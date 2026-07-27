from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import types

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    TESTED_PYDDX_VERSION,
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
    PyDDXReactionFieldLinearMap,
    _PyDDXRuntime,
    mace_polar_density_to_pyddx_multipoles,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    NeutralDensityCoordinates,
    UnmixedDensityResidualLinearization,
    project_neutral_density_tangent,
    solve_adjoint,
)


class _FakeModel:
    """Quadratic fake with the same source/state surface as pyddx 0.8.0."""

    def __init__(
        self,
        model,
        sphere_centres,
        sphere_radii,
        solvent_epsilon,
        **_kwargs,
    ):
        self.model_name = str(model)
        self.options = dict(_kwargs)
        self.n_proc = int(_kwargs["n_proc"])
        self.sphere_centres = np.asarray(sphere_centres, dtype=float)
        self.sphere_radii = np.asarray(sphere_radii, dtype=float)
        self.solvent_epsilon = float(solvent_epsilon)
        self.n_spheres = self.sphere_centres.shape[1]
        self.n_basis = 4
        self.n_cav = 4 * self.n_spheres
        self.has_force_enabled = True
        self.scale = 1.0 + 0.01 * float(np.sum(self.sphere_centres))

    @staticmethod
    def _raw_density(multipoles):
        values = np.asarray(multipoles, dtype=float)
        raw = np.empty((values.shape[1], 4), dtype=float)
        raw[:, 0] = values[0] * np.sqrt(4.0 * np.pi)
        raw[:, 1:] = values[1:].T * Bohr * np.sqrt(4.0 * np.pi / 3.0)
        return raw

    def multipole_psi(self, multipoles):
        return self._raw_density(multipoles).T.copy()

    def multipole_electrostatics(self, multipoles, derivative_order=-1):
        raw = self._raw_density(multipoles)
        result = {"phi": raw.reshape(-1).copy()}
        if derivative_order != 0:
            result["e"] = np.zeros((3, self.n_cav), dtype=float)
        return result


class _FakeState:
    instances = []

    def __init__(self, model, psi, phi):
        self.model = model
        self.fill_guess_calls = 0
        self.fill_guess_adjoint_calls = 0
        self.solve_calls = 0
        self.solve_adjoint_calls = 0
        self.update_problem_calls = 0
        self._set_problem(psi, phi)
        self.instances.append(self)

    def _set_problem(self, psi, phi):
        self.raw = np.asarray(psi, dtype=float).T.copy()
        np.testing.assert_allclose(
            np.asarray(phi, dtype=float),
            self.raw.reshape(-1),
        )
        self.x = self.model.scale * self.raw.T
        self.xi = -self.model.scale * self.raw.reshape(-1)

    def update_problem(self, psi, phi, _elec_field=None):
        self.update_problem_calls += 1
        self._set_problem(psi, phi)

    def fill_guess(self, _tolerance):
        self.fill_guess_calls += 1

    def fill_guess_adjoint(self, _tolerance):
        self.fill_guess_adjoint_calls += 1

    def solve(self, _tolerance):
        self.solve_calls += 1

    def solve_adjoint(self, _tolerance):
        self.solve_adjoint_calls += 1

    def energy(self):
        return 0.5 * self.model.scale * float(np.vdot(self.raw, self.raw))

    def solvation_force_terms(self, _electrostatics):
        value = 0.005 * float(np.vdot(self.raw, self.raw))
        return np.full((3, self.model.n_spheres), value)

    def multipole_force_terms(self, _multipoles):
        return np.zeros((3, self.model.n_spheres), dtype=float)


class _MatrixDensityResponse:
    def __init__(self, matrix, atom_count):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = int(atom_count)

    def jvp(self, field_direction):
        return (
            self.matrix @ np.asarray(field_direction, dtype=float).reshape(-1)
        ).reshape(self.atom_count, 4)

    def vjp(self, density_cotangent):
        return (
            self.matrix.T
            @ np.asarray(density_cotangent, dtype=float).reshape(-1)
        ).reshape(self.atom_count, 4)


@dataclass(frozen=True)
class _FakeRuntimeBundle:
    runtime: _PyDDXRuntime


@pytest.fixture
def fake_runtime():
    _FakeState.instances.clear()
    return _FakeRuntimeBundle(
        runtime=_PyDDXRuntime(
            version=TESTED_PYDDX_VERSION,
            module=types.SimpleNamespace(Model=_FakeModel, State=_FakeState),
        )
    )


def _reaction_map(fake_runtime):
    return PyDDXPCMReactionFieldLinearMap(
        np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]]),
        np.asarray([1.2, 1.5]),
        dielectric=78.39,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-12,
        _runtime=fake_runtime.runtime,
    )


def test_mace_density_to_pyddx_multipoles_preserves_l1_order_and_units():
    density = np.asarray(
        [
            [0.3, 0.4, -0.2, 0.1],
            [-0.3, -0.1, 0.5, -0.6],
        ]
    )

    multipoles = mace_polar_density_to_pyddx_multipoles(density)

    np.testing.assert_allclose(
        multipoles[0],
        density[:, 0] / np.sqrt(4.0 * np.pi),
    )
    np.testing.assert_allclose(
        multipoles[1:].T,
        density[:, 1:] / (Bohr * np.sqrt(4.0 * np.pi / 3.0)),
    )


def test_pyddx_reaction_map_apply_adjoint_and_energy_identity(fake_runtime):
    reaction = _reaction_map(fake_runtime)
    assert reaction._model.model_name == "pcm"
    assert reaction.runtime_provenance["model"] == "pcm"
    assert reaction.runtime_provenance["method"] == "ddPCM"
    assert reaction.runtime_provenance["dielectric_scaling"] == 1.0
    assert reaction._model.options["n_proc"] == 1
    assert reaction.runtime_provenance["n_proc"] == 1
    density = np.asarray([[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]])
    cotangent = np.asarray([[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]])
    scale = reaction._model.scale

    field = reaction.apply(density)
    expected_field = density_to_external_field_order(scale * density) * Hartree
    np.testing.assert_allclose(field, expected_field)

    energy_ev = reaction.polarization_energy_hartree(density) * Hartree
    assert energy_ev == pytest.approx(
        0.5
        * float(
            np.vdot(
                density,
                external_field_to_density_order(field),
            )
        )
    )

    adjoint = reaction.adjoint(cotangent)
    expected_adjoint = scale * external_field_to_density_order(cotangent) * Hartree
    np.testing.assert_allclose(adjoint, expected_adjoint)
    assert float(np.vdot(cotangent, field)) == pytest.approx(
        float(np.vdot(adjoint, density))
    )


def test_pyddx_cosmo_scales_energy_field_adjoint_and_position_vjp(
    fake_runtime,
):
    positions = np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]])
    radii = np.asarray([1.2, 1.5])
    dielectric = 78.39
    common = {
        "dielectric": dielectric,
        "lmax": 7,
        "n_lebedev": 302,
        "solver_tolerance": 1.0e-12,
        "_runtime": fake_runtime.runtime,
    }
    pcm = PyDDXPCMReactionFieldLinearMap(positions, radii, **common)
    cosmo = PyDDXCOSMOReactionFieldLinearMap(positions, radii, **common)
    density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )
    cotangent = np.asarray(
        [[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]]
    )
    expected_scale = (dielectric - 1.0) / dielectric

    assert cosmo._model.model_name == "cosmo"
    assert cosmo.runtime_provenance["model"] == "cosmo"
    assert cosmo.runtime_provenance["method"] == "ddCOSMO"
    assert cosmo.runtime_provenance["dielectric_scaling"] == pytest.approx(
        expected_scale
    )
    assert cosmo.runtime_provenance["dielectric_scaling_source"] == (
        "pyddx-0.8.0-host-applied-(epsilon-1)/epsilon"
    )
    assert cosmo.polarization_energy_hartree(density) == pytest.approx(
        expected_scale * pcm.polarization_energy_hartree(density)
    )
    np.testing.assert_allclose(
        cosmo.apply(density),
        expected_scale * pcm.apply(density),
    )
    np.testing.assert_allclose(
        cosmo.adjoint(cotangent),
        expected_scale * pcm.adjoint(cotangent),
    )
    np.testing.assert_allclose(
        cosmo.full_position_vjp(density, cotangent),
        expected_scale * pcm.full_position_vjp(density, cotangent),
    )


def test_pyddx_generic_map_rejects_unsupported_continuum_model(fake_runtime):
    with pytest.raises(ValueError, match="continuum_model.*pcm.*cosmo"):
        PyDDXReactionFieldLinearMap(
            np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            continuum_model="cpcm",
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            _runtime=fake_runtime.runtime,
        )


def test_pyddx_reaction_map_passes_and_reports_configured_thread_count(
    fake_runtime,
):
    reaction = PyDDXPCMReactionFieldLinearMap(
        np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]]),
        np.asarray([1.2, 1.5]),
        dielectric=78.39,
        lmax=7,
        n_lebedev=302,
        n_proc=4,
        _runtime=fake_runtime.runtime,
    )

    assert reaction._model.options["n_proc"] == 4
    assert reaction.runtime_provenance["n_proc"] == 4


@pytest.mark.parametrize("n_proc", [True, 0, -1, 1.5])
def test_pyddx_reaction_map_rejects_invalid_thread_count(
    fake_runtime,
    n_proc,
):
    with pytest.raises(ValueError, match="n_proc.*positive integer"):
        PyDDXPCMReactionFieldLinearMap(
            np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            n_proc=n_proc,
            _runtime=fake_runtime.runtime,
        )


def test_pyddx_reaction_map_rejects_backend_thread_count_mismatch(
    fake_runtime,
):
    class _MismatchedThreadModel(_FakeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_proc = 1

    runtime = _PyDDXRuntime(
        version=TESTED_PYDDX_VERSION,
        module=types.SimpleNamespace(
            Model=_MismatchedThreadModel,
            State=_FakeState,
        ),
    )
    with pytest.raises(RuntimeError, match="thread count"):
        PyDDXPCMReactionFieldLinearMap(
            np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]]),
            np.asarray([1.2, 1.5]),
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            n_proc=4,
            _runtime=runtime,
        )


def test_pyddx_scf_path_reuses_previous_state_solutions_as_guesses(
    fake_runtime,
):
    reaction = _reaction_map(fake_runtime)
    first_density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )
    second_density = 0.75 * first_density

    first_field = reaction.apply_scf(first_density)
    second_field = reaction.apply_scf(second_density)

    assert len(_FakeState.instances) == 1
    state = _FakeState.instances[0]
    assert state.fill_guess_calls == 1
    assert state.fill_guess_adjoint_calls == 1
    assert state.update_problem_calls == 1
    assert state.solve_calls == 2
    assert state.solve_adjoint_calls == 2
    np.testing.assert_allclose(second_field, 0.75 * first_field)
    assert np.isfinite(
        reaction.scf_polarization_energy_hartree(second_density)
    )
    assert state.update_problem_calls == 2
    assert state.solve_calls == 3
    assert state.solve_adjoint_calls == 2
    provenance = reaction.runtime_provenance
    assert provenance["scf_state_creations"] == 1
    assert provenance["scf_state_updates"] == 2
    assert provenance["scf_forward_warm_starts"] == 2
    assert provenance["scf_adjoint_warm_starts"] == 1

    reaction.apply(second_density)
    reaction.apply(second_density)
    assert len(_FakeState.instances) == 3
    assert provenance == reaction.runtime_provenance


def test_pyddx_scf_path_discards_failed_warm_state(fake_runtime):
    reaction = _reaction_map(fake_runtime)
    density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )
    reaction.apply_scf(density)
    failed_state = _FakeState.instances[0]

    def fail_update(_psi, _phi):
        raise RuntimeError("synthetic update failure")

    failed_state.update_problem = fail_update
    with pytest.raises(RuntimeError, match="synthetic update failure"):
        reaction.apply_scf(0.75 * density)

    reaction.apply_scf(0.5 * density)
    assert len(_FakeState.instances) == 2
    assert _FakeState.instances[1] is not failed_state
    assert reaction.runtime_provenance["scf_state_creations"] == 2


def test_pyddx_full_position_vjp_uses_complete_energy_polarization_identity(
    fake_runtime,
):
    reaction = _reaction_map(fake_runtime)
    density = np.asarray([[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]])
    field_cotangent = np.asarray([[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]])
    raw_left = external_field_to_density_order(field_cotangent)

    gradient = reaction.full_position_vjp(density, field_cotangent)
    expected_value = 0.01 * float(np.vdot(raw_left, density)) * Hartree / Bohr
    np.testing.assert_allclose(
        gradient,
        np.full((2, 3), expected_value),
    )


def test_pyddx_exposes_fixed_source_polarization_position_gradient(
    fake_runtime,
):
    reaction = _reaction_map(fake_runtime)
    density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )

    gradient = reaction.polarization_position_gradient_ev_per_angstrom(
        density
    )

    expected_value = (
        0.005 * float(np.vdot(density, density)) * Hartree / Bohr
    )
    np.testing.assert_allclose(
        gradient,
        np.full((2, 3), expected_value),
    )


def test_pyddx_map_composes_with_fixed_point_adjoint_and_total_cds_gradient(
    fake_runtime,
):
    atom_count = 2
    flat_dimension = 4 * atom_count
    coordinates = NeutralDensityCoordinates(atom_count)
    neutral_projector = np.column_stack(
        [
            project_neutral_density_tangent(unit.reshape(atom_count, 4)).reshape(
                -1
            )
            for unit in np.eye(flat_dimension)
        ]
    )
    shifted_identity = np.roll(np.eye(flat_dimension), 1, axis=1)
    density_response = _MatrixDensityResponse(
        neutral_projector
        @ (2.0e-4 * np.eye(flat_dimension) + 5.0e-5 * shifted_identity),
        atom_count,
    )
    base_density = project_neutral_density_tangent(
        np.asarray(
            [
                [0.12, 0.03, -0.02, 0.04],
                [-0.12, -0.01, 0.05, -0.03],
            ]
        )
    )
    intrinsic_linear = np.linspace(
        -0.03,
        0.04,
        flat_dimension,
    )
    intrinsic_hessian = (
        2.0e-3 * np.eye(flat_dimension)
        + 5.0e-4 * (shifted_identity + shifted_identity.T)
    )
    positions = np.asarray([[-0.7, 0.1, 0.2], [0.8, -0.2, -0.1]])
    radii = np.asarray([1.2, 1.5])
    cds_gradient_hartree_per_angstrom = np.asarray(
        [[0.001, -0.002, 0.003], [-0.001, 0.002, -0.003]]
    )

    def state_at(displaced_positions):
        reaction = PyDDXPCMReactionFieldLinearMap(
            displaced_positions,
            radii,
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            solver_tolerance=1.0e-12,
            _runtime=fake_runtime.runtime,
        )
        reduced_response = np.column_stack(
            [
                coordinates.reduce(
                    density_response.jvp(
                        reaction.apply(
                            coordinates.expand(unit),
                        )
                    )
                )
                for unit in np.eye(coordinates.dimension)
            ]
        )
        reduced_density = np.linalg.solve(
            np.eye(coordinates.dimension) - reduced_response,
            coordinates.reduce(base_density),
        )
        density = coordinates.expand(reduced_density)
        field = reaction.apply(density)
        flat_field = field.reshape(-1)
        intrinsic_energy = float(
            intrinsic_linear @ flat_field
            + 0.5 * flat_field @ intrinsic_hessian @ flat_field
        )
        polarization_energy = (
            reaction.polarization_energy_hartree(density) * Hartree
        )
        continuum_energy = intrinsic_energy + polarization_energy
        cds_energy = float(
            np.vdot(
                cds_gradient_hartree_per_angstrom,
                displaced_positions - positions,
            )
        )
        return (
            reaction,
            density,
            field,
            continuum_energy,
            continuum_energy + cds_energy * Hartree,
        )

    reaction, density, field, _, _ = state_at(positions)
    np.testing.assert_allclose(
        density,
        base_density + density_response.jvp(field),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    flat_field = field.reshape(-1)
    intrinsic_field_gradient = (
        intrinsic_linear + intrinsic_hessian @ flat_field
    ).reshape(atom_count, 4)
    residual = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=reaction,
        density_response=density_response,
    )
    physical_rhs = fixed_cavity_energy_density_gradient(
        reaction,
        reaction_field_values=field,
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
    )
    adjoint = solve_adjoint(
        residual,
        physical_rhs,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-13,
    )
    zero_coordinates = np.zeros((atom_count, 3))
    continuum_analytic = continuum_coupled_solvation_coordinate_gradient(
        reaction,
        density_response,
        density_coefficients=density,
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
        adjoint_solution=adjoint.solution,
        adjoint_density_position_vjp=zero_coordinates,
        solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
        gas_forces_ev_per_angstrom=zero_coordinates,
    )
    total_analytic = assemble_total_solvation_coordinate_gradient(
        continuum_analytic,
        cds_gradient_hartree_per_angstrom,
    )

    step = 1.0e-5
    displaced_continuum_energies = []
    displaced_total_energies = []
    for sign in (-1.0, 1.0):
        displaced = positions.copy()
        displaced[0, 0] += sign * step
        displaced_state = state_at(displaced)
        displaced_continuum_energies.append(displaced_state[-2])
        displaced_total_energies.append(displaced_state[-1])
    continuum_finite_difference = (
        displaced_continuum_energies[1] - displaced_continuum_energies[0]
    ) / (2.0 * step)
    total_finite_difference = (
        displaced_total_energies[1] - displaced_total_energies[0]
    ) / (2.0 * step)

    assert adjoint.relative_residual <= 1.0e-12
    assert continuum_analytic[0, 0] == pytest.approx(
        continuum_finite_difference,
        abs=2.0e-8,
        rel=2.0e-7,
    )
    assert (
        total_analytic.total_position_gradient_hartree_per_angstrom[0, 0]
        * Hartree
    ) == pytest.approx(
        total_finite_difference,
        abs=2.0e-8,
        rel=2.0e-7,
    )


def test_pyddx_response_fails_closed_for_untested_runtime(fake_runtime):
    runtime = _PyDDXRuntime(
        version="0.9.0",
        module=fake_runtime.runtime.module,
    )
    with pytest.raises(RuntimeError, match="tested only"):
        PyDDXPCMReactionFieldLinearMap(
            np.zeros((2, 3)),
            np.ones(2),
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            _runtime=runtime,
        )


def test_pyddx_response_rejects_invalid_geometry_and_density(fake_runtime):
    with pytest.raises(ValueError, match="positions"):
        PyDDXPCMReactionFieldLinearMap(
            np.zeros((2, 2)),
            np.ones(2),
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            _runtime=fake_runtime.runtime,
        )

    reaction = _reaction_map(fake_runtime)
    with pytest.raises(ValueError, match="density_direction"):
        reaction.apply(np.zeros((2, 3)))


def test_real_pyddx_full_position_vjp_matches_same_energy_finite_difference():
    pyddx = pytest.importorskip("pyddx")
    if str(pyddx.__version__) != TESTED_PYDDX_VERSION:
        pytest.skip("The optional real-runtime canary is version locked.")

    positions = np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]])
    radii = np.asarray([1.2, 1.5])
    density = np.asarray([[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]])
    reaction = PyDDXPCMReactionFieldLinearMap(
        positions,
        radii,
        dielectric=78.39,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-11,
    )
    multipoles = mace_polar_density_to_pyddx_multipoles(density)
    ddx_potential = reaction._model.multipole_electrostatics(
        multipoles,
        derivative_order=0,
    )["phi"]
    maple_potential = point_multipole_potential(
        np.asarray(reaction._model.cavity, dtype=float).T,
        positions,
        density,
    )
    np.testing.assert_allclose(
        ddx_potential,
        maple_potential,
        atol=1.0e-13,
        rtol=1.0e-13,
    )

    field = reaction.apply(density)
    energy_ev = reaction.polarization_energy_hartree(density) * Hartree
    assert energy_ev == pytest.approx(
        0.5
        * float(
            np.vdot(
                density,
                external_field_to_density_order(field),
            )
        ),
        abs=1.0e-10,
    )
    density_step = 1.0e-5
    displaced_density_energies_hartree = []
    for sign in (-1.0, 1.0):
        displaced_density = density.copy()
        displaced_density[0, 1] += sign * density_step
        displaced_density_energies_hartree.append(
            reaction.polarization_energy_hartree(displaced_density)
        )
    density_finite_difference = (
        displaced_density_energies_hartree[1]
        - displaced_density_energies_hartree[0]
    ) / (2.0 * density_step)
    raw_reaction_gradient_hartree = (
        external_field_to_density_order(field) / Hartree
    )
    assert raw_reaction_gradient_hartree[0, 1] == pytest.approx(
        density_finite_difference,
        abs=1.0e-9,
        rel=1.0e-7,
    )

    field_cotangent_probe = np.asarray(
        [[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]]
    )
    adjoint = reaction.adjoint(field_cotangent_probe)
    assert float(np.vdot(field_cotangent_probe, field)) == pytest.approx(
        float(np.vdot(adjoint, density)),
        abs=1.0e-10,
    )

    field_cotangent = 0.5 * density_to_external_field_order(density)
    analytic = reaction.full_position_vjp(
        density,
        field_cotangent,
    )

    step = 1.0e-4
    energies = []
    for sign in (-1.0, 1.0):
        displaced = positions.copy()
        displaced[0, 0] += sign * step
        displaced_reaction = PyDDXPCMReactionFieldLinearMap(
            displaced,
            radii,
            dielectric=78.39,
            lmax=7,
            n_lebedev=302,
            solver_tolerance=1.0e-11,
        )
        energies.append(
            displaced_reaction.polarization_energy_hartree(density) * Hartree
        )
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)

    assert analytic[0, 0] == pytest.approx(
        finite_difference,
        abs=2.0e-6,
        rel=2.0e-5,
    )


def test_real_pyddx_pcm_converges_to_scaled_cosmo_at_high_dielectric():
    pyddx = pytest.importorskip("pyddx")
    if str(pyddx.__version__) != TESTED_PYDDX_VERSION:
        pytest.skip("The optional real-runtime canary is version locked.")

    positions = np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]])
    radii = np.asarray([1.2, 1.5])
    density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )
    cotangent = np.asarray(
        [[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]]
    )
    common = {
        "dielectric": 1.0e8,
        "lmax": 7,
        "n_lebedev": 302,
        "solver_tolerance": 1.0e-11,
    }
    pcm = PyDDXPCMReactionFieldLinearMap(positions, radii, **common)
    cosmo = PyDDXCOSMOReactionFieldLinearMap(positions, radii, **common)

    assert cosmo.polarization_energy_hartree(density) == pytest.approx(
        pcm.polarization_energy_hartree(density),
        abs=5.0e-10,
        rel=1.0e-8,
    )
    np.testing.assert_allclose(
        cosmo.apply(density),
        pcm.apply(density),
        atol=5.0e-8,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        cosmo.full_position_vjp(density, cotangent),
        pcm.full_position_vjp(density, cotangent),
        atol=5.0e-8,
        rtol=1.0e-7,
    )


def test_real_pyddx_scaled_cosmo_position_vjp_matches_energy_difference():
    pyddx = pytest.importorskip("pyddx")
    if str(pyddx.__version__) != TESTED_PYDDX_VERSION:
        pytest.skip("The optional real-runtime canary is version locked.")

    positions = np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]])
    radii = np.asarray([1.2, 1.5])
    density = np.asarray(
        [[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]]
    )
    common = {
        "dielectric": 78.39,
        "lmax": 7,
        "n_lebedev": 302,
        "solver_tolerance": 1.0e-11,
    }
    reaction = PyDDXCOSMOReactionFieldLinearMap(
        positions,
        radii,
        **common,
    )
    analytic = reaction.full_position_vjp(
        density,
        0.5 * density_to_external_field_order(density),
    )

    step = 1.0e-4
    energies = []
    for sign in (-1.0, 1.0):
        displaced = positions.copy()
        displaced[0, 0] += sign * step
        displaced_reaction = PyDDXCOSMOReactionFieldLinearMap(
            displaced,
            radii,
            **common,
        )
        energies.append(
            displaced_reaction.polarization_energy_hartree(density)
            * Hartree
        )
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)

    assert analytic[0, 0] == pytest.approx(
        finite_difference,
        abs=2.0e-6,
        rel=2.0e-5,
    )


def test_real_pyddx_one_and_four_threads_are_numerically_equivalent(tmp_path):
    pyddx = pytest.importorskip("pyddx")
    if str(pyddx.__version__) != TESTED_PYDDX_VERSION:
        pytest.skip("The optional real-runtime canary is version locked.")

    script = """
import sys
import numpy as np
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXPCMReactionFieldLinearMap,
)
positions = np.asarray([[-0.7, 0.0, 0.1], [0.8, 0.2, -0.1]])
radii = np.asarray([1.2, 1.5])
density = np.asarray([[0.2, 0.1, -0.3, 0.4], [-0.2, 0.5, 0.2, -0.1]])
cotangent = np.asarray([[0.3, -0.2, 0.5, 0.1], [-0.4, 0.6, -0.1, 0.2]])
reaction = PyDDXPCMReactionFieldLinearMap(
    positions,
    radii,
    dielectric=78.39,
    lmax=7,
    n_lebedev=302,
    n_proc=int(sys.argv[1]),
    solver_tolerance=1.0e-11,
)
np.savez(
    sys.argv[2],
    field=reaction.apply(density),
    energy=reaction.polarization_energy_hartree(density),
    position_vjp=reaction.full_position_vjp(density, cotangent),
)
"""
    outputs = []
    for n_proc in (1, 4):
        output = tmp_path / f"pyddx-nproc-{n_proc}.npz"
        subprocess.run(
            [sys.executable, "-c", script, str(n_proc), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(np.load(output))

    np.testing.assert_allclose(
        outputs[1]["field"],
        outputs[0]["field"],
        atol=1.0e-12,
        rtol=0.0,
    )
    assert float(outputs[1]["energy"]) == pytest.approx(
        float(outputs[0]["energy"]),
        abs=1.0e-13,
        rel=0.0,
    )
    np.testing.assert_allclose(
        outputs[1]["position_vjp"],
        outputs[0]["position_vjp"],
        atol=1.0e-11,
        rtol=0.0,
    )
