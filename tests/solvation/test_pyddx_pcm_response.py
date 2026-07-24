from __future__ import annotations

from dataclasses import dataclass
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
    PyDDXPCMReactionFieldLinearMap,
    _PyDDXRuntime,
    mace_polar_density_to_pyddx_multipoles,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
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
        _model,
        sphere_centres,
        sphere_radii,
        solvent_epsilon,
        **_kwargs,
    ):
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
    def __init__(self, model, psi, phi):
        self.model = model
        self.raw = np.asarray(psi, dtype=float).T.copy()
        np.testing.assert_allclose(
            np.asarray(phi, dtype=float),
            self.raw.reshape(-1),
        )
        self.x = model.scale * self.raw.T
        self.xi = -model.scale * self.raw.reshape(-1)

    @staticmethod
    def fill_guess(_tolerance):
        return None

    @staticmethod
    def fill_guess_adjoint(_tolerance):
        return None

    @staticmethod
    def solve(_tolerance):
        return None

    @staticmethod
    def solve_adjoint(_tolerance):
        return None

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


def test_pyddx_map_composes_with_fixed_point_adjoint_and_full_gradient(
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
        return reaction, density, field, intrinsic_energy + polarization_energy

    reaction, density, field, _ = state_at(positions)
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
    analytic = continuum_coupled_solvation_coordinate_gradient(
        reaction,
        density_response,
        density_coefficients=density,
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
        adjoint_solution=adjoint.solution,
        adjoint_density_position_vjp=zero_coordinates,
        solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
        gas_forces_ev_per_angstrom=zero_coordinates,
    )

    step = 1.0e-5
    displaced_energies = []
    for sign in (-1.0, 1.0):
        displaced = positions.copy()
        displaced[0, 0] += sign * step
        displaced_energies.append(state_at(displaced)[-1])
    finite_difference = (
        displaced_energies[1] - displaced_energies[0]
    ) / (2.0 * step)

    assert adjoint.relative_residual <= 1.0e-12
    assert analytic[0, 0] == pytest.approx(
        finite_difference,
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
        displaced_density_energies_hartree[1] - displaced_density_energies_hartree[0]
    ) / (2.0 * density_step)
    raw_reaction_gradient_hartree = external_field_to_density_order(field) / Hartree
    assert raw_reaction_gradient_hartree[0, 1] == pytest.approx(
        density_finite_difference,
        abs=1.0e-9,
        rel=1.0e-7,
    )

    field_cotangent_probe = np.asarray([[0.7, -0.4, 0.2, 0.1], [-0.3, 0.6, -0.5, 0.8]])
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
