from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    PCMSolverExternalMEPCavityResponse,
    SurfaceChargeState,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    ExactGTOFieldProjector,
    MACEPolarGTOFieldProjectionSpec,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    continuum_coupled_solvation_coordinate_gradient,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION,
    AtomCenteredSurfacePCMReactionFieldLinearMap,
    FixedCavityPCMReactionFieldLinearMap,
)


class _FakeSymmetricPCMSolverSession:
    def __init__(
        self,
        atom_positions_angstrom: np.ndarray,
        cavity_centers_bohr: np.ndarray,
        response_matrix: np.ndarray,
        *,
        symmetric: bool = True,
    ):
        self.atomic_numbers = np.ones(len(atom_positions_angstrom))
        self.coordinates_bohr = np.asarray(atom_positions_angstrom) / Bohr
        self.cavity_centers_bohr = np.asarray(cavity_centers_bohr, dtype=float)
        self.cavity_areas_bohr2 = np.ones(len(cavity_centers_bohr))
        self.response_operator_is_symmetric = symmetric
        self._response_matrix = np.asarray(response_matrix, dtype=float)
        self.solve_calls = 0

    def compute_asc(self, mep: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(mep, dtype=float)

    def solve(self, mep: np.ndarray) -> dict[str, np.ndarray | float]:
        self.solve_calls += 1
        potential = np.asarray(mep, dtype=float)
        asc = self.compute_asc(potential)
        return {
            "asc": asc,
            "polarization_energy": 0.5 * float(np.dot(potential, asc)),
        }


class _ZeroDensityResponse:
    @staticmethod
    def vjp(density_cotangent: np.ndarray) -> np.ndarray:
        return np.zeros_like(density_cotangent)


class _AtomCenteredMovingResponse:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    operator_derivative_contract_version = 1
    surface_motion_contract_version = (
        ATOM_CENTERED_SURFACE_MOTION_CONTRACT_VERSION
    )
    energy_response_is_reciprocal = True

    def __init__(self, atom_positions_angstrom: np.ndarray):
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        self._positions = positions.copy()
        self._atomic_numbers = np.ones(len(positions))
        self._parents = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)
        self._offsets_bohr = np.asarray(
            [
                [-2.1, 0.4, 0.2],
                [1.8, -0.5, 0.7],
                [-1.7, 1.3, -0.4],
                [2.2, 0.6, -0.8],
                [0.5, -2.0, 1.1],
                [-0.9, 1.8, -1.4],
            ]
        )
        rng = np.random.default_rng(20260725)
        base = rng.normal(scale=0.05, size=(6, 6))
        self._base_response = base + base.T
        derivative = rng.normal(scale=0.002, size=(3, 3, 6, 6))
        self._response_derivative = derivative + derivative.swapaxes(-1, -2)
        self._response_matrix = self._base_response + np.einsum(
            "ak,akij->ij",
            positions,
            self._response_derivative,
        )

    @property
    def atomic_numbers(self):
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self):
        return self._positions.copy() / Bohr

    @property
    def cavity_radii_angstrom(self):
        return np.full(len(self._positions), 1.5)

    @property
    def surface_points_bohr(self):
        return (
            self._positions[self._parents] / Bohr
            + self._offsets_bohr
        )

    @property
    def surface_areas_bohr2(self):
        return np.ones(len(self._parents))

    @property
    def surface_parent_atom_indices(self):
        return self._parents.copy()

    def apply_energy_conjugate(self, surface_potential_hartree_per_e):
        return self._response_matrix @ np.asarray(
            surface_potential_hartree_per_e,
            dtype=float,
        )

    def solve(self, surface_potential_hartree_per_e):
        potential = np.asarray(
            surface_potential_hartree_per_e,
            dtype=float,
        )
        charge = self.apply_energy_conjugate(potential)
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=charge,
            adjoint_surface_charge_e=charge,
            energy_conjugate_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(np.dot(potential, charge)),
        )

    def operator_position_vjp(self, left, right):
        return np.einsum(
            "i,akij,j->ak",
            np.asarray(left, dtype=float),
            self._response_derivative,
            np.asarray(right, dtype=float),
        )


def _operator(*, symmetric: bool = True):
    positions = np.asarray(
        [
            [-0.7, 0.1, 0.2],
            [0.8, -0.2, 0.3],
            [0.2, 0.9, -0.4],
        ]
    )
    centers = np.asarray(
        [
            [-3.0, 0.0, 0.5],
            [2.5, 1.0, -0.2],
            [0.3, -2.8, 1.2],
            [1.4, 2.2, 2.0],
            [-1.7, 1.8, -2.1],
        ]
    )
    rng = np.random.default_rng(20260724)
    factor = rng.normal(size=(centers.shape[0], centers.shape[0]))
    response = factor + factor.T
    session = _FakeSymmetricPCMSolverSession(
        positions,
        centers,
        response,
        symmetric=symmetric,
    )
    continuum_response = PCMSolverExternalMEPCavityResponse(
        session,
        cavity_radii_angstrom=np.full(len(positions), 1.5),
    )
    return (
        FixedCavityPCMReactionFieldLinearMap(
            continuum_response,
            positions,
        ),
        positions,
        session,
        continuum_response,
    )


def _moving_operator(atom_positions_angstrom: np.ndarray):
    response = _AtomCenteredMovingResponse(atom_positions_angstrom)
    return AtomCenteredSurfacePCMReactionFieldLinearMap(
        response,
        atom_positions_angstrom,
    )


def test_fixed_cavity_pcm_apply_and_adjoint_obey_discrete_pairing():
    operator, positions, _, _ = _operator()
    rng = np.random.default_rng(19)
    density_direction = rng.normal(size=(len(positions), 4))
    field_cotangent = rng.normal(size=(len(positions), 4))

    applied = operator.apply(density_direction)
    adjoint = operator.adjoint(field_cotangent)

    assert np.vdot(field_cotangent, applied) == pytest.approx(
        np.vdot(adjoint, density_direction),
        rel=2.0e-13,
        abs=2.0e-11,
    )


def test_fixed_cavity_pcm_adjoint_matches_the_full_discrete_transpose():
    operator, positions, _, _ = _operator()
    dimension = 4 * len(positions)
    basis = np.eye(dimension)
    forward_matrix = np.column_stack(
        [
            operator.apply(vector.reshape(len(positions), 4)).reshape(-1)
            for vector in basis
        ]
    )
    adjoint_matrix = np.column_stack(
        [
            operator.adjoint(vector.reshape(len(positions), 4)).reshape(-1)
            for vector in basis
        ]
    )

    np.testing.assert_allclose(
        adjoint_matrix,
        forward_matrix.T,
        rtol=2.0e-13,
        atol=2.0e-11,
    )


def test_fixed_cavity_pcm_map_is_linear():
    operator, positions, _, _ = _operator()
    rng = np.random.default_rng(23)
    first = rng.normal(size=(len(positions), 4))
    second = rng.normal(size=(len(positions), 4))

    np.testing.assert_allclose(
        operator.apply(0.7 * first - 1.3 * second),
        0.7 * operator.apply(first) - 1.3 * operator.apply(second),
        rtol=2.0e-13,
        atol=2.0e-11,
    )


def test_fixed_cavity_pcm_scf_field_and_energy_share_one_root_snapshot():
    operator, positions, session, _ = _operator()
    density = np.zeros((len(positions), 4))
    density[:, 0] = [-0.2, 0.05, 0.15]

    field = operator.apply_scf(density)
    snapshot = operator.scf_snapshot(density)
    energy = operator.scf_polarization_energy_hartree(density)

    assert session.solve_calls == 1
    assert energy == snapshot.polarization_energy_hartree
    assert snapshot.density_coefficients.flags.writeable is False
    np.testing.assert_allclose(
        field[:, 0],
        snapshot.reaction_potential_hartree_per_e * Hartree,
    )
    np.testing.assert_allclose(
        field[:, 1:],
        snapshot.reaction_gradient_hartree_per_e_bohr * Hartree / Bohr,
    )


def test_exact_gto_drive_keeps_energy_and_model_spaces_distinct():
    _, positions, session, continuum_response = _operator()
    matrix = np.asarray(
        [
            [3.544907701811032, 0.0, 0.0, 0.0],
            [3.544907701811032, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.771474235728387],
            [0.0, 5.771474235728387, 0.0, 0.0],
            [0.0, 0.0, 5.771474235728387, 0.0],
            [0.0, 0.0, 0.0, 11.542948471456774],
            [0.0, 11.542948471456774, 0.0, 0.0],
            [0.0, 0.0, 11.542948471456774, 0.0],
        ]
    )
    projector = ExactGTOFieldProjector(
        MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=(1.5, 3.0),
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=matrix,
        )
    )
    operator = FixedCavityPCMReactionFieldLinearMap(
        continuum_response,
        positions,
        model_field_projector=projector,
        model_field_gauge="atomic-center-mean-zero-v1",
    )
    density = np.asarray(
        [
            [-0.20, 0.10, -0.30, 0.40],
            [0.05, -0.20, 0.15, -0.10],
            [0.15, 0.10, 0.15, -0.30],
        ]
    )

    drive = operator.apply_scf_drive(density)
    snapshot = operator.scf_snapshot(density)
    expected_mep = point_multipole_potential(
        session.cavity_centers_bohr,
        positions,
        density,
    )
    expected_potential, expected_gradient = (
        point_asc_reaction_potential_gradient(
        positions,
        session.cavity_centers_bohr,
        snapshot.asc_e,
        )
    )

    np.testing.assert_allclose(snapshot.mep_hartree_per_e, expected_mep)
    np.testing.assert_allclose(
        drive.density_dual_field_ev[:, 0],
        expected_potential * Hartree,
    )
    np.testing.assert_allclose(
        drive.density_dual_field_ev[:, 1:],
        expected_gradient * Hartree / Bohr,
    )
    np.testing.assert_allclose(
        drive.model_field_features,
        projector.project_asc(
            positions,
            session.cavity_centers_bohr,
            snapshot.asc_e,
        ),
    )
    assert drive.projector == "exact-gto-v1"
    assert drive.model_field_gauge == "atomic-center-mean-zero-v1"
    assert drive.model_field_gauge_reference_ev == pytest.approx(
        float(np.mean(expected_potential)) * Hartree
    )
    assert np.isfinite(drive.model_field_gauge_reference_ev)
    assert snapshot.density_reaction_coupling_hartree == pytest.approx(
        2.0 * snapshot.polarization_energy_hartree,
        rel=2.0e-13,
        abs=2.0e-11,
    )


def test_centered_local_jet_uses_same_atomic_mean_as_exact_gto():
    _, positions, session, continuum_response = _operator()
    density = np.asarray(
        [
            [-0.20, 0.10, -0.30, 0.40],
            [0.05, -0.20, 0.15, -0.10],
            [0.15, 0.10, 0.15, -0.30],
        ]
    )
    operator = FixedCavityPCMReactionFieldLinearMap(
        continuum_response,
        positions,
        model_field_gauge="atomic-center-mean-zero-v1",
    )

    drive = operator.apply_scf_drive(density)
    expected_reference = float(
        np.mean(drive.density_dual_field_ev[:, 0])
    )

    assert drive.model_field_features is None
    assert drive.model_local_field_ev is not None
    assert drive.model_field_gauge == "atomic-center-mean-zero-v1"
    assert drive.model_field_gauge_reference_ev == pytest.approx(
        expected_reference
    )
    np.testing.assert_allclose(
        drive.model_local_field_ev[:, 0],
        drive.density_dual_field_ev[:, 0] - expected_reference,
    )
    np.testing.assert_allclose(
        drive.model_local_field_ev[:, 1:],
        drive.density_dual_field_ev[:, 1:],
    )
    assert session.solve_calls == 1


def test_current_pcmsolver_map_fails_closed_for_full_coordinate_gradient():
    operator, positions, _, _ = _operator()
    atom_count = len(positions)
    zero_density = np.zeros((atom_count, 4))
    zero_coordinates = np.zeros((atom_count, 3))

    with pytest.raises(
        NotImplementedError,
        match="full reaction-field coordinate derivative",
    ):
        continuum_coupled_solvation_coordinate_gradient(
            operator,
            _ZeroDensityResponse(),
            density_coefficients=zero_density,
            intrinsic_energy_field_gradient=zero_density,
            adjoint_solution=zero_density,
            adjoint_density_position_vjp=zero_coordinates,
            solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
            gas_forces_ev_per_angstrom=zero_coordinates,
        )


def test_atom_centered_surface_pcm_full_position_vjp_matches_rebuilt_energy_fd():
    positions = np.asarray(
        [
            [-0.7, 0.1, 0.2],
            [0.8, -0.2, 0.3],
            [0.2, 0.9, -0.4],
        ]
    )
    rng = np.random.default_rng(20260725)
    density = rng.normal(size=(len(positions), 4))
    field_cotangent = rng.normal(size=(len(positions), 4))
    operator = _moving_operator(positions)

    analytic = operator.full_position_vjp(density, field_cotangent)
    finite_difference = np.empty_like(positions)
    step_angstrom = 1.0e-6

    def scalar(position_values: np.ndarray) -> float:
        displaced = _moving_operator(position_values)
        return float(np.vdot(field_cotangent, displaced.apply(density)))

    for atom_index in range(len(positions)):
        for coordinate in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            finite_difference[atom_index, coordinate] = (
                scalar(plus) - scalar(minus)
            ) / (2.0 * step_angstrom)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=3.0e-7,
        atol=3.0e-7,
    )


def test_atom_centered_surface_pcm_map_rejects_missing_or_invalid_motion_contract():
    positions = np.asarray(
        [
            [-0.7, 0.1, 0.2],
            [0.8, -0.2, 0.3],
            [0.2, 0.9, -0.4],
        ]
    )
    response = _AtomCenteredMovingResponse(positions)
    response.surface_motion_contract_version = None
    with pytest.raises(NotImplementedError, match="surface-motion"):
        AtomCenteredSurfacePCMReactionFieldLinearMap(response, positions)

    response = _AtomCenteredMovingResponse(positions)
    response.surface_motion_contract_version = 2
    with pytest.raises(ValueError, match="surface-motion"):
        AtomCenteredSurfacePCMReactionFieldLinearMap(response, positions)

    response = _AtomCenteredMovingResponse(positions)
    response._parents[-1] = -1
    with pytest.raises(ValueError, match="parent"):
        AtomCenteredSurfacePCMReactionFieldLinearMap(response, positions)


def test_fixed_cavity_pcm_map_can_reuse_surface_at_displaced_solute_positions():
    operator, positions, session, _ = _operator()
    density = np.asarray(
        [
            [0.2, -0.1, 0.3, -0.4],
            [-0.3, 0.5, -0.2, 0.1],
            [0.1, -0.4, 0.2, 0.3],
        ]
    )
    displaced_positions = positions.copy()
    displaced_positions[1, 2] += 0.015
    original_field = operator.apply(density)

    displaced = operator.at_solute_positions(displaced_positions)
    applied = displaced.apply(density)
    mep = point_multipole_potential(
        session.cavity_centers_bohr,
        displaced_positions,
        density,
    )
    asc = session.compute_asc(mep)
    potential, gradient = point_asc_reaction_potential_gradient(
        displaced_positions,
        session.cavity_centers_bohr,
        asc,
    )
    expected = np.concatenate(
        (
            (potential * Hartree)[:, None],
            gradient * Hartree / Bohr,
        ),
        axis=1,
    )

    np.testing.assert_allclose(applied, expected, rtol=2.0e-13, atol=2.0e-11)
    np.testing.assert_allclose(
        operator.apply(density),
        original_field,
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_cavity_pcm_map_rejects_surface_motion_beyond_readback_tolerance():
    operator, positions, session, _ = _operator()
    original_centers = session.cavity_centers_bohr.copy()

    session.cavity_centers_bohr[0, 0] += 0.5e-12 / Bohr
    operator.at_solute_positions(positions)

    session.cavity_centers_bohr = original_centers.copy()
    session.cavity_centers_bohr[0, 0] += 1.0e-8 / Bohr
    with pytest.raises(RuntimeError, match="surface changed"):
        operator.at_solute_positions(positions)


def test_fixed_surface_pcm_position_vjp_matches_central_difference():
    operator, positions, session, _ = _operator()
    rng = np.random.default_rng(29)
    density = rng.normal(size=(len(positions), 4))
    field_cotangent = rng.normal(size=(len(positions), 4))

    analytic = operator.position_vjp(density, field_cotangent)
    finite_difference = np.empty_like(positions)
    step_angstrom = 1.0e-6

    def scalar(position_values: np.ndarray) -> float:
        mep = point_multipole_potential(
            session.cavity_centers_bohr,
            position_values,
            density,
        )
        asc = session.compute_asc(mep)
        potential, gradient = point_asc_reaction_potential_gradient(
            position_values,
            session.cavity_centers_bohr,
            asc,
        )
        field = np.concatenate(
            (
                (potential * Hartree)[:, None],
                gradient * Hartree / Bohr,
            ),
            axis=1,
        )
        return float(np.vdot(field_cotangent, field))

    for atom_index in range(len(positions)):
        for coordinate in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            finite_difference[atom_index, coordinate] = (
                scalar(plus) - scalar(minus)
            ) / (2.0 * step_angstrom)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-7,
    )


def test_fixed_cavity_pcm_map_requires_hermitivized_response():
    with pytest.raises(ValueError, match="MATRIXSYMM=TRUE"):
        _operator(symmetric=False)


def test_fixed_cavity_pcm_map_rejects_unknown_continuum_contract_version():
    _, positions, _, response = _operator()
    response.contract_version = 2

    with pytest.raises(ValueError, match="contract version"):
        FixedCavityPCMReactionFieldLinearMap(response, positions)


def test_pcmsolver_map_fails_closed_without_operator_derivative():
    operator, positions, _, _ = _operator()

    with pytest.raises(NotImplementedError, match="operator derivative"):
        operator.continuum_operator_position_vjp(
            np.zeros((len(positions), 4)),
            np.zeros((len(positions), 4)),
        )


def test_fixed_cavity_pcm_map_rejects_geometry_mismatch():
    _, positions, _, response = _operator()
    shifted = positions.copy()
    shifted[0, 0] += 1.0e-5

    with pytest.raises(ValueError, match="does not match"):
        FixedCavityPCMReactionFieldLinearMap(response, shifted)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.zeros((2, 4)),
        np.full((3, 4), np.nan),
    ],
)
def test_fixed_cavity_pcm_map_rejects_invalid_blocks(bad_values):
    operator, _, _, _ = _operator()

    with pytest.raises(ValueError, match="finite with shape"):
        operator.apply(bad_values)
    with pytest.raises(ValueError, match="finite with shape"):
        operator.adjoint(bad_values)
    with pytest.raises(ValueError, match="finite with shape"):
        operator.position_vjp(bad_values, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="finite with shape"):
        operator.position_vjp(np.zeros((3, 4)), bad_values)
