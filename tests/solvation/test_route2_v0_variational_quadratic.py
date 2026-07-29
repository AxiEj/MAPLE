from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_variational_quadratic import (
    V0_VARIATIONAL_QUADRATIC_CONSTRUCTION,
    embed_mace_polar_l1_density_in_single_radial_gto,
    solve_route2_v0_variational_quadratic,
)


ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-variational-quadratic-prereg-v1.json"
)


class _ReciprocalSurfaceResponse:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._positions = np.asarray(positions_angstrom, dtype=float)
        self._points = np.asarray(surface_points_bohr, dtype=float)
        self._response = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self):
        return np.ones(self.atom_count)

    @property
    def reference_positions_bohr(self):
        return self._positions / Bohr

    @property
    def cavity_radii_angstrom(self):
        return np.full(self.atom_count, 1.5)

    @property
    def surface_points_bohr(self):
        return self._points.copy()

    @property
    def surface_areas_bohr2(self):
        return np.ones(len(self._points))

    def apply_energy_conjugate(self, potential):
        return self._response @ np.asarray(potential, dtype=float)


def _operator() -> FixedCavityGTOGalerkinOperator:
    positions = np.asarray([[-0.8, 0.2, -0.1], [0.9, -0.3, 0.4]], dtype=float)
    rng = np.random.default_rng(20260729)
    points = rng.normal(size=(20, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= rng.uniform(4.0, 6.0, size=(20, 1))
    factor = rng.normal(size=(20, 20))
    response = -(factor.T @ factor) / 80.0
    return FixedCavityGTOGalerkinOperator(
        _ReciprocalSurfaceResponse(positions, points, response),
        positions,
        AtomCenteredL1GTOBasis((MACE_POLAR_DENSITY_SIGMA_ANGSTROM,)),
    )


def _frozen_density(operator: FixedCavityGTOGalerkinOperator) -> np.ndarray:
    density = np.asarray(
        [[0.35, 0.13, -0.08, 0.06], [-0.35, -0.03, 0.09, -0.11]],
        dtype=float,
    )
    return embed_mace_polar_l1_density_in_single_radial_gto(
        density,
        operator.basis,
    )


def _stable_curvature(operator: FixedCavityGTOGalerkinOperator) -> np.ndarray:
    return np.eye(operator.coefficient_count) * 10.0


def test_single_radial_mace_embedding_is_exact_and_not_a_projection():
    operator = _operator()
    native = np.asarray([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])

    embedded = embed_mace_polar_l1_density_in_single_radial_gto(
        native,
        operator.basis,
    )

    assert embedded.shape == (2, 1, 4)
    np.testing.assert_array_equal(embedded[:, 0, :], native)
    assert embedded.flags.writeable is False


def test_mace_embedding_rejects_a_new_radial_decomposition_or_width():
    native = np.zeros((2, 4))

    with pytest.raises(ValueError, match="single-radial"):
        embed_mace_polar_l1_density_in_single_radial_gto(
            native,
            AtomCenteredL1GTOBasis((1.5, 3.0)),
        )
    with pytest.raises(ValueError, match="density width"):
        embed_mace_polar_l1_density_in_single_radial_gto(
            native,
            AtomCenteredL1GTOBasis((1.6,)),
        )


def test_quadratic_preregistration_locks_the_no_training_physical_boundary():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == "route2-v0-variational-quadratic-prereg-v1"
    assert protocol["status"] == (
        "structural-kernel-preregistered-before-physical-curvature-binding"
    )
    construction = protocol["construction"]
    assert construction["name"] == V0_VARIATIONAL_QUADRATIC_CONSTRUCTION
    assert "0.5 delta_c^T H_R delta_c" in construction["electronic_functional"]
    assert "legacy learned local-jet and exact-GTO" in construction["induced_density"]

    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "density_projection_or_radial_decomposition": False,
        "learned_response_jacobian_symmetrization": False,
        "response_tempering_or_eigenvalue_clipping": False,
        "per_record_or_per_solvent_method_selection_from_error": False,
        "legacy_public_route_changed": False,
    }
    binding = protocol["physical_curvature_binding_gate"]
    assert any("no target solvation value" in item for item in binding[
        "required_before_any_chemistry_score"
    ])
    forbidden = " ".join(binding["forbidden_shortcuts"])
    assert "isolated-atom hardness" in forbidden
    assert "rejected MACE learned field update" in forbidden
    gates = " ".join(protocol["structural_acceptance_gates"])
    assert "strictly positive definite" in gates
    assert "symmetric and nonpositive" in gates
    qeq = protocol["frozen_physical_bindings"][
        "rappe_goddard_hardness_same_basis_monopole_tangent"
    ]
    assert qeq["construction"] == (
        "route2-v0-rappe-goddard-hardness-same-basis-monopole-v1"
    )
    assert qeq["parameter_table_sha256"] == (
        "5d2b405b78dd59b95da89b69fb10b409bcb3f0f85921fc8cc4a67dd2aba8a618"
    )
    assert "Every l=1 coefficient is an exact homogeneous KKT constraint" in (
        qeq["response_subspace"]
    )


def test_quadratic_kkt_state_is_stationary_charge_conserving_and_one_ledger():
    operator = _operator()
    state = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=_frozen_density(operator),
        operator=operator,
        electronic_curvature_coefficient_dual=_stable_curvature(operator),
    )

    assert state.construction == V0_VARIATIONAL_QUADRATIC_CONSTRUCTION
    assert state.stationarity_residual_inf < 1.0e-11
    assert state.charge_constraint_residual_e < 1.0e-12
    assert state.stationary_total_charge_e == pytest.approx(0.0, abs=1.0e-12)
    assert state.electronic_minimum_neutral_curvature > 0.0
    assert state.joint_minimum_neutral_curvature > 0.0
    np.testing.assert_allclose(
        state.stationary_density_coefficients,
        state.frozen_density_coefficients + state.induced_density_coefficients,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert state.continuum_polarization_energy_hartree == pytest.approx(
        state.continuum_snapshot.polarization_energy_hartree,
        abs=1.0e-13,
    )
    assert state.solute_continuum_energy_hartree == pytest.approx(
        state.electronic_induction_energy_hartree
        + state.continuum_polarization_energy_hartree,
        abs=1.0e-13,
    )
    assert state.stationary_total_energy_hartree == pytest.approx(
        state.solute_continuum_energy_hartree,
        abs=1.0e-13,
    )


def test_joint_response_is_reciprocal_passive_and_matches_kkt_finite_difference():
    operator = _operator()
    frozen = _frozen_density(operator)
    curvature = _stable_curvature(operator)
    state = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=curvature,
    )
    response = state.joint_external_dual_response

    np.testing.assert_allclose(response, response.T, rtol=0.0, atol=2.0e-14)
    assert np.max(np.linalg.eigvalsh(response)) <= 2.0e-14
    np.testing.assert_allclose(
        response @ state.charge_constraint_vector,
        np.zeros(operator.coefficient_count),
        rtol=0.0,
        atol=2.0e-14,
    )

    direction = np.random.default_rng(17).normal(size=operator.coefficient_count)
    direction -= state.charge_constraint_vector * (
        np.dot(state.charge_constraint_vector, direction)
        / np.dot(state.charge_constraint_vector, state.charge_constraint_vector)
    )
    step = 1.0e-6
    plus = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=curvature,
        external_coefficient_dual_hartree=operator.basis.unflatten(
            step * direction,
            atom_count=operator.atom_count,
        ),
    )
    minus = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=curvature,
        external_coefficient_dual_hartree=operator.basis.unflatten(
            -step * direction,
            atom_count=operator.atom_count,
        ),
    )
    finite_difference = (
        plus.stationary_density_coefficients.reshape(-1)
        - minus.stationary_density_coefficients.reshape(-1)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference,
        response @ direction,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_additional_linear_constraints_freeze_unmodeled_dipole_response_exactly():
    operator = _operator()
    frozen = _frozen_density(operator)
    coefficient_shape = frozen.shape
    dipole_indices = [
        np.ravel_multi_index((atom, radial, component), coefficient_shape)
        for atom in range(coefficient_shape[0])
        for radial in range(coefficient_shape[1])
        for component in (1, 2, 3)
    ]
    dipole_constraints = np.eye(operator.coefficient_count)[dipole_indices]

    state = solve_route2_v0_variational_quadratic(
        frozen_density_coefficients=frozen,
        operator=operator,
        electronic_curvature_coefficient_dual=_stable_curvature(operator),
        induced_constraints=dipole_constraints,
    )

    assert state.additional_constraint_residual_inf < 1.0e-12
    np.testing.assert_allclose(
        state.induced_density_coefficients[:, :, 1:],
        np.zeros_like(state.induced_density_coefficients[:, :, 1:]),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.induced_constraint_matrix @ state.joint_external_dual_response,
        np.zeros((len(dipole_indices) + 1, operator.coefficient_count)),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_quadratic_kkt_rejects_nonvariational_or_unstable_curvature():
    operator = _operator()
    frozen = _frozen_density(operator)
    nonsymmetric = _stable_curvature(operator)
    nonsymmetric[0, 1] = 0.5
    with pytest.raises(ValueError, match="must be symmetric"):
        solve_route2_v0_variational_quadratic(
            frozen_density_coefficients=frozen,
            operator=operator,
            electronic_curvature_coefficient_dual=nonsymmetric,
        )

    continuum = operator.galerkin_matrix_hartree
    scale = max(1.0, float(np.max(np.abs(np.linalg.eigvalsh(continuum)))))
    unstable = -0.5 * continuum + 1.0e-6 * scale * np.eye(
        operator.coefficient_count
    )
    with pytest.raises(RuntimeError, match="Joint electronic-continuum curvature"):
        solve_route2_v0_variational_quadratic(
            frozen_density_coefficients=frozen,
            operator=operator,
            electronic_curvature_coefficient_dual=unstable,
        )


def test_quadratic_kkt_rejects_a_frozen_charge_mismatch():
    operator = _operator()
    frozen = _frozen_density(operator).copy()
    frozen[0, 0, 0] += 0.01

    with pytest.raises(ValueError, match="total-charge constraint"):
        solve_route2_v0_variational_quadratic(
            frozen_density_coefficients=frozen,
            operator=operator,
            electronic_curvature_coefficient_dual=_stable_curvature(operator),
        )
