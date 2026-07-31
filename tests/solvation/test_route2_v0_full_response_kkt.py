from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_independent_particle_response as atomic_response,
    route2_v0_atomic_independent_particle_surface as atomic_surface,
    route2_v0_full_response_kkt as full_kkt,
    route2_v0_response_kernel as response_kernel,
)


ROOT = Path(__file__).resolve().parents[2]


class _LinearReciprocalContinuum:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        atomic_numbers: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._positions_bohr = (
            np.asarray(positions_angstrom, dtype=float)
            / atomic_surface.BOHR_ANGSTROM
        )
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._surface_points_bohr = np.asarray(surface_points_bohr, dtype=float)
        self._response_matrix = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_bohr.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return self._surface_points_bohr.copy()

    def apply_energy_conjugate(self, potential: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(potential, dtype=float)


def _baseline() -> atomic_response.Route2V0AtomicIndependentParticleBaseline:
    atom_numbers = np.asarray([1, 1])
    map_to_dipoles = np.zeros((6, 6))
    map_to_dipoles[:3, :3] = -np.eye(3)
    map_to_dipoles[3:, 3:] = -np.eye(3)
    return atomic_response.Route2V0AtomicIndependentParticleBaseline(
        atomic_numbers=atom_numbers,
        response_weights_hartree_inverse=np.ones(6),
        transition_density_dipoles_ebohr=np.vstack((np.eye(3), np.eye(3))),
        atom_dipole_map_coefficient_to_ebohr=map_to_dipoles,
        transition_charges_e=np.zeros(6),
        coefficient_slices=(slice(0, 3), slice(3, 6)),
    )


def _completion() -> response_kernel.Route2V0ResponseKernelCompletion:
    baseline = _baseline()
    return response_kernel.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=np.vstack(
            (0.5 * np.eye(3), 0.5 * np.eye(3))
        ),
        molecular_polarizability_bohr3=2.0 * np.eye(3),
        charge_constraint_vector=None,
    )


def _molecular_moment_completion(
) -> response_kernel.Route2V0MolecularMomentResponseKernelCompletion:
    baseline = _baseline()
    return response_kernel.complete_molecular_moment_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        molecular_polarizability_bohr3=2.0 * np.eye(3),
        charge_constraint_vector=None,
    )


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.8, -0.3, 0.5]])
    numbers = np.asarray([1, 1])
    points = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [-2.5, 1.5, 0.7],
            [1.1, -2.7, 1.8],
            [-1.2, -0.8, -2.5],
            [2.2, 1.3, -1.7],
        ]
    )
    operator = np.asarray(
        [
            [0.11, -0.03, 0.02, 0.04, 0.01, -0.05],
            [-0.02, 0.08, 0.05, -0.01, 0.06, 0.03],
            [0.04, 0.02, 0.09, -0.04, 0.01, 0.07],
            [0.03, -0.05, 0.01, 0.08, -0.02, 0.02],
            [-0.06, 0.01, -0.04, 0.02, 0.07, -0.01],
            [0.05, 0.03, -0.02, -0.07, 0.04, 0.06],
            [-0.01, 0.06, 0.04, 0.03, -0.05, 0.08],
        ]
    )
    return positions, numbers, points, operator


def _coupling() -> atomic_surface.AtomicIndependentParticleSurfaceCoupling:
    positions, _, points, operator = _geometry()
    return atomic_surface.AtomicIndependentParticleSurfaceCoupling(
        baseline=_baseline(),
        atom_positions_angstrom=positions,
        surface_points_bohr=points,
        surface_operator_hartree_per_e_per_coefficient=operator,
    )


def _continuum(response_scale: float = -0.02) -> _LinearReciprocalContinuum:
    positions, numbers, points, _ = _geometry()
    return _LinearReciprocalContinuum(
        positions,
        numbers,
        points,
        response_scale * np.eye(len(points)),
    )


def test_atomic_independent_particle_surface_uses_the_exact_matrix_transpose():
    coupling = _coupling()
    coefficients = np.asarray([0.4, -0.2, 0.1, -0.3, 0.5, -0.1])
    charge = np.asarray([0.1, -0.2, 0.4, -0.1, 0.3, -0.2, 0.05])

    assert coupling.source_duality_error(coefficients, charge) < 1.0e-14
    assert coupling.surface_potential(coefficients).shape == (7,)
    assert coupling.surface_to_coefficient_dual(charge).shape == (6,)


def test_full_response_kkt_preregistration_locks_common_scalar_boundary():
    protocol = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-full-response-kkt-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    theory = (
        ROOT / "docs/implicit-solvation/ROUTE2_V0_FULL_RESPONSE_KKT_THEORY.md"
    ).read_text(encoding="utf-8")

    assert protocol["status"] == (
        "structural-kernel-implemented-before-physical-continuum-binding"
    )
    assert protocol["construction"]["name"] == "route2-v0-full-response-kkt-v1"
    assert all(value is False for value in protocol["hard_constraints"].values())
    forbidden = " ".join(protocol["forbidden_shortcuts"])
    assert "FreeSolv" in forbidden
    assert "MNSol" in forbidden
    assert "eigenvalue" in forbidden
    assert "same coefficient/dual space" in theory
    assert "not a permanent-density model" in theory


def test_full_response_kkt_has_one_ledger_and_support_constrained_stationarity():
    coupling = _coupling()
    permanent = np.asarray([0.08, -0.04, 0.03, 0.01, -0.05, 0.02, 0.06])
    external = np.asarray([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])
    state = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=_completion(),
        continuum=_continuum(),
        permanent_surface_potential_hartree_per_e=permanent,
        external_coefficient_dual_hartree=external,
    )

    assert np.linalg.norm(state.induced_coefficients) > 1.0e-8
    assert state.source_duality_error_hartree < 1.0e-14
    assert state.continuum_linearity_error_e < 1.0e-14
    assert state.kkt_residual_inf_hartree < 1.0e-12
    assert state.support_constraint_residual_inf < 1.0e-12
    assert state.support_minimum_curvature_hartree > state.support_stability_threshold_hartree
    np.testing.assert_allclose(
        state.total_surface_charge_e,
        state.permanent_surface_charge_e + state.induced_surface_charge_e,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert state.stationary_total_energy_hartree == pytest.approx(
        state.electronic_induction_energy_hartree
        + state.continuum_polarization_energy_hartree
        + state.external_work_hartree,
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        state.external_response_coefficient_dual,
        state.external_response_coefficient_dual.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.max(np.linalg.eigvalsh(state.external_response_coefficient_dual)) <= 1.0e-12


def test_full_response_kkt_accepts_the_molecular_moment_completion():
    """The corrected no-fit response still shares the same scalar KKT solve."""

    state = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=_coupling(),
        completion=_molecular_moment_completion(),
        continuum=_continuum(),
        permanent_surface_potential_hartree_per_e=np.asarray(
            [0.08, -0.04, 0.03, 0.01, -0.05, 0.02, 0.06]
        ),
        external_coefficient_dual_hartree=np.asarray(
            [0.02, -0.01, 0.03, -0.02, 0.01, 0.04]
        ),
    )

    assert state.kkt_residual_inf_hartree < 1.0e-12
    assert state.support_constraint_residual_inf < 1.0e-12
    assert state.support_minimum_curvature_hartree > state.support_stability_threshold_hartree
    np.testing.assert_allclose(
        state.external_response_coefficient_dual,
        state.external_response_coefficient_dual.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.max(np.linalg.eigvalsh(state.external_response_coefficient_dual)) <= 1.0e-12


def test_full_response_kkt_obeys_external_and_permanent_envelope_identities():
    coupling = _coupling()
    completion = _completion()
    continuum = _continuum()
    permanent = np.asarray([0.08, -0.04, 0.03, 0.01, -0.05, 0.02, 0.06])
    external = np.asarray([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])
    state = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=permanent,
        external_coefficient_dual_hartree=external,
    )
    step = 1.0e-6
    direction = np.eye(6)[0]
    plus = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=permanent,
        external_coefficient_dual_hartree=external + step * direction,
    )
    minus = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=permanent,
        external_coefficient_dual_hartree=external - step * direction,
    )
    np.testing.assert_allclose(
        (plus.induced_coefficients - minus.induced_coefficients) / (2.0 * step),
        state.external_response_coefficient_dual[:, 0],
        rtol=0.0,
        atol=1.0e-9,
    )
    finite_difference = (
        plus.stationary_total_energy_hartree - minus.stationary_total_energy_hartree
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(state.induced_coefficients[0], abs=1.0e-9)

    plus_scale = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=(1.0 + step) * permanent,
        external_coefficient_dual_hartree=external,
    )
    minus_scale = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=(1.0 - step) * permanent,
        external_coefficient_dual_hartree=external,
    )
    permanent_fd = (
        plus_scale.stationary_total_energy_hartree
        - minus_scale.stationary_total_energy_hartree
    ) / (2.0 * step)
    assert permanent_fd == pytest.approx(
        float(permanent @ state.total_surface_charge_e),
        rel=2.0e-8,
        abs=2.0e-10,
    )


def test_full_response_kkt_response_does_not_identify_permanent_reference():
    coupling = _coupling()
    completion = _completion()
    continuum = _continuum()
    external = np.asarray([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])
    reference_coefficients = np.asarray(
        [0.09, -0.03, 0.06, 0.02, -0.04, 0.05]
    )
    absent_reference = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        external_coefficient_dual_hartree=external,
    )
    shifted_reference = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=coupling,
        completion=completion,
        continuum=continuum,
        permanent_surface_potential_hartree_per_e=coupling.surface_potential(
            reference_coefficients
        ),
        external_coefficient_dual_hartree=external,
    )

    # The stationary linear response is set by the curvature only.  A neutral
    # reference-state shift changes the permanent source and on-shell scalar,
    # but cannot be inferred from reciprocity/passivity of that response.
    np.testing.assert_allclose(
        shifted_reference.external_response_coefficient_dual,
        absent_reference.external_response_coefficient_dual,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.linalg.norm(
        shifted_reference.total_surface_charge_e
        - absent_reference.total_surface_charge_e
    ) > 1.0e-8
    assert shifted_reference.stationary_total_energy_hartree != pytest.approx(
        absent_reference.stationary_total_energy_hartree,
        abs=1.0e-12,
    )


def test_full_response_kkt_rejects_nonreciprocal_or_unstable_continuum():
    coupling = _coupling()
    positions, numbers, points, _ = _geometry()
    nonreciprocal = -0.02 * np.eye(len(points))
    nonreciprocal[0, 1] = 0.1
    with pytest.raises(RuntimeError, match="not reciprocal"):
        full_kkt.solve_route2_v0_full_response_kkt(
            coupling=coupling,
            completion=_completion(),
            continuum=_LinearReciprocalContinuum(
                positions, numbers, points, nonreciprocal
            ),
        )
    with pytest.raises(RuntimeError, match="not positive on support"):
        full_kkt.solve_route2_v0_full_response_kkt(
            coupling=coupling,
            completion=_completion(),
            continuum=_continuum(response_scale=-100.0),
        )
    with pytest.raises(RuntimeError, match="not passive"):
        full_kkt.solve_route2_v0_full_response_kkt(
            coupling=coupling,
            completion=_completion(),
            continuum=_continuum(response_scale=0.02),
        )


def test_full_response_state_rejects_a_mutated_energy_or_response_ledger():
    state = full_kkt.solve_route2_v0_full_response_kkt(
        coupling=_coupling(),
        completion=_completion(),
        continuum=_continuum(),
        external_coefficient_dual_hartree=np.asarray(
            [0.02, -0.01, 0.03, -0.02, 0.01, 0.04]
        ),
    )

    with pytest.raises(ValueError, match="energy ledger"):
        replace(state, stationary_total_energy_hartree=1.0)
    with pytest.raises(ValueError, match="external response"):
        replace(
            state,
            external_response_coefficient_dual=np.eye(
                len(state.induced_coefficients)
            ),
        )
