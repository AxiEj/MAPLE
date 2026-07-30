from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    route2_v0_response_kernel as response_kernel,
)


ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> dict[str, np.ndarray]:
    coefficient_count = 8
    atom_dipole_map = np.zeros((6, coefficient_count), dtype=float)
    atom_dipole_map[:, :6] = np.eye(6)
    baseline = np.diag([1.2, 0.9, 1.1, 1.4, 0.8, 1.3, 0.7, 0.0])
    partition = np.vstack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    polarizability = np.asarray(
        [[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 1.2]],
        dtype=float,
    )
    charge_constraint = np.eye(coefficient_count)[-1]
    return {
        "baseline_response_covariance_coefficient_dual": baseline,
        "atom_dipole_map_coefficient_to_ebohr": atom_dipole_map,
        "atomic_dipole_partition_molecular_to_ebohr": partition,
        "molecular_polarizability_bohr3": polarizability,
        "charge_constraint_vector": charge_constraint,
    }


def test_response_kernel_completion_preserves_moments_null_response_and_scalar():
    inputs = _inputs()
    state = response_kernel.complete_route2_v0_response_kernel(**inputs)
    atom_sum = np.hstack((np.eye(3), np.eye(3)))
    molecular_dipole_map = atom_sum @ inputs["atom_dipole_map_coefficient_to_ebohr"]
    response = state.response_covariance_coefficient_dual

    np.testing.assert_allclose(
        inputs["atom_dipole_map_coefficient_to_ebohr"]
        @ response
        @ inputs["atom_dipole_map_coefficient_to_ebohr"].T,
        state.target_atom_dipole_covariance_bohr3,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        molecular_dipole_map @ response @ molecular_dipole_map.T,
        inputs["molecular_polarizability_bohr3"],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response @ inputs["charge_constraint_vector"],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert state.completed_minimum_eigenvalue >= -1.0e-12
    assert state.atom_covariance_error < 1.0e-12
    assert state.molecular_polarizability_error < 1.0e-12
    assert state.moore_penrose_error < 1.0e-12
    assert state.charge_nullspace_projection_error < 1.0e-12
    np.testing.assert_allclose(
        state.response_support_constraints
        @ inputs["charge_constraint_vector"],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    all_constraints = np.vstack(
        (inputs["charge_constraint_vector"], state.response_support_constraints)
    )
    assert np.linalg.matrix_rank(all_constraints) == (
        response.shape[0] - np.linalg.matrix_rank(state.response_support_projector)
    )

    external_dual = np.asarray([0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.7, 0.6])
    induced = -response @ external_dual
    np.testing.assert_allclose(
        state.response_support_constraints @ induced,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.electronic_curvature_coefficient_dual @ induced
        + state.response_support_projector @ external_dual,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_response_kernel_completion_preserves_baseline_null_moment_response():
    state = response_kernel.complete_route2_v0_response_kernel(**_inputs())

    assert state.conditional_null_covariance_coefficient_dual[6, 6] == pytest.approx(
        0.7,
        abs=1.0e-12,
    )
    assert state.response_covariance_coefficient_dual[6, 6] == pytest.approx(
        0.7,
        abs=1.0e-12,
    )


def test_response_kernel_separates_a_rotated_charge_null_mode_from_kkt_rows():
    generator = np.random.default_rng(20260731)
    coefficient_count = 8
    rotation, _ = np.linalg.qr(
        generator.normal(size=(coefficient_count, coefficient_count))
    )
    charge_constraint = rotation[:, -1]
    baseline = rotation @ np.diag(
        [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.0]
    ) @ rotation.T
    atom_dipole_map = generator.normal(size=(6, coefficient_count))
    partition = np.vstack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    state = response_kernel.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=baseline,
        atom_dipole_map_coefficient_to_ebohr=atom_dipole_map,
        atomic_dipole_partition_molecular_to_ebohr=partition,
        molecular_polarizability_bohr3=np.diag([1.7, 1.4, 1.1]),
        charge_constraint_vector=charge_constraint,
    )

    np.testing.assert_allclose(
        state.response_support_constraints @ charge_constraint,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    all_constraints = np.vstack(
        (charge_constraint, state.response_support_constraints)
    )
    assert np.linalg.matrix_rank(all_constraints) == (
        coefficient_count
        - np.linalg.matrix_rank(state.response_support_projector)
    )


def test_response_kernel_rejects_a_partition_that_does_not_sum_to_identity():
    inputs = _inputs()
    inputs["atomic_dipole_partition_molecular_to_ebohr"][3:] = 0.5 * np.eye(3)

    with pytest.raises(ValueError, match="sum exactly to the molecular dipole"):
        response_kernel.complete_route2_v0_response_kernel(**inputs)


def test_response_kernel_rejects_nonpositive_baseline_covariance_without_clipping():
    inputs = _inputs()
    inputs["baseline_response_covariance_coefficient_dual"][6, 6] = -0.1

    with pytest.raises(ValueError, match="positive semidefinite"):
        response_kernel.complete_route2_v0_response_kernel(**inputs)


def test_response_kernel_rejects_an_atom_covariance_with_missing_support():
    inputs = _inputs()
    inputs["atom_dipole_map_coefficient_to_ebohr"][-1] = 0.0

    with pytest.raises(
        ValueError,
        match="Baseline atom-dipole covariance must be positive",
    ):
        response_kernel.complete_route2_v0_response_kernel(**inputs)


def test_response_kernel_rejects_charge_response_drift_without_repair():
    inputs = _inputs()
    inputs["baseline_response_covariance_coefficient_dual"][-1, -1] = 0.1

    with pytest.raises(ValueError, match="charge-neutral response constraint"):
        response_kernel.complete_route2_v0_response_kernel(**inputs)


def test_response_kernel_preregistration_locks_the_no_fit_source_boundary():
    protocol = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-response-kernel-completion-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    theory = (
        ROOT
        / "docs/implicit-solvation/ROUTE2_V0_RESPONSE_KERNEL_THEORY.md"
    ).read_text(encoding="utf-8")

    assert protocol["status"] == (
        "structural-kernel-implemented-before-source-bound-response-data"
    )
    assert protocol["construction"]["name"] == (
        "route2-v0-response-kernel-completion-v1"
    )
    hard_constraints = protocol["hard_constraints"]
    assert hard_constraints["post_training"] is False
    assert hard_constraints["fine_tuning"] is False
    assert hard_constraints["experimental_solvation_fit"] is False
    assert hard_constraints["response_eigenvalue_clipping"] is False
    assert hard_constraints["physical_response_kernel_execution"] is False
    forbidden = " ".join(protocol["forbidden_shortcuts"])
    assert "FreeSolv" in forbidden
    assert "MNSol" in forbidden
    assert "radial width" in forbidden
    assert "Schur-complement" in theory
    assert "physical response-kernel asset" in theory
