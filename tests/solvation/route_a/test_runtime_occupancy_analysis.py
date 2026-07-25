from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import kB

from maple.function.dispatcher.solvfe.analysis import ReducedPotentialTable
from maple.function.dispatcher.solvfe.membership import soft_occupancy_weights
from maple.function.dispatcher.solvfe.occupancy_analysis import (
    SoftOccupancyAnalysisInput,
    SoftOccupancyContract,
    SoftOccupancyDistribution,
    estimate_soft_occupancy,
)


def _contract(
    *,
    ensemble_role: str = "reference-product",
    target_state_index: int = 1,
    sampling_measure_id: str = "reference-soft-occupancy-v3",
    active_occupancy_max: int = 1,
    reconstruction_bin_index: int = 1,
) -> SoftOccupancyContract:
    return SoftOccupancyContract(
        ensemble_role=ensemble_role,
        target_state_index=target_state_index,
        sampling_measure_id=sampling_measure_id,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="5" * 64,
        sample_independence_hash="6" * 64,
        temperature_k=298.15,
        boundary_conditions="periodic-3d",
        active_occupancy_max=active_occupancy_max,
        reconstruction_bin_index=reconstruction_bin_index,
    )


def _table(
    *,
    measure_id: str = "reference-soft-occupancy-v3",
    boundary_conditions: str = "periodic-3d",
) -> ReducedPotentialTable:
    sample_count = 240
    return ReducedPotentialTable.create(
        u_kn=np.vstack(
            [
                np.full(sample_count, 0.4),
                np.zeros(sample_count),
            ]
        ),
        N_k=(120, 120),
        row_labels=("biased", "target"),
        frame_ids=tuple(f"frame-{index}" for index in range(sample_count)),
        beta=1.0 / (kB * 298.15),
        measure_id=measure_id,
        boundary_conditions=boundary_conditions,
    )


def test_soft_occupancy_estimator_uses_explicit_target_and_full_covariance():
    phase = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    occupancy_weights = np.column_stack(
        (
            0.2 + 0.05 * np.sin(phase),
            0.5 + 0.04 * np.cos(phase),
            np.zeros_like(phase),
        )
    )
    occupancy_weights[:, 2] = 1.0 - np.sum(
        occupancy_weights[:, :2],
        axis=1,
    )
    analysis_input = SoftOccupancyAnalysisInput.create(
        table=_table(),
        contract=_contract(),
        occupancy_weights=occupancy_weights,
    )

    estimate = estimate_soft_occupancy(analysis_input)
    distribution = estimate.distribution

    assert analysis_input.target_state_index == 1
    assert distribution.ensemble_role == "reference-product"
    assert distribution.occupancies == (0, 1)
    assert distribution.probabilities == pytest.approx([0.2, 0.5])
    assert distribution.tail_probability == pytest.approx(0.3)
    assert (
        np.sum(distribution.probabilities) + distribution.tail_probability
        == pytest.approx(1.0)
    )
    assert distribution.covariance_of_mean.shape == (3, 3)
    assert distribution.covariance_of_mean @ np.ones(3) == pytest.approx(
        np.zeros(3),
        abs=1.0e-12,
    )
    assert np.linalg.eigvalsh(
        distribution.covariance_of_mean
    ).min() >= -1.0e-12
    assert estimate.covariance_projection_max_abs < 1.0e-10
    variance_0 = (0.05**2 / 2.0) / 240
    variance_1 = (0.04**2 / 2.0) / 240
    expected_covariance = np.asarray(
        [
            [variance_0, 0.0, -variance_0],
            [0.0, variance_1, -variance_1],
            [-variance_0, -variance_1, variance_0 + variance_1],
        ]
    )
    assert estimate.raw_covariance_of_mean == pytest.approx(
        expected_covariance,
        rel=1.0e-10,
        abs=1.0e-15,
    )
    assert distribution.covariance_of_mean == pytest.approx(
        expected_covariance,
        rel=1.0e-10,
        abs=1.0e-15,
    )
    assert estimate.content_hash


def test_constant_soft_occupancy_has_exact_zero_covariance():
    occupancy_weights = np.tile([0.25, 0.5, 0.25], (240, 1))
    estimate = estimate_soft_occupancy(
        SoftOccupancyAnalysisInput.create(
            table=_table(),
            contract=_contract(),
            occupancy_weights=occupancy_weights,
        )
    )

    assert estimate.distribution.probabilities == pytest.approx(
        [0.25, 0.5]
    )
    assert estimate.distribution.tail_probability == pytest.approx(0.25)
    assert estimate.distribution.covariance_of_mean == pytest.approx(
        np.zeros((3, 3)),
        abs=1.0e-15,
    )


@pytest.mark.parametrize(
    ("table", "contract", "weights", "message"),
    [
        (
            _table(measure_id="wrong"),
            _contract(),
            np.tile([0.2, 0.8], (240, 1)),
            "sampling measure",
        ),
        (
            _table(boundary_conditions="nonperiodic"),
            _contract(),
            np.tile([0.2, 0.8], (240, 1)),
            "boundary",
        ),
        (
            _table(),
            _contract(target_state_index=0),
            np.tile([0.2, 0.8], (240, 1)),
            "target state label",
        ),
        (
            _table(),
            _contract(),
            np.tile([0.2, 0.7, 0.0], (240, 1)),
            "normalization",
        ),
    ],
)
def test_soft_occupancy_input_rejects_measure_or_probability_drift(
    table,
    contract,
    weights,
    message,
):
    with pytest.raises(ValueError, match=message):
        SoftOccupancyAnalysisInput.create(
            table=table,
            contract=contract,
            occupancy_weights=weights,
        )


def test_distribution_rejects_covariance_that_violates_simplex_nullspace():
    with pytest.raises(ValueError, match="simplex"):
        SoftOccupancyDistribution.create(
            ensemble_role="coupled-solution",
            probabilities=np.asarray([0.4, 0.6]),
            tail_probability=0.0,
            covariance_of_mean=np.eye(3) * 0.01,
            membership_definition_hash="1" * 64,
            observation_volume_hash="2" * 64,
            solute_measure_hash="3" * 64,
            water_hamiltonian_hash="4" * 64,
            system_hamiltonian_hash="5" * 64,
            temperature_k=298.15,
            boundary_conditions="periodic-3d",
            source_artifact_hash="6" * 64,
            estimator="test",
        )


@pytest.mark.parametrize("molecule_count", [64, 128, 256, 512])
def test_large_water_box_uses_preregistered_support_and_tail_bin(
    molecule_count,
):
    memberships = np.full((240, molecule_count), 0.01)
    raw_weights = soft_occupancy_weights(
        memberships,
        active_occupancy_max=8,
    )
    estimate = estimate_soft_occupancy(
        SoftOccupancyAnalysisInput.create(
            table=_table(),
            contract=_contract(
                active_occupancy_max=8,
                reconstruction_bin_index=2,
            ),
            occupancy_weights=raw_weights,
        )
    )

    expected = np.asarray(
        [
            math.comb(molecule_count, occupancy)
            * 0.01**occupancy
            * 0.99 ** (molecule_count - occupancy)
            for occupancy in range(9)
        ]
    )
    assert estimate.distribution.probabilities == pytest.approx(
        expected,
        rel=1.0e-10,
        abs=1.0e-15,
    )
    assert estimate.distribution.tail_probability == pytest.approx(
        1.0 - float(np.sum(expected)),
        rel=1.0e-10,
        abs=1.0e-14,
    )
    assert estimate.distribution.covariance_of_mean == pytest.approx(
        np.zeros((10, 10)),
        abs=1.0e-15,
    )
