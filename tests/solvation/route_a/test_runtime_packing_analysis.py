from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase.units import kB

from maple.function.dispatcher.solvfe.analysis import ReducedPotentialTable
from maple.function.dispatcher.solvfe.occupancy_analysis import (
    SoftOccupancyAnalysisInput,
    SoftOccupancyContract,
    SoftOccupancyDistribution,
    estimate_soft_occupancy,
)
from maple.function.dispatcher.solvfe.packing_analysis import (
    PackingAnalysisInput,
    estimate_soft_packing,
)
from maple.function.dispatcher.solvfe.packing_contract import (
    PackingBiasState,
    PackingSchedule,
)
from maple.function.dispatcher.solvfe.qct_ledger import (
    PackingOccupancyBridge,
)


def _schedule() -> PackingSchedule:
    # Deliberately descending so target state is not row zero.
    return PackingSchedule(
        states=(
            PackingBiasState("full-field", 1.0),
            PackingBiasState("unbiased", 0.0),
        ),
        target_state_index=1,
        full_field_state_index=0,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash="7" * 64,
        conditioning_measure_id="product-soft-packing-v3",
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        oxygen_atom_map_hash="5" * 64,
        solute_atom_map_hash="6" * 64,
        cell_hash="7" * 64,
        active_occupancy_max=1,
        temperature_k=298.15,
        ensemble="NVT",
        pressure_bar=None,
        boundary_conditions="periodic-3d",
    )


def _table(
    *,
    row_labels=("full-field", "unbiased"),
    boundary_conditions="periodic-3d",
    full_bias=2.0,
) -> ReducedPotentialTable:
    sample_count = 200
    return ReducedPotentialTable.create(
        u_kn=np.vstack(
            [
                np.full(sample_count, full_bias),
                np.zeros(sample_count),
            ]
        ),
        N_k=(100, 100),
        row_labels=row_labels,
        frame_ids=tuple(f"frame-{index}" for index in range(sample_count)),
        beta=1.0 / (kB * 298.15),
        measure_id="product-soft-packing-v3",
        boundary_conditions=boundary_conditions,
    )


def _analysis_input(**overrides) -> PackingAnalysisInput:
    values = {
        "table": _table(),
        "schedule": _schedule(),
        "log_empty_weights": np.full(200, -2.0),
    }
    values.update(overrides)
    return PackingAnalysisInput.create(**values)


def test_packing_analysis_input_binds_rows_to_explicit_bias_scales():
    analysis_input = _analysis_input()

    assert analysis_input.target_state_index == 1
    assert analysis_input.full_field_state_index == 0
    assert analysis_input.content_hash
    assert analysis_input.log_empty_weights.flags.writeable is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"table": _table(row_labels=("unbiased", "full-field"))},
            "row labels",
        ),
        (
            {"table": _table(boundary_conditions="nonperiodic")},
            "boundary",
        ),
        (
            {"table": _table(full_bias=1.5)},
            "bias relation",
        ),
    ],
)
def test_packing_analysis_input_rejects_contract_drift(overrides, message):
    with pytest.raises(ValueError, match=message):
        _analysis_input(**overrides)


def test_soft_packing_estimator_uses_explicit_target_index_and_closes_cycle():
    rt_kcal = 0.001987204258640831 * 298.15
    result = estimate_soft_packing(
        _analysis_input(),
        overlap_min=0.03,
        effective_samples_min=100.0,
        bar_disagreement_kcal_max=0.2,
        packing_se_kcal_max=0.2,
        estimator_agreement_kcal_max=1.0e-8,
        estimator_agreement_z_max=3.0,
    )

    assert result["status"] == "passed"
    assert result["failure_codes"] == []
    assert result["target_state_index"] == 1
    assert result["full_field_state_index"] == 0
    assert result["delta_f_target_to_full"] == pytest.approx(2.0)
    assert result["p0_from_free_energy"] == pytest.approx(np.exp(-2.0))
    assert result["p0_from_reweighted_expectation"] == pytest.approx(
        np.exp(-2.0)
    )
    assert result["packing_free_energy_kcal_mol"] == pytest.approx(
        2.0 * rt_kcal
    )
    assert result[
        "estimator_disagreement_kcal_mol"
    ] == pytest.approx(0.0, abs=1.0e-12)
    assert result["estimator_agreement_bootstrap"]["replicates"] == 64
    assert np.isfinite(
        result["paired_estimator_covariance_kcal2_mol2"]
    )
    assert np.isfinite(
        result["estimator_disagreement_standard_error_kcal_mol"]
    )
    with pytest.raises(ValueError, match="estimate_soft_packing"):
        replace(result, content_hash="0" * 64)
    assert np.isfinite(
        result["p0_estimator_difference_standard_error"]
    )
    assert result["packing_result_sha256"]


def test_packing_result_and_reference_occupancy_share_one_hash_bound_dataset():
    analysis_input = _analysis_input()
    packing_result = estimate_soft_packing(
        analysis_input,
        overlap_min=0.03,
        effective_samples_min=100.0,
        bar_disagreement_kcal_max=0.2,
        packing_se_kcal_max=0.2,
        estimator_agreement_kcal_max=1.0e-8,
        estimator_agreement_z_max=3.0,
    )
    p0 = np.exp(-2.0)
    raw_weights = np.tile(
        [p0, 0.4, 1.0 - p0 - 0.4],
        (200, 1),
    )
    occupancy_input = SoftOccupancyAnalysisInput.create(
        table=analysis_input.table,
        contract=SoftOccupancyContract(
            ensemble_role="reference-product",
            target_state_index=1,
            target_state_label="unbiased",
            sampling_measure_id="product-soft-packing-v3",
            membership_definition_hash="1" * 64,
            observation_volume_hash="2" * 64,
            boundary_adapter_hash="7" * 64,
            solute_measure_hash="3" * 64,
            water_hamiltonian_hash="4" * 64,
            system_hamiltonian_hash="5" * 64,
            sample_independence_hash="6" * 64,
            temperature_k=298.15,
            boundary_conditions="periodic-3d",
            active_occupancy_max=1,
            reconstruction_bin_index=1,
        ),
        occupancy_weights=raw_weights,
    )
    reference_estimate = estimate_soft_occupancy(occupancy_input)
    reference = reference_estimate.distribution
    bridge = PackingOccupancyBridge.create(
        packing_input=analysis_input,
        packing_result=packing_result,
        reference_input=occupancy_input,
        reference_estimate=reference_estimate,
        agreement_abs_tolerance=1.0e-12,
        agreement_z_max=3.0,
    )

    assert bridge.status == "passed"
    assert bridge.reference_distribution_hash == reference.content_hash
    assert bridge.reduced_potential_table_hash == analysis_input.table.state_hash
    with pytest.raises(ValueError, match="hash-bound packing result"):
        PackingOccupancyBridge.create(
            packing_input=analysis_input,
            packing_result=dict(packing_result),
            reference_input=occupancy_input,
            reference_estimate=reference_estimate,
            agreement_abs_tolerance=1.0e-12,
            agreement_z_max=3.0,
        )

    mutated = SoftOccupancyDistribution.create(
        ensemble_role="reference-product",
        probabilities=np.asarray(
            [
                reference.probabilities[0] + 0.1,
                reference.probabilities[1] - 0.1,
            ]
        ),
        tail_probability=reference.tail_probability,
        covariance_of_mean=reference.covariance_of_mean,
        membership_definition_hash=reference.membership_definition_hash,
        observation_volume_hash=reference.observation_volume_hash,
        boundary_adapter_hash=reference.boundary_adapter_hash,
        solute_measure_hash=reference.solute_measure_hash,
        water_hamiltonian_hash=reference.water_hamiltonian_hash,
        system_hamiltonian_hash=reference.system_hamiltonian_hash,
        temperature_k=reference.temperature_k,
        boundary_conditions=reference.boundary_conditions,
        source_artifact_hash=reference.source_artifact_hash,
        estimator="deliberate arbitrary p0 mutation",
    )
    assert mutated.content_hash != reference.content_hash
    with pytest.raises(ValueError, match="validated factory"):
        PackingOccupancyBridge(
            role="shared-packing-occupancy-analysis",
            reference_distribution_hash=mutated.content_hash,
            reference_estimate_hash=reference_estimate.content_hash,
            reference_analysis_input_hash=occupancy_input.content_hash,
            packing_analysis_input_hash=analysis_input.content_hash,
            packing_result_hash=packing_result.content_hash,
            reduced_potential_table_hash=analysis_input.table.state_hash,
            periodic_boundary_adapter_hash=mutated.boundary_adapter_hash,
            p0_reference=float(mutated.probabilities[0]),
            p0_from_free_energy=float(mutated.probabilities[0]),
            p0_from_expectation=float(mutated.probabilities[0]),
            agreement_abs_tolerance=1.0e-12,
            agreement_z_max=3.0,
            maximum_standardized_residual=0.0,
            status="passed",
            content_hash="1" * 64,
        )
