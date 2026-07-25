from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from maple.function.dispatcher.solvfe.association_conditioning import (
    SoftEffectiveVolumeEstimate,
)
from maple.function.dispatcher.solvfe.finite_qct import (
    enumerate_finite_soft_qct,
    enumerate_labeled_cluster_partition,
)
from maple.function.dispatcher.solvfe.membership import (
    soft_occupancy_weights,
)
from maple.function.dispatcher.solvfe.occupancy_analysis import (
    SoftOccupancyDistribution,
)
from maple.function.dispatcher.solvfe.qct_ledger import (
    BoundaryMeasureBridge,
    ConditionedQCTProfile,
    PackingOccupancyBridge,
    QCTJointCovariance,
    build_cluster_qct_profile,
    evaluate_soft_qct_ledger,
)


TEMPERATURE_K = 298.15
R_KCAL = 0.001987204258640831


def _volume_artifact(
    value: float,
    *,
    index: int,
    boundary_adapter_hash: str,
) -> SoftEffectiveVolumeEstimate:
    return SoftEffectiveVolumeEstimate(
        volume_angstrom3=value,
        replicate_standard_error_angstrom3=0.0,
        tail_upper_bound_angstrom3=0.0,
        replicate_estimates_angstrom3=(value, value),
        samples_per_replicate=16,
        integration_box_angstrom=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash=boundary_adapter_hash,
        solute_geometry_hash=f"{index + 1:x}" * 64,
        content_hash=f"{index + 3:x}" * 64,
    )


def _synthetic_packing_bridge(
    reference: SoftOccupancyDistribution,
) -> PackingOccupancyBridge:
    p0 = float(reference.probabilities[0])
    return PackingOccupancyBridge(
        role="shared-packing-occupancy-analysis",
        reference_distribution_hash=reference.content_hash,
        reference_analysis_input_hash="e" * 64,
        packing_analysis_input_hash="f" * 64,
        reduced_potential_table_hash="0" * 64,
        p0_reference=p0,
        p0_from_free_energy=p0,
        p0_from_expectation=p0,
        maximum_standardized_residual=0.0,
        status="passed",
        content_hash="1" * 64,
    )


def _finite_system(*, reverse_water_labels: bool = False):
    memberships = np.asarray(
        [
            [0.25, 0.10],
            [0.40, 0.25],
            [0.75, 0.55],
            [0.20, 0.45],
        ]
    )
    if reverse_water_labels:
        memberships = memberships[:, ::-1]
    return enumerate_finite_soft_qct(
        reference_energies_kcal_mol=np.asarray([0.0, 0.4, 1.1, 0.7]),
        coupling_energies_kcal_mol=np.asarray([-1.0, 0.2, -0.4, 0.8]),
        soft_occupancy_weights=soft_occupancy_weights(memberships),
        temperature_k=TEMPERATURE_K,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
    )


def test_finite_soft_qct_is_invariant_to_identical_water_label_permutation():
    original = _finite_system()
    permuted = _finite_system(reverse_water_labels=True)

    assert permuted.reference_distribution.probabilities == pytest.approx(
        original.reference_distribution.probabilities
    )
    assert permuted.coupled_distribution.probabilities == pytest.approx(
        original.coupled_distribution.probabilities
    )
    assert (
        permuted.conditioned_profile.free_energies_kcal_mol
        == pytest.approx(
            original.conditioned_profile.free_energies_kcal_mol
        )
    )
    assert permuted.exact_excess_free_energy_kcal_mol == pytest.approx(
        original.exact_excess_free_energy_kcal_mol
    )


def test_finite_soft_qct_closes_direct_fixed_n_and_multi_n_paths():
    finite = _finite_system()
    joint = QCTJointCovariance.block_diagonal(
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        relation="exact-enumeration",
        evidence_hash="7" * 64,
    )
    result = evaluate_soft_qct_ledger(
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        packing_bridge=PackingOccupancyBridge.exact_enumeration(
            finite.reference_distribution
        ),
        joint_covariance=joint,
        closure_abs_tolerance_kcal_mol=1.0e-10,
        closure_z_max=3.0,
        bridge_abs_tolerance_kcal_mol=1.0e-10,
        bridge_z_max=3.0,
        occupancy_abs_tolerance=1.0e-12,
        tail_probability_max=0.005,
    )

    assert result["status"] == "passed"
    assert result["failure_codes"] == []
    assert result["multi_n_free_energy_kcal_mol"] == pytest.approx(
        finite.exact_excess_free_energy_kcal_mol,
        abs=1.0e-12,
    )
    assert result["direct_n0_free_energy_kcal_mol"] == pytest.approx(
        finite.exact_excess_free_energy_kcal_mol,
        abs=1.0e-12,
    )
    assert result["fixed_n_free_energies_kcal_mol"] == pytest.approx(
        np.full(3, finite.exact_excess_free_energy_kcal_mol),
        abs=1.0e-12,
    )
    assert result["reconstructed_coupled_probabilities"] == pytest.approx(
        finite.coupled_distribution.probabilities,
        abs=1.0e-12,
    )
    assert result["packing_free_energy_kcal_mol"] == pytest.approx(
        -R_KCAL
        * TEMPERATURE_K
        * math.log(finite.reference_distribution.probabilities[0])
    )


def test_periodic_bridge_rejects_nonzero_reference_occupancy_mutation():
    finite = _finite_system()
    original = finite.reference_distribution.probabilities
    mutated_probabilities = np.asarray(
        [
            original[0],
            original[1] + 0.08,
            original[2] - 0.08,
        ]
    )
    mutated = SoftOccupancyDistribution.create(
        ensemble_role="reference-product",
        probabilities=mutated_probabilities,
        tail_probability=0.0,
        covariance_of_mean=np.zeros((4, 4)),
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="5" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="8" * 64,
        estimator="deliberate p(n>0) mutation",
    )
    joint = QCTJointCovariance.block_diagonal(
        reference=mutated,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        relation="exact-enumeration",
        evidence_hash="9" * 64,
    )
    result = evaluate_soft_qct_ledger(
        reference=mutated,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        packing_bridge=_synthetic_packing_bridge(mutated),
        joint_covariance=joint,
        closure_abs_tolerance_kcal_mol=1.0e-10,
        closure_z_max=3.0,
        bridge_abs_tolerance_kcal_mol=1.0e-10,
        bridge_z_max=3.0,
        occupancy_abs_tolerance=1.0e-12,
        tail_probability_max=0.005,
    )

    assert result["status"] == "failed"
    assert "QCT_PERIODIC_BRIDGE_FAILED" in result["failure_codes"]


def test_labeled_cluster_builder_counts_density_volume_and_factorial_once():
    finite = _finite_system()
    rho = 0.03332953803622
    volumes = np.asarray([18.0, 13.0])
    boundary_bridge = BoundaryMeasureBridge.create(
        membership_surface_hash="2" * 64,
        periodic_adapter_hash="c" * 64,
        nonperiodic_adapter_hash="d" * 64,
    )
    volume_artifacts = tuple(
        _volume_artifact(
            value,
            index=index,
            boundary_adapter_hash=boundary_bridge.nonperiodic_adapter_hash,
        )
        for index, value in enumerate(volumes)
    )
    rt = R_KCAL * TEMPERATURE_K
    coupling_factors = np.asarray([1.7, 0.65])
    release_factors = np.asarray([0.85, 1.15])
    outer_water_factor = 0.9
    outer_cluster_factors = np.asarray([1.1, 0.75, 1.3])
    oracle = enumerate_labeled_cluster_partition(
        edge_coupling_boltzmann_factors=coupling_factors,
        edge_effective_volumes_angstrom3=volumes,
        edge_release_boltzmann_factors=release_factors,
        outer_cluster_boltzmann_factors=outer_cluster_factors,
        outer_water_boltzmann_factor=outer_water_factor,
        water_number_density_per_angstrom3=rho,
        temperature_k=TEMPERATURE_K,
    )
    alchemical_edges = -rt * np.log(coupling_factors)
    release_edges = -rt * np.log(release_factors)
    outer_cluster = -rt * np.log(outer_cluster_factors)
    outer_water = -rt * math.log(outer_water_factor)
    builder_kwargs = {
        "edge_alchemical_free_energies_kcal_mol": alchemical_edges,
        "effective_volume_estimates": volume_artifacts,
        "edge_release_free_energies_kcal_mol": release_edges,
        "outer_cluster_free_energies_kcal_mol": outer_cluster,
        "outer_water_free_energy_kcal_mol": outer_water,
        "water_number_density_per_angstrom3": rho,
        "covariance_of_mean": np.zeros((3, 3)),
        "temperature_k": TEMPERATURE_K,
        "membership_definition_hash": "1" * 64,
        "observation_volume_hash": "2" * 64,
        "solute_measure_hash": "3" * 64,
        "water_hamiltonian_hash": "4" * 64,
        "reference_hamiltonian_hash": "5" * 64,
        "coupled_hamiltonian_hash": "6" * 64,
        "outer_contract_hash": "8" * 64,
        "boundary_measure_bridge": boundary_bridge,
        "hamiltonian_bridge_hash": (
            finite.conditioned_profile.hamiltonian_bridge_hash
        ),
        "joint_provenance_graph_hash": "9" * 64,
        "covariance_method": "exact-enumeration",
        "approximation_role": "maple-cluster-continuum-v3",
    }
    build = build_cluster_qct_profile(**builder_kwargs)

    assert build.profile.free_energies_kcal_mol == pytest.approx(
        oracle.free_energies_kcal_mol
    )
    assert build.rows[0]["occupancy"] == 0
    assert build.rows[0]["volume_density_kcal_mol"] == 0.0
    assert build.rows[0]["symmetry_kcal_mol"] == 0.0
    assert build.rows[2]["volume_density_kcal_mol"] == pytest.approx(
        -rt * math.log(rho * volumes[0])
        - rt * math.log(rho * volumes[1])
    )
    assert build.rows[2]["symmetry_kcal_mol"] == pytest.approx(
        rt * math.log(2.0)
    )
    assert build.rows[2]["outer_cycle_kcal_mol"] == pytest.approx(
        outer_cluster[2] - 2.0 * outer_water
    )
    bad_kwargs = dict(builder_kwargs)
    bad_kwargs["effective_volume_estimates"] = (
        replace(
            volume_artifacts[0],
            boundary_adapter_hash="e" * 64,
        ),
        volume_artifacts[1],
    )
    with pytest.raises(ValueError, match="boundary bridge"):
        build_cluster_qct_profile(**bad_kwargs)

@pytest.mark.parametrize("duplicate", ["density", "symmetry"])
def test_toy_falsifier_rejects_density_or_factorial_double_counting(duplicate):
    finite = _finite_system()
    values = np.array(
        finite.conditioned_profile.free_energies_kcal_mol,
        copy=True,
    )
    rt = R_KCAL * TEMPERATURE_K
    if duplicate == "density":
        rho = 0.03332953803622
        volumes = np.asarray([18.0, 13.0])
        values += np.concatenate(
            ([0.0], np.cumsum(-rt * np.log(rho * volumes)))
        )
    else:
        values += np.asarray(
            [rt * math.lgamma(n + 1.0) for n in range(3)]
        )
    perturbed = ConditionedQCTProfile.create(
        free_energies_kcal_mol=values,
        covariance_of_mean=np.zeros((3, 3)),
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
        hamiltonian_bridge_hash=(
            finite.conditioned_profile.hamiltonian_bridge_hash
        ),
        temperature_k=TEMPERATURE_K,
        boundary_conditions="nonperiodic-cluster",
        source_artifact_hash="8" * 64,
        estimator=f"deliberate-duplicate-{duplicate}",
        approximation_role="maple-cluster-continuum-v3",
    )
    joint = QCTJointCovariance.block_diagonal(
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=perturbed,
        conditional_coupling=finite.conditional_coupling_profile,
        relation="exact-enumeration",
        evidence_hash="9" * 64,
    )
    result = evaluate_soft_qct_ledger(
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=perturbed,
        conditional_coupling=finite.conditional_coupling_profile,
        packing_bridge=PackingOccupancyBridge.exact_enumeration(
            finite.reference_distribution
        ),
        joint_covariance=joint,
        closure_abs_tolerance_kcal_mol=1.0e-10,
        closure_z_max=3.0,
        bridge_abs_tolerance_kcal_mol=1.0e-10,
        bridge_z_max=3.0,
        occupancy_abs_tolerance=1.0e-12,
        tail_probability_max=0.005,
    )

    assert result["status"] == "failed"
    assert "QCT_FIXED_N_CLOSURE_FAILED" in result["failure_codes"]
    assert "QCT_OCCUPANCY_RECONSTRUCTION_FAILED" in result["failure_codes"]


def test_joint_covariance_propagates_simplex_and_profile_uncertainty():
    finite = _finite_system()
    count = 3
    probability_count = count + 1
    projector = (
        np.eye(probability_count)
        - np.ones((probability_count, probability_count))
        / probability_count
    )
    occupancy_covariance = 2.0e-6 * projector
    profile_covariance = np.diag([1.0e-4, 2.0e-4, 3.0e-4])

    reference = SoftOccupancyDistribution.create(
        ensemble_role="reference-product",
        probabilities=finite.reference_distribution.probabilities,
        tail_probability=0.0,
        covariance_of_mean=occupancy_covariance,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="5" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="a" * 64,
        estimator="synthetic covariance test",
    )
    coupled = SoftOccupancyDistribution.create(
        ensemble_role="coupled-solution",
        probabilities=finite.coupled_distribution.probabilities,
        tail_probability=0.0,
        covariance_of_mean=occupancy_covariance,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="6" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="b" * 64,
        estimator="synthetic covariance test",
    )
    profile = ConditionedQCTProfile.create(
        free_energies_kcal_mol=(
            finite.conditioned_profile.free_energies_kcal_mol
        ),
        covariance_of_mean=profile_covariance,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
        hamiltonian_bridge_hash=(
            finite.conditioned_profile.hamiltonian_bridge_hash
        ),
        temperature_k=TEMPERATURE_K,
        boundary_conditions="nonperiodic-cluster",
        source_artifact_hash="c" * 64,
        estimator="synthetic covariance test",
        approximation_role="maple-cluster-continuum-v3",
    )
    joint = QCTJointCovariance.block_diagonal(
        reference=reference,
        coupled=coupled,
        profile=profile,
        conditional_coupling=finite.conditional_coupling_profile,
        relation="independent-block-diagonal",
        evidence_hash="d" * 64,
    )
    packing_bridge = _synthetic_packing_bridge(reference)
    result = evaluate_soft_qct_ledger(
        reference=reference,
        coupled=coupled,
        profile=profile,
        conditional_coupling=finite.conditional_coupling_profile,
        packing_bridge=packing_bridge,
        joint_covariance=joint,
        closure_abs_tolerance_kcal_mol=1.0e-10,
        closure_z_max=3.0,
        bridge_abs_tolerance_kcal_mol=1.0e-10,
        bridge_z_max=3.0,
        occupancy_abs_tolerance=1.0e-12,
        tail_probability_max=0.005,
    )

    assert result["status"] == "passed"
    assert result["multi_n_standard_error_kcal_mol"] > 0.0
    assert result["direct_n0_standard_error_kcal_mol"] > 0.0
    assert np.all(result["fixed_n_standard_errors_kcal_mol"] > 0.0)
    assert np.all(
        result["fixed_n_minus_multi_n_standard_errors_kcal_mol"] > 0.0
    )


def test_joint_covariance_rejects_missing_simplex_cross_nullspace():
    finite = _finite_system()
    matrix = np.zeros((14, 14))
    matrix[0, 4] = matrix[4, 0] = 1.0e-4

    with pytest.raises(ValueError, match="normalization nullspace"):
        QCTJointCovariance.create(
            reference=finite.reference_distribution,
            coupled=finite.coupled_distribution,
            profile=finite.conditioned_profile,
            conditional_coupling=finite.conditional_coupling_profile,
            matrix=matrix,
            relation="exact-enumeration",
            evidence_hash="7" * 64,
        )


def test_joint_covariance_rejects_naked_shared_bootstrap_claim():
    finite = _finite_system()
    count = len(finite.conditioned_profile.occupancies)
    matrix = np.zeros((4 * count + 2, 4 * count + 2))

    with pytest.raises(ValueError, match="bootstrap evidence"):
        QCTJointCovariance.create(
            reference=finite.reference_distribution,
            coupled=finite.coupled_distribution,
            profile=finite.conditioned_profile,
            conditional_coupling=finite.conditional_coupling_profile,
            matrix=matrix,
            relation="shared-provenance-graph-bootstrap",
            evidence_hash="7" * 64,
        )
    with pytest.raises(ValueError, match="cannot claim shared"):
        QCTJointCovariance.block_diagonal(
            reference=finite.reference_distribution,
            coupled=finite.coupled_distribution,
            profile=finite.conditioned_profile,
            conditional_coupling=finite.conditional_coupling_profile,
            relation="shared-provenance-graph-bootstrap",
            evidence_hash="7" * 64,
        )
