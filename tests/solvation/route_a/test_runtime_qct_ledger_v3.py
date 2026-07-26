from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from ase.units import kB

from maple.function.dispatcher.solvfe.analysis import ReducedPotentialTable
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
from maple.function.dispatcher.solvfe.protocol import canonical_sha256
from maple.function.dispatcher.solvfe.qct_ledger import (
    BoundaryMeasureBridge,
    ConditionalCouplingProfile,
    ConditionedQCTProfile,
    PackingOccupancyBridge,
    QCTBootstrapEvidence,
    QCTJointCovariance,
    build_cluster_qct_profile,
    build_periodic_qct_profile,
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
    return SoftEffectiveVolumeEstimate.create(
        tail_upper_bound_angstrom3=0.0,
        replicate_estimates_angstrom3=(value, value),
        integration_box_angstrom=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash=boundary_adapter_hash,
        solute_geometry_hash=f"{index + 1:x}" * 64,
        sobol_power=4,
        seed=index,
        tail_log_tolerance=20.0,
    )


def _finite_system(
    *,
    reverse_water_labels: bool = False,
    solute_measure_hash: str = "3" * 64,
):
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
        solute_measure_hash=solute_measure_hash,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
    )


def test_finite_oracle_derives_and_binds_periodic_boundary_identity():
    finite = _finite_system()
    expected = canonical_sha256(
        {
            "contract_id": "finite-soft-qct-periodic-boundary-adapter-v3",
            "membership_definition_hash": "1" * 64,
            "observation_volume_hash": "2" * 64,
            "solute_measure_hash": "3" * 64,
            "boundary_conditions": "periodic-3d",
            "coordinate_adapter": (
                "identity over caller-enumerated finite configurations"
            ),
            "occupancy_input": (
                "precomputed framewise soft weights under this declared "
                "periodic adapter"
            ),
        }
    )

    assert finite.reference_distribution.boundary_adapter_hash == expected
    assert finite.coupled_distribution.boundary_adapter_hash == expected
    assert (
        finite.conditioned_profile.periodic_boundary_adapter_hash
        == expected
    )
    assert (
        finite.conditional_coupling_profile.periodic_boundary_adapter_hash
        == expected
    )
    changed = _finite_system(solute_measure_hash="9" * 64)
    assert changed.reference_distribution.boundary_adapter_hash != expected
    assert changed.source_artifact_hash != finite.source_artifact_hash


def _bootstrap_columns(finite, *, replicates: int = 3):
    count = len(finite.conditioned_profile.occupancies)
    labels = tuple(
        [f"p[{n}]" for n in range(count)]
        + [f"p[n>={count}]"]
        + [f"x[{n}]" for n in range(count)]
        + [f"x[n>={count}]"]
        + [f"A[{n}]_kcal_mol" for n in range(count)]
        + [f"g[{n}]_kcal_mol" for n in range(count)]
    )
    return {
        label: np.full(replicates, float(index))
        for index, label in enumerate(labels)
    }


def _shared_packing_reference_bridge():
    sample_count = 200
    table = ReducedPotentialTable.create(
        u_kn=np.vstack(
            (
                np.full(sample_count, 2.0),
                np.zeros(sample_count),
            )
        ),
        N_k=(100, 100),
        row_labels=("full-field", "unbiased"),
        frame_ids=tuple(f"shared-frame-{index}" for index in range(sample_count)),
        beta=1.0 / (kB * TEMPERATURE_K),
        measure_id="shared-packing-bootstrap-v3",
        boundary_conditions="periodic-3d",
    )
    schedule = PackingSchedule(
        states=(
            PackingBiasState("full-field", 1.0),
            PackingBiasState("unbiased", 0.0),
        ),
        target_state_index=1,
        full_field_state_index=0,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash="7" * 64,
        conditioning_measure_id="shared-packing-bootstrap-v3",
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        oxygen_atom_map_hash="5" * 64,
        solute_atom_map_hash="6" * 64,
        cell_hash="7" * 64,
        active_occupancy_max=1,
        temperature_k=TEMPERATURE_K,
        ensemble="NVT",
        pressure_bar=None,
        boundary_conditions="periodic-3d",
    )
    packing_input = PackingAnalysisInput.create(
        table=table,
        schedule=schedule,
        log_empty_weights=np.full(sample_count, -2.0),
    )
    packing_result = estimate_soft_packing(
        packing_input,
        overlap_min=0.03,
        effective_samples_min=100.0,
        bar_disagreement_kcal_max=0.2,
        packing_se_kcal_max=0.2,
        estimator_agreement_kcal_max=1.0e-8,
        estimator_agreement_z_max=3.0,
    )
    p0 = math.exp(-2.0)
    occupancy_input = SoftOccupancyAnalysisInput.create(
        table=table,
        contract=SoftOccupancyContract(
            ensemble_role="reference-product",
            target_state_index=1,
            target_state_label="unbiased",
            sampling_measure_id="shared-packing-bootstrap-v3",
            membership_definition_hash="1" * 64,
            observation_volume_hash="2" * 64,
            boundary_adapter_hash="7" * 64,
            solute_measure_hash="3" * 64,
            water_hamiltonian_hash="4" * 64,
            system_hamiltonian_hash="5" * 64,
            sample_independence_hash="6" * 64,
            temperature_k=TEMPERATURE_K,
            boundary_conditions="periodic-3d",
            active_occupancy_max=1,
            reconstruction_bin_index=1,
        ),
        occupancy_weights=np.tile(
            [p0, 0.4, 1.0 - p0 - 0.4],
            (sample_count, 1),
        ),
    )
    reference_estimate = estimate_soft_occupancy(occupancy_input)
    bridge = PackingOccupancyBridge.create(
        packing_input=packing_input,
        packing_result=packing_result,
        reference_input=occupancy_input,
        reference_estimate=reference_estimate,
        agreement_abs_tolerance=1.0e-12,
        agreement_z_max=3.0,
    )
    return reference_estimate.distribution, bridge


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
        evidence_hash=finite.source_artifact_hash,
    )
    with pytest.raises(ValueError, match="validated factory"):
        QCTJointCovariance(
            matrix=joint.matrix,
            variable_labels=joint.variable_labels,
            reference_distribution_hash=(
                joint.reference_distribution_hash
            ),
            coupled_distribution_hash=joint.coupled_distribution_hash,
            conditioned_profile_hash=joint.conditioned_profile_hash,
            conditional_coupling_profile_hash=(
                joint.conditional_coupling_profile_hash
            ),
            relation=joint.relation,
            evidence_hash=joint.evidence_hash,
            content_hash=joint.content_hash,
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


def test_periodic_bridge_uses_nonzero_reference_occupancies():
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
        boundary_adapter_hash=(
            finite.reference_distribution.boundary_adapter_hash
        ),
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="5" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="8" * 64,
        estimator="deliberate p(n>0) mutation",
    )
    mutated_profile = build_periodic_qct_profile(
        reference=mutated,
        conditional_coupling=finite.conditional_coupling_profile,
        covariance_of_mean=np.zeros((3, 3)),
        approximation_role="periodic-explicit-reference",
    )
    with pytest.raises(ValueError, match="internal finite-state oracle"):
        build_periodic_qct_profile(
            reference=mutated,
            conditional_coupling=finite.conditional_coupling_profile,
            covariance_of_mean=np.zeros((3, 3)),
            approximation_role="exact-enumerable-bridge",
        )

    assert mutated_profile.free_energies_kcal_mol[0] == pytest.approx(
        finite.conditioned_profile.free_energies_kcal_mol[0]
    )
    assert not np.allclose(
        mutated_profile.free_energies_kcal_mol[1:],
        finite.conditioned_profile.free_energies_kcal_mol[1:],
    )


def test_periodic_bridge_rejects_boundary_adapter_mismatch():
    finite = _finite_system()
    conditional = finite.conditional_coupling_profile
    with pytest.raises(ValueError, match="internal finite-system oracle"):
        ConditionalCouplingProfile.create(
            free_energies_kcal_mol=conditional.free_energies_kcal_mol,
            covariance_of_mean=conditional.covariance_of_mean,
            membership_definition_hash=(
                conditional.membership_definition_hash
            ),
            observation_volume_hash=conditional.observation_volume_hash,
            periodic_boundary_adapter_hash=(
                conditional.periodic_boundary_adapter_hash
            ),
            solute_measure_hash=conditional.solute_measure_hash,
            water_hamiltonian_hash=conditional.water_hamiltonian_hash,
            reference_hamiltonian_hash=(
                conditional.reference_hamiltonian_hash
            ),
            coupled_hamiltonian_hash=conditional.coupled_hamiltonian_hash,
            hamiltonian_bridge_hash=conditional.hamiltonian_bridge_hash,
            temperature_k=conditional.temperature_k,
            source_artifact_hash=conditional.source_artifact_hash,
            estimator="exact finite enumeration of Z_X,n/Z_0,n",
        )
    mismatched = ConditionalCouplingProfile.create(
        free_energies_kcal_mol=conditional.free_energies_kcal_mol,
        covariance_of_mean=conditional.covariance_of_mean,
        membership_definition_hash=conditional.membership_definition_hash,
        observation_volume_hash=conditional.observation_volume_hash,
        periodic_boundary_adapter_hash="b" * 64,
        solute_measure_hash=conditional.solute_measure_hash,
        water_hamiltonian_hash=conditional.water_hamiltonian_hash,
        reference_hamiltonian_hash=conditional.reference_hamiltonian_hash,
        coupled_hamiltonian_hash=conditional.coupled_hamiltonian_hash,
        hamiltonian_bridge_hash=conditional.hamiltonian_bridge_hash,
        temperature_k=conditional.temperature_k,
        source_artifact_hash="c" * 64,
        estimator="boundary-adapter mismatch fixture",
    )

    with pytest.raises(ValueError, match="reference/conditional hashes"):
        build_periodic_qct_profile(
            reference=finite.reference_distribution,
            conditional_coupling=mismatched,
            covariance_of_mean=np.zeros((3, 3)),
            approximation_role="periodic-explicit-reference",
        )


def test_labeled_cluster_builder_counts_density_volume_and_factorial_once():
    finite = _finite_system()
    rho = 0.03332953803622
    volumes = np.asarray([18.0, 13.0])
    boundary_bridge = BoundaryMeasureBridge.create(
        reference=finite.reference_distribution,
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
        "reference_distribution": finite.reference_distribution,
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
    with pytest.raises(ValueError, match="validated factory"):
        replace(
            volume_artifacts[0],
            volume_angstrom3=999.0,
        )
    bad_kwargs = dict(builder_kwargs)
    bad_kwargs["effective_volume_estimates"] = (
        _volume_artifact(
            volumes[0],
            index=0,
            boundary_adapter_hash="e" * 64,
        ),
        volume_artifacts[1],
    )
    with pytest.raises(ValueError, match="boundary bridge"):
        build_cluster_qct_profile(**bad_kwargs)
    bad_kwargs = dict(builder_kwargs)
    bad_kwargs["temperature_k"] = TEMPERATURE_K + 1.0
    with pytest.raises(ValueError, match="temperatures differ"):
        build_cluster_qct_profile(**bad_kwargs)
    bad_kwargs = dict(builder_kwargs)
    bad_kwargs["reference_hamiltonian_hash"] = "f" * 64
    with pytest.raises(ValueError, match="reference distribution"):
        build_cluster_qct_profile(**bad_kwargs)


def test_validated_qct_artifacts_cannot_be_replaced():
    finite = _finite_system()

    for artifact in (
        finite.reference_distribution,
        finite.conditional_coupling_profile,
        finite.conditioned_profile,
    ):
        with pytest.raises(ValueError, match="validated factory"):
            replace(artifact, content_hash="0" * 64)


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
        periodic_boundary_adapter_hash=(
            finite.reference_distribution.boundary_adapter_hash
        ),
        boundary_measure_bridge_hash="e" * 64,
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
        relation="independent-block-diagonal",
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

    reference = finite.reference_distribution
    coupled = SoftOccupancyDistribution.create(
        ensemble_role="coupled-solution",
        probabilities=finite.coupled_distribution.probabilities,
        tail_probability=0.0,
        covariance_of_mean=occupancy_covariance,
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash=(
            finite.reference_distribution.boundary_adapter_hash
        ),
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
        periodic_boundary_adapter_hash=(
            finite.reference_distribution.boundary_adapter_hash
        ),
        boundary_measure_bridge_hash="e" * 64,
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
    packing_bridge = PackingOccupancyBridge.exact_enumeration(reference)
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
            evidence_hash=finite.source_artifact_hash,
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


def test_joint_covariance_rejects_broadcastable_bootstrap_matrix():
    finite = _finite_system()

    with pytest.raises(ValueError, match="canonical ledger variable labels"):
        QCTBootstrapEvidence.create(
            replicate_columns={"p[0]": np.zeros(3)},
            reference=finite.reference_distribution,
            coupled=finite.coupled_distribution,
            profile=finite.conditioned_profile,
            conditional_coupling=finite.conditional_coupling_profile,
            provenance_graph_hash="7" * 64,
        )


def test_shared_bootstrap_binds_column_order_and_source_artifacts():
    finite = _finite_system()
    columns = _bootstrap_columns(finite)
    reversed_columns = dict(reversed(tuple(columns.items())))
    evidence = QCTBootstrapEvidence.create(
        replicate_columns=reversed_columns,
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        provenance_graph_hash="7" * 64,
    )
    count = len(finite.conditioned_profile.occupancies)
    matrix = np.zeros((4 * count + 2, 4 * count + 2))

    assert evidence.variable_labels == tuple(columns)
    assert evidence.replicate_values[0] == pytest.approx(
        np.arange(len(columns), dtype=float)
    )
    joint = QCTJointCovariance.create(
        reference=finite.reference_distribution,
        coupled=finite.coupled_distribution,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        matrix=matrix,
        relation="shared-provenance-graph-bootstrap",
        evidence_hash=evidence.content_hash,
        bootstrap_evidence=evidence,
    )
    assert joint.variable_labels == evidence.variable_labels

    unrelated_coupled = SoftOccupancyDistribution.create(
        ensemble_role="coupled-solution",
        probabilities=finite.coupled_distribution.probabilities,
        tail_probability=0.0,
        covariance_of_mean=np.zeros((4, 4)),
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash=(
            finite.reference_distribution.boundary_adapter_hash
        ),
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="6" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="b" * 64,
        estimator="unrelated bootstrap source",
    )
    unrelated_evidence = QCTBootstrapEvidence.create(
        replicate_columns=columns,
        reference=finite.reference_distribution,
        coupled=unrelated_coupled,
        profile=finite.conditioned_profile,
        conditional_coupling=finite.conditional_coupling_profile,
        provenance_graph_hash="7" * 64,
    )
    with pytest.raises(ValueError, match="bootstrap evidence"):
        QCTJointCovariance.create(
            reference=finite.reference_distribution,
            coupled=finite.coupled_distribution,
            profile=finite.conditioned_profile,
            conditional_coupling=finite.conditional_coupling_profile,
            matrix=matrix,
            relation="shared-provenance-graph-bootstrap",
            evidence_hash=unrelated_evidence.content_hash,
            bootstrap_evidence=unrelated_evidence,
        )


def test_shared_bootstrap_propagates_nonzero_cross_block_covariance():
    reference, packing_bridge = _shared_packing_reference_bridge()
    z = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    p = np.append(
        reference.probabilities,
        reference.tail_probability,
    )
    x = np.asarray([0.35, 0.55, 0.10])
    rt = R_KCAL * TEMPERATURE_K
    target_mu = -1.0
    conditioned = target_mu - rt * np.log(
        x[:2] / reference.probabilities[0]
    )
    conditional = conditioned + rt * np.log(
        reference.probabilities / reference.probabilities[0]
    )
    columns = {
        "p[0]": np.full(len(z), p[0]),
        "p[1]": np.full(len(z), p[1]),
        "p[n>=2]": np.full(len(z), p[2]),
        "x[0]": x[0] + 0.006 * z,
        "x[1]": x[1] - 0.004 * z,
        "x[n>=2]": x[2] - 0.002 * z,
        "A[0]_kcal_mol": conditioned[0] + 0.030 * z,
        "A[1]_kcal_mol": conditioned[1] - 0.020 * z,
        "g[0]_kcal_mol": conditional[0] + 0.010 * z,
        "g[1]_kcal_mol": conditional[1] + 0.015 * z,
    }
    values = np.column_stack(tuple(columns.values()))
    covariance = np.cov(values, rowvar=False, ddof=1)
    coupled = SoftOccupancyDistribution.create(
        ensemble_role="coupled-solution",
        probabilities=x[:2],
        tail_probability=x[2],
        covariance_of_mean=covariance[3:6, 3:6],
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        boundary_adapter_hash="7" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        system_hamiltonian_hash="6" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="periodic-3d",
        source_artifact_hash="8" * 64,
        estimator="correlated-bootstrap coupled occupancy",
    )
    profile = ConditionedQCTProfile.create(
        free_energies_kcal_mol=conditioned,
        covariance_of_mean=covariance[6:8, 6:8],
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        periodic_boundary_adapter_hash="7" * 64,
        boundary_measure_bridge_hash="9" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
        hamiltonian_bridge_hash="a" * 64,
        temperature_k=TEMPERATURE_K,
        boundary_conditions="nonperiodic-cluster",
        source_artifact_hash="b" * 64,
        estimator="correlated-bootstrap conditioned rows",
        approximation_role="maple-cluster-continuum-v3",
    )
    conditional_coupling = ConditionalCouplingProfile.create(
        free_energies_kcal_mol=conditional,
        covariance_of_mean=covariance[8:10, 8:10],
        membership_definition_hash="1" * 64,
        observation_volume_hash="2" * 64,
        periodic_boundary_adapter_hash="7" * 64,
        solute_measure_hash="3" * 64,
        water_hamiltonian_hash="4" * 64,
        reference_hamiltonian_hash="5" * 64,
        coupled_hamiltonian_hash="6" * 64,
        hamiltonian_bridge_hash="a" * 64,
        temperature_k=TEMPERATURE_K,
        source_artifact_hash="c" * 64,
        estimator="correlated-bootstrap conditional coupling",
    )
    evidence = QCTBootstrapEvidence.create(
        replicate_columns=columns,
        reference=reference,
        coupled=coupled,
        profile=profile,
        conditional_coupling=conditional_coupling,
        provenance_graph_hash="d" * 64,
    )
    shared = QCTJointCovariance.create(
        reference=reference,
        coupled=coupled,
        profile=profile,
        conditional_coupling=conditional_coupling,
        matrix=evidence.covariance,
        relation="shared-provenance-graph-bootstrap",
        evidence_hash=evidence.content_hash,
        bootstrap_evidence=evidence,
    )
    independent = QCTJointCovariance.block_diagonal(
        reference=reference,
        coupled=coupled,
        profile=profile,
        conditional_coupling=conditional_coupling,
        relation="independent-block-diagonal",
        evidence_hash="e" * 64,
    )
    evaluation_kwargs = {
        "reference": reference,
        "coupled": coupled,
        "profile": profile,
        "conditional_coupling": conditional_coupling,
        "packing_bridge": packing_bridge,
        "closure_abs_tolerance_kcal_mol": 1.0e-10,
        "closure_z_max": 3.0,
        "bridge_abs_tolerance_kcal_mol": 1.0e-10,
        "bridge_z_max": 3.0,
        "occupancy_abs_tolerance": 1.0e-12,
        "tail_probability_max": 0.5,
    }
    shared_result = evaluate_soft_qct_ledger(
        joint_covariance=shared,
        **evaluation_kwargs,
    )
    independent_result = evaluate_soft_qct_ledger(
        joint_covariance=independent,
        **evaluation_kwargs,
    )

    assert shared.matrix[3, 6] != pytest.approx(0.0)
    assert shared_result["status"] == "passed"
    assert independent_result["status"] == "passed"
    assert shared_result[
        "direct_n0_standard_error_kcal_mol"
    ] != pytest.approx(
        independent_result["direct_n0_standard_error_kcal_mol"]
    )
    assert shared_result[
        "multi_n_standard_error_kcal_mol"
    ] != pytest.approx(
        independent_result["multi_n_standard_error_kcal_mol"]
    )
