from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import aggregate_mnsol_response_partition as aggregation  # pyright: ignore[reportMissingImports]  # noqa: E402

ALLOWED_METHODS = aggregation.ALLOWED_METHODS
FULL_METHODS = aggregation.FULL_METHODS
SELECTION_FINGERPRINT = "e" * 64
SELECTION_ARTIFACT_SHA256 = "c" * 64
PROTOCOL_FINGERPRINT = "d" * 64
DEFAULT_DATASET_FINGERPRINT = "f" * 64
DEFAULT_SOURCE_TABLE = "1" * 64
DEFAULT_SOURCE_BUNDLE = "2" * 64
CURRENT_HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()


def test_replay_v2_artifact_contract_is_versioned():
    assert aggregation.runner.ARTIFACT_NAME == (
        "route2-mnsol-macepolar-response-ablation-v4"
    )
    assert aggregation.runner.SCHEMA_VERSION == 4
    assert aggregation.runner.SCF_CONVERGENCE_CONTRACT_VERSION == (
        "route2-scf-convergence-evidence-v3"
    )
    assert aggregation.AGGREGATOR_ARTIFACT_NAME == (
        "route2-mnsol-macepolar-two-member-matrix-v4"
    )
    assert aggregation.SCHEMA_VERSION == 4
    assert aggregation.MACE_SCF_FINITE_RESOLUTION_REASON == (
        "finite-resolution-stagnation-v2"
    )
    assert aggregation.MACE_SCF_MAX_FINITE_FLOAT64_ULP_DISTANCE == (
        0xFFDFFFFFFFFFFFFE
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_actual_residual_objective_formula"
        ]
        == "max(monopole/tau_monopole,dipole/tau_dipole)"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_growth_rejection_inequality"
        ]
        == "Phi_trial > growth_limit * Phi_anchor"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT["scf_accepted_residual_source"]
        == "evaluated-actual-unmixed-physical-residual"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT["scf_rejected_growth_action"]
        == "reject-trial-rollback-prior-accepted-anchor-one-picard-restart"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT["scf_rejected_attempt_status"]
        == "rejected-anderson-actual-residual-growth"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_rejected_attempts_count_toward_max_iterations"
        ]
        is True
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_rejected_attempts_excluded_from_anderson_samples"
        ]
        is True
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_rejected_attempts_excluded_from_best_state_selection"
        ]
        is True
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_rejected_attempts_excluded_from_finite_resolution_window"
        ]
        is True
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_finite_resolution_history_source"
        ]
        == "accepted-solver-states-only"
    )
    assert (
        aggregation.runner.SCF_SOLVER_CONTRACT[
            "scf_finite_resolution_policy_version"
        ]
        == "finite-resolution-stagnation-v2"
    )


def _scf_convergence(
    *,
    online_candidate_iteration: int,
    solvent: str,
    scale: float = 1.0,
    reason: str = "finite-resolution-stagnation-v2",
):
    finite_resolution = reason == "finite-resolution-stagnation-v2"
    convergence = {
        "reason": reason,
        "online_candidate_iteration": online_candidate_iteration,
        "final_monopole_residual_e": (
            (4.0e-11 if finite_resolution else 5.0e-13) * scale
        ),
        "final_dipole_residual_e_angstrom": (
            (3.0e-11 if finite_resolution else 4.0e-13) * scale
        ),
    }
    if reason == "nominal-density-and-energy-v1":
        convergence["runtime_identity"] = None
        convergence["history_window"] = None
        convergence["fresh_map_replay"] = None
        return convergence

    convergence["runtime_identity"] = {
        "profile": aggregation.CONTINUUM_EQUATION_TO_PROFILE["ddpcm"],
        "solvent": solvent,
        "continuum_equation": "ddpcm",
        "mace_checkpoint_identifier": "polar-1-m",
        "mace_checkpoint_release_url": (
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_polar_1/MACE-POLAR-1-M.model"
        ),
        "mace_checkpoint_sha256": (
            "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
        ),
        "mace_checkpoint_size_bytes": 68_133_235,
        "mace_torch_version": "0.3.16",
        "graph_longrange_version": "0.4.0",
        "mace_long_range_evaluator_profile": (
            "graph-longrange-molecular-realspace-v1"
        ),
        "mace_dtype": "torch.float64",
        "device": "cpu",
        "torch_threads": 1,
        "torch_version": "2.12.0+cu130",
        "pyddx_version": "0.8.0",
        "pyddx_n_proc": 1,
        "pyddx_solver_tolerance": 1.0e-12,
        "continuum_dielectric": (
            aggregation.route2_solvent_spec(solvent).descriptors.dielectric
        ),
        "lmax": 15,
        "n_lebedev": 1202,
        "eta": 0.1,
        "atomic_numbers_sha256": "cc" * 32,
        "positions_angstrom_sha256": "dd" * 32,
        "cavity_radii_angstrom_sha256": "ee" * 32,
    }
    convergence["history_window"] = {
        "start_iteration": online_candidate_iteration - 6,
        "end_iteration": online_candidate_iteration,
        "root_monopole_span_e": 5.0e-13 * scale,
        "root_dipole_span_e_angstrom": 6.0e-13 * scale,
        "residual_monopole_span_e": 7.0e-13 * scale,
        "residual_dipole_span_e_angstrom": 8.0e-13 * scale,
        "potential_span_ev": 3.0e-11 * scale,
        "gradient_span_ev_per_angstrom": 4.0e-11 * scale,
        "maximum_monopole_residual_e": 4.0e-11 * scale,
        "maximum_dipole_residual_e_angstrom": 3.0e-11 * scale,
        "maximum_energy_delta_ev": 3.0e-11 * scale,
        "intrinsic_energy_span_ev": 2.0e-11 * scale,
    }
    convergence["fresh_map_replay"] = {
        "replay_count": 3,
        "evaluation_count": 4,
        "includes_online_candidate": True,
        "cold_replay_field_arrays_identical": True,
        "cold_replay_response_arrays_identical": True,
        "field_sha256_by_evaluation": ["cc" * 32] + ["aa" * 32] * 3,
        "response_sha256_by_evaluation": ["dd" * 32] + ["bb" * 32] * 3,
        "maximum_online_to_replay_potential_delta_ev_per_e": 1.0e-11 * scale,
        "maximum_online_to_replay_gradient_delta_ev_per_e_angstrom": 1.0e-11
        * scale,
        "maximum_online_to_replay_monopole_response_delta_e": 2.0e-13 * scale,
        "maximum_online_to_replay_dipole_response_delta_e_angstrom": 2.0e-13
        * scale,
        "maximum_online_to_replay_potential_ulp": 8,
        "maximum_online_to_replay_gradient_ulp": 8,
        "maximum_online_to_replay_monopole_ulp": 4,
        "maximum_online_to_replay_dipole_ulp": 4,
        "maximum_monopole_residual_e": 4.0e-11 * scale,
        "maximum_dipole_residual_e_angstrom": 3.0e-11 * scale,
        "intrinsic_ledger_span_ev": 2.0e-11 * scale,
        "pcm_ledger_span_ev": 3.0e-11 * scale,
        "electrostatic_ledger_span_ev": 4.0e-11 * scale,
        "maximum_polarization_identity_error_ev": 5.0e-11 * scale,
    }
    return convergence


def _dataset(fingerprint: str = DEFAULT_DATASET_FINGERPRINT):
    return {
        "source_artifact_sha256": fingerprint,
        "table_sha256": DEFAULT_SOURCE_TABLE,
        "normalized_bundle_sha256": DEFAULT_SOURCE_BUNDLE,
    }


def _selection_manifest():
    return {
        "artifact": "route2-mnsol-partition-selection-v1",
        "selection_fingerprint": SELECTION_FINGERPRINT,
        "partition": "development",
        "selection_status": "frozen-before-partition-run",
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "source_artifact_sha256": "d" * 64,
            "table_sha256": "3" * 64,
            "normalized_bundle_sha256": "4" * 64,
        },
        "selected_records": [
            {
                "selection_index": 0,
                "canonical_solvent": "water",
                "partition": "development",
                "opaque_record_id": "0" * 64,
                "geometry_sha256": "a" * 64,
                "prior_pilot_geometry_overlap": False,
            },
            {
                "selection_index": 1,
                "canonical_solvent": "methanol",
                "partition": "development",
                "opaque_record_id": "1" * 64,
                "geometry_sha256": "b" * 64,
                "prior_pilot_geometry_overlap": True,
            },
        ],
    }


def _fragment(
    shard_index: int,
    *,
    run_kind: str = "partition-record-shard",
    stage: str = "scf",
    status: str = "complete",
    methods: tuple[str, ...] = FULL_METHODS,
    method_scale: float = 1.0,
    continuum_equation: str = "ddpcm",
    continuum_profile: str = aggregation.CONTINUUM_EQUATION_TO_PROFILE["ddpcm"],
    include_resolved_paths: bool = True,
    include_scf_convergence: bool = True,
    scf_reason: str | None = None,
):
    canonical_solvent = "water" if shard_index == 0 else "methanol"
    total = -4.0 + 0.8 * method_scale
    method_rows: dict[str, dict[str, object]] = {
        method: {
            "total_solvation_kcal_mol": total,
            "signed_error_kcal_mol": 0.8 * method_scale,
            "absolute_error_kcal_mol": 0.8 * method_scale,
            "wall_seconds": 2.0,
            "solute_polarization_kcal_mol": -1.0,
            "continuum_polarization_kcal_mol": -2.5,
            "electrostatic_kcal_mol": -3.5,
            "smd_cds_kcal_mol": total + 3.5,
        }
        for method in methods
    }
    if include_scf_convergence:
        scf = _scf_convergence(
            online_candidate_iteration=7 + shard_index,
            solvent=canonical_solvent,
            scale=method_scale,
            reason=scf_reason or "finite-resolution-stagnation-v2",
        )
        method_rows["mace_scf_l1"]["scf_convergence"] = scf
        method_rows["mace_scf_l1"].update(
            {
                "scf_iterations": scf["online_candidate_iteration"],
                "unmixed_density_residual_inf_e": max(
                    scf["final_monopole_residual_e"],
                    scf["final_dipole_residual_e_angstrom"],
                ),
                "half_coupling_identity_error_ev": (
                    5.0e-11 * method_scale
                    if scf["fresh_map_replay"] is not None
                    else 1.0e-13 * method_scale
                ),
            }
        )
    checkpoint_aimnet = {
        "sha256": "3" * 64,
        "size_bytes": 123,
        "identifier": "aimnet-id",
        "resolved_path": "/tmp/aimnet.pt",
    }
    checkpoint_mace = {
        **aggregation.EXPECTED_MACE_POLAR_CHECKPOINT,
        "resolved_path": "/tmp/mace.pt",
    }
    if not include_resolved_paths:
        checkpoint_aimnet.pop("resolved_path")
        checkpoint_mace.pop("resolved_path")

    return {
        "artifact": aggregation.runner.ARTIFACT_NAME,
        "schema_version": aggregation.runner.SCHEMA_VERSION,
        "scf_convergence_contract_version": (
            aggregation.runner.SCF_CONVERGENCE_CONTRACT_VERSION
        ),
        "scf_solver_contract": copy.deepcopy(aggregation.runner.SCF_SOLVER_CONTRACT),
        "status": status,
        "complete_panel": False,
        "run_kind": run_kind,
        "maximum_response_stage": stage,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "selection_fingerprint": SELECTION_FINGERPRINT,
        "execution_git_head": CURRENT_HEAD,
        "do_not_commit": True,
        "visibility": "private-user-supplied-mnsol-row-level",
        "evaluated_methods": list(methods),
        "continuum_equation": continuum_equation,
        "continuum_profile": continuum_profile,
        "dataset": {
            "source_artifact_sha256": DEFAULT_DATASET_FINGERPRINT,
            "table_sha256": DEFAULT_SOURCE_TABLE,
            "normalized_bundle_sha256": DEFAULT_SOURCE_BUNDLE,
        },
        "checkpoints": {
            "aimnet2": checkpoint_aimnet,
            "mace_polar": checkpoint_mace,
        },
        "records": [
            {
                "selection_index": shard_index,
                "canonical_solvent": canonical_solvent,
                "partition": "development",
                "opaque_record_id": "0" * 64 if shard_index == 0 else "1" * 64,
                "geometry_sha256": "a" * 64 if shard_index == 0 else "b" * 64,
                "prior_pilot_geometry_overlap": shard_index == 1,
                "experimental_delta_g_kcal_mol": -4.0,
                "methods": method_rows,
            }
        ],
    }


def _run_aggregation(
    fragments,
    *,
    selection_records,
    dataset: dict | None = None,
):
    return aggregation.aggregate_private_two_member_shards(
        fragments,
        selection_records=selection_records,
        selection_artifact_sha256=SELECTION_ARTIFACT_SHA256,
        selection_fingerprint=SELECTION_FINGERPRINT,
        protocol_fingerprint=PROTOCOL_FINGERPRINT,
        dataset=_dataset() if dataset is None else dataset,
    )


def test_aggregator_cli_imports_the_checkout_when_run_as_a_script():
    completed = subprocess.run(
        [
            sys.executable,
            str(
                BENCHMARK_DIR
                / "aggregate_mnsol_response_partition.py"
            ),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--private-shard" in completed.stdout


def test_partition_selection_manifest_validation_covers_complete_indices_once():
    expected = aggregation._selection_records(_selection_manifest())
    assert expected.keys() == {0, 1}
    assert expected[0]["canonical_solvent"] == "water"
    assert expected[1]["canonical_solvent"] == "methanol"
    assert expected[0]["prior_pilot_geometry_overlap"] is False
    assert expected[1]["prior_pilot_geometry_overlap"] is True


def test_partition_selection_rejects_sealed_confirmation():
    selection = _selection_manifest()
    selection["partition"] = "confirmation"
    for row in selection["selected_records"]:
        row["partition"] = "confirmation"

    with pytest.raises(ValueError, match="partition is unsupported"):
        aggregation._selection_records(selection)


def test_aggregate_rejects_missing_or_duplicate_selection_indices_across_shards():
    selection = _selection_manifest()
    selection_records = aggregation._selection_records(selection)

    first = _fragment(0)
    duplicate = _fragment(0)

    with pytest.raises(ValueError, match="selection indices"):
        _run_aggregation(
            [first, duplicate],
            selection_records=selection_records,
        )

    second_missing = _fragment(0)
    with pytest.raises(ValueError, match="must cover all frozen partition indices"):
        _run_aggregation(
            [second_missing],
            selection_records=selection_records,
        )


def test_aggregate_rejects_cross_partition_or_bad_provenance_and_method_gates():
    selection = _selection_manifest()
    selection_records = aggregation._selection_records(selection)

    cross_partition = _fragment(1)
    cross_partition["records"][0]["partition"] = "confirmation"

    with pytest.raises(ValueError, match="cross-partition"):
        _run_aggregation(
            [_fragment(0), cross_partition],
            selection_records=selection_records,
        )

    with pytest.raises(ValueError, match="status"):
        _run_aggregation(
            [_fragment(0), _fragment(1, status="running")],
            selection_records=selection_records,
        )

    with pytest.raises(ValueError, match="scf_solver_contract"):
        missing = _fragment(1)
        missing.pop("scf_solver_contract")
        _run_aggregation(
            [_fragment(0), missing],
            selection_records=selection_records,
        )

    with pytest.raises(ValueError, match="scf_solver_contract"):
        drifted = _fragment(1)
        drifted["scf_solver_contract"]["scf_mixing"] = 0.75
        _run_aggregation(
            [_fragment(0), drifted],
            selection_records=selection_records,
        )

    with pytest.raises(ValueError, match="method"):
        _run_aggregation(
            [
                _fragment(
                    0,
                    methods=(
                        "mace_fixed_l1",
                        "mace_scf_l1",
                        "mace_one_shot_l1",
                    ),
                ),
                _fragment(1),
            ],
            selection_records=selection_records,
        )

    execution_drift = _fragment(1)
    execution_drift["execution_git_head"] = "8" * 40
    with pytest.raises(ValueError, match="execution heads"):
        _run_aggregation(
            [_fragment(0), execution_drift],
            selection_records=selection_records,
        )

    protocol_drift = _fragment(1)
    protocol_drift["protocol_fingerprint"] = "9" * 64
    with pytest.raises(ValueError, match="protocol fingerprint"):
        _run_aggregation(
            [_fragment(0), protocol_drift],
            selection_records=selection_records,
        )

    audit_checkpoint_drift = _fragment(1)
    audit_checkpoint_drift["checkpoints"]["aimnet2"]["sha256"] = "6" * 64
    with pytest.raises(ValueError, match="checkpoint provenance"):
        _run_aggregation(
            [_fragment(0), audit_checkpoint_drift],
            selection_records=selection_records,
        )

    overlap_drift = _fragment(1)
    overlap_drift["records"][0]["prior_pilot_geometry_overlap"] = False
    with pytest.raises(ValueError, match="identity drifted"):
        _run_aggregation(
            [_fragment(0), overlap_drift],
            selection_records=selection_records,
        )


def test_aggregate_rejects_missing_mace_scf_l1_scf_convergence_evidence():
    selection_records = aggregation._selection_records(_selection_manifest())
    missing = _fragment(0)
    missing["records"][0]["methods"]["mace_scf_l1"].pop("scf_convergence")

    with pytest.raises(ValueError, match="scf_convergence is required"):
        _run_aggregation(
            [_fragment(1), missing],
            selection_records=selection_records,
        )


def test_aggregate_rejects_unknown_mace_scf_l1_scf_convergence_reason():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad_reason = _fragment(1, scf_reason="unknown")

    with pytest.raises(ValueError, match="reason is unsupported"):
        _run_aggregation(
            [_fragment(0), bad_reason],
            selection_records=selection_records,
        )


def test_aggregate_rejects_nominal_with_extra_scf_convergence_evidence():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad_nominal = _fragment(0, scf_reason="nominal-density-and-energy-v1")
    bad_nominal["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "runtime_identity"
    ] = {
        "profile": "synthetic",
        "device": "cpu",
        "dtype": "torch.float64",
        "torch_threads": 1,
    }

    with pytest.raises(ValueError, match="must be null"):
        _run_aggregation(
            [_fragment(1), bad_nominal],
            selection_records=selection_records,
        )


def test_aggregate_rejects_incomplete_mace_scf_l1_scf_convergence_evidence():
    selection_records = aggregation._selection_records(_selection_manifest())
    incomplete = _fragment(1)
    del incomplete["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]

    with pytest.raises(ValueError, match="fresh_map_replay"):
        _run_aggregation(
            [_fragment(0), incomplete],
            selection_records=selection_records,
        )


def test_aggregate_rejects_finite_non_true_replay_flags():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]["cold_replay_field_arrays_identical"] = False

    with pytest.raises(
        ValueError,
        match="fresh_map_replay cold_replay_field_arrays_identical",
    ):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_legacy_map_replay_evidence_field():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]["all_field_arrays_identical"] = True

    with pytest.raises(ValueError, match="exactly the v2 evidence fields"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_finite_bad_replay_digest():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]["field_sha256_by_evaluation"] = ["zz" * 32] * 4

    with pytest.raises(ValueError, match="field_sha256_by_evaluation"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_inconsistent_map_replay_hash_vector():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]["field_sha256_by_evaluation"] = [
        "aa" * 32,
        "aa" * 32,
        "bb" * 32,
        "cc" * 32,
    ]

    with pytest.raises(
        ValueError,
        match="entries 1..3 must match",
    ):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_identical_hashes_with_nonzero_replay_metrics():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    replay = bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]
    replay["field_sha256_by_evaluation"][0] = (
        replay["field_sha256_by_evaluation"][1]
    )

    with pytest.raises(
        ValueError,
        match="identical online/cold field hashes require zero",
    ):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


@pytest.mark.parametrize(
    ("delta", "ulp"),
    (
        (0.0, 1),
        (1.0e-11, 0),
    ),
)
def test_aggregate_rejects_inconsistent_replay_delta_and_ulp(delta, ulp):
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    replay = bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]
    replay["maximum_online_to_replay_potential_delta_ev_per_e"] = delta
    replay["maximum_online_to_replay_potential_ulp"] = ulp

    with pytest.raises(ValueError, match="must be zero together"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_replay_ulp_outside_finite_float64_range():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    replay = bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]
    replay["maximum_online_to_replay_potential_ulp"] = (
        aggregation.MACE_SCF_MAX_FINITE_FLOAT64_ULP_DISTANCE + 1
    )

    with pytest.raises(ValueError, match="exceeds the finite float64 ULP range"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_finite_metric_above_frozen_gate():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "history_window"
    ]["root_monopole_span_e"] = 2.1e-12

    with pytest.raises(ValueError, match="exceeds the frozen convergence gate"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_finite_runtime_identity_drift():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "runtime_identity"
    ]["pyddx_version"] = "0.8.1"

    with pytest.raises(ValueError, match="drifted from the frozen runtime lock"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_wrong_solvent_dielectric():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(0)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "runtime_identity"
    ]["continuum_dielectric"] = 2.0

    with pytest.raises(ValueError, match="frozen water descriptor"):
        _run_aggregation(
            [_fragment(1, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_missing_scf_iterations():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"].pop("scf_iterations")

    with pytest.raises(ValueError, match="scf_iterations"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_final_residual_above_history_maximum():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    method = bad["records"][0]["methods"]["mace_scf_l1"]
    method["scf_convergence"]["final_monopole_residual_e"] = 9.0e-11
    method["unmixed_density_residual_inf_e"] = 9.0e-11

    with pytest.raises(ValueError, match="history-window maximum"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_map_replay_residual_contradiction():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "fresh_map_replay"
    ]["maximum_monopole_residual_e"] = 0.0

    with pytest.raises(ValueError, match="final monopole residual exceeds the replay residual"):
        _run_aggregation(
            [_fragment(0, scf_reason="nominal-density-and-energy-v1"), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_nonfinite_mace_scf_l1_scf_convergence_metric():
    selection_records = aggregation._selection_records(_selection_manifest())
    bad = _fragment(1)
    bad["records"][0]["methods"]["mace_scf_l1"]["scf_convergence"][
        "history_window"
    ]["root_dipole_span_e_angstrom"] = float("inf")

    with pytest.raises(ValueError, match="history_window root_dipole_span_e_angstrom"):
        _run_aggregation(
            [_fragment(0), bad],
            selection_records=selection_records,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("identifier", "fine-tuned-private-model"),
        ("release_url", "https://example.invalid/model"),
        ("sha256", "5" * 64),
        ("size_bytes", 456),
    ),
)
def test_aggregate_rejects_any_nonofficial_mace_checkpoint_metadata(field, value):
    selection_records = aggregation._selection_records(_selection_manifest())
    drifted = _fragment(1)
    drifted["checkpoints"]["mace_polar"][field] = value

    with pytest.raises(ValueError, match="frozen official"):
        _run_aggregation(
            [_fragment(0), drifted],
            selection_records=selection_records,
        )


def test_aggregate_private_and_public_outputs_include_two_member_metrics_and_no_private_rows():
    selection = _selection_manifest()
    selection_records = aggregation._selection_records(selection)

    private, public = _run_aggregation(
        [
            _fragment(0, method_scale=1.0, scf_reason="nominal-density-and-energy-v1"),
            _fragment(1, method_scale=2.0),
        ],
        selection_records=selection_records,
    )

    assert private["artifact"] == aggregation.AGGREGATOR_ARTIFACT_NAME
    assert private["schema_version"] == aggregation.SCHEMA_VERSION
    assert (
        private["scf_convergence_contract_version"]
        == aggregation.runner.SCF_CONVERGENCE_CONTRACT_VERSION
    )
    assert private["scf_solver_contract"] == aggregation.runner.SCF_SOLVER_CONTRACT
    assert private["run_kind"] == "partition-two-member-matrix"
    assert private["source_run_kind"] == "partition-record-shard"
    assert private["status"] == "complete"
    assert private["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
    assert private["selection_artifact_sha256"] == SELECTION_ARTIFACT_SHA256
    assert private["selection_fingerprint"] == SELECTION_FINGERPRINT
    assert "scientific_identity" in private
    assert "claim_boundary" in private
    assert private["complete_panel"] is True
    assert len(private["records"]) == 2
    assert private["source_shards"]["count"] == 2
    assert len(private["source_shard_canonical_sha256"]) == 2
    assert {frozenset(record["methods"].keys()) for record in private["records"]} == {
        frozenset(ALLOWED_METHODS)
    }
    scf_summary = private["scf_convergence_summary"]
    assert scf_summary["record_count"] == 2
    assert scf_summary["nominal_count"] == 1
    assert scf_summary["finite_count"] == 1
    assert scf_summary["reason_counts"] == {
        "nominal-density-and-energy-v1": 1,
        "finite-resolution-stagnation-v2": 1,
    }
    assert scf_summary["max_final_monopole_residual_e"] == pytest.approx(
        8.0e-11
    )
    assert scf_summary["max_final_dipole_residual_e_angstrom"] == pytest.approx(
        6.0e-11
    )
    assert (
        scf_summary["finite_history_window_maximums"]["root_dipole_span_e_angstrom"]
        == pytest.approx(1.2e-12)
    )
    assert (
        scf_summary["finite_history_window_maximums"]["gradient_span_ev_per_angstrom"]
        == pytest.approx(8.0e-11)
    )
    assert (
        scf_summary["finite_map_replay_maximums"]["electrostatic_ledger_span_ev"]
        == pytest.approx(8.0e-11)
    )
    assert (
        "finite-precision approximate fixed-point candidate"
        in private["claim_boundary"]
    )
    assert public["scf_solver_contract"] == aggregation.runner.SCF_SOLVER_CONTRACT

    metrics = public["aggregate_metrics"]
    for method in ALLOWED_METHODS:
        assert metrics[method]["record_count"] == 2
        assert "ge_1_0_count" in metrics[method]
        assert "ge_1_5_fraction" in metrics[method]

    assert public["artifact"] == aggregation.AGGREGATOR_ARTIFACT_NAME
    assert public["run_kind"] == "partition-two-member-matrix"
    assert public["source_run_kind"] == "partition-record-shard"
    assert public["status"] == "complete"
    assert public["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
    assert public["selection_artifact_sha256"] == SELECTION_ARTIFACT_SHA256
    assert public["selection_fingerprint"] == SELECTION_FINGERPRINT
    assert "scientific_identity" in public
    assert "claim_boundary" in public
    assert "records" not in public
    assert public["selection"]["record_count"] == 2
    assert public["selection"]["partition"] == "development"
    assert public["selection"]["unique_geometry_count"] == 2
    assert public["selection"]["prior_pilot_geometry_overlap_record_count"] == 1
    assert (
        public["selection"]["prior_pilot_geometry_overlap_unique_geometry_count"] == 1
    )
    assert "resolved_path" not in public["member_checkpoint"]
    assert "source_runner_checkpoints" not in public
    assert "source_shard_canonical_sha256" not in public
    assert public["source_execution"]["execution_git_head"] == CURRENT_HEAD
    assert len(public["source_execution"]["execution_git_tree_sha1"]) == 40
    assert len(public["source_execution"]["source_runner_git_blob_sha1"]) == 40
    assert len(public["source_execution"]["source_runner_sha256"]) == 64
    assert public["scf_convergence_summary"] == scf_summary
    assert (
        "finite-precision approximate fixed-point candidate"
        in public["claim_boundary"]
    )
    assert all(
        "resolved_path" not in checkpoint
        for checkpoint in private["source_runner_checkpoints"].values()
    )


def test_aggregate_accepts_full_five_method_input_and_outputs_only_two_members():
    selection = _selection_manifest()
    selection_records = aggregation._selection_records(selection)

    private, public = _run_aggregation(
        [_fragment(0), _fragment(1)],
        selection_records=selection_records,
    )

    assert {frozenset(record["methods"].keys()) for record in private["records"]} == {
        frozenset(ALLOWED_METHODS)
    }
    assert sorted(public["member_checkpoint"]) == [
        "identifier",
        "release_url",
        "sha256",
        "size_bytes",
    ]


def test_private_rows_retain_full_mace_scf_l1_scf_convergence_evidence():
    selection_records = aggregation._selection_records(_selection_manifest())
    private, _ = _run_aggregation(
        [_fragment(0, scf_reason="nominal-density-and-energy-v1"), _fragment(1)],
        selection_records=selection_records,
    )

    row = private["records"][0]["methods"]["mace_scf_l1"]
    assert row["scf_convergence"]["reason"] == "nominal-density-and-energy-v1"
    assert row["scf_convergence"]["online_candidate_iteration"] == 7
    assert row["scf_convergence"]["runtime_identity"] is None
    assert row["scf_convergence"]["history_window"] is None
    assert row["scf_convergence"]["fresh_map_replay"] is None
    assert row["scf_convergence"]["final_monopole_residual_e"] == pytest.approx(
        5.0e-13
    )
    assert row["scf_convergence"]["final_dipole_residual_e_angstrom"] == pytest.approx(
        4.0e-13
    )
    finite_row = private["records"][1]["methods"]["mace_scf_l1"]
    assert finite_row["scf_convergence"]["reason"] == (
        "finite-resolution-stagnation-v2"
    )
    assert (
        finite_row["scf_convergence"]["fresh_map_replay"]["replay_count"]
        == 3
    )
    assert isinstance(
        finite_row["scf_convergence"]["history_window"]["start_iteration"],
        int,
    )
    assert isinstance(
        finite_row["scf_convergence"]["history_window"]["end_iteration"],
        int,
    )


def test_aggregate_accepts_distinct_online_and_repeatable_cold_replay_hashes():
    selection_records = aggregation._selection_records(_selection_manifest())
    finite = _fragment(1)
    replay = finite["records"][0]["methods"]["mace_scf_l1"][
        "scf_convergence"
    ]["fresh_map_replay"]
    replay["field_sha256_by_evaluation"] = ["cc" * 32] + ["aa" * 32] * 3
    replay["response_sha256_by_evaluation"] = ["dd" * 32] + ["bb" * 32] * 3
    replay["maximum_online_to_replay_potential_ulp"] = 17
    replay["maximum_online_to_replay_monopole_ulp"] = 5

    private, public = _run_aggregation(
        [_fragment(0, scf_reason="nominal-density-and-energy-v1"), finite],
        selection_records=selection_records,
    )

    accepted = private["records"][1]["methods"]["mace_scf_l1"][
        "scf_convergence"
    ]["fresh_map_replay"]
    assert accepted["field_sha256_by_evaluation"][0] != (
        accepted["field_sha256_by_evaluation"][1]
    )
    assert len(set(accepted["field_sha256_by_evaluation"][1:])) == 1
    assert public["scf_convergence_summary"][
        "finite_map_replay_ulp_maximums"
    ]["maximum_online_to_replay_potential_ulp"] == 17


def test_threshold_metrics_treat_exact_targets_as_failures_of_strict_bounds():
    selection_records = aggregation._selection_records(_selection_manifest())

    _, public = _run_aggregation(
        [
            _fragment(0, method_scale=1.25),
            _fragment(1, method_scale=1.875),
        ],
        selection_records=selection_records,
    )

    for method in ALLOWED_METHODS:
        metrics = public["aggregate_metrics"][method]
        assert metrics["ge_1_0_count"] == 2
        assert metrics["ge_1_0_fraction"] == pytest.approx(1.0)
        assert metrics["ge_1_5_count"] == 1
        assert metrics["ge_1_5_fraction"] == pytest.approx(0.5)


def test_aggregate_rejects_continuum_drift_between_shards():
    selection = _selection_manifest()
    selection_records = aggregation._selection_records(selection)
    other_equation = "ddcosmo"
    bad = _fragment(
        1,
        continuum_equation=other_equation,
        continuum_profile=aggregation.CONTINUUM_EQUATION_TO_PROFILE[other_equation],
    )
    with pytest.raises(ValueError, match="requires the frozen ddPCM"):
        _run_aggregation(
            [_fragment(0), bad],
            selection_records=selection_records,
        )


def test_aggregate_rejects_internally_consistent_ddcosmo_matrix():
    selection_records = aggregation._selection_records(_selection_manifest())
    equation = "ddcosmo"
    fragments = [
        _fragment(
            index,
            continuum_equation=equation,
            continuum_profile=aggregation.CONTINUUM_EQUATION_TO_PROFILE[equation],
        )
        for index in (0, 1)
    ]

    with pytest.raises(ValueError, match="requires the frozen ddPCM"):
        _run_aggregation(
            fragments,
            selection_records=selection_records,
        )


def test_aggregate_rejects_nonexistent_shared_execution_commit():
    selection_records = aggregation._selection_records(_selection_manifest())
    fragments = [_fragment(0), _fragment(1)]
    for fragment in fragments:
        fragment["execution_git_head"] = "7" * 40

    with pytest.raises(ValueError, match="not a local Git commit"):
        _run_aggregation(
            fragments,
            selection_records=selection_records,
        )


def test_aggregate_rejects_drifted_method_ledger():
    selection_records = aggregation._selection_records(_selection_manifest())
    drifted = _fragment(1)
    drifted["records"][0]["methods"]["mace_scf_l1"]["absolute_error_kcal_mol"] = 99.0

    with pytest.raises(ValueError, match="absolute-error ledger"):
        _run_aggregation(
            [_fragment(0), drifted],
            selection_records=selection_records,
        )


def test_selection_manifest_rejects_non_hex_record_identity():
    selection = _selection_manifest()
    selection["selected_records"][0]["opaque_record_id"] = "z" * 64

    with pytest.raises(ValueError, match="hex"):
        aggregation._selection_records(selection)


def test_aggregate_rejects_non_hex_dataset_metadata():
    selection_records = aggregation._selection_records(_selection_manifest())

    with pytest.raises(ValueError, match="source_artifact_sha256"):
        _run_aggregation(
            [_fragment(0), _fragment(1)],
            selection_records=selection_records,
            dataset={
                "source_artifact_sha256": "not-hex",
                "table_sha256": DEFAULT_SOURCE_TABLE,
                "normalized_bundle_sha256": DEFAULT_SOURCE_BUNDLE,
            },
        )


def test_output_paths_keep_private_rows_below_omx_and_public_under_docs():
    private = ROOT / ".omx" / "benchmarks" / "unused-private.json"
    public = ROOT / "docs" / "implicit-solvation" / "benchmarks" / "unused-public.json"
    assert aggregation._validated_output_paths(private, public) == (
        private.resolve(),
        public.resolve(),
    )

    with pytest.raises(ValueError, match="must remain below"):
        aggregation._validated_output_paths(
            ROOT / "private-row-leak.json",
            public,
        )
    with pytest.raises(ValueError, match="must remain below"):
        aggregation._validated_output_paths(
            private,
            ROOT / "public-output.json",
        )
