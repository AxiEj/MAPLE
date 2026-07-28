from __future__ import annotations

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
):
    total = -4.0 + 0.8 * method_scale
    method_rows = {
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
        "artifact": "route2-mnsol-macepolar-response-ablation-v1",
        "schema_version": 1,
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
                "canonical_solvent": "water" if shard_index == 0 else "methanol",
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
        [_fragment(0, method_scale=1.0), _fragment(1, method_scale=2.0)],
        selection_records=selection_records,
    )

    assert private["artifact"] == "route2-mnsol-macepolar-two-member-matrix-v1"
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

    metrics = public["aggregate_metrics"]
    for method in ALLOWED_METHODS:
        assert metrics[method]["record_count"] == 2
        assert "ge_1_0_count" in metrics[method]
        assert "ge_1_5_fraction" in metrics[method]

    assert public["artifact"] == "route2-mnsol-macepolar-two-member-matrix-v1"
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
