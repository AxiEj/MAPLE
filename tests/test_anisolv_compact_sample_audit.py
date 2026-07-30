from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "docs" / "pretrained-solvation-hub" / "run_anisolv_compact_sample_audit.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "anisolv-compact-upstream-sample-audit-2026-07-30.json"
)
CONFIGURED_UPSTREAM = os.environ.get("MAPLE_ANISOLV_AUDIT_ROOT")
WATCHLIST = ROOT / "docs" / "pretrained-solvation-hub" / "research_watchlist.yaml"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "anisolv_compact_sample_audit", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_revision_identity_gate_fails_closed(monkeypatch, tmp_path):
    def fake_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "0" * 40
        raise AssertionError(f"unexpected git call: {args!r}")

    monkeypatch.setattr(AUDIT, "_git", fake_git)

    with pytest.raises(ValueError, match="revision"):
        AUDIT.verify_upstream_identity(tmp_path)


def test_blob_identity_gate_rejects_wrong_size_and_hash(tmp_path):
    candidate = tmp_path / "blob"
    candidate.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="size"):
        AUDIT._verify_file(
            tmp_path,
            {"path": "blob", "size_bytes": 6, "sha256": "unused"},
            label="synthetic",
        )

    with pytest.raises(ValueError, match="SHA256"):
        AUDIT._verify_file(
            tmp_path,
            {"path": "blob", "size_bytes": 5, "sha256": "0" * 64},
            label="synthetic",
        )


def test_ignored_checkout_policy_allows_only_inert_bytecode_paths():
    AUDIT._verify_ignored_paths_are_bytecode_only(
        [
            "__pycache__/module.cpython-311.pyc",
            "anisolv/_backbone/__pycache__/rotation.cpython-311.pyc",
        ]
    )

    for unsafe in (
        "sibling.py",
        "__pycache__/native.so",
        "__pycache__/executable",
        "../__pycache__/escape.pyc",
    ):
        with pytest.raises(ValueError, match="ignored non-bytecode"):
            AUDIT._verify_ignored_paths_are_bytecode_only([unsafe])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_finite_scalar_gate_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="not finite"):
        AUDIT._finite_float(value, label="synthetic")


def test_atomic_json_writer_rejects_nonfinite_payload_without_replacing_output(
    tmp_path,
):
    output = tmp_path / "audit.json"
    output.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON compliant"):
        AUDIT._write_json_atomic(output, {"unsafe": float("nan")})

    assert output.read_text(encoding="utf-8") == '{"preserved": true}\n'
    assert list(tmp_path.iterdir()) == [output]


def test_frozen_sample_artifact_preserves_scientific_and_gpu_boundaries():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert payload["identity"]["revision"] == AUDIT.UPSTREAM_REVISION
    assert payload["identity"]["tree"] == AUDIT.UPSTREAM_TREE
    assert payload["identity"]["checkpoint"]["sha256"] == AUDIT.CHECKPOINT["sha256"]
    assert payload["panel"]["geometry_and_source_row_sha256"] == (
        AUDIT.PANEL_IDENTITY_SHA256
    )
    assert payload["panel"]["experimental_label_values_embedded"] is False
    assert payload["panel"]["experimental_label_values_used_for_identity"] is False
    assert payload["panel"]["record_count"] == 6
    assert len(payload["panel"]["records"]) == 6
    assert payload["panel"]["vacuum_gate"] == {
        "correction_energy_ev": 0.0,
        "correction_forces_exactly_zero": True,
        "correction_forces_shape": [3, 3],
        "solute": "water",
    }
    assert payload["metrics"] is None
    assert "diagnostic_experimental_comparison" not in payload
    assert payload["route4_accuracy_gate"] == {
        "accuracy_metric_reporting_allowed": False,
        "evaluated": False,
        "ideal_max_error_kcal_mol": 1.0,
        "ideal_passed": False,
        "required_max_error_kcal_mol": 1.5,
        "required_passed": False,
        "reason": (
            "Accuracy evaluation is forbidden: this panel has six records and "
            "no predeclared ten-functional-group taxonomy."
        ),
    }
    assert payload["functional_group_accuracy_gate"] == {
        "accuracy_metric_reporting_allowed": False,
        "functional_group_taxonomy": None,
        "minimum_distinct_functional_groups": 10,
        "minimum_record_count": 10,
        "observed_distinct_functional_groups": 0,
        "observed_record_count": 6,
        "passes": False,
        "record_assignments_predeclared": False,
        "scope": "runtime_or_interface_smoke_only",
    }
    assert payload["scientific_scope"]["independent_experimental_holdout"] is False
    assert payload["scientific_scope"]["literature_accuracy_certification"] is False
    assert payload["scientific_scope"]["experimental_labels_used_for_accuracy"] is False
    assert (
        payload["scientific_scope"]["complete_standard_state_cycle_executed"] is False
    )
    assert payload["panel"]["training_overlap"] == "unknown"
    assert payload["gpu_acceleration"]["gpu_executed"] is False
    assert payload["gpu_acceleration"]["gpu_parity_demonstrated"] is False
    assert set(payload["gpu_acceleration"]["reduced_precision_controls"].values()) == {
        False,
        "highest",
    }
    assert payload["model_runtime_fingerprint"]["parameter_devices"] == ["cpu"]
    assert payload["model_runtime_fingerprint"]["parameter_dtypes"] == ["torch.float32"]
    isolation = payload["source_execution_isolation"]
    assert isolation["git_archive_from_verified_tree"] is True
    assert isolation["ignored_checkout_files_excluded_from_execution"] is True
    assert isolation["isolated_import_parent"] is True
    assert isolation["isolated_pycache_root"] is True
    assert isolation["loaded_anisolv_module_origins_verified"] is True
    origins = payload["model_runtime_fingerprint"]["loaded_source_modules"]
    assert origins["anisolv"] == "__init__.py"
    assert all(not value.startswith("/") for value in origins.values())


def test_frozen_artifact_recomputes_all_row_integrity_without_accuracy():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = payload["panel"]["records"]
    assert len(records) == payload["panel"]["record_count"]

    for record in records:
        force_digest = hashlib.sha256(
            AUDIT._canonical_bytes(record["correction_forces_ev_per_angstrom"])
        ).hexdigest()
        assert record["correction_forces_sha256"] == force_digest

        assert math.isfinite(record["correction_energy_ev"])
        assert "experiment_kcal_mol" not in record
        assert "prediction_kcal_mol" not in record
        assert "error_kcal_mol" not in record
        assert "absolute_error_kcal_mol" not in record

    assert (
        payload["panel"]["geometry_and_source_row_sha256"]
        == AUDIT.PANEL_IDENTITY_SHA256
    )


def test_watchlist_binds_the_exact_negative_development_artifact():
    watchlist = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    candidate = next(
        model for model in watchlist["models"] if model["model_id"] == "anisolv-compact"
    )
    audit = candidate["development_audit"]

    assert audit["artifact"] == ARTIFACT.relative_to(WATCHLIST.parent).as_posix()
    assert audit["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert "no_experimental_accuracy_calculation" in audit["status"]
    assert "gpu_parity" in audit["status"]


def test_frozen_artifact_equals_fresh_exact_upstream_audit(tmp_path):
    if CONFIGURED_UPSTREAM is None:
        pytest.skip(
            "Set MAPLE_ANISOLV_AUDIT_ROOT to run the external-checkout integration test."
        )
    upstream = Path(CONFIGURED_UPSTREAM)
    if not upstream.is_dir():
        pytest.fail(f"Configured AniSolv audit checkout does not exist: {upstream}")

    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--upstream-root",
            str(upstream),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.read_bytes() == ARTIFACT.read_bytes()
