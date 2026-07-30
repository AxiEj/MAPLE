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
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_anisolv_compact_force_consistency_audit.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "anisolv-compact-force-consistency-audit-2026-07-30.json"
)
CONFIGURED_UPSTREAM = os.environ.get("MAPLE_ANISOLV_AUDIT_ROOT")
WATCHLIST = ROOT / "docs" / "pretrained-solvation-hub" / "research_watchlist.yaml"
MODEL_CARDS = ROOT / "maple" / "function" / "calculator" / "model_cards"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "anisolv_compact_force_consistency_audit", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _values(array_record):
    values = array_record["values"]
    assert (
        array_record["sha256"] == hashlib.sha256(_canonical_bytes(values)).hexdigest()
    )
    return values


def _max_abs_difference(left, right):
    return max(
        abs(left_row[column] - right_row[column])
        for left_row, right_row in zip(left, right)
        for column in range(len(left_row))
    )


def test_frozen_force_audit_enforces_the_energy_only_disposition():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert payload["identity"]["revision"] == AUDIT.SOURCE_AUDIT.UPSTREAM_REVISION
    assert payload["identity"]["tree"] == AUDIT.SOURCE_AUDIT.UPSTREAM_TREE
    assert payload["identity"]["checkpoint"]["sha256"] == (
        AUDIT.SOURCE_AUDIT.CHECKPOINT["sha256"]
    )
    assert payload["identity"]["panel_geometry_and_source_row_sha256"] == (
        AUDIT.SOURCE_AUDIT.PANEL_IDENTITY_SHA256
    )
    assert payload["identity"]["experimental_label_values_used_for_identity"] is False
    assert payload["finite_difference"]["step_angstrom"] == 1.0e-5
    assert payload["disposition"] == {
        "maple_runtime_scope": "single_point_scalar_energy_only",
        "optimization_frequency_md_forbidden": True,
        "upstream_force_admission": "rejected",
    }

    metrics = payload["metrics"]
    assert metrics["energy_rotation_tolerance_passed"] is True
    assert metrics["energy_rotation_max_abs_drift_ev"] == 0.0
    assert metrics["original_orientation_passed"] is False
    assert metrics["rotated_orientation_passed"] is True
    assert metrics[
        "original_orientation_max_abs_difference_ev_per_angstrom"
    ] == pytest.approx(0.005704464218729888)
    assert metrics[
        "rotated_orientation_max_abs_difference_ev_per_angstrom"
    ] == pytest.approx(1.0869843358740638e-09)

    records = {record["g2_name"]: record for record in payload["records"]}
    assert set(records) == {"H2O", "CH3OH"}
    assert records["H2O"]["original"][
        "max_abs_force_difference_ev_per_angstrom"
    ] == pytest.approx(0.004636657201139)
    assert records["CH3OH"]["original"][
        "max_abs_force_difference_ev_per_angstrom"
    ] == pytest.approx(0.005704464218729888)

    fingerprint = payload["model_runtime_fingerprint"]
    assert fingerprint["parameter_devices"] == ["cpu"]
    assert fingerprint["parameter_dtypes"] == ["torch.float64"]
    assert fingerprint["cache_key"]["dtype"] == "torch.float64"
    assert payload["runtime"]["device"] == "cpu"
    assert payload["runtime"]["dtype"] == "torch.float64"
    assert (
        payload["source_execution_isolation"]["git_archive_from_verified_tree"] is True
    )
    assert (
        payload["source_execution_isolation"][
            "ignored_checkout_files_excluded_from_execution"
        ]
        is True
    )


def test_frozen_force_audit_recomputes_arrays_digests_and_metrics():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    original_maximum = 0.0
    rotated_maximum = 0.0
    energy_maximum = 0.0
    for record in payload["records"]:
        assert math.isfinite(record["energy_ev"])
        assert math.isfinite(record["rotated"]["energy_ev"])
        energy_drift = abs(record["rotated"]["energy_ev"] - record["energy_ev"])
        assert record["energy_rotation_abs_drift_ev"] == energy_drift
        energy_maximum = max(energy_maximum, energy_drift)

        original_autograd = _values(
            record["original"]["autograd_forces_ev_per_angstrom"]
        )
        original_fd = _values(
            record["original"]["finite_difference_forces_ev_per_angstrom"]
        )
        original_difference = _max_abs_difference(original_autograd, original_fd)
        assert (
            record["original"]["max_abs_force_difference_ev_per_angstrom"]
            == original_difference
        )
        original_maximum = max(original_maximum, original_difference)

        rotated_autograd = _values(record["rotated"]["autograd_forces_ev_per_angstrom"])
        rotated_fd = _values(
            record["rotated"]["finite_difference_forces_ev_per_angstrom"]
        )
        rotated_difference = _max_abs_difference(rotated_autograd, rotated_fd)
        assert (
            record["rotated"]["max_abs_force_difference_ev_per_angstrom"]
            == rotated_difference
        )
        rotated_maximum = max(rotated_maximum, rotated_difference)

        rotated_back = _values(
            record["rotation_covariance"]["rotated_forces_mapped_back_ev_per_angstrom"]
        )
        covariance_difference = _max_abs_difference(original_autograd, rotated_back)
        assert (
            record["rotation_covariance"]["max_abs_force_difference_ev_per_angstrom"]
            == covariance_difference
        )
        _values(record["positions_angstrom"])
        _values(record["rotated"]["positions_angstrom"])

    metrics = payload["metrics"]
    assert metrics["energy_rotation_max_abs_drift_ev"] == energy_maximum
    assert (
        metrics["original_orientation_max_abs_difference_ev_per_angstrom"]
        == original_maximum
    )
    assert (
        metrics["rotated_orientation_max_abs_difference_ev_per_angstrom"]
        == rotated_maximum
    )
    _values(payload["rotation"]["matrix"])


def test_force_artifact_hash_is_bound_by_watchlist_and_model_cards():
    expected_sha256 = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    watchlist = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    candidate = next(
        model for model in watchlist["models"] if model["model_id"] == "anisolv-compact"
    )
    audit = candidate["force_consistency_audit"]

    assert audit["artifact"] == ARTIFACT.relative_to(WATCHLIST.parent).as_posix()
    assert audit["artifact_sha256"] == expected_sha256
    assert "force_rejected" in audit["status"]

    for name in ("anisolv-compact.yaml", "anisolv-uma.yaml"):
        card = json.loads((MODEL_CARDS / name).read_text(encoding="utf-8"))
        evidence = card["force_consistency_evidence"]
        assert evidence["artifact"] == ARTIFACT.relative_to(ROOT).as_posix()
        assert evidence["artifact_sha256"] == expected_sha256
        assert evidence["status"].startswith("rejected_")


def test_force_artifact_equals_fresh_exact_upstream_audit(tmp_path):
    if CONFIGURED_UPSTREAM is None:
        pytest.skip(
            "Set MAPLE_ANISOLV_AUDIT_ROOT to run the external-checkout integration test."
        )
    upstream = Path(CONFIGURED_UPSTREAM)
    if not upstream.is_dir():
        pytest.fail(f"Configured AniSolv audit checkout does not exist: {upstream}")

    output = tmp_path / "force-audit.json"
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
