from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-matrix-free-feedback-gain-two-state-v1.json"
)
ARTIFACT_SHA256 = "7683c93ae373341de0dcd8563e48a092a9ff231965c326d68cc21de1830dae48"
DIAGNOSTIC_SOURCE = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_thermodynamic_diagnostics.py"
)
SYNTHETIC_TEST_SOURCE = "tests/solvation/test_route2_thermodynamic_diagnostics.py"
DIAGNOSTIC_SYMBOL = "matrix_free_fixed_point_feedback_gain_diagnostic"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_feedback_gain_artifact_is_anonymous_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    assert artifact["artifact"] == ("route2-matrix-free-feedback-gain-two-state-v1")
    assert artifact["schema_version"] == 1
    assert artifact["visibility"] == ("public-anonymous-fixed-state-evidence")
    assert [row["state_id"] for row in artifact["records"]] == [
        "state-01",
        "state-02",
    ]
    serialized = json.dumps(artifact, sort_keys=True).lower()
    for private_token in (
        "acetic",
        "methoxybenzoic",
        "experimental",
        "solute_name",
        "entry_number",
    ):
        assert private_token not in serialized

    source = artifact["execution_source"]
    assert source["execution_git_head"] == ("ca5aed354e06f679a2903b209874393200a9f859")
    landed_commit = source["implementation_landed_commit"]
    assert landed_commit == "f833d8e2af5435631bd67040c82c6d02d856f192"
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", landed_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )

    for path in (DIAGNOSTIC_SOURCE, SYNTHETIC_TEST_SOURCE):
        identity = source["files"][path]
        assert _sha256(ROOT / path) == identity["sha256"]
        committed_bytes = _git_blob_bytes(landed_commit, path)
        assert hashlib.sha256(committed_bytes).hexdigest() == identity["sha256"]
        blob = subprocess.run(
            ["git", "rev-parse", f"{landed_commit}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert blob == identity["landed_git_blob_sha1"]

    checkpoint = source["checkpoint_binding"]
    assert checkpoint["checkpoint_sha256_recorded"] is False
    assert "rather than a checkpoint-bound rerun" in checkpoint["limitation"]


def test_feedback_gain_artifact_matches_dense_oracles_and_records_cost():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = artifact["records"]

    assert len(records) == 2
    for row in records:
        assert row["matrix_free_sigma_max_estimate"] == pytest.approx(
            row["dense_oracle_sigma_max"],
            abs=2.0e-10,
            rel=0.0,
        )
        assert row["absolute_difference"] <= 2.0e-10
        assert row["relative_singular_triplet_residual"] < 1.0e-9
        assert row["feedback_jvp_applications"] > 0
        assert row["feedback_vjp_applications"] > 0
        assert row["dense_oracle_seconds"] > 0.0
        assert row["matrix_free_seconds"] > 0.0
        assert row["speedup_excluding_model_load_and_setup"] > 1.0

    aggregate = artifact["aggregate"]
    assert aggregate["state_count"] == len(records)
    assert aggregate["maximum_sigma_max_estimate"] == max(
        row["matrix_free_sigma_max_estimate"] for row in records
    )
    assert aggregate["maximum_absolute_dense_difference"] == max(
        row["absolute_difference"] for row in records
    )
    assert aggregate["maximum_relative_singular_triplet_residual"] == max(
        row["relative_singular_triplet_residual"] for row in records
    )
    assert aggregate["mean_matrix_free_seconds"] == pytest.approx(
        sum(row["matrix_free_seconds"] for row in records) / len(records),
        abs=1.0e-12,
    )
    assert aggregate["mean_dense_oracle_seconds"] == pytest.approx(
        sum(row["dense_oracle_seconds"] for row in records) / len(records),
        abs=1.0e-12,
    )

    boundary = artifact["claim_boundary"]
    assert "not a certified nonlinear contraction upper bound" in boundary
    assert "Banach certificate" in boundary
    assert artifact["diagnostic_protocol"]["chemistry_accuracy_evaluated"] is False


def test_feedback_gain_diagnostic_remains_outside_public_calculation_paths():
    for path in (ROOT / "maple").rglob("*.py"):
        if path.relative_to(ROOT).as_posix() == DIAGNOSTIC_SOURCE:
            continue
        assert DIAGNOSTIC_SYMBOL not in path.read_text(encoding="utf-8")

    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["diagnostic_protocol"]["default_scf_path_changed"] is False


def test_feedback_gain_documentation_rejects_certificate_language():
    formulas = (ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md").read_text(
        encoding="utf-8"
    )
    benchmark = (ROOT / "docs/implicit-solvation/benchmarks/README.md").read_text(
        encoding="utf-8"
    )
    validation = (ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md").read_text(
        encoding="utf-8"
    )
    normalized_validation = " ".join(validation.split())

    assert "certified uniform bound" in formulas
    assert "not a Banach contraction certificate" in benchmark
    assert "not certified upper bounds" in normalized_validation
