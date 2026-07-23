from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core


def _protocol_path(tmp_path: Path) -> Path:
    protocol = json.loads(
        (BENCHMARK_DIR / "route2-protocol.json").read_text(encoding="utf-8")
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path


def test_load_protocol_rejects_missing_required_key(tmp_path):
    path = _protocol_path(tmp_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    del protocol["result_schema_version"]
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        core.load_protocol(path)


def test_load_protocol_rejects_artifact_with_non_https_url(tmp_path):
    path = _protocol_path(tmp_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["dataset"]["artifacts"][0]["url"] = "http://example.invalid/database.txt"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="requires an HTTPS source URL"):
        core.load_protocol(path)


def test_load_protocol_rejects_bad_bootstrap_confidence(tmp_path):
    path = _protocol_path(tmp_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["statistics"]["bootstrap_confidence"] = 0
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="bootstrap_confidence must be strictly"):
        core.load_protocol(path)


def test_safe_extract_tar_blocks_parent_relative_path(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("bad", encoding="utf-8")

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="../evil.txt")

    with pytest.raises(ValueError, match="Unsafe path in dataset archive"):
        core.safe_extract_tar(archive, tmp_path / "extract")


def test_summarize_errors_empty_errors_records_failure_rate_and_ci_none():
    metrics = core.summarize_errors(
        [],
        expected_count=3,
        resamples=100,
        confidence=0.95,
        seed=2026,
    )

    assert metrics["n"] == 0
    assert metrics["expected_count"] == 3
    assert metrics["failure_count"] == 3
    assert metrics["failure_rate"] == pytest.approx(1.0)
    assert metrics["bootstrap_ci"] == {"mse": None, "mae": None, "rmse": None}
    assert metrics["max_absolute_error"] is None


def test_summarize_errors_rejects_non_finite_input():
    errors = [0.0, float("nan")]
    with pytest.raises(ValueError, match="Summary errors must all be finite"):
        core.summarize_errors(
            errors,
            expected_count=2,
            resamples=10,
            confidence=0.95,
            seed=2026,
        )


def test_partition_for_smiles_is_deterministic_and_forced_development():
    partition_a = core.partition_for_smiles(
        "CCO", seed="seed-2026", development_fraction=0.5, forced_development=False
    )
    partition_b = core.partition_for_smiles(
        "CCO", seed="seed-2026", development_fraction=0.5, forced_development=False
    )

    assert partition_a == partition_b
    assert partition_a in {"development", "confirmation"}
    assert (
        core.partition_for_smiles(
            "CCO", seed="seed-2026", development_fraction=0.5, forced_development=True
        )
        == "development"
    )
