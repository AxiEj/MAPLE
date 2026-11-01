"""Validate source-bound deterministic Route 1 algorithm contract evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from maple.function.calculator.algorithm_contracts import (
    CANONICAL_ROWS,
    CANONICAL_ROWS_BY_ID,
    EXPECTED_CAPABILITIES,
    HOSTILE_ROW_IDS,
    expected_outcome,
    fixture_identity,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_payload(artifact: dict) -> bytes:
    payload = dict(artifact)
    payload.pop("content_sha256", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replay_nodes() -> None:
    for node in dict.fromkeys(spec.test_node for spec in CANONICAL_ROWS):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", node],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            detail = (completed.stdout + completed.stderr).strip()
            raise ValueError(f"algorithm contract replay failed: {node}\n{detail}")


def validate_artifact(
    artifact: dict,
    *,
    expected_source_snapshot: dict,
    replay: bool = False,
) -> dict:
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("algorithm contract artifact schema mismatch")
    if artifact.get("artifact_type") != "route1-advanced-algorithm-contracts":
        raise ValueError("algorithm contract artifact type mismatch")
    if (
        artifact.get("content_sha256")
        != hashlib.sha256(canonical_payload(artifact)).hexdigest()
    ):
        raise ValueError("algorithm contract content seal mismatch")
    if artifact.get("source_before") != expected_source_snapshot:
        raise ValueError("algorithm contract source-before snapshot mismatch")
    if artifact.get("source_after") != expected_source_snapshot:
        raise ValueError("algorithm contract source changed during execution")
    capabilities = artifact.get("capability_keys")
    if capabilities != sorted(EXPECTED_CAPABILITIES):
        raise ValueError("algorithm contract capability set mismatch")
    rows = artifact.get("rows")
    if not isinstance(rows, dict) or set(rows) != set(CANONICAL_ROWS_BY_ID):
        raise ValueError("algorithm contract canonical rows are missing or changed")
    positive = set()
    hostile = set()
    for row_id, row in rows.items():
        if not isinstance(row_id, str) or not isinstance(row, dict):
            raise ValueError("algorithm contract row is malformed")
        spec = CANONICAL_ROWS_BY_ID[row_id]
        if row.get("passed") is not True:
            raise ValueError(f"algorithm contract row failed: {row_id}")
        capability = row.get("capability_key")
        if capability != spec.capability:
            raise ValueError(f"algorithm contract capability binding changed: {row_id}")
        if spec.capability is None:
            hostile.add(row_id)
        else:
            positive.add(spec.capability)
        implementations = row.get("implementation_files_sha256")
        if not isinstance(implementations, dict) or set(implementations) != {
            spec.implementation
        }:
            raise ValueError(
                f"algorithm contract implementation binding missing: {row_id}"
            )
        for relative, digest in implementations.items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"algorithm contract implementation drift: {row_id}")
        test_file = spec.test_node.split("::", 1)[0]
        if (
            row.get("test_file") != test_file
            or not (ROOT / test_file).is_file()
            or _sha256(ROOT / test_file) != row.get("test_file_sha256")
            or row.get("test_node") != spec.test_node
        ):
            raise ValueError(f"algorithm contract test binding drift: {row_id}")
        outcome = expected_outcome(spec)
        if row.get("fixture_identity") != fixture_identity(spec):
            raise ValueError(f"algorithm contract fixture binding drift: {row_id}")
        if row.get("expected") != outcome or row.get("observed") != outcome:
            raise ValueError(f"algorithm contract outcome mismatch: {row_id}")
    if positive != EXPECTED_CAPABILITIES or hostile != HOSTILE_ROW_IDS:
        raise ValueError("algorithm contract positive or hostile rows are incomplete")
    if replay:
        _replay_nodes()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_git_sha": expected_source_snapshot["git_head"],
        "source_inventory_aggregate_sha256": expected_source_snapshot[
            "files_aggregate_sha256"
        ],
        "capability_keys": sorted(EXPECTED_CAPABILITIES),
        "rows": rows,
    }
