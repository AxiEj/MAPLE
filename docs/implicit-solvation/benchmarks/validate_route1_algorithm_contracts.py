"""Validate source-bound deterministic Route 1 algorithm contract evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from maple.function.dispatcher.method_registry import CapabilityKey


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
HOSTILE_ROW_IDS = frozenset({
    "hostile.unknown-key", "hostile.forged-binding", "hostile.source-drift",
    "hostile.wrong-checkpoint-state", "hostile.unsupported-charge-spin",
    "hostile.unsupported-provider-gb", "hostile.stale-energy-force",
    "hostile.bounded-outcome", "hostile.false-ts", "hostile.unknown-stage",
})
EXPECTED_CAPABILITIES = frozenset(key.value for key in CapabilityKey)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_payload(artifact: dict) -> bytes:
    payload = dict(artifact)
    payload.pop("content_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def validate_artifact(artifact: dict, *, expected_source_snapshot: dict) -> dict:
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("algorithm contract artifact schema mismatch")
    if artifact.get("artifact_type") != "route1-advanced-algorithm-contracts":
        raise ValueError("algorithm contract artifact type mismatch")
    if artifact.get("content_sha256") != hashlib.sha256(canonical_payload(artifact)).hexdigest():
        raise ValueError("algorithm contract content seal mismatch")
    if artifact.get("source_before") != expected_source_snapshot:
        raise ValueError("algorithm contract source-before snapshot mismatch")
    if artifact.get("source_after") != expected_source_snapshot:
        raise ValueError("algorithm contract source changed during execution")
    capabilities = artifact.get("capability_keys")
    if not isinstance(capabilities, list) or set(capabilities) != EXPECTED_CAPABILITIES:
        raise ValueError("algorithm contract capability set mismatch")
    rows = artifact.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("algorithm contract rows are missing")
    positive = set()
    hostile = set()
    for row_id, row in rows.items():
        if not isinstance(row_id, str) or not isinstance(row, dict):
            raise ValueError("algorithm contract row is malformed")
        if row.get("passed") is not True:
            raise ValueError(f"algorithm contract row failed: {row_id}")
        capability = row.get("capability_key")
        if capability is None:
            hostile.add(row_id)
        elif capability in EXPECTED_CAPABILITIES:
            positive.add(capability)
        else:
            raise ValueError(f"algorithm contract row has unknown capability: {row_id}")
        implementations = row.get("implementation_files_sha256")
        if not isinstance(implementations, dict) or not implementations:
            raise ValueError(f"algorithm contract implementation binding missing: {row_id}")
        for relative, digest in implementations.items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"algorithm contract implementation drift: {row_id}")
        test_file = row.get("test_file")
        if (
            not isinstance(test_file, str)
            or not (ROOT / test_file).is_file()
            or _sha256(ROOT / test_file) != row.get("test_file_sha256")
            or not isinstance(row.get("test_node"), str)
            or not row["test_node"].startswith(test_file + "::")
        ):
            raise ValueError(f"algorithm contract test binding drift: {row_id}")
        for field in ("fixture_identity", "expected", "observed"):
            if field not in row:
                raise ValueError(f"algorithm contract row lacks {field}: {row_id}")
    if positive != EXPECTED_CAPABILITIES or hostile != HOSTILE_ROW_IDS:
        raise ValueError("algorithm contract positive or hostile rows are incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_git_sha": expected_source_snapshot["git_head"],
        "source_inventory_aggregate_sha256": expected_source_snapshot["files_aggregate_sha256"],
        "capability_keys": sorted(EXPECTED_CAPABILITIES),
        "rows": rows,
    }
