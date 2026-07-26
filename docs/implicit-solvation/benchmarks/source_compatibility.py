"""Audit documented post-execution source changes without rewriting evidence.

The compatibility record explains why historical evidence hashes differ from
the current safety-hardened source. It never authorizes execution or sealing
under a protocol whose implementation freeze no longer matches byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_core import artifact_content_sha256, load_json, sha256_file


COMPATIBILITY_ARTIFACT = (
    Path(__file__).resolve().parent
    / "route1-production-safety-source-compatibility-2026-07-27.json"
)


def load_source_compatibility() -> dict[str, Any]:
    artifact = load_json(COMPATIBILITY_ARTIFACT)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type")
        != "route1-postexecution-production-safety-source-compatibility"
        or artifact.get("content_sha256") != artifact_content_sha256(artifact)
        or artifact.get("historical_evidence_bytes_unchanged") is not True
        or artifact.get("frozen_results_recomputed") is not False
        or artifact.get("scientific_claim_promoted") is not False
    ):
        raise ValueError("Invalid Route 1 production-safety compatibility artifact.")
    return artifact


def validate_frozen_source(
    repository_root: str | Path,
    relative_path: str,
    historical_sha256: str,
) -> dict[str, Any]:
    """Audit an exact freeze or one documented post-execution source change."""
    root = Path(repository_root).resolve()
    source_path = (root / relative_path).resolve()
    try:
        normalized = source_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Frozen source escapes the repository: {relative_path}."
        ) from exc
    if normalized != relative_path or not source_path.is_file():
        raise ValueError(f"Frozen source path is invalid: {relative_path}.")
    observed_sha256 = sha256_file(source_path)
    if observed_sha256 == historical_sha256:
        return {
            "mode": "exact-historical-freeze",
            "path": relative_path,
            "historical_sha256": historical_sha256,
            "observed_sha256": observed_sha256,
        }

    artifact = load_source_compatibility()
    matches = [
        record
        for record in artifact.get("source_changes", [])
        if record.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"Frozen implementation changed: {relative_path}.")
    record = matches[0]
    historical_hashes = record.get("historical_sha256", [])
    if (
        not isinstance(historical_hashes, list)
        or historical_sha256 not in historical_hashes
        or observed_sha256 != record.get("current_sha256")
        or record.get("historical_numerical_results_recomputed") is not False
        or record.get("scientific_claim_reuse_authorized") is not False
    ):
        raise ValueError(f"Frozen implementation changed: {relative_path}.")
    return {
        "mode": "documented-postexecution-production-safety-change",
        "path": relative_path,
        "historical_sha256": historical_sha256,
        "observed_sha256": observed_sha256,
        "change_class": record.get("change_class"),
    }


def require_exact_frozen_sources(
    repository_root: str | Path,
    source_records: list[dict[str, Any]],
) -> None:
    """Require byte-exact protocol sources before execution or sealing."""
    root = Path(repository_root).resolve()
    for record in source_records:
        relative_path = record["path"]
        historical_sha256 = record["sha256"]
        audit = validate_frozen_source(
            root,
            relative_path,
            historical_sha256,
        )
        if audit["mode"] != "exact-historical-freeze":
            raise ValueError(
                "Frozen implementation changed and the compatibility record is "
                "historical-audit-only; execution, resume, and sealing remain "
                f"unauthorized: {relative_path}."
            )
