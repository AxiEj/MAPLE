#!/usr/bin/env python3
"""Verify and aggregate independent Route A bulk-water NPT replicas."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maple.function.dispatcher.solvfe.bulk_water import (
    BulkWaterValidationError,
)
from maple.function.dispatcher.solvfe.bulk_water_campaign import (
    BulkWaterCampaignConfig,
    evaluate_bulk_water_replica_campaign,
)
from maple.function.dispatcher.solvfe.protocol import canonical_sha256
from maple.function.dispatcher.solvfe.provenance import (
    collect_implementation_provenance,
)


IMPLEMENTATION_PATHS = {
    "maple/function/dispatcher/solvfe/bulk_water.py",
    "maple/function/dispatcher/solvfe/bulk_water_campaign.py",
    "maple/function/dispatcher/solvfe/bulk_water_evidence.py",
    "maple/function/dispatcher/solvfe/bulk_water_npt.py",
    "maple/function/dispatcher/solvfe/protocol.py",
    "maple/function/dispatcher/solvfe/provenance.py",
    "examples/solvation/route_a/aggregate_bulk_water_npt_replicas.py",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-replicas", type=int, default=3)
    parser.add_argument(
        "--density-relative-error-limit",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--replica-density-spread-limit",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "replicas",
        type=Path,
        nargs="+",
        help="Hash-bound NPT artifact directories.",
    )
    return parser


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BulkWaterValidationError(
                f"OUTPUT_EXISTS: refusing to overwrite {path}"
            ) from exc
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _campaign_artifact(
    campaign: dict,
    *,
    replica_paths: Sequence[Path],
) -> dict:
    campaign_hash = campaign.get("campaign_hash")
    campaign_preimage = dict(campaign)
    campaign_preimage.pop("campaign_hash", None)
    if (
        not isinstance(campaign_hash, str)
        or canonical_sha256(campaign_preimage) != campaign_hash
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_HASH_MISMATCH: aggregate changed before publication."
        )
    result_hashes = campaign.get("replica_result_hashes")
    if (
        not isinstance(result_hashes, list)
        or len(result_hashes) != len(replica_paths)
        or not all(
            isinstance(result_hash, str)
            and len(result_hash) == 64
            for result_hash in result_hashes
        )
    ):
        raise BulkWaterValidationError(
            "CAMPAIGN_HASH_MISMATCH: aggregate does not bind every replica."
        )
    artifact = {
        "schema": "maple-route-a-bulk-water-campaign-artifact-v1",
        "campaign": campaign,
        "replicas": [
            {
                "path": path.as_posix(),
                "result_hash": result_hash,
            }
            for path, result_hash in zip(
                replica_paths,
                result_hashes,
                strict=True,
            )
        ],
        "implementation": collect_implementation_provenance(
            PROJECT_ROOT,
            IMPLEMENTATION_PATHS,
            schema=(
                "maple-route-a-bulk-water-campaign-implementation-v1"
            ),
        ),
        "interpretation": (
            "This artifact can pass only the independent NPT density campaign. "
            "Finite-size, external RDF, cross-engine, Hamiltonian-freeze and "
            "Route A accuracy gates remain outside this artifact and false."
        ),
    }
    artifact["artifact_hash"] = canonical_sha256(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise BulkWaterValidationError(
            f"OUTPUT_EXISTS: refusing to overwrite {output}"
        )
    replica_paths = [
        path.expanduser().resolve() for path in args.replicas
    ]
    if len(set(replica_paths)) != len(replica_paths):
        raise BulkWaterValidationError(
            "CAMPAIGN_ARTIFACT_REUSE: replica paths must be distinct."
        )
    campaign = evaluate_bulk_water_replica_campaign(
        replica_paths,
        config=BulkWaterCampaignConfig(
            required_replicas=args.required_replicas,
            density_relative_error_limit=(
                args.density_relative_error_limit
            ),
            replica_density_spread_limit=(
                args.replica_density_spread_limit
            ),
        ),
    )
    artifact = _campaign_artifact(
        campaign,
        replica_paths=replica_paths,
    )
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return (
        0
        if campaign["gates"]["npt_density_accuracy_passed"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
