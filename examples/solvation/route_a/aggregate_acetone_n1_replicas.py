"""Validate and aggregate independent acetone + one-water replicas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maple.function.dispatcher.solvfe.analysis import (  # noqa: E402
    diagnose_independent_replicas,
)
from maple.function.dispatcher.solvfe.protocol import (  # noqa: E402
    RouteAProtocol,
    canonical_sha256,
    raw_sha256,
)
from maple.function.dispatcher.solvfe.sampling import (  # noqa: E402
    AlchemicalSampleSet,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_replica(path: Path, protocol_hash: str) -> dict[str, Any]:
    root = path.resolve()
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (root / "run-request.json").read_text(encoding="utf-8")
    )
    if manifest.get("artifact_type") != (
        "route-a-acetone-n1-independent-replica"
    ):
        raise ValueError(f"Unexpected replica artifact at {root}.")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError(f"Replica protocol mismatch at {root}.")
    request_preimage = {
        key: value
        for key, value in request.items()
        if key not in {"schema_version", "run_hash"}
    }
    if canonical_sha256(request_preimage) != request.get("run_hash"):
        raise ValueError(f"Replica request hash mismatch at {root}.")
    if manifest.get("run_hash") != request.get("run_hash"):
        raise ValueError(f"Replica manifest/request mismatch at {root}.")
    source_manifest = json.loads(
        (root / "runtime-source" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    declared_source_manifest_hash = source_manifest.pop(
        "artifact_sha256",
        None,
    )
    if (
        declared_source_manifest_hash != canonical_sha256(source_manifest)
        or source_manifest.get("run_hash") != request.get("run_hash")
    ):
        raise ValueError(f"Replica source snapshot manifest mismatch at {root}.")
    source_files = source_manifest.get("files")
    if not isinstance(source_files, dict):
        raise ValueError(f"Replica source snapshot is incomplete at {root}.")
    if {
        relative: record.get("sha256")
        for relative, record in source_files.items()
    } != request.get("runtime_source_sha256"):
        raise ValueError(f"Replica source hash map mismatch at {root}.")
    for relative, record in source_files.items():
        source_path = root / "runtime-source" / relative
        if (
            not source_path.is_file()
            or raw_sha256(source_path) != record.get("sha256")
        ):
            raise ValueError(f"Replica source snapshot drift at {root}.")
    samples = AlchemicalSampleSet.load(root / "samples")
    if samples.content_hash != manifest.get("sample_content_hash"):
        raise ValueError(f"Replica sample hash mismatch at {root}.")
    if manifest.get("scientific_status") != "fixed-n-development-diagnostic":
        raise ValueError(f"Replica scientific status mismatch at {root}.")
    return {
        "path": root.as_posix(),
        "manifest": manifest,
        "request": request,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "replicas",
        type=Path,
        nargs="+",
        help="Completed independent replica directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol = RouteAProtocol.load(args.protocol)
    replicas = [
        _load_replica(path, protocol.content_hash) for path in args.replicas
    ]
    manifests = [item["manifest"] for item in replicas]
    thresholds = protocol.data["thresholds"]

    invariant_fields = (
        "schedule_hash",
        "move_config_hash",
    )
    for field in invariant_fields:
        observed = {manifest[field] for manifest in manifests}
        if len(observed) != 1:
            raise ValueError(f"Replica invariant '{field}' differs.")
    model_hashes = {
        manifest["model_provenance"]["checkpoint_sha256"]
        for manifest in manifests
    }
    if len(model_hashes) != 1:
        raise ValueError("Replica model checkpoint hashes differ.")
    for field in (
        "initial_samples_sha256",
        "initial_frame_sha256",
        "runtime_environment",
        "runtime_source_sha256",
    ):
        observed = {
            canonical_sha256(item["request"][field]) for item in replicas
        }
        if len(observed) != 1:
            raise ValueError(f"Replica request field '{field}' differs.")
    nonseed_configs = []
    for manifest in manifests:
        config = dict(manifest["run_config"])
        config.pop("seed")
        nonseed_configs.append(canonical_sha256(config))
    if len(set(nonseed_configs)) != 1:
        raise ValueError("Replica run configurations differ beyond seed.")

    diagnostics = diagnose_independent_replicas(
        estimates_kcal_mol=tuple(
            float(manifest["mbar_delta_g_D_to_P_kcal_mol"])
            for manifest in manifests
        ),
        standard_errors_kcal_mol=tuple(
            float(manifest["mbar_uncertainty_kcal_mol"])
            for manifest in manifests
        ),
        replica_ids=tuple(
            str(manifest["run_hash"]) for manifest in manifests
        ),
        seeds=tuple(
            int(manifest["run_config"]["seed"]) for manifest in manifests
        ),
        per_replica_status=tuple(
            str(manifest["scientific_gate_status"])
            for manifest in manifests
        ),
        minimum_replicas=int(
            thresholds["replica_min_count"]["value"]
        ),
        pairwise_z_max=float(
            thresholds["replica_pairwise_z_max"]["value"]
        ),
    )
    result = {
        "schema_version": 1,
        "artifact_type": "route-a-acetone-n1-replica-aggregate",
        "scientific_status": "fixed-n-development-diagnostic",
        "protocol_sha256": protocol.content_hash,
        "replica_paths": [item["path"] for item in replicas],
        "replica_run_hashes": [
            manifest["run_hash"] for manifest in manifests
        ],
        "replica_sample_hashes": [
            manifest["sample_content_hash"] for manifest in manifests
        ],
        "diagnostics": diagnostics,
        "aggregator_source_sha256": raw_sha256(Path(__file__)),
        "warning": (
            "This aggregate is only the fixed-n acetone development edge. "
            "It omits p0 and the conditioned multi-n QCT cycle."
        ),
    }
    result["artifact_sha256"] = canonical_sha256(result)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
