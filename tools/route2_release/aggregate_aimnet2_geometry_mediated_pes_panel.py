#!/usr/bin/env python3
"""Independently aggregate all AIMNet2 geometry-mediated PES shards.

The verifier accepts one exact continuum-specific seventeen-shard contract,
recomputes every shard and panel decision from raw records, and writes a
fail-closed aggregate outside the checkout.  A negative aggregate is retained
as scientific evidence and exits with status 2; it never admits a capability.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.solvation.release import (
    AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    RepositorySnapshot,
    aimnet2_geometry_mediated_pes_continuum_contract,
    aimnet2_geometry_mediated_pes_molecule,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    summarize_aimnet2_geometry_mediated_pes_panel,
    summarize_aimnet2_geometry_mediated_pes_shard,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file
from maple.solvation.models.aimnet2 import (
    AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
    AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
)
from tools.route2_release.aimnet2_geometry_mediated_common import (
    geometry_mediated_continuum_protocol,
)

RUNTIME_KIND = "reconstructed-python-float64"
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
_INVARIANT_IDENTITY_KEYS = (
    "scalar_id",
    "profile_id",
    "model_provider_id",
    "model_configuration_sha256",
    "model_provenance_sha256",
    "continuum_provider_id",
    "source_space_sha256",
    "field_space_sha256",
    "pairing_sha256",
    "model_runtime",
)


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    path: Path
    payload: dict[str, object]
    summary: dict[str, object]
    runtime_signature: str
    protocol_signature: str
    identity_signature: str
    source_hash_signature: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the exact 17-shard AIMNet2 geometry-mediated PES panel."
        )
    )
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument(
        "--continuum",
        choices=("harmonic-point", "harmonic-ddpcm-water"),
        default="harmonic-point",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _measurement(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "contract_version": payload.get("contract_version"),
        "panel_asset_sha256": payload.get("panel_asset_sha256"),
        "protocol": payload.get("protocol"),
        "identity": payload.get("identity"),
        "records": payload.get("records"),
        "summary": payload.get("summary"),
    }


def _runtime_signature(payload: Mapping[str, object]) -> str:
    runtime = _mapping(payload.get("runtime"), name="runtime record")
    numpy_record = _mapping(runtime.get("numpy"), name="NumPy runtime record")
    return canonical_json_sha256(
        {
            "device": payload.get("device"),
            "dtype": payload.get("dtype"),
            "implementation": runtime.get("implementation"),
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "machine": runtime.get("machine"),
            "cpu_model": runtime.get("cpu_model"),
            "packages": runtime.get("packages"),
            "numpy_version": numpy_record.get("version"),
            "numpy_show_config": numpy_record.get("show_config"),
            "torch": runtime.get("torch"),
            "thread_environment": runtime.get("environment"),
        }
    )


def _protocol_signature(
    payload: Mapping[str, object],
    *,
    molecule_index: int,
    continuum_kind: str,
) -> str:
    protocol = dict(_mapping(payload.get("protocol"), name="shard protocol"))
    if (
        protocol.get("aimnet_runtime") != RUNTIME_KIND
        or protocol.get("continuum_kind") != continuum_kind
        or tuple(protocol.get("directional_steps_A", ()))
        != PES_PANEL_DIRECTIONAL_STEPS_A
    ):
        raise ValueError("PES shard runtime, continuum, or step protocol changed.")
    continuum = dict(_mapping(protocol.get("continuum"), name="continuum protocol"))
    molecule = aimnet2_geometry_mediated_pes_molecule(molecule_index)
    expected_continuum = geometry_mediated_continuum_protocol(
        molecule.atoms, continuum_kind
    )
    if continuum != expected_continuum:
        raise ValueError("Continuum protocol changed from the frozen contract.")
    continuum.pop("radii_A")
    protocol["continuum"] = continuum
    return canonical_json_sha256(protocol)


def _identity_signature(
    payload: Mapping[str, object], *, records: Sequence[Mapping[str, object]]
) -> str:
    identity = _mapping(payload.get("identity"), name="shard identity")
    if any(key not in identity for key in _INVARIANT_IDENTITY_KEYS):
        raise ValueError("PES shard identity is incomplete.")
    model_configuration = identity.get("model_configuration_sha256")
    continuum_configuration = identity.get("continuum_configuration_sha256")
    for record in records:
        model_topology = _mapping(
            record.get("center_model_topology"), name="model topology"
        )
        continuum_topology = _mapping(
            record.get("center_continuum_topology"), name="continuum topology"
        )
        if model_topology.get("configuration_sha256") != model_configuration:
            raise ValueError("Model identity and topology configuration disagree.")
        if continuum_topology.get("configuration_sha256") != continuum_configuration:
            raise ValueError("Continuum identity and topology configuration disagree.")
    return canonical_json_sha256(
        {key: identity[key] for key in _INVARIANT_IDENTITY_KEYS}
    )


def _load_shards(
    paths: Sequence[Path],
    *,
    continuum_kind: str = "harmonic-point",
) -> tuple[_LoadedShard, ...]:
    continuum_contract = aimnet2_geometry_mediated_pes_continuum_contract(
        continuum_kind
    )
    loaded: list[_LoadedShard] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"PES shard artifact must be a mapping: {path}.")
        if (
            payload.get("schema_version")
            != continuum_contract.shard_artifact_schema_version
            or payload.get("artifact_kind") != continuum_contract.shard_artifact_kind
            or payload.get("capabilities") != NO_CAPABILITIES
            or payload.get("working_tree_clean") is not True
            or payload.get("contract_version")
            != continuum_contract.shard_contract_version
            or payload.get("panel_asset_sha256") != PES_PANEL_ASSET_SHA256
            or payload.get("aimnet_runtime") != RUNTIME_KIND
            or payload.get("continuum_kind") != continuum_kind
            or payload.get("device") != "cpu"
            or payload.get("dtype") != "float64"
        ):
            raise ValueError(f"Unexpected PES shard artifact contract: {path}.")
        expected_measurement = canonical_json_sha256(_measurement(payload))
        if payload.get("measurement_sha256") != expected_measurement:
            raise ValueError(f"PES shard measurement digest is invalid: {path}.")

        cached_summary = _mapping(payload.get("summary"), name="cached shard summary")
        raw_index = cached_summary.get("molecule_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError("PES shard molecule index must be an integer.")
        records = _sequence(payload.get("records"), name="raw shard records")
        normalized_records = tuple(
            _mapping(record, name="raw geometry record") for record in records
        )
        recomputed = summarize_aimnet2_geometry_mediated_pes_shard(
            molecule_index=raw_index,
            records=normalized_records,
            continuum_kind=continuum_kind,
        )
        if canonical_json_sha256(cached_summary) != canonical_json_sha256(recomputed):
            raise ValueError(f"PES shard cached summary is not reproducible: {path}.")
        expected_status = (
            "diagnostic-gates-passed-not-admitted"
            if recomputed["diagnostic_gates_passed"]
            else "diagnostic-gates-failed-not-admitted"
        )
        if payload.get("status") != expected_status:
            raise ValueError(f"PES shard status disagrees with raw records: {path}.")
        checkpoint = _mapping(payload.get("checkpoint"), name="checkpoint record")
        if (
            checkpoint.get("sha256") != AIMNET2_WB97M_D3_CHECKPOINT_SHA256
            or checkpoint.get("bytes") != AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES
        ):
            raise ValueError("PES shard checkpoint differs from the frozen contract.")
        loaded.append(
            _LoadedShard(
                path=path,
                payload=payload,
                summary=recomputed,
                runtime_signature=_runtime_signature(payload),
                protocol_signature=_protocol_signature(
                    payload,
                    molecule_index=raw_index,
                    continuum_kind=continuum_kind,
                ),
                identity_signature=_identity_signature(
                    payload,
                    records=normalized_records,
                ),
                source_hash_signature=canonical_json_sha256(
                    payload.get("source_files_sha256")
                ),
            )
        )

    if not loaded:
        raise ValueError("At least one PES shard artifact is required.")
    invariant_sets = {
        "execution Git head": {
            item.payload.get("execution_git_head") for item in loaded
        },
        "execution Git tree": {
            item.payload.get("execution_git_tree") for item in loaded
        },
        "checkpoint": {
            _mapping(item.payload.get("checkpoint"), name="checkpoint record").get(
                "sha256"
            )
            for item in loaded
        },
        "runtime": {item.runtime_signature for item in loaded},
        "protocol": {item.protocol_signature for item in loaded},
        "identity": {item.identity_signature for item in loaded},
        "source hash ledger": {item.source_hash_signature for item in loaded},
    }
    changed = [name for name, values in invariant_sets.items() if len(values) != 1]
    if changed:
        raise ValueError("All PES shards must share one " + ", ".join(changed) + ".")
    indices = [int(item.summary["molecule_index"]) for item in loaded]
    if len(indices) != len(set(indices)):
        raise ValueError("PES shard molecule indices must be unique.")
    return tuple(sorted(loaded, key=lambda item: int(item.summary["molecule_index"])))


def main() -> None:
    args = _parse_args()
    continuum_contract = aimnet2_geometry_mediated_pes_continuum_contract(
        args.continuum
    )
    repository = RepositorySnapshot.capture(REPOSITORY_ROOT)
    shards = _load_shards(args.shard, continuum_kind=args.continuum)
    expected_count = len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)
    if len(shards) != expected_count:
        raise ValueError(
            f"The AIMNet2 geometry-mediated PES panel requires exactly "
            f"{expected_count} shard artifacts."
        )
    raw_shards = [
        {
            "molecule_index": int(shard.summary["molecule_index"]),
            "records": shard.payload["records"],
        }
        for shard in shards
    ]
    summary = summarize_aimnet2_geometry_mediated_pes_panel(
        raw_shards,
        continuum_kind=args.continuum,
    )
    first = shards[0].payload
    if repository.head != first.get(
        "execution_git_head"
    ) or repository.tree != first.get("execution_git_tree"):
        raise RuntimeError(
            "PES-panel aggregation requires the exact shard execution Git tree."
        )
    source_hashes = first.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("PES shard source-file hash ledger is missing.")
    if committed_source_hashes(repository, source_hashes) != source_hashes:
        raise RuntimeError("PES shard sources do not match the exact Git tree.")

    verifier_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            "tools/route2_release/aggregate_aimnet2_geometry_mediated_pes_panel.py",
            "maple/solvation/release/geometry_mediated_panel.py",
        ),
    )
    input_artifacts = [
        {
            "molecule_index": int(shard.summary["molecule_index"]),
            "molecule_id": str(shard.summary["molecule_id"]),
            "path": str(shard.path),
            "bytes": shard.path.stat().st_size,
            "sha256": sha256_file(shard.path),
            "measurement_sha256": shard.payload["measurement_sha256"],
            "diagnostic_gates_passed": bool(shard.summary["diagnostic_gates_passed"]),
        }
        for shard in shards
    ]
    payload: dict[str, object] = {
        "schema_version": continuum_contract.panel_artifact_schema_version,
        "artifact_kind": continuum_contract.panel_artifact_kind,
        "status": (
            "diagnostic-panel-passed-not-admitted"
            if summary["diagnostic_gates_passed"]
            else "diagnostic-panel-failed-not-admitted"
        ),
        "claim_boundary": (
            "Independent raw-record recomputation of the exact seventeen-shard "
            f"H/C/N/O AIMNet2 geometry-mediated {args.continuum} panel. This is "
            "diagnostic domain evidence, not solvent calibration, chemical "
            "accuracy, fixed-R mutual polarization, global C1/C2 proof, or "
            "E/F/H/V/M, OPT, FREQ/TS/IRC, or MD admission."
        ),
        "capabilities": NO_CAPABILITIES,
        "contract_version": continuum_contract.panel_contract_version,
        "continuum_kind": args.continuum,
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "source_files_sha256": source_hashes,
        "verifier_source_files_sha256": committed_source_hashes(
            repository, verifier_paths
        ),
        "verifier_runtime": runtime_record(),
        "checkpoint": first["checkpoint"],
        "input_artifacts": input_artifacts,
        "aggregate_measurement_sha256": canonical_json_sha256(raw_shards),
        "panel_summary": summary,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_PES_PANEL="
        + json.dumps(
            {
                "artifact": artifact,
                "status": payload["status"],
                "diagnostic_gates_passed": summary["diagnostic_gates_passed"],
                "failed_molecule_ids": summary["failed_molecule_ids"],
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )
    if not summary["diagnostic_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
