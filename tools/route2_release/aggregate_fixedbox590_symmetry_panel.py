#!/usr/bin/env python3
"""Independently recompute the frozen 20-molecule symmetry/loop gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    SYMMETRY_PANEL_CONTRACT_VERSION,
    SYMMETRY_PANEL_LOOP_AMPLITUDES_A,
    SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
    SYMMETRY_PANEL_ROTATION_COUNT,
    SYMMETRY_PANEL_TRANSLATION_A,
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    runtime_record,
    summarize_bidirectional_loop_record,
    summarize_rigid_symmetry,
    summarize_symmetry_panel,
    symmetry_panel_rotations,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file

SCHEMA_VERSION = "route2-fixedbox590-symmetry-panel-aggregate-v1"
SHARD_SCHEMA_VERSION = "route2-fixedbox590-symmetry-panel-shard-v1"
CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _runtime_signature(payload):
    runtime = payload.get("runtime", {})
    return canonical_json_sha256(
        {
            "device": payload.get("device"),
            "dtype": payload.get("dtype"),
            "platform": runtime.get("platform"),
            "machine": runtime.get("machine"),
            "packages": runtime.get("packages"),
            "numpy_version": runtime.get("numpy", {}).get("version"),
            "torch": runtime.get("torch"),
            "thread_environment": runtime.get("environment"),
            "numerical_determinism": payload.get("numerical_determinism"),
        }
    )


def _load_shards(paths):
    payloads = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text())
        contract = payload.get("panel_contract")
        if (
            payload.get("schema_version") != SHARD_SCHEMA_VERSION
            or payload.get("capabilities") != CAPABILITIES
            or not isinstance(contract, dict)
            or contract.get("contract_version") != SYMMETRY_PANEL_CONTRACT_VERSION
            or contract.get("asset_sha256") != PES_PANEL_ASSET_SHA256
            or contract.get("variant") != PES_CARTESIAN_PANEL_VARIANT
            or contract.get("rotation_count") != SYMMETRY_PANEL_ROTATION_COUNT
            or tuple(contract.get("translation_A", ()))
            != SYMMETRY_PANEL_TRANSLATION_A
            or tuple(contract.get("loop_amplitudes_A", ()))
            != SYMMETRY_PANEL_LOOP_AMPLITUDES_A
            or contract.get("loop_subdivisions_per_edge")
            != SYMMETRY_PANEL_LOOP_SUBDIVISIONS
        ):
            raise ValueError(f"Unexpected symmetry shard contract: {path}.")
        expected = canonical_json_sha256(
            {"contract": contract, "measurements": payload.get("measurements")}
        )
        if payload.get("measurement_sha256") != expected:
            raise ValueError(f"Symmetry shard measurement digest is invalid: {path}.")
        payloads.append((path, payload))
    if any(
        len(values) != 1
        for values in (
            {payload["execution_git_head"] for _, payload in payloads},
            {payload["execution_git_tree"] for _, payload in payloads},
            {payload["checkpoint"]["sha256"] for _, payload in payloads},
            {_runtime_signature(payload) for _, payload in payloads},
        )
    ):
        raise ValueError(
            "All symmetry shards must share one source tree, checkpoint, and runtime."
        )
    return payloads


def _raw_record(raw, molecule):
    atoms = panel_geometries(molecule)[PES_CARTESIAN_PANEL_VARIANT]
    if (
        raw.get("geometry_sha256") != geometry_sha256(atoms)
        or raw.get("atomic_numbers") != atoms.numbers.tolist()
        or not np.array_equal(np.asarray(raw.get("positions_A")), atoms.positions)
    ):
        raise ValueError("Symmetry base geometry differs from the frozen asset.")
    base = raw.get("base")
    rigid = raw.get("rigid_symmetry")
    loop = raw.get("closed_loop")
    if not all(isinstance(value, dict) for value in (base, rigid, loop)):
        raise ValueError("Symmetry raw measurement sections are missing.")
    raw_rotations = rigid.get("rotations")
    expected_rotations = symmetry_panel_rotations(molecule.molecule_id)
    if not isinstance(raw_rotations, list) or len(raw_rotations) != len(
        expected_rotations
    ):
        raise ValueError("Symmetry rotation coverage is incomplete.")
    # Runner summaries intentionally omit large raw rotated arrays. They are
    # retained in `_raw_rigid_records` exclusively for independent recomputation.
    private = raw.get("_raw_rigid_records")
    if not isinstance(private, dict):
        raise ValueError("Symmetry raw rigid arrays are missing.")
    translation = private.get("translation")
    raw_rotation_arrays = private.get("rotations")
    if not isinstance(translation, dict) or not isinstance(raw_rotation_arrays, list):
        raise ValueError("Symmetry raw rigid array records are invalid.")
    recomputed_rigid = summarize_rigid_symmetry(
        positions_A=atoms.positions,
        base_energy_eV=base["energy_eV"],
        base_forces_eV_per_A=base["forces_eV_per_A"],
        base_source=base["source"],
        base_topology_hash=base["topology_hash"],
        translation_record=translation,
        permutation_record=private.get("permutation"),
        rotation_records=[
            {
                **item,
                "expected_rotation_matrix": expected.tolist(),
            }
            for item, expected in zip(
                raw_rotation_arrays, expected_rotations, strict=True
            )
        ],
    )
    recomputed_loop = summarize_bidirectional_loop_record(loop)
    return {
        "molecule_id": molecule.molecule_id,
        "contract_version": SYMMETRY_PANEL_CONTRACT_VERSION,
        "geometry_sha256": geometry_sha256(atoms),
        "rigid_symmetry": recomputed_rigid,
        "closed_loop": recomputed_loop,
    }


def main():
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    shards = _load_shards(args.shard)
    molecules = {item.molecule_id: item for item in load_pes_panel()}
    records = []
    source_hashes = None
    artifacts = []
    for path, payload in shards:
        artifacts.append(
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
        current = payload.get("source_files_sha256")
        if source_hashes is None:
            source_hashes = current
        elif current != source_hashes:
            raise ValueError("Symmetry shards have different source ledgers.")
        for raw in payload.get("measurements", ()):
            molecule = molecules.get(str(raw.get("molecule_id")))
            if molecule is None:
                raise ValueError("Symmetry shard contains an unknown molecule.")
            records.append(_raw_record(raw, molecule))
    summary = summarize_symmetry_panel(records)
    first = shards[0][1]
    if (
        repository.head != first["execution_git_head"]
        or repository.tree != first["execution_git_tree"]
    ):
        raise RuntimeError("Symmetry aggregator must use the exact shard source tree.")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Symmetry source ledger is missing.")
    if committed_source_hashes(repository, source_hashes) != source_hashes:
        raise RuntimeError("Symmetry shard sources do not match the Git tree.")
    verifier_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
            "maple/solvation/release/symmetry_panel.py",
            "maple/solvation/release/pes_validation.py",
        ),
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if summary["all_gates_passed"] else "fail",
        "claim_boundary": (
            "Independent raw-value recomputation of the 20-molecule rigid-symmetry "
            "and bidirectional cold/warm closed-loop panel; not Tier admission, "
            "complete solvation free energy, chemical accuracy, H/FREQ/TS/HVP/NVE/MD."
        ),
        "capabilities": CAPABILITIES,
        "execution_git_head": first["execution_git_head"],
        "execution_git_tree": first["execution_git_tree"],
        "exact_command": shlex.join(sys.argv),
        "checkpoint_sha256": first["checkpoint"]["sha256"],
        "source_files_sha256": source_hashes,
        "verifier_source_files_sha256": committed_source_hashes(
            repository, verifier_paths
        ),
        "verifier_runtime": runtime_record(),
        "input_artifacts": artifacts,
        "symmetry_panel_summary": summary,
    }
    output["aggregate_measurement_sha256"] = canonical_json_sha256(records)
    target = args.output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate artifact: {target}")
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, target, output)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_SYMMETRY_PANEL_AGGREGATE="
        + json.dumps(
            {
                "path": str(target),
                "sha256": file_record["sha256"],
                "status": output["status"],
                "capabilities": CAPABILITIES,
            },
            sort_keys=True,
        )
    )
    if not summary["all_gates_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
