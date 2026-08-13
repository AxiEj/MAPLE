#!/usr/bin/env python3
"""Independently recompute the residual-refinement force-error panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
    ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    PRIMAL_REFINEMENT_TOLERANCES,
    RESIDUAL_FORCE_CONTRACT_VERSION,
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    runtime_record,
    summarize_residual_force_panel,
    summarize_residual_force_refinement,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file

SCHEMA_VERSION = "route2-fixedbox590-residual-force-panel-aggregate-v1"
SHARD_SCHEMA_VERSION = "route2-fixedbox590-residual-force-panel-shard-v1"
CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _runtime_signature(payload: dict[str, object]) -> str:
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


def _validate_contract(contract: object, path: Path) -> None:
    if not isinstance(contract, dict) or (
        contract.get("contract_version") != RESIDUAL_FORCE_CONTRACT_VERSION
        or contract.get("asset_sha256") != PES_PANEL_ASSET_SHA256
        or contract.get("variant") != PES_CARTESIAN_PANEL_VARIANT
        or tuple(contract.get("primal_tolerances", ()))
        != PRIMAL_REFINEMENT_TOLERANCES
        or tuple(contract.get("adjoint_relative_tolerances", ()))
        != ADJOINT_REFINEMENT_RELATIVE_TOLERANCES
        or tuple(contract.get("adjoint_absolute_tolerances", ()))
        != ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES
    ):
        raise ValueError(f"Unexpected residual-force shard contract: {path}.")


def _load_shards(paths):
    payloads = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SHARD_SCHEMA_VERSION
            or payload.get("capabilities") != CAPABILITIES
        ):
            raise ValueError(f"Unexpected residual-force shard schema: {path}.")
        contract = payload.get("panel_contract")
        _validate_contract(contract, path)
        expected = canonical_json_sha256(
            {"contract": contract, "measurements": payload.get("measurements")}
        )
        if payload.get("measurement_sha256") != expected:
            raise ValueError(f"Residual-force shard digest is invalid: {path}.")
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
            "All residual-force shards must share one source tree, checkpoint, "
            "and numerical runtime signature."
        )
    return payloads


def _raw_record(raw, molecule):
    atoms = panel_geometries(molecule)[PES_CARTESIAN_PANEL_VARIANT]
    if (
        raw.get("geometry_sha256") != geometry_sha256(atoms)
        or raw.get("atomic_numbers") != atoms.numbers.tolist()
        or not np.array_equal(
            np.asarray(raw.get("positions_A"), dtype=float), atoms.positions
        )
    ):
        raise ValueError("Residual-force geometry differs from the frozen asset.")
    topology_hash = str(raw.get("topology_hash"))
    if len(topology_hash) != 64 or any(
        character not in "0123456789abcdef" for character in topology_hash
    ):
        raise ValueError("Residual-force topology hash is invalid.")
    recomputed = summarize_residual_force_refinement(
        raw.get("primal_levels", ()), raw.get("adjoint_levels", ())
    )
    return {
        "molecule_id": molecule.molecule_id,
        "geometry_sha256": geometry_sha256(atoms),
        "topology_hash": topology_hash,
        **recomputed,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    shards = _load_shards(args.shard)
    molecules = {item.molecule_id: item for item in load_pes_panel()}
    records = []
    source_hashes = None
    artifact_records = []
    for path, payload in shards:
        artifact_records.append(
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
        current_hashes = payload.get("source_files_sha256")
        if source_hashes is None:
            source_hashes = current_hashes
        elif current_hashes != source_hashes:
            raise ValueError("Residual-force shards have different source ledgers.")
        for raw in payload.get("measurements", ()):
            molecule = molecules.get(str(raw.get("molecule_id")))
            if molecule is None:
                raise ValueError("Residual-force shard contains an unknown molecule.")
            records.append(_raw_record(raw, molecule))

    expected_ids = tuple(item.molecule_id for item in load_pes_panel())
    summary = summarize_residual_force_panel(records, expected_ids)
    first = shards[0][1]
    if (
        repository.head != first["execution_git_head"]
        or repository.tree != first["execution_git_tree"]
    ):
        raise RuntimeError(
            "Residual-force aggregator checkout must be the exact shard tree."
        )
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Residual-force source-file hash ledger is missing.")
    if committed_source_hashes(repository, source_hashes) != source_hashes:
        raise RuntimeError("Residual-force shard sources do not match the Git tree.")
    verifier_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            "tools/route2_release/aggregate_fixedbox590_residual_force_panel.py",
            "maple/solvation/release/residual_force.py",
            "maple/solvation/release/pes_panel.py",
        ),
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if summary["all_gates_passed"] else "fail",
        "claim_boundary": (
            "Independent raw-value recomputation of the 20-molecule empirical "
            "residual-refinement force-error estimate; not a rigorous bound, "
            "Tier admission, complete solvation free energy, or accuracy evidence."
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
        "input_artifacts": artifact_records,
        "residual_force_panel_summary": summary,
    }
    output["aggregate_measurement_sha256"] = canonical_json_sha256(records)
    target = args.output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate artifact: {target}")
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, target, output)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_RESIDUAL_FORCE_AGGREGATE="
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
