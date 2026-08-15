#!/usr/bin/env python3
"""Independently recompute the frozen full-Cartesian PES-panel gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    PES_CARTESIAN_PANEL_CONTRACT_VERSION,
    PES_CARTESIAN_PANEL_STEPS_A,
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    runtime_record,
    summarize_cartesian_force_differences,
    summarize_cartesian_pes_panel,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file

from aggregate_fixedbox590_pes_panel import _extended_record, _root_record

SCHEMA_VERSION = "route2-fixedbox590-cartesian-panel-aggregate-v1"
SHARD_SCHEMA_VERSION = "route2-fixedbox590-cartesian-panel-shard-v1"
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


def _load_shards(
    paths,
    *,
    shard_schema_version=SHARD_SCHEMA_VERSION,
    contract_version=PES_CARTESIAN_PANEL_CONTRACT_VERSION,
    expected_profile_id=None,
    expected_scalar_id=None,
):
    payloads = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload.get("panel_contract")
        if (
            payload.get("schema_version") != shard_schema_version
            or payload.get("capabilities") != CAPABILITIES
            or not isinstance(contract, dict)
            or contract.get("contract_version") != contract_version
            or contract.get("asset_sha256") != PES_PANEL_ASSET_SHA256
            or contract.get("variant") != PES_CARTESIAN_PANEL_VARIANT
            or tuple(contract.get("cartesian_steps_A", ()))
            != PES_CARTESIAN_PANEL_STEPS_A
        ):
            raise ValueError(f"Unexpected Cartesian-panel shard contract: {path}.")
        identities = payload.get("identities")
        if not isinstance(identities, dict):
            raise ValueError(f"Cartesian shard identity binding is missing: {path}.")
        if expected_profile_id is not None and (
            contract.get("profile_id") != expected_profile_id
            or identities.get("profile_id") != expected_profile_id
        ):
            raise ValueError(f"Cartesian shard uses a different profile: {path}.")
        if expected_scalar_id is not None and (
            contract.get("scalar_id") != expected_scalar_id
            or identities.get("scalar_id") != expected_scalar_id
        ):
            raise ValueError(f"Cartesian shard uses a different scalar: {path}.")
        expected = canonical_json_sha256(
            {
                "contract": contract,
                "measurements": payload.get("measurements"),
            }
        )
        if payload.get("measurement_sha256") != expected:
            raise ValueError(f"Cartesian shard measurement digest is invalid: {path}.")
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
            "All Cartesian shards must share one source tree, checkpoint, and "
            "numerical runtime signature."
        )
    return payloads


def _raw_cartesian(record, molecule, *, require_extended_gates=False):
    atoms = panel_geometries(molecule)[PES_CARTESIAN_PANEL_VARIANT]
    if record.get("geometry_sha256") != geometry_sha256(atoms):
        raise ValueError("Cartesian base geometry differs from the frozen asset.")
    forces = np.asarray(record.get("forces_eV_per_A"), dtype=float)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("Cartesian raw force array is invalid.")
    analytic = -forces
    cartesian = record.get("cartesian_force_fd")
    if not isinstance(cartesian, dict):
        raise ValueError("Cartesian raw displacement record is missing.")
    raw_steps = cartesian.get("raw_displaced_components")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(
        PES_CARTESIAN_PANEL_STEPS_A
    ):
        raise ValueError("Cartesian raw step coverage is incomplete.")
    samples = []
    maximum_primal = float(record.get("maximum_primal_residual"))
    topology_hashes = {str(record.get("topology_hash"))}
    for raw_step, step in zip(raw_steps, PES_CARTESIAN_PANEL_STEPS_A):
        if float(raw_step.get("step_A")) != step:
            raise ValueError("Cartesian raw step changed from the contract.")
        components = raw_step.get("components")
        if not isinstance(components, list) or len(components) != 3 * len(atoms):
            raise ValueError("Cartesian raw component coverage is incomplete.")
        finite_difference = np.empty_like(analytic)
        seen: set[tuple[int, int]] = set()
        for raw in components:
            atom, axis = int(raw.get("atom")), int(raw.get("axis"))
            if not 0 <= atom < len(atoms) or not 0 <= axis < 3 or (atom, axis) in seen:
                raise ValueError("Cartesian raw component identity is invalid.")
            seen.add((atom, axis))
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            if raw.get("plus_geometry_sha256") != geometry_sha256(plus) or raw.get(
                "minus_geometry_sha256"
            ) != geometry_sha256(minus):
                raise ValueError("Cartesian displaced geometry hash is invalid.")
            values = tuple(
                float(raw.get(name))
                for name in (
                    "plus_energy_eV",
                    "minus_energy_eV",
                    "plus_primal_residual",
                    "minus_primal_residual",
                )
            )
            if not all(np.isfinite(value) for value in values):
                raise ValueError("Cartesian raw energy/residual values must be finite.")
            plus_energy, minus_energy, plus_residual, minus_residual = values
            finite_difference[atom, axis] = (plus_energy - minus_energy) / (2.0 * step)
            maximum_primal = max(maximum_primal, plus_residual, minus_residual)
            topology_hashes.update(
                (raw.get("plus_topology_hash"), raw.get("minus_topology_hash"))
            )
        if len(seen) != 3 * len(atoms):
            raise ValueError("Cartesian raw components are duplicated or missing.")
        samples.append((step, finite_difference))
    recomputed = summarize_cartesian_force_differences(analytic, samples)
    recomputed.update(
        {
            "maximum_displaced_primal_residual": maximum_primal,
            "topology_hashes": sorted(topology_hashes),
            "fixed_topology": len(topology_hashes) == 1,
        }
    )
    root = _root_record(record.get("cold_warm"))
    result = {
        "molecule_id": molecule.molecule_id,
        "variant": PES_CARTESIAN_PANEL_VARIANT,
        "component_count": 3 * len(atoms),
        "cartesian_force_fd": recomputed,
        "cold_warm": root,
        "maximum_primal_residual": max(maximum_primal, root["maximum_primal_residual"]),
        "adjoint_residual": float(record.get("adjoint_residual")),
    }
    if require_extended_gates:
        result.update(_extended_record(record))
    return result


def aggregate_cartesian_panel(
    args: argparse.Namespace,
    *,
    schema_version: str = SCHEMA_VERSION,
    shard_schema_version: str = SHARD_SCHEMA_VERSION,
    contract_version: str = PES_CARTESIAN_PANEL_CONTRACT_VERSION,
    expected_profile_id: str | None = None,
    expected_scalar_id: str | None = None,
    require_extended_gates: bool = False,
    verifier_required_paths=(
        "tools/route2_release/aggregate_fixedbox590_cartesian_panel.py",
        "maple/solvation/release/pes_panel.py",
        "maple/solvation/release/pes_validation.py",
    ),
    output_marker: str = "ROUTE2_FIXEDBOX590_CARTESIAN_PANEL_AGGREGATE",
) -> None:
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    shards = _load_shards(
        args.shard,
        shard_schema_version=shard_schema_version,
        contract_version=contract_version,
        expected_profile_id=expected_profile_id,
        expected_scalar_id=expected_scalar_id,
    )
    molecules = {item.molecule_id: item for item in load_pes_panel()}
    records = []
    source_hashes = None
    artifact_records = []
    for path, payload in shards:
        artifact_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        current_hashes = payload.get("source_files_sha256")
        if source_hashes is None:
            source_hashes = current_hashes
        elif current_hashes != source_hashes:
            raise ValueError(
                "Cartesian shards have different source-file hash ledgers."
            )
        for raw in payload.get("measurements", ()):
            molecule = molecules.get(str(raw.get("molecule_id")))
            if molecule is None:
                raise ValueError("Cartesian shard contains an unknown molecule.")
            records.append(
                _raw_cartesian(
                    raw, molecule, require_extended_gates=require_extended_gates
                )
            )

    summary = summarize_cartesian_pes_panel(records)
    extended_passed = not require_extended_gates or all(
        record["all_root_multistart_gates_passed"]
        and record["scalar_identity"]["gate_passed"]
        and record["domain"]["all_gates_passed"]
        for record in records
    )
    first = shards[0][1]
    if (
        repository.head != first["execution_git_head"]
        or repository.tree != first["execution_git_tree"]
    ):
        raise RuntimeError(
            "Cartesian aggregator checkout must be the exact shard source tree."
        )
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Cartesian shard source-file hash ledger is missing.")
    if committed_source_hashes(repository, source_hashes) != source_hashes:
        raise RuntimeError("Cartesian shard sources do not match the exact Git tree.")
    verifier_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=verifier_required_paths,
    )
    output = {
        "schema_version": schema_version,
        "status": "pass" if summary["all_gates_passed"] and extended_passed else "fail",
        "claim_boundary": (
            "Independent raw-value recomputation of the preregistered 20-molecule "
            "reference-geometry Cartesian same-scalar panel. This does not itself "
            "enable E/F/H/V/M, prove chemical accuracy, complete solvation free "
            "energy, Hessian, frequency, TS, or MD support."
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
        "cartesian_panel_summary": summary,
        "extended_gate_summary": {
            "required": require_extended_gates,
            "record_count": len(records),
            "all_gates_passed": extended_passed,
        },
    }
    output["aggregate_measurement_sha256"] = canonical_json_sha256(records)
    target = args.output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate artifact: {target}")
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, target, output)
    repository.assert_unchanged()
    print(
        output_marker
        + "="
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
    if not summary["all_gates_passed"] or not extended_passed:
        sys.exit(2)


def main() -> None:
    aggregate_cartesian_panel(_parse_args())


if __name__ == "__main__":
    main()
