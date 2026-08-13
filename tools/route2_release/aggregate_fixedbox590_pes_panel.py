#!/usr/bin/env python3
"""Independently aggregate all frozen fixed-box590 PES-panel shards.

This verifier recomputes gates from raw energies, geometries, forces, and
residuals. It never trusts a shard's cached pass/fail booleans and cannot alter
the capability registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_CONTRACT_VERSION,
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_VARIANT_NAMES,
    RepositorySnapshot,
    canonical_json_sha256,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_directions,
    panel_geometries,
    panel_paths,
    runtime_record,
    stretch_tangent,
    summarize_pes_panel,
    summarize_pes_paths,
    torsion_tangent,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file
from maple.solvation.release.pes_validation import summarize_directional_derivatives

SCHEMA_VERSION = "route2-fixedbox590-pes-panel-aggregate-v1"
SHARD_SCHEMA_VERSION = "route2-fixedbox590-pes-panel-shard-v1"
CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _raw_direction(record, atoms, direction, topology_hash, analytic):
    displaced = record.get("displaced_states")
    if not isinstance(displaced, list) or len(displaced) != len(
        PES_PANEL_DIRECTIONAL_STEPS_A
    ):
        raise ValueError("Directional record has incomplete raw displaced states.")
    analytic = float(analytic)
    if not np.isfinite(analytic):
        raise ValueError("Analytic raw-force projection must be finite.")
    samples = []
    maximum_primal = 0.0
    topology_hashes = {topology_hash}
    for raw, step in zip(displaced, PES_PANEL_DIRECTIONAL_STEPS_A):
        if float(raw.get("step_A")) != step:
            raise ValueError("Directional steps changed from the frozen contract.")
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        if raw.get("plus_geometry_sha256") != geometry_sha256(plus) or raw.get(
            "minus_geometry_sha256"
        ) != geometry_sha256(minus):
            raise ValueError("Directional displaced geometry hash is invalid.")
        plus_energy = float(raw.get("plus_energy_eV"))
        minus_energy = float(raw.get("minus_energy_eV"))
        plus_residual = float(raw.get("plus_primal_residual"))
        minus_residual = float(raw.get("minus_primal_residual"))
        if not all(
            np.isfinite(value)
            for value in (plus_energy, minus_energy, plus_residual, minus_residual)
        ):
            raise ValueError("Directional raw values must be finite.")
        maximum_primal = max(maximum_primal, plus_residual, minus_residual)
        topology_hashes.update(
            (raw.get("plus_topology_hash"), raw.get("minus_topology_hash"))
        )
        samples.append((step, plus_energy, minus_energy))
    recomputed = summarize_directional_derivatives(analytic, samples)
    recomputed.update(
        {
            "maximum_primal_residual": maximum_primal,
            "fixed_topology": len(topology_hashes) == 1,
            "topology_hashes": sorted(topology_hashes),
        }
    )
    return recomputed


def _root_record(record):
    cold = record.get("cold")
    warm = record.get("warm")
    if not isinstance(cold, dict) or not isinstance(warm, dict):
        raise ValueError("Cold/warm root leaves are missing.")
    cold_source = np.asarray(cold.get("source"), dtype=float)
    warm_source = np.asarray(warm.get("source"), dtype=float)
    cold_field = np.asarray(cold.get("field"), dtype=float)
    warm_field = np.asarray(warm.get("field"), dtype=float)
    if (
        cold_source.shape != warm_source.shape
        or cold_field.shape != warm_field.shape
        or cold_source.ndim != 2
        or cold_field.shape != cold_source.shape
        or not all(
            np.all(np.isfinite(value))
            for value in (cold_source, warm_source, cold_field, warm_field)
        )
    ):
        raise ValueError("Cold/warm source and field arrays are invalid.")
    source_difference = float(np.linalg.norm(cold_source - warm_source))
    source_relative = source_difference / max(
        float(np.linalg.norm(cold_source)),
        float(np.linalg.norm(warm_source)),
        1.0e-15,
    )
    energy_difference = abs(
        float(cold.get("total_energy_eV")) - float(warm.get("total_energy_eV"))
    )
    cold_residual = float(cold.get("actual_unmixed_residual_norm"))
    warm_residual = float(warm.get("actual_unmixed_residual_norm"))
    if not all(
        np.isfinite(value)
        for value in (energy_difference, cold_residual, warm_residual)
    ):
        raise ValueError("Cold/warm energies and residuals must be finite.")
    return {
        "numerically_equivalent": (
            source_relative <= 1.0e-8 and energy_difference <= 1.0e-8
        ),
        "source_l2_difference": source_difference,
        "source_relative_difference": source_relative,
        "field_l2_difference": float(np.linalg.norm(cold_field - warm_field)),
        "energy_abs_difference_eV": energy_difference,
        "maximum_primal_residual": max(cold_residual, warm_residual),
        "gates": {
            "source_relative_le_1e-8": source_relative <= 1.0e-8,
            "energy_le_1e-8_eV": energy_difference <= 1.0e-8,
        },
    }


def _load_shards(paths):
    payloads = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise ValueError(f"Unexpected shard schema: {path}.")
        if payload.get("capabilities") != CAPABILITIES:
            raise ValueError(f"Shard capabilities are not all false: {path}.")
        contract = payload.get("panel_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("contract_version") != PES_PANEL_CONTRACT_VERSION
            or contract.get("asset_sha256") != PES_PANEL_ASSET_SHA256
        ):
            raise ValueError(f"Shard uses a different PES-panel contract: {path}.")
        expected_measurement_sha256 = canonical_json_sha256(
            {
                "contract": payload["panel_contract"],
                "measurements": payload.get("measurements"),
                "path_measurements": payload.get("path_measurements"),
            }
        )
        if payload.get("measurement_sha256") != expected_measurement_sha256:
            raise ValueError(f"Shard measurement digest is invalid: {path}.")
        payloads.append((path, payload))
    heads = {payload["execution_git_head"] for _, payload in payloads}
    trees = {payload["execution_git_tree"] for _, payload in payloads}
    checkpoints = {payload["checkpoint"]["sha256"] for _, payload in payloads}
    runtime_signatures = {
        canonical_json_sha256(
            {
                "device": payload.get("device"),
                "dtype": payload.get("dtype"),
                "platform": payload.get("runtime", {}).get("platform"),
                "machine": payload.get("runtime", {}).get("machine"),
                "packages": payload.get("runtime", {}).get("packages"),
                "numpy_version": payload.get("runtime", {})
                .get("numpy", {})
                .get("version"),
                "torch": payload.get("runtime", {}).get("torch"),
                "thread_environment": payload.get("runtime", {}).get("environment"),
                "numerical_determinism": payload.get("numerical_determinism"),
            }
        )
        for _, payload in payloads
    }
    if (
        len(heads) != 1
        or len(trees) != 1
        or len(checkpoints) != 1
        or len(runtime_signatures) != 1
    ):
        raise ValueError(
            "All PES shards must share one source tree, checkpoint, and numerical "
            "runtime signature."
        )
    return payloads


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    shards = _load_shards(args.shard)
    base_records = []
    path_records = []
    source_hashes = None
    artifact_records = []
    molecules = {item.molecule_id: item for item in load_pes_panel()}
    frozen_paths = {
        (point.path_name, point.point_label): point
        for path in panel_paths().values()
        for point in path
    }
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
            raise ValueError("PES shards have different source-file hash ledgers.")
        for raw in payload.get("measurements", ()):
            molecule = molecules.get(str(raw.get("molecule_id")))
            variant = str(raw.get("variant"))
            if molecule is None or variant not in PES_PANEL_VARIANT_NAMES:
                raise ValueError("Shard contains an unknown base geometry.")
            atoms = panel_geometries(molecule)[variant]
            if raw.get("geometry_sha256") != geometry_sha256(atoms):
                raise ValueError("Shard base geometry differs from the frozen asset.")
            directions = panel_directions(atoms, molecule.molecule_id)
            forces = np.asarray(raw.get("forces_eV_per_A"), dtype=float)
            if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
                raise ValueError("Shard base force array is invalid.")
            gradient = -forces
            directional = {
                name: _raw_direction(
                    raw["directional_force_fd"][name],
                    atoms,
                    directions[name],
                    raw["topology_hash"],
                    float(np.vdot(gradient, directions[name])),
                )
                for name in PES_PANEL_DIRECTION_NAMES
            }
            maximum_primal = max(
                float(raw["maximum_primal_residual"]),
                *(record["maximum_primal_residual"] for record in directional.values()),
            )
            base_records.append(
                {
                    "molecule_id": molecule.molecule_id,
                    "variant": variant,
                    "directional_force_fd": directional,
                    "cold_warm": (root := _root_record(raw["cold_warm"])),
                    "topology_hash": raw["topology_hash"],
                    "maximum_primal_residual": max(
                        maximum_primal, root["maximum_primal_residual"]
                    ),
                    "adjoint_residual": float(raw["adjoint_residual"]),
                }
            )
        for raw in payload.get("path_measurements", ()):
            key = (str(raw.get("path_name")), str(raw.get("point_label")))
            point = frozen_paths.get(key)
            if point is None or raw.get("geometry_sha256") != geometry_sha256(
                point.atoms
            ):
                raise ValueError("Shard contains an unknown or changed path geometry.")
            tangent = (
                torsion_tangent(point.atoms)
                if point.path_name.startswith("trans-butane")
                else stretch_tangent(point.atoms)
            )
            unit_direction = tangent / np.linalg.norm(tangent)
            forces = np.asarray(raw.get("forces_eV_per_A"), dtype=float)
            if forces.shape != (len(point.atoms), 3) or not np.all(np.isfinite(forces)):
                raise ValueError("Shard path force array is invalid.")
            gradient = -forces
            local = _raw_direction(
                raw["local_tangent_force_fd"],
                point.atoms,
                unit_direction,
                raw["topology_hash"],
                float(np.vdot(gradient, unit_direction)),
            )
            root = _root_record(raw["cold_warm"])
            path_records.append(
                {
                    **{
                        name: raw[name]
                        for name in (
                            "path_name",
                            "molecule_id",
                            "point_label",
                            "coordinate_name",
                            "coordinate_value",
                            "coordinate_unit",
                            "geometry_sha256",
                            "energy_eV",
                            "topology_hash",
                            "adjoint_residual",
                        )
                    },
                    "cold_warm": root,
                    "maximum_primal_residual": max(
                        float(raw["maximum_primal_residual"]),
                        local["maximum_primal_residual"],
                        root["maximum_primal_residual"],
                    ),
                    "local_tangent_force_fd": local,
                }
            )

    panel_summary = summarize_pes_panel(base_records)
    path_summary = summarize_pes_paths(path_records)
    aggregate_passed = (
        panel_summary["all_gates_passed"] and path_summary["all_gates_passed"]
    )
    first_payload = shards[0][1]
    if (
        repository.head != first_payload["execution_git_head"]
        or repository.tree != first_payload["execution_git_tree"]
    ):
        raise RuntimeError(
            "Aggregator checkout must be the exact source tree used by every shard."
        )
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Shard source-file hash ledger is missing.")
    rebound_shard_sources = committed_source_hashes(repository, source_hashes)
    if rebound_shard_sources != source_hashes:
        raise RuntimeError("Shard source-file hashes do not match the exact Git tree.")
    verifier_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            "tools/route2_release/aggregate_fixedbox590_pes_panel.py",
            "maple/solvation/release/pes_panel.py",
            "maple/solvation/release/pes_validation.py",
        ),
    )
    verifier_hashes = committed_source_hashes(repository, verifier_paths)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if aggregate_passed else "fail",
        "claim_boundary": (
            "Independent raw-value recomputation of the preregistered same-scalar "
            "PES panel. This does not enable E/F/H/V/M, prove chemical accuracy, "
            "complete solvation free energy, Hessian, frequency, TS, or MD support."
        ),
        "capabilities": CAPABILITIES,
        "execution_git_head": first_payload["execution_git_head"],
        "execution_git_tree": first_payload["execution_git_tree"],
        "exact_command": shlex.join(sys.argv),
        "checkpoint_sha256": first_payload["checkpoint"]["sha256"],
        "source_files_sha256": source_hashes,
        "verifier_source_files_sha256": verifier_hashes,
        "verifier_runtime": runtime_record(),
        "input_artifacts": artifact_records,
        "panel_summary": panel_summary,
        "path_summary": path_summary,
    }
    output["aggregate_measurement_sha256"] = canonical_json_sha256(
        {"base": base_records, "paths": path_records}
    )
    target = args.output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate artifact: {target}")
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, target, output)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_PES_PANEL_AGGREGATE="
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
    if not aggregate_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
