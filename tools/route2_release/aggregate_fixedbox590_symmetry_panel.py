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
from maple.solvation.api.profiles import (
    DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
)
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
    closed_loop_work,
    closed_rectangular_loop,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_directions,
    panel_geometries,
    reverse_closed_path,
    runtime_record,
    summarize_bidirectional_loop_record,
    summarize_rigid_symmetry,
    summarize_symmetry_panel,
    symmetry_panel_rotations,
    write_external_json_artifact,
)
from maple.solvation.release.evidence import sha256_file

from aggregate_fixedbox590_pes_panel import _root_record

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


def _load_shards(
    paths,
    *,
    shard_schema_version=SHARD_SCHEMA_VERSION,
    contract_version=SYMMETRY_PANEL_CONTRACT_VERSION,
    expected_profile_id=DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    expected_scalar_id=None,
):
    payloads = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text())
        contract = payload.get("panel_contract")
        if (
            payload.get("schema_version") != shard_schema_version
            or payload.get("capabilities") != CAPABILITIES
            or not isinstance(contract, dict)
            or contract.get("contract_version") != contract_version
            or contract.get("profile_id") != expected_profile_id
            or payload.get("identities", {}).get("profile_id") != expected_profile_id
            or contract.get("asset_sha256") != PES_PANEL_ASSET_SHA256
            or contract.get("variant") != PES_CARTESIAN_PANEL_VARIANT
            or contract.get("rotation_count") != SYMMETRY_PANEL_ROTATION_COUNT
            or tuple(contract.get("translation_A", ())) != SYMMETRY_PANEL_TRANSLATION_A
            or tuple(contract.get("loop_amplitudes_A", ()))
            != SYMMETRY_PANEL_LOOP_AMPLITUDES_A
            or contract.get("loop_subdivisions_per_edge")
            != SYMMETRY_PANEL_LOOP_SUBDIVISIONS
        ):
            raise ValueError(f"Unexpected symmetry shard contract: {path}.")
        if expected_scalar_id is not None and (
            contract.get("scalar_id") != expected_scalar_id
            or payload.get("identities", {}).get("scalar_id") != expected_scalar_id
        ):
            raise ValueError(f"Symmetry shard uses a different scalar: {path}.")
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


def _raw_record(
    raw,
    molecule,
    *,
    contract_version=SYMMETRY_PANEL_CONTRACT_VERSION,
    require_domain_gates=False,
):
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
    loop_for_summary = dict(loop)
    if require_domain_gates:
        directions = panel_directions(atoms, molecule.molecule_id)
        first = directions["seeded-internal"]
        raw_second = directions["radial-internal"]
        second = raw_second - float(np.vdot(first, raw_second)) * first
        second /= float(np.linalg.norm(second))
        forward = closed_rectangular_loop(
            subdivisions_per_edge=SYMMETRY_PANEL_LOOP_SUBDIVISIONS
        )
        reverse = reverse_closed_path(forward)

        def expected_geometry(coefficient):
            result = atoms.copy()
            result.positions += (
                SYMMETRY_PANEL_LOOP_AMPLITUDES_A[0] * coefficient[0] * first
                + SYMMETRY_PANEL_LOOP_AMPLITUDES_A[1] * coefficient[1] * second
            )
            return result

        traversal_coefficients = {
            "cold_forward": forward,
            "cold_reverse": reverse,
            "warm_forward": forward,
            "warm_reverse": reverse,
        }
        raw_traversals = loop.get("raw_traversals")
        if not isinstance(raw_traversals, dict):
            raise ValueError("Symmetry loop raw traversals are missing.")
        recomputed_residuals = []
        recomputed_topologies = []
        for name, coefficients in traversal_coefficients.items():
            points = raw_traversals.get(name)
            if not isinstance(points, list) or len(points) != len(coefficients):
                raise ValueError("Symmetry loop raw traversal coverage is incomplete.")
            positions = []
            forces = []
            for point, coefficient in zip(points, coefficients, strict=True):
                expected = expected_geometry(coefficient)
                point_positions = np.asarray(point.get("positions_A"), dtype=float)
                point_forces = np.asarray(point.get("forces_eV_per_A"), dtype=float)
                residuals = (
                    float(point.get("primal_residual")),
                    float(point.get("adjoint_residual")),
                )
                if (
                    point.get("geometry_sha256") != geometry_sha256(expected)
                    or not np.array_equal(point_positions, expected.positions)
                    or point_forces.shape != (len(atoms), 3)
                    or not np.all(np.isfinite(point_forces))
                    or not all(
                        np.isfinite(value) and value >= 0.0 for value in residuals
                    )
                ):
                    raise ValueError("Symmetry loop raw traversal point is invalid.")
                positions.append(point_positions)
                forces.append(point_forces)
                recomputed_residuals.append(residuals)
                recomputed_topologies.append(str(point.get("topology_hash")))
            loop_for_summary[name] = closed_loop_work(
                positions,
                forces,
                subdivisions_per_edge=SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
            )
        loop_for_summary["maximum_primal_residual"] = max(
            value[0] for value in recomputed_residuals
        )
        loop_for_summary["maximum_adjoint_residual"] = max(
            value[1] for value in recomputed_residuals
        )
        loop_for_summary["topology_hashes"] = sorted(set(recomputed_topologies))

        roots = loop.get("cold_warm_roots")
        if not isinstance(roots, dict):
            raise ValueError("Symmetry loop cold/warm root records are missing.")
        recomputed_roots = []
        for name, coefficients in (("forward", forward), ("reverse", reverse)):
            records = roots.get(name)
            if not isinstance(records, list) or len(records) != len(coefficients):
                raise ValueError("Symmetry loop cold/warm root coverage is incomplete.")
            for record, coefficient in zip(records, coefficients, strict=True):
                if record.get("geometry_sha256") != geometry_sha256(
                    expected_geometry(coefficient)
                ):
                    raise ValueError("Symmetry loop root geometry binding is invalid.")
                recomputed_roots.append(_root_record(record, require_field_gate=True))
        loop_for_summary["all_cold_warm_roots"] = all(
            record["numerically_equivalent"] and all(record["gates"].values())
            for record in recomputed_roots
        )

        raw_repeats = loop.get("raw_warm_repeat_records")
        if not isinstance(raw_repeats, list) or len(raw_repeats) != len(forward):
            raise ValueError("Symmetry loop warm-repeat records are incomplete.")
        warm_repeat_passed = True
        for record, coefficient in zip(raw_repeats, forward, strict=True):
            forward_source = np.asarray(record.get("forward_source"), dtype=float)
            reverse_source = np.asarray(record.get("reverse_source"), dtype=float)
            energies = (
                float(record.get("forward_energy_eV")),
                float(record.get("reverse_energy_eV")),
            )
            if (
                record.get("geometry_sha256")
                != geometry_sha256(expected_geometry(coefficient))
                or forward_source.shape != reverse_source.shape
                or forward_source.ndim != 2
                or not np.all(np.isfinite(forward_source))
                or not np.all(np.isfinite(reverse_source))
                or not all(np.isfinite(value) for value in energies)
            ):
                raise ValueError("Symmetry loop warm-repeat record is invalid.")
            source_relative = float(
                np.linalg.norm(forward_source - reverse_source)
            ) / max(
                float(np.linalg.norm(forward_source)),
                float(np.linalg.norm(reverse_source)),
                1.0e-15,
            )
            warm_repeat_passed &= bool(
                source_relative <= 1.0e-8 and abs(energies[0] - energies[1]) <= 1.0e-8
            )
        loop_for_summary["warm_forward_reverse_repeat"] = warm_repeat_passed

        domain_records = loop.get("domain_records")
        if not isinstance(domain_records, list) or len(domain_records) != len(forward):
            raise ValueError("Symmetry loop domain records are incomplete.")
        domain_passed = True
        for record, coefficient in zip(domain_records, forward, strict=True):
            if record.get("geometry_sha256") != geometry_sha256(
                expected_geometry(coefficient)
            ):
                raise ValueError("Symmetry loop domain geometry binding is invalid.")
            values = tuple(
                float(record.get(name))
                for name in (
                    "minimum_center_distance_A",
                    "minimum_relative_basis_singular_value",
                    "surface_minimum_eigenvalue",
                    "surface_condition_number",
                )
            )
            if not all(np.isfinite(value) for value in values):
                raise ValueError("Symmetry loop domain values must be finite.")
            domain_passed &= bool(
                values[0] > 0.0
                and values[1] > 1.0e-10
                and values[2] > 1.0e-12
                and values[3] <= 1.0e12
            )
    recomputed_loop = summarize_bidirectional_loop_record(loop_for_summary)
    if require_domain_gates:
        recomputed_loop["all_domain_gates_passed"] = domain_passed
        recomputed_loop["all_gates_passed"] = bool(
            recomputed_loop["all_gates_passed"] and domain_passed
        )
    return {
        "molecule_id": molecule.molecule_id,
        "contract_version": contract_version,
        "geometry_sha256": geometry_sha256(atoms),
        "rigid_symmetry": recomputed_rigid,
        "closed_loop": recomputed_loop,
    }


def aggregate_symmetry_panel(
    args,
    *,
    schema_version=SCHEMA_VERSION,
    shard_schema_version=SHARD_SCHEMA_VERSION,
    contract_version=SYMMETRY_PANEL_CONTRACT_VERSION,
    expected_profile_id=DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    expected_scalar_id=None,
    require_domain_gates=False,
    verifier_required_paths=(
        "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
        "maple/solvation/release/symmetry_panel.py",
        "maple/solvation/release/pes_validation.py",
    ),
    output_marker="ROUTE2_FIXEDBOX590_SYMMETRY_PANEL_AGGREGATE",
):
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
    artifacts = []
    for path, payload in shards:
        artifacts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
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
            records.append(
                _raw_record(
                    raw,
                    molecule,
                    contract_version=contract_version,
                    require_domain_gates=require_domain_gates,
                )
            )
    summary = summarize_symmetry_panel(records, contract_version=contract_version)
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
        required_paths=verifier_required_paths,
    )
    output = {
        "schema_version": schema_version,
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
    if not summary["all_gates_passed"]:
        sys.exit(2)


def main():
    aggregate_symmetry_panel(_parse_args())


if __name__ == "__main__":
    main()
