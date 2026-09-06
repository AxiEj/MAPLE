#!/usr/bin/env python3
"""Preregister and audit fixed local-whitened ETB-beta2 molecular coordinates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SELF_REPO_PATH = (
    "tools/route2_release/run_local_whitened_etb2_condition_gate.py"
)
ARTIFACT = "route2-local-whitened-etb2-condition-gate-v1"
PREREG_ARTIFACT = f"{ARTIFACT}-preregistration"
RESULT_ARTIFACT = f"{ARTIFACT}-result"
EXPECTED_CASE_COUNT = 12
GLOBAL_CONDITION_MAXIMUM = 1.0e10
CONSTRAINT_GRAM_CONDITION_MAXIMUM = 1.0e10
CONSTRAINT_RANK_RELATIVE_TOLERANCE = 1.0e-12
CONSTRAINT_RESIDUAL_MAXIMUM = 1.0e-10
MOMENT_LENGTH_UNIT_BOHR = 1.0
ATOMIC_ASSET = SOURCE_ROOT / (
    "docs/implicit-solvation/benchmarks/route2-etb2-atomic-whitening-v1/"
    "asset.json"
)
DEVELOPMENT_PREREGISTRATION = SOURCE_ROOT / (
    "docs/route2/evidence/auxiliary-density-standard-basis-ladder-20260824/"
    "preregistration.json"
)
FAILED_LADDER_RESULT = SOURCE_ROOT / (
    "docs/route2/evidence/auxiliary-density-standard-basis-ladder-20260824/"
    "result.json"
)
PRO_ANSWER = SOURCE_ROOT / (
    "docs/route2/evidence/auxiliary-density-conditioning-architecture-pro-20260824/"
    "answer.md"
)
SOURCE_PATHS = (
    SELF_REPO_PATH,
    "maple/solvation/reference/atomic_auxiliary_whitening.py",
    "maple/solvation/reference/pyscf_pcmsolver.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object.")
    return value


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    path.chmod(0o444)


def _write_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def create_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    result = args.result.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if result.exists():
        raise ValueError("result path must not exist before preregistration.")
    development = _load_object(
        DEVELOPMENT_PREREGISTRATION.resolve(strict=True),
        name="development preregistration",
    )
    asset = _load_object(ATOMIC_ASSET.resolve(strict=True), name="atomic asset")
    if (
        development.get("artifact")
        != "route2-auxiliary-density-standard-basis-ladder-v1-preregistration"
        or len(development.get("cases", ())) != EXPECTED_CASE_COUNT
        or asset.get("artifact")
        != "route2-etb2-isolated-atom-coulomb-whitening-asset-v1"
        or asset.get("status")
        != "frozen-before-any-molecular-local-whitening-audit"
    ):
        raise ValueError("parent development/atomic identities are incompatible.")
    payload: dict[str, Any] = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-molecular-local-whitening-condition-result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_files_sha256": {
            relative: _sha256(SOURCE_ROOT / relative) for relative in SOURCE_PATHS
        },
        "parents": {
            "development_preregistration": _file_record(
                DEVELOPMENT_PREREGISTRATION
            ),
            "failed_standard_ladder_result": _file_record(FAILED_LADDER_RESULT),
            "atomic_whitening_asset": _file_record(ATOMIC_ASSET),
            "conditioning_pro_advice": _file_record(PRO_ANSWER),
            "advice_is_non_authoritative": True,
        },
        "cases": development["cases"],
        "architecture": {
            "basis": "pyscf.df.aug_etb(beta=2.0)",
            "local_transform": (
                "fixed isolated-atom symmetric Coulomb inverse square root per "
                "element and l; identical for every m"
            ),
            "molecular_global_transform_or_pivot": False,
            "mode_truncation_or_regularization": False,
            "physical_span_changed": False,
            "source_field_duality": "c_old=T c_new; u_new=T.T u_old",
        },
        "gates": {
            "constraint_rank": 4,
            "constraint_rank_relative_tolerance": (
                CONSTRAINT_RANK_RELATIVE_TOLERANCE
            ),
            "global_constraint_nullspace_condition_max": (
                GLOBAL_CONDITION_MAXIMUM
            ),
            "constraint_gram_condition_max": (
                CONSTRAINT_GRAM_CONDITION_MAXIMUM
            ),
            "constraint_residual_max": CONSTRAINT_RESIDUAL_MAXIMUM,
            "moment_length_unit_bohr": MOMENT_LENGTH_UNIT_BOHR,
            "every_case_required": True,
        },
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "model_output_or_training_read": False,
            "capability_admitted": False,
            "forward_high_precision_gate_completed": False,
            "allowed_decision": (
                "close auxiliary route on failure or advance to forward-stability "
                "and confirmation gates on pass"
            ),
        },
        "result_path": str(result),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    _write_exclusive(output, payload)
    return payload


def _validate_preregistration(path: Path) -> dict[str, Any]:
    payload = _load_object(path, name="preregistration")
    unsigned = dict(payload)
    observed = unsigned.pop("preregistration_sha256", None)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("status")
        != "locked-before-first-molecular-local-whitening-condition-result"
        or observed != _canonical_sha256(unsigned)
    ):
        raise ValueError("local-whitening preregistration drifted.")
    for relative, expected in payload["source_files_sha256"].items():
        if _sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"preregistered source {relative!r} drifted.")
    for record in payload["parents"].values():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path_value = Path(str(record["path"])).resolve(strict=True)
        if _sha256(path_value) != record["sha256"]:
            raise ValueError("preregistered parent evidence drifted.")
    if len(payload.get("cases", ())) != EXPECTED_CASE_COUNT:
        raise ValueError("local-whitening case panel drifted.")
    return payload


def _atomic_asset_records(asset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["element"]): record for record in asset["records"]}


def _load_atomic_transform(
    asset_root: Path,
    record: dict[str, Any],
) -> np.ndarray:
    path = (asset_root / str(record["transform_file"])).resolve(strict=True)
    if _sha256(path) != record["transform_file_sha256"]:
        raise RuntimeError("atomic whitening transform file drifted.")
    values = np.load(path, allow_pickle=False)
    expected = int(record["auxiliary_dimension"])
    if values.shape != (expected, expected) or not np.all(np.isfinite(values)):
        raise RuntimeError("atomic whitening transform is invalid.")
    return np.asarray(values, dtype=np.float64)


def _moment_constraints(auxiliary: Any) -> np.ndarray:
    from pyscf import dft

    grids = dft.gen_grid.Grids(auxiliary)
    grids.level = 4
    grids.build(with_non0tab=False)
    dimension = int(auxiliary.nao_nr())
    charge = np.zeros(dimension, dtype=np.float64)
    first = np.zeros((3, dimension), dtype=np.float64)
    for start in range(0, len(grids.coords), 30000):
        stop = min(start + 30000, len(grids.coords))
        coordinates = np.asarray(grids.coords[start:stop], dtype=np.float64)
        weights = np.asarray(grids.weights[start:stop], dtype=np.float64)
        ao = np.asarray(dft.numint.eval_ao(auxiliary, coordinates), dtype=np.float64)
        charge += np.einsum("g,gp->p", weights, ao, optimize=True)
        first += np.einsum(
            "g,gx,gp->xp", weights, coordinates, ao, optimize=True
        )
    return np.vstack((charge, first / MOMENT_LENGTH_UNIT_BOHR))


def _case_record(
    case: dict[str, Any],
    *,
    asset_root: Path,
    asset_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from pyscf import df, lib
    from scipy import linalg

    from maple.solvation.reference.pyscf_pcmsolver import (
        closed_shell_density_from_orbitals,
    )

    started = time.perf_counter()
    checkpoint = Path(str(case["files"]["checkpoint"]["path"]))
    molecule = lib.chkfile.load_mol(str(checkpoint))
    molecule.verbose = 0
    basis = df.aug_etb(molecule, beta=2.0)
    auxiliary = df.addons.make_auxmol(molecule, auxbasis=basis)
    dimension = int(auxiliary.nao_nr())
    transform = np.zeros((dimension, dimension), dtype=np.float64)
    slices = np.asarray(auxiliary.aoslice_by_atom(), dtype=np.int64)
    for atom_index, symbol in enumerate(molecule.elements):
        record = asset_records[str(symbol)]
        atomic_transform = _load_atomic_transform(asset_root, record)
        start, stop = (int(value) for value in slices[atom_index, 2:4])
        if stop - start != atomic_transform.shape[0]:
            raise RuntimeError("molecular and atomic ETB2 basis dimensions differ.")
        transform[start:stop, start:stop] = atomic_transform
        atomic_basis = _load_object(
            (asset_root / str(record["basis_file"])).resolve(strict=True),
            name="atomic basis",
        )
        if basis[str(symbol)] != atomic_basis[str(symbol)]:
            raise RuntimeError("molecular and atomic ETB2 basis definitions differ.")
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)
    whitened_metric = transform.T @ metric @ transform
    whitened_metric = 0.5 * (whitened_metric + whitened_metric.T)
    constraints = _moment_constraints(auxiliary) @ transform

    charge_row = constraints[0]
    positions = np.asarray(molecule.atom_coords(), dtype=np.float64)
    weights = np.empty(len(positions), dtype=np.float64)
    for atom_index in range(len(positions)):
        start, stop = (int(value) for value in slices[atom_index, 2:4])
        weights[atom_index] = float(np.vdot(charge_row[start:stop], charge_row[start:stop]))
    if np.sum(weights) <= 0.0:
        raise RuntimeError("local charge carriers vanished.")
    origin = np.sum(weights[:, None] * positions, axis=0) / np.sum(weights)
    centered_constraints = np.array(constraints, copy=True)
    centered_constraints[1:] -= origin[:, None] * charge_row[None, :]
    singular_values = linalg.svdvals(centered_constraints)
    rank = int(
        np.count_nonzero(
            singular_values
            > singular_values[0] * CONSTRAINT_RANK_RELATIVE_TOLERANCE
        )
    )
    if rank != 4:
        raise RuntimeError("local-whitened molecular constraints lack rank four.")
    q_matrix, _r_matrix = linalg.qr(
        centered_constraints.T,
        mode="full",
        pivoting=False,
        check_finite=False,
    )
    nullspace = np.ascontiguousarray(q_matrix[:, 4:])
    restricted = nullspace.T @ whitened_metric @ nullspace
    restricted = 0.5 * (restricted + restricted.T)
    eigenvalues = linalg.eigvalsh(restricted, check_finite=False, driver="evr")
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("local-whitened restricted metric is not positive.")
    global_condition = float(eigenvalues[-1] / eigenvalues[0])
    constraint_gram = centered_constraints @ centered_constraints.T
    constraint_eigenvalues = linalg.eigvalsh(
        0.5 * (constraint_gram + constraint_gram.T),
        check_finite=False,
    )
    constraint_condition = float(
        constraint_eigenvalues[-1] / constraint_eigenvalues[0]
    )

    overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    density, density_record = closed_shell_density_from_orbitals(
        lib.chkfile.load(str(checkpoint), "scf/mo_coeff"),
        lib.chkfile.load(str(checkpoint), "scf/mo_occ"),
        overlap,
    )
    position_integrals = np.asarray(
        molecule.intor_symmetric("int1e_r", comp=3), dtype=np.float64
    )
    electrons = float(np.einsum("ij,ji", density, overlap, optimize=True))
    first_moment = np.einsum(
        "xij,ji->x", position_integrals, density, optimize=True
    )
    target = np.concatenate(
        ([electrons], (first_moment - origin * electrons) / MOMENT_LENGTH_UNIT_BOHR)
    )
    projected = centered_constraints.T @ linalg.solve(
        constraint_gram,
        target,
        assume_a="pos",
        check_finite=False,
    )
    residual = float(
        np.max(np.abs(centered_constraints @ projected - target))
    )
    passes = bool(
        global_condition <= GLOBAL_CONDITION_MAXIMUM
        and constraint_condition <= CONSTRAINT_GRAM_CONDITION_MAXIMUM
        and residual <= CONSTRAINT_RESIDUAL_MAXIMUM
    )
    return {
        "compound_id": case["compound_id"],
        "name": case["name"],
        "atom_count": len(positions),
        "auxiliary_dimension": dimension,
        "constraint_rank": rank,
        "translation_covariant_origin_bohr": origin.tolist(),
        "global_restricted_minimum_eigenvalue": float(eigenvalues[0]),
        "global_restricted_maximum_eigenvalue": float(eigenvalues[-1]),
        "global_restricted_condition_number": global_condition,
        "constraint_gram_condition_number": constraint_condition,
        "constraint_max_absolute_residual": residual,
        "density_record": density_record,
        "case_passed": passes,
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    preregistration = _validate_preregistration(preregistration_path)
    result_path = Path(str(preregistration["result_path"])).resolve()
    if result_path.exists():
        raise FileExistsError(result_path)
    asset = _load_object(ATOMIC_ASSET, name="atomic asset")
    asset_root = ATOMIC_ASSET.parent
    asset_records = _atomic_asset_records(asset)
    records = []
    for case in preregistration["cases"]:
        record = _case_record(
            case,
            asset_root=asset_root,
            asset_records=asset_records,
        )
        records.append(record)
        print(
            f"{record['compound_id']} global_condition="
            f"{record['global_restricted_condition_number']:.8e} "
            f"constraint_condition={record['constraint_gram_condition_number']:.8e} "
            f"pass={record['case_passed']}",
            flush=True,
        )
    maximum_global = max(
        float(record["global_restricted_condition_number"]) for record in records
    )
    maximum_constraint = max(
        float(record["constraint_gram_condition_number"]) for record in records
    )
    maximum_residual = max(
        float(record["constraint_max_absolute_residual"]) for record in records
    )
    gates = {
        "global_condition": maximum_global <= GLOBAL_CONDITION_MAXIMUM,
        "constraint_condition": (
            maximum_constraint <= CONSTRAINT_GRAM_CONDITION_MAXIMUM
        ),
        "constraint_residual": maximum_residual <= CONSTRAINT_RESIDUAL_MAXIMUM,
        "all_cases": all(bool(record["case_passed"]) for record in records),
    }
    result: dict[str, Any] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "preregistration_file_sha256": _sha256(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "record_count": len(records),
        "records": records,
        "aggregate": {
            "maximum_global_restricted_condition_number": maximum_global,
            "maximum_constraint_gram_condition_number": maximum_constraint,
            "maximum_constraint_absolute_residual": maximum_residual,
        },
        "gates": gates,
        "claim_boundary": preregistration["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("preregister")
    create.add_argument("--result", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    value = (
        create_preregistration(args)
        if args.command == "preregister"
        else run(args)
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
