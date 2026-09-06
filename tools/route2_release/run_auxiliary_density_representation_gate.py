#!/usr/bin/env python3
"""Preregister and run a twelve-molecule auxiliary-density basis gate.

The underlying gas-phase QM checkpoints and exterior MEP probes were generated
by an earlier preregistered no-solvation-label source audit.  This gate asks a
new, narrower question: can PySCF's standard ``make_auxbasis`` Coulomb-fitting
basis represent each frozen QM density while enforcing exact electron count
and electronic first moment?  It performs no model training and cannot admit a
source head or solvation capability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np


ARTIFACT = "route2-auxiliary-density-representation-gate-v1"
PREREG_ARTIFACT = f"{ARTIFACT}-preregistration"
RESULT_ARTIFACT = f"{ARTIFACT}-result"
EXPECTED_CASE_COUNT = 12
PARENT_ARTIFACT = (
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-execution-edfae78e"
)
PARENT_STATUS = "complete-source-diagnostic-no-acceptance-threshold"
MEAN_RELATIVE_ERROR_MAX = 0.05
MAXIMUM_CASE_RELATIVE_ERROR_MAX = 0.10
METRIC_EIGENVALUE_RELATIVE_CUTOFF = 1.0e-12
MOMENT_GRID_LEVEL = 4
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/run_auxiliary_density_representation_gate.py"


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
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object.")
    return payload


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
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _write_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _case_files(static_root: Path, compound_id: str) -> dict[str, Path]:
    root = static_root / "records" / compound_id / "qm-attempt-001"
    return {
        "checkpoint": root / "gas/gas.chk",
        "checkpoint_ledger": root / "gas/gas.json",
        "surface": root / "surface.npz",
        "qm_mep": root / "qm-static-mep.npz",
    }


def create_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    static_root = args.static_root.expanduser().resolve(strict=True)
    parent_path = args.parent_execution.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if result_path.exists():
        raise ValueError("result path must not exist before preregistration.")
    parent = _load_object(parent_path, name="parent execution")
    records = parent.get("records")
    if (
        parent.get("artifact") != PARENT_ARTIFACT
        or parent.get("status") != PARENT_STATUS
        or not isinstance(records, list)
        or len(records) != EXPECTED_CASE_COUNT
    ):
        raise ValueError("parent twelve-case source execution is incompatible.")
    cases = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("parent source record is malformed.")
        compound_id = str(record["compound_id"])
        files = _case_files(static_root, compound_id)
        cases.append(
            {
                "compound_id": compound_id,
                "name": str(record["name"]),
                "chemical_class": str(record["chemical_class"]),
                "functional_group": record.get("functional_group"),
                "files": {key: _file_record(path) for key, path in files.items()},
            }
        )
    import pyscf

    payload: dict[str, Any] = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-twelve-case-auxiliary-density-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_files_sha256": {SELF_REPO_PATH: _sha256(SOURCE_ROOT / SELF_REPO_PATH)},
        "parent_execution": _file_record(parent_path),
        "static_root": str(static_root),
        "cases": cases,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "pyscf_init": _file_record(Path(pyscf.__file__).resolve(strict=True)),
        },
        "method": {
            "orbital_basis": "checkpoint-defined def2-TZVPD",
            "auxiliary_basis": "pyscf.df.make_auxbasis(checkpoint molecule)",
            "density_projection": "Coulomb metric",
            "metric_eigenvalue_relative_cutoff": METRIC_EIGENVALUE_RELATIVE_CUTOFF,
            "moment_grid_level": MOMENT_GRID_LEVEL,
            "hard_constraints": "exact electron count and electronic first moment",
            "probe_surface": "frozen geometry-only SMD-radius exterior shell",
        },
        "gates": {
            "mean_area_relative_error_max": MEAN_RELATIVE_ERROR_MAX,
            "maximum_case_area_relative_error_max": (
                MAXIMUM_CASE_RELATIVE_ERROR_MAX
            ),
            "every_constraint_max_absolute_error": 1.0e-8,
        },
        "selection_history": {
            "four_case_make_auxbasis_probe_opened_before_this_lock": True,
            "these_twelve_qm_mep_values_previously_opened_by_parent": True,
            "no_result_from_this_basis_on_the_twelve_case_panel_opened": True,
        },
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "fit_or_training_performed": False,
            "capability_admitted": False,
            "allowed_decision": "reject or retain this basis for later head training",
        },
        "result_path": str(result_path),
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
        or payload.get("schema_version") != 1
        or payload.get("status")
        != "locked-before-first-twelve-case-auxiliary-density-evaluation"
        or observed != _canonical_sha256(unsigned)
    ):
        raise ValueError("auxiliary-density preregistration identity drifted.")
    if payload.get("source_root") != str(SOURCE_ROOT):
        raise ValueError("preregistration source root drifted.")
    for relative, expected in payload["source_files_sha256"].items():
        if _sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"preregistered source {relative!r} drifted.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("preregistered case panel drifted.")
    for case in cases:
        for name, record in case["files"].items():
            file_path = Path(str(record["path"])).resolve(strict=True)
            if _sha256(file_path) != record["sha256"]:
                raise ValueError(f"case file {name!r} drifted.")
    if payload["claim_boundary"] != {
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "capability_admitted": False,
        "allowed_decision": "reject or retain this basis for later head training",
    }:
        raise ValueError("claim boundary drifted.")
    return payload


def _project_case(case: Mapping[str, Any]) -> dict[str, Any]:
    from ase import Atoms
    from pyscf import df, dft, gto, lib

    from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
        build_route2_smd_exterior_probe_surface,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        route2_coulomb_radii,
    )
    from maple.function.route2_smd_profiles import (
        DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    )

    started = time.perf_counter()
    files = case["files"]
    checkpoint = Path(str(files["checkpoint"]["path"]))
    molecule = lib.chkfile.load_mol(str(checkpoint))
    coefficients = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_coeff"))
    occupations = np.asarray(lib.chkfile.load(str(checkpoint), "scf/mo_occ"))
    density = np.einsum(
        "pi,i,qi->pq", coefficients, occupations, coefficients, optimize=True
    )
    with np.load(str(files["surface"]["path"])) as state:
        points = np.asarray(state["surface_points_bohr"], dtype=np.float64)
    with np.load(str(files["qm_mep"]["path"])) as state:
        reference = np.asarray(
            state["surface_potential_hartree_per_e"], dtype=np.float64
        )
        positions_angstrom = np.asarray(
            state["atom_positions_angstrom"], dtype=np.float64
        )
        atomic_numbers = np.asarray(state["atomic_numbers"], dtype=np.int64)
    atoms = Atoms(numbers=atomic_numbers, positions=positions_angstrom)
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    )
    surface = build_route2_smd_exterior_probe_surface(
        positions_angstrom, radii, clearance_angstrom=1.0
    )
    if (
        surface.surface_points_bohr.shape != points.shape
        or np.max(np.abs(surface.surface_points_bohr - points)) > 1.0e-7
    ):
        raise RuntimeError("frozen geometry-only probe surface did not replay.")

    auxiliary = df.addons.make_auxmol(
        molecule, auxbasis=df.make_auxbasis(molecule)
    )
    metric = auxiliary.intor("int2c2e")
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    retained = eigenvalues > (
        float(np.max(eigenvalues)) * METRIC_EIGENVALUE_RELATIVE_CUTOFF
    )
    inverse = eigenvectors[:, retained] @ (
        eigenvectors[:, retained].T / eigenvalues[retained, None]
    )
    three_center = df.incore.aux_e2(
        molecule, auxiliary, intor="int3c2e", aosym="s1"
    )
    rhs = np.einsum("ijp,ij->p", three_center, density, optimize=True)
    raw = inverse @ rhs

    grids = dft.gen_grid.Grids(auxiliary)
    grids.level = MOMENT_GRID_LEVEL
    grids.build(with_non0tab=False)
    charge_moment = np.zeros(auxiliary.nao_nr())
    first_moment = np.zeros((3, auxiliary.nao_nr()))
    for start in range(0, len(grids.coords), 30000):
        stop = min(start + 30000, len(grids.coords))
        coordinates = grids.coords[start:stop]
        weights = grids.weights[start:stop]
        ao = dft.numint.eval_ao(auxiliary, coordinates)
        charge_moment += np.einsum("g,gp->p", weights, ao)
        first_moment += np.einsum("g,gx,gp->xp", weights, coordinates, ao)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    position_integrals = molecule.intor_symmetric("int1e_r", comp=3)
    target = np.concatenate(
        (
            [float(np.einsum("ij,ji", density, overlap))],
            np.einsum("xij,ji->x", position_integrals, density),
        )
    )
    constraints = np.vstack((charge_moment, first_moment))
    residual = target - constraints @ raw
    schur = constraints @ inverse @ constraints.T
    constrained = raw + inverse @ constraints.T @ np.linalg.solve(schur, residual)

    point_charges = gto.fakemol_for_charges(points)
    potential_operator = gto.mole.intor_cross(
        "int2c2e", auxiliary, point_charges
    )
    nuclear = sum(
        charge / np.linalg.norm(points - center, axis=1)
        for charge, center in zip(
            molecule.atom_charges(), molecule.atom_coords()
        )
    )

    def metrics(values: np.ndarray) -> dict[str, float]:
        predicted = nuclear - values @ potential_operator
        difference = predicted - reference
        weights = surface.quadrature_weights
        return {
            "weighted_relative_l2": float(
                np.sqrt(
                    np.sum(weights * difference**2)
                    / np.sum(weights * reference**2)
                )
            ),
            "correlation": float(np.corrcoef(predicted, reference)[0, 1]),
            "maximum_absolute_error_hartree_per_e": float(
                np.max(np.abs(difference))
            ),
        }

    return {
        "compound_id": case["compound_id"],
        "name": case["name"],
        "chemical_class": case["chemical_class"],
        "functional_group": case["functional_group"],
        "atom_count": len(atoms),
        "auxiliary_dimension": auxiliary.nao_nr(),
        "metric_rank": int(np.count_nonzero(retained)),
        "metric_condition_number": float(
            np.max(eigenvalues[retained]) / np.min(eigenvalues[retained])
        ),
        "raw_moment_max_absolute_error": float(
            np.max(np.abs(constraints @ raw - target))
        ),
        "constrained_moment_max_absolute_error": float(
            np.max(np.abs(constraints @ constrained - target))
        ),
        "raw": metrics(raw),
        "constrained": metrics(constrained),
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    payload = _validate_preregistration(preregistration_path)
    result_path = Path(str(payload["result_path"])).resolve()
    if result_path.exists():
        raise FileExistsError(result_path)
    records = []
    for case in payload["cases"]:
        record = _project_case(case)
        records.append(record)
        print(
            f"{record['compound_id']} constrained_relative="
            f"{record['constrained']['weighted_relative_l2']:.8f}",
            flush=True,
        )
    values = np.asarray(
        [record["constrained"]["weighted_relative_l2"] for record in records]
    )
    constraint_errors = np.asarray(
        [record["constrained_moment_max_absolute_error"] for record in records]
    )
    mean_error = float(np.mean(values))
    maximum_error = float(np.max(values))
    result: dict[str, Any] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "pass" if (
            mean_error <= MEAN_RELATIVE_ERROR_MAX
            and maximum_error <= MAXIMUM_CASE_RELATIVE_ERROR_MAX
            and float(np.max(constraint_errors)) <= 1.0e-8
        ) else "fail",
        "preregistration_file_sha256": _sha256(preregistration_path),
        "preregistration_sha256": payload["preregistration_sha256"],
        "record_count": len(records),
        "records": records,
        "aggregate": {
            "mean_constrained_area_relative_error": mean_error,
            "maximum_constrained_area_relative_error": maximum_error,
            "maximum_constraint_absolute_error": float(
                np.max(constraint_errors)
            ),
        },
        "gates": {
            "mean": mean_error <= MEAN_RELATIVE_ERROR_MAX,
            "maximum": maximum_error <= MAXIMUM_CASE_RELATIVE_ERROR_MAX,
            "constraints": float(np.max(constraint_errors)) <= 1.0e-8,
        },
        "claim_boundary": payload["claim_boundary"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("preregister")
    create_parser.add_argument("--static-root", type=Path, required=True)
    create_parser.add_argument("--parent-execution", type=Path, required=True)
    create_parser.add_argument("--result", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    result = (
        create_preregistration(args) if args.command == "preregister" else run(args)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
