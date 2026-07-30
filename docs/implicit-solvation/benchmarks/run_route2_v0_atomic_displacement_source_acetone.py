#!/usr/bin/env python3
"""Falsify a frozen free-atom translation-tangent MACE-MDP source against QM MEP.

The candidate combines only two already frozen, non-fitted inputs:

* the MACE-MDP acetone polarizability plus exact atomic induced-dipole
  partition; and
* the v2 radial potential of an infinitesimally translated spherical
  free-atom Hartree-Fock electron cloud.

It compares the resulting response potential at every point in an immutable
516-point exterior QM-MEP validation set.  It never invokes a continuum,
changes a radial shape or a MACE weight, reads a solvation label, runs an
accuracy panel, or makes a force/PES claim.  A pass can only admit this exact
source to a separately preregistered same-basis common-scalar/KKT gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-v0-atomic-displacement-source-acetone-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-source-acetone-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_atomic_displacement_source_acetone.py"
)
SOURCE_MODULE_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_displacement_response.py"
)
TABLE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-hf-def2-tzvpd-v2.npz"
)
TABLE_MANIFEST_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-hf-def2-tzvpd-v2.json"
)
TABLE_PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-hf-def2-tzvpd-prereg-v2.json"
)
TABLE_AUDIT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-atomic-displacement-hf-def2-tzvpd-v2-reproducibility-audit.json"
)
ATOMIC_MAP_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json"
)
RESPONSE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-acetone-response-v1.json"
)
RAW_QM_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
)
POINTS_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-bohr.npy"
)
POINTS_PROVENANCE_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/reproducers/"
    "route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-provenance.json"
)
SOURCE_RELATIVE_PATHS = (RUNNER_RELATIVE_PATH, SOURCE_MODULE_RELATIVE_PATH)
INPUT_RELATIVE_PATHS = (
    TABLE_RELATIVE_PATH,
    TABLE_MANIFEST_RELATIVE_PATH,
    TABLE_PREREG_RELATIVE_PATH,
    TABLE_AUDIT_RELATIVE_PATH,
    ATOMIC_MAP_RELATIVE_PATH,
    RESPONSE_RELATIVE_PATH,
    RAW_QM_RELATIVE_PATH,
    POINTS_RELATIVE_PATH,
    POINTS_PROVENANCE_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
DIRECTIONS = ("x", "y", "z")
FIELD_STEPS = (3.0e-4, 1.0e-3)
SELECTED_FIELD_STEP = 3.0e-4
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17}
EXPECTED_QM_METHOD = {
    "electronic_structure": "omegaB97M-V",
    "pyscf_xc_token": "wb97m-v",
    "basis": "def2-tzvpd",
    "reference": "RKS",
    "density_fitting": True,
    "charge": 0,
    "spin": 0,
    "semilocal_grid_level": 3,
    "nonlocal_grid_profile": "50x194-SG1",
    "scf_energy_tolerance_hartree": 1.0e-10,
    "scf_gradient_tolerance": 1.0e-7,
    "maximum_scf_cycles": 100,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object.")
    return payload


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": bool(value <= maximum)}


def _require_clean_tracked_checkout() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The atomic-displacement QM-MEP falsifier requires a clean tracked "
            f"checkout; git reported:\n{status}"
        )
    for relative in (*SOURCE_RELATIVE_PATHS, *INPUT_RELATIVE_PATHS):
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _validate_preregistration() -> tuple[dict[str, Any], dict[str, str], str]:
    preregistration = _load_json(
        DEFAULT_PREREGISTRATION,
        label="atomic-displacement acetone QM-MEP preregistration",
    )
    if (
        preregistration.get("protocol_id")
        != "route2-v0-atomic-displacement-source-acetone-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The atomic-displacement source protocol is not frozen.")
    expected_sources = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }
    expected_inputs = {
        relative: _sha256(REPO_ROOT / relative) for relative in INPUT_RELATIVE_PATHS
    }
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict) or (
        contract.get("source_sha256") != expected_sources
        or contract.get("input_sha256") != expected_inputs
    ):
        raise RuntimeError("The frozen atomic-displacement QM-MEP inputs changed.")
    if preregistration.get("qm_method") != EXPECTED_QM_METHOD:
        raise RuntimeError("The frozen QM-MEP method changed after preregistration.")
    if preregistration.get("finite_field_protocol") != {
        "directions": list(DIRECTIONS),
        "field_steps_au": list(FIELD_STEPS),
        "selected_reporting_step_au": SELECTED_FIELD_STEP,
        "signs": [-1, 1],
    }:
        raise RuntimeError("The frozen QM-MEP finite-field protocol changed.")
    return preregistration, expected_inputs, _sha256(DEFAULT_PREREGISTRATION)


def _load_source_module() -> Any:
    path = REPO_ROOT / SOURCE_MODULE_RELATIVE_PATH
    specification = importlib.util.spec_from_file_location(
        "route2_v0_atomic_displacement_response_frozen_source",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the frozen atomic-displacement source module.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_atomic_displacement_table(source_module: Any) -> Any:
    table_path = REPO_ROOT / TABLE_RELATIVE_PATH
    manifest_path = REPO_ROOT / TABLE_MANIFEST_RELATIVE_PATH
    prereg_path = REPO_ROOT / TABLE_PREREG_RELATIVE_PATH
    audit_path = REPO_ROOT / TABLE_AUDIT_RELATIVE_PATH
    manifest = _load_json(manifest_path, label="atomic-displacement table manifest")
    prereg = _load_json(prereg_path, label="atomic-displacement table preregistration")
    audit = _load_json(audit_path, label="atomic-displacement table audit")
    if (
        manifest.get("artifact") != "route2-v0-atomic-displacement-hf-def2-tzvpd-v2"
        or manifest.get("status") != "pass"
        or manifest.get("table", {}).get("sha256") != _sha256(table_path)
        or manifest.get("preregistration", {}).get("sha256") != _sha256(prereg_path)
        or prereg.get("protocol_id")
        != "route2-v0-atomic-displacement-hf-def2-tzvpd-v2-prereg"
        or audit.get("decision", {}).get("status") != "pass"
        or audit.get("deterministic_repeat", {}).get("array_and_archive_byte_identity")
        is not True
        or audit.get("deterministic_repeat", {}).get("identical_table_sha256")
        != _sha256(table_path)
    ):
        raise RuntimeError("The v2 atomic-displacement table provenance is invalid.")
    try:
        archive = np.load(table_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot load the v2 atomic-displacement table.") from exc
    with archive:
        required = {"radial_grid_bohr", "atomic_numbers"}
        if not required.issubset(archive.files):
            raise RuntimeError("The v2 atomic-displacement table omits core arrays.")
        radial_grid = np.asarray(archive["radial_grid_bohr"], dtype=float)
        atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        enclosed = {
            int(number): np.asarray(
                archive[f"enclosed_electrons_Z{int(number)}"], dtype=float
            )
            for number in atomic_numbers
        }
    if manifest.get("radial_grid_sha256") != _sha256_array(radial_grid):
        raise RuntimeError("The v2 atomic-displacement radial grid hash is invalid.")
    records = manifest.get("results")
    if not isinstance(records, list) or len(records) != len(enclosed):
        raise RuntimeError("The v2 atomic-displacement records are incomplete.")
    by_number = {int(record["atomic_number"]): record for record in records}
    if set(by_number) != set(enclosed):
        raise RuntimeError("The v2 atomic-displacement element sets disagree.")
    for number, values in enclosed.items():
        if by_number[number].get("enclosed_electrons_sha256") != _sha256_array(values):
            raise RuntimeError(
                f"The v2 atomic-displacement enclosed-electron hash is invalid for Z={number}."
            )
    return source_module.Route2V0AtomicDisplacementResponseTable(
        radial_grid_bohr=radial_grid,
        enclosed_electrons_by_atomic_number=enclosed,
        table_sha256=_sha256(table_path),
        manifest_sha256=_sha256(manifest_path),
    )


def _load_mace_mdp_inputs() -> tuple[
    tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    atomic = _load_json(
        REPO_ROOT / ATOMIC_MAP_RELATIVE_PATH,
        label="frozen MACE-MDP atomic partition",
    )
    response = _load_json(
        REPO_ROOT / RESPONSE_RELATIVE_PATH,
        label="frozen MACE-MDP response",
    )
    if (
        atomic.get("artifact") != "route2-v0-mace-mdp-atomic-map-acetone-v1"
        or atomic.get("status") != "pass"
        or atomic.get("decision", {}).get("verdict")
        != "admit-atomic-moment-partition-for-source-map-gates-only"
        or response.get("artifact") != "route2-v0-mace-mdp-acetone-response-v1"
        or response.get("status") != "pass"
        or response.get("scientific_falsification", {}).get("verdict")
        != "admit-frozen-mace-mdp-response-coefficients-only"
    ):
        raise RuntimeError("The frozen MACE-MDP response inputs are not admissible.")
    atomic_system = atomic.get("system")
    response_system = response.get("system")
    decomposition = atomic.get("mace_mdp_atomic_decomposition")
    response_values = response.get("mace_mdp_response")
    if not all(
        isinstance(value, dict)
        for value in (atomic_system, response_system, decomposition, response_values)
    ):
        raise TypeError("The frozen MACE-MDP response payload is incomplete.")
    symbols = tuple(atomic_system.get("atom_symbols", ()))
    positions = np.asarray(atomic_system.get("positions_angstrom"), dtype=float)
    response_positions = np.asarray(
        response_system.get("positions_angstrom"), dtype=float
    )
    weights = np.asarray(decomposition.get("atomic_dipole_weights"), dtype=float)
    polarizability = np.asarray(
        response_values.get("polarizability_bohr3"), dtype=float
    )
    if (
        len(symbols) != 10
        or positions.shape != (10, 3)
        or not np.array_equal(response_positions, positions)
        or weights.shape != (10, 3, 3)
        or polarizability.shape != (3, 3)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(polarizability))
        or any(symbol not in ATOMIC_NUMBERS for symbol in symbols)
    ):
        raise RuntimeError("The frozen MACE-MDP acetone inputs are invalid.")
    return symbols, positions, weights, polarizability


def _load_frozen_qm_mep_points() -> np.ndarray:
    provenance_path = REPO_ROOT / POINTS_PROVENANCE_RELATIVE_PATH
    points_path = REPO_ROOT / POINTS_RELATIVE_PATH
    raw_path = REPO_ROOT / RAW_QM_RELATIVE_PATH
    provenance = _load_json(provenance_path, label="frozen exterior QM-MEP points")
    if (
        provenance.get("artifact")
        != "route2-v0-atomic-displacement-source-acetone-v1-frozen-exterior-qm-mep-points"
        or provenance.get("status") != "pass"
        or provenance.get("origin", {}).get("raw_qm_induced_mep_sha256")
        != _sha256(raw_path)
        or provenance.get("points", {}).get("file_sha256") != _sha256(points_path)
    ):
        raise RuntimeError("The frozen exterior QM-MEP point provenance is invalid.")
    try:
        points = np.asarray(np.load(points_path, allow_pickle=False), dtype=float)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot load frozen exterior QM-MEP points.") from exc
    if (
        points.shape != (516, 3)
        or not np.all(np.isfinite(points))
        or provenance.get("points", {}).get("array_sha256") != _sha256_array(points)
    ):
        raise RuntimeError("The frozen exterior QM-MEP point set is invalid.")
    return points


def _load_qm_responses(
    *,
    points: np.ndarray,
    positions: np.ndarray,
) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    raw_path = REPO_ROOT / RAW_QM_RELATIVE_PATH
    payload = _load_json(raw_path, label="frozen independent QM induced-MEP response")
    if (
        payload.get("status") != "pass"
        or payload.get("method") != EXPECTED_QM_METHOD
        or payload.get("finite_field_protocol")
        != {
            "directions": list(DIRECTIONS),
            "field_steps_au": list(FIELD_STEPS),
            "signs": [-1, 1],
        }
    ):
        raise RuntimeError("The frozen QM induced-MEP response is invalid.")
    raw_input = payload.get("input")
    raw_responses = payload.get("central_difference_responses")
    if not isinstance(raw_input, dict) or not isinstance(raw_responses, dict):
        raise TypeError("The frozen QM induced-MEP response is incomplete.")
    if (
        raw_input.get("source_point_count") != len(points)
        or raw_input.get("source_points_bohr_sha256") != _sha256_array(points)
        or not np.array_equal(
            np.asarray(raw_input.get("positions_angstrom"), dtype=float), positions
        )
    ):
        raise RuntimeError("The frozen QM induced-MEP geometry or point set changed.")
    parsed: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for step in FIELD_STEPS:
        records = raw_responses.get(f"{step:.1e}")
        if not isinstance(records, dict) or set(records) != set(DIRECTIONS):
            raise RuntimeError("The frozen QM induced-MEP directions are incomplete.")
        parsed[step] = {}
        for direction in DIRECTIONS:
            record = records[direction]
            if not isinstance(record, dict):
                raise TypeError("A frozen QM induced-MEP direction record is invalid.")
            potential = np.asarray(
                record.get("electronic_potential_response_hartree_per_e_per_field_au"),
                dtype=float,
            )
            dipole = np.asarray(
                record.get("molecular_dipole_response_bohr3"), dtype=float
            )
            if (
                potential.shape != (len(points),)
                or dipole.shape != (3,)
                or not np.all(np.isfinite(potential))
                or not np.all(np.isfinite(dipole))
            ):
                raise RuntimeError("A frozen QM induced-MEP response array is invalid.")
            parsed[step][direction] = {"potential": potential, "dipole": dipole}
    return parsed


def _response_tensor(
    responses: dict[float, dict[str, dict[str, np.ndarray]]],
    *,
    step: float,
    observable: str,
) -> np.ndarray:
    if observable not in {"potential", "dipole"}:
        raise ValueError(f"Unsupported frozen QM observable: {observable}.")
    try:
        tensor = np.column_stack(
            [responses[step][direction][observable] for direction in DIRECTIONS]
        )
    except KeyError as exc:
        raise RuntimeError("The frozen QM response tensor is incomplete.") from exc
    if tensor.ndim != 2 or not np.all(np.isfinite(tensor)):
        raise RuntimeError("The frozen QM response tensor is invalid.")
    return tensor


def _candidate_source_checks(
    *,
    weights: np.ndarray,
    polarizability: np.ndarray,
) -> dict[str, dict[str, float | bool]]:
    partition_error = float(
        np.linalg.norm(np.sum(weights, axis=0) - np.eye(3), ord="fro")
    )
    atom_partitioned = np.einsum("aij,jk->aik", weights, polarizability)
    response_error = float(
        np.linalg.norm(np.sum(atom_partitioned, axis=0) - polarizability, ord="fro")
    )
    checks = {
        "atomic_partition_moment_identity": {
            "value": partition_error,
            "maximum": 1.0e-12,
        },
        "atomic_partition_response_identity": {
            "value": response_error,
            "maximum": 1.0e-12,
        },
    }
    for check in checks.values():
        check["passes"] = bool(float(check["value"]) <= float(check["maximum"]))
    return checks


def main() -> int:
    arguments = _parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    git_head = _require_clean_tracked_checkout()
    preregistration, input_hashes, preregistration_sha = _validate_preregistration()
    source_module = _load_source_module()
    table = _load_atomic_displacement_table(source_module)
    symbols, positions, weights, polarizability = _load_mace_mdp_inputs()
    points = _load_frozen_qm_mep_points()
    qm_responses = _load_qm_responses(points=points, positions=positions)
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in symbols])
    source_checks = _candidate_source_checks(
        weights=weights,
        polarizability=polarizability,
    )
    numerical_gates = preregistration.get("numerical_gates")
    scientific_gates = preregistration.get("scientific_falsification_gates")
    if not isinstance(numerical_gates, dict) or not isinstance(scientific_gates, dict):
        raise TypeError("The source falsifier preregistration omits registered gates.")
    for name, check in source_checks.items():
        if float(check["maximum"]) != float(numerical_gates[f"{name}_max"]):
            raise RuntimeError(f"The registered {name} threshold changed.")

    candidate_potentials: dict[str, np.ndarray] = {}
    candidate_dipoles: dict[str, np.ndarray] = {}
    candidate_records: dict[str, dict[str, object]] = {}
    for direction_index, direction in enumerate(DIRECTIONS):
        field = np.zeros(3)
        field[direction_index] = 1.0
        induced_dipole = polarizability @ field
        potential = table.induced_potential(
            points,
            atomic_numbers,
            positions,
            weights,
            induced_dipole,
        )
        candidate_potentials[direction] = potential
        candidate_dipoles[direction] = induced_dipole
        candidate_records[direction] = {
            "candidate_induced_dipole_bohr3": induced_dipole.tolist(),
            "candidate_potential_response_sha256": _sha256_array(potential),
        }
    candidate_mep = np.column_stack(
        [candidate_potentials[direction] for direction in DIRECTIONS]
    )
    candidate_dipole = np.column_stack(
        [candidate_dipoles[direction] for direction in DIRECTIONS]
    )
    qm_mep = {
        step: _response_tensor(qm_responses, step=step, observable="potential")
        for step in FIELD_STEPS
    }
    qm_dipole = {
        step: _response_tensor(qm_responses, step=step, observable="dipole")
        for step in FIELD_STEPS
    }
    comparisons_by_step: dict[str, dict[str, object]] = {}
    for step in FIELD_STEPS:
        per_direction: dict[str, object] = {}
        for direction_index, direction in enumerate(DIRECTIONS):
            per_direction[direction] = {
                "qm_induced_dipole_bohr3": qm_dipole[step][:, direction_index].tolist(),
                "qm_potential_response_sha256": _sha256_array(
                    qm_mep[step][:, direction_index]
                ),
                "mep_response_relative_error": _relative_frobenius(
                    candidate_potentials[direction], qm_mep[step][:, direction_index]
                ),
                "dipole_response_relative_error": _relative_frobenius(
                    candidate_dipoles[direction], qm_dipole[step][:, direction_index]
                ),
            }
        comparisons_by_step[f"{step:.1e}"] = per_direction

    selected_comparison = comparisons_by_step[f"{SELECTED_FIELD_STEP:.1e}"]
    selected_direction_errors = [
        float(selected_comparison[direction]["mep_response_relative_error"])
        for direction in DIRECTIONS
    ]
    numerical_checks: dict[str, dict[str, float | bool]] = {
        **source_checks,
        "qm_mep_step_consistency_relative_frobenius": _upper_check(
            _relative_frobenius(qm_mep[FIELD_STEPS[0]], qm_mep[FIELD_STEPS[1]]),
            float(numerical_gates["qm_mep_step_consistency_relative_frobenius_max"]),
        ),
        "qm_dipole_step_consistency_relative_frobenius": _upper_check(
            _relative_frobenius(qm_dipole[FIELD_STEPS[0]], qm_dipole[FIELD_STEPS[1]]),
            float(numerical_gates["qm_dipole_step_consistency_relative_frobenius_max"]),
        ),
    }
    scientific_checks: dict[str, dict[str, float | bool]] = {
        "mep_response_relative_frobenius": _upper_check(
            _relative_frobenius(candidate_mep, qm_mep[SELECTED_FIELD_STEP]),
            float(scientific_gates["mep_response_relative_frobenius_max"]),
        ),
        "mep_response_relative_direction_max": _upper_check(
            max(selected_direction_errors),
            float(scientific_gates["mep_response_relative_direction_max"]),
        ),
        "induced_dipole_response_relative_frobenius": _upper_check(
            _relative_frobenius(candidate_dipole, qm_dipole[SELECTED_FIELD_STEP]),
            float(scientific_gates["induced_dipole_response_relative_frobenius_max"]),
        ),
    }
    numerical_pass = all(bool(check["passes"]) for check in numerical_checks.values())
    scientific_pass = all(bool(check["passes"]) for check in scientific_checks.values())
    passes_all = numerical_pass and scientific_pass
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass" if passes_all else "reject",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "source_files_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVE_PATHS
        },
        "input_files_sha256": input_hashes,
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "source_point_count": len(points),
            "source_points_bohr_sha256": _sha256_array(points),
        },
        "source_representation": {
            "construction": "frozen-free-atom-electron-translation-tangent",
            "radial_asset": "route2-v0-atomic-displacement-hf-def2-tzvpd-v2",
            "atomic_dipole_partition": "p_a=W_a p from frozen MACE-MDP atomic map; no reweighting",
            "molecular_dipole_response": "p=alpha_MDP f with the raw frozen MACE-MDP tensor; no symmetrization, rescaling, clipping, or fit",
            "potential": "sum_a (p_a dot (r-R_a)) N_Za(|r-R_a|)/(Z_a |r-R_a|^3)",
            "source_point_boundary": "All 516 frozen exterior points are QM-MEP validation points only; no PCM surface, cavity selection, or continuum response is evaluated.",
            "table_sha256": table.table_sha256,
            "table_manifest_sha256": table.manifest_sha256,
        },
        "candidate_by_direction": candidate_records,
        "comparison_by_qm_field_step": comparisons_by_step,
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-frozen-atomic-displacement-source-to-same-basis-common-scalar-kkt-gates-only"
                if passes_all
                else "reject-frozen-atomic-displacement-source"
            ),
            "admission_boundary": (
                "A pass admits only this exact frozen source to a separately "
                "registered same-basis common-scalar/KKT, continuum-duality, and "
                "force gate. It does not establish a molecular density, stable "
                "energy functional, PCM result, force/PES result, nonpolar term, "
                "or experimental solvation accuracy."
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_resolved_sha256": _sha256(Path(sys.executable).resolve()),
            "boundary": "This is an array-level gas-phase QM-MEP source falsifier using frozen raw QM data; it is not an end-to-end Route-2 or QM runtime comparison.",
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
