#!/usr/bin/env python3
"""Falsify or admit one frozen MACE-MDP Gaussian induced-density source.

The already-audited MACE-MDP molecular polarizability and atomic moment
partition define a neutral induced ``l<=1`` Gaussian coefficient map.  This
runner compares its external induced electrostatic potential against a new
fixed-geometry QM finite-field reference at a frozen exterior point set.

No continuum response, solvation energy, experimental solvation label, model
training, width tuning, response repair, or force claim is involved.  A pass
only permits the source map to enter the next common-scalar KKT gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase.units import Bohr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    gaussian_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_mace_mdp_moment_source import (
    atomic_partition_to_l1_gto_coefficients,
    induced_dipole_source_map,
)

ARTIFACT_ID = "route2-v0-mace-mdp-induced-source-acetone-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-mace-mdp-induced-source-acetone-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_mace_mdp_induced_source_acetone.py"
)
QM_HELPER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2_v0_qm_induced_mep.py"
)
SOURCE_MAP_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_mace_mdp_moment_source.py"
)
GTO_DENSITY_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/gto_density.py"
)
GTO_GALERKIN_RELATIVE_PATH = (
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py"
)
ATOMIC_MAP_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json"
)
RESPONSE_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-acetone-response-v1.json"
)
QM_POLARIZABILITY_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)
ARCHIVE_RELATIVE_DIRECTORY = (
    ".omx/benchmarks/"
    "route2-exact-gto-fixed-geometry-canary-v1-408ca3f-20260727/"
    "maple.out.implicit"
)
ARCHIVE_STATE_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/route2-state.npz"
ARCHIVE_MANIFEST_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/manifest.json"
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    QM_HELPER_RELATIVE_PATH,
    SOURCE_MAP_RELATIVE_PATH,
    GTO_DENSITY_RELATIVE_PATH,
    GTO_GALERKIN_RELATIVE_PATH,
)
INPUT_RELATIVE_PATHS = (
    ATOMIC_MAP_ARTIFACT_RELATIVE_PATH,
    RESPONSE_ARTIFACT_RELATIVE_PATH,
    QM_POLARIZABILITY_ARTIFACT_RELATIVE_PATH,
    ARCHIVE_STATE_RELATIVE_PATH,
    ARCHIVE_MANIFEST_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
DEFAULT_ATOMIC_MAP_ARTIFACT = REPO_ROOT / ATOMIC_MAP_ARTIFACT_RELATIVE_PATH
DEFAULT_RESPONSE_ARTIFACT = REPO_ROOT / RESPONSE_ARTIFACT_RELATIVE_PATH
DEFAULT_QM_POLARIZABILITY_ARTIFACT = REPO_ROOT / QM_POLARIZABILITY_ARTIFACT_RELATIVE_PATH
DEFAULT_ARCHIVE_STATE = REPO_ROOT / ARCHIVE_STATE_RELATIVE_PATH
DEFAULT_ARCHIVE_MANIFEST = REPO_ROOT / ARCHIVE_MANIFEST_RELATIVE_PATH
DEFAULT_PYSCF_PYTHON = Path("/home/axie/.cache/maple-envs/pyscf-smd-qmref/bin/python")
DIRECTIONS = ("x", "y", "z")
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
EXPECTED_FIELD_STEPS = (3.0e-4, 1.0e-3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--atomic-map-artifact", type=Path, default=DEFAULT_ATOMIC_MAP_ARTIFACT)
    parser.add_argument("--response-artifact", type=Path, default=DEFAULT_RESPONSE_ARTIFACT)
    parser.add_argument(
        "--qm-polarizability-artifact",
        type=Path,
        default=DEFAULT_QM_POLARIZABILITY_ARTIFACT,
    )
    parser.add_argument("--archive-state", type=Path, default=DEFAULT_ARCHIVE_STATE)
    parser.add_argument("--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST)
    parser.add_argument("--pyscf-python", type=Path, default=DEFAULT_PYSCF_PYTHON)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": bool(value <= maximum)}


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The induced-source QM canary requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _validate_preregistration(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str], str]:
    if args.preregistration.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("Only the tracked induced-source preregistration is allowed.")
    preregistration = _load_json(
        args.preregistration, label="MACE-MDP induced-source preregistration"
    )
    if (
        preregistration.get("protocol_id")
        != "route2-v0-mace-mdp-induced-source-acetone-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The MACE-MDP induced-source protocol is not frozen.")
    if preregistration.get("qm_method") != EXPECTED_QM_METHOD:
        raise RuntimeError("The induced-source QM method changed after freeze.")
    field_protocol = preregistration.get("finite_field_protocol")
    expected_field_protocol = {
        "field_units": "Hartree per e Bohr",
        "field_steps_au": list(EXPECTED_FIELD_STEPS),
        "directions": list(DIRECTIONS),
        "signs": [-1, 1],
        "selected_reporting_step_au": 3.0e-4,
    }
    if field_protocol != expected_field_protocol:
        raise RuntimeError("The induced-source finite-field protocol changed.")
    representation = preregistration.get("source_representation")
    if not isinstance(representation, dict):
        raise TypeError("The induced-source preregistration lacks a source map.")
    if (
        representation.get("sigma_angstrom") != MACE_POLAR_DENSITY_SIGMA_ANGSTROM
        or representation.get("radial_count") != 1
        or representation.get("induced_monopoles") != "exactly zero"
        or representation.get("molecular_dipole_input_unit") != "e bohr"
        or representation.get("coefficient_unit") != "e angstrom"
    ):
        raise RuntimeError("The induced-source representation changed after freeze.")

    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise TypeError("The induced-source preregistration lacks a contract.")
    expected_sources = {relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS}
    if contract.get("source_sha256") != expected_sources:
        raise RuntimeError("An induced-source implementation changed after freeze.")
    expected_inputs = {
        ATOMIC_MAP_ARTIFACT_RELATIVE_PATH: _sha256(args.atomic_map_artifact),
        RESPONSE_ARTIFACT_RELATIVE_PATH: _sha256(args.response_artifact),
        QM_POLARIZABILITY_ARTIFACT_RELATIVE_PATH: _sha256(
            args.qm_polarizability_artifact
        ),
        ARCHIVE_STATE_RELATIVE_PATH: _sha256(args.archive_state),
        ARCHIVE_MANIFEST_RELATIVE_PATH: _sha256(args.archive_manifest),
    }
    if contract.get("input_sha256") != expected_inputs:
        raise RuntimeError("A frozen induced-source input changed after freeze.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("The induced-source preregistration lacks a runtime contract.")
    if (
        runtime.get("pyscf_version") != "2.13.1"
        or runtime.get("python_resolved_sha256") != _sha256(args.pyscf_python)
        or runtime.get("threads") != args.threads
        or runtime.get("max_memory_mb") != args.max_memory_mb
    ):
        raise RuntimeError("The induced-source QM runtime changed after freeze.")
    return preregistration, expected_inputs, _sha256(args.preregistration)


def _load_geometry_and_points(
    manifest: Path,
    state: Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    payload = _load_json(manifest, label="frozen acetone source manifest")
    symbols = tuple(payload.get("elements", ()))
    positions = np.asarray(payload.get("positions_angstrom"), dtype=float)
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The induced-source state is not frozen acetone.")
    if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen source geometry is invalid.")
    with np.load(state) as archive:
        points = np.asarray(archive["cavity_centers_bohr"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise RuntimeError("The frozen exterior source-point array is invalid.")
    return symbols, positions, points


def _load_source_inputs(
    *,
    atomic_map: Path,
    response: Path,
    qm_polarizability: Path,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    atomic = _load_json(atomic_map, label="MACE-MDP atomic map")
    response_artifact = _load_json(response, label="MACE-MDP response artifact")
    qm_artifact = _load_json(qm_polarizability, label="QM polarizability artifact")
    if (
        atomic.get("artifact") != "route2-v0-mace-mdp-atomic-map-acetone-v1"
        or atomic.get("status") != "pass"
        or atomic.get("decision", {}).get("verdict")
        != "admit-atomic-moment-partition-for-source-map-gates-only"
    ):
        raise RuntimeError("The frozen atomic moment-partition artifact is invalid.")
    if (
        response_artifact.get("artifact") != "route2-v0-mace-mdp-acetone-response-v1"
        or response_artifact.get("status") != "pass"
        or response_artifact.get("scientific_falsification", {}).get("verdict")
        != "admit-frozen-mace-mdp-response-coefficients-only"
    ):
        raise RuntimeError("The frozen MACE-MDP response artifact is invalid.")
    if (
        qm_artifact.get("artifact") != "route2-v0-qeq-acetone-qm-field-v1"
        or qm_artifact.get("status") != "pass"
    ):
        raise RuntimeError("The frozen QM polarizability artifact is invalid.")
    atomic_system = atomic.get("system")
    if not isinstance(atomic_system, dict) or not np.array_equal(
        np.asarray(atomic_system.get("positions_angstrom"), dtype=float), positions
    ):
        raise RuntimeError("The atomic moment partition has a different geometry.")
    decomposition = atomic.get("mace_mdp_atomic_decomposition")
    response_values = response_artifact.get("mace_mdp_response")
    if not isinstance(decomposition, dict) or not isinstance(response_values, dict):
        raise TypeError("The frozen MACE-MDP artifact payloads are incomplete.")
    weights = np.asarray(decomposition.get("atomic_dipole_weights"), dtype=float)
    polarizability = np.asarray(
        response_values.get("canonical_polarizability_bohr3"), dtype=float
    )
    if weights.shape != (10, 3, 3) or polarizability.shape != (3, 3):
        raise RuntimeError("The frozen MACE-MDP source coefficients are invalid.")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(polarizability)):
        raise RuntimeError("The frozen MACE-MDP source coefficients are non-finite.")
    return weights, polarizability, qm_artifact


def _validate_qm_helper_result(
    payload: dict[str, Any],
    *,
    manifest: Path,
    state: Path,
    preregistration: dict[str, Any],
    pyscf_python: Path,
) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    if payload.get("schema_version") != 1 or payload.get("status") != "pass":
        raise RuntimeError("The isolated induced-MEP QM helper did not pass.")
    if payload.get("method") != preregistration["qm_method"]:
        raise RuntimeError("The isolated induced-MEP helper changed the QM method.")
    expected_protocol = {
        "field_steps_au": list(EXPECTED_FIELD_STEPS),
        "directions": list(DIRECTIONS),
        "signs": [-1, 1],
    }
    if payload.get("finite_field_protocol") != expected_protocol:
        raise RuntimeError("The isolated induced-MEP helper changed its protocol.")
    input_payload = payload.get("input")
    if not isinstance(input_payload, dict) or (
        input_payload.get("manifest_sha256") != _sha256(manifest)
        or input_payload.get("state_sha256") != _sha256(state)
    ):
        raise RuntimeError("The isolated induced-MEP helper used different inputs.")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("pyscf") != "2.13.1"
        or runtime.get("python_resolved_sha256") != _sha256(pyscf_python)
        or runtime.get("threads") != 8
    ):
        raise RuntimeError("The isolated induced-MEP helper runtime changed.")
    responses = payload.get("central_difference_responses")
    if not isinstance(responses, dict):
        raise TypeError("The induced-MEP helper omitted central differences.")
    parsed: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    point_count = int(input_payload.get("source_point_count", -1))
    for step in EXPECTED_FIELD_STEPS:
        by_direction = responses.get(f"{step:.1e}")
        if not isinstance(by_direction, dict) or set(by_direction) != set(DIRECTIONS):
            raise RuntimeError("The induced-MEP field directions are incomplete.")
        parsed[step] = {}
        for direction in DIRECTIONS:
            record = by_direction[direction]
            if not isinstance(record, dict):
                raise TypeError("The induced-MEP response record is invalid.")
            potential = np.asarray(
                record.get("electronic_potential_response_hartree_per_e_per_field_au"),
                dtype=float,
            )
            dipole = np.asarray(
                record.get("molecular_dipole_response_bohr3"), dtype=float
            )
            if (
                potential.shape != (point_count,)
                or dipole.shape != (3,)
                or not np.all(np.isfinite(potential))
                or not np.all(np.isfinite(dipole))
            ):
                raise RuntimeError("The induced-MEP response array is invalid.")
            parsed[step][direction] = {"potential": potential, "dipole": dipole}
    return parsed


def _qm_response_tensor(
    responses: dict[float, dict[str, dict[str, np.ndarray]]],
    *,
    step: float,
    observable: str,
) -> np.ndarray:
    """Stack a frozen QM response by Cartesian field direction.

    Keeping this extraction separate prevents a candidate response array from
    being accidentally substituted for the independent QM reproduction check.
    ``observable`` is deliberately limited to the two arrays emitted by the
    isolated helper.
    """

    if observable not in {"potential", "dipole"}:
        raise ValueError(f"Unsupported QM response observable: {observable}.")
    try:
        columns = [responses[step][axis][observable] for axis in DIRECTIONS]
    except KeyError as error:
        raise RuntimeError("The frozen QM response tensor is incomplete.") from error
    tensor = np.column_stack(columns)
    if tensor.ndim != 2 or not np.all(np.isfinite(tensor)):
        raise RuntimeError("The frozen QM response tensor is invalid.")
    return tensor


def _source_map_checks(
    *,
    weights: np.ndarray,
    positions: np.ndarray,
    points: np.ndarray,
    sigma_angstrom: float,
    duality_seed: int,
) -> tuple[dict[str, dict[str, float | bool]], np.ndarray]:
    basis = AtomCenteredL1GTOBasis((sigma_angstrom,))
    source_map = induced_dipole_source_map(weights)
    source_map_matrix = source_map.reshape(-1, 3)
    constraints = basis.molecular_charge_dipole_constraints(positions)
    source_moments = constraints @ source_map_matrix
    surface_operator = basis.surface_operator(points, positions)
    rng = np.random.default_rng(duality_seed)
    test_dipole = rng.normal(size=3)
    coefficients = atomic_partition_to_l1_gto_coefficients(weights, test_dipole)
    direct_surface = gaussian_multipole_potential(
        points,
        positions,
        coefficients[:, 0, :],
        sigma_angstrom=sigma_angstrom,
    )
    matrix_surface = surface_operator @ coefficients.reshape(-1)
    surface_cotangent = rng.normal(size=points.shape[0])
    dual_left = float(surface_cotangent @ (surface_operator @ coefficients.reshape(-1)))
    dual_right = float(coefficients.reshape(-1) @ (surface_operator.T @ surface_cotangent))
    checks = {
        "induced_charge_identity": {
            "value": float(np.max(np.abs(source_moments[0]))),
            "maximum": 1.0e-12,
        },
        "induced_dipole_identity_ebohr": {
            "value": float(
                np.linalg.norm(source_moments[1:] / Bohr - np.eye(3), ord="fro")
            ),
            "maximum": 1.0e-12,
        },
        "gaussian_surface_source_identity": {
            "value": _relative_frobenius(matrix_surface, direct_surface),
            "maximum": 1.0e-12,
        },
        "surface_source_duality_hartree": {
            "value": abs(dual_left - dual_right),
            "maximum": 1.0e-12,
        },
        "atomic_partition_moment_identity": {
            "value": float(np.linalg.norm(np.sum(weights, axis=0) - np.eye(3), ord="fro")),
            "maximum": 1.0e-12,
        },
    }
    for check in checks.values():
        check["passes"] = bool(float(check["value"]) <= float(check["maximum"]))
    return checks, source_map


def main() -> int:
    args = _parse_args()
    for name in (
        "preregistration",
        "atomic_map_artifact",
        "response_artifact",
        "qm_polarizability_artifact",
        "archive_state",
        "archive_manifest",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.pyscf_python = args.pyscf_python.absolute()
    if not args.pyscf_python.is_file():
        raise FileNotFoundError(args.pyscf_python)
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if output.exists() or work_dir.exists():
        raise FileExistsError(output if output.exists() else work_dir)
    git_head = _require_clean_source()
    preregistration, input_hashes, preregistration_sha = _validate_preregistration(args)
    symbols, positions, points = _load_geometry_and_points(
        args.archive_manifest, args.archive_state
    )
    del symbols
    weights, polarizability, qm_polarizability_artifact = _load_source_inputs(
        atomic_map=args.atomic_map_artifact,
        response=args.response_artifact,
        qm_polarizability=args.qm_polarizability_artifact,
        positions=positions,
    )
    numerical_gates = preregistration.get("numerical_gates")
    scientific_gates = preregistration.get("scientific_falsification_gates")
    if not isinstance(numerical_gates, dict) or not isinstance(scientific_gates, dict):
        raise TypeError("The induced-source preregistration omits its gates.")
    duality_seed = int(preregistration["source_representation"]["duality_seed"])
    source_checks, source_map = _source_map_checks(
        weights=weights,
        positions=positions,
        points=points,
        sigma_angstrom=MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
        duality_seed=duality_seed,
    )
    for name, check in source_checks.items():
        expected = float(numerical_gates[f"{name}_max"])
        if float(check["maximum"]) != expected:
            raise RuntimeError(f"The frozen {name} numerical threshold changed.")

    work_dir.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    qm_raw_path = work_dir / "qm-induced-mep.json"
    helper_stdout_path = work_dir / "qm-induced-mep.stdout.json"
    helper_stderr_path = work_dir / "qm-induced-mep.stderr.log"
    helper_command = [
        str(args.pyscf_python),
        str(REPO_ROOT / QM_HELPER_RELATIVE_PATH),
        "--manifest",
        str(args.archive_manifest),
        "--state",
        str(args.archive_state),
        "--threads",
        str(args.threads),
        "--max-memory-mb",
        str(args.max_memory_mb),
        "--output",
        str(qm_raw_path),
    ]
    with (
        helper_stdout_path.open("x", encoding="utf-8") as stdout_handle,
        helper_stderr_path.open("x", encoding="utf-8") as stderr_handle,
    ):
        subprocess.run(
            helper_command,
            cwd=work_dir,
            check=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    qm_raw = _load_json(qm_raw_path, label="isolated QM induced-MEP result")
    qm_responses = _validate_qm_helper_result(
        qm_raw,
        manifest=args.archive_manifest,
        state=args.archive_state,
        preregistration=preregistration,
        pyscf_python=args.pyscf_python,
    )

    mep_by_step: dict[float, np.ndarray] = {}
    dipole_by_step: dict[float, np.ndarray] = {}
    candidate_records: dict[str, dict[str, object]] = {}
    selected_step = float(preregistration["finite_field_protocol"]["selected_reporting_step_au"])
    for step in EXPECTED_FIELD_STEPS:
        mep_columns: list[np.ndarray] = []
        dipole_columns: list[np.ndarray] = []
        per_direction: dict[str, object] = {}
        for direction_index, direction in enumerate(DIRECTIONS):
            field = np.zeros(3)
            field[direction_index] = 1.0
            induced_dipole = polarizability @ field
            coefficients = atomic_partition_to_l1_gto_coefficients(
                weights, induced_dipole
            )
            candidate_potential = gaussian_multipole_potential(
                points,
                positions,
                coefficients[:, 0, :],
                sigma_angstrom=MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
            )
            qm_potential = qm_responses[step][direction]["potential"]
            qm_dipole = qm_responses[step][direction]["dipole"]
            mep_columns.append(candidate_potential)
            dipole_columns.append(induced_dipole)
            per_direction[direction] = {
                "candidate_induced_dipole_bohr3": induced_dipole.tolist(),
                "qm_induced_dipole_bohr3": qm_dipole.tolist(),
                "candidate_potential_response_sha256": _sha256_array(
                    candidate_potential
                ),
                "qm_potential_response_sha256": _sha256_array(qm_potential),
                "mep_response_relative_error": _relative_frobenius(
                    candidate_potential, qm_potential
                ),
                "dipole_response_relative_error": _relative_frobenius(
                    induced_dipole, qm_dipole
                ),
            }
        mep_by_step[step] = np.column_stack(mep_columns)
        dipole_by_step[step] = np.column_stack(dipole_columns)
        candidate_records[f"{step:.1e}"] = per_direction

    qm_mep_by_step = {
        step: _qm_response_tensor(qm_responses, step=step, observable="potential")
        for step in EXPECTED_FIELD_STEPS
    }
    qm_dipole_by_step = {
        step: _qm_response_tensor(qm_responses, step=step, observable="dipole")
        for step in EXPECTED_FIELD_STEPS
    }
    qm_mep_step_consistency = _relative_frobenius(
        qm_mep_by_step[EXPECTED_FIELD_STEPS[0]],
        qm_mep_by_step[EXPECTED_FIELD_STEPS[1]],
    )
    qm_dipole_step_consistency = _relative_frobenius(
        qm_dipole_by_step[EXPECTED_FIELD_STEPS[0]],
        qm_dipole_by_step[EXPECTED_FIELD_STEPS[1]],
    )
    prior_qm = qm_polarizability_artifact.get("qm_reference", {})
    prior_alpha = np.asarray(
        prior_qm.get("selected_symmetric_polarizability_bohr3"), dtype=float
    )
    if prior_alpha.shape != (3, 3):
        raise RuntimeError("The frozen prior QM polarizability is invalid.")
    current_qm_alpha = qm_dipole_by_step[selected_step]
    numerical_checks: dict[str, dict[str, float | bool]] = {
        **source_checks,
        "qm_mep_step_consistency_relative_frobenius": _upper_check(
            qm_mep_step_consistency,
            float(numerical_gates["qm_mep_step_consistency_relative_frobenius_max"]),
        ),
        "qm_dipole_step_consistency_relative_frobenius": _upper_check(
            qm_dipole_step_consistency,
            float(
                numerical_gates["qm_dipole_step_consistency_relative_frobenius_max"]
            ),
        ),
        "qm_dipole_reproduction_relative_frobenius": _upper_check(
            _relative_frobenius(current_qm_alpha, prior_alpha),
            float(
                numerical_gates["qm_dipole_reproduction_relative_frobenius_max"]
            ),
        ),
    }
    selected_direction_records = cast(
        dict[str, dict[str, object]], candidate_records[f"{selected_step:.1e}"]
    )
    selected_mep = mep_by_step[selected_step]
    selected_qm_mep = qm_mep_by_step[selected_step]
    selected_dipole = dipole_by_step[selected_step]
    selected_qm_dipole = qm_dipole_by_step[selected_step]
    per_direction_errors: list[float] = []
    for direction in DIRECTIONS:
        value = selected_direction_records[direction].get(
            "mep_response_relative_error"
        )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("The selected MEP direction record is invalid.")
        per_direction_errors.append(float(value))
    per_direction_mep_error = max(per_direction_errors)
    scientific_checks: dict[str, dict[str, float | bool]] = {
        "mep_response_relative_frobenius": _upper_check(
            _relative_frobenius(selected_mep, selected_qm_mep),
            float(scientific_gates["mep_response_relative_frobenius_max"]),
        ),
        "mep_response_relative_direction_max": _upper_check(
            per_direction_mep_error,
            float(scientific_gates["mep_response_relative_direction_max"]),
        ),
        "induced_dipole_response_relative_frobenius": _upper_check(
            _relative_frobenius(selected_dipole, selected_qm_dipole),
            float(
                scientific_gates[
                    "induced_dipole_response_relative_frobenius_max"
                ]
            ),
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
            relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
        },
        "input_files_sha256": input_hashes,
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "positions_angstrom": positions.tolist(),
            "source_point_count": int(points.shape[0]),
            "source_points_bohr_sha256": _sha256_array(points),
        },
        "source_representation": {
            **preregistration["source_representation"],
            "source_map_shape": list(source_map.shape),
            "source_map_sha256": _sha256_array(source_map),
            "mace_mdp_polarizability_bohr3": polarizability.tolist(),
            "source_point_boundary": (
                "The frozen points are an exterior QM validation set only. No "
                "continuum response, surface charge, PCM energy, cavity choice, "
                "or solvation calculation is evaluated here."
            ),
        },
        "qm_reference": {
            "method": qm_raw["method"],
            "selected_step_au": selected_step,
            "raw_helper": {
                "path": str(qm_raw_path),
                "sha256": _sha256(qm_raw_path),
                "total_elapsed_seconds": qm_raw["runtime"]["total_elapsed_seconds"],
            },
            "response_definition": qm_raw["response_definition"],
        },
        "candidate_by_field_step": candidate_records,
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-frozen-mace-mdp-one-radial-l1-source-to-common-scalar-kkt-gates-only"
                if passes_all
                else "reject-frozen-mace-mdp-one-radial-l1-source"
            ),
            "admission_boundary": (
                "A pass admits only this exact frozen one-radial neutral induced "
                "source map to a separately registered common-scalar KKT and "
                "continuum gate. It is not a general radial-density proof, an "
                "energy/force model, a smooth-cavity result, a nonpolar model, "
                "or an experimental solvation accuracy result."
            ),
        },
        "runtime": {
            "pyscf_python": str(args.pyscf_python),
            "pyscf_python_sha256": _sha256(args.pyscf_python),
            "threads": args.threads,
            "max_memory_mb": args.max_memory_mb,
            "total_elapsed_seconds": time.perf_counter() - total_started,
            "boundary": (
                "This is a QM source-falsification run, not an end-to-end Route-2 "
                "or MACE-versus-QM runtime comparison."
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
