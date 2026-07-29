#!/usr/bin/env python3
"""Compare the frozen V0-Q monopole curvature with QM finite fields.

This source-bound acetone canary evaluates the gas-phase polarizability of the
published-hardness, monopole-only V0-Q tangent and an independent
omegaB97M-V/def2-TZVPD finite-field reference at exactly the same geometry.
It does not train or fine-tune anything, use an experimental solvation value,
or rescale the QEq response after seeing the comparison.

The result is a physical response falsifier, not a solvation benchmark.  Even
a passing polarizability screen would not supply a nonpolar functional, a
solution-phase force/PES, or a public Route-2 profile.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import ase
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_gas_response import (  # noqa: E402
    RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION,
    build_route2_v0_rappe_goddard_gas_monopole_response,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_monopole import (  # noqa: E402
    RAPPE_GODDARD_QEQ_PARAMETER_SHA256,
)


ARTIFACT_ID = "route2-v0-qeq-acetone-qm-field-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-qeq-acetone-qm-field-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_qeq_acetone_qm_field.py"
)
QM_HELPER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2_v0_qm_finite_field.py"
)
ARCHIVE_RELATIVE_DIRECTORY = (
    ".omx/benchmarks/"
    "route2-exact-gto-fixed-geometry-canary-v1-408ca3f-20260727/"
    "maple.out.implicit"
)
ARCHIVE_STATE_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/route2-state.npz"
ARCHIVE_RESULT_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/route2-result.json"
ARCHIVE_MANIFEST_RELATIVE_PATH = f"{ARCHIVE_RELATIVE_DIRECTORY}/manifest.json"
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    QM_HELPER_RELATIVE_PATH,
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_qeq_gas_response.py"
    ),
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/route2_v0_qeq_monopole.py",
    "maple/function/calculator/extra_correction/charge/data/qeq.dat",
)
INPUT_RELATIVE_PATHS = (
    ARCHIVE_STATE_RELATIVE_PATH,
    ARCHIVE_RESULT_RELATIVE_PATH,
    ARCHIVE_MANIFEST_RELATIVE_PATH,
)
DEFAULT_PREREGISTRATION = REPO_ROOT / PREREG_RELATIVE_PATH
DEFAULT_ARCHIVE_STATE = REPO_ROOT / ARCHIVE_STATE_RELATIVE_PATH
DEFAULT_ARCHIVE_RESULT = REPO_ROOT / ARCHIVE_RESULT_RELATIVE_PATH
DEFAULT_ARCHIVE_MANIFEST = REPO_ROOT / ARCHIVE_MANIFEST_RELATIVE_PATH
DEFAULT_PYSCF_PYTHON = Path(
    "/home/axie/.cache/maple-envs/pyscf-smd-qmref/bin/python"
)
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
EXPECTED_NUMERICAL_GATES = {
    "qm_antisymmetry_relative_frobenius_max": 0.02,
    "energy_dipole_diagonal_relative_max": 0.02,
    "field_step_consistency_relative_frobenius_max": 0.02,
    "qm_minimum_eigenvalue_bohr3_min": -0.01,
    "qeq_charge_constraint_residual_e_max": 1.0e-10,
    "qeq_stationarity_residual_hartree_per_e_max": 1.0e-10,
}
EXPECTED_SCIENTIFIC_GATES = {
    "relative_frobenius_mismatch_max": 0.2,
    "trace_ratio_min": 0.8,
    "trace_ratio_max": 1.2,
    "principal_value_relative_max": 0.3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--archive-state", type=Path, default=DEFAULT_ARCHIVE_STATE)
    parser.add_argument("--archive-result", type=Path, default=DEFAULT_ARCHIVE_RESULT)
    parser.add_argument("--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST)
    parser.add_argument("--pyscf-python", type=Path, default=DEFAULT_PYSCF_PYTHON)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=8000)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
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


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object.")
    return payload


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The V0-Q QM-field canary requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS}


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _require_leq(value: float, ceiling: float, *, name: str) -> None:
    if not np.isfinite(value) or value > ceiling:
        raise RuntimeError(f"{name} failed: {value:.16e} > {ceiling:.16e}.")


def _require_geq(value: float, floor: float, *, name: str) -> None:
    if not np.isfinite(value) or value < floor:
        raise RuntimeError(f"{name} failed: {value:.16e} < {floor:.16e}.")


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _validate_preregistration(
    path: Path,
    *,
    supplied_inputs: dict[str, Path],
    pyscf_python: Path,
    threads: int,
    max_memory_mb: int,
) -> tuple[dict[str, object], str]:
    if path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("Only the tracked acetone QM-field preregistration is allowed.")
    preregistration = _load_object(path, label="QM-field preregistration")
    if (
        preregistration.get("protocol_id")
        != "route2-v0-qeq-acetone-qm-field-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The V0-Q QM-field preregistration identity is invalid.")
    if preregistration.get("qm_method") != EXPECTED_QM_METHOD:
        raise RuntimeError("The preregistered QM method changed after review.")
    field_protocol = preregistration.get("finite_field_protocol")
    if (
        not isinstance(field_protocol, dict)
        or field_protocol.get("field_steps_au") != [3.0e-4, 1.0e-3]
        or field_protocol.get("directions") != list(DIRECTIONS)
        or field_protocol.get("signs") != [-1, 1]
        or field_protocol.get("selected_reporting_step_au") != 3.0e-4
    ):
        raise RuntimeError("The finite-field protocol changed after review.")
    if preregistration.get("numerical_gates") != EXPECTED_NUMERICAL_GATES:
        raise RuntimeError("The finite-field numerical gates changed after review.")
    scientific_gates = preregistration.get("scientific_falsification_gates")
    if not isinstance(scientific_gates, dict) or {
        key: scientific_gates.get(key) for key in EXPECTED_SCIENTIFIC_GATES
    } != EXPECTED_SCIENTIFIC_GATES:
        raise RuntimeError("The physical falsification gates changed after review.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("The QM-field preregistration omits its execution contract.")
    expected_sources = contract.get("source_sha256")
    if not isinstance(expected_sources, dict) or set(expected_sources) != set(
        SOURCE_RELATIVE_PATHS
    ):
        raise RuntimeError("The QM-field source binding set is incomplete.")
    for relative, expected in expected_sources.items():
        if _sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Tracked source {relative} changed after the freeze.")
    expected_inputs = contract.get("input_sha256")
    if not isinstance(expected_inputs, dict) or set(expected_inputs) != set(
        INPUT_RELATIVE_PATHS
    ):
        raise RuntimeError("The QM-field input binding set is incomplete.")
    for relative, supplied in supplied_inputs.items():
        if supplied.resolve() != (REPO_ROOT / relative).resolve():
            raise RuntimeError(f"The QM-field canary forbids substituting {relative}.")
        if _sha256(supplied) != expected_inputs[relative]:
            raise RuntimeError(f"Frozen input {relative} changed after the freeze.")

    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("The QM-field preregistration omits its runtime binding.")
    if (
        _sha256(pyscf_python.resolve()) != runtime.get("python_resolved_sha256")
        or threads != runtime.get("threads")
        or max_memory_mb != runtime.get("max_memory_mb")
    ):
        raise RuntimeError("The PySCF runtime does not match the preregistration.")
    return preregistration, _sha256(path)


def _validate_qm_helper_result(
    payload: dict[str, object],
    *,
    archive_manifest: Path,
    preregistration: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if payload.get("schema_version") != 1 or payload.get("status") != "pass":
        raise RuntimeError("The isolated QM finite-field helper did not pass.")
    if payload.get("method") != EXPECTED_QM_METHOD:
        raise RuntimeError("The isolated QM helper changed the frozen method.")
    expected_protocol = {
        "field_steps_au": [3.0e-4, 1.0e-3],
        "directions": list(DIRECTIONS),
        "signs": [-1, 1],
    }
    if payload.get("finite_field_protocol") != expected_protocol:
        raise RuntimeError("The isolated QM helper changed the field protocol.")

    helper_input = payload.get("input")
    if (
        not isinstance(helper_input, dict)
        or helper_input.get("manifest_sha256") != _sha256(archive_manifest)
    ):
        raise RuntimeError("The isolated QM helper used a different geometry source.")

    contract = preregistration["execution_contract"]
    expected_runtime = contract["runtime"]
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("pyscf") != expected_runtime["pyscf_version"]
        or runtime.get("python_resolved_sha256")
        != expected_runtime["python_resolved_sha256"]
        or runtime.get("pyscf_init_sha256")
        != expected_runtime["pyscf_init_sha256"]
        or runtime.get("threads") != expected_runtime["threads"]
    ):
        raise RuntimeError("The isolated QM helper runtime changed after the freeze.")

    zero_field = payload.get("zero_field")
    field_records = payload.get("field_records")
    if not isinstance(zero_field, dict) or not isinstance(field_records, list):
        raise RuntimeError("The isolated QM helper output is incomplete.")
    if len(field_records) != 12 or not all(
        isinstance(record, dict) for record in field_records
    ):
        raise RuntimeError("The isolated QM helper must return twelve field records.")
    observed_cases = {
        (record.get("step_au"), record.get("direction"), record.get("sign"))
        for record in field_records
    }
    expected_cases = {
        (step, direction, sign)
        for step in (3.0e-4, 1.0e-3)
        for direction in DIRECTIONS
        for sign in (-1, 1)
    }
    if observed_cases != expected_cases:
        raise RuntimeError("The isolated QM helper field cases are incomplete.")
    return zero_field, field_records


def _load_frozen_geometry(
    *,
    archive_state: Path,
    archive_result: Path,
    archive_manifest: Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    manifest = _load_object(archive_manifest, label="archived acetone manifest")
    symbols = manifest.get("elements")
    positions = np.asarray(manifest.get("positions_angstrom"), dtype=float)
    expected_symbols = ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H")
    if tuple(symbols) != expected_symbols or positions.shape != (10, 3):
        raise RuntimeError("The archived source is not the frozen acetone geometry.")
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("The archived acetone geometry contains nonfinite values.")
    result = _load_object(archive_result, label="archived MACE result")
    checkpoint = result.get("mace_polar_checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != (
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    ):
        raise RuntimeError("The frozen source is not the declared MACE-POLAR-1-M state.")
    with np.load(archive_state) as archive:
        density = np.asarray(archive["gas_density_coefficients"], dtype=float)
    if density.shape != (10, 4) or not np.all(np.isfinite(density)):
        raise RuntimeError("The frozen MACE gas density is invalid.")
    return expected_symbols, positions, density, checkpoint


def _finite_field_tensor(
    zero_energy: float,
    records: list[dict[str, object]],
    *,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    dipole_tensor = np.empty((3, 3), dtype=float)
    energy_diagonal = np.empty(3, dtype=float)
    for direction in range(3):
        plus = next(
            record
            for record in records
            if record["step_au"] == step
            and record["direction"] == DIRECTIONS[direction]
            and record["sign"] == 1
        )
        minus = next(
            record
            for record in records
            if record["step_au"] == step
            and record["direction"] == DIRECTIONS[direction]
            and record["sign"] == -1
        )
        plus_dipole = np.asarray(plus["dipole_e_bohr"], dtype=float)
        minus_dipole = np.asarray(minus["dipole_e_bohr"], dtype=float)
        dipole_tensor[:, direction] = (plus_dipole - minus_dipole) / (2.0 * step)
        energy_diagonal[direction] = -(
            float(plus["energy_hartree"])
            + float(minus["energy_hartree"])
            - 2.0 * zero_energy
        ) / step**2
    return dipole_tensor, energy_diagonal


def main() -> int:
    args = _parse_args()
    git_head = _require_clean_source()
    for name in (
        "preregistration",
        "archive_state",
        "archive_result",
        "archive_manifest",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    pyscf_python = args.pyscf_python.absolute()
    if not pyscf_python.is_file():
        raise FileNotFoundError(pyscf_python)
    args.pyscf_python = pyscf_python
    if args.threads < 1 or args.max_memory_mb < 1:
        raise ValueError("Threads and memory must be positive.")
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if output.exists() or work_dir.exists():
        raise FileExistsError(output if output.exists() else work_dir)
    supplied_inputs = {
        ARCHIVE_STATE_RELATIVE_PATH: args.archive_state,
        ARCHIVE_RESULT_RELATIVE_PATH: args.archive_result,
        ARCHIVE_MANIFEST_RELATIVE_PATH: args.archive_manifest,
    }
    preregistration, preregistration_sha = _validate_preregistration(
        args.preregistration,
        supplied_inputs=supplied_inputs,
        pyscf_python=args.pyscf_python,
        threads=args.threads,
        max_memory_mb=args.max_memory_mb,
    )
    symbols, positions, frozen_density, checkpoint = _load_frozen_geometry(
        archive_state=args.archive_state,
        archive_result=args.archive_result,
        archive_manifest=args.archive_manifest,
    )
    method = preregistration["qm_method"]
    field_protocol = preregistration["finite_field_protocol"]
    numerical_gates = preregistration["numerical_gates"]
    scientific_gates = preregistration["scientific_falsification_gates"]
    field_steps = tuple(float(value) for value in field_protocol["field_steps_au"])
    if field_steps != (3.0e-4, 1.0e-3):
        raise RuntimeError("The finite-field steps do not match the frozen protocol.")

    work_dir.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    qm_raw_path = work_dir / "qm-field.json"
    helper_stdout_path = work_dir / "qm-field.stdout.json"
    helper_stderr_path = work_dir / "qm-field.stderr.log"
    helper_command = [
        str(args.pyscf_python),
        str(REPO_ROOT / QM_HELPER_RELATIVE_PATH),
        "--manifest",
        str(args.archive_manifest),
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
    qm_raw = _load_object(qm_raw_path, label="isolated QM finite-field result")
    zero_record, field_records = _validate_qm_helper_result(
        qm_raw,
        archive_manifest=args.archive_manifest,
        preregistration=preregistration,
    )
    zero_energy = float(zero_record["energy_hartree"])
    zero_dipole = np.asarray(zero_record["dipole_e_bohr"], dtype=float)
    zero_cycles = int(zero_record["scf_cycles"])
    zero_elapsed = float(zero_record["elapsed_seconds"])

    qm_by_step: dict[str, dict[str, object]] = {}
    raw_qm_tensors: dict[float, np.ndarray] = {}
    for step in field_steps:
        tensor, energy_diagonal = _finite_field_tensor(
            zero_energy,
            field_records,
            step=step,
        )
        symmetric = 0.5 * (tensor + tensor.T)
        antisymmetry_relative = float(
            np.linalg.norm(0.5 * (tensor - tensor.T))
            / max(np.linalg.norm(symmetric), 1.0e-30)
        )
        energy_dipole_relative = float(
            np.max(
                np.abs(energy_diagonal - np.diag(symmetric))
                / np.maximum(np.abs(np.diag(symmetric)), 1.0e-30)
            )
        )
        raw_qm_tensors[step] = tensor
        qm_by_step[f"{step:.1e}"] = {
            "dipole_derivative_tensor_bohr3": tensor.tolist(),
            "symmetric_tensor_bohr3": symmetric.tolist(),
            "energy_second_derivative_diagonal_bohr3": energy_diagonal.tolist(),
            "antisymmetry_relative_frobenius": antisymmetry_relative,
            "energy_dipole_diagonal_relative_max": energy_dipole_relative,
            "symmetric_eigenvalues_bohr3": np.linalg.eigvalsh(symmetric).tolist(),
        }

    selected_step = min(field_steps)
    selected_raw = raw_qm_tensors[selected_step]
    qm_polarizability = 0.5 * (selected_raw + selected_raw.T)
    step_consistency = _relative_frobenius(
        raw_qm_tensors[min(field_steps)],
        raw_qm_tensors[max(field_steps)],
    )
    selected_record = qm_by_step[f"{selected_step:.1e}"]
    _require_leq(
        selected_record["antisymmetry_relative_frobenius"],
        numerical_gates["qm_antisymmetry_relative_frobenius_max"],
        name="QM polarizability antisymmetry",
    )
    _require_leq(
        selected_record["energy_dipole_diagonal_relative_max"],
        numerical_gates["energy_dipole_diagonal_relative_max"],
        name="QM energy/dipole finite-field consistency",
    )
    _require_leq(
        step_consistency,
        numerical_gates["field_step_consistency_relative_frobenius_max"],
        name="QM finite-field step consistency",
    )
    _require_geq(
        float(np.min(np.linalg.eigvalsh(qm_polarizability))),
        numerical_gates["qm_minimum_eigenvalue_bohr3_min"],
        name="QM minimum polarizability eigenvalue",
    )

    qeq_response = build_route2_v0_rappe_goddard_gas_monopole_response(
        symbols,
        positions,
    )
    qeq_charge_residual = 0.0
    qeq_stationarity_residual = 0.0
    for direction in range(3):
        field = np.zeros(3)
        field[direction] = selected_step
        state = qeq_response.solve_uniform_field(field)
        qeq_charge_residual = max(
            qeq_charge_residual,
            state.charge_constraint_residual_e,
        )
        qeq_stationarity_residual = max(
            qeq_stationarity_residual,
            state.stationarity_residual_inf_hartree_per_e,
        )
    _require_leq(
        qeq_charge_residual,
        numerical_gates["qeq_charge_constraint_residual_e_max"],
        name="QEq charge constraint",
    )
    _require_leq(
        qeq_stationarity_residual,
        numerical_gates["qeq_stationarity_residual_hartree_per_e_max"],
        name="QEq stationarity",
    )

    qeq_polarizability = qeq_response.polarizability_bohr3
    qm_eigenvalues = np.linalg.eigvalsh(qm_polarizability)
    qeq_eigenvalues = np.linalg.eigvalsh(qeq_polarizability)
    relative_frobenius_mismatch = _relative_frobenius(
        qeq_polarizability,
        qm_polarizability,
    )
    trace_ratio = float(np.trace(qeq_polarizability) / np.trace(qm_polarizability))
    principal_value_relative_max = float(
        np.max(
            np.abs(qeq_eigenvalues - qm_eigenvalues)
            / np.maximum(np.abs(qm_eigenvalues), 1.0e-30)
        )
    )
    scientific_checks = {
        "relative_frobenius_mismatch": {
            "value": relative_frobenius_mismatch,
            "maximum": scientific_gates["relative_frobenius_mismatch_max"],
            "passes": relative_frobenius_mismatch
            <= scientific_gates["relative_frobenius_mismatch_max"],
        },
        "trace_ratio": {
            "value": trace_ratio,
            "minimum": scientific_gates["trace_ratio_min"],
            "maximum": scientific_gates["trace_ratio_max"],
            "passes": scientific_gates["trace_ratio_min"]
            <= trace_ratio
            <= scientific_gates["trace_ratio_max"],
        },
        "principal_value_relative_max": {
            "value": principal_value_relative_max,
            "maximum": scientific_gates["principal_value_relative_max"],
            "passes": principal_value_relative_max
            <= scientific_gates["principal_value_relative_max"],
        },
    }
    physical_pass = all(check["passes"] for check in scientific_checks.values())
    total_elapsed = time.perf_counter() - total_started

    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "status": "pass",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "preregistration": {
            "path": PREREG_RELATIVE_PATH,
            "sha256": preregistration_sha,
            "protocol_id": preregistration["protocol_id"],
        },
        "claim_boundary": (
            "This one-acetone gas-phase finite-field comparison is a physical "
            "falsifier for the frozen no-training QEq-monopole V0-Q curvature. "
            "It uses no experimental solvation value and cannot certify total "
            "solvation accuracy, a nonpolar functional, solution-phase forces, "
            "a smooth PES, optimization, scan, MD, NVE, or a public profile."
        ),
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
            "frozen_mace_gas_density_sha256": _sha256_array(frozen_density),
            "mace_checkpoint": checkpoint,
        },
        "qm_reference": {
            "method": method,
            "zero_field": {
                "energy_hartree": zero_energy,
                "dipole_e_bohr": zero_dipole.tolist(),
                "scf_cycles": zero_cycles,
                "elapsed_seconds": zero_elapsed,
                "semilocal_grid_point_count": int(
                    zero_record["semilocal_grid_point_count"]
                ),
                "nonlocal_grid_point_count": int(
                    zero_record["nonlocal_grid_point_count"]
                ),
            },
            "field_records": field_records,
            "polarizability_by_step": qm_by_step,
            "selected_step_au": selected_step,
            "selected_symmetric_polarizability_bohr3": qm_polarizability.tolist(),
            "selected_eigenvalues_bohr3": qm_eigenvalues.tolist(),
            "field_step_consistency_relative_frobenius": step_consistency,
        },
        "qeq_control": {
            "construction": RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION,
            "parameter_table_sha256": RAPPE_GODDARD_QEQ_PARAMETER_SHA256,
            "polarizability_bohr3": qeq_polarizability.tolist(),
            "eigenvalues_bohr3": qeq_eigenvalues.tolist(),
            "minimum_neutral_curvature_hartree_per_e2": (
                qeq_response.electronic_minimum_neutral_curvature_hartree_per_e2
            ),
            "charge_constraint_residual_e_max": qeq_charge_residual,
            "stationarity_residual_hartree_per_e_max": qeq_stationarity_residual,
        },
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": physical_pass,
            "verdict": (
                "provisional-one-geometry-screen-pass"
                if physical_pass
                else "reject-fixed-qeq-monopole-curvature"
            ),
            "decision_rule": preregistration["decision_rule"],
        },
        "runtime": {
            "main_python": platform.python_version(),
            "main_python_executable": str(Path(sys.executable).resolve()),
            "main_numpy": np.__version__,
            "ase": ase.__version__,
            "pyscf_helper": qm_raw["runtime"],
            "helper_command": helper_command,
            "helper_stdout_sha256": _sha256(helper_stdout_path),
            "helper_stderr_sha256": _sha256(helper_stderr_path),
            "raw_qm_sha256": _sha256(qm_raw_path),
            "total_elapsed_seconds": total_elapsed,
        },
        "official_references": {
            "pyscf_dipole_implementation": "https://pyscf.org/_modules/pyscf/scf/hf.html",
            "pyscf_integral_api": "https://pyscf.org/pyscf_api_docs/pyscf.gto.html",
            "rappe_goddard_qeq": "https://doi.org/10.1021/j100161a070",
            "qeq_polarizability_scaling_limit": "https://doi.org/10.1063/1.2908071",
            "acks2": "https://doi.org/10.1063/1.4791569",
        },
    }
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
