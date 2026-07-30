#!/usr/bin/env python3
"""Check frozen MACE-MDP property coordinate derivatives against full FD.

MACE-MDP is still not an energy or force model.  This audit only establishes
whether its documented calculator derivative path for dipole and
polarizability is numerically consistent enough to become an explicit
coefficient derivative in a later, separately defined stationary scalar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_ID = "route2-v0-mace-mdp-acetone-derivatives-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-mace-mdp-acetone-derivatives-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_v0_mace_mdp_acetone_derivatives.py"
)
RESPONSE_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-acetone-response-v1.json"
)
ATOMIC_MAP_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json"
)
QM_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)
DEFAULT_MODEL = Path("/home/axie/.cache/mace/MACE-MDP.model")
MACE_CALCULATOR_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/calculators/mace.py"
)
MACE_MODELS_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/modules/models.py"
)
MACE_UTILS_PATH = Path(
    "/home/axie/miniconda3/envs/maple/lib/python3.11/site-packages/"
    "mace/modules/utils.py"
)


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


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite numeric value.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The MACE-MDP derivative audit requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    _git("ls-files", "--error-unmatch", RUNNER_RELATIVE_PATH)
    _git("ls-files", "--error-unmatch", PREREG_RELATIVE_PATH)
    return _git("rev-parse", "HEAD")


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _relative_magnitude(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value) / max(np.linalg.norm(reference), 1.0e-30))


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _runtime_versions() -> dict[str, str]:
    return {
        "ase": metadata.version("ase"),
        "mace_torch": metadata.version("mace-torch"),
        "numpy": metadata.version("numpy"),
        "torch": metadata.version("torch"),
    }


def _validate_preregistration() -> tuple[dict[str, Any], str]:
    preregistration_path = REPO_ROOT / PREREG_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, label="MACE-MDP derivative preregistration"
    )
    if (
        preregistration.get("protocol_id")
        != "route2-v0-mace-mdp-acetone-derivatives-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The MACE-MDP derivative protocol is not frozen.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise TypeError("The derivative preregistration omits its contract.")
    if contract.get("source_sha256") != {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
    }:
        raise RuntimeError("The MACE-MDP derivative runner changed after freeze.")
    required_paths = (
        RESPONSE_ARTIFACT_RELATIVE_PATH,
        ATOMIC_MAP_ARTIFACT_RELATIVE_PATH,
        QM_ARTIFACT_RELATIVE_PATH,
    )
    for relative_path in required_paths:
        artifact = _load_json(REPO_ROOT / relative_path, label=relative_path)
        if artifact.get("status") != "pass":
            raise RuntimeError(f"Prerequisite artifact did not pass: {relative_path}")
    expected_inputs = {
        **{
            relative_path: _sha256(REPO_ROOT / relative_path)
            for relative_path in required_paths
        },
        str(MACE_CALCULATOR_PATH): _sha256(MACE_CALCULATOR_PATH),
        str(MACE_MODELS_PATH): _sha256(MACE_MODELS_PATH),
        str(MACE_UTILS_PATH): _sha256(MACE_UTILS_PATH),
    }
    if contract.get("input_sha256") != expected_inputs:
        raise RuntimeError("A MACE-MDP derivative input changed after freeze.")
    model = contract.get("model")
    if not isinstance(model, dict):
        raise TypeError("The derivative preregistration omits model identity.")
    if (
        not DEFAULT_MODEL.is_file()
        or model.get("path") != str(DEFAULT_MODEL)
        or model.get("sha256") != _sha256(DEFAULT_MODEL)
        or model.get("size_bytes") != DEFAULT_MODEL.stat().st_size
    ):
        raise RuntimeError("The MACE-MDP checkpoint does not match the freeze.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("The derivative preregistration omits runtime identity.")
    if (
        runtime.get("versions") != _runtime_versions()
        or runtime.get("device") != "cpu"
        or runtime.get("default_dtype") != "float64"
        or runtime.get("torch_force_no_weights_only_load") != "1"
        or os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1"
    ):
        raise RuntimeError("The MACE-MDP derivative runtime does not match freeze.")
    return preregistration, _sha256(preregistration_path)


def _frozen_geometry() -> tuple[tuple[str, ...], np.ndarray]:
    artifact = _load_json(
        REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH, label="frozen acetone QM artifact"
    )
    system = artifact.get("system")
    if not isinstance(system, dict):
        raise TypeError("The frozen QM artifact has no system identity.")
    symbols = tuple(system.get("atom_symbols", ()))
    positions = np.asarray(system.get("positions_angstrom"), dtype=float)
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The frozen system is not the declared acetone geometry.")
    if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen acetone geometry is invalid.")
    return symbols, positions


def _calculator() -> object:
    calculator_module = import_module("mace.calculators.mace")
    calculator_type = getattr(calculator_module, "MACECalculator", None)
    if calculator_type is None:
        raise RuntimeError("The pinned MACE runtime has no MACECalculator.")
    return calculator_type(
        model_paths=str(DEFAULT_MODEL),
        model_type="DipolePolarizabilityMACE",
        default_dtype="float64",
        device="cpu",
    )


def _property_pair(calculator: object, atoms: object) -> tuple[np.ndarray, np.ndarray]:
    get_property = getattr(calculator, "get_property", None)
    if get_property is None:
        raise RuntimeError("The pinned MACE calculator has no get_property method.")
    dipole = np.asarray(get_property("dipole", atoms), dtype=float)
    polarizability = np.asarray(get_property("polarizability", atoms), dtype=float)
    if (
        dipole.shape != (3,)
        or polarizability.shape != (3, 3)
        or not np.all(np.isfinite(dipole))
        or not np.all(np.isfinite(polarizability))
    ):
        raise RuntimeError("MACE-MDP did not return finite dipole/polarizability.")
    return dipole, polarizability


def _analytic_derivatives(
    calculator: object, atoms: object, atom_count: int
) -> tuple[np.ndarray, np.ndarray]:
    method = getattr(calculator, "get_dielectric_derivatives", None)
    if method is None:
        raise RuntimeError("The pinned MACE calculator has no dielectric derivatives.")
    result = method(atoms)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("MACE-MDP did not return dipole/polarizability derivatives.")
    dipole_derivative = np.asarray(result[0], dtype=float)
    polarizability_derivative = np.asarray(result[1], dtype=float)
    if (
        dipole_derivative.shape != (3, atom_count, 3)
        or polarizability_derivative.shape != (9, atom_count, 3)
        or not np.all(np.isfinite(dipole_derivative))
        or not np.all(np.isfinite(polarizability_derivative))
    ):
        raise RuntimeError(
            "MACE-MDP derivative shapes must be (3,n_atoms,3) and (9,n_atoms,3)."
        )
    return dipole_derivative, polarizability_derivative


def _finite_difference_derivatives(
    calculator: object,
    symbols: tuple[str, ...],
    positions: np.ndarray,
    step_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    from ase import Atoms

    atom_count = len(symbols)
    dipole_derivative = np.empty((3, atom_count, 3), dtype=float)
    polarizability_derivative = np.empty((9, atom_count, 3), dtype=float)
    for atom in range(atom_count):
        for axis in range(3):
            displacement = np.zeros_like(positions)
            displacement[atom, axis] = step_angstrom
            plus_dipole, plus_polarizability = _property_pair(
                calculator, Atoms(symbols=symbols, positions=positions + displacement)
            )
            minus_dipole, minus_polarizability = _property_pair(
                calculator, Atoms(symbols=symbols, positions=positions - displacement)
            )
            dipole_derivative[:, atom, axis] = (plus_dipole - minus_dipole) / (
                2.0 * step_angstrom
            )
            polarizability_derivative[:, atom, axis] = (
                plus_polarizability.reshape(-1) - minus_polarizability.reshape(-1)
            ) / (2.0 * step_angstrom)
    return dipole_derivative, polarizability_derivative


def _upper_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": value <= maximum}


def main() -> int:
    args = _parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    git_head = _require_clean_source()
    preregistration, preregistration_sha = _validate_preregistration()
    symbols, positions = _frozen_geometry()
    protocol = preregistration.get("finite_difference_protocol")
    gates = preregistration.get("numerical_gates")
    if not isinstance(protocol, dict) or not isinstance(gates, dict):
        raise TypeError("The derivative preregistration omits protocol/gates.")
    step = _number(protocol.get("coordinate_step_angstrom"), label="FD step")
    if step != 1.0e-4:
        raise RuntimeError(
            "The derivative finite-difference step changed after freeze."
        )

    total_started = time.perf_counter()
    calculator = _calculator()
    from ase import Atoms

    atoms = Atoms(symbols=symbols, positions=positions)
    public_dipole, public_polarizability = _property_pair(calculator, atoms)
    analytic_dipole, analytic_polarizability = _analytic_derivatives(
        calculator, atoms, len(symbols)
    )
    finite_dipole, finite_polarizability = _finite_difference_derivatives(
        calculator, symbols, positions, step
    )
    total_seconds = time.perf_counter() - total_started

    analytic_polarizability_tensor = analytic_polarizability.reshape(
        3, 3, len(symbols), 3
    )
    derivative_antisymmetry = _relative_norm(
        analytic_polarizability_tensor,
        np.swapaxes(analytic_polarizability_tensor, 0, 1),
    )
    dipole_translation = np.sum(analytic_dipole, axis=1)
    polarizability_translation = np.sum(analytic_polarizability, axis=1)
    checks = {
        "dipole_derivative_finite_difference": _upper_check(
            _relative_norm(analytic_dipole, finite_dipole),
            _number(
                gates.get("derivative_relative_frobenius_max"),
                label="derivative gate",
            ),
        ),
        "polarizability_derivative_finite_difference": _upper_check(
            _relative_norm(analytic_polarizability, finite_polarizability),
            _number(
                gates.get("derivative_relative_frobenius_max"),
                label="derivative gate",
            ),
        ),
        "polarizability_derivative_symmetry": _upper_check(
            derivative_antisymmetry,
            _number(
                gates.get("polarizability_derivative_antisymmetry_max"),
                label="polarizability symmetry gate",
            ),
        ),
        "dipole_translation_identity": _upper_check(
            _relative_magnitude(dipole_translation, analytic_dipole),
            _number(gates.get("translation_relative_max"), label="translation gate"),
        ),
        "polarizability_translation_identity": _upper_check(
            _relative_magnitude(polarizability_translation, analytic_polarizability),
            _number(gates.get("translation_relative_max"), label="translation gate"),
        ),
    }
    passes_all = all(bool(check["passes"]) for check in checks.values())
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
        "source_files_sha256": {
            RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
        },
        "input_files_sha256": {
            RESPONSE_ARTIFACT_RELATIVE_PATH: _sha256(
                REPO_ROOT / RESPONSE_ARTIFACT_RELATIVE_PATH
            ),
            ATOMIC_MAP_ARTIFACT_RELATIVE_PATH: _sha256(
                REPO_ROOT / ATOMIC_MAP_ARTIFACT_RELATIVE_PATH
            ),
            QM_ARTIFACT_RELATIVE_PATH: _sha256(REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH),
            str(MACE_CALCULATOR_PATH): _sha256(MACE_CALCULATOR_PATH),
            str(MACE_MODELS_PATH): _sha256(MACE_MODELS_PATH),
            str(MACE_UTILS_PATH): _sha256(MACE_UTILS_PATH),
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
        },
        "finite_difference_protocol": {
            "coordinate_step_angstrom": step,
            "coordinate_components": "all 3N Cartesian components",
            "central_difference_evaluations": 2 * len(symbols) * 3,
        },
        "mace_mdp_properties": {
            "dipole_eangstrom": public_dipole.tolist(),
            "polarizability_eangstrom2_per_volt": public_polarizability.tolist(),
            "analytic_dipole_derivative_e_per_component": analytic_dipole.tolist(),
            "finite_difference_dipole_derivative_e_per_component": (
                finite_dipole.tolist()
            ),
            "analytic_polarizability_derivative_eangstrom_per_volt": (
                analytic_polarizability.tolist()
            ),
            "finite_difference_polarizability_derivative_eangstrom_per_volt": (
                finite_polarizability.tolist()
            ),
            "dipole_translation_sum_e_per_angstrom": dipole_translation.tolist(),
            "polarizability_translation_sum_eangstrom_per_volt": (
                polarizability_translation.tolist()
            ),
        },
        "numerical_checks": checks,
        "decision": {
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-mace-mdp-property-coordinate-derivatives-only"
                if passes_all
                else "reject-mace-mdp-property-coordinate-derivatives"
            ),
            "admission_boundary": (
                "A pass admits only frozen dipole/polarizability coordinate "
                "derivatives as explicit coefficient derivatives in a later "
                "common scalar. It does not create a model energy, force, "
                "density/GTO map, continuum source, force certificate, or "
                "solvation-accuracy result."
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "versions": _runtime_versions(),
            "device": "cpu",
            "default_dtype": "float64",
            "model_path": str(DEFAULT_MODEL),
            "model_sha256": _sha256(DEFAULT_MODEL),
            "model_size_bytes": DEFAULT_MODEL.stat().st_size,
            "torch_force_no_weights_only_load": os.environ.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
            ),
            "total_seconds": total_seconds,
            "mace_calculator_source_sha256": _sha256(MACE_CALCULATOR_PATH),
            "mace_models_source_sha256": _sha256(MACE_MODELS_PATH),
            "mace_utils_source_sha256": _sha256(MACE_UTILS_PATH),
        },
    }
    _write_exclusive_json(output_path, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
