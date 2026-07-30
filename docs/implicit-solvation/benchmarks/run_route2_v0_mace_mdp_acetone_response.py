#!/usr/bin/env python3
"""Screen frozen MACE-MDP response coefficients against a QM field reference.

MACE-MDP is intentionally *not* treated as an energy or force model here.  It
only supplies a frozen molecular dipole/polarizability coefficient candidate
for a future, explicitly defined moment-space scalar.  This one-geometry,
gas-phase calculation neither invokes a continuum nor reads an experimental
solvation label, so it cannot make an accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_ID = "route2-v0-mace-mdp-acetone-response-v1"
PREREG_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-mace-mdp-acetone-response-prereg-v1.json"
)
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_v0_mace_mdp_acetone_response.py"
)
QM_ARTIFACT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/route2-v0-qeq-acetone-qm-field-v1.json"
)
DEFAULT_MODEL = Path("/home/axie/.cache/mace/MACE-MDP.model")

# CODATA values used to convert the MACE-MDP documented e Å²/V output into
# atomic-unit polarizability (bohr³).  We retain the derivation in the
# artifact instead of silently assuming the commonly quoted Å³ convention.
BOHR_RADIUS_METERS = 5.29177210903e-11
BOHR_RADIUS_ANGSTROM = 0.529177210903
HARTREE_JOULE = 4.3597447222071e-18
ELEMENTARY_CHARGE_COULOMB = 1.602176634e-19
ATOMIC_FIELD_VOLT_PER_ANGSTROM = (
    HARTREE_JOULE / (ELEMENTARY_CHARGE_COULOMB * BOHR_RADIUS_METERS) / 1.0e10
)
POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT = (
    ATOMIC_FIELD_VOLT_PER_ANGSTROM / BOHR_RADIUS_ANGSTROM
)
BOHR_PER_ANGSTROM = 1.0 / BOHR_RADIUS_ANGSTROM


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
            "The MACE-MDP coefficient screen requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    _git("ls-files", "--error-unmatch", RUNNER_RELATIVE_PATH)
    _git("ls-files", "--error-unmatch", PREREG_RELATIVE_PATH)
    return _git("rev-parse", "HEAD")


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _relative_antisymmetry(tensor: np.ndarray) -> float:
    return float(
        np.linalg.norm(tensor - tensor.T) / max(np.linalg.norm(tensor), 1.0e-30)
    )


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


def _validate_preregistration() -> tuple[dict[str, Any], dict[str, Any], str]:
    preregistration_path = REPO_ROOT / PREREG_RELATIVE_PATH
    qm_path = REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, label="MACE-MDP coefficient preregistration"
    )
    qm_artifact = _load_json(qm_path, label="frozen acetone QM artifact")

    if (
        preregistration.get("protocol_id")
        != "route2-v0-mace-mdp-acetone-response-prereg-v1"
        or preregistration.get("status") != "frozen-before-execution"
    ):
        raise RuntimeError("The MACE-MDP protocol is not the frozen preregistration.")
    contract = preregistration.get("execution_contract")
    if not isinstance(contract, dict):
        raise TypeError("The MACE-MDP preregistration omits its execution contract.")
    if contract.get("source_sha256") != {
        RUNNER_RELATIVE_PATH: _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH)
    }:
        raise RuntimeError("The MACE-MDP runner changed after preregistration.")
    if contract.get("input_sha256") != {QM_ARTIFACT_RELATIVE_PATH: _sha256(qm_path)}:
        raise RuntimeError("The frozen acetone QM input changed after preregistration.")

    model_contract = contract.get("model")
    if not isinstance(model_contract, dict):
        raise TypeError("The MACE-MDP preregistration omits model identity.")
    if (
        not DEFAULT_MODEL.is_file()
        or model_contract.get("path") != str(DEFAULT_MODEL)
        or model_contract.get("sha256") != _sha256(DEFAULT_MODEL)
        or model_contract.get("size_bytes") != DEFAULT_MODEL.stat().st_size
    ):
        raise RuntimeError("The MACE-MDP checkpoint does not match the freeze.")

    runtime_contract = contract.get("runtime")
    if not isinstance(runtime_contract, dict):
        raise TypeError("The MACE-MDP preregistration omits runtime identity.")
    versions = _runtime_versions()
    if runtime_contract.get("versions") != versions:
        raise RuntimeError("The MACE-MDP package runtime does not match the freeze.")
    if runtime_contract.get("device") != "cpu":
        raise RuntimeError("The MACE-MDP protocol permits CPU inference only.")
    if runtime_contract.get("default_dtype") != "float64":
        raise RuntimeError("The MACE-MDP protocol requires float64 inference.")
    if (
        runtime_contract.get("torch_force_no_weights_only_load") != "1"
        or os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1"
    ):
        raise RuntimeError(
            "Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 before starting Python; "
            "it is part of the frozen MACE-MDP deserialization runtime."
        )

    conversion = preregistration.get("unit_conversion")
    if not isinstance(conversion, dict):
        raise TypeError("The MACE-MDP preregistration omits unit conversion.")
    if not np.isclose(
        _number(
            conversion.get("eangstrom2_per_volt_to_bohr3"),
            label="MACE-MDP polarizability conversion",
        ),
        POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("The frozen MACE-MDP polarizability conversion changed.")

    if qm_artifact.get("artifact") != "route2-v0-qeq-acetone-qm-field-v1":
        raise RuntimeError("The frozen QM artifact identity is invalid.")
    if qm_artifact.get("status") != "pass":
        raise RuntimeError("The frozen QM reference did not pass its numerical gates.")
    return preregistration, qm_artifact, _sha256(preregistration_path)


def _frozen_geometry(qm_artifact: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray]:
    system = qm_artifact.get("system")
    if not isinstance(system, dict):
        raise TypeError("The frozen QM artifact has no system identity.")
    symbols = tuple(system.get("atom_symbols", ()))
    positions = np.asarray(system.get("positions_angstrom"), dtype=float)
    if symbols != ("C", "C", "O", "C", "H", "H", "H", "H", "H", "H"):
        raise RuntimeError("The frozen system is not the declared acetone geometry.")
    if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("The frozen acetone coordinates are invalid.")
    return symbols, positions


def _qm_reference(qm_artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    qm_reference = qm_artifact.get("qm_reference")
    if not isinstance(qm_reference, dict):
        raise TypeError("The frozen QM artifact has no polarizability reference.")
    polarizability = np.asarray(
        qm_reference.get("selected_symmetric_polarizability_bohr3"), dtype=float
    )
    zero_field = qm_reference.get("zero_field")
    if not isinstance(zero_field, dict):
        raise TypeError("The frozen QM artifact has no zero-field dipole.")
    dipole = np.asarray(zero_field.get("dipole_e_bohr"), dtype=float)
    if (
        polarizability.shape != (3, 3)
        or dipole.shape != (3,)
        or not np.all(np.isfinite(polarizability))
        or not np.all(np.isfinite(dipole))
    ):
        raise RuntimeError("The frozen QM response values are invalid.")
    return polarizability, dipole


def _run_mace_mdp(
    symbols: tuple[str, ...], positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    # These imports occur only after the preregistered runtime contract is
    # checked, so a different MACE type cannot silently enter the screen.
    from ase import Atoms

    mace_module = import_module("mace.calculators.mace")
    calculator_type = getattr(mace_module, "MACECalculator", None)
    if calculator_type is None:
        raise RuntimeError("The pinned MACE runtime has no MACECalculator.")

    initialization_started = time.perf_counter()
    calculator = calculator_type(
        model_paths=str(DEFAULT_MODEL),
        model_type="DipolePolarizabilityMACE",
        default_dtype="float64",
        device="cpu",
    )
    initialization_elapsed = time.perf_counter() - initialization_started
    atoms = Atoms(symbols=symbols, positions=positions)
    inference_started = time.perf_counter()
    dipole = np.asarray(calculator.get_property("dipole", atoms), dtype=float)
    polarizability = np.asarray(
        calculator.get_property("polarizability", atoms), dtype=float
    )
    inference_elapsed = time.perf_counter() - inference_started
    if (
        dipole.shape != (3,)
        or polarizability.shape != (3, 3)
        or not np.all(np.isfinite(dipole))
        or not np.all(np.isfinite(polarizability))
    ):
        raise RuntimeError("MACE-MDP did not return finite 3-vector/3x3 response.")
    return dipole, polarizability, initialization_elapsed, inference_elapsed


def _upper_bound_check(value: float, maximum: float) -> dict[str, float | bool]:
    return {"value": value, "maximum": maximum, "passes": value <= maximum}


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    git_head = _require_clean_source()
    preregistration, qm_artifact, preregistration_sha = _validate_preregistration()
    symbols, positions = _frozen_geometry(qm_artifact)
    qm_polarizability, qm_dipole = _qm_reference(qm_artifact)

    total_started = time.perf_counter()
    dipole_eangstrom, raw_polarizability, initialization_seconds, inference_seconds = (
        _run_mace_mdp(symbols, positions)
    )
    total_seconds = time.perf_counter() - total_started
    canonical_polarizability = 0.5 * (raw_polarizability + raw_polarizability.T)
    polarizability_bohr3 = (
        raw_polarizability * POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT
    )
    canonical_polarizability_bohr3 = 0.5 * (
        polarizability_bohr3 + polarizability_bohr3.T
    )
    dipole_ebohr = dipole_eangstrom * BOHR_PER_ANGSTROM
    candidate_eigenvalues = np.linalg.eigvalsh(canonical_polarizability_bohr3)
    qm_eigenvalues = np.linalg.eigvalsh(qm_polarizability)

    numerical_gates = preregistration.get("numerical_gates")
    scientific_gates = preregistration.get("scientific_falsification_gates")
    if not isinstance(numerical_gates, dict) or not isinstance(scientific_gates, dict):
        raise TypeError(
            "The MACE-MDP preregistration omits numerical/scientific gates."
        )
    numerical_checks: dict[str, dict[str, float | bool]] = {
        "raw_polarizability_antisymmetry": _upper_bound_check(
            _relative_antisymmetry(raw_polarizability),
            float(numerical_gates["raw_antisymmetry_relative_frobenius_max"]),
        ),
        "minimum_canonical_eigenvalue": {
            "value": float(np.min(candidate_eigenvalues)),
            "minimum": float(numerical_gates["minimum_canonical_eigenvalue_bohr3_min"]),
            "passes": bool(
                float(np.min(candidate_eigenvalues))
                > float(numerical_gates["minimum_canonical_eigenvalue_bohr3_min"])
            ),
        },
    }
    scientific_checks: dict[str, dict[str, float | bool]] = {
        "relative_frobenius_mismatch": _upper_bound_check(
            _relative_frobenius(polarizability_bohr3, qm_polarizability),
            float(scientific_gates["relative_frobenius_mismatch_max"]),
        ),
        "trace_ratio": {
            "value": float(
                np.trace(polarizability_bohr3) / np.trace(qm_polarizability)
            ),
            "minimum": float(scientific_gates["trace_ratio_min"]),
            "maximum": float(scientific_gates["trace_ratio_max"]),
        },
        "principal_value_relative_max": _upper_bound_check(
            float(
                np.max(
                    np.abs(candidate_eigenvalues - qm_eigenvalues)
                    / np.maximum(np.abs(qm_eigenvalues), 1.0e-30)
                )
            ),
            float(scientific_gates["principal_value_relative_max"]),
        ),
    }
    trace_check = scientific_checks["trace_ratio"]
    trace_check["passes"] = bool(
        float(trace_check["minimum"])
        <= float(trace_check["value"])
        <= float(trace_check["maximum"])
    )
    numerical_pass = all(bool(check["passes"]) for check in numerical_checks.values())
    scientific_pass = all(bool(check["passes"]) for check in scientific_checks.values())
    passes_all = numerical_pass and scientific_pass

    qm_runtime = qm_artifact.get("runtime")
    qm_total_seconds = (
        None
        if not isinstance(qm_runtime, dict)
        else qm_runtime.get("total_elapsed_seconds")
    )
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
            QM_ARTIFACT_RELATIVE_PATH: _sha256(REPO_ROOT / QM_ARTIFACT_RELATIVE_PATH)
        },
        "claim_boundary": preregistration["claim_boundary"],
        "hard_constraints": preregistration["hard_constraints"],
        "system": {
            "compound_id": "mobley_3867265",
            "name": "acetone",
            "atom_symbols": list(symbols),
            "positions_angstrom": positions.tolist(),
        },
        "mace_mdp_response": {
            "model_type": "DipolePolarizabilityMACE",
            "dipole_eangstrom": dipole_eangstrom.tolist(),
            "dipole_ebohr": dipole_ebohr.tolist(),
            "raw_polarizability_eangstrom2_per_volt": raw_polarizability.tolist(),
            "canonical_polarizability_eangstrom2_per_volt": (
                canonical_polarizability.tolist()
            ),
            "polarizability_bohr3": polarizability_bohr3.tolist(),
            "canonical_polarizability_bohr3": canonical_polarizability_bohr3.tolist(),
            "canonical_eigenvalues_bohr3": candidate_eigenvalues.tolist(),
            "canonicalization_boundary": (
                "The raw tensor is used for every acceptance comparison. The "
                "listed canonical tensor is only the float64 symmetric form "
                "needed for a later scalar-function construction, and is "
                "permitted only after the raw antisymmetry gate passes. It is "
                "not a response repair, rescaling, clipping, or fitted edit."
            ),
        },
        "qm_reference": {
            "method": qm_artifact["qm_reference"].get("method"),
            "selected_step_au": qm_artifact["qm_reference"].get("selected_step_au"),
            "dipole_ebohr": qm_dipole.tolist(),
            "selected_symmetric_polarizability_bohr3": qm_polarizability.tolist(),
            "selected_eigenvalues_bohr3": qm_eigenvalues.tolist(),
            "source_artifact": QM_ARTIFACT_RELATIVE_PATH,
        },
        "non_gating_diagnostics": {
            "dipole_relative_error": float(
                np.linalg.norm(dipole_ebohr - qm_dipole)
                / max(np.linalg.norm(qm_dipole), 1.0e-30)
            ),
            "runtime": {
                "model_initialization_seconds": initialization_seconds,
                "coefficient_inference_seconds": inference_seconds,
                "screen_total_seconds": total_seconds,
                "frozen_qm_reference_total_seconds": qm_total_seconds,
                "screen_to_qm_runtime_ratio": (
                    None
                    if not isinstance(qm_total_seconds, (int, float))
                    or qm_total_seconds <= 0.0
                    else total_seconds / float(qm_total_seconds)
                ),
                "boundary": (
                    "This is a single fixed-geometry coefficient-screen timing, "
                    "not an end-to-end implicit-solvation runtime or throughput "
                    "claim."
                ),
            },
        },
        "numerical_checks": numerical_checks,
        "scientific_falsification": {
            "checks": scientific_checks,
            "passes_all_registered_checks": passes_all,
            "verdict": (
                "admit-frozen-mace-mdp-response-coefficients-only"
                if passes_all
                else "reject-frozen-mace-mdp-response-coefficients"
            ),
            "admission_boundary": (
                "Admission is limited to defining and testing a new explicit "
                "moment-space scalar with an energy-conjugate source map, a "
                "common continuum functional, coordinate derivatives, and force "
                "gates. MACE-MDP is not thereby an energy/force model, and this "
                "does not permit an experimental-solvation accuracy panel."
            ),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "versions": _runtime_versions(),
            "device": "cpu",
            "default_dtype": "float64",
            "model_path": str(DEFAULT_MODEL),
            "model_sha256": _sha256(DEFAULT_MODEL),
            "model_size_bytes": DEFAULT_MODEL.stat().st_size,
            "torch_force_no_weights_only_load": os.environ.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
            ),
        },
        "unit_conversion": {
            "source_unit": "e angstrom^2 / volt",
            "target_unit": "bohr^3",
            "bohr_radius_meters": BOHR_RADIUS_METERS,
            "bohr_radius_angstrom": BOHR_RADIUS_ANGSTROM,
            "hartree_joule": HARTREE_JOULE,
            "elementary_charge_coulomb": ELEMENTARY_CHARGE_COULOMB,
            "atomic_field_volt_per_angstrom": ATOMIC_FIELD_VOLT_PER_ANGSTROM,
            "eangstrom2_per_volt_to_bohr3": (
                POLARIZABILITY_BOHR3_PER_EANGSTROM2_PER_VOLT
            ),
        },
    }
    _write_exclusive_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
