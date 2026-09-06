#!/usr/bin/env python3
"""Validate and seal one preregistered VQM24 localized-field QM record.

This is the post-Gate-0 batch sealer.  The original Gate-0 sealer is kept
immutable because its source hash is part of the first-record evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_nonuniform_qm_record.py"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-localized-field-qm-record-level3-prereg-v1"
)
PREREGISTRATION_STATUS = "locked-before-level3-record-qm-execution"
RESPONSE_ARTIFACT = (
    "route2-vqm24-qm-localized-point-charge-response-level3-v1"
)
RESULT_ARTIFACT = "route2-vqm24-localized-field-qm-record-level3-result-v1"
STEP_CONSISTENCY_MAXIMUM = 0.02
ELECTRON_COUNT_ERROR_MAXIMUM = 1.0e-7
DENSITY_BINDING_RESIDUAL_MAXIMUM = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path.resolve())


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0e-30)
    )


def _copy_exclusive(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


def _validate_preregistration(
    preregistration_path: Path,
) -> dict[str, Any]:
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status") != PREREGISTRATION_STATUS
    ):
        raise ValueError("VQM24 level-3 record preregistration is invalid.")
    for label, record in preregistration["input_files"].items():
        path = Path(str(record["path"])).resolve(strict=True)
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"VQM24 preregistered input drifted: {label}.")
    return preregistration


def _energy_diagnostics(
    *,
    zero_energy: float,
    steps: np.ndarray,
    perturbation_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    records = {
        (
            int(record["mode_index"]),
            float(record["step_e"]),
            int(record["sign"]),
        ): record
        for record in perturbation_records
    }
    diagnostics: list[dict[str, Any]] = []
    all_curvatures_negative = True
    for mode_index in range(4):
        values: list[dict[str, float]] = []
        for step in steps:
            key_negative = (mode_index, float(step), -1)
            key_positive = (mode_index, float(step), 1)
            if key_negative not in records or key_positive not in records:
                raise RuntimeError("VQM24 response omits a frozen perturbation.")
            negative = float(records[key_negative]["energy_hartree"])
            positive = float(records[key_positive]["energy_hartree"])
            slope = (positive - negative) / (2.0 * float(step))
            curvature = (positive + negative - 2.0 * zero_energy) / float(
                step * step
            )
            all_curvatures_negative &= curvature < 0.0
            values.append(
                {
                    "step_e": float(step),
                    "central_slope_hartree_per_e": slope,
                    "central_curvature_hartree_per_e2": curvature,
                }
            )
        diagnostics.append(
            {
                "mode_index": mode_index,
                "by_step": values,
                "slope_step_relative_difference": abs(
                    values[0]["central_slope_hartree_per_e"]
                    - values[1]["central_slope_hartree_per_e"]
                )
                / max(
                    abs(values[0]["central_slope_hartree_per_e"]),
                    abs(values[1]["central_slope_hartree_per_e"]),
                    1.0e-30,
                ),
                "curvature_step_relative_difference": abs(
                    values[0]["central_curvature_hartree_per_e2"]
                    - values[1]["central_curvature_hartree_per_e2"]
                )
                / max(
                    abs(values[0]["central_curvature_hartree_per_e2"]),
                    abs(values[1]["central_curvature_hartree_per_e2"]),
                    1.0e-30,
                ),
            }
        )
    return diagnostics, all_curvatures_negative


def seal(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    gas_directory = args.gas_directory.expanduser().resolve(strict=True)
    response_directory = args.response_directory.expanduser().resolve(strict=True)
    evidence_directory = args.evidence_directory.expanduser().resolve()
    if evidence_directory.exists():
        raise FileExistsError(evidence_directory)

    preregistration = _validate_preregistration(preregistration_path)
    expected_gas_directory = Path(
        preregistration["outputs"]["gas_directory"]
    ).resolve()
    expected_response_directory = Path(
        preregistration["outputs"]["response_directory"]
    ).resolve()
    if gas_directory != expected_gas_directory:
        raise ValueError("Gas directory differs from the preregistration.")
    if response_directory != expected_response_directory:
        raise ValueError("Response directory differs from the preregistration.")

    source_files = {
        "gas_json": gas_directory / "gas.json",
        "gas_checkpoint": gas_directory / "gas.chk",
        "response_json": response_directory / "response.json",
        "response_npz": response_directory / "response.npz",
    }
    for path in source_files.values():
        path.resolve(strict=True)

    gas = json.loads(source_files["gas_json"].read_text())
    response = json.loads(source_files["response_json"].read_text())
    if gas.get("status") != "pass":
        raise RuntimeError("VQM24 gas checkpoint helper did not pass.")
    if (
        response.get("artifact") != RESPONSE_ARTIFACT
        or response.get("status") != "success"
    ):
        raise RuntimeError("VQM24 localized response helper did not succeed.")
    if gas["numerics"]["nonlocal_grid_profile"] != "level":
        raise RuntimeError("Gas checkpoint does not use the frozen level profile.")
    if int(gas["numerics"]["nonlocal_grid_level"]) != 3:
        raise RuntimeError("Gas checkpoint does not use the frozen NLC level 3.")
    if response["method"]["nonlocal_grid_profile"] != "PySCF-level-3":
        raise RuntimeError("Response does not use the frozen PySCF level-3 grid.")

    input_files = preregistration["input_files"]
    if gas["input"]["mol2_sha256"] != input_files["mol2"]["sha256"]:
        raise RuntimeError("Gas checkpoint used a different geometry input.")
    if (
        gas["source"]["runner_sha256"]
        != preregistration["source"]["gas_runner"]["sha256"]
    ):
        raise RuntimeError("Gas runner differs from the preregistration.")
    gas_checkpoint_sha256 = _sha256(source_files["gas_checkpoint"])
    if gas_checkpoint_sha256 != gas["result"]["checkpoint"]["sha256"]:
        raise RuntimeError("Gas checkpoint hash differs from its ledger.")
    if response["input"]["checkpoint_sha256"] != gas_checkpoint_sha256:
        raise RuntimeError("Response used a different gas checkpoint.")
    if response["input"]["modes_npz_sha256"] != input_files["modes"]["sha256"]:
        raise RuntimeError("Response used different localized modes.")
    if response["input"]["surface_npz_sha256"] != input_files["surface"]["sha256"]:
        raise RuntimeError("Response used a different exterior surface.")
    if (
        response["source"]["runner_sha256"]
        != preregistration["source"]["response_runner"]["sha256"]
    ):
        raise RuntimeError("Response runner differs from the preregistration.")
    if _sha256(source_files["response_npz"]) != response["output"]["npz_sha256"]:
        raise RuntimeError("Response NPZ hash differs from its ledger.")

    with np.load(source_files["response_npz"], allow_pickle=False) as state:
        steps = np.asarray(state["field_steps_e"], dtype=np.float64)
        mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"],
            dtype=np.float64,
        )
    if (
        tuple(steps.tolist()) != (3.0e-4, 1.0e-3)
        or mep.ndim != 3
        or mep.shape[:2] != (2, 4)
        or dipole.shape != (2, 4, 3)
    ):
        raise RuntimeError("VQM24 response arrays violate the frozen protocol.")

    records = response["perturbation_records"]
    mep_step = _relative(mep[0], mep[1])
    dipole_step = _relative(dipole[0], dipole[1])
    maximum_electron_error = max(
        float(record["electron_count_error_e"]) for record in records
    )
    cycles = [int(record["scf_cycles"]) for record in records]
    energy_diagnostics, all_curvatures_negative = _energy_diagnostics(
        zero_energy=float(gas["result"]["energy_hartree"]),
        steps=steps,
        perturbation_records=records,
    )
    density_binding_residual = float(
        gas["result"]["checkpoint_density_binding_residual_inf"]
    )
    gates = {
        "mep_step_consistency": mep_step <= STEP_CONSISTENCY_MAXIMUM,
        "dipole_step_consistency": dipole_step <= STEP_CONSISTENCY_MAXIMUM,
        "electron_count": maximum_electron_error <= ELECTRON_COUNT_ERROR_MAXIMUM,
        "gas_density_binding": (
            density_binding_residual <= DENSITY_BINDING_RESIDUAL_MAXIMUM
        ),
        "all_scf_converged": len(records) == 16,
        "all_energy_curvatures_negative": all_curvatures_negative,
    }

    evidence_directory.mkdir(parents=True, exist_ok=False)
    copied = {
        "preregistration": evidence_directory / "preregistration.json",
        "gas_json": evidence_directory / "gas.json",
        "gas_checkpoint": evidence_directory / "gas.chk",
        "response_json": evidence_directory / "qm-response.json",
        "response_npz": evidence_directory / "qm-response.npz",
    }
    _copy_exclusive(preregistration_path, copied["preregistration"])
    for key, source in source_files.items():
        _copy_exclusive(source, copied[key])
    optional_logs = {
        "gas_log": gas_directory.parent / "gas-level3.run.log",
        "response_log": response_directory.parent / "response-level3.run.log",
    }
    for key, source in optional_logs.items():
        if source.is_file():
            copied[key] = evidence_directory / source.name
            _copy_exclusive(source, copied[key])

    record = preregistration["selection"]["record"]
    payload: dict[str, Any] = {
        "artifact": RESULT_ARTIFACT,
        "schema_version": 1,
        "status": "pass-numerical-gate-a-record" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "stratum": record["stratum"],
            "split": record["split"],
            "chemical_formula_hill": record["chemical_formula_hill"],
            "dataset_index": int(record["dataset_index"]),
            "record_sha256": record["record_sha256"],
        },
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "original_preregistration_path": str(preregistration_path),
            "original_preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
        },
        "files": {
            key: {"path": _portable_path(path), "sha256": _sha256(path)}
            for key, path in sorted(copied.items())
        },
        "gas": {
            "energy_hartree": float(gas["result"]["energy_hartree"]),
            "scf_cycles": int(gas["result"]["scf_cycles"]),
            "elapsed_seconds": float(gas["result"]["elapsed_seconds"]),
            "density_binding_residual_inf": density_binding_residual,
            "nonlocal_grid_point_count": int(
                gas["result"]["nonlocal_grid_point_count"]
            ),
        },
        "response": {
            "mode_count": 4,
            "field_steps_e": steps.tolist(),
            "surface_point_count": int(mep.shape[2]),
            "mep_step_relative_difference": mep_step,
            "dipole_step_relative_difference": dipole_step,
            "maximum_electron_count_error_e": maximum_electron_error,
            "minimum_scf_cycles": min(cycles),
            "maximum_scf_cycles": max(cycles),
            "elapsed_seconds": float(response["runtime"]["total_elapsed_seconds"]),
            "energy_diagnostics": energy_diagnostics,
        },
        "gates": gates,
        "claim_boundary": {
            "independent_qm_training_datum_generated": True,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    result_path = evidence_directory / "result.json"
    with result_path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    readme = evidence_directory / "README.md"
    readme.write_text(
        "# VQM24 nonuniform-field QM Gate-A record\n\n"
        f"- Stratum: `{record['stratum']}`\n"
        f"- Formula: `{record['chemical_formula_hill']}`\n"
        f"- Status: `{payload['status']}`\n"
        "- Electronic structure: `RKS omegaB97M-V/def2-TZVPD`\n"
        "- Semilocal/NLC grids: `PySCF level 3`\n\n"
        "This is independent gas-phase QM energy/MEP/dipole response data for "
        "the observable-supervised scalar-head program. It contains no VQM24 "
        "energy label, PCM/cavity target, experimental solvation target, fit, "
        "or MAPLE capability admission.\n"
    )
    copied["preregistration"].chmod(0o444)
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--gas-directory", type=Path, required=True)
    parser.add_argument("--response-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
