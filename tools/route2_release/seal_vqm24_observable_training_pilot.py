#!/usr/bin/env python3
"""Seal the one-step observable-record replay against fluorine Gate-A."""

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
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_observable_training_pilot.py"
RUNNER_REPO_PATH = "tools/route2_release/run_vqm24_observable_training_record.py"
ARTIFACT = "route2-vqm24-observable-training-fluorine-pilot-result-v1"


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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0e-30)
    )


def seal(args: argparse.Namespace) -> dict[str, Any]:
    pilot_json_path = args.pilot_json.expanduser().resolve(strict=True)
    pilot_npz_path = args.pilot_npz.expanduser().resolve(strict=True)
    finite_directory = args.finite_evidence.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    pilot = json.loads(pilot_json_path.read_text())
    finite = json.loads((finite_directory / "qm-response.json").read_text())
    if pilot.get("artifact") != "route2-vqm24-observable-training-record-level3-v1":
        raise ValueError("Observable pilot has the wrong artifact identity.")
    if pilot.get("status") != "success" or finite.get("status") != "success":
        raise RuntimeError("Observable pilot or finite-field oracle did not succeed.")
    if _sha256(pilot_npz_path) != pilot["output"]["npz_sha256"]:
        raise RuntimeError("Observable pilot NPZ differs from its ledger.")
    runner = SOURCE_ROOT / RUNNER_REPO_PATH
    if pilot["source"]["runner_sha256"] != _sha256(runner):
        raise RuntimeError("Observable pilot runner changed after execution.")

    with np.load(pilot_npz_path, allow_pickle=False) as state:
        pilot_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        pilot_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"],
            dtype=np.float64,
        )
        enthalpy_slope = np.asarray(
            state["central_enthalpy_slope_hartree_per_e"],
            dtype=np.float64,
        )
        zero_source_mep = np.asarray(
            state["zero_total_source_mep_hartree_per_e"],
            dtype=np.float64,
        )
    with np.load(finite_directory / "qm-response.npz", allow_pickle=False) as state:
        steps = np.asarray(state["field_steps_e"], dtype=np.float64)
        finite_mep = np.asarray(
            state["induced_surface_mep_hartree_per_e_per_source_e"],
            dtype=np.float64,
        )
        finite_dipole = np.asarray(
            state["induced_dipole_e_bohr_per_source_e"],
            dtype=np.float64,
        )
    if tuple(steps.tolist()) != (3.0e-4, 1.0e-3):
        raise RuntimeError("Fluorine oracle has different finite-field steps.")
    mep_replay = _relative(pilot_mep, finite_mep[1])
    dipole_replay = _relative(pilot_dipole, finite_dipole[1])
    conjugacy = _relative(enthalpy_slope, zero_source_mep)

    finite_records = {
        (int(record["mode_index"]), int(record["sign"])): record
        for record in finite["perturbation_records"]
        if float(record["step_e"]) == 1.0e-3
    }
    pilot_records = {
        (int(record["mode_index"]), int(record["sign"])): record
        for record in pilot["perturbation_records"]
    }
    energy_replay = max(
        abs(
            float(pilot_records[key]["electronic_scf_energy_hartree"])
            - float(finite_records[key]["energy_hartree"])
        )
        for key in finite_records
    )
    gates = {
        "mep_large_step_replay": mep_replay <= 1.0e-8,
        "dipole_large_step_replay": dipole_replay <= 1.0e-8,
        "electronic_energy_large_step_replay": energy_replay <= 1.0e-9,
        "total_enthalpy_conjugacy": conjugacy <= 1.0e-5,
        "electron_count": (
            float(pilot["numerical_checks"]["maximum_electron_count_error_e"])
            <= 1.0e-10
        ),
        "external_nuclear_coupling_present": bool(
            pilot["finite_field_protocol"][
                "external_nuclear_coupling_included_in_total_enthalpy"
            ]
        ),
        "no_curvature_label": not bool(
            pilot["finite_field_protocol"][
                "numerical_energy_curvature_label_emitted"
            ]
        ),
    }
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-observable-record-pilot" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "runner_path": RUNNER_REPO_PATH,
            "runner_sha256": _sha256(runner),
            "finite_evidence_path": str(finite_directory.relative_to(SOURCE_ROOT)),
            "finite_result_sha256": _sha256(finite_directory / "result.json"),
        },
        "metrics": {
            "mep_large_step_replay_relative": mep_replay,
            "dipole_large_step_replay_relative": dipole_replay,
            "electronic_energy_large_step_replay_abs_hartree": energy_replay,
            "enthalpy_slope_vs_zero_source_mep_relative": conjugacy,
            "maximum_electron_count_error_e": float(
                pilot["numerical_checks"]["maximum_electron_count_error_e"]
            ),
            "scf_cycle_range": [
                int(pilot["numerical_checks"]["minimum_scf_cycles"]),
                int(pilot["numerical_checks"]["maximum_scf_cycles"]),
            ],
            "wall_seconds": float(pilot["runtime"]["total_elapsed_seconds"]),
        },
        "gates": gates,
        "claim_boundary": {
            "observable_record_runner_validated_on_opened_pilot": all(gates.values()),
            "fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(pilot_json_path, output_directory / "observable.json")
    shutil.copy2(pilot_npz_path, output_directory / "observable.npz")
    result_path = output_directory / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result_path.chmod(0o444)
    (output_directory / "README.md").write_text(
        "# VQM24 observable-training record pilot\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "The new one-step record replays the already opened fluorine Gate-A "
        "large-step response and verifies the full molecular external-enthalpy "
        "conjugacy after adding the external point-charge--nuclear term. It "
        "does not train a model or admit a MAPLE capability.\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--pilot-npz", type=Path, required=True)
    parser.add_argument("--finite-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
