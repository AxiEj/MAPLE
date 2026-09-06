#!/usr/bin/env python3
"""Validate unopened CPKS records against the frozen finite-field oracle."""

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
SELF_REPO_PATH = "tools/route2_release/seal_vqm24_static_cpks_gate_b.py"
PREREGISTRATION_ARTIFACT = "route2-vqm24-static-cpks-gate-b-prereg-v1"
CPKS_ARTIFACT = "route2-vqm24-static-cpks-observable-response-level3-v1"
ARTIFACT = "route2-vqm24-static-cpks-gate-b-result-v1"


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


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    return path.resolve(strict=True)


def _symmetric_relative(left: np.ndarray, right: np.ndarray) -> float:
    numerator = 2.0 * float(np.linalg.norm(left - right))
    denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
    return numerator / max(denominator, np.finfo(float).tiny)


def _finite_curvatures(
    gas: dict[str, Any],
    response: dict[str, Any],
    steps: np.ndarray,
) -> np.ndarray:
    records = {
        (
            int(record["mode_index"]),
            float(record["step_e"]),
            int(record["sign"]),
        ): record
        for record in response["perturbation_records"]
    }
    zero = float(gas["result"]["energy_hartree"])
    values = np.empty((len(steps), 4), dtype=np.float64)
    for step_index, step in enumerate(steps):
        for mode_index in range(4):
            negative = float(records[(mode_index, float(step), -1)]["energy_hartree"])
            positive = float(records[(mode_index, float(step), 1)]["energy_hartree"])
            values[step_index, mode_index] = (
                positive + negative - 2.0 * zero
            ) / float(step * step)
    return values


def _mode_relative_maximum(left: np.ndarray, right: np.ndarray) -> float:
    return max(
        _symmetric_relative(left[index], right[index])
        for index in range(len(left))
    )


def seal(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status")
        != "locked-before-unopened-cpks-validation"
    ):
        raise ValueError("Static CPKS Gate-B preregistration is invalid.")
    runner_path = _resolve_recorded_path(preregistration["source"]["runner_path"])
    if _sha256(runner_path) != preregistration["source"]["runner_sha256"]:
        raise RuntimeError("Static CPKS runner changed after preregistration.")

    gates_contract = preregistration["comparison_contract"]
    record_results: list[dict[str, Any]] = []
    copied_sources: list[tuple[Path, Path]] = []
    for record in preregistration["validation_records"]:
        stratum = str(record["stratum"])
        cpks_json_path = Path(record["outputs"]["json"]).resolve(strict=True)
        cpks_npz_path = Path(record["outputs"]["npz"]).resolve(strict=True)
        cpks = json.loads(cpks_json_path.read_text())
        if cpks.get("artifact") != CPKS_ARTIFACT or cpks.get("status") != "success":
            raise RuntimeError(f"Static CPKS helper failed for {stratum}.")
        if cpks["source"]["runner_sha256"] != preregistration["source"]["runner_sha256"]:
            raise RuntimeError(f"Static CPKS source drifted for {stratum}.")
        if _sha256(cpks_npz_path) != cpks["output"]["npz_sha256"]:
            raise RuntimeError(f"Static CPKS NPZ drifted for {stratum}.")
        for label, expected in record["inputs"].items():
            if _sha256(Path(expected["path"])) != expected["sha256"]:
                raise RuntimeError(f"Gate-B input drifted for {stratum}: {label}.")

        finite_result_path = _resolve_recorded_path(
            record["finite_evidence_result"]["path"]
        )
        if _sha256(finite_result_path) != record["finite_evidence_result"]["sha256"]:
            raise RuntimeError(f"Finite-field evidence drifted for {stratum}.")
        finite_directory = finite_result_path.parent
        finite_gas = json.loads((finite_directory / "gas.json").read_text())
        finite_response = json.loads(
            (finite_directory / "qm-response.json").read_text()
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
        with np.load(cpks_npz_path, allow_pickle=False) as state:
            cpks_mep = np.asarray(
                state["induced_surface_mep_hartree_per_e_per_source_e"],
                dtype=np.float64,
            )
            cpks_dipole = np.asarray(
                state["induced_dipole_e_bohr_per_source_e"],
                dtype=np.float64,
            )
            cpks_curvature = np.asarray(
                state["energy_curvature_hartree_per_e2"],
                dtype=np.float64,
            )
        if tuple(steps.tolist()) != (3.0e-4, 1.0e-3):
            raise RuntimeError("Finite-field steps differ from Gate-B.")
        if cpks_mep.shape != finite_mep.shape[1:] or cpks_dipole.shape != (4, 3):
            raise RuntimeError(f"Static CPKS observable shape changed for {stratum}.")

        finite_curvature = _finite_curvatures(
            finite_gas,
            finite_response,
            steps,
        )
        small_index, large_index = 0, 1
        mep_relative = _symmetric_relative(cpks_mep, finite_mep[large_index])
        dipole_relative = _symmetric_relative(
            cpks_dipole,
            finite_dipole[large_index],
        )
        mep_mode_relative = _mode_relative_maximum(
            cpks_mep,
            finite_mep[large_index],
        )
        dipole_mode_relative = _mode_relative_maximum(
            cpks_dipole,
            finite_dipole[large_index],
        )
        mep_two_step = _symmetric_relative(
            finite_mep[small_index],
            finite_mep[large_index],
        )
        dipole_two_step = _symmetric_relative(
            finite_dipole[small_index],
            finite_dipole[large_index],
        )
        curvature_absolute = float(
            np.max(np.abs(cpks_curvature - finite_curvature[large_index]))
        )
        curvature_two_step_absolute = float(
            np.max(
                np.abs(
                    finite_curvature[small_index] - finite_curvature[large_index]
                )
            )
        )
        scf_noise_bound = (
            2.0 * float(gates_contract["scf_energy_tolerance_hartree"])
            / float(steps[large_index] ** 2)
        )
        curvature_absolute_budget = max(
            curvature_two_step_absolute,
            scf_noise_bound,
        )

        response = cpks["response"]
        maximum_residual_frobenius = max(
            float(value)
            for value in response["final_residual_relative_frobenius"]
        )
        maximum_residual_infinity = max(
            float(value)
            for value in response["final_residual_relative_infinity"]
        )
        maximum_energy_identity_error = max(
            abs(float(value))
            for value in response["energy_identity_error_hartree_per_e2"]
        )
        maximum_electron_number_error = max(
            abs(float(value))
            for value in response[
                "electron_number_derivative_e_per_source_e"
            ]
        )
        maximum_passivity_eigenvalue = max(
            float(value)
            for value in response[
                "susceptibility_symmetric_eigenvalues_hartree_per_e2"
            ]
        )
        record_gates = {
            "mep_global_relative": (
                mep_relative
                <= float(gates_contract["mep_symmetric_relative_maximum"])
            ),
            "mep_mode_relative": (
                mep_mode_relative
                <= float(gates_contract["mep_symmetric_relative_maximum"])
            ),
            "mep_within_two_step": mep_relative <= mep_two_step,
            "dipole_global_relative": (
                dipole_relative
                <= float(gates_contract["dipole_symmetric_relative_maximum"])
            ),
            "dipole_mode_relative": (
                dipole_mode_relative
                <= float(gates_contract["dipole_symmetric_relative_maximum"])
            ),
            "dipole_within_two_step": dipole_relative <= dipole_two_step,
            "curvature_absolute": curvature_absolute <= curvature_absolute_budget,
            "cpks_residual_frobenius": (
                maximum_residual_frobenius
                <= float(
                    gates_contract[
                        "direct_cpks_residual_relative_frobenius_maximum"
                    ]
                )
            ),
            "cpks_residual_infinity": (
                maximum_residual_infinity
                <= float(
                    gates_contract[
                        "direct_cpks_residual_relative_infinity_maximum"
                    ]
                )
            ),
            "checkpoint_canonical_fock": (
                abs(
                    float(
                        cpks["checkpoint_consistency"][
                            "canonical_fock_residual_relative_frobenius"
                        ]
                    )
                )
                <= float(
                    gates_contract[
                        "checkpoint_canonical_fock_residual_relative_maximum"
                    ]
                )
            ),
            "checkpoint_rebuilt_energy": (
                abs(
                    float(
                        cpks["checkpoint_consistency"][
                            "rebuilt_energy_error_hartree"
                        ]
                    )
                )
                <= float(
                    gates_contract[
                        "checkpoint_rebuilt_energy_abs_hartree_maximum"
                    ]
                )
            ),
            "energy_identity": (
                maximum_energy_identity_error
                <= float(
                    gates_contract[
                        "energy_identity_abs_hartree_per_e2_maximum"
                    ]
                )
            ),
            "electron_number": (
                maximum_electron_number_error
                <= float(
                    gates_contract[
                        "electron_number_derivative_abs_maximum"
                    ]
                )
            ),
            "reciprocity": (
                float(
                    response["susceptibility_reciprocity_relative_frobenius"]
                )
                <= float(
                    gates_contract["reciprocity_relative_frobenius_maximum"]
                )
            ),
            "passivity": (
                maximum_passivity_eigenvalue
                <= float(
                    gates_contract[
                        "passivity_maximum_eigenvalue_hartree_per_e2"
                    ]
                )
            ),
        }
        record_results.append(
            {
                "stratum": stratum,
                "status": "pass" if all(record_gates.values()) else "fail",
                "gates": record_gates,
                "metrics": {
                    "mep_symmetric_relative": mep_relative,
                    "mep_mode_symmetric_relative_maximum": mep_mode_relative,
                    "mep_finite_two_step_symmetric_relative": mep_two_step,
                    "dipole_symmetric_relative": dipole_relative,
                    "dipole_mode_symmetric_relative_maximum": (
                        dipole_mode_relative
                    ),
                    "dipole_finite_two_step_symmetric_relative": dipole_two_step,
                    "curvature_absolute_error_hartree_per_e2": (
                        curvature_absolute
                    ),
                    "curvature_absolute_budget_hartree_per_e2": (
                        curvature_absolute_budget
                    ),
                    "curvature_finite_two_step_absolute_hartree_per_e2": (
                        curvature_two_step_absolute
                    ),
                    "maximum_cpks_residual_relative_frobenius": (
                        maximum_residual_frobenius
                    ),
                    "maximum_cpks_residual_relative_infinity": (
                        maximum_residual_infinity
                    ),
                    "maximum_energy_identity_error_hartree_per_e2": (
                        maximum_energy_identity_error
                    ),
                    "maximum_electron_number_derivative_abs": (
                        maximum_electron_number_error
                    ),
                    "reciprocity_relative_frobenius": float(
                        response[
                            "susceptibility_reciprocity_relative_frobenius"
                        ]
                    ),
                    "maximum_passivity_eigenvalue_hartree_per_e2": (
                        maximum_passivity_eigenvalue
                    ),
                    "cpks_total_elapsed_seconds": float(
                        cpks["runtime"]["total_elapsed_seconds"]
                    ),
                    "finite_field_total_elapsed_seconds": float(
                        finite_response["runtime"]["total_elapsed_seconds"]
                    ),
                },
                "source": {
                    "cpks_json_sha256": _sha256(cpks_json_path),
                    "cpks_npz_sha256": _sha256(cpks_npz_path),
                    "finite_result_path": str(
                        finite_result_path.relative_to(SOURCE_ROOT)
                    ),
                    "finite_result_sha256": _sha256(finite_result_path),
                },
            }
        )
        destination = output_directory / stratum
        copied_sources.extend(
            [
                (cpks_json_path, destination / "cpks.json"),
                (cpks_npz_path, destination / "cpks.npz"),
            ]
        )

    all_records_pass = all(record["status"] == "pass" for record in record_results)
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "pass-cpks-batch-backend" if all_records_pass else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sealer_path": SELF_REPO_PATH,
            "sealer_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "cpks_runner_sha256": preregistration["source"]["runner_sha256"],
        },
        "records": sorted(record_results, key=lambda record: record["stratum"]),
        "aggregate": {
            "record_count": len(record_results),
            "all_records_pass": all_records_pass,
            "maximum_mep_symmetric_relative": max(
                record["metrics"]["mep_symmetric_relative"]
                for record in record_results
            ),
            "maximum_dipole_symmetric_relative": max(
                record["metrics"]["dipole_symmetric_relative"]
                for record in record_results
            ),
            "maximum_cpks_residual_relative_frobenius": max(
                record["metrics"]["maximum_cpks_residual_relative_frobenius"]
                for record in record_results
            ),
            "median_wall_time_ratio_cpks_over_finite": float(
                np.median(
                    [
                        record["metrics"]["cpks_total_elapsed_seconds"]
                        / record["metrics"]["finite_field_total_elapsed_seconds"]
                        for record in record_results
                    ]
                )
            ),
        },
        "claim_boundary": {
            "cpks_batch_backend_admitted_for_same_q_to_zero_qm_target": (
                all_records_pass
            ),
            "independent_qm_training_data_generated": True,
            "model_fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "vqm24_energy_target_used": False,
            "pcm_or_cavity_used": False,
            "maple_capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preregistration_path, output_directory / "preregistration.json")
    for source, destination in copied_sources:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result_path = output_directory / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result_path.chmod(0o444)
    (output_directory / "README.md").write_text(
        "# VQM24 static CPKS Gate-B\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "This gate compares unopened F/P/S/Br static CPKS observable responses "
        "with the already frozen two-step finite-field oracle. Passing admits "
        "only a faster generator for the same q-to-zero independent QM target; "
        "it does not train a model, measure model accuracy, read solvation or "
        "VQM24 energy targets, or admit a MAPLE capability.\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
