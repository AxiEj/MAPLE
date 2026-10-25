"""Label-free numerical and timing harness for AnySolv/ConSolv GPU evidence.

The harness owns neither model loading nor admission.  Callers provide one
explicit execution-mode evaluator, which keeps runtime policy out of this
measurement module and makes the frozen protocol independently testable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

Evaluate = Callable[[np.ndarray, np.ndarray, str, bool], Mapping[str, Any]]
Synchronize = Callable[[], None]
FROZEN_PANEL_SHA256 = "fbfeb576eb37c7444190c497a9b0c459387d1a7ffd7395424779ea854713ec70"


def canonical_sha256(value: Any) -> str:
    """Hash a finite JSON value using the frozen-panel serialization."""

    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def load_frozen_panel(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Load a panel only when its exact bytes and label-free flags match."""

    panel_path = Path(path)
    payload = panel_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != str(expected_sha256).strip().lower():
        raise ValueError(
            f"Frozen solvation GPU panel SHA256 mismatch: expected {expected_sha256}, got {actual}."
        )
    panel = json.loads(payload)
    if panel.get("experimental_labels_opened") is not False:
        raise ValueError("Numerical GPU panel must not open experimental labels.")
    if panel.get("confirmation_partition_opened") is not False:
        raise ValueError(
            "Numerical GPU panel must not open the confirmation partition."
        )
    if canonical_sha256(panel) != actual:
        raise ValueError("Frozen panel is not in canonical serialization form.")
    return panel


def _geometry(case: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    numbers = np.asarray(case["numbers"], dtype=np.int64)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    if numbers.ndim != 1 or positions.shape != (len(numbers), 3):
        raise ValueError(f"Malformed frozen geometry for {case.get('case_id')!r}.")
    identity = {
        "numbers": numbers.tolist(),
        "positions_angstrom": positions.tolist(),
        "charge": case["charge"],
        "multiplicity": case["multiplicity"],
    }
    if canonical_sha256(identity) != case["geometry_sha256"]:
        raise ValueError(f"Frozen geometry SHA256 mismatch for {case['case_id']!r}.")
    return numbers, positions


def _normalized_result(raw: Mapping[str, Any], *, need_forces: bool) -> dict[str, Any]:
    energy = float(raw["energy_ev"])
    if not math.isfinite(energy):
        raise ValueError("Evaluator energy_ev must be finite.")
    normalized: dict[str, Any] = {"energy_ev": energy}
    if need_forces:
        forces = np.asarray(raw["forces_ev_per_angstrom"], dtype=np.float64)
        if (
            forces.ndim != 2
            or forces.shape[1:] != (3,)
            or not np.isfinite(forces).all()
        ):
            raise ValueError("Evaluator forces must be a finite N x 3 array.")
        normalized["forces_ev_per_angstrom"] = forces.tolist()
    normalized["output_sha256"] = canonical_sha256(normalized)
    return normalized


def _evaluate_record(
    evaluate: Evaluate,
    case_id: str,
    numbers: np.ndarray,
    positions: np.ndarray,
    solvent: str,
    need_forces: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record: dict[str, Any] = {"case_id": case_id, "status": "failed"}
    try:
        normalized = _normalized_result(
            evaluate(numbers.copy(), positions.copy(), solvent, need_forces),
            need_forces=need_forces,
        )
        if (
            need_forces
            and np.asarray(normalized["forces_ev_per_angstrom"]).shape
            != positions.shape
        ):
            raise ValueError(
                "Evaluator force shape does not match the frozen geometry."
            )
        record.update(status="success", **normalized)
        return record, normalized
    except Exception as exc:  # failure-complete scientific ledger
        record.update(error_type=type(exc).__name__, error=str(exc))
        return record, None


def _timing_summary(seconds: list[float]) -> dict[str, Any]:
    values = np.asarray(seconds, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Timing samples must be a non-empty finite sequence.")
    return {
        "count": int(values.size),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "samples_seconds": values.tolist(),
    }


def _collect_timing(
    evaluate: Evaluate,
    case: Mapping[str, Any],
    *,
    need_forces: bool,
    synchronize: Synchronize,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    numbers, positions = _geometry(case)
    warmup = int(timing["warmup"])
    repeats = int(timing["repetitions"])
    try:
        for index in range(warmup):
            delta = (
                0.001
                * (index + 1)
                * np.sin(np.arange(positions.size)).reshape(positions.shape)
            )
            evaluate(numbers.copy(), positions + delta, case["solvent"], need_forces)
            synchronize()

        fresh_seconds: list[float] = []
        fresh_geometry_sha256s: list[str] = []
        for index in range(repeats):
            changed = positions + 0.001 * (index + 1) * np.sin(
                np.arange(positions.size)
            ).reshape(positions.shape)
            fresh_geometry_sha256s.append(
                canonical_sha256(
                    {
                        "numbers": numbers.tolist(),
                        "positions_angstrom": changed.tolist(),
                    }
                )
            )
            synchronize()
            started = time.perf_counter()
            _normalized_result(
                evaluate(numbers.copy(), changed, case["solvent"], need_forces),
                need_forces=need_forces,
            )
            synchronize()
            fresh_seconds.append(time.perf_counter() - started)

        exact_input_seconds: list[float] = []
        for _ in range(repeats):
            synchronize()
            started = time.perf_counter()
            _normalized_result(
                evaluate(
                    numbers.copy(), positions.copy(), case["solvent"], need_forces
                ),
                need_forces=need_forces,
            )
            synchronize()
            exact_input_seconds.append(time.perf_counter() - started)
        return {
            "status": "success",
            "case_id": case["case_id"],
            "warmup_count": warmup,
            "fresh_inference": {
                **_timing_summary(fresh_seconds),
                "geometry_sha256s": fresh_geometry_sha256s,
            },
            "exact_input_repetition": {
                **_timing_summary(exact_input_seconds),
                "classification": (
                    "identical-input evaluation; cache hit not assumed; "
                    "not inference acceleration"
                ),
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "case_id": case["case_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def collect_numerical_panel(
    evaluate: Evaluate,
    panel: Mapping[str, Any],
    route: str,
    *,
    inference_mode: str = "reference",
    synchronize: Synchronize = lambda: None,
    include_timing: bool = False,
) -> dict[str, Any]:
    """Evaluate one route/mode and return a failure-complete JSON ledger."""

    route = str(route).strip().lower()
    if route not in {"anysolv", "consolv"}:
        raise ValueError("route must be anysolv or consolv.")
    actual_panel_sha256 = canonical_sha256(panel)
    if actual_panel_sha256 != FROZEN_PANEL_SHA256:
        raise ValueError(
            "Solvation GPU collection requires the exact preregistered frozen panel."
        )
    inference_mode = str(inference_mode).strip().lower()
    if inference_mode not in panel["modes"]:
        raise ValueError(
            f"inference_mode must be one of the frozen modes {panel['modes']!r}."
        )
    if (
        panel.get("experimental_labels_opened") is not False
        or panel.get("confirmation_partition_opened") is not False
    ):
        raise ValueError("Solvation GPU collection accepts label-free panels only.")

    scalar_cases = [case for case in panel["scalar_cases"] if case["route"] == route]
    vacuum_cases = [case for case in panel["vacuum_cases"] if case["route"] == route]
    expected = panel["expected_counts_per_gpu_mode"]
    expected_scalar = (
        expected[f"{route}_scalar"]
        if route == "anysolv"
        else expected["consolv_solution"]
    )
    expected_vacuum = expected[f"{route}_vacuum"]
    if len(scalar_cases) != expected_scalar or len(vacuum_cases) != expected_vacuum:
        raise ValueError(
            "Frozen scalar/vacuum case counts do not match their declared denominators."
        )
    scalar_records: list[dict[str, Any]] = []
    vacuum_records: list[dict[str, Any]] = []
    successful: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    need_forces = route == "consolv"

    for case in scalar_cases:
        numbers, positions = _geometry(case)
        record, result = _evaluate_record(
            evaluate, case["case_id"], numbers, positions, case["solvent"], need_forces
        )
        scalar_records.append(record)
        if result is not None:
            successful[case["case_id"]] = (result, numbers, positions)
    for case in vacuum_cases:
        numbers, positions = _geometry(case)
        record, _ = _evaluate_record(
            evaluate, case["case_id"], numbers, positions, "vacuum", need_forces
        )
        vacuum_records.append(record)

    fd_records: list[dict[str, Any]] = []
    motion_records: list[dict[str, Any]] = []
    sampled_records: list[dict[str, Any]] = []
    if route == "consolv":
        if len(panel["finite_difference_cases"]) != expected["consolv_fd_components"]:
            raise ValueError("Frozen FD cases do not match their declared denominator.")
        if len(panel["motion_cases"]) != expected["consolv_motion"]:
            raise ValueError(
                "Frozen motion cases do not match their declared denominator."
            )
        scalar_by_id = {case["case_id"]: case for case in scalar_cases}
        for fd in panel["finite_difference_cases"]:
            base = scalar_by_id[fd["base_case_id"]]
            numbers, positions = _geometry(base)
            plus, minus = positions.copy(), positions.copy()
            index = (fd["atom_index"], fd["component_index"])
            plus[index] += fd["displacement_angstrom"]
            minus[index] -= fd["displacement_angstrom"]
            plus_record, plus_result = _evaluate_record(
                evaluate, fd["case_id"] + ":plus", numbers, plus, base["solvent"], False
            )
            minus_record, minus_result = _evaluate_record(
                evaluate,
                fd["case_id"] + ":minus",
                numbers,
                minus,
                base["solvent"],
                False,
            )
            record = {
                "case_id": fd["case_id"],
                "plus": plus_record,
                "minus": minus_record,
                "status": "failed",
            }
            base_result = successful.get(fd["base_case_id"])
            if (
                plus_result is not None
                and minus_result is not None
                and base_result is not None
            ):
                finite_difference = -(
                    plus_result["energy_ev"] - minus_result["energy_ev"]
                ) / (2.0 * fd["displacement_angstrom"])
                analytic = base_result[0]["forces_ev_per_angstrom"][fd["atom_index"]][
                    fd["component_index"]
                ]
                record.update(
                    status="success",
                    finite_difference_force_ev_per_angstrom=finite_difference,
                    analytic_force_ev_per_angstrom=analytic,
                    absolute_error_ev_per_angstrom=abs(finite_difference - analytic),
                )
            else:
                record.update(
                    error_type="ReferenceEvaluationError",
                    error="FD comparison needs successful base, plus, and minus evaluations.",
                )
            fd_records.append(record)

        rotation = np.asarray(panel["rotation_matrix"], dtype=np.float64)
        for motion in panel["motion_cases"]:
            numbers, positions = _geometry(motion)
            record, transformed = _evaluate_record(
                evaluate, motion["case_id"], numbers, positions, motion["solvent"], True
            )
            output = {
                "case_id": motion["case_id"],
                "evaluation": record,
                "status": "failed",
            }
            base_id = f"consolv:{motion['molecule']}:{motion['solvent']}"
            base_result = successful.get(base_id)
            if transformed is not None and base_result is not None:
                expected_force = np.asarray(base_result[0]["forces_ev_per_angstrom"])
                if motion["transform"] == "rotation":
                    expected_force = expected_force @ rotation.T
                elif motion["transform"] == "permutation":
                    expected_force = expected_force[::-1]
                observed = np.asarray(transformed["forces_ev_per_angstrom"])
                output.update(
                    status="success",
                    absolute_energy_error_ev=abs(
                        transformed["energy_ev"] - base_result[0]["energy_ev"]
                    ),
                    maximum_force_covariance_error_ev_per_angstrom=float(
                        np.max(np.abs(observed - expected_force))
                    ),
                )
            else:
                output.update(
                    error_type="ReferenceEvaluationError",
                    error="Motion comparison needs successful base and transformed evaluations.",
                )
            motion_records.append(output)

        for replay_id, initial_id in zip(
            panel["sampled_force_probe"]["case_ids"],
            panel["sampled_force_probe"]["initial_case_ids"],
        ):
            base = scalar_by_id[initial_id]
            numbers, positions = _geometry(base)
            steps: list[dict[str, Any]] = []
            replay = {"case_id": replay_id, "status": "success", "steps": steps}
            for step in range(panel["sampled_force_probe"]["steps"]):
                evaluated, result = _evaluate_record(
                    evaluate,
                    f"{replay_id}:step:{step}",
                    numbers,
                    positions,
                    base["solvent"],
                    True,
                )
                if result is None:
                    replay.update(
                        status="failed",
                        error_type="ForceReplayEvaluationError",
                        error=f"Force replay failed at step {step}.",
                    )
                    steps.append(evaluated)
                    break
                positions = positions + panel["sampled_force_probe"][
                    "step_size"
                ] * np.asarray(result["forces_ev_per_angstrom"])
                steps.append(
                    {**evaluated, "next_positions_angstrom": positions.tolist()}
                )
            sampled_records.append(replay)

    timing_record = None
    if include_timing:
        timing_case = next(
            case
            for case in scalar_cases
            if case["molecule"] == "C2H5OH" and case["solvent"] == "ethanol"
        )
        timing_record = _collect_timing(
            evaluate,
            timing_case,
            need_forces=need_forces,
            synchronize=synchronize,
            timing=panel["timing"],
        )

    ledgers = (
        scalar_records + vacuum_records + fd_records + motion_records + sampled_records
    )
    failures = sum(record.get("status") != "success" for record in ledgers)
    internal_context_coverage: list[dict[str, Any]] = []
    if route == "consolv":
        failed_scalar_ids = [
            record["case_id"]
            for record in scalar_records
            if record["status"] != "success"
        ]
        context_rows = [
            row
            for row in panel["context_matrix"]
            if row["inference_mode"] == inference_mode
        ]
        expected_contexts = panel["internal_mechanical_contexts"]["consolv"]
        if [row["task"] for row in context_rows] != expected_contexts:
            raise ValueError(
                "Frozen ConSolv context matrix does not cover the declared contexts."
            )
        for row in context_rows:
            internal_context_coverage.append(
                {
                    "task": row["task"],
                    "inference_mode": inference_mode,
                    "status": "failed" if failed_scalar_ids else "success",
                    "reused_solution_case_ids": list(row["solution_case_ids"]),
                    "expected_energy_pairs": row["expected_energy_pairs"],
                    "expected_force_components": row["expected_force_components"],
                    "failed_case_ids": failed_scalar_ids,
                    "scope": panel["context_scope"],
                }
            )
    return {
        "schema_version": 1,
        "kind": "solvation_gpu_label_free_numerical_measurement",
        "route": route,
        "inference_mode": inference_mode,
        "panel_sha256": actual_panel_sha256,
        "experimental_labels_opened": False,
        "confirmation_partition_opened": False,
        "counts": {
            "scalar": len(scalar_records),
            "vacuum": len(vacuum_records),
            "finite_difference_components": len(fd_records),
            "motion_comparisons": len(motion_records),
            "sampled_force_replays": len(sampled_records),
            "failed_records": failures,
        },
        "expected_counts": {
            "scalar": expected_scalar,
            "vacuum": expected_vacuum,
            "finite_difference_components": (
                expected["consolv_fd_components"] if route == "consolv" else 0
            ),
            "motion_comparisons": (
                expected["consolv_motion"] if route == "consolv" else 0
            ),
            "sampled_force_replays": (
                panel["sampled_force_probe"]["expected_replay_records"]
                if route == "consolv"
                else 0
            ),
        },
        "scalar_records": scalar_records,
        "vacuum_records": vacuum_records,
        "finite_difference_records": fd_records,
        "motion_records": motion_records,
        "sampled_force_records": sampled_records,
        "internal_context_coverage": internal_context_coverage,
        "timing": timing_record,
    }
