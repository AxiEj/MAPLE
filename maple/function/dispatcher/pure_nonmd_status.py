"""Machine-readable execution status for pure MACE-POLAR non-MD v2 jobs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_STATUS_FIELDS = (
    "converged",
    "termination_reason",
    "iterations",
    "final_metrics",
)
REQUIRED_V4_OUTCOME_FIELDS = (
    "completed", "task_success", "termination_reason", "iterations",
    "final_metrics", "geometry",
)
REQUIRED_V4_OUTCOME_V2_FIELDS = REQUIRED_V4_OUTCOME_FIELDS + (
    "numerical_stationary_point_validated",
    "certificate_status",
    "scientific_certified",
)
_V4_OUTCOME_RESERVED_FIELDS = frozenset({
    "schema", "workflow", "method", "completed", "task_success",
    "termination_class", "termination_reason", "iterations",
    "optimizer_converged", "path_converged", "scientific_certified",
    "numerical_stationary_point_validated", "certificate_status",
    "candidate_only", "rejected", "final_metrics", "geometry", "trace",
})


def _validate_outcome_state(
    *, completed, task_success, candidate_only, rejected,
    numerical_stationary_point_validated, certificate_status,
    scientific_certified,
) -> None:
    for name, value in (
        ("completed", completed), ("task_success", task_success),
        ("candidate_only", candidate_only), ("rejected", rejected),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool.")
    for name, value in (
        ("numerical_stationary_point_validated", numerical_stationary_point_validated),
        ("scientific_certified", scientific_certified),
    ):
        _optional_strict_bool(value, name=name)
    if certificate_status not in {
        "unavailable", "passed", "failed", "not_applicable"
    }:
        raise ValueError("Invalid certificate_status.")
    if task_success and (not completed or candidate_only or rejected):
        raise ValueError(
            "Task success requires completion and forbids candidate/rejected state."
        )
    if rejected and completed:
        raise ValueError("A rejected workflow cannot be completed.")
    if certificate_status == "unavailable" and scientific_certified is not None:
        raise ValueError("Unavailable certification requires scientific_certified=None.")
    if certificate_status == "failed" and task_success:
        raise ValueError("A failed certificate cannot accompany task success.")
    if scientific_certified is False and task_success:
        raise ValueError("An uncertified scientific candidate cannot be successful.")


@dataclass(frozen=True, slots=True)
class WorkflowGeometry:
    """Immutable molecular geometry captured when an algorithm returns."""

    symbols: tuple[str, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.symbols or len(self.positions_angstrom) != len(self.symbols):
            raise ValueError("Workflow geometry symbols and positions must align.")
        if any(
            len(row) != 3 or not all(np.isfinite(value) for value in row)
            for row in self.positions_angstrom
        ):
            raise ValueError("Workflow geometry must contain finite 3D positions.")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "WorkflowGeometry":
        try:
            symbols = tuple(str(symbol) for symbol in record["symbols"])
            positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Workflow geometry record is invalid.") from exc
        if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
            raise ValueError("Workflow geometry must contain finite N-by-3 positions.")
        return cls(
            symbols=symbols,
            positions_angstrom=tuple(
                tuple(float(value) for value in row) for row in positions
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "positions_angstrom": [list(row) for row in self.positions_angstrom],
        }


def workflow_geometry_snapshot(
    records: Iterable[dict[str, Any]],
) -> tuple[WorkflowGeometry, ...]:
    """Copy external geometry records into immutable outcome-owned values."""

    return tuple(WorkflowGeometry.from_record(record) for record in records)


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    """Version-neutral task result before profile-specific serialization."""

    workflow: str
    method: str
    completed: bool
    task_success: bool
    termination_reason: str
    termination_class: str
    iterations: int
    final_metrics: dict[str, Any]
    geometry: tuple[WorkflowGeometry, ...]
    trace: tuple[dict[str, Any], ...] = ()
    optimizer_converged: bool | None = None
    path_converged: bool | None = None
    scientific_certified: bool | None = None
    numerical_stationary_point_validated: bool | None = None
    certificate_status: str = "not_applicable"
    candidate_only: bool = False
    rejected: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not self.termination_reason:
            raise ValueError("termination_reason must be non-empty")
        if not self.geometry or any(
            not isinstance(item, WorkflowGeometry) for item in self.geometry
        ):
            raise ValueError("Workflow outcome requires an immutable geometry snapshot.")
        _validate_outcome_state(
            completed=self.completed,
            task_success=self.task_success,
            candidate_only=self.candidate_only,
            rejected=self.rejected,
            numerical_stationary_point_validated=(
                self.numerical_stationary_point_validated
            ),
            certificate_status=self.certificate_status,
            scientific_certified=self.scientific_certified,
        )
        reserved = sorted(_V4_OUTCOME_RESERVED_FIELDS.intersection(self.extra))
        if reserved:
            raise ValueError(
                "Workflow outcome extra fields cannot override reserved keys: "
                + ", ".join(reserved)
            )

    def to_v4_record(self) -> dict[str, Any]:
        record = dict(self.extra)
        record.update({
            "schema": "maple-route2-workflow-outcome-v1",
            "workflow": self.workflow,
            "method": self.method,
            "completed": self.completed,
            "task_success": self.task_success,
            "termination_class": self.termination_class,
            "termination_reason": self.termination_reason,
            "iterations": self.iterations,
            "optimizer_converged": self.optimizer_converged,
            "path_converged": self.path_converged,
            "scientific_certified": self.scientific_certified,
            "candidate_only": self.candidate_only,
            "rejected": self.rejected,
            "final_metrics": dict(self.final_metrics),
            "geometry": [item.to_record() for item in self.geometry],
            "trace": list(self.trace),
        })
        return record

    def to_route2_record(self, outcome_policy: str) -> dict[str, Any]:
        if outcome_policy not in {
            "numerical_stationary", "numerical_irc_auto", "exploratory_irc",
            "single_point_frames", "frequency_quantities",
        }:
            return self.to_v4_record()
        record = self.to_v4_record()
        record.update(
            schema="maple-route2-workflow-outcome-v2",
            numerical_stationary_point_validated=(
                self.numerical_stationary_point_validated
            ),
            certificate_status=self.certificate_status,
        )
        return record

    @classmethod
    def from_internal_status(
        cls, status: dict[str, Any], *, capability: Any
    ) -> "WorkflowOutcome":
        converged = bool(status["converged"])
        workflow = str(status["workflow"])
        metrics = dict(status["final_metrics"])
        candidate_only = capability.execution_class == "candidate_only"
        numerical_policy = capability.outcome_policy in {
            "numerical_stationary", "numerical_irc_auto"
        }
        exploratory_policy = capability.outcome_policy == "exploratory_irc"
        path_policy = capability.outcome_policy == "path_success"
        if capability.outcome_policy in {"candidate_validation", "numerical_stationary"}:
            path_policy = capability.image_policy == "multi"
        optimizer_converged = metrics.get("optimizer_converged")
        path_converged = metrics.get("path_converged")
        if numerical_policy or exploratory_policy:
            optimizer_converged = _optional_strict_bool(
                optimizer_converged, name="optimizer_converged"
            )
            path_converged = _optional_strict_bool(
                path_converged, name="path_converged"
            )
        if optimizer_converged is None and capability.outcome_policy in {
            "candidate_validation", "numerical_stationary"
        }:
            optimizer_converged = _nested_optimizer_converged(status.get("refinement"))
        if optimizer_converged is None and not path_policy and workflow in {"opt", "ts"}:
            optimizer_converged = converged
        if path_converged is None and path_policy:
            path_converged = converged
        extra = {
            key: value for key, value in status.items()
            if key not in {
                "schema", "workflow", "method", "executed", "termination_class",
                "converged", "termination_reason", "iterations", "final_metrics",
                "geometry", "trace",
            }
        }
        termination_class = str(status["termination_class"])
        if capability.outcome_policy == "fixed_steps" and converged:
            termination_class = "completed"
        try:
            geometry = workflow_geometry_snapshot(status["geometry"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Internal workflow status requires authoritative geometry."
            ) from exc
        numerical_validated = _optional_strict_bool(
            metrics.get("numerical_stationary_point_validated"),
            name="numerical_stationary_point_validated",
        )
        if numerical_validated is None and capability.outcome_policy == "numerical_stationary":
            numerical_validated = _nested_numerical_stationary_validated(
                status.get("refinement")
            )
        if numerical_policy and converged and numerical_validated is not True:
            raise ValueError(
                "Successful numerical workflow requires numerical stationary validation."
            )
        task_success = converged and not candidate_only
        if numerical_policy:
            task_success = bool(converged and numerical_validated is True)
        if exploratory_policy:
            task_success = False
        return cls(
            workflow=workflow,
            method=str(status["method"]),
            completed=(
                bool(status.get("executed", True))
                and status.get("termination_class") != "execution_failure"
            ),
            task_success=task_success,
            optimizer_converged=(
                None if optimizer_converged is None else bool(optimizer_converged)
            ),
            path_converged=(
                None if path_converged is None else bool(path_converged)
            ),
            scientific_certified=(
                None if numerical_policy or exploratory_policy
                else False if candidate_only else None
            ),
            numerical_stationary_point_validated=numerical_validated,
            certificate_status=(
                "unavailable" if numerical_policy or exploratory_policy
                else "not_applicable"
            ),
            candidate_only=candidate_only,
            rejected=not bool(status.get("executed", True)),
            termination_reason=(
                "stationary_point_certificate_unavailable"
                if candidate_only and converged
                else str(status["termination_reason"])
            ),
            termination_class=(
                "validation_failure" if candidate_only and converged
                else termination_class
            ),
            iterations=int(status["iterations"]),
            final_metrics=metrics,
            geometry=geometry,
            trace=tuple(status.get("trace", ())),
            extra=extra,
        )


def _nested_optimizer_converged(value: Any) -> bool | None:
    """Read an explicitly named optimizer result from refinement records."""

    if not isinstance(value, dict):
        return None
    metrics = value.get("final_metrics")
    if isinstance(metrics, dict) and "optimizer_converged" in metrics:
        return _optional_strict_bool(
            metrics["optimizer_converged"], name="nested optimizer_converged"
        )
    nested = value.get("refinement")
    explicit = _nested_optimizer_converged(nested)
    if explicit is not None:
        return explicit
    if value.get("workflow") == "ts" and "converged" in value:
        return _optional_strict_bool(
            value["converged"], name="nested refinement converged"
        )
    return None


def _nested_numerical_stationary_validated(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    metrics = value.get("final_metrics")
    if isinstance(metrics, dict) and "numerical_stationary_point_validated" in metrics:
        return _optional_strict_bool(
            metrics["numerical_stationary_point_validated"],
            name="nested numerical_stationary_point_validated",
        )
    return _nested_numerical_stationary_validated(value.get("refinement"))


def _optional_strict_bool(value: Any, *, name: str) -> bool | None:
    if value is None or type(value) is bool:
        return value
    raise TypeError(f"{name} must be bool or None.")


class NonMDWorkflowFailure(RuntimeError):
    """A failed task with a measured, typed outcome that must survive dispatch."""

    def __init__(self, status: dict[str, Any]):
        if status["converged"]:
            raise ValueError("A workflow failure cannot carry a converged status.")
        self.status = dict(status)
        super().__init__(str(status["termination_reason"]))


def is_pure_nonmd_v2(atoms: Any) -> bool:
    """Return whether all supplied images use the canonical pure non-MD v2 PES."""

    from maple.function.calculator.route2 import (
        is_pure_mace_polar_nonmd_calculator,
    )

    if hasattr(atoms, "multiatoms"):
        images = list(atoms.multiatoms)
    else:
        images = list(atoms) if isinstance(atoms, (list, tuple)) else [atoms]
    return bool(images) and all(
        is_pure_mace_polar_nonmd_calculator(getattr(image, "calc", None))
        for image in images
    )


def is_pure_torch_calculator(atoms: Any) -> bool:
    """Return whether all images use a canonical analytic Torch facade."""

    from maple.function.calculator.route2 import (
        is_pure_mace_polar_torch_calculator,
    )

    if hasattr(atoms, "multiatoms"):
        images = list(atoms.multiatoms)
    else:
        images = list(atoms) if isinstance(atoms, (list, tuple)) else [atoms]
    return bool(images) and all(
        is_pure_mace_polar_torch_calculator(getattr(image, "calc", None))
        for image in images
    )


def is_pure_torch_v4(atoms: Any) -> bool:
    from maple.function.route2_smd_profiles import (
        PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
    )

    if not is_pure_torch_calculator(atoms):
        return False
    if hasattr(atoms, "multiatoms"):
        images = list(atoms.multiatoms)
    else:
        images = list(atoms) if isinstance(atoms, (list, tuple)) else [atoms]
    return all(
        getattr(
            getattr(getattr(image, "calc", None), "raw_calculator", getattr(image, "calc", None)),
            "profile_spec", None,
        ).workflow_policy_id == PURE_TORCH_FP32_V4_WORKFLOW_POLICY
        for image in images
    )


def is_route2_workflow_status_enabled(atoms: Any) -> bool:
    return is_pure_nonmd_v2(atoms) or is_pure_torch_v4(atoms)


def add_v4_outcome_metrics(
    atoms: Any, metrics: dict[str, Any], **values: Any
) -> dict[str, Any]:
    """Add fields consumed only by the V4 outcome projection."""

    if is_pure_torch_v4(atoms):
        metrics.update(values)
    return metrics


def geometry_record(atoms: Any) -> dict[str, Any]:
    return {
        "symbols": list(atoms.get_chemical_symbols()),
        "positions_angstrom": np.asarray(
            atoms.get_positions(), dtype=np.float64
        ).tolist(),
    }


def geometries_record(atoms: Any) -> list[dict[str, Any]]:
    if hasattr(atoms, "multiatoms"):
        images = list(atoms.multiatoms)
    else:
        images = list(atoms) if isinstance(atoms, (list, tuple)) else [atoms]
    return [geometry_record(image) for image in images]


def make_status(
    *,
    workflow: str,
    method: str,
    converged: bool,
    termination_reason: str,
    iterations: int,
    final_metrics: dict[str, Any],
    atoms: Any,
    trace: Iterable[dict[str, Any]] = (),
    executed: bool = True,
    termination_class: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the canonical v2 status without converting a cap into success."""

    if int(iterations) < 0:
        raise ValueError("iterations must be non-negative")
    if not termination_reason:
        raise ValueError("termination_reason must be non-empty")
    if termination_class is None:
        termination_class = (
            "converged" if converged else "bounded_nonconvergence"
        )
    allowed_classes = {
        "converged", "bounded_nonconvergence", "validation_failure",
        "execution_failure",
    }
    if termination_class not in allowed_classes:
        raise ValueError(f"invalid termination_class: {termination_class}")
    record = {
        "schema": "maple-pure-nonmd-status-v2",
        "workflow": str(workflow),
        "method": str(method),
        "executed": bool(executed),
        "termination_class": termination_class,
        "converged": bool(converged),
        "termination_reason": str(termination_reason),
        "iterations": int(iterations),
        "final_metrics": dict(final_metrics),
        "geometry": geometries_record(atoms),
        "trace": list(trace),
    }
    record.update(extra)
    return record


def calculator_provenance(atoms: Any) -> dict[str, Any]:
    if hasattr(atoms, "multiatoms"):
        atoms = atoms.multiatoms[0]
    elif isinstance(atoms, (list, tuple)):
        atoms = atoms[0]
    calculator = getattr(atoms, "calc", None)
    calculator = getattr(calculator, "raw_calculator", calculator)
    spec = getattr(calculator, "profile_spec", None)
    return {
        "profile": getattr(spec, "name", None),
        "scalar_contract_id": getattr(spec, "scalar_contract_id", None),
        "actual_device": str(getattr(calculator, "device", "unknown")),
    }


def status_path(output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    return path.with_name(path.stem + "_status.json")


def _numpy_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return item
    raise TypeError(f"Unsupported status value type: {type(value).__name__}")


def write_status(output: str | os.PathLike[str], status: dict[str, Any]) -> Path:
    """Atomically persist one canonical status sidecar."""

    schema = status.get("schema")
    required = (
        REQUIRED_V4_OUTCOME_V2_FIELDS
        if schema == "maple-route2-workflow-outcome-v2"
        else REQUIRED_V4_OUTCOME_FIELDS
        if schema == "maple-route2-workflow-outcome-v1"
        else REQUIRED_STATUS_FIELDS
    )
    missing = [field for field in required if field not in status]
    if missing:
        raise ValueError(f"status is missing required fields: {', '.join(missing)}")
    if schema == "maple-route2-workflow-outcome-v2":
        for field_name in ("optimizer_converged", "path_converged"):
            _optional_strict_bool(status.get(field_name), name=field_name)
        _validate_outcome_state(
            completed=status["completed"],
            task_success=status["task_success"],
            candidate_only=status["candidate_only"],
            rejected=status["rejected"],
            numerical_stationary_point_validated=(
                status["numerical_stationary_point_validated"]
            ),
            certificate_status=status["certificate_status"],
            scientific_certified=status["scientific_certified"],
        )
    target = status_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True, allow_nan=False,
                   default=_numpy_json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def optimization_metrics(atoms: Any) -> dict[str, float | None]:
    names = ("max_f", "rms_f", "max_dp", "rms_dp")
    return {
        name: (float(getattr(atoms, name)) if hasattr(atoms, name) else None)
        for name in names
    }


def optimization_converged(atoms: Any) -> bool:
    thresholds = {
        "max_f": "f_max_th",
        "rms_f": "f_rms_th",
        "max_dp": "dp_max_th",
        "rms_dp": "dp_rms_th",
    }
    return all(
        hasattr(atoms, metric)
        and hasattr(atoms, threshold)
        and float(getattr(atoms, metric)) <= float(getattr(atoms, threshold))
        for metric, threshold in thresholds.items()
    )


__all__ = [
    "REQUIRED_STATUS_FIELDS",
    "REQUIRED_V4_OUTCOME_FIELDS",
    "WorkflowOutcome",
    "WorkflowGeometry",
    "add_v4_outcome_metrics",
    "calculator_provenance",
    "geometries_record",
    "geometry_record",
    "is_pure_nonmd_v2",
    "is_pure_torch_calculator",
    "is_pure_torch_v4",
    "is_route2_workflow_status_enabled",
    "make_status",
    "optimization_converged",
    "optimization_metrics",
    "status_path",
    "write_status",
    "workflow_geometry_snapshot",
]
