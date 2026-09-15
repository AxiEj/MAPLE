"""Machine-readable execution status for pure MACE-POLAR non-MD v2 jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_STATUS_FIELDS = (
    "converged",
    "termination_reason",
    "iterations",
    "final_metrics",
)


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

    missing = [field for field in REQUIRED_STATUS_FIELDS if field not in status]
    if missing:
        raise ValueError(f"status is missing required fields: {', '.join(missing)}")
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
    "calculator_provenance",
    "geometries_record",
    "geometry_record",
    "is_pure_nonmd_v2",
    "make_status",
    "optimization_converged",
    "optimization_metrics",
    "status_path",
    "write_status",
]
