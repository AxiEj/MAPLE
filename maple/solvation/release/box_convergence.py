"""Preregistered fixed-box convergence summaries for Route-2 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

BOX_LENGTHS_A = (32, 40, 48, 56)
TAIL_TOTAL_ENERGY_TOLERANCE_EV = 1.0e-4
TAIL_CONTINUUM_ENERGY_TOLERANCE_EV = 1.0e-4
TAIL_FORCE_RMS_TOLERANCE_EV_PER_A = 1.0e-4
TAIL_FORCE_MAX_TOLERANCE_EV_PER_A = 5.0e-4
TAIL_SOURCE_RELATIVE_TOLERANCE = 1.0e-5
MAXIMUM_PRIMAL_RESIDUAL = 1.0e-12
MAXIMUM_ADJOINT_RESIDUAL = 1.0e-10


def _number(record: Mapping[str, object], name: str) -> float:
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _array(record: Mapping[str, object], name: str, *, components: int) -> np.ndarray:
    values = np.asarray(record[name], dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != components
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"{name} must be finite with shape (atom_count, {components})."
        )
    return values


def _identity(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty identity.")
    return value.strip()


def summarize_box_convergence(
    raw_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare the preregistered 32/40/48/56-A fixed-box family.

    The 56-A calculation is the finite reference and the 48->56 tail decides
    the convergence gate.  This is a numerical-operator diagnostic, not a
    chemical-accuracy or capability-admission decision.
    """

    records = tuple(raw_records)
    lengths = tuple(int(record.get("box_length_A", -1)) for record in records)
    if lengths != BOX_LENGTHS_A:
        raise ValueError(
            "box records must contain exactly the preregistered ordered lengths "
            f"{BOX_LENGTHS_A}."
        )
    profile_ids = tuple(_identity(record, "profile_id") for record in records)
    model_ids = tuple(_identity(record, "model_profile_id") for record in records)
    configurations = tuple(
        _identity(record, "model_configuration_sha256") for record in records
    )
    if len(set(profile_ids)) != len(records):
        raise ValueError("Each box must use a distinct profile identity.")
    if len(set(model_ids)) != len(records):
        raise ValueError("Each box must use a distinct model-profile identity.")
    if len(set(configurations)) != len(records):
        raise ValueError("Each box must use a distinct model configuration hash.")

    validated: list[dict[str, object]] = []
    force_shape: tuple[int, int] | None = None
    source_shape: tuple[int, int] | None = None
    for record in records:
        forces = _array(record, "forces_eV_per_A", components=3)
        source_values = np.asarray(record["source"], dtype=float)
        if (
            source_values.ndim != 2
            or source_values.shape[0] < 1
            or source_values.shape[1] < 1
            or not np.all(np.isfinite(source_values))
        ):
            raise ValueError("source must be a non-empty finite two-dimensional array.")
        if force_shape is None:
            force_shape, source_shape = forces.shape, source_values.shape
        if forces.shape != force_shape or source_values.shape != source_shape:
            raise ValueError("All box records must use identical force/source shapes.")
        primal = _number(record, "primal_residual")
        adjoint = _number(record, "adjoint_residual")
        if primal < 0.0 or primal > MAXIMUM_PRIMAL_RESIDUAL:
            raise ValueError("Each box primal residual must be within 1e-12.")
        if adjoint < 0.0 or adjoint > MAXIMUM_ADJOINT_RESIDUAL:
            raise ValueError("Each box adjoint residual must be within 1e-10.")
        validated.append(
            {
                **dict(record),
                "total_energy_eV": _number(record, "total_energy_eV"),
                "continuum_energy_eV": _number(record, "continuum_energy_eV"),
                "forces_eV_per_A": forces.tolist(),
                "source": source_values.tolist(),
            }
        )

    reference = validated[-1]
    reference_forces = np.asarray(reference["forces_eV_per_A"])
    reference_source = np.asarray(reference["source"])
    comparisons: list[dict[str, object]] = []
    for record in validated:
        force_error = np.asarray(record["forces_eV_per_A"]) - reference_forces
        source_error = np.asarray(record["source"]) - reference_source
        comparisons.append(
            {
                "box_length_A": record["box_length_A"],
                "total_energy_difference_to_56A_eV": (
                    record["total_energy_eV"] - reference["total_energy_eV"]
                ),
                "continuum_energy_difference_to_56A_eV": (
                    record["continuum_energy_eV"] - reference["continuum_energy_eV"]
                ),
                "force_rms_difference_to_56A_eV_per_A": float(
                    np.sqrt(np.mean(force_error**2))
                ),
                "force_max_difference_to_56A_eV_per_A": float(
                    np.max(np.abs(force_error))
                ),
                "source_relative_difference_to_56A": float(
                    np.linalg.norm(source_error)
                    / max(np.linalg.norm(reference_source), 1.0e-15)
                ),
            }
        )
    tail = comparisons[-2]
    tail_summary = {
        "total_energy_abs_eV": abs(tail["total_energy_difference_to_56A_eV"]),
        "continuum_energy_abs_eV": abs(tail["continuum_energy_difference_to_56A_eV"]),
        "force_rms_eV_per_A": tail["force_rms_difference_to_56A_eV_per_A"],
        "force_max_eV_per_A": tail["force_max_difference_to_56A_eV_per_A"],
        "source_relative": tail["source_relative_difference_to_56A"],
    }
    gates = {
        "tail_total_energy_abs_le_1e-4_eV": (
            tail_summary["total_energy_abs_eV"] <= TAIL_TOTAL_ENERGY_TOLERANCE_EV
        ),
        "tail_continuum_energy_abs_le_1e-4_eV": (
            tail_summary["continuum_energy_abs_eV"]
            <= TAIL_CONTINUUM_ENERGY_TOLERANCE_EV
        ),
        "tail_force_rms_le_1e-4_eV_per_A": (
            tail_summary["force_rms_eV_per_A"] <= TAIL_FORCE_RMS_TOLERANCE_EV_PER_A
        ),
        "tail_force_max_le_5e-4_eV_per_A": (
            tail_summary["force_max_eV_per_A"] <= TAIL_FORCE_MAX_TOLERANCE_EV_PER_A
        ),
        "tail_source_relative_le_1e-5": (
            tail_summary["source_relative"] <= TAIL_SOURCE_RELATIVE_TOLERANCE
        ),
    }
    return {
        "schema_version": "route2-fixed-box-convergence-summary-v1",
        "reference_box_length_A": BOX_LENGTHS_A[-1],
        "tail_pair": list(BOX_LENGTHS_A[-2:]),
        "records": validated,
        "comparisons_to_56A": comparisons,
        "tail": tail_summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


__all__ = [
    "BOX_LENGTHS_A",
    "MAXIMUM_ADJOINT_RESIDUAL",
    "MAXIMUM_PRIMAL_RESIDUAL",
    "TAIL_CONTINUUM_ENERGY_TOLERANCE_EV",
    "TAIL_FORCE_MAX_TOLERANCE_EV_PER_A",
    "TAIL_FORCE_RMS_TOLERANCE_EV_PER_A",
    "TAIL_SOURCE_RELATIVE_TOLERANCE",
    "TAIL_TOTAL_ENERGY_TOLERANCE_EV",
    "summarize_box_convergence",
]
