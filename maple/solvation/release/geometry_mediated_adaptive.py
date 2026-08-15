"""Adaptive same-scalar directional audit for geometry-mediated Route 2.

This module deliberately delegates numerical differentiation to
``scipy.differentiate.derivative``.  It records every scalar-energy abscissa,
replays the pinned SciPy algorithm from those raw values, and independently
checks the analytic gradient and the complete sampled model/cavity stratum.

The audit is diagnostic-only.  A successful record is local evidence inside
one guarded stratum; it is not a global smoothness proof or task admission.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
import math
from typing import Any

import numpy as np

from .geometry_mediated import (
    GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A,
    GEOMETRY_MEDIATED_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A,
    GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_TOLERANCE,
    GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A,
)

GEOMETRY_MEDIATED_ADAPTIVE_FD_SCHEMA_VERSION = (
    "route2-geometry-mediated-adaptive-directional-audit-v1"
)
GEOMETRY_MEDIATED_ADAPTIVE_FD_SCIPY_VERSION = "1.17.1"
GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER = 4
GEOMETRY_MEDIATED_ADAPTIVE_FD_INITIAL_STEP_A = 6.4e-3
GEOMETRY_MEDIATED_ADAPTIVE_FD_STEP_FACTOR = 2.0
GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER = 8
GEOMETRY_MEDIATED_ADAPTIVE_FD_ATOL_EV_PER_A = (
    GEOMETRY_MEDIATED_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
)
GEOMETRY_MEDIATED_ADAPTIVE_FD_RTOL = GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_TOLERANCE

_CONTINUUM_EVENT_MARGIN_FIELDS = (
    "minimum_point_source_shell_margin_angstrom",
    "minimum_sphere_tangency_margin_angstrom",
    "minimum_cavity_active_set_clearance_angstrom",
)


def _require_scipy_derivative():
    scipy_version = version("scipy")
    if scipy_version != GEOMETRY_MEDIATED_ADAPTIVE_FD_SCIPY_VERSION:
        raise RuntimeError(
            "The adaptive Route-2 derivative audit is pinned to SciPy "
            f"{GEOMETRY_MEDIATED_ADAPTIVE_FD_SCIPY_VERSION}; received "
            f"{scipy_version}."
        )
    from scipy.differentiate import derivative

    return derivative


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    digest = value.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return digest


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _adaptive_protocol() -> dict[str, object]:
    return {
        "implementation": "scipy.differentiate.derivative",
        "scipy_version": GEOMETRY_MEDIATED_ADAPTIVE_FD_SCIPY_VERSION,
        "order": GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER,
        "initial_step_A": GEOMETRY_MEDIATED_ADAPTIVE_FD_INITIAL_STEP_A,
        "step_factor": GEOMETRY_MEDIATED_ADAPTIVE_FD_STEP_FACTOR,
        "maxiter": GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER,
        "step_direction": 0,
        "atol_eV_per_A": GEOMETRY_MEDIATED_ADAPTIVE_FD_ATOL_EV_PER_A,
        "rtol": GEOMETRY_MEDIATED_ADAPTIVE_FD_RTOL,
    }


def _run_derivative(function: Callable[[np.ndarray], np.ndarray]):
    derivative = _require_scipy_derivative()
    return derivative(
        function,
        0.0,
        tolerances={
            "atol": GEOMETRY_MEDIATED_ADAPTIVE_FD_ATOL_EV_PER_A,
            "rtol": GEOMETRY_MEDIATED_ADAPTIVE_FD_RTOL,
        },
        maxiter=GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER,
        order=GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER,
        initial_step=GEOMETRY_MEDIATED_ADAPTIVE_FD_INITIAL_STEP_A,
        step_factor=GEOMETRY_MEDIATED_ADAPTIVE_FD_STEP_FACTOR,
        step_direction=0,
    )


def _result_record(result: object) -> dict[str, object]:
    return {
        "success": bool(getattr(result, "success")),
        "status": int(getattr(result, "status")),
        "derivative_eV_per_A": _finite_float(
            getattr(result, "df"), name="SciPy derivative"
        ),
        "error_estimate_eV_per_A": _finite_float(
            getattr(result, "error"),
            name="SciPy derivative error estimate",
            nonnegative=True,
        ),
        "iterations": _positive_integer(
            getattr(result, "nit"), name="SciPy derivative iteration count"
        ),
        "nfev": _positive_integer(
            getattr(result, "nfev"), name="SciPy derivative evaluation count"
        ),
        "x_A": _finite_float(getattr(result, "x"), name="SciPy derivative x"),
    }


def capture_geometry_mediated_adaptive_directional_samples(
    sample_at_displacement: Callable[[float], Mapping[str, object]],
) -> dict[str, object]:
    """Run the pinned adaptive derivative and retain every raw sample.

    ``sample_at_displacement`` must return the total scalar energy, exact
    coordinates, and model/continuum topology records for the requested signed
    displacement in Angstrom.  Repeated SciPy requests are served from one
    in-memory cache so each abscissa has exactly one scientific measurement.
    """

    if not callable(sample_at_displacement):
        raise TypeError("sample_at_displacement must be callable.")
    cache: dict[str, dict[str, object]] = {}

    def evaluate(displacements: np.ndarray) -> np.ndarray:
        values = np.asarray(displacements, dtype=float)
        energies = np.empty_like(values)
        for index in np.ndindex(values.shape):
            displacement = float(values[index])
            key = displacement.hex()
            if key not in cache:
                raw = _mapping(
                    sample_at_displacement(displacement),
                    name="adaptive displacement sample",
                )
                energy = _finite_float(raw.get("energy_eV"), name="sample energy")
                positions = np.asarray(raw.get("positions_A"), dtype=float)
                if (
                    positions.ndim != 2
                    or positions.shape[0] < 2
                    or positions.shape[1] != 3
                    or not np.all(np.isfinite(positions))
                ):
                    raise ValueError(
                        "adaptive sample positions must be finite with shape (N,3)."
                    )
                cache[key] = {
                    "displacement_A": displacement,
                    "displacement_hex": key,
                    "energy_eV": energy,
                    "positions_A": positions.tolist(),
                    "model_topology": dict(
                        _mapping(raw.get("model_topology"), name="model topology")
                    ),
                    "continuum_topology": dict(
                        _mapping(
                            raw.get("continuum_topology"),
                            name="continuum topology",
                        )
                    ),
                }
            energies[index] = float(cache[key]["energy_eV"])
        return energies

    result = _run_derivative(evaluate)
    samples = sorted(cache.values(), key=lambda record: float(record["displacement_A"]))
    result_record = _result_record(result)
    if result_record["nfev"] != len(samples):
        raise RuntimeError(
            "SciPy evaluation count disagrees with the unique adaptive sample cache."
        )
    return {
        "schema_version": GEOMETRY_MEDIATED_ADAPTIVE_FD_SCHEMA_VERSION,
        "protocol": _adaptive_protocol(),
        "samples": samples,
        "scipy_result": result_record,
    }


def _validate_protocol(raw: object) -> None:
    protocol = _mapping(raw, name="adaptive protocol")
    expected = _adaptive_protocol()
    if set(protocol) != set(expected):
        raise ValueError("adaptive protocol schema changed from the frozen contract.")
    for name, value in expected.items():
        if protocol[name] != value:
            raise ValueError(f"adaptive protocol field {name!r} changed.")


def _model_topology(record: Mapping[str, object]) -> tuple[str, float]:
    return (
        _sha256(record.get("topology_sha256"), name="model topology"),
        _finite_float(
            record.get("minimum_cutoff_margin_angstrom"),
            name="model cutoff margin",
            nonnegative=True,
        ),
    )


def _continuum_topology(
    record: Mapping[str, object],
) -> tuple[str, int, dict[str, float]]:
    digest = _sha256(record.get("cavity_topology_sha256"), name="continuum topology")
    count = _positive_integer(
        record.get("cavity_active_node_count", record.get("coefficient_count")),
        name="continuum topology dimension",
    )
    margins = {
        name: _finite_float(record[name], name=name, nonnegative=True)
        for name in _CONTINUUM_EVENT_MARGIN_FIELDS
        if name in record
    }
    return digest, count, margins


def _relative_displacement_bound(displacement: np.ndarray) -> float:
    bound = 0.0
    for first in range(len(displacement)):
        for second in range(first + 1, len(displacement)):
            bound = max(
                bound,
                float(np.linalg.norm(displacement[second] - displacement[first])),
            )
    return bound


def _same_float(raw: object, expected: float, *, name: str) -> None:
    value = _finite_float(raw, name=name)
    if value != expected:
        raise ValueError(f"SciPy result disagrees for {name}.")


def summarize_geometry_mediated_adaptive_directional_audit(
    *,
    analytic_gradient_eV_per_A: object,
    direction: object,
    center_positions_A: object,
    center_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    adaptive_record: Mapping[str, object],
    reciprocity_audit: Mapping[str, object],
) -> dict[str, object]:
    """Recompute a pinned adaptive derivative and all sampled event gates."""

    if adaptive_record.get("schema_version") != (
        GEOMETRY_MEDIATED_ADAPTIVE_FD_SCHEMA_VERSION
    ):
        raise ValueError("adaptive directional schema version is invalid.")
    _validate_protocol(adaptive_record.get("protocol"))
    analytic = np.asarray(analytic_gradient_eV_per_A, dtype=float)
    positions = np.asarray(center_positions_A, dtype=float)
    tangent = np.asarray(direction, dtype=float)
    if (
        analytic.ndim != 2
        or analytic.shape[0] < 2
        or analytic.shape[1] != 3
        or positions.shape != analytic.shape
        or tangent.shape != analytic.shape
        or not np.all(np.isfinite(analytic))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(tangent))
    ):
        raise ValueError(
            "analytic gradient, positions, and direction must be finite matching "
            "(N,3) arrays."
        )
    if not np.isclose(np.linalg.norm(tangent), 1.0, rtol=0.0, atol=2.0e-14):
        raise ValueError("adaptive direction must have unit Euclidean norm.")
    if not np.allclose(np.sum(tangent, axis=0), 0.0, rtol=0.0, atol=2.0e-14):
        raise ValueError("adaptive direction must contain no rigid translation.")

    samples_raw = adaptive_record.get("samples")
    if not isinstance(samples_raw, Sequence) or isinstance(samples_raw, (str, bytes)):
        raise TypeError("adaptive samples must be a sequence.")
    samples: dict[str, dict[str, Any]] = {}
    for raw_value in samples_raw:
        raw = _mapping(raw_value, name="adaptive sample")
        displacement = _finite_float(
            raw.get("displacement_A"), name="adaptive displacement"
        )
        key = raw.get("displacement_hex")
        if key != displacement.hex() or key in samples:
            raise ValueError("adaptive sample displacement identity is invalid.")
        sample_positions = np.asarray(raw.get("positions_A"), dtype=float)
        expected_positions = positions + displacement * tangent
        if not np.array_equal(sample_positions, expected_positions):
            raise ValueError(
                "adaptive sample coordinates do not match its displacement."
            )
        samples[key] = {
            "displacement_A": displacement,
            "energy_eV": _finite_float(raw.get("energy_eV"), name="sample energy"),
            "positions_A": sample_positions,
            "model_topology": _mapping(
                raw.get("model_topology"), name="sample model topology"
            ),
            "continuum_topology": _mapping(
                raw.get("continuum_topology"), name="sample continuum topology"
            ),
        }
    if not samples:
        raise ValueError("adaptive samples must be non-empty.")

    requested: set[str] = set()

    def replay(displacements: np.ndarray) -> np.ndarray:
        values = np.asarray(displacements, dtype=float)
        energies = np.empty_like(values)
        for index in np.ndindex(values.shape):
            key = float(values[index]).hex()
            requested.add(key)
            if key not in samples:
                raise ValueError(
                    "adaptive replay requested an unrecorded displacement."
                )
            energies[index] = samples[key]["energy_eV"]
        return energies

    result = _run_derivative(replay)
    if set(samples) != requested:
        raise ValueError("adaptive record contains an unrequested displacement.")
    recomputed = _result_record(result)
    raw_result = _mapping(adaptive_record.get("scipy_result"), name="SciPy result")
    if set(raw_result) != set(recomputed):
        raise ValueError("SciPy result schema changed from the frozen contract.")
    for name in ("success", "status", "iterations", "nfev"):
        if raw_result[name] != recomputed[name]:
            raise ValueError(f"SciPy result disagrees for {name}.")
    for name in (
        "derivative_eV_per_A",
        "error_estimate_eV_per_A",
        "x_A",
    ):
        _same_float(raw_result[name], float(recomputed[name]), name=name)

    center_model_hash, center_model_margin = _model_topology(center_model_topology)
    center_continuum_hash, center_continuum_count, center_event_margins = (
        _continuum_topology(center_continuum_topology)
    )
    all_event_fields = set(center_event_margins)
    parsed_topologies: dict[str, tuple[str, float, str, int, dict[str, float]]] = {}
    for key, sample in samples.items():
        model_hash, model_margin = _model_topology(sample["model_topology"])
        continuum_hash, continuum_count, event_margins = _continuum_topology(
            sample["continuum_topology"]
        )
        all_event_fields.update(event_margins)
        parsed_topologies[key] = (
            model_hash,
            model_margin,
            continuum_hash,
            continuum_count,
            event_margins,
        )

    continuum_event_applicable = bool(all_event_fields)
    all_event_margins_available = (
        continuum_event_applicable
        and all(
            set(event_margins) == all_event_fields
            for _, _, _, _, event_margins in parsed_topologies.values()
        )
        and set(center_event_margins) == all_event_fields
    )
    topology_records: list[dict[str, object]] = []
    all_same_stratum = True
    all_neighbor_guards = True
    all_event_guards = all_event_margins_available
    for key in sorted(samples, key=lambda item: samples[item]["displacement_A"]):
        sample = samples[key]
        (
            model_hash,
            model_margin,
            continuum_hash,
            continuum_count,
            event_margins,
        ) = parsed_topologies[key]
        displacement = sample["positions_A"] - positions
        relative_bound = _relative_displacement_bound(displacement)
        same_stratum = (
            model_hash == center_model_hash
            and continuum_hash == center_continuum_hash
            and continuum_count == center_continuum_count
        )
        neighbor_lower = min(center_model_margin, model_margin) - relative_bound
        neighbor_gate = neighbor_lower >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
        event_lower_bounds = {
            name: min(center_event_margins[name], event_margins[name]) - relative_bound
            for name in all_event_fields
            if name in center_event_margins and name in event_margins
        }
        event_gate = (
            all_event_margins_available
            and set(event_lower_bounds) == all_event_fields
            and all(
                value >= GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A
                for value in event_lower_bounds.values()
            )
        )
        all_same_stratum = all_same_stratum and same_stratum
        all_neighbor_guards = all_neighbor_guards and neighbor_gate
        all_event_guards = all_event_guards and event_gate
        topology_records.append(
            {
                "displacement_A": sample["displacement_A"],
                "displacement_hex": key,
                "relative_displacement_bound_A": relative_bound,
                "same_model_and_cavity_stratum": same_stratum,
                "neighbor_cutoff_segment_lower_bound_A": neighbor_lower,
                "neighbor_cutoff_guard_passed": neighbor_gate,
                "continuum_event_segment_lower_bounds_A": event_lower_bounds,
                "continuum_event_guard_passed": event_gate,
                "gate_passed": same_stratum and neighbor_gate and event_gate,
            }
        )

    analytic_directional = float(np.vdot(analytic, tangent))
    numerical_directional = float(recomputed["derivative_eV_per_A"])
    absolute_error = abs(analytic_directional - numerical_directional)
    relative_error = absolute_error / max(abs(numerical_directional), 1.0e-15)
    relative_applies = (
        abs(numerical_directional)
        >= GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A
    )
    analytic_gate = absolute_error <= (
        GEOMETRY_MEDIATED_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
    ) and (
        not relative_applies
        or relative_error <= GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_TOLERANCE
    )
    convergence_gate = (
        recomputed["success"] is True
        and recomputed["status"] == 0
        and float(recomputed["error_estimate_eV_per_A"])
        < GEOMETRY_MEDIATED_ADAPTIVE_FD_ATOL_EV_PER_A
        + GEOMETRY_MEDIATED_ADAPTIVE_FD_RTOL * abs(numerical_directional)
    )
    reciprocity_gate = reciprocity_audit.get("gate_passed") is True
    topology_gate = (
        all_same_stratum
        and all_neighbor_guards
        and continuum_event_applicable
        and all_event_margins_available
        and all_event_guards
    )
    return {
        "schema_version": GEOMETRY_MEDIATED_ADAPTIVE_FD_SCHEMA_VERSION,
        "protocol": _adaptive_protocol(),
        "scipy_result": recomputed,
        "scipy_result_recomputed": True,
        "analytic_directional_eV_per_A": analytic_directional,
        "adaptive_directional_eV_per_A": numerical_directional,
        "absolute_error_eV_per_A": absolute_error,
        "relative_error": relative_error,
        "relative_gate_applies": relative_applies,
        "adaptive_convergence_gate_passed": convergence_gate,
        "analytic_agreement_gate_passed": analytic_gate,
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "topology": {
            "center_model_topology_sha256": center_model_hash,
            "center_continuum_topology_sha256": center_continuum_hash,
            "sample_count": len(topology_records),
            "all_samples_same_stratum": all_same_stratum,
            "all_neighbor_cutoff_guards_passed": all_neighbor_guards,
            "continuum_event_guard_applicable": continuum_event_applicable,
            "continuum_event_margin_fields": sorted(all_event_fields),
            "all_continuum_event_margins_available": all_event_margins_available,
            "all_continuum_event_guards_passed": all_event_guards,
            "all_segment_guards_passed": topology_gate,
            "records": topology_records,
        },
        "gate_passed": (
            convergence_gate and analytic_gate and reciprocity_gate and topology_gate
        ),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }


__all__ = [
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_ATOL_EV_PER_A",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_INITIAL_STEP_A",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_MAXITER",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_ORDER",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_RTOL",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_SCHEMA_VERSION",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_SCIPY_VERSION",
    "GEOMETRY_MEDIATED_ADAPTIVE_FD_STEP_FACTOR",
    "capture_geometry_mediated_adaptive_directional_samples",
    "summarize_geometry_mediated_adaptive_directional_audit",
]
