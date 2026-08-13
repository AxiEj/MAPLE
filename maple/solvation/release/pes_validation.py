"""Pure numerical primitives for same-scalar PES evidence.

These helpers know nothing about MACE, a continuum backend, or capability
admission.  They define and evaluate preregistered geometry/path diagnostics so
the real-stack runner does not validate its own differentiation formulas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-4
_DIRECTIONAL_RELATIVE_TOLERANCE = 2.0e-3
_DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A = 1.0e-3
_CARTESIAN_RMS_TOLERANCE_EV_PER_A = 5.0e-4
_CARTESIAN_MAXIMUM_TOLERANCE_EV_PER_A = 2.0e-3
_CARTESIAN_PLATEAU_RMS_EV_PER_A = 5.0e-5
_CARTESIAN_PLATEAU_MAXIMUM_EV_PER_A = 2.0e-4
_CARTESIAN_PLATEAU_SPREAD_FACTOR = 1.5
_CARTESIAN_MINIMUM_OBSERVED_ORDER = 1.5
_CARTESIAN_TERMINAL_GROWTH_FACTOR = 1.25
_LOOP_ABSOLUTE_TOLERANCE_EV = 1.0e-5
_LOOP_RELATIVE_TOLERANCE = 1.0e-3


def _positions(values: object, name: str = "positions") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (3, 3).")
    return np.array(array, copy=True)


def _normalized_internal(values: np.ndarray, name: str) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    direction = direction - np.mean(direction, axis=0, keepdims=True)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} produced a singular internal direction.")
    return direction / norm


def _gram_schmidt(
    candidate: np.ndarray,
    accepted: Sequence[np.ndarray],
    name: str,
) -> np.ndarray:
    orthogonal = np.array(candidate, copy=True)
    for prior in accepted:
        orthogonal -= float(np.vdot(prior, orthogonal)) * prior
    return _normalized_internal(orthogonal, name)


def water_vibrational_directions(positions: object) -> dict[str, np.ndarray]:
    """Return three deterministic translation-free orthonormal water modes.

    These are Cartesian collective coordinates used only to preregister a
    compact distorted-water panel.  They are not mass-weighted normal modes.
    """

    geometry = _positions(positions)
    first_bond = geometry[1] - geometry[0]
    second_bond = geometry[2] - geometry[0]
    first_norm = float(np.linalg.norm(first_bond))
    second_norm = float(np.linalg.norm(second_bond))
    if min(first_norm, second_norm) <= 1.0e-12:
        raise ValueError("Water O-H bonds must be nonzero.")
    first_unit = first_bond / first_norm
    second_unit = second_bond / second_norm

    symmetric_seed = np.stack((-(first_unit + second_unit), first_unit, second_unit))
    asymmetric_seed = np.stack((first_unit - second_unit, -first_unit, second_unit))
    bend_seed = np.stack((-(first_unit + second_unit), second_unit, first_unit))
    symmetric = _gram_schmidt(symmetric_seed, (), "symmetric_stretch")
    asymmetric = _gram_schmidt(asymmetric_seed, (symmetric,), "asymmetric_stretch")
    bend = _gram_schmidt(bend_seed, (symmetric, asymmetric), "bend")
    return {
        "symmetric_stretch": symmetric,
        "asymmetric_stretch": asymmetric,
        "bend": bend,
    }


def displace_positions(
    base_positions: object,
    directions: Mapping[str, object],
    coefficients_A: Mapping[str, float],
) -> np.ndarray:
    """Displace a geometry along named normalized Cartesian directions."""

    result = _positions(base_positions, "base_positions")
    if not coefficients_A:
        return result
    for name, coefficient in coefficients_A.items():
        if name not in directions:
            raise ValueError(f"unknown collective direction {name!r}.")
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("collective-coordinate coefficients must be finite.")
        direction = _positions(directions[name], f"direction {name!r}")
        result += value * direction
    return result


def water_geometry_descriptors(positions: object) -> dict[str, float]:
    geometry = _positions(positions)
    first = geometry[1] - geometry[0]
    second = geometry[2] - geometry[0]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if min(first_norm, second_norm) <= 1.0e-12:
        raise ValueError("Water O-H bonds must be nonzero.")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return {"oh1_A": first_norm, "oh2_A": second_norm, "hoh_angle_deg": angle}


def closed_rectangular_loop(
    *, subdivisions_per_edge: int = 4
) -> tuple[tuple[float, float], ...]:
    """Return a counter-clockwise closed rectangle with Simpson-compatible edges."""

    if (
        type(subdivisions_per_edge) is not int
        or subdivisions_per_edge < 2
        or subdivisions_per_edge % 2
    ):
        raise ValueError("subdivisions_per_edge must be a positive even integer.")
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    points: list[tuple[float, float]] = []
    for start, stop in zip(corners, (*corners[1:], corners[0])):
        for index in range(subdivisions_per_edge):
            fraction = index / subdivisions_per_edge
            points.append(
                (
                    start[0] + fraction * (stop[0] - start[0]),
                    start[1] + fraction * (stop[1] - start[1]),
                )
            )
    points.append(points[0])
    return tuple(points)


def reverse_closed_path(
    path: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    values = tuple((float(first), float(second)) for first, second in path)
    if len(values) < 2 or values[0] != values[-1]:
        raise ValueError("path must be closed before reversal.")
    return tuple(reversed(values))


def summarize_directional_derivatives(
    analytic_derivative_eV_per_A: float,
    energy_samples: Sequence[tuple[float, float, float]],
) -> dict[str, object]:
    """Compare one analytic derivative with central energy differences."""

    analytic = float(analytic_derivative_eV_per_A)
    if not math.isfinite(analytic):
        raise ValueError("analytic directional derivative must be finite.")
    records: list[dict[str, object]] = []
    seen: set[float] = set()
    for raw_step, raw_plus, raw_minus in energy_samples:
        step, plus, minus = float(raw_step), float(raw_plus), float(raw_minus)
        if not all(math.isfinite(value) for value in (step, plus, minus)) or step <= 0:
            raise ValueError("directional samples must be finite with positive steps.")
        if step in seen:
            raise ValueError("directional displacement steps must be unique.")
        seen.add(step)
        finite_difference = (plus - minus) / (2.0 * step)
        absolute_error = abs(analytic - finite_difference)
        relative_applies = (
            abs(finite_difference) >= _DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A
        )
        relative_error = absolute_error / max(abs(finite_difference), 1.0e-15)
        absolute_pass = absolute_error <= _DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
        relative_pass = (
            not relative_applies or relative_error <= _DIRECTIONAL_RELATIVE_TOLERANCE
        )
        records.append(
            {
                "step_A": step,
                "analytic_eV_per_A": analytic,
                "finite_difference_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": absolute_error,
                "relative_error": relative_error,
                "relative_gate_applies": relative_applies,
                "gates": {
                    "absolute_le_5e-4_eV_per_A": absolute_pass,
                    "relative_le_2e-3_away_from_zero": relative_pass,
                },
                "gate_passed": absolute_pass and relative_pass,
            }
        )
    if not records:
        raise ValueError("at least one directional sample is required.")
    return {
        "records": records,
        "all_gates_passed": all(record["gate_passed"] for record in records),
    }


def summarize_cartesian_force_differences(
    analytic_gradient_eV_per_A: object,
    finite_difference_samples: Sequence[tuple[float, object]],
) -> dict[str, object]:
    """Gate full Cartesian gradients at multiple central-difference steps.

    Steps must be supplied in strictly decreasing order.  Above ten percent of
    the preregistered RMS/maximum error budgets, the first-to-last refinement
    must show better than first-order behavior (observed order at least 1.5)
    and the terminal RMS error may not grow by more than 25 percent.  Once all
    step errors lie below those ten-percent floors, a numerical plateau is
    accepted explicitly rather than manufacturing an apparent convergence
    order from roundoff.
    """

    analytic = np.asarray(analytic_gradient_eV_per_A, dtype=float)
    if (
        analytic.ndim != 2
        or analytic.shape[0] < 1
        or analytic.shape[1] != 3
        or not np.all(np.isfinite(analytic))
    ):
        raise ValueError("analytic Cartesian gradient must be finite with shape (N, 3).")
    records: list[dict[str, object]] = []
    previous_step = math.inf
    for raw_step, raw_finite_difference in finite_difference_samples:
        step = float(raw_step)
        finite_difference = np.asarray(raw_finite_difference, dtype=float)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or step >= previous_step
            or finite_difference.shape != analytic.shape
            or not np.all(np.isfinite(finite_difference))
        ):
            raise ValueError(
                "Cartesian samples require strictly decreasing positive steps and "
                "finite gradients matching the analytic shape."
            )
        previous_step = step
        error = analytic - finite_difference
        rms = float(np.sqrt(np.mean(error**2)))
        maximum = float(np.max(np.abs(error)))
        records.append(
            {
                "step_A": step,
                "analytic_gradient_eV_per_A": analytic.tolist(),
                "finite_difference_gradient_eV_per_A": finite_difference.tolist(),
                "error_eV_per_A": error.tolist(),
                "rms_error_eV_per_A": rms,
                "maximum_error_eV_per_A": maximum,
                "relative_frobenius_error": float(
                    np.linalg.norm(error)
                    / max(float(np.linalg.norm(finite_difference)), 1.0e-15)
                ),
                "gates": {
                    "rms_le_5e-4_eV_per_A": (
                        rms <= _CARTESIAN_RMS_TOLERANCE_EV_PER_A
                    ),
                    "maximum_le_2e-3_eV_per_A": (
                        maximum <= _CARTESIAN_MAXIMUM_TOLERANCE_EV_PER_A
                    ),
                },
            }
        )
    if len(records) < 3:
        raise ValueError("Cartesian force validation requires at least three steps.")

    first = records[0]
    last = records[-1]
    first_rms = float(first["rms_error_eV_per_A"])
    last_rms = float(last["rms_error_eV_per_A"])
    ratio = float(first["step_A"]) / float(last["step_A"])
    observed_order = (
        None
        if last_rms == 0.0
        else math.log(max(first_rms, 1.0e-300) / last_rms) / math.log(ratio)
    )
    rms_values = [float(record["rms_error_eV_per_A"]) for record in records]
    low_error_plateau = all(
        float(record["rms_error_eV_per_A"])
        <= _CARTESIAN_PLATEAU_RMS_EV_PER_A
        and float(record["maximum_error_eV_per_A"])
        <= _CARTESIAN_PLATEAU_MAXIMUM_EV_PER_A
        for record in records
    ) and max(rms_values) <= _CARTESIAN_PLATEAU_SPREAD_FACTOR * max(
        min(rms_values), 1.0e-300
    )
    terminal_nondivergent = last_rms <= max(
        _CARTESIAN_PLATEAU_RMS_EV_PER_A,
        _CARTESIAN_TERMINAL_GROWTH_FACTOR
        * float(records[-2]["rms_error_eV_per_A"]),
    )
    order_or_plateau = low_error_plateau or (
        last_rms == 0.0 and first_rms > 0.0
    ) or (
        observed_order is not None
        and observed_order >= _CARTESIAN_MINIMUM_OBSERVED_ORDER
    )
    per_step = all(all(record["gates"].values()) for record in records)
    convergence = {
        "contract": "central-order-or-ten-percent-error-plateau-v1",
        "observed_first_to_last_order": observed_order,
        "minimum_observed_order": _CARTESIAN_MINIMUM_OBSERVED_ORDER,
        "low_error_plateau": low_error_plateau,
        "plateau_rms_eV_per_A": _CARTESIAN_PLATEAU_RMS_EV_PER_A,
        "plateau_maximum_eV_per_A": _CARTESIAN_PLATEAU_MAXIMUM_EV_PER_A,
        "plateau_spread_factor_limit": _CARTESIAN_PLATEAU_SPREAD_FACTOR,
        "terminal_growth_factor_limit": _CARTESIAN_TERMINAL_GROWTH_FACTOR,
        "gates": {
            "central_order_ge_1p5_or_low_error_plateau": order_or_plateau,
            "terminal_rms_nondivergent": terminal_nondivergent,
        },
    }
    return {
        "records": records,
        "convergence": convergence,
        "all_gates_passed": per_step
        and order_or_plateau
        and terminal_nondivergent,
    }


def _path_arrays(
    positions: Sequence[object], forces: Sequence[object]
) -> tuple[np.ndarray, np.ndarray]:
    position_values = np.asarray(positions, dtype=float)
    force_values = np.asarray(forces, dtype=float)
    if (
        position_values.ndim != 3
        or position_values.shape[2] != 3
        or position_values.shape != force_values.shape
        or position_values.shape[0] < 3
        or not np.all(np.isfinite(position_values))
        or not np.all(np.isfinite(force_values))
    ):
        raise ValueError("positions and forces must be equally shaped finite paths.")
    if not np.array_equal(position_values[0], position_values[-1]):
        raise ValueError("path must be geometrically closed.")
    return position_values, force_values


def closed_loop_work(
    positions: Sequence[object],
    forces: Sequence[object],
    *,
    subdivisions_per_edge: int,
) -> dict[str, object]:
    """Integrate force work over a four-edge loop using composite Simpson."""

    if (
        type(subdivisions_per_edge) is not int
        or subdivisions_per_edge < 2
        or subdivisions_per_edge % 2
    ):
        raise ValueError("subdivisions_per_edge must be a positive even integer.")
    position_values, force_values = _path_arrays(positions, forces)
    expected = 4 * subdivisions_per_edge + 1
    if len(position_values) != expected:
        raise ValueError(f"four-edge loop must contain exactly {expected} points.")

    edge_work: list[float] = []
    edge_absolute_work: list[float] = []
    for edge in range(4):
        start = edge * subdivisions_per_edge
        stop = start + subdivisions_per_edge
        segment_positions = position_values[start : stop + 1]
        segment_forces = force_values[start : stop + 1]
        increment = (
            segment_positions[-1] - segment_positions[0]
        ) / subdivisions_per_edge
        projected = np.einsum("nij,ij->n", segment_forces, increment)
        weights = np.ones(subdivisions_per_edge + 1)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        edge_work.append(float(np.dot(weights, projected) / 3.0))
        edge_absolute_work.append(float(np.dot(weights, np.abs(projected)) / 3.0))
    work = float(sum(edge_work))
    absolute_work = float(sum(edge_absolute_work))
    threshold = max(
        _LOOP_ABSOLUTE_TOLERANCE_EV,
        _LOOP_RELATIVE_TOLERANCE * absolute_work,
    )
    return {
        "integration": "composite-simpson-four-equal-edges",
        "subdivisions_per_edge": subdivisions_per_edge,
        "edge_work_eV": edge_work,
        "edge_absolute_work_eV": edge_absolute_work,
        "simpson_work_eV": work,
        "sum_absolute_work_eV": absolute_work,
        "gate_threshold_eV": threshold,
        "gate_passed": abs(work) <= threshold,
    }


__all__ = [
    "closed_loop_work",
    "closed_rectangular_loop",
    "displace_positions",
    "reverse_closed_path",
    "summarize_cartesian_force_differences",
    "summarize_directional_derivatives",
    "water_geometry_descriptors",
    "water_vibrational_directions",
]
