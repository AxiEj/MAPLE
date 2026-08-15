"""Fail-closed diagnostics for the AIMNet2 geometry-mediated scalar.

These pure helpers validate measurements produced by a real-stack runner.
They do not import AIMNet2, torch, pyddx, or ASE, and they never admit a Route-2
capability.  In particular, a passing local directional audit remains
non-admissible while pyddx exposes no measured post-solve residual and while
the laboratory-frame cavity grid fails the frozen rigid-rotation contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from maple.solvation.coupling.geometry_mediated import (
    GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E,
    GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE,
    GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E,
    GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV,
    GEOMETRY_MEDIATED_RECIPROCITY_PROBES,
    GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE,
    GEOMETRY_MEDIATED_RECIPROCITY_SEED,
    GEOMETRY_MEDIATED_SOURCE_GRADIENT_RECIPROCITY_RELATIVE_TOLERANCE,
)

from .pes_validation import summarize_cartesian_force_differences

GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-admission-audit-v2"
)
GEOMETRY_MEDIATED_COORDINATE_SEED = 20260816
GEOMETRY_MEDIATED_ROTATION_SEED = 20260817
GEOMETRY_MEDIATED_ROTATION_COUNT = 3
GEOMETRY_MEDIATED_COORDINATE_STEPS_A = (1.0e-3, 5.0e-4, 2.5e-4)
GEOMETRY_MEDIATED_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-4
GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_TOLERANCE = 2.0e-3
GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A = 1.0e-3
GEOMETRY_MEDIATED_NUMERICAL_PLATEAU_EV_PER_A = 5.0e-5
GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A = 2.0e-2
GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A = 2.0e-2
GEOMETRY_MEDIATED_ROTATION_ENERGY_TOLERANCE_EV = 1.0e-6
GEOMETRY_MEDIATED_ROTATION_FORCE_RELATIVE_TOLERANCE = 1.0e-4
GEOMETRY_MEDIATED_ROTATION_SOURCE_RELATIVE_TOLERANCE = 1.0e-6
GEOMETRY_MEDIATED_NET_FORCE_TOLERANCE_EV_PER_A = 1.0e-5
GEOMETRY_MEDIATED_TORQUE_TOLERANCE_EV = 1.0e-4
GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV = 1.0e-10
GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE = 1.0e-10
GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A = 1.0e-9


def _array(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.array(result, copy=True)


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _model_topology(record: Mapping[str, object]) -> tuple[str, float | None]:
    digest = _sha(record.get("topology_sha256"), name="model topology")
    raw_margin = record.get("minimum_cutoff_margin_angstrom")
    if raw_margin is None:
        margin = None
    else:
        if isinstance(raw_margin, bool):
            raise TypeError("model cutoff margin must be numeric.")
        margin = float(raw_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("model cutoff margin must be finite and non-negative.")
    return digest, margin


def _optional_margin(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    margin = float(value)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return margin


def _continuum_topology(
    record: Mapping[str, object],
) -> tuple[str, float | None, float | None, bool, bool]:
    count = record.get("cavity_active_node_count", record.get("coefficient_count"))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            "continuum topology requires a positive active-node or coefficient count."
        )
    point_margin = _optional_margin(
        record.get("minimum_point_source_shell_margin_angstrom"),
        name="point-source shell margin",
    )
    sphere_margin = _optional_margin(
        record.get("minimum_sphere_tangency_margin_angstrom"),
        name="sphere-tangency margin",
    )
    point_marker = record.get("point_source_topology_sha256")
    sphere_marker = record.get("sphere_pair_topology_sha256")
    if point_marker is not None:
        _sha(point_marker, name="point-source topology")
    if sphere_marker is not None:
        _sha(sphere_marker, name="sphere-pair topology")
    return (
        _sha(record.get("cavity_topology_sha256"), name="continuum topology"),
        point_margin,
        sphere_margin,
        point_marker is not None or point_margin is not None,
        sphere_marker is not None or sphere_margin is not None,
    )


def _continuum_margin_status(
    *margins: float | None,
    applicable: bool | None = None,
) -> tuple[bool, bool, bool]:
    """Return applicability, availability, and the optional event-margin gate."""

    detected = any(value is not None for value in margins)
    applicable = detected if applicable is None else applicable or detected
    available = applicable and all(value is not None for value in margins)
    passed = not applicable or (
        available
        and all(
            value is not None and value >= GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A
            for value in margins
        )
    )
    return applicable, available, passed


def _finite_scalar(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _relative_pair(first: float, second: float, *, floor: float = 1.0e-15) -> float:
    return abs(first - second) / max(abs(first), abs(second), floor)


def _assert_recomputed_float(
    raw: object,
    expected: float,
    *,
    name: str,
) -> None:
    value = _finite_scalar(raw, name=name, nonnegative=True)
    if not math.isclose(value, expected, rel_tol=1.0e-13, abs_tol=1.0e-15):
        raise ValueError(f"{name} disagrees with raw reciprocity measurements.")


def summarize_geometry_mediated_reciprocity_audit(
    record: Mapping[str, object],
    *,
    reaction_field: object,
) -> dict[str, object]:
    """Recompute the metric, adjoint, charge-FD, and gauge audit gate."""

    seed = record.get("seed")
    requested = record.get("requested_probe_count")
    effective = record.get("effective_probe_count")
    if seed != GEOMETRY_MEDIATED_RECIPROCITY_SEED:
        raise ValueError("reciprocity seed changed from the scalar contract.")
    if requested != GEOMETRY_MEDIATED_RECIPROCITY_PROBES or effective != requested:
        raise ValueError("reciprocity probe coverage changed from the scalar contract.")
    bilinear = record.get("bilinear_records")
    charge_fd = record.get("charge_directional_fd_records")
    if (
        not isinstance(bilinear, Sequence)
        or isinstance(bilinear, (str, bytes))
        or len(bilinear) != GEOMETRY_MEDIATED_RECIPROCITY_PROBES
        or not isinstance(charge_fd, Sequence)
        or isinstance(charge_fd, (str, bytes))
        or len(charge_fd)
        != GEOMETRY_MEDIATED_RECIPROCITY_PROBES
        * len(GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E)
    ):
        raise ValueError("reciprocity raw-record coverage is incomplete.")

    reciprocity_absolute: list[float] = []
    reciprocity_relative: list[float] = []
    adjoint_absolute: list[float] = []
    adjoint_relative: list[float] = []
    for expected_probe, raw in enumerate(bilinear):
        if not isinstance(raw, Mapping) or raw.get("probe_index") != expected_probe:
            raise ValueError("bilinear reciprocity probe ordering is invalid.")
        left = _finite_scalar(raw.get("left_P_right_eV"), name="left P right")
        right = _finite_scalar(raw.get("right_P_left_eV"), name="right P left")
        adjoint = _finite_scalar(raw.get("apply_adjoint_eV"), name="apply adjoint")
        rec_abs = abs(left - right)
        rec_rel = _relative_pair(left, right)
        adj_abs = abs(left - adjoint)
        adj_rel = _relative_pair(left, adjoint)
        _assert_recomputed_float(
            raw.get("reciprocity_absolute_error_eV"),
            rec_abs,
            name="reciprocity absolute error",
        )
        _assert_recomputed_float(
            raw.get("reciprocity_relative_error"),
            rec_rel,
            name="reciprocity relative error",
        )
        _assert_recomputed_float(
            raw.get("apply_adjoint_absolute_error_eV"),
            adj_abs,
            name="apply-adjoint absolute error",
        )
        _assert_recomputed_float(
            raw.get("apply_adjoint_relative_error"),
            adj_rel,
            name="apply-adjoint relative error",
        )
        reciprocity_absolute.append(rec_abs)
        reciprocity_relative.append(rec_rel)
        adjoint_absolute.append(adj_abs)
        adjoint_relative.append(adj_rel)

    charge_absolute: list[float] = []
    charge_relative: list[float] = []
    expected_charge_records = tuple(
        (probe, step)
        for probe in range(GEOMETRY_MEDIATED_RECIPROCITY_PROBES)
        for step in GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E
    )
    for raw, (expected_probe, expected_step) in zip(
        charge_fd, expected_charge_records, strict=True
    ):
        if (
            not isinstance(raw, Mapping)
            or raw.get("probe_index") != expected_probe
            or _finite_scalar(raw.get("step_e"), name="charge FD step") != expected_step
        ):
            raise ValueError("charge-direction FD probe/step ordering is invalid.")
        analytic = _finite_scalar(raw.get("analytic_eV_per_e"), name="charge analytic")
        finite_difference = _finite_scalar(
            raw.get("finite_difference_eV_per_e"), name="charge finite difference"
        )
        absolute = abs(analytic - finite_difference)
        relative = _relative_pair(analytic, finite_difference, floor=1.0e-12)
        _assert_recomputed_float(
            raw.get("absolute_error_eV_per_e"),
            absolute,
            name="charge FD absolute error",
        )
        _assert_recomputed_float(
            raw.get("relative_error"),
            relative,
            name="charge FD relative error",
        )
        charge_absolute.append(absolute)
        charge_relative.append(relative)

    maxima = {
        "maximum_reciprocity_absolute_error_eV": max(reciprocity_absolute),
        "maximum_reciprocity_relative_error": max(reciprocity_relative),
        "maximum_apply_adjoint_absolute_error_eV": max(adjoint_absolute),
        "maximum_apply_adjoint_relative_error": max(adjoint_relative),
        "maximum_charge_fd_absolute_error_eV_per_e": max(charge_absolute),
        "maximum_charge_fd_relative_error": max(charge_relative),
    }
    for name, expected in maxima.items():
        _assert_recomputed_float(record.get(name), expected, name=name)

    source_gradient_error = _finite_scalar(
        record.get("source_gradient_half_error_eV_per_source_unit"),
        name="source-gradient reciprocity error",
        nonnegative=True,
    )
    gauge_norm = _finite_scalar(
        record.get("charge_gauge_vjp_norm_eV_per_A"),
        name="charge-gauge VJP norm",
        nonnegative=True,
    )
    field = np.asarray(reaction_field, dtype=float)
    if field.ndim != 2 or field.shape[0] < 2 or not np.all(np.isfinite(field)):
        raise ValueError("reaction_field must be a finite two-dimensional array.")
    thresholds = record.get("thresholds")
    expected_thresholds = {
        "reciprocity_absolute_eV": (
            GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
        ),
        "reciprocity_relative": GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE,
        "charge_fd_absolute_eV_per_e": (
            GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E
        ),
        "charge_fd_relative": GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE,
        "charge_gauge_vjp_norm_eV_per_A": (
            GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A
        ),
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(
        expected_thresholds
    ):
        raise ValueError("reciprocity threshold schema changed from the contract.")
    if any(
        _finite_scalar(thresholds[name], name=f"reciprocity threshold {name}")
        != expected
        for name, expected in expected_thresholds.items()
    ):
        raise ValueError("reciprocity thresholds changed from the scalar contract.")
    source_gradient_threshold = (
        GEOMETRY_MEDIATED_SOURCE_GRADIENT_RECIPROCITY_RELATIVE_TOLERANCE
        * max(1.0, float(np.linalg.norm(field)))
    )
    gate = (
        source_gradient_error <= source_gradient_threshold
        and maxima["maximum_reciprocity_absolute_error_eV"]
        <= GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
        and maxima["maximum_reciprocity_relative_error"]
        <= GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE
        and maxima["maximum_apply_adjoint_absolute_error_eV"]
        <= GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
        and maxima["maximum_apply_adjoint_relative_error"]
        <= GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE
        and maxima["maximum_charge_fd_absolute_error_eV_per_e"]
        <= GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E
        and maxima["maximum_charge_fd_relative_error"]
        <= GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE
        and gauge_norm <= GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("reciprocity gate disagrees with raw measurements.")
    return {
        "seed": seed,
        "probe_count": requested,
        **maxima,
        "source_gradient_half_error_eV_per_source_unit": source_gradient_error,
        "source_gradient_threshold_eV_per_source_unit": source_gradient_threshold,
        "charge_gauge_vjp_norm_eV_per_A": gauge_norm,
        "gate_passed": gate,
    }


def geometry_mediated_coordinate_direction(atom_count: int) -> np.ndarray:
    """Return the frozen, translation-free coordinate direction."""

    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 2
    ):
        raise ValueError("coordinate audit requires at least two atoms.")
    result = np.random.default_rng(GEOMETRY_MEDIATED_COORDINATE_SEED).normal(
        size=(atom_count, 3)
    )
    result -= np.mean(result, axis=0, keepdims=True)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("frozen coordinate direction is singular.")
    return result / norm


def geometry_mediated_rotations() -> tuple[np.ndarray, ...]:
    """Return the three preregistered proper rotations."""

    rotations: list[np.ndarray] = []
    for index in range(GEOMETRY_MEDIATED_ROTATION_COUNT):
        matrix = np.random.default_rng(GEOMETRY_MEDIATED_ROTATION_SEED + index).normal(
            size=(3, 3)
        )
        rotation, triangular = np.linalg.qr(matrix)
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        rotation = rotation @ np.diag(signs)
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), rtol=0.0, atol=2.0e-14
        ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=2.0e-14):
            raise RuntimeError("frozen rigid rotation lost proper orthogonality.")
        rotation.setflags(write=False)
        rotations.append(rotation)
    return tuple(rotations)


def geometry_mediated_trial_step_guard(
    *,
    center_positions_A: object,
    trial_positions_A: object,
    center_model_topology: Mapping[str, object],
    trial_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    trial_continuum_topology: Mapping[str, object],
) -> dict[str, object]:
    """Conservatively certify one straight trial segment stays in one stratum.

    Pair distances, point/source shell event functions, and sphere-tangency
    event functions are 1-Lipschitz with respect to pair-relative atomic
    displacement.  Subtracting the maximum pair-relative displacement from
    both endpoint margins therefore gives a deliberately conservative lower
    bound for the full line segment.  The certificate is local: it does not
    establish global smoothness or stationary-solve conditioning.
    """

    center = np.asarray(center_positions_A, dtype=float)
    if (
        center.ndim != 2
        or center.shape[0] < 1
        or center.shape[1] != 3
        or not np.all(np.isfinite(center))
    ):
        raise ValueError("center positions must be finite with shape (N,3).")
    trial = _array(trial_positions_A, shape=center.shape, name="trial positions")
    displacement = trial - center
    relative_displacement_bound = 0.0
    for first in range(len(center)):
        for second in range(first + 1, len(center)):
            relative_displacement_bound = max(
                relative_displacement_bound,
                float(np.linalg.norm(displacement[second] - displacement[first])),
            )

    center_model_hash, center_model_margin = _model_topology(center_model_topology)
    trial_model_hash, trial_model_margin = _model_topology(trial_model_topology)
    (
        center_continuum_hash,
        center_point_margin,
        center_sphere_margin,
        center_point_applicable,
        center_sphere_applicable,
    ) = _continuum_topology(center_continuum_topology)
    (
        trial_continuum_hash,
        trial_point_margin,
        trial_sphere_margin,
        trial_point_applicable,
        trial_sphere_applicable,
    ) = _continuum_topology(trial_continuum_topology)

    def certified_lower_bound(
        first: float | None, second: float | None
    ) -> float | None:
        if first is None or second is None:
            return None
        return min(first, second) - relative_displacement_bound

    neighbor_lower = certified_lower_bound(center_model_margin, trial_model_margin)
    point_lower = certified_lower_bound(center_point_margin, trial_point_margin)
    sphere_lower = certified_lower_bound(center_sphere_margin, trial_sphere_margin)
    point_applicable = center_point_applicable or trial_point_applicable
    sphere_applicable = center_sphere_applicable or trial_sphere_applicable
    model_topology_match = center_model_hash == trial_model_hash
    continuum_topology_match = center_continuum_hash == trial_continuum_hash
    neighbor_gate = (
        neighbor_lower is not None
        and neighbor_lower >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
    )
    point_available = center_point_margin is not None and trial_point_margin is not None
    point_gate = not point_applicable or (
        point_available
        and point_lower is not None
        and point_lower >= GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A
    )
    sphere_available = (
        center_sphere_margin is not None and trial_sphere_margin is not None
    )
    sphere_gate = not sphere_applicable or (
        sphere_available
        and sphere_lower is not None
        and sphere_lower >= GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A
    )
    gate = (
        model_topology_match
        and continuum_topology_match
        and neighbor_gate
        and point_gate
        and sphere_gate
    )
    return {
        "schema_version": GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
        "straight_segment_only": True,
        "relative_displacement_bound_A": relative_displacement_bound,
        "same_model_topology": model_topology_match,
        "same_continuum_topology": continuum_topology_match,
        "neighbor_cutoff": {
            "center_margin_A": center_model_margin,
            "trial_margin_A": trial_model_margin,
            "certified_segment_lower_bound_A": neighbor_lower,
            "guard_A": GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A,
            "gate_passed": neighbor_gate,
        },
        "point_source_shell": {
            "applicable": point_applicable,
            "margins_available": point_available,
            "center_margin_A": center_point_margin,
            "trial_margin_A": trial_point_margin,
            "certified_segment_lower_bound_A": point_lower,
            "guard_A": GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A,
            "gate_passed": point_gate,
        },
        "sphere_tangency": {
            "applicable": sphere_applicable,
            "margins_available": sphere_available,
            "center_margin_A": center_sphere_margin,
            "trial_margin_A": trial_sphere_margin,
            "certified_segment_lower_bound_A": sphere_lower,
            "guard_A": GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A,
            "gate_passed": sphere_gate,
        },
        "gate_passed": gate,
    }


def summarize_geometry_mediated_directional_audit(
    *,
    analytic_gradient_eV_per_A: object,
    direction: object,
    center_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    reciprocity_audit: Mapping[str, object],
    expected_steps_A: Sequence[float] = GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
) -> dict[str, object]:
    """Gate a preregistered same-scalar coordinate derivative within one stratum."""

    analytic = np.asarray(analytic_gradient_eV_per_A, dtype=float)
    if (
        analytic.ndim != 2
        or analytic.shape[0] < 2
        or analytic.shape[1] != 3
        or not np.all(np.isfinite(analytic))
    ):
        raise ValueError("analytic gradient must be finite with shape (N,3).")
    tangent = _array(direction, shape=analytic.shape, name="direction")
    if not np.isclose(np.linalg.norm(tangent), 1.0, rtol=0.0, atol=2.0e-14):
        raise ValueError("direction must have unit Euclidean norm.")
    if not np.allclose(np.sum(tangent, axis=0), 0.0, rtol=0.0, atol=2.0e-14):
        raise ValueError("direction must contain no rigid translation component.")
    center_model_hash, center_margin = _model_topology(center_model_topology)
    (
        center_continuum_hash,
        center_continuum_margin,
        center_sphere_tangency_margin,
        center_continuum_applicable,
        center_sphere_applicable,
    ) = _continuum_topology(center_continuum_topology)
    expected_steps = tuple(float(value) for value in expected_steps_A)
    if (
        len(expected_steps) < 3
        or any(not math.isfinite(value) or value <= 0.0 for value in expected_steps)
        or any(
            second >= first for first, second in zip(expected_steps, expected_steps[1:])
        )
    ):
        raise ValueError(
            "directional audit steps must be finite, positive, and strictly "
            "decreasing."
        )
    if tuple(float(record.get("step_A")) for record in samples) != expected_steps:
        raise ValueError("directional samples changed from the frozen step sequence.")

    analytic_directional = float(np.vdot(analytic, tangent))
    records: list[dict[str, object]] = []
    all_topologies_match = True
    all_margins_safe = center_margin is not None and (
        center_margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
    )
    any_continuum_margin_applicable = center_continuum_applicable
    all_continuum_margins_available = center_continuum_margin is not None
    _, _, all_continuum_margins_safe = _continuum_margin_status(
        center_continuum_margin,
        applicable=center_continuum_applicable,
    )
    any_sphere_margin_applicable = center_sphere_applicable
    all_sphere_margins_available = center_sphere_tangency_margin is not None
    _, _, all_sphere_margins_safe = _continuum_margin_status(
        center_sphere_tangency_margin,
        applicable=center_sphere_applicable,
    )
    for sample in samples:
        step = float(sample["step_A"])
        plus = float(sample["plus_energy_eV"])
        minus = float(sample["minus_energy_eV"])
        if not all(math.isfinite(value) for value in (step, plus, minus)):
            raise ValueError("directional energy samples must be finite.")
        finite_difference = (plus - minus) / (2.0 * step)
        absolute_error = abs(analytic_directional - finite_difference)
        relative_error = absolute_error / max(abs(finite_difference), 1.0e-15)
        relative_applies = (
            abs(finite_difference)
            >= GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_FLOOR_EV_PER_A
        )
        plus_model_hash, plus_margin = _model_topology(sample["plus_model_topology"])
        minus_model_hash, minus_margin = _model_topology(sample["minus_model_topology"])
        (
            plus_continuum_hash,
            plus_continuum_margin,
            plus_sphere_tangency_margin,
            plus_continuum_applicable,
            plus_sphere_applicable,
        ) = _continuum_topology(sample["plus_continuum_topology"])
        (
            minus_continuum_hash,
            minus_continuum_margin,
            minus_sphere_tangency_margin,
            minus_continuum_applicable,
            minus_sphere_applicable,
        ) = _continuum_topology(sample["minus_continuum_topology"])
        topology_match = (
            plus_model_hash == center_model_hash == minus_model_hash
            and plus_continuum_hash == center_continuum_hash == minus_continuum_hash
        )
        margins = (center_margin, plus_margin, minus_margin)
        margin_safe = all(
            margin is not None and margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
            for margin in margins
        )
        (
            continuum_margin_applicable,
            continuum_margin_available,
            continuum_margin_safe,
        ) = _continuum_margin_status(
            center_continuum_margin,
            plus_continuum_margin,
            minus_continuum_margin,
            applicable=(
                center_continuum_applicable
                or plus_continuum_applicable
                or minus_continuum_applicable
            ),
        )
        (
            sphere_margin_applicable,
            sphere_margin_available,
            sphere_margin_safe,
        ) = _continuum_margin_status(
            center_sphere_tangency_margin,
            plus_sphere_tangency_margin,
            minus_sphere_tangency_margin,
            applicable=(
                center_sphere_applicable
                or plus_sphere_applicable
                or minus_sphere_applicable
            ),
        )
        all_topologies_match = all_topologies_match and topology_match
        all_margins_safe = all_margins_safe and margin_safe
        any_continuum_margin_applicable = (
            any_continuum_margin_applicable or continuum_margin_applicable
        )
        all_continuum_margins_available = (
            all_continuum_margins_available and continuum_margin_available
        )
        all_continuum_margins_safe = (
            all_continuum_margins_safe and continuum_margin_safe
        )
        any_sphere_margin_applicable = (
            any_sphere_margin_applicable or sphere_margin_applicable
        )
        all_sphere_margins_available = (
            all_sphere_margins_available and sphere_margin_available
        )
        all_sphere_margins_safe = all_sphere_margins_safe and sphere_margin_safe
        numerical_gate = (
            absolute_error <= GEOMETRY_MEDIATED_DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
            and (
                not relative_applies
                or relative_error <= GEOMETRY_MEDIATED_DIRECTIONAL_RELATIVE_TOLERANCE
            )
        )
        records.append(
            {
                "step_A": step,
                "analytic_eV_per_A": analytic_directional,
                "finite_difference_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": absolute_error,
                "relative_error": relative_error,
                "relative_gate_applies": relative_applies,
                "same_model_and_cavity_stratum": topology_match,
                "neighbor_cutoff_guard_passed": margin_safe,
                "continuum_event_guard_applicable": continuum_margin_applicable,
                "continuum_event_margin_available": continuum_margin_available,
                "continuum_event_guard_passed": continuum_margin_safe,
                "sphere_tangency_guard_applicable": sphere_margin_applicable,
                "sphere_tangency_margin_available": sphere_margin_available,
                "sphere_tangency_guard_passed": sphere_margin_safe,
                "numerical_gate_passed": numerical_gate,
                "gate_passed": (
                    numerical_gate
                    and topology_match
                    and margin_safe
                    and continuum_margin_safe
                    and sphere_margin_safe
                ),
            }
        )

    errors = [float(record["absolute_error_eV_per_A"]) for record in records]
    low_error_plateau = all(
        error <= GEOMETRY_MEDIATED_NUMERICAL_PLATEAU_EV_PER_A for error in errors
    )
    refinement_observed = any(
        second <= first / 1.2 for first, second in zip(errors, errors[1:])
    )
    terminal_nondivergent = errors[-1] <= max(
        GEOMETRY_MEDIATED_NUMERICAL_PLATEAU_EV_PER_A,
        1.25 * errors[0],
    )
    convergence_gate = (low_error_plateau or refinement_observed) and (
        terminal_nondivergent
    )
    reciprocity_gate = reciprocity_audit.get("gate_passed") is True
    return {
        "schema_version": GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
        "analytic_directional_eV_per_A": analytic_directional,
        "records": records,
        "convergence": {
            "low_error_plateau": low_error_plateau,
            "refinement_observed": refinement_observed,
            "terminal_nondivergent": terminal_nondivergent,
            "gate_passed": convergence_gate,
        },
        "topology": {
            "center_model_topology_sha256": center_model_hash,
            "center_continuum_topology_sha256": center_continuum_hash,
            "all_stencils_same_stratum": all_topologies_match,
            "all_neighbor_cutoff_guards_passed": all_margins_safe,
            "continuum_event_guard_applicable": any_continuum_margin_applicable,
            "all_continuum_event_margins_available": (all_continuum_margins_available),
            "all_continuum_event_guards_passed": all_continuum_margins_safe,
            "sphere_tangency_guard_applicable": any_sphere_margin_applicable,
            "all_sphere_tangency_margins_available": (all_sphere_margins_available),
            "all_sphere_tangency_guards_passed": all_sphere_margins_safe,
        },
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "gate_passed": (
            all(bool(record["gate_passed"]) for record in records)
            and convergence_gate
            and reciprocity_gate
        ),
    }


def summarize_geometry_mediated_cartesian_audit(
    *,
    analytic_gradient_eV_per_A: object,
    center_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    reciprocity_audit: Mapping[str, object],
) -> dict[str, object]:
    """Recompute and gate a complete Cartesian same-scalar FD panel.

    The numerical convergence logic is the existing Route-2 PES primitive;
    this wrapper adds exact component coverage, raw-energy recomputation,
    model/cavity stratum equality, cutoff guards, and the optional harmonic
    point/source event margin.  It never admits a public capability.
    """

    analytic = np.asarray(analytic_gradient_eV_per_A, dtype=float)
    if (
        analytic.ndim != 2
        or analytic.shape[0] < 2
        or analytic.shape[1] != 3
        or not np.all(np.isfinite(analytic))
    ):
        raise ValueError("analytic gradient must be finite with shape (N,3).")
    center_model_hash, center_model_margin = _model_topology(center_model_topology)
    (
        center_continuum_hash,
        center_continuum_margin,
        center_sphere_tangency_margin,
        center_continuum_applicable,
        center_sphere_applicable,
    ) = _continuum_topology(center_continuum_topology)
    records = tuple(samples)
    if tuple(float(record.get("step_A")) for record in records) != (
        GEOMETRY_MEDIATED_COORDINATE_STEPS_A
    ):
        raise ValueError("Cartesian samples changed from the frozen step sequence.")

    numerical_samples: list[tuple[float, np.ndarray]] = []
    normalized_samples: list[dict[str, object]] = []
    all_topologies_match = True
    all_neighbor_guards = (
        center_model_margin is not None
        and center_model_margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
    )
    any_continuum_guard_applicable = center_continuum_applicable
    all_continuum_margins_available = center_continuum_margin is not None
    _, _, all_continuum_guards = _continuum_margin_status(
        center_continuum_margin,
        applicable=center_continuum_applicable,
    )
    any_sphere_guard_applicable = center_sphere_applicable
    all_sphere_margins_available = center_sphere_tangency_margin is not None
    _, _, all_sphere_guards = _continuum_margin_status(
        center_sphere_tangency_margin,
        applicable=center_sphere_applicable,
    )
    expected_components = analytic.size
    for raw_step in records:
        step = float(raw_step["step_A"])
        components = tuple(raw_step.get("components", ()))
        if len(components) != expected_components:
            raise ValueError("Cartesian sample has incomplete component coverage.")
        finite_difference = np.empty_like(analytic)
        seen: set[tuple[int, int]] = set()
        normalized_components: list[dict[str, object]] = []
        for raw in components:
            raw_atom = raw.get("atom")
            raw_axis = raw.get("axis")
            if (
                isinstance(raw_atom, bool)
                or not isinstance(raw_atom, (int, np.integer))
                or isinstance(raw_axis, bool)
                or not isinstance(raw_axis, (int, np.integer))
            ):
                raise TypeError("Cartesian component indices must be integers.")
            atom = int(raw_atom)
            axis = int(raw_axis)
            identity = (atom, axis)
            if (
                not 0 <= atom < analytic.shape[0]
                or not 0 <= axis < 3
                or identity in seen
            ):
                raise ValueError("Cartesian component identity is invalid.")
            seen.add(identity)
            plus_energy = float(raw.get("plus_energy_eV"))
            minus_energy = float(raw.get("minus_energy_eV"))
            if not all(math.isfinite(value) for value in (plus_energy, minus_energy)):
                raise ValueError("Cartesian displaced energies must be finite.")
            finite_difference[atom, axis] = (plus_energy - minus_energy) / (2.0 * step)

            plus_model_hash, plus_model_margin = _model_topology(
                raw["plus_model_topology"]
            )
            minus_model_hash, minus_model_margin = _model_topology(
                raw["minus_model_topology"]
            )
            (
                plus_continuum_hash,
                plus_continuum_margin,
                plus_sphere_tangency_margin,
                plus_continuum_applicable,
                plus_sphere_applicable,
            ) = _continuum_topology(raw["plus_continuum_topology"])
            (
                minus_continuum_hash,
                minus_continuum_margin,
                minus_sphere_tangency_margin,
                minus_continuum_applicable,
                minus_sphere_applicable,
            ) = _continuum_topology(raw["minus_continuum_topology"])
            topology_match = (
                plus_model_hash == center_model_hash == minus_model_hash
                and plus_continuum_hash == center_continuum_hash == minus_continuum_hash
            )
            neighbor_guard = all(
                margin is not None
                and margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
                for margin in (
                    center_model_margin,
                    plus_model_margin,
                    minus_model_margin,
                )
            )
            (
                continuum_guard_applicable,
                continuum_margin_available,
                continuum_guard,
            ) = _continuum_margin_status(
                center_continuum_margin,
                plus_continuum_margin,
                minus_continuum_margin,
                applicable=(
                    center_continuum_applicable
                    or plus_continuum_applicable
                    or minus_continuum_applicable
                ),
            )
            (
                sphere_guard_applicable,
                sphere_margin_available,
                sphere_guard,
            ) = _continuum_margin_status(
                center_sphere_tangency_margin,
                plus_sphere_tangency_margin,
                minus_sphere_tangency_margin,
                applicable=(
                    center_sphere_applicable
                    or plus_sphere_applicable
                    or minus_sphere_applicable
                ),
            )
            all_topologies_match = all_topologies_match and topology_match
            all_neighbor_guards = all_neighbor_guards and neighbor_guard
            any_continuum_guard_applicable = (
                any_continuum_guard_applicable or continuum_guard_applicable
            )
            all_continuum_margins_available = (
                all_continuum_margins_available and continuum_margin_available
            )
            all_continuum_guards = all_continuum_guards and continuum_guard
            any_sphere_guard_applicable = (
                any_sphere_guard_applicable or sphere_guard_applicable
            )
            all_sphere_margins_available = (
                all_sphere_margins_available and sphere_margin_available
            )
            all_sphere_guards = all_sphere_guards and sphere_guard
            normalized_components.append(
                {
                    "atom": atom,
                    "axis": axis,
                    "plus_energy_eV": plus_energy,
                    "minus_energy_eV": minus_energy,
                    "same_model_and_cavity_stratum": topology_match,
                    "neighbor_cutoff_guard_passed": neighbor_guard,
                    "continuum_event_guard_applicable": (continuum_guard_applicable),
                    "continuum_event_margin_available": (continuum_margin_available),
                    "continuum_event_guard_passed": continuum_guard,
                    "sphere_tangency_guard_applicable": sphere_guard_applicable,
                    "sphere_tangency_margin_available": sphere_margin_available,
                    "sphere_tangency_guard_passed": sphere_guard,
                    "plus_model_topology": raw["plus_model_topology"],
                    "minus_model_topology": raw["minus_model_topology"],
                    "plus_continuum_topology": raw["plus_continuum_topology"],
                    "minus_continuum_topology": raw["minus_continuum_topology"],
                }
            )
        if len(seen) != expected_components:
            raise ValueError("Cartesian components are duplicated or missing.")
        numerical_samples.append((step, finite_difference))
        normalized_samples.append({"step_A": step, "components": normalized_components})

    numerical = summarize_cartesian_force_differences(analytic, numerical_samples)
    reciprocity_gate = reciprocity_audit.get("gate_passed") is True
    topology = {
        "center_model_topology_sha256": center_model_hash,
        "center_continuum_topology_sha256": center_continuum_hash,
        "all_stencils_same_stratum": all_topologies_match,
        "all_neighbor_cutoff_guards_passed": all_neighbor_guards,
        "continuum_event_guard_applicable": any_continuum_guard_applicable,
        "all_continuum_event_margins_available": (all_continuum_margins_available),
        "all_continuum_event_guards_passed": all_continuum_guards,
        "sphere_tangency_guard_applicable": any_sphere_guard_applicable,
        "all_sphere_tangency_margins_available": all_sphere_margins_available,
        "all_sphere_tangency_guards_passed": all_sphere_guards,
    }
    gate = (
        numerical["all_gates_passed"] is True
        and all_topologies_match
        and all_neighbor_guards
        and all_continuum_guards
        and all_sphere_guards
        and reciprocity_gate
    )
    return {
        "schema_version": GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
        "component_count": expected_components,
        "raw_displaced_components": normalized_samples,
        "records": numerical["records"],
        "convergence": numerical["convergence"],
        "topology": topology,
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "gate_passed": gate,
    }


def summarize_geometry_mediated_rotation_audit(
    *,
    positions_A: object,
    base_energy_eV: float,
    base_forces_eV_per_A: object,
    base_source: object,
    base_model_topology: Mapping[str, object],
    base_continuum_topology: Mapping[str, object],
    rotation_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Gate rigid-frame invariance and exact discrete-stratum preservation."""

    positions = np.asarray(positions_A, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3) with at least two atoms.")
    positions = _array(positions, shape=positions.shape, name="positions")
    forces = _array(base_forces_eV_per_A, shape=positions.shape, name="base forces")
    source = _array(base_source, shape=(len(positions), 4), name="base point-l0 source")
    if not np.array_equal(source[:, 1:], np.zeros_like(source[:, 1:])):
        raise ValueError(
            "geometry-mediated rotation audit requires exact point-l0 source."
        )
    energy = float(base_energy_eV)
    if not math.isfinite(energy):
        raise ValueError("base energy must be finite.")
    base_model_hash, base_margin = _model_topology(base_model_topology)
    (
        base_continuum_hash,
        base_continuum_margin,
        base_sphere_tangency_margin,
        base_continuum_applicable,
        base_sphere_applicable,
    ) = _continuum_topology(base_continuum_topology)
    expected_rotations = geometry_mediated_rotations()
    records = tuple(rotation_records)
    if len(records) != len(expected_rotations):
        raise ValueError("rotation records changed from the frozen count.")

    measured: list[dict[str, object]] = []
    for index, (record, expected_rotation) in enumerate(
        zip(records, expected_rotations)
    ):
        rotation = _array(record.get("rotation_matrix"), shape=(3, 3), name="rotation")
        if not np.array_equal(rotation, expected_rotation):
            raise ValueError("rotation record changed from the frozen matrix.")
        rotated_energy = float(record["energy_eV"])
        rotated_forces = _array(
            record["forces_eV_per_A"], shape=forces.shape, name="rotated forces"
        )
        rotated_source = _array(
            record["source"], shape=source.shape, name="rotated source"
        )
        model_hash, margin = _model_topology(record["model_topology"])
        (
            continuum_hash,
            continuum_margin,
            sphere_tangency_margin,
            continuum_applicable,
            sphere_applicable,
        ) = _continuum_topology(record["continuum_topology"])
        energy_error = abs(rotated_energy - energy)
        force_relative = float(
            np.linalg.norm(rotated_forces - forces @ rotation.T)
            / max(float(np.linalg.norm(forces)), 1.0e-15)
        )
        source_relative = float(
            np.linalg.norm(rotated_source - source)
            / max(float(np.linalg.norm(source)), 1.0e-15)
        )
        topology_match = (
            model_hash == base_model_hash and continuum_hash == base_continuum_hash
        )
        margin_safe = (
            base_margin is not None
            and margin is not None
            and min(base_margin, margin) >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
        )
        (
            continuum_margin_applicable,
            continuum_margin_available,
            continuum_margin_safe,
        ) = _continuum_margin_status(
            base_continuum_margin,
            continuum_margin,
            applicable=base_continuum_applicable or continuum_applicable,
        )
        (
            sphere_margin_applicable,
            sphere_margin_available,
            sphere_margin_safe,
        ) = _continuum_margin_status(
            base_sphere_tangency_margin,
            sphere_tangency_margin,
            applicable=base_sphere_applicable or sphere_applicable,
        )
        numerical_gate = (
            energy_error <= GEOMETRY_MEDIATED_ROTATION_ENERGY_TOLERANCE_EV
            and force_relative <= GEOMETRY_MEDIATED_ROTATION_FORCE_RELATIVE_TOLERANCE
            and source_relative <= GEOMETRY_MEDIATED_ROTATION_SOURCE_RELATIVE_TOLERANCE
        )
        measured.append(
            {
                "index": index,
                "rotation_matrix": rotation.tolist(),
                "energy_absolute_error_eV": energy_error,
                "force_covariance_relative_error": force_relative,
                "source_covariance_relative_error": source_relative,
                "same_model_and_cavity_stratum": topology_match,
                "neighbor_cutoff_guard_passed": margin_safe,
                "continuum_event_guard_applicable": continuum_margin_applicable,
                "continuum_event_margin_available": continuum_margin_available,
                "continuum_event_guard_passed": continuum_margin_safe,
                "sphere_tangency_guard_applicable": sphere_margin_applicable,
                "sphere_tangency_margin_available": sphere_margin_available,
                "sphere_tangency_guard_passed": sphere_margin_safe,
                "numerical_gate_passed": numerical_gate,
                "gate_passed": (
                    numerical_gate
                    and topology_match
                    and margin_safe
                    and continuum_margin_safe
                    and sphere_margin_safe
                ),
            }
        )

    centered = positions - np.mean(positions, axis=0, keepdims=True)
    net_force = float(np.linalg.norm(np.sum(forces, axis=0)))
    torque = float(np.linalg.norm(np.sum(np.cross(centered, forces), axis=0)))
    rigid_body_gate = (
        net_force <= GEOMETRY_MEDIATED_NET_FORCE_TOLERANCE_EV_PER_A
        and torque <= GEOMETRY_MEDIATED_TORQUE_TOLERANCE_EV
    )
    return {
        "schema_version": GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
        "records": measured,
        "base_net_force_norm_eV_per_A": net_force,
        "base_torque_norm_eV": torque,
        "rigid_body_gate_passed": rigid_body_gate,
        "all_rotation_topologies_match": all(
            bool(record["same_model_and_cavity_stratum"]) for record in measured
        ),
        "continuum_event_guard_applicable": any(
            bool(record["continuum_event_guard_applicable"]) for record in measured
        ),
        "all_continuum_event_margins_available": all(
            bool(record["continuum_event_margin_available"]) for record in measured
        ),
        "sphere_tangency_guard_applicable": any(
            bool(record["sphere_tangency_guard_applicable"]) for record in measured
        ),
        "all_sphere_tangency_margins_available": all(
            bool(record["sphere_tangency_margin_available"]) for record in measured
        ),
        "gate_passed": (
            rigid_body_gate and all(bool(record["gate_passed"]) for record in measured)
        ),
    }


def geometry_mediated_admission_decision(
    *,
    deterministic_replay_passed: bool,
    directional_audit: Mapping[str, object],
    cartesian_audit: Mapping[str, object],
    rotation_audit: Mapping[str, object],
    post_solve_residual_available: bool,
) -> dict[str, bool]:
    """Return an explicit diagnostic/admission decision without promoting tiers."""

    local = (
        deterministic_replay_passed is True
        and directional_audit.get("gate_passed") is True
        and cartesian_audit.get("gate_passed") is True
        and rotation_audit.get("gate_passed") is True
    )
    tier_f_prerequisites = local and post_solve_residual_available is True
    return {
        "deterministic_replay_passed": deterministic_replay_passed is True,
        "directional_metric_topology_gate_passed": (
            directional_audit.get("gate_passed") is True
        ),
        "cartesian_metric_topology_gate_passed": (
            cartesian_audit.get("gate_passed") is True
        ),
        "rotation_topology_gate_passed": rotation_audit.get("gate_passed") is True,
        "post_solve_residual_available": post_solve_residual_available is True,
        "local_diagnostic_gates_passed": local,
        "tier_f_prerequisites_passed": tier_f_prerequisites,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "opt_admitted": False,
        "hessian_freq_ts_irc_admitted": False,
        "md_admitted": False,
        "tier_v_mutual_polarization_admitted": False,
    }


__all__ = [
    "GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION",
    "GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A",
    "GEOMETRY_MEDIATED_COORDINATE_STEPS_A",
    "GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A",
    "GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV",
    "GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A",
    "GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE",
    "geometry_mediated_admission_decision",
    "geometry_mediated_coordinate_direction",
    "geometry_mediated_rotations",
    "geometry_mediated_trial_step_guard",
    "summarize_geometry_mediated_reciprocity_audit",
    "summarize_geometry_mediated_cartesian_audit",
    "summarize_geometry_mediated_directional_audit",
    "summarize_geometry_mediated_rotation_audit",
]
