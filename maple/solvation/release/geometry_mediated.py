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

GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-admission-audit-v1"
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
GEOMETRY_MEDIATED_ROTATION_ENERGY_TOLERANCE_EV = 1.0e-6
GEOMETRY_MEDIATED_ROTATION_FORCE_RELATIVE_TOLERANCE = 1.0e-4
GEOMETRY_MEDIATED_ROTATION_SOURCE_RELATIVE_TOLERANCE = 1.0e-6
GEOMETRY_MEDIATED_NET_FORCE_TOLERANCE_EV_PER_A = 1.0e-5
GEOMETRY_MEDIATED_TORQUE_TOLERANCE_EV = 1.0e-4


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
        margin = float(raw_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("model cutoff margin must be finite and non-negative.")
    return digest, margin


def _continuum_topology(record: Mapping[str, object]) -> str:
    count = record.get("cavity_active_node_count", record.get("coefficient_count"))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            "continuum topology requires a positive active-node or coefficient count."
        )
    return _sha(record.get("cavity_topology_sha256"), name="continuum topology")


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


def summarize_geometry_mediated_directional_audit(
    *,
    analytic_gradient_eV_per_A: object,
    direction: object,
    center_model_topology: Mapping[str, object],
    center_continuum_topology: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    reciprocity_audit: Mapping[str, object],
) -> dict[str, object]:
    """Gate a three-step same-scalar coordinate derivative within one stratum."""

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
    center_continuum_hash = _continuum_topology(center_continuum_topology)
    if tuple(float(record.get("step_A")) for record in samples) != (
        GEOMETRY_MEDIATED_COORDINATE_STEPS_A
    ):
        raise ValueError("directional samples changed from the frozen step sequence.")

    analytic_directional = float(np.vdot(analytic, tangent))
    records: list[dict[str, object]] = []
    all_topologies_match = True
    all_margins_safe = center_margin is not None and (
        center_margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
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
        plus_continuum_hash = _continuum_topology(sample["plus_continuum_topology"])
        minus_continuum_hash = _continuum_topology(sample["minus_continuum_topology"])
        topology_match = (
            plus_model_hash == center_model_hash == minus_model_hash
            and plus_continuum_hash == center_continuum_hash == minus_continuum_hash
        )
        margins = (center_margin, plus_margin, minus_margin)
        margin_safe = all(
            margin is not None and margin >= GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A
            for margin in margins
        )
        all_topologies_match = all_topologies_match and topology_match
        all_margins_safe = all_margins_safe and margin_safe
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
                "numerical_gate_passed": numerical_gate,
                "gate_passed": numerical_gate and topology_match and margin_safe,
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
        },
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "gate_passed": (
            all(bool(record["gate_passed"]) for record in records)
            and convergence_gate
            and reciprocity_gate
        ),
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
    base_continuum_hash = _continuum_topology(base_continuum_topology)
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
        continuum_hash = _continuum_topology(record["continuum_topology"])
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
                "numerical_gate_passed": numerical_gate,
                "gate_passed": numerical_gate and topology_match and margin_safe,
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
        "gate_passed": (
            rigid_body_gate and all(bool(record["gate_passed"]) for record in measured)
        ),
    }


def geometry_mediated_admission_decision(
    *,
    deterministic_replay_passed: bool,
    directional_audit: Mapping[str, object],
    rotation_audit: Mapping[str, object],
    post_solve_residual_available: bool,
) -> dict[str, bool]:
    """Return an explicit diagnostic/admission decision without promoting tiers."""

    local = (
        deterministic_replay_passed is True
        and directional_audit.get("gate_passed") is True
        and rotation_audit.get("gate_passed") is True
    )
    tier_f_prerequisites = local and post_solve_residual_available is True
    return {
        "deterministic_replay_passed": deterministic_replay_passed is True,
        "directional_metric_topology_gate_passed": (
            directional_audit.get("gate_passed") is True
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
    "GEOMETRY_MEDIATED_COORDINATE_STEPS_A",
    "GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A",
    "geometry_mediated_admission_decision",
    "geometry_mediated_coordinate_direction",
    "geometry_mediated_rotations",
    "summarize_geometry_mediated_directional_audit",
    "summarize_geometry_mediated_rotation_audit",
]
