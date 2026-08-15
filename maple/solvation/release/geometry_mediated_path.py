"""Bidirectional closed-loop evidence for the explicit AIMNet2/PCM scalar.

This contract is intentionally separate from the implicit-state cold/warm loop
contracts.  The geometry-mediated scalar has no outer electronic fixed point:
each point is an independent evaluation of the same explicit
``R -> q_AIMNet2(R)`` composite scalar.  The audit recomputes both force-work
integrals, same-coordinate replay errors, stationary-continuum gates, and a
conservative no-event certificate for every straight path segment.

Passing this water-only diagnostic does not admit E/F/H/V/M, optimization,
frequency/TS/IRC, or molecular dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from ase import Atoms

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import (
    GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A,
    GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A,
    GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV,
    GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE,
    geometry_mediated_trial_step_guard,
    summarize_geometry_mediated_reciprocity_audit,
)
from .geometry_mediated_panel import (
    aimnet2_geometry_mediated_pes_molecule,
    summarize_aimnet2_geometry_mediated_stationarity,
)
from .pes_panel import panel_directions, panel_geometries
from .pes_validation import (
    closed_loop_work,
    closed_rectangular_loop,
    reverse_closed_path,
)

AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-water-loop-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-water-loop-summary-v1"
)
AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A = (0.02, 0.02)
AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS = 4
AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_COORDINATE_NAMES = (
    "seeded-internal",
    "orthogonal-radial-internal",
)
AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_ANTISYMMETRY_TOLERANCE_EV = 1.0e-10


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def aimnet2_geometry_mediated_water_loop_directions() -> tuple[np.ndarray, np.ndarray]:
    """Return the two frozen, translation-free orthonormal water directions."""

    molecule = aimnet2_geometry_mediated_pes_molecule(0)
    atoms = panel_geometries(molecule)["reference"]
    directions = panel_directions(atoms, molecule.molecule_id)
    first = np.array(directions["seeded-internal"], copy=True)
    raw_second = np.asarray(directions["radial-internal"], dtype=float)
    second = raw_second - float(np.vdot(first, raw_second)) * first
    norm = float(np.linalg.norm(second))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError("frozen water loop directions became linearly dependent.")
    second /= norm
    for name, direction in zip(
        AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_COORDINATE_NAMES,
        (first, second),
        strict=True,
    ):
        if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=2.0e-14):
            raise RuntimeError(f"{name} lost unit normalization.")
        if not np.allclose(np.sum(direction, axis=0), 0.0, rtol=0.0, atol=2.0e-14):
            raise RuntimeError(f"{name} gained a rigid translation component.")
        direction.setflags(write=False)
    if not np.isclose(float(np.vdot(first, second)), 0.0, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("frozen water loop directions lost orthogonality.")
    return first, second


def aimnet2_geometry_mediated_water_loop_coefficients(
    *, reverse: bool = False
) -> tuple[tuple[float, float], ...]:
    """Return the exact forward or reverse rectangular coefficient path."""

    forward = closed_rectangular_loop(
        subdivisions_per_edge=AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS
    )
    return reverse_closed_path(forward) if reverse else forward


def aimnet2_geometry_mediated_water_loop_atoms(
    coefficient: Sequence[float],
) -> Atoms:
    """Construct one exact water-loop geometry from its two coefficients."""

    values = np.asarray(coefficient, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("water-loop coefficient must be a finite two-vector.")
    molecule = aimnet2_geometry_mediated_pes_molecule(0)
    atoms = panel_geometries(molecule)["reference"].copy()
    first, second = aimnet2_geometry_mediated_water_loop_directions()
    atoms.positions += (
        AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A[0] * values[0] * first
        + AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A[1] * values[1] * second
    )
    return atoms


def _array(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.array(result, copy=True)


def _point_summary(
    raw: Mapping[str, object],
    *,
    expected_coefficient: tuple[float, float],
) -> dict[str, object]:
    coefficient = tuple(float(value) for value in raw.get("coefficient", ()))
    if coefficient != expected_coefficient:
        raise ValueError("water-loop coefficient changed from the frozen path.")
    expected_atoms = aimnet2_geometry_mediated_water_loop_atoms(coefficient)
    positions = _array(
        raw.get("positions_A"), shape=(len(expected_atoms), 3), name="loop positions"
    )
    numbers = np.asarray(raw.get("atomic_numbers"))
    if not np.array_equal(numbers, expected_atoms.numbers) or not np.array_equal(
        positions, expected_atoms.positions
    ):
        raise ValueError("water-loop atomic numbers or positions changed.")
    if raw.get("geometry_sha256") != geometry_sha256(expected_atoms):
        raise ValueError("water-loop geometry SHA changed from the frozen path.")

    energy = _mapping(raw.get("energy"), name="loop energy")
    vacuum = _finite_float(energy.get("vacuum_energy_eV"), name="vacuum energy")
    continuum = _finite_float(
        energy.get("continuum_energy_eV"), name="continuum energy"
    )
    total = _finite_float(energy.get("total_energy_eV"), name="total energy")
    if not math.isclose(total, vacuum + continuum, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("water-loop scalar energy ledger does not close.")

    forces = _array(
        raw.get("forces_eV_per_A"),
        shape=(len(expected_atoms), 3),
        name="loop forces",
    )
    gradient = _array(
        raw.get("total_gradient_eV_per_A"),
        shape=forces.shape,
        name="loop gradient",
    )
    if not np.array_equal(forces, -gradient):
        raise ValueError("water-loop force is not the exact negative scalar gradient.")
    source = _array(
        raw.get("source"), shape=(len(expected_atoms), 4), name="loop source"
    )
    reaction = _array(
        raw.get("reaction_field"),
        shape=source.shape,
        name="loop reaction field",
    )
    if not np.array_equal(
        source[:, 1:], np.zeros_like(source[:, 1:])
    ) or not math.isclose(
        float(np.sum(source[:, 0])), 0.0, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("water-loop source violates the neutral point-l0 contract.")
    half_coupling = 0.5 * float(np.vdot(source, reaction))
    if not math.isclose(continuum, half_coupling, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("water-loop continuum energy violates half coupling.")

    model_topology = _mapping(raw.get("model_topology"), name="model topology")
    continuum_topology = _mapping(
        raw.get("continuum_topology"), name="continuum topology"
    )
    _sha(model_topology.get("topology_sha256"), name="model topology")
    _sha(continuum_topology.get("cavity_topology_sha256"), name="continuum topology")
    _sha(
        continuum_topology.get("point_source_topology_sha256"),
        name="point-source topology",
    )
    _sha(
        continuum_topology.get("sphere_pair_topology_sha256"),
        name="sphere-pair topology",
    )
    stationarity = summarize_aimnet2_geometry_mediated_stationarity(
        _mapping(raw.get("stationarity"), name="stationarity audit")
    )
    if stationarity["state_dimension"] != 4 * len(expected_atoms):
        raise ValueError("water-loop stationarity dimension is not surface_lmax=1.")
    if continuum_topology.get("coefficient_count") != stationarity["state_dimension"]:
        raise ValueError("water-loop topology and stationarity dimensions disagree.")
    reciprocity = _mapping(raw.get("reciprocity"), name="reciprocity audit")
    reciprocity_summary = summarize_geometry_mediated_reciprocity_audit(
        reciprocity,
        reaction_field=reaction,
    )

    return {
        "coefficient": list(coefficient),
        "geometry_sha256": geometry_sha256(expected_atoms),
        "positions_A": positions.tolist(),
        "vacuum_energy_eV": vacuum,
        "continuum_energy_eV": continuum,
        "total_energy_eV": total,
        "forces_eV_per_A": forces.tolist(),
        "source": source.tolist(),
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
        "stationarity": stationarity,
        "reciprocity": reciprocity_summary,
    }


def _traversal_summary(
    records: Sequence[Mapping[str, object]],
    *,
    reverse: bool,
) -> dict[str, object]:
    coefficients = aimnet2_geometry_mediated_water_loop_coefficients(reverse=reverse)
    values = tuple(records)
    if len(values) != len(coefficients):
        raise ValueError("water-loop traversal has the wrong point count.")
    points = [
        _point_summary(raw, expected_coefficient=coefficient)
        for raw, coefficient in zip(values, coefficients, strict=True)
    ]
    work = closed_loop_work(
        [point["positions_A"] for point in points],
        [point["forces_eV_per_A"] for point in points],
        subdivisions_per_edge=AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS,
    )
    segment_guards = [
        geometry_mediated_trial_step_guard(
            center_positions_A=first["positions_A"],
            trial_positions_A=second["positions_A"],
            center_model_topology=first["model_topology"],
            trial_model_topology=second["model_topology"],
            center_continuum_topology=first["continuum_topology"],
            trial_continuum_topology=second["continuum_topology"],
        )
        for first, second in zip(points, points[1:])
    ]
    return {
        "points": points,
        "work": work,
        "segment_guards": segment_guards,
    }


def _repeat_record(
    first: Mapping[str, object], second: Mapping[str, object]
) -> dict[str, object]:
    if first["geometry_sha256"] != second["geometry_sha256"]:
        raise ValueError("same-coordinate replay paired different geometries.")
    energy_error = abs(
        float(first["total_energy_eV"]) - float(second["total_energy_eV"])
    )
    source_error = float(
        np.linalg.norm(np.asarray(first["source"]) - np.asarray(second["source"]))
    )
    force_error = float(
        np.linalg.norm(
            np.asarray(first["forces_eV_per_A"]) - np.asarray(second["forces_eV_per_A"])
        )
    )
    gate = (
        energy_error <= GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV
        and source_error <= GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE
        and force_error <= GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A
    )
    return {
        "geometry_sha256": first["geometry_sha256"],
        "energy_absolute_error_eV": energy_error,
        "source_difference_norm": source_error,
        "force_difference_norm_eV_per_A": force_error,
        "gate_passed": gate,
    }


def _optional_topology_margin(topology: Mapping[str, object], key: str) -> float | None:
    raw = topology.get(key)
    if raw is None:
        return None
    value = _finite_float(raw, name=key)
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative.")
    return value


def summarize_aimnet2_geometry_mediated_water_loop(
    *,
    forward_records: Sequence[Mapping[str, object]],
    reverse_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute the frozen water closed-loop and every fail-closed gate."""

    forward = _traversal_summary(forward_records, reverse=False)
    reverse = _traversal_summary(reverse_records, reverse=True)
    forward_points = forward["points"]
    reverse_points = reverse["points"]
    paired_replays = [
        _repeat_record(first, second)
        for first, second in zip(forward_points, reversed(reverse_points), strict=True)
    ]
    closure_replays = [
        _repeat_record(forward_points[0], forward_points[-1]),
        _repeat_record(reverse_points[0], reverse_points[-1]),
    ]
    all_points = (*forward_points, *reverse_points)
    all_segment_guards = (
        *forward["segment_guards"],
        *reverse["segment_guards"],
    )
    model_hashes = {
        str(point["model_topology"]["topology_sha256"]) for point in all_points
    }
    continuum_hashes = {
        str(point["continuum_topology"]["cavity_topology_sha256"])
        for point in all_points
    }
    neighbor_margins = [
        _optional_topology_margin(
            point["model_topology"], "minimum_cutoff_margin_angstrom"
        )
        for point in all_points
    ]
    point_margins = [
        _optional_topology_margin(
            point["continuum_topology"],
            "minimum_point_source_shell_margin_angstrom",
        )
        for point in all_points
    ]
    sphere_margins = [
        _optional_topology_margin(
            point["continuum_topology"],
            "minimum_sphere_tangency_margin_angstrom",
        )
        for point in all_points
    ]

    forward_work = float(forward["work"]["simpson_work_eV"])
    reverse_work = float(reverse["work"]["simpson_work_eV"])
    gates = {
        "forward_loop_work": forward["work"]["gate_passed"] is True,
        "reverse_loop_work": reverse["work"]["gate_passed"] is True,
        "forward_reverse_antisymmetry_le_1e-10_eV": abs(forward_work + reverse_work)
        <= AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_ANTISYMMETRY_TOLERANCE_EV,
        "all_same_coordinate_replays": all(
            record["gate_passed"] is True
            for record in (*paired_replays, *closure_replays)
        ),
        "all_stationarity_audits": all(
            point["stationarity"]["gate_passed"] is True for point in all_points
        ),
        "all_reciprocity_metric_charge_gauge_audits": all(
            point["reciprocity"]["gate_passed"] is True for point in all_points
        ),
        "fixed_model_topology": len(model_hashes) == 1,
        "fixed_continuum_topology": len(continuum_hashes) == 1,
        "all_straight_segments_certified": all(
            guard["gate_passed"] is True for guard in all_segment_guards
        ),
        "all_neighbor_cutoff_margins_available": all(
            value is not None for value in neighbor_margins
        ),
        "all_point_source_shell_margins_available": all(
            value is not None for value in point_margins
        ),
        "all_sphere_tangency_margins_available": all(
            value is not None for value in sphere_margins
        ),
    }
    stationarity_records = [point["stationarity"] for point in all_points]
    reciprocity_records = [point["reciprocity"] for point in all_points]
    return {
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SCHEMA_VERSION,
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION,
        "molecule_id": "water",
        "variant": "reference",
        "coordinate_names": list(AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_COORDINATE_NAMES),
        "coordinate_directions": [
            direction.tolist()
            for direction in aimnet2_geometry_mediated_water_loop_directions()
        ],
        "amplitudes_A": list(AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A),
        "subdivisions_per_edge": (AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS),
        "point_count_per_traversal": len(forward_points),
        "forward": forward,
        "reverse": reverse,
        "paired_forward_reverse_replays": paired_replays,
        "closure_replays": closure_replays,
        "forward_reverse_work_sum_eV": forward_work + reverse_work,
        "maximum_same_coordinate_energy_error_eV": max(
            float(record["energy_absolute_error_eV"])
            for record in (*paired_replays, *closure_replays)
        ),
        "maximum_same_coordinate_source_error": max(
            float(record["source_difference_norm"])
            for record in (*paired_replays, *closure_replays)
        ),
        "maximum_same_coordinate_force_error_eV_per_A": max(
            float(record["force_difference_norm_eV_per_A"])
            for record in (*paired_replays, *closure_replays)
        ),
        "maximum_stationarity_absolute_residual_eV_per_e": max(
            float(record["absolute_residual_eV_per_e"])
            for record in stationarity_records
        ),
        "maximum_stationarity_relative_residual": max(
            float(record["relative_residual"]) for record in stationarity_records
        ),
        "maximum_surface_condition_number": max(
            float(record["surface_condition_number"]) for record in stationarity_records
        ),
        "maximum_reciprocity_absolute_error_eV": max(
            float(record["maximum_reciprocity_absolute_error_eV"])
            for record in reciprocity_records
        ),
        "maximum_reciprocity_relative_error": max(
            float(record["maximum_reciprocity_relative_error"])
            for record in reciprocity_records
        ),
        "maximum_apply_adjoint_absolute_error_eV": max(
            float(record["maximum_apply_adjoint_absolute_error_eV"])
            for record in reciprocity_records
        ),
        "maximum_apply_adjoint_relative_error": max(
            float(record["maximum_apply_adjoint_relative_error"])
            for record in reciprocity_records
        ),
        "maximum_charge_fd_absolute_error_eV_per_e": max(
            float(record["maximum_charge_fd_absolute_error_eV_per_e"])
            for record in reciprocity_records
        ),
        "maximum_charge_fd_relative_error": max(
            float(record["maximum_charge_fd_relative_error"])
            for record in reciprocity_records
        ),
        "maximum_source_gradient_half_error_eV_per_source_unit": max(
            float(record["source_gradient_half_error_eV_per_source_unit"])
            for record in reciprocity_records
        ),
        "maximum_charge_gauge_vjp_norm_eV_per_A": max(
            float(record["charge_gauge_vjp_norm_eV_per_A"])
            for record in reciprocity_records
        ),
        "minimum_neighbor_cutoff_margin_A": (
            min(float(value) for value in neighbor_margins if value is not None)
            if all(value is not None for value in neighbor_margins)
            else None
        ),
        "minimum_point_source_shell_margin_A": (
            min(float(value) for value in point_margins if value is not None)
            if all(value is not None for value in point_margins)
            else None
        ),
        "minimum_sphere_tangency_margin_A": (
            min(float(value) for value in sphere_margins if value is not None)
            if all(value is not None for value in sphere_margins)
            else None
        ),
        "neighbor_cutoff_guard_A": GEOMETRY_MEDIATED_NEIGHBOR_CUTOFF_GUARD_A,
        "continuum_event_guard_A": GEOMETRY_MEDIATED_CONTINUUM_EVENT_GUARD_A,
        "gates": gates,
        "diagnostic_gates_passed": all(gates.values()),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }


__all__ = [
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A",
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_ANTISYMMETRY_TOLERANCE_EV",
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_COORDINATE_NAMES",
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS",
    "aimnet2_geometry_mediated_water_loop_atoms",
    "aimnet2_geometry_mediated_water_loop_coefficients",
    "aimnet2_geometry_mediated_water_loop_directions",
    "summarize_aimnet2_geometry_mediated_water_loop",
]
