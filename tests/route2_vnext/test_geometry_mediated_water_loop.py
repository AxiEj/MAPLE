from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from _geometry_mediated_records import synthetic_reciprocity_record

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated_path import (
    AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A,
    AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS,
    aimnet2_geometry_mediated_water_loop_atoms,
    aimnet2_geometry_mediated_water_loop_coefficients,
    aimnet2_geometry_mediated_water_loop_directions,
    summarize_aimnet2_geometry_mediated_water_loop,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stationarity(*, gate: bool = True) -> dict[str, object]:
    return {
        "absolute_residual_eV_per_e": 1.0e-14 if gate else 1.0e-4,
        "relative_residual": 1.0e-15,
        "surface_condition_number": 100.0,
        "state_dimension": 12,
        "thresholds": {
            "absolute_residual_eV_per_e": 1.0e-10,
            "relative_residual": 1.0e-10,
            "surface_condition_number": 1.0e12,
        },
        "gate_passed": gate,
        "capability_admitted": False,
    }


def _model_topology(*, margin: float = 1.0, label: str = "model"):
    return {
        "topology_sha256": _sha(label),
        "minimum_cutoff_margin_angstrom": margin,
    }


def _continuum_topology(*, margin: float = 1.0, label: str = "continuum"):
    return {
        "cavity_topology_sha256": _sha(label),
        "coefficient_count": 12,
        "point_source_topology_sha256": _sha(f"{label}:point"),
        "sphere_pair_topology_sha256": _sha(f"{label}:sphere"),
        "minimum_point_source_shell_margin_angstrom": margin,
        "minimum_sphere_tangency_margin_angstrom": margin,
    }


def _point(coefficient, *, margin: float = 1.0):
    atoms = aimnet2_geometry_mediated_water_loop_atoms(coefficient)
    positions = np.asarray(atoms.positions, dtype=float)
    total = 0.5 * float(np.vdot(positions, positions))
    gradient = positions
    source = np.zeros((3, 4))
    reaction = np.zeros_like(source)
    return {
        "coefficient": list(coefficient),
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": positions.tolist(),
        "energy": {
            "vacuum_energy_eV": total,
            "continuum_energy_eV": 0.0,
            "total_energy_eV": total,
        },
        "forces_eV_per_A": (-gradient).tolist(),
        "total_gradient_eV_per_A": gradient.tolist(),
        "source": source.tolist(),
        "reaction_field": reaction.tolist(),
        "model_topology": _model_topology(margin=margin),
        "continuum_topology": _continuum_topology(margin=margin),
        "stationarity": _stationarity(),
        "reciprocity": synthetic_reciprocity_record(),
    }


def _records(*, reverse: bool = False, margin: float = 1.0):
    return [
        _point(coefficient, margin=margin)
        for coefficient in aimnet2_geometry_mediated_water_loop_coefficients(
            reverse=reverse
        )
    ]


def test_water_loop_definition_is_frozen_translation_free_and_orthonormal():
    assert AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION.endswith("-v1")
    assert AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_AMPLITUDES_A == (0.02, 0.02)
    assert AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_SUBDIVISIONS == 4
    forward = aimnet2_geometry_mediated_water_loop_coefficients()
    reverse = aimnet2_geometry_mediated_water_loop_coefficients(reverse=True)
    assert len(forward) == len(reverse) == 17
    assert forward[0] == forward[-1]
    assert reverse == tuple(reversed(forward))
    first, second = aimnet2_geometry_mediated_water_loop_directions()
    np.testing.assert_allclose(np.sum(first, axis=0), 0.0, atol=2.0e-14, rtol=0.0)
    np.testing.assert_allclose(np.sum(second, axis=0), 0.0, atol=2.0e-14, rtol=0.0)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.linalg.norm(second) == pytest.approx(1.0)
    assert float(np.vdot(first, second)) == pytest.approx(0.0, abs=2.0e-14)


def test_water_loop_recomputes_bidirectional_work_replay_and_segment_gates():
    forward = _records()
    reverse = _records(reverse=True)
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward,
        reverse_records=reverse,
    )
    assert summary["point_count_per_traversal"] == 17
    assert summary["forward"]["work"]["gate_passed"] is True
    assert summary["reverse"]["work"]["gate_passed"] is True
    assert abs(summary["forward_reverse_work_sum_eV"]) < 1.0e-14
    assert summary["gates"]["all_same_coordinate_replays"] is True
    assert summary["gates"]["all_straight_segments_certified"] is True
    assert summary["minimum_sphere_tangency_margin_A"] == pytest.approx(1.0)
    assert summary["diagnostic_gates_passed"] is True
    assert all(value is False for value in summary["capabilities"].values())
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False

    serialized_forward = json.loads(json.dumps(forward, sort_keys=True))
    serialized_reverse = json.loads(json.dumps(reverse, sort_keys=True))
    replay = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=serialized_forward,
        reverse_records=serialized_reverse,
    )
    assert replay["diagnostic_gates_passed"] is True


def test_water_loop_detects_same_coordinate_path_dependence():
    forward = _records()
    reverse = _records(reverse=True)
    reverse[4]["energy"]["vacuum_energy_eV"] += 1.0e-5
    reverse[4]["energy"]["total_energy_eV"] += 1.0e-5
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward,
        reverse_records=reverse,
    )
    assert summary["gates"]["all_same_coordinate_replays"] is False
    assert summary["maximum_same_coordinate_energy_error_eV"] == pytest.approx(1.0e-5)
    assert summary["diagnostic_gates_passed"] is False


def test_water_loop_detects_nonconservative_discrete_force_samples():
    forward = _records()
    reverse = _records(reverse=True)
    forward_index = 1
    reverse_index = len(reverse) - 1 - forward_index
    perturbation = aimnet2_geometry_mediated_water_loop_directions()[0]
    for record in (forward[forward_index], reverse[reverse_index]):
        record["forces_eV_per_A"] = (
            np.asarray(record["forces_eV_per_A"]) + perturbation
        ).tolist()
        record["total_gradient_eV_per_A"] = (
            np.asarray(record["total_gradient_eV_per_A"]) - perturbation
        ).tolist()
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward,
        reverse_records=reverse,
    )
    assert summary["gates"]["all_same_coordinate_replays"] is True
    assert summary["gates"]["forward_loop_work"] is False
    assert summary["gates"]["reverse_loop_work"] is False
    assert summary["diagnostic_gates_passed"] is False


def test_water_loop_segment_certificate_is_stricter_than_endpoint_margins():
    forward = _records(margin=0.025)
    reverse = _records(reverse=True, margin=0.025)
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward,
        reverse_records=reverse,
    )
    assert summary["minimum_neighbor_cutoff_margin_A"] == pytest.approx(0.025)
    assert summary["gates"]["all_neighbor_cutoff_margins_available"] is True
    assert summary["gates"]["all_straight_segments_certified"] is False
    assert any(
        guard["relative_displacement_bound_A"] > 0.0
        and guard["neighbor_cutoff"]["gate_passed"] is False
        for guard in summary["forward"]["segment_guards"]
    )
    assert summary["diagnostic_gates_passed"] is False


def test_water_loop_fails_closed_when_harmonic_tangency_margin_is_missing():
    forward = _records()
    reverse = _records(reverse=True)
    del forward[3]["continuum_topology"]["minimum_sphere_tangency_margin_angstrom"]
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward,
        reverse_records=reverse,
    )
    assert summary["minimum_sphere_tangency_margin_A"] is None
    assert summary["gates"]["all_sphere_tangency_margins_available"] is False
    assert summary["gates"]["all_straight_segments_certified"] is False
    assert summary["diagnostic_gates_passed"] is False


def test_water_loop_rejects_dishonest_stationarity_gate():
    forward = _records()
    reverse = _records(reverse=True)
    forward[0]["stationarity"]["absolute_residual_eV_per_e"] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_aimnet2_geometry_mediated_water_loop(
            forward_records=forward,
            reverse_records=reverse,
        )


def test_water_loop_recomputes_reciprocity_instead_of_trusting_boolean_gate():
    forward = _records()
    reverse = _records(reverse=True)
    forward[0]["reciprocity"]["charge_directional_fd_records"][0][
        "finite_difference_eV_per_e"
    ] += 1.0e-3
    with pytest.raises(ValueError, match="disagrees"):
        summarize_aimnet2_geometry_mediated_water_loop(
            forward_records=forward,
            reverse_records=reverse,
        )
