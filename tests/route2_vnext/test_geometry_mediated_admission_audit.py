from __future__ import annotations

import copy

import numpy as np

from maple.solvation.release.geometry_mediated import (
    GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
    geometry_mediated_admission_decision,
    geometry_mediated_coordinate_direction,
    geometry_mediated_rotations,
    summarize_geometry_mediated_directional_audit,
    summarize_geometry_mediated_rotation_audit,
)


def _model_topology(digest: str = "1" * 64):
    return {
        "topology_sha256": digest,
        "minimum_cutoff_margin_angstrom": 1.0,
    }


def _continuum_topology(digest: str = "2" * 64):
    return {
        "cavity_topology_sha256": digest,
        "cavity_active_node_count": 12,
    }


def _directional_audit():
    direction = geometry_mediated_coordinate_direction(3)
    gradient = 2.0 * direction
    center = -10.0
    samples = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        cubic = 0.1 * step**3
        samples.append(
            {
                "step_A": step,
                "plus_energy_eV": center + 2.0 * step + cubic,
                "minus_energy_eV": center - 2.0 * step - cubic,
                "plus_model_topology": _model_topology(),
                "minus_model_topology": _model_topology(),
                "plus_continuum_topology": _continuum_topology(),
                "minus_continuum_topology": _continuum_topology(),
            }
        )
    return (
        summarize_geometry_mediated_directional_audit(
            analytic_gradient_eV_per_A=gradient,
            direction=direction,
            center_model_topology=_model_topology(),
            center_continuum_topology=_continuum_topology(),
            samples=samples,
            reciprocity_audit={"gate_passed": True},
        ),
        samples,
        gradient,
        direction,
    )


def _rotation_audit():
    positions = np.asarray([[-0.7, 0.0, 0.0], [0.5, 0.4, 0.0], [0.2, -0.4, 0.0]])
    positions -= np.mean(positions, axis=0, keepdims=True)
    forces = -positions
    source = np.zeros((3, 4))
    source[:, 0] = (-0.4, 0.2, 0.2)
    records = []
    for rotation in geometry_mediated_rotations():
        records.append(
            {
                "rotation_matrix": rotation.tolist(),
                "energy_eV": -10.0,
                "forces_eV_per_A": (forces @ rotation.T).tolist(),
                "source": source.tolist(),
                "model_topology": _model_topology(),
                "continuum_topology": _continuum_topology(),
            }
        )
    return (
        summarize_geometry_mediated_rotation_audit(
            positions_A=positions,
            base_energy_eV=-10.0,
            base_forces_eV_per_A=forces,
            base_source=source,
            base_model_topology=_model_topology(),
            base_continuum_topology=_continuum_topology(),
            rotation_records=records,
        ),
        records,
        positions,
        forces,
        source,
    )


def test_geometry_mediated_audits_pass_only_for_one_metric_and_topology_stratum():
    directional, _, _, _ = _directional_audit()
    rotation, _, _, _, _ = _rotation_audit()

    assert directional["gate_passed"] is True
    assert directional["topology"]["all_stencils_same_stratum"] is True
    assert directional["convergence"]["gate_passed"] is True
    assert rotation["gate_passed"] is True
    assert rotation["all_rotation_topologies_match"] is True

    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=True,
        directional_audit=directional,
        rotation_audit=rotation,
        post_solve_residual_available=False,
    )
    assert decision["local_diagnostic_gates_passed"] is True
    assert decision["tier_f_prerequisites_passed"] is False
    assert decision["public_energy_admitted"] is False
    assert decision["public_force_admitted"] is False
    assert decision["opt_admitted"] is False
    assert decision["md_admitted"] is False
    assert decision["tier_v_mutual_polarization_admitted"] is False


def test_geometry_mediated_directional_audit_fails_on_cavity_event_or_reciprocity():
    _, samples, gradient, direction = _directional_audit()
    changed = copy.deepcopy(samples)
    changed[-1]["plus_continuum_topology"] = _continuum_topology("3" * 64)
    result = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=gradient,
        direction=direction,
        center_model_topology=_model_topology(),
        center_continuum_topology=_continuum_topology(),
        samples=changed,
        reciprocity_audit={"gate_passed": True},
    )
    assert result["gate_passed"] is False
    assert result["topology"]["all_stencils_same_stratum"] is False

    failed_metric = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=gradient,
        direction=direction,
        center_model_topology=_model_topology(),
        center_continuum_topology=_continuum_topology(),
        samples=samples,
        reciprocity_audit={"gate_passed": False},
    )
    assert failed_metric["gate_passed"] is False


def test_geometry_mediated_rotation_audit_fails_on_lab_grid_topology_change():
    _, records, positions, forces, source = _rotation_audit()
    changed = copy.deepcopy(records)
    changed[0]["continuum_topology"] = _continuum_topology("4" * 64)
    result = summarize_geometry_mediated_rotation_audit(
        positions_A=positions,
        base_energy_eV=-10.0,
        base_forces_eV_per_A=forces,
        base_source=source,
        base_model_topology=_model_topology(),
        base_continuum_topology=_continuum_topology(),
        rotation_records=changed,
    )
    assert result["all_rotation_topologies_match"] is False
    assert result["gate_passed"] is False
