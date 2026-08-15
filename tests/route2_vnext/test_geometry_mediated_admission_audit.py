from __future__ import annotations

import copy

import numpy as np

from maple.solvation.release.geometry_mediated import (
    GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
    geometry_mediated_admission_decision,
    geometry_mediated_coordinate_direction,
    geometry_mediated_rotations,
    geometry_mediated_trial_step_guard,
    summarize_geometry_mediated_cartesian_audit,
    summarize_geometry_mediated_directional_audit,
    summarize_geometry_mediated_rotation_audit,
)


def _model_topology(digest: str = "1" * 64):
    return {
        "topology_sha256": digest,
        "minimum_cutoff_margin_angstrom": 1.0,
    }


def _continuum_topology(
    digest: str = "2" * 64,
    *,
    margin: float | None = None,
    sphere_margin: float | None = None,
    harmonic: bool = False,
):
    result = {
        "cavity_topology_sha256": digest,
        "cavity_active_node_count": 12,
    }
    if margin is not None:
        result["minimum_point_source_shell_margin_angstrom"] = margin
    if sphere_margin is not None:
        result["minimum_sphere_tangency_margin_angstrom"] = sphere_margin
    if harmonic:
        result["point_source_topology_sha256"] = "3" * 64
        result["sphere_pair_topology_sha256"] = "4" * 64
    return result


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


def _cartesian_audit():
    gradient = np.arange(1.0, 10.0).reshape(3, 3) / 10.0
    center = -10.0
    samples = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        components = []
        for atom in range(3):
            for axis in range(3):
                derivative = gradient[atom, axis]
                cubic = 0.1 * step**3
                components.append(
                    {
                        "atom": atom,
                        "axis": axis,
                        "plus_energy_eV": center + derivative * step + cubic,
                        "minus_energy_eV": center - derivative * step - cubic,
                        "plus_model_topology": _model_topology(),
                        "minus_model_topology": _model_topology(),
                        "plus_continuum_topology": _continuum_topology(),
                        "minus_continuum_topology": _continuum_topology(),
                    }
                )
        samples.append({"step_A": step, "components": components})
    return (
        summarize_geometry_mediated_cartesian_audit(
            analytic_gradient_eV_per_A=gradient,
            center_model_topology=_model_topology(),
            center_continuum_topology=_continuum_topology(),
            samples=samples,
            reciprocity_audit={"gate_passed": True},
        ),
        samples,
        gradient,
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
    cartesian, _, _ = _cartesian_audit()
    rotation, _, _, _, _ = _rotation_audit()

    assert directional["gate_passed"] is True
    assert directional["topology"]["all_stencils_same_stratum"] is True
    assert directional["topology"]["continuum_event_guard_applicable"] is False
    assert directional["topology"]["all_continuum_event_margins_available"] is False
    assert directional["topology"]["all_continuum_event_guards_passed"] is True
    assert directional["convergence"]["gate_passed"] is True
    assert cartesian["gate_passed"] is True
    assert cartesian["component_count"] == 9
    assert all(cartesian["convergence"]["gates"].values())
    assert cartesian["topology"]["continuum_event_guard_applicable"] is False
    assert rotation["gate_passed"] is True
    assert rotation["all_rotation_topologies_match"] is True

    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=True,
        directional_audit=directional,
        cartesian_audit=cartesian,
        rotation_audit=rotation,
        post_solve_residual_available=False,
    )
    assert decision["local_diagnostic_gates_passed"] is True
    assert decision["tier_f_prerequisites_passed"] is False
    assert decision["cartesian_metric_topology_gate_passed"] is True
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


def test_geometry_mediated_directional_audit_requires_declared_harmonic_margins():
    _, samples, gradient, direction = _directional_audit()
    harmonic = _continuum_topology(
        margin=0.5,
        sphere_margin=0.5,
        harmonic=True,
    )
    changed = copy.deepcopy(samples)
    for sample in changed:
        sample["plus_continuum_topology"] = copy.deepcopy(harmonic)
        sample["minus_continuum_topology"] = copy.deepcopy(harmonic)
    del changed[-1]["minus_continuum_topology"][
        "minimum_sphere_tangency_margin_angstrom"
    ]
    result = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=gradient,
        direction=direction,
        center_model_topology=_model_topology(),
        center_continuum_topology=harmonic,
        samples=changed,
        reciprocity_audit={"gate_passed": True},
    )
    assert result["topology"]["sphere_tangency_guard_applicable"] is True
    assert result["topology"]["all_sphere_tangency_margins_available"] is False
    assert result["topology"]["all_sphere_tangency_guards_passed"] is False
    assert result["gate_passed"] is False


def test_geometry_mediated_directional_audit_accepts_only_explicit_step_contract():
    _, _, gradient, direction = _directional_audit()
    steps = (4.0e-4, 2.0e-4, 1.0e-4)
    samples = []
    for step in steps:
        cubic = 0.1 * step**3
        samples.append(
            {
                "step_A": step,
                "plus_energy_eV": -10.0 + 2.0 * step + cubic,
                "minus_energy_eV": -10.0 - 2.0 * step - cubic,
                "plus_model_topology": _model_topology(),
                "minus_model_topology": _model_topology(),
                "plus_continuum_topology": _continuum_topology(),
                "minus_continuum_topology": _continuum_topology(),
            }
        )

    result = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=gradient,
        direction=direction,
        center_model_topology=_model_topology(),
        center_continuum_topology=_continuum_topology(),
        samples=samples,
        reciprocity_audit={"gate_passed": True},
        expected_steps_A=steps,
    )
    assert result["gate_passed"] is True
    assert tuple(record["step_A"] for record in result["records"]) == steps

    with np.testing.assert_raises_regex(ValueError, "strictly decreasing"):
        summarize_geometry_mediated_directional_audit(
            analytic_gradient_eV_per_A=gradient,
            direction=direction,
            center_model_topology=_model_topology(),
            center_continuum_topology=_continuum_topology(),
            samples=samples,
            reciprocity_audit={"gate_passed": True},
            expected_steps_A=(1.0e-4, 2.0e-4, 4.0e-4),
        )


def test_geometry_mediated_cartesian_audit_fails_closed_on_coverage_and_events():
    _, samples, gradient = _cartesian_audit()
    incomplete = copy.deepcopy(samples)
    incomplete[0]["components"].pop()
    with np.testing.assert_raises_regex(ValueError, "incomplete component coverage"):
        summarize_geometry_mediated_cartesian_audit(
            analytic_gradient_eV_per_A=gradient,
            center_model_topology=_model_topology(),
            center_continuum_topology=_continuum_topology(),
            samples=incomplete,
            reciprocity_audit={"gate_passed": True},
        )

    event = copy.deepcopy(samples)
    for step in event:
        for component in step["components"]:
            component["plus_continuum_topology"] = _continuum_topology(margin=0.01)
            component["minus_continuum_topology"] = _continuum_topology(margin=0.01)
    result = summarize_geometry_mediated_cartesian_audit(
        analytic_gradient_eV_per_A=gradient,
        center_model_topology=_model_topology(),
        center_continuum_topology=_continuum_topology(margin=0.01),
        samples=event,
        reciprocity_audit={"gate_passed": True},
    )
    assert result["topology"]["all_continuum_event_guards_passed"] is False
    assert result["topology"]["continuum_event_guard_applicable"] is True
    assert result["topology"]["all_continuum_event_margins_available"] is True
    assert result["gate_passed"] is False


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


def test_geometry_mediated_trial_step_guard_certifies_only_the_full_segment():
    center = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    trial = center.copy()
    trial[1, 0] += 0.01
    harmonic = _continuum_topology(
        margin=0.4,
        sphere_margin=0.3,
        harmonic=True,
    )
    result = geometry_mediated_trial_step_guard(
        center_positions_A=center,
        trial_positions_A=trial,
        center_model_topology=_model_topology(),
        trial_model_topology=_model_topology(),
        center_continuum_topology=harmonic,
        trial_continuum_topology=harmonic,
    )
    np.testing.assert_allclose(
        result["relative_displacement_bound_A"], 0.01, atol=1.0e-15, rtol=0.0
    )
    assert result["neighbor_cutoff"]["gate_passed"] is True
    assert result["point_source_shell"]["gate_passed"] is True
    assert result["sphere_tangency"]["gate_passed"] is True
    assert result["gate_passed"] is True

    translated = geometry_mediated_trial_step_guard(
        center_positions_A=center,
        trial_positions_A=center + np.asarray([2.0, -1.0, 0.5]),
        center_model_topology=_model_topology(),
        trial_model_topology=_model_topology(),
        center_continuum_topology=harmonic,
        trial_continuum_topology=harmonic,
    )
    assert translated["relative_displacement_bound_A"] < 1.0e-15
    assert translated["gate_passed"] is True


def test_geometry_mediated_trial_step_guard_fails_on_missing_margin_or_long_step():
    center = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    harmonic = _continuum_topology(
        margin=0.1,
        sphere_margin=0.1,
        harmonic=True,
    )
    missing_sphere = copy.deepcopy(harmonic)
    del missing_sphere["minimum_sphere_tangency_margin_angstrom"]
    missing = geometry_mediated_trial_step_guard(
        center_positions_A=center,
        trial_positions_A=center,
        center_model_topology=_model_topology(),
        trial_model_topology=_model_topology(),
        center_continuum_topology=harmonic,
        trial_continuum_topology=missing_sphere,
    )
    assert missing["sphere_tangency"]["applicable"] is True
    assert missing["sphere_tangency"]["margins_available"] is False
    assert missing["gate_passed"] is False

    trial = center.copy()
    trial[1, 0] += 0.09
    long_step = geometry_mediated_trial_step_guard(
        center_positions_A=center,
        trial_positions_A=trial,
        center_model_topology=_model_topology(),
        trial_model_topology=_model_topology(),
        center_continuum_topology=harmonic,
        trial_continuum_topology=harmonic,
    )
    np.testing.assert_allclose(
        long_step["point_source_shell"]["certified_segment_lower_bound_A"],
        0.01,
        atol=1.0e-15,
        rtol=0.0,
    )
    assert long_step["gate_passed"] is False

    changed = geometry_mediated_trial_step_guard(
        center_positions_A=center,
        trial_positions_A=center,
        center_model_topology=_model_topology(),
        trial_model_topology=_model_topology("9" * 64),
        center_continuum_topology=harmonic,
        trial_continuum_topology=harmonic,
    )
    assert changed["same_model_topology"] is False
    assert changed["gate_passed"] is False
