from __future__ import annotations

from ase import Atoms
from types import SimpleNamespace
import json
from pathlib import Path
import pytest

from maple.function.dispatcher.pure_nonmd_status import (
    WorkflowOutcome,
    WorkflowGeometry,
    add_v4_outcome_metrics,
    geometries_record,
    make_status,
    write_status,
    workflow_geometry_snapshot,
)


def _water():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.info.update(charge=0, mult=1)
    return atoms


def _geometry(value=None):
    return workflow_geometry_snapshot(geometries_record(value or _water()))


def test_fixed_step_outcome_does_not_invent_optimizer_convergence():
    outcome = WorkflowOutcome(
        workflow="md",
        method="nve",
        completed=True,
        task_success=True,
        termination_reason="requested_steps_completed",
        termination_class="completed",
        iterations=8,
        final_metrics={"completed_steps": 8},
        geometry=_geometry(),
    )
    record = outcome.to_v4_record()
    assert record["schema"] == "maple-route2-workflow-outcome-v1"
    assert record["completed"] is True
    assert record["task_success"] is True
    assert record["optimizer_converged"] is None
    assert record["scientific_certified"] is None
    assert "converged" not in record


def test_candidate_can_converge_without_becoming_task_success():
    outcome = WorkflowOutcome(
        workflow="ts",
        method="prfo",
        completed=True,
        task_success=False,
        optimizer_converged=True,
        scientific_certified=False,
        candidate_only=True,
        termination_reason="stationary_point_certificate_unavailable",
        termination_class="validation_failure",
        iterations=3,
        final_metrics={},
        geometry=_geometry(),
    )
    record = outcome.to_v4_record()
    assert record["optimizer_converged"] is True
    assert record["scientific_certified"] is False
    assert record["candidate_only"] is True
    assert record["task_success"] is False


def test_candidate_refinement_preserves_path_and_optimizer_results_separately():
    status = {
        "workflow": "ts",
        "method": "nebts",
        "executed": True,
        "termination_class": "validation_failure",
        "converged": False,
        "termination_reason": "index_one_validation_failed",
        "iterations": 4,
        "final_metrics": {"path_converged": True},
        "geometry": geometries_record(_water()),
        "trace": [],
        "refinement": {
            "refinement": {
                "workflow": "ts",
                "converged": False,
                "final_metrics": {"optimizer_converged": True},
            }
        },
    }
    capability = SimpleNamespace(
        execution_class="candidate_only",
        outcome_policy="candidate_validation",
        image_policy="multi",
    )
    record = WorkflowOutcome.from_internal_status(
        status, capability=capability
    ).to_v4_record()
    assert record["completed"] is True
    assert record["path_converged"] is True
    assert record["optimizer_converged"] is True
    assert record["scientific_certified"] is False
    assert record["task_success"] is False


def test_internal_outcome_preserves_authoritative_algorithm_geometry_snapshot():
    input_atoms = [_water(), _water()]
    algorithm_geometry = [
        {
            "symbols": ["O", "H", "H"],
            "positions_angstrom": [
                [
                    [0.0, 0.0, float(frame)],
                    [0.96, 0.0, float(frame)],
                    [-0.24, 0.93, float(frame)],
                ][index]
                for index in range(3)
            ],
        }
        for frame in (0, 1, 2)
    ]
    status = {
        "workflow": "ts",
        "method": "cistring",
        "executed": True,
        "termination_class": "converged",
        "converged": True,
        "termination_reason": "path_force_criteria_satisfied",
        "iterations": 2,
        "final_metrics": {"image_count": 3, "path_converged": True},
        "geometry": algorithm_geometry,
        "trace": [],
    }
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="path_success",
        image_policy="multi",
    )
    outcome = WorkflowOutcome.from_internal_status(status, capability=capability)
    input_atoms[0].positions[:] = 99.0
    status["geometry"][0]["positions_angstrom"][0][2] = 99.0
    record = outcome.to_v4_record()
    assert len(record["geometry"]) == record["final_metrics"]["image_count"] == 3
    assert [frame["positions_angstrom"][0][2] for frame in record["geometry"]] == [
        0.0,
        1.0,
        2.0,
    ]


@pytest.mark.parametrize(
    "geometry",
    [
        None,
        [{"symbols": ["H"], "positions_angstrom": [[0.0, 0.0]]}],
        [{"symbols": ["H"], "positions_angstrom": [[0.0, 0.0, float("nan")]]}],
        [{"symbols": ["H", "H"], "positions_angstrom": [[0.0, 0.0, 0.0]]}],
    ],
)
def test_internal_outcome_rejects_missing_or_invalid_geometry(geometry):
    status = {
        "workflow": "sp",
        "method": "sp",
        "executed": True,
        "termination_class": "converged",
        "converged": True,
        "termination_reason": "calculation_completed",
        "iterations": 0,
        "final_metrics": {},
        "trace": [],
    }
    if geometry is not None:
        status["geometry"] = geometry
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="task_success",
        image_policy="single",
    )
    with pytest.raises(ValueError, match="authoritative geometry"):
        WorkflowOutcome.from_internal_status(status, capability=capability)


def test_geometry_value_type_rejects_direct_invalid_construction():
    with pytest.raises(ValueError):
        WorkflowGeometry(symbols=("H",), positions_angstrom=((0.0, 0.0),))


def test_partial_execution_failure_is_not_completed():
    status = {
        "workflow": "md",
        "method": "nve",
        "executed": True,
        "termination_class": "execution_failure",
        "converged": False,
        "termination_reason": "integration_exception",
        "iterations": 2,
        "final_metrics": {"completed_steps": 2},
        "geometry": geometries_record(_water()),
        "trace": [],
    }
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="fixed_steps",
        image_policy="single",
    )
    record = WorkflowOutcome.from_internal_status(
        status, capability=capability
    ).to_v4_record()
    assert record["completed"] is False
    assert record["task_success"] is False


def test_fixed_steps_success_uses_completed_terminal_class():
    status = {
        "workflow": "md",
        "method": "nve",
        "executed": True,
        "termination_class": "converged",
        "converged": True,
        "termination_reason": "requested_steps_completed",
        "iterations": 8,
        "final_metrics": {"completed_steps": 8},
        "geometry": geometries_record(_water()),
        "trace": [],
    }
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="fixed_steps",
        image_policy="single",
    )
    record = WorkflowOutcome.from_internal_status(
        status, capability=capability
    ).to_v4_record()
    assert record["termination_class"] == "completed"
    assert record["task_success"] is True
    assert record["optimizer_converged"] is None
    assert record["path_converged"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed": False, "task_success": True},
        {"completed": True, "task_success": False, "rejected": True},
        {
            "completed": True,
            "task_success": False,
            "extra": {"task_success": True},
        },
    ],
)
def test_outcome_invariants_and_reserved_fields_cannot_be_overridden(kwargs):
    base = dict(
        workflow="md",
        method="nve",
        completed=False,
        task_success=False,
        termination_reason="test",
        termination_class="validation_failure",
        iterations=0,
        final_metrics={},
        geometry=_geometry(),
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        WorkflowOutcome(**base)


def test_v2_status_projection_remains_exact(monkeypatch):
    atoms = _water()
    record = make_status(
        workflow="opt",
        method="lbfgs",
        converged=False,
        termination_reason="maximum_iterations_reached",
        termination_class="bounded_nonconvergence",
        iterations=2,
        final_metrics={"max_f": 0.1},
        atoms=atoms,
    )
    assert record == {
        "schema": "maple-pure-nonmd-status-v2",
        "workflow": "opt",
        "method": "lbfgs",
        "executed": True,
        "termination_class": "bounded_nonconvergence",
        "converged": False,
        "termination_reason": "maximum_iterations_reached",
        "iterations": 2,
        "final_metrics": {"max_f": 0.1},
        "geometry": [
            {
                "symbols": ["O", "H", "H"],
                "positions_angstrom": [
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ],
            }
        ],
        "trace": [],
    }
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: False,
    )
    metrics = {"max_f": 0.1}
    assert add_v4_outcome_metrics(atoms, metrics, optimizer_converged=True) == {
        "max_f": 0.1
    }


def test_numerical_stationary_v2_allows_success_with_unknown_certificate():
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="numerical_stationary",
        image_policy="single",
    )
    status = make_status(
        workflow="ts",
        method="prfo",
        converged=True,
        termination_reason="computed_index_one_stationary_point",
        iterations=4,
        final_metrics={
            "optimizer_converged": True,
            "numerical_stationary_point_validated": True,
        },
        atoms=_water(),
    )
    outcome = WorkflowOutcome.from_internal_status(status, capability=capability)
    record = outcome.to_route2_record(capability.outcome_policy)
    assert record["schema"] == "maple-route2-workflow-outcome-v2"
    assert record["task_success"] is True
    assert record["numerical_stationary_point_validated"] is True
    assert record["certificate_status"] == "unavailable"
    assert record["scientific_certified"] is None
    assert record["candidate_only"] is False


def test_missing_numerical_stationary_validation_cannot_be_successful():
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="numerical_stationary",
        image_policy="single",
    )
    status = make_status(
        workflow="ts",
        method="dimer",
        converged=True,
        termination_reason="optimizer_converged",
        iterations=3,
        final_metrics={"optimizer_converged": True},
        atoms=_water(),
    )
    with pytest.raises(ValueError, match="numerical stationary validation"):
        WorkflowOutcome.from_internal_status(status, capability=capability)


def test_fixed_step_policy_stays_on_v1_serializer():
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="fixed_steps",
        image_policy="single",
    )
    status = make_status(
        workflow="md",
        method="nve",
        converged=True,
        termination_reason="requested_steps_completed",
        iterations=8,
        final_metrics={"completed_steps": 8},
        atoms=_water(),
    )
    outcome = WorkflowOutcome.from_internal_status(status, capability=capability)
    assert outcome.to_route2_record("fixed_steps") == outcome.to_v4_record()
    assert outcome.to_v4_record()["schema"] == "maple-route2-workflow-outcome-v1"


def test_numerical_path_reads_stationary_validation_from_refinement():
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="numerical_stationary",
        image_policy="multi",
    )
    status = make_status(
        workflow="ts",
        method="nebts",
        converged=True,
        termination_reason="computed_index_one_stationary_point",
        iterations=6,
        final_metrics={"path_converged": True},
        atoms=[_water(), _water(), _water()],
        refinement=make_status(
            workflow="ts",
            method="prfo",
            converged=True,
            termination_reason="computed_index_one_stationary_point",
            iterations=2,
            final_metrics={
                "optimizer_converged": True,
                "numerical_stationary_point_validated": True,
            },
            atoms=_water(),
        ),
    )
    record = WorkflowOutcome.from_internal_status(
        status, capability=capability
    ).to_route2_record(capability.outcome_policy)
    assert record["task_success"] is True
    assert record["path_converged"] is True
    assert record["optimizer_converged"] is True
    assert record["numerical_stationary_point_validated"] is True


def test_v2_rejects_string_booleans_at_internal_and_persistence_boundaries(tmp_path):
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="numerical_stationary",
        image_policy="multi",
    )
    status = make_status(
        workflow="ts",
        method="nebts",
        converged=True,
        termination_reason="computed_index_one_stationary_point",
        iterations=2,
        final_metrics={"path_converged": True},
        atoms=[_water(), _water(), _water()],
        refinement=make_status(
            workflow="ts",
            method="prfo",
            converged=True,
            termination_reason="computed_index_one_stationary_point",
            iterations=1,
            final_metrics={
                "optimizer_converged": True,
                "numerical_stationary_point_validated": "false",
            },
            atoms=_water(),
        ),
    )
    with pytest.raises(TypeError, match="bool or None"):
        WorkflowOutcome.from_internal_status(status, capability=capability)

    valid_status = make_status(
        workflow="ts",
        method="prfo",
        converged=True,
        termination_reason="computed_index_one_stationary_point",
        iterations=1,
        final_metrics={
            "optimizer_converged": True,
            "numerical_stationary_point_validated": True,
        },
        atoms=_water(),
    )
    record = WorkflowOutcome.from_internal_status(
        valid_status,
        capability=SimpleNamespace(
            execution_class="bounded_execution",
            outcome_policy="numerical_stationary",
            image_policy="single",
        ),
    ).to_route2_record("numerical_stationary")
    record["task_success"] = "false"
    with pytest.raises(TypeError, match="task_success"):
        write_status(tmp_path / "bad.out", record)


@pytest.mark.parametrize(
    "updates",
    [
        {"completed": None},
        {"task_success": True, "completed": False},
        {"task_success": True, "candidate_only": True},
        {"task_success": True, "rejected": True},
        {"certificate_status": "unavailable", "scientific_certified": True},
        {"certificate_status": "failed", "task_success": True},
        {"certificate_status": "stringly-typed"},
    ],
)
def test_v2_persistence_rejects_malformed_or_inconsistent_records(
    tmp_path, updates
):
    capability = SimpleNamespace(
        execution_class="bounded_execution",
        outcome_policy="numerical_stationary",
        image_policy="single",
    )
    status = make_status(
        workflow="ts", method="prfo", converged=True,
        termination_reason="computed_index_one_stationary_point",
        iterations=1,
        final_metrics={
            "optimizer_converged": True,
            "numerical_stationary_point_validated": True,
        },
        atoms=_water(),
    )
    record = WorkflowOutcome.from_internal_status(
        status, capability=capability
    ).to_route2_record(capability.outcome_policy)
    record.update(updates)
    with pytest.raises((TypeError, ValueError)):
        write_status(tmp_path / "invalid.out", record)


@pytest.mark.parametrize(
    "fixture_name",
    ["v4_md_nve_outcome_v1.json", "v4_md_nvt_langevin_outcome_v1.json"],
)
def test_fixed_step_md_v1_serialization_matches_frozen_real_receipt_bytes(
    fixture_name,
):
    path = Path(__file__).with_name("fixtures") / fixture_name
    expected_bytes = path.read_bytes()
    expected = json.loads(expected_bytes)
    core = {
        "schema", "workflow", "method", "completed", "task_success",
        "termination_class", "termination_reason", "iterations",
        "optimizer_converged", "path_converged", "scientific_certified",
        "candidate_only", "rejected", "final_metrics", "geometry", "trace",
    }
    outcome = WorkflowOutcome(
        workflow=expected["workflow"],
        method=expected["method"],
        completed=expected["completed"],
        task_success=expected["task_success"],
        termination_reason=expected["termination_reason"],
        termination_class=expected["termination_class"],
        iterations=expected["iterations"],
        final_metrics=expected["final_metrics"],
        geometry=workflow_geometry_snapshot(expected["geometry"]),
        trace=tuple(expected["trace"]),
        optimizer_converged=expected["optimizer_converged"],
        path_converged=expected["path_converged"],
        scientific_certified=expected["scientific_certified"],
        candidate_only=expected["candidate_only"],
        rejected=expected["rejected"],
        extra={key: value for key, value in expected.items() if key not in core},
    )
    actual_bytes = (
        json.dumps(outcome.to_v4_record(), indent=2, sort_keys=True) + "\n"
    ).encode()
    assert actual_bytes == expected_bytes
