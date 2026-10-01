from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "docs/implicit-solvation/benchmarks/validate_cha_continuum_branch_study.py"
)
PREPARED = (
    Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
    / ".omx/benchmarks/route1-cha-analytic-v1-20260930/water-preparation-v1"
)
PROTOCOL = PREPARED.parent / "prospective-protocol-v1.json"
TOPOLOGY_SIGNATURE = {
    "active_pairs": [[0, 1], [0, 2]],
    "fully_occluded_pairs": [[1, 2]],
}


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_cha_continuum_branch_study", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_point_force_has_the_physical_negative_gradient_sign_and_scale():
    study = _module()
    h = 5.0e-4
    slope = 7.25
    offset = -3.0
    energies = {delta: offset + slope * delta for delta in (-2 * h, -h, h, 2 * h)}

    assert study.five_point_force(energies, h) == pytest.approx(-slope, abs=1.0e-12)


def test_denominator_classification_preserves_all_nine_and_marks_sign_event():
    study = _module()
    directions = [(atom, axis) for atom in range(3) for axis in range(3)]
    center_signs = [-1, 1, 1]
    denominators = [
        {
            "atom_index": atom,
            "axis": axis,
            "samples": [
                {
                    "status": "success",
                    "weighted_signs_e": center_signs,
                    "r6_topology_signature": TOPOLOGY_SIGNATURE,
                    "cha_size_branch": "below-10A",
                },
                {
                    "status": "success",
                    "weighted_signs_e": center_signs,
                    "r6_topology_signature": TOPOLOGY_SIGNATURE,
                    "cha_size_branch": "below-10A",
                },
            ],
        }
        for atom, axis in directions
    ]
    denominators[4]["samples"][1]["weighted_signs_e"] = [-1, -1, 1]

    result = study.classify_denominators(
        center_signs,
        denominators,
        directions,
        center_topology_signature=TOPOLOGY_SIGNATURE,
        center_size_branch="below-10A",
    )

    assert result["status"] == "EVENT"
    assert result["denominator_count"] == 9
    assert result["event_denominator_count"] == 1
    assert result["failed_denominator_count"] == 0
    assert len(result["denominators"]) == 9
    assert result["denominators"][4]["classification"] == "SIGN_EVENT"
    assert result["force_gate_applicable"] is False


def test_missing_denominator_is_retained_as_typed_failure_not_dropped():
    study = _module()
    directions = [(atom, axis) for atom in range(3) for axis in range(3)]
    denominators = [
        {
            "atom_index": atom,
            "axis": axis,
            "samples": [
                {
                    "status": "success",
                    "weighted_signs_e": [-1, 1, 1],
                    "r6_topology_signature": TOPOLOGY_SIGNATURE,
                    "cha_size_branch": "below-10A",
                }
            ],
        }
        for atom, axis in directions[:-1]
    ]

    result = study.classify_denominators(
        [-1, 1, 1],
        denominators,
        directions,
        center_topology_signature=TOPOLOGY_SIGNATURE,
        center_size_branch="below-10A",
    )

    assert result["status"] == "FAILED"
    assert result["denominator_count"] == 9
    assert result["failed_denominator_count"] == 1
    assert result["denominators"][-1]["classification"] == "MISSING_DENOMINATOR"
    assert result["denominators"][-1]["error_type"] == "MissingDenominator"


def test_reference_budget_propagates_r6_to_energy_and_stays_nonrigorous():
    study = _module()

    def polar_energy(inverse_cube):
        values = np.asarray(inverse_cube, dtype=float)
        return float(np.sum(values**2))

    budget = study.reference_scalar_budget(
        inverse_cube_96=np.array([1.0, 2.0, 3.0]),
        inverse_cube_128=np.array([1.0 + 1.0e-13, 2.0, 3.0]),
        inverse_cube_quadrature_error_per_angstrom3=0.0,
        polar_energy_from_inverse_cube=polar_energy,
        cavity_volume_error_angstrom3=1.0e-13,
        dispersion_error_kcal_mol=1.0e-13,
        gate_kcal_mol=1.0e-10,
    )

    assert budget["r6_raw_inverse_cube_difference_recorded_only"] > 0.0
    assert budget["polar_energy_sensitivity_envelope_kcal_mol"] > 0.0
    assert budget["combined_estimate_kcal_mol"] < 1.0e-10
    assert budget["within_numeric_gate"] is True
    assert budget["rigorous_bound"] is False
    assert budget["admission_status"] == "UNRESOLVED_NONRIGOROUS_ESTIMATE"


def test_reference_budget_marks_unresolved_when_numeric_budget_is_too_large():
    study = _module()
    budget = study.reference_scalar_budget(
        inverse_cube_96=np.array([1.0]),
        inverse_cube_128=np.array([1.01]),
        inverse_cube_quadrature_error_per_angstrom3=0.0,
        polar_energy_from_inverse_cube=lambda values: float(np.sum(values)),
        cavity_volume_error_angstrom3=0.0,
        dispersion_error_kcal_mol=0.0,
        gate_kcal_mol=1.0e-10,
    )
    assert budget["within_numeric_gate"] is False
    assert budget["admission_status"] == "UNRESOLVED_REFERENCE_BUDGET"


def test_load_inputs_rejects_tamper_and_returns_immutable_coordinates(tmp_path):
    study = _module()
    loaded = study.load_study_inputs(PREPARED, PROTOCOL)
    assert loaded.topology.content_sha256 == loaded.topology_sha256
    assert loaded.positions_angstrom[0] == (0.011, 0.404, 0.0)
    with pytest.raises(TypeError):
        loaded.positions_angstrom[0][0] = 2.0

    copied = tmp_path / "prepared"
    copied.mkdir()
    for name in ("topology.json", "coordinates.json", "preparation-record.json"):
        (copied / name).write_bytes((PREPARED / name).read_bytes())
    artifact = json.loads((copied / "topology.json").read_text())
    artifact["cha_radii_angstrom"][0] += 0.01
    (copied / "topology.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="topology|hash|SHA"):
        study.load_study_inputs(copied, PROTOCOL)


def test_source_snapshot_detects_concurrent_mutation(tmp_path):
    study = _module()
    source = tmp_path / "source.py"
    source.write_text("first\n")
    before = study.snapshot_files([source])
    source.write_text("second\n")
    after = study.snapshot_files([source])
    assert before != after
    with pytest.raises(RuntimeError, match="changed during study"):
        study.require_unchanged_snapshot(before, after)


def test_reference_budget_includes_meridian_error_with_energy_units():
    study = _module()
    budget = study.reference_scalar_budget(
        inverse_cube_96=np.array([1.0]),
        inverse_cube_128=np.array([1.0]),
        inverse_cube_quadrature_error_per_angstrom3=1.0e-8,
        polar_energy_from_inverse_cube=lambda value: 100.0 * float(value[0]),
        cavity_volume_error_angstrom3=0.0,
        dispersion_error_kcal_mol=0.0,
        gate_kcal_mol=1.0e-10,
    )
    assert budget["polar_energy_sensitivity_envelope_kcal_mol"] == pytest.approx(
        1.0e-6, rel=1e-7
    )
    assert budget["within_numeric_gate"] is False


@pytest.mark.parametrize("error", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    [
        "inverse_cube_quadrature_error_per_angstrom3",
        "cavity_volume_error_angstrom3",
        "dispersion_error_kcal_mol",
    ],
)
def test_reference_budget_rejects_invalid_error_estimates(field, error):
    study = _module()
    arguments = dict(
        inverse_cube_96=np.array([1.0]),
        inverse_cube_128=np.array([1.0]),
        inverse_cube_quadrature_error_per_angstrom3=0.0,
        polar_energy_from_inverse_cube=lambda value: float(value[0]),
        cavity_volume_error_angstrom3=0.0,
        dispersion_error_kcal_mol=0.0,
        gate_kcal_mol=1.0e-10,
    )
    arguments[field] = error
    with pytest.raises(ValueError, match="finite|nonnegative"):
        study.reference_scalar_budget(**arguments)


def test_real_reference_dataclass_and_unit_explicit_error_are_consumed():
    study = _module()
    inputs = study.load_study_inputs(PREPARED, PROTOCOL)
    positions = np.asarray(inputs.positions_angstrom)
    _, reference = study._reference_point(positions, inputs.topology)
    budget = study._center_budget(reference, positions, inputs.topology, 1.0e-10)
    assert budget["r6_quadrature_error_estimate_per_angstrom3"] > 0.0
    assert np.isfinite(budget["combined_estimate_kcal_mol"])
    assert budget["rigorous_bound"] is False


def test_force_rms_cannot_be_diluted_by_zero_error_components():
    study = _module()
    errors = np.zeros((3, 3, 4))
    errors[:, :, 0] = 7.0e-6
    metrics = study.force_comparison_metrics(errors, max_gate=2.0e-5, rms_gate=5.0e-6)
    assert metrics["rms_kcal_mol_per_angstrom"] < 5.0e-6
    assert metrics["per_component"]["polar"]["passed"] is False
    assert metrics["passed"] is False


def test_missing_numeric_values_are_explicit_json_null_not_nonstandard_nan():
    study = _module()
    result = study._jsonable({"not_evaluated": np.array([float("nan")])})
    assert result == {"not_evaluated": [None]}
    json.dumps(result, allow_nan=False)


def test_serializer_rejects_dataclass_types_instead_of_treating_them_as_values():
    from dataclasses import dataclass

    @dataclass
    class NotAnInstance:
        number: int

    with pytest.raises(TypeError, match="type objects"):
        _module()._jsonable(NotAnInstance)


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "r6_topology_signature",
            {"active_pairs": [[0, 1]], "fully_occluded_pairs": [[0, 2], [1, 2]]},
        ),
        ("cha_size_branch", "above-10A"),
    ],
)
def test_topology_or_size_change_is_an_event_not_regular(field, value):
    study = _module()
    sample = {
        "status": "success",
        "weighted_signs_e": [-1, 1, 1],
        "r6_topology_signature": TOPOLOGY_SIGNATURE,
        "cha_size_branch": "below-10A",
    }
    sample[field] = value
    result = study.classify_denominators(
        [-1, 1, 1],
        [{"atom_index": 0, "axis": 0, "samples": [sample]}],
        [(0, 0)],
        center_topology_signature=TOPOLOGY_SIGNATURE,
        center_size_branch="below-10A",
    )
    assert result["status"] == "EVENT"
    assert result["force_gate_applicable"] is False


def test_dependency_snapshot_includes_live_geometry_and_budget_helpers():
    study = _module()
    names = {path.name for path in study.SOURCE_FILES}
    assert {"torch_sphere_union_geometry.py", "torch_dense_budget.py"} <= names


def test_structured_branch_failure_retains_reason_and_raw_margins():
    study = _module()
    from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb import (
        ChaBranchFailureReason,
        ContinuumChaBranchError,
    )

    error = ContinuumChaBranchError(
        ChaBranchFailureReason.SIZE_SWITCH, "event", (("size_margin", 0.0),)
    )
    record = study._exception_record(error)
    assert record["status"] == "branch_event"
    assert record["reason"] == ChaBranchFailureReason.SIZE_SWITCH.value
    assert record["raw_margins"] == [["size_margin", 0.0]]


def test_mixed_vector_norms_do_not_stand_in_for_scalar_uncertainties():
    study = _module()
    inputs = study.load_study_inputs(PREPARED, PROTOCOL)
    positions = np.asarray(inputs.positions_angstrom)
    _, reference = study._reference_point(positions, inputs.topology)
    reference.diagnostics["cavity_scalar_error_estimate_angstrom3"] = None
    budget = study._center_budget(reference, positions, inputs.topology, 1e-10)
    assert budget["admission_status"] == "UNRESOLVED_SCALAR_ERROR_UNAVAILABLE"
    assert budget["within_numeric_gate"] is None


def test_reported_generic_energy_budget_includes_scalar_crosscheck_discrepancy():
    study = _module()
    inputs = study.load_study_inputs(PREPARED, PROTOCOL)
    positions = np.asarray(inputs.positions_angstrom)
    _, reference = study._reference_point(positions, inputs.topology)
    reference.diagnostics["dispersion_generic_minus_scalar_reference_kcal_mol"] = -0.002
    budget = study._center_budget(reference, positions, inputs.topology, 1e-10)
    assert budget["dispersion_error_kcal_mol"] >= 0.002
    assert budget["within_numeric_gate"] is False


def test_crosscheck_discrepancy_cannot_hide_invalid_negative_scalar_error():
    study = _module()
    inputs = study.load_study_inputs(PREPARED, PROTOCOL)
    positions = np.asarray(inputs.positions_angstrom)
    _, reference = study._reference_point(positions, inputs.topology)
    reference.diagnostics["dispersion_scalar_error_estimate_kcal_mol"] = -0.001
    reference.diagnostics["dispersion_generic_minus_scalar_reference_kcal_mol"] = 0.002
    with pytest.raises(ValueError, match="nonnegative"):
        study._center_budget(reference, positions, inputs.topology, 1e-10)


def test_five_point_reference_uncertainty_has_force_units_and_all_weights():
    study = _module()
    h = 5e-4
    errors = {offset: 2e-12 for offset in (-2 * h, -h, h, 2 * h)}
    assert study.five_point_force_uncertainty(errors, h) == pytest.approx(6e-9)
    errors[h] = -1e-12
    with pytest.raises(ValueError, match="nonnegative"):
        study.five_point_force_uncertainty(errors, h)


def test_every_stencil_scalar_budget_is_checked_before_force_gate(monkeypatch):
    study = _module()
    center = {
        "status": "success",
        "weighted_signs_e": [-1, 1, 1],
        "r6_topology_signature": TOPOLOGY_SIGNATURE,
        "cha_size_branch": "below-10A",
        "analytic_forces_kcal_mol_per_angstrom": {
            name: np.zeros((3, 3)).tolist() for name in study.COMPONENTS
        },
    }
    reference = {
        "status": "success",
        "effective_charges_e": [-1, 1, 1],
        "energies_kcal_mol": {name: 0.0 for name in study.COMPONENTS},
    }
    requested_forces = []

    def production(*args, **kwargs):
        requested_forces.append(kwargs.get("need_forces", True))
        return dict(center)

    monkeypatch.setattr(study, "_production_point", production)
    monkeypatch.setattr(study, "_reference_point", lambda *args: (reference, object()))
    calls = []

    def budget(*args):
        calls.append(args)
        return {
            "within_numeric_gate": len(calls) != 1,
            "rigorous_bound": False,
            "polar_energy_sensitivity_envelope_kcal_mol": 0.0,
            "cavity_error_kcal_mol": 0.0,
            "dispersion_error_kcal_mol": 0.0,
            "combined_estimate_kcal_mol": 0.0,
        }

    monkeypatch.setattr(study, "_center_budget", budget)
    protocol = {
        "fd_steps_angstrom": [1e-3, 5e-4],
        "cartesian_directions": [(i, j) for i in range(3) for j in range(3)],
        "gates": {
            "reference_scalar_kcal_mol": 1e-10,
            "force_max_kcal_mol_per_angstrom": 2e-5,
            "force_rms_kcal_mol_per_angstrom": 5e-6,
            "two_step_disagreement_kcal_mol_per_angstrom": 1e-5,
        },
    }
    result = study._evaluate_stencil(np.zeros((3, 3)), center, None, "pin", protocol)
    assert len(calls) == 54
    assert requested_forces == [False] * 54
    assert result["status"] == "REGULAR"
    assert result["force_gate"]["scalar_reference_budgets_passed"] is False
    assert result["force_gate"]["passed"] is False


def test_scalar_only_stencil_operation_preserves_values_without_autograd(monkeypatch):
    import torch

    study = _module()
    inputs = study.load_study_inputs(PREPARED, PROTOCOL)
    positions = study.scaled_water_geometry(inputs.positions_angstrom, 0.96)
    positions[1, 2] += 0.0005
    full = study._production_point(
        positions, inputs.topology, inputs.topology_sha256, 64
    )

    def forbidden_gradient(*args, **kwargs):
        raise AssertionError("Unused stencil gradients must not be evaluated")

    monkeypatch.setattr(torch.autograd, "grad", forbidden_gradient)
    scalar = study._production_point(
        positions, inputs.topology, inputs.topology_sha256, 64, need_forces=False
    )
    assert scalar["analytic_forces_evaluated"] is False
    assert scalar["analytic_forces_kcal_mol_per_angstrom"] is None
    for name in study.COMPONENTS:
        assert scalar["energies_kcal_mol"][name] == pytest.approx(
            full["energies_kcal_mol"][name], abs=2e-12, rel=0
        )
    np.testing.assert_allclose(
        scalar["weighted_signs_e"], full["weighted_signs_e"], atol=1e-14, rtol=0
    )
    assert scalar["r6_topology_signature"] == full["r6_topology_signature"]
    assert scalar["cha_size_branch"] == full["cha_size_branch"]
