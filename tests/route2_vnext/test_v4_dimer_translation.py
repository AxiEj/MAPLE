from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.dispatcher.pure_nonmd_status import NonMDWorkflowFailure
from maple.function.dispatcher.ts.algorithm.dimer import Dimer, DimerParams
from maple.function.dispatcher.ts.algorithm.dimer_translation import (
    DimerTranslationAttempt,
)


POLICY = "v4_fixed_force_scale_evaluated_translation_v1"
N = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _set_item(container, key, value):
    container[key] = value


class ExactQuadratic:
    def __init__(self, hessian, *, fail_on_call=None):
        self.hessian = np.asarray(hessian, dtype=np.float64)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def get_hvp(self, atoms, direction, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic canonical HVP failure")
        x = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        energy = 0.5 * x @ self.hessian @ x
        force = -(self.hessian @ x)
        hvp = self.hessian @ direction
        return (
            torch.as_tensor(hvp / EV2HARTREE, dtype=torch.float64),
            torch.as_tensor(force / EV2HARTREE, dtype=torch.float64),
            torch.tensor(energy / EV2HARTREE, dtype=torch.float64),
        )


class AdjustedTrialQuadratic(ExactQuadratic):
    def get_hvp(self, atoms, direction, *args, **kwargs):
        result = super().get_hvp(atoms, direction, *args, **kwargs)
        if self.calls == 2:
            positions = np.asarray(atoms.positions, dtype=np.float64).copy()
            positions.reshape(-1)[1] += 0.003
            atoms.set_positions(positions)
        return result


@pytest.fixture(autouse=True)
def _route_all_jobs_through_v4(monkeypatch):
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )


def _job(tmp_path, hessian, x, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    calculator = ExactQuadratic(hessian, fail_on_call=overrides.pop("fail_on_call", None))
    atoms = Atoms(
        "H2", positions=np.asarray(x, dtype=np.float64).reshape(2, 3),
        info={"charge": 0, "mult": 1},
    )
    atoms.calc = LegacyHartreeJobView(calculator)
    params = {
        "n_init": "given",
        "n_given": N.copy(),
        "remove_rigid": False,
        "use_hvp": True,
        "rot_max_iter": 5,
        "step0": 0.2,
        "step_max": 0.15,
        "trust_radius": 0.15,
        "max_iter": 1,
        "f_max_th": 1e-12,
        "f_rms_th": 1e-12,
        "rot_f_max_th": 1e-12,
        "rot_f_rms_th": 1e-12,
        "save_traj": False,
    }
    params.update(overrides)
    job = Dimer(str(tmp_path / "dimer.out"), atoms, params)
    return job, atoms, calculator


def _metrics(status):
    metrics = status["final_metrics"]
    assert isinstance(metrics, dict)
    assert metrics["translation_policy"] == POLICY
    assert status["iterations"] == metrics["completed_translations"]
    assert metrics["completed_translations"] == metrics["accepted_translations"]
    assert metrics["accepted_translations"] <= metrics["attempted_translations"]
    return metrics


def _attempt(metrics, beta=0.2):
    attempt = metrics["final_attempt"]
    assert isinstance(attempt, dict)
    assert attempt["ordinal"] == metrics["attempted_translations"]
    assert attempt["beta_A2_per_Ha"] == pytest.approx(beta)
    assert attempt["beta_unit"] == "A2/Ha"
    return attempt


def _assert_rotation_observables(record, hessian, positions):
    assert isinstance(record, dict)
    x = np.asarray(positions, dtype=np.float64).reshape(-1)
    matrix = np.asarray(hessian, dtype=np.float64)
    assert record["energy_hartree"] == pytest.approx(0.5 * x @ matrix @ x)
    np.testing.assert_allclose(
        record["forces_Ha_per_A"], (-(matrix @ x)).reshape(-1, 3)
    )
    assert record["curvature_Ha_per_A2"] == pytest.approx(
        np.linalg.eigvalsh(matrix)[0]
    )


@pytest.mark.parametrize("step_max", [0.10, 0.15])
def test_inactive_cap_reaches_same_two_step_geometry(tmp_path, step_max):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    job, atoms, calculator = _job(
        tmp_path, hessian, [0.01, 0, 0, 0, 0, 0],
        step_max=step_max, max_iter=2,
    )

    status = job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), [0.0064, 0, 0, 0, 0, 0])
    metrics = _metrics(status)
    assert (metrics["attempted_translations"], metrics["accepted_translations"]) == (2, 2)
    assert _attempt(metrics)["clipped"] is False
    assert calculator.calls == 3


@pytest.mark.parametrize(
    ("step_max", "expected"), [(0.10, 0.90), (0.15, 0.85)],
)
def test_active_cap_clips_raw_step_without_changing_beta(tmp_path, step_max, expected):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    job, atoms, _ = _job(
        tmp_path, hessian, [1, 0, 0, 0, 0, 0], step_max=step_max,
    )

    status = job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), [expected, 0, 0, 0, 0, 0])
    metrics = _metrics(status)
    attempt = _attempt(metrics)
    np.testing.assert_allclose(attempt["raw_step_angstrom"], [-0.2, 0, 0, 0, 0, 0])
    assert attempt["cap_angstrom"] == pytest.approx(step_max)
    assert attempt["clipped"] is True
    assert attempt["current_force_norm_Ha_per_A"] == pytest.approx(1.0)
    assert attempt["trial_force_norm_Ha_per_A"] == pytest.approx(expected)
    _assert_rotation_observables(
        attempt["current_rotation"], hessian, [1, 0, 0, 0, 0, 0]
    )
    _assert_rotation_observables(
        attempt["trial_rotation"], hessian, [expected, 0, 0, 0, 0, 0]
    )
    _assert_rotation_observables(
        metrics["rotation"], hessian, [expected, 0, 0, 0, 0, 0]
    )


def test_stable_anisotropic_trial_is_accepted_from_evaluated_force(tmp_path):
    hessian = np.diag([-1.0, 0.25, 8.0, 0.5, 1.0, 2.0])
    job, atoms, calculator = _job(tmp_path, hessian, [0.01, 0.02, 0.01, 0, 0, 0])

    status = job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), [0.008, 0.019, -0.006, 0, 0, 0])
    attempt = _attempt(_metrics(status))
    assert attempt["accepted"] is True
    assert attempt["stage"] == "evaluated"
    assert attempt["current_force_norm_Ha_per_A"] == pytest.approx(0.08077747210701756)
    assert attempt["trial_force_norm_Ha_per_A"] == pytest.approx(0.048893378897351736)
    assert calculator.calls == 2


def test_stiff_trial_is_rejected_and_authoritative_state_is_rolled_back(tmp_path):
    hessian = np.diag([-1.0, 12.0, 1.0, 1.0, 1.0, 1.0])
    initial = np.array([0.01, 0.02, 0, 0, 0, 0])
    job, atoms, calculator = _job(tmp_path, hessian, initial)

    status = job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), initial)
    assert status["termination_reason"] == "translation_force_merit_not_decreased"
    metrics = _metrics(status)
    assert (metrics["attempted_translations"], metrics["accepted_translations"]) == (1, 0)
    attempt = _attempt(metrics)
    np.testing.assert_allclose(attempt["proposed_positions_angstrom"], [0.008, -0.028, 0, 0, 0, 0])
    assert attempt["accepted"] is False
    assert attempt["current_force_norm_Ha_per_A"] == pytest.approx(0.24020824298928628)
    assert attempt["trial_force_norm_Ha_per_A"] == pytest.approx(0.3360952246016001)
    _assert_rotation_observables(attempt["current_rotation"], hessian, initial)
    _assert_rotation_observables(
        attempt["trial_rotation"], hessian, [0.008, -0.028, 0, 0, 0, 0]
    )
    _assert_rotation_observables(metrics["rotation"], hessian, initial)
    assert calculator.calls == 2


@pytest.mark.parametrize("scale_energy,scale_length", [(3.0, 2.0), (7.0, 0.5)])
@pytest.mark.parametrize(
    ("diagonal", "initial", "accepted"),
    [
        ([-1.0, 0.25, 8.0, 0.5, 1.0, 2.0], [0.01, 0.02, 0.01, 0, 0, 0], True),
        ([-1.0, 12.0, 1.0, 1.0, 1.0, 1.0], [0.01, 0.02, 0, 0, 0, 0], False),
    ],
)
def test_translation_decision_is_covariant_under_energy_and_length_units(
    tmp_path, scale_energy, scale_length, diagonal, initial, accepted,
):
    hessian = np.diag(diagonal)
    reference, _, _ = _job(
        tmp_path / "reference", hessian, initial, kappa_to_flip=-0.9
    )
    transformed, _, _ = _job(
        tmp_path / "transformed",
        scale_energy * hessian / scale_length**2,
        scale_length * np.asarray(initial),
        step0=scale_length**2 * 0.2 / scale_energy,
        step_max=scale_length * 0.15,
        trust_radius=scale_length * 0.15,
        f_max_th=scale_energy * 1e-12 / scale_length,
        f_rms_th=scale_energy * 1e-12 / scale_length,
        rot_f_max_th=scale_energy * 1e-12 / scale_length**2,
        rot_f_rms_th=scale_energy * 1e-12 / scale_length**2,
        kappa_to_flip=scale_energy * -0.9 / scale_length**2,
    )
    for job, factor in ((reference, 1.0), (transformed, scale_length)):
        setattr(job.atoms, "dp_max_th", factor * 1e-12)
        setattr(job.atoms, "dp_rms_th", factor * 1e-12)

    base_status = reference.run()
    scaled_status = transformed.run()

    base = _metrics(base_status)
    scaled = _metrics(scaled_status)
    base_attempt = _attempt(base)
    assert base_attempt["accepted"] is accepted
    scaled_beta = scale_length**2 * 0.2 / scale_energy
    scaled_attempt = _attempt(scaled, scaled_beta)
    assert scaled_attempt["accepted"] is accepted
    assert scaled_status["termination_reason"] == base_status["termination_reason"]
    assert scaled_status["iterations"] == base_status["iterations"]
    np.testing.assert_allclose(
        np.asarray(scaled_attempt["applied_step_angstrom"]) / scale_length,
        base_attempt["applied_step_angstrom"], atol=1e-14,
    )


def test_zero_translation_direction_stops_without_trial_evaluation(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial = [0.01, 0, 0, 0, 0, 0]
    job, atoms, calculator = _job(tmp_path, hessian, initial, kappa_to_flip=-2.0)

    status = job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), initial)
    assert status["termination_reason"] == "zero_translation_direction"
    attempt = _attempt(_metrics(status))
    assert attempt["stage"] == "zero_direction"
    assert attempt["trial_rotation"] is None
    assert attempt["trial_force_norm_Ha_per_A"] is None
    _assert_rotation_observables(attempt["current_rotation"], hessian, initial)
    _assert_rotation_observables(_metrics(status)["rotation"], hessian, initial)
    assert calculator.calls == 1


def test_trial_evaluation_failure_retains_prior_state_and_original_error(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial = [0.01, 0, 0, 0, 0, 0]
    job, atoms, _ = _job(tmp_path, hessian, initial, fail_on_call=2)

    with pytest.raises(NonMDWorkflowFailure) as caught:
        job.run()

    np.testing.assert_allclose(atoms.positions.reshape(-1), initial)
    status = caught.value.status
    assert status["termination_class"] == "execution_failure"
    metrics = _metrics(status)
    attempt = _attempt(metrics)
    assert attempt["stage"] == "evaluation_failed"
    assert attempt["error_type"] == "RuntimeError"
    assert attempt["error_message"] == "synthetic canonical HVP failure"
    assert attempt["trial_rotation"] is None
    _assert_rotation_observables(attempt["current_rotation"], hessian, initial)
    _assert_rotation_observables(metrics["rotation"], hessian, initial)


def test_initial_success_has_no_translation_attempt(monkeypatch, tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    job, _, calculator = _job(tmp_path, hessian, np.zeros(6))
    monkeypatch.setattr(
        importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO"),
        "_pure_frozen_ts_postcondition",
        lambda atoms: ({"workflow_success": True}, None),
    )

    status = job.run()

    assert status["converged"] is True
    metrics = _metrics(status)
    assert metrics["final_attempt"] is None
    assert metrics["attempted_translations"] == 0
    _assert_rotation_observables(metrics["rotation"], hessian, np.zeros(6))
    assert calculator.calls == 1


def test_zero_iteration_cap_has_null_attempt_and_preserves_per_atom_rms(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x = np.array([1.0, 0, 0, 0, 0.4, 0])
    job, _, calculator = _job(tmp_path, hessian, x, max_iter=0)

    status = job.run()

    metrics = _metrics(status)
    expected_force = -(hessian @ x).reshape(2, 3)
    expected_rms = np.sqrt(np.mean(np.sum(expected_force**2, axis=1)))
    assert metrics["rms_f"] == pytest.approx(expected_rms)
    assert metrics["final_attempt"] is None
    assert metrics["attempted_translations"] == 0
    assert calculator.calls == 1


def test_defaults_keep_fixed_scale_geometric_caps_and_iteration_budget():
    params = DimerParams()
    assert params.step0 == pytest.approx(0.2)
    assert params.step_max == pytest.approx(0.15)
    assert params.trust_radius == pytest.approx(0.15)
    assert params.max_iter == 200


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("step0", 0.0), ("step0", np.inf),
        ("step_max", -0.1), ("step_max", np.nan),
        ("trust_radius", 0.0), ("trust_radius", np.inf),
        ("max_iter", -1), ("max_iter", 1.5), ("max_iter", True),
    ],
)
def test_invalid_translation_control_is_rejected_before_physical_evaluation(
    tmp_path, name, value,
):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    calculator = ExactQuadratic(hessian)
    atoms = Atoms("H2", positions=np.zeros((2, 3)), info={"charge": 0, "mult": 1})
    atoms.calc = LegacyHartreeJobView(calculator)
    params = {"n_init": "given", "n_given": N, "remove_rigid": False, name: value}

    with pytest.raises((TypeError, ValueError)):
        Dimer(str(tmp_path / "invalid.out"), atoms, params)

    assert calculator.calls == 0


def test_trial_metrics_use_evaluated_geometry_displacement(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial = np.array([0.01, 0, 0, 0, 0, 0], dtype=np.float64)
    calculator = AdjustedTrialQuadratic(hessian)
    atoms = Atoms(
        "H2", positions=initial.reshape(2, 3), info={"charge": 0, "mult": 1}
    )
    atoms.calc = LegacyHartreeJobView(calculator)
    job = Dimer(
        str(tmp_path / "adjusted.out"),
        atoms,
        {
            "n_init": "given",
            "n_given": N.copy(),
            "remove_rigid": False,
            "rot_max_iter": 5,
            "max_iter": 1,
            "f_max_th": 1e-12,
            "f_rms_th": 1e-12,
            "rot_f_max_th": 1e-12,
            "rot_f_rms_th": 1e-12,
            "save_traj": False,
        },
    )

    status = job.run()

    actual_displacement = atoms.positions.reshape(-1) - initial
    expected_rms = np.sqrt(
        np.mean(np.sum(actual_displacement.reshape(-1, 3) ** 2, axis=1))
    )
    metrics = _metrics(status)
    attempt = _attempt(metrics)
    assert metrics["rms_dp"] == pytest.approx(expected_rms)
    np.testing.assert_allclose(
        attempt["evaluated_displacement_angstrom"], actual_displacement
    )
    np.testing.assert_allclose(
        attempt["proposed_positions_angstrom"], [0.008, 0, 0, 0, 0, 0]
    )
    assert metrics["rms_dp"] != pytest.approx(
        np.sqrt(np.mean(np.asarray(attempt["applied_step_angstrom"]) ** 2))
    )


def test_zero_budget_reason_is_independent_of_rotation_convergence(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    hessian[0, 1] = hessian[1, 0] = 0.2
    job, _, calculator = _job(
        tmp_path,
        hessian,
        [0.01, 0.02, 0, 0, 0, 0],
        max_iter=0,
        rot_max_iter=1,
    )

    status = job.run()

    assert status["termination_reason"] == "maximum_iterations_reached"
    metrics = _metrics(status)
    assert metrics["rotation"]["converged"] is False
    assert metrics["attempted_translations"] == 0
    assert calculator.calls == 1


def test_initial_rotation_failure_is_structured_without_fake_state(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial = np.array([0.01, 0, 0, 0, 0, 0], dtype=np.float64)
    job, atoms, calculator = _job(
        tmp_path, hessian, initial, fail_on_call=1
    )

    with pytest.raises(NonMDWorkflowFailure) as caught:
        job.run()

    status = caught.value.status
    metrics = _metrics(status)
    assert status["termination_class"] == "execution_failure"
    assert status["termination_reason"] == "initial_rotation_evaluation_failed"
    assert status["iterations"] == 0
    assert metrics["attempted_translations"] == 0
    assert metrics["accepted_translations"] == 0
    assert metrics["completed_translations"] == 0
    assert metrics["final_attempt"] is None
    assert "rotation" not in metrics
    assert "max_f" not in metrics
    assert metrics["error_type"] == "RuntimeError"
    assert metrics["error"] == "synthetic canonical HVP failure"
    np.testing.assert_allclose(atoms.positions.reshape(-1), initial)
    assert calculator.calls == 1
    json.dumps(status, allow_nan=False)


def test_accepted_direction_survives_following_hvp_failure(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    hessian[0, 1] = hessian[1, 0] = 0.2
    initial = np.array([0.01, 0.02, 0, 0, 0, 0], dtype=np.float64)
    job, atoms, calculator = _job(
        tmp_path,
        hessian,
        initial,
        max_iter=2,
        fail_on_call=4,
    )

    with pytest.raises(NonMDWorkflowFailure) as caught:
        job.run()

    status = caught.value.status
    metrics = _metrics(status)
    assert status["iterations"] == 1
    assert metrics["attempted_translations"] == 2
    assert metrics["accepted_translations"] == 1
    assert metrics["completed_translations"] == 1
    assert metrics["final_attempt"]["stage"] == "evaluation_failed"
    accepted_direction = np.asarray(metrics["rotation"]["direction"])
    np.testing.assert_allclose(job.n, accepted_direction)
    assert not np.allclose(job.n, N)
    np.testing.assert_allclose(
        atoms.positions,
        np.asarray(status["geometry"][0]["positions_angstrom"]),
    )
    assert calculator.calls == 4


def test_attempt_snapshots_are_deeply_immutable_and_records_are_independent():
    rotation = {
        "direction": [1.0, 0.0],
        "positions_angstrom": [[0.1, 0.2, 0.3]],
        "energy_hartree": 0.5,
        "forces_Ha_per_A": [[-0.1, 0.0, 0.0]],
        "curvature_Ha_per_A2": -1.0,
    }
    attempt = DimerTranslationAttempt(
        ordinal=1,
        stage="evaluated",
        current_rotation=rotation,
        current_positions_angstrom=np.array([0.1, 0.2, 0.3]),
        proposed_positions_angstrom=np.array([0.08, 0.2, 0.3]),
        translation_force_Ha_per_A=np.array([-0.1, 0.0, 0.0]),
        beta_A2_per_Ha=0.2,
        raw_step_angstrom=np.array([-0.02, 0.0, 0.0]),
        applied_step_angstrom=np.array([-0.02, 0.0, 0.0]),
        evaluated_displacement_angstrom=np.array([-0.02, 0.0, 0.0]),
        cap_angstrom=0.15,
        clipped=False,
        current_force_norm_Ha_per_A=0.1,
        trial_force_norm_Ha_per_A=0.08,
        trial_rotation=rotation,
        accepted=True,
        reason="full_force_norm_strictly_decreased",
    )

    with pytest.raises(TypeError):
        _set_item(attempt.current_rotation, "direction", (0.0, 1.0))
    positions = attempt.current_rotation["positions_angstrom"]
    with pytest.raises(TypeError):
        _set_item(positions, 0, (9.0, 9.0, 9.0))
    forces = attempt.current_rotation["forces_Ha_per_A"]
    with pytest.raises(TypeError):
        _set_item(forces, 0, (9.0, 9.0, 9.0))
    record = attempt.to_record()
    current_record = record["current_rotation"]
    assert isinstance(current_record, dict)
    record_positions = current_record["positions_angstrom"]
    assert isinstance(record_positions, list)
    record_positions[0][0] = 9.0
    assert attempt.to_record()["current_rotation"]["positions_angstrom"][0][0] == 0.1


def test_huge_finite_beta_overflow_is_typed_and_json_strict(tmp_path):
    hessian = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    initial = np.array([2.0, 0, 0, 0, 0, 0], dtype=np.float64)
    job, atoms, calculator = _job(
        tmp_path, hessian, initial, step0=np.finfo(np.float64).max
    )

    with pytest.raises(NonMDWorkflowFailure) as caught:
        job.run()

    status = caught.value.status
    metrics = _metrics(status)
    attempt = _attempt(metrics, np.finfo(np.float64).max)
    assert status["termination_class"] == "execution_failure"
    assert status["termination_reason"] == "nonfinite_translation_proposal"
    assert status["iterations"] == 0
    assert metrics["attempted_translations"] == 1
    assert metrics["accepted_translations"] == 0
    assert metrics["completed_translations"] == 0
    assert attempt["stage"] == "nonfinite_proposal"
    assert attempt["proposed_positions_angstrom"] is None
    assert attempt["raw_step_angstrom"] is None
    assert attempt["applied_step_angstrom"] is None
    assert attempt["evaluated_displacement_angstrom"] is None
    np.testing.assert_allclose(atoms.positions.reshape(-1), initial)
    assert calculator.calls == 1
    json.dumps(status, allow_nan=False)
