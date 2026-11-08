from __future__ import annotations

from types import SimpleNamespace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.dispatcher.ts.algorithm.dimer import (
    Dimer,
    DimerParams,
    DimerRotationResult,
    _euclidean_rigid_basis,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.calculator.calculator_base import EV2HARTREE
from maple.solvation.numerical_precision import hessian_antisymmetry_tolerance
from maple.function.dispatcher.pure_nonmd_status import NonMDWorkflowFailure


class _HVP:
    def __init__(self, matrix, force=None):
        self.matrix = np.asarray(matrix, dtype=float)
        self.force = np.zeros(self.matrix.shape[0]) if force is None else force
        self.calls = 0

    def get_hvp(self, atoms, direction):
        self.calls += 1
        return (
            torch.as_tensor(self.matrix @ direction, dtype=torch.float64),
            torch.as_tensor(self.force, dtype=torch.float64),
            torch.tensor(-1.0, dtype=torch.float64),
        )


def _water():
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
        info={"charge": 0, "mult": 1},
    )


def _operator(atoms, eigenvalues):
    rigid = _euclidean_rigid_basis(atoms)
    u, _, _ = np.linalg.svd(rigid, full_matrices=True)
    internal = u[:, rigid.shape[1] :]
    return internal @ np.diag(eigenvalues) @ internal.T


def _job(atoms, matrix, *, rot_max_iter=5):
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.atoms.calc = _HVP(matrix)
    job.params = DimerParams(
        use_hvp=True,
        rot_max_iter=rot_max_iter,
        rot_f_max_th=1e-10,
        rot_f_rms_th=1e-10,
        remove_rigid=True,
    )
    job.M = None
    return job


def test_v4_ritz_rotation_converges_without_fixed_alpha_iteration():
    atoms = _water()
    matrix = _operator(atoms, [-2.0, 1.0, 4.0])
    job = _job(atoms, matrix)
    seed = np.arange(1.0, 10.0)

    result = job._v4_rotation(seed)

    assert result.converged is True
    assert result.curvature == pytest.approx(-2.0)
    assert result.hvp_count <= 5
    assert result.subspace_dimension == result.hvp_count
    assert result.max_residual <= job.params.rot_f_max_th
    assert isinstance(job.atoms.calc, _HVP)
    assert job.atoms.calc.calls == result.hvp_count


def test_positive_curvature_rotation_is_numerically_resolved_but_not_a_saddle():
    atoms = _water()
    job = _job(atoms, _operator(atoms, [1.0, 2.0, 3.0]))
    result = job._v4_rotation(np.arange(1.0, 10.0))
    assert result.converged is True
    assert result.curvature > 0.0


def test_remove_rigid_false_uses_full_space_without_silent_projection():
    atoms = _water()
    matrix = np.diag(np.arange(1.0, 10.0))
    job = _job(atoms, matrix)
    job.params.remove_rigid = False
    result = job._v4_rotation(np.arange(1.0, 10.0))
    assert result.rigid_rank == 0
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.h_direction, matrix @ result.direction, atol=1e-12
    )


def test_rigid_rank_is_five_for_linear_and_six_for_nonlinear_molecule():
    linear = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
    assert _euclidean_rigid_basis(linear).shape[1] == 5
    assert _euclidean_rigid_basis(_water()).shape[1] == 6


def test_source_admitted_hessian_is_not_rejected_by_a_projected_cartesian_gate():
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "ritz_projected_symmetry_counterexample.json"
        ).read_text()
    )
    atoms = Atoms(fixture["symbols"], positions=fixture["positions_angstrom"])
    matrix = np.asarray(fixture["hessian_eV_per_A2"])
    source_budget = hessian_antisymmetry_tolerance(
        torch.tensor(matrix, dtype=torch.float32)
    )
    assert np.max(np.abs(matrix - matrix.T)) <= source_budget
    basis = np.asarray(fixture["soft_basis"])
    projected = basis.T @ matrix @ basis
    wrong_budget = hessian_antisymmetry_tolerance(
        torch.tensor(projected, dtype=torch.float32)
    )
    assert np.max(np.abs(projected - projected.T)) > wrong_budget
    job = _job(atoms, matrix * EV2HARTREE)
    defaults = DimerParams()
    job.params.rot_f_max_th = defaults.rot_f_max_th
    job.params.rot_f_rms_th = defaults.rot_f_rms_th

    rotation = job._v4_rotation((basis[:, 0] + basis[:, 1]) / np.sqrt(2))

    assert rotation.converged
    assert rotation.projected_antisymmetry > 0.0
    assert rotation.max_residual <= defaults.rot_f_max_th
    assert rotation.rms_residual <= defaults.rot_f_rms_th
    record = job._rotation_record(rotation)
    assert (
        record["projected_antisymmetry_role"] == "diagnostic_only_no_transported_bound"
    )
    assert record["projected_antisymmetry_Ha_per_A2"] == rotation.projected_antisymmetry


def test_full_action_residual_not_small_symmetric_residual_owns_convergence():
    atoms = _water()
    rigid = _euclidean_rigid_basis(atoms)
    basis, _, _ = np.linalg.svd(rigid, full_matrices=True)
    internal = basis[:, rigid.shape[1] :]
    operator = np.array([[-2.0, 4.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    job = _job(atoms, internal @ operator @ internal.T)

    rotation = job._v4_rotation(internal @ np.array([1.0, 1.0, 0.0]))

    assert rotation.converged is False
    assert rotation.projected_antisymmetry > 1.0
    assert rotation.max_residual > job.params.rot_f_max_th


def test_nonfinite_projected_action_still_fails_closed():
    atoms = _water()
    matrix = np.eye(9)
    matrix[0, 0] = np.nan
    job = _job(atoms, matrix)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        job._v4_rotation(np.arange(1.0, 10.0))


def test_v4_mass_weighting_is_explicitly_pending(monkeypatch, tmp_path):
    atoms = _water()
    atoms.calc = LegacyHartreeJobView(_HVP(np.eye(9)))
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    with pytest.raises(ValueError, match="mass weighting is not defined"):
        Dimer(
            str(tmp_path / "dimer.out"),
            atoms,
            {"use_mass_weight": True},
        )


def _rotation(atoms, *, force, converged=True, reason="rotation_residual_converged"):
    dimension = 3 * len(atoms)
    return DimerRotationResult(
        direction=np.ones(dimension) / np.sqrt(dimension),
        h_direction=np.zeros(dimension),
        curvature=-1.0,
        max_residual=0.0 if converged else 1.0,
        rms_residual=0.0 if converged else 1.0,
        converged=converged,
        hvp_count=2,
        subspace_dimension=2,
        rigid_rank=6,
        projected_antisymmetry=0.0,
        force=np.asarray(force, dtype=float),
        energy=-1.0,
        positions_angstrom=np.asarray(atoms.positions, dtype=float),
        termination_reason=reason,
    )


def test_v4_cap_receipt_uses_last_evaluated_post_translation_state(
    tmp_path, monkeypatch
):
    atoms = _water()
    atoms.calc = SimpleNamespace()
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.output = str(tmp_path / "dimer.out")
    job.params = DimerParams(max_iter=1, save_traj=False)
    job.alpha = 0.2
    job.n = np.ones(9) / 3.0
    job.M = None
    first = _rotation(atoms, force=np.ones(9))
    rotations = []

    def rotate(direction):
        result = first if not rotations else _rotation(atoms, force=np.full(9, 0.5))
        rotations.append(result)
        return result

    monkeypatch.setattr(job, "_v4_rotation", rotate)

    status = job._run_v4()

    assert status["converged"] is False
    assert status["iterations"] == 1
    assert status["final_metrics"]["completed_translations"] == 1
    np.testing.assert_allclose(
        status["geometry"][0]["positions_angstrom"],
        status["final_metrics"]["rotation"]["positions_angstrom"],
    )
    np.testing.assert_allclose(atoms.positions, rotations[-1].positions_angstrom)


def test_v4_initial_force_and_rotation_success_uses_zero_translations(
    tmp_path, monkeypatch
):
    atoms = _water()
    atoms.calc = SimpleNamespace()
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.output = str(tmp_path / "dimer.out")
    job.params = DimerParams(max_iter=20, save_traj=False)
    job.alpha = 0.2
    job.n = np.ones(9) / 3.0
    job.M = None
    rotation = _rotation(atoms, force=np.zeros(9))
    monkeypatch.setattr(job, "_v4_rotation", lambda direction: rotation)
    monkeypatch.setattr(
        importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO"),
        "_pure_frozen_ts_postcondition",
        lambda atoms: (
            {
                "workflow_success": True,
                "numerical_stationary_point_validated": True,
            },
            None,
        ),
    )

    status = job._run_v4()

    assert status["converged"] is True
    assert status["iterations"] == 0
    assert status["final_metrics"]["completed_translations"] == 0
    assert status["final_metrics"]["rotation"]["converged"] is True


def test_terminal_cartesian_hessian_gate_still_blocks_task_success(
    tmp_path, monkeypatch
):
    atoms = _water()
    atoms.calc = SimpleNamespace()
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.output = str(tmp_path / "dimer.out")
    job.params = DimerParams(max_iter=20, save_traj=False)
    job.alpha = 0.2
    job.n = np.ones(9) / 3.0
    job.M = None
    rotation = _rotation(atoms, force=np.zeros(9))
    monkeypatch.setattr(job, "_v4_rotation", lambda direction: rotation)

    def reject_hessian(atoms):
        raise RuntimeError("full Cartesian Hessian source gate failed")

    monkeypatch.setattr(
        importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO"),
        "_pure_frozen_ts_postcondition",
        reject_hessian,
    )
    status = job._run_v4()

    assert status["converged"] is False
    assert status["termination_class"] == "validation_failure"
    assert status["termination_reason"] == "index_one_validation_failed"
    assert "Cartesian Hessian" in status["final_metrics"]["postcondition_error"]


def test_v4_preserves_per_atom_rms_force_and_displacement(monkeypatch, tmp_path):
    atoms = _water()
    atoms.calc = SimpleNamespace()
    initial = atoms.positions.copy()
    for name, value in (
        ("f_max_th", 0.00085),
        ("f_rms_th", 0.00055),
        ("dp_max_th", 0.00110),
        ("dp_rms_th", 0.00075),
    ):
        setattr(atoms, name, value)
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.output = str(tmp_path / "dimer.out")
    job.params = DimerParams(max_iter=1, save_traj=False)
    job.alpha = 0.2
    job.n = np.ones(9) / 3.0
    job.M = None
    initial_force = np.full(9, 0.0004)
    trial_force = np.full(9, 0.00035)
    rotations = []

    def rotate(direction):
        force = initial_force if not rotations else trial_force
        result = _rotation(atoms, force=force)
        rotations.append(result)
        return result

    monkeypatch.setattr(job, "_v4_rotation", rotate)
    monkeypatch.setattr(
        importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO"),
        "_pure_frozen_ts_postcondition",
        lambda atoms: ({"workflow_success": True}, None),
    )

    status = job._run_v4()

    assert status["final_metrics"]["rms_f"] == pytest.approx(np.sqrt(3) * 0.00035)
    actual_step = atoms.positions - initial
    expected_rms_step = np.sqrt(np.mean(np.sum(actual_step**2, axis=1)))
    assert status["final_metrics"]["rms_dp"] == pytest.approx(expected_rms_step)
    assert status["final_metrics"]["rms_dp"] != pytest.approx(
        np.sqrt(np.mean(actual_step**2))
    )
    assert status["final_metrics"]["final_attempt"]["accepted"] is True
    assert status["converged"] is False


def test_post_translation_hvp_exception_restores_and_preserves_structured_state(
    tmp_path, monkeypatch
):
    atoms = _water()
    atoms.calc = SimpleNamespace()
    job = object.__new__(Dimer)
    job.atoms = atoms
    job.output = str(tmp_path / "dimer.out")
    job.params = DimerParams(max_iter=2, save_traj=False)
    job.alpha = 0.2
    job.n = np.ones(9) / 3.0
    job.M = None
    authoritative = _rotation(atoms, force=np.ones(9))
    calls = 0

    def rotate(direction):
        nonlocal calls
        calls += 1
        if calls == 1:
            return authoritative
        raise RuntimeError("synthetic HVP failure")

    monkeypatch.setattr(job, "_v4_rotation", rotate)
    with pytest.raises(NonMDWorkflowFailure) as caught:
        job._run_v4()
    status = caught.value.status
    assert status["termination_class"] == "execution_failure"
    assert status["final_metrics"]["error_type"] == "RuntimeError"
    assert status["final_metrics"]["error"] == "synthetic HVP failure"
    assert status["final_metrics"]["completed_translations"] == 0
    assert status["final_metrics"]["rotation"]["hvp_count"] == 2
    np.testing.assert_allclose(atoms.positions, authoritative.positions_angstrom)
    np.testing.assert_allclose(
        status["geometry"][0]["positions_angstrom"],
        authoritative.positions_angstrom,
    )
