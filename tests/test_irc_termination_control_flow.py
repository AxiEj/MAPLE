from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.irc.algorithm.eulerpc import EulerPC
from maple.function.dispatcher.irc.algorithm.gs import GS
from maple.function.dispatcher.irc.algorithm.hpc import HPC
from maple.function.dispatcher.irc.algorithm.lqa import LQA
from maple.function.dispatcher.irc.parameters import (
    EulerPCParams,
    GSParams,
    HPCParams,
    LQAParams,
)


_ALGORITHMS = (
    (GS, GSParams),
    (LQA, LQAParams),
    (HPC, HPCParams),
    (EulerPC, EulerPCParams),
)


def _controlled_integrator(algorithm, params, tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    integrator = algorithm(atoms, str(tmp_path / "irc.out"), params=params)
    integrator._D = np.ones(3)
    integrator._step_len_mw = 0.1
    if hasattr(integrator, "_step_len_umw"):
        integrator._step_len_umw = 0.1

    integrator._print_header = lambda *_: None
    integrator._print_conv_thresholds = lambda *_: None
    integrator._print_iter_line = lambda *_: None
    integrator._print_hurray = lambda: None
    integrator._get_hessian_cart = lambda: np.zeros((3, 3))
    integrator._bofill_update = lambda hessian, *_: hessian
    integrator._bfgs_update = lambda hessian, *_: hessian

    def energy_forces(q_mw):
        coordinates = np.asarray(q_mw, dtype=float).reshape(1, 3)
        integrator.atoms.set_positions(coordinates)
        return -1.0, np.full(3, 0.1)

    integrator._energy_forces_from_mw = energy_forces
    integrator._gradient_mw_from_forces = lambda forces: -np.asarray(forces)
    return integrator


def _nonzero_micro_step(integrator):
    displacement = np.array([1.0e-3, 0.0, 0.0])
    integrator.mw_coords = integrator.mw_coords + displacement
    if hasattr(integrator, "_early_converged"):
        integrator._early_converged = False
    return displacement, np.ones(3)


@pytest.mark.parametrize(("algorithm", "params_type"), _ALGORITHMS)
def test_each_integrator_reports_true_maximum_step_exhaustion(
    algorithm,
    params_type,
    tmp_path,
):
    params = params_type(
        max_steps=1,
        f_max_th=1.0e-6,
        f_rms_th=1.0e-6,
        print_each=False,
        write_traj=False,
    )
    integrator = _controlled_integrator(algorithm, params, tmp_path)
    integrator._micro_step = lambda: _nonzero_micro_step(integrator)

    branch = integrator._one_side(
        forward=True,
        sign=1.0,
        q_ts_cart=np.zeros(3),
        v_neg_mw=np.array([1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert branch["status"]["termination_reason"] == "maximum_steps"
    assert branch["status"]["converged"] is False
    assert branch["status"]["iterations_attempted"] == 1
    assert branch["status"]["accepted_macro_steps"] == 1


def test_gs_reports_zero_gradient_without_accepting_a_macro_step(tmp_path):
    params = GSParams(
        max_steps=2,
        f_max_th=1.0e-6,
        f_rms_th=1.0e-6,
        print_each=False,
        write_traj=False,
    )
    integrator = _controlled_integrator(GS, params, tmp_path)
    gradient_calls = iter((np.ones(3), np.zeros(3)))
    integrator._gradient_mw_from_forces = lambda _: next(gradient_calls)

    branch = integrator._one_side(
        forward=True,
        sign=1.0,
        q_ts_cart=np.zeros(3),
        v_neg_mw=np.array([1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert branch["status"]["termination_reason"] == "zero_gradient"
    assert branch["status"]["iterations_attempted"] == 1
    assert branch["status"]["accepted_macro_steps"] == 0


@pytest.mark.parametrize(
    ("algorithm", "params"),
    [
        (LQA, LQAParams()),
        (HPC, HPCParams()),
    ],
)
def test_integrator_reports_step_too_small_without_accepting_it(
    algorithm,
    params,
    tmp_path,
):
    params.max_steps = 2
    params.f_max_th = 1.0e-6
    params.f_rms_th = 1.0e-6
    params.print_each = False
    params.write_traj = False
    integrator = _controlled_integrator(algorithm, params, tmp_path)
    integrator._micro_step = lambda: (np.zeros(3), np.ones(3))

    branch = integrator._one_side(
        forward=False,
        sign=-1.0,
        q_ts_cart=np.zeros(3),
        v_neg_mw=np.array([1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert branch["status"]["termination_reason"] == "step_too_small"
    assert branch["status"]["iterations_attempted"] == 1
    assert branch["status"]["accepted_macro_steps"] == 0


def test_eulerpc_reports_predictor_stall_separately_from_force_convergence(
    tmp_path,
):
    params = EulerPCParams(
        max_steps=2,
        f_max_th=1.0e-6,
        f_rms_th=1.0e-6,
        print_each=False,
        write_traj=False,
    )
    integrator = _controlled_integrator(EulerPC, params, tmp_path)

    def stalled_micro_step():
        displacement = np.array([1.0e-3, 0.0, 0.0])
        integrator.mw_coords = integrator.mw_coords + displacement
        integrator._early_converged = True
        return displacement, np.ones(3)

    integrator._micro_step = stalled_micro_step
    branch = integrator._one_side(
        forward=True,
        sign=1.0,
        q_ts_cart=np.zeros(3),
        v_neg_mw=np.array([1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert branch["status"]["termination_reason"] == "predictor_stalled"
    assert branch["status"]["force_criteria_satisfied"] is False
    assert branch["status"]["converged"] is False
    assert branch["status"]["iterations_attempted"] == 1
    assert branch["status"]["accepted_macro_steps"] == 1
