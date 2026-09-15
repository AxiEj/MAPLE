from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from maple.function.dispatcher.irc.algorithm.gs import GS
from maple.function.dispatcher.irc.algorithm.hpc import HPC
from maple.function.dispatcher.irc.algorithm.eulerpc import EulerPC
from maple.function.dispatcher.irc.algorithm.lqa import LQA
from maple.function.dispatcher.irc import irc as irc_module


def _atoms():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0, free_energy=0.0, forces=np.zeros((2, 3), dtype=np.float64)
    )
    return atoms


def _side(reason="endpoint_force_criteria_satisfied", converged=True):
    return {
        "title": "side",
        "records": [
            {
                "E": -1.0,
                "maxG": 0.0 if converged else 1.0,
                "rmsG": 0.0 if converged else 1.0,
                "x": np.zeros((2, 3), dtype=np.float64),
            }
        ],
        "E_ts": 0.0,
        "converged": converged,
        "termination_reason": reason,
        "termination_class": "converged" if converged else "bounded_nonconvergence",
        "iterations": 0,
        "final_metrics": {},
        "endpoint_minimum_verified": False,
    }


@pytest.mark.parametrize("algorithm_type", [GS, HPC, EulerPC, LQA])
def test_gs_uses_one_preselected_mode_for_opposite_directions(monkeypatch, tmp_path, algorithm_type):
    atoms = _atoms()
    gs = algorithm_type(atoms, output=str(tmp_path / "irc.out"), paras={"write_traj": False})
    mode = np.arange(1.0, 7.0)
    gs.preselected_mode_mw = mode
    calls = []

    monkeypatch.setattr(gs, "_get_hessian_cart", lambda: pytest.fail("raw TS Hessian reselected"))

    def one_side(**kwargs):
        calls.append((kwargs["sign"], kwargs["v_neg_mw"].copy()))
        return _side()

    monkeypatch.setattr(gs, "_one_side", one_side)
    monkeypatch.setattr(gs, "_merge_and_mark_ts", lambda forward, backward: {})

    result = gs.run()

    assert [sign for sign, _ in calls] == [+1.0, -1.0]
    assert np.array_equal(calls[0][1], mode)
    assert np.array_equal(calls[1][1], mode)
    assert result["completed_directions"] == ["forward", "backward"]


@pytest.mark.parametrize("algorithm_type", [GS, HPC, EulerPC, LQA])
def test_gs_preserves_forward_when_backward_raises(monkeypatch, tmp_path, algorithm_type):
    atoms = _atoms()
    gs = algorithm_type(atoms, output=str(tmp_path / "irc.out"), paras={"write_traj": False})
    gs.preselected_mode_mw = np.ones(6)

    def one_side(*, forward, **kwargs):
        if forward:
            return _side()
        gs._active_direction_log = _side("corrector_failure", False)
        raise RuntimeError("mBS Richardson extrapolation did not converge")

    monkeypatch.setattr(gs, "_one_side", one_side)
    result = gs.run()

    assert result["completed_directions"] == ["forward", "backward"]
    assert result["forward"]["converged"] is True
    assert result["backward"]["converged"] is False
    assert result["backward"]["termination_reason"] == "corrector_failure"
    assert result["backward"]["termination_class"] == "execution_failure"
    assert result["backward"]["records"]


def test_wrapper_preserves_algorithm_side_statuses(monkeypatch, tmp_path):
    atoms = _atoms()
    mode = np.ones(6) / np.sqrt(6.0)
    diagnostic = {"resolved_negative_count": 1}
    monkeypatch.setattr(
        irc_module, "_preselect_pure_nonmd_internal_mode", lambda atoms: (mode, diagnostic)
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2", lambda atoms: True
    )

    class FakeGS:
        def __init__(self, *args, **kwargs):
            self.p = SimpleNamespace(f_max_th=0.1, f_rms_th=0.1)

        def run(self):
            return {
                "forward": _side("maximum_steps_reached", False),
                "backward": _side("microcycle_limit_reached", False),
                "completed_directions": ["forward", "backward"],
            }

    monkeypatch.setattr("maple.function.dispatcher.irc.algorithm.GS", FakeGS)
    status = irc_module.IRC({}, str(tmp_path / "irc.out"), atoms, "gs").run()

    assert status["termination_class"] == "bounded_nonconvergence"
    assert status["directions"]["forward"]["termination_reason"] == "maximum_steps_reached"
    assert status["directions"]["backward"]["termination_reason"] == "microcycle_limit_reached"
    assert status["directions"]["backward"]["endpoint_minimum_verified"] is False
    assert status["trace"][1]["records"][0]["positions_angstrom"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_mode_validation_failure_is_not_bounded_nonconvergence(monkeypatch, tmp_path):
    atoms = _atoms()
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2", lambda atoms: True
    )
    monkeypatch.setattr(
        irc_module,
        "_preselect_pure_nonmd_internal_mode",
        lambda atoms: (_ for _ in ()).throw(
            irc_module.InvalidStartingHessian("IRC requires exactly one uncertainty-resolved negative internal mode.")
        ),
    )

    status = irc_module.IRC({}, str(tmp_path / "irc.out"), atoms, "gs").run()

    assert status["converged"] is False
    assert status["termination_class"] == "validation_failure"
    assert status["termination_reason"] == "invalid_starting_hessian_index"


def test_gs_microcycle_cap_stops_without_accepting_a_false_endpoint(monkeypatch, tmp_path):
    atoms = _atoms()
    runner = GS(atoms, output=str(tmp_path / "micro.out"), paras={
        "write_traj": False, "max_steps": 2, "max_micro_cycles": 0,
    })
    runner.preselected_mode_mw = np.ones(6)
    monkeypatch.setattr(runner, "_get_hessian_cart", lambda: np.eye(6))
    monkeypatch.setattr(runner, "_energy_forces_from_mw", lambda q: (-1.0, np.ones((2, 3))))
    result = runner.run()
    for name in ("forward", "backward"):
        assert result[name]["converged"] is False
        assert result[name]["termination_reason"] == "microcycle_limit_reached"
        assert len(result[name]["records"]) == 1
