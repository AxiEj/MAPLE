"""RED input/schema contracts for advanced IRC methods on immutable baseline."""
from __future__ import annotations

import os

import pytest
from maple.function.dispatcher.irc.algorithm import EulerPC, HPC, LQA
from ase import Atoms






def test_nested_eulerpc_parameters_reach_plain_algorithm():
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    job = EulerPC(
        atoms,
        output=os.devnull,
        paras={"eulerpc": {"max_steps": 0, "max_pred_steps": 17, "write_traj": False}},
    )
    assert job.p.max_steps == 0
    assert job.p.max_pred_steps == 17
    assert job.p.write_traj is False


def test_unknown_nested_eulerpc_parameter_is_rejected():
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match="(?i)unknown.*parameter"):
        EulerPC(atoms, output=os.devnull, paras={"eulerpc": {"bogus": 1}})


@pytest.mark.parametrize(
    ("method", "job_type"),
    [("hpc", HPC), ("lqa", LQA), ("eulerpc", EulerPC)],
)
def test_plain_algorithm_rejects_unknown_parameters(method, job_type):
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match="(?i)unknown.*parameter"):
        job_type(atoms, output=os.devnull, paras={method: {"bogus": 1}})


@pytest.mark.parametrize("bad_steps", [-1, 1.5, True])
def test_plain_algorithm_rejects_invalid_step_budgets(bad_steps):
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match="max_steps"):
        EulerPC(
            atoms,
            output=os.devnull,
            paras={"eulerpc": {"max_steps": bad_steps}},
        )
