"""RED input/schema contracts for advanced IRC methods on immutable baseline."""
from __future__ import annotations

import os

import pytest
from maple.function.dispatcher.irc.algorithm import EulerPC, GS, HPC, LQA
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


@pytest.mark.parametrize(
    "parameters",
    [
        {"MAX_STEPS": 1, "max_steps": 2},
        {"MAX_POINTS": 1, "max_steps": 2},
    ],
)
def test_plain_algorithm_rejects_case_and_alias_duplicates(parameters):
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match="(?i)conflicting.*max_steps"):
        HPC(atoms, output=os.devnull, paras={"hpc": parameters})


@pytest.mark.parametrize("name", ["mbs_points", "mbs_max_k"])
def test_plain_algorithm_requires_two_mbs_samples(name):
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match=name):
        HPC(atoms, output=os.devnull, paras={"hpc": {name: 1}})


def test_existing_gs_schema_keeps_zero_hessian_recalculation_disable():
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    job = GS(atoms, output=os.devnull, paras={"gs": {"hessian_recalc": 0}})
    assert job.p.hessian_recalc == 0


@pytest.mark.parametrize("bad_steps", [-1, 1.5, True])
def test_plain_algorithm_rejects_invalid_step_budgets(bad_steps):
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(ValueError, match="max_steps"):
        EulerPC(
            atoms,
            output=os.devnull,
            paras={"eulerpc": {"max_steps": bad_steps}},
        )
