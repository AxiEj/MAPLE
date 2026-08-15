from __future__ import annotations

from pathlib import Path

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
    validate_irc_params,
)
from maple.function.read.command_control import CommandControl


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_command_control_rejects_unknown_irc_parameter_with_suggestion():
    with pytest.raises(ValueError, match="Did you mean 'f_max_th'"):
        CommandControl.from_settings(["#irc(method=gs,f_max_thh=1e-8)"])


def test_checked_in_irc_header_accepts_its_global_model_and_device():
    input_lines = (
        REPOSITORY_ROOT / "examples" / "irc" / "gs" / "inp1.inp"
    ).read_text(encoding="utf-8").splitlines()

    command = CommandControl.from_settings(input_lines)

    assert command.task == "irc"
    assert command.params["model"] == "ani-1xnr"
    assert command.params["device"] == "gpu0"
    assert command.params["method"] == "gs"


def test_command_control_rejects_optimization_level_as_an_irc_noop():
    with pytest.raises(ValueError, match="set f_max_th and f_rms_th explicitly"):
        CommandControl.from_settings(
            ["#model=aimnet2", "#irc(method=gs)", "#level=tight"]
        )


def test_command_control_rejects_parameter_from_a_different_irc_method():
    with pytest.raises(ValueError, match="Unknown IRC parameter: 'euler_n'"):
        CommandControl.from_settings(["#irc(method=gs,euler_n=10)"])


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("#irc(method=gs,max_micro_cycles=0)", "max_micro_cycles"),
        ("#irc(method=gs,hessian_recalc=0)", "hessian_recalc"),
        ("#irc(method=gs,hessian_update=sr1)", "hessian_update"),
        ("#irc(method=lqa,euler_n=0)", "euler_n"),
        ("#irc(method=hpc,dwi_n=3)", "positive even integer"),
        ("#irc(method=hpc,mbs_max_k=1)", "mbs_max_k"),
        ("#irc(method=hpc,mbs_points=1)", "mbs_points"),
        ("#irc(method=hpc,mbs_tol=0)", "mbs_tol"),
        ("#irc(method=eulerpc,max_pred_steps=0)", "max_pred_steps"),
        ("#irc(method=eulerpc,loose_cycles=-1)", "loose_cycles"),
    ],
)
def test_command_control_rejects_invalid_method_specific_irc_values(
    setting,
    message,
):
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings([setting])


def test_command_control_normalizes_integral_float_before_integrator_use():
    command = CommandControl.from_settings(["#irc(method=gs,max_steps=50.0)"])

    assert command.params["max_steps"] == 50
    assert type(command.params["max_steps"]) is int


def test_command_control_canonicalizes_a_legacy_alias_before_defaults_merge():
    command = CommandControl.from_settings(["#irc(method=gs,max_points=7)"])

    assert command.params["max_steps"] == 7
    assert "max_points" not in command.params


@pytest.mark.parametrize(
    "settings",
    [
        ["#irc(method=gs)", "#max_points=7"],
        ["#max_points=7", "#irc(method=gs)"],
    ],
)
def test_command_control_canonicalizes_flat_legacy_alias_in_either_order(settings):
    command = CommandControl.from_settings(settings)

    assert command.params["max_steps"] == 7
    assert "max_points" not in command.params


def test_command_control_rejects_alias_and_canonical_key_together():
    with pytest.raises(ValueError, match="supplied more than once"):
        CommandControl.from_settings(
            ["#irc(method=gs,max_steps=8,max_points=7)"]
        )


def test_command_control_rejects_repeated_exact_inline_irc_key():
    with pytest.raises(ValueError, match="Duplicate nested parameter: 'max_steps'"):
        CommandControl.from_settings(
            ["#irc(method=gs,max_steps=8,max_steps=7)"]
        )


def test_command_control_rejects_inline_and_flat_irc_collision():
    with pytest.raises(ValueError, match="supplied more than once"):
        CommandControl.from_settings(
            ["#irc(method=gs,max_steps=8)", "#max_points=7"]
        )


@pytest.mark.parametrize(
    ("method", "params", "attribute"),
    [
        ("gs", GSParams(max_micro_cycles=20.0), "max_micro_cycles"),
        ("lqa", LQAParams(euler_n=5000.0), "euler_n"),
        ("hpc", HPCParams(mbs_points=20.0), "mbs_points"),
        ("eulerpc", EulerPCParams(max_pred_steps=500.0), "max_pred_steps"),
    ],
)
def test_direct_parameter_validation_normalizes_integral_float(
    method,
    params,
    attribute,
):
    validate_irc_params(params, method)

    assert type(getattr(params, attribute)) is int


@pytest.mark.parametrize(
    ("algorithm", "params", "message"),
    [
        (GS, GSParams(max_micro_cycles=0), "max_micro_cycles"),
        (LQA, LQAParams(euler_n=0), "euler_n"),
        (HPC, HPCParams(mbs_points=1), "mbs_points"),
        (EulerPC, EulerPCParams(max_pred_steps=0), "max_pred_steps"),
    ],
)
def test_integrator_direct_api_validates_its_method_controls_before_running(
    algorithm,
    params,
    message,
    tmp_path,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    integrator = algorithm(atoms, str(tmp_path / "irc.out"), params=params)

    with pytest.raises(ValueError, match=message):
        integrator.run()


def test_explicit_nested_direct_api_block_rejects_unknown_parameter(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="Unknown GS IRC parameter"):
        GS(
            atoms,
            str(tmp_path / "irc.out"),
            paras={"gs": {"f_max_thh": 1.0e-8}},
        )
