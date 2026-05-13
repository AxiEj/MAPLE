import torch
import pytest

from maple.function.calculator.set_calculator import SetClaculator
from maple.function.read.command_control import CommandControl


def test_command_control_accepts_aimnet_coulomb_options(tmp_path):
    output = tmp_path / "parse.out"

    control = CommandControl.from_settings(
        [
            "#model=aimnet2(coulomb=DSF,cutoff=12.5,dsf_alpha=0.3)",
        ],
        output_path=str(output),
    )

    assert control.params["model"] == "aimnet2"
    assert control.params["model_options"] == {
        "coulomb": "dsf",
        "cutoff": 12.5,
        "dsf_alpha": 0.3,
    }


def test_command_control_rejects_unknown_aimnet_option(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported AIMNet2 option"):
        CommandControl.from_settings(
            [
                "#model=aimnet2(pme_cutoff=12)",
            ],
            output_path=str(output),
        )


def test_command_control_rejects_unknown_aimnet_coulomb_method(tmp_path):
    output = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Unsupported AIMNet2 Coulomb method"):
        CommandControl.from_settings(
            [
                "#model=aimnet2(coulomb=pme)",
            ],
            output_path=str(output),
        )


def test_set_calculator_rejects_unknown_aimnet_option(tmp_path):
    with pytest.raises(ValueError, match="Unsupported AIMNet2 option"):
        SetClaculator(
            torch.device("cpu"),
            "aimnet2",
            str(tmp_path / "factory.out"),
            model_options={"pme_cutoff": 12},
        ).set_calculator()


def test_set_calculator_passes_aimnet_options(monkeypatch, tmp_path):
    captured = {}

    class FakeAIMNet2Calculator:
        supported_hessian_modes = ("analytic", "numerical")

        def __init__(self, **kwargs):
            captured.update(kwargs)

    from maple.function.calculator.aimnet import _aimnet2_calculator as aimnet_module

    monkeypatch.setattr(aimnet_module, "AIMNet2Calculator", FakeAIMNet2Calculator)
    monkeypatch.setattr(SetClaculator, "_ensure_model_file", lambda self, model_name: tmp_path / "aimnet2.pt")

    calculator = SetClaculator(
        torch.device("cpu"),
        "aimnet2",
        str(tmp_path / "factory.out"),
        model_options={"coulomb": "dsf", "cutoff": 12.5, "dsf_alpha": 0.3},
    ).set_calculator()

    assert calculator is not None
    assert captured["model"] == "aimnet2"
    assert captured["coulomb_method"] == "dsf"
    assert captured["cutoff"] == 12.5
    assert captured["dsf_alpha"] == 0.3

