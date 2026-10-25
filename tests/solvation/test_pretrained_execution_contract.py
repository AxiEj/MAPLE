"""Route-scoped header/device defaults and narrow scalar composition gates."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader
from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.calculator.solvation_execution import SolvationExecution

ANY = (
    "#solv(implicit=water,method=anysolv,experimental=true,source_root=/pinned/source)"
)
CON = "#model=consolv(native_solvent=ethanol)"


@pytest.mark.parametrize(
    "settings,key", [(["#model=aimnet2", ANY], "solv"), ([CON], "model_options")]
)
def test_defaults_are_route_scoped_cuda32_reference(settings, key):
    params = CommandControl.from_settings([*settings, "#sp"]).as_dict()
    assert params["device"] == "cuda:0"
    assert params[key]["dtype"] == "float32"
    assert params[key]["inference"] == "reference"
    gas = CommandControl.from_settings(["#model=aimnet2", "#sp"]).as_dict()
    assert "device" not in gas and "model_options" not in gas


@pytest.mark.parametrize(
    "setting,key",
    [
        (ANY[:-1] + ",dtype=float64)", "solv"),
        (CON[:-1] + ",dtype=float64)", "model_options"),
    ],
)
def test_explicit_cpu64_controls_are_owned_by_the_component(setting, key):
    header = [setting] if key == "model_options" else ["#model=aimnet2", setting]
    params = CommandControl.from_settings([*header, "#device=cpu", "#sp"]).as_dict()
    assert params["device"] == "cpu"
    assert params[key]["dtype"] == "float64"


@pytest.mark.parametrize(
    "option",
    [
        "dtype=float16",
        "dtype=123",
        "inference=true",
        "inference=fast_gpu",
        "dtype=float64,inference=optimized",
    ],
)
@pytest.mark.parametrize("route", ["anysolv", "consolv"])
def test_invalid_execution_options_fail_at_the_header(option, route):
    header = (
        [CON[:-1] + f",{option})"]
        if route == "consolv"
        else ["#model=aimnet2", ANY[:-1] + f",{option})"]
    )
    with pytest.raises(ValueError, match="dtype|inference|requires CUDA"):
        CommandControl.from_settings([*header, "#sp"])


@pytest.mark.parametrize("option", ["dtype=float32", "inference=optimized"])
def test_new_options_do_not_leak_into_other_solvation_routes(option):
    with pytest.raises(ValueError, match="Unknown.*solvation"):
        CommandControl.from_settings(
            ["#model=aimnet2", "#sp", f"#solv(explicit=water,{option})"]
        )


@pytest.mark.parametrize("header", [[CON], ["#model=aimnet2", ANY]])
def test_reader_never_falls_back_from_requested_solvent_cuda(
    tmp_path, monkeypatch, header
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = InputReader()
    reader.output = str(tmp_path / "reader.out")
    with pytest.raises(ValueError, match="unavailable; CPU fallback is disabled"):
        reader.settings_command([*header, "#sp"])
    assert "Falling back to CPU" not in (tmp_path / "reader.out").read_text()


def test_unrelated_reader_keeps_existing_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = InputReader()
    reader.output = str(tmp_path / "gas.out")
    reader.settings_command(["#model=aimnet2", "#device=cuda:0", "#sp"])
    assert str(reader.device) == "cpu"
    assert "Falling back to CPU" in (tmp_path / "gas.out").read_text()


def test_non_string_solvent_device_fails_clearly():
    with pytest.raises(ValueError, match="device name"):
        CommandControl.from_settings([CON, "#device=0", "#sp"])


def test_scalar_composite_qualifies_before_resolving_any_weights(tmp_path, monkeypatch):
    from maple.function.calculator import set_calculator as module

    class Vacuum(CalcABC):
        __module__ = "maple.function.calculator.aimnet._aimnet2_calculator"
        MODEL_NAMES = ("aimnet2",)
        OPTION_KEYS = ()

    setter = SetCalculator(
        "cuda:0",
        "aimnet2",
        str(tmp_path / "out"),
        implicit="anysolv",
        solvent="water",
        task="sp",
        solvation_options={"source_root": "/pinned/source", "experimental": True},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(setter, "_discover_calculator_class", lambda _: Vacuum)
    monkeypatch.setattr(
        setter,
        "_resolve_model_path",
        lambda *args: pytest.fail("weights resolved before composite admission"),
    )
    calls = []
    card = SimpleNamespace(validate_task=lambda task: calls.append(task))
    monkeypatch.setattr(module, "load_model_provenance_card", lambda name, root: card)

    def reject(self, card, **kwargs):
        raise ValueError("composite evidence rejected")

    monkeypatch.setattr(SolvationExecution, "validate_card", reject)
    with pytest.raises(ValueError, match="composite evidence rejected"):
        setter.set_calculator()
    assert calls == ["sp"]


def test_composite_pin_never_authorizes_a_different_base_checkpoint(tmp_path):
    path = tmp_path / "base.pt"
    path.write_bytes(b"qualified")
    setter = SetCalculator("cpu", "aimnet2", str(tmp_path / "out"), task="sp")
    setter._anysolv_composite_card = SimpleNamespace(
        payload={
            "checkpoints": {
                "gas": {"checkpoint_sha256": hashlib.sha256(b"qualified").hexdigest()}
            }
        }
    )
    setter._validate_anysolv_base_checkpoint(path)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        setter._validate_anysolv_base_checkpoint(path)
