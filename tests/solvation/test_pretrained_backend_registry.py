from __future__ import annotations

import pytest

from maple.function.calculator.set_calculator import SetCalculator

BACKEND_MODELS = [
    "aimnet2-cpcms-v2",
    "mace-off24-medium",
    "aceff-2.0",
]


@pytest.mark.parametrize("model", BACKEND_MODELS)
def test_backend_registration_discoverable_without_instantiation(model):
    calc = SetCalculator(device="cpu", model=model, output="out.log")
    cls = calc._discover_calculator_class(model)
    assert cls is not None


@pytest.mark.parametrize("model", BACKEND_MODELS)
def test_backend_requires_explicit_model_path(model, tmp_path):
    calc = SetCalculator(
        device="cpu",
        model=model,
        output=str(tmp_path / "out.log"),
    )

    with pytest.raises(ValueError, match="model_path"):
        calc.set_calculator()
