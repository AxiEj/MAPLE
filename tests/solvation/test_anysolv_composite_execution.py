"""The scalar composition owns both gas and solvent execution precision."""

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
from maple.function.calculator.calculator_base import CalcABC, EV2HARTREE
from maple.function.calculator.solvation_execution import SolvationExecution
from maple.function.calculator.extra_correction.implicit.result import SolvationResult


class _Gas(torch.nn.Module):
    def __init__(self, strict=False):
        super().__init__()
        self.register_buffer("weight", torch.eye(3))
        self.strict = strict
        self.seen = []

    def forward(self, data):
        if self.strict:
            assert not torch.is_autocast_enabled("cpu")
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert all(
                not value.is_floating_point() or value.dtype == self.weight.dtype
                for value in data.values()
            )
        self.seen.append(data["coord"].dtype)
        return {"energy": (data["coord"] @ self.weight).square().sum().reshape(1)}


class _Correction:
    supported_properties = {"energy"}

    def evaluate(self, atoms, *, need_forces=False, calculator=None):
        assert not need_forces
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.backends.cudnn.allow_tf32
        return SolvationResult(
            energy_hartree=0.01, quantity_kind="geometry_level_energy_correction"
        )


def _calculator(strict=False):
    calculator = AIMNet2Calculator.__new__(AIMNet2Calculator)
    CalcABC.__init__(calculator)
    calculator.device = torch.device("cpu")
    calculator.model_name = "aimnet2"
    calculator.model = _Gas(strict)
    calculator.solvent_correction = _Correction() if strict else None
    calculator._build_data = lambda coord, atoms: {
        "coord": coord,
        "charge": torch.zeros(1, dtype=torch.float32),
    }
    return calculator


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_whole_composition_owns_precision_and_dtype_under_hostile_autocast(dtype):
    calculator = _calculator(strict=True)
    execution = SolvationExecution("cpu", dtype)
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0.7, 0, 0]], info={"charge": 0, "mult": 1}
    )
    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        calculator._set_scalar_composite_execution(execution)
        assert torch.is_autocast_enabled("cpu")
        calculator.calculate(atoms, properties=["energy"])
        assert torch.is_autocast_enabled("cpu")
    assert before == (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    assert calculator.model.seen == [getattr(torch, dtype)]
    assert calculator.model.weight.dtype == getattr(torch, dtype)
    assert calculator.results["energy"] == pytest.approx(0.49 * EV2HARTREE + 0.01)
    with pytest.raises(ValueError, match="constructor-immutable"):
        calculator._set_scalar_composite_execution(execution)


def test_standalone_aimnet2_keeps_its_native_fp32_path():
    calculator = _calculator()
    calculator.calculate(
        Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]]), properties=["energy"]
    )
    assert calculator.model.seen == [torch.float32]
    assert not hasattr(calculator, "_scalar_composite_execution")
