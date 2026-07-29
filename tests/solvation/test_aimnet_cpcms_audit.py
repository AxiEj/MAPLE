from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from typing import Dict

from maple.function.calculator.set_calculator import _BUILTIN_NAME_TO_MODULE
from maple.function.calculator.aimnet import _aimnet2_cpcms_calculator as aim_mod
from maple.function.calculator.aimnet._aimnet2_cpcms_calculator import (
    AIMNet2CPCMSCalculator,
)


class _ShapeProbeModel(torch.nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coord = data["coord"]
        numbers = data["numbers"]
        charge = data["charge"]
        torch._assert(coord.dim() == 3, "coord_dim")
        torch._assert(coord.shape[0] == 1, "coord_batch")
        torch._assert(coord.shape[1] == numbers.shape[1], "ncoord_matches_numbers")
        torch._assert(numbers.dim() == 2 and numbers.shape[0] == 1, "numbers_shape")
        torch._assert(charge.dim() == 1 and charge.shape[0] == 1, "charge_shape")
        return {"energy": (coord * coord).sum()}


class _QuadraticModel(torch.nn.Module):
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.offset = torch.tensor(float(offset), dtype=torch.float32)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coord = data["coord"]
        return {"energy": (coord * coord).sum() + self.offset}


class _NoForcesModel(torch.nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        coord = data["coord"]
        return (coord * coord).sum()


def _write_script_model(path, model):
    scripted = torch.jit.script(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))
    return path


def _write_model(path, *, offset=0.0, shape_only=False, tensor_output=False):
    if shape_only:
        return _write_script_model(path, _ShapeProbeModel())
    if tensor_output:
        return _write_script_model(path, _NoForcesModel())
    return _write_script_model(path, _QuadraticModel(offset=offset))


def _water_atoms() -> Atoms:
    atoms = Atoms(
        symbols="H2O",
        positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.239987, 0.927297, 0.0)],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def _water_pbc_atoms() -> Atoms:
    atoms = _water_atoms()
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(True)
    return atoms


def _finite_difference_force(
    atoms: Atoms, calc: AIMNet2CPCMSCalculator, idx: int, delta: float = 1e-3
) -> float:
    base = atoms.get_positions().copy()
    plus = atoms.copy()
    pos_plus = base.copy()
    pos_plus.flat[idx] += delta
    plus.set_positions(pos_plus)
    ep = calc.get_potential_energy(plus)
    minus = atoms.copy()
    pos_minus = base.copy()
    pos_minus.flat[idx] -= delta
    minus.set_positions(pos_minus)
    em = calc.get_potential_energy(minus)
    return -(ep - em) / (2 * delta)


@pytest.fixture
def official_sha(monkeypatch):
    monkeypatch.setattr(
        aim_mod, "_sha256_for_path", lambda _path: aim_mod.DEFAULT_CHECKPOINT_SHA256
    )


def test_aimnet2_cpcms_rejects_additive_implicit_and_solvent(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="additive implicit"):
        AIMNet2CPCMSCalculator(device="cpu", model_path=str(model), implicit="gbsa")
    with pytest.raises(ValueError, match="does not accept additive solvent"):
        AIMNet2CPCMSCalculator(device="cpu", model_path=str(model), solvent="water")


def test_aimnet2_cpcms_rejects_user_supplied_model_card(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt")
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{}")
    with pytest.raises(
        ValueError, match="does not accept user-supplied model_card_path"
    ):
        AIMNet2CPCMSCalculator(
            device="cpu", model_path=str(model), model_card_path=str(bogus)
        )


def test_aimnet2_cpcms_shape_contract(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt", shape_only=True)
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    energy = calc.get_potential_energy(_water_atoms())
    assert isinstance(energy, float)
    assert "model_metadata" in calc.results


def test_aimnet2_cpcms_forces_are_autograd_gradients(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt", offset=0.125)
    atoms = _water_atoms()
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    forces = calc.get_forces(atoms)
    expected = np.array(
        [_finite_difference_force(atoms, calc, i, 1e-4) for i in range(3 * len(atoms))]
    )
    assert forces.shape == (len(atoms), 3)
    assert np.allclose(forces.reshape(-1), expected, atol=1e-3, rtol=1e-3)


def test_aimnet2_cpcms_numerical_hessian_is_square_and_symmetric(
    tmp_path, official_sha
):
    model = _write_model(tmp_path / "model.pt", offset=0.0)
    atoms = _water_atoms()
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    hessian = calc.get_hessian(atoms)
    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    assert np.allclose(hessian, hessian.T, atol=1e-8)


def test_aimnet2_cpcms_tensor_output_is_accepted(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt", tensor_output=True)
    atoms = _water_atoms()
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    energy = calc.get_potential_energy(atoms)
    assert isinstance(energy, float)


def test_aimnet2_cpcms_rejects_non_singleton_multiplicity_and_pbc(
    tmp_path, official_sha
):
    model = _write_model(tmp_path / "model.pt")
    atoms = _water_atoms()
    atoms.info["mult"] = 2
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    with pytest.raises(ValueError, match="multiplicity=1"):
        calc.get_potential_energy(atoms)
    with pytest.raises(NotImplementedError, match="no-PBC"):
        calc.get_potential_energy(_water_pbc_atoms())


def test_aimnet2_cpcms_attests_model_metadata_and_provenance(tmp_path, official_sha):
    model = _write_model(tmp_path / "model.pt", offset=0.3)
    atoms = _water_atoms()
    calc = AIMNet2CPCMSCalculator(device="cpu", model_path=str(model))
    calc.calculate(atoms)
    assert calc.results["model_metadata"]["model_name"] == "aimnet2-cpcms-v2"
    assert calc.results["model_metadata"]["model_card_path"].endswith(
        "aimnet2-cpcms-v2.yaml"
    )
    assert calc.results["provenance"]["calculator"] == "AIMNet2CPCMSCalculator"


def test_aimnet2_cpcms_is_registered():
    assert "aimnet2-cpcms-v2" in _BUILTIN_NAME_TO_MODULE
    assert (
        _BUILTIN_NAME_TO_MODULE["aimnet2-cpcms-v2"]
        == "maple.function.calculator.aimnet._aimnet2_cpcms_calculator"
    )
