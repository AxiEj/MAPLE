from __future__ import annotations

import json
import os
import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.consolv import _consolv_calculator as module
from maple.function.calculator.consolv._consolv_calculator import (
    CONSOLV_DEFAULT_MODEL_PATH,
    CONSOLV_TORCH_CHECKPOINT_SHA256,
    ConSolvCalculator,
)


class _FakeTorchModel:
    metadata = {"source_revision": module.CONSOLV_SOURCE_REVISION}

    def __init__(self):
        self.calls = []

    def energy_ev(self, numbers, positions, solvent_index):
        self.calls.append((numbers.clone(), positions.clone(), solvent_index, False))
        return torch.tensor(1.0, dtype=torch.float64)

    def energy_and_forces_ev(self, numbers, positions, solvent_index):
        self.calls.append((numbers.clone(), positions.clone(), solvent_index, True))
        return torch.tensor(1.0, dtype=torch.float64), torch.ones_like(positions)


def _atoms(symbols="CO"):
    atoms = Atoms(symbols, positions=np.arange(len(symbols) * 3).reshape(-1, 3) / 10)
    atoms.info.update(charge=0, mult=1)
    return atoms


@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "consolv-torch-v1.pt"
    path.write_bytes(b"test checkpoint")
    model = _FakeTorchModel()
    monkeypatch.setattr(module, "_sha256", lambda _: CONSOLV_TORCH_CHECKPOINT_SHA256)
    monkeypatch.setattr(module, "load_consolv_torch_checkpoint", lambda *a, **k: model)
    return path, model


def test_torch_units_solvent_index_and_metadata(fake_checkpoint):
    path, model = fake_checkpoint
    calculator = ConSolvCalculator(
        device="cpu", native_solvent="ethanol", model_path=str(path)
    )
    calculator.calculate(_atoms(), properties=("energy", "forces"))
    assert calculator.results["energy"] == pytest.approx(EV2HARTREE)
    assert calculator.results["forces"] == pytest.approx(np.ones((2, 3)) * EV2HARTREE)
    assert model.calls[0][2:] == (19, True)
    assert calculator.results["model_metadata"]["runtime"] == "pure_torch"
    assert calculator.results["model_metadata"]["compute_dtype"] == "float64"


def test_energy_request_does_not_compute_or_expose_forces(fake_checkpoint):
    path, model = fake_checkpoint
    calculator = ConSolvCalculator(
        device="cpu", native_solvent="toluene", model_path=str(path)
    )
    calculator.calculate(_atoms(), properties=("energy",))
    assert "forces" not in calculator.results
    assert model.calls[0][2:] == (45, False)


@pytest.mark.parametrize("through_ase", [False, True])
def test_attached_correction_rejected_before_computing_or_reusing_cache(
    fake_checkpoint, through_ase
):
    path, model = fake_checkpoint
    calculator = ConSolvCalculator(
        device="cpu", native_solvent="ethanol", model_path=path
    )
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    calculator.solvent_correction = object()
    with pytest.raises(ValueError, match="native.*correction"):
        if through_ase:
            atoms.get_potential_energy()
        else:
            calculator.calculate(atoms)
    assert calculator.results == {}
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"implicit": "gb", "solvent": "water"}, "native solution potential"),
        ({"native_solvent": "water"}, "frozen 66"),
        ({"native_solvent": "unknown"}, "frozen 66"),
        ({"execution_task": "opt"}, "explicitly forbids"),
        ({"device": "cuda"}, "no frozen no-loss"),
    ],
)
def test_composition_solvent_task_and_device_fail_closed(
    fake_checkpoint, kwargs, message
):
    path, _ = fake_checkpoint
    kwargs.setdefault("native_solvent", "ethanol")
    kwargs.setdefault("device", "cpu")
    kwargs["model_path"] = str(path)
    with pytest.raises(ValueError, match=message):
        ConSolvCalculator(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda atoms: atoms.info.update(charge=1), "neutral closed-shell"),
        (lambda atoms: atoms.info.update(mult=2), "neutral closed-shell"),
        (lambda atoms: atoms.set_pbc(True), "no-PBC"),
        (lambda atoms: atoms.set_atomic_numbers([6, 14]), "unsupported atomic"),
        (
            lambda atoms: atoms.set_positions([[0, 0, 0], [20, 0, 0]]),
            "geometry-connected",
        ),
        (lambda atoms: atoms.positions.__setitem__((0, 0), np.nan), "finite"),
    ],
)
def test_system_domain_fail_closed(fake_checkpoint, mutator, message):
    path, _ = fake_checkpoint
    calculator = ConSolvCalculator(
        device="cpu", native_solvent="ethanol", model_path=str(path)
    )
    atoms = _atoms()
    mutator(atoms)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        calculator.calculate(atoms)


def test_checkpoint_hash_checked_before_load(tmp_path, monkeypatch):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"wrong")
    monkeypatch.setattr(
        module,
        "load_consolv_torch_checkpoint",
        lambda *a, **k: pytest.fail("must reject before loading"),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        ConSolvCalculator(device="cpu", native_solvent="ethanol", model_path=str(path))


@pytest.mark.parametrize("model_path", [None, "", "   ", 7, False, object()])
def test_explicit_model_path_never_falls_back(model_path):
    error = ValueError if isinstance(model_path, str) else TypeError
    with pytest.raises(error, match="model_path must be a non-empty path string"):
        ConSolvCalculator(device="cpu", native_solvent="ethanol", model_path=model_path)


def test_constructor_configuration_is_read_only(fake_checkpoint):
    path, _ = fake_checkpoint
    calculator = ConSolvCalculator(
        device="cpu", native_solvent="ethanol", model_path=path
    )
    expected = {
        "device": "cuda",
        "native_solvent": "toluene",
        "solvent_index": 45,
        "model_path": path.with_name("other.pt"),
    }
    for name, value in expected.items():
        with pytest.raises(AttributeError):
            setattr(calculator, name, value)
        with pytest.raises(AttributeError, match="constructor"):
            calculator.set(**{name: value})
    assert calculator.device == "cpu"
    assert calculator.native_solvent == "ethanol"
    assert calculator.solvent_index == 19
    assert calculator.model_path == path.resolve()


def test_build_kwargs_preserves_explicit_empty_model_path():
    kwargs = ConSolvCalculator.build_kwargs_from_options(
        "consolv", {"native_solvent": "ethanol", "model_path": ""}
    )
    assert kwargs["model_path"] == ""


def test_build_kwargs_omits_unspecified_model_path():
    kwargs = ConSolvCalculator.build_kwargs_from_options(
        "consolv", {"native_solvent": "ethanol"}
    )
    assert "model_path" not in kwargs


@pytest.mark.parametrize("value", [None, "", "   ", 7, False])
def test_factory_rejects_invalid_consolv_path_before_resolution(
    tmp_path, monkeypatch, value
):
    from maple.function.calculator.set_calculator import SetCalculator

    setter = SetCalculator(
        "cpu",
        "consolv",
        str(tmp_path / "run.out"),
        task="sp",
        model_options={"native_solvent": "ethanol", "model_path": value},
    )
    monkeypatch.setattr(
        setter, "_discover_calculator_class", lambda _: ConSolvCalculator
    )
    monkeypatch.setattr(
        setter,
        "_resolve_model_path",
        lambda *args: pytest.fail("default resolution reached"),
    )
    with pytest.raises(ValueError, match="non-empty path string"):
        setter.set_calculator()


@pytest.mark.parametrize("value", ["", "123", "false"])
def test_header_rejects_invalid_consolv_path(value):
    from maple.function.read.command_control import CommandControl

    with pytest.raises(ValueError, match="non-empty path string"):
        CommandControl.from_settings(
            [f"#model=consolv(native_solvent=ethanol,model_path={value})", "#sp"]
        )


def test_ase_cache_and_state_remain_mutable_with_read_only_configuration(
    fake_checkpoint,
):
    path, model = fake_checkpoint
    atoms = _atoms()
    atoms.calc = ConSolvCalculator(
        device="cpu", native_solvent="ethanol", model_path=path
    )
    atoms.get_potential_energy()
    atoms.get_potential_energy()
    assert len(model.calls) == 1
    atoms.positions[0, 0] += 0.01
    atoms.get_potential_energy()
    assert len(model.calls) == 2
    atoms.info["mult"] = 2
    with pytest.raises(ValueError, match="neutral closed-shell"):
        atoms.get_potential_energy()
    assert atoms.calc.results == {}
    assert len(model.calls) == 2


def test_model_card_is_native_sp_only_and_no_gpu_admission():
    payload = json.loads(module.CONSOLV_CARD.read_text())
    assert payload["capabilities"]["solvation_mode"] == "native"
    assert payload["capabilities"]["forces"] is True
    assert payload["capabilities"]["hessian"] == "none"
    assert payload["capabilities"]["supports_md"] is False
    assert set(payload["forbidden_tasks"]) == {
        "opt",
        "scan",
        "ts",
        "irc",
        "frequency",
        "md",
        "absolute_solvation_free_energy",
        "alchemical_free_energy",
    }
    assert payload["gpu_acceleration"]["no_loss_parity_verified"] is False


@pytest.mark.skipif(
    os.environ.get("MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS") != "1"
    or not CONSOLV_DEFAULT_MODEL_PATH.is_file(),
    reason="set MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS=1 with the frozen checkpoint",
)
def test_real_checkpoint_direct_calculator_two_solvents():
    atoms = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.927, 0]],
        info={"charge": 0, "mult": 1},
    )
    energies = []
    for solvent in ("ethanol", "toluene"):
        calculator = ConSolvCalculator(device="cpu", native_solvent=solvent)
        calculator.calculate(atoms, properties=("energy", "forces"))
        assert np.isfinite(calculator.results["energy"])
        assert np.isfinite(calculator.results["forces"]).all()
        energies.append(calculator.results["energy"])
    assert energies[0] != energies[1]
