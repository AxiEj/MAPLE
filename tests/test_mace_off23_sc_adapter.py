from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE


class _Delegate:
    def __init__(self, *, result=None):
        self.result = result
        self.results = {}
        self.calls = []

    def calculate(self, atoms, properties, system_changes):
        self.calls.append((atoms.copy(), tuple(properties), tuple(system_changes)))
        if self.result is None:
            positions = np.asarray(atoms.positions, dtype=float)
            self.results = {
                "energy": float(np.sum(positions**2)),
                "forces": -2.0 * positions,
            }
        else:
            self.results = dict(self.result)


def _checkpoint(tmp_path):
    path = tmp_path / "MACE-OFF23-SC_swa.model"
    path.write_bytes(b"test checkpoint")
    return path


def _atoms(symbols="HO", *, pbc=False):
    atoms = Atoms(symbols)
    atoms.positions = np.arange(len(atoms) * 3, dtype=float).reshape((-1, 3)) / 10.0
    if pbc:
        atoms.cell = [8.0, 8.0, 8.0]
        atoms.pbc = True
    atoms.info.update(charge=0, mult=1)
    return atoms


def _calculator(tmp_path, monkeypatch, *, delegate=None, **kwargs):
    from maple.function.calculator.mace import _maceoff23_sc_calculator as module

    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.MACE_OFF23_SC_SHA256,
    )
    delegate = delegate or _Delegate()
    captured = {}

    def factory(model_path, device):
        captured.update(model_path=model_path, device=device)
        return delegate

    calculator = module.MACEOFF23SCCalculator(
        device="cpu",
        model_path=str(checkpoint),
        calculator_factory=factory,
        **kwargs,
    )
    return module, calculator, delegate, captured


def test_official_identity_hash_and_cpu_float64_delegate_are_sealed(
    tmp_path, monkeypatch
):
    module, calculator, _delegate, captured = _calculator(tmp_path, monkeypatch)

    assert calculator.model_name == "mace-off23-sc"
    assert captured == {
        "model_path": str(tmp_path / "MACE-OFF23-SC_swa.model"),
        "device": "cpu",
    }
    assert calculator.provenance == {
        "provider": "mace-off23-sc",
        "checkpoint_sha256": module.MACE_OFF23_SC_SHA256,
        "checkpoint_path": str(tmp_path / "MACE-OFF23-SC_swa.model"),
        "source_revision": module.MACE_OFF23_SC_OFFICIAL_IDENTITY["source_revision"],
        "protocol_revision": module.MACE_OFF23_SC_OFFICIAL_IDENTITY[
            "protocol_revision"
        ],
        "model_card": str(module.MACE_OFF23_SC_CARD),
        "runtime_scope": "full_coupling_explicit_system_pes_only",
    }

    card = json.loads(module.MACE_OFF23_SC_CARD.read_text(encoding="utf-8"))
    assert card["checkpoint_sha256"] == module.MACE_OFF23_SC_SHA256
    assert card["capabilities"]["supports_absolute_solvation"] is False
    assert card["capabilities"]["supports_alchemical_lambda"] is False
    assert card["capabilities"]["solvation_mode"] == "none"
    assert calculator.capabilities.solvation_mode == "none"
    assert card["capabilities"]["requires_topology"] is False
    assert card["gpu_acceleration"]["no_loss_parity_verified"] is False
    assert card["gpu_acceleration"]["status"] == (
        "cuda_runtime_verified_formal_experimental_accuracy_admission_pending"
    )
    assert card["gpu_acceleration"]["allowed_tasks"] == []
    assert card["gpu_acceleration"]["allowed_inference_modes"] == []
    assert any(
        "not GPU-unsupported" in note or "supports GPU execution" in note
        for note in card["notes"]
    )


def test_energy_forces_and_pbc_multifragment_forward_through_official_delegate(
    tmp_path, monkeypatch
):
    _module, calculator, delegate, _captured = _calculator(tmp_path, monkeypatch)
    atoms = _atoms("HHOHHO", pbc=True)

    calculator.calculate(atoms, properties=("energy", "forces"))

    expected_energy_ev = float(np.sum(atoms.positions**2))
    np.testing.assert_allclose(
        calculator.results["energy"],
        expected_energy_ev * EV2HARTREE,
    )
    np.testing.assert_allclose(
        calculator.results["free_energy"],
        expected_energy_ev * EV2HARTREE,
    )
    np.testing.assert_allclose(
        calculator.results["forces"],
        -2.0 * atoms.positions * EV2HARTREE,
    )
    assert delegate.calls[0][1] == ("energy", "forces")
    assert calculator.results["model_metadata"]["provider"] == "mace-off23-sc"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("mace-off24-sc", "pinned to model identity"),
        ("forged", "pinned to model identity"),
    ],
)
def test_rejects_checkpoint_identity_substitution(
    tmp_path, monkeypatch, model, message
):
    from maple.function.calculator.mace import _maceoff23_sc_calculator as module

    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.MACE_OFF23_SC_SHA256,
    )
    with pytest.raises(ValueError, match=message):
        module.MACEOFF23SCCalculator(
            "cpu",
            model=model,
            model_path=str(checkpoint),
            calculator_factory=lambda *_args: _Delegate(),
        )


def test_rejects_missing_or_wrong_checkpoint(tmp_path):
    from maple.function.calculator.mace import _maceoff23_sc_calculator as module

    with pytest.raises(ValueError, match="explicit model_path"):
        module.MACEOFF23SCCalculator("cpu")

    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.MACEOFF23SCCalculator(
            "cpu",
            model_path=str(checkpoint),
            calculator_factory=lambda *_args: _Delegate(),
        )


def test_gpu_fails_before_checkpoint_or_delegate_resolution(tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    setter = SetCalculator(
        device="cuda",
        model="mace-off23-sc",
        output=str(tmp_path / "maple.out"),
        model_options={"model_path": str(tmp_path / "does-not-exist.model")},
        task="sp",
    )
    with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
        setter.set_calculator()


def test_d4_request_fails_closed_before_checkpoint_loading(tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    checkpoint = _checkpoint(tmp_path)
    setter = SetCalculator(
        device="cpu",
        model="mace-off23-sc",
        output=str(tmp_path / "maple.out"),
        d4=True,
        model_options={"model_path": str(checkpoint)},
        task="sp",
    )

    with pytest.raises(ValueError, match="does not support D4"):
        setter.set_calculator()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({}, r"explicit atoms\.info"),
        ({"charge": 0}, r"explicit atoms\.info"),
        ({"mult": 1}, r"explicit atoms\.info"),
        ({"charge": 1, "mult": 1}, "neutral singlets"),
        ({"charge": 0, "mult": 2}, "neutral singlets"),
        ({"charge": float("nan"), "mult": 1}, "finite integer"),
    ],
)
def test_rejects_missing_or_out_of_domain_charge_and_multiplicity(
    tmp_path, monkeypatch, updates, message
):
    _module, calculator, _delegate, _captured = _calculator(tmp_path, monkeypatch)
    atoms = _atoms("HO")
    atoms.info.clear()
    atoms.info.update(updates)

    with pytest.raises(ValueError, match=message):
        calculator.get_potential_energy(atoms)


def test_rejects_unsupported_elements_and_partial_periodicity(tmp_path, monkeypatch):
    _module, calculator, _delegate, _captured = _calculator(tmp_path, monkeypatch)

    unsupported = _atoms("Si")
    with pytest.raises(ValueError, match="does not enable elements: Si"):
        calculator.get_potential_energy(unsupported)

    partial = _atoms("HHO", pbc=True)
    partial.pbc = [True, True, False]
    with pytest.raises(ValueError, match="fully periodic or non-periodic"):
        calculator.get_potential_energy(partial)


@pytest.mark.parametrize(
    ("implicit", "solvent"),
    [
        ("gb", "water"),
        ("pb", "water"),
        ("smd", "water"),
        ("none", "water"),
    ],
)
def test_rejects_implicit_or_named_solvent_composition(
    tmp_path, monkeypatch, implicit, solvent
):
    with pytest.raises(ValueError, match="explicit-system PES"):
        _calculator(
            tmp_path,
            monkeypatch,
            implicit=implicit,
            solvent=solvent,
        )


@pytest.mark.parametrize(
    ("result", "properties", "message"),
    [
        ({}, ("energy",), "did not return energy"),
        ({"energy": float("nan")}, ("energy",), "energy must be finite"),
        ({"energy": 0.0}, ("energy", "forces"), "did not return forces"),
        (
            {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]]},
            ("energy", "forces"),
            r"shape \(N, 3\)",
        ),
        (
            {
                "energy": 0.0,
                "forces": [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]],
            },
            ("energy", "forces"),
            "forces must be finite",
        ),
    ],
)
def test_rejects_invalid_delegate_results(
    tmp_path, monkeypatch, result, properties, message
):
    delegate = _Delegate(result=result)
    _module, calculator, _delegate, _captured = _calculator(
        tmp_path,
        monkeypatch,
        delegate=delegate,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        calculator.calculate(_atoms("HO"), properties=properties)


def test_rejects_hessian_and_thermodynamic_task_claims(tmp_path, monkeypatch):
    _module, calculator, _delegate, _captured = _calculator(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="unsupported properties"):
        calculator.calculate(_atoms("HO"), properties=("hessian",))
    with pytest.raises(ValueError, match="forbids task 'absolute_solvation"):
        calculator.validate_task("absolute_solvation_free_energy")
    with pytest.raises(ValueError, match="forbids task 'alchemical_free_energy"):
        calculator.validate_task("alchemical_free_energy")


@pytest.mark.parametrize("requested", ["mace-off23-sc", "maceoff23sc"])
def test_set_calculator_discovers_cpu_only_backend(tmp_path, requested):
    from maple.function.calculator.set_calculator import SetCalculator

    setter = SetCalculator(
        device="cpu",
        model=requested,
        output=str(tmp_path / "maple.out"),
    )
    assert (
        setter._discover_calculator_class(requested).__name__ == "MACEOFF23SCCalculator"
    )
