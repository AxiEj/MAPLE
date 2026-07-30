from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE


class _HarmonicDelegate:
    def __init__(
        self,
        *,
        force_scale: float = 1.0,
        result: dict[str, object] | None = None,
    ):
        self.force_scale = force_scale
        self.result = result
        self.results: dict[str, object] = {}
        self.calls: list[tuple[Atoms, tuple[str, ...], tuple[str, ...]]] = []

    def calculate(self, atoms, properties, system_changes):
        self.calls.append((atoms.copy(), tuple(properties), tuple(system_changes)))
        if self.result is not None:
            self.results = dict(self.result)
            return
        positions = np.asarray(atoms.positions, dtype=float)
        self.results = {"energy": 0.5 * float(np.sum(positions**2))}
        if "forces" in properties:
            self.results["forces"] = -self.force_scale * positions


class _StatefulHarmonicDelegate:
    """Mimic an upstream delegate that trusts ASE's system_changes."""

    def __init__(self):
        self.positions: np.ndarray | None = None
        self.results: dict[str, object] = {}

    def calculate(self, atoms, properties, system_changes):
        if self.positions is None or "positions" in system_changes:
            self.positions = np.asarray(atoms.positions, dtype=float).copy()
        self.results = {"energy": 0.5 * float(np.sum(self.positions**2))}
        if "forces" in properties:
            self.results["forces"] = -self.positions


class _ContextRecorder:
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    @contextmanager
    def __call__(self, device: str) -> Iterator[None]:
        self.events.append(("enter", device))
        try:
            yield
        finally:
            self.events.append(("exit", device))


def _checkpoint(tmp_path: Path, size: str = "small") -> Path:
    filename = {
        "small": "fennix-bio1S.fnx",
        "medium": "fennix-bio1M.fnx",
    }[size]
    path = tmp_path / filename
    path.write_bytes(f"test {size} checkpoint".encode())
    return path


def _atoms(symbols: str = "CO", *, pbc: bool = False) -> Atoms:
    atoms = Atoms(symbols)
    atoms.positions = np.arange(len(atoms) * 3, dtype=float).reshape((-1, 3)) / 10.0
    if pbc:
        atoms.cell = [8.0, 8.0, 8.0]
        atoms.pbc = True
    atoms.info.update(charge=0, mult=1)
    return atoms


def _runtime_receipt(module) -> dict[str, str]:
    return {
        "distribution": "FeNNol",
        "version": module.FENNOL_DISTRIBUTION_VERSION,
        "ase_sha256": module.FENNOL_ASE_SHA256,
        "preprocessing_sha256": module.FENNOL_PREPROCESSING_SHA256,
    }


def _calculator(
    tmp_path: Path,
    monkeypatch,
    *,
    size: str = "small",
    model: str = "fennix-bio1",
    delegate: _HarmonicDelegate | _StatefulHarmonicDelegate | None = None,
    context_recorder: _ContextRecorder | None = None,
    **kwargs,
):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    checkpoint = _checkpoint(tmp_path, size)
    expected = module.FENNIX_BIO1_CHECKPOINTS[size]["sha256"]
    monkeypatch.setattr(module, "_sha256", lambda _path: expected)
    delegate = delegate or _HarmonicDelegate()
    context_recorder = context_recorder or _ContextRecorder()
    captured: dict[str, object] = {}

    def factory(
        model_path: str,
    ) -> _HarmonicDelegate | _StatefulHarmonicDelegate:
        captured["model_path"] = model_path
        captured["factory_in_context"] = context_recorder.events[-1] == (
            "enter",
            "cpu",
        )
        return delegate

    calculator = module.FeNNixBio1Calculator(
        device="cpu",
        model=model,
        model_path=str(checkpoint),
        size=size,
        calculator_factory=factory,
        device_context_factory=context_recorder,
        runtime_validator=lambda: _runtime_receipt(module),
        **kwargs,
    )
    return module, calculator, delegate, context_recorder, captured


def test_official_factory_seals_float64_highest_without_fast_preprocessing(
    monkeypatch,
):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    captured = {}

    class FakeFENNIXCalculator:
        results: dict[str, object] = {}

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def calculate(self, atoms, properties, system_changes):
            raise AssertionError("factory contract test must not execute the delegate")

    fake_package = types.ModuleType("fennol")
    fake_ase = types.ModuleType("fennol.ase")
    setattr(fake_ase, "FENNIXCalculator", FakeFENNIXCalculator)
    monkeypatch.setitem(sys.modules, "fennol", fake_package)
    monkeypatch.setitem(sys.modules, "fennol.ase", fake_ase)

    module._official_calculator_factory("/models/fennix-bio1S.fnx")

    assert captured == {
        "model": "/models/fennix-bio1S.fnx",
        "gpu_preprocessing": False,
        "use_float64": True,
        "matmul_prec": "highest",
    }


@pytest.mark.parametrize("size", ["small", "medium"])
def test_pinned_identity_hash_license_and_precision_contract(
    tmp_path, monkeypatch, size
):
    module, calculator, _delegate, context, captured = _calculator(
        tmp_path,
        monkeypatch,
        size=size,
        model=f"fennix-bio1-{size}",
    )

    assert calculator.model_name == "fennix-bio1"
    assert calculator.model_size == size
    assert captured == {
        "model_path": str(tmp_path / module.FENNIX_BIO1_CHECKPOINTS[size]["filename"]),
        "factory_in_context": True,
    }
    assert context.events == [("enter", "cpu"), ("exit", "cpu")]
    assert calculator.provenance["checkpoint_sha256"] == (
        module.FENNIX_BIO1_CHECKPOINTS[size]["sha256"]
    )
    assert calculator.provenance["source_revision"] == (
        module.FENNIX_BIO1_SOURCE_REVISION
    )
    assert calculator.provenance["runtime_receipt"] == _runtime_receipt(module)
    assert calculator.provenance["checkpoint_source_revision"] == (
        module.FENNIX_BIO1_PMC_REVISION
    )
    assert calculator.provenance["precision"] == {
        "requested_float64": True,
        "cpu_neighborlist_coordinate_dtype": "float32",
        "effective_precision": "unverified_mixed_or_checkpoint_serialized",
        "end_to_end_float64_verified": False,
        "matmul_precision": "highest",
        "gpu_preprocessing": False,
    }

    card = json.loads(module.FENNIX_BIO1_CARD.read_text(encoding="utf-8"))
    assert card["license"] == {
        "runtime_code": "GNU LGPL-3.0",
        "checkpoint": ("Academic Software License; academic non-commercial use only"),
    }
    assert card["precision"]["requested_float64"] is True
    assert card["precision"]["cpu_neighborlist_coordinate_dtype"] == "float32"
    assert card["precision"]["effective_precision"] == (
        "unverified_mixed_or_checkpoint_serialized"
    )
    assert card["precision"]["end_to_end_float64_verified"] is False
    assert card["precision"]["training_or_fine_tuning"] is False
    assert card["precision"]["additional_compile_or_fast_mode_enabled"] is False
    assert card["source"]["distribution_version"] == (
        module.FENNOL_DISTRIBUTION_VERSION
    )
    assert card["source"]["ase_adapter_sha256"] == module.FENNOL_ASE_SHA256
    assert card["source"]["preprocessing_sha256"] == (
        module.FENNOL_PREPROCESSING_SHA256
    )
    assert "representability boundary" in card["domain_evidence"]["contract"]
    assert card["recommended_tasks"] == []
    assert card["mechanically_exposed_tasks_pending_task_specific_validation"]
    assert {"stress", "npt"}.issubset(card["forbidden_tasks"])
    assert card["gpu_acceleration"]["no_loss_parity_verified"] is False
    assert card["gpu_acceleration"]["allowed_tasks"] == []
    assert card["gpu_acceleration"]["allowed_inference_modes"] == []
    assert card["capabilities"]["supports_absolute_solvation"] is False
    assert card["capabilities"]["supports_alchemical_lambda"] is False
    assert card["capabilities"]["solvation_mode"] == "none"


def test_energy_forces_free_energy_and_context_forwarding(tmp_path, monkeypatch):
    _module, calculator, delegate, context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    atoms = _atoms("HHOHHO", pbc=True)

    calculator.calculate(atoms, properties=("energy", "forces", "free_energy"))

    expected_energy_ev = 0.5 * float(np.sum(atoms.positions**2))
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
        -atoms.positions * EV2HARTREE,
    )
    assert isinstance(delegate, _HarmonicDelegate)
    assert delegate.calls[0][1] == ("energy", "forces")
    assert context.events[-2:] == [("enter", "cpu"), ("exit", "cpu")]
    assert calculator.results["model_metadata"]["runtime_scope"] == (
        "explicit_system_pes_only"
    )


def test_numerical_hessian_uses_common_mechanics_after_force_check(
    tmp_path, monkeypatch
):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    atoms = _atoms("He")

    hessian = calculator.get_hessian(atoms, delta=1.0e-3)

    np.testing.assert_allclose(
        hessian,
        np.eye(3) * EV2HARTREE,
        atol=1.0e-10,
    )


def test_numerical_hessian_fails_closed_for_nonconservative_delegate(
    tmp_path, monkeypatch
):
    delegate = _HarmonicDelegate(force_scale=0.5)
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
        delegate=delegate,
    )

    with pytest.raises(ValueError, match="force/energy consistency check failed"):
        calculator.get_hessian(_atoms("He"))


def test_delegate_is_resynchronized_after_common_hessian_restores_wrapper_state(
    tmp_path, monkeypatch
):
    delegate = _StatefulHarmonicDelegate()
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
        delegate=delegate,
    )
    atoms = _atoms("He")
    atoms.calc = calculator
    original_positions = atoms.positions.copy()

    atoms.get_potential_energy()
    calculator.get_hessian(atoms)
    forces = atoms.get_forces()

    np.testing.assert_allclose(forces, -original_positions * EV2HARTREE)


@pytest.mark.parametrize(
    ("model", "size", "message"),
    [
        ("forged", "small", "pinned to the 'fennix-bio1' model family"),
        ("fennix-bio1-small", "medium", "conflicts with size"),
        ("fennix-bio1", "large", "small' or 'medium"),
    ],
)
def test_rejects_identity_or_size_substitution(
    tmp_path, monkeypatch, model, size, message
):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.FENNIX_BIO1_CHECKPOINTS["small"]["sha256"],
    )
    with pytest.raises(ValueError, match=message):
        module.FeNNixBio1Calculator(
            "cpu",
            model=model,
            size=size,
            model_path=str(checkpoint),
            calculator_factory=lambda _path: _HarmonicDelegate(),
            device_context_factory=_ContextRecorder(),
        )


def test_rejects_missing_wrong_or_other_variant_checkpoint(tmp_path, monkeypatch):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    with pytest.raises(ValueError, match="explicit model_path"):
        module.FeNNixBio1Calculator("cpu")

    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.FeNNixBio1Calculator(
            "cpu",
            model_path=str(checkpoint),
            calculator_factory=lambda _path: _HarmonicDelegate(),
            device_context_factory=_ContextRecorder(),
        )

    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.FENNIX_BIO1_CHECKPOINTS["medium"]["sha256"],
    )
    with pytest.raises(ValueError, match="medium.*requested.*small"):
        module.FeNNixBio1Calculator(
            "cpu",
            model_path=str(checkpoint),
            calculator_factory=lambda _path: _HarmonicDelegate(),
            device_context_factory=_ContextRecorder(),
        )


@pytest.mark.parametrize(
    ("execution_task", "message"),
    [
        ("absolute_solvation_free_energy", "explicitly forbids"),
        ("not-a-maple-task", "Unsupported or unaudited"),
    ],
)
def test_direct_cpu_construction_validates_task_before_checkpoint_resolution(
    execution_task, message
):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    with pytest.raises(ValueError, match=message):
        module.FeNNixBio1Calculator(
            "cpu",
            execution_task=execution_task,
        )


def test_runtime_receipt_must_match_pinned_distribution_and_source_hashes(
    tmp_path, monkeypatch
):
    from maple.function.calculator.fennol import _fennix_bio1_calculator as module

    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        module,
        "_sha256",
        lambda _path: module.FENNIX_BIO1_CHECKPOINTS["small"]["sha256"],
    )
    factory_called = False

    def factory(_path):
        nonlocal factory_called
        factory_called = True
        return _HarmonicDelegate()

    bad_receipt = _runtime_receipt(module)
    bad_receipt["ase_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact pinned FeNNol runtime"):
        module.FeNNixBio1Calculator(
            "cpu",
            model_path=str(checkpoint),
            calculator_factory=factory,
            device_context_factory=_ContextRecorder(),
            runtime_validator=lambda: bad_receipt,
        )
    assert factory_called is False


def test_gpu_fails_before_checkpoint_or_runtime_resolution(tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    setter = SetCalculator(
        device="cuda",
        model="fennix-bio1",
        output=str(tmp_path / "maple.out"),
        model_options={"model_path": str(tmp_path / "does-not-exist.fnx")},
        task="sp",
    )
    with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
        setter.set_calculator()


def test_d4_request_fails_before_checkpoint_loading(tmp_path):
    from maple.function.calculator.set_calculator import SetCalculator

    checkpoint = _checkpoint(tmp_path)
    setter = SetCalculator(
        device="cpu",
        model="fennix-bio1",
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
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    atoms = _atoms("CO")
    atoms.info.clear()
    atoms.info.update(updates)

    with pytest.raises(ValueError, match=message):
        calculator.get_potential_energy(atoms)


def test_cached_property_cannot_bypass_mutated_charge_or_multiplicity(
    tmp_path, monkeypatch
):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    atoms = _atoms("CO")
    atoms.calc = calculator
    atoms.get_potential_energy()

    atoms.info["charge"] = 1
    with pytest.raises(ValueError, match="neutral singlets"):
        atoms.get_potential_energy()

    atoms.info.update(charge=0, mult=2)
    with pytest.raises(ValueError, match="neutral singlets"):
        atoms.get_potential_energy()


def test_neutral_hydrogen_is_rejected_as_inconsistent_singlet(tmp_path, monkeypatch):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(ValueError, match="even electron count"):
        calculator.get_potential_energy(_atoms("H"))


@pytest.mark.parametrize(
    ("initial_charges", "message"),
    [
        ([0.25, 0.0], "initial-charge sum must match"),
        ([float("nan"), 0.0], "finite initial-charge vector"),
    ],
)
def test_initial_charge_vector_must_be_finite_and_match_declared_charge(
    tmp_path, monkeypatch, initial_charges, message
):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    atoms = _atoms("CO")
    atoms.set_initial_charges(initial_charges)

    with pytest.raises(ValueError, match=message):
        calculator.get_potential_energy(atoms)


def test_rejects_unsupported_atomic_number_and_partial_periodicity(
    tmp_path, monkeypatch
):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )

    radium = Atoms(numbers=[88], positions=[[0.0, 0.0, 0.0]])
    radium.info.update(charge=0, mult=1)
    with pytest.raises(ValueError, match="atomic numbers 1 through 86"):
        calculator.get_potential_energy(radium)

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
        ("anisolv", "water"),
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
    delegate = _HarmonicDelegate(result=result)
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
        delegate=delegate,
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        calculator.calculate(_atoms("CO"), properties=properties)


def test_rejects_thermodynamic_claims_and_discovers_all_aliases(tmp_path, monkeypatch):
    _module, calculator, _delegate, _context, _captured = _calculator(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(ValueError, match="forbids task 'absolute_solvation"):
        calculator.validate_task("absolute_solvation_free_energy")
    with pytest.raises(ValueError, match="forbids task 'alchemical_free_energy"):
        calculator.validate_task("alchemical_free_energy")
    with pytest.raises(ValueError, match="forbids task 'stress"):
        calculator.validate_task("stress")
    with pytest.raises(ValueError, match="forbids task 'npt"):
        calculator.validate_task("npt")

    from maple.function.calculator.set_calculator import SetCalculator

    for requested in (
        "fennix-bio1",
        "fennixbio1",
        "fennix-bio1-small",
        "fennix-bio1-medium",
    ):
        setter = SetCalculator(
            device="cpu",
            model=requested,
            output=str(tmp_path / "maple.out"),
        )
        assert (
            setter._discover_calculator_class(requested).__name__
            == "FeNNixBio1Calculator"
        )
