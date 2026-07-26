from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.mace import _mace_upstream_calculator as backend


class _FakeUpstream:
    def __init__(self, calls):
        self.calls = calls
        self.results = {}
        self.r_max = 6.0

    def calculate(self, atoms, properties, system_changes):
        self.calls.append(
            {
                "atoms": atoms.copy(),
                "properties": tuple(properties),
                "system_changes": tuple(system_changes),
            }
        )
        self.results = {
            "energy": -12.5,
            "free_energy": -12.5,
            "forces": np.full((len(atoms), 3), 0.25),
            "stress": np.arange(6, dtype=float),
        }


def _checkpoint(tmp_path):
    path = tmp_path / "model.model"
    path.write_bytes(b"pinned checkpoint")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_is_checked_before_the_mace_runtime_is_imported(
    tmp_path, monkeypatch
):
    checkpoint, _ = _checkpoint(tmp_path)

    def forbidden_import(**kwargs):
        raise AssertionError("runtime import occurred before hash validation")

    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        forbidden_import,
    )
    with pytest.raises(ValueError, match="MODEL_HASH_MISMATCH"):
        backend.MACEOff24Provider(
            device="cpu",
            model="maceoff24m",
            checkpoint=str(checkpoint),
            sha256="0" * 64,
            license_ack=True,
        )


def test_model_cutoff_must_come_from_loaded_upstream_model(
    tmp_path,
    monkeypatch,
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    upstream = _FakeUpstream([])
    upstream.r_max = float("nan")
    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        lambda **kwargs: upstream,
    )

    with pytest.raises(RuntimeError, match="interaction cutoff"):
        backend.MACEOff24Provider(
            device="cpu",
            model="maceoff24m",
            checkpoint=str(checkpoint),
            sha256=sha256,
            license_ack=True,
        )


def test_off24_forwards_periodic_cell_energy_forces_and_stress(
    tmp_path, monkeypatch
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    calls = []
    construction = {}

    def fake_construct(**kwargs):
        construction.update(kwargs)
        return _FakeUpstream(calls)

    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        fake_construct,
    )
    calculator = backend.MACEOff24Provider(
        device="cpu",
        model="maceoff24m",
        checkpoint=str(checkpoint),
        sha256=sha256,
        license_ack=True,
    )
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[12, 12, 12],
        pbc=True,
    )
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])

    assert construction["kind"] == "off24"
    assert construction["checkpoint"] == checkpoint.resolve()
    assert calls[0]["atoms"].get_pbc().all()
    assert np.array_equal(calls[0]["atoms"].cell.array, atoms.cell.array)
    assert calculator.results["energy"] == -12.5
    assert calculator.results["forces"].shape == (3, 3)
    assert calculator.results["stress"].shape == (6,)
    assert calculator.provenance["interaction_cutoff_angstrom"] == 6.0


def test_off24_retains_coevaluated_results_for_ase_cache(
    tmp_path,
    monkeypatch,
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    calls = []
    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        lambda **kwargs: _FakeUpstream(calls),
    )
    calculator = backend.MACEOff24Provider(
        device="cpu",
        model="maceoff24m",
        checkpoint=str(checkpoint),
        sha256=sha256,
        license_ack=True,
    )
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[12, 12, 12],
        pbc=True,
        calculator=calculator,
    )

    assert atoms.get_potential_energy() == pytest.approx(-12.5)
    assert atoms.get_forces().shape == (3, 3)
    assert atoms.get_stress().shape == (6,)
    assert len(calls) == 1


def test_omol_maps_closed_shell_multiplicity_to_spin_one(
    tmp_path, monkeypatch
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    calls = []
    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        lambda **kwargs: _FakeUpstream(calls),
    )
    calculator = backend.MACEOMOLProvider(
        device="cpu",
        model="maceomol",
        checkpoint=str(checkpoint),
        sha256=sha256,
        license_ack=True,
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.info.update(charge=0, mult=1)
    calculator.calculate(atoms, properties=["energy", "forces"])

    evaluated = calls[0]["atoms"]
    assert evaluated.info["charge"] == 0.0
    assert evaluated.info["spin"] == 1.0
    assert "stress" not in calculator.results


def test_omol_rejects_conflicting_spin_zero_before_evaluation(
    tmp_path, monkeypatch
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    calls = []
    monkeypatch.setattr(
        backend,
        "_construct_upstream_calculator",
        lambda **kwargs: _FakeUpstream(calls),
    )
    calculator = backend.MACEOMOLProvider(
        device="cpu",
        model="maceomol",
        checkpoint=str(checkpoint),
        sha256=sha256,
        license_ack=True,
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.info.update(charge=0, mult=1, spin=0)

    with pytest.raises(ValueError, match="spin=1"):
        calculator.calculate(atoms, properties=["energy"])
    assert calls == []


def test_abi_failure_is_typed_and_does_not_mutate_environment(
    tmp_path, monkeypatch
):
    checkpoint, sha256 = _checkpoint(tmp_path)
    original = os.environ.get("LD_LIBRARY_PATH")

    def fail_import():
        raise ImportError("CXXABI_1.3.15 not found")

    monkeypatch.setattr(backend, "_import_mace_calculator", fail_import)
    with pytest.raises(RuntimeError, match="RUNTIME_ABI_INCOMPATIBLE"):
        backend.MACEOff24Provider(
            device="cpu",
            model="maceoff24m",
            checkpoint=str(checkpoint),
            sha256=sha256,
            license_ack=True,
        )
    assert os.environ.get("LD_LIBRARY_PATH") == original
