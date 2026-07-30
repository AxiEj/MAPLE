from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.backends import (
    CompositeCalculator,
    PotentialBackend,
    PotentialResult,
)
from maple.function.calculator.extra_correction.implicit import (
    anisolv as anisolv_module,
)
from maple.function.calculator.extra_correction.implicit.anisolv import (
    ANISOLV_COMPACT_CHECKPOINT_SHA256,
    AniSolvCompactBackend,
    AniSolvCompactResult,
    SUPPORTED_ANISOLV_SOLVENTS,
    resolve_anisolv_solvent,
)
from maple.function.calculator.model_capabilities import ModelCapabilities
from maple.function.calculator.model_capabilities import load_model_provenance_card


class RecordingRuntime:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def evaluate(self, atomic_numbers, positions_angstrom, **kwargs):
        self.calls.append((atomic_numbers.copy(), positions_angstrom.copy(), kwargs))
        return AniSolvCompactResult(
            energy_ev=-2.0,
            forces_ev_per_angstrom=np.full((len(atomic_numbers), 3), 0.5),
            provenance={"runtime": "recording"},
        )


class ZeroPotential(PotentialBackend):
    capabilities = ModelCapabilities(
        energy=True,
        forces=True,
        conservative_forces=True,
        hessian="finite_difference",
        energy_reference="absolute",
    )

    def evaluate(self, atoms, *, need_forces=False, need_hessian=False):
        return PotentialResult(
            energy_hartree=0.0,
            forces_hartree_per_angstrom=(
                np.zeros((len(atoms), 3), dtype=float) if need_forces else None
            ),
            provenance={"provider": "zero"},
        )


@pytest.fixture
def water():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


@pytest.fixture
def compact_model(tmp_path):
    path = tmp_path / "model1_compact.pt"
    path.write_bytes(b"test checkpoint")
    return path


def _backend(monkeypatch, compact_model, solvent="thf"):
    monkeypatch.setattr(
        anisolv_module,
        "_sha256",
        lambda _path: ANISOLV_COMPACT_CHECKPOINT_SHA256,
    )
    created = []

    def factory(**kwargs):
        runtime = RecordingRuntime(**kwargs)
        created.append(runtime)
        return runtime

    backend = AniSolvCompactBackend(
        model_path=str(compact_model),
        solvent=solvent,
        device="cpu",
        runtime_factory=factory,
    )
    return backend, created[0]


def test_resolver_whitelists_only_21_explicitly_validated_solvents():
    assert len(SUPPORTED_ANISOLV_SOLVENTS) == 21
    assert resolve_anisolv_solvent("THF").upstream_name == "tetrahydrofuran"
    assert resolve_anisolv_solvent("DMSO").upstream_name == "dimethyl sulfoxide (DMSO)"
    with pytest.raises(ValueError, match="only the 21 solvents"):
        resolve_anisolv_solvent("pyridine")


def test_backend_requires_exact_public_checkpoint_identity(monkeypatch, compact_model):
    monkeypatch.setattr(anisolv_module, "_sha256", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        AniSolvCompactBackend(model_path=str(compact_model), solvent="water")

    with pytest.raises(ValueError, match="pinned to the official checkpoint"):
        AniSolvCompactBackend(
            model_path=str(compact_model),
            solvent="water",
            checkpoint_sha256="0" * 64,
        )


def test_backend_converts_upstream_energy_and_preserves_explicit_state(
    monkeypatch, compact_model, water
):
    backend, runtime = _backend(monkeypatch, compact_model)

    result = backend.evaluate(water)

    assert result.energy_hartree == pytest.approx(-2.0 / anisolv_module.EV_PER_HARTREE)
    assert result.forces_hartree_per_angstrom is None
    assert result.components_hartree["anisolv_compact_delta"] == result.energy_hartree
    assert result.provenance["solvent"] == "tetrahydrofuran"
    assert result.provenance["supports_absolute_solvation"] is False
    assert result.provenance["charge"] == 0
    assert result.provenance["multiplicity"] == 1
    assert result.provenance["runtime"] == "recording"
    assert runtime.init_kwargs["device"] == "cpu"
    assert runtime.calls[0][2]["solvent"].upstream_name == "tetrahydrofuran"
    assert runtime.calls[0][2]["need_forces"] is False


def test_backend_is_legal_for_energy_composition_but_rejects_forces(
    monkeypatch, compact_model, water
):
    backend, _ = _backend(monkeypatch, compact_model, solvent="methanol")
    calc = CompositeCalculator(ZeroPotential(), backend)

    assert calc.get_potential_energy(water) == pytest.approx(
        -2.0 / anisolv_module.EV_PER_HARTREE
    )
    with pytest.raises(NotImplementedError, match="not rotation-covariant"):
        calc.get_forces(water)
    with pytest.raises(NotImplementedError, match="Only scalar single-point"):
        backend.evaluate(water, need_forces=True)
    assert backend.capabilities.supports_absolute_solvation is False


def test_backend_rejects_runtime_provenance_that_overrides_pinned_identity(
    monkeypatch, compact_model, water
):
    backend, _ = _backend(monkeypatch, compact_model)

    class ForgedRuntime:
        def evaluate(self, *_args, **_kwargs):
            return AniSolvCompactResult(
                energy_ev=-2.0,
                provenance={
                    "checkpoint_sha256": "forged",
                    "supports_absolute_solvation": True,
                },
            )

    backend._runtime = ForgedRuntime()
    with pytest.raises(ValueError, match="reserved provenance"):
        backend.evaluate(water)


def test_backend_rejects_unaudited_electronic_state_and_pbc(
    monkeypatch, compact_model, water
):
    backend, _ = _backend(monkeypatch, compact_model)
    del water.info["mult"]
    with pytest.raises(ValueError, match="explicit atoms.info"):
        backend.evaluate(water)

    water.info["mult"] = 1
    water.info["charge"] = 1
    with pytest.raises(ValueError, match="neutral singlet"):
        backend.evaluate(water)

    water.info["charge"] = 0
    water.info["mult"] = 2
    with pytest.raises(ValueError, match="neutral singlet"):
        backend.evaluate(water)

    water.info["mult"] = 1
    water.pbc = True
    with pytest.raises(NotImplementedError, match="non-periodic"):
        backend.evaluate(water)


def test_model_card_keeps_the_energy_only_and_free_energy_boundary():
    card = load_model_provenance_card(
        "anisolv-compact",
        model_card_root=Path(__file__).parents[2]
        / "maple/function/calculator/model_cards",
    )

    assert card.capabilities.solvation_mode == "additive"
    assert card.capabilities.forces is False
    assert card.capabilities.conservative_forces is False
    assert card.capabilities.supports_charge is False
    assert card.capabilities.supports_multiplicity is False
    assert card.capabilities.supports_absolute_solvation is False
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("absolute_solvation_free_energy")
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("opt")
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("frequency")
