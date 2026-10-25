from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import anysolv


class FakeRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def evaluate(
        self, numbers, positions, *, charge, multiplicity, solvent, scalar_only=False
    ):
        self.calls.append(
            (numbers.copy(), positions.copy(), charge, multiplicity, solvent)
        )
        return -0.25, np.ones_like(positions) * 999.0


class NeutralOnlyBase:
    SUPPORTS_CHARGE_MULT = False


class ElectronicStateBase:
    SUPPORTS_CHARGE_MULT = True


@pytest.fixture
def backend(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_moe.pt"
    checkpoint.write_bytes(b"model")
    source = tmp_path / "anysolv"
    source.mkdir()
    monkeypatch.setattr(
        anysolv,
        "_sha256",
        lambda path: anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
    )
    monkeypatch.setattr(
        anysolv,
        "_validate_source",
        lambda root: {
            "source_revision": anysolv.ANYSOLV_SOURCE_REVISION,
            "source_root": str(root),
        },
    )
    monkeypatch.setattr(anysolv, "_OfficialAnySolvRuntime", FakeRuntime)
    return anysolv.AnySolvBackend(
        dtype="float64",
        solvent="water",
        model_path=str(checkpoint),
        source_root=str(source),
        device="cpu",
        execution_task="sp",
    )


def water():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]])
    atoms.info.update(charge=0, mult=1)
    return atoms


def test_energy_only_result_has_geometry_level_quantity(backend):
    result = backend.evaluate(water())
    assert result.energy_hartree == pytest.approx(-0.25 / anysolv.EV_PER_HARTREE)
    assert result.forces_hartree_per_angstrom is None
    assert result.quantity_kind == "geometry_level_energy_correction"
    assert "delta_g" not in " ".join(result.components_hartree)
    assert result.provenance["forces_exposed"] is False
    assert result.provenance["scientific_status"] == "experimental_single_point_only"
    assert result.provenance["validated_multifragment_domain"] == "unknown"


def test_force_request_fails_before_runtime(backend):
    with pytest.raises(NotImplementedError, match="forces are disabled"):
        backend.evaluate(water(), need_forces=True)
    assert backend._runtime.calls == []


def test_invalid_state_and_pbc_fail_closed(backend):
    atoms = water()
    del atoms.info["mult"]
    with pytest.raises(ValueError, match="explicit"):
        backend.evaluate(atoms)
    atoms = water()
    atoms.pbc = True
    with pytest.raises(NotImplementedError, match="PBC"):
        backend.evaluate(atoms)


def test_only_frozen_trained_solvents_are_accepted(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_moe.pt"
    checkpoint.write_bytes(b"model")
    source = tmp_path / "anysolv"
    source.mkdir()
    monkeypatch.setattr(
        anysolv, "_sha256", lambda _: anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(anysolv, "_validate_source", lambda root: {})
    monkeypatch.setattr(anysolv, "_OfficialAnySolvRuntime", FakeRuntime)
    with pytest.raises(ValueError, match="36-solvent"):
        anysolv.AnySolvBackend(
            device="cpu",
            dtype="float64",
            solvent="untrained-solvent",
            model_path=str(checkpoint),
            source_root=str(source),
        )


def test_checkpoint_mismatch_fails_before_runtime(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_moe.pt"
    checkpoint.write_bytes(b"wrong")
    source = tmp_path / "anysolv"
    source.mkdir()
    monkeypatch.setattr(
        anysolv, "_validate_source", lambda root: pytest.fail("source touched")
    )
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        anysolv.AnySolvBackend(
            device="cpu",
            dtype="float64",
            solvent="water",
            model_path=str(checkpoint),
            source_root=str(source),
        )


def test_default_checkpoint_uses_maple_model_storage_contract():
    assert anysolv.ANYSOLV_MOE_DEFAULT_PATH.parts[-3:] == (
        "model",
        "anysolv",
        "model_moe.pt",
    )


@pytest.mark.parametrize("source_root", [None, "", "   ", 7, False])
def test_invalid_source_path_rejected_before_checkpoint_access(tmp_path, source_root):
    with pytest.raises((TypeError, ValueError), match="source_root"):
        anysolv.AnySolvBackend(
            device="cpu",
            dtype="float64",
            solvent="water",
            model_path=tmp_path / "absent.pt",
            source_root=source_root,
        )


@pytest.mark.skipif(
    os.environ.get("MAPLE_TEST_ANYSOLV_MOE") != "1",
    reason=(
        "requires the authorized 1.1 GB AnySolv-MoE checkpoint; set "
        "MAPLE_TEST_ANYSOLV_MOE=1 for the explicit integration test"
    ),
)
def test_authorized_checkpoint_matches_frozen_sha256():
    assert (
        anysolv.ANYSOLV_MOE_DEFAULT_PATH.is_file()
    ), "MAPLE_TEST_ANYSOLV_MOE=1 requires model/anysolv/model_moe.pt"
    assert (
        anysolv._sha256(anysolv.ANYSOLV_MOE_DEFAULT_PATH)
        == anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256
    )


def test_configuration_is_constructor_immutable(backend):
    with pytest.raises(AttributeError):
        backend.solvent = "toluene"
    with pytest.raises(AttributeError):
        backend.model_path = "/tmp/other.pt"
    with pytest.raises(AttributeError):
        backend.source_root = "/tmp/other-source"


def test_empty_and_out_of_capacity_atomic_numbers_fail(backend):
    empty = Atoms()
    empty.info.update(charge=0, mult=1)
    with pytest.raises(ValueError, match="at least one atom"):
        backend.evaluate(empty)

    outside = Atoms(numbers=[100], positions=[[0.0, 0.0, 0.0]])
    outside.info.update(charge=0, mult=1)
    with pytest.raises(ValueError, match="capacity 1..99"):
        backend.evaluate(outside)


def test_each_frame_rechecks_state_against_neutral_only_base(backend):
    atoms = water()
    backend.evaluate(atoms, calculator=NeutralOnlyBase())
    assert len(backend._runtime.calls) == 1

    atoms.info["charge"] = 1
    with pytest.raises(ValueError, match="base calculator"):
        backend.evaluate(atoms, calculator=NeutralOnlyBase())
    assert len(backend._runtime.calls) == 1

    atoms.info.update(charge=0, mult=2)
    with pytest.raises(ValueError, match="base calculator"):
        backend.evaluate(atoms, calculator=NeutralOnlyBase())
    assert len(backend._runtime.calls) == 1


def test_mixed_frames_cannot_bypass_first_frame_state_check(backend):
    neutral = water()
    charged = water()
    charged.info["charge"] = -1

    backend.evaluate(neutral, calculator=NeutralOnlyBase())
    with pytest.raises(ValueError, match="charge/multiplicity"):
        backend.evaluate(charged, calculator=NeutralOnlyBase())
    assert len(backend._runtime.calls) == 1


def test_charge_capable_base_allows_explicit_state_to_reach_anysolv(backend):
    atoms = water()
    atoms.info.update(charge=1, mult=2)
    backend.evaluate(atoms, calculator=ElectronicStateBase())
    assert backend._runtime.calls[-1][2:4] == (1, 2)


def test_direct_delta_diagnostic_accepts_explicit_state_without_base(backend):
    atoms = water()
    atoms.info.update(charge=-1, mult=2)
    backend.evaluate(atoms, calculator=None)
    assert backend._runtime.calls[-1][2:4] == (-1, 2)


@pytest.mark.parametrize(
    "supports_charge,supports_mult",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("charge,mult", [(0, 1), (1, 1), (0, 2), (1, 2)])
def test_composite_uses_base_charge_and_multiplicity_capabilities_independently(
    backend, supports_charge, supports_mult, charge, mult
):
    from maple.function.calculator.backends import CompositeCalculator, PotentialResult
    from maple.function.calculator.model_capabilities import ModelCapabilities

    class Base:
        capabilities = ModelCapabilities(
            energy=True,
            supports_charge=supports_charge,
            supports_multiplicity=supports_mult,
        )

        def evaluate(self, *args, **kwargs):
            return PotentialResult(energy_hartree=1.0)

    atoms = water()
    atoms.info.update(charge=charge, mult=mult)
    calculator = CompositeCalculator(Base(), backend)
    # The wrapper flag must not override the authoritative potential capabilities.
    calculator.SUPPORTS_CHARGE_MULT = True
    atoms.calc = calculator
    allowed = (charge == 0 or supports_charge) and (mult == 1 or supports_mult)
    if allowed:
        assert np.isfinite(atoms.get_potential_energy())
        assert backend._runtime.calls[-1][2:4] == (charge, mult)
    else:
        with pytest.raises(ValueError, match="base calculator"):
            atoms.get_potential_energy()
        assert calculator.results == {}
        assert backend._runtime.calls == []


@pytest.mark.parametrize("state", [{"charge": 1}, {"mult": 2}])
def test_composite_state_mutation_rechecks_real_backend_before_runtime(backend, state):
    from maple.function.calculator.backends import CompositeCalculator, PotentialResult
    from maple.function.calculator.model_capabilities import ModelCapabilities

    class Base:
        capabilities = ModelCapabilities(energy=True)

        def evaluate(self, *args, **kwargs):
            return PotentialResult(energy_hartree=1.0)

    atoms = water()
    atoms.calc = CompositeCalculator(Base(), backend)
    atoms.get_potential_energy()
    assert len(backend._runtime.calls) == 1
    atoms.info.update(state)
    with pytest.raises(ValueError, match="base calculator"):
        atoms.get_potential_energy()
    assert atoms.calc.results == {}
    assert len(backend._runtime.calls) == 1


@pytest.mark.parametrize("mode", ["native", "alchemical", "property_only"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_direct_evaluation_rejects_non_vacuum_context_before_runtime(
    backend, mode, wrapped
):
    from types import SimpleNamespace
    from maple.function.calculator.backends import CompositeCalculator
    from maple.function.calculator.model_capabilities import ModelCapabilities

    base = SimpleNamespace(
        capabilities=ModelCapabilities(energy=True, solvation_mode=mode)
    )
    calculator = CompositeCalculator(base, backend) if wrapped else base
    with pytest.raises(ValueError, match="vacuum"):
        backend.evaluate(water(), calculator=calculator)
    assert backend._runtime.calls == []


@pytest.mark.parametrize(
    ("charge", "mult", "message"),
    [
        (-101, 1, "charge embedding capacity"),
        (101, 1, "charge embedding capacity"),
        (0, 101, "multiplicity embedding capacity"),
    ],
)
def test_electronic_state_embedding_capacity_is_checked(backend, charge, mult, message):
    atoms = water()
    atoms.info.update(charge=charge, mult=mult)
    with pytest.raises(ValueError, match=message):
        backend.evaluate(atoms)
    assert backend._runtime.calls == []


@pytest.mark.parametrize(
    ("model_path", "source_root", "message"),
    [
        ("", "/source", "model_path cannot be empty"),
        (None, "/source", "model_path must be"),
        (123, "/source", "model_path must be"),
        ("/checkpoint", "", "source_root cannot be empty"),
        ("/checkpoint", 123, "source_root must be"),
    ],
)
def test_explicit_paths_never_fall_back(
    backend, monkeypatch, model_path, source_root, message
):
    monkeypatch.setattr(
        anysolv, "_sha256", lambda _: anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    with pytest.raises((TypeError, ValueError), match=message):
        anysolv.AnySolvBackend(
            device="cpu",
            dtype="float64",
            solvent="water",
            model_path=model_path,
            source_root=source_root,
        )


def test_official_runtime_uses_explicit_package_spec_without_sys_path_pollution(
    tmp_path,
):
    source = tmp_path / "anysolv"
    source.mkdir()
    (source / "__init__.py").write_text(
        "class Model: pass\n"
        "class InferenceSettings:\n"
        "    def __repr__(self): return 'reference'\n"
        "def load_model(*args, **kwargs): return Model()\n"
    )
    (source / "data.py").write_text("def build_atomic_data(*args, **kwargs): pass\n")
    (source / "solvent.py").write_text(
        "def get_solvent_vector(*args, **kwargs): pass\n"
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    parent = str(source.parent)
    before = list(sys.path)
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "anysolv" or name.startswith("anysolv.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        runtime = anysolv._OfficialAnySolvRuntime(
            dtype="float64",
            source_root=source,
            checkpoint=checkpoint,
            device="cpu",
            checkpoint_sha256=anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
        )
        assert runtime._resident.model is not None
        assert sys.path == before
        assert parent not in sys.path or parent in before
    finally:
        for name in tuple(sys.modules):
            if name == "anysolv" or name.startswith("anysolv."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_disconnected_multifragment_input_is_mechanically_accepted(backend):
    atoms = water() + water()
    atoms.positions[3:] += [8.0, 0.0, 0.0]
    atoms.info.update(charge=0, mult=1)
    result = backend.evaluate(atoms)
    assert np.isfinite(result.energy_hartree)
    assert backend._runtime.calls[-1][0].shape == (6,)


@pytest.mark.parametrize("state", [{"charge": 1}, {"mult": 2}])
def test_calcabc_state_change_revalidates_base_before_runtime(backend, state):
    from maple.function.calculator.calculator_base import CalcABC

    class NeutralBase(CalcABC):
        SUPPORTS_CHARGE_MULT = False
        MODEL_ENERGY_UNIT = "hartree"
        implemented_properties = ["energy", "free_energy"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            atoms = super().calculate(atoms, properties, system_changes)
            self._finalize_results(atoms, energy=1.0)

    atoms = water()
    calculator = NeutralBase()
    calculator.solvent_correction = backend
    atoms.calc = calculator

    assert np.isfinite(atoms.get_potential_energy())
    assert len(backend._runtime.calls) == 1

    atoms.info.update(state)
    with pytest.raises(ValueError, match="base calculator"):
        atoms.get_potential_energy()
    assert calculator.results == {}
    assert len(backend._runtime.calls) == 1


@pytest.mark.parametrize("charge,mult", [(-100, 1), (100, 100)])
def test_electronic_state_embedding_boundaries_reach_runtime(backend, charge, mult):
    atoms = water()
    atoms.info.update(charge=charge, mult=mult)
    backend.evaluate(atoms)
    assert backend._runtime.calls[-1][2:4] == (charge, mult)


@pytest.mark.parametrize("module_file", [None, "/unvalidated/anysolv/__init__.py"])
def test_official_runtime_rejects_preimported_package_without_pinned_location(
    tmp_path, monkeypatch, module_file
):
    package = ModuleType("anysolv")
    if module_file is not None:
        package.__file__ = module_file
    monkeypatch.setitem(sys.modules, "anysolv", package)
    before = list(sys.path)
    with pytest.raises(ImportError, match="filesystem location|expected pinned source"):
        anysolv._OfficialAnySolvRuntime(
            dtype="float64",
            source_root=tmp_path,
            checkpoint=tmp_path / "model.pt",
            device="cpu",
            checkpoint_sha256=anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
        )
    assert sys.path == before


def test_failed_source_import_leaves_no_modules_or_path_residue(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "anysolv", raising=False)
    (tmp_path / "__init__.py").write_text(
        "from . import partial\nraise RuntimeError('source failed')\n"
    )
    (tmp_path / "partial.py").write_text("sentinel = 1\n")
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="source failed"):
        anysolv._OfficialAnySolvRuntime(
            dtype="float64",
            source_root=tmp_path,
            checkpoint=tmp_path / "model.pt",
            device="cpu",
            checkpoint_sha256=anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
        )
    assert sys.path == before
    assert not any(
        name == "anysolv" or name.startswith("anysolv.") for name in sys.modules
    )


def test_force_rejection_model_card_binds_exact_frozen_result():
    root = Path(__file__).resolve().parents[2]
    card = json.loads(
        (anysolv.ANYSOLV_MODEL_CARD_ROOT / "anysolv-moe.yaml").read_text()
    )
    evidence = card["force_consistency_evidence"]
    artifact = root / evidence["artifact"]
    assert artifact.is_file()
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest() == evidence["artifact_sha256"]
    )
    assert evidence["status"] == "rejected_orientation_dependent_upstream_forces"
    assert card["capabilities"]["forces"] is False
    assert card["capabilities"]["supports_multifragment"] is True
    mechanical = card["multifragment_mechanics_evidence"]
    artifact = root / mechanical["artifact"]
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        == mechanical["artifact_sha256"]
    )
    frozen = json.loads(artifact.read_text())
    assert frozen["checkpoint_sha256"] == card["checkpoint_sha256"]
    assert frozen["source_revision"] == card["source"]["revision"]
    assert (
        frozen["minimum_cross_fragment_distance_angstrom"]
        > frozen["upstream_graph_cutoff_angstrom"]
    )
    assert frozen["added_bonds"] is False
    assert frozen["validated_multifragment_domain"] == "unknown"


def test_execution_policy_provenance_and_immutability(backend):
    assert backend.device == "cpu"
    assert backend.dtype == "float64"
    assert backend.inference == "reference"
    assert backend._runtime.kwargs["dtype"] == "float64"
    assert backend.provenance["compute_dtype"] == "float64"
    assert backend.provenance["source_digest"] == anysolv._SOURCE_DIGEST
    for name, value in (
        ("device", "cuda:1"),
        ("dtype", "float32"),
        ("inference", "optimized"),
    ):
        with pytest.raises(AttributeError):
            setattr(backend, name, value)


def test_default_cuda32_policy_validates_card_before_loading(monkeypatch):
    import torch
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        anysolv.SolvationExecution, "runtime_context", lambda self: {"live": True}
    )
    calls = []

    def blocked(device, **kwargs):
        calls.append((device, kwargs))
        raise ValueError("missing CUDA32 diagnostic evidence")

    card = SimpleNamespace(validate_device=blocked)
    monkeypatch.setattr(anysolv, "load_model_provenance_card", lambda *args: card)
    monkeypatch.setattr(
        anysolv,
        "_OfficialAnySolvRuntime",
        lambda **kwargs: pytest.fail("loader reached"),
    )
    with pytest.raises(ValueError, match="missing CUDA32 diagnostic evidence"):
        anysolv.AnySolvBackend(solvent="water")
    assert calls == [
        (
            "cuda:0",
            {
                "task": "sp",
                "inference_mode": "reference",
                "runtime_context": {"live": True},
            },
        )
    ]
    assert anysolv.AnySolvBackend.__dataclass_fields__["dtype"].default == "float32"


def test_default_unavailable_cuda_has_no_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        anysolv,
        "_OfficialAnySolvRuntime",
        lambda **kwargs: pytest.fail("loader reached"),
    )
    with pytest.raises(ValueError, match="unavailable.*fallback is disabled"):
        anysolv.AnySolvBackend(solvent="water")


@pytest.mark.parametrize("device", [None, "none"])
def test_missing_literal_device_rejected_before_runtime_loading(monkeypatch, device):
    monkeypatch.setattr(anysolv, "_OfficialAnySolvRuntime", lambda **kwargs: pytest.fail("loader reached"))
    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        anysolv.AnySolvBackend(solvent="water", device=device)
