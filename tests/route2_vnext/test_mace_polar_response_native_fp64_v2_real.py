"""Contracts for the bounded native-FP64 v2 direct-autograd oracle."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from ase import Atoms


def test_import_does_not_mutate_default_dtype():
    torch = pytest.importorskip("torch")
    original = torch.get_default_dtype()
    importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    assert torch.get_default_dtype() is original


def test_oracle_rejects_cases_outside_water_methane_cap():
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    with pytest.raises(ValueError, match="water and methane"):
        module.validate_oracle_case(("C", "H", "H", "H", "H", "H"))
    with pytest.raises(ValueError, match="water and methane"):
        module.validate_oracle_case(("O", "H"))


def test_oracle_case_accepts_ordered_water_and_methane_only():
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    assert module.validate_oracle_case(("O", "H", "H")) == "water"
    assert module.validate_oracle_case(("C", "H", "H", "H", "H")) == "methane"


class _Model:
    device = "cpu"
    dtype = pytest.importorskip("torch").float64

    def configuration_sha256(self):
        return "a" * 64

    def energy_source_torch(self, atoms, positions):
        del atoms
        torch = pytest.importorskip("torch")
        q = positions[:, 0].square()
        q = q - q.mean()
        learned = torch.stack(
            (q, positions[:, 1], positions[:, 2], positions.sum(1)), 1
        )
        return 0.1 * positions.square().sum(), learned


class _DriftingModel(_Model):
    def __init__(self):
        self.digest = "a" * 64

    def configuration_sha256(self):
        return self.digest

    def energy_source_torch(self, atoms, positions):
        result = super().energy_source_torch(atoms, positions)
        self.digest = "d" * 64
        return result


class _Continuum:
    device = "cpu"

    def configuration_sha256(self):
        return "b" * 64

    def energy_torch(self, positions, source):
        return 0.02 * positions.square().sum() + 0.03 * source.square().sum()


class _CDS:
    device = "cpu"

    def configuration_sha256(self):
        return "c" * 64

    def energy_torch(self, positions):
        return 0.01 * positions.pow(3).sum()


def test_synthetic_oracle_is_direct_autograd_and_hvp_matches_full_hessian():
    torch = pytest.importorskip("torch")
    original_dtype = torch.get_default_dtype()
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    pes = module.MACEPolarNativeFP64DirectOracle._for_testing(
        model=_Model(),
        continuum=_Continuum(),
        solvent_term=_CDS(),
        symbols=("O", "H", "H"),
        solvent="water",
        device="cpu",
    )
    atoms = Atoms("OH2", positions=[[0.1, 0.2, 0.3], [0.9, 0.0, 0.0], [-0.2, 0.8, 0.0]])
    atoms.info.update(charge=0, mult=1)
    direction = np.arange(9, dtype=float).reshape(3, 3) / 10
    result = pes.evaluate_hessian(atoms)
    hvp = pes.hessian_vector_product(atoms, direction)
    np.testing.assert_allclose(
        hvp.ravel(), result.hessian_eV_per_A2 @ direction.ravel()
    )
    assert result.diagnostics["implementation"] == "direct-autograd"
    assert result.diagnostics["structured_response"] is False
    assert result.provider_id.startswith("unregistered-engineering-test-")
    assert result.scalar_contract_id.startswith("unregistered-engineering-test-")
    assert torch.get_default_dtype() is original_dtype


def test_testing_oracle_uses_explicit_engineering_identities():
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    pes = module.MACEPolarNativeFP64DirectOracle._for_testing(
        model=_Model(),
        continuum=_Continuum(),
        solvent_term=_CDS(),
        symbols=("O", "H", "H"),
        solvent="water",
        device="cpu",
    )
    assert pes.provider_id.startswith("unregistered-engineering-test-")
    assert pes.profile_id.startswith("unregistered-engineering-test-")
    assert pes.scalar_contract_id.startswith("unregistered-engineering-test-")


def test_cuda_preparation_precedes_continuum_allocation(monkeypatch):
    torch = pytest.importorskip("torch")
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    import maple.solvation.experimental.cuda_execution as cuda_execution
    import maple.solvation.continuum.torch_ddpcm as torch_ddpcm

    events = []
    monkeypatch.setattr(
        cuda_execution, "prepare_cuda_execution", lambda: events.append("prepare")
    )

    class StopAfterOrderCheck(Exception):
        pass

    def continuum(*args, **kwargs):
        assert events == ["prepare"]
        raise StopAfterOrderCheck

    monkeypatch.setattr(torch_ddpcm, "TorchDDPCM", continuum)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(StopAfterOrderCheck):
            module.build_smd_mace_polar_native_fp64_v2_oracle(
                ("O", "H", "H"), solvent="water", device="cuda:0"
            )
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize(
    "method", ["get_potential_energy", "get_forces", "hessian_vector_product"]
)
def test_oracle_rechecks_configuration_after_evaluation(method):
    module = importlib.import_module(
        "maple.solvation.experimental.mace_polar_native_fp64_oracle"
    )
    pes = module.MACEPolarNativeFP64DirectOracle._for_testing(
        model=_DriftingModel(),
        continuum=_Continuum(),
        solvent_term=_CDS(),
        symbols=("O", "H", "H"),
        solvent="water",
        device="cpu",
    )
    atoms = Atoms("OH2", positions=[[0.1, 0.2, 0.3], [0.9, 0.0, 0.0], [-0.2, 0.8, 0.0]])
    atoms.info.update(charge=0, mult=1)
    arguments = (atoms,)
    if method == "hessian_vector_product":
        arguments += (np.ones((3, 3)),)
    with pytest.raises(RuntimeError, match="configuration"):
        getattr(pes, method)(*arguments)
