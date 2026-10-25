"""One-graph energy/force handoff across the Torch PES and ASE facade."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.function.calculator.route2 import PureMACEPolarTorchCalculator
from maple.function.route2_smd_profiles import route2_smd_profile_spec
from maple.solvation.derivatives.analytic import AnalyticForceEvaluation
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.experimental.mace_polar_torch import MACEPolarTorchPES
from maple.solvation.models.base import model_input_sha256

CPU_V3 = "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v3"
CPU_V4 = "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4"
V3_EVIDENCE = "route2-pure-macepolar-torch-analytic-v3-contract-tests"
V4_EVIDENCE = "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
PROVIDER_ID = "maple.pure-macepolar-frozen-ddpcm-legacy-cds.torch-v3"


def _water(*, charge=0, mult=1):
    return Atoms(
        "OH2",
        positions=[[0.1, 0.2, 0.1], [1.05, 0.2, 0.1], [-0.14, 1.12, 0.1]],
        info={"charge": charge, "mult": mult},
    )


class _CountingModel:
    device = "cpu"

    def __init__(self, dtype):
        self.dtype = dtype
        self.calls = 0
        self.requires_grad = []

    def configuration_sha256(self):
        return "a" * 64

    def energy_source_torch(self, atoms, positions):
        self.calls += 1
        self.requires_grad.append(positions.requires_grad)
        x, y, z = positions.unbind(-1)
        source = torch.stack((x * 0, y**2, z**2, x**2), dim=-1)
        return 0.07 * positions.square().sum(), source


class _CountingContinuum:
    def __init__(self):
        self.calls = 0

    def configuration_sha256(self):
        return "b" * 64

    def energy_torch(self, positions, source):
        self.calls += 1
        cartesian_dipoles = source[:, [4, 2, 3]]
        return 0.2 * source.square().sum() + 0.1 * (positions * cartesian_dipoles).sum()


class _CountingCDS:
    def __init__(self):
        self.calls = 0

    def configuration_sha256(self):
        return "c" * 64

    def energy_torch(self, positions):
        self.calls += 1
        return 0.05 * positions.pow(3).sum()


def _polynomial_pes(*, fp32):
    model = _CountingModel(torch.float32 if fp32 else torch.float64)
    continuum = _CountingContinuum()
    cds = _CountingCDS()
    profile = CPU_V4 if fp32 else CPU_V3
    pes = MACEPolarTorchPES._for_testing(
        symbols=("O", "H", "H"),
        model=model,
        continuum=continuum,
        solvent_term=cds,
        device="cpu",
        requested_profile=profile,
    )
    return pes, model, continuum, cds


def _expected(atoms, dtype):
    coordinates = atoms.positions.astype(dtype)
    energy = np.sum(
        0.07 * coordinates**2 + 0.15 * coordinates**3 + 0.2 * coordinates**4
    )
    forces = -(0.14 * coordinates + 0.45 * coordinates**2 + 0.8 * coordinates**3)
    return energy, forces


@pytest.mark.parametrize(
    ("fp32", "numpy_dtype", "atol"),
    [(False, np.float64, 1.0e-12), (True, np.float32, 2.0e-6)],
)
def test_pes_force_evaluation_uses_one_forward_and_preserves_values(
    fp32, numpy_dtype, atol
):
    atoms = _water()
    pes, model, continuum, cds = _polynomial_pes(fp32=fp32)

    evaluation = pes.evaluate_forces(atoms)

    expected_energy, expected_forces = _expected(atoms, numpy_dtype)
    assert evaluation.energy_eV == pytest.approx(expected_energy, abs=atol)
    np.testing.assert_allclose(evaluation.forces_eV_per_A, expected_forces, atol=atol)
    assert (model.calls, continuum.calls, cds.calls) == (1, 1, 1)
    assert model.requires_grad == [True]


@pytest.mark.parametrize(
    ("fp32", "expected_dtype"), [(False, np.float64), (True, np.float32)]
)
def test_standalone_get_forces_preserves_native_dtype_and_returns_a_copy(
    fp32, expected_dtype
):
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=fp32)

    first = pes.get_forces(atoms)
    baseline = first.copy()
    first[:] = 0.0
    second = pes.get_forces(atoms)

    assert first.dtype == second.dtype == expected_dtype
    np.testing.assert_array_equal(second, baseline)
    assert not np.shares_memory(first, second)


def test_force_evaluation_is_detached_immutable_data_without_a_torch_graph():
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=True)

    evaluation = pes.evaluate_forces(atoms)

    assert isinstance(evaluation.energy_eV, float)
    assert isinstance(evaluation.forces_eV_per_A, np.ndarray)
    assert evaluation.forces_eV_per_A.dtype == np.float64
    assert not evaluation.forces_eV_per_A.flags.writeable
    assert not any(
        torch.is_tensor(getattr(evaluation, record.name))
        for record in fields(evaluation)
    )


def test_energy_only_path_uses_one_graph_free_forward():
    atoms = _water()
    pes, model, continuum, cds = _polynomial_pes(fp32=True)

    energy = pes.get_potential_energy(atoms)

    assert np.isfinite(energy)
    assert (model.calls, continuum.calls, cds.calls) == (1, 1, 1)
    assert model.requires_grad == [False]


def test_pes_rejects_model_input_mutation_during_force_evaluation(monkeypatch):
    atoms = _water()
    pes, _, continuum, _ = _polynomial_pes(fp32=True)
    original = continuum.energy_torch

    def mutate_after_source_audit(positions, source):
        value = original(positions, source)
        atoms.info["charge"] = 2
        return value

    monkeypatch.setattr(continuum, "energy_torch", mutate_after_source_audit)

    with pytest.raises(RuntimeError, match="model inputs changed"):
        pes.evaluate_forces(atoms)


def _force_evaluation(atoms, spec, *, energy=-2.5, force=0.125, **overrides):
    payload = {
        "energy_eV": energy,
        "forces_eV_per_A": np.full((len(atoms), 3), force),
        "geometry_sha256": overrides.pop("geometry_sha256", geometry_sha256(atoms)),
        "model_input_sha256": model_input_sha256(atoms),
        "configuration_sha256": "f" * 64,
        "scalar_contract_id": spec.scalar_contract_id,
        "device": "cpu",
        "component_energies_eV": {"vacuum": energy, "ddpcm": 0.0, "cds": 0.0},
        "provider_id": PROVIDER_ID,
        "profile_id": spec.name,
        "verification_evidence_id": (
            V4_EVIDENCE if spec.name.endswith("v4") else V3_EVIDENCE
        ),
        "computation_dtype": "float32" if spec.name.endswith("v4") else "float64",
    }
    payload.update(overrides)
    return AnalyticForceEvaluation(**payload)


class _StructuredPES:
    provider_id = PROVIDER_ID

    def __init__(self, spec):
        self.scalar_contract_id = spec.scalar_contract_id
        self.device = "cpu"
        self.spec = spec
        self.evaluate_calls = 0
        self.energy_calls = 0
        self.force_calls = 0
        self.fail = False
        self.override = {}

    def configuration_sha256(self):
        return "f" * 64

    def evaluate_forces(self, atoms):
        self.evaluate_calls += 1
        if self.fail:
            raise RuntimeError("injected structured force failure")
        return _force_evaluation(atoms, self.spec, **self.override)

    def get_potential_energy(self, atoms):
        self.energy_calls += 1
        return -2.5

    def get_forces(self, atoms):
        self.force_calls += 1
        return np.full((len(atoms), 3), 0.125)

    def evaluate_hessian(self, atoms):
        return SimpleNamespace(
            hessian_eV_per_A2=np.eye(3 * len(atoms)),
            forces_eV_per_A=np.full((len(atoms), 3), 0.125),
            energy_eV=-2.5,
            derivative_method="torch-autograd",
        )

    def hessian_vector_product(self, atoms, direction):
        return np.asarray(direction, dtype=float)


def _calculator(atoms, *, version=3, pes=None):
    spec = route2_smd_profile_spec(CPU_V4 if version == 4 else CPU_V3)
    selected = pes or _StructuredPES(spec)
    calculator = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=selected,
    )
    return calculator, selected


def test_ase_force_request_commits_energy_and_forces_from_one_structured_evaluation():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    atoms.calc = calculator

    np.testing.assert_allclose(atoms.get_forces(), 0.125)

    assert atoms.get_potential_energy() == pytest.approx(-2.5)
    assert pes.evaluate_calls == 1
    assert pes.energy_calls == pes.force_calls == 0


def test_ase_repeated_reads_at_same_input_are_cache_hits():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    atoms.calc = calculator

    atoms.get_forces()
    atoms.get_potential_energy()
    atoms.get_forces()

    assert pes.evaluate_calls == 1


def test_ase_energy_only_request_keeps_the_scalar_path():
    atoms = _water()
    calculator, pes = _calculator(atoms)

    calculator.calculate(atoms, properties=["energy"])

    assert calculator.results == {"energy": -2.5, "free_energy": -2.5}
    assert pes.energy_calls == 1
    assert pes.evaluate_calls == pes.force_calls == 0


@pytest.mark.parametrize("mutation", ["geometry", "charge"])
def test_ase_valid_input_changes_invalidate_detached_results(mutation):
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    atoms.calc = calculator
    atoms.get_forces()

    if mutation == "geometry":
        atoms.positions[0, 0] += 0.01
    elif mutation == "charge":
        atoms.info["charge"] = 2
    calculator.get_property("forces", atoms)

    assert pes.evaluate_calls == 2


def test_ase_invalid_multiplicity_never_returns_a_cached_force():
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    atoms.calc = calculator
    atoms.get_forces()
    atoms.info["mult"] = 3

    with pytest.raises(ValueError, match="closed-shell"):
        calculator.get_property("forces", atoms)

    assert pes.evaluate_calls == 1


def test_force_failure_clears_stale_results_and_does_not_commit_a_new_cache_key():
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    calculator.calculate(atoms, properties=["energy", "forces"])
    assert calculator.results
    pes.fail = True
    atoms.positions[0, 0] += 0.01

    with pytest.raises(RuntimeError, match="structured force failure"):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert calculator.results == {}
    assert calculator._cache_key is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("geometry_sha256", "0" * 64, "geometry"),
        ("model_input_sha256", "0" * 64, "model input"),
        ("configuration_sha256", "0" * 64, "configuration"),
        ("scalar_contract_id", "wrong-scalar", "scalar"),
        ("provider_id", "wrong-provider", "provider"),
    ],
)
def test_ase_rejects_a_structured_evaluation_with_wrong_identity(field, value, message):
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.override[field] = value

    with pytest.raises(RuntimeError, match=message):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert calculator.results == {}
    assert calculator._cache_key is None


@pytest.mark.parametrize("mutation", ["geometry", "charge", "multiplicity"])
def test_mid_evaluation_input_mutation_prevents_atomic_commit(mutation):
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    original = pes.evaluate_forces

    def mutate_during_evaluation(target):
        evaluation = original(target)
        if mutation == "geometry":
            target.positions[0, 0] += 0.01
        elif mutation == "charge":
            target.info["charge"] = 2
        else:
            target.info["mult"] = 3
        return evaluation

    pes.evaluate_forces = mutate_during_evaluation

    with pytest.raises((RuntimeError, ValueError), match="changed|input|closed-shell"):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert calculator.results == {}
    assert calculator._cache_key is None


def test_ase_force_request_has_no_fallback_to_separate_energy_or_force_methods():
    atoms = _water()
    calculator, pes = _calculator(atoms)

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert pes.evaluate_calls == 1
    assert pes.energy_calls == 0
    assert pes.force_calls == 0


def test_ase_rejects_an_unstructured_force_result_without_fallback():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.evaluate_forces = lambda target: (-2.5, np.zeros((len(target), 3)))

    with pytest.raises(RuntimeError, match="analytic force evaluation"):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert calculator.results == {}
    assert calculator._cache_key is None
    assert pes.energy_calls == pes.force_calls == 0
