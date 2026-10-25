"""Single-graph analytic HVP handoff across the Torch PES and ASE facade."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.function.calculator.route2 import PureMACEPolarTorchCalculator
from maple.function.route2_smd_profiles import route2_smd_profile_spec
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.analytic import (
    AnalyticForceEvaluation,
    AnalyticHVPEvaluation,
)
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

    def configuration_sha256(self):
        return "a" * 64

    def energy_source_torch(self, atoms, positions):
        self.calls += 1
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
    pes = MACEPolarTorchPES._for_testing(
        symbols=("O", "H", "H"),
        model=model,
        continuum=continuum,
        solvent_term=cds,
        device="cpu",
        requested_profile=CPU_V4 if fp32 else CPU_V3,
    )
    return pes, model, continuum, cds


def _polynomial_oracle(atoms, direction, dtype):
    coordinates = atoms.positions.astype(dtype)
    probe = np.asarray(direction, dtype=dtype)
    energy = np.sum(
        0.07 * coordinates**2 + 0.15 * coordinates**3 + 0.2 * coordinates**4
    )
    forces = -(0.14 * coordinates + 0.45 * coordinates**2 + 0.8 * coordinates**3)
    hvp = (0.14 + 0.9 * coordinates + 2.4 * coordinates**2) * probe
    return energy, forces, hvp


@pytest.mark.parametrize(
    ("fp32", "numpy_dtype", "atol"),
    [(False, np.float64, 1.0e-12), (True, np.float32, 2.0e-6)],
)
def test_pes_hvp_evaluation_uses_one_forward_and_preserves_all_values(
    fp32, numpy_dtype, atol
):
    atoms = _water()
    direction = np.linspace(-0.7, 0.9, 9).reshape(3, 3)
    pes, model, continuum, cds = _polynomial_pes(fp32=fp32)

    evaluation = pes.evaluate_hvp(atoms, direction)

    energy, forces, hvp = _polynomial_oracle(atoms, direction, numpy_dtype)
    assert evaluation.force_evaluation.energy_eV == pytest.approx(energy, abs=atol)
    np.testing.assert_allclose(
        evaluation.force_evaluation.forces_eV_per_A, forces, atol=atol
    )
    np.testing.assert_allclose(evaluation.hvp_eV_per_A2, hvp, atol=atol)
    assert (model.calls, continuum.calls, cds.calls) == (1, 1, 1)


def test_composed_pes_hvp_uses_central_force_plus_two_hvp_reverse_operations(
    monkeypatch,
):
    atoms = _water()
    direction = np.ones((3, 3))
    pes, *_ = _polynomial_pes(fp32=False)
    original_grad = torch.autograd.grad
    create_graph_values = []

    def count_grad(*args, **kwargs):
        create_graph_values.append(bool(kwargs.get("create_graph", False)))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", count_grad)
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "evaluate_hessian",
        lambda *_: pytest.fail("direct HVP must not materialize a dense Hessian"),
    )

    pes.evaluate_hvp(atoms, direction)

    assert create_graph_values == [False, True, False]


def test_standalone_pes_hvp_uses_only_two_hvp_reverse_operations(monkeypatch):
    atoms = _water()
    direction = np.ones((3, 3))
    pes, *_ = _polynomial_pes(fp32=False)
    original_grad = torch.autograd.grad
    create_graph_values = []

    def count_grad(*args, **kwargs):
        create_graph_values.append(bool(kwargs.get("create_graph", False)))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", count_grad)
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "evaluate_hessian",
        lambda *_: pytest.fail("direct HVP must not materialize a dense Hessian"),
    )

    pes.hessian_vector_product(atoms, direction)

    assert create_graph_values == [True, False]


def test_composed_pes_hvp_returns_the_create_graph_false_central_force(monkeypatch):
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=True)
    original_grad = torch.autograd.grad
    central_gradient = 11.0

    def distinguish_central_gradient(*args, **kwargs):
        if not kwargs.get("create_graph", False) and kwargs.get("retain_graph") is True:
            return (torch.full_like(args[1], central_gradient),)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", distinguish_central_gradient)

    evaluation = pes.evaluate_hvp(atoms, np.ones((3, 3)))

    np.testing.assert_array_equal(
        evaluation.force_evaluation.forces_eV_per_A, -central_gradient
    )


class _LinearModel(_CountingModel):
    def energy_source_torch(self, atoms, positions):
        self.calls += 1
        x, y, z = positions.unbind(-1)
        source = torch.stack((x * 0, y * 0, z * 0, x * 0), dim=-1)
        return 0.07 * positions.sum(), source


class _LinearContinuum(_CountingContinuum):
    def energy_torch(self, positions, source):
        self.calls += 1
        return 0.0 * (positions.sum() + source.sum())


class _LinearCDS(_CountingCDS):
    def energy_torch(self, positions):
        self.calls += 1
        return 0.03 * positions.sum()


class _ConstantModel(_CountingModel):
    def energy_source_torch(self, atoms, positions):
        self.calls += 1
        x, y, z = positions.unbind(-1)
        source = torch.stack((x * 0, y * 0, z * 0, x * 0), dim=-1)
        return 0.0 * positions.sum(), source


class _ConstantCDS(_CountingCDS):
    def energy_torch(self, positions):
        self.calls += 1
        return 0.0 * positions.sum()


@pytest.mark.parametrize(
    ("model_type", "cds_type"),
    [(_LinearModel, _LinearCDS), (_ConstantModel, _ConstantCDS)],
    ids=["linear", "constant"],
)
def test_pes_constant_or_linear_scalar_returns_exact_zero_hvp_without_fallback(
    model_type, cds_type
):
    atoms = _water()
    pes = MACEPolarTorchPES._for_testing(
        symbols=("O", "H", "H"),
        model=model_type(torch.float64),
        continuum=_LinearContinuum(),
        solvent_term=cds_type(),
        device="cpu",
        requested_profile=CPU_V3,
    )

    evaluation = pes.evaluate_hvp(atoms, np.ones((3, 3)))

    np.testing.assert_array_equal(evaluation.hvp_eV_per_A2, 0.0)


def test_pes_zero_direction_returns_exact_zero_hvp():
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=True)

    evaluation = pes.evaluate_hvp(atoms, np.zeros((3, 3)))

    np.testing.assert_array_equal(evaluation.hvp_eV_per_A2, 0.0)


@pytest.mark.parametrize(
    "direction", [np.ones(9), np.full((3, 3), np.nan), np.full((3, 3), np.inf)]
)
def test_pes_invalid_direction_fails_before_model_execution(direction):
    atoms = _water()
    pes, model, continuum, cds = _polynomial_pes(fp32=False)

    with pytest.raises(ValueError, match="direction|finite|shape"):
        pes.evaluate_hvp(atoms, direction)

    assert (model.calls, continuum.calls, cds.calls) == (0, 0, 0)


def test_pes_resource_preflight_fails_before_positions_or_graph(monkeypatch):
    import maple.solvation.experimental.mace_polar_torch as module

    atoms = _water()
    pes, model, continuum, cds = _polynomial_pes(fp32=False)
    object.__setattr__(pes, "_testing", False)
    monkeypatch.setattr(
        module,
        "_second_order_resource_preflight",
        lambda *_: (_ for _ in ()).throw(MemoryError("injected preflight")),
    )
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "_positions",
        lambda *_args, **_kwargs: pytest.fail("positions created before preflight"),
    )

    with pytest.raises(MemoryError, match="injected preflight"):
        pes.evaluate_hvp(atoms, np.ones((3, 3)))

    assert (model.calls, continuum.calls, cds.calls) == (0, 0, 0)


def test_pes_hvp_records_actual_float32_probe_not_unrounded_request():
    atoms = _water()
    requested = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
    pes, *_ = _polynomial_pes(fp32=True)

    evaluation = pes.evaluate_hvp(atoms, requested)

    actual = requested.astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(evaluation.direction, actual)
    assert np.any(evaluation.direction != requested)


def test_pes_hvp_evaluation_is_detached_immutable_data_without_a_torch_graph():
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=True)

    evaluation = pes.evaluate_hvp(atoms, np.ones((3, 3)))

    assert not evaluation.direction.flags.writeable
    assert not evaluation.hvp_eV_per_A2.flags.writeable
    assert not evaluation.force_evaluation.forces_eV_per_A.flags.writeable
    assert not any(
        torch.is_tensor(getattr(evaluation, item.name)) for item in fields(evaluation)
    )
    assert not any(
        torch.is_tensor(getattr(evaluation.force_evaluation, item.name))
        for item in fields(evaluation.force_evaluation)
    )


@pytest.mark.parametrize(
    ("fp32", "expected_dtype"), [(False, np.float64), (True, np.float32)]
)
def test_standalone_hvp_preserves_native_dtype_and_returns_writable_copies(
    fp32, expected_dtype
):
    atoms = _water()
    direction = np.ones((3, 3))
    pes, *_ = _polynomial_pes(fp32=fp32)

    first = pes.hessian_vector_product(atoms, direction)
    baseline = first.copy()
    first[:] = 0.0
    second = pes.hessian_vector_product(atoms, direction)

    assert first.dtype == second.dtype == expected_dtype
    assert first.flags.writeable and second.flags.writeable
    np.testing.assert_array_equal(second, baseline)
    assert not np.shares_memory(first, second)


def test_standalone_hvp_adapter_forbids_other_observable_methods(monkeypatch):
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=False)
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "evaluate_hessian",
        lambda *_: pytest.fail("dense fallback forbidden"),
    )
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "evaluate_forces",
        lambda *_: pytest.fail("force fallback forbidden"),
    )
    monkeypatch.setattr(
        MACEPolarTorchPES,
        "get_potential_energy",
        lambda *_: pytest.fail("energy fallback forbidden"),
    )

    result = pes.hessian_vector_product(atoms, np.ones((3, 3)))

    assert result.shape == (3, 3)


def test_pes_rejects_caller_direction_mutation_during_evaluation(monkeypatch):
    atoms = _water()
    direction = np.ones((3, 3))
    pes, *_ = _polynomial_pes(fp32=False)
    original = MACEPolarTorchPES._energy_components

    def mutate_after_forward(self, target, positions):
        result = original(self, target, positions)
        direction[0, 0] = 2.0
        return result

    monkeypatch.setattr(MACEPolarTorchPES, "_energy_components", mutate_after_forward)

    with pytest.raises(RuntimeError, match="direction.*changed|changed.*direction"):
        pes.evaluate_hvp(atoms, direction)


def test_pes_rejects_direction_mutation_after_force_record_materialization(
    monkeypatch,
):
    atoms = _water()
    direction = np.ones((3, 3))
    pes, *_ = _polynomial_pes(fp32=False)
    original = MACEPolarTorchPES._force_evaluation_from_graph

    def mutate_after_force_record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        direction[0, 0] = 2.0
        return result

    monkeypatch.setattr(
        MACEPolarTorchPES,
        "_force_evaluation_from_graph",
        mutate_after_force_record,
    )

    with pytest.raises(RuntimeError, match="direction.*changed|changed.*direction"):
        pes.evaluate_hvp(atoms, direction)


def test_pes_rejects_atom_mutation_after_force_record_materialization(monkeypatch):
    atoms = _water()
    pes, *_ = _polynomial_pes(fp32=False)
    original = MACEPolarTorchPES._force_evaluation_from_graph

    def mutate_after_force_record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        atoms.positions[0, 0] += 0.01
        return result

    monkeypatch.setattr(
        MACEPolarTorchPES,
        "_force_evaluation_from_graph",
        mutate_after_force_record,
    )

    with pytest.raises(RuntimeError, match="model inputs changed"):
        pes.evaluate_hvp(atoms, np.ones((3, 3)))


@pytest.mark.parametrize("mutation", ["geometry", "charge", "configuration"])
def test_pes_rejects_model_or_configuration_mutation_during_hvp(monkeypatch, mutation):
    atoms = _water()
    pes, _, continuum, _ = _polynomial_pes(fp32=True)
    original = continuum.energy_torch

    def mutate_after_source(positions, source):
        value = original(positions, source)
        if mutation == "geometry":
            atoms.positions[0, 0] += 0.01
        elif mutation == "charge":
            atoms.info["charge"] = 2
        else:
            continuum.configuration_sha256 = lambda: "d" * 64
        return value

    monkeypatch.setattr(continuum, "energy_torch", mutate_after_source)

    with pytest.raises(RuntimeError, match="configuration|model inputs changed"):
        pes.evaluate_hvp(atoms, np.ones((3, 3)))


def _force_evaluation(atoms, spec, *, energy=-2.5, force=0.125, **overrides):
    values = {
        "energy_eV": energy,
        "forces_eV_per_A": np.full((len(atoms), 3), force),
        "geometry_sha256": geometry_sha256(atoms),
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
    values.update(overrides)
    return AnalyticForceEvaluation(**values)


class _StructuredHVPPES:
    provider_id = PROVIDER_ID

    def __init__(self, spec):
        self.scalar_contract_id = spec.scalar_contract_id
        self.device = "cpu"
        self.spec = spec
        self.evaluate_calls = 0
        self.old_hvp_calls = 0
        self.force_calls = 0
        self.energy_calls = 0
        self.callback = None
        self.force_overrides = {}
        self.direction_override = None
        self.hvp_override = None
        self.result_override = None
        self.tamper_hvp = None
        self.tamper_evaluation_sha256 = None
        self.tamper_force_field = None
        self.mutate_received_direction = False
        self.configuration = "f" * 64

    def configuration_sha256(self):
        return self.configuration

    def evaluate_hvp(self, atoms, direction):
        self.evaluate_calls += 1
        force = _force_evaluation(atoms, self.spec, **self.force_overrides)
        actual_direction = np.array(
            direction,
            dtype=np.float32 if self.spec.name.endswith("v4") else np.float64,
            copy=True,
        )
        result = AnalyticHVPEvaluation(
            force_evaluation=force,
            direction=(
                actual_direction
                if self.direction_override is None
                else self.direction_override
            ),
            hvp_eV_per_A2=(
                2.0 * actual_direction
                if self.hvp_override is None
                else self.hvp_override
            ),
        )
        if self.callback is not None:
            self.callback(atoms)
        if self.mutate_received_direction:
            direction[:] = 99.0
        if self.tamper_hvp is not None:
            object.__setattr__(result, "hvp_eV_per_A2", self.tamper_hvp)
        if self.tamper_evaluation_sha256 is not None:
            object.__setattr__(
                result, "evaluation_sha256", self.tamper_evaluation_sha256
            )
        if self.tamper_force_field is not None:
            field, value = self.tamper_force_field
            object.__setattr__(result.force_evaluation, field, value)
        return result if self.result_override is None else self.result_override

    def hessian_vector_product(self, *_):
        self.old_hvp_calls += 1
        pytest.fail("legacy unstructured HVP method must not be called")

    def get_forces(self, *_):
        self.force_calls += 1
        pytest.fail("separate force method must not be called")

    def get_potential_energy(self, *_):
        self.energy_calls += 1
        pytest.fail("separate energy method must not be called")


def _calculator(atoms, *, version=3, pes=None):
    spec = route2_smd_profile_spec(CPU_V4 if version == 4 else CPU_V3)
    selected = pes or _StructuredHVPPES(spec)
    calculator = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=selected,
    )
    return calculator, selected


@pytest.mark.parametrize("shape", [(9,), (1, 9), (3, 3), (9, 1)])
def test_ase_hvp_accepts_any_3n_shape_and_returns_legacy_writable_tensors(shape):
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    direction = np.linspace(-0.4, 0.4, 9).reshape(shape)

    hvp, forces, energy = calculator.get_hvp(atoms, direction)

    assert pes.evaluate_calls == 1
    assert pes.old_hvp_calls == pes.force_calls == pes.energy_calls == 0
    assert hvp.shape == forces.shape == (9,)
    assert energy.shape == ()
    assert hvp.dtype == forces.dtype == energy.dtype == torch.float64
    assert hvp.device.type == forces.device.type == energy.device.type == "cpu"
    assert hvp.is_contiguous() and forces.is_contiguous()
    np.testing.assert_allclose(hvp.numpy(), 2.0 * direction.reshape(-1))
    np.testing.assert_allclose(forces.numpy(), 0.125)
    assert float(energy) == pytest.approx(-2.5)
    hvp[0] = 99.0
    forces[0] = 99.0


def test_ase_hvp_outputs_remain_cpu_float64_under_meta_default_device():
    atoms = _water()
    calculator, _ = _calculator(atoms)

    with torch.device("meta"):
        hvp, forces, energy = calculator.get_hvp(atoms, np.ones((3, 3)))

    assert hvp.shape == forces.shape == (9,)
    assert energy.shape == ()
    assert hvp.device.type == forces.device.type == energy.device.type == "cpu"
    assert hvp.dtype == forces.dtype == energy.dtype == torch.float64


def test_ase_hvp_returns_defensive_copies_on_repeated_calls():
    atoms = _water()
    calculator, _ = _calculator(atoms)
    direction = np.ones((3, 3))
    first = calculator.get_hvp(atoms, direction)
    first[0][:] = 99.0
    first[1][:] = 99.0
    first[2].fill_(99.0)

    second = calculator.get_hvp(atoms, direction)

    np.testing.assert_array_equal(second[0].numpy(), 2.0)
    np.testing.assert_array_equal(second[1].numpy(), 0.125)
    assert float(second[2]) == pytest.approx(-2.5)


def test_ase_hvp_passes_a_private_direction_copy_to_the_pes():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    direction = np.ones((3, 3))
    pes.mutate_received_direction = True

    calculator.get_hvp(atoms, direction)

    np.testing.assert_array_equal(direction, 1.0)


def _seed_calculator_state(calculator):
    cache_key = ("1" * 64, "2" * 64, "profile", "scalar", "cpu")
    hessian_key = ("3" * 64, "4" * 64, "profile", "scalar", "cpu")
    hessian_evaluation = object()
    results = {"energy": 7.0, "forces": np.ones((3, 3))}
    calculator.results = results
    calculator._cache_key = cache_key
    calculator._hessian_cache_key = hessian_key
    calculator._hessian_evaluation = hessian_evaluation
    return results, cache_key, hessian_key, hessian_evaluation


def _assert_calculator_state(calculator, expected):
    results, cache_key, hessian_key, hessian_evaluation = expected
    assert calculator.results is results
    assert calculator._cache_key is cache_key
    assert calculator._hessian_cache_key is hessian_key
    assert calculator._hessian_evaluation is hessian_evaluation


def test_ase_hvp_success_leaves_all_result_and_hessian_caches_unchanged():
    atoms = _water()
    calculator, _ = _calculator(atoms)
    expected = _seed_calculator_state(calculator)

    calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


@pytest.mark.parametrize(
    "direction", [np.ones(8), np.full(9, np.nan), np.full(9, np.inf)]
)
def test_ase_invalid_direction_fails_before_pes_and_preserves_caches(direction):
    atoms = _water()
    calculator, pes = _calculator(atoms)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(ValueError, match="3N|finite|direction"):
        calculator.get_hvp(atoms, direction)

    assert pes.evaluate_calls == 0
    _assert_calculator_state(calculator, expected)


def test_ase_constraints_fail_before_pes_and_preserve_caches():
    from ase.constraints import FixAtoms

    atoms = _water()
    atoms.set_constraint(FixAtoms(indices=[0]))
    calculator, pes = _calculator(atoms)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(ValueError, match="constrained"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    assert pes.evaluate_calls == 0
    _assert_calculator_state(calculator, expected)


def test_ase_rejects_unstructured_hvp_result_without_old_method_fallback():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.result_override = (np.ones((3, 3)), np.ones((3, 3)), -2.5)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="analytic HVP evaluation"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    assert pes.old_hvp_calls == pes.force_calls == pes.energy_calls == 0
    _assert_calculator_state(calculator, expected)


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("model_input_sha256", "0" * 64, "model input"),
        ("configuration_sha256", "0" * 64, "configuration"),
        ("geometry_sha256", "0" * 64, "geometry"),
        ("provider_id", "wrong-provider", "provider"),
        ("scalar_contract_id", "wrong-scalar", "scalar"),
        ("profile_id", "wrong-profile", "profile"),
        ("device", "cuda:0", "device"),
    ],
)
def test_ase_rejects_forged_nested_force_identity_and_preserves_caches(
    override, value, message
):
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.force_overrides[override] = value
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match=message):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


def test_ase_rejects_direction_forgery_and_preserves_caches():
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    pes.direction_override = np.zeros((3, 3), dtype=np.float32)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="direction"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


@pytest.mark.parametrize(
    "hvp", [np.ones((9,)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)]
)
def test_ase_rejects_forged_hvp_values_and_preserves_caches(hvp):
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.tamper_hvp = hvp
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="HVP|invalid"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


def test_ase_rejects_post_construction_finite_hvp_mutation_and_preserves_caches():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.tamper_hvp = np.full((3, 3), 7.0)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="HVP|content|evaluation"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


def test_ase_rejects_hvp_digest_mutation_and_preserves_caches():
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.tamper_evaluation_sha256 = "0" * 64
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="HVP|content|evaluation"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy_eV", -9.0),
        ("forces_eV_per_A", np.full((3, 3), 0.75)),
    ],
)
def test_ase_rejects_nested_force_value_mutation_and_preserves_caches(field, value):
    atoms = _water()
    calculator, pes = _calculator(atoms)
    pes.tamper_force_field = (field, value)
    expected = _seed_calculator_state(calculator)

    with pytest.raises(RuntimeError, match="HVP|force|content|evaluation"):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


@pytest.mark.parametrize("mutation", ["geometry", "charge", "multiplicity", "config"])
def test_ase_rejects_mid_evaluation_input_drift_and_preserves_caches(mutation):
    atoms = _water()
    calculator, pes = _calculator(atoms, version=4)
    expected = _seed_calculator_state(calculator)

    def mutate(target):
        if mutation == "geometry":
            target.positions[0, 0] += 0.01
        elif mutation == "charge":
            target.info["charge"] = 2
        elif mutation == "multiplicity":
            target.info["mult"] = 3
        else:
            pes.configuration = "e" * 64

    pes.callback = mutate

    with pytest.raises(
        (RuntimeError, ValueError), match="changed|input|configuration|closed-shell"
    ):
        calculator.get_hvp(atoms, np.ones((3, 3)))

    _assert_calculator_state(calculator, expected)


def test_ase_rejects_caller_direction_mutation_and_preserves_caches():
    atoms = _water()
    direction = np.ones((3, 3))
    calculator, pes = _calculator(atoms)
    expected = _seed_calculator_state(calculator)
    pes.callback = lambda _atoms: direction.__setitem__((0, 0), 2.0)

    with pytest.raises(RuntimeError, match="direction.*changed|changed.*direction"):
        calculator.get_hvp(atoms, direction)

    _assert_calculator_state(calculator, expected)
