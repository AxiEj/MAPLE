from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from maple.function.calculator.consolv._torch_model import (
    CHECKPOINT_IDENTITIES,
    FORMAT,
    load_consolv_torch_checkpoint,
)

MODEL_DIR = Path(
    os.environ.get(
        "MAPLE_CONSOLV_TEST_MODEL_DIR",
        Path(__file__).resolve().parents[2] / "maple/function/calculator/model/consolv",
    )
)
CHECKPOINT = MODEL_DIR / "consolv-torch-v1.pt"
VACUUM_CHECKPOINT = CHECKPOINT.with_name("consolv-vacuum-torch-v1.pt")
RUN_REAL_ASSETS = os.environ.get("MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS") == "1"
real_asset_test = pytest.mark.skipif(
    not RUN_REAL_ASSETS or not (CHECKPOINT.is_file() and VACUUM_CHECKPOINT.is_file()),
    reason="set MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS=1 with both frozen checkpoints",
)


def _water():
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
        dtype=torch.float64,
    )
    return numbers, positions


@real_asset_test
def test_checkpoint_is_safe_complete_and_jax_free():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    assert payload["format"] == FORMAT
    assert len(payload["parameter_map"]) == 60
    assert payload["metadata"]["parameter_count"] == 768338
    assert payload["unused_source_parameters"] == [
        "mace_consol/layer_0/linear_readout_layer/linear::w[0,0] 64x0e,1x0e"
    ]
    before = {
        name for name in sys.modules if name.startswith(("jax", "haiku", "e3nn_jax"))
    }
    model = load_consolv_torch_checkpoint(CHECKPOINT, dtype=torch.float64)
    after = {
        name for name in sys.modules if name.startswith(("jax", "haiku", "e3nn_jax"))
    }
    assert after == before
    assert model.metadata["implementation"] == "independent_pure_torch"


@real_asset_test
def test_paired_vacuum_checkpoint_is_solvent_invariant_and_finite():
    payload = torch.load(
        VACUUM_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
    )
    assert payload["metadata"]["potential_state"] == "vacuum"
    model = load_consolv_torch_checkpoint(
        VACUUM_CHECKPOINT, dtype=torch.float64, expected_state="vacuum"
    )
    numbers, positions = _water()
    energy_a, forces_a = model.energy_and_forces_ev(numbers, positions, 19)
    energy_b, forces_b = model.energy_and_forces_ev(numbers, positions, 45)
    assert torch.isfinite(energy_a)
    assert torch.isfinite(forces_a).all()
    assert torch.equal(energy_a, energy_b)
    assert torch.equal(forces_a, forces_b)


@real_asset_test
def test_two_solvents_are_finite_and_distinct():
    model = load_consolv_torch_checkpoint(CHECKPOINT, dtype=torch.float64)
    numbers, positions = _water()
    ethanol, f_ethanol = model.energy_and_forces_ev(numbers, positions, 19)
    toluene, f_toluene = model.energy_and_forces_ev(numbers, positions, 45)
    assert torch.isfinite(ethanol)
    assert torch.isfinite(toluene)
    assert torch.isfinite(f_ethanol).all()
    assert torch.isfinite(f_toluene).all()
    assert ethanol.item() != toluene.item()


@real_asset_test
def test_frozen_conversion_reference_for_two_solvents():
    """Lock the independent Torch port to the reconstructed published graph."""
    model = load_consolv_torch_checkpoint(CHECKPOINT, dtype=torch.float64)
    numbers = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    positions = torch.tensor(
        [
            [-0.748, 0.015, 0.024],
            [0.748, -0.015, -0.024],
            [1.424, 1.171, 0.106],
            [-1.129, -0.514, 0.899],
            [-1.104, 1.047, 0.050],
            [-1.136, -0.482, -0.892],
            [1.116, -0.558, -0.895],
            [1.104, -0.604, 0.837],
            [2.367, 0.999, 0.092],
        ],
        dtype=torch.float64,
    )
    references = {
        19: (-34.93867043776235, [-0.62102622, 0.27658415, -0.80794168]),
        45: (-34.87228928881432, [-0.68025142, 0.27134016, -0.81029111]),
    }
    for solvent, (energy_ref, force_ref) in references.items():
        energy, forces = model.energy_and_forces_ev(numbers, positions, solvent)
        assert abs(energy.item() - energy_ref) < 1.0e-5
        assert torch.allclose(
            forces[0], torch.tensor(force_ref, dtype=torch.float64), atol=5.0e-4, rtol=0
        )


@real_asset_test
def test_energy_forces_fd_translation_rotation_and_permutation():
    model = load_consolv_torch_checkpoint(CHECKPOINT, dtype=torch.float64)
    numbers, positions = _water()
    energy, forces = model.energy_and_forces_ev(numbers, positions, 19)
    step = 1.0e-4
    moved = positions.clone()
    moved[1, 2] += step
    plus = model.energy_ev(numbers, moved, 19)
    moved[1, 2] -= 2 * step
    minus = model.energy_ev(numbers, moved, 19)
    fd = -(plus - minus) / (2 * step)
    assert torch.allclose(fd, forces[1, 2], atol=1e-5, rtol=1e-7)

    shift = torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64)
    shifted_energy = model.energy_ev(numbers, positions + shift, 19)
    assert torch.allclose(shifted_energy, energy, atol=1e-8, rtol=1e-12)

    angle = 0.61
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    rotated_energy, rotated_forces = model.energy_and_forces_ev(
        numbers, positions @ rotation.T, 19
    )
    assert torch.allclose(rotated_energy, energy, atol=1e-7, rtol=1e-12)
    assert torch.allclose(rotated_forces, forces @ rotation.T, atol=1e-6, rtol=1e-10)

    permutation = torch.tensor([0, 2, 1])
    permuted = model.energy_ev(numbers[permutation], positions[permutation], 19)
    assert torch.allclose(permuted, energy, atol=1e-8, rtol=1e-12)


def test_loader_rejects_unrecognized_synthetic_checkpoint(tmp_path):
    checkpoint = tmp_path / "fake.pt"
    torch.save({"format": FORMAT, "metadata": {}}, checkpoint)
    with pytest.raises(ValueError, match="Unrecognized ConSolv Torch checkpoint"):
        load_consolv_torch_checkpoint(checkpoint)


def test_loader_rejects_unknown_state_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="Unsupported ConSolv potential state"):
        load_consolv_torch_checkpoint(tmp_path / "absent.pt", expected_state="gas")


def test_raw_diagnostic_loader_is_not_exported_from_supported_package():
    from maple.function.calculator import consolv

    assert consolv.__all__ == ["ConSolvCalculator"]
    assert not hasattr(consolv, "load_consolv_torch_checkpoint")


@pytest.mark.parametrize(
    "controls",
    [
        {"device": "mps"},
        {"dtype": torch.float16},
        {"dtype": None},
    ],
)
def test_diagnostic_loader_rejects_runtime_controls_before_file_access(
    monkeypatch, controls
):
    monkeypatch.setattr(Path, "is_file", lambda _: pytest.fail("file access reached"))
    with pytest.raises(ValueError, match="CPU|float32 or float64"):
        load_consolv_torch_checkpoint("not-accessed.pt", **controls)


@pytest.mark.parametrize(
    "actual,requested", [("solution", "vacuum"), ("vacuum", "solution")]
)
def test_loader_rejects_swapped_synthetic_identity_before_deserialization(
    tmp_path, monkeypatch, actual, requested
):
    from maple.function.calculator.consolv import _torch_model

    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"synthetic identity test")
    monkeypatch.setattr(
        _torch_model, "_sha256", lambda _: CHECKPOINT_IDENTITIES[actual]
    )
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("deserialization reached")
    )
    with pytest.raises(ValueError, match=f"is '{actual}', not requested '{requested}'"):
        load_consolv_torch_checkpoint(path, expected_state=requested)


@real_asset_test
def test_solution_and_vacuum_identities_cannot_be_interchanged():
    import hashlib

    assert (
        hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        == CHECKPOINT_IDENTITIES["solution"]
    )
    assert (
        hashlib.sha256(VACUUM_CHECKPOINT.read_bytes()).hexdigest()
        == CHECKPOINT_IDENTITIES["vacuum"]
    )
    with pytest.raises(ValueError, match="is 'vacuum', not requested 'solution'"):
        load_consolv_torch_checkpoint(VACUUM_CHECKPOINT)
    with pytest.raises(ValueError, match="is 'solution', not requested 'vacuum'"):
        load_consolv_torch_checkpoint(CHECKPOINT, expected_state="vacuum")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loader_cpu_dtype_uses_safe_load_and_preserves_identity(
    tmp_path, monkeypatch, dtype
):
    from maple.function.calculator.consolv import _torch_model as module

    path = tmp_path / "identity.pt"
    path.write_bytes(b"mock identity")
    monkeypatch.setattr(module, "_sha256", lambda _: CHECKPOINT_IDENTITIES["solution"])
    calls = {}

    def load(path, **kwargs):
        calls.update(kwargs)
        return {"metadata": {"source_revision": module.SOURCE_REVISION}}

    class Model(torch.nn.Module):
        def __init__(self, payload, *, inference):
            super().__init__()
            self.register_buffer("weight", torch.ones(2, dtype=torch.float64))

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(module, "_ConSolvTorchModel", Model)
    model = module.load_consolv_torch_checkpoint(path, dtype=dtype)
    assert calls == {"map_location": "cpu", "weights_only": True, "mmap": True}
    assert model.weight.dtype == dtype
    assert model.weight.device.type == "cpu"
    assert not model.training


def _tiny_optimized_model():
    from maple.function.calculator.consolv._torch_model import _ConSolvTorchModel

    model = _ConSolvTorchModel.__new__(_ConSolvTorchModel)
    torch.nn.Module.__init__(model)
    model.register_buffer("dtype_reference", torch.ones(1, dtype=torch.float32))
    model.inference = "optimized"
    model._compiled_energy = None
    model.compilation_active = False
    model._energy_graph = (
        lambda species, positions, solvent, sender, receiver: positions.square().sum()
    )
    return model


def _tiny_compile_args(positions):
    return (
        torch.zeros(2, dtype=torch.long),
        positions,
        0,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )


def test_optimized_wrapper_without_backend_execution_is_rejected(monkeypatch):
    model = _tiny_optimized_model()
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        model._optimized_energy(
            *_tiny_compile_args(torch.ones((2, 3), requires_grad=True))
        )
    assert not model.compilation_active


def test_optimized_backend_execution_is_lazy_and_preserves_ad(monkeypatch):
    from torch._dynamo.backends import registry

    model = _tiny_optimized_model()
    calls = []
    monkeypatch.setattr(
        registry, "lookup_backend", lambda name: lambda graph, inputs, **kwargs: graph
    )

    def compile(fn, *, backend, fullgraph, dynamic):
        assert fullgraph is True
        assert dynamic is False
        calls.append("compile")
        return backend(fn, [])

    monkeypatch.setattr(torch, "compile", compile)
    assert not model.compilation_active
    for _ in range(2):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        value = model._optimized_energy(*_tiny_compile_args(x))
        assert model.compilation_active
        assert torch.equal(torch.autograd.grad(value, x)[0], 2 * x)
    assert calls == ["compile"]


def test_optimized_compile_error_is_not_retried_as_eager(monkeypatch):
    model = _tiny_optimized_model()

    def fail(*args, **kwargs):
        raise RuntimeError("compile unavailable")

    monkeypatch.setattr(torch, "compile", fail)
    with pytest.raises(RuntimeError, match="fallback is disabled") as error:
        model._optimized_energy(*_tiny_compile_args(torch.ones((2, 3))))
    assert "compile unavailable" in str(error.value.__cause__)
    assert not model.compilation_active


def test_e3nn_fx_tensor_product_matches_reference_without_global_mutation():
    import e3nn
    from e3nn import o3
    from maple.function.calculator.consolv._torch_model import _FXFullTensorProduct

    defaults = e3nn.get_optimization_defaults().copy()
    reference = o3.FullTensorProduct("2x0e", "1x1o+1x2e")
    candidate = _FXFullTensorProduct("2x0e", "1x1o+1x2e")
    assert isinstance(candidate._compiled_main_left_right, torch.fx.GraphModule)
    assert e3nn.get_optimization_defaults() == defaults
    x, y = torch.randn(3, 2), torch.randn(3, 8, requires_grad=True)
    expected, actual = reference(x, y), candidate(x, y)
    assert torch.equal(expected, actual)
    assert torch.equal(
        torch.autograd.grad(expected.sum(), y)[0],
        torch.autograd.grad(actual.sum(), y)[0],
    )


@real_asset_test
def test_cpu_fp32_buffers_energy_and_forces_use_selected_dtype():
    model = load_consolv_torch_checkpoint(CHECKPOINT, device="cpu", dtype=torch.float32)
    numbers, positions = _water()
    energy, forces = model.energy_and_forces_ev(numbers, positions.float(), 19)
    assert energy.dtype == torch.float32
    assert forces.dtype == torch.float32
    assert all(
        not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers()
    )


def test_neighbor_edges_are_rebuilt_after_coordinate_change():
    from maple.function.calculator.consolv._torch_model import _ConSolvTorchModel

    model = _tiny_optimized_model()
    model.inference = "reference"
    received = []

    def graph(species, positions, solvent, sender, receiver):
        received.append((species.clone(), sender.clone(), receiver.clone()))
        return positions.square().sum()

    model._energy_graph = graph
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    numbers = torch.tensor([8, 1, 1])
    model.energy_native(numbers, positions, 19)
    positions[2, 0] = 7.0
    model.energy_native(numbers, positions, 19)
    model.energy_native(torch.tensor([6, 1, 1]), positions, 19)
    assert received[0][1].numel() == 6
    assert received[1][1].numel() == 2
    assert received[2][0][0] == 5


@real_asset_test
def test_cpu_fullgraph_trace_preserves_energy_force_and_fresh_geometry():
    """Traceability check only: the CPU graph backend is not an optimized receipt."""
    from maple.function.calculator.consolv._torch_model import _ConSolvTorchModel

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    graph_model = _ConSolvTorchModel(payload, inference="optimized").float()
    reference = load_consolv_torch_checkpoint(
        CHECKPOINT, device="cpu", dtype=torch.float32
    )
    graphs = []

    def graph_backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    traced = torch.compile(
        graph_model._energy_graph, backend=graph_backend, fullgraph=True
    )
    numbers, base = _water()
    for solvent, displacement in [(19, 0.0), (19, 0.01), (45, 0.02)]:
        positions = base.float()
        positions[1, 2] += displacement
        expected_energy, expected_forces = reference.energy_and_forces_ev(
            numbers, positions, solvent
        )
        positions.requires_grad_(True)
        sender, receiver, _ = graph_model._valid_edges(positions)
        actual_energy = (
            traced(numbers - 1, positions, solvent, sender, receiver) / 96.485
        )
        actual_forces = -torch.autograd.grad(actual_energy, positions)[0]
        assert torch.allclose(actual_energy, expected_energy, atol=1e-6, rtol=0)
        assert torch.allclose(actual_forces, expected_forces, atol=1e-5, rtol=0)
    assert graphs
