"""Regression coverage for the CUDA-FP32 ddPCM operator batch path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import maple.solvation.continuum.torch_ddpcm as torch_ddpcm_module
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM

SYMBOLS = ("C", "O", "N")
RADII = np.array([1.70, 1.52, 1.55])
POSITIONS = np.array([[0.0, 0.0, 0.0], [2.45, 0.25, -0.10], [-0.35, 2.60, 0.40]])
SOURCE = np.zeros((3, 8))
SOURCE[:, (0, 2, 3, 4)] = np.array(
    [
        [0.30, 0.05, -0.03, 0.02],
        [-0.20, -0.02, 0.04, -0.01],
        [-0.10, -0.03, -0.01, -0.01],
    ]
)
DIRECTION = np.array([[0.20, -0.10, 0.05], [-0.15, 0.25, -0.10], [0.05, -0.15, 0.20]])
SOURCE_DIRECTION = np.zeros((3, 8))
SOURCE_DIRECTION[:, (0, 2, 3, 4)] = np.array(
    [
        [0.04, -0.02, 0.01, 0.03],
        [-0.01, 0.03, -0.02, 0.01],
        [-0.03, -0.01, 0.01, -0.04],
    ]
)


def _functional(*, dtype=torch.float32, device="cpu"):
    return TorchDDPCM(
        SYMBOLS,
        RADII,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        dtype=dtype,
        device=device,
        max_dense_bytes=200_000_000,
    )


def _energy_gradient_hvp(functional, *, operator=None, device="cpu"):
    def evaluate():
        positions = torch.tensor(
            POSITIONS, dtype=torch.float32, device=device, requires_grad=True
        )
        source = torch.tensor(SOURCE, dtype=torch.float32, device=device)
        direction = torch.tensor(DIRECTION, dtype=torch.float32, device=device)
        state = functional._state_torch(positions, source, enforce_checks=True)
        gradient = torch.autograd.grad(state.energy, positions, create_graph=True)[0]
        hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
        return state, gradient, hvp

    if operator is None:
        return evaluate()
    implementation = getattr(TorchDDPCM, operator)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(TorchDDPCM, "_operators", implementation)
        return evaluate()


def _source_gradient_and_mixed_coordinate_derivative(functional, *, operator):
    def evaluate():
        positions = torch.tensor(POSITIONS, dtype=torch.float32, requires_grad=True)
        source = torch.tensor(SOURCE, dtype=torch.float32, requires_grad=True)
        source_direction = torch.tensor(SOURCE_DIRECTION, dtype=torch.float32)
        energy = functional._state_torch(positions, source, enforce_checks=True).energy
        source_gradient = torch.autograd.grad(energy, source, create_graph=True)[0]
        mixed = torch.autograd.grad(
            (source_gradient * source_direction).sum(), positions
        )[0]
        return source_gradient, mixed

    implementation = getattr(TorchDDPCM, operator)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(TorchDDPCM, "_operators", implementation)
        return evaluate()


def test_batched_operators_match_reference_across_pair_chunks(monkeypatch):
    functional = _functional()
    positions = torch.tensor(POSITIONS, dtype=torch.float32)
    geometry = functional._geometry(positions)
    monkeypatch.setattr(torch_ddpcm_module, "_CUDA_PAIR_CHUNK_SIZE", 2)

    reference = functional._operators_reference(positions, geometry)
    batched = functional._operators_batched(positions, geometry)

    for observed, expected in zip(batched, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2.0e-6, atol=2.0e-6)


def test_batched_full_scalar_matches_reference_derivatives_across_chunks(monkeypatch):
    functional = _functional()
    monkeypatch.setattr(torch_ddpcm_module, "_CUDA_PAIR_CHUNK_SIZE", 2)
    reference = _energy_gradient_hvp(functional, operator="_operators_reference")
    batched = _energy_gradient_hvp(functional, operator="_operators_batched")

    torch.testing.assert_close(
        batched[0].energy, reference[0].energy, rtol=2.0e-6, atol=2.0e-6
    )
    torch.testing.assert_close(batched[1], reference[1], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(batched[2], reference[2], rtol=2.0e-4, atol=2.0e-4)
    assert batched[0].topology == reference[0].topology
    assert batched[0].assessment_gate_failures == reference[0].assessment_gate_failures
    assert batched[0].assessment_gate_failures == ()


def test_batched_source_gradient_and_mixed_coordinate_derivative_match_reference(
    monkeypatch,
):
    functional = _functional()
    monkeypatch.setattr(torch_ddpcm_module, "_CUDA_PAIR_CHUNK_SIZE", 2)
    reference = _source_gradient_and_mixed_coordinate_derivative(
        functional, operator="_operators_reference"
    )
    batched = _source_gradient_and_mixed_coordinate_derivative(
        functional, operator="_operators_batched"
    )

    torch.testing.assert_close(batched[0], reference[0], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(batched[1], reference[1], rtol=2.0e-4, atol=2.0e-4)


@pytest.mark.parametrize(
    ("device_type", "dtype", "expected"),
    [
        ("cpu", torch.float32, "reference"),
        ("cpu", torch.float64, "reference"),
        ("cuda", torch.float64, "reference"),
        ("cuda", torch.float32, "batched"),
    ],
)
def test_operator_dispatch_changes_only_cuda_float32(
    monkeypatch, device_type, dtype, expected
):
    functional = _functional()
    calls = []
    reference_result = object()
    batched_result = object()

    def reference(self, positions, geometry):
        calls.append("reference")
        return reference_result

    def batched(self, positions, geometry):
        calls.append("batched")
        return batched_result

    monkeypatch.setattr(TorchDDPCM, "_operators_reference", reference)
    monkeypatch.setattr(TorchDDPCM, "_operators_batched", batched)
    positions = SimpleNamespace(
        device=SimpleNamespace(type=device_type),
        dtype=dtype,
    )

    result = functional._operators(positions, ())

    assert result is (batched_result if expected == "batched" else reference_result)
    assert calls == [expected]


def test_batched_full_scalar_keeps_residual_gate_fail_closed(monkeypatch):
    functional = _functional()
    positions = torch.tensor(POSITIONS, dtype=torch.float32)
    source = torch.tensor(SOURCE, dtype=torch.float32)
    monkeypatch.setattr(
        TorchDDPCM,
        "_relative_residual",
        staticmethod(lambda matrix, solution, rhs: 2.0e-5),
    )
    monkeypatch.setattr(TorchDDPCM, "_operators", TorchDDPCM._operators_batched)

    with pytest.raises(RuntimeError, match="residual exceeded"):
        functional.energy_torch(positions, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_stability_uses_exact_gesvd_driver(monkeypatch):
    matrix = torch.eye(3, dtype=torch.float32, device="cuda:0")
    calls = []
    original = torch.linalg.svdvals

    def recording_svdvals(value, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", recording_svdvals)
    diagnostics, failures = TorchDDPCM._assess_matrix_stability("synthetic", matrix)

    assert failures == ()
    assert diagnostics.numerical_rank == 3
    assert calls == [{"driver": "gesvd"}]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_exact_svd_keeps_rank_deficiency_rejection():
    matrix = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float32, device="cuda:0"))

    with pytest.raises(RuntimeError, match="rank deficient or ill-conditioned"):
        TorchDDPCM._validate_matrix_stability("synthetic", matrix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_dispatch_matches_reference_scalar_and_derivatives():
    functional = _functional(device="cuda:0")
    reference = _energy_gradient_hvp(
        functional, operator="_operators_reference", device="cuda:0"
    )
    dispatched = _energy_gradient_hvp(functional, device="cuda:0")

    torch.testing.assert_close(
        dispatched[0].energy, reference[0].energy, rtol=2.0e-6, atol=2.0e-6
    )
    torch.testing.assert_close(dispatched[1], reference[1], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(dispatched[2], reference[2], rtol=2.0e-4, atol=2.0e-4)
    assert dispatched[0].topology == reference[0].topology
    assert dispatched[0].assessment_gate_failures == ()
    positions = torch.tensor(POSITIONS, dtype=torch.float32, device="cuda:0")
    source = torch.tensor(SOURCE, dtype=torch.float32, device="cuda:0")
    receipt = functional.diagnostics(positions, source)["execution_receipt"]
    assert receipt["operator_assembly"] == "bounded-pair-batched-cuda-fp32-v1"
    assert receipt["svd_driver"] == "gesvd"
