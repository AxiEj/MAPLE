"""Regression coverage for CUDA-FP32 batched legacy-SMD atomic tensions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from maple.solvation.nonpolar.legacy_smd_cds import (
    _atomic_tensions,
    _atomic_tensions_batched,
    _atomic_tensions_reference,
    _switch,
)

SYMBOLS = ("H", "C", "C", "O", "N", "S", "S", "P")
POSITIONS = (
    (0.00, 0.00, 0.00),
    (1.12, 0.07, -0.03),
    (2.58, -0.24, 0.18),
    (2.34, 0.88, 0.11),
    (1.41, 1.66, -0.17),
    (-0.31, 2.13, 0.29),
    (-1.72, 2.47, -0.13),
    (0.82, 3.23, -0.41),
)
SIGMA = {
    1: 1.1,
    6: 2.1,
    7: 3.1,
    8: 4.1,
    15: 5.1,
    16: 6.1,
    101: 0.11,
    102: 0.12,
    103: 0.13,
    104: 0.14,
    105: 0.15,
    106: 0.16,
    107: 0.17,
    110: 0.20,
    111: 0.21,
    114: 0.24,
    115: 0.25,
    116: 0.26,
    118: 0.28,
}
HSIGMA = {1: -0.31, 6: -0.32, 7: -0.33, 8: -0.34, 15: -0.35, 16: -0.36}


def _derivatives(operator):
    positions = torch.tensor(POSITIONS, dtype=torch.float32, requires_grad=True)
    weights = torch.linspace(0.7, 1.2, len(SYMBOLS), dtype=torch.float32)
    direction = torch.linspace(
        -0.4, 0.5, positions.numel(), dtype=torch.float32
    ).reshape_as(positions)
    tensions = operator(SYMBOLS, positions, SIGMA, HSIGMA)
    scalar = (tensions * weights).sum()
    gradient = torch.autograd.grad(scalar, positions, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
    return tensions, gradient, hvp


def test_batched_tensions_preserve_all_element_formulas_and_second_ad():
    reference = _derivatives(_atomic_tensions_reference)
    batched = _derivatives(_atomic_tensions_batched)

    torch.testing.assert_close(batched[0], reference[0], rtol=2.0e-6, atol=2.0e-6)
    torch.testing.assert_close(batched[1], reference[1], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(batched[2], reference[2], rtol=2.0e-4, atol=2.0e-4)
    assert bool(torch.isfinite(batched[0]).all())
    assert bool(torch.isfinite(batched[1]).all())
    assert bool(torch.isfinite(batched[2]).all())


def test_batched_reference_zero_switch_remains_zero_but_checks_cutoff_first():
    symbols = ("H", "F")  # H/F has no RKK reference, hence reference == 0.
    inside = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32)
    for operator in (_atomic_tensions_reference, _atomic_tensions_batched):
        observed = operator(symbols, inside, {1: 1.0, 9: 2.0}, {9: 7.0})
        torch.testing.assert_close(observed, torch.tensor([1.0, 2.0]))

    boundary = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
    for operator in (_atomic_tensions_reference, _atomic_tensions_batched):
        with pytest.raises(RuntimeError, match="cutoff boundary"):
            operator(
                symbols,
                boundary,
                {1: 1.0, 9: 2.0},
                {9: 7.0},
                branch_tolerance=2.0e-8,
            )


def test_near_float32_cutoff_keeps_host_branch_and_sequential_denominator(monkeypatch):
    # float32(1.43) is below the Python-double cutoff 1.33 + 0.10, but equals
    # a once-cast float32 cutoff. The retained path must therefore be active
    # and must form `(distance - 1.33) - 0.10` exactly as the reference does.
    distance = torch.tensor(1.43, dtype=torch.float32)
    assert float(distance) < 1.33 + 0.10
    original_exp = torch.exp
    reference_inputs = []

    def capture_reference(value):
        reference_inputs.append(value.detach().clone())
        return original_exp(value)

    monkeypatch.setattr(torch, "exp", capture_reference)
    reference = _switch(distance, 1.33, 0.10, branch_tolerance=1.0e-11)
    assert len(reference_inputs) == 1
    expected_exponent = reference_inputs[0]

    batched_inputs = []

    def capture_batched(value):
        batched_inputs.append(value.detach().clone())
        return original_exp(value)

    monkeypatch.setattr(torch, "exp", capture_batched)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0]], dtype=torch.float32
    )
    observed = _atomic_tensions_batched(("C", "O"), positions, {6: 1.0, 8: 2.0}, {})

    assert len(batched_inputs) == 1
    torch.testing.assert_close(
        batched_inputs[0],
        expected_exponent.expand_as(batched_inputs[0]),
        rtol=0,
        atol=0,
    )
    assert torch.equal(observed, torch.tensor([1.0, 2.0]))
    assert (
        reference == 0.0
    )  # active exponential underflows, rather than taking zero branch


def test_batched_path_has_no_per_pair_host_scalar_reads(monkeypatch):
    module = __import__(
        "maple.solvation.nonpolar.legacy_smd_cds", fromlist=["_atomic_tensions"]
    )
    original = module._host
    calls = []

    def counted(value):
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(module, "_host", counted)
    positions = torch.tensor(POSITIONS, dtype=torch.float32)
    _atomic_tensions_batched(SYMBOLS, positions, SIGMA, HSIGMA)

    # Only the N fractional-power exact-zero guard remains scalar. All pair
    # topology choices come from one bulk distance snapshot.
    assert calls == [()]


@pytest.mark.parametrize(
    ("device_type", "dtype", "expected"),
    [
        ("cpu", torch.float32, "reference"),
        ("cpu", torch.float64, "reference"),
        ("cuda", torch.float64, "reference"),
        ("cuda", torch.float32, "batched"),
    ],
)
def test_dispatch_changes_only_cuda_float32(monkeypatch, device_type, dtype, expected):
    calls = []
    reference_result = object()
    batched_result = object()

    def reference(*args, **kwargs):
        calls.append("reference")
        return reference_result

    def batched(*args, **kwargs):
        calls.append("batched")
        return batched_result

    module = __import__(
        "maple.solvation.nonpolar.legacy_smd_cds", fromlist=["_atomic_tensions"]
    )
    monkeypatch.setattr(module, "_atomic_tensions_reference", reference)
    monkeypatch.setattr(module, "_atomic_tensions_batched", batched)
    positions = SimpleNamespace(device=SimpleNamespace(type=device_type), dtype=dtype)

    result = _atomic_tensions((), positions, {}, {})

    assert result is (batched_result if expected == "batched" else reference_result)
    assert calls == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_batched_tensions_match_retained_reference():
    positions = torch.tensor(POSITIONS, dtype=torch.float32, device="cuda:0")
    reference = _atomic_tensions_reference(SYMBOLS, positions, SIGMA, HSIGMA)
    observed = _atomic_tensions(SYMBOLS, positions, SIGMA, HSIGMA)

    torch.testing.assert_close(observed, reference, rtol=2.0e-6, atol=2.0e-6)
