"""Precision-policy tests for the original dense :class:`TorchDDPCM`.

These tests intentionally exercise the production implementation in
``maple.solvation.continuum.torch_ddpcm``.  They do not use a matrix-free or
research-only continuum backend, and all derivative checks use Torch AD.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import maple.solvation.continuum.torch_ddpcm as torch_ddpcm_module
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM


POSITIONS = np.array([[0.0, 0.0, 0.0], [2.4, 0.2, -0.1]])
RADII = np.array([1.7, 1.5])
SOURCE = np.zeros((2, 8))
SOURCE[:, (0, 2, 3, 4)] = np.array(
    [[0.4, 0.1, -0.1, 0.2], [-0.4, -0.05, 0.12, -0.18]]
)
DIRECTION = np.array([[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2]])

FP64_BASELINE_ENERGY_EV = -0.7020509554498835
FP64_BASELINE_GRADIENT_EV_PER_A = np.array(
    [
        [0.3112872198172133, 0.024089379261210554, -0.03956457201257505],
        [-0.3112872198172133, -0.024089379261210547, 0.03956457201257505],
    ]
)
FP64_BASELINE_HVP_EV_PER_A2 = np.array(
    [
        [0.4256627120720591, -0.30232582879374376, -0.11761292188897599],
        [-0.42566271207205936, 0.30232582879374376, 0.117612921888976],
    ]
)


def _functional(
    dtype: torch.dtype,
    *,
    lmax: int = 2,
    n_lebedev: int = 50,
    solve_residual_tolerance: float = 1.0e-12,
) -> TorchDDPCM:
    return TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=lmax,
        n_lebedev=n_lebedev,
        dtype=dtype,
        max_dense_bytes=1_000_000_000,
        solve_residual_tolerance=solve_residual_tolerance,
    )


def _tensors(dtype: torch.dtype, *, requires_grad: bool = False):
    positions = torch.tensor(POSITIONS, dtype=dtype, requires_grad=requires_grad)
    source = torch.tensor(SOURCE, dtype=dtype)
    return positions, source


def _raw_state(functional: TorchDDPCM, positions, source):
    return functional._state_torch(positions, source, enforce_checks=False)


class _FloatingOutputDtypeAudit(TorchDispatchMode):
    """Record the floating/complex dtype of every dispatched Torch output."""

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[tuple[str, tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        result = func(*args, **({} if kwargs is None else kwargs))
        leaves, _ = tree_flatten(result)
        for value in leaves:
            if torch.is_tensor(value) and (
                value.is_floating_point() or value.is_complex()
            ):
                self.observed.append((str(func), tuple(value.shape), value.dtype))
        return result


def test_constructor_accepts_only_float32_and_float64() -> None:
    assert _functional(torch.float32).dtype is torch.float32
    assert _functional(torch.float64).dtype is torch.float64
    for rejected in (torch.float16, torch.bfloat16, torch.complex64):
        with pytest.raises(TypeError, match="float32.*float64"):
            _functional(rejected)


def test_default_float64_reproduces_frozen_energy_gradient_and_hvp() -> None:
    functional = TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        max_dense_bytes=100_000_000,
    )
    positions, source = _tensors(torch.float64, requires_grad=True)
    energy = functional.energy_torch(positions, source)
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    direction = torch.tensor(DIRECTION, dtype=torch.float64)
    hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]

    assert functional.dtype is torch.float64
    assert float(energy.detach()) == FP64_BASELINE_ENERGY_EV
    np.testing.assert_array_equal(
        gradient.detach().numpy(), FP64_BASELINE_GRADIENT_EV_PER_A
    )
    np.testing.assert_array_equal(hvp.detach().numpy(), FP64_BASELINE_HVP_EV_PER_A2)


def test_explicit_float64_and_default_have_identical_policy_and_resources() -> None:
    default = TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        max_dense_bytes=100_000_000,
    )
    explicit = TorchDDPCM(
        ("C", "O"),
        RADII,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        dtype=torch.float64,
        max_dense_bytes=100_000_000,
    )
    assert dict(default.numerical_policy) == dict(explicit.numerical_policy)
    for order in (0, 1, 2):
        assert default.estimate_resources(derivative_order=order) == (
            explicit.estimate_resources(derivative_order=order)
        )


def test_float32_dense_resource_estimates_use_four_byte_scalars() -> None:
    fp32 = _functional(torch.float32, lmax=7, n_lebedev=302)
    fp64 = _functional(torch.float64, lmax=7, n_lebedev=302)
    for order in (0, 1, 2):
        estimate32 = fp32.estimate_resources(derivative_order=order)
        estimate64 = fp64.estimate_resources(derivative_order=order)
        assert estimate32.bytes_per_matrix * 2 == estimate64.bytes_per_matrix
        assert estimate32.forward_peak_bytes * 2 == estimate64.forward_peak_bytes
        assert estimate32.first_order_peak_bytes * 2 == (
            estimate64.first_order_peak_bytes
        )
        assert estimate32.second_order_peak_bytes * 2 == (
            estimate64.second_order_peak_bytes
        )


def test_float32_builds_full_dense_operators_and_uses_exactly_two_direct_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = torch.linalg.solve

    def recording_solve(matrix, rhs, *args, **kwargs):
        calls.append((tuple(matrix.shape), tuple(rhs.shape)))
        return original(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", recording_solve)
    state = _raw_state(functional, positions, source)
    basis = len(RADII) * (functional.lmax + 1) ** 2

    assert tuple(state.L.shape) == (basis, basis)
    assert tuple(state.D.shape) == (basis, basis)
    assert calls == [((basis, basis), (basis,)), ((basis, basis), (basis,))]


def test_original_dense_module_has_no_matrix_free_or_research_backend_import() -> None:
    source = inspect.getsource(torch_ddpcm_module)
    assert ".omx" not in source
    assert "matrixfree" not in source.lower()
    assert "gmres" not in source.lower()


def test_float32_raw_state_keeps_every_floating_tensor_in_float32() -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    state = _raw_state(functional, positions, source)
    graph_tensors = (
        state.energy,
        state.L,
        state.D,
        state.rhs,
        state.dielectric_rhs,
        state.solution,
    )
    assert all(value.dtype is torch.float32 for value in graph_tensors)
    assert not any(
        value.dtype in (torch.float64, torch.complex128) for value in graph_tensors
    )


def test_float32_forward_gradient_and_hvp_never_dispatch_float64_outputs() -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32, requires_grad=True)
    direction = torch.tensor(DIRECTION, dtype=torch.float32)
    audit = _FloatingOutputDtypeAudit()

    with audit:
        energy = _raw_state(functional, positions, source).energy
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]

    assert audit.observed
    assert all(
        dtype in (torch.float32, torch.complex64) for _, _, dtype in audit.observed
    ), [
        entry
        for entry in audit.observed
        if entry[2] in (torch.float64, torch.complex128)
    ]
    assert energy.dtype is torch.float32
    assert gradient.dtype is torch.float32
    assert hvp.dtype is torch.float32


def test_energy_rejects_input_dtype_mismatch() -> None:
    functional = _functional(torch.float32)
    positions = torch.tensor(POSITIONS, dtype=torch.float32)
    source = torch.tensor(SOURCE, dtype=torch.float64)
    with pytest.raises(ValueError, match="continuum scalar contract"):
        functional.energy_torch(positions, source)


def test_float32_residual_gate_remains_exactly_one_e_minus_twelve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    monkeypatch.setattr(
        TorchDDPCM, "_relative_residual", staticmethod(lambda matrix, solution, rhs: 2e-12)
    )

    with pytest.raises(RuntimeError, match="residual exceeded"):
        functional.energy_torch(positions, source)
    with pytest.raises(RuntimeError, match="residual exceeded"):
        functional.diagnostics(positions, source)
    raw = functional.diagnostics(positions, source, allow_unqualified=True)
    assert raw["scientific_admitted"] is False
    assert "solve_residual" in raw["failed_gates"]
    assert raw["reps_relative_residual"] == 2e-12
    assert raw["l_relative_residual"] == 2e-12
    assert np.isfinite(raw["energy_eV"])


@pytest.mark.parametrize("invalid", (1, None, "false"))
def test_unqualified_diagnostics_requires_an_actual_boolean(invalid) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    with pytest.raises(TypeError, match="allow_unqualified.*bool"):
        functional.diagnostics(positions, source, allow_unqualified=invalid)


def test_unqualified_diagnostics_false_is_strict_and_true_is_observational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    monkeypatch.setattr(
        TorchDDPCM, "_relative_residual", staticmethod(lambda matrix, solution, rhs: 2e-12)
    )
    with pytest.raises(RuntimeError, match="residual exceeded"):
        functional.diagnostics(positions, source, allow_unqualified=False)
    raw = functional.diagnostics(positions, source, allow_unqualified=True)
    assert raw["assessment_gate_pass"] is False
    assert raw["scientific_admitted"] is False


def test_unqualified_diagnostics_reports_the_actual_dense_execution_receipt() -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    receipt = functional.diagnostics(
        positions, source, allow_unqualified=True
    )["execution_receipt"]
    assert receipt == {
        "contract": "torch-ddpcm-dense-execution-receipt-v1",
        "observation_mode": "unqualified-raw",
        "allow_unqualified": True,
        "dtype": "torch.float32",
        "device": "cpu",
        "lmax": 1,
        "n_lebedev": 194,
        "grid_sha256": (
            "5c5d366f54a23faf3e1a4611fe02d432e51804bb09f5d779283a94ed8e19bf89"
        ),
        "eta": 0.1,
        "dielectric": 78.39,
        "backend": "dense-direct-two-solve",
        "storage": "full-dense",
        "source_tile": "not-applicable-full-dense",
        "linear_solver": "torch.linalg.solve",
        "linear_solve_count": 2,
        "solve_residual_tolerance": 1.0e-12,
        "effective_solve_residual_limit": 1.0e-12,
        "topology_margin": 1.0e-8,
        "L_shape": [8, 8],
        "D_shape": [8, 8],
        "positions_shape": [2, 3],
        "source_shape": [2, 8],
        "positions_dtype": "torch.float32",
        "source_dtype": "torch.float32",
        "positions_device": "cpu",
        "source_device": "cpu",
        "state_tensors": {
            "energy": {"shape": [], "dtype": "torch.float32", "device": "cpu"},
            "L": {"shape": [8, 8], "dtype": "torch.float32", "device": "cpu"},
            "D": {"shape": [8, 8], "dtype": "torch.float32", "device": "cpu"},
            "rhs": {"shape": [8], "dtype": "torch.float32", "device": "cpu"},
            "dielectric_rhs": {
                "shape": [8],
                "dtype": "torch.float32",
                "device": "cpu",
            },
            "solution": {
                "shape": [8],
                "dtype": "torch.float32",
                "device": "cpu",
            },
        },
        "autocast_policy": "forbidden",
        "autocast_active": False,
        "tf32_policy": "forbidden",
        "tf32_active": False,
    }


def test_unqualified_observer_does_not_hide_a_dense_solve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)

    def fail(*args, **kwargs):
        raise RuntimeError("injected solve failure")

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(RuntimeError, match="dense linear solve failed"):
        functional.diagnostics(positions, source, allow_unqualified=True)


def test_unqualified_observer_rejects_a_nonfinite_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    monkeypatch.setattr(
        torch, "dot", lambda left, right: left.new_tensor(float("nan"))
    )
    with pytest.raises(RuntimeError, match="non-finite energy"):
        functional.diagnostics(positions, source, allow_unqualified=True)


def test_float32_policy_forbids_autocast_and_tf32() -> None:
    policy = dict(_functional(torch.float32).numerical_policy)
    assert policy["autocast"] == "forbidden"
    assert policy["tf32"] == "forbidden"


def test_float32_execution_rejects_active_cpu_autocast() -> None:
    functional = _functional(torch.float32, lmax=1, n_lebedev=194)
    positions, source = _tensors(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            functional.diagnostics(positions, source, allow_unqualified=True)


def test_float32_energy_gradient_and_hvp_are_analytic_and_close_to_float64() -> None:
    results = {}
    for dtype in (torch.float32, torch.float64):
        functional = _functional(dtype, lmax=1, n_lebedev=194)
        positions, source = _tensors(dtype, requires_grad=True)
        energy = _raw_state(functional, positions, source).energy
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        direction = torch.tensor(DIRECTION, dtype=dtype)
        hvp = torch.autograd.grad((gradient * direction).sum(), positions)[0]
        results[dtype] = (
            float(energy.detach()),
            gradient.detach().cpu().double().numpy(),
            hvp.detach().cpu().double().numpy(),
        )
        assert energy.dtype is dtype
        assert gradient.dtype is dtype
        assert hvp.dtype is dtype
        assert np.isfinite(results[dtype][0])
        assert np.all(np.isfinite(results[dtype][1]))
        assert np.all(np.isfinite(results[dtype][2]))

    fp32, fp64 = results[torch.float32], results[torch.float64]
    assert abs(fp32[0] - fp64[0]) <= 5.0e-5
    np.testing.assert_allclose(fp32[1], fp64[1], rtol=0.0, atol=5.0e-4)
    np.testing.assert_allclose(fp32[2], fp64[2], rtol=0.0, atol=5.0e-4)


def test_dtype_is_bound_into_configuration_identity() -> None:
    fp32 = _functional(torch.float32)
    fp64 = _functional(torch.float64)
    assert fp32.configuration_sha256() != fp64.configuration_sha256()
    assert fp32.numerical_policy["dtype"] == "torch.float32"
    assert fp64.numerical_policy["dtype"] == "torch.float64"
