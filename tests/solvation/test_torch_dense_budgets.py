from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")


def _budget(**overrides):
    from maple.function.calculator.extra_correction.implicit.torch_dense_budget import (
        DenseTorchBudget,
    )

    values = {
        "max_pair_graph_bytes": 1_000_000,
        "max_pair_work_units": 1_000_000,
        "max_dense_hessian_bytes": 1_000_000,
        "max_dense_hessian_sweeps": 1_000,
    }
    values.update(overrides)
    return DenseTorchBudget(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_pair_graph_bytes", -1),
        ("max_pair_work_units", True),
        ("max_dense_hessian_bytes", float("inf")),
        ("max_dense_hessian_sweeps", float("nan")),
    ],
)
def test_budget_limits_reject_negative_boolean_and_nonfinite_values(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        _budget(**{field: value})


@pytest.mark.parametrize("atom_count", [-1, 0, True, 1.0])
@pytest.mark.parametrize("admission", ["pair", "hessian"])
def test_admission_helpers_require_a_positive_integer_atom_count(atom_count, admission):
    budget = _budget()
    with pytest.raises((TypeError, ValueError), match="atom_count"):
        if admission == "pair":
            budget.admit_pair_graph(atom_count, derivative_order=0, label="test")
        else:
            budget.admit_dense_hessian(atom_count, label="test")


@pytest.mark.parametrize("derivative_order", [True, 1.0, -1, 3])
def test_pair_admission_requires_an_integer_supported_derivative_order(
    derivative_order,
):
    with pytest.raises((TypeError, ValueError), match="derivative_order"):
        _budget().admit_pair_graph(1, derivative_order=derivative_order, label="test")


def _obc_geometry_backend(count: int, budget):
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import TorchOBC2

    backend = TorchOBC2.__new__(TorchOBC2)
    backend.dtype = torch.float64
    backend.device = torch.device("cpu")
    backend._charges = torch.zeros(count, dtype=torch.float64)
    backend._or = torch.full((count,), 0.14, dtype=torch.float64)
    backend._sr = torch.full((count,), 0.112, dtype=torch.float64)
    backend.resource_budget = budget
    backend._provenance = {}
    backend.last_derivative_provenance = None
    backend.parameters = SimpleNamespace(
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
        nonpolar="none",
    )
    return backend


def test_obc_pair_graph_rejected_before_dense_displacement_allocation(monkeypatch):
    backend = _obc_geometry_backend(3, _budget(max_pair_graph_bytes=1))
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    original_sub = torch.Tensor.__sub__

    def reject_dense_pair_subtraction(left, right):
        if left.ndim == 3 or right.ndim == 3:
            raise AssertionError("dense pair allocation was attempted")
        return original_sub(left, right)

    monkeypatch.setattr(torch.Tensor, "__sub__", reject_dense_pair_subtraction)
    with pytest.raises(RuntimeError, match="OBC-II.*pair-graph memory budget"):
        backend._geometry(positions, derivative_order=0)


def test_obc_energy_force_and_hvp_have_explicit_pair_work_charges():
    count = 3
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    backend = _obc_geometry_backend(
        count,
        _budget(max_pair_work_units=2 * count * count),
    )

    backend._geometry(positions, derivative_order=0)
    backend._geometry(positions, derivative_order=1)
    with pytest.raises(RuntimeError, match="OBC-II.*pair-work budget"):
        backend._geometry(positions, derivative_order=2)


def test_obc_dense_hessian_sweep_budget_is_separate_from_hvp():
    count = 2
    backend = _obc_geometry_backend(
        count,
        _budget(max_dense_hessian_sweeps=3),
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    direction = np.ones(3 * count, dtype=np.float64)

    assert np.isfinite(backend.hvp(atoms, direction)).all()
    with pytest.raises(RuntimeError, match="OBC-II.*Hessian sweep budget"):
        backend.hessian(atoms)


def test_dense_hessian_materialization_has_its_own_memory_budget():
    with pytest.raises(RuntimeError, match="dense-Hessian memory budget"):
        _budget(max_dense_hessian_bytes=1).admit_dense_hessian(1, label="test")


def _cha_case(*, requires_grad=False):
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.3, 0.0], [0.2, 1.1, 0.1]],
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
        torch.tensor([0.35, -0.55, 0.2], dtype=torch.float64),
        torch.tensor([1.7, 1.5, 1.2], dtype=torch.float64),
        torch.tensor([0.40, 0.50, 0.48], dtype=torch.float64),
    )


def test_cha_pair_graph_rejected_before_dense_displacement_allocation(monkeypatch):
    from maple.function.calculator.extra_correction.implicit.torch_chagb import (
        cha_polar_from_inverse_born,
    )

    original_sub = torch.Tensor.__sub__

    def reject_dense_pair_subtraction(left, right):
        if left.ndim == 3 or right.ndim == 3:
            raise AssertionError("dense pair allocation was attempted")
        return original_sub(left, right)

    monkeypatch.setattr(torch.Tensor, "__sub__", reject_dense_pair_subtraction)
    with pytest.raises(RuntimeError, match="CHA.*pair-graph memory budget"):
        cha_polar_from_inverse_born(
            *_cha_case(), resource_budget=_budget(max_pair_graph_bytes=1)
        )


def test_cha_differentiable_graph_uses_second_derivative_pair_budget():
    from maple.function.calculator.extra_correction.implicit.torch_chagb import (
        cha_polar_from_inverse_born,
    )

    count = 3
    budget = _budget(max_pair_work_units=count * count)
    cha_polar_from_inverse_born(*_cha_case(), resource_budget=budget)
    with pytest.raises(RuntimeError, match="CHA.*pair-work budget"):
        cha_polar_from_inverse_born(
            *_cha_case(requires_grad=True), resource_budget=budget
        )


def test_default_budgets_do_not_change_cha_numerics():
    from maple.function.calculator.extra_correction.implicit.torch_chagb import (
        cha_polar_from_inverse_born,
    )

    inputs = _cha_case()
    default = cha_polar_from_inverse_born(*inputs)
    explicit = cha_polar_from_inverse_born(*inputs, resource_budget=_budget())

    torch.testing.assert_close(default.polar_kcal_mol, explicit.polar_kcal_mol)
    torch.testing.assert_close(
        default.effective_charges_e, explicit.effective_charges_e
    )
