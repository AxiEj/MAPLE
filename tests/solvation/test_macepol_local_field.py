from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


torch = pytest.importorskip("torch")

from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator


class _FieldRecorder:
    def __init__(self):
        self.values = None

    def set_node_potential_gradient(self, values):
        self.values = values


class _QuadraticFieldModel:
    def __init__(self, recorder: _FieldRecorder):
        self.recorder = recorder

    def __call__(
        self,
        _batch,
        *,
        compute_force,
        compute_stress,
        compute_hessian,
    ):
        assert compute_force is False
        assert compute_stress is False
        assert compute_hessian is False
        values = self.recorder.values
        assert values is not None
        return {
            "energy": values.square().sum().reshape(1),
            "density_coefficients": 3.0 * values,
        }


class _FailingFieldModel:
    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("synthetic model failure")


def _calculator_with_model(model, recorder: _FieldRecorder):
    calculator = object.__new__(MACEPolCalculator)
    calculator.device = torch.device("cpu")
    calculator.dtype = torch.float64
    calculator._reaction_projector = recorder
    calculator.model = model
    calculator._batch_dict = lambda _atoms: {"synthetic": torch.tensor(1.0)}
    return calculator


def test_polar_output_torch_preserves_local_field_autograd_graph():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _QuadraticFieldModel(recorder),
        recorder,
    )
    atoms = Atoms("OH", positions=np.zeros((2, 3)))
    potential = torch.tensor([0.2, -0.1], dtype=torch.float32, requires_grad=True)
    gradient = torch.tensor(
        [[0.3, -0.4, 0.5], [-0.6, 0.7, -0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = calculator.polar_output_torch(
        atoms,
        node_potential_ev=potential,
        node_gradient_ev_per_angstrom=gradient,
    )
    potential_derivative, gradient_derivative = torch.autograd.grad(
        output["energy"].sum(),
        (potential, gradient),
        retain_graph=True,
    )
    density_potential_derivative, density_gradient_derivative = torch.autograd.grad(
        output["density_coefficients"].sum(),
        (potential, gradient),
    )

    torch.testing.assert_close(potential_derivative, 2.0 * potential)
    torch.testing.assert_close(gradient_derivative, 2.0 * gradient)
    torch.testing.assert_close(
        density_potential_derivative,
        torch.full_like(potential, 3.0),
    )
    torch.testing.assert_close(
        density_gradient_derivative,
        torch.full_like(gradient, 3.0),
    )
    assert output["energy"].dtype == torch.float64
    assert recorder.values is None


def test_polar_output_torch_clears_local_field_after_model_failure():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(_FailingFieldModel(), recorder)
    atoms = Atoms("H")

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        calculator.polar_output_torch(
            atoms,
            node_potential_ev=torch.zeros(1, requires_grad=True),
            node_gradient_ev_per_angstrom=torch.zeros(
                (1, 3),
                requires_grad=True,
            ),
        )

    assert recorder.values is None
