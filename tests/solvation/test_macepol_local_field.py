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


class _PositionFieldModel:
    def __init__(self, recorder: _FieldRecorder):
        self.recorder = recorder

    def __call__(
        self,
        batch,
        *,
        compute_force,
        compute_stress,
        compute_hessian,
    ):
        assert compute_stress is False
        assert compute_hessian is False
        positions = batch["positions"]
        values = self.recorder.values
        assert values is not None
        energy = (
            positions.square().sum()
            + (positions * values[:, 1:]).sum()
            + (positions[:, 0] * values[:, 0]).sum()
        ).reshape(1)
        forces = None
        if compute_force:
            forces = -torch.autograd.grad(
                energy.sum(),
                positions,
                retain_graph=True,
            )[0]
        return {
            "energy": energy,
            "density_coefficients": values,
            "dipole": torch.zeros((1, 3), dtype=positions.dtype),
            "forces": forces,
        }


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


def test_density_response_linearization_returns_matching_jvp_and_vjp():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _QuadraticFieldModel(recorder),
        recorder,
    )
    atoms = Atoms("OH", positions=np.zeros((2, 3)))
    potential = np.asarray([0.2, -0.1])
    gradient = np.asarray(
        [[0.3, -0.4, 0.5], [-0.6, 0.7, -0.8]],
    )
    field_direction = np.asarray(
        [[0.9, -0.8, 0.7, -0.6], [0.5, -0.4, 0.3, -0.2]],
    )
    density_cotangent = np.asarray(
        [[-0.3, 0.2, -0.1, 0.4], [0.5, -0.6, 0.7, -0.8]],
    )
    linearization = calculator.linearize_density_response(
        atoms,
        node_potential_ev=potential,
        node_gradient_ev_per_angstrom=gradient,
    )

    np.testing.assert_allclose(
        linearization.jvp(field_direction),
        3.0 * field_direction,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        linearization.vjp(density_cotangent),
        3.0 * density_cotangent,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert recorder.values is None


def test_polar_state_returns_fixed_local_field_force_matching_energy_difference():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _PositionFieldModel(recorder),
        recorder,
    )
    calculator._batch_dict = lambda atoms: {
        "positions": torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            requires_grad=True,
        )
    }
    atoms = Atoms(
        "OH",
        positions=[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
    )
    potential = np.asarray([0.2, -0.1])
    gradient = np.asarray(
        [[0.3, -0.4, 0.5], [-0.6, 0.7, -0.8]],
    )

    state, _ = calculator.polar_state(
        atoms,
        node_potential_ev=potential,
        node_gradient_ev_per_angstrom=gradient,
        compute_forces=True,
    )
    analytic = state.fixed_field_forces_ev_per_angstrom
    assert analytic is not None

    # This exact quadratic synthetic oracle permits a much tighter tolerance
    # than the separate float64 real-model canary; it is not a production
    # MACE-POLAR force-accuracy threshold.
    step_angstrom = 1.0e-6
    finite_difference = np.zeros_like(atoms.positions)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += step_angstrom
            minus.positions[atom_index, axis] -= step_angstrom
            plus_state, _ = calculator.polar_state(
                plus,
                node_potential_ev=potential,
                node_gradient_ev_per_angstrom=gradient,
            )
            minus_state, _ = calculator.polar_state(
                minus,
                node_potential_ev=potential,
                node_gradient_ev_per_angstrom=gradient,
            )
            finite_difference[atom_index, axis] = -(
                plus_state.energy_ev - minus_state.energy_ev
            ) / (2.0 * step_angstrom)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=1.0e-9,
        atol=1.0e-9,
    )


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
