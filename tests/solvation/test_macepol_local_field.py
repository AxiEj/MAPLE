from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from ase import Atoms


torch = pytest.importorskip("torch")

from maple.function.calculator.mace._macepol_calculator import (
    MACEPolCalculator,
    _LocalReactionFieldProjector,
)
from maple.function.calculator.mace._macepol_long_range import (
    MACEPolarLongRangeEvaluator,
)
from maple.function.route2_smd_profiles import (
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
)


class _FieldRecorder:
    def __init__(self):
        self.values = None

    def set_node_potential_gradient(self, values):
        self.values = values

    @contextmanager
    def use_node_potential_gradient(self, values):
        previous = self.values
        self.values = values
        try:
            yield
        finally:
            self.values = previous

    @contextmanager
    def use_model_field_features(self, values):
        previous = self.values
        self.values = values
        try:
            yield
        finally:
            self.values = previous


class _ProjectorStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("matrix", torch.eye(4, dtype=torch.float64))

    def forward(self, _batch, _positions, _field):
        raise AssertionError("The upstream projector is not used by this test.")


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


class _DisconnectedFieldModel:
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
        return {
            "energy": torch.ones(1),
            "density_coefficients": torch.zeros((1, 4)),
        }


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


class _PositionDependentDensityModel:
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
        assert compute_force is False
        assert compute_stress is False
        assert compute_hessian is False
        positions = batch["positions"]
        values = self.recorder.values
        assert values is not None
        position_features = torch.cat(
            (positions[:, :1].square(), positions),
            dim=1,
        )
        return {
            "energy": positions.square().sum().reshape(1),
            "density_coefficients": 2.0 * values + position_features,
        }


class _FeatureStateModel:
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
            "density_coefficients": values[:, :4],
            "dipole": torch.zeros((1, 3), dtype=values.dtype),
        }


def _calculator_with_model(model, recorder: _FieldRecorder):
    calculator = object.__new__(MACEPolCalculator)
    calculator.device = torch.device("cpu")
    calculator.dtype = torch.float64
    calculator._reaction_projector = recorder
    calculator._long_range_evaluator = (
        MACEPolarLongRangeEvaluator.from_profile(
            MACEPOL_MOLECULAR_REALSPACE_PROFILE
        )
    )
    calculator.model = model
    calculator._batch_dict = lambda _atoms: {"synthetic": torch.tensor(1.0)}
    return calculator


def test_local_reaction_field_context_restores_nested_outer_state():
    projector = _LocalReactionFieldProjector(_ProjectorStub())
    outer = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    inner = -outer

    with projector.use_node_potential_gradient(outer):
        assert projector._node_potential_gradient is outer
        with projector.use_node_potential_gradient(inner):
            assert projector._node_potential_gradient is inner
        assert projector._node_potential_gradient is outer

    assert projector._node_potential_gradient is None


def test_projected_reaction_features_bypass_the_local_jet_matrix():
    projector = _LocalReactionFieldProjector(_ProjectorStub())
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    batch = torch.zeros(2, dtype=torch.long)
    positions = torch.zeros((2, 3), dtype=torch.float64)

    with projector.use_model_field_features(features):
        actual = projector(batch, positions, torch.zeros((1, 4)))

    torch.testing.assert_close(
        actual,
        features.to(dtype=torch.float64),
    )
    assert projector._model_field_features is None


def test_projector_rejects_ambiguous_local_and_preprojected_field_state():
    projector = _LocalReactionFieldProjector(_ProjectorStub())
    local = torch.zeros((1, 4))
    features = torch.zeros((1, 4))

    with projector.use_node_potential_gradient(local):
        with pytest.raises(RuntimeError, match="already installed"):
            with projector.use_model_field_features(features):
                pass


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


def test_polar_output_torch_preserves_preprojected_feature_autograd_graph():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _QuadraticFieldModel(recorder),
        recorder,
    )
    atoms = Atoms("OH", positions=np.zeros((2, 3)))
    features = torch.tensor(
        [[0.2, -0.1, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = calculator.polar_output_torch(
        atoms,
        model_field_features=features,
    )
    derivative = torch.autograd.grad(output["energy"].sum(), features)[0]

    torch.testing.assert_close(derivative, 2.0 * features)
    assert output["energy"].dtype == torch.float64
    assert recorder.values is None


@pytest.mark.parametrize("preprojected", [False, True])
def test_polar_state_copies_immutable_numpy_field_inputs_without_warning(
    preprojected,
):
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _FeatureStateModel(recorder),
        recorder,
    )
    atoms = Atoms("OH", positions=np.zeros((2, 3)))
    values = np.arange(
        16 if preprojected else 8,
        dtype=float,
    ).reshape(2, -1)
    values.setflags(write=False)
    kwargs = (
        {"model_field_features": values}
        if preprojected
        else {
            "node_potential_ev": values[:, 0],
            "node_gradient_ev_per_angstrom": values[:, 1:],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state, _ = calculator.polar_state(atoms, **kwargs)

    assert state.density_coefficients.shape == (2, 4)
    assert not any(
        "not writable" in str(item.message)
        for item in caught
    )


def test_mace_projection_spec_is_read_from_the_loaded_checkpoint():
    upstream = torch.nn.Module()
    upstream.register_buffer(
        "matrix",
        torch.arange(32, dtype=torch.float64).reshape(8, 4),
    )
    calculator = object.__new__(MACEPolCalculator)
    calculator._reaction_projector = _LocalReactionFieldProjector(upstream)
    calculator.graph_longrange_version = "0.4.0"
    calculator.model = SimpleNamespace(
        electric_potential_descriptor=SimpleNamespace(
            feature_basis=SimpleNamespace(
                sigmas=[1.5, 3.0],
                max_l=1,
                normalize="receiver",
            )
        )
    )

    spec = calculator.route2_gto_field_projection_spec()

    assert spec.receiver_sigmas_angstrom == (1.5, 3.0)
    assert spec.receiver_max_l == 1
    assert spec.receiver_normalization == "receiver"
    np.testing.assert_array_equal(
        spec.upstream_matrix,
        np.arange(32, dtype=float).reshape(8, 4),
    )
    assert spec.graph_longrange_version == "0.4.0"
    assert len(spec.upstream_matrix_sha256) == 64


def test_mace_projection_spec_fails_closed_on_unvalidated_graph_longrange():
    upstream = torch.nn.Module()
    upstream.register_buffer(
        "matrix",
        torch.arange(32, dtype=torch.float64).reshape(8, 4),
    )
    calculator = object.__new__(MACEPolCalculator)
    calculator._reaction_projector = _LocalReactionFieldProjector(upstream)
    calculator.graph_longrange_version = "0.4.1"
    calculator.model = SimpleNamespace(
        electric_potential_descriptor=SimpleNamespace(
            feature_basis=SimpleNamespace(
                sigmas=[1.5, 3.0],
                max_l=1,
                normalize="receiver",
            )
        )
    )

    with pytest.raises(RuntimeError, match="pinned to graph-longrange 0.4.0"):
        calculator.route2_gto_field_projection_spec()


def test_intrinsic_energy_field_gradient_matches_exact_quadratic_model():
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

    field_gradient = calculator.intrinsic_energy_field_gradient(
        atoms,
        node_potential_ev=potential,
        node_gradient_ev_per_angstrom=gradient,
    )

    np.testing.assert_allclose(
        field_gradient,
        2.0 * np.concatenate((potential[:, None], gradient), axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert recorder.values is None


def test_intrinsic_energy_field_gradient_rejects_disconnected_model_energy():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _DisconnectedFieldModel(),
        recorder,
    )
    atoms = Atoms("H")

    with pytest.raises(RuntimeError, match="disconnected"):
        calculator.intrinsic_energy_field_gradient(
            atoms,
            node_potential_ev=np.zeros(1),
            node_gradient_ev_per_angstrom=np.zeros((1, 3)),
        )

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


def test_density_position_vjp_matches_exact_fixed_field_coordinate_derivative():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _PositionDependentDensityModel(recorder),
        recorder,
    )
    atoms = Atoms(
        "OH",
        positions=[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
    )
    batch_positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float64,
    )
    calculator._batch_dict = lambda atoms: {"positions": batch_positions}
    potential = np.asarray([0.2, -0.1])
    gradient = np.asarray(
        [[0.3, -0.4, 0.5], [-0.6, 0.7, -0.8]],
    )
    cotangent = np.asarray(
        [[0.7, -0.2, 0.4, -0.6], [-0.3, 0.8, -0.5, 0.9]],
    )

    analytic = calculator.density_position_vjp(
        atoms,
        node_potential_ev=potential,
        node_gradient_ev_per_angstrom=gradient,
        density_cotangent=cotangent,
    )
    expected = cotangent[:, 1:].copy()
    expected[:, 0] += 2.0 * atoms.positions[:, 0] * cotangent[:, 0]

    np.testing.assert_allclose(analytic, expected, rtol=1.0e-13, atol=1.0e-13)
    assert batch_positions.requires_grad is False
    assert recorder.values is None


def test_density_position_vjp_rejects_coordinate_disconnected_density():
    recorder = _FieldRecorder()
    calculator = _calculator_with_model(
        _QuadraticFieldModel(recorder),
        recorder,
    )
    atoms = Atoms("H")
    batch_positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float64,
    )
    calculator._batch_dict = lambda atoms: {"positions": batch_positions}

    with pytest.raises(RuntimeError, match="disconnected"):
        calculator.density_position_vjp(
            atoms,
            node_potential_ev=np.zeros(1),
            node_gradient_ev_per_angstrom=np.zeros((1, 3)),
            density_cotangent=np.zeros((1, 4)),
        )

    assert batch_positions.requires_grad is False
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
