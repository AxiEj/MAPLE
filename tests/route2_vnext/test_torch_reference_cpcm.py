from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.solvation.continuum.torch_reference_cpcm import (  # noqa: E402
    FixedCPCMTopology,
    TorchFixedTopologyAmplitudeSWIGCPCMOracle,
    TorchReferenceCPCM,
    TorchSyntheticDenseCPCM,
)


def _oracle():
    atom_positions = np.array(
        [[-0.65, 0.10, -0.05], [0.70, -0.08, 0.12]], dtype=np.float64
    )
    topology = FixedCPCMTopology(
        reference_atom_positions=atom_positions,
        node_positions=np.array(
            [
                [-1.45, 0.10, -0.05],
                [-0.65, 0.90, -0.05],
                [-0.65, 0.10, 0.75],
                [1.50, -0.08, 0.12],
                [0.70, -0.88, 0.12],
                [0.70, -0.08, -0.68],
            ],
            dtype=np.float64,
        ),
        node_weights=np.array([0.8, 1.1, 0.9, 1.2, 0.7, 1.0]),
        node_widths=np.array([0.25, 0.28, 0.24, 0.27, 0.26, 0.29]),
        node_owners=np.array([0, 0, 0, 1, 1, 1]),
        source_widths=np.array([0.31, 0.34]),
        epsilon=40.0,
        spd_margin=0.75,
    )
    positions = torch.tensor(atom_positions, dtype=torch.float64, requires_grad=True)
    source = torch.tensor([0.42, -0.31], dtype=torch.float64, requires_grad=True)
    return TorchSyntheticDenseCPCM(topology), positions, source


def test_module_import_is_torch_lazy():
    module_path = (
        Path(__file__).parents[2] / "maple/solvation/continuum/torch_reference_cpcm.py"
    )
    code = r"""
import builtins
import importlib.util
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('eager torch import')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import maple.solvation
spec = importlib.util.spec_from_file_location('_torch_reference_cpcm_lazy_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(module_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_topology_is_detached_immutable_ordered_and_hash_stable():
    oracle, _, _ = _oracle()
    topology = oracle.topology
    assert topology.node_count == 6
    assert topology.atom_count == 2
    assert topology.node_owners.tolist() == [0, 0, 0, 1, 1, 1]
    assert len(topology.topology_hash) == 64
    with pytest.raises(ValueError):
        topology.node_positions[0, 0] = 999.0
    with pytest.raises(FrozenInstanceError):
        topology.epsilon = 2.0
    clone, _, _ = _oracle()
    assert clone.topology.topology_hash == topology.topology_hash
    assert TorchReferenceCPCM is TorchSyntheticDenseCPCM
    assert "synthetic" in TorchSyntheticDenseCPCM.scalar_id
    assert TorchSyntheticDenseCPCM.capability_admitted is False


def test_spd_reciprocal_half_coupling_and_dense_source_derivative():
    oracle, positions, source = _oracle()
    state = oracle.solve(positions, source)
    eigenvalues = torch.linalg.eigvalsh(state.A)
    assert float(torch.min(eigenvalues).detach()) > 0.0
    source_energy = 0.5 * torch.dot(source, state.field)
    surface_energy = 0.5 * torch.dot(state.surface_charge, state.potential)
    torch.testing.assert_close(state.energy, source_energy, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(state.energy, surface_energy, atol=1e-13, rtol=1e-13)
    assert float(state.energy.detach()) < 0.0

    _, source_gradient = oracle.autograd_energy_gradients(positions, source)
    torch.testing.assert_close(source_gradient, state.field, atol=2e-13, rtol=2e-13)
    dense = oracle.dense_source_jacobian(positions)
    torch.testing.assert_close(dense, dense.T, atol=1e-13, rtol=1e-13)
    direction = torch.tensor([0.17, -0.23], dtype=torch.float64)
    _, field_jvp = oracle.field_jvp(
        positions,
        source,
        torch.zeros_like(positions),
        direction,
    )
    torch.testing.assert_close(field_jvp, dense @ direction, atol=2e-13, rtol=2e-13)


def test_unrolled_and_dense_implicit_coordinate_gradient_match_finite_difference():
    oracle, positions, source = _oracle()
    autograd_coordinate, _ = oracle.autograd_energy_gradients(positions, source)
    implicit_coordinate = oracle.implicit_coordinate_gradient(positions, source)
    torch.testing.assert_close(
        autograd_coordinate, implicit_coordinate, atol=5e-12, rtol=5e-12
    )

    direction = torch.tensor(
        [[0.11, -0.07, 0.05], [-0.03, 0.09, -0.04]], dtype=torch.float64
    )
    step = 2.0e-5
    plus = oracle.energy((positions.detach() + step * direction), source.detach())
    minus = oracle.energy((positions.detach() - step * direction), source.detach())
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = torch.sum(autograd_coordinate * direction)
    torch.testing.assert_close(analytic, finite_difference, atol=2e-10, rtol=2e-9)


def test_joint_field_jvp_vjp_dot_product_identity():
    oracle, positions, source = _oracle()
    d_positions = torch.tensor(
        [[0.02, -0.04, 0.01], [-0.03, 0.05, -0.02]], dtype=torch.float64
    )
    d_source = torch.tensor([0.07, -0.09], dtype=torch.float64)
    cotangent = torch.tensor([-0.13, 0.19], dtype=torch.float64)
    _, jvp = oracle.field_jvp(positions, source, d_positions, d_source)
    position_vjp, source_vjp = oracle.field_vjp(positions, source, cotangent)
    lhs = torch.dot(cotangent, jvp)
    rhs = torch.sum(position_vjp * d_positions) + torch.dot(source_vjp, d_source)
    torch.testing.assert_close(lhs, rhs, atol=2e-12, rtol=2e-12)


def test_joint_energy_hessian_is_symmetric_to_1e_8():
    oracle, positions, source = _oracle()
    direction_a = (
        torch.tensor([[0.04, -0.02, 0.03], [-0.01, 0.05, -0.06]], dtype=torch.float64),
        torch.tensor([0.08, -0.03], dtype=torch.float64),
    )
    direction_b = (
        torch.tensor([[-0.03, 0.07, -0.02], [0.06, -0.04, 0.01]], dtype=torch.float64),
        torch.tensor([-0.05, 0.09], dtype=torch.float64),
    )
    _, hvp_a = oracle.energy_hvp(positions, source, *direction_a)
    _, hvp_b = oracle.energy_hvp(positions, source, *direction_b)
    a_h_b = torch.sum(direction_a[0] * hvp_b[0]) + torch.dot(direction_a[1], hvp_b[1])
    b_h_a = torch.sum(direction_b[0] * hvp_a[0]) + torch.dot(direction_b[1], hvp_a[1])
    assert abs(float(a_h_b - b_h_a)) <= 1.0e-8


def test_cold_replay_is_bitwise_deterministic():
    first, positions, source = _oracle()
    second, positions_2, source_2 = _oracle()
    state_1 = first.solve(positions, source)
    state_2 = second.solve(positions_2, source_2)
    assert state_1.topology_hash == state_2.topology_hash
    for name in (
        "node_positions",
        "A",
        "B",
        "potential",
        "surface_charge",
        "field",
        "energy",
    ):
        assert torch.equal(getattr(state_1, name), getattr(state_2, name))


@pytest.mark.parametrize(
    "case, expected",
    [
        ("positions_dtype", TypeError),
        ("source_dtype", TypeError),
        ("positions_nonfinite", ValueError),
        ("source_nonfinite", ValueError),
    ],
)
def test_runtime_dtype_and_nonfinite_inputs_fail_closed(case, expected):
    oracle, positions, source = _oracle()
    if case == "positions_dtype":
        positions = positions.detach().to(torch.float32)
    elif case == "source_dtype":
        source = source.detach().to(torch.float32)
    elif case == "positions_nonfinite":
        positions = positions.detach().clone()
        positions[0, 0] = torch.nan
    else:
        source = source.detach().clone()
        source[0] = torch.inf
    with pytest.raises(expected):
        oracle.solve(positions, source)


def test_invalid_topology_fails_closed():
    atoms = np.zeros((2, 3))
    nodes = np.zeros((2, 3))
    with pytest.raises(ValueError, match="positive"):
        FixedCPCMTopology(atoms, nodes, [1.0, 0.0], [0.2, 0.2], [0, 1], [0.3, 0.3])
    with pytest.raises(ValueError, match="owner"):
        FixedCPCMTopology(atoms, nodes, [1.0, 1.0], [0.2, 0.2], [0, 2], [0.3, 0.3])
    with pytest.raises(ValueError, match="integers"):
        FixedCPCMTopology(atoms, nodes, [1.0, 1.0], [0.2, 0.2], [0, 0.5], [0.3, 0.3])
    with pytest.raises(ValueError, match="same node count"):
        FixedCPCMTopology(atoms, nodes, [1.0], [0.2, 0.2], [0, 1], [0.3, 0.3])


def _production_pair():
    from maple.solvation.continuum.fixed_topology_cpcm import FixedTopologyCPCMBackend

    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    unit_sphere = np.column_stack((directions, np.full(6, 1.0 / 6.0)))
    dielectric = 32.5
    backend = FixedTopologyCPCMBackend(
        ("H", "H"),
        np.array([1.35, 1.45]),
        dielectric=dielectric,
        unit_sphere=unit_sphere,
        switching_constant=1.15,
        runtime_version="torch-parity-grid-v1",
    )
    return backend, dielectric


def _parity_oracle(backend, dielectric, positions):
    snapshot = backend.surface_provider.build_state(positions)
    return TorchFixedTopologyAmplitudeSWIGCPCMOracle(
        snapshot,
        dielectric=dielectric,
        switching_constant=backend.surface_provider.switching_constant,
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[-0.95, 0.02, -0.01], [0.95, -0.02, 0.01]]),
        np.array([[-0.80, 0.06, 0.03], [0.86, -0.04, -0.02]]),
        np.array([[-1.08, -0.05, 0.04], [1.01, 0.08, -0.03]]),
    ],
)
def test_amplitude_swig_oracle_matches_production_energy_field_and_source_maps(
    positions,
):
    from maple.function.calculator.extra_correction.implicit.gto_density import (
        point_multipole_potential,
    )

    backend, dielectric = _production_pair()
    oracle = _parity_oracle(backend, dielectric, positions)
    source_np = np.array(
        [[0.31, -0.08, 0.05, 0.11], [-0.27, 0.04, -0.09, 0.03]], dtype=float
    )
    direction_np = np.array(
        [[-0.07, 0.02, 0.01, -0.03], [0.06, -0.05, 0.04, 0.02]], dtype=float
    )
    cotangent_np = np.array(
        [[0.12, -0.04, 0.03, 0.08], [-0.09, 0.05, -0.02, 0.07]], dtype=float
    )
    position_tensor = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    source_tensor = torch.tensor(source_np, dtype=torch.float64, requires_grad=True)
    state = oracle.solve(position_tensor, source_tensor)
    production = backend.build_state(positions, source_np)
    _, _, torch_hessian, torch_b = oracle.assemble(position_tensor)
    production_response = backend._legacy_response(positions)
    np.testing.assert_allclose(
        torch_hessian.detach().numpy(),
        production_response.surface_hessian,
        rtol=2e-13,
        atol=2e-13,
    )
    basis = np.eye(source_np.size).reshape(source_np.size, *source_np.shape)
    production_b = np.column_stack(
        [
            point_multipole_potential(
                production_response.surface_points_bohr, positions, vector
            )
            for vector in basis
        ]
    )
    np.testing.assert_allclose(
        torch_b.detach().numpy(), production_b, rtol=2e-13, atol=2e-13
    )

    assert oracle.capability_admitted is False
    assert oracle.hessian_capability_admitted is False
    assert oracle.field_to_source_indices == (0, 2, 3, 1)
    assert state.topology_hash == production.surface.topology_hash
    np.testing.assert_allclose(
        state.field.detach().numpy(), production.reaction_field, rtol=2e-12, atol=2e-12
    )
    assert float(state.energy.detach()) == pytest.approx(
        production.polarization_energy_ev, rel=2e-12, abs=2e-12
    )
    q_field = state.field[:, (0, 2, 3, 1)]
    torch.testing.assert_close(
        state.energy, 0.5 * torch.sum(source_tensor * q_field), atol=2e-12, rtol=2e-12
    )
    _, source_energy_gradient = oracle.energy_gradients(
        position_tensor, source_tensor, create_graph=False
    )
    torch.testing.assert_close(source_energy_gradient, q_field, atol=3e-12, rtol=3e-12)

    jvp = oracle.source_jvp(
        position_tensor,
        source_tensor,
        torch.tensor(direction_np, dtype=torch.float64),
    )
    np.testing.assert_allclose(
        jvp.detach().numpy(),
        backend.source_jvp(positions, source_np, direction_np),
        rtol=3e-12,
        atol=3e-12,
    )
    vjp = oracle.source_vjp(
        position_tensor,
        source_tensor,
        torch.tensor(cotangent_np, dtype=torch.float64),
    )
    np.testing.assert_allclose(
        vjp.detach().numpy(),
        backend.source_vjp(positions, source_np, cotangent_np),
        rtol=3e-12,
        atol=3e-12,
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[-0.95, 0.02, -0.01], [0.95, -0.02, 0.01]]),
        np.array([[-0.80, 0.06, 0.03], [0.86, -0.04, -0.02]]),
        np.array([[-1.08, -0.05, 0.04], [1.01, 0.08, -0.03]]),
    ],
)
def test_amplitude_swig_oracle_coordinate_energy_gradient_matches_production(positions):
    backend, dielectric = _production_pair()
    oracle = _parity_oracle(backend, dielectric, positions)
    source_np = np.array(
        [[0.31, -0.08, 0.05, 0.11], [-0.27, 0.04, -0.09, 0.03]], dtype=float
    )
    position_tensor = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    source_tensor = torch.tensor(source_np, dtype=torch.float64, requires_grad=True)
    coordinate_gradient, _ = oracle.energy_gradients(
        position_tensor, source_tensor, create_graph=False
    )
    # E=0.5<c,Pc>; production coordinate_vjp differentiates <y,Pc>.
    energy_field_cotangent = 0.5 * source_np[:, [0, 3, 1, 2]]
    production_gradient = backend.coordinate_vjp(
        positions, source_np, energy_field_cotangent
    ).reshape(2, 3)
    np.testing.assert_allclose(
        coordinate_gradient.detach().numpy(),
        production_gradient,
        rtol=2e-9,
        atol=2e-9,
    )


def test_amplitude_swig_oracle_raw_hvp_is_symmetric_but_not_admitted():
    backend, dielectric = _production_pair()
    positions = np.array([[-0.86, 0.04, 0.02], [0.90, -0.03, -0.01]])
    oracle = _parity_oracle(backend, dielectric, positions)
    r = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(
        [[0.29, -0.06, 0.04, 0.08], [-0.25, 0.03, -0.07, 0.02]],
        dtype=torch.float64,
        requires_grad=True,
    )
    a = (
        torch.tensor([[0.02, -0.03, 0.01], [-0.01, 0.04, -0.02]], dtype=torch.float64),
        torch.tensor(
            [[0.03, -0.02, 0.01, 0.04], [-0.01, 0.02, -0.03, 0.01]], dtype=torch.float64
        ),
    )
    b = (
        torch.tensor([[-0.03, 0.01, 0.02], [0.04, -0.02, 0.01]], dtype=torch.float64),
        torch.tensor(
            [[-0.02, 0.04, -0.01, 0.03], [0.02, -0.01, 0.04, -0.03]],
            dtype=torch.float64,
        ),
    )
    _, ha = oracle.energy_hvp(r, c, *a)
    _, hb = oracle.energy_hvp(r, c, *b)
    a_h_b = torch.sum(a[0] * hb[0]) + torch.sum(a[1] * hb[1])
    b_h_a = torch.sum(b[0] * ha[0]) + torch.sum(b[1] * ha[1])
    assert abs(float((a_h_b - b_h_a).detach())) <= 1.0e-8
    assert oracle.hessian_capability_admitted is False


def test_parity_oracle_is_sealed_content_addressed_and_snapshot_validated():
    backend, dielectric = _production_pair()
    positions = np.array([[-0.86, 0.04, 0.02], [0.90, -0.03, -0.01]])
    snapshot = backend.surface_provider.build_state(positions)
    oracle = _parity_oracle(backend, dielectric, positions)
    assert len(oracle.configuration_sha256) == 64
    assert len(oracle.provenance_sha256) == 64
    assert (
        oracle.topology_input_scope
        == "validated-production-snapshot-not-independent-grid-asset"
    )
    with pytest.raises(AttributeError, match="immutable"):
        oracle.dielectric = 2.0

    oracle._weights.setflags(write=True)
    oracle._weights[0] *= 2.0
    with pytest.raises(RuntimeError, match="configuration fingerprint"):
        _ = oracle.configuration_sha256

    class SnapshotLookalike:
        pass

    lookalike = SnapshotLookalike()
    for name in snapshot.__dataclass_fields__:
        setattr(lookalike, name, getattr(snapshot, name))
    with pytest.raises(TypeError, match="validated FixedTopologySurfaceSnapshot"):
        TorchFixedTopologyAmplitudeSWIGCPCMOracle(
            lookalike,
            dielectric=dielectric,
            switching_constant=backend.surface_provider.switching_constant,
        )

    with pytest.raises(ValueError, match="topology_hash"):
        TorchFixedTopologyAmplitudeSWIGCPCMOracle(
            replace(snapshot, topology_hash="0" * 64),
            dielectric=dielectric,
            switching_constant=backend.surface_provider.switching_constant,
        )
