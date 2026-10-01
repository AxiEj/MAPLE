"""Research-only contracts against the unchanged dense ddPCM operator."""

import math

import numpy as np
import pytest
import torch

from maple.solvation.continuum.ddpcm_response_operators import DDPCMResponseOperators
from streamed_ddpcm import StreamedDDPCM, solve_diagnostic


def fixture(n=3, lmax=2, n_lebedev=50, tile=2):
    symbols = ("C", "O", "H")[:n]
    radii = np.array([1.7, 1.5, 1.2])[:n]
    r = torch.tensor(
        [[0, 0, 0], [1.8, 0.23, 0.12], [-0.27, 1.45, 0.32]][:n],
        dtype=torch.float64,
    )
    s = torch.tensor(
        [[0.4, 0.1, -0.1, 0.2], [-0.5, -0.05, 0.12, -0.18], [0.1, 0.02, -0.03, 0.04]][
            :n
        ],
        dtype=torch.float64,
    )
    kwargs = dict(dielectric=78.39, lmax=lmax, n_lebedev=n_lebedev)
    dense = DDPCMResponseOperators(symbols, radii, **kwargs).build(r, s)
    streamed = StreamedDDPCM(symbols, radii, r, source_tile=tile, **kwargs)
    return dense, streamed, r, s


def chain_fixture(n, tile=2):
    index = np.arange(n, dtype=float)
    r = torch.tensor(
        np.column_stack(
            (1.63 * index, 0.31 * np.sin(1.37 * index), 0.23 * np.cos(0.79 * index))
        ),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(901)
    source = 0.025 * torch.randn((n, 4), generator=generator, dtype=torch.float64)
    source[:, 0] -= source[:, 0].mean()
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    old = DDPCMResponseOperators(("C",) * n, np.full(n, 1.7), **kwargs).build(r, source)
    new = StreamedDDPCM(("C",) * n, np.full(n, 1.7), r, source_tile=tile, **kwargs)
    return old, new, r, source


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("tile", [1, 2, 5])
def test_geometry_rhs_and_topology(n, tile):
    dense, op, _, source = fixture(n, tile=tile)
    rhs, psi = op.rhs_psi(source)
    torch.testing.assert_close(rhs, dense.rhs, atol=2e-12, rtol=2e-13)
    torch.testing.assert_close(psi, dense.psi, atol=2e-13, rtol=2e-13)
    for name in ("ui", "gi", "fi"):
        torch.testing.assert_close(
            getattr(op, name), getattr(dense.geometry, name), atol=0, rtol=0
        )
    assert op.topology == dense.geometry.topology


@pytest.mark.parametrize("kind", ["L", "D", "A", "C"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("columns", [1, 3])
def test_actions_and_transposes(kind, transpose, columns):
    dense, op, _, _ = fixture()
    identity = torch.eye(op.dimension, dtype=torch.float64)
    matrices = {
        "L": dense.L,
        "D": dense.D,
        "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * identity - dense.D,
        "C": 2 * math.pi * identity - dense.D,
    }
    generator = torch.Generator().manual_seed(42)
    x = torch.randn((op.dimension, columns), generator=generator, dtype=torch.float64)
    if columns == 1:
        x = x[:, 0]
    matrix = matrices[kind].T if transpose else matrices[kind]
    torch.testing.assert_close(
        op.apply(kind, x, transpose=transpose), matrix @ x, atol=2e-11, rtol=2e-12
    )


def test_full_resolution_action():
    dense, op, _, _ = fixture(2, lmax=15, n_lebedev=1202, tile=2)
    generator = torch.Generator().manual_seed(512)
    x = torch.randn(op.dimension, generator=generator, dtype=torch.float64)
    for kind, matrix in (("L", dense.L), ("D", dense.D)):
        for transpose in (False, True):
            expected = (matrix.T if transpose else matrix) @ x
            torch.testing.assert_close(
                op.apply(kind, x, transpose=transpose), expected, atol=2e-11, rtol=2e-12
            )


def test_no_retained_dense_or_pair_geometry():
    _, op, _, _ = fixture()
    assert op.storage_report()["dense_operator_bytes"] == 0
    assert op.storage_report()["retained_pair_grid_bytes"] == 0
    forbidden = {(op.dimension, op.dimension), (3, 50, 3), (3, 50, 3, 3)}
    # Grid nodes (N,G,3) are permitted: the last 3 is Cartesian, not a source axis.
    for name, tensor in op.retained_tensors():
        if name != "nodes":
            assert tuple(tensor.shape) not in forbidden


def test_mutated_input_does_not_change_geometry():
    _, op, r, s = fixture()
    x = torch.ones(op.dimension, dtype=torch.float64)
    before = op.apply("D", x)
    r.add_(10)
    s.mul_(0)
    torch.testing.assert_close(op.apply("D", x), before, atol=0, rtol=0)


def test_invalid_inputs_and_unknown_operator():
    _, op, _, s = fixture()
    with pytest.raises(ValueError, match="FP64"):
        op.apply("L", torch.zeros(op.dimension, dtype=torch.float32))
    with pytest.raises(ValueError, match="operator"):
        op.apply("Q", torch.zeros(op.dimension, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        op.rhs_psi(s * float("nan"))


def test_diagnostic_solver_preserves_energy_but_not_admitted():
    dense, op, _, s = fixture()
    result = solve_diagnostic(op, s)
    from maple.solvation.api.units import HARTREE_TO_EV

    eye = torch.eye(op.dimension, dtype=torch.float64)
    a = 2 * math.pi * (78.39 + 1) / (78.39 - 1) * eye - dense.D
    y = torch.linalg.solve(a, (2 * math.pi * eye - dense.D) @ dense.rhs)
    z = torch.linalg.solve(dense.L, y)
    expected = (HARTREE_TO_EV / 2) * dense.psi.dot(z)
    assert abs(result["energy_eV"] - float(expected)) < 1e-10
    assert result["scientific_admitted"] is False
    assert result["stability_certified"] is False
    assert max(row["true_relative_residual"] for row in result["solves"]) <= 1e-12


def test_solver_fails_closed_when_not_converged():
    _, op, _, s = fixture()
    with pytest.raises(RuntimeError, match="converge"):
        solve_diagnostic(op, s, maxiter=1, restart=1)


@pytest.mark.parametrize("name", ["positions", "nodes", "fi", "ui", "gi"])
def test_geometry_tensor_mutation_fails_closed(name):
    _, op, _, _ = fixture()
    getattr(op, name).add_(0.001)
    with pytest.raises(RuntimeError, match="mutated"):
        op.apply("D", torch.ones(op.dimension, dtype=torch.float64))


def test_constant_mutation_and_control_reassignment_fail_closed():
    _, op, _, _ = fixture()
    op.parameters._weights.mul_(0.99)
    with pytest.raises(RuntimeError, match="mutated"):
        op.rhs_psi(torch.zeros((op.n, 4), dtype=torch.float64))
    _, op, _, _ = fixture()
    op.epsilon = 2.0
    with pytest.raises(RuntimeError, match="control"):
        op.apply("A", torch.ones(op.dimension, dtype=torch.float64))


def test_rhs_width_has_an_explicit_bound():
    _, op, _, _ = fixture()
    with pytest.raises(MemoryError, match="RHS"):
        op.apply("D", torch.ones((op.dimension, 9), dtype=torch.float64))
    with pytest.raises(ValueError, match="RHS"):
        op.apply("D", torch.ones((op.dimension, 0), dtype=torch.float64))


def test_switch_endpoint_rejection_matches_old_guard():
    r = torch.tensor([[0.0, 0.0, 0.0], [1.95, 0.0, 0.0]], dtype=torch.float64)
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    with pytest.raises(RuntimeError, match="endpoint"):
        DDPCMResponseOperators(("H", "H"), [1.0, 1.0], **kwargs).build(
            r, torch.zeros((2, 4), dtype=torch.float64)
        )
    with pytest.raises(RuntimeError, match="endpoint"):
        StreamedDDPCM(("H", "H"), [1.0, 1.0], r, **kwargs)


@pytest.mark.parametrize("kind", ["L", "D", "A", "C"])
def test_independent_bilinear_pairing_and_repeat(kind):
    _, op, _, _ = fixture()
    generator = torch.Generator().manual_seed(811)
    x, y = [
        torch.randn(op.dimension, generator=generator, dtype=torch.float64)
        for _ in range(2)
    ]
    first = op.apply(kind, x)
    torch.testing.assert_close(op.apply(kind, x), first, atol=0, rtol=0)
    assert abs(float(y.dot(first) - x.dot(op.apply(kind, y, transpose=True)))) < 1e-11


@pytest.mark.parametrize(
    "positions,radii",
    [
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [2.0, 0.8]),
        ([[0.0, 0.0, 0.0], [1.9501, 0.0, 0.0]], [1.0, 1.0]),
        ([[0.0, 0.0, 0.0], [4.0, 0.2, 0.1]], [1.0, 1.0]),
    ],
)
def test_adversarial_topology_certificates(positions, radii):
    r = torch.tensor(positions, dtype=torch.float64)
    s = torch.zeros((2, 4), dtype=torch.float64)
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    old = DDPCMResponseOperators(("C", "O"), radii, **kwargs).build(r, s)
    new = StreamedDDPCM(("C", "O"), radii, r, **kwargs)
    assert new.topology == old.geometry.topology
    if radii[0] == 2.0:
        assert new.topology.buried_plateau_count > 0
    if positions[1][0] == 1.9501:
        assert len(new.topology.one_sided_nodes) > 0


def test_coincident_node_and_nonfinite_geometry_rejected():
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    r = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="coincides"):
        StreamedDDPCM(("C", "C"), [1.0, 1.0], r, **kwargs)
    r[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        StreamedDDPCM(("C", "C"), [1.0, 1.0], r, **kwargs)


def test_every_topology_field_is_bound():
    _, op, _, _ = fixture()
    object.__setattr__(op.topology, "active_node_count", 0)
    with pytest.raises(RuntimeError, match="control"):
        op.apply("L", torch.ones(op.dimension, dtype=torch.float64))


def test_parameter_controls_are_bound():
    _, op, _, _ = fixture()
    object.__setattr__(op.parameters, "dielectric", 2.0)
    with pytest.raises(RuntimeError, match="control"):
        op.diagonal("A")


def test_topology_proof_metadata_has_a_fail_closed_budget(monkeypatch):
    import streamed_ddpcm

    monkeypatch.setattr(streamed_ddpcm, "_TOPOLOGY_PROOF_LIMIT_BYTES", 1)
    r = torch.tensor([[0.0, 0.0, 0.0], [1.9501, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(MemoryError, match="proof metadata"):
        StreamedDDPCM(("H", "H"), [1.0, 1.0], r, dielectric=78.39, lmax=2, n_lebedev=50)


def test_tile_workspace_rejected_before_geometry():
    with pytest.raises(MemoryError, match="tile"):
        StreamedDDPCM(
            ("C",) * 100,
            np.full(100, 1.7),
            torch.zeros((100, 3), dtype=torch.float64),
            dielectric=78.39,
            source_tile=100,
        )


@pytest.mark.parametrize("tile", [1, 2, 3, 5])
def test_full_resolution_all_actions_blocks_rhs_and_certificate(tile):
    dense, op, _, source = fixture(3, lmax=15, n_lebedev=1202, tile=tile)
    rhs, psi = op.rhs_psi(source)
    torch.testing.assert_close(rhs, dense.rhs, atol=2e-12, rtol=2e-13)
    torch.testing.assert_close(psi, dense.psi, atol=2e-13, rtol=2e-13)
    assert op.topology == dense.geometry.topology
    generator = torch.Generator().manual_seed(81001)
    x = torch.randn((op.dimension, 3), dtype=torch.float64, generator=generator)
    eye = torch.eye(op.dimension, dtype=torch.float64)
    matrices = {
        "L": dense.L,
        "D": dense.D,
        "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * eye - dense.D,
        "C": 2 * math.pi * eye - dense.D,
    }
    for kind, matrix in matrices.items():
        for transpose in (False, True):
            torch.testing.assert_close(
                op.apply(kind, x, transpose=transpose),
                (matrix.T if transpose else matrix) @ x,
                atol=2e-11,
                rtol=2e-12,
            )


def test_action_allocation_has_no_dense_operator_or_global_pair_grid():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    _, op, _, _ = chain_fixture(4)
    x = torch.ones((op.dimension, 3), dtype=torch.float64)

    class ForbidDense(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_leaves(out):
                if not torch.is_tensor(t):
                    continue
                shape = tuple(t.shape)
                assert shape != (op.dimension, op.dimension)
                assert shape != (op.n, op.g, op.n)
                assert shape != (op.n, op.g, op.n, 3)
            return out

    with ForbidDense():
        op.apply("D", x)
        op.apply("D", x, transpose=True)


@pytest.mark.parametrize("n", [4, 6, 10])
@pytest.mark.parametrize("tile", [1, 3, 10, 12])
def test_oracle_panel_reaches_ten_atoms(n, tile):
    old, op, _, source = chain_fixture(n, tile)
    assert op.topology == old.geometry.topology
    rhs, psi = op.rhs_psi(source)
    torch.testing.assert_close(rhs, old.rhs, atol=2e-12, rtol=2e-13)
    torch.testing.assert_close(psi, old.psi, atol=2e-13, rtol=2e-13)
    eye = torch.eye(op.dimension, dtype=torch.float64)
    matrices = {
        "L": old.L,
        "D": old.D,
        "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * eye - old.D,
        "C": 2 * math.pi * eye - old.D,
    }
    generator = torch.Generator().manual_seed(791)
    x = torch.randn((op.dimension, 2), dtype=torch.float64, generator=generator)
    for kind, matrix in matrices.items():
        for transpose in (False, True):
            torch.testing.assert_close(
                op.apply(kind, x, transpose=transpose),
                (matrix.T if transpose else matrix) @ x,
                atol=2e-11,
                rtol=2e-12,
            )
    for kind in ("L", "A"):
        torch.testing.assert_close(
            op.diagonal(kind), matrices[kind].diagonal(), atol=2e-13, rtol=2e-13
        )


def test_constructor_has_no_global_pair_arrays():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    n, g, q = 4, 50, 9

    class Monitor(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for value in tree_leaves(out):
                if torch.is_tensor(value):
                    assert tuple(value.shape) not in {
                        (n * q, n * q),
                        (n, g, n),
                        (n, g, n, 3),
                    }
            return out

    r = torch.tensor(
        [[i * 1.63, 0.2 * math.sin(i), 0.17 * math.cos(i)] for i in range(n)],
        dtype=torch.float64,
    )
    with Monitor():
        op = StreamedDDPCM(
            ("C",) * n, [1.7] * n, r, dielectric=78.39, lmax=2, n_lebedev=g
        )
    assert op.dimension == n * q


def test_response_protocol_is_explicitly_closed():
    _, op, r, _ = fixture()
    assert op.capabilities["apply"] and op.capabilities["transpose"]
    assert not any(
        op.capabilities[k]
        for k in (
            "jvp",
            "vjp",
            "directional_vjp",
            "hvp",
            "hessian",
            "stability_certified",
            "rhs_jvp",
            "rhs_vjp",
            "rhs_directional_vjp",
            "psi_jvp",
            "psi_vjp",
            "psi_directional_vjp",
        )
    )
    x = torch.ones(op.dimension, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        op.jvp("L", x, r)
    with pytest.raises(NotImplementedError):
        op.vjp("L", x, x)
    with pytest.raises(NotImplementedError):
        op.directional_vjp("L", x, x, r)
    with pytest.raises(TypeError):
        op.capabilities["hessian"] = True
    source = torch.zeros((op.n, 4), dtype=torch.float64)
    for prefix in ("rhs", "psi"):
        with pytest.raises(NotImplementedError):
            getattr(op, prefix + "_jvp")(source, r, source)
        with pytest.raises(NotImplementedError):
            getattr(op, prefix + "_vjp")(source, x)
        with pytest.raises(NotImplementedError):
            getattr(op, prefix + "_directional_vjp")(source, x, r, source)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("resolution", [(2, 50), (15, 1202)])
def test_retained_resource_bound_includes_every_bound_tensor(n, resolution):
    from streamed_ddpcm import known_resource_estimate

    lmax, n_lebedev = resolution
    _, op, _, _ = fixture(n, lmax=lmax, n_lebedev=n_lebedev)
    estimated = known_resource_estimate(n, lmax, n_lebedev, op.tile)
    actual = sum(t.numel() * t.element_size() for _, t in op.retained_tensors())
    assert estimated["retained_upper_bytes"] >= actual


def test_combined_resource_model_preflights_before_arrays():
    from streamed_ddpcm import known_resource_estimate

    low = known_resource_estimate(300, 15, 1202, 8, 1)
    high = known_resource_estimate(300, 15, 1202, 8, 8)
    assert (
        low["estimated_known_peak_bytes"] < high["estimated_known_peak_bytes"] < 1024**3
    )
    with pytest.raises(MemoryError, match="Combined"):
        StreamedDDPCM(
            ("C",) * 10000,
            [1.7] * 10000,
            torch.zeros((10000, 3), dtype=torch.float64),
            dielectric=78.39,
        )


def test_iterative_diagnostic_accounts_for_krylov_and_history():
    _, op, _, source = fixture()
    with pytest.raises(MemoryError, match="Krylov"):
        solve_diagnostic(op, source, maxiter=10**8)
    with pytest.raises(ValueError, match="restart"):
        solve_diagnostic(op, source, restart=True)
    report = op.storage_report()["combined_known_resource_estimate"]
    assert report["mace_workspace_bytes"] is None
    assert report["cds_workspace_bytes"] is None
    assert report["mace_cds_capability_available"] is False
