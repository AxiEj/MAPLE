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
        [[0.4, 0.1, -0.1, 0.2], [-0.5, -0.05, 0.12, -0.18],
         [0.1, 0.02, -0.03, 0.04]][:n], dtype=torch.float64,
    )
    kwargs = dict(dielectric=78.39, lmax=lmax, n_lebedev=n_lebedev)
    dense = DDPCMResponseOperators(symbols, radii, **kwargs).build(r, s)
    streamed = StreamedDDPCM(symbols, radii, r, source_tile=tile, **kwargs)
    return dense, streamed, r, s


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("tile", [1, 2, 5])
def test_geometry_rhs_and_topology(n, tile):
    dense, op, _, source = fixture(n, tile=tile)
    rhs, psi = op.rhs_psi(source)
    torch.testing.assert_close(rhs, dense.rhs, atol=2e-12, rtol=2e-13)
    torch.testing.assert_close(psi, dense.psi, atol=2e-13, rtol=2e-13)
    for name in ("ui", "gi", "fi"):
        torch.testing.assert_close(getattr(op, name), getattr(dense.geometry, name),
                                   atol=0, rtol=0)
    assert op.topology == dense.geometry.topology


@pytest.mark.parametrize("kind", ["L", "D", "A", "C"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("columns", [1, 3])
def test_actions_and_transposes(kind, transpose, columns):
    dense, op, _, _ = fixture()
    identity = torch.eye(op.dimension, dtype=torch.float64)
    matrices = {"L": dense.L, "D": dense.D,
                "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * identity - dense.D,
                "C": 2 * math.pi * identity - dense.D}
    generator = torch.Generator().manual_seed(42)
    x = torch.randn((op.dimension, columns), generator=generator, dtype=torch.float64)
    if columns == 1:
        x = x[:, 0]
    matrix = matrices[kind].T if transpose else matrices[kind]
    torch.testing.assert_close(op.apply(kind, x, transpose=transpose), matrix @ x,
                               atol=2e-11, rtol=2e-12)


def test_full_resolution_action():
    dense, op, _, _ = fixture(2, lmax=15, n_lebedev=1202, tile=2)
    generator = torch.Generator().manual_seed(512)
    x = torch.randn(op.dimension, generator=generator, dtype=torch.float64)
    for kind, matrix in (("L", dense.L), ("D", dense.D)):
        for transpose in (False, True):
            expected = (matrix.T if transpose else matrix) @ x
            torch.testing.assert_close(op.apply(kind, x, transpose=transpose), expected,
                                       atol=2e-11, rtol=2e-12)


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
    getattr(op, name).add_(.001)
    with pytest.raises(RuntimeError, match="mutated"):
        op.apply("D", torch.ones(op.dimension, dtype=torch.float64))


def test_constant_mutation_and_control_reassignment_fail_closed():
    _, op, _, _ = fixture()
    op.parameters._weights.mul_(.99)
    with pytest.raises(RuntimeError, match="mutated"):
        op.rhs_psi(torch.zeros((op.n, 4), dtype=torch.float64))
    _, op, _, _ = fixture()
    op.epsilon = 2.
    with pytest.raises(RuntimeError, match="control"):
        op.apply("A", torch.ones(op.dimension, dtype=torch.float64))


def test_rhs_width_has_an_explicit_bound():
    _, op, _, _ = fixture()
    with pytest.raises(MemoryError, match="RHS"):
        op.apply("D", torch.ones((op.dimension, 9), dtype=torch.float64))
    with pytest.raises(ValueError, match="RHS"):
        op.apply("D", torch.ones((op.dimension, 0), dtype=torch.float64))


def test_switch_endpoint_rejection_matches_old_guard():
    r = torch.tensor([[0., 0., 0.], [1.95, 0., 0.]], dtype=torch.float64)
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    with pytest.raises(RuntimeError, match="endpoint"):
        DDPCMResponseOperators(("H", "H"), [1., 1.], **kwargs).build(
            r, torch.zeros((2, 4), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="endpoint"):
        StreamedDDPCM(("H", "H"), [1., 1.], r, **kwargs)


@pytest.mark.parametrize("kind", ["L", "D", "A", "C"])
def test_independent_bilinear_pairing_and_repeat(kind):
    _, op, _, _ = fixture()
    generator = torch.Generator().manual_seed(811)
    x, y = [torch.randn(op.dimension, generator=generator, dtype=torch.float64)
            for _ in range(2)]
    first = op.apply(kind, x)
    torch.testing.assert_close(op.apply(kind, x), first, atol=0, rtol=0)
    assert abs(float(y.dot(first) - x.dot(op.apply(kind, y, transpose=True)))) < 1e-11


@pytest.mark.parametrize("positions,radii", [
    ([[0.,0.,0.],[0.,0.,0.]], [2., .8]),
    ([[0.,0.,0.],[1.9501,0.,0.]], [1., 1.]),
    ([[0.,0.,0.],[4.,.2,.1]], [1., 1.]),
])
def test_adversarial_topology_certificates(positions, radii):
    r = torch.tensor(positions, dtype=torch.float64)
    s = torch.zeros((2,4), dtype=torch.float64)
    kwargs = dict(dielectric=78.39,lmax=2,n_lebedev=50)
    old = DDPCMResponseOperators(("C","O"),radii,**kwargs).build(r,s)
    new = StreamedDDPCM(("C","O"),radii,r,**kwargs)
    assert new.topology == old.geometry.topology
    if radii[0] == 2.:
        assert new.topology.buried_plateau_count > 0
    if positions[1][0] == 1.9501:
        assert len(new.topology.one_sided_nodes) > 0


def test_coincident_node_and_nonfinite_geometry_rejected():
    kwargs = dict(dielectric=78.39,lmax=2,n_lebedev=50)
    r = torch.tensor([[0.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
    with pytest.raises(RuntimeError,match="coincides"):
        StreamedDDPCM(("C","C"),[1.,1.],r,**kwargs)
    r[0,0] = float("nan")
    with pytest.raises(ValueError,match="finite"):
        StreamedDDPCM(("C","C"),[1.,1.],r,**kwargs)


def test_every_topology_field_is_bound():
    _,op,_,_=fixture()
    object.__setattr__(op.topology,"active_node_count",0)
    with pytest.raises(RuntimeError,match="control"):
        op.apply("L",torch.ones(op.dimension,dtype=torch.float64))


def test_parameter_controls_are_bound():
    _,op,_,_=fixture()
    object.__setattr__(op.parameters,"dielectric",2.)
    with pytest.raises(RuntimeError,match="control"):
        op.diagonal("A")


def test_tile_workspace_rejected_before_geometry():
    with pytest.raises(MemoryError,match="tile"):
        StreamedDDPCM(("C",)*100,np.full(100,1.7),
                      torch.zeros((100,3),dtype=torch.float64),
                      dielectric=78.39,source_tile=100)


@pytest.mark.parametrize("tile", [1,2,3,5])
def test_full_resolution_all_actions_blocks_rhs_and_certificate(tile):
    dense,op,_,source=fixture(3,lmax=15,n_lebedev=1202,tile=tile)
    rhs,psi=op.rhs_psi(source)
    torch.testing.assert_close(rhs,dense.rhs,atol=2e-12,rtol=2e-13)
    torch.testing.assert_close(psi,dense.psi,atol=2e-13,rtol=2e-13)
    assert op.topology == dense.geometry.topology
    generator=torch.Generator().manual_seed(81001)
    x=torch.randn((op.dimension,3),dtype=torch.float64,generator=generator)
    eye=torch.eye(op.dimension,dtype=torch.float64)
    matrices={"L":dense.L,"D":dense.D,
              "A":2*math.pi*(78.39+1)/(78.39-1)*eye-dense.D,
              "C":2*math.pi*eye-dense.D}
    for kind,matrix in matrices.items():
        for transpose in (False,True):
            torch.testing.assert_close(op.apply(kind,x,transpose=transpose),
                                       (matrix.T if transpose else matrix)@x,
                                       atol=2e-11,rtol=2e-12)


def test_action_allocation_has_no_dense_operator_or_global_pair_grid():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    _,op,_,_=fixture()
    x=torch.ones((op.dimension,3),dtype=torch.float64)
    class ForbidDense(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            for t in tree_leaves(out):
                if not torch.is_tensor(t):
                    continue
                shape=tuple(t.shape)
                assert shape != (op.dimension,op.dimension)
                # Nodes are N,G,3. N=3 makes that shape ambiguous; check 4D
                # pair Cartesian arrays and global pair scalar tensors elsewhere.
                assert shape != (op.n,op.g,op.n,3)
            return out
    with ForbidDense():
        op.apply("D",x)
        op.apply("D",x,transpose=True)
