"""Parity and fail-closed contracts for the isolated fused D value backend."""

import math

import pytest
import torch

from fused_value_operator import FusedValueStreamedDDPCM
from test_streamed_ddpcm import chain_fixture, fixture


def _candidate(r, n, *, lmax, n_lebedev, tile=2, columns_compile=False):
    symbols = ("C", "O", "H")[:n] if n <= 3 else ("C",) * n
    radii = [1.7, 1.5, 1.2][:n] if n <= 3 else [1.7] * n
    return FusedValueStreamedDDPCM(
        symbols,
        radii,
        r,
        dielectric=78.39,
        lmax=lmax,
        n_lebedev=n_lebedev,
        source_tile=tile,
        compile_kernels=columns_compile,
    )


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("tile", [1, 2, 5])
@pytest.mark.parametrize("columns", [1, 3])
def test_reduced_dense_actions_blocks_and_transposes(n, tile, columns):
    dense, _, r, _ = fixture(n, lmax=2, n_lebedev=50)
    op = _candidate(r, n, lmax=2, n_lebedev=50, tile=tile)
    generator = torch.Generator().manual_seed(7400 + n + columns)
    x = torch.randn((op.dimension, columns), generator=generator, dtype=torch.float64)
    if columns == 1:
        x = x[:, 0]
    identity = torch.eye(op.dimension, dtype=torch.float64)
    matrices = {
        "D": dense.D,
        "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * identity - dense.D,
        "C": 2 * math.pi * identity - dense.D,
    }
    for kind, matrix in matrices.items():
        for transpose in (False, True):
            torch.testing.assert_close(
                op.apply(kind, x, transpose=transpose),
                (matrix.T if transpose else matrix) @ x,
                atol=2e-11,
                rtol=2e-12,
            )


@pytest.mark.parametrize("n", range(2, 11))
def test_reduced_chain_dense_parity_and_pairing(n):
    dense, _, r, _ = chain_fixture(n)
    op = FusedValueStreamedDDPCM(
        ("C",) * n,
        [1.7] * n,
        r,
        dielectric=78.39,
        lmax=2,
        n_lebedev=50,
        source_tile=3,
    )
    generator = torch.Generator().manual_seed(8100 + n)
    x = torch.randn((op.dimension, 2), generator=generator, dtype=torch.float64)
    y = torch.randn((op.dimension, 2), generator=generator, dtype=torch.float64)
    forward = op.apply("D", x)
    transpose = op.apply("D", y, transpose=True)
    torch.testing.assert_close(forward, dense.D @ x, atol=2e-11, rtol=2e-12)
    torch.testing.assert_close(transpose, dense.D.T @ y, atol=2e-11, rtol=2e-12)
    torch.testing.assert_close(
        (y * forward).sum(), (x * transpose).sum(), atol=2e-11, rtol=2e-12
    )


@pytest.mark.parametrize("n", [2, 3])
def test_full_resolution_dense_d_and_transpose_parity(n):
    dense, _, r, _ = fixture(n, lmax=15, n_lebedev=1202)
    op = _candidate(r, n, lmax=15, n_lebedev=1202, tile=2)
    generator = torch.Generator().manual_seed(9150)
    x = torch.randn((op.dimension, 2), generator=generator, dtype=torch.float64)
    torch.testing.assert_close(op.apply("D", x), dense.D @ x, atol=2e-11, rtol=2e-12)
    torch.testing.assert_close(
        op.apply("D", x, transpose=True), dense.D.T @ x, atol=2e-11, rtol=2e-12
    )


def test_fully_buried_target_row_is_pruned_without_erasing_transpose_output():
    r = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    source = torch.zeros((2, 4), dtype=torch.float64)
    kwargs = dict(dielectric=78.39, lmax=2, n_lebedev=50)
    from maple.solvation.continuum.ddpcm_response_operators import (
        DDPCMResponseOperators,
    )

    dense = DDPCMResponseOperators(("C", "O"), [2.0, 0.8], **kwargs).build(r, source)
    op = FusedValueStreamedDDPCM(("C", "O"), [2.0, 0.8], r, source_tile=2, **kwargs)

    # Constructor certification is still the complete, all-node base contract.
    assert op.topology == dense.geometry.topology
    torch.testing.assert_close(op.ui, dense.geometry.ui, atol=0, rtol=0)
    assert int((op.ui[0] > 0).sum()) == 50
    assert int((op.ui[1] > 0).sum()) == 0
    assert op.topology.buried_plateau_count == 50

    generator = torch.Generator().manual_seed(9901)
    x = torch.zeros(op.dimension, dtype=torch.float64)
    x[: op.q] = torch.randn(op.q, generator=generator, dtype=torch.float64)
    forward = op.apply("D", x)
    transpose = op.apply("D", x, transpose=True)
    torch.testing.assert_close(forward, dense.D @ x, atol=2e-11, rtol=2e-12)
    torch.testing.assert_close(transpose, dense.D.T @ x, atol=2e-11, rtol=2e-12)
    assert torch.count_nonzero(forward[op.q :]) == 0
    # This block was accumulated while processing target row zero.  The later
    # empty-active-row branch must preserve it rather than zeroing result[1].
    assert float(torch.linalg.vector_norm(transpose[op.q :])) > 0


def test_metadata_capabilities_binding_and_compile_entry_hook():
    _, _, r, _ = fixture(2, lmax=2, n_lebedev=50)
    op = _candidate(r, 2, lmax=2, n_lebedev=50)
    report = op.storage_report()
    assert report["fused_d_materializes_source_grid_basis"] is False
    assert report["fused_d_materialized_kernel_tile_bytes"] == 0
    assert report["kernel_tile_bytes"] == report["reference_l_kernel_tile_bytes"] > 0
    assert report["fused_d_exact_zero_node_pruning"].startswith("ui > 0")
    assert report["fused_d_largest_named_tile_rank"] == "SxGxK"
    assert len(report["kernel_source_sha256"]["fused_value_operator.py"]) == 64
    assert report["analytic_derivative_kernel_qualified"] is False
    for capability in ("jvp", "vjp", "hvp", "hessian", "stability_certified"):
        assert op.capabilities[capability] is False
    op._fused_source_id = "mutated"
    with pytest.raises(RuntimeError, match="controls"):
        op.apply("D", torch.ones(op.dimension, dtype=torch.float64))

    # The compiler path is an explicit constructor hook; expensive compilation is
    # intentionally left to the separately supervised benchmark process.
    compiled = _candidate(r, 2, lmax=2, n_lebedev=50, columns_compile=False)
    assert compiled.storage_report()["compiled_value_kernels"] is False
