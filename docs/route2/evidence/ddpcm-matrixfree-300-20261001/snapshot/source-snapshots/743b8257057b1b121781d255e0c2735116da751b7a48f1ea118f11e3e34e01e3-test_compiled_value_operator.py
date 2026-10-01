"""Full-resolution value parity of the separately identified compiled kernel."""

import math

import torch

from compiled_value_operator import CompiledValueStreamedDDPCM
from test_streamed_ddpcm import fixture


def test_compiled_full_resolution_actions_are_not_a_derivative_claim():
    # This is an isolated test process, matching the native-FP64 caller contract.
    torch.set_default_dtype(torch.float64)
    dense, eager, r, source = fixture(3, lmax=15, n_lebedev=1202)
    generator = torch.Generator().manual_seed(6010)
    x = torch.randn((eager.dimension, 3), dtype=torch.float64, generator=generator)
    identity = torch.eye(eager.dimension, dtype=torch.float64)
    matrices = {
        "L": dense.L,
        "D": dense.D,
        "A": 2 * math.pi * (78.39 + 1) / (78.39 - 1) * identity - dense.D,
        "C": 2 * math.pi * identity - dense.D,
    }
    for tile in (1, 2, 3, 5):
        op = CompiledValueStreamedDDPCM(
            ("C", "O", "H"), [1.7, 1.5, 1.2], r, dielectric=78.39, source_tile=tile
        )
        assert op.topology == dense.geometry.topology
        rhs, psi = op.rhs_psi(source)
        torch.testing.assert_close(rhs, dense.rhs, atol=2e-12, rtol=2e-13)
        torch.testing.assert_close(psi, dense.psi, atol=2e-13, rtol=2e-13)
        for kind, matrix in matrices.items():
            for transpose in (False, True):
                actual = op.apply(kind, x, transpose=transpose)
                torch.testing.assert_close(
                    actual,
                    (matrix.T if transpose else matrix) @ x,
                    atol=2e-11,
                    rtol=2e-12,
                )
                assert not actual.requires_grad
        assert op.storage_report()["analytic_derivative_kernel_qualified"] is False
