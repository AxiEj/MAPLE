"""Exact autograd Hessian helpers shared by ML calculators.

These helpers keep Hessian acceleration on the analytic/autograd path.  They
never fall back to finite differences, so callers can use them in precision-
critical TS optimizers without changing the Hessian definition.
"""
from __future__ import annotations

import torch


def hessian_loop(
    energy: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    output_dof: int | None = None,
    input_dof: int | None = None,
) -> torch.Tensor:
    """Assemble an exact Hessian by row-wise reverse-mode autograd.

    Parameters
    ----------
    energy
        Scalar energy tensor.
    coordinates
        Coordinate tensor with ``requires_grad=True``.
    output_dof, input_dof
        Optional leading flattened DOF counts.  They are useful for padded
        model inputs such as AIMNet2's sentinel coordinate row, where the
        physical Hessian is the top-left ``3N x 3N`` block.
    """
    grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].reshape(-1)
    n_rows = grad.numel() if output_dof is None else int(output_dof)
    n_cols = grad.numel() if input_dof is None else int(input_dof)

    hessian = torch.empty(
        (n_rows, grad.numel()),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    for row in range(n_rows):
        hessian[row] = torch.autograd.grad(
            grad[row],
            coordinates,
            retain_graph=True,
        )[0].reshape(-1)

    return hessian[:, :n_cols]


def hessian_batched_vjp(
    energy: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    output_dof: int | None = None,
    input_dof: int | None = None,
) -> torch.Tensor:
    """Assemble an exact Hessian with PyTorch batched VJPs.

    ``torch.autograd.grad(..., is_grads_batched=True)`` uses PyTorch's vmap
    backend to compute a batch of vector-Jacobian products in one call.  This
    is mathematically the same Hessian as :func:`hessian_loop`; unsupported
    operator stacks should raise, letting callers retry the loop path.
    """
    grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].reshape(-1)
    n_all = grad.numel()
    n_rows = n_all if output_dof is None else int(output_dof)
    n_cols = n_all if input_dof is None else int(input_dof)

    grad_outputs = torch.eye(
        n_all,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )[:n_rows]
    hessian = torch.autograd.grad(
        grad,
        coordinates,
        grad_outputs=grad_outputs,
        retain_graph=True,
        is_grads_batched=True,
    )[0].reshape(n_rows, n_all)

    return hessian[:, :n_cols]
