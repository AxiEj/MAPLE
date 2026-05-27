"""Exact autograd Hessian helpers shared by ML calculators.

These helpers keep Hessian acceleration on the analytic/autograd path.  They
never fall back to finite differences, so callers can use them in precision-
critical TS optimizers without changing the Hessian definition.
"""
from __future__ import annotations

import operator
import warnings

import torch

AUTO_BATCH_SIZE = "auto"
AUTO_BATCH_TARGET_FRACTION = 0.75


def _is_auto_batch_size(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO_BATCH_SIZE


def _positive_int_auto_or_none(value, name: str):
    if value is None:
        return None
    if _is_auto_batch_size(value):
        return AUTO_BATCH_SIZE
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, 'auto', or None, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, 'auto', or None, got {value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, 'auto', or None, got {value!r}")
    return value


def _positive_int_or_none(value, name: str) -> int | None:
    value = _positive_int_auto_or_none(value, name)
    if value == AUTO_BATCH_SIZE:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return value


def _physical_dof(grad: torch.Tensor, output_dof: int | None, input_dof: int | None) -> tuple[int, int, int]:
    n_all = grad.numel()
    n_rows = n_all if output_dof is None else int(output_dof)
    n_cols = n_all if input_dof is None else int(input_dof)
    if n_rows < 0 or n_cols < 0 or n_rows > n_all or n_cols > n_all:
        raise ValueError(
            "Invalid Hessian DOF slice: "
            f"output_dof={output_dof}, input_dof={input_dof}, available={n_all}"
        )
    return n_all, n_rows, n_cols


def _auto_hessian_batch_size(
    coordinates: torch.Tensor,
    *,
    n_rows: int,
    n_cols: int,
    n_all: int,
    target_fraction: float = AUTO_BATCH_TARGET_FRACTION,
) -> int:
    """Estimate rows per batched VJP using about 75% of free CUDA memory."""
    if n_rows <= 1:
        return max(1, n_rows)
    if not getattr(coordinates, "is_cuda", False) or not torch.cuda.is_available():
        return n_rows
    try:
        free_bytes, _ = torch.cuda.mem_get_info(coordinates.device)
    except Exception:
        return n_rows
    # grad_outputs + returned block + autograd/vmap temporaries. This is a
    # heuristic, not a physics change; OOM still falls back to exact row loop.
    bytes_per_row = max(n_all, n_cols, 1) * coordinates.element_size() * 8
    target_bytes = max(1.0, float(free_bytes) * float(target_fraction))
    return max(1, min(n_rows, int(target_bytes // bytes_per_row)))


def _row_loop_from_grad(
    grad: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    n_rows: int,
    n_cols: int,
) -> torch.Tensor:
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


def _batched_vjp_from_grad(
    grad: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    n_rows: int,
    n_cols: int,
    batch_size: int,
) -> torch.Tensor:
    n_all = grad.numel()
    hessian = torch.empty(
        (n_rows, n_cols),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        rows = torch.arange(start, stop, device=coordinates.device)
        grad_outputs = torch.zeros(
            (stop - start, n_all),
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        grad_outputs[torch.arange(stop - start, device=coordinates.device), rows] = 1.0
        block = torch.autograd.grad(
            grad,
            coordinates,
            grad_outputs=grad_outputs,
            retain_graph=True,
            is_grads_batched=True,
        )[0].reshape(stop - start, n_all)
        hessian[start:stop] = block[:, :n_cols]
    return hessian


def hessian_loop(
    energy: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    output_dof: int | None = None,
    input_dof: int | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Assemble an exact Hessian by reverse-mode autograd.

    ``batch_size`` is an optional *row chunk* size for analytic Hessians.  When
    it is absent or ``1``, the historical row-by-row loop is used.  Values
    larger than one use PyTorch batched VJPs over at most ``batch_size`` Hessian
    rows at a time and fall back to the exact row loop if that operator stack is
    unsupported.  This changes memory/speed only; it never changes the Hessian
    definition or switches to finite differences.
    """
    grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].reshape(-1)
    _, n_rows, n_cols = _physical_dof(grad, output_dof, input_dof)
    batch_size = _positive_int_auto_or_none(batch_size, "batch_size")
    if batch_size == AUTO_BATCH_SIZE:
        batch_size = _auto_hessian_batch_size(
            coordinates,
            n_rows=n_rows,
            n_cols=n_cols,
            n_all=grad.numel(),
        )

    if batch_size is None or batch_size <= 1:
        return _row_loop_from_grad(grad, coordinates, n_rows=n_rows, n_cols=n_cols)

    try:
        return _batched_vjp_from_grad(
            grad,
            coordinates,
            n_rows=n_rows,
            n_cols=n_cols,
            batch_size=batch_size,
        )
    except (RuntimeError, TypeError):
        return _row_loop_from_grad(grad, coordinates, n_rows=n_rows, n_cols=n_cols)


def hessian_batched_vjp(
    energy: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    output_dof: int | None = None,
    input_dof: int | None = None,
    batch_size: int | None = None,
    warn_on_fallback: bool = False,
) -> torch.Tensor:
    """Assemble an exact Hessian with PyTorch batched VJPs.

    ``torch.autograd.grad(..., is_grads_batched=True)`` uses PyTorch's vmap
    backend to compute vector-Jacobian products.  ``batch_size`` caps how many
    Hessian rows are requested per autograd call; ``None`` preserves the old
    full-matrix batched VJP behavior.  ``batch_size=1`` delegates to the exact
    row loop to minimize memory.  If the PyTorch operator stack does not
    support batched VJPs, the helper falls back to the exact row loop; set
    ``warn_on_fallback=True`` to make that performance fallback visible.
    """
    batch_size = _positive_int_auto_or_none(batch_size, "batch_size")
    if batch_size == 1:
        return hessian_loop(
            energy,
            coordinates,
            output_dof=output_dof,
            input_dof=input_dof,
            batch_size=1,
        )

    grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].reshape(-1)
    n_all, n_rows, n_cols = _physical_dof(grad, output_dof, input_dof)
    if batch_size == AUTO_BATCH_SIZE:
        batch_size = _auto_hessian_batch_size(
            coordinates,
            n_rows=n_rows,
            n_cols=n_cols,
            n_all=n_all,
        )

    try:
        if batch_size is None:
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

        return _batched_vjp_from_grad(
            grad,
            coordinates,
            n_rows=n_rows,
            n_cols=n_cols,
            batch_size=batch_size,
        )
    except (RuntimeError, TypeError) as exc:
        if warn_on_fallback:
            warnings.warn(
                "Batched VJP Hessian fell back to the exact row loop: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return _row_loop_from_grad(grad, coordinates, n_rows=n_rows, n_cols=n_cols)
