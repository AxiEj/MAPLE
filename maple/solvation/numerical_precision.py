"""Shared dtype-aware numerical tolerances for solvation calculations.

The factors in this module are engineering budgets for known floating-point
workflows.  They are not forward-error guarantees for arbitrary algorithms.
"""

from __future__ import annotations

import math
from typing import Any

NUMERICAL_SAFETY_FACTOR = 100.0
SOURCE_CHARGE_ACCUMULATION_SAFETY = 8.0
FLOAT64_SOLVE_RESIDUAL_TOLERANCE = 1.0e-12
BASE_TOPOLOGY_MARGIN = 1.0e-8
BASE_SOURCE_CHARGE_TOLERANCE_E = 1.0e-8


def default_solve_residual_tolerance(dtype: Any) -> float:
    """Return the strict scaled-residual budget for ``dtype``.

    FP64 retains the historical ``1e-12`` contract.  FP32 uses ``100*eps``:
    a deliberately conservative engineering allowance for the implemented
    ``||Ax-b||_2/max(||b||_2, 1)`` metric after two dense direct solves.  It is
    not a normwise backward-error theorem.
    """

    torch = __import__("torch")
    if dtype is torch.float64:
        return FLOAT64_SOLVE_RESIDUAL_TOLERANCE
    if dtype is torch.float32:
        return NUMERICAL_SAFETY_FACTOR * float(torch.finfo(dtype).eps)
    raise TypeError("numerical precision policy supports only float32 and float64.")


def default_topology_margin(dtype: Any) -> float:
    """Return a branch-distance margin resolvable in ``dtype``."""

    torch = __import__("torch")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("numerical precision policy supports only float32 and float64.")
    return max(
        BASE_TOPOLOGY_MARGIN,
        NUMERICAL_SAFETY_FACTOR * float(torch.finfo(dtype).eps),
    )


def source_charge_tolerance(charges: Any) -> float:
    """Return a scale-aware neutrality tolerance for a Torch charge tensor.

    The summation component uses the standard sequential-sum rounding bound
    ``gamma_(n-1) = (n-1)u / (1-(n-1)u)``, with unit roundoff ``u=eps/2``.
    A factor of eight budgets the small number of upstream accumulation and
    conversion stages used by the source pipeline.  This is an engineering
    allowance, not a guarantee for arbitrary preprocessing.  Charges are
    never modified or renormalized.
    """

    torch = __import__("torch")
    if not torch.is_tensor(charges):
        raise TypeError("charges must be a Torch tensor.")
    if charges.dtype not in (torch.float32, torch.float64):
        raise TypeError("charges must have dtype torch.float32 or torch.float64.")
    if charges.numel() < 1:
        raise ValueError("charges must be nonempty.")
    if not bool(torch.isfinite(charges).all().detach().cpu()):
        raise ValueError("charges must be finite.")

    u = float(torch.finfo(charges.dtype).eps) / 2.0
    k = charges.numel() - 1
    ku = k * u
    if ku >= 1.0:
        raise ValueError("charge tensor is too large for the summation error model.")
    gamma = ku / (1.0 - ku)
    sum_abs = float(torch.sum(torch.abs(charges.detach())).cpu())
    rounding_budget = SOURCE_CHARGE_ACCUMULATION_SAFETY * gamma * sum_abs
    if not math.isfinite(rounding_budget):  # defensive after finite input check
        raise ValueError("charge tolerance calculation overflowed.")
    return max(BASE_SOURCE_CHARGE_TOLERANCE_E, rounding_budget)


def hessian_antisymmetry_tolerance(hessian: Any) -> float:
    """Engineering symmetry budget, not an accuracy or stability guarantee.

    Preserve the absolute FP64 budget (eV/Angstrom squared). For FP32,
    also allow 100 eps times the largest Hessian component. This accounts
    for the scale of nested-AD accumulation without symmetrizing the result.
    """
    torch = __import__("torch")
    if hessian.dtype is torch.float64:
        return 1.0e-4
    if hessian.dtype is not torch.float32:
        raise TypeError("Hessian precision must be float32 or float64.")
    scale = float(hessian.detach().abs().max().cpu())
    if not math.isfinite(scale):
        raise ValueError("Hessian must be finite.")
    return max(1.0e-4, NUMERICAL_SAFETY_FACTOR * torch.finfo(hessian.dtype).eps * scale)


__all__ = [
    "BASE_SOURCE_CHARGE_TOLERANCE_E",
    "BASE_TOPOLOGY_MARGIN",
    "FLOAT64_SOLVE_RESIDUAL_TOLERANCE",
    "NUMERICAL_SAFETY_FACTOR",
    "SOURCE_CHARGE_ACCUMULATION_SAFETY",
    "default_solve_residual_tolerance",
    "default_topology_margin",
    "source_charge_tolerance",
    "hessian_antisymmetry_tolerance",
]
