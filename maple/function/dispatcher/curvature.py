"""Numerically guarded quasi-Newton curvature updates."""

from __future__ import annotations

import numpy as np


def bofill_update(
    hessian: np.ndarray,
    displacement: np.ndarray,
    gradient_change: np.ndarray,
) -> np.ndarray:
    """Return a symmetric Bofill update, or the input for a null secant.

    The null-secant case is legitimate when two evaluations have identical
    gradients.  Returning the unchanged Hessian avoids the ``0 / 0`` terms in
    both the Murtagh-Sargent update and the Bofill mixing factor.
    """
    hessian = np.asarray(hessian, dtype=np.float64)
    dx = np.asarray(displacement, dtype=np.float64).reshape(-1)
    dg = np.asarray(gradient_change, dtype=np.float64).reshape(-1)
    dx2 = float(dx @ dx)
    dg2 = float(dg @ dg)
    if dx2 <= 1e-16 or dg2 <= 1e-16:
        return hessian.copy()

    residual = dg - hessian @ dx
    residual2 = float(residual @ residual)
    residual_dot_step = float(residual @ dx)
    psb = (
        (np.outer(residual, dx) + np.outer(dx, residual)) / dx2
        - residual_dot_step * np.outer(dx, dx) / (dx2 * dx2)
    )
    if abs(residual_dot_step) > 1e-8 and residual2 > 1e-16:
        ms = np.outer(residual, residual) / residual_dot_step
        ratio = residual_dot_step**2 / (dx2 * residual2)
        phi = float(np.clip(1.0 - ratio, 0.0, 1.0))
    else:
        ms = np.zeros_like(hessian)
        phi = 1.0
    updated = hessian + (1.0 - phi) * ms + phi * psb
    return 0.5 * (updated + updated.T)
