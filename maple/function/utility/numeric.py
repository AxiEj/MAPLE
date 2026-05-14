# -*- coding: utf-8 -*-
"""Numeric coercion helpers shared across optimizer / TS / IRC algorithms.

These two functions appeared verbatim in several algorithm modules
(optimization/_common.py, ts/algorithm/PRFO.py, ts/algorithm/neb.py, ...).
Centralising them here keeps the float64/torch/scalar coercion contract
single-sourced, so future numeric-typing fixes propagate automatically.
"""
from typing import Optional

import numpy as np


def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def vec1d(x, n_expected: Optional[int] = None) -> np.ndarray:
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v
