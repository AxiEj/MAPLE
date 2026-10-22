"""Unweighted prior-centered least-squares solver for CDS coefficients.

Column RMS scaling is used only to condition the numerical solve.  It does not
add a penalty or change the least-squares objective.  If the supported design
matrix is rank deficient, ``numpy.linalg.lstsq`` returns the representative
with minimum Euclidean norm in scaled correction coordinates.  Equivalently,
this is the solution with minimum RMS-scaled change from ``prior``; the fitted
coefficients are then not uniquely identified by the training data.
"""

from __future__ import annotations

import numpy as np


def _validated_inputs(X_train, y_train, prior):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    theta_prior = np.asarray(prior, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or theta_prior.ndim != 1:
        raise ValueError("X_train must be 2D; y_train and prior must be 1D")
    if X.shape[0] == 0 or X.shape[0] != y.size or X.shape[1] != theta_prior.size:
        raise ValueError("incompatible or empty training dimensions")
    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
        and np.isfinite(theta_prior).all()
    ):
        raise ValueError("training inputs and prior must be finite")
    return X, y, theta_prior


def fit(X_train, y_train, prior):
    """Fit an ordinary unweighted least-squares correction around ``prior``.

    Exactly zero training columns are held bitwise at their prior values.  For
    supported columns, the returned solution minimizes ``||y - X @ theta||^2``.
    A rank-deficient fit is successful but is explicitly marked as non-unique;
    its null-space bases describe coefficient changes invisible to training.
    """
    X, y, theta_prior = _validated_inputs(X_train, y_train, prior)
    n_rows, n_columns = X.shape
    column_coverage = np.count_nonzero(X, axis=0).astype(np.int64, copy=False)
    supported = column_coverage > 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scales = np.sqrt(np.mean(np.square(X), axis=0))
    if not np.isfinite(scales).all() or np.any(scales[supported] <= 0.0):
        raise ValueError("column RMS scales must be finite and positive when supported")
    supported_indices = np.flatnonzero(supported)
    W = X[:, supported] / scales[supported]
    if not np.isfinite(W).all():
        raise ValueError("scaled training design must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        target_correction = y - X @ theta_prior
    if not np.isfinite(target_correction).all():
        raise FloatingPointError("prior-centered training target must be finite")

    # Use and report an explicit relative cutoff rather than relying on a
    # version-dependent default in numpy.linalg.lstsq.
    rcond = float(max(W.shape) * np.finfo(np.float64).eps)
    scaled_correction, _, rank, singular_values = np.linalg.lstsq(
        W, target_correction, rcond=rcond
    )
    if not (
        np.isfinite(scaled_correction).all() and np.isfinite(singular_values).all()
    ):
        raise FloatingPointError("least-squares solution or singular values are nonfinite")
    rank = int(rank)
    cutoff = float(rcond * singular_values[0]) if singular_values.size else 0.0

    correction = np.zeros(n_columns, dtype=np.float64)
    theta = theta_prior.copy()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        correction[supported] = scaled_correction / scales[supported]
        theta[supported] = theta_prior[supported] + correction[supported]

    with np.errstate(over="ignore", invalid="ignore"):
        residual = y - X @ theta
        train_sse = float(residual @ residual)
    if not (
        np.isfinite(correction).all()
        and np.isfinite(theta).all()
        and np.isfinite(residual).all()
        and np.isfinite(train_sse)
    ):
        raise FloatingPointError("fitted coefficients, residuals, and SSE must be finite")
    n_supported = int(supported_indices.size)
    nullity = n_supported - rank

    if n_supported:
        # A reduced SVD contains all right singular vectors for tall/square W.
        # Wide W needs the full Vh to expose its additional null directions.
        _, _, vh = np.linalg.svd(W, full_matrices=W.shape[0] < n_supported)
        null_scaled = vh[rank:].T.copy()
    else:
        null_scaled = np.empty((0, 0), dtype=np.float64)
    null_coefficients = np.zeros((n_columns, nullity), dtype=np.float64)
    if nullity:
        null_coefficients[supported, :] = null_scaled / scales[supported, None]
        identifiable_supported = (
            np.max(np.abs(null_scaled), axis=1) <= np.sqrt(np.finfo(np.float64).eps)
        )
    else:
        identifiable_supported = np.ones(n_supported, dtype=bool)
    identifiable = np.zeros(n_columns, dtype=bool)
    identifiable[supported] = identifiable_supported

    full_column_rank = rank == n_supported
    if singular_values.size and full_column_rank:
        condition_number = float(singular_values[0] / singular_values[-1])
    elif singular_values.size:
        condition_number = float("inf")
    else:
        condition_number = 0.0
    diagnostics = {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "n_supported": n_supported,
        "supported_indices": supported_indices,
        "full_column_rank": full_column_rank,
        "unique_supported_solution": full_column_rank,
        "representative": (
            "unique least-squares solution"
            if full_column_rank
            else "minimum RMS-scaled change from prior; coefficients are not uniquely identified"
        ),
        "nullity": int(nullity),
        "null_space_basis_scaled": null_scaled,
        "null_space_basis_coefficient": null_coefficients,
        "identifiable_mask": identifiable,
        "rcond": rcond,
        "rcond_cutoff": cutoff,
        "condition_number": condition_number,
        "residual_l2": float(np.sqrt(train_sse)),
    }
    return {
        "theta": theta,
        "rank": rank,
        "singular_values": singular_values,
        "column_coverage": column_coverage,
        "supported_mask": supported,
        "scales": scales,
        "coefficient_correction": correction,
        "train_sse": train_sse,
        "diagnostics": diagnostics,
    }
