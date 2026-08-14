"""Fail-closed reduced stability diagnostics for strict Route-2 candidates.

The module is deliberately provider agnostic and dependency light.  Callers
must first obtain the model response Jacobian and continuum reduced Hessian
from the same sealed scalar graphs used by their stationary state.  This module
then evaluates the local Legendre/passivity, feedback, residual-Jacobian, and
combined-Hessian conditions without changing either response map.

Passing one local diagnostic is implementation evidence only.  It does not
admit a scalar, force, Hessian, variational, or MD capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return result


def _matrix(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[0] != result.shape[1]
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be one non-empty finite square matrix.")
    return np.array(result, copy=True)


def _sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _rows(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def dense_matrix_from_action(
    action: Callable[[np.ndarray], object],
    dimension: int,
    *,
    name: str,
) -> np.ndarray:
    """Materialize one small diagnostic matrix from a matrix-free action."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if not callable(action):
        raise TypeError("action must be callable.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be non-empty.")
    basis = np.eye(dimension)
    columns = []
    for index in range(dimension):
        column = np.asarray(action(basis[:, index]), dtype=float)
        if column.shape != (dimension,) or not np.all(np.isfinite(column)):
            raise ValueError(
                f"{name} action column {index} is non-finite or has wrong shape."
            )
        columns.append(column)
    result = np.column_stack(columns)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VariationalStabilityThresholds:
    symmetry_relative: float = 1.0e-10
    curvature_sign_relative: float = 1.0e-8
    absolute_eigenvalue: float = 1.0e-12
    model_invertibility_relative: float = 1.0e-6
    feedback_imaginary_relative: float = 1.0e-10
    feedback_spectral_radius_maximum: float = 0.95
    residual_minimum_singular_relative: float = 1.0e-3
    combined_positive_relative: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in (
            "symmetry_relative",
            "curvature_sign_relative",
            "absolute_eigenvalue",
            "model_invertibility_relative",
            "feedback_imaginary_relative",
            "feedback_spectral_radius_maximum",
            "residual_minimum_singular_relative",
            "combined_positive_relative",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        if self.feedback_spectral_radius_maximum >= 1.0:
            raise ValueError("feedback_spectral_radius_maximum must be below one.")
        for name in (
            "symmetry_relative",
            "curvature_sign_relative",
            "model_invertibility_relative",
            "feedback_imaginary_relative",
            "residual_minimum_singular_relative",
            "combined_positive_relative",
        ):
            if getattr(self, name) >= 1.0:
                raise ValueError(f"{name} must be below one.")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _symmetric_record(
    values: np.ndarray,
    *,
    thresholds: VariationalStabilityThresholds,
) -> tuple[dict[str, object], np.ndarray]:
    symmetric = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    frobenius = float(np.linalg.norm(values, ord="fro"))
    defect = float(np.linalg.norm(values - values.T, ord="fro"))
    relative_defect = defect / max(frobenius, np.finfo(float).tiny)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    sign_tolerance = (
        thresholds.absolute_eigenvalue + thresholds.curvature_sign_relative * scale
    )
    return (
        {
            "shape": list(values.shape),
            "sha256": _sha256(values),
            "frobenius_norm": frobenius,
            "symmetry_absolute_defect": defect,
            "symmetry_relative_defect": relative_defect,
            "symmetry_gate_passed": (relative_defect <= thresholds.symmetry_relative),
            "eigenvalues": [float(value) for value in eigenvalues],
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "maximum_absolute_eigenvalue": scale,
            "minimum_absolute_eigenvalue": float(np.min(np.abs(eigenvalues))),
            "nonpositive_gate_passed": bool(eigenvalues[-1] <= sign_tolerance),
            "positive_gate_passed": bool(
                eigenvalues[0] >= thresholds.combined_positive_relative * scale
            ),
            "sign_tolerance": sign_tolerance,
        },
        eigenvalues,
    )


def variational_stability_diagnostic(
    model_response_jacobian: object,
    continuum_reduced_hessian: object,
    *,
    thresholds: VariationalStabilityThresholds = VariationalStabilityThresholds(),
) -> dict[str, object]:
    """Evaluate one local reduced strict-variational stability diagnostic.

    ``model_response_jacobian`` is ``J_M = dc/dxi`` in conjugate reduced
    source/field coordinates. ``continuum_reduced_hessian`` is ``H_G`` for the
    same source chart.  For the positive convention used by the current strict
    candidate, passivity requires ``J_M <= 0``.  When ``J_M`` is invertible,
    the eliminated source Hessian is ``-J_M^-1 + H_G``.
    """

    if not isinstance(thresholds, VariationalStabilityThresholds):
        raise TypeError("thresholds must be VariationalStabilityThresholds.")
    model = _matrix(model_response_jacobian, name="model_response_jacobian")
    continuum = _matrix(continuum_reduced_hessian, name="continuum_reduced_hessian")
    if model.shape != continuum.shape:
        raise ValueError("model and continuum reduced matrices must have equal shape.")

    model_record, model_eigenvalues = _symmetric_record(model, thresholds=thresholds)
    continuum_record, _ = _symmetric_record(continuum, thresholds=thresholds)
    model_scale = float(model_record["maximum_absolute_eigenvalue"])
    model_invertible = bool(
        np.min(np.abs(model_eigenvalues))
        >= thresholds.model_invertibility_relative * model_scale
    )

    feedback = model @ continuum
    feedback_eigenvalues = np.linalg.eigvals(feedback)
    feedback_order = np.lexsort((feedback_eigenvalues.imag, feedback_eigenvalues.real))
    feedback_eigenvalues = feedback_eigenvalues[feedback_order]
    spectral_radius = float(np.max(np.abs(feedback_eigenvalues)))
    feedback_scale = max(spectral_radius, np.finfo(float).tiny)
    maximum_imaginary = float(np.max(np.abs(feedback_eigenvalues.imag)))
    feedback_imaginary_gate = bool(
        maximum_imaginary
        <= thresholds.feedback_imaginary_relative * feedback_scale
        + thresholds.absolute_eigenvalue
    )
    feedback_nonnegative_gate = bool(
        float(np.min(feedback_eigenvalues.real))
        >= -(
            thresholds.curvature_sign_relative * feedback_scale
            + thresholds.absolute_eigenvalue
        )
    )
    feedback_radius_gate = bool(
        spectral_radius <= thresholds.feedback_spectral_radius_maximum
    )

    residual = np.eye(model.shape[0]) - feedback
    residual_singular_values = np.linalg.svd(residual, compute_uv=False)
    residual_scale = max(float(residual_singular_values[0]), np.finfo(float).tiny)
    residual_minimum_ratio = float(residual_singular_values[-1] / residual_scale)
    residual_gate = bool(
        residual_minimum_ratio >= thresholds.residual_minimum_singular_relative
    )

    combined = None
    combined_record = None
    if model_invertible:
        combined = -np.linalg.solve(model, np.eye(model.shape[0])) + continuum
        combined_record, _ = _symmetric_record(combined, thresholds=thresholds)
    combined_gate = bool(
        combined_record is not None
        and combined_record["symmetry_gate_passed"]
        and combined_record["positive_gate_passed"]
    )

    decisions = {
        "model_reciprocity_gate_passed": bool(model_record["symmetry_gate_passed"]),
        "model_passivity_gate_passed": bool(model_record["nonpositive_gate_passed"]),
        "model_local_invertibility_gate_passed": model_invertible,
        "continuum_reciprocity_gate_passed": bool(
            continuum_record["symmetry_gate_passed"]
        ),
        "continuum_nonpositive_curvature_gate_passed": bool(
            continuum_record["nonpositive_gate_passed"]
        ),
        "feedback_real_spectrum_gate_passed": feedback_imaginary_gate,
        "feedback_nonnegative_spectrum_gate_passed": feedback_nonnegative_gate,
        "feedback_contraction_gate_passed": feedback_radius_gate,
        "residual_local_nonsingularity_gate_passed": residual_gate,
        "combined_hessian_positive_gate_passed": combined_gate,
    }
    decisions["all_local_stability_gates_passed"] = all(decisions.values())

    return {
        "schema": "route2-variational-reduced-stability-diagnostic-v1",
        "dimension": model.shape[0],
        "thresholds": thresholds.as_dict(),
        "model_response_jacobian": {
            **model_record,
            "matrix": [list(row) for row in _rows(model)],
        },
        "continuum_reduced_hessian": {
            **continuum_record,
            "matrix": [list(row) for row in _rows(continuum)],
        },
        "feedback": {
            "matrix_sha256": _sha256(feedback),
            "eigenvalues_real": [float(value) for value in feedback_eigenvalues.real],
            "eigenvalues_imaginary": [
                float(value) for value in feedback_eigenvalues.imag
            ],
            "spectral_radius": spectral_radius,
            "maximum_absolute_imaginary": maximum_imaginary,
        },
        "residual_jacobian": {
            "matrix_sha256": _sha256(residual),
            "singular_values": [float(value) for value in residual_singular_values],
            "minimum_to_maximum_singular_ratio": residual_minimum_ratio,
        },
        "combined_hessian": (
            None
            if combined_record is None or combined is None
            else {
                **combined_record,
                "matrix": [list(row) for row in _rows(combined)],
            }
        ),
        "decisions": decisions,
        "admitted": False,
        "claim_boundary": (
            "One local reduced spectrum cannot prove global passivity, root "
            "uniqueness, PES smoothness, physical accuracy, or any Route-2 tier."
        ),
    }


__all__ = [
    "VariationalStabilityThresholds",
    "dense_matrix_from_action",
    "variational_stability_diagnostic",
]
