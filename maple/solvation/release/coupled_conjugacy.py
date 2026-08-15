"""Terminal coupling-active-space conjugacy audit for original MACE-POLAR.

The original source and intrinsic energy need not be compared component by
component when the continuum only sees a quotient of source space.  The last
legitimate loophole is therefore tested after the actual operators:

``B M(u) + s L.T grad_u E(u) = 0`` and ``B M_u L`` self-adjoint on the
continuum coefficient chart.  A material failure at one state is a decisive
counterexample.  Passing finitely many states is explicitly not a global
variational proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

COUPLED_CONJUGACY_CONTRACT_VERSION = "route2-coupled-conjugacy-terminal-audit-v1"
COUPLED_CONJUGACY_RELATIVE_TOLERANCE = 1.0e-9
COUPLED_CURL_RELATIVE_TOLERANCE = 1.0e-9


def _matrix(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite matrix.")
    return np.array(result, copy=True)


def _vector(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return np.array(result, copy=True)


def _weighted_norm(values: np.ndarray, metric: np.ndarray) -> float:
    dual = np.linalg.solve(metric, values)
    squared = float(np.vdot(values, dual))
    if not math.isfinite(squared) or squared < -1.0e-12:
        raise RuntimeError("boundary metric produced an invalid dual norm.")
    return math.sqrt(max(0.0, squared))


@dataclass(frozen=True, slots=True)
class CoupledConjugacyAuditResult:
    energy_gradient_sign: int
    energy_semantics_verified: bool
    boundary_dimension: int
    source_dimension: int
    native_field_dimension: int
    residual_absolute: float
    residual_relative: float
    residual_tolerance: float
    coupled_curl_absolute_frobenius: float
    coupled_curl_relative_frobenius: float
    curl_tolerance: float
    source_boundary_norm: float
    energy_boundary_norm: float
    response_frobenius: float
    direct_identity_passed: bool
    coupled_reciprocity_passed: bool
    contract_version: str = COUPLED_CONJUGACY_CONTRACT_VERSION

    @property
    def terminal_no_go_witness_detected(self) -> bool:
        return (not self.coupled_reciprocity_passed) or (
            self.energy_semantics_verified and not self.direct_identity_passed
        )

    @property
    def direct_identity_interpretable(self) -> bool:
        return self.energy_semantics_verified

    @property
    def material_direct_defect_detected(self) -> bool:
        return not self.direct_identity_passed

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "energy_gradient_sign": self.energy_gradient_sign,
            "energy_semantics_verified": self.energy_semantics_verified,
            "boundary_dimension": self.boundary_dimension,
            "source_dimension": self.source_dimension,
            "native_field_dimension": self.native_field_dimension,
            "residual_absolute": self.residual_absolute,
            "residual_relative": self.residual_relative,
            "residual_tolerance": self.residual_tolerance,
            "coupled_curl_absolute_frobenius": (self.coupled_curl_absolute_frobenius),
            "coupled_curl_relative_frobenius": (self.coupled_curl_relative_frobenius),
            "curl_tolerance": self.curl_tolerance,
            "source_boundary_norm": self.source_boundary_norm,
            "energy_boundary_norm": self.energy_boundary_norm,
            "response_frobenius": self.response_frobenius,
            "direct_identity_passed": self.direct_identity_passed,
            "direct_identity_interpretable": self.direct_identity_interpretable,
            "material_direct_defect_detected": self.material_direct_defect_detected,
            "coupled_reciprocity_passed": self.coupled_reciprocity_passed,
            "terminal_no_go_witness_detected": (self.terminal_no_go_witness_detected),
            "claim_boundary": (
                "A coupled-curl failure is terminal. A direct energy/source "
                "failure is terminal only after the consumed energy has verified "
                "external-enthalpy semantics and sign. A finite pass is not a "
                "global proof."
            ),
        }


def analyze_coupled_conjugacy(
    *,
    source_to_boundary: object,
    boundary_to_native_field: object,
    original_source: object,
    intrinsic_energy_native_field_gradient: object,
    source_native_field_jacobian: object,
    boundary_metric: object | None = None,
    energy_gradient_sign: int = 1,
    energy_semantics_verified: bool = False,
    residual_tolerance: float = COUPLED_CONJUGACY_RELATIVE_TOLERANCE,
    curl_tolerance: float = COUPLED_CURL_RELATIVE_TOLERANCE,
) -> CoupledConjugacyAuditResult:
    """Audit conjugacy after the exact source and receiver coupling maps.

    ``B`` has shape ``(n_boundary, n_source)`` and ``L`` has shape
    ``(n_native_field, n_boundary)``.  The source Jacobian is
    ``d source / d native_field``.  ``boundary_metric`` identifies coefficient
    vectors with covectors for numerical norms; it does not alter the raw
    stationarity covector ``B c + s L.T grad(E)``.
    """

    if energy_gradient_sign not in (-1, 1):
        raise ValueError("energy_gradient_sign must be -1 or +1.")
    if type(energy_semantics_verified) is not bool:
        raise TypeError("energy_semantics_verified must be exactly bool.")
    for name, value in (
        ("residual_tolerance", residual_tolerance),
        ("curl_tolerance", curl_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    source_operator = _matrix(source_to_boundary, name="source_to_boundary B")
    receiver_operator = _matrix(
        boundary_to_native_field, name="boundary_to_native_field L"
    )
    source = _vector(original_source, name="original_source")
    gradient = _vector(
        intrinsic_energy_native_field_gradient,
        name="intrinsic_energy_native_field_gradient",
    )
    jacobian = _matrix(
        source_native_field_jacobian, name="source_native_field_jacobian"
    )
    boundary_dimension, source_dimension = source_operator.shape
    native_field_dimension = receiver_operator.shape[0]
    if receiver_operator.shape[1] != boundary_dimension:
        raise ValueError("L boundary dimension does not match B.")
    if source.shape != (source_dimension,):
        raise ValueError("original_source dimension does not match B.")
    if gradient.shape != (native_field_dimension,):
        raise ValueError("intrinsic-energy gradient dimension does not match L.")
    if jacobian.shape != (source_dimension, native_field_dimension):
        raise ValueError("source Jacobian must have shape (source, native_field).")

    metric = (
        np.eye(boundary_dimension)
        if boundary_metric is None
        else _matrix(boundary_metric, name="boundary_metric")
    )
    if metric.shape != (boundary_dimension, boundary_dimension):
        raise ValueError("boundary_metric has the wrong shape.")
    metric_scale = max(1.0, float(np.linalg.norm(metric, ord="fro")))
    if not np.allclose(metric, metric.T, rtol=0.0, atol=2.0e-13 * metric_scale):
        raise ValueError("boundary_metric must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    if eigenvalues[0] <= 1.0e-14:
        raise ValueError("boundary_metric must be positive definite.")

    source_covector = source_operator @ source
    energy_covector = receiver_operator.T @ gradient
    residual = source_covector + energy_gradient_sign * energy_covector
    source_norm = _weighted_norm(source_covector, metric)
    energy_norm = _weighted_norm(energy_covector, metric)
    residual_norm = _weighted_norm(residual, metric)
    residual_relative = residual_norm / max(
        source_norm + energy_norm, np.finfo(float).tiny
    )

    # B J_M L is a boundary covector response.  Whiten covector and vector
    # charts with the same SPD metric so the relative defect is dimensionally
    # consistent and invariant under a common coefficient-basis change.
    raw_response = source_operator @ jacobian @ receiver_operator
    metric_eigenvalues, metric_vectors = np.linalg.eigh(metric)
    inverse_sqrt_metric = (
        metric_vectors @ np.diag(1.0 / np.sqrt(metric_eigenvalues)) @ metric_vectors.T
    )
    response_operator = inverse_sqrt_metric @ raw_response @ inverse_sqrt_metric
    curl = response_operator - response_operator.T
    response_norm = float(np.linalg.norm(response_operator, ord="fro"))
    curl_norm = float(np.linalg.norm(curl, ord="fro"))
    curl_relative = curl_norm / max(response_norm, np.finfo(float).tiny)

    return CoupledConjugacyAuditResult(
        energy_gradient_sign=energy_gradient_sign,
        energy_semantics_verified=energy_semantics_verified,
        boundary_dimension=boundary_dimension,
        source_dimension=source_dimension,
        native_field_dimension=native_field_dimension,
        residual_absolute=residual_norm,
        residual_relative=residual_relative,
        residual_tolerance=float(residual_tolerance),
        coupled_curl_absolute_frobenius=curl_norm,
        coupled_curl_relative_frobenius=curl_relative,
        curl_tolerance=float(curl_tolerance),
        source_boundary_norm=source_norm,
        energy_boundary_norm=energy_norm,
        response_frobenius=response_norm,
        direct_identity_passed=(residual_relative <= residual_tolerance),
        coupled_reciprocity_passed=(curl_relative <= curl_tolerance),
    )


__all__ = [
    "COUPLED_CONJUGACY_CONTRACT_VERSION",
    "COUPLED_CONJUGACY_RELATIVE_TOLERANCE",
    "COUPLED_CURL_RELATIVE_TOLERANCE",
    "CoupledConjugacyAuditResult",
    "analyze_coupled_conjugacy",
]
