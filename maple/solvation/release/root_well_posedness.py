"""Local and domain-level certificates for operational Route-2 roots.

Anderson, DIIS, damping, and a small final residual are solver evidence, not
uniqueness proofs.  This module reports the quantities that actually matter
for ``r(y)=y-F(y)``: the smallest singular value of ``I-J_F``, a weighted
contraction norm, and a weighted strong-monotonicity margin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

ROOT_WELL_POSEDNESS_CONTRACT_VERSION = "route2-root-well-posedness-v1"
DEFAULT_LOCAL_SINGULAR_VALUE_FLOOR = 1.0e-8
DEFAULT_CERTIFICATE_MARGIN = 1.0e-8


def _matrix(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[0] != result.shape[1]
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be a non-empty finite square matrix.")
    return np.array(result, copy=True)


@dataclass(frozen=True, slots=True)
class RootWellPosednessCertificate:
    dimension: int
    residual_jacobian_sigma_min: float
    residual_jacobian_sigma_max: float
    residual_jacobian_condition_number: float
    weighted_contraction_norm: float
    weighted_strong_monotonicity_margin: float
    local_singular_value_floor: float
    certificate_margin: float
    local_implicit_branch_certified: bool
    contraction_at_state: bool
    strong_monotonicity_at_state: bool
    domain_invariance_certified: bool
    domain_uniform_bound_certified: bool
    global_unique_root_certified: bool
    contract_version: str = ROOT_WELL_POSEDNESS_CONTRACT_VERSION

    def residual_error_bound(self, residual_norm: float) -> float | None:
        """Return a rigorous domain bound, never a pointwise-Jacobian heuristic."""

        if not np.isfinite(residual_norm) or residual_norm < 0.0:
            raise ValueError("residual_norm must be finite and non-negative.")
        if not self.global_unique_root_certified:
            return None
        if self.contraction_at_state and self.weighted_contraction_norm < 1.0:
            return residual_norm / (1.0 - self.weighted_contraction_norm)
        if (
            self.strong_monotonicity_at_state
            and self.weighted_strong_monotonicity_margin > 0.0
        ):
            return residual_norm / self.weighted_strong_monotonicity_margin
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "dimension": self.dimension,
            "residual_jacobian_sigma_min": self.residual_jacobian_sigma_min,
            "residual_jacobian_sigma_max": self.residual_jacobian_sigma_max,
            "residual_jacobian_condition_number": (
                self.residual_jacobian_condition_number
            ),
            "weighted_contraction_norm": self.weighted_contraction_norm,
            "weighted_strong_monotonicity_margin": (
                self.weighted_strong_monotonicity_margin
            ),
            "local_singular_value_floor": self.local_singular_value_floor,
            "certificate_margin": self.certificate_margin,
            "local_implicit_branch_certified": self.local_implicit_branch_certified,
            "contraction_at_state": self.contraction_at_state,
            "strong_monotonicity_at_state": self.strong_monotonicity_at_state,
            "domain_invariance_certified": self.domain_invariance_certified,
            "domain_uniform_bound_certified": self.domain_uniform_bound_certified,
            "global_unique_root_certified": self.global_unique_root_certified,
            "claim_boundary": (
                "Single-state spectra certify at most a local implicit branch. "
                "Global uniqueness additionally requires an invariant domain and "
                "a uniform contraction or strong-monotonicity bound."
            ),
        }


def certify_root_well_posedness(
    state_map_jacobian: object,
    *,
    coordinate_metric: object | None = None,
    local_singular_value_floor: float = DEFAULT_LOCAL_SINGULAR_VALUE_FLOOR,
    certificate_margin: float = DEFAULT_CERTIFICATE_MARGIN,
    domain_invariance_certified: bool = False,
    domain_uniform_bound_certified: bool = False,
) -> RootWellPosednessCertificate:
    """Certify one Jacobian of ``F`` in ``r(y)=y-F(y)``.

    Passing ``domain_*`` booleans is intentionally insufficient on its own:
    the measured local contraction/monotonicity condition must also pass.
    Callers are responsible for binding those domain claims to external
    evidence; this numerical object never infers them from solver history.
    """

    jacobian = _matrix(state_map_jacobian, name="state_map_jacobian")
    dimension = jacobian.shape[0]
    for name, value in (
        ("local_singular_value_floor", local_singular_value_floor),
        ("certificate_margin", certificate_margin),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (
        ("domain_invariance_certified", domain_invariance_certified),
        ("domain_uniform_bound_certified", domain_uniform_bound_certified),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be exactly bool.")

    metric = (
        np.eye(dimension)
        if coordinate_metric is None
        else _matrix(coordinate_metric, name="coordinate_metric")
    )
    if metric.shape != jacobian.shape:
        raise ValueError("coordinate_metric must match the reduced dimension.")
    metric_scale = max(1.0, float(np.linalg.norm(metric, ord="fro")))
    if not np.allclose(metric, metric.T, rtol=0.0, atol=2.0e-13 * metric_scale):
        raise ValueError("coordinate_metric must be symmetric.")
    metric_eigenvalues, metric_vectors = np.linalg.eigh(0.5 * (metric + metric.T))
    if metric_eigenvalues[0] <= 1.0e-14:
        raise ValueError("coordinate_metric must be positive definite.")
    sqrt_metric = (
        metric_vectors @ np.diag(np.sqrt(metric_eigenvalues)) @ metric_vectors.T
    )
    inverse_sqrt_metric = (
        metric_vectors @ np.diag(1.0 / np.sqrt(metric_eigenvalues)) @ metric_vectors.T
    )

    residual_jacobian = np.eye(dimension) - jacobian
    singular_values = np.linalg.svd(residual_jacobian, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    condition = math.inf if sigma_min == 0.0 else sigma_max / sigma_min

    weighted_jacobian = sqrt_metric @ jacobian @ inverse_sqrt_metric
    contraction_norm = float(np.linalg.svd(weighted_jacobian, compute_uv=False)[0])
    symmetric_residual = 0.5 * (
        metric @ residual_jacobian + residual_jacobian.T @ metric
    )
    normalized_monotonicity = (
        inverse_sqrt_metric @ symmetric_residual @ inverse_sqrt_metric
    )
    normalized_monotonicity = 0.5 * (
        normalized_monotonicity + normalized_monotonicity.T
    )
    monotonicity_margin = float(np.linalg.eigvalsh(normalized_monotonicity)[0])

    local = sigma_min >= local_singular_value_floor
    contraction = contraction_norm <= 1.0 - certificate_margin
    monotone = monotonicity_margin >= certificate_margin
    global_unique = (
        domain_invariance_certified
        and domain_uniform_bound_certified
        and (contraction or monotone)
    )
    return RootWellPosednessCertificate(
        dimension=dimension,
        residual_jacobian_sigma_min=sigma_min,
        residual_jacobian_sigma_max=sigma_max,
        residual_jacobian_condition_number=condition,
        weighted_contraction_norm=contraction_norm,
        weighted_strong_monotonicity_margin=monotonicity_margin,
        local_singular_value_floor=float(local_singular_value_floor),
        certificate_margin=float(certificate_margin),
        local_implicit_branch_certified=local,
        contraction_at_state=contraction,
        strong_monotonicity_at_state=monotone,
        domain_invariance_certified=domain_invariance_certified,
        domain_uniform_bound_certified=domain_uniform_bound_certified,
        global_unique_root_certified=global_unique,
    )


def dense_state_map_jacobian(
    state_map_jvp: object,
    *,
    dimension: int,
    maximum_dimension: int = 128,
) -> np.ndarray:
    """Materialize a bounded audit Jacobian from a matrix-free JVP."""

    if not callable(state_map_jvp):
        raise TypeError("state_map_jvp must be callable.")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if (
        isinstance(maximum_dimension, bool)
        or not isinstance(maximum_dimension, int)
        or maximum_dimension < 1
    ):
        raise ValueError("maximum_dimension must be a positive integer.")
    if dimension > maximum_dimension:
        raise ValueError("dense root certificate exceeds its bounded audit dimension.")
    columns = []
    for index in range(dimension):
        direction = np.zeros(dimension)
        direction[index] = 1.0
        column = np.asarray(state_map_jvp(direction), dtype=float)
        if column.shape != (dimension,) or not np.all(np.isfinite(column)):
            raise ValueError("state_map_jvp returned an invalid reduced vector.")
        columns.append(column)
    result = np.column_stack(columns)
    result.setflags(write=False)
    return result


__all__ = [
    "DEFAULT_CERTIFICATE_MARGIN",
    "DEFAULT_LOCAL_SINGULAR_VALUE_FLOOR",
    "ROOT_WELL_POSEDNESS_CONTRACT_VERSION",
    "RootWellPosednessCertificate",
    "certify_root_well_posedness",
    "dense_state_map_jacobian",
]
