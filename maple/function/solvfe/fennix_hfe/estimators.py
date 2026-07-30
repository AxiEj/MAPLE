"""Deterministic estimator primitives for fixed FeNNix HFE trajectories.

This module contains no sampling, model evaluation, file access, adaptive window
selection, or admission logic.  Inputs are already-sampled, fixed-window scalar
observations and every result retains explicit units and provenance.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np

EV_TO_KCAL_PER_MOL = 23.06054783061903
EV_TO_KJ_PER_MOL = EV_TO_KCAL_PER_MOL * 4.184
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ENERGY_TO_EV = {
    "eV": 1.0,
    "kcal/mol": 1.0 / EV_TO_KCAL_PER_MOL,
    "kJ/mol": 1.0 / EV_TO_KJ_PER_MOL,
}


@dataclass(frozen=True)
class EstimatorProvenance:
    """Content identity and nonempty provenance for estimator inputs."""

    input_sha256: str
    protocol_sha256: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("input_sha256", "protocol_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
            object.__setattr__(self, name, value.lower())
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be a nonempty string.")
        object.__setattr__(self, "provenance", self.provenance.strip())


@dataclass(frozen=True)
class LambdaLegIntegral:
    """Composite-Simpson and trapezoid integrals for one native lambda leg."""

    grid: tuple[float, ...]
    mean_derivative: tuple[float, ...]
    simpson: float
    trapezoid: float
    absolute_discrepancy: float
    energy_unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class FixedWindowTIResult:
    """Signed sum of predeclared native-leg integrals."""

    leg_count: int
    sign: int
    simpson: float
    trapezoid: float
    absolute_discrepancy: float
    energy_unit: str
    provenance: EstimatorProvenance


class BAROverlapError(ValueError):
    """Raised when BAR has no numerically resolvable bidirectional overlap."""


@dataclass(frozen=True)
class BARResult:
    """Adjacent-state bidirectional BAR solution in reduced-energy units."""

    delta_f: float
    residual: float
    iterations: int
    converged: bool
    forward_sample_count: int
    reverse_sample_count: int
    reduced_unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class ReweightingDiagnostics:
    """Symmetric overlap and directional exponential-reweighting ESS."""

    overlap: float
    forward_effective_sample_size: float
    reverse_effective_sample_size: float
    forward_ess_fraction: float
    reverse_ess_fraction: float
    reduced_unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class AutocorrelationEstimate:
    """Deterministic initial-positive-sequence integrated autocorrelation."""

    sample_count: int
    integrated_autocorrelation_time: float
    effective_sample_size: float
    maximum_lag_used: int
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class MovingBlockBootstrapResult:
    """Fixed-seed moving-block bootstrap result for a scalar sample mean."""

    sample_count: int
    block_length: int
    resample_count: int
    seed: int
    estimate: float
    standard_error: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    bootstrap_means: tuple[float, ...]
    unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class HalfTrajectoryDiagnostic:
    """First/second-half scalar stationarity diagnostic."""

    sample_count: int
    first_half_count: int
    second_half_count: int
    first_half_mean: float
    second_half_mean: float
    signed_mean_shift: float
    absolute_mean_shift: float
    unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class InterwalkerDiagnostic:
    """Deterministic between-walker mean-spread diagnostic."""

    walker_count: int
    samples_per_walker: int
    pooled_mean: float
    walker_means: tuple[float, ...]
    walker_mean_standard_deviation: float
    maximum_absolute_walker_mean_deviation: float
    unit: str
    provenance: EstimatorProvenance


@dataclass(frozen=True)
class HalfTrajectoryGateResult:
    """Explicit-threshold first/second-half diagnostic decision."""

    diagnostic: HalfTrajectoryDiagnostic
    maximum_absolute_mean_shift: float
    passed: bool


@dataclass(frozen=True)
class InterwalkerGateResult:
    """Explicit-threshold interwalker diagnostic decision."""

    diagnostic: InterwalkerDiagnostic
    maximum_walker_mean_standard_deviation: float
    maximum_absolute_walker_mean_deviation: float
    passed: bool


def _finite_1d(
    values: Sequence[Real] | np.ndarray, name: str, minimum: int
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(
            f"{name} must be one-dimensional with at least {minimum} values."
        )
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _nonempty_unit(unit: str, *, energy: bool = False, reduced: bool = False) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a nonempty string.")
    value = unit.strip()
    if energy and value not in _ENERGY_TO_EV:
        raise ValueError("energy unit must be one of: eV, kcal/mol, kJ/mol.")
    if reduced and value not in {"kT", "dimensionless"}:
        raise ValueError("reduced unit must be 'kT' or 'dimensionless'.")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _convert_energy(value: float, source: str, target: str) -> float:
    return float(value) * _ENERGY_TO_EV[source] / _ENERGY_TO_EV[target]


def _quadratic_panel_integral(x: np.ndarray, y: np.ndarray) -> float:
    """Integrate the unique quadratic through three distinct ordered points."""

    x0, x1, x2 = (float(v) for v in x)
    y0, y1, y2 = (float(v) for v in y)
    a, b = x0, x2

    def weight(xi: float, xj: float, xk: float) -> float:
        denominator = (xi - xj) * (xi - xk)
        antiderivative = lambda z: z**3 / 3.0 - (xj + xk) * z**2 / 2.0 + xj * xk * z
        return (antiderivative(b) - antiderivative(a)) / denominator

    return math.fsum(
        (
            y0 * weight(x0, x1, x2),
            y1 * weight(x1, x0, x2),
            y2 * weight(x2, x0, x1),
        )
    )


def integrate_native_lambda_leg(
    grid: Sequence[Real] | np.ndarray,
    mean_derivative: Sequence[Real] | np.ndarray,
    *,
    energy_unit: str,
    provenance: EstimatorProvenance,
) -> LambdaLegIntegral:
    """Integrate a fixed native-leg grid without inserting or tuning windows.

    Simpson panels use each consecutive triplet.  If the number of intervals is
    odd, the final interval is integrated by trapezoid and disclosed by the
    Simpson-versus-all-trapezoid discrepancy.
    """

    x = _finite_1d(grid, "grid", 2)
    y = _finite_1d(mean_derivative, "mean_derivative", 2)
    if x.shape != y.shape:
        raise ValueError("grid and mean_derivative must have exactly the same shape.")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("grid must be strictly increasing.")
    unit = _nonempty_unit(energy_unit, energy=True)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")

    trapezoid = float(np.trapezoid(y, x))
    panel_stop = x.size - 1 if (x.size - 1) % 2 == 0 else x.size - 2
    terms = [
        _quadratic_panel_integral(x[index : index + 3], y[index : index + 3])
        for index in range(0, panel_stop, 2)
    ]
    if panel_stop < x.size - 1:
        terms.append(0.5 * float(x[-1] - x[-2]) * float(y[-1] + y[-2]))
    simpson = math.fsum(terms)
    return LambdaLegIntegral(
        grid=tuple(float(value) for value in x),
        mean_derivative=tuple(float(value) for value in y),
        simpson=simpson,
        trapezoid=trapezoid,
        absolute_discrepancy=abs(simpson - trapezoid),
        energy_unit=unit,
        provenance=provenance,
    )


def fixed_window_ti_sum(
    legs: Sequence[LambdaLegIntegral],
    *,
    sign: int,
    output_energy_unit: str,
    provenance: EstimatorProvenance,
) -> FixedWindowTIResult:
    """Sum fixed native-leg TI integrals with an explicit thermodynamic sign."""

    if (
        not isinstance(sign, Integral)
        or isinstance(sign, bool)
        or int(sign) not in {-1, 1}
    ):
        raise ValueError("sign must be exactly +1 or -1.")
    if not legs:
        raise ValueError("legs must contain at least one native-leg integral.")
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    target = _nonempty_unit(output_energy_unit, energy=True)
    simpson_terms: list[float] = []
    trapezoid_terms: list[float] = []
    for leg in legs:
        if not isinstance(leg, LambdaLegIntegral):
            raise TypeError("every leg must be a LambdaLegIntegral.")
        source = _nonempty_unit(leg.energy_unit, energy=True)
        simpson_terms.append(_convert_energy(leg.simpson, source, target))
        trapezoid_terms.append(_convert_energy(leg.trapezoid, source, target))
    simpson = int(sign) * math.fsum(simpson_terms)
    trapezoid = int(sign) * math.fsum(trapezoid_terms)
    return FixedWindowTIResult(
        leg_count=len(legs),
        sign=int(sign),
        simpson=simpson,
        trapezoid=trapezoid,
        absolute_discrepancy=abs(simpson - trapezoid),
        energy_unit=target,
        provenance=provenance,
    )


def _expit_negative(value: np.ndarray | float) -> np.ndarray | float:
    """Return 1/(1+exp(value)) without overflow."""

    array = np.asarray(value, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    exponential = np.exp(-array[positive])
    result[positive] = exponential / (1.0 + exponential)
    exponential = np.exp(array[~positive])
    result[~positive] = 1.0 / (1.0 + exponential)
    if result.ndim == 0:
        return float(result)
    return result


def solve_adjacent_bar(
    forward_reduced_work: Sequence[Real] | np.ndarray,
    reverse_reduced_work: Sequence[Real] | np.ndarray,
    *,
    provenance: EstimatorProvenance,
    reduced_unit: str = "kT",
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 256,
) -> BARResult:
    """Solve adjacent-state bidirectional BAR by stable bracketed bisection.

    Reverse work follows the B-to-A convention.  ``delta_f`` is therefore the
    reduced free energy F_B-F_A.
    """

    forward = _finite_1d(forward_reduced_work, "forward_reduced_work", 1)
    reverse = _finite_1d(reverse_reduced_work, "reverse_reduced_work", 1)
    unit = _nonempty_unit(reduced_unit, reduced=True)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a finite positive number.")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be a finite positive number.")
    iteration_limit = _positive_int(maximum_iterations, "maximum_iterations")
    sample_log_ratio = math.log(float(forward.size) / float(reverse.size))

    def equation(delta_f: float) -> float:
        lhs = np.asarray(_expit_negative(forward - delta_f + sample_log_ratio))
        rhs = np.asarray(_expit_negative(reverse + delta_f - sample_log_ratio))
        return float(np.sum(lhs, dtype=np.float64) - np.sum(rhs, dtype=np.float64))

    scale = max(1.0, float(np.max(np.abs(forward))), float(np.max(np.abs(reverse))))
    lower, upper = -2.0 * scale, 2.0 * scale
    f_lower, f_upper = equation(lower), equation(upper)
    expansions = 0
    while not (f_lower <= 0.0 <= f_upper) and expansions < 64:
        lower *= 2.0
        upper *= 2.0
        f_lower, f_upper = equation(lower), equation(upper)
        expansions += 1
    if not (f_lower <= 0.0 <= f_upper):
        raise RuntimeError("BAR root could not be bracketed from finite work inputs.")

    midpoint = 0.5 * (lower + upper)
    residual = equation(midpoint)
    converged = abs(residual) <= tolerance_value
    iterations = 0
    while not converged and iterations < iteration_limit:
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        midpoint = 0.5 * (lower + upper)
        residual = equation(midpoint)
        iterations += 1
        converged = abs(residual) <= tolerance_value or (
            upper - lower <= tolerance_value * max(1.0, abs(midpoint))
        )
    forward_overlap = float(
        np.mean(_expit_negative(forward - midpoint + sample_log_ratio))
    )
    reverse_overlap = float(
        np.mean(_expit_negative(reverse + midpoint - sample_log_ratio))
    )
    numerical_overlap_floor = 32.0 * np.finfo(np.float64).eps
    if min(forward_overlap, reverse_overlap) <= numerical_overlap_floor:
        raise BAROverlapError(
            "BAR inputs have no numerically resolvable bidirectional overlap."
        )
    return BARResult(
        delta_f=midpoint,
        residual=residual,
        iterations=iterations,
        converged=converged,
        forward_sample_count=int(forward.size),
        reverse_sample_count=int(reverse.size),
        reduced_unit=unit,
        provenance=provenance,
    )


def _exponential_ess(log_weights: np.ndarray) -> float:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    total = float(np.sum(weights, dtype=np.float64))
    squared = float(np.dot(weights, weights))
    return total * total / squared


def bidirectional_reweighting_diagnostics(
    forward_reduced_work: Sequence[Real] | np.ndarray,
    reverse_reduced_work: Sequence[Real] | np.ndarray,
    delta_f: Real,
    *,
    provenance: EstimatorProvenance,
    reduced_unit: str = "kT",
) -> ReweightingDiagnostics:
    """Compute symmetric BAR overlap and directional exponential-weight ESS."""

    forward = _finite_1d(forward_reduced_work, "forward_reduced_work", 1)
    reverse = _finite_1d(reverse_reduced_work, "reverse_reduced_work", 1)
    if isinstance(delta_f, bool) or not isinstance(delta_f, Real):
        raise ValueError("delta_f must be finite.")
    delta = float(delta_f)
    if not math.isfinite(delta):
        raise ValueError("delta_f must be finite.")
    unit = _nonempty_unit(reduced_unit, reduced=True)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    sample_log_ratio = math.log(float(forward.size) / float(reverse.size))
    forward_overlap = float(
        np.mean(_expit_negative(forward - delta + sample_log_ratio))
    )
    reverse_overlap = float(
        np.mean(_expit_negative(reverse + delta - sample_log_ratio))
    )
    overlap = min(1.0, max(0.0, forward_overlap + reverse_overlap))
    forward_ess = _exponential_ess(-forward)
    reverse_ess = _exponential_ess(-reverse)
    return ReweightingDiagnostics(
        overlap=overlap,
        forward_effective_sample_size=forward_ess,
        reverse_effective_sample_size=reverse_ess,
        forward_ess_fraction=forward_ess / float(forward.size),
        reverse_ess_fraction=reverse_ess / float(reverse.size),
        reduced_unit=unit,
        provenance=provenance,
    )


def integrated_autocorrelation_time(
    values: Sequence[Real] | np.ndarray,
    *,
    provenance: EstimatorProvenance,
    maximum_lag: int | None = None,
) -> AutocorrelationEstimate:
    """Estimate IAT using Geyer's deterministic initial-positive pair sequence."""

    samples = _finite_1d(values, "values", 2)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    if maximum_lag is None:
        lag_limit = int(samples.size - 1)
    else:
        lag_limit = _positive_int(maximum_lag, "maximum_lag")
        if lag_limit >= samples.size:
            raise ValueError("maximum_lag must be smaller than the sample count.")
    centered = samples - float(np.mean(samples))
    variance_sum = float(np.dot(centered, centered))
    if variance_sum == 0.0:
        return AutocorrelationEstimate(
            int(samples.size), 1.0, float(samples.size), 0, provenance
        )
    correlations = np.empty(lag_limit + 1, dtype=np.float64)
    correlations[0] = 1.0
    for lag in range(1, lag_limit + 1):
        correlations[lag] = (
            float(np.dot(centered[:-lag], centered[lag:])) / variance_sum
        )
    correlation_sum = 0.0
    used_lag = 0
    lag = 1
    while lag <= lag_limit:
        pair = float(correlations[lag])
        if lag + 1 <= lag_limit:
            pair += float(correlations[lag + 1])
        if pair <= 0.0:
            break
        correlation_sum += pair
        used_lag = min(lag + 1, lag_limit)
        lag += 2
    tau = max(1.0, 1.0 + 2.0 * correlation_sum)
    return AutocorrelationEstimate(
        sample_count=int(samples.size),
        integrated_autocorrelation_time=tau,
        effective_sample_size=min(float(samples.size), float(samples.size) / tau),
        maximum_lag_used=used_lag,
        provenance=provenance,
    )


def moving_block_bootstrap_mean(
    values: Sequence[Real] | np.ndarray,
    *,
    block_length: int,
    resample_count: int,
    seed: int,
    confidence_level: float,
    unit: str,
    provenance: EstimatorProvenance,
) -> MovingBlockBootstrapResult:
    """Bootstrap a mean from circular moving blocks using a supplied fixed seed."""

    samples = _finite_1d(values, "values", 2)
    block = _positive_int(block_length, "block_length")
    resamples = _positive_int(resample_count, "resample_count")
    if block > samples.size:
        raise ValueError("block_length cannot exceed the sample count.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be a finite number in (0, 1).")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be a finite number in (0, 1).")
    result_unit = _nonempty_unit(unit)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")

    generator = np.random.default_rng(int(seed))
    block_count = math.ceil(samples.size / block)
    offsets = np.arange(block, dtype=np.int64)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        starts = generator.integers(0, samples.size - block + 1, size=block_count)
        indices = starts[:, None] + offsets[None, :]
        resample = samples[indices.ravel()[: samples.size]]
        means[index] = float(np.mean(resample))
    tail = 0.5 * (1.0 - confidence)
    quantiles = np.asarray(np.quantile(means, (tail, 1.0 - tail)), dtype=np.float64)
    low = float(quantiles[0])
    high = float(quantiles[1])
    standard_error = float(np.std(means, ddof=1)) if resamples > 1 else 0.0
    return MovingBlockBootstrapResult(
        sample_count=int(samples.size),
        block_length=block,
        resample_count=resamples,
        seed=int(seed),
        estimate=float(np.mean(samples)),
        standard_error=standard_error,
        confidence_level=confidence,
        confidence_interval=(low, high),
        bootstrap_means=tuple(float(value) for value in means),
        unit=result_unit,
        provenance=provenance,
    )


def first_second_half_diagnostic(
    values: Sequence[Real] | np.ndarray,
    *,
    unit: str,
    provenance: EstimatorProvenance,
) -> HalfTrajectoryDiagnostic:
    """Compare nonoverlapping first and second halves of a scalar trajectory."""

    samples = _finite_1d(values, "values", 4)
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    result_unit = _nonempty_unit(unit)
    midpoint = samples.size // 2
    first = samples[:midpoint]
    second = samples[midpoint:]
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    shift = second_mean - first_mean
    return HalfTrajectoryDiagnostic(
        sample_count=int(samples.size),
        first_half_count=int(first.size),
        second_half_count=int(second.size),
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        signed_mean_shift=shift,
        absolute_mean_shift=abs(shift),
        unit=result_unit,
        provenance=provenance,
    )


def interwalker_diagnostic(
    values: Sequence[Sequence[Real]] | np.ndarray,
    *,
    unit: str,
    provenance: EstimatorProvenance,
) -> InterwalkerDiagnostic:
    """Compare equal-length walker means without pooling away walker identity."""

    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError(
            "values must have shape (at least 2 walkers, at least 2 samples)."
        )
    if raw.dtype.kind not in "iuf":
        raise ValueError("values must contain real numeric values.")
    samples = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must contain only finite values.")
    if not isinstance(provenance, EstimatorProvenance):
        raise TypeError("provenance must be an EstimatorProvenance.")
    result_unit = _nonempty_unit(unit)
    walker_count = int(samples.shape[0])
    means = tuple(
        float(np.mean(samples[index, :], dtype=np.float64))
        for index in range(walker_count)
    )
    pooled = float(np.mean(samples))
    squared_deviations = tuple((value - pooled) ** 2 for value in means)
    standard_deviation = math.sqrt(math.fsum(squared_deviations) / (walker_count - 1))
    return InterwalkerDiagnostic(
        walker_count=walker_count,
        samples_per_walker=int(samples.shape[1]),
        pooled_mean=pooled,
        walker_means=means,
        walker_mean_standard_deviation=standard_deviation,
        maximum_absolute_walker_mean_deviation=max(
            abs(value - pooled) for value in means
        ),
        unit=result_unit,
        provenance=provenance,
    )


def _nonnegative_threshold(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return threshold


def evaluate_half_trajectory_gate(
    diagnostic: HalfTrajectoryDiagnostic,
    *,
    maximum_absolute_mean_shift: Real,
) -> HalfTrajectoryGateResult:
    """Apply a predeclared absolute-shift threshold to a half diagnostic."""

    if not isinstance(diagnostic, HalfTrajectoryDiagnostic):
        raise TypeError("diagnostic must be a HalfTrajectoryDiagnostic.")
    threshold = _nonnegative_threshold(
        maximum_absolute_mean_shift, "maximum_absolute_mean_shift"
    )
    return HalfTrajectoryGateResult(
        diagnostic=diagnostic,
        maximum_absolute_mean_shift=threshold,
        passed=diagnostic.absolute_mean_shift <= threshold,
    )


def evaluate_interwalker_gate(
    diagnostic: InterwalkerDiagnostic,
    *,
    maximum_walker_mean_standard_deviation: Real,
    maximum_absolute_walker_mean_deviation: Real,
) -> InterwalkerGateResult:
    """Apply predeclared spread thresholds to an interwalker diagnostic."""

    if not isinstance(diagnostic, InterwalkerDiagnostic):
        raise TypeError("diagnostic must be an InterwalkerDiagnostic.")
    standard_deviation_threshold = _nonnegative_threshold(
        maximum_walker_mean_standard_deviation,
        "maximum_walker_mean_standard_deviation",
    )
    maximum_deviation_threshold = _nonnegative_threshold(
        maximum_absolute_walker_mean_deviation,
        "maximum_absolute_walker_mean_deviation",
    )
    return InterwalkerGateResult(
        diagnostic=diagnostic,
        maximum_walker_mean_standard_deviation=standard_deviation_threshold,
        maximum_absolute_walker_mean_deviation=maximum_deviation_threshold,
        passed=(
            diagnostic.walker_mean_standard_deviation <= standard_deviation_threshold
            and diagnostic.maximum_absolute_walker_mean_deviation
            <= maximum_deviation_threshold
        ),
    )


def sha256_numeric_inputs(*arrays: Sequence[Real] | np.ndarray) -> str:
    """Return a deterministic SHA-256 over shapes and canonical float64 bytes."""

    if not arrays:
        raise ValueError("at least one numeric input array is required.")
    digest = hashlib.sha256()
    for index, values in enumerate(arrays):
        array = _finite_1d(values, f"arrays[{index}]", 1)
        canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "AutocorrelationEstimate",
    "BAROverlapError",
    "BARResult",
    "EV_TO_KCAL_PER_MOL",
    "EV_TO_KJ_PER_MOL",
    "EstimatorProvenance",
    "FixedWindowTIResult",
    "HalfTrajectoryDiagnostic",
    "HalfTrajectoryGateResult",
    "InterwalkerDiagnostic",
    "InterwalkerGateResult",
    "LambdaLegIntegral",
    "MovingBlockBootstrapResult",
    "ReweightingDiagnostics",
    "bidirectional_reweighting_diagnostics",
    "evaluate_half_trajectory_gate",
    "evaluate_interwalker_gate",
    "first_second_half_diagnostic",
    "fixed_window_ti_sum",
    "integrate_native_lambda_leg",
    "integrated_autocorrelation_time",
    "interwalker_diagnostic",
    "moving_block_bootstrap_mean",
    "sha256_numeric_inputs",
    "solve_adjacent_bar",
]
