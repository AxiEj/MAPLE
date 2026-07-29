"""Reciprocal reference mapping for a smooth 1D-RISM short-range Cvv asset.

The source-defined Amber ``SMEAR`` split makes ``c^sr_ab(r)`` finite at the
origin.  This module maps that radial short-range remainder to a periodic
Cartesian kernel through its declared spherical Fourier transform,

``C_ab(k) = 4*pi int r^2 c^sr_ab(r) sinc(k r) dr``.

The resulting grid array is normalized so that the existing discrete HNC
convolution ``dv * ifft(fft(c) * fft(delta_rho))`` uses exactly ``C_ab(k)`` as
its multiplier.  It is a small-grid mathematical/admission control only: it
does not select a tail tolerance from solvation errors, supply a short-range
solute potential, or make a physical liquid/force/accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase.units import Bohr

from .route2_v0_rism_bulk import Route2V0RismShortRangeDirectCorrelation
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION = (
    "route2-v0-rism-short-range-reciprocal-control-v1"
)


def _immutable_real_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return one finite immutable real array with an optional exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"a finite array with {shape}"
        raise ValueError(f"{name} must be {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _reverse_displacement(values: np.ndarray) -> np.ndarray:
    """Return one periodic kernel evaluated at negative displacement."""

    indices = np.ix_(*((-np.arange(extent)) % extent for extent in values.shape))
    return values[indices]


def _validated_tail_controls(
    radial: Route2V0RismShortRangeDirectCorrelation,
    *,
    tail_start_angstrom: float,
    tail_tolerance_dimensionless: float,
) -> tuple[float, float, float]:
    """Validate a preregistered source-tail truncation control."""

    start = float(tail_start_angstrom)
    tolerance = float(tail_tolerance_dimensionless)
    radii = radial.radii_angstrom
    if not math.isfinite(start) or start < 0.0 or start > float(radii[-1]):
        raise ValueError(
            "RISM short-range tail start must be finite and within the source radial grid."
        )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("RISM short-range tail tolerance must be finite and positive.")
    indices = radii >= start
    if not np.any(indices):
        raise ValueError("RISM short-range tail region has no radial samples.")
    maximum = float(np.max(np.abs(radial.values_dimensionless[..., indices])))
    if maximum > tolerance:
        raise ValueError(
            "RISM short-range tail exceeds its preregistered truncation tolerance."
        )
    return start, tolerance, maximum


def _radial_quadrature_weights(
    radial: Route2V0RismShortRangeDirectCorrelation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Bohr radii and trapezoid weights for the spherical transform."""

    radii_bohr = np.asarray(radial.radii_angstrom / Bohr, dtype=float)
    spacing = np.diff(radii_bohr)
    expected = float(spacing[0])
    scale = max(1.0, abs(expected))
    if np.max(np.abs(spacing - expected)) > 1.0e-12 * scale:
        raise ValueError(
            "RISM reciprocal control requires a uniform radial source grid."
        )
    if abs(float(radii_bohr[0])) > 1.0e-12 * scale:
        raise ValueError(
            "RISM reciprocal control requires the source-defined r=0 short-range value."
        )
    weights = np.full(radii_bohr.size, expected, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    weights *= radii_bohr**2
    return radii_bohr, weights


def _reciprocal_wavevector_squared(grid: RegularCartesianGrid) -> np.ndarray:
    """Return ``|k|^2`` in FFT order for one periodic Cartesian cell."""

    axes = tuple(
        2.0 * math.pi * np.fft.fftfreq(grid.shape[axis], d=grid.spacing_bohr[axis])
        for axis in range(3)
    )
    return sum(axis_values**2 for axis_values in np.meshgrid(*axes, indexing="ij"))


def _short_range_fourier_kernel(
    radial: Route2V0RismShortRangeDirectCorrelation,
    grid: RegularCartesianGrid,
) -> np.ndarray:
    """Return periodic real-space samples with the declared radial multiplier."""

    radii_bohr, weights = _radial_quadrature_weights(radial)
    wavevector_squared = _reciprocal_wavevector_squared(grid)
    maximum_wavevector = math.sqrt(float(np.max(wavevector_squared)))
    radial_nyquist = math.pi / float(radii_bohr[1] - radii_bohr[0])
    if maximum_wavevector > radial_nyquist * (1.0 + 1.0e-12):
        raise ValueError(
            "Cartesian reciprocal grid exceeds the source radial Nyquist wavevector."
        )

    unique_squared, inverse = np.unique(
        wavevector_squared.reshape(-1),
        return_inverse=True,
    )
    site_count = radial.metadata.site_count
    transformed = np.empty((site_count, site_count, unique_squared.size), dtype=float)
    weighted_values = radial.values_dimensionless * weights
    chunk_size = 256
    for first in range(0, unique_squared.size, chunk_size):
        last = min(first + chunk_size, unique_squared.size)
        wavevectors = np.sqrt(unique_squared[first:last])
        sinc = np.sinc(np.outer(wavevectors, radii_bohr) / math.pi)
        transformed[..., first:last] = (
            4.0
            * math.pi
            * np.einsum(
                "kr,abr->abk",
                sinc,
                weighted_values,
                optimize=True,
            )
        )

    reciprocal = transformed[..., inverse].reshape(
        (site_count, site_count, *grid.shape)
    )
    periodic_complex = np.fft.ifftn(
        reciprocal / grid.volume_element_bohr3,
        axes=(2, 3, 4),
    )
    imaginary_error = float(np.max(np.abs(np.imag(periodic_complex))))
    scale = max(1.0, float(np.max(np.abs(periodic_complex))))
    if imaginary_error > 1.0e-12 * scale:
        raise RuntimeError("RISM reciprocal control produced a complex kernel.")
    return np.real(periodic_complex)


@dataclass(frozen=True)
class Route2V0RismShortRangeReciprocalControl:
    """One audited periodic representation of a smooth radial Cvv remainder."""

    radial: Route2V0RismShortRangeDirectCorrelation
    grid: RegularCartesianGrid
    tail_start_angstrom: float
    tail_tolerance_dimensionless: float
    direct_correlation_dimensionless: np.ndarray
    construction: str = V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION
    maximum_tail_abs: float = field(init=False)

    def __post_init__(self) -> None:
        if self.construction != V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 RISM reciprocal control construction."
            )
        start, tolerance, maximum = _validated_tail_controls(
            self.radial,
            tail_start_angstrom=self.tail_start_angstrom,
            tail_tolerance_dimensionless=self.tail_tolerance_dimensionless,
        )
        count = self.radial.metadata.site_count
        direct = _immutable_real_array(
            self.direct_correlation_dimensionless,
            name="RISM reciprocal-control direct correlation",
            shape=(count, count, *self.grid.shape),
        )
        reciprocity_error = 0.0
        for left in range(count):
            for right in range(count):
                difference = direct[left, right] - _reverse_displacement(
                    direct[right, left]
                )
                reciprocity_error = max(
                    reciprocity_error,
                    float(np.max(np.abs(difference))),
                )
        scale = max(1.0, float(np.max(np.abs(direct))))
        if reciprocity_error > 1.0e-12 * scale:
            raise ValueError(
                "RISM reciprocal-control direct correlation must be reciprocal."
            )
        object.__setattr__(self, "tail_start_angstrom", start)
        object.__setattr__(self, "tail_tolerance_dimensionless", tolerance)
        object.__setattr__(self, "maximum_tail_abs", maximum)
        object.__setattr__(self, "direct_correlation_dimensionless", direct)

    @classmethod
    def from_radial(
        cls,
        *,
        radial: Route2V0RismShortRangeDirectCorrelation,
        grid: RegularCartesianGrid,
        tail_start_angstrom: float,
        tail_tolerance_dimensionless: float,
    ) -> "Route2V0RismShortRangeReciprocalControl":
        """Build a source-grid quadrature control without radial interpolation."""

        _validated_tail_controls(
            radial,
            tail_start_angstrom=tail_start_angstrom,
            tail_tolerance_dimensionless=tail_tolerance_dimensionless,
        )
        direct = _short_range_fourier_kernel(radial, grid)
        return cls(
            radial=radial,
            grid=grid,
            tail_start_angstrom=tail_start_angstrom,
            tail_tolerance_dimensionless=tail_tolerance_dimensionless,
            direct_correlation_dimensionless=direct,
        )


__all__ = [
    "Route2V0RismShortRangeReciprocalControl",
    "V0_RISM_SHORT_RANGE_RECIPROCAL_CONSTRUCTION",
]
