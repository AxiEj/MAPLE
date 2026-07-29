"""Neutral periodic Coulomb operator for the Route-2 V0 liquid reference.

The radial 1D-RISM direct correlation has the native long-range convention
``c_ab^lr(r) = -q_a q_b / r``.  This module supplies the *positive*
``q_a q_b / r`` term as one reciprocal, zero-average periodic Poisson
operator.  Its scalar and derivative are deliberately implemented together:

``E_lr = 1/2 int rho_Q(r) V_Q(r) dr`` and
``delta E_lr / delta delta_rho_a = q_a V_Q``.

Here ``rho_Q = sum_a q_a delta_rho_a`` and ``V_Q`` solves the periodic
Poisson equation with its zero Fourier mode omitted.  The input must therefore
be neutral; silently adding a uniform compensating background would introduce
a different free-energy convention.  The operator is a mathematical control
for the future RISM long-range term only.  It does not interpolate a radial
short-range Cvv asset, define a physical liquid functional, or make a
solvation/force claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase.units import Bohr

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_PERIODIC_COULOMB_CONSTRUCTION = "route2-v0-periodic-coulomb-poisson-v1"


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


def amber_rism_qv_to_sqrt_bohr(
    qv_sqrt_kT_angstrom: np.ndarray,
) -> np.ndarray:
    """Convert Amber's native QV radial scale from Angstrom to Bohr length.

    Amber labels QV as ``sqrt(kT A)``.  The only convention used here is its
    documented direct-correlation tail: ``q_a q_b / r_A`` is dimensionless.
    Thus converting the radial coordinate to Bohr requires exactly
    ``q_a -> q_a / sqrt(Bohr)``.  This routine does not reinterpret QV as an
    elementary charge or introduce a new electrostatic prefactor.
    """

    qv = _immutable_real_array(qv_sqrt_kT_angstrom, name="Amber RISM QV scale")
    if qv.ndim != 1 or qv.size < 1:
        raise ValueError(
            "Amber RISM QV scale must be a nonempty one-dimensional array."
        )
    result = np.array(qv / math.sqrt(Bohr), copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0PeriodicCoulombOperator:
    """One neutral, zero-average periodic inverse-distance operator.

    ``grid`` defines the periodic cell as ``shape * spacing`` in Bohr.  The
    Fourier Green function is ``4*pi/k^2`` for all nonzero reciprocal vectors
    and is exactly zero at ``k=0``.  Inputs are rejected unless their integral
    is neutral within the declared relative tolerance, so that the omitted
    mode is a gauge choice rather than an unannounced background convention.
    """

    grid: RegularCartesianGrid
    construction: str = V0_PERIODIC_COULOMB_CONSTRUCTION
    _fourier_green_bohr2: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.construction != V0_PERIODIC_COULOMB_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 periodic Coulomb construction.")
        reciprocal_axes = tuple(
            2.0
            * math.pi
            * np.fft.fftfreq(self.grid.shape[axis], d=self.grid.spacing_bohr[axis])
            for axis in range(3)
        )
        wavevector_squared = sum(
            axis_values**2
            for axis_values in np.meshgrid(*reciprocal_axes, indexing="ij")
        )
        green = np.zeros(self.grid.shape, dtype=float)
        nonzero = wavevector_squared > 0.0
        green[nonzero] = 4.0 * math.pi / wavevector_squared[nonzero]
        green.setflags(write=False)
        object.__setattr__(self, "_fourier_green_bohr2", green)

    @property
    def cell_lengths_bohr(self) -> np.ndarray:
        """Return the periodic cell lengths fixed by the grid geometry."""

        result = np.asarray(self.grid.shape, dtype=float) * self.grid.spacing_bohr
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result

    @property
    def fourier_green_bohr2(self) -> np.ndarray:
        """Return the immutable ``4*pi/k^2`` multiplier with zero mode omitted."""

        return self._fourier_green_bohr2

    def _validate_charge_density(self, values: np.ndarray) -> np.ndarray:
        return _immutable_real_array(
            values,
            name="Periodic Coulomb charge density",
            shape=self.grid.shape,
        )

    def integrated_charge(self, charge_density: np.ndarray) -> float:
        """Return the grid integral of one scaled charge-density field."""

        density = self._validate_charge_density(charge_density)
        return float(self.grid.volume_element_bohr3 * np.sum(density))

    def _require_neutral(
        self,
        charge_density: np.ndarray,
        *,
        relative_tolerance: float,
    ) -> np.ndarray:
        tolerance = float(relative_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Periodic Coulomb neutrality tolerance must be finite in (0, 1)."
            )
        density = self._validate_charge_density(charge_density)
        volume = self.grid.volume_element_bohr3
        net_charge = float(volume * np.sum(density))
        scale = max(1.0, float(volume * np.sum(np.abs(density))))
        if abs(net_charge) > tolerance * scale:
            raise ValueError(
                "Periodic Coulomb charge density must be neutral; a uniform "
                "background is not an admitted Route-2 V0 convention."
            )
        return density

    def potential_from_charge_density(
        self,
        charge_density: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Apply the zero-average periodic Poisson inverse to a neutral field."""

        density = self._require_neutral(
            charge_density,
            relative_tolerance=neutrality_relative_tolerance,
        )
        potential_complex = np.fft.ifftn(
            self._fourier_green_bohr2 * np.fft.fftn(density)
        )
        imaginary_error = float(np.max(np.abs(np.imag(potential_complex))))
        scale = max(1.0, float(np.max(np.abs(potential_complex))))
        if imaginary_error > 1.0e-12 * scale:
            raise RuntimeError(
                "Periodic Coulomb Poisson solve produced a complex field."
            )
        potential = np.real(potential_complex)
        # The omitted k=0 Fourier coefficient fixes this gauge.  The explicit
        # subtraction only removes roundoff and preserves the linear operator.
        potential -= float(np.mean(potential))
        potential.setflags(write=False)
        return potential

    def dimensionless_energy(
        self,
        charge_density: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return ``1/2 int rho_Q V_Q`` from the same Poisson operator."""

        density = self._require_neutral(
            charge_density,
            relative_tolerance=neutrality_relative_tolerance,
        )
        potential = self.potential_from_charge_density(
            density,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        energy = (
            0.5 * self.grid.volume_element_bohr3 * float(np.sum(density * potential))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Periodic Coulomb scalar is non-finite.")
        return energy

    def _validate_site_density_and_scales(
        self,
        density_difference_bohr3: np.ndarray,
        site_charge_scales_sqrt_bohr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        density = _immutable_real_array(
            density_difference_bohr3,
            name="Periodic Coulomb site density difference",
        )
        if (
            density.ndim != 4
            or density.shape[0] < 1
            or density.shape[1:] != self.grid.shape
        ):
            raise ValueError(
                "Periodic Coulomb site density difference must have shape "
                f"(site, {self.grid.shape[0]}, {self.grid.shape[1]}, "
                f"{self.grid.shape[2]})."
            )
        scales = _immutable_real_array(
            site_charge_scales_sqrt_bohr,
            name="Periodic Coulomb site charge scales",
            shape=(density.shape[0],),
        )
        return density, scales

    def weighted_charge_density_from_site_density_difference(
        self,
        density_difference_bohr3: np.ndarray,
        site_charge_scales_sqrt_bohr: np.ndarray,
    ) -> np.ndarray:
        """Return ``rho_Q = sum_a q_a delta_rho_a`` on this periodic grid."""

        density, scales = self._validate_site_density_and_scales(
            density_difference_bohr3,
            site_charge_scales_sqrt_bohr,
        )
        result = np.einsum("a,axyz->xyz", scales, density, optimize=True)
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result

    def site_kernel_from_density_difference(
        self,
        density_difference_bohr3: np.ndarray,
        site_charge_scales_sqrt_bohr: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Return ``q_a V_Q``: the derivative of the declared scalar term."""

        density, scales = self._validate_site_density_and_scales(
            density_difference_bohr3,
            site_charge_scales_sqrt_bohr,
        )
        weighted = self.weighted_charge_density_from_site_density_difference(
            density,
            scales,
        )
        potential = self.potential_from_charge_density(
            weighted,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        result = scales.reshape((-1, 1, 1, 1)) * potential
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result

    def dimensionless_energy_from_site_density_difference(
        self,
        density_difference_bohr3: np.ndarray,
        site_charge_scales_sqrt_bohr: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return the scalar whose site derivative is ``q_a V_Q``."""

        weighted = self.weighted_charge_density_from_site_density_difference(
            density_difference_bohr3,
            site_charge_scales_sqrt_bohr,
        )
        return self.dimensionless_energy(
            weighted,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )


__all__ = [
    "Route2V0PeriodicCoulombOperator",
    "V0_PERIODIC_COULOMB_CONSTRUCTION",
    "amber_rism_qv_to_sqrt_bohr",
]
