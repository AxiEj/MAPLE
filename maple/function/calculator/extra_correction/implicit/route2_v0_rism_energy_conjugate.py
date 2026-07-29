"""Energy-conjugate periodic RISM kernel for the Route-2 V0 controls.

A parsed Amber 1D-RISM direct correlation has the source-defined split

``c_ab(r) = c_ab^sr(r) - q_a q_b erf(r / eta) / r``.

The finite short-range Cartesian transform and the periodic Coulomb operator
must therefore enter one scalar together.  This module constructs the
reciprocal full kernel

``C_full_ab(k) = C_sr_ab(k) - q_a q_b G_eta(k)``,

where ``G_eta`` is the same zero-average smeared ``4*pi/k^2`` multiplier used
by :class:`Route2V0PeriodicCoulombOperator`.  Its excess scalar is equivalently

``F_ex = -kBT/2 <delta_n, C_sr * delta_n>
        + kBT/2 <rho_Q, G_eta * rho_Q>``.

Consequently its site derivative is ``-C_sr * delta_n + q_a V_Q``.  The
constructed full periodic direct correlation gives exactly that derivative
through the existing HNC convolution pairing.

This is a representation/control layer only.  It does not turn the local
cSPC/E parser control into a complete physical solvent asset, synthesize a
solute--solvent short-range interaction, choose a pressure correction, or make
any solvation, force, PES, or accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase.units import Bohr, Hartree, kB

from .route2_v0_periodic_coulomb import (
    Route2V0PeriodicCoulombOperator,
    amber_rism_qv_to_sqrt_bohr,
)
from .route2_v0_rism_reciprocal import Route2V0RismShortRangeReciprocalControl
from .route2_v0_site_hnc import Route2V0SiteHNCAsset, Route2V0SiteHNCFunctional
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION = "route2-v0-rism-energy-conjugate-v1"


def _same_grid(left: RegularCartesianGrid, right: RegularCartesianGrid) -> bool:
    """Return whether both periodic operators use one exact Cartesian grid."""

    return bool(
        left.shape == right.shape
        and left.layout == right.layout
        and np.array_equal(left.origin_bohr, right.origin_bohr)
        and np.array_equal(left.spacing_bohr, right.spacing_bohr)
    )


def _immutable_real_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return one finite immutable real array with an exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0RismEnergyConjugateKernel:
    """One complete periodic RISM correlation representation on a fixed grid.

    The input short-range control is already gated for native radial origin,
    tail, Nyquist, and reciprocal pairing.  This layer checks that its native
    SMEAR and QV conventions are joined to exactly the matching periodic
    Coulomb operator, so a caller cannot attach an arbitrary electrostatic
    prefactor or a separately tuned long-range term.
    """

    short_range: Route2V0RismShortRangeReciprocalControl
    periodic_coulomb: Route2V0PeriodicCoulombOperator
    construction: str = V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION
    _site_charge_scales_sqrt_bohr: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _bulk_number_density_bohr3: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _kbt_hartree: float = field(init=False, repr=False, compare=False)
    _short_range_asset: Route2V0SiteHNCAsset = field(
        init=False,
        repr=False,
        compare=False,
    )
    _full_asset: Route2V0SiteHNCAsset = field(init=False, repr=False, compare=False)
    _short_range_functional: Route2V0SiteHNCFunctional = field(
        init=False,
        repr=False,
        compare=False,
    )
    _full_functional: Route2V0SiteHNCFunctional = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.short_range, Route2V0RismShortRangeReciprocalControl):
            raise TypeError(
                "Energy-conjugate RISM kernel requires a short-range control."
            )
        if not isinstance(self.periodic_coulomb, Route2V0PeriodicCoulombOperator):
            raise TypeError("Energy-conjugate RISM kernel requires a Coulomb operator.")
        if self.construction != V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 energy-conjugate RISM construction.")
        if not _same_grid(self.short_range.grid, self.periodic_coulomb.grid):
            raise ValueError(
                "Energy-conjugate RISM short-range and periodic Coulomb grids differ."
            )

        metadata = self.short_range.radial.metadata
        expected_smear_bohr = metadata.coulomb_smear_angstrom / Bohr
        smear_scale = max(1.0, abs(expected_smear_bohr))
        if not math.isclose(
            self.periodic_coulomb.smear_bohr,
            expected_smear_bohr,
            rel_tol=0.0,
            abs_tol=1.0e-12 * smear_scale,
        ):
            raise ValueError(
                "Energy-conjugate RISM periodic Coulomb SMEAR must match the "
                "native short-range source SMEAR."
            )

        charge_scales = amber_rism_qv_to_sqrt_bohr(
            metadata.site_charges_sqrt_kT_angstrom
        )
        bulk = _immutable_real_array(
            metadata.bulk_number_density_angstrom3 * Bohr**3,
            name="Energy-conjugate RISM bulk site density",
            shape=(metadata.site_count,),
        )
        kbt = float(kB * metadata.temperature_kelvin / Hartree)
        if not math.isfinite(kbt) or kbt <= 0.0:
            raise RuntimeError("Energy-conjugate RISM kBT conversion is invalid.")

        short_range_asset = Route2V0SiteHNCAsset(
            grid=self.short_range.grid,
            site_names=metadata.site_names,
            bulk_number_density_bohr3=bulk,
            direct_correlation_dimensionless=(
                self.short_range.direct_correlation_dimensionless
            ),
            kbt_hartree=kbt,
        )
        full_direct = self._compose_full_direct_correlation(charge_scales)
        full_asset = Route2V0SiteHNCAsset(
            grid=self.short_range.grid,
            site_names=metadata.site_names,
            bulk_number_density_bohr3=bulk,
            direct_correlation_dimensionless=full_direct,
            kbt_hartree=kbt,
        )
        zero_external = np.zeros((metadata.site_count, *self.short_range.grid.shape))
        short_range_functional = Route2V0SiteHNCFunctional(
            short_range_asset,
            zero_external,
        )
        full_functional = Route2V0SiteHNCFunctional(full_asset, zero_external)

        object.__setattr__(self, "_site_charge_scales_sqrt_bohr", charge_scales)
        object.__setattr__(self, "_bulk_number_density_bohr3", bulk)
        object.__setattr__(self, "_kbt_hartree", kbt)
        object.__setattr__(self, "_short_range_asset", short_range_asset)
        object.__setattr__(self, "_full_asset", full_asset)
        object.__setattr__(self, "_short_range_functional", short_range_functional)
        object.__setattr__(self, "_full_functional", full_functional)

    def _compose_full_direct_correlation(
        self,
        charge_scales_sqrt_bohr: np.ndarray,
    ) -> np.ndarray:
        """Return the full reciprocal kernel from the exact short/long split."""

        grid = self.short_range.grid
        short_range_fourier = np.fft.fftn(
            self.short_range.direct_correlation_dimensionless,
            axes=(2, 3, 4),
        )
        long_range_fourier = (
            -np.einsum(
                "a,b,xyz->abxyz",
                charge_scales_sqrt_bohr,
                charge_scales_sqrt_bohr,
                self.periodic_coulomb.fourier_green_bohr2,
                optimize=True,
            )
            / grid.volume_element_bohr3
        )
        full_complex = np.fft.ifftn(
            short_range_fourier + long_range_fourier,
            axes=(2, 3, 4),
        )
        imaginary_error = float(np.max(np.abs(np.imag(full_complex))))
        scale = max(1.0, float(np.max(np.abs(full_complex))))
        if imaginary_error > 1.0e-12 * scale:
            raise RuntimeError(
                "Energy-conjugate RISM full direct correlation is unexpectedly complex."
            )
        result = np.array(np.real(full_complex), dtype=float, copy=True)
        result.setflags(write=False)
        return result

    @property
    def grid(self) -> RegularCartesianGrid:
        """Return the one shared periodic Cartesian grid."""

        return self.short_range.grid

    @property
    def site_charge_scales_sqrt_bohr(self) -> np.ndarray:
        """Return the native-QV scales converted only to Bohr length units."""

        return self._site_charge_scales_sqrt_bohr

    @property
    def bulk_number_density_bohr3(self) -> np.ndarray:
        """Return bulk site densities converted from the source XVV convention."""

        return self._bulk_number_density_bohr3

    @property
    def site_multiplicity(self) -> np.ndarray:
        """Return the immutable source-declared count of each site type per molecule."""

        return self.short_range.radial.metadata.site_multiplicity

    @property
    def kbt_hartree(self) -> float:
        """Return source-temperature ``kBT`` in Hartree."""

        return self._kbt_hartree

    @property
    def direct_correlation_dimensionless(self) -> np.ndarray:
        """Return the full periodic direct correlation for the existing HNC pairing."""

        return self._full_asset.direct_correlation_dimensionless

    @property
    def site_hnc_asset(self) -> Route2V0SiteHNCAsset:
        """Return the full energy-conjugate HNC asset for a later liquid solve."""

        return self._full_asset

    def _density_difference(self, values: np.ndarray) -> np.ndarray:
        return _immutable_real_array(
            values,
            name="Energy-conjugate RISM site density difference",
            shape=(self.short_range.radial.metadata.site_count, *self.grid.shape),
        )

    def short_range_convolution(
        self, density_difference_bohr3: np.ndarray
    ) -> np.ndarray:
        """Return ``c_sr * delta_n`` in the exact existing HNC convention."""

        difference = self._density_difference(density_difference_bohr3)
        return self._short_range_functional.convolve_direct_correlation(difference)

    def full_convolution(self, density_difference_bohr3: np.ndarray) -> np.ndarray:
        """Return ``c_full * delta_n`` in the exact existing HNC convention."""

        difference = self._density_difference(density_difference_bohr3)
        return self._full_functional.convolve_direct_correlation(difference)

    def dimensionless_excess_gradient(
        self,
        density_difference_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Return the derivative of the split excess scalar at each site/grid point."""

        difference = self._density_difference(density_difference_bohr3)
        short_range = self.short_range_convolution(difference)
        long_range = self.periodic_coulomb.site_kernel_from_density_difference(
            difference,
            self.site_charge_scales_sqrt_bohr,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        result = np.array(-short_range + long_range, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def excess_energy_components_hartree(
        self,
        density_difference_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> tuple[float, float]:
        """Return the short-range and positive long-range excess components."""

        difference = self._density_difference(density_difference_bohr3)
        short_range = float(
            -0.5
            * self.kbt_hartree
            * self.grid.volume_element_bohr3
            * np.sum(difference * self.short_range_convolution(difference))
        )
        long_range = float(
            self.kbt_hartree
            * self.periodic_coulomb.dimensionless_energy_from_site_density_difference(
                difference,
                self.site_charge_scales_sqrt_bohr,
                neutrality_relative_tolerance=neutrality_relative_tolerance,
            )
        )
        if not all(math.isfinite(value) for value in (short_range, long_range)):
            raise RuntimeError(
                "Energy-conjugate RISM excess components are non-finite."
            )
        return short_range, long_range

    def excess_energy_hartree(
        self,
        density_difference_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return the one split scalar for the complete periodic RISM kernel."""

        return float(
            sum(
                self.excess_energy_components_hartree(
                    density_difference_bohr3,
                    neutrality_relative_tolerance=neutrality_relative_tolerance,
                )
            )
        )


__all__ = [
    "Route2V0RismEnergyConjugateKernel",
    "V0_RISM_ENERGY_CONJUGATE_CONSTRUCTION",
]
