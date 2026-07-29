"""Variational multi-site HNC reference functional for Route-2 V0-FD-S.

This module provides a mathematical reference kernel for a molecular-liquid
backend that consumes an externally supplied grid potential and a frozen,
reciprocal site--site direct-correlation asset.  For site densities
``rho_a(r)``, bulk densities ``rho_a^b``, direct correlations ``c_ab``, and
external potentials ``u_a``, it evaluates the HNC grand-potential difference

``Omega = kBT * int[rho log(rho/rho_b) - delta_rho]
        - kBT/2 * int[delta_rho_a c_ab delta_rho_b]
        + int[rho_a u_a]``.

The stationary equation is the gradient of that same scalar.  This is a
synthetic/reference engine only: it carries no real solvent susceptibility,
short-range MACE--solvent interaction, pressure correction, standard-state
term, or solvation accuracy claim.  The Picard mixing parameter is numerical
acceleration, never part of the physical functional.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_SITE_HNC_CONSTRUCTION = "route2-v0-site-hnc-variational-reference-v1"


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
) -> np.ndarray:
    """Return one finite immutable floating-point array."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        expected = "a finite array" if shape is None else f"a finite array with {shape}"
        if positive:
            expected += " and strictly positive entries"
        raise ValueError(f"{name} must be {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _site_names(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a stable ordered site-name sequence."""

    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("HNC site names must be nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("HNC site names must be unique.")
    return names


def _reverse_displacement(values: np.ndarray) -> np.ndarray:
    """Return a periodic kernel evaluated at negative displacement."""

    indices = np.ix_(*((-np.arange(extent)) % extent for extent in values.shape))
    return values[indices]


@dataclass(frozen=True)
class Route2V0SiteHNCAsset:
    """One frozen reciprocal direct-correlation asset on a regular grid.

    ``direct_correlation_dimensionless[a, b]`` uses FFT displacement order:
    zero displacement is index ``[0, 0, 0]`` and negative displacements wrap
    around the three spatial axes.  The invariant
    ``c_ab(r) == c_ba(-r)`` makes the discrete excess functional scalar.
    """

    grid: RegularCartesianGrid
    site_names: tuple[str, ...]
    bulk_number_density_bohr3: np.ndarray
    direct_correlation_dimensionless: np.ndarray
    kbt_hartree: float
    construction: str = V0_SITE_HNC_CONSTRUCTION

    def __post_init__(self) -> None:
        names = _site_names(self.site_names)
        count = len(names)
        bulk = _immutable_array(
            self.bulk_number_density_bohr3,
            name="HNC bulk number density",
            shape=(count,),
            positive=True,
        )
        direct = _immutable_array(
            self.direct_correlation_dimensionless,
            name="HNC direct correlation",
            shape=(count, count, *self.grid.shape),
        )
        kbt = float(self.kbt_hartree)
        if not math.isfinite(kbt) or kbt <= 0.0:
            raise ValueError("HNC kBT must be finite and positive in Hartree.")
        if self.construction != V0_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 HNC construction.")

        maximum_error = 0.0
        for left in range(count):
            for right in range(count):
                difference = direct[left, right] - _reverse_displacement(
                    direct[right, left]
                )
                maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
        scale = max(1.0, float(np.max(np.abs(direct))))
        if maximum_error > 1.0e-12 * scale:
            raise ValueError("HNC direct correlations must obey c_ab(r) == c_ba(-r).")

        object.__setattr__(self, "site_names", names)
        object.__setattr__(self, "bulk_number_density_bohr3", bulk)
        object.__setattr__(self, "direct_correlation_dimensionless", direct)
        object.__setattr__(self, "kbt_hartree", kbt)

    @property
    def site_count(self) -> int:
        """Return the number of liquid site densities."""

        return len(self.site_names)


@dataclass(frozen=True)
class Route2V0SiteHNCState:
    """One stationary solution of the declared discrete HNC functional."""

    density_bohr3: np.ndarray
    grand_potential_hartree: float
    ideal_contribution_hartree: float
    excess_contribution_hartree: float
    external_contribution_hartree: float
    residual_inf: float
    iterations: int
    construction: str = V0_SITE_HNC_CONSTRUCTION

    def __post_init__(self) -> None:
        density = _immutable_array(
            self.density_bohr3,
            name="HNC stationary density",
            positive=True,
        )
        energies = (
            "grand_potential_hartree",
            "ideal_contribution_hartree",
            "excess_contribution_hartree",
            "external_contribution_hartree",
            "residual_inf",
        )
        for name in energies:
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name == "residual_inf" and value < 0.0):
                raise ValueError(f"HNC state {name} must be finite.")
            object.__setattr__(self, name, value)
        try:
            iterations = int(self.iterations)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "HNC iteration count must be a nonnegative integer."
            ) from exc
        if (
            isinstance(self.iterations, (bool, np.bool_))
            or iterations != self.iterations
            or iterations < 0
        ):
            raise ValueError("HNC iteration count must be nonnegative.")
        if self.construction != V0_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 HNC construction.")
        object.__setattr__(self, "density_bohr3", density)
        object.__setattr__(self, "iterations", iterations)


@dataclass(frozen=True)
class Route2V0SiteHNCFunctional:
    """One external-potential instance of the frozen multi-site HNC scalar."""

    asset: Route2V0SiteHNCAsset
    external_potential_hartree: np.ndarray
    construction: str = V0_SITE_HNC_CONSTRUCTION

    def __post_init__(self) -> None:
        potential = _immutable_array(
            self.external_potential_hartree,
            name="HNC external potential",
            shape=(self.asset.site_count, *self.asset.grid.shape),
        )
        if self.construction != V0_SITE_HNC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 HNC construction.")
        object.__setattr__(self, "external_potential_hartree", potential)

    @property
    def _bulk_grid(self) -> np.ndarray:
        return self.asset.bulk_number_density_bohr3.reshape(
            (self.asset.site_count, 1, 1, 1)
        )

    def _validate_density(self, values: np.ndarray) -> np.ndarray:
        return _immutable_array(
            values,
            name="HNC density",
            shape=(self.asset.site_count, *self.asset.grid.shape),
            positive=True,
        )

    def convolve_direct_correlation(
        self, density_delta_bohr3: np.ndarray
    ) -> np.ndarray:
        """Return ``sum_b c_ab * delta_rho_b`` in dimensionless units."""

        delta = _immutable_array(
            density_delta_bohr3,
            name="HNC density difference",
            shape=(self.asset.site_count, *self.asset.grid.shape),
        )
        direct_fft = np.fft.fftn(
            self.asset.direct_correlation_dimensionless,
            axes=(2, 3, 4),
        )
        density_fft = np.fft.fftn(delta, axes=(1, 2, 3))
        convolution = self.asset.grid.volume_element_bohr3 * np.fft.ifftn(
            np.einsum("abxyz,bxyz->axyz", direct_fft, density_fft, optimize=True),
            axes=(1, 2, 3),
        )
        imaginary_error = float(np.max(np.abs(np.imag(convolution))))
        scale = max(1.0, float(np.max(np.abs(convolution))))
        if imaginary_error > 1.0e-12 * scale:
            raise RuntimeError("HNC reciprocal convolution produced a complex field.")
        result = np.real(convolution)
        result.setflags(write=False)
        return result

    def dimensionless_gradient(self, density_bohr3: np.ndarray) -> np.ndarray:
        """Return ``beta * delta Omega / delta rho`` on the grid."""

        density = self._validate_density(density_bohr3)
        bulk = self._bulk_grid
        gradient = (
            np.log(density / bulk)
            - self.convolve_direct_correlation(density - bulk)
            + self.external_potential_hartree / self.asset.kbt_hartree
        )
        gradient.setflags(write=False)
        return gradient

    def energy_components(
        self, density_bohr3: np.ndarray
    ) -> tuple[float, float, float]:
        """Return ideal, HNC-excess, and external scalar contributions."""

        density = self._validate_density(density_bohr3)
        bulk = self._bulk_grid
        delta = density - bulk
        volume = self.asset.grid.volume_element_bohr3
        ideal = float(
            self.asset.kbt_hartree
            * volume
            * np.sum(density * np.log(density / bulk) - delta)
        )
        excess = float(
            -0.5
            * self.asset.kbt_hartree
            * volume
            * np.sum(delta * self.convolve_direct_correlation(delta))
        )
        external = float(volume * np.sum(density * self.external_potential_hartree))
        if not all(math.isfinite(value) for value in (ideal, excess, external)):
            raise RuntimeError("HNC free-energy components are non-finite.")
        return ideal, excess, external

    def grand_potential_hartree(self, density_bohr3: np.ndarray) -> float:
        """Return the one scalar whose gradient defines the HNC stationarity."""

        return float(sum(self.energy_components(density_bohr3)))

    def stationary_state(
        self,
        density_bohr3: np.ndarray,
        *,
        iterations: int,
    ) -> Route2V0SiteHNCState:
        """Build one auditable state without claiming convergence by itself."""

        density = self._validate_density(density_bohr3)
        residual_inf = float(np.max(np.abs(self.dimensionless_gradient(density))))
        ideal, excess, external = self.energy_components(density)
        return Route2V0SiteHNCState(
            density_bohr3=density,
            grand_potential_hartree=ideal + excess + external,
            ideal_contribution_hartree=ideal,
            excess_contribution_hartree=excess,
            external_contribution_hartree=external,
            residual_inf=residual_inf,
            iterations=iterations,
        )

    def solve_picard(
        self,
        *,
        initial_density_bohr3: np.ndarray | None = None,
        residual_tolerance: float = 1.0e-10,
        picard_mixing: float = 0.2,
        max_iterations: int = 1000,
    ) -> Route2V0SiteHNCState:
        """Solve the scalar HNC stationarity equation with numerical Picard mixing.

        The physical equation is always the un-mixed functional gradient.  The
        mixing controls only numerical iteration and cannot be reported as a
        solvent-model parameter.
        """

        tolerance = float(residual_tolerance)
        mixing = float(picard_mixing)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("HNC residual tolerance must be finite and positive.")
        if not math.isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("HNC Picard mixing must be in (0, 1].")
        try:
            maximum = int(max_iterations)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "HNC maximum iterations must be a positive integer."
            ) from exc
        if (
            isinstance(max_iterations, (bool, np.bool_))
            or maximum != max_iterations
            or maximum < 1
        ):
            raise ValueError("HNC maximum iterations must be a positive integer.")

        if initial_density_bohr3 is None:
            density = np.broadcast_to(
                self._bulk_grid,
                (self.asset.site_count, *self.asset.grid.shape),
            ).copy()
        else:
            density = np.array(self._validate_density(initial_density_bohr3), copy=True)

        for iteration in range(maximum + 1):
            state = self.stationary_state(density, iterations=iteration)
            if state.residual_inf <= tolerance:
                return state
            if iteration == maximum:
                break
            log_ratio = np.log(density / self._bulk_grid)
            log_ratio -= mixing * self.dimensionless_gradient(density)
            if not np.all(np.isfinite(log_ratio)):
                raise RuntimeError(
                    "HNC Picard iteration produced a non-finite log density."
                )
            density = self._bulk_grid * np.exp(log_ratio)
            if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
                raise RuntimeError("HNC Picard iteration produced an invalid density.")

        raise RuntimeError(
            "HNC Picard solver did not meet the declared stationarity tolerance "
            f"within {maximum} iterations."
        )


__all__ = [
    "Route2V0SiteHNCAsset",
    "Route2V0SiteHNCFunctional",
    "Route2V0SiteHNCState",
    "V0_SITE_HNC_CONSTRUCTION",
]
