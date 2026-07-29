"""Energy-conjugate nonlocal dielectric control for no-training Route-2 V0.

This module defines only the electrostatic reaction field of a *fixed*,
neutral charge density on a periodic Cartesian grid.  Given one frozen,
real-even relative dielectric spectrum ``epsilon(k) >= 1``, it uses the one
scalar

``G_pol[rho] = 1/2 int rho(r) V_reac[rho](r) dr``

with ``V_reac(k) = 4*pi*(epsilon(k)**-1 - 1)*rho(k)/k**2`` for ``k != 0``.
The potential returned by :meth:`reaction_potential_hartree_per_e` is the
functional derivative of :meth:`polarization_energy_hartree`; real-even
spectra make the discrete operator self-adjoint and ``epsilon >= 1`` makes its
polarization energy non-positive.

It is deliberately *not* a total solvation model.  A scalar bulk dielectric
constant is insufficient to identify a molecular solvent response at finite
wave vector, and this control supplies neither a cavity, a short-range
solute--solvent interaction, dispersion, a liquid free-energy functional, nor
a charged-solute convention.  The zero reciprocal mode is omitted only after
requiring a neutral source; silently adding a periodic background would change
the free-energy convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_NONLOCAL_DIELECTRIC_CONSTRUCTION = "route2-v0-nonlocal-dielectric-v1"
V0_NONLOCAL_DIELECTRIC_SCOPE = "electrostatic-reaction-control-only-v1"
V0_NONLOCAL_DIELECTRIC_RECIPROCITY_RELATIVE_TOLERANCE = 1.0e-12


def _immutable_real_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return one finite immutable real array with an optional exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array.") from exc
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = (
            "a finite array" if shape is None else f"a finite array with shape {shape}"
        )
        raise ValueError(f"{name} must be {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _reciprocal_partner(values: np.ndarray) -> np.ndarray:
    """Return the periodic ``k -> -k`` partner of one three-dimensional array."""

    axes = tuple((-np.arange(length)) % length for length in values.shape)
    return values[np.ix_(*axes)]


@dataclass(frozen=True)
class Route2V0NonlocalDielectricOperator:
    """One frozen, passive, periodic nonlocal dielectric reaction operator.

    ``dielectric_spectrum`` is the relative dielectric ``epsilon(k)`` sampled
    on the FFT grid defined by ``grid``.  It must be finite, real, reciprocal
    even, and no smaller than one.  No empirical parameter is inferred here:
    callers must supply the entire spectrum from a separately frozen solvent
    asset.  A constant spectrum is mathematically permitted as a continuum
    diagnostic, but this class intentionally offers only the electrostatic
    component and cannot be promoted to a molecular total free energy.
    """

    grid: RegularCartesianGrid
    dielectric_spectrum: np.ndarray
    construction: str = V0_NONLOCAL_DIELECTRIC_CONSTRUCTION
    response_scope: str = V0_NONLOCAL_DIELECTRIC_SCOPE
    reciprocity_relative_tolerance: float = (
        V0_NONLOCAL_DIELECTRIC_RECIPROCITY_RELATIVE_TOLERANCE
    )
    _dielectric_spectrum: np.ndarray = field(init=False, repr=False, compare=False)
    _reaction_green_bohr2: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.construction != V0_NONLOCAL_DIELECTRIC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 nonlocal dielectric construction.")
        if self.response_scope != V0_NONLOCAL_DIELECTRIC_SCOPE:
            raise ValueError("Unsupported Route-2 nonlocal dielectric response scope.")
        tolerance = float(self.reciprocity_relative_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Nonlocal dielectric reciprocity tolerance must be finite in (0, 1)."
            )
        dielectric = _immutable_real_array(
            self.dielectric_spectrum,
            name="Nonlocal dielectric spectrum",
            shape=self.grid.shape,
        )
        if np.any(dielectric < 1.0):
            raise ValueError(
                "Nonlocal dielectric spectrum must satisfy epsilon(k) >= 1."
            )
        reciprocal_error = float(
            np.max(np.abs(dielectric - _reciprocal_partner(dielectric)))
        )
        reciprocal_scale = max(1.0, float(np.max(np.abs(dielectric))))
        if reciprocal_error > tolerance * reciprocal_scale:
            raise ValueError(
                "Nonlocal dielectric spectrum must be reciprocal-even under k -> -k."
            )

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
        green[nonzero] = (
            4.0
            * math.pi
            * (1.0 / dielectric[nonzero] - 1.0)
            / wavevector_squared[nonzero]
        )
        green.setflags(write=False)
        object.__setattr__(self, "dielectric_spectrum", dielectric)
        object.__setattr__(self, "reciprocity_relative_tolerance", tolerance)
        object.__setattr__(self, "_dielectric_spectrum", dielectric)
        object.__setattr__(self, "_reaction_green_bohr2", green)

    @property
    def cell_lengths_bohr(self) -> np.ndarray:
        """Return the periodic cell lengths fixed by the grid geometry."""

        result = np.asarray(self.grid.shape, dtype=float) * self.grid.spacing_bohr
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result

    @property
    def fourier_dielectric_spectrum(self) -> np.ndarray:
        """Return the immutable, caller-supplied relative dielectric spectrum."""

        return self._dielectric_spectrum

    @property
    def fourier_reaction_green_bohr2(self) -> np.ndarray:
        """Return ``4*pi*(epsilon^-1 - 1)/k^2`` with its zero mode omitted."""

        return self._reaction_green_bohr2

    def _validate_charge_density(self, values: np.ndarray) -> np.ndarray:
        return _immutable_real_array(
            values,
            name="Nonlocal dielectric charge density",
            shape=self.grid.shape,
        )

    def integrated_charge_e(self, charge_density_e_per_bohr3: np.ndarray) -> float:
        """Return the grid integral of one charge density in elementary charge."""

        density = self._validate_charge_density(charge_density_e_per_bohr3)
        return float(self.grid.volume_element_bohr3 * np.sum(density))

    def _require_neutral(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        relative_tolerance: float,
    ) -> np.ndarray:
        tolerance = float(relative_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(
                "Nonlocal dielectric neutrality tolerance must be finite in (0, 1)."
            )
        density = self._validate_charge_density(charge_density_e_per_bohr3)
        volume = self.grid.volume_element_bohr3
        net_charge = float(volume * np.sum(density))
        scale = max(1.0, float(volume * np.sum(np.abs(density))))
        if abs(net_charge) > tolerance * scale:
            raise ValueError(
                "Nonlocal dielectric charge density must be neutral; a periodic "
                "background is not an admitted Route-2 V0 convention."
            )
        return density

    def reaction_potential_hartree_per_e(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> np.ndarray:
        """Return the zero-average reaction potential from the declared scalar."""

        density = self._require_neutral(
            charge_density_e_per_bohr3,
            relative_tolerance=neutrality_relative_tolerance,
        )
        potential_complex = np.fft.ifftn(
            self._reaction_green_bohr2 * np.fft.fftn(density)
        )
        imaginary_error = float(np.max(np.abs(np.imag(potential_complex))))
        scale = max(1.0, float(np.max(np.abs(potential_complex))))
        if imaginary_error > 1.0e-12 * scale:
            raise RuntimeError(
                "Nonlocal dielectric reaction solve produced a complex field."
            )
        potential = np.array(np.real(potential_complex), dtype=float, copy=True)
        # k=0 is omitted as a gauge.  The subtraction removes only roundoff.
        potential -= float(np.mean(potential))
        potential.setflags(write=False)
        return potential

    def polarization_energy_hartree(
        self,
        charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return ``1/2 int rho V_reac`` in Hartree from the same operator."""

        density = self._require_neutral(
            charge_density_e_per_bohr3,
            relative_tolerance=neutrality_relative_tolerance,
        )
        potential = self.reaction_potential_hartree_per_e(
            density,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        energy = (
            0.5 * self.grid.volume_element_bohr3 * float(np.sum(density * potential))
        )
        if not math.isfinite(energy):
            raise RuntimeError("Nonlocal dielectric polarization scalar is non-finite.")
        return energy

    def reaction_pairing_hartree(
        self,
        left_charge_density_e_per_bohr3: np.ndarray,
        right_charge_density_e_per_bohr3: np.ndarray,
        *,
        neutrality_relative_tolerance: float = 1.0e-12,
    ) -> float:
        """Return ``int left * V_reac[right]`` for one reciprocal-pairing check."""

        left = self._require_neutral(
            left_charge_density_e_per_bohr3,
            relative_tolerance=neutrality_relative_tolerance,
        )
        right_potential = self.reaction_potential_hartree_per_e(
            right_charge_density_e_per_bohr3,
            neutrality_relative_tolerance=neutrality_relative_tolerance,
        )
        return float(self.grid.volume_element_bohr3 * np.sum(left * right_potential))


__all__ = [
    "Route2V0NonlocalDielectricOperator",
    "V0_NONLOCAL_DIELECTRIC_CONSTRUCTION",
    "V0_NONLOCAL_DIELECTRIC_RECIPROCITY_RELATIVE_TOLERANCE",
    "V0_NONLOCAL_DIELECTRIC_SCOPE",
]
