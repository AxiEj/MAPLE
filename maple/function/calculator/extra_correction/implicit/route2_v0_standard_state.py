r"""Exact standard-state scalar for the no-training Route-2 V0-AQ-C ledger.

The V0-AQ-C electronic/continuum stationary difference is a microscopic
infinite-dilution excess free energy.  A reported solvation standard free
energy additionally needs an explicitly declared thermodynamic reference.
For the common ideal-gas pressure standard ``p°`` and ideal-solution
concentration standard ``c°``, the required conversion is

.. math::

   \Delta G^\circ_{p\to c}(T)
   = RT\log\!\left(\frac{c^\circ RT}{p^\circ}\right).

This module implements that exact, solute-independent scalar in SI units and
converts it to the energy units used by the rest of MAPLE.  It is not an
empirical nonpolar correction, solvent-radius adjustment, or fitted offset.
It has no coordinate derivative and may be added to a V0 total only when the
underlying stationary auxiliary calculation declares the matching excess-free-
energy reference.  In particular, it cannot turn an electrostatic-only block
into a total solvation model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ase.units import Hartree

V0_STANDARD_STATE_CONSTRUCTION = "route2-v0-standard-state-v1"
V0_STANDARD_STATE_SCOPE = "ideal-gas-pressure-to-ideal-solution-concentration-v1"

# Exact SI definitions / CODATA 2018 values.  Keeping these local makes the
# conversion auditable rather than depending on an implicit unit convention.
_MOLAR_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
_AVOGADRO_PER_MOL = 6.02214076e23
_ELEMENTARY_CHARGE_J_PER_EV = 1.602176634e-19
_JOULE_PER_KILOCALORIE = 4184.0
_MOL_PER_M3_PER_MOL_PER_LITER = 1000.0


def _finite_positive(value: object, *, name: str) -> float:
    """Return one finite positive real scalar without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class Route2V0StandardStateCorrection:
    """Immutable ideal-standard-state conversion at one temperature.

    ``correction_*`` is the additive conversion from an ideal-gas pressure
    standard to an ideal-solution concentration standard.  For the familiar
    ``1 atm -> 1 mol/L`` convention at 298.15 K it is positive, about
    1.89 kcal/mol.  The sign is fixed by the declared direction; callers must
    not negate it to improve agreement with a benchmark.
    """

    temperature_kelvin: float
    gas_standard_pressure_pascal: float
    solution_standard_concentration_mol_per_liter: float
    dimensionless_standard_ratio: float
    correction_joule_per_mol: float
    correction_kcal_per_mol: float
    correction_hartree_per_molecule: float
    temperature_derivative_joule_per_mol_kelvin: float
    construction: str = V0_STANDARD_STATE_CONSTRUCTION
    scope: str = V0_STANDARD_STATE_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "temperature_kelvin",
            "gas_standard_pressure_pascal",
            "solution_standard_concentration_mol_per_liter",
            "dimensionless_standard_ratio",
        ):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name=name)
            )
        for name in (
            "correction_joule_per_mol",
            "correction_kcal_per_mol",
            "correction_hartree_per_molecule",
            "temperature_derivative_joule_per_mol_kelvin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.construction != V0_STANDARD_STATE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 standard-state construction.")
        if self.scope != V0_STANDARD_STATE_SCOPE:
            raise ValueError("Unsupported Route-2 V0 standard-state scope.")

    @property
    def is_total_solvation_asset(self) -> bool:
        """Return false: this exact convention is only one ledger component."""

        return False

    @property
    def coordinate_gradient_hartree_per_bohr(self) -> float:
        """Return the exact nuclear-coordinate derivative of this scalar."""

        return 0.0


def evaluate_route2_v0_standard_state_correction(
    *,
    temperature_kelvin: float,
    gas_standard_pressure_pascal: float = 101_325.0,
    solution_standard_concentration_mol_per_liter: float = 1.0,
) -> Route2V0StandardStateCorrection:
    """Evaluate ``RT log(c° RT / p°)`` with explicit SI standards.

    The quantity inside the logarithm is dimensionless after converting
    ``c°`` from mol/L to mol/m³.  No solvent property or experimental
    solvation value appears in this calculation.
    """

    temperature = _finite_positive(temperature_kelvin, name="temperature_kelvin")
    pressure = _finite_positive(
        gas_standard_pressure_pascal,
        name="gas_standard_pressure_pascal",
    )
    concentration = _finite_positive(
        solution_standard_concentration_mol_per_liter,
        name="solution_standard_concentration_mol_per_liter",
    )
    ratio = (
        concentration
        * _MOL_PER_M3_PER_MOL_PER_LITER
        * _MOLAR_GAS_CONSTANT_J_PER_MOL_K
        * temperature
        / pressure
    )
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise RuntimeError("Standard-state ratio is invalid.")
    log_ratio = math.log(ratio)
    correction_joule_per_mol = _MOLAR_GAS_CONSTANT_J_PER_MOL_K * temperature * log_ratio
    correction_kcal_per_mol = correction_joule_per_mol / _JOULE_PER_KILOCALORIE
    correction_ev_per_molecule = (
        correction_joule_per_mol / _AVOGADRO_PER_MOL / _ELEMENTARY_CHARGE_J_PER_EV
    )
    correction_hartree_per_molecule = correction_ev_per_molecule / Hartree
    temperature_derivative = _MOLAR_GAS_CONSTANT_J_PER_MOL_K * (log_ratio + 1.0)
    return Route2V0StandardStateCorrection(
        temperature_kelvin=temperature,
        gas_standard_pressure_pascal=pressure,
        solution_standard_concentration_mol_per_liter=concentration,
        dimensionless_standard_ratio=ratio,
        correction_joule_per_mol=correction_joule_per_mol,
        correction_kcal_per_mol=correction_kcal_per_mol,
        correction_hartree_per_molecule=correction_hartree_per_molecule,
        temperature_derivative_joule_per_mol_kelvin=temperature_derivative,
    )


__all__ = [
    "V0_STANDARD_STATE_CONSTRUCTION",
    "V0_STANDARD_STATE_SCOPE",
    "Route2V0StandardStateCorrection",
    "evaluate_route2_v0_standard_state_correction",
]
