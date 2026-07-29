"""Parameter-free frozen-density Pauli-overlap control for Route-2 V0.

For two nonnegative *electron* reference densities on one common Cartesian
grid, this module defines the Thomas--Fermi nonadditive kinetic scalar

``T_TF^nad[n_a, n_b] = C_TF int ((n_a + n_b)^(5/3) - n_a^(5/3) - n_b^(5/3))``.

Its two returned potentials are the exact discrete functional derivatives of
that same scalar.  Convexity makes the scalar nonnegative, so it is a
well-defined parameter-free frozen-density Pauli-overlap control.  It does
not turn a promolecular reference into the MACE density, supply a solvent
electron-density asset, model dispersion, or define a physical liquid free
energy.  In particular, no fitted overlap coefficient or target-solvation
label enters this primitive.

The Thomas--Fermi integrand is continuously differentiable at zero density,
but is not twice differentiable there.  Consequently this V0 control is not
admitted as a Newton/Hessian or production-PES component until a smooth,
source-provenanced short-range functional is derived together with the liquid
state it couples to.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_FROZEN_DENSITY_PAULI_CONSTRUCTION = "route2-v0-frozen-density-tf-nad-v1"
V0_FROZEN_DENSITY_PAULI_SCOPE = "frozen-density-pauli-overlap-control-only-v1"
THOMAS_FERMI_KINETIC_COEFFICIENT = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0)


def _immutable_nonnegative_density(
    values: np.ndarray,
    *,
    grid: RegularCartesianGrid,
    name: str,
) -> np.ndarray:
    """Return one finite, nonnegative, immutable electron-density grid."""

    try:
        density = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real density array.") from exc
    if (
        density.shape != grid.shape
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise ValueError(
            f"{name} must be finite, nonnegative, and have shape {grid.shape}."
        )
    result = np.array(density, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0FrozenDensityPauliOverlap:
    """One frozen-density Thomas--Fermi nonadditive kinetic scalar.

    Both density inputs are independent electron reference densities in
    electrons per Bohr cubed.  They deliberately remain separate from the
    frozen MACE Gaussian electrostatic source.  The two density channels are
    exchange symmetric; their labels only identify which functional derivative
    a caller requests.
    """

    grid: RegularCartesianGrid
    solute_electron_density_e_per_bohr3: np.ndarray
    solvent_electron_density_e_per_bohr3: np.ndarray
    construction: str = V0_FROZEN_DENSITY_PAULI_CONSTRUCTION
    interaction_scope: str = V0_FROZEN_DENSITY_PAULI_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise ValueError("Frozen-density Pauli control requires a regular grid.")
        if self.construction != V0_FROZEN_DENSITY_PAULI_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 V0 frozen-density Pauli construction."
            )
        if self.interaction_scope != V0_FROZEN_DENSITY_PAULI_SCOPE:
            raise ValueError("Unsupported Route-2 V0 frozen-density Pauli scope.")
        solute = _immutable_nonnegative_density(
            self.solute_electron_density_e_per_bohr3,
            grid=self.grid,
            name="Solute electron density",
        )
        solvent = _immutable_nonnegative_density(
            self.solvent_electron_density_e_per_bohr3,
            grid=self.grid,
            name="Solvent electron density",
        )
        object.__setattr__(self, "solute_electron_density_e_per_bohr3", solute)
        object.__setattr__(self, "solvent_electron_density_e_per_bohr3", solvent)

    def _total_density_e_per_bohr3(self) -> np.ndarray:
        """Return the finite nonnegative total reference density."""

        return (
            self.solute_electron_density_e_per_bohr3
            + self.solvent_electron_density_e_per_bohr3
        )

    def nonadditive_kinetic_energy_hartree(self) -> float:
        """Return the parameter-free Thomas--Fermi nonadditive kinetic scalar."""

        solute = self.solute_electron_density_e_per_bohr3
        solvent = self.solvent_electron_density_e_per_bohr3
        total = self._total_density_e_per_bohr3()
        integrand = (
            total ** (5.0 / 3.0) - solute ** (5.0 / 3.0) - solvent ** (5.0 / 3.0)
        )
        energy = float(
            THOMAS_FERMI_KINETIC_COEFFICIENT
            * self.grid.volume_element_bohr3
            * np.sum(integrand)
        )
        if not math.isfinite(energy):
            raise RuntimeError("Frozen-density Pauli scalar is non-finite.")
        return energy

    def solute_potential_hartree_per_e(self) -> np.ndarray:
        """Return ``delta T_TF^nad / delta n_solute`` on the common grid."""

        total = self._total_density_e_per_bohr3()
        potential = (
            (5.0 / 3.0)
            * THOMAS_FERMI_KINETIC_COEFFICIENT
            * (
                total ** (2.0 / 3.0)
                - self.solute_electron_density_e_per_bohr3 ** (2.0 / 3.0)
            )
        )
        result = np.array(potential, dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def solvent_potential_hartree_per_e(self) -> np.ndarray:
        """Return ``delta T_TF^nad / delta n_solvent`` on the common grid."""

        total = self._total_density_e_per_bohr3()
        potential = (
            (5.0 / 3.0)
            * THOMAS_FERMI_KINETIC_COEFFICIENT
            * (
                total ** (2.0 / 3.0)
                - self.solvent_electron_density_e_per_bohr3 ** (2.0 / 3.0)
            )
        )
        result = np.array(potential, dtype=float, copy=True)
        result.setflags(write=False)
        return result


__all__ = [
    "Route2V0FrozenDensityPauliOverlap",
    "THOMAS_FERMI_KINETIC_COEFFICIENT",
    "V0_FROZEN_DENSITY_PAULI_CONSTRUCTION",
    "V0_FROZEN_DENSITY_PAULI_SCOPE",
]
