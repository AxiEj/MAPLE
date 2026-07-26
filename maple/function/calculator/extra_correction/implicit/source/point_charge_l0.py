"""Fixed atom-centred point-charge source for continuum baselines.

This source is intentionally separate from the MACE-POLAR mutual-polarization
engine.  It embeds scalar charges into the existing continuum ``l<=1`` block
only at the provider boundary, with all dipole channels fixed to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from ase.units import Hartree

from ..electrostatic_pairing import MACE_POLAR_L1_PAIRING
from ..gto_density import point_multipole_potential


POINT_CHARGE_TOTAL_TOLERANCE_E = 1.0e-10


@dataclass(frozen=True)
class PointChargeL0Source:
    """One immutable fixed point-charge source."""

    charges_e: np.ndarray
    declared_total_charge_e: float
    source_model: str

    def __post_init__(self) -> None:
        charges = np.array(self.charges_e, dtype=float, copy=True)
        if (
            charges.ndim != 1
            or charges.size == 0
            or not np.all(np.isfinite(charges))
        ):
            raise ValueError(
                "Point-charge source requires a finite non-empty charge vector."
            )
        declared = float(self.declared_total_charge_e)
        if not math.isfinite(declared):
            raise ValueError("Declared point-charge total must be finite.")
        if abs(float(np.sum(charges)) - declared) > (
            POINT_CHARGE_TOTAL_TOLERANCE_E
        ):
            raise ValueError(
                "Point-charge source does not conserve its declared molecular "
                f"charge (sum={float(np.sum(charges)):.12g} e, "
                f"declared={declared:.12g} e)."
            )
        if not self.source_model:
            raise ValueError("Point-charge source requires a source-model name.")
        charges.setflags(write=False)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "declared_total_charge_e", declared)

    @property
    def continuum_multipole_coefficients(self) -> np.ndarray:
        """Return an ``(N,4)`` provider block with zero dipole channels."""

        coefficients = np.zeros((self.charges_e.size, 4), dtype=float)
        coefficients[:, 0] = self.charges_e
        coefficients.setflags(write=False)
        return coefficients

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "solute_source": "point-charge-l0",
            "source_model": self.source_model,
            "polarization_response": "fixed",
            "declared_total_charge_e": self.declared_total_charge_e,
        }

    def surface_potential_hartree_per_e(
        self,
        points_bohr: np.ndarray,
        atom_positions_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the point-charge MEP without inventing higher multipoles."""

        return point_multipole_potential(
            points_bohr,
            atom_positions_angstrom,
            self.continuum_multipole_coefficients,
        )


@dataclass(frozen=True)
class FixedChargeContinuumState:
    """Continuum electrostatic state for one fixed point-charge source."""

    continuum_multipole_coefficients: np.ndarray
    reaction_field_values_ev: np.ndarray
    polarization_energy_hartree: float
    energy_identity_error_ev: float
    source_model: str
    solute_source: str = "point-charge-l0"
    polarization_response: str = "fixed"

    def __post_init__(self) -> None:
        coefficients = np.array(
            self.continuum_multipole_coefficients,
            dtype=float,
            copy=True,
        )
        field = np.array(
            self.reaction_field_values_ev,
            dtype=float,
            copy=True,
        )
        if (
            coefficients.ndim != 2
            or coefficients.shape[1] != 4
            or field.shape != coefficients.shape
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(field))
        ):
            raise ValueError(
                "Fixed-charge continuum arrays must be finite matching "
                "(n_atoms, 4) blocks."
            )
        if np.any(coefficients[:, 1:] != 0.0):
            raise ValueError(
                "A point-charge-l0 state cannot contain dipole coefficients."
            )
        if not math.isfinite(self.polarization_energy_hartree):
            raise ValueError("Continuum polarization energy must be finite.")
        if (
            not math.isfinite(self.energy_identity_error_ev)
            or self.energy_identity_error_ev < 0.0
        ):
            raise ValueError(
                "Continuum energy-identity error must be finite and nonnegative."
            )
        if not self.source_model:
            raise ValueError("Fixed-charge continuum state needs a source model.")
        if self.solute_source != "point-charge-l0":
            raise ValueError("Invalid fixed-charge solute-source identity.")
        if self.polarization_response != "fixed":
            raise ValueError("Invalid fixed-charge polarization identity.")
        coefficients.setflags(write=False)
        field.setflags(write=False)
        object.__setattr__(
            self,
            "continuum_multipole_coefficients",
            coefficients,
        )
        object.__setattr__(self, "reaction_field_values_ev", field)


def solve_fixed_charge_continuum(
    reaction_field: Any,
    source: PointChargeL0Source,
    *,
    energy_identity_tolerance_ev: float = 1.0e-8,
) -> FixedChargeContinuumState:
    """Solve one fixed-source continuum state and check its scalar ledger."""

    if energy_identity_tolerance_ev <= 0.0:
        raise ValueError("Energy-identity tolerance must be positive.")
    coefficients = source.continuum_multipole_coefficients
    field = np.asarray(
        reaction_field.apply_scf(coefficients),
        dtype=float,
    )
    if (
        field.shape != coefficients.shape
        or not np.all(np.isfinite(field))
    ):
        raise RuntimeError(
            "Continuum reaction field must be finite with shape "
            f"{coefficients.shape}; received {field.shape}."
        )
    polarization_energy_hartree = float(
        reaction_field.scf_polarization_energy_hartree(coefficients)
    )
    if not math.isfinite(polarization_energy_hartree):
        raise RuntimeError("Continuum polarization energy is non-finite.")

    half_coupling_ev = 0.5 * MACE_POLAR_L1_PAIRING.pair(
        coefficients,
        field,
    )
    provider_energy_ev = polarization_energy_hartree * Hartree
    identity_error_ev = abs(half_coupling_ev - provider_energy_ev)
    if identity_error_ev > energy_identity_tolerance_ev:
        raise RuntimeError(
            "Fixed point-charge continuum failed the half-coupling identity "
            f"(absolute error={identity_error_ev:.3e} eV)."
        )

    return FixedChargeContinuumState(
        continuum_multipole_coefficients=coefficients,
        reaction_field_values_ev=field,
        polarization_energy_hartree=polarization_energy_hartree,
        energy_identity_error_ev=identity_error_ev,
        source_model=source.source_model,
    )
