"""Parameter-free Ewald separation of MDP and POLAR permanent sources.

The MACE-MDP atomwise ``(q, p)`` partition has the desired public molecular
moments but is not a reliable near-field PCM source.  Conversely, the
zero-field MACE-POLAR source has substantially better cavity-surface
electrostatics but does not reproduce the independent MACE-MDP dipole.

This module defines one target-free direct-sum source using the checkpoint's
published charge-density width ``sigma = 1.5 Angstrom``::

    V = V_point[c_POLAR] + V_gaussian_sigma[c_MDP - c_POLAR].

The point branch therefore owns the singular/short-range topology, while the
Gaussian correction restores the MDP multipoles at long range without adding
a fitted mixing coefficient.  The construction is an evaluation candidate,
not an admitted source, energy, force, or solvation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    cartesian_multipoles,
)


def _source4(values: object, *, name: str) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N,4).")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be finite.")
    contiguous = np.ascontiguousarray(source, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(source.shape)


def _molecular_moments(
    positions_angstrom: object,
    source4: np.ndarray,
) -> tuple[float, np.ndarray]:
    positions = np.asarray(positions_angstrom, dtype=float)
    if positions.shape != (len(source4), 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"positions_angstrom must be finite with shape ({len(source4)},3)."
        )
    charges, dipoles = cartesian_multipoles(source4)
    molecular_dipole = np.sum(charges[:, None] * positions + dipoles, axis=0)
    return float(np.sum(charges)), molecular_dipole


@dataclass(frozen=True, slots=True)
class EwaldGaugeSeparatedPermanentSource:
    """One immutable point-plus-Gaussian permanent-source candidate."""

    point_source4_raw_l1: np.ndarray
    gaussian_correction4_raw_l1: np.ndarray
    sigma_angstrom: float = MACE_POLAR_DENSITY_SIGMA_ANGSTROM

    def __post_init__(self) -> None:
        point = _source4(self.point_source4_raw_l1, name="point_source4_raw_l1")
        correction = _source4(
            self.gaussian_correction4_raw_l1,
            name="gaussian_correction4_raw_l1",
        )
        if correction.shape != point.shape:
            raise ValueError("Point and Gaussian source blocks must have equal shape.")
        sigma = float(self.sigma_angstrom)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_angstrom must be finite and positive.")
        if sigma != MACE_POLAR_DENSITY_SIGMA_ANGSTROM:
            raise ValueError(
                "The zero-training candidate is frozen to the official "
                f"MACE-POLAR width {MACE_POLAR_DENSITY_SIGMA_ANGSTROM} Angstrom."
            )
        object.__setattr__(self, "point_source4_raw_l1", point)
        object.__setattr__(self, "gaussian_correction4_raw_l1", correction)
        object.__setattr__(self, "sigma_angstrom", sigma)

    @property
    def atom_count(self) -> int:
        return int(self.point_source4_raw_l1.shape[0])

    @property
    def far_field_source4_raw_l1(self) -> np.ndarray:
        """Return the exact unscreened far-field multipole coefficients."""

        result = self.point_source4_raw_l1 + self.gaussian_correction4_raw_l1
        result.setflags(write=False)
        return result

    def molecular_closure(
        self,
        positions_angstrom: object,
    ) -> dict[str, object]:
        """Return exact direct-sum charge and far-field dipole diagnostics."""

        point_charge, point_dipole = _molecular_moments(
            positions_angstrom, self.point_source4_raw_l1
        )
        correction_charge, correction_dipole = _molecular_moments(
            positions_angstrom, self.gaussian_correction4_raw_l1
        )
        total_charge, total_dipole = _molecular_moments(
            positions_angstrom, self.far_field_source4_raw_l1
        )
        return {
            "point_total_charge_e": point_charge,
            "gaussian_correction_total_charge_e": correction_charge,
            "far_field_total_charge_e": total_charge,
            "point_molecular_dipole_eangstrom": point_dipole.copy(),
            "gaussian_correction_molecular_dipole_eangstrom": (
                correction_dipole.copy()
            ),
            "far_field_molecular_dipole_eangstrom": total_dipole.copy(),
        }


def build_ewald_gauge_separated_permanent_source(
    *,
    mdp_source4_raw_l1: object,
    polar_zero_source4_raw_l1: object,
    charge_tolerance_e: float = 1.0e-10,
) -> EwaldGaugeSeparatedPermanentSource:
    """Build the frozen POLAR-point plus Gaussian MDP-correction source.

    Both checkpoints must describe the same total charge.  A charge mismatch
    would turn the Gaussian branch into an arbitrary long-range monopole
    correction and is therefore rejected rather than projected or rescaled.
    """

    mdp = _source4(mdp_source4_raw_l1, name="mdp_source4_raw_l1")
    polar = _source4(
        polar_zero_source4_raw_l1,
        name="polar_zero_source4_raw_l1",
    )
    if mdp.shape != polar.shape:
        raise ValueError("MDP and POLAR sources must have equal shape.")
    tolerance = float(charge_tolerance_e)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("charge_tolerance_e must be finite and positive.")
    charge_difference = float(np.sum(mdp[:, 0]) - np.sum(polar[:, 0]))
    if abs(charge_difference) > tolerance:
        raise RuntimeError(
            "MDP and POLAR checkpoint sources do not have the same total charge."
        )
    return EwaldGaugeSeparatedPermanentSource(
        point_source4_raw_l1=polar,
        gaussian_correction4_raw_l1=mdp - polar,
    )


__all__ = [
    "EwaldGaugeSeparatedPermanentSource",
    "build_ewald_gauge_separated_permanent_source",
]
