"""Fixed-density, energy-conjugate continuum kernel for Route-2 V0-FD.

This module deliberately has no MACE field-response input.  Given a frozen
gas-phase MACE-POLAR density and a surface potential generated from that same
density, it evaluates exactly one fixed-geometry continuum scalar,

``U_pol(v) = 0.5 * v.T @ q_energy(v)``.

The continuum adapter owns the response ``q_energy``.  The state is therefore
the stationary solvent response to a *fixed* solute source, not a new
MACE--PCM fixed point and not a post-hoc correction of the rejected scalar
response V0 experiment.  It is internal research infrastructure only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Bohr

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
    SurfaceChargeState,
)
from .gto_density import point_multipole_potential


V0_FROZEN_DENSITY_CONSTRUCTION = "route2-v0-fixed-density-v1"


def _immutable_density(values: np.ndarray) -> np.ndarray:
    """Validate and freeze Route-2 ``l<=1`` density coefficients."""

    density = np.asarray(values, dtype=float)
    if (
        density.ndim != 2
        or density.shape[0] == 0
        or density.shape[1] != 4
        or not np.all(np.isfinite(density))
    ):
        raise ValueError(
            "Frozen density coefficients must be finite with shape (n_atoms, 4)."
        )
    result = np.array(density, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_positions(values: np.ndarray, *, atom_count: int) -> np.ndarray:
    """Validate and freeze the atomic centres of the frozen MACE source."""

    positions = np.asarray(values, dtype=float)
    if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            "Frozen-density atom positions must be finite with shape "
            f"({atom_count}, 3); received {positions.shape}."
        )
    result = np.array(positions, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0FrozenDensityState:
    """One auditable fixed-density continuum evaluation at fixed geometry.

    ``surface_charge_state`` is retained rather than flattened so that direct,
    adjoint, and energy-conjugate surface charges remain inspectable.  No
    charge projection is permitted: the supplied density must already satisfy
    its declared total-charge constraint.
    """

    density_coefficients: np.ndarray
    atom_positions_angstrom: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_state: SurfaceChargeState
    target_total_charge_e: float
    total_charge_e: float
    electrostatic_energy_hartree: float
    construction: str = V0_FROZEN_DENSITY_CONSTRUCTION

    def __post_init__(self) -> None:
        density = _immutable_density(self.density_coefficients)
        positions = _immutable_positions(
            self.atom_positions_angstrom,
            atom_count=density.shape[0],
        )
        potential = np.asarray(self.surface_potential_hartree_per_e, dtype=float)
        if (
            potential.shape
            != self.surface_charge_state.surface_potential_hartree_per_e.shape
            or not np.all(np.isfinite(potential))
        ):
            raise ValueError("Frozen-density surface potential is invalid.")
        potential = np.array(potential, dtype=float, copy=True)
        potential.setflags(write=False)
        if not np.array_equal(
            potential,
            self.surface_charge_state.surface_potential_hartree_per_e,
        ):
            raise ValueError(
                "The frozen-density surface potential must equal the continuum "
                "surface-charge state's potential."
            )
        target_charge = float(self.target_total_charge_e)
        total_charge = float(self.total_charge_e)
        energy = float(self.electrostatic_energy_hartree)
        if not all(
            math.isfinite(value) for value in (target_charge, total_charge, energy)
        ):
            raise ValueError("Frozen-density scalar values must be finite.")
        if self.construction != V0_FROZEN_DENSITY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 frozen-density construction.")
        density_charge = float(np.sum(density[:, 0]))
        charge_tolerance = max(1.0e-12, 1.0e-10 * abs(density_charge))
        if abs(total_charge - density_charge) > charge_tolerance:
            raise ValueError(
                "Frozen-density total charge must equal the supplied density's "
                "monopole sum."
            )
        expected_energy = float(self.surface_charge_state.polarization_energy_hartree)
        tolerance = max(1.0e-12, 1.0e-10 * abs(expected_energy))
        if abs(energy - expected_energy) > tolerance:
            raise ValueError(
                "Frozen-density electrostatic energy must equal the matching "
                "energy-conjugate continuum polarization energy."
            )
        object.__setattr__(self, "density_coefficients", density)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "surface_potential_hartree_per_e", potential)
        object.__setattr__(self, "target_total_charge_e", target_charge)
        object.__setattr__(self, "total_charge_e", total_charge)
        object.__setattr__(self, "electrostatic_energy_hartree", energy)


def evaluate_route2_v0_frozen_density(
    *,
    density_coefficients: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    continuum: ExternalMEPCavityResponse,
    target_total_charge_e: float = 0.0,
    total_charge_tolerance_e: float = 1.0e-10,
) -> Route2V0FrozenDensityState:
    """Evaluate the V0-FD electrostatic scalar without learned field response.

    Mathematically, an energy-conjugate continuum response is the gradient of
    a quadratic surface-potential functional.  This function derives the
    point-multipole source potential from the supplied frozen MACE density at
    the declared atom positions, then evaluates ``0.5 * v.T @ q_energy(v)``.
    Supplying a precomputed surface potential is deliberately unsupported: it
    would let the source and energy pair drift apart.  The only solute-state
    invariant enforced here is total charge; unlike the legacy SCF path, no
    density projection or mixing is performed.
    """

    density = _immutable_density(density_coefficients)
    if density.shape[0] != int(continuum.atom_count):
        raise ValueError(
            "Frozen density atom count does not match the continuum cavity."
        )
    if (
        getattr(continuum, "contract_version", None)
        != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported external-MEP continuum-response contract.")
    if not bool(continuum.energy_response_is_reciprocal):
        raise ValueError(
            "Route-2 V0-FD requires an energy-conjugate reciprocal continuum "
            "response."
        )
    tolerance = float(total_charge_tolerance_e)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Total-charge tolerance must be finite and positive.")
    target_charge = float(target_total_charge_e)
    if not math.isfinite(target_charge):
        raise ValueError("Target total charge must be finite.")
    total_charge = float(np.sum(density[:, 0]))
    if abs(total_charge - target_charge) > tolerance:
        raise ValueError(
            "Frozen density violates its total-charge constraint "
            f"(observed={total_charge:.16e} e, target={target_charge:.16e} e)."
        )
    positions = _immutable_positions(
        atom_positions_angstrom,
        atom_count=density.shape[0],
    )
    reference_positions = np.asarray(
        continuum.reference_positions_bohr,
        dtype=float,
    )
    if reference_positions.shape != positions.shape or not np.allclose(
        positions,
        reference_positions * Bohr,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Frozen-density source geometry does not match the continuum cavity."
        )
    potential = point_multipole_potential(
        np.asarray(continuum.surface_points_bohr, dtype=float),
        positions,
        density,
    )
    state = continuum.solve(potential)
    return Route2V0FrozenDensityState(
        density_coefficients=density,
        atom_positions_angstrom=positions,
        surface_potential_hartree_per_e=potential,
        surface_charge_state=state,
        target_total_charge_e=target_charge,
        total_charge_e=total_charge,
        electrostatic_energy_hartree=float(state.polarization_energy_hartree),
    )


__all__ = [
    "Route2V0FrozenDensityState",
    "V0_FROZEN_DENSITY_CONSTRUCTION",
    "evaluate_route2_v0_frozen_density",
]
