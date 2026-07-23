"""Water SMD Coulomb radii and native CDS energy.

The parameterization follows Marenich, Cramer, and Truhlar,
J. Phys. Chem. B 2009, 113, 6378-6396 (DOI: 10.1021/jp810292n).
The aqueous atomic-tension geometry functions are independently implemented
from the published/NWChem reference equations.  SASA is evaluated with a
deterministic high-order spherical quadrature and no quantum-chemistry runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


HARTREE_TO_KCAL_MOL = 627.5094740631
SASA_PROBE_RADIUS_ANGSTROM = 0.4
SASA_GRID_POINTS = 5810


SMD_WATER_COULOMB_RADII_ANGSTROM = {
    "H": 1.20,
    "C": 1.85,
    "N": 1.89,
    "O": 1.52,
    "F": 1.73,
    "P": 2.47,
    "S": 2.12,
    "Cl": 2.49,
    # SMD18 revision recommended by the Minnesota solvation database.
    "Br": 2.60,
    "I": 2.74,
}


_SASA_VDW_RADII_ANGSTROM = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}


_WATER_BASE_TENSION = {
    "H": 48.69,
    "C": 129.74,
    "F": 38.18,
    "S": -9.10,
    "Cl": 9.82,
    "Br": -8.72,
}


_SWITCH_PARAMETERS = {
    ("H", "C"): (1.55, 0.30),
    ("H", "O"): (1.55, 0.30),
    ("C", "H"): (1.55, 0.30),
    ("C", "C"): (1.84, 0.30),
    ("C", "N"): (1.84, 0.30),
    ("C", "O"): (1.84, 0.30),
    ("C", "F"): (1.84, 0.30),
    ("C", "P"): (2.20, 0.30),
    ("C", "S"): (2.20, 0.30),
    ("C", "Cl"): (2.10, 0.30),
    ("C", "Br"): (2.30, 0.30),
    ("C", "I"): (2.60, 0.30),
    ("N", "C"): (1.84, 0.30),
    ("O", "C"): (1.33, 0.10),
    ("O", "N"): (1.50, 0.30),
    ("O", "O"): (1.80, 0.30),
    ("O", "P"): (2.10, 0.30),
}


@dataclass(frozen=True)
class SMDCDSResult:
    energy_hartree: float
    energy_kcal_mol: float
    atom_areas_angstrom2: np.ndarray
    atom_tensions_cal_mol_angstrom2: np.ndarray
    total_area_angstrom2: float
    grid_points_per_atom: int


def _validate_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(symbol) for symbol in symbols)
    unsupported = sorted(set(normalized).difference(SMD_WATER_COULOMB_RADII_ANGSTROM))
    if unsupported:
        raise ValueError(
            "Route-2 SMD water currently supports H/C/N/O/F/P/S/Cl/Br/I only; "
            f"unsupported elements: {', '.join(unsupported)}."
        )
    return normalized


def smd_water_coulomb_radii(symbols) -> np.ndarray:
    normalized = _validate_symbols(tuple(symbols))
    return np.asarray(
        [SMD_WATER_COULOMB_RADII_ANGSTROM[symbol] for symbol in normalized],
        dtype=float,
    )


def smd_sasa_radii(symbols) -> np.ndarray:
    normalized = _validate_symbols(tuple(symbols))
    return np.asarray(
        [
            _SASA_VDW_RADII_ANGSTROM[symbol] + SASA_PROBE_RADIUS_ANGSTROM
            for symbol in normalized
        ],
        dtype=float,
    )


def _switch(distance: float, reference: float, width: float) -> float:
    if distance >= reference + width:
        return 0.0
    return math.exp(width / (distance - width - reference))


def aqueous_atomic_surface_tensions(
    symbols,
    positions_angstrom: np.ndarray,
) -> np.ndarray:
    """Return aqueous SMD atomic surface tensions in cal/(mol Å²)."""

    symbols = _validate_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    if positions.shape != (len(symbols), 3):
        raise ValueError("SMD coordinates must have shape (n_atoms, 3).")
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )

    tensions = np.zeros(len(symbols), dtype=float)
    for i, symbol in enumerate(symbols):
        tension = _WATER_BASE_TENSION.get(symbol, 0.0)
        if symbol in {"F", "S", "Cl", "Br", "I", "P"}:
            tensions[i] = tension
            continue

        if symbol == "H":
            t_hc = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("H", "C")])
                for j, other in enumerate(symbols)
                if other == "C"
            )
            # The aqueous H-O coefficient is zero, but retaining the published
            # switching branch documents the complete water functional.
            _ = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("H", "O")])
                for j, other in enumerate(symbols)
                if other == "O"
            )
            tensions[i] = tension - 60.77 * t_hc
            continue

        if symbol == "C":
            t_cc = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("C", "C")])
                for j, other in enumerate(symbols)
                if j != i and other == "C"
            )
            tensions[i] = tension - 72.95 * t_cc
            continue

        if symbol == "N":
            t_nc = 0.0
            t_nc_carbonyl = 0.0
            for j, other in enumerate(symbols):
                if other != "C":
                    continue
                carbon_environment = 0.0
                carbon_oxygen_environment = 0.0
                for k, neighbor in enumerate(symbols):
                    if k in {i, j}:
                        continue
                    parameters = _SWITCH_PARAMETERS.get(("C", neighbor), (0.0, 0.0))
                    if parameters[1] > 0.0:
                        coordination = _switch(distances[j, k], *parameters)
                        carbon_environment += coordination
                        if neighbor == "O":
                            carbon_oxygen_environment += coordination
                nc_coordination = _switch(
                    distances[i, j], *_SWITCH_PARAMETERS[("N", "C")]
                )
                t_nc += nc_coordination * carbon_environment**2
                t_nc_carbonyl += nc_coordination * carbon_oxygen_environment
            tensions[i] = (
                tension
                - 48.22 * t_nc**1.3
                + 84.10 * t_nc_carbonyl
            )
            continue

        if symbol == "O":
            t_oc = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("O", "C")])
                for j, other in enumerate(symbols)
                if other == "C"
            )
            t_on = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("O", "N")])
                for j, other in enumerate(symbols)
                if other == "N"
            )
            t_op = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("O", "P")])
                for j, other in enumerate(symbols)
                if other == "P"
            )
            tensions[i] = tension + 68.69 * t_oc + 121.98 * t_on + 68.85 * t_op
            continue

        tensions[i] = tension
    return tensions


def _fibonacci_sphere(number: int) -> np.ndarray:
    if number < 32:
        raise ValueError("SASA spherical quadrature requires at least 32 points.")
    indices = np.arange(number, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / number
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = indices * golden_angle
    return np.column_stack((radius_xy * np.cos(phi), radius_xy * np.sin(phi), z))


def solvent_accessible_surface_areas(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> np.ndarray:
    """Deterministic Shrake-Rupley SASA with equal-area spherical nodes."""

    positions = np.asarray(positions_angstrom, dtype=float)
    radii = np.asarray(radii_angstrom, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("SASA coordinates must have shape (n_atoms, 3).")
    if radii.shape != (positions.shape[0],) or np.any(radii <= 0):
        raise ValueError("SASA radii must be one positive value per atom.")

    directions = _fibonacci_sphere(int(grid_points))
    areas = np.empty(positions.shape[0], dtype=float)
    for atom_index, (center, radius) in enumerate(
        zip(positions, radii, strict=True)
    ):
        surface_points = center + radius * directions
        exposed = np.ones(grid_points, dtype=bool)
        for other_index, (other_center, other_radius) in enumerate(
            zip(positions, radii, strict=True)
        ):
            if atom_index == other_index:
                continue
            # Skip non-overlapping expanded spheres before allocating the
            # point-to-neighbour distance array.
            if np.linalg.norm(center - other_center) >= radius + other_radius:
                continue
            exposed &= (
                np.einsum(
                    "ij,ij->i",
                    surface_points - other_center,
                    surface_points - other_center,
                )
                >= other_radius**2
            )
            if not np.any(exposed):
                break
        areas[atom_index] = (
            4.0 * math.pi * radius * radius * np.count_nonzero(exposed) / grid_points
        )
    return areas


def smd_water_cds(
    symbols,
    positions_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> SMDCDSResult:
    """Compute the aqueous SMD cavity/dispersion/solvent-structure term."""

    symbols = _validate_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    tensions = aqueous_atomic_surface_tensions(symbols, positions)
    areas = solvent_accessible_surface_areas(
        positions,
        smd_sasa_radii(symbols),
        grid_points=grid_points,
    )
    energy_kcal_mol = float(np.dot(tensions, areas) / 1000.0)
    return SMDCDSResult(
        energy_hartree=energy_kcal_mol / HARTREE_TO_KCAL_MOL,
        energy_kcal_mol=energy_kcal_mol,
        atom_areas_angstrom2=areas,
        atom_tensions_cal_mol_angstrom2=tensions,
        total_area_angstrom2=float(areas.sum()),
        grid_points_per_atom=int(grid_points),
    )
