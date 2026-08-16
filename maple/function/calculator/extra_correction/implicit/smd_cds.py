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

from ....route2_smd_profiles import (
    CANONICAL_SMD_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_SMD_PROFILE,
    GAFF2_CARBONYL_O_PROFILE,
    SUPPORTED_DDPCM_SMD_PROFILES,
    SUPPORTED_PCMSOLVER_SMD_PROFILES,
    SUPPORTED_ROUTE2_SMD_PROFILES,
    route2_smd_profile_spec,
)
from ....route2_solvents import route2_solvent_spec

# Historical callers import these profile constants from ``smd_cds``.  Keep
# that compatibility boundary explicit while new code imports the registry
# directly from ``route2_smd_profiles``.
_LEGACY_PROFILE_REEXPORTS = (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_SMD_PROFILE,
    GAFF2_CARBONYL_O_PROFILE,
    SUPPORTED_DDPCM_SMD_PROFILES,
    SUPPORTED_PCMSOLVER_SMD_PROFILES,
)

HARTREE_TO_KCAL_MOL = 627.5094740631
SASA_PROBE_RADIUS_ANGSTROM = 0.4
SASA_GRID_POINTS = 5810
GAFF2_CARBONYL_O_RADIUS_ANGSTROM = 1.70


# Atomic-number-indexed SMD Coulomb radii for the Route-2 element domain.
SMD_WATER_COULOMB_RADII_ANGSTROM = {
    "H": 1.20,
    "C": 1.85,
    "N": 1.89,
    "O": 1.52,
    "F": 1.73,
    "P": 2.12,
    "S": 2.49,
    "Cl": 2.38,
    # SMD18 revision recommended by the Minnesota solvation database.
    "Br": 2.60,
    "I": 2.74,
}

PYSCF_SMD_COULOMB_RADII_ANGSTROM = {
    "H": 1.20,
    "C": 1.85,
    "N": 1.89,
    "O": 1.52,
    "F": 1.73,
    "P": 2.12,
    "S": 2.49,
    "Cl": 2.38,
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
    ("N", "C3"): (1.225, 0.065),
    ("O", "C"): (1.33, 0.10),
    ("O", "N"): (1.50, 0.30),
    ("O", "O"): (1.80, 0.30),
    ("O", "P"): (2.10, 0.30),
}

_HYDROGEN_CARBON_TENSION_COEFFICIENT = -60.77
_CARBON_CARBON_TENSION_COEFFICIENT = -72.95
_NITROGEN_COORDINATION_TENSION_COEFFICIENT = -48.22
_NITROGEN_COORDINATION_POWER = 1.3
_NITROGEN_SHORT_RANGE_C3_TENSION_COEFFICIENT = 84.10
_OXYGEN_TENSION_COEFFICIENTS = {
    "C": 68.69,
    "N": 121.98,
    "P": 68.85,
}


SMD_WATER_TENSION_PARAMETER_NAMES = (
    "base:H",
    "base:C",
    "base:N",
    "base:O",
    "base:F",
    "base:P",
    "base:S",
    "base:Cl",
    "base:Br",
    "base:I",
    "environment:H-C",
    "environment:H-O",
    "environment:C-C",
    "environment:N-C-coordination-power-1.3",
    "environment:N-C3-short-range",
    "environment:O-C",
    "environment:O-N",
    "environment:O-P",
)
_SMD_WATER_TENSION_PARAMETER_INDEX = {
    name: index for index, name in enumerate(SMD_WATER_TENSION_PARAMETER_NAMES)
}
SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2 = np.frombuffer(
    np.asarray(
        [
            _WATER_BASE_TENSION.get("H", 0.0),
            _WATER_BASE_TENSION.get("C", 0.0),
            _WATER_BASE_TENSION.get("N", 0.0),
            _WATER_BASE_TENSION.get("O", 0.0),
            _WATER_BASE_TENSION.get("F", 0.0),
            _WATER_BASE_TENSION.get("P", 0.0),
            _WATER_BASE_TENSION.get("S", 0.0),
            _WATER_BASE_TENSION.get("Cl", 0.0),
            _WATER_BASE_TENSION.get("Br", 0.0),
            _WATER_BASE_TENSION.get("I", 0.0),
            _HYDROGEN_CARBON_TENSION_COEFFICIENT,
            0.0,
            _CARBON_CARBON_TENSION_COEFFICIENT,
            _NITROGEN_COORDINATION_TENSION_COEFFICIENT,
            _NITROGEN_SHORT_RANGE_C3_TENSION_COEFFICIENT,
            _OXYGEN_TENSION_COEFFICIENTS["C"],
            _OXYGEN_TENSION_COEFFICIENTS["N"],
            _OXYGEN_TENSION_COEFFICIENTS["P"],
        ],
        dtype=np.float64,
    ).tobytes(),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SMDCDSResult:
    energy_hartree: float
    energy_kcal_mol: float
    atom_areas_angstrom2: np.ndarray
    atom_tensions_cal_mol_angstrom2: np.ndarray
    total_area_angstrom2: float
    grid_points_per_atom: int


def validate_smd_symbols(
    symbols: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    normalized = tuple(str(symbol) for symbol in symbols)
    unsupported = sorted(set(normalized).difference(SMD_WATER_COULOMB_RADII_ANGSTROM))
    if unsupported:
        raise ValueError(
            "Route-2 SMD currently supports H/C/N/O/F/P/S/Cl/Br/I only; "
            f"unsupported elements: {', '.join(unsupported)}."
        )
    return normalized


def smd_coulomb_radii(symbols, *, solvent: str) -> np.ndarray:
    """Return SMD Coulomb radii for one registered solvent.

    SMD equation 16 makes the oxygen radius depend on the solvent
    hydrogen-bond acidity.  All other currently supported elemental radii
    remain the locked SMD/SMD18 values.
    """

    normalized = validate_smd_symbols(tuple(symbols))
    solvent_spec = route2_solvent_spec(solvent)
    radii = np.asarray(
        [PYSCF_SMD_COULOMB_RADII_ANGSTROM[symbol] for symbol in normalized],
        dtype=float,
    )
    acidity = solvent_spec.descriptors.hydrogen_bond_acidity
    oxygen_radius = 1.52 if acidity >= 0.43 else 1.52 + 1.8 * (0.43 - acidity)
    radii[np.asarray(normalized) == "O"] = oxygen_radius
    return radii


def smd_water_coulomb_radii(symbols) -> np.ndarray:
    return smd_coulomb_radii(symbols, solvent="water")


def route2_coulomb_radii(
    symbols,
    *,
    solvent: str,
    atom_types=None,
    profile: str = CANONICAL_SMD_PROFILE,
) -> np.ndarray:
    """Return versioned Route-2 electrostatic cavity radii."""

    normalized_symbols = validate_smd_symbols(tuple(symbols))
    solvent_spec = route2_solvent_spec(solvent)
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in SUPPORTED_ROUTE2_SMD_PROFILES:
        raise ValueError(f"Unsupported Route 2 SMD profile: {profile}.")

    profile_spec = route2_smd_profile_spec(normalized_profile)
    if not profile_spec.supports_solvent(solvent_spec.name):
        raise ValueError(
            f"Route 2 profile={normalized_profile} does not support "
            f"solvent={solvent_spec.name}."
        )

    if profile_spec.coulomb_radii_policy == "smd-water-reference-smd18-v1":
        radii = np.asarray(
            [SMD_WATER_COULOMB_RADII_ANGSTROM[symbol] for symbol in normalized_symbols],
            dtype=float,
        )
    else:
        radii = smd_coulomb_radii(
            normalized_symbols,
            solvent=solvent_spec.name,
        )
    if not profile_spec.uses_gaff2_carbonyl_oxygen:
        return radii

    if atom_types is None:
        raise ValueError(
            f"{normalized_profile} requires one GAFF/GAFF2 atom type per atom."
        )
    normalized_atom_types = tuple(
        str(atom_type).strip().lower() for atom_type in atom_types
    )
    if len(normalized_atom_types) != len(normalized_symbols):
        raise ValueError(
            f"{normalized_profile} requires one GAFF/GAFF2 atom type per atom."
        )

    carbonyl_oxygen = np.asarray(
        [
            symbol == "O" and atom_type == "o"
            for symbol, atom_type in zip(
                normalized_symbols, normalized_atom_types, strict=True
            )
        ],
        dtype=bool,
    )
    radii[carbonyl_oxygen] = GAFF2_CARBONYL_O_RADIUS_ANGSTROM
    return radii


def route2_water_coulomb_radii(
    symbols,
    *,
    atom_types=None,
    profile: str = CANONICAL_SMD_PROFILE,
) -> np.ndarray:
    """Return the versioned Route-2 electrostatic cavity radii."""
    return route2_coulomb_radii(
        symbols,
        solvent="water",
        atom_types=atom_types,
        profile=profile,
    )


def smd_sasa_radii(symbols) -> np.ndarray:
    normalized = validate_smd_symbols(tuple(symbols))
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


def _switch_with_derivative(
    distance: float,
    reference: float,
    width: float,
) -> tuple[float, float]:
    """Return the SMD switching value and its distance derivative."""

    value = _switch(distance, reference, width)
    if value == 0.0:
        return 0.0, 0.0
    denominator = distance - reference - width
    derivative = -width * value / (denominator * denominator)
    return value, derivative


def _validated_tension_coefficients(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    expected = (len(SMD_WATER_TENSION_PARAMETER_NAMES),)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(
            "SMD water tension coefficients must be finite with shape " f"{expected}."
        )
    return values


def aqueous_atomic_surface_tension_basis(
    symbols,
    positions_angstrom: np.ndarray,
) -> np.ndarray:
    """Return the linear aqueous SMD atomic-tension design matrix.

    Each row describes one atomic surface tension and each column follows
    :data:`SMD_WATER_TENSION_PARAMETER_NAMES`.  The published stock SMD
    tensions are exactly ``basis @ stock_coefficients``.  The aqueous H-O
    branch is retained even though its published stock coefficient is zero,
    allowing a future preregistered linear refit without changing geometry
    functions after observing data.
    """

    symbols = validate_smd_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    if positions.shape != (len(symbols), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("SMD coordinates must be finite with shape (n_atoms, 3).")
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )
    basis = np.zeros(
        (len(symbols), len(SMD_WATER_TENSION_PARAMETER_NAMES)),
        dtype=float,
    )
    for i, symbol in enumerate(symbols):
        basis[i, _SMD_WATER_TENSION_PARAMETER_INDEX[f"base:{symbol}"]] = 1.0
        if symbol in {"F", "S", "Cl", "Br", "I", "P"}:
            continue

        if symbol == "H":
            basis[i, _SMD_WATER_TENSION_PARAMETER_INDEX["environment:H-C"]] = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("H", "C")])
                for j, other in enumerate(symbols)
                if other == "C"
            )
            basis[i, _SMD_WATER_TENSION_PARAMETER_INDEX["environment:H-O"]] = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("H", "O")])
                for j, other in enumerate(symbols)
                if other == "O"
            )
            continue

        if symbol == "C":
            basis[i, _SMD_WATER_TENSION_PARAMETER_INDEX["environment:C-C"]] = sum(
                _switch(distances[i, j], *_SWITCH_PARAMETERS[("C", "C")])
                for j, other in enumerate(symbols)
                if j != i and other == "C"
            )
            continue

        if symbol == "N":
            t_nc = 0.0
            t_nc3 = 0.0
            for j, other in enumerate(symbols):
                if other != "C":
                    continue
                carbon_environment = 0.0
                for k, neighbor in enumerate(symbols):
                    if k in {i, j}:
                        continue
                    parameters = _SWITCH_PARAMETERS.get(("C", neighbor), (0.0, 0.0))
                    if parameters[1] > 0.0:
                        carbon_environment += _switch(distances[j, k], *parameters)
                nc_coordination = _switch(
                    distances[i, j], *_SWITCH_PARAMETERS[("N", "C")]
                )
                t_nc += nc_coordination * carbon_environment**2
                t_nc3 += _switch(distances[i, j], *_SWITCH_PARAMETERS[("N", "C3")])
            basis[
                i,
                _SMD_WATER_TENSION_PARAMETER_INDEX[
                    "environment:N-C-coordination-power-1.3"
                ],
            ] = (
                t_nc**_NITROGEN_COORDINATION_POWER
            )
            basis[
                i,
                _SMD_WATER_TENSION_PARAMETER_INDEX["environment:N-C3-short-range"],
            ] = t_nc3
            continue

        if symbol == "O":
            for neighbor in ("C", "N", "P"):
                basis[
                    i,
                    _SMD_WATER_TENSION_PARAMETER_INDEX[f"environment:O-{neighbor}"],
                ] = sum(
                    _switch(
                        distances[i, j],
                        *_SWITCH_PARAMETERS[("O", neighbor)],
                    )
                    for j, other in enumerate(symbols)
                    if other == neighbor
                )
    return basis


def aqueous_atomic_surface_tensions_from_coefficients(
    symbols,
    positions_angstrom: np.ndarray,
    coefficients_cal_mol_angstrom2: np.ndarray,
) -> np.ndarray:
    """Evaluate aqueous atomic tensions for one frozen linear coefficient set."""

    coefficients = _validated_tension_coefficients(coefficients_cal_mol_angstrom2)
    return (
        aqueous_atomic_surface_tension_basis(symbols, positions_angstrom) @ coefficients
    )


def aqueous_atomic_surface_tensions(
    symbols,
    positions_angstrom: np.ndarray,
) -> np.ndarray:
    """Return published aqueous SMD atomic tensions in cal/(mol Å²)."""

    return aqueous_atomic_surface_tensions_from_coefficients(
        symbols,
        positions_angstrom,
        SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    )


def aqueous_cds_tension_design_row(
    symbols,
    positions_angstrom: np.ndarray,
    atom_areas_angstrom2: np.ndarray,
) -> np.ndarray:
    """Contract atomic areas with the linear tension basis.

    The returned row has units such that its dot product with coefficients in
    ``cal/(mol Å²)`` is the CDS energy in ``kcal/mol``.  Area generation is an
    independent, explicitly versioned concern and is therefore supplied by the
    caller rather than hidden in this function.
    """

    symbols = validate_smd_symbols(tuple(symbols))
    areas = np.asarray(atom_areas_angstrom2, dtype=float)
    if areas.shape != (len(symbols),) or not np.all(np.isfinite(areas)):
        raise ValueError("SMD atom areas must be finite with shape (n_atoms,).")
    basis = aqueous_atomic_surface_tension_basis(symbols, positions_angstrom)
    return areas @ basis / 1000.0


def aqueous_atomic_surface_tension_position_vjp_from_coefficients(
    symbols,
    positions_angstrom: np.ndarray,
    tension_cotangent: np.ndarray,
    coefficients_cal_mol_angstrom2: np.ndarray,
) -> np.ndarray:
    """Differentiate a pairing with a frozen linear aqueous tension model.

    This returns
    ``d <tension_cotangent, basis(R) @ coefficients> / dR`` without constructing
    a dense tension-by-coordinate Jacobian.  Solvent-accessible surface-area
    derivatives are separate and deliberately absent, so this is not a complete
    CDS gradient.
    """

    symbols = validate_smd_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    cotangent = np.asarray(tension_cotangent, dtype=float)
    coefficients = _validated_tension_coefficients(coefficients_cal_mol_angstrom2)
    expected_positions_shape = (len(symbols), 3)
    if positions.shape != expected_positions_shape or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError("SMD coordinates must be finite with shape (n_atoms, 3).")
    if cotangent.shape != (len(symbols),) or not np.all(np.isfinite(cotangent)):
        raise ValueError(
            "SMD atomic-tension cotangent must be finite with shape (n_atoms,)."
        )

    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )
    pair_cotangents = np.zeros((len(symbols), len(symbols)), dtype=float)

    def add_pair(first: int, second: int, value: float) -> None:
        if first == second or value == 0.0:
            return
        lower, upper = sorted((first, second))
        pair_cotangents[lower, upper] += value

    def switch(
        first: int,
        second: int,
        pair: tuple[str, str],
    ) -> tuple[float, float]:
        return _switch_with_derivative(
            float(distances[first, second]),
            *_SWITCH_PARAMETERS[pair],
        )

    for i, symbol in enumerate(symbols):
        objective_cotangent = float(cotangent[i])
        if objective_cotangent == 0.0:
            continue
        if symbol in {"F", "S", "Cl", "Br", "I", "P"}:
            continue

        if symbol == "H":
            for j, other in enumerate(symbols):
                if other not in {"C", "O"}:
                    continue
                _, derivative = switch(i, j, ("H", other))
                add_pair(
                    i,
                    j,
                    objective_cotangent
                    * coefficients[
                        _SMD_WATER_TENSION_PARAMETER_INDEX[f"environment:H-{other}"]
                    ]
                    * derivative,
                )
            continue

        if symbol == "C":
            for j, other in enumerate(symbols):
                if j == i or other != "C":
                    continue
                _, derivative = switch(i, j, ("C", "C"))
                add_pair(
                    i,
                    j,
                    objective_cotangent
                    * coefficients[
                        _SMD_WATER_TENSION_PARAMETER_INDEX["environment:C-C"]
                    ]
                    * derivative,
                )
            continue

        if symbol == "N":
            carbon_branches = []
            t_nc = 0.0
            for j, other in enumerate(symbols):
                if other != "C":
                    continue
                carbon_environment = 0.0
                environment_branches = []
                for k, neighbor in enumerate(symbols):
                    if k in {i, j}:
                        continue
                    parameters = _SWITCH_PARAMETERS.get(
                        ("C", neighbor),
                        (0.0, 0.0),
                    )
                    if parameters[1] <= 0.0:
                        continue
                    coordination, coordination_derivative = _switch_with_derivative(
                        float(distances[j, k]),
                        *parameters,
                    )
                    carbon_environment += coordination
                    environment_branches.append((k, coordination_derivative))
                nc_coordination, nc_derivative = switch(i, j, ("N", "C"))
                _, nc3_derivative = switch(i, j, ("N", "C3"))
                t_nc += nc_coordination * carbon_environment**2
                carbon_branches.append(
                    (
                        j,
                        nc_coordination,
                        nc_derivative,
                        nc3_derivative,
                        carbon_environment,
                        environment_branches,
                    )
                )

            t_nc_gradient = (
                0.0
                if t_nc == 0.0
                else coefficients[
                    _SMD_WATER_TENSION_PARAMETER_INDEX[
                        "environment:N-C-coordination-power-1.3"
                    ]
                ]
                * _NITROGEN_COORDINATION_POWER
                * t_nc ** (_NITROGEN_COORDINATION_POWER - 1.0)
            )
            for (
                j,
                nc_coordination,
                nc_derivative,
                nc3_derivative,
                carbon_environment,
                environment_branches,
            ) in carbon_branches:
                nc_cotangent = (
                    objective_cotangent * t_nc_gradient * carbon_environment**2
                )
                add_pair(
                    i,
                    j,
                    nc_cotangent * nc_derivative
                    + objective_cotangent
                    * coefficients[
                        _SMD_WATER_TENSION_PARAMETER_INDEX[
                            "environment:N-C3-short-range"
                        ]
                    ]
                    * nc3_derivative,
                )

                environment_cotangent = (
                    objective_cotangent
                    * t_nc_gradient
                    * nc_coordination
                    * 2.0
                    * carbon_environment
                )
                for k, coordination_derivative in environment_branches:
                    add_pair(
                        j,
                        k,
                        environment_cotangent * coordination_derivative,
                    )
            continue

        if symbol == "O":
            for j, other in enumerate(symbols):
                parameter_name = f"environment:O-{other}"
                parameter_index = _SMD_WATER_TENSION_PARAMETER_INDEX.get(parameter_name)
                if parameter_index is None:
                    continue
                _, derivative = switch(i, j, ("O", other))
                add_pair(
                    i,
                    j,
                    objective_cotangent * coefficients[parameter_index] * derivative,
                )

    position_vjp = np.zeros_like(positions)
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            pair_cotangent = pair_cotangents[first, second]
            if pair_cotangent == 0.0:
                continue
            distance = float(distances[first, second])
            if distance <= 1.0e-14:
                raise ValueError(
                    "Distinct atoms with geometry-dependent SMD tensions "
                    "cannot occupy the same position."
                )
            direction = (positions[first] - positions[second]) / distance
            contribution = pair_cotangent * direction
            position_vjp[first] += contribution
            position_vjp[second] -= contribution
    return position_vjp


def aqueous_atomic_surface_tension_position_vjp(
    symbols,
    positions_angstrom: np.ndarray,
    tension_cotangent: np.ndarray,
) -> np.ndarray:
    """Differentiate a pairing with the published aqueous SMD tensions."""

    return aqueous_atomic_surface_tension_position_vjp_from_coefficients(
        symbols,
        positions_angstrom,
        tension_cotangent,
        SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    )


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
    for atom_index, (center, radius) in enumerate(zip(positions, radii, strict=True)):
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


def _swig_switch_with_derivative(
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Lange--Herbert SWIG switch and its first derivative."""

    coordinate = np.asarray(coordinate, dtype=float)
    values = np.empty_like(coordinate)
    derivatives = np.zeros_like(coordinate)
    inside = coordinate <= 0.0
    outside = coordinate >= 1.0
    switching = ~(inside | outside)
    x = coordinate[switching]
    values[inside] = 0.0
    values[outside] = 1.0
    values[switching] = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
    derivatives[switching] = 30.0 * x**2 * (1.0 - x) ** 2
    return values, derivatives


def _fibonacci_swig_inspired_surface_areas_and_position_vjp(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    area_cotangent: np.ndarray | None,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate Fibonacci-grid, SWIG-inspired areas and one optional VJP.

    The switching function and radius parameterization follow Lange and
    Herbert, J. Chem. Phys. 133, 244111 (2010), but are evaluated on MAPLE's
    deterministic equal-area Fibonacci nodes instead of the published
    Lebedev grid. This research candidate is not an implementation of the
    original SWIG discretization and does not include Gaussian ASC
    electrostatics.
    """

    positions = np.asarray(positions_angstrom, dtype=float)
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "Fibonacci-SWIG-inspired coordinates must be finite with shape "
            "(n_atoms, 3)."
        )
    if (
        radii.shape != (positions.shape[0],)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            "Fibonacci-SWIG-inspired radii must be one finite positive value "
            "per atom."
        )

    cotangent = None
    if area_cotangent is not None:
        cotangent = np.asarray(area_cotangent, dtype=float)
        if cotangent.shape != (positions.shape[0],) or not np.all(
            np.isfinite(cotangent)
        ):
            raise ValueError(
                "Fibonacci-SWIG-inspired area cotangent must be finite with "
                "shape (n_atoms,)."
            )

    number = int(grid_points)
    directions = _fibonacci_sphere(number)
    point_weight = 4.0 * math.pi / number
    switching_radii = radii * math.sqrt(14.0 / number)
    radius_ratios = radii / switching_radii
    square_roots = np.sqrt(radius_ratios**2 - 1.0 / 28.0)
    # This is algebraically identical to the published/PySCF expression
    # 1/2 + ratio - sqrt(ratio**2 - 1/28), but avoids cancellation for dense
    # surface grids.
    alpha = 0.5 + (1.0 / 28.0) / (radius_ratios + square_roots)
    inner_radii = radii - alpha * switching_radii
    outer_radii = inner_radii + switching_radii

    atom_count = positions.shape[0]
    areas = np.empty(atom_count, dtype=float)
    position_vjp = None if cotangent is None else np.zeros_like(positions)
    center_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )

    for atom_index, (center, radius) in enumerate(zip(positions, radii, strict=True)):
        surface_points = center + radius * directions
        candidates = np.flatnonzero(
            (np.arange(atom_count) != atom_index)
            & (center_distances[atom_index] < radius + outer_radii)
        )
        area_scale = point_weight * radius * radius
        if candidates.size == 0:
            areas[atom_index] = area_scale * number
            continue

        displacements = surface_points[:, None, :] - positions[candidates][None, :, :]
        distances = np.linalg.norm(displacements, axis=2)
        switch_coordinates = (
            distances - inner_radii[candidates][None, :]
        ) / switching_radii[candidates][None, :]
        switches, switch_derivatives = _swig_switch_with_derivative(switch_coordinates)
        point_switches = np.prod(switches, axis=1)
        areas[atom_index] = area_scale * float(np.sum(point_switches))

        if position_vjp is None or cotangent[atom_index] == 0.0:
            continue

        prefix = np.ones((number, candidates.size + 1), dtype=float)
        prefix[:, 1:] = np.cumprod(switches, axis=1)
        suffix = np.ones((number, candidates.size + 1), dtype=float)
        suffix[:, :-1] = np.cumprod(switches[:, ::-1], axis=1)[:, ::-1]
        products_without_neighbor = prefix[:, :-1] * suffix[:, 1:]

        active = switch_derivatives != 0.0
        if np.any(active & (distances <= 1.0e-14)):
            raise ValueError(
                "An active Fibonacci-SWIG-inspired surface point cannot "
                "coincide with a neighbouring atom center."
            )
        inverse_distances = np.zeros_like(distances)
        inverse_distances[active] = 1.0 / distances[active]
        derivative_coefficients = (
            float(cotangent[atom_index])
            * area_scale
            * products_without_neighbor
            * switch_derivatives
            * inverse_distances
            / switching_radii[candidates][None, :]
        )
        contributions = derivative_coefficients[:, :, None] * displacements
        position_vjp[atom_index] += np.sum(contributions, axis=(0, 1))
        np.add.at(
            position_vjp,
            candidates,
            -np.sum(contributions, axis=0),
        )

    return areas, position_vjp


def fibonacci_swig_inspired_solvent_accessible_surface_areas(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> np.ndarray:
    """Return Fibonacci-grid, SWIG-inspired atomwise accessible areas.

    The canonical Route-2 SMD energy continues to use
    :func:`solvent_accessible_surface_areas`.  This separately named function
    is an experimental, internally energy-consistent candidate. It is not
    equivalent to the published Lebedev-SWIG discretization.
    """

    areas, _ = _fibonacci_swig_inspired_surface_areas_and_position_vjp(
        positions_angstrom,
        radii_angstrom,
        area_cotangent=None,
        grid_points=grid_points,
    )
    return areas


def fibonacci_swig_inspired_solvent_accessible_surface_area_position_vjp(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    area_cotangent: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> np.ndarray:
    """Differentiate a pairing with the Fibonacci/SWIG-inspired areas."""

    _, position_vjp = _fibonacci_swig_inspired_surface_areas_and_position_vjp(
        positions_angstrom,
        radii_angstrom,
        area_cotangent=area_cotangent,
        grid_points=grid_points,
    )
    assert position_vjp is not None
    return position_vjp


def smd_water_cds(
    symbols,
    positions_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> SMDCDSResult:
    """Compute the aqueous SMD cavity/dispersion/solvent-structure term."""

    symbols = validate_smd_symbols(tuple(symbols))
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


def smd_water_cds_fibonacci_swig_inspired(
    symbols,
    positions_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> SMDCDSResult:
    """Compute an experimental CDS term with Fibonacci/SWIG-inspired areas.

    This function is deliberately separate from :func:`smd_water_cds`; it is
    neither Lebedev-SWIG nor selected by the public Route-2 provider.
    """

    symbols = validate_smd_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    tensions = aqueous_atomic_surface_tensions(symbols, positions)
    areas = fibonacci_swig_inspired_solvent_accessible_surface_areas(
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


def smd_water_cds_fibonacci_swig_inspired_position_gradient(
    symbols,
    positions_angstrom: np.ndarray,
    *,
    grid_points: int = SASA_GRID_POINTS,
) -> np.ndarray:
    """Return this experimental discrete CDS gradient in hartree/angstrom."""

    symbols = validate_smd_symbols(tuple(symbols))
    positions = np.asarray(positions_angstrom, dtype=float)
    tensions = aqueous_atomic_surface_tensions(symbols, positions)
    energy_scale = 1.0 / (1000.0 * HARTREE_TO_KCAL_MOL)
    (
        areas,
        area_gradient,
    ) = _fibonacci_swig_inspired_surface_areas_and_position_vjp(
        positions,
        smd_sasa_radii(symbols),
        area_cotangent=tensions * energy_scale,
        grid_points=grid_points,
    )
    assert area_gradient is not None
    tension_gradient = aqueous_atomic_surface_tension_position_vjp(
        symbols,
        positions,
        areas * energy_scale,
    )
    return area_gradient + tension_gradient
