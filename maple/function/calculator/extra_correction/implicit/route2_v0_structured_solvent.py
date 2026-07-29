"""MACE-native electrostatic grid source for the no-training Route-2 V0-FD-S path.

At a fixed nuclear geometry, this module evaluates only the all-space Gaussian
``l<=1`` MACE-POLAR potential on a regular Cartesian grid,

``phi_c0^G(r) = <c0, Coulomb_Gaussian(r; R)>``.

It deliberately does *not* construct a complete solute--solvent interaction
or a solvation free energy.  A molecular liquid functional still requires a
separately provenanced positive short-range source, a frozen solvent asset,
and an independently verified MDFT/3D-RISM backend.  Returning a total
interaction before those inputs exist would silently invent the missing Pauli
repulsion and dispersion physics.

The grid values are in Hartree per elementary charge.  A solvent site with
charge ``z`` couples through ``u_el = z * phi_c0^G`` in Hartree.  No MACE field
response, charge projection, empirical correction, or atom-charge surrogate
is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    gaussian_multipole_potential,
)


V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION = (
    "route2-v0-fixed-density-structured-electrostatic-grid-v1"
)
V0_STRUCTURED_SOLVENT_GRID_LAYOUT = "x-y-z-c-order-v1"


def _immutable_vector(
    values: np.ndarray,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    """Return one immutable finite Cartesian vector."""

    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (3,).")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _grid_shape(values: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a positive, three-dimensional Cartesian grid shape."""

    if len(values) != 3:
        raise ValueError("Grid shape must contain exactly three dimensions.")
    shape: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Grid dimensions must be positive integers.")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Grid dimensions must be positive integers.") from exc
        if integer != value or integer < 1:
            raise ValueError("Grid dimensions must be positive integers.")
        shape.append(integer)
    return tuple(shape)  # type: ignore[return-value]


def _immutable_density(values: np.ndarray) -> np.ndarray:
    """Freeze finite Route-2 ``l<=1`` MACE density coefficients."""

    density = np.asarray(values, dtype=float)
    if (
        density.ndim != 2
        or density.shape[0] == 0
        or density.shape[1] != 4
        or not np.all(np.isfinite(density))
    ):
        raise ValueError(
            "MACE density coefficients must be finite with shape (n_atoms, 4)."
        )
    result = np.array(density, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_positions(values: np.ndarray, *, atom_count: int) -> np.ndarray:
    """Freeze finite atom centres in Angstrom."""

    positions = np.asarray(values, dtype=float)
    if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            "MACE atom positions must be finite with shape "
            f"({atom_count}, 3); received {positions.shape}."
        )
    result = np.array(positions, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RegularCartesianGrid:
    """One regular Cartesian grid with positions expressed in Bohr.

    The three-dimensional array layout is ``(x, y, z)`` in NumPy C order.
    ``points_bohr()`` returns the matching flattened coordinate order, so
    reshaping a pointwise result back to :attr:`shape` is unambiguous.
    """

    origin_bohr: np.ndarray
    spacing_bohr: np.ndarray
    shape: tuple[int, int, int]
    layout: str = V0_STRUCTURED_SOLVENT_GRID_LAYOUT

    def __post_init__(self) -> None:
        origin = _immutable_vector(self.origin_bohr, name="Grid origin")
        spacing = _immutable_vector(
            self.spacing_bohr,
            name="Grid spacing",
            positive=True,
        )
        shape = _grid_shape(self.shape)
        if self.layout != V0_STRUCTURED_SOLVENT_GRID_LAYOUT:
            raise ValueError("Unsupported Route-2 structured-solvent grid layout.")
        object.__setattr__(self, "origin_bohr", origin)
        object.__setattr__(self, "spacing_bohr", spacing)
        object.__setattr__(self, "shape", shape)

    @property
    def point_count(self) -> int:
        """Return the number of Cartesian grid points."""

        return int(np.prod(self.shape))

    @property
    def volume_element_bohr3(self) -> float:
        """Return the uniform integration cell volume in Bohr cubed."""

        return float(np.prod(self.spacing_bohr))

    def points_bohr(self) -> np.ndarray:
        """Return grid coordinates ordered consistently with a C-order volume."""

        axes = tuple(
            self.origin_bohr[axis]
            + self.spacing_bohr[axis] * np.arange(self.shape[axis], dtype=float)
            for axis in range(3)
        )
        coordinates = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        ).reshape((-1, 3))
        coordinates.setflags(write=False)
        return coordinates


@dataclass(frozen=True)
class Route2V0StructuredSolventElectrostaticSource:
    """Immutable MACE Gaussian electrostatic source on one liquid grid.

    This object deliberately marks its scope as electrostatic-only.  The grid
    cannot stand in for the short-range potential, a solvent functional, or a
    total solvation free energy.
    """

    grid: RegularCartesianGrid
    density_coefficients: np.ndarray
    atom_positions_angstrom: np.ndarray
    potential_hartree_per_e: np.ndarray
    target_total_charge_e: float
    total_charge_e: float
    construction: str = V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION

    interaction_scope: str = "electrostatic-grid-source-only-v1"

    def __post_init__(self) -> None:
        density = _immutable_density(self.density_coefficients)
        positions = _immutable_positions(
            self.atom_positions_angstrom,
            atom_count=density.shape[0],
        )
        potential = np.asarray(self.potential_hartree_per_e, dtype=float)
        if potential.shape != self.grid.shape or not np.all(np.isfinite(potential)):
            raise ValueError(
                "Structured-solvent potential must be finite with the grid shape."
            )
        potential = np.array(potential, dtype=float, copy=True)
        potential.setflags(write=False)
        target_charge = float(self.target_total_charge_e)
        total_charge = float(self.total_charge_e)
        if not all(math.isfinite(value) for value in (target_charge, total_charge)):
            raise ValueError("Structured-solvent source charges must be finite.")
        density_charge = float(np.sum(density[:, 0]))
        tolerance = max(1.0e-12, 1.0e-10 * abs(density_charge))
        if abs(total_charge - density_charge) > tolerance:
            raise ValueError(
                "Structured-solvent total charge must equal the MACE monopole sum."
            )
        if self.construction != V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 structured-solvent construction.")
        if self.interaction_scope != "electrostatic-grid-source-only-v1":
            raise ValueError("Unsupported Route-2 structured-solvent source scope.")
        object.__setattr__(self, "density_coefficients", density)
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(self, "potential_hartree_per_e", potential)
        object.__setattr__(self, "target_total_charge_e", target_charge)
        object.__setattr__(self, "total_charge_e", total_charge)

    def site_electrostatic_energy_hartree(self, site_charge_e: float) -> np.ndarray:
        """Return the electrostatic site-energy grid ``z * phi_c0^G``.

        This is only the electrostatic summand of a future molecular-liquid
        external potential.  It intentionally does not synthesize a
        short-range interaction for the site.
        """

        charge = float(site_charge_e)
        if not math.isfinite(charge):
            raise ValueError("Solvent site charge must be finite.")
        energy = np.array(charge * self.potential_hartree_per_e, copy=True)
        energy.setflags(write=False)
        return energy


def evaluate_route2_v0_structured_solvent_electrostatic_source(
    *,
    density_coefficients: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    grid: RegularCartesianGrid,
    target_total_charge_e: float = 0.0,
    total_charge_tolerance_e: float = 1.0e-10,
) -> Route2V0StructuredSolventElectrostaticSource:
    """Evaluate a frozen MACE Gaussian electrostatic source on ``grid``.

    The only supported Gaussian width is
    :data:`MACE_POLAR_DENSITY_SIGMA_ANGSTROM`; changing it would alter the
    frozen MACE source rather than refine grid resolution.  A caller must
    supply the same zero-field MACE density used by the V0-FD construction.
    The function validates, but does not repair, the total charge.
    """

    density = _immutable_density(density_coefficients)
    positions = _immutable_positions(
        atom_positions_angstrom,
        atom_count=density.shape[0],
    )
    target_charge = float(target_total_charge_e)
    tolerance = float(total_charge_tolerance_e)
    if not math.isfinite(target_charge):
        raise ValueError("Target total charge must be finite.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Total-charge tolerance must be finite and positive.")
    total_charge = float(np.sum(density[:, 0]))
    if abs(total_charge - target_charge) > tolerance:
        raise ValueError(
            "Frozen MACE density violates its total-charge constraint "
            f"(observed={total_charge:.16e} e, target={target_charge:.16e} e)."
        )
    potential = gaussian_multipole_potential(
        grid.points_bohr(),
        positions,
        density,
        sigma_angstrom=MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
    ).reshape(grid.shape)
    return Route2V0StructuredSolventElectrostaticSource(
        grid=grid,
        density_coefficients=density,
        atom_positions_angstrom=positions,
        potential_hartree_per_e=potential,
        target_total_charge_e=target_charge,
        total_charge_e=total_charge,
    )


__all__ = [
    "RegularCartesianGrid",
    "Route2V0StructuredSolventElectrostaticSource",
    "V0_STRUCTURED_SOLVENT_ELECTROSTATIC_GRID_CONSTRUCTION",
    "V0_STRUCTURED_SOLVENT_GRID_LAYOUT",
    "evaluate_route2_v0_structured_solvent_electrostatic_source",
]
