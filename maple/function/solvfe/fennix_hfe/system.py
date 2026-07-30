"""Fail-closed system validation for the FeNNix-Bio1 HFE mechanics kernel."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .types import FeNNixAlchemicalSystem

FENNIX_ALCHEMICAL_CUTOFF_ANGSTROM = 7.5


def validate_fennix_alchemical_system(
    system: FeNNixAlchemicalSystem,
) -> FeNNixAlchemicalSystem:
    """Validate and normalize the currently admitted neutral water-box boundary."""

    if not isinstance(system, FeNNixAlchemicalSystem):
        raise TypeError("system must be a FeNNixAlchemicalSystem.")
    species = np.asarray(system.atomic_numbers)
    coordinates = np.asarray(system.coordinates_angstrom, dtype=np.float64)
    molecule_ids = np.asarray(system.molecule_ids)
    natoms = len(system.atomic_numbers)
    if natoms == 0 or species.shape != (natoms,) or coordinates.shape != (natoms, 3):
        raise ValueError(
            "FeNNix alchemical systems require N atoms and N x 3 coordinates."
        )
    if species.dtype.kind not in "iu" or np.any(species < 1) or np.any(species > 86):
        raise ValueError("FeNNix atomic numbers must be integers in [1, 86].")
    if not np.isfinite(coordinates).all():
        raise ValueError("FeNNix coordinates must be finite angstrom values.")
    if molecule_ids.shape != (natoms,) or molecule_ids.dtype.kind not in "iu":
        raise ValueError("FeNNix molecule_ids must contain one integer per atom.")
    if tuple(bool(value) for value in system.pbc) != (True, True, True):
        raise ValueError(
            "FeNNix alchemical systems require full three-dimensional periodic PBC."
        )
    if (
        system.total_charge_e != 0
        or system.multiplicity != 1
        or system.solute_charge_e != 0
        or system.solute_multiplicity != 1
    ):
        raise ValueError(
            "FeNNix alchemical systems are admitted only as neutral closed-shell systems."
        )

    solute = np.asarray(system.solute_atom_indices)
    if solute.ndim != 1 or solute.size == 0 or solute.dtype.kind not in "iu":
        raise ValueError("FeNNix requires explicit integer indices for one solute.")
    if (
        np.any(solute < 0)
        or np.any(solute >= natoms)
        or np.unique(solute).size != solute.size
    ):
        raise ValueError("FeNNix solute_atom_indices must be unique in-range indices.")
    solute_ids = np.unique(molecule_ids[solute])
    if solute_ids.size != 1:
        raise ValueError("All solute atoms must belong to exactly one molecule_id.")
    solute_id = int(solute_ids[0])
    expected_solute = np.flatnonzero(molecule_ids == solute_id)
    if not np.array_equal(np.sort(solute), expected_solute):
        raise ValueError(
            "solute_atom_indices must select the complete single solute molecule."
        )
    if int(species[solute].sum()) % 2:
        raise ValueError(
            "The declared neutral singlet solute must have an even electron count."
        )
    solvent_ids = [
        int(value) for value in np.unique(molecule_ids) if int(value) != solute_id
    ]
    if not solvent_ids:
        raise ValueError(
            "FeNNix HFE mechanics require at least one explicit water molecule."
        )
    for molecule_id in solvent_ids:
        composition = sorted(
            int(value) for value in species[molecule_ids == molecule_id]
        )
        if composition != [1, 1, 8]:
            raise ValueError(
                "Every non-solute molecule must be one neutral water H2O molecule."
            )

    cell = np.asarray(system.cell_angstrom, dtype=np.float64)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("FeNNix requires a finite 3 x 3 cell in angstrom.")
    volume = float(np.linalg.det(cell))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("FeNNix requires a positive-volume periodic cell.")
    face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[0], cell[2])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ]
    )
    if np.any(face_areas <= 0.0):
        raise ValueError("FeNNix requires a non-degenerate periodic cell.")
    skin = float(system.neighbor_skin_angstrom)
    if not math.isfinite(skin) or skin < 0.0:
        raise ValueError(
            "FeNNix neighbor skin must be a finite non-negative angstrom value."
        )
    minimum_height = float(np.min(volume / face_areas))
    required_height = 2.0 * (FENNIX_ALCHEMICAL_CUTOFF_ANGSTROM + skin)
    if minimum_height <= required_height:
        raise ValueError(
            "FeNNix minimum cell height must be greater than "
            f"2 * (7.5 angstrom + skin) = {required_height:g} angstrom."
        )

    return replace(
        system,
        atomic_numbers=tuple(int(value) for value in species),
        coordinates_angstrom=tuple(tuple(float(x) for x in row) for row in coordinates),
        cell_angstrom=tuple(tuple(float(x) for x in row) for row in cell),
        molecule_ids=tuple(int(value) for value in molecule_ids),
        solute_atom_indices=tuple(int(value) for value in np.sort(solute)),
        pbc=(True, True, True),
        neighbor_skin_angstrom=skin,
    )
