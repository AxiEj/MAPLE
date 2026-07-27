"""Shared chemical-domain validation for Route-2 solvation providers."""

from __future__ import annotations

import numpy as np
from ase.data import covalent_radii


SUPPORTED_ELEMENTS = frozenset(
    {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
)
FORMALLY_CHARGED_TRIPOS_TYPES = frozenset({"n.4", "c.cat", "o.co2"})
MIN_MOLECULAR_MASS_DA = 16.0
MAX_MOLECULAR_MASS_DA = 500.0
CONNECTEDNESS_BOND_SCALE = 1.25


def _covalent_component_count(atoms) -> int:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atom_count = len(atoms)
    neighbours: list[list[int]] = [[] for _ in range(atom_count)]
    for first in range(atom_count):
        first_radius = float(covalent_radii[int(atoms.numbers[first])])
        for second in range(first + 1, atom_count):
            second_radius = float(
                covalent_radii[int(atoms.numbers[second])]
            )
            threshold = CONNECTEDNESS_BOND_SCALE * (
                first_radius + second_radius
            )
            distance = float(
                np.linalg.norm(positions[first] - positions[second])
            )
            if distance <= threshold:
                neighbours[first].append(second)
                neighbours[second].append(first)

    visited: set[int] = set()
    component_count = 0
    for start in range(atom_count):
        if start in visited:
            continue
        component_count += 1
        visited.add(start)
        pending = [start]
        while pending:
            atom = pending.pop()
            for neighbour in neighbours[atom]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
    return component_count


def validate_route2_domain(atoms) -> None:
    """Require one neutral, closed-shell, non-periodic organic molecule."""

    if atoms is None or len(atoms) == 0:
        raise ValueError("Route 2 requires one non-empty molecule.")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.shape != (len(atoms), 3) or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError("Route 2 coordinates must be finite.")
    symbols = tuple(atoms.get_chemical_symbols())
    unsupported = sorted(set(symbols).difference(SUPPORTED_ELEMENTS))
    if unsupported:
        raise ValueError(
            "Route 2 supports H/C/N/O/F/P/S/Cl/Br/I only; unsupported elements: "
            + ", ".join(unsupported)
            + "."
        )
    molecular_mass = float(np.sum(atoms.get_masses()))
    if not MIN_MOLECULAR_MASS_DA <= molecular_mass <= MAX_MOLECULAR_MASS_DA:
        raise ValueError(
            "Route 2 v1 is validated for neutral organics from 16 to 500 Da; "
            f"received {molecular_mass:.6f} Da."
        )
    if "charge" not in atoms.info or "mult" not in atoms.info:
        raise ValueError(
            "Route 2 requires explicit molecular charge and multiplicity "
            "metadata; declare the neutral closed-shell state as '0 1'."
        )
    try:
        charge = float(atoms.info["charge"])
        multiplicity_value = float(atoms.info["mult"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Route 2 requires numeric charge=0 and multiplicity=1 metadata."
        ) from exc
    if not multiplicity_value.is_integer():
        raise ValueError("Route 2 multiplicity metadata must be an integer.")
    multiplicity = int(multiplicity_value)
    if charge != 0.0 or multiplicity != 1:
        raise ValueError(
            "Route 2 v1 supports neutral closed-shell molecules only "
            f"(received charge={charge:g}, multiplicity={multiplicity})."
        )
    electron_count = int(np.sum(np.asarray(atoms.numbers, dtype=int)))
    if electron_count % 2 != (multiplicity - 1) % 2:
        raise ValueError(
            "Route 2 electron-count parity is inconsistent with the declared "
            f"multiplicity={multiplicity}: the neutral molecule has "
            f"{electron_count} electrons."
        )
    component_count = _covalent_component_count(atoms)
    if component_count != 1:
        raise ValueError(
            "Route 2 requires one connected molecule; the 1.25x covalent-"
            f"radius graph contains {component_count} components."
        )
    mol2 = atoms.info.get("mol2")
    if isinstance(mol2, dict):
        atom_types = {
            str(atom_type).strip().lower()
            for atom_type in mol2.get("atom_types", ())
        }
        charged_markers = sorted(
            atom_types.intersection(FORMALLY_CHARGED_TRIPOS_TYPES)
        )
        if charged_markers:
            raise ValueError(
                "Route 2 v1 excludes salts and zwitterions; the MOL2 uses "
                "formally charged Tripos atom type(s): "
                + ", ".join(charged_markers)
                + "."
            )
    if np.any(atoms.get_pbc()):
        raise ValueError("Route 2 SMD is non-periodic.")


__all__ = [
    "CONNECTEDNESS_BOND_SCALE",
    "FORMALLY_CHARGED_TRIPOS_TYPES",
    "MAX_MOLECULAR_MASS_DA",
    "MIN_MOLECULAR_MASS_DA",
    "SUPPORTED_ELEMENTS",
    "validate_route2_domain",
]
