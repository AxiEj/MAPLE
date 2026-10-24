"""Tripos MOL2 reader for implicit-solvent inputs.

The reader deliberately consumes only information explicitly present in MOL2:
coordinates, atom types, bonds, substructure labels, and partial charges.  It
does not infer bond orders, protonation states, or missing hydrogens from XYZ
geometry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


MOL2_CHARGE_TOL = 1.0e-4


_GAFF_TYPES_BY_ELEMENT = {
    # Exact GAFF/GAFF2 labels from the sibling Route 1 mapping.
    "C": (
        "c", "c1", "c2", "c3", "c5", "c6", "ca", "cc",
        "cd", "ce", "cf", "cg", "ch", "cp", "cq", "cs",
        "cu", "cv", "cx", "cy", "cz",
    ),
    "H": (
        "h1", "h2", "h3", "h4", "h5", "ha", "hb", "hc",
        "hn", "ho", "hp", "hs", "hw", "hx",
    ),
    "F": (
        "f",
    ),
    "Cl": (
        "cl",
    ),
    "Br": (
        "br",
    ),
    "I": (
        "i",
    ),
    "N": (
        "n", "n+", "n1", "n2", "n3", "n4", "n5", "n6",
        "n7", "n8", "n9", "na", "nb", "nc", "nd", "ne",
        "nf", "nh", "ni", "nj", "nk", "nl", "nm", "nn",
        "no", "np", "nq", "ns", "nt", "nu", "nv", "nx",
        "ny", "nz",
    ),
    "O": (
        "o", "oh", "op", "oq", "os", "ow",
    ),
    "P": (
        "p2", "p3", "p4", "p5", "pb", "pc", "pd", "pe",
        "pf", "px", "py",
    ),
    "S": (
        "s", "s2", "s4", "s6", "sh", "sp", "sq", "ss",
        "sx", "sy",
    ),
}
_GAFF_ELEMENT_BY_TYPE = {
    atom_type: element
    for element, atom_types in _GAFF_TYPES_BY_ELEMENT.items()
    for atom_type in atom_types
}


def _element_from_mol2(atom_name: str, atom_type: str) -> str:
    token = atom_type.split(".", 1)[0].strip()
    if token in atomic_numbers and atomic_numbers[token] > 0:
        return token
    gaff_element = _GAFF_ELEMENT_BY_TYPE.get(atom_type.strip())
    if gaff_element is not None:
        return gaff_element
    raise ValueError(
        "Cannot determine an element from the explicit Tripos or GAFF/GAFF2 "
        f"MOL2 atom type {atom_type!r} (atom name {atom_name!r})."
    )


def _sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for raw in lines:
        line = raw.rstrip("\n")
        if line.upper().startswith("@<TRIPOS>"):
            current = line[9:].strip().upper()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return sections


def _connected(natoms: int, bonds: list[list[object]]) -> bool:
    if natoms <= 1:
        return True
    adjacency = [[] for _ in range(natoms)]
    for i, j, _ in bonds:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    visited = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == natoms


class MOL2Reader:
    """Read one connected molecule from a Tripos MOL2 file into ASE Atoms."""

    def __new__(
        cls,
        file_path: str,
        charge: Optional[int] = None,
        mult: Optional[int] = None,
        base_dir: Optional[str] = None,
        *,
        validate_charge: bool = False,
    ) -> Atoms:
        path = Path(file_path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MOL2 file not found: {path}")

        sections = _sections(path.read_text(encoding="utf-8", errors="replace").splitlines())
        molecule = [line.strip() for line in sections.get("MOLECULE", []) if line.strip()]
        atom_lines = [line for line in sections.get("ATOM", []) if line.strip()]
        bond_lines = [line for line in sections.get("BOND", []) if line.strip()]
        if len(molecule) < 3 or not atom_lines:
            raise ValueError(f"Invalid MOL2 file {path}: MOLECULE and ATOM sections are required.")

        name = molecule[0]
        try:
            counts = molecule[1].split()
            declared_atoms = int(counts[0])
            declared_bonds = int(counts[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid MOL2 counts line in {path}: {molecule[1]!r}") from exc
        molecule_type = molecule[2]
        charge_type = molecule[3] if len(molecule) >= 4 else "NO_CHARGES"

        symbols: list[str] = []
        positions: list[list[float]] = []
        names: list[str] = []
        atom_types: list[str] = []
        subst_ids: list[int] = []
        subst_names: list[str] = []
        charges: list[float] = []
        atom_id_to_index: dict[int, int] = {}
        charges_present = True

        for index, line in enumerate(atom_lines):
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Invalid MOL2 ATOM record in {path}: {line!r}")
            try:
                atom_id = int(fields[0])
                xyz = [float(fields[2]), float(fields[3]), float(fields[4])]
            except ValueError as exc:
                raise ValueError(f"Invalid MOL2 ATOM record in {path}: {line!r}") from exc
            if not np.all(np.isfinite(xyz)):
                raise ValueError(f"MOL2 coordinates must be finite in {path}: {line!r}")
            if atom_id in atom_id_to_index:
                raise ValueError(f"Duplicate MOL2 atom id {atom_id} in {path}.")
            atom_id_to_index[atom_id] = index
            atom_name, atom_type = fields[1], fields[5]
            symbols.append(_element_from_mol2(atom_name, atom_type))
            positions.append(xyz)
            names.append(atom_name)
            atom_types.append(atom_type)
            subst_ids.append(int(fields[6]) if len(fields) >= 7 else 1)
            subst_names.append(fields[7] if len(fields) >= 8 else "MOL")
            if len(fields) >= 9:
                try:
                    charges.append(float(fields[8]))
                except ValueError as exc:
                    raise ValueError(f"Invalid MOL2 partial charge in {path}: {line!r}") from exc
            else:
                charges_present = False
                charges.append(0.0)

        if not np.all(np.isfinite(charges)):
            raise ValueError(f"MOL2 partial charges must be finite in {path}.")

        bonds: list[list[object]] = []
        for line in bond_lines:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Invalid MOL2 BOND record in {path}: {line!r}")
            try:
                i = atom_id_to_index[int(fields[1])]
                j = atom_id_to_index[int(fields[2])]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid MOL2 bond atom id in {path}: {line!r}") from exc
            bonds.append([i, j, fields[3]])

        if declared_atoms != len(symbols) or declared_bonds != len(bonds):
            raise ValueError(
                f"MOL2 counts mismatch in {path}: declared {declared_atoms} atoms/{declared_bonds} "
                f"bonds, read {len(symbols)} atoms/{len(bonds)} bonds."
            )
        if not _connected(len(symbols), bonds):
            raise ValueError("Implicit solvation currently requires one connected molecule per MOL2 file.")
        if mult is not None and mult < 1:
            raise ValueError(f"Invalid multiplicity: {mult}. Must be >= 1.")
        if validate_charge:
            if not charges_present or charge_type.upper() == "NO_CHARGES":
                raise ValueError("#charge(source=mol2) requires a MOL2 file with per-atom partial charges.")
            if charge is None:
                raise ValueError(
                    "#charge(source=mol2) requires an explicit molecular charge/multiplicity line before MOL2."
                )
            actual = float(np.sum(charges))
            if abs(actual - float(charge)) > MOL2_CHARGE_TOL:
                raise ValueError(
                    "MOL2 partial-charge sum does not match the declared molecular charge: "
                    f"sum={actual:.8f}, declared={charge}. Charges are never silently renormalized."
                )

        atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=np.float64))
        if charges_present:
            atoms.set_initial_charges(np.asarray(charges, dtype=np.float64))
        if charge is not None:
            atoms.info["charge"] = int(charge)
        if mult is not None:
            atoms.info["mult"] = int(mult)
            atoms.info["spin"] = (int(mult) - 1) / 2
        atoms.info["mol2"] = {
            "path": os.fspath(path),
            "name": name,
            "molecule_type": molecule_type,
            "charge_type": charge_type,
            "atom_names": names,
            "atom_types": atom_types,
            "subst_ids": subst_ids,
            "subst_names": subst_names,
            "bonds": bonds,
            "charges_present": charges_present,
        }
        return atoms
