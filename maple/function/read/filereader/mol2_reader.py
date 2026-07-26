"""Tripos MOL2 reader for implicit-solvent inputs.

The reader deliberately consumes only information explicitly present in MOL2:
coordinates, atom types, bonds, substructure labels, and partial charges.  It
does not infer bond orders, protonation states, or missing hydrogens from XYZ
geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


MOL2_CHARGE_TOL = 1.0e-4
MOL2_ATOM_ID_ARRAY = "_maple_mol2_atom_id"
MOL2_IDENTITY_SHA256_KEY = "identity_sha256"
_MOL2_IDENTITY_FIELDS = (
    "atom_ids",
    "atom_names",
    "atom_types",
    "subst_ids",
    "subst_names",
    "bonds",
    "component_ids",
    "component_count",
)


def mol2_identity_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash the atom/type/substructure/bond identity frozen by MOL2Reader."""
    payload = {
        field: metadata.get(field)
        for field in _MOL2_IDENTITY_FIELDS
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


_GAFF_TYPES_BY_ELEMENT = {
    # Exact GAFF/GAFF2 atom-type tables, rather than a first-character guess.
    # The ambiguous lowercase labels ``ca`` and ``na`` therefore remain
    # aromatic carbon and nitrogen, while case-correct ``Ca`` and ``Na`` are
    # explicit element tokens.
    "C": frozenset(
        {
            "c",
            "c1",
            "c2",
            "c3",
            "c5",
            "c6",
            "ca",
            "cc",
            "cd",
            "ce",
            "cf",
            "cg",
            "ch",
            "cp",
            "cq",
            "cs",
            "cu",
            "cv",
            "cx",
            "cy",
            "cz",
        }
    ),
    "H": frozenset(
        {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "ha",
            "hb",
            "hc",
            "hn",
            "ho",
            "hp",
            "hs",
            "hw",
            "hx",
        }
    ),
    "F": frozenset({"f"}),
    "Cl": frozenset({"cl"}),
    "Br": frozenset({"br"}),
    "I": frozenset({"i"}),
    "N": frozenset(
        {
            "n",
            "n+",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "na",
            "nb",
            "nc",
            "nd",
            "ne",
            "nf",
            "nh",
            "ni",
            "nj",
            "nk",
            "nl",
            "nm",
            "nn",
            "no",
            "np",
            "nq",
            "ns",
            "nt",
            "nu",
            "nv",
            "nx",
            "ny",
            "nz",
        }
    ),
    "O": frozenset({"o", "oh", "op", "oq", "os", "ow"}),
    "P": frozenset({"p2", "p3", "p4", "p5", "pb", "pc", "pd", "pe", "pf", "px", "py"}),
    "S": frozenset({"s", "s2", "s4", "s6", "sh", "sp", "sq", "ss", "sx", "sy"}),
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


def _connected_components(
    natoms: int, bonds: list[list[int | str]]
) -> tuple[list[int], int]:
    adjacency = [[] for _ in range(natoms)]
    for i, j, _ in bonds:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    component_ids = [-1] * natoms
    component_count = 0
    for start in range(natoms):
        if component_ids[start] >= 0:
            continue
        component_ids[start] = component_count
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if component_ids[neighbor] < 0:
                    component_ids[neighbor] = component_count
                    stack.append(neighbor)
        component_count += 1
    return component_ids, component_count


class MOL2Reader:
    """Read a Tripos MOL2 topology into ASE Atoms.

    Disconnected components remain fail-closed unless a caller explicitly
    enables them for a prebuilt explicit-inner/implicit-outer cluster.
    """

    def __new__(
        cls,
        file_path: str,
        charge: Optional[int] = None,
        mult: Optional[int] = None,
        base_dir: Optional[str] = None,
        *,
        validate_charge: bool = False,
        allow_disconnected: bool = False,
    ) -> Atoms:
        path = Path(file_path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MOL2 file not found: {path}")

        source_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        source_text = source_bytes.decode("utf-8", errors="replace")
        sections = _sections(source_text.splitlines())
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
        atom_ids: list[int] = []
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
            if atom_id in atom_id_to_index:
                raise ValueError(f"Duplicate MOL2 atom id {atom_id} in {path}.")
            atom_id_to_index[atom_id] = index
            atom_name, atom_type = fields[1], fields[5]
            if not np.isfinite(xyz).all():
                raise ValueError(f"Non-finite MOL2 coordinate in {path}: {line!r}")
            symbols.append(_element_from_mol2(atom_name, atom_type))
            positions.append(xyz)
            names.append(atom_name)
            atom_types.append(atom_type)
            atom_ids.append(atom_id)
            subst_ids.append(int(fields[6]) if len(fields) >= 7 else 1)
            subst_names.append(fields[7] if len(fields) >= 8 else "MOL")
            if len(fields) >= 9:
                try:
                    partial_charge = float(fields[8])
                except ValueError as exc:
                    raise ValueError(f"Invalid MOL2 partial charge in {path}: {line!r}") from exc
                if not np.isfinite(partial_charge):
                    raise ValueError(f"Non-finite MOL2 partial charge in {path}: {line!r}")
                charges.append(partial_charge)
            else:
                charges_present = False
                charges.append(0.0)

        bonds: list[list[int | str]] = []
        bond_ids: set[int] = set()
        bonded_pairs: set[tuple[int, int]] = set()
        for line in bond_lines:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Invalid MOL2 BOND record in {path}: {line!r}")
            try:
                bond_id = int(fields[0])
                i = atom_id_to_index[int(fields[1])]
                j = atom_id_to_index[int(fields[2])]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid MOL2 bond atom id in {path}: {line!r}") from exc
            if bond_id in bond_ids:
                raise ValueError(f"Duplicate MOL2 bond id {bond_id} in {path}.")
            bond_ids.add(bond_id)
            if i == j:
                raise ValueError(f"Self-referential MOL2 bond in {path}: {line!r}")
            bonded_pair = (min(i, j), max(i, j))
            if bonded_pair in bonded_pairs:
                raise ValueError(
                    f"Duplicate MOL2 bond between one atom pair in {path}: {line!r}"
                )
            bonded_pairs.add(bonded_pair)
            bonds.append([i, j, fields[3]])

        if declared_atoms != len(symbols) or declared_bonds != len(bonds):
            raise ValueError(
                f"MOL2 counts mismatch in {path}: declared {declared_atoms} atoms/{declared_bonds} "
                f"bonds, read {len(symbols)} atoms/{len(bonds)} bonds."
            )
        component_ids, component_count = _connected_components(len(symbols), bonds)
        if component_count > 1 and not allow_disconnected:
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
        atoms.new_array(MOL2_ATOM_ID_ARRAY, np.asarray(atom_ids, dtype=np.int64))
        if charges_present:
            atoms.set_initial_charges(np.asarray(charges, dtype=np.float64))
        if charge is not None:
            atoms.info["charge"] = int(charge)
        if mult is not None:
            atoms.info["mult"] = int(mult)
            atoms.info["spin"] = (int(mult) - 1) / 2
        metadata = {
            "path": os.fspath(path),
            "source_sha256": source_sha256,
            "name": name,
            "molecule_type": molecule_type,
            "charge_type": charge_type,
            "atom_ids": atom_ids,
            "atom_names": names,
            "atom_types": atom_types,
            "subst_ids": subst_ids,
            "subst_names": subst_names,
            "bonds": bonds,
            "charges_present": charges_present,
            "component_ids": component_ids,
            "component_count": component_count,
            "component_charge_sums_e": (
                np.bincount(
                    component_ids,
                    weights=charges,
                    minlength=component_count,
                ).astype(float).tolist()
                if charges_present
                else None
            ),
        }
        metadata[MOL2_IDENTITY_SHA256_KEY] = mol2_identity_sha256(metadata)
        atoms.info["mol2"] = metadata
        return atoms
