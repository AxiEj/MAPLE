"""Immutable MOL2 topology used by the independent OBC-II preparation path.

Only identity-bearing facts already validated by :class:`MOL2Reader` are
consumed.  No geometry-based bond inference or OpenMM topology object is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    mol2_identity_sha256,
)


def _canonical_json_bytes(value: Any) -> bytes:
    """Match the canonical JSON convention used by the sealed OBC oracle."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class OBC2Atom:
    """One atom in the frozen portable topology."""

    index: int
    atomic_number: int
    symbol: str
    mol2_atom_id: int
    name: str
    mol2_type: str
    component_id: int
    bonded_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OBC2Bond:
    """One MOL2 bond; OBC-II uses connectivity but not the bond-type token."""

    atom1: int
    atom2: int
    mol2_type: str

    @property
    def indices(self) -> tuple[int, int]:
        return self.atom1, self.atom2


@dataclass(frozen=True, slots=True)
class OBC2Topology:
    """Canonical, position-independent OBC-II topology snapshot.

    The portable mbondi2 domain is intentionally stricter than the upstream
    helper's unchecked ``bondeds[0]`` access: every hydrogen (and therefore any
    ASE-representable deuterium, which has atomic number 1) must have exactly one
    bonded neighbor.
    """

    atoms: tuple[OBC2Atom, ...]
    bonds: tuple[OBC2Bond, ...]
    component_count: int
    mol2_identity_sha256: str
    identity_sha256: str

    @classmethod
    def from_atoms(cls, atoms) -> "OBC2Topology":
        metadata = atoms.info.get("mol2")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Independent OBC-II preparation requires validated MOL2 topology metadata."
            )
        expected_identity = mol2_identity_sha256(metadata)
        observed_identity = metadata.get(MOL2_IDENTITY_SHA256_KEY)
        if observed_identity != expected_identity:
            raise ValueError(
                "MOL2 topology identity changed after MOL2Reader; refusing OBC-II preparation."
            )

        count = len(atoms)
        if count == 0:
            raise ValueError("OBC-II topology requires at least one atom.")
        atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
        atom_ids = _integer_tuple(metadata.get("atom_ids"), count, "atom ids")
        atom_names = _string_tuple(metadata.get("atom_names"), count, "atom names")
        atom_types = _string_tuple(metadata.get("atom_types"), count, "atom types")
        component_ids = _integer_tuple(
            metadata.get("component_ids"), count, "component ids"
        )
        if any(number <= 0 for number in atomic_numbers):
            raise ValueError("OBC-II atomic numbers must be positive.")
        if len(set(atom_ids)) != count:
            raise ValueError("MOL2 atom ids must be unique.")

        identity_array = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
        if identity_array is None or tuple(int(value) for value in identity_array) != atom_ids:
            raise ValueError(
                "Current ASE atom order does not match the frozen MOL2 atom identity."
            )

        unique_components = sorted(set(component_ids))
        if unique_components != list(range(len(unique_components))):
            raise ValueError("MOL2 component ids must be dense and zero-based.")
        component_count = metadata.get("component_count")
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, (int, np.integer))
            or int(component_count) != len(unique_components)
        ):
            raise ValueError("MOL2 component count does not match component ids.")

        raw_bonds = metadata.get("bonds")
        if not isinstance(raw_bonds, list | tuple):
            raise ValueError("MOL2 bonds must be an explicit sequence.")
        adjacency: list[list[int]] = [[] for _ in range(count)]
        seen_pairs: set[tuple[int, int]] = set()
        bond_records: list[OBC2Bond] = []
        for record in raw_bonds:
            if not isinstance(record, list | tuple) or len(record) != 3:
                raise ValueError("Each MOL2 bond must contain two indices and a bond type.")
            left, right, bond_type = record
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, (int, np.integer))
                or not isinstance(right, (int, np.integer))
            ):
                raise ValueError("MOL2 bond indices must be integers.")
            left, right = int(left), int(right)
            if not (0 <= left < count and 0 <= right < count):
                raise ValueError("MOL2 bond index is out of range.")
            if left == right:
                raise ValueError("MOL2 topology contains a self-referential bond.")
            pair = (min(left, right), max(left, right))
            if pair in seen_pairs:
                raise ValueError("MOL2 topology contains a duplicate bond.")
            if component_ids[left] != component_ids[right]:
                raise ValueError("MOL2 bond crosses declared component identities.")
            seen_pairs.add(pair)
            adjacency[left].append(right)
            adjacency[right].append(left)
            bond_records.append(OBC2Bond(left, right, str(bond_type)))

        _validate_component_connectivity(component_ids, adjacency)
        for index, number in enumerate(atomic_numbers):
            if number == 1 and len(adjacency[index]) != 1:
                raise ValueError(
                    "Portable mbondi2 hydrogen connectivity must contain exactly one "
                    f"unambiguous bonded neighbor; atom {index} has {len(adjacency[index])}."
                )

        atom_records = tuple(
            OBC2Atom(
                index=index,
                atomic_number=atomic_numbers[index],
                symbol=symbols[index],
                mol2_atom_id=atom_ids[index],
                name=atom_names[index],
                mol2_type=atom_types[index],
                component_id=component_ids[index],
                bonded_indices=tuple(sorted(adjacency[index])),
            )
            for index in range(count)
        )
        payload = {
            "atomic_numbers": list(atomic_numbers),
            "symbols": list(symbols),
            "atom_ids": list(atom_ids),
            "atom_names": list(atom_names),
            "atom_types": list(atom_types),
            "bonds": [
                [bond.atom1, bond.atom2, bond.mol2_type] for bond in bond_records
            ],
            "component_ids": list(component_ids),
            "component_count": int(component_count),
            "mol2_identity_sha256": str(observed_identity),
        }
        identity_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return cls(
            atoms=atom_records,
            bonds=tuple(bond_records),
            component_count=int(component_count),
            mol2_identity_sha256=str(observed_identity),
            identity_sha256=identity_sha256,
        )

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return tuple(atom.atomic_number for atom in self.atoms)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(atom.symbol for atom in self.atoms)

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return tuple(atom.mol2_atom_id for atom in self.atoms)

    @property
    def atom_names(self) -> tuple[str, ...]:
        return tuple(atom.name for atom in self.atoms)

    @property
    def atom_types(self) -> tuple[str, ...]:
        return tuple(atom.mol2_type for atom in self.atoms)

    @property
    def component_ids(self) -> tuple[int, ...]:
        return tuple(atom.component_id for atom in self.atoms)

    @property
    def bonded_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(atom.bonded_indices for atom in self.atoms)

    @property
    def bond_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(bond.indices for bond in self.bonds)


def _integer_tuple(values, count: int, label: str) -> tuple[int, ...]:
    if not isinstance(values, list | tuple) or len(values) != count:
        raise ValueError(f"MOL2 {label} do not match the atom count.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"MOL2 {label} must contain integers.")
    return tuple(int(value) for value in values)


def _string_tuple(values, count: int, label: str) -> tuple[str, ...]:
    if not isinstance(values, list | tuple) or len(values) != count:
        raise ValueError(f"MOL2 {label} do not match the atom count.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"MOL2 {label} must contain non-empty strings.")
    return tuple(values)


def _validate_component_connectivity(
    component_ids: tuple[int, ...], adjacency: list[list[int]]
) -> None:
    """Reject stale component labels rather than reinterpreting connectivity."""
    for component in sorted(set(component_ids)):
        members = {i for i, value in enumerate(component_ids) if value == component}
        pending = [min(members)]
        visited: set[int] = set()
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            pending.extend(neighbor for neighbor in adjacency[index] if neighbor in members)
        if visited != members:
            raise ValueError(
                f"MOL2 component {component} is disconnected from its declared membership."
            )

