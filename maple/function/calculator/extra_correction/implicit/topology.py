from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ase.data import atomic_numbers, chemical_symbols


def _integral_charge(value: Any, *, field: str) -> int:
    """Normalize scalar/quantity-like formal charges without truncation."""
    raw = value
    for attribute in ("m", "magnitude"):
        candidate = getattr(raw, attribute, None)
        if candidate is not None:
            raw = candidate
            break
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite integer, got {value!r}.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be a finite integer, got {value!r}.")
    return int(numeric)


@dataclass(frozen=True)
class CanonicalTopology:
    """Canonicalized per-source topology used by solvent providers.

    Instances are intentionally immutable and strictly validated. Any mismatch in
    atom count/order/elements/bonds/mapping raises immediately (fail-closed).
    """

    source: str
    symbols: tuple[str, ...]
    atom_names: tuple[str, ...]
    atom_types: tuple[str, ...]
    mappings: tuple[int | None, ...]
    formal_charges: tuple[int | None, ...]
    total_formal_charge: int | None
    bonds: tuple[tuple[int, int, float], ...]
    fragments: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def nfragments(self) -> int:
        return max(self.fragments) + 1 if self.fragments else 0

    @property
    def formal_charge_sum(self) -> int | None:
        if any(charge is None for charge in self.formal_charges):
            return None
        return sum(int(charge) for charge in self.formal_charges)

    def validate_atoms(self, atoms) -> None:
        """Reject atom-count or atom-order drift between topology and coordinates."""
        symbols = tuple(atoms.get_chemical_symbols())
        if len(symbols) != self.natoms:
            raise ValueError(
                f"{self.source}: topology atom count {self.natoms} does not match "
                f"coordinate atom count {len(symbols)}."
            )
        if symbols != self.symbols:
            raise ValueError(
                f"{self.source}: topology/coordinate element order mismatch; "
                "MAPLE will not guess an atom mapping."
            )
        declared = getattr(atoms, "info", {}).get("charge")
        if declared is not None and self.total_formal_charge is not None:
            declared_charge = _integral_charge(
                declared,
                field=f"{self.source} coordinate total charge",
            )
            if declared_charge != self.total_formal_charge:
                raise ValueError(
                    f"{self.source}: topology total formal charge "
                    f"{self.total_formal_charge} does not match coordinate "
                    f"total charge {declared_charge}."
                )

    def to_openmm_topology(self, *, residue_name: str = "MOL"):
        """Export the validated canonical topology without guessing atom mappings."""
        try:
            from openmm import app
        except Exception as exc:  # pragma: no cover - exercised by integration tests
            raise ImportError(
                "OpenMM is required to export an OpenMM topology. Install with "
                "`pip install 'maple[implicit-gb]'`."
            ) from exc

        topology = app.Topology()
        chain = topology.addChain("A")
        residue = topology.addResidue(residue_name, chain)
        openmm_atoms = []
        for symbol, name in zip(self.symbols, self.atom_names):
            element = app.Element.getBySymbol(symbol)
            openmm_atoms.append(topology.addAtom(name, element, residue))

        for i, j, _ in self.bonds:
            topology.addBond(openmm_atoms[i], openmm_atoms[j])
        return topology


class TopologyProvider:
    """Construct canonical topologies from multiple sources with strict validation."""

    @staticmethod
    def _validate_bond(bond: Sequence[Any], natoms: int) -> tuple[int, int, float]:
        if len(bond) < 3:
            raise ValueError(f"Invalid bond record: {bond!r}")
        i, j, raw_order = bond[0], bond[1], bond[2]
        try:
            i_i = int(i)
            j_j = int(j)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bond atom index in {bond!r}.") from exc
        if i_i < 0 or j_j < 0 or i_i >= natoms or j_j >= natoms:
            raise ValueError(f"Bond atom index out of range in {bond!r}.")
        if i_i == j_j:
            raise ValueError(f"Self-bond is invalid in {bond!r}.")

        try:
            order = float(raw_order)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bond order in {bond!r}.") from exc
        if order <= 0:
            raise ValueError(f"Bond order must be > 0 in {bond!r}.")
        return (i_i, j_j, order)

    @staticmethod
    def _validate_mappings(
        mappings: Iterable[Any], natoms: int
    ) -> tuple[int | None, ...]:
        normalized: list[int | None] = []
        used: set[int] = set()
        for item in mappings:
            if item is None:
                normalized.append(None)
                continue
            try:
                mapping = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid atom mapping value {item!r}.") from exc
            if mapping <= 0:
                raise ValueError(f"Atom mapping must be positive: {mapping!r}.")
            if mapping in used:
                raise ValueError(f"Duplicate mapping id detected: {mapping!r}.")
            used.add(mapping)
            normalized.append(mapping)
        if len(normalized) != natoms:
            raise ValueError(
                f"Mapping count {len(normalized)} does not match atom count {natoms}."
            )
        return tuple(normalized)

    @staticmethod
    def _validate_formal_charges(
        formal_charges: Sequence[Any] | None, natoms: int
    ) -> tuple[int | None, ...]:
        if formal_charges is None:
            return tuple([None] * natoms)

        if len(formal_charges) != natoms:
            raise ValueError(
                f"Formal charge count {len(formal_charges)} does not match atom count {natoms}."
            )

        normalized: list[int | None] = []
        for index, item in enumerate(formal_charges):
            if item is None:
                normalized.append(None)
                continue
            normalized.append(
                _integral_charge(item, field=f"formal charge for atom {index}")
            )
        return tuple(normalized)

    @staticmethod
    def _component_ids(
        natoms: int, bonds: Sequence[tuple[int, int, float]]
    ) -> tuple[int, ...]:
        if natoms == 0:
            return tuple()
        adjacency: list[list[int]] = [[] for _ in range(natoms)]
        for i, j, _ in bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        component = [-1] * natoms
        current = 0
        for start in range(natoms):
            if component[start] != -1:
                continue
            stack = [start]
            component[start] = current
            while stack:
                atom = stack.pop()
                for nbr in adjacency[atom]:
                    if component[nbr] == -1:
                        component[nbr] = current
                        stack.append(nbr)
            current += 1
        return tuple(component)

    @staticmethod
    def _validate_open_atoms(
        symbols: Sequence[str],
        atom_names: Sequence[str],
        atom_types: Sequence[str],
        bonds: Sequence[tuple[int, int, float]],
        mappings: Sequence[int | None],
        formal_charges: Sequence[int | None] | None = None,
        total_formal_charge: Any | None = None,
        *,
        source: str,
        require_single_fragment: bool,
        metadata: dict[str, Any] | None = None,
    ) -> CanonicalTopology:
        symbols_tuple = tuple(symbols)
        natoms = len(symbols_tuple)
        if natoms == 0:
            raise ValueError(f"{source}: topology must contain at least one atom.")

        if len(atom_names) != natoms:
            raise ValueError(
                f"{source}: atom_names length {len(atom_names)} must match atom count {natoms}."
            )
        if len(atom_types) != natoms:
            raise ValueError(
                f"{source}: atom_types length {len(atom_types)} must match atom count {natoms}."
            )

        bonds_tuple = tuple(
            TopologyProvider._validate_bond(edge, natoms) for edge in bonds
        )
        mapping_tuple = TopologyProvider._validate_mappings(mappings, natoms)
        formal_charge_tuple = TopologyProvider._validate_formal_charges(
            formal_charges, natoms
        )
        normalized_total_charge = (
            None
            if total_formal_charge is None
            else _integral_charge(
                total_formal_charge, field=f"{source} total formal charge"
            )
        )
        known_formal_sum = (
            None
            if any(charge is None for charge in formal_charge_tuple)
            else sum(int(charge) for charge in formal_charge_tuple)
        )
        if normalized_total_charge is None:
            normalized_total_charge = known_formal_sum
        elif (
            known_formal_sum is not None and known_formal_sum != normalized_total_charge
        ):
            raise ValueError(
                f"{source}: per-atom formal charges sum to {known_formal_sum}, "
                f"not declared total charge {normalized_total_charge}."
            )
        fragments = TopologyProvider._component_ids(natoms, bonds_tuple)

        if require_single_fragment and len({*fragments}) > 1:
            raise ValueError(
                f"{source}: topology contains {len({*fragments})} disconnected fragments; "
                "split fragments or set require_single_fragment=False."
            )

        return CanonicalTopology(
            source=source,
            symbols=symbols_tuple,
            atom_names=tuple(atom_names),
            atom_types=tuple(atom_types),
            mappings=mapping_tuple,
            formal_charges=formal_charge_tuple,
            total_formal_charge=normalized_total_charge,
            bonds=bonds_tuple,
            fragments=fragments,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _element_from_int(number: Any) -> str:
        try:
            n = int(number)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported atomic number {number!r} in topology data."
            ) from exc
        if n < 1 or n >= len(chemical_symbols):
            raise ValueError(f"Unsupported atomic number {n} in topology data.")
        symbol = chemical_symbols[n]
        if not symbol:
            raise ValueError(f"Unsupported atomic number {n} in topology data.")
        return symbol

    @classmethod
    def from_mol2_atoms(
        cls, atoms, *, require_single_fragment: bool = True
    ) -> CanonicalTopology:
        metadata = atoms.info.get("mol2") if getattr(atoms, "info", None) else None
        if not isinstance(metadata, dict):
            raise ValueError("MOL2 topology requires atoms.info['mol2'] metadata.")
        mol2_symbols = list(atoms.get_chemical_symbols())
        atom_names = metadata.get("atom_names")
        if atom_names is None:
            atom_names = mol2_symbols
        atom_types = metadata.get("atom_types")
        if atom_types is None:
            atom_types = tuple("") * len(mol2_symbols)
        bonds = metadata.get("bonds", ())
        if not isinstance(bonds, list):
            raise ValueError("MOL2 metadata 'bonds' must be a list of [i, j, order].")

        return cls._validate_open_atoms(
            mol2_symbols,
            atom_names,
            atom_types,
            bonds,
            [None] * len(mol2_symbols),
            formal_charges=None,
            total_formal_charge=atoms.info.get("charge"),
            source="MOL2",
            require_single_fragment=require_single_fragment,
            metadata=metadata,
        )

    @classmethod
    def from_openmm_topology(
        cls, topology, *, require_single_fragment: bool = True
    ) -> CanonicalTopology:
        try:
            app = __import__("openmm.app", fromlist=["Topology"])
        except ModuleNotFoundError as exc:
            raise ImportError(
                "OpenMM topology support is optional; install with `pip install 'maple[implicit-gb]'`."
            ) from exc

        if not isinstance(topology, app.Topology):
            raise ValueError(
                "from_openmm_topology expects an openmm.app.Topology object."
            )

        topology_atoms = list(topology.atoms())
        if not topology_atoms:
            raise ValueError("OpenMM topology has no atoms.")

        symbols: list[str] = []
        names: list[str] = []
        mappings: list[int | None] = []
        formal_charges: list[int | None] = []
        atom_types: list[str] = [""] * len(topology_atoms)

        for idx, atom in enumerate(topology_atoms):
            if atom.element is None:
                raise ValueError(f"OpenMM atom {idx} does not expose element symbol.")
            symbol = atom.element.symbol
            if symbol not in atomic_numbers:
                raise ValueError(
                    f"OpenMM atom {idx} has unsupported element symbol {symbol!r}."
                )
            symbols.append(symbol)
            names.append(str(atom.name))
            atom_id = getattr(atom, "id", None)
            mappings.append(
                atom_id if isinstance(atom_id, int) and atom_id > 0 else None
            )
            raw_charge = getattr(atom, "formal_charge", None)
            formal_charges.append(raw_charge)

        bonds: list[tuple[int, int, float]] = []
        for bond in topology.bonds():
            a, b = bond.atom1, bond.atom2
            if a not in topology_atoms or b not in topology_atoms:
                raise ValueError("OpenMM bond references atom outside topology.")
            i, j = topology_atoms.index(a), topology_atoms.index(b)
            raw_order = getattr(bond, "order", 1)
            if raw_order is None:
                raw_order = 1
            try:
                order = float(raw_order)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot parse OpenMM bond order for bond {i}-{j}."
                ) from exc
            bonds.append((i, j, order))

        return cls._validate_open_atoms(
            symbols,
            names,
            atom_types,
            bonds,
            mappings,
            formal_charges=formal_charges,
            total_formal_charge=None,
            source="OpenMM-Topology",
            require_single_fragment=require_single_fragment,
            metadata={"source": "openmm.app.Topology"},
        )

    @classmethod
    def from_openff_molecule(
        cls, molecule, *, require_single_fragment: bool = True
    ) -> CanonicalTopology:
        if not hasattr(molecule, "atoms"):
            raise ValueError("OpenFF molecules should provide an atoms collection.")

        mol_atoms = list(molecule.atoms)
        if not mol_atoms:
            raise ValueError("OpenFF molecule has no atoms.")

        symbols: list[str] = []
        names: list[str] = []
        mappings: list[int | None] = []
        formal_charges: list[int | None] = []
        atom_types: list[str] = []
        properties = getattr(molecule, "properties", {})
        molecule_atom_map = (
            properties.get("atom_map", {}) if isinstance(properties, Mapping) else {}
        )
        if not isinstance(molecule_atom_map, Mapping):
            raise ValueError(
                "OpenFF molecule properties['atom_map'] must be a mapping."
            )

        for index, atom in enumerate(mol_atoms):
            symbol = getattr(atom, "element_symbol", None) or getattr(
                atom, "symbol", None
            )
            if symbol is None:
                atomic_number = getattr(atom, "atomic_number", None)
                if atomic_number is None:
                    raise ValueError(
                        f"OpenFF atom {index} missing atomic symbol/number."
                    )
                symbol = cls._element_from_int(atomic_number)
            atom_name = getattr(atom, "atom_name", None)
            if atom_name is None:
                atom_name = getattr(atom, "name", None)
            names.append(str(atom_name or index + 1))
            symbols.append(symbol)
            atom_types.append(str(getattr(atom, "atom_type", "")))
            mapping = getattr(atom, "atom_map", None)
            if mapping is None:
                mapping = molecule_atom_map.get(index)
            mappings.append(mapping)
            raw_formal_charge = getattr(atom, "formal_charge", None)
            formal_charges.append(raw_formal_charge)

        raw_bonds = getattr(molecule, "bonds", None)
        if raw_bonds is None:
            raw_bonds = []

        bonds: list[tuple[Any, Any, Any]] = []
        for bond in raw_bonds:
            i = getattr(bond, "atom1_index", None)
            j = getattr(bond, "atom2_index", None)
            if i is None or j is None:
                a1 = getattr(bond, "atom1", None)
                a2 = getattr(bond, "atom2", None)
                if (
                    a1 is not None
                    and a2 is not None
                    and hasattr(a1, "molecule_atom_index")
                ):
                    i = a1.molecule_atom_index
                    j = a2.molecule_atom_index
            if i is None or j is None:
                raise ValueError(
                    f"OpenFF bond entry {bond!r} does not expose atom indices."
                )
            order = (
                getattr(bond, "bond_order", None)
                or getattr(bond, "order", None)
                or getattr(bond, "bond_order_fraction", None)
            )
            if order is None:
                raise ValueError(f"OpenFF bond {i}-{j} does not expose bond order.")
            bonds.append((i, j, order))

        return cls._validate_open_atoms(
            symbols,
            names,
            atom_types,
            bonds,
            mappings,
            formal_charges=formal_charges,
            total_formal_charge=None,
            source="OpenFF",
            require_single_fragment=require_single_fragment,
            metadata={"source": "openff toolkit molecule"},
        )

    @classmethod
    def from_sdf_file(
        cls,
        file_path: str | Path,
        *,
        require_single_fragment: bool = True,
    ) -> CanonicalTopology:
        if importlib.util.find_spec("rdkit") is None:
            raise ImportError(
                "SDF/MOL support requires optional RDKit dependency; install rdkit-pypi."
            )
        from rdkit import Chem

        candidate = Path(file_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"SDF/MOL file not found: {candidate}")

        mol = Chem.MolFromMolFile(str(candidate), removeHs=False)
        if mol is None:
            raise ValueError(f"Unable to parse SDF/MOL file: {candidate}")

        symbols: list[str] = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_names: list[str] = [
            atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else str(i + 1)
            for i, atom in enumerate(mol.GetAtoms())
        ]
        atom_types: list[str] = [atom.GetSymbol() for atom in mol.GetAtoms()]
        formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        mapping_props = [
            atom.GetAtomMapNum() if atom.HasProp("molAtomMapNumber") else None
            for atom in mol.GetAtoms()
        ]
        raw_bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
            for bond in mol.GetBonds()
        ]

        return cls._validate_open_atoms(
            symbols,
            atom_names,
            atom_types,
            raw_bonds,
            mapping_props,
            formal_charges,
            total_formal_charge=sum(formal_charges),
            source="SDF",
            require_single_fragment=require_single_fragment,
            metadata={"source": str(candidate), "rdkit_name": str(type(mol).__name__)},
        )


def canonicalize_topology(topology_source: Any) -> CanonicalTopology:
    """Auto-dispatch to canonical providers; raise for unsupported source objects."""
    if isinstance(topology_source, CanonicalTopology):
        return topology_source

    info = getattr(topology_source, "info", None)
    if (
        hasattr(topology_source, "get_chemical_symbols")
        and isinstance(info, dict)
        and isinstance(info.get("mol2"), dict)
    ):
        return TopologyProvider.from_mol2_atoms(topology_source)

    module = type(topology_source).__module__
    if module.startswith("openff"):
        return TopologyProvider.from_openff_molecule(topology_source)
    if module.startswith("openmm") and hasattr(topology_source, "atoms"):
        return TopologyProvider.from_openmm_topology(topology_source)

    if isinstance(topology_source, (str, Path)):
        path = Path(topology_source)
        suffix = path.suffix.lower()
        if suffix in {".sdf", ".mol"}:
            return TopologyProvider.from_sdf_file(path)
        raise ValueError(f"Unsupported topology file type: {suffix!r}.")

    raise TypeError(
        "Topology source is unsupported; pass MOL2 atoms.info['mol2'], an OpenMM topology, "
        "an OpenFF Molecule-like object, or a SDF/MOL path."
    )
