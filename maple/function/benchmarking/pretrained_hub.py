"""Benchmark identity, leakage, metrics, and artifact contracts for Route 4.

The module deliberately does not download or redistribute benchmark data.  It
binds local records to a complete protocol identity, enforces conservative
training-overlap labels, and writes reproducible result artifacts.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import CodeType, MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MINIMUM_ACCURACY_FUNCTIONAL_GROUPS = 10
MINIMUM_ACCURACY_RECORDS = 10
MINIMUM_MATCHED_RUNTIME_REPEATS = 3
MATCHED_PERFORMANCE_PRECISION_POLICIES = frozenset({"float32", "float64"})
FORMAL_ACCURACY_WORKER_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
END_TO_END_RUNTIME_SCOPE_COMPONENTS = frozenset(
    {
        "input_preparation",
        "geometry_or_sampling",
        "prediction_or_qm",
        "analysis",
        "artifact_write",
    }
)
FORMAL_FUNCTIONAL_GROUP_TAXONOMY_ID = "maple-rdkit-structural-functional-groups"
FORMAL_FUNCTIONAL_GROUP_TAXONOMY_FINGERPRINT = (
    "4928108edfe8529a9302484cca01d60860bb5bc53458db668c79994c91f65db2"
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_ids(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must contain one or more non-empty identifiers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate identifiers.")
    return normalized


def _normalized_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA256.")
    return normalized


def _canonical_json_value(value: Any, *, field: str) -> Any:
    """Return a deterministic JSON value or reject opaque runtime state."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite float.")
        return value
    if isinstance(value, Enum):
        return _canonical_json_value(value.value, field=field)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(
                    f"{field} contains mapping keys that collide after string "
                    "normalization."
                )
            normalized[normalized_key] = _canonical_json_value(
                item,
                field=f"{field}.{normalized_key}",
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported runtime state of type "
        f"{type(value).__name__!r}; formal adapters must expose only canonical "
        "JSON configuration and artifact-backed implementation state."
    )


@dataclass(frozen=True)
class FunctionalGroupDefinition:
    """One structure-query-backed functional-group family."""

    group_id: str
    label: str
    smarts: str
    source_entry: str

    def __post_init__(self) -> None:
        group_id = str(self.group_id).strip()
        if (
            not group_id
            or not group_id[0].isalpha()
            or any(
                not (character.islower() or character.isdigit() or character == "_")
                for character in group_id
            )
        ):
            raise ValueError(
                "Functional-group group_id must be a lowercase snake-case identifier."
            )
        for field in ("label", "smarts", "source_entry"):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"Functional-group {field} must be non-empty.")
            object.__setattr__(self, field, value)
        object.__setattr__(self, "group_id", group_id)


@dataclass(frozen=True)
class FunctionalGroupTaxonomyManifest:
    """Frozen, reviewable vocabulary that prevents arbitrary class labels."""

    taxonomy_id: str
    version: str
    source_citation: str
    assignment_policy: str
    definitions: tuple[FunctionalGroupDefinition, ...]
    primary_precedence: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "taxonomy_id",
            "version",
            "source_citation",
            "assignment_policy",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(
                    f"Functional-group taxonomy {field} must be non-empty."
                )
            object.__setattr__(self, field, value)
        definitions = tuple(self.definitions)
        if len(definitions) < MINIMUM_ACCURACY_FUNCTIONAL_GROUPS:
            raise ValueError(
                "A formal accuracy taxonomy must define at least 10 functional-group "
                "families."
            )
        if any(not isinstance(item, FunctionalGroupDefinition) for item in definitions):
            raise TypeError(
                "Functional-group taxonomy definitions must contain "
                "FunctionalGroupDefinition instances."
            )
        group_ids = tuple(item.group_id for item in definitions)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("Functional-group taxonomy contains duplicate group IDs.")
        primary_precedence = _normalized_ids(
            self.primary_precedence,
            field="functional-group primary precedence",
        )
        if set(primary_precedence) != set(group_ids):
            raise ValueError(
                "Functional-group primary_precedence must contain every defined "
                "group ID exactly once."
            )
        object.__setattr__(
            self,
            "definitions",
            tuple(sorted(definitions, key=lambda item: item.group_id)),
        )
        object.__setattr__(self, "primary_precedence", primary_precedence)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.definitions)

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "FunctionalGroupTaxonomyManifest":
        try:
            raw_definitions = payload["definitions"]
        except KeyError as exc:
            raise ValueError("Functional-group taxonomy lacks definitions.") from exc
        if not isinstance(raw_definitions, Sequence) or isinstance(
            raw_definitions, (str, bytes)
        ):
            raise TypeError("Functional-group taxonomy definitions must be a sequence.")
        definitions = tuple(
            FunctionalGroupDefinition(
                group_id=item["group_id"],
                label=item["label"],
                smarts=item["smarts"],
                source_entry=item["source_entry"],
            )
            for item in raw_definitions
        )
        try:
            return cls(
                taxonomy_id=payload["taxonomy_id"],
                version=payload["version"],
                source_citation=payload["source_citation"],
                assignment_policy=payload["assignment_policy"],
                definitions=definitions,
                primary_precedence=tuple(payload["primary_precedence"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Functional-group taxonomy lacks required field {exc.args[0]!r}."
            ) from exc


def load_functional_group_taxonomy(
    path: str | Path,
    *,
    expected_sha256: str,
) -> FunctionalGroupTaxonomyManifest:
    """Load an exact taxonomy artifact and fail closed on byte drift."""

    source = Path(path)
    data = source.read_bytes()
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != str(expected_sha256).strip().lower():
        raise ValueError(
            "Functional-group taxonomy SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}."
        )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("Functional-group taxonomy is not valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Functional-group taxonomy root must be an object.")
    return FunctionalGroupTaxonomyManifest.from_mapping(payload)


@dataclass(frozen=True)
class FunctionalGroupAssignment:
    """Predeclared, structure-bound functional-group labels for one record."""

    record_id: str
    structure_identifier: str
    primary_group: str
    matched_groups: tuple[str, ...]
    assignment_evidence: str

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        if not record_id:
            raise ValueError("Functional-group record_id must be non-empty.")
        structure_identifier = str(self.structure_identifier).strip()
        if not structure_identifier:
            raise ValueError("Functional-group structure_identifier must be non-empty.")
        primary_group = str(self.primary_group).strip()
        if not primary_group:
            raise ValueError("Functional-group primary_group must be non-empty.")
        groups = tuple(
            sorted(
                _normalized_ids(
                    self.matched_groups,
                    field="matched functional groups",
                )
            )
        )
        if primary_group not in groups:
            raise ValueError(
                "Functional-group primary_group must be one of matched_groups."
            )
        assignment_evidence = str(self.assignment_evidence).strip()
        if not assignment_evidence:
            raise ValueError("Functional-group assignment_evidence must be non-empty.")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "structure_identifier", structure_identifier)
        object.__setattr__(self, "primary_group", primary_group)
        object.__setattr__(self, "matched_groups", groups)
        object.__setattr__(self, "assignment_evidence", assignment_evidence)

    @property
    def groups(self) -> tuple[str, ...]:
        """Compatibility alias for all structurally matched group families."""

        return self.matched_groups


_MOLECULAR_INPUT_FORMATS = frozenset(
    {
        "smiles",
        "inchi",
        "sdf",
        "sdf_conformers",
        "mol",
        "mol2",
        "pdb",
        "headerless_xyz",
    }
)


def _v2000_declared_stereo_identity(
    mol_block: str,
    molecule: Any,
    *,
    chem: Any,
) -> str:
    """Capture only molfile-declared stereo, excluding 3D-inferred tetrahedra."""

    lines = mol_block.splitlines()
    counts_index = next(
        (index for index, line in enumerate(lines) if "V2000" in line),
        None,
    )
    if counts_index is None:
        if any("V3000" in line for line in lines):
            raise ValueError(
                "Formal SDF-conformer receipts currently require V2000 records "
                "so declared stereochemistry can be audited."
            )
        raise ValueError("Formal SDF-conformer record lacks a V2000 counts line.")
    try:
        counts = lines[counts_index].split()
        atom_count = int(counts[0])
        bond_count = int(counts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Formal SDF-conformer record has an invalid V2000 counts line."
        ) from exc
    if (
        atom_count != molecule.GetNumAtoms()
        or counts_index + 1 + atom_count + bond_count > len(lines)
    ):
        raise ValueError(
            "Formal SDF-conformer V2000 counts do not match the parsed molecule."
        )

    declared_tetrahedral_atoms: set[int] = set()
    atom_lines = lines[counts_index + 1 : counts_index + 1 + atom_count]
    for atom_index, line in enumerate(atom_lines):
        try:
            parity = int(line.split()[6])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "Formal SDF-conformer record has an invalid V2000 atom line."
            ) from exc
        if parity in {1, 2}:
            declared_tetrahedral_atoms.add(atom_index)

    unknown_double_bonds: list[tuple[int, int]] = []
    bond_lines = lines[
        counts_index + 1 + atom_count : counts_index + 1 + atom_count + bond_count
    ]
    for line in bond_lines:
        try:
            begin, end, bond_type, stereo_code = map(int, line.split()[:4])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Formal SDF-conformer record has an invalid V2000 bond line."
            ) from exc
        if not 1 <= begin <= atom_count or not 1 <= end <= atom_count:
            raise ValueError(
                "Formal SDF-conformer record has an out-of-range V2000 bond."
            )
        if bond_type == 1 and stereo_code in {1, 6}:
            declared_tetrahedral_atoms.add(begin - 1)
        if bond_type == 2 and stereo_code == 3:
            unknown_double_bonds.append((begin - 1, end - 1))

    declared = chem.Mol(molecule)
    for atom in declared.GetAtoms():
        if atom.GetIdx() not in declared_tetrahedral_atoms:
            atom.SetChiralTag(chem.ChiralType.CHI_UNSPECIFIED)
    for begin, end in unknown_double_bonds:
        bond = declared.GetBondBetweenAtoms(begin, end)
        if bond is None:
            raise ValueError(
                "Formal SDF-conformer V2000 bond is absent from the parsed " "molecule."
            )
        bond.SetStereo(chem.BondStereo.STEREOANY)
        for atom_index in (begin, end):
            for neighboring_bond in declared.GetAtomWithIdx(atom_index).GetBonds():
                if neighboring_bond.GetIdx() != bond.GetIdx():
                    neighboring_bond.SetBondDir(chem.BondDir.NONE)
    if declared.GetNumConformers():
        declared.GetConformer().Set3D(False)
    declared = chem.RemoveHs(declared, sanitize=True)
    identity = chem.MolToSmiles(
        declared,
        canonical=True,
        isomericSmiles=True,
    )
    if not identity:
        raise ValueError(
            "Formal SDF-conformer record lacks a declared-stereo identity."
        )
    return identity


def _canonical_structure_from_molecular_payload(
    payload: bytes,
    *,
    molecular_input_format: str,
) -> str:
    """Parse one exact molecular payload and derive its canonical graph."""

    input_format = str(molecular_input_format).strip().casefold()
    if input_format not in _MOLECULAR_INPUT_FORMATS:
        raise ValueError(
            "molecular_input_format must be one of: "
            + ", ".join(sorted(_MOLECULAR_INPUT_FORMATS))
            + "."
        )
    if not payload:
        raise ValueError("Molecular-input payload must be non-empty.")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "Verified molecular-input receipts require RDKit to parse the exact "
            "payload used for prediction."
        ) from exc

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Molecular-input payload must be UTF-8 text.") from exc

    molecule = None
    if input_format == "headerless_xyz":
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError(
                "A formal headerless XYZ payload must contain one or more "
                "non-empty atom lines before optional trailing blank lines."
            )
        normalized_lines = []
        periodic_table = Chem.GetPeriodicTable()
        for index, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    "Every formal headerless XYZ atom line must contain exactly "
                    "four whitespace-separated fields."
                )
            element = fields[0]
            try:
                atomic_number = int(periodic_table.GetAtomicNumber(element))
            except RuntimeError as exc:
                raise ValueError(
                    f"Headerless XYZ line {index} has an unknown element symbol."
                ) from exc
            if (
                atomic_number <= 0
                or periodic_table.GetElementSymbol(atomic_number) != element
            ):
                raise ValueError(
                    f"Headerless XYZ line {index} has a non-canonical element symbol."
                )
            try:
                coordinates = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Headerless XYZ line {index} has invalid coordinates."
                ) from exc
            if any(not math.isfinite(value) for value in coordinates):
                raise ValueError(
                    f"Headerless XYZ line {index} coordinates must be finite."
                )
            normalized_lines.append(
                f"{element} {coordinates[0]:.17g} {coordinates[1]:.17g} "
                f"{coordinates[2]:.17g}"
            )
        xyz_block = (
            f"{len(normalized_lines)}\nMAPLE verified headerless XYZ\n"
            + "\n".join(normalized_lines)
            + "\n"
        )
        molecule = Chem.MolFromXYZBlock(xyz_block)
        if molecule is None or molecule.GetNumAtoms() != len(normalized_lines):
            raise ValueError("The formal headerless XYZ payload cannot be parsed.")
        try:
            from rdkit.Chem import rdDetermineBonds

            rdDetermineBonds.DetermineBonds(
                molecule,
                charge=0,
                allowChargedFragments=False,
                embedChiral=False,
            )
            Chem.SanitizeMol(molecule)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "The formal headerless XYZ payload does not yield one neutral "
                "canonical molecular graph."
            ) from exc
        if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0 or any(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        ):
            raise ValueError(
                "Formal headerless XYZ molecular inputs must be neutral and "
                "closed-shell."
            )
        molecule = Chem.RemoveHs(molecule, sanitize=True)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(
                "Formal headerless XYZ molecular inputs must contain one connected "
                "molecular graph."
            )
    elif input_format == "smiles":
        rows = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(rows) != 1:
            raise ValueError(
                "A formal SMILES molecular-input receipt must contain exactly one "
                "non-comment record."
            )
        molecule = Chem.MolFromSmiles(rows[0].split()[0])
    elif input_format == "inchi":
        identifier = text.strip()
        if "\n" in identifier or not identifier.startswith("InChI="):
            raise ValueError(
                "A formal InChI molecular-input receipt must contain exactly one "
                "standard InChI record."
            )
        molecule = Chem.MolFromInchi(identifier)
    elif input_format == "sdf_conformers":
        blocks = [block for block in text.split("$$$$") if block.strip()]
        if not 1 <= len(blocks) <= 5:
            raise ValueError(
                "A formal SDF-conformer molecular-input receipt must contain "
                "between one and five molecule records."
            )
        molecules = tuple(
            Chem.ForwardSDMolSupplier(
                io.BytesIO(payload),
                sanitize=True,
                removeHs=False,
                strictParsing=True,
            )
        )
        if len(molecules) != len(blocks):
            raise ValueError(
                "Formal SDF-conformer record boundaries do not match parsed "
                "molecules."
            )
        if any(item is None for item in molecules):
            raise ValueError(
                "Every formal SDF-conformer record must parse as a molecule."
            )
        canonical_graphs = tuple(
            Chem.MolToSmiles(
                Chem.RemoveHs(Chem.Mol(item), sanitize=True),
                canonical=True,
                isomericSmiles=False,
            )
            for item in molecules
        )
        if any(not graph for graph in canonical_graphs):
            raise ValueError(
                "Every formal SDF-conformer record must yield a canonical graph."
            )
        if len(set(canonical_graphs)) != 1:
            raise ValueError(
                "Formal SDF-conformer records must represent the same canonical "
                "molecular graph."
            )
        stereo_identities = tuple(
            _v2000_declared_stereo_identity(block, item, chem=Chem)
            for block, item in zip(blocks, molecules, strict=True)
        )
        if len(set(stereo_identities)) != 1:
            raise ValueError(
                "Formal SDF-conformer records must preserve the same explicitly "
                "declared stereochemistry; distinct stereoisomers cannot be "
                "averaged."
            )
        return f"SMILES:{canonical_graphs[0]}"
    elif input_format == "sdf":
        blocks = [block.strip() for block in text.split("$$$$") if block.strip()]
        if len(blocks) != 1:
            raise ValueError(
                "A formal SDF molecular-input receipt must contain exactly one "
                "molecule record."
            )
        molecule = Chem.MolFromMolBlock(
            blocks[0],
            sanitize=True,
            removeHs=True,
            strictParsing=True,
        )
    elif input_format == "mol":
        molecule = Chem.MolFromMolBlock(
            text,
            sanitize=True,
            removeHs=True,
            strictParsing=True,
        )
    elif input_format == "mol2":
        molecule = Chem.MolFromMol2Block(
            text,
            sanitize=True,
            removeHs=True,
            cleanupSubstructures=True,
        )
    elif input_format == "pdb":
        molecule = Chem.MolFromPDBBlock(
            text,
            sanitize=True,
            removeHs=True,
            proximityBonding=True,
        )
    if molecule is None:
        raise ValueError(
            "The exact molecular-input payload cannot be parsed as the declared "
            f"{input_format!r} format."
        )
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if not canonical:
        raise ValueError(
            "The exact molecular-input payload did not yield a canonical graph."
        )
    return f"SMILES:{canonical}"


@dataclass(frozen=True, init=False)
class MolecularInputReceipt:
    """Receipt derived from the exact bytes supplied at the featurization boundary."""

    solute_structure_identifier: str
    molecular_input_sha256: str
    molecular_input_locator: str
    molecular_input_format: str
    _source_path: str
    _archive_member: str | None

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "MolecularInputReceipt":
        raise TypeError(
            "MolecularInputReceipt cannot be constructed from metadata; use "
            "MolecularInputReceipt.from_file()."
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        molecular_input_format: str,
        archive_member: str | None = None,
        locator: str | None = None,
    ) -> "MolecularInputReceipt":
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Molecular-input receipt source must be a regular file.")
        normalized_member = (
            str(archive_member).strip() if archive_member is not None else None
        )
        if normalized_member == "":
            raise ValueError("archive_member must be non-empty when provided.")
        input_format = str(molecular_input_format).strip().casefold()
        payload = cls._read_source_payload(source, normalized_member)
        canonical_structure = _canonical_structure_from_molecular_payload(
            payload,
            molecular_input_format=input_format,
        )
        if locator is None:
            if normalized_member is None:
                normalized_locator = f"file://{source}"
            else:
                normalized_locator = f"zip://{source}!/{normalized_member}"
        else:
            normalized_locator = str(locator).strip()
            if not normalized_locator:
                raise ValueError("molecular_input_locator must be non-empty.")

        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "solute_structure_identifier",
            canonical_structure,
        )
        object.__setattr__(
            instance,
            "molecular_input_sha256",
            hashlib.sha256(payload).hexdigest(),
        )
        object.__setattr__(
            instance,
            "molecular_input_locator",
            normalized_locator,
        )
        object.__setattr__(instance, "molecular_input_format", input_format)
        object.__setattr__(instance, "_source_path", str(source))
        object.__setattr__(instance, "_archive_member", normalized_member)
        return instance

    @staticmethod
    def _read_source_payload(
        source: Path,
        archive_member: str | None,
    ) -> bytes:
        if archive_member is None:
            return source.read_bytes()
        try:
            with zipfile.ZipFile(source) as archive:
                return archive.read(archive_member)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(
                "Molecular-input archive member cannot be read exactly."
            ) from exc

    def read_verified_payload(self) -> bytes:
        """Re-read and verify the exact bytes that the model input must use."""

        source = Path(self._source_path)
        payload = self._read_source_payload(source, self._archive_member)
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != self.molecular_input_sha256:
            raise ValueError(
                "Molecular-input source bytes changed after the receipt was created."
            )
        observed_structure = _canonical_structure_from_molecular_payload(
            payload,
            molecular_input_format=self.molecular_input_format,
        )
        if observed_structure != self.solute_structure_identifier:
            raise ValueError(
                "Molecular-input source graph changed after the receipt was created."
            )
        return payload


@dataclass(frozen=True, init=False)
class BenchmarkRecordInput:
    """Exact, receipt-backed molecular input for one accuracy record."""

    record_id: str
    solute_structure_identifier: str
    molecular_input_sha256: str
    molecular_input_locator: str
    molecular_input_format: str
    _receipt: MolecularInputReceipt

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "BenchmarkRecordInput":
        raise TypeError(
            "BenchmarkRecordInput requires a verified receipt; use "
            "BenchmarkRecordInput.from_receipt()."
        )

    @classmethod
    def from_receipt(
        cls,
        *,
        record_id: str,
        receipt: MolecularInputReceipt,
    ) -> "BenchmarkRecordInput":
        normalized_record_id = str(record_id).strip()
        if not normalized_record_id:
            raise ValueError("Benchmark-record input record_id must be non-empty.")
        if not isinstance(receipt, MolecularInputReceipt):
            raise TypeError(
                "BenchmarkRecordInput requires a MolecularInputReceipt generated "
                "from an actual file or archive member."
            )
        receipt.read_verified_payload()
        instance = object.__new__(cls)
        object.__setattr__(instance, "record_id", normalized_record_id)
        object.__setattr__(
            instance,
            "solute_structure_identifier",
            receipt.solute_structure_identifier,
        )
        object.__setattr__(
            instance,
            "molecular_input_sha256",
            receipt.molecular_input_sha256,
        )
        object.__setattr__(
            instance,
            "molecular_input_locator",
            receipt.molecular_input_locator,
        )
        object.__setattr__(
            instance,
            "molecular_input_format",
            receipt.molecular_input_format,
        )
        object.__setattr__(instance, "_receipt", receipt)
        return instance

    def verify_source(self) -> MolecularInputReceipt:
        """Fail closed unless all public fields still match the verified source."""

        self._receipt.read_verified_payload()
        for field in (
            "solute_structure_identifier",
            "molecular_input_sha256",
            "molecular_input_locator",
            "molecular_input_format",
        ):
            if getattr(self, field) != getattr(self._receipt, field):
                raise ValueError(
                    f"Benchmark-record input {field} differs from its verified "
                    "molecular-input receipt."
                )
        return self._receipt

    def canonical_payload(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "solute_structure_identifier": self.solute_structure_identifier,
            "molecular_input_sha256": self.molecular_input_sha256,
            "molecular_input_locator": self.molecular_input_locator,
            "molecular_input_format": self.molecular_input_format,
        }


@dataclass(frozen=True, init=False)
class BenchmarkExperimentalReference:
    """Record-level label derived from and bound to an exact JSON artifact."""

    record_id: str
    value: float
    unit: str
    provenance: str
    source_artifact_sha256: str
    source_record_locator: str
    _source_path: str

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "BenchmarkExperimentalReference":
        raise TypeError(
            "BenchmarkExperimentalReference cannot be constructed from metadata; "
            "use BenchmarkExperimentalReference.from_json_file()."
        )

    @staticmethod
    def _normalized_record_id(record_id: str) -> str:
        normalized = str(record_id).strip()
        if not normalized:
            raise ValueError("Experimental-reference record_id must be non-empty.")
        return normalized

    @staticmethod
    def _expected_locator(record_id: str) -> str:
        return f"$.{record_id}"

    @classmethod
    def _parse_entry(
        cls,
        payload: bytes,
        *,
        record_id: str,
        source_record_locator: str,
    ) -> tuple[float, str, str]:
        expected_locator = cls._expected_locator(record_id)
        if source_record_locator != expected_locator:
            raise ValueError(
                "Experimental-reference source_record_locator must identify the "
                f"top-level JSON entry {expected_locator!r}."
            )
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                "Experimental-reference source artifact is not valid JSON."
            ) from exc
        if not isinstance(document, Mapping):
            raise TypeError(
                "Experimental-reference source artifact root must be a JSON object."
            )
        try:
            entry = document[record_id]
        except KeyError as exc:
            raise ValueError(
                f"Experimental-reference source lacks record {record_id!r}."
            ) from exc
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Experimental-reference entry {record_id!r} must be a JSON object."
            )
        try:
            raw_value = entry["value"]
            raw_unit = entry["unit"]
            raw_provenance = entry["provenance"]
        except KeyError as exc:
            raise ValueError(
                f"Experimental-reference entry {record_id!r} lacks required field "
                f"{exc.args[0]!r}."
            ) from exc
        if isinstance(raw_value, bool):
            raise ValueError("Experimental-reference value must be a finite number.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Experimental-reference value must be a finite number."
            ) from exc
        if not math.isfinite(value):
            raise ValueError("Experimental-reference value must be finite.")
        unit = str(raw_unit).strip()
        if unit != "kcal/mol":
            raise ValueError(
                "Formal solvation accuracy references must use the explicit "
                "'kcal/mol' unit."
            )
        provenance = str(raw_provenance).strip()
        if not provenance:
            raise ValueError("Experimental-reference provenance must be non-empty.")
        return value, unit, provenance

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        record_id: str,
        source_record_locator: str | None = None,
    ) -> "BenchmarkExperimentalReference":
        """Read one exact top-level record and freeze its source bytes."""

        normalized_record_id = cls._normalized_record_id(record_id)
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(
                "Experimental-reference source must be a regular JSON file."
            )
        locator = (
            cls._expected_locator(normalized_record_id)
            if source_record_locator is None
            else str(source_record_locator).strip()
        )
        if not locator:
            raise ValueError(
                "Experimental-reference source_record_locator must be non-empty."
            )
        payload = source.read_bytes()
        value, unit, provenance = cls._parse_entry(
            payload,
            record_id=normalized_record_id,
            source_record_locator=locator,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "record_id", normalized_record_id)
        object.__setattr__(instance, "value", value)
        object.__setattr__(instance, "unit", unit)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(
            instance,
            "source_artifact_sha256",
            hashlib.sha256(payload).hexdigest(),
        )
        object.__setattr__(instance, "source_record_locator", locator)
        object.__setattr__(instance, "_source_path", str(source))
        return instance

    def verify_source(self) -> "BenchmarkExperimentalReference":
        """Re-read the exact artifact and reject byte, locator, or value drift."""

        source = Path(self._source_path)
        payload = source.read_bytes()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != self.source_artifact_sha256:
            raise ValueError(
                "Experimental-reference source bytes changed after the receipt "
                "was created."
            )
        value, unit, provenance = self._parse_entry(
            payload,
            record_id=self.record_id,
            source_record_locator=self.source_record_locator,
        )
        if value != self.value or unit != self.unit or provenance != self.provenance:
            raise ValueError(
                "Experimental-reference source entry changed after the receipt "
                "was created."
            )
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Return the path-independent reference identity used for hashing."""

        return {
            "record_id": self.record_id,
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_record_locator": self.source_record_locator,
        }


def _formal_taxonomy_queries(
    taxonomy: FunctionalGroupTaxonomyManifest,
) -> tuple[Any, dict[str, Any]]:
    if taxonomy.taxonomy_id != FORMAL_FUNCTIONAL_GROUP_TAXONOMY_ID:
        raise ValueError(
            "Accuracy evaluation accepts only the frozen MAPLE structural "
            "functional-group taxonomy."
        )
    if taxonomy.fingerprint != FORMAL_FUNCTIONAL_GROUP_TAXONOMY_FINGERPRINT:
        raise ValueError(
            "Functional-group taxonomy semantic fingerprint is not the reviewed "
            "formal accuracy taxonomy."
        )
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "Formal functional-group accuracy evaluation requires RDKit to verify "
            "every structure and frozen SMARTS assignment."
        ) from exc

    queries: dict[str, Any] = {}
    for definition in taxonomy.definitions:
        query = Chem.MolFromSmarts(definition.smarts)
        if query is None:
            raise ValueError(
                f"Frozen SMARTS is invalid for group {definition.group_id!r}."
            )
        queries[definition.group_id] = query
    return Chem, queries


def _canonical_structure_molecule(
    structure_identifier: str,
    *,
    chem: Any,
) -> tuple[str, Any]:
    identifier = str(structure_identifier).strip()
    if identifier.startswith("SMILES:"):
        raw = identifier.removeprefix("SMILES:").strip()
        molecule = chem.MolFromSmiles(raw)
        if molecule is None:
            raise ValueError(
                f"Functional-group structure_identifier has invalid SMILES: {raw!r}."
            )
        canonical = chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        return f"SMILES:{canonical}", molecule
    if identifier.startswith("InChI="):
        molecule = chem.MolFromInchi(identifier)
        if molecule is None:
            raise ValueError("Functional-group structure_identifier has invalid InChI.")
        return chem.MolToInchi(molecule), molecule
    raise ValueError(
        "Functional-group structure_identifier must be an explicit SMILES:<value> "
        "or standard InChI=<value> identifier."
    )


def _derive_functional_group_assignment(
    *,
    record_id: str,
    structure_identifier: str,
    taxonomy: FunctionalGroupTaxonomyManifest,
    assignment_evidence: str,
    chem: Any,
    queries: Mapping[str, Any],
) -> FunctionalGroupAssignment:
    canonical_structure, molecule = _canonical_structure_molecule(
        structure_identifier,
        chem=chem,
    )
    matched = tuple(
        group_id
        for group_id in taxonomy.group_ids
        if molecule.HasSubstructMatch(queries[group_id])
    )
    if not matched:
        raise ValueError(
            "The record has no structural match in the frozen functional-group "
            "taxonomy and cannot contribute to an accuracy panel."
        )
    matched_set = set(matched)
    primary = next(
        (
            group_id
            for group_id in taxonomy.primary_precedence
            if group_id in matched_set
        ),
        None,
    )
    if primary is None:
        raise RuntimeError(
            "Frozen functional-group precedence does not cover all SMARTS matches."
        )
    return FunctionalGroupAssignment(
        record_id=record_id,
        structure_identifier=canonical_structure,
        primary_group=primary,
        matched_groups=matched,
        assignment_evidence=assignment_evidence,
    )


def derive_functional_group_assignment(
    *,
    record_id: str,
    structure_identifier: str,
    taxonomy: FunctionalGroupTaxonomyManifest,
    assignment_evidence: str,
) -> FunctionalGroupAssignment:
    """Derive all matches and the unique primary family from the frozen taxonomy."""

    chem, queries = _formal_taxonomy_queries(taxonomy)
    return _derive_functional_group_assignment(
        record_id=record_id,
        structure_identifier=structure_identifier,
        taxonomy=taxonomy,
        assignment_evidence=assignment_evidence,
        chem=chem,
        queries=queries,
    )


@dataclass(frozen=True)
class SolventComposition:
    """Physical liquid composition tied to one benchmark record.

    Components are canonicalized with their mole fractions as pairs, so a
    permutation of a physical mixture cannot create a different benchmark
    identity.  Zero-fraction entries are forbidden rather than silently
    dropped: their presence usually signals an ambiguous input convention.
    """

    record_id: str
    components: tuple[str, ...]
    mole_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        if not record_id:
            raise ValueError("Solvent composition record_id must be non-empty.")
        components = _normalized_ids(self.components, field="solvent components")
        if len(components) != len(self.mole_fractions):
            raise ValueError(
                "Solvent components and mole_fractions must have the same length."
            )

        fractions: list[float] = []
        for fraction in self.mole_fractions:
            try:
                value = float(fraction)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Solvent mole fractions must be finite numbers."
                ) from exc
            if not math.isfinite(value):
                raise ValueError("Solvent mole fractions must be finite numbers.")
            if value <= 0.0:
                raise ValueError("Solvent mole fractions must be strictly positive.")
            fractions.append(value)

        if not math.isclose(sum(fractions), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Solvent mole fractions must sum to one.")

        ordered = tuple(sorted(zip(components, fractions), key=lambda pair: pair[0]))
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "components", tuple(pair[0] for pair in ordered))
        object.__setattr__(self, "mole_fractions", tuple(pair[1] for pair in ordered))


@dataclass(frozen=True)
class BenchmarkIdentity:
    """Everything that must match before two accuracy panels are compared."""

    dataset: str
    dataset_version: str
    record_ids: tuple[str, ...]
    temperature_kelvin: float
    standard_state: str
    protonation_policy: str
    tautomer_policy: str
    conformer_policy: str
    geometry_protocol: str
    solvent_protocol: str
    potential: str
    solvation_backend: str
    cavity_model: str
    sampling_protocol: str
    estimator: str
    experimental_provenance: str
    target_quantity: "BenchmarkQuantity"
    record_solvent_compositions: tuple[SolventComposition, ...] = ()
    record_inputs: tuple[BenchmarkRecordInput, ...] = ()
    record_experimental_references: tuple[BenchmarkExperimentalReference, ...] = ()
    accuracy_adapter_id: str | None = None
    accuracy_adapter_fingerprint: str | None = None
    functional_group_taxonomy: FunctionalGroupTaxonomyManifest | None = None
    record_functional_groups: tuple[FunctionalGroupAssignment, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "dataset",
            "dataset_version",
            "standard_state",
            "protonation_policy",
            "tautomer_policy",
            "conformer_policy",
            "geometry_protocol",
            "solvent_protocol",
            "potential",
            "solvation_backend",
            "cavity_model",
            "sampling_protocol",
            "estimator",
            "experimental_provenance",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must be explicit and non-empty.")
        if (
            not math.isfinite(float(self.temperature_kelvin))
            or self.temperature_kelvin <= 0
        ):
            raise ValueError("temperature_kelvin must be finite and positive.")
        object.__setattr__(
            self,
            "record_ids",
            _normalized_ids(self.record_ids, field="record_ids"),
        )
        if not isinstance(self.target_quantity, BenchmarkQuantity):
            try:
                object.__setattr__(
                    self,
                    "target_quantity",
                    BenchmarkQuantity(str(self.target_quantity)),
                )
            except ValueError as exc:
                raise ValueError("target_quantity is invalid.") from exc
        compositions = tuple(self.record_solvent_compositions)
        if compositions:
            if any(not isinstance(item, SolventComposition) for item in compositions):
                raise TypeError(
                    "record_solvent_compositions must contain SolventComposition instances."
                )
            by_record_id = {item.record_id: item for item in compositions}
            if len(by_record_id) != len(compositions):
                raise ValueError(
                    "record_solvent_compositions contains duplicate record identifiers."
                )
            if set(by_record_id) != set(self.record_ids):
                raise ValueError(
                    "record_solvent_compositions must contain exactly one composition "
                    "for every benchmark record ID."
                )
            object.__setattr__(
                self,
                "record_solvent_compositions",
                tuple(by_record_id[record_id] for record_id in self.record_ids),
            )

        record_inputs = tuple(self.record_inputs)
        if record_inputs:
            if any(
                not isinstance(item, BenchmarkRecordInput) for item in record_inputs
            ):
                raise TypeError(
                    "record_inputs must contain BenchmarkRecordInput instances."
                )
            inputs_by_record_id = {item.record_id: item for item in record_inputs}
            if len(inputs_by_record_id) != len(record_inputs):
                raise ValueError("record_inputs contains duplicate record identifiers.")
            if set(inputs_by_record_id) != set(self.record_ids):
                raise ValueError(
                    "record_inputs must contain exactly one molecular input for "
                    "every benchmark record ID."
                )
            object.__setattr__(
                self,
                "record_inputs",
                tuple(inputs_by_record_id[record_id] for record_id in self.record_ids),
            )

        references = tuple(self.record_experimental_references)
        if references:
            if any(
                not isinstance(item, BenchmarkExperimentalReference)
                for item in references
            ):
                raise TypeError(
                    "record_experimental_references must contain "
                    "BenchmarkExperimentalReference instances."
                )
            references_by_record_id = {item.record_id: item for item in references}
            if len(references_by_record_id) != len(references):
                raise ValueError(
                    "record_experimental_references contains duplicate record "
                    "identifiers."
                )
            if set(references_by_record_id) != set(self.record_ids):
                raise ValueError(
                    "record_experimental_references must contain exactly one "
                    "frozen reference for every benchmark record ID."
                )
            for reference in references:
                reference.verify_source()
            object.__setattr__(
                self,
                "record_experimental_references",
                tuple(
                    references_by_record_id[record_id] for record_id in self.record_ids
                ),
            )

        adapter_id = (
            str(self.accuracy_adapter_id).strip()
            if self.accuracy_adapter_id is not None
            else ""
        )
        adapter_fingerprint = (
            str(self.accuracy_adapter_fingerprint).strip().lower()
            if self.accuracy_adapter_fingerprint is not None
            else ""
        )
        if bool(adapter_id) != bool(adapter_fingerprint):
            raise ValueError(
                "accuracy_adapter_id and accuracy_adapter_fingerprint must be "
                "declared together."
            )
        if adapter_fingerprint:
            adapter_fingerprint = _normalized_sha256(
                adapter_fingerprint,
                field="accuracy_adapter_fingerprint",
            )
        object.__setattr__(
            self,
            "accuracy_adapter_id",
            adapter_id or None,
        )
        object.__setattr__(
            self,
            "accuracy_adapter_fingerprint",
            adapter_fingerprint or None,
        )

        taxonomy = self.functional_group_taxonomy
        assignments = tuple(self.record_functional_groups)
        if assignments:
            if not isinstance(taxonomy, FunctionalGroupTaxonomyManifest):
                raise ValueError(
                    "record_functional_groups requires a frozen "
                    "FunctionalGroupTaxonomyManifest."
                )
            if any(
                not isinstance(item, FunctionalGroupAssignment) for item in assignments
            ):
                raise TypeError(
                    "record_functional_groups must contain "
                    "FunctionalGroupAssignment instances."
                )
            by_record_id = {item.record_id: item for item in assignments}
            if len(by_record_id) != len(assignments):
                raise ValueError(
                    "record_functional_groups contains duplicate record identifiers."
                )
            if set(by_record_id) != set(self.record_ids):
                raise ValueError(
                    "record_functional_groups must contain exactly one "
                    "functional-group assignment for every benchmark record ID."
                )
            if not record_inputs:
                raise ValueError(
                    "record_functional_groups requires an exact, hash-bound "
                    "BenchmarkRecordInput for every benchmark record."
                )
            if not references:
                raise ValueError(
                    "record_functional_groups requires a frozen "
                    "BenchmarkExperimentalReference for every benchmark record."
                )
            if not adapter_id:
                raise ValueError(
                    "record_functional_groups requires a registered accuracy "
                    "adapter identity and implementation fingerprint."
                )
            chem, queries = _formal_taxonomy_queries(taxonomy)
            canonical_inputs_by_record_id: dict[str, BenchmarkRecordInput] = {}
            for record_input in self.record_inputs:
                receipt = record_input.verify_source()
                canonical_structure, _molecule = _canonical_structure_molecule(
                    receipt.solute_structure_identifier,
                    chem=chem,
                )
                if canonical_structure != receipt.solute_structure_identifier:
                    raise ValueError(
                        "Verified molecular-input receipt is not canonically encoded."
                    )
                canonical_inputs_by_record_id[record_input.record_id] = record_input
            object.__setattr__(
                self,
                "record_inputs",
                tuple(
                    canonical_inputs_by_record_id[record_id]
                    for record_id in self.record_ids
                ),
            )
            allowed_groups = set(taxonomy.group_ids)
            unknown_groups = sorted(
                {
                    group
                    for assignment in assignments
                    for group in assignment.matched_groups
                    if group not in allowed_groups
                }
            )
            if unknown_groups:
                raise ValueError(
                    "record_functional_groups contains groups absent from the frozen "
                    "taxonomy: " + ", ".join(unknown_groups) + "."
                )
            verified_by_record_id: dict[str, FunctionalGroupAssignment] = {}
            for assignment in assignments:
                record_input = canonical_inputs_by_record_id[assignment.record_id]
                assignment_structure, _molecule = _canonical_structure_molecule(
                    assignment.structure_identifier,
                    chem=chem,
                )
                if assignment_structure != record_input.solute_structure_identifier:
                    raise ValueError(
                        f"Functional-group structure for record "
                        f"{assignment.record_id!r} does not match the exact "
                        "benchmark molecular input."
                    )
                derived = _derive_functional_group_assignment(
                    record_id=assignment.record_id,
                    structure_identifier=record_input.solute_structure_identifier,
                    taxonomy=taxonomy,
                    assignment_evidence=assignment.assignment_evidence,
                    chem=chem,
                    queries=queries,
                )
                if assignment.matched_groups != derived.matched_groups:
                    raise ValueError(
                        f"Functional-group matches for record {assignment.record_id!r} "
                        "do not equal the frozen SMARTS result."
                    )
                if assignment.primary_group != derived.primary_group:
                    raise ValueError(
                        f"Primary functional group for record {assignment.record_id!r} "
                        "does not follow the frozen precedence."
                    )
                verified_by_record_id[assignment.record_id] = derived
            object.__setattr__(
                self,
                "record_functional_groups",
                tuple(
                    verified_by_record_id[record_id] for record_id in self.record_ids
                ),
            )
        elif taxonomy is not None:
            raise ValueError(
                "functional_group_taxonomy requires record-level assignments."
            )
        object.__setattr__(self, "functional_group_taxonomy", taxonomy)

        if self.target_quantity is BenchmarkQuantity.ABSOLUTE_SOLVATION_FREE_ENERGY:
            if not any(character.isdigit() for character in str(self.standard_state)):
                raise ValueError(
                    "Absolute solvation free-energy identities require a quantitative "
                    "standard state (for example, 1M)."
                )
            estimator = str(self.estimator).casefold().replace("_", "-")
            has_thermodynamic_estimator = any(
                token in estimator
                for token in (
                    "thermodynamic-integration",
                    "free-energy-perturbation",
                    "bar",
                    "mbar",
                    "fep",
                    "harmonic",
                )
            )
            if not has_thermodynamic_estimator:
                raise ValueError(
                    "Absolute solvation free-energy identities require a thermodynamic "
                    "estimator (TI, BAR, MBAR, FEP, or an explicitly named harmonic protocol)."
                )
            sampling = str(self.sampling_protocol).casefold().replace("_", "-")
            has_ensemble_or_harmonic_protocol = any(
                token in sampling
                for token in (
                    "molecular-dynamics",
                    "trajectory",
                    "replica",
                    "monte-carlo",
                    "mcmc",
                    "harmonic",
                    "sampling",
                )
            )
            if not has_ensemble_or_harmonic_protocol:
                raise ValueError(
                    "Absolute solvation free-energy identities require a sampling protocol; "
                    "a fixed-geometry or single-point energy difference is not an absolute "
                    "free-energy result."
                )

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record_ids"] = list(self.record_ids)
        payload["record_inputs"] = [
            record_input.canonical_payload() for record_input in self.record_inputs
        ]
        payload["record_experimental_references"] = [
            reference.canonical_payload()
            for reference in self.record_experimental_references
        ]
        return payload

    def _hash_payload(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._hash_payload(self.canonical_payload())

    @property
    def panel_fingerprint(self) -> str:
        """Fingerprint for scientifically comparable panels (identity minus potential/model id)."""
        payload = self.canonical_payload()
        payload = dict(payload)
        payload.pop("potential", None)
        payload.pop("accuracy_adapter_id", None)
        payload.pop("accuracy_adapter_fingerprint", None)
        return self._hash_payload(payload)

    @property
    def matched_task_fingerprint(self) -> str:
        """Workload identity shared by distinct candidate and QM implementations.

        The potential, solvation implementation, cavity implementation, and
        prediction adapter may differ because those are the methods being
        compared.  The sampling and estimator protocols remain bound: removing
        them would permit timing a cheaper workflow than the one whose accuracy
        was accepted.
        """

        payload = dict(self.canonical_payload())
        for field in (
            "potential",
            "solvation_backend",
            "cavity_model",
            "accuracy_adapter_id",
            "accuracy_adapter_fingerprint",
        ):
            payload.pop(field, None)
        return self._hash_payload(payload)


class BenchmarkQuantity(str, Enum):
    """The physical quantity a Route 4 benchmark is allowed to summarize."""

    ABSOLUTE_SOLVATION_FREE_ENERGY = "absolute_solvation_free_energy"
    GEOMETRY_LEVEL_SOLUTION_PMF = "geometry_level_solution_pmf"
    PROPERTY_PREDICTION = "property_prediction"


class LeakageStatus(str, Enum):
    KNOWN_OVERLAP = "known_overlap"
    OVERLAP_UNKNOWN = "overlap_unknown"
    STRICT_HOLDOUT = "strict_holdout"


@dataclass(frozen=True)
class ModelTrainingEvidence:
    """Auditable evidence about the exact records used to train a model."""

    training_datasets: tuple[str, ...] = ()
    known_training_record_ids: tuple[str, ...] = ()
    explicitly_excluded_datasets: tuple[str, ...] = ()
    explicitly_excluded_record_ids: tuple[str, ...] = ()
    record_accounting_complete: bool = False
    evidence_source: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "training_datasets",
            "known_training_record_ids",
            "explicitly_excluded_datasets",
            "explicitly_excluded_record_ids",
        ):
            values = tuple(
                value
                for value in (str(item).strip() for item in getattr(self, field))
                if value
            )
            object.__setattr__(self, field, values)


@dataclass(frozen=True)
class LeakageAudit:
    status: LeakageStatus
    overlapping_record_ids: tuple[str, ...]
    rationale: str
    evidence_source: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "overlapping_record_ids": list(self.overlapping_record_ids),
            "rationale": self.rationale,
            "evidence_source": self.evidence_source,
        }


def audit_training_overlap(
    identity: BenchmarkIdentity,
    evidence: ModelTrainingEvidence,
) -> LeakageAudit:
    """Label overlap conservatively; missing evidence is never a holdout."""

    benchmark_ids = set(identity.record_ids)
    overlaps = tuple(
        sorted(benchmark_ids.intersection(evidence.known_training_record_ids))
    )
    if overlaps:
        return LeakageAudit(
            LeakageStatus.KNOWN_OVERLAP,
            overlaps,
            "One or more benchmark record identifiers are documented training records.",
            evidence.evidence_source,
        )

    dataset_key = identity.dataset.casefold()
    used_datasets = {item.casefold() for item in evidence.training_datasets}
    excluded_datasets = {
        item.casefold() for item in evidence.explicitly_excluded_datasets
    }
    excluded_records = set(evidence.explicitly_excluded_record_ids)
    every_record_excluded = benchmark_ids.issubset(excluded_records)

    if (
        evidence.record_accounting_complete
        and (dataset_key in excluded_datasets or every_record_excluded)
        and dataset_key not in used_datasets
    ):
        return LeakageAudit(
            LeakageStatus.STRICT_HOLDOUT,
            (),
            "Complete training accounting explicitly excludes this dataset or every benchmark record.",
            evidence.evidence_source,
        )

    if dataset_key in used_datasets:
        rationale = (
            "The benchmark dataset is named in training evidence, but exact record overlap "
            "cannot be resolved."
        )
    else:
        rationale = (
            "Training evidence does not completely account for every benchmark record; "
            "absence of a documented overlap is not holdout proof."
        )
    return LeakageAudit(
        LeakageStatus.OVERLAP_UNKNOWN,
        (),
        rationale,
        evidence.evidence_source,
    )


class ValidationStage(str, Enum):
    """Ordered validation stages for a single immutable model identity."""

    DEVELOPMENT = "development"
    CONFIRMATION = "confirmation"
    INDEPENDENT_EXTRAPOLATION = "independent_extrapolation"


@dataclass(frozen=True)
class ValidationPanel:
    """One declared benchmark panel and its allowed role in model selection."""

    model_id: str
    stage: ValidationStage
    identity: BenchmarkIdentity
    training_evidence: ModelTrainingEvidence
    selection_allowed: bool
    independent_experimental_source: str | None = None

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        if not model_id:
            raise ValueError("Validation panel model_id must be non-empty.")
        object.__setattr__(self, "model_id", model_id)
        if not isinstance(self.stage, ValidationStage):
            try:
                object.__setattr__(self, "stage", ValidationStage(self.stage))
            except ValueError as exc:
                raise ValueError("Validation panel stage is invalid.") from exc
        if self.stage is ValidationStage.DEVELOPMENT:
            if not self.selection_allowed:
                raise ValueError(
                    "Development panels must declare selection_allowed=true."
                )
        elif self.selection_allowed:
            raise ValueError(
                "Confirmation and independent extrapolation panels must be sealed "
                "from model selection."
            )

        source = self.independent_experimental_source
        normalized_source = str(source).strip() if source is not None else None
        if self.stage is ValidationStage.INDEPENDENT_EXTRAPOLATION:
            if not normalized_source:
                raise ValueError(
                    "Independent extrapolation panels require an explicit external "
                    "experimental source."
                )
        object.__setattr__(self, "independent_experimental_source", normalized_source)


class ValidationLedger:
    """Fail-closed declaration ledger for development, confirmation, and final panels.

    The ledger prevents a final extrapolation panel from reusing a record that
    previously participated in model selection or confirmation.  It also
    requires complete training-record accounting before an independent final
    panel can be declared.
    """

    def __init__(self, panels: Sequence[ValidationPanel] = ()) -> None:
        self._panels: list[ValidationPanel] = []
        for panel in panels:
            self.add(panel)

    @property
    def panels(self) -> tuple[ValidationPanel, ...]:
        return tuple(self._panels)

    def add(self, panel: ValidationPanel) -> None:
        if not isinstance(panel, ValidationPanel):
            raise TypeError("ValidationLedger accepts ValidationPanel instances only.")
        same_model = [
            entry for entry in self._panels if entry.model_id == panel.model_id
        ]
        stages = {entry.stage for entry in same_model}
        if panel.stage in stages:
            raise ValueError(
                f"Model {panel.model_id!r} already has a {panel.stage.value!r} panel."
            )

        required_previous = {
            ValidationStage.DEVELOPMENT: (),
            ValidationStage.CONFIRMATION: (ValidationStage.DEVELOPMENT,),
            ValidationStage.INDEPENDENT_EXTRAPOLATION: (
                ValidationStage.DEVELOPMENT,
                ValidationStage.CONFIRMATION,
            ),
        }[panel.stage]
        missing = [stage.value for stage in required_previous if stage not in stages]
        if missing:
            raise ValueError(
                f"Model {panel.model_id!r} cannot declare {panel.stage.value!r} "
                "before " + ", ".join(missing) + "."
            )

        used_record_ids = {
            record_id for entry in same_model for record_id in entry.identity.record_ids
        }
        repeated = tuple(
            sorted(used_record_ids.intersection(panel.identity.record_ids))
        )
        if repeated:
            raise ValueError(
                "Validation panels for one model must not reuse record IDs; "
                f"repeated: {', '.join(repeated)}."
            )

        if panel.stage is ValidationStage.INDEPENDENT_EXTRAPOLATION:
            leakage = audit_training_overlap(panel.identity, panel.training_evidence)
            if leakage.status is not LeakageStatus.STRICT_HOLDOUT:
                raise ValueError(
                    "Independent extrapolation requires complete training accounting "
                    "and a strict training holdout."
                )

        self._panels.append(panel)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = np.sign(left[j] - left[i])
            dy = np.sign(right[j] - right[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    if denominator == 0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def _metric_bundle(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    absolute = np.abs(error)
    ss_total = float(np.sum((reference - reference.mean()) ** 2))
    r_squared = (
        float(1.0 - np.sum(error**2) / ss_total) if ss_total > 0 else float("nan")
    )
    return {
        "mae_kcal_mol": float(absolute.mean()),
        "rmse_kcal_mol": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "maximum_absolute_error_kcal_mol": float(absolute.max()),
        "mean_signed_error_kcal_mol": float(error.mean()),
        "r_squared": r_squared,
        "spearman_rho": _correlation(
            _average_ranks(reference), _average_ranks(prediction)
        ),
        "kendall_tau_b": _kendall_tau_b(reference, prediction),
    }


def audit_accuracy_panel_coverage(identity: BenchmarkIdentity) -> dict[str, Any]:
    """Report the non-negotiable record and functional-group accuracy gates."""

    if not isinstance(identity, BenchmarkIdentity):
        raise TypeError("Accuracy coverage requires a BenchmarkIdentity.")
    taxonomy = identity.functional_group_taxonomy
    primary_groups = sorted(
        {assignment.primary_group for assignment in identity.record_functional_groups}
    )
    record_count_passes = len(identity.record_ids) >= MINIMUM_ACCURACY_RECORDS
    functional_group_count_passes = (
        len(primary_groups) >= MINIMUM_ACCURACY_FUNCTIONAL_GROUPS
    )
    assignments_complete = len(identity.record_functional_groups) == len(
        identity.record_ids
    )
    record_inputs_complete = len(identity.record_inputs) == len(identity.record_ids)
    experimental_references_complete = len(
        identity.record_experimental_references
    ) == len(identity.record_ids)
    adapter_identity_declared = bool(
        identity.accuracy_adapter_id and identity.accuracy_adapter_fingerprint
    )
    result = {
        "functional_group_taxonomy": (
            taxonomy.taxonomy_id if taxonomy is not None else None
        ),
        "minimum_record_count": MINIMUM_ACCURACY_RECORDS,
        "observed_record_count": len(identity.record_ids),
        "record_count_passes": record_count_passes,
        "minimum_distinct_functional_groups": (MINIMUM_ACCURACY_FUNCTIONAL_GROUPS),
        "observed_count": len(primary_groups),
        "observed_functional_groups": primary_groups,
        "record_assignments_complete": assignments_complete,
        "record_inputs_complete": record_inputs_complete,
        "experimental_references_complete": experimental_references_complete,
        "accuracy_adapter_identity_declared": adapter_identity_declared,
        "passes": (
            record_count_passes
            and functional_group_count_passes
            and assignments_complete
            and record_inputs_complete
            and experimental_references_complete
            and adapter_identity_declared
            and taxonomy is not None
        ),
    }
    if taxonomy is not None:
        result["functional_group_taxonomy_fingerprint"] = taxonomy.fingerprint
        result["assignment_basis"] = (
            "distinct predeclared primary functional-group families"
        )
    return result


def _require_accuracy_panel(identity: BenchmarkIdentity) -> dict[str, Any]:
    coverage = audit_accuracy_panel_coverage(identity)
    if identity.functional_group_taxonomy is None:
        raise ValueError(
            "Accuracy evaluation requires a frozen functional-group taxonomy manifest."
        )
    if coverage["observed_count"] < MINIMUM_ACCURACY_FUNCTIONAL_GROUPS:
        raise ValueError(
            "Accuracy evaluation requires at least 10 distinct functional groups, "
            "counted as predeclared primary families with structure-bound record "
            "assignments."
        )
    if coverage["observed_record_count"] < MINIMUM_ACCURACY_RECORDS:
        raise ValueError("Accuracy evaluation requires at least 10 records.")
    if not coverage["record_assignments_complete"]:
        raise ValueError(
            "Accuracy evaluation requires complete record-level functional-group "
            "assignments."
        )
    if not coverage["record_inputs_complete"]:
        raise ValueError(
            "Accuracy evaluation requires complete, hash-bound molecular inputs for "
            "every record."
        )
    if not coverage["experimental_references_complete"]:
        raise ValueError(
            "Accuracy evaluation requires frozen experimental values and source "
            "identities for every record."
        )
    if not coverage["accuracy_adapter_identity_declared"]:
        raise ValueError(
            "Accuracy evaluation requires a registered adapter identity and "
            "implementation fingerprint."
        )
    return coverage


def summarize_runtime_smoke(
    identity: BenchmarkIdentity,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize execution coverage without computing experimental accuracy."""

    if not isinstance(identity, BenchmarkIdentity):
        raise TypeError("summarize_runtime_smoke requires a BenchmarkIdentity.")
    if not records:
        raise ValueError("At least one runtime-smoke record is required.")

    seen: set[str] = set()
    evaluated = 0
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("Every record must define a non-empty record_id.")
        if record_id in seen:
            raise ValueError(f"Duplicate benchmark record_id: {record_id}")
        seen.add(record_id)
        try:
            prediction = float(record.get("predicted_kcal_mol"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(prediction):
            evaluated += 1

    if seen != set(identity.record_ids):
        raise ValueError(
            "Prediction record IDs must match the declared benchmark identity exactly."
        )
    record_count = len(records)
    return {
        "identity_fingerprint": identity.fingerprint,
        "panel_fingerprint": identity.panel_fingerprint,
        "matched_task_fingerprint": identity.matched_task_fingerprint,
        "target_quantity": identity.target_quantity.value,
        "record_count": record_count,
        "evaluated_count": evaluated,
        "failure_count": record_count - evaluated,
        "coverage": evaluated / record_count,
        "failure_rate": (record_count - evaluated) / record_count,
        "metrics": None,
        "per_record_errors": None,
        "accuracy_evaluation_performed": False,
        "accuracy_metric_reporting_allowed": False,
        "functional_group_coverage": audit_accuracy_panel_coverage(identity),
        "scope": "runtime_or_interface_smoke_only",
    }


def _summarize_verified_predictions(
    identity: BenchmarkIdentity,
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize records emitted only by the trusted accuracy runner."""

    if not isinstance(identity, BenchmarkIdentity):
        raise TypeError("summarize_predictions requires a BenchmarkIdentity.")
    if not records:
        raise ValueError("At least one benchmark record is required.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    functional_group_coverage = _require_accuracy_panel(identity)
    functional_group_assignments = {
        assignment.record_id: assignment
        for assignment in identity.record_functional_groups
    }
    experimental_references = {
        reference.record_id: reference
        for reference in identity.record_experimental_references
    }
    benchmark_inputs = {
        record_input.record_id: record_input for record_input in identity.record_inputs
    }

    seen: set[str] = set()
    usable: list[tuple[str, float, float, str, BenchmarkRecordInput]] = []
    failures = 0
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("Every record must define a non-empty record_id.")
        if record_id in seen:
            raise ValueError(f"Duplicate benchmark record_id: {record_id}")
        seen.add(record_id)
        if record_id not in benchmark_inputs:
            raise ValueError(
                f"Benchmark record {record_id!r} is absent from the exact input "
                "manifest."
            )
        expected_input = benchmark_inputs[record_id]
        expected_input.verify_source()
        observed_receipt = record.get("molecular_input_receipt")
        if not isinstance(observed_receipt, MolecularInputReceipt):
            raise ValueError(
                f"Benchmark record {record_id!r} lacks a verified "
                "molecular_input_receipt from the actual featurization input."
            )
        observed_receipt.read_verified_payload()
        if (
            observed_receipt.solute_structure_identifier
            != expected_input.solute_structure_identifier
        ):
            raise ValueError(
                f"Actual prediction-input structure for record {record_id!r} does "
                "not match the frozen benchmark molecular input."
            )
        if (
            observed_receipt.molecular_input_sha256
            != expected_input.molecular_input_sha256
        ):
            raise ValueError(
                f"Actual prediction-input hash for record {record_id!r} does not "
                "match the frozen benchmark molecular input."
            )
        if (
            observed_receipt.molecular_input_locator
            != expected_input.molecular_input_locator
        ):
            raise ValueError(
                f"Actual prediction-input locator for record {record_id!r} does not "
                "match the frozen benchmark molecular input."
            )
        if (
            observed_receipt.molecular_input_format
            != expected_input.molecular_input_format
        ):
            raise ValueError(
                f"Actual prediction-input format for record {record_id!r} does not "
                "match the frozen benchmark molecular input."
            )
        expected_reference = experimental_references[record_id].verify_source()
        reference_value = expected_reference.value
        experimental_provenance = expected_reference.provenance
        prediction = record.get("predicted_kcal_mol")

        try:
            prediction_value = float(prediction)
        except (TypeError, ValueError):
            failures += 1
            continue
        if not math.isfinite(prediction_value):
            failures += 1
            continue
        usable.append(
            (
                record_id,
                reference_value,
                prediction_value,
                experimental_provenance,
                expected_input,
            )
        )

    if seen != set(identity.record_ids):
        raise ValueError(
            "Prediction record IDs must match the declared benchmark identity exactly."
        )
    if failures:
        raise ValueError(
            "Accuracy evaluation requires a finite prediction for every record; "
            "failed predictions may be summarized only as runtime smoke."
        )

    count = len(usable)
    result: dict[str, Any] = {
        "identity_fingerprint": identity.fingerprint,
        "panel_fingerprint": identity.panel_fingerprint,
        "matched_task_fingerprint": identity.matched_task_fingerprint,
        "target_quantity": identity.target_quantity.value,
        "record_count": len(records),
        "evaluated_count": count,
        "failure_count": failures,
        "coverage": count / len(records),
        "failure_rate": failures / len(records),
        "accuracy_evaluation_performed": True,
        "accuracy_metric_reporting_allowed": True,
        "functional_group_coverage": functional_group_coverage,
    }
    reference = np.asarray([item[1] for item in usable], dtype=float)
    prediction = np.asarray([item[2] for item in usable], dtype=float)
    result["metrics"] = _metric_bundle(reference, prediction)
    by_record_id = {
        record_id: (
            reference_value,
            prediction_value,
            experimental_provenance,
            record_input,
        )
        for (
            record_id,
            reference_value,
            prediction_value,
            experimental_provenance,
            record_input,
        ) in usable
    }
    result["per_record_errors"] = [
        {
            "record_id": record_id,
            "primary_functional_group": (
                functional_group_assignments[record_id].primary_group
            ),
            "functional_groups": list(
                functional_group_assignments[record_id].matched_groups
            ),
            "functional_group_structure_identifier": (
                functional_group_assignments[record_id].structure_identifier
            ),
            "functional_group_assignment_evidence": (
                functional_group_assignments[record_id].assignment_evidence
            ),
            "solute_structure_identifier": (
                by_record_id[record_id][3].solute_structure_identifier
            ),
            "molecular_input_sha256": (
                by_record_id[record_id][3].molecular_input_sha256
            ),
            "molecular_input_locator": (
                by_record_id[record_id][3].molecular_input_locator
            ),
            "molecular_input_format": (
                by_record_id[record_id][3].molecular_input_format
            ),
            "experimental_provenance": by_record_id[record_id][2],
            "experimental_unit": experimental_references[record_id].unit,
            "experimental_source_artifact_sha256": (
                experimental_references[record_id].source_artifact_sha256
            ),
            "experimental_source_record_locator": (
                experimental_references[record_id].source_record_locator
            ),
            "experimental_kcal_mol": by_record_id[record_id][0],
            "predicted_kcal_mol": by_record_id[record_id][1],
            "signed_error_kcal_mol": (
                by_record_id[record_id][1] - by_record_id[record_id][0]
            ),
            "absolute_error_kcal_mol": abs(
                by_record_id[record_id][1] - by_record_id[record_id][0]
            ),
        }
        for record_id in identity.record_ids
    ]

    if bootstrap_samples == 0:
        result["mae_bootstrap_95_ci_kcal_mol"] = None
    else:
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, count, size=(bootstrap_samples, count))
        absolute_error = np.abs(prediction - reference)
        bootstrap_mae = absolute_error[indices].mean(axis=1)
        low, high = np.percentile(bootstrap_mae, (2.5, 97.5))
        result["mae_bootstrap_95_ci_kcal_mol"] = [float(low), float(high)]
    return result


class VerifiedMolecularInput:
    """One path-free input handle that records actual payload consumption."""

    __slots__ = (
        "_payload",
        "_solute_structure_identifier",
        "_molecular_input_sha256",
        "_molecular_input_format",
        "_consumption_count",
    )

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "VerifiedMolecularInput":
        raise TypeError(
            "VerifiedMolecularInput is runner-owned and cannot be constructed "
            "directly."
        )

    @classmethod
    def _from_verified_payload(
        cls,
        payload: bytes,
        *,
        solute_structure_identifier: str,
        molecular_input_sha256: str,
        molecular_input_format: str,
    ) -> "VerifiedMolecularInput":
        if not isinstance(payload, bytes):
            raise TypeError("Verified molecular-input payload must be bytes.")
        normalized_sha256 = _normalized_sha256(
            molecular_input_sha256,
            field="molecular_input_sha256",
        )
        if hashlib.sha256(payload).hexdigest() != normalized_sha256:
            raise ValueError(
                "Isolated molecular-input bytes differ from the parent-verified "
                "payload hash."
            )
        normalized_format = str(molecular_input_format).strip().casefold()
        observed_structure = _canonical_structure_from_molecular_payload(
            payload,
            molecular_input_format=normalized_format,
        )
        if observed_structure != str(solute_structure_identifier).strip():
            raise ValueError(
                "Isolated molecular-input graph differs from the parent-verified "
                "structure."
            )
        instance = object.__new__(cls)
        instance._payload = payload
        instance._solute_structure_identifier = observed_structure
        instance._molecular_input_sha256 = normalized_sha256
        instance._molecular_input_format = normalized_format
        instance._consumption_count = 0
        return instance

    def read(self) -> bytes:
        """Return the verified payload and attest that the predictor consumed it."""

        self._consumption_count += 1
        return self._payload

    @property
    def solute_structure_identifier(self) -> str:
        return self._solute_structure_identifier

    @property
    def molecular_input_sha256(self) -> str:
        return self._molecular_input_sha256

    @property
    def molecular_input_format(self) -> str:
        return self._molecular_input_format

    @property
    def was_consumed(self) -> bool:
        return self._consumption_count > 0


@dataclass(frozen=True)
class AccuracyPredictionContext:
    """Label-free scientific context exposed to one reviewed predictor."""

    record_id: str
    molecular_input: VerifiedMolecularInput
    temperature_kelvin: float
    standard_state: str
    protonation_policy: str
    tautomer_policy: str
    conformer_policy: str
    geometry_protocol: str
    solvent_protocol: str
    solvent_components: tuple[str, ...]
    solvent_mole_fractions: tuple[float, ...]
    potential: str
    solvation_backend: str
    cavity_model: str
    sampling_protocol: str
    estimator: str
    target_quantity: BenchmarkQuantity
    implementation_artifact_paths: Mapping[str, str]


class AccuracyModelAdapter:
    """Base class for statically registered and code-reviewed accuracy adapters."""

    def predict(
        self,
        *,
        context: AccuracyPredictionContext,
    ) -> Any:
        raise NotImplementedError


_REQUIRED_ACCURACY_ADAPTER_ARTIFACT_ROLES = frozenset(
    {
        "adapter_code",
        "featurizer_code",
        "checkpoint",
        "dependency_lock",
    }
)


@dataclass(frozen=True, init=False)
class AccuracyAdapterArtifactReceipt:
    """Exact source-controlled implementation artifact used by one adapter."""

    role: str
    artifact_sha256: str
    _source_path: str

    def __new__(
        cls,
        *_args: Any,
        **_kwargs: Any,
    ) -> "AccuracyAdapterArtifactReceipt":
        raise TypeError(
            "AccuracyAdapterArtifactReceipt cannot be constructed from metadata; "
            "use AccuracyAdapterArtifactReceipt.from_file()."
        )

    @classmethod
    def from_file(
        cls,
        *,
        role: str,
        path: str | Path,
    ) -> "AccuracyAdapterArtifactReceipt":
        normalized_role = str(role).strip().casefold()
        if (
            not normalized_role
            or not normalized_role[0].isalpha()
            or any(
                not (character.islower() or character.isdigit() or character == "_")
                for character in normalized_role
            )
        ):
            raise ValueError(
                "Accuracy-adapter artifact role must be a lowercase snake-case "
                "identifier."
            )
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Accuracy-adapter artifact source must be a regular file.")
        instance = object.__new__(cls)
        object.__setattr__(instance, "role", normalized_role)
        object.__setattr__(
            instance,
            "artifact_sha256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        object.__setattr__(instance, "_source_path", str(source))
        return instance

    def verify_source(self) -> "AccuracyAdapterArtifactReceipt":
        observed_sha256 = hashlib.sha256(
            Path(self._source_path).read_bytes()
        ).hexdigest()
        if observed_sha256 != self.artifact_sha256:
            raise ValueError(
                f"Accuracy-adapter {self.role!r} artifact bytes changed after "
                "registration."
            )
        return self

    def canonical_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_sha256": self.artifact_sha256,
        }


def _normalized_code_constant(value: Any) -> Any:
    """Serialize one immutable code constant without CPython quickening state."""

    if isinstance(value, CodeType):
        return {"code": _normalized_code_payload(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_normalized_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_normalized_code_constant(item) for item in value]
        return {
            "frozenset": sorted(
                items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            )
        }
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"ellipsis": True}
    return _canonical_json_value(value, field="adapter code constant")


def _normalized_code_payload(code: CodeType) -> dict[str, Any]:
    """Return stable public bytecode semantics, excluding paths and line numbers."""

    return {
        "name": code.co_name,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_hex": code.co_code.hex(),
        "constants": [_normalized_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "exceptiontable_hex": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _adapter_runtime_implementation_payload(
    adapter: AccuracyModelAdapter,
) -> dict[str, Any]:
    """Fingerprint live executable code and all declared JSON runtime state."""

    adapter_type = type(adapter)
    predict_method = getattr(adapter_type, "predict", None)
    code = getattr(predict_method, "__code__", None)
    if not isinstance(code, CodeType):
        raise TypeError(
            "Formal accuracy adapters must implement predict() as auditable "
            "Python code."
        )
    closure_values: list[Any] = []
    for index, cell in enumerate(getattr(predict_method, "__closure__", None) or ()):
        try:
            closure_value = cell.cell_contents
        except ValueError as exc:
            raise ValueError(
                "Formal accuracy adapter predict() contains an empty closure cell."
            ) from exc
        closure_values.append(
            _canonical_json_value(
                closure_value,
                field=f"adapter predict closure[{index}]",
            )
        )

    class_state: dict[str, Any] = {}
    for name, value in vars(adapter_type).items():
        if name.startswith("__") or name == "predict":
            continue
        if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
            continue
        class_state[name] = _canonical_json_value(
            value,
            field=f"adapter class state.{name}",
        )

    return {
        "adapter_class_module": adapter_type.__module__,
        "adapter_class_qualname": adapter_type.__qualname__,
        "predict_code_sha256": _canonical_sha256(_normalized_code_payload(code)),
        "predict_defaults": _canonical_json_value(
            getattr(predict_method, "__defaults__", None),
            field="adapter predict defaults",
        ),
        "predict_keyword_defaults": _canonical_json_value(
            getattr(predict_method, "__kwdefaults__", None),
            field="adapter predict keyword defaults",
        ),
        "predict_closure": closure_values,
        "class_state": class_state,
        "instance_state": _canonical_json_value(
            dict(getattr(adapter, "__dict__", {})),
            field="adapter instance state",
        ),
    }


@dataclass(frozen=True, init=False)
class AccuracyAdapterRegistration:
    """Immutable reviewed adapter plus runner-verifiable implementation evidence."""

    adapter_id: str
    adapter: AccuracyModelAdapter
    implementation_artifacts: tuple[AccuracyAdapterArtifactReceipt, ...]
    configuration_json: str

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "AccuracyAdapterRegistration":
        raise TypeError(
            "AccuracyAdapterRegistration must be built from verified components; "
            "use AccuracyAdapterRegistration.from_components()."
        )

    @classmethod
    def from_components(
        cls,
        *,
        adapter_id: str,
        adapter: AccuracyModelAdapter,
        implementation_artifacts: Sequence[AccuracyAdapterArtifactReceipt],
        configuration: Mapping[str, Any],
    ) -> "AccuracyAdapterRegistration":
        normalized_adapter_id = str(adapter_id).strip()
        if not normalized_adapter_id:
            raise ValueError("Accuracy-adapter registration ID must be non-empty.")
        if not isinstance(adapter, AccuracyModelAdapter):
            raise TypeError(
                "Accuracy-adapter registration requires an AccuracyModelAdapter."
            )
        adapter_type = type(adapter)
        if "<locals>" in adapter_type.__qualname__:
            raise ValueError(
                "Formal accuracy adapter classes must be module-level and importable "
                "inside a fresh interpreter."
            )
        if not hasattr(adapter, "__dict__"):
            raise TypeError(
                "Formal accuracy adapters must expose all runtime state through a "
                "canonical JSON instance dictionary."
            )
        artifacts = tuple(implementation_artifacts)
        if any(
            not isinstance(item, AccuracyAdapterArtifactReceipt) for item in artifacts
        ):
            raise TypeError(
                "Accuracy-adapter implementation_artifacts must contain verified "
                "AccuracyAdapterArtifactReceipt instances."
            )
        roles = tuple(item.role for item in artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError(
                "Accuracy-adapter implementation artifacts contain duplicate roles."
            )
        adapter_required_roles = getattr(
            adapter_type,
            "required_artifact_roles",
            tuple(sorted(_REQUIRED_ACCURACY_ADAPTER_ARTIFACT_ROLES)),
        )
        if not isinstance(adapter_required_roles, (tuple, list, frozenset, set)):
            raise TypeError(
                "Accuracy-adapter required_artifact_roles must be a sequence of "
                "lowercase snake-case role names."
            )
        required_roles = frozenset(str(role).strip() for role in adapter_required_roles)
        if not required_roles or any(
            not role
            or not role[0].isalpha()
            or any(
                not (character.islower() or character.isdigit() or character == "_")
                for character in role
            )
            for role in required_roles
        ):
            raise ValueError(
                "Accuracy-adapter required_artifact_roles contains an invalid role."
            )
        missing_roles = sorted(required_roles.difference(roles))
        if missing_roles:
            raise ValueError(
                "Accuracy-adapter registration lacks required implementation "
                "artifact roles: " + ", ".join(missing_roles) + "."
            )
        adapter_source = inspect.getsourcefile(adapter_type)
        if adapter_source is None:
            raise ValueError("Formal accuracy adapter source file cannot be resolved.")
        registered_adapter_source = next(
            item for item in artifacts if item.role == "adapter_code"
        )
        if Path(adapter_source).resolve(strict=True) != Path(
            registered_adapter_source._source_path
        ).resolve(strict=True):
            raise ValueError(
                "The adapter_code artifact must be the exact source file that "
                "defines the registered adapter class."
            )
        for artifact in artifacts:
            artifact.verify_source()
        if not isinstance(configuration, Mapping):
            raise TypeError(
                "Accuracy-adapter configuration must be a canonical JSON mapping."
            )
        precision_policy = str(
            vars(adapter_type).get("accuracy_precision_policy", "")
        ).strip()
        if not precision_policy:
            raise ValueError(
                "Every concrete formal accuracy-adapter class must explicitly "
                "declare a non-empty accuracy_precision_policy."
            )
        supplied_precision = configuration.get("accuracy_precision_policy")
        if (
            supplied_precision is not None
            and str(supplied_precision).strip() != precision_policy
        ):
            raise ValueError(
                "Accuracy-adapter configuration precision differs from the "
                "adapter's executable precision contract."
            )
        configuration_with_precision = dict(configuration)
        configuration_with_precision["accuracy_precision_policy"] = precision_policy
        normalized_configuration = _canonical_json_value(
            configuration_with_precision,
            field="accuracy-adapter configuration",
        )
        configuration_json = json.dumps(
            normalized_configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "adapter_id", normalized_adapter_id)
        object.__setattr__(instance, "adapter", adapter)
        object.__setattr__(
            instance,
            "implementation_artifacts",
            tuple(sorted(artifacts, key=lambda item: item.role)),
        )
        object.__setattr__(instance, "configuration_json", configuration_json)
        instance.compute_fingerprint()
        return instance

    def compute_fingerprint(self) -> str:
        artifacts = []
        for artifact in self.implementation_artifacts:
            artifact.verify_source()
            artifacts.append(artifact.canonical_payload())
        return _canonical_sha256(
            {
                "adapter_id": self.adapter_id,
                "configuration": json.loads(self.configuration_json),
                "implementation_artifacts": artifacts,
                "runtime_implementation": _adapter_runtime_implementation_payload(
                    self.adapter
                ),
            }
        )

    def isolated_execution_spec(self) -> dict[str, Any]:
        """Return only implementation evidence needed by a fresh worker."""

        return {
            "adapter_id": self.adapter_id,
            "adapter_module": type(self.adapter).__module__,
            "adapter_qualname": type(self.adapter).__qualname__,
            "adapter_instance_state": _canonical_json_value(
                dict(self.adapter.__dict__),
                field="adapter instance state",
            ),
            "configuration": json.loads(self.configuration_json),
            "implementation_artifacts": [
                {
                    **artifact.canonical_payload(),
                    "source_path": artifact._source_path,
                }
                for artifact in self.implementation_artifacts
            ],
            "expected_fingerprint": self.compute_fingerprint(),
        }


def _resolve_isolated_adapter_type(
    *,
    module_name: str,
    qualname: str,
) -> type[AccuracyModelAdapter]:
    module = importlib.import_module(module_name)
    resolved: Any = module
    for component in qualname.split("."):
        if not component or component == "<locals>":
            raise ValueError(
                "Formal accuracy adapter qualname is not safely importable."
            )
        resolved = getattr(resolved, component)
    if not isinstance(resolved, type) or not issubclass(
        resolved,
        AccuracyModelAdapter,
    ):
        raise TypeError(
            "Freshly imported accuracy adapter is not an AccuracyModelAdapter class."
        )
    return resolved


def _registration_from_isolated_spec(
    spec: Mapping[str, Any],
) -> AccuracyAdapterRegistration:
    """Rebuild one adapter only from verified files and canonical JSON state."""

    artifacts: list[AccuracyAdapterArtifactReceipt] = []
    for item in spec["implementation_artifacts"]:
        receipt = AccuracyAdapterArtifactReceipt.from_file(
            role=item["role"],
            path=item["source_path"],
        )
        if receipt.artifact_sha256 != item["artifact_sha256"]:
            raise ValueError(
                f"Isolated accuracy-adapter {receipt.role!r} artifact hash differs "
                "from the parent registration."
            )
        artifacts.append(receipt)

    adapter_type = _resolve_isolated_adapter_type(
        module_name=str(spec["adapter_module"]),
        qualname=str(spec["adapter_qualname"]),
    )
    adapter = object.__new__(adapter_type)
    state = spec["adapter_instance_state"]
    if not isinstance(state, Mapping):
        raise TypeError("Isolated accuracy-adapter state must be a JSON object.")
    for name, value in state.items():
        normalized_name = str(name)
        if not normalized_name.isidentifier() or normalized_name.startswith("__"):
            raise ValueError(
                "Isolated accuracy-adapter state keys must be non-dunder Python "
                "identifiers."
            )
        object.__setattr__(adapter, normalized_name, value)

    registration = AccuracyAdapterRegistration.from_components(
        adapter_id=str(spec["adapter_id"]),
        adapter=adapter,
        implementation_artifacts=tuple(artifacts),
        configuration=spec["configuration"],
    )
    if registration.compute_fingerprint() != spec["expected_fingerprint"]:
        raise ValueError(
            "Freshly imported accuracy adapter does not match the frozen "
            "implementation fingerprint."
        )
    return registration


def _prediction_context_from_isolated_job(
    job: Mapping[str, Any],
    *,
    implementation_artifact_paths: Mapping[str, str],
) -> AccuracyPredictionContext:
    molecular_spec = job["molecular_input"]
    try:
        payload = base64.b64decode(
            str(molecular_spec["payload_base64"]),
            validate=True,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Isolated molecular-input payload is not valid base64."
        ) from exc
    handle = VerifiedMolecularInput._from_verified_payload(
        payload,
        solute_structure_identifier=molecular_spec["solute_structure_identifier"],
        molecular_input_sha256=molecular_spec["molecular_input_sha256"],
        molecular_input_format=molecular_spec["molecular_input_format"],
    )
    context_spec = job["context"]
    return AccuracyPredictionContext(
        record_id=str(job["record_id"]),
        molecular_input=handle,
        temperature_kelvin=float(context_spec["temperature_kelvin"]),
        standard_state=str(context_spec["standard_state"]),
        protonation_policy=str(context_spec["protonation_policy"]),
        tautomer_policy=str(context_spec["tautomer_policy"]),
        conformer_policy=str(context_spec["conformer_policy"]),
        geometry_protocol=str(context_spec["geometry_protocol"]),
        solvent_protocol=str(context_spec["solvent_protocol"]),
        solvent_components=tuple(context_spec["solvent_components"]),
        solvent_mole_fractions=tuple(
            float(value) for value in context_spec["solvent_mole_fractions"]
        ),
        potential=str(context_spec["potential"]),
        solvation_backend=str(context_spec["solvation_backend"]),
        cavity_model=str(context_spec["cavity_model"]),
        sampling_protocol=str(context_spec["sampling_protocol"]),
        estimator=str(context_spec["estimator"]),
        target_quantity=BenchmarkQuantity(str(context_spec["target_quantity"])),
        implementation_artifact_paths=MappingProxyType(
            {
                str(role): str(path)
                for role, path in implementation_artifact_paths.items()
            }
        ),
    )


def _adapter_import_root(registration_spec: Mapping[str, Any]) -> Path:
    """Return the import root for the exact registered adapter source file."""

    adapter_source = next(
        Path(item["source_path"])
        for item in registration_spec["implementation_artifacts"]
        if item["role"] == "adapter_code"
    )
    module_parts = str(registration_spec["adapter_module"]).split(".")
    levels = (
        len(module_parts)
        if adapter_source.name == "__init__.py"
        else (len(module_parts) - 1)
    )
    import_root = adapter_source.parent
    for _ in range(levels):
        import_root = import_root.parent
    return import_root


def _execute_isolated_accuracy_request(
    registration_spec: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one label-free request inside an already-confined interpreter."""

    try:
        import_root = str(_adapter_import_root(registration_spec))
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        registration = _registration_from_isolated_spec(registration_spec)
        expected_fingerprint = registration.compute_fingerprint()
        implementation_artifact_paths = MappingProxyType(
            {
                artifact.role: artifact._source_path
                for artifact in registration.implementation_artifacts
            }
        )
        predictions: list[dict[str, Any]] = []
        for job in jobs:
            context = _prediction_context_from_isolated_job(
                job,
                implementation_artifact_paths=implementation_artifact_paths,
            )
            prediction = registration.adapter.predict(context=context)
            if not context.molecular_input.was_consumed:
                raise RuntimeError(
                    "Predictor did not consume the runner-bound molecular input for "
                    f"record {context.record_id!r}."
                )
            try:
                prediction_value = float(prediction)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Accuracy evaluation requires a finite prediction for every "
                    "record; failed predictions may be summarized only as runtime "
                    "smoke."
                ) from exc
            if not math.isfinite(prediction_value):
                raise ValueError(
                    "Accuracy evaluation requires a finite prediction for every "
                    "record; failed predictions may be summarized only as runtime "
                    "smoke."
                )
            predictions.append(
                {
                    "record_id": context.record_id,
                    "predicted_kcal_mol": prediction_value,
                }
            )
        observed_fingerprint = registration.compute_fingerprint()
        if observed_fingerprint != expected_fingerprint:
            raise ValueError(
                "Registered accuracy adapter implementation changed during isolated "
                "panel execution."
            )
        return {
            "ok": True,
            "predictions": predictions,
            "accuracy_adapter_fingerprint": observed_fingerprint,
            "accuracy_precision_policy": str(
                registration.adapter.accuracy_precision_policy
            ),
        }
    except BaseException as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


_ACCURACY_WORKER_RESPONSE_PREFIX = "MAPLE_ACCURACY_RESPONSE="


def _isolated_accuracy_worker_cli() -> None:
    """Read one JSON request and emit one machine-readable response line."""

    try:
        request = json.loads(sys.stdin.read())
        response = _execute_isolated_accuracy_request(
            request["registration_spec"],
            request["jobs"],
        )
    except BaseException as exc:
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    sys.__stdout__.write(
        _ACCURACY_WORKER_RESPONSE_PREFIX
        + json.dumps(
            response,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    )
    sys.__stdout__.flush()


def _configured_formal_accuracy_adapters() -> dict[str, AccuracyAdapterRegistration]:
    """Build only explicitly configured source-controlled formal adapters."""

    registrations: dict[str, AccuracyAdapterRegistration] = {}
    c3net_source_root = os.environ.get("MAPLE_C3NET_FORMAL_SOURCE_ROOT")
    c3net_source_bundle = os.environ.get("MAPLE_C3NET_FORMAL_SOURCE_BUNDLE")
    if bool(c3net_source_root) != bool(c3net_source_bundle):
        raise RuntimeError(
            "Formal C3Net registration requires both "
            "MAPLE_C3NET_FORMAL_SOURCE_ROOT and "
            "MAPLE_C3NET_FORMAL_SOURCE_BUNDLE."
        )
    if c3net_source_root:
        from .c3net_accuracy import build_c3net_formal_accuracy_registration

        registration = build_c3net_formal_accuracy_registration(
            source_root=c3net_source_root,
            source_bundle=c3net_source_bundle,
        )
        registrations[registration.adapter_id] = registration

    atomicese_source_root = os.environ.get("MAPLE_ATOMICESE_FORMAL_SOURCE_ROOT")
    if atomicese_source_root:
        from .atomicese_accuracy import (
            build_atomicese_formal_accuracy_registration,
        )

        registration = build_atomicese_formal_accuracy_registration(
            source_root=atomicese_source_root,
        )
        if registration.adapter_id in registrations:
            raise RuntimeError(
                "Formal accuracy-adapter configuration produced a duplicate ID."
            )
        registrations[registration.adapter_id] = registration
    return registrations


# Production accuracy remains fail-closed unless exact, source-controlled
# implementation artifacts are explicitly configured before module import.
# Tests may monkeypatch the registry inside their isolated process; there is no
# public runtime registration API.
_FORMAL_ACCURACY_ADAPTERS: Mapping[str, AccuracyAdapterRegistration] = MappingProxyType(
    _configured_formal_accuracy_adapters()
)


def _registered_accuracy_adapter(
    identity: BenchmarkIdentity,
) -> AccuracyAdapterRegistration:
    adapter_id = identity.accuracy_adapter_id
    if not adapter_id:
        raise ValueError("Accuracy identity lacks a registered adapter ID.")
    try:
        registration = _FORMAL_ACCURACY_ADAPTERS[adapter_id]
    except KeyError as exc:
        raise RuntimeError(
            f"No reviewed formal accuracy adapter is registered for " f"{adapter_id!r}."
        ) from exc
    if not isinstance(registration, AccuracyAdapterRegistration):
        raise TypeError(
            f"Registered accuracy adapter {adapter_id!r} has an invalid registration."
        )
    if registration.adapter_id != adapter_id:
        raise ValueError(
            "Registered accuracy adapter ID does not match its registry key."
        )
    observed_fingerprint = registration.compute_fingerprint()
    if observed_fingerprint != identity.accuracy_adapter_fingerprint:
        raise ValueError(
            "Registered accuracy adapter implementation fingerprint does not "
            "match the frozen benchmark identity."
        )
    return registration


def _label_free_prediction_job(
    identity: BenchmarkIdentity,
    *,
    record_id: str,
    record_input: BenchmarkRecordInput,
) -> dict[str, Any]:
    compositions = {
        composition.record_id: composition
        for composition in identity.record_solvent_compositions
    }
    composition = compositions.get(record_id)
    receipt = record_input.verify_source()
    payload = receipt.read_verified_payload()
    return {
        "record_id": record_id,
        "molecular_input": {
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "solute_structure_identifier": receipt.solute_structure_identifier,
            "molecular_input_sha256": receipt.molecular_input_sha256,
            "molecular_input_format": receipt.molecular_input_format,
        },
        "context": {
            "temperature_kelvin": float(identity.temperature_kelvin),
            "standard_state": str(identity.standard_state),
            "protonation_policy": str(identity.protonation_policy),
            "tautomer_policy": str(identity.tautomer_policy),
            "conformer_policy": str(identity.conformer_policy),
            "geometry_protocol": str(identity.geometry_protocol),
            "solvent_protocol": str(identity.solvent_protocol),
            "solvent_components": (
                list(composition.components) if composition is not None else []
            ),
            "solvent_mole_fractions": (
                list(composition.mole_fractions) if composition is not None else []
            ),
            "potential": str(identity.potential),
            "solvation_backend": str(identity.solvation_backend),
            "cavity_model": str(identity.cavity_model),
            "sampling_protocol": str(identity.sampling_protocol),
            "estimator": str(identity.estimator),
            "target_quantity": identity.target_quantity.value,
        },
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _accuracy_sandbox_readonly_roots() -> tuple[Path, ...]:
    """Return trusted runtime trees visible inside the formal worker sandbox."""

    module_path = Path(__file__).resolve(strict=True)
    package_tree = module_path.parents[2]
    candidates = (
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path(sys.prefix),
        Path(sys.base_prefix),
        package_tree,
    )
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        normalized = candidate.absolute()
        if any(normalized == existing for existing in roots):
            continue
        roots.append(normalized)
    return tuple(roots)


def _assert_accuracy_sandbox_hides_references(
    *,
    identity: BenchmarkIdentity,
    registration: AccuracyAdapterRegistration,
    readonly_roots: Sequence[Path],
) -> None:
    """Fail closed if any experimental source would be mounted in the worker."""

    artifact_paths = tuple(
        Path(artifact._source_path).resolve(strict=True)
        for artifact in registration.implementation_artifacts
    )
    for reference in identity.record_experimental_references:
        reference_path = Path(reference._source_path).resolve(strict=True)
        if any(
            reference_path == artifact_path
            or os.path.samefile(reference_path, artifact_path)
            for artifact_path in artifact_paths
        ):
            raise ValueError(
                "An experimental-reference artifact cannot also be a formal "
                "accuracy-adapter implementation artifact."
            )
        if any(_path_is_within(reference_path, root) for root in readonly_roots):
            raise ValueError(
                "Experimental-reference artifacts must remain outside every "
                "read-only runtime tree mounted in the formal accuracy sandbox."
            )


def _accuracy_sandbox_command(
    *,
    registration: AccuracyAdapterRegistration,
    readonly_roots: Sequence[Path],
) -> list[str]:
    """Build one bubblewrap command exposing runtime code and exact artifacts."""

    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise RuntimeError(
            "Formal accuracy evaluation requires bubblewrap filesystem isolation; "
            "no bwrap executable is available, so execution is fail-closed."
        )
    command = [
        bubblewrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
    ]
    for root in readonly_roots:
        command.extend(("--ro-bind", str(root), str(root)))
    for artifact in registration.implementation_artifacts:
        source = Path(artifact._source_path).resolve(strict=True)
        if any(_path_is_within(source, root) for root in readonly_roots):
            continue
        command.extend(("--ro-bind", str(source), str(source)))
    repository_root = Path(__file__).resolve(strict=True).parents[3]
    worker_code = (
        "import sys;"
        f"sys.path.insert(0,{str(repository_root)!r});"
        "from maple.function.benchmarking.pretrained_hub "
        "import _isolated_accuracy_worker_cli;"
        "_isolated_accuracy_worker_cli()"
    )
    command.extend(
        (
            "--proc",
            "/proc",
            "--dev-bind",
            "/dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "USER",
            "maple",
            "--setenv",
            "LOGNAME",
            "maple",
            "--setenv",
            "TORCHINDUCTOR_CACHE_DIR",
            "/tmp/torchinductor",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            sys.executable,
            "-I",
            "-c",
            worker_code,
        )
    )
    return command


def _limit_accuracy_worker_process() -> None:
    """Disable core dumps without constraining model scratch-artifact sizes."""

    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class _AccuracyWorkerOutputLimitExceeded(Exception):
    """Internal signal for bounded stdout/stderr capture."""


def _accuracy_worker_timeout_seconds(
    registration: AccuracyAdapterRegistration,
    *,
    job_count: int,
) -> float:
    """Return a panel deadline covering every declared per-record budget."""

    configuration = json.loads(registration.configuration_json)
    try:
        configured_record_timeout = float(configuration.get("timeout_seconds", 30.0))
    except (TypeError, ValueError):
        configured_record_timeout = 30.0
    if not math.isfinite(configured_record_timeout) or configured_record_timeout <= 0:
        configured_record_timeout = 30.0
    return max(
        120.0,
        60.0 + max(1, int(job_count)) * configured_record_timeout,
    )


def _terminate_accuracy_worker(process: subprocess.Popen[bytes]) -> None:
    """Kill the isolated worker session and reap its direct process."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _run_bounded_accuracy_worker(
    command: Sequence[str],
    *,
    serialized_request: str,
    timeout_seconds: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one worker with aggregate, backpressured stdout/stderr capture."""

    request_bytes = serialized_request.encode("utf-8")
    with tempfile.TemporaryFile(mode="w+b") as stdin_handle:
        stdin_handle.write(request_bytes)
        stdin_handle.seek(0)
        process = subprocess.Popen(
            list(command),
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            preexec_fn=_limit_accuracy_worker_process,
            start_new_session=True,
            env=dict(env),
        )
        if process.stdout is None or process.stderr is None:
            _terminate_accuracy_worker(process)
            raise RuntimeError("Formal accuracy worker pipes were not created.")

        captured: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
        total_output_bytes = 0
        deadline = time.monotonic() + timeout_seconds
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        cmd=list(command),
                        timeout=timeout_seconds,
                    )
                events = selector.select(timeout=remaining)
                if not events:
                    raise subprocess.TimeoutExpired(
                        cmd=list(command),
                        timeout=timeout_seconds,
                    )
                for key, _mask in events:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    total_output_bytes += len(chunk)
                    if total_output_bytes > FORMAL_ACCURACY_WORKER_MAX_OUTPUT_BYTES:
                        raise _AccuracyWorkerOutputLimitExceeded
                    captured[key.data].append(chunk)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=list(command),
                    timeout=timeout_seconds,
                )
            returncode = process.wait(timeout=remaining)
        except (
            _AccuracyWorkerOutputLimitExceeded,
            subprocess.TimeoutExpired,
        ):
            _terminate_accuracy_worker(process)
            raise
        finally:
            selector.close()
            for stream in (process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()

    return subprocess.CompletedProcess(
        args=list(command),
        returncode=returncode,
        stdout=b"".join(captured["stdout"]).decode("utf-8", errors="replace"),
        stderr=b"".join(captured["stderr"]).decode("utf-8", errors="replace"),
    )


def _run_isolated_accuracy_predictions(
    *,
    identity: BenchmarkIdentity,
    registration: AccuracyAdapterRegistration,
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute a label-free panel in a fresh filesystem-confined interpreter."""

    readonly_roots = _accuracy_sandbox_readonly_roots()
    _assert_accuracy_sandbox_hides_references(
        identity=identity,
        registration=registration,
        readonly_roots=readonly_roots,
    )
    request = {
        "registration_spec": registration.isolated_execution_spec(),
        "jobs": list(jobs),
    }
    command = _accuracy_sandbox_command(
        registration=registration,
        readonly_roots=readonly_roots,
    )
    worker_timeout_seconds = _accuracy_worker_timeout_seconds(
        registration,
        job_count=len(jobs),
    )
    serialized_request = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    try:
        completed = _run_bounded_accuracy_worker(
            command,
            serialized_request=serialized_request,
            timeout_seconds=worker_timeout_seconds,
            env={
                "HOME": "/tmp",
                "PATH": str(Path(sys.executable).parent),
                "PYTHONNOUSERSITE": "1",
                "TMPDIR": "/tmp",
            },
        )
    except _AccuracyWorkerOutputLimitExceeded as exc:
        raise RuntimeError(
            "Filesystem-confined accuracy predictor exceeded the runner-owned "
            "aggregate stdout/stderr output limit."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Formal accuracy adapter exceeded its runner-owned whole-panel "
            f"{worker_timeout_seconds:g}-second timeout."
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Formal accuracy adapter could not start in a fresh filesystem " "sandbox."
        ) from exc
    stdout_text = completed.stdout
    stderr_text = completed.stderr
    if completed.returncode != 0:
        diagnostic = stderr_text.strip()
        if len(diagnostic) > 2000:
            diagnostic = diagnostic[-2000:]
        raise RuntimeError(
            "Filesystem-confined accuracy predictor exited unsuccessfully with "
            f"code {completed.returncode}: {diagnostic}"
        )
    response_lines = [
        line.removeprefix(_ACCURACY_WORKER_RESPONSE_PREFIX)
        for line in stdout_text.splitlines()
        if line.startswith(_ACCURACY_WORKER_RESPONSE_PREFIX)
    ]
    if not response_lines:
        raise RuntimeError(
            "Filesystem-confined accuracy predictor exited without a verified "
            "transcript."
        )
    try:
        response = json.loads(response_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Filesystem-confined accuracy predictor returned malformed JSON."
        ) from exc
    if not isinstance(response, Mapping):
        raise RuntimeError(
            "Isolated accuracy predictor returned an invalid transcript envelope."
        )
    if not response.get("ok"):
        error_types = {
            "AttributeError": AttributeError,
            "TypeError": TypeError,
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
        }
        error_type = error_types.get(str(response.get("error_type")), RuntimeError)
        raise error_type(
            "Isolated accuracy predictor failed: "
            + str(response.get("error_message", "unknown worker error"))
        )
    return dict(response)


def run_accuracy_panel(
    identity: BenchmarkIdentity,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Execute and summarize a formal panel at one trusted input boundary.

    The frozen identity supplies experimental labels and selects a statically
    registered, code-reviewed model adapter.  The runner owns each payload
    handle and never accepts a caller-provided predictor or prediction table.
    """

    if not isinstance(identity, BenchmarkIdentity):
        raise TypeError("run_accuracy_panel requires a BenchmarkIdentity.")
    _require_accuracy_panel(identity)
    registration = _registered_accuracy_adapter(identity)
    record_inputs = {item.record_id: item for item in identity.record_inputs}
    jobs = tuple(
        _label_free_prediction_job(
            identity,
            record_id=record_id,
            record_input=record_inputs[record_id],
        )
        for record_id in identity.record_ids
    )
    isolated = _run_isolated_accuracy_predictions(
        identity=identity,
        registration=registration,
        jobs=jobs,
    )
    observed_adapter_fingerprint = str(isolated["accuracy_adapter_fingerprint"]).strip()
    if observed_adapter_fingerprint != identity.accuracy_adapter_fingerprint:
        raise ValueError(
            "Isolated accuracy adapter fingerprint does not match the frozen "
            "benchmark identity."
        )
    if registration.compute_fingerprint() != observed_adapter_fingerprint:
        raise ValueError(
            "Parent accuracy-adapter registration changed during isolated panel "
            "execution."
        )
    observed_precision_policy = str(
        isolated.get("accuracy_precision_policy", "")
    ).strip()
    registered_precision_policy = str(
        registration.adapter.accuracy_precision_policy
    ).strip()
    if (
        not observed_precision_policy
        or observed_precision_policy != registered_precision_policy
    ):
        raise ValueError(
            "Isolated accuracy precision policy does not match the registered "
            "adapter contract."
        )
    predictions = isolated["predictions"]
    if not isinstance(predictions, Sequence) or isinstance(
        predictions,
        (str, bytes),
    ):
        raise RuntimeError(
            "Isolated accuracy predictor returned an invalid prediction transcript."
        )
    predictions_by_record_id: dict[str, float] = {}
    for item in predictions:
        if not isinstance(item, Mapping):
            raise RuntimeError(
                "Isolated accuracy prediction transcript contains an invalid row."
            )
        record_id = str(item.get("record_id", "")).strip()
        if not record_id or record_id in predictions_by_record_id:
            raise RuntimeError(
                "Isolated accuracy prediction transcript has missing or duplicate "
                "record IDs."
            )
        prediction = float(item["predicted_kcal_mol"])
        if not math.isfinite(prediction):
            raise RuntimeError(
                "Isolated accuracy prediction transcript contains a non-finite value."
            )
        predictions_by_record_id[record_id] = prediction
    if set(predictions_by_record_id) != set(identity.record_ids):
        raise RuntimeError(
            "Isolated accuracy prediction record IDs do not match the frozen panel."
        )

    for reference in identity.record_experimental_references:
        reference.verify_source()
    executed_records = [
        {
            "record_id": record_id,
            "molecular_input_receipt": record_inputs[record_id].verify_source(),
            "predicted_kcal_mol": predictions_by_record_id[record_id],
        }
        for record_id in identity.record_ids
    ]
    summary = _summarize_verified_predictions(
        identity,
        executed_records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary["accuracy_adapter_id"] = registration.adapter_id
    summary["accuracy_adapter_fingerprint"] = observed_adapter_fingerprint
    summary["accuracy_precision_policy"] = observed_precision_policy
    summary["experimental_references_verified_from_artifacts"] = True
    summary["predictions_executed_in_label_free_spawned_interpreter"] = True
    summary["predictions_executed_in_filesystem_sandbox"] = True
    summary["molecular_inputs_passed_as_verified_bytes_without_source_paths"] = True
    return summary


def summarize_predictions(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Reject free prediction tables; the registered runner returns the summary."""

    raise TypeError(
        "Direct prediction-table summarization is forbidden for formal accuracy; "
        "use run_accuracy_panel() with a frozen registered adapter identity."
    )


@dataclass(frozen=True)
class PairedComparison:
    panel_fingerprint: str
    run_fingerprint_a: str
    run_fingerprint_b: str
    record_count: int
    functional_group_count: int
    mean_delta_absolute_error_kcal_mol: float
    wins_a: int
    ties: int
    wins_b: int
    metrics_a: Mapping[str, float]
    metrics_b: Mapping[str, float]
    per_record_errors_a: tuple[Mapping[str, Any], ...]
    per_record_errors_b: tuple[Mapping[str, Any], ...]

    @property
    def benchmark_fingerprint(self) -> str:
        """Backward-compatible name for the shared scientific panel identity."""
        return self.panel_fingerprint


def paired_comparison(
    identity_a: BenchmarkIdentity,
    identity_b: BenchmarkIdentity,
) -> PairedComparison:
    """Execute and compare two registered adapters on one scientific panel.

    Full run identifiers may differ (e.g., potential/model version) as long as the
    panel context used for the comparison is identical.
    """

    if not isinstance(identity_a, BenchmarkIdentity) or not isinstance(
        identity_b, BenchmarkIdentity
    ):
        raise TypeError(
            "Paired accuracy comparison requires two BenchmarkIdentity instances."
        )
    if identity_a.panel_fingerprint != identity_b.panel_fingerprint:
        raise ValueError(
            "Paired accuracy comparison requires identical scientific panel identities."
        )
    functional_group_coverage = _require_accuracy_panel(identity_a)
    _require_accuracy_panel(identity_b)
    summary_a = run_accuracy_panel(identity_a, bootstrap_samples=0)
    summary_b = run_accuracy_panel(identity_b, bootstrap_samples=0)

    def collect(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[float, float]]:
        collected: dict[str, tuple[float, float]] = {}
        for record in records:
            record_id = str(record.get("record_id", "")).strip()
            if record_id in collected:
                raise ValueError(f"Duplicate paired record_id: {record_id}")
            try:
                reference = float(record["experimental_kcal_mol"])
                prediction = float(record["predicted_kcal_mol"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Paired record {record_id!r} lacks a finite prediction/reference."
                ) from exc
            if not (
                record_id and math.isfinite(reference) and math.isfinite(prediction)
            ):
                raise ValueError(
                    f"Paired record {record_id!r} lacks a finite prediction/reference."
                )
            collected[record_id] = (reference, prediction)
        return collected

    left = collect(summary_a["per_record_errors"])
    right = collect(summary_b["per_record_errors"])
    expected = set(identity_a.record_ids)
    if set(left) != expected or set(right) != expected:
        raise ValueError(
            "Paired comparison requires exactly one successful result for every identity record."
        )

    deltas: list[float] = []
    wins_a = ties = wins_b = 0
    for record_id in identity_a.record_ids:
        reference_a, prediction_a = left[record_id]
        reference_b, prediction_b = right[record_id]
        if reference_a != reference_b:
            raise ValueError(
                f"Experimental value differs for paired record {record_id!r}."
            )
        error_a = abs(prediction_a - reference_a)
        error_b = abs(prediction_b - reference_b)
        deltas.append(error_a - error_b)
        if math.isclose(error_a, error_b, rel_tol=0.0, abs_tol=1e-12):
            ties += 1
        elif error_a < error_b:
            wins_a += 1
        else:
            wins_b += 1

    return PairedComparison(
        panel_fingerprint=identity_a.panel_fingerprint,
        run_fingerprint_a=identity_a.fingerprint,
        run_fingerprint_b=identity_b.fingerprint,
        record_count=len(identity_a.record_ids),
        functional_group_count=functional_group_coverage["observed_count"],
        mean_delta_absolute_error_kcal_mol=float(np.mean(deltas)),
        wins_a=wins_a,
        ties=ties,
        wins_b=wins_b,
        metrics_a=summary_a["metrics"],
        metrics_b=summary_b["metrics"],
        per_record_errors_a=tuple(summary_a["per_record_errors"]),
        per_record_errors_b=tuple(summary_b["per_record_errors"]),
    )


def _accuracy_signature_for_performance(
    summary: Mapping[str, Any],
    *,
    maximum_absolute_error_limit_kcal_mol: float,
    label: str,
) -> tuple[str, str, str, tuple[tuple[str, str, float, str], ...]]:
    """Validate one formal accuracy result before any speed comparison."""

    if not isinstance(summary, Mapping):
        raise TypeError(f"{label} accuracy summary must be a mapping.")
    try:
        limit = float(maximum_absolute_error_limit_kcal_mol)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "maximum_absolute_error_limit_kcal_mol must be finite and positive."
        ) from exc
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(
            "maximum_absolute_error_limit_kcal_mol must be finite and positive."
        )
    required_attestations = (
        "accuracy_evaluation_performed",
        "accuracy_metric_reporting_allowed",
        "experimental_references_verified_from_artifacts",
        "predictions_executed_in_label_free_spawned_interpreter",
        "predictions_executed_in_filesystem_sandbox",
        "molecular_inputs_passed_as_verified_bytes_without_source_paths",
    )
    missing_attestations = [
        field for field in required_attestations if summary.get(field) is not True
    ]
    if missing_attestations:
        raise ValueError(
            f"{label} accuracy summary lacks formal-runner attestations: "
            + ", ".join(missing_attestations)
            + "."
        )
    for digest_field in (
        "identity_fingerprint",
        "panel_fingerprint",
        "matched_task_fingerprint",
        "accuracy_adapter_fingerprint",
    ):
        _normalized_sha256(
            str(summary.get(digest_field, "")).strip().lower(),
            field=f"{label} {digest_field}",
        )
    if not str(summary.get("accuracy_adapter_id", "")).strip():
        raise ValueError(f"{label} accuracy summary lacks an adapter identity.")
    target_quantity = str(summary.get("target_quantity", "")).strip()
    if not target_quantity:
        raise ValueError(f"{label} accuracy summary lacks a target quantity.")
    precision_policy = str(summary.get("accuracy_precision_policy", "")).strip()
    if not precision_policy:
        raise ValueError(f"{label} accuracy summary lacks a trusted precision policy.")

    coverage = summary.get("functional_group_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("passes") is not True:
        raise ValueError(
            f"{label} accuracy summary did not pass the formal panel coverage gate."
        )
    try:
        record_count = int(summary["record_count"])
        evaluated_count = int(summary["evaluated_count"])
        group_count = int(coverage["observed_count"])
        observed_record_count = int(coverage["observed_record_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} accuracy summary lacks finite formal accuracy metrics."
        ) from exc
    observed_groups = coverage.get("observed_functional_groups")
    if not isinstance(observed_groups, Sequence) or isinstance(
        observed_groups, (str, bytes)
    ):
        raise ValueError(
            f"{label} accuracy summary lacks formal functional-group evidence."
        )
    normalized_observed_groups = tuple(str(item).strip() for item in observed_groups)
    if (
        record_count < MINIMUM_ACCURACY_RECORDS
        or evaluated_count != record_count
        or observed_record_count != record_count
        or group_count < MINIMUM_ACCURACY_FUNCTIONAL_GROUPS
        or len(normalized_observed_groups) != group_count
        or any(not item for item in normalized_observed_groups)
        or len(set(normalized_observed_groups)) != group_count
    ):
        raise ValueError(
            f"{label} is not eligible for performance admission because its "
            "formal accuracy gate failed."
        )
    task_fingerprint = str(summary.get("matched_task_fingerprint", "")).strip().lower()
    _normalized_sha256(
        task_fingerprint,
        field=f"{label} matched_task_fingerprint",
    )
    records = summary.get("per_record_errors")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"{label} accuracy summary lacks per-record evidence.")
    signature: list[tuple[str, str, float, str]] = []
    predictions: list[float] = []
    references: list[float] = []
    primary_groups: list[str] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{label} per-record accuracy evidence contains an invalid row."
            )
        try:
            record_id = str(item["record_id"]).strip()
            input_sha256 = _normalized_sha256(
                str(item["molecular_input_sha256"]),
                field=f"{label} molecular_input_sha256",
            )
            experimental = float(item["experimental_kcal_mol"])
            prediction = float(item["predicted_kcal_mol"])
            signed_error = float(item["signed_error_kcal_mol"])
            absolute_error = float(item["absolute_error_kcal_mol"])
            primary_group = str(item["primary_functional_group"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} per-record accuracy evidence is incomplete."
            ) from exc
        expected_signed_error = prediction - experimental
        if (
            not record_id
            or not primary_group
            or not all(
                math.isfinite(value)
                for value in (
                    experimental,
                    prediction,
                    signed_error,
                    absolute_error,
                )
            )
            or not math.isclose(
                signed_error,
                expected_signed_error,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                absolute_error,
                abs(expected_signed_error),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(f"{label} per-record accuracy evidence is incomplete.")
        signature.append((record_id, input_sha256, experimental, primary_group))
        references.append(experimental)
        predictions.append(prediction)
        primary_groups.append(primary_group)
    if (
        len(signature) != record_count
        or len({item[0] for item in signature}) != record_count
        or len(set(signature)) != record_count
        or set(primary_groups) != set(normalized_observed_groups)
    ):
        raise ValueError(
            f"{label} per-record accuracy evidence has missing or duplicate rows."
        )
    recomputed_metrics = _metric_bundle(
        np.asarray(references, dtype=float),
        np.asarray(predictions, dtype=float),
    )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} accuracy summary lacks formal accuracy metrics.")
    for metric_name in (
        "mae_kcal_mol",
        "rmse_kcal_mol",
        "median_absolute_error_kcal_mol",
        "maximum_absolute_error_kcal_mol",
        "mean_signed_error_kcal_mol",
    ):
        try:
            declared_value = float(metrics[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} accuracy summary lacks finite formal accuracy metrics."
            ) from exc
        expected_value = recomputed_metrics[metric_name]
        if not math.isfinite(declared_value) or not math.isclose(
            declared_value,
            expected_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{label} declared accuracy metrics do not match its per-record "
                "evidence."
            )
    if not recomputed_metrics["maximum_absolute_error_kcal_mol"] < limit:
        raise ValueError(
            f"{label} is not eligible for performance admission because its "
            "formal accuracy gate failed."
        )
    return task_fingerprint, target_quantity, precision_policy, tuple(signature)


def _validated_runtime_samples(
    values: Sequence[float],
    *,
    field: str,
    minimum_count: int,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of timing samples.")
    samples: list[float] = []
    for value in values:
        try:
            sample = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite positive seconds.") from exc
        if not math.isfinite(sample) or sample <= 0:
            raise ValueError(f"{field} must contain finite positive seconds.")
        samples.append(sample)
    if len(samples) < minimum_count:
        raise ValueError(f"{field} requires at least {minimum_count} timing samples.")
    return tuple(samples)


def evaluate_matched_qm_performance(
    candidate_identity: BenchmarkIdentity,
    qm_identity: BenchmarkIdentity,
    *,
    maximum_absolute_error_limit_kcal_mol: float,
    candidate_full_task_seconds: Sequence[float],
    qm_full_task_seconds: Sequence[float],
    candidate_cold_start_seconds: Sequence[float] | None = None,
    qm_cold_start_seconds: Sequence[float] | None = None,
    candidate_warm_state_seconds: Sequence[float] | None = None,
    qm_warm_state_seconds: Sequence[float] | None = None,
    candidate_scope_components: Iterable[str],
    qm_scope_components: Iterable[str],
    accepted_candidate_precision_policy: str,
    timed_candidate_precision_policy: str,
    accepted_qm_precision_policy: str,
    timed_qm_precision_policy: str,
    hardware_fingerprint: str,
) -> dict[str, Any]:
    """Evaluate the strict matched-QM speed policy after formal accuracy passes.

    Formal accuracy evidence is rerun from source-verifying benchmark identities.
    Caller-supplied summary mappings are deliberately rejected because, without
    a signing service, their task fingerprints and group labels are forgeable.

    The caller-supplied timing arrays are sufficient to reject a slow method,
    but not to approve production performance.  Positive admission additionally
    requires a future model-specific trusted runner whose hash-bound receipts
    prove the executed commands, inputs, environment, hardware, and complete
    task boundary.
    """

    if not isinstance(candidate_identity, BenchmarkIdentity) or not isinstance(
        qm_identity,
        BenchmarkIdentity,
    ):
        raise TypeError(
            "Matched-QM performance evaluation requires candidate and QM "
            "BenchmarkIdentity inputs so trusted formal accuracy can be rerun."
        )
    if (
        candidate_identity.matched_task_fingerprint
        != qm_identity.matched_task_fingerprint
    ):
        raise ValueError(
            "Matched-QM performance admission requires the same trusted benchmark "
            "task; geometry, sampling, estimator, records, molecular inputs, and "
            "experimental references must match."
        )
    candidate_accuracy_summary = run_accuracy_panel(
        candidate_identity,
        bootstrap_samples=0,
    )
    qm_accuracy_summary = run_accuracy_panel(
        qm_identity,
        bootstrap_samples=0,
    )
    (
        candidate_task,
        candidate_quantity,
        trusted_candidate_precision,
        candidate_records,
    ) = _accuracy_signature_for_performance(
        candidate_accuracy_summary,
        maximum_absolute_error_limit_kcal_mol=(maximum_absolute_error_limit_kcal_mol),
        label="candidate",
    )
    (
        qm_task,
        qm_quantity,
        trusted_qm_precision,
        qm_records,
    ) = _accuracy_signature_for_performance(
        qm_accuracy_summary,
        maximum_absolute_error_limit_kcal_mol=(maximum_absolute_error_limit_kcal_mol),
        label="QM baseline",
    )
    if (
        candidate_task != qm_task
        or candidate_quantity != qm_quantity
        or candidate_records != qm_records
    ):
        raise ValueError(
            "Matched-QM performance admission requires the same frozen panel, "
            "record order, molecular-input hashes, and experimental references."
        )

    def normalized_scope(values: Iterable[str], *, label: str) -> frozenset[str]:
        scope = frozenset(str(value).strip() for value in values)
        if scope != END_TO_END_RUNTIME_SCOPE_COMPONENTS:
            raise ValueError(
                f"{label} runtime scope must include exactly the full end-to-end "
                "task boundary."
            )
        return scope

    normalized_scope(candidate_scope_components, label="candidate")
    normalized_scope(qm_scope_components, label="QM baseline")

    accepted_candidate_precision = str(accepted_candidate_precision_policy).strip()
    timed_candidate_precision = str(timed_candidate_precision_policy).strip()
    accepted_qm_precision = str(accepted_qm_precision_policy).strip()
    timed_qm_precision = str(timed_qm_precision_policy).strip()
    if (
        accepted_candidate_precision not in MATCHED_PERFORMANCE_PRECISION_POLICIES
        or accepted_qm_precision not in MATCHED_PERFORMANCE_PRECISION_POLICIES
        or accepted_candidate_precision != trusted_candidate_precision
        or accepted_qm_precision != trusted_qm_precision
        or timed_candidate_precision != accepted_candidate_precision
        or timed_qm_precision != accepted_qm_precision
        or accepted_candidate_precision != accepted_qm_precision
    ):
        raise ValueError(
            "Timed precision policies must be explicit float32 or float64 and "
            "exactly match the accuracy-accepted policies, and both methods must "
            "use the same precision policy; "
            "reduced-precision speedups are forbidden."
        )
    normalized_hardware_fingerprint = str(hardware_fingerprint).strip()
    if not normalized_hardware_fingerprint:
        raise ValueError("hardware_fingerprint must be explicit and non-empty.")

    candidate_full = _validated_runtime_samples(
        candidate_full_task_seconds,
        field="candidate_full_task_seconds",
        minimum_count=MINIMUM_MATCHED_RUNTIME_REPEATS,
    )
    qm_full = _validated_runtime_samples(
        qm_full_task_seconds,
        field="qm_full_task_seconds",
        minimum_count=MINIMUM_MATCHED_RUNTIME_REPEATS,
    )

    def optional_runtime_pair(
        candidate_values: Sequence[float] | None,
        qm_values: Sequence[float] | None,
        *,
        candidate_field: str,
        qm_field: str,
        minimum_count: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if candidate_values is None and qm_values is None:
            return None
        if candidate_values is None or qm_values is None:
            raise ValueError(
                f"{candidate_field} and {qm_field} must be supplied together."
            )
        return (
            _validated_runtime_samples(
                candidate_values,
                field=candidate_field,
                minimum_count=minimum_count,
            ),
            _validated_runtime_samples(
                qm_values,
                field=qm_field,
                minimum_count=minimum_count,
            ),
        )

    cold_samples = optional_runtime_pair(
        candidate_cold_start_seconds,
        qm_cold_start_seconds,
        candidate_field="candidate_cold_start_seconds",
        qm_field="qm_cold_start_seconds",
        minimum_count=1,
    )
    warm_samples = optional_runtime_pair(
        candidate_warm_state_seconds,
        qm_warm_state_seconds,
        candidate_field="candidate_warm_state_seconds",
        qm_field="qm_warm_state_seconds",
        minimum_count=MINIMUM_MATCHED_RUNTIME_REPEATS,
    )

    def timing_bundle(
        candidate: tuple[float, ...],
        qm: tuple[float, ...],
    ) -> dict[str, Any]:
        candidate_median = float(np.median(candidate))
        qm_median = float(np.median(qm))
        return {
            "candidate_seconds": list(candidate),
            "qm_seconds": list(qm),
            "candidate_median_seconds": candidate_median,
            "qm_median_seconds": qm_median,
            "speedup_qm_over_candidate": qm_median / candidate_median,
        }

    full_task = timing_bundle(candidate_full, qm_full)
    cold_start = timing_bundle(*cold_samples) if cold_samples is not None else None
    warm_state = timing_bundle(*warm_samples) if warm_samples is not None else None
    full_task_passed = full_task["speedup_qm_over_candidate"] > 1.0
    admission_status = (
        "blocked_pending_trusted_timing_receipts"
        if full_task_passed
        else "rejected_not_faster_than_qm"
    )
    return {
        "schema_version": 1,
        "matched_task_fingerprint": candidate_task,
        "record_count": len(candidate_records),
        "accuracy_precondition_passed": True,
        "maximum_absolute_error_limit_kcal_mol": float(
            maximum_absolute_error_limit_kcal_mol
        ),
        "hardware_fingerprint": normalized_hardware_fingerprint,
        "precision_policy": {
            "candidate": accepted_candidate_precision,
            "qm": accepted_qm_precision,
            "timed_matches_accuracy_accepted": True,
        },
        "end_to_end_scope_components": sorted(END_TO_END_RUNTIME_SCOPE_COMPONENTS),
        "full_task": {
            **full_task,
            "strict_speedup_required": ">1",
            "gate_passed": full_task_passed,
        },
        "cold_start_diagnostic": cold_start,
        "warm_state_diagnostic": warm_state,
        "timing_evidence_status": "caller_supplied_policy_inputs_not_admission_receipts",
        "performance_policy_gate_passed": full_task_passed,
        "performance_admission_passed": False,
        "admission_status": admission_status,
        "blocking_reason": (
            "positive speed policy result lacks model-specific trusted timing "
            "receipts"
            if full_task_passed
            else None
        ),
        "rejection_reason": (
            None
            if full_task_passed
            else "matched end-to-end candidate runtime is not faster than QM"
        ),
    }


class BenchmarkResultStore:
    """Atomic, path-safe writer for the Route-4 benchmark artifact schema."""

    REQUIRED_ARTIFACTS = frozenset(
        {
            "model_card.yaml",
            "protocol.yaml",
            "environment.lock",
            "per_record.csv",
            "failures.csv",
            "diagnostics.json",
            "uncertainty.csv",
            "leakage_report.json",
            "speed.json",
            "summary.json",
        }
    )

    def __init__(self, root: str | Path, model_id: str) -> None:
        if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
            raise ValueError("model_id must be one path-safe directory name.")
        self.directory = Path(root) / model_id
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_filename(filename: str) -> str:
        normalized = str(filename).strip()
        if (
            not normalized
            or Path(normalized).name != normalized
            or normalized in {".", ".."}
        ):
            raise ValueError("artifact filename must be one path-safe file name.")
        return normalized

    def _atomic_text(self, filename: str, text: str) -> Path:
        filename = self._validate_filename(filename)
        target = self.directory / filename
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as handle:
            handle.write(text)
            temp_name = handle.name
        os.replace(temp_name, target)
        return target

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        return self._atomic_text(
            filename,
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )

    def write_json_yaml(self, filename: str, payload: Mapping[str, Any]) -> Path:
        if not filename.endswith((".yaml", ".yml")):
            raise ValueError("JSON-compatible YAML filename must end in .yaml or .yml.")
        return self.write_json(filename, payload)

    def write_csv(
        self,
        filename: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str],
    ) -> Path:
        filename = self._validate_filename(filename)
        if not fieldnames:
            raise ValueError("CSV fieldnames must not be empty.")
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            temp_name = handle.name
        target = self.directory / filename
        os.replace(temp_name, target)
        return target

    def write_environment_lock(self, text: str) -> Path:
        if not text.strip():
            raise ValueError("environment.lock content must not be empty.")
        return self._atomic_text("environment.lock", text.rstrip() + "\n")

    def missing_required_artifacts(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name in self.REQUIRED_ARTIFACTS
                if not (self.directory / name).is_file()
            )
        )

    def assert_complete(self) -> None:
        missing = self.missing_required_artifacts()
        if missing:
            raise RuntimeError(
                "Benchmark result directory is incomplete; missing: "
                + ", ".join(missing)
            )
