"""Pinned, read-only FreeSolv water-box builder for FeNNix HFE mechanics.

This module deliberately uses the FreeSolv GROMACS files only as a source of
atom ordering, coordinates, molecule grouping, and the periodic cell.  It does
not parse or use GAFF energies, force constants, Lennard-Jones parameters, or
partial charges.
"""

from __future__ import annotations

import hashlib
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .system import validate_fennix_alchemical_system
from .types import FeNNixAlchemicalSystem

PINNED_FREESOLV_COMMIT = "6c7d19b4b565537365ffd22006aa2cd4643200c6"
PINNED_SOLVATED_ARCHIVE_SHA256 = (
    "170707f608e12ae46c04997004a2534132a0bbd16f3edd5ff6b1fb2890fdc98f"
)
PINNED_MOL2_ARCHIVE_SHA256 = (
    "15ece6114442a8eb5e720c0ad25bb48e4a6a35392da36f72b80a6f2f415fd133"
)

_SOLVATED_ROOT = "gromacs_solvated"
_MOL2_ROOT = "mol2files_gaff"
_KNOWN_SOLVATED_METADATA_MEMBER = "gromacs_solvated/Icon\r"
_COMPOUND_RE = re.compile(r"mobley_[0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOLVATED_MEMBER_RE = re.compile(r"gromacs_solvated/mobley_[0-9]+\.(?:gro|top)")
_MOL2_MEMBER_RE = re.compile(r"mol2files_gaff/mobley_[0-9]+\.mol2")
_SECTION_RE = re.compile(r"^\[\s*([A-Za-z_]+)\s*\]$")
_ELEMENTS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


@dataclass(frozen=True)
class FreeSolvMemberPin:
    """Exact content identity for one regular archive member."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_member_name(self.name)
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("FreeSolv member sha256 must be 64 lowercase hex digits.")


@dataclass(frozen=True)
class FreeSolvSolvatedManifest:
    """Output-blind identity contract for one pinned solvated FreeSolv input."""

    compound_id: str
    gro: FreeSolvMemberPin
    topology: FreeSolvMemberPin
    source_commit: str = PINNED_FREESOLV_COMMIT
    solvated_archive_sha256: str = PINNED_SOLVATED_ARCHIVE_SHA256
    mol2: FreeSolvMemberPin | None = None
    mol2_archive_sha256: str | None = None

    def __post_init__(self) -> None:
        if _COMPOUND_RE.fullmatch(self.compound_id) is None:
            raise ValueError("FreeSolv compound_id must have form mobley_<digits>.")
        if self.source_commit != PINNED_FREESOLV_COMMIT:
            raise ValueError("FreeSolv manifest must bind the pinned source commit.")
        if self.solvated_archive_sha256 != PINNED_SOLVATED_ARCHIVE_SHA256:
            raise ValueError("FreeSolv manifest must bind the pinned solvated archive.")
        expected_gro = f"{_SOLVATED_ROOT}/{self.compound_id}.gro"
        expected_top = f"{_SOLVATED_ROOT}/{self.compound_id}.top"
        if self.gro.name != expected_gro or self.topology.name != expected_top:
            raise ValueError(
                "FreeSolv GRO/topology member names must match compound_id exactly."
            )
        if self.mol2 is None:
            if self.mol2_archive_sha256 is not None:
                raise ValueError(
                    "A mol2 archive hash requires an exact mol2 member pin."
                )
        else:
            if self.mol2.name != f"{_MOL2_ROOT}/{self.compound_id}.mol2":
                raise ValueError(
                    "FreeSolv MOL2 member name must match compound_id exactly."
                )
            if self.mol2_archive_sha256 != PINNED_MOL2_ARCHIVE_SHA256:
                raise ValueError("FreeSolv manifest must bind the pinned MOL2 archive.")


@dataclass(frozen=True)
class FreeSolvAtomMapping:
    """One immutable archive-to-FeNNix atom mapping entry."""

    system_index: int
    gro_atom_number: int
    molecule_id: int
    residue_name: str
    atom_name: str
    atomic_number: int
    role: str


@dataclass(frozen=True)
class FreeSolvBuildReceipt:
    """Content and parser provenance for a constructed FeNNix system."""

    compound_id: str
    source_commit: str
    solvated_archive_sha256: str
    gro_member: FreeSolvMemberPin
    topology_member: FreeSolvMemberPin
    mol2_archive_sha256: str | None
    mol2_member: FreeSolvMemberPin | None
    builder_source_sha256: str
    atom_count: int
    solute_atom_count: int
    water_count: int
    cell_angstrom: tuple[tuple[float, float, float], ...]
    atom_mapping: tuple[FreeSolvAtomMapping, ...]
    topology_usage: str = "atom_and_molecule_ordering_counts_only"
    experimental_labels_used: bool = False
    force_field_parameters_used: bool = False

    def __post_init__(self) -> None:
        if self.experimental_labels_used or self.force_field_parameters_used:
            raise ValueError(
                "FreeSolv builder receipts cannot admit labels or force-field parameters."
            )


@dataclass(frozen=True)
class FreeSolvBuildResult:
    """Validated system plus its complete immutable construction receipt."""

    system: FeNNixAlchemicalSystem
    receipt: FreeSolvBuildReceipt


@dataclass(frozen=True)
class _GroAtom:
    residue_number: int
    residue_name: str
    atom_name: str
    atom_number: int
    coordinates_angstrom: tuple[float, float, float]


@dataclass(frozen=True)
class _TopologyLayout:
    solute_name: str
    solute_atom_names: tuple[str, ...]
    water_count: int


def _validate_member_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("FreeSolv archive member name must be nonempty.")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(f"Unsafe FreeSolv archive member name: {name!r}.")


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _read_verified_archive_member(
    archive_path: Path,
    *,
    archive_sha256: str,
    member_pin: FreeSolvMemberPin,
    archive_kind: str,
) -> bytes:
    observed_archive_sha256 = _sha256_path(archive_path)
    if observed_archive_sha256 != archive_sha256:
        raise ValueError(
            f"FreeSolv {archive_kind} archive sha256 mismatch: "
            f"expected {archive_sha256}, observed {observed_archive_sha256}."
        )

    seen: set[str] = set()
    selected: bytes | None = None
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"Cannot read FreeSolv {archive_kind} archive.") from exc
    with archive:
        for member in archive:
            _validate_member_name(member.name.rstrip("/") or member.name)
            if member.name in seen:
                raise ValueError(f"Duplicate FreeSolv archive member: {member.name!r}.")
            seen.add(member.name)

            if member.isdir():
                expected_root = (
                    _SOLVATED_ROOT if archive_kind == "solvated" else _MOL2_ROOT
                )
                if member.name.rstrip("/") != expected_root:
                    raise ValueError(
                        f"Unexpected FreeSolv archive directory: {member.name!r}."
                    )
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(
                    f"FreeSolv archives admit regular files only: {member.name!r}."
                )

            if archive_kind == "solvated":
                allowed = bool(_SOLVATED_MEMBER_RE.fullmatch(member.name))
                allowed = allowed or member.name == _KNOWN_SOLVATED_METADATA_MEMBER
            else:
                allowed = bool(_MOL2_MEMBER_RE.fullmatch(member.name))
            if not allowed:
                raise ValueError(
                    f"Unexpected FreeSolv {archive_kind} archive member: "
                    f"{member.name!r}."
                )
            if member.name == member_pin.name:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"Cannot read FreeSolv member {member_pin.name!r}."
                    )
                selected = extracted.read()

    if selected is None:
        raise ValueError(f"Missing FreeSolv member {member_pin.name!r}.")
    observed_member_sha256 = hashlib.sha256(selected).hexdigest()
    if observed_member_sha256 != member_pin.sha256:
        raise ValueError(
            f"FreeSolv member sha256 mismatch for {member_pin.name!r}: "
            f"expected {member_pin.sha256}, observed {observed_member_sha256}."
        )
    return selected


def _decode_text(data: bytes, label: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"FreeSolv {label} must be valid UTF-8 text.") from exc
    if "\x00" in text:
        raise ValueError(f"FreeSolv {label} cannot contain NUL bytes.")
    return text


def _parse_gro(
    data: bytes,
) -> tuple[
    tuple[_GroAtom, ...],
    tuple[tuple[float, float, float], ...],
]:
    lines = _decode_text(data, "GRO member").splitlines()
    if len(lines) < 3:
        raise ValueError("FreeSolv GRO member is truncated.")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError("FreeSolv GRO atom count must be an integer.") from exc
    if atom_count <= 0 or len(lines) != atom_count + 3:
        raise ValueError("FreeSolv GRO line count does not match its atom count.")

    atoms: list[_GroAtom] = []
    for offset, line in enumerate(lines[2 : 2 + atom_count], start=1):
        if len(line) < 44:
            raise ValueError(f"FreeSolv GRO atom line {offset} is truncated.")
        try:
            residue_number = int(line[0:5])
            residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_number = int(line[15:20])
            xyz = tuple(float(line[start : start + 8]) * 10.0 for start in (20, 28, 36))
        except ValueError as exc:
            raise ValueError(f"Cannot parse FreeSolv GRO atom line {offset}.") from exc
        if (
            not residue_name
            or not atom_name
            or residue_number <= 0
            or atom_number != offset
            or not all(math.isfinite(value) for value in xyz)
        ):
            raise ValueError(f"Invalid FreeSolv GRO atom line {offset}.")
        atoms.append(
            _GroAtom(
                residue_number,
                residue_name,
                atom_name,
                atom_number,
                (xyz[0], xyz[1], xyz[2]),
            )
        )

    try:
        box = tuple(float(value) * 10.0 for value in lines[-1].split())
    except ValueError as exc:
        raise ValueError("Cannot parse FreeSolv GRO periodic cell.") from exc
    if len(box) == 3:
        cell = ((box[0], 0.0, 0.0), (0.0, box[1], 0.0), (0.0, 0.0, box[2]))
    elif len(box) == 9:
        cell = (
            (box[0], box[3], box[4]),
            (box[5], box[1], box[6]),
            (box[7], box[8], box[2]),
        )
    else:
        raise ValueError("FreeSolv GRO cell must contain exactly 3 or 9 values.")
    if not all(math.isfinite(value) for row in cell for value in row):
        raise ValueError("FreeSolv GRO periodic cell must be finite.")
    return tuple(atoms), cell


def _parse_topology(data: bytes, compound_id: str) -> _TopologyLayout:
    section = ""
    solute_name: str | None = None
    atom_names: list[str] = []
    molecule_rows: list[tuple[str, int]] = []
    for raw_line in _decode_text(data, "topology member").splitlines():
        content = raw_line.split(";", 1)[0].strip()
        if not content or content.startswith("#"):
            continue
        match = _SECTION_RE.fullmatch(content)
        if match is not None:
            section = match.group(1).lower()
            continue
        fields = content.split()
        if section == "moleculetype" and solute_name is None:
            if len(fields) != 2 or fields[0] != compound_id:
                raise ValueError("FreeSolv topology has an unexpected solute molecule.")
            solute_name = fields[0]
        elif section == "atoms":
            if len(fields) < 5:
                raise ValueError("Malformed FreeSolv topology atom ordering row.")
            try:
                atom_index = int(fields[0])
            except ValueError as exc:
                raise ValueError("Malformed FreeSolv topology atom index.") from exc
            if atom_index != len(atom_names) + 1:
                raise ValueError("FreeSolv topology atom ordering is not contiguous.")
            atom_names.append(fields[4])
        elif section == "molecules":
            if len(fields) != 2:
                raise ValueError("Malformed FreeSolv topology molecule count row.")
            try:
                count = int(fields[1])
            except ValueError as exc:
                raise ValueError("Malformed FreeSolv topology molecule count.") from exc
            if count <= 0:
                raise ValueError("FreeSolv topology molecule counts must be positive.")
            molecule_rows.append((fields[0], count))

    if solute_name != compound_id or not atom_names:
        raise ValueError("FreeSolv topology lacks one explicit solute atom ordering.")
    if (
        len(molecule_rows) != 2
        or molecule_rows[0] != (compound_id, 1)
        or molecule_rows[1][0] != "SOL"
    ):
        raise ValueError(
            "FreeSolv topology must contain exactly one solute followed by water."
        )
    return _TopologyLayout(compound_id, tuple(atom_names), molecule_rows[1][1])


def _atomic_number(atom_name: str) -> int:
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from atom name {atom_name!r}.")
    normalized = letters[0].upper() + letters[1:].lower()
    for length in (2, 1):
        symbol = normalized[:length]
        if symbol in _ELEMENTS:
            return _ELEMENTS[symbol]
    raise ValueError(f"Unsupported or ambiguous atom name {atom_name!r}.")


def _parse_mol2_atomic_numbers(data: bytes) -> tuple[int, ...]:
    lines = _decode_text(data, "MOL2 member").splitlines()
    try:
        start = lines.index("@<TRIPOS>ATOM") + 1
    except ValueError as exc:
        raise ValueError("FreeSolv MOL2 member lacks an ATOM section.") from exc
    numbers: list[int] = []
    expected_index = 1
    for line in lines[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError("Malformed FreeSolv MOL2 atom row.")
        try:
            atom_index = int(fields[0])
        except ValueError as exc:
            raise ValueError("Malformed FreeSolv MOL2 atom index.") from exc
        if atom_index != expected_index:
            raise ValueError("FreeSolv MOL2 atom ordering is not contiguous.")
        numbers.append(_atomic_number(fields[1]))
        expected_index += 1
    if not numbers:
        raise ValueError("FreeSolv MOL2 member has no atoms.")
    return tuple(numbers)


def _group_gro_molecules(
    atoms: tuple[_GroAtom, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    groups: list[list[int]] = []
    group_keys: list[tuple[int, str]] = []
    for index, atom in enumerate(atoms):
        key = (atom.residue_number, atom.residue_name)
        if not group_keys or key != group_keys[-1]:
            if key in group_keys:
                raise ValueError("FreeSolv GRO molecule residues must be contiguous.")
            group_keys.append(key)
            groups.append([])
        groups[-1].append(index)
    molecule_ids = tuple(
        molecule_id for molecule_id, indices in enumerate(groups) for _ in indices
    )
    return molecule_ids, tuple(tuple(indices) for indices in groups)


def build_fennix_system_from_freesolv(
    manifest: FreeSolvSolvatedManifest,
    solvated_archive_path: str | Path,
    *,
    mol2_archive_path: str | Path | None = None,
    neighbor_skin_angstrom: float = 1.0,
) -> FreeSolvBuildResult:
    """Construct and validate one pinned FreeSolv system without extraction writes."""

    if not isinstance(manifest, FreeSolvSolvatedManifest):
        raise TypeError("manifest must be a FreeSolvSolvatedManifest.")
    solvated_path = Path(solvated_archive_path)
    gro_data = _read_verified_archive_member(
        solvated_path,
        archive_sha256=manifest.solvated_archive_sha256,
        member_pin=manifest.gro,
        archive_kind="solvated",
    )
    topology_data = _read_verified_archive_member(
        solvated_path,
        archive_sha256=manifest.solvated_archive_sha256,
        member_pin=manifest.topology,
        archive_kind="solvated",
    )
    mol2_data: bytes | None = None
    if manifest.mol2 is not None:
        if mol2_archive_path is None:
            raise ValueError("Pinned MOL2 validation requires mol2_archive_path.")
        assert manifest.mol2_archive_sha256 is not None
        mol2_data = _read_verified_archive_member(
            Path(mol2_archive_path),
            archive_sha256=manifest.mol2_archive_sha256,
            member_pin=manifest.mol2,
            archive_kind="mol2",
        )
    elif mol2_archive_path is not None:
        raise ValueError("mol2_archive_path requires an exact MOL2 manifest pin.")

    atoms, cell = _parse_gro(gro_data)
    topology = _parse_topology(topology_data, manifest.compound_id)
    molecule_ids, molecule_groups = _group_gro_molecules(atoms)
    if len(molecule_groups) != topology.water_count + 1:
        raise ValueError("FreeSolv GRO molecule count disagrees with topology.")
    solute_indices = molecule_groups[0]
    if tuple(atoms[index].atom_name for index in solute_indices) != (
        topology.solute_atom_names
    ):
        raise ValueError("FreeSolv GRO solute ordering disagrees with topology.")
    if any(atoms[index].residue_name != "MOL" for index in solute_indices):
        raise ValueError("FreeSolv GRO first molecule must be the MOL solute.")

    atomic_numbers: list[int] = []
    for group_number, indices in enumerate(molecule_groups):
        if group_number == 0:
            atomic_numbers.extend(
                _atomic_number(atoms[index].atom_name) for index in indices
            )
            continue
        names = tuple(atoms[index].atom_name for index in indices)
        residues = tuple(atoms[index].residue_name for index in indices)
        if names != ("OW", "HW1", "HW2") or residues != ("SOL", "SOL", "SOL"):
            raise ValueError(
                "Each FreeSolv solvent molecule must be ordered as SOL OW/HW1/HW2."
            )
        atomic_numbers.extend((8, 1, 1))

    if mol2_data is not None:
        mol2_numbers = _parse_mol2_atomic_numbers(mol2_data)
        solute_numbers = tuple(atomic_numbers[index] for index in solute_indices)
        if mol2_numbers != solute_numbers:
            raise ValueError(
                "FreeSolv MOL2 solute atomic identities/order disagree with GRO/topology."
            )

    system = validate_fennix_alchemical_system(
        FeNNixAlchemicalSystem(
            atomic_numbers=tuple(atomic_numbers),
            coordinates_angstrom=tuple(atom.coordinates_angstrom for atom in atoms),
            cell_angstrom=cell,
            molecule_ids=molecule_ids,
            solute_atom_indices=solute_indices,
            neighbor_skin_angstrom=neighbor_skin_angstrom,
        )
    )
    mapping = tuple(
        FreeSolvAtomMapping(
            system_index=index,
            gro_atom_number=atom.atom_number,
            molecule_id=system.molecule_ids[index],
            residue_name=atom.residue_name,
            atom_name=atom.atom_name,
            atomic_number=system.atomic_numbers[index],
            role="solute" if index in solute_indices else "water",
        )
        for index, atom in enumerate(atoms)
    )
    receipt = FreeSolvBuildReceipt(
        compound_id=manifest.compound_id,
        source_commit=manifest.source_commit,
        solvated_archive_sha256=manifest.solvated_archive_sha256,
        gro_member=manifest.gro,
        topology_member=manifest.topology,
        mol2_archive_sha256=manifest.mol2_archive_sha256,
        mol2_member=manifest.mol2,
        builder_source_sha256=_sha256_path(Path(__file__)),
        atom_count=len(system.atomic_numbers),
        solute_atom_count=len(system.solute_atom_indices),
        water_count=topology.water_count,
        cell_angstrom=system.cell_angstrom,
        atom_mapping=mapping,
    )
    return FreeSolvBuildResult(system=system, receipt=receipt)
