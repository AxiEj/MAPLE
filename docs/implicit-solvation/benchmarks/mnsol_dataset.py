#!/usr/bin/env python3
"""License-aware MNSol-v2012 ingestion and aggregate coverage auditing.

The MNSol distribution contains unpublished material and is not redistributed
by MAPLE.  This module reads a user-supplied archive or extracted directory,
verifies the pinned table contract, and emits aggregate metadata only.  It is
deliberately independent of every continuum provider and solute model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
from itertools import islice
import json
import math
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Mapping, Sequence
from zipfile import ZipFile, ZipInfo

from ase.data import atomic_masses, chemical_symbols, covalent_radii

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic

MNSOL_V2012_TABLE_NAME = "MNSol_alldata.txt"
MNSOL_MAX_SOURCE_ARCHIVE_BYTES = 128 * 1024 * 1024
MNSOL_MAX_ZIP_MEMBERS = 20_000
MNSOL_MAX_ZIP_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MNSOL_MAX_ZIP_COMPRESSION_RATIO = 100.0
MNSOL_MAX_TABLE_BYTES = 8 * 1024 * 1024
MNSOL_MAX_GEOMETRY_COUNT = 5_000
MNSOL_MAX_GEOMETRY_BYTES = 1024 * 1024
MNSOL_MAX_TOTAL_GEOMETRY_BYTES = 64 * 1024 * 1024
MNSOL_V2012_HEADER = (
    "No.",
    "FileHandle",
    "SoluteName",
    "Formula",
    "Subset",
    "Charge",
    "Level1",
    "Level2",
    "Level3",
    "Solvent",
    "DeltaGsolv",
    "type",
    "eps",
    "n",
    "alpha",
    "beta",
    "gamma",
    "phi**2",
    "psi**2",
    "beta**2",
    "H",
    "C",
    "HC",
    "CC",
    "CC2",
    "N",
    "HN",
    "HN2",
    "CN",
    "NC",
    "NC2",
    "NC3",
    "O",
    "HO",
    "HO2",
    "OC",
    "CO2",
    "ON",
    "OO",
    "F",
    "Cl",
    "Br",
    "I",
    "FC",
    "ClC",
    "BrC",
    "IC",
    "Si",
    "OSi",
    "P",
    "HP",
    "OP",
    "S",
    "HS",
    "OS",
    "SP",
    "SS",
    "TotalArea",
)
MNSOL_V2012_SUBSETS = frozenset(f"[{letter}]" for letter in "abcdefghijk")
MNSOL_V2012_PROCESS_TYPES = frozenset({"abs", "rel"})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_protocol_value(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"MNSol protocol requires a 64-character {field} SHA256.")
    return digest


def _finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"MNSol row {row_number} has a non-numeric {field}: {value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"MNSol row {row_number} has a non-finite {field}: {value!r}.")
    return parsed


def _integer(value: str, *, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"MNSol row {row_number} has a non-integer {field}: {value!r}."
        ) from exc
    return parsed


@dataclass(frozen=True)
class MNSolPanelSolvent:
    """One dataset-facing solvent identity in the frozen validation panel."""

    canonical_name: str
    mnsol_name: str
    neutral_absolute_validation: bool
    expected_all_records: int
    expected_neutral_absolute_records: int
    rationale: str


@dataclass(frozen=True)
class MNSolProtocol:
    """Validated immutable subset of the tracked MNSol protocol."""

    protocol_id: str
    fingerprint: str
    expected_table_sha256: str
    expected_normalized_bundle_sha256: str
    expected_record_count: int
    expected_unique_solutes: int
    expected_absolute_solvents: int
    expected_transfer_labels: int
    temperature_k: float
    standard_state: str
    molecular_charge: int
    multiplicity: int
    process_type: str
    geometry_policy: str
    allowed_atomic_numbers: frozenset[int]
    molecular_mass_da: tuple[float, float]
    required_component_count: int
    connectedness_bond_scale: float
    partition_strategy: str
    development_fraction: float
    partition_seed: str
    solvent_generalization_strategy: str
    pooled_random_split_is_not_sufficient: bool
    small_n_policy: str
    panel: tuple[MNSolPanelSolvent, ...]


@dataclass(frozen=True)
class MNSolGeometry:
    """One MNSol gas-phase geometry in the distribution's native format."""

    handle: str
    formula: str
    geometry_level_tag: str
    charge: int
    multiplicity: int
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    sha256: str

    @property
    def molecular_mass_da(self) -> float:
        return float(sum(atomic_masses[number] for number in self.atomic_numbers))

    @property
    def elements(self) -> frozenset[str]:
        return frozenset(chemical_symbols[number] for number in self.atomic_numbers)


@dataclass(frozen=True)
class MNSolRecord:
    """The public scientific fields of one MNSol table row."""

    entry_number: int
    geometry_handle: str
    solute_name: str
    formula: str
    subset: str
    charge: int
    classification: tuple[int, int, int]
    solvent: str
    delta_g_kcal_mol: float
    process_type: str
    solvent_descriptors: tuple[float, ...]
    raw_row_sha256: str


@dataclass(frozen=True)
class MNSolDataset:
    """A verified in-memory MNSol dataset without provider dependencies."""

    records: tuple[MNSolRecord, ...]
    geometries: Mapping[str, MNSolGeometry]
    table_sha256: str
    normalized_bundle_sha256: str
    source_artifact_sha256: str | None
    source_kind: str


@dataclass(frozen=True)
class MNSolEligibleRecord:
    """One Route-2-domain row with its geometry and frozen partition."""

    canonical_solvent: str
    record: MNSolRecord
    geometry: MNSolGeometry
    partition: str


def load_mnsol_protocol(path: str | Path) -> MNSolProtocol:
    """Load and strictly validate the tracked MNSol-v2012 protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MNSol protocol must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Only MNSol protocol schema version 1 is supported.")
    if payload.get("benchmark_kind") != "mnsol-v2012-dataset-protocol":
        raise ValueError(
            "MNSol protocol benchmark_kind must be " "'mnsol-v2012-dataset-protocol'."
        )
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("version") != "2012":
        raise ValueError("MNSol protocol must pin dataset version 2012.")
    if dataset.get("acquisition_policy") != "user-supplied-no-auto-download":
        raise ValueError(
            "MNSol acquisition policy must remain user-supplied-no-auto-download."
        )
    protocol_id = str(payload.get("protocol_id", "")).strip()
    if not protocol_id:
        raise ValueError("MNSol protocol_id must be non-empty.")
    expected_sha256 = _sha256_protocol_value(dataset.get("table_sha256"), field="table")
    expected_bundle_sha256 = _sha256_protocol_value(
        dataset.get("normalized_bundle_sha256"), field="normalized-bundle"
    )
    temperature_k = dataset.get("temperature_k")
    if (
        not isinstance(temperature_k, (int, float))
        or isinstance(temperature_k, bool)
        or float(temperature_k) != 298.0
    ):
        raise ValueError("MNSol protocol temperature_k must remain 298.0 K.")
    standard_state = str(dataset.get("standard_state", "")).strip()
    if standard_state != "1M-ideal-gas-to-1M-ideal-solution":
        raise ValueError(
            "MNSol protocol standard_state must remain "
            "'1M-ideal-gas-to-1M-ideal-solution'."
        )

    domain = payload.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("MNSol protocol requires a domain object.")
    molecular_charge = domain.get("molecular_charge")
    multiplicity = domain.get("multiplicity")
    process_type = str(domain.get("process_type", "")).strip()
    if (
        not isinstance(molecular_charge, int)
        or isinstance(molecular_charge, bool)
        or molecular_charge != 0
    ):
        raise ValueError("MNSol Route-2 protocol molecular_charge must remain 0.")
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity != 1
    ):
        raise ValueError("MNSol Route-2 protocol multiplicity must remain 1.")
    if process_type != "abs":
        raise ValueError("MNSol Route-2 protocol process_type must remain 'abs'.")
    geometry_policy = str(domain.get("geometry_policy", "")).strip()
    if geometry_policy != "one-fixed-MNSol-M06-2X-MG3S-gas-geometry-no-optimization":
        raise ValueError(
            "MNSol geometry_policy must retain the fixed M06-2X/MG3S geometry."
        )
    allowed_elements = domain.get("allowed_elements")
    if not isinstance(allowed_elements, list) or not allowed_elements:
        raise ValueError("MNSol protocol allowed_elements must be a non-empty list.")
    try:
        allowed_atomic_numbers = frozenset(
            chemical_symbols.index(str(symbol)) for symbol in allowed_elements
        )
    except ValueError as exc:
        raise ValueError("MNSol protocol contains an unknown element symbol.") from exc
    mass_domain = domain.get("molecular_mass_da")
    if (
        not isinstance(mass_domain, list)
        or len(mass_domain) != 2
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in mass_domain
        )
        or not 0.0 < float(mass_domain[0]) < float(mass_domain[1])
    ):
        raise ValueError("MNSol molecular_mass_da must be [positive_min, max].")
    connectedness = domain.get("connectedness")
    if not isinstance(connectedness, dict):
        raise ValueError("MNSol protocol requires a connectedness object.")
    required_component_count = connectedness.get("required_component_count")
    if (
        not isinstance(required_component_count, int)
        or isinstance(required_component_count, bool)
        or required_component_count != 1
    ):
        raise ValueError(
            "MNSol Route-2 protocol requires exactly one covalent component."
        )
    if connectedness.get("radius_source") != "ase.data.covalent_radii":
        raise ValueError(
            "MNSol connectedness radius_source must be " "'ase.data.covalent_radii'."
        )
    connectedness_bond_scale = connectedness.get("bond_scale")
    if (
        not isinstance(connectedness_bond_scale, (int, float))
        or isinstance(connectedness_bond_scale, bool)
        or not 1.0 <= float(connectedness_bond_scale) <= 1.5
    ):
        raise ValueError("MNSol connectedness bond_scale must be in [1.0, 1.5].")

    partition = payload.get("partition")
    if not isinstance(partition, dict):
        raise ValueError("MNSol protocol requires a partition object.")
    partition_strategy = str(partition.get("strategy", "")).strip()
    if partition_strategy != "sha256-geometry-handle-group-v1":
        raise ValueError(
            "MNSol partition strategy must remain " "'sha256-geometry-handle-group-v1'."
        )
    if partition.get("group_key") != "FileHandle":
        raise ValueError("MNSol partitions must group all rows by FileHandle.")
    fraction = partition.get("development_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0.0 < float(fraction) < 1.0
    ):
        raise ValueError(
            "MNSol development_fraction must be strictly between zero and one."
        )
    seed = str(partition.get("seed", "")).strip()
    if not seed:
        raise ValueError("MNSol partition seed must be non-empty.")
    solvent_generalization = partition.get("solvent_generalization")
    if not isinstance(solvent_generalization, dict):
        raise ValueError("MNSol partition requires a solvent_generalization object.")
    solvent_generalization_strategy = str(
        solvent_generalization.get("strategy", "")
    ).strip()
    if (
        solvent_generalization_strategy
        != "per-solvent-reporting-plus-leave-one-solvent-out-sensitivity"
    ):
        raise ValueError(
            "MNSol solvent-generalization strategy must retain per-solvent "
            "reporting plus leave-one-solvent-out sensitivity."
        )
    pooled_random_split_is_not_sufficient = solvent_generalization.get(
        "pooled_random_split_is_not_sufficient"
    )
    if pooled_random_split_is_not_sufficient is not True:
        raise ValueError(
            "MNSol protocol must state that a pooled random split is insufficient."
        )
    small_n_policy = str(solvent_generalization.get("small_n_policy", "")).strip()
    if not small_n_policy:
        raise ValueError("MNSol solvent-generalization small_n_policy is required.")

    expected_record_count = int(dataset.get("expected_record_count", -1))
    expected_unique_solutes = int(dataset.get("expected_unique_solutes", -1))
    expected_absolute_solvents = int(dataset.get("expected_absolute_solvents", -1))
    expected_transfer_labels = int(dataset.get("expected_transfer_labels", -1))
    if (
        min(
            expected_record_count,
            expected_unique_solutes,
            expected_absolute_solvents,
        )
        <= 0
        or expected_transfer_labels < 0
    ):
        raise ValueError(
            "MNSol protocol record, solute, and absolute-solvent counts must be "
            "positive; transfer-label count cannot be negative."
        )

    raw_panel = payload.get("solvent_panel")
    if not isinstance(raw_panel, list) or not raw_panel:
        raise ValueError("MNSol solvent_panel must be a non-empty list.")
    panel: list[MNSolPanelSolvent] = []
    for item in raw_panel:
        if not isinstance(item, dict):
            raise ValueError("Every MNSol solvent_panel entry must be an object.")
        neutral_validation = item.get("neutral_absolute_validation")
        if not isinstance(neutral_validation, bool):
            raise ValueError(
                "MNSol neutral_absolute_validation values must be booleans."
            )
        panel.append(
            MNSolPanelSolvent(
                canonical_name=str(item.get("canonical_name", "")).strip(),
                mnsol_name=str(item.get("mnsol_name", "")).strip(),
                neutral_absolute_validation=neutral_validation,
                expected_all_records=int(item.get("expected_all_records", -1)),
                expected_neutral_absolute_records=int(
                    item.get("expected_neutral_absolute_records", -1)
                ),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    canonical_names = [item.canonical_name for item in panel]
    mnsol_names = [item.mnsol_name for item in panel]
    if (
        any(not name for name in canonical_names + mnsol_names)
        or len(canonical_names) != len(set(canonical_names))
        or len(mnsol_names) != len(set(mnsol_names))
    ):
        raise ValueError("MNSol solvent-panel names must be non-empty and unique.")
    if sum(item.neutral_absolute_validation for item in panel) < 10:
        raise ValueError(
            "MNSol protocol must retain at least ten neutral absolute-validation "
            "solvents."
        )
    for item in panel:
        if item.expected_all_records < 0 or item.expected_neutral_absolute_records < 0:
            raise ValueError("MNSol solvent-panel expected counts cannot be negative.")
        if (
            item.neutral_absolute_validation
            and item.expected_neutral_absolute_records == 0
        ):
            raise ValueError(
                f"Neutral validation solvent {item.canonical_name!r} has no "
                "expected neutral absolute records."
            )

    fingerprint = _sha256_bytes(canonical_json_bytes(payload))
    return MNSolProtocol(
        protocol_id=protocol_id,
        fingerprint=fingerprint,
        expected_table_sha256=expected_sha256,
        expected_normalized_bundle_sha256=expected_bundle_sha256,
        expected_record_count=expected_record_count,
        expected_unique_solutes=expected_unique_solutes,
        expected_absolute_solvents=expected_absolute_solvents,
        expected_transfer_labels=expected_transfer_labels,
        temperature_k=float(temperature_k),
        standard_state=standard_state,
        molecular_charge=molecular_charge,
        multiplicity=multiplicity,
        process_type=process_type,
        geometry_policy=geometry_policy,
        allowed_atomic_numbers=allowed_atomic_numbers,
        molecular_mass_da=(float(mass_domain[0]), float(mass_domain[1])),
        required_component_count=required_component_count,
        connectedness_bond_scale=float(connectedness_bond_scale),
        partition_strategy=partition_strategy,
        development_fraction=float(fraction),
        partition_seed=seed,
        solvent_generalization_strategy=solvent_generalization_strategy,
        pooled_random_split_is_not_sufficient=(pooled_random_split_is_not_sufficient),
        small_n_policy=small_n_policy,
        panel=tuple(panel),
    )


def _parse_geometry(handle: str, payload: bytes) -> MNSolGeometry:
    try:
        lines = payload.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"MNSol geometry {handle!r} is not UTF-8 text.") from exc
    title_fields = lines[0].split() if lines else []
    if len(title_fields) < 3 or title_fields[0] != handle:
        raise ValueError(f"MNSol geometry {handle!r} has a mismatched title line.")
    if title_fields[-1] != "m062x_mg3s_geom":
        raise ValueError(
            f"MNSol geometry {handle!r} does not identify the frozen "
            "M06-2X/MG3S geometry level."
        )
    body = [line.strip() for line in lines[1:] if line.strip()]
    if len(body) < 2:
        raise ValueError(f"MNSol geometry {handle!r} is incomplete.")
    state = body[0].split()
    if len(state) != 2:
        raise ValueError(
            f"MNSol geometry {handle!r} must contain exactly charge and multiplicity."
        )
    charge = _integer(state[0], field="geometry charge", row_number=0)
    multiplicity = _integer(state[1], field="geometry multiplicity", row_number=0)
    if multiplicity <= 0:
        raise ValueError(f"MNSol geometry {handle!r} has invalid multiplicity.")

    atomic_numbers: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(body[1:], start=1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"MNSol geometry {handle!r} coordinate row {line_number} "
                "must have four fields."
            )
        atomic_number = _integer(
            fields[0], field="atomic number", row_number=line_number
        )
        if atomic_number <= 0 or atomic_number >= len(chemical_symbols):
            raise ValueError(
                f"MNSol geometry {handle!r} contains unsupported atomic number "
                f"{atomic_number}."
            )
        xyz = tuple(
            _finite_float(value, field="coordinate", row_number=line_number)
            for value in fields[1:]
        )
        atomic_numbers.append(atomic_number)
        coordinates.append((xyz[0], xyz[1], xyz[2]))
    return MNSolGeometry(
        handle=handle,
        formula=title_fields[1],
        geometry_level_tag=title_fields[-1],
        charge=charge,
        multiplicity=multiplicity,
        atomic_numbers=tuple(atomic_numbers),
        coordinates_angstrom=tuple(coordinates),
        sha256=_sha256_bytes(payload),
    )


def _covalent_component_count(geometry: MNSolGeometry, *, bond_scale: float) -> int:
    atom_count = len(geometry.atomic_numbers)
    neighbours: list[list[int]] = [[] for _ in range(atom_count)]
    for first in range(atom_count):
        first_xyz = geometry.coordinates_angstrom[first]
        first_radius = float(covalent_radii[geometry.atomic_numbers[first]])
        for second in range(first + 1, atom_count):
            second_xyz = geometry.coordinates_angstrom[second]
            threshold = bond_scale * (
                first_radius + float(covalent_radii[geometry.atomic_numbers[second]])
            )
            squared_distance = sum(
                (first_xyz[axis] - second_xyz[axis]) ** 2 for axis in range(3)
            )
            if squared_distance <= threshold**2:
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


def _parse_table(payload: bytes) -> tuple[MNSolRecord, ...]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("MNSol table is not UTF-8 text.") from exc
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    try:
        header = tuple(next(reader))
    except StopIteration as exc:
        raise ValueError("MNSol table is empty.") from exc
    if header != MNSOL_V2012_HEADER:
        raise ValueError("MNSol-v2012 table header does not match the frozen contract.")

    records: list[MNSolRecord] = []
    seen_entries: set[int] = set()
    for physical_row, fields in enumerate(reader, start=2):
        if not fields or all(not field.strip() for field in fields):
            continue
        if len(fields) != len(MNSOL_V2012_HEADER):
            raise ValueError(
                f"MNSol row {physical_row} has {len(fields)} columns; "
                f"expected {len(MNSOL_V2012_HEADER)}."
            )
        entry_number = _integer(
            fields[0], field="database entry number", row_number=physical_row
        )
        if entry_number in seen_entries:
            raise ValueError(f"Duplicate MNSol database entry number: {entry_number}.")
        seen_entries.add(entry_number)
        subset = fields[4].strip()
        if subset not in MNSOL_V2012_SUBSETS:
            raise ValueError(f"MNSol row {physical_row} has invalid subset {subset!r}.")
        process_type = fields[11].strip()
        if process_type not in MNSOL_V2012_PROCESS_TYPES:
            raise ValueError(
                f"MNSol row {physical_row} has invalid process type "
                f"{process_type!r}."
            )
        if (process_type == "rel") != (subset == "[h]"):
            raise ValueError(
                f"MNSol row {physical_row} has inconsistent subset/process type."
            )
        for index, name in enumerate(MNSOL_V2012_HEADER[12:], start=12):
            _finite_float(fields[index], field=name, row_number=physical_row)
        for index, name in (
            (1, "FileHandle"),
            (2, "SoluteName"),
            (3, "Formula"),
            (9, "Solvent"),
        ):
            if not fields[index].strip():
                raise ValueError(f"MNSol row {physical_row} has an empty {name} field.")
        canonical_row = "\t".join(fields).encode("utf-8")
        records.append(
            MNSolRecord(
                entry_number=entry_number,
                geometry_handle=fields[1].strip(),
                solute_name=fields[2].strip(),
                formula=fields[3].strip(),
                subset=subset,
                charge=_integer(
                    fields[5], field="solute charge", row_number=physical_row
                ),
                classification=(
                    _integer(fields[6], field="Level1", row_number=physical_row),
                    _integer(fields[7], field="Level2", row_number=physical_row),
                    _integer(fields[8], field="Level3", row_number=physical_row),
                ),
                solvent=fields[9].strip(),
                delta_g_kcal_mol=_finite_float(
                    fields[10], field="DeltaGsolv", row_number=physical_row
                ),
                process_type=process_type,
                solvent_descriptors=tuple(float(value) for value in fields[12:20]),
                raw_row_sha256=_sha256_bytes(canonical_row),
            )
        )
    return tuple(records)


def _validate_logical_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in MNSol source: {name!r}.")
    return path


def _regular_file_size(path: Path, *, label: str) -> int:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"MNSol {label} must be a regular file.")
    return status.st_size


def _read_path_limited(path: Path, *, limit: int, label: str) -> bytes:
    if _regular_file_size(path, label=label) > limit:
        raise ValueError(f"MNSol {label} exceeds the {limit}-byte safety limit.")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"MNSol {label} exceeds the {limit}-byte safety limit.")
    return payload


def _read_zip_member_limited(
    archive: ZipFile,
    info: ZipInfo,
    *,
    limit: int,
    label: str,
) -> bytes:
    if info.file_size < 0 or info.file_size > limit:
        raise ValueError(f"MNSol {label} exceeds the {limit}-byte safety limit.")
    with archive.open(info, "r") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"MNSol {label} exceeds the {limit}-byte safety limit.")
    return payload


def _normalized_bundle_sha256(
    table_payload: bytes, geometry_payloads: Mapping[str, bytes]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"maple-mnsol-normalized-bundle-v1\0")
    members = {MNSOL_V2012_TABLE_NAME: table_payload}
    members.update(
        {
            f"all_solutes/{handle}.xyz": payload
            for handle, payload in geometry_payloads.items()
        }
    )
    for name in sorted(members):
        payload = members[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _read_source(
    source: Path,
) -> tuple[bytes, dict[str, bytes], str, str | None]:
    if source.is_file() and source.suffix.lower() == ".zip":
        if source.stat().st_size > MNSOL_MAX_SOURCE_ARCHIVE_BYTES:
            raise ValueError(
                "MNSol zip archive exceeds the compressed-source safety limit."
            )
        source_artifact_sha256 = sha256_file(source)
        with ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MNSOL_MAX_ZIP_MEMBERS:
                raise ValueError("MNSol zip archive contains too many members.")
            if sum(max(0, info.file_size) for info in infos) > (
                MNSOL_MAX_ZIP_UNCOMPRESSED_BYTES
            ):
                raise ValueError(
                    "MNSol zip archive exceeds the uncompressed-size safety limit."
                )
            if any(
                info.file_size > 0
                and (
                    info.compress_size <= 0
                    or info.file_size / info.compress_size
                    > MNSOL_MAX_ZIP_COMPRESSION_RATIO
                )
                for info in infos
            ):
                raise ValueError(
                    "MNSol zip archive exceeds the compression-ratio safety limit."
                )
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("MNSol zip archive contains duplicate member paths.")
            entries = [(_validate_logical_path(info.filename), info) for info in infos]
            logical_names = [str(path) for path, _ in entries]
            if len(logical_names) != len(set(logical_names)):
                raise ValueError(
                    "MNSol zip archive contains duplicate logical member paths."
                )
            table_candidates = [
                (path, info)
                for path, info in entries
                if path.name == MNSOL_V2012_TABLE_NAME and "__MACOSX" not in path.parts
            ]
            if len(table_candidates) != 1:
                raise ValueError(
                    "MNSol zip archive must contain exactly one "
                    f"{MNSOL_V2012_TABLE_NAME}."
                )
            table_path, table_info = table_candidates[0]
            geometry_parent = table_path.parent / "all_solutes"
            geometry_entries = [
                (path, info)
                for path, info in entries
                if path.parent == geometry_parent and path.suffix.lower() == ".xyz"
            ]
            if len(geometry_entries) > MNSOL_MAX_GEOMETRY_COUNT:
                raise ValueError("MNSol zip archive contains too many geometries.")
            geometry_stems = [path.stem for path, _ in geometry_entries]
            if len(geometry_stems) != len(set(geometry_stems)):
                raise ValueError(
                    "MNSol zip archive contains duplicate geometry handles."
                )
            if sum(max(0, info.file_size) for _, info in geometry_entries) > (
                MNSOL_MAX_TOTAL_GEOMETRY_BYTES
            ):
                raise ValueError(
                    "MNSol zip archive geometry payload exceeds the safety limit."
                )
            geometry_payloads = {
                path.stem: _read_zip_member_limited(
                    archive,
                    info,
                    limit=MNSOL_MAX_GEOMETRY_BYTES,
                    label=f"geometry {path.stem!r}",
                )
                for path, info in geometry_entries
            }
            table_payload = _read_zip_member_limited(
                archive,
                table_info,
                limit=MNSOL_MAX_TABLE_BYTES,
                label="table",
            )
        return table_payload, geometry_payloads, "zip", source_artifact_sha256

    if source.is_dir():
        table_candidates = list(
            islice(
                (
                    path
                    for path in source.rglob(MNSOL_V2012_TABLE_NAME)
                    if "__MACOSX" not in path.parts
                ),
                2,
            )
        )
        if len(table_candidates) != 1:
            raise ValueError(
                "Extracted MNSol source must contain exactly one "
                f"{MNSOL_V2012_TABLE_NAME}."
            )
        table_path = table_candidates[0]
        geometry_dir = table_path.parent / "all_solutes"
        if not geometry_dir.is_dir():
            raise ValueError("Extracted MNSol source is missing all_solutes/.")
        geometry_paths = list(
            islice(
                geometry_dir.glob("*.xyz"),
                MNSOL_MAX_GEOMETRY_COUNT + 1,
            )
        )
        if len(geometry_paths) > MNSOL_MAX_GEOMETRY_COUNT:
            raise ValueError("Extracted MNSol source contains too many geometries.")
        geometry_stems = [path.stem for path in geometry_paths]
        if len(geometry_stems) != len(set(geometry_stems)):
            raise ValueError(
                "Extracted MNSol source contains duplicate geometry handles."
            )
        if sum(
            max(
                0,
                _regular_file_size(
                    path,
                    label=f"geometry {path.stem!r}",
                ),
            )
            for path in geometry_paths
        ) > (MNSOL_MAX_TOTAL_GEOMETRY_BYTES):
            raise ValueError(
                "Extracted MNSol geometry payload exceeds the safety limit."
            )
        geometry_payloads = {
            path.stem: _read_path_limited(
                path,
                limit=MNSOL_MAX_GEOMETRY_BYTES,
                label=f"geometry {path.stem!r}",
            )
            for path in geometry_paths
        }
        return (
            _read_path_limited(
                table_path,
                limit=MNSOL_MAX_TABLE_BYTES,
                label="table",
            ),
            geometry_payloads,
            "directory",
            None,
        )

    raise ValueError(
        "MNSol source must be a user-supplied .zip archive or extracted directory."
    )


def load_mnsol_v2012(source: str | Path, protocol: MNSolProtocol) -> MNSolDataset:
    """Read and validate a user-supplied MNSol-v2012 distribution."""

    table_payload, geometry_payloads, source_kind, artifact_sha256 = _read_source(
        Path(source)
    )
    table_sha256 = _sha256_bytes(table_payload)
    if table_sha256 != protocol.expected_table_sha256:
        raise ValueError(
            "MNSol table hash mismatch: expected "
            f"{protocol.expected_table_sha256}, observed {table_sha256}."
        )
    normalized_bundle_sha256 = _normalized_bundle_sha256(
        table_payload, geometry_payloads
    )
    if normalized_bundle_sha256 != protocol.expected_normalized_bundle_sha256:
        raise ValueError(
            "MNSol normalized bundle hash mismatch: expected "
            f"{protocol.expected_normalized_bundle_sha256}, observed "
            f"{normalized_bundle_sha256}."
        )
    records = _parse_table(table_payload)
    if len(records) != protocol.expected_record_count:
        raise ValueError(
            f"MNSol record count mismatch: expected {protocol.expected_record_count}, "
            f"observed {len(records)}."
        )
    expected_entries = list(range(1, protocol.expected_record_count + 1))
    if [record.entry_number for record in records] != expected_entries:
        raise ValueError(
            "MNSol database entry numbers are not the frozen 1..N sequence."
        )

    record_handles = {record.geometry_handle for record in records}
    if len(record_handles) != protocol.expected_unique_solutes:
        raise ValueError(
            "MNSol unique-solute count mismatch: expected "
            f"{protocol.expected_unique_solutes}, observed {len(record_handles)}."
        )
    if set(geometry_payloads) != record_handles:
        missing = sorted(record_handles - set(geometry_payloads))
        extra = sorted(set(geometry_payloads) - record_handles)
        raise ValueError(
            "MNSol geometry/table identities do not reconcile "
            f"(missing={missing[:5]}, extra={extra[:5]})."
        )
    geometries = {
        handle: _parse_geometry(handle, payload)
        for handle, payload in geometry_payloads.items()
    }
    for record in records:
        geometry = geometries[record.geometry_handle]
        if record.charge != geometry.charge:
            raise ValueError(
                f"MNSol charge mismatch for {record.geometry_handle}: "
                f"table={record.charge}, geometry={geometry.charge}."
            )
        if record.formula != geometry.formula:
            raise ValueError(
                f"MNSol formula mismatch for {record.geometry_handle}: "
                f"table={record.formula!r}, geometry={geometry.formula!r}."
            )

    absolute_solvents = {
        record.solvent for record in records if record.process_type == "abs"
    }
    transfer_labels = {
        record.solvent for record in records if record.process_type == "rel"
    }
    if len(absolute_solvents) != protocol.expected_absolute_solvents:
        raise ValueError(
            "MNSol absolute-solvent count mismatch: expected "
            f"{protocol.expected_absolute_solvents}, observed "
            f"{len(absolute_solvents)}."
        )
    if len(transfer_labels) != protocol.expected_transfer_labels:
        raise ValueError(
            "MNSol transfer-label count mismatch: expected "
            f"{protocol.expected_transfer_labels}, observed {len(transfer_labels)}."
        )

    return MNSolDataset(
        records=records,
        geometries=MappingProxyType(geometries),
        table_sha256=table_sha256,
        normalized_bundle_sha256=normalized_bundle_sha256,
        source_artifact_sha256=artifact_sha256,
        source_kind=source_kind,
    )


def partition_for_mnsol_handle(
    handle: str,
    protocol: MNSolProtocol,
) -> str:
    """Return the solute-group partition without inspecting experiment values."""

    key = f"{protocol.partition_seed}\0{handle}".encode("utf-8")
    score = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "development" if score < protocol.development_fraction else "confirmation"


def _geometry_domain_exclusion(
    geometry: MNSolGeometry,
    protocol: MNSolProtocol,
) -> str | None:
    mass_min, mass_max = protocol.molecular_mass_da
    if geometry.multiplicity != protocol.multiplicity:
        return "non-singlet_geometry"
    if not set(geometry.atomic_numbers).issubset(protocol.allowed_atomic_numbers):
        return "unsupported_element"
    if not mass_min <= geometry.molecular_mass_da <= mass_max:
        return "outside_mass_domain"
    if (
        _covalent_component_count(
            geometry,
            bond_scale=protocol.connectedness_bond_scale,
        )
        != protocol.required_component_count
    ):
        return "disconnected_geometry"
    return None


def eligible_mnsol_records(
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
) -> tuple[MNSolEligibleRecord, ...]:
    """Return neutral absolute rows inside the frozen Route-2 domain."""

    solvent_names = {
        item.mnsol_name: item.canonical_name
        for item in protocol.panel
        if item.neutral_absolute_validation
    }
    eligible: list[MNSolEligibleRecord] = []
    for record in dataset.records:
        canonical_solvent = solvent_names.get(record.solvent)
        if canonical_solvent is None:
            continue
        if (
            record.process_type != protocol.process_type
            or record.charge != protocol.molecular_charge
        ):
            continue
        geometry = dataset.geometries[record.geometry_handle]
        if _geometry_domain_exclusion(geometry, protocol) is not None:
            continue
        eligible.append(
            MNSolEligibleRecord(
                canonical_solvent=canonical_solvent,
                record=record,
                geometry=geometry,
                partition=partition_for_mnsol_handle(
                    record.geometry_handle,
                    protocol,
                ),
            )
        )
    return tuple(eligible)


# Retain the private spelling for older fixture callers while new benchmark
# code uses the explicit public helper above.
_partition_for_handle = partition_for_mnsol_handle


def build_coverage_manifest(
    dataset: MNSolDataset, protocol: MNSolProtocol
) -> dict[str, object]:
    """Build a deterministic aggregate-only coverage manifest."""

    charge_counts = Counter(record.charge for record in dataset.records)
    process_counts = Counter(record.process_type for record in dataset.records)
    subset_counts = Counter(record.subset for record in dataset.records)
    descriptor_inconsistencies: list[str] = []
    for solvent in {record.solvent for record in dataset.records}:
        descriptor_sets = {
            record.solvent_descriptors
            for record in dataset.records
            if record.solvent == solvent
        }
        if len(descriptor_sets) != 1:
            descriptor_inconsistencies.append(solvent)

    panel_rows: list[dict[str, object]] = []
    eligible_records: list[MNSolRecord] = []
    exclusion_counts: Counter[str] = Counter()
    for panel_solvent in protocol.panel:
        records = [
            record
            for record in dataset.records
            if record.solvent == panel_solvent.mnsol_name
        ]
        neutral_absolute = [
            record
            for record in records
            if record.process_type == protocol.process_type
            and record.charge == protocol.molecular_charge
        ]
        if len(records) != panel_solvent.expected_all_records:
            raise ValueError(
                f"MNSol count mismatch for {panel_solvent.canonical_name}: "
                f"expected {panel_solvent.expected_all_records}, observed "
                f"{len(records)}."
            )
        if len(neutral_absolute) != panel_solvent.expected_neutral_absolute_records:
            raise ValueError(
                "MNSol neutral-absolute count mismatch for "
                f"{panel_solvent.canonical_name}: expected "
                f"{panel_solvent.expected_neutral_absolute_records}, observed "
                f"{len(neutral_absolute)}."
            )

        eligible_for_solvent = 0
        if panel_solvent.neutral_absolute_validation:
            for record in neutral_absolute:
                geometry = dataset.geometries[record.geometry_handle]
                exclusion = _geometry_domain_exclusion(geometry, protocol)
                if exclusion is None:
                    eligible_records.append(record)
                    eligible_for_solvent += 1
                else:
                    exclusion_counts[exclusion] += 1
        panel_rows.append(
            {
                "canonical_name": panel_solvent.canonical_name,
                "mnsol_name": panel_solvent.mnsol_name,
                "all_records": len(records),
                "absolute_neutral_records": len(neutral_absolute),
                "ionic_records": sum(record.charge != 0 for record in records),
                "neutral_absolute_validation": (
                    panel_solvent.neutral_absolute_validation
                ),
                "route2_domain_eligible_records": eligible_for_solvent,
                "rationale": panel_solvent.rationale,
            }
        )

    eligible_handles = {record.geometry_handle for record in eligible_records}
    partition_by_handle = {
        handle: partition_for_mnsol_handle(handle, protocol)
        for handle in eligible_handles
    }
    partition_record_counts = Counter(
        partition_by_handle[record.geometry_handle] for record in eligible_records
    )
    partition_solute_counts = Counter(partition_by_handle.values())
    observed_partitions_by_handle: dict[str, set[str]] = {}
    for record in eligible_records:
        observed_partitions_by_handle.setdefault(record.geometry_handle, set()).add(
            partition_by_handle[record.geometry_handle]
        )
    cross_solvent_leakage = any(
        len(partitions) != 1 for partitions in observed_partitions_by_handle.values()
    )
    if cross_solvent_leakage:
        raise RuntimeError("MNSol FileHandle partitioning leaked a solute.")

    return {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "source_kind": dataset.source_kind,
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "record_count": len(dataset.records),
            "unique_solute_count": len(dataset.geometries),
            "absolute_solvent_count": len(
                {
                    record.solvent
                    for record in dataset.records
                    if record.process_type == "abs"
                }
            ),
            "transfer_label_count": len(
                {
                    record.solvent
                    for record in dataset.records
                    if record.process_type == "rel"
                }
            ),
        },
        "integrity": {
            "table_geometry_identity_reconciled": True,
            "table_geometry_charge_reconciled": True,
            "table_geometry_formula_reconciled": True,
            "geometry_level_tag": "m062x_mg3s_geom",
            "solvent_descriptor_inconsistencies": sorted(descriptor_inconsistencies),
            "raw_rows_emitted": False,
        },
        "aggregate_counts": {
            "process_type": dict(sorted(process_counts.items())),
            "charge": {
                str(charge): count for charge, count in sorted(charge_counts.items())
            },
            "subset": dict(sorted(subset_counts.items())),
        },
        "solvent_panel": panel_rows,
        "initial_neutral_absolute_scope": {
            "pre_domain_record_count": sum(
                row["absolute_neutral_records"]
                for row in panel_rows
                if row["neutral_absolute_validation"]
            ),
            "eligible_record_count": len(eligible_records),
            "eligible_unique_solute_count": len(eligible_handles),
            "aggregate_exclusion_counts": dict(sorted(exclusion_counts.items())),
            "process_type": protocol.process_type,
            "charge": protocol.molecular_charge,
            "multiplicity": protocol.multiplicity,
            "required_covalent_component_count": protocol.required_component_count,
            "connectedness_bond_scale": protocol.connectedness_bond_scale,
            "connectedness_radius_source": "ase.data.covalent_radii",
            "geometry_policy": protocol.geometry_policy,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
        },
        "partition": {
            "strategy": protocol.partition_strategy,
            "group_key": "FileHandle",
            "same_solute_cross_solvent_leakage": cross_solvent_leakage,
            "development_fraction": protocol.development_fraction,
            "seed": protocol.partition_seed,
            "record_counts": {
                name: partition_record_counts.get(name, 0)
                for name in ("development", "confirmation")
            },
            "unique_solute_counts": {
                name: partition_solute_counts.get(name, 0)
                for name in ("development", "confirmation")
            },
        },
        "solvent_generalization": {
            "strategy": protocol.solvent_generalization_strategy,
            "neutral_absolute_solvent_count": sum(
                item.neutral_absolute_validation for item in protocol.panel
            ),
            "single_pooled_random_split_is_sufficient": (
                not protocol.pooled_random_split_is_not_sufficient
            ),
            "small_n_policy": protocol.small_n_policy,
        },
        "claim_boundary": (
            "Dataset ingestion and aggregate coverage only. No MACE-POLAR, "
            "AIMNet, continuum, CDS, accuracy, force, or PES calculation was run."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a user-supplied MNSol-v2012 distribution."
    )
    parser.add_argument("action", choices=("inspect",))
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    manifest = build_coverage_manifest(dataset, protocol)
    write_json_atomic(args.output, manifest)
    scope = manifest["initial_neutral_absolute_scope"]
    print(
        "Verified MNSol-v2012: "
        f"{manifest['dataset']['record_count']} records, "
        f"{scope['eligible_record_count']} initial Route-2-domain records, "
        f"{scope['eligible_unique_solute_count']} unique solutes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
