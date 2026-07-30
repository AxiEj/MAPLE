#!/usr/bin/env python3
"""Freeze SolProp-mix candidate-holdout metadata without reading target values.

This audit is deliberately non-scoring.  It verifies the exact official
SolProp-mix archive and workbook, then reads only the two worksheets labelled
as unused three- and four-component VLE data.  Experimental ``Gsolv`` cells
are counted structurally but their values are never decoded, embedded,
compared, or used for model selection.  The result remains a candidate
holdout, not a strict final holdout, until record-level overlap accounting is
complete for every candidate model family.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

AUDITED_ON = "2026-07-30"
ARCHIVE_SIZE_BYTES = 288_947_625
ARCHIVE_SHA256 = "670915e5bf86d5457bc2f43301e5a02e77a66b529535fcd0eddfadeb72ed73a5"
SOURCE_REPOSITORY = "https://gitlab.kuleuven.be/creas/vermeiregroup/solprop"
SOURCE_REVISION = "80043ce09eb8802517c35b59254f8e9c181f2dac"
ZENODO_RECORD = "https://zenodo.org/records/14238055"
PAPER_URL = "https://arxiv.org/abs/2412.01982"
MINIMUM_REQUIRED_SOLVENTS = 10
PUBLISHED_MODEL_DATA_BOUNDARY = {
    "source": PAPER_URL,
    "pretraining_datasets": ["CombiSolv-QM", "BinarySolv-QM"],
    "fine_tuning_datasets": ["CombiSolv-Exp"],
    "published_test_datasets": ["BinarySolv-Exp", "TernarySolv-Exp"],
    "split_unit": "solute",
    "published_test_solutes_excluded_from": ["CombiSolv-Exp"],
    "published_test_solute_overlap": {
        "synthetic_cosmo_rs_datasets": "some_overlap_reported",
    },
    "scope_warning": (
        "The published exclusions apply to BinarySolv-Exp and TernarySolv-Exp. "
        "They do not certify either Not used worksheet audited here."
    ),
}

WORKBOOK_MEMBER = "Files/Data/solprop-mix.xlsx"
DICTIONARY_MEMBER = (
    "Files/DataPreparation/inchi_dictionaries/" "DDB_to_inchi_Dortmund_key.csv"
)
ARCHIVE_README_MEMBER = "Files/README.txt"
MODEL_README_MEMBER = "Files/SolProp_ML-StaticCodeGsolv/README.md"

PINNED_MEMBERS = {
    WORKBOOK_MEMBER: {
        "size_bytes": 12_342_482,
        "crc32": "7e4b3d2a",
        "sha256": ("a20e5720b3a395386dbf522137378435f464a4cbd21574d36d3f713eb0fa0e16"),
    },
    DICTIONARY_MEMBER: {
        "size_bytes": 217_440,
        "crc32": "c130a571",
        "sha256": ("47dcb97393d9abbe0bd415a730f5f5ecc8b3ab18dd7083bc2eeb6dba17fcf363"),
    },
    ARCHIVE_README_MEMBER: {
        "size_bytes": 9_677,
        "crc32": "da436916",
        "sha256": ("b7b1627f9a8ee6cd8f2949c88a3008aa6ec25a6f9dcf997023eb3840e2cf4fdd"),
    },
    MODEL_README_MEMBER: {
        "size_bytes": 7_770,
        "crc32": "8a8d346a",
        "sha256": ("5af355681ae5ed606eac01916a387f18642f0ee84b1b866259dadba5665f69a0"),
    },
}

WORKBOOK_XML = "xl/workbook.xml"
WORKBOOK_RELATIONSHIPS_XML = "xl/_rels/workbook.xml.rels"
PINNED_WORKBOOK_MEMBERS = {
    WORKBOOK_XML: {
        "size_bytes": 1_405,
        "crc32": "d0101792",
        "sha256": ("72e277382291e1df898056ae9d187991e799e8928415946e7199f32e4b54cd3b"),
    },
    WORKBOOK_RELATIONSHIPS_XML: {
        "size_bytes": 1_973,
        "crc32": "8ac9313d",
        "sha256": ("fe9dd6b9a3e7f9c94edec3f1fe3f84d61f5332f263442af9d2ad2acca5535e46"),
    },
    "xl/worksheets/sheet10.xml": {
        "size_bytes": 29_709_431,
        "crc32": "4183538b",
        "sha256": ("fd00cb0d2009da2d552880de0faa9e989086646a86fd82d7e948092f53aa88a9"),
    },
    "xl/worksheets/sheet11.xml": {
        "size_bytes": 5_735_891,
        "crc32": "3498d940",
        "sha256": ("745400d626a7557f1a9fd03d9752c3c0cc9fbe3a281c7eeca733737bb875b3dd"),
    },
}

EXPECTED_WORKSHEETS = [
    "From IDAC - has water",
    "From IDAC - no water",
    "From VLE - Bin Solv - has water",
    "From VLE - Bin Solv - no water",
    "From VLE - Ter Solv - has water",
    "From VLE - Ter Solv - no water",
    "Dataset Statistics",
    "Not used - From VLE - Mono Solv",
    "Not used - From IDAC",
    "Not used - From 3-comp VLE",
    "Not used - From 4-comp VLE",
]

THREE_COMPONENT_HEADER = [
    "inchi_solute",
    "DDB_name_solute",
    "inchi_solvent1",
    "DDB_name_solvent1",
    "inchi_solvent2",
    "DDB_name_solvent2",
    "frac_solvent1",
    "T (K)",
    "P (kPa)",
    "Gsolv (kcal/mol)",
    "solute_exp_liq_mol_frac",
    "Authors",
    "Journal",
    "Volume",
    "Issue",
    "Pages",
    "Year",
    "Title",
]
FOUR_COMPONENT_HEADER = [
    "inchi_solute",
    "DDB_name_solute",
    "inchi_solvent1",
    "DDB_name_solvent1",
    "inchi_solvent2",
    "DDB_name_solvent2",
    "inchi_solvent3",
    "DDB_name_solvent3",
    "frac_solvent1",
    "frac_solvent2",
    "T (K)",
    "P (kPa)",
    "Gsolv (kcal/mol)",
    "solute_exp_liq_mol_frac",
    "Authors",
    "Journal",
    "Volume",
    "Issue",
    "Pages",
    "Year",
    "Title",
]

TARGET_SHEETS = {
    "Not used - From 3-comp VLE": {
        "sheet_name": "Not used - From 3-comp VLE",
        "header": THREE_COMPONENT_HEADER,
        "expected_rows": 25_833,
        "summary_key": "not_used_from_3_component_vle",
        "worksheet_path": "xl/worksheets/sheet10.xml",
        "size_bytes": 29_709_431,
        "crc32": "4183538b",
        "sha256": ("fd00cb0d2009da2d552880de0faa9e989086646a86fd82d7e948092f53aa88a9"),
        "source_system_component_count": 3,
        "solvent_component_count": 2,
    },
    "Not used - From 4-comp VLE": {
        "sheet_name": "Not used - From 4-comp VLE",
        "header": FOUR_COMPONENT_HEADER,
        "expected_rows": 4_351,
        "summary_key": "not_used_from_4_component_vle",
        "worksheet_path": "xl/worksheets/sheet11.xml",
        "size_bytes": 5_735_891,
        "crc32": "3498d940",
        "sha256": ("745400d626a7557f1a9fd03d9752c3c0cc9fbe3a281c7eeca733737bb875b3dd"),
        "source_system_component_count": 4,
        "solvent_component_count": 3,
    },
}

LABEL_COLUMN = "Gsolv (kcal/mol)"
NUMERIC_COLUMNS = {
    "frac_solvent1",
    "frac_solvent2",
    "T (K)",
    "P (kPa)",
    "solute_exp_liq_mol_frac",
}
CORE_PROVENANCE_COLUMNS = ("Authors", "Journal", "Year", "Title")
CITATION_COLUMNS = (
    "Authors",
    "Journal",
    "Volume",
    "Issue",
    "Pages",
    "Year",
    "Title",
)
FRACTION_ROUNDOFF_TOLERANCE = Decimal("1e-12")

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCUMENT_RELATIONSHIP_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
ROW_TAG = f"{{{SPREADSHEET_NS}}}row"
CELL_TAG = f"{{{SPREADSHEET_NS}}}c"
TEXT_TAG = f"{{{SPREADSHEET_NS}}}t"
VALUE_TAG = f"{{{SPREADSHEET_NS}}}v"
FORMULA_TAG = f"{{{SPREADSHEET_NS}}}f"
CELL_REFERENCE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _column_index(reference: str) -> int:
    match = CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"Invalid worksheet cell reference {reference!r}.")
    index = 0
    for character in match.group(1):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _cell_row(reference: str) -> int:
    match = CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"Invalid worksheet cell reference {reference!r}.")
    return int(match.group(2))


def _cell_value(cell: ElementTree.Element) -> str:
    """Decode a permitted non-label XLSX cell into a stripped text token."""

    if cell.find(FORMULA_TAG) is not None:
        raise ValueError("The candidate worksheet contains a formula cell.")
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(TEXT_TAG)).strip()
    if cell_type in (None, "n", "str"):
        value = cell.find(VALUE_TAG)
        return "" if value is None or value.text is None else value.text.strip()
    raise ValueError(f"Unsupported XLSX cell type {cell_type!r}.")


def _header_value(cell: ElementTree.Element) -> str:
    """Decode a schema header without entering the data-value decoder."""

    if cell.find(FORMULA_TAG) is not None:
        raise ValueError("The candidate worksheet header contains a formula.")
    if cell.attrib.get("t") != "inlineStr":
        raise ValueError("The candidate worksheet header is not inline text.")
    return "".join(node.text or "" for node in cell.iter(TEXT_TAG)).strip()


def _decimal(value: str, *, label: str) -> Decimal:
    if not value:
        raise ValueError(f"{label} is missing.")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not a valid decimal.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} is not finite.")
    return parsed


def _decimal_token(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _range(values: list[Decimal]) -> list[float]:
    if not values:
        raise ValueError("Cannot summarize an empty numeric range.")
    return [float(min(values)), float(max(values))]


def _canonical_composition(
    row: dict[str, str],
    *,
    solvent_component_count: int,
) -> tuple[tuple[tuple[str, str], ...], tuple[Decimal, ...], Decimal]:
    explicit = [
        _decimal(row[f"frac_solvent{index}"], label=f"solvent fraction {index}")
        for index in range(1, solvent_component_count)
    ]
    for value in explicit:
        if value < -FRACTION_ROUNDOFF_TOLERANCE or value > (
            Decimal(1) + FRACTION_ROUNDOFF_TOLERANCE
        ):
            raise ValueError("A solvent fraction lies outside [0, 1].")

    corrected = [min(Decimal(1), max(Decimal(0), value)) for value in explicit]
    adjustment = sum(
        (abs(before - after) for before, after in zip(explicit, corrected)),
        Decimal(0),
    )
    explicit_total = sum(corrected, Decimal(0))
    if explicit_total > Decimal(1):
        excess = explicit_total - Decimal(1)
        if excess > FRACTION_ROUNDOFF_TOLERANCE:
            raise ValueError("The explicit solvent fractions sum above one.")
        largest = max(range(len(corrected)), key=corrected.__getitem__)
        corrected[largest] -= excess
        adjustment += excess
        explicit_total = Decimal(1)

    inferred = Decimal(1) - explicit_total
    fractions = [*corrected, inferred]
    if any(value < 0 or value > 1 for value in fractions):
        raise ValueError("A derived solvent fraction lies outside [0, 1].")
    if sum(fractions, Decimal(0)) != Decimal(1):
        raise ValueError("Canonical solvent fractions do not sum exactly to one.")

    component_fractions: dict[str, Decimal] = {}
    for index, fraction in enumerate(fractions, start=1):
        inchi = row[f"inchi_solvent{index}"]
        if not inchi.startswith("InChI=1S/"):
            raise ValueError(f"Solvent {index} does not have a standard InChI.")
        if fraction > 0:
            component_fractions[inchi] = (
                component_fractions.get(inchi, Decimal(0)) + fraction
            )
    if not component_fractions:
        raise ValueError("The physical solvent composition has no active component.")
    physical_components = tuple(
        sorted(
            (inchi, _decimal_token(fraction))
            for inchi, fraction in component_fractions.items()
        )
    )
    return physical_components, tuple(fractions), adjustment


def _row_cells(row: ElementTree.Element) -> dict[int, ElementTree.Element]:
    row_number_text = row.attrib.get("r")
    if row_number_text is None or not row_number_text.isdigit():
        raise ValueError("The candidate worksheet has an invalid row reference.")
    row_number = int(row_number_text)
    cells: dict[int, ElementTree.Element] = {}
    for cell in row:
        if cell.tag != CELL_TAG:
            continue
        reference = cell.attrib.get("r")
        if reference is None or _cell_row(reference) != row_number:
            raise ValueError("The candidate worksheet has a misplaced cell.")
        column = _column_index(reference)
        if column in cells:
            raise ValueError("The candidate worksheet repeats a cell column.")
        if cell.find(FORMULA_TAG) is not None:
            raise ValueError("The candidate worksheet contains a formula cell.")
        cells[column] = cell
    return cells


def _sheet_analysis(
    worksheet: bytes,
    specification: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_header = specification["header"]
    label_index = expected_header.index(LABEL_COLUMN)
    expected_rows = specification["expected_rows"]
    solvent_component_count = specification["solvent_component_count"]

    solutes: set[str] = set()
    solvents: set[str] = set()
    mixtures: set[tuple[tuple[str, str], ...]] = set()
    solute_mixtures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    citations: set[tuple[str, ...]] = set()
    record_identity_counts: Counter[tuple[Any, ...]] = Counter()
    temperatures: list[Decimal] = []
    pressures: list[Decimal] = []
    solvent_fractions: list[Decimal] = []
    solute_liquid_fractions: list[Decimal] = []
    optional_missing: Counter[str] = Counter()
    label_cells = 0
    incomplete_core_provenance = 0
    records_with_zero_fraction_components = 0
    zero_fraction_component_count = 0
    roundoff_adjustment_count = 0
    maximum_roundoff_adjustment = Decimal(0)
    parsed_header: list[str] | None = None
    record_count = 0

    try:
        events = ElementTree.iterparse(io.BytesIO(worksheet), events=("end",))
        for _, element in events:
            if element.tag != ROW_TAG:
                continue
            cells = _row_cells(element)
            row_number = int(element.attrib["r"])
            if row_number == 1:
                if parsed_header is not None:
                    raise ValueError("The candidate worksheet repeats its header.")
                if set(cells) != set(range(len(expected_header))):
                    raise ValueError("The candidate worksheet header columns drifted.")
                parsed_header = [
                    _header_value(cells[index]) for index in range(len(expected_header))
                ]
                if parsed_header != expected_header:
                    raise ValueError(
                        "The candidate worksheet header does not match the pinned schema."
                    )
                element.clear()
                continue

            if parsed_header is None:
                raise ValueError("The candidate worksheet data precedes its header.")
            expected_row_number = record_count + 2
            if row_number != expected_row_number:
                raise ValueError(
                    "The candidate worksheet has a missing or reordered data row."
                )
            if any(index >= len(expected_header) for index in cells):
                raise ValueError("The candidate worksheet has an unexpected column.")

            label_cell = cells.get(label_index)
            if label_cell is None or label_cell.find(VALUE_TAG) is None:
                raise ValueError("An experimental label cell is structurally missing.")
            if label_cell.attrib.get("t") not in (None, "n"):
                raise ValueError("An experimental label cell is not numeric.")
            label_cells += 1

            row_values = {
                name: (
                    ""
                    if index == label_index or index not in cells
                    else _cell_value(cells[index])
                )
                for index, name in enumerate(expected_header)
            }
            solute = row_values["inchi_solute"]
            if not solute.startswith("InChI=1S/"):
                raise ValueError("A solute does not have a standard InChI.")
            mixture, declared_fractions, roundoff_adjustment = _canonical_composition(
                row_values,
                solvent_component_count=solvent_component_count,
            )
            row_zero_fraction_count = sum(
                fraction == 0 for fraction in declared_fractions
            )
            if row_zero_fraction_count:
                records_with_zero_fraction_components += 1
                zero_fraction_component_count += row_zero_fraction_count
            if roundoff_adjustment:
                roundoff_adjustment_count += 1
                maximum_roundoff_adjustment = max(
                    maximum_roundoff_adjustment,
                    roundoff_adjustment,
                )

            temperature = _decimal(row_values["T (K)"], label="temperature")
            pressure = _decimal(row_values["P (kPa)"], label="pressure")
            solute_fraction = _decimal(
                row_values["solute_exp_liq_mol_frac"],
                label="solute liquid mole fraction",
            )
            if temperature <= 0:
                raise ValueError("A temperature is not positive.")
            if pressure <= 0:
                raise ValueError("A pressure is not positive.")
            if solute_fraction < 0 or solute_fraction > 1:
                raise ValueError("A solute liquid mole fraction lies outside [0, 1].")

            citation = tuple(row_values[column] for column in CITATION_COLUMNS)
            if any(not row_values[column] for column in CORE_PROVENANCE_COLUMNS):
                incomplete_core_provenance += 1
            for column in CITATION_COLUMNS:
                if not row_values[column]:
                    optional_missing[column] += 1

            identity = (
                solute,
                mixture,
                _decimal_token(temperature),
                _decimal_token(pressure),
                _decimal_token(solute_fraction),
                citation,
            )
            solutes.add(solute)
            solvents.update(component for component, _ in mixture)
            mixtures.add(mixture)
            solute_mixtures.add((solute, mixture))
            citations.add(citation)
            record_identity_counts[identity] += 1
            temperatures.append(temperature)
            pressures.append(pressure)
            solvent_fractions.extend(declared_fractions)
            solute_liquid_fractions.append(solute_fraction)
            record_count += 1
            element.clear()
    except ElementTree.ParseError as exc:
        raise ValueError("The candidate worksheet is not valid XML.") from exc

    if parsed_header is None:
        raise ValueError("The candidate worksheet has no header.")
    if record_count != expected_rows:
        raise ValueError(
            f"Pinned candidate worksheet changed; expected {expected_rows} rows, "
            f"found {record_count}."
        )

    ordered_identity_sha256 = _sha256_bytes(
        b"\n".join(
            _canonical_bytes(identity)
            for identity in sorted(record_identity_counts, key=_canonical_bytes)
            for _ in range(record_identity_counts[identity])
        )
    )
    duplicate_count = record_count - len(record_identity_counts)
    summary = {
        "sheet_name": specification["sheet_name"],
        "source_family": "VLE",
        "source_system_component_count": specification["source_system_component_count"],
        "solvent_component_count": solvent_component_count,
        "record_count": record_count,
        "unique_record_identity_count": len(record_identity_counts),
        "duplicate_record_identity_count": duplicate_count,
        "unique_solute_count": len(solutes),
        "unique_solvent_count": len(solvents),
        "unique_mixture_count": len(mixtures),
        "unique_solute_mixture_count": len(solute_mixtures),
        "unique_citation_count": len(citations),
        "records_with_incomplete_core_provenance": incomplete_core_provenance,
        "records_with_zero_fraction_declared_components": (
            records_with_zero_fraction_components
        ),
        "zero_fraction_declared_component_count": zero_fraction_component_count,
        "missing_optional_provenance_cells": dict(sorted(optional_missing.items())),
        "all_inchi_identifiers_use_standard_prefix": True,
        "temperature_range_K": _range(temperatures),
        "pressure_range_kPa": _range(pressures),
        "solvent_fraction_range": _range(solvent_fractions),
        "solute_liquid_mole_fraction_range": _range(solute_liquid_fractions),
        "composition_roundoff_adjustment_count": roundoff_adjustment_count,
        "maximum_composition_roundoff_adjustment": float(maximum_roundoff_adjustment),
        "label_cells_present_but_values_not_decoded": label_cells,
        "experimental_label_values_embedded": False,
        "models_scored": [],
        "record_identities_embedded": False,
        "mixture_identity_basis": (
            "positive-fraction physical components; nominal source-system "
            "cardinality and zero-fraction declarations remain provenance"
        ),
        "ordered_record_identity_sha256": ordered_identity_sha256,
        "header_sha256": _sha256_bytes(_canonical_bytes(parsed_header)),
        "worksheet_size_bytes": len(worksheet),
        "worksheet_sha256": _sha256_bytes(worksheet),
    }
    internal = {
        "solutes": solutes,
        "solvents": solvents,
        "mixtures": mixtures,
        "solute_mixtures": solute_mixtures,
        "citations": citations,
        "record_identity_counts": record_identity_counts,
        "temperatures": temperatures,
        "pressures": pressures,
        "solvent_fractions": solvent_fractions,
        "solute_liquid_fractions": solute_liquid_fractions,
    }
    return summary, internal


def _sheet_summary(
    worksheet: bytes,
    specification: dict[str, Any],
) -> dict[str, Any]:
    """Return the public summary used by unit tests and the release audit."""

    return _sheet_analysis(worksheet, specification)[0]


def _verified_member_record(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "crc32": specification["crc32"],
        "sha256": specification["sha256"],
        "size_bytes": specification["size_bytes"],
        "verified": True,
    }


def _read_verified_member(
    archive: zipfile.ZipFile,
    member: str,
    specification: dict[str, Any],
) -> bytes:
    matches = [item for item in archive.infolist() if item.filename == member]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one ZIP member named {member!r}.")
    info = matches[0]
    if info.flag_bits & 0x1:
        raise ValueError(f"Pinned ZIP member {member!r} is unexpectedly encrypted.")
    if info.file_size != specification["size_bytes"]:
        raise ValueError(f"Pinned ZIP member {member!r} has the wrong size.")
    if f"{info.CRC:08x}" != specification["crc32"]:
        raise ValueError(f"Pinned ZIP member {member!r} has the wrong CRC32.")
    data = archive.read(info)
    if _sha256_bytes(data) != specification["sha256"]:
        raise ValueError(f"Pinned ZIP member {member!r} has the wrong SHA256.")
    return data


def _worksheet_map(workbook_xml: bytes, relationships_xml: bytes) -> dict[str, str]:
    try:
        workbook_root = ElementTree.fromstring(workbook_xml)
        relationships_root = ElementTree.fromstring(relationships_xml)
    except ElementTree.ParseError as exc:
        raise ValueError("The pinned workbook metadata is not valid XML.") from exc

    relationships = {}
    relationship_tag = f"{{{PACKAGE_RELATIONSHIP_NS}}}Relationship"
    for relationship in relationships_root.findall(relationship_tag):
        relationship_id = relationship.attrib.get("Id")
        target = relationship.attrib.get("Target")
        relationship_type = relationship.attrib.get("Type", "")
        if not relationship_id or not target:
            raise ValueError("The workbook has an invalid relationship.")
        if relationship.attrib.get("TargetMode") == "External":
            continue
        if relationship_type.endswith("/worksheet"):
            relationships[relationship_id] = target.lstrip("/")

    sheets = workbook_root.find(f"{{{SPREADSHEET_NS}}}sheets")
    if sheets is None:
        raise ValueError("The pinned workbook has no sheet map.")
    result = {}
    for sheet in sheets:
        name = sheet.attrib.get("name")
        relationship_id = sheet.attrib.get(f"{{{DOCUMENT_RELATIONSHIP_NS}}}id")
        if not name or relationship_id not in relationships:
            raise ValueError("The pinned workbook has an unresolved worksheet.")
        if name in result:
            raise ValueError("The pinned workbook repeats a worksheet name.")
        result[name] = relationships[relationship_id]
    if list(result) != EXPECTED_WORKSHEETS:
        raise ValueError("The pinned workbook sheet order or names changed.")
    return result


def _candidate_union(
    panels: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    summaries = [summary for summary, _ in panels]
    internals = [internal for _, internal in panels]
    identity_counts: Counter[tuple[Any, ...]] = Counter()
    for internal in internals:
        identity_counts.update(internal["record_identity_counts"])

    def union_set(key: str) -> set[Any]:
        result: set[Any] = set()
        for internal in internals:
            result.update(internal[key])
        return result

    temperatures = [
        value for internal in internals for value in internal["temperatures"]
    ]
    pressures = [value for internal in internals for value in internal["pressures"]]
    solvent_fractions = [
        value for internal in internals for value in internal["solvent_fractions"]
    ]
    solute_fractions = [
        value for internal in internals for value in internal["solute_liquid_fractions"]
    ]
    record_count = sum(summary["record_count"] for summary in summaries)
    return {
        "record_count": record_count,
        "unique_record_identity_count": len(identity_counts),
        "duplicate_record_identity_count": record_count - len(identity_counts),
        "unique_solute_count": len(union_set("solutes")),
        "unique_solvent_count": len(union_set("solvents")),
        "unique_mixture_count": len(union_set("mixtures")),
        "unique_solute_mixture_count": len(union_set("solute_mixtures")),
        "unique_citation_count": len(union_set("citations")),
        "records_with_incomplete_core_provenance": sum(
            summary["records_with_incomplete_core_provenance"] for summary in summaries
        ),
        "records_with_zero_fraction_declared_components": sum(
            summary["records_with_zero_fraction_declared_components"]
            for summary in summaries
        ),
        "zero_fraction_declared_component_count": sum(
            summary["zero_fraction_declared_component_count"] for summary in summaries
        ),
        "label_cells_present_but_values_not_decoded": sum(
            summary["label_cells_present_but_values_not_decoded"]
            for summary in summaries
        ),
        "temperature_range_K": _range(temperatures),
        "pressure_range_kPa": _range(pressures),
        "solvent_fraction_range": _range(solvent_fractions),
        "solute_liquid_mole_fraction_range": _range(solute_fractions),
        "composition_roundoff_adjustment_count": sum(
            summary["composition_roundoff_adjustment_count"] for summary in summaries
        ),
        "maximum_composition_roundoff_adjustment": max(
            summary["maximum_composition_roundoff_adjustment"] for summary in summaries
        ),
        "experimental_label_values_embedded": False,
        "models_scored": [],
        "record_identities_embedded": False,
        "mixture_identity_basis": (
            "positive-fraction physical components; nominal source-system "
            "cardinality and zero-fraction declarations remain provenance"
        ),
        "ordered_record_identity_sha256": _sha256_bytes(
            b"\n".join(
                _canonical_bytes(identity)
                for identity in sorted(identity_counts, key=_canonical_bytes)
                for _ in range(identity_counts[identity])
            )
        ),
    }


def audit_archive(archive_path: Path) -> dict[str, Any]:
    """Return a deterministic, metadata-only audit of the exact release."""

    if not archive_path.is_file():
        raise ValueError(f"SolProp-mix archive does not exist: {archive_path}")
    if archive_path.stat().st_size != ARCHIVE_SIZE_BYTES:
        raise ValueError("SolProp-mix archive size does not match the pinned release.")
    if _sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError(
            "SolProp-mix archive SHA256 does not match the pinned release."
        )

    try:
        with zipfile.ZipFile(archive_path) as archive:
            verified_outer = {
                member: _read_verified_member(archive, member, specification)
                for member, specification in PINNED_MEMBERS.items()
            }
    except zipfile.BadZipFile as exc:
        raise ValueError("SolProp-mix archive is not a valid ZIP file.") from exc

    workbook_bytes = verified_outer[WORKBOOK_MEMBER]
    try:
        with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as workbook:
            verified_inner = {
                member: _read_verified_member(workbook, member, specification)
                for member, specification in PINNED_WORKBOOK_MEMBERS.items()
            }
    except zipfile.BadZipFile as exc:
        raise ValueError("SolProp-mix workbook is not a valid XLSX file.") from exc

    worksheet_map = _worksheet_map(
        verified_inner[WORKBOOK_XML],
        verified_inner[WORKBOOK_RELATIONSHIPS_XML],
    )
    for name, specification in TARGET_SHEETS.items():
        if worksheet_map[name] != specification["worksheet_path"]:
            raise ValueError(f"Target worksheet path changed for {name!r}.")

    panels: list[tuple[dict[str, Any], dict[str, Any]]] = []
    candidate_panels = {}
    for name, specification in TARGET_SHEETS.items():
        summary, internal = _sheet_analysis(
            verified_inner[specification["worksheet_path"]],
            specification,
        )
        if summary["sheet_name"] != name:
            raise ValueError("Candidate worksheet name binding failed.")
        candidate_panels[specification["summary_key"]] = summary
        panels.append((summary, internal))
    union = _candidate_union(panels)

    minimum_coverage_passes = union["unique_solvent_count"] >= MINIMUM_REQUIRED_SOLVENTS
    no_duplicate_identities = union["duplicate_record_identity_count"] == 0
    complete_provenance = union["records_with_incomplete_core_provenance"] == 0
    return {
        "schema_version": 1,
        "audited_on": AUDITED_ON,
        "dataset": "SolProp-mix not-used VLE candidate holdout",
        "identity": {
            "archive": {
                "sha256": ARCHIVE_SHA256,
                "size_bytes": ARCHIVE_SIZE_BYTES,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "zenodo_record": ZENODO_RECORD,
                "verified": True,
            },
            "members": {
                member: _verified_member_record(specification)
                for member, specification in sorted(PINNED_MEMBERS.items())
            },
            "workbook_members": {
                member: _verified_member_record(specification)
                for member, specification in sorted(PINNED_WORKBOOK_MEMBERS.items())
            },
            "worksheet_names_and_order_verified": True,
        },
        "paper": {
            "url": PAPER_URL,
            "candidate_sheet_interpretation": (
                "the upstream Not used labels are provenance metadata only; "
                "they do not prove absence from every model training or test family"
            ),
        },
        "candidate_panels": candidate_panels,
        "candidate_union": union,
        "validation_scope": {
            "experimental_label_cells_counted_without_decoding_values": True,
            "experimental_label_values_embedded": False,
            "experimental_label_values_used_for_model_selection": False,
            "maximum_error_computed": False,
            "model_prediction_sheets_read": False,
            "model_predictions_read": False,
            "models_scored": [],
            "record_identities_embedded": False,
        },
        "overlap_audit": {
            "status": "pending",
            "blocking_reason": (
                "record-complete canonical training and test ledgers are not locally "
                "pinned for every candidate model family"
            ),
            "training_overlap_proven_absent": False,
            "published_solprop_mix_model_data_boundary": (
                PUBLISHED_MODEL_DATA_BOUNDARY
            ),
            "candidate_identity_available_without_labels": {
                "exact_record_fields": [
                    "solute standard InChI",
                    "positive-fraction physical solvent InChI/composition pairs",
                    "temperature",
                    "pressure",
                    "solute liquid mole fraction",
                    "citation tuple",
                ],
                "solute_structure_key": "standard InChI",
                "solvent_structure_key": "standard InChI",
                "experimental_target_required_for_join": False,
                "record_values_embedded": False,
            },
            "required_join_levels": {
                "exact_record": (
                    "solute + physical solvent composition + temperature + pressure "
                    "+ solute liquid mole fraction + citation"
                ),
                "solute_structure": (
                    "canonical solute identity regardless of solvent or condition"
                ),
                "solvent_system": (
                    "canonical positive-fraction solvent identities and mole fractions"
                ),
                "source_provenance": "authors + journal + year + title",
            },
            "required_canonical_joins": [
                "FreeSolv",
                "MNSol",
                "dGsolvDB",
                "CombiSolv",
                "Solv@TUM",
                "SolProp",
                "QM-derived candidate training corpora",
                "every admitted candidate model training and test family",
            ],
        },
        "final_holdout_gates": {
            "minimum_solvent_coverage": {
                "observed_unique_solvents": union["unique_solvent_count"],
                "minimum_required": MINIMUM_REQUIRED_SOLVENTS,
                "passes": minimum_coverage_passes,
            },
            "unique_record_identities": {
                "duplicate_record_identity_count": union[
                    "duplicate_record_identity_count"
                ],
                "passes": no_duplicate_identities,
            },
            "complete_core_provenance": {
                "records_with_incomplete_core_provenance": union[
                    "records_with_incomplete_core_provenance"
                ],
                "passes": complete_provenance,
            },
            "training_overlap_absent": {
                "status": "pending",
                "passes": False,
            },
            "passes_all": False,
        },
        "candidate_holdout": True,
        "strict_final_holdout": False,
        "acceptance_eligible": False,
        "gpu_acceleration": {
            "relevant_to_static_audit": False,
            "gpu_execution_performed": False,
            "no_precision_loss_claim": False,
        },
        "route4_decision": {
            "status": "metadata_only_candidate_not_consumed_for_scoring",
            "allowed_future_use": (
                "overlap auditing before any separately authorized blind scoring"
            ),
            "forbidden_uses": [
                "training",
                "fine_tuning",
                "calibration",
                "model_selection",
                "scoring_before_overlap_accounting",
                "claiming_strict_final_holdout",
                "claiming_never_used_from_sheet_names_alone",
            ],
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            os.environ.get(
                "MAPLE_SOLPROPMIX_AUDIT_ARCHIVE",
                "/home/axie/.cache/maple-benchmarks/solprop-mix/Files.zip",
            )
        ),
        help="Exact official SolProp-mix Files.zip archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks"
            / "solpropmix-not-used-candidate-holdout-audit-2026-07-30.json"
        ),
        help="Destination for the deterministic metadata-only audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = audit_archive(args.archive)
    _write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
