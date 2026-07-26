#!/usr/bin/env python3
"""Run SolProp external-water structural/source audit (label-free, source-only)."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]

try:
    from rdkit import Chem  # type: ignore

    if Chem is None:
        raise ImportError("RDKit import returned None")
except (ImportError, OSError):  # pragma: no cover - optional
    Chem = None  # type: ignore

try:
    import openpyxl  # type: ignore
except ImportError:  # pragma: no cover - optional
    openpyxl = None  # type: ignore


ALLOWED_PROTOCOL_KEYS = {
    "schema_version",
    "protocol_id",
    "claim_scope",
    "route1_boundary",
    "dataset",
    "analysis",
    "counts",
    "columns",
    "freesolv_reference",
    "literature",
}

REQUIRED_COUNT_KEYS = {
    "total_rows",
    "water_rows",
    "overlap_connectivity_rows",
    "source_has_freesolv_rows",
    "source_has_freesolv_mismatch_rows",
    "nonoverlap_rows",
    "candidate_rows",
    "filter_exclusions",
    "candidate_small_rows",
    "candidate_medium_rows",
    "candidate_large_rows",
    "candidate_source_abraham_rows",
    "candidate_source_compsol_rows",
    "candidate_source_both_rows",
    "candidate_multi_measurement_rows",
    "candidate_three_plus_measurement_rows",
}

TARGET_ARCHIVE_SHA = "f66bb046bc1d5b8471d36436e485d0cb9d95fb7111aa2db4deb85fdbbea6b766"
TARGET_ARCHIVE_MD5 = "677d6f107df6fbd0e6d3963e32f06bdb"
TARGET_ARCHIVE_SIZE = 268_574_239
TARGET_TABLE_SHA = "a68bb5ba4f120846f8250cb79f18afd6439cebba4e7d418f2e54ba38919d91b7"
TARGET_TABLE_SIZE = 466_840
TARGET_ROWS = 8780


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_smiles(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).lower()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a JSON integer.")
    return value


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _load_xlsx_rows(
    path: Path,
    *,
    columns: dict[str, str],
    sheet_name: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """Read only identity/provenance columns; never load dGsolv value columns."""
    if openpyxl is None:
        raise RuntimeError("openpyxl is required for XLSX source input")
    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"XLSX source has no {sheet_name!r} worksheet.")
        sheet = workbook[sheet_name]
        header_raw = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        if header_raw is None:
            raise ValueError("XLSX source file is empty.")
        header = [str(cell or "").strip() for cell in header_raw]
        if any(not value for value in header):
            raise ValueError("XLSX source file has an empty header cell.")
        normalized: dict[str, int] = {}
        for index, value in enumerate(header, start=1):
            key = _normalize_header(value)
            if key in normalized:
                raise ValueError(f"Duplicate normalized XLSX header: {value!r}.")
            normalized[key] = index
        selected: list[tuple[str, int]] = []
        for role in ("source", "solvent", "solute_smiles", "solute_inchi"):
            expected = str(columns[role])
            index = normalized.get(_normalize_header(expected))
            if index is None:
                raise ValueError(
                    f"XLSX source is missing required column {expected!r}."
                )
            selected.append((expected, index))

        values_by_column: dict[str, list[str]] = {}
        for name, index in selected:
            values_by_column[name] = [
                "" if row[0] is None else str(row[0]).strip()
                for row in sheet.iter_rows(
                    min_row=2,
                    min_col=index,
                    max_col=index,
                    values_only=True,
                )
            ]
        lengths = {len(values) for values in values_by_column.values()}
        if len(lengths) != 1:
            raise ValueError("Selected XLSX columns have inconsistent row counts.")
        row_count = lengths.pop()
        rows = [
            {name: values[index] for name, values in values_by_column.items()}
            for index in range(row_count)
        ]
        return rows, [name for name, _index in selected]
    finally:
        workbook.close()


def _load_source_rows(
    path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, str]], list[str]]:
    if path.suffix.lower() != ".xlsx":
        raise ValueError(
            "The frozen label-free audit accepts only the pinned XLSX table."
        )
    return _load_xlsx_rows(
        path,
        columns=protocol["columns"],
        sheet_name=str(protocol["analysis"]["sheet_name"]),
    )


def _resolve_column(
    row: dict[str, str], aliases: list[str] | tuple[str, ...] | set[str]
) -> str | None:
    normalized = {_normalize_header(key): key for key in row}
    for alias in aliases:
        key = normalized.get(_normalize_header(alias))
        if key is not None:
            return key
    return None


def _extract_fields(
    row: dict[str, str],
    columns: dict[str, Any],
) -> tuple[str, str, str | None, str | None]:
    source_key = _resolve_column(
        row,
        (
            [columns["source"]]
            if isinstance(columns["source"], str)
            else columns["source"]
        ),
    )
    solvent_key = _resolve_column(
        row,
        (
            [columns["solvent"]]
            if isinstance(columns["solvent"], str)
            else columns["solvent"]
        ),
    )
    solute_smiles_key = _resolve_column(
        row,
        (
            [columns["solute_smiles"]]
            if isinstance(columns["solute_smiles"], str)
            else columns["solute_smiles"]
        ),
    )
    solute_inchi_key = _resolve_column(
        row,
        (
            [columns.get("solute_inchi", "solute_inchi")]
            if isinstance(columns.get("solute_inchi", "solute_inchi"), str)
            else columns.get("solute_inchi", ["solute_inchi"])
        ),
    )
    if source_key is None or solvent_key is None or solute_smiles_key is None:
        raise ValueError(
            "Missing one of required columns (source / solvent / solute_smiles)."
        )

    return (
        str(row.get(source_key, "")).strip(),
        str(row.get(solvent_key, "")).strip(),
        (str(row.get(solute_smiles_key, "")).strip() if solute_smiles_key else None),
        (str(row.get(solute_inchi_key, "")).strip() if solute_inchi_key else None),
    )


def _require_rdkit() -> Any:
    if Chem is None:
        raise RuntimeError("RDKit is required for SolProp audit connectivity checks.")
    return Chem


def _inchi_connectivity_block(smiles: str | None, inchi: str | None) -> str:
    Chem_ = _require_rdkit()
    mol = None
    if inchi:
        mol = Chem_.MolFromInchi(inchi)
    if mol is None and smiles:
        mol = Chem_.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES/InChI parsing failed")
    key = Chem_.MolToInchiKey(mol)
    if not isinstance(key, str) or "-" not in key:
        raise ValueError("Malformed InChIKey from RDKit")
    return key.split("-")[0]


def _classify_structure(smiles: str) -> dict[str, Any]:
    Chem_ = _require_rdkit()
    mol = Chem_.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot parse solute smiles")
    if mol.GetNumAtoms() == 0:
        raise ValueError("Empty molecule")
    radical_electrons = sum(
        int(atom.GetNumRadicalElectrons()) for atom in mol.GetAtoms()
    )
    if radical_electrons != 0:
        raise ValueError("Molecule is radical")
    if Chem_.GetFormalCharge(mol) != 0:
        raise ValueError("Molecule is not neutral")
    if len(Chem_.GetMolFrags(mol)) != 1:
        raise ValueError("Molecule is disconnected")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    inchikey = Chem_.MolToInchiKey(mol)
    return {
        "atoms": int(mol.GetNumAtoms()),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "elements": sorted(elements),
        "element_signature": "|".join(sorted(elements)),
        "canonical_isomeric_smiles": Chem_.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        ),
        "canonical_nonisomeric_smiles": Chem_.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        ),
        "inchikey": inchikey,
        "radical_electrons": radical_electrons,
    }


def _structure_filter_reason(
    elements: set[str], heavy_atoms: int, allowed_elements: set[str]
) -> str | None:
    if "C" not in elements:
        return "no_carbon"
    if not set(elements).issubset(allowed_elements):
        return "unsupported_elements"
    if heavy_atoms <= 0:
        return "zero_heavy_atoms"
    return None


def _is_water(solvent: str, aliases: Iterable[str]) -> bool:
    if not solvent:
        return False
    canonical = _canonical_smiles(solvent)
    if canonical in {_canonical_smiles(a) for a in aliases}:
        return True
    Chem_ = _require_rdkit()
    mol = Chem_.MolFromSmiles(solvent)
    if mol is None:
        raise ValueError(f"Cannot parse solvent smiles: {solvent!r}")
    return Chem_.MolToSmiles(mol, canonical=True) == "O"


def _source_has_freesolv(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", " ", str(value).strip().lower())
    return re.search(r"\bfreesolv\b", normalized) is not None


def _parse_source_entries(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = [value]
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    entries = []
    for item in parsed:
        text = str(item).strip().strip("\"'").strip()
        if text:
            entries.append(text)
    return entries


def _source_families(entries: Iterable[str]) -> list[str]:
    joined = " ".join(entries).lower()
    families = []
    if "abraham" in joined:
        families.append("Abraham")
    if "compsol" in joined:
        families.append("CompSol")
    if "freesolv" in joined:
        families.append("FreeSolv")
    return families


def _build_reference_connectivity(
    database: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for compound_id, record in database.items():
        smiles = str((record or {}).get("smiles", "")).strip()
        if not smiles:
            raise ValueError(f"FreeSolv record {compound_id} missing smiles")
        connectivity = _inchi_connectivity_block(smiles=smiles, inchi=None)
        mapping.setdefault(connectivity, []).append(compound_id)
    return {key: tuple(sorted(values)) for key, values in mapping.items()}


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = core.load_json(path)
    if not isinstance(protocol, dict):
        raise TypeError("Protocol must be a JSON object.")
    missing = sorted(ALLOWED_PROTOCOL_KEYS - set(protocol))
    if missing:
        raise ValueError("Protocol missing required keys: " + ", ".join(missing))

    if _as_int(protocol["schema_version"], "schema_version") != 1:
        raise ValueError("Protocol schema_version must be 1.")
    if protocol.get("protocol_id") != "maple-route1-solprop-external-water-audit-v1":
        raise ValueError("Unexpected protocol_id for SolProp audit.")

    boundary = protocol.get("route1_boundary", {})
    expected_boundary = {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    if boundary != expected_boundary:
        raise ValueError("Route-1 boundary mismatch in protocol.")

    dataset = protocol["dataset"]
    expected_archive = dataset.get("expected_archive", {})
    if str(dataset.get("version", "")).strip() != "v1.2":
        raise ValueError("Protocol dataset version must be v1.2.")
    if (
        _as_int(
            expected_archive.get("size_bytes"), "dataset.expected_archive.size_bytes"
        )
        != TARGET_ARCHIVE_SIZE
    ):
        raise ValueError("Protocol archive size mismatch.")
    if str(expected_archive.get("sha256", "")).strip() != TARGET_ARCHIVE_SHA:
        raise ValueError("Protocol archive SHA256 mismatch.")
    if str(expected_archive.get("md5", "")).strip() != TARGET_ARCHIVE_MD5:
        raise ValueError("Protocol archive MD5 mismatch.")
    source_table = dataset.get("source_table", {})
    if source_table.get("name") != "CombiSolv-Exp.xlsx":
        raise ValueError("Protocol source-table name mismatch.")
    if (
        _as_int(source_table.get("expected_rows"), "dataset.source_table.expected_rows")
        != TARGET_ROWS
    ):
        raise ValueError(f"Protocol source-table row count must be {TARGET_ROWS}.")
    if (
        _as_int(source_table.get("size_bytes"), "dataset.source_table.size_bytes")
        != TARGET_TABLE_SIZE
    ):
        raise ValueError("Protocol source-table size mismatch.")
    if str(source_table.get("sha256", "")).strip() != TARGET_TABLE_SHA:
        raise ValueError("Protocol source-table SHA256 mismatch.")

    counts = protocol["counts"]
    if not isinstance(counts, dict) or not REQUIRED_COUNT_KEYS.issubset(counts):
        raise ValueError("Protocol count block missing required fields.")
    for key in REQUIRED_COUNT_KEYS:
        if _as_int(counts[key], f"counts.{key}") < 0:
            raise ValueError(f"Protocol count {key} must be non-negative.")

    cols = protocol["columns"]
    if not all(item in cols for item in ["source", "solvent", "solute_smiles"]):
        raise ValueError("Protocol columns must define source/solvent/solute_smiles.")

    analysis = protocol["analysis"]
    if (
        not isinstance(analysis.get("water_smiles"), list)
        or not analysis["water_smiles"]
    ):
        raise ValueError("Protocol analysis.water_smiles must be a non-empty list.")
    if analysis.get("sheet_name") != "data":
        raise ValueError("Protocol analysis must select the data worksheet.")
    size_bins = analysis.get("size_bins_heavy_atoms")
    if not isinstance(size_bins, dict):
        raise TypeError("Protocol analysis.size_bins_heavy_atoms must be an object.")
    small_max = _as_int(
        size_bins.get("small_max"), "analysis.size_bins_heavy_atoms.small_max"
    )
    medium_max = _as_int(
        size_bins.get("medium_max"), "analysis.size_bins_heavy_atoms.medium_max"
    )
    if small_max < 1 or medium_max <= small_max:
        raise ValueError("Protocol heavy-atom size-bin boundaries are invalid.")
    allowed_elements = analysis.get("allowed_elements")
    if not isinstance(allowed_elements, list) or allowed_elements != [
        "H",
        "C",
        "N",
        "O",
        "F",
        "S",
        "Cl",
    ]:
        raise ValueError("Protocol Route 1 element domain is not frozen correctly.")

    freesolv = protocol["freesolv_reference"]
    freesolv_path = (
        REPOSITORY_ROOT / str(freesolv["database_json"]).lstrip("/")
    ).resolve()
    if not freesolv_path.is_file():
        raise ValueError("Pinned FreeSolv database path does not exist.")
    if core.sha256_file(freesolv_path) != str(freesolv.get("sha256", "")):
        raise ValueError("Pinned FreeSolv database hash mismatch.")

    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _audit_rows(
    protocol: dict[str, Any],
    source_rows: list[dict[str, str]],
    freesolv_connectivity: dict[str, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], list[dict[str, Any]]]:
    counts = {name: 0 for name in REQUIRED_COUNT_KEYS}
    exclusion_counts = {
        "not_water": 0,
        "solvent_parse_failed": 0,
        "structure_parse_failed": 0,
        "no_identifier": 0,
        "no_carbon": 0,
        "unsupported_elements": 0,
        "zero_heavy_atoms": 0,
        "disconnected": 0,
        "charged": 0,
        "radical": 0,
    }
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    water_aliases = protocol["analysis"]["water_smiles"]
    allowed_elements = set(protocol["analysis"]["allowed_elements"])
    size_bins = protocol["analysis"]["size_bins_heavy_atoms"]

    for index, row in enumerate(source_rows):
        counts["total_rows"] += 1
        try:
            source_name, solvent, solute_smiles, solute_inchi = _extract_fields(
                row, protocol["columns"]
            )
        except ValueError:
            exclusion_counts["no_identifier"] += 1
            exclusions.append({"row_index": index, "reason_code": "no_identifier"})
            continue

        try:
            is_water = _is_water(solvent, water_aliases)
        except (RuntimeError, ValueError) as exc:
            exclusion_counts["solvent_parse_failed"] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": "solvent_parse_failed",
                    "detail": str(exc),
                }
            )
            continue

        if not is_water:
            exclusion_counts["not_water"] += 1
            continue
        counts["water_rows"] += 1
        source_entries = _parse_source_entries(source_name)
        source_families = _source_families(source_entries)
        source_has_freesolv = "FreeSolv" in source_families
        if source_has_freesolv:
            counts["source_has_freesolv_rows"] += 1

        try:
            connectivity = _inchi_connectivity_block(
                smiles=solute_smiles or None,
                inchi=solute_inchi or None,
            )
        except (RuntimeError, ValueError) as exc:
            exclusion_counts["structure_parse_failed"] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": "structure_parse_failed",
                    "detail": str(exc),
                }
            )
            continue

        connectivity_overlap = connectivity in freesolv_connectivity
        if connectivity_overlap:
            counts["overlap_connectivity_rows"] += 1
        if connectivity_overlap != source_has_freesolv:
            counts["source_has_freesolv_mismatch_rows"] += 1
            exclusions.append(
                {
                    "workbook_row_number": index + 2,
                    "reason_code": "freesolv_connectivity_source_mismatch",
                    "connectivity_block": connectivity,
                    "source_has_freesolv": source_has_freesolv,
                }
            )
        if connectivity_overlap or source_has_freesolv:
            continue

        counts["nonoverlap_rows"] += 1
        if not solute_smiles:
            exclusion_counts["structure_parse_failed"] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": "structure_parse_failed",
                    "detail": "Missing solute smiles",
                }
            )
            continue

        try:
            details = _classify_structure(solute_smiles)
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if "charged" in message:
                reason = "charged"
            elif "radical" in message:
                reason = "radical"
            elif "disconnected" in message:
                reason = "disconnected"
            elif "Empty" in message:
                reason = "zero_heavy_atoms"
            else:
                reason = "structure_parse_failed"
            exclusion_counts[reason] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": f"structure_{reason}",
                    "detail": str(exc),
                }
            )
            continue

        reason = _structure_filter_reason(
            set(details["elements"]), details["heavy_atoms"], allowed_elements
        )
        if reason is not None:
            exclusion_counts[reason] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": reason,
                    "connectivity_block": connectivity,
                }
            )
            continue

        candidates.append(
            {
                "candidate_id": connectivity,
                "workbook_row_number": index + 2,
                "solvent_smiles": solvent,
                "solute_smiles": solute_smiles,
                "solute_inchi": solute_inchi or None,
                "connectivity_block": connectivity,
                "inchikey": details["inchikey"],
                "canonical_isomeric_smiles": details["canonical_isomeric_smiles"],
                "canonical_nonisomeric_smiles": details["canonical_nonisomeric_smiles"],
                "elements": details["elements"],
                "atoms": details["atoms"],
                "heavy_atoms": details["heavy_atoms"],
                "element_signature": details["element_signature"],
                "radical_electrons": details["radical_electrons"],
                "source_entries": source_entries,
                "source_families": source_families,
                "measurement_count_from_sources": len(source_entries),
                "source_has_freesolv": source_has_freesolv,
            }
        )
        counts["candidate_rows"] += 1
        if details["heavy_atoms"] <= _as_int(
            size_bins["small_max"], "analysis.size_bins_heavy_atoms.small_max"
        ):
            counts["candidate_small_rows"] += 1
        elif details["heavy_atoms"] <= _as_int(
            size_bins["medium_max"], "analysis.size_bins_heavy_atoms.medium_max"
        ):
            counts["candidate_medium_rows"] += 1
        else:
            counts["candidate_large_rows"] += 1
        has_abraham = "Abraham" in source_families
        has_compsol = "CompSol" in source_families
        counts["candidate_source_abraham_rows"] += int(has_abraham)
        counts["candidate_source_compsol_rows"] += int(has_compsol)
        counts["candidate_source_both_rows"] += int(has_abraham and has_compsol)
        counts["candidate_multi_measurement_rows"] += int(len(source_entries) >= 2)
        counts["candidate_three_plus_measurement_rows"] += int(len(source_entries) >= 3)

    counts["filter_exclusions"] = counts["nonoverlap_rows"] - counts["candidate_rows"]
    connectivities = [record["connectivity_block"] for record in candidates]
    if protocol["analysis"].get("require_unique_candidate_connectivity") and len(
        connectivities
    ) != len(set(connectivities)):
        raise ValueError("Candidate manifest contains duplicate connectivity blocks.")

    for candidate in candidates:
        payload = dict(candidate)
        candidate["row_sha256"] = core.sha256_bytes(core.canonical_json_bytes(payload))

    return candidates, counts, exclusion_counts, exclusions


def audit(
    protocol_path: Path,
    source_path: Path,
    output_manifest: Path,
    output_artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, protocol_fingerprint = _load_protocol(protocol_path)

    source_table = protocol["dataset"]["source_table"]
    if source_path.name != source_table["name"]:
        raise ValueError("Source table filename does not match the frozen protocol.")
    if source_path.stat().st_size != _as_int(
        source_table["size_bytes"], "dataset.source_table.size_bytes"
    ):
        raise ValueError("Source table size does not match the frozen protocol.")
    observed_source_sha = core.sha256_file(source_path)
    if observed_source_sha != source_table["sha256"]:
        raise ValueError("Source table hash does not match the frozen protocol.")

    freesolv = protocol["freesolv_reference"]
    freesolv_path = (
        REPOSITORY_ROOT / str(freesolv["database_json"]).lstrip("/")
    ).resolve()
    freesolv_db = _load_json(freesolv_path)
    reference_connectivity = _build_reference_connectivity(freesolv_db)

    source_rows, loaded_columns = _load_source_rows(source_path, protocol)
    candidates, counts, exclusion_counts, exclusions = _audit_rows(
        protocol,
        source_rows,
        reference_connectivity,
    )

    expected_counts = protocol["counts"]
    for key in REQUIRED_COUNT_KEYS:
        if counts[key] != _as_int(expected_counts[key], f"counts.{key}"):
            raise ValueError(
                f"Count mismatch for {key}: expected {expected_counts[key]}, observed {counts[key]}."
            )

    candidates.sort(key=lambda record: record["workbook_row_number"])

    source_fail_count = (
        exclusion_counts["not_water"] + exclusion_counts["solvent_parse_failed"]
    )

    manifest = {
        "schema_version": 1,
        "manifest_id": "maple-route1-solprop-external-water-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "protocol_path": _portable_path(protocol_path),
        "protocol_path_sha256": core.sha256_file(protocol_path),
        "source_dataset": {
            "name": source_path.name,
            "sha256": observed_source_sha,
            "size_bytes": source_path.stat().st_size,
            "worksheet": protocol["analysis"]["sheet_name"],
            "loaded_columns": loaded_columns,
            "experimental_value_columns_loaded": [],
        },
        "analysis_schema": "external-water-only",
        "counts": {
            **counts,
            "source_fail_count": source_fail_count,
        },
        "exclusion_counts": exclusion_counts,
        "exclusions": exclusions,
        "records": candidates,
    }
    manifest["content_sha256"] = core.artifact_content_sha256(manifest)
    core.write_json_atomic(output_manifest, manifest)

    source_manifest_sha256 = core.sha256_file(output_manifest)

    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-solprop-external-water-audit-v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol_fingerprint,
            "protocol_path": str(protocol_path.resolve()),
            "protocol_path_sha256": core.sha256_file(protocol_path),
            "source_table": {
                "name": source_path.name,
                "sha256": observed_source_sha,
                "size_bytes": source_path.stat().st_size,
                "worksheet": protocol["analysis"]["sheet_name"],
                "loaded_columns": loaded_columns,
                "experimental_value_columns_loaded": [],
            },
            "freesolv_database": _portable_path(freesolv_path),
            "freesolv_database_sha256": str(freesolv["sha256"]),
            "counts": counts,
            "source_fail_count": source_fail_count,
            "filter_failure_by_reason": exclusion_counts,
            "record_count": len(candidates),
            "source_manifest_relative_path": _portable_path(output_manifest),
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_content_sha256": manifest["content_sha256"],
            "command_provenance": core.command_provenance(
                Path(__file__),
                {
                    "protocol": _portable_path(protocol_path),
                    "source": source_path.name,
                    "manifest": _portable_path(output_manifest),
                    "artifact": _portable_path(output_artifact),
                },
                repository_root=REPOSITORY_ROOT,
                environment_variables=(
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ),
            ),
        }
    )
    core.write_json_atomic(output_artifact, artifact)

    return artifact, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = Path(args.protocol).resolve()
    source = Path(args.source).resolve()
    manifest = Path(args.manifest).resolve()
    artifact = Path(args.artifact).resolve()

    if not protocol.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol}")
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    audit(protocol, source, manifest, artifact)
    return core.load_json(artifact)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SolProp external-water audit.")
    parser.add_argument(
        "--protocol",
        default=SCRIPT_DIR / "solprop_external_water_audit_protocol.json",
        help="Frozen protocol path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="SolProp source table (CSV/JSON/XLSX).",
    )
    parser.add_argument(
        "--manifest",
        default=SCRIPT_DIR / "solprop_external_water_source_manifest.json",
        help="Source manifest output path.",
    )
    parser.add_argument(
        "--artifact",
        default=SCRIPT_DIR / "route1-solprop-external-water-audit-2026-07-26.json",
        help="Audit artifact output path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
