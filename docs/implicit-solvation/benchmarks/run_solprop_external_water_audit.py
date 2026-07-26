#!/usr/bin/env python3
"""Run SolProp external-water structural/source audit (label-free, source-only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402

try:
    from rdkit import Chem  # type: ignore
    if Chem is None:
        raise ImportError("RDKit import returned None")
except Exception:  # pragma: no cover - optional
    Chem = None  # type: ignore

try:
    import openpyxl  # type: ignore
except Exception:  # pragma: no cover - optional
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
}

REQUIRED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl"}
TARGET_SHA = "f66bb046bc1d5b8471d36436e485d0cb9d95fb7111aa2db4deb85fdbbea6b766"
TARGET_MD5 = "677d6f107df6fbd0e6d3963e32f06bdb"
TARGET_SIZE = 268_574_239
TARGET_ROWS = 8780


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_smiles(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).lower()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV source file has no header row.")
        for row in reader:
            rows.append({str(key): ("" if value is None else str(value).strip()) for key, value in row.items()})
    return rows


def _load_xlsx_rows(path: Path) -> list[dict[str, str]]:
    if openpyxl is None:
        raise RuntimeError("openpyxl is required for XLSX source input")
    workbook = openpyxl.load_workbook(str(path), read_only=True)
    try:
        sheet = workbook.active
        it = iter(sheet.iter_rows(values_only=True))
        header_raw = next(it)
        if header_raw is None:
            raise ValueError("XLSX source file is empty.")
        header = [str(cell or "").strip() for cell in header_raw]
        if any(not value for value in header):
            raise ValueError("XLSX source file has an empty header cell.")
        rows: list[dict[str, str]] = []
        for row in it:
            rows.append(
                {
                    header[index]: ("" if value is None else str(value).strip())
                    for index, value in enumerate(row)
                }
            )
        return rows
    finally:
        workbook.close()


def _load_source_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise ValueError("JSON source table must be a list of row objects.")
        rows: list[dict[str, str]] = []
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("Each JSON source row must be an object.")
            rows.append({str(key): "" if value is None else str(value) for key, value in row.items()})
        return rows
    if suffix in {".csv", ".tsv"}:
        return _load_csv_rows(path)
    if suffix in {".xlsx", ".xlsm", ".xltx"}:
        return _load_xlsx_rows(path)
    raise ValueError("Unsupported source format; expected csv/json/xlsx.")


def _resolve_column(row: dict[str, str], aliases: list[str] | tuple[str, ...] | set[str]) -> str | None:
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
        [columns["source"]] if isinstance(columns["source"], str) else columns["source"],
    )
    solvent_key = _resolve_column(
        row,
        [columns["solvent"]] if isinstance(columns["solvent"], str) else columns["solvent"],
    )
    solute_smiles_key = _resolve_column(
        row,
        [columns["solute_smiles"]]
        if isinstance(columns["solute_smiles"], str)
        else columns["solute_smiles"],
    )
    solute_inchi_key = _resolve_column(
        row,
        [columns.get("solute_inchi", "solute_inchi")]
        if isinstance(columns.get("solute_inchi", "solute_inchi"), str)
        else columns.get("solute_inchi", ["solute_inchi"]),
    )
    if source_key is None or solvent_key is None or solute_smiles_key is None:
        raise ValueError("Missing one of required columns (source / solvent / solute_smiles).")

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
    if mol.GetNumRadicalElectrons() != 0:
        raise ValueError("Molecule is radical")
    if Chem_.GetFormalCharge(mol) != 0:
        raise ValueError("Molecule is not neutral")
    if len(Chem_.GetMolFrags(mol)) != 1:
        raise ValueError("Molecule is disconnected")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return {
        "atoms": int(mol.GetNumAtoms()),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "elements": sorted(elements),
        "element_signature": "|".join(sorted(elements)),
    }


def _structure_filter_reason(elements: set[str], heavy_atoms: int) -> str | None:
    if "C" not in elements and "Si" not in elements:
        return "no_carbon"
    if not set(elements).issubset(REQUIRED_ELEMENTS):
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
    return normalized == "freesolv" or normalized.startswith("freesolv")


def _build_reference_connectivity(database: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for compound_id, record in database.items():
        smiles = str((record or {}).get("smiles", "")).strip()
        if not smiles:
            raise ValueError(f"FreeSolv record {compound_id} missing smiles")
        connectivity = _inchi_connectivity_block(smiles=smiles, inchi=None)
        existing = mapping.get(connectivity)
        if existing is not None and existing != compound_id:
            raise ValueError(
                f"Duplicate FreeSolv connectivity key {connectivity!r} for {existing!r} and {compound_id!r}."
            )
        mapping[connectivity] = compound_id
    return mapping


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = core.load_json(path)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object.")
    missing = sorted(ALLOWED_PROTOCOL_KEYS - set(protocol))
    if missing:
        raise ValueError("Protocol missing required keys: " + ", ".join(missing))

    if int(protocol["schema_version"]) != 1:
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
    if int(dataset.get("expected_rows", -1)) != TARGET_ROWS:
        raise ValueError(f"Protocol dataset expected_rows must be {TARGET_ROWS}.")
    if int(expected_archive.get("size_bytes", -1)) != TARGET_SIZE:
        raise ValueError("Protocol archive size mismatch.")
    if str(expected_archive.get("sha256", "")).strip() != TARGET_SHA:
        raise ValueError("Protocol archive SHA256 mismatch.")
    if str(expected_archive.get("md5", "")).strip() != TARGET_MD5:
        raise ValueError("Protocol archive MD5 mismatch.")

    counts = protocol["counts"]
    if not isinstance(counts, dict) or not REQUIRED_COUNT_KEYS.issubset(counts):
        raise ValueError("Protocol count block missing required fields.")
    for key in REQUIRED_COUNT_KEYS:
        if int(counts[key]) < 0:
            raise ValueError(f"Protocol count {key} must be non-negative.")

    cols = protocol["columns"]
    if not all(item in cols for item in ["source", "solvent", "solute_smiles"]):
        raise ValueError("Protocol columns must define source/solvent/solute_smiles.")

    analysis = protocol["analysis"]
    if not isinstance(analysis.get("water_smiles"), list) or not analysis["water_smiles"]:
        raise ValueError("Protocol analysis.water_smiles must be a non-empty list.")

    freesolv = protocol["freesolv_reference"]
    freesolv_path = (REPOSITORY_ROOT / str(freesolv["database_json"]).lstrip("/")).resolve()
    if not freesolv_path.is_file():
        raise ValueError("Pinned FreeSolv database path does not exist.")
    if core.sha256_file(freesolv_path) != str(freesolv.get("sha256", "")):
        raise ValueError("Pinned FreeSolv database hash mismatch.")

    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _audit_rows(
    protocol: dict[str, Any],
    source_rows: list[dict[str, str]],
    freesolv_connectivity: dict[str, str],
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
        except Exception as exc:
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

        try:
            connectivity = _inchi_connectivity_block(
                smiles=solute_smiles or None,
                inchi=solute_inchi or None,
            )
        except Exception as exc:
            exclusion_counts["structure_parse_failed"] += 1
            exclusions.append(
                {
                    "row_index": index,
                    "reason_code": "structure_parse_failed",
                    "detail": str(exc),
                }
            )
            continue

        if connectivity in freesolv_connectivity:
            counts["overlap_connectivity_rows"] += 1
            if _source_has_freesolv(source_name):
                counts["source_has_freesolv_rows"] += 1
            else:
                counts["source_has_freesolv_mismatch_rows"] += 1
                exclusions.append(
                    {
                        "row_index": index,
                        "reason_code": "source_has_no_freesolv_tag",
                        "connectivity_block": connectivity,
                    }
                )
            continue

        counts["nonoverlap_rows"] += 1
        if not solute_smiles:
            exclusion_counts["structure_parse_failed"] += 1
            exclusions.append(
                {"row_index": index, "reason_code": "structure_parse_failed", "detail": "Missing solute smiles"}
            )
            continue

        try:
            details = _classify_structure(solute_smiles)
        except Exception as exc:
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

        reason = _structure_filter_reason(set(details["elements"]), details["heavy_atoms"])
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
                "row_index": index,
                "source_all": source_name,
                "solvent_smiles": solvent,
                "solute_smiles": solute_smiles,
                "solute_inchi": solute_inchi or None,
                "connectivity_block": connectivity,
                "structure_elements": details["elements"],
                "atoms": details["atoms"],
                "heavy_atoms": details["heavy_atoms"],
                "element_signature": details["element_signature"],
            }
        )
        counts["candidate_rows"] += 1

    counts["filter_exclusions"] = counts["nonoverlap_rows"] - counts["candidate_rows"]

    for candidate in candidates:
        candidate["row_sha256"] = core.sha256_bytes(
            core.canonical_json_bytes(
                {
                    "row_index": candidate["row_index"],
                    "source_all": candidate["source_all"],
                }
            )
        )

    return candidates, counts, exclusion_counts, exclusions


def audit(
    protocol_path: Path,
    source_path: Path,
    output_manifest: Path,
    output_artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, protocol_fingerprint = _load_protocol(protocol_path)

    expected_archive = protocol["dataset"]["expected_archive"]
    if expected_archive.get("name") and expected_archive["name"] == source_path.name:
        observed_archive_sha = core.sha256_file(source_path)
        if observed_archive_sha != expected_archive.get("sha256"):
            raise ValueError(
                "Source archive hash mismatch: "
                f"expected {expected_archive.get('sha256')}, observed {observed_archive_sha}"
            )

    freesolv = protocol["freesolv_reference"]
    freesolv_path = (REPOSITORY_ROOT / str(freesolv["database_json"]).lstrip("/")).resolve()
    freesolv_db = _load_json(freesolv_path)
    reference_connectivity = _build_reference_connectivity(freesolv_db)

    source_rows = _load_source_rows(source_path)
    candidates, counts, exclusion_counts, exclusions = _audit_rows(
        protocol,
        source_rows,
        reference_connectivity,
    )

    expected_counts = protocol["counts"]
    for key in REQUIRED_COUNT_KEYS:
        if int(counts[key]) != int(expected_counts[key]):
            raise ValueError(f"Count mismatch for {key}: expected {expected_counts[key]}, observed {counts[key]}.")

    candidates.sort(key=lambda record: record["row_index"])

    source_fail_count = exclusion_counts["not_water"] + exclusion_counts["solvent_parse_failed"]

    manifest = {
        "schema_version": 1,
        "manifest_id": "maple-route1-solprop-external-water-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_path_sha256": core.sha256_file(protocol_path),
        "source_dataset_path": str(source_path.resolve()),
        "source_dataset_sha256": core.sha256_file(source_path),
        "source_dataset_name": source_path.name,
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
            "source_path": str(source_path.resolve()),
            "source_archive_name": source_path.name,
            "source_archive_sha256": core.sha256_file(source_path),
            "freesolv_database": str(freesolv_path),
            "freesolv_database_sha256": str(freesolv["sha256"]),
            "counts": counts,
            "source_fail_count": source_fail_count,
            "filter_failure_by_reason": exclusion_counts,
            "record_count": len(candidates),
            "generated_at_utc": _utc_now(),
            "source_manifest_relative_path": str(output_manifest.relative_to(REPOSITORY_ROOT)),
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_content_sha256": manifest["content_sha256"],
            "command_provenance": core.command_provenance(
                Path(__file__),
                {
                    "protocol": str(protocol_path.resolve()),
                    "source": str(source_path),
                    "manifest": str(output_manifest),
                    "artifact": str(output_artifact),
                },
                repository_root=REPOSITORY_ROOT,
                environment_variables=("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"),
            ),
            "records": candidates,
        }
    )
    artifact["source_manifest_content_sha256"] = manifest["content_sha256"]
    artifact["source_manifest_sha256"] = core.sha256_file(output_manifest)
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
