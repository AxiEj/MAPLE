#!/usr/bin/env python3
"""Audit quarantined 3D-RISM source assets without opening result tables.

Misin, Palmer, and Fedorov (2016) publish a ZIP containing both solvent
susceptibilities and per-solute solvation-result CSV files.  This tool reads
only the explicitly allowlisted ``.mdl``, ``.sh``, and ``.xvv`` members needed
to establish what the solvent models actually are.  It never opens a member
under the result-table prefix and emits no source labels or source paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]


MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ZIP_MEMBERS = 2_000
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0
ALLOWED_PROTOCOL_KEYS = {
    "schema_version",
    "protocol_id",
    "benchmark_kind",
    "claim_scope",
    "route1_boundary",
    "source",
    "required_panel_canonical_names",
    "candidate_assets",
    "expected_model",
    "containment",
    "literature",
}
ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}


def _digest(value: object, length: int, field: str) -> str:
    result = str(value).lower()
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a {length}-character hexadecimal digest.")
    return result


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a JSON integer.")
    return value


def _as_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a JSON number.")
    return float(value)


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Coarse-grain asset protocol must be a JSON object.")
    unexpected = sorted(set(protocol) - ALLOWED_PROTOCOL_KEYS)
    missing = sorted(ALLOWED_PROTOCOL_KEYS - set(protocol))
    if unexpected or missing:
        details = []
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        if missing:
            details.append("missing=" + ", ".join(missing))
        raise ValueError("Invalid coarse-grain asset protocol keys: " + "; ".join(details))
    if protocol["schema_version"] != 1:
        raise ValueError("Only coarse-grain asset protocol schema version 1 is supported.")
    if protocol["benchmark_kind"] != "route1-misin-coarse-grain-asset-audit":
        raise ValueError("Unexpected coarse-grain asset protocol benchmark_kind.")
    if protocol["route1_boundary"] != ROUTE1_BOUNDARY:
        raise ValueError("Coarse-grain asset audit must preserve the Route 1 boundary.")

    source = protocol["source"]
    if not isinstance(source, dict):
        raise ValueError("Coarse-grain asset protocol requires a source object.")
    if source.get("acquisition_policy") != "user-supplied-no-auto-download":
        raise ValueError("Coarse-grain source acquisition must remain user-supplied.")
    archive = source.get("archive")
    if not isinstance(archive, dict) or archive.get("name") != "jp6b05352_si_002.zip":
        raise ValueError("Unexpected Misin supporting archive.")
    _digest(archive.get("md5"), 32, "source.archive.md5")
    _digest(archive.get("sha256"), 64, "source.archive.sha256")
    if _as_int(archive.get("size_bytes"), "source.archive.size_bytes") <= 0:
        raise ValueError("source.archive.size_bytes must be positive.")
    for key in ("asset_prefix", "result_prefix"):
        value = str(source.get(key, ""))
        if not value.endswith("/") or value.startswith("/") or ".." in PurePosixPath(value).parts:
            raise ValueError(f"source.{key} must be a safe relative directory prefix.")

    panel = protocol["required_panel_canonical_names"]
    if not isinstance(panel, list) or len(panel) <= 10 or len(panel) != len(set(panel)):
        raise ValueError("The required panel must name more than ten unique solvents.")
    if any(not isinstance(name, str) or not name for name in panel):
        raise ValueError("Required panel names must be non-empty strings.")

    assets = protocol["candidate_assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("Candidate assets must be a non-empty list.")
    seen: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("Each candidate asset must be an object.")
        canonical_name = str(asset.get("canonical_name", ""))
        source_stem = str(asset.get("source_stem", ""))
        if (
            not canonical_name
            or not source_stem
            or canonical_name not in panel
            or canonical_name in seen
            or PurePosixPath(source_stem).name != source_stem
        ):
            raise ValueError("Candidate asset mappings must be unique and panel-bound.")
        seen.add(canonical_name)

    expected = protocol["expected_model"]
    if not isinstance(expected, dict) or expected != {
        "closure": "hnc",
        "temperature_k": 298.15,
        "generator_dieleps": 2.0,
        "xvv_dielectric": 1.0,
        "xvv_grid_points": 16384,
        "site_count": 1,
        "species_count": 1,
        "site_charge": 0.0,
    }:
        raise ValueError("Expected source model must remain the pinned neutral one-site HNC form.")

    containment = protocol["containment"]
    if not isinstance(containment, dict) or containment != {
        "experimental_values_loaded": False,
        "label_file_content_read": False,
        "assets_redistributed": False,
        "candidate_is_general_multisolvent_provider": False,
        "candidate_can_count_toward_15_solvent_gate": False,
        "candidate_can_select_or_score_against_mnsol": False,
    }:
        raise ValueError("Containment must remain label-free and fail closed.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _safe_member(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe ZIP member: {info.filename!r}.")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"ZIP member is too large: {info.filename!r}.")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise ValueError(f"Suspicious ZIP compression ratio: {info.filename!r}.")


def _read_asset(archive: ZipFile, name: str) -> bytes:
    try:
        with archive.open(name, "r") as handle:
            return handle.read(MAX_MEMBER_BYTES + 1)
    except KeyError as exc:
        raise ValueError(f"Missing required source asset: {name}") from exc


def _flag_values(text: str, flag: str) -> list[str]:
    match = re.search(
        rf"^%FLAG {re.escape(flag)}\r?\n%FORMAT\([^\n]+\)\r?\n(?P<values>.*?)(?=^%FLAG |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing %FLAG {flag}.")
    values = match.group("values").split()
    if not values:
        raise ValueError(f"Empty %FLAG {flag}.")
    return values


def _float_values(text: str, flag: str, count: int) -> list[float]:
    values = _flag_values(text, flag)
    if len(values) < count:
        raise ValueError(f"%FLAG {flag} has too few values.")
    try:
        return [float(value) for value in values[:count]]
    except ValueError as exc:
        raise ValueError(f"%FLAG {flag} is not numeric.") from exc


def _parse_asset(
    archive: ZipFile,
    *,
    asset_prefix: str,
    source_stem: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    names = {suffix: f"{asset_prefix}{source_stem}{suffix}" for suffix in (".mdl", ".sh", ".xvv")}
    payloads = {suffix: _read_asset(archive, name) for suffix, name in names.items()}
    try:
        mdl = payloads[".mdl"].decode("ascii")
        generator = payloads[".sh"].decode("ascii")
        xvv = payloads[".xvv"].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Source asset {source_stem!r} is not ASCII text.") from exc

    mdl_sites = _float_values(mdl, "POINTERS", 2)
    mdl_charge = _float_values(mdl, "CHG", 1)[0]
    xvv_pointers = _float_values(xvv, "POINTERS", 3)
    xvv_thermo = _float_values(xvv, "THERMO", 2)
    xvv_charge = _float_values(xvv, "QV", 1)[0]
    closure = re.search(r"CLOSUR\s*=\s*['\"]?([A-Za-z0-9+_-]+)", generator, re.IGNORECASE)
    conditions = re.search(
        r"TEMPER\s*=\s*([-+0-9.eE]+)\s*,\s*DIEps\s*=\s*([-+0-9.eE]+)",
        generator,
        re.IGNORECASE,
    )
    if not closure or not conditions:
        raise ValueError(f"Source generator input is incomplete for {source_stem!r}.")
    observed = {
        "closure": closure.group(1).lower(),
        "temperature_k": float(conditions.group(1)),
        "generator_dieleps": float(conditions.group(2)),
        "xvv_dielectric": xvv_thermo[1],
        "xvv_grid_points": int(xvv_pointers[0]),
        "site_count": int(xvv_pointers[1]),
        "species_count": int(xvv_pointers[2]),
        "site_charge": xvv_charge,
    }
    if int(mdl_sites[0]) != observed["site_count"] or int(mdl_sites[1]) != observed["species_count"]:
        raise ValueError(f"Source model/XVV site topology disagrees for {source_stem!r}.")
    if abs(mdl_charge - observed["site_charge"]) > 1e-12:
        raise ValueError(f"Source model/XVV charge disagrees for {source_stem!r}.")
    for key, expected_value in expected.items():
        value = observed[key]
        if isinstance(expected_value, float):
            if abs(value - expected_value) > 1e-12:
                raise ValueError(f"Unexpected {key} for {source_stem!r}: {value!r}.")
        elif value != expected_value:
            raise ValueError(f"Unexpected {key} for {source_stem!r}: {value!r}.")
    return {
        "source_stem": source_stem,
        "model_sha256": core.sha256_bytes(payloads[".mdl"]),
        "generator_input_sha256": core.sha256_bytes(payloads[".sh"]),
        "xvv_sha256": core.sha256_bytes(payloads[".xvv"]),
        **observed,
    }


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archive(
    protocol: dict[str, Any], protocol_fingerprint: str, archive_path: str | Path
) -> dict[str, Any]:
    """Inspect only named model inputs and XVV assets from a supplied ZIP."""
    source_path = Path(archive_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Supporting archive is not a file: {source_path}")
    source = protocol["source"]
    expected_archive = source["archive"]
    if source_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Supporting archive exceeds the configured size limit.")
    if source_path.stat().st_size != expected_archive["size_bytes"]:
        raise ValueError("Supporting archive size mismatch.")
    observed_sha256 = core.sha256_file(source_path)
    observed_md5 = _md5_file(source_path)
    if observed_sha256 != expected_archive["sha256"] or observed_md5 != expected_archive["md5"]:
        raise ValueError("Supporting archive identity mismatch.")

    try:
        with ZipFile(source_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS:
                raise ValueError("Supporting archive has too many ZIP members.")
            for member in members:
                _safe_member(member)
            member_names = [member.filename for member in members]
            if len(member_names) != len(set(member_names)):
                raise ValueError("Supporting archive has duplicate ZIP member names.")
            asset_prefix = source["asset_prefix"]
            result_prefix = source["result_prefix"]
            result_csv_members = sum(
                1
                for member in members
                if (
                    member.filename.startswith(result_prefix)
                    and PurePosixPath(member.filename).suffix.lower() == ".csv"
                )
            )
            if result_csv_members <= 0:
                raise ValueError("Supporting archive no longer declares result-table members.")
            assets = []
            for mapping in protocol["candidate_assets"]:
                observed = _parse_asset(
                    archive,
                    asset_prefix=asset_prefix,
                    source_stem=mapping["source_stem"],
                    expected=protocol["expected_model"],
                )
                assets.append({"canonical_name": mapping["canonical_name"], **observed})
    except BadZipFile as exc:
        raise ValueError("Supporting archive is not a valid ZIP file.") from exc

    required_names = protocol["required_panel_canonical_names"]
    provided_names = {asset["canonical_name"] for asset in assets}
    result = {
        "artifact_type": "route1-misin-coarse-grain-asset-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol_fingerprint,
        "route1_boundary": protocol["route1_boundary"],
        "containment": protocol["containment"],
        "source_archive": {
            "name": expected_archive["name"],
            "size_bytes": expected_archive["size_bytes"],
            "md5": observed_md5,
            "sha256": observed_sha256,
            "source_path_recorded": False,
            "result_csv_member_count_detected": result_csv_members,
            "label_file_content_read": False,
        },
        "coverage": {
            "required_panel_solvents": len(required_names),
            "candidate_assets": len(assets),
            "missing_required_panel_solvents": [
                name for name in required_names if name not in provided_names
            ],
        },
        "model_properties": protocol["expected_model"],
        "candidate_assets": assets,
        "conclusion": {
            "status": "blocked_for_general_multisolvent_provider",
            "reason_codes": [
                "neutral_single_site_coarse_grain",
                "no_molecular_electrostatics_or_hydrogen_bond_sites",
                "incomplete_15_solvent_coverage",
                "supporting_archive_contains_result_tables",
            ],
            "accuracy_claim": "none",
            "product_or_runtime_change": False,
        },
    }
    return core.seal_artifact(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol, fingerprint = _load_protocol(args.protocol)
    artifact = audit_archive(protocol, fingerprint, args.archive)
    artifact["command_provenance"] = core.command_provenance(
        __file__,
        {
            "protocol": str(args.protocol),
            "archive_name": artifact["source_archive"]["name"],
            "archive_sha256": artifact["source_archive"]["sha256"],
            "output": str(args.output),
        },
        repository_root=REPOSITORY_ROOT,
    )
    core.seal_artifact(artifact)
    core.write_json_atomic(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
