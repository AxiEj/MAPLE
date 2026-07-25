#!/usr/bin/env python3
"""Run the pre-registered, label-exposed Route 1 FreeSolv reserve evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
import run_chagb_nonpolar as chagb  # noqa: E402
from maple.function.calculator.extra_correction.implicit.charges import (  # noqa: E402
    prepare_charges,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_freesolv_reserve_protocol.json"
DEFAULT_MANIFEST = SCRIPT_DIR / "route1_freesolv_reserve_source_manifest.json"
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/route1-freesolv-reserve-20260725"
)
DEFAULT_ENERGY_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
)
DEFAULT_SCORE_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-score-2026-07-25.json"
)

ENDPOINTS = (
    "am1bcc_obc2_ace",
    "am1bcc_chagb_pbsa_cavity_dispersion",
)
KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184
GBNSR6_LOCK = threading.Lock()


def _raw_protocol(path: str | Path) -> dict[str, Any]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only Route 1 reserve protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != (
        "maple-route1-freesolv-label-exposed-reserve-v1"
    ):
        raise ValueError("Unexpected Route 1 reserve protocol id.")
    expected_boundary = {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    if protocol.get("route1_boundary") != expected_boundary:
        raise ValueError("Reserve protocol violates the Route 1 boundary.")
    design = protocol.get("evaluation_design", {})
    required_design = {
        "historical_label_exposure": True,
        "independent_blind_confirmation": False,
        "energy_phase_reads_labels": False,
        "energy_artifact_is_sealed_before_scoring": True,
        "score_phase_reads_labels": True,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_post_score_tuning": True,
    }
    if any(design.get(key) is not value for key, value in required_design.items()):
        raise ValueError("Reserve protocol does not preserve the label-use boundary.")
    if tuple(protocol.get("endpoints", {})) != ENDPOINTS:
        raise ValueError("Reserve endpoint order or membership changed.")
    rule = protocol.get("pre_registered_decision_rule", {})
    if rule.get("production_default_may_change") is not False:
        raise ValueError("Reserve protocol cannot change the product default.")
    if rule.get("runtime_force_capability_may_be_inferred") is not False:
        raise ValueError("Reserve energy scores cannot establish force capability.")
    if int(rule.get("required_case_count", -1)) != int(
        protocol["source_evidence"]["expected_case_count"]
    ):
        raise ValueError("Decision-rule case count does not match source evidence.")
    return protocol


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = _raw_protocol(path)
    manifest_hash = str(protocol["source_evidence"]["source_manifest_sha256"])
    if len(manifest_hash) != 64 or set(manifest_hash) - set("0123456789abcdef"):
        raise ValueError("Source manifest hash must be frozen before energy execution.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _repository_path(relative: str) -> Path:
    path = (REPOSITORY_ROOT / relative).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("Source evidence path escapes the repository.") from exc
    return path


def prepare_source_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol = _raw_protocol(protocol_path)
    evidence = protocol["source_evidence"]
    prepared_path = _repository_path(evidence["prepared_manifest_relative_path"])
    if core.sha256_file(prepared_path) != evidence["prepared_manifest_sha256"]:
        raise ValueError("Pinned prepared FreeSolv manifest hash mismatch.")
    prepared = core.load_json(prepared_path)
    source_partition = protocol["evaluation_design"]["source_partition"]
    records = []
    for candidate in prepared["candidates"]:
        if candidate["partition"] != source_partition:
            continue
        records.append(
            {
                "compound_id": candidate["compound_id"],
                "source_mol2_relative_path": candidate["mol2_relative_path"],
                "source_mol2_sha256": candidate["mol2_sha256"],
                "dataset_record_sha256": candidate["dataset_record_sha256"],
                "structure_group_sha256": candidate["structure_group_sha256"],
                "atom_count": int(candidate["atom_count"]),
                "heavy_atom_count": int(candidate["heavy_atom_count"]),
                "elements": list(candidate["elements"]),
                "bins": {
                    "element_class": candidate["element_class"],
                    "size": candidate["size_bin"],
                    "heteroatom_count": candidate["heteroatom_bin"],
                    "flexibility": candidate["flexibility_bin"],
                },
            }
        )
    records.sort(key=lambda record: record["compound_id"])
    expected = int(evidence["expected_case_count"])
    if len(records) != expected or len({row["compound_id"] for row in records}) != expected:
        raise ValueError(
            f"Expected {expected} unique reserve cases, found {len(records)}."
        )
    source_dir = Path(args.source_dir).resolve()
    for record in records:
        mol2_path = source_dir / record["source_mol2_relative_path"]
        if core.sha256_file(mol2_path) != record["source_mol2_sha256"]:
            raise ValueError(
                f"Pinned source MOL2 hash mismatch: {record['compound_id']}."
            )
    manifest = {
        "schema_version": 1,
        "manifest_id": "maple-route1-freesolv-reserve-label-free-inputs-v1",
        "protocol_id": protocol["protocol_id"],
        "source_designation": protocol["evaluation_design"]["designation"],
        "source_prepared_sha256": evidence["prepared_manifest_sha256"],
        "case_count": len(records),
        "records": records,
    }
    payload = json.dumps(manifest, sort_keys=True).lower()
    if "experimental" in payload:
        raise AssertionError("Label-free source manifest contains a forbidden field.")
    output = Path(args.manifest).resolve()
    core.write_json_atomic(output, manifest)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": core.sha256_file(output),
                "case_count": len(records),
            },
            indent=2,
        )
    )
    return manifest


def load_source_manifest(
    protocol_path: str | Path,
    protocol: dict[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    expected_hash = protocol["source_evidence"]["source_manifest_sha256"]
    if core.sha256_file(path) != expected_hash:
        raise ValueError("Frozen reserve source-manifest hash mismatch.")
    manifest = core.load_json(path)
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Reserve source-manifest protocol mismatch.")
    if int(manifest.get("case_count", -1)) != int(
        protocol["source_evidence"]["expected_case_count"]
    ):
        raise ValueError("Reserve source-manifest case count mismatch.")
    records = manifest.get("records", [])
    ids = [record["compound_id"] for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Reserve source-manifest ids must be sorted and unique.")
    if "experimental" in json.dumps(manifest, sort_keys=True).lower():
        raise ValueError("Energy source manifest contains a label field.")
    return manifest


def resolve_providers(
    protocol: dict[str, Any], amber_bin: str | Path | None
) -> dict[str, Any]:
    executables = chagb.resolve_executables(protocol, amber_bin)
    observed_openmm = importlib.metadata.version("openmm")
    expected_openmm = protocol["providers"]["openmm"]["required_version"]
    if observed_openmm != expected_openmm:
        raise ValueError(
            f"OpenMM version mismatch: expected {expected_openmm}, "
            f"observed {observed_openmm}."
        )
    return {
        "executables": executables,
        "openmm": {
            "version": observed_openmm,
            "platform": protocol["providers"]["openmm"]["platform"],
        },
    }


def _record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "records" / f"{compound_id}.json"


def _verify_source_mol2(row: dict[str, Any], source_dir: Path) -> Path:
    path = source_dir / row["source_mol2_relative_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if core.sha256_file(path) != row["source_mol2_sha256"]:
        raise ValueError(f"Source MOL2 hash mismatch: {row['compound_id']}.")
    return path


def _prepare_am1bcc(
    atoms,
    *,
    protocol: dict[str, Any],
    executable: str,
    audit_dir: Path,
) -> tuple[list[float], dict[str, Any]]:
    charge = protocol["charge"]
    result = prepare_charges(
        atoms,
        {
            "source": charge["source"],
            "method": charge["method"],
            "mode": charge["mode"],
            "geometry": charge["geometry"],
            "executable": executable,
        },
        audit_dir,
    )
    values = [float(value) for value in result.charges]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("AM1-BCC returned an empty or non-finite charge vector.")
    if abs(sum(values)) > 1.0e-8:
        raise ValueError("AM1-BCC charge vector does not close to neutral.")
    return values, dict(result.provenance)


def _evaluate_case(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_dir: Path,
    work_dir: Path,
    providers: dict[str, Any],
) -> dict[str, str]:
    compound_id = row["compound_id"]
    destination = _record_path(work_dir, compound_id)
    if destination.is_file():
        existing = core.load_json(destination)
        if (
            existing.get("status") == "success"
            and existing.get("protocol_fingerprint") == fingerprint
            and existing.get("source_mol2_sha256") == row["source_mol2_sha256"]
            and core.artifact_content_sha256(existing) == existing.get("content_sha256")
        ):
            return {"compound_id": compound_id, "status": "skipped"}
        raise ValueError(f"Existing record is incompatible: {destination}")

    source_mol2 = _verify_source_mol2(row, source_dir)
    atoms = MOL2Reader(
        str(source_mol2),
        charge=int(protocol["charge"]["net_charge"]),
        mult=int(protocol["charge"]["multiplicity"]),
    )
    executables = providers["executables"]
    charges, charge_provenance = _prepare_am1bcc(
        atoms,
        protocol=protocol,
        executable=executables["antechamber"]["path"],
        audit_dir=work_dir / "provider-audit" / compound_id / "am1bcc",
    )
    charge_hash = core.sha256_bytes(core.canonical_json_bytes(charges))

    endpoint = protocol["endpoints"]["am1bcc_obc2_ace"]
    openmm_provider = OpenMMGB(
        atoms,
        charges,
        model=endpoint["openmm_model"],
        nonpolar=endpoint["openmm_nonpolar"],
        platform=providers["openmm"]["platform"],
    )
    openmm_result = openmm_provider.evaluate(atoms, need_forces=False)
    obc2_polar = (
        float(openmm_result.components_hartree["polar"]) * KCAL_PER_HARTREE
    )
    ace_nonpolar = (
        float(openmm_result.components_hartree["nonpolar"]) * KCAL_PER_HARTREE
    )
    obc2_total = float(openmm_result.energy_hartree) * KCAL_PER_HARTREE

    with tempfile.TemporaryDirectory(prefix="route1-reserve-", dir="/tmp") as temporary:
        case_dir = Path(temporary)
        (case_dir / "m.mol2").write_text(
            chagb.mol2_with_charges(
                source_mol2.read_text(encoding="utf-8"), charges
            ),
            encoding="utf-8",
        )
        chagb._run(  # noqa: SLF001
            [
                executables["parmchk2"]["path"],
                "-i",
                "m.mol2",
                "-f",
                "mol2",
                "-o",
                "f",
                "-s",
                protocol["topology"]["parmchk2_mode"],
            ],
            case_dir,
            label="parmchk2",
        )
        chagb.require_no_frcmod_nonbonded_overrides(
            (case_dir / "f").read_text(encoding="utf-8")
        )
        (case_dir / "tleap.in").write_text(
            "\n".join(
                [
                    f"source {protocol['topology']['tleap_source']}",
                    f"set default PBradii {protocol['topology']['pb_radii']}",
                    "MOL = loadmol2 m.mol2",
                    "loadamberparams f",
                    "saveamberparm MOL p c",
                    "quit",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        chagb._run(  # noqa: SLF001
            [executables["tleap"]["path"], "-f", "tleap.in"],
            case_dir,
            label="tleap",
        )
        (case_dir / "gb.in").write_text(
            chagb._gbnsr6_input(protocol), encoding="utf-8"  # noqa: SLF001
        )
        with GBNSR6_LOCK:
            chagb._run(  # noqa: SLF001
                [
                    executables["gbnsr6"]["path"],
                    "-O",
                    "-i",
                    "gb.in",
                    "-o",
                    "gb.out",
                    "-p",
                    "p",
                    "-c",
                    "c",
                ],
                case_dir,
                label="gbnsr6",
            )
        gb = chagb.parse_gbnsr6_components(
            (case_dir / "gb.out").read_text(encoding="utf-8")
        )
        (case_dir / "pb.in").write_text(
            chagb._pbsa_input(protocol), encoding="utf-8"  # noqa: SLF001
        )
        chagb._run(  # noqa: SLF001
            [
                executables["pbsa"]["path"],
                "-O",
                "-i",
                "pb.in",
                "-o",
                "pb.out",
                "-p",
                "p",
                "-c",
                "c",
            ],
            case_dir,
            label="pbsa",
        )
        pb = chagb.parse_pbsa_components(
            (case_dir / "pb.out").read_text(encoding="utf-8")
        )

    chagb_total = gb["chagb_polar"] + pb["cavity"] + pb["dispersion"]
    values_to_check = (
        obc2_polar,
        ace_nonpolar,
        obc2_total,
        gb["chagb_polar"],
        gb["surface_tension"],
        pb["cavity"],
        pb["dispersion"],
        chagb_total,
    )
    if not all(math.isfinite(value) for value in values_to_check):
        raise ValueError("A provider returned a non-finite energy component.")

    record = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-freesolv-reserve-energy-record",
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "compound_id": compound_id,
            "status": "success",
            "source_mol2_sha256": row["source_mol2_sha256"],
            "dataset_record_sha256": row["dataset_record_sha256"],
            "am1bcc_charges_e": charges,
            "am1bcc_charge_vector_sha256": charge_hash,
            "charge_provenance": charge_provenance,
            "components_kcal_mol": {
                "obc2_polar": obc2_polar,
                "ace_nonpolar": ace_nonpolar,
                "chagb_polar": gb["chagb_polar"],
                "gbnsr6_surface_tension": gb["surface_tension"],
                "pbsa_cavity": pb["cavity"],
                "pbsa_dispersion": pb["dispersion"],
            },
            "predictions_kcal_mol": {
                "am1bcc_obc2_ace": obc2_total,
                "am1bcc_chagb_pbsa_cavity_dispersion": chagb_total,
            },
            "provider_provenance": {
                "openmm": openmm_result.provenance,
                "executables_sha256": {
                    name: details["sha256"]
                    for name, details in sorted(executables.items())
                },
            },
        }
    )
    if "experimental" in json.dumps(record, sort_keys=True).lower():
        raise AssertionError("Energy record contains a label field.")
    core.write_json_atomic(destination, record)
    return {"compound_id": compound_id, "status": "success"}


def _load_energy_record(
    path: Path,
    row: dict[str, Any],
    *,
    fingerprint: str,
) -> dict[str, Any]:
    record = core.load_json(path)
    if record.get("status") != "success":
        raise ValueError(f"Energy record is not successful: {path}.")
    if record.get("compound_id") != row["compound_id"]:
        raise ValueError(f"Energy record id mismatch: {path}.")
    if record.get("protocol_fingerprint") != fingerprint:
        raise ValueError(f"Energy record protocol mismatch: {path}.")
    if record.get("source_mol2_sha256") != row["source_mol2_sha256"]:
        raise ValueError(f"Energy record source mismatch: {path}.")
    if core.artifact_content_sha256(record) != record.get("content_sha256"):
        raise ValueError(f"Energy record content hash mismatch: {path}.")
    if "experimental" in json.dumps(record, sort_keys=True).lower():
        raise ValueError(f"Energy record contains a label field: {path}.")
    return record


def seal_energy_artifact(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_source_manifest(protocol_path, protocol, manifest_path)
    work_dir = Path(args.work_dir).resolve()
    records = []
    record_hashes = {}
    for row in manifest["records"]:
        path = _record_path(work_dir, row["compound_id"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing energy record: {path}.")
        record = _load_energy_record(path, row, fingerprint=fingerprint)
        records.append(record)
        record_hashes[row["compound_id"]] = core.sha256_file(path)
    expected = int(protocol["source_evidence"]["expected_case_count"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} energy records, found {len(records)}.")
    providers = resolve_providers(protocol, args.amber_bin)
    output = Path(args.energy_artifact).resolve()
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-freesolv-reserve-label-free-energy",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "protocol_fingerprint": fingerprint,
            "route1_contract": protocol["route1_boundary"],
            "source_designation": protocol["evaluation_design"]["designation"],
            "source_manifest_sha256": core.sha256_file(manifest_path),
            "case_count": len(records),
            "endpoint_names": list(ENDPOINTS),
            "provider_provenance": providers,
            "record_file_sha256": record_hashes,
            "records": records,
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "phase": "seal-energy",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "work_dir": Path(args.work_dir).as_posix(),
                    "energy_artifact": output.relative_to(REPOSITORY_ROOT).as_posix(),
                    "amber_bin": args.amber_bin,
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    if "experimental" in json.dumps(artifact, sort_keys=True).lower():
        raise AssertionError("Sealed energy artifact contains a label field.")
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "energy_artifact": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "case_count": len(records),
            },
            indent=2,
        )
    )
    return artifact


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest = load_source_manifest(protocol_path, protocol, args.manifest)
    providers = resolve_providers(protocol, args.amber_bin)
    source_dir = Path(args.source_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    expected_workers = int(protocol["providers"]["execution_controls"]["workers"])
    if args.workers != expected_workers:
        raise ValueError(
            f"Protocol freezes --workers={expected_workers}; got {args.workers}."
        )

    successes = skips = failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _evaluate_case,
                row,
                protocol=protocol,
                fingerprint=fingerprint,
                source_dir=source_dir,
                work_dir=work_dir,
                providers=providers,
            ): row["compound_id"]
            for row in manifest["records"]
        }
        for future in as_completed(futures):
            compound_id = futures[future]
            try:
                status = future.result()["status"]
            except Exception as exc:
                failures += 1
                print(f"FAIL {compound_id}: {exc}", file=sys.stderr, flush=True)
            else:
                if status == "skipped":
                    skips += 1
                else:
                    successes += 1
                print(f"{status.upper()} {compound_id}", flush=True)
    if failures:
        raise RuntimeError(
            f"Reserve energy phase failed for {failures} cases "
            f"({successes} completed, {skips} skipped)."
        )
    print(
        f"Reserve energy phase complete: {successes} completed, "
        f"{skips} skipped, 0 failed."
    )
    seal_energy_artifact(args)


def _point_metrics(errors: list[float]) -> dict[str, float | int | None]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mse": None,
            "mae": None,
            "rmse": None,
            "max_absolute_error": None,
        }
    return {
        "n": int(values.size),
        "mse": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max_absolute_error": float(np.max(np.abs(values))),
    }


def load_energy_artifact(
    path: str | Path,
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    fingerprint: str,
    manifest_path: Path,
) -> dict[str, Any]:
    artifact = core.load_json(path)
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Sealed reserve energy-artifact content hash mismatch.")
    if artifact.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError("Sealed reserve energy-artifact protocol hash mismatch.")
    if artifact.get("protocol_fingerprint") != fingerprint:
        raise ValueError("Sealed reserve energy-artifact protocol fingerprint mismatch.")
    if artifact.get("source_manifest_sha256") != core.sha256_file(manifest_path):
        raise ValueError("Sealed reserve energy-artifact source hash mismatch.")
    if int(artifact.get("case_count", -1)) != int(
        protocol["source_evidence"]["expected_case_count"]
    ):
        raise ValueError("Sealed reserve energy-artifact case count mismatch.")
    if "experimental" in json.dumps(artifact, sort_keys=True).lower():
        raise ValueError("Sealed reserve energy artifact contains a label field.")
    return artifact


def _verify_score_sources(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evidence = protocol["source_evidence"]
    prepared_path = _repository_path(evidence["prepared_manifest_relative_path"])
    if core.sha256_file(prepared_path) != evidence["prepared_manifest_sha256"]:
        raise ValueError("Pinned prepared FreeSolv manifest hash mismatch.")
    development_path = SCRIPT_DIR / evidence["development_artifact"]
    if core.sha256_file(development_path) != evidence["development_artifact_sha256"]:
        raise ValueError("Pinned development artifact hash mismatch.")
    prepared = core.load_json(prepared_path)
    source_partition = protocol["evaluation_design"]["source_partition"]
    candidates = {
        row["compound_id"]: row
        for row in prepared["candidates"]
        if row["partition"] == source_partition
    }
    manifest_ids = {row["compound_id"] for row in manifest["records"]}
    if set(candidates) != manifest_ids:
        raise ValueError("Scoring labels do not reconcile with the source manifest.")
    for row in manifest["records"]:
        candidate = candidates[row["compound_id"]]
        if candidate["mol2_sha256"] != row["source_mol2_sha256"]:
            raise ValueError(f"Scoring MOL2 mismatch: {row['compound_id']}.")
        if candidate["dataset_record_sha256"] != row["dataset_record_sha256"]:
            raise ValueError(f"Scoring record mismatch: {row['compound_id']}.")
    return candidates, core.load_json(development_path)


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_source_manifest(protocol_path, protocol, manifest_path)
    energy_path = Path(args.energy_artifact).resolve()
    energy = load_energy_artifact(
        energy_path,
        protocol_path=protocol_path,
        protocol=protocol,
        fingerprint=fingerprint,
        manifest_path=manifest_path,
    )

    # This is the intentional unsealing boundary: labels are read only after the
    # complete label-free energy artifact has passed every hash/count check above.
    candidates, development = _verify_score_sources(protocol, manifest)
    energy_records = {
        record["compound_id"]: record for record in energy["records"]
    }
    errors = {endpoint: [] for endpoint in ENDPOINTS}
    rows = []
    for source in manifest["records"]:
        compound_id = source["compound_id"]
        candidate = candidates[compound_id]
        record = energy_records[compound_id]
        label = float(candidate["experimental_kcal_mol"])
        predictions = {
            endpoint: float(record["predictions_kcal_mol"][endpoint])
            for endpoint in ENDPOINTS
        }
        signed_errors = {
            endpoint: predictions[endpoint] - label for endpoint in ENDPOINTS
        }
        for endpoint in ENDPOINTS:
            errors[endpoint].append(signed_errors[endpoint])
        rows.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": label,
                "experimental_uncertainty_kcal_mol": float(
                    candidate["experimental_uncertainty_kcal_mol"]
                ),
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": signed_errors,
                "bins": source["bins"],
            }
        )

    statistics = protocol["statistics"]
    expected = int(protocol["source_evidence"]["expected_case_count"])
    metrics = {
        endpoint: core.summarize_errors(
            errors[endpoint],
            expected_count=expected,
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, endpoint in enumerate(ENDPOINTS)
    }
    paired = chagb.paired_absolute_error_gain(
        errors["am1bcc_obc2_ace"],
        errors["am1bcc_chagb_pbsa_cavity_dispersion"],
        resamples=int(statistics["bootstrap_resamples"]),
        confidence=float(statistics["bootstrap_confidence"]),
        seed=int(statistics["bootstrap_seed"]) + 100,
    )
    strata: dict[str, Any] = {}
    for bin_name in statistics["strata"]:
        strata[bin_name] = {}
        for value in sorted({row["bins"][bin_name] for row in rows}):
            selected = [row for row in rows if row["bins"][bin_name] == value]
            strata[bin_name][value] = {
                endpoint: _point_metrics(
                    [
                        row["signed_errors_kcal_mol"][endpoint]
                        for row in selected
                    ]
                )
                for endpoint in ENDPOINTS
            }

    development_methods = development["methods"]
    development_paired = development["paired_absolute_error_gain_vs_obc2_ace"][
        "chagb_pbsa_cavity_dispersion"
    ]
    dev_to_reserve = {
        "am1bcc_obc2_ace": {
            metric: float(metrics["am1bcc_obc2_ace"][metric])
            - float(development_methods["obc2_ace"][metric])
            for metric in ("mae", "rmse", "max_absolute_error")
        },
        "am1bcc_chagb_pbsa_cavity_dispersion": {
            metric: float(
                metrics["am1bcc_chagb_pbsa_cavity_dispersion"][metric]
            )
            - float(development_methods["chagb_pbsa_cavity_dispersion"][metric])
            for metric in ("mae", "rmse", "max_absolute_error")
        },
        "paired_mean_mae_gain_kcal_mol": (
            float(paired["mean_mae_gain_kcal_mol"])
            - float(development_paired["mean_mae_gain_kcal_mol"])
        ),
    }

    rule = protocol["pre_registered_decision_rule"]
    baseline_rule = rule["baseline_reporting_gate"]
    baseline_metrics = metrics["am1bcc_obc2_ace"]
    baseline_pass = (
        baseline_metrics["failure_count"] == 0
        and baseline_metrics["mae"] <= baseline_rule["mae_max_kcal_mol"]
        and baseline_metrics["rmse"] <= baseline_rule["rmse_max_kcal_mol"]
        and baseline_metrics["max_absolute_error"]
        <= baseline_rule["max_absolute_error_max_kcal_mol"]
    )
    candidate_rule = rule["retain_chagb_sp_accuracy_profile_only_if"]
    candidate_metrics = metrics["am1bcc_chagb_pbsa_cavity_dispersion"]
    candidate_gate_results = {
        "full_coverage": (
            baseline_metrics["failure_count"] == 0
            and candidate_metrics["failure_count"] == 0
            and baseline_metrics["n"] == expected
            and candidate_metrics["n"] == expected
        ),
        "paired_mean_mae_gain_at_least_threshold": (
            paired["mean_mae_gain_kcal_mol"]
            >= candidate_rule["paired_mean_mae_gain_min_kcal_mol"]
        ),
        "paired_bootstrap_ci_lower_above_zero": paired["bootstrap_ci"][0] > 0.0,
        "rmse_not_worse": candidate_metrics["rmse"] <= baseline_metrics["rmse"],
        "max_absolute_error_not_worse": (
            candidate_metrics["max_absolute_error"]
            <= baseline_metrics["max_absolute_error"]
        ),
    }
    candidate_pass = all(candidate_gate_results.values())
    largest = sorted(
        rows,
        key=lambda row: abs(
            row["signed_errors_kcal_mol"][
                "am1bcc_chagb_pbsa_cavity_dispersion"
            ]
        ),
        reverse=True,
    )
    output = Path(args.score_artifact).resolve()
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-freesolv-label-exposed-reserve-score",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "protocol_fingerprint": fingerprint,
            "claim_scope": protocol["claim_scope"],
            "route1_contract": protocol["route1_boundary"],
            "evaluation_design": protocol["evaluation_design"],
            "source_manifest_sha256": core.sha256_file(manifest_path),
            "sealed_energy_artifact_sha256": core.sha256_file(energy_path),
            "sealed_energy_content_sha256": energy["content_sha256"],
            "case_count": len(rows),
            "methods": metrics,
            "paired_absolute_error_gain": paired,
            "strata": strata,
            "development_to_reserve_metric_delta": dev_to_reserve,
            "decision": {
                "baseline_reporting_gate_passed": baseline_pass,
                "candidate_gate_results": candidate_gate_results,
                "chagb_sp_accuracy_profile_retained": candidate_pass,
                "production_default_changed": False,
                "runtime_force_capability_established": False,
                "post_score_tuning_allowed": False,
                "failure_action": rule["failure_action"],
            },
            "largest_candidate_absolute_errors": largest[:10],
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "phase": "score",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "energy_artifact": energy_path.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "score_artifact": output.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                },
                repository_root=REPOSITORY_ROOT,
            ),
            "records": rows,
        }
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "score_artifact": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "methods": {
                    endpoint: {
                        metric: metrics[endpoint][metric]
                        for metric in ("mae", "rmse", "max_absolute_error")
                    }
                    for endpoint in ENDPOINTS
                },
                "paired_absolute_error_gain": paired,
                "decision": artifact["decision"],
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare-manifest", "run", "seal-energy", "score", "all"),
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--energy-artifact", default=DEFAULT_ENERGY_ARTIFACT)
    parser.add_argument("--score-artifact", default=DEFAULT_SCORE_ARTIFACT)
    parser.add_argument("--amber-bin")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "prepare-manifest":
        prepare_source_manifest(args)
        return
    if args.phase in {"run", "all"}:
        run_energy(args)
    elif args.phase == "seal-energy":
        seal_energy_artifact(args)
    if args.phase in {"score", "all"}:
        score(args)


if __name__ == "__main__":
    main()
