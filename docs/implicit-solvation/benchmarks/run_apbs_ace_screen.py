#!/usr/bin/env python3
"""Run the label-blind Route-1 APBS molecular-surface PB plus ACE screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    summarize_errors,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.apbs_pb import (  # noqa: E402
    APBSLPB,
)
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_MOL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184
METHOD_KEYS = (
    "am1bcc_obc2_ace",
    "am1bcc_apbs_mol_sasa",
    "am1bcc_apbs_mol_ace",
)
PROTOCOL_IDS = {
    "maple-route1-am1bcc-apbs-mol-ace-v1",
    "maple-route1-am1bcc-apbs-mol-ace-fine-v1",
}


def load_evaluation_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only APBS/ACE protocol schema version 1 is supported.")
    if protocol.get("protocol_id") not in PROTOCOL_IDS:
        raise ValueError("Unexpected APBS/ACE protocol ID.")
    boundary = protocol.get("execution_boundary", {})
    required = {
        "energy_phase_reads_experimental_labels": False,
        "confirmation_remains_closed": True,
        "no_experimental_fit_or_residual_model": True,
        "no_gas_phase_mm_energy": True,
        "no_mlip_retraining": True,
    }
    for key, expected in required.items():
        if boundary.get(key) is not expected:
            raise ValueError(f"Protocol execution boundary must set {key}={expected}.")
    polar = protocol["polar"]
    if polar.get("surface_definition") != "mol":
        raise ValueError("This screen is frozen to the APBS molecular surface.")
    if polar.get("calculation", {}).get("force") != "no":
        raise ValueError("APBS molecular-surface PB must remain SP-energy-only.")
    expected = int(protocol["source_evidence"]["expected_case_count"])
    if expected <= 0:
        raise ValueError("expected_case_count must be positive.")
    parent = protocol.get("parent_numerical_evidence")
    if parent is not None:
        for key, hash_key in (
            ("parent_protocol", "parent_protocol_sha256"),
            ("grid_followup_protocol", "grid_followup_protocol_sha256"),
            ("grid_followup_summary", "grid_followup_summary_sha256"),
        ):
            artifact = protocol_path.parent / parent[key]
            if sha256_file(artifact) != parent[hash_key]:
                raise ValueError(f"Fine-grid parent evidence hash mismatch: {key}.")
        if parent.get("selection_was_label_free") is not True:
            raise ValueError("Fine-grid selection must remain label-free.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _artifact_path(protocol_path: Path, relative: str) -> Path:
    path = protocol_path.parent / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_label_free_manifest(
    protocol_path: str | Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    evidence = protocol["source_evidence"]
    path = _artifact_path(protocol_path, evidence["source_manifest"])
    if sha256_file(path) != evidence["source_manifest_sha256"]:
        raise ValueError("Frozen APBS/ACE source manifest hash mismatch.")
    manifest = load_json(path)
    if int(manifest.get("case_count", -1)) != int(evidence["expected_case_count"]):
        raise ValueError("Source manifest case count does not match the protocol.")
    payload = json.dumps(manifest, sort_keys=True).lower()
    if "experimental" in payload:
        raise ValueError("Energy source manifest contains an experimental-label field.")
    ids = [row["compound_id"] for row in manifest["records"]]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Source manifest compound IDs must be sorted and unique.")
    selected = manifest["grid_sensitivity_selection"]["compound_ids"]
    if (
        selected != sorted(selected)
        or len(selected) != len(set(selected))
        or not set(selected).issubset(ids)
    ):
        raise ValueError(
            "Grid-sensitivity selection must be sorted, unique, and known."
        )
    return manifest


def resolve_apbs(
    protocol: dict[str, Any], executable: str | Path | None = None
) -> dict[str, str]:
    specification = protocol["providers"]["apbs"]
    requested = str(executable or specification["default_path"])
    located = shutil.which(requested)
    if located is None:
        candidate = Path(requested)
        if not candidate.is_file():
            raise FileNotFoundError(f"Required APBS executable not found: {requested}")
        located = str(candidate.resolve())
    observed = sha256_file(located)
    if observed != specification["sha256"]:
        raise ValueError(
            "APBS executable hash mismatch: "
            f"expected {specification['sha256']}, observed {observed}."
        )
    completed = subprocess.run(
        [located, "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30.0,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    match = re.search(
        r"\b(?:APBS\s+Version|Version)\s+([0-9]+(?:\.[0-9]+)+)",
        completed.stdout + "\n" + completed.stderr,
        re.IGNORECASE,
    )
    version = match.group(1) if match else "unknown"
    if version != specification["required_version"]:
        raise ValueError(
            f"APBS version mismatch: expected {specification['required_version']}, "
            f"observed {version}."
        )
    return {
        "path": str(Path(located).resolve()),
        "sha256": observed,
        "version": version,
    }


def apply_execution_controls(protocol: dict[str, Any], workers: int) -> None:
    controls = protocol["execution_controls"]
    maximum = int(controls["maximum_concurrent_apbs_processes"])
    if workers > maximum:
        raise ValueError(
            f"--workers={workers} exceeds the frozen APBS concurrency limit {maximum}."
        )
    for name, value in controls["subprocess_environment"].items():
        os.environ[str(name)] = str(value)


def _provider(
    atoms,
    charges_e: Iterable[float],
    *,
    protocol: dict[str, Any],
    executable: str,
    grid: dict[str, Any],
) -> APBSLPB:
    polar = protocol["polar"]
    nonpolar = protocol["nonpolar_endpoints"]["apbs_sasa_control"]
    return APBSLPB(
        atoms,
        charges_e,
        executable=executable,
        grid_spacing=float(grid["spacing_angstrom"]),
        grid_points=int(grid["points_per_axis"]),
        probe_radius=float(polar["solvent_probe_radius_angstrom"]),
        surface_tension=float(nonpolar["surface_tension_kj_mol_angstrom2"]),
        pressure=float(nonpolar["pressure_kj_mol_angstrom3"]),
    )


def _input_hashes(provider: APBSLPB, atoms) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="maple-apbs-input-", dir="/tmp") as tmp:
        pqr = Path(tmp) / "molecule.pqr"
        provider.write_pqr(pqr, atoms)
        rendered = provider.render_input(pqr.name).encode("utf-8")
        return {
            "pqr_sha256": sha256_file(pqr),
            "apbs_input_sha256": sha256_bytes(rendered),
        }


def evaluate_apbs_provider(
    atoms,
    charges_e: Iterable[float],
    *,
    protocol: dict[str, Any],
    executable: str,
    grid: dict[str, Any],
) -> dict[str, Any]:
    provider = _provider(
        atoms,
        charges_e,
        protocol=protocol,
        executable=executable,
        grid=grid,
    )
    input_hashes = _input_hashes(provider, atoms)
    result = provider.evaluate(atoms, need_forces=False)
    polar = float(result.components_hartree["polar"]) * KCAL_PER_MOL_PER_HARTREE
    nonpolar = (
        float(result.components_hartree["nonpolar"]) * KCAL_PER_MOL_PER_HARTREE
    )
    if not math.isfinite(polar) or not math.isfinite(nonpolar):
        raise ValueError("APBS returned a non-finite energy component.")
    return {
        "polar_kcal_mol": polar,
        "apbs_sasa_nonpolar_kcal_mol": nonpolar,
        "input_sha256": input_hashes,
        "provenance": result.provenance,
    }


def _record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "records" / f"{compound_id}.json"


def _grid_record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "grid_records" / f"{compound_id}.json"


def _verify_source_row(
    row: dict[str, Any], source_work_dir: Path
) -> tuple[Path, list[float], str]:
    compound_id = row["compound_id"]
    source_mol2 = source_work_dir / row["source_mol2_relative_path"]
    if not source_mol2.is_file():
        raise FileNotFoundError(source_mol2)
    observed_mol2_hash = sha256_file(source_mol2)
    if observed_mol2_hash != row["source_mol2_sha256"]:
        raise ValueError(f"Frozen source MOL2 hash mismatch for {compound_id}.")
    charges = [float(value) for value in row["am1bcc_charges_e"]]
    if not charges or not all(math.isfinite(value) for value in charges):
        raise ValueError(f"AM1-BCC charge vector is invalid for {compound_id}.")
    return source_mol2, charges, observed_mol2_hash


def _evaluate_case(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_work_dir: Path,
    work_dir: Path,
    apbs: dict[str, str],
) -> dict[str, str]:
    compound_id = row["compound_id"]
    destination = _record_path(work_dir, compound_id)
    if destination.is_file():
        existing = load_json(destination)
        if (
            existing.get("status") == "success"
            and existing.get("protocol_fingerprint") == fingerprint
            and existing.get("source_record_sha256") == row["source_record_sha256"]
            and existing.get("source_mol2_sha256") == row["source_mol2_sha256"]
        ):
            return {"compound_id": compound_id, "status": "skipped"}
        raise ValueError(f"Existing record is incompatible or failed: {destination}")

    source_mol2, charges, observed_mol2_hash = _verify_source_row(
        row, source_work_dir
    )
    atoms = MOL2Reader(str(source_mol2), charge=0, mult=1)
    if len(atoms) != len(charges):
        raise ValueError(f"MOL2/AM1-BCC charge length mismatch for {compound_id}.")
    observed_span = float(np.ptp(np.asarray(atoms.positions), axis=0).max())
    if abs(observed_span - float(row["maximum_cartesian_span_angstrom"])) > 1.0e-9:
        raise ValueError(f"Frozen geometry span changed for {compound_id}.")
    evaluated = evaluate_apbs_provider(
        atoms,
        charges,
        protocol=protocol,
        executable=apbs["path"],
        grid=protocol["polar"]["grid"],
    )
    ace = float(row["openmm_ace_nonpolar_kcal_mol"])
    if not math.isfinite(ace):
        raise ValueError(f"Frozen ACE component is non-finite for {compound_id}.")
    polar = evaluated["polar_kcal_mol"]
    apbs_sasa = evaluated["apbs_sasa_nonpolar_kcal_mol"]
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "compound_id": compound_id,
        "source_partition": protocol["source_partition"],
        "source_record_sha256": row["source_record_sha256"],
        "source_mol2_sha256": observed_mol2_hash,
        "am1bcc_charge_vector_sha256": sha256_bytes(canonical_json_bytes(charges)),
        "status": "success",
        "components_kcal_mol": {
            "apbs_mol_lpb_polar": polar,
            "apbs_sasa_nonpolar": apbs_sasa,
            "openmm_ace_nonpolar": ace,
        },
        "predictions_kcal_mol": {
            "am1bcc_apbs_mol_sasa": polar + apbs_sasa,
            "am1bcc_apbs_mol_ace": polar + ace,
        },
        "input_sha256": evaluated["input_sha256"],
        "provider_provenance": {
            "polar_and_sasa": evaluated["provenance"],
            "ace": {
                "provider": "openmm",
                "provider_version": protocol["providers"]["openmm"][
                    "required_version"
                ],
                "model": "ace",
                "radius_profile": "mbondi2",
                "source_record_sha256": row["source_record_sha256"],
                "gas_phase_mm_energy_used": False,
            },
        },
        "provider_sha256": {"apbs": apbs["sha256"]},
    }
    write_json_atomic(destination, record)
    return {"compound_id": compound_id, "status": "success"}


def _run_parallel(
    rows: list[dict[str, Any]],
    worker,
    *,
    workers: int,
    phase_name: str,
) -> None:
    successes = skips = failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, row): row["compound_id"] for row in rows}
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
            f"{phase_name} failed for {failures} cases "
            f"({successes} completed, {skips} skipped)."
        )
    print(
        f"{phase_name} complete: {successes} completed, {skips} skipped, 0 failed."
    )


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_evaluation_protocol(protocol_path)
    apply_execution_controls(protocol, args.workers)
    manifest = load_label_free_manifest(protocol_path, protocol)
    apbs = resolve_apbs(protocol, args.apbs)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest["records"]
    if args.limit is not None:
        rows = rows[: args.limit]
    _run_parallel(
        rows,
        lambda row: _evaluate_case(
            row,
            protocol=protocol,
            fingerprint=fingerprint,
            source_work_dir=source_work_dir,
            work_dir=work_dir,
            apbs=apbs,
        ),
        workers=args.workers,
        phase_name="Energy phase",
    )


def _evaluate_grid_case(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_work_dir: Path,
    work_dir: Path,
    apbs: dict[str, str],
) -> dict[str, str]:
    compound_id = row["compound_id"]
    destination = _grid_record_path(work_dir, compound_id)
    main_path = _record_path(work_dir, compound_id)
    if not main_path.is_file():
        raise FileNotFoundError(f"Main-grid record is absent: {main_path}")
    main_hash = sha256_file(main_path)
    main = load_json(main_path)
    if main.get("protocol_fingerprint") != fingerprint:
        raise ValueError(f"Main-grid protocol mismatch for {compound_id}.")
    if destination.is_file():
        existing = load_json(destination)
        if (
            existing.get("status") == "success"
            and existing.get("protocol_fingerprint") == fingerprint
            and existing.get("main_grid_record_sha256") == main_hash
        ):
            return {"compound_id": compound_id, "status": "skipped"}
        raise ValueError(
            f"Existing grid record is incompatible or failed: {destination}"
        )
    source_mol2, charges, _observed_hash = _verify_source_row(row, source_work_dir)
    atoms = MOL2Reader(str(source_mol2), charge=0, mult=1)
    evaluated = evaluate_apbs_provider(
        atoms,
        charges,
        protocol=protocol,
        executable=apbs["path"],
        grid=protocol["grid_sensitivity"]["reference_grid"],
    )
    main_polar = float(main["components_kcal_mol"]["apbs_mol_lpb_polar"])
    reference_polar = evaluated["polar_kcal_mol"]
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "compound_id": compound_id,
        "status": "success",
        "selection_policy": "label-free frozen source-manifest selection",
        "source_mol2_sha256": row["source_mol2_sha256"],
        "main_grid_record_sha256": main_hash,
        "main_grid": protocol["polar"]["grid"],
        "reference_grid": protocol["grid_sensitivity"]["reference_grid"],
        "main_grid_polar_kcal_mol": main_polar,
        "reference_grid_polar_kcal_mol": reference_polar,
        "signed_difference_reference_minus_main_kcal_mol": (
            reference_polar - main_polar
        ),
        "reference_input_sha256": evaluated["input_sha256"],
        "provider_sha256": {"apbs": apbs["sha256"]},
    }
    write_json_atomic(destination, record)
    return {"compound_id": compound_id, "status": "success"}


def run_grid_sensitivity(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_evaluation_protocol(protocol_path)
    apply_execution_controls(protocol, args.workers)
    manifest = load_label_free_manifest(protocol_path, protocol)
    selected = set(manifest["grid_sensitivity_selection"]["compound_ids"])
    rows = [row for row in manifest["records"] if row["compound_id"] in selected]
    apbs = resolve_apbs(protocol, args.apbs)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    _run_parallel(
        rows,
        lambda row: _evaluate_grid_case(
            row,
            protocol=protocol,
            fingerprint=fingerprint,
            source_work_dir=source_work_dir,
            work_dir=work_dir,
            apbs=apbs,
        ),
        workers=args.workers,
        phase_name="Grid-sensitivity phase",
    )


def _point_metrics(errors: Iterable[float]) -> dict[str, float | int | None]:
    values = np.asarray(list(errors), dtype=np.float64)
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


def paired_absolute_error_gain(
    baseline_errors: Iterable[float],
    candidate_errors: Iterable[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    baseline = np.asarray(list(baseline_errors), dtype=np.float64)
    candidate = np.asarray(list(candidate_errors), dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("Paired gain requires non-empty, equally sized error arrays.")
    gains = np.abs(baseline) - np.abs(candidate)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, gains.size, size=(resamples, gains.size))
    samples = np.mean(gains[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    tolerance = 1.0e-12
    return {
        "n": int(gains.size),
        "mean_mae_gain_kcal_mol": float(np.mean(gains)),
        "bootstrap_ci": [
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1.0 - alpha)),
        ],
        "bootstrap_probability_gain_gt_zero": float(np.mean(samples > 0.0)),
        "case_outcomes": {
            "improved": int(np.sum(gains > tolerance)),
            "unchanged": int(np.sum(np.abs(gains) <= tolerance)),
            "worsened": int(np.sum(gains < -tolerance)),
        },
    }


def _verify_summary_sources(
    protocol_path: Path, protocol: dict[str, Any], source_work_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = protocol["source_evidence"]
    for key, hash_key in (
        ("base_protocol", "base_protocol_sha256"),
        ("development_summary", "development_summary_sha256"),
        ("strongest_energy_only_summary", "strongest_energy_only_summary_sha256"),
    ):
        path = _artifact_path(protocol_path, evidence[key])
        if sha256_file(path) != evidence[hash_key]:
            raise ValueError(f"Frozen source artifact hash mismatch: {key}.")
    development = load_json(
        _artifact_path(protocol_path, evidence["development_summary"])
    )
    if development["protocol_fingerprint"] != evidence["base_protocol_fingerprint"]:
        raise ValueError("Development summary base protocol fingerprint mismatch.")
    strongest = load_json(
        _artifact_path(protocol_path, evidence["strongest_energy_only_summary"])
    )
    manifest = load_label_free_manifest(protocol_path, protocol)
    for row in manifest["records"]:
        source_record = source_work_dir / row["source_record_relative_path"]
        if sha256_file(source_record) != row["source_record_sha256"]:
            raise ValueError(f"Source record hash mismatch for {row['compound_id']}.")
        source_mol2 = source_work_dir / row["source_mol2_relative_path"]
        if sha256_file(source_mol2) != row["source_mol2_sha256"]:
            raise ValueError(f"Source MOL2 hash mismatch for {row['compound_id']}.")
    return manifest, development, strongest


def _grid_summary(
    manifest: dict[str, Any],
    *,
    work_dir: Path,
    fingerprint: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    differences: list[float] = []
    record_hashes: dict[str, str] = {}
    for compound_id in manifest["grid_sensitivity_selection"]["compound_ids"]:
        path = _grid_record_path(work_dir, compound_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing grid-sensitivity result: {path}")
        record = load_json(path)
        if (
            record.get("status") != "success"
            or record.get("protocol_fingerprint") != fingerprint
        ):
            raise ValueError(f"Invalid grid-sensitivity result: {path}")
        main_path = _record_path(work_dir, compound_id)
        if record["main_grid_record_sha256"] != sha256_file(main_path):
            raise ValueError(f"Grid result main-record hash mismatch: {compound_id}")
        differences.append(
            float(record["signed_difference_reference_minus_main_kcal_mol"])
        )
        record_hashes[compound_id] = sha256_file(path)
    absolute = np.abs(np.asarray(differences, dtype=np.float64))
    gates = protocol["grid_sensitivity"]
    result = {
        "n": len(differences),
        "mean_signed_difference_kcal_mol": float(np.mean(differences)),
        "mean_absolute_difference_kcal_mol": float(np.mean(absolute)),
        "p90_absolute_difference_kcal_mol": float(np.quantile(absolute, 0.9)),
        "maximum_absolute_difference_kcal_mol": float(np.max(absolute)),
    }
    result["gate"] = {
        "maximum_threshold_kcal_mol": float(
            gates["maximum_absolute_difference_kcal_mol"]
        ),
        "p90_threshold_kcal_mol": float(
            gates["p90_absolute_difference_kcal_mol"]
        ),
        "maximum_pass": (
            result["maximum_absolute_difference_kcal_mol"]
            <= float(gates["maximum_absolute_difference_kcal_mol"])
        ),
        "p90_pass": (
            result["p90_absolute_difference_kcal_mol"]
            <= float(gates["p90_absolute_difference_kcal_mol"])
        ),
    }
    result["gate"]["passed"] = (
        result["gate"]["maximum_pass"] and result["gate"]["p90_pass"]
    )
    return result, record_hashes


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_evaluation_protocol(protocol_path)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    manifest, development, strongest = _verify_summary_sources(
        protocol_path, protocol, source_work_dir
    )
    expected = int(protocol["source_evidence"]["expected_case_count"])
    errors: dict[str, list[float]] = {key: [] for key in METHOD_KEYS}
    rows: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}

    for source in manifest["records"]:
        compound_id = source["compound_id"]
        result_path = _record_path(work_dir, compound_id)
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing energy result: {result_path}")
        result = load_json(result_path)
        if (
            result.get("status") != "success"
            or result.get("protocol_fingerprint") != fingerprint
        ):
            raise ValueError(f"Invalid APBS/ACE energy result: {result_path}")
        if result.get("source_record_sha256") != source["source_record_sha256"]:
            raise ValueError(f"Energy result source mismatch: {compound_id}")
        if result.get("source_mol2_sha256") != source["source_mol2_sha256"]:
            raise ValueError(f"Energy result MOL2 mismatch: {compound_id}")
        charge_hash = sha256_bytes(canonical_json_bytes(source["am1bcc_charges_e"]))
        if result.get("am1bcc_charge_vector_sha256") != charge_hash:
            raise ValueError(f"Energy result charge-vector mismatch: {compound_id}")
        if "experimental" in json.dumps(result, sort_keys=True).lower():
            raise ValueError(
                f"Energy result leaked an experimental label: {compound_id}"
            )
        record_hashes[compound_id] = sha256_file(result_path)

        source_record = load_json(
            source_work_dir / source["source_record_relative_path"]
        )
        experimental = float(source_record["experimental_kcal_mol"])
        predictions = {
            "am1bcc_obc2_ace": float(source_record["predicted_kcal_mol"]),
            **{
                key: float(value)
                for key, value in result["predictions_kcal_mol"].items()
            },
        }
        signed_errors = {
            key: value - experimental for key, value in predictions.items()
        }
        for key in METHOD_KEYS:
            errors[key].append(signed_errors[key])
        rows.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": experimental,
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": signed_errors,
                "bins": source_record["bins"],
            }
        )

    statistics = protocol["statistics"]
    method_metrics = {
        key: summarize_errors(
            errors[key],
            expected_count=expected,
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]),
        )
        for key in METHOD_KEYS
    }
    paired = {
        "obc2_ace_to_apbs_mol_ace": paired_absolute_error_gain(
            errors["am1bcc_obc2_ace"],
            errors["am1bcc_apbs_mol_ace"],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]),
        ),
        "apbs_mol_sasa_to_apbs_mol_ace": paired_absolute_error_gain(
            errors["am1bcc_apbs_mol_sasa"],
            errors["am1bcc_apbs_mol_ace"],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + 1,
        ),
    }
    strata: dict[str, Any] = {}
    for bin_name in ("element_class", "size", "heteroatom_count", "flexibility"):
        values = sorted({row["bins"][bin_name] for row in rows})
        strata[bin_name] = {}
        for value in values:
            selected = [row for row in rows if row["bins"][bin_name] == value]
            strata[bin_name][value] = {
                key: _point_metrics(
                    row["signed_errors_kcal_mol"][key] for row in selected
                )
                for key in METHOD_KEYS
            }
    grid, grid_hashes = _grid_summary(
        manifest,
        work_dir=work_dir,
        fingerprint=fingerprint,
        protocol=protocol,
    )
    gates = protocol["prospective_decision_gates"]
    candidate = method_metrics["am1bcc_apbs_mol_ace"]
    baseline = method_metrics["am1bcc_obc2_ace"]
    apbs_control = method_metrics["am1bcc_apbs_mol_sasa"]
    baseline_gain = paired["obc2_ace_to_apbs_mol_ace"]
    decision_checks = {
        "coverage": len(rows) == gates["coverage"]["required_success_count"],
        "minimum_mae_gain": (
            baseline_gain["mean_mae_gain_kcal_mol"]
            >= gates["accuracy_against_obc2_ace"]["minimum_mae_gain_kcal_mol"]
        ),
        "paired_ci_lower_above_zero": baseline_gain["bootstrap_ci"][0] > 0.0,
        "rmse_not_above_obc2_ace": candidate["rmse"] <= baseline["rmse"],
        "mae_below_apbs_sasa": candidate["mae"] < apbs_control["mae"],
        "rmse_below_apbs_sasa": candidate["rmse"] < apbs_control["rmse"],
        "grid_sensitivity": grid["gate"]["passed"],
    }
    pass_result = gates["product_role"].get(
        "pass_result",
        "eligible_for_explicit_sp_runtime_integration_pending_confirmation",
    )
    decision = (
        pass_result
        if all(decision_checks.values())
        else "rejected_for_runtime_promotion"
    )
    ranked = sorted(
        rows,
        key=lambda row: abs(row["signed_errors_kcal_mol"]["am1bcc_apbs_mol_ace"]),
        reverse=True,
    )
    apbs = resolve_apbs(protocol, args.apbs)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "claim_scope": protocol["claim_scope"],
        "selection_disclosure": protocol["selection_disclosure"],
        "parent_numerical_evidence": protocol.get("parent_numerical_evidence"),
        "case_count": len(rows),
        "success_count": len(rows),
        "failure_count": expected - len(rows),
        "label_use_boundary": {
            "energy_phase": "No experimental labels read.",
            "summary_phase": (
                "Experimental FreeSolv values read only after all energy and "
                "grid records existed."
            ),
            "experimental_fit_or_residual_model": False,
            "gas_phase_mm_energy": False,
            "mlip_retraining": False,
        },
        "methods": method_metrics,
        "paired_absolute_error_gain": paired,
        "strata": strata,
        "grid_sensitivity": grid,
        "decision": {
            "result": decision,
            "checks": decision_checks,
            "scope": "SP energy only; APBS molecular-surface forces are not exposed.",
            "confirmation_remains_closed": True,
        },
        "strongest_prior_energy_only_endpoint": {
            "method": "am1bcc_chagb_cavity_dispersion",
            "metrics": strongest["methods"]["am1bcc_chagb_cavity_dispersion"],
            "summary_sha256": protocol["source_evidence"][
                "strongest_energy_only_summary_sha256"
            ],
        },
        "largest_candidate_absolute_errors": ranked[:10],
        "record_sha256": record_hashes,
        "grid_record_sha256": grid_hashes,
        "provider_sha256": {"apbs": apbs["sha256"]},
        "source_development_summary_sha256": protocol["source_evidence"][
            "development_summary_sha256"
        ],
        "source_baseline_metrics": development["methods"]["am1bcc/obc2"],
        "interpretation": (
            "Development-only fixed-geometry comparison. APBS supplies only the "
            "molecular-surface LPBE polar term; the APBS APOLAR control and the "
            "OpenMM ACE candidate are mutually exclusive nonpolar endpoints. "
            "No gas-phase MLIP or MM energy enters the scored correction."
        ),
    }
    write_json_atomic(args.summary, summary)
    print(f"Wrote deterministic summary to {Path(args.summary).resolve()}")
    return summary


def _default_paths() -> dict[str, Path]:
    return {
        "protocol": SCRIPT_DIR / "apbs_ace_protocol.json",
        "source_work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
        ),
        "work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-apbs-ace-route1-20260725"
        ),
        "summary": SCRIPT_DIR / "freesolv-am1bcc-apbs-ace-2026-07-25.json",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("run", "grid", "summarize", "all")
    )
    parser.add_argument("--protocol", default=str(defaults["protocol"]))
    parser.add_argument("--source-work-dir", default=str(defaults["source_work_dir"]))
    parser.add_argument("--work-dir", default=str(defaults["work_dir"]))
    parser.add_argument("--summary", default=str(defaults["summary"]))
    parser.add_argument("--apbs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.phase in {"run", "all"}:
        run_energy(args)
    if args.phase in {"grid", "all"}:
        run_grid_sensitivity(args)
    if args.phase in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
