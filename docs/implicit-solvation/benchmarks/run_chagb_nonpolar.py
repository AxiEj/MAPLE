#!/usr/bin/env python3
"""Run and summarize the label-blind Route-1 CHA-GB/nonpolar benchmark."""

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
import threading
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
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

GBNSR6_PATTERN = re.compile(
    r"EGB\s*=\s*([-+0-9.Ee]+).*?ESURF\s*=\s*([-+0-9.Ee]+)", re.DOTALL
)
PBSA_PATTERN = re.compile(r"ECAVITY\s*=\s*([-+0-9.Ee]+)\s+EDISPER\s*=\s*([-+0-9.Ee]+)")
METHOD_KEYS = (
    "am1bcc_obc2_ace",
    "am1bcc_chagb_surface_tension",
    "am1bcc_chagb_cavity_dispersion",
)
GBNSR6_LOCK = threading.Lock()


def load_evaluation_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only CHA-GB/nonpolar protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-am1bcc-chagb-nonpolar-v1":
        raise ValueError("Unexpected CHA-GB/nonpolar protocol ID.")
    boundary = protocol.get("execution_boundary", {})
    if boundary.get("energy_phase_reads_experimental_labels") is not False:
        raise ValueError("Energy execution must be explicitly label-blind.")
    if boundary.get("no_experimental_fit_or_residual_model") is not True:
        raise ValueError(
            "Protocol must prohibit experimental fits and residual models."
        )
    expected = int(protocol["source_evidence"]["expected_case_count"])
    if expected <= 0:
        raise ValueError("expected_case_count must be positive.")
    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, fingerprint


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
        raise ValueError("Frozen label-free source manifest hash mismatch.")
    manifest = load_json(path)
    if int(manifest.get("case_count", -1)) != int(evidence["expected_case_count"]):
        raise ValueError("Source manifest case count does not match the protocol.")
    payload = json.dumps(manifest, sort_keys=True).lower()
    if "experimental" in payload:
        raise ValueError("Energy source manifest contains an experimental-label field.")
    ids = [row["compound_id"] for row in manifest["records"]]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Source manifest compound IDs must be sorted and unique.")
    return manifest


def resolve_executables(
    protocol: dict[str, Any], amber_bin: str | Path | None = None
) -> dict[str, dict[str, str]]:
    resolved: dict[str, dict[str, str]] = {}
    root = Path(amber_bin).resolve() if amber_bin else None
    for name, specification in protocol["providers"]["executables"].items():
        candidate = root / name if root else Path(specification["default_path"])
        if not candidate.is_file():
            located = shutil.which(str(candidate))
            if located is None:
                raise FileNotFoundError(f"Required executable not found: {candidate}")
            candidate = Path(located)
        observed = sha256_file(candidate)
        if observed != specification["sha256"]:
            raise ValueError(
                f"{name} executable hash mismatch: expected {specification['sha256']}, "
                f"observed {observed}."
            )
        resolved[name] = {"path": str(candidate), "sha256": observed}
    return resolved


def parse_gbnsr6_components(text: str) -> dict[str, float]:
    matches = GBNSR6_PATTERN.findall(text)
    if not matches:
        raise ValueError("GBNSR6 output does not contain final EGB/ESURF components.")
    polar, surface = (float(value) for value in matches[-1])
    if not math.isfinite(polar) or not math.isfinite(surface):
        raise ValueError("GBNSR6 returned a non-finite energy component.")
    return {"chagb_polar": polar, "surface_tension": surface}


def parse_pbsa_components(text: str) -> dict[str, float]:
    matches = PBSA_PATTERN.findall(text)
    if not matches:
        raise ValueError(
            "PBSA output does not contain final ECAVITY/EDISPER components."
        )
    cavity, dispersion = (float(value) for value in matches[-1])
    if not math.isfinite(cavity) or not math.isfinite(dispersion):
        raise ValueError("PBSA returned a non-finite energy component.")
    return {"cavity": cavity, "dispersion": dispersion}


def mol2_with_charges(text: str, charges_e: Iterable[float]) -> str:
    charges = [float(value) for value in charges_e]
    if not charges or not all(math.isfinite(value) for value in charges):
        raise ValueError("AM1-BCC charge vector must be non-empty and finite.")
    lines = text.splitlines()
    output: list[str] = []
    atom_index = 0
    in_atom_section = False
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            in_atom_section = line.strip() == "@<TRIPOS>ATOM"
            output.append(line)
            continue
        if in_atom_section and line.strip():
            fields = line.split()
            if len(fields) < 9:
                raise ValueError("Malformed MOL2 atom row.")
            if atom_index >= len(charges):
                raise ValueError("MOL2 contains more atoms than the AM1-BCC vector.")
            fields[8] = f"{charges[atom_index]:.12f}"
            output.append(" ".join(fields))
            atom_index += 1
        else:
            output.append(line)
    if atom_index != len(charges):
        raise ValueError(
            f"MOL2/charge length mismatch: {atom_index} atoms, {len(charges)} charges."
        )
    return "\n".join(output) + "\n"


def require_no_frcmod_nonbonded_overrides(text: str) -> None:
    lines = text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "NONBON"
        )
    except StopIteration as exc:
        raise ValueError("parmchk2 frcmod does not contain a NONBON section.") from exc
    overrides = [line for line in lines[start + 1 :] if line.strip()]
    if overrides:
        raise ValueError(
            "Pinned FreeSolv atom types unexpectedly require frcmod nonbonded "
            "overrides: " + "; ".join(overrides)
        )


def _run(command: list[str], cwd: Path, *, label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-1000:].strip()
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}: {detail}"
        )


def _gbnsr6_input(protocol: dict[str, Any]) -> str:
    polar = protocol["polar"]
    nonpolar = protocol["nonpolar_endpoints"]["gbnsr6_surface_tension"]
    return (
        "Route-1 AM1-BCC CHA-GB/GBNSR6\n"
        "&cntrl\n"
        "  inp=1,\n"
        "/\n"
        "&gb\n"
        f"  epsin={polar['epsin']}, epsout={polar['epsout']}, "
        f"istrng={polar['istrng_molar']},\n"
        f"  dprob={polar['dprob_angstrom']}, space={polar['space_angstrom']}, "
        f"arcres={polar['arcres_angstrom']},\n"
        f"  alpb={polar['alpb']}, chagb={polar['chagb']}, "
        f"radiopt={polar['radiopt']},\n"
        f"  ROH={polar['roh_angstrom']}, tau={polar['tau']}, "
        f"rbornstat={polar['rbornstat']},\n"
        f"  cavity_surften={nonpolar['cavity_surften_kcal_mol_angstrom2']},\n"
        "/\n"
    )


def _pbsa_input(protocol: dict[str, Any]) -> str:
    model = protocol["nonpolar_endpoints"]["pbsa_cavity_dispersion"]
    return (
        "Route-1 AM1-BCC PBSA cavity/dispersion\n"
        "&cntrl\n"
        f"  inp={model['inp']},\n"
        "/\n"
        "&pb\n"
        f"  npbverb=1, istrng={model['istrng_millimolar']}, "
        f"fillratio={model['fillratio']}, saopt={model['saopt']},\n"
        f"  decompopt={model['decompopt']}, use_rmin={model['use_rmin']}, "
        f"sprob={model['sprob_angstrom']}, vprob={model['vprob_angstrom']},\n"
        f"  rhow_effect={model['rhow_effect']}, use_sav={model['use_sav']},\n"
        f"  cavity_surften={model['cavity_surften']}, "
        f"cavity_offset={model['cavity_offset_kcal_mol']},\n"
        "/\n"
    )


def _record_path(work_dir: Path, compound_id: str) -> Path:
    return work_dir / "records" / f"{compound_id}.json"


def _evaluate_case(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_work_dir: Path,
    work_dir: Path,
    executables: dict[str, dict[str, str]],
) -> dict[str, Any]:
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

    source_mol2 = source_work_dir / row["source_mol2_relative_path"]
    if not source_mol2.is_file():
        raise FileNotFoundError(source_mol2)
    observed_mol2_hash = sha256_file(source_mol2)
    if observed_mol2_hash != row["source_mol2_sha256"]:
        raise ValueError(f"Frozen source MOL2 hash mismatch for {compound_id}.")
    charges = [float(value) for value in row["am1bcc_charges_e"]]
    charge_vector_sha256 = sha256_bytes(canonical_json_bytes(charges))

    with tempfile.TemporaryDirectory(prefix="m1-", dir="/tmp") as temporary:
        case_dir = Path(temporary)
        local_mol2 = case_dir / "m.mol2"
        local_mol2.write_text(
            mol2_with_charges(source_mol2.read_text(encoding="utf-8"), charges),
            encoding="utf-8",
        )

        _run(
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
        require_no_frcmod_nonbonded_overrides(
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
        _run(
            [executables["tleap"]["path"], "-f", "tleap.in"],
            case_dir,
            label="tleap",
        )

        (case_dir / "g.in").write_text(_gbnsr6_input(protocol), encoding="utf-8")
        with GBNSR6_LOCK:
            _run(
                [
                    executables["gbnsr6"]["path"],
                    "-O",
                    "-i",
                    "g.in",
                    "-o",
                    "g.out",
                    "-p",
                    "p",
                    "-c",
                    "c",
                ],
                case_dir,
                label="gbnsr6",
            )
        gb = parse_gbnsr6_components((case_dir / "g.out").read_text(encoding="utf-8"))

        (case_dir / "pb.in").write_text(_pbsa_input(protocol), encoding="utf-8")
        _run(
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
        pb = parse_pbsa_components((case_dir / "pb.out").read_text(encoding="utf-8"))
    chagb_surface = gb["chagb_polar"] + gb["surface_tension"]
    chagb_cavity_dispersion = gb["chagb_polar"] + pb["cavity"] + pb["dispersion"]
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "compound_id": compound_id,
        "source_partition": protocol["source_partition"],
        "source_record_sha256": row["source_record_sha256"],
        "source_mol2_sha256": observed_mol2_hash,
        "am1bcc_charge_vector_sha256": charge_vector_sha256,
        "status": "success",
        "components_kcal_mol": {
            "chagb_polar": gb["chagb_polar"],
            "gbnsr6_surface_tension": gb["surface_tension"],
            "pbsa_cavity": pb["cavity"],
            "pbsa_dispersion": pb["dispersion"],
        },
        "predictions_kcal_mol": {
            "am1bcc_chagb_surface_tension": chagb_surface,
            "am1bcc_chagb_cavity_dispersion": chagb_cavity_dispersion,
        },
        "topology_provenance": {
            "force_field": protocol["topology"]["force_field"],
            "atom_types": protocol["topology"]["atom_types"],
            "pb_radii": protocol["topology"]["pb_radii"],
            "bonded_terms_used_in_reported_components": False,
        },
        "provider_sha256": {
            name: details["sha256"] for name, details in sorted(executables.items())
        },
    }
    write_json_atomic(destination, record)
    return {"compound_id": compound_id, "status": "success"}


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_evaluation_protocol(protocol_path)
    manifest = load_label_free_manifest(protocol_path, protocol)
    executables = resolve_executables(protocol, args.amber_bin)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    successes = skips = failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _evaluate_case,
                row,
                protocol=protocol,
                fingerprint=fingerprint,
                source_work_dir=source_work_dir,
                work_dir=work_dir,
                executables=executables,
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
            f"Energy phase failed for {failures} cases "
            f"({successes} completed, {skips} skipped)."
        )
    print(f"Energy phase complete: {successes} completed, {skips} skipped, 0 failed.")


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = protocol["source_evidence"]
    for key, hash_key in (
        ("base_protocol", "base_protocol_sha256"),
        ("development_summary", "development_summary_sha256"),
    ):
        path = _artifact_path(protocol_path, evidence[key])
        if sha256_file(path) != evidence[hash_key]:
            raise ValueError(f"Frozen source artifact hash mismatch: {key}.")
    development = load_json(
        _artifact_path(protocol_path, evidence["development_summary"])
    )
    if development["protocol_fingerprint"] != evidence["base_protocol_fingerprint"]:
        raise ValueError("Development summary base protocol fingerprint mismatch.")
    manifest = load_label_free_manifest(protocol_path, protocol)
    for row in manifest["records"]:
        source_record = source_work_dir / row["source_record_relative_path"]
        if sha256_file(source_record) != row["source_record_sha256"]:
            raise ValueError(f"Source record hash mismatch for {row['compound_id']}.")
        source_mol2 = source_work_dir / row["source_mol2_relative_path"]
        if sha256_file(source_mol2) != row["source_mol2_sha256"]:
            raise ValueError(f"Source MOL2 hash mismatch for {row['compound_id']}.")
    return manifest, development


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_evaluation_protocol(protocol_path)
    source_work_dir = Path(args.source_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    manifest, development = _verify_summary_sources(
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
        if result.get("status") != "success":
            raise ValueError(f"Energy result is not successful: {result_path}")
        if result.get("protocol_fingerprint") != fingerprint:
            raise ValueError(f"Energy result protocol mismatch: {compound_id}")
        if result.get("source_record_sha256") != source["source_record_sha256"]:
            raise ValueError(f"Energy result source mismatch: {compound_id}")
        if result.get("source_mol2_sha256") != source["source_mol2_sha256"]:
            raise ValueError(f"Energy result MOL2 mismatch: {compound_id}")
        charge_vector_sha256 = sha256_bytes(
            canonical_json_bytes(source["am1bcc_charges_e"])
        )
        if result.get("am1bcc_charge_vector_sha256") != charge_vector_sha256:
            raise ValueError(f"Energy result charge-vector mismatch: {compound_id}")
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
        "obc2_ace_to_chagb_surface_tension": paired_absolute_error_gain(
            errors["am1bcc_obc2_ace"],
            errors["am1bcc_chagb_surface_tension"],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]),
        ),
        "obc2_ace_to_chagb_cavity_dispersion": paired_absolute_error_gain(
            errors["am1bcc_obc2_ace"],
            errors["am1bcc_chagb_cavity_dispersion"],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + 1,
        ),
        "chagb_surface_tension_to_cavity_dispersion": paired_absolute_error_gain(
            errors["am1bcc_chagb_surface_tension"],
            errors["am1bcc_chagb_cavity_dispersion"],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + 2,
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

    ranked = sorted(
        rows,
        key=lambda row: abs(
            row["signed_errors_kcal_mol"]["am1bcc_chagb_cavity_dispersion"]
        ),
        reverse=True,
    )
    executables = resolve_executables(protocol, args.amber_bin)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": protocol["source_partition"],
        "claim_scope": protocol["claim_scope"],
        "case_count": len(rows),
        "success_count": len(rows),
        "failure_count": expected - len(rows),
        "label_use_boundary": {
            "energy_phase": "No experimental labels read.",
            "summary_phase": "Experimental FreeSolv values read only after all energy records existed.",
            "experimental_fit_or_residual_model": False,
        },
        "methods": method_metrics,
        "paired_absolute_error_gain": paired,
        "strata": strata,
        "largest_cavity_dispersion_absolute_errors": ranked[:10],
        "record_sha256": record_hashes,
        "provider_sha256": {
            name: details["sha256"] for name, details in sorted(executables.items())
        },
        "source_development_summary_sha256": protocol["source_evidence"][
            "development_summary_sha256"
        ],
        "source_baseline_metrics": development["methods"]["am1bcc/obc2"],
        "interpretation": (
            "Development-only fixed-geometry comparison. CHA-GB changes the polar "
            "continuum endpoint; PBSA cavity/dispersion changes only the nonpolar "
            "endpoint. Neither endpoint was fit to FreeSolv labels in this workflow."
        ),
    }
    write_json_atomic(args.summary, summary)
    print(f"Wrote deterministic summary to {Path(args.summary).resolve()}")
    return summary


def _default_paths() -> dict[str, Path]:
    return {
        "protocol": SCRIPT_DIR / "chagb_nonpolar_protocol.json",
        "source_work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
        ),
        "work_dir": (
            REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724"
        ),
        "summary": SCRIPT_DIR / "freesolv-am1bcc-chagb-nonpolar-2026-07-24.json",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "summarize", "all"))
    parser.add_argument("--protocol", default=str(defaults["protocol"]))
    parser.add_argument("--source-work-dir", default=str(defaults["source_work_dir"]))
    parser.add_argument("--work-dir", default=str(defaults["work_dir"]))
    parser.add_argument("--summary", default=str(defaults["summary"]))
    parser.add_argument("--amber-bin")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.phase in {"run", "all"}:
        run_energy(args)
    if args.phase in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
