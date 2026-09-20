#!/usr/bin/env python3
"""Read-only cross-case qualification of a completed foundation panel.

Creates a NEW postflight record. Never modifies or reseals native run records.
In particular, a precision diagnostic cannot replace a failed CPU panel case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_core import sha256_file
from run_route1_foundation_panel import (
    PANEL,
    PROTOCOL,
    ROOT,
    array_hash,
    hessian_quality,
    save,
    summarize,
)


def cross_case_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(records)
    errors: list[str] = []
    if not summary["complete"]:
        errors.append("missing, duplicate or unexpected endpoint/molecule cases")
    checkpoints = set()
    by_key = {}
    for record in records:
        key = (record["endpoint"], record["molecule"])
        by_key[key] = record
        precision = record.get("inference_precision", {})
        checkpoint = precision.get("original_checkpoint_sha256", "")
        if (
            len(checkpoint) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint)
            or precision.get("effective_dtype") != "float64"
            or precision.get("requested_dtype") != "float64"
            or precision.get("numerical_curvature_prepared") is not True
        ):
            errors.append(f"{key}: missing checkpoint identity or float64 precision")
        checkpoints.add(checkpoint)
        before, after = (
            record.get("identity_before", {}),
            record.get("identity_after", {}),
        )
        for field, vector in (
            ("charges_sha256", "charges_e"),
            ("radii_sha256", "radii_angstrom"),
        ):
            if not before.get(field) or before.get(field) != after.get(field):
                errors.append(f"{key}: fixed {field} changed/missing")
            for values in (before, after):
                if vector not in values or array_hash(values[vector]) != values.get(
                    field
                ):
                    errors.append(f"{key}: {field} does not bind its vector")
        provider = record.get("provider_provenance", {})
        if key[0] == "obc2":
            expected = {
                "provider": "openmm",
                "model": "obc2",
                "nonpolar": "ace",
                "profile": "obc2-mbondi2",
                "platform": "CPU",
            }
        else:
            expected = {
                "provider": "ddx",
                "model": "lpb",
                "nonpolar": "none",
                "profile": "ddlpb-union-mbondi2-v1",
                "polar_only": True,
                "solvent_kappa_inverse_angstrom": 0.1,
                "reference_only": True,
                "fixed_charge": True,
                "fixed_radius": True,
                "absolute_solvation_free_energy_claim": False,
                "radii": "mbondi2",
                "required_provider_version": "0.8.0",
            }
            settings = {
                "lmax": 9,
                "n_lebedev": 302,
                "eta": 0.1,
                "shift": 0.0,
                "solver_tolerance": 1e-10,
                "n_proc": 1,
            }
            if any(
                provider.get("settings", {}).get(k) != value
                for k, value in settings.items()
            ):
                errors.append(f"{key}: wrong numerical LPB resolution/settings")
        if any(provider.get(k) != value for k, value in expected.items()):
            errors.append(f"{key}: wrong endpoint scalar/configuration")
    if len(checkpoints) != 1:
        errors.append("ANI checkpoint differs across cases")
    for name in PANEL:
        pair = [by_key.get((endpoint, name), {}) for endpoint in ("obc2", "ddlpb")]
        if not pair[0].get("input_sha256") or pair[0].get("input_sha256") != pair[
            1
        ].get("input_sha256"):
            errors.append(f"{name}: paired input differs/missing")
        for field in ("charges_sha256", "radii_sha256"):
            values = [r.get("identity_before", {}).get(field) for r in pair]
            if values[0] is None or values[0] != values[1]:
                errors.append(f"{name}: paired {field} differs/missing")
    return {
        "passed": not errors,
        "errors": errors,
        "checkpoint_sha256": sorted(checkpoints),
    }


def bind_curvature(folder: Path, label: str) -> dict[str, Any]:
    raw, bindings = [], []
    for step in PROTOCOL["hessian_steps_angstrom"]:
        path = folder / f"{label}-raw-hessian-{step:g}.npy"
        matrix = np.load(path, allow_pickle=False)
        raw.append(matrix)
        norm = float(np.linalg.norm(matrix))
        bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "relative_raw_asymmetry_frobenius": float(
                    np.linalg.norm(matrix - matrix.T) / norm
                )
                if norm
                else 0.0,
            }
        )
    scale = max(float(np.linalg.norm(h)) for h in raw)
    return {
        "arrays": bindings,
        "quality": hessian_quality(raw),
        "relative_raw_refinement_frobenius": float(
            np.linalg.norm(raw[0] - raw[1]) / scale
        )
        if scale
        else 0.0,
    }


def check_evidence(
    run_dir: Path, inputs_dir: Path, lineage: Path, execution_environment: Path
) -> dict[str, Any]:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    original = json.loads((run_dir / "summary.json").read_text())
    records = original["cases"]
    identity = cross_case_identity(records)
    errors = list(identity["errors"])
    if original.get("source_unchanged_during_run") is not True:
        errors.append("native run did not pass its source-freeze check")
    checkpoint_path = ROOT / "maple/function/calculator/model/ani2x.pt"
    if identity["checkpoint_sha256"] != [sha256_file(checkpoint_path)]:
        errors.append("checkpoint does not match the live expected ANI2x file")
    execution = json.loads(execution_environment.read_text())
    if len(execution) != 1 or any(
        execution[0]["environment"].get(key) != "1"
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OPENMM_CPU_THREADS",
        )
    ):
        errors.append("single-thread execution was not evidenced")
    elif str(run_dir) not in execution[0]["command"]:
        errors.append("execution environment belongs to a different run")
    if len(execution) == 1 and (
        execution[0].get("cwd") != str(ROOT)
        or execution[0]["environment"].get("PYTHONPATH") != str(ROOT)
    ):
        errors.append("execution cwd/PYTHONPATH belongs to a different checkout")
    manifest = inputs_dir / "manifest.json"
    prepared = json.loads(manifest.read_text())["records"]
    lineage_record = json.loads(lineage.read_text())
    if (
        sha256_file(manifest) != protocol["input_manifest_sha256"]
        or sha256_file(manifest) != lineage_record["prepared_manifest_sha256"]
    ):
        errors.append("prepared manifest differs from run/lineage")
    if len(prepared) != len(PANEL) or {r["name"] for r in prepared} != set(PANEL):
        errors.append("input manifest does not contain exactly the six locked records")
    expected = {r["name"]: r for r in prepared}
    reconstructed = {r["name"]: r for r in lineage_record["records"]}
    if len(lineage_record["records"]) != len(PANEL) or set(reconstructed) != set(PANEL):
        errors.append("reconstruction must contain six unique locked records")
    generator_names = {
        "prepare_inputs.py",
        "prepare_rounded_inputs.py",
        "reconstruct_input_lineage.py",
    }
    if set(lineage_record["generator_sha256"]) != generator_names or any(
        sha256_file(lineage.parent / name) != digest
        for name, digest in lineage_record["generator_sha256"].items()
    ):
        errors.append("input generator source hash mismatch")
    lineage_controls = {
        "seed": 20260913,
        "rdkit_version": "2024.09.2",
        "embedding": "ETKDGv3",
        "embedding_return_code_reconstructed": 0,
        "uff_max_iterations": 200,
        "uff_return_code_reconstructed": 0,
        "quantization_angstrom": 0.001,
        "original_reference_byte_exact_reconstruction": True,
        "rounded_reference_byte_exact_reconstruction": True,
    }
    for name, reconstruction in reconstructed.items():
        if any(reconstruction.get(k) != value for k, value in lineage_controls.items()):
            errors.append(f"{name}: wrong input reconstruction controls/outcome")
        for path, field in (
            (
                lineage.parent / "inputs" / name / "reference.mol2",
                "original_reference_sha256",
            ),
            (
                lineage.parent / "inputs" / name / "preparation.json",
                "failed_original_preparation_sha256",
            ),
            (inputs_dir / name / "reference.mol2", "rounded_reference_sha256"),
            (inputs_dir / name / "preparation.json", "preparation_record_sha256"),
        ):
            if sha256_file(path) != reconstruction.get(field):
                errors.append(
                    f"{name}: original/preparation lineage file changed ({field})"
                )
    for record in records:
        name = record["molecule"]
        if (
            name not in expected
            or name not in reconstructed
            or expected[name]["status"] != "prepared"
            or sha256_file(inputs_dir / name / "fixed.mol2")
            != expected[name]["fixed_sha256"]
            or record.get("input_sha256") != expected[name]["fixed_sha256"]
            or reconstructed[name]["fixed_mol2_sha256"]
            != expected[name]["fixed_sha256"]
        ):
            errors.append(f"{name}: fixed-input lineage mismatch")
    for path, digest in protocol["source_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            errors.append(f"source changed: {path}")
    arrays = {}
    native_artifacts = {}
    for record in records:
        key = f"{record['endpoint']}-{record['molecule']}"
        folder = run_dir / key
        try:
            if json.loads((folder / "result.json").read_text()) != record:
                errors.append(f"{key}: case result differs from native summary")
            native_artifacts[key] = {
                str(p): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()
            }
            arrays[key] = {"panel": bind_curvature(folder, "panel")}
            if record["molecule"] in PROTOCOL["minima"]:
                arrays[key]["minimum"] = bind_curvature(folder, "minimum")
            for label, values in arrays[key].items():
                expected_quality = (
                    record["hessian_quality"]
                    if label == "panel"
                    else record["minimum"]["hessian_quality"]
                )
                if values["quality"] != expected_quality:
                    errors.append(
                        f"{key}/{label}: raw arrays do not reproduce recorded quality"
                    )
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{key}: incomplete/corrupt raw curvature: {exc}")
    return {
        "protocol_id": "route1-foundation-evidence-postflight-v1",
        "native_summary_sha256": sha256_file(run_dir / "summary.json"),
        "native_protocol_sha256": sha256_file(run_dir / "protocol.json"),
        "postflight_script_sha256": sha256_file(__file__),
        "benchmark_core_sha256_at_postflight": sha256_file(
            Path(__file__).with_name("benchmark_core.py")
        ),
        "lineage_sha256": sha256_file(lineage),
        "execution_environment_sha256": sha256_file(execution_environment),
        "cross_case_identity": identity,
        "curvature": arrays,
        "native_case_artifacts_sha256": native_artifacts,
        "errors": errors,
        "evidence_integrity_passed": not errors,
        "all_numerical_cases_accepted": bool(original["all_accepted"] and not errors),
        "native_records_unchanged": True,
        "numerical_summary": {k: v for k, v in original.items() if k != "cases"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--execution-environment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record = check_evidence(
        args.run_dir, args.inputs_dir, args.lineage, args.execution_environment
    )
    save(args.output, record)
    return 0 if record["evidence_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
