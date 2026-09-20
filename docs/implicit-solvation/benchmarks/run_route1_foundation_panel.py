#!/usr/bin/env python3
"""Small, fixed-input Route 1 E/F/H panel and OPT->FREQ acceptance anchors.

This is numerical/workflow qualification, not solvation-accuracy admission.
Every requested case survives in the denominator, including native failures.
Output directories must be new; source, controls and raw arrays are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_core import command_provenance, sha256_file, write_json_atomic
from stationary_point_checks import analyze_stationary_point

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.optimization.optimization import Optimization
from maple.function.read.filereader.mol2_reader import MOL2Reader

PANEL = ("water", "ammonia", "methanol", "ethane", "fluoromethane", "benzene")
MINIMA = ("water", "methanol")
ENDPOINTS = {
    "obc2": {
        "implicit": "gb",
        "method": "gb",
        "provider": "openmm",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": "ace",
        "platform": "CPU",
        "experimental": True,
    },
    "ddlpb": {
        "implicit": "pb",
        "method": "pb",
        "provider": "ddx",
        "model": "lpb",
        "profile": "ddlpb-union-mbondi2-v1",
        "nonpolar": "none",
        "solvent_kappa_inverse_angstrom": 0.1,
        "experimental": True,
    },
}
PROTOCOL = {
    "id": "route1-foundation-panel-20260913-v1",
    "panel": PANEL,
    "minima": MINIMA,
    "endpoints": ENDPOINTS,
    "force_fd_steps_angstrom": (2e-4, 1e-4),
    "hessian_steps_angstrom": (5e-4, 2.5e-4),
    "force_max_error_hartree_per_angstrom": 2e-5,
    "force_rms_error_hartree_per_angstrom": 5e-6,
    "replay_energy_force_tolerance": 1e-10,
    "hessian_absolute_budget": 5e-4,
    "hessian_relative_budget": 1e-3,
    "significant_negative_cutoff_cm1": 30.0,
    "optimizer": {
        "method": "lbfgs",
        "curvature": 5.0,
        "max_step": 0.05,
        "max_iter": 100,
        "memory": 5,
        "verbose": 1,
    },
    "solver_stops": {
        "f_max_th": 1e-4,
        "f_rms_th": 7.5e-5,
        "dp_max_th": 3e-4,
        "dp_rms_th": 2e-4,
    },
    "stationary_force_max": 2.5e-4,
    "stationary_force_rms": 1.5e-4,
    "stationary_step_max": 1e-3,
    "stationary_step_rms": 6e-4,
    "model_options": {"hessian": "numerical", "dtype": "float64"},
    "charge_options": {
        "source": "mol2",
        "mode": "fixed",
        "geometry": "keep",
        "label": "am1bcc-frozen-route1-foundation-20260913",
    },
    "claim_scope": "water-only numerical foundation; no accuracy or chemistry generalization",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save(path: Path, value: Any) -> None:
    write_json_atomic(path, json_safe(value))


def array_hash(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def identity(provider: Any) -> dict[str, Any]:
    charges = np.asarray(provider.charges)
    radii = np.asarray(provider.radius_result.radii_angstrom)
    return {
        "charges_e": charges,
        "radii_angstrom": radii,
        "charges_sha256": array_hash(charges),
        "radii_sha256": array_hash(radii),
    }


def evaluate(atoms: Atoms) -> tuple[float, np.ndarray]:
    # A force solve also returns E; the reverse ordering duplicates native solves.
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).copy()
    energy = float(atoms.get_potential_energy(force_consistent=True))
    if (
        forces.shape != (len(atoms), 3)
        or not np.isfinite(forces).all()
        or not np.isfinite(energy)
    ):
        raise ValueError("Incomplete or nonfinite native E/F")
    return energy, forces


def differentiate(atoms: Atoms, step: float, *, hessian: bool) -> np.ndarray:
    """Full central differences; always restore the reference on failure."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite positive difference step required")
    positions = atoms.get_positions().copy()
    dimension = positions.size
    result = np.empty((dimension, dimension) if hessian else (dimension,))
    try:
        for j in range(dimension):
            values = []
            for sign in (1, -1):
                displaced = positions.copy().reshape(-1)
                displaced[j] += sign * step
                atoms.set_positions(displaced.reshape((-1, 3)))
                values.append(
                    np.asarray(atoms.get_forces()).reshape(-1).copy()
                    if hessian
                    else float(atoms.get_potential_energy(force_consistent=True))
                )
            difference = -(values[0] - values[1]) / (2 * step)
            if hessian:
                result[:, j] = difference
            else:
                result[j] = difference
    finally:
        atoms.set_positions(positions)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite raw finite difference")
    return result if hessian else result.reshape((-1, 3))


def hessian_quality(raw: list[np.ndarray]) -> dict[str, Any]:
    if len(raw) != 2 or any(h.ndim != 2 or h.shape[0] != h.shape[1] for h in raw):
        raise ValueError("two square raw Hessians required")
    if raw[0].shape != raw[1].shape or not all(np.isfinite(h).all() for h in raw):
        raise ValueError("matching finite raw Hessians required")
    scales = [float(np.max(np.abs(h))) for h in raw]
    budgets = [5e-4 + 1e-3 * scale for scale in scales]
    asymmetry = [float(np.max(np.abs(h - h.T))) for h in raw]
    raw_change = float(np.max(np.abs(raw[0] - raw[1])))
    sym_change = float(np.max(np.abs((raw[0] + raw[0].T - raw[1] - raw[1].T) / 2)))
    return {
        "raw_maximum_asymmetry": asymmetry,
        "raw_maximum_step_change": raw_change,
        "symmetrized_maximum_step_change_diagnostic_only": sym_change,
        "budgets": budgets,
        "passed": bool(
            all(a <= b for a, b in zip(asymmetry, budgets))
            and raw_change <= max(budgets)
        ),
    }


def curvature(
    atoms: Atoms, folder: Path, label: str
) -> tuple[list[np.ndarray], dict[str, Any]]:
    raw = []
    for step in PROTOCOL["hessian_steps_angstrom"]:
        matrix = differentiate(atoms, step, hessian=True)
        raw.append(matrix)
        np.save(folder / f"{label}-raw-hessian-{step:g}.npy", matrix)
    return raw, hessian_quality(raw)


def minimum_anchor(atoms: Any, folder: Path) -> dict[str, Any]:
    for key, value in PROTOCOL["solver_stops"].items():
        setattr(atoms, key, value)
    output = folder / "opt.out"
    optimized = Optimization(dict(PROTOCOL["optimizer"]), str(output), atoms).run()
    energy, forces = evaluate(optimized)
    text = output.read_text()
    iterations = re.findall(r"Iteration:\s*(\d+)", text)
    status = bool(re.search(r"LBFGS converged at iteration \d+", text))
    max_step = float(getattr(optimized, "max_dp", np.inf))
    rms_step = float(getattr(optimized, "rms_dp", np.inf))
    # Save the optimizer outcome before expensive postflight curvature.
    record: dict[str, Any] = {
        "optimizer_converged": status,
        "iterations": int(iterations[-1]) if iterations else None,
        "maximum_last_step_angstrom": max_step,
        "rms_last_step_angstrom": rms_step,
        "energy_hartree": energy,
        "forces_hartree_per_angstrom": forces,
        "positions_angstrom": optimized.get_positions(),
        "accepted": False,
    }
    save(folder / "minimum.json", record)
    raw, quality = curvature(optimized, folder, "minimum")
    analyses = [analyze_stationary_point(optimized, h, forces) for h in raw]
    record.update(hessian_quality=quality, stationary_analyses=analyses)
    record["accepted"] = bool(
        status
        and quality["passed"]
        and all(a["is_minimum"] for a in analyses)
        and max_step <= PROTOCOL["stationary_step_max"]
        and rms_step <= PROTOCOL["stationary_step_rms"]
    )
    save(folder / "minimum.json", record)
    return record


def run_case(source: Path, endpoint: str, folder: Path) -> dict[str, Any]:
    import torch

    folder.mkdir()
    started = time.monotonic()
    record: dict[str, Any] = {
        "molecule": source.parent.name,
        "endpoint": endpoint,
        "source": str(source),
        "accepted": False,
        "status": "running",
    }
    try:
        record["input_sha256"] = sha256_file(source)
        atoms = MOL2Reader(str(source), charge=0, mult=1)
        settings = dict(ENDPOINTS[endpoint])
        method = settings.pop("implicit")
        settings["implicit"] = "water"
        calculator = SetCalculator(
            torch.device("cpu"),
            "ani2x",
            str(folder / "calculator.out"),
            atoms=atoms,
            implicit=method,
            solvent="water",
            model_options=dict(PROTOCOL["model_options"]),
            solvation_options=settings,
            charge_options=dict(PROTOCOL["charge_options"]),
        ).set_calculator()
        atoms.calc = calculator
        provider = calculator.solvent_correction.provider
        before = identity(provider)
        record.update(
            identity_before=before,
            provider_provenance=provider.provenance,
            inference_precision=calculator.inference_precision_provenance,
        )
        energy, forces = evaluate(atoms)
        repeat_energy, repeat_forces = evaluate(atoms)
        positions = atoms.get_positions().copy()
        displaced = positions.copy()
        displaced[0, 0] += 0.013
        atoms.set_positions(displaced)
        evaluate(atoms)
        atoms.set_positions(positions)
        replay_energy, replay_forces = evaluate(atoms)
        replay_errors = [
            abs(energy - repeat_energy),
            abs(energy - replay_energy),
            float(np.max(np.abs(forces - repeat_forces))),
            float(np.max(np.abs(forces - replay_forces))),
        ]
        record.update(
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces,
            positions_angstrom=positions,
            replay_errors=replay_errors,
            replay_passed=bool(max(replay_errors) <= 1e-10),
        )
        save(folder / "result.json", record)
        force_checks = []
        for step in PROTOCOL["force_fd_steps_angstrom"]:
            numerical = differentiate(atoms, step, hessian=False)
            error = numerical - forces
            maximum, rms = (
                float(np.max(np.abs(error))),
                float(np.sqrt(np.mean(error**2))),
            )
            force_checks.append(
                {
                    "step_angstrom": step,
                    "numerical_force": numerical,
                    "error": error,
                    "maximum_error": maximum,
                    "rms_error": rms,
                    "passed": maximum <= 2e-5 and rms <= 5e-6,
                }
            )
        record["force_checks"] = force_checks
        save(folder / "result.json", record)
        _, quality = curvature(atoms, folder, "panel")
        record["hessian_quality"] = quality
        save(folder / "result.json", record)
        if source.parent.name in MINIMA:
            record["minimum"] = minimum_anchor(atoms, folder)
        after = identity(provider)
        record.update(
            identity_after=after,
            identity_unchanged=all(
                before[key] == after[key] for key in ("charges_sha256", "radii_sha256")
            ),
        )
        record["accepted"] = bool(
            record["replay_passed"]
            and quality["passed"]
            and all(check["passed"] for check in force_checks)
            and record["identity_unchanged"]
            and record.get("minimum", {"accepted": True})["accepted"]
            and record["inference_precision"].get("effective_dtype") == "float64"
        )
        record["status"] = "accepted" if record["accepted"] else "failed_checks"
    except Exception as exc:  # noqa: BLE001 -- preserve native failures in the full denominator
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    record["wall_seconds"] = time.monotonic() - started
    save(folder / "result.json", record)
    print(endpoint, source.parent.name, record["status"], flush=True)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(e, m) for e in ENDPOINTS for m in PANEL}
    actual = [(r["endpoint"], r["molecule"]) for r in records]
    complete = len(actual) == len(expected) and set(actual) == expected
    return {
        "expected_count": len(expected),
        "executed_count": len(records),
        "accepted_count": sum(r["accepted"] for r in records),
        "missing": sorted(expected - set(actual)),
        "complete": complete,
        "all_accepted": bool(complete and all(r["accepted"] for r in records)),
        "selected_cases_all_accepted": bool(
            records and all(r["accepted"] for r in records)
        ),
        "cases": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", choices=tuple(ENDPOINTS), action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("stationary_point_checks.py")]
    # TS runs have a separate frozen manifest; this runner never invokes them.
    sources.extend(
        p
        for p in sorted((ROOT / "maple").rglob("*.py"))
        if "dispatcher/ts/" not in p.as_posix()
    )
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    manifest = args.inputs_dir / "manifest.json"
    prepared = json.loads(manifest.read_text())["records"]
    if len(prepared) != len(PANEL) or {r["name"] for r in prepared} != set(PANEL):
        raise ValueError("input manifest must contain exactly the six locked molecules")
    for record in prepared:
        if (
            record["status"] != "prepared"
            or sha256_file(args.inputs_dir / record["name"] / "fixed.mol2")
            != record["fixed_sha256"]
        ):
            raise ValueError(f"input preparation/hash mismatch: {record['name']}")
    provenance = {
        "protocol": PROTOCOL,
        "python": platform.python_version(),
        "command": command_provenance(
            __file__,
            vars(args),
            repository_root=ROOT,
            environment_variables=(
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONPATH",
            ),
        ),
        "input_manifest_sha256": sha256_file(manifest),
        "source_sha256": source_hashes,
        "source_scope": "runner/helper and MAPLE Python excluding unexercised TS dispatcher",
    }
    save(args.output_dir / "protocol.json", provenance)
    records = []
    for endpoint in args.endpoint or ENDPOINTS:
        for name in PANEL:
            records.append(
                run_case(
                    args.inputs_dir / name / "fixed.mol2",
                    endpoint,
                    args.output_dir / f"{endpoint}-{name}",
                )
            )
            save(args.output_dir / "summary.json", summarize(records))
    unchanged = all(
        sha256_file(ROOT / p) == digest for p, digest in source_hashes.items()
    )
    summary = summarize(records)
    summary["source_unchanged_during_run"] = unchanged
    summary["all_accepted"] = summary["all_accepted"] and unchanged
    summary["selected_cases_all_accepted"] = (
        summary["selected_cases_all_accepted"] and unchanged
    )
    save(args.output_dir / "summary.json", summary)
    # A subset process can finish normally; canonical full acceptance stays false.
    return 0 if summary["selected_cases_all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
