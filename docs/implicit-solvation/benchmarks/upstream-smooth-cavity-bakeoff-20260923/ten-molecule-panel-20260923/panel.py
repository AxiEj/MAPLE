#!/usr/bin/env python3
"""Frozen ten-species implementation-parity panel; no fitting or source edits.

This private wrapper reuses the reviewed v3 canary without changing its
scientific scalar or numerical gates. Every species runs in a fresh process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ase import __version__ as ase_version
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "tools/route2_release/run_pure_mace_polar_torch_canary.py"
SPECIES = (
    ("water", "H2O", "bent oxide"),
    ("methane", "CH4", "saturated hydrocarbon"),
    ("ammonia", "NH3", "pyramidal nitrogen hydride"),
    ("carbon-monoxide", "CO", "heteronuclear diatomic"),
    ("carbon-dioxide", "CO2", "linear oxide"),
    ("hydrogen-cyanide", "HCN", "nitrile"),
    ("formaldehyde", "H2CO", "carbonyl"),
    ("hydrogen-peroxide", "H2O2", "peroxide"),
    ("acetylene", "C2H2", "alkyne"),
    ("hydrogen-fluoride", "HF", "hydrogen halide"),
)
SOLVENT = "water"
DEVICE = "cpu"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _runner_module():
    spec = importlib.util.spec_from_file_location("frozen_torch_canary", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the reviewed v3 canary runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare(root: Path, checkpoint: Path) -> dict:
    runner = _runner_module()
    if root.exists():
        raise FileExistsError(f"Panel output already exists: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cases = []
    for name, ase_key, category in SPECIES:
        atoms = molecule(ase_key)
        if len(atoms) > 5:
            raise ValueError(f"Dense v3 atom envelope exceeded: {ase_key}")
        atoms.info.update(charge=0, mult=1)
        cases.append(
            {
                "name": name,
                "ase_key": ase_key,
                "category": category,
                "symbols": atoms.get_chemical_symbols(),
                "positions_angstrom": atoms.positions.tolist(),
                "charge": 0,
                "multiplicity": 1,
                "solvent": SOLVENT,
            }
        )
    root.mkdir(parents=True)
    source_manifest = runner._source_manifest()
    _write(root / "source-manifest.json", source_manifest)
    protocol = {
        "purpose": "implementation parity and local derivative consistency, not experimental accuracy",
        "status": "preregistered-before-ten-runs",
        "cases": cases,
        "case_count": 10,
        "geometry_source": "ASE 3.27.0 ase.build.molecule G2/extra database; exact coordinates frozen above",
        "ase_version": ase_version,
        "checkpoint_sha256": _sha256(checkpoint),
        "wrapper_sha256": _sha256(Path(__file__)),
        "runner_sha256": _sha256(RUNNER),
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "device": DEVICE,
        "solvent": SOLVENT,
        "limits": runner.LIMITS,
        "audit_steps_angstrom": runner.AUDIT_STEPS_ANGSTROM,
        "selection_rule": "fixed list; no replacement after seeing failures",
        "fit_performed": False,
        "physical_accuracy_tested": False,
    }
    _write(root / "protocol.json", protocol)
    return protocol


def _single(case: str, output_dir: Path, checkpoint: Path) -> int:
    runner = _runner_module()
    selected = {name: ase_key for name, ase_key, _ in SPECIES}
    if case not in selected:
        raise ValueError(case)
    runner.CASES[case] = (selected[case], SOLVENT)
    sys.argv = [
        str(RUNNER),
        "--case",
        case,
        "--device",
        DEVICE,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]
    return runner.main()


def _run(root: Path, checkpoint: Path) -> int:
    protocol = _prepare(root, checkpoint)
    started = time.perf_counter()
    exits = {}
    for index, (name, _ase_key, _category) in enumerate(SPECIES, start=1):
        log = root / f"{index:02d}-{name}.log"
        print(f"[{index}/10] {name}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            complete = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--single",
                    name,
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(root / name),
                ],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exits[name] = complete.returncode
        _write(root / "exit-codes.json", exits)
        print(f"[{index}/10] {name}: exit {complete.returncode}", flush=True)
    rows = []
    for name, _ase_key, _category in SPECIES:
        path = root / name / "result.json"
        if path.is_file():
            value = json.loads(path.read_text())
            rows.append(
                {
                    "name": name,
                    "exit_code": exits[name],
                    "pass": value.get("pass", False),
                    "energy_error_eV": value.get("parity_errors", {}).get("energy_eV"),
                    "force_error_eV_per_A": value.get("parity_errors", {}).get(
                        "force_eV_per_A"
                    ),
                    "hvp_vs_dense_error_eV_per_A2": value.get("parity_errors", {}).get(
                        "hvp_vs_dense_eV_per_A2"
                    ),
                    "independent_hvp_errors_eV_per_A2": [
                        audit.get("maximum_error_eV_per_A2")
                        for audit in value.get(
                            "independent_legacy_gradient_fd_audit", []
                        )
                    ],
                    "raw_hessian_antisymmetry_eV_per_A2": value.get("torch", {}).get(
                        "raw_antisymmetry_eV_per_A2"
                    ),
                    "error": value.get("error"),
                    "source_unchanged": value.get("source_unchanged"),
                    "result_sha256": _sha256(path),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "exit_code": exits[name],
                    "pass": False,
                    "error": "result.json missing",
                }
            )
    current_source = _runner_module()._source_manifest()
    summary = {
        "protocol_sha256": _sha256(root / "protocol.json"),
        "case_count": len(rows),
        "passed_count": sum(row["pass"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "source_manifest_unchanged": current_source
        == json.loads((root / "source-manifest.json").read_text()),
        "wrapper_sha256_unchanged": _sha256(Path(__file__))
        == protocol["wrapper_sha256"],
        "rows": rows,
        "physical_accuracy_tested": False,
        "fit_performed": False,
    }
    _write(root / "summary.json", summary)
    return (
        0
        if summary["passed_count"] == 10 and summary["source_manifest_unchanged"]
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--single", choices=[name for name, _, _ in SPECIES])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.single:
        if args.output_dir is None:
            parser.error("--single requires --output-dir")
        return _single(args.single, args.output_dir, args.checkpoint)
    if args.output_root is None:
        parser.error("--output-root is required")
    return _run(args.output_root, args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
