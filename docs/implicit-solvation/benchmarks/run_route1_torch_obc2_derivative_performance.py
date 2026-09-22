#!/usr/bin/env python3
"""Label-free timing of numerical and analytic Route 1 curvature paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core  # noqa: E402
from maple.function.calculator.set_calculator import SetClaculator  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

CASES = {
    "methanol": (
        REPOSITORY_ROOT
        / "tests/solvation/data/amber_gb_reference/audit/methanol/normalized.mol2"
    ),
    "methyl-hexanoate": (
        REPOSITORY_ROOT / "tests/solvation/data/amber_gb_reference/audit/"
        "methyl-hexanoate/normalized.mol2"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(case: str, mol2: Path, mode: str, work_dir: Path):
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    platform = "Reference" if mode == "analytic" else "CPU"
    factory = SetClaculator(
        "cpu",
        "ani2x",
        str(work_dir / f"{case}-{mode}.out"),
        atoms=atoms,
        implicit="gb",
        solvent="water",
        model_options={"hessian": mode, "dtype": "float64"},
        solvation_options={
            "implicit": "water",
            "method": "gb",
            "provider": "openmm",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
            "platform": platform,
            "experimental": True,
        },
        charge_options={
            "source": "mol2",
            "mode": "fixed",
            "geometry": "keep",
        },
        task_context={"task": "freq", "method": "mw"},
    )
    calculator = factory.set_calculator()
    atoms.calc = calculator
    return atoms, calculator


def _timed(function, samples: int) -> dict[str, object]:
    function()  # unmeasured graph/context warm-up
    values = []
    for _ in range(samples):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    return {
        "samples_seconds": values,
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "result_shape": list(np.asarray(result).shape),
    }


def run(output: Path, work_dir: Path, samples: int) -> dict[str, object]:
    if samples < 1:
        raise ValueError("samples must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for case, mol2 in CASES.items():
        analytic_atoms, analytic = _build(case, mol2, "analytic", work_dir)
        numerical_atoms, numerical = _build(case, mol2, "numerical", work_dir)
        analytic_timing = _timed(lambda: analytic.get_hessian(analytic_atoms), samples)
        numerical_timing = _timed(
            lambda: numerical.get_hessian(numerical_atoms), samples
        )

        generator = np.random.default_rng(20260922)
        direction = generator.normal(size=3 * len(analytic_atoms))
        direction /= np.linalg.norm(direction)
        explicit_hvp = _timed(
            lambda: analytic.get_hessian(analytic_atoms) @ direction, samples
        )
        direct_hvp = _timed(
            lambda: analytic.get_hvp(analytic_atoms, direction)[0], samples
        )
        records[case] = {
            "atom_count": len(analytic_atoms),
            "mol2": str(mol2.relative_to(REPOSITORY_ROOT)),
            "mol2_sha256": _sha256(mol2),
            "analytic_hessian": analytic_timing,
            "complete_force_numerical_hessian": numerical_timing,
            "analytic_over_numerical_speedup": (
                numerical_timing["median_seconds"] / analytic_timing["median_seconds"]
            ),
            "explicit_hessian_then_multiply": explicit_hvp,
            "direct_hvp": direct_hvp,
            "direct_hvp_speedup": (
                explicit_hvp["median_seconds"] / direct_hvp["median_seconds"]
            ),
        }

    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-torch-obc2-derivative-performance",
        "recorded_date": "2026-09-22",
        "label_reads": False,
        "samples_after_warmup": samples,
        "gas_backend": "ani2x-float64-cpu",
        "analytic_solvent_backend": "torch-obc2-float64-cpu",
        "analytic_runtime_oracle": "openmm-reference",
        "numerical_runtime_provider": "openmm-cpu",
        "records": records,
        "interpretation": {
            "universal_performance_claim": False,
            "hydration_accuracy_claim": False,
            "openmm_runtime_default_changed": False,
        },
        "command_provenance": {
            "script": str(SCRIPT_PATH.relative_to(REPOSITORY_ROOT)),
            "script_sha256": _sha256(SCRIPT_PATH),
        },
        "content_sha256": None,
    }
    artifact["content_sha256"] = core.artifact_content_sha256(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    artifact = run(args.output.resolve(), args.work_dir.resolve(), args.samples)
    print(json.dumps(artifact["records"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
