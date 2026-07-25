#!/usr/bin/env python3
"""Audit the deterministic CPU product path against OpenMM Reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    command_provenance,
    load_json,
    seal_artifact,
    sha256_file,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    DEFAULT_CPU_PLATFORM_PROPERTIES,
    DEFAULT_OPENMM_PLATFORM,
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184
ENERGY_TOLERANCE_KCAL_MOL = 2.0e-5
FORCE_TOLERANCE_KCAL_MOL_ANGSTROM = 2.0e-5


def _case_mol2(manifest_path: Path, case: dict[str, Any]) -> Path:
    path = Path(case["mol2"])
    path = path if path.is_absolute() else manifest_path.parent / path
    path = path.resolve()
    observed = sha256_file(path)
    expected = str(case.get("mol2_sha256", ""))
    if observed != expected:
        raise ValueError(
            f"MOL2 hash mismatch for {case.get('case_id')}: "
            f"expected {expected}, observed {observed}."
        )
    return path


def _expected_model_support(case: dict[str, Any], model: str) -> bool:
    expectation = case.get("model_expectations", {}).get(model)
    return True if expectation is None else bool(expectation["openmm_supported"])


def _expected_lcpo_support(case: dict[str, Any]) -> bool:
    expectation = case.get("lcpo_expectation")
    return True if expectation is None else bool(expectation["openmm_supported"])


def _failure(exc: Exception) -> dict[str, str]:
    return {"exception_class": type(exc).__name__, "reason": str(exc)}


def _result_difference(left, right) -> dict[str, float]:
    component_names = sorted(
        set(left.components_hartree) | set(right.components_hartree)
    )
    component_difference = max(
        (
            abs(
                float(left.components_hartree.get(name, 0.0))
                - float(right.components_hartree.get(name, 0.0))
            )
            * KCAL_PER_HARTREE
            for name in component_names
        ),
        default=0.0,
    )
    force_difference = float(
        np.max(
            np.abs(
                np.asarray(left.forces_hartree_per_angstrom)
                - np.asarray(right.forces_hartree_per_angstrom)
            )
        )
        * KCAL_PER_HARTREE
    )
    return {
        "energy_abs_kcal_mol": (
            abs(float(left.energy_hartree) - float(right.energy_hartree))
            * KCAL_PER_HARTREE
        ),
        "component_max_abs_kcal_mol": component_difference,
        "force_max_abs_kcal_mol_angstrom": force_difference,
    }


def _evaluate_triplet(atoms, charges, *, model: str, nonpolar: str):
    reference = OpenMMGB(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform="Reference",
    )
    cpu_first = OpenMMGB(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform=DEFAULT_OPENMM_PLATFORM,
    )
    cpu_second = OpenMMGB(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform=DEFAULT_OPENMM_PLATFORM,
    )
    return (
        reference,
        cpu_first,
        cpu_second,
        reference.evaluate(atoms, need_forces=True),
        cpu_first.evaluate(atoms, need_forces=True),
        cpu_second.evaluate(atoms, need_forces=True),
    )


def _evaluate_platform(
    atoms,
    charges,
    *,
    model: str,
    nonpolar: str,
    platform_name: str,
):
    try:
        provider = OpenMMGB(
            atoms,
            charges,
            model=model,
            nonpolar=nonpolar,
            platform=platform_name,
        )
        result = provider.evaluate(atoms, need_forces=True)
    except Exception as exc:
        return None, None, _failure(exc)
    return provider, result, None


def _audit_component(
    atoms,
    charges,
    *,
    model: str,
    nonpolar: str,
    expected_support: bool,
) -> tuple[dict[str, Any], tuple[OpenMMGB, OpenMMGB] | None]:
    reference, reference_result, reference_failure = _evaluate_platform(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform_name="Reference",
    )
    cpu_first, cpu_first_result, cpu_first_failure = _evaluate_platform(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform_name=DEFAULT_OPENMM_PLATFORM,
    )
    cpu_second, cpu_second_result, cpu_second_failure = _evaluate_platform(
        atoms,
        charges,
        model=model,
        nonpolar=nonpolar,
        platform_name=DEFAULT_OPENMM_PLATFORM,
    )
    observed_support = {
        "Reference": reference is not None,
        "CPU_first": cpu_first is not None,
        "CPU_second": cpu_second is not None,
    }
    support_matches = all(
        observed is expected_support for observed in observed_support.values()
    )
    if not support_matches:
        record = {
            "status": "failure",
            "expected_supported": expected_support,
            "observed_support": observed_support,
            "failures": {
                name: failure
                for name, failure in (
                    ("Reference", reference_failure),
                    ("CPU_first", cpu_first_failure),
                    ("CPU_second", cpu_second_failure),
                )
                if failure is not None
            },
        }
        return record, None

    if not expected_support:
        return {
            "status": "expected-unavailable",
            "expected_supported": False,
            "observed_support": observed_support,
            "failures": {
                "Reference": reference_failure,
                "CPU_first": cpu_first_failure,
                "CPU_second": cpu_second_failure,
            },
        }, None

    assert reference is not None and reference_result is not None
    assert cpu_first is not None and cpu_first_result is not None
    assert cpu_second is not None and cpu_second_result is not None
    cpu_vs_reference = _result_difference(cpu_first_result, reference_result)
    cpu_repeat = _result_difference(cpu_first_result, cpu_second_result)
    record = {
        "status": "success",
        "expected_supported": True,
        "observed_support": observed_support,
        "cpu_vs_reference": cpu_vs_reference,
        "cpu_independent_context_repeat": cpu_repeat,
        "cpu_provenance": {
            "platform": cpu_first.provenance["platform"],
            "platform_properties": cpu_first.provenance["platform_properties"],
        },
        "reference_provenance": {
            "platform": reference.provenance["platform"],
            "platform_properties": reference.provenance["platform_properties"],
        },
    }
    return record, (reference, cpu_first)


def _time_evaluations(
    reference: OpenMMGB,
    cpu: OpenMMGB,
    atoms,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    timings = {"Reference": [], "CPU": []}
    providers = {"Reference": reference, "CPU": cpu}
    for index in range(warmups + samples):
        order = ("Reference", "CPU") if index % 2 == 0 else ("CPU", "Reference")
        for name in order:
            start = time.perf_counter_ns()
            providers[name].evaluate(atoms, need_forces=True)
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
            if index >= warmups:
                timings[name].append(elapsed_ms)
    summaries = {
        name: {
            "n": len(values),
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for name, values in timings.items()
    }
    summaries["cpu_speedup_over_reference"] = (
        summaries["Reference"]["median_ms"] / summaries["CPU"]["median_ms"]
    )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("provider") != "amber":
        raise ValueError("Platform audit requires the pinned AmberTools manifest.")
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("warmups must be non-negative and samples must be positive.")

    records: list[dict[str, Any]] = []
    numeric_records: list[dict[str, float]] = []
    repeat_records: list[dict[str, float]] = []
    speedups: list[float] = []
    polar_statuses: list[str] = []
    lcpo_statuses: list[str] = []

    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        atoms = MOL2Reader(
            str(_case_mol2(manifest_path, case)),
            charge=int(case.get("charge", 0)),
            mult=int(case.get("multiplicity", 1)),
        )
        charges = np.asarray(case["charges_e"], dtype=np.float64)
        if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
            raise ValueError(f"Invalid charge vector for {case_id}.")

        model_records = []
        for model in manifest["models"]:
            polar, _providers = _audit_component(
                atoms,
                charges,
                model=model,
                nonpolar="none",
                expected_support=_expected_model_support(case, model),
            )
            polar_statuses.append(polar["status"])
            if polar["status"] == "success":
                numeric_records.append(polar["cpu_vs_reference"])
                repeat_records.append(polar["cpu_independent_context_repeat"])

            lcpo, _providers = _audit_component(
                atoms,
                charges,
                model=model,
                nonpolar="lcpo",
                expected_support=(
                    _expected_model_support(case, model)
                    and _expected_lcpo_support(case)
                ),
            )
            lcpo_statuses.append(lcpo["status"])
            if lcpo["status"] == "success":
                numeric_records.append(lcpo["cpu_vs_reference"])
                repeat_records.append(lcpo["cpu_independent_context_repeat"])
            model_records.append(
                {"model": model, "polar": polar, "complete_lcpo": lcpo}
            )

        (
            reference_speed,
            cpu_speed,
            _cpu_repeat,
            _reference_result,
            _cpu_result,
            _cpu_repeat_result,
        ) = _evaluate_triplet(
            atoms,
            charges,
            model="obc2",
            nonpolar="ace",
        )
        timing = _time_evaluations(
            reference_speed,
            cpu_speed,
            atoms,
            warmups=args.warmups,
            samples=args.samples,
        )
        speedups.append(float(timing["cpu_speedup_over_reference"]))
        records.append(
            {
                "case_id": case_id,
                "compound_id": case.get("compound_id"),
                "atom_count": len(atoms),
                "model_records": model_records,
                "obc2_ace_energy_force_timing": timing,
            }
        )

    def maximum(key: str, values: list[dict[str, float]]) -> float:
        return max((record[key] for record in values), default=0.0)

    summary = {
        "case_count": len(records),
        "model_slot_count": len(records) * len(manifest["models"]),
        "polar_status_counts": {
            status: polar_statuses.count(status)
            for status in sorted(set(polar_statuses))
        },
        "lcpo_status_counts": {
            status: lcpo_statuses.count(status) for status in sorted(set(lcpo_statuses))
        },
        "cpu_vs_reference_maxima": {
            "energy_abs_kcal_mol": maximum(
                "energy_abs_kcal_mol", numeric_records
            ),
            "component_max_abs_kcal_mol": maximum(
                "component_max_abs_kcal_mol", numeric_records
            ),
            "force_max_abs_kcal_mol_angstrom": maximum(
                "force_max_abs_kcal_mol_angstrom", numeric_records
            ),
        },
        "cpu_independent_context_repeat_maxima": {
            "energy_abs_kcal_mol": maximum(
                "energy_abs_kcal_mol", repeat_records
            ),
            "component_max_abs_kcal_mol": maximum(
                "component_max_abs_kcal_mol", repeat_records
            ),
            "force_max_abs_kcal_mol_angstrom": maximum(
                "force_max_abs_kcal_mol_angstrom", repeat_records
            ),
        },
        "obc2_ace_energy_force_speedup": {
            "minimum": min(speedups),
            "median": statistics.median(speedups),
            "maximum": max(speedups),
        },
    }
    numeric_pass = (
        summary["cpu_vs_reference_maxima"]["energy_abs_kcal_mol"]
        <= ENERGY_TOLERANCE_KCAL_MOL
        and summary["cpu_vs_reference_maxima"][
            "component_max_abs_kcal_mol"
        ]
        <= ENERGY_TOLERANCE_KCAL_MOL
        and summary["cpu_vs_reference_maxima"][
            "force_max_abs_kcal_mol_angstrom"
        ]
        <= FORCE_TOLERANCE_KCAL_MOL_ANGSTROM
    )
    repeat_pass = all(
        value == 0.0
        for value in summary["cpu_independent_context_repeat_maxima"].values()
    )
    support_pass = (
        polar_statuses.count("failure") == 0
        and lcpo_statuses.count("failure") == 0
    )
    local_speed_observation_pass = (
        summary["obc2_ace_energy_force_speedup"]["minimum"] > 1.0
    )

    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-openmm-platform-audit",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "route": {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "role": "Baseline/Product Route",
            "formula": (
                "E_solution(R)=E_MLIP,gas(R)+"
                "G_polar(R,q_fixed)+G_nonpolar(R)"
            ),
            "gas_phase_mm_energy": False,
            "hydration_label_residual": False,
            "retraining": False,
        },
        "claim_scope": (
            "Same-host warm OpenMM solvent-backend audit. It tests platform "
            "equivalence, independent-context determinism, and local OBC-II/ACE "
            "energy+force evaluation time; it does not change solvent physics."
        ),
        "claim_boundaries": [
            "The speedup applies to the OpenMM solvent correction only.",
            (
                "This is not a claim that MLIP plus implicit solvent is faster "
                "than bare MM."
            ),
            (
                "Platform timing is host-, OpenMM-build-, molecule-, and "
                "workload-specific."
            ),
            "No experimental hydration labels enter this audit.",
        ],
        "source_manifest": {
            "path": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
            "provider": manifest["provider"],
            "provider_version": manifest["provider_version"],
        },
        "platform_contract": {
            "product_default": DEFAULT_OPENMM_PLATFORM,
            "product_properties": dict(DEFAULT_CPU_PLATFORM_PROPERTIES),
            "parity_control": "Reference",
            "parity_control_properties": {},
        },
        "tolerances": {
            "energy_abs_kcal_mol": ENERGY_TOLERANCE_KCAL_MOL,
            "component_max_abs_kcal_mol": ENERGY_TOLERANCE_KCAL_MOL,
            "force_max_abs_kcal_mol_angstrom": (
                FORCE_TOLERANCE_KCAL_MOL_ANGSTROM
            ),
            "independent_context_repeat_required_exact": True,
        },
        "checks": {
            "support_expectations_match": support_pass,
            "cpu_reference_numeric_equivalence": numeric_pass,
            "cpu_independent_context_repeat_exact": repeat_pass,
            "same_host_cpu_faster_for_every_timed_case": (
                local_speed_observation_pass
            ),
        },
        "all_checks_pass": (
            support_pass
            and numeric_pass
            and repeat_pass
            and local_speed_observation_pass
        ),
        "summary": summary,
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "openmm": importlib.metadata.version("openmm"),
        },
        "command_provenance": command_provenance(
            __file__,
            {
                "manifest": args.manifest,
                "output": args.output,
                "warmups": args.warmups,
                "samples": args.samples,
            },
            repository_root=REPOSITORY_ROOT,
            environment_variables=("OPENMM_CPU_THREADS",),
        ),
    }
    seal_artifact(artifact)
    write_json_atomic(args.output, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="tests/solvation/data/amber_gb_reference/manifest.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/implicit-solvation/benchmarks/"
            "route1-openmm-platform-audit-2026-07-25.json"
        ),
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
