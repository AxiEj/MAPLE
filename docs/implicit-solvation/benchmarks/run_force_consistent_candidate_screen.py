#!/usr/bin/env python3
"""Screen the remaining maintained analytical Route-1 GB/nonpolar candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    command_provenance,
    seal_artifact,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    KJ_PER_MOL_PER_HARTREE,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    OpenMMGB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

DEFAULT_SOURCE_WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "chagb_nonpolar_source_manifest.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "route1-force-consistent-candidate-screen-2026-07-24.json"
GB_MODELS = ("hct", "obc1", "obc2", "gbn", "gbn2")
ALPB_ALPHA = 0.571412
SOLUTE_DIELECTRIC = 1.0
SOLVENT_DIELECTRIC = 78.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_724


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(errors: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(errors), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty error vector.")
    return {
        "n": int(values.size),
        # FreeSolv convention: MSE is mean signed error, not mean-squared error.
        "mse_kcal_mol": float(np.mean(values)),
        "mae_kcal_mol": float(np.mean(np.abs(values))),
        "rmse_kcal_mol": float(np.sqrt(np.mean(values**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(values))),
    }


def _paired_gain(
    baseline_errors: Iterable[float], candidate_errors: Iterable[float]
) -> tuple[float, list[float], dict[str, int]]:
    baseline = np.asarray(list(baseline_errors), dtype=float)
    candidate = np.asarray(list(candidate_errors), dtype=float)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError("Paired errors must have the same non-zero shape.")
    gains = np.abs(baseline) - np.abs(candidate)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, gains.size, size=(BOOTSTRAP_RESAMPLES, gains.size))
    bootstrap = np.mean(gains[indices], axis=1)
    tolerance = 1.0e-12
    return (
        float(np.mean(gains)),
        [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        {
            "improved": int(np.sum(gains > tolerance)),
            "unchanged": int(np.sum(np.abs(gains) <= tolerance)),
            "worsened": int(np.sum(gains < -tolerance)),
        },
    )


def _combined_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _record_path(records_dir: Path, compound_id: str, model: str) -> Path:
    return records_dir / f"{compound_id}__am1bcc__{model}.json"


def _same_case_gbn2_comparison(
    records_dir: Path, compound_ids: list[str]
) -> dict[str, float | int]:
    obc2_errors: list[float] = []
    gbn2_errors: list[float] = []
    for compound_id in compound_ids:
        gbn2 = json.loads(
            _record_path(records_dir, compound_id, "gbn2").read_text(encoding="utf-8")
        )
        if gbn2["status"] != "success":
            continue
        obc2 = json.loads(
            _record_path(records_dir, compound_id, "obc2").read_text(encoding="utf-8")
        )
        if obc2["status"] != "success":
            raise ValueError(f"OBC-II failed on the GBn2 subset: {compound_id}.")
        experimental = float(gbn2["experimental_kcal_mol"])
        if float(obc2["experimental_kcal_mol"]) != experimental:
            raise ValueError(f"Experimental label mismatch for {compound_id}.")
        obc2_errors.append(float(obc2["predicted_kcal_mol"]) - experimental)
        gbn2_errors.append(float(gbn2["predicted_kcal_mol"]) - experimental)
    obc2_mae = float(np.mean(np.abs(obc2_errors)))
    gbn2_mae = float(np.mean(np.abs(gbn2_errors)))
    return {
        "case_count": len(gbn2_errors),
        "obc2_mae_kcal_mol": obc2_mae,
        "gbn2_mae_kcal_mol": gbn2_mae,
        "paired_mae_change_gbn2_minus_obc2_kcal_mol": gbn2_mae - obc2_mae,
    }


def _screen_alpb(
    records_dir: Path, compound_ids: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    beta = SOLUTE_DIELECTRIC / SOLVENT_DIELECTRIC
    scale = 1.0 / (1.0 + ALPB_ALPHA * beta)
    models: dict[str, Any] = {}
    raw_metrics: dict[str, Any] = {}
    for model in GB_MODELS:
        baseline_errors: list[float] = []
        candidate_errors: list[float] = []
        failure_count = 0
        maximum_charge_sum = 0.0
        for compound_id in compound_ids:
            record = json.loads(
                _record_path(records_dir, compound_id, model).read_text(
                    encoding="utf-8"
                )
            )
            if record["status"] != "success":
                failure_count += 1
                continue
            charge_sum = abs(sum(float(value) for value in record["charges_e"]))
            maximum_charge_sum = max(maximum_charge_sum, charge_sum)
            if charge_sum > 1.0e-10:
                raise ValueError(
                    f"Neutral ALPB simplification invalid for {compound_id}/{model}."
                )
            experimental = float(record["experimental_kcal_mol"])
            polar = float(record["components_kcal_mol"]["polar"])
            nonpolar = float(record["components_kcal_mol"]["nonpolar"])
            baseline = float(record["predicted_kcal_mol"]) - experimental
            candidate = scale * polar + nonpolar - experimental
            baseline_errors.append(baseline)
            candidate_errors.append(candidate)
        gain, confidence_interval, outcomes = _paired_gain(
            baseline_errors, candidate_errors
        )
        baseline_metrics = _metrics(baseline_errors)
        candidate_metrics = _metrics(candidate_errors)
        models[model] = {
            "case_count": len(candidate_errors),
            "failure_count": failure_count,
            "scale_factor": scale,
            "maximum_absolute_charge_sum_e": maximum_charge_sum,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "paired_mae_gain_kcal_mol": gain,
            "paired_mae_gain_bootstrap_ci_kcal_mol": confidence_interval,
            "case_outcomes": outcomes,
            "decision": "reject-product-default",
        }
        raw_metrics[model] = {
            "success_count": len(candidate_errors),
            "failure_count": failure_count,
            **baseline_metrics,
        }
    return (
        {
            "formula": (
                "For neutral Q=0: G_ALPB,polar = "
                "G_GB,polar/(1 + alpha*epsilon_in/epsilon_out)"
            ),
            "alpha": ALPB_ALPHA,
            "solute_dielectric": SOLUTE_DIELECTRIC,
            "solvent_dielectric": SOLVENT_DIELECTRIC,
            "electrostatic_size_term": (
                "The Q^2/A term vanishes for the declared neutral domain."
            ),
            "derivative_capability": (
                "Analytical in this neutral domain: scale the same GB polar "
                "energy and force by one constant."
            ),
            "models": models,
            "decision_reason": (
                "The default OBC-II paired MAE gain is only about "
                "0.00235 kcal/mol and its bootstrap interval crosses zero."
            ),
        },
        raw_metrics,
    )


def _screen_lcpo(
    source_work_dir: Path,
    records_dir: Path,
    manifest_records: list[dict[str, Any]],
) -> dict[str, Any]:
    kcal_per_hartree = KJ_PER_MOL_PER_HARTREE / 4.184
    supported: list[str] = []
    failures: list[dict[str, str]] = []
    baseline_errors: list[float] = []
    candidate_errors: list[float] = []
    records: list[dict[str, Any]] = []
    maximum_polar_difference = 0.0
    provider_version: str | None = None
    for row in manifest_records:
        compound_id = row["compound_id"]
        mol2 = source_work_dir / row["source_mol2_relative_path"]
        if _sha256_file(mol2) != row["source_mol2_sha256"]:
            raise ValueError(f"Frozen MOL2 hash mismatch for {compound_id}.")
        atoms = MOL2Reader(str(mol2), charge=0, mult=1)
        try:
            provider = OpenMMGB(
                atoms,
                np.asarray(row["am1bcc_charges_e"], dtype=np.float64),
                model="obc2",
                nonpolar="lcpo",
                platform="Reference",
            )
            result = provider.evaluate(atoms)
        except ValueError as exc:
            failures.append({"compound_id": compound_id, "reason": str(exc)})
            continue

        source_record = _record_path(records_dir, compound_id, "obc2")
        if _sha256_file(source_record) != row["source_record_sha256"]:
            raise ValueError(f"Frozen OBC-II record hash mismatch for {compound_id}.")
        source = json.loads(source_record.read_text(encoding="utf-8"))
        if (
            source.get("status") != "success"
            or source.get("nonpolar") != "ace"
            or not np.array_equal(
                np.asarray(source["charges_e"], dtype=np.float64),
                np.asarray(row["am1bcc_charges_e"], dtype=np.float64),
            )
        ):
            raise ValueError(f"Unexpected OBC-II/ACE source record for {compound_id}.")

        baseline_prediction = float(source["predicted_kcal_mol"])
        candidate_prediction = float(result.energy_hartree * kcal_per_hartree)
        experimental = float(source["experimental_kcal_mol"])
        baseline_error = baseline_prediction - experimental
        candidate_error = candidate_prediction - experimental
        source_polar_kcal_mol = float(source["components_kcal_mol"]["polar"])
        ace_nonpolar_kcal_mol = float(
            source["components_kcal_mol"]["nonpolar"]
        )
        polar_kcal_mol = float(
            result.components_hartree["polar"] * kcal_per_hartree
        )
        lcpo_nonpolar_kcal_mol = float(
            result.components_hartree["nonpolar"] * kcal_per_hartree
        )
        polar_difference = polar_kcal_mol - source_polar_kcal_mol
        if not (
            np.isclose(
                baseline_prediction,
                source_polar_kcal_mol + ace_nonpolar_kcal_mol,
                rtol=0.0,
                atol=1.0e-10,
            )
            and np.isclose(
                candidate_prediction,
                polar_kcal_mol + lcpo_nonpolar_kcal_mol,
                rtol=0.0,
                atol=1.0e-10,
            )
            and np.isclose(
                candidate_prediction - baseline_prediction,
                lcpo_nonpolar_kcal_mol - ace_nonpolar_kcal_mol,
                rtol=0.0,
                atol=1.0e-10,
            )
        ):
            raise ValueError(
                f"ACE/LCPO component identity failed for {compound_id}."
            )
        maximum_polar_difference = max(
            maximum_polar_difference,
            abs(polar_difference),
        )
        provider_version = str(provider.provenance["provider_version"])
        supported.append(compound_id)
        baseline_errors.append(baseline_error)
        candidate_errors.append(candidate_error)
        records.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": experimental,
                "baseline_obc2_ace_kcal_mol": baseline_prediction,
                "candidate_obc2_lcpo_kcal_mol": candidate_prediction,
                "baseline_signed_error_kcal_mol": baseline_error,
                "candidate_signed_error_kcal_mol": candidate_error,
                "obc2_polar_kcal_mol": polar_kcal_mol,
                "ace_nonpolar_kcal_mol": ace_nonpolar_kcal_mol,
                "lcpo_nonpolar_kcal_mol": lcpo_nonpolar_kcal_mol,
                "obc2_polar_difference_kcal_mol": polar_difference,
            }
        )

    gain, confidence_interval, outcomes = _paired_gain(
        baseline_errors,
        candidate_errors,
    )
    reason_counts = Counter(failure["reason"] for failure in failures)
    expected = len(manifest_records)
    return {
        "implementation": "OpenMM CustomGBForce plus openmm.app.internal.lcpo",
        "provider_version": provider_version,
        "evaluation_platform": "Reference",
        "derivative_capability": "analytical where parameterized",
        "supported_count": len(supported),
        "unsupported_count": len(failures),
        "coverage_fraction": len(supported) / expected,
        "unsupported_reason_counts": dict(sorted(reason_counts.items())),
        "unsupported_compound_ids": [
            failure["compound_id"] for failure in failures
        ],
        "first_unsupported_cases": failures[:10],
        "same_supported_case_comparison": {
            "case_count": len(records),
            "baseline_obc2_ace": _metrics(baseline_errors),
            "candidate_obc2_lcpo": _metrics(candidate_errors),
            "paired_mae_gain_ace_minus_lcpo_kcal_mol": gain,
            "paired_mae_gain_bootstrap_ci_kcal_mol": confidence_interval,
            "case_outcomes": outcomes,
            "maximum_obc2_polar_difference_kcal_mol": (
                maximum_polar_difference
            ),
            "records": records,
        },
        "decision": "reject-product-default",
        "decision_reason": (
            "The upstream topology parameter table rejects 72/526 molecules, "
            "and LCPO is less accurate than ACE on the identical 454-case "
            "supported subset; dropping failures or inventing fallback "
            "parameters is not allowed."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_work_dir = Path(args.source_work_dir).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_records = manifest["records"]
    compound_ids = [row["compound_id"] for row in manifest_records]
    if (
        int(manifest.get("case_count", -1)) != 526
        or compound_ids != sorted(compound_ids)
        or len(compound_ids) != len(set(compound_ids))
    ):
        raise ValueError("Expected the frozen sorted 526-case source manifest.")
    records_dir = source_work_dir / "records/development"
    source_records = [
        _record_path(records_dir, compound_id, model)
        for compound_id in compound_ids
        for model in GB_MODELS
    ]
    missing = [path for path in source_records if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} frozen source records.")

    alpb, raw_metrics = _screen_alpb(records_dir, compound_ids)
    lcpo = _screen_lcpo(source_work_dir, records_dir, manifest_records)
    same_case_gbn2 = _same_case_gbn2_comparison(records_dir, compound_ids)
    obc2 = raw_metrics["obc2"]
    gbn2 = raw_metrics["gbn2"]
    payload = {
        "schema_version": 1,
        "artifact_type": "route1-force-consistent-candidate-screen",
        "case_count": len(compound_ids),
        "source_partition": "development",
        "metric_definitions": {
            "mse_kcal_mol": (
                "mean signed error in kcal/mol; not mean-squared error"
            ),
            "mae_kcal_mol": "mean absolute error in kcal/mol",
            "rmse_kcal_mol": "root-mean-square error in kcal/mol",
        },
        "route1_boundary": {
            "fixed_am1bcc_charges": True,
            "gas_phase_mm_energy": False,
            "retraining": False,
            "hydration_label_residual": False,
        },
        "label_use_boundary": {
            "source_records_contain_development_labels": True,
            "candidate_parameters_fit_here": False,
            "lcpo_predictions_recomputed_from_frozen_coordinates_and_charges": True,
            "interpretation": (
                "Post hoc development screen of literature-fixed physical "
                "models; not label-blind confirmation and not certification."
            ),
        },
        "source_evidence": {
            "manifest": str(source_manifest_path),
            "manifest_sha256": _sha256_file(source_manifest_path),
            "source_record_count": len(source_records),
            "source_records_combined_sha256": _combined_hash(source_records),
        },
        "baseline_obc2_ace": {
            "success_count": obc2["success_count"],
            "failure_count": obc2["failure_count"],
            "mse_kcal_mol": obc2["mse_kcal_mol"],
            "mae_kcal_mol": obc2["mae_kcal_mol"],
            "rmse_kcal_mol": obc2["rmse_kcal_mol"],
            "maximum_absolute_error_kcal_mol": obc2["maximum_absolute_error_kcal_mol"],
        },
        "candidates": {
            "neutral_alpb": alpb,
            "gbn2": {
                "implementation": "OpenMM GBSAGBn2Force / Amber igb=8",
                "derivative_capability": "analytical",
                "success_count": gbn2["success_count"],
                "failure_count": gbn2["failure_count"],
                "mse_kcal_mol": gbn2["mse_kcal_mol"],
                "mae_kcal_mol": gbn2["mae_kcal_mol"],
                "rmse_kcal_mol": gbn2["rmse_kcal_mol"],
                "maximum_absolute_error_kcal_mol": gbn2[
                    "maximum_absolute_error_kcal_mol"
                ],
                "same_515_case_comparison": same_case_gbn2,
                "decision": "reject-product-default",
                "decision_reason": (
                    "Worse development MAE than OBC-II/ACE and 11 "
                    "phosphorus-containing molecules fail closed."
                ),
            },
            "lcpo": lcpo,
        },
        "decision": {
            "product_default_changed": False,
            "retained_default": "AM1-BCC/OBC-II/ACE",
            "reason": (
                "No remaining maintained analytical candidate passes both a "
                "material accuracy gate and the full declared applicability "
                "gate; LCPO also worsens accuracy on its supported subset."
            ),
        },
        "literature": {
            "alpb": "Sigalov, Fenley, and Onufriev, J. Chem. Phys. 124, 124902 (2006), DOI:10.1063/1.2177251",
            "gbn2": "Nguyen, Roe, and Simmerling, JCTC 2013, DOI:10.1021/ct3010485",
            "lcpo": (
                "Weiser, Shenkin, and Still, J. Comput. Chem. 20, 217-230 "
                "(1999), DOI:10.1002/(SICI)1096-987X(19990130)"
                "20:2<217::AID-JCC4>3.0.CO;2-A"
            ),
            "amber25_manual": "https://ambermd.org/doc12/Amber25.pdf",
        },
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUDA_VISIBLE_DEVICES",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
            ),
        ),
    }
    seal_artifact(payload)
    output = Path(args.output).resolve()
    write_json_atomic(output, payload)
    print(f"Wrote Route-1 analytical candidate screen to {output}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-work-dir", default=str(DEFAULT_SOURCE_WORK_DIR))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
