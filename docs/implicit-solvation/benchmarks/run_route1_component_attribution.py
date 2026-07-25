#!/usr/bin/env python3
"""Attribute the CHA-GB endpoint gain to frozen polar/nonpolar components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
from run_apbs_ace_screen import paired_absolute_error_gain  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_component_attribution_protocol.json"
DEFAULT_MANIFEST = SCRIPT_DIR / "apbs_ace_source_manifest.json"
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_CHAGB_RECORD_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724/records"
)
DEFAULT_APBS_RECORD_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-apbs-ace-fine-route1-20260725/records"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "route1-chagb-component-attribution-2026-07-25.json"

METHODS = (
    "obc2_ace",
    "obc2_pbsa_cavity_dispersion",
    "chagb_ace",
    "chagb_surface_tension",
    "chagb_pbsa_cavity_dispersion",
    "apbs_ace",
    "apbs_pbsa_cavity_dispersion",
)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError(
            "Only Route 1 component-attribution schema version 1 is supported."
        )
    if protocol.get("protocol_id") != "maple-route1-chagb-component-attribution-v1":
        raise ValueError("Unexpected Route 1 component-attribution protocol id.")
    expected_boundary = {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": ("E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)"),
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    if protocol.get("route1_boundary") != expected_boundary:
        raise ValueError("Component attribution violates the Route 1 boundary.")
    if protocol["label_use_boundary"]["experimental_fit_or_residual"]:
        raise ValueError("Route 1 component attribution forbids residual fitting.")
    if set(protocol["component_endpoints"]) != set(METHODS):
        raise ValueError("Component-attribution endpoint set is incomplete.")
    return protocol


def _verify_source_evidence(protocol: dict[str, Any], manifest_path: Path) -> None:
    evidence = protocol["source_evidence"]
    if core.sha256_file(manifest_path) != evidence["source_manifest_sha256"]:
        raise ValueError("Component-attribution source-manifest hash mismatch.")
    for name_key, hash_key in (
        ("source_development_summary", "source_development_summary_sha256"),
        ("chagb_summary", "chagb_summary_sha256"),
        ("fine_apbs_summary", "fine_apbs_summary_sha256"),
    ):
        path = SCRIPT_DIR / evidence[name_key]
        if core.sha256_file(path) != evidence[hash_key]:
            raise ValueError(f"Frozen evidence hash mismatch: {path.name}.")


def _load_success_record(path: Path, compound_id: str) -> dict[str, Any]:
    record = core.load_json(path)
    if record.get("status") != "success":
        raise ValueError(f"Component record is not successful: {path}.")
    if record.get("compound_id") != compound_id:
        raise ValueError(f"Component record compound id mismatch: {path}.")
    return record


def _component_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_kcal_mol": float(array.mean()),
        "standard_deviation_kcal_mol": float(array.std()),
        "p05_kcal_mol": float(np.quantile(array, 0.05)),
        "p95_kcal_mol": float(np.quantile(array, 0.95)),
    }


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    manifest_path = Path(args.manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    chagb_record_dir = Path(args.chagb_record_dir).resolve()
    apbs_record_dir = Path(args.apbs_record_dir).resolve()
    output = Path(args.output).resolve()

    protocol = load_protocol(protocol_path)
    _verify_source_evidence(protocol, manifest_path)
    manifest = core.load_json(manifest_path)
    if manifest.get("case_count") != protocol["source_evidence"]["expected_case_count"]:
        raise ValueError("Component-attribution case count differs from protocol.")

    errors: dict[str, list[float]] = {name: [] for name in METHODS}
    component_values: dict[str, list[float]] = {
        "obc2_polar": [],
        "chagb_polar": [],
        "apbs_polar": [],
        "ace_nonpolar": [],
        "gbnsr6_surface_tension": [],
        "pbsa_cavity_dispersion": [],
    }
    records: list[dict[str, Any]] = []
    for source in manifest["records"]:
        compound_id = source["compound_id"]
        source_record_path = source_dir / source["source_record_relative_path"]
        if core.sha256_file(source_record_path) != source["source_record_sha256"]:
            raise ValueError(f"Source-record hash mismatch: {compound_id}.")
        source_record = core.load_json(source_record_path)
        chagb_record = _load_success_record(
            chagb_record_dir / f"{compound_id}.json", compound_id
        )
        apbs_record = _load_success_record(
            apbs_record_dir / f"{compound_id}.json", compound_id
        )
        if (
            chagb_record["source_record_sha256"] != source["source_record_sha256"]
            or apbs_record["source_record_sha256"] != source["source_record_sha256"]
        ):
            raise ValueError(f"Component source mismatch: {compound_id}.")

        source_components = source_record["components_kcal_mol"]
        chagb_components = chagb_record["components_kcal_mol"]
        apbs_components = apbs_record["components_kcal_mol"]
        components = {
            "obc2_polar": float(source_components["polar"]),
            "chagb_polar": float(chagb_components["chagb_polar"]),
            "apbs_polar": float(apbs_components["apbs_mol_lpb_polar"]),
            "ace_nonpolar": float(source_components["nonpolar"]),
            "gbnsr6_surface_tension": float(chagb_components["gbnsr6_surface_tension"]),
            "pbsa_cavity_dispersion": float(
                chagb_components["pbsa_cavity"] + chagb_components["pbsa_dispersion"]
            ),
        }
        predictions = {
            "obc2_ace": (components["obc2_polar"] + components["ace_nonpolar"]),
            "obc2_pbsa_cavity_dispersion": (
                components["obc2_polar"] + components["pbsa_cavity_dispersion"]
            ),
            "chagb_ace": (components["chagb_polar"] + components["ace_nonpolar"]),
            "chagb_surface_tension": (
                components["chagb_polar"] + components["gbnsr6_surface_tension"]
            ),
            "chagb_pbsa_cavity_dispersion": (
                components["chagb_polar"] + components["pbsa_cavity_dispersion"]
            ),
            "apbs_ace": (components["apbs_polar"] + components["ace_nonpolar"]),
            "apbs_pbsa_cavity_dispersion": (
                components["apbs_polar"] + components["pbsa_cavity_dispersion"]
            ),
        }
        experimental = float(source_record["experimental_kcal_mol"])
        signed_errors = {name: predictions[name] - experimental for name in METHODS}
        for name in METHODS:
            errors[name].append(signed_errors[name])
        for name, value in components.items():
            component_values[name].append(value)
        records.append(
            {
                "compound_id": compound_id,
                "source_record_sha256": source["source_record_sha256"],
                "experimental_kcal_mol": experimental,
                "components_kcal_mol": components,
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": signed_errors,
                "bins": source_record["bins"],
            }
        )

    records.sort(key=lambda record: record["compound_id"])
    statistics = protocol["statistics"]
    metrics = {
        name: core.summarize_errors(
            errors[name],
            expected_count=len(records),
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, name in enumerate(METHODS)
    }
    baseline = "obc2_ace"
    paired = {
        name: paired_absolute_error_gain(
            errors[baseline],
            errors[name],
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + 100 + index,
        )
        for index, name in enumerate(METHODS)
        if name != baseline
    }

    factorial = protocol["attribution"]["factorial_endpoints"]
    baseline_mae = float(metrics[factorial["baseline"]]["mae"])
    polar_only_mae = float(metrics[factorial["polar_only_swap"]]["mae"])
    nonpolar_only_mae = float(metrics[factorial["nonpolar_only_swap"]]["mae"])
    joint_mae = float(metrics[factorial["joint_swap"]]["mae"])
    polar_shapley = 0.5 * (
        (baseline_mae - polar_only_mae) + (nonpolar_only_mae - joint_mae)
    )
    nonpolar_shapley = 0.5 * (
        (baseline_mae - nonpolar_only_mae) + (polar_only_mae - joint_mae)
    )
    interaction = (polar_only_mae - baseline_mae) - (joint_mae - nonpolar_only_mae)
    total_gain = baseline_mae - joint_mae
    if not np.isclose(polar_shapley + nonpolar_shapley, total_gain, atol=1.0e-12):
        raise AssertionError("Component Shapley values do not close.")

    polar_gain = paired[factorial["polar_only_swap"]]
    nonpolar_gain = paired[factorial["nonpolar_only_swap"]]
    joint_gain = paired[factorial["joint_swap"]]
    polar_only_pass = (
        polar_gain["mean_mae_gain_kcal_mol"] > 0.0
        and polar_gain["bootstrap_ci"][0] > 0.0
    )
    nonpolar_only_pass = (
        nonpolar_gain["mean_mae_gain_kcal_mol"] > 0.0
        and nonpolar_gain["bootstrap_ci"][0] > 0.0
    )
    joint_pass = (
        joint_gain["mean_mae_gain_kcal_mol"] > 0.0
        and joint_gain["bootstrap_ci"][0] > 0.0
    )
    co_calibration_evidence = (
        not polar_only_pass and not nonpolar_only_pass and joint_pass
    )

    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-chagb-component-attribution",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "claim_scope": protocol["claim_scope"],
            "route1_contract": protocol["route1_boundary"],
            "label_use_boundary": protocol["label_use_boundary"],
            "source_partition": "development",
            "case_count": len(records),
            "source_manifest_sha256": core.sha256_file(manifest_path),
            "component_endpoints": protocol["component_endpoints"],
            "methods": metrics,
            "paired_absolute_error_gain_vs_obc2_ace": paired,
            "component_statistics": {
                name: _component_statistics(values)
                for name, values in component_values.items()
            },
            "component_differences": {
                "chagb_minus_obc2_polar": _component_statistics(
                    (
                        np.asarray(component_values["chagb_polar"])
                        - np.asarray(component_values["obc2_polar"])
                    ).tolist()
                ),
                "pbsa_cavity_dispersion_minus_ace": _component_statistics(
                    (
                        np.asarray(component_values["pbsa_cavity_dispersion"])
                        - np.asarray(component_values["ace_nonpolar"])
                    ).tolist()
                ),
            },
            "factorial_shapley_attribution": {
                "endpoint_mae_kcal_mol": {
                    "baseline": baseline_mae,
                    "polar_only_swap": polar_only_mae,
                    "nonpolar_only_swap": nonpolar_only_mae,
                    "joint_swap": joint_mae,
                },
                "total_joint_mae_gain_kcal_mol": total_gain,
                "polar_swap_shapley_gain_kcal_mol": polar_shapley,
                "nonpolar_swap_shapley_gain_kcal_mol": nonpolar_shapley,
                "marginal_gain_interaction_kcal_mol": interaction,
                "closure_residual_kcal_mol": (
                    polar_shapley + nonpolar_shapley - total_gain
                ),
                "interpretation": protocol["attribution"]["interpretation"],
            },
            "decision": {
                "isolated_chagb_polar_swap_passed": polar_only_pass,
                "isolated_pbsa_nonpolar_swap_passed": nonpolar_only_pass,
                "joint_chagb_pbsa_swap_passed_development_gate": joint_pass,
                "co_calibrated_pairing_evidence": co_calibration_evidence,
                "runtime_provider_added": False,
                "production_default_changed": False,
                "confirmation_remains_closed": True,
                "next_research_target": (
                    "A conservative implementation must preserve a jointly "
                    "justified CHA-like polar and matching nonpolar/radius "
                    "parameterization. Porting only PBSA cavity/dispersion to "
                    "OBC-II or only CHA-GB polar to ACE is rejected."
                ),
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "source_dir": Path(args.source_dir).as_posix(),
                    "chagb_record_dir": Path(args.chagb_record_dir).as_posix(),
                    "apbs_record_dir": Path(args.apbs_record_dir).as_posix(),
                    "output": output.relative_to(REPOSITORY_ROOT).as_posix(),
                },
                repository_root=REPOSITORY_ROOT,
            ),
            "records": records,
        }
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "methods": {
                    name: {
                        "mae": metrics[name]["mae"],
                        "rmse": metrics[name]["rmse"],
                        "max_absolute_error": metrics[name]["max_absolute_error"],
                    }
                    for name in METHODS
                },
                "factorial_shapley_attribution": artifact[
                    "factorial_shapley_attribution"
                ],
                "decision": artifact["decision"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--chagb-record-dir", default=DEFAULT_CHAGB_RECORD_DIR)
    parser.add_argument("--apbs-record-dir", default=DEFAULT_APBS_RECORD_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
