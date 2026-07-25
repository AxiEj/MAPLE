#!/usr/bin/env python3
"""Evaluate and summarize a Route 1 low-level OPT -> high-level final-SP probe."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent

import sys

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_chagb_nonpolar as chagb

HARTREE_TO_KCAL_MOL = 627.5094740631


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only two-level protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-two-level-final-sp-development-v1":
        raise ValueError("Unexpected two-level final-SP protocol id.")
    if protocol["route1_boundary"] != {
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
        "fixed_charge": "AM1-BCC",
    }:
        raise ValueError("Two-level protocol violates the frozen Route 1 boundary.")
    for item in protocol["pinned_inputs"].values():
        source = REPOSITORY_ROOT / item["path"]
        if core.sha256_file(source) != item["sha256"]:
            raise ValueError(f"Pinned input hash mismatch: {item['path']}")
    return protocol


def mol2_with_positions_and_charges(
    source_text: str,
    positions_angstrom: list[list[float]],
    charges_e: list[float],
) -> str:
    if len(positions_angstrom) != len(charges_e):
        raise ValueError("Position and charge counts differ.")
    output: list[str] = []
    in_atoms = False
    atom_index = 0
    for line in source_text.splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            output.append(line)
            continue
        if not in_atoms or not line.strip():
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 9 or atom_index >= len(positions_angstrom):
            raise ValueError("Malformed or overlong MOL2 atom section.")
        position = positions_angstrom[atom_index]
        fields[2:5] = [f"{float(value):.10f}" for value in position]
        fields[8] = f"{float(charges_e[atom_index]):.12f}"
        output.append(" ".join(fields))
        atom_index += 1
    if atom_index != len(positions_angstrom):
        raise ValueError("MOL2 atom count differs from the final geometry.")
    return "\n".join(output) + "\n"


def high_level_components(
    source_mol2: Path,
    positions_angstrom: list[list[float]],
    charges_e: list[float],
    *,
    protocol: dict[str, Any],
    executables: dict[str, dict[str, Any]],
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="maple-route1-two-level-") as temporary:
        work_dir = Path(temporary)
        (work_dir / "molecule.mol2").write_text(
            mol2_with_positions_and_charges(
                source_mol2.read_text(encoding="utf-8"),
                positions_angstrom,
                charges_e,
            ),
            encoding="utf-8",
        )
        chagb._run(
            [
                executables["parmchk2"]["path"],
                "-i",
                "molecule.mol2",
                "-f",
                "mol2",
                "-o",
                "molecule.frcmod",
                "-s",
                protocol["topology"]["parmchk2_mode"],
            ],
            work_dir,
            label="parmchk2",
        )
        chagb.require_no_frcmod_nonbonded_overrides(
            (work_dir / "molecule.frcmod").read_text(encoding="utf-8")
        )
        (work_dir / "tleap.in").write_text(
            "\n".join(
                [
                    f"source {protocol['topology']['tleap_source']}",
                    f"set default PBradii {protocol['topology']['pb_radii']}",
                    "MOL = loadmol2 molecule.mol2",
                    "loadamberparams molecule.frcmod",
                    "saveamberparm MOL molecule.prmtop molecule.inpcrd",
                    "quit",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        chagb._run(
            [executables["tleap"]["path"], "-f", "tleap.in"],
            work_dir,
            label="tleap",
        )
        (work_dir / "gbnsr6.in").write_text(
            chagb._gbnsr6_input(protocol), encoding="utf-8"
        )
        chagb._run(
            [
                executables["gbnsr6"]["path"],
                "-O",
                "-i",
                "gbnsr6.in",
                "-o",
                "gbnsr6.out",
                "-p",
                "molecule.prmtop",
                "-c",
                "molecule.inpcrd",
            ],
            work_dir,
            label="gbnsr6",
        )
        gb = chagb.parse_gbnsr6_components(
            (work_dir / "gbnsr6.out").read_text(encoding="utf-8")
        )
        (work_dir / "pbsa.in").write_text(chagb._pbsa_input(protocol), encoding="utf-8")
        chagb._run(
            [
                executables["pbsa"]["path"],
                "-O",
                "-i",
                "pbsa.in",
                "-o",
                "pbsa.out",
                "-p",
                "molecule.prmtop",
                "-c",
                "molecule.inpcrd",
            ],
            work_dir,
            label="pbsa",
        )
        pb = chagb.parse_pbsa_components(
            (work_dir / "pbsa.out").read_text(encoding="utf-8")
        )
    return {
        "chagb_polar": gb["chagb_polar"],
        "pbsa_cavity": pb["cavity"],
        "pbsa_dispersion": pb["dispersion"],
    }


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    high_protocol_path = (
        REPOSITORY_ROOT / protocol["pinned_inputs"]["high_level_protocol"]["path"]
    )
    high_protocol, high_protocol_fingerprint = chagb.load_evaluation_protocol(
        high_protocol_path
    )
    executables = chagb.resolve_executables(high_protocol, None)

    relaxation_record_dir = Path(args.relaxation_record_dir).resolve()
    base_work_dir = Path(args.base_work_dir).resolve()
    expected_ids = protocol["source_selection"]["compound_ids"]
    records = []
    for compound_id in expected_ids:
        relaxation_path = relaxation_record_dir / f"{compound_id}.json"
        relaxation = core.load_json(relaxation_path)
        if relaxation["compound_id"] != compound_id:
            raise ValueError(f"Relaxation record id mismatch for {compound_id}.")
        charge_path = base_work_dir / "charges" / compound_id / "am1bcc.json"
        charge_record = core.load_json(charge_path)
        charges_e = [float(value) for value in charge_record["charges_e"]]
        source_mol2 = base_work_dir / "dataset/mol2files_gaff" / f"{compound_id}.mol2"

        successful_gas = [
            branch["gas"]["final_energy"]["gas_energy_hartree"]
            for branch in relaxation["branches"]
            if branch["gas"]["status"] == "success"
        ]
        if not successful_gas:
            raise ValueError(f"No converged gas branch for {compound_id}.")
        gas_minimum_hartree = min(successful_gas)
        candidates = []
        for branch in relaxation["branches"]:
            solution = branch["solution"]
            if solution["status"] != "success":
                continue
            components = high_level_components(
                source_mol2,
                solution["final_positions_angstrom"],
                charges_e,
                protocol=high_protocol,
                executables=executables,
            )
            correction = sum(components.values())
            gas_reorganization = (
                solution["final_energy"]["gas_energy_hartree"] - gas_minimum_hartree
            ) * HARTREE_TO_KCAL_MOL
            candidates.append(
                {
                    "partition_state_index": branch["partition_state_index"],
                    "start_roles": branch["roles"],
                    "final_geometry_sha256": solution["final_geometry_sha256"],
                    "gas_reorganization_cost_kcal_mol": gas_reorganization,
                    "components_kcal_mol": components,
                    "final_sp_solvation_kcal_mol": correction,
                    "two_level_relaxed_transfer_kcal_mol": (
                        gas_reorganization + correction
                    ),
                }
            )
        if not candidates:
            raise ValueError(f"No converged solution branch for {compound_id}.")

        low_selected_index = relaxation["result"][
            "solution_minimum_partition_state_index"
        ]
        if not any(
            item["partition_state_index"] == low_selected_index for item in candidates
        ):
            raise ValueError(f"Low-level selected minimum is absent for {compound_id}.")
        high_selected = min(
            candidates, key=lambda item: item["two_level_relaxed_transfer_kcal_mol"]
        )
        records.append(
            {
                "compound_id": compound_id,
                "source_relaxation_record_sha256": core.sha256_file(relaxation_path),
                "source_mol2_sha256": core.sha256_file(source_mol2),
                "charge_record_sha256": core.sha256_file(charge_path),
                "charge_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(charges_e)
                ),
                "candidate_count": len(candidates),
                "low_level_selected_partition_state_index": low_selected_index,
                "high_level_selected_partition_state_index": high_selected[
                    "partition_state_index"
                ],
                "selection_changed": (
                    high_selected["partition_state_index"] != low_selected_index
                ),
                "selected": high_selected,
                "candidates": sorted(
                    candidates, key=lambda item: item["partition_state_index"]
                ),
            }
        )

    output = Path(args.output).resolve()
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-two-level-opt-final-sp-rerank-energy-probe",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "case_count": len(records),
            "candidate_count": sum(item["candidate_count"] for item in records),
            "source_selection": protocol["source_selection"],
            "route1_boundary": protocol["route1_boundary"],
            "formula": protocol["formula"],
            "optimization_potential": protocol["optimization_potential"],
            "final_sp_potential": protocol["final_sp_potential"],
            "derivative_contract": protocol["derivative_contract"],
            "label_boundary": protocol["label_boundary"],
            "high_level_protocol_fingerprint": high_protocol_fingerprint,
            "provider_sha256": {
                name: details["sha256"]
                for name, details in sorted(
                    high_protocol["providers"]["executables"].items()
                )
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "command": "energy",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "relaxation_record_dir": Path(args.relaxation_record_dir).name,
                    "base_work_dir": Path(args.base_work_dir).name,
                    "output": output.name,
                },
                repository_root=REPOSITORY_ROOT,
            ),
            "records": records,
        }
    )
    core.write_json_atomic(output, artifact)
    print(f"{output} sha256={core.sha256_file(output)}")


def _metrics(records: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    errors = [record["errors_kcal_mol"][key] for record in records]
    return {
        "count": len(errors),
        "mae_kcal_mol": sum(abs(value) for value in errors) / len(errors),
        "rmse_kcal_mol": math.sqrt(
            sum(value * value for value in errors) / len(errors)
        ),
        "mse_kcal_mol": sum(errors) / len(errors),
        "max_abs_error_kcal_mol": max(abs(value) for value in errors),
    }


def _paired_comparison(
    records: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    gains = [
        abs(record["errors_kcal_mol"][baseline])
        - abs(record["errors_kcal_mol"][candidate])
        for record in records
    ]
    return {
        "mean_absolute_error_gain_kcal_mol": sum(gains) / len(gains),
        "improved_case_count": sum(value > 1.0e-12 for value in gains),
        "worsened_case_count": sum(value < -1.0e-12 for value in gains),
        "unchanged_case_count": sum(abs(value) <= 1.0e-12 for value in gains),
        "per_case_gain_kcal_mol": {
            record["compound_id"]: gain for record, gain in zip(records, gains)
        },
    }


def run_summary(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    energy_path = Path(args.energy_artifact).resolve()
    energy = core.load_json(energy_path)
    if core.artifact_content_sha256(energy) != energy.get("content_sha256"):
        raise ValueError("Two-level energy artifact content hash is inconsistent.")
    if energy.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError("Two-level energy artifact protocol hash is inconsistent.")

    relaxation_record_dir = Path(args.relaxation_record_dir).resolve()
    high_level_record_dir = Path(args.high_level_record_dir).resolve()
    records = []
    for item in energy["records"]:
        compound_id = item["compound_id"]
        relaxation_path = relaxation_record_dir / f"{compound_id}.json"
        if core.sha256_file(relaxation_path) != item["source_relaxation_record_sha256"]:
            raise ValueError(f"Relaxation record hash mismatch for {compound_id}.")
        relaxation = core.load_json(relaxation_path)
        high_level = core.load_json(high_level_record_dir / f"{compound_id}.json")
        experimental = float(relaxation["experimental_kcal_mol"])
        low_selected = next(
            candidate
            for candidate in item["candidates"]
            if candidate["partition_state_index"]
            == item["low_level_selected_partition_state_index"]
        )
        predictions = {
            "low_level_relaxed": relaxation["result"][
                "relaxed_transfer_energy_kcal_mol"
            ],
            "fixed_geometry_high_level": high_level["predictions_kcal_mol"][
                "am1bcc_chagb_cavity_dispersion"
            ],
            "two_level_low_selected": low_selected[
                "two_level_relaxed_transfer_kcal_mol"
            ],
            "two_level_high_reranked": item["selected"][
                "two_level_relaxed_transfer_kcal_mol"
            ],
        }
        records.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": experimental,
                "predictions_kcal_mol": predictions,
                "errors_kcal_mol": {
                    key: value - experimental for key, value in predictions.items()
                },
                "selection_changed": item["selection_changed"],
            }
        )

    method_keys = list(records[0]["predictions_kcal_mol"])
    paired = {
        f"{baseline}_to_two_level_high_reranked": _paired_comparison(
            records, baseline, "two_level_high_reranked"
        )
        for baseline in (
            "low_level_relaxed",
            "fixed_geometry_high_level",
            "two_level_low_selected",
        )
    }
    fixed_comparison = paired["fixed_geometry_high_level_to_two_level_high_reranked"]
    output = Path(args.output).resolve()
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": (
                "route1-two-level-opt-final-sp-rerank-energy-probe-summary"
            ),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "energy_artifact_sha256": core.sha256_file(energy_path),
            "source_selection": protocol["source_selection"],
            "label_boundary": protocol["label_boundary"],
            "case_count": len(records),
            "candidate_count": energy["candidate_count"],
            "selection_changed_case_count": sum(
                record["selection_changed"] for record in records
            ),
            "metrics": {key: _metrics(records, key) for key in method_keys},
            "paired_comparisons": paired,
            "decision": {
                "disposition": "do-not-promote-two-level-default",
                "product_default_changed": False,
                "fixed_geometry_accuracy_improved": (
                    fixed_comparison["mean_absolute_error_gain_kcal_mol"] > 0.0
                ),
                "force_claim_for_final_sp": False,
                "reason": (
                    "On this six-case label-exposed diagnostic, high-level "
                    "reranking worsens every case relative to the fixed-geometry "
                    "high-level endpoint."
                ),
            },
            "interpretation": (
                "The two-level construction is scientifically valid when both "
                "potentials are exposed, but this small development probe does not "
                "show an accuracy gain and cannot justify a product default."
            ),
            "records": records,
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "command": "summarize",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "energy_artifact": energy_path.name,
                    "relaxation_record_dir": Path(args.relaxation_record_dir).name,
                    "high_level_record_dir": Path(args.high_level_record_dir).name,
                    "output": output.name,
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    core.write_json_atomic(output, artifact)
    print(f"{output} sha256={core.sha256_file(output)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    energy = subparsers.add_parser("energy", help="Run all high-level single points.")
    energy.add_argument(
        "--protocol",
        default=BENCHMARK_DIR / "two_level_final_sp_protocol.json",
        type=Path,
    )
    energy.add_argument("--relaxation-record-dir", required=True, type=Path)
    energy.add_argument("--base-work-dir", required=True, type=Path)
    energy.add_argument("--output", required=True, type=Path)
    energy.set_defaults(handler=run_energy)

    summary = subparsers.add_parser(
        "summarize", help="Score and compare the frozen two-level energy artifact."
    )
    summary.add_argument(
        "--protocol",
        default=BENCHMARK_DIR / "two_level_final_sp_protocol.json",
        type=Path,
    )
    summary.add_argument("--energy-artifact", required=True, type=Path)
    summary.add_argument("--relaxation-record-dir", required=True, type=Path)
    summary.add_argument("--high-level-record-dir", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    summary.set_defaults(handler=run_summary)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
