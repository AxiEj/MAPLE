#!/usr/bin/env python3
"""Run and summarize the Route 1 GBr6 development-only energy screen."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent

import sys

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import benchmark_core as core
from maple.function.calculator.extra_correction.implicit.openmm_gb import (
    build_openmm_topology,
)
from maple.function.calculator.extra_correction.implicit.radii import (
    OpenMMAmberGBRadiusProvider,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

ENERGY_PATTERN = re.compile(r"Delta Ggbr6\s*=\s*([-+0-9.Ee]+)")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only GBr6 screen protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-gbr6-development-screen-v1":
        raise ValueError("Unexpected GBr6 screen protocol id.")
    if protocol["route1_boundary"] != {
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
        "fixed_charge": "AM1-BCC",
    }:
        raise ValueError("GBr6 protocol violates the frozen Route 1 boundary.")
    return protocol


def parse_gbr6_energy(stdout: str) -> float:
    match = ENERGY_PATTERN.search(stdout)
    if match is None:
        raise ValueError("GBr6 output does not contain a Delta Ggbr6 energy.")
    energy = float(match.group(1))
    if not np.isfinite(energy):
        raise ValueError("GBr6 returned a non-finite energy.")
    return energy


def render_qcd(atoms, charges: np.ndarray, radii_angstrom: np.ndarray) -> str:
    if charges.shape != (len(atoms),) or radii_angstrom.shape != (len(atoms),):
        raise ValueError("GBr6 charge/radius count does not match the atom count.")
    if not np.isfinite(charges).all() or not np.isfinite(radii_angstrom).all():
        raise ValueError("GBr6 charges and radii must be finite.")
    if np.any(radii_angstrom <= 0.0):
        raise ValueError("GBr6 radii must be positive.")

    lines = []
    for index, (atom, position, charge, radius) in enumerate(
        zip(atoms, atoms.positions, charges, radii_angstrom), start=1
    ):
        atom_marker = "H" if atom.symbol == "H" else atom.symbol[0]
        lines.append(
            f"ATOM {index:5d} MOL {atom_marker:1s} "
            f"{position[0]:12.6f} {position[1]:12.6f} {position[2]:12.6f} "
            f"{charge:14.10f} {radius:8.4f}"
        )
    return "\n".join(lines) + "\n"


def run_gbr6(
    executable: Path,
    qcd_text: str,
    endpoint: dict[str, Any],
) -> float:
    with tempfile.TemporaryDirectory(prefix="maple-route1-gbr6-") as temporary:
        work_dir = Path(temporary)
        (work_dir / "molecule.qcd").write_text(qcd_text, encoding="utf-8")
        stdin = "\n".join(
            [
                "molecule.qcd",
                str(endpoint["solute_dielectric"]),
                str(endpoint["solvent_dielectric"]),
                str(endpoint["salt_mM"]),
                str(endpoint["temperature_K"]),
            ]
        )
        completed = subprocess.run(
            [str(executable)],
            input=stdin + "\n",
            text=True,
            capture_output=True,
            cwd=work_dir,
            timeout=30,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            "GBr6 failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_gbr6_energy(completed.stdout)


def _verify_external_source(
    protocol: dict[str, Any],
    *,
    archive: Path,
    fortran_source: Path,
    executable: Path,
) -> dict[str, str]:
    if not executable.is_file():
        raise FileNotFoundError(f"GBr6 executable not found: {executable}")
    observed = {
        "archive_sha256": core.sha256_file(archive),
        "fortran_sha256": core.sha256_file(fortran_source),
        "executable_sha256": core.sha256_file(executable),
    }
    expected = protocol["source"]
    if observed["archive_sha256"] != expected["archive_sha256"]:
        raise ValueError("GBr6 archive SHA-256 does not match the frozen protocol.")
    if observed["fortran_sha256"] != expected["fortran_sha256"]:
        raise ValueError("GBr6 Fortran SHA-256 does not match the frozen protocol.")
    return observed


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    source_hashes = _verify_external_source(
        protocol,
        archive=Path(args.source_archive).resolve(),
        fortran_source=Path(args.fortran_source).resolve(),
        executable=Path(args.executable).resolve(),
    )
    manifest_path = Path(args.manifest).resolve()
    manifest = core.load_json(manifest_path)
    if "experimental" in json.dumps(manifest, sort_keys=True).lower():
        raise ValueError("GBr6 energy phase requires a label-free source manifest.")

    records_spec = manifest["records"]
    limit = len(records_spec) if args.limit is None else int(args.limit)
    if limit <= 0 or limit > len(records_spec):
        raise ValueError(f"--limit must be between 1 and {len(records_spec)}.")

    source_work_dir = Path(args.source_work_dir).resolve()
    executable = Path(args.executable).resolve()
    radius_provider = OpenMMAmberGBRadiusProvider("gbn")
    endpoint = protocol["energy_endpoint"]
    records = []
    for row in records_spec[:limit]:
        mol2_path = source_work_dir / row["source_mol2_relative_path"]
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        charges = np.asarray(row["am1bcc_charges_e"], dtype=np.float64)
        radius_result = radius_provider.assign(build_openmm_topology(atoms))
        if radius_result.profile != endpoint["radius_profile"]:
            raise ValueError(
                "Radius provider profile differs from the frozen protocol."
            )
        energy = run_gbr6(
            executable,
            render_qcd(atoms, charges, radius_result.radii_angstrom),
            endpoint,
        )
        records.append(
            {
                "compound_id": row["compound_id"],
                "gbr6_polar_kcal_mol": energy,
                "source_mol2_sha256": row["source_mol2_sha256"],
                "source_record_sha256": row["source_record_sha256"],
                "charge_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(charges.tolist())
                ),
                "radius_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(radius_result.radii_angstrom.tolist())
                ),
            }
        )

    output = Path(args.output).resolve()
    command = core.command_provenance(
        __file__,
        {
            "command": "energy",
            "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "source_work_dir": Path(args.source_work_dir).name,
            "source_archive": Path(args.source_archive).name,
            "fortran_source": Path(args.fortran_source).name,
            "executable": Path(args.executable).name,
            "limit": limit,
            "output": output.name,
        },
        repository_root=REPOSITORY_ROOT,
    )
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-gbr6-label-free-energy-screen",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "case_count": len(records),
            "label_use_boundary": protocol["execution_boundary"],
            "route1_boundary": protocol["route1_boundary"],
            "energy_endpoint": endpoint,
            "source": {
                **source_hashes,
                "archive_url": protocol["source"]["archive_url"],
                "fortran_path": protocol["source"]["fortran_path"],
                "license": protocol["source"]["license"],
            },
            "radius_provider_provenance": radius_result.provenance,
            "command_provenance": command,
            "records": records,
        }
    )
    core.write_json_atomic(output, artifact)
    print(f"{output} sha256={core.sha256_file(output)}")


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    errors = np.asarray(
        [row[key] - row["experimental_kcal_mol"] for row in rows],
        dtype=np.float64,
    )
    return {
        "n": int(errors.size),
        "mse_kcal_mol": float(errors.mean()),
        "mae_kcal_mol": float(np.abs(errors).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors * errors))),
        "max_absolute_error_kcal_mol": float(np.abs(errors).max()),
    }


def _paired_gain(
    rows: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    baseline_errors = np.asarray(
        [abs(row[baseline] - row["experimental_kcal_mol"]) for row in rows],
        dtype=np.float64,
    )
    candidate_errors = np.asarray(
        [abs(row[candidate] - row["experimental_kcal_mol"]) for row in rows],
        dtype=np.float64,
    )
    gain = baseline_errors - candidate_errors
    rng = np.random.default_rng(20260724)
    sample_indices = rng.integers(
        0, gain.size, size=(10_000, gain.size), endpoint=False
    )
    means = gain[sample_indices].mean(axis=1)
    return {
        "mean_mae_gain_kcal_mol": float(gain.mean()),
        "bootstrap_ci_95_kcal_mol": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "improved_case_count": int(np.count_nonzero(gain > 1.0e-12)),
        "unchanged_case_count": int(np.count_nonzero(np.abs(gain) <= 1.0e-12)),
        "worsened_case_count": int(np.count_nonzero(gain < -1.0e-12)),
    }


def run_summary(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    energy_path = Path(args.energy_artifact).resolve()
    energy = core.load_json(energy_path)
    if core.artifact_content_sha256(energy) != energy.get("content_sha256"):
        raise ValueError("GBr6 energy artifact content hash is inconsistent.")
    if energy.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError("GBr6 energy artifact protocol hash is inconsistent.")
    if energy["case_count"] != 526:
        raise ValueError("Frozen GBr6 development summary requires all 526 cases.")

    source_work_dir = Path(args.source_work_dir).resolve()
    high_level_record_dir = Path(args.high_level_record_dir).resolve()
    seen: set[str] = set()
    rows = []
    for item in energy["records"]:
        compound_id = item["compound_id"]
        if compound_id in seen:
            raise ValueError(f"Duplicate GBr6 compound id: {compound_id}")
        seen.add(compound_id)
        source_path = (
            source_work_dir
            / "records/development"
            / f"{compound_id}__am1bcc__obc2.json"
        )
        if core.sha256_file(source_path) != item["source_record_sha256"]:
            raise ValueError(f"Source record hash mismatch for {compound_id}.")
        source_record = core.load_json(source_path)
        high_record = core.load_json(high_level_record_dir / f"{compound_id}.json")
        components = high_record["components_kcal_mol"]
        pbsa_nonpolar = components["pbsa_cavity"] + components["pbsa_dispersion"]
        rows.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": source_record["experimental_kcal_mol"],
                "obc2_ace_kcal_mol": source_record["predicted_kcal_mol"],
                "chagb_cavity_dispersion_kcal_mol": high_record["predictions_kcal_mol"][
                    "am1bcc_chagb_cavity_dispersion"
                ],
                "gbr6_cavity_dispersion_kcal_mol": (
                    item["gbr6_polar_kcal_mol"] + pbsa_nonpolar
                ),
            }
        )

    method_keys = [
        "obc2_ace_kcal_mol",
        "chagb_cavity_dispersion_kcal_mol",
        "gbr6_cavity_dispersion_kcal_mol",
    ]
    output = Path(args.output).resolve()
    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-gbr6-development-summary",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "energy_artifact_sha256": core.sha256_file(energy_path),
            "case_count": len(rows),
            "label_use_boundary": protocol["execution_boundary"],
            "methods": {key: _metrics(rows, key) for key in method_keys},
            "paired_gain": {
                "obc2_to_gbr6": _paired_gain(
                    rows, "obc2_ace_kcal_mol", "gbr6_cavity_dispersion_kcal_mol"
                ),
                "chagb_to_gbr6": _paired_gain(
                    rows,
                    "chagb_cavity_dispersion_kcal_mol",
                    "gbr6_cavity_dispersion_kcal_mol",
                ),
            },
            "decision": {
                "disposition": "reject-product-provider",
                "accuracy_gate_passed": False,
                "released_force_interface_identified": False,
                "maintained_dependency_gate_passed": False,
                "product_default_changed": False,
                "reason": (
                    "GBr6/PBSA cavity-dispersion is less accurate than both frozen "
                    "comparators, and the released program exposes no force interface."
                ),
            },
            "interpretation": (
                "Development-only fixed-geometry energy screen with no fit. "
                "Experimental labels were opened only for this summary."
            ),
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "command": "summarize",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "energy_artifact": energy_path.name,
                    "source_work_dir": Path(args.source_work_dir).name,
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

    energy = subparsers.add_parser("energy", help="Run label-free GBr6 energies.")
    energy.add_argument(
        "--protocol",
        default=BENCHMARK_DIR / "gbr6_screen_protocol.json",
        type=Path,
    )
    energy.add_argument(
        "--manifest",
        default=BENCHMARK_DIR / "chagb_nonpolar_source_manifest.json",
        type=Path,
    )
    energy.add_argument("--source-work-dir", required=True, type=Path)
    energy.add_argument("--source-archive", required=True, type=Path)
    energy.add_argument("--fortran-source", required=True, type=Path)
    energy.add_argument("--executable", required=True, type=Path)
    energy.add_argument("--limit", type=int)
    energy.add_argument("--output", required=True, type=Path)
    energy.set_defaults(handler=run_energy)

    summary = subparsers.add_parser(
        "summarize", help="Open development labels and summarize a complete screen."
    )
    summary.add_argument(
        "--protocol",
        default=BENCHMARK_DIR / "gbr6_screen_protocol.json",
        type=Path,
    )
    summary.add_argument("--energy-artifact", required=True, type=Path)
    summary.add_argument("--source-work-dir", required=True, type=Path)
    summary.add_argument("--high-level-record-dir", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    summary.set_defaults(handler=run_summary)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
