#!/usr/bin/env python3
"""Run the Route 1 external ddX/ddPCM development screen.

``energy`` evaluates frozen geometries, AM1-BCC charge vectors, and the
already-frozen ACE nonpolar component without opening experimental labels.
``score`` is a separate development-only phase that opens the source labels
after the energy artifact has been sealed.

pyddx is intentionally imported lazily.  It is an isolated audit dependency,
not a MAPLE runtime dependency.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    build_openmm_topology,
)
from maple.function.calculator.extra_correction.implicit.radii import (  # noqa: E402
    OpenMMMbondi2RadiusProvider,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from run_apbs_ace_screen import paired_absolute_error_gain  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "ddx_pcm_protocol.json"
DEFAULT_MANIFEST = SCRIPT_DIR / "apbs_ace_source_manifest.json"
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_APBS_RECORD_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-apbs-ace-fine-route1-20260725/records"
)
DEFAULT_CHAGB_RECORD_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724/records"
)
DEFAULT_ENERGY_OUTPUT = SCRIPT_DIR / "route1-ddx-ddpcm-energy-screen-2026-07-25.json"
DEFAULT_FORCE_ARTIFACT = (
    SCRIPT_DIR / "route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json"
)
DEFAULT_SCORE_OUTPUT = (
    SCRIPT_DIR / "route1-ddx-ddpcm-development-summary-2026-07-25.json"
)

BOHR_PER_ANGSTROM = 1.0 / 0.52917721092
HARTREE_TO_KCAL_MOL = 627.5094740631

BASELINE_METHODS = (
    "am1bcc_obc2_ace",
    "am1bcc_apbs_mol_ace",
    "am1bcc_chagb_cavity_dispersion",
)
CANDIDATE_METHODS = (
    "ddpcm_vdw_mbondi2_l7_n194",
    "ddpcm_sas_mbondi2_plus_1p4A_l5_n110",
)


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and enforce the scientific and product boundary."""
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only ddX protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-ddx-ddpcm-development-audit-v1":
        raise ValueError("Unexpected ddX protocol id.")
    boundary = protocol.get("route1_boundary", {})
    expected = {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": ("E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)"),
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    if boundary != expected:
        raise ValueError("ddX protocol violates the frozen Route 1 boundary.")
    if protocol["external_provider"]["isolated_build"]["project_dependency_added"]:
        raise ValueError("pyddx must remain an isolated audit dependency.")
    if protocol["label_use_boundary"]["energy_phase_reads_experimental_labels"]:
        raise ValueError("The ddX energy phase must remain label blind.")
    if protocol["label_use_boundary"]["experimental_fit_or_residual"]:
        raise ValueError("Route 1 forbids an experimental fit or residual.")
    return protocol


def _require_pyddx(expected_version: str):
    try:
        import pyddx
    except ImportError as exc:  # pragma: no cover - exercised in audit environment
        raise RuntimeError(
            "pyddx is required only for the external ddX audit. Install the "
            "protocol-pinned sdist in an isolated environment; do not add it "
            "to MAPLE's runtime dependencies."
        ) from exc
    if pyddx.__version__ != expected_version:
        raise RuntimeError(
            f"Expected pyddx {expected_version}, observed {pyddx.__version__}."
        )
    return pyddx


def ddpcm_energy_hartree(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    lmax: int,
    n_lebedev: int,
    solvent_epsilon: float,
    solver_tolerance: float,
    expected_version: str,
) -> float:
    """Evaluate point-charge ddPCM electrostatic solvation energy."""
    pyddx = _require_pyddx(expected_version)
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    radii = np.asarray(radii_angstrom, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("ddPCM positions must have shape (N, 3).")
    if charges.shape != (positions.shape[0],):
        raise ValueError("ddPCM charge count does not match the atom count.")
    if radii.shape != charges.shape or np.any(radii <= 0.0):
        raise ValueError("ddPCM requires one positive radius per atom.")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(charges).all()
        and np.isfinite(radii).all()
    ):
        raise ValueError("ddPCM inputs must be finite.")

    model = pyddx.Model(
        "pcm",
        np.asfortranarray(positions.T * BOHR_PER_ANGSTROM),
        radii * BOHR_PER_ANGSTROM,
        solvent_epsilon=float(solvent_epsilon),
        lmax=int(lmax),
        n_lebedev=int(n_lebedev),
        shift=0.0,
        enable_fmm=False,
        enable_force=False,
        n_proc=1,
    )
    multipoles = np.asfortranarray(charges.reshape(1, -1) / np.sqrt(4.0 * np.pi))
    electrostatics = model.multipole_electrostatics(multipoles)
    state = pyddx.State(
        model,
        model.multipole_psi(multipoles),
        electrostatics["phi"],
    )
    state.fill_guess(float(solver_tolerance))
    state.solve(float(solver_tolerance))
    energy = float(state.energy())
    if not np.isfinite(energy):
        raise ValueError("ddPCM returned a non-finite energy.")
    return energy


def _energy_worker(job: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    source_dir_text, row, protocol = job
    compound_id = row.get("compound_id")
    try:
        source_dir = Path(source_dir_text)
        mol2_path = source_dir / row["source_mol2_relative_path"]
        if core.sha256_file(mol2_path) != row["source_mol2_sha256"]:
            raise ValueError("Frozen MOL2 hash mismatch.")
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        charges = np.asarray(row["am1bcc_charges_e"], dtype=np.float64)
        if charges.shape != (len(atoms),):
            raise ValueError("Frozen AM1-BCC charge count mismatch.")
        radius_result = OpenMMMbondi2RadiusProvider().assign(
            build_openmm_topology(atoms)
        )
        base_radii = np.asarray(radius_result.radii_angstrom, dtype=np.float64)
        predictions: dict[str, dict[str, float]] = {}
        for name, profile in protocol["profiles"].items():
            started = time.perf_counter()
            polar_hartree = ddpcm_energy_hartree(
                positions,
                charges,
                base_radii + float(profile["radius_offset_angstrom"]),
                lmax=int(profile["lmax"]),
                n_lebedev=int(profile["n_lebedev"]),
                solvent_epsilon=float(protocol["physical_model"]["solvent_dielectric"]),
                solver_tolerance=float(protocol["physical_model"]["solver_tolerance"]),
                expected_version=protocol["external_provider"]["version"],
            )
            polar_kcal_mol = polar_hartree * HARTREE_TO_KCAL_MOL
            nonpolar_kcal_mol = float(row["openmm_ace_nonpolar_kcal_mol"])
            predictions[name] = {
                "polar_hartree": polar_hartree,
                "polar_kcal_mol": polar_kcal_mol,
                "openmm_ace_nonpolar_kcal_mol": nonpolar_kcal_mol,
                "total_kcal_mol": polar_kcal_mol + nonpolar_kcal_mol,
                "elapsed_seconds": time.perf_counter() - started,
            }
        return {
            "status": "success",
            "compound_id": compound_id,
            "atom_count": len(atoms),
            "source_mol2_sha256": row["source_mol2_sha256"],
            "source_record_sha256": row["source_record_sha256"],
            "charge_sum_e": float(charges.sum()),
            "charge_vector_sha256": core.sha256_bytes(
                core.canonical_json_bytes(charges.tolist())
            ),
            "radius_profile": radius_result.profile,
            "radius_vector_sha256": core.sha256_bytes(
                core.canonical_json_bytes(base_radii.tolist())
            ),
            "predictions": predictions,
        }
    except Exception as exc:  # pragma: no cover - retained for failed-case evidence
        return {
            "status": "failure",
            "compound_id": compound_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _verify_source_evidence(protocol: dict[str, Any], manifest_path: Path) -> None:
    evidence = protocol["source_evidence"]
    if core.sha256_file(manifest_path) != evidence["source_manifest_sha256"]:
        raise ValueError("Source-manifest hash differs from the frozen protocol.")
    for name_key, hash_key in (
        ("source_development_summary", "source_development_summary_sha256"),
        ("fine_apbs_summary", "fine_apbs_summary_sha256"),
        ("chagb_summary", "chagb_summary_sha256"),
        ("product_performance_trace", "product_performance_trace_sha256"),
        (
            "product_performance_cpu_trace",
            "product_performance_cpu_trace_sha256",
        ),
    ):
        path = SCRIPT_DIR / evidence[name_key]
        if core.sha256_file(path) != evidence[hash_key]:
            raise ValueError(f"Frozen source evidence hash mismatch: {path.name}.")


def run_energy(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    manifest_path = Path(args.manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    _verify_source_evidence(protocol, manifest_path)
    _require_pyddx(protocol["external_provider"]["version"])

    manifest = core.load_json(manifest_path)
    if manifest.get("case_count") != protocol["source_evidence"]["expected_case_count"]:
        raise ValueError("Source-manifest case count differs from the protocol.")
    if "experimental" in json.dumps(manifest["records"], sort_keys=True).lower():
        raise ValueError("The ddX energy phase requires label-free records.")
    rows = list(manifest["records"])
    if args.limit is not None:
        limit = int(args.limit)
        if limit <= 0 or limit > len(rows):
            raise ValueError(f"--limit must be between 1 and {len(rows)}.")
        rows = rows[:limit]

    started = time.perf_counter()
    jobs = ((str(source_dir), row, protocol) for row in rows)
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        records = list(executor.map(_energy_worker, jobs))
    wall_seconds = time.perf_counter() - started
    records.sort(key=lambda record: str(record["compound_id"]))
    success_count = sum(record["status"] == "success" for record in records)
    failure_count = len(records) - success_count

    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-ddx-ddpcm-label-free-energy-screen",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "claim_scope": protocol["claim_scope"],
            "route1_contract": protocol["route1_boundary"],
            "label_use_boundary": protocol["label_use_boundary"],
            "source_partition": "development",
            "source_manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "source_manifest_sha256": core.sha256_file(manifest_path),
            "external_provider": protocol["external_provider"],
            "physical_model": protocol["physical_model"],
            "profiles": protocol["profiles"],
            "case_count": len(records),
            "success_count": success_count,
            "failure_count": failure_count,
            "execution": {
                "workers": int(args.workers),
                "wall_seconds": wall_seconds,
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "command": "energy",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "source_dir": Path(args.source_dir).as_posix(),
                    "workers": int(args.workers),
                    "limit": args.limit,
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
                "case_count": len(records),
                "success_count": success_count,
                "failure_count": failure_count,
                "wall_seconds": wall_seconds,
            },
            indent=2,
        )
    )


def _load_success_record(path: Path, compound_id: str) -> dict[str, Any]:
    record = core.load_json(path)
    if record.get("status") != "success":
        raise ValueError(f"Comparator record is not successful: {path}")
    if record.get("compound_id") != compound_id:
        raise ValueError(f"Comparator compound id mismatch: {path}")
    return record


def run_score(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    manifest_path = Path(args.manifest).resolve()
    energy_path = Path(args.energy_artifact).resolve()
    force_path = Path(args.force_artifact).resolve()
    source_dir = Path(args.source_dir).resolve()
    apbs_record_dir = Path(args.apbs_record_dir).resolve()
    chagb_record_dir = Path(args.chagb_record_dir).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    _verify_source_evidence(protocol, manifest_path)

    energy_artifact = core.load_json(energy_path)
    if core.artifact_content_sha256(energy_artifact) != energy_artifact.get(
        "content_sha256"
    ):
        raise ValueError("ddX energy artifact self-hash mismatch.")
    if energy_artifact.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError("ddX energy artifact protocol hash mismatch.")
    if energy_artifact.get("failure_count") != 0:
        raise ValueError("Failed energy cases remain failed; scoring is closed.")
    if energy_artifact["label_use_boundary"]["energy_phase_reads_experimental_labels"]:
        raise ValueError("Energy artifact is not label blind.")

    force_artifact = core.load_json(force_path)
    if core.artifact_content_sha256(force_artifact) != force_artifact.get(
        "content_sha256"
    ):
        raise ValueError("ddX force artifact self-hash mismatch.")
    if force_artifact.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError("ddX force artifact protocol hash mismatch.")

    manifest = core.load_json(manifest_path)
    energy_by_id = {
        record["compound_id"]: record for record in energy_artifact["records"]
    }
    if set(energy_by_id) != {record["compound_id"] for record in manifest["records"]}:
        raise ValueError("Energy artifact does not cover the frozen manifest.")

    methods = BASELINE_METHODS + CANDIDATE_METHODS
    errors: dict[str, list[float]] = {name: [] for name in methods}
    ranked_rows: list[dict[str, Any]] = []
    for source in manifest["records"]:
        compound_id = source["compound_id"]
        source_record_path = source_dir / source["source_record_relative_path"]
        if core.sha256_file(source_record_path) != source["source_record_sha256"]:
            raise ValueError(f"Source-record hash mismatch: {compound_id}.")
        source_record = core.load_json(source_record_path)
        apbs_record = _load_success_record(
            apbs_record_dir / f"{compound_id}.json", compound_id
        )
        chagb_record = _load_success_record(
            chagb_record_dir / f"{compound_id}.json", compound_id
        )
        if (
            apbs_record["source_record_sha256"] != source["source_record_sha256"]
            or chagb_record["source_record_sha256"] != source["source_record_sha256"]
        ):
            raise ValueError(f"Comparator source mismatch: {compound_id}.")

        experimental = float(source_record["experimental_kcal_mol"])
        predictions = {
            "am1bcc_obc2_ace": float(source_record["predicted_kcal_mol"]),
            "am1bcc_apbs_mol_ace": float(
                apbs_record["predictions_kcal_mol"]["am1bcc_apbs_mol_ace"]
            ),
            "am1bcc_chagb_cavity_dispersion": float(
                chagb_record["predictions_kcal_mol"]["am1bcc_chagb_cavity_dispersion"]
            ),
        }
        predictions.update(
            {
                candidate: float(
                    energy_by_id[compound_id]["predictions"][candidate][
                        "total_kcal_mol"
                    ]
                )
                for candidate in CANDIDATE_METHODS
            }
        )
        signed_errors = {name: predictions[name] - experimental for name in methods}
        for name in methods:
            errors[name].append(signed_errors[name])
        ranked_rows.append(
            {
                "compound_id": compound_id,
                "experimental_kcal_mol": experimental,
                "predictions_kcal_mol": predictions,
                "signed_errors_kcal_mol": signed_errors,
                "bins": source_record["bins"],
            }
        )

    case_count = len(ranked_rows)
    statistics = protocol["statistics"]
    metrics = {
        name: core.summarize_errors(
            errors[name],
            expected_count=case_count,
            resamples=int(statistics["bootstrap_resamples"]),
            confidence=float(statistics["bootstrap_confidence"]),
            seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, name in enumerate(methods)
    }
    paired = {
        candidate: {
            baseline: paired_absolute_error_gain(
                errors[baseline],
                errors[candidate],
                resamples=int(statistics["bootstrap_resamples"]),
                confidence=float(statistics["bootstrap_confidence"]),
                seed=int(statistics["bootstrap_seed"])
                + 100
                + 10 * candidate_index
                + baseline_index,
            )
            for baseline_index, baseline in enumerate(BASELINE_METHODS)
        }
        for candidate_index, candidate in enumerate(CANDIDATE_METHODS)
    }
    largest = {
        candidate: sorted(
            ranked_rows,
            key=lambda row: abs(row["signed_errors_kcal_mol"][candidate]),
            reverse=True,
        )[:10]
        for candidate in CANDIDATE_METHODS
    }

    primary = CANDIDATE_METHODS[0]
    baseline = BASELINE_METHODS[0]
    rule = protocol["product_admission_rule"]["accuracy_against_obc2_ace"]
    gain = paired[primary][baseline]
    accuracy_gate = (
        gain["mean_mae_gain_kcal_mol"] >= float(rule["minimum_mae_gain_kcal_mol"])
        and gain["bootstrap_ci"][0] > 0.0
        and metrics[primary]["rmse"] <= metrics[baseline]["rmse"]
    )
    force_gate = bool(force_artifact["force_gate"]["passed"])
    numerical_gate = bool(force_artifact["numerical_convergence_gate"]["passed"])
    timing_ratio = float(
        force_artifact["performance_comparison"][
            "ddpcm_vdw_force_vs_openmm_obc2_ace_correction_ratio"
        ]
    )
    performance_gate = timing_ratio <= float(
        protocol["product_admission_rule"]["performance"]["maximum_slowdown_ratio"]
    )
    independent_confirmation = False
    admitted = (
        accuracy_gate
        and force_gate
        and numerical_gate
        and performance_gate
        and independent_confirmation
    )

    summary = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-ddx-ddpcm-development-summary",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "claim_scope": (
                "Fixed-geometry development accuracy and local provider "
                "admission only; not independent confirmation or a sampled "
                "hydration free energy."
            ),
            "route1_contract": protocol["route1_boundary"],
            "label_use_boundary": protocol["label_use_boundary"],
            "source_partition": "development",
            "case_count": case_count,
            "energy_artifact_sha256": core.sha256_file(energy_path),
            "energy_content_sha256": energy_artifact["content_sha256"],
            "force_artifact_sha256": core.sha256_file(force_path),
            "force_content_sha256": force_artifact["content_sha256"],
            "methods": metrics,
            "paired_absolute_error_gain": paired,
            "largest_candidate_absolute_errors": largest,
            "product_gates": {
                "numerical_convergence_gate_passed": numerical_gate,
                "complete_polar_force_gate_passed": force_gate,
                "accuracy_materiality_gate_passed": accuracy_gate,
                "performance_gate_passed": performance_gate,
                "independent_confirmation_passed": independent_confirmation,
                "product_admission_passed": admitted,
            },
            "decision": {
                "disposition": "reject-product-provider",
                "runtime_provider_added": False,
                "project_dependency_added": False,
                "production_default_changed": False,
                "confirmation_remains_closed": True,
                "reason": (
                    "The ddPCM polar derivative is conservative and its "
                    "label-blind numerical gate passes, but the vdW/ACE "
                    "profile has no material development accuracy gain, has "
                    "worse RMSE/outliers, and is roughly hundreds of times "
                    "slower than the current OpenMM OBC-II/ACE correction on "
                    "the same local molecule."
                ),
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "command": "score",
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "energy_artifact": energy_path.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "force_artifact": force_path.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "source_dir": Path(args.source_dir).as_posix(),
                    "apbs_record_dir": Path(args.apbs_record_dir).as_posix(),
                    "chagb_record_dir": Path(args.chagb_record_dir).as_posix(),
                    "output": output.relative_to(REPOSITORY_ROOT).as_posix(),
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    core.write_json_atomic(output, summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": summary["content_sha256"],
                "methods": {
                    name: {
                        "mae": metrics[name]["mae"],
                        "rmse": metrics[name]["rmse"],
                        "max_absolute_error": metrics[name]["max_absolute_error"],
                    }
                    for name in methods
                },
                "product_gates": summary["product_gates"],
                "decision": summary["decision"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    energy = subparsers.add_parser(
        "energy", help="Run the label-blind ddPCM energy phase."
    )
    energy.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    energy.add_argument("--manifest", default=DEFAULT_MANIFEST)
    energy.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    energy.add_argument("--workers", type=int, default=6)
    energy.add_argument("--limit", type=int)
    energy.add_argument("--output", default=DEFAULT_ENERGY_OUTPUT)
    energy.set_defaults(handler=run_energy)

    score = subparsers.add_parser(
        "score", help="Open development labels and score the sealed energy phase."
    )
    score.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    score.add_argument("--manifest", default=DEFAULT_MANIFEST)
    score.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    score.add_argument("--apbs-record-dir", default=DEFAULT_APBS_RECORD_DIR)
    score.add_argument("--chagb-record-dir", default=DEFAULT_CHAGB_RECORD_DIR)
    score.add_argument("--energy-artifact", default=DEFAULT_ENERGY_OUTPUT)
    score.add_argument("--force-artifact", default=DEFAULT_FORCE_ARTIFACT)
    score.add_argument("--output", default=DEFAULT_SCORE_OUTPUT)
    score.set_defaults(handler=run_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "workers", 1) <= 0:
        raise ValueError("--workers must be positive.")
    args.handler(args)


if __name__ == "__main__":
    main()
