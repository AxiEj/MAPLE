#!/usr/bin/env python3
"""Run the preregistered self-consistent MACE-POLAR multisolvent pilot.

The private output contains row-level MNSol values and must remain under
``.omx``.  The tracked public summary contains aggregate metrics only.  Both
ddPCM and scaled ddCOSMO are evaluated through MAPLE's public Route-2 path and
the same shared fixed-point engine.  This is a bounded diagnostic, not a
replacement for complete MNSol development/confirmation evaluation.

The default path remains the frozen ten-record panel.  ``--record-index`` is a
single-record engineering smoke only: both outputs must remain under ``.omx``,
are marked ``do_not_commit``, and cannot support population or solvent-ranking
claims.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import import_module
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import ase
from ase import Atoms
import numpy as np
import torch

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_partition import (
    PARTITION_ARTIFACT,
    indexed_partition_record,
    validate_frozen_mnsol_partition_selection,
)
from mnsol_pilot import validate_frozen_mnsol_pilot_selection
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
    SCF_ANDERSON_COEFFICIENT_L1_LIMIT,
    SCF_ANDERSON_DEPTH,
    SCF_ANDERSON_REGULARIZATION,
    SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT,
    SCF_ANDERSON_STEP_RATIO_LIMIT,
    SCF_DENSITY_TOLERANCE,
    SCF_ENERGY_TOLERANCE_EV,
    SCF_MAX_ITERATIONS,
    SCF_MIXING,
    SCF_SOLVER,
    SCF_TOTAL_CHARGE_E,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
)

ARTIFACT_NAME = "route2-mnsol-macepolar-multisolvent-pilot-v1"
FULL_PANEL_RECORD_COUNT = 10
METHOD_PROFILES = (
    ("ddpcm", DDPCM_MULTISOLVENT_SMD_PROFILE),
    ("ddcosmo", DDCOSMO_MULTISOLVENT_SMD_PROFILE),
)


def _execution_git_head() -> str:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "The MNSol pilot must run from a clean Git checkout so its source "
            "hashes remain tied to one execution commit."
        )
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Could not resolve a full execution Git commit.")
    return head


def _source_hashes() -> dict[str, str]:
    implicit_root = "maple/function/calculator/extra_correction/implicit"
    paths = (
        "maple/function/calculator/mace/_macepol_calculator.py",
        "maple/function/calculator/calculator_base.py",
        "maple/function/calculator/set_calculator.py",
        f"{implicit_root}/correction.py",
        f"{implicit_root}/ddpcm_smd.py",
        f"{implicit_root}/electrostatic_pairing.py",
        f"{implicit_root}/pyddx_pcm_response.py",
        f"{implicit_root}/pyscf_smd_cds.py",
        f"{implicit_root}/route2_engine.py",
        f"{implicit_root}/route2_fixed_point.py",
        f"{implicit_root}/route2_response.py",
        f"{implicit_root}/smd_cds.py",
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_partition.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_macepolar_multisolvent_pilot.py"
        ),
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in paths}


def _require_private_path(path: Path, *, kind: str) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            f"{kind} must remain below the repository '.omx' directory."
        ) from exc
    return resolved


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("Pilot timing samples must be finite and nonnegative.")
    return {
        "sample_count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "maximum": float(max(values)),
    }


def _metrics(
    records: list[dict[str, object]],
    method: str,
) -> dict[str, float | int]:
    method_records = [record["methods"][method] for record in records]
    errors = np.asarray(
        [record["signed_error_kcal_mol"] for record in method_records],
        dtype=float,
    )
    predicted = np.asarray(
        [record["total_solvation_kcal_mol"] for record in method_records],
        dtype=float,
    )
    experimental = np.asarray(
        [record["experimental_delta_g_kcal_mol"] for record in records],
        dtype=float,
    )
    if (
        errors.size == 0
        or not np.all(np.isfinite(errors))
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(experimental))
    ):
        raise RuntimeError("MNSol pilot metrics require finite nonempty records.")
    return {
        "record_count": int(errors.size),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(np.abs(errors))),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(errors))),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predicted)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experimental)),
        "mean_solute_polarization_kcal_mol": float(
            np.mean(
                [record["solute_polarization_kcal_mol"] for record in method_records]
            )
        ),
        "mean_continuum_polarization_kcal_mol": float(
            np.mean(
                [record["continuum_polarization_kcal_mol"] for record in method_records]
            )
        ),
        "mean_electrostatic_kcal_mol": float(
            np.mean([record["electrostatic_kcal_mol"] for record in method_records])
        ),
        "mean_smd_cds_energy_kcal_mol": float(
            np.mean([record["smd_cds_energy_kcal_mol"] for record in method_records])
        ),
        "mean_scf_iterations": float(
            np.mean([record["scf_iterations"] for record in method_records])
        ),
        "maximum_scf_iterations": int(
            max(record["scf_iterations"] for record in method_records)
        ),
        "maximum_unmixed_density_residual_e": float(
            max(record["unmixed_density_residual_inf_e"] for record in method_records)
        ),
        "maximum_half_coupling_identity_error_ev": float(
            max(record["half_coupling_identity_error_ev"] for record in method_records)
        ),
    }


def _paired_method_comparison(
    records: list[dict[str, object]],
) -> dict[str, float | int]:
    energy_differences = np.asarray(
        [
            record["methods"]["ddcosmo"]["total_solvation_kcal_mol"]
            - record["methods"]["ddpcm"]["total_solvation_kcal_mol"]
            for record in records
        ],
        dtype=float,
    )
    pcm_errors = np.asarray(
        [record["methods"]["ddpcm"]["absolute_error_kcal_mol"] for record in records],
        dtype=float,
    )
    cosmo_errors = np.asarray(
        [record["methods"]["ddcosmo"]["absolute_error_kcal_mol"] for record in records],
        dtype=float,
    )
    if (
        energy_differences.size == 0
        or not np.all(np.isfinite(energy_differences))
        or not np.all(np.isfinite(pcm_errors))
        or not np.all(np.isfinite(cosmo_errors))
    ):
        raise RuntimeError("Paired MNSol comparison requires finite nonempty records.")
    tolerance = 1.0e-12
    return {
        "record_count": int(energy_differences.size),
        "mean_ddcosmo_minus_ddpcm_kcal_mol": float(np.mean(energy_differences)),
        "minimum_ddcosmo_minus_ddpcm_kcal_mol": float(np.min(energy_differences)),
        "maximum_ddcosmo_minus_ddpcm_kcal_mol": float(np.max(energy_differences)),
        "ddcosmo_lower_absolute_error_count": int(
            np.sum(cosmo_errors < pcm_errors - tolerance)
        ),
        "ddpcm_lower_absolute_error_count": int(
            np.sum(pcm_errors < cosmo_errors - tolerance)
        ),
        "absolute_error_tie_count": int(
            np.sum(np.abs(cosmo_errors - pcm_errors) <= tolerance)
        ),
        "tie_tolerance_kcal_mol": tolerance,
    }


def _settings(solvent: str, profile: str) -> dict[str, object]:
    return CommandControl.from_settings(
        [
            "#model=macepol-m",
            "#sp",
            (
                f"#solv(implicit={solvent},method=smd,provider=pyddx,"
                f"profile={profile},response=scf,standard_state=1m,"
                "experimental=true)"
            ),
        ]
    ).as_dict()


def _atoms(selected) -> Atoms:
    geometry = selected.eligible_record.geometry
    atoms = Atoms(
        numbers=geometry.atomic_numbers,
        positions=geometry.coordinates_angstrom,
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def _load_calculator(
    first_atoms: Atoms,
    first_solvent: str,
    work_dir: Path,
):
    parameters = _settings(
        first_solvent,
        DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    return SetCalculator(
        "cpu",
        parameters["model"],
        str(work_dir / "model-load.out"),
        atoms=first_atoms,
        d4=bool(parameters.get("d4", False)),
        implicit=parameters["solv"]["method"],
        solvent=parameters["solv"]["implicit"],
        model_options=parameters.get("model_options"),
        solvation_options=parameters["solv"],
        charge_options=parameters.get("charge") or {},
    ).set_calculator()


def _evaluate_method(
    *,
    calculator,
    atoms: Atoms,
    selected,
    method: str,
    profile: str,
    work_dir: Path,
) -> dict[str, object]:
    parameters = _settings(selected.canonical_solvent, profile)
    output = work_dir / selected.opaque_record_id / method / "maple.out"
    output.parent.mkdir(parents=True, exist_ok=True)
    calculator.solvent_correction = ImplicitSolvationCorrection(
        atoms,
        parameters.get("charge") or {},
        parameters["solv"],
        output=output,
    )
    calculator.reset()
    atoms.calc = calculator
    started = time.perf_counter()
    combined_energy_hartree = float(atoms.get_potential_energy())
    wall_seconds = time.perf_counter() - started

    result = calculator.results.get("solvation")
    if not isinstance(result, dict):
        raise RuntimeError("Public MAPLE path did not return a solvation ledger.")
    components = result["components_hartree"]
    provenance = result["provenance"]
    expected_model = method
    if provenance.get("electrostatics_model") != expected_model:
        raise RuntimeError("Public MAPLE path selected the wrong continuum equation.")
    if provenance.get("profile") != profile:
        raise RuntimeError("Public MAPLE path selected the wrong profile.")
    if abs(combined_energy_hartree - result["combined_energy_hartree"]) > 1.0e-12:
        raise RuntimeError("Common finalizer combined-energy ledger drifted.")

    correction = calculator.solvent_correction
    audit_dir = Path(correction.audit_dir)
    audit_path = audit_dir / f"route2-{method}-result.json"
    state_path = audit_dir / f"route2-{method}-state.npz"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with np.load(state_path) as state:
        root_density = np.asarray(
            state["density_coefficients"],
            dtype=float,
        )
    last_scf = audit["scf"]["history"][-1]
    delta_g_hartree = float(result["delta_g_solv_hartree"])
    delta_g_kcal_mol = delta_g_hartree * HARTREE_TO_KCAL_MOL
    experiment = selected.eligible_record.record.delta_g_kcal_mol
    return {
        "profile": profile,
        "gas_energy_hartree": float(result["gas_energy_hartree"]),
        "combined_energy_hartree": combined_energy_hartree,
        "delta_g_solv_hartree": delta_g_hartree,
        "total_solvation_kcal_mol": delta_g_kcal_mol,
        "signed_error_kcal_mol": delta_g_kcal_mol - experiment,
        "absolute_error_kcal_mol": abs(delta_g_kcal_mol - experiment),
        "solute_polarization_kcal_mol": (
            float(components["solute_polarization"]) * HARTREE_TO_KCAL_MOL
        ),
        "continuum_polarization_kcal_mol": (
            float(components["pcm_polarization"]) * HARTREE_TO_KCAL_MOL
        ),
        "electrostatic_kcal_mol": (
            float(components["electrostatic"]) * HARTREE_TO_KCAL_MOL
        ),
        "smd_cds_energy_kcal_mol": (float(components["cds"]) * HARTREE_TO_KCAL_MOL),
        "scf_iterations": int(audit["scf"]["iterations"]),
        "unmixed_density_residual_inf_e": float(last_scf["density_residual_e"]),
        "scf_convergence": dict(audit["scf"]["convergence"]),
        "half_coupling_identity_error_ev": float(
            audit["polarization_energy_identity_error_ev"]
        ),
        "root_density_monopole_sum_e": float(np.sum(root_density[:, 0])),
        "runtime_provenance": provenance,
        "timing_seconds": {"public_energy": wall_seconds},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--pilot-selection",
        type=Path,
        help=(
            "Frozen ten-record pilot manifest used only to validate prior-"
            "inspection overlap in a complete partition selection."
        ),
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--record-index",
        type=int,
        help="Run one zero-based selected record as a private smoke test.",
    )
    return parser


def _indexed_selection(
    full_selection: Sequence[object],
    record_index: int | None,
) -> tuple[list[tuple[int, object]], bool]:
    if len(full_selection) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError(
            "The frozen MNSol MACE-POLAR pilot must contain exactly "
            f"{FULL_PANEL_RECORD_COUNT} records."
        )
    if record_index is None:
        return list(enumerate(full_selection)), True
    if not 0 <= record_index < len(full_selection):
        raise ValueError(f"--record-index must lie in [0, {len(full_selection) - 1}].")
    return [(record_index, full_selection[record_index])], False


def _validated_output_paths(
    *,
    private_output: Path,
    public_output: Path,
    work_dir: Path,
    complete_panel: bool,
) -> tuple[Path, Path, Path]:
    private = _require_private_path(
        private_output,
        kind="Row-level MNSol pilot output",
    )
    work = _require_private_path(
        work_dir,
        kind="MNSol provider audit work directory",
    )
    public = public_output.resolve()
    if not complete_panel:
        public = _require_private_path(
            public,
            kind="Single-record derived MNSol smoke output",
        )
    return private, public, work


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    execution_git_head = _execution_git_head()

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    partition_shard = selection_manifest.get("artifact") == PARTITION_ARTIFACT
    if partition_shard:
        if args.pilot_selection is None:
            raise ValueError(
                "Complete MNSol partition selection requires "
                "--pilot-selection to classify prior inspection overlap."
            )
        pilot_manifest = json.loads(args.pilot_selection.read_text(encoding="utf-8"))
        full_selection = validate_frozen_mnsol_partition_selection(
            selection_manifest,
            dataset,
            protocol,
            pilot_manifest,
        )
        indexed_selection = indexed_partition_record(
            full_selection,
            args.record_index,
        )
        complete_panel = False
    else:
        full_selection = validate_frozen_mnsol_pilot_selection(
            selection_manifest,
            dataset,
            protocol,
        )
        indexed_selection, complete_panel = _indexed_selection(
            full_selection,
            args.record_index,
        )
    private_output, public_output, work_dir = _validated_output_paths(
        private_output=args.private_output,
        public_output=args.public_output,
        work_dir=args.work_dir,
        complete_panel=complete_panel,
    )
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)

    torch.set_num_threads(1)
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    calculator = _load_calculator(
        _atoms(indexed_selection[0][1]),
        indexed_selection[0][1].canonical_solvent,
        work_dir,
    )
    model_load_seconds = time.perf_counter() - load_started
    if str(calculator.dtype) != "torch.float64":
        raise RuntimeError("Route-2 MNSol pilot requires float64 MACE-POLAR.")
    supported_atomic_numbers = frozenset(calculator.atomic_numbers)

    records: list[dict[str, object]] = []
    for ordinal, (selection_index, selected) in enumerate(
        indexed_selection,
        start=1,
    ):
        prior_pilot_overlap = (
            bool(selected.prior_pilot_geometry_overlap) if partition_shard else False
        )
        print(
            f"[{ordinal}/{len(indexed_selection)}] index={selection_index} "
            f"solvent={selected.canonical_solvent}",
            flush=True,
        )
        atoms = _atoms(selected)
        unsupported = sorted(
            set(int(number) for number in atoms.numbers) - supported_atomic_numbers
        )
        if unsupported:
            raise RuntimeError(
                "Preregistered MNSol record is outside the loaded "
                f"MACE-POLAR element domain: {unsupported}."
            )
        methods = {
            method: _evaluate_method(
                calculator=calculator,
                atoms=atoms,
                selected=selected,
                method=method,
                profile=profile,
                work_dir=work_dir,
            )
            for method, profile in METHOD_PROFILES
        }
        if (
            abs(
                methods["ddpcm"]["gas_energy_hartree"]
                - methods["ddcosmo"]["gas_energy_hartree"]
            )
            > 1.0e-12
        ):
            raise RuntimeError(
                "The paired continuum equations did not share one gas energy."
            )
        item = selected.eligible_record
        records.append(
            {
                "selection_index": selection_index,
                "canonical_solvent": selected.canonical_solvent,
                "mnsol_solvent": item.record.solvent,
                "partition": item.partition,
                "opaque_record_id": selected.opaque_record_id,
                "entry_number": item.record.entry_number,
                "geometry_handle": item.record.geometry_handle,
                "geometry_sha256": item.geometry.sha256,
                "solute_name": item.record.solute_name,
                "formula": item.record.formula,
                "atom_count": len(atoms),
                "subset": item.record.subset,
                "prior_pilot_geometry_overlap": prior_pilot_overlap,
                "experimental_delta_g_kcal_mol": (item.record.delta_g_kcal_mol),
                "methods": methods,
            }
        )

    total_wall_seconds = time.perf_counter() - wall_started
    if complete_panel and len(records) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError("The full MNSol pilot did not produce all ten records.")
    method_metrics = {
        method: _metrics(records, method) for method, _profile in METHOD_PROFILES
    }
    paired_comparison = _paired_method_comparison(records)
    checkpoint = dict(calculator.mace_polar_checkpoint_provenance)
    run_kind = (
        "partition-record-shard"
        if partition_shard
        else ("ten-record-panel" if complete_panel else "single-record-smoke")
    )
    if complete_panel:
        claim_boundary = (
            "This experiment-blind ten-record MNSol pilot is an engineering "
            "and early chemical diagnostic for self-consistent MACE-POLAR "
            "coarse residual point-(l<=1) multipoles with ddPCM or scaled "
            "ddCOSMO plus SMD-CDS. One record per solvent cannot certify "
            "accuracy, solvent generalization, exact-GTO forces, Gaussian "
            "solute sources, original SMD equivalence, C-PCM, COSMO-RS, a "
            "smooth solution-phase PES, OPT, TS, scan, or MD."
        )
    elif partition_shard:
        claim_boundary = (
            "One-record bounded MNSol confirmation-partition shard for "
            "provenance, convergence, energy-ledger, and timing inspection "
            "only. It cannot be aggregated until the complete frozen "
            "partition has been evaluated."
        )
    else:
        claim_boundary = (
            "One-record bounded MNSol pilot smoke for provenance, "
            "convergence, energy-ledger, and timing inspection only."
        )
    if not complete_panel:
        claim_boundary += (
            " Its derived single-row values remain private under .omx and "
            "cannot support population accuracy, solvent ranking, "
            "generalization, method selection, force, PES, OPT, TS, scan, "
            "or MD claims."
        )
    private_artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "complete_panel": complete_panel,
        "run_kind": run_kind,
        "execution_git_head": execution_git_head,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "dataset": {
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
        },
        "checkpoint": checkpoint,
        "aggregate_metrics": method_metrics,
        "paired_method_comparison": paired_comparison,
        "records": records,
    }
    write_json_atomic(private_output, private_artifact)

    method_timings = {
        method: _timing_summary(
            [
                record["methods"][method]["timing_seconds"]["public_energy"]
                for record in records
            ]
        )
        for method, _profile in METHOD_PROFILES
    }
    pyddx_runtime = import_module("pyddx")
    pyscf_runtime = import_module("pyscf")
    public_artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": (
            "public-aggregate-only"
            if complete_panel
            else "private-single-record-smoke-do-not-commit"
        ),
        "do_not_commit": not complete_panel,
        "execution_git_head": execution_git_head,
        "complete_panel": complete_panel,
        "run_kind": run_kind,
        "scientific_identity": {
            "solute_energy_model": ("official unmodified MACE-POLAR-1-M checkpoint"),
            "solute_source": ("MACE-POLAR coarse residual point-multipole-l<=1"),
            "polarization_response": "self-consistent",
            "reaction_field_projector": "local-jet",
            "continuum_equations": ["pyddx ddPCM", "pyddx ddCOSMO"],
            "nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "energy_composition": (
                "DeltaG_solv = DeltaE_MACE_intrinsic + U_continuum " "+ G_SMD-CDS"
            ),
            "strict_original_smd_equivalence": False,
            "mutual_ml_continuum_polarization": True,
            "cpcm_included": False,
            "cosmo_rs_included": False,
        },
        "claim_boundary": claim_boundary,
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
            "row_level_data_emitted": not complete_panel,
        },
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_artifact": args.selection.name,
        "selection_artifact_sha256": sha256_file(args.selection),
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "selection_record_count": len(records),
        "selection_full_preregistered_record_count": len(full_selection),
        "selection_indices": [int(record["selection_index"]) for record in records],
        "selection_prior_pilot_geometry_overlap_count": sum(
            bool(record["prior_pilot_geometry_overlap"]) for record in records
        ),
        "selection_solvent_count": len(
            {record["canonical_solvent"] for record in records}
        ),
        "selection_partition_counts": dict(
            sorted(Counter(record["partition"] for record in records).items())
        ),
        "checkpoint": {
            "identifier": checkpoint["identifier"],
            "release_url": checkpoint["release_url"],
            "size_bytes": checkpoint["size_bytes"],
            "sha256": checkpoint["sha256"],
            "redistributed": False,
        },
        "continuum_parameters": {
            "profiles": {method: profile for method, profile in METHOD_PROFILES},
            "coulomb_radii_policy": ("PySCF 2.13.1 SMD solvent-acidity-dependent"),
            "dielectric_source": "PySCF 2.13.1 SMD solvent_db",
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "n_proc": 1,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
            "full_profile_numerics": True,
        },
        "scf_parameters": {
            "solver": SCF_SOLVER,
            "mixing": SCF_MIXING,
            "density_tolerance_e": SCF_DENSITY_TOLERANCE,
            "energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
            "maximum_iterations": SCF_MAX_ITERATIONS,
            "anderson_depth": SCF_ANDERSON_DEPTH,
            "anderson_regularization": SCF_ANDERSON_REGULARIZATION,
            "anderson_coefficient_l1_limit": (SCF_ANDERSON_COEFFICIENT_L1_LIMIT),
            "anderson_step_ratio_limit": (SCF_ANDERSON_STEP_RATIO_LIMIT),
            "anderson_residual_growth_limit": (SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT),
            "total_charge_e": SCF_TOTAL_CHARGE_E,
            "residual_definition": (
                "unmixed neutral-tangent Pi0[M(P(c))-c]"
            ),
        },
        "aggregate_metrics": method_metrics,
        "paired_method_comparison": paired_comparison,
        "timing_seconds": {
            "comparison_status": (
                "metadata-only-not-a-speed-ranking; fixed "
                "ddPCM-then-ddCOSMO order and process-level cache effects "
                "are not randomized"
            ),
            "method_execution_order": [method for method, _profile in METHOD_PROFILES],
            "model_load": model_load_seconds,
            "methods": method_timings,
            "total_wall": total_wall_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "torch": torch.__version__,
            "mace_torch": version("mace-torch"),
            "graph_longrange": version("graph-longrange"),
            "pyddx": pyddx_runtime.__version__,
            "pyscf": pyscf_runtime.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "source_files_sha256": _source_hashes(),
        "references": {
            "mace_polar": "https://arxiv.org/abs/2602.19411",
            "mnsol_manual": ("https://comp.chem.umn.edu/mnsol/MNSol-v2012_Manual.pdf"),
            "smd_doi": "10.1021/jp810292n",
            "ddx_documentation": "https://ddsolvation.github.io/ddX/",
        },
    }
    write_json_atomic(public_output, public_artifact)
    print(
        f"Wrote private {private_output} and summary {public_output} "
        f"for {len(records)} preregistered records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
