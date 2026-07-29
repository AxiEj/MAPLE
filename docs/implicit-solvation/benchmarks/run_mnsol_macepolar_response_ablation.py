#!/usr/bin/env python3
"""Compare fixed and polarizable solute response on frozen MNSol records.

The selected rows are neutral absolute experimental MNSol-v2012 free
energies. Row-level data remain below ``.omx``; public output is aggregate
only. Within one run every method shares the selected continuum equation,
cavity radii, dielectric, and SMD-CDS so paired differences isolate source
representation and ML response.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import ase
import numpy as np
import torch

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_partition import (
    MNSolPartitionSelection,
    PARTITION_ARTIFACT,
    indexed_partition_record,
    validate_frozen_mnsol_partition_selection,
)
from mnsol_pilot import (
    MNSolPilotSelection,
    validate_frozen_mnsol_pilot_selection,
)
from mnsol_response_ablation import (
    ABLATION_METHODS,
    aggregate_method_metrics,
    compose_method_ledger,
    paired_method_comparison,
    solve_fixed_multipole_continuum,
)
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNet2Calculator,
)
from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
    ENERGY_IDENTITY_TOLERANCE_EV,
    NEUTRAL_DENSITY_TOLERANCE,
    SCF_ANDERSON_COEFFICIENT_L1_LIMIT,
    SCF_ANDERSON_DEPTH,
    SCF_ANDERSON_REGULARIZATION,
    SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT,
    SCF_ANDERSON_STEP_RATIO_LIMIT,
    SCF_DENSITY_TOLERANCE,
    SCF_DIPOLE_TOLERANCE_E_ANGSTROM,
    SCF_ENERGY_TOLERANCE_EV,
    SCF_FINITE_RESOLUTION_MAP_REPLAY_COUNT,
    SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM,
    SCF_FINITE_RESOLUTION_GRADIENT_SPAN_TOLERANCE_EV_PER_ANGSTROM,
    SCF_FINITE_RESOLUTION_HISTORY_LENGTH,
    SCF_FINITE_RESOLUTION_LEDGER_SPAN_TOLERANCE_EV,
    SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E,
    SCF_FINITE_RESOLUTION_POLICY_VERSION,
    SCF_FINITE_RESOLUTION_POTENTIAL_SPAN_TOLERANCE_EV,
    SCF_MAX_ITERATIONS,
    SCF_MIXING,
    SCF_SOLVER,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
    PyDDXReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    SCF_ACCEPTED_RESIDUAL_SOURCE,
    SCF_ACTUAL_RESIDUAL_OBJECTIVE_FORMULA,
    SCF_FINITE_RESOLUTION_HISTORY_SOURCE,
    SCF_REJECTED_GROWTH_ACTION,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import (
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
    DDPCM_MULTISOLVENT_SMD_PROFILE,
)
from maple.function.route2_solvents import route2_solvent_spec

ARTIFACT_NAME = "route2-mnsol-macepolar-response-ablation-v5"
SCHEMA_VERSION = 5
SCF_CONVERGENCE_CONTRACT_VERSION = "route2-scf-convergence-evidence-v4"
SCF_SOLVER_CONTRACT = {
    "scf_solver": SCF_SOLVER,
    "scf_mixing": SCF_MIXING,
    "pyddx_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
    "scf_actual_residual_objective_formula": (
        SCF_ACTUAL_RESIDUAL_OBJECTIVE_FORMULA
    ),
    "scf_growth_rejection_inequality": (
        "Phi_trial > growth_limit * Phi_anchor"
    ),
    "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
    "scf_dipole_tolerance_e_angstrom": SCF_DIPOLE_TOLERANCE_E_ANGSTROM,
    "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
    "scf_maximum_iterations": SCF_MAX_ITERATIONS,
    "scf_accepted_residual_source": SCF_ACCEPTED_RESIDUAL_SOURCE,
    "scf_rejected_growth_action": SCF_REJECTED_GROWTH_ACTION,
    "scf_rejected_attempt_status": (
        "rejected-anderson-actual-residual-growth"
    ),
    "scf_rejected_attempts_count_toward_max_iterations": True,
    "scf_rejected_attempts_excluded_from_anderson_samples": True,
    "scf_rejected_attempts_excluded_from_best_state_selection": True,
    "scf_rejected_attempts_excluded_from_finite_resolution_window": True,
    "scf_finite_resolution_history_source": (
        SCF_FINITE_RESOLUTION_HISTORY_SOURCE
    ),
    "scf_finite_resolution_policy_version": SCF_FINITE_RESOLUTION_POLICY_VERSION,
    "scf_anderson_depth": SCF_ANDERSON_DEPTH,
    "scf_anderson_regularization": SCF_ANDERSON_REGULARIZATION,
    "scf_anderson_coefficient_l1_limit": SCF_ANDERSON_COEFFICIENT_L1_LIMIT,
    "scf_anderson_step_ratio_limit": SCF_ANDERSON_STEP_RATIO_LIMIT,
    "scf_anderson_residual_growth_limit": SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT,
}
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL * EV2HARTREE
FUNCTIONAL_GROUP_COVERAGE = (
    "halogenated-hydrocarbon",
    "ketone",
    "aromatic-hydrocarbon",
    "nitro",
    "amide",
    "cyclic-diether",
    "phenol",
    "thiophenol",
    "alcohol",
    "carboxylic-acid",
)
PAIRED_COMPARISONS = (
    ("aimnet2_fixed_l0", "mace_fixed_l0"),
    ("mace_fixed_l0", "mace_fixed_l1"),
    ("mace_fixed_l1", "mace_one_shot_l1"),
    ("mace_fixed_l1", "mace_scf_l1"),
    ("mace_one_shot_l1", "mace_scf_l1"),
)
MAXIMUM_RESPONSE_STAGES = ("one-shot", "scf")
SelectionRecord = MNSolPilotSelection | MNSolPartitionSelection


@dataclass(frozen=True)
class ContinuumArm:
    """One versioned continuum-equation arm with all other axes held fixed."""

    equation: str
    display_name: str
    profile: str
    reaction_field_type: Callable[..., PyDDXReactionFieldLinearMap]


CONTINUUM_ARMS = {
    "ddpcm": ContinuumArm(
        equation="ddpcm",
        display_name="ddPCM",
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
        reaction_field_type=PyDDXPCMReactionFieldLinearMap,
    ),
    "ddcosmo": ContinuumArm(
        equation="ddcosmo",
        display_name="ddCOSMO",
        profile=DDCOSMO_MULTISOLVENT_SMD_PROFILE,
        reaction_field_type=PyDDXCOSMOReactionFieldLinearMap,
    ),
}
MACE_BOOTSTRAP_SETTING_KEYS = ("model", "d4", "model_options", "charge")


def _continuum_arm(equation: str) -> ContinuumArm:
    try:
        return CONTINUUM_ARMS[equation]
    except KeyError as exc:
        supported = ", ".join(CONTINUUM_ARMS)
        raise ValueError(
            f"Unsupported continuum equation {equation!r}; choose {supported}."
        ) from exc


def _validate_mace_bootstrap_profile_compatibility(
    mace_runtime,
    *,
    solvent: str,
    continuum_arm: ContinuumArm,
) -> None:
    """Fail closed if the shared MACE bootstrap diverges between profiles."""

    baseline = mace_runtime._settings(
        solvent,
        DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    selected = mace_runtime._settings(solvent, continuum_arm.profile)
    drifted = [
        key
        for key in MACE_BOOTSTRAP_SETTING_KEYS
        if baseline.get(key) != selected.get(key)
    ]
    baseline_solvation = baseline.get("solv")
    selected_solvation = selected.get("solv")
    if not isinstance(baseline_solvation, dict) or not isinstance(
        selected_solvation,
        dict,
    ):
        raise TypeError("MACE bootstrap requires mapping solvation settings.")
    baseline_implicit_kwargs = MACEPolCalculator.build_implicit_solvent_kwargs(
        baseline_solvation
    )
    selected_implicit_kwargs = MACEPolCalculator.build_implicit_solvent_kwargs(
        selected_solvation
    )
    if baseline_implicit_kwargs != selected_implicit_kwargs:
        drifted.append("implicit_solvent_kwargs")
    if drifted:
        raise RuntimeError(
            "Selected continuum profile changes MACE bootstrap settings "
            f"{drifted}; the response-ablation runner cannot isolate only "
            "the continuum equation."
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
            "Response ablation requires a clean Git checkout so source hashes "
            "remain tied to one execution commit."
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


def _require_private_path(path: Path, *, kind: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / ".omx").resolve())
    except ValueError as exc:
        raise ValueError(f"{kind} must remain below '.omx'.") from exc
    return resolved


def _validated_output_paths(
    *,
    private_output: Path,
    public_output: Path,
    work_dir: Path,
    complete_panel: bool,
) -> tuple[Path, Path, Path]:
    private = _require_private_path(
        private_output,
        kind="Row-level response-ablation output",
    )
    work = _require_private_path(
        work_dir,
        kind="Provider audit work directory",
    )
    public = public_output.resolve()
    if not complete_panel:
        public = _require_private_path(
            public,
            kind="Single-record response-ablation output",
        )
    return private, public, work


def _selection_indices(records: Sequence[Mapping[str, Any]]) -> list[int]:
    indices: list[int] = []
    for record in records:
        selection_index = record["selection_index"]
        if isinstance(selection_index, bool) or not isinstance(selection_index, int):
            raise TypeError("Selection indices must be integers.")
        indices.append(selection_index)
    return indices


def _row_level_data_emitted(*, complete_panel: bool) -> bool:
    # A single-record aggregate reproduces that record's experimental value.
    return not complete_panel


def _evaluated_methods(maximum_response_stage: str) -> tuple[str, ...]:
    if maximum_response_stage == "one-shot":
        return ABLATION_METHODS[:-1]
    if maximum_response_stage == "scf":
        return ABLATION_METHODS
    raise ValueError(
        "Maximum response stage must be one of "
        + ", ".join(MAXIMUM_RESPONSE_STAGES)
        + "."
    )


def _validated_evaluated_methods(
    *,
    maximum_response_stage: str,
    partition_shard: bool,
) -> tuple[str, ...]:
    methods = _evaluated_methods(maximum_response_stage)
    if maximum_response_stage != "scf" and not partition_shard:
        raise ValueError(
            "A stage-bounded response-ablation run is allowed only for a "
            "private frozen MNSol partition shard."
        )
    return methods


def _source_hashes() -> dict[str, str]:
    root = "maple/function/calculator/extra_correction/implicit"
    paths = (
        "maple/function/calculator/aimnet/_aimnet2_calculator.py",
        "maple/function/calculator/mace/_macepol_calculator.py",
        f"{root}/ddpcm_smd.py",
        f"{root}/electrostatic_pairing.py",
        f"{root}/pyddx_pcm_response.py",
        f"{root}/pyscf_smd_cds.py",
        f"{root}/route2_engine.py",
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_partition.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        "docs/implicit-solvation/benchmarks/mnsol_response_ablation.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_macepolar_multisolvent_pilot.py"
        ),
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_macepolar_response_ablation.py"
        ),
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in paths}


def _reaction_field(atoms, solvent: str, continuum_arm: ContinuumArm):
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent=solvent,
        profile=continuum_arm.profile,
    )
    reaction = continuum_arm.reaction_field_type(
        atoms.get_positions(),
        radii,
        dielectric=route2_solvent_spec(solvent).descriptors.dielectric,
        lmax=DDPCM_LMAX,
        n_lebedev=DDPCM_N_LEBEDEV,
        n_proc=1,
        solver_tolerance=DDPCM_SOLVER_TOLERANCE,
        eta=DDPCM_ETA,
    )
    return reaction, radii


def _fixed_source(
    atoms,
    solvent: str,
    coefficients: np.ndarray,
    continuum_arm: ContinuumArm,
):
    started = time.perf_counter()
    reaction, radii = _reaction_field(atoms, solvent, continuum_arm)
    state = solve_fixed_multipole_continuum(
        reaction,
        coefficients,
        charge_tolerance_e=NEUTRAL_DENSITY_TOLERANCE,
        energy_identity_tolerance_ev=ENERGY_IDENTITY_TOLERANCE_EV,
    )
    return state, radii, time.perf_counter() - started


def _fixed_ledger(experiment, cds, model_seconds, continuum_seconds, state):
    continuum_kcal = float(state["polarization_energy_hartree"]) * HARTREE_TO_KCAL_MOL
    return compose_method_ledger(
        experimental_kcal_mol=experiment,
        solute_polarization_kcal_mol=0.0,
        continuum_polarization_kcal_mol=continuum_kcal,
        smd_cds_kcal_mol=cds.energy_kcal_mol,
        wall_seconds=model_seconds + continuum_seconds,
        extra={"half_coupling_identity_error_ev": state["energy_identity_error_ev"]},
    )


def _validated_scf_ledger(
    scf: Mapping[str, Any],
    *,
    gas_energy_ev: float,
    cds_energy_kcal_mol: float,
    experimental_kcal_mol: float,
) -> dict[str, object]:
    if (
        abs(float(scf["gas_energy_hartree"]) - gas_energy_ev * EV2HARTREE)
        > 1.0e-10
    ):
        raise RuntimeError("Direct and public MACE gas energies differ.")
    if (
        abs(float(scf["smd_cds_energy_kcal_mol"]) - cds_energy_kcal_mol)
        > 1.0e-9
    ):
        raise RuntimeError("Direct and public SMD-CDS energies differ.")
    timing = scf["timing_seconds"]
    if not isinstance(timing, Mapping):
        raise TypeError("SCF timing metadata must be a mapping.")
    return compose_method_ledger(
        experimental_kcal_mol=experimental_kcal_mol,
        solute_polarization_kcal_mol=float(
            scf["solute_polarization_kcal_mol"]
        ),
        continuum_polarization_kcal_mol=float(
            scf["continuum_polarization_kcal_mol"]
        ),
        smd_cds_kcal_mol=float(scf["smd_cds_energy_kcal_mol"]),
        wall_seconds=float(timing["public_energy"]),
        extra={
            "scf_iterations": scf["scf_iterations"],
            "unmixed_density_residual_inf_e": scf[
                "unmixed_density_residual_inf_e"
            ],
            "half_coupling_identity_error_ev": scf[
                "half_coupling_identity_error_ev"
            ],
            "scf_convergence": dict(scf["scf_convergence"]),
        },
    )


def _run_audited_scf_stage(
    *,
    evaluator: Callable[[], Mapping[str, Any]],
    on_failure: Callable[[Exception], None],
    gas_energy_ev: float,
    cds_energy_kcal_mol: float,
    experimental_kcal_mol: float,
) -> dict[str, object]:
    try:
        scf = evaluator()
        return _validated_scf_ledger(
            scf,
            gas_energy_ev=gas_energy_ev,
            cds_energy_kcal_mol=cds_energy_kcal_mol,
            experimental_kcal_mol=experimental_kcal_mol,
        )
    except Exception as exc:
        on_failure(exc)
        raise


def _validate_experimental_selection(selection, protocol) -> dict[str, object]:
    if protocol.temperature_k != 298.0:
        raise RuntimeError("The frozen comparison requires 298 K.")
    if protocol.standard_state != "1M-ideal-gas-to-1M-ideal-solution":
        raise RuntimeError("The MNSol standard-state contract drifted.")
    subsets: Counter[str] = Counter()
    for selected in selection:
        record = selected.eligible_record.record
        if record.process_type != "abs" or record.charge != 0:
            raise RuntimeError(
                "Only neutral absolute gas-to-solvent MNSol records are valid."
            )
        if not math.isfinite(record.delta_g_kcal_mol):
            raise RuntimeError("MNSol experimental values must be finite.")
        subsets[record.subset] += 1
    return {
        "all_records_absolute_gas_to_solvent": True,
        "all_records_neutral": True,
        "all_values_finite": True,
        "subset_counts": dict(sorted(subsets.items())),
    }


def _metrics(records, methods: Sequence[str]):
    return {
        method: aggregate_method_metrics(records, method) for method in methods
    }


def _comparisons(records, methods: Sequence[str]):
    evaluated = frozenset(methods)
    return {
        f"{left}__to__{right}": paired_method_comparison(
            records,
            left=left,
            right=right,
        )
        for left, right in PAIRED_COMPARISONS
        if left in evaluated and right in evaluated
    }


def _private_artifact(
    *,
    execution_git_head,
    protocol,
    selection_manifest,
    dataset,
    aimnet_checkpoint,
    mace_checkpoint,
    records,
    status,
    complete_panel,
    run_kind,
    evaluated_methods,
    maximum_response_stage,
    continuum_arm,
):
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "scf_convergence_contract_version": (
            SCF_CONVERGENCE_CONTRACT_VERSION
        ),
        "scf_solver_contract": deepcopy(SCF_SOLVER_CONTRACT),
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": status,
        "complete_panel": complete_panel,
        "run_kind": run_kind,
        "evaluated_methods": list(evaluated_methods),
        "maximum_response_stage": maximum_response_stage,
        "continuum_equation": continuum_arm.equation,
        "continuum_profile": continuum_arm.profile,
        "execution_git_head": execution_git_head,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "dataset": {
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
        },
        "checkpoints": {
            "aimnet2": {
                "sha256": sha256_file(aimnet_checkpoint),
                "bytes": aimnet_checkpoint.stat().st_size,
            },
            "mace_polar": dict(mace_checkpoint),
        },
        "completed_record_count": len(records),
        "records": records,
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
            "Frozen ten-record pilot manifest used to validate prior-pilot "
            "geometry overlap for a complete MNSol partition selection."
        ),
    )
    parser.add_argument("--aimnet2-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--record-index",
        type=int,
        help="Run one zero-based selected record as a smoke test.",
    )
    parser.add_argument(
        "--maximum-response-stage",
        choices=MAXIMUM_RESPONSE_STAGES,
        default="scf",
        help=(
            "Stop after the one-shot response or continue through strict SCF. "
            "The one-shot bound is allowed only for private partition shards."
        ),
    )
    parser.add_argument(
        "--continuum-equation",
        choices=tuple(CONTINUUM_ARMS),
        default="ddpcm",
        help=(
            "Select one continuum equation while holding the cavity, source, "
            "response stages, and SMD-CDS model fixed."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    aimnet_checkpoint = args.aimnet2_checkpoint.resolve()
    if not aimnet_checkpoint.is_file():
        raise FileNotFoundError(aimnet_checkpoint)
    execution_git_head = _execution_git_head()

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    partition_shard = selection_manifest.get("artifact") == PARTITION_ARTIFACT
    partition_name: str | None = None
    pilot_selection_path: Path | None = None
    full_selection: tuple[SelectionRecord, ...]
    indexed_selection: list[tuple[int, SelectionRecord]]
    if partition_shard:
        if args.pilot_selection is None:
            raise ValueError(
                "Complete MNSol partition selection requires "
                "--pilot-selection to validate prior-pilot overlap."
            )
        pilot_selection_path = cast(Path, args.pilot_selection)
        pilot_manifest = json.loads(
            pilot_selection_path.read_text(encoding="utf-8")
        )
        full_selection = validate_frozen_mnsol_partition_selection(
            selection_manifest,
            dataset,
            protocol,
            pilot_manifest,
        )
        indexed_selection = cast(
            list[tuple[int, SelectionRecord]],
            indexed_partition_record(
                full_selection,
                args.record_index,
            ),
        )
        partition_name = str(selection_manifest["partition"])
        complete_panel = False
    else:
        full_selection = validate_frozen_mnsol_pilot_selection(
            selection_manifest,
            dataset,
            protocol,
        )
        if len(full_selection) != len(FUNCTIONAL_GROUP_COVERAGE):
            raise RuntimeError(
                "Functional-group coverage no longer matches selection."
            )
        if args.record_index is None:
            indexed_selection = list(enumerate(full_selection))
        else:
            if not 0 <= args.record_index < len(full_selection):
                raise ValueError(
                    "--record-index must lie in "
                    f"[0, {len(full_selection) - 1}]."
                )
            indexed_selection = [
                (args.record_index, full_selection[args.record_index])
            ]
        complete_panel = len(indexed_selection) == len(full_selection)
    maximum_response_stage = str(args.maximum_response_stage)
    continuum_arm = _continuum_arm(str(args.continuum_equation))
    evaluated_methods = _validated_evaluated_methods(
        maximum_response_stage=maximum_response_stage,
        partition_shard=partition_shard,
    )
    experimental_checks = _validate_experimental_selection(
        full_selection,
        protocol,
    )
    run_kind = (
        (
            "partition-record-shard"
            if maximum_response_stage == "scf"
            else "partition-record-shard-through-one-shot"
        )
        if partition_shard
        else ("ten-record-panel" if complete_panel else "single-record-smoke")
    )
    partition_display = partition_name or "pilot"
    private_output, public_output, work_dir = _validated_output_paths(
        private_output=args.private_output,
        public_output=args.public_output,
        work_dir=args.work_dir,
        complete_panel=complete_panel,
    )
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)

    # Reuse the existing validated public MACE benchmark path rather than
    # duplicating calculator construction or the same-root SCF audit ledger.
    mace_runtime = import_module("run_mnsol_macepolar_multisolvent_pilot")
    torch.set_num_threads(1)
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    aimnet = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(aimnet_checkpoint),
    )
    aimnet_load_seconds = time.perf_counter() - load_started
    load_started = time.perf_counter()
    _validate_mace_bootstrap_profile_compatibility(
        mace_runtime,
        solvent=indexed_selection[0][1].canonical_solvent,
        continuum_arm=continuum_arm,
    )
    mace = mace_runtime._load_calculator(
        mace_runtime._atoms(indexed_selection[0][1]),
        indexed_selection[0][1].canonical_solvent,
        work_dir,
    )
    mace_load_seconds = time.perf_counter() - load_started
    if str(mace.dtype) != "torch.float64":
        raise RuntimeError("Response ablation requires float64 MACE-POLAR.")
    mace_checkpoint = dict(mace.mace_polar_checkpoint_provenance)
    supported_numbers = frozenset(mace.atomic_numbers)

    records: list[dict[str, Any]] = []
    for ordinal, (selection_index, selected) in enumerate(
        indexed_selection,
        start=1,
    ):
        print(
            f"[{ordinal}/{len(indexed_selection)}] index={selection_index} "
            f"solvent={selected.canonical_solvent}",
            flush=True,
        )
        item = selected.eligible_record
        record = item.record
        prior_pilot_overlap = (
            selected.prior_pilot_geometry_overlap
            if isinstance(selected, MNSolPartitionSelection)
            else False
        )
        atoms = mace_runtime._atoms(selected)
        unsupported = sorted(set(map(int, atoms.numbers)) - supported_numbers)
        if unsupported:
            raise RuntimeError(f"Unsupported MACE-POLAR elements: {unsupported}.")

        cds_started = time.perf_counter()
        cds = pyscf_smd_cds(
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            solvent=selected.canonical_solvent,
        )
        cds_seconds = time.perf_counter() - cds_started

        aimnet_started = time.perf_counter()
        aimnet_state = aimnet.charge_state(atoms)
        aimnet_seconds = time.perf_counter() - aimnet_started
        aimnet_coefficients = np.zeros((len(atoms), 4), dtype=float)
        aimnet_coefficients[:, 0] = aimnet_state.charges_e
        aimnet_continuum, radii, aimnet_continuum_seconds = _fixed_source(
            atoms,
            selected.canonical_solvent,
            aimnet_coefficients,
            continuum_arm,
        )

        mace_gas_started = time.perf_counter()
        gas_state, _ = mace.polar_state(atoms)
        mace_gas_seconds = time.perf_counter() - mace_gas_started
        gas_density = np.asarray(gas_state.density_coefficients, dtype=float)
        if (
            gas_density.shape != (len(atoms), 4)
            or not np.all(np.isfinite(gas_density))
            or abs(float(np.sum(gas_density[:, 0]))) > NEUTRAL_DENSITY_TOLERANCE
        ):
            raise RuntimeError("MACE-POLAR gas density violates its contract.")
        gas_density_l0 = gas_density.copy()
        gas_density_l0[:, 1:] = 0.0
        mace_l0, l0_radii, mace_l0_seconds = _fixed_source(
            atoms,
            selected.canonical_solvent,
            gas_density_l0,
            continuum_arm,
        )
        mace_l1, l1_radii, mace_l1_seconds = _fixed_source(
            atoms,
            selected.canonical_solvent,
            gas_density,
            continuum_arm,
        )
        if not (np.array_equal(radii, l0_radii) and np.array_equal(radii, l1_radii)):
            raise RuntimeError("Paired methods did not share one cavity.")

        field = np.asarray(mace_l1["reaction_field_values_ev"], dtype=float)
        one_shot_started = time.perf_counter()
        one_shot_state, _ = mace.polar_state(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )
        one_shot_seconds = time.perf_counter() - one_shot_started
        one_shot_density = np.asarray(
            one_shot_state.density_coefficients,
            dtype=float,
        )
        one_shot_delta_e = (
            float(one_shot_state.energy_ev) - float(gas_state.energy_ev)
        ) * EV_TO_KCAL_MOL

        experiment = record.delta_g_kcal_mol
        shared_fixed_time = cds_seconds
        methods = {
            "aimnet2_fixed_l0": _fixed_ledger(
                experiment,
                cds,
                aimnet_seconds + shared_fixed_time,
                aimnet_continuum_seconds,
                aimnet_continuum,
            ),
            "mace_fixed_l0": _fixed_ledger(
                experiment,
                cds,
                mace_gas_seconds + shared_fixed_time,
                mace_l0_seconds,
                mace_l0,
            ),
            "mace_fixed_l1": _fixed_ledger(
                experiment,
                cds,
                mace_gas_seconds + shared_fixed_time,
                mace_l1_seconds,
                mace_l1,
            ),
            "mace_one_shot_l1": compose_method_ledger(
                experimental_kcal_mol=experiment,
                solute_polarization_kcal_mol=one_shot_delta_e,
                continuum_polarization_kcal_mol=(
                    cast(float, mace_l1["polarization_energy_hartree"])
                    * HARTREE_TO_KCAL_MOL
                ),
                smd_cds_kcal_mol=cds.energy_kcal_mol,
                wall_seconds=(
                    mace_gas_seconds + cds_seconds + mace_l1_seconds + one_shot_seconds
                ),
                extra={
                    "diagnostic_not_a_fixed_point": True,
                    "unmixed_density_residual_inf_e": float(
                        np.max(np.abs(one_shot_density - gas_density))
                    ),
                    "half_coupling_identity_error_ev": mace_l1[
                        "energy_identity_error_ev"
                    ],
                },
            ),
        }

        record_payload = {
            "selection_index": selection_index,
            "canonical_solvent": selected.canonical_solvent,
            "partition": item.partition,
            "prior_pilot_geometry_overlap": prior_pilot_overlap,
            "opaque_record_id": selected.opaque_record_id,
            "entry_number": record.entry_number,
            "geometry_handle": record.geometry_handle,
            "geometry_sha256": item.geometry.sha256,
            "solute_name": record.solute_name,
            "formula": record.formula,
            "atom_count": len(atoms),
            "subset": record.subset,
            "process_type": record.process_type,
            "charge": record.charge,
            "experimental_delta_g_kcal_mol": experiment,
            "cavity_radii_angstrom": radii.tolist(),
            "aimnet2_charges_e": aimnet_state.charges_e.tolist(),
            "aimnet2_raw_charge_residual_e": (
                aimnet_state.raw_charge_residual_e
            ),
            "mace_gas_density_coefficients": gas_density.tolist(),
            "methods": methods,
        }
        if not partition_shard:
            record_payload["functional_group_class"] = (
                FUNCTIONAL_GROUP_COVERAGE[selection_index]
            )
        records.append(record_payload)

        if maximum_response_stage == "scf":
            write_json_atomic(
                private_output,
                _private_artifact(
                    execution_git_head=execution_git_head,
                    protocol=protocol,
                    selection_manifest=selection_manifest,
                    dataset=dataset,
                    aimnet_checkpoint=aimnet_checkpoint,
                    mace_checkpoint=mace_checkpoint,
                    records=records,
                    status="running-scf",
                    complete_panel=complete_panel,
                    run_kind=run_kind,
                    evaluated_methods=tuple(methods),
                    maximum_response_stage=maximum_response_stage,
                    continuum_arm=continuum_arm,
                ),
            )
            def _persist_scf_failure(exc: Exception) -> None:
                partial_methods = tuple(methods)
                failure = _private_artifact(
                    execution_git_head=execution_git_head,
                    protocol=protocol,
                    selection_manifest=selection_manifest,
                    dataset=dataset,
                    aimnet_checkpoint=aimnet_checkpoint,
                    mace_checkpoint=mace_checkpoint,
                    records=records,
                    status="failed-during-scf",
                    complete_panel=complete_panel,
                    run_kind=run_kind,
                    evaluated_methods=partial_methods,
                    maximum_response_stage=maximum_response_stage,
                    continuum_arm=continuum_arm,
                )
                failure.update(
                    {
                        "failed_method": "mace_scf_l1",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "aggregate_metrics": _metrics(records, partial_methods),
                        "paired_method_comparisons": _comparisons(
                            records,
                            partial_methods,
                        ),
                        "actual_total_wall_seconds": (
                            time.perf_counter() - wall_started
                        ),
                    }
                )
                write_json_atomic(private_output, failure)

            methods["mace_scf_l1"] = _run_audited_scf_stage(
                evaluator=lambda: mace_runtime._evaluate_method(
                    calculator=mace,
                    atoms=atoms,
                    selected=selected,
                    method=continuum_arm.equation,
                    profile=continuum_arm.profile,
                    work_dir=work_dir,
                ),
                on_failure=_persist_scf_failure,
                gas_energy_ev=float(gas_state.energy_ev),
                cds_energy_kcal_mol=float(cds.energy_kcal_mol),
                experimental_kcal_mol=float(experiment),
            )

        if tuple(methods) != evaluated_methods:
            raise RuntimeError("Response-ablation method order drifted.")
        write_json_atomic(
            private_output,
            _private_artifact(
                execution_git_head=execution_git_head,
                protocol=protocol,
                selection_manifest=selection_manifest,
                dataset=dataset,
                aimnet_checkpoint=aimnet_checkpoint,
                mace_checkpoint=mace_checkpoint,
                records=records,
                status="running",
                complete_panel=complete_panel,
                run_kind=run_kind,
                evaluated_methods=evaluated_methods,
                maximum_response_stage=maximum_response_stage,
                continuum_arm=continuum_arm,
            ),
        )

    metrics = _metrics(records, evaluated_methods)
    comparisons = _comparisons(records, evaluated_methods)
    total_wall_seconds = time.perf_counter() - wall_started
    private = _private_artifact(
        execution_git_head=execution_git_head,
        protocol=protocol,
        selection_manifest=selection_manifest,
        dataset=dataset,
        aimnet_checkpoint=aimnet_checkpoint,
        mace_checkpoint=mace_checkpoint,
        records=records,
        status="complete",
        complete_panel=complete_panel,
        run_kind=run_kind,
        evaluated_methods=evaluated_methods,
        maximum_response_stage=maximum_response_stage,
        continuum_arm=continuum_arm,
    )
    private.update(
        {
            "aggregate_metrics": metrics,
            "paired_method_comparisons": comparisons,
            "actual_total_wall_seconds": total_wall_seconds,
        }
    )
    write_json_atomic(private_output, private)

    pyddx = import_module("pyddx")
    pyscf = import_module("pyscf")
    coverage = (
        list(FUNCTIONAL_GROUP_COVERAGE)
        if complete_panel
        else (
            []
            if partition_shard
            else [FUNCTIONAL_GROUP_COVERAGE[indexed_selection[0][0]]]
        )
    )
    if complete_panel:
        claim_boundary = (
            "Paired diagnostic over ten solvents and ten post-selection "
            "functional-group classes; one point per class/solvent cannot "
            "certify population accuracy or separate class from solvent."
        )
    elif partition_shard:
        claim_boundary = (
            f"One-record frozen MNSol {partition_display}-partition shard for "
            "paired AIMNet2/MACE response diagnosis only. It cannot be "
            "aggregated until the complete partition is evaluated."
        )
    else:
        claim_boundary = (
            "One-record engineering smoke; no ten-record accuracy claim."
        )
    if maximum_response_stage == "one-shot":
        claim_boundary += (
            " This stage-bounded run deliberately did not attempt the "
            "self-consistent method and cannot be interpreted as an SCF result."
        )
    public = {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "scf_convergence_contract_version": (
            SCF_CONVERGENCE_CONTRACT_VERSION
        ),
        "scf_solver_contract": deepcopy(SCF_SOLVER_CONTRACT),
        "visibility": (
            "public-aggregate-only"
            if complete_panel
            else "private-single-record-smoke-do-not-commit"
        ),
        "do_not_commit": not complete_panel,
        "execution_git_head": execution_git_head,
        "complete_panel": complete_panel,
        "run_kind": run_kind,
        "evaluated_methods": list(evaluated_methods),
        "maximum_response_stage": maximum_response_stage,
        "continuum_equation": continuum_arm.equation,
        "continuum_profile": continuum_arm.profile,
        "scientific_identity": {
            "shared_electrostatics": f"pyddx {continuum_arm.display_name}",
            "shared_cavity": "PySCF 2.13.1 SMD Coulomb radii",
            "shared_nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "reaction_field_projector": "local-jet",
            "strict_original_smd_equivalence": False,
            "methods": {
                "aimnet2_fixed_l0": "AIMNet2 charges; U(q)+G_CDS",
                "mace_fixed_l0": "gas MACE monopoles; U(c0_l0)+G_CDS",
                "mace_fixed_l1": "gas MACE l<=1; U(c0)+G_CDS",
                "mace_one_shot_l1": (
                    "diagnostic c0->P(c0)->M(P(c0)); "
                    "DeltaEint+U(c0,P(c0))+G_CDS; not a fixed point"
                ),
                "mace_scf_l1": (
                    "nominal same-root c*=M(P(c*)) or energy-only "
                    "finite-resolution approximate candidate under the "
                    "frozen residual policy; DeltaEint+U(c*)+G_CDS"
                ),
            },
        },
        "claim_boundary": claim_boundary,
        "experimental_reference": {
            "dataset": "Minnesota Solvation Database",
            "version": "2012",
            "doi": "10.13020/3eks-j059",
            "official_homepage": "https://comp.chem.umn.edu/mnsol/",
            "temperature_k": protocol.temperature_k,
            "standard_state": protocol.standard_state,
            "source_artifact_sha256": dataset.source_artifact_sha256,
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "selected_record_checks": experimental_checks,
            "measurement_basis": (
                "experimental partition coefficients, Henry constants, or "
                "experimental solubility and vapor-pressure combinations"
            ),
            "neutral_average_uncertainty_kcal_mol": 0.2,
            "uncertainty_scope": (
                "dataset-average estimate; not a per-record standard deviation"
            ),
            "row_level_data_emitted": _row_level_data_emitted(
                complete_panel=complete_panel
            ),
            "row_level_disclosure_reason": (
                "A one-record aggregate reproduces that record's experimental "
                "value and therefore remains private below .omx."
                if not complete_panel
                else "Only aggregate statistics over the complete panel are emitted."
            ),
        },
        "selection": {
            "artifact_sha256": sha256_file(args.selection),
            "fingerprint": selection_manifest["selection_fingerprint"],
            "record_count": len(records),
            "full_preregistered_record_count": len(full_selection),
            "partition": partition_name,
            "selection_indices": _selection_indices(records),
            "prior_pilot_geometry_overlap_count": sum(
                bool(record["prior_pilot_geometry_overlap"])
                for record in records
            ),
            "used_experimental_values": False,
            "used_model_outputs": False,
            "functional_group_assignment": (
                f"not assigned for {partition_display}-partition shards"
                if partition_shard
                else "post-selection descriptive labels; not selection criteria"
            ),
            "functional_group_coverage": coverage,
            "solvent_count": len({record["canonical_solvent"] for record in records}),
            "partition_counts": dict(
                sorted(Counter(r["partition"] for r in records).items())
            ),
        },
        "aggregate_metrics": metrics,
        "paired_method_comparisons": comparisons,
        "checkpoints": {
            "aimnet2": {
                "sha256": sha256_file(aimnet_checkpoint),
                "size_bytes": aimnet_checkpoint.stat().st_size,
            },
            "mace_polar": {
                "identifier": mace_checkpoint["identifier"],
                "sha256": mace_checkpoint["sha256"],
                "size_bytes": mace_checkpoint["size_bytes"],
                "release_url": mace_checkpoint["release_url"],
            },
        },
        "numerics": {
            "profile": continuum_arm.profile,
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
            "scf_mixing": SCF_MIXING,
            "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
            "scf_dipole_tolerance_e_angstrom": (
                SCF_DIPOLE_TOLERANCE_E_ANGSTROM
            ),
            "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
            "scf_maximum_iterations": SCF_MAX_ITERATIONS,
            "scf_attempted": maximum_response_stage == "scf",
            "scf_finite_resolution_policy": (
                {
                    "version": SCF_FINITE_RESOLUTION_POLICY_VERSION,
                    "profile_scope": DDPCM_MULTISOLVENT_SMD_PROFILE,
                    "history_length": SCF_FINITE_RESOLUTION_HISTORY_LENGTH,
                    "map_replay_count": (
                        SCF_FINITE_RESOLUTION_MAP_REPLAY_COUNT
                    ),
                    "monopole_residual_ceiling_e": (
                        SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E
                    ),
                    "dipole_residual_ceiling_e_angstrom": (
                        SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM
                    ),
                    "potential_span_tolerance_ev": (
                        SCF_FINITE_RESOLUTION_POTENTIAL_SPAN_TOLERANCE_EV
                    ),
                    "gradient_span_tolerance_ev_per_angstrom": (
                        SCF_FINITE_RESOLUTION_GRADIENT_SPAN_TOLERANCE_EV_PER_ANGSTROM
                    ),
                    "ledger_span_tolerance_ev": (
                        SCF_FINITE_RESOLUTION_LEDGER_SPAN_TOLERANCE_EV
                    ),
                }
                if continuum_arm.equation == "ddpcm"
                else None
            ),
        },
        "timing_seconds": {
            "aimnet2_model_load": aimnet_load_seconds,
            "mace_polar_model_load": mace_load_seconds,
            "actual_total_wall": total_wall_seconds,
            "per_method_definition": (
                "phase-summed estimated standalone time; shared CDS/gas "
                "phases are charged to each method that requires them"
            ),
            "status": "metadata-only-not-a-randomized-speed-ranking",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "torch": torch.__version__,
            "mace_torch": version("mace-torch"),
            "graph_longrange": version("graph-longrange"),
            "pyddx": pyddx.__version__,
            "pyscf": pyscf.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "source_files_sha256": _source_hashes(),
        "references": {
            "mace_polar": "https://arxiv.org/abs/2602.19411",
            "smd_doi": "10.1021/jp810292n",
            "ddx": "https://ddsolvation.github.io/ddX/",
            "aimnet2": "https://isayevlab.github.io/aimnetcentral/models/guide/",
        },
    }
    if partition_shard:
        assert pilot_selection_path is not None
        public["selection"]["pilot_selection_artifact_sha256"] = sha256_file(
            pilot_selection_path
        )
    write_json_atomic(public_output, public)
    print(
        f"Wrote {len(records)} record(s) to {private_output} and "
        f"{public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
