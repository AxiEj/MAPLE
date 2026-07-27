#!/usr/bin/env python3
"""Compare fixed and polarizable solute response on the frozen MNSol pilot.

The selected rows are neutral absolute experimental MNSol-v2012 free
energies. Row-level data remain below ``.omx``; public output is aggregate
only. Every method shares ddPCM, cavity radii, dielectric, and SMD-CDS so the
paired differences isolate source representation and ML response.
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
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import ase
from ase.units import Hartree
import numpy as np
import torch

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import validate_frozen_mnsol_pilot_selection
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
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
    ENERGY_IDENTITY_TOLERANCE_EV,
    NEUTRAL_DENSITY_TOLERANCE,
    SCF_DENSITY_TOLERANCE,
    SCF_ENERGY_TOLERANCE_EV,
    SCF_MAX_ITERATIONS,
    SCF_MIXING,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.route2_smd_profiles import DDPCM_MULTISOLVENT_SMD_PROFILE
from maple.function.route2_solvents import route2_solvent_spec

ARTIFACT_NAME = "route2-mnsol-macepolar-response-ablation-v1"
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL / Hartree
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


def _reaction_field(atoms, solvent: str):
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent=solvent,
        profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
    )
    reaction = PyDDXPCMReactionFieldLinearMap(
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


def _fixed_source(atoms, solvent: str, coefficients: np.ndarray):
    started = time.perf_counter()
    reaction, radii = _reaction_field(atoms, solvent)
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


def _metrics(records):
    return {
        method: aggregate_method_metrics(records, method) for method in ABLATION_METHODS
    }


def _comparisons(records):
    return {
        f"{left}__to__{right}": paired_method_comparison(
            records,
            left=left,
            right=right,
        )
        for left, right in PAIRED_COMPARISONS
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
):
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": status,
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
    parser.add_argument("--aimnet2-checkpoint", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--record-index",
        type=int,
        help="Run one zero-based selected record as a smoke test.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    private_output = _require_private_path(
        args.private_output,
        kind="Row-level response-ablation output",
    )
    work_dir = _require_private_path(
        args.work_dir,
        kind="Provider audit work directory",
    )
    aimnet_checkpoint = args.aimnet2_checkpoint.resolve()
    if not aimnet_checkpoint.is_file():
        raise FileNotFoundError(aimnet_checkpoint)
    execution_git_head = _execution_git_head()

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    full_selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )
    experimental_checks = _validate_experimental_selection(
        full_selection,
        protocol,
    )
    if len(full_selection) != len(FUNCTIONAL_GROUP_COVERAGE):
        raise RuntimeError("Functional-group coverage no longer matches selection.")
    if args.record_index is None:
        indexed_selection = list(enumerate(full_selection))
    else:
        if not 0 <= args.record_index < len(full_selection):
            raise ValueError(
                f"--record-index must lie in [0, {len(full_selection) - 1}]."
            )
        indexed_selection = [(args.record_index, full_selection[args.record_index])]
    complete_panel = len(indexed_selection) == len(full_selection)
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

    records: list[dict[str, object]] = []
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
        )
        mace_l1, l1_radii, mace_l1_seconds = _fixed_source(
            atoms,
            selected.canonical_solvent,
            gas_density,
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

        scf = mace_runtime._evaluate_method(
            calculator=mace,
            atoms=atoms,
            selected=selected,
            method="ddpcm",
            profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
            work_dir=work_dir,
        )
        if abs(scf["gas_energy_hartree"] - gas_state.energy_ev / Hartree) > 1.0e-10:
            raise RuntimeError("Direct and public MACE gas energies differ.")
        if abs(scf["smd_cds_energy_kcal_mol"] - cds.energy_kcal_mol) > 1.0e-9:
            raise RuntimeError("Direct and public SMD-CDS energies differ.")

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
                    float(mace_l1["polarization_energy_hartree"]) * HARTREE_TO_KCAL_MOL
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
            "mace_scf_l1": compose_method_ledger(
                experimental_kcal_mol=experiment,
                solute_polarization_kcal_mol=scf["solute_polarization_kcal_mol"],
                continuum_polarization_kcal_mol=scf["continuum_polarization_kcal_mol"],
                smd_cds_kcal_mol=scf["smd_cds_energy_kcal_mol"],
                wall_seconds=scf["timing_seconds"]["public_energy"],
                extra={
                    "scf_iterations": scf["scf_iterations"],
                    "unmixed_density_residual_inf_e": scf[
                        "unmixed_density_residual_inf_e"
                    ],
                    "half_coupling_identity_error_ev": scf[
                        "half_coupling_identity_error_ev"
                    ],
                },
            ),
        }
        if tuple(methods) != ABLATION_METHODS:
            raise RuntimeError("Response-ablation method order drifted.")

        records.append(
            {
                "selection_index": selection_index,
                "functional_group_class": FUNCTIONAL_GROUP_COVERAGE[selection_index],
                "canonical_solvent": selected.canonical_solvent,
                "partition": item.partition,
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
                "aimnet2_raw_charge_residual_e": (aimnet_state.raw_charge_residual_e),
                "mace_gas_density_coefficients": gas_density.tolist(),
                "methods": methods,
            }
        )
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
            ),
        )

    metrics = _metrics(records)
    comparisons = _comparisons(records)
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
    )
    private.update(
        {
            "complete_panel": complete_panel,
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
        else [FUNCTIONAL_GROUP_COVERAGE[indexed_selection[0][0]]]
    )
    public = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "public-aggregate-only",
        "execution_git_head": execution_git_head,
        "complete_panel": complete_panel,
        "run_kind": "ten-record-panel" if complete_panel else "single-record-smoke",
        "scientific_identity": {
            "shared_electrostatics": "pyddx ddPCM",
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
                "mace_scf_l1": ("same-root c*=M(P(c*)); DeltaEint+U(c*)+G_CDS"),
            },
        },
        "claim_boundary": (
            "Paired diagnostic over ten solvents and ten post-selection "
            "functional-group classes; one point per class/solvent cannot "
            "certify population accuracy or separate class from solvent."
            if complete_panel
            else "One-record engineering smoke; no ten-record accuracy claim."
        ),
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
            "row_level_data_emitted": False,
        },
        "selection": {
            "artifact_sha256": sha256_file(args.selection),
            "fingerprint": selection_manifest["selection_fingerprint"],
            "record_count": len(records),
            "full_preregistered_record_count": len(full_selection),
            "used_experimental_values": False,
            "used_model_outputs": False,
            "functional_group_assignment": (
                "post-selection descriptive labels; not selection criteria"
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
            "profile": DDPCM_MULTISOLVENT_SMD_PROFILE,
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
            "scf_mixing": SCF_MIXING,
            "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
            "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
            "scf_maximum_iterations": SCF_MAX_ITERATIONS,
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
    write_json_atomic(args.public_output, public)
    print(
        f"Wrote {len(records)} record(s) to {private_output} and "
        f"{args.public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
