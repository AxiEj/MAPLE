#!/usr/bin/env python3
"""Run a frozen three-record MACE-POLAR solvent-diversity canary.

The canary reuses the ten-record pilot's dataset validation, MAPLE public
calculation path, physical ledgers, and descriptive metrics.  It has a
separate preregistration and artifact identity so three rows can never be
mistaken for the ten-record pilot or a population-level benchmark.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import ase
import numpy as np
import torch

from benchmark_core import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import (
    MNSolPilotSelection,
    validate_frozen_mnsol_pilot_selection,
)
import run_mnsol_macepolar_multisolvent_pilot as pilot

ARTIFACT_NAME = "route2-mnsol-macepolar-solvent-diversity-canary-v1"
PREREG_ARTIFACT = (
    "route2-mnsol-macepolar-solvent-diversity-canary-prereg-v1"
)
RUN_KIND = "three-record-solvent-diversity-canary"
CANARY_SOLVENTS = ("water", "dimethylsulfoxide", "hexane")
CLAIM_BOUNDARY = (
    "This frozen three-record water/DMSO/hexane run is a bounded "
    "solvent-diversity canary for provenance, strict SCF convergence, "
    "energy-ledger closure, descriptive experiment error, and timing only. "
    "It cannot certify population accuracy, solvent generalization, "
    "continuum-equation ranking, forces, a smooth solution-phase PES, OPT, "
    "TS, scan, or MD."
)


def _selection_fingerprint(records: object) -> str:
    return sha256_bytes(canonical_json_bytes(records))


def _validated_canary_selection(
    prereg: dict[str, Any],
    full_selection: Sequence[MNSolPilotSelection],
    *,
    protocol_id: str,
    protocol_fingerprint: str,
    parent_selection_fingerprint: str,
) -> list[tuple[int, MNSolPilotSelection]]:
    expected = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 1,
        "run_kind": RUN_KIND,
        "protocol_id": protocol_id,
        "protocol_fingerprint": protocol_fingerprint,
        "parent_selection_fingerprint": parent_selection_fingerprint,
    }
    mismatches = [
        key for key, value in expected.items() if prereg.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "MNSol canary preregistration mismatch: "
            + ", ".join(sorted(mismatches))
            + "."
        )
    records = prereg.get("selected_records")
    if not isinstance(records, list) or len(records) != len(CANARY_SOLVENTS):
        raise ValueError("The MNSol canary must freeze exactly three records.")
    if prereg.get("selection_fingerprint") != _selection_fingerprint(records):
        raise ValueError("The MNSol canary selection fingerprint is invalid.")

    selected: list[tuple[int, MNSolPilotSelection]] = []
    observed_solvents: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each MNSol canary record must be an object.")
        index = record.get("selection_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(full_selection)
        ):
            raise ValueError("MNSol canary selection_index is invalid.")
        parent = full_selection[index]
        solvent = str(getattr(parent, "canonical_solvent"))
        opaque_id = str(getattr(parent, "opaque_record_id"))
        if record.get("canonical_solvent") != solvent:
            raise ValueError("MNSol canary solvent no longer matches its parent.")
        if record.get("opaque_record_id") != opaque_id:
            raise ValueError("MNSol canary record no longer matches its parent.")
        observed_solvents.append(solvent)
        selected.append((index, parent))
    if tuple(observed_solvents) != CANARY_SOLVENTS:
        raise ValueError(
            "The MNSol canary solvent order must remain water, "
            "dimethylsulfoxide, hexane."
        )
    return selected


def _validated_output_paths(
    *,
    private_output: Path,
    summary_output: Path,
    work_dir: Path,
) -> tuple[Path, Path, Path]:
    return (
        pilot._require_private_path(
            private_output,
            kind="Row-level MNSol canary output",
        ),
        pilot._require_private_path(
            summary_output,
            kind="Derived MNSol canary summary",
        ),
        pilot._require_private_path(
            work_dir,
            kind="MNSol canary provider audit work directory",
        ),
    )


def _repo_relative_path(path: Path, *, kind: str) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"{kind} must remain inside the repository.") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    execution_git_head = pilot._execution_git_head()
    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    parent_manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    full_selection = validate_frozen_mnsol_pilot_selection(
        parent_manifest,
        dataset,
        protocol,
    )
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("parent_selection_artifact") != args.selection.name:
        raise ValueError(
            "The MNSol canary preregistration points to a different parent "
            "selection artifact."
        )
    prereg_relative_path = _repo_relative_path(
        args.prereg,
        kind="MNSol canary preregistration",
    )
    indexed_selection = _validated_canary_selection(
        prereg,
        full_selection,
        protocol_id=protocol.protocol_id,
        protocol_fingerprint=protocol.fingerprint,
        parent_selection_fingerprint=parent_manifest["selection_fingerprint"],
    )
    private_output, summary_output, work_dir = _validated_output_paths(
        private_output=args.private_output,
        summary_output=args.summary_output,
        work_dir=args.work_dir,
    )
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)

    torch.set_num_threads(1)
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    calculator = pilot._load_calculator(
        pilot._atoms(indexed_selection[0][1]),
        indexed_selection[0][1].canonical_solvent,
        work_dir,
    )
    model_load_seconds = time.perf_counter() - load_started
    if str(calculator.dtype) != "torch.float64":
        raise RuntimeError("The MNSol canary requires float64 MACE-POLAR.")
    supported_atomic_numbers = frozenset(calculator.atomic_numbers)

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
        atoms = pilot._atoms(selected)
        unsupported = sorted(
            set(int(number) for number in atoms.numbers)
            - supported_atomic_numbers
        )
        if unsupported:
            raise RuntimeError(
                "Preregistered MNSol canary record is outside the loaded "
                f"MACE-POLAR element domain: {unsupported}."
            )
        methods = {
            method: pilot._evaluate_method(
                calculator=calculator,
                atoms=atoms,
                selected=selected,
                method=method,
                profile=profile,
                work_dir=work_dir,
            )
            for method, profile in pilot.METHOD_PROFILES
        }
        if (
            abs(
                cast(float, methods["ddpcm"]["gas_energy_hartree"])
                - cast(float, methods["ddcosmo"]["gas_energy_hartree"])
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
                "experimental_delta_g_kcal_mol": (
                    item.record.delta_g_kcal_mol
                ),
                "methods": methods,
            }
        )

    method_metrics = {
        method: pilot._metrics(records, method)
        for method, _profile in pilot.METHOD_PROFILES
    }
    paired_comparison = pilot._paired_method_comparison(records)
    total_wall_seconds = time.perf_counter() - wall_started
    method_timings = {
        method: pilot._timing_summary(
            [
                record["methods"][method]["timing_seconds"]["public_energy"]
                for record in records
            ]
        )
        for method, _profile in pilot.METHOD_PROFILES
    }
    checkpoint = dict(calculator.mace_polar_checkpoint_provenance)

    common = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "do_not_commit": True,
        "run_kind": RUN_KIND,
        "execution_git_head": execution_git_head,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "parent_selection_artifact": args.selection.name,
        "parent_selection_artifact_sha256": sha256_file(args.selection),
        "parent_selection_fingerprint": parent_manifest[
            "selection_fingerprint"
        ],
        "canary_prereg_artifact": args.prereg.name,
        "canary_prereg_artifact_sha256": sha256_file(args.prereg),
        "canary_selection_fingerprint": prereg["selection_fingerprint"],
        "selection_record_count": len(records),
        "selection_full_preregistered_record_count": len(full_selection),
        "selection_indices": [
            int(record["selection_index"]) for record in records
        ],
        "selection_solvents": [
            str(record["canonical_solvent"]) for record in records
        ],
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
        },
        "checkpoint": {
            "identifier": checkpoint["identifier"],
            "release_url": checkpoint["release_url"],
            "size_bytes": checkpoint["size_bytes"],
            "sha256": checkpoint["sha256"],
            "redistributed": False,
        },
        "aggregate_metrics": method_metrics,
        "paired_method_comparison": paired_comparison,
    }
    private_artifact = {
        **common,
        "visibility": "private-user-supplied-mnsol-row-level",
        "records": records,
    }
    write_json_atomic(private_output, private_artifact)

    pyddx_runtime = import_module("pyddx")
    pyscf_runtime = import_module("pyscf")
    source_hashes = pilot._source_hashes()
    source_hashes[prereg_relative_path] = sha256_file(args.prereg)
    source_hashes[
        "docs/implicit-solvation/benchmarks/"
        "run_mnsol_macepolar_solvent_diversity_canary.py"
    ] = sha256_file(Path(__file__))
    summary_artifact = {
        **common,
        "visibility": "private-aggregate-canary-do-not-commit",
        "scientific_identity": {
            "solute_energy_model": (
                "official unmodified MACE-POLAR-1-M checkpoint"
            ),
            "solute_source": (
                "MACE-POLAR coarse residual point-multipole-l<=1"
            ),
            "polarization_response": "self-consistent",
            "reaction_field_projector": "local-jet",
            "continuum_equations": ["pyddx ddPCM", "pyddx ddCOSMO"],
            "nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "energy_composition": (
                "DeltaG_solv = DeltaE_MACE_intrinsic + U_continuum "
                "+ G_SMD-CDS"
            ),
            "strict_original_smd_equivalence": False,
            "mutual_ml_continuum_polarization": True,
            "cpcm_included": False,
            "cosmo_rs_included": False,
        },
        "continuum_parameters": {
            "profiles": {
                method: profile for method, profile in pilot.METHOD_PROFILES
            },
            "coulomb_radii_policy": (
                "PySCF 2.13.1 SMD solvent-acidity-dependent"
            ),
            "dielectric_source": "PySCF 2.13.1 SMD solvent_db",
            "lmax": pilot.DDPCM_LMAX,
            "n_lebedev": pilot.DDPCM_N_LEBEDEV,
            "n_proc": 1,
            "solver_tolerance": pilot.DDPCM_SOLVER_TOLERANCE,
            "eta": pilot.DDPCM_ETA,
            "full_profile_numerics": True,
        },
        "scf_parameters": {
            "solver": pilot.SCF_SOLVER,
            "mixing": pilot.SCF_MIXING,
            "density_tolerance_e": pilot.SCF_DENSITY_TOLERANCE,
            "energy_tolerance_ev": pilot.SCF_ENERGY_TOLERANCE_EV,
            "maximum_iterations": pilot.SCF_MAX_ITERATIONS,
            "anderson_depth": pilot.SCF_ANDERSON_DEPTH,
            "anderson_regularization": pilot.SCF_ANDERSON_REGULARIZATION,
            "anderson_coefficient_l1_limit": (
                pilot.SCF_ANDERSON_COEFFICIENT_L1_LIMIT
            ),
            "anderson_step_ratio_limit": (
                pilot.SCF_ANDERSON_STEP_RATIO_LIMIT
            ),
            "anderson_residual_growth_limit": (
                pilot.SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT
            ),
            "total_charge_e": pilot.SCF_TOTAL_CHARGE_E,
            "residual_definition": (
                "unmixed neutral-tangent Pi0[M(P(c))-c]"
            ),
        },
        "timing_seconds": {
            "comparison_status": (
                "descriptive-only; fixed ddPCM-then-ddCOSMO order and "
                "process-level cache effects are not randomized"
            ),
            "method_execution_order": [
                method for method, _profile in pilot.METHOD_PROFILES
            ],
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
        "source_files_sha256": source_hashes,
        "references": {
            "mace_polar": "https://arxiv.org/abs/2602.19411",
            "mnsol_manual": (
                "https://comp.chem.umn.edu/mnsol/"
                "MNSol-v2012_Manual.pdf"
            ),
            "smd_doi": "10.1021/jp810292n",
            "ddx_documentation": "https://ddsolvation.github.io/ddX/",
        },
    }
    write_json_atomic(summary_output, summary_artifact)
    print(
        f"Wrote private {private_output} and summary {summary_output} "
        f"for {len(records)} frozen canary records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
