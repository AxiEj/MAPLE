#!/usr/bin/env python3
"""Run the preregistered ten-solvent AIMNet2 fixed-charge MNSol pilot.

The private output contains row-level MNSol values and must remain under
``.omx``.  The tracked public summary contains aggregate metrics only.  This is
an intentionally small diagnostic, not a replacement for complete MNSol
development/confirmation evaluation.
"""

from __future__ import annotations

import argparse
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
import pyddx
import pyscf
import torch

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import validate_frozen_mnsol_pilot_selection
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_PADDED_SENTINEL_TOLERANCE_E,
    AIMNET2_RAW_CHARGE_TOLERANCE_E,
    AIMNet2Calculator,
)
from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    DDPCM_ETA,
    DDPCM_LMAX,
    DDPCM_N_LEBEDEV,
    DDPCM_SOLVER_TOLERANCE,
)
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    route2_coulomb_radii,
)
from maple.function.calculator.extra_correction.implicit.source import (
    PointChargeL0Source,
    solve_fixed_charge_continuum,
)
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_PROFILE,
)
from maple.function.route2_solvents import route2_solvent_spec

ARTIFACT_NAME = "route2-mnsol-aimnet2-multisolvent-pilot-v1"
ENERGY_IDENTITY_TOLERANCE_EV = 1.0e-8
METHODS = (
    ("ddpcm", PyDDXPCMReactionFieldLinearMap),
    ("ddcosmo", PyDDXCOSMOReactionFieldLinearMap),
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
    paths = (
        "maple/function/calculator/aimnet/_aimnet2_calculator.py",
        (
            "maple/function/calculator/extra_correction/implicit/source/"
            "point_charge_l0.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "pyddx_pcm_response.py"
        ),
        ("maple/function/calculator/extra_correction/implicit/" "ddpcm_smd.py"),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "electrostatic_pairing.py"
        ),
        ("maple/function/calculator/extra_correction/implicit/" "gto_density.py"),
        ("maple/function/calculator/extra_correction/implicit/" "pyscf_smd_cds.py"),
        ("maple/function/calculator/extra_correction/implicit/" "pyscf_runtime.py"),
        ("maple/function/calculator/extra_correction/implicit/" "route2_derivative.py"),
        ("maple/function/calculator/extra_correction/implicit/" "smd_cds.py"),
        "maple/function/route2_smd_profiles.py",
        "maple/function/route2_solvents.py",
        "docs/implicit-solvation/benchmarks/benchmark_core.py",
        "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
        "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
        (
            "docs/implicit-solvation/benchmarks/"
            "run_mnsol_aimnet2_multisolvent_pilot.py"
        ),
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in paths}


def _require_private_output(path: Path) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            "Row-level MNSol pilot output must remain below the repository "
            "'.omx' directory and must not be committed."
        ) from exc
    return resolved


def _metrics(
    records: list[dict[str, object]],
    method: str,
) -> dict[str, float | int]:
    errors = np.asarray(
        [record["methods"][method]["signed_error_kcal_mol"] for record in records],
        dtype=float,
    )
    predicted = np.asarray(
        [record["methods"][method]["total_solvation_kcal_mol"] for record in records],
        dtype=float,
    )
    experimental = np.asarray(
        [record["experimental_delta_g_kcal_mol"] for record in records],
        dtype=float,
    )
    if (
        errors.size != 10
        or not np.all(np.isfinite(errors))
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(experimental))
    ):
        raise RuntimeError("MNSol pilot metrics require ten finite records.")
    return {
        "record_count": int(errors.size),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(np.abs(errors))),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(errors))),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predicted)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experimental)),
        "mean_polarization_energy_kcal_mol": float(
            np.mean(
                [
                    record["methods"][method]["polarization_energy_kcal_mol"]
                    for record in records
                ]
            )
        ),
        "mean_smd_cds_energy_kcal_mol": float(
            np.mean([record["smd_cds_energy_kcal_mol"] for record in records])
        ),
        "maximum_half_coupling_identity_error_ev": float(
            max(
                record["methods"][method]["half_coupling_identity_error_ev"]
                for record in records
            )
        ),
    }


def _timing_summary(
    values: list[float],
) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("Pilot timing samples must be finite and nonnegative.")
    return {
        "sample_count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "maximum": float(max(values)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    private_output = _require_private_output(args.private_output)
    execution_git_head = _execution_git_head()

    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )

    torch.set_num_threads(1)
    wall_started = time.perf_counter()
    load_started = time.perf_counter()
    calculator = AIMNet2Calculator(
        torch.device("cpu"),
        model="aimnet2",
        model_path=str(checkpoint),
    )
    model_load_seconds = time.perf_counter() - load_started

    records: list[dict[str, object]] = []
    for selected in selection:
        item = selected.eligible_record
        geometry = item.geometry
        record = item.record
        atoms = Atoms(
            numbers=geometry.atomic_numbers,
            positions=geometry.coordinates_angstrom,
        )
        atoms.info.update(charge=0, mult=1)
        solvent_spec = route2_solvent_spec(selected.canonical_solvent)

        charge_started = time.perf_counter()
        charge_state = calculator.charge_state(atoms)
        charge_seconds = time.perf_counter() - charge_started
        source = PointChargeL0Source(
            charges_e=charge_state.charges_e,
            declared_total_charge_e=0.0,
            source_model="aimnet2-nqe",
        )
        radii = route2_coulomb_radii(
            atoms.get_chemical_symbols(),
            solvent=selected.canonical_solvent,
            profile=DDPCM_MULTISOLVENT_SMD_PROFILE,
        )

        cds_started = time.perf_counter()
        cds = pyscf_smd_cds(
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            solvent=selected.canonical_solvent,
        )
        cds_seconds = time.perf_counter() - cds_started

        method_records: dict[str, dict[str, object]] = {}
        for method, reaction_class in METHODS:
            build_started = time.perf_counter()
            reaction_field = reaction_class(
                atoms.get_positions(),
                radii,
                dielectric=solvent_spec.descriptors.dielectric,
                lmax=DDPCM_LMAX,
                n_lebedev=DDPCM_N_LEBEDEV,
                n_proc=1,
                solver_tolerance=DDPCM_SOLVER_TOLERANCE,
                eta=DDPCM_ETA,
            )
            build_seconds = time.perf_counter() - build_started
            solve_started = time.perf_counter()
            continuum = solve_fixed_charge_continuum(
                reaction_field,
                source,
                energy_identity_tolerance_ev=(ENERGY_IDENTITY_TOLERANCE_EV),
            )
            solve_seconds = time.perf_counter() - solve_started
            polarization_kcal_mol = (
                continuum.polarization_energy_hartree * HARTREE_TO_KCAL_MOL
            )
            total_kcal_mol = polarization_kcal_mol + cds.energy_kcal_mol
            method_records[method] = {
                "polarization_energy_hartree": (continuum.polarization_energy_hartree),
                "polarization_energy_kcal_mol": polarization_kcal_mol,
                "smd_cds_energy_kcal_mol": cds.energy_kcal_mol,
                "total_solvation_kcal_mol": total_kcal_mol,
                "signed_error_kcal_mol": (total_kcal_mol - record.delta_g_kcal_mol),
                "absolute_error_kcal_mol": abs(
                    total_kcal_mol - record.delta_g_kcal_mol
                ),
                "half_coupling_identity_error_ev": (continuum.energy_identity_error_ev),
                "runtime_provenance": reaction_field.runtime_provenance,
                "timing_seconds": {
                    "build": build_seconds,
                    "solve": solve_seconds,
                },
            }

        records.append(
            {
                "canonical_solvent": selected.canonical_solvent,
                "mnsol_solvent": record.solvent,
                "partition": item.partition,
                "opaque_record_id": selected.opaque_record_id,
                "entry_number": record.entry_number,
                "geometry_handle": record.geometry_handle,
                "geometry_sha256": geometry.sha256,
                "solute_name": record.solute_name,
                "formula": record.formula,
                "atom_count": len(atoms),
                "experimental_delta_g_kcal_mol": (record.delta_g_kcal_mol),
                "aimnet2_energy_ev": charge_state.energy_ev,
                "raw_charge_residual_e": (charge_state.raw_charge_residual_e),
                "projected_charge_sum_e": (charge_state.projected_charge_sum_e),
                "cavity_radii_angstrom": radii.tolist(),
                "smd_cds_energy_hartree": cds.energy_hartree,
                "smd_cds_energy_kcal_mol": cds.energy_kcal_mol,
                "smd_cds_runtime_provenance": dict(cds.runtime_provenance),
                "methods": method_records,
                "timing_seconds": {
                    "aimnet2_charge": charge_seconds,
                    "smd_cds": cds_seconds,
                },
            }
        )

    total_wall_seconds = time.perf_counter() - wall_started
    method_metrics = {method: _metrics(records, method) for method, _ in METHODS}
    private_artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "execution_git_head": execution_git_head,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "dataset": {
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
        },
        "checkpoint": {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "redistributed": False,
        },
        "aggregate_metrics": method_metrics,
        "records": records,
    }
    write_json_atomic(private_output, private_artifact)

    shared_timings = {
        "aimnet2_charge": _timing_summary(
            [record["timing_seconds"]["aimnet2_charge"] for record in records]
        ),
        "smd_cds": _timing_summary(
            [record["timing_seconds"]["smd_cds"] for record in records]
        ),
    }
    method_timings = {
        method: {
            phase: _timing_summary(
                [
                    record["methods"][method]["timing_seconds"][phase]
                    for record in records
                ]
            )
            for phase in ("build", "solve")
        }
        for method, _ in METHODS
    }
    public_artifact = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "public-aggregate-only",
        "execution_git_head": execution_git_head,
        "scientific_identity": {
            "solute_energy_model": ("AIMNet2 gas-phase wB97M-D3 checkpoint"),
            "solute_source": "AIMNet2 NQE point-charge-l0",
            "polarization_response": "fixed",
            "continuum_equations": ["pyddx ddPCM", "pyddx ddCOSMO"],
            "nonpolar_model": "PySCF 2.13.1 SMD-CDS",
            "energy_composition": (
                "DeltaG_solv = U_polarization(fixed AIMNet2 charges) " "+ G_SMD-CDS"
            ),
            "strict_original_smd_equivalence": False,
            "mutual_ml_continuum_polarization": False,
            "cpcm_included": False,
            "cosmo_rs_included": False,
        },
        "claim_boundary": (
            "This experiment-blind ten-record MNSol pilot is an engineering "
            "and early chemical diagnostic for a fixed AIMNet2 point-charge "
            "source with ddPCM or scaled ddCOSMO plus SMD-CDS. One record per "
            "solvent cannot certify accuracy, solvent generalization, "
            "MACE-POLAR, C-PCM, COSMO-RS, forces, self-consistent solute "
            "polarization, or a solution-phase PES."
        ),
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "standard_state": protocol.standard_state,
            "temperature_k": protocol.temperature_k,
            "row_level_data_emitted": False,
        },
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_artifact": args.selection.name,
        "selection_artifact_sha256": sha256_file(args.selection),
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "selection_record_count": len(records),
        "selection_solvent_count": len(
            {record["canonical_solvent"] for record in records}
        ),
        "checkpoint": {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "redistributed": False,
        },
        "charge_output_parameters": {
            "raw_total_charge_tolerance_e": (AIMNET2_RAW_CHARGE_TOLERANCE_E),
            "padded_sentinel_tolerance_e": (AIMNET2_PADDED_SENTINEL_TOLERANCE_E),
            "projection": "uniform-affine-float-residue-only",
        },
        "continuum_parameters": {
            "shared_parameter_profile": (DDPCM_MULTISOLVENT_SMD_PROFILE),
            "shared_parameter_profile_scope": (
                "solvent descriptors, Coulomb radii, and SMD-CDS; "
                "ddCOSMO remains a separately named equation override"
            ),
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "n_proc": 1,
            "solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "eta": DDPCM_ETA,
            "full_profile_numerics": True,
            "dielectric_source": "PySCF 2.13.1 SMD solvent_db",
            "coulomb_radii_policy": ("PySCF 2.13.1 SMD solvent-acidity-dependent"),
            "standard_state_correction_kcal_mol": 0.0,
        },
        "aggregate_metrics": method_metrics,
        "timing_seconds": {
            "model_load": model_load_seconds,
            "total_wall": total_wall_seconds,
            "shared": shared_timings,
            "methods": method_timings,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "pyddx": pyddx.__version__,
            "pyscf": pyscf.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "references": {
            "mnsol_manual": (
                "https://comp.chem.umn.edu/mnsol/" "MNSol-v2012_Manual.pdf"
            ),
            "smd_doi": "10.1021/jp810292n",
            "ddx_documentation": "https://ddsolvation.github.io/ddX/",
            "aimnet2_model_guide": (
                "https://isayevlab.github.io/aimnetcentral/models/guide/"
            ),
        },
        "source_files_sha256": _source_hashes(),
    }
    write_json_atomic(args.public_output, public_artifact)
    print(args.private_output)
    print(args.public_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
