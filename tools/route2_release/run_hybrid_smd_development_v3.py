from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from ase import Atoms
import torch

REPO_ROOT = Path(
    os.environ.get("MAPLE_ROUTE2_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
BENCHMARK_ROOT = REPO_ROOT / "docs/implicit-solvation/benchmarks"
MDP_CHECKPOINT = Path("/home/axie/.cache/mace/MACE-MDP.model")
POLAR_CHECKPOINT = Path("/home/axie/.cache/mace/MACEPOLAR1Mmodel")
EXPECTED_RECORD_COUNT = 505
INPUT_FILES = {
    "dataset_zip_sha256": "MNSolDatabase_v2012.zip",
    "development_selection_sha256": (
        "route2-mnsol-development-selection-v1.private.json"
    ),
    "protocol_sha256": "route2-mnsol-protocol-v1.json",
    "pilot_selection_sha256": "route2-mnsol-pilot-selection-v1.json",
}
SOURCE_FILES = (
    "maple/function/calculator/extra_correction/implicit/pyscf_smd_cds.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_ddx.py",
    "maple/solvation/experimental/mace_mdp_polar_solvated_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/solvent_terms.py",
)

if not REPO_ROOT.is_dir():
    raise RuntimeError(
        "MAPLE_ROUTE2_SOURCE_ROOT does not identify the isolated source snapshot."
    )
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(BENCHMARK_ROOT))

from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012  # noqa: E402
from mnsol_partition import validate_frozen_mnsol_partition_selection  # noqa: E402

from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    HARTREE_TO_KCAL_MOL,
)
from maple.solvation.api.profiles import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.units import HARTREE_TO_EV  # noqa: E402
from maple.solvation.experimental import (  # noqa: E402
    build_smd_mace_mdp_polar_hybrid_ddx_pes,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=MDP_CHECKPOINT)
    parser.add_argument("--polar-checkpoint", type=Path, default=POLAR_CHECKPOINT)
    return parser.parse_args()


def _validate_preregistration(
    prereg: Mapping[str, Any],
    *,
    input_root: Path,
    preregistration_path: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> None:
    if prereg.get("artifact_id") != "route2-hybrid-smd-development-prereg-v3":
        raise ValueError("Unknown preregistration identity.")
    if prereg.get("status") != "locked-before-first-v3-hybrid-evaluation":
        raise ValueError("Preregistration is not locked.")
    if prereg.get("partition") != "development":
        raise ValueError("Only the development partition is permitted.")
    if prereg.get("confirmation_partition_opened") is not False:
        raise ValueError("The confirmation partition must remain sealed.")
    if prereg.get("record_count") != EXPECTED_RECORD_COUNT:
        raise ValueError("Preregistration must bind exactly 505 records.")
    target = prereg.get("hard_accuracy_target")
    if not isinstance(target, Mapping) or target != {
        "comparison": "<=",
        "metric": "mean_absolute_error_kcal_mol",
        "threshold_kcal_mol": 1.5,
    }:
        raise ValueError("Hard accuracy target drifted.")
    if prereg.get("fitting_or_calibration_permitted") is not False:
        raise ValueError("Fitting or calibration is forbidden.")
    expected_hashes = {
        key: _sha256(input_root / relative) for key, relative in INPUT_FILES.items()
    }
    expected_hashes.update(
        {
            "runner_sha256": _sha256(Path(__file__)),
            "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
            "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        }
    )
    for key, expected in expected_hashes.items():
        if prereg.get(key) != expected:
            raise ValueError(f"Preregistration {key} drifted.")
    source_hashes = {name: _sha256(REPO_ROOT / name) for name in SOURCE_FILES}
    if prereg.get("source_files_sha256") != source_hashes:
        raise ValueError("Preregistered implementation sources drifted.")
    if prereg.get("preregistration_path") != str(preregistration_path):
        raise ValueError("Preregistration path drifted.")
    method = prereg.get("method")
    expected_method = {
        "scalar": "G_ddX,pol(MACE-MDP permanent point l<=1 + MACE-POLAR induced Gaussian l<=1) + PySCF SMD CDS",
        "permanent_source": "official frozen MACE-MDP q/p checkpoint",
        "induced_response": "official frozen MACE-POLAR-1-M zero-anchored response",
        "mace_polar_long_range_evaluator": MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
        "continuum_model": "pcm",
        "lmax": 15,
        "n_lebedev": 1202,
        "solver_tolerance": 1.0e-12,
        "eta": 0.1,
        "n_proc": 1,
        "device": "cpu",
        "standard_state": "MNSol protocol gas-to-solution 1M convention",
    }
    if method != expected_method:
        raise ValueError("Preregistered hybrid method drifted.")


def _record_identity(
    *,
    prereg_sha256: str,
    runner_sha256: str,
    mdp_checkpoint_sha256: str,
    polar_checkpoint_sha256: str,
    selection_index: int,
) -> str:
    return _canonical_sha256(
        {
            "contract": "route2-hybrid-smd-development-record-v3",
            "preregistration_sha256": prereg_sha256,
            "runner_sha256": runner_sha256,
            "mdp_checkpoint_sha256": mdp_checkpoint_sha256,
            "polar_checkpoint_sha256": polar_checkpoint_sha256,
            "selection_index": selection_index,
        }
    )


def main() -> int:
    args = _parse_args()
    if not 0 <= args.start < args.stop <= EXPECTED_RECORD_COUNT:
        raise ValueError("--start/--stop must define a nonempty subset of 0..505.")
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    if output_dir.is_relative_to(REPO_ROOT) or output_dir.is_relative_to(input_root):
        raise ValueError("Output records must be outside source and input roots.")
    for relative in INPUT_FILES.values():
        (input_root / relative).resolve(strict=True)
    prereg = json.loads(preregistration_path.read_text())
    _validate_preregistration(
        prereg,
        input_root=input_root,
        preregistration_path=preregistration_path,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    prereg_sha256 = _sha256(preregistration_path)
    runner_sha256 = _sha256(Path(__file__))
    mdp_checkpoint_sha256 = _sha256(mdp_checkpoint)
    polar_checkpoint_sha256 = _sha256(polar_checkpoint)

    protocol = load_mnsol_protocol(input_root / INPUT_FILES["protocol_sha256"])
    dataset = load_mnsol_v2012(input_root / INPUT_FILES["dataset_zip_sha256"], protocol)
    selection_manifest = json.loads(
        (input_root / INPUT_FILES["development_selection_sha256"]).read_text()
    )
    pilot = json.loads((input_root / INPUT_FILES["pilot_selection_sha256"]).read_text())
    selected_records = validate_frozen_mnsol_partition_selection(
        selection_manifest,
        dataset,
        protocol,
        pilot,
    )
    if len(selected_records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Frozen development selection does not contain 505 rows.")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_started = time.perf_counter()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device="cpu",
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    model_load_seconds = time.perf_counter() - model_started
    ev_to_kcal = HARTREE_TO_KCAL_MOL / HARTREE_TO_EV

    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    shard_started = time.perf_counter()
    for ordinal, selection_index in enumerate(range(args.start, args.stop), start=1):
        output = output_dir / f"index-{selection_index:03d}.json"
        record_identity = _record_identity(
            prereg_sha256=prereg_sha256,
            runner_sha256=runner_sha256,
            mdp_checkpoint_sha256=mdp_checkpoint_sha256,
            polar_checkpoint_sha256=polar_checkpoint_sha256,
            selection_index=selection_index,
        )
        if output.is_file():
            existing = json.loads(output.read_text())
            if existing.get("record_identity_sha256") == record_identity:
                completed.append(existing)
                print(
                    f"[{ordinal}/{args.stop - args.start}] index={selection_index} "
                    f"resume={existing.get('status')}",
                    flush=True,
                )
                continue
            raise RuntimeError(f"Refusing stale hybrid record: {output}")

        selected = selected_records[selection_index]
        geometry = selected.eligible_record.geometry
        atoms = Atoms(
            numbers=geometry.atomic_numbers,
            positions=geometry.coordinates_angstrom,
            info={"charge": geometry.charge, "mult": geometry.multiplicity},
        )
        experimental = float(selected.eligible_record.record.delta_g_kcal_mol)
        record: dict[str, Any] = {
            "artifact": "route2-hybrid-smd-development-record-v3",
            "do_not_commit": True,
            "partition": "development",
            "confirmation_partition_opened": False,
            "selection_index": selection_index,
            "opaque_record_id": selected.opaque_record_id,
            "geometry_sha256": geometry.sha256,
            "dataset_row_sha256": selected.eligible_record.record.raw_row_sha256,
            "selection_score_sha256": selected.selection_score_sha256,
            "canonical_solvent": selected.canonical_solvent,
            "atom_count": len(atoms),
            "record_identity_sha256": record_identity,
            "preregistration_sha256": prereg_sha256,
            "runner_sha256": runner_sha256,
            "mdp_checkpoint_sha256": mdp_checkpoint_sha256,
            "polar_checkpoint_sha256": polar_checkpoint_sha256,
            "hybrid_configuration_sha256": hybrid.configuration_sha256(),
            "experimental_delta_g_kcal_mol": experimental,
            "claim_boundary": (
                "Frozen MNSol development-only hybrid full-solvation energy; "
                "no fitting/calibration, sealed confirmation, force/virial/Hessian "
                "accuracy, or production admission."
            ),
        }
        solve_started = time.perf_counter()
        try:
            pes = build_smd_mace_mdp_polar_hybrid_ddx_pes(
                hybrid,
                tuple(atoms.get_chemical_symbols()),
                solvent=selected.canonical_solvent,
                continuum_model="pcm",
                lmax=15,
                n_lebedev=1202,
                solver_tolerance=1.0e-12,
                eta=0.1,
                n_proc=1,
            )
            state = pes.solve(atoms)
            predicted = state.solvation_energy_eV * ev_to_kcal
            signed_error = predicted - experimental
            record.update(
                {
                    "status": "pass",
                    "pes_configuration_sha256": pes.configuration_sha256(),
                    "state_sha256": state.state_sha256,
                    "electrostatic_root_sha256": state.electrostatic_state.root_sha256,
                    "continuum_state_sha256": (
                        state.electrostatic_state.continuum_state_sha256
                    ),
                    "root_residual_eV": state.electrostatic_state.primal_residual_ev,
                    "cold_iterations": state.electrostatic_state.cold_iterations,
                    "wide_iterations": state.electrostatic_state.wide_iterations,
                    "vacuum_energy_eV": state.vacuum_energy_eV,
                    "continuum_polarization_kcal_mol": (
                        state.polarization_energy_eV * ev_to_kcal
                    ),
                    "smd_cds_kcal_mol": state.cds_energy_eV * ev_to_kcal,
                    "predicted_delta_g_kcal_mol": predicted,
                    "signed_error_kcal_mol": signed_error,
                    "absolute_error_kcal_mol": abs(signed_error),
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "provider-failure",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                }
            )
        record["solve_wall_seconds"] = time.perf_counter() - solve_started
        _write_json_atomic(output, record)
        completed.append(record)
        print(
            f"[{ordinal}/{args.stop - args.start}] index={selection_index} "
            f'solvent={selected.canonical_solvent} status={record["status"]} '
            f'abs_err={record.get("absolute_error_kcal_mol", float("nan")):.6f} '
            f'wall={record["solve_wall_seconds"]:.2f}s',
            flush=True,
        )

    successes = [record for record in completed if record["status"] == "pass"]
    errors = np.asarray(
        [record["absolute_error_kcal_mol"] for record in successes],
        dtype=float,
    )
    summary = {
        "artifact": "route2-hybrid-smd-development-shard-v3",
        "status": "pass" if len(successes) == len(completed) else "provider-failure",
        "do_not_commit": True,
        "partition": "development",
        "confirmation_partition_opened": False,
        "start": args.start,
        "stop": args.stop,
        "record_count": len(completed),
        "success_count": len(successes),
        "failure_count": len(completed) - len(successes),
        "selection_indices": [record["selection_index"] for record in completed],
        "preregistration_sha256": prereg_sha256,
        "runner_sha256": runner_sha256,
        "mdp_checkpoint_sha256": mdp_checkpoint_sha256,
        "polar_checkpoint_sha256": polar_checkpoint_sha256,
        "hybrid_configuration_sha256": hybrid.configuration_sha256(),
        "model_load_seconds": model_load_seconds,
        "mean_absolute_error_kcal_mol": (
            float(np.mean(errors)) if len(errors) else None
        ),
        "root_mean_square_error_kcal_mol": (
            float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None
        ),
        "maximum_absolute_error_kcal_mol": (
            float(np.max(errors)) if len(errors) else None
        ),
        "wall_seconds": time.perf_counter() - shard_started,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    _write_json_atomic(
        output_dir / f"shard-summary-{args.start:03d}-{args.stop:03d}.json",
        summary,
    )
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
