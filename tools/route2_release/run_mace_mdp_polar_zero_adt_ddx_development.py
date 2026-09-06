"""Run target-unused predictions for the frozen zero-training v3 hybrid."""

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

from ase import Atoms
import torch

REPO_ROOT = Path(
    os.environ.get("MAPLE_ROUTE2_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
BENCHMARK_ROOT = REPO_ROOT / "docs/implicit-solvation/benchmarks"
MDP_CHECKPOINT = Path("/home/axie/.cache/mace/MACE-MDP.model")
POLAR_CHECKPOINT = Path("/home/axie/.cache/mace/MACEPOLAR1Mmodel")
EXPECTED_RECORD_COUNT = 505
EXPECTED_PREREGISTRATION_ID = "route2-mace-mdp-polar-zero-adt-ddx-development-prereg-v1"
RECORD_ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-ddx-development-record-v1"
SHARD_ARTIFACT_ID = "route2-mace-mdp-polar-zero-adt-ddx-development-shard-v1"
DDX_LMAX = 12
DDX_N_LEBEDEV = 1202
RANDOM_SEED = 20260818
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
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/route2_solvents.py",
    "maple/solvation/api/profiles.py",
    "maple/solvation/continuum/separated_source_adt_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/adt_radial_shape.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/coupling/neutral_atom_penetration.py",
    "maple/solvation/experimental/mace_mdp_polar_adt_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/uniform_response_manifold.py",
    "maple/solvation/release/uniform_susceptibility_replacement.py",
    "maple/solvation/solvent_terms.py",
    "tools/route2_release/aggregate_mace_mdp_polar_zero_adt_ddx_development.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_ddx_development_preregistration.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_coupled_screen_preregistration.py",
    "tools/route2_release/create_mace_mdp_polar_zero_adt_stage_b_preregistration.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_ddx_development.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_coupled_screen.py",
    "tools/route2_release/run_mace_mdp_polar_zero_adt_stage_b.py",
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
    smd_coulomb_radii,
)
from maple.function.route2_solvents import route2_solvent_spec  # noqa: E402
from maple.solvation.api.profiles import (  # noqa: E402
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.units import HARTREE_TO_EV  # noqa: E402
from maple.solvation.continuum import SeparatedSourceDDXBackend  # noqa: E402
from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (  # noqa: E402
    MACE_MDPPolarCanonicalADTDDXEnergy,
    POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
)
from maple.solvation.models import (  # noqa: E402
    MACEPolarOriginalSourceNativeFieldAdapter,
    MACEPolarZeroFieldPointPermanentSource,
    build_mace_mdp_moment_adapter,
    build_mdp_polar_role_separated_adt_response,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.solvent_terms import PySCFSMDCDSTerm  # noqa: E402


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


def _write_json_exclusive(path: Path, payload: object) -> None:
    """Publish one immutable prediction artifact without silent replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=MDP_CHECKPOINT)
    parser.add_argument("--polar-checkpoint", type=Path, default=POLAR_CHECKPOINT)
    parser.add_argument(
        "--polar-device",
        choices=("cpu", "cuda"),
        default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cuda"),
    )
    return parser.parse_args()


def _validate_preregistration(
    prereg: Mapping[str, Any],
    *,
    input_root: Path,
    preregistration_path: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
    polar_device: str,
) -> None:
    payload = dict(prereg)
    prereg_digest = payload.pop("preregistration_sha256", None)
    if prereg_digest != _canonical_sha256(payload):
        raise ValueError("Preregistration self hash drifted.")
    if prereg.get("artifact_id") != EXPECTED_PREREGISTRATION_ID:
        raise ValueError("Unknown preregistration identity.")
    if prereg.get("status") != "locked-after-stage-b-pass-before-zero-adt-505":
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
    if prereg.get("development_targets_previously_opened") is not True:
        raise ValueError("Development-target history must be disclosed.")
    if prereg.get("profile_selected_from_experimental_targets") is not False:
        raise ValueError("Zero-ADT v3 profile must be selected target-free.")
    if (
        prereg.get("prediction_runner_uses_experimental_targets") is not False
        or prereg.get("prediction_runner_emits_experimental_targets") is not False
    ):
        raise ValueError("Prediction target-use boundary drifted.")
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
        "profile_id": POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
        "scalar": (
            "G_ddX,pol(MACE-POLAR zero-field permanent point l<=1 + "
            "MACE-POLAR nonuniform Gaussian residual l<=1 with its uniform "
            "tangent removed + canonical MDP-alpha ADT dipoles) + PySCF "
            "stock SMD CDS"
        ),
        "permanent_source": "official frozen MACE-POLAR zero-field point q/p",
        "nonuniform_response": (
            "official frozen MACE-POLAR-1-M zero-anchored residual response "
            "minus its exact zero-field uniform tangent"
        ),
        "uniform_response": (
            "canonical ADT lift of the official frozen MACE-MDP molecular "
            "polarizability"
        ),
        "mace_polar_long_range_evaluator": MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
        "continuum_model": "pcm",
        "lmax": DDX_LMAX,
        "n_lebedev": DDX_N_LEBEDEV,
        "solver_tolerance": 1.0e-12,
        "eta": 0.1,
        "n_proc": 1,
        "mdp_device": "cpu",
        "polar_device": "cuda",
        "standard_state": "MNSol protocol gas-to-solution 1M convention",
        "cds": "unmodified PySCF 2.13.1 stock SMD CDS",
        "fitting_or_calibration": False,
    }
    if method != expected_method:
        raise ValueError("Preregistered hybrid method drifted.")
    if polar_device != method["polar_device"]:
        raise ValueError("Runtime MACE-POLAR device differs from preregistration.")
    stage_b_prereg_path = Path(
        str(prereg.get("stage_b_preregistration_path", ""))
    ).expanduser()
    stage_b_aggregate_path = Path(
        str(prereg.get("stage_b_aggregate_path", ""))
    ).expanduser()
    if (
        not stage_b_prereg_path.is_absolute()
        or not stage_b_aggregate_path.is_absolute()
    ):
        raise ValueError("Stage-B evidence paths must be absolute.")
    stage_b_prereg_path = stage_b_prereg_path.resolve(strict=True)
    stage_b_aggregate_path = stage_b_aggregate_path.resolve(strict=True)
    if _sha256(stage_b_prereg_path) != prereg.get(
        "stage_b_preregistration_file_sha256"
    ) or _sha256(stage_b_aggregate_path) != prereg.get("stage_b_aggregate_file_sha256"):
        raise ValueError("Stage-B evidence bytes drifted.")
    stage_b_prereg = json.loads(stage_b_prereg_path.read_text())
    stage_b_prereg_digest = stage_b_prereg.pop("preregistration_sha256", None)
    if stage_b_prereg_digest != _canonical_sha256(
        stage_b_prereg
    ) or stage_b_prereg_digest != prereg.get("stage_b_preregistration_artifact_sha256"):
        raise ValueError("Stage-B preregistration identity drifted.")
    stage_b_aggregate = json.loads(stage_b_aggregate_path.read_text())
    aggregate_digest = stage_b_aggregate.pop("aggregate_sha256", None)
    if (
        aggregate_digest != _canonical_sha256(stage_b_aggregate)
        or aggregate_digest != prereg.get("stage_b_aggregate_artifact_sha256")
        or stage_b_aggregate.get("status") != "pass-stage-b"
        or stage_b_aggregate.get("stage_b_passed") is not True
        or stage_b_aggregate.get("pass_count") != 12
        or stage_b_aggregate.get("failure_count") != 0
        or stage_b_aggregate.get("record_count") != 12
        or stage_b_aggregate.get("preregistration_artifact_sha256")
        != stage_b_prereg_digest
    ):
        raise ValueError("Stage-B pass identity drifted.")


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
            "contract": RECORD_ARTIFACT_ID,
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
        polar_device=args.polar_device,
    )
    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The preregistered CUDA MACE-POLAR path is unavailable.")
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

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_started = time.perf_counter()
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    response = build_mdp_polar_role_separated_adt_response(
        mdp=mdp,
        base=MACEPolarOriginalSourceNativeFieldAdapter(radial),
        source_root=REPO_ROOT,
    )
    permanent = MACEPolarZeroFieldPointPermanentSource(response)
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
            existing_digest = existing.pop("record_sha256", None)
            existing_valid = existing_digest == _canonical_sha256(existing)
            existing["record_sha256"] = existing_digest
            if (
                existing_valid
                and existing.get("record_identity_sha256") == record_identity
            ):
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
            info={
                "charge": geometry.charge,
                "multiplicity": geometry.multiplicity,
            },
        )
        record: dict[str, Any] = {
            "artifact": RECORD_ARTIFACT_ID,
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
            "response_configuration_sha256": response.configuration_sha256(),
            "profile_id": POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
            "experimental_targets_parsed_by_selection_loader": True,
            "experimental_targets_used_by_prediction": False,
            "experimental_targets_emitted": False,
            "claim_boundary": (
                "Frozen MNSol development-only zero-training v3 M0/M1 prediction. "
                "The historical selection loader parses the source table, but no "
                "target enters or is emitted by prediction. Comparison is deferred "
                "until exact 505-record closure. No fitting/calibration, confirmation "
                "access, force, virial, Hessian, or production admission."
            ),
        }
        solve_started = time.perf_counter()
        try:
            symbols = tuple(atoms.get_chemical_symbols())
            solvent = route2_solvent_spec(selected.canonical_solvent)
            continuum = SeparatedSourceDDXBackend(
                symbols,
                smd_coulomb_radii(symbols, solvent=solvent.name),
                continuum_model="pcm",
                dielectric=solvent.descriptors.dielectric,
                lmax=DDX_LMAX,
                n_lebedev=DDX_N_LEBEDEV,
                solver_tolerance=1.0e-12,
                eta=0.1,
                n_proc=1,
            )
            evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
                atoms,
                permanent=permanent,
                response=response,
                continuum=continuum,
            )
            if (
                evaluator.profile_id
                != POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
            ):
                raise RuntimeError("Prediction evaluator instantiated another profile.")
            state = evaluator.solve(atoms)
            cds_term = PySCFSMDCDSTerm(symbols, solvent.name)
            cds_state = cds_term.evaluate(atoms, need_gradient=False)
            predicted_m0 = state.polarization_energy_ev * ev_to_kcal
            predicted_m1 = (
                state.polarization_energy_ev + cds_state.energy_eV
            ) * ev_to_kcal
            record.update(
                {
                    "status": "pass",
                    "evaluator_configuration_sha256": (
                        evaluator.configuration_sha256()
                    ),
                    "root_sha256": state.root_sha256,
                    "continuum_state_sha256": state.continuum_state_sha256,
                    "root_residual_eV": state.primal_residual_ev,
                    "root_start_iterations": {
                        item.label: item.iterations for item in state.root_starts
                    },
                    "maximum_root_iterations": max(
                        item.iterations for item in state.root_starts
                    ),
                    "vacuum_energy_eV": state.vacuum_energy_ev,
                    "continuum_polarization_kcal_mol": (
                        state.polarization_energy_ev * ev_to_kcal
                    ),
                    "smd_cds_configuration_sha256": (cds_term.configuration_sha256()),
                    "smd_cds_state_sha256": cds_state.state_sha256,
                    "smd_cds_kcal_mol": cds_state.energy_eV * ev_to_kcal,
                    "m0_electrostatic_only": {
                        "predicted_delta_g_kcal_mol": predicted_m0,
                    },
                    "m1_electrostatic_plus_stock_smd_cds": {
                        "predicted_delta_g_kcal_mol": predicted_m1,
                    },
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
        record["record_sha256"] = _canonical_sha256(record)
        _write_json_exclusive(output, record)
        completed.append(record)
        print(
            f"[{ordinal}/{args.stop - args.start}] index={selection_index} "
            f'solvent={selected.canonical_solvent} status={record["status"]} '
            f'm0_prediction={record.get("m0_electrostatic_only", {}).get("predicted_delta_g_kcal_mol", float("nan")):.6f} '
            f'm1_prediction={record.get("m1_electrostatic_plus_stock_smd_cds", {}).get("predicted_delta_g_kcal_mol", float("nan")):.6f} '
            f'wall={record["solve_wall_seconds"]:.2f}s',
            flush=True,
        )

    successes = [record for record in completed if record["status"] == "pass"]
    summary = {
        "artifact": SHARD_ARTIFACT_ID,
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
        "profile_id": POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
        "response_configuration_sha256": response.configuration_sha256(),
        "model_load_seconds": model_load_seconds,
        "experimental_targets_parsed_by_selection_loader": True,
        "experimental_targets_used_by_prediction": False,
        "experimental_targets_emitted": False,
        "record_sha256s": [record["record_sha256"] for record in completed],
        "wall_seconds": time.perf_counter() - shard_started,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    _write_json_exclusive(
        output_dir / f"shard-summary-{args.start:03d}-{args.stop:03d}.json",
        summary,
    )
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
