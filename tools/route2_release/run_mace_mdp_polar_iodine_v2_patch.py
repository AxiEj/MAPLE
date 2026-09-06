#!/usr/bin/env python3
"""Preregister and replay the iodine failures from canonical-ADT v1.

The frozen 505-record canonical-ADT v1 run cannot evaluate iodine because its
neutral-atom ADT asset stops before Z=53.  This narrow additive replay changes
only the ADT translation-shape registry to the independently audited
role-separated v2 registry.  It does not select records from their errors,
fit a parameter, open the confirmation partition, or reinterpret the
non-iodine predictions.

The script deliberately provides two separate commands.  ``preregister`` must
be executed first and binds the exact implementation, checkpoints, inputs,
parent failures, and role-separated iodine assets.  ``run`` then refuses any
drift and evaluates only records with the exact missing-Z=53 provider reason.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, cast


PREREG_ARTIFACT = "route2-mace-mdp-polar-iodine-v2-patch-prereg-v1"
RECORD_ARTIFACT = "route2-mace-mdp-polar-iodine-v2-patch-record-v1"
SUMMARY_ARTIFACT = "route2-mace-mdp-polar-iodine-v2-patch-summary-v1"
PARENT_PREREG_ARTIFACT = (
    "route2-mace-mdp-polar-canonical-adt-ddx-development-prereg-v1"
)
PARENT_RECORD_ARTIFACT = (
    "route2-mace-mdp-polar-canonical-adt-ddx-development-record-v1"
)
EXPECTED_INDICES = (114, 237, 292, 323, 343, 358, 370, 386, 455, 462)
EXPECTED_PARENT_FAILURE = "canonical ADT asset has no elements [53]."
ROLE_SEPARATED_PROFILE_ID = (
    "route2-research-mace-mdp-point-polar-residual-role-separated-adt-"
    "ddx-operational-v2"
)
ROLE_SEPARATED_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-role-separated-adt-ddx.impl.v2"
)
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
    "maple/solvation/solvent_terms.py",
    "tools/route2_release/run_mace_mdp_polar_iodine_v2_patch.py",
)
ASSET_FILES = (
    "docs/implicit-solvation/benchmarks/route2-adt-radial-shape-iodine-ecp-"
    "valence-v1.json",
    "docs/implicit-solvation/benchmarks/route2-adt-radial-shape-iodine-ecp-"
    "valence-v1.npz",
    "docs/implicit-solvation/benchmarks/route2-neutral-atom-penetration-"
    "gaussian-mixture-v1.json",
    "docs/implicit-solvation/benchmarks/route2-neutral-atom-penetration-"
    "gaussian-mixture-v1.npz",
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


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return payload


def _validate_parent_preregistration(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path, name="parent preregistration")
    if payload.get("artifact_id") != PARENT_PREREG_ARTIFACT:
        raise ValueError("Unknown parent preregistration artifact.")
    if payload.get("status") != (
        "locked-after-target-free-prequalification-before-canonical-adt-505"
    ):
        raise ValueError("Parent preregistration is not the frozen v1 run.")
    if payload.get("confirmation_partition_opened") is not False:
        raise ValueError("Parent preregistration opened confirmation data.")
    if payload.get("record_count") != 505:
        raise ValueError("Parent preregistration does not bind 505 records.")
    if payload.get("profile_selected_from_experimental_targets") is not False:
        raise ValueError("Parent profile was not target-free selected.")
    return payload


def _validate_parent_failures(
    records_dir: Path,
    *,
    parent_preregistration_sha256: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index in EXPECTED_INDICES:
        path = records_dir / f"index-{index:03d}.json"
        payload = _load_mapping(path, name=f"parent record {index}")
        if payload.get("artifact") != PARENT_RECORD_ARTIFACT:
            raise ValueError(f"Parent record {index} has the wrong artifact.")
        if payload.get("selection_index") != index:
            raise ValueError(f"Parent record {index} selection identity drifted.")
        if payload.get("preregistration_sha256") != parent_preregistration_sha256:
            raise ValueError(f"Parent record {index} preregistration drifted.")
        if payload.get("confirmation_partition_opened") is not False:
            raise ValueError(f"Parent record {index} opened confirmation data.")
        if payload.get("status") != "provider-failure":
            raise ValueError(f"Parent record {index} is not a provider failure.")
        if payload.get("failure_type") != "ValueError":
            raise ValueError(f"Parent record {index} failure type drifted.")
        if payload.get("failure_message") != EXPECTED_PARENT_FAILURE:
            raise ValueError(f"Parent record {index} failure reason drifted.")
        result[str(index)] = {
            "file_sha256": _sha256(path),
            "record_identity_sha256": payload.get("record_identity_sha256"),
            "geometry_sha256": payload.get("geometry_sha256"),
            "opaque_record_id": payload.get("opaque_record_id"),
            "canonical_solvent": payload.get("canonical_solvent"),
        }
    return result


def _source_hashes(source_root: Path) -> dict[str, str]:
    return {name: _sha256(source_root / name) for name in SOURCE_FILES}


def _asset_hashes(source_root: Path) -> dict[str, str]:
    return {name: _sha256(source_root / name) for name in ASSET_FILES}


def _input_hashes(input_root: Path) -> dict[str, str]:
    return {key: _sha256(input_root / name) for key, name in INPUT_FILES.items()}


def create_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    input_root = args.input_root.expanduser().resolve(strict=True)
    parent_preregistration = args.parent_preregistration.expanduser().resolve(
        strict=True
    )
    parent_records = args.parent_records.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Iodine v2 output directory must initially be empty.")
    script = Path(__file__).resolve()
    if script.parent.parent.parent != source_root:
        raise ValueError("Script must belong to the declared source root.")
    _validate_parent_preregistration(parent_preregistration)
    parent_sha = _sha256(parent_preregistration)
    failures = _validate_parent_failures(
        parent_records, parent_preregistration_sha256=parent_sha
    )
    payload: dict[str, Any] = {
        "artifact_id": PREREG_ARTIFACT,
        "schema_version": 1,
        "status": "locked-after-v1-iodine-provider-failures-before-v2-replay",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_indices": list(EXPECTED_INDICES),
        "selection_reason": "exact-v1-provider-gap-for-atomic-number-53",
        "development_targets_previously_opened": True,
        "experimental_targets_used_for_method_selection": False,
        "fitting_or_calibration_permitted": False,
        "confirmation_partition_opened": False,
        "parent_preregistration_path": str(parent_preregistration),
        "parent_preregistration_sha256": parent_sha,
        "parent_failure_records_path": str(parent_records),
        "parent_failure_records": failures,
        "source_root": str(source_root),
        "source_files_sha256": _source_hashes(source_root),
        "asset_files_sha256": _asset_hashes(source_root),
        "input_root": str(input_root),
        "input_files_sha256": _input_hashes(input_root),
        "mdp_checkpoint_sha256": _sha256(mdp_checkpoint),
        "polar_checkpoint_sha256": _sha256(polar_checkpoint),
        "output_dir": str(output_dir),
        "method": {
            "profile_id": ROLE_SEPARATED_PROFILE_ID,
            "provider_id": ROLE_SEPARATED_PROVIDER_ID,
            "change_from_parent": (
                "role-separated ADT translation-shape registry adds the "
                "audited 25-electron iodine def2-ECP valence shape"
            ),
            "permanent_source": "unchanged official MACE-MDP point q/p",
            "nonuniform_response": (
                "unchanged official MACE-POLAR-1-M zero-anchored residual"
            ),
            "continuum": "unchanged ddPCM lmax=8 nleb=1202",
            "cds": "unchanged PySCF 2.13.1 stock SMD CDS",
        },
        "claim_boundary": (
            "Development-only provider remediation for the exact iodine "
            "failures. Targets in the parent development records were already "
            "read historically but are not used to choose or parameterize the "
            "iodine asset. No confirmation, fitting, force, Hessian, virial, "
            "variational, or public-admission claim."
        ),
    }
    payload["preregistration_payload_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(output, payload)
    return payload


def _validate_preregistration(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path,
    source_root: Path,
    input_root: Path,
    parent_preregistration: Path,
    parent_records: Path,
    output_dir: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> None:
    if preregistration.get("artifact_id") != PREREG_ARTIFACT:
        raise ValueError("Unknown iodine patch preregistration.")
    if preregistration.get("status") != (
        "locked-after-v1-iodine-provider-failures-before-v2-replay"
    ):
        raise ValueError("Iodine patch preregistration is not locked.")
    expected_payload = dict(preregistration)
    digest = expected_payload.pop("preregistration_payload_sha256", None)
    if digest != _canonical_sha256(expected_payload):
        raise ValueError("Iodine patch preregistration self hash drifted.")
    if preregistration.get("selection_indices") != list(EXPECTED_INDICES):
        raise ValueError("Iodine patch selection changed.")
    if preregistration.get("experimental_targets_used_for_method_selection") is not False:
        raise ValueError("Iodine patch must remain target-independent.")
    if preregistration.get("fitting_or_calibration_permitted") is not False:
        raise ValueError("Iodine patch cannot fit or calibrate parameters.")
    if preregistration.get("confirmation_partition_opened") is not False:
        raise ValueError("Iodine patch cannot open confirmation data.")
    exact_paths = {
        "source_root": source_root,
        "input_root": input_root,
        "parent_preregistration_path": parent_preregistration,
        "parent_failure_records_path": parent_records,
        "output_dir": output_dir,
    }
    for key, path in exact_paths.items():
        if preregistration.get(key) != str(path):
            raise ValueError(f"Preregistration {key} drifted.")
    if preregistration.get("source_files_sha256") != _source_hashes(source_root):
        raise ValueError("Iodine patch implementation sources drifted.")
    if preregistration.get("asset_files_sha256") != _asset_hashes(source_root):
        raise ValueError("Iodine patch assets drifted.")
    if preregistration.get("input_files_sha256") != _input_hashes(input_root):
        raise ValueError("Iodine patch inputs drifted.")
    if preregistration.get("mdp_checkpoint_sha256") != _sha256(mdp_checkpoint):
        raise ValueError("MACE-MDP checkpoint drifted.")
    if preregistration.get("polar_checkpoint_sha256") != _sha256(polar_checkpoint):
        raise ValueError("MACE-POLAR checkpoint drifted.")
    parent_sha = _sha256(parent_preregistration)
    if preregistration.get("parent_preregistration_sha256") != parent_sha:
        raise ValueError("Parent preregistration bytes drifted.")
    failures = _validate_parent_failures(
        parent_records, parent_preregistration_sha256=parent_sha
    )
    if preregistration.get("parent_failure_records") != failures:
        raise ValueError("Parent iodine failure records drifted.")
    if _sha256(preregistration_path) == "":  # pragma: no cover - defensive
        raise RuntimeError("Unreachable empty preregistration digest.")


def _error_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    import numpy as np

    errors = np.asarray(
        [record[key]["signed_error_kcal_mol"] for record in records], dtype=float
    )
    absolute = np.abs(errors)
    return {
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(absolute)),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
    }


def run_patch(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    input_root = args.input_root.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    parent_preregistration = args.parent_preregistration.expanduser().resolve(
        strict=True
    )
    parent_records = args.parent_records.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    preregistration = _load_mapping(
        preregistration_path, name="iodine patch preregistration"
    )
    _validate_preregistration(
        preregistration,
        preregistration_path=preregistration_path,
        source_root=source_root,
        input_root=input_root,
        parent_preregistration=parent_preregistration,
        parent_records=parent_records,
        output_dir=output_dir,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(source_root / "docs/implicit-solvation/benchmarks"))

    from ase import Atoms
    import torch

    from mnsol_dataset import (  # pyright: ignore[reportMissingImports]
        load_mnsol_protocol,
        load_mnsol_v2012,
    )
    from mnsol_partition import (  # pyright: ignore[reportMissingImports]
        validate_frozen_mnsol_partition_selection,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        HARTREE_TO_KCAL_MOL,
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.api.units import HARTREE_TO_EV
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
        MACE_MDPPolarCanonicalADTDDXEnergy,
        ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
        ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID,
    )
    from maple.solvation.models import (
        MACE_MDPPermanentSourceAdapter,
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_mdp_polar_role_separated_adt_response,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.solvent_terms import PySCFSMDCDSTerm

    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(20260818)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260818)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    protocol = load_mnsol_protocol(input_root / INPUT_FILES["protocol_sha256"])
    dataset = load_mnsol_v2012(
        input_root / INPUT_FILES["dataset_zip_sha256"], protocol
    )
    selection = json.loads(
        (input_root / INPUT_FILES["development_selection_sha256"]).read_text()
    )
    pilot = json.loads(
        (input_root / INPUT_FILES["pilot_selection_sha256"]).read_text()
    )
    selected_records = validate_frozen_mnsol_partition_selection(
        selection, dataset, protocol, pilot
    )
    if len(selected_records) != 505:
        raise ValueError("Frozen development selection is not exactly 505 rows.")

    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = build_mdp_polar_role_separated_adt_response(
        mdp=cast(Any, mdp),
        base=cast(Any, MACEPolarOriginalSourceNativeFieldAdapter(radial)),
        source_root=source_root,
    )
    permanent = MACE_MDPPermanentSourceAdapter(mdp)
    prereg_sha = _sha256(preregistration_path)
    ev_to_kcal = HARTREE_TO_KCAL_MOL / HARTREE_TO_EV
    completed: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in EXPECTED_INDICES:
        output = output_dir / f"index-{index:03d}.json"
        identity = _canonical_sha256(
            {
                "contract": RECORD_ARTIFACT,
                "preregistration_sha256": prereg_sha,
                "selection_index": index,
                "parent_record_sha256": preregistration[
                    "parent_failure_records"
                ][str(index)]["file_sha256"],
            }
        )
        if output.is_file():
            existing = _load_mapping(output, name=f"iodine output {index}")
            if existing.get("record_identity_sha256") != identity:
                raise RuntimeError(f"Refusing stale iodine patch record {output}.")
            completed.append(existing)
            continue
        selected = selected_records[index]
        geometry = selected.eligible_record.geometry
        expected_parent = preregistration["parent_failure_records"][str(index)]
        if (
            geometry.sha256 != expected_parent["geometry_sha256"]
            or selected.opaque_record_id != expected_parent["opaque_record_id"]
            or selected.canonical_solvent != expected_parent["canonical_solvent"]
        ):
            raise RuntimeError(f"Frozen iodine geometry identity {index} drifted.")
        atoms = Atoms(
            numbers=geometry.atomic_numbers,
            positions=geometry.coordinates_angstrom,
            info={"charge": geometry.charge, "multiplicity": geometry.multiplicity},
        )
        experimental = float(selected.eligible_record.record.delta_g_kcal_mol)
        solvent = route2_solvent_spec(selected.canonical_solvent)
        symbols = tuple(atoms.get_chemical_symbols())
        started = time.perf_counter()
        continuum = SeparatedSourceDDXBackend(
            symbols,
            smd_coulomb_radii(symbols, solvent=solvent.name),
            continuum_model="pcm",
            dielectric=solvent.descriptors.dielectric,
            lmax=8,
            n_lebedev=1202,
            solver_tolerance=1.0e-12,
            eta=0.1,
            n_proc=1,
        )
        evaluator = MACE_MDPPolarCanonicalADTDDXEnergy(
            atoms,
            permanent=cast(Any, permanent),
            response=response,
            continuum=continuum,
        )
        if (
            evaluator.profile_id != ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
            or evaluator.provider_id != ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
        ):
            raise RuntimeError("Evaluator did not select role-separated ADT v2.")
        state = evaluator.solve(atoms)
        cds_term = PySCFSMDCDSTerm(symbols, solvent.name)
        cds_state = cds_term.evaluate(atoms, need_gradient=False)
        predicted_m0 = state.polarization_energy_ev * ev_to_kcal
        predicted_m1 = (
            state.polarization_energy_ev + cds_state.energy_eV
        ) * ev_to_kcal
        record: dict[str, Any] = {
            "artifact": RECORD_ARTIFACT,
            "status": "pass",
            "do_not_commit": True,
            "partition": "development",
            "confirmation_partition_opened": False,
            "selection_index": index,
            "opaque_record_id": selected.opaque_record_id,
            "geometry_sha256": geometry.sha256,
            "dataset_row_sha256": selected.eligible_record.record.raw_row_sha256,
            "canonical_solvent": selected.canonical_solvent,
            "atom_count": len(atoms),
            "record_identity_sha256": identity,
            "preregistration_sha256": prereg_sha,
            "parent_record_sha256": expected_parent["file_sha256"],
            "profile_id": evaluator.profile_id,
            "provider_id": evaluator.provider_id,
            "response_configuration_sha256": response.configuration_sha256(),
            "evaluator_configuration_sha256": evaluator.configuration_sha256(),
            "root_sha256": state.root_sha256,
            "continuum_state_sha256": state.continuum_state_sha256,
            "root_residual_eV": state.primal_residual_ev,
            "experimental_delta_g_kcal_mol": experimental,
            "continuum_polarization_kcal_mol": predicted_m0,
            "smd_cds_configuration_sha256": cds_term.configuration_sha256(),
            "smd_cds_state_sha256": cds_state.state_sha256,
            "smd_cds_kcal_mol": cds_state.energy_eV * ev_to_kcal,
            "m0_electrostatic_only": {
                "predicted_delta_g_kcal_mol": predicted_m0,
                "signed_error_kcal_mol": predicted_m0 - experimental,
                "absolute_error_kcal_mol": abs(predicted_m0 - experimental),
            },
            "m1_electrostatic_plus_stock_smd_cds": {
                "predicted_delta_g_kcal_mol": predicted_m1,
                "signed_error_kcal_mol": predicted_m1 - experimental,
                "absolute_error_kcal_mol": abs(predicted_m1 - experimental),
            },
            "solve_wall_seconds": time.perf_counter() - started,
            "claim_boundary": preregistration["claim_boundary"],
        }
        _write_json_atomic(output, record)
        completed.append(record)
        print(
            f"index={index} status=pass m1_abs_err="
            f"{record['m1_electrostatic_plus_stock_smd_cds']['absolute_error_kcal_mol']:.6f}",
            flush=True,
        )

    summary: dict[str, Any] = {
        "artifact": SUMMARY_ARTIFACT,
        "status": "pass",
        "selection_indices": list(EXPECTED_INDICES),
        "record_count": len(completed),
        "preregistration_sha256": prereg_sha,
        "profile_id": ROLE_SEPARATED_PROFILE_ID,
        "provider_id": ROLE_SEPARATED_PROVIDER_ID,
        "m0_metrics": _error_metrics(completed, "m0_electrostatic_only"),
        "m1_metrics": _error_metrics(
            completed, "m1_electrostatic_plus_stock_smd_cds"
        ),
        "confirmation_partition_opened": False,
        "fitting_or_calibration_performed": False,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-preregistration", type=Path, required=True)
    parser.add_argument("--parent-records", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    _common_parser(preregister)
    preregister.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    _common_parser(execute)
    execute.add_argument("--preregistration", type=Path, required=True)
    execute.add_argument("--polar-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "preregister":
        payload = create_preregistration(args)
    else:
        payload = run_patch(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
