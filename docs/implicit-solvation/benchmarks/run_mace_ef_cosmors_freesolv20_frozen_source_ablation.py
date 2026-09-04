#!/usr/bin/env python3
"""Generate a byte-bound frozen-zero-field control for FreeSolv-20 v3."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (  # noqa: E402
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from maple.function.cosmors_torch.cosmospace import (  # noqa: E402
    OPEN_COSMORS_24A_PARAMETERS,
)
from maple.function.cosmors_torch.segment_cosmo import (  # noqa: E402
    TorchSegmentCOSMO,
    TorchSegmentCOSMOConfig,
)
from maple.function.cosmors_torch.surface import (  # noqa: E402
    build_sigma_profile,
    write_sigma_profile,
)
from maple.function.cosmors_torch.thermodynamics import (  # noqa: E402
    open24a_solvation_free_energy,
)
from maple.function.mlip_cosmo_rs import open_cosmors_24a_cavity_radii  # noqa: E402
from run_mace_ef_cosmors_freesolv20 import (  # noqa: E402
    _bind_source_files,
    _canonical_sha256,
    _git_blob_sha256,
    _repository_relative,
    _resolve_artifact_path,
    _runtime_versions,
    _sha256_file,
    _write_json_atomic,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v3.json"
OUTPUT_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-frozen-source-ablation-v3.json"
WORK_DIR = REPOSITORY_ROOT / ".omx/benchmarks/mace-ef-cosmors-freesolv20-frozen-v3"
EPSILON_COEFFICIENT = 1.0e-30
HARTREE_TO_KCAL_MOL = 627.5094740631


def _assert_source_compatible(artifact: dict[str, object], commit: str) -> None:
    hashes = artifact.get("source_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Input artifact lacks committed source-file binding.")
    for relative, expected in hashes.items():
        if _git_blob_sha256(commit, str(relative)) != expected:
            raise RuntimeError(
                f"Input artifact source {relative!r} is incompatible with {commit}."
            )


def _prediction_summary(
    records: list[dict[str, object]], key: str
) -> dict[str, object]:
    if not records:
        return {
            "count": 0,
            "mean_signed_error_kcal_mol": None,
            "mae_kcal_mol": None,
            "rmse_kcal_mol": None,
            "maximum_absolute_error_kcal_mol": None,
            "within_1_kcal_mol_count": 0,
            "within_2_kcal_mol_count": 0,
            "worst_record": None,
        }
    errors = np.asarray(
        [float(record[key]["signed_error_kcal_mol"]) for record in records]
    )
    absolute = np.abs(errors)
    worst = int(np.argmax(absolute))
    return {
        "count": len(records),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mae_kcal_mol": float(np.mean(absolute)),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "within_1_kcal_mol_count": int(np.sum(absolute <= 1.0)),
        "within_2_kcal_mol_count": int(np.sum(absolute <= 2.0)),
        "worst_record": {
            "compound_id": records[worst]["compound_id"],
            "name": records[worst]["name"],
            "signed_error_kcal_mol": float(errors[worst]),
        },
    }


def _frozen_profile(
    xyz_path: Path,
    *,
    name: str,
    checkpoint: Path,
    device: str,
    angular_degree: int,
):
    atoms = read(xyz_path, index=0)
    atomic_numbers = tuple(int(value) for value in atoms.numbers)
    symbols = tuple(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.positions, dtype=np.float64)
    electronic = MACEPolarEFEnergyModel(
        MACEPolarEFConfig(
            checkpoint_path=str(checkpoint),
            atomic_numbers=atomic_numbers,
            total_charge=0,
            spin_multiplicity=1,
            device=device,
        )
    )
    gas = electronic.evaluate(
        positions,
        np.zeros((len(atomic_numbers), 4), dtype=np.float64),
    )
    continuum = TorchSegmentCOSMO(
        TorchSegmentCOSMOConfig(
            atomic_numbers=atomic_numbers,
            radii_angstrom=tuple(open_cosmors_24a_cavity_radii(symbols)),
            angular_degree=angular_degree,
        )
    )
    surface = continuum.surface(
        positions,
        np.asarray(gas.conjugate_source_raw, dtype=np.float64),
        name=name,
    )
    profile = replace(
        build_sigma_profile(surface),
        dielectric_energy_role="total-solvated-minus-gas",
        molecular_charge_e=surface.molecular_charge_e.new_tensor(0.0),
    )
    source = np.asarray(gas.conjugate_source_raw, dtype=float)
    return profile, {
        "source_charge_e": float(np.sum(source[:, 0])),
        "source_maximum_absolute_component": float(np.max(np.abs(source))),
        "surface_area_angstrom2": float(surface.cavity_area_angstrom2.detach()),
        "surface_segment_count": int(surface.segment_areas_angstrom2.numel()),
        "dielectric_kcal_mol": float(surface.dielectric_energy_hartree.detach())
        * HARTREE_TO_KCAL_MOL,
    }


def main() -> int:
    import torch

    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    if primary.get("artifact") != "mace-ef-cosmors-freesolv20-diverse-v3":
        raise ValueError("Frozen-source v2 requires the byte-bound v3 primary.")
    if primary.get("status") != "complete":
        raise ValueError("Frozen-source v2 requires a complete primary artifact.")
    execution_git_head, source_files_sha256 = _bind_source_files((Path(__file__),))
    _assert_source_compatible(primary, execution_git_head)

    checkpoint = Path(primary["model"]["checkpoint_path"])
    if _sha256_file(checkpoint) != primary["model"]["checkpoint_sha256"]:
        raise RuntimeError("MACE-EF checkpoint bytes drifted from the primary.")
    device = str(primary["model"]["device"])
    angular_degree = int(primary["model"]["angular_degree"])
    maximum_iterations = int(primary["model"]["cosmospace_maximum_iterations"])
    full_parameters = replace(
        OPEN_COSMORS_24A_PARAMETERS,
        maximum_iterations=maximum_iterations,
    )
    no_hb_parameters = replace(
        OPEN_COSMORS_24A_PARAMETERS,
        name="open24a-ablation-no-hydrogen-bond",
        hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
        maximum_iterations=maximum_iterations,
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    profile_dir = WORK_DIR / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    water_xyz = _resolve_artifact_path(primary["water_profile"]["xyz_path"])
    if _sha256_file(water_xyz) != primary["water_profile"]["xyz_sha256"]:
        raise RuntimeError("Primary water geometry drifted before frozen replay.")
    water, water_record = _frozen_profile(
        water_xyz,
        name="water-frozen-zero-field",
        checkpoint=checkpoint,
        device=device,
        angular_degree=angular_degree,
    )
    water_profile_path = WORK_DIR / "water.frozen.torch-cosmors.json"
    write_sigma_profile(water, water_profile_path)
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-freesolv20-frozen-source-ablation-v3",
        "status": "running",
        "scientific_status": "running-mechanism-diagnostic",
        "admission_eligible": False,
        "diagnostic_only": True,
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "execution_git_head": execution_git_head,
        "source_files_sha256": source_files_sha256,
        "primary_artifact": {
            "path": _repository_relative(PRIMARY_PATH),
            "sha256": _sha256_file(PRIMARY_PATH),
            "artifact": primary["artifact"],
        },
        "runtime": _runtime_versions(),
        "method": {
            "description": "Evaluate the same checkpoint once at zero field, freeze its energy-conjugate atomwise l<=1 source, and solve the same Torch segment conductor without an electronic fixed point.",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "device": device,
            "angular_degree": angular_degree,
            "full_cosmospace_parameter_sha256": _canonical_sha256(
                asdict(full_parameters)
            ),
            "frozen_energy_ledger": "one-half source/reaction-field pairing (Torch segment-COSMO boundary energy)",
            "geometry_relaxation_or_conformer_averaging": False,
            "post_hoc_shift_or_rescaling": False,
        },
        "claim_boundary": {
            "does_establish": [
                "A paired mechanism intervention that removes the MACE-EF electronic fixed point at the same 20 geometries."
            ],
            "does_not_establish": [
                "Frozen dielectric energy as a total hydration free energy.",
                "Accuracy or admission of frozen MACE-EF sources or open24a surface transfer.",
            ],
        },
        "water_frozen_source": {
            **water_record,
            "profile_path": _repository_relative(water_profile_path),
            "profile_sha256": _sha256_file(water_profile_path),
        },
        "records": [],
        "failures": [],
    }
    _write_json_atomic(OUTPUT_PATH, artifact)
    started = time.perf_counter()
    for index, original in enumerate(primary["records"], start=1):
        print(
            f"[{index:02d}/20] {original['compound_id']} {original['name']}", flush=True
        )
        try:
            xyz_spec = original["generated_xyz"]
            xyz_path = _resolve_artifact_path(xyz_spec["path"])
            if _sha256_file(xyz_path) != xyz_spec["sha256"]:
                raise RuntimeError("Primary generated geometry drifted.")
            profile, frozen_source = _frozen_profile(
                xyz_path,
                name=str(original["compound_id"]),
                checkpoint=checkpoint,
                device=device,
                angular_degree=angular_degree,
            )
            frozen_profile_path = (
                profile_dir / f"{original['compound_id']}.frozen.torch-cosmors.json"
            )
            write_sigma_profile(profile, frozen_profile_path)
            full = open24a_solvation_free_energy(
                profile,
                water,
                temperature_k=298.15,
                ring_atom_count=int(original["ring_atom_count"]),
                solvent_liquid_molar_volume_cm3_mol=18.06863632,
                cosmospace_parameters=full_parameters,
            )
            no_hb = open24a_solvation_free_energy(
                profile,
                water,
                temperature_k=298.15,
                ring_atom_count=int(original["ring_atom_count"]),
                solvent_liquid_molar_volume_cm3_mol=18.06863632,
                cosmospace_parameters=no_hb_parameters,
            )
            experimental = float(original["experimental_kcal_mol"])
            frozen_dielectric = float(frozen_source["dielectric_kcal_mol"])
            full_value = float(full.delta_g_solvation_kcal_mol.detach())
            no_hb_value = float(no_hb.delta_g_solvation_kcal_mol.detach())
            scf_dielectric = float(
                original["model_result"]["ledger_kcal_mol"]["dielectric"]
            )
            record = {
                "ordinal": int(original["ordinal"]),
                "compound_id": original["compound_id"],
                "name": original["name"],
                "chemical_class": original["chemical_class"],
                "experimental_kcal_mol": experimental,
                "generated_xyz": dict(xyz_spec),
                "frozen_profile": {
                    "path": _repository_relative(frozen_profile_path),
                    "sha256": _sha256_file(frozen_profile_path),
                },
                "frozen_source": frozen_source,
                "frozen_dielectric_mechanism_only": {
                    "predicted_kcal_mol": frozen_dielectric,
                    "signed_difference_from_experiment_kcal_mol": (
                        frozen_dielectric - experimental
                    ),
                    "not_total_hydration_free_energy": True,
                },
                "scf_dielectric_primary": {
                    "predicted_kcal_mol": scf_dielectric,
                    "signed_difference_from_experiment_kcal_mol": (
                        scf_dielectric - experimental
                    ),
                    "scf_minus_frozen_kcal_mol": scf_dielectric - frozen_dielectric,
                    "passivity_passed": bool(
                        original["mace_ef_surface"]["electronic_passivity"][
                            "passivity_passed"
                        ]
                    ),
                    "not_total_hydration_free_energy": True,
                },
                "frozen_full_open24a": {
                    "predicted_kcal_mol": full_value,
                    "signed_error_kcal_mol": full_value - experimental,
                    "result": full.as_dict(),
                },
                "frozen_no_hydrogen_bond": {
                    "predicted_kcal_mol": no_hb_value,
                    "signed_error_kcal_mol": no_hb_value - experimental,
                    "result": no_hb.as_dict(),
                },
            }
            artifact["records"].append(record)
            print(
                f"    dielectric frozen={frozen_dielectric:+.3f}, "
                f"SCF={scf_dielectric:+.3f}, full={full_value:+.3f}, "
                f"no-HB={no_hb_value:+.3f}",
                flush=True,
            )
            del profile, full, no_hb
            torch.cuda.empty_cache()
        except Exception as exc:
            artifact["failures"].append(
                {
                    "compound_id": original["compound_id"],
                    "name": original["name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
        _write_json_atomic(OUTPUT_PATH, artifact)

    records = artifact["records"]
    artifact["summaries"] = {
        "frozen_full_open24a": _prediction_summary(records, "frozen_full_open24a"),
        "frozen_no_hydrogen_bond": _prediction_summary(
            records, "frozen_no_hydrogen_bond"
        ),
    }
    if records:
        response_shifts = np.asarray(
            [
                float(item["scf_dielectric_primary"]["scf_minus_frozen_kcal_mol"])
                for item in records
            ]
        )
        artifact["scf_response_shift"] = {
            "interpretation": "paired dielectric-component mechanism diagnostic, not total hydration accuracy",
            "mean_kcal_mol": float(np.mean(response_shifts)),
            "mean_absolute_kcal_mol": float(np.mean(np.abs(response_shifts))),
            "maximum_absolute_kcal_mol": float(np.max(np.abs(response_shifts))),
        }
    else:
        artifact["scf_response_shift"] = None
    artifact["runtime_seconds"] = time.perf_counter() - started
    artifact["completed_at_utc"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    artifact["status"] = (
        "complete"
        if len(records) == 20 and not artifact["failures"]
        else "complete-with-failures"
    )
    artifact["scientific_status"] = (
        "complete-mechanism-diagnostic"
        if artifact["status"] == "complete"
        else "incomplete-mechanism-diagnostic"
    )
    _write_json_atomic(OUTPUT_PATH, artifact)
    print(json.dumps(artifact["summaries"], indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(artifact["scf_response_shift"], indent=2, sort_keys=True), flush=True
    )
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
