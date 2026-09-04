#!/usr/bin/env python3
"""Run the preregistered two-map MACE-EF/conductor-COSMO diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (  # noqa: E402
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (  # noqa: E402
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from maple.function.cosmors_torch.segment_cosmo import (  # noqa: E402
    HARTREE_EV,
    TorchSegmentCOSMO,
    TorchSegmentCOSMOConfig,
)
from maple.function.mlip_cosmo_rs import open_cosmors_24a_cavity_radii  # noqa: E402
from run_mace_ef_cosmors_freesolv20 import (  # noqa: E402
    _bind_source_files,
    _git_blob_sha256,
    _repository_relative,
    _runtime_versions,
    _sha256_file,
    _write_json_atomic,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v3.json"
PREREGISTRATION_PATH = SCRIPT_DIR / "mace-ef-cosmo-freesolv20-two-step-prereg-v1.json"
BUNDLE_MANIFEST_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.json"
BUNDLE_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
OUTPUT_PATH = SCRIPT_DIR / "mace-ef-cosmo-freesolv20-two-step-v1.json"
HARTREE_TO_KCAL_MOL = 627.5094740631
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL / HARTREE_EV
MAP_APPLICATION_COUNT = 2


def _assert_source_bound(artifact: dict[str, object]) -> None:
    commit = artifact.get("execution_git_head")
    hashes = artifact.get("source_files_sha256")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("Input artifact lacks a full execution Git commit.")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Input artifact lacks committed source-file binding.")
    for relative, expected in hashes.items():
        if _git_blob_sha256(commit, str(relative)) != expected:
            raise RuntimeError(
                f"Input artifact source {relative!r} is not bound at {commit}."
            )


def _load_geometries_from_bundle(
    primary: dict[str, object],
    bundle_manifest: dict[str, object],
) -> dict[str, Any]:
    expected_members = dict(bundle_manifest["member_sha256"])
    geometries = {}
    with TemporaryDirectory(prefix="maple-mace-ef-two-step-") as temporary:
        root = Path(temporary)
        with tarfile.open(BUNDLE_PATH, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for record in primary["records"]:
                compound_id = str(record["compound_id"])
                member_name = f"geometries/{compound_id}.xyz"
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"Profile bundle lacks {member_name!r}.")
                expected = str(record["generated_xyz"]["sha256"])
                if expected_members.get(member_name) != expected:
                    raise RuntimeError(
                        f"Geometry manifest and primary disagree for {compound_id}."
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read {member_name!r}.")
                payload = extracted.read()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise RuntimeError(f"Geometry {compound_id} failed SHA256.")
                path = root / f"{compound_id}.xyz"
                path.write_bytes(payload)
                geometries[compound_id] = read(path, index=0)
    return geometries


def _source_array(state: object) -> np.ndarray:
    source = np.asarray(state.conjugate_source_raw, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 4 or not np.all(np.isfinite(source)):
        raise RuntimeError("MACE-EF returned an invalid conjugate source.")
    if abs(float(np.sum(source[:, 0]))) > 2.0e-5:
        raise RuntimeError("MACE-EF two-step source violated the neutral-charge gate.")
    return source


def _two_step_response(
    electronic: object,
    continuum: object,
    positions_angstrom: np.ndarray,
) -> dict[str, object]:
    """Apply exactly two unmixed source maps; never evaluate a third map."""

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    zero_field = np.zeros((positions.shape[0], 4), dtype=np.float64)
    gas = electronic.evaluate(positions, zero_field)
    source0 = _source_array(gas)
    field0 = np.asarray(
        continuum.drive_cartesian(positions, source0),
        dtype=np.float64,
    )
    first = electronic.evaluate(positions, field0)
    source1 = _source_array(first)
    field1 = np.asarray(
        continuum.drive_cartesian(positions, source1),
        dtype=np.float64,
    )
    second = electronic.evaluate(positions, field1)
    source2 = _source_array(second)
    field2 = np.asarray(
        continuum.drive_cartesian(positions, source2),
        dtype=np.float64,
    )

    sources = (source0, source1, source2)
    fields = (field0, field1, field2)
    stages = []
    for index, (source, field) in enumerate(zip(sources, fields, strict=True)):
        energy_ev = float(continuum.energy(positions, source))
        pairing_ev = MACE_POLAR_L1_PAIRING.pair(source, field)
        stages.append(
            {
                "source_index": index,
                "cosmo_boundary_energy_ev": energy_ev,
                "cosmo_boundary_energy_kcal_mol": energy_ev * EV_TO_KCAL_MOL,
                "source_field_pairing_ev": pairing_ev,
                "half_coupling_identity_error_ev": abs(energy_ev - 0.5 * pairing_ev),
                "source_charge_e": float(np.sum(source[:, 0])),
                "source_maximum_absolute_component": float(np.max(np.abs(source))),
                "field_maximum_absolute_component": float(np.max(np.abs(field))),
            }
        )
    first_update = float(np.max(np.abs(source1 - source0)))
    second_update = float(np.max(np.abs(source2 - source1)))
    return {
        "map_applications": MAP_APPLICATION_COUNT,
        "electronic_evaluation_count": 3,
        "third_map_evaluated": False,
        "stages": stages,
        "source_updates": {
            "first_maximum_absolute_component": first_update,
            "second_maximum_absolute_component": second_update,
            "second_to_first_ratio": (
                None if first_update == 0.0 else second_update / first_update
            ),
        },
    }


def _distribution(values: list[float], *, units: str) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "units": units}
    return {
        "count": int(array.size),
        "units": units,
        "mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(np.abs(array))),
        "median_absolute": float(np.median(np.abs(array))),
        "maximum_absolute": float(np.max(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array**2))),
    }


def main() -> int:
    import torch

    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    bundle_manifest = json.loads(BUNDLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if primary.get("status") != "complete":
        raise ValueError("Two-step response requires the complete v3 primary.")
    if (
        primary.get("artifact")
        != preregistration["record_selection"]["primary_artifact"]
    ):
        raise ValueError("Two-step preregistration and primary artifact differ.")
    if preregistration["response_map"]["map_applications"] != MAP_APPLICATION_COUNT:
        raise ValueError("Implemented map count drifted from preregistration.")
    if _sha256_file(BUNDLE_PATH) != bundle_manifest["archive"]["sha256"]:
        raise RuntimeError("Committed geometry/profile bundle failed SHA256.")
    if bundle_manifest["input_artifacts"]["primary"]["sha256"] != _sha256_file(
        PRIMARY_PATH
    ):
        raise RuntimeError("Geometry bundle is not bound to the v3 primary.")
    _assert_source_bound(primary)
    _assert_source_bound(bundle_manifest)
    execution_git_head, source_files_sha256 = _bind_source_files(
        (Path(__file__), PREREGISTRATION_PATH, BUNDLE_MANIFEST_PATH, BUNDLE_PATH)
    )

    checkpoint = Path(primary["model"]["checkpoint_path"])
    if _sha256_file(checkpoint) != primary["model"]["checkpoint_sha256"]:
        raise RuntimeError("MACE-EF checkpoint bytes drifted from the primary.")
    device = str(primary["model"]["device"])
    angular_degree = int(primary["model"]["angular_degree"])
    geometries = _load_geometries_from_bundle(primary, bundle_manifest)

    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact": preregistration["preregistration_id"],
        "status": "running",
        "scientific_status": "running-partial-component-diagnostic",
        "diagnostic_only": True,
        "admission_eligible": False,
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "execution_git_head": execution_git_head,
        "source_files_sha256": source_files_sha256,
        "preregistration": {
            "path": _repository_relative(PREREGISTRATION_PATH),
            "sha256": _sha256_file(PREREGISTRATION_PATH),
            "id": preregistration["preregistration_id"],
        },
        "primary_artifact": {
            "path": _repository_relative(PRIMARY_PATH),
            "sha256": _sha256_file(PRIMARY_PATH),
            "artifact": primary["artifact"],
        },
        "geometry_bundle": {
            "path": _repository_relative(BUNDLE_PATH),
            "sha256": _sha256_file(BUNDLE_PATH),
            "manifest_path": _repository_relative(BUNDLE_MANIFEST_PATH),
            "manifest_sha256": _sha256_file(BUNDLE_MANIFEST_PATH),
        },
        "runtime": {
            **_runtime_versions(),
            "torch_version": str(torch.__version__),
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
        },
        "method": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "angular_degree": angular_degree,
            "response_map": preregistration["response_map"],
            "reported_energy": preregistration["reported_energy"],
        },
        "claim_boundary": preregistration["claim_boundary"],
        "records": [],
        "failures": [],
    }
    _write_json_atomic(OUTPUT_PATH, artifact)
    started = time.perf_counter()
    for ordinal, primary_record in enumerate(primary["records"], start=1):
        compound_id = str(primary_record["compound_id"])
        print(f"[{ordinal:02d}/20] {compound_id} {primary_record['name']}", flush=True)
        try:
            atoms = geometries[compound_id]
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
            continuum = TorchSegmentCOSMO(
                TorchSegmentCOSMOConfig(
                    atomic_numbers=atomic_numbers,
                    radii_angstrom=tuple(open_cosmors_24a_cavity_radii(symbols)),
                    angular_degree=angular_degree,
                )
            )
            result = _two_step_response(electronic, continuum, positions)
            experimental = float(primary_record["experimental_kcal_mol"])
            converged = (
                float(
                    primary_record["mace_ef_surface"]["boundary_polarization_energy_ev"]
                )
                * EV_TO_KCAL_MOL
            )
            stages = result["stages"]
            for stage in stages:
                value = float(stage["cosmo_boundary_energy_kcal_mol"])
                stage["partial_minus_total_freesolv_label_kcal_mol"] = (
                    value - experimental
                )
                stage["difference_from_converged_boundary_kcal_mol"] = value - converged
                stage["not_total_hydration_free_energy"] = True
            record = {
                "ordinal": int(primary_record["ordinal"]),
                "compound_id": compound_id,
                "name": primary_record["name"],
                "chemical_class": primary_record["chemical_class"],
                "experimental_total_hydration_label_kcal_mol": experimental,
                "electronic_passivity_passed": bool(
                    primary_record["mace_ef_surface"]["electronic_passivity"][
                        "passivity_passed"
                    ]
                ),
                "converged_boundary_energy_kcal_mol": converged,
                **result,
            }
            artifact["records"].append(record)
            print(
                "    "
                + ", ".join(
                    f"U{stage['source_index']}="
                    f"{stage['cosmo_boundary_energy_kcal_mol']:+.3f}"
                    for stage in stages
                )
                + f", |dc2|={result['source_updates']['second_maximum_absolute_component']:.3e}",
                flush=True,
            )
            del electronic, continuum
            torch.cuda.empty_cache()
        except Exception as exc:
            artifact["failures"].append(
                {
                    "ordinal": int(primary_record["ordinal"]),
                    "compound_id": compound_id,
                    "name": primary_record["name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
        _write_json_atomic(OUTPUT_PATH, artifact)

    records = artifact["records"]
    stage_summaries = {}
    for index in range(3):
        stage_summaries[f"c{index}"] = {
            "partial_component_difference_from_total_freesolv_label": _distribution(
                [
                    float(
                        record["stages"][index][
                            "partial_minus_total_freesolv_label_kcal_mol"
                        ]
                    )
                    for record in records
                ],
                units="kcal/mol",
            ),
            "difference_from_converged_boundary": _distribution(
                [
                    float(
                        record["stages"][index][
                            "difference_from_converged_boundary_kcal_mol"
                        ]
                    )
                    for record in records
                ],
                units="kcal/mol",
            ),
        }
    ratios = [
        float(record["source_updates"]["second_to_first_ratio"])
        for record in records
        if record["source_updates"]["second_to_first_ratio"] is not None
    ]
    artifact["summaries"] = {
        "stages": stage_summaries,
        "source_update": {
            "first_maximum_absolute_component": _distribution(
                [
                    float(record["source_updates"]["first_maximum_absolute_component"])
                    for record in records
                ],
                units="source-component",
            ),
            "second_maximum_absolute_component": _distribution(
                [
                    float(record["source_updates"]["second_maximum_absolute_component"])
                    for record in records
                ],
                units="source-component",
            ),
            "second_to_first_ratio": _distribution(ratios, units="dimensionless"),
            "contracting_record_count": sum(value < 1.0 for value in ratios),
            "amplifying_or_equal_record_count": sum(value >= 1.0 for value in ratios),
        },
        "electronic_passivity": {
            "passed_count": sum(
                record["electronic_passivity_passed"] is True for record in records
            ),
            "failed_count": sum(
                record["electronic_passivity_passed"] is not True for record in records
            ),
            "two_step_route_admission_eligible": False,
        },
    }
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
        "complete-partial-component-diagnostic"
        if artifact["status"] == "complete"
        else "incomplete-partial-component-diagnostic"
    )
    artifact["interpretation"] = {
        "automatic_accuracy_verdict": None,
        "reason": "The reported conductor-COSMO term is only one component of total hydration free energy.",
        "iteration_count_was_user_specified_before_this_run": True,
        "iteration_count_may_not_be_changed_using_this_result": True,
        "stationary_solution_claimed": False,
    }
    _write_json_atomic(OUTPUT_PATH, artifact)
    print(json.dumps(artifact["summaries"], indent=2, sort_keys=True), flush=True)
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
