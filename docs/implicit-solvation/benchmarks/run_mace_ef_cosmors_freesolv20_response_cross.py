#!/usr/bin/env python3
"""Run the preregistered 20/20 sigma-tail and 2x2 source-state diagnostic."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.cosmors_torch.cosmospace import (  # noqa: E402
    OPEN_COSMORS_24A_PARAMETERS,
)
from maple.function.cosmors_torch.surface import read_sigma_profile  # noqa: E402
from maple.function.cosmors_torch.thermodynamics import (  # noqa: E402
    open24a_solvation_free_energy,
)
from run_mace_ef_cosmors_freesolv20 import (  # noqa: E402
    _bind_source_files,
    _canonical_sha256,
    _repository_relative,
    _resolve_artifact_path,
    _runtime_versions,
    _sha256_file,
    _write_json_atomic,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v3.json"
FROZEN_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-frozen-source-ablation-v2.json"
PREREGISTRATION_PATH = (
    SCRIPT_DIR / "mace-ef-cosmors-freesolv20-response-cross-prereg-v1.json"
)
OUTPUT_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-response-cross-v1.json"
EPSILON_COEFFICIENT = 1.0e-30


def _to_device(profile, device: str):
    updates = {}
    for field in (
        "sigma_e_per_angstrom2",
        "sigma_orthogonal_e_per_angstrom2",
        "areas_angstrom2",
        "atomic_numbers",
        "hydrogen_bond_donor_weight",
        "hydrogen_bond_acceptor_weight",
        "ionic_contact_weight",
        "cavity_volume_angstrom3",
        "dielectric_energy_hartree",
        "molecular_charge_e",
    ):
        updates[field] = getattr(profile, field).to(device=device)
    return replace(profile, **updates)


def _checked_profile(spec: dict[str, object], *, device: str):
    path = _resolve_artifact_path(spec["path"])
    if _sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"Frozen sigma-profile bytes drifted: {path}")
    return _to_device(read_sigma_profile(path), device)


def _tail_metrics(profile) -> dict[str, float]:
    sigma = profile.sigma_e_per_angstrom2.detach()
    area = profile.areas_angstrom2.detach()
    threshold = (
        OPEN_COSMORS_24A_PARAMETERS.hydrogen_bond_sigma_threshold_e_per_angstrom2
    )
    negative_excess = (-sigma - threshold).clamp(min=0.0)
    positive_excess = (sigma - threshold).clamp(min=0.0)
    return {
        "sigma_min_e_per_angstrom2": float(sigma.min().cpu()),
        "sigma_max_e_per_angstrom2": float(sigma.max().cpu()),
        "total_area_angstrom2": float(area.sum().cpu()),
        "negative_hb_active_area_angstrom2": float(
            area[sigma < -threshold].sum().cpu()
        ),
        "positive_hb_active_area_angstrom2": float(area[sigma > threshold].sum().cpu()),
        "negative_hb_excess_area_e": float((area * negative_excess).sum().cpu()),
        "positive_hb_excess_area_e": float((area * positive_excess).sum().cpu()),
        "area_weighted_sigma_second_moment_e2_per_angstrom2": float(
            (area * sigma.square()).sum().cpu()
        ),
        "area_weighted_sigma_fourth_moment_e4_per_angstrom6": float(
            (area * sigma**4).sum().cpu()
        ),
    }


def _evaluate_arm(
    solute,
    water,
    *,
    parameters,
    ring_atom_count: int,
    experimental_kcal_mol: float,
) -> dict[str, object]:
    result = open24a_solvation_free_energy(
        solute,
        water,
        temperature_k=298.15,
        ring_atom_count=ring_atom_count,
        solvent_liquid_molar_volume_cm3_mol=18.06863632,
        cosmospace_parameters=parameters,
    )
    predicted = float(result.delta_g_solvation_kcal_mol.detach().cpu())
    return {
        "predicted_kcal_mol": predicted,
        "signed_error_kcal_mol": predicted - experimental_kcal_mol,
        "result": result.as_dict(),
    }


def _arm_summary(records: list[dict[str, object]], arm: str) -> dict[str, object]:
    errors = np.asarray(
        [float(record["arms"][arm]["signed_error_kcal_mol"]) for record in records]
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


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean_kcal_mol": float(np.mean(array)),
        "mean_absolute_kcal_mol": float(np.mean(np.abs(array))),
        "median_absolute_kcal_mol": float(np.median(np.abs(array))),
        "rmse_kcal_mol": float(np.sqrt(np.mean(array**2))),
        "maximum_absolute_kcal_mol": float(np.max(np.abs(array))),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def main() -> int:
    import torch

    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if primary.get("artifact") != preregistration.get("primary_selection_id"):
        raise ValueError("Response-cross preregistration and primary selection differ.")
    if primary.get("status") != "complete" or frozen.get("status") != "complete":
        raise ValueError(
            "Response-cross requires complete primary and frozen controls."
        )
    execution_git_head, source_files_sha256 = _bind_source_files(
        (Path(__file__), PREREGISTRATION_PATH)
    )
    if {
        primary.get("execution_git_head"),
        frozen.get("execution_git_head"),
        execution_git_head,
    } != {execution_git_head}:
        raise RuntimeError("All response-cross inputs must use one committed Git tree.")
    if frozen["primary_artifact"]["sha256"] != _sha256_file(PRIMARY_PATH):
        raise RuntimeError(
            "Frozen control is not bound to the current primary artifact."
        )

    primary_ids = [record["compound_id"] for record in primary["records"]]
    frozen_ids = [record["compound_id"] for record in frozen["records"]]
    if primary_ids != frozen_ids or len(primary_ids) != 20:
        raise RuntimeError(
            "Primary and frozen controls must contain the same 20 records."
        )
    frozen_by_id = {record["compound_id"]: record for record in frozen["records"]}

    device = str(primary["model"]["device"])
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
    scf_water = _checked_profile(
        {
            "path": primary["water_profile"]["path"],
            "sha256": primary["water_profile"]["sha256"],
        },
        device=device,
    )
    frozen_water = _checked_profile(
        {
            "path": frozen["water_frozen_source"]["profile_path"],
            "sha256": frozen["water_frozen_source"]["profile_sha256"],
        },
        device=device,
    )
    water_tail_metrics = {
        "frozen_zero_field": _tail_metrics(frozen_water),
        "self_consistent": _tail_metrics(scf_water),
    }

    arm_specs = (
        ("frozen_solute__frozen_water__full", "frozen", "frozen", full_parameters),
        (
            "frozen_solute__frozen_water__no_hydrogen_bond",
            "frozen",
            "frozen",
            no_hb_parameters,
        ),
        (
            "self_consistent_solute__frozen_water__full",
            "self_consistent",
            "frozen",
            full_parameters,
        ),
        (
            "self_consistent_solute__frozen_water__no_hydrogen_bond",
            "self_consistent",
            "frozen",
            no_hb_parameters,
        ),
        (
            "frozen_solute__self_consistent_water__full",
            "frozen",
            "self_consistent",
            full_parameters,
        ),
        (
            "frozen_solute__self_consistent_water__no_hydrogen_bond",
            "frozen",
            "self_consistent",
            no_hb_parameters,
        ),
        (
            "self_consistent_solute__self_consistent_water__full",
            "self_consistent",
            "self_consistent",
            full_parameters,
        ),
        (
            "self_consistent_solute__self_consistent_water__no_hydrogen_bond",
            "self_consistent",
            "self_consistent",
            no_hb_parameters,
        ),
    )
    arm_names = [item[0] for item in arm_specs]
    if arm_names != preregistration.get("arms"):
        raise RuntimeError(
            "Implemented response-cross arms drifted from preregistration."
        )

    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-freesolv20-response-cross-v1",
        "status": "running",
        "scientific_status": "running-mechanism-diagnostic",
        "admission_eligible": False,
        "diagnostic_only": True,
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
        "frozen_control_artifact": {
            "path": _repository_relative(FROZEN_PATH),
            "sha256": _sha256_file(FROZEN_PATH),
            "artifact": frozen["artifact"],
        },
        "runtime": _runtime_versions(),
        "method": {
            "temperature_k": 298.15,
            "device": device,
            "full_cosmospace_parameter_sha256": _canonical_sha256(
                asdict(full_parameters)
            ),
            "no_hydrogen_bond_cosmospace_parameter_sha256": _canonical_sha256(
                asdict(no_hb_parameters)
            ),
            "epsilon_coefficient": EPSILON_COEFFICIENT,
            "post_hoc_shift_or_rescaling": False,
            "geometry_relaxation_or_conformer_averaging": False,
            "experimental_values_used_to_define_arms": False,
        },
        "water_tail_metrics": water_tail_metrics,
        "records": [],
        "failures": [],
        "claim_boundary": preregistration["claim_boundary"],
    }
    _write_json_atomic(OUTPUT_PATH, artifact)
    started = time.perf_counter()
    for index, primary_record in enumerate(primary["records"], start=1):
        compound_id = primary_record["compound_id"]
        print(f"[{index:02d}/20] {compound_id} {primary_record['name']}", flush=True)
        try:
            frozen_record = frozen_by_id[compound_id]
            scf_solute = _checked_profile(
                primary_record["mace_ef_profile"], device=device
            )
            frozen_solute = _checked_profile(
                frozen_record["frozen_profile"], device=device
            )
            solutes = {"frozen": frozen_solute, "self_consistent": scf_solute}
            waters = {"frozen": frozen_water, "self_consistent": scf_water}
            experimental = float(primary_record["experimental_kcal_mol"])
            arms = {
                name: _evaluate_arm(
                    solutes[solute_state],
                    waters[water_state],
                    parameters=parameters,
                    ring_atom_count=int(primary_record["ring_atom_count"]),
                    experimental_kcal_mol=experimental,
                )
                for name, solute_state, water_state, parameters in arm_specs
            }
            ss_full = arms["self_consistent_solute__self_consistent_water__full"]
            if not math.isclose(
                float(ss_full["predicted_kcal_mol"]),
                float(primary_record["predicted_kcal_mol"]),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise RuntimeError("Self-consistent/full replay drifted from primary.")
            ff_full = arms["frozen_solute__frozen_water__full"]
            ff_no_hb = arms["frozen_solute__frozen_water__no_hydrogen_bond"]
            if not math.isclose(
                float(ff_full["predicted_kcal_mol"]),
                float(frozen_record["frozen_full_open24a"]["predicted_kcal_mol"]),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ) or not math.isclose(
                float(ff_no_hb["predicted_kcal_mol"]),
                float(frozen_record["frozen_no_hydrogen_bond"]["predicted_kcal_mol"]),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise RuntimeError("Frozen/full replay drifted from frozen control.")

            def h(solute_state: str, water_state: str) -> float:
                prefix = f"{solute_state}_solute__{water_state}_water"
                full = float(arms[f"{prefix}__full"]["predicted_kcal_mol"])
                no_hb = float(arms[f"{prefix}__no_hydrogen_bond"]["predicted_kcal_mol"])
                return full - no_hb

            h_ff = h("frozen", "frozen")
            h_sf = h("self_consistent", "frozen")
            h_fs = h("frozen", "self_consistent")
            h_ss = h("self_consistent", "self_consistent")
            frozen_tail = _tail_metrics(frozen_solute)
            scf_tail = _tail_metrics(scf_solute)
            tail_ratios = {
                key: _safe_ratio(scf_tail[key], frozen_tail[key])
                for key in preregistration["tail_metrics"]
            }
            contrasts = {
                "h_frozen_solute__frozen_water_kcal_mol": h_ff,
                "h_self_consistent_solute__frozen_water_kcal_mol": h_sf,
                "h_frozen_solute__self_consistent_water_kcal_mol": h_fs,
                "h_self_consistent_solute__self_consistent_water_kcal_mol": h_ss,
                "solute_response_at_frozen_water_kcal_mol": h_sf - h_ff,
                "solute_response_at_self_consistent_water_kcal_mol": h_ss - h_fs,
                "water_response_at_frozen_solute_kcal_mol": h_fs - h_ff,
                "water_response_at_self_consistent_solute_kcal_mol": h_ss - h_sf,
                "nonlinear_interaction_kcal_mol": h_ss - h_sf - h_fs + h_ff,
                "full_total_scf_minus_frozen_kcal_mol": float(
                    arms["self_consistent_solute__self_consistent_water__full"][
                        "predicted_kcal_mol"
                    ]
                )
                - float(
                    arms["frozen_solute__frozen_water__full"]["predicted_kcal_mol"]
                ),
                "no_hydrogen_bond_total_scf_minus_frozen_kcal_mol": float(
                    arms[
                        "self_consistent_solute__self_consistent_water__no_hydrogen_bond"
                    ]["predicted_kcal_mol"]
                )
                - float(
                    arms["frozen_solute__frozen_water__no_hydrogen_bond"][
                        "predicted_kcal_mol"
                    ]
                ),
            }
            artifact["records"].append(
                {
                    "ordinal": int(primary_record["ordinal"]),
                    "compound_id": compound_id,
                    "name": primary_record["name"],
                    "chemical_class": primary_record["chemical_class"],
                    "experimental_kcal_mol": experimental,
                    "mace_ef_passivity_passed": bool(
                        primary_record["mace_ef_surface"]["electronic_passivity"][
                            "passivity_passed"
                        ]
                    ),
                    "tail_metrics": {
                        "frozen_zero_field": frozen_tail,
                        "self_consistent": scf_tail,
                        "self_consistent_to_frozen_ratio": tail_ratios,
                    },
                    "arms": arms,
                    "contrasts": contrasts,
                }
            )
            print(
                f"    H_FF={h_ff:+.3f}, H_SF={h_sf:+.3f}, "
                f"H_FS={h_fs:+.3f}, H_SS={h_ss:+.3f}",
                flush=True,
            )
            del scf_solute, frozen_solute, arms
            torch.cuda.empty_cache()
        except Exception as exc:
            artifact["failures"].append(
                {
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
    artifact["arm_summaries"] = {arm: _arm_summary(records, arm) for arm in arm_names}
    contrast_names = tuple(records[0]["contrasts"]) if records else ()
    artifact["contrast_summaries"] = {
        name: _distribution([float(record["contrasts"][name]) for record in records])
        for name in contrast_names
    }
    passivity_failures = [
        record["compound_id"]
        for record in records
        if record["mace_ef_passivity_passed"] is not True
    ]
    artifact["electronic_passivity"] = {
        "passed_count": len(records) - len(passivity_failures),
        "failed_count": len(passivity_failures),
        "failed_compound_ids": passivity_failures,
        "scf_route_admission_eligible": False,
    }
    artifact["interpretation"] = {
        "automatic_root_cause_verdict": None,
        "reason": "The preregistration defines falsifiers but no post-hoc materiality threshold; contrasts are reported without fitting or relabelling.",
        "hard_stop_applied": bool(passivity_failures),
        "hard_stop": preregistration["hard_stop"],
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
        "complete-mechanism-diagnostic"
        if artifact["status"] == "complete"
        else "incomplete-mechanism-diagnostic"
    )
    _write_json_atomic(OUTPUT_PATH, artifact)
    print(json.dumps(artifact["contrast_summaries"], indent=2, sort_keys=True))
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
