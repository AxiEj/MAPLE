#!/usr/bin/env python3
"""Ablate open24a COSMOspace misfit and hydrogen-bond terms on FreeSolv-20."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.cosmors_torch.cosmospace import (  # noqa: E402
    GAS_CONSTANT_J_PER_MOL_K,
    OPEN_COSMORS_24A_PARAMETERS,
)
from maple.function.cosmors_torch.fixed_structure import _load_species  # noqa: E402
from maple.function.cosmors_torch.surface import read_sigma_profile  # noqa: E402
from maple.function.cosmors_torch.thermodynamics import (  # noqa: E402
    open24a_solvation_free_energy,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v2.json"
OUTPUT_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-component-ablation-v1.json"
WORK_DIR = REPOSITORY_ROOT / ".omx/benchmarks/mace-ef-cosmors-freesolv20-diverse-v2"
EPSILON_COEFFICIENT = 1.0e-30
JOULE_TO_KCAL = 1.0 / 4184.0
TEMPERATURE_K = 298.15
RT_KCAL_MOL = GAS_CONSTANT_J_PER_MOL_K * TEMPERATURE_K * JOULE_TO_KCAL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _summary(records: list[dict[str, object]], variant: str) -> dict[str, object]:
    errors = np.asarray(
        [float(item["variants"][variant]["signed_error_kcal_mol"]) for item in records]
    )
    absolute = np.abs(errors)
    worst_index = int(np.argmax(absolute))
    return {
        "count": len(records),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mae_kcal_mol": float(np.mean(absolute)),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error_kcal_mol": float(np.median(absolute)),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "within_1_kcal_mol_count": int(np.sum(absolute <= 1.0)),
        "within_2_kcal_mol_count": int(np.sum(absolute <= 2.0)),
        "worst_record": {
            "compound_id": records[worst_index]["compound_id"],
            "name": records[worst_index]["name"],
            "signed_error_kcal_mol": float(errors[worst_index]),
        },
    }


def main() -> int:
    import torch

    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    device = str(primary["model"]["device"])
    checkpoint = Path(primary["model"]["checkpoint_path"])
    water = read_sigma_profile(WORK_DIR / "water.torch-cosmors.json")
    variants = {
        "no_hydrogen_bond": replace(
            OPEN_COSMORS_24A_PARAMETERS,
            name="open24a-ablation-no-hydrogen-bond",
            hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
            maximum_iterations=20000,
        ),
        "no_misfit": replace(
            OPEN_COSMORS_24A_PARAMETERS,
            name="open24a-ablation-no-misfit",
            misfit_alpha_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
            maximum_iterations=20000,
        ),
        "no_misfit_or_hydrogen_bond": replace(
            OPEN_COSMORS_24A_PARAMETERS,
            name="open24a-ablation-no-misfit-or-hydrogen-bond",
            misfit_alpha_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
            hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
            maximum_iterations=20000,
        ),
    }
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-freesolv20-component-ablation-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "primary_artifact": {
            "path": str(PRIMARY_PATH),
            "sha256": _sha256(PRIMARY_PATH),
        },
        "method": {
            "description": "Reuse the frozen primary inputs and MACE-EF surface path; set one or both positive COSMOspace interaction coefficients to 1e-30.",
            "temperature_k": TEMPERATURE_K,
            "rt_kcal_mol": RT_KCAL_MOL,
            "epsilon_coefficient": EPSILON_COEFFICIENT,
            "maximum_iterations": 20000,
            "interpretation": "Mechanism attribution only; ablated variants are not fitted physical models.",
        },
        "records": [],
        "failures": [],
    }
    started = time.perf_counter()
    for index, original in enumerate(primary["records"], start=1):
        print(
            f"[{index:02d}/20] {original['compound_id']} {original['name']}", flush=True
        )
        try:
            profile, _, source = _load_species(
                {
                    "name": original["compound_id"],
                    "xyz": original["mace_ef_surface"]["xyz"],
                    "charge": 0,
                    "multiplicity": 1,
                    "ring_atom_count": int(original["ring_atom_count"]),
                },
                base=WORK_DIR,
                checkpoint_path=checkpoint,
                device=device,
                angular_degree=int(primary["model"]["angular_degree"]),
                scf_payload={},
            )
            regenerated_dielectric = (
                float(profile.dielectric_energy_hartree.detach().cpu()) * 627.5094740631
            )
            primary_dielectric = float(
                original["model_result"]["ledger_kcal_mol"]["dielectric"]
            )
            if not np.isclose(
                regenerated_dielectric, primary_dielectric, rtol=0.0, atol=1.0e-2
            ):
                raise RuntimeError(
                    "Regenerated MACE-EF dielectric term differs from the primary artifact: "
                    f"{regenerated_dielectric} versus {primary_dielectric}."
                )
            record = {
                "ordinal": int(original["ordinal"]),
                "compound_id": original["compound_id"],
                "name": original["name"],
                "chemical_class": original["chemical_class"],
                "experimental_kcal_mol": float(original["experimental_kcal_mol"]),
                "mace_ef_passivity_passed": bool(
                    source["electronic_passivity"]["passivity_passed"]
                ),
                "variants": {
                    "full_primary": {
                        "predicted_kcal_mol": float(original["predicted_kcal_mol"]),
                        "signed_error_kcal_mol": float(
                            original["signed_error_kcal_mol"]
                        ),
                        "residual_log_activity": float(
                            original["model_result"]["log_activity"]["residual"]
                        ),
                        "cosmospace_iterations": int(
                            original["model_result"]["cosmospace_iterations"]
                        ),
                    }
                },
            }
            for variant, parameters in variants.items():
                result = open24a_solvation_free_energy(
                    profile,
                    water,
                    temperature_k=TEMPERATURE_K,
                    ring_atom_count=int(original["ring_atom_count"]),
                    solvent_liquid_molar_volume_cm3_mol=18.06863632,
                    cosmospace_parameters=parameters,
                )
                predicted = float(result.delta_g_solvation_kcal_mol.detach().cpu())
                experimental = float(original["experimental_kcal_mol"])
                record["variants"][variant] = {
                    "predicted_kcal_mol": predicted,
                    "signed_error_kcal_mol": predicted - experimental,
                    "residual_log_activity": float(
                        result.activity.residual_log_activity.detach().cpu()
                    ),
                    "residual_kcal_mol": float(
                        result.activity.residual_log_activity.detach().cpu()
                    )
                    * RT_KCAL_MOL,
                    "cosmospace_iterations": int(
                        result.activity.cosmospace.iterations.detach().cpu()
                    ),
                }
            artifact["records"].append(record)
            print(
                "    "
                + ", ".join(
                    f"{name}={record['variants'][name]['predicted_kcal_mol']:+.3f}"
                    for name in variants
                ),
                flush=True,
            )
            del profile
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
    records = artifact["records"]
    artifact["summaries"] = {
        variant: _summary(records, variant) for variant in ("full_primary", *variants)
    }
    artifact["passivity"] = {
        "passed_count": sum(item["mace_ef_passivity_passed"] for item in records),
        "failed_count": sum(not item["mace_ef_passivity_passed"] for item in records),
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
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["summaries"], indent=2, sort_keys=True), flush=True)
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
