#!/usr/bin/env python3
"""Audit HB-active sigma tails for frozen and SCF MACE-EF profiles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from maple.function.cosmors_torch.cosmospace import (  # noqa: E402
    OPEN_COSMORS_24A_PARAMETERS,
)
from maple.function.cosmors_torch.fixed_structure import _load_species  # noqa: E402
from maple.function.cosmors_torch.surface import read_sigma_profile  # noqa: E402
from run_mace_ef_cosmors_freesolv20_frozen_source_ablation import (  # noqa: E402
    _frozen_profile,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v2.json"
OUTPUT_PATH = SCRIPT_DIR / "mace-ef-cosmors-hbond-tail-audit-v1.json"
WORK_DIR = REPOSITORY_ROOT / ".omx/benchmarks/mace-ef-cosmors-freesolv20-diverse-v2"
TARGETS = {
    "mobley_1636752",
    "mobley_3034976",
    "mobley_20524",
    "mobley_4639255",
}


def _metrics(profile) -> dict[str, float]:
    sigma = profile.sigma_e_per_angstrom2.detach().cpu()
    area = profile.areas_angstrom2.detach().cpu()
    threshold = (
        OPEN_COSMORS_24A_PARAMETERS.hydrogen_bond_sigma_threshold_e_per_angstrom2
    )
    negative_excess = torch.clamp(-(sigma + threshold), min=0.0)
    positive_excess = torch.clamp(sigma - threshold, min=0.0)
    return {
        "total_area_angstrom2": float(torch.sum(area)),
        "sigma_min_e_per_angstrom2": float(torch.min(sigma)),
        "sigma_max_e_per_angstrom2": float(torch.max(sigma)),
        "negative_hb_active_area_angstrom2": float(torch.sum(area[sigma < -threshold])),
        "positive_hb_active_area_angstrom2": float(torch.sum(area[sigma > threshold])),
        "negative_hb_excess_area_e": float(torch.sum(area * negative_excess)),
        "positive_hb_excess_area_e": float(torch.sum(area * positive_excess)),
        "area_weighted_sigma_second_moment_e2_per_angstrom2": float(
            torch.sum(area * sigma.square())
        ),
        "area_weighted_sigma_fourth_moment_e4_per_angstrom6": float(
            torch.sum(area * sigma**4)
        ),
    }


def _ratio(scf: dict[str, float], frozen: dict[str, float], key: str) -> float | None:
    denominator = frozen[key]
    return None if denominator == 0.0 else scf[key] / denominator


def main() -> int:
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    checkpoint = Path(primary["model"]["checkpoint_path"])
    device = str(primary["model"]["device"])
    angular_degree = int(primary["model"]["angular_degree"])
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-hbond-tail-audit-v1",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "threshold_e_per_angstrom2": (
            OPEN_COSMORS_24A_PARAMETERS.hydrogen_bond_sigma_threshold_e_per_angstrom2
        ),
        "records": [],
        "failures": [],
    }
    frozen_water, _ = _frozen_profile(
        Path(primary["water_profile"]["source"]["xyz"]),
        name="water-frozen-zero-field",
        checkpoint=checkpoint,
        device=device,
        angular_degree=angular_degree,
    )
    scf_water = read_sigma_profile(WORK_DIR / "water.torch-cosmors.json")
    artifact["water"] = {
        "frozen": _metrics(frozen_water),
        "scf": _metrics(scf_water),
    }
    for original in primary["records"]:
        if original["compound_id"] not in TARGETS:
            continue
        print(f"{original['compound_id']} {original['name']}", flush=True)
        try:
            frozen, _ = _frozen_profile(
                Path(original["mace_ef_surface"]["xyz"]),
                name=f"{original['compound_id']}-frozen-zero-field",
                checkpoint=checkpoint,
                device=device,
                angular_degree=angular_degree,
            )
            scf, _, _ = _load_species(
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
                angular_degree=angular_degree,
                scf_payload={},
            )
            frozen_metrics = _metrics(frozen)
            scf_metrics = _metrics(scf)
            ratio_keys = (
                "negative_hb_active_area_angstrom2",
                "positive_hb_active_area_angstrom2",
                "negative_hb_excess_area_e",
                "positive_hb_excess_area_e",
                "area_weighted_sigma_second_moment_e2_per_angstrom2",
            )
            artifact["records"].append(
                {
                    "compound_id": original["compound_id"],
                    "name": original["name"],
                    "frozen": frozen_metrics,
                    "scf": scf_metrics,
                    "scf_to_frozen_ratio": {
                        key: _ratio(scf_metrics, frozen_metrics, key)
                        for key in ratio_keys
                    },
                }
            )
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
    artifact["status"] = (
        "complete"
        if len(artifact["records"]) == len(TARGETS) and not artifact["failures"]
        else "complete-with-failures"
    )
    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
