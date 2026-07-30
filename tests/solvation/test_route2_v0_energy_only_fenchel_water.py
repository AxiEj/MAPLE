from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"


def test_energy_only_fenchel_preregistration_forbids_posthoc_concavity_repairs():
    preregistration = json.loads(
        (BENCHMARKS / "route2-v0-energy-only-fenchel-water-prereg-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preregistration["status"] == "preregistered-before-execution"
    assert "MACE density output" in " ".join(preregistration["prohibited_actions"])
    assert "clip, rescale, symmetrize" in " ".join(
        preregistration["prohibited_actions"]
    )
    assert (
        preregistration["gates"]["neutral_hessian_antisymmetric_frobenius_ratio_max"]
        == 1.0e-8
    )
    assert (
        preregistration["gates"]["neutral_hessian_maximum_eigenvalue_ev_max"] == 1.0e-8
    )
    assert "rejected" in preregistration["stop_condition"]
