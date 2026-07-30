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


def test_energy_only_fenchel_artifact_rejects_the_frozen_checkpoint_before_pcm():
    artifact = json.loads(
        (BENCHMARKS / "route2-v0-energy-only-fenchel-water-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert artifact["status"] == "fail"
    assert artifact["failed_gates"] == ["neutral_hessian_concavity"]
    assert artifact["gates"]["passed"] == {
        "neutral_hessian_concavity": False,
        "neutral_hessian_reciprocity": True,
    }
    assert (
        artifact["energy_scalar"]["neutral_hessian_maximum_eigenvalue_ev"]
        > artifact["gates"]["thresholds"]["neutral_hessian_maximum_eigenvalue_ev_max"]
    )
    assert artifact["scientific_identity"]["mace_density_output_used"] is False
    assert (
        artifact["scientific_identity"]["experimental_solvation_labels_read"] is False
    )
    assert artifact["scientific_identity"]["continuum_fixed_point_solved"] is False
