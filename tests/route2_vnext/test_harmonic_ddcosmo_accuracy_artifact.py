from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/route2-mnsol10-harmonic-ddcosmo-preregistration-v1.json"
)
ARTIFACT = (
    ROOT
    / "docs/route2/evidence/mace-polar-point-l1-harmonic-ddcosmo-mnsol10-accuracy-v1.json"
)


def test_frozen_harmonic_ddcosmo_mnsol10_artifact_passes_its_preregistered_gate():
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    metrics = artifact["aggregate_metrics"]

    assert artifact["status"] == "pass"
    assert artifact["record_count"] == 10
    assert metrics["record_count"] == 10
    assert metrics["solvent_count"] == 10
    assert metrics["target_mae_kcal_mol"] == pytest.approx(1.5, abs=0.0)
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
        0.9194258753853435, rel=0.0, abs=1.0e-12
    )
    assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
        1.647941711352706, rel=0.0, abs=1.0e-12
    )
    assert artifact["gates"] == {
        "complete_frozen_panel": True,
        "mae_at_most_1_5_kcal_mol": True,
        "source_charge": True,
        "ten_distinct_solvents": True,
    }
    assert artifact["harmonic_ddcosmo_preregistration"] == {
        "artifact": PREREGISTRATION.name,
        "sha256": digest,
    }
    assert preregistration["scientific_target"]["maximum"] == 1.5
    assert preregistration["profile"]["surface_lmax"] == 5
    assert preregistration["profile"]["partition_lmax"] == 10
