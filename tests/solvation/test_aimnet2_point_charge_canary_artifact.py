from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-aimnet2-point-charge-ddpcm-canary-v1.json"
)

def test_aimnet2_point_charge_canary_is_source_bound_and_claim_bounded():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-aimnet2-point-charge-ddpcm-canary-v1"
    )
    assert artifact["execution_git_head"] == (
        "37c3ed715c63c80045c2362a1580fdb503321344"
    )
    identity = artifact["scientific_identity"]
    assert identity["solute_source"] == "AIMNet2 NQE point-charge-l0"
    assert identity["polarization_response"] == "fixed"
    assert identity["continuum"] == "pyddx ddPCM"
    assert identity["cds_included"] is False
    assert identity["total_solvation_free_energy_reported"] is False
    assert identity["mutual_ml_continuum_polarization"] is False
    assert artifact["checkpoint"]["sha256"] == (
        "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
    )
    assert artifact["checkpoint"]["redistributed"] is False
    assert artifact["charge_output_parameters"] == {
        "padded_sentinel_tolerance_e": 1.0e-6,
        "projection": "uniform-affine-float-residue-only",
        "raw_total_charge_tolerance_e": 1.0e-4,
    }
    assert artifact["continuum_parameters"][
        "full_profile_numerical_equivalence"
    ] is False
    assert artifact["continuum_parameters"][
        "reduced_canary_discretization"
    ] is True

    assert_source_files_match_execution_commit(ROOT, artifact)

    records = artifact["records"]
    assert [record["molecule"] for record in records] == [
        "H2O",
        "NH3",
        "CH4",
        "CH3COCH3",
    ]
    for record in records:
        assert abs(record["raw_charge_residual_e"]) <= 1.0e-4
        assert abs(record["projected_charge_sum_e"]) <= 1.0e-12
        assert record["half_coupling_identity_error_ev"] <= 1.0e-8
        assert np.isfinite(record["polarization_energy_hartree"])
        timing = record["timing_seconds"]
        assert timing["aimnet2_charge_median"] > 0.0
        assert timing["ddpcm_build"] > 0.0
        assert timing["ddpcm_solve"] > 0.0
