from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1.json"
)


def test_aimnet2_continuum_equation_canary_is_orthogonal_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1"
    )
    assert artifact["schema_version"] == 1
    assert artifact["execution_git_head"] == (
        "6b3ba1e7e063cee13f890ccfb4a9f93e126a81ff"
    )
    assert artifact["checkpoint"]["sha256"] == (
        "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
    )
    identity = artifact["scientific_identity"]
    assert identity["solute_source"] == "AIMNet2 NQE point-charge-l0"
    assert identity["polarization_response"] == "fixed"
    assert identity["continuum_equations"] == [
        "pyddx ddPCM",
        "pyddx ddCOSMO",
    ]
    assert identity["orthogonal_comparison"] == (
        "same geometry/source/radii/dielectric/grid; continuum equation only"
    )
    assert identity["cds_included"] is False
    assert identity["total_solvation_free_energy_reported"] is False
    assert identity["mutual_ml_continuum_polarization"] is False
    assert identity["cosmo_rs_included"] is False
    assert "does not validate SMD-CDS" in artifact["claim_boundary"]
    assert "C-PCM" in artifact["claim_boundary"]
    assert "COSMO-RS" in artifact["claim_boundary"]

    parameters = artifact["continuum_parameters"]
    expected_scale = (
        parameters["dielectric"] - 1.0
    ) / parameters["dielectric"]
    assert parameters["cosmo_dielectric_scaling"] == pytest.approx(
        expected_scale
    )
    assert parameters["reduced_canary_discretization"] is True
    assert parameters["full_profile_numerical_equivalence"] is False

    assert_source_files_match_execution_commit(ROOT, artifact)

    records = artifact["records"]
    assert [record["molecule"] for record in records] == [
        "H2O",
        "CH3COCH3",
    ]
    expected_energies = {
        "H2O": {
            "ddpcm": -6.685055998258688,
            "ddcosmo": -6.722310055461649,
            "difference": -0.03725405720296049,
        },
        "CH3COCH3": {
            "ddpcm": -4.918674670945484,
            "ddcosmo": -4.947223246048024,
            "difference": -0.02854857510253961,
        },
    }
    for record in records:
        assert abs(record["raw_charge_residual_e"]) <= 1.0e-4
        assert abs(record["projected_charge_sum_e"]) <= 1.0e-12
        expected = expected_energies[record["molecule"]]
        for method, model, label, scaling in (
            ("ddpcm", "pcm", "ddPCM", 1.0),
            ("ddcosmo", "cosmo", "ddCOSMO", expected_scale),
        ):
            result = record["methods"][method]
            assert result["polarization_energy_kcal_mol"] == pytest.approx(
                expected[method],
                abs=1.0e-12,
            )
            assert result["half_coupling_identity_error_ev"] <= 1.0e-8
            assert np.isfinite(result["polarization_energy_hartree"])
            provenance = result["runtime_provenance"]
            assert provenance["model"] == model
            assert provenance["method"] == label
            assert provenance["dielectric_scaling"] == pytest.approx(
                scaling
            )
            assert result["timing_seconds"]["build"] > 0.0
            assert result["timing_seconds"]["solve"] > 0.0
        assert record["ddcosmo_minus_ddpcm_kcal_mol"] == pytest.approx(
            expected["difference"],
            abs=1.0e-12,
        )
