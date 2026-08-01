from __future__ import annotations

import json
from pathlib import Path

import pytest
from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-macepolar-direct-pcm-multisolvent-pilot-v2-"
    "execution-e1f8acb1.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_direct_pcm_mnsol_pilot_is_public_aggregate_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-mnsol-macepolar-direct-pcm-multisolvent-pilot-v2"
    )
    assert artifact["execution_git_head"] == (
        "e1f8acb1d886b289001640b97cc65dac50c1422a"
    )
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["do_not_commit"] is False
    assert artifact["complete_panel"] is True
    assert artifact["dataset"]["row_level_data_emitted"] is False
    assert "records" not in artifact
    assert {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "experimental_delta_g_kcal_mol",
        "density_coefficients",
    }.isdisjoint(_all_keys(artifact))

    coverage = artifact["selection_functional_group_coverage"]
    assert coverage == {
        "class_count": 10,
        "classes": [
            "halogenated-hydrocarbon",
            "ketone",
            "aromatic-hydrocarbon",
            "nitro",
            "amide",
            "cyclic-diether",
            "phenol",
            "thiophenol",
            "alcohol",
            "carboxylic-acid",
        ],
        "status": "post-selection-descriptive",
        "used_for_selection": False,
    }
    assert artifact["selection_record_count"] == 10
    assert artifact["selection_solvent_count"] == 10

    expected = {
        "ddpcm": {
            "mae": 1.527666939836195,
            "rmse": 1.992072483595031,
            "mse": -1.1591246005708276,
            "maximum": 4.095270295007756,
        },
        "ddcosmo": {
            "mae": 1.8619346859141483,
            "rmse": 2.2917559495179995,
            "mse": -1.5337524493462995,
            "maximum": 4.295427677264431,
        },
    }
    for method, reference in expected.items():
        metrics = artifact["aggregate_metrics"][method]
        assert metrics["record_count"] == 10
        assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
            reference["mae"], abs=1.0e-12
        )
        assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
            reference["rmse"], abs=1.0e-12
        )
        assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(
            reference["mse"], abs=1.0e-12
        )
        assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
            reference["maximum"], abs=1.0e-12
        )
        assert metrics["maximum_absolute_error_kcal_mol"] >= 1.5
        assert metrics["maximum_half_coupling_identity_error_ev"] <= 3.0e-16

    paired = artifact["paired_method_comparison"]
    assert paired["record_count"] == 10
    assert paired["ddpcm_lower_absolute_error_count"] == 8
    assert paired["ddcosmo_lower_absolute_error_count"] == 2
    assert paired["absolute_error_tie_count"] == 0

    identity = artifact["scientific_identity"]
    assert identity["electrostatic_energy_ledger"] == "pcm-half-coupling-only-v1"
    assert identity["mutual_ml_continuum_polarization"] is True
    assert identity["nonpolar_model"] == "PySCF 2.13.1 SMD-CDS"
    assert identity["strict_original_smd_equivalence"] is False
    assert identity["cpcm_included"] is False
    assert identity["cosmo_rs_included"] is False
    assert "One record per solvent cannot certify accuracy" in artifact["claim_boundary"]

    assert_source_files_match_execution_commit(ROOT, artifact)
