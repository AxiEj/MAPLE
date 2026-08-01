from __future__ import annotations

import json
from pathlib import Path

import pytest
from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-macepolar-frozen-source-direct-pcm-multisolvent-"
    "pilot-v1-execution-ae427ea7.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_frozen_source_mnsol_pilot_is_public_aggregate_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-mnsol-macepolar-frozen-source-direct-pcm-" "multisolvent-pilot-v1"
    )
    assert artifact["execution_git_head"] == (
        "ae427ea7537c5d50673d99a566a250642b45707b"
    )
    assert artifact["visibility"] == "public-aggregate-only"
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
    assert coverage["class_count"] == 10
    assert len(set(coverage["classes"])) == 10
    assert coverage["status"] == "post-selection-descriptive"
    assert coverage["used_for_selection"] is False
    assert artifact["selection_record_count"] == 10
    assert artifact["selection_solvent_count"] == 10

    expected = {
        "ddpcm": {
            "mae": 0.8635007355065989,
            "rmse": 0.9902395026416793,
            "maximum": 1.6454379135383834,
        },
        "ddcosmo": {
            "mae": 0.9154387617956268,
            "rmse": 1.020783151682132,
            "maximum": 1.6450882111475043,
        },
    }
    for method, reference in expected.items():
        metrics = artifact["aggregate_metrics"][method]
        assert metrics["record_count"] == 10
        assert metrics["response_mode"] == "frozen"
        assert metrics["fixed_point_applicable"] is False
        assert metrics["mean_scf_iterations"] is None
        assert metrics["maximum_scf_iterations"] is None
        assert metrics["maximum_unmixed_density_residual_e"] is None
        assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
            reference["mae"], abs=1.0e-12
        )
        assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
            reference["rmse"], abs=1.0e-12
        )
        assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
            reference["maximum"], abs=1.0e-12
        )
        assert metrics["maximum_absolute_error_kcal_mol"] >= 1.5

    identity = artifact["scientific_identity"]
    contract = artifact["frozen_source_contract"]
    assert identity["polarization_response"] == "frozen"
    assert identity["mutual_ml_continuum_polarization"] is False
    assert identity["electrostatic_energy_ledger"] == ("pcm-half-coupling-only-v1")
    assert contract["field_conditioned_model_evaluated"] is False
    assert contract["fixed_point_applicable"] is False
    assert artifact["scf_parameters"] is None

    paired = artifact["paired_method_comparison"]
    assert paired["record_count"] == 10
    assert paired["ddpcm_lower_absolute_error_count"] == 5
    assert paired["ddcosmo_lower_absolute_error_count"] == 5

    assert_source_files_match_execution_commit(ROOT, artifact)
