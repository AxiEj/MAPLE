from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-macepolar-response-ablation-v1.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_all_keys(item) for item in value.values()),
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_mnsol_response_ablation_is_experimental_aggregate_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == ("route2-mnsol-macepolar-response-ablation-v1")
    assert artifact["execution_git_head"] == (
        "96dbb70003c487f1dd3d3507e4a8962fa8eca5f9"
    )
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["complete_panel"] is True
    assert artifact["run_kind"] == "ten-record-panel"
    assert "records" not in artifact
    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "aimnet2_charges_e",
        "mace_gas_density_coefficients",
    }
    assert forbidden.isdisjoint(_all_keys(artifact))

    experiment = artifact["experimental_reference"]
    assert experiment["dataset"] == "Minnesota Solvation Database"
    assert experiment["version"] == "2012"
    assert experiment["doi"] == "10.13020/3eks-j059"
    assert experiment["temperature_k"] == 298.0
    assert experiment["standard_state"] == ("1M-ideal-gas-to-1M-ideal-solution")
    assert experiment["row_level_data_emitted"] is False
    assert experiment["selected_record_checks"] == {
        "all_records_absolute_gas_to_solvent": True,
        "all_records_neutral": True,
        "all_values_finite": True,
        "subset_counts": {"[a]": 1, "[g]": 9},
    }
    assert experiment["table_sha256"] == (
        "6dba4397764d1ca665c5dac653b9963bd64f784c353311a72c15e42897b90156"
    )
    assert experiment["normalized_bundle_sha256"] == (
        "6465a65a024cd06872cb9812381184ed6e9b1a528adfe12be9d5d43b4aec75d8"
    )

    assert artifact["selection"]["record_count"] == 10
    assert artifact["selection"]["solvent_count"] == 10
    assert artifact["selection"]["used_experimental_values"] is False
    assert artifact["selection"]["used_model_outputs"] is False
    assert artifact["selection"]["partition_counts"] == {
        "confirmation": 8,
        "development": 2,
    }

    expected = {
        "aimnet2_fixed_l0": (1.143062508322529, 1.3515629726549894),
        "mace_fixed_l0": (3.3945757262053275, 3.7780524212518665),
        "mace_fixed_l1": (0.8635007355068123, 0.9902395026418327),
        "mace_one_shot_l1": (1.0392391268919141, 1.1906186306917923),
        "mace_scf_l1": (0.9905378109300497, 1.3224669770152702),
    }
    for method, (mae, rmse) in expected.items():
        metrics = artifact["aggregate_metrics"][method]
        assert metrics["record_count"] == 10
        assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
            mae,
            abs=1.0e-12,
        )
        assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
            rmse,
            abs=1.0e-12,
        )

    fixed_to_scf = artifact["paired_method_comparisons"][
        "mace_fixed_l1__to__mace_scf_l1"
    ]
    assert fixed_to_scf["left_lower_absolute_error_count"] == 6
    assert fixed_to_scf["right_lower_absolute_error_count"] == 4
    assert fixed_to_scf[
        "right_minus_left_mean_absolute_error_kcal_mol"
    ] == pytest.approx(0.1270370754232374, abs=1.0e-12)
    assert fixed_to_scf["right_minus_left_mean_energy_kcal_mol"] == pytest.approx(
        -0.813225537053923, abs=1.0e-12
    )

    identity = artifact["scientific_identity"]
    assert identity["shared_electrostatics"] == "pyddx ddPCM"
    assert identity["reaction_field_projector"] == "local-jet"
    assert identity["strict_original_smd_equivalence"] is False
    assert artifact["timing_seconds"]["status"] == (
        "metadata-only-not-a-randomized-speed-ranking"
    )
    assert_source_files_match_execution_commit(ROOT, artifact)
