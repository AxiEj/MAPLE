from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1.json"
)


def _load() -> dict[str, object]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_full_matrix_artifact_is_complete_public_and_source_bound():
    artifact = _load()

    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "c0c9f316529fce87641922419fc500d827b9a5582a69ade73404a6d6f6ba056d"
    )
    assert artifact["artifact"] == (
        "route2-mnsol-aimnet2-full-frozen-charge-matrix-v1"
    )
    assert artifact["measurement_sha256"] == (
        "10b68d1fbbbc9eede7feefcb64daa2cb7be1bd0ab4a00186586eaf8965361039"
    )
    assert artifact["execution_git_head"] == (
        "2a33c2f39016f4c545ddf82eef0a2b6144710a24"
    )
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["complete_panel"] is True
    assert artifact["do_not_commit"] is False
    assert artifact["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert artifact["dataset"]["row_level_data_emitted"] is False
    assert "records" not in artifact
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_full_matrix_artifact_retains_accuracy_and_holdout_boundaries():
    artifact = _load()
    coverage = artifact["coverage"]
    assert coverage["eligible_record_count"] == 653
    assert coverage["completed_record_count"] == 653
    assert coverage["failed_record_count"] == 0
    assert coverage["unique_geometry_count"] == 395
    assert coverage["partition_record_counts"] == {
        "confirmation": 148,
        "development": 505,
    }
    assert coverage["untouched_confirmation_record_count"] == 102
    assert coverage["solvent_record_counts"]["water"] == 387

    metrics = artifact["aggregate_metrics"]
    all_records = metrics["all_records"]["methods"]
    assert all_records["ddpcm"]["mean_absolute_error_kcal_mol"] == pytest.approx(
        2.5442693466168476,
        abs=1.0e-12,
    )
    assert all_records["ddcosmo"]["mean_absolute_error_kcal_mol"] == pytest.approx(
        2.4392623938207794,
        abs=1.0e-12,
    )
    untouched = metrics["untouched_confirmation"]["methods"]
    assert untouched["ddpcm"]["mean_absolute_error_kcal_mol"] == pytest.approx(
        2.907465687221704,
        abs=1.0e-12,
    )
    assert untouched["ddcosmo"]["mean_absolute_error_kcal_mol"] == pytest.approx(
        2.7732506391019154,
        abs=1.0e-12,
    )
    water = metrics["by_solvent"]["water"]["methods"]
    assert water["ddpcm"]["mean_signed_error_kcal_mol"] == pytest.approx(
        3.383622998683818,
        abs=1.0e-12,
    )
    assert water["ddcosmo"]["mean_signed_error_kcal_mol"] == pytest.approx(
        3.3397256812796465,
        abs=1.0e-12,
    )

    identity = artifact["scientific_identity"]
    assert identity["continuum_field_supplied_to_aimnet2"] is False
    assert identity["electronic_scf_iteration"] is False
    assert identity["same_as_harmonic_force_scalar"] is False
    assert artifact["selection"]["selection_used_experimental_values"] is False
    assert artifact["selection"]["selection_used_model_outputs"] is False
