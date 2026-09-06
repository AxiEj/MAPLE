from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "tools"
    / "route2_release"
    / "analyze_canonical_adt_paired_partial.py"
)
SPEC = importlib.util.spec_from_file_location("paired_adt_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records(tmp_path: Path) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    common = {
        "selection_index": 0,
        "dataset_row_sha256": "a" * 64,
        "geometry_sha256": "b" * 64,
        "opaque_record_id": "c" * 64,
        "selection_score_sha256": "d" * 64,
        "canonical_solvent": "water",
        "experimental_delta_g_kcal_mol": -5.0,
        "mdp_checkpoint_sha256": "e" * 64,
        "polar_checkpoint_sha256": "f" * 64,
        "partition": "development",
        "confirmation_partition_opened": False,
        "status": "pass",
        "smd_cds_kcal_mol": 1.0,
    }
    baseline = {
        **common,
        "artifact": MODULE.BASELINE_ARTIFACT,
        "continuum_polarization_kcal_mol": -5.5,
        "predicted_delta_g_kcal_mol": -4.5,
    }
    candidate = {
        **common,
        "artifact": MODULE.CANDIDATE_ARTIFACT,
        "continuum_polarization_kcal_mol": -5.4,
        "m1_electrostatic_plus_stock_smd_cds": {
            "predicted_delta_g_kcal_mol": -4.4,
        },
    }
    (baseline_dir / "index-000.json").write_text(json.dumps(baseline))
    (candidate_dir / "index-000.json").write_text(json.dumps(candidate))
    return baseline_dir, candidate_dir


def test_paired_analysis_separates_m0_from_unchanged_cds(tmp_path: Path) -> None:
    baseline_dir, candidate_dir = _records(tmp_path)
    result = MODULE.analyze(baseline_dir, candidate_dir)
    assert result["matched_record_count"] == 1
    assert result["paired_change"]["m0_mae_improvement_kcal_mol"] == pytest.approx(0.1)
    assert result["paired_change"]["m1_mae_improvement_kcal_mol"] == pytest.approx(-0.1)
    assert result["paired_change"][
        "stock_smd_cds_maximum_absolute_difference_kcal_mol"
    ] == pytest.approx(0.0)
    assert result["confirmation_partition_opened"] is False


def test_paired_analysis_rejects_identity_or_cds_drift(tmp_path: Path) -> None:
    baseline_dir, candidate_dir = _records(tmp_path)
    candidate_path = candidate_dir / "index-000.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["geometry_sha256"] = "0" * 64
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="geometry_sha256"):
        MODULE.analyze(baseline_dir, candidate_dir)

    baseline_dir, candidate_dir = _records(tmp_path / "second")
    candidate_path = candidate_dir / "index-000.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["smd_cds_kcal_mol"] = 1.2
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="SMD-CDS drift"):
        MODULE.analyze(baseline_dir, candidate_dir)
