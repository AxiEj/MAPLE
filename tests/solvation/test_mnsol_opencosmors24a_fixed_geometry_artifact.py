from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-opencosmors24a-fixed-geometry-v1.json"
)
AGGREGATOR_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "aggregate_mnsol_opencosmors24a_fixed_geometry.py"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_opencosmors_artifact_is_complete_private_safe_and_dual_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == ("route2-mnsol-opencosmors24a-fixed-geometry-v1")
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["do_not_commit"] is False
    assert artifact["complete_panel"] is True
    assert artifact["record_execution_git_head"] == (
        "6421f18db70f626fa710cb72c4bed0f8cd15b396"
    )
    assert artifact["aggregation_git_head"] == (
        "eb5b7e1402be3a4e9fcd5a5fc36b7185052a6667"
    )
    assert artifact["preregistration"]["sha256"] == (
        "67924ba47e286d976cf0dc9f93d3dd2f2fa73768b71fd7e9535b1a35fb976234"
    )
    assert artifact["selection"]["record_count"] == 10
    assert artifact["selection"]["solvent_count"] == 10
    assert artifact["experimental_reference"]["doi"] == "10.13020/3eks-j059"
    assert artifact["experimental_reference"]["row_level_data_emitted"] is False
    assert artifact["scientific_identity"]["training_overlap_status"] == (
        "known-overlap-training-domain-reproduction"
    )
    assert (
        artifact["scientific_identity"]["strict_published_24a_geometry_workflow"]
        is False
    )

    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "opencosmors_delta_g_kcal_mol",
        "input_bundle",
        "orca_path",
        "opencosmors_path",
        "records",
    }
    assert forbidden.isdisjoint(_all_keys(artifact))

    sources = artifact["source_files_sha256"]
    for relative_path, digest in sources.items():
        commit = (
            artifact["aggregation_git_head"]
            if relative_path == AGGREGATOR_PATH
            else artifact["record_execution_git_head"]
        )
        assert _git_blob_sha256(commit, relative_path) == digest


def test_opencosmors_aggregate_metrics_and_paired_results_are_immutable():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    metrics = artifact["aggregate_metrics"]

    assert metrics["record_count"] == 10
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
        0.6460055,
        abs=1.0e-12,
    )
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
        0.8037994501282641,
        abs=1.0e-12,
    )
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(
        0.5791647,
        abs=1.0e-12,
    )
    assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
        1.508105,
        abs=1.0e-12,
    )
    assert metrics["mean_wall_seconds"] == pytest.approx(
        25.80324181060132,
        abs=1.0e-12,
    )

    expected = {
        "aimnet2_fixed_l0__pyscf_swig_iefpcm": (1.1297309378064637, 8, 2),
        "aimnet2_fixed_l0__pyscf_swig_cpcm": (1.0395967316091395, 6, 4),
        "aimnet2_fixed_l0__pyscf_swig_cosmo": (1.1142242850072284, 7, 3),
        "mace_fixed_l1__pyscf_swig_iefpcm": (0.8780065988654894, 7, 3),
        "mace_fixed_l1__pyscf_swig_cpcm": (0.9127298962577448, 7, 3),
        "mace_fixed_l1__pyscf_swig_cosmo": (0.8737904414643213, 7, 3),
    }
    comparisons = artifact["paired_method_comparisons"]
    for method_id, (baseline_mae, wins, losses) in expected.items():
        comparison = comparisons[method_id]
        reconstructed_baseline = (
            metrics["mean_absolute_error_kcal_mol"]
            - comparison["mean_opencosmors_minus_baseline_absolute_error_kcal_mol"]
        )
        assert reconstructed_baseline == pytest.approx(
            baseline_mae,
            abs=1.0e-12,
        )
        assert comparison["opencosmors_lower_absolute_error_count"] == wins
        assert comparison["absolute_error_tie_count"] == 0
        assert comparison["opencosmors_higher_absolute_error_count"] == losses


def test_opencosmors_fragment_provenance_covers_each_index_once():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fragments = artifact["fragment_provenance"]

    indices = [
        index for fragment in fragments for index in fragment["selection_indices"]
    ]
    assert sorted(indices) == list(range(10))
    assert len(indices) == len(set(indices))
    assert sum(fragment["record_count"] for fragment in fragments) == 10
    assert all(len(fragment["sha256"]) == 64 for fragment in fragments)
    assert all(
        fragment["execution_git_head"] == artifact["record_execution_git_head"]
        for fragment in fragments
    )
    assert artifact["timing_seconds"]["summed_record_wall"] == pytest.approx(
        258.0324181060132,
        abs=1.0e-12,
    )
    assert artifact["timing_seconds"]["aggregation_overhead_included"] is False
