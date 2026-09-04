from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PRIMARY_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-diverse-v3.json"
FROZEN_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-frozen-source-ablation-v2.json"
CROSS_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-response-cross-v1.json"
PREREGISTRATION_PATH = (
    BENCHMARKS / "mace-ef-cosmors-freesolv20-response-cross-prereg-v1.json"
)
BUNDLE_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
BUNDLE_MANIFEST_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-profile-bundle-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_clean_replay_is_source_bound_and_scientifically_fail_closed():
    artifact = _load(PRIMARY_PATH)

    assert _sha256(PRIMARY_PATH) == (
        "a23c847d7e6ec8f18662d205edaaf36a0d5e0d6ea3a863c6481a38c515b6c83f"
    )
    assert artifact["artifact"] == "mace-ef-cosmors-freesolv20-diverse-v3"
    assert artifact["status"] == "complete"
    assert artifact["scientific_status"] == "complete-negative-result"
    assert artifact["admission_eligible"] is False
    assert artifact["diagnostic_only"] is True
    assert artifact["failed_gates"] == [
        "mace_ef_uniform_field_passivity_all_records",
        "open24a_surface_parameterization_equivalence",
    ]
    assert artifact["scientific_gates"] == {
        "accuracy_gate_preregistered": False,
        "mace_ef_uniform_field_passivity_all_records": False,
        "open24a_surface_parameterization_equivalence": False,
        "runtime_complete": True,
    }
    assert len(artifact["electronic_passivity_failures"]) == 15
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_primary_metrics_reconstruct_from_all_twenty_records():
    artifact = _load(PRIMARY_PATH)
    records = artifact["records"]

    assert len(records) == 20
    assert artifact["failures"] == []
    errors = np.asarray(
        [
            float(record["predicted_kcal_mol"] - record["experimental_kcal_mol"])
            for record in records
        ]
    )
    summary = artifact["summary"]
    assert summary["count"] == 20
    assert summary["mae_kcal_mol"] == pytest.approx(np.mean(np.abs(errors)))
    assert summary["rmse_kcal_mol"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        np.max(np.abs(errors))
    )
    assert summary["mae_kcal_mol"] == pytest.approx(9.19267882897089)
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        52.250202031629286
    )
    assert all(record["model_result"]["workflow_diagnostic_only"] for record in records)
    assert all(len(record["generated_xyz"]["sha256"]) == 64 for record in records)
    assert all(len(record["mace_ef_profile"]["sha256"]) == 64 for record in records)
    assert all(
        not Path(record["mace_ef_profile"]["path"]).is_absolute() for record in records
    )


def test_frozen_control_is_paired_source_bound_and_not_total_component_accuracy():
    primary = _load(PRIMARY_PATH)
    artifact = _load(FROZEN_PATH)

    assert _sha256(FROZEN_PATH) == (
        "e020795286af6b56a38fcdc1fe441de8328d87265b32a78416dbdb560162556d"
    )
    assert artifact["status"] == "complete"
    assert artifact["scientific_status"] == "complete-mechanism-diagnostic"
    assert artifact["admission_eligible"] is False
    assert artifact["primary_artifact"]["sha256"] == _sha256(PRIMARY_PATH)
    assert_source_files_match_execution_commit(ROOT, artifact)
    assert [record["compound_id"] for record in artifact["records"]] == [
        record["compound_id"] for record in primary["records"]
    ]
    assert all(
        record["frozen_dielectric_mechanism_only"]["not_total_hydration_free_energy"]
        for record in artifact["records"]
    )
    assert artifact["summaries"]["frozen_full_open24a"][
        "mae_kcal_mol"
    ] == pytest.approx(2.1134715081829016)
    assert artifact["summaries"]["frozen_no_hydrogen_bond"][
        "mae_kcal_mol"
    ] == pytest.approx(1.7705160084852092)
    assert (
        "not total hydration accuracy"
        in artifact["scf_response_shift"]["interpretation"]
    )


def test_response_cross_replays_preregistered_arms_and_exact_input_artifacts():
    artifact = _load(CROSS_PATH)
    preregistration = _load(PREREGISTRATION_PATH)

    assert _sha256(CROSS_PATH) == (
        "f34a4c70a0f996d05ed69136f897302bb4aaa7297b88c20aedb5fcc5da49247b"
    )
    assert artifact["status"] == "complete"
    assert artifact["scientific_status"] == "complete-mechanism-diagnostic"
    assert artifact["admission_eligible"] is False
    assert artifact["diagnostic_only"] is True
    assert artifact["failures"] == []
    assert len(artifact["records"]) == 20
    assert artifact["preregistration"]["sha256"] == _sha256(PREREGISTRATION_PATH)
    assert artifact["primary_artifact"]["sha256"] == _sha256(PRIMARY_PATH)
    assert artifact["frozen_control_artifact"]["sha256"] == _sha256(FROZEN_PATH)
    assert_source_files_match_execution_commit(ROOT, artifact)
    for record in artifact["records"]:
        assert set(record["arms"]) == set(preregistration["arms"])


def test_response_cross_contrasts_reconstruct_without_post_hoc_relabelling():
    artifact = _load(CROSS_PATH)

    for record in artifact["records"]:
        arms = record["arms"]

        def h(solute: str, water: str) -> float:
            prefix = f"{solute}_solute__{water}_water"
            return float(arms[f"{prefix}__full"]["predicted_kcal_mol"]) - float(
                arms[f"{prefix}__no_hydrogen_bond"]["predicted_kcal_mol"]
            )

        h_ff = h("frozen", "frozen")
        h_sf = h("self_consistent", "frozen")
        h_fs = h("frozen", "self_consistent")
        h_ss = h("self_consistent", "self_consistent")
        contrasts = record["contrasts"]
        assert contrasts["solute_response_at_frozen_water_kcal_mol"] == pytest.approx(
            h_sf - h_ff
        )
        assert contrasts[
            "solute_response_at_self_consistent_water_kcal_mol"
        ] == pytest.approx(h_ss - h_fs)
        assert contrasts["water_response_at_frozen_solute_kcal_mol"] == pytest.approx(
            h_fs - h_ff
        )
        assert contrasts[
            "water_response_at_self_consistent_solute_kcal_mol"
        ] == pytest.approx(h_ss - h_sf)
        assert contrasts["nonlinear_interaction_kcal_mol"] == pytest.approx(
            h_ss - h_sf - h_fs + h_ff
        )

    summaries = artifact["contrast_summaries"]
    assert summaries["solute_response_at_frozen_water_kcal_mol"][
        "mean_absolute_kcal_mol"
    ] == pytest.approx(4.025901912889331)
    assert summaries["water_response_at_frozen_solute_kcal_mol"][
        "mean_absolute_kcal_mol"
    ] == pytest.approx(0.7383990624420544)
    assert summaries["nonlinear_interaction_kcal_mol"][
        "mean_absolute_kcal_mol"
    ] == pytest.approx(2.441591764084246)
    assert summaries["full_total_scf_minus_frozen_kcal_mol"][
        "mean_absolute_kcal_mol"
    ] == pytest.approx(7.853806166483793)
    assert summaries["no_hydrogen_bond_total_scf_minus_frozen_kcal_mol"][
        "mean_absolute_kcal_mol"
    ] == pytest.approx(2.0345699818027576)


def test_response_cross_freezes_twenty_record_tail_metrics_and_passivity_stop():
    artifact = _load(CROSS_PATH)

    required = {
        "area_weighted_sigma_second_moment_e2_per_angstrom2",
        "area_weighted_sigma_fourth_moment_e4_per_angstrom6",
        "negative_hb_active_area_angstrom2",
        "positive_hb_active_area_angstrom2",
        "negative_hb_excess_area_e",
        "positive_hb_excess_area_e",
    }
    for record in artifact["records"]:
        tails = record["tail_metrics"]
        assert required.issubset(tails["frozen_zero_field"])
        assert required.issubset(tails["self_consistent"])
    assert artifact["electronic_passivity"]["passed_count"] == 5
    assert artifact["electronic_passivity"]["failed_count"] == 15
    assert artifact["electronic_passivity"]["scf_route_admission_eligible"] is False
    assert artifact["interpretation"]["hard_stop_applied"] is True
    assert artifact["interpretation"]["automatic_root_cause_verdict"] is None


def test_profile_bundle_is_deterministic_complete_and_source_bound():
    manifest = _load(BUNDLE_MANIFEST_PATH)

    assert _sha256(BUNDLE_MANIFEST_PATH) == (
        "14d9a9436e6e837bd766507c43d2cf5b1879dd1d8c0118c4fe754a7cddea85a7"
    )
    assert manifest["status"] == "complete"
    assert manifest["scientific_result"] is False
    assert manifest["archive"] == {
        "format": "deterministic-ustar-gzip-mtime-zero",
        "member_count": 63,
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
        ),
        "sha256": "989688eca71577c11fa623c599adbf4538394e9be3841613d73df36218b356c3",
    }
    assert _sha256(BUNDLE_PATH) == manifest["archive"]["sha256"]
    assert manifest["input_artifacts"]["primary"]["sha256"] == _sha256(PRIMARY_PATH)
    assert manifest["input_artifacts"]["frozen_control"]["sha256"] == _sha256(
        FROZEN_PATH
    )
    assert manifest["input_artifacts"]["response_cross"]["sha256"] == _sha256(
        CROSS_PATH
    )
    assert_source_files_match_execution_commit(ROOT, manifest)

    with tarfile.open(BUNDLE_PATH, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(manifest["member_sha256"])
        assert len(members) == 63
        for member in members:
            assert member.isfile()
            assert member.mtime == 0
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert (
                hashlib.sha256(extracted.read()).hexdigest()
                == manifest["member_sha256"][member.name]
            )
