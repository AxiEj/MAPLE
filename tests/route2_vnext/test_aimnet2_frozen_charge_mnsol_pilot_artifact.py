from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-mnsol-pilot-replay-b758aede"
)
HISTORICAL = (
    ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "route2-mnsol-aimnet2-multisolvent-pilot-v1.json"
)
EXECUTION_HEAD = "b758aede9d52693c2e45412c5f4582eee4ef0300"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
SELECTION_FINGERPRINT = (
    "967979795b3bd1483db93f7b463f9a65f3f2c0859f69228100d2adc107138fba"
)
DATASET_BUNDLE_SHA256 = (
    "6465a65a024cd06872cb9812381184ed6e9b1a528adfe12be9d5d43b4aec75d8"
)
METRIC_KEYS = (
    "maximum_absolute_error_kcal_mol",
    "maximum_half_coupling_identity_error_ev",
    "mean_absolute_error_kcal_mol",
    "mean_experimental_delta_g_kcal_mol",
    "mean_polarization_energy_kcal_mol",
    "mean_predicted_delta_g_kcal_mol",
    "mean_signed_error_kcal_mol",
    "mean_smd_cds_energy_kcal_mol",
    "record_count",
    "root_mean_square_error_kcal_mol",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksums() -> dict[str, str]:
    return {
        name: digest
        for digest, name in (
            line.split(maxsplit=1)
            for line in (EVIDENCE / "SHA256SUMS")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }


def _assert_public_accuracy_artifact(artifact: dict[str, object]) -> None:
    assert artifact["artifact"] == "route2-mnsol-aimnet2-multisolvent-pilot-v1"
    assert artifact["schema_version"] == 1
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["complete_panel"] is True
    assert artifact["do_not_commit"] is False
    assert artifact["run_kind"] == "ten-record-panel"
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["selection_fingerprint"] == SELECTION_FINGERPRINT
    assert artifact["selection_record_count"] == 10
    assert artifact["selection_solvent_count"] == 10
    assert artifact["selection_partition_counts"] == {
        "confirmation": 8,
        "development": 2,
    }
    assert artifact["dataset"]["normalized_bundle_sha256"] == DATASET_BUNDLE_SHA256
    assert artifact["dataset"]["standard_state"] == (
        "1M-ideal-gas-to-1M-ideal-solution"
    )
    assert artifact["dataset"]["row_level_data_emitted"] is False

    identity = artifact["scientific_identity"]
    assert identity["polarization_response"] == "fixed"
    assert identity["mutual_ml_continuum_polarization"] is False
    assert identity["solute_source"] == "AIMNet2 NQE point-charge-l0"
    assert identity["nonpolar_model"] == "PySCF 2.13.1 SMD-CDS"
    assert identity["energy_composition"] == (
        "DeltaG_solv = U_polarization(fixed AIMNet2 charges) + G_SMD-CDS"
    )

    continuum = artifact["continuum_parameters"]
    assert continuum["lmax"] == 15
    assert continuum["n_lebedev"] == 1202
    assert continuum["eta"] == pytest.approx(0.1, rel=0.0, abs=0.0)
    assert continuum["solver_tolerance"] == pytest.approx(1.0e-12)
    assert continuum["standard_state_correction_kcal_mol"] == 0.0
    assert artifact["charge_quality"] == {
        "maximum_absolute_projected_charge_sum_e": 1.1102230246251565e-16,
        "maximum_absolute_raw_charge_residual_e": 1.4901161193847656e-07,
    }
    assert len(artifact["source_files_sha256"]) == 17
    assert not ({"records", "rows", "geometries"} & set(artifact))


def test_current_frozen_charge_mnsol_replay_checksums_and_source_binding():
    checksums = _checksums()
    expected = {path.name for path in EVIDENCE.iterdir() if path.name != "SHA256SUMS"}
    assert set(checksums) == expected
    for name, digest in checksums.items():
        assert _sha256(EVIDENCE / name) == digest

    primary = _load(EVIDENCE / "primary-public.json")
    replay = _load(EVIDENCE / "replay-public.json")
    _assert_public_accuracy_artifact(primary)
    _assert_public_accuracy_artifact(replay)
    assert_source_files_match_execution_commit(ROOT, primary)


def test_current_frozen_charge_mnsol_scientific_aggregates_replay_exactly():
    primary = _load(EVIDENCE / "primary-public.json")
    replay = _load(EVIDENCE / "replay-public.json")
    for key in (
        "aggregate_metrics",
        "charge_output_parameters",
        "charge_quality",
        "checkpoint",
        "continuum_parameters",
        "dataset",
        "paired_method_comparison",
        "protocol_fingerprint",
        "protocol_id",
        "scientific_identity",
        "selection_fingerprint",
        "selection_indices",
        "selection_partition_counts",
    ):
        assert replay[key] == primary[key]

    assert primary["aggregate_metrics"]["ddpcm"] == {
        "maximum_absolute_error_kcal_mol": 2.23223982326141,
        "maximum_half_coupling_identity_error_ev": 3.2751579226442118e-15,
        "mean_absolute_error_kcal_mol": 1.1430625083225296,
        "mean_experimental_delta_g_kcal_mol": -4.817,
        "mean_polarization_energy_kcal_mol": -2.4714575076033327,
        "mean_predicted_delta_g_kcal_mol": -3.783687273368531,
        "mean_signed_error_kcal_mol": 1.0333127266314686,
        "mean_smd_cds_energy_kcal_mol": -1.3122297657651987,
        "record_count": 10,
        "root_mean_square_error_kcal_mol": 1.3515629726548866,
    }
    assert primary["aggregate_metrics"]["ddcosmo"] == {
        "maximum_absolute_error_kcal_mol": 2.1932768162931726,
        "maximum_half_coupling_identity_error_ev": 3.4139358007223564e-15,
        "mean_absolute_error_kcal_mol": 1.0120975743588152,
        "mean_experimental_delta_g_kcal_mol": -4.817,
        "mean_polarization_energy_kcal_mol": -2.674766737317466,
        "mean_predicted_delta_g_kcal_mol": -3.986996503082665,
        "mean_signed_error_kcal_mol": 0.8300034969173351,
        "mean_smd_cds_energy_kcal_mol": -1.3122297657651987,
        "record_count": 10,
        "root_mean_square_error_kcal_mol": 1.274316391038506,
    }


def test_current_frozen_charge_mnsol_reproduces_historical_metrics_without_relabeling():
    current = _load(EVIDENCE / "primary-public.json")
    historical = _load(HISTORICAL)
    assert current["checkpoint"] == historical["checkpoint"]
    assert current["dataset"] == historical["dataset"]
    assert current["selection_fingerprint"] == historical["selection_fingerprint"]
    assert current["scientific_identity"] == historical["scientific_identity"]
    for method in ("ddpcm", "ddcosmo"):
        for key in METRIC_KEYS:
            assert current["aggregate_metrics"][method][key] == pytest.approx(
                historical["aggregate_metrics"][method][key],
                rel=0.0,
                abs=5.0e-12,
            )
