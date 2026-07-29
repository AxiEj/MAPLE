from __future__ import annotations

import json

import pytest

from maple.function.benchmarking import (
    BenchmarkIdentity,
    BenchmarkResultStore,
    LeakageStatus,
    ModelTrainingEvidence,
    audit_training_overlap,
    paired_comparison,
    summarize_predictions,
)


def _identity(**overrides):
    values = {
        "dataset": "FreeSolv",
        "dataset_version": "0.52",
        "record_ids": ("mobley_1", "mobley_2"),
        "temperature_kelvin": 298.15,
        "standard_state": "1M",
        "protonation_policy": "dataset",
        "tautomer_policy": "dataset",
        "conformer_policy": "dataset-3d",
        "geometry_protocol": "fixed-input",
        "solvent_protocol": "water",
        "potential": "example-pes-v1",
        "solvation_backend": "example-solvent-v1",
        "cavity_model": "provider-native",
        "sampling_protocol": "single-point",
        "estimator": "energy-difference",
        "experimental_provenance": "FreeSolv database value",
    }
    values.update(overrides)
    return BenchmarkIdentity(**values)


def _records(predictions=(0.0, 2.0)):
    return [
        {
            "record_id": "mobley_1",
            "experimental_kcal_mol": 1.0,
            "predicted_kcal_mol": predictions[0],
        },
        {
            "record_id": "mobley_2",
            "experimental_kcal_mol": 3.0,
            "predicted_kcal_mol": predictions[1],
        },
    ]


def test_identity_is_stable_and_changes_with_any_protocol_dimension():
    identity = _identity()
    assert len(identity.fingerprint) == 64
    assert identity.fingerprint == _identity().fingerprint
    assert identity.fingerprint != _identity(cavity_model="different").fingerprint
    assert identity.fingerprint != _identity(potential="alternative").fingerprint
    assert identity.panel_fingerprint == _identity(potential="alternative").panel_fingerprint
    assert identity.panel_fingerprint != _identity(solvent_protocol="methanol").panel_fingerprint


def test_identity_rejects_duplicate_or_implicit_records():
    with pytest.raises(ValueError, match="duplicate"):
        _identity(record_ids=("same", "same"))
    with pytest.raises(ValueError, match="non-empty"):
        _identity(estimator="")


def test_leakage_audit_is_conservative():
    identity = _identity()
    overlap = audit_training_overlap(
        identity,
        ModelTrainingEvidence(
            known_training_record_ids=("mobley_2",),
            evidence_source="training manifest",
        ),
    )
    assert overlap.status is LeakageStatus.KNOWN_OVERLAP
    assert overlap.overlapping_record_ids == ("mobley_2",)

    unknown = audit_training_overlap(identity, ModelTrainingEvidence())
    assert unknown.status is LeakageStatus.OVERLAP_UNKNOWN

    holdout = audit_training_overlap(
        identity,
        ModelTrainingEvidence(
            explicitly_excluded_datasets=("FreeSolv",),
            record_accounting_complete=True,
            evidence_source="signed split manifest",
        ),
    )
    assert holdout.status is LeakageStatus.STRICT_HOLDOUT


def test_named_training_dataset_without_record_manifest_is_not_known_overlap():
    audit = audit_training_overlap(
        _identity(),
        ModelTrainingEvidence(training_datasets=("FreeSolv",)),
    )
    assert audit.status is LeakageStatus.OVERLAP_UNKNOWN


def test_metrics_include_coverage_correlations_and_seeded_bootstrap():
    summary = summarize_predictions(_records(), bootstrap_samples=200, seed=7)
    assert summary["coverage"] == 1.0
    assert summary["metrics"]["mae_kcal_mol"] == pytest.approx(1.0)
    assert summary["metrics"]["rmse_kcal_mol"] == pytest.approx(1.0)
    assert summary["metrics"]["spearman_rho"] == pytest.approx(1.0)
    assert summary["metrics"]["kendall_tau_b"] == pytest.approx(1.0)
    assert summary["mae_bootstrap_95_ci_kcal_mol"] == pytest.approx([1.0, 1.0])


def test_metrics_count_nonfinite_predictions_as_failures():
    records = _records()
    records[1]["predicted_kcal_mol"] = None
    summary = summarize_predictions(records, bootstrap_samples=0)
    assert summary["evaluated_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["coverage"] == 0.5


def test_paired_comparison_requires_panel_identity_match():
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _identity(),
            _records(),
            _identity(dataset="MNSol"),
            _records(),
        )


def test_paired_comparison_requires_complete_successful_record_matrix():
    with pytest.raises(ValueError, match="exactly one successful"):
        paired_comparison(_identity(), _records()[:1], _identity(), _records())


def test_paired_comparison_reports_wins_and_delta():
    comparison = paired_comparison(
        _identity(),
        _records((1.0, 2.5)),
        _identity(potential="alternative"),
        _records((0.0, 4.0)),
    )
    assert comparison.record_count == 2
    assert comparison.wins_a == 2
    assert comparison.mean_delta_absolute_error_kcal_mol == pytest.approx(-0.75)
    assert comparison.benchmark_fingerprint == _identity().panel_fingerprint
    assert comparison.panel_fingerprint == _identity().panel_fingerprint
    assert comparison.run_fingerprint_a == _identity().fingerprint
    assert comparison.run_fingerprint_b == _identity(
        potential="alternative"
    ).fingerprint


def test_paired_comparison_rejects_solvent_protocol_or_backend_mismatch():
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _identity(),
            _records(),
            _identity(solvation_backend="different-backend"),
            _records(),
        )
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _identity(),
            _records(),
            _identity(solvent_protocol="octanol"),
            _records(),
        )


def test_result_store_is_path_safe_and_requires_full_artifact_set(tmp_path):
    with pytest.raises(ValueError, match="path-safe"):
        BenchmarkResultStore(tmp_path, "../escape")

    store = BenchmarkResultStore(tmp_path, "model-a")
    with pytest.raises(RuntimeError, match="incomplete"):
        store.assert_complete()
    with pytest.raises(ValueError, match="path-safe"):
        store.write_json("../escape.json", {"unsafe": True})
    with pytest.raises(ValueError, match="path-safe"):
        store.write_csv(
            "nested/escape.csv",
            [],
            fieldnames=("record_id",),
        )

    store.write_json_yaml("model_card.yaml", {"model_id": "model-a"})
    store.write_json_yaml("protocol.yaml", {"fingerprint": _identity().fingerprint})
    store.write_environment_lock("python==3.11")
    store.write_csv(
        "per_record.csv",
        _records(),
        fieldnames=("record_id", "experimental_kcal_mol", "predicted_kcal_mol"),
    )
    store.write_csv(
        "failures.csv",
        [],
        fieldnames=("record_id", "reason"),
    )
    store.write_json("diagnostics.json", {"status": "ok"})
    store.write_csv(
        "uncertainty.csv",
        [],
        fieldnames=("record_id", "uncertainty_kcal_mol"),
    )
    store.write_json("leakage_report.json", {"status": "overlap_unknown"})
    store.write_json("speed.json", {"energy_evaluations_per_second": 1.0})
    store.write_json("summary.json", {"mae_kcal_mol": 1.0})
    store.assert_complete()

    model_card = json.loads(
        (store.directory / "model_card.yaml").read_text(encoding="utf-8")
    )
    assert model_card["model_id"] == "model-a"
