from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SCORER_PATH = BENCHMARK_DIR / "score_multi_mlip_phase_specific_qrrho.py"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_phase_specific_qrrho_protocol.json"

spec = importlib.util.spec_from_file_location("route1_qrrho_scoring", SCORER_PATH)
assert spec is not None and spec.loader is not None
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def test_paired_bootstrap_uses_paired_compound_resampling():
    result = scoring.paired_bootstrap_mae_improvement(
        baseline_kcal_mol=np.asarray([2.0, -2.0, 2.0]),
        qrrho_kcal_mol=np.asarray([0.0, 0.0, 0.0]),
        experimental_kcal_mol=np.asarray([0.0, 0.0, 0.0]),
        resamples=1000,
        random_seed=20260725,
    )

    assert result["point_improvement_kcal_mol"] == pytest.approx(2.0)
    assert result["bootstrap_lower_kcal_mol"] == pytest.approx(2.0)
    assert result["bootstrap_upper_kcal_mol"] == pytest.approx(2.0)
    assert len(result["distribution_sha256"]) == 64


def _synthetic_energy_artifact(protocol):
    records = []
    for model_index, model in enumerate(protocol["models"]):
        for case in protocol["cases"]:
            primary = 0.0 if model_index < 2 else 2.0
            variants = {
                variant_id: {"hydration_prediction_kcal_mol": primary}
                for variant_id in [
                    "primary",
                    "grimme-entropy-only",
                    "qrrho-otlyotov-omega100",
                    "qrrho-omega50",
                    "qrrho-omega150",
                ]
            }
            records.append(
                {
                    "model": model["name"],
                    "compound_id": case["compound_id"],
                    "prediction": {
                        "single_reference_endpoint_baseline_kcal_mol": 2.0,
                        "primary_hydration_prediction_kcal_mol": primary,
                        "variants": variants,
                    },
                }
            )
    return {"records": records}


def test_preregistered_scoring_gates_are_applied_without_posthoc_choices():
    protocol, _fingerprint, _manifest = scoring.load_qrrho_protocol(PROTOCOL_PATH)
    artifact = _synthetic_energy_artifact(protocol)
    labels = {
        case["compound_id"]: {"experimental_kcal_mol": 0.0}
        for case in protocol["cases"]
    }

    result = scoring.score_predictions(
        protocol=protocol,
        energy_artifact=artifact,
        labels=labels,
    )

    assert result["route_gates"]["accuracy"]["passed"] is True
    assert result["route_gates"]["accuracy"]["positive_model_count"] == 2
    assert result["route_gates"]["cross_model_direction"]["agreement_case_count"] == 6
    assert result["route_gates"]["cross_model_direction"]["passed"] is True
    assert result["route_gates"]["low_frequency_stability"]["passed"] is True
    assert result["route_passed_all_preregistered_development_gates"] is True
    assert result["decision"] == (
        "eligible-only-for-preregistered-larger-heldout-study"
    )

    first_model = protocol["models"][0]["name"]
    for case in protocol["cases"][:3]:
        entry = next(
            record
            for record in artifact["records"]
            if record["model"] == first_model
            and record["compound_id"] == case["compound_id"]
        )
        entry["prediction"]["variants"]["qrrho-omega50"][
            "hydration_prediction_kcal_mol"
        ] = 2.0

    unstable = scoring.score_predictions(
        protocol=protocol,
        energy_artifact=artifact,
        labels=labels,
    )
    assert unstable["route_gates"]["low_frequency_stability"]["passed"] is False
    assert unstable["route_passed_all_preregistered_development_gates"] is False
    assert unstable["decision"] == "development-falsification-failed-no-promotion"


def test_score_command_cannot_open_labels_before_seal_validation(
    tmp_path,
    monkeypatch,
):
    invalid_artifact = tmp_path / "incomplete.json"
    invalid_artifact.write_text("{}\n", encoding="utf-8")
    label_accessed = False

    def forbidden_label_access(**_kwargs):
        nonlocal label_accessed
        label_accessed = True
        raise AssertionError("Labels were opened before seal validation.")

    monkeypatch.setattr(scoring, "_load_labels_after_seal", forbidden_label_access)
    monkeypatch.setattr(scoring, "SCRIPT_DIR", tmp_path)
    args = type(
        "Args",
        (),
        {
            "protocol": str(PROTOCOL_PATH),
            "energy_artifact": str(invalid_artifact),
            "output": str(tmp_path / "score.json"),
        },
    )()

    with pytest.raises(ValueError, match="content SHA256"):
        scoring.score_command(args)
    assert label_accessed is False


def test_semantically_incomplete_self_hashed_records_cannot_unlock_labels(
    tmp_path,
    monkeypatch,
):
    protocol, fingerprint, manifest = scoring.load_qrrho_protocol(PROTOCOL_PATH)
    raw_root = tmp_path / "artifact-raw"
    record_dir = raw_root / "records"
    record_dir.mkdir(parents=True)
    entries = []
    for model in protocol["models"]:
        for case in protocol["cases"]:
            record = scoring.seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": (
                        "route1-multi-mlip-phase-specific-qrrho-model-case"
                    ),
                    "protocol_id": protocol["protocol_id"],
                    "protocol_fingerprint": fingerprint,
                    "source_partition": "development",
                    "status": "success",
                    "model": model,
                    "compound_id": case["compound_id"],
                    "name": case["name"],
                    "flexibility_bin": case["flexibility_bin"],
                    "claim_boundary": protocol["claim_boundary"],
                    "prediction": {},
                }
            )
            path = record_dir / f"{model['name']}--{case['compound_id']}.json"
            scoring.write_json_atomic(path, record)
            entries.append(
                {
                    "model": model["name"],
                    "compound_id": case["compound_id"],
                    "record": f"artifact-raw/records/{path.name}",
                    "record_file_sha256": scoring.sha256_file(path),
                    "record_content_sha256": record["content_sha256"],
                    "prediction": {},
                }
            )
    artifact = scoring.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": ("route1-multi-mlip-phase-specific-qrrho-energy-artifact"),
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": fingerprint,
            "source_partition": "development",
            "status": "sealed-before-label-scoring",
            "source_manifest_content_sha256": manifest["content_sha256"],
            "durable_raw_root": "artifact-raw",
            "durable_record_schema_version": 1,
            "model_case_count": len(entries),
            "records": entries,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    artifact_path = tmp_path / "artifact.json"
    scoring.write_json_atomic(artifact_path, artifact)
    monkeypatch.setattr(scoring, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(scoring, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        scoring,
        "_resolve_repository_file",
        lambda path, *, name: tmp_path / path,
    )
    label_accessed = False

    def forbidden_label_access(**_kwargs):
        nonlocal label_accessed
        label_accessed = True
        raise AssertionError("Labels were opened before semantic record validation.")

    monkeypatch.setattr(scoring, "_load_labels_after_seal", forbidden_label_access)
    args = type(
        "Args",
        (),
        {
            "protocol": str(PROTOCOL_PATH),
            "energy_artifact": str(artifact_path),
            "output": str(tmp_path / "score.json"),
        },
    )()

    with pytest.raises(ValueError, match="Invalid sealed model-case record"):
        scoring.score_command(args)
    assert label_accessed is False
