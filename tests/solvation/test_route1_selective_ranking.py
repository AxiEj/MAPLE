from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import seal_artifact  # pyright: ignore[reportMissingImports]
from selective_ranking import (  # pyright: ignore[reportMissingImports]
    STATUS_CERTIFIED,
    STATUS_OUT_OF_DOMAIN,
    STATUS_UNCERTAIN,
    SelectiveRankCalibrator,
    selective_rank_prediction,
    split_conformal_radius,
    validate_calibration_artifact,
)


PROTOCOL_PATH = BENCHMARK_DIR / "route1_selective_ranking_protocol_v1.json"
PHYSDISTILL_PROTOCOL_PATH = BENCHMARK_DIR / "route1_physdistill_protocol_v1.json"


def _calibration_artifact(
    allowed_prediction_evidence_sha256: list[str] | None = None,
):
    if allowed_prediction_evidence_sha256 is None:
        allowed_prediction_evidence_sha256 = [
            str(_prediction_evidence()["content_sha256"])
        ]
    return seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-selective-ranking-calibration",
            "protocol_id": "route1-selective-ranking-v1",
            "source_energy_artifact_sha256": "a" * 64,
            "boundary": {
                "energy_correction_applied": False,
                "energy_residual_fit": False,
                "endpoint_selection": False,
                "calibrated_before_test_labels": True,
                "series_disjoint_from_test": True,
                "external_independent_series": True,
            },
            "calibration_parameters": {
                "miscoverage": 0.25,
                "minimum_series_per_domain": 3,
                "series_error_statistic": (
                    "maximum_absolute_pair_error_per_complete_series"
                ),
            },
            "provenance": {
                "calibration_series_manifest_sha256": "b" * 64,
                "held_out_test_series_manifest_sha256": "c" * 64,
                "held_out_test_energy_artifact_sha256": "2" * 64,
                "series_disjointness_audit_sha256": "d" * 64,
                "domain_schema_sha256": "e" * 64,
                "endpoint_fingerprint_sha256": "f" * 64,
                "pair_disagreement_audit_protocol_sha256": "1" * 64,
                "prediction_evidence_manifest_sha256": "4" * 64,
                "held_out_test_series_ids": ["test-series-1", "test-series-2"],
                "allowed_prediction_evidence_sha256": (
                    allowed_prediction_evidence_sha256
                ),
            },
            "domain_keys": ["charge_class", "size_bin"],
            "series_records": [
                {
                    "series_id": "cal-neutral-small-1",
                    "domain": {"charge_class": "neutral", "size_bin": "small"},
                    "maximum_pair_error_kcal_mol": 0.2,
                },
                {
                    "series_id": "cal-neutral-small-2",
                    "domain": {"charge_class": "neutral", "size_bin": "small"},
                    "maximum_pair_error_kcal_mol": 0.3,
                },
                {
                    "series_id": "cal-neutral-small-3",
                    "domain": {"charge_class": "neutral", "size_bin": "small"},
                    "maximum_pair_error_kcal_mol": 0.4,
                },
            ],
        }
    )


def _calibrator(*evidence_records) -> SelectiveRankCalibrator:
    if not evidence_records:
        evidence_records = (_prediction_evidence(),)
    return SelectiveRankCalibrator(
        _calibration_artifact(
            [str(evidence["content_sha256"]) for evidence in evidence_records]
        )
    )


def _prediction_evidence(
    *,
    delta: float = -0.6,
    domain: dict[str, object] | None = None,
    series_id: str = "test-series-1",
    charge_order_disagreement: bool = False,
    endpoint_order_disagreement: bool = False,
):
    return seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-selective-ranking-prediction-evidence",
            "series_id": series_id,
            "first_id": "ligand-a",
            "second_id": "ligand-b",
            "domain": domain
            or {"charge_class": "neutral", "size_bin": "small"},
            "predicted_delta_kcal_mol": delta,
            "held_out_test_series_manifest_sha256": "c" * 64,
            "test_energy_artifact_sha256": "2" * 64,
            "endpoint_fingerprint_sha256": "f" * 64,
            "domain_schema_sha256": "e" * 64,
            "pair_disagreement_audit_protocol_sha256": "1" * 64,
            "pair_disagreement_audit_artifact_sha256": "3" * 64,
            "charge_order_disagreement": charge_order_disagreement,
            "endpoint_order_disagreement": endpoint_order_disagreement,
            "experimental_test_labels_loaded": False,
            "energy_correction_applied": False,
        }
    )


def test_current_selective_ranking_protocol_is_explicitly_unactivated():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["status"] == "blocked_no_qualified_calibration"
    assert protocol["current_free_solv_scaffold_diagnostic_allowed_as_calibration"] is False
    assert protocol["route1_boundary"]["energy_correction_applied"] is False
    assert protocol["route1_boundary"]["energy_residual_fit"] is False
    assert protocol["allowed_rank_status"] == [
        STATUS_CERTIFIED,
        STATUS_UNCERTAIN,
        STATUS_OUT_OF_DOMAIN,
    ]


def test_physdistill_is_research_only_and_rejects_experimental_residuals():
    protocol = json.loads(PHYSDISTILL_PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["status"] == "research_only_unimplemented"
    boundary = protocol["route1_boundary"]
    assert boundary["experimental_hydration_residual_target"] is False
    assert boundary["direct_unstructured_total_energy_residual"] is False
    assert boundary["conservative_force_required"] is True
    assert protocol["teacher_boundary"]["training_and_validation_separation_required"]
    assert "all_3N_finite_difference_force_parity" in protocol[
        "required_pre_promotion_gates"
    ]


def test_split_conformal_radius_uses_complete_series_order_statistic():
    assert split_conformal_radius([0.2, 0.3, 0.4], miscoverage=0.25) == 0.4
    with pytest.raises(ValueError, match="non-negative"):
        split_conformal_radius([-0.1], miscoverage=0.25)
    with pytest.raises(ValueError, match="too few independent series"):
        split_conformal_radius([0.2, 0.3, 0.4], miscoverage=0.1)


def test_selective_ranking_refuses_unattainable_finite_sample_coverage():
    evidence = _prediction_evidence(delta=-1.0)
    artifact = _calibration_artifact([str(evidence["content_sha256"])])
    artifact["calibration_parameters"]["miscoverage"] = 0.1
    artifact = seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    calibrator = SelectiveRankCalibrator(artifact)
    prediction = selective_rank_prediction(
        calibrator=calibrator,
        evidence=evidence,
    )

    assert prediction["rank_status"] == STATUS_OUT_OF_DOMAIN
    assert prediction["rank_prediction"] is None
    assert prediction["ood_reasons"] == ["insufficient-exact-domain-calibration"]


def test_calibration_requires_independent_series_and_separate_energy_reference():
    artifact = _calibration_artifact()
    normalized = validate_calibration_artifact(artifact)
    assert normalized["source_energy_artifact_sha256"] == "a" * 64

    artifact["boundary"]["series_disjoint_from_test"] = False
    artifact = seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="scientific boundary"):
        validate_calibration_artifact(artifact)


def test_calibration_binds_quantile_endpoint_domain_and_disjoint_test_ids():
    artifact = _calibration_artifact()
    artifact["provenance"]["held_out_test_series_ids"].append(
        "cal-neutral-small-1"
    )
    artifact = seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="series IDs overlap"):
        validate_calibration_artifact(artifact)

    artifact = _calibration_artifact()
    artifact["provenance"]["endpoint_fingerprint_sha256"] = "not-a-digest"
    artifact = seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="provenance is invalid"):
        validate_calibration_artifact(artifact)


def test_selective_ranking_certifies_only_outside_the_conformal_interval():
    evidence = _prediction_evidence(delta=-0.6)
    prediction = selective_rank_prediction(
        calibrator=_calibrator(evidence),
        evidence=evidence,
    )

    assert prediction["rank_status"] == STATUS_CERTIFIED
    assert prediction["rank_prediction"] == "first_better"
    assert prediction["rank_interval"] == pytest.approx([-1.0, -0.2])
    assert prediction["energy_correction_applied"] is False
    assert prediction["rank_coverage"]["series_count"] == 3

    uncertain_evidence = _prediction_evidence(delta=-0.2)
    uncertain = selective_rank_prediction(
        calibrator=_calibrator(uncertain_evidence),
        evidence=uncertain_evidence,
    )
    assert uncertain["rank_status"] == STATUS_UNCERTAIN
    assert uncertain["rank_prediction"] is None
    assert uncertain["ood_reasons"] == ["interval-crosses-zero"]


def test_disagreement_forces_uncertain_and_unseen_domain_refuses():
    disagreement_evidence = _prediction_evidence(
        charge_order_disagreement=True
    )
    disagreement = selective_rank_prediction(
        calibrator=_calibrator(disagreement_evidence),
        evidence=disagreement_evidence,
    )
    assert disagreement["rank_status"] == STATUS_UNCERTAIN
    assert disagreement["rank_prediction"] is None
    assert disagreement["ood_reasons"] == ["charge-order-disagreement"]

    unseen_evidence = _prediction_evidence(
        domain={"charge_class": "charged", "size_bin": "small"}
    )
    unseen = selective_rank_prediction(
        calibrator=_calibrator(unseen_evidence),
        evidence=unseen_evidence,
    )
    assert unseen["rank_status"] == STATUS_OUT_OF_DOMAIN
    assert unseen["rank_prediction"] is None
    assert unseen["ood_reasons"] == ["insufficient-exact-domain-calibration"]

    unavailable = selective_rank_prediction(
        calibrator=None,
        evidence=None,
    )
    assert unavailable["rank_status"] == STATUS_OUT_OF_DOMAIN
    assert unavailable["ood_reasons"] == ["no-qualified-series-disjoint-calibration"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint_fingerprint_sha256", "9" * 64),
        ("domain_schema_sha256", "8" * 64),
        ("held_out_test_series_manifest_sha256", "7" * 64),
        ("test_energy_artifact_sha256", "6" * 64),
    ],
)
def test_prediction_refuses_mismatched_sealed_provenance(
    field: str,
    value: str,
):
    evidence = _prediction_evidence()
    evidence[field] = value
    evidence = seal_artifact(
        {key: item for key, item in evidence.items() if key != "content_sha256"}
    )

    prediction = selective_rank_prediction(
        calibrator=_calibrator(),
        evidence=evidence,
    )

    assert prediction["rank_status"] == STATUS_OUT_OF_DOMAIN
    assert prediction["ood_reasons"] == ["prediction-provenance-mismatch"]


@pytest.mark.parametrize("series_id", ["cal-neutral-small-1", "unknown-series"])
def test_prediction_refuses_calibration_or_unregistered_series(series_id: str):
    evidence = _prediction_evidence(series_id=series_id)
    prediction = selective_rank_prediction(
        calibrator=_calibrator(evidence),
        evidence=evidence,
    )

    assert prediction["rank_status"] == STATUS_OUT_OF_DOMAIN
    assert prediction["ood_reasons"] == ["prediction-provenance-mismatch"]


def test_prediction_refuses_resealed_delta_or_pair_substitution():
    calibrator = _calibrator()
    for field, value in (
        ("predicted_delta_kcal_mol", -100.0),
        ("first_id", "forged-ligand"),
    ):
        evidence = _prediction_evidence()
        evidence[field] = value
        evidence = seal_artifact(
            {
                key: item
                for key, item in evidence.items()
                if key != "content_sha256"
            }
        )

        prediction = selective_rank_prediction(
            calibrator=calibrator,
            evidence=evidence,
        )

        assert prediction["rank_status"] == STATUS_OUT_OF_DOMAIN
        assert prediction["ood_reasons"] == ["prediction-provenance-mismatch"]
