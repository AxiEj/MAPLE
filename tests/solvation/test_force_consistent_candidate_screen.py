from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks"
    / "route1-force-consistent-candidate-screen-2026-07-24.json"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _content_sha256(artifact: dict) -> str:
    payload = {
        key: value for key, value in artifact.items() if key != "content_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_candidate_screen_is_nonfitted_route1_development_evidence():
    artifact = _artifact()

    assert artifact["schema_version"] == 1
    assert artifact["content_sha256"] == _content_sha256(artifact)
    assert artifact["case_count"] == 526
    assert artifact["source_evidence"]["source_record_count"] == 526 * 5
    assert artifact["metric_definitions"]["mse_kcal_mol"] == (
        "mean signed error in kcal/mol; not mean-squared error"
    )
    assert artifact["command_provenance"]["script_sha256"] == hashlib.sha256(
        (
            REPOSITORY_ROOT
            / "docs/implicit-solvation/benchmarks"
            / "run_force_consistent_candidate_screen.py"
        ).read_bytes()
    ).hexdigest()
    assert artifact["route1_boundary"] == {
        "fixed_am1bcc_charges": True,
        "gas_phase_mm_energy": False,
        "retraining": False,
        "hydration_label_residual": False,
    }
    assert artifact["label_use_boundary"]["candidate_parameters_fit_here"] is False
    assert artifact["decision"]["product_default_changed"] is False


def test_neutral_alpb_has_no_material_obc2_accuracy_gain():
    artifact = _artifact()
    alpb = artifact["candidates"]["neutral_alpb"]["models"]["obc2"]

    assert alpb["case_count"] == 526
    assert alpb["scale_factor"] == pytest.approx(0.9927734691268697)
    assert alpb["baseline"]["mae_kcal_mol"] == pytest.approx(1.7603512076636878)
    assert alpb["candidate"]["mae_kcal_mol"] == pytest.approx(1.7580030028156772)
    assert alpb["paired_mae_gain_kcal_mol"] == pytest.approx(0.00234820484801287)
    assert alpb["paired_mae_gain_bootstrap_ci_kcal_mol"][0] < 0.0
    assert alpb["decision"] == "reject-product-default"


def test_gbn2_and_lcpo_fail_accuracy_or_applicability_gates():
    artifact = _artifact()
    gbn2 = artifact["candidates"]["gbn2"]
    lcpo = artifact["candidates"]["lcpo"]

    assert gbn2["success_count"] == 515
    assert gbn2["failure_count"] == 11
    assert gbn2["mae_kcal_mol"] > artifact["baseline_obc2_ace"]["mae_kcal_mol"]
    assert gbn2["same_515_case_comparison"]["obc2_mae_kcal_mol"] == pytest.approx(
        1.6614874200887932
    )
    assert gbn2["same_515_case_comparison"][
        "paired_mae_change_gbn2_minus_obc2_kcal_mol"
    ] == pytest.approx(0.2324960736424576)
    assert gbn2["decision"] == "reject-product-default"

    assert lcpo["supported_count"] == 454
    assert lcpo["unsupported_count"] == 72
    assert lcpo["coverage_fraction"] == pytest.approx(454 / 526)
    supported = lcpo["same_supported_case_comparison"]
    assert supported["baseline_obc2_ace"]["mae_kcal_mol"] == pytest.approx(
        1.8007923374290695
    )
    assert supported["candidate_obc2_lcpo"]["mae_kcal_mol"] == pytest.approx(
        2.2502427688773343
    )
    assert supported["paired_mae_gain_ace_minus_lcpo_kcal_mol"] == pytest.approx(
        -0.44945043144826485
    )
    assert supported["paired_mae_gain_bootstrap_ci_kcal_mol"][1] < 0.0
    assert supported["case_outcomes"] == {
        "improved": 169,
        "unchanged": 0,
        "worsened": 285,
    }
    assert supported["baseline_obc2_ace"]["mse_kcal_mol"] == pytest.approx(
        sum(record["baseline_signed_error_kcal_mol"] for record in supported["records"])
        / supported["case_count"]
    )
    assert supported["candidate_obc2_lcpo"]["mse_kcal_mol"] == pytest.approx(
        sum(
            record["candidate_signed_error_kcal_mol"]
            for record in supported["records"]
        )
        / supported["case_count"]
    )
    for record in supported["records"]:
        polar = record["obc2_polar_kcal_mol"]
        assert record["baseline_obc2_ace_kcal_mol"] == pytest.approx(
            polar + record["ace_nonpolar_kcal_mol"],
            abs=1.0e-10,
        )
        assert record["candidate_obc2_lcpo_kcal_mol"] == pytest.approx(
            polar + record["lcpo_nonpolar_kcal_mol"],
            abs=1.0e-10,
        )
        assert (
            record["candidate_obc2_lcpo_kcal_mol"]
            - record["baseline_obc2_ace_kcal_mol"]
        ) == pytest.approx(
            record["lcpo_nonpolar_kcal_mol"]
            - record["ace_nonpolar_kcal_mol"],
            abs=1.0e-10,
        )
    assert supported["maximum_obc2_polar_difference_kcal_mol"] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert lcpo["decision"] == "reject-product-default"
