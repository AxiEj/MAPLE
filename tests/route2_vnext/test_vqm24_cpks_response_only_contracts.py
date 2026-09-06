from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "docs/route2/evidence/vqm24-cpks-response-only-pilot-20260827"
COVERAGE_PREREGISTRATION = ROOT / (
    "docs/route2/preregistrations/vqm24-cpks-response-only-coverage-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_cpks_response_only_pilot_narrows_target_without_rewriting_gate_b() -> None:
    payload = json.loads((PILOT / "result.json").read_text())
    protocol = json.loads((PILOT / "protocol.json").read_text())
    assert payload["status"] == "pass-response-only-pilot"
    assert payload["aggregate"]["pass_count"] == 4
    assert payload["aggregate"]["maximum_mep_symmetric_relative"] == pytest.approx(
        6.194549829016144e-05, abs=0.0
    )
    assert payload["decision"] == {
        "cpks_generator_admitted": False,
        "energy_curvature_target_authorized": False,
        "full_32_record_coverage_preregistration_authorized": True,
        "model_training_authorized": False,
        "new_QM_response_generated": False,
        "response_only_pilot_passed": True,
    }
    assert protocol["excluded_gate_names"] == ["curvature_absolute"]
    assert "curvature_absolute" not in protocol["required_gate_names"]
    assert all(
        record["excluded_diagnostic"] == {"curvature_absolute": passed}
        for record, passed in zip(
            payload["records"], [True, True, False, True], strict=True
        )
    )
    historical_sealer = PILOT / "source/seal_vqm24_cpks_response_only_pilot.py"
    assert _sha256(historical_sealer) == protocol["source"]["sealer_sha256"]
    assert payload["claim_boundary"]["new_QM_calculation_run"] is False


def test_cpks_response_only_coverage_is_frozen_for_all_train_records() -> None:
    payload = json.loads(COVERAGE_PREREGISTRATION.read_text())
    assert payload["artifact"] == (
        "route2-vqm24-cpks-response-only-coverage-prereg-v1"
    )
    assert payload["status"] == "locked-before-coverage-execution"
    expected = payload.pop("preregistration_sha256")
    assert _canonical_sha256(payload) == expected
    assert len(payload["records"]) == 32
    assert len({record["record_id"] for record in payload["records"]}) == 32
    assert payload["decision"]["energy_curvature_target_or_gate"] is False
    assert payload["decision"]["all_32_records_must_pass_every_gate"] is True
    for name in ("record_runner", "sealer"):
        assert _sha256(ROOT / payload["source"][f"{name}_path"]) == payload[
            "source"
        ][f"{name}_sha256"]
    assert _sha256(ROOT / payload["source"]["cpks_runner_path"]) == payload[
        "source"
    ]["cpks_runner_sha256"]

