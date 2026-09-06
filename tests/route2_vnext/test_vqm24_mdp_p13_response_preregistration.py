from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-mdp-p13-block-passive-response-pilot-v1.json"
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


def test_vqm24_p13_response_pilot_was_frozen_before_training() -> None:
    payload = json.loads(PREREGISTRATION.read_text())
    assert payload["artifact"] == (
        "route2-vqm24-mdp-p13-block-passive-response-pilot-prereg-v1"
    )
    assert payload["status"] == "locked-before-training"
    expected = payload.pop("preregistration_sha256")
    assert _canonical_sha256(payload) == expected
    assert len(payload["train_record_ids"]) == 32
    split = payload["selection"]["record_sha256s_by_split"]
    assert {name: len(values) for name, values in split.items()} == {
        "train": 32,
        "validation": 14,
        "blind": 14,
    }
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["blind"]))
    assert not (set(split["validation"]) & set(split["blind"]))
    assert payload["candidate_sequence"] == ["A2", "A3"]
    assert payload["architecture"]["parameter_cap"] == 2500
    assert payload["optimizer"]["audit_points_used_in_gradient"] is False
    assert payload["selection"]["validation_and_blind_data_unopened"] is True
    forbidden = set(payload["inputs"]["forbidden_targets"])
    assert "experimental solvation energies" in forbidden
    assert "original MACE-POLAR source or response" in forbidden
    for name in ("factor", "radial_coupling", "l2_coupling"):
        entry = payload["source"][name]
        assert _sha256(ROOT / entry["path"]) == entry["sha256"]
    # The response module intentionally advanced after this frozen pilot when
    # the exact analytic O2 derivative backend was added.  Historical identity
    # remains bound by the preregistration digest rather than being rewritten.
    response = payload["source"]["response"]
    assert response["sha256"] == (
        "58f996f442f85fedc48e801ca745e585cf120f48adf22a7b61d2a4f05fb9d360"
    )
    assert _sha256(ROOT / response["path"]) != response["sha256"]
    assert _sha256(ROOT / payload["source"]["training_runner_path"]) == payload[
        "source"
    ]["training_runner_sha256"]
