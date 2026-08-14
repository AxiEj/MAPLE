from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "route2" / "evidence" / "mace-conjugacy-nogo-d17c35ac"
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "d17c35acb3de71e9780741d57b476de3f589d9d7"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "d89b620ed31cfdc0dfd7a89d03551251008458fe62e60c4eacda020255ab6909"
FILE_SHA256 = {
    "measurements.json": (
        "3fcc90a01f6fb76d1ddf9a60ad296c8d07602baea9ed24c76a56a5af60b77bb6"
    ),
    "cold-replay.json": (
        "95d46a0d09427aadf53666448d74dd414739113beda19c3a68ece81082b7cf47"
    ),
    "runner.log": ("63f3e8904fb2c3990ccf9f68f51442c017ca45ef7385e4877e2b7baf7e467b20"),
    "cold-replay.log": (
        "185226694e37b0b25a73a5a066c388e450d6d68afb97791c929db45f0d7ff893"
    ),
}
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_state_is_a_decisive_negative_canary(state: dict[str, object]) -> None:
    assert state["capabilities"] == NO_CAPABILITIES
    assert state["intrinsic_energy_directional_fd_gate_passed"] is False

    ad = state["intrinsic_energy_directional_ad"]
    assert isinstance(ad, dict)
    assert ad["implementation_consistent"] is True
    assert ad["status"] == "post-preregistration-implementation-diagnostic"

    source_jvp_vjp = state["source_jvp_vjp"]
    assert isinstance(source_jvp_vjp, dict)
    assert source_jvp_vjp["gate_passed"] is True

    analysis = state["analysis"]
    assert isinstance(analysis, dict)
    assert analysis["contract_version"] == "route2-mace-conjugacy-nogo-v1"
    assert analysis["no_go_witness_detected"] is True

    missing = analysis["missing_subspace"]
    assert isinstance(missing, dict)
    assert missing["no_go_witness_detected"] is True
    full = missing["full"]
    reduced = missing["gauge_reduced"]
    assert isinstance(full, dict)
    assert isinstance(reduced, dict)
    assert full["numerically_zero"] is False
    assert reduced["numerically_zero"] is False
    assert float(full["relative_l2"]) > 0.7
    assert float(reduced["relative_l2"]) > 0.7
    assert float(full["absolute_l2"]) > 1.0e8 * float(full["threshold"])
    assert float(reduced["absolute_l2"]) > 1.0e8 * float(reduced["threshold"])

    for family in ("direct", "intrinsic_stationarity"):
        signed = analysis[family]
        assert isinstance(signed, dict)
        assert set(signed) == {"-1", "1"}
        for result in signed.values():
            assert isinstance(result, dict)
            assert result["all_allowed_directions_zero"] is False
            assert result["gauge_direction_zero"] is False
            for space in ("full", "gauge_reduced"):
                residual = result[space]
                assert isinstance(residual, dict)
                assert residual["numerically_zero"] is False

    reciprocity = analysis["gauge_reduced_reciprocity"]
    assert isinstance(reciprocity, dict)
    assert reciprocity["reciprocal"] is False
    assert float(reciprocity["relative_frobenius"]) > 1.0


def test_real_checkpoint_conjugacy_nogo_is_source_bound_and_fail_closed():
    artifact = _load(PRIMARY)

    assert artifact["schema_version"] == (
        "route2-mace-conjugacy-nogo-real-checkpoint-v2"
    )
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-tier-v-conjugacy-nogo-canary"
    )
    assert artifact["status"] == "no-go-witness-detected"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES

    decision = artifact["decision"]
    assert decision == {
        "forward_reverse_ad_consistent_for_all_states": True,
        "frozen_energy_fd_gate_passed_for_all_states": False,
        "no_go_witness_detected": True,
        "original_energy_original_source_common_scalar": (
            "formally-ruled-out-by-counterexample"
        ),
        "tier_v_admitted": False,
    }

    states = artifact["states"]
    assert isinstance(states, dict)
    assert set(states) == {"zero-field", "nonzero-deterministic-field"}
    for state in states.values():
        assert isinstance(state, dict)
        _assert_state_is_a_decisive_negative_canary(state)

    measured = {
        "protocol": artifact["protocol"],
        "states": artifact["states"],
        "decision": artifact["decision"],
    }
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_real_checkpoint_conjugacy_nogo_cold_replay_is_exact():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)

    for artifact in (primary, replay):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["working_tree_clean"] is True
        assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)

    for key in ("protocol", "states", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]

    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected


def test_real_checkpoint_logs_preserve_runtime_warnings():
    for name in ("runner.log", "cold-replay.log"):
        log = (EVIDENCE / name).read_text(encoding="utf-8")
        assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in log
        assert "converting models to float64" in log
        assert "Cuequivariance acceleration will be disabled" in log
        assert '"no_go_witness_detected": true' in log
        assert '"tier_v_admitted": false' in log
