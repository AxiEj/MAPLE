from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs" / "implicit-solvation" / "benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import freeze_obc2_am1bcc_526_reference as freezer


@pytest.fixture(scope="module")
def bound_members():
    _manifest, bindings, aggregate = freezer._load_manifest_members()
    assert aggregate == freezer.EXPECTED_SAFE_MEMBER_BINDING_SHA256
    assert len(bindings) == 526
    return bindings


def test_raw_hash_is_allowed_but_json_loads_member_bytes_is_blocked(bound_members):
    binding = bound_members[0]
    record = freezer.ROOT / freezer.CORPUS_ROOT_REL / binding.record_relative_path
    assert freezer.core.sha256_file(record) == binding.record_sha256

    guard = freezer.SemanticDecoderGuard(bound_members)
    with guard.active():
        with pytest.raises(PermissionError, match="semantic decoder denied"):
            json.loads(record.read_bytes())

    assert len(guard.blocked) == 1
    assert guard.blocked[0]["api"] == "json.loads"
    assert guard.blocked[0]["sha256"] == binding.record_sha256


def test_json_load_member_path_is_blocked_while_safe_manifest_is_allowed(
    bound_members,
):
    binding = bound_members[0]
    record = freezer.ROOT / freezer.CORPUS_ROOT_REL / binding.record_relative_path
    manifest = freezer.ROOT / freezer.MANIFEST_REL

    guard = freezer.SemanticDecoderGuard(bound_members)
    with guard.active():
        with record.open("r", encoding="utf-8") as handle:
            with pytest.raises(PermissionError, match="semantic decoder denied"):
                json.load(handle)
        safe = json.loads(manifest.read_bytes())

    assert safe["case_count"] == 526
    assert len(guard.blocked) == 1
    assert guard.blocked[0]["api"] == "json.load"


def test_guard_probe_receipt_records_real_blocks(bound_members):
    guard = freezer.SemanticDecoderGuard(bound_members)
    receipt = freezer._exercise_decoder_guard(guard, bound_members)

    assert receipt["enforced"] is True
    assert receipt["forbidden_member_digest_count"] == 526
    assert receipt["blocked_probe_count"] == 2
    assert receipt["runtime_violation_count"] == 0
    assert guard.blocked == []


@pytest.mark.parametrize(
    "field",
    ["experimental_value", "score", "signed_error_kcal_mol", "mae"],
)
def test_stage0_output_schema_rejects_sensitive_fields(field):
    with pytest.raises(ValueError, match="Forbidden Stage-0 output field"):
        freezer._assert_output_key_policy({field: 0.0})
