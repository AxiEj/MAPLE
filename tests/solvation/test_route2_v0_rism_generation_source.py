from __future__ import annotations

import hashlib
import json

import pytest
from route2_v0_asset_fixture import write_route2_v0_test_manifest

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_generation_source import (
    normalized_rism1d_xvv_sha256,
    parse_route2_v0_rism_generation_source,
)


def _payload(tmp_path) -> dict:
    write_route2_v0_test_manifest(tmp_path)
    return json.loads((tmp_path / "provenance/cSPCE.json").read_text(encoding="utf-8"))


def test_rism_generation_source_binds_two_converged_source_identical_runs(
    tmp_path,
):
    source = parse_route2_v0_rism_generation_source(json.dumps(_payload(tmp_path)))

    assert len(source.runs) == 2
    assert source.runs[0].primary_iterations == 1
    assert source.runs[0].temperature_derivative_iterations == 1
    assert source.runs[0].primary_final_residual == pytest.approx(5.0e-13)
    assert source.runs[0].temperature_derivative_final_residual == pytest.approx(
        4.0e-13
    )
    assert source.normalized_xvv_sha256 == source.runs[1].normalized_xvv_sha256
    source.verify_xvv_reproduction(tmp_path / "bulk/cSPCE.xvv")
    assert set(source.excluded_target_label_sets) >= {
        "mnsol",
        "freesolv",
        "development",
        "confirmation",
        "blind",
    }


def test_rism_generation_source_rejects_transcript_hash_drift(tmp_path):
    payload = _payload(tmp_path)
    payload["runs"][0]["transcript"] += "mutated\n"

    with pytest.raises(ValueError, match="transcript content hash"):
        parse_route2_v0_rism_generation_source(json.dumps(payload))


def test_rism_generation_source_rejects_unconverged_residual(tmp_path):
    payload = _payload(tmp_path)
    transcript = payload["runs"][0]["transcript"].replace(
        "5.0000000000000000E-13",
        "5.0000000000000000E-10",
    )
    payload["runs"][0]["transcript"] = transcript
    payload["runs"][0]["transcript_sha256"] = hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="residual tolerance"):
        parse_route2_v0_rism_generation_source(json.dumps(payload))


def test_rism_generation_source_rejects_independent_xvv_mismatch(tmp_path):
    payload = _payload(tmp_path)
    payload["runs"][1]["normalized_xvv_sha256"] = "2" * 64

    with pytest.raises(ValueError, match="timestamp-normalized XVV"):
        parse_route2_v0_rism_generation_source(json.dumps(payload))


def test_rism_generation_source_rejects_unreconstructable_raw_xvv(tmp_path):
    payload = _payload(tmp_path)
    payload["runs"][1]["xvv_first_line"] = payload["runs"][1]["xvv_first_line"].replace(
        "00:00:02", "00:00:03"
    )
    source = parse_route2_v0_rism_generation_source(json.dumps(payload))

    with pytest.raises(ValueError, match="cannot be reconstructed"):
        source.verify_xvv_reproduction(tmp_path / "bulk/cSPCE.xvv")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("raw_xvv_sha256", "distinct raw XVV evidence"),
        ("transcript_sha256", "distinct transcript evidence"),
    ],
)
def test_rism_generation_source_rejects_duplicate_run_evidence(
    tmp_path,
    field,
    message,
):
    payload = _payload(tmp_path)
    payload["runs"][1][field] = payload["runs"][0][field]
    if field == "transcript_sha256":
        payload["runs"][1]["transcript"] = payload["runs"][0]["transcript"]

    with pytest.raises(ValueError, match=message):
        parse_route2_v0_rism_generation_source(json.dumps(payload))


def test_rism_generation_source_rejects_version_metadata_drift(tmp_path):
    payload = _payload(tmp_path)
    second = payload["runs"][1]
    second["xvv_first_line"] = second["xvv_first_line"].replace(
        "V0001.001",
        "V0002.001",
    )
    frozen = (tmp_path / "bulk/cSPCE.xvv").read_bytes()
    body = frozen.partition(b"\n")[2]
    second["raw_xvv_sha256"] = hashlib.sha256(
        second["xvv_first_line"].encode("utf-8") + b"\n" + body
    ).hexdigest()
    source = parse_route2_v0_rism_generation_source(json.dumps(payload))

    with pytest.raises(ValueError, match="non-DATE XVV metadata"):
        source.verify_xvv_reproduction(tmp_path / "bulk/cSPCE.xvv")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda policy: policy.__setitem__("fine_tuning", True),
            "fine_tuning must be explicitly false",
        ),
        (
            lambda policy: policy["excluded_target_label_sets"].remove("blind"),
            "must exclude target-label sets",
        ),
    ],
)
def test_rism_generation_source_rejects_target_driven_policy(
    tmp_path,
    mutate,
    message,
):
    payload = _payload(tmp_path)
    mutate(payload["no_target_policy"])

    with pytest.raises(ValueError, match=message):
        parse_route2_v0_rism_generation_source(json.dumps(payload))


def test_normalized_xvv_hash_normalizes_only_date_value(tmp_path):
    left = tmp_path / "left.xvv"
    right = tmp_path / "right.xvv"
    changed = tmp_path / "changed.xvv"
    version_drift = tmp_path / "version-drift.xvv"
    whitespace_drift = tmp_path / "whitespace-drift.xvv"
    line_ending_drift = tmp_path / "line-ending-drift.xvv"
    left.write_bytes(
        b"%VERSION VERSION_STAMP=V0001.001 DATE=01:01:00 00:00:01\n%FLAG XVV\n1.0\n"
    )
    right.write_bytes(
        b"%VERSION VERSION_STAMP=V0001.001 DATE=01:01:00 00:00:02\n%FLAG XVV\n1.0\n"
    )
    changed.write_bytes(
        b"%VERSION VERSION_STAMP=V0001.001 DATE=01:01:00 00:00:01\n%FLAG XVV\n2.0\n"
    )
    version_drift.write_bytes(
        b"%VERSION VERSION_STAMP=V0002.001 DATE=01:01:00 00:00:02\n%FLAG XVV\n1.0\n"
    )
    whitespace_drift.write_bytes(
        b"%VERSION VERSION_STAMP=V0001.001 DATE=01:01:00 00:00:02 \n%FLAG XVV\n1.0\n"
    )
    line_ending_drift.write_bytes(
        b"%VERSION VERSION_STAMP=V0001.001 DATE=01:01:00 00:00:02\r\n%FLAG XVV\n1.0\n"
    )

    assert normalized_rism1d_xvv_sha256(left) == normalized_rism1d_xvv_sha256(right)
    assert normalized_rism1d_xvv_sha256(left) != normalized_rism1d_xvv_sha256(changed)
    assert normalized_rism1d_xvv_sha256(left) != normalized_rism1d_xvv_sha256(
        version_drift
    )
    assert normalized_rism1d_xvv_sha256(left) != normalized_rism1d_xvv_sha256(
        whitespace_drift
    )
    assert normalized_rism1d_xvv_sha256(left) != normalized_rism1d_xvv_sha256(
        line_ending_drift
    )
