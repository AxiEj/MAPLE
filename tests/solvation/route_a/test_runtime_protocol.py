from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from maple.function.dispatcher.solvfe.protocol import (
    ProtocolIntegrityError,
    RouteAProtocol,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs" / "solvation" / "route-a" / "protocol-v1.json"


def test_runtime_protocol_loader_verifies_schema_self_hash_and_artifacts():
    loaded = RouteAProtocol.load(PROTOCOL, project_root=ROOT)

    assert loaded.content_hash == loaded.data["protocol_sha256"]
    assert loaded.data["protocol_version"] == "1.4.0"
    assert loaded.data["tail_envelope_contract"][
        "conservative_ratio_ceiling"
    ] == 0.55
    assert loaded.verified_artifact_count == len(
        loaded.data["artifact_references"]
    )


def test_runtime_protocol_loader_rejects_self_rehashed_semantic_mutation(
    tmp_path,
):
    data = json.loads(PROTOCOL.read_text())
    data["tail_envelope_contract"]["conservative_ratio_ceiling"] = 0.8
    payload = copy.deepcopy(data)
    payload.pop("protocol_sha256")
    from maple.function.dispatcher.solvfe.protocol import canonical_sha256

    data["protocol_sha256"] = canonical_sha256(payload)
    mutated = tmp_path / "protocol.json"
    mutated.write_text(json.dumps(data))

    with pytest.raises(ProtocolIntegrityError, match="schema"):
        RouteAProtocol.load(mutated, project_root=ROOT)

