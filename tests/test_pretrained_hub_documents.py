from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "docs/pretrained-solvation-hub"


def test_mnsol_reference_preserves_the_frozen_route2_split_without_rows():
    reference = json.loads(
        (HUB / "mnsol-partition-reference.json").read_text(encoding="utf-8")
    )

    assert reference["partitions"]["development"]["record_count"] == 505
    assert reference["partitions"]["development"]["unique_solute_count"] == 312
    assert reference["partitions"]["confirmation"]["record_count"] == 148
    assert reference["partitions"]["confirmation"]["unique_solute_count"] == 83
    assert reference["total"] == {"record_count": 653, "unique_solute_count": 395}
    assert reference["confirmation_policy"]["sealed"] is True
    assert reference["confirmation_policy"][
        "failed_confirmation_must_not_trigger_tuning"
    ] is True
    assert "records" not in reference


def test_watchlist_has_no_executable_placeholder_backends():
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    statuses = {
        model["model_id"]: model["adapter_status"] for model in watchlist["models"]
    }

    assert statuses["consolv"] == "blocked_by_public_runtime_and_weights"
    assert statuses["twin"] == "blocked_by_public_runtime_and_weights"
    assert statuses["mace-off24-sc"] == "blocked_by_checkpoint_identity"
    assert watchlist["property_only_baselines"]["registry"] == "benchmark_only"


def test_hub_document_keeps_free_energy_and_route_boundaries_explicit():
    text = (HUB / "README.md").read_text(encoding="utf-8")

    assert "not a claim that an MLIP alone computes" in text
    assert "not thermochemical Gibbs energy" in text
    assert "does not duplicate the Route 3 `#solvfe`" in text
    assert "does not add a public `#bindfe`" in text
    assert "never ranked against" in text


def test_upstream_artifact_manifest_pins_real_files_and_unknowns():
    payload = json.loads(
        (HUB / "upstream-artifacts.json").read_text(encoding="utf-8")
    )
    artifacts = payload["artifacts"]
    assert len(artifacts) >= 7
    assert len({item["model_id"] for item in artifacts}) == len(artifacts)
    for item in artifacts:
        assert len(item["source_revision"]) == 40
        assert item["size_bytes"] > 0
        assert len(item["sha256"]) == 64
        int(item["sha256"], 16)
        assert item["artifact_license"]

    by_id = {item["model_id"]: item for item in artifacts}
    assert by_id["aimnet2-cpcms-v2"]["size_bytes"] == 9280334
    assert by_id["gnnis-reference"]["repository_license"] == "MIT-0"
    assert by_id["mace-off24-medium"]["sha256"].startswith("e5ccf583")
    assert by_id["mace-off23-sc"]["scientific_status"].endswith(
        "protocol-bridge-pending"
    )
    assert "aceff_examples@3c59fb3" in by_id["aceff-2.0"]["unit_evidence"]
