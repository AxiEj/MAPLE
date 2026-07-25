from __future__ import annotations

import re

from .conftest import DOCS_DIR, load_json

FAIL_PATH = DOCS_DIR / "failure-contract-v1.json"


def test_failure_contract_distinguishes_catalog_entry_and_record_fields():
    failure = load_json(FAIL_PATH)
    catalog = failure["catalog"]

    assert catalog["catalog_entry_required_fields"] == [
        "category",
        "code",
        "scientific_gate",
        "fail_closed",
        "recovery_hint",
    ]
    assert catalog["failure_record_required_fields"] == [
        "code",
        "stage",
        "state_id",
        "message",
        "thresholds",
        "observed",
        "recovery_hint",
    ]
    assert catalog["required_fields"] == ["code","category","scientific_gate","fail_closed","recovery_hint"]
    assert failure["runtime"]["required_fields"] == catalog["failure_record_required_fields"]
    assert set(catalog["failure_record_required_fields"]) != set(catalog["catalog_entry_required_fields"])


def test_failure_contract_codes_are_unique_stable_with_repeatable_categories():
    failure = load_json(FAIL_PATH)
    codes = [row["code"] for row in failure["catalog"]["failure_codes"]]

    assert len(codes) == len(set(codes))

    categories = [row["category"] for row in failure["catalog"]["failure_codes"]]
    assert len(categories) > len(set(categories)), "categories are expected to repeat"

    for code in codes:
        assert re.fullmatch(r"[A-Z0-9_]+", code), code

    for row in failure["catalog"]["failure_codes"]:
        assert isinstance(row["code"], str) and row["code"]
        assert isinstance(row["category"], str) and row["category"]
        assert isinstance(row["scientific_gate"], bool)
        assert isinstance(row["fail_closed"], bool)
        assert isinstance(row["recovery_hint"], str) and row["recovery_hint"].strip()

    assert len(failure["catalog"]["known_categories"]) >= 8


def test_benchmark_leakage_hint_is_partition_locked_and_holdout_bound():
    failure = load_json(FAIL_PATH)
    benchmark = next(
        code
        for code in failure["catalog"]["failure_codes"]
        if code["code"] == "BENCHMARK_LEAKAGE"
    )

    hint = benchmark["recovery_hint"].lower()
    assert "partition" in hint
    assert "protocol" in hint
    assert "holdout" in hint
    assert "route-2" not in hint
    assert "parity" not in hint
    assert "selection" in hint


def test_pcmsolver_warning_is_fail_closed_and_blocks_scientific_claims():
    failure = load_json(FAIL_PATH)
    warning = next(
        row
        for row in failure["catalog"]["failure_codes"]
        if row["code"] == "SMD_WARNING"
    )
    assert warning["fail_closed"] is True
    assert warning["scientific_gate"] is True
    assert "native or PEDRA" in warning["recovery_hint"]

    topology = next(
        row
        for row in failure["catalog"]["failure_codes"]
        if row["code"] == "SMD_CAVITY_TOPOLOGY_DISCONTINUITY"
    )
    hint = topology["recovery_hint"]
    assert "no automatic fallback" in hint
    assert "new protocol version" in hint
    assert "regenerated cavity golden" in hint
