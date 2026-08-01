from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "route2-v0-freesolv12-mace-localized-response-execution-62a8e413.json"
)


def test_localized_response_execution_records_the_frozen_rejection():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["artifact"] == "route2-v0-freesolv12-mace-localized-response-v1"
    assert payload["status"] == "complete-response-gate-reject"
    assert payload["success_count"] == 12
    assert payload["failure_count"] == 0
    assert len(payload["records"]) == 12
    assert payload["disposition"] == {
        "all_records_pass_registered_response_gates": False,
        "continuum_or_solvation_energy_invoked": False,
        "experimental_solvation_labels_read": False,
        "learned_fixed_point_admitted_as_variational": False,
        "next_gate": (
            "Use this broad spatial-response result to admit or reject a "
            "separately scalar V0 response candidate; never repair a failed "
            "record with a fitted scale or experimental solvation residual."
        ),
    }

    assert all(record["status"] == "success" for record in payload["records"])
    assert all(
        all(check["passes"] for check in record["numerical_checks"].values())
        for record in payload["records"]
    )
    assert all(
        record["passes_all_registered_checks"] is False
        for record in payload["records"]
    )
    assert all(
        any(
            check["passes"] is False
            for check in record["scientific_checks"].values()
        )
        for record in payload["records"]
    )
    assert payload["run_lock"]["execution_git_head"] == (
        "62a8e4139727be226eade8cdc435a05758702ab9"
    )
    assert payload["run_lock"]["experimental_solvation_labels_read"] is False
