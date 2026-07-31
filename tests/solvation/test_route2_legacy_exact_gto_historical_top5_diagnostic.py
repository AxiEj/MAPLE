from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-legacy-exact-gto-historical-top5-diagnostic-v1.json"
)


def test_high_error_diagnostic_is_preserved_without_becoming_an_acceptance_claim():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["artifact"] == "route2-legacy-exact-gto-historical-top5-diagnostic-v1"
    assert artifact["status"] == "diagnostic-only-not-an-acceptance-panel"
    assert "cannot be reported as Route-2 V0 accuracy" in artifact["claim_boundary"]
    assert "cannot replace" in artifact["claim_boundary"]

    selection = artifact["selection"]
    assert selection == {
        "method": "retired qeq-gto-full-h + gbn2",
        "ranking": "top five absolute signed errors descending",
        "record_count": 5,
        "source_commit": "c77eb6a295cbb39a1b6e66d21190a192cc8ef050",
        "source_path": (
            "docs/implicit-solvation/benchmarks/"
            "freesolv10-qeq-gto-full-h-2026-07-22.json"
        ),
        "source_sha256": (
            "893ad62e476e6375602d82231def6b0ad22cab60128c8068894daaf4a6ee8c4c"
        ),
    }

    policy = artifact["no_target_policy"]
    assert all(
        policy[key] is False
        for key in (
            "post_training",
            "fine_tuning",
            "experimental_solvation_fit",
            "map_or_uq_calibration",
            "radius_adjustment",
            "half_coupling_adjustment",
            "post_selection_parameter_change",
            "experimental_labels_used_to_rank_current_method",
        )
    )
    assert policy["experimental_labels_used_to_rank_diagnostic_records"] is True

    method = artifact["current_method"]
    assert method["scientific_status"] == "legacy-nonvariational-fixedpoint-diagnostic"
    assert method["public_capability"] == "experimental-energy-only"
    assert method["source_receiver_contract"] == (
        "known-nonconjugate-point-source-gto-receiver"
    )
    assert "Route-2 V0 accuracy" in method["prohibited_claims"]
    assert "acceptance-panel pass" in method["prohibited_claims"]


def test_high_error_diagnostic_values_and_nonvariational_contract_are_locked():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = artifact["records"]

    expected = (
        ("mobley_6973347", "ethyl acetate", 7.041442082076966, 0.6866509988069964),
        ("mobley_4883284", "aniline", 5.809357466972302, 0.22708489683263178),
        ("mobley_7532833", "acetonitrile", 3.559916049026574, 0.7075493098774284),
        ("mobley_2198613", "chloroethane", 3.3235498063361715, 0.49964912482428103),
        ("mobley_7015518", "methoxymethane", 3.256791586002884, 1.0664428045869072),
    )
    assert len(records) == len(expected)

    current_errors = []
    for record, (compound_id, name, old_error, new_error) in zip(records, expected):
        assert record["compound_id"] == compound_id
        assert record["name"] == name
        previous = record["historical_retired_gbn2"]
        current = record["legacy_exact_gto_diagnostic"]
        assert previous["absolute_error_kcal_mol"] == pytest.approx(old_error, abs=1e-12)
        assert current["absolute_error_kcal_mol"] == pytest.approx(new_error, abs=1e-12)
        assert current["absolute_error_change_vs_retired_gbn2_kcal_mol"] < 0.0
        assert current["density_residual_inf_e"] < 1.0e-5
        assert len(record["frozen_mol2_sha256"]) == 64
        assert all(
            len(record["runtime_evidence"][key]) == 64
            for key in (
                "raw_audit_sha256",
                "public_record_sha256",
                "public_manifest_sha256",
            )
        )
        contract = record["immutable_evaluation_record"]["source_receiver_contract"]
        assert contract["common_stationary_electronic_functional_established"] is False
        assert contract["continuum_pairing_established"] is False
        assert contract["public_capability"] == "experimental-energy-only"
        assert contract["solute_source"] == "point-multipole-l1"
        assert contract["reaction_field_receiver"] == "exact-gto-v1"
        current_errors.append(current["absolute_error_kcal_mol"])

    summary = artifact["summary"]
    assert summary["maximum_absolute_error_kcal_mol"] == pytest.approx(
        max(current_errors), abs=1e-12
    )
    assert summary["mean_absolute_error_kcal_mol"] == pytest.approx(
        sum(current_errors) / len(current_errors), abs=1e-12
    )
    assert all(error < 1.5 for error in current_errors)
    assert summary["all_selected_records_strictly_below_1_5_kcal_mol"] is True
    assert summary["new_error_lower_than_retired_gbn2_for_all_selected_records"] is True
    assert math.isfinite(summary["panel_wall_seconds"])
