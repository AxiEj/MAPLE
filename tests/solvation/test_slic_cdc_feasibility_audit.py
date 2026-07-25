from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks"
    / "route1-slic-cdc-feasibility-audit-2026-07-25.json"
)
REPLAY_PATH = AUDIT_PATH.parent / "route1-slic-cdc-si-table-replay-2026-07-25.json"
REPLAY_RUNNER = AUDIT_PATH.parent / "replay_slic_cdc_si_table.py"
RETAINED_SI = (
    REPOSITORY_ROOT / ".omx/research/slic-cdc-paper-20260725/ct2c00248_si_001.pdf"
)
DOCUMENTATION_DIR = AUDIT_PATH.parent.parent


def _load_audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def _load_replay() -> dict:
    return json.loads(REPLAY_PATH.read_text(encoding="utf-8"))


def _load_replay_runner():
    specification = importlib.util.spec_from_file_location(
        "replay_slic_cdc_si_table",
        REPLAY_RUNNER,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical_content_sha256(artifact: dict) -> str:
    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def test_slic_cdc_audit_has_a_self_consistent_fingerprint():
    audit = _load_audit()
    assert _canonical_content_sha256(audit) == audit["content_sha256"]
    assert audit["schema_version"] == 1
    assert audit["protocol"]["protocol_id"] == "route1-slic-cdc-feasibility-v1"
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_slic_cdc_si_replay_is_source_bound_and_matches_the_audit():
    audit = _load_audit()
    replay = _load_replay()
    fingerprints = audit["source_fingerprints"]
    reanalysis = audit["candidate"]["independent_si_table_reanalysis"]
    replay_record = reanalysis["replay"]

    assert _canonical_content_sha256(replay) == replay["content_sha256"]
    assert (
        hashlib.sha256(REPLAY_RUNNER.read_bytes()).hexdigest()
        == fingerprints["si_replay_runner_sha256"]
    )
    assert (
        hashlib.sha256(REPLAY_PATH.read_bytes()).hexdigest()
        == fingerprints["si_replay_artifact_sha256"]
    )
    assert replay_record["runner_sha256"] == fingerprints["si_replay_runner_sha256"]
    assert (
        replay_record["artifact_file_sha256"]
        == fingerprints["si_replay_artifact_sha256"]
    )
    assert replay_record["artifact_content_sha256"] == replay["content_sha256"]
    assert replay["source"]["pdf_sha256"] == fingerprints["journal_si_pdf_sha256"]
    assert (
        replay["source"]["layout_text_sha256"]
        == fingerprints["journal_si_layout_text_sha256"]
    )
    assert (
        replay["source"]["raw_text_sha256"]
        == fingerprints["journal_si_raw_text_sha256"]
    )
    assert replay["extraction"] == {
        "layout_and_raw_rows_identical": True,
        "normalized_duplicate_names": [
            "2-methylbut-2-ene",
            "3-methylbut-1-ene",
        ],
        "normalized_unique_names": 492,
        "parsed_rows": 494,
        "row_payload_sha256": (
            "7f4b39f12973bac16a6e1397ffa55af99cbbbd810e0d90efa0dc3a0a59dd9d5e"
        ),
        "training_names_listed": 63,
        "training_table_s4": {
            "column_count": 3,
            "extraction_mode": "layout",
            "name_payload_sha256": (
                "63c1d71a32e784bea8f64f319e3b7ac227ea226a7f51eba2ddae8a9524341b1a"
            ),
            "row_count": 21,
        },
    }

    for audit_key, replay_key in (
        ("all_parsed_rows", "all_parsed_rows"),
        ("listed_training_rows", "listed_training_rows"),
        (
            "remaining_rows_after_excluding_listed_training_names",
            "remaining_rows_after_excluding_listed_training_names",
        ),
    ):
        audit_metrics = reanalysis[audit_key]
        replay_metrics = replay["metrics"][replay_key]
        for metric in (
            "mae_kcal_per_mol",
            "maximum_absolute_error_kcal_per_mol",
            "mean_signed_error_kcal_per_mol",
            "rmse_kcal_per_mol",
        ):
            assert replay_metrics[metric] == pytest.approx(audit_metrics[metric])
        if "n" in audit_metrics:
            assert replay_metrics["n"] == audit_metrics["n"]


def test_slic_cdc_si_replay_from_retained_source_when_available():
    if not RETAINED_SI.is_file():
        pytest.skip("The audit-only published SI PDF is not retained in this checkout.")

    runner = _load_replay_runner()
    assert runner.replay(RETAINED_SI) == _load_replay()


def test_slic_cdc_is_physical_and_am1bcc_compatible_but_label_parameterized():
    audit = _load_audit()
    route = audit["route_contract"]
    candidate = audit["candidate"]
    parameterization = candidate["parameterization"]

    assert route["default_charge"] == "AM1-BCC"
    assert not route["gas_phase_mm_energy"]
    assert not route["learned_hydration_residual"]
    assert candidate["route1_formula_compatible"]
    assert candidate["fixed_charge_compatibility"]["charge_model"] == "AM1-BCC"
    assert not candidate["gas_phase_mm_energy_in_product_potential"]
    assert not candidate["learned_solvent_correction"]
    assert parameterization["slic_cdc_parameter_count_for_delta_g"] == 38
    assert parameterization["uses_experimental_hydration_labels"]
    assert parameterization["uses_explicit_solvent_component_labels"]
    assert parameterization["training_count_stated_in_text"] == 65
    assert parameterization["training_names_listed_in_table_s4"] == 63


def test_slic_cdc_si_reanalysis_preserves_discrepancies_and_holdout_boundary():
    audit = _load_audit()
    accuracy = audit["candidate"]["published_accuracy"]
    reanalysis = audit["candidate"]["independent_si_table_reanalysis"]
    all_rows = reanalysis["all_parsed_rows"]
    training = reanalysis["listed_training_rows"]
    remaining = reanalysis["remaining_rows_after_excluding_listed_training_names"]

    assert reanalysis["layout_and_raw_numeric_results_identical"]
    assert reanalysis["parsed_rows"] == 494
    assert reanalysis["normalized_unique_names"] == 492
    assert accuracy["chemrxiv_preprint_rmse_kcal_per_mol"] == pytest.approx(1.15)
    assert accuracy["si_table_s10_footer"]["rmse_kcal_per_mol"] == pytest.approx(0.98)
    assert all_rows["mae_kcal_per_mol"] == pytest.approx(0.8127732793522272)
    assert all_rows["rmse_kcal_per_mol"] == pytest.approx(1.1523325227185854)
    assert training["n"] == 63
    assert training["rmse_kcal_per_mol"] == pytest.approx(0.8526234026032047)
    assert remaining["n"] == 431
    assert remaining["mae_kcal_per_mol"] == pytest.approx(0.8261716937354994)
    assert remaining["rmse_kcal_per_mol"] == pytest.approx(1.1898341634139245)
    assert (
        "not a prospectively sealed independent MAPLE reserve"
        in remaining["classification"]
    )


def test_related_upstreams_do_not_supply_a_complete_force_capable_slic_cdc():
    audit = _load_audit()
    upstream = audit["upstream_implementation_audit"]
    journal = upstream["journal_figshare"]
    pbj = upstream["pbj_related_upstream"]
    historical = upstream["historical_matlab_upstream"]
    decision = audit["overall_decision"]

    assert journal["file_count"] == 1
    assert not journal["code_archive_present"]
    assert not journal["data_archive_present"]
    assert pbj["main_contains_slic_electrostatics"]
    assert pbj["main_contains_simple_sasa_nonpolar_energy"]
    assert not pbj["main_contains_complete_published_slic_cdc"]
    assert not pbj["force_interface"]["atom_resolved_complete_solvation_force_present"]
    assert not pbj["force_interface"]["nonpolar_force_present"]
    assert not pbj["force_interface"]["slic_force_regression_tests_present"]
    assert not historical["complete_2022_neutral_molecule_slic_cdc"]
    decision_flags = {
        key: decision[key]
        for key in (
            "energy_only_final_sp_watch_item",
            "free_solv_product_screen_opened",
            "new_runtime_provider_added",
            "opt_scan_md_claim_allowed",
            "paper_based_reimplementation_planned",
            "speed_claim_allowed",
            "status",
        )
    }
    assert decision_flags == {
        "energy_only_final_sp_watch_item": True,
        "free_solv_product_screen_opened": False,
        "new_runtime_provider_added": False,
        "opt_scan_md_claim_allowed": False,
        "paper_based_reimplementation_planned": False,
        "speed_claim_allowed": False,
        "status": "scientifically_promising_no_complete_upstream_provider",
    }
    assert decision["force_capable_watch_priority_unchanged"] == [
        "GBMV2",
        "GBSW",
        "FACTS",
    ]


def test_route1_docs_preserve_the_slic_cdc_admission_boundary():
    specification = (DOCUMENTATION_DIR / "ROUTE1_PRODUCT_SPEC.md").read_text(
        encoding="utf-8"
    )
    formulas = (DOCUMENTATION_DIR / "FORMULAS_AND_REFERENCES.md").read_text(
        encoding="utf-8"
    )
    validation = (DOCUMENTATION_DIR / "VALIDATION_STATUS.md").read_text(
        encoding="utf-8"
    )
    normalized_validation = " ".join(validation.split())
    benchmark = (AUDIT_PATH.parent / "README.md").read_text(encoding="utf-8")
    assert "SLIC/CDC physical-provider feasibility boundary" in specification
    assert "scientifically_promising_no_complete_upstream_provider" in specification
    assert "route1-slic-cdc-feasibility-audit-2026-07-25.json" in specification
    assert "10.1021/acs.jctc.2c00248" in formulas
    assert "route1-slic-cdc-si-table-replay-2026-07-25.json" in formulas
    assert "38 fitted physical-model parameters" in normalized_validation
    assert "494" in validation
    assert "1.152" in validation
    assert "not an independent confirmation" in normalized_validation
    assert "PBJ is not the published SLIC/CDC model" in validation
    assert "No SLIC/CDC runtime provider" in benchmark
    assert "replay_slic_cdc_si_table.py" in benchmark
