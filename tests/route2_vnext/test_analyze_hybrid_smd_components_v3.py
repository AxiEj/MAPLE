from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.route2_release.analyze_hybrid_smd_components_v3 import (
    HybridComponentAnalysisError,
    _canonical_sha256,
    _sha256,
    analyze_hybrid_components,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resign_audit(payload: dict[str, object]) -> None:
    payload.pop("verification_sha256", None)
    payload["verification_sha256"] = _canonical_sha256(payload)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    records = tmp_path / "records"
    bindings = {
        "preregistration_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "mdp_checkpoint_sha256": "3" * 64,
        "polar_checkpoint_sha256": "4" * 64,
        "hybrid_configuration_sha256": "5" * 64,
    }
    rows = (
        (0, "water", -4.0, -5.0, 0.5),
        (1, "water", -2.0, -1.0, -0.5),
        (2, "toluene", -3.0, -5.0, 1.5),
    )
    manifest = []
    for index, solvent, experimental, electrostatic, cds in rows:
        predicted = electrostatic + cds
        signed = predicted - experimental
        path = records / f"index-{index:03d}.json"
        _write(
            path,
            {
                "artifact": "route2-hybrid-smd-development-record-v3",
                "selection_index": index,
                "status": "pass",
                "partition": "development",
                "confirmation_partition_opened": False,
                "do_not_commit": True,
                "canonical_solvent": solvent,
                "experimental_delta_g_kcal_mol": experimental,
                "continuum_polarization_kcal_mol": electrostatic,
                "smd_cds_kcal_mol": cds,
                "predicted_delta_g_kcal_mol": predicted,
                "signed_error_kcal_mol": signed,
                "absolute_error_kcal_mol": abs(signed),
                **bindings,
            },
        )
        manifest.append((index, _sha256(path)))
    audit: dict[str, object] = {
        "artifact": "route2-hybrid-smd-development-integrity-audit-v1",
        "evidence_contract_version": 3,
        "status": "accuracy-failure",
        "expected_record_count": 3,
        "record_count": 3,
        "success_count": 3,
        "failure_count": 0,
        "missing_selection_indices": [],
        "failed_selection_indices": [],
        "partition": "development",
        "confirmation_partition_opened": False,
        "record_files_manifest_sha256": _canonical_sha256(manifest),
        **bindings,
    }
    audit["verification_sha256"] = _canonical_sha256(audit)
    audit_path = tmp_path / "audit.json"
    _write(audit_path, audit)
    return records, audit_path


def test_component_analysis_reports_paired_m0_m1_without_fitting(tmp_path: Path):
    records, audit = _fixture(tmp_path)
    result = analyze_hybrid_components(
        input_dir=records,
        integrity_audit_path=audit,
        expected_count=3,
    )

    assert result["status"] == "complete"
    assert result["record_count"] == 3
    assert result["fitting_or_calibration_performed"] is False
    assert result["confirmation_partition_opened"] is False
    assert set(result["per_solvent"]) == {"toluene", "water"}
    assert result["water_records"]["record_count"] == 2
    assert result["nonaqueous_records"]["record_count"] == 1

    all_records = result["all_records"]
    m0 = all_records["m0_hybrid_electrostatic_only"]
    m1 = all_records["m1_hybrid_electrostatic_plus_stock_smd_cds"]
    assert m0["mean_absolute_error_kcal_mol"] == pytest.approx(4.0 / 3.0)
    assert m1["mean_absolute_error_kcal_mol"] == pytest.approx(0.5)
    assert m1["q95_absolute_error_kcal_mol"] == pytest.approx(0.5)
    assert all_records["paired_stock_cds_effect"] == {
        "mean_absolute_error_change_m1_minus_m0_kcal_mol": pytest.approx(-5.0 / 6.0),
        "helped_count": 3,
        "worsened_count": 0,
        "tied_count": 0,
    }
    assert len(result["analysis_sha256"]) == 64


def test_component_analysis_rejects_incomplete_or_nonterminal_evidence(tmp_path: Path):
    records, audit = _fixture(tmp_path)
    (records / "index-002.json").unlink()
    with pytest.raises(HybridComponentAnalysisError, match="incomplete"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )

    records, audit = _fixture(tmp_path / "nonterminal")
    payload = json.loads(audit.read_text())
    payload["status"] = "incomplete"
    _resign_audit(payload)
    _write(audit, payload)
    with pytest.raises(HybridComponentAnalysisError, match="terminal"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )


def test_component_analysis_rejects_ledger_or_manifest_drift(tmp_path: Path):
    records, audit = _fixture(tmp_path)
    record = records / "index-001.json"
    payload = json.loads(record.read_text())
    payload["smd_cds_kcal_mol"] += 0.25
    _write(record, payload)
    with pytest.raises(HybridComponentAnalysisError, match="does not close"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )

    records, audit = _fixture(tmp_path / "manifest")
    payload = json.loads(audit.read_text())
    payload["record_files_manifest_sha256"] = "0" * 64
    _resign_audit(payload)
    _write(audit, payload)
    with pytest.raises(HybridComponentAnalysisError, match="manifest drifted"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )


def test_component_analysis_rejects_provider_or_configuration_drift(tmp_path: Path):
    records, audit = _fixture(tmp_path)
    record = records / "index-000.json"
    payload = json.loads(record.read_text())
    payload["hybrid_configuration_sha256"] = "f" * 64
    _write(record, payload)
    with pytest.raises(HybridComponentAnalysisError, match="configuration"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )


def test_component_analysis_rejects_tampered_integrity_audit(tmp_path: Path):
    records, audit = _fixture(tmp_path)
    payload = json.loads(audit.read_text())
    payload["status"] = "pass"
    _write(audit, payload)
    with pytest.raises(HybridComponentAnalysisError, match="self-hash"):
        analyze_hybrid_components(
            input_dir=records,
            integrity_audit_path=audit,
            expected_count=3,
        )
