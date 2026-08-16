from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from maple.solvation.release.evidence import canonical_json_sha256, sha256_file
from tools.route2_release import create_maple_cds_w1_m2_preregistration as creator
from tools.route2_release import fit_maple_cds_w1_m2_scale as fitter
from tools.route2_release.analyze_hybrid_smd_components_v3 import (
    HybridComponentAnalysisError,
    _sha256,
)
from tools.route2_release.fit_maple_cds_w1_m2_scale import (
    EXPECTED_NONAQUEOUS_COUNT,
    EXPECTED_WATER_COUNT,
    M2FitError,
    _fit_water_scale,
)


def _records(*, alpha: float, water_noise: float = 0.0):
    rows = []
    for index in range(EXPECTED_WATER_COUNT):
        stock = 0.5 + 0.01 * (index % 17)
        electrostatic = -5.0 - 0.02 * (index % 11)
        experimental = electrostatic + alpha * stock
        if index == 0:
            experimental += water_noise
        rows.append(
            {
                "canonical_solvent": "water",
                "continuum_polarization_kcal_mol": electrostatic,
                "smd_cds_kcal_mol": stock,
                "experimental_delta_g_kcal_mol": experimental,
            }
        )
    for index in range(EXPECTED_NONAQUEOUS_COUNT):
        stock = 0.7
        electrostatic = -3.0
        rows.append(
            {
                "canonical_solvent": "toluene",
                "continuum_polarization_kcal_mol": electrostatic,
                "smd_cds_kcal_mol": stock,
                "experimental_delta_g_kcal_mol": electrostatic + stock,
            }
        )
    return rows


def test_closed_form_recovers_one_water_scale_without_touching_nonaqueous():
    result = _fit_water_scale(_records(alpha=0.4))
    assert result["alpha"] == pytest.approx(0.4, abs=2.0e-15)
    assert result["alpha_in_preregistered_domain"] is True
    assert result["water_m2"]["mean_absolute_error_kcal_mol"] < 1.0e-14
    assert result["water_development_decision"] == "pass"
    assert (
        result["mixed_505_water_m2_nonaqueous_m1"]["mean_absolute_error_kcal_mol"]
        < 1.0e-14
    )


def test_out_of_domain_scale_is_rejected_without_clipping():
    result = _fit_water_scale(_records(alpha=-0.25))
    assert result["alpha"] == pytest.approx(-0.25, abs=2.0e-15)
    assert result["alpha_in_preregistered_domain"] is False
    assert result["water_development_decision"] == "fail"


def test_record_scope_is_exact_and_predictor_must_have_information():
    rows = _records(alpha=0.4)
    with pytest.raises(M2FitError, match="counts drifted"):
        _fit_water_scale(rows[:-1])

    for row in rows:
        row["smd_cds_kcal_mol"] = 0.0
        row["experimental_delta_g_kcal_mol"] = row["continuum_polarization_kcal_mol"]
    with pytest.raises(M2FitError, match="no finite scale information"):
        _fit_water_scale(rows)


class _FakeSnapshot:
    def __init__(self, root: Path):
        self.root = root
        self.head = "a" * 40
        self.tree = "b" * 40

    def assert_unchanged(self) -> None:
        return None


class _FakeRepositorySnapshot:
    @classmethod
    def capture(cls, root: Path) -> _FakeSnapshot:
        return _FakeSnapshot(Path(root).resolve())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resign_audit(payload: dict[str, object]) -> None:
    payload.pop("verification_sha256", None)
    payload["verification_sha256"] = canonical_json_sha256(payload)


def _install_synthetic_source(monkeypatch: pytest.MonkeyPatch, source_root: Path):
    source_root.mkdir()
    runtime = ({"runtime": "synthetic-m2-test-v1"}, "c" * 64)
    for module in (creator, fitter):
        monkeypatch.setattr(module, "_SOURCE_ROOT", source_root)
        monkeypatch.setattr(module, "RepositorySnapshot", _FakeRepositorySnapshot)
        monkeypatch.setattr(module, "normalized_runtime_identity", lambda: runtime)
    monkeypatch.setattr(creator, "committed_source_hashes", lambda *_: {})


def _write_bound_505_records(root: Path, parent_sha256: str) -> dict[str, object]:
    records = root / "records"
    bindings = {
        "preregistration_sha256": parent_sha256,
        "runner_sha256": "2" * 64,
        "mdp_checkpoint_sha256": "3" * 64,
        "polar_checkpoint_sha256": "4" * 64,
        "hybrid_configuration_sha256": "5" * 64,
    }
    manifest = []
    for index, row in enumerate(_records(alpha=0.4)):
        continuum = float(row["continuum_polarization_kcal_mol"])
        cds = float(row["smd_cds_kcal_mol"])
        predicted = continuum + cds
        path = records / f"index-{index:03d}.json"
        _write_json(
            path,
            {
                "artifact": "route2-hybrid-smd-development-record-v3",
                "selection_index": index,
                "status": "pass",
                "partition": "development",
                "confirmation_partition_opened": False,
                "do_not_commit": True,
                "predicted_delta_g_kcal_mol": predicted,
                **row,
                **bindings,
            },
        )
        manifest.append((index, _sha256(path)))
    audit: dict[str, object] = {
        "artifact": "route2-hybrid-smd-development-integrity-audit-v1",
        "evidence_contract_version": 3,
        "status": "pass",
        "expected_record_count": EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT,
        "record_count": EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT,
        "success_count": EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT,
        "failure_count": 0,
        "missing_selection_indices": [],
        "failed_selection_indices": [],
        "partition": "development",
        "confirmation_partition_opened": False,
        "record_files_manifest_sha256": canonical_json_sha256(manifest),
        **bindings,
    }
    _resign_audit(audit)
    return audit


def test_m2_preregistration_and_fit_are_cross_bound_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "source"
    _install_synthetic_source(monkeypatch, source_root)
    evidence_root = tmp_path / "hybrid-v3"
    records = evidence_root / "records"
    records.mkdir(parents=True)
    (records / "index-000.json").write_text("{}\n")
    parent_path = evidence_root / "preregistration.json"
    parent = {
        "artifact_id": creator.PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
        "source_git_head": "d" * 40,
        "preregistration_path": str(parent_path),
        "records_path": str(records),
    }
    _write_json(parent_path, parent)

    preregistration_path = tmp_path / "external" / "m2-preregistration.json"
    preregistration = creator.create(
        argparse.Namespace(
            source_root=source_root,
            parent_hybrid_preregistration=parent_path,
            hybrid_evidence_root=evidence_root,
            output=preregistration_path,
        )
    )
    state = preregistration["hybrid_evidence_state_at_lock"]
    assert state["record_file_count_at_lock"] == 1
    assert state["record_contents_read"] is False
    assert all(state["terminal_evidence_absent"].values())

    parent_sha256 = sha256_file(parent_path)
    audit = _write_bound_505_records(evidence_root, parent_sha256)
    audit_path = evidence_root / "audits/independent-integrity-audit.json"
    _write_json(audit_path, audit)

    with pytest.raises(
        creator.M2PreregistrationError, match="before all 505 v3 records"
    ):
        creator.create(
            argparse.Namespace(
                source_root=source_root,
                parent_hybrid_preregistration=parent_path,
                hybrid_evidence_root=evidence_root,
                output=tmp_path / "external" / "too-late.json",
            )
        )

    output = tmp_path / "external" / "m2-fit.json"
    result = fitter.fit(
        argparse.Namespace(
            source_root=source_root,
            preregistration=preregistration_path,
            input_dir=records,
            integrity_audit=audit_path,
            output=output,
        )
    )
    assert result["status"] == "development-pass"
    assert result["result"]["alpha"] == pytest.approx(0.4, abs=2.0e-15)
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        fitter.fit(
            argparse.Namespace(
                source_root=source_root,
                preregistration=preregistration_path,
                input_dir=records,
                integrity_audit=audit_path,
                output=output,
            )
        )
    assert output.read_bytes() == original_output

    mismatched = dict(audit)
    mismatched["preregistration_sha256"] = "e" * 64
    _resign_audit(mismatched)
    _write_json(audit_path, mismatched)
    with pytest.raises(M2FitError, match="not bound to the preregistered parent"):
        fitter.fit(
            argparse.Namespace(
                source_root=source_root,
                preregistration=preregistration_path,
                input_dir=records,
                integrity_audit=audit_path,
                output=tmp_path / "external" / "mismatch.json",
            )
        )

    confirmation = dict(audit)
    confirmation["confirmation_partition_opened"] = True
    _resign_audit(confirmation)
    _write_json(audit_path, confirmation)
    with pytest.raises(
        HybridComponentAnalysisError, match="confirmation_partition_opened"
    ):
        fitter.fit(
            argparse.Namespace(
                source_root=source_root,
                preregistration=preregistration_path,
                input_dir=records,
                integrity_audit=audit_path,
                output=tmp_path / "external" / "confirmation.json",
            )
        )
