from __future__ import annotations

import csv
import importlib.util
import io
import json
from typing import Any, cast
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_multisolvent_source_protocol.json"
ACCURACY_CONTRACT_PATH = (
    BENCHMARK_DIR / "route1_multisolvent_accuracy_contract_v1.json"
)
ASSET_CONTRACT_PATH = BENCHMARK_DIR / "route1_custom_solvent_asset_contract_v1.json"
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-mnsol-multisolvent-source-audit-2026-07-29.json"
)
SPEC = importlib.util.spec_from_file_location(
    "route1_multisolvent_source_audit",
    BENCHMARK_DIR / "route1_multisolvent_source_audit.py",
)
assert SPEC is not None
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _fake_protocol() -> dict[str, Any]:
    protocol = cast(dict[str, Any], json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
    for entry in cast(list[dict[str, Any]], protocol["panel"]):
        entry["expected_all_records"] = 7
        entry["expected_neutral_absolute_records"] = 7
    return protocol


def _write_fake_archive(
    tmp_path: Path,
    protocol: dict[str, Any],
    *,
    invalid_label_bytes: bool = False,
) -> tuple[Path, str]:
    headers = [
        "No.",
        "FileHandle",
        "Charge",
        "Solvent",
        "DeltaGsolv",
        "type",
        "eps",
        "n",
        "alpha",
        "beta",
        "gamma",
    ]
    secret_label = "SECRET_EXPERIMENTAL_DELTA_G_MUST_NOT_ESCAPE"
    table = io.StringIO(newline="")
    writer = csv.writer(table, delimiter="\t", lineterminator="\n")
    writer.writerow(headers)
    geometries: list[str] = []
    row_number = 1
    for solvent_index, entry in enumerate(cast(list[dict[str, Any]], protocol["panel"])):
        for sample_index in range(7):
            handle = f"solute_{solvent_index}_{sample_index}"
            geometries.append(handle)
            writer.writerow(
                [
                    row_number,
                    handle,
                    0,
                    entry["mnsol_name"],
                    secret_label,
                    "abs",
                    "1.0",
                    "1.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ]
            )
            row_number += 1
    table_bytes = table.getvalue().encode("utf-8")
    if invalid_label_bytes:
        table_bytes = table_bytes.replace(secret_label.encode("utf-8"), b"\xff")
    archive_path = tmp_path / "MNSolDatabase_v2012.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("MNSolDatabase_v2012/MNSol_alldata.txt", table_bytes)
        for handle in geometries:
            archive.writestr(f"MNSolDatabase_v2012/all_solutes/{handle}.xyz", "0\n\n")

    dataset = cast(dict[str, Any], protocol["dataset"])
    archive = dataset["archive"]
    archive["size_bytes"] = archive_path.stat().st_size
    archive["sha256"] = audit.core.sha256_file(archive_path)
    source_table = dataset["table"]
    source_table["size_bytes"] = len(table_bytes)
    source_table["sha256"] = audit.core.sha256_bytes(table_bytes)
    source_table["expected_rows"] = row_number - 1
    return archive_path, secret_label


def _write_protocol(path: Path, protocol: dict[str, Any]) -> Path:
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def test_protocol_freezes_a_label_free_fifteen_solvent_gate() -> None:
    protocol, fingerprint = audit._load_protocol(PROTOCOL_PATH)
    benchmark_readme = (BENCHMARK_DIR / "README.md").read_text(encoding="utf-8")

    assert len(fingerprint) == 64
    assert protocol["dataset"]["acquisition_policy"] == "user-supplied-no-auto-download"
    assert len(protocol["panel"]) == 15
    assert sum(
        entry["expected_neutral_absolute_records"] for entry in protocol["panel"]
    ) == 1106
    assert all(
        entry["expected_neutral_absolute_records"] >= 7
        for entry in protocol["panel"]
    )
    assert protocol["evaluation_design"]["experimental_values_loaded"] is False
    assert protocol["evaluation_design"]["fit_or_tuning_allowed"] is False
    assert protocol["evaluation_design"]["pooled_random_split_is_sufficient"] is False
    assert "Label-free MNSol multi-solvent source audit" in benchmark_readme
    assert ARTIFACT_PATH.name in benchmark_readme


def test_accuracy_contract_requires_all_records_and_an_unseen_external_final() -> None:
    contract = json.loads(ACCURACY_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert (
        contract["status"]
        == "blocked_research_comparator_not_product_route1_candidate"
    )
    source = contract["source_evidence"]
    assert source["coverage_artifact_file_sha256"] == audit.core.sha256_file(
        ARTIFACT_PATH
    )
    assert source["coverage_artifact_content_sha256"] == json.loads(
        ARTIFACT_PATH.read_text(encoding="utf-8")
    )["content_sha256"]
    assert source["source_protocol_file_sha256"] == audit.core.sha256_file(
        PROTOCOL_PATH
    )
    candidate = contract["candidate_admission"]
    assert (
        candidate["route_identity"]
        == "separate_benchmark_only_3drism_research_comparator"
    )
    assert candidate["candidate_is_product_route1_endpoint"] is False
    assert candidate["candidate_may_satisfy_product_route1_accuracy_gate"] is False
    assert candidate["product_route1_formula_remains_unchanged"] is True
    assert candidate["asset_contract_file_sha256"] == audit.core.sha256_file(
        ASSET_CONTRACT_PATH
    )
    assert candidate["required_asset_audit_count"] == 15
    assert candidate["candidate_selection_from_mnsol_labels"] is False
    assert candidate["solvent_specific_error_offset"] is False
    thermodynamics = contract["thermodynamic_admission"]
    assert thermodynamics == {
        "mnsol_target_temperature_kelvin": 298.0,
        "existing_298_15_kelvin_assets_are_accuracy_eligible": False,
        "mnsol_standard_state": "Ben-Naim 1 M ideal gas to 1 M ideal solution",
        "neutral_excess_chemical_potential_standard_state_shift_kcal_mol": 0.0,
        "rt_ln_24_46_is_a_pressure_standard_state_shift": True,
        "rt_ln_24_46_applied_to_neutral_excess_chemical_potential": False,
        "pressure_correction_is_not_a_standard_state_conversion": True,
        "fixed_geometry_energy_can_promote_absolute_solvation_accuracy": False,
        "ensemble_protocol_required_before_absolute_solvation_scoring": True,
    }
    separation = contract["label_separation"]
    assert separation["label_free_energy_artifact_sealed_before_scoring"] is True
    assert separation["labels_may_be_used_for_residual_training"] is False
    milestone = contract["accuracy_gates"]["development_physical_milestone"]
    ultimate = contract["accuracy_gates"]["ultimate_internal_target"]
    assert milestone == {
        "required_solvents": 15,
        "required_records": 1106,
        "coverage_fraction": 1.0,
        "every_record_absolute_error_strictly_below_kcal_mol": 1.5,
    }
    assert ultimate["every_record_absolute_error_strictly_below_kcal_mol"] == 1.0
    external = contract["external_final"]
    assert external["status"] == "not_selected"
    assert external["record_identity_overlap_with_mnsol_or_freesolv"] is False
    assert external["used_in_any_previous_score"] is False
    assert external["one_shot_only"] is True
    assert contract["activation"]["accuracy_claim_allowed"] is False
    assert contract["activation"]["custom_solvent_runtime_allowed"] is False


def test_protocol_rejects_a_panel_with_only_ten_solvents(tmp_path: Path) -> None:
    protocol = _fake_protocol()
    protocol["panel"] = protocol["panel"][:10]

    with pytest.raises(ValueError, match="more than ten solvents"):
        audit._load_protocol(_write_protocol(tmp_path / "protocol.json", protocol))


def test_label_free_audit_never_emits_secret_experimental_values(
    tmp_path: Path,
) -> None:
    protocol = _fake_protocol()
    archive_path, secret_label = _write_fake_archive(tmp_path, protocol)
    loaded_protocol, fingerprint = audit._load_protocol(
        _write_protocol(tmp_path / "protocol.json", protocol)
    )

    artifact = audit.audit_archive(loaded_protocol, fingerprint, archive_path)
    serialized = json.dumps(artifact, sort_keys=True)

    assert artifact["content_sha256"] == audit.core.artifact_content_sha256(artifact)
    assert artifact["counts"] == {
        "panel_solvents": 15,
        "panel_neutral_absolute_records": 105,
        "panel_unique_neutral_geometry_pairs": 105,
    }
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []
    assert artifact["source_table"]["metadata_columns_loaded"] == [
        "FileHandle",
        "Charge",
        "Solvent",
        "type",
    ]
    assert artifact["source_archive"]["source_path_recorded"] is False
    assert secret_label not in serialized
    assert "DeltaGsolv" not in serialized
    assert "records" not in artifact


def test_audit_fails_closed_when_pinned_archive_identity_changes(tmp_path: Path) -> None:
    protocol = _fake_protocol()
    archive_path, _secret_label = _write_fake_archive(tmp_path, protocol)
    protocol["dataset"]["archive"]["sha256"] = "0" * 64
    loaded_protocol, fingerprint = audit._load_protocol(
        _write_protocol(tmp_path / "protocol.json", protocol)
    )

    with pytest.raises(ValueError, match="archive SHA256 mismatch"):
        audit.audit_archive(loaded_protocol, fingerprint, archive_path)


def test_label_free_audit_does_not_decode_experimental_value_bytes(
    tmp_path: Path,
) -> None:
    protocol = _fake_protocol()
    archive_path, _secret_label = _write_fake_archive(
        tmp_path,
        protocol,
        invalid_label_bytes=True,
    )
    loaded_protocol, fingerprint = audit._load_protocol(
        _write_protocol(tmp_path / "protocol.json", protocol)
    )

    artifact = audit.audit_archive(loaded_protocol, fingerprint, archive_path)

    assert artifact["counts"]["panel_neutral_absolute_records"] == 105
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []


def test_committed_source_audit_is_aggregate_only_and_self_hashed() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == audit.core.artifact_content_sha256(artifact)
    assert artifact["counts"] == {
        "panel_neutral_absolute_records": 1106,
        "panel_solvents": 15,
        "panel_unique_neutral_geometry_pairs": 1106,
    }
    assert artifact["source_archive"]["source_path_recorded"] is False
    assert artifact["source_table"]["experimental_value_columns_loaded"] == []
    assert all(
        entry["neutral_absolute_records"] >= 7 for entry in artifact["panel"]
    )
    serialized = json.dumps(artifact, sort_keys=True)
    assert "DeltaGsolv" not in serialized
    assert "records" not in artifact
