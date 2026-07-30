from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_solpropmix_candidate_holdout_audit.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "solpropmix-not-used-candidate-holdout-audit-2026-07-30.json"
)
WATCHLIST = ROOT / "docs" / "pretrained-solvation-hub" / "research_watchlist.yaml"
CONFIGURED_ARCHIVE = os.environ.get("MAPLE_SOLPROPMIX_AUDIT_ARCHIVE")


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "solpropmix_candidate_holdout_audit",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _column_name(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _inline_cell(reference: str, value: str) -> str:
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def _numeric_cell(reference: str, value: str) -> str:
    return f'<c r="{reference}" t="n"><v>{escape(value)}</v></c>'


def _synthetic_sheet(*, label_value: str = "SECRET-LABEL") -> bytes:
    specification = AUDIT.TARGET_SHEETS["Not used - From 3-comp VLE"]
    headers = specification["header"]
    header_cells = "".join(
        _inline_cell(f"{_column_name(index)}1", value)
        for index, value in enumerate(headers)
    )
    values = {
        "inchi_solute": "InChI=1S/CH4/h1H4",
        "DDB_name_solute": "1: Methane",
        "inchi_solvent1": "InChI=1S/H2O/h1H2",
        "DDB_name_solvent1": "174: Water",
        "inchi_solvent2": "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",
        "DDB_name_solvent2": "4: Acetone",
        "frac_solvent1": "0.25",
        "T (K)": "298.15",
        "P (kPa)": "101.325",
        "solute_exp_liq_mol_frac": "0.01",
        "Authors": "A. Author",
        "Journal": "Journal",
        "Volume": "1",
        "Issue": "",
        "Pages": "1-2",
        "Year": "2020",
        "Title": "Source title",
    }
    data_cells = []
    for index, header in enumerate(headers):
        reference = f"{_column_name(index)}2"
        if header == AUDIT.LABEL_COLUMN:
            data_cells.append(_numeric_cell(reference, label_value))
        elif header in AUDIT.NUMERIC_COLUMNS:
            data_cells.append(_numeric_cell(reference, values[header]))
        elif values.get(header):
            data_cells.append(_inline_cell(reference, values[header]))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        f'<dimension ref="A1:R2"/><sheetData>'
        f'<row r="1">{header_cells}</row>'
        f'<row r="2">{"".join(data_cells)}</row>'
        "</sheetData></worksheet>"
    )
    return xml.encode()


def test_static_parser_never_decodes_the_experimental_label_value(monkeypatch):
    specification = {
        **AUDIT.TARGET_SHEETS["Not used - From 3-comp VLE"],
        "expected_rows": 1,
    }
    original = AUDIT._cell_value

    def guarded_value(cell):
        if AUDIT._column_index(cell.attrib["r"]) == 9:
            raise AssertionError("experimental label value was decoded")
        return original(cell)

    monkeypatch.setattr(AUDIT, "_cell_value", guarded_value)
    summary = AUDIT._sheet_summary(_synthetic_sheet(), specification)

    assert summary["record_count"] == 1
    assert summary["label_cells_present_but_values_not_decoded"] == 1
    assert summary["experimental_label_values_embedded"] is False
    assert summary["models_scored"] == []
    assert "records" not in summary


def test_static_parser_rejects_schema_drift_formulas_and_invalid_composition():
    specification = {
        **AUDIT.TARGET_SHEETS["Not used - From 3-comp VLE"],
        "expected_rows": 1,
    }

    drifted = _synthetic_sheet().replace(b"inchi_solute", b"wrong_header", 1)
    with pytest.raises(ValueError, match="header"):
        AUDIT._sheet_summary(drifted, specification)

    formula = _synthetic_sheet().replace(
        b'<c r="H2" t="n"><v>298.15</v></c>',
        b'<c r="H2" t="n"><f>1+1</f><v>2</v></c>',
    )
    with pytest.raises(ValueError, match="formula"):
        AUDIT._sheet_summary(formula, specification)

    invalid_fraction = _synthetic_sheet().replace(b"<v>0.25</v>", b"<v>1.25</v>", 1)
    with pytest.raises(ValueError, match="fraction"):
        AUDIT._sheet_summary(invalid_fraction, specification)


def test_physical_mixture_identity_omits_zero_fraction_declared_components():
    three_component = {
        "inchi_solvent1": "InChI=1S/H2O/h1H2",
        "inchi_solvent2": "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",
        "frac_solvent1": "0.25",
    }
    four_component = {
        **three_component,
        "inchi_solvent3": "InChI=1S/CH4/h1H4",
        "frac_solvent2": "0.75",
    }

    three_identity, three_declared, _ = AUDIT._canonical_composition(
        three_component,
        solvent_component_count=2,
    )
    four_identity, four_declared, _ = AUDIT._canonical_composition(
        four_component,
        solvent_component_count=3,
    )

    assert three_identity == four_identity
    assert three_declared == (AUDIT.Decimal("0.25"), AUDIT.Decimal("0.75"))
    assert four_declared == (
        AUDIT.Decimal("0.25"),
        AUDIT.Decimal("0.75"),
        AUDIT.Decimal("0"),
    )
    assert all(component != "InChI=1S/CH4/h1H4" for component, _ in four_identity)


def test_frozen_candidate_artifact_preserves_non_scoring_non_final_boundary():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert payload["candidate_holdout"] is True
    assert payload["strict_final_holdout"] is False
    assert payload["identity"]["archive"]["sha256"] == AUDIT.ARCHIVE_SHA256
    assert payload["identity"]["archive"]["size_bytes"] == AUDIT.ARCHIVE_SIZE_BYTES
    assert payload["identity"]["members"][AUDIT.WORKBOOK_MEMBER]["sha256"] == (
        AUDIT.PINNED_MEMBERS[AUDIT.WORKBOOK_MEMBER]["sha256"]
    )
    assert payload["identity"]["members"][AUDIT.DICTIONARY_MEMBER]["sha256"] == (
        AUDIT.PINNED_MEMBERS[AUDIT.DICTIONARY_MEMBER]["sha256"]
    )
    assert payload["validation_scope"] == {
        "experimental_label_cells_counted_without_decoding_values": True,
        "experimental_label_values_embedded": False,
        "experimental_label_values_used_for_model_selection": False,
        "maximum_error_computed": False,
        "model_prediction_sheets_read": False,
        "model_predictions_read": False,
        "models_scored": [],
        "record_identities_embedded": False,
    }
    assert payload["overlap_audit"]["status"] == "pending"
    assert payload["overlap_audit"]["training_overlap_proven_absent"] is False
    boundary = payload["overlap_audit"]["published_solprop_mix_model_data_boundary"]
    assert boundary["pretraining_datasets"] == [
        "CombiSolv-QM",
        "BinarySolv-QM",
    ]
    assert boundary["fine_tuning_datasets"] == ["CombiSolv-Exp"]
    assert boundary["published_test_datasets"] == [
        "BinarySolv-Exp",
        "TernarySolv-Exp",
    ]
    assert boundary["split_unit"] == "solute"
    assert "do not certify" in boundary["scope_warning"]
    join_identity = payload["overlap_audit"][
        "candidate_identity_available_without_labels"
    ]
    assert join_identity["experimental_target_required_for_join"] is False
    assert join_identity["record_values_embedded"] is False
    assert set(payload["overlap_audit"]["required_join_levels"]) == {
        "exact_record",
        "solute_structure",
        "solvent_system",
        "source_provenance",
    }
    assert payload["final_holdout_gates"]["passes_all"] is False
    assert payload["final_holdout_gates"]["minimum_solvent_coverage"]["passes"] is True
    assert payload["final_holdout_gates"]["unique_record_identities"]["passes"] is False
    assert payload["final_holdout_gates"]["complete_core_provenance"]["passes"] is False
    assert payload["final_holdout_gates"]["training_overlap_absent"]["passes"] is False


def test_frozen_candidate_artifact_locks_identity_coverage_and_defects():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    three = payload["candidate_panels"]["not_used_from_3_component_vle"]
    four = payload["candidate_panels"]["not_used_from_4_component_vle"]

    assert three["record_count"] == 25_833
    assert three["unique_solute_count"] == 281
    assert three["unique_solvent_count"] == 269
    assert three["unique_mixture_count"] == 24_233
    assert three["duplicate_record_identity_count"] == 42
    assert three["records_with_incomplete_core_provenance"] == 48
    assert three["records_with_zero_fraction_declared_components"] == 27
    assert three["temperature_range_K"] == [288.15, 646.75]
    assert three["pressure_range_kPa"] == [1.44, 18_399.999]
    assert three["experimental_label_values_embedded"] is False

    assert four["record_count"] == 4_351
    assert four["unique_solute_count"] == 79
    assert four["unique_solvent_count"] == 80
    assert four["unique_mixture_count"] == 4_319
    assert four["duplicate_record_identity_count"] == 0
    assert four["records_with_incomplete_core_provenance"] == 0
    assert four["records_with_zero_fraction_declared_components"] == 20
    assert four["temperature_range_K"] == [303.15, 529.9]
    assert four["pressure_range_kPa"] == [0.12978, 500.0]
    assert four["experimental_label_values_embedded"] is False

    union = payload["candidate_union"]
    assert union["record_count"] == 30_184
    assert union["unique_solute_count"] == 293
    assert union["unique_solvent_count"] == 279
    assert union["unique_mixture_count"] == 28_551
    assert union["unique_solute_mixture_count"] == 29_117
    assert union["unique_citation_count"] == 833
    assert union["duplicate_record_identity_count"] == 43
    assert union["records_with_incomplete_core_provenance"] == 48
    assert union["records_with_zero_fraction_declared_components"] == 47
    assert union["mixture_identity_basis"].startswith(
        "positive-fraction physical components"
    )
    assert "records" not in union


def test_frozen_artifact_recomputes_summary_hashes_and_watchlist_binding():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    for key, specification in AUDIT.TARGET_SHEETS.items():
        summary_key = specification["summary_key"]
        summary = payload["candidate_panels"][summary_key]
        assert (
            summary["header_sha256"]
            == hashlib.sha256(
                AUDIT._canonical_bytes(specification["header"])
            ).hexdigest()
        )
        assert summary["worksheet_sha256"] == specification["sha256"]
        assert summary["worksheet_size_bytes"] == specification["size_bytes"]
        assert summary["sheet_name"] == key

    watchlist = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    candidate = next(
        model for model in watchlist["models"] if model["model_id"] == "solprop-mix-exp"
    )
    audit = candidate["candidate_holdout_audit"]
    assert audit["artifact"] == ARTIFACT.relative_to(WATCHLIST.parent).as_posix()
    assert audit["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert audit["strict_final_holdout"] is False
    assert audit["overlap_audit_status"] == "pending"
    assert audit["model_scoring_performed"] is False


def test_frozen_artifact_equals_fresh_exact_archive_audit(tmp_path):
    if CONFIGURED_ARCHIVE is None:
        pytest.skip(
            "Set MAPLE_SOLPROPMIX_AUDIT_ARCHIVE to run the exact archive integration test."
        )
    archive = Path(CONFIGURED_ARCHIVE)
    if not archive.is_file():
        pytest.fail(f"Configured SolProp-mix archive does not exist: {archive}")

    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive",
            str(archive),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.read_bytes() == ARTIFACT.read_bytes()
