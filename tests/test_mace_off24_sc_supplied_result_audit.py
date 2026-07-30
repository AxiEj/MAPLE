from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_mace_off24_sc_supplied_result_audit.py"
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("mace_off24_sc_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _records_with_constant_error(error: float):
    return [
        {
            "name": "record-a",
            "smiles": "C",
            "experimental_kcal_mol": 0.0,
            "predicted_kcal_mol": error,
        },
        {
            "name": "record-b",
            "smiles": "CC",
            "experimental_kcal_mol": 1.0,
            "predicted_kcal_mol": 1.0 + error,
        },
    ]


def _hydration_csv_text() -> str:
    rows = [",".join(AUDIT.HYDRATION_HEADER)]
    names = ["N,N-dimethylbenzamide", "1,4-dioxane"]
    names.extend(f"compound-{index}" for index in range(2, 36))
    for index, name in enumerate(names):
        rows.append(
            f"{name},SMILES-{index},{index}.0,{index + 0.1}," "0.2,0.3,0.4,0.5,0.6,0.7"
        )
    return "\n".join(rows)


def _octanol_table_text() -> str:
    rows = [
        "unrelated prefix",
        "Table S2: Solvation free energies in kcal/mol in octanol",
        "Compound SMILES Exp ExpError MACE MACEError GAFF",
    ]
    rows.extend(
        f"compound-{index} SMILES-{index} "
        f"{index}.0 0.1 {index + 0.25} 0.2 {index + 0.5}"
        for index in range(10)
    )
    rows.extend(
        [
            "S3        FreeSolv hydration free energy results",
            "unrelated suffix",
        ]
    )
    return "\n".join(rows)


def test_hydration_parser_recovers_unquoted_comma_names_from_the_right():
    records = AUDIT._read_hydration_csv(_hydration_csv_text())

    assert len(records) == 36
    assert records[0]["name"] == "N,N-dimethylbenzamide"
    assert records[1]["name"] == "1,4-dioxane"
    assert records[0]["smiles"] == "SMILES-0"


def test_octanol_parser_extracts_exact_table_s2_section():
    records = AUDIT._read_octanol_table(_octanol_table_text())

    assert len(records) == 10
    assert records[0] == {
        "name": "compound-0",
        "smiles": "SMILES-0",
        "experimental_kcal_mol": 0.0,
        "predicted_kcal_mol": 0.25,
    }
    assert records[-1]["name"] == "compound-9"


def test_metric_formulas_and_maximum_error_record():
    records = [
        {
            "name": "a",
            "smiles": "C",
            "experimental_kcal_mol": 0.0,
            "predicted_kcal_mol": 1.0,
        },
        {
            "name": "b",
            "smiles": "CC",
            "experimental_kcal_mol": 2.0,
            "predicted_kcal_mol": 0.0,
        },
        {
            "name": "c",
            "smiles": "CCC",
            "experimental_kcal_mol": 4.0,
            "predicted_kcal_mol": 5.0,
        },
    ]

    metrics, maximum = AUDIT._metrics(records)

    assert metrics["mae_kcal_mol"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse_kcal_mol"] == pytest.approx(math.sqrt(2.0))
    assert metrics["maximum_absolute_error_kcal_mol"] == 2.0
    assert metrics["mean_signed_error_kcal_mol"] == 0.0
    assert maximum == {"record_name": "b", "absolute_error_kcal_mol": 2.0}


def test_maximum_error_target_is_strictly_less_than_1_5():
    passing = AUDIT._panel_summary(
        _records_with_constant_error(1.499),
        dataset="synthetic",
        solvent="water",
        evidence_identity={},
    )
    boundary = AUDIT._panel_summary(
        _records_with_constant_error(1.5),
        dataset="synthetic",
        solvent="water",
        evidence_identity={},
    )

    assert passing["passes_maximum_error_target"] is True
    assert boundary["passes_maximum_error_target"] is False


def test_rounding_bound_rules_out_the_article_table_1_discrepancy():
    comparison = AUDIT._article_table_1_hydration_comparison(
        {
            "mae_kcal_mol": 0.7638888888888888,
            "rmse_kcal_mol": 0.8343893309214564,
        }
    )

    assert comparison["metric_gaps_recomputed_minus_reported_kcal_mol"] == (
        pytest.approx(
            {
                "mae_kcal_mol": 0.0738888888888888,
                "rmse_kcal_mol": 0.0343893309214564,
            }
        )
    )
    assert (
        comparison["rounding_bounds_kcal_mol"][
            "combined_conservative_display_upper_bound"
        ]
        == 0.03
    )
    assert comparison["rounding_alone_can_explain"] is False


def test_audit_payload_and_atomic_artifact_regenerate_deterministically(
    tmp_path, monkeypatch
):
    csv_path = tmp_path / "hydration.csv"
    pdf_path = tmp_path / "supporting.pdf"
    csv_path.write_text(_hydration_csv_text(), encoding="utf-8")
    pdf_path.write_bytes(b"synthetic PDF placeholder")
    monkeypatch.setattr(AUDIT, "_verify_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(AUDIT, "_extract_pdf_text", lambda path: _octanol_table_text())

    first_payload = AUDIT.audit_supplied_results(csv_path, pdf_path)
    second_payload = AUDIT.audit_supplied_results(csv_path, pdf_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    AUDIT._write_json_atomic(first_output, first_payload)
    AUDIT._write_json_atomic(second_output, second_payload)

    assert first_payload == second_payload
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_payload["validation_panels"]["hydration"]["record_count"] == 36
    assert first_payload["validation_panels"]["octanol"]["record_count"] == 10
