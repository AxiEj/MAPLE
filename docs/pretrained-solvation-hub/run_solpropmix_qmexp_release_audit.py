#!/usr/bin/env python3
"""Reproduce the negative SolProp-mix QMExp release audit.

The frozen panel is split deliberately: ``panel-inputs.json`` is the only file
passed to the model worker, while labels and the official workbook prediction
columns remain in ``experimental-references.json``.  This is a development
panel/release-reproduction audit, not a blind holdout and not a PES/force audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from rdkit import Chem

AUDITED_ON = "2026-07-31"
ZENODO_RECORD = "https://zenodo.org/records/15587866"
ARCHIVE_SIZE_BYTES = 118_240_334
ARCHIVE_MD5 = "60831b5fd10e06f9f44845a2d6bc0367"
ARCHIVE_SHA256 = "9aef449e30a85eca1636580a7741bbb828fddda434c4cb50443f26f4d28dcf1e"
WORKBOOK_MEMBER = "Data/solprop-mix_v1.1.xlsx"
WORKBOOK_SIZE_BYTES = 15_837_066
WORKBOOK_MD5 = "b428a724a50fe66184746775cda0df24"
WORKBOOK_SHA256 = "793f325e6edddca3334b368f68edfc368027e472ad779e7e5d52ff5d1e1a001d"
SOURCE_REPOSITORY = "https://gitlab.kuleuven.be/creas/vermeiregroup/solprop"
SOURCE_REVISION = "80043ce09eb8802517c35b59254f8e9c181f2dac"
SOURCE_TREE = "bc7b933d6cf4c55c8ad10236383cfd09afbbfdc2"
TAXONOMY_RELATIVE_PATH = (
    "docs/pretrained-solvation-hub/functional-group-taxonomy-v1.json"
)
TAXONOMY_SHA256 = "46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36"
AUDIT_RUNNER_RELATIVE_PATH = (
    "docs/pretrained-solvation-hub/run_solpropmix_qmexp_release_audit.py"
)
ADAPTER_RELATIVE_PATH = "maple/function/solvfe/solpropmix_property.py"
EXPECTED_ADAPTER_SHA256 = (
    "0674b07888fc466fa3dd2c5adc17070d3434e513d44770e99b6af8485d4282e6"
)
EXPECTED_MODEL_WORKER_SHA256 = (
    "849d9f73610a35da497b55b6e12358a097a8dd61e084d67b1044daa62d577cc7"
)
BENCHMARK_RELATIVE_PATH = (
    "docs/pretrained-solvation-hub/benchmarks/"
    "solpropmix-qmexp-release-audit-2026-07-31"
)
RAW_EXECUTION_ARTIFACT_NAME = "raw-live-execution.json"
IMPORTED_RESULT_NAME = "imported-fixture-result.json"
IMPORTED_ROWS_NAME = "imported-fixture-rows.csv"
RELEASE_MAPPING_RELATIVE_PATH = (
    f"{BENCHMARK_RELATIVE_PATH}/v1.0-v1.1-exact-chemical-mapping.json"
)
RELEASE_MAPPING_SHA256 = (
    "945084372a4b004f2613460654f46076dd0e56bad3c54dd5d89a75a0def7babb"
)
SELECTION_PROVENANCE_RELATIVE_PATH = (
    f"{BENCHMARK_RELATIVE_PATH}/selection-provenance.json"
)
SELECTION_PROVENANCE_SHA256 = (
    "ea297a5ce3244661a75cd2f6f882fecc4b5223a7eff815dc3a8b0ba231f323e3"
)
PAPER_URL = "https://arxiv.org/abs/2412.01982"
REFERENCE_TEMPERATURE_K = 298.15
ROUND_OFF_TOLERANCE_KCAL_MOL = 5e-12
HEADLINE_MAE_MAX = 0.25
HEADLINE_RMSE_MAX = 0.37
VOLATILE_RUNTIME_RECEIPT_KEYS = {
    "worker_stderr_sha256",
    "worker_stderr_truncated",
    "worker_stdout_sha256",
}
ROW_SPECIFIC_RUNTIME_RECEIPT_KEYS = {
    "canonical_solute_smiles",
    "solvent_components",
}

EXPECTED_GROUPS = (
    "amide",
    "ester",
    "carboxylic_acid",
    "aldehyde",
    "ketone",
    "alcohol",
    "epoxide",
    "ether",
    "amine",
    "nitrile",
)
CHECKPOINT_SHA256 = (
    "e887f9d424134250eb6ae871e2d142f5f5e04a6a9b2e41c37e2d96935f30ffd5",
    "c6f005771274988e6c8b4c28f12d6172d2514fd1bbe543031871f79ab60ab02d",
    "44277a9bbdd06122c039c58d051b19b5f8ea4d9b6fcebae5c6408d903883fce2",
    "ca7cd715b235c0c69366dda9c1d06aac837a6cfdb959dee4db8a8d0366407edb",
    "45d2a83015c4965adb982e8a6f07d48899b1d45050d375aa9e77bb9f46856fcf",
    "5881eddaef2c22b167ed0de043743a917b1a6fe852f642443b08ba41552f10c1",
    "a571f5e48f52f61a028c82fbd5c7857133c719f808892483c9b27b0ff17b4b3c",
    "47256bfc9bf66d9cca38c378aca3a307317b891580979f6d18694abf1e9ae853",
    "4d6db1fc5099708fba40df71e2517be8ebf9816caee2ad38f099c2a4d178f97b",
    "0decb8b662177edae77fa66007ba5f9ad1abfe3e6405ffbb35110dc837a1660b",
)
STATIC_SHA256 = {
    "solvation_predictor/data/__init__.py": "2639b97ff1fe63a3c9e88402aa9d1369305be046762483db834325da22948a78",
    "solvation_predictor/data/data.py": "fd7bc4454b981686b3ef7fbb86d5069f9cb5eb7b4bbc454e3876ac61775aa83e",
    "solvation_predictor/data/Scaler.py": "5e4a65084db838238f14386af66672184d93764764b694b138fa97cf13dffa73",
    "solvation_predictor/data/Splitter.py": "f5244d923b5afec61e22d6d9ded1bef081545415b9b0efefe7282d8ab5e7d1ac",
}
WORKBOOK_PANELS = {
    "binary_nonaqueous": (
        "From IDAC - no water",
        "From VLE - Bin Solv - no water",
    ),
    "binary_all": (
        "From IDAC - has water",
        "From IDAC - no water",
        "From VLE - Bin Solv - has water",
        "From VLE - Bin Solv - no water",
    ),
    "ternary_all": (
        "From VLE - Ter Solv - has water",
        "From VLE - Ter Solv - no water",
    ),
}
EXPECTED_WORKBOOK_COUNTS = {
    "binary_nonaqueous": 22_564,
    "binary_all": 30_030,
    "ternary_all": 4_242,
}
LABEL_KEYS = {
    "experimental_gsolv_kcal_mol",
    "published_solpropmix_qm_exp_g298_kcal_mol",
    "published_solpropmix_qm_exp_h298_kcal_mol",
    "published_solpropmix_qm_exp_gt_kcal_mol",
}
INPUT_KEYS = (
    "row_id",
    "primary_functional_group",
    "all_matched_groups",
    "canonical_nonisomeric_smiles",
    "solute_inchi",
    "solute_name",
    "solvent_inchis",
    "solvent_names",
    "solvent_mole_fractions",
    "temperature_kelvin",
)
_EXECUTE_LIVE_CAPABILITY = object()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _render(value: Any) -> str:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_execute_live_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Remove stream-only receipt hashes from the bound execute-live artifact."""
    stable = json.loads(_render(raw))
    for payload in stable["modes"].values():
        for row in payload["rows"]:
            receipt = row["runtime_receipt"]
            for key in VOLATILE_RUNTIME_RECEIPT_KEYS:
                receipt.pop(key, None)
    return stable


def _metrics(
    errors: Iterable[float], *, stable_squares: bool = True
) -> dict[str, float | int]:
    values = tuple(float(value) for value in errors)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Metric inputs must be non-empty and finite.")
    squared_sum = (
        math.fsum(value * value for value in values)
        if stable_squares
        else sum(value * value for value in values)
    )
    return {
        "count": len(values),
        "mae_kcal_mol": math.fsum(abs(value) for value in values) / len(values),
        "rmse_kcal_mol": math.sqrt(squared_sum / len(values)),
        "maxae_kcal_mol": max(abs(value) for value in values),
    }


def _column_letter(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _row_id(row: Mapping[str, Any]) -> str:
    return f'{row["sheet"]}!{int(row["excel_row_number"])}'


def split_selection(
    selection: Sequence[Mapping[str, Any]],
    workbook_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a previously frozen, output-blind selection from its labels."""
    inputs = []
    references = []
    for selected in selection:
        row_id = _row_id(selected)
        item = {key: selected[key] for key in INPUT_KEYS if key != "row_id"}
        item["row_id"] = row_id
        inputs.append(item)
        prediction_column = "SolProp-mix_QM_Exp_GsolvT"
        experimental_column = "Gsolv (kcal/mol)"
        headers = workbook_header_columns(workbook_path, str(selected["sheet"]))
        experimental_index = headers[experimental_column]
        h298_index = headers["SolProp-mix_QM_Exp_Hsolv298"]
        g298_index = headers["SolProp-mix_QM_Exp_Gsolv298"]
        prediction_index = headers[prediction_column]
        excel_row = int(selected["excel_row_number"])
        references.append(
            {
                "row_id": row_id,
                "sheet": selected["sheet"],
                "excel_row_number": excel_row,
                "zero_based_dataframe_row_index": int(selected["zero_based_row_index"]),
                "experimental": {
                    "header": experimental_column,
                    "header_cell": f"{_column_letter(experimental_index)}1",
                    "value_cell": f"{_column_letter(experimental_index)}{excel_row}",
                    "value_kcal_mol": selected["experimental_gsolv_kcal_mol"],
                },
                "published_qmexp": {
                    "g298_header": "SolProp-mix_QM_Exp_Gsolv298",
                    "g298_header_cell": f"{_column_letter(g298_index)}1",
                    "g298_value_cell": f"{_column_letter(g298_index)}{excel_row}",
                    "g298_value_kcal_mol": selected[
                        "published_solpropmix_qm_exp_g298_kcal_mol"
                    ],
                    "h298_header": "SolProp-mix_QM_Exp_Hsolv298",
                    "h298_header_cell": f"{_column_letter(h298_index)}1",
                    "h298_value_cell": f"{_column_letter(h298_index)}{excel_row}",
                    "h298_value_kcal_mol": selected[
                        "published_solpropmix_qm_exp_h298_kcal_mol"
                    ],
                    "temperature_header": prediction_column,
                    "temperature_header_cell": f"{_column_letter(prediction_index)}1",
                    "temperature_value_cell": f"{_column_letter(prediction_index)}{excel_row}",
                    "temperature_value_kcal_mol": selected[
                        "published_solpropmix_qm_exp_gt_kcal_mol"
                    ],
                },
                "literature_reference": {
                    "authors": selected["authors"],
                    "journal": selected["journal"],
                    "year": selected["year"],
                    "title": selected["title"],
                },
            }
        )
    panel = {
        "schema_version": 1,
        "selection_boundary": {
            "selection_author_attests_chosen_without_experimental_or_model_outputs": True,
            "selection_author_attests_rows_not_changed_after_outputs_were_read": True,
            "development_panel_not_blind_holdout": True,
            "primary_group_precedence_frozen": list(EXPECTED_GROUPS),
            "taxonomy": {"path": TAXONOMY_RELATIVE_PATH, "sha256": TAXONOMY_SHA256},
            "exact_release_mapping_sha256": RELEASE_MAPPING_SHA256,
        },
        "model_worker_may_read_only_this_file": True,
        "records": inputs,
    }
    refs = {
        "schema_version": 1,
        "must_not_be_supplied_to_model_worker": True,
        "workbook": {
            "member": WORKBOOK_MEMBER,
            "size_bytes": WORKBOOK_SIZE_BYTES,
            "md5": WORKBOOK_MD5,
            "sha256": WORKBOOK_SHA256,
        },
        "records": references,
    }
    _validate_panel(panel, refs)
    return panel, refs


def _validate_panel(panel: Mapping[str, Any], references: Mapping[str, Any]) -> None:
    rows = panel["records"]
    refs = references["records"]
    if len(rows) != 10 or len(refs) != 10:
        raise ValueError("Release audit requires exactly ten frozen rows.")
    groups = tuple(row["primary_functional_group"] for row in rows)
    if groups != EXPECTED_GROUPS or len(set(groups)) != 10:
        raise ValueError(
            "Release audit requires the exact ten distinct primary groups."
        )
    for row in rows:
        if LABEL_KEYS.intersection(row):
            raise ValueError(
                "Model-worker panel contains a forbidden label or workbook prediction."
            )
        if set(row) != set(INPUT_KEYS):
            raise ValueError("Model-worker panel schema drifted.")
        fractions = tuple(float(value) for value in row["solvent_mole_fractions"])
        if not 1 <= len(fractions) <= 3 or len(fractions) != len(row["solvent_inchis"]):
            raise ValueError("Panel solvent composition is invalid.")
        if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Panel solvent fractions must sum to one.")
    if [row["row_id"] for row in rows] != [row["row_id"] for row in refs]:
        raise ValueError("Panel/reference row identities do not align.")
    boundary = panel["selection_boundary"]
    if boundary.get("exact_release_mapping_sha256") != RELEASE_MAPPING_SHA256:
        raise ValueError("Panel release-mapping identity changed.")
    if (
        boundary.get(
            "selection_author_attests_chosen_without_experimental_or_model_outputs"
        )
        is not True
        or boundary.get(
            "selection_author_attests_rows_not_changed_after_outputs_were_read"
        )
        is not True
    ):
        raise ValueError("Panel selection attestation is missing.")


SHEET_PATHS = {
    "From IDAC - has water": "xl/worksheets/sheet1.xml",
    "From IDAC - no water": "xl/worksheets/sheet2.xml",
    "From VLE - Bin Solv - has water": "xl/worksheets/sheet3.xml",
    "From VLE - Bin Solv - no water": "xl/worksheets/sheet4.xml",
    "From VLE - Ter Solv - has water": "xl/worksheets/sheet5.xml",
    "From VLE - Ter Solv - no water": "xl/worksheets/sheet6.xml",
}
XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ROW_TAG = f"{{{XML_NS}}}row"
CELL_TAG = f"{{{XML_NS}}}c"
VALUE_TAG = f"{{{XML_NS}}}v"
TEXT_TAG = f"{{{XML_NS}}}t"


def _shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ()
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return tuple(
        "".join(node.text or "" for node in item.iter(TEXT_TAG)) for item in root
    )


def _cell_value(
    cell: ElementTree.Element,
    shared_strings: Sequence[str] = (),
) -> str | float | None:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(TEXT_TAG))
    value = cell.find(VALUE_TAG)
    if value is None or value.text is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    return float(value.text)


def workbook_header_columns(workbook_path: Path, sheet: str) -> dict[str, int]:
    """Return one-based Excel column indices by exact header name."""
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _shared_strings(archive)
        with archive.open(SHEET_PATHS[sheet]) as handle:
            for _, row in ElementTree.iterparse(handle, events=("end",)):
                if row.tag != ROW_TAG:
                    continue
                if int(row.attrib["r"]) != 1:
                    raise ValueError(f"Missing header row in {sheet}.")
                result = {}
                for cell in row.findall(CELL_TAG):
                    reference = cell.attrib["r"]
                    letters = reference.rstrip("0123456789")
                    index = 0
                    for character in letters:
                        index = index * 26 + ord(character) - ord("A") + 1
                    result[str(_cell_value(cell, shared_strings))] = index
                return result
    raise ValueError(f"Missing worksheet {sheet}.")


def _scan_sheet(
    handle,
    wanted_cells: set[str],
    shared_strings: Sequence[str],
) -> tuple[list[float], dict[str, Any], int]:
    errors: list[float] = []
    captured: dict[str, Any] = {}
    experimental_column = None
    prediction_column = None
    count = 0
    for _, row in ElementTree.iterparse(handle, events=("end",)):
        if row.tag != ROW_TAG:
            continue
        row_number = int(row.attrib["r"])
        cells = {
            cell.attrib["r"]: _cell_value(cell, shared_strings)
            for cell in row.findall(CELL_TAG)
        }
        for reference in wanted_cells.intersection(cells):
            captured[reference] = cells[reference]
        if row_number == 1:
            by_header = {
                value: reference.rstrip("0123456789")
                for reference, value in cells.items()
            }
            experimental_column = by_header["Gsolv (kcal/mol)"]
            prediction_column = by_header["SolProp-mix_QM_Exp_GsolvT"]
        else:
            experimental = cells.get(f"{experimental_column}{row_number}")
            prediction = cells.get(f"{prediction_column}{row_number}")
            if experimental is not None and prediction is not None:
                errors.append(float(prediction) - float(experimental))
                count += 1
        row.clear()
    return errors, captured, count


def audit_workbook(
    workbook_path: Path, references: Mapping[str, Any]
) -> dict[str, Any]:
    """Stream the exact workbook once for provenance and all published metrics."""
    if workbook_path.stat().st_size != WORKBOOK_SIZE_BYTES:
        raise ValueError("Official SolProp-mix v1.1 workbook size mismatch.")
    if _md5_file(workbook_path) != WORKBOOK_MD5:
        raise ValueError("Official SolProp-mix v1.1 workbook MD5 mismatch.")
    if _sha256_file(workbook_path) != WORKBOOK_SHA256:
        raise ValueError("Official SolProp-mix v1.1 workbook SHA256 mismatch.")
    wanted: dict[str, set[str]] = {name: set() for name in SHEET_PATHS}
    expected_cells: dict[tuple[str, str], Any] = {}
    for reference in references["records"]:
        sheet = reference["sheet"]
        for section in (reference["experimental"], reference["published_qmexp"]):
            for key, value in section.items():
                if key.endswith("_cell"):
                    wanted[sheet].add(value)
                    value_key = key.replace("_cell", "")
                    expected_key = value_key.replace("value", "value_kcal_mol")
                    expected_cells[(sheet, value)] = section.get(
                        expected_key, section.get(value_key)
                    )
    sheet_errors: dict[str, list[float]] = {}
    sheet_counts: dict[str, int] = {}
    captured: dict[tuple[str, str], Any] = {}
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _shared_strings(archive)
        for sheet, member in SHEET_PATHS.items():
            with archive.open(member) as handle:
                errors, values, count = _scan_sheet(
                    handle, wanted[sheet], shared_strings
                )
            sheet_errors[sheet] = errors
            sheet_counts[sheet] = count
            captured.update({(sheet, cell): value for cell, value in values.items()})
    for key, expected in expected_cells.items():
        observed = captured.get(key)
        if isinstance(expected, (int, float)):
            if observed is None or not math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Workbook provenance mismatch at {key[0]}!{key[1]}.")
        elif observed != expected:
            raise ValueError(f"Workbook header mismatch at {key[0]}!{key[1]}.")
    result: dict[str, Any] = {}
    for panel_name, sheets in WORKBOOK_PANELS.items():
        errors = [value for sheet in sheets for value in sheet_errors[sheet]]
        metrics = _metrics(errors)
        if metrics["count"] != EXPECTED_WORKBOOK_COUNTS[panel_name]:
            raise ValueError(f"Published workbook row count changed for {panel_name}.")
        result[panel_name] = {
            **metrics,
            "sheets": {sheet: sheet_counts[sheet] for sheet in sheets},
            "prediction_source": "published_workbook_column_not_live_checkpoint_execution",
        }
    return result


def reproduce_published_workbook_columns(workbook_path: Path) -> dict[str, Any]:
    """Compatibility helper for tests without selected-row provenance."""
    return audit_workbook(workbook_path, {"records": []})


def audit_release_mapping(
    repository_root: Path,
    panel: Mapping[str, Any],
    references: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate frozen v1.0→v1.1 row mapping and recompute closeness claims."""
    mapping_path = repository_root / RELEASE_MAPPING_RELATIVE_PATH
    selection_path = repository_root / SELECTION_PROVENANCE_RELATIVE_PATH
    if _sha256_file(mapping_path) != RELEASE_MAPPING_SHA256:
        raise ValueError("Frozen v1.0→v1.1 release mapping identity changed.")
    if _sha256_file(selection_path) != SELECTION_PROVENANCE_SHA256:
        raise ValueError("Frozen selection-provenance identity changed.")
    mapping = _required_mapping(
        json.loads(mapping_path.read_text(encoding="utf-8")), "release mapping"
    )
    selection = _required_mapping(
        json.loads(selection_path.read_text(encoding="utf-8")),
        "selection provenance",
    )
    mapping_rows = _required_list(mapping.get("records"), "release mapping records")
    selection_rows = _required_list(
        selection.get("records"), "selection provenance records"
    )
    panel_rows = _required_list(panel.get("records"), "panel records")
    reference_rows = _required_list(references.get("records"), "reference records")
    if not (
        len(mapping_rows)
        == len(selection_rows)
        == len(panel_rows)
        == len(reference_rows)
        == len(scored_rows)
        == 10
    ):
        raise ValueError("Release mapping must align exactly with all ten panel rows.")
    attestation = _required_mapping(
        selection.get("selection_author_attestation"), "selection attestation"
    )
    if attestation.get("independently_provable_from_frozen_artifacts") is not False:
        raise ValueError("Selection provenance must preserve its proof limitation.")

    v1_0_deltas = []
    v1_1_deltas = []
    closer_count = 0
    for index, (
        mapping_value,
        selection_value,
        panel_value,
        reference_value,
        scored,
    ) in enumerate(
        zip(
            mapping_rows,
            selection_rows,
            panel_rows,
            reference_rows,
            scored_rows,
            strict=True,
        )
    ):
        mapping_row = _required_mapping(mapping_value, f"mapping row {index}")
        selection_row = _required_mapping(selection_value, f"selection row {index}")
        panel_row = _required_mapping(panel_value, f"panel row {index}")
        reference_row = _required_mapping(reference_value, f"reference row {index}")
        new_row_id = f'{mapping_row["sheet"]}!{int(mapping_row["new_excel_row_v1_1"])}'
        if not (
            new_row_id
            == panel_row["row_id"]
            == reference_row["row_id"]
            == scored["row_id"]
            == selection_row["new_row_id_v1_1"]
        ):
            raise ValueError("Release mapping row identity does not align.")
        for key in (
            "primary_functional_group",
            "solute_inchi",
            "solute_name",
            "solvent_inchis",
            "solvent_mole_fractions",
            "temperature_kelvin",
        ):
            if mapping_row[key] != panel_row[key]:
                raise ValueError(f"Release mapping input mismatch for {key}.")
            if selection_row[key] != panel_row[key]:
                raise ValueError(f"Selection provenance input mismatch for {key}.")
        if LABEL_KEYS.intersection(selection_row):
            raise ValueError("Selection provenance must remain label-free.")
        experimental = _finite_number(
            reference_row["experimental"]["value_kcal_mol"],
            f"reference row {index} experiment",
        )
        if mapping_row["experimental_gsolv_kcal_mol"] != experimental:
            raise ValueError("Release mapping experimental value mismatch.")
        v1_1_prediction = _finite_number(
            reference_row["published_qmexp"]["temperature_value_kcal_mol"],
            f"reference row {index} v1.1 prediction",
        )
        if mapping_row["qmexp_prediction_v1_1_kcal_mol"] != v1_1_prediction:
            raise ValueError("Release mapping v1.1 prediction mismatch.")
        v1_0_prediction = _finite_number(
            mapping_row["qmexp_prediction_v1_0_kcal_mol"],
            f"mapping row {index} v1.0 prediction",
        )
        expected_delta = v1_1_prediction - v1_0_prediction
        if (
            mapping_row["qmexp_prediction_delta_v1_1_minus_v1_0_kcal_mol"]
            != expected_delta
        ):
            raise ValueError("Release mapping prediction delta mismatch.")
        live = _finite_number(
            scored["cpu_fp64_prediction_kcal_mol"],
            f"scored row {index} live prediction",
        )
        old_delta = abs(live - v1_0_prediction)
        new_delta = abs(live - v1_1_prediction)
        v1_0_deltas.append(old_delta)
        v1_1_deltas.append(new_delta)
        closer_count += int(new_delta < old_delta)

    v1_0_mean = math.fsum(v1_0_deltas) / len(v1_0_deltas)
    v1_1_mean = math.fsum(v1_1_deltas) / len(v1_1_deltas)
    return {
        "mapping_path": RELEASE_MAPPING_RELATIVE_PATH,
        "mapping_sha256": RELEASE_MAPPING_SHA256,
        "selection_provenance_path": SELECTION_PROVENANCE_RELATIVE_PATH,
        "selection_provenance_sha256": SELECTION_PROVENANCE_SHA256,
        "v1_0_mean_abs_live_delta_kcal_mol": v1_0_mean,
        "v1_0_max_abs_live_delta_kcal_mol": max(v1_0_deltas),
        "v1_1_mean_abs_live_delta_kcal_mol": v1_1_mean,
        "v1_1_max_abs_live_delta_kcal_mol": max(v1_1_deltas),
        "v1_1_rows_closer_to_live_count": closer_count,
        "row_count": len(v1_1_deltas),
        "v1_1_broadly_closer_than_v1_0": (
            v1_1_mean < v1_0_mean and closer_count > len(v1_1_deltas) / 2
        ),
        "historical_output_blind_selection_is_author_attestation_not_independent_proof": True,
    }


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys.")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    return value


def _required_sequence(value: Any, label: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _canonical_smiles(structure: str, label: str) -> str:
    molecule = (
        Chem.MolFromInchi(structure)
        if structure.startswith("InChI=")
        else Chem.MolFromSmiles(structure)
    )
    if molecule is None:
        raise ValueError(f"{label} is not a valid molecular structure.")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _normalized_live_rows(
    raw: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    joined_rows = _required_list(raw.get("rows"), "live rows")
    modes = _required_mapping(raw.get("modes"), "live modes")
    panel_rows = _required_list(panel.get("records"), "panel records")
    mode_rows: dict[str, list[Any]] = {}
    for mode in ("cpu", "cuda:0"):
        payload = _required_mapping(modes.get(mode), f"{mode} payload")
        mode_rows[mode] = _required_list(payload.get("rows"), f"{mode} rows")
    if not (
        len(joined_rows)
        == len(mode_rows["cpu"])
        == len(mode_rows["cuda:0"])
        == len(panel_rows)
        == 10
    ):
        raise ValueError("Live result must contain ten aligned CPU/GPU rows.")

    expected_safety = {
        "deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "amp_autocast_enabled": False,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
    }
    representative_receipts: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for row_index, input_value in enumerate(panel_rows):
        input_row = _required_mapping(input_value, f"panel row {row_index}")
        joined = _required_mapping(joined_rows[row_index], f"joined row {row_index}")
        row_id = input_row["row_id"]
        if joined.get("row_id") != row_id:
            raise ValueError(
                "Joined live-result row identity does not match the panel."
            )
        if (
            joined.get("primary_functional_group")
            != input_row["primary_functional_group"]
        ):
            raise ValueError("Live result group order does not match the frozen panel.")

        predictions: dict[str, float] = {}
        models_by_mode: dict[str, list[dict[str, Any]]] = {}
        for mode, expected_device, joined_key in (
            ("cpu", "cpu", "cpu_prediction_kcal_mol"),
            ("cuda:0", "cuda:0", "gpu_prediction_kcal_mol"),
        ):
            mode_row = _required_mapping(
                mode_rows[mode][row_index], f"{mode} row {row_index}"
            )
            raw_models = _required_list(
                mode_row.get("models"), f"{mode} row {row_index} models"
            )
            if len(raw_models) != 10:
                raise ValueError("Each live row requires all ten ensemble members.")
            models = []
            for model_index, model_value in enumerate(raw_models):
                model = _required_mapping(
                    model_value, f"{mode} row {row_index} model {model_index}"
                )
                if model.get("model_index") != model_index:
                    raise ValueError(
                        "Live ensemble model indices must be exactly 0..9."
                    )
                models.append(
                    {
                        "model_index": model_index,
                        "g_temperature_kcal_mol": _finite_number(
                            model.get("g_temperature_kcal_mol"),
                            f"{mode} row {row_index} model {model_index} prediction",
                        ),
                    }
                )
            prediction = _finite_number(
                mode_row.get("prediction"), f"{mode} row {row_index} prediction"
            )
            ensemble_mean = math.fsum(
                model["g_temperature_kcal_mol"] for model in models
            ) / len(models)
            if prediction != ensemble_mean:
                raise ValueError(
                    "Live mode prediction does not equal the ten-model ensemble mean."
                )
            joined_prediction = _finite_number(
                joined.get(joined_key),
                f"joined {mode} row {row_index} prediction",
            )
            if joined_prediction != prediction:
                raise ValueError(
                    "Joined live prediction does not match its mode-row prediction."
                )

            receipt = _required_mapping(
                mode_row.get("runtime_receipt"),
                f"{mode} row {row_index} runtime receipt",
            )
            if receipt.get("execution_device") != expected_device:
                raise ValueError(
                    "Runtime receipt execution device does not match mode."
                )
            if (
                receipt.get("execution_dtype") != "float64"
                or receipt.get("precision") != "float64_promoted"
            ):
                raise ValueError("Release audit requires promoted float64 execution.")
            if any(receipt.get(key) != value for key, value in expected_safety.items()):
                raise ValueError(
                    "Strict deterministic no-reduced-precision policy was not observed."
                )
            if receipt.get("source_revision") != SOURCE_REVISION:
                raise ValueError("Runtime receipt source revision mismatch.")
            if receipt.get("source_tree") != SOURCE_TREE:
                raise ValueError("Runtime receipt source tree mismatch.")
            if receipt.get("worker_sha256") != EXPECTED_MODEL_WORKER_SHA256:
                raise ValueError("Runtime receipt worker identity mismatch.")
            if tuple(receipt.get("checkpoint_sha256", ())) != CHECKPOINT_SHA256:
                raise ValueError("Runtime receipt checkpoint identity mismatch.")
            if dict(receipt.get("static_sha256", ())) != STATIC_SHA256:
                raise ValueError(
                    "Runtime receipt supplemental-source identity mismatch."
                )
            if (
                receipt.get("canonical_solute_smiles")
                != input_row["canonical_nonisomeric_smiles"]
            ):
                raise ValueError(
                    "Runtime receipt solute identity does not match panel."
                )
            receipt_solvents = _required_sequence(
                receipt.get("solvent_components"),
                f"{mode} row {row_index} receipt solvents",
            )
            if len(receipt_solvents) != len(input_row["solvent_inchis"]):
                raise ValueError("Runtime receipt solvent count does not match panel.")
            observed_components = sorted(
                (
                    str(
                        _required_mapping(component, "receipt solvent").get(
                            "canonical_smiles"
                        )
                    ),
                    _finite_number(
                        _required_mapping(component, "receipt solvent").get(
                            "mole_fraction"
                        ),
                        "receipt solvent mole fraction",
                    ),
                )
                for component in receipt_solvents
            )
            expected_components = sorted(
                (
                    _canonical_smiles(str(inchi), "panel solvent"),
                    float(fraction),
                )
                for inchi, fraction in zip(
                    input_row["solvent_inchis"],
                    input_row["solvent_mole_fractions"],
                    strict=True,
                )
            )
            if observed_components != expected_components:
                raise ValueError(
                    "Runtime receipt solvent identity does not match panel."
                )

            invariant_receipt = {
                key: value
                for key, value in receipt.items()
                if key
                not in VOLATILE_RUNTIME_RECEIPT_KEYS | ROW_SPECIFIC_RUNTIME_RECEIPT_KEYS
            }
            previous = representative_receipts.setdefault(mode, invariant_receipt)
            if previous != invariant_receipt:
                raise ValueError(
                    f"{mode} runtime receipt invariants changed between panel rows."
                )
            predictions[mode] = prediction
            models_by_mode[mode] = models

        normalized.append(
            {
                "row_id": row_id,
                "cpu_prediction_kcal_mol": predictions["cpu"],
                "gpu_prediction_kcal_mol": predictions["cuda:0"],
                "models": [
                    {
                        "model_index": model_index,
                        "cpu_g_temperature_kcal_mol": models_by_mode["cpu"][
                            model_index
                        ]["g_temperature_kcal_mol"],
                        "gpu_g_temperature_kcal_mol": models_by_mode["cuda:0"][
                            model_index
                        ]["g_temperature_kcal_mol"],
                    }
                    for model_index in range(10)
                ],
            }
        )
    return normalized, representative_receipts


def _gpu_policy(
    rows: Sequence[Mapping[str, Any]], tolerance: float = ROUND_OFF_TOLERANCE_KCAL_MOL
) -> dict[str, Any]:
    exact_row_deteriorations = [
        row["row_id"]
        for row in rows
        if row["gpu_abs_error_kcal_mol"] > row["cpu_abs_error_kcal_mol"]
    ]
    prediction_bitwise_mismatches = [
        row["row_id"]
        for row in rows
        if row["gpu_fp64_prediction_kcal_mol"] != row["cpu_fp64_prediction_kcal_mol"]
    ]
    model_bitwise_mismatches = [
        f'{row["row_id"]}:model{model["model_index"]}'
        for row in rows
        for model in row["model_deltas"]
        if model["abs_gpu_minus_cpu_kcal_mol"] != 0.0
    ]
    row_deteriorations = [
        row["row_id"]
        for row in rows
        if row["gpu_abs_error_kcal_mol"] > row["cpu_abs_error_kcal_mol"] + tolerance
    ]
    model_deteriorations = [
        f'{row["row_id"]}:model{model["model_index"]}'
        for row in rows
        for model in row["model_deltas"]
        if model["abs_gpu_minus_cpu_kcal_mol"] > tolerance
    ]
    maximum = max(
        model["abs_gpu_minus_cpu_kcal_mol"]
        for row in rows
        for model in row["model_deltas"]
    )
    passes_strict_zero_loss = not (
        exact_row_deteriorations
        or prediction_bitwise_mismatches
        or model_bitwise_mismatches
    )
    return {
        "policy": (
            "strict_zero_loss_required_for_gpu_admission; f64_roundoff_tolerance "
            "reported_as_diagnostic_only"
        ),
        "tolerance_kcal_mol": tolerance,
        "exact_row_error_deteriorations": exact_row_deteriorations,
        "prediction_bitwise_mismatches": prediction_bitwise_mismatches,
        "model_bitwise_mismatches": model_bitwise_mismatches,
        "row_error_deteriorations": row_deteriorations,
        "model_deteriorations": model_deteriorations,
        "max_abs_per_model_gpu_minus_cpu_kcal_mol": maximum,
        "passes_roundoff_parity": not row_deteriorations and not model_deteriorations,
        "passes_strict_zero_loss": passes_strict_zero_loss,
        "bitwise_equal_claimed": passes_strict_zero_loss,
        "gpu_admission_granted": False,
    }


def _build_audit(
    panel: Mapping[str, Any],
    references: Mapping[str, Any],
    raw_live_results: Mapping[str, Any],
    workbook_path: Path,
    *,
    repository_root: Path,
    raw_input_sha256: str,
    canonical_live_execution: bool,
) -> dict[str, Any]:
    evidence_origin = (
        "execute_live" if canonical_live_execution else "imported_trusted_fixture"
    )
    _validate_panel(panel, references)
    published = audit_workbook(workbook_path, references)
    live, runtime_receipts = _normalized_live_rows(raw_live_results, panel)
    reference_by_id = {row["row_id"]: row for row in references["records"]}
    input_by_id = {row["row_id"]: row for row in panel["records"]}
    scored_rows = []
    for prediction in live:
        row_id = prediction["row_id"]
        input_row = input_by_id[row_id]
        reference = reference_by_id[row_id]
        experimental = float(reference["experimental"]["value_kcal_mol"])
        published_value = float(
            reference["published_qmexp"]["temperature_value_kcal_mol"]
        )
        cpu = float(prediction["cpu_prediction_kcal_mol"])
        gpu = float(prediction["gpu_prediction_kcal_mol"])
        models = []
        for model in prediction["models"]:
            delta = (
                model["gpu_g_temperature_kcal_mol"]
                - model["cpu_g_temperature_kcal_mol"]
            )
            models.append(
                {
                    **model,
                    "gpu_minus_cpu_kcal_mol": delta,
                    "abs_gpu_minus_cpu_kcal_mol": abs(delta),
                }
            )
        scored_rows.append(
            {
                "row_id": row_id,
                "primary_functional_group": input_row["primary_functional_group"],
                "solute": {
                    "name": input_row["solute_name"],
                    "inchi": input_row["solute_inchi"],
                },
                "solvent_composition": [
                    {"name": name, "inchi": inchi, "mole_fraction": fraction}
                    for name, inchi, fraction in zip(
                        input_row["solvent_names"],
                        input_row["solvent_inchis"],
                        input_row["solvent_mole_fractions"],
                        strict=True,
                    )
                ],
                "temperature_kelvin": input_row["temperature_kelvin"],
                "experimental_gsolv_kcal_mol": experimental,
                "published_workbook_qmexp_prediction_kcal_mol": published_value,
                "cpu_fp64_prediction_kcal_mol": cpu,
                "cpu_signed_error_kcal_mol": cpu - experimental,
                "cpu_abs_error_kcal_mol": abs(cpu - experimental),
                "gpu_fp64_prediction_kcal_mol": gpu,
                "gpu_signed_error_kcal_mol": gpu - experimental,
                "gpu_abs_error_kcal_mol": abs(gpu - experimental),
                "live_cpu_minus_published_workbook_kcal_mol": cpu - published_value,
                "model_deltas": models,
            }
        )
    cpu_metrics = _metrics(
        (row["cpu_signed_error_kcal_mol"] for row in scored_rows),
        stable_squares=False,
    )
    gpu_metrics = _metrics(
        (row["gpu_signed_error_kcal_mol"] for row in scored_rows),
        stable_squares=False,
    )
    release_mapping = audit_release_mapping(
        repository_root,
        panel,
        references,
        scored_rows,
    )
    gpu = _gpu_policy(scored_rows)

    adapter_path = repository_root / ADAPTER_RELATIVE_PATH
    taxonomy_path = repository_root / TAXONOMY_RELATIVE_PATH
    audit_runner_path = repository_root / AUDIT_RUNNER_RELATIVE_PATH
    adapter_sha = _sha256_file(adapter_path)
    taxonomy_sha = _sha256_file(taxonomy_path)
    audit_runner_sha = _sha256_file(audit_runner_path)
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise ValueError(
            "Adapter source changed; rerun and review the scientific audit."
        )
    if taxonomy_sha != TAXONOMY_SHA256:
        raise ValueError("Functional-group taxonomy identity changed.")
    worker_path = adapter_path.with_name("_solpropmix_worker.py")
    if not worker_path.is_file():
        raise ValueError("Adapter isolated model-worker source is missing.")
    worker_sha = _sha256_file(worker_path)
    if worker_sha != EXPECTED_MODEL_WORKER_SHA256:
        raise ValueError(
            "Isolated model-worker source changed; rerun and review the audit."
        )

    accuracy_passes = (
        cpu_metrics["mae_kcal_mol"] <= HEADLINE_MAE_MAX
        and cpu_metrics["rmse_kcal_mol"] <= HEADLINE_RMSE_MAX
    )
    reported_gpu_policy = dict(gpu)
    if not canonical_live_execution:
        reported_gpu_policy.update(
            {
                "passes_strict_zero_loss": False,
                "bitwise_equal_claimed": False,
                "gpu_admission_granted": False,
            }
        )
    execution_section = {
        "evidence_origin": evidence_origin,
        "live_execution_claimed": canonical_live_execution,
        "scientific_or_admission_claim_eligible": canonical_live_execution,
        "precision": "float64_promoted",
        "ensemble_size": 10,
        "rows": scored_rows,
        "cpu_metrics": cpu_metrics,
        "gpu_metrics": gpu_metrics,
        "release_vs_workbook": {
            "mean_abs_delta_kcal_mol": release_mapping[
                "v1_1_mean_abs_live_delta_kcal_mol"
            ],
            "max_abs_delta_kcal_mol": release_mapping[
                "v1_1_max_abs_live_delta_kcal_mol"
            ],
            "upstream_oracle_reproduced": (
                canonical_live_execution
                and release_mapping["v1_1_max_abs_live_delta_kcal_mol"]
                <= ROUND_OFF_TOLERANCE_KCAL_MOL
            ),
            "v1_0_mean_abs_live_delta_kcal_mol": release_mapping[
                "v1_0_mean_abs_live_delta_kcal_mol"
            ],
            "v1_0_max_abs_live_delta_kcal_mol": release_mapping[
                "v1_0_max_abs_live_delta_kcal_mol"
            ],
            "v1_1_rows_closer_to_live_count": release_mapping[
                "v1_1_rows_closer_to_live_count"
            ],
            "v1_1_broadly_closer_than_v1_0_on_frozen_panel": (
                canonical_live_execution
                and release_mapping["v1_1_broadly_closer_than_v1_0"]
            ),
        },
        "gpu_no_loss_policy": reported_gpu_policy,
    }
    result = {
        "schema_version": 1,
        "audited_on": AUDITED_ON,
        "acceptance_eligible": False,
        "evidence_provenance": {
            "origin": evidence_origin,
            "raw_input_sha256": raw_input_sha256,
            "raw_input_normalization": (
                "execute_live_json_with_volatile_worker_stream_receipt_keys_omitted"
                if canonical_live_execution
                else "none_exact_imported_file_bytes"
            ),
            "canonical_live_execution_evidence": canonical_live_execution,
            "trusted_fixture_not_independent_execution": not canonical_live_execution,
        },
        "identity": {
            "official_archive": {
                "zenodo_record": ZENODO_RECORD,
                "filename": "Data.zip",
                "size_bytes": ARCHIVE_SIZE_BYTES,
                "md5": ARCHIVE_MD5,
                "sha256": ARCHIVE_SHA256,
            },
            "workbook": {
                "member": WORKBOOK_MEMBER,
                "size_bytes": WORKBOOK_SIZE_BYTES,
                "md5": WORKBOOK_MD5,
                "sha256": WORKBOOK_SHA256,
            },
            "source": {
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "tree": SOURCE_TREE,
            },
            "checkpoint_sha256": [
                {"model_index": index, "filename": f"model{index}.pt", "sha256": digest}
                for index, digest in enumerate(CHECKPOINT_SHA256)
            ],
            "supplemental_runtime_sha256": STATIC_SHA256,
            "taxonomy": {"path": TAXONOMY_RELATIVE_PATH, "sha256": taxonomy_sha},
            "execution_code": {
                "audit_runner_path": AUDIT_RUNNER_RELATIVE_PATH,
                "audit_runner_sha256": audit_runner_sha,
                "adapter_path": ADAPTER_RELATIVE_PATH,
                "adapter_sha256": adapter_sha,
                "isolated_model_worker_path": "maple/function/solvfe/_solpropmix_worker.py",
                "isolated_model_worker_sha256": worker_sha,
            },
            "panel_inputs_sha256": _sha256_bytes(_canonical_bytes(panel)),
            "experimental_references_sha256": _sha256_bytes(
                _canonical_bytes(references)
            ),
            "release_mapping": {
                "path": release_mapping["mapping_path"],
                "sha256": release_mapping["mapping_sha256"],
            },
            "selection_provenance": {
                "path": release_mapping["selection_provenance_path"],
                "sha256": release_mapping["selection_provenance_sha256"],
            },
            "runtime_receipts": runtime_receipts,
        },
        "capability_boundary": {
            "property_only_scalar_solvation_prediction": True,
            "neutral_solute_and_solvents_only": True,
            "one_to_three_solvent_components": True,
            "official_public_entrypoint_documented_max_solvent_components": 2,
            "third_solvent_dynamic_slot_path": (
                "code-compatible but not explicitly documented by the official "
                "public prediction entrypoints"
            ),
            "potential_energy_surface": False,
            "forces": False,
            "geometry_optimization": False,
            "frequency": False,
            "free_energy_sampling_protocol": False,
        },
        "validation_scope": {
            "panel_role": "frozen_output_blind_development_panel_not_final_holdout",
            "row_count": len(scored_rows),
            "distinct_primary_functional_group_count": len(
                {row["primary_functional_group"] for row in scored_rows}
            ),
            "paper_dataset_aggregate_claimed_from_ten_rows": False,
            "paper_or_workbook_published_column_reproduction_separate_from_live_weight_execution": True,
            "historical_output_blind_selection_is_author_attestation_not_independent_proof": True,
            "matched_qm_speed_test_performed": False,
            "runtime_used_for_admission": False,
            "canonical_live_execution_evidence": canonical_live_execution,
            "imported_fixture_used": not canonical_live_execution,
        },
        "published_workbook_column_reproduction": {
            "description": "Recomputed from official workbook experimental and SolProp-mix_QM_Exp_GsolvT columns; not live checkpoint execution.",
            "paper": PAPER_URL,
            "panels": published,
        },
        "runtime_diagnostic": {
            "source": "separate frozen runtime-diagnostic.json",
            "matched_qm_baseline": False,
            "admission_evidence": False,
            "volatile_worker_stream_hashes_omitted_from_canonical_result": True,
        },
        "admission_gates": {
            "upstream_oracle_reproduced": False,
            "ten_group_accuracy_gate_passes": (
                canonical_live_execution and accuracy_passes
            ),
            "gpu_f64_roundoff_parity_observed": (
                canonical_live_execution and gpu["passes_roundoff_parity"]
            ),
            "gpu_strict_zero_loss_passes": (
                canonical_live_execution and gpu["passes_strict_zero_loss"]
            ),
            "accuracy_admission_eligible": False,
            "gpu_admission_eligible": False,
            "matched_qm_speed_eligible": False,
            "final_holdout_eligible": False,
            "passes_all": False,
        },
        "route4_decision": {
            "status": (
                "rejected_for_admission"
                if canonical_live_execution
                else "imported_fixture_not_admission_evidence"
            ),
            "reason": (
                "The v1.1 workbook is broadly closer to live released-weight "
                "execution, but the exact workbook oracle is not reproduced and "
                "the frozen ten-group accuracy gate still fails."
                if canonical_live_execution
                else "Imported trusted fixture values are diagnostic only and "
                "cannot establish live execution, scientific, GPU, or admission claims."
            ),
            "adapter_may_be_used_as_unadmitted_research_property_backend": (
                canonical_live_execution
            ),
        },
    }
    result[
        (
            "live_released_weight_execution"
            if canonical_live_execution
            else "imported_trusted_fixture_evaluation"
        )
    ] = execution_section
    return result


def build_audit(
    panel: Mapping[str, Any],
    references: Mapping[str, Any],
    raw_live_results: Mapping[str, Any],
    workbook_path: Path,
    *,
    repository_root: Path,
    raw_input_sha256: str,
) -> dict[str, Any]:
    """Evaluate an imported trusted fixture without granting live-execution claims."""
    return _build_audit(
        panel,
        references,
        raw_live_results,
        workbook_path,
        repository_root=repository_root,
        raw_input_sha256=raw_input_sha256,
        canonical_live_execution=False,
    )


def _build_execute_live_audit(
    panel: Mapping[str, Any],
    references: Mapping[str, Any],
    raw_live_results: Mapping[str, Any],
    workbook_path: Path,
    *,
    repository_root: Path,
    raw_input_sha256: str,
    capability: object,
) -> dict[str, Any]:
    if capability is not _EXECUTE_LIVE_CAPABILITY:
        raise ValueError(
            "Canonical live evidence requires the execute-live capability."
        )
    return _build_audit(
        panel,
        references,
        raw_live_results,
        workbook_path,
        repository_root=repository_root,
        raw_input_sha256=raw_input_sha256,
        canonical_live_execution=True,
    )


def write_rows_csv(
    audit: Mapping[str, Any],
    path: Path,
    *,
    section: str = "live_released_weight_execution",
) -> None:
    """Write the row-wise experimental/prediction/error view from canonical data."""
    rows = audit[section]["rows"]
    fieldnames = [
        "row_id",
        "primary_functional_group",
        "solute_name",
        "solvent_composition",
        "temperature_kelvin",
        "experimental_gsolv_kcal_mol",
        "published_workbook_qmexp_prediction_kcal_mol",
        "cpu_fp64_prediction_kcal_mol",
        "cpu_signed_error_kcal_mol",
        "cpu_abs_error_kcal_mol",
        "gpu_fp64_prediction_kcal_mol",
        "gpu_signed_error_kcal_mol",
        "gpu_abs_error_kcal_mol",
        *(f"model_{index}_gpu_minus_cpu_kcal_mol" for index in range(10)),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = {
                "row_id": row["row_id"],
                "primary_functional_group": row["primary_functional_group"],
                "solute_name": row["solute"]["name"],
                "solvent_composition": _canonical_bytes(
                    row["solvent_composition"]
                ).decode("utf-8"),
                "temperature_kelvin": row["temperature_kelvin"],
                "experimental_gsolv_kcal_mol": row["experimental_gsolv_kcal_mol"],
                "published_workbook_qmexp_prediction_kcal_mol": row[
                    "published_workbook_qmexp_prediction_kcal_mol"
                ],
                "cpu_fp64_prediction_kcal_mol": row["cpu_fp64_prediction_kcal_mol"],
                "cpu_signed_error_kcal_mol": row["cpu_signed_error_kcal_mol"],
                "cpu_abs_error_kcal_mol": row["cpu_abs_error_kcal_mol"],
                "gpu_fp64_prediction_kcal_mol": row["gpu_fp64_prediction_kcal_mol"],
                "gpu_signed_error_kcal_mol": row["gpu_signed_error_kcal_mol"],
                "gpu_abs_error_kcal_mol": row["gpu_abs_error_kcal_mol"],
            }
            for model in row["model_deltas"]:
                rendered[f'model_{model["model_index"]}_gpu_minus_cpu_kcal_mol'] = (
                    model["gpu_minus_cpu_kcal_mol"]
                )
            writer.writerow(rendered)


def execute_live(
    panel: Mapping[str, Any],
    *,
    source_root: Path,
    static_root: Path,
    weights_root: Path,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Execute only label-free panel inputs through the pinned adapter."""
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    from maple.function.solvfe.solpropmix_property import SolPropMixPropertyAdapter

    _validate_panel(
        panel, {"records": [{"row_id": row["row_id"]} for row in panel["records"]]}
    )
    adapter = SolPropMixPropertyAdapter(
        source_root,
        static_root,
        weights_root,
        python_executable=python_executable,
    )
    modes: dict[str, Any] = {}
    for device in ("cpu", "cuda:0"):
        rows = []
        for row in panel["records"]:
            result = adapter.predict(
                row["solute_inchi"],
                list(
                    zip(
                        row["solvent_inchis"],
                        row["solvent_mole_fractions"],
                        strict=True,
                    )
                ),
                temperature_kelvin=row["temperature_kelvin"],
                device=device,
                precision="float64_promoted",
            )
            rows.append(
                {
                    "models": [asdict(value) for value in result.model_predictions],
                    "prediction": result.predicted_solvation_free_energy_kcal_mol,
                    "runtime_receipt": asdict(result.runtime_receipt),
                }
            )
        modes[device] = {"rows": rows}
    joined = [
        {
            "row_id": row["row_id"],
            "primary_functional_group": row["primary_functional_group"],
            "cpu_prediction_kcal_mol": modes["cpu"]["rows"][index]["prediction"],
            "gpu_prediction_kcal_mol": modes["cuda:0"]["rows"][index]["prediction"],
        }
        for index, row in enumerate(panel["records"])
    ]
    # Normalize dataclass field names to the historical raw execution schema.
    for mode in modes.values():
        for row in mode["rows"]:
            row["models"] = [
                {
                    "model_index": item["model_index"],
                    "g298_kcal_mol": item["g298_kcal_mol"],
                    "h298_kcal_mol": item["h298_kcal_mol"],
                    "g_temperature_kcal_mol": item["g_temperature_kcal_mol"],
                }
                for item in row["models"]
            ]
    return {"rows": joined, "modes": modes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = (
        Path(__file__).resolve().parent
        / "benchmarks"
        / "solpropmix-qmexp-release-audit-2026-07-31"
    )
    parser.add_argument("--output-dir", type=Path, default=directory)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--selection-source", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live-results", type=Path)
    source.add_argument("--execute-live", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--python-executable", type=Path)
    args = parser.parse_args()
    canonical_directory = directory.resolve()
    if args.live_results and args.output_dir.resolve() == canonical_directory:
        parser.error(
            "--live-results is an imported trusted fixture and cannot overwrite "
            "the canonical benchmark directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "panel-inputs.json"
    references_path = args.output_dir / "experimental-references.json"
    if args.selection_source:
        selection = json.loads(args.selection_source.read_text(encoding="utf-8"))
        panel, references = split_selection(selection, args.workbook)
        panel_path.write_text(_render(panel), encoding="utf-8")
        references_path.write_text(_render(references), encoding="utf-8")
    else:
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        references = json.loads(references_path.read_text(encoding="utf-8"))
    if args.execute_live:
        required = (args.source_root, args.static_root, args.weights_root)
        if any(value is None for value in required):
            parser.error(
                "--execute-live requires --source-root, --static-root, and --weights-root"
            )
        raw_live = execute_live(
            panel,
            source_root=args.source_root,
            static_root=args.static_root,
            weights_root=args.weights_root,
            python_executable=args.python_executable,
        )
        raw_artifact_path = args.output_dir / RAW_EXECUTION_ARTIFACT_NAME
        raw_artifact_path.write_text(
            _render(_stable_execute_live_artifact(raw_live)),
            encoding="utf-8",
        )
        raw_input_sha256 = _sha256_file(raw_artifact_path)
    else:
        assert args.live_results is not None
        raw_live = json.loads(args.live_results.read_text(encoding="utf-8"))
        raw_input_sha256 = _sha256_file(args.live_results)
    root = Path(__file__).resolve().parents[2]
    if args.execute_live:
        audit = _build_execute_live_audit(
            panel,
            references,
            raw_live,
            args.workbook,
            repository_root=root,
            raw_input_sha256=raw_input_sha256,
            capability=_EXECUTE_LIVE_CAPABILITY,
        )
        result_name = "result.json"
        rows_name = "rows.csv"
        section = "live_released_weight_execution"
    else:
        audit = build_audit(
            panel,
            references,
            raw_live,
            args.workbook,
            repository_root=root,
            raw_input_sha256=raw_input_sha256,
        )
        result_name = IMPORTED_RESULT_NAME
        rows_name = IMPORTED_ROWS_NAME
        section = "imported_trusted_fixture_evaluation"
    (args.output_dir / result_name).write_text(_render(audit), encoding="utf-8")
    write_rows_csv(audit, args.output_dir / rows_name, section=section)
    print(_render(audit), end="")


if __name__ == "__main__":
    main()
