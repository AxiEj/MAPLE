from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/pretrained-solvation-hub/run_solpropmix_qmexp_release_audit.py"
DIRECTORY = (
    ROOT
    / "docs/pretrained-solvation-hub/benchmarks/solpropmix-qmexp-release-audit-2026-07-31"
)
PANEL = DIRECTORY / "panel-inputs.json"
REFERENCES = DIRECTORY / "experimental-references.json"
RESULT = DIRECTORY / "result.json"
RUNTIME = DIRECTORY / "runtime-diagnostic.json"
DEFAULT_WORKBOOK = Path(
    "/tmp/solpropmix-v11-audit/data-extracted-20260731/Data/solprop-mix_v1.1.xlsx"
)
WORKBOOK = Path(os.environ.get("MAPLE_SOLPROPMIX_QMEXP_WORKBOOK", DEFAULT_WORKBOOK))


def _load_module():
    specification = importlib.util.spec_from_file_location(
        "solpropmix_release_audit", SCRIPT
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AUDIT = _load_module()
RAW_EXECUTION = DIRECTORY / AUDIT.RAW_EXECUTION_ARTIFACT_NAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(values, *, stable_squares=True):
    values = tuple(values)
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


def _raw_live_from_frozen_result():
    panel = json.loads(PANEL.read_text())
    result = json.loads(RESULT.read_text())
    scored = result["live_released_weight_execution"]["rows"]
    joined = []
    modes = {"cpu": {"rows": []}, "cuda:0": {"rows": []}}
    for input_row, scored_row in zip(panel["records"], scored, strict=True):
        joined.append(
            {
                "row_id": input_row["row_id"],
                "primary_functional_group": input_row["primary_functional_group"],
                "cpu_prediction_kcal_mol": scored_row["cpu_fp64_prediction_kcal_mol"],
                "gpu_prediction_kcal_mol": scored_row["gpu_fp64_prediction_kcal_mol"],
            }
        )
        for mode, prefix in (("cpu", "cpu"), ("cuda:0", "gpu")):
            receipt = copy.deepcopy(result["identity"]["runtime_receipts"][mode])
            receipt["canonical_solute_smiles"] = input_row[
                "canonical_nonisomeric_smiles"
            ]
            receipt["solvent_components"] = [
                {"canonical_smiles": smiles, "mole_fraction": fraction}
                for smiles, fraction in sorted(
                    (
                        AUDIT._canonical_smiles(inchi, "test solvent"),
                        fraction,
                    )
                    for inchi, fraction in zip(
                        input_row["solvent_inchis"],
                        input_row["solvent_mole_fractions"],
                        strict=True,
                    )
                )
            ]
            modes[mode]["rows"].append(
                {
                    "prediction": scored_row[f"{prefix}_fp64_prediction_kcal_mol"],
                    "models": [
                        {
                            "model_index": model["model_index"],
                            "g_temperature_kcal_mol": model[
                                f"{prefix}_g_temperature_kcal_mol"
                            ],
                        }
                        for model in scored_row["model_deltas"]
                    ],
                    "runtime_receipt": receipt,
                }
            )
    return {"rows": joined, "modes": modes}, panel


def test_panel_is_exactly_ten_distinct_groups_and_label_free():
    panel = json.loads(PANEL.read_text())
    references = json.loads(REFERENCES.read_text())
    rows = panel["records"]
    groups = [row["primary_functional_group"] for row in rows]

    assert len(rows) >= 10
    assert len(rows) == len(set(groups)) == 10
    assert tuple(groups) == AUDIT.EXPECTED_GROUPS
    assert panel["model_worker_may_read_only_this_file"] is True
    assert (
        panel["selection_boundary"][
            "selection_author_attests_chosen_without_experimental_or_model_outputs"
        ]
        is True
    )
    assert (
        panel["selection_boundary"]["exact_release_mapping_sha256"]
        == AUDIT.RELEASE_MAPPING_SHA256
    )
    assert references["must_not_be_supplied_to_model_worker"] is True
    assert all(not AUDIT.LABEL_KEYS.intersection(row) for row in rows)
    assert [row["row_id"] for row in rows] == [
        row["row_id"] for row in references["records"]
    ]


def test_live_schema_rejects_cross_row_and_cross_field_corruption():
    raw, panel = _raw_live_from_frozen_result()
    normalized, _ = AUDIT._normalized_live_rows(raw, panel)
    assert len(normalized) == 10
    direct_execute_shape = copy.deepcopy(raw)
    for mode in ("cpu", "cuda:0"):
        for row in direct_execute_shape["modes"][mode]["rows"]:
            receipt = row["runtime_receipt"]
            receipt["solvent_components"] = tuple(receipt["solvent_components"])
            receipt["checkpoint_sha256"] = tuple(receipt["checkpoint_sha256"])
            receipt["static_sha256"] = tuple(
                tuple(item) for item in receipt["static_sha256"]
            )
    assert len(AUDIT._normalized_live_rows(direct_execute_shape, panel)[0]) == 10

    corruptions = []
    bad = copy.deepcopy(raw)
    bad["modes"]["cpu"]["rows"][1]["runtime_receipt"]["execution_dtype"] = "float32"
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["modes"]["cuda:0"]["rows"][1]["runtime_receipt"]["execution_device"] = "cpu"
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["modes"]["cpu"]["rows"][1]["runtime_receipt"]["checkpoint_sha256"][0] = "bad"
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["modes"]["cpu"]["rows"][1]["prediction"] = 12345.0
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["modes"]["cpu"]["rows"][1]["models"][0]["model_index"] = 999
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["rows"][1]["cpu_prediction_kcal_mol"] += 0.5
    corruptions.append(bad)
    bad = copy.deepcopy(raw)
    bad["modes"]["cpu"]["rows"][1]["runtime_receipt"]["solvent_components"][0][
        "canonical_smiles"
    ] = "C"
    corruptions.append(bad)

    for corrupted in corruptions:
        with pytest.raises((TypeError, ValueError)):
            AUDIT._normalized_live_rows(corrupted, panel)


def test_selection_and_release_mapping_are_frozen_with_proof_limitations():
    panel = json.loads(PANEL.read_text())
    references = json.loads(REFERENCES.read_text())
    result = json.loads(RESULT.read_text())
    selection_path = ROOT / AUDIT.SELECTION_PROVENANCE_RELATIVE_PATH
    mapping_path = ROOT / AUDIT.RELEASE_MAPPING_RELATIVE_PATH
    selection = json.loads(selection_path.read_text())

    assert _sha256(selection_path) == AUDIT.SELECTION_PROVENANCE_SHA256
    assert _sha256(mapping_path) == AUDIT.RELEASE_MAPPING_SHA256
    assert (
        selection["selection_author_attestation"][
            "independently_provable_from_frozen_artifacts"
        ]
        is False
    )
    assert all(not AUDIT.LABEL_KEYS.intersection(row) for row in selection["records"])
    mapping = AUDIT.audit_release_mapping(
        ROOT,
        panel,
        references,
        result["live_released_weight_execution"]["rows"],
    )
    assert mapping["v1_1_rows_closer_to_live_count"] == 9
    assert mapping["v1_1_broadly_closer_than_v1_0"] is True


def test_v11_header_name_provenance_tracks_sigma_column_insertions():
    references = json.loads(REFERENCES.read_text())["records"]
    binary = references[:-1]
    ternary = references[-1]
    assert all(row["experimental"]["value_cell"].startswith("J") for row in binary)
    assert all(
        row["published_qmexp"]["h298_value_cell"].startswith("U") for row in binary
    )
    assert all(
        row["published_qmexp"]["g298_value_cell"].startswith("V") for row in binary
    )
    assert all(
        row["published_qmexp"]["temperature_value_cell"].startswith("W")
        for row in binary
    )
    assert ternary["experimental"]["value_cell"].startswith("M")
    assert ternary["published_qmexp"]["h298_value_cell"].startswith("P")
    assert ternary["published_qmexp"]["g298_value_cell"].startswith("Q")
    assert ternary["published_qmexp"]["temperature_value_cell"].startswith("R")


def test_frozen_identity_hashes_and_result_input_hashes_are_recomputed():
    panel = json.loads(PANEL.read_text())
    references = json.loads(REFERENCES.read_text())
    result = json.loads(RESULT.read_text())
    identity = result["identity"]

    assert (
        identity["panel_inputs_sha256"]
        == hashlib.sha256(AUDIT._canonical_bytes(panel)).hexdigest()
    )
    assert (
        identity["experimental_references_sha256"]
        == hashlib.sha256(AUDIT._canonical_bytes(references)).hexdigest()
    )
    assert identity["taxonomy"]["sha256"] == _sha256(
        ROOT / AUDIT.TAXONOMY_RELATIVE_PATH
    )
    assert identity["execution_code"]["adapter_sha256"] == _sha256(
        ROOT / AUDIT.ADAPTER_RELATIVE_PATH
    )
    assert identity["execution_code"]["isolated_model_worker_sha256"] == _sha256(
        ROOT / identity["execution_code"]["isolated_model_worker_path"]
    )
    assert identity["execution_code"]["audit_runner_sha256"] == _sha256(
        ROOT / AUDIT.AUDIT_RUNNER_RELATIVE_PATH
    )
    assert identity["release_mapping"] == {
        "path": AUDIT.RELEASE_MAPPING_RELATIVE_PATH,
        "sha256": AUDIT.RELEASE_MAPPING_SHA256,
    }
    assert identity["selection_provenance"] == {
        "path": AUDIT.SELECTION_PROVENANCE_RELATIVE_PATH,
        "sha256": AUDIT.SELECTION_PROVENANCE_SHA256,
    }
    assert identity["official_archive"] == {
        "zenodo_record": AUDIT.ZENODO_RECORD,
        "filename": "Data.zip",
        "size_bytes": AUDIT.ARCHIVE_SIZE_BYTES,
        "md5": AUDIT.ARCHIVE_MD5,
        "sha256": AUDIT.ARCHIVE_SHA256,
    }
    assert identity["workbook"] == {
        "member": AUDIT.WORKBOOK_MEMBER,
        "size_bytes": AUDIT.WORKBOOK_SIZE_BYTES,
        "md5": AUDIT.WORKBOOK_MD5,
        "sha256": AUDIT.WORKBOOK_SHA256,
    }
    assert identity["source"]["revision"] == AUDIT.SOURCE_REVISION
    assert identity["source"]["tree"] == AUDIT.SOURCE_TREE
    assert (
        tuple(row["sha256"] for row in identity["checkpoint_sha256"])
        == AUDIT.CHECKPOINT_SHA256
    )
    assert result["evidence_provenance"] == {
        "origin": "execute_live",
        "raw_input_sha256": _sha256(RAW_EXECUTION),
        "raw_input_normalization": (
            "execute_live_json_with_volatile_worker_stream_receipt_keys_omitted"
        ),
        "canonical_live_execution_evidence": True,
        "trusted_fixture_not_independent_execution": False,
    }
    raw_execution = json.loads(RAW_EXECUTION.read_text(encoding="utf-8"))
    for payload in raw_execution["modes"].values():
        for row in payload["rows"]:
            assert AUDIT.VOLATILE_RUNTIME_RECEIPT_KEYS.isdisjoint(
                row["runtime_receipt"]
            )
    assert result["live_released_weight_execution"]["evidence_origin"] == "execute_live"
    assert result["live_released_weight_execution"]["live_execution_claimed"] is True


def test_all_live_row_and_aggregate_metrics_recompute_independently():
    result = json.loads(RESULT.read_text())
    execution = result["live_released_weight_execution"]
    cpu_errors = []
    gpu_errors = []
    release_deltas = []
    model_deltas = []
    for row in execution["rows"]:
        experimental = row["experimental_gsolv_kcal_mol"]
        cpu_error = row["cpu_fp64_prediction_kcal_mol"] - experimental
        gpu_error = row["gpu_fp64_prediction_kcal_mol"] - experimental
        assert row["cpu_signed_error_kcal_mol"] == pytest.approx(cpu_error, abs=1e-15)
        assert row["gpu_signed_error_kcal_mol"] == pytest.approx(gpu_error, abs=1e-15)
        assert row["cpu_abs_error_kcal_mol"] == pytest.approx(abs(cpu_error), abs=1e-15)
        assert row["gpu_abs_error_kcal_mol"] == pytest.approx(abs(gpu_error), abs=1e-15)
        cpu_errors.append(cpu_error)
        gpu_errors.append(gpu_error)
        release_deltas.append(abs(row["live_cpu_minus_published_workbook_kcal_mol"]))
        assert len(row["model_deltas"]) == 10
        for model in row["model_deltas"]:
            delta = (
                model["gpu_g_temperature_kcal_mol"]
                - model["cpu_g_temperature_kcal_mol"]
            )
            assert model["gpu_minus_cpu_kcal_mol"] == delta
            assert model["abs_gpu_minus_cpu_kcal_mol"] == abs(delta)
            model_deltas.append(abs(delta))

    assert execution["cpu_metrics"] == _metrics(cpu_errors, stable_squares=False)
    assert execution["gpu_metrics"] == _metrics(gpu_errors, stable_squares=False)
    assert execution["cpu_metrics"]["mae_kcal_mol"] == 0.5098988620625835
    assert execution["cpu_metrics"]["rmse_kcal_mol"] == 0.7100644511098742
    assert execution["cpu_metrics"]["maxae_kcal_mol"] == 1.6089863795855495
    oracle = execution["release_vs_workbook"]
    assert oracle["mean_abs_delta_kcal_mol"] == math.fsum(release_deltas) / 10
    assert oracle["max_abs_delta_kcal_mol"] == max(release_deltas) == 0.2105751123873958
    assert oracle["mean_abs_delta_kcal_mol"] == 0.023138089950940088
    assert oracle["v1_0_mean_abs_live_delta_kcal_mol"] == 0.20559967169078072
    assert oracle["v1_0_max_abs_live_delta_kcal_mol"] == 0.5388906851377113
    assert oracle["v1_1_rows_closer_to_live_count"] == 9
    assert oracle["v1_1_broadly_closer_than_v1_0_on_frozen_panel"] is True
    assert oracle["upstream_oracle_reproduced"] is False
    assert execution["gpu_no_loss_policy"][
        "max_abs_per_model_gpu_minus_cpu_kcal_mol"
    ] == max(model_deltas)
    assert max(model_deltas) == 2.6645352591003757e-15


def test_rows_csv_is_regenerated_from_canonical_result(tmp_path):
    result = json.loads(RESULT.read_text())
    regenerated = tmp_path / "rows.csv"
    AUDIT.write_rows_csv(result, regenerated)
    assert regenerated.read_bytes() == (DIRECTORY / "rows.csv").read_bytes()

    with regenerated.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["primary_functional_group"] for row in rows} == set(
        AUDIT.EXPECTED_GROUPS
    )


def test_any_single_row_or_model_deterioration_blocks_gpu_policy():
    result = json.loads(RESULT.read_text())
    rows = copy.deepcopy(result["live_released_weight_execution"]["rows"])
    policy = AUDIT._gpu_policy(rows)
    assert policy["passes_roundoff_parity"] is True
    assert policy["passes_strict_zero_loss"] is False
    assert len(policy["exact_row_error_deteriorations"]) == 3
    assert len(policy["prediction_bitwise_mismatches"]) == 4
    assert len(policy["model_bitwise_mismatches"]) == 55

    rows[0]["gpu_abs_error_kcal_mol"] += 1e-5
    assert AUDIT._gpu_policy(rows)["passes_roundoff_parity"] is False
    rows = copy.deepcopy(result["live_released_weight_execution"]["rows"])
    rows[0]["model_deltas"][0]["abs_gpu_minus_cpu_kcal_mol"] = 1e-5
    assert AUDIT._gpu_policy(rows)["passes_roundoff_parity"] is False


def test_negative_admissions_and_runtime_boundary_are_frozen():
    result = json.loads(RESULT.read_text())
    diagnostic = json.loads(RUNTIME.read_text())
    gates = result["admission_gates"]
    boundary = result["capability_boundary"]

    assert result["acceptance_eligible"] is False
    assert gates["upstream_oracle_reproduced"] is False
    assert gates["ten_group_accuracy_gate_passes"] is False
    assert gates["gpu_f64_roundoff_parity_observed"] is True
    assert gates["gpu_strict_zero_loss_passes"] is False
    assert gates["accuracy_admission_eligible"] is False
    assert gates["gpu_admission_eligible"] is False
    assert gates["matched_qm_speed_eligible"] is False
    assert gates["final_holdout_eligible"] is False
    assert gates["passes_all"] is False
    assert (
        result["validation_scope"]["paper_dataset_aggregate_claimed_from_ten_rows"]
        is False
    )
    assert result["validation_scope"]["matched_qm_speed_test_performed"] is False
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["matched_qm_baseline"] is False
    assert diagnostic["admission_evidence"] is False
    assert diagnostic["canonical_result_omits_volatile_worker_stream_hashes"] is True
    assert [row["row_id"] for row in diagnostic["row_runtime_seconds"]] == [
        row["row_id"] for row in json.loads(PANEL.read_text())["records"]
    ]
    assert boundary["official_public_entrypoint_documented_max_solvent_components"] == 2
    assert "not explicitly documented" in boundary["third_solvent_dynamic_slot_path"]
    assert (
        result["runtime_diagnostic"][
            "volatile_worker_stream_hashes_omitted_from_canonical_result"
        ]
        is True
    )
    for receipt in result["identity"]["runtime_receipts"].values():
        assert AUDIT.VOLATILE_RUNTIME_RECEIPT_KEYS.isdisjoint(receipt)
    assert result["route4_decision"]["status"] == "rejected_for_admission"


def test_coordinated_imported_fixture_cannot_masquerade_as_live_execution(
    tmp_path, monkeypatch, capsys
):
    if not WORKBOOK.is_file():
        pytest.skip(f"SolProp-mix v1.1 workbook not available: {WORKBOOK}")
    raw, _ = _raw_live_from_frozen_result()
    for row_index, joined in enumerate(raw["rows"]):
        for mode, joined_key in (
            ("cpu", "cpu_prediction_kcal_mol"),
            ("cuda:0", "gpu_prediction_kcal_mol"),
        ):
            mode_row = raw["modes"][mode]["rows"][row_index]
            for model_index, model in enumerate(mode_row["models"]):
                if mode == "cpu":
                    model["g_temperature_kcal_mol"] += 100.0
                else:
                    model["g_temperature_kcal_mol"] = raw["modes"]["cpu"]["rows"][
                        row_index
                    ]["models"][model_index]["g_temperature_kcal_mol"]
            forged_mean = (
                math.fsum(
                    model["g_temperature_kcal_mol"] for model in mode_row["models"]
                )
                / 10
            )
            mode_row["prediction"] = forged_mean
            joined[joined_key] = forged_mean

    imported_path = tmp_path / "coordinated-forgery.json"
    imported_path.write_text(AUDIT._render(raw), encoding="utf-8")
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    references = json.loads(REFERENCES.read_text(encoding="utf-8"))
    direct_result = AUDIT.build_audit(
        panel,
        references,
        raw,
        WORKBOOK,
        repository_root=ROOT,
        raw_input_sha256=_sha256(imported_path),
    )
    assert "live_released_weight_execution" not in direct_result
    assert (
        direct_result["evidence_provenance"]["canonical_live_execution_evidence"]
        is False
    )
    with pytest.raises(TypeError):
        AUDIT.build_audit(
            panel,
            references,
            raw,
            WORKBOOK,
            repository_root=ROOT,
            raw_input_sha256=_sha256(imported_path),
            evidence_origin="execute_live",
        )

    output = tmp_path / "imported-output"
    output.mkdir()
    shutil.copy2(PANEL, output / PANEL.name)
    shutil.copy2(REFERENCES, output / REFERENCES.name)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output-dir",
            str(output),
            "--workbook",
            str(WORKBOOK),
            "--live-results",
            str(imported_path),
        ],
    )
    AUDIT.main()
    capsys.readouterr()

    assert not (output / "result.json").exists()
    assert not (output / "rows.csv").exists()
    imported_result = json.loads(
        (output / AUDIT.IMPORTED_RESULT_NAME).read_text(encoding="utf-8")
    )
    assert "live_released_weight_execution" not in imported_result
    fixture = imported_result["imported_trusted_fixture_evaluation"]
    assert fixture["evidence_origin"] == "imported_trusted_fixture"
    assert fixture["live_execution_claimed"] is False
    assert fixture["scientific_or_admission_claim_eligible"] is False
    assert fixture["gpu_no_loss_policy"]["passes_strict_zero_loss"] is False
    assert fixture["gpu_no_loss_policy"]["bitwise_equal_claimed"] is False
    assert imported_result["evidence_provenance"] == {
        "origin": "imported_trusted_fixture",
        "raw_input_sha256": _sha256(imported_path),
        "raw_input_normalization": "none_exact_imported_file_bytes",
        "canonical_live_execution_evidence": False,
        "trusted_fixture_not_independent_execution": True,
    }
    assert all(value is False for value in imported_result["admission_gates"].values())
    assert (
        imported_result["route4_decision"]["status"]
        == "imported_fixture_not_admission_evidence"
    )


def test_imported_fixture_cannot_overwrite_canonical_benchmark(monkeypatch, capsys):
    raw, _ = _raw_live_from_frozen_result()
    imported_path = DIRECTORY / ".test-imported-fixture.json"
    imported_path.write_text(AUDIT._render(raw), encoding="utf-8")
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--output-dir",
                str(DIRECTORY),
                "--workbook",
                str(WORKBOOK),
                "--live-results",
                str(imported_path),
            ],
        )
        with pytest.raises(SystemExit):
            AUDIT.main()
        assert (
            "cannot overwrite the canonical benchmark directory"
            in capsys.readouterr().err
        )
    finally:
        imported_path.unlink(missing_ok=True)


def _independent_workbook_metrics(path: Path):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    row_tag = f"{{{ns}}}row"
    cell_tag = f"{{{ns}}}c"
    value_tag = f"{{{ns}}}v"
    text_tag = f"{{{ns}}}t"
    errors = {}
    with zipfile.ZipFile(path) as archive:
        shared = ()
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = tuple(
                "".join(node.text or "" for node in item.iter(text_tag))
                for item in root
            )
        for sheet, member in AUDIT.SHEET_PATHS.items():
            current = []
            experimental_column = prediction_column = None
            with archive.open(member) as handle:
                for _, row in ElementTree.iterparse(handle, events=("end",)):
                    if row.tag != row_tag:
                        continue
                    number = int(row.attrib["r"])
                    values = {}
                    for cell in row.findall(cell_tag):
                        reference = cell.attrib["r"]
                        if cell.attrib.get("t") == "inlineStr":
                            value = "".join(
                                node.text or "" for node in cell.iter(text_tag)
                            )
                        else:
                            node = cell.find(value_tag)
                            if node is None:
                                value = None
                            elif cell.attrib.get("t") == "s":
                                value = shared[int(node.text)]
                            else:
                                value = float(node.text)
                        values[reference] = value
                    if number == 1:
                        headers = {
                            value: reference.rstrip("0123456789")
                            for reference, value in values.items()
                        }
                        experimental_column = headers["Gsolv (kcal/mol)"]
                        prediction_column = headers["SolProp-mix_QM_Exp_GsolvT"]
                    else:
                        observed = values.get(f"{experimental_column}{number}")
                        predicted = values.get(f"{prediction_column}{number}")
                        if observed is not None and predicted is not None:
                            current.append(predicted - observed)
                    row.clear()
            errors[sheet] = current
    return {
        name: _metrics(value for sheet in sheets for value in errors[sheet])
        for name, sheets in AUDIT.WORKBOOK_PANELS.items()
    }


@pytest.mark.skipif(
    not WORKBOOK.is_file(), reason="Pinned official SolProp-mix workbook unavailable"
)
def test_workbook_provenance_and_published_columns_round_trip_independently():
    assert _sha256(WORKBOOK) == AUDIT.WORKBOOK_SHA256
    result = json.loads(RESULT.read_text())
    observed = _independent_workbook_metrics(WORKBOOK)
    published = result["published_workbook_column_reproduction"]["panels"]
    for name, metrics in observed.items():
        for key, value in metrics.items():
            assert published[name][key] == value
    assert observed["binary_nonaqueous"] == {
        "count": 22564,
        "mae_kcal_mol": 0.24954483260577912,
        "rmse_kcal_mol": 0.36882203382673406,
        "maxae_kcal_mol": 3.7406858271985826,
    }
    assert observed["binary_all"]["count"] == 30030
    assert observed["binary_all"]["mae_kcal_mol"] == 0.2899802382859586
    assert observed["binary_all"]["rmse_kcal_mol"] == 0.4463043000208012
    assert observed["ternary_all"]["count"] == 4242
    assert observed["ternary_all"]["mae_kcal_mol"] == 0.2350860982612734
    assert observed["ternary_all"]["rmse_kcal_mol"] == 0.3438563818309633
    AUDIT.audit_workbook(WORKBOOK, json.loads(REFERENCES.read_text()))
