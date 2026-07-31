from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/pretrained-solvation-hub/run_solpropmix_qmexp_broad_audit.py"
DIRECTORY = (
    ROOT
    / "docs/pretrained-solvation-hub/benchmarks/solpropmix-qmexp-broad-audit-2026-07-31"
)
PANEL = DIRECTORY / "panel-inputs.json"
REFERENCES = DIRECTORY / "experimental-references.json"
ROWS = DIRECTORY / "rows.csv"
TERNARY_ROWS = DIRECTORY / "ternary-rows.csv"
CROSSCHECK = DIRECTORY / "adapter-crosscheck.json"
RESULT = DIRECTORY / "result.json"
EXPECTED_CHECKPOINT_SHA256 = [
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
]
# Test-owned identities: coordinated artifact/result edits cannot redefine these.
EXPECTED_FILE_SHA256 = {
    "adapter-crosscheck.json": "344657e8da67945a5c2fde3e6b7b877401a64631317d1e69809ea813327d1386",
    "experimental-references.json": "96ae81e15db0ed5cdbafdc85f6d7d2ba53cebde8ab2cf13d34f1f82bbafe6906",
    "panel-inputs.json": "6e593c52eadcc6f1d3793161e80a30e31fe5bb83e21dd0fa1379db2bbb43bd24",
    "result.json": "02f7dc64b6f4379c68dbb4999225b1400aefd50ab89df3a405ba864061a9f01c",
    "rows.csv": "626f4c7f1372b5036360654a7efb9b158eed00fda2809efc823f5d1947d82084",
    "ternary-rows.csv": "87535555c65fead1f32e15e5a0f09e6eabacc5280754ef0bd73166f6c89dad8a",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metrics(errors: list[float]) -> dict[str, float | int]:
    return {
        "n": len(errors),
        "mae_kcal_mol": math.fsum(abs(value) for value in errors) / len(errors),
        "rmse_kcal_mol": math.sqrt(
            math.fsum(value * value for value in errors) / len(errors)
        ),
        "maxae_kcal_mol": max(abs(value) for value in errors),
    }


def _assert_metrics(actual, expected) -> None:
    assert actual["n"] == expected["n"]
    for key in ("mae_kcal_mol", "rmse_kcal_mol", "maxae_kcal_mol"):
        assert math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=2e-15)


def _assert_fixed_bundle_hashes(directory: Path) -> None:
    assert EXPECTED_FILE_SHA256
    for name, expected in EXPECTED_FILE_SHA256.items():
        assert _sha256(directory / name) == expected


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("solpropmix_broad_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_hashes_and_code_identities_are_exact() -> None:
    result = json.loads(RESULT.read_text())
    _assert_fixed_bundle_hashes(DIRECTORY)
    for name, expected in result["artifact_sha256"].items():
        assert _sha256(DIRECTORY / name) == expected
    identity = result["scientific_identity"]
    assert identity["source_repository"] == (
        "https://gitlab.kuleuven.be/creas/vermeiregroup/solprop"
    )
    assert identity["source_revision"] == ("80043ce09eb8802517c35b59254f8e9c181f2dac")
    assert identity["source_tree"] == ("bc7b933d6cf4c55c8ad10236383cfd09afbbfdc2")
    assert identity["data_release_version"] == "v1.1"
    assert identity["data_release_record"] == 15587866
    assert _sha256(RUNNER) == identity["runner_sha256"]
    assert (
        _sha256(ROOT / identity["adapter_relative_path"]) == identity["adapter_sha256"]
    )
    assert _sha256(ROOT / identity["worker_relative_path"]) == identity["worker_sha256"]
    assert identity["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert len(set(identity["checkpoint_sha256"])) == 10
    assert len(identity["static_module_sha256"]) == 31
    assert identity["static_module_sha256"]["solvation_predictor/inp.py"] == (
        "e46b809a992a382794858d7cdc6406cff017ac4b37fd5fda92b27055d5eca167"
    )
    assert (
        identity["static_module_sha256"]["solvation_predictor/train/train.py"]
        == "556fd9b8ed5dcd531d2e58b845ecc95fb43357faf31c28c7863a7a396cd8cab6"
    )
    assert identity["workbook_sha256"] == (
        "793f325e6edddca3334b368f68edfc368027e472ad779e7e5d52ff5d1e1a001d"
    )


def test_main_panel_is_label_separated_and_has_exact_row_provenance() -> None:
    panel = json.loads(PANEL.read_text())
    references = json.loads(REFERENCES.read_text())
    rows = _read_rows(ROWS)
    inputs = panel["records"]
    labels = references["records"]
    assert panel["model_worker_may_read_only_this_file"] is True
    assert panel["selection_labels_read_after_indices_frozen"] is True
    assert references["must_not_be_supplied_to_model_worker"] is True
    assert len(inputs) == len(labels) == len(rows) == 1000
    assert len({row["primary_functional_group"] for row in inputs}) == 15
    assert [row["row_id"] for row in inputs] == [row["row_id"] for row in labels]
    assert [row["row_id"] for row in inputs] == [row["row_id"] for row in rows]
    forbidden = {
        "experimental_gsolv_kcal_mol",
        "cpu_prediction_kcal_mol",
        "gpu_prediction_kcal_mol",
        "cpu_abs_error_kcal_mol",
        "gpu_abs_error_kcal_mol",
    }
    assert all(not forbidden.intersection(row) for row in inputs)
    assert all("experimental_gsolv" not in row["input_cells"] for row in inputs)
    for input_row, label_row, row in zip(inputs, labels, rows, strict=True):
        assert row["row_id"] == f'{row["sheet"]}!{row["excel_row_number"]}'
        assert row["row_id"] == input_row["row_id"] == label_row["row_id"]
        for field in (
            "sheet",
            "primary_functional_group",
            "solute_name",
            "solute_inchi",
            "canonical_solute_smiles",
        ):
            assert row[field] == str(input_row[field])
        assert json.loads(row["solvent_pairs"]) == input_row["solvent_pairs"]
        assert json.loads(row["solvent_names"]) == input_row["solvent_names"]
        assert (
            json.loads(row["canonical_solvent_components"])
            == input_row["canonical_solvent_components"]
        )
        assert (
            float(row["experimental_gsolv_kcal_mol"])
            == label_row["experimental_gsolv_kcal_mol"]
        )
        for field in (
            "published_workbook_qmexp_h298_kcal_mol",
            "published_workbook_qmexp_g298_kcal_mol",
            "published_workbook_qmexp_gt_kcal_mol",
        ):
            assert float(row[field]) == label_row[field]
        cells = json.loads(row["input_cells"])
        assert {
            key: value for key, value in cells.items() if key != "experimental_gsolv"
        } == input_row["input_cells"]
        assert cells["experimental_gsolv"] == label_row["experimental_value_cell"]
        assert cells["solute_inchi"].endswith(row["excel_row_number"])
        assert cells["temperature_kelvin"].endswith(row["excel_row_number"])


def test_all_main_row_errors_and_aggregate_metrics_recompute() -> None:
    result = json.loads(RESULT.read_text())
    rows = _read_rows(ROWS)
    cpu_errors: list[float] = []
    gpu_errors: list[float] = []
    cpu_by_group: dict[str, list[float]] = defaultdict(list)
    gpu_by_group: dict[str, list[float]] = defaultdict(list)
    exact_worsening = 0
    exact_worsening_ids: list[str] = []
    over_tolerance = 0
    over_tolerance_ids: list[str] = []
    bitwise_mismatches = 0
    for row in rows:
        experiment = float(row["experimental_gsolv_kcal_mol"])
        cpu_prediction = float(row["cpu_prediction_kcal_mol"])
        gpu_prediction = float(row["gpu_prediction_kcal_mol"])
        cpu_error = cpu_prediction - experiment
        gpu_error = gpu_prediction - experiment
        assert cpu_error == float(row["cpu_signed_error_kcal_mol"])
        assert gpu_error == float(row["gpu_signed_error_kcal_mol"])
        assert abs(cpu_error) == float(row["cpu_abs_error_kcal_mol"])
        assert abs(gpu_error) == float(row["gpu_abs_error_kcal_mol"])
        assert gpu_prediction - cpu_prediction == float(
            row["gpu_minus_cpu_prediction_kcal_mol"]
        )
        cpu_errors.append(cpu_error)
        gpu_errors.append(gpu_error)
        cpu_by_group[row["primary_functional_group"]].append(cpu_error)
        gpu_by_group[row["primary_functional_group"]].append(gpu_error)
        if abs(gpu_error) > abs(cpu_error):
            exact_worsening += 1
            exact_worsening_ids.append(row["row_id"])
        if abs(gpu_error) > abs(cpu_error) + 5e-12:
            over_tolerance += 1
            over_tolerance_ids.append(row["row_id"])
        bitwise_mismatches += gpu_prediction != cpu_prediction

    lane = result["main_supported_lane"]
    _assert_metrics(_metrics(cpu_errors), lane["overall_cpu"])
    _assert_metrics(_metrics(gpu_errors), lane["overall_gpu"])
    assert lane["overall_cpu"] == {
        "n": 1000,
        "mae_kcal_mol": 0.2600320951491671,
        "rmse_kcal_mol": 0.3768962032607486,
        "maxae_kcal_mol": 1.8977049801370343,
    }
    assert len(cpu_by_group) == len(gpu_by_group) == 15
    for group, errors in cpu_by_group.items():
        _assert_metrics(
            _metrics(errors), lane["by_primary_functional_group"][group]["cpu"]
        )
        _assert_metrics(
            _metrics(gpu_by_group[group]),
            lane["by_primary_functional_group"][group]["gpu"],
        )
    gpu = lane["gpu_vs_cpu"]
    assert exact_worsening == gpu["gpu_worsens_abs_error_exact_count"] == 218
    assert over_tolerance == gpu["gpu_worsens_abs_error_over_5e12_count"] == 0
    assert exact_worsening_ids == gpu["gpu_worsen_exact_row_ids"]
    assert over_tolerance_ids == gpu["gpu_worsen_over_5e12_row_ids"] == []
    assert (
        bitwise_mismatches == gpu["ensemble_prediction_bitwise_mismatch_count"] == 496
    )


def test_strict_gpu_and_accuracy_gates_reject_admission_and_speed() -> None:
    result = json.loads(RESULT.read_text())
    assert result["artifact_construction"] == {
        "mode": "exact_canonical_reconstruction_from_fixed_scored_inputs",
        "not_a_new_live_run": True,
        "fixed_frozen_input_sha256": {
            key: value
            for key, value in EXPECTED_FILE_SHA256.items()
            if key != "result.json"
        },
    }
    policy = result["strict_float64_policy"]
    assert policy == {
        "execution_dtype": "float64",
        "deterministic_algorithms": True,
        "tf32": False,
        "amp": False,
        "fp16": False,
        "bf16": False,
        "diagnostic_tolerance_kcal_mol": 5e-12,
        "zero_loss_requires_exact_no_error_worsening": True,
    }
    lane = result["main_supported_lane"]
    assert lane["gpu_vs_cpu"]["strict_zero_loss"] is False
    assert lane["gpu_vs_cpu"]["gpu_admission"] is False
    assert lane["accuracy_gate"]["mae_pass"] is False
    assert lane["accuracy_gate"]["rmse_pass"] is False
    assert lane["accuracy_gate"]["overall_pass"] is False
    assert lane["matched_qm_speed_eligible"] is False
    assert "wall-clock" in lane["speed_boundary"]
    assert "development/non-blind" in lane["training_holdout_boundary"]


def test_adapter_crosscheck_covers_three_pure_three_binary_on_both_devices() -> None:
    result = json.loads(RESULT.read_text())
    checks = json.loads(CROSSCHECK.read_text())
    assert checks == result["adapter_crosscheck"]["rows"]
    assert len(checks) == 12
    assert {row["device"] for row in checks} == {"cpu", "cuda:0"}
    base_rows = {row["row_id"] for row in checks}
    assert sum(row.startswith("From IDAC") for row in base_rows) == 3
    assert sum(row.startswith("From VLE - Bin") for row in base_rows) == 3
    assert all(row["prediction_abs_delta_kcal_mol"] <= 5e-12 for row in checks)
    assert all(
        row["max_model_prediction_abs_delta_kcal_mol"] <= 5e-12 for row in checks
    )
    for row in checks:
        model_deltas = row["model_prediction_abs_deltas_kcal_mol"]
        assert [item["model_index"] for item in model_deltas] == list(range(10))
        assert all(
            item["prediction_abs_delta_kcal_mol"] <= 5e-12 for item in model_deltas
        )
    assert result["adapter_crosscheck"]["all_per_model_and_ensemble_pass"] is True


def test_ternary_is_separate_100_row_undocumented_diagnostic() -> None:
    result = json.loads(RESULT.read_text())
    rows = _read_rows(TERNARY_ROWS)
    lane = result["ternary_undocumented_dynamic_slot_diagnostic"]
    assert len(rows) == 100
    assert lane["not_part_of_main_lane_or_admission"] is True
    cpu_errors = [float(row["cpu_signed_error_kcal_mol"]) for row in rows]
    gpu_errors = [float(row["gpu_signed_error_kcal_mol"]) for row in rows]
    _assert_metrics(_metrics(cpu_errors), lane["overall_cpu"])
    _assert_metrics(_metrics(gpu_errors), lane["overall_gpu"])
    assert lane["overall_cpu"] == {
        "n": 100,
        "mae_kcal_mol": 0.18568599085530849,
        "rmse_kcal_mol": 0.2630909605347411,
        "maxae_kcal_mol": 1.426683398055653,
    }
    live_deltas = [
        abs(float(row["live_cpu_minus_workbook_qmexp_kcal_mol"])) for row in rows
    ]
    assert math.isclose(
        math.fsum(live_deltas) / len(live_deltas),
        lane["mean_abs_live_cpu_vs_workbook_qmexp_gt_kcal_mol"],
        rel_tol=0.0,
        abs_tol=2e-15,
    )
    assert lane["mean_abs_live_cpu_vs_workbook_qmexp_gt_kcal_mol"] == (
        0.06220292400566668
    )
    assert lane["max_abs_live_cpu_vs_workbook_qmexp_gt_kcal_mol"] == (
        0.38707916302832324
    )
    exact_ids = [
        row["row_id"]
        for row in rows
        if abs(float(row["gpu_signed_error_kcal_mol"]))
        > abs(float(row["cpu_signed_error_kcal_mol"]))
    ]
    tolerance_ids = [
        row["row_id"]
        for row in rows
        if abs(float(row["gpu_signed_error_kcal_mol"]))
        > abs(float(row["cpu_signed_error_kcal_mol"])) + 5e-12
    ]
    gpu = lane["gpu_vs_cpu"]
    assert exact_ids == gpu["gpu_worsen_exact_row_ids"]
    assert len(exact_ids) == gpu["gpu_worsens_abs_error_exact_count"] == 14
    assert tolerance_ids == gpu["gpu_worsen_over_5e12_row_ids"] == []
    assert gpu["gpu_worsens_abs_error_over_5e12_count"] == 0
    assert (
        sum(
            row["cpu_prediction_kcal_mol"] != row["gpu_prediction_kcal_mol"]
            for row in rows
        )
        == gpu["ensemble_prediction_bitwise_mismatch_count"]
        == 49
    )


def test_runner_has_no_frozen_host_paths_or_volatile_canonical_fields() -> None:
    source = RUNNER.read_text()
    result = json.loads(RESULT.read_text())
    assert "/home/axie/" not in source
    assert "/tmp/" not in source
    assert "runtime_seconds" not in json.dumps(result)
    assert "execution_locations" not in result
    assert any(
        "stream hashes" in exclusion
        for exclusion in result["canonical_result_exclusions"]
    )
    assert "--workbook" in source
    assert "--static-root" in source
    assert "--weights-root" in source
    main_source = source[source.index("def main() -> None:") :]
    assert (
        main_source.index("authenticate_official_assets(")
        < main_source.index("snapshot_official_data(")
        < main_source.index("configure_runtime(STATIC_ROOT)")
    )
    assert 'TemporaryDirectory(prefix="solpropmix-static-")' in source
    assert "sys.path.insert(0, str(clean_root))" in source
    assert "sys.path.insert(0, str(static_root))" not in source
    assert "python_executable.resolve(strict=True)" in source


@pytest.mark.parametrize(
    "target",
    [
        "panel-inputs.json",
        "experimental-references.json",
        "rows.csv",
        "adapter-crosscheck.json",
    ],
)
def test_fixed_hashes_reject_four_independent_artifact_tampers(
    tmp_path: Path, target: str
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(DIRECTORY, copied)
    path = copied / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AssertionError):
        _assert_fixed_bundle_hashes(copied)


def test_fixed_hashes_reject_synchronized_rows_and_result_hash_tamper(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(DIRECTORY, copied)
    rows = copied / "rows.csv"
    rows.write_text(rows.read_text().replace("-4.049969329113636", "-4.0", 1))
    result_path = copied / "result.json"
    result = json.loads(result_path.read_text())
    result["artifact_sha256"]["rows.csv"] = _sha256(rows)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with pytest.raises(AssertionError):
        _assert_fixed_bundle_hashes(copied)


@pytest.mark.parametrize("tamper", ["row", "device", "model_index", "delta", "pass"])
def test_runner_rejects_adapter_crosscheck_semantic_tamper(tamper: str) -> None:
    module = _load_runner_module()
    checks = json.loads(CROSSCHECK.read_text())
    if tamper == "row":
        checks[0]["row_id"] = "forged!1"
    elif tamper == "device":
        checks[0]["device"] = "cuda:0"
    elif tamper == "model_index":
        checks[0]["model_prediction_abs_deltas_kcal_mol"][0]["model_index"] = 9
    elif tamper == "delta":
        checks[0]["prediction_abs_delta_kcal_mol"] = 1e-12
    else:
        checks[0]["passes_5e12"] = False
    with pytest.raises(RuntimeError):
        module._validate_crosscheck(checks)


def test_main_authenticates_before_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner_module()
    configured = False

    def reject(*_args) -> None:
        raise RuntimeError("authentication sentinel")

    def configure(_root: Path) -> None:
        nonlocal configured
        configured = True

    monkeypatch.setattr(module, "authenticate_official_assets", reject)
    monkeypatch.setattr(module, "configure_runtime", configure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--workbook",
            str(tmp_path / "workbook.xlsx"),
            "--source-root",
            str(tmp_path / "source"),
            "--static-root",
            str(tmp_path / "static"),
            "--weights-root",
            str(tmp_path / "weights"),
            "--python-executable",
            str(tmp_path / "python"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(RuntimeError, match="authentication sentinel"):
        module.main()
    assert configured is False


def test_configure_runtime_copies_only_authenticated_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner_module()
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    trusted = supplied / "trusted.py"
    trusted.write_text("TRUSTED = True\n")
    (supplied / "__pycache__").mkdir()
    (supplied / "__pycache__/trusted.cpython-311.pyc").write_bytes(b"malicious")
    (supplied / "native.so").write_bytes(b"malicious")
    module.__dict__["OFFICIAL_STATIC_PYTHON_SHA256"] = {"trusted.py": _sha256(trusted)}
    imported_roots: list[Path] = []

    def fake_import(_name: str):
        imported_roots.append(Path(sys.path[0]))
        return SimpleNamespace(
            DataPoint=object,
            DatapointList=object,
            DataTensor=object,
            MolencoderDatabase=object,
            GsolvHsolv=object,
            load_checkpoint=object,
            load_scaler=object,
        )

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    try:
        module.configure_runtime(supplied)
        assert imported_roots
        clean_root = imported_roots[0]
        assert clean_root != supplied
        assert (clean_root / "trusted.py").read_bytes() == trusted.read_bytes()
        assert not list(clean_root.rglob("*.pyc"))
        assert not list(clean_root.rglob("*.so"))
    finally:
        if module.STATIC_RUNTIME is not None:
            module.STATIC_RUNTIME.cleanup()
        if imported_roots and str(imported_roots[0]) in sys.path:
            sys.path.remove(str(imported_roots[0]))


def _fake_data_assets(tmp_path: Path) -> tuple[Path, Path, str, tuple[str, ...]]:
    workbook = tmp_path / "workbook.xlsx"
    workbook.write_bytes(b"authenticated workbook")
    weights = tmp_path / "weights"
    weights.mkdir()
    hashes = []
    for index in range(10):
        path = weights / f"model{index}.pt"
        path.write_bytes(f"authenticated weight {index}".encode())
        hashes.append(_sha256(path))
    return workbook, weights, _sha256(workbook), tuple(hashes)


def test_private_data_snapshot_rehashes_destinations(tmp_path: Path) -> None:
    module = _load_runner_module()
    workbook, weights, workbook_hash, weight_hashes = _fake_data_assets(tmp_path)
    module.__dict__["WORKBOOK_SHA256"] = workbook_hash
    module.__dict__["SOLPROPMIX_CHECKPOINT_SHA256"] = weight_hashes
    try:
        private_workbook, private_weights = module.snapshot_official_data(
            workbook, weights
        )
        assert private_workbook != workbook
        assert private_weights != weights
        assert _sha256(private_workbook) == workbook_hash
        assert [
            _sha256(private_weights / f"model{index}.pt") for index in range(10)
        ] == list(weight_hashes)
    finally:
        if module.DATA_RUNTIME is not None:
            module.DATA_RUNTIME.cleanup()


@pytest.mark.parametrize("replace_target", ["workbook", "weight"])
def test_private_data_snapshot_rejects_post_auth_source_replacement(
    tmp_path: Path, replace_target: str
) -> None:
    module = _load_runner_module()
    workbook, weights, workbook_hash, weight_hashes = _fake_data_assets(tmp_path)
    module.__dict__["WORKBOOK_SHA256"] = workbook_hash
    module.__dict__["SOLPROPMIX_CHECKPOINT_SHA256"] = weight_hashes
    if replace_target == "workbook":
        workbook.write_bytes(b"replaced after authentication")
    else:
        (weights / "model4.pt").write_bytes(b"replaced after authentication")
    with pytest.raises(RuntimeError, match="private .* snapshot identity mismatch"):
        module.snapshot_official_data(workbook, weights)
    assert module.DATA_RUNTIME is None


def test_frozen_input_snapshot_survives_post_auth_source_replacement(
    tmp_path: Path,
) -> None:
    module = _load_runner_module()
    copied = tmp_path / "canonical"
    shutil.copytree(DIRECTORY, copied)
    try:
        private = module.snapshot_canonical_frozen_inputs(copied)
        original_private_hash = _sha256(private / "rows.csv")
        (copied / "rows.csv").write_bytes(b"replaced after authentication")
        assert _sha256(private / "rows.csv") == original_private_hash
        assert original_private_hash == EXPECTED_FILE_SHA256["rows.csv"]
    finally:
        if module.FROZEN_RUNTIME is not None:
            module.FROZEN_RUNTIME.cleanup()


def test_cli_rejects_coordinated_prediction_and_derived_field_tamper(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(DIRECTORY, copied)
    rows_path = copied / "rows.csv"
    rows = _read_rows(rows_path)
    row = rows[0]
    experiment = float(row["experimental_gsolv_kcal_mol"])
    published = float(row["published_workbook_qmexp_gt_kcal_mol"])
    cpu = float(row["cpu_prediction_kcal_mol"]) + 1.0
    gpu = float(row["gpu_prediction_kcal_mol"]) + 1.0
    cpu_error = cpu - experiment
    gpu_error = gpu - experiment
    delta = gpu - cpu
    row.update(
        {
            "cpu_prediction_kcal_mol": repr(cpu),
            "gpu_prediction_kcal_mol": repr(gpu),
            "cpu_signed_error_kcal_mol": repr(cpu_error),
            "gpu_signed_error_kcal_mol": repr(gpu_error),
            "cpu_abs_error_kcal_mol": repr(abs(cpu_error)),
            "gpu_abs_error_kcal_mol": repr(abs(gpu_error)),
            "gpu_minus_cpu_prediction_kcal_mol": repr(delta),
            "abs_gpu_minus_cpu_prediction_kcal_mol": repr(abs(delta)),
            "gpu_worsens_abs_error_exact": str(abs(gpu_error) > abs(cpu_error)),
            "gpu_worsens_abs_error_over_5e12": str(
                abs(gpu_error) > abs(cpu_error) + 5e-12
            ),
            "live_cpu_minus_workbook_qmexp_kcal_mol": repr(cpu - published),
        }
    )
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--workbook",
            str(tmp_path / "missing.xlsx"),
            "--source-root",
            str(tmp_path / "missing-source"),
            "--static-root",
            str(tmp_path / "missing-static"),
            "--weights-root",
            str(tmp_path / "missing-weights"),
            "--python-executable",
            sys.executable,
            "--output-dir",
            str(output),
            "--assemble-from-frozen",
            str(copied),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "canonical frozen input identity mismatch: rows.csv" in process.stderr
    assert not (output / "result.json").exists()
