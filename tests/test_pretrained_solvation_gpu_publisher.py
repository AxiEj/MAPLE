"""CPU-only synthetic metadata tests; no scientific card admission or model assets."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

PUBLISHER = (
    Path(__file__).resolve().parents[1]
    / "maple/function/benchmarking/solvation_gpu_evidence.py"
)
spec = importlib.util.spec_from_file_location("gpu_publisher", PUBLISHER)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def synthetic(tmp_path):
    source = tmp_path / "synthetic.py"
    source.write_text("# synthetic source, no model\n")
    sources = {str(source): publisher.digest(source)}
    geom = "a" * 64
    case = dict(
        case_id="anysolv:synthetic:water",
        route="anysolv",
        molecule="synthetic",
        numbers=[1],
        geometry_sha256=geom,
    )
    vacuum = {**case, "case_id": "anysolv:synthetic:vacuum"}
    panel = dict(
        scalar_cases=[case],
        vacuum_cases=[vacuum],
        finite_difference_cases=[],
        motion_cases=[],
        sampled_force_probe={"case_ids": []},
        timing={"warmup": 3, "repetitions": 10},
    )
    expectation = dict(
        required_sources=sources,
        source_revision="synthetic-revision",
        checkpoints={"gas": "b" * 64, "correction": "c" * 64},
        external_sources={"synthetic.py": publisher.digest(source)},
    )
    reports = {}
    for key, device, dtype, mode in [
        ("cpu64", "cpu", "float64", "reference"),
        ("cpu32", "cpu", "float32", "reference"),
        ("reference", "cuda:0", "float32", "reference"),
        ("optimized", "cuda:0", "float32", "optimized"),
    ]:
        execution = dict(
            device=device,
            compute_dtype=dtype,
            inference_mode=mode,
            matmul_precision="highest",
            tf32=False,
            autocast=False,
        )
        precision = dict(
            compute_dtype=dtype,
            matmul_precision="highest",
            matmul_tf32=False,
            cudnn_tf32=False,
            autocast=False,
            compiler_options=(
                {
                    "use_fast_math": False,
                    "cuda.use_fast_math": False,
                    "cutlass.use_fast_math": False,
                    "force_same_precision": True,
                }
                if mode == "optimized"
                else {}
            ),
        )
        runtime = (
            dict(
                accelerator_backend="cuda",
                hardware="synthetic",
                software_versions={"torch": "synthetic"},
                precision_state=copy.deepcopy(precision),
                precision_controls={
                    "compute_dtype": dtype,
                    "matmul_precision": "highest",
                    **{k: False for k in publisher.GPU_FORBIDDEN_PRECISION_CONTROLS},
                },
            )
            if device.startswith("cuda")
            else None
        )
        counts = dict(
            scalar=1,
            vacuum=1,
            finite_difference_components=0,
            motion_comparisons=0,
            sampled_force_replays=0,
        )
        reports[key] = dict(
            route="anysolv",
            panel_sha256=publisher.canonical(panel),
            inference_mode=mode,
            execution=execution,
            effective_precision=precision,
            runtime=runtime,
            experimental_labels_opened=False,
            confirmation_partition_opened=False,
            initialization_error=None,
            source_drift=False,
            source_sha256_before=copy.deepcopy(sources),
            source_sha256_after=copy.deepcopy(sources),
            counts={**counts, "failed_records": 0},
            expected_counts=counts,
            scalar_records=[
                dict(case_id=case["case_id"], status="success", energy_ev=1.0)
            ],
            vacuum_records=[
                dict(case_id=vacuum["case_id"], status="success", energy_ev=0.0)
            ],
            finite_difference_records=[],
            motion_records=[],
            sampled_force_records=[],
            adapter=dict(
                source_revision=expectation["source_revision"],
                source_identity=dict(
                    source_revision=expectation["source_revision"],
                    source_root=str(tmp_path),
                ),
                checkpoint={"sha256": expectation["checkpoints"]["correction"]},
            ),
            gas_base=dict(
                checkpoint={"sha256": expectation["checkpoints"]["gas"]},
                gas_reference_compute_dtype=dtype,
                gas_records=[
                    dict(
                        case_id=case["case_id"],
                        status="success",
                        geometry_sha256=geom,
                        gas_ev=2.0,
                    )
                ],
                combined_records=[
                    dict(
                        case_id=case["case_id"],
                        status="success",
                        geometry_sha256=geom,
                        atom_count=1,
                        gas_ev=2.0,
                        correction_ev=1.0,
                        combined_ev=3.0,
                    )
                ],
            ),
            timing=dict(
                status="success",
                case_id=case["case_id"],
                warmup_count=3,
                fresh_inference={"samples_seconds": [0.01] * 10},
            ),
        )
    return tmp_path, panel, expectation, reports


def write_reports(folder, reports):
    for key, report in reports.items():
        device = "cpu" if key.startswith("cpu") else "cuda"
        dtype = "float64" if key == "cpu64" else "float32"
        mode = "reference" if key.startswith("cpu") else key
        (folder / f"anysolv.{device}.{dtype}.{mode}.json").write_text(
            json.dumps(report)
        )


def load(synthetic):
    folder, panel, expectation, reports = synthetic
    write_reports(folder, reports)
    return publisher.load_reports(folder, "anysolv", panel, expectation)


def test_synthetic_reports_pass_content_identity_checks(synthetic):
    reports, _ = load(synthetic)
    assert set(reports) == {"cpu64", "cpu32", "reference", "optimized"}


@pytest.mark.parametrize(
    "field,value",
    [
        (("route",), "consolv"),
        (("panel_sha256",), "f" * 64),
        (("execution", "compute_dtype"), "float32"),
        (("execution", "device"), "cuda:0"),
        (("execution", "inference_mode"), "optimized"),
        (("inference_mode",), "optimized"),
        (("effective_precision", "compute_dtype"), "float32"),
        (("adapter", "source_revision"), "wrong"),
        (("adapter", "source_identity", "source_revision"), "wrong"),
        (("adapter", "checkpoint", "sha256"), "f" * 64),
        (("gas_base", "checkpoint", "sha256"), "f" * 64),
        (("gas_base", "gas_reference_compute_dtype"), "float32"),
        (("source_drift",), True),
        (("initialization_error",), "failed"),
        (("counts", "failed_records"), 1),
        (("expected_counts", "scalar"), 0),
    ],
)
def test_rejects_embedded_identity_swaps_before_publication(synthetic, field, value):
    row = synthetic[3]["cpu64"]
    for key in field[:-1]:
        row = row[key]
    row[field[-1]] = value
    with pytest.raises(ValueError):
        load(synthetic)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "status",
        "gas_missing",
        "combined_duplicate",
        "geometry",
        "sum",
    ],
)
def test_rejects_case_and_combination_corruption(synthetic, change):
    report = synthetic[3]["cpu64"]
    if change == "missing":
        report["scalar_records"].clear()
    elif change == "duplicate":
        report["scalar_records"] *= 2
    elif change == "status":
        report["scalar_records"][0]["status"] = "failed"
    elif change == "gas_missing":
        report["gas_base"]["gas_records"].clear()
    elif change == "combined_duplicate":
        report["gas_base"]["combined_records"] *= 2
    elif change == "geometry":
        report["gas_base"]["combined_records"][0]["geometry_sha256"] = "f" * 64
    else:
        report["gas_base"]["combined_records"][0]["combined_ev"] = 4.0
    with pytest.raises(ValueError):
        load(synthetic)


@pytest.mark.parametrize("change", ["before", "both", "live", "missing", "pin"])
def test_rejects_source_mismatch_in_cpu_receipt(synthetic, change):
    folder, _, expectation, reports = synthetic
    report = reports["cpu64"]
    name = next(iter(expectation["required_sources"]))
    if change == "live":
        Path(name).write_text("# changed\n")
    elif change == "missing":
        report["source_sha256_before"] = report["source_sha256_after"] = {}
    elif change == "pin":
        expectation["external_sources"]["synthetic.py"] = "f" * 64
    else:
        report["source_sha256_before"][name] = "f" * 64
        if change == "both":
            report["source_sha256_after"][name] = "f" * 64
    with pytest.raises(ValueError):
        load(synthetic)


def test_cpu32_file_cannot_impersonate_cpu64(synthetic):
    synthetic[3]["cpu64"] = copy.deepcopy(synthetic[3]["cpu32"])
    with pytest.raises(ValueError, match="dtype mismatch"):
        load(synthetic)


def test_gpu_runtime_precision_must_match(synthetic):
    synthetic[3]["optimized"]["runtime"]["precision_controls"]["tf32"] = True
    with pytest.raises(ValueError, match="precision controls"):
        load(synthetic)


def test_no_composite_timing_fallback(synthetic):
    with pytest.raises(ValueError, match="Composite performance receipt is required"):
        publisher.supplemental(
            "aimnet2-anysolv",
            {},
            {"scalar_cases": []},
            ["optimized"],
            "scalar_composition/v1",
            {},
            "a" * 64,
            {},
        )


def test_consolv_checkpoint_and_state_are_bound(synthetic):
    folder, panel, expectation, reports = synthetic
    panel["scalar_cases"][0]["route"] = "consolv"
    panel["vacuum_cases"][0]["route"] = "consolv"
    for cases in (panel["scalar_cases"], panel["vacuum_cases"]):
        cases[0]["case_id"] = cases[0]["case_id"].replace("anysolv", "consolv")
    expectation["checkpoints"] = {"solution": "b" * 64, "vacuum": "c" * 64}
    report = reports["cpu64"]
    report.update(route="consolv", panel_sha256=publisher.canonical(panel))
    report["scalar_records"][0]["case_id"] = panel["scalar_cases"][0]["case_id"]
    report["vacuum_records"][0]["case_id"] = panel["vacuum_cases"][0]["case_id"]
    report["timing"]["case_id"] = panel["scalar_cases"][0]["case_id"]
    report["adapter"].update(
        checkpoints={k: {"sha256": v} for k, v in expectation["checkpoints"].items()},
        source_state_unchanged=True,
        state_sha256_before={"solution": "a"},
        state_sha256_after={"solution": "a"},
    )
    publisher.validate_report(
        report, "consolv", panel, expectation, "cpu", "float64", "reference"
    )
    for change in ("checkpoint", "state"):
        bad = copy.deepcopy(report)
        if change == "checkpoint":
            bad["adapter"]["checkpoints"]["solution"]["sha256"] = "f" * 64
        else:
            bad["adapter"]["state_sha256_after"]["solution"] = "changed"
        with pytest.raises(ValueError):
            publisher.validate_report(
                bad, "consolv", panel, expectation, "cpu", "float64", "reference"
            )


@pytest.fixture
def composite(synthetic, monkeypatch):
    folder, panel, expectation, reports = synthetic
    monkeypatch.setattr(publisher, "ROOT", folder)
    for name in ("calculator_base", "set_calculator"):
        source = folder / f"maple/function/calculator/{name}.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic composite source\n")
    sources = {str(p): publisher.digest(p) for p in folder.rglob("*.py")}
    data = dict(
        schema_version=1,
        model_id="aimnet2-anysolv",
        route="anysolv",
        scope="public_production_calculate_gas_plus_correction",
        panel_sha256=publisher.canonical(panel),
        checkpoints=copy.deepcopy(expectation["checkpoints"]),
        source_sha256_before=sources,
        source_sha256_after=copy.deepcopy(sources),
        source_drift=False,
        source_revision=expectation["source_revision"],
        source_identity={
            "source_revision": expectation["source_revision"],
            "source_root": str(folder),
        },
    )
    for mode in ("reference", "optimized"):
        data[mode] = copy.deepcopy(reports[mode])
        data[mode]["resident_models"] = ["gas", "correction"]
        data[mode]["timing"]["cuda_synchronized"] = True
        data[mode]["timing"]["fresh_inference"]["geometry_sha256s"] = [
            f"{i:064x}" for i in range(10)
        ]
        data[mode]["memory"] = dict(
            device_total_bytes=2**31,
            peak_reserved_bytes=2**30,
            free_bytes_at_peak=2**30,
            measured_headroom_bytes=2**30,
        )
    path = folder / "synthetic-composite.json"
    return path, panel, expectation, data


def test_independent_combined_performance_schema(composite):
    path, panel, expectation, data = composite
    path.write_text(json.dumps(data))
    assert publisher.load_composite_performance(path, panel, expectation) == data


@pytest.mark.parametrize(
    "field,value",
    [
        (("scope",), "correction_only"),
        (("panel_sha256",), "wrong"),
        (("checkpoints", "gas"), "wrong"),
        (("source_drift",), True),
        (("reference", "execution", "compute_dtype"), "float64"),
        (("optimized", "timing", "cuda_synchronized"), False),
        (("optimized", "timing", "warmup_count"), 2),
        (("optimized", "timing", "fresh_inference", "samples_seconds"), [0.01] * 9),
        (
            ("optimized", "timing", "fresh_inference", "geometry_sha256s"),
            ["a" * 64] * 10,
        ),
        (("optimized", "resident_models"), ["correction"]),
        (("optimized", "memory", "measured_headroom_bytes"), 0),
    ],
)
def test_combined_performance_rejects_wrong_scope_controls_and_footprint(
    composite, field, value
):
    path, panel, expectation, data = composite
    row = data
    for key in field[:-1]:
        row = row[key]
    row[field[-1]] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        publisher.load_composite_performance(path, panel, expectation)


def test_historical_failures_are_packaged_with_verified_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    dest = tmp_path / "packaged"
    dest.mkdir()
    monkeypatch.setattr(publisher, "DEST", dest)
    source = tmp_path / "raw"
    source.mkdir()
    protocol = source / "force-audit-protocol-v3.json"
    protocol.write_text('{"synthetic": true}')
    (dest / "implementation-addendum.json").write_text(
        json.dumps({"force_methodology_protocol_sha256": publisher.digest(protocol)})
    )
    failure = source / "synthetic-failure.json"
    failure.write_text('{"status":"failed","error":1.0}')
    index = source / "preserved-failure-index.json"
    index.write_text(json.dumps({"files": {failure.name: publisher.digest(failure)}}))
    protocol.write_text(
        json.dumps({"complete_failure_index": {"sha256": publisher.digest(index)}})
    )
    (dest / "implementation-addendum.json").write_text(
        json.dumps({"force_methodology_protocol_sha256": publisher.digest(protocol)})
    )
    files, references = publisher.package_methodology(
        protocol, publisher.digest(protocol)
    )
    assert failure.name in files and "preserved_failure_index" in references
    failure.write_text("changed")
    with pytest.raises(ValueError, match="failure SHA"):
        publisher.package_methodology(protocol, publisher.digest(protocol))


def test_publisher_contains_no_optimization_removable_assertions():
    import ast

    assert not any(
        isinstance(node, ast.Assert)
        for node in ast.walk(ast.parse(PUBLISHER.read_text()))
    )
