"""Fail-closed contracts for the frozen ddPCM response benchmark."""

import json
import hashlib
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from tools.route2_release import run_ddpcm_response_benchmark as runner


def _protocol():
    protocol = runner.protocol_policy()
    protocol["requested_devices"] = ["cpu"]
    return protocol


def test_source_manifest_binds_extracted_response_core():
    path = "maple/solvation/experimental/mace_polar_response_core.py"
    manifest = runner.source_manifest()
    assert path in manifest["candidate_files"]
    assert manifest["candidate_files"][path] == runner.sha(runner.ROOT / path)


def _result(mode, atom_count, value=0.0):
    if mode == "hvp":
        return {"hvp_eV_A2": [[value] * 3 for _ in range(atom_count)]}
    return {
        "energy_eV": value,
        "forces_eV_A": [[value] * 3 for _ in range(atom_count)],
        "hessian_eV_A2": [[value] * (3 * atom_count) for _ in range(3 * atom_count)],
    }


def _timed_row(backend, case, mode, protocol_sha, *, elapsed=None, memory=None):
    elapsed = elapsed or ([10.0] * 3 if backend == "reference" else [1.0] * 3)
    memory = memory or ([1000] * 3 if backend == "reference" else [100] * 3)
    atom_count = {"water": 3, "methane": 5}[case]
    return {
        "protocol_sha256": protocol_sha,
        "backend": backend,
        "device": "cpu",
        "case": case,
        "mode": mode,
        "atom_count": atom_count,
        "completed": True,
        "resource_preflight": (
            {"estimated_total_bytes": 10_000} if backend == "response" else None
        ),
        "measurements": [
            {
                "index": index,
                "elapsed_seconds": elapsed[index],
                "uss_sampled_peak_bytes": memory[index],
                "result": _result(mode, atom_count),
            }
            for index in range(3)
        ],
    }


def _write_row(root, record, suffix=""):
    identity = "-".join(
        str(record[key]) for key in ("backend", "device", "case", "mode")
    )
    path = root / "rows" / f"{identity}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def _complete_output(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n")
    protocol_sha = runner.sha(protocol_path)
    monkeypatch.setattr(runner, "verify", lambda args: _protocol())
    for case in ("water", "methane"):
        for mode in ("hvp", "hessian"):
            for backend in ("reference", "response"):
                _write_row(tmp_path, _timed_row(backend, case, mode, protocol_sha))
    for case in ("water", "methane", "acetone-10", "hexane-20"):
        _write_row(
            tmp_path,
            {
                "protocol_sha256": protocol_sha,
                "backend": "response",
                "device": "cpu",
                "case": case,
                "mode": "validation",
                "completed": True,
                "result": {"validation_pass": True},
            },
        )
    for case in ("hexadecane-50", "alkylamine-100"):
        atom_count = 50 if case == "hexadecane-50" else 100
        _write_row(
            tmp_path,
            {
                "protocol_sha256": protocol_sha,
                "backend": "response",
                "device": "cpu",
                "case": case,
                "mode": "resource",
                "atom_count": atom_count,
                "completed": True,
                "within_limit": False,
                "resource": {
                    "atom_count": atom_count,
                    "derivative_order": 2,
                    "host_peak_bytes": 1000,
                    "device_peak_bytes": 0,
                    "conservative_peak_bytes": 1000,
                },
            },
        )
    for case in ("water", "methane", "acetone-10"):
        _write_row(
            tmp_path,
            {
                "protocol_sha256": protocol_sha,
                "backend": "response",
                "device": "cpu",
                "case": case,
                "mode": "continuum-memory",
                "completed": True,
                "result": {
                    "incremental_peak_bytes": 100,
                    "continuum_estimate_bytes": 1000,
                    "resource_envelope_pass": True,
                    "claim_boundary": "incremental only",
                },
            },
        )
    return SimpleNamespace(output_dir=tmp_path, checkpoint=tmp_path / "model")


def _record(tmp_path, identity):
    backend, device, case, mode = identity
    path = tmp_path / "rows" / f"{backend}-{device}-{case}-{mode}.json"
    return path, json.loads(path.read_text())


def test_summarize_rejects_a_missing_required_row(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, _ = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    path.unlink()
    with pytest.raises(RuntimeError, match="Missing preregistered rows"):
        runner.summarize(args)


def test_summarize_rejects_a_missing_timing_repetition(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"].pop()
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="timing repetitions"):
        runner.summarize(args)


def test_summarize_rejects_duplicate_row_identity(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    _, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    _write_row(tmp_path, row, suffix="-duplicate")
    with pytest.raises(RuntimeError, match="Duplicate result row identity"):
        runner.summarize(args)


def test_summarize_rejects_an_unpreregistered_row_identity(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    protocol_sha = runner.sha(tmp_path / "protocol.json")
    _write_row(
        tmp_path,
        _timed_row("response", "water", "hvp", protocol_sha),
        suffix="-unexpected",
    ).write_text(
        json.dumps(
            {
                **_timed_row("response", "water", "hvp", protocol_sha),
                "case": "not-in-panel",
            }
        )
    )
    with pytest.raises(RuntimeError, match="Unexpected result row identities"):
        runner.summarize(args)


def test_summarize_rejects_row_from_another_protocol(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["protocol_sha256"] = "wrong"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="protocol binding failed"):
        runner.summarize(args)


def test_summarize_rejects_repetition_indexes_out_of_order(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"][1]["index"] = 0
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="repetition indexes"):
        runner.summarize(args)


def test_summarize_rejects_result_shapes_that_would_broadcast(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"][0]["result"]["hvp_eV_A2"] = [[0.0, 0.0]]
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        runner.summarize(args)


def test_summarize_rejects_result_shape_bound_to_wrong_atom_count(
    tmp_path, monkeypatch
):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    for measurement in row["measurements"]:
        measurement["result"]["hvp_eV_A2"] = [[0.0] * 3 for _ in range(5)]
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        runner.summarize(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("elapsed_seconds", 0.0),
        ("elapsed_seconds", np.inf),
        ("uss_sampled_peak_bytes", -1),
    ],
)
def test_summarize_rejects_invalid_timing_or_memory_measurement(
    tmp_path, monkeypatch, field, value
):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"][0][field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="finite and positive"):
        runner.summarize(args)


def test_summarize_does_not_allow_one_slow_repetition_to_hide_behind_median(
    tmp_path, monkeypatch
):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"][2]["elapsed_seconds"] = 10.0
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    item = next(
        x for x in report["comparisons"] if x["case"] == "water" and x["mode"] == "hvp"
    )
    assert item["performance_pass"] is False


def test_summarize_reports_timing_median_mad_and_repetition_order(
    tmp_path, monkeypatch
):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    for measurement, elapsed in zip(row["measurements"], (1.0, 2.0, 3.0), strict=True):
        measurement["elapsed_seconds"] = elapsed
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    item = next(
        x for x in report["comparisons"] if x["case"] == "water" and x["mode"] == "hvp"
    )
    assert item["new_time_median_seconds"] == 2.0
    assert item["new_time_mad_seconds"] == 1.0
    assert item["repetition_order"] == [0, 1, 2]


def test_summarize_marks_any_numerical_mismatch_as_failed(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["measurements"][2]["result"]["hvp_eV_A2"][0][0] = 2e-6
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    item = next(
        x for x in report["comparisons"] if x["case"] == "water" and x["mode"] == "hvp"
    )
    assert item["accuracy_pass"] is False
    assert report["implementation_pass"] is False


def test_full_pes_resource_underestimate_fails_performance_gate(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "water", "hvp"))
    row["resource_preflight"]["estimated_total_bytes"] = 99
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    item = next(
        x for x in report["comparisons"] if x["case"] == "water" and x["mode"] == "hvp"
    )
    assert item["resource_envelope_pass"] is False
    assert item["performance_pass"] is False


def test_continuum_resource_underestimate_fails_stage1_gate(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "acetone-10", "continuum-memory"))
    row["result"]["resource_envelope_pass"] = False
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["stage1_pass_before_final_review"] is False


def test_stage1_rejects_arbitrary_hexane20_runtime_failure(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "hexane-20", "validation"))
    row.update(completed=False, failure={"type": "RuntimeError", "message": "boom"})
    row.pop("result")
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["stage1_pass_before_final_review"] is False


def test_stage1_accepts_explicit_hexane20_memoryerror_boundary(tmp_path, monkeypatch):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "hexane-20", "validation"))
    row.update(completed=False, failure={"type": "MemoryError", "message": "bounded"})
    row["resource_boundary"] = {
        "preflight_rejected": True,
        "derivative_order": 2,
        "message": "bounded",
    }
    row.pop("result")
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["stage1_pass_before_final_review"] is True


def test_stage1_rejects_hexane20_memoryerror_without_preflight_boundary(
    tmp_path, monkeypatch
):
    args = _complete_output(tmp_path, monkeypatch)
    path, row = _record(tmp_path, ("response", "cpu", "hexane-20", "validation"))
    row.update(completed=False, failure={"type": "MemoryError", "message": "later"})
    row.pop("result")
    path.write_text(json.dumps(row))
    runner.summarize(args)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["stage1_pass_before_final_review"] is False


def test_main_rejects_devices_outside_frozen_cpu_cuda0_set(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "run", "--output-dir", "/tmp/out", "--device", "cuda:1"],
    )
    with pytest.raises(SystemExit):
        runner.main()


def test_gpu_timing_guard_rejects_a_foreign_compute_process(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: f"{runner.os.getpid()}\n424242\n",
    )
    with pytest.raises(RuntimeError, match="foreign compute processes.*424242"):
        runner.assert_gpu_idle_for_timing()


def test_gpu_timing_guard_fails_clearly_without_nvidia_smi(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(runner.subprocess, "check_output", unavailable)
    with pytest.raises(
        RuntimeError, match="nvidia-smi compute-process guard unavailable"
    ):
        runner.assert_gpu_idle_for_timing()


def test_environment_records_torch_threads_and_numpy_blas(monkeypatch):
    monkeypatch.setattr(runner.np, "show_config", lambda: print("BLAS witness"))
    record = runner.environment()
    assert record["torch_threads"]["intraop"] > 0
    assert record["torch_threads"]["interop"] > 0
    assert record["numpy_blas"]["show_config"] == "BLAS witness\n"


def test_environment_binds_ambient_default_dtype_without_changing_it(monkeypatch):
    import torch

    monkeypatch.setattr(runner.np, "show_config", lambda: None)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        single = runner.environment()
        assert torch.get_default_dtype() == torch.float32
        torch.set_default_dtype(torch.float64)
        double = runner.environment()
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original)
    assert single["torch_default_dtype"] == "torch.float32"
    assert double["torch_default_dtype"] == "torch.float64"


def test_response_test_collection_does_not_change_process_default_dtype():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, torch; "
            "before = torch.get_default_dtype(); "
            "runpy.run_path('tests/route2_vnext/test_ddpcm_response.py'); "
            "assert torch.get_default_dtype() == before",
        ],
        cwd=runner.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_verify_rejects_default_dtype_drift(tmp_path, monkeypatch):
    import torch

    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    source = {"candidate_files": {}}
    hardware = {"platform": "frozen"}
    monkeypatch.setattr(runner.np, "show_config", lambda: None)
    original = torch.get_default_dtype()
    protocol = {
        **runner.protocol_policy(),
        "source_manifest": source,
        "environment": runner.environment(),
        "hardware_context": hardware,
        "checkpoint_sha256": runner.sha(checkpoint),
        "git_head": "head",
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "source_manifest", lambda: source)
    monkeypatch.setattr(runner, "hardware_context", lambda: hardware)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kwargs: "head\n" if "rev-parse" in command else b"",
    )
    try:
        changed = torch.float64 if original == torch.float32 else torch.float32
        torch.set_default_dtype(changed)
        with pytest.raises(RuntimeError, match="source/runtime drift"):
            runner.verify(SimpleNamespace(output_dir=tmp_path, checkpoint=checkpoint))
    finally:
        torch.set_default_dtype(original)


def _valid_source_diagnostics():
    return {
        "source_active_sha256": "a" * 64,
        "source_embedded_sha256": "b" * 64,
        "source_witnesses": {
            "schema": "macepolar-response-source-witness-v1",
            "active_columns": [0, 2, 3, 4],
            "inactive_columns": [1, 5, 6, 7],
            "inactive_nonzero_count": 0,
            "inactive_exact_zero": True,
            "charge_sum_e": 0.0,
            "charge_atol_e": 1e-8,
            "neutral_charge_pass": True,
            "probe_policy": "normalized-linspace-minus1-plus1-v1",
            "probe_sha256": "c" * 64,
            "source_jacobian_frobenius_norm": 2e-12,
            "source_jvp_norm": 2e-12,
            "source_vjp_norm": 2e-12,
            "mixed_R_source_max_abs": 2e-12,
            "weighted_source_curvature_hvp_norm": 2e-12,
        },
    }


def test_source_witness_gate_rejects_an_omitted_witness_payload():
    diagnostics = _valid_source_diagnostics()
    diagnostics.pop("source_witnesses")
    assert runner._source_witnesses_pass(diagnostics, 1e-12) is False


@pytest.mark.parametrize(
    "field",
    [
        "source_jacobian_frobenius_norm",
        "source_jvp_norm",
        "source_vjp_norm",
        "mixed_R_source_max_abs",
        "weighted_source_curvature_hvp_norm",
    ],
)
@pytest.mark.parametrize("mutation", ["omit", "zero"])
def test_source_witness_gate_rejects_each_missing_or_zero_nonzero_witness(
    field, mutation
):
    diagnostics = _valid_source_diagnostics()
    if mutation == "omit":
        diagnostics["source_witnesses"].pop(field)
    else:
        diagnostics["source_witnesses"][field] = 0.0
    assert runner._source_witnesses_pass(diagnostics, 1e-12) is False


def test_source_witness_gate_accepts_complete_bound_witnesses():
    assert runner._source_witnesses_pass(_valid_source_diagnostics(), 1e-12) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_columns", [0, 1, 2, 3]),
        ("inactive_nonzero_count", 1),
        ("inactive_exact_zero", False),
        ("charge_sum_e", 2e-8),
        ("charge_sum_e", np.nan),
        ("neutral_charge_pass", False),
        ("probe_sha256", "not-a-sha"),
    ],
)
def test_source_witness_gate_rejects_invalid_binding_evidence(field, value):
    diagnostics = _valid_source_diagnostics()
    diagnostics["source_witnesses"][field] = value
    assert runner._source_witnesses_pass(diagnostics, 1e-12) is False


@pytest.mark.parametrize("field", ["source_active_sha256", "source_embedded_sha256"])
def test_source_witness_gate_rejects_missing_top_level_source_hash(field):
    diagnostics = _valid_source_diagnostics()
    diagnostics.pop(field)
    assert runner._source_witnesses_pass(diagnostics, 1e-12) is False


def test_prepare_records_hardware_runtime_snapshot(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(runner, "CHECKPOINT_SHA", runner.sha(checkpoint))
    monkeypatch.setattr(runner, "source_manifest", lambda: {"candidate_files": {}})
    monkeypatch.setattr(runner, "environment", lambda: {"versions": {}})
    snapshot = {
        "platform": "test-platform",
        "cpu_count": 8,
        "torch_cuda_version": "12.8",
        "nvidia_smi_gpu_inventory": "0, GPU, uuid, driver",
    }
    monkeypatch.setattr(runner, "hardware_context", lambda: snapshot)

    def git_output(command, **kwargs):
        return "head\n" if "rev-parse" in command else b"diff"

    monkeypatch.setattr(runner.subprocess, "check_output", git_output)
    args = SimpleNamespace(output_dir=tmp_path / "output", checkpoint=checkpoint)
    runner.prepare(args)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    assert protocol["hardware_context"] == snapshot
    assert protocol["source_witness_floor"] == 1e-12


def test_verify_rejects_tracked_diff_drift(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    source = {"candidate_files": {}}
    runtime = {"versions": {}}
    hardware = {"platform": "frozen"}
    protocol = {
        **runner.protocol_policy(),
        "source_manifest": source,
        "environment": runtime,
        "hardware_context": hardware,
        "checkpoint_sha256": runner.sha(checkpoint),
        "git_head": "head",
        "tracked_diff_sha256": hashlib.sha256(b"frozen diff").hexdigest(),
    }
    tmp_path.joinpath("protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "source_manifest", lambda: source)
    monkeypatch.setattr(runner, "environment", lambda: runtime)
    monkeypatch.setattr(runner, "hardware_context", lambda: hardware)

    def git_output(command, **kwargs):
        return "head\n" if "rev-parse" in command else b"changed diff"

    monkeypatch.setattr(runner.subprocess, "check_output", git_output)
    args = SimpleNamespace(output_dir=tmp_path, checkpoint=checkpoint)
    with pytest.raises(RuntimeError, match="tracked diff drift"):
        runner.verify(args)


def _leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, (*prefix, key))
    else:
        yield prefix


@pytest.mark.parametrize("path", list(_leaf_paths(runner.protocol_policy())))
def test_verify_rejects_mutation_of_every_frozen_policy_field(
    tmp_path, monkeypatch, path
):
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    source, runtime, hardware = {}, {}, {}
    protocol = {
        **runner.protocol_policy(),
        "source_manifest": source,
        "environment": runtime,
        "hardware_context": hardware,
        "checkpoint_sha256": runner.sha(checkpoint),
        "git_head": "head",
        "tracked_diff_sha256": "d" * 64,
    }
    cursor = protocol
    for key in path[:-1]:
        cursor = cursor[key]
    value = cursor[path[-1]]
    cursor[path[-1]] = (
        not value
        if isinstance(value, bool)
        else (
            value + 1
            if isinstance(value, (int, float))
            else (
                [*value, "mutation"] if isinstance(value, list) else value + "-mutation"
            )
        )
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "source_manifest", lambda: source)
    monkeypatch.setattr(runner, "environment", lambda: runtime)
    monkeypatch.setattr(runner, "hardware_context", lambda: hardware)
    with pytest.raises(RuntimeError, match="policy drift"):
        runner.verify(SimpleNamespace(output_dir=tmp_path, checkpoint=checkpoint))


class _FakePes:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def hessian_vector_product(self, atoms, vector):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("evaluation failed")
        return np.zeros_like(vector)

    def _preflight(self, derivative_order):
        assert derivative_order == 2
        return {"estimated_total_bytes": 1_000_000_000}


def _run_args(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        checkpoint=tmp_path / "model",
        backend="response",
        device="cpu",
        case="water",
        mode="hvp",
    )


def _install_run_fakes(tmp_path, monkeypatch, pes):
    panel = tmp_path / "panel.json"
    geometry = {
        "symbols": ["H"],
        "positions_angstrom": [[0.0, 0.0, 0.0]],
        "charge": 0,
        "multiplicity": 1,
    }
    case = {
        "name": "water",
        "solvent": "water",
        **geometry,
        "geometry_sha256": runner.canonical_sha(geometry),
    }
    panel.write_text(json.dumps({"cases": [case]}))
    monkeypatch.setattr(runner, "PANEL", panel)
    tmp_path.joinpath("protocol.json").write_text("{}\n")
    monkeypatch.setattr(runner, "build_pes", lambda args, selected: pes)


def test_run_marks_completed_false_when_postrun_verification_fails(
    tmp_path, monkeypatch
):
    _install_run_fakes(tmp_path, monkeypatch, _FakePes())
    calls = 0

    def verify(_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("postrun drift")

    monkeypatch.setattr(runner, "verify", verify)
    with pytest.raises(SystemExit):
        runner.run(_run_args(tmp_path))
    row = json.loads((tmp_path / "rows/response/cpu/water/hvp.json").read_text())
    assert row["completed"] is False
    assert row["failure"]["message"] == "postrun drift"


def test_run_retains_completed_repetitions_when_later_evaluation_fails(
    tmp_path, monkeypatch
):
    _install_run_fakes(tmp_path, monkeypatch, _FakePes(fail_on_call=3))
    monkeypatch.setattr(runner, "verify", lambda args: None)
    with pytest.raises(SystemExit):
        runner.run(_run_args(tmp_path))
    row = json.loads((tmp_path / "rows/response/cpu/water/hvp.json").read_text())
    assert row["completed"] is False
    assert [item["index"] for item in row["measurements"]] == [0]


def test_cuda_measurements_record_starting_and_peak_allocator_bytes(
    tmp_path, monkeypatch
):
    _install_run_fakes(tmp_path, monkeypatch, _FakePes())
    monkeypatch.setattr(
        runner,
        "verify",
        lambda args: {"source_witness_floor": runner.SOURCE_WITNESS_FLOOR},
    )
    guards = []

    def gpu_guard():
        evidence = {"sequence": len(guards)}
        guards.append(evidence)
        return evidence

    monkeypatch.setattr(runner, "assert_gpu_idle_for_timing", gpu_guard)
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 11)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 22)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 33)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 44)
    args = _run_args(tmp_path)
    args.device = "cuda:0"
    runner.run(args)
    row = json.loads((tmp_path / "rows/response/cuda-0/water/hvp.json").read_text())
    for measurement in row["measurements"]:
        assert measurement["cuda_start_allocated_bytes"] == 11
        assert measurement["cuda_start_reserved_bytes"] == 22
        assert measurement["cuda_peak_allocated_bytes"] == 33
        assert measurement["cuda_peak_reserved_bytes"] == 44
        assert measurement["cuda_process_guard_before"] is not None
        assert measurement["cuda_process_guard_after"] is not None
    assert len(guards) == 2 * runner.REPETITIONS


def test_benchmark_lock_rejects_concurrent_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "benchmark.lock")

    def flock(_fd, operation):
        if operation & runner.fcntl.LOCK_NB:
            raise BlockingIOError("busy")

    monkeypatch.setattr(runner.fcntl, "flock", flock)
    with pytest.raises(RuntimeError, match="Another host benchmark"):
        with runner.benchmark_lock():
            pass


def test_run_holds_host_lock_across_warmup_and_all_repetitions(tmp_path, monkeypatch):
    active = {"lock": False}

    class LockCheckingPes(_FakePes):
        def hessian_vector_product(self, atoms, vector):
            assert active["lock"] is True
            return super().hessian_vector_product(atoms, vector)

    @contextlib.contextmanager
    def lock():
        active["lock"] = True
        try:
            yield {"test": True}
        finally:
            active["lock"] = False

    pes = LockCheckingPes()
    _install_run_fakes(tmp_path, monkeypatch, pes)
    monkeypatch.setattr(runner, "verify", lambda args: {})
    monkeypatch.setattr(runner, "benchmark_lock", lock)
    runner.run(_run_args(tmp_path))
    assert pes.calls == 1 + runner.REPETITIONS
    assert active["lock"] is False
    row = json.loads((tmp_path / "rows/response/cpu/water/hvp.json").read_text())
    assert [item["stage"] for item in row["source_checks"]] == [
        "after-warmup",
        "after-repetition-0",
        "after-repetition-1",
        "after-repetition-2",
    ]


def test_resource_estimate_records_device_specific_peak_location():
    from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse

    cpu = TorchDDPCMResponse._resource_estimate(50, 15, 1202, 2, 4_000_000_000, "cpu")
    cuda = TorchDDPCMResponse._resource_estimate(50, 15, 1202, 2, 4_000_000_000, "cuda")
    assert cpu.host_peak_bytes == cpu.conservative_peak_bytes
    assert cpu.device_peak_bytes == 0
    assert cuda.host_peak_bytes == cuda.coefficient_table_bytes
    assert cuda.device_peak_bytes > 0
    assert cuda.host_peak_bytes + cuda.device_peak_bytes == cuda.conservative_peak_bytes
