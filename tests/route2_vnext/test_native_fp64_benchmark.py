"""Fail-closed contracts for the native-FP64 v2 qualification runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from tools.route2_release import run_ddpcm_response_native_fp64_v2_benchmark as runner


def test_protocol_is_new_identity_and_preserves_fixed_gates():
    policy = runner.protocol_policy()
    assert policy["schema"] == "ddpcm-native-fp64-v2-qualification-v1"
    assert policy["warmups"] == 1
    assert policy["repetitions"] == 3
    assert policy["audit_steps_angstrom"] == [2e-5, 1e-5, 5e-6]
    assert policy["validation_limits"]["force_fd_eV_A2"] == 1e-4
    assert policy["validation_limits"]["fine_step_agreement_eV_A2"] == 1e-4
    assert policy["resource_policy"]["second_order_limit_bytes"] == 4_000_000_000
    assert policy["scientific_admitted"] is False


def test_performance_cases_are_only_capped_water_and_methane():
    policy = runner.protocol_policy()
    identities = runner.required_row_identities(policy)
    timing = {case for _, _, case, mode in identities if mode in {"hvp", "hessian"}}
    assert timing == {"water", "methane"}
    assert all(backend in {"oracle", "response"} for backend, _, _, _ in identities)


def test_initialize_process_fp64_is_exactly_once(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(runner, "_PROCESS_FP64_INITIALIZED", False)
    original = torch.get_default_dtype()
    try:
        runner.initialize_process_fp64()
        assert torch.get_default_dtype() is torch.float64
        with pytest.raises(RuntimeError, match="exactly once"):
            runner.initialize_process_fp64()
    finally:
        torch.set_default_dtype(original)
        monkeypatch.setattr(runner, "_PROCESS_FP64_INITIALIZED", False)


def test_write_json_never_overwrites(tmp_path):
    path = tmp_path / "row.json"
    runner.write_json_new(path, {"first": True})
    with pytest.raises(FileExistsError, match="Preserve prior evidence"):
        runner.write_json_new(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def _measurement(value, n):
    return {
        "elapsed_seconds": value,
        "uss_sampled_peak_bytes": int(value * 100),
        "result": {
            "energy_eV": 0.0,
            "forces_eV_A": [[0.0] * 3 for _ in range(n)],
            "hessian_eV_A2": [[0.0] * (3 * n) for _ in range(3 * n)],
        },
    }


def test_summarize_rejects_incomplete_matrix(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n")
    monkeypatch.setattr(
        runner, "verify_protocol", lambda args: runner.protocol_policy()
    )
    args = SimpleNamespace(output_dir=tmp_path, checkpoint=tmp_path / "model")
    with pytest.raises(RuntimeError, match="Missing preregistered rows"):
        runner.summarize(args)


def test_performance_gate_requires_memory_ranges_not_to_overlap():
    policy = runner.protocol_policy()["performance"]
    comparison = runner.performance_statistics(
        oracle_times=[10.0, 11.0, 12.0],
        response_times=[1.0, 2.0, 3.0],
        oracle_memory=[100.0, 110.0, 120.0],
        response_memory=[80.0, 105.0, 115.0],
        mode="hessian",
        policy=policy,
    )
    assert comparison["time_ranges_nonoverlap"] is True
    assert comparison["memory_ranges_nonoverlap"] is False
    assert comparison["performance_conditions_pass"] is False
    assert comparison["raw_memory_bytes"]["oracle"] == [100.0, 110.0, 120.0]
    assert "memory_mad_bytes" in comparison


def test_verify_rejects_drift_in_reused_common_runner(tmp_path, monkeypatch):
    frozen = runner.source_manifest()
    common_name = "tools/route2_release/run_ddpcm_response_benchmark.py"
    assert common_name in frozen["files_sha256"]
    protocol = {
        **runner.protocol_policy(),
        "source_manifest": frozen,
        "runtime_environment": {},
        "hardware_context": {},
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    drifted = json.loads(json.dumps(frozen))
    drifted["files_sha256"][common_name] = "0" * 64
    monkeypatch.setattr(runner, "source_manifest", lambda: drifted)
    args = SimpleNamespace(output_dir=tmp_path, checkpoint=tmp_path / "model")
    with pytest.raises(RuntimeError, match="source manifest drift"):
        runner.verify_protocol(args)


def test_fd_audit_does_not_relabel_unrelated_failure_as_topology_rejection(
    monkeypatch,
):
    monkeypatch.setattr(
        runner, "_topology", lambda *args: (_ for _ in ()).throw(MemoryError("oom"))
    )
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(MemoryError, match="oom"):
        runner._finite_difference_audit(
            pes=object(),
            reference=object(),
            atoms=atoms,
            direction=np.zeros((1, 3)),
            hvp=np.zeros(3),
            step=2e-5,
            central={},
        )


def test_source_manifest_contains_full_structured_response_closure():
    files = runner.source_manifest()["files_sha256"]
    required = {
        "maple/solvation/continuum/ddpcm_response_operators.py",
        "maple/solvation/continuum/response_tensor_binding.py",
        "maple/solvation/continuum/response_topology.py",
        "maple/solvation/continuum/solid_harmonic_response.py",
        "maple/solvation/derivatives/response.py",
    }
    assert required <= set(files)
