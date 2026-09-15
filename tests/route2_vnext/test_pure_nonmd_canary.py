from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.route2_release import run_pure_mace_polar_nonmd_canary as canary


def _protocol() -> dict:
    return json.loads(canary.PROTOCOL.read_text())


def test_family_selector_expands_only_requested_shard():
    selected = canary.select_cases(_protocol(), ("sp", "irc"))
    assert selected == (
        "sp", "irc.gs", "irc.gs-complete", "irc.hpc", "irc.eulerpc", "irc.lqa"
    )


def test_all_selector_covers_every_preregistered_variant_once():
    protocol = _protocol()
    selected = canary.select_cases(protocol, ("all",))
    assert selected == tuple(protocol["case_order"])
    assert len(selected) == len(set(selected))
    assert {
        "opt.lbfgs", "opt.rfo", "opt.sd", "opt.cg", "opt.sdcg",
        "scan.rigid", "scan.relaxed-lbfgs", "scan.relaxed-sd",
        "scan.relaxed-cg", "scan.relaxed-sdcg", "freq.mw", "ts.prfo",
        "ts.neb", "ts.cineb", "ts.nebts", "ts.string", "ts.cistring",
        "ts.stringts", "ts.dimer", "ts.autoneb", "irc.gs", "irc.hpc",
        "irc.gs-complete", "irc.eulerpc", "irc.lqa", "sp",
    } == set(selected)


def test_unknown_or_duplicate_selector_is_rejected():
    with pytest.raises(ValueError, match="unknown task selector"):
        canary.select_cases(_protocol(), ("md",))
    with pytest.raises(ValueError, match="more than once"):
        canary.select_cases(_protocol(), ("sp", "sp"))


def test_rendered_cpu_and_cuda_inputs_use_same_fixture_and_distinct_identity():
    protocol = _protocol()
    cpu = canary.render_input(protocol, "cpu", "ts.neb")
    cuda = canary.render_input(protocol, "cuda:0", "ts.neb")
    assert "#device=cpu" in cpu
    assert "#device=cuda:0" in cuda
    assert protocol["identities"]["cpu"]["profile"] in cpu
    assert protocol["identities"]["cuda"]["profile"] in cuda
    assert cpu.split("\n\n", 1)[1] == cuda.split("\n\n", 1)[1]
    assert cpu.count("0 1") == 3


def test_timeout_is_executed_false_and_retains_partial_output(tmp_path):
    protocol = _protocol()

    def timeout_runner(command, *, cwd, env, timeout):
        return canary.ProcessResult(
            returncode=None,
            stdout="partial stdout",
            stderr="partial stderr",
            timed_out=True,
            elapsed_seconds=timeout,
        )

    record = canary.execute_case(
        protocol, "cpu", "opt.lbfgs", tmp_path / "case", {}, timeout_runner
    )
    assert record["executed"] is False
    assert record["numerical"]["converged"] is None
    assert record["infrastructure_status"] == "timeout"
    assert (tmp_path / "case/stdout.log").read_text() == "partial stdout"
    assert (tmp_path / "case/stderr.log").read_text() == "partial stderr"


def test_bounded_nonconvergence_is_executed_but_not_converged(tmp_path):
    protocol = _protocol()

    def capped_runner(command, *, cwd, env, timeout):
        output = Path(command[-1])
        output.with_name(output.stem + "_status.json").write_text(json.dumps({
            "schema": "maple-pure-nonmd-status-v2",
            "workflow": "opt",
            "method": "lbfgs",
            "executed": True,
            "converged": False,
            "termination_reason": "maximum_iterations_reached",
            "termination_class": "bounded_nonconvergence",
            "iterations": 8,
            "final_metrics": {"max_force": 0.02},
            "geometry": [{"symbols": ["O", "H", "H"],
                          "positions_angstrom": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}],
            "trace": [{"iteration": 8, "max_force": 0.02}],
            "actual_device": "cpu",
            "profile": protocol["identities"]["cpu"]["profile"],
            "scalar": protocol["identities"]["cpu"]["scalar_id"],
        }))
        return canary.ProcessResult(1, "bounded", "", False, 1.0)

    record = canary.execute_case(
        protocol, "cpu", "opt.lbfgs", tmp_path / "case", {}, capped_runner
    )
    assert record["executed"] is True
    assert record["finite"] is True
    assert record["numerical"] == {
        "converged": False,
        "status": "maximum_iterations_reached",
        "iterations": 8,
        "final_metrics": {"max_force": 0.02},
    }
    assert record["infrastructure_status"] == "completed_nonconverged"


def test_missing_status_is_fail_closed_even_after_zero_exit(tmp_path):
    def runner(command, *, cwd, env, timeout):
        return canary.ProcessResult(0, "done", "", False, 0.1)

    record = canary.execute_case(
        _protocol(), "cpu", "sp", tmp_path / "case", {}, runner
    )
    assert record["executed"] is False
    assert record["infrastructure_status"] == "missing_status"


def test_runner_exception_is_recorded_instead_of_aborting_shard(tmp_path):
    def runner(command, *, cwd, env, timeout):
        raise OSError("cannot launch")

    record = canary.execute_case(
        _protocol(), "cpu", "sp", tmp_path / "case", {}, runner
    )
    assert record["executed"] is False
    assert record["infrastructure_status"] == "runner_exception"
    assert record["runner_error"] == "OSError: cannot launch"
    assert "cannot launch" in (tmp_path / "case/stderr.log").read_text()


def test_source_change_stops_new_cases_and_preserves_completed_case(tmp_path, monkeypatch):
    protocol = _protocol()
    snapshots = iter(({"a": "before"}, {"a": "changed"}))
    monkeypatch.setattr(canary, "source_manifest", lambda: next(snapshots))
    calls = []

    def fake_execute(protocol, device, case_id, case_dir, env, process_runner):
        calls.append(case_id)
        return {"executed": True, "finite": True,
                "infrastructure_status": "completed",
                "numerical": {"converged": True}}

    monkeypatch.setattr(canary, "execute_case", fake_execute)
    result = canary.run_selected_cases(
        protocol, ("cpu",), ("sp", "opt.lbfgs"), tmp_path, {}, object()
    )
    assert calls == ["sp"]
    assert list(result["cases"]) == ["cpu/sp"]
    assert result["source_unchanged"] is False
    assert result["stop_reason"] == "source_changed"


def test_output_directory_must_be_new(tmp_path):
    with pytest.raises(FileExistsError):
        canary.create_output_directory(tmp_path)


def test_cpu_replay_and_cuda_parity_use_preregistered_tolerances():
    protocol = _protocol()
    reference = protocol["cpu_prechange_reference"]
    metrics = {
        "energy_eV": reference["energy_eV"],
        "forces_eV_per_A": reference["forces_eV_per_A"],
    }
    records = {
        device + "/sp": {"numerical": {"final_metrics": metrics}}
        for device in ("cpu", "cuda:0")
    }
    result = canary.measurement_comparisons(
        protocol, records, ("cpu", "cuda:0"), ("sp",)
    )
    assert result["required_checks_pass"] is True
    assert result["cpu_replay"]["energy_eV"]["pass"] is True
    assert result["cpu_cuda_parity"]["cuda:0/forces_eV_per_A"]["pass"] is True


def test_cuda_only_shard_does_not_require_an_absent_cpu_comparison():
    result = canary.measurement_comparisons(
        _protocol(), {}, ("cuda:0",), ("sp",)
    )
    assert result["required_checks_pass"] is True


def test_real_timeout_preserves_text_output_without_duplication(tmp_path):
    import os
    import sys
    from tools.route2_release.run_pure_mace_polar_nonmd_canary import _run_subprocess
    result = _run_subprocess(
        [sys.executable, '-u', '-c', "import sys,time; print('BEFORE_TIMEOUT'); print('ERROR_TRACE', file=sys.stderr); time.sleep(10)"],
        cwd=tmp_path, env=os.environ, timeout=0.3,
    )
    assert result.timed_out
    assert result.stdout.count('BEFORE_TIMEOUT') == 1
    assert result.stderr.count('ERROR_TRACE') == 1
