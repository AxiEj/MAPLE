from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import cgroup_probe
from cgroup_probe import (
    HARD_MEMORY_BYTES,
    _artifact_paths,
    _cleanup_owned_unit,
    _critical_event_deltas,
    _ensure_new_artifacts,
    _launch,
    _owned_unit_state,
    _validate_script,
    _worker,
    run_cgroup_command_for_testing,
)


def test_probe_allowlist_and_no_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        _validate_script(
            "../../tools/route2_release/run_ddpcm_response_benchmark.py", []
        )
    with pytest.raises(ValueError, match="include --output"):
        _validate_script("probe_native.py", ["--atoms", "10"])
    output = tmp_path / "receipt.json"
    paths = _artifact_paths(output)
    _ensure_new_artifacts(paths)
    paths["child_stdout"].write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        _ensure_new_artifacts(paths)


@pytest.mark.parametrize("artifact", ["receipt", "worker", "child_stdout"])
def test_child_output_cannot_collide_with_wrapper_artifacts(
    tmp_path: Path, artifact: str
) -> None:
    wrapper = tmp_path / "receipt.json"
    child = _artifact_paths(wrapper)[artifact]
    with pytest.raises(ValueError, match="collides"):
        _validate_script(
            "probe_native.py",
            ["--output", str(child)],
            wrapper_output=wrapper,
        )


def test_artifact_conflict_retains_failed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    paths = _artifact_paths(output)
    paths["systemd_stdout"].write_text("prior", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        _launch(output=output, wall_limit_seconds=30.0, command=["/bin/true"])
    payload = json.loads(output.read_text())
    assert payload["completed"] is False
    assert payload["failure"]["type"] == "FileExistsError"


def test_real_owned_systemd_scope_enforces_memory_and_swap_limits(
    tmp_path: Path,
) -> None:
    receipt = run_cgroup_command_for_testing(
        ["/bin/true"], tmp_path / "scope-receipt.json", wall_limit_seconds=30.0
    )
    assert receipt["completed"] is True
    worker = receipt["worker"]
    assert worker["hard_limits"]["memory_max_bytes"] == HARD_MEMORY_BYTES
    assert worker["hard_limits"]["memory_swap_max_bytes"] == 0
    assert worker["kernel_memory_peak_bytes"] > 0
    assert worker["kernel_memory_peak_bytes"] <= HARD_MEMORY_BYTES
    assert worker["kernel_memory_peak_within_limit"] is True
    assert worker["sampled_supervisor"]["is_hard_rss_enforcement"] is False
    assert ".scope" in worker["cgroup_membership"]
    assert worker["kernel_memory_events_clean"] is True
    assert worker["kernel_critical_event_deltas"] == {
        "max": 0,
        "oom": 0,
        "oom_kill": 0,
        "oom_group_kill": 0,
    }


def test_child_exit_zero_with_oom_events_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {"max": 2, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    after = {"max": 3, "oom": 1, "oom_kill": 1, "oom_group_kill": 0}
    assert _critical_event_deltas(before, after) == {
        "max": 1,
        "oom": 1,
        "oom_kill": 1,
        "oom_group_kill": 0,
    }
    monkeypatch.setattr(cgroup_probe, "_owned_cgroup", lambda: (tmp_path, "/x.scope"))
    monkeypatch.setattr(
        cgroup_probe,
        "_validate_hard_limits",
        lambda _path: {
            "memory_max_bytes": HARD_MEMORY_BYTES,
            "memory_swap_max_bytes": 0,
            "memory_events_before": before,
            "memory_current_before_bytes": 1,
        },
    )
    monkeypatch.setattr(cgroup_probe, "_read_events", lambda _path: after)
    monkeypatch.setattr(cgroup_probe, "_read_scalar", lambda _path: 1)
    monkeypatch.setattr(
        cgroup_probe,
        "run_guarded",
        lambda *_args, **_kwargs: {"exit_status": 0, "termination_reason": None},
    )
    worker_output = tmp_path / "worker.json"
    exit_status = _worker(
        worker_output=worker_output,
        child_stdout=tmp_path / "stdout",
        child_stderr=tmp_path / "stderr",
        wall_limit_seconds=10.0,
        command=["/bin/true"],
    )
    payload = json.loads(worker_output.read_text())
    assert exit_status == 1
    assert payload["completed"] is False
    assert payload["failure"]["type"] == "KernelMemoryEventIncrease"


def test_child_exit_zero_with_peak_over_memory_max_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = {"max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    monkeypatch.setattr(cgroup_probe, "_owned_cgroup", lambda: (tmp_path, "/x.scope"))
    monkeypatch.setattr(
        cgroup_probe,
        "_validate_hard_limits",
        lambda _path: {
            "memory_max_bytes": HARD_MEMORY_BYTES,
            "memory_swap_max_bytes": 0,
            "memory_events_before": events,
            "memory_current_before_bytes": 1,
        },
    )
    monkeypatch.setattr(cgroup_probe, "_read_events", lambda _path: events)
    monkeypatch.setattr(
        cgroup_probe, "_read_scalar", lambda _path: HARD_MEMORY_BYTES + 1
    )
    monkeypatch.setattr(
        cgroup_probe,
        "run_guarded",
        lambda *_args, **_kwargs: {"exit_status": 0, "termination_reason": None},
    )
    worker_output = tmp_path / "worker.json"
    exit_status = _worker(
        worker_output=worker_output,
        child_stdout=tmp_path / "stdout",
        child_stderr=tmp_path / "stderr",
        wall_limit_seconds=10.0,
        command=["/bin/true"],
    )
    payload = json.loads(worker_output.read_text())
    assert exit_status == 1
    assert payload["failure"]["type"] == "KernelMemoryPeakExceeded"


@pytest.mark.parametrize("mode", ["missing", "malformed", "exception", "hash-drift"])
def test_launch_failures_retain_receipt_and_cleanup_exact_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    output = tmp_path / f"{mode}.json"
    calls = []
    monkeypatch.setattr(cgroup_probe, "verify_tracked_freeze", lambda: {"head": "x"})
    hashes = iter(({"a.py": "1"}, {"a.py": "2"})) if mode == "hash-drift" else None
    monkeypatch.setattr(
        cgroup_probe,
        "research_python_hashes",
        (lambda: next(hashes)) if hashes is not None else (lambda: {"a.py": "1"}),
    )
    monkeypatch.setattr(
        cgroup_probe,
        "_cleanup_owned_unit",
        lambda unit: calls.append(unit)
        or {"unit": unit, "verified_inactive_and_empty": True},
    )
    monkeypatch.setattr(
        cgroup_probe,
        "_owned_unit_state",
        lambda _unit: {"verified_inactive_and_empty": True},
    )

    def fake_run_guarded(*_args, **_kwargs):
        if mode == "exception":
            raise RuntimeError("observer failed")
        worker = _artifact_paths(output)["worker"]
        if mode == "malformed":
            worker.write_text("not-json", encoding="utf-8")
        elif mode == "hash-drift":
            worker.write_text(
                json.dumps(
                    {
                        "schema": "matrixfree-owned-cgroup-worker-v1",
                        "scientific_admitted": False,
                        "gate_complete": False,
                        "completed": True,
                        "cgroup_membership": "/fake.scope",
                        "hard_limits": {
                            "memory_max_bytes": HARD_MEMORY_BYTES,
                            "memory_swap_max_bytes": 0,
                        },
                        "kernel_memory_events_clean": True,
                        "kernel_critical_event_deltas": {
                            "max": 0,
                            "oom": 0,
                            "oom_kill": 0,
                            "oom_group_kill": 0,
                        },
                        "kernel_memory_peak_bytes": 1,
                        "kernel_memory_peak_within_limit": True,
                        "sampled_supervisor": {
                            "exit_status": 0,
                            "termination_reason": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
        return {"exit_status": 0, "termination_reason": None}

    monkeypatch.setattr(cgroup_probe, "run_guarded", fake_run_guarded)
    receipt = _launch(output=output, wall_limit_seconds=30.0, command=["/bin/true"])
    assert receipt["completed"] is False
    assert "failure" in receipt
    assert output.is_file()
    assert calls == [receipt["unit"]]
    assert (
        receipt["owned_unit_cleanup_on_failure"]["verified_inactive_and_empty"] is True
    )


def test_cleanup_requires_inactive_and_empty_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cgroup_probe, "_systemctl", lambda command: {"command": command}
    )
    states = iter(
        (
            {"verified_inactive_and_empty": False},
            {"verified_inactive_and_empty": True},
        )
    )
    monkeypatch.setattr(cgroup_probe, "_owned_unit_state", lambda _unit: next(states))
    evidence = _cleanup_owned_unit("maple-mf-cap-test.scope")
    assert evidence["verified_inactive_and_empty"] is True
    assert evidence["scope"] == "deterministic-owned-unit-only"


def test_unit_query_failure_cannot_be_mistaken_for_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_query(command):
        return {
            "command": command,
            "exit_status": 1,
            "stdout": "",
            "stderr": "Failed to connect to bus",
        }

    monkeypatch.setattr(cgroup_probe, "_systemctl", failed_query)
    state = _owned_unit_state("maple-mf-cap-query-failure.scope")
    assert state["queries_valid"] is False
    assert state["verified_inactive_and_empty"] is False


def test_explicit_not_found_unit_is_verified_inactive_and_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_found(command):
        if command[0] == "is-active":
            return {
                "command": command,
                "exit_status": 4,
                "stdout": "unknown\n",
                "stderr": "",
            }
        return {
            "command": command,
            "exit_status": 0,
            "stdout": "LoadState=not-found\nActiveState=inactive\nControlGroup=\n",
            "stderr": "",
        }

    monkeypatch.setattr(cgroup_probe, "_systemctl", not_found)
    state = _owned_unit_state("maple-mf-cap-not-found.scope")
    assert state["explicit_not_found"] is True
    assert state["verified_inactive_and_empty"] is True


def test_real_orphan_descendant_invalidates_success_and_is_cleaned(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-c",
        "import subprocess; subprocess.Popen(['/bin/sleep', '3'])",
    ]
    receipt = run_cgroup_command_for_testing(
        command, tmp_path / "orphan-receipt.json", wall_limit_seconds=30.0
    )
    assert receipt["completed"] is False
    assert (
        receipt["owned_unit_state_after_worker"]["verified_inactive_and_empty"] is False
    )
    cleanup = receipt["owned_unit_cleanup_on_failure"]
    assert cleanup["unit"] == receipt["unit"]
    assert cleanup["verified_inactive_and_empty"] is True
