from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "route2_release" / "run_mace_conjugacy_nogo.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("route2_mace_nogo_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_retains_failed_preregistered_fd_gate_as_negative_evidence():
    runner = _load_runner()
    record = runner._error_record(0.054460477272, 0.054459986)

    assert record["gate_passed"] is False
    assert record["absolute_error_eV"] > 2.0e-7
    assert record["relative_error"] > 2.0e-6


def test_post_preregistration_ad_crosscheck_is_explicit_and_not_an_fd_relabel():
    runner = _load_runner()
    record = runner._energy_ad_record(0.05446047727235914, 0.054460477278798235)

    assert record["implementation_consistent"] is True
    assert record["absolute_error_eV"] == pytest.approx(6.4390923149e-12)
    assert record["status"] == "post-preregistration-implementation-diagnostic"
    assert runner.POST_PREREGISTRATION_AD_ABSOLUTE_TOLERANCE_EV == 2.0e-10
    assert runner.POST_PREREGISTRATION_AD_RELATIVE_TOLERANCE == 5.0e-9
    assert runner.SCHEMA_VERSION.endswith("-v2")
    assert runner.FD_STEPS == (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
