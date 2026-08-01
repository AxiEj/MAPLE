"""Pure protocol gates for the fixed-topology Route-2 NVE benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _runner_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/implicit-solvation/benchmarks/"
        "run_route2_fc_aswig_d2_force_nve.py"
    )
    specification = importlib.util.spec_from_file_location(
        "route2_fc_aswig_d2_force_nve_runner", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _trajectory(
    timestep_fs: float,
    drift_ev: float,
    *,
    reversal: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "timestep_fs": timestep_fs,
        "maximum_absolute_total_energy_drift_ev": drift_ev,
        "physical_time_fs": 0.20,
        "surface_cardinalities": [860],
        "all_nominal_roots": True,
        "all_local_multistart_agreement": True,
        "maximum_net_force_hartree_per_angstrom": 1.0e-13,
    }
    if reversal:
        result["velocity_reversal"] = {
            "position_error_angstrom": 1.0e-10,
            "velocity_error_au": 1.0e-11,
            "total_energy_error_ev": 1.0e-9,
        }
    return result


def test_nve_gate_accepts_same_duration_second_order_trend():
    runner = _runner_module()
    result = runner._nve_gate(
        [
            _trajectory(0.10, 1.0e-6),
            _trajectory(0.05, 2.0e-7, reversal=True),
            _trajectory(0.025, 3.0e-8),
        ]
    )
    assert result["passed"]
    assert result["checks"]["second_order_drift_trend"]
    assert result["surface_cardinalities"] == [860]


def test_nve_gate_rejects_nonconvergent_timestep_trend():
    runner = _runner_module()
    result = runner._nve_gate(
        [
            _trajectory(0.10, 1.0e-6),
            _trajectory(0.05, 9.0e-7, reversal=True),
            _trajectory(0.025, 8.0e-7),
        ]
    )
    assert not result["passed"]
    assert not result["checks"]["second_order_drift_trend"]


def test_research_bridge_cannot_be_publicly_constructed():
    runner = _runner_module()
    with pytest.raises(PermissionError, match="internal benchmark bridge"):
        runner._Route2DirectPCMResearchNVECalculator(None, {})


def test_nve_gate_treats_sub_noise_drift_as_unresolved_not_nonconvergent():
    runner = _runner_module()
    result = runner._nve_gate(
        [
            _trajectory(0.10, 1.0e-10),
            _trajectory(0.05, 2.0e-10, reversal=True),
            _trajectory(0.025, 3.0e-10),
        ]
    )
    assert result["passed"]
    assert result["checks"]["second_order_drift_trend"]
