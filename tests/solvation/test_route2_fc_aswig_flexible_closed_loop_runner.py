"""Pure geometry and work-gate tests for the flexible FC-aSWIG loop."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _runner_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/implicit-solvation/benchmarks/"
        "run_route2_fc_aswig_d2_force_flexible_closed_loop.py"
    )
    specification = importlib.util.spec_from_file_location(
        "route2_fc_aswig_d2_force_flexible_closed_loop_runner", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sample(offset: tuple[float, float]) -> dict[str, object]:
    return {
        "offset_degrees": list(offset),
        "offset_radians": list(np.deg2rad(offset)),
        "total_energy_hartree": -10.0,
        "generalized_forces_hartree_per_radian": [0.1, -0.2],
        "surface_size": 1720,
        "scf_reason": "nominal-density-and-energy-v1",
        "local_multistart_agreed": True,
        "force_sum_hartree_per_angstrom": [1.0e-13, 0.0, 0.0],
    }


def test_closed_loop_gate_accepts_zero_work_and_rejects_large_work():
    runner = _runner_module()
    samples = {label: _sample(offset) for label, offset in {
        "A": (-0.3, -0.3), "AB_mid": (0.0, -0.3), "B": (0.3, -0.3),
        "BC_mid": (0.3, 0.0), "C": (0.3, 0.3), "CD_mid": (0.0, 0.3),
        "D": (-0.3, 0.3), "DA_mid": (-0.3, 0.0),
    }.items()}
    passed = runner._gate(samples, {"A_to_B": 0.0, "B_to_C": 0.0, "C_to_D": 0.0, "D_to_A": 0.0})
    assert passed["passed"]

    failed = runner._gate(samples, {"A_to_B": 1.0e-3, "B_to_C": 0.0, "C_to_D": 0.0, "D_to_A": 0.0})
    assert not failed["passed"]
    assert not failed["checks"]["closed_work"]


def test_two_torsion_positions_leave_each_axis_atom_stationary_for_its_rotation():
    runner = _runner_module()
    positions = np.zeros((20, 3), dtype=float)
    positions[4] = (0.0, 0.0, 0.0)
    positions[5] = (1.0, 0.0, 0.0)
    positions[6] = (1.0, 1.0, 0.0)
    for index in range(20):
        if index not in {4, 5, 6}:
            positions[index] = (0.1 * index, 0.3 + 0.01 * index, -0.2)
    rotated = runner.two_torsion_positions(positions, (0.3, -0.2))
    # The first axis endpoints do not move under its own operation; the second
    # terminal rotation cannot move atoms before that terminal fragment.
    assert np.allclose(rotated[4], positions[4])
    assert np.allclose(rotated[5], positions[5])
