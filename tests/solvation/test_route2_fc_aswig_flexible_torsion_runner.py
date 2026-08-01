"""Pure geometry and admission tests for the flexible FC-aSWIG torsion gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _runner_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/implicit-solvation/benchmarks/"
        "run_route2_fc_aswig_d2_force_flexible_torsion.py"
    )
    specification = importlib.util.spec_from_file_location(
        "route2_fc_aswig_d2_force_flexible_torsion_runner", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _positions() -> np.ndarray:
    """A nondegenerate mock with all required 20 atom positions."""

    positions = np.zeros((20, 3), dtype=float)
    positions[4] = (0.0, 0.0, 0.0)
    positions[5] = (1.0, 0.0, 0.0)
    for index in range(20):
        if index not in {4, 5}:
            positions[index] = (float(index) * 0.1, 0.4 + 0.03 * index, -0.2)
    return positions


def _sample(angle: float, *, error: float = 0.0) -> dict[str, object]:
    return {
        "angle_degrees": angle,
        "surface_size": 1720,
        "scf_reason": "nominal-density-and-energy-v1",
        "local_multistart_agreed": True,
        "force_sum_hartree_per_angstrom": [1.0e-13, 0.0, 0.0],
        "analytic_generalized_force_hartree_per_radian": 1.0,
        "total_energy_hartree": -2.0,
        "synthetic_error": error,
    }


def test_rigid_torsion_tangent_agrees_with_centered_coordinate_difference():
    runner = _runner_module()
    positions = _positions()
    step = 1.0e-6
    plus = runner.rotate_fragment(positions, angle_radians=step)
    minus = runner.rotate_fragment(positions, angle_radians=-step)
    numeric = (plus - minus) / (2.0 * step)
    analytic = runner.torsional_tangent(positions)
    assert np.max(np.abs(numeric - analytic)) < 1.0e-10
    assert np.allclose(plus[runner.AXIS_ORIGIN_INDEX], positions[runner.AXIS_ORIGIN_INDEX])
    assert np.allclose(plus[runner.AXIS_DIRECTION_INDEX], positions[runner.AXIS_DIRECTION_INDEX])


def test_gate_accepts_second_order_torsion_error_and_rejects_bad_trend():
    runner = _runner_module()
    samples = {angle: _sample(angle) for angle in runner.ANGLES_DEGREES}
    central = [
        {"absolute_error_ev_per_radian": 8.0e-5},
        {"absolute_error_ev_per_radian": 2.0e-5},
        {"absolute_error_ev_per_radian": 5.0e-6},
    ]
    path = [{"absolute_error_ev_per_radian": 1.0e-5}]
    accepted = runner._gate(samples, central, path)
    assert accepted["passed"]

    rejected = runner._gate(
        samples,
        [
            {"absolute_error_ev_per_radian": 5.0e-6},
            {"absolute_error_ev_per_radian": 5.0e-5},
            {"absolute_error_ev_per_radian": 5.0e-6},
        ],
        path,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["second_order_finite_difference_trend"]
