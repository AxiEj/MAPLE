from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _runner_module():
    path = (
        Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_atomic_displacement_source_acetone.py"
    )
    specification = importlib.util.spec_from_file_location(
        "route2_v0_atomic_displacement_source_runner", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _qm_responses() -> dict[float, dict[str, dict[str, np.ndarray]]]:
    return {
        3.0e-4: {
            "x": {
                "potential": np.asarray([1.0, 2.0]),
                "dipole": np.asarray([1.0, 0.0, 0.0]),
            },
            "y": {
                "potential": np.asarray([3.0, 4.0]),
                "dipole": np.asarray([0.0, 2.0, 0.0]),
            },
            "z": {
                "potential": np.asarray([5.0, 6.0]),
                "dipole": np.asarray([0.0, 0.0, 3.0]),
            },
        },
        1.0e-3: {
            "x": {
                "potential": np.asarray([1.1, 2.2]),
                "dipole": np.asarray([1.1, 0.0, 0.0]),
            },
            "y": {
                "potential": np.asarray([3.3, 4.4]),
                "dipole": np.asarray([0.0, 2.2, 0.0]),
            },
            "z": {
                "potential": np.asarray([5.5, 6.6]),
                "dipole": np.asarray([0.0, 0.0, 3.3]),
            },
        },
    }


def test_response_tensor_uses_the_frozen_qm_arrays():
    runner = _runner_module()

    tensor = runner._response_tensor(_qm_responses(), step=3.0e-4, observable="dipole")

    np.testing.assert_allclose(tensor, np.diag([1.0, 2.0, 3.0]))


def test_response_tensor_rejects_an_unknown_observable():
    runner = _runner_module()

    with pytest.raises(ValueError, match="Unsupported"):
        runner._response_tensor(_qm_responses(), step=3.0e-4, observable="candidate")


def test_candidate_source_checks_require_exact_atomic_moment_partition():
    runner = _runner_module()
    weights = np.stack((0.25 * np.eye(3), 0.75 * np.eye(3)))
    polarizability = np.diag([1.0, 2.0, 3.0])

    checks = runner._candidate_source_checks(
        weights=weights,
        polarizability=polarizability,
    )

    assert all(check["passes"] for check in checks.values())
    broken = runner._candidate_source_checks(
        weights=0.9 * weights,
        polarizability=polarizability,
    )
    assert not broken["atomic_partition_moment_identity"]["passes"]
    assert not broken["atomic_partition_response_identity"]["passes"]
