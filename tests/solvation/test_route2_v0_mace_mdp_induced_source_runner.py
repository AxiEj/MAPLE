from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _runner_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_mace_mdp_induced_source_acetone.py"
    )
    specification = importlib.util.spec_from_file_location(
        "route2_v0_mace_mdp_induced_source_runner", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _qm_responses() -> dict[float, dict[str, dict[str, np.ndarray]]]:
    result: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for step, factor in ((3.0e-4, 1.0), (1.0e-3, 1.1)):
        result[step] = {
            "x": {
                "potential": np.asarray([factor, 2.0 * factor]),
                "dipole": np.asarray([factor, 0.0, 0.0]),
            },
            "y": {
                "potential": np.asarray([3.0 * factor, 4.0 * factor]),
                "dipole": np.asarray([0.0, 2.0 * factor, 0.0]),
            },
            "z": {
                "potential": np.asarray([5.0 * factor, 6.0 * factor]),
                "dipole": np.asarray([0.0, 0.0, 3.0 * factor]),
            },
        }
    return result


def test_qm_response_tensor_uses_the_qm_dipoles_not_candidate_values():
    runner = _runner_module()
    responses = _qm_responses()

    tensor = runner._qm_response_tensor(
        responses, step=3.0e-4, observable="dipole"
    )

    np.testing.assert_allclose(tensor, np.diag([1.0, 2.0, 3.0]))


def test_qm_response_tensor_rejects_unknown_or_incomplete_observables():
    runner = _runner_module()
    responses = _qm_responses()

    with pytest.raises(ValueError, match="Unsupported"):
        runner._qm_response_tensor(responses, step=3.0e-4, observable="candidate")

    del responses[3.0e-4]["z"]
    with pytest.raises(RuntimeError, match="incomplete"):
        runner._qm_response_tensor(responses, step=3.0e-4, observable="potential")
