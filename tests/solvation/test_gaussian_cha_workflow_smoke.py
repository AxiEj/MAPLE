"""Real-checkpoint development smoke, not full FREQ/TS qualification.

Private, pinned inputs are optional for a fresh clone. These tests never
download inputs and never claim that a water minimum is a transition state.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.parametrize("case_index", range(15))
def test_real_water_analytic_frequency_and_ts_interfaces(
    tmp_path, monkeypatch, case_index
):
    root = Path(__file__).resolve().parents[2]
    script = root / "docs/implicit-solvation/benchmarks/run_cha_gaussian_freq_ts.py"
    spec = importlib.util.spec_from_file_location("cha_real_workflow_smoke", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol, protocol_hash = runner.load_protocol()
    checkpoint = root / "maple/function/calculator/model/maceoff23m.pt"
    topology_path = runner.ROOT / protocol["topology_path"]
    if not checkpoint.is_file() or not topology_path.is_file():
        pytest.skip("pinned private water/checkpoint inputs are not installed")
    _, topology = runner.load_topology(protocol)
    case = protocol["cases"][case_index]
    from maple.function.calculator import calculator_base
    from maple.function.calculator.mace._mace_cha_analytic_calculator import (
        MACEChaAnalyticCalculator,
    )
    from maple.function.calculator.extra_correction.implicit.gaussian_cha_analytic_correction import (
        GaussianChaAnalyticCorrection,
    )
    from maple.function.dispatcher.ts.algorithm.dimer import Dimer

    def forbidden(*args, **kwargs):
        pytest.fail("runtime FD or dense Hessian was reached from a direct-HVP path")

    monkeypatch.setattr(calculator_base, "numerical_hessian_from_atoms", forbidden)
    monkeypatch.setattr(Dimer, "_finite_difference_hn", forbidden)
    result = {
        "status": "DEVELOPMENT_SMOKE_ONLY_NOT_QUALIFIED",
        "case_identity": case,
        "protocol_sha256": protocol_hash,
        "runner_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "runtime_sources_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                root
                / "maple/function/calculator/mace/_mace_cha_analytic_calculator.py",
                root
                / "maple/function/calculator/extra_correction/implicit/gaussian_cha_analytic_correction.py",
                root / "maple/function/dispatcher/ts/algorithm/PRFO.py",
                root / "maple/function/dispatcher/ts/algorithm/dimer.py",
                root / "maple/function/dispatcher/frequency/frequency.py",
            )
        },
        "true_ts_claim": False,
    }
    try:
        result["frequency"] = runner.run_frequency_workflow(
            case, topology, tmp_path, protocol
        )
        frequencies = np.asarray(result["frequency"]["frequencies_cm1"])
        assert frequencies.shape == (9,)
        assert np.isfinite(frequencies).all()
        assert np.count_nonzero(np.abs(frequencies) <= 1.0) == 6
        assert np.count_nonzero(frequencies > 10.0) == 3
        result["prfo"] = runner.run_prfo_workflow(case, topology, tmp_path, protocol)
        assert result["prfo"]["initial_material_negative_count"] == 0
        assert not result["prfo"]["raw_algorithm_converged"]
        with (
            patch.object(MACEChaAnalyticCalculator, "get_hessian", forbidden),
            patch.object(MACEChaAnalyticCalculator, "_analytic_hessian", forbidden),
            patch.object(GaussianChaAnalyticCorrection, "get_hessian", forbidden),
            patch.object(calculator_base, "hessian_via_double_autograd", forbidden),
        ):
            result["dimer"] = runner.run_dimer_workflow(
                case, topology, tmp_path, protocol
            )
        assert result["dimer"]["derivative_mode"] == "hvp"
        assert not result["dimer"]["raw_algorithm_converged"]
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (tmp_path / "development-smoke.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
