from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "run_route2_gto_pcm_energy_projection_canary.py"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "route2_gto_pcm_energy_projection_canary_test",
        RUNNER,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_projection_runner_rejects_a_different_pyscf_interpreter(tmp_path):
    runner = _load_runner()
    fake_interpreter = tmp_path / "python"
    fake_interpreter.write_bytes(b"not the preregistered interpreter")

    with pytest.raises(RuntimeError, match="does not match the preregistration"):
        runner._validate_pyscf_interpreter(
            fake_interpreter,
            {"pyscf_python_resolved_sha256": "0" * 64},
        )
