"""Lightweight distribution and public-example contracts; no model assets."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from maple.function.read.command_control import CommandControl

ROOT = Path(__file__).resolve().parents[2]


def test_generic_maple_import_does_not_load_optional_model_runtimes():
    code = """
import importlib.abc
import sys

blocked = {'torch', 'jax', 'haiku', 'e3nn', 'e3nn_jax', 'fairchem', 'anysolv'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('generic import loaded optional runtime: ' + fullname)
sys.meta_path.insert(0, Guard())
import maple
import maple.function.calculator.set_calculator
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    ["gas", "anysolv_water", "anysolv_toluene", "consolv_ethanol", "consolv_toluene"],
)
def test_public_examples_parse_without_weights(name):
    content = (ROOT / "examples/pretrained_solvation" / (name + ".inp")).read_text()
    headers = [line for line in content.splitlines() if line.startswith("#")]
    control = CommandControl.from_settings(headers)
    assert control.task == "sp"
    assert control.get("device") == ("cpu" if name == "gas" else "cuda:0")
    if name.startswith("consolv"):
        assert control.get("model") == "consolv"
    else:
        assert control.get("model") == "aimnet2"


def test_runtime_model_cards_and_registries_keep_the_same_frozen_identities():
    cards = ROOT / "maple/function/calculator/model_cards"
    manifest = json.loads(
        (ROOT / "docs/pretrained-solvation-hub/upstream-artifacts.json").read_text()
    )
    artifacts = {item["model_id"]: item for item in manifest["artifacts"]}
    anysolv = json.loads((cards / "anysolv-moe.yaml").read_text())
    consolv = json.loads((cards / "consolv.yaml").read_text())
    assert artifacts["anysolv-moe"]["sha256"] == anysolv["checkpoint_sha256"]
    for model_id in ("consolv-torch-v1", "consolv-vacuum-torch-v1"):
        assert (
            artifacts[model_id]["sha256"]
            == consolv["checkpoints"][model_id]["checkpoint_sha256"]
        )


def test_gpu_source_and_evidence_dependencies_are_package_local():
    import hashlib

    for model in ("anysolv-moe", "consolv", "aimnet2-anysolv"):
        card = json.loads(
            (
                ROOT / "maple/function/calculator/model_cards" / f"{model}.yaml"
            ).read_text()
        )
        sources = card["gpu_acceleration"]["numerical_qualification"][
            "implementation_source_sha256s"
        ]
        assert "maple/function/benchmarking/solvation_gpu_evidence.py" in sources
        assert all(name.startswith("maple/") for name in sources)
        for name, expected in sources.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
