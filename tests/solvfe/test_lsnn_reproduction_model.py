from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "docs" / "implicit-solvation" / "benchmarks" / "lsnn_pilot"


def _load_model_module():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "maple_lsnn_reproduction_model", PILOT / "lsnn_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module


def test_pilot_manifest_is_label_free():
    panel = json.loads((PILOT / "panel.json").read_text(encoding="utf-8"))
    for molecule in panel["molecules"]:
        assert "experimental_kcal_mol" not in molecule
        assert "experimental_uncertainty_kcal_mol" not in molecule


def test_reconstructed_energy_is_scriptable_and_zero_when_decoupled():
    torch, module = _load_model_module()
    features = torch.tensor(
        [
            [-0.1087, 0.1504859, 0.15929745, 0.733756, 0.506378, 0.205844, 1],
            [0.0272, 0.1004859, 0.14328807, 0.78844, 0.798699, 0.437334, 0],
            [0.0272, 0.1004859, 0.14328807, 0.78844, 0.798699, 0.437334, 0],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0554, 0.0800, 0.0496], [0.0683, -0.0813, -0.0254]],
        dtype=torch.float32,
    )
    model = module.LSNNV1Energy(features).eval()
    scripted = torch.jit.script(model)

    decoupled = scripted(positions, torch.tensor(0.0), torch.tensor(0.0))
    assert decoupled.item() == pytest.approx(0.0, abs=1.0e-8)

    differentiable_positions = positions.clone().requires_grad_(True)
    coupled = model(
        differentiable_positions,
        torch.tensor(1.0),
        torch.tensor(1.0),
    )
    gradient = torch.autograd.grad(coupled, differentiable_positions)[0]
    assert torch.isfinite(coupled)
    assert torch.isfinite(gradient).all()
