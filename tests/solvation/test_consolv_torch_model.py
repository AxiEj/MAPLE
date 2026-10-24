from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from maple.function.calculator.consolv._torch_model import (
    CHECKPOINT_IDENTITIES,
    FORMAT,
    load_consolv_torch_checkpoint,
)

MODEL_DIR = Path(
    os.environ.get(
        "MAPLE_CONSOLV_TEST_MODEL_DIR",
        Path(__file__).resolve().parents[2] / "maple/function/calculator/model/consolv",
    )
)
CHECKPOINT = MODEL_DIR / "consolv-torch-v1.pt"
VACUUM_CHECKPOINT = CHECKPOINT.with_name("consolv-vacuum-torch-v1.pt")
RUN_REAL_ASSETS = os.environ.get("MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS") == "1"
real_asset_test = pytest.mark.skipif(
    not RUN_REAL_ASSETS or not (CHECKPOINT.is_file() and VACUUM_CHECKPOINT.is_file()),
    reason="set MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS=1 with both frozen checkpoints",
)


def _water():
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
        dtype=torch.float64,
    )
    return numbers, positions


@real_asset_test
def test_checkpoint_is_safe_complete_and_jax_free():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    assert payload["format"] == FORMAT
    assert len(payload["parameter_map"]) == 60
    assert payload["metadata"]["parameter_count"] == 768338
    assert payload["unused_source_parameters"] == [
        "mace_consol/layer_0/linear_readout_layer/linear::w[0,0] 64x0e,1x0e"
    ]
    before = {
        name for name in sys.modules if name.startswith(("jax", "haiku", "e3nn_jax"))
    }
    model = load_consolv_torch_checkpoint(CHECKPOINT)
    after = {
        name for name in sys.modules if name.startswith(("jax", "haiku", "e3nn_jax"))
    }
    assert after == before
    assert model.metadata["implementation"] == "independent_pure_torch"


@real_asset_test
def test_paired_vacuum_checkpoint_is_solvent_invariant_and_finite():
    payload = torch.load(
        VACUUM_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
    )
    assert payload["metadata"]["potential_state"] == "vacuum"
    model = load_consolv_torch_checkpoint(VACUUM_CHECKPOINT, expected_state="vacuum")
    numbers, positions = _water()
    energy_a, forces_a = model.energy_and_forces_ev(numbers, positions, 19)
    energy_b, forces_b = model.energy_and_forces_ev(numbers, positions, 45)
    assert torch.isfinite(energy_a)
    assert torch.isfinite(forces_a).all()
    assert torch.equal(energy_a, energy_b)
    assert torch.equal(forces_a, forces_b)


@real_asset_test
def test_two_solvents_are_finite_and_distinct():
    model = load_consolv_torch_checkpoint(CHECKPOINT)
    numbers, positions = _water()
    ethanol, f_ethanol = model.energy_and_forces_ev(numbers, positions, 19)
    toluene, f_toluene = model.energy_and_forces_ev(numbers, positions, 45)
    assert torch.isfinite(ethanol)
    assert torch.isfinite(toluene)
    assert torch.isfinite(f_ethanol).all()
    assert torch.isfinite(f_toluene).all()
    assert ethanol.item() != toluene.item()


@real_asset_test
def test_frozen_conversion_reference_for_two_solvents():
    """Lock the independent Torch port to the reconstructed published graph."""
    model = load_consolv_torch_checkpoint(CHECKPOINT)
    numbers = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    positions = torch.tensor(
        [
            [-0.748, 0.015, 0.024],
            [0.748, -0.015, -0.024],
            [1.424, 1.171, 0.106],
            [-1.129, -0.514, 0.899],
            [-1.104, 1.047, 0.050],
            [-1.136, -0.482, -0.892],
            [1.116, -0.558, -0.895],
            [1.104, -0.604, 0.837],
            [2.367, 0.999, 0.092],
        ],
        dtype=torch.float64,
    )
    references = {
        19: (-34.93867043776235, [-0.62102622, 0.27658415, -0.80794168]),
        45: (-34.87228928881432, [-0.68025142, 0.27134016, -0.81029111]),
    }
    for solvent, (energy_ref, force_ref) in references.items():
        energy, forces = model.energy_and_forces_ev(numbers, positions, solvent)
        assert abs(energy.item() - energy_ref) < 1.0e-5
        assert torch.allclose(
            forces[0], torch.tensor(force_ref, dtype=torch.float64), atol=5.0e-4, rtol=0
        )


@real_asset_test
def test_energy_forces_fd_translation_rotation_and_permutation():
    model = load_consolv_torch_checkpoint(CHECKPOINT)
    numbers, positions = _water()
    energy, forces = model.energy_and_forces_ev(numbers, positions, 19)
    step = 1.0e-4
    moved = positions.clone()
    moved[1, 2] += step
    plus = model.energy_ev(numbers, moved, 19)
    moved[1, 2] -= 2 * step
    minus = model.energy_ev(numbers, moved, 19)
    fd = -(plus - minus) / (2 * step)
    assert torch.allclose(fd, forces[1, 2], atol=1e-5, rtol=1e-7)

    shift = torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64)
    shifted_energy = model.energy_ev(numbers, positions + shift, 19)
    assert torch.allclose(shifted_energy, energy, atol=1e-8, rtol=1e-12)

    angle = 0.61
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    rotated_energy, rotated_forces = model.energy_and_forces_ev(
        numbers, positions @ rotation.T, 19
    )
    assert torch.allclose(rotated_energy, energy, atol=1e-7, rtol=1e-12)
    assert torch.allclose(rotated_forces, forces @ rotation.T, atol=1e-6, rtol=1e-10)

    permutation = torch.tensor([0, 2, 1])
    permuted = model.energy_ev(numbers[permutation], positions[permutation], 19)
    assert torch.allclose(permuted, energy, atol=1e-8, rtol=1e-12)


def test_loader_rejects_unrecognized_synthetic_checkpoint(tmp_path):
    checkpoint = tmp_path / "fake.pt"
    torch.save({"format": FORMAT, "metadata": {}}, checkpoint)
    with pytest.raises(ValueError, match="Unrecognized ConSolv Torch checkpoint"):
        load_consolv_torch_checkpoint(checkpoint)


def test_loader_rejects_unknown_state_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="Unsupported ConSolv potential state"):
        load_consolv_torch_checkpoint(tmp_path / "absent.pt", expected_state="gas")


def test_raw_diagnostic_loader_is_not_exported_from_supported_package():
    from maple.function.calculator import consolv

    assert consolv.__all__ == ["ConSolvCalculator"]
    assert not hasattr(consolv, "load_consolv_torch_checkpoint")


@pytest.mark.parametrize(
    "controls",
    [
        {"device": "cuda"},
        {"device": "mps"},
        {"device": "none"},
        {"dtype": torch.float32},
        {"dtype": torch.float16},
        {"dtype": None},
    ],
)
def test_diagnostic_loader_rejects_runtime_controls_before_file_access(
    monkeypatch, controls
):
    monkeypatch.setattr(Path, "is_file", lambda _: pytest.fail("file access reached"))
    with pytest.raises(ValueError, match="CPU|FP64"):
        load_consolv_torch_checkpoint("not-accessed.pt", **controls)


@pytest.mark.parametrize(
    "actual,requested", [("solution", "vacuum"), ("vacuum", "solution")]
)
def test_loader_rejects_swapped_synthetic_identity_before_deserialization(
    tmp_path, monkeypatch, actual, requested
):
    from maple.function.calculator.consolv import _torch_model

    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"synthetic identity test")
    monkeypatch.setattr(
        _torch_model, "_sha256", lambda _: CHECKPOINT_IDENTITIES[actual]
    )
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("deserialization reached")
    )
    with pytest.raises(ValueError, match=f"is '{actual}', not requested '{requested}'"):
        load_consolv_torch_checkpoint(path, expected_state=requested)


@real_asset_test
def test_solution_and_vacuum_identities_cannot_be_interchanged():
    import hashlib

    assert (
        hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        == CHECKPOINT_IDENTITIES["solution"]
    )
    assert (
        hashlib.sha256(VACUUM_CHECKPOINT.read_bytes()).hexdigest()
        == CHECKPOINT_IDENTITIES["vacuum"]
    )
    with pytest.raises(ValueError, match="is 'vacuum', not requested 'solution'"):
        load_consolv_torch_checkpoint(VACUUM_CHECKPOINT)
    with pytest.raises(ValueError, match="is 'solution', not requested 'vacuum'"):
        load_consolv_torch_checkpoint(CHECKPOINT, expected_state="vacuum")
