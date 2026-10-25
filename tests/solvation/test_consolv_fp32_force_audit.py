"""FP32 forces use an independently named higher-precision FD audit."""

import os

import pytest
import torch
from ase.build import molecule

from maple.function.calculator.consolv._torch_model import load_consolv_torch_checkpoint
from maple.function.calculator.consolv._consolv_calculator import (
    CONSOLV_DEFAULT_MODEL_PATH,
)


@pytest.mark.skipif(
    os.environ.get("MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS") != "1",
    reason="explicit local checkpoint opt-in required",
)
def test_pure_fp32_force_vs_cpu64_fd_preserves_frozen_h_and_tolerance():
    reference = load_consolv_torch_checkpoint(
        CONSOLV_DEFAULT_MODEL_PATH, device="cpu", dtype=torch.float64
    )
    candidate = load_consolv_torch_checkpoint(
        CONSOLV_DEFAULT_MODEL_PATH, device="cpu", dtype=torch.float32
    )
    atoms = molecule("C6H6")
    numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
    positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
    count = 0
    for solvent in (19, 45):
        ref_energy, ref_forces = reference.energy_and_forces_ev(
            numbers, positions, solvent
        )
        energy, forces = candidate.energy_and_forces_ev(
            numbers, positions.float(), solvent
        )
        assert energy.dtype == forces.dtype == torch.float32
        assert all(
            not b.is_floating_point() or b.dtype == torch.float32
            for b in candidate.buffers()
        )
        assert torch.max(torch.abs(forces.double() - ref_forces)) <= 1e-4
        for atom in range(len(atoms)):
            for component in range(3):
                plus, minus = positions.clone(), positions.clone()
                plus[atom, component] += 0.002
                minus[atom, component] -= 0.002
                fd = (
                    -(
                        reference.energy_ev(numbers, plus, solvent)
                        - reference.energy_ev(numbers, minus, solvent)
                    )
                    / 0.004
                )
                assert abs(float(fd) - float(ref_forces[atom, component])) <= 0.002
                assert abs(float(fd) - float(forces[atom, component])) <= 0.002
                count += 1
    assert count == 72
