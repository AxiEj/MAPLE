"""Controlled statistical checks for C-rescale, not production NPT admission."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat
from maple.function.dispatcher.md.utils import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
)


class ZeroConfigurationalPressure(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def _ideal_gas_samples(
    *, timestep: float, steps: int, burn_in: int, stride: int, seed: int
) -> np.ndarray:
    n_atoms = 2
    temperature = 300.0
    pressure = 5000.0
    k_t_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    scale = k_t_ev * EV_PER_ANG3_TO_BAR / pressure
    initial_volume = (n_atoms + 1) * scale
    cell_length = initial_volume ** (1.0 / 3.0)
    atoms = Atoms(
        "H2",
        positions=np.zeros((n_atoms, 3)),
        cell=np.eye(3) * cell_length,
        pbc=True,
        calculator=ZeroConfigurationalPressure(),
    )
    barostat = CRescaleBarostat(
        atoms,
        pressure=pressure,
        temperature=temperature,
        tau_p=100.0,
        timestep=timestep,
        compressibility=5.0e-4,
        rng=np.random.default_rng(seed),
        n_dof=3 * n_atoms,
    )
    velocities = np.zeros((n_atoms, 3))
    samples = []
    for step in range(steps):
        barostat.apply(velocities)
        if step >= burn_in and step % stride == 0:
            samples.append(atoms.get_volume())
    return np.asarray(samples)


@pytest.mark.slow
def test_ideal_gas_volume_moments_converge_at_two_timesteps() -> None:
    seed_sequence = np.random.SeedSequence(0)
    seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(4)]
    coarse = np.concatenate(
        [
            _ideal_gas_samples(
                timestep=2.0, steps=10_000, burn_in=2_000, stride=10, seed=seed
            )
            for seed in seeds
        ]
    )
    fine = np.concatenate(
        [
            _ideal_gas_samples(
                timestep=0.5, steps=30_000, burn_in=5_000, stride=25, seed=seed
            )
            for seed in seeds
        ]
    )

    k_t_ev = 300.0 * KELVIN_TO_HARTREE * HARTREE_TO_EV
    scale = k_t_ev * EV_PER_ANG3_TO_BAR / 5000.0
    expected_mean = 3.0 * scale
    expected_variance = 3.0 * scale**2

    for samples in (coarse, fine):
        assert np.all(np.isfinite(samples))
        assert np.all(samples > 0.0)
        assert samples.mean() == pytest.approx(expected_mean, rel=0.10)
        assert samples.var(ddof=1) == pytest.approx(expected_variance, rel=0.30)

    assert coarse.mean() == pytest.approx(fine.mean(), rel=0.08)
    assert coarse.var(ddof=1) == pytest.approx(fine.var(ddof=1), rel=0.25)
