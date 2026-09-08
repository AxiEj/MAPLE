"""Small, independent checks of scientific formulas used by MAPLE."""

import math

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.frequency.frequency import (
    AMU,
    ANG2_TO_M2,
    CM_TO_M,
    K_B,
    R_GAS,
    C,
    H,
    MWFrequency,
)
from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat
from maple.function.dispatcher.md.utils import (
    EV_PER_ANG3_TO_BAR,
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
)


def _harmonic_entropy(wavenumber_cm1: float, temperature: float) -> float:
    frequency_hz = wavenumber_cm1 * C / CM_TO_M
    x = H * frequency_hz / (K_B * temperature)
    return R_GAS * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def _grimme_qrrho_entropy(wavenumber_cm1: float, temperature: float) -> float:
    """Independent transcription of the Grimme/GoodVibes interpolation."""
    frequency_hz = wavenumber_cm1 * C / CM_TO_M
    mu = H / (8.0 * math.pi**2 * frequency_hz)
    b_av = 1.0e-44
    mu_reduced = mu * b_av / (mu + b_av)
    rotor = R_GAS * (
        0.5
        + math.log(
            math.sqrt(8.0 * math.pi**3 * mu_reduced * K_B * temperature / H**2)
        )
    )
    weight = 1.0 / (1.0 + (100.0 / wavenumber_cm1) ** 4)
    harmonic = _harmonic_entropy(wavenumber_cm1, temperature)
    return weight * harmonic + (1.0 - weight) * rotor


@pytest.mark.parametrize("wavenumber_cm1", [10.0, 50.0, 100.0, 500.0])
def test_grimme_entropy_matches_independent_qrrho_formula(
    tmp_path, wavenumber_cm1: float
) -> None:
    temperature = 298.15
    job = MWFrequency(str(tmp_path / "frequency.out"), Atoms("H"), ilowfreq=2)
    actual = job._grimme_entropy(
        wavenumber_cm1,
        temperature,
        _harmonic_entropy(wavenumber_cm1, temperature),
    )

    assert actual == pytest.approx(
        _grimme_qrrho_entropy(wavenumber_cm1, temperature), rel=2.0e-14
    )


def test_grimme_average_inertia_has_the_documented_si_value(tmp_path) -> None:
    job = MWFrequency(str(tmp_path / "frequency.out"), Atoms("H"), ilowfreq=2)
    inertia_si = job._grimme_bav_amuA2() * AMU * ANG2_TO_M2

    assert inertia_si == pytest.approx(1.0e-44, rel=2.0e-15)


def test_qeq_charges_match_an_independent_gaussian_radius_solve() -> None:
    torch = pytest.importorskip("torch")
    from maple.function.calculator.extra_correction.charge.qeq import (
        COULOMB_EV_ANGSTROM,
        QEqTorch,
    )

    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    atoms.info["charge"] = 1
    qeq = QEqTorch(device="cpu")
    actual = qeq(atoms).numpy()

    chi, hardness, radii = qeq._get_param_tensor(
        atoms.get_chemical_symbols(), dtype=torch.float64
    )
    positions = atoms.get_positions()
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    p = 1.0 / np.sqrt(radii.numpy()[:, None] ** 2 + radii.numpy()[None, :] ** 2)
    screened = np.zeros_like(distances)
    nonzero = distances > 0.0
    screened[nonzero] = np.vectorize(math.erf)(p[nonzero] * distances[nonzero]) / distances[nonzero]

    n_atoms = len(atoms)
    system = np.zeros((n_atoms + 1, n_atoms + 1))
    system[:n_atoms, :n_atoms] = np.diag(hardness.numpy())
    system[:n_atoms, :n_atoms] += COULOMB_EV_ANGSTROM * screened
    system[n_atoms, :n_atoms] = 1.0
    system[:n_atoms, n_atoms] = 1.0
    rhs = np.r_[-chi.numpy(), float(atoms.info["charge"])]
    expected = np.linalg.solve(system, rhs)[:-1]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-12)
    assert actual.sum() == pytest.approx(1.0, abs=1.0e-12)


def test_crescale_noise_prefactor_converts_inverse_bar_to_angstrom3_per_ev() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    temperature = 300.0
    compressibility = 4.5e-5
    timestep = 2.0
    tau_p = 1000.0
    barostat = CRescaleBarostat(
        atoms,
        pressure=1.0,
        temperature=temperature,
        tau_p=tau_p,
        timestep=timestep,
        compressibility=compressibility,
        rng=np.random.default_rng(0),
    )
    k_t_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV
    expected = math.sqrt(
        2.0
        * k_t_ev
        * (compressibility * EV_PER_ANG3_TO_BAR)
        * timestep
        / tau_p
    )

    assert barostat._noise_prefactor == pytest.approx(expected, rel=2.0e-15)
