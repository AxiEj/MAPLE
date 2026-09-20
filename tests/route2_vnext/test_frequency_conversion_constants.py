from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.dispatcher.frequency.frequency import (
    AMU,
    ANG2_TO_M2,
    C,
    CM_TO_M,
    ELECTRON_VOLT_J,
    EV_PER_ANGSTROM2_AMU_TO_CM1,
    H,
    HARTREE_J,
    HARTREE_PER_ANGSTROM2_AMU_TO_CM1,
    HARTREE_TO_EV,
    KJ_PER_MOL_TO_KCAL,
    MWFrequency,
    NA,
    PureFrozenMWFrequency,
)


class _UnitMassAtoms:
    def __len__(self) -> int:
        return 1

    def get_masses(self) -> np.ndarray:
        return np.ones(1)


def _legacy_frequencies(eigenvalues_hartree: np.ndarray) -> np.ndarray:
    job = MWFrequency.__new__(MWFrequency)
    job.atoms = _UnitMassAtoms()
    job.verbosity = 0
    job._project_hessian = lambda matrix: np.asarray(matrix)
    job._eigh = lambda matrix: np.linalg.eigh(np.asarray(matrix))
    frequencies, _ = job.compute_frequencies(
        np.diag(np.asarray(eigenvalues_hartree, dtype=float))
    )
    return frequencies


def _pure_frequencies(eigenvalues_ev: np.ndarray) -> np.ndarray:
    job = PureFrozenMWFrequency.__new__(PureFrozenMWFrequency)
    job.mode_analysis = SimpleNamespace(
        eigenvalues_eV_per_A2_amu=np.asarray(eigenvalues_ev, dtype=float),
        modes_cartesian=np.eye(len(eigenvalues_ev)),
    )
    frequencies, _ = job.compute_frequencies(np.empty((0, 0)))
    return frequencies


def test_conversion_constants_follow_the_si_wavenumber_definition() -> None:
    expected_hartree = np.sqrt(HARTREE_J / (AMU * ANG2_TO_M2)) / (
        2.0 * np.pi * C * 100.0
    )
    expected_ev = np.sqrt(ELECTRON_VOLT_J / (AMU * ANG2_TO_M2)) / (
        2.0 * np.pi * C * 100.0
    )

    assert HARTREE_PER_ANGSTROM2_AMU_TO_CM1 == pytest.approx(
        expected_hartree,
        rel=1.0e-15,
    )
    assert EV_PER_ANGSTROM2_AMU_TO_CM1 == pytest.approx(
        expected_ev,
        rel=1.0e-15,
    )
    assert HARTREE_PER_ANGSTROM2_AMU_TO_CM1 == pytest.approx(
        2720.228649394269,
        rel=1.0e-15,
    )
    assert EV_PER_ANGSTROM2_AMU_TO_CM1 == pytest.approx(
        521.4708983725066,
        rel=1.0e-15,
    )


def test_legacy_and_pure_paths_agree_for_positive_negative_and_zero_curvature() -> None:
    eigenvalues_hartree = np.asarray([-1.0, 0.0, 4.0])
    legacy = _legacy_frequencies(eigenvalues_hartree)
    pure = _pure_frequencies(eigenvalues_hartree * HARTREE_TO_EV)

    # The legacy sorter reports zero, imaginary, then real modes.  The pure
    # analyzer preserves the analyzer's internal-mode order.
    np.testing.assert_allclose(
        legacy,
        [
            0.0,
            -HARTREE_PER_ANGSTROM2_AMU_TO_CM1,
            2.0 * HARTREE_PER_ANGSTROM2_AMU_TO_CM1,
        ],
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        pure,
        [
            -HARTREE_PER_ANGSTROM2_AMU_TO_CM1,
            0.0,
            2.0 * HARTREE_PER_ANGSTROM2_AMU_TO_CM1,
        ],
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sort(legacy),
        np.sort(pure),
        rtol=1.0e-15,
        atol=0.0,
    )


def test_corrected_frequency_is_propagated_to_zero_point_energy() -> None:
    job = MWFrequency.__new__(MWFrequency)
    job.atoms = _UnitMassAtoms()
    job.temperature = 298.15
    job.nu_floor_cm1 = 1.0
    job.ilowfreq = 0
    job._is_linear_molecule = lambda: True
    job._trans_rot_entropy = lambda: (0.0, 0.0)

    frequency = HARTREE_PER_ANGSTROM2_AMU_TO_CM1
    thermo = job.compute_thermo(np.asarray([-frequency, 0.0, frequency]))
    expected_zpe_kjmol = 0.5 * H * C * NA * frequency / CM_TO_M * 1.0e-3

    assert thermo.zpe_kjmol == pytest.approx(expected_zpe_kjmol, rel=1.0e-15)


def test_thermochemistry_output_preserves_gibbs_correction_units() -> None:
    job = MWFrequency.__new__(MWFrequency)
    job.atoms = _UnitMassAtoms()
    job.temperature = 298.15
    job.pressure_kpa = 101.325
    job.nu_floor_cm1 = 1.0
    job.ilowfreq = 0
    job._is_linear_molecule = lambda: True
    job._trans_rot_entropy = lambda: (125.0, 45.0)

    thermo = job.compute_thermo(np.asarray([100.0, 500.0, 1500.0]))
    expected_g_kjmol = (
        thermo.h_total_kjmol - job.temperature * thermo.s_total_jmolK * 1.0e-3
    )
    assert thermo.g_correction_kjmol == pytest.approx(expected_g_kjmol)

    output: list[str] = []
    job.log_info = lambda messages: output.extend(messages)
    job._write_thermochemistry(thermo)
    report = "".join(output)

    def reported_value(label: str) -> float:
        line = next(line for line in report.splitlines() if line.startswith(label))
        return float(line.split("...", maxsplit=1)[1].split()[0])

    reported_h = reported_value("Total enthalpy correction")
    reported_minus_ts = reported_value("Total entropy correction")
    reported_g = reported_value("Final Gibbs free energy corr.")

    expected_minus_ts_kcal = (
        -job.temperature * thermo.s_total_jmolK * 1.0e-3 * KJ_PER_MOL_TO_KCAL
    )
    assert reported_minus_ts == pytest.approx(expected_minus_ts_kcal, abs=0.0051)
    # Each displayed contribution is independently rounded to 0.01 kcal/mol.
    assert reported_h + reported_minus_ts == pytest.approx(
        reported_g,
        abs=0.0151,
    )
