from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.thermochemistry import IdealGasThermo

from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.frequency import (
    Frequency,
    FrequencyParams,
    MWFrequency,
)
from maple.function.dispatcher.frequency.normal_modes import (
    analyze_cartesian_hessian,
    rigid_body_subspaces,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.read.command_control import CommandControl

WATER_MASSES_AMU = np.array([15.999, 1.008, 1.008])
WATER_POSITIONS_A = np.array(
    [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
)


def _water_with_known_ev_hessian() -> tuple[Atoms, np.ndarray]:
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    vibrational_basis = subspaces.vibrational_basis_mass_weighted
    hessian_mass_weighted = (
        vibrational_basis @ np.diag([1.0, 4.0, 9.0]) @ vibrational_basis.T
    )
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    hessian = (
        square_root_mass[:, None] * hessian_mass_weighted * square_root_mass[None, :]
    )
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=WATER_POSITIONS_A,
        masses=WATER_MASSES_AMU,
    )
    return atoms, hessian


def _ase_rrho_totals(
    atoms: Atoms,
    vibrational_frequencies_cm1: np.ndarray,
    *,
    temperature: float,
    pressure_kpa: float,
    symmetry_number: int,
) -> tuple[float, float, float]:
    if len(atoms) == 1:
        geometry = "monatomic"
    else:
        moments = np.sort(atoms.get_moments_of_inertia())
        geometry = "linear" if moments[0] / moments[-1] < 1.0e-2 else "nonlinear"
    multiplicity = int(atoms.info.get("mult", 1))
    thermo = IdealGasThermo(
        vib_energies=np.asarray(vibrational_frequencies_cm1) * units.invcm,
        geometry=geometry,
        potentialenergy=0.0,
        atoms=atoms,
        symmetrynumber=symmetry_number,
        spin=0.5 * (multiplicity - 1),
    )
    enthalpy_eV = thermo.get_enthalpy(temperature, verbose=False)
    entropy_eV_per_K = thermo.get_entropy(
        temperature,
        pressure_kpa * 1000.0,
        verbose=False,
    )
    gibbs_eV = thermo.get_gibbs_energy(
        temperature,
        pressure_kpa * 1000.0,
        verbose=False,
    )
    eV_to_kJ_per_mol = units.mol / units.kJ
    return (
        enthalpy_eV * eV_to_kJ_per_mol,
        entropy_eV_per_K * eV_to_kJ_per_mol * 1000.0,
        gibbs_eV * eV_to_kJ_per_mol,
    )


def test_public_mw_frequency_reuses_unit_explicit_mass_metric_analysis(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()
    expected = analyze_cartesian_hessian(
        hessian,
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )
    job = MWFrequency(str(tmp_path / "freq.out"), atoms)
    job.verbosity = 0

    frequencies, modes = job.compute_frequencies(hessian)

    assert frequencies.shape == (9,)
    assert modes.shape == (9, 9)
    assert frequencies[:6] == pytest.approx(np.zeros(6), abs=0.0)
    assert frequencies[6:] == pytest.approx(
        expected.frequencies_cm1,
        rel=0.0,
        abs=2.0e-11,
    )
    mass_metric = np.diag(np.repeat(WATER_MASSES_AMU, 3))
    assert modes @ mass_metric @ modes.T == pytest.approx(
        np.eye(9),
        abs=3.0e-14,
    )


def test_public_mw_frequency_rejects_asymmetric_hessian(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()
    hessian[0, 1] += 1.0e-2
    job = MWFrequency(str(tmp_path / "freq.out"), atoms)
    job.verbosity = 0

    with pytest.raises(ValueError, match="not symmetric"):
        job.compute_frequencies(hessian)


def test_public_mw_frequency_accepts_reportable_float32_scale_asymmetry(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()
    expected = analyze_cartesian_hessian(
        hessian,
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )
    hessian[0, 1] += 5.0e-6
    job = MWFrequency(str(tmp_path / "freq.out"), atoms)
    job.verbosity = 0

    frequencies, _ = job.compute_frequencies(hessian)

    assert frequencies[6:] == pytest.approx(
        expected.frequencies_cm1,
        rel=2.0e-6,
    )


def test_public_mw_frequency_rejects_hidden_rigid_body_curvature(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    translation = subspaces.translation_basis_mass_weighted[:, 0]
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    spurious_mass_weighted = 0.01 * np.outer(translation, translation)
    hessian += (
        square_root_mass[:, None] * spurious_mass_weighted * square_root_mass[None, :]
    )
    job = MWFrequency(str(tmp_path / "freq.out"), atoms)
    job.verbosity = 0

    with pytest.raises(ValueError, match="rigid-body invariance"):
        job.compute_frequencies(hessian)


@pytest.mark.parametrize(
    ("atoms", "vibrational_frequencies_cm1", "symmetry_number"),
    [
        (
            Atoms(
                numbers=[8, 1, 1],
                positions=WATER_POSITIONS_A,
                masses=WATER_MASSES_AMU,
            ),
            np.array([1595.0, 3657.0, 3756.0]),
            2,
        ),
        (
            Atoms(
                "N2",
                positions=[[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]],
            ),
            np.array([2358.0]),
            2,
        ),
        (Atoms("Ar", positions=[[0.0, 0.0, 0.0]]), np.array([]), 1),
    ],
)
def test_rrho_thermochemistry_matches_ase(
    tmp_path,
    atoms,
    vibrational_frequencies_cm1,
    symmetry_number,
):
    temperature = 298.15
    pressure_kpa = 101.325
    job = MWFrequency(
        str(tmp_path / "freq.out"),
        atoms,
        temperature=temperature,
        pressure_kpa=pressure_kpa,
        symmetry_number=symmetry_number,
        ilowfreq=0,
    )
    rigid_count = 3 if len(atoms) == 1 else (5 if len(atoms) == 2 else 6)
    frequencies = np.concatenate((np.zeros(rigid_count), vibrational_frequencies_cm1))

    result = job.compute_thermo(frequencies)
    expected_h, expected_s, expected_g = _ase_rrho_totals(
        atoms,
        vibrational_frequencies_cm1,
        temperature=temperature,
        pressure_kpa=pressure_kpa,
        symmetry_number=symmetry_number,
    )

    assert result.h_total_kjmol == pytest.approx(expected_h, abs=1.0e-4)
    assert result.s_total_jmolK == pytest.approx(expected_s, abs=1.0e-4)
    assert result.g_correction_kjmol == pytest.approx(expected_g, abs=3.0e-5)


def test_rrho_electronic_entropy_uses_explicit_multiplicity(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()
    atoms.info["mult"] = 2
    job = MWFrequency(
        str(tmp_path / "freq.out"),
        atoms,
        symmetry_number=2,
        ilowfreq=0,
    )
    frequencies = np.array([0.0] * 6 + [1595.0, 3657.0, 3756.0])

    result = job.compute_thermo(frequencies)
    _, expected_entropy, _ = _ase_rrho_totals(
        atoms,
        frequencies[6:],
        temperature=job.temperature,
        pressure_kpa=job.pressure_kpa,
        symmetry_number=job.symmetry_number,
    )

    assert result.s_elec_jmolK == pytest.approx(8.314462618 * np.log(2.0), abs=1e-8)
    assert result.s_total_jmolK == pytest.approx(expected_entropy, abs=1.0e-4)


def test_rrho_thermochemistry_rejects_transition_state_mode(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()
    job = MWFrequency(str(tmp_path / "freq.out"), atoms, ilowfreq=0)

    with pytest.raises(ValueError, match="TS thermochemistry"):
        job.compute_thermo(np.array([0.0] * 6 + [-250.0, 1595.0, 3657.0]))


def test_rrho_keeps_soft_positive_vibrations_instead_of_dropping_them(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()
    job = MWFrequency(
        str(tmp_path / "freq.out"),
        atoms,
        symmetry_number=2,
        ilowfreq=0,
    )
    frequencies = np.array([0.0] * 6 + [1.0, 1595.0, 3657.0])

    result = job.compute_thermo(frequencies)
    expected_h, expected_s, expected_g = _ase_rrho_totals(
        atoms,
        frequencies[6:],
        temperature=job.temperature,
        pressure_kpa=job.pressure_kpa,
        symmetry_number=job.symmetry_number,
    )

    assert result.h_total_kjmol == pytest.approx(expected_h, abs=1.0e-4)
    assert result.s_total_jmolK == pytest.approx(expected_s, abs=1.0e-4)
    assert result.g_correction_kjmol == pytest.approx(expected_g, abs=3.0e-5)


def test_rrho_rejects_zero_vibrational_mode(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()
    job = MWFrequency(str(tmp_path / "freq.out"), atoms, ilowfreq=0)

    with pytest.raises(ValueError, match="strictly positive"):
        job.compute_thermo(np.array([0.0] * 7 + [1595.0, 3657.0]))


@pytest.mark.parametrize("method", ["nonmw", "both"])
def test_public_frequency_rejects_nonphysical_methods(tmp_path, method):
    atoms, _ = _water_with_known_ev_hessian()

    with pytest.raises(ValueError, match="mass-weighted"):
        Frequency(
            str(tmp_path / "freq.out"),
            atoms,
            params=FrequencyParams(method=method),
        )


def test_public_frequency_rejects_unvalidated_low_frequency_models(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()

    with pytest.raises(ValueError, match="RRHO"):
        Frequency(
            str(tmp_path / "freq.out"),
            atoms,
            params=FrequencyParams(ilowfreq=2),
        )


def test_frequency_parameter_alias_keeps_documented_verbosity(tmp_path):
    atoms, _ = _water_with_known_ev_hessian()

    frequency = Frequency(
        str(tmp_path / "freq.out"),
        atoms,
        paras={"method": "mw", "verbosity": 2, "ilowfreq": 0},
    )

    assert frequency.params.verbose == 2


@pytest.mark.parametrize(
    ("paras", "message"),
    [
        ({"ilow": 0}, "Unknown FREQ parameter"),
        ({"treat_imag_as_real": "Ture"}, "true or false"),
        ({"symmetry_number": 1.5}, "positive integer"),
        ({"temperature": float("nan")}, "finite"),
    ],
)
def test_frequency_rejects_ambiguous_or_invalid_parameters(
    tmp_path,
    paras,
    message,
):
    atoms, _ = _water_with_known_ev_hessian()

    with pytest.raises(ValueError, match=message):
        Frequency(str(tmp_path / "freq.out"), atoms, paras=paras)


def test_frequency_run_uses_ev_hessian_and_reports_consistent_entropy(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()

    class _EVHessianCalculator:
        @staticmethod
        def get_forces(_atoms):
            return np.zeros((3, 3))

        @staticmethod
        def get_hessian(_atoms):
            return hessian.copy()

    calculator = _EVHessianCalculator()
    atoms.calc = calculator
    output = tmp_path / "freq.out"
    driver = Frequency(
        str(output),
        atoms,
        paras={
            "method": "mw",
            "verbosity": 10,
            "ilowfreq": 0,
            "symmetry_number": 2,
            "device": "gpu0",
        },
    )
    analysis = analyze_cartesian_hessian(
        hessian,
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )
    rrho = MWFrequency(
        str(tmp_path / "unused.out"),
        atoms,
        symmetry_number=2,
        ilowfreq=0,
    ).compute_thermo(np.concatenate((np.zeros(6), analysis.frequencies_cm1)))

    driver.run()

    text = output.read_text(encoding="utf-8")
    assert atoms.calc is calculator
    for frequency in analysis.frequencies_cm1:
        assert f"{frequency:10.2f} cm**-1" in text
    assert text.count("Translational entropy") == 1
    entropy_correction_kcal = -driver.params.temperature * rrho.s_total_jmolK / 4184.0
    assert (
        f"Total entropy correction         ...   "
        f"{entropy_correction_kcal:10.2f} kcal/mol"
    ) in text
    summary = output.with_suffix(".sum").read_text(encoding="utf-8")
    assert "Number of Vib. Modes:   3" in summary
    assert "S_elec:" in summary


def test_frequency_run_rejects_nonstationary_geometry_before_hessian(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()
    observed = {"hessian_calls": 0}

    class _NonstationaryCalculator:
        @staticmethod
        def get_forces(_atoms):
            forces = np.zeros((3, 3))
            forces[0, 0] = 2.0e-3
            return forces

        @staticmethod
        def get_hessian(_atoms):
            observed["hessian_calls"] += 1
            return hessian.copy()

    atoms.calc = _NonstationaryCalculator()
    driver = Frequency(
        str(tmp_path / "freq.out"),
        atoms,
        paras={"method": "mw", "ilowfreq": 0},
    )

    with pytest.raises(ValueError, match="stationary"):
        driver.run()
    assert observed["hessian_calls"] == 0


def test_frequency_rejects_private_legacy_hartree_calculator_view(tmp_path):
    atoms, hessian = _water_with_known_ev_hessian()

    class _RawCalculator:
        @staticmethod
        def get_forces(_atoms):
            return np.zeros((3, 3))

        @staticmethod
        def get_hessian(_atoms):
            return hessian.copy()

    atoms.calc = LegacyHartreeJobView(_RawCalculator())
    driver = Frequency(
        str(tmp_path / "freq.out"),
        atoms,
        paras={"method": "mw", "ilowfreq": 0},
    )

    with pytest.raises(RuntimeError, match="raw ASE calculator"):
        driver.run()


def test_frequency_rejects_periodic_or_constrained_systems(tmp_path):
    periodic, _ = _water_with_known_ev_hessian()
    periodic.set_cell([10.0, 10.0, 10.0])
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="non-periodic"):
        Frequency(str(tmp_path / "periodic.out"), periodic)

    constrained, _ = _water_with_known_ev_hessian()
    constrained.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NotImplementedError, match="constraints"):
        Frequency(str(tmp_path / "constrained.out"), constrained)


@pytest.mark.parametrize("method", ["nonmw", "both"])
def test_command_control_rejects_nonphysical_frequency_methods(method):
    with pytest.raises(ValueError, match="not implemented"):
        CommandControl.from_settings([f"#freq(method={method})"])


def test_command_control_defaults_to_rrho_mass_weighted_frequency():
    command = CommandControl.from_settings(["#freq"])

    assert command.params["method"] == "mw"
    assert command.params["ilowfreq"] == 0
    assert command.params["stationarity_tolerance_ev_per_a"] == pytest.approx(1.0e-3)
    assert command.params["hessian_symmetry_relative_tolerance"] == pytest.approx(
        1.0e-6
    )
    assert command.params["n_freqs_to_print"] == 10
    assert command.params["imag_tol_cm1"] == pytest.approx(10.0)


def test_command_control_rejects_unknown_frequency_parameter():
    with pytest.raises(ValueError, match="Unknown FREQ parameter.*ilow"):
        CommandControl.from_settings(["#freq(ilow=2)"])


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("#freq(ilowfreq=2)", "only ilowfreq=0"),
        ("#freq(treat_imag_as_real=Ture)", "true or false"),
        ("#freq(symmetry_number=1.5)", "positive integer"),
        ("#freq(stationarity_tolerance_ev_per_a=0)", "positive number"),
        (
            "#freq(hessian_symmetry_relative_tolerance=-1)",
            "non-negative",
        ),
        ("#freq(n_freqs_to_print=-1)", "non-negative integer"),
        ("#freq(imag_tol_cm1=-1)", "non-negative"),
    ],
)
def test_command_control_rejects_invalid_frequency_values(setting, message):
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings([setting])


def test_dispatcher_frequency_keeps_raw_ase_calculator_unit_boundary(
    tmp_path,
    monkeypatch,
):
    atoms, _ = _water_with_known_ev_hessian()
    raw_calculator = SimpleNamespace()
    atoms.calc = raw_calculator
    observed: dict[str, object] = {}

    class _ProbeFrequency:
        def __init__(self, *, output, atoms, paras):
            del output, paras
            observed["calculator"] = atoms.calc

        @staticmethod
        def run():
            return None

    import maple.function.dispatcher.frequency as frequency_package

    monkeypatch.setattr(frequency_package, "Frequency", _ProbeFrequency)
    command = SimpleNamespace(params={"method": "mw"})

    Dispatcher()(command, "freq", atoms, str(tmp_path / "freq.out"))

    assert observed["calculator"] is raw_calculator
    assert not isinstance(observed["calculator"], LegacyHartreeJobView)
    assert atoms.calc is raw_calculator
