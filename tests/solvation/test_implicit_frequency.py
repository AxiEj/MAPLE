from __future__ import annotations

import re

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixInternals

from maple.function.dispatcher.frequency.frequency import (
    Frequency,
    FrequencyParams,
    HARTREE_AMU_ANGSTROM2_TO_CM1,
    MWFrequency,
    NonMWFrequency,
    ThermoResults,
)
from maple.function.engine import engine
from maple.function.timer import timer


def _reported_kcal_mol(output_text, label):
    match = re.search(
        rf"^{re.escape(label)}\s+\.\.\.\s+([-+]?\d+\.\d+)\s+kcal/mol$",
        output_text,
        flags=re.MULTILINE,
    )
    assert match is not None
    return float(match.group(1))


def test_thermochemistry_report_closes_enthalpy_entropy_and_gibbs(tmp_path):
    output_path = tmp_path / "thermochemistry.out"
    frequency = MWFrequency(
        str(output_path),
        Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.96],
                [0.92, 0.0, -0.24],
            ],
        ),
    )
    temperature = frequency.temperature
    thermo = ThermoResults(
        zpe_kjmol=10.0,
        h_trans_kjmol=5.0,
        h_rot_kjmol=2.0,
        h_vib_thermal_kjmol=3.0,
        s_trans_jmolK=50.0,
        s_rot_jmolK=20.0,
        s_vib_jmolK=30.0,
        g_correction_kjmol=20.0 - temperature * 100.0e-3,
    )

    frequency._write_thermochemistry(thermo)
    output_text = output_path.read_text(encoding="utf-8")
    enthalpy = _reported_kcal_mol(
        output_text,
        "Total enthalpy correction",
    )
    entropy = _reported_kcal_mol(
        output_text,
        "Total entropy correction",
    )
    gibbs = _reported_kcal_mol(
        output_text,
        "Final Gibbs free energy corr.",
    )

    assert entropy == pytest.approx(
        -temperature * 100.0e-3 / 4.184,
        abs=0.01,
    )
    assert gibbs == pytest.approx(enthalpy + entropy, abs=0.02)


def test_linear_diatomic_projection_retains_the_stretch_mode(tmp_path):
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, -0.6], [0.0, 0.0, 0.6]],
    )
    frequency = MWFrequency(str(tmp_path / "co.out"), atoms)
    masses = atoms.get_masses()
    stretch = np.zeros(6, dtype=np.float64)
    stretch[2] = np.sqrt(masses[1])
    stretch[5] = -np.sqrt(masses[0])
    stretch /= np.linalg.norm(stretch)
    stretch_hessian = np.outer(stretch, stretch)

    basis = frequency._build_translation_rotation_basis(
        masses,
        atoms.get_positions(),
    )
    projected = frequency._project_hessian(stretch_hessian)

    assert np.linalg.matrix_rank(basis) == 5
    np.testing.assert_allclose(
        projected,
        stretch_hessian,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.linalg.matrix_rank(projected, tol=1.0e-12) == 1


def test_linear_triatomic_projection_retains_four_vibrations(tmp_path):
    atoms = Atoms(
        "CO2",
        positions=[
            [0.0, 0.0, -1.16],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.16],
        ],
    )
    frequency = MWFrequency(str(tmp_path / "co2.out"), atoms)
    projected = frequency._project_hessian(np.eye(9))

    assert frequency._is_linear_molecule()
    assert np.linalg.matrix_rank(projected, tol=1.0e-12) == 4


def test_mass_weighting_precedes_rigid_body_projection(tmp_path):
    atoms = Atoms(
        "HOF",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [1.22, 1.05, 0.0],
        ],
    )
    frequency = MWFrequency(str(tmp_path / "hof.out"), atoms)
    generator = np.random.default_rng(20260725)
    factor = generator.normal(size=(9, 9))
    cartesian_hessian = factor.T @ factor + np.eye(9)
    inverse_sqrt_mass = np.repeat(
        1.0 / np.sqrt(atoms.get_masses()),
        3,
    )
    mass_weighted_hessian = (
        cartesian_hessian * inverse_sqrt_mass[None, :] * inverse_sqrt_mass[:, None]
    )
    rigid_basis = frequency._build_translation_rotation_basis(
        atoms.get_masses(),
        atoms.get_positions(),
    )
    left_vectors, singular_values, _ = np.linalg.svd(
        rigid_basis,
        full_matrices=False,
    )
    tolerance = np.finfo(np.float64).eps * max(rigid_basis.shape) * singular_values[0]
    orthonormal_basis = left_vectors[
        :,
        singular_values > tolerance,
    ]
    projector = np.eye(9) - orthonormal_basis @ orthonormal_basis.T
    expected_hessian = projector @ mass_weighted_hessian @ projector
    expected_eigenvalues = np.linalg.eigvalsh(expected_hessian)
    expected_frequencies = (
        np.sqrt(expected_eigenvalues[expected_eigenvalues > 1.0e-12]) * 2720.22864939427
    )

    frequencies, _ = frequency.compute_frequencies(cartesian_hessian)

    assert HARTREE_AMU_ANGSTROM2_TO_CM1 == pytest.approx(
        2720.22864939427,
        abs=1.0e-10,
    )
    assert len(expected_frequencies) == 3
    np.testing.assert_allclose(
        frequencies[-3:],
        expected_frequencies,
        atol=1.0e-8,
        rtol=1.0e-12,
    )


def test_frequency_constructor_rejects_unknown_low_frequency_treatment(
    tmp_path,
):
    with pytest.raises(ValueError, match="ilowfreq"):
        Frequency(
            str(tmp_path / "invalid-low-frequency.out"),
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            params=FrequencyParams(ilowfreq=99),
        )
    with pytest.raises(ValueError, match="ilowfreq"):
        MWFrequency(
            str(tmp_path / "invalid-direct-low-frequency.out"),
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            ilowfreq=99,
        )


def test_grimme_rotor_entropy_uses_finite_mean_molecular_inertia(tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.92, 0.0, -0.24],
        ],
    )
    frequency = MWFrequency(
        str(tmp_path / "rotor-entropy.out"),
        atoms,
        ilowfreq=2,
    )
    temperature = 298.15
    nu_cm1 = 10.0
    h = 6.62607015e-34
    k_b = 1.380649e-23
    r_gas = 8.31446261815324
    amu_angstrom2_to_kg_m2 = 1.66053906660e-47
    average_inertia = np.mean(atoms.get_moments_of_inertia()) * amu_angstrom2_to_kg_m2
    mu = h / (8.0 * np.pi**2 * nu_cm1 * 2.99792458e10)
    mu_effective = mu * average_inertia / (mu + average_inertia)
    expected = r_gas * (
        0.5 + np.log(np.sqrt(8.0 * np.pi**3 * mu_effective * k_b * temperature / h**2))
    )

    observed = frequency._free_rotor_entropy(
        nu_cm1,
        temperature,
        Bav_amuA2=None,
    )

    assert observed == pytest.approx(expected, rel=1.0e-12)
    assert np.isfinite(
        frequency._free_rotor_entropy(
            1.0e-9,
            temperature,
            Bav_amuA2=None,
        )
    )


def test_minenkov_qrrho_interpolates_zpe_and_thermal_internal_energy(tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.92, 0.0, -0.24],
        ],
    )
    frequency = MWFrequency(
        str(tmp_path / "minenikov-energy.out"),
        atoms,
        ilowfreq=3,
        omega0_cm1=100.0,
    )
    temperature = frequency.temperature
    nu_cm1 = 10.0
    h = 6.62607015e-34
    c_cm_s = 2.99792458e10
    n_a = 6.02214076e23
    r_gas = 8.31446261815324
    mode_energy_j_mol = h * c_cm_s * nu_cm1 * n_a
    x = mode_energy_j_mol / (r_gas * temperature)
    zpe_harmonic_j_mol = 0.5 * mode_energy_j_mol
    thermal_harmonic_j_mol = mode_energy_j_mol / np.expm1(x)
    free_rotor_j_mol = 0.5 * r_gas * temperature
    weight = 1.0 / (1.0 + (100.0 / nu_cm1) ** 4)
    expected_vibrational_internal_energy_kj_mol = (
        weight * (zpe_harmonic_j_mol + thermal_harmonic_j_mol)
        + (1.0 - weight) * free_rotor_j_mol
    ) * 1.0e-3

    thermo = frequency.compute_thermo(np.asarray([nu_cm1]))

    assert (thermo.zpe_kjmol + thermo.h_vib_thermal_kjmol) == pytest.approx(
        expected_vibrational_internal_energy_kj_mol,
        rel=1.0e-12,
    )
    assert thermo.zpe_kjmol == pytest.approx(
        weight * zpe_harmonic_j_mol * 1.0e-3,
        rel=1.0e-12,
    )


def test_grimme_entropy_only_keeps_harmonic_internal_energy(tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.92, 0.0, -0.24],
        ],
    )
    frequency = MWFrequency(
        str(tmp_path / "grimme-energy.out"),
        atoms,
        ilowfreq=2,
        omega0_cm1=100.0,
    )
    temperature = frequency.temperature
    nu_cm1 = 10.0
    h = 6.62607015e-34
    c_cm_s = 2.99792458e10
    n_a = 6.02214076e23
    r_gas = 8.31446261815324
    mode_energy_j_mol = h * c_cm_s * nu_cm1 * n_a
    x = mode_energy_j_mol / (r_gas * temperature)
    expected_harmonic_internal_energy_kj_mol = (
        0.5 * mode_energy_j_mol + mode_energy_j_mol / np.expm1(x)
    ) * 1.0e-3
    harmonic_entropy_j_mol_k = r_gas * (x / np.expm1(x) - np.log1p(-np.exp(-x)))
    weight = 1.0 / (1.0 + (100.0 / nu_cm1) ** 4)
    expected_entropy_j_mol_k = weight * harmonic_entropy_j_mol_k + (
        1.0 - weight
    ) * frequency._free_rotor_entropy(
        nu_cm1,
        temperature,
        Bav_amuA2=None,
    )

    thermo = frequency.compute_thermo(np.asarray([nu_cm1]))

    assert (thermo.zpe_kjmol + thermo.h_vib_thermal_kjmol) == pytest.approx(
        expected_harmonic_internal_energy_kj_mol,
        rel=1.0e-12,
    )
    assert thermo.s_vib_jmolK == pytest.approx(expected_entropy_j_mol_k, rel=1.0e-12)


def test_internal_rotational_thermo_keeps_explicit_sub_five_cm1_mode(tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.92, 0.0, -0.24],
        ],
    )
    frequency = MWFrequency(
        str(tmp_path / "explicit-vibrational-mode.out"),
        atoms,
        ilowfreq=0,
        nu_floor_cm1=1.0,
    )

    thermo = frequency.compute_internal_rotational_thermo(
        np.asarray([1.0], dtype=np.float64)
    )

    assert thermo.zpe_kjmol > 0.0
    assert thermo.h_vib_thermal_kjmol > 0.0
    assert thermo.s_vib_jmolK > 0.0
    assert thermo.h_trans_kjmol == 0.0
    assert thermo.s_trans_jmolK == 0.0
    assert thermo.g_correction_kjmol == pytest.approx(
        thermo.h_total_kjmol
        - frequency.temperature * thermo.s_total_jmolK * 1.0e-3,
        rel=1.0e-12,
    )


def test_frequency_verbosity_reaches_the_concrete_job(tmp_path, monkeypatch):
    observed = {}

    def record_run(job):
        observed["verbosity"] = job.verbosity

    monkeypatch.setattr(MWFrequency, "run", record_run)
    frequency = Frequency(
        str(tmp_path / "verbosity.out"),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        params=FrequencyParams(verbosity=10),
    )

    frequency.run()

    assert observed["verbosity"] == 10


@pytest.mark.parametrize(
    "constraint",
    [
        FixAtoms(indices=[0]),
        FixInternals(bonds=[[0.96, [0, 1]]]),
    ],
    ids=["fixed-atom", "fixed-internal"],
)
def test_frequency_fails_closed_for_constraints(tmp_path, constraint):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.92, 0.0, -0.24],
        ],
    )
    atoms.set_constraint(constraint)

    with pytest.raises(ValueError, match="active-coordinate Hessian"):
        MWFrequency(
            str(tmp_path / "constrained-frequency.out"),
            atoms,
        ).run()


def test_direct_api_rejects_nonmw_implicit_frequency(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = Calculator()
    atoms.calc.solvent_correction = object()

    with pytest.raises(ValueError, match="only method=mw"):
        Frequency(
            str(tmp_path / "implicit-nonmw-driver.out"),
            atoms,
            params=FrequencyParams(method="nonmw"),
        ).run()

    with pytest.raises(ValueError, match="only method=mw"):
        NonMWFrequency(
            str(tmp_path / "implicit-nonmw-concrete.out"),
            atoms,
        ).run()


def test_direct_api_rejects_polarizable_implicit_frequency(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = Calculator()
    atoms.calc.solvent_correction = type(
        "PolarizableCorrection",
        (),
        {
            "mode": "polarizable",
            "supported_properties": {"energy", "forces"},
        },
    )()

    with pytest.raises(ValueError, match="mode=fixed"):
        MWFrequency(
            str(tmp_path / "implicit-polarizable-mw.out"),
            atoms,
        ).run()


def test_direct_api_rejects_periodic_implicit_frequency(tmp_path):
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.calc = Calculator()
    atoms.calc.SUPPORTS_IMPLICIT_SOLVATION = True
    atoms.calc.hessian = "numerical"
    atoms.calc.solvent_correction = type(
        "FixedCorrection",
        (),
        {
            "mode": "fixed",
            "supported_properties": {"energy", "forces"},
        },
    )()

    with pytest.raises(ValueError, match="non-periodic"):
        MWFrequency(
            str(tmp_path / "implicit-periodic-mw.out"),
            atoms,
        ).run()


@pytest.mark.parametrize(
    "correction_attributes, calculator_attributes, message",
    [
        (
            {"supported_properties": {"energy", "forces"}},
            {
                "SUPPORTS_IMPLICIT_SOLVATION": True,
                "hessian": "numerical",
            },
            "mode=fixed",
        ),
        (
            {"mode": "fixed", "supported_properties": {"energy"}},
            {
                "SUPPORTS_IMPLICIT_SOLVATION": True,
                "hessian": "numerical",
            },
            "force support",
        ),
        (
            {
                "mode": "fixed",
                "supported_properties": {"energy", "forces"},
            },
            {
                "SUPPORTS_IMPLICIT_SOLVATION": False,
                "hessian": "numerical",
            },
            "SUPPORTS_IMPLICIT_SOLVATION=True",
        ),
        (
            {
                "mode": "fixed",
                "supported_properties": {"energy", "forces"},
            },
            {
                "SUPPORTS_IMPLICIT_SOLVATION": True,
                "hessian": "analytic",
            },
            "hessian=numerical",
        ),
    ],
    ids=[
        "undeclared-charge-mode",
        "energy-only-correction",
        "calculator-without-composition-capability",
        "analytic-gas-hessian",
    ],
)
def test_direct_api_fails_closed_for_incomplete_implicit_frequency_contract(
    tmp_path,
    correction_attributes,
    calculator_attributes,
    message,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = Calculator()
    for name, value in calculator_attributes.items():
        setattr(atoms.calc, name, value)
    atoms.calc.solvent_correction = type(
        "TestCorrection",
        (),
        correction_attributes,
    )()

    with pytest.raises(ValueError, match=message):
        MWFrequency(
            str(tmp_path / "incomplete-implicit-contract.out"),
            atoms,
        ).run()


@pytest.mark.parametrize("model", ["maceoff23m", "aimnet2", "ani2x"])
def test_actual_engine_frequency_differentiates_the_complete_mlip_gb_force(
    model,
    water_mol2,
    tmp_path,
):
    input_path = tmp_path / "water-freq.inp"
    output_path = tmp_path / "water-freq.out"
    input_path.write_text(
        "\n".join(
            [
                f"#model={model}(hessian=numerical)",
                "#freq(method=mw,ilowfreq=2,verbosity=1)",
                "#device=cpu",
                "#charge(source=mol2,label=fixed-water-smoke)",
                (
                    "#solv(implicit=water,method=gb,provider=openmm,"
                    "model=obc2,profile=obc2-mbondi2,nonpolar=ace,"
                    "platform=Reference,experimental=true)"
                ),
                "",
                "0 1",
                f"MOL2 {water_mol2}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    timer.reset()
    maple_engine = engine()
    maple_engine(str(input_path), str(output_path))

    calculator = maple_engine.atoms.calc
    correction = calculator.solvent_correction
    assert calculator.hessian == "numerical"
    assert "forces" in correction.supported_properties

    composed_hessian = np.asarray(
        calculator.get_hessian(maple_engine.atoms),
        dtype=np.float64,
    )
    calculator.solvent_correction = None
    try:
        gas_hessian = np.asarray(
            calculator.get_hessian(maple_engine.atoms),
            dtype=np.float64,
        )
    finally:
        calculator.solvent_correction = correction

    expected_shape = (3 * len(maple_engine.atoms),) * 2
    assert composed_hessian.shape == expected_shape
    assert np.isfinite(composed_hessian).all()
    np.testing.assert_allclose(
        composed_hessian,
        composed_hessian.T,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.linalg.norm(composed_hessian - gas_hessian) > 1.0e-6

    output_text = output_path.read_text(encoding="utf-8")
    assert "Starting frequency analysis calculation" in output_text
    assert "IMPLICIT-SOLVENT FREQUENCY BOUNDARY" in output_text
    assert "not a solution-standard-state Gibbs energy" in output_text
    assert "not an absolute solvation free energy calculation" in output_text
    assert "Final Gibbs free energy corr." in output_text
    assert "Frequency analysis completed" in output_text
    assert "ERROR:" not in output_text
