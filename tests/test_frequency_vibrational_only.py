from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.frequency import (
    BothFrequency,
    FREQUENCY_CONVERSION_CM1,
    Frequency,
    FrequencyParams,
    MWFrequency,
    NonMWFrequency,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.read.command_control import CommandControl


class _HessianCalculator:
    FREQUENCY_HESSIAN_UNITS = "hartree/angstrom^2"

    def __init__(self, thermochemistry=None):
        if thermochemistry is not None:
            self.FREQUENCY_THERMOCHEMISTRY = thermochemistry

    def get_hessian(self, atoms):
        return np.eye(3 * len(atoms))


class _ASEHessianCalculator(Calculator):
    FREQUENCY_THERMOCHEMISTRY = "none"

    def __init__(self, hessian_eV_per_A2):
        super().__init__()
        self.hessian_eV_per_A2 = np.asarray(hessian_eV_per_A2, dtype=float)

    def get_hessian(self, atoms):
        return self.hessian_eV_per_A2.copy()


class _DiagnosticASEHessianCalculator(_ASEHessianCalculator):
    def get_hessian(self, atoms):
        hessian = super().get_hessian(atoms)
        self.hessian_diagnostics = {
            "evaluation_sha256": "abc123",
            "coarse_step_angstrom": 0.002,
            "fine_step_angstrom": 0.001,
            "maximum_error_estimate_eV_per_A2": 0.004,
            "maximum_antisymmetry_eV_per_A2": 0.0003,
            "topology_step_reductions_used": 2,
            "topology_guard_status": "pass",
            "topology_observation_coverage": "partial",
            "unobservable_topology_components": ["buried-cavity birth"],
        }
        return hessian


class _LegacyHartreeHessianCalculator:
    FREQUENCY_THERMOCHEMISTRY = "none"
    FREQUENCY_HESSIAN_UNITS = "hartree/angstrom^2"

    def __init__(self, hessian_hartree_per_A2):
        self.hessian_hartree_per_A2 = np.asarray(
            hessian_hartree_per_A2, dtype=float
        )

    def get_hessian(self, atoms):
        return self.hessian_hartree_per_A2.copy()


def _atoms(thermochemistry=None):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.calc = _HessianCalculator(thermochemistry)
    return atoms


def _bond_hessian(n_atoms, atom_i, atom_j, unit_vector, force_constant):
    gradient = np.zeros(3 * n_atoms)
    gradient[3 * atom_i : 3 * atom_i + 3] = -unit_vector
    gradient[3 * atom_j : 3 * atom_j + 3] = unit_vector
    return force_constant * np.outer(gradient, gradient)


def _reported_frequencies(path):
    values = []
    pattern = re.compile(r"^\s*\d+:\s+([-+]?\d+(?:\.\d+)?)\s+cm\*\*-1")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            values.append(float(match.group(1)))
    return np.asarray(values)


def _nist_hartree_angstrom_amu_to_cm1():
    hartree_j = 4.3597447222071e-18
    atomic_mass_kg = 1.66053906660e-27
    speed_of_light_cm_per_s = 2.99792458e10
    return np.sqrt(hartree_j / (atomic_mass_kg * 1.0e-20)) / (
        2.0 * np.pi * speed_of_light_cm_per_s
    )


@pytest.fixture
def deterministic_modes(monkeypatch):
    frequencies = np.array([-1.25, 0.0, 25.0, 100.0, 200.0, 300.0])
    modes = np.eye(6)
    monkeypatch.setattr(
        MWFrequency,
        "compute_frequencies",
        lambda self, hessian: (frequencies.copy(), modes.copy()),
    )
    return frequencies


def test_default_frequency_run_retains_gas_thermochemistry(
    tmp_path, deterministic_modes
):
    output = tmp_path / "gas.out"
    Frequency(output=str(output), atoms=_atoms()).run()

    text = output.read_text()
    assert "THERMOCHEMISTRY AT" in text
    assert "Final Gibbs free energy corr." in text
    assert "HESSIAN DIAGNOSTICS" not in text


def test_vibrational_only_run_and_summary_skip_all_thermochemistry(
    tmp_path, monkeypatch, deterministic_modes
):
    output = tmp_path / "vibrational.out"
    monkeypatch.setattr(
        MWFrequency,
        "compute_thermo",
        lambda self, frequencies: pytest.fail(
            "vibrational-only analysis must not compute gas thermochemistry"
        ),
    )

    Frequency(
        output=str(output),
        atoms=_atoms(),
        paras={"freq": {"thermochemistry": "none", "verbose": 10}},
    ).run()

    text = output.read_text()
    summary = output.with_suffix(".sum").read_text()
    assert "VIBRATIONAL FREQUENCIES" in text
    assert "NORMAL MODES" in text
    assert "-1.25 cm**-1" in text
    assert "THERMOCHEMISTRY" not in text
    assert "GIBBS FREE ENERGY" not in text
    assert "VIBRATIONAL ANALYSIS SUMMARY" in summary
    assert "-1.25" in summary
    assert "near-zero numerical/uncertain mode" in text
    assert "near-zero uncertain" in summary
    assert "Reliable Vibrational Modes (|nu| >= 5.0 cm^-1): 4" in summary
    assert "Near-Zero Numerical/Uncertain Modes: 2" in summary
    assert "THERMODYNAMIC" not in summary
    assert "Temperature:" not in summary
    assert "Pressure:" not in summary
    assert "ZPE:" not in summary
    assert "G_corr" not in summary


def test_restricted_calculator_enforces_vibrational_only_for_direct_use(
    tmp_path, monkeypatch, deterministic_modes
):
    output = tmp_path / "restricted.out"
    monkeypatch.setattr(
        MWFrequency,
        "compute_thermo",
        lambda self, frequencies: pytest.fail(
            "calculator restriction must prevent gas thermochemistry"
        ),
    )

    frequency = Frequency(output=str(output), atoms=_atoms("none"))
    assert frequency.params.thermochemistry == "none"
    frequency.run()
    assert "THERMOCHEMISTRY" not in output.read_text()


@pytest.mark.parametrize(
    "params", [FrequencyParams(), FrequencyParams(verbose=10)]
)
def test_unspecified_dataclass_thermochemistry_obeys_calculator_policy(
    tmp_path, params
):
    frequency = Frequency(
        output=str(tmp_path / "restricted-default.out"),
        atoms=_atoms("none"),
        params=params,
    )
    assert frequency.params.thermochemistry == "none"


def test_explicit_dataclass_gas_conflicts_with_restricted_calculator(tmp_path):
    with pytest.raises(ValueError, match="restricts frequency thermochemistry to 'none'"):
        Frequency(
            output=str(tmp_path / "restricted-gas.out"),
            atoms=_atoms("none"),
            params=FrequencyParams(thermochemistry="gas"),
        )


def test_direct_mw_frequency_enforces_calculator_policy(
    tmp_path, monkeypatch, deterministic_modes
):
    output = tmp_path / "direct-mw.out"
    job = MWFrequency(output=str(output), atoms=_atoms("none"))
    assert job.thermochemistry == "none"
    monkeypatch.setattr(
        job,
        "compute_thermo",
        lambda frequencies: pytest.fail(
            "direct MWFrequency must honor calculator thermochemistry policy"
        ),
    )

    job.run()
    assert "THERMOCHEMISTRY" not in output.read_text()


def test_direct_mw_frequency_rejects_imaginary_frequency_rewriting(
    tmp_path, deterministic_modes
):
    job = MWFrequency(
        output=str(tmp_path / "direct-rewrite.out"),
        atoms=_atoms("none"),
    )
    job.treat_imag_as_real = True

    with pytest.raises(ValueError, match="treat_imag_as_real"):
        job.run()


def test_restricted_wrapped_calculator_rejects_explicit_gas_request(tmp_path):
    raw = _HessianCalculator("none")
    atoms = _atoms()
    atoms.calc = LegacyHartreeJobView(raw)

    with pytest.raises(ValueError, match="restricts frequency thermochemistry to 'none'"):
        Frequency(
            output=str(tmp_path / "forbidden.out"),
            atoms=atoms,
            paras={"thermochemistry": "gas"},
        )


def test_command_parser_preserves_vibrational_only_parameter(tmp_path):
    command = CommandControl.from_settings(
        ["#freq(thermochemistry=none,verbose=10)"],
        output_path=str(tmp_path / "parse.out"),
    )

    assert command.task == "freq"
    assert command.params["thermochemistry"] == "none"
    assert command.params["verbose"] == 10


@pytest.mark.parametrize("value", [None, "rrho", "off", True])
def test_invalid_thermochemistry_parameter_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="thermochemistry must be one of"):
        Frequency(
            output=str(tmp_path / "invalid.out"),
            atoms=_atoms(),
            paras={"thermochemistry": value},
        )


@pytest.mark.parametrize("value", [None, "invalid"])
def test_direct_frequency_rejects_invalid_thermochemistry(tmp_path, value):
    with pytest.raises(ValueError, match="thermochemistry must be one of"):
        MWFrequency(
            output=str(tmp_path / "direct-invalid.out"),
            atoms=_atoms(),
            thermochemistry=value,
        )


def test_vibrational_only_rejects_imaginary_frequency_rewriting(tmp_path):
    with pytest.raises(ValueError, match="treat_imag_as_real"):
        Frequency(
            output=str(tmp_path / "invalid.out"),
            atoms=_atoms(),
            params=FrequencyParams(
                thermochemistry="none", treat_imag_as_real=True
            ),
        )


@pytest.mark.parametrize("method", ["nonmw", "both"])
def test_vibrational_only_rejects_nonphysical_frequency_methods(
    tmp_path, method
):
    with pytest.raises(ValueError, match="supports only the mass-weighted method"):
        Frequency(
            output=str(tmp_path / "nonphysical.out"),
            atoms=_atoms(),
            paras={"method": method, "thermochemistry": "none"},
        )


@pytest.mark.parametrize("job_type", [NonMWFrequency, BothFrequency])
def test_direct_nonphysical_vibrational_only_jobs_fail_before_calculation(
    tmp_path, job_type
):
    job = job_type(
        output=str(tmp_path / "direct-nonphysical.out"),
        atoms=_atoms(),
        thermochemistry="none",
    )

    with pytest.raises(ValueError, match="supports only the mass-weighted method"):
        job.run()


def test_legacy_gas_summary_still_contains_thermodynamic_values(
    tmp_path, deterministic_modes
):
    output = tmp_path / "gas-summary.out"
    Frequency(
        output=str(output),
        atoms=_atoms(),
        paras={"verbose": 10},
    ).run()

    summary = output.with_suffix(".sum").read_text()
    assert "THERMODYNAMIC SUMMARY" in summary
    assert "ZPE:" in summary
    assert "G_corr (total):" in summary


def test_unspecified_dataclass_thermochemistry_keeps_generic_gas_default(tmp_path):
    frequency = Frequency(
        output=str(tmp_path / "generic-default.out"),
        atoms=_atoms(),
        params=FrequencyParams(verbose=10),
    )
    assert frequency.params.thermochemistry == "gas"


@pytest.mark.parametrize("first_restricted", [False, True])
def test_frequency_params_can_be_reused_across_calculator_policies(
    tmp_path, first_restricted
):
    params = FrequencyParams(verbose=10)
    original_thermochemistry = params.thermochemistry
    policies = ("none", None) if first_restricted else (None, "none")

    resolved = []
    for index, policy in enumerate(policies):
        frequency = Frequency(
            output=str(tmp_path / f"reuse-{index}.out"),
            atoms=_atoms(policy),
            params=params,
        )
        resolved.append(frequency.params.thermochemistry)

    expected = ["none", "gas"] if first_restricted else ["gas", "none"]
    assert resolved == expected
    assert params.thermochemistry is original_thermochemistry
    assert params.verbose == 10


def test_paras_updates_do_not_mutate_caller_owned_frequency_params(tmp_path):
    params = FrequencyParams(verbose=1, method="mw")
    original_thermochemistry = params.thermochemistry

    frequency = Frequency(
        output=str(tmp_path / "paras-copy.out"),
        atoms=_atoms(),
        params=params,
        paras={"verbose": 10, "thermochemistry": "none"},
    )

    assert frequency.params.verbose == 10
    assert frequency.params.thermochemistry == "none"
    assert params.verbose == 1
    assert params.method == "mw"
    assert params.thermochemistry is original_thermochemistry


@pytest.mark.parametrize("symbols,bond_length", [("H2", 0.74), ("HCl", 1.27)])
def test_linear_diatomic_retains_reduced_mass_bond_frequency(
    tmp_path, symbols, bond_length
):
    atoms = Atoms(
        symbols,
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]],
    )
    force_constant = 0.42
    hessian = _bond_hessian(2, 0, 1, np.array([0.0, 0.0, 1.0]), force_constant)
    atoms.calc = _LegacyHartreeHessianCalculator(hessian)
    job = MWFrequency(output=str(tmp_path / "diatomic.out"), atoms=atoms)

    frequencies, _ = job.compute_frequencies(hessian)
    masses = atoms.get_masses()
    expected = _nist_hartree_angstrom_amu_to_cm1() * np.sqrt(
        force_constant * (1.0 / masses[0] + 1.0 / masses[1])
    )

    assert FREQUENCY_CONVERSION_CM1 == pytest.approx(
        2720.2286493942706, rel=1.0e-15
    )
    assert np.sum(np.abs(frequencies) < 1.0e-4) == 5
    assert frequencies[-1] == pytest.approx(expected, rel=1.0e-12)


def test_nonlinear_water_mass_metric_projection_matches_generalized_modes(
    tmp_path,
):
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]]
    )
    atoms = Atoms("OH2", positions=positions)
    unit_oh1 = (positions[1] - positions[0]) / np.linalg.norm(
        positions[1] - positions[0]
    )
    unit_oh2 = (positions[2] - positions[0]) / np.linalg.norm(
        positions[2] - positions[0]
    )
    hessian = _bond_hessian(3, 0, 1, unit_oh1, 0.51)
    hessian += _bond_hessian(3, 0, 2, unit_oh2, 0.37)
    atoms.calc = _LegacyHartreeHessianCalculator(hessian)
    job = MWFrequency(output=str(tmp_path / "water.out"), atoms=atoms)

    frequencies, _ = job.compute_frequencies(hessian)
    inv_sqrt_mass = np.repeat(1.0 / np.sqrt(atoms.get_masses()), 3)
    generalized = hessian * inv_sqrt_mass[:, None] * inv_sqrt_mass[None, :]
    expected_eigenvalues = np.linalg.eigvalsh(generalized)
    expected = _nist_hartree_angstrom_amu_to_cm1() * np.sqrt(
        expected_eigenvalues[-2:]
    )

    rigid_basis = job._build_translation_rotation_basis(
        atoms.get_masses(), atoms.get_positions()
    )
    assert np.linalg.matrix_rank(rigid_basis) == 6
    np.testing.assert_allclose(frequencies[-2:], expected, rtol=1.0e-12, atol=1.0e-9)

    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_3n = np.kron(np.eye(3), rotation)
    rotated_atoms = atoms.copy()
    rotated_atoms.positions = (positions + np.array([2.1, -0.8, 1.3])) @ rotation.T
    rotated_hessian = rotation_3n @ hessian @ rotation_3n.T
    rotated_job = MWFrequency(
        output=str(tmp_path / "rotated-water.out"),
        atoms=rotated_atoms,
        thermochemistry="none",
    )
    rotated_frequencies, _ = rotated_job.compute_frequencies(rotated_hessian)
    np.testing.assert_allclose(
        rotated_frequencies[-2:], frequencies[-2:], rtol=1.0e-12, atol=1.0e-9
    )


def test_raw_wrapped_dispatcher_and_legacy_hartree_frequency_invariance(tmp_path):
    atoms_template = Atoms(
        "HCl", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.27]]
    )
    hessian_eV = _bond_hessian(
        2, 0, 1, np.array([0.0, 0.0, 1.0]), 8.4
    )

    raw_atoms = atoms_template.copy()
    raw_calculator = _ASEHessianCalculator(hessian_eV)
    raw_atoms.calc = raw_calculator
    raw_output = tmp_path / "raw.out"
    raw_job = MWFrequency(output=str(raw_output), atoms=raw_atoms)
    raw_job.verbosity = 2
    raw_job.run()
    assert raw_atoms.calc is raw_calculator

    wrapped_atoms = atoms_template.copy()
    wrapped_atoms.calc = LegacyHartreeJobView(_ASEHessianCalculator(hessian_eV))
    wrapped_output = tmp_path / "wrapped.out"
    Frequency(
        output=str(wrapped_output),
        atoms=wrapped_atoms,
        paras={"verbose": 2},
    ).run()

    dispatcher_atoms = atoms_template.copy()
    dispatcher_calculator = _ASEHessianCalculator(hessian_eV)
    dispatcher_atoms.calc = dispatcher_calculator
    dispatcher_output = tmp_path / "dispatcher.out"
    Dispatcher()(
        SimpleNamespace(params={"method": "mw", "verbose": 2}),
        "freq",
        dispatcher_atoms,
        str(dispatcher_output),
    )
    assert dispatcher_atoms.calc is dispatcher_calculator

    legacy_atoms = atoms_template.copy()
    legacy_atoms.calc = _LegacyHartreeHessianCalculator(hessian_eV * EV2HARTREE)
    legacy_output = tmp_path / "legacy.out"
    Frequency(
        output=str(legacy_output),
        atoms=legacy_atoms,
        paras={"verbose": 2},
    ).run()

    raw = _reported_frequencies(raw_output)
    assert raw.size == 6
    for output in (wrapped_output, dispatcher_output, legacy_output):
        np.testing.assert_allclose(_reported_frequencies(output), raw, atol=0.01)


def test_ambiguous_non_ase_hessian_units_fail_closed(tmp_path):
    class _AmbiguousCalculator:
        def get_hessian(self, atoms):
            return np.eye(3 * len(atoms))

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = _AmbiguousCalculator()
    job = MWFrequency(output=str(tmp_path / "ambiguous.out"), atoms=atoms)

    with pytest.raises(RuntimeError, match="FREQUENCY_HESSIAN_UNITS"):
        job.run()


@pytest.mark.parametrize("wrapped", [False, True])
def test_hessian_diagnostics_are_reported_with_public_units_and_caveat(
    tmp_path, wrapped
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    hessian_eV = _bond_hessian(
        2, 0, 1, np.array([0.0, 0.0, 1.0]), 8.4
    )
    calculator = _DiagnosticASEHessianCalculator(hessian_eV)
    atoms.calc = LegacyHartreeJobView(calculator) if wrapped else calculator
    output = tmp_path / f"diagnostics-{wrapped}.out"

    Frequency(output=str(output), atoms=atoms, paras={"verbose": 2}).run()

    text = output.read_text()
    assert "HESSIAN DIAGNOSTICS" in text
    assert "Evaluation SHA256: abc123" in text
    assert "Coarse step: 0.002 Angstrom" in text
    assert "Fine step: 0.001 Angstrom" in text
    assert "Maximum error estimate: 0.004 eV/Angstrom^2" in text
    assert "Maximum antisymmetry: 0.0003 eV/Angstrom^2" in text
    assert "Topology step reductions used: 2" in text
    assert "Topology guard status: pass" in text
    assert "Topology observation coverage: partial" in text
    assert "Unobservable topology components: ['buried-cavity birth']" in text
    assert "WARNING: Topology observation coverage is partial" in text
    assert "were not validated by this Hessian evaluation" in text
