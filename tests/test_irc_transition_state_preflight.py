from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.calculator.calculator_base import HARTREE2EV
from maple.function.dispatcher.frequency.normal_modes import (
    EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1,
    rigid_body_subspaces,
)
from maple.function.dispatcher.irc.preflight import (
    IRCPreflightParams,
    validate_irc_transition_state,
)
from maple.function.dispatcher.irc.path import finalize_irc_branch
from maple.function.read.command_control import CommandControl

WATER_MASSES_AMU = np.array([15.999, 1.008, 1.008])
WATER_POSITIONS_A = np.array(
    [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
)


def _water_legacy_hessian(
    vibrational_eigenvalues_eV_per_A2_amu: object,
) -> tuple[Atoms, np.ndarray]:
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    basis = subspaces.vibrational_basis_mass_weighted
    mass_weighted = (
        basis
        @ np.diag(np.asarray(vibrational_eigenvalues_eV_per_A2_amu, dtype=float))
        @ basis.T
    )
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    hessian_eV_per_A2 = (
        square_root_mass[:, None] * mass_weighted * square_root_mass[None, :]
    )
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=WATER_POSITIONS_A,
        masses=WATER_MASSES_AMU,
    )
    return atoms, hessian_eV_per_A2 / HARTREE2EV


def test_irc_preflight_selects_the_projected_first_order_saddle_mode():
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])

    result = validate_irc_transition_state(
        atoms,
        hessian_hartree,
        np.zeros((3, 3)),
        IRCPreflightParams(),
    )

    expected_frequency = -EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1
    assert result.assessment.imaginary_frequency_cm1 == pytest.approx(
        expected_frequency,
        abs=2.0e-11,
    )
    assert result.negative_eigenvalue_hartree_per_A2_amu == pytest.approx(
        -1.0 / HARTREE2EV,
        abs=1.0e-14,
    )
    rigid = rigid_body_subspaces(
        WATER_MASSES_AMU,
        WATER_POSITIONS_A,
    )
    rigid_basis = np.concatenate(
        (
            rigid.translation_basis_mass_weighted,
            rigid.rotation_basis_mass_weighted,
        ),
        axis=1,
    )
    assert rigid_basis.T @ result.negative_mode_mass_weighted == pytest.approx(
        np.zeros(rigid.rigid_rank),
        abs=2.0e-14,
    )
    sign_pivot = int(np.argmax(np.abs(result.negative_mode_mass_weighted)))
    assert result.negative_mode_mass_weighted[sign_pivot] > 0.0


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        ([1.0, 4.0, 9.0], "exactly one robust imaginary"),
        ([-1.0, -0.25, 9.0], "exactly one robust imaginary"),
        (
            [-((20.0 / EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1) ** 2), 4.0, 9.0],
            "ambiguous non-positive",
        ),
    ],
)
def test_irc_preflight_rejects_non_first_order_saddle_signatures(
    eigenvalues,
    message,
):
    atoms, hessian_hartree = _water_legacy_hessian(eigenvalues)

    with pytest.raises(ValueError, match=message):
        validate_irc_transition_state(
            atoms,
            hessian_hartree,
            np.zeros((3, 3)),
            IRCPreflightParams(),
        )


def test_irc_preflight_rejects_nonstationary_geometry():
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])
    forces_hartree_per_A = np.zeros((3, 3))
    forces_hartree_per_A[0, 0] = 2.0e-3 / HARTREE2EV

    with pytest.raises(ValueError, match="stationary geometry"):
        validate_irc_transition_state(
            atoms,
            hessian_hartree,
            forces_hartree_per_A,
            IRCPreflightParams(),
        )


def test_irc_preflight_rejects_hidden_rigid_body_curvature():
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])
    subspaces = rigid_body_subspaces(WATER_MASSES_AMU, WATER_POSITIONS_A)
    translation = subspaces.translation_basis_mass_weighted[:, 0]
    square_root_mass = np.sqrt(np.repeat(WATER_MASSES_AMU, 3))
    spurious_mass_weighted_eV = 0.01 * np.outer(translation, translation)
    spurious_cartesian_hartree = (
        square_root_mass[:, None]
        * spurious_mass_weighted_eV
        * square_root_mass[None, :]
        / HARTREE2EV
    )

    with pytest.raises(ValueError, match="rigid-body invariance"):
        validate_irc_transition_state(
            atoms,
            hessian_hartree + spurious_cartesian_hartree,
            np.zeros((3, 3)),
            IRCPreflightParams(),
        )


def test_irc_preflight_rejects_asymmetric_hessian():
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])
    hessian_hartree[0, 1] += 0.1 / HARTREE2EV

    with pytest.raises(ValueError, match="not symmetric"):
        validate_irc_transition_state(
            atoms,
            hessian_hartree,
            np.zeros((3, 3)),
            IRCPreflightParams(),
        )


@pytest.mark.parametrize("domain", ["periodic", "constrained"])
def test_irc_preflight_rejects_unsupported_molecular_domains(domain):
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])
    if domain == "periodic":
        atoms.set_cell([10.0, 10.0, 10.0])
        atoms.set_pbc(True)
        expected_exception = ValueError
        message = "non-periodic"
    else:
        atoms.set_constraint(FixAtoms(indices=[0]))
        expected_exception = NotImplementedError
        message = "constraints"

    with pytest.raises(expected_exception, match=message):
        validate_irc_transition_state(
            atoms,
            hessian_hartree,
            np.zeros((3, 3)),
            IRCPreflightParams(),
        )


def test_irc_preflight_rejects_targeting_a_second_negative_mode():
    atoms, hessian_hartree = _water_legacy_hessian([-1.0, 4.0, 9.0])

    with pytest.raises(ValueError, match="target_mode=1"):
        validate_irc_transition_state(
            atoms,
            hessian_hartree,
            np.zeros((3, 3)),
            IRCPreflightParams(target_mode=2),
        )


def test_command_control_exposes_the_shared_irc_preflight_defaults():
    command = CommandControl.from_settings(["#irc(method=gs)"])

    assert command.params["target_mode"] == 1
    assert command.params["stationarity_tolerance_ev_per_a"] == pytest.approx(1.0e-3)
    assert command.params["hessian_symmetry_relative_tolerance"] == pytest.approx(
        1.0e-6
    )
    assert command.params["rigid_mode_tolerance_cm1"] == pytest.approx(5.0)
    assert command.params["transition_state_imaginary_threshold_cm1"] == (
        pytest.approx(50.0)
    )
    assert command.params["step_length_bohr"] == pytest.approx(0.10)
    assert command.params["max_steps"] == 50
    assert command.params["f_max_th"] == pytest.approx(2.0e-3)
    assert command.params["f_rms_th"] == pytest.approx(5.0e-4)
    assert command.params["path_energy_tolerance_hartree"] == pytest.approx(1.0e-7)
    assert command.params["require_converged_endpoints"] is True


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("#irc(method=gs,target_mode=2)", "target_mode=1"),
        (
            "#irc(method=gs,stationarity_tolerance_ev_per_a=0)",
            "finite positive",
        ),
        (
            "#irc(method=gs,transition_state_imaginary_threshold_cm1=0)",
            "finite positive",
        ),
        ("#irc(method=gs,max_steps=0)", "max_steps"),
        ("#irc(method=gs,step_length_bohr=0)", "step_length_bohr"),
        ("#irc(method=gs,f_max_th=0)", "f_max_th"),
        (
            "#irc(method=gs,path_energy_tolerance_hartree=0)",
            "path_energy_tolerance_hartree",
        ),
        (
            "#irc(method=gs,require_converged_endpoints=1)",
            "require_converged_endpoints",
        ),
    ],
)
def test_command_control_rejects_invalid_irc_preflight_parameters(
    setting,
    message,
):
    with pytest.raises(ValueError, match=message):
        CommandControl.from_settings([setting])


class _ZeroLegacyCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


@pytest.mark.parametrize(
    ("module_name", "class_name", "params_name"),
    [
        ("gs", "GS", "GSParams"),
        ("lqa", "LQA", "LQAParams"),
        ("hpc", "HPC", "HPCParams"),
        ("eulerpc", "EulerPC", "EulerPCParams"),
    ],
)
def test_every_irc_integrator_uses_the_shared_projected_preflight_mode(
    tmp_path,
    monkeypatch,
    module_name,
    class_name,
    params_name,
):
    module = importlib.import_module(
        f"maple.function.dispatcher.irc.algorithm.{module_name}"
    )
    integrator_class = getattr(module, class_name)
    params_class = getattr(module, params_name)
    atoms = Atoms("H2", positions=[[0.0, 0.0, -0.4], [0.0, 0.0, 0.4]])
    atoms.calc = _ZeroLegacyCalculator()
    integrator = integrator_class(
        atoms,
        output=str(tmp_path / f"{module_name}.out"),
        params=params_class(write_traj=False),
    )
    expected_mode = np.arange(6, dtype=float) + 1.0
    observed_modes: list[np.ndarray] = []

    def _fake_preflight(atoms_arg, hessian_arg, forces_arg, params_arg):
        assert atoms_arg is atoms
        assert hessian_arg.shape == (6, 6)
        assert forces_arg.shape == (2, 3)
        assert params_arg is integrator.p
        return SimpleNamespace(
            negative_mode_mass_weighted=expected_mode,
            negative_eigenvalue_hartree_per_A2_amu=-0.25,
            assessment=SimpleNamespace(imaginary_frequency_cm1=-100.0),
            maximum_force_eV_per_A=0.0,
            rigid_residual_cm1=0.0,
        )

    monkeypatch.setattr(module, "validate_irc_transition_state", _fake_preflight)
    monkeypatch.setattr(
        integrator,
        "_get_hessian_cart",
        lambda: np.eye(6),
    )

    def _fake_one_side(**kwargs):
        observed_modes.append(np.array(kwargs["v_neg_mw"], copy=True))
        direction = "forward" if kwargs["forward"] else "backward"
        displaced = atoms.get_positions().copy()
        displaced[0, 0] = 0.1 if kwargs["forward"] else -0.1
        return finalize_irc_branch(
            title=f"{direction.upper()} IRC",
            direction=direction,
            records=[
                {
                    "E": -0.1,
                    "maxG": 0.0,
                    "rmsG": 0.0,
                    "x": displaced,
                }
            ],
            transition_state_energy_hartree=kwargs["E_ts"],
            termination_reason="force_converged",
            iterations_attempted=0,
            maximum_force_threshold_hartree_per_A=integrator.p.f_max_th,
            rms_force_threshold_hartree_per_A=integrator.p.f_rms_th,
        )

    monkeypatch.setattr(integrator, "_one_side", _fake_one_side)

    result = integrator.run()

    assert len(observed_modes) == 2
    assert observed_modes[0] == pytest.approx(expected_mode)
    assert observed_modes[1] == pytest.approx(expected_mode)
    assert result["summary"]["converged"] is True
    assert result["summary"]["ts_index"] == 2
    assert result["summary"]["records"][1]["point_kind"] == "transition_state"
    assert result["summary"]["records"][1]["x"] == pytest.approx(atoms.get_positions())


@pytest.mark.parametrize(
    ("module_name", "class_name", "params_name"),
    [
        ("gs", "GS", "GSParams"),
        ("lqa", "LQA", "LQAParams"),
        ("hpc", "HPC", "HPCParams"),
        ("eulerpc", "EulerPC", "EulerPCParams"),
    ],
)
def test_every_irc_integrator_accepts_a_converged_initial_displacement(
    tmp_path,
    monkeypatch,
    module_name,
    class_name,
    params_name,
):
    module = importlib.import_module(
        f"maple.function.dispatcher.irc.algorithm.{module_name}"
    )
    integrator_class = getattr(module, class_name)
    params_class = getattr(module, params_name)
    atoms = Atoms("H2", positions=[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    atoms.calc = _ZeroLegacyCalculator()
    integrator = integrator_class(
        atoms,
        output=str(tmp_path / f"{module_name}.out"),
        params=params_class(write_traj=False, print_each=False),
    )
    integrator._D = np.ones(6)
    integrator._step_len_mw = 0.1
    if hasattr(integrator, "_step_len_umw"):
        integrator._step_len_umw = 0.1

    def _zero_energy_forces(q_mw):
        atoms.set_positions(np.asarray(q_mw).reshape(-1, 3))
        return -0.1, np.zeros(6)

    monkeypatch.setattr(integrator, "_energy_forces_from_mw", _zero_energy_forces)
    monkeypatch.setattr(integrator, "_get_hessian_cart", lambda: np.eye(6))
    monkeypatch.setattr(
        integrator,
        "_micro_step",
        lambda: pytest.fail("macro propagation must not run after initial convergence"),
    )

    branch = integrator._one_side(
        forward=True,
        sign=1.0,
        q_ts_cart=atoms.get_positions().reshape(-1).copy(),
        v_neg_mw=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert branch["status"]["converged"] is True
    assert branch["status"]["termination_reason"] == "force_converged"
    assert branch["status"]["iterations_attempted"] == 0
    assert branch["status"]["accepted_macro_steps"] == 0
    assert len(branch["records"]) == 1
