from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.ts.algorithm.dimer import Dimer


class _QuadraticCalculator(Calculator):
    """Toy calculator whose numeric results intentionally use MAPLE Hartree units."""

    def __init__(self, hessian: np.ndarray, *, implicit: bool = False):
        super().__init__()
        self.implemented_properties = ["energy", "forces"]
        self.hessian_matrix = np.asarray(hessian, dtype=np.float64)
        self.hvp_calls = 0
        self.force_calls = 0
        if implicit:
            self.solvent_correction = SimpleNamespace(
                mode="fixed",
                supported_properties={"energy", "forces"},
            )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        assert atoms is not None
        super().calculate(atoms, properties, system_changes)
        x = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
        self.force_calls += 1
        self.results = {
            "energy": 0.5 * float(x @ self.hessian_matrix @ x),
            "forces": (-(self.hessian_matrix @ x)).reshape(-1, 3),
        }

    def get_hvp(self, atoms, n):
        self.hvp_calls += 1
        x = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
        n = np.asarray(n, dtype=np.float64).reshape(-1)
        return (
            self.hessian_matrix @ n,
            -(self.hessian_matrix @ x),
            0.5 * float(x @ self.hessian_matrix @ x),
        )


def _atoms(calculator, positions=None):
    atoms = Atoms(
        "H2",
        positions=(
            [[0.2, -0.1, 0.3], [0.4, 0.1, -0.2]] if positions is None else positions
        ),
    )
    atoms.calc = calculator
    return atoms


def _dimer(tmp_path, calculator, **params):
    defaults = {
        "remove_rigid": False,
        "n_init": "given",
        "n_given": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "save_traj": False,
        "max_iter": 1,
    }
    defaults.update(params)
    return Dimer(str(tmp_path / "dimer.out"), _atoms(calculator), defaults)


def test_implicit_auto_mode_uses_complete_force_finite_difference(tmp_path):
    gas_hessian = np.diag([-1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    solvent_hessian = np.diag([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    complete_hessian = gas_hessian + solvent_hessian

    class ImplicitCalculator(_QuadraticCalculator):
        def get_hvp(self, atoms, n):
            self.hvp_calls += 1
            raise AssertionError("gas-only HVP must not be called")

    calculator = ImplicitCalculator(complete_hessian, implicit=True)
    dimer = _dimer(tmp_path, calculator)
    center = dimer.atoms.get_positions().copy()

    hn, forces, energy = dimer._evaluate_derivatives(center.reshape(-1), dimer.n)

    assert hn == pytest.approx(complete_hessian @ dimer.n, abs=1.0e-12)
    assert hn != pytest.approx(gas_hessian @ dimer.n)
    assert forces == pytest.approx(-(complete_hessian @ center.reshape(-1)))
    assert energy == pytest.approx(
        0.5 * center.reshape(-1) @ complete_hessian @ center.reshape(-1)
    )
    assert calculator.hvp_calls == 0
    assert dimer.atoms.get_positions() == pytest.approx(center)


def test_finite_difference_failure_restores_center_positions(tmp_path):
    class FailingSideCalculator(_QuadraticCalculator):
        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            assert atoms is not None
            if atoms.positions[0, 0] < 0.2:
                raise RuntimeError("side evaluation failed")
            super().calculate(atoms, properties, system_changes)

    calculator = FailingSideCalculator(np.eye(6), implicit=True)
    dimer = _dimer(tmp_path, calculator)
    center = dimer.atoms.get_positions().copy()

    with pytest.raises(RuntimeError, match="side evaluation failed"):
        dimer._evaluate_derivatives(center.reshape(-1), dimer.n)

    assert dimer.atoms.get_positions() == pytest.approx(center)


def test_nonfinite_finite_difference_force_is_rejected(tmp_path):
    class NonfiniteSideCalculator(_QuadraticCalculator):
        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            assert atoms is not None
            super().calculate(atoms, properties, system_changes)
            if atoms.positions[0, 0] > 0.2:
                self.results["forces"][0, 0] = np.nan

    calculator = NonfiniteSideCalculator(np.eye(6), implicit=True)
    dimer = _dimer(tmp_path, calculator)

    with pytest.raises(ValueError, match="finite-difference force.*non-finite"):
        dimer._evaluate_derivatives(dimer.atoms.positions.reshape(-1), dimer.n)


def test_nonfinite_center_energy_is_rejected(tmp_path):
    class NonfiniteCenterCalculator(_QuadraticCalculator):
        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            assert atoms is not None
            super().calculate(atoms, properties, system_changes)
            if np.isclose(atoms.positions[0, 0], 0.2):
                self.results["energy"] = np.nan

    calculator = NonfiniteCenterCalculator(np.eye(6), implicit=True)
    dimer = _dimer(tmp_path, calculator)

    with pytest.raises(ValueError, match="center energy.*non-finite"):
        dimer._evaluate_derivatives(dimer.atoms.positions.reshape(-1), dimer.n)


def test_gas_auto_mode_preserves_calculator_hvp_path(tmp_path):
    hessian = np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    calculator = _QuadraticCalculator(hessian)
    dimer = _dimer(tmp_path, calculator)

    hn, _, _ = dimer._evaluate_derivatives(dimer.atoms.positions.reshape(-1), dimer.n)

    assert hn == pytest.approx(hessian @ dimer.n)
    assert calculator.hvp_calls == 1
    assert calculator.force_calls == 0


def test_fd_backend_without_recommendation_keeps_legacy_step(tmp_path):
    class OrdinaryCalculator(_QuadraticCalculator):
        def prepare_numerical_derivatives(self):
            return None

    dimer = _dimer(tmp_path, OrdinaryCalculator(np.eye(6), implicit=True))
    assert dimer.derivative_mode == "finite_difference"
    assert dimer.params.delta == 0.005


def test_hvp_mode_does_not_prepare_finite_difference_precision(tmp_path):
    class HVPCalculator(_QuadraticCalculator):
        def prepare_numerical_derivatives(self):
            raise AssertionError("HVP mode must not prepare finite differences")

    dimer = _dimer(tmp_path, HVPCalculator(np.eye(6)))
    dimer.run()
    assert dimer.derivative_mode == "hvp"


@pytest.mark.parametrize("explicit_delta", [None, 0.001])
def test_fd_prepares_backend_before_force_initialization(tmp_path, explicit_delta):
    class PreparedCalculator(_QuadraticCalculator):
        ready = False

        def prepare_numerical_derivatives(self):
            self.ready = True
            return 0.0005

        def calculate(self, *args, **kwargs):
            assert self.ready, "numerical precision must be prepared before force evaluation"
            return super().calculate(*args, **kwargs)

    calculator = PreparedCalculator(np.eye(6), implicit=True)
    options = {"delta": explicit_delta} if explicit_delta is not None else {}
    dimer = _dimer(tmp_path, calculator, n_init="force", **options)
    assert calculator.ready is True
    assert dimer.params.delta == (explicit_delta if explicit_delta is not None else 0.0005)


def test_hvp_callback_is_used_without_calculator_hvp(tmp_path):
    hessian = np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    calculator = _QuadraticCalculator(hessian)
    atoms = _atoms(calculator)
    observed = {}

    def callback(callback_atoms, n):
        observed["atoms"] = callback_atoms
        observed["n"] = np.asarray(n).copy()
        return hessian @ n

    dimer = Dimer(
        str(tmp_path / "callback.out"),
        atoms,
        {
            "use_hvp": True,
            "remove_rigid": False,
            "n_init": "given",
            "n_given": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        hvp_fn=callback,
    )

    hn, _, _ = dimer._evaluate_derivatives(atoms.positions.reshape(-1), dimer.n)

    assert hn == pytest.approx(hessian @ dimer.n)
    assert observed == {"atoms": atoms, "n": pytest.approx(dimer.n)}
    assert calculator.hvp_calls == 0


def test_explicit_implicit_hvp_is_rejected(tmp_path):
    calculator = _QuadraticCalculator(np.eye(6), implicit=True)

    with pytest.raises(ValueError, match="implicit-solvent.*finite-difference"):
        _dimer(tmp_path, calculator, use_hvp=True)

    assert calculator.hvp_calls == 0
    assert calculator.force_calls == 0


@pytest.mark.parametrize("delta", [False, 0.0, -0.1, np.inf, np.nan])
def test_invalid_finite_difference_delta_is_rejected(tmp_path, delta):
    calculator = _QuadraticCalculator(np.eye(6), implicit=True)

    with pytest.raises(ValueError, match="delta must be finite and positive"):
        _dimer(tmp_path, calculator, delta=delta)


@pytest.mark.parametrize("name,value", [("max_iter", 0), ("rot_max_iter", 1.5)])
def test_invalid_iteration_controls_are_rejected(tmp_path, name, value):
    calculator = _QuadraticCalculator(np.eye(6))

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        _dimer(tmp_path, calculator, **{name: value})


@pytest.mark.parametrize(
    "params,message",
    [
        ({"n_init": "unknown"}, "n_init must be"),
        ({"n_init": "given", "n_given": None}, "n_given is required"),
    ],
)
def test_invalid_initial_direction_policy_is_rejected(tmp_path, params, message):
    calculator = _QuadraticCalculator(np.eye(6))

    with pytest.raises(ValueError, match=message):
        _dimer(tmp_path, calculator, **params)


def test_hvp_error_does_not_fall_back_to_finite_difference(tmp_path):
    class BrokenHvpCalculator(_QuadraticCalculator):
        def get_hvp(self, atoms, n):
            self.hvp_calls += 1
            raise RuntimeError("broken HVP")

    calculator = BrokenHvpCalculator(np.eye(6))
    dimer = _dimer(tmp_path, calculator)

    with pytest.raises(RuntimeError, match="broken HVP"):
        dimer._evaluate_derivatives(dimer.atoms.positions.reshape(-1), dimer.n)

    assert calculator.hvp_calls == 1
    assert calculator.force_calls == 0


def test_zero_force_minimum_is_not_reported_as_converged_ts_candidate(tmp_path):
    calculator = _QuadraticCalculator(np.eye(6))
    dimer = _dimer(
        tmp_path,
        calculator,
        use_hvp=False,
    )
    dimer.atoms.set_positions(np.zeros((2, 3)))

    dimer.run()

    output = (tmp_path / "dimer.out").read_text()
    assert "converged TS candidate" not in output
    assert "did not converge to a TS candidate" in output


def test_positive_flip_threshold_cannot_turn_minimum_into_negative_mode(tmp_path):
    calculator = _QuadraticCalculator(np.eye(6))
    dimer = _dimer(
        tmp_path,
        calculator,
        use_hvp=False,
        kappa_to_flip=2.0,
    )
    dimer.atoms.set_positions(np.zeros((2, 3)))

    dimer.run()

    output = (tmp_path / "dimer.out").read_text()
    assert "Negative Mode:          No" in output
    assert "Mode Flip:              Yes" in output
    assert "converged TS candidate" not in output
    assert "did not converge to a TS candidate" in output


def test_stationary_negative_mode_can_converge_as_ts_candidate(tmp_path):
    calculator = _QuadraticCalculator(np.diag([-1.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    dimer = _dimer(tmp_path, calculator, use_hvp=False)
    dimer.atoms.set_positions(np.zeros((2, 3)))

    dimer.run()

    output = (tmp_path / "dimer.out").read_text()
    assert "converged TS candidate at iteration 1" in output
    assert "Post-search frequency/IRC validation remains required" in output


def test_gas_threshold_overrides_and_step_metrics_are_preserved(tmp_path):
    calculator = _QuadraticCalculator(np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    dimer = _dimer(tmp_path, calculator)
    atoms = dimer.atoms
    thresholds = ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
    for name in thresholds:
        setattr(atoms, name, 100.0)
    expected_force_norms = np.linalg.norm(atoms.get_forces(), axis=1)

    dimer.run()

    # Loose caller thresholds must not silently revert to parameter defaults.
    assert "converged TS candidate" in (tmp_path / "dimer.out").read_text()
    assert all(getattr(atoms, name) == 100.0 for name in thresholds)
    assert getattr(atoms, "max_f", None) == pytest.approx(expected_force_norms.max())
    assert getattr(atoms, "rms_f", None) == pytest.approx(
        np.sqrt(np.mean(expected_force_norms**2))
    )
    for name in ("max_dp", "rms_dp"):
        assert np.isfinite(getattr(atoms, name))


def test_finite_difference_rejects_constraints_instead_of_wrong_derivatives(tmp_path):
    calculator = _QuadraticCalculator(np.eye(6), implicit=True)
    atoms = _atoms(calculator)
    from ase.constraints import FixAtoms

    atoms.set_constraint(FixAtoms(indices=[0]))

    with pytest.raises(
        NotImplementedError, match="does not yet support ASE constraints"
    ):
        Dimer(
            str(tmp_path / "constraint.out"),
            atoms,
            {"remove_rigid": False, "n_init": "random"},
        )


def test_existing_trajectory_is_replaced_at_start_of_run(tmp_path):
    calculator = _QuadraticCalculator(np.eye(6))
    dimer = _dimer(tmp_path, calculator, use_hvp=False, save_traj=True)
    dimer.atoms.set_positions(np.zeros((2, 3)))
    trajectory = tmp_path / "dimer_dimer_traj.xyz"
    trajectory.write_text("stale trajectory\n")

    dimer.run()

    assert "stale trajectory" not in trajectory.read_text()
