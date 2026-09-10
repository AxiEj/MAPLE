"""Synthetic task-outcome tests; real-checkpoint CLI evidence is separate."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher import Dispatcher
from maple.function.dispatcher.aimnet2_experimental import (
    OPTIMIZATION_FORCE_TARGET_EV_PER_A,
    WORKFLOW_ID,
)
from maple.function.dispatcher.frequency.frequency import FrequencyParams
from maple.function.dispatcher.frequency.stationary_points import (
    assess_stationary_point,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView


class _Calculator(Calculator):
    implemented_properties = ("energy", "free_energy", "forces")
    experimental_workflow_id = WORKFLOW_ID

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint_path = checkpoint
        self.evaluation_count = 0
        self.hessian_call_count = 0
        self.last_domain_guard = {"gate_passed": True}
        self.provenance = {"claim": "synthetic task test"}
        self.scalar = SimpleNamespace(
            scalar_id="synthetic-total",
            profile_id="synthetic-profile",
            fingerprint_sha256=lambda: "synthetic-fingerprint",
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.evaluation_count += 1
        self.results = {
            "energy": -1.0,
            "free_energy": -1.0,
            "forces": np.zeros((len(atoms), 3)),
        }

    def get_hessian(self, atoms):
        self.hessian_call_count += 1
        self.evaluation_count += 18 * len(atoms) + 2
        return np.eye(3 * len(atoms))


@pytest.fixture
def system(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic-not-a-real-model")
    atoms = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]],
        info={"charge": 0, "mult": 1},
    )
    atoms.calc = _Calculator(checkpoint)
    return atoms, str(tmp_path / "job.out")


def _install_optimizer(monkeypatch, *, converged=True, metrics=True, iterations=1):
    class Optimizer:
        def __init__(self, atoms, output, paras):
            self.atoms = atoms
            self.params = SimpleNamespace(**paras)

        def run(self):
            assert isinstance(self.atoms.calc, LegacyHartreeJobView)
            self.atoms.get_forces()
            for _ in range(iterations):
                self.atoms.positions[0, 0] += 1e-7
                self.atoms.get_forces()
            if metrics:
                self.atoms.max_f = 0.0 if converged else 1.0
                self.atoms.rms_f = 0.0
                self.atoms.max_dp = 0.0
                self.atoms.rms_dp = 0.0
            return self.atoms

        def check_convergence(self, atoms):
            return converged

    for module, name in (
        ("maple.function.dispatcher.optimization.algorithm.LBFGS", "LBFGS"),
        ("maple.function.dispatcher.ts.algorithm.PRFO", "PRFO"),
    ):
        monkeypatch.setattr(importlib.import_module(module), name, Optimizer)


def _read(output):
    return json.loads(Path(output + ".experimental.json").read_text())


@pytest.mark.parametrize("iterations,max_iter", [(1, 3), (3, 3)])
def test_opt_counts_initial_iterations_and_forced_final_replay(
    system, monkeypatch, iterations, max_iter
):
    atoms, output = system
    raw = atoms.calc
    _install_optimizer(monkeypatch, iterations=iterations)
    result = Dispatcher()(
        SimpleNamespace(
            params={"method": "lbfgs", "max_iter": max_iter, "level": "superloose"}
        ),
        "opt",
        atoms,
        output,
    )
    assert result["run_status"] == "converged"
    assert result["scientific_admission"] is False
    assert result["optimizer_initial_evaluation_count"] == 1
    assert result["optimizer_iteration_evaluation_count"] == iterations
    assert result["postflight_evaluation_count"] == 1
    assert result["total_evaluation_count"] == iterations + 2
    assert result["evaluation_budget"] == max_iter + 2
    assert atoms.calc is raw
    from maple.function.calculator.calculator_base import EV2HARTREE

    assert atoms.f_max_th / EV2HARTREE == pytest.approx(
        OPTIMIZATION_FORCE_TARGET_EV_PER_A
    )


@pytest.mark.parametrize("metrics", [True, False])
def test_nonconvergence_and_stale_metrics_are_failures(system, monkeypatch, metrics):
    atoms, output = system
    raw = atoms.calc
    for name in ("max_f", "rms_f", "max_dp", "rms_dp"):
        setattr(atoms, name, 0.0)
    _install_optimizer(monkeypatch, converged=False, metrics=metrics, iterations=2)
    with pytest.raises(RuntimeError, match="did not converge"):
        Dispatcher()(
            SimpleNamespace(params={"method": "lbfgs", "max_iter": 2}),
            "opt",
            atoms,
            output,
        )
    result = _read(output)
    assert result["run_status"] == "failed"
    assert result["total_evaluation_count"] == 4
    assert result["postflight_evaluation_count"] == 1
    assert atoms.calc is raw


@pytest.mark.parametrize("max_iter", [0, -1, True, 1.5])
def test_iteration_budget_must_be_positive_integer(system, max_iter):
    atoms, output = system
    with pytest.raises(ValueError, match="max_iter"):
        Dispatcher()(
            SimpleNamespace(params={"method": "lbfgs", "max_iter": max_iter}),
            "opt",
            atoms,
            output,
        )
    assert _read(output)["run_status"] == "failed"
    assert atoms.calc.evaluation_count == 0


@pytest.mark.parametrize(
    "modes,passes",
    [([-500, 800, 1000], True), ([500, 800, 1000], False), ([-500, -800, 1000], False)],
)
def test_ts_requires_final_shared_saddle_assessment(system, monkeypatch, modes, passes):
    atoms, output = system
    _install_optimizer(monkeypatch)
    calls = []

    class Frequency:
        def __init__(self, output, atoms, paras):
            assert not isinstance(atoms.calc, LegacyHartreeJobView)
            self.params = FrequencyParams(stationary_point=paras["stationary_point"])
            self.atoms = atoms

        def run(self):
            self.atoms.calc.get_hessian(self.atoms)
            calls.append(True)
            assess_stationary_point(
                modes, target="transition_state", imaginary_threshold_cm1=50.0
            )

    monkeypatch.setattr(
        importlib.import_module("maple.function.dispatcher.frequency"),
        "Frequency",
        Frequency,
    )
    run = lambda: Dispatcher()(
        SimpleNamespace(params={"method": "prfo", "max_iter": 2}), "ts", atoms, output
    )
    if passes:
        assert run()["run_status"] == "converged"
    else:
        with pytest.raises(ValueError, match="exactly one"):
            run()
        assert _read(output)["run_status"] == "failed"
    assert calls == [True]


@pytest.mark.parametrize(
    "params",
    [
        {"treat_imag_as_real": True},
        {"stationary_point": "transition_state"},
        {"stationarity_tolerance_ev_per_a": 0.1},
        {"rigid_mode_tolerance_cm1": 50.0},
        {"transition_state_imaginary_threshold_cm1": 1.0},
    ],
)
def test_freq_does_not_relax_stationarity_or_hide_negative_modes(system, params):
    atoms, output = system
    with pytest.raises(ValueError):
        Dispatcher()(
            SimpleNamespace(params={"method": "mw", **params}), "freq", atoms, output
        )
    assert _read(output)["run_status"] == "failed"


def test_direct_dispatch_rejects_unsupported_tasks(system):
    atoms, output = system
    with pytest.raises(ValueError, match="supports SP"):
        Dispatcher()(SimpleNamespace(params={"method": "neb"}), "ts", atoms, output)
    assert _read(output)["total_evaluation_count"] == 0


def test_optimizer_without_evaluated_iteration_is_not_success(system, monkeypatch):
    atoms, output = system
    _install_optimizer(monkeypatch, iterations=0)
    with pytest.raises(RuntimeError, match="no finite geometry displacement"):
        Dispatcher()(
            SimpleNamespace(params={"method": "lbfgs", "max_iter": 2}),
            "opt",
            atoms,
            output,
        )
    assert _read(output)["run_status"] == "failed"
    assert _read(output)["optimizer_iteration_evaluation_count"] == 0


@pytest.mark.parametrize("failure", ["nan_energy", "nan_force", "force_shape", "guard"])
def test_sp_validates_total_results_and_guard_before_success(
    system, monkeypatch, failure
):
    atoms, output = system
    calculator = atoms.calc
    original = calculator.calculate

    def calculate(atoms=None, properties=("energy",), system_changes=all_changes):
        original(atoms, properties, system_changes)
        if failure == "nan_energy":
            calculator.results["energy"] = np.nan
        elif failure == "nan_force":
            calculator.results["forces"][0, 0] = np.nan
        elif failure == "force_shape":
            calculator.results["forces"] = np.zeros((1, 3))
        else:
            calculator.last_domain_guard = {"gate_passed": False}

    monkeypatch.setattr(calculator, "calculate", calculate)
    monkeypatch.setattr(
        Dispatcher, "_dispatch", lambda self, *args: atoms.get_potential_energy()
    )
    with pytest.raises((ValueError, RuntimeError)):
        Dispatcher()(SimpleNamespace(params={}), "sp", atoms, output)
    result = _read(output)
    assert result["run_status"] == "failed"
    assert result["scientific_admission"] is False


def test_ts_budget_is_derived_from_iterations_recalcs_and_atom_count():
    from maple.function.dispatcher.aimnet2_experimental import _work_budget

    assert _work_budget("ts", {"max_iter": 24, "recalc": 6}, 4) == (588, 5)


def test_rejected_frequency_index_retains_actual_spectrum_without_an_extra_hessian(
    system, monkeypatch
):
    from maple.function.dispatcher.frequency.normal_modes import (
        rigid_body_subspaces,
        EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1,
    )

    atoms, output = system
    calculator = atoms.calc
    expected = np.array([-500.0, 800.0, 1000.0])
    subspace = rigid_body_subspaces(
        atoms.get_masses(), atoms.positions
    ).vibrational_basis_mass_weighted
    eigenvalues = (
        np.sign(expected) * (expected / EV_PER_ANGSTROM2_AMU_TO_WAVENUMBER_CM1) ** 2
    )
    massroot = np.sqrt(np.repeat(atoms.get_masses(), 3))
    hessian = (
        massroot[:, None]
        * (subspace @ np.diag(eigenvalues) @ subspace.T)
        * massroot[None, :]
    )

    def get_hessian(atoms):
        calculator.hessian_call_count += 1
        calculator.evaluation_count += 18 * len(atoms) + 2
        calculator.last_hessian_diagnostics = {
            "status": "passed",
            "returned_hessian_eV_per_A2": hessian.tolist(),
            "sidecar_path": "synthetic-hessian-record.json",
            "schema_version": "synthetic",
        }
        return hessian

    monkeypatch.setattr(calculator, "get_hessian", get_hessian, raising=False)
    with pytest.raises(ValueError, match="minimum requires"):
        Dispatcher()(SimpleNamespace(params={"method": "mw"}), "freq", atoms, output)
    result = _read(output)
    assert result["run_status"] == "failed"
    np.testing.assert_allclose(
        result["vibrational_frequencies_cm1"], expected, atol=1e-8
    )
    assert result["hessian_call_count"] == 1
    assert (
        result["last_hessian_diagnostic"]["sidecar_path"]
        == "synthetic-hessian-record.json"
    )


@pytest.mark.parametrize("task,method", [("opt", "lbfgs"), ("ts", "prfo")])
def test_optimizer_cannot_fake_convergence_with_same_geometry_replays(
    system, monkeypatch, task, method
):
    atoms, output = system

    class NoMovement:
        def __init__(self, atoms, output, paras):
            self.atoms = atoms

        def run(self):
            self.atoms.get_forces()
            raw = self.atoms.calc._calculator
            raw.calculate(self.atoms, properties=("forces",))
            for name in ("max_f", "rms_f", "max_dp", "rms_dp"):
                setattr(self.atoms, name, 0.0)

        def check_convergence(self, atoms):
            return True

    module, name = (
        ("maple.function.dispatcher.optimization.algorithm.LBFGS", "LBFGS")
        if task == "opt"
        else ("maple.function.dispatcher.ts.algorithm.PRFO", "PRFO")
    )
    monkeypatch.setattr(importlib.import_module(module), name, NoMovement)
    with pytest.raises(RuntimeError, match="no finite geometry displacement"):
        Dispatcher()(
            SimpleNamespace(params={"method": method, "max_iter": 2}),
            task,
            atoms,
            output,
        )
    assert _read(output)["run_status"] == "failed"
    assert _read(output)["maximum_total_displacement_A"] == 0


def test_v2_ts_work_budget_uses_its_own_four_step_numerical_cost():
    from maple.function.dispatcher.aimnet2_experimental import _work_budget

    assert _work_budget(
        "ts", {"max_iter": 24, "recalc": 6}, 4, hessian_evaluations_per_atom=24
    ) == (708, 5)


def test_fresh_replay_must_also_satisfy_stricter_rms_force_threshold(
    system, monkeypatch
):
    from maple.function.calculator.calculator_base import EV2HARTREE

    atoms, output = system
    calculator = atoms.calc
    _install_optimizer(monkeypatch, iterations=1)
    original = calculator.calculate

    def calculate(atoms=None, properties=("energy",), system_changes=all_changes):
        original(atoms, properties, system_changes)
        if calculator.evaluation_count == 3:
            calculator.results["forces"][:] = 1e-6

    monkeypatch.setattr(calculator, "calculate", calculate)
    with pytest.raises(RuntimeError, match="RMS"):
        Dispatcher()(
            SimpleNamespace(
                params={"method": "lbfgs", "max_iter": 2, "f_rms_th": 1e-8 * EV2HARTREE}
            ),
            "opt",
            atoms,
            output,
        )
    result = _read(output)
    assert result["run_status"] == "failed"
    assert result["maximum_force_eV_per_A"] < 1e-5
    assert result["rms_force_eV_per_A"] > 1e-8
