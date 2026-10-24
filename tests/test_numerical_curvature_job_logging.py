"""Numerical curvature must be visible in the consuming job's .out file."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator.calculator_base import CalcABC
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.frequency import MWFrequency, NonMWFrequency
from maple.function.dispatcher.irc.algorithm import EulerPC, GS, HPC, LQA
from maple.function.dispatcher.jobABC import JobABC
from maple.function.dispatcher.optimization.algorithm.RFO import RFO
from maple.function.dispatcher.scan.scan import Scan
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.algorithm.dimer import Dimer


HESSIAN_WARNING = "WARNING: Numerical Hessian"
DIRECTIONAL_WARNING = "WARNING: Numerical directional curvature"


class Quadratic(CalcABC):
    """Exercise real CalcABC analytic/force-FD dispatch without model assets."""

    implemented_properties = ["energy", "free_energy", "forces"]
    SUPPORTED_HESSIAN_MODES = ("numerical", "analytic")

    def __init__(self, atoms, mode="numerical", *, diagnostics=True, saddle=False):
        super().__init__()
        if mode is not None:
            self.hessian = mode
        self.reference = atoms.positions.copy()
        self.matrix = np.eye(3 * len(atoms))
        if saddle:
            self.matrix[0, 0] = -1.0
        self.diagnostics = diagnostics
        self.hessian_calls = 0
        self.hvp_calls = 0
        self.fail_analytic = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = (self.atoms.positions - self.reference).reshape(-1)
        energy = float(x @ self.matrix @ x / 2)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -(self.matrix @ x).reshape(-1, 3),
        }

    def prepare_numerical_derivatives(self):
        return 0.003

    def _analytic_hessian(self, atoms):
        if self.fail_analytic:
            raise NotImplementedError("analytic Hessian unavailable")
        return self.matrix.copy()

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        result = super().get_hessian(atoms)
        if not self.diagnostics:
            self.last_numerical_hessian_diagnostics = None
        return result

    def get_hvp(self, atoms, n):
        self.hvp_calls += 1
        x = (atoms.positions - self.reference).reshape(-1)
        return self.matrix @ n, -(self.matrix @ x), float(x @ self.matrix @ x / 2)


def make_atoms(mode="numerical", **kwargs):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = Quadratic(atoms, mode, **kwargs)
    for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
        setattr(atoms, name, 0.001)
    return atoms


def read_output(path):
    return path.read_text() if path.exists() else ""


@pytest.mark.parametrize("job_class", [MWFrequency, NonMWFrequency, RFO, GS, HPC, EulerPC, LQA, PRFO])
@pytest.mark.parametrize("mode,diagnostics", [("numerical", True), ("numerical", False), (None, False), ("analytic", True)])
def test_all_hessian_consumers_warn_once_on_actual_numerical_use(
    tmp_path, job_class, mode, diagnostics
):
    atoms = make_atoms(mode, diagnostics=diagnostics, saddle=True)
    output = tmp_path / "consumer.out"
    if job_class in (MWFrequency, NonMWFrequency, PRFO):
        job = job_class(output=str(output), atoms=atoms)
    else:
        job = job_class(atoms, output=str(output))

    def consume():
        if job_class in (MWFrequency, NonMWFrequency):
            return job.get_hessian()
        if job_class is RFO:
            return job._calculate_hessian(atoms)
        if job_class is PRFO:
            return job.check_saddle_curvature(atoms)
        return job._get_hessian_cart()

    assert HESSIAN_WARNING not in read_output(output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        consume()
        consume()
    text = read_output(output)
    assert atoms.calc.hessian_calls == 2
    if mode == "analytic":
        assert HESSIAN_WARNING not in text
        assert "Numerical Hessian diagnostics" not in text
    else:
        assert text.count(HESSIAN_WARNING) == 1
        assert "hessian=numerical" in text
        if diagnostics:
            assert "0.003 Angstrom" in text
            assert text.count("Numerical Hessian diagnostics") == 2
        else:
            assert "not reported" in text
            assert "Numerical Hessian diagnostics" not in text


@pytest.mark.parametrize("method", ["mw", "nonmw"])
@pytest.mark.parametrize("mode", ["numerical", "analytic"])
def test_dispatcher_frequency_writes_warning_to_job_out(tmp_path, method, mode):
    atoms = make_atoms(mode)
    output = tmp_path / "frequency.out"
    Dispatcher()(SimpleNamespace(params={"method": method}), "freq", atoms, str(output))
    text = output.read_text()
    assert "Starting frequency analysis" in text
    assert text.count(HESSIAN_WARNING) == int(mode == "numerical")
    assert atoms.calc.hessian_calls == 1


@pytest.mark.parametrize("mode", ["numerical", "analytic"])
def test_prfo_run_and_fresh_final_certification_share_one_warning(tmp_path, mode):
    atoms = make_atoms(mode, saddle=True)
    output = tmp_path / "prfo.out"
    Dispatcher()(
        SimpleNamespace(params={"method": "prfo", "prfo": {
            "max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02,
        }}),
        "ts", atoms, str(output),
    )
    text = output.read_text()
    assert atoms.calc.hessian_calls == 2
    assert "Final saddle curvature: 1" in text
    assert text.count(HESSIAN_WARNING) == int(mode == "numerical")


@pytest.mark.parametrize("method", ["gs", "hpc", "eulerpc", "lqa"])
@pytest.mark.parametrize("mode", ["numerical", "analytic"])
def test_dispatcher_irc_initial_and_direction_hessians(tmp_path, method, mode):
    atoms = make_atoms(mode, diagnostics=False, saddle=True)
    output = tmp_path / "irc.out"
    Dispatcher()(
        SimpleNamespace(params={
            "method": method,
            "irc": {"max_steps": 0, "write_traj": False},
        }),
        "irc", atoms, str(output),
    )
    assert atoms.calc.hessian_calls > 1
    assert output.read_text().count(HESSIAN_WARNING) == int(mode == "numerical")


def test_scan_suboptimizations_share_warning_but_separate_jobs_do_not(tmp_path):
    atoms = make_atoms(diagnostics=False)
    output = tmp_path / "scan.out"
    scan = Scan(
        str(output), atoms, method="rfo", constraints=[[1, 2, 0.01, 1]],
        params={"rfo": {"max_iter": 1}},
    )
    # Isolate repeated optimizer delegation from unsupported FixInternals FD.
    scan._run_optimizer(atoms)
    first_calls = atoms.calc.hessian_calls
    scan._run_optimizer(atoms)
    assert first_calls > 0
    assert atoms.calc.hessian_calls > first_calls
    assert output.read_text().count(HESSIAN_WARNING) == 1
    other_output = tmp_path / "other.out"
    RFO(atoms, str(other_output))._calculate_hessian(atoms)
    assert other_output.read_text().count(HESSIAN_WARNING) == 1


@pytest.mark.parametrize("use_hvp,implicit", [(False, False), (None, True), (True, False), (None, False)])
def test_dimer_warns_only_force_difference_directional_curvature(tmp_path, use_hvp, implicit):
    atoms = make_atoms("analytic", saddle=True)
    if implicit:
        atoms.calc.solvent_correction = SimpleNamespace(
            mode="fixed", supported_properties={"energy", "forces"},
        )
    output = tmp_path / "dimer.out"
    dimer = Dimer(str(output), atoms, {
        "dimer": {"use_hvp": use_hvp, "delta": 0.007, "remove_rigid": False,
                  "n_init": "given", "n_given": np.eye(9)[0], "max_iter": 1,
                  "save_traj": False, "save_metrics": False},
    })
    assert "WARNING:" not in read_output(output)
    dimer.run()
    dimer._evaluate_derivatives(atoms.positions.reshape(-1), dimer.n)
    text = output.read_text()
    numerical = use_hvp is False or (use_hvp is None and implicit)
    assert text.count(DIRECTIONAL_WARNING) == int(numerical)
    assert HESSIAN_WARNING not in text
    assert atoms.calc.hessian_calls == 0
    if numerical:
        assert "not a full Hessian" in text
        assert "finite_difference" in text
        assert "0.007 Angstrom" in text
        assert atoms.calc.hvp_calls == 0
    else:
        assert atoms.calc.hvp_calls > 0


def test_dimer_reports_backend_selected_step_without_warning_at_construction(tmp_path):
    atoms = make_atoms()
    output = tmp_path / "step.out"
    dimer = Dimer(str(output), atoms, {"dimer": {
        "use_hvp": False, "remove_rigid": False,
        "n_init": "given", "n_given": np.eye(9)[0],
    }})
    assert "WARNING:" not in read_output(output)
    dimer._evaluate_derivatives(atoms.positions.reshape(-1), dimer.n)
    assert "0.003 Angstrom" in output.read_text()


@pytest.mark.parametrize("jobtype,params", [("freq", {}), ("ts", {"method": "prfo", "prfo": {"max_iter": 1}}), ("opt", {"method": "rfo", "rfo": {"max_iter": 1}}), ("irc", {"method": "gs"})])
def test_explicit_analytic_failure_does_not_downgrade_or_warn(tmp_path, jobtype, params):
    atoms = make_atoms("analytic")
    atoms.calc.fail_analytic = True
    output = tmp_path / "failed.out"
    with pytest.raises(NotImplementedError, match="analytic Hessian unavailable"):
        Dispatcher()(SimpleNamespace(params=params), jobtype, atoms, str(output))
    assert atoms.calc.hessian == "analytic"
    assert HESSIAN_WARNING not in read_output(output)
    assert atoms.calc.last_numerical_hessian_diagnostics is None


def test_sp_never_warns_about_configured_but_unused_hessian(tmp_path):
    atoms = make_atoms()
    output = tmp_path / "sp.out"
    Dispatcher()(SimpleNamespace(params={}), "sp", atoms, str(output))
    assert atoms.calc.hessian_calls == 0
    assert "WARNING:" not in output.read_text()


def test_analytic_mode_ignores_stale_numerical_diagnostics(tmp_path):
    class Job(JobABC):
        def run(self):
            pass

    atoms = make_atoms()
    atoms.calc.get_hessian(atoms)
    atoms.calc.hessian = "analytic"
    output = tmp_path / "stale.out"
    Job(str(output)).log_numerical_hessian_diagnostics(atoms.calc)
    assert "Numerical" not in read_output(output)


@pytest.mark.parametrize("metadata,numerical", [
    ({"hessian": "numerical"}, True),
    ({"SUPPORTED_HESSIAN_MODES": ("numerical",)}, True),
    ({"SUPPORTED_HESSIAN_MODES": ("analytic", "numerical")}, False),
    ({}, False),
])
def test_backend_without_diagnostics_attribute_uses_only_declared_mode(
    tmp_path, metadata, numerical
):
    atoms = make_atoms()
    atoms.calc = SimpleNamespace(
        get_hessian=lambda current: np.eye(3 * len(current)), **metadata,
    )
    output = tmp_path / "no-diagnostics.out"
    job = MWFrequency(str(output), atoms)
    job.get_hessian()
    job.get_hessian()
    text = read_output(output)
    assert text.count(HESSIAN_WARNING) == int(numerical)
    if numerical:
        assert "step not reported by backend" in text
    assert "Numerical Hessian diagnostics" not in text
