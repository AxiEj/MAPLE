from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from tools.route2_release import run_pure_mace_polar_workflow_canary as canary


class _Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(np.sum(atoms.positions**2)),
            "forces": -2.0 * atoms.positions,
        }


def test_ammonia_height_is_invariant_to_proper_rotation_and_translation():
    positions = np.array([[0, 0, 0.2], [1, 0, 0], [0, 1, 0], [-1, -1, 0]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert canary.signed_ammonia_height(positions) == pytest.approx(0.2)
    assert canary.signed_ammonia_height(
        positions @ rotation + [3, 2, 1]
    ) == pytest.approx(0.2)


def test_ammonia_height_distinguishes_the_two_inversion_endpoints():
    positions = np.array([[0, 0, 0.2], [1, 0, 0], [0, 1, 0], [-1, -1, 0]])
    plus = canary.signed_ammonia_height(positions)
    positions[0, 2] *= -1
    assert plus * canary.signed_ammonia_height(positions) < 0.0


def test_ammonia_height_rejects_a_degenerate_hydrogen_plane():
    with pytest.raises(ValueError, match="degenerate"):
        canary.signed_ammonia_height(np.zeros((4, 3)))


@pytest.mark.parametrize("normal_termination", [False, True])
def test_relaxation_requires_actual_convergence_not_just_returned_atoms(
    tmp_path, monkeypatch, normal_termination
):
    atoms = Atoms("H2", positions=[[1, 0, 0], [-1, 0, 0]])
    atoms.calc = _Quadratic()
    output = tmp_path / "opt.out"
    monkeypatch.setattr(canary, "_command", lambda *args, **kwargs: object())

    def dispatch(command, current, path):
        current.positions *= 0.5
        for name in ("max_f", "rms_f", "max_dp", "rms_dp"):
            setattr(current, name, 0.0)
        for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
            setattr(current, name, 0.01)
        path.write_text(
            "LBFGS converged at iteration 1."
            if normal_termination
            else "LBFGS did NOT converge after 1 iterations."
        )

    monkeypatch.setattr(canary, "_dispatch", dispatch)
    result = canary._relax(
        {},
        atoms,
        output,
        {"method": "lbfgs", "max_iter": 1, "max_step": 0.1, "level": "extratight"},
    )
    assert result["optimizer_converged"] is normal_termination
    assert result["final_energy_eV"] < result["initial_energy_eV"]


def test_canary_rejects_freq_without_optimized_water(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "canary",
            "--checkpoint",
            "unused",
            "--output-dir",
            "unused",
            "--tasks",
            "freq",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        canary.main()
    assert exc.value.code == 2


def test_frequency_failure_retains_completed_optimization_and_trace(
    monkeypatch, tmp_path
):
    protocol = json.loads(canary.PROTOCOL.read_text())
    fake = SimpleNamespace(last_hessian_evaluation=None)
    monkeypatch.setattr(canary, "_attach", lambda *args: fake)
    monkeypatch.setattr(canary, "_identity", lambda *args: {"test_only": True})
    monkeypatch.setattr(canary, "_command", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        canary,
        "_relax",
        lambda *args: {
            "optimizer_converged": True,
            "initial_energy_eV": 1.0,
            "final_energy_eV": 0.0,
        },
    )

    def fail(*args):
        raise RuntimeError("deliberate Hessian spike")

    monkeypatch.setattr(canary, "_dispatch", fail)
    result = canary._water_workflows(protocol, tmp_path, True)
    assert result["opt"]["pass"] is True
    assert result["freq"]["pass"] is False
    assert result["freq"]["failure"]["stage"] == "freq"
    assert "deliberate Hessian spike" in result["freq"]["failure"]["traceback"]
    assert json.loads((tmp_path / "water-progress.json").read_text()) == result


def test_ts_initial_hessian_failure_retains_stage_geometry_and_trace(
    monkeypatch, tmp_path
):
    protocol = json.loads(canary.PROTOCOL.read_text())
    fake = SimpleNamespace(last_hessian_evaluation=None)
    monkeypatch.setattr(canary, "_attach", lambda *args: fake)
    monkeypatch.setattr(canary, "_identity", lambda *args: {"test_only": True})

    def fail(*args):
        raise RuntimeError("initial stencil failed")

    monkeypatch.setattr(canary, "_mode_record", fail)
    result = canary._ts(protocol, tmp_path)
    assert result["pass"] is False
    assert result["stage"] == "initial-index-check"
    assert result["failure"]["stage"] == "initial-index-check"
    assert len(result["failure"]["positions_angstrom"]) == 4
    assert "initial stencil failed" in result["failure"]["traceback"]
    assert json.loads((tmp_path / "ts-progress.json").read_text()) == result
