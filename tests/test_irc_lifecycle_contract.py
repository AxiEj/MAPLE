"""RED contracts for truthful HPC/LQA/EulerPC IRC lifecycle behavior.

This file deliberately imports the immutable 21dbbb57 baseline.  It uses only a
small analytic ASE calculator; it is not a model or qualification test.
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes
from types import SimpleNamespace

from maple.function.calculator.calculator_base import CalcABC
from maple.function.dispatcher.irc.algorithm import EulerPC, HPC, LQA
from maple.function.dispatcher.irc.irc import IRC

METHODS = {"hpc": HPC, "lqa": LQA, "eulerpc": EulerPC}




class InternalDoubleWell(CalcABC):
    """Two-atom internal double well with exact rigid translation zero modes."""

    implemented_properties = ["energy", "free_energy", "forces"]
    SUPPORTED_HESSIAN_MODES = ("analytic",)

    def __init__(self, atoms: Atoms, *, real_force_bias: float = 0.0):
        super().__init__()
        self.hessian = "analytic"
        self.r0 = float(atoms.positions[1, 0] - atoms.positions[0, 0])
        self.real_force_bias = float(real_force_bias)
        self.energy_calls = 0
        self.force_calls = 0
        self.hessian_calls = 0

    def _values(self, atoms: Atoms):
        s = float((atoms.positions[1, 0] - atoms.positions[0, 0]) - self.r0)
        # E(s) = -s^2/2 + s^4/4.  At s=0 the internal mode is negative.
        e = -0.5 * s * s + 0.25 * s**4
        deds = -s + s**3
        f = np.zeros((len(atoms), 3), dtype=float)
        f[0, 0], f[1, 0] = deds, -deds
        # Optional uniform bias is deliberately a real force at the evaluated point.
        f[:, 1] += self.real_force_bias
        return e, f, s

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, _ = self._values(self.atoms)
        self.energy_calls += 1
        self.force_calls += 1
        self.results = {"energy": e, "free_energy": e, "forces": f}

    def _analytic_hessian(self, atoms: Atoms):
        self.hessian_calls += 1
        _, _, s = self._values(atoms)
        k = -1.0 + 3.0 * s * s
        h = np.zeros((3 * len(atoms), 3 * len(atoms)), dtype=float)
        h[0, 0] = h[3, 3] = k
        h[0, 3] = h[3, 0] = -k
        return h


def make_ts(*, bias: float = 0.0, implicit: bool = False) -> Atoms:
    atoms = Atoms("HH", positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    atoms.calc = InternalDoubleWell(atoms, real_force_bias=bias)
    if implicit:
        atoms.calc.solvent_correction = SimpleNamespace(mode="fixed")
    return atoms


def _params(method: str, max_steps: int = 0) -> dict:
    values = {
        "max_steps": max_steps,
        "write_traj": False,
        "print_each": False,
        "step_length_bohr": 0.1,
        "f_max_th": 1e-3,
        "f_rms_th": 1e-3,
    }
    if method == "hpc":
        values.update(euler_n=1, mbs_max_k=2, mbs_points=2, mbs_tol=1e9)
    elif method == "lqa":
        values.update(euler_n=1)
    else:
        values.update(max_pred_steps=1, loose_cycles=0, mbs_max_k=2, mbs_points=2, mbs_tol=1e9)
    return {"irc": values}


def _path_images(result: dict) -> list[np.ndarray]:
    """Read only public result data; tolerate the planned additive field names."""
    summary = result["summary"]
    raw = summary.get("images", summary.get("records", summary.get("trajectory")))
    assert raw is not None, "IRC summary must expose its ordered physical images"
    images = []
    for item in raw:
        if isinstance(item, Atoms):
            images.append(item.get_positions().copy())
        elif isinstance(item, dict):
            value = item.get("x", item.get("positions", item.get("atoms")))
            images.append(value.get_positions().copy() if isinstance(value, Atoms) else np.asarray(value))
        else:
            images.append(np.asarray(item))
    return images


@pytest.mark.parametrize("method", METHODS)
def test_zero_budget_keeps_exact_ts_and_uses_one_initial_hessian(tmp_path, method):
    atoms = make_ts()
    before = atoms.positions.copy()
    result = METHODS[method](atoms, str(tmp_path / f"{method}.out"), paras=_params(method)).run()
    assert np.array_equal(atoms.positions, before)
    assert atoms.calc.hessian_calls == 1
    images = _path_images(result)
    assert len(images) == 1
    assert np.array_equal(images[0], before)
    assert result["summary"]["ts_index"] == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_mass", [0.0, -1.0, np.nan])
def test_invalid_masses_are_rejected_instead_of_replaced(tmp_path, method, bad_mass):
    atoms = make_ts()
    atoms.set_masses([bad_mass, 1.0])
    with pytest.raises(ValueError, match="(?i)mass|finite|positive"):
        METHODS[method](atoms, str(tmp_path / f"{method}.out"), paras=_params(method)).run()


@pytest.mark.parametrize("method", METHODS)
def test_exact_physical_order_and_ts_index(tmp_path, method):
    """Merged order is reversed(backward) + exact input TS + forward."""
    atoms = make_ts()
    exact_ts = atoms.positions.copy()
    result = METHODS[method](atoms, str(tmp_path / f"{method}.out"), paras=_params(method, 1)).run()
    images = _path_images(result)
    backward_count = len(result["backward"]["records"])
    assert result["summary"]["ts_index"] == backward_count
    assert sum(np.array_equal(image, exact_ts) for image in images) == 1
    assert np.array_equal(images[backward_count], exact_ts)
    if backward_count:
        assert images[0][1, 0] < exact_ts[1, 0]
    if len(images) > backward_count + 1:
        assert images[-1][1, 0] > exact_ts[1, 0]


@pytest.mark.parametrize("method", METHODS)
def test_public_gas_wrapper_preserves_existing_in_place_ownership(tmp_path, method):
    atoms = make_ts()
    before = atoms.positions.copy()
    IRC(_params(method, 1), str(tmp_path / f"{method}.out"), atoms, method).run()
    # Gas algorithms historically own and update the caller object in place.
    assert not np.array_equal(atoms.positions, before)




def test_eulerpc_predicted_gradient_cannot_claim_real_force_convergence(tmp_path):
    atoms = make_ts(bias=0.05)
    result = EulerPC(
        atoms,
        str(tmp_path / "eulerpc.out"),
        paras={"eulerpc": {
            **_params("eulerpc", 1)["irc"],
            "f_max_th": 1e-3,
            "f_rms_th": 1e-3,
        }},
    ).run()
    # Any predicted-gradient shortcut must yield to the real final force.
    for side in ("forward", "backward"):
        record = result[side]["records"][-1]
        assert record["maxG"] >= 0.05
        assert record.get("converged") is False
    assert result["summary"].get("status") == "bounded_nonconverged"

@pytest.mark.parametrize("method", METHODS)
def test_zero_negative_mode_is_rejected_without_moving_caller(tmp_path, method):
    atoms = make_ts()
    before = atoms.positions.copy()

    def positive_hessian(_atoms):
        atoms.calc.hessian_calls += 1
        h = np.zeros((6, 6), dtype=float)
        h[0, 0] = h[3, 3] = 1.0
        h[0, 3] = h[3, 0] = -1.0
        return h

    atoms.calc._analytic_hessian = positive_hessian
    with pytest.raises(RuntimeError, match="(?i)no negative"):
        METHODS[method](atoms, str(tmp_path / f"{method}.out"), paras=_params(method)).run()
    assert np.array_equal(atoms.positions, before)

@pytest.mark.parametrize("method", METHODS)
def test_written_full_trajectory_is_backward_ts_forward(tmp_path, method):
    from ase.io import read

    atoms = make_ts()
    exact_ts = atoms.positions.copy()
    params = _params(method, 1)
    params["irc"]["write_traj"] = True
    output = tmp_path / f"{method}.out"
    result = METHODS[method](atoms, str(output), paras=params).run()
    written = [image.get_positions() for image in read(tmp_path / f"{method}_full.xyz", ":")]
    backward = [record["x"] for record in result["backward"]["records"]]
    forward = [record["x"] for record in result["forward"]["records"]]
    expected = list(reversed(backward)) + [exact_ts] + forward
    assert len(written) == len(expected)
    for observed, wanted in zip(written, expected):
        assert np.allclose(observed, wanted, atol=5e-9, rtol=0.0)


@pytest.mark.parametrize("method", METHODS)
def test_two_step_budget_reports_actual_macro_counts(tmp_path, method):
    atoms = make_ts()
    result = METHODS[method](
        atoms,
        str(tmp_path / f"{method}.out"),
        paras=_params(method, 2),
    ).run()
    assert result["forward"]["iterations"] == 2
    assert result["backward"]["iterations"] == 2
    assert result["iterations"] == 4
    assert result["forward"]["termination_reason"] == "maximum_steps"
    assert result["backward"]["termination_reason"] == "maximum_steps"


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_start_hessian_refuses_and_restores_caller(tmp_path, method):
    atoms = make_ts()
    before = atoms.positions.copy()

    def nonfinite_hessian(_atoms):
        atoms.calc.hessian_calls += 1
        return np.full((6, 6), np.nan)

    atoms.calc._analytic_hessian = nonfinite_hessian
    with pytest.raises(ValueError, match="(?i)finite"):
        METHODS[method](
            atoms,
            str(tmp_path / f"{method}.out"),
            paras=_params(method),
        ).run()
    assert np.array_equal(atoms.positions, before)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["energy", "forces"])
def test_nonfinite_start_energy_or_forces_refuse_and_restore_caller(
    tmp_path, method, kind
):
    atoms = make_ts()
    before = atoms.positions.copy()
    original_values = atoms.calc._values

    def nonfinite_values(changed):
        energy, forces, coordinate = original_values(changed)
        if kind == "energy":
            energy = np.nan
        else:
            forces[:] = np.nan
        return energy, forces, coordinate

    atoms.calc._values = nonfinite_values
    with pytest.raises(ValueError, match="(?i)finite"):
        METHODS[method](
            atoms,
            str(tmp_path / f"{method}.out"),
            paras=_params(method),
        ).run()
    assert np.array_equal(atoms.positions, before)


@pytest.mark.parametrize("method", METHODS)
def test_mutating_hessian_provider_refuses_and_restores_caller(tmp_path, method):
    atoms = make_ts()
    before = atoms.positions.copy()

    def mutating_hessian(changed):
        changed.positions[:] += 1.0
        return np.eye(6)

    atoms.calc._analytic_hessian = mutating_hessian
    with pytest.raises(ValueError, match="(?i)mutated scientific Atoms state"):
        METHODS[method](
            atoms,
            str(tmp_path / f"{method}.out"),
            paras=_params(method),
        ).run()
    assert np.array_equal(atoms.positions, before)


@pytest.mark.parametrize("method", METHODS)
def test_failed_run_restores_published_calculator_cache_in_place(tmp_path, method):
    atoms = make_ts()
    calculator = atoms.calc
    cached_atoms = atoms.copy()
    cached_results = {
        "energy": -7.0,
        "nested": {"values": np.array([1.0, 2.0])},
    }
    cached_solvation = object()
    cached_correction = object()
    calculator.atoms = cached_atoms
    calculator.results = cached_results
    calculator.solvation_result = cached_solvation
    calculator._evaluated_correction = cached_correction
    initial_positions = atoms.positions.copy()
    original_values = calculator._values

    def fail_after_displacement(changed):
        if not np.array_equal(changed.positions, initial_positions):
            cached_results["nested"]["values"][0] = np.nan
            calculator.solvation_result = object()
            calculator._evaluated_correction = object()
            raise RuntimeError("injected displaced-point failure")
        return original_values(changed)

    calculator._values = fail_after_displacement
    with pytest.raises(RuntimeError, match="injected displaced-point failure"):
        METHODS[method](
            atoms,
            str(tmp_path / f"{method}.out"),
            paras=_params(method, 1),
        ).run()

    assert np.array_equal(atoms.positions, initial_positions)
    assert calculator.atoms is cached_atoms
    assert calculator.results is cached_results
    np.testing.assert_array_equal(
        calculator.results["nested"]["values"], np.array([1.0, 2.0])
    )
    assert calculator.solvation_result is cached_solvation
    assert calculator._evaluated_correction is cached_correction


@pytest.mark.parametrize("method", METHODS)
def test_successful_gas_run_keeps_final_endpoint_cache(tmp_path, method):
    atoms = make_ts()
    calculator = atoms.calc
    cached_atoms = atoms.copy()
    cached_results = {"energy": -7.0}
    calculator.atoms = cached_atoms
    calculator.results = cached_results

    result = METHODS[method](
        atoms,
        str(tmp_path / f"{method}.out"),
        paras=_params(method, 1),
    ).run()

    endpoint = result["backward"]["records"][-1]
    assert calculator.atoms is not cached_atoms
    assert calculator.results is not cached_results
    assert calculator.results["energy"] == pytest.approx(endpoint["E"])
    np.testing.assert_array_equal(calculator.atoms.positions, endpoint["x"])
    np.testing.assert_array_equal(atoms.positions, endpoint["x"])


@pytest.mark.parametrize("method", METHODS)
def test_path_summary_rows_follow_exact_ordered_path(tmp_path, method):
    atoms = make_ts()
    result = METHODS[method](
        atoms,
        str(tmp_path / f"{method}.out"),
        paras=_params(method, 2),
    ).run()

    ordered = result["ordered_path"]
    summary = result["summary"]
    assert summary["ts_index"] == len(result["backward"]["records"])
    assert len(summary["images"]) == len(summary["merged_rows"]) == len(ordered)
    assert (
        len(summary["energies"])
        == len(summary["relative_energies_kcal_mol"])
        == len(ordered)
    )
    for index, (record, image, energy, relative, row) in enumerate(
        zip(
            ordered,
            summary["images"],
            summary["energies"],
            summary["relative_energies_kcal_mol"],
            summary["merged_rows"],
        )
    ):
        np.testing.assert_array_equal(image, record["x"])
        assert energy == record["E"]
        assert relative == pytest.approx(
            (record["E"] - summary["E_ref"]) * 627.509474
        )
        assert f"{record['E']:14.6f}" in row
        assert f"{relative:12.6f}" in row
        assert ("<= TS" in row) is (index == summary["ts_index"])


@pytest.mark.parametrize("method", METHODS)
def test_path_summary_is_written_for_zero_step_run(tmp_path, method):
    output = tmp_path / f"{method}.out"
    result = METHODS[method](
        make_ts(),
        str(output),
        paras=_params(method, 0),
    ).run()

    text = output.read_text()
    assert f"{method.upper()}-IRC PATH SUMMARY" in text
    assert "Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G)" in text
    assert text.count("<= TS") == 1
    assert result["summary"]["merged_rows"] == [
        line + "\n" for line in text.splitlines() if "<= TS" in line
    ]
