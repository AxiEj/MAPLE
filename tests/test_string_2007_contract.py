from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import (
    PRFOResult,
    PathStageResult,
    RunStatus,
)
from maple.function.dispatcher.ts.algorithm.string import (
    GSM,
    STRING_PARAMETER_METADATA,
)
from maple.function.dispatcher.ts.algorithm import string as string_module


prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")








class DoubleWellCalculator(Calculator):
    """Analytic V=(x^2-1)^2+2y^2 in one atom's Cartesian container."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, _ = np.asarray(atoms.positions[0], dtype=np.float64)
        energy = (x * x - 1.0) ** 2 + 2.0 * y * y
        forces = np.array(
            [[-4.0 * x * (x * x - 1.0), -4.0 * y, 0.0]],
            dtype=np.float64,
        )
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": forces,
        }


class InvalidCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": np.nan,
            "free_energy": np.nan,
            "forces": np.zeros((len(atoms), 3)),
        }


class BrokenCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        raise RuntimeError("programmer defect")


def image(x: float, y: float, calculator=None) -> Atoms:
    atoms = Atoms("H", positions=[[x, y, 0.0]])
    atoms.calc = calculator or DoubleWellCalculator()
    atoms.info.update({"source_id": "analytic-double-well", "topology_id": "H-1"})
    atoms.set_masses([1.25])
    atoms.f_max_th = 1.1e-7
    atoms.f_rms_th = 2.2e-7
    atoms.dp_max_th = 3.3e-7
    atoms.dp_rms_th = 4.4e-7
    return atoms


def path(points) -> list[Atoms]:
    calculator = DoubleWellCalculator()
    return [image(x, y, calculator) for x, y in points]


def independent_equal_arc(points: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] == 0.0:
        return points.copy()
    keep = np.concatenate(([True], np.diff(cumulative) > 0.0))
    target = np.linspace(0.0, cumulative[-1], len(points))
    result = np.empty_like(points)
    for axis in range(points.shape[1]):
        result[:, axis] = np.interp(
            target,
            cumulative[keep],
            points[keep, axis],
        )
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def independent_force(points: np.ndarray) -> np.ndarray:
    result = np.zeros_like(points)
    x = points[:, 0]
    result[:, 0] = -4.0 * x * (x * x - 1.0)
    result[:, 1] = -4.0 * points[:, 1]
    return result


def job(tmp_path, images, **overrides) -> GSM:
    parameters = {
        "n_images": len(images),
        "refine": "cistring",
        "max_iter_relax": 1,
        "max_iter_cistring": 1,
        "lbfgs_memory": 0,
        "lbfgs_curvature": 10.0,
        "lbfgs_max_step": 10.0,
        "reparam_every": 1,
        "string_f_max_th": 1.0e-14,
        "string_f_rms_th": 1.0e-14,
        "cistring_f_max_th": 1.0e-14,
        "cistring_f_rms_th": 1.0e-14,
    }
    parameters.update(overrides)
    return GSM(
        str(tmp_path / "string.out"),
        images[0],
        images[-1],
        paras={"string": parameters},
    )


def positions(images) -> np.ndarray:
    return np.array([atoms.positions.reshape(-1) for atoms in images])


def test_ordinary_uses_full_force_then_equal_arc_reparameterization(tmp_path):
    images = path([(-1.0, 0.0), (-0.55, 0.16), (0.15, 0.10), (1.0, 0.0)])
    runner = job(tmp_path, images)
    initial = positions(images)
    evolved = initial.copy()
    evolved[1:-1] += 0.1 * independent_force(initial)[1:-1]
    expected = independent_equal_arc(evolved)

    result = runner.relax_string(images)

    np.testing.assert_allclose(positions(result.images), expected, atol=1.0e-13, rtol=0.0)
    np.testing.assert_array_equal(result.images[0].positions, images[0].positions)
    np.testing.assert_array_equal(result.images[-1].positions, images[-1].positions)


def test_ci_formula_freezes_tau0_and_anchors_latest_ci_during_split_reparam(tmp_path):
    images = path(
        [
            (-1.0, 0.0),
            (-0.74, 0.08),
            (-0.31, 0.15),
            (0.08, 0.11),
            (0.52, 0.08),
            (0.78, 0.02),
            (1.0, 0.0),
        ]
    )
    runner = job(tmp_path, images, max_iter_cistring=2)
    initial = positions(images)
    ci_index = 3
    tau0 = initial[ci_index + 1] - initial[ci_index - 1]
    tau0 /= np.linalg.norm(tau0)
    expected = initial.copy()
    first_ci_after_ode = None
    for _ in range(2):
        forces = independent_force(expected)
        evolved = expected.copy()
        evolved[1:-1] += 0.1 * forces[1:-1]
        ci_force = forces[ci_index] - 2.0 * np.dot(forces[ci_index], tau0) * tau0
        evolved[ci_index] = expected[ci_index] + 0.1 * ci_force
        if first_ci_after_ode is None:
            first_ci_after_ode = evolved[ci_index].copy()
        left = independent_equal_arc(evolved[: ci_index + 1])
        right = independent_equal_arc(evolved[ci_index:])
        expected[: ci_index + 1] = left
        expected[ci_index:] = right
        expected[ci_index] = evolved[ci_index]

    result = runner.relax_cistring(images, ci_index=ci_index)

    np.testing.assert_allclose(positions(result.images), expected, atol=1.0e-12, rtol=0.0)
    np.testing.assert_array_equal(result.images[0].positions, images[0].positions)
    np.testing.assert_array_equal(result.images[-1].positions, images[-1].positions)
    assert not np.array_equal(first_ci_after_ode, initial[ci_index])


def test_global_residual_checks_regular_images_on_both_sides_and_full_ci_force(tmp_path):
    images = path(
        [
            (-1.0, 0.0),
            (-0.72, 0.12),
            (-0.38, 0.0),
            (0.0, 0.0),
            (0.38, 0.0),
            (0.72, -0.21),
            (1.0, 0.0),
        ]
    )
    runner = job(tmp_path, images)
    _, forces = runner._evaluate_images(images)
    residual = runner._string_residuals(images, forces, ci_index=3)

    assert residual.regular_max > 0.8
    assert residual.regular_rms > 0.0
    assert residual.ci_max == pytest.approx(0.0, abs=1.0e-15)
    assert residual.ci_rms == pytest.approx(0.0, abs=1.0e-15)

    shifted = path(
        [
            (-1.0, 0.0),
            (-0.72, 0.0),
            (-0.38, 0.0),
            (0.12, 0.0),
            (0.38, 0.0),
            (0.72, 0.0),
            (1.0, 0.0),
        ]
    )
    _, shifted_forces = runner._evaluate_images(shifted)
    shifted_residual = runner._string_residuals(shifted, shifted_forces, ci_index=3)
    assert shifted_residual.ci_max > 0.0


def test_stage_snapshots_have_same_geometry_final_composed_ef_and_do_not_alias(tmp_path):
    images = path([(-1.0, 0.0), (-0.4, 0.12), (0.1, 0.07), (0.55, 0.03), (1.0, 0.0)])
    runner = job(tmp_path, images)
    ordinary = runner.relax_string(images)
    ordinary_positions = positions(ordinary.images).copy()
    ci_index = int(np.argmax(ordinary.energies_hartree[1:-1])) + 1
    climbing = runner.relax_cistring(ordinary.images, ci_index=ci_index)

    np.testing.assert_array_equal(positions(ordinary.images), ordinary_positions)
    for stage in (ordinary, climbing):
        fresh_e, fresh_f = runner._evaluate_images(list(stage.images))
        np.testing.assert_allclose(stage.energies_hartree, fresh_e, atol=1.0e-14, rtol=0.0)
        for observed, expected in zip(stage.forces_hartree_per_angstrom, fresh_f):
            np.testing.assert_allclose(observed, expected, atol=1.0e-14, rtol=0.0)
        assert all(image.info["source_id"] == "analytic-double-well" for image in stage.images)
        assert all(image.calc is images[0].calc for image in stage.images)
        assert all(image.get_masses()[0] == pytest.approx(1.25) for image in stage.images)
        assert all(image.f_max_th == pytest.approx(1.1e-7) for image in stage.images)
        assert all(image.f_rms_th == pytest.approx(2.2e-7) for image in stage.images)
        assert all(image.dp_max_th == pytest.approx(3.3e-7) for image in stage.images)
        assert all(image.dp_rms_th == pytest.approx(4.4e-7) for image in stage.images)


def test_actual_ordinary_and_ci_stages_converge_to_known_index_one_saddle(tmp_path):
    x = np.linspace(-1.0, 1.0, 8)
    y = 0.18 * np.cos(0.5 * np.pi * x) + 0.08 * x
    y[[0, -1]] = 0.0
    images = path(zip(x, y))
    runner = job(
        tmp_path,
        images,
        max_iter_relax=2500,
        max_iter_cistring=3000,
        lbfgs_curvature=50.0,
        string_f_max_th=1.0e-7,
        string_f_rms_th=1.0e-7,
        cistring_f_max_th=2.0e-7,
        cistring_f_rms_th=2.0e-7,
    )

    ordinary = runner.relax_string(images)
    ci_index = int(np.argmax(ordinary.energies_hartree[1:-1])) + 1
    climbing = runner.relax_cistring(ordinary.images, ci_index=ci_index)

    assert ordinary.status is RunStatus.CONVERGED
    assert climbing.status is RunStatus.CONVERGED
    np.testing.assert_array_equal(climbing.images[0].positions, images[0].positions)
    np.testing.assert_array_equal(climbing.images[-1].positions, images[-1].positions)
    np.testing.assert_allclose(
        climbing.images[ci_index].positions[0, :2],
        [0.0, 0.0],
        atol=5.0e-8,
        rtol=0.0,
    )
    assert np.linalg.norm(climbing.forces_hartree_per_angstrom[ci_index]) < 2.0e-7
    analytic_hessian = np.diag([-4.0, 4.0, 0.0])
    assert np.count_nonzero(np.linalg.eigvalsh(analytic_hessian) < -1.0e-12) == 1


def test_zero_budget_is_bounded_and_stringts_never_starts_prfo(tmp_path, monkeypatch):
    calls = []

    class ForbiddenPRFO:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(prfo_module, "PRFO", ForbiddenPRFO)
    images = path([(-1.0, 0.0), (-0.4, 0.1), (0.2, 0.05), (1.0, 0.0)])
    runner = job(
        tmp_path,
        images,
        refine="stringts",
        max_iter_relax=0,
        prfo_params={"max_iter": 9},
        project_rigid_modes=True,
    )
    stage = runner.relax_string(images)
    assert stage.status is RunStatus.BOUNDED_NONCONVERGED
    assert stage.budgets == {"maximum_iterations": 0, "used_iterations": 0}
    assert calls == []
    assert not list(tmp_path.glob("*_stringts_ts.xyz"))


def test_expected_invalid_numerical_state_refuses_but_programmer_error_propagates(tmp_path):
    invalid = [image(-1.0, 0.0, InvalidCalculator()), image(0.0, 0.0, InvalidCalculator()), image(1.0, 0.0, InvalidCalculator())]
    invalid_runner = job(tmp_path, invalid)
    refused = invalid_runner.relax_string(invalid)
    assert refused.status is RunStatus.REFUSED
    assert "finite" in refused.termination_reason.lower()

    broken = [image(-1.0, 0.0, BrokenCalculator()), image(0.0, 0.0, BrokenCalculator()), image(1.0, 0.0, BrokenCalculator())]
    broken_runner = job(tmp_path, broken)
    with pytest.raises(RuntimeError, match="programmer defect"):
        broken_runner.relax_string(broken)


def test_cistring_run_never_calls_prfo(tmp_path, monkeypatch):
    calls = []

    class ForbiddenPRFO:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(prfo_module, "PRFO", ForbiddenPRFO)
    monkeypatch.setattr(
        string_module,
        "kabsch_align",
        lambda _left, right: (right.copy(), 0.0, np.eye(3), np.zeros(3)),
    )
    images = path([(-1.0, 0.0), (1.0, 0.0)])
    runner = GSM(
        str(tmp_path / "string.out"),
        images[0],
        images[-1],
        paras={
            "string": {
                "n_images": 5,
                "max_iter_grow": 1,
                "merge_threshold": 10.0,
                "grow_relax_iters": 0,
                "refine": "cistring",
                "max_iter_relax": 0,
                "max_iter_cistring": 0,
                "string_f_max_th": 10.0,
                "string_f_rms_th": 10.0,
                "cistring_f_max_th": 10.0,
                "cistring_f_rms_th": 10.0,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_predict_and_relax_one",
        lambda last, direction, *_: (
            runner._copy_with_inherit(last),
            0.0,
            0.0,
        ),
    )

    result = runner.run()

    assert result["cistring_stage"].status is RunStatus.CONVERGED
    assert result["prfo_result"] is None
    assert calls == []


def test_stringts_caller_rejects_a_bounded_ci_stage_before_prfo(tmp_path, monkeypatch):
    calls = []

    class ForbiddenPRFO:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(prfo_module, "PRFO", ForbiddenPRFO)
    images = path([(-1.0, 0.0), (0.1, 0.0), (1.0, 0.0)])
    runner = job(tmp_path, images, refine="stringts")
    energies, forces = runner._evaluate_images(images)
    bounded = PathStageResult.capture(
        status=RunStatus.BOUNDED_NONCONVERGED,
        images=images,
        energies_hartree=energies,
        forces_hartree_per_angstrom=forces,
        budgets={"maximum_iterations": 0, "used_iterations": 0},
        termination_reason="cistring_iteration_limit",
        candidate=images[1],
    )

    with pytest.raises(ValueError, match="converged CI-String"):
        runner.restart_run(bounded, 1, str(tmp_path / "string"))
    assert calls == []


def test_unmerged_growth_limit_skips_ordinary_ci_and_prfo(tmp_path, monkeypatch):
    images = path([(-1.0, 0.0), (1.0, 0.0)])
    monkeypatch.setattr(
        string_module,
        "kabsch_align",
        lambda _left, right: (right.copy(), 0.0, np.eye(3), np.zeros(3)),
    )
    runner = GSM(
        str(tmp_path / "string.out"),
        images[0],
        images[-1],
        paras={
            "string": {
                "n_images": 5,
                "max_iter_grow": 1,
                "merge_threshold": 1.0e-12,
                "grow_relax_iters": 0,
                "refine": "stringts",
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "_predict_and_relax_one",
        lambda last, direction, *_: (
            runner._copy_with_inherit(last),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        runner,
        "relax_string",
        lambda *_: pytest.fail("unmerged growth must skip ordinary relaxation"),
    )
    monkeypatch.setattr(
        runner,
        "relax_cistring",
        lambda *_args, **_kwargs: pytest.fail("unmerged growth must skip CI-String"),
    )
    monkeypatch.setattr(
        runner,
        "restart_run",
        lambda *_args, **_kwargs: pytest.fail("unmerged growth must skip PRFO"),
    )

    result = runner.run()

    assert result["status"] == "bounded_nonconverged"
    assert result["termination_reason"] == "growth_limit"
    assert result["refinement_skipped_reason"] == "growth_not_merged"


def bounded_prfo_result(atoms: Atoms, output: str) -> PRFOResult:
    energy = float(atoms.get_potential_energy(force_consistent=True))
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    atoms.max_f = float(np.max(np.abs(forces)))
    atoms.rms_f = float(np.sqrt(np.mean(forces**2)))
    atoms.max_dp = 0.1
    atoms.rms_dp = 0.05
    return PRFOResult.capture(
        status=RunStatus.BOUNDED_NONCONVERGED,
        atoms=atoms,
        iterations=2,
        max_iterations=2,
        force_displacement_converged=False,
        termination_reason="maximum_iterations",
        energy_hartree=energy,
        forces_hartree_per_angstrom=forces,
        certification=None,
        output_paths={"candidate": output},
    )


def test_stringts_uses_typed_prfo_params_reports_actual_force_and_writes_no_false_ts(
    tmp_path, monkeypatch
):
    calls = []

    class RecorderPRFO:
        def __init__(self, output, atoms, paras):
            calls.append({"output": output, "atoms": atoms.copy(), "paras": paras})
            self.atoms = atoms
            self.output = output

        def run(self):
            self.atoms.set_positions([[0.2, 0.1, 0.0]])
            return bounded_prfo_result(self.atoms, self.output)

    monkeypatch.setattr(prfo_module, "PRFO", RecorderPRFO)
    images = path([(-1.0, 0.0), (-0.5, 0.0), (0.15, 0.0), (0.6, 0.0), (1.0, 0.0)])
    runner = job(
        tmp_path,
        images,
        refine="stringts",
        prfo_params={"max_iter": 2, "trust_radius": 0.12},
        project_rigid_modes=True,
    )
    energies, forces = runner._evaluate_images(images)
    stage = PathStageResult.capture(
        status=RunStatus.CONVERGED,
        images=images,
        energies_hartree=energies,
        forces_hartree_per_angstrom=forces,
        budgets={"maximum_iterations": 1, "used_iterations": 1},
        termination_reason="cistring_converged",
        candidate=images[2],
    )

    result = runner.restart_run(stage, 2, str(tmp_path / "string"))

    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert len(calls) == 1
    assert calls[0]["paras"]["prfo"]["max_iter"] == 2
    assert calls[0]["paras"]["prfo"]["trust_radius"] == pytest.approx(0.12)
    assert calls[0]["paras"]["prfo"]["project_rigid_modes"] is True
    assert calls[0]["paras"]["prfo"]["f_max_th"] == pytest.approx(1.1e-7)
    assert calls[0]["paras"]["prfo"]["f_rms_th"] == pytest.approx(2.2e-7)
    assert calls[0]["paras"]["prfo"]["dp_max_th"] == pytest.approx(3.3e-7)
    assert calls[0]["paras"]["prfo"]["dp_rms_th"] == pytest.approx(4.4e-7)
    ts_lines = [
        line
        for line in Path(runner.output).read_text().splitlines()
        if line.lstrip().startswith("TS candidate")
    ]
    assert len(ts_lines) == 1
    assert "0.00000     0.00000" not in ts_lines[0]
    assert (tmp_path / "string_stringts_candidate.xyz").exists()
    assert not (tmp_path / "string_stringts_ts.xyz").exists()


def test_parameter_metadata_states_active_inactive_and_cross_field_rules():
    assert set(STRING_PARAMETER_METADATA["field_defaults"]) == STRING_PARAMETER_METADATA["all_fields"]
    assert STRING_PARAMETER_METADATA["stage_fields"]["ordinary"] == {
        "max_iter_relax",
        "string_f_max_th",
        "string_f_rms_th",
        "lbfgs_memory",
        "lbfgs_curvature",
        "lbfgs_max_step",
        "reparam_every",
    }
    assert "max_iter_cistring" in STRING_PARAMETER_METADATA["stage_fields"]["ci"]
    assert "prfo_params" in STRING_PARAMETER_METADATA["stage_fields"]["ts"]
    categories = STRING_PARAMETER_METADATA["stage_fields"]
    assert categories["selector"] == {"refine"}
    owned = [name for names in categories.values() for name in names]
    assert len(owned) == len(set(owned))
    assert set(owned) == STRING_PARAMETER_METADATA["all_fields"]
    assert "refine=None" in STRING_PARAMETER_METADATA["inactive_rules"]
    assert any("reparam_every == 1" in rule for rule in STRING_PARAMETER_METADATA["cross_field_rules"])


def test_invalid_2007_and_nested_prfo_parameters_fail_at_string_boundary(tmp_path):
    images = path([(-1.0, 0.0), (0.0, 0.1), (1.0, 0.0)])
    with pytest.raises(ValueError, match="reparam_every must be 1"):
        job(tmp_path, images, reparam_every=3)
    with pytest.raises(ValueError, match="project_rigid_modes"):
        job(
            tmp_path,
            images,
            refine="stringts",
            prfo_params={"project_rigid_modes": True},
        )
    with pytest.raises(ValueError, match="unknown PRFO parameter"):
        job(tmp_path, images, refine="stringts", prfo_params={"not_a_parameter": 1})
    with pytest.raises(ValueError, match="project_rigid_modes.*stringts"):
        job(tmp_path, images, refine="cistring", project_rigid_modes=True)
    with pytest.raises(ValueError, match="project_rigid_modes.*stringts"):
        job(tmp_path, images, refine=None, project_rigid_modes=True)


def test_stringts_rejects_mismatched_endpoint_job_thresholds(tmp_path):
    images = path([(-1.0, 0.0), (0.0, 0.1), (1.0, 0.0)])
    images[-1].dp_rms_th = 9.9e-7
    with pytest.raises(ValueError, match="matching endpoint convergence thresholds"):
        job(tmp_path, images, refine="stringts")


def test_growth_only_accepts_default_inactive_ts_controls(tmp_path):
    images = path([(-1.0, 0.0), (0.0, 0.1), (1.0, 0.0)])
    runner = job(
        tmp_path,
        images,
        refine=None,
        max_iter_grow=0,
        prfo_params={},
        project_rigid_modes=False,
    )

    result = runner.run()

    assert result["status"] == "bounded_nonconverged"
    assert result["refinement_skipped_reason"] == "growth_not_merged"
    assert result["prfo_result"] is None


def test_refused_prfo_without_final_ef_is_preserved_without_reevaluation_or_artifact(
    tmp_path, monkeypatch
):
    class RefusingPRFO:
        def __init__(self, output, atoms, paras):
            self.output = output
            self.atoms = atoms

        def run(self):
            return PRFOResult.capture(
                status=RunStatus.REFUSED,
                atoms=self.atoms,
                iterations=0,
                max_iterations=4,
                force_displacement_converged=False,
                termination_reason="curvature_evidence_unavailable",
                energy_hartree=None,
                forces_hartree_per_angstrom=None,
                certification=None,
                output_paths={},
            )

    monkeypatch.setattr(prfo_module, "PRFO", RefusingPRFO)
    images = path([(-1.0, 0.0), (-0.5, 0.0), (0.15, 0.0), (0.6, 0.0), (1.0, 0.0)])
    runner = job(tmp_path, images, refine="stringts")
    energies, forces = runner._evaluate_images(images)
    stage = PathStageResult.capture(
        status=RunStatus.CONVERGED,
        images=images,
        energies_hartree=energies,
        forces_hartree_per_angstrom=forces,
        budgets={"maximum_iterations": 1, "used_iterations": 1},
        termination_reason="cistring_converged",
        candidate=images[2],
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_image",
        lambda *_args: pytest.fail("a refused PRFO candidate must not be re-evaluated"),
    )

    result = runner.restart_run(stage, 2, str(tmp_path / "refused"))

    assert result.status is RunStatus.REFUSED
    assert result.termination_reason == "curvature_evidence_unavailable"
    assert result.energy_hartree is None
    assert result.forces_hartree_per_angstrom is None
    assert not (tmp_path / "refused_stringts_candidate.xyz").exists()
    log = Path(runner.output).read_text()
    assert "final energy/forces unavailable" in log
    assert "0.00000     0.00000" not in log


def test_stringts_retires_old_certified_output_before_growth_limit(tmp_path):
    images = path([(-1.0, 0.0), (0.0, 0.1), (1.0, 0.0)])
    runner = job(tmp_path, images, refine="stringts", max_iter_grow=0)
    certified = tmp_path / "string_stringts_ts.xyz"
    certified.write_bytes(b"old-validated-ts\n")

    result = runner.run()

    assert result["status"] == "bounded_nonconverged"
    assert not certified.exists()
    archive = tmp_path / "string_stringts_ts.previous-1.xyz"
    assert archive.read_bytes() == b"old-validated-ts\n"
    assert str(archive) in Path(runner.output).read_text()
