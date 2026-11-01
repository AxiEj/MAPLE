from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.curvature import CurvatureValidationError
from maple.function.dispatcher.numerical_state import NumericalStateError
from maple.function.dispatcher.result_contracts import (
    PRFOResult,
    RunStatus,
    SaddleCertification,
    geometry_sha256,
)
from maple.function.dispatcher.ts.algorithm import neb as neb_module
from maple.function.dispatcher.ts.algorithm.neb import (
    NEB,
    NEBParams,
    improved_tangent,
    run_neb_stage,
)
from maple.function.utility import Molecules


class BondCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        delta = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(delta))
        unit = delta / distance
        derivative = 2.0 * (distance - 1.25)
        forces = np.zeros_like(atoms.positions)
        forces[0] = derivative * unit
        forces[1] = -derivative * unit
        energy = float((distance - 1.25) ** 2)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }


class PerpendicularCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = float(-np.sum(atoms.positions[:, 2]))
        forces = np.zeros_like(atoms.positions)
        forces[:, 2] = 1.0
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }


def images():
    calculator = BondCalculator()
    result = []
    for half_distance in (0.5, 0.65, 0.8):
        image = Atoms(
            "HH",
            positions=[[-half_distance, 0.0, 0.0], [half_distance, 0.0, 0.0]],
            masses=[1.1, 2.2],
        )
        image.calc = calculator
        image.info["source_identity"] = "analytic-bond-v1"
        image.new_array("path_atom_identity", np.array([17, 29], dtype=np.int64))
        for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
            setattr(image, name, 1.0e-3)
        result.append(image)
    return result


def nonconverged_images():
    result = images()
    calculator = PerpendicularCalculator()
    for image in result:
        image.calc = calculator
    return result


def test_stage_api_zero_and_one_step_budgets_are_truthful(tmp_path):
    zero_images = nonconverged_images()
    zero_endpoints = (
        zero_images[0].get_positions().copy(),
        zero_images[-1].get_positions().copy(),
    )
    zero = run_neb_stage(
        zero_images,
        params=NEBParams(max_iter=0, ifidpp=0),
        stage_kind="base",
        output=str(tmp_path / "zero.out"),
    )
    assert zero.status is RunStatus.BOUNDED_NONCONVERGED
    assert dict(zero.budgets) == {"maximum_iterations": 0, "used_iterations": 0}
    assert len(zero_images) == 3
    np.testing.assert_array_equal(zero_images[0].get_positions(), zero_endpoints[0])
    np.testing.assert_array_equal(zero_images[-1].get_positions(), zero_endpoints[1])

    one_images = nonconverged_images()
    before = one_images[1].get_positions().copy()
    one = run_neb_stage(
        one_images,
        params=NEBParams(max_iter=1, ifidpp=0),
        stage_kind="base",
        output=str(tmp_path / "one.out"),
    )
    assert dict(one.budgets) == {"maximum_iterations": 1, "used_iterations": 1}
    assert not np.array_equal(one_images[1].get_positions(), before)


def test_stage_result_energy_force_and_snapshot_are_aligned(tmp_path):
    mutable_images = images()
    stage = run_neb_stage(
        mutable_images,
        params=NEBParams(
            max_iter=1,
            ifidpp=0,
            neb_f_max_th=1.0e6,
            neb_f_rms_th=1.0e6,
        ),
        stage_kind="base",
        output=str(tmp_path / "aligned.out"),
    )
    for index, snapshot in enumerate(stage.images):
        assert snapshot.calc is mutable_images[0].calc
        assert snapshot.info["source_identity"] == "analytic-bond-v1"
        np.testing.assert_array_equal(snapshot.get_masses(), [1.1, 2.2])
        np.testing.assert_array_equal(snapshot.arrays["path_atom_identity"], [17, 29])
        assert stage.energies_hartree[index] == pytest.approx(
            snapshot.get_potential_energy(force_consistent=True)
        )
        np.testing.assert_allclose(
            stage.forces_hartree_per_angstrom[index], snapshot.get_forces()
        )
    mutable_images[1].positions[0, 0] -= 0.3
    assert not np.array_equal(stage.images[1].positions, mutable_images[1].positions)


def test_interpolated_image_preserves_calculator_domain_and_source_identity(tmp_path):
    endpoints = images()[::2]
    calculator = endpoints[0].calc
    result = NEB(
        str(tmp_path / "interpolated.out"),
        Molecules(endpoints),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "ifidpp": 0,
                "refine": None,
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
            }
        },
    ).run()
    assert len(result["images"]) == 3
    for image in result["images"]:
        assert image.calc is calculator
        assert image.info["source_identity"] == "analytic-bond-v1"
        np.testing.assert_array_equal(image.get_masses(), [1.1, 2.2])
        np.testing.assert_array_equal(image.arrays["path_atom_identity"], [17, 29])


def test_last_allowed_base_step_can_converge_and_start_ci(tmp_path, monkeypatch):
    calls = []

    class ConvergesAfterOneStep:
        def __init__(self, *args, **kwargs):
            self.stop_calls = 0

        def should_stop(self, gradient, fmax, frms):
            self.stop_calls += 1
            return self.stop_calls >= 2

        def two_loop(self, gradient):
            return np.zeros_like(gradient)

        def step_limit(self, step):
            return step

        def update(self, *args):
            return None

    monkeypatch.setattr(neb_module, "LBFGSDriver", ConvergesAfterOneStep)
    path = nonconverged_images()
    job = NEB(
        str(tmp_path / "last.out"),
        Molecules([path[0], path[-1]]),
        paras={"neb": {"n_images": 1, "max_iter": 1, "ifidpp": 0, "refine": "cineb"}},
    )
    monkeypatch.setattr(job, "restart_run", lambda *args, **kwargs: calls.append("ci"))
    result = job.run()
    assert result["base_stage"].status is RunStatus.CONVERGED
    assert calls == ["ci"]


def test_bounded_base_blocks_ci_and_prfo(tmp_path, monkeypatch):
    path = nonconverged_images()
    old_current = tmp_path / "bounded_nebts_ts.xyz"
    old_current.write_bytes(b"old certified output\n")
    job = NEB(
        str(tmp_path / "bounded.out"),
        Molecules([path[0], path[-1]]),
        paras={"neb": {"n_images": 1, "max_iter": 0, "ifidpp": 0, "refine": "nebts"}},
    )
    monkeypatch.setattr(
        job,
        "restart_run",
        lambda *args, **kwargs: pytest.fail("bounded base must block refinement"),
    )
    result = job.run()
    assert result["status"] == RunStatus.BOUNDED_NONCONVERGED.value
    assert result["ci_stage"] is None
    assert result["prfo_result"] is None
    assert not old_current.exists()
    assert (tmp_path / "bounded_nebts_ts.previous-1.xyz").read_bytes() == (
        b"old certified output\n"
    )


def test_ci_uses_its_memory_budget_and_returns_final_state(tmp_path, monkeypatch):
    observed_memories = []
    original = neb_module.LBFGSDriver

    class RecordingDriver(original):
        def __init__(self, *args, **kwargs):
            observed_memories.append(kwargs["m"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(neb_module, "LBFGSDriver", RecordingDriver)
    mutable_images = images()
    stage = run_neb_stage(
        mutable_images,
        params=NEBParams(
            max_iter=7,
            cineb_max_iter=0,
            cilbfgs_m=37,
            cineb_f_max_th=1.0e6,
            cineb_f_rms_th=1.0e6,
            neb_f_max_th=1.0e6,
            neb_f_rms_th=1.0e6,
        ),
        stage_kind="ci",
        output=str(tmp_path / "ci.out"),
    )
    assert observed_memories == [37]
    assert dict(stage.budgets) == {"maximum_iterations": 0, "used_iterations": 0}
    assert stage.status is RunStatus.CONVERGED
    for index, image in enumerate(stage.images):
        assert stage.energies_hartree[index] == pytest.approx(
            image.get_potential_energy(force_consistent=True)
        )
        np.testing.assert_allclose(stage.forces_hartree_per_angstrom[index], image.get_forces())


def test_ci_force_matches_henkelman_jonsson_formula(tmp_path):
    mutable_images = images()
    params = NEBParams(
        max_iter=0,
        cineb_max_iter=0,
        cineb_f_max_th=1.0e6,
        cineb_f_rms_th=1.0e6,
        neb_f_max_th=1.0e6,
        neb_f_rms_th=1.0e6,
    )
    stage = run_neb_stage(
        mutable_images,
        params=params,
        stage_kind="ci",
        output=str(tmp_path / "formula.out"),
    )
    climbing_index = max(
        range(1, len(stage.images) - 1),
        key=lambda index: stage.energies_hartree[index],
    )
    coordinates = [image.get_positions() for image in stage.images]
    tangent = improved_tangent(
        coordinates[climbing_index - 1],
        coordinates[climbing_index],
        coordinates[climbing_index + 1],
        stage.energies_hartree[climbing_index - 1],
        stage.energies_hartree[climbing_index],
        stage.energies_hartree[climbing_index + 1],
    )
    true_force = stage.forces_hartree_per_angstrom[climbing_index].reshape(-1)
    expected = true_force - 2.0 * np.dot(true_force, tangent) * tangent
    job = NEB(str(tmp_path / "formula-job.out"), Molecules(mutable_images), paras={"neb": {"n_images": 1}})
    projected, _, observed_index = job.cineb_forces(
        mutable_images,
        list(stage.energies_hartree),
        params.k_max,
        raw_forces=list(stage.forces_hartree_per_angstrom),
    )
    assert observed_index == climbing_index
    np.testing.assert_allclose(projected[climbing_index].reshape(-1), expected)


def test_public_ci_result_matches_its_final_geometry(tmp_path):
    path = images()
    job = NEB(
        str(tmp_path / "public-ci.out"),
        Molecules(path),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "cineb",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
            }
        },
    )
    result = job.run()
    assert result["ci_stage"].status is RunStatus.CONVERGED
    fresh_energies, raw_forces = job._evaluate_images(result["images"])
    np.testing.assert_allclose(result["energies"], fresh_energies)
    projected, _, _ = job.cineb_forces(
        result["images"], fresh_energies, job.params.k_max, raw_forces=raw_forces
    )
    for observed, expected in zip(result["projected_forces"], projected):
        np.testing.assert_allclose(observed, expected)


def test_bounded_prfo_candidate_stays_separate_and_writes_no_ts(tmp_path, monkeypatch):
    observed = {}

    class BoundedPRFO:
        def __init__(self, output, atoms, paras):
            observed["paras"] = paras
            self.atoms = atoms

        def run(self):
            candidate = self.atoms.copy()
            candidate.calc = self.atoms.calc
            candidate.positions[0, 1] += 0.4
            return PRFOResult.capture(
                status=RunStatus.BOUNDED_NONCONVERGED,
                atoms=candidate,
                iterations=1,
                max_iterations=3,
                force_displacement_converged=False,
                termination_reason="maximum_iterations",
                energy_hartree=candidate.get_potential_energy(force_consistent=True),
                forces_hartree_per_angstrom=candidate.get_forces(),
                certification=None,
                output_paths={},
            )

    prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")
    monkeypatch.setattr(prfo_module, "PRFO", BoundedPRFO)
    path = images()
    job = NEB(
        str(tmp_path / "neb.out"),
        Molecules(path),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "nebts",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
                "prfo_max_iter": 3,
                "prfo_project_rigid_modes": True,
            }
        },
    )
    result = job.run()
    assert result["status"] == RunStatus.BOUNDED_NONCONVERGED.value
    assert observed["paras"] == {
        "prfo": {"max_iter": 3, "project_rigid_modes": True}
    }
    assert result["candidate"] is not result["images"][job._ci_climbing_index]
    assert all(
        not np.array_equal(result["candidate"].positions, image.positions)
        for image in result["images"]
    )
    assert not (tmp_path / "neb_nebts_ts.xyz").exists()


def test_only_certified_prfo_writes_nebts_file(tmp_path, monkeypatch):
    class CertifiedPRFO:
        def __init__(self, output, atoms, paras):
            self.atoms = atoms

        def run(self):
            candidate = self.atoms.copy()
            candidate.calc = self.atoms.calc
            candidate.positions[0, 1] += 0.05
            certification = SaddleCertification(
                geometry_sha256=geometry_sha256(candidate),
                raw_hessian_sha256="a" * 64,
                precision_profile="strict-generic",
                material_negative_mode_count=1,
                force_displacement_converged=True,
            )
            return PRFOResult.capture(
                status=RunStatus.CONVERGED,
                atoms=candidate,
                iterations=1,
                max_iterations=3,
                force_displacement_converged=True,
                termination_reason="certified_index_one",
                energy_hartree=candidate.get_potential_energy(force_consistent=True),
                forces_hartree_per_angstrom=candidate.get_forces(),
                certification=certification,
                output_paths={},
            )

    prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")
    monkeypatch.setattr(prfo_module, "PRFO", CertifiedPRFO)
    old_current = tmp_path / "certified_nebts_ts.xyz"
    old_current.write_bytes(b"old certified bytes\n")
    path = images()
    job = NEB(
        str(tmp_path / "certified.out"),
        Molecules(path),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "nebts",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
                "prfo_max_iter": 3,
            }
        },
    )
    result = job.run()
    assert result["status"] == RunStatus.CONVERGED.value
    assert result["certification"].certified_ts
    assert old_current.is_file()
    assert old_current.read_bytes() != b"old certified bytes\n"
    assert (tmp_path / "certified_nebts_ts.previous-1.xyz").read_bytes() == (
        b"old certified bytes\n"
    )
    assert not list(tmp_path.glob("certified_nebts_ts.xyz.*.tmp"))


def test_provider_mutation_is_restored_and_refused(tmp_path):
    class MutatingCalculator(BondCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            atoms.info["provider_mutation"] = True
            super().calculate(atoms, properties, system_changes)

    mutable_images = images()
    calculator = MutatingCalculator()
    for image in mutable_images:
        image.calc = calculator
    with pytest.raises(CurvatureValidationError, match="mutated scientific Atoms state"):
        run_neb_stage(
            mutable_images,
            params=NEBParams(max_iter=0),
            stage_kind="base",
            output=str(tmp_path / "mutating.out"),
        )
    assert all("provider_mutation" not in image.info for image in mutable_images)


def test_complex_complete_potential_force_is_refused_before_float_cast(tmp_path):
    class ComplexForceCalculator(BondCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["forces"] = self.results["forces"].astype(np.complex128)
            self.results["forces"][0, 2] = 1.0j

    mutable_images = images()
    calculator = ComplexForceCalculator()
    for image in mutable_images:
        image.calc = calculator
    with pytest.raises(NumericalStateError, match="Complex"):
        run_neb_stage(
            mutable_images,
            params=NEBParams(max_iter=0),
            stage_kind="base",
            output=str(tmp_path / "complex.out"),
        )


def test_unknown_stage_refuses_before_evaluation(tmp_path):
    mutable_images = images()
    with pytest.raises(ValueError, match="stage_kind"):
        run_neb_stage(
            mutable_images,
            params=NEBParams(),
            stage_kind="prfo",
            output=str(tmp_path / "unknown.out"),
        )
    assert not Path(tmp_path / "unknown_image_traj.xyz").exists()


def test_nebts_uses_original_endpoint_thresholds_after_interpolation(
    tmp_path, monkeypatch
):
    observed = {}

    class BoundedPRFO:
        def __init__(self, output, atoms, paras):
            observed["thresholds"] = {
                name: getattr(atoms, name)
                for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
            }
            self.atoms = atoms

        def run(self):
            return PRFOResult.capture(
                status=RunStatus.BOUNDED_NONCONVERGED,
                atoms=self.atoms,
                iterations=0,
                max_iterations=0,
                force_displacement_converged=False,
                termination_reason="maximum_iterations",
                energy_hartree=self.atoms.get_potential_energy(force_consistent=True),
                forces_hartree_per_angstrom=self.atoms.get_forces(),
                certification=None,
                output_paths={},
            )

    prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")
    monkeypatch.setattr(prfo_module, "PRFO", BoundedPRFO)
    endpoints = images()[::2]
    expected = {
        "f_max_th": 1.1e-4,
        "f_rms_th": 2.2e-4,
        "dp_max_th": 3.3e-4,
        "dp_rms_th": 4.4e-4,
    }
    for endpoint in endpoints:
        for name, value in expected.items():
            setattr(endpoint, name, value)

    NEB(
        str(tmp_path / "thresholds.out"),
        Molecules(endpoints),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "nebts",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
                "prfo_max_iter": 0,
            }
        },
    ).run()
    assert observed["thresholds"] == expected


def test_neb_refuses_inconsistent_endpoint_thresholds(tmp_path):
    endpoints = images()[::2]
    endpoints[0].f_max_th = 1.0e-4
    endpoints[1].f_max_th = 2.0e-4
    with pytest.raises(ValueError, match="endpoint thresholds disagree"):
        NEB(
            str(tmp_path / "threshold-mismatch.out"),
            Molecules(endpoints),
            paras={"neb": {"n_images": 1}},
        )


def test_requested_nebts_rerun_archives_old_current_before_bounded_attempt(
    tmp_path, monkeypatch
):
    old_bytes = b"previous certified TS\n"
    current = tmp_path / "rerun_nebts_ts.xyz"
    current.write_bytes(old_bytes)

    class BoundedPRFO:
        def __init__(self, output, atoms, paras):
            self.atoms = atoms

        def run(self):
            assert not current.exists()
            return PRFOResult.capture(
                status=RunStatus.BOUNDED_NONCONVERGED,
                atoms=self.atoms,
                iterations=0,
                max_iterations=0,
                force_displacement_converged=False,
                termination_reason="maximum_iterations",
                energy_hartree=self.atoms.get_potential_energy(force_consistent=True),
                forces_hartree_per_angstrom=self.atoms.get_forces(),
                certification=None,
                output_paths={},
            )

    prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")
    monkeypatch.setattr(prfo_module, "PRFO", BoundedPRFO)
    path = images()
    job = NEB(
        str(tmp_path / "rerun.out"),
        Molecules(path),
        paras={
            "neb": {
                "n_images": 1,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "nebts",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
                "prfo_max_iter": 0,
            }
        },
    )
    result = job.run()
    assert result["status"] == RunStatus.BOUNDED_NONCONVERGED.value
    assert not current.exists()
    archive = Path(job.nebts_retired_output)
    assert archive.name == "rerun_nebts_ts.previous-1.xyz"
    assert archive.read_bytes() == old_bytes
    assert str(archive) in (tmp_path / "rerun.out").read_text()


def test_nebts_retirement_refuses_symlink_without_touching_target(tmp_path):
    unrelated = tmp_path / "unrelated.xyz"
    unrelated.write_bytes(b"unrelated valid bytes\n")
    current = tmp_path / "linked_nebts_ts.xyz"
    current.symlink_to(unrelated)
    with pytest.raises(ValueError, match="symlink or non-file"):
        neb_module._retire_owned_nebts_output(str(tmp_path / "linked.out"))
    assert current.is_symlink()
    assert unrelated.read_bytes() == b"unrelated valid bytes\n"


def test_ci_wrapper_keeps_exact_selected_internal_index_with_duplicate_pose(tmp_path):
    class ConstantCalculator(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, energy, force_x):
            super().__init__()
            self.energy = energy
            self.force_x = force_x

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            forces = np.zeros_like(atoms.positions)
            forces[:, 0] = self.force_x
            self.results = {
                "energy": self.energy,
                "free_energy": self.energy,
                "forces": forces,
            }

    path = []
    for position, energy, force_x in (
        (0.0, 0.0, 10.0),
        (0.0, 5.0, 20.0),
        (1.0, 1.0, 30.0),
        (2.0, 0.0, 40.0),
    ):
        atom = Atoms("H", positions=[[position, 0.0, 0.0]])
        atom.calc = ConstantCalculator(energy, force_x)
        path.append(atom)
    job = NEB(
        str(tmp_path / "duplicate.out"),
        Molecules(path),
        paras={
            "neb": {
                "n_images": 2,
                "max_iter": 0,
                "cineb_max_iter": 0,
                "ifidpp": 0,
                "refine": "cineb",
                "neb_f_max_th": 1.0e6,
                "neb_f_rms_th": 1.0e6,
                "cineb_f_max_th": 1.0e6,
                "cineb_f_rms_th": 1.0e6,
            }
        },
    )
    stage = job._run_cineb_stage(path)
    assert job._ci_climbing_index == 1
    assert job._ci_candidate is path[1]
    assert job._ci_candidate.calc is path[1].calc
    np.testing.assert_array_equal(job._ci_projected_forces[0], np.zeros((1, 3)))
    assert getattr(stage, "_stage_diagnostics")["climbing_image_index"] == 1


def test_explicit_ci_displacement_tolerance_blocks_force_only_convergence(
    tmp_path, monkeypatch
):
    class ForcePassesAfterLargeMove:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def ci_should_stop(self, regular, climbing):
            self.calls += 1
            return self.calls >= 2

        def two_loop(self, gradient):
            step = np.zeros_like(gradient)
            step[0] = 0.2
            return step

        def step_limit(self, step):
            return step

        def update(self, *args):
            return None

    monkeypatch.setattr(neb_module, "LBFGSDriver", ForcePassesAfterLargeMove)
    stage = run_neb_stage(
        images(),
        params=NEBParams(
            cineb_max_iter=1,
            ci_displacement_tol=0.05,
            neb_f_max_th=1.0e6,
            neb_f_rms_th=1.0e6,
            cineb_f_max_th=1.0e6,
            cineb_f_rms_th=1.0e6,
        ),
        stage_kind="ci",
        output=str(tmp_path / "ci-displacement.out"),
    )
    assert stage.status is RunStatus.BOUNDED_NONCONVERGED
    assert stage.termination_reason == "maximum_iterations"
    assert getattr(stage, "_stage_diagnostics")[
        "ci_max_displacement_angstrom"
    ] == pytest.approx(0.2)


def test_explicit_ci_displacement_tolerance_accepts_initial_converged_point(
    tmp_path, monkeypatch
):
    class InitiallyConverged:
        def __init__(self, *args, **kwargs):
            pass

        def ci_should_stop(self, regular, climbing):
            return True

    monkeypatch.setattr(neb_module, "LBFGSDriver", InitiallyConverged)
    stage = run_neb_stage(
        images(),
        params=NEBParams(
            cineb_max_iter=0,
            ci_displacement_tol=0.0,
            neb_f_max_th=1.0e6,
            neb_f_rms_th=1.0e6,
            cineb_f_max_th=1.0e6,
            cineb_f_rms_th=1.0e6,
        ),
        stage_kind="ci",
        output=str(tmp_path / "initial-ci-displacement.out"),
    )
    assert stage.status is RunStatus.CONVERGED
    assert getattr(stage, "_stage_diagnostics")[
        "ci_max_displacement_angstrom"
    ] == 0.0


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, 1.0j, True, "0.1"])
def test_ci_displacement_tolerance_must_be_finite_and_nonnegative(tmp_path, value):
    with pytest.raises(ValueError, match="ci_displacement_tol"):
        run_neb_stage(
            images(),
            params=NEBParams(ci_displacement_tol=value),
            stage_kind="ci",
            output=str(tmp_path / "invalid-ci-displacement.out"),
        )
