from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import (
    PRFOResult,
    PathStageResult,
    RunStatus,
    SaddleCertification,
)
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.ts import TransitionState


class Quadratic(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, reference, hessian):
        super().__init__()
        self.reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        self._hessian = np.asarray(hessian, dtype=np.float64)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions.reshape(-1) - self.reference
        gradient = self._hessian @ displacement
        energy = float(0.5 * displacement @ gradient)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -gradient.reshape(-1, 3),
        }

    def get_hessian(self, atoms):
        return self._hessian.copy()


class MutatingFinalHessian(Quadratic):
    def __init__(self, reference, hessian):
        super().__init__(reference, hessian)
        self.hessian_calls = 0

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        if self.hessian_calls == 2:
            atoms.positions[0, 0] = 1.0
        return self._hessian.copy()


class ComplexHessian(Quadratic):
    def get_hessian(self, atoms):
        hessian = self._hessian.astype(np.complex128)
        hessian[0, 0] += 1000.0j
        return hessian


class NonfiniteEnergy(Quadratic):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = np.nan
        self.results["free_energy"] = np.nan


class NonfiniteFinalHessian(Quadratic):
    def __init__(self, reference, hessian):
        super().__init__(reference, hessian)
        self.hessian_calls = 0

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        if self.hessian_calls == 2:
            return np.full_like(self._hessian, np.nan)
        return self._hessian.copy()


class ProgrammerErrorHessian(Quadratic):
    def get_hessian(self, atoms):
        raise TypeError("programmer error sentinel")


def _stationary_hydrogen(*, negative_modes: int) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    diagonal = np.ones(3)
    diagonal[:negative_modes] = -1.0
    atoms.calc = Quadratic(atoms.positions, np.diag(diagonal))
    return atoms


def test_certified_success(tmp_path):
    source = _stationary_hydrogen(negative_modes=1)
    result = PRFO(
        str(tmp_path / "certified.out"),
        source,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert isinstance(result, PRFOResult)
    assert result.status is RunStatus.CONVERGED
    assert result.certified_ts is True
    assert result.force_displacement_converged is True
    assert result.geometry_sha256
    assert result.forces_hartree_per_angstrom.flags.writeable is False
    assert result.curvature["negative_mode_count"] == 1
    assert (tmp_path / "certified_prfo_ts.xyz").is_file()
    frozen_positions = result.atoms.get_positions().copy()
    source.positions[0, 0] = 3.0
    np.testing.assert_array_equal(result.atoms.get_positions(), frozen_positions)
    np.testing.assert_array_equal(result.forces_hartree_per_angstrom, 0.0)


def test_budget_returns_typed_result(tmp_path):
    result = PRFO(
        str(tmp_path / "budget.out"),
        _stationary_hydrogen(negative_modes=1),
        paras={"max_iter": 0},
    ).run()

    assert isinstance(result, PRFOResult)
    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.certified_ts is False
    assert result.iterations == 0


@pytest.mark.parametrize("negative_modes", [0, 2])
def test_failure_never_writes_ts_file(tmp_path, negative_modes):
    result = PRFO(
        str(tmp_path / "minimum.out"),
        _stationary_hydrogen(negative_modes=negative_modes),
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is RunStatus.REFUSED
    assert result.certified_ts is False
    assert not (tmp_path / "minimum_prfo_ts.xyz").exists()
    assert (tmp_path / "minimum_prfo_candidate.xyz").is_file()


def test_transition_state_union_contract(tmp_path):
    certified = TransitionState(
        str(tmp_path / "public-certified.out"),
        _stationary_hydrogen(negative_modes=1),
        {"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
        method="prfo",
    )
    geometry = certified.run()
    assert isinstance(geometry, Atoms)
    assert certified.last_result.status is RunStatus.CONVERGED

    bounded = TransitionState(
        str(tmp_path / "public-bounded.out"),
        _stationary_hydrogen(negative_modes=1),
        {"max_iter": 0},
        method="prfo",
    )
    result = bounded.run()
    assert isinstance(result, PRFOResult)
    assert bounded.last_result is result


def test_complex_hessian_never_certifies_or_writes_ts(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = ComplexHessian(atoms.positions, np.diag([-1.0, 1.0, 1.0]))

    result = PRFO(
        str(tmp_path / "complex.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.certified_ts is False
    assert "complex" in result.termination_reason.lower()
    assert not (tmp_path / "complex_prfo_ts.xyz").exists()


def test_final_hessian_mutation_refuses_and_restores_geometry(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    initial = atoms.positions.copy()
    atoms.calc = MutatingFinalHessian(initial, np.diag([-1.0, 1.0, 1.0]))

    result = PRFO(
        str(tmp_path / "mutating.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.certified_ts is False
    assert "mutated" in result.termination_reason
    np.testing.assert_array_equal(atoms.positions, initial)
    np.testing.assert_array_equal(result.atoms.positions, initial)
    assert not (tmp_path / "mutating_prfo_ts.xyz").exists()


def test_nonfinite_final_state_never_publishes_ts(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = NonfiniteEnergy(atoms.positions, np.diag([-1.0, 1.0, 1.0]))

    result = PRFO(
        str(tmp_path / "nan.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.certified_ts is False
    assert "finite" in result.termination_reason
    assert not (tmp_path / "nan_prfo_ts.xyz").exists()


def test_expected_invalid_final_hessian_sets_last_result(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = NonfiniteFinalHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )
    transition = TransitionState(
        str(tmp_path / "invalid-final.out"),
        atoms,
        {"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
        method="prfo",
    )

    result = transition.run()

    assert isinstance(result, PRFOResult)
    assert result.status is RunStatus.REFUSED
    assert transition.last_result is result
    assert "finite" in result.termination_reason
    assert not (tmp_path / "invalid-final_prfo_ts.xyz").exists()


def test_new_attempt_replaces_prior_certification_and_removes_stale_ts(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    transition = TransitionState(
        str(tmp_path / "repeat.out"),
        atoms,
        {"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
        method="prfo",
    )
    assert isinstance(transition.run(), Atoms)
    first_result = transition.last_result
    assert (tmp_path / "repeat_prfo_ts.xyz").is_file()
    atoms.calc = NonfiniteEnergy(atoms.positions, np.diag([-1.0, 1.0, 1.0]))

    second_result = transition.run()

    assert second_result.status is RunStatus.REFUSED
    assert transition.last_result is second_result
    assert transition.last_result is not first_result
    assert not (tmp_path / "repeat_prfo_ts.xyz").exists()


def test_programmer_error_propagates_without_stale_result(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = ProgrammerErrorHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )
    transition = TransitionState(
        str(tmp_path / "programmer-error.out"),
        atoms,
        {"max_iter": 1},
        method="prfo",
    )

    with pytest.raises(TypeError, match="programmer error sentinel"):
        transition.run()

    assert transition.last_result is None


def test_shared_result_interfaces_freeze_snapshots():
    atoms = _stationary_hydrogen(negative_modes=1)
    stage = PathStageResult.capture(
        status=RunStatus.BOUNDED_NONCONVERGED,
        images=[atoms],
        energies_hartree=[0.0],
        forces_hartree_per_angstrom=[np.zeros((1, 3))],
        budgets={"maximum_iterations": 0, "used_iterations": 0},
        termination_reason="maximum_iterations",
    )
    assert stage.energies_hartree.flags.writeable is False
    assert stage.forces_hartree_per_angstrom[0].flags.writeable is False
    assert stage.geometry_sha256[0]
    assert set(RunStatus) == {
        RunStatus.CONVERGED,
        RunStatus.BOUNDED_NONCONVERGED,
        RunStatus.FAILED,
        RunStatus.REFUSED,
    }
    assert SaddleCertification(
        geometry_sha256=stage.geometry_sha256[0],
        raw_hessian_sha256=None,
        precision_profile=None,
        material_negative_mode_count=None,
        force_displacement_converged=False,
        refusal_reason="not_evaluated",
    ).certified_ts is False


def test_all_repo_callers_migrated():
    root = Path(__file__).resolve().parents[1]
    owned_callers = [
        root / "maple/function/dispatcher/ts/ts.py",
        root / "maple/function/dispatcher/ts/algorithm/neb.py",
        root / "maple/function/dispatcher/ts/algorithm/string.py",
    ]
    for path in owned_callers:
        text = path.read_text()
        assert "prfo.run()" in text
        assert "certified_ts" in text or "last_result" in text
    assert "images.insert(hei + 1, ts_opt)" not in owned_callers[1].read_text()
    assert "images_ts.insert" not in owned_callers[2].read_text()
    benchmark_callers = [
        root / "docs/implicit-solvation/benchmarks/run_mlip_device_canary.py",
        root
        / "docs/implicit-solvation/benchmarks/run_route1_stationary_validation.py",
    ]
    for path in benchmark_callers:
        text = path.read_text()
        assert "prfo_result" in text
        assert "certified_ts" in text
