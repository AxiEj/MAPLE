from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import (
    PRFOResult,
    PathStageResult,
    RunStatus,
    SaddleCertification,
    geometry_sha256,
)
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.ts import TransitionState
from maple.function.calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticIdentityError,
)

prfo_module = importlib.import_module("maple.function.dispatcher.ts.algorithm.PRFO")


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


class ComplexScalarEnergy(Quadratic):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = np.complex128(1000.0j)
        self.results["free_energy"] = np.complex128(1000.0j)


class MutatingThenFailingFinalHessian(Quadratic):
    def __init__(self, reference, hessian):
        super().__init__(reference, hessian)
        self.hessian_calls = 0

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        if self.hessian_calls == 2:
            atoms.positions[0, 0] = 1.0
            raise ValueError("Cartesian Hessian must contain only finite values.")
        return self._hessian.copy()


class MutatingZeroBudgetEnergy(Quadratic):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        atoms.positions[0, 0] = 2.0
        super().calculate(atoms, properties, system_changes)


class MutatingFinalChargeHessian(Quadratic):
    def __init__(self, reference, hessian):
        super().__init__(reference, hessian)
        self.hessian_calls = 0

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        if self.hessian_calls == 2:
            atoms.info["charge"] = 1
        return self._hessian.copy()


class MutatingScientificStateHessian(Quadratic):
    def __init__(self, reference, hessian):
        super().__init__(reference, hessian)
        self.hessian_calls = 0

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        if self.hessian_calls == 2:
            atoms.set_masses(np.full(len(atoms), 9.0))
            atoms.set_initial_charges(np.full(len(atoms), 0.5))
            atoms.set_cell(np.eye(3) * 4.0)
            atoms.set_pbc(True)
            atoms.set_constraint([])
            atoms.info["mult"] = 3
            atoms.calc = object()
        return self._hessian.copy()


class AnalyticIdentityFailureHessian(Quadratic):
    def get_hessian(self, atoms):
        raise AnalyticIdentityError("AIM source identity drift")


class AnalyticFailureEF(Quadratic):
    def __init__(self, reference, hessian, *, failure, fail_property):
        super().__init__(reference, hessian)
        self.failure = failure
        self.fail_property = fail_property

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if self.fail_property == "energy":
            atoms.positions[0, 0] = 3.0
            raise self.failure("dedicated analytic E/F refusal")
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        if self.fail_property == "forces":
            atoms.positions[0, 0] = 3.0
            raise self.failure("dedicated analytic E/F refusal")
        return super().get_forces(atoms)


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


def test_complex_numpy_scalar_energy_never_certifies(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = ComplexScalarEnergy(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )

    result = PRFO(
        str(tmp_path / "complex-scalar.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is RunStatus.REFUSED
    assert "complex" in result.termination_reason.lower()
    assert not (tmp_path / "complex-scalar_prfo_ts.xyz").exists()


def test_result_capture_rejects_complex_scalar_before_float_cast():
    atoms = _stationary_hydrogen(negative_modes=1)
    with pytest.raises(ValueError, match="real finite scalar"):
        PRFOResult.capture(
            status=RunStatus.BOUNDED_NONCONVERGED,
            atoms=atoms,
            iterations=0,
            max_iterations=0,
            force_displacement_converged=False,
            termination_reason="complex",
            energy_hartree=np.complex128(1000.0j),
            forces_hartree_per_angstrom=np.zeros((1, 3)),
            certification=None,
            output_paths={},
        )


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


def test_final_hessian_mutation_then_error_restores_before_propagation(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    initial = atoms.positions.copy()
    atoms.calc = MutatingThenFailingFinalHessian(
        initial, np.diag([-1.0, 1.0, 1.0])
    )

    with pytest.raises(ValueError, match="finite"):
        PRFO(
            str(tmp_path / "mutating-error.out"), atoms,
            paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
        ).run()

    np.testing.assert_array_equal(atoms.positions, initial)


def test_final_hessian_restores_complete_scientific_atoms_state(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.info.update({"charge": 0, "mult": 1})
    initial_masses = atoms.get_masses().copy()
    initial_charges = atoms.get_initial_charges().copy()
    initial_cell = np.asarray(atoms.cell).copy()
    initial_pbc = atoms.get_pbc().copy()
    calculator = MutatingScientificStateHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )
    atoms.calc = calculator

    result = PRFO(
        str(tmp_path / "scientific-state.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is RunStatus.REFUSED
    np.testing.assert_array_equal(atoms.get_masses(), initial_masses)
    np.testing.assert_array_equal(atoms.get_initial_charges(), initial_charges)
    np.testing.assert_array_equal(np.asarray(atoms.cell), initial_cell)
    np.testing.assert_array_equal(atoms.get_pbc(), initial_pbc)
    assert atoms.info == {"charge": 0, "mult": 1}
    assert atoms.calc is calculator


def test_zero_budget_energy_mutation_restores_before_refusal(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    initial = atoms.positions.copy()
    atoms.calc = MutatingZeroBudgetEnergy(
        initial, np.diag([-1.0, 1.0, 1.0])
    )

    result = PRFO(
        str(tmp_path / "zero-mutating.out"), atoms,
        paras={"max_iter": 0},
    ).run()

    assert result.status is RunStatus.REFUSED
    np.testing.assert_array_equal(atoms.positions, initial)
    np.testing.assert_array_equal(result.atoms.positions, initial)


def test_final_hessian_charge_mutation_refuses_and_restores(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.info["charge"] = 0
    atoms.calc = MutatingFinalChargeHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )

    result = PRFO(
        str(tmp_path / "charge-mutating.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is RunStatus.REFUSED
    assert atoms.info["charge"] == 0
    assert result.atoms.info["charge"] == 0


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


def test_dedicated_analytic_identity_failure_returns_refusal(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = AnalyticIdentityFailureHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )
    transition = TransitionState(
        str(tmp_path / "identity-failure.out"), atoms,
        {"max_iter": 1}, method="prfo",
    )

    result = transition.run()

    assert result.status is RunStatus.REFUSED
    assert transition.last_result is result
    assert "identity drift" in result.termination_reason


@pytest.mark.parametrize(
    ("failure", "fail_property"),
    [
        (AnalyticIdentityError, "energy"),
        (AnalyticCapabilityError, "forces"),
    ],
)
def test_dedicated_analytic_ef_failure_returns_refusal_and_restores(
    tmp_path, failure, fail_property
):
    atoms = _stationary_hydrogen(negative_modes=1)
    initial = atoms.positions.copy()
    atoms.calc = AnalyticFailureEF(
        initial,
        np.diag([-1.0, 1.0, 1.0]),
        failure=failure,
        fail_property=fail_property,
    )
    transition = TransitionState(
        str(tmp_path / f"{fail_property}-failure.out"),
        atoms,
        {"max_iter": 1},
        method="prfo",
    )

    result = transition.run()

    assert result.status is RunStatus.REFUSED
    assert transition.last_result is result
    assert "dedicated analytic E/F refusal" in result.termination_reason
    np.testing.assert_array_equal(atoms.positions, initial)


@pytest.mark.parametrize("failure", [TypeError, ValueError, RuntimeError])
@pytest.mark.parametrize("fail_property", ["energy", "forces"])
def test_programmer_ef_failure_propagates_after_restore_and_clears_result(
    tmp_path, failure, fail_property
):
    atoms = _stationary_hydrogen(negative_modes=1)
    initial = atoms.positions.copy()
    atoms.calc = AnalyticFailureEF(
        initial,
        np.diag([-1.0, 1.0, 1.0]),
        failure=failure,
        fail_property=fail_property,
    )
    transition = TransitionState(
        str(tmp_path / f"programmer-{fail_property}.out"),
        atoms,
        {"max_iter": 1},
        method="prfo",
    )

    with pytest.raises(failure, match="dedicated analytic E/F refusal"):
        transition.run()

    assert transition.last_result is None
    np.testing.assert_array_equal(atoms.positions, initial)


@pytest.mark.parametrize("bad_mass", [0.0, -1.0, np.nan])
def test_invalid_mass_returns_typed_refusal(tmp_path, bad_mass):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.set_masses([bad_mass])
    transition = TransitionState(
        str(tmp_path / "invalid-mass.out"), atoms, {"max_iter": 1}, method="prfo"
    )

    result = transition.run()

    assert result.status is RunStatus.REFUSED
    assert transition.last_result is result
    assert "positive atomic masses" in result.termination_reason


def test_invalid_mass_in_updated_curvature_returns_typed_refusal(tmp_path, monkeypatch):
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    atoms.calc = Quadratic(
        np.zeros(3),
        np.diag([-1.0, 1.0, 1.0]),
    )
    original = prfo_module.build_mass_weighted_mode_basis
    calls = 0

    def invalidate_on_updated_curvature(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            args[0].set_masses([0.0])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        prfo_module,
        "build_mass_weighted_mode_basis",
        invalidate_on_updated_curvature,
    )
    transition = TransitionState(
        str(tmp_path / "invalid-updated-mass.out"),
        atoms,
        {
            "max_iter": 2,
            "recalc": 2,
            "f_max_th": 0.0,
            "f_rms_th": 0.0,
            "dp_max_th": 0.0,
            "dp_rms_th": 0.0,
        },
        method="prfo",
    )

    result = transition.run()

    assert calls == 2
    assert result.status is RunStatus.REFUSED
    assert transition.last_result is result
    assert "positive atomic masses" in result.termination_reason


def test_late_hessian_refusal_preserves_valid_ef_and_convergence(tmp_path):
    atoms = _stationary_hydrogen(negative_modes=1)
    atoms.calc = NonfiniteFinalHessian(
        atoms.positions, np.diag([-1.0, 1.0, 1.0])
    )

    result = PRFO(
        str(tmp_path / "late-h.out"), atoms,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is RunStatus.REFUSED
    assert result.force_displacement_converged is True
    assert result.energy_hartree == pytest.approx(0.0)
    np.testing.assert_array_equal(result.forces_hartree_per_angstrom, 0.0)
    assert result.convergence_metrics == {
        "max_f": 0.0, "rms_f": 0.0, "max_dp": 0.0, "rms_dp": 0.0,
    }


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"geometry_sha256": "x"},
        {"raw_hessian_sha256": "x"},
        {"precision_profile": "forged-profile"},
        {"material_negative_mode_count": True},
        {"material_negative_mode_count": 0},
        {"force_displacement_converged": 1},
        {"force_displacement_converged": False},
    ],
)
def test_malformed_certification_cannot_be_constructed(overrides):
    values = {
        "geometry_sha256": "a" * 64,
        "raw_hessian_sha256": "b" * 64,
        "precision_profile": "strict-generic",
        "material_negative_mode_count": 1,
        "force_displacement_converged": True,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        SaddleCertification(**values)


def test_nested_certification_diagnostics_are_deeply_immutable():
    source = {"nested": {"values": [1, 2]}}
    certification = SaddleCertification(
        geometry_sha256="a" * 64,
        raw_hessian_sha256="b" * 64,
        precision_profile="strict-generic",
        material_negative_mode_count=1,
        force_displacement_converged=True,
        diagnostics=source,
    )
    source["nested"]["values"].append(3)

    assert certification.diagnostics["nested"]["values"] == (1, 2)
    assert isinstance(certification.diagnostics["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        certification.diagnostics["nested"]["new"] = 1


def test_path_stage_rejects_foreign_candidate_certification():
    image = _stationary_hydrogen(negative_modes=1)
    candidate = image.copy()
    candidate.positions[0, 0] = 1.0
    certification = SaddleCertification(
        geometry_sha256="a" * 64,
        raw_hessian_sha256="b" * 64,
        precision_profile="strict-generic",
        material_negative_mode_count=1,
        force_displacement_converged=True,
    )

    with pytest.raises(ValueError, match="certification|geometry"):
        PathStageResult.capture(
            status=RunStatus.BOUNDED_NONCONVERGED,
            images=[image],
            budgets={"maximum_iterations": 1, "used_iterations": 1},
            termination_reason="candidate",
            candidate=candidate,
            certification=certification,
        )


def test_path_stage_accepts_certification_for_its_candidate_geometry():
    image = _stationary_hydrogen(negative_modes=1)
    candidate = image.copy()
    candidate.positions[0, 0] = 1.0
    certification = SaddleCertification(
        geometry_sha256=geometry_sha256(candidate),
        raw_hessian_sha256="b" * 64,
        precision_profile="strict-generic",
        material_negative_mode_count=1,
        force_displacement_converged=True,
    )

    stage = PathStageResult.capture(
        status=RunStatus.CONVERGED,
        images=[image],
        budgets={"maximum_iterations": 1, "used_iterations": 1},
        termination_reason="certified_candidate",
        candidate=candidate,
        certification=certification,
    )

    assert stage.certification is certification


@pytest.mark.parametrize(
    "budgets",
    [
        {"maximum_iterations": True, "used_iterations": 0},
        {"maximum_iterations": 1, "used_iterations": np.int64(0)},
        {"maximum_iterations": 0, "used_iterations": 1},
    ],
)
def test_result_budgets_require_exact_coherent_counts(budgets):
    image = _stationary_hydrogen(negative_modes=1)
    with pytest.raises((TypeError, ValueError)):
        PathStageResult.capture(
            status=RunStatus.BOUNDED_NONCONVERGED,
            images=[image],
            budgets=budgets,
            termination_reason="budget",
        )


@pytest.mark.parametrize(
    "override",
    [
        {"iterations": True},
        {"max_iterations": np.int64(0)},
        {"force_displacement_converged": 1},
    ],
)
def test_prfo_capture_requires_exact_count_and_boolean_types(override):
    values = {
        "status": RunStatus.BOUNDED_NONCONVERGED,
        "atoms": _stationary_hydrogen(negative_modes=1),
        "iterations": 0,
        "max_iterations": 0,
        "force_displacement_converged": False,
        "termination_reason": "budget",
        "energy_hartree": 0.0,
        "forces_hartree_per_angstrom": np.zeros((1, 3)),
        "certification": None,
        "output_paths": {},
    }
    values.update(override)
    with pytest.raises(TypeError):
        PRFOResult.capture(**values)


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
