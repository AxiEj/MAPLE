from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

from maple.function.calculator.extra_correction.implicit.numerical_force import (
    EvaluationAuditContext,
    NumericalForceBudget,
    NumericalForceProvider,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult


class _SentinelCalculator(Calculator):
    implemented_properties = []


class _AnalyticEnergyProvider:
    """Deterministic energy-only endpoint used to specify the wrapper contract."""

    supported_properties = frozenset({"energy"})

    def __init__(self, *, fail_on_call: int | None = None, nonfinite=False):
        self.fail_on_call = fail_on_call
        self.nonfinite = nonfinite
        self.calls: list[tuple[np.ndarray, EvaluationAuditContext]] = []
        self._lock = Lock()

    def evaluate(self, atoms, *, need_forces=False, audit_context=None):
        assert need_forces is False
        assert isinstance(audit_context, EvaluationAuditContext)
        assert audit_context.path.is_dir()
        assert not atoms.constraints
        xyz = np.asarray(atoms.positions, dtype=np.float64)
        with self._lock:
            self.calls.append((xyz.copy(), audit_context))
            call_number = len(self.calls)
        if self.fail_on_call == call_number:
            raise RuntimeError("selected provider failed")
        polar = 1.25 + float(np.sum(0.5 * xyz**2))
        nonpolar = -0.4 + float(np.sum(2.0 * xyz))
        energy = polar + nonpolar
        if self.nonfinite:
            energy = np.nan
        return SolvationResult(
            energy_hartree=energy,
            components_hartree={"polar": polar, "nonpolar": nonpolar},
            provenance={"provider": "analytic-energy-only"},
        )


def _atoms() -> Atoms:
    return Atoms("H", positions=[[0.25, -0.5, 0.75]])


def _wrapper(provider, tmp_path: Path, **overrides) -> NumericalForceProvider:
    options = {
        "force_step_angstrom": 1.0e-3,
        "force_check_step_angstrom": 5.0e-4,
        "max_scalar_evaluations": 100,
        "max_raw_records": 110,
        "max_audit_bytes": 1_000_000,
        "audit_dir": tmp_path / "numerical-force-audit",
    }
    options.update(overrides)
    return NumericalForceProvider(provider, **options)


def test_returns_center_scalar_and_negative_energy_gradient_in_hartree_per_angstrom(
    tmp_path,
):
    atoms = _atoms()
    result = _wrapper(_AnalyticEnergyProvider(), tmp_path).evaluate(
        atoms, need_forces=True
    )

    expected_energy = 1.25 + np.sum(0.5 * atoms.positions**2) - 0.4 + np.sum(
        2.0 * atoms.positions
    )
    assert result.energy_hartree == pytest.approx(expected_energy)
    assert result.forces_hartree_per_angstrom == pytest.approx(
        -(atoms.positions + 2.0)
    )


def test_returns_primary_and_independent_check_vectors_for_all_cartesian_coordinates(
    tmp_path,
):
    result = _wrapper(_AnalyticEnergyProvider(), tmp_path).evaluate(
        _atoms(), need_forces=True
    )

    diagnostic = result.provenance["numerical_force"]
    assert np.asarray(diagnostic["primary_forces_hartree_per_angstrom"]).shape == (
        1,
        3,
    )
    assert np.asarray(diagnostic["check_forces_hartree_per_angstrom"]).shape == (
        1,
        3,
    )
    assert diagnostic["max_abs_difference_hartree_per_angstrom"] == pytest.approx(
        0.0, abs=1.0e-10
    )
    assert diagnostic["rms_difference_hartree_per_angstrom"] == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_preserves_exact_center_components_while_differentiating_their_total(tmp_path):
    atoms = _atoms()
    result = _wrapper(_AnalyticEnergyProvider(), tmp_path).evaluate(
        atoms, need_forces=True
    )

    assert result.components_hartree == pytest.approx(
        {
            "polar": 1.25 + float(np.sum(0.5 * atoms.positions**2)),
            "nonpolar": -0.4 + float(np.sum(2.0 * atoms.positions)),
        }
    )
    assert result.energy_hartree == pytest.approx(sum(result.components_hartree.values()))


def test_force_call_uses_one_center_and_both_centered_stencils(tmp_path):
    provider = _AnalyticEnergyProvider()
    _wrapper(provider, tmp_path).evaluate(_atoms(), need_forces=True)

    # d=3N and C_force=1+4d.
    assert len(provider.calls) == 13
    contexts = [context for _, context in provider.calls]
    assert sum(context.step_role == "center" for context in contexts) == 1
    assert sum(context.step_role == "primary" for context in contexts) == 6
    assert sum(context.step_role == "check" for context in contexts) == 6


def test_only_wrapper_advertises_testing_only_non_native_forces(tmp_path):
    provider = _AnalyticEnergyProvider()
    wrapper = _wrapper(provider, tmp_path)

    assert provider.supported_properties == frozenset({"energy"})
    assert wrapper.supported_properties == frozenset({"energy", "forces"})
    assert wrapper.testing_only is True
    assert wrapper.native_force is False
    assert wrapper.derivative_source == (
        "centered-finite-difference-of-selected-provider-total"
    )


def test_energy_only_request_does_not_promote_or_call_a_force_fallback(tmp_path):
    provider = _AnalyticEnergyProvider()
    result = _wrapper(provider, tmp_path).evaluate(_atoms(), need_forces=False)

    assert len(provider.calls) == 1
    assert result.forces_hartree_per_angstrom is None
    assert provider.supported_properties == frozenset({"energy"})


def test_detached_displacements_remove_constraints_and_preserve_caller_state(tmp_path):
    atoms = _atoms()
    constraint = FixAtoms(indices=[0])
    atoms.set_constraint(constraint)
    calculator = _SentinelCalculator()
    atoms.calc = calculator
    before = atoms.positions.copy()
    constraint_list = atoms.constraints

    _wrapper(_AnalyticEnergyProvider(), tmp_path).evaluate(atoms, need_forces=True)

    assert atoms.positions == pytest.approx(before)
    assert atoms.calc is calculator
    assert atoms.constraints is constraint_list
    assert atoms.constraints[0] is constraint


def test_provider_exception_restores_caller_geometry_calculator_and_constraints(tmp_path):
    atoms = _atoms()
    constraint = FixAtoms(indices=[0])
    atoms.set_constraint(constraint)
    calculator = _SentinelCalculator()
    atoms.calc = calculator
    before = atoms.positions.copy()
    constraint_list = atoms.constraints

    with pytest.raises(RuntimeError, match="selected provider failed"):
        _wrapper(_AnalyticEnergyProvider(fail_on_call=4), tmp_path).evaluate(
            atoms, need_forces=True
        )

    assert atoms.positions == pytest.approx(before)
    assert atoms.calc is calculator
    assert atoms.constraints is constraint_list
    assert atoms.constraints[0] is constraint


def test_nonfinite_selected_provider_scalar_fails_without_returning_partial_force(
    tmp_path,
):
    provider = _AnalyticEnergyProvider(nonfinite=True)
    with pytest.raises(ValueError, match="non-finite"):
        _wrapper(provider, tmp_path).evaluate(_atoms(), need_forces=True)
    assert len(provider.calls) == 1


def test_each_scalar_attempt_has_a_unique_immutable_precreated_audit_context(tmp_path):
    provider = _AnalyticEnergyProvider()
    _wrapper(provider, tmp_path).evaluate(_atoms(), need_forces=True)
    contexts = [context for _, context in provider.calls]

    assert len({context.path for context in contexts}) == 13
    assert len({(context.force_call_id, context.scalar_ordinal) for context in contexts}) == 13
    with pytest.raises(FrozenInstanceError):
        contexts[0].scalar_ordinal = 99  # type: ignore[misc]


def test_repeated_and_concurrent_calls_use_unique_contexts_and_atomic_lifetime_counts(
    tmp_path,
):
    provider = _AnalyticEnergyProvider()
    wrapper = _wrapper(
        provider,
        tmp_path,
        max_scalar_evaluations=52,
        max_raw_records=56,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: wrapper.evaluate(_atoms(), need_forces=True), range(4)))

    assert all(result.forces_hartree_per_angstrom is not None for result in results)
    contexts = [context for _, context in provider.calls]
    assert len(contexts) == 52
    assert len({context.path for context in contexts}) == 52
    assert len({context.force_call_id for context in contexts}) == 4
    assert wrapper.budget_snapshot["consumed_scalar_evaluations"] == 52
    assert wrapper.budget_snapshot["consumed_raw_records"] == 56


def test_complete_force_budget_is_reserved_atomically_before_first_scalar_launch(tmp_path):
    provider = _AnalyticEnergyProvider()
    wrapper = _wrapper(
        provider,
        tmp_path,
        max_scalar_evaluations=12,
        max_raw_records=14,
    )

    with pytest.raises(RuntimeError, match="scalar.*budget"):
        wrapper.evaluate(_atoms(), need_forces=True)
    assert provider.calls == []
    assert wrapper.budget_snapshot["consumed_scalar_evaluations"] == 0
    assert wrapper.budget_snapshot["reserved_scalar_evaluations"] == 0


def test_failed_launched_force_consumes_its_entire_scalar_and_record_reservation(tmp_path):
    provider = _AnalyticEnergyProvider(fail_on_call=3)
    wrapper = _wrapper(provider, tmp_path, max_scalar_evaluations=13, max_raw_records=14)

    with pytest.raises(RuntimeError, match="selected provider failed"):
        wrapper.evaluate(_atoms(), need_forces=True)
    assert len(provider.calls) == 3
    assert wrapper.budget_snapshot["consumed_scalar_evaluations"] == 13
    assert wrapper.budget_snapshot["consumed_raw_records"] == 14
    with pytest.raises(RuntimeError, match="scalar.*budget"):
        wrapper.evaluate(_atoms(), need_forces=True)
    assert len(provider.calls) == 3


def test_invalid_geometry_releases_reservation_tail_before_any_provider_launch(tmp_path):
    provider = _AnalyticEnergyProvider()
    wrapper = _wrapper(provider, tmp_path)
    atoms = _atoms()
    atoms.positions[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite.*coordinates"):
        wrapper.evaluate(atoms, need_forces=True)
    assert provider.calls == []
    assert wrapper.budget_snapshot["consumed_scalar_evaluations"] == 0
    assert wrapper.budget_snapshot["reserved_scalar_evaluations"] == 0


def test_task_reservation_children_do_not_double_count_and_completion_reconciles():
    budget = NumericalForceBudget(
        max_scalar_evaluations=30,
        max_raw_records=40,
        max_audit_bytes=2_000,
    )
    task = budget.reserve_task(
        scalar_evaluations=26,
        raw_records=29,
        audit_bytes=1_000,
    )
    child = task.reserve_complete_force(atom_count=1, audit_bytes=400)
    child.mark_provider_launch()
    child.complete(actual_audit_bytes=40)

    assert budget.snapshot["reserved_scalar_evaluations"] == 26
    assert budget.snapshot["consumed_scalar_evaluations"] == 0
    task.reconcile(
        actual_scalar_evaluations=13,
        actual_raw_records=14,
        actual_audit_bytes=40,
    )
    assert budget.snapshot["reserved_scalar_evaluations"] == 0
    assert budget.snapshot["consumed_scalar_evaluations"] == 13
    assert budget.snapshot["consumed_raw_records"] == 14
    assert budget.snapshot["consumed_audit_bytes"] == 40


def test_unlaunched_task_abort_releases_all_reserved_capacity():
    budget = NumericalForceBudget(
        max_scalar_evaluations=30,
        max_raw_records=40,
        max_audit_bytes=2_000,
    )
    task = budget.reserve_task(
        scalar_evaluations=26,
        raw_records=29,
        audit_bytes=1_000,
    )
    task.abort()

    assert budget.snapshot["reserved_scalar_evaluations"] == 0
    assert budget.snapshot["consumed_scalar_evaluations"] == 0
    assert budget.snapshot["consumed_raw_records"] == 0
    assert budget.snapshot["consumed_audit_bytes"] == 0


def test_launched_task_failure_keeps_full_reservation_consumed():
    budget = NumericalForceBudget(
        max_scalar_evaluations=30,
        max_raw_records=40,
        max_audit_bytes=2_000,
    )
    task = budget.reserve_task(
        scalar_evaluations=26,
        raw_records=29,
        audit_bytes=1_000,
    )
    task.mark_provider_launch()
    task.abort()

    assert budget.snapshot["reserved_scalar_evaluations"] == 0
    assert budget.snapshot["consumed_scalar_evaluations"] == 26
    assert budget.snapshot["consumed_raw_records"] == 29
    assert budget.snapshot["consumed_audit_bytes"] == 1_000
