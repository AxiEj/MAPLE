from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator import calculator_base
from maple.function.calculator.calculator_base import CalcABC, HARTREE2EV
from maple.function.calculator.route2 import (
    PureMACEPolarDDXCalculator,
    PureMACEPolarTorchCalculator,
)
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
    route2_smd_profile_spec,
)


def _atoms() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        info={"charge": 0, "mult": 1},
    )


class _DDXPES:
    provider_id = "test.lifecycle.ddx"

    def __init__(self, scalar_contract_id: str, *, fail: bool = False) -> None:
        self.scalar_contract_id = scalar_contract_id
        self.fail = fail
        self.fail_force_x: float | None = None
        self.solve_calls = 0
        self.force_calls = 0
        self.hessian_calls = 0

    def configuration_sha256(self) -> str:
        return "d" * 64

    def solve(self, atoms):
        self.solve_calls += 1
        if self.fail:
            raise RuntimeError("injected DDX evaluation failure")
        return SimpleNamespace(total_energy_eV=4.25)

    def evaluate_forces(self, atoms, *, central_state=None):
        self.force_calls += 1
        if (
            self.fail_force_x is not None
            and float(atoms.positions[0, 0]) == self.fail_force_x
        ):
            raise RuntimeError("injected changed-geometry failure")
        state = central_state or self.solve(atoms)
        return SimpleNamespace(
            central_state=state,
            total_forces_eV_per_A=np.full((len(atoms), 3), 0.375),
        )

    def evaluate_hessian(self, atoms):
        self.hessian_calls += 1
        return SimpleNamespace(hessian_eV_per_A2=np.eye(3 * len(atoms)))


class _TorchPES:
    provider_id = "test.lifecycle.torch"

    def __init__(
        self,
        scalar_contract_id: str,
        *,
        fail: bool = False,
        drift_after_evaluation: bool = False,
    ) -> None:
        self.scalar_contract_id = scalar_contract_id
        self.device = "cpu"
        self.fail = fail
        self.drift_after_evaluation = drift_after_evaluation
        self.energy_calls = 0

    def configuration_sha256(self) -> str:
        if self.drift_after_evaluation and self.energy_calls:
            return "f" * 64
        return "e" * 64

    def get_potential_energy(self, atoms) -> float:
        self.energy_calls += 1
        if self.fail:
            raise RuntimeError("injected Torch evaluation failure")
        return 6.5


class _OrdinaryCalculator(CalcABC):
    implemented_properties = ["energy", "free_energy"]

    def __init__(self, source_unit: str) -> None:
        super().__init__()
        self.MODEL_ENERGY_UNIT = source_unit
        self.solvent_correction = object()
        self.correction_calls = 0

    def implicit_solv_energy(self, atoms) -> torch.Tensor:
        self.correction_calls += 1
        return torch.tensor(0.125)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        target = super().calculate(atoms, properties, system_changes)
        self._finalize_results(target, energy=2.75)


def _ddx(profile_name: str, *, fail: bool = False):
    atoms = _atoms()
    spec = route2_smd_profile_spec(profile_name)
    assert spec.scalar_contract_id is not None
    pes = _DDXPES(spec.scalar_contract_id, fail=fail)
    calc = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        model="macepolm",
        profile_spec=spec,
        pes=pes,
    )
    return atoms, calc, pes


def _torch(
    profile_name: str,
    *,
    fail: bool = False,
    drift_after_evaluation: bool = False,
):
    atoms = _atoms()
    spec = route2_smd_profile_spec(profile_name)
    assert spec.scalar_contract_id is not None
    pes = _TorchPES(
        spec.scalar_contract_id,
        fail=fail,
        drift_after_evaluation=drift_after_evaluation,
    )
    calc = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=pes,
    )
    return atoms, calc, pes


_FAMILIES = (
    pytest.param(
        lambda: _ddx(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE),
        id="ddx-v1",
    ),
    pytest.param(
        lambda: _ddx(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE),
        id="ddx-v2",
    ),
    pytest.param(
        lambda: _torch(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE),
        id="torch-v3",
    ),
    pytest.param(
        lambda: _torch(PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE),
        id="torch-v4",
    ),
)


def _evaluator_calls(pes) -> int:
    return int(getattr(pes, "solve_calls", 0)) + int(
        getattr(pes, "energy_calls", 0)
    )


def _install_lifecycle_spies(monkeypatch):
    calls = {"calculate": 0, "finalize": 0}
    calculate = CalcABC.calculate
    finalize = CalcABC._finalize_results

    def counted_calculate(self, *args, **kwargs):
        calls["calculate"] += 1
        return calculate(self, *args, **kwargs)

    def counted_finalize(self, *args, **kwargs):
        calls["finalize"] += 1
        return finalize(self, *args, **kwargs)

    monkeypatch.setattr(CalcABC, "calculate", counted_calculate)
    monkeypatch.setattr(CalcABC, "_finalize_results", counted_finalize)
    return calls


def test_route2_facades_directly_inherit_the_common_calculator_parent():
    assert PureMACEPolarDDXCalculator.__bases__ == (CalcABC,)
    assert PureMACEPolarTorchCalculator.__bases__ == (CalcABC,)


def test_result_composition_is_closed_and_route2_declares_complete_total_pes():
    composition = calculator_base.ResultComposition
    assert tuple(composition) == (
        composition.NATIVE_PLUS_OPTIONAL_ADDITIVE,
        composition.COMPLETE_TOTAL_PES,
    )
    assert CalcABC.RESULT_COMPOSITION is composition.NATIVE_PLUS_OPTIONAL_ADDITIVE
    assert (
        PureMACEPolarDDXCalculator.RESULT_COMPOSITION
        is composition.COMPLETE_TOTAL_PES
    )
    assert (
        PureMACEPolarTorchCalculator.RESULT_COMPOSITION
        is composition.COMPLETE_TOTAL_PES
    )


@pytest.mark.parametrize("factory", _FAMILIES)
def test_public_miss_uses_parent_and_finalizer_once_then_public_hit_uses_neither(
    factory, monkeypatch
):
    atoms, calc, pes = factory()
    atoms.calc = calc
    calls = _install_lifecycle_spies(monkeypatch)

    first = atoms.get_potential_energy()
    attempted = _evaluator_calls(pes)
    second = atoms.get_potential_energy()

    assert first == second
    assert calls == {"calculate": 1, "finalize": 1}
    assert attempted == 1
    assert _evaluator_calls(pes) == attempted


@pytest.mark.parametrize("factory", _FAMILIES)
def test_warm_cache_rejects_additive_correction_before_cached_return(factory, monkeypatch):
    atoms, calc, pes = factory()
    atoms.calc = calc
    atoms.get_potential_energy()
    previous_results = dict(calc.results)
    attempted = _evaluator_calls(pes)
    calc.solvent_correction = object()
    calls = _install_lifecycle_spies(monkeypatch)

    with pytest.raises(ValueError, match="complete total-PES"):
        atoms.get_potential_energy()

    assert calls == {"calculate": 0, "finalize": 0}
    assert _evaluator_calls(pes) == attempted
    assert calc.results == previous_results


@pytest.mark.parametrize("factory", _FAMILIES)
def test_direct_invalid_composition_enters_parent_but_not_finalizer_or_evaluator(
    factory, monkeypatch
):
    atoms, calc, pes = factory()
    calc.solvent_correction = object()
    calls = _install_lifecycle_spies(monkeypatch)

    with pytest.raises(ValueError, match="complete total-PES"):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"calculate": 1, "finalize": 0}
    assert _evaluator_calls(pes) == 0


@pytest.mark.parametrize("factory", _FAMILIES)
def test_direct_finalizer_rejects_invalid_composition_before_publication(factory):
    atoms, calc, _ = factory()
    calc.solvent_correction = object()
    calc.results = {"energy": -91.0}

    with pytest.raises(ValueError, match="complete total-PES"):
        calc._finalize_results(atoms, energy=7.0, unit="eV")

    assert calc.results == {"energy": -91.0}


@pytest.mark.parametrize(
    "factory,cache_names",
    (
        pytest.param(
            lambda: _ddx(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
            ),
            ("_energy_cache_key", "_force_cache_key"),
            id="ddx",
        ),
        pytest.param(
            lambda: _torch(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
            ),
            ("_cache_key",),
            id="torch",
        ),
    ),
)
def test_finalizer_failure_occurs_after_one_evaluation_without_cache_commit(
    factory, cache_names, monkeypatch
):
    atoms, calc, pes = factory()
    calls = {"calculate": 0, "finalize": 0}
    calculate = CalcABC.calculate

    def counted_calculate(self, *args, **kwargs):
        calls["calculate"] += 1
        return calculate(self, *args, **kwargs)

    def failing_finalize(self, *args, **kwargs):
        calls["finalize"] += 1
        raise RuntimeError("injected finalizer failure")

    monkeypatch.setattr(CalcABC, "calculate", counted_calculate)
    monkeypatch.setattr(CalcABC, "_finalize_results", failing_finalize)

    with pytest.raises(RuntimeError, match="injected finalizer failure"):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"calculate": 1, "finalize": 1}
    assert _evaluator_calls(pes) == 1
    assert calc.results == {}
    assert all(getattr(calc, name) is None for name in cache_names)


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(
            lambda: _ddx(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
            ),
            id="ddx",
        ),
        pytest.param(
            lambda: _torch(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
            ),
            id="torch",
        ),
    ),
)
def test_partial_finalizer_write_is_removed_without_retrying_evaluation(
    factory, monkeypatch
):
    atoms, calc, pes = factory()
    calls = {"finalize": 0}

    def partially_writing_finalize(self, *args, **kwargs):
        calls["finalize"] += 1
        self.results = {"energy": 123.0}
        raise RuntimeError("injected finalizer failure after partial write")

    monkeypatch.setattr(CalcABC, "_finalize_results", partially_writing_finalize)

    with pytest.raises(
        RuntimeError, match="injected finalizer failure after partial write"
    ):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"finalize": 1}
    assert _evaluator_calls(pes) == 1
    assert calc.results == {}


def test_ddx_changed_identity_failure_invalidates_old_caches_before_retry():
    atoms, calc, pes = _ddx(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE
    )
    calc.calculate(atoms, properties=["forces"], system_changes=all_changes)
    calc.get_hessian_evaluation(atoms)
    assert (pes.solve_calls, pes.force_calls, pes.hessian_calls) == (1, 1, 1)

    changed = atoms.copy()
    changed.positions[0, 0] += 0.125
    pes.fail_force_x = float(changed.positions[0, 0])
    with pytest.raises(RuntimeError, match="injected changed-geometry failure"):
        calc.calculate(changed, properties=["forces"], system_changes=all_changes)

    assert calc.results == {}
    assert calc._energy_cache_key is None
    assert calc._last_energy_state is None
    assert calc._force_cache_key is None
    assert calc._force_evaluation is None
    assert calc._hessian_cache_key is None
    assert calc._hessian_evaluation is None

    pes.fail_force_x = None
    calc.calculate(atoms, properties=["forces"], system_changes=all_changes)
    calc.get_hessian_evaluation(atoms)
    assert (pes.solve_calls, pes.force_calls, pes.hessian_calls) == (2, 3, 2)
    np.testing.assert_array_equal(calc.results["forces"], np.full((3, 3), 0.375))


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(
            lambda: _ddx(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
                fail=True,
            ),
            id="ddx",
        ),
        pytest.param(
            lambda: _torch(
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
                fail=True,
            ),
            id="torch",
        ),
    ),
)
def test_evaluator_failure_has_one_attempt_and_never_reaches_finalizer(
    factory, monkeypatch
):
    atoms, calc, pes = factory()
    calls = _install_lifecycle_spies(monkeypatch)

    with pytest.raises(RuntimeError, match="injected .* evaluation failure"):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"calculate": 1, "finalize": 0}
    assert _evaluator_calls(pes) == 1
    assert calc.results == {}


def test_target_identity_prevalidation_precedes_the_common_parent(monkeypatch):
    atoms, calc, pes = _torch(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )
    atoms.symbols[1] = "F"
    calls = _install_lifecycle_spies(monkeypatch)

    with pytest.raises(ValueError, match="configured symbols"):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"calculate": 0, "finalize": 0}
    assert _evaluator_calls(pes) == 0


def test_identity_drift_after_evaluation_prevents_finalization_and_cache_commit(
    monkeypatch,
):
    atoms, calc, pes = _torch(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
        drift_after_evaluation=True,
    )
    calls = _install_lifecycle_spies(monkeypatch)

    with pytest.raises(RuntimeError, match="changed during evaluation"):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)

    assert calls == {"calculate": 1, "finalize": 0}
    assert _evaluator_calls(pes) == 1
    assert calc.results == {}
    assert calc._cache_key is None


@pytest.mark.parametrize(
    ("source_unit", "expected"),
    (
        pytest.param("eV", 2.75 + 0.125 * HARTREE2EV, id="ev-native"),
        pytest.param("hartree", (2.75 + 0.125) * HARTREE2EV, id="hartree-native"),
    ),
)
def test_ordinary_calcabc_composes_one_additive_correction_in_public_ev(
    source_unit, expected
):
    atoms = _atoms()
    calc = _OrdinaryCalculator(source_unit)
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(expected, abs=1.0e-12)
    assert calc.correction_calls == 1
    assert calc.results["free_energy"] == calc.results["energy"]
