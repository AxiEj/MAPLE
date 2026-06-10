"""UMA MD contract: neighbor cutoff + unit contract + read-only task_name.

UMA extends FAIR Chemistry's ``FAIRChemCalculator`` rather than MAPLE's
``CalcABC``, so it must declare the MAPLE MD unit and neighbor-cutoff contracts
explicitly — otherwise the MD admission gate (``validate_energy_force_units`` +
``validate_pbc_neighbor_cutoff``) rejects it for PBC MD even with valid weights.
The PBC auto-task switch additionally hits a FAIR Chemistry compatibility risk:
``task_name`` is a read-only property, so MAPLE must assign to the backing
``_task_name`` attribute and refresh ``implemented_properties`` itself.

The tests stub the FAIR Chemistry predictor unit so they run without weights
(but require ``fairchem-core`` to be importable — UMACalculator hard-imports it
at module load).
"""

import pytest

pytest.importorskip("fairchem")

import torch
from ase import Atoms

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from maple.function.calculator.uma._uma_calculator import (
    UMA_NEIGHBOR_CUTOFF_A,
    UMACalculator,
)


# ── stub predictor unit ─────────────────────────────────────────────────────


class _StubTask:
    """Minimal stand-in for ``fairchem.core``'s per-task descriptor."""

    def __init__(self, prop):
        self.property = prop


class _StubInferenceSettings:
    # ``FAIRChemCalculator.__init__`` reads both fields when constructing its
    # ``AtomicData.from_ase`` partial. The dtype is only stored on the partial,
    # never executed, so float32 is enough for these construction-only tests.
    external_graph_gen = False
    base_precision_dtype = torch.float32


class _StubPredictUnit:
    """Enough surface for ``FAIRChemCalculator.__init__`` and the MAPLE auto-task switch.

    ``dataset_to_tasks`` lists the per-task properties that drive
    ``implemented_properties``; the ``omat`` entry includes ``stress`` so the
    PBC-side assertion below sees the refresh actually take effect.
    """

    inference_settings = _StubInferenceSettings()
    dataset_to_tasks = {
        "omol": [_StubTask("energy"), _StubTask("forces")],
        "omat": [_StubTask("energy"), _StubTask("forces"), _StubTask("stress")],
    }


def _build_stub_calculator(monkeypatch, *, task=None):
    monkeypatch.setattr(
        UMACalculator,
        "_build_predictor",
        staticmethod(lambda *args, **kwargs: _StubPredictUnit()),
    )
    return UMACalculator(device=torch.device("cpu"), task=task)


# ── class-level contracts ───────────────────────────────────────────────────


def test_uma_class_declares_neighbor_cutoff_constant():
    assert UMACalculator.neighbor_cutoff_A == UMA_NEIGHBOR_CUTOFF_A == 6.0
    assert UMACalculator.maple_requires_single_image_mic is False
    assert UMACalculator.maple_periodic_neighborlist_multi_image_safe is True


def test_uma_class_declares_maple_md_unit_contract():
    assert UMACalculator.maple_energy_unit == MAPLE_ENERGY_UNIT == "Ha"
    assert UMACalculator.maple_force_unit == MAPLE_FORCE_UNIT == "Ha/A"
    assert UMACalculator.maple_stress_unit == ASE_STRESS_UNIT == "eV/A^3"
    assert UMACalculator.maple_pbc_md_supported is True
    assert UMACalculator.maple_stress_supported is True


# ── instance contracts ─────────────────────────────────────────────────────


def test_uma_instance_carries_neighbor_cutoff_for_admission(monkeypatch):
    calc = _build_stub_calculator(monkeypatch)
    # Admission gates resolve the cutoff via ``getattr`` — must be readable as
    # a finite, positive float on a built instance, not only on the class.
    assert float(calc.neighbor_cutoff_A) == pytest.approx(6.0)


def test_uma_instance_declares_unit_contract(monkeypatch):
    calc = _build_stub_calculator(monkeypatch)
    assert calc.maple_energy_unit == "Ha"
    assert calc.maple_force_unit == "Ha/A"
    assert calc.maple_stress_unit == "eV/A^3"


# ── PBC auto-task switch + read-only task_name guard ───────────────────────


def test_pbc_auto_task_switch_assigns_through_backing_attribute(monkeypatch):
    calc = _build_stub_calculator(monkeypatch)  # task=None → auto-task on
    pbc_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    # No AttributeError on FAIRChemCalculator's read-only ``task_name`` property.
    calc._set_task_from_atoms(pbc_atoms)
    assert calc.task_name == "omat"


def test_pbc_auto_task_switch_refreshes_implemented_properties(monkeypatch):
    calc = _build_stub_calculator(monkeypatch)  # default omol from parent init
    # Sanity: the parent init seeded the omol property set.
    assert "stress" not in calc.implemented_properties
    pbc_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    calc._set_task_from_atoms(pbc_atoms)
    # After the switch, omat's stress property must be visible (otherwise NPT
    # would silently miss stress until the next full re-init).
    assert "stress" in calc.implemented_properties
    assert "free_energy" in calc.implemented_properties


def test_explicit_task_disables_auto_switch(monkeypatch):
    calc = _build_stub_calculator(monkeypatch, task="omat")
    # Explicit task pins ``_auto_task=False``; a non-PBC system must not flip
    # the task back to omol behind the user's back.
    iso_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], pbc=False)
    calc._set_task_from_atoms(iso_atoms)
    assert calc.task_name == "omat"
