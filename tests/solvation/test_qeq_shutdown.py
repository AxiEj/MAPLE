from __future__ import annotations

import inspect

import numpy as np
import pytest

from maple.function.calculator.extra_correction import implicit
from maple.function.calculator.extra_correction.charge.qeq import QEqGTO
from maple.function.calculator.extra_correction.implicit import charges
from maple.function.calculator.extra_correction.implicit.charges import prepare_charges
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
from maple.function.read.filereader.mol2_reader import MOL2Reader


@pytest.mark.parametrize(
    "options",
    [
        {"source": "maple", "method": "qeq-gto", "mode": "fixed"},
        {"source": "maple", "method": "qeq", "mode": "fixed"},
        {"source": "maple", "method": "cqeq-gto", "mode": "polarizable"},
        {"source": "maple", "method": "cqeq", "mode": "polarizable"},
        {"source": "maple", "method": "am1bcc", "mode": "polarizable"},
        {"source": "mol2", "mode": "polarizable"},
    ],
)
def test_runtime_charge_api_blocks_qeq_and_polarizable_modes_before_providers(
    water_mol2, tmp_path, monkeypatch, options
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider_calls = []

    def unexpected_provider(*_args, **_kwargs):
        provider_calls.append(True)
        raise AssertionError("disabled charge selection reached a provider")

    monkeypatch.setattr(QEqGTO, "solve", unexpected_provider)
    monkeypatch.setattr(charges, "_ambertools_charges", unexpected_provider)
    monkeypatch.setattr(charges, "_mol2_charges", unexpected_provider)

    with pytest.raises(ValueError, match="QEq/CQEq charge models are disabled"):
        prepare_charges(atoms, options, tmp_path)

    assert provider_calls == []


def test_deleted_polarizable_cqeq_gb_surface_is_absent():
    assert not hasattr(ImplicitSolvationCorrection, "_evaluate_polarizable_cqeq_gb")
    assert not hasattr(OpenMMGB, "polar_charge_hessian_ev")
    assert not hasattr(QEqGTO, "solve_variational")
    assert not hasattr(QEqGTO, "charge_gradient_ev")
    assert not hasattr(charges, "QEQ_EXPERIMENTAL_PROVENANCE")
    assert not hasattr(implicit, "QEqGTO")
    assert "QEqGTO" not in implicit.__all__
    assert "charges" not in inspect.signature(OpenMMGB.evaluate).parameters


def test_fixed_mol2_provider_remains_available_after_cqeq_removal(
    water_mol2, tmp_path
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    result = prepare_charges(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        tmp_path,
    )

    assert result.mode == "fixed"
    assert result.method == "mol2-fixed"
    assert np.allclose(result.charges, atoms.get_initial_charges())
    assert np.isfinite(result.charges).all()
