from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.analytic_protocol import (
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    ScalarCellIdentity,
    ScalarPointIdentity,
    freeze_identity_options,
    geometry_sha256,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.uma import _uma_calculator as uma_module
from maple.function.calculator.uma._uma_calculator import UMACalculator


def _tiny_checkpoint(path: Path) -> None:
    checkpoint = SimpleNamespace(
        model_config={
            "model_id": "UMA-test",
            "supports_single_atoms": False,
            "backbone": {
                "charge_balanced_channels": 8,
                "composition_dropout": 0.0,
                "dataset_mapping": {"omol": 0},
            },
            "heads": {"omol": {"dataset_mapping": {"omol": 0}}},
        },
        model_state_dict={"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        ema_state_dict={
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "step": torch.tensor(7, dtype=torch.int64),
        },
        tasks_config={"omol": {"property": "energy"}},
    )
    torch.save(checkpoint, path)


def test_compat_checkpoint_is_content_addressed_and_seals_semantic_receipt(
    tmp_path, monkeypatch
):
    source = tmp_path / "original.pt"
    _tiny_checkpoint(source)
    cache = tmp_path / "cache"
    legacy = cache / "maple_compat" / "uma-s-1p1.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"untrusted legacy cache")
    monkeypatch.setattr(uma_module, "CACHE_DIR", cache)

    compat = Path(UMACalculator._prepare_compat_checkpoint("uma-s-1p1", str(source)))

    assert compat != legacy
    assert compat.name.startswith("uma-s-1p1-maple-compat-v")
    assert legacy.read_bytes() == b"untrusted legacy cache"
    receipt_path = compat.with_suffix(compat.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source"]["path"] == str(source.resolve())
    assert receipt["output"]["path"] == str(compat.resolve())
    assert (
        receipt["tensor_payload"]["before_sha256"]
        == receipt["tensor_payload"]["after_sha256"]
    )
    assert receipt["tensor_payload"]["exactly_equal"] is True
    assert receipt["config_diff"]


def test_analytic_capability_does_not_remove_explicit_numerical_mode():
    assert UMACalculator.SUPPORTED_HESSIAN_MODES == ("analytic", "numerical")


def test_public_energy_force_calculation_does_not_initialize_hessian_predictor(
    monkeypatch,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    calculator = UMACalculator.__new__(UMACalculator)
    calculator._auto_task = False
    calculator._task_name = "omol"
    calculator.solvent_correction = None
    calculator.results = {}
    calculator._hessian_predictor_unit = None
    calculator._validate_task_atoms_compatibility = lambda _atoms: None
    calculator._validate_charge_spin_task_compatibility = lambda _atoms: (0, 1)
    calculator._set_task_from_atoms = lambda _atoms: None

    def fake_super_calculate(self, _atoms, _properties, _changes):
        self.results = {
            "energy": 2.0,
            "free_energy": 2.0,
            "forces": np.ones((2, 3)),
        }

    monkeypatch.setattr(
        uma_module.FAIRChemCalculator, "calculate", fake_super_calculate
    )
    monkeypatch.setattr(
        uma_module, "calculator_execution", lambda _calc: {"backend": "fake"}
    )

    result = calculator.calculate(atoms, properties=["energy", "forces"])

    assert calculator._hessian_predictor_unit is None
    assert result["energy"] == pytest.approx(2.0 * uma_module.EV2HARTREE)
    np.testing.assert_allclose(
        result["forces"], np.ones((2, 3)) * uma_module.EV2HARTREE
    )


def test_admitted_energy_force_identity_drift_is_transactional_and_precedes_solvent(
    monkeypatch,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    atoms.info.update(charge=0, mult=1)

    def cell(component, *, model):
        return ScalarCellIdentity(
            component=component,
            implementation=f"fixture-{component}-v1",
            model=model,
            checkpoint_path="/fixture/uma.pt" if component == "gas" else None,
            checkpoint_sha256="a" * 64 if component == "gas" else None,
            device="cpu",
            effective_dtype=(
                "torch.float32" if component == "gas" else "torch.float64"
            ),
            d4=None,
            source_energy_unit="eV" if component == "gas" else "hartree",
            cutoff_guard_policy=None,
            cutoff_guard_margin_angstrom=None,
            options=freeze_identity_options({"fixture": True}),
        )

    def point(identity_cell):
        return ScalarPointIdentity(
            cell=identity_cell,
            geometry_sha256=geometry_sha256(atoms),
            atomic_numbers=(1, 1),
            charge=0.0,
            multiplicity=1,
            atom_topology_sha256=None,
            neighbor_graph_sha256=None,
            active_cutoffs_angstrom=(),
            minimum_cutoff_gap_angstrom=None,
            response_settings=freeze_identity_options({"fixture": True}),
        )

    admitted_gas_cell = cell("gas", model="uma-s-1p1")
    drifted_gas_cell = cell("gas", model="uma-s-1p2")
    solvent_cell = cell("solvent", model="obc2-mbondi2")
    gas_admission = AnalyticDerivativeAdmission(
        profile_id="fixture-gas-v1",
        evidence_id="fixture-gas-evidence-v1",
        cell=admitted_gas_cell,
        operations=frozenset({"energy_forces"}),
    )
    solvent_admission = AnalyticDerivativeAdmission(
        profile_id="fixture-solvent-v1",
        evidence_id="fixture-solvent-evidence-v1",
        cell=solvent_cell,
        operations=frozenset({"energy_forces"}),
    )
    composition = AnalyticCompositionAdmission(
        profile_id="fixture-composition-v1",
        evidence_id="fixture-composition-evidence-v1",
        gas_admission=gas_admission,
        solvent_admission=solvent_admission,
        operations=frozenset({"energy_forces"}),
    )

    class Correction:
        supported_properties = frozenset({"energy", "forces"})
        analytic_derivative_admission = solvent_admission
        evaluate_calls = 0

        def analytic_scalar_identity(self, _atoms):
            return point(solvent_cell)

        def evaluate(self, _atoms, *, need_forces=False):
            self.evaluate_calls += 1
            return SolvationResult(
                energy_hartree=0.1,
                forces_hartree_per_angstrom=(np.zeros((2, 3)) if need_forces else None),
                components_hartree={"polar": 0.1, "nonpolar": 0.0},
                provenance={"provider": "fixture"},
            )

    correction = Correction()
    calculator = UMACalculator.__new__(UMACalculator)
    calculator.results = {"energy": 7.0, "forces": np.full((2, 3), 3.0)}
    calculator.execution_provenance = {"prior": True}
    calculator.solvation_result = {"prior": "solvent"}
    calculator.solvent_correction = correction
    calculator.analytic_derivative_admission = gas_admission
    calculator.analytic_composition_admission = composition
    calculator._predictor_unit = SimpleNamespace(
        inference_settings=SimpleNamespace(base_precision_dtype=torch.float32)
    )
    calculator.device = torch.device("cpu")
    calculator._hessian_predictor_unit = None
    calculator._analytic_energy_forces_operation = lambda _atoms, *, need_forces: (
        AnalyticOperationResult.energy_forces(
            identity=point(drifted_gas_cell),
            energy_hartree=-1.0,
            forces_hartree_per_angstrom=(np.zeros((2, 3)) if need_forces else None),
        )
    )

    def fail_raw_path(*_args, **_kwargs):
        raise AssertionError("admitted UMA E/F must not use the raw FAIR-Chem path")

    monkeypatch.setattr(uma_module.FAIRChemCalculator, "calculate", fail_raw_path)
    monkeypatch.setattr(
        uma_module, "calculator_execution", lambda _calc: {"backend": "fixture"}
    )

    prior_forces = calculator.results["forces"].copy()
    prior_execution = calculator.execution_provenance
    prior_solvation = calculator.solvation_result
    with pytest.raises(AnalyticIdentityError, match="cell identity"):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert correction.evaluate_calls == 0
    assert calculator.results["energy"] == 7.0
    np.testing.assert_array_equal(calculator.results["forces"], prior_forces)
    assert calculator.execution_provenance is prior_execution
    assert calculator.solvation_result is prior_solvation
    assert calculator._hessian_predictor_unit is None


def test_analytic_hessian_gas_failure_preserves_prior_mode_diagnostics():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    calculator = UMACalculator.__new__(UMACalculator)
    calculator.hessian = "analytic"
    prior_analytic = {"profile": "prior-analytic"}
    prior_numerical = {"step": 0.002}
    calculator.last_analytic_hessian_provenance = prior_analytic
    calculator.last_numerical_hessian_diagnostics = prior_numerical

    def fail_gas(_atoms):
        raise RuntimeError("gas-H failed")

    calculator._analytic_hessian_operation = fail_gas

    with pytest.raises(RuntimeError, match="gas-H failed"):
        calculator.get_hessian(atoms)

    assert calculator.last_analytic_hessian_provenance is prior_analytic
    assert calculator.last_numerical_hessian_diagnostics is prior_numerical


def test_analytic_hessian_solvent_failure_preserves_prior_mode_diagnostics(
    monkeypatch,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    calculator = UMACalculator.__new__(UMACalculator)
    calculator.hessian = "analytic"
    calculator.solvent_correction = SimpleNamespace(provider="fixture")
    calculator.analytic_composition_admission = SimpleNamespace(profile_id="fixture")
    calculator.analytic_derivative_admission = SimpleNamespace(profile_id="gas")
    calculator._last_uma_hessian_operation_provenance = {"operation": "fixture"}
    calculator._analytic_hessian_operation = lambda _atoms: object()
    prior_analytic = {"profile": "prior-analytic"}
    prior_numerical = {"step": 0.002}
    calculator.last_analytic_hessian_provenance = prior_analytic
    calculator.last_numerical_hessian_diagnostics = prior_numerical

    def fail_solvent(*_args, **_kwargs):
        raise RuntimeError("solvent-H failed")

    monkeypatch.setattr(uma_module, "compose_analytic_hessian", fail_solvent)

    with pytest.raises(RuntimeError, match="solvent-H failed"):
        calculator.get_hessian(atoms)

    assert calculator.last_analytic_hessian_provenance is prior_analytic
    assert calculator.last_numerical_hessian_diagnostics is prior_numerical


def test_successful_analytic_hessian_commits_then_clears_numerical_diagnostics(
    monkeypatch,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    calculator = UMACalculator.__new__(UMACalculator)
    calculator.hessian = "analytic"
    calculator.solvent_correction = None
    calculator.analytic_composition_admission = None
    calculator.analytic_derivative_admission = None
    calculator._last_uma_hessian_operation_provenance = {"operation": "fixture"}
    calculator._analytic_hessian_operation = lambda _atoms: object()
    calculator.last_analytic_hessian_provenance = {"profile": "prior-analytic"}
    calculator.last_numerical_hessian_diagnostics = {"step": 0.002}
    expected = np.eye(6)
    provenance = {"profile": "new-analytic"}
    monkeypatch.setattr(
        uma_module,
        "compose_analytic_hessian",
        lambda *_args, **_kwargs: (expected, provenance),
    )

    result = calculator.get_hessian(atoms)

    assert result is expected
    assert calculator.last_analytic_hessian_provenance is provenance
    assert calculator.last_numerical_hessian_diagnostics is None


def test_successful_numerical_hessian_commits_then_clears_analytic_provenance(
    monkeypatch,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    calculator = UMACalculator.__new__(UMACalculator)
    calculator.hessian = "numerical"
    calculator.solvent_correction = None
    calculator.last_analytic_hessian_provenance = {"profile": "prior-analytic"}
    calculator.last_numerical_hessian_diagnostics = {"step": 0.004}
    expected = np.eye(6)
    diagnostics = {"step": 0.002, "scheme": "central"}
    monkeypatch.setattr(
        uma_module,
        "numerical_hessian_from_atoms",
        lambda *_args, **_kwargs: (expected, diagnostics),
    )

    result = calculator.get_hessian(atoms)

    assert result is expected
    assert calculator.last_numerical_hessian_diagnostics == diagnostics
    assert calculator.last_analytic_hessian_provenance is None


def test_direct_hvp_head_graph_context_restores_training_and_forbids_dense_hessian():
    from maple.function.calculator.uma._analytic_adapter import efs_head_graph_context

    class RegressConfig:
        hessian = False

    class MLP_EFS_Head(torch.nn.Module):
        __module__ = "fairchem.core.models.uma.escn_md"

        def __init__(self):
            super().__init__()
            self.regress_config = RegressConfig()

    head = MLP_EFS_Head()
    head.eval()
    backbone = torch.nn.Linear(1, 1)
    backbone.eval()

    with efs_head_graph_context([head], backbone=backbone):
        assert head.training is True
        assert backbone.training is False
        assert head.regress_config.hessian is False

    assert head.training is False
    assert backbone.training is False


def test_direct_hvp_head_graph_context_restores_state_after_failure():
    from maple.function.calculator.uma._analytic_adapter import efs_head_graph_context

    head = SimpleNamespace(
        training=False,
        regress_config=SimpleNamespace(hessian=False),
        __class__=SimpleNamespace(),
    )
    # A non-FAIR-Chem head is rejected before any state change.
    with pytest.raises(RuntimeError, match="audited FAIR-Chem UMA EFS head"):
        with efs_head_graph_context([head], backbone=SimpleNamespace(training=False)):
            pass
    assert head.training is False
