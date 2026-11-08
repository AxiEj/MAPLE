from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.derivatives.computed_stationary import (
    _assess_internal_operator,
    analyze_identified_hessian,
    assess_computed_stationary_point,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.analytic import AnalyticHessianEvaluation
from maple.solvation.derivatives.molecular_modes import analyze_hessian_evaluation
from maple.function.calculator.route2 import PureMACEPolarTorchCalculator
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    route2_smd_profile_spec,
)


def _evaluation(hessian: np.ndarray, atoms: Atoms):
    return AnalyticHessianEvaluation(
        hessian_eV_per_A2=hessian,
        forces_eV_per_A=np.zeros((len(atoms), 3)),
        energy_eV=0.0,
        evaluation_sha256="",
        geometry_sha256=geometry_sha256(atoms),
        configuration_sha256="c" * 64,
        scalar_contract_id="test-scalar",
        component_energies_eV={"test": 0.0},
        diagnostics={},
        provider_id="test-provider",
        profile_id="test-profile",
        verification_evidence_id="test-evidence",
        computation_dtype="float32",
        device="cpu",
    )


def _context(
    evaluation: AnalyticHessianEvaluation, atoms: Atoms, *, return_calculator=False
):
    spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    bound = AnalyticHessianEvaluation(
        hessian_eV_per_A2=evaluation.hessian_eV_per_A2,
        forces_eV_per_A=evaluation.forces_eV_per_A,
        energy_eV=evaluation.energy_eV,
        geometry_sha256=evaluation.geometry_sha256,
        configuration_sha256="c" * 64,
        scalar_contract_id=spec.scalar_contract_id,
        device="cpu",
        component_energies_eV=evaluation.component_energies_eV,
        diagnostics=evaluation.diagnostics,
        provider_id="test-provider",
        profile_id=spec.name,
        verification_evidence_id=(
            "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
        ),
        computation_dtype="float32",
    )

    class PES:
        scalar_contract_id = spec.scalar_contract_id
        provider_id = "test-provider"
        device = "cpu"

        def __init__(self):
            self.calls = []

        @staticmethod
        def configuration_sha256():
            return "c" * 64

        def evaluate_hessian(self, atoms):
            self.calls.append("hessian")
            return bound

    calculator = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=PES(),
    )
    context = calculator.get_identified_hessian_evaluation(atoms)
    return (context, calculator) if return_calculator else context


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        ([1.0, 2.0], (0, 2, 0)),
        ([-1.0, 2.0], (1, 1, 0)),
        ([-2.0, -1.0], (2, 0, 0)),
        ([0.0, 0.0], (0, 0, 2)),
    ],
)
def test_stored_operator_inertia_is_separate_from_hessian_uncertainty(
    diagonal, expected
):
    operator = np.diag(diagonal)
    result = _assess_internal_operator(
        operator,
        rigid_internal_overlap=np.zeros((0, len(diagonal))),
    )
    assert (
        result.computed_negative_count,
        result.computed_positive_count,
        result.computed_unresolved_count,
    ) == expected
    assert result.independent_hessian_uncertainty is None
    assert result.interval_certified is None
    assert result.computed_operator_quality_passed is True


def test_resolution_radius_keeps_near_zero_computed_mode_unresolved():
    operator = np.diag([1.0, 1e-15])
    result = _assess_internal_operator(
        operator,
        rigid_internal_overlap=np.zeros((0, 2)),
    )
    assert result.computed_positive_count == 1
    assert result.computed_unresolved_count == 1
    assert result.computed_resolution_radius > 1e-15
    assert result.computed_operator_quality_passed is True


def test_bad_eigensystem_fails_stored_operator_quality():
    operator = np.diag([-1.0, 2.0])
    bad_vectors = np.array([[1.0, 1.0], [0.0, 1.0]])
    result = _assess_internal_operator(
        operator,
        rigid_internal_overlap=np.zeros((0, 2)),
        eigensystem=(np.array([-1.0, 2.0]), bad_vectors),
    )
    assert result.computed_operator_quality_passed is False
    assert result.eigenvector_orthogonality_defect > result.quality_gate


@pytest.mark.parametrize(
    "operator",
    [np.array([[np.nan]]), np.ones((2, 3)), np.empty((0, 0))],
)
def test_invalid_stored_operator_fails_closed(operator):
    with pytest.raises(ValueError):
        _assess_internal_operator(
            operator,
            rigid_internal_overlap=np.zeros((0, operator.shape[-1])),
        )


def test_assessment_binds_geometry_mass_hessian_and_evaluation_identity():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    hessian = np.eye(9)
    evaluation = _evaluation(hessian, atoms)
    result = assess_computed_stationary_point(atoms, _context(evaluation, atoms))
    assert len(result.geometry_sha256) == 64
    assert len(result.mass_sha256) == 64
    assert len(result.hessian_sha256) == 64
    assert len(result.evaluation_sha256) == 64
    assert result.scalar_contract_id.endswith("-v4")
    assert result.computation_dtype == "float32"
    assert result.device == "cpu"
    assert result.independent_hessian_uncertainty is None


def test_nonpositive_mass_and_unidentified_evaluation_fail_closed():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    evaluation = _evaluation(np.eye(9), atoms)
    context = _context(evaluation, atoms)
    atoms.set_masses([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="masses"):
        assess_computed_stationary_point(atoms, context)

    atoms.set_masses([16.0, 1.0, 1.0])
    with pytest.raises(TypeError, match="identified analytic Hessian context"):
        assess_computed_stationary_point(atoms, evaluation)


def test_charge_change_rejects_an_old_hessian_at_identical_geometry():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    evaluation = _evaluation(np.eye(9), atoms)
    context = _context(evaluation, atoms)
    atoms.info["charge"] = 1
    atoms.info["mult"] = 2

    with pytest.raises(ValueError, match="model-input identity"):
        assess_computed_stationary_point(atoms, context)


def test_identified_analysis_runs_one_eigensolve(monkeypatch):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
        info={"charge": 0, "mult": 1},
    )
    evaluation = _evaluation(np.eye(9), atoms)
    original = np.linalg.eigh
    calls = []

    def counted(matrix):
        calls.append(np.asarray(matrix).shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    combined = analyze_identified_hessian(atoms, _context(evaluation, atoms))

    assert calls == [(3, 3)]
    assert combined.stationary_assessment.computed_minimum is True

    with pytest.raises(TypeError, match="unexpected keyword"):
        assess_computed_stationary_point(
            atoms, _context(evaluation, atoms), mode_analysis=object()
        )


def test_v4_postcondition_uses_computed_index_without_certifying_interval(
    monkeypatch,
):
    from maple.function.dispatcher.ts.algorithm.PRFO import (
        _pure_frozen_ts_postcondition,
    )

    atoms = Atoms(
        "NH3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
    )
    atoms.info.update(charge=0, mult=1)
    seed = _evaluation(np.eye(12), atoms)
    basis = analyze_hessian_evaluation(atoms, seed).internal_basis_mass_weighted
    spectrum = np.array([-2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mass_weighted = basis @ np.diag(spectrum) @ basis.T
    root_mass = np.sqrt(np.repeat(atoms.get_masses(), 3))
    hessian = root_mass[:, None] * mass_weighted * root_mass[None, :]
    evaluation = _evaluation(hessian, atoms)
    context, calculator = _context(evaluation, atoms, return_calculator=True)
    atoms.calc = calculator
    monkeypatch.setattr(
        "maple.function.calculator.route2.is_pure_mace_polar_workflow_calculator",
        lambda calc: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )

    diagnostic, interval_analysis = _pure_frozen_ts_postcondition(atoms)

    assert interval_analysis.is_resolved_index_one is False
    assert set(interval_analysis.statuses) == {"uncertain"}
    assert diagnostic["workflow_success"] is True
    assert diagnostic["numerical_stationary_point_validated"] is True
    assert diagnostic["certificate_status"] == "unavailable"
    assert diagnostic["scientific_certified"] is None
    assert calculator.pes.calls.count("hessian") == 1
