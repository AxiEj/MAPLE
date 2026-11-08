from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.route2 import PureMACEPolarTorchCalculator
from maple.function.dispatcher.frequency.frequency import Frequency
from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.quantities import (
    CartesianCurvatureResult,
    FrequencyQuantityResult,
)
from maple.function.dispatcher.legacy_units import LegacyHartreeJobView
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
    route2_smd_profile_spec,
    route2_workflow_capability,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.analytic import AnalyticHessianEvaluation


def _atoms(*, deuterium: bool = False) -> Atoms:
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
    )
    atoms.info.update(charge=0, mult=1)
    if deuterium:
        atoms.arrays["masses"] = np.array([15.999, 2.01410177812, 2.01410177812])
    return atoms


class _StoredHessianPES:
    provider_id = "test-provider"
    device = "cpu"

    def __init__(self, spec, atoms: Atoms, hessian: np.ndarray):
        self.scalar_contract_id = spec.scalar_contract_id
        self._spec = spec
        self._hessian = np.array(hessian, dtype=np.float64, copy=True)
        self._geometry_sha256 = geometry_sha256(atoms)
        self.calls = 0

    @staticmethod
    def configuration_sha256() -> str:
        return "c" * 64

    def evaluate_hessian(self, atoms: Atoms) -> AnalyticHessianEvaluation:
        self.calls += 1
        assert geometry_sha256(atoms) == self._geometry_sha256
        return AnalyticHessianEvaluation(
            hessian_eV_per_A2=self._hessian,
            forces_eV_per_A=np.zeros((len(atoms), 3)),
            energy_eV=-1.0,
            geometry_sha256=self._geometry_sha256,
            configuration_sha256=self.configuration_sha256(),
            scalar_contract_id=self.scalar_contract_id,
            device="cpu",
            component_energies_eV={"test": -1.0},
            diagnostics={"raw_skew_is_error_bound": False},
            provider_id=self.provider_id,
            profile_id=self._spec.name,
            verification_evidence_id=(
                "route2-pure-macepolar-torch-fp32-v4-engineering-tests"
            ),
            computation_dtype="float32",
        )


def _calculator(atoms: Atoms, hessian: np.ndarray):
    spec = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    )
    pes = _StoredHessianPES(spec, atoms, hessian)
    calculator = PureMACEPolarTorchCalculator._from_test_pes(
        atoms=atoms,
        solvent="water",
        device="cpu",
        profile_spec=spec,
        pes=pes,
    )
    return calculator, pes


def _run(tmp_path: Path, method: str, *, deuterium: bool = False):
    atoms = _atoms(deuterium=deuterium)
    diagonal = np.array([-4.0, -1.0, -0.25, 0.125, 0.5, 1.0, 2.0, 4.0, 8.0])
    hessian = np.diag(diagonal)
    hessian[0, 1] = 0.2
    calculator, pes = _calculator(atoms, hessian)
    atoms.calc = LegacyHartreeJobView(calculator)
    output = tmp_path / f"{method}.out"
    result = Frequency(
        output=str(output), atoms=atoms, paras={"freq": {"method": method}}
    ).run()
    assert isinstance(result, FrequencyQuantityResult)
    return result, output.read_text(), pes, hessian


def test_cartesian_curvature_is_mass_independent_and_preserves_raw_signs(tmp_path):
    protium, text_h, _, raw = _run(tmp_path, "cartesian-curvature")
    deuterium, text_d, _, _ = _run(tmp_path, "cartesian-curvature", deuterium=True)

    assert isinstance(protium, FrequencyQuantityResult)
    assert isinstance(protium.cartesian_curvature, CartesianCurvatureResult)
    assert protium.canonical_method == "cartesian-curvature"
    assert protium.physical_mass_weighted is None
    np.testing.assert_allclose(
        protium.cartesian_curvature.eigenvalues_eV_per_A2,
        deuterium.cartesian_curvature.eigenvalues_eV_per_A2,
    )
    expected = np.linalg.eigvalsh(0.5 * (raw + raw.T))
    np.testing.assert_allclose(
        protium.cartesian_curvature.eigenvalues_eV_per_A2, expected
    )
    assert protium.cartesian_curvature.eigenvalues_eV_per_A2[0] < 0.0
    assert (
        protium.cartesian_curvature.raw_maximum_antisymmetry_eV_per_A2
        == pytest.approx(0.2)
    )
    assert protium.cartesian_curvature.independent_hessian_uncertainty is None
    assert protium.cartesian_curvature.interval_certified is None
    assert "cm^-1" not in text_h + text_d
    assert "ZPE" not in text_h + text_d
    assert "entropy" not in (text_h + text_d).lower()
    assert "gibbs" not in (text_h + text_d).lower()


def test_both_uses_one_hessian_and_reports_physical_plus_cartesian(tmp_path):
    result, text, pes, _ = _run(tmp_path, "both")

    assert pes.calls == 1
    assert result.canonical_method == "both"
    assert result.physical_mass_weighted is not None
    assert result.cartesian_curvature is not None
    assert result.physical_mass_weighted.thermochemistry_available is True
    assert result.cartesian_curvature.thermochemistry_available is False
    assert "MASS-WEIGHTED" in text
    assert "CARTESIAN CURVATURE" in text


def test_isotope_substitution_changes_mw_not_cartesian_spectrum(tmp_path):
    protium, _, _, _ = _run(tmp_path, "both")
    deuterium, _, _, _ = _run(tmp_path, "both", deuterium=True)
    assert protium.physical_mass_weighted is not None
    assert deuterium.physical_mass_weighted is not None

    np.testing.assert_allclose(
        protium.cartesian_curvature.eigenvalues_eV_per_A2,
        deuterium.cartesian_curvature.eigenvalues_eV_per_A2,
    )
    assert not np.allclose(
        protium.physical_mass_weighted.eigenvalues_eV_per_A2_amu,
        deuterium.physical_mass_weighted.eigenvalues_eV_per_A2_amu,
    )


def test_nonmw_alias_is_canonicalized_only_for_v4():
    command = CommandControl.from_settings(
        [
            "#model=macepolm",
            "#device=cpu",
            "#freq(method=nonmw)",
            "#solv(method=smd,provider=torch,implicit=water,experimental=true)",
        ]
    )
    assert command.params["method"] == "nonmw"
    capability = route2_workflow_capability(
        PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
        task="freq",
        method="cartesian-curvature",
    )
    assert capability.outcome_policy == "frequency_quantities"
    assert "deprecated_alias" not in command.params

    v3 = route2_smd_profile_spec(
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE
    )
    assert v3.workflow_policy_id is None
    with pytest.raises(ValueError, match="only with method=mw"):
        CommandControl.from_settings(
            [
                "#model=macepolm",
                "#device=cpu",
                "#freq(method=nonmw)",
                f"#solv(method=smd,provider=torch,profile={v3.name},implicit=water,experimental=true)",
            ]
        )


@pytest.mark.parametrize("method", ["cartesian-curvature", "both"])
def test_v4_frequency_quantity_policy_is_v2(method):
    capability = route2_workflow_capability(
        PURE_TORCH_FP32_V4_WORKFLOW_POLICY, task="freq", method=method
    )
    assert capability.execution_class == "bounded_execution"
    assert capability.outcome_policy == "frequency_quantities"


def test_unknown_frequency_method_still_fails_closed():
    with pytest.raises(ValueError, match="no capability"):
        route2_workflow_capability(
            PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
            task="freq",
            method="made-up-curvature",
        )


def test_quantity_record_owns_exact_source_and_unit_identity(tmp_path):
    result, _, _, _ = _run(tmp_path, "cartesian-curvature")
    record = result.to_final_metrics()
    cartesian = record["cartesian_curvature"]

    assert cartesian["eigenvalue_unit"] == "eV/angstrom^2"
    assert cartesian["eigenvector_basis"] == "full_cartesian_3N"
    assert cartesian["stored_operator"] == "symmetrized_cartesian_hessian"
    assert cartesian["thermochemistry_available"] is False
    assert len(cartesian["geometry_sha256"]) == 64
    assert len(cartesian["model_input_sha256"]) == 64
    assert len(cartesian["hessian_sha256"]) == 64
    assert len(cartesian["stored_operator_sha256"]) == 64
    assert len(cartesian["evaluation_sha256"]) == 64
    assert cartesian["computation_dtype"] == "float32"
    assert cartesian["device"] == "cpu"
    serialized = json.dumps(record, sort_keys=True).lower()
    for forbidden in ("cm^-1", "wavenumber", "zpe", "entropy", "gibbs"):
        assert forbidden not in serialized


def test_dispatcher_serializes_frequency_quantities_as_outcome_v2(tmp_path):
    atoms = _atoms()
    hessian = np.diag(np.arange(-4.0, 5.0))
    calculator, pes = _calculator(atoms, hessian)
    atoms.calc = calculator
    output = tmp_path / "dispatcher-cartesian.out"

    record = Dispatcher()(
        {"method": "cartesian-curvature", "treat_imag_as_real": False},
        "freq",
        atoms,
        str(output),
    )

    assert isinstance(record, dict)
    assert pes.calls == 1
    assert record["schema"] == "maple-route2-workflow-outcome-v2"
    assert record["method"] == "cartesian-curvature"
    assert record["task_success"] is True
    assert record["certificate_status"] == "not_applicable"
    assert record["final_metrics"]["canonical_method"] == "cartesian-curvature"
    assert "physical_mass_weighted" not in record["final_metrics"]


def test_raw_dispatcher_nonmw_uses_the_same_alias_normalization(tmp_path):
    atoms = _atoms()
    calculator, pes = _calculator(atoms, np.eye(9))
    atoms.calc = calculator

    record = Dispatcher()(
        {"method": "nonmw", "treat_imag_as_real": False},
        "freq",
        atoms,
        str(tmp_path / "raw-alias.out"),
    )

    assert isinstance(record, dict)
    assert pes.calls == 1
    assert record["method"] == "cartesian-curvature"
    assert record["final_metrics"]["deprecated_alias"] == "nonmw"


def test_non_v4_cartesian_curvature_rejects_before_hessian_evaluation(tmp_path):
    class Calculator:
        def __init__(self):
            self.calls = 0

        def get_hessian(self, atoms):
            self.calls += 1
            return np.eye(3 * len(atoms))

    atoms = _atoms()
    calculator = Calculator()
    atoms.calc = calculator
    with pytest.raises(ValueError, match="V4-only"):
        Frequency(
            output=str(tmp_path / "non-v4-cartesian.out"),
            atoms=atoms,
            paras={"freq": {"method": "cartesian-curvature"}},
        ).run()
    assert calculator.calls == 0


def test_deprecated_alias_marker_cannot_be_injected(tmp_path):
    atoms = _atoms()
    calculator, pes = _calculator(atoms, np.eye(9))
    atoms.calc = LegacyHartreeJobView(calculator)
    with pytest.raises(ValueError, match="deprecated_alias"):
        Frequency(
            output=str(tmp_path / "spoofed-alias.out"),
            atoms=atoms,
            paras={
                "freq": {
                    "method": "cartesian-curvature",
                    "deprecated_alias": "nonmw",
                }
            },
        )
    atoms.calc = calculator
    with pytest.raises(ValueError, match="deprecated_alias"):
        Dispatcher()(
            {
                "method": "both",
                "deprecated_alias": "nonmw",
                "treat_imag_as_real": False,
            },
            "freq",
            atoms,
            str(tmp_path / "spoofed-dispatcher.out"),
        )
    assert pes.calls == 0


def test_frequency_quantity_variants_reject_contradictory_payloads(tmp_path):
    cartesian, _, _, _ = _run(tmp_path, "cartesian-curvature")
    both, _, _, _ = _run(tmp_path, "both")
    assert both.physical_mass_weighted is not None
    with pytest.raises(ValueError, match="forbids physical"):
        FrequencyQuantityResult(
            canonical_method="cartesian-curvature",
            cartesian_curvature=cartesian.cartesian_curvature,
            physical_mass_weighted=both.physical_mass_weighted,
        )
    with pytest.raises(ValueError, match="requires physical"):
        FrequencyQuantityResult(
            canonical_method="both",
            cartesian_curvature=cartesian.cartesian_curvature,
        )
    with pytest.raises(ValueError, match="alias"):
        FrequencyQuantityResult(
            canonical_method="both",
            cartesian_curvature=cartesian.cartesian_curvature,
            physical_mass_weighted=both.physical_mass_weighted,
            deprecated_alias="nonmw",
        )
    with pytest.raises(ValueError, match="one Hessian"):
        FrequencyQuantityResult(
            canonical_method="both",
            cartesian_curvature=cartesian.cartesian_curvature,
            physical_mass_weighted=replace(
                both.physical_mass_weighted, hessian_sha256="0" * 64
            ),
        )


def test_cartesian_qa_failure_is_retained_as_task_failure(monkeypatch, tmp_path):
    import maple.function.dispatcher.frequency.quantities as quantity_module

    original_eigh = np.linalg.eigh

    def invalid_eigh(matrix):
        eigenvalues, eigenvectors = original_eigh(matrix)
        return eigenvalues, np.zeros_like(eigenvectors)

    monkeypatch.setattr(quantity_module.np.linalg, "eigh", invalid_eigh)
    atoms = _atoms()
    calculator, _ = _calculator(atoms, np.eye(9))
    atoms.calc = calculator
    output = tmp_path / "qa-failure.out"
    with pytest.raises(RuntimeError, match="computed_operator_qa_failed"):
        Dispatcher()(
            {"method": "cartesian-curvature", "treat_imag_as_real": False},
            "freq",
            atoms,
            str(output),
        )
    status = json.loads((tmp_path / "qa-failure_status.json").read_text())
    assert status["task_success"] is False
    assert status["termination_reason"] == "computed_operator_qa_failed"
    qa = status["final_metrics"]["cartesian_curvature"]["computed_operator_qa"]
    assert qa["passed"] is False
    assert qa["eigenvector_orthogonality_defect"] > qa["quality_gate"]
    assert (
        status["final_metrics"]["cartesian_curvature"][
            "independent_hessian_uncertainty"
        ]
        is None
    )


def test_both_physical_record_binds_masses_hessian_and_effective_knobs(tmp_path):
    result, _, _, _ = _run(tmp_path, "both", deuterium=True)
    physical = result.physical_mass_weighted
    assert physical is not None
    cartesian = result.cartesian_curvature
    np.testing.assert_allclose(
        physical.masses_amu, [15.999, 2.01410177812, 2.01410177812]
    )
    assert len(physical.mass_sha256) == 64
    assert physical.hessian_sha256 == cartesian.hessian_sha256
    assert physical.evaluation_sha256 == cartesian.evaluation_sha256
    assert physical.temperature_K == pytest.approx(298.15)
    assert physical.pressure_kPa == pytest.approx(101.325)
    assert physical.ilowfreq == 2
    assert physical.mode_basis == "cartesian_displacements"
    assert physical.mode_normalization == "Q.T@M@Q=I"


def test_both_reuses_full_mw_lifecycle_with_one_hessian(tmp_path):
    atoms = _atoms()
    calculator, pes = _calculator(atoms, np.eye(9))
    atoms.calc = LegacyHartreeJobView(calculator)
    output = tmp_path / "both-verbose.out"
    result = Frequency(
        output=str(output),
        atoms=atoms,
        paras={"freq": {"method": "both", "verbose": 10}},
    ).run()

    assert isinstance(result, FrequencyQuantityResult)
    assert pes.calls == 1
    assert getattr(calculator, "last_frequency_mode_analysis", None) is not None
    assert output.with_suffix(".sum").is_file()
    text = output.read_text()
    assert "Starting frequency analysis calculation" in text
    assert "Frequency analysis completed" in text
