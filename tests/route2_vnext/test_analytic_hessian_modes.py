from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.frequency.frequency import (
    FrequencyBase,
    PureFrozenMWFrequency,
)
from maple.solvation.derivatives.analytic import AnalyticHessianEvaluation
from maple.solvation.derivatives.molecular_modes import (
    analyze_hessian_evaluation,
    hessian_numerical_diagnostics,
    hessian_numerical_summary,
)

_REGISTERED_CPU_SCALAR_ID = (
    "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v3"
)
_CONTRACT_EVIDENCE_ID = "route2-pure-macepolar-torch-analytic-v3-contract-tests"
_REGISTERED_CPU_V4_SCALAR_ID = (
    "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4"
)
_FP32_V4_EVIDENCE_ID = "route2-pure-macepolar-torch-fp32-v4-engineering-tests"


def _evaluation(*, hessian: np.ndarray | None = None) -> AnalyticHessianEvaluation:
    raw = np.eye(9) if hessian is None else hessian
    return AnalyticHessianEvaluation(
        hessian_eV_per_A2=raw,
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id="route2-test-analytic-scalar-v1",
        device="cuda:0",
        component_energies_eV={"vacuum": -1.0, "ddpcm": -0.2, "cds": -0.05},
        diagnostics={
            "topology": {"regular": True, "margins": [0.2, 0.4]},
            "solver": {"residual": np.float64(1.0e-13)},
        },
    )


def test_analytic_hessian_evaluation_is_immutable_and_content_addressed():
    first = _evaluation()
    second = _evaluation()

    assert first.evaluation_sha256 == second.evaluation_sha256
    assert len(first.derivative_policy_sha256) == 64
    assert first.derivative_method == "torch-autograd"
    assert first.dtype == "float64"
    assert first.computation_dtype == "float64"
    assert first.storage_dtype == "float64"
    assert first.numerical_uncertainty_eV_per_A2 is None
    assert isinstance(first.component_energies_eV, MappingProxyType)
    assert isinstance(first.diagnostics, MappingProxyType)
    assert isinstance(first.diagnostics["topology"], MappingProxyType)
    assert first.diagnostics["topology"]["margins"] == (0.2, 0.4)
    assert not first.hessian_eV_per_A2.flags.writeable
    assert not first.forces_eV_per_A.flags.writeable
    with pytest.raises(ValueError):
        first.hessian_eV_per_A2[0, 0] = 2.0
    with pytest.raises(TypeError):
        first.diagnostics["new"] = True


def test_default_fp64_preserves_legacy_payload_digest():
    evaluation = _evaluation()

    assert evaluation.evaluation_sha256 == (
        "5c68eae13ce46bfc355d5ca39bd797e39f8ed627a70eefe6ab832f48f22d6da4"
    )


def test_fp32_computation_is_hash_bound_but_arrays_remain_float64_storage():
    fp64 = _evaluation()
    fp32 = AnalyticHessianEvaluation(
        hessian_eV_per_A2=np.eye(9, dtype=np.float32),
        forces_eV_per_A=np.zeros((3, 3), dtype=np.float32),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id="unregistered-engineering-test-fp32-scalar",
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"scientific_release_admitted": False},
        computation_dtype="float32",
    )

    assert fp32.dtype == "float32"
    assert fp32.computation_dtype == "float32"
    assert fp32.storage_dtype == "float64"
    assert fp32.hessian_eV_per_A2.dtype == np.float64
    assert fp32.forces_eV_per_A.dtype == np.float64
    assert fp32.evaluation_sha256 != fp64.evaluation_sha256


def test_registered_torch_v4_requires_bound_fp32_evidence():
    common = dict(
        hessian_eV_per_A2=np.eye(9),
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id=_REGISTERED_CPU_V4_SCALAR_ID,
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"scientific_release_admitted": False},
        provider_id="provider-v4",
        profile_id="profile-v4",
        computation_dtype="float32",
    )
    with pytest.raises(ValueError, match="require non-empty"):
        AnalyticHessianEvaluation(**common)
    with pytest.raises(ValueError, match="bound FP32 engineering"):
        AnalyticHessianEvaluation(
            **common, verification_evidence_id="unverified-fp32-claim"
        )

    admitted = AnalyticHessianEvaluation(
        **common, verification_evidence_id=_FP32_V4_EVIDENCE_ID
    )
    assert admitted.dtype == "float32"


@pytest.mark.parametrize(
    "scalar_contract_id,verification_evidence_id,computation_dtype,match",
    [
        (
            _REGISTERED_CPU_SCALAR_ID,
            _CONTRACT_EVIDENCE_ID,
            "float32",
            "v3 analytic results require float64 computation",
        ),
        (
            _REGISTERED_CPU_V4_SCALAR_ID,
            _FP32_V4_EVIDENCE_ID,
            "float64",
            "v4 analytic results require float32 computation",
        ),
    ],
)
def test_registered_torch_precision_must_match_version(
    scalar_contract_id, verification_evidence_id, computation_dtype, match
):
    with pytest.raises(ValueError, match=match):
        AnalyticHessianEvaluation(
            hessian_eV_per_A2=np.eye(9),
            forces_eV_per_A=np.zeros((3, 3)),
            energy_eV=-1.25,
            geometry_sha256="a" * 64,
            configuration_sha256="b" * 64,
            scalar_contract_id=scalar_contract_id,
            device="cpu",
            component_energies_eV={"vacuum": -1.25},
            diagnostics={"scientific_release_admitted": False},
            provider_id="provider",
            profile_id="profile",
            verification_evidence_id=verification_evidence_id,
            computation_dtype=computation_dtype,
        )


@pytest.mark.parametrize("invalid", ["", "float16", "torch.float32", None])
def test_computation_dtype_rejects_unsupported_values(invalid):
    kwargs = dict(
        hessian_eV_per_A2=np.eye(9),
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id="route2-test-analytic-scalar-v1",
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"regular": True},
        computation_dtype=invalid,
    )
    with pytest.raises(ValueError, match="computation_dtype"):
        AnalyticHessianEvaluation(**kwargs)


def test_analytic_hessian_hash_covers_raw_antisymmetry_and_diagnostics():
    raw = np.eye(9)
    raw[0, 1] = 0.25
    evaluation = _evaluation(hessian=raw)
    assert evaluation.maximum_antisymmetry_eV_per_A2 == pytest.approx(0.25)

    changed = AnalyticHessianEvaluation(
        hessian_eV_per_A2=raw,
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id="route2-test-analytic-scalar-v1",
        device="cuda:0",
        component_energies_eV={"vacuum": -1.0, "ddpcm": -0.2, "cds": -0.05},
        diagnostics={"topology": {"regular": False}},
    )
    assert changed.evaluation_sha256 != evaluation.evaluation_sha256


def test_registered_torch_v3_requires_explicit_bound_identities():
    kwargs = dict(
        hessian_eV_per_A2=np.eye(9),
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id=_REGISTERED_CPU_SCALAR_ID,
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"scientific_release_admitted": False},
    )
    for field in ("provider_id", "profile_id", "verification_evidence_id"):
        incomplete = {
            **kwargs,
            "provider_id": "provider-v3",
            "profile_id": "profile-v3",
            "verification_evidence_id": _CONTRACT_EVIDENCE_ID,
        }
        incomplete.pop(field)
        with pytest.raises(ValueError, match="require non-empty"):
            AnalyticHessianEvaluation(**incomplete)

    with pytest.raises(ValueError, match="bound contract verification"):
        AnalyticHessianEvaluation(
            **kwargs,
            provider_id="provider-v3",
            profile_id="profile-v3",
            verification_evidence_id="unverified-claim",
        )


def test_registered_identity_and_verification_fields_are_hash_bound():
    common = dict(
        hessian_eV_per_A2=np.eye(9),
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id=_REGISTERED_CPU_SCALAR_ID,
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"scientific_release_admitted": False},
        profile_id="profile-v3",
        verification_evidence_id=_CONTRACT_EVIDENCE_ID,
    )
    first = AnalyticHessianEvaluation(**common, provider_id="provider-v3")
    changed = AnalyticHessianEvaluation(**common, provider_id="other-provider-v3")

    assert first.evaluation_sha256 != changed.evaluation_sha256
    record = hessian_numerical_diagnostics(first)
    assert record["provider_id"] == "provider-v3"
    assert record["profile_id"] == "profile-v3"
    assert record["dtype"] == "float64"
    assert record["verification_evidence_id"] == _CONTRACT_EVIDENCE_ID
    summary = hessian_numerical_summary(first)
    assert "provider=provider-v3" in summary
    assert "profile=profile-v3" in summary
    assert "dtype=float64" in summary
    assert f"verification evidence={_CONTRACT_EVIDENCE_ID}" in summary


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"hessian_eV_per_A2": np.zeros((8, 8))}, "Hessian shape"),
        ({"hessian_eV_per_A2": np.full((9, 9), np.nan)}, "finite"),
        ({"forces_eV_per_A": np.zeros((9,))}, "forces_eV_per_A"),
        ({"energy_eV": np.inf}, "energy_eV"),
        ({"geometry_sha256": "not-a-digest"}, "geometry_sha256"),
        ({"scalar_contract_id": ""}, "scalar_contract_id"),
        ({"component_energies_eV": {"vacuum": np.nan}}, "finite"),
        ({"diagnostics": {"bad": object()}}, "JSON-compatible"),
    ],
)
def test_analytic_hessian_rejects_invalid_content(changes, match):
    kwargs = dict(
        hessian_eV_per_A2=np.eye(9),
        forces_eV_per_A=np.zeros((3, 3)),
        energy_eV=-1.25,
        geometry_sha256="a" * 64,
        configuration_sha256="b" * 64,
        scalar_contract_id="route2-test-analytic-scalar-v1",
        device="cpu",
        component_energies_eV={"vacuum": -1.25},
        diagnostics={"regular": True},
    )
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        AnalyticHessianEvaluation(**kwargs)


def test_analytic_modes_keep_eigenpairs_but_decline_certification():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    )
    analysis = analyze_hessian_evaluation(atoms, _evaluation())

    assert analysis.internal_dimension == 3
    assert analysis.eigenvalues_eV_per_A2_amu.shape == (3,)
    assert analysis.modes_cartesian.shape == (3, 9)
    assert analysis.richardson_error_eV_per_A2 is None
    assert analysis.symmetric_error_envelope_eV_per_A2 is None
    assert analysis.uncertainty_eV_per_A2_amu is None
    assert analysis.eigenvalue_intervals_eV_per_A2_amu is None
    assert analysis.statuses == ("uncertain",) * 3
    assert not analysis.is_resolved_minimum
    assert not analysis.is_resolved_index_one


def test_analytic_diagnostics_have_no_displaced_stencil_or_fake_zero_error():
    evaluation = _evaluation()
    record = hessian_numerical_diagnostics(evaluation)

    assert record["derivative_method"] == "torch-autograd"
    assert record["numerical_uncertainty"] is None
    assert record["eigensolver_device"] == "cpu"
    assert record["derivative_device"] == "cuda:0"
    assert record["model_device"] == "cuda:0"
    assert record["solvent_device"] == "cuda:0"
    assert record["provider_id"] is None
    assert record["profile_id"] is None
    assert record["dtype"] == "float64"
    assert record["verification_evidence_id"] is None
    assert "coarse_step_angstrom" not in record
    assert "stencil_order_per_column" not in record
    assert "maximum_richardson_error_eV_per_A2" not in record
    summary = hessian_numerical_summary(evaluation)
    assert "Numerical uncertainty: unavailable" in summary
    assert "eigensolver: cpu" in summary
    assert "Stencil" not in summary


def test_frequency_output_formats_unavailable_analytic_uncertainty(
    monkeypatch, tmp_path
):
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    )
    evaluation = _evaluation()
    analysis = analyze_hessian_evaluation(atoms, evaluation)
    output = tmp_path / "analytic-freq.out"
    job = PureFrozenMWFrequency.__new__(PureFrozenMWFrequency)
    job.output = str(output)
    job.mode_analysis = analysis
    job.hessian_evaluation = evaluation
    monkeypatch.setattr(FrequencyBase, "_write_output", lambda *args: None)

    job._write_output(np.array([]), np.empty((0, 9)), object())

    text = output.read_text()
    assert "Numerical uncertainty: unavailable" in text
    assert "eigensolver: cpu" in text
    assert "uncertainty unavailable; modes are not certified" in text
