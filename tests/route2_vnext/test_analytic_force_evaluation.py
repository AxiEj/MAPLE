from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

from maple.solvation.derivatives.analytic import (
    ANALYTIC_FORCE_DERIVATIVE_POLICY_SHA256,
    ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256,
    AnalyticForceEvaluation,
)

REGISTERED_V3 = (
    "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v3"
)
REGISTERED_V4 = (
    "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v4"
)
V3_EVIDENCE = "route2-pure-macepolar-torch-analytic-v3-contract-tests"
V4_EVIDENCE = "route2-pure-macepolar-torch-fp32-v4-engineering-tests"


def _evaluation(**overrides) -> AnalyticForceEvaluation:
    values = {
        "forces_eV_per_A": np.arange(9, dtype=np.float32).reshape(3, 3) / 10,
        "energy_eV": -1.25,
        "geometry_sha256": "a" * 64,
        "model_input_sha256": "b" * 64,
        "configuration_sha256": "c" * 64,
        "scalar_contract_id": "route2-test-force-scalar-v1",
        "device": "cuda:0",
        "component_energies_eV": {
            "vacuum": -1.0,
            "ddpcm": -0.2,
            "cds": -0.05,
        },
        "computation_dtype": "float32",
    }
    values.update(overrides)
    return AnalyticForceEvaluation(**values)


def test_force_evaluation_is_immutable_content_addressed_and_force_specific():
    first = _evaluation()
    second = _evaluation()

    assert first.evaluation_sha256 == second.evaluation_sha256
    assert first.derivative_policy_sha256 == ANALYTIC_FORCE_DERIVATIVE_POLICY_SHA256
    assert first.derivative_policy_sha256 != ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256
    assert first.derivative_method == "torch-autograd"
    assert first.dtype == first.computation_dtype == "float32"
    assert first.storage_dtype == "float64"
    assert first.forces_eV_per_A.dtype == np.float64
    assert not first.forces_eV_per_A.flags.writeable
    assert isinstance(first.component_energies_eV, MappingProxyType)
    with pytest.raises(ValueError):
        first.forces_eV_per_A[0, 0] = 4.0
    with pytest.raises(TypeError):
        first.component_energies_eV["new"] = 1.0
    with pytest.raises(FrozenInstanceError):
        first.energy_eV = 2.0


def test_model_input_identity_is_hash_bound_independently_of_geometry():
    neutral = _evaluation(model_input_sha256="1" * 64)
    charged = _evaluation(model_input_sha256="2" * 64)

    assert neutral.geometry_sha256 == charged.geometry_sha256
    assert neutral.evaluation_sha256 != charged.evaluation_sha256


def test_model_input_identity_is_required():
    values = {
        "forces_eV_per_A": np.zeros((1, 3)),
        "energy_eV": 0.0,
        "geometry_sha256": "a" * 64,
        "configuration_sha256": "c" * 64,
        "scalar_contract_id": "route2-test-force-scalar-v1",
        "device": "cpu",
        "component_energies_eV": {"vacuum": 0.0},
    }
    with pytest.raises(TypeError, match="model_input_sha256"):
        AnalyticForceEvaluation(**values)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("forces_eV_per_A", np.zeros((3, 2)), "shape"),
        ("forces_eV_per_A", np.full((3, 3), np.nan), "finite"),
        ("energy_eV", float("inf"), "finite"),
        ("model_input_sha256", "not-a-digest", "model_input_sha256"),
        ("geometry_sha256", "A" * 64, "geometry_sha256"),
        ("component_energies_eV", {}, "must not be empty"),
        ("computation_dtype", "float16", "computation_dtype"),
    ],
)
def test_force_evaluation_rejects_invalid_content(field, value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _evaluation(**{field: value})


@pytest.mark.parametrize(
    ("scalar", "dtype", "evidence"),
    [
        (REGISTERED_V3, "float64", V3_EVIDENCE),
        (REGISTERED_V4, "float32", V4_EVIDENCE),
    ],
)
def test_registered_force_results_require_bound_identity(scalar, dtype, evidence):
    common = {
        "scalar_contract_id": scalar,
        "computation_dtype": dtype,
        "provider_id": "provider",
        "profile_id": "profile",
    }
    with pytest.raises(ValueError, match="require non-empty"):
        _evaluation(**common)
    with pytest.raises(ValueError, match="verification evidence"):
        _evaluation(**common, verification_evidence_id="unverified")

    admitted = _evaluation(**common, verification_evidence_id=evidence)
    assert admitted.verification_evidence_id == evidence


@pytest.mark.parametrize(
    ("scalar", "dtype", "evidence", "match"),
    [
        (REGISTERED_V3, "float32", V3_EVIDENCE, "v3.*float64"),
        (REGISTERED_V4, "float64", V4_EVIDENCE, "v4.*float32"),
    ],
)
def test_registered_force_precision_matches_version(scalar, dtype, evidence, match):
    with pytest.raises(ValueError, match=match):
        _evaluation(
            scalar_contract_id=scalar,
            computation_dtype=dtype,
            provider_id="provider",
            profile_id="profile",
            verification_evidence_id=evidence,
        )


def test_forged_force_evaluation_digest_is_rejected():
    original = _evaluation()
    with pytest.raises(ValueError, match="does not match analytic force content"):
        replace(original, energy_eV=-1.5)
