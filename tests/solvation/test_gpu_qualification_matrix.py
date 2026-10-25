"""CPU-only synthetic matrix mechanics; these records are not scientific evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from maple.function.calculator.gpu_qualification import (
    PERFORMANCE_CONTROLS,
    required_numerical_comparisons,
    validate_numerical_contract,
    validate_numerical_qualification,
)
from maple.function.calculator.model_capabilities import _gpu_output_sha256


def synthetic_matrix(
    root, *, forces=True, archetype=None, modes=("reference",), version="v1"
):
    """Minimal complete synthetic example; caller may serialize both returned maps."""
    archetype = archetype or ("native_energy_forces" if forces else "scalar_correction")
    protocol = f"{archetype}/{version}"
    source = root / "synthetic_implementation.py"
    if not source.exists():
        source.write_text("# synthetic implementation identity only\n")
    sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    contracts, comparisons = {}, {}
    for name, spec in required_numerical_comparisons(protocol, modes).items():
        contract = {
            **spec,
            "comparison": "equal" if spec["kind"] == "vacuum_zero" else "absolute",
            "threshold": 0.0 if spec["kind"] == "vacuum_zero" else 1e-4,
            "expected_count": 1,
            "cases": {"case-1": "a" * 64},
            "configuration": (
                {"step_A": 0.002}
                if "fd" in spec["kind"]
                else {"panel_id": "synthetic-only"}
            ),
        }
        contracts[name] = contract
        pair = _pair(name, [0.0], [0.0])
        comparisons[name] = {
            "contract": copy.deepcopy(contract),
            "paired_outputs": [pair],
            "comparison_count": 1,
            "observed_maximum": 0.0,
            "passed": True,
        }
    performance_contracts, performance = {}, {}
    if "optimized" in modes:
        contract = {**PERFORMANCE_CONTROLS, "cases": {"case-1": "a" * 64}}
        performance_contracts["optimized"] = contract
        pair = _pair("performance_optimized", [0.02] * 10, [0.01] * 10)
        pair.update(
            eager_median_s=0.02,
            candidate_median_s=0.01,
            device_total_bytes=2 * 1024**3,
            peak_reserved_bytes=1024**3,
            free_bytes_at_peak=1024**3,
            measured_headroom_bytes=1024**3,
        )
        performance["optimized"] = {
            "contract": copy.deepcopy(contract),
            "paired_outputs": [pair],
        }
    card = dict(
        protocol_id=protocol,
        implementation_source_sha256s=sources,
        required_contracts=contracts,
        performance_contracts=performance_contracts,
    )
    artifact = dict(
        protocol_id=protocol,
        implementation_source_sha256s=copy.deepcopy(sources),
        comparisons=comparisons,
        performance=performance,
    )
    return card, artifact


def _pair(name, reference, candidate):
    pair = {
        "record_id": "case-1",
        "configuration_sha256": "a" * 64,
        "reference_values": reference,
        "accelerator_values": candidate,
    }
    for role in ("reference", "accelerator"):
        pair[f"{role}_sha256"] = _gpu_output_sha256(
            record_id="case-1",
            observable=name,
            role=role,
            values=pair[f"{role}_values"],
        )
    return pair


def _validate(root, card, artifact, *, modes=("reference",), forces=True):
    canonical = validate_numerical_contract(
        card,
        modes=modes,
        forces=forces,
        expected_archetype=card["protocol_id"].split("/")[0],
    )
    validate_numerical_qualification(canonical, artifact, evidence_root=root)


@pytest.mark.parametrize(
    "archetype,forces",
    [
        ("scalar_correction", False),
        ("native_energy_forces", True),
        ("scalar_composition", False),
    ],
)
@pytest.mark.parametrize("modes", [("reference",), ("reference", "optimized")])
def test_complete_synthetic_matrix(tmp_path, archetype, forces, modes):
    card, artifact = synthetic_matrix(
        tmp_path, archetype=archetype, forces=forces, modes=modes
    )
    _validate(tmp_path, card, artifact, modes=modes, forces=forces)


@pytest.mark.parametrize(
    "name",
    [
        "independent_fd_cuda32_reference",
        "force_drift_cpu32",
        "solution_minus_vacuum_drift_cuda32_reference",
        "rigid_motion_force_cuda32_reference",
        "sampled_coordinates_cuda32_reference",
        "energy_drift_cpu32",
    ],
)
def test_force_model_cannot_omit_critical_comparison(tmp_path, name):
    card, artifact = synthetic_matrix(tmp_path)
    del artifact["comparisons"][name]
    with pytest.raises(ValueError, match="exactly"):
        _validate(tmp_path, card, artifact)
    del card["required_contracts"][name]
    with pytest.raises(ValueError, match="exactly"):
        _validate(tmp_path, card, artifact)


@pytest.mark.parametrize(
    "archetype,name",
    [
        ("scalar_correction", "vacuum_zero_cpu32"),
        ("scalar_correction", "energy_drift_cuda32_reference"),
        ("scalar_composition", "composite_scalar_cuda32_reference"),
        ("scalar_composition", "correction_parity_cuda32_reference"),
    ],
)
def test_scalar_model_cannot_omit_critical_comparison(tmp_path, archetype, name):
    card, artifact = synthetic_matrix(tmp_path, forces=False, archetype=archetype)
    del card["required_contracts"][name]
    del artifact["comparisons"][name]
    with pytest.raises(ValueError, match="exactly"):
        _validate(tmp_path, card, artifact, forces=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "Hartree"),
        ("threshold", 1.0),
        ("expected_count", 2),
        ("comparison", "non_degradation"),
        ("candidate", {"dtype": "float64"}),
    ],
)
def test_artifact_cannot_change_frozen_contract(tmp_path, field, value):
    card, artifact = synthetic_matrix(tmp_path)
    artifact["comparisons"]["force_drift_cpu32"]["contract"][field] = value
    with pytest.raises(ValueError, match="differs"):
        _validate(tmp_path, card, artifact)


@pytest.mark.parametrize(
    "mutation", ["values", "hash", "count", "aggregate", "case_hash", "extra"]
)
def test_comparisons_replay_hashes_counts_and_aggregates(tmp_path, mutation):
    card, artifact = synthetic_matrix(tmp_path)
    record = artifact["comparisons"]["force_drift_cpu32"]
    pair = record["paired_outputs"][0]
    if mutation == "values":
        pair["accelerator_values"] = [1.0]
    elif mutation == "hash":
        pair["accelerator_sha256"] = "b" * 64
    elif mutation == "count":
        record["comparison_count"] = 2
    elif mutation == "aggregate":
        record["observed_maximum"] = 1.0
    elif mutation == "case_hash":
        pair["configuration_sha256"] = "b" * 64
    else:
        artifact["comparisons"]["unrequested"] = record
    with pytest.raises(ValueError):
        _validate(tmp_path, card, artifact)


def test_equal_nonzero_vacuum_is_not_exact_zero(tmp_path):
    card, artifact = synthetic_matrix(tmp_path, forces=False)
    artifact["comparisons"]["vacuum_zero_cpu32"]["paired_outputs"] = [
        _pair("vacuum_zero_cpu32", [1.0], [1.0])
    ]
    with pytest.raises(ValueError, match="actual exact zero"):
        _validate(tmp_path, card, artifact, forces=False)


@pytest.mark.parametrize(
    "mutation", ["missing", "mismatch", "tamper", "escape", "absolute", "symlink"]
)
@pytest.mark.parametrize("forces", [False, True])
def test_implementation_source_identity_is_live_and_safe(tmp_path, mutation, forces):
    card, artifact = synthetic_matrix(tmp_path, forces=forces)
    if mutation == "missing":
        card["implementation_source_sha256s"] = {}
    elif mutation == "mismatch":
        artifact["implementation_source_sha256s"]["synthetic_implementation.py"] = (
            "b" * 64
        )
    elif mutation == "tamper":
        (tmp_path / "synthetic_implementation.py").write_text("# changed\n")
    elif mutation == "symlink":
        (tmp_path / "alias.py").symlink_to(tmp_path / "synthetic_implementation.py")
        card["implementation_source_sha256s"] = {
            "alias.py": next(iter(card["implementation_source_sha256s"].values()))
        }
        artifact["implementation_source_sha256s"] = card[
            "implementation_source_sha256s"
        ]
    else:
        name = "../escape.py" if mutation == "escape" else "/tmp/escape.py"
        card["implementation_source_sha256s"] = {name: "a" * 64}
    with pytest.raises(ValueError):
        _validate(tmp_path, card, artifact, forces=forces)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "summary",
        "samples",
        "median",
        "slow",
        "memory",
        "controls",
        "hash",
        "free_memory",
    ],
)
@pytest.mark.parametrize("forces", [False, True])
def test_optimized_requires_actual_speed_and_memory_inputs(tmp_path, mutation, forces):
    modes = ("reference", "optimized")
    card, artifact = synthetic_matrix(tmp_path, modes=modes, forces=forces)
    record = artifact["performance"]["optimized"]
    pair = record["paired_outputs"][0]
    if mutation == "missing":
        artifact["performance"] = {}
    elif mutation == "summary":
        artifact["performance"]["optimized"] = {"passed": True}
    elif mutation == "samples":
        del pair["accelerator_values"]
    elif mutation == "median":
        pair["candidate_median_s"] = 0.001
    elif mutation == "slow":
        pair.update(_pair("performance_optimized", [0.02] * 10, [0.03] * 10))
        pair["candidate_median_s"] = 0.03
    elif mutation == "memory":
        pair["peak_reserved_bytes"] = pair["device_total_bytes"] - 1
    elif mutation == "controls":
        record["contract"]["repetitions"] = 9
    elif mutation == "free_memory":
        pair["free_bytes_at_peak"] = 1
    else:
        pair["accelerator_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        _validate(tmp_path, card, artifact, modes=modes, forces=forces)


def test_reference_only_cannot_claim_optimized_performance(tmp_path):
    card, artifact = synthetic_matrix(tmp_path)
    artifact["performance"]["optimized"] = {"passed": True}
    with pytest.raises(ValueError, match="performance"):
        _validate(tmp_path, card, artifact)


def test_v2_fd_dtype_explicit_and_failed_fp32_diagnostic_preserved(tmp_path):
    card, artifact = synthetic_matrix(tmp_path, version="v2")
    name = "same_precision_fd_diagnostic_cuda32_reference"
    record = artifact["comparisons"][name]
    record.update(
        paired_outputs=[_pair(name, [0.0], [0.00210])],
        observed_maximum=0.00210,
        passed=False,
    )
    assert (
        card["required_contracts"]["independent_fd_cuda32_reference"]["reference"][
            "dtype"
        ]
        == "float64"
    )
    assert record["contract"]["reference"]["dtype"] == "float32"
    _validate(tmp_path, card, artifact)
    card["required_contracts"]["independent_fd_cuda32_reference"]["reference"][
        "dtype"
    ] = "float32"
    with pytest.raises(ValueError, match="dtype"):
        _validate(tmp_path, card, artifact)


@pytest.mark.parametrize(
    "forces,name",
    [
        (True, "independent_fd_cuda32_reference"),
        (True, "force_drift_cpu32"),
        (False, "energy_drift_cpu32"),
    ],
)
def test_hash_valid_failed_numerical_outputs_still_reject(tmp_path, forces, name):
    card, artifact = synthetic_matrix(tmp_path, forces=forces)
    artifact["comparisons"][name].update(
        paired_outputs=[_pair(name, [0.0], [0.01])], observed_maximum=0.01, passed=False
    )
    with pytest.raises(ValueError, match="failed frozen"):
        _validate(tmp_path, card, artifact, forces=forces)


@pytest.mark.parametrize(
    "field,value", [("unit", "Hartree"), ("threshold", 0.01), ("expected_count", True)]
)
def test_card_cannot_weaken_canonical_requirements(tmp_path, field, value):
    card, artifact = synthetic_matrix(tmp_path)
    card["required_contracts"]["force_drift_cpu32"][field] = value
    with pytest.raises(ValueError):
        _validate(tmp_path, card, artifact)


def test_bounded_device_admission_requires_embedded_matrix(tmp_path):
    from test_gpu_acceleration_policy import (
        BOUNDED_RUNTIME_CONTEXT,
        _bounded_card,
        _write_bounded_artifact,
    )

    evidence = tmp_path / "evidence.json"
    _write_bounded_artifact(evidence)
    payload = json.loads(evidence.read_text())
    del payload["numerical_qualification"]
    evidence.write_text(json.dumps(payload))
    card = _bounded_card(evidence)
    with pytest.raises(ValueError, match="numerical_qualification"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="reference",
            evidence_root=tmp_path,
            runtime_context=BOUNDED_RUNTIME_CONTEXT,
        )


def test_bounded_card_cannot_omit_matrix_contract(tmp_path):
    from test_gpu_acceleration_policy import _bounded_card, _write_bounded_artifact
    from maple.function.calculator.model_capabilities import ModelCardError

    evidence = tmp_path / "evidence.json"
    _write_bounded_artifact(evidence)
    with pytest.raises(ModelCardError, match="Numerical qualification"):
        _bounded_card(evidence, numerical_qualification=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_count", 2),
        ("cases", {"other": "b" * 64}),
        ("threshold", 2e-3),
        ("configuration", {"step_A": 0.002, "different": True}),
    ],
)
def test_v2_cannot_change_fd_panel_or_threshold(tmp_path, field, value):
    card, artifact = synthetic_matrix(tmp_path, version="v2")
    card["required_contracts"]["independent_fd_cuda32_reference"][field] = value
    with pytest.raises(ValueError, match="identical"):
        _validate(tmp_path, card, artifact)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("scalar_correction", "scalar_composition"),
        ("scalar_composition", "scalar_correction"),
    ],
)
def test_quantity_archetype_rejects_scalar_protocol_swap(tmp_path, actual, expected):
    card, _ = synthetic_matrix(tmp_path, forces=False, archetype=actual)
    with pytest.raises(ValueError, match="archetype mismatch"):
        validate_numerical_contract(
            card, modes=("reference",), forces=False, expected_archetype=expected
        )


@pytest.mark.parametrize("invalid_quantity", [None, [], "unknown"])
def test_bounded_card_requires_explicit_quantity_binding(
    tmp_path, monkeypatch, invalid_quantity
):
    from test_gpu_acceleration_policy import _bounded_card, _write_bounded_artifact
    from maple.function.calculator.model_capabilities import ModelProvenanceCard

    evidence = tmp_path / "synthetic.json"
    _write_bounded_artifact(evidence, capabilities={"energy": True})
    _bounded_card(evidence, capabilities={"energy": True}, allowed_tasks=["sp"])
    # The ordinary CPU card stays parseable without a quantity kind.
    ModelProvenanceCard.from_payload(
        {"model_id": "synthetic", "capabilities": {"energy": True}},
        model_name="synthetic",
    )
    with pytest.raises(ValueError, match="archetype mismatch"):
        _bounded_card(
            evidence,
            capabilities={"energy": True},
            allowed_tasks=["sp"],
            quantity_kind="combined_solution_energy",
        )

    original = ModelProvenanceCard.from_payload.__func__

    def without_quantity(cls, payload, model_name):
        payload = copy.deepcopy(payload)
        payload.pop("quantity_kind", None)
        if invalid_quantity is not None:
            payload["quantity_kind"] = invalid_quantity
        return original(cls, payload, model_name)

    with monkeypatch.context() as patch:
        patch.setattr(
            ModelProvenanceCard, "from_payload", classmethod(without_quantity)
        )
        with pytest.raises(ValueError, match="requires explicit quantity_kind"):
            _bounded_card(evidence, capabilities={"energy": True}, allowed_tasks=["sp"])
    with pytest.raises(ValueError, match="Quantity kind/capability"):
        _bounded_card(evidence, quantity_kind="geometry_level_energy_correction")


@pytest.mark.parametrize(
    "key",
    [
        "implementation_addendum",
        "preserved_failure_index",
        "composite_production_performance",
    ],
)
def test_packaged_methodology_references_are_sha_bound(tmp_path, key):
    card, artifact = synthetic_matrix(tmp_path)
    source = tmp_path / "synthetic-methodology.json"
    source.write_text('{"synthetic": true}')
    reference = {
        "artifact": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    for name, contract in card["required_contracts"].items():
        contract["configuration"][key] = reference
        artifact["comparisons"][name]["contract"] = copy.deepcopy(contract)
    _validate(tmp_path, card, artifact)
    source.write_text('{"synthetic": "changed"}')
    with pytest.raises(ValueError, match="Methodology artifact SHA256 mismatch"):
        _validate(tmp_path, card, artifact)
    source.unlink()
    with pytest.raises(ValueError, match="Methodology artifact is unavailable"):
        _validate(tmp_path, card, artifact)
