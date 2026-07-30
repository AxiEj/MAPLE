from __future__ import annotations

import hashlib
import json

import pytest

from maple.function.calculator.model_capabilities import (
    GPU_FORBIDDEN_PRECISION_CONTROLS,
    GPU_STRICT_ACCURACY_CONTRACTS,
    ModelCardError,
    ModelCapabilities,
    ModelProvenanceCard,
)
from maple.function.calculator.set_calculator import SetCalculator

PRECISION_STATE = {
    "compute_dtype": "float64",
    "matmul_precision": "highest",
    **{name: False for name in GPU_FORBIDDEN_PRECISION_CONTROLS},
}
COMPARISON_MANIFEST_NAME = "synthetic-comparison-manifest.json"


def _output_sha256(record_id, observable, role, values):
    payload = {
        "observable": observable,
        "record_id": record_id,
        "role": role,
        "values": values,
    }
    return hashlib.sha256(
        (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode()
    ).hexdigest()


def _paired_outputs(observable, count):
    reference = [float(index) for index in range(count)]
    accelerator = list(reference)
    return [
        {
            "record_id": "case-1",
            "reference_values": reference,
            "accelerator_values": accelerator,
            "reference_sha256": _output_sha256(
                "case-1", observable, "reference", reference
            ),
            "accelerator_sha256": _output_sha256(
                "case-1", observable, "accelerator", accelerator
            ),
        }
    ]


def _with_paired_outputs(observables):
    return {
        name: {
            **record,
            "paired_outputs": _paired_outputs(name, record["comparison_count"]),
        }
        for name, record in observables.items()
    }


def _comparison_manifest_bytes(observables, task_modes):
    payload = {
        "schema_version": 1,
        "comparison_manifest_id": "synthetic-panel-v1",
        "model_id": "synthetic",
        "model_version": "1",
        "checkpoint_sha256s": ["c" * 64],
        "task_modes": task_modes,
        "records": [
            {
                "record_id": "case-1",
                "panel_id": "synthetic-panel",
                "configuration_sha256": "e" * 64,
                "reference_sha256": "f" * 64,
                "observable_output_sha256s": {
                    name: {
                        "reference": record["paired_outputs"][0]["reference_sha256"],
                        "accelerator": record["paired_outputs"][0][
                            "accelerator_sha256"
                        ],
                    }
                    for name, record in observables.items()
                },
            }
        ],
        "observable_counts": {
            name: record["comparison_count"] for name, record in observables.items()
        },
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


DEFAULT_OBSERVABLE_NAMES = (
    "coverage",
    "energy",
    "literature_panel_metrics",
    "maximum_error",
    "per_case_accuracy",
)
DEFAULT_OBSERVABLES = _with_paired_outputs(
    {name: {"comparison_count": 4} for name in DEFAULT_OBSERVABLE_NAMES}
)
COMPARISON_MANIFEST_BYTES = _comparison_manifest_bytes(
    DEFAULT_OBSERVABLES,
    ["sp:default"],
)
COMPARISON_MANIFEST_SHA256 = hashlib.sha256(COMPARISON_MANIFEST_BYTES).hexdigest()
RUNTIME_CONTEXT = {
    "accelerator_backend": "cuda",
    "hardware": "synthetic-gpu",
    "software_versions": {"runtime": "1"},
    "precision_controls": PRECISION_STATE,
}


def _contracts(capabilities):
    names = ModelCapabilities.from_payload(
        {"capabilities": capabilities}
    ).gpu_required_observables()
    return {
        name: {
            "unit": "synthetic-unit",
            "comparison": GPU_STRICT_ACCURACY_CONTRACTS.get(name, "absolute"),
            "threshold": 0.0,
        }
        for name in names
    }


def _card(*, capabilities=None, forbidden_tasks=None, **gpu_acceleration):
    capabilities = capabilities or {"energy": True}
    if gpu_acceleration.get("no_loss_parity_verified") is True:
        gpu_acceleration.setdefault("compute_dtype", "float64")
        gpu_acceleration.setdefault("matmul_precision", "highest")
        gpu_acceleration.setdefault("allowed_hardware", ["synthetic-gpu"])
        gpu_acceleration.setdefault("required_runtime_packages", ["runtime"])
        gpu_acceleration.setdefault(
            "comparison_manifest_artifact",
            COMPARISON_MANIFEST_NAME,
        )
        gpu_acceleration.setdefault("comparison_manifest_id", "synthetic-panel-v1")
        gpu_acceleration.setdefault(
            "comparison_manifest_sha256",
            COMPARISON_MANIFEST_SHA256,
        )
        gpu_acceleration.setdefault(
            "observable_contracts",
            _contracts(capabilities),
        )
    payload = {
        "model_id": "synthetic",
        "version": "1",
        "checkpoint_sha256": "c" * 64,
        "capabilities": capabilities,
    }
    if forbidden_tasks is not None:
        payload["forbidden_tasks"] = forbidden_tasks
    if gpu_acceleration:
        payload["gpu_acceleration"] = gpu_acceleration
    return ModelProvenanceCard.from_payload(payload, model_name="synthetic")


def _write_parity_artifact(
    path,
    *,
    manifest_observables=None,
    manifest_task_modes=None,
    **overrides,
):
    contracts = _contracts({"energy": True})
    observables = {
        name: {
            "passed": True,
            "comparison_count": 4,
            "unit": contract["unit"],
            "comparison": contract["comparison"],
            "threshold": contract["threshold"],
            "observed_maximum": 0.0,
        }
        for name, contract in contracts.items()
    }
    observables = _with_paired_outputs(observables)
    manifest_observables = _with_paired_outputs(manifest_observables or observables)
    manifest_task_modes = manifest_task_modes or ["sp:default"]
    manifest_bytes = _comparison_manifest_bytes(
        manifest_observables,
        manifest_task_modes,
    )
    (path.parent / COMPARISON_MANIFEST_NAME).write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    payload = {
        "schema_version": 2,
        "verdict": "pass_no_loss",
        "model_id": "synthetic",
        "model_version": "1",
        "checkpoint_sha256s": ["c" * 64],
        "comparison_manifest_id": "synthetic-panel-v1",
        "comparison_manifest_sha256": manifest_sha256,
        "allowed_tasks": ["sp"],
        "allowed_inference_modes": ["default"],
        "validated_task_modes": ["sp:default"],
        "precision_controls": {
            "reference_dtype": "float64",
            "accelerator_dtype": "float64",
            "same_scalar_precision": True,
            "matmul_precision": "highest",
            "autocast": False,
            "bfloat16": False,
            "fast_math": False,
            "float16": False,
            "reduced_matmul": False,
            "relaxed_convergence": False,
            "shortened_sampling": False,
            "tf32": False,
        },
        "runtime": {
            "reference_backend": "cpu",
            "accelerator_backend": "cuda",
            "hardware": "synthetic-gpu",
            "software_versions": {"runtime": "1"},
        },
        "observables": observables,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_gpu_policy_defaults_closed_but_leaves_cpu_unchanged():
    card = _card()

    card.validate_device("cpu", task="sp")
    for accelerator in (
        0,
        None,
        "auto",
        "cuda:0",
        "gpu",
        "mps",
        "xpu",
        "hip",
        "rocm",
        "unknown-accelerator",
    ):
        with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
            card.validate_device(accelerator, task="sp")


def test_gpu_policy_rejects_malformed_or_unbound_verified_evidence():
    with pytest.raises(ModelCardError, match="must be a mapping"):
        ModelProvenanceCard.from_payload(
            {
                "model_id": "synthetic",
                "capabilities": {"energy": True},
                "gpu_acceleration": True,
            },
            model_name="synthetic",
        )

    with pytest.raises(ModelCardError, match="requires a parity evidence artifact"):
        _card(no_loss_parity_verified=True, allowed_tasks=["sp"])

    with pytest.raises(ModelCardError, match="repository-relative path"):
        _card(
            no_loss_parity_verified=True,
            evidence_artifact="../outside.json",
            evidence_sha256="a" * 64,
            allowed_tasks=["sp"],
            allowed_inference_modes=["default"],
        )

    with pytest.raises(
        ModelCardError,
        match="comparison_manifest_artifact.*repository-relative",
    ):
        _card(
            no_loss_parity_verified=True,
            evidence_artifact="gpu-parity.json",
            evidence_sha256="a" * 64,
            comparison_manifest_artifact="../outside-panel.json",
            allowed_tasks=["sp"],
            allowed_inference_modes=["default"],
        )

    with pytest.raises(ModelCardError, match="allowed_inference_modes"):
        _card(
            no_loss_parity_verified=True,
            evidence_artifact="gpu-parity.json",
            evidence_sha256="a" * 64,
            allowed_tasks=["sp"],
        )

    with pytest.raises(ModelCardError, match="concrete compute_dtype"):
        _card(
            no_loss_parity_verified=True,
            evidence_artifact="gpu-parity.json",
            evidence_sha256="a" * 64,
            allowed_tasks=["sp"],
            allowed_inference_modes=["default"],
            compute_dtype="upstream-controlled",
        )


def test_verified_gpu_policy_is_hash_task_and_inference_mode_scoped(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    card.validate_device(
        "cuda",
        task="sp",
        inference_mode="default",
        evidence_root=tmp_path,
        runtime_context=RUNTIME_CONTEXT,
    )
    with pytest.raises(ValueError, match="explicit inference mode"):
        card.validate_device(
            "cuda",
            task="sp",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )
    with pytest.raises(ValueError, match="no admitted parity evidence for task"):
        card.validate_device(
            "cuda",
            task="frequency",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )
    with pytest.raises(ValueError, match="inference mode 'turbo'"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="turbo",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_rejects_missing_or_changed_evidence(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    expected_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=expected_sha256,
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    evidence.unlink()
    with pytest.raises(ValueError, match="evidence.*is missing"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )

    evidence.write_text('{"result":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_recomputes_comparison_manifest_hash(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )
    manifest = tmp_path / COMPARISON_MANIFEST_NAME

    manifest.unlink()
    with pytest.raises(ValueError, match="comparison manifest.*is missing"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )

    manifest.write_text('{"id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="comparison manifest SHA256 mismatch"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_comparison_manifest_cannot_be_empty_even_when_hashes_match(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    manifest = tmp_path / COMPARISON_MANIFEST_NAME
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["records"] = []
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_payload["comparison_manifest_sha256"] = manifest_sha256
    evidence.write_text(
        json.dumps(evidence_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        comparison_manifest_sha256=manifest_sha256,
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="records must not be empty"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_parity_counts_must_match_the_frozen_manifest(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    manifest = tmp_path / COMPARISON_MANIFEST_NAME
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["observable_counts"]["energy"] = 5
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_payload["comparison_manifest_sha256"] = manifest_sha256
    evidence.write_text(
        json.dumps(evidence_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        comparison_manifest_sha256=manifest_sha256,
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="comparison_count does not match"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_parity_requires_hash_bound_outputs_and_recomputed_aggregates(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    del payload["observables"]["energy"]["paired_outputs"]
    evidence.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="paired_outputs"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )

    _write_parity_artifact(evidence)
    manifest = tmp_path / COMPARISON_MANIFEST_NAME
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    pair = payload["observables"]["energy"]["paired_outputs"][0]
    pair["accelerator_values"][0] = 1.0
    pair["accelerator_sha256"] = _output_sha256(
        pair["record_id"],
        "energy",
        "accelerator",
        pair["accelerator_values"],
    )
    manifest_payload["records"][0]["observable_output_sha256s"]["energy"][
        "accelerator"
    ] = pair["accelerator_sha256"]
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload["comparison_manifest_sha256"] = manifest_sha256
    evidence.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        comparison_manifest_sha256=manifest_sha256,
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="observed_maximum.*recomputed"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_rejects_opaque_or_precision_reduced_artifacts(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    (tmp_path / COMPARISON_MANIFEST_NAME).write_bytes(COMPARISON_MANIFEST_BYTES)
    evidence.write_text('{"result":"no-loss"}\n', encoding="utf-8")
    opaque = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )
    with pytest.raises(ValueError, match="schema_version"):
        opaque.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )

    precision_controls = {
        "reference_dtype": "float64",
        "accelerator_dtype": "float32",
        "same_scalar_precision": True,
        "matmul_precision": "highest",
        "autocast": False,
        "bfloat16": False,
        "fast_math": False,
        "float16": False,
        "reduced_matmul": False,
        "relaxed_convergence": False,
        "shortened_sampling": False,
        "tf32": False,
    }
    _write_parity_artifact(evidence, precision_controls=precision_controls)
    reduced = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )
    with pytest.raises(ValueError, match="dtypes must match"):
        reduced.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_rejects_artifact_selected_loose_threshold(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["observables"]["energy"]["threshold"] = 999.0
    payload["observables"]["energy"]["observed_maximum"] = 999.0
    evidence.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="threshold looser than the model-card"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_requires_task_and_all_exposed_observables(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(
        evidence,
        observables={
            "energy": {
                "passed": True,
                "comparison_count": 1,
                "unit": "synthetic-unit",
                "comparison": "absolute",
                "threshold": 0.0,
                "observed_maximum": 0.0,
            }
        },
    )
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="explicit execution task"):
        card.validate_device("cuda", evidence_root=tmp_path)
    with pytest.raises(ValueError, match="observables must exactly match"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_verified_gpu_policy_requires_every_capability_exposed_task(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    capabilities = {
        "energy": True,
        "forces": True,
        "conservative_forces": True,
        "hessian": "finite_difference",
        "supports_md": True,
    }
    contracts = _contracts(capabilities)
    mechanics_observables = {
        name: {
            "passed": True,
            "comparison_count": 4,
            "unit": contract["unit"],
            "comparison": contract["comparison"],
            "threshold": contract["threshold"],
            "observed_maximum": 0.0,
        }
        for name, contract in contracts.items()
    }
    mechanics_observables = _with_paired_outputs(mechanics_observables)
    exposed_tasks = ["frequency", "irc", "md", "opt", "scan", "sp", "ts"]
    _write_parity_artifact(
        evidence,
        manifest_observables=mechanics_observables,
        manifest_task_modes=[f"{task}:default" for task in exposed_tasks],
        observables=mechanics_observables,
        allowed_tasks=exposed_tasks,
        validated_task_modes=[f"{task}:default" for task in exposed_tasks],
    )
    gpu_fields = {
        "no_loss_parity_verified": True,
        "status": "verified",
        "evidence_artifact": evidence.name,
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "comparison_manifest_sha256": hashlib.sha256(
            (tmp_path / COMPARISON_MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "allowed_tasks": ["sp"],
        "allowed_inference_modes": ["default"],
    }
    with pytest.raises(ModelCardError, match="every mechanically exposed task"):
        _card(capabilities=capabilities, **gpu_fields)

    gpu_fields["allowed_tasks"] = exposed_tasks
    gpu_fields["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    card = _card(capabilities=capabilities, **gpu_fields)
    for exposed_task in exposed_tasks:
        card.validate_device(
            "cuda",
            task=exposed_task,
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context=RUNTIME_CONTEXT,
        )


def test_force_api_requires_md_trajectory_parity_even_when_md_is_forbidden():
    capabilities = {
        "energy": True,
        "forces": True,
        "conservative_forces": True,
    }
    assert (
        "trajectory_or_sampled_observables"
        in ModelCapabilities.from_payload(
            {"capabilities": capabilities}
        ).gpu_required_observables()
    )

    with pytest.raises(
        ModelCardError,
        match="forbidden_tasks do not waive parity coverage",
    ):
        _card(
            capabilities=capabilities,
            forbidden_tasks=["md"],
            no_loss_parity_verified=True,
            status="verified",
            evidence_artifact="synthetic-gpu-parity.json",
            evidence_sha256="a" * 64,
            allowed_tasks=["irc", "opt", "scan", "sp", "ts"],
            allowed_inference_modes=["default"],
        )


def test_verified_gpu_policy_is_bound_to_current_hardware_and_runtime(tmp_path):
    evidence = tmp_path / "synthetic-gpu-parity.json"
    _write_parity_artifact(evidence)
    card = _card(
        no_loss_parity_verified=True,
        status="verified",
        evidence_artifact=evidence.name,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        allowed_tasks=["sp"],
        allowed_inference_modes=["default"],
    )

    with pytest.raises(ValueError, match="hardware does not match"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context={
                **RUNTIME_CONTEXT,
                "hardware": "different-gpu",
            },
        )
    with pytest.raises(ValueError, match="'runtime'.*does not match"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context={
                **RUNTIME_CONTEXT,
                "software_versions": {"runtime": "2"},
            },
        )
    with pytest.raises(ValueError, match="precision state must record tf32=false"):
        card.validate_device(
            "cuda",
            task="sp",
            inference_mode="default",
            evidence_root=tmp_path,
            runtime_context={
                **RUNTIME_CONTEXT,
                "precision_controls": {
                    **PRECISION_STATE,
                    "tf32": True,
                },
            },
        )


@pytest.mark.parametrize(
    "model_id",
    [
        "mace-off24-medium",
        "aimnet2-cpcms-v2",
        "aceff-2.0",
    ],
)
def test_route4_set_calculator_rejects_cuda_before_checkpoint_resolution(
    model_id, tmp_path
):
    setter = SetCalculator(
        device="cuda:0",
        model=model_id,
        output=str(tmp_path / "maple.out"),
        task="sp",
    )

    with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
        setter.set_calculator()


def test_route4_set_calculator_rejects_accelerator_without_an_execution_task(
    tmp_path,
):
    setter = SetCalculator(
        device=0,
        model="mace-off24-medium",
        output=str(tmp_path / "maple.out"),
    )

    with pytest.raises(ValueError, match="explicit execution task"):
        setter.set_calculator()


def test_route4_set_calculator_rejects_cuda_when_model_has_no_gpu_evidence_card(
    tmp_path,
):
    setter = SetCalculator(
        device="cuda:0",
        model="uma",
        output=str(tmp_path / "maple.out"),
        task="sp",
    )

    with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
        setter.set_calculator()


@pytest.mark.parametrize("model_id", ["uma", "aimnet2"])
def test_uncarded_registered_models_reject_accelerators_even_without_a_task(
    model_id,
    tmp_path,
):
    setter = SetCalculator(
        device="cuda:0",
        model=model_id,
        output=str(tmp_path / "maple.out"),
    )

    with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
        setter.set_calculator()


def test_calculator_without_stable_model_identity_cannot_use_an_accelerator(tmp_path):
    class AnonymousCalculator:
        MODEL_NAMES = ()

    setter = SetCalculator(
        device="cuda:0",
        model="anonymous",
        output=str(tmp_path / "maple.out"),
    )

    with pytest.raises(ValueError, match="stable MODEL_NAMES identity"):
        setter._validate_registered_model_acceleration(AnonymousCalculator)


def test_direct_route4_constructors_cannot_bypass_the_gpu_gate(tmp_path):
    from maple.function.calculator.aceff._aceff2_calculator import AceFF2Calculator
    from maple.function.calculator.aimnet._aimnet2_cpcms_calculator import (
        AIMNet2CPCMSCalculator,
    )
    from maple.function.calculator.extra_correction.implicit.anisolv import (
        AniSolvCompactBackend,
    )
    from maple.function.calculator.mace._maceoff24_calculator import (
        MACEOFF24MediumCalculator,
    )
    from maple.function.calculator.uma._anisolv_uma_calculator import (
        AniSolvUMACalculator,
    )
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    placeholder = tmp_path / "placeholder.model"
    placeholder.write_bytes(b"not loaded because the GPU gate runs first")
    constructors = (
        lambda: MACEOFF24MediumCalculator(
            device="cuda:0",
            model_path=str(placeholder),
            execution_task="sp",
        ),
        lambda: AIMNet2CPCMSCalculator(
            device="cuda:0",
            model_path=str(placeholder),
            execution_task="sp",
        ),
        lambda: AceFF2Calculator(
            device="cuda:0",
            model_path=str(placeholder),
            execution_task="sp",
        ),
        lambda: AniSolvCompactBackend(
            device="cuda:0",
            model_path=str(placeholder),
            solvent="water",
            execution_task="sp",
        ),
        lambda: AniSolvUMACalculator(
            device="cuda:0",
            base_model_path=str(placeholder),
            solvation_model_path=str(placeholder),
            execution_task="sp",
        ),
        lambda: UMACalculator(
            device="cuda:0",
            checkpoint_path=str(placeholder),
        ),
    )

    for construct in constructors:
        with pytest.raises(ValueError, match="no frozen no-loss CPU/GPU parity"):
            construct()


@pytest.mark.parametrize(
    "model_id",
    [
        "aimnet2-cpcms-v2",
        "mace-off24-medium",
        "aceff-2.0",
        "anisolv-compact",
        "anisolv-uma",
        "gnnis-reference",
    ],
)
def test_current_route4_gpu_capable_cards_are_explicitly_unverified(model_id):
    from pathlib import Path

    from maple.function.calculator.model_capabilities import (
        load_model_provenance_card,
    )

    root = (
        Path(__file__).resolve().parents[2]
        / "maple"
        / "function"
        / "calculator"
        / "model_cards"
    )
    card = load_model_provenance_card(model_id, root)

    assert card.gpu_acceleration.no_loss_parity_verified is False
    assert card.gpu_acceleration.status.startswith("blocked_pending_")
