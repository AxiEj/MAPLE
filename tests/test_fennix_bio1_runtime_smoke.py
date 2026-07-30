from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "pretrained-solvation-hub" / "run_fennix_bio1_runtime_smoke.py"
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "fennix-bio1-runtime-smoke-2026-07-31.json"
)
DEFAULT_RECEIPT_ROOT = Path(
    "/home/axie/.cache/maple-benchmarks/fennix/runtime-smoke-2026-07-31"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fennix_bio1_runtime_smoke", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _contains_timing_key(value) -> bool:
    if isinstance(value, dict):
        return any(
            "seconds" in key or _contains_timing_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_timing_key(item) for item in value)
    return False


def test_frozen_runtime_smoke_preserves_real_values_and_closed_gates():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["verdict"] == "runtime_mechanics_available_gpu_admission_blocked"
    assert payload["acceptance_eligible"] is False
    assert payload["identity"] == {
        "checkpoint_parameter_dtypes": ["float32"],
        "checkpoint_sha256": AUDIT.CHECKPOINT_SHA256,
        "fennol_revision": AUDIT.FENNOL_REVISION,
        "model_distribution_revision": AUDIT.MODEL_DISTRIBUTION_REVISION,
    }
    precision = payload["precision_observation"]
    assert precision["requested_precision"] == "float64"
    assert precision["cpu_preprocessed_coordinate_dtype"] == "float64"
    assert precision["gpu_preprocessed_coordinate_dtype"] == "float64"
    assert "float64" in precision["cpu_output_dtypes"]
    assert "float64" in precision["gpu_output_dtypes"]
    assert precision["cpu_gpu_bitwise_exact"] is False
    assert precision["absolute_energy_difference_ev"] == pytest.approx(
        3.3306690738754696e-16, rel=0, abs=0
    )
    assert precision["maximum_absolute_force_difference_ev_per_angstrom"] == (
        pytest.approx(1.7763568394002505e-15, rel=0, abs=0)
    )

    adapter = payload["runtime_receipts"]["maple_adapter_cpu"]
    assert adapter["energy_hartree"] == pytest.approx(
        -0.014841685682622237, rel=0, abs=0
    )
    assert adapter["hessian_shape"] == [9, 9]
    assert adapter["hessian_finite"] is True
    assert adapter["hessian_symmetric"] is True

    alchemical = payload["native_alchemical_kernel_mechanics_only"]
    assert alchemical["progress_schedule"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert alchemical["all_energy_coordinate_cell_and_lambda_derivatives_finite"]
    assert alchemical["all_reported_output_dtypes_float64"]
    assert alchemical["cpu_gpu_bitwise_exact"] is False
    assert alchemical["cpu_gpu_maximum_absolute_differences"] == {
        "atomic_energy_sum": pytest.approx(7.771561172376096e-16, rel=0, abs=0),
        "dE_dlambda_e_model_units": pytest.approx(1.2212453270876722e-15, rel=0, abs=0),
        "dE_dlambda_v_model_units": pytest.approx(2.0599841277224584e-17, rel=0, abs=0),
        "max_coordinate_gradient": pytest.approx(2.220446049250313e-15, rel=0, abs=0),
    }
    upcast = payload["runtime_parameter_upcast_mechanics_only"]
    assert upcast["original_checkpoint_file_unchanged"] is True
    assert upcast["original_checkpoint_parameter_dtypes"] == ["float32"]
    assert upcast["derived_runtime_parameter_dtypes"] == ["float64"]
    assert upcast["upcast_vs_original_changes_model_outputs"] is True
    assert upcast["largest_original_vs_upcast_reported_summary_difference"] == (
        pytest.approx(3.537531757802359e-08, rel=0, abs=0)
    )
    assert upcast["full_force_arrays_saved"] is False
    assert upcast["full_cell_gradient_arrays_saved"] is False
    assert max(upcast["upcast_cpu_gpu_maximum_absolute_differences"].values()) == (
        pytest.approx(1.3322676295501878e-15, rel=0, abs=0)
    )

    gates = payload["formal_gates"]
    assert gates == {
        "absolute_hydration_free_energy_protocol_run": False,
        "cpu_gpu_experimental_no_degradation_verified": False,
        "experimental_accuracy_evaluated": False,
        "gpu_admitted": False,
        "matched_qm_accuracy_passed": False,
        "matched_qm_timing_eligible": False,
        "multi_solvent_validated": False,
        "performance_claim_allowed": False,
        "ten_record_ten_distinct_primary_functional_group_panel_run": False,
    }
    assert "float32" in payload["rejected_paths"]["tinker_float32_bridge"]
    assert _contains_timing_key(payload) is False


def test_build_artifact_rejects_precision_or_bit_exact_reinterpretation():
    frozen = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cpu = deepcopy(frozen["runtime_receipts"]["cpu"])
    gpu = deepcopy(frozen["runtime_receipts"]["gpu"])
    comparison = deepcopy(frozen["runtime_receipts"]["comparison"])
    alchemical_cpu = deepcopy(
        frozen["runtime_receipts"]["native_alchemical_kernel_cpu"]
    )
    alchemical_gpu = deepcopy(
        frozen["runtime_receipts"]["native_alchemical_kernel_gpu"]
    )
    alchemical_upcast_cpu = deepcopy(
        frozen["runtime_receipts"]["native_alchemical_kernel_upcast_cpu"]
    )
    alchemical_upcast_gpu = deepcopy(
        frozen["runtime_receipts"]["native_alchemical_kernel_upcast_gpu"]
    )
    alchemical_upcast_comparison = deepcopy(
        frozen["runtime_receipts"]["native_alchemical_kernel_upcast_comparison"]
    )
    adapter = deepcopy(frozen["runtime_receipts"]["maple_adapter_cpu"])
    pip_freeze = "\n".join(frozen["environment_receipts"]["pip_freeze"]) + "\n"
    nvidia_smi = frozen["environment_receipts"]["nvidia_smi_csv"] + "\n"
    artifact_manifest = "\n".join(
        frozen["environment_receipts"]["original_artifact_sha256_manifest"]
    )
    source_hashes = frozen["source_receipt_sha256"]

    cpu["preprocessed_dtypes"]["coordinates"] = "float32"
    with pytest.raises(ValueError, match="not preprocessed as float64"):
        AUDIT.build_artifact(
            cpu=cpu,
            gpu=gpu,
            comparison=comparison,
            alchemical_cpu=alchemical_cpu,
            alchemical_gpu=alchemical_gpu,
            alchemical_upcast_cpu=alchemical_upcast_cpu,
            alchemical_upcast_gpu=alchemical_upcast_gpu,
            alchemical_upcast_comparison=alchemical_upcast_comparison,
            maple_adapter=adapter,
            pip_freeze=pip_freeze,
            nvidia_smi=nvidia_smi,
            artifact_manifest=artifact_manifest,
            source_hashes=source_hashes,
        )

    cpu = deepcopy(frozen["runtime_receipts"]["cpu"])
    comparison["cpu_gpu_bitwise_exact"] = True
    with pytest.raises(ValueError, match="non-bit-exact"):
        AUDIT.build_artifact(
            cpu=cpu,
            gpu=gpu,
            comparison=comparison,
            alchemical_cpu=alchemical_cpu,
            alchemical_gpu=alchemical_gpu,
            alchemical_upcast_cpu=alchemical_upcast_cpu,
            alchemical_upcast_gpu=alchemical_upcast_gpu,
            alchemical_upcast_comparison=alchemical_upcast_comparison,
            maple_adapter=adapter,
            pip_freeze=pip_freeze,
            nvidia_smi=nvidia_smi,
            artifact_manifest=artifact_manifest,
            source_hashes=source_hashes,
        )


def test_exact_cached_receipts_reproduce_frozen_artifact(tmp_path):
    receipt_root = Path(
        os.environ.get("MAPLE_FENNIX_RUNTIME_SMOKE_ROOT", DEFAULT_RECEIPT_ROOT)
    )
    required = {
        "cpu": receipt_root / "cpu-float64.json",
        "gpu": receipt_root / "gpu-float64.json",
        "comparison": receipt_root / "comparison.json",
        "alchemical_cpu": receipt_root / "alchemical-kernel-cpu-float64.json",
        "alchemical_gpu": receipt_root / "alchemical-kernel-gpu-float64.json",
        "alchemical_upcast_cpu": (receipt_root / "alchemical-kernel-cpu-upcast64.json"),
        "alchemical_upcast_gpu": (receipt_root / "alchemical-kernel-gpu-upcast64.json"),
        "alchemical_upcast_comparison": (
            receipt_root / "alchemical-kernel-upcast64-comparison.json"
        ),
        "maple_adapter": receipt_root / "maple-adapter-cpu-float64.json",
        "pip_freeze": receipt_root / "pip-freeze.txt",
        "nvidia_smi": receipt_root / "nvidia-smi.csv",
        "artifact_sha256": receipt_root / "artifact-sha256.txt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        pytest.skip("FeNNix runtime receipts are unavailable: " + ", ".join(missing))

    output = tmp_path / "runtime-smoke.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "combine",
            "--cpu",
            str(required["cpu"]),
            "--gpu",
            str(required["gpu"]),
            "--comparison",
            str(required["comparison"]),
            "--alchemical-cpu",
            str(required["alchemical_cpu"]),
            "--alchemical-gpu",
            str(required["alchemical_gpu"]),
            "--alchemical-upcast-cpu",
            str(required["alchemical_upcast_cpu"]),
            "--alchemical-upcast-gpu",
            str(required["alchemical_upcast_gpu"]),
            "--alchemical-upcast-comparison",
            str(required["alchemical_upcast_comparison"]),
            "--maple-adapter",
            str(required["maple_adapter"]),
            "--pip-freeze",
            str(required["pip_freeze"]),
            "--nvidia-smi",
            str(required["nvidia_smi"]),
            "--artifact-sha256",
            str(required["artifact_sha256"]),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == ARTIFACT.read_bytes()
