#!/usr/bin/env python3
"""Run and freeze the production FeNNix native-kernel mechanics smoke.

CPU and GPU workers execute in separate processes because JAX backend
selection is process-global.  This script records mechanics only: it performs
no sampling, HFE estimation, experimental accuracy evaluation, matched-QM
timing, or performance admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

CHECKPOINT_SHA256 = "5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a"
CHECKPOINT_SOURCE_REVISION = "83f299b81c1d62e2a15c892280559a7c0cc2fac3"
FENNOL_SOURCE_REVISION = "d62b8740343b803a2b864140ec79e347f8ba034e"
ORIGINAL_PARAMETER_TREE_FINGERPRINT = (
    "e0554dc2af0a652de5f42be138eb69449d519c89bad972e56c88403488e06e11"
)
DERIVED_PARAMETER_TREE_FINGERPRINT = (
    "0fe05dfdbe83b110a3a7c4bc42c3d5c7400645169721ad2d9aa3bf836e735d5c"
)
FIXED_SPECIES_ENCODING_FLOAT64_SHA256 = (
    "6e9ddf0e4b7c91aa0c8b6fdd4af617ec87f5abc0f51c0c7262395569c22dbe5b"
)
RUNTIME_VERSION = "2026.6.29"
CUTOFF_ANGSTROM = 7.5
MINIMUM_ENERGY_SPAN_EV = 1.0e-3
MINIMUM_ACTIVE_DERIVATIVE_EV = 1.0e-3
PROGRESS = (0.0, 0.25, 0.5, 0.75, 1.0)
ATOMIC_NUMBERS = (8, 1, 1, 8, 1, 1)
COORDINATES_ANGSTROM = (
    (2.0, 2.0, 2.0),
    (2.9572, 2.0, 2.0),
    (1.760013, 2.927297, 2.0),
    (6.0, 2.0, 2.0),
    (6.9572, 2.0, 2.0),
    (5.760013, 2.927297, 2.0),
)
CELL_ANGSTROM = (
    (18.0, 0.0, 0.0),
    (0.0, 18.0, 0.0),
    (0.0, 0.0, 18.0),
)
MOLECULE_IDS = (0, 0, 0, 1, 1, 1)
SOLUTE_ATOM_INDICES = (0, 1, 2)
EXPECTED_RUNTIME_SOURCE_SHA256 = {
    "fennol/ase.py": "6d6a9dbf40fce96b8589a3702de7de190f8301d0827f629fbfad2c1e4a2f97d7",
    "fennol/models/embeddings/charge_embeddings.py": (
        "b6e731a217bca04d9ed0b3a472175839283fce4b2f34edd14c25e0b3ecd96114"
    ),
    "fennol/models/fennix.py": (
        "2e092b782b2fcabf2b1d355274d34888faa3b6aca8bee60e44fcea71245991af"
    ),
    "fennol/models/misc/encodings.py": (
        "cfe1a486efc1f0a35d16a50dc09c5f28f717bb25d0cc63254f9e201aee3f1aae"
    ),
    "fennol/models/modules.py": (
        "afa28295acf9935cfa633d7ad03bec7100b1c7788f2b80ec0fa89db4e2cb19f0"
    ),
    "fennol/models/physics/repulsion.py": (
        "ee952d840432090a38bf9eedda2429b7f3a94c4b2f0d1f9dac27d2297f165c22"
    ),
    "fennol/models/preprocessing.py": (
        "153519f469e112b6f1d2bfe7f243c241db476f273543abd0540ab658d59c96d4"
    ),
    "fennol/utils/periodic_table.py": (
        "e6b2dd254305c0c065fdcb41249fa80ed92e8b0e00b16dfa0a039ea3f313006c"
    ),
}
EXPECTED_MAPLE_KERNEL_SOURCE_SHA256 = {
    "__init__.py": "1c37c43276769db8cbb5e6c1cf3599acafe90f7ea6670a1af9d62c03efa127d7",
    "kernel.py": "5d0aa3f31d293529849f7747bd47089656b8dbfca76865777c7935b3291c874d",
    "system.py": "fb18b37ad5f93567bb7b4d982f798113f244ea80003569a8b4063b26c12edfe8",
    "types.py": "eb724276ff0ffbc328ebce000dd12ea7fc48750e98561c26fa99f5532b115d1c",
}
FORMAL_GATES = {
    "absolute_hydration_free_energy_protocol_run": False,
    "experimental_accuracy_evaluated": False,
    "ten_record_ten_distinct_primary_functional_group_panel_run": False,
    "cpu_gpu_experimental_no_degradation_verified": False,
    "gpu_production_admitted": False,
    "matched_qm_timing_eligible": False,
    "performance_claim_allowed": False,
}
EXPECTED_ALCHEMICAL_PARAMETERS = {
    "graph_softcore_v_angstrom": 0.5,
    "repulsion_softcore_angstrom": 0.5,
    "repulsion_power_m": 2,
    "graph_softcore_provenance": "pinned_fennol_source_default",
    "repulsion_power_provenance": "pinned_fennol_source_default",
    "repulsion_softcore_provenance": (
        "maple_output_blind_reconstruction_not_paper_exact"
    ),
    "scientific_scope": (
        "maple_owned_softcore_reconstruction_mechanics_not_paper_reproduction"
    ),
}
EXPECTED_KERNEL_SCOPE = (
    "maple_owned_softcore_reconstruction_mechanics_not_paper_reproduction_"
    "not_hfe_not_accuracy_not_gpu_admission_not_performance"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_types():
    """Load only the production FeNNix package in the isolated FeNNol env."""

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    __import__("maple.function")

    solvfe = types.ModuleType("maple.function.solvfe")
    solvfe.__path__ = [str(root / "maple" / "function" / "solvfe")]
    solvfe.__package__ = "maple.function.solvfe"
    sys.modules["maple.function.solvfe"] = solvfe
    from maple.function.solvfe.fennix_hfe import (
        FeNNixAlchemicalParameters,
        FeNNixAlchemicalKernel,
        FeNNixAlchemicalSystem,
    )

    return (
        FeNNixAlchemicalKernel,
        FeNNixAlchemicalParameters,
        FeNNixAlchemicalSystem,
    )


def _worker(checkpoint: Path, output: Path, device: str) -> None:
    (
        FeNNixAlchemicalKernel,
        FeNNixAlchemicalParameters,
        FeNNixAlchemicalSystem,
    ) = _production_types()
    if _sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("FeNNix checkpoint SHA256 does not match the pinned model.")
    is_gpu = device.split(":", 1)[0] in {"gpu", "cuda"}
    kernel = FeNNixAlchemicalKernel(
        checkpoint,
        device=device,
        allow_unadmitted_gpu_mechanics=is_gpu,
        alchemical_parameters=FeNNixAlchemicalParameters(
            graph_softcore_v_angstrom=0.5,
            repulsion_softcore_angstrom=0.5,
            repulsion_power_m=2,
        ),
    )
    system = FeNNixAlchemicalSystem(
        atomic_numbers=ATOMIC_NUMBERS,
        coordinates_angstrom=COORDINATES_ANGSTROM,
        cell_angstrom=CELL_ANGSTROM,
        molecule_ids=MOLECULE_IDS,
        solute_atom_indices=SOLUTE_ATOM_INDICES,
    )
    rows = []
    for progress in PROGRESS:
        result = kernel.evaluate(system, progress)
        rows.append(
            {
                "progress": progress,
                "lambda_e": result.lambda_state.lambda_e,
                "lambda_v": result.lambda_state.lambda_v,
                "energy_ev": result.energy_ev,
                "forces_ev_per_angstrom": result.forces_ev_per_angstrom,
                "cell_gradient_ev_per_angstrom": (result.cell_gradient_ev_per_angstrom),
                "virial_ev": result.virial_ev,
                "denergy_dlambda_e_ev": result.denergy_dlambda_e_ev,
                "denergy_dlambda_v_ev": result.denergy_dlambda_v_ev,
                "denergy_dprogress_ev": result.denergy_dprogress_ev,
                "formal_admission": {
                    "mechanics_evidence_only": True,
                    "hfe_admitted": result.hfe_admitted,
                    "accuracy_admitted": result.accuracy_admitted,
                    "gpu_admitted": result.gpu_admitted,
                    "performance_admitted": result.performance_admitted,
                },
            }
        )
    identity = {
        name: getattr(kernel.identity, name)
        for name in kernel.identity.__dataclass_fields__
    }
    identity["runtime_source_sha256"] = dict(identity["runtime_source_sha256"])
    identity["runtime_package_versions"] = dict(identity["runtime_package_versions"])
    identity["alchemical_parameters"] = asdict(kernel.alchemical_parameters)
    root = Path(__file__).resolve().parents[2]
    kernel_sources = sorted(
        (root / "maple" / "function" / "solvfe" / "fennix_hfe").glob("*.py")
    )
    distance = math.dist(COORDINATES_ANGSTROM[0], COORDINATES_ANGSTROM[3])
    payload = {
        "scope": (
            "interacting_native_fennix_bio1m_mechanics_only_"
            "not_hfe_not_accuracy_not_gpu_admission_not_performance"
        ),
        "device_request": device,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_file_sha256": CHECKPOINT_SHA256,
        "identity": identity,
        "maple_kernel_source_sha256": {
            path.name: _sha256_file(path) for path in kernel_sources
        },
        "system": {
            "atomic_numbers": ATOMIC_NUMBERS,
            "coordinates_angstrom": COORDINATES_ANGSTROM,
            "cell_angstrom": CELL_ANGSTROM,
            "molecule_ids": MOLECULE_IDS,
            "solute_atom_indices": SOLUTE_ATOM_INDICES,
            "solute_solvent_oxygen_distance_angstrom": distance,
            "inside_cutoff": distance < CUTOFF_ANGSTROM,
        },
        "rows": rows,
        "formal_gates": FORMAL_GATES,
    }
    validate_receipt(payload, expected_platform="gpu" if is_gpu else "cpu")
    _write_json(output, payload)


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
    else:
        raise TypeError("Expected a numeric scalar or nested numeric list.")


def _max_nested_difference(left: Any, right: Any) -> float:
    left_values = tuple(_flatten_numbers(left))
    right_values = tuple(_flatten_numbers(right))
    if len(left_values) != len(right_values):
        raise ValueError("CPU/GPU observable shapes differ.")
    return max(
        (
            abs(left_value - right_value)
            for left_value, right_value in zip(left_values, right_values)
        ),
        default=0.0,
    )


def _expected_system() -> dict[str, Any]:
    return {
        "atomic_numbers": list(ATOMIC_NUMBERS),
        "coordinates_angstrom": [list(row) for row in COORDINATES_ANGSTROM],
        "cell_angstrom": [list(row) for row in CELL_ANGSTROM],
        "molecule_ids": list(MOLECULE_IDS),
        "solute_atom_indices": list(SOLUTE_ATOM_INDICES),
        "solute_solvent_oxygen_distance_angstrom": 4.0,
        "inside_cutoff": True,
    }


def _nontrivial_metrics(receipt: dict[str, Any]) -> dict[str, float | bool]:
    energies = [float(row["energy_ev"]) for row in receipt["rows"]]
    energy_span = max(energies) - min(energies)
    active_derivative = max(
        abs(float(row["denergy_dprogress_ev"])) for row in receipt["rows"]
    )
    passed = (
        energy_span > MINIMUM_ENERGY_SPAN_EV
        and active_derivative > MINIMUM_ACTIVE_DERIVATIVE_EV
    )
    return {
        "energy_span_ev": energy_span,
        "maximum_absolute_active_derivative_ev": active_derivative,
        "minimum_energy_span_ev_exclusive": MINIMUM_ENERGY_SPAN_EV,
        "minimum_active_derivative_ev_exclusive": MINIMUM_ACTIVE_DERIVATIVE_EV,
        "passed": passed,
    }


def _validate_identity(identity: dict[str, Any], platform: str) -> None:
    expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_source_revision": CHECKPOINT_SOURCE_REVISION,
        "source_revision": FENNOL_SOURCE_REVISION,
        "runtime_distribution": "FeNNol",
        "runtime_version": RUNTIME_VERSION,
        "original_parameter_tree_fingerprint": (ORIGINAL_PARAMETER_TREE_FINGERPRINT),
        "derived_parameter_tree_fingerprint": DERIVED_PARAMETER_TREE_FINGERPRINT,
        "fixed_species_encoding_float64_sha256": (
            FIXED_SPECIES_ENCODING_FLOAT64_SHA256
        ),
        "model_id": "fennix-bio1",
        "model_variant": "medium",
        "energy_unit": "eV",
        "length_unit": "angstrom",
        "jax_enable_x64": True,
        "matmul_precision": "highest",
        "tf32_enabled": False,
        "gpu_production_admitted": False,
        "scientific_scope": EXPECTED_KERNEL_SCOPE,
        "alchemical_parameters": EXPECTED_ALCHEMICAL_PARAMETERS,
    }
    for name, expected_value in expected.items():
        if identity.get(name) != expected_value:
            raise ValueError(f"FeNNix {name} identity mismatch.")
    if identity.get("runtime_source_sha256") != EXPECTED_RUNTIME_SOURCE_SHA256:
        raise ValueError("FeNNix runtime source identity mismatch.")
    if identity.get("alchemical_parameters") != EXPECTED_ALCHEMICAL_PARAMETERS:
        raise ValueError("FeNNix alchemical parameter provenance mismatch.")
    versions = identity.get("runtime_package_versions", {})
    required = {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "flax": "0.12.8",
        "numpy": "2.4.6",
    }
    if any(versions.get(name) != version for name, version in required.items()):
        raise ValueError("FeNNix runtime package identity mismatch.")
    gpu_versions = {
        "jax-cuda12-pjrt": "0.10.2",
        "jax-cuda12-plugin": "0.10.2",
    }
    expected_gpu_versions = (
        gpu_versions if platform == "gpu" else {name: "N/A" for name in gpu_versions}
    )
    if any(
        versions.get(name) != version for name, version in expected_gpu_versions.items()
    ):
        raise ValueError("FeNNix platform runtime package identity mismatch.")


def validate_receipt(
    receipt: dict[str, Any],
    *,
    expected_platform: str,
) -> dict[str, float | bool]:
    """Validate identity, full observables, system geometry, and coupling."""

    if receipt.get("checkpoint_file_sha256") != CHECKPOINT_SHA256:
        raise ValueError("FeNNix checkpoint file identity mismatch.")
    _validate_identity(receipt.get("identity", {}), expected_platform)
    distance = float(receipt["system"]["solute_solvent_oxygen_distance_angstrom"])
    if not distance < CUTOFF_ANGSTROM:
        raise ValueError("FeNNix smoke system is outside the 7.5 angstrom cutoff.")
    if _canonical_bytes(receipt.get("system")) != _canonical_bytes(_expected_system()):
        raise ValueError("FeNNix interacting-system identity mismatch.")
    if receipt.get("maple_kernel_source_sha256") != EXPECTED_MAPLE_KERNEL_SOURCE_SHA256:
        raise ValueError("FeNNix MAPLE kernel source identity mismatch.")
    if [float(row["progress"]) for row in receipt.get("rows", [])] != list(PROGRESS):
        raise ValueError("FeNNix smoke must use the frozen five-state schedule.")
    for row in receipt["rows"]:
        for name in (
            "energy_ev",
            "denergy_dlambda_e_ev",
            "denergy_dlambda_v_ev",
            "denergy_dprogress_ev",
        ):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"FeNNix {name} must be finite.")
        expected_shapes = {
            "forces_ev_per_angstrom": 18,
            "cell_gradient_ev_per_angstrom": 9,
            "virial_ev": 9,
        }
        for name, expected_size in expected_shapes.items():
            values = tuple(_flatten_numbers(row[name]))
            if len(values) != expected_size or not all(map(math.isfinite, values)):
                raise ValueError(f"FeNNix {name} is incomplete or non-finite.")
        if row.get("formal_admission") != {
            "mechanics_evidence_only": True,
            "hfe_admitted": False,
            "accuracy_admitted": False,
            "gpu_admitted": False,
            "performance_admitted": False,
        }:
            raise ValueError("FeNNix row admission boundary changed.")
    if receipt.get("formal_gates") != FORMAL_GATES:
        raise ValueError("FeNNix formal gates must remain closed.")
    metrics = _nontrivial_metrics(receipt)
    if not metrics["passed"]:
        raise ValueError(
            "FeNNix coupling is trivial: both energy span and active derivative "
            "must exceed 1e-3 eV."
        )
    return metrics


def compare_receipts(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    cpu_metrics = validate_receipt(cpu, expected_platform="cpu")
    gpu_metrics = validate_receipt(gpu, expected_platform="gpu")
    if cpu["system"] != gpu["system"]:
        raise ValueError("FeNNix CPU/GPU receipts do not use the same system.")
    identity_fields = (
        "checkpoint_sha256",
        "source_revision",
        "checkpoint_source_revision",
        "original_parameter_tree_fingerprint",
        "derived_parameter_tree_fingerprint",
        "fixed_species_encoding_float64_sha256",
    )
    for name in identity_fields:
        if cpu["identity"][name] != gpu["identity"][name]:
            raise ValueError(f"FeNNix CPU/GPU {name} mismatch.")
    if cpu["maple_kernel_source_sha256"] != gpu["maple_kernel_source_sha256"]:
        raise ValueError("FeNNix CPU/GPU MAPLE kernel source identity mismatch.")
    if (
        cpu["identity"]["alchemical_parameters"]
        != gpu["identity"]["alchemical_parameters"]
    ):
        raise ValueError("FeNNix CPU/GPU alchemical parameter identity mismatch.")
    observable_names = (
        "energy_ev",
        "forces_ev_per_angstrom",
        "cell_gradient_ev_per_angstrom",
        "virial_ev",
        "denergy_dlambda_e_ev",
        "denergy_dlambda_v_ev",
        "denergy_dprogress_ev",
    )
    per_progress = []
    for cpu_row, gpu_row in zip(cpu["rows"], gpu["rows"]):
        per_progress.append(
            {
                "progress": cpu_row["progress"],
                "max_abs_diff": {
                    name: _max_nested_difference(cpu_row[name], gpu_row[name])
                    for name in observable_names
                },
            }
        )
    global_max = max(
        difference
        for row in per_progress
        for difference in row["max_abs_diff"].values()
    )
    return {
        "scope": (
            "interacting_cpu_gpu_native_kernel_mechanics_parity_"
            "not_hfe_not_accuracy_not_gpu_admission_not_performance"
        ),
        "same_system": True,
        "system_sha256": _canonical_sha256(cpu["system"]),
        "same_checkpoint": True,
        "same_original_parameter_tree_fingerprint": True,
        "same_derived_parameter_tree_fingerprint": True,
        "same_fixed_species_encoding_float64_sha256": True,
        "same_maple_kernel_source_sha256": True,
        "same_alchemical_parameters": True,
        "inside_cutoff": True,
        "solute_solvent_oxygen_distance_angstrom": 4.0,
        "cpu_nontrivial_coupling": cpu_metrics,
        "gpu_nontrivial_coupling": gpu_metrics,
        "per_progress": per_progress,
        "global_max_abs_diff": global_max,
        "formal_gates": FORMAL_GATES,
    }


def build_artifact(
    *,
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    source_comparison: dict[str, Any],
    cpu_sha256: str,
    gpu_sha256: str,
    comparison_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    comparison = compare_receipts(cpu, gpu)
    if source_comparison.get("cpu_receipt_sha256") != cpu_sha256:
        raise ValueError("FeNNix source comparison CPU receipt identity mismatch.")
    if source_comparison.get("gpu_receipt_sha256") != gpu_sha256:
        raise ValueError("FeNNix source comparison GPU receipt identity mismatch.")
    if (
        source_comparison.get("global_max_abs_diff")
        != comparison["global_max_abs_diff"]
    ):
        raise ValueError("FeNNix source comparison observable mismatch.")
    if source_comparison.get("formal_gates") != FORMAL_GATES:
        raise ValueError("FeNNix source comparison formal gates changed.")
    return {
        "schema_version": 1,
        "audited_on": "2026-07-31",
        "verdict": "production_native_kernel_mechanics_available_admission_blocked",
        "acceptance_eligible": False,
        "scope": (
            "production_fennix_native_kernel_mechanics_only_"
            "not_hfe_not_accuracy_not_gpu_admission_not_performance"
        ),
        "reproduction": {
            "runner": (
                "docs/pretrained-solvation-hub/"
                "run_fennix_bio1_native_kernel_smoke.py"
            ),
            "worker_isolation": "one_process_per_cpu_or_gpu_backend",
            "commands": {
                "run": (
                    "python run_fennix_bio1_native_kernel_smoke.py run "
                    "--python PINNED_FENNOL_PYTHON --checkpoint FENNIX_BIO1M "
                    "--output-dir RECEIPTS"
                ),
                "combine": (
                    "python run_fennix_bio1_native_kernel_smoke.py combine "
                    "--cpu CPU_JSON --gpu GPU_JSON "
                    "--comparison SOURCE_COMPARISON_JSON --output ARTIFACT_JSON"
                ),
            },
        },
        "source_sha256": {
            "cpu_receipt": cpu_sha256,
            "gpu_receipt": gpu_sha256,
            "comparison_receipt": comparison_sha256,
            "runner": runner_sha256,
        },
        "identity": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_source_revision": CHECKPOINT_SOURCE_REVISION,
            "fennol_source_revision": FENNOL_SOURCE_REVISION,
            "original_parameter_tree_fingerprint": (
                ORIGINAL_PARAMETER_TREE_FINGERPRINT
            ),
            "derived_parameter_tree_fingerprint": (DERIVED_PARAMETER_TREE_FINGERPRINT),
            "fixed_species_encoding_float64_sha256": (
                FIXED_SPECIES_ENCODING_FLOAT64_SHA256
            ),
            "runtime_source_sha256": EXPECTED_RUNTIME_SOURCE_SHA256,
            "maple_kernel_source_sha256": cpu["maple_kernel_source_sha256"],
            "alchemical_parameters": EXPECTED_ALCHEMICAL_PARAMETERS,
            "system_sha256": comparison["system_sha256"],
        },
        "cpu_receipt": cpu,
        "gpu_receipt": gpu,
        "source_comparison_receipt": source_comparison,
        "recomputed_same_system_comparison": comparison,
        "formal_gates": FORMAL_GATES,
        "scientific_boundary": {
            "full_energy_force_cell_gradient_virial_lambda_arrays_saved": True,
            "nontrivial_interacting_coupling_required": True,
            "explicit_graph_softcore_provenance_bound": True,
            "explicit_repulsion_softcore_provenance_bound": True,
            "protocol_reconstruction_not_paper_reproduction": True,
            "mechanics_only": True,
            "sampling_run": False,
            "lambda_abf_run": False,
            "hfe_estimated": False,
            "experimental_records_used": 0,
            "distinct_primary_functional_groups_used": 0,
            "gpu_admitted": False,
            "matched_qm_timing_eligible": False,
            "performance_claim_allowed": False,
        },
    }


def _gpu_environment(base: dict[str, str], python: Path) -> dict[str, str]:
    environment = dict(base)
    environment.pop("JAX_PLATFORMS", None)
    environment.update(
        {
            "JAX_ENABLE_X64": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda",
        }
    )
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment_root = python.absolute().parent.parent
    library_dirs = sorted(
        str(path.resolve())
        for path in (environment_root / "cuda-libs").glob("*/lib")
        if path.is_dir()
    )
    library_dirs.extend(
        sorted(
            str(path.resolve())
            for cuda_root in Path("/usr/local").glob("cuda*")
            for path in (cuda_root / "extras" / "CUPTI").glob("lib64")
            if path.is_dir()
        )
    )
    if environment.get("LD_LIBRARY_PATH"):
        library_dirs.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    return environment


def _run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = [
        str(args.python),
        str(Path(__file__).resolve()),
        "_worker",
        "--checkpoint",
        str(args.checkpoint),
    ]
    paths = {}
    for platform, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
        output = args.output_dir / f"native-kernel-interacting-{platform}.json"
        environment = dict(os.environ)
        if platform == "gpu":
            environment = _gpu_environment(environment, args.python)
        else:
            environment.update(
                {
                    "JAX_PLATFORMS": "cpu",
                    "JAX_ENABLE_X64": "1",
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                }
            )
        subprocess.run(
            [*worker, "--device", device, "--output", str(output)],
            check=True,
            env=environment,
        )
        paths[platform] = output
    cpu = _load_json(paths["cpu"])
    gpu = _load_json(paths["gpu"])
    comparison = compare_receipts(cpu, gpu)
    comparison.update(
        {
            "cpu_receipt": str(paths["cpu"].resolve()),
            "gpu_receipt": str(paths["gpu"].resolve()),
            "cpu_receipt_sha256": _sha256_file(paths["cpu"]),
            "gpu_receipt_sha256": _sha256_file(paths["gpu"]),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        }
    )
    _write_json(
        args.output_dir / "native-kernel-interacting-cpu-gpu-comparison.json",
        comparison,
    )


def _combine(args: argparse.Namespace) -> None:
    payload = build_artifact(
        cpu=_load_json(args.cpu),
        gpu=_load_json(args.gpu),
        source_comparison=_load_json(args.comparison),
        cpu_sha256=_sha256_file(args.cpu),
        gpu_sha256=_sha256_file(args.gpu),
        comparison_sha256=_sha256_file(args.comparison),
        runner_sha256=_sha256_file(Path(__file__).resolve()),
    )
    _write_json(args.output, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("_worker")
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    combine = commands.add_parser("combine")
    combine.add_argument("--cpu", type=Path, required=True)
    combine.add_argument("--gpu", type=Path, required=True)
    combine.add_argument("--comparison", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "_worker":
        _worker(args.checkpoint, args.output, args.device)
    elif args.command == "run":
        _run(args)
    else:
        _combine(args)


if __name__ == "__main__":
    main()
