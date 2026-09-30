#!/usr/bin/env python3
"""Qualify the native-FP64 v2 response against direct AD and native pyddx.

The command is intentionally a serialized process entry.  ``main`` selects
Torch's process default float64 exactly once before importing or building a
MACE model.  Importing this module has no global dtype side effect.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

common = importlib.import_module("tools.route2_release.run_ddpcm_response_benchmark")

EVIDENCE = ROOT / "docs/route2/evidence"
PANEL = EVIDENCE / "route2-ddpcm-response-performance-geometries-v1.json"
BASELINE = EVIDENCE / "route2-pure-macepolar-torch-v3-source-manifest.json"
BASELINE_SHA256 = "527b7aebf3623c6cc0e7ada61fa1aca508d29b2c772c2dd908f3fdffadfa6d99"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
REPETITIONS = 3
DIRECTION_SEED = 20260922
AUDIT_STEPS = (2e-5, 1e-5, 5e-6)
_PROCESS_FP64_INITIALIZED = False

_SOURCES = (
    "maple/solvation/models/mace_polar_native_fp64.py",
    "maple/solvation/nonpolar/native_smd_cds_parameters.py",
    "maple/solvation/nonpolar/native_smd_cds.py",
    "maple/solvation/experimental/mace_polar_native_fp64_oracle.py",
    "maple/solvation/experimental/mace_polar_response_core.py",
    "maple/solvation/experimental/mace_polar_response_native_fp64_v2.py",
    "maple/solvation/continuum/torch_ddpcm.py",
    "maple/solvation/continuum/ddpcm_response.py",
    "tests/route2_vnext/test_mace_polar_native_fp64.py",
    "tests/route2_vnext/test_native_smd_cds.py",
    "tests/route2_vnext/test_mace_polar_response_native_fp64_v2.py",
    "tests/route2_vnext/test_mace_polar_response_native_fp64_v2_real.py",
    "tests/route2_vnext/test_native_fp64_benchmark.py",
    "tools/route2_release/run_ddpcm_response_benchmark.py",
    "tools/route2_release/run_ddpcm_response_native_fp64_v2_benchmark.py",
)


def initialize_process_fp64() -> None:
    """Set the serialized qualification process policy once and never restore it."""
    global _PROCESS_FP64_INITIALIZED
    if _PROCESS_FP64_INITIALIZED:
        raise RuntimeError("The CLI may initialize process FP64 exactly once.")
    import torch

    torch.set_default_dtype(torch.float64)
    _PROCESS_FP64_INITIALIZED = True


def protocol_policy() -> dict[str, object]:
    return {
        "schema": "ddpcm-native-fp64-v2-qualification-v1",
        "status": "frozen-before-qualification",
        "process_policy_id": "serialized-process-default-float64-v1",
        "model_policy_id": "mace-polar-native-fp64-v2",
        "required_identity": {
            "fp64_policy_id": "macepolar-native-fp64-serialized-process-policy-v2",
            "model_dtype": "torch.float64",
            "input_dtype": "torch.float64",
            "output_dtypes": ["torch.float64", "torch.float64"],
            "process_default_dtype": "torch.float64",
            "graph_longrange_version": "0.4.0",
            "graph_longrange_realspace_sha256": (
                "2cf7e098f4960490e6e9a5868ca1ea01ca23beb56ff8ce2cd0a5fe8615aec268"
            ),
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "implementation_ids": {
            "oracle": "native-fp64-direct-autograd-oracle-v2",
            "response": "native-fp64-structured-response-v2",
        },
        "warmups": 1,
        "repetitions": REPETITIONS,
        "direction_seed": DIRECTION_SEED,
        "audit_steps_angstrom": list(AUDIT_STEPS),
        "requested_devices": ["cpu", "cuda:0"],
        "performance_cases": ["water", "methane"],
        "native_validation_case": "acetone-10",
        "accuracy_limits": {
            "energy_eV": 1e-10,
            "force_eV_A": 1e-8,
            "hessian_eV_A2": 1e-6,
            "hvp_eV_A2": 1e-6,
        },
        "validation_limits": {
            "hvp_vs_full_eV_A2": 1e-6,
            "raw_antisymmetry_eV_A2": 1e-4,
            "translation_null_mode_eV_A2": 1e-6,
            "net_force_eV_A": 1e-8,
            "force_fd_eV_A2": 1e-4,
            "fine_step_agreement_eV_A2": 1e-4,
        },
        "performance": {
            "maximum_median_time_ratio": 0.9,
            "maximum_hvp_memory_ratio": 0.95,
            "maximum_hessian_memory_ratio": 0.8,
            "require_observed_ranges_nonoverlap": True,
            "outlier_removal": False,
        },
        "resource_policy": {
            "second_order_limit_bytes": 4_000_000_000,
            "maximum_oracle_atoms": 5,
            "one_time_16_gib_authorization_reused": False,
        },
        "physical_configuration": {
            "lmax": 15,
            "n_lebedev": 1202,
            "eta": 0.1,
            "dtype": "float64",
            "fit_performed": False,
            "ddpcm_equations": "ddx-0.8.0-two-dense-solves-general-source",
            "cds": "pyscf-2.13.1-native-fortran-literal-semantics",
        },
        "scientific_admitted": False,
        "stage1_pass": False,
        "release_admitted": False,
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json_new(path: Path, value: object) -> None:
    """Atomically create evidence without permitting replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"Preserve prior evidence; refusing overwrite: {path}"
        ) from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write(payload)


def source_manifest() -> dict[str, object]:
    baseline = json.loads(BASELINE.read_text())
    if len(baseline) != 352 or canonical_sha(baseline) != BASELINE_SHA256:
        raise RuntimeError("Reviewed original-352 inventory changed.")
    for name, digest in baseline.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Reviewed original-352 source changed: {name}")
    missing = [name for name in _SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Native-FP64 v2 source set is incomplete: {missing}")
    common_manifest = common.source_manifest()
    files = dict(common_manifest["candidate_files"])
    files.update({name: sha(ROOT / name) for name in _SOURCES})
    return {
        "schema": "native-fp64-v2-source-manifest-v1",
        "original_352_manifest_sha256": BASELINE_SHA256,
        "common_source_manifest_sha256": canonical_sha(common_manifest),
        "files_sha256": dict(sorted(files.items())),
        "panel_sha256": sha(PANEL),
    }


def runtime_environment() -> dict[str, object]:
    import torch

    names = ("torch", "numpy", "ase", "pyscf", "pyddx", "mace-torch", "graph-longrange")
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    numpy_configuration = io.StringIO()
    with contextlib.redirect_stdout(numpy_configuration):
        np.show_config()
    return {
        "process_default_dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__,
        "versions": versions,
        "platform": platform.platform(),
        "torch_cuda_version": torch.version.cuda,
        "torch_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "numpy_blas": {"show_config": numpy_configuration.getvalue()},
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def hardware_context() -> dict[str, object]:
    import torch

    try:
        inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        inventory_error = None
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        inventory = None
        inventory_error = getattr(error, "output", "") or str(error)
    return {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi_gpu_inventory": inventory,
        "nvidia_smi_error": inventory_error,
    }


def prepare(args) -> None:
    if sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Official MACE-POLAR checkpoint changed.")
    if str(runtime_environment()["process_default_dtype"]) != "torch.float64":
        raise RuntimeError("Preparation requires the serialized FP64 process entry.")
    payload = {
        **protocol_policy(),
        "source_manifest": source_manifest(),
        "runtime_environment": runtime_environment(),
        "hardware_context": hardware_context(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=ROOT)
        ).hexdigest(),
    }
    write_json_new(args.output_dir / "protocol.json", payload)
    print(f"prepared {args.output_dir / 'protocol.json'}", flush=True)


def verify_protocol(args) -> dict[str, object]:
    path = args.output_dir / "protocol.json"
    protocol = json.loads(path.read_text())
    expected = protocol_policy()
    if {key: protocol.get(key) for key in expected} != expected:
        raise RuntimeError("Frozen native-FP64 v2 protocol policy drift.")
    if protocol.get("source_manifest") != source_manifest():
        raise RuntimeError("Frozen native-FP64 v2 source manifest drift.")
    if protocol.get("runtime_environment") != runtime_environment():
        raise RuntimeError("Frozen native-FP64 v2 runtime drift.")
    if protocol.get("hardware_context") != hardware_context():
        raise RuntimeError("Frozen native-FP64 v2 hardware drift.")
    if sha(args.checkpoint) != protocol.get("checkpoint_sha256"):
        raise RuntimeError("Frozen checkpoint drift.")
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip() != protocol.get("git_head"):
        raise RuntimeError("Frozen git revision drift.")
    if hashlib.sha256(
        subprocess.check_output(["git", "diff"], cwd=ROOT)
    ).hexdigest() != protocol.get("tracked_diff_sha256"):
        raise RuntimeError("Frozen tracked diff drift.")
    return protocol


def _case(name: str) -> dict[str, object]:
    cases = json.loads(PANEL.read_text())["cases"]
    try:
        return next(case for case in cases if case["name"] == name)
    except StopIteration as error:
        raise ValueError(f"Unknown frozen geometry case: {name}") from error


def _atoms(case):
    from ase import Atoms

    atoms = Atoms(case["symbols"], positions=case["positions_angstrom"])
    atoms.info.update(charge=case["charge"], mult=case["multiplicity"])
    payload = {
        "symbols": case["symbols"],
        "positions_angstrom": case["positions_angstrom"],
        "charge": case["charge"],
        "multiplicity": case["multiplicity"],
    }
    if canonical_sha(payload) != case["geometry_sha256"]:
        raise RuntimeError("Frozen geometry binding failed.")
    return atoms


def _direction(atom_count: int) -> np.ndarray:
    vector = np.random.default_rng(DIRECTION_SEED).normal(size=(atom_count, 3))
    return vector / np.linalg.norm(vector)


def _build(backend: str, case, args):
    if backend == "oracle":
        from maple.solvation.experimental.mace_polar_native_fp64_oracle import (
            build_smd_mace_polar_native_fp64_v2_oracle as builder,
        )
    else:
        from maple.solvation.experimental.mace_polar_response_native_fp64_v2 import (
            build_smd_mace_polar_response_native_fp64_v2_pes as builder,
        )
    return builder(
        case["symbols"],
        solvent=case["solvent"],
        device=args.device,
        checkpoint_path=args.checkpoint,
    )


def _component_identities(pes) -> dict[str, object]:
    model_metadata = pes.model.metadata()
    components = {
        "model": pes.model.configuration_sha256(),
        "ddpcm": pes.continuum.configuration_sha256(),
        "cds": pes.solvent_term.configuration_sha256(),
    }
    return {
        "provider_id": pes.provider_id,
        "profile_id": pes.profile_id,
        "scalar_contract_id": pes.scalar_contract_id,
        "pes_configuration_sha256": pes.configuration_sha256(),
        "component_configuration_sha256": components,
        "implementation_id": (
            "native-fp64-direct-autograd-oracle-v2"
            if "oracle" in pes.provider_id
            else "native-fp64-structured-response-v2"
        ),
        "physical_contract_sha256": canonical_sha(
            {
                "model": components["model"],
                "cds": components["cds"],
                "solvent": pes.solvent,
                "symbols": list(pes.symbols),
                "ddpcm_equations": "ddx-0.8.0-two-dense-solves-general-source",
                "lmax": 15,
                "n_lebedev": 1202,
                "eta": 0.1,
            }
        ),
        "fp64_runtime": {
            key: model_metadata.get(key)
            for key in (
                "fp64_policy_id",
                "model_dtype",
                "input_dtype",
                "output_dtypes",
                "process_default_dtype",
                "torch_version",
                "mace_torch_version",
                "graph_longrange_version",
                "graph_longrange_realspace_sha256",
                "checkpoint_sha256",
                "base_configuration_sha256",
            )
        },
    }


def _tensor_sha256(value) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_identity(pes, atoms) -> dict[str, str]:
    import torch

    from maple.solvation.coupling.exact_gto import (
        mace_polar_learned_source_embedding_matrix,
    )

    positions = torch.tensor(
        atoms.positions, dtype=torch.float64, device=pes.device, requires_grad=False
    )
    vacuum, learned = pes.model.energy_source_torch(atoms, positions)
    embedding = torch.tensor(
        mace_polar_learned_source_embedding_matrix(),
        dtype=torch.float64,
        device=pes.device,
    )
    embedded = learned @ embedding.T
    return {
        "active_sha256": _tensor_sha256(learned),
        "embedded_sha256": _tensor_sha256(embedded),
        "vacuum_sha256": _tensor_sha256(vacuum.reshape(1)),
    }


def _resource_preflight(pes, backend: str) -> dict[str, object]:
    return pes._resource_preflight() if backend == "oracle" else pes._preflight(2)


def _serialize_hessian(result) -> dict[str, object]:
    return {
        "energy_eV": result.energy_eV,
        "forces_eV_A": result.forces_eV_per_A.tolist(),
        "hessian_eV_A2": result.hessian_eV_per_A2.tolist(),
        "raw_antisymmetry_eV_A2": result.maximum_antisymmetry_eV_per_A2,
        "evaluation_sha256": result.evaluation_sha256,
        "configuration_sha256": result.configuration_sha256,
        "provider_id": result.provider_id,
        "profile_id": result.profile_id,
        "scalar_contract_id": result.scalar_contract_id,
        "derivative_method": result.derivative_method,
        "components_eV": dict(result.component_energies_eV),
        "diagnostics": common.plain(result.diagnostics),
    }


def _topology(pes, atoms) -> dict[str, str]:
    import torch

    positions = torch.tensor(atoms.positions, dtype=torch.float64, device=pes.device)
    if hasattr(pes.continuum, "_operator"):
        polar = pes.continuum._operator._geometry(positions).topology.topology_sha256
    else:
        from tools.route2_release.run_pure_mace_polar_torch_canary import _topology

        polar = _topology(pes, atoms)["ddpcm"]
    cds = pes.solvent_term.evaluate_torch(positions).diagnostics.dareal.topology_sha256
    model = pes.model.topology_diagnostics(atoms)["pair_mask_order_sha256"]
    return {"ddpcm": polar, "cds": cds, "model": model}


def _native_reference(atoms, args, solvent: str):
    from maple.solvation.models.mace_polar import (
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.experimental.mace_polar_frozen_ddx import (
        build_smd_mace_polar_frozen_point_ddx_pes,
    )

    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device, checkpoint_path=args.checkpoint
    )
    return build_smd_mace_polar_frozen_point_ddx_pes(
        model, tuple(atoms.get_chemical_symbols()), solvent=solvent
    )


def _finite_difference_audit(
    *, pes, reference, atoms, direction, hvp, step: float, central
) -> dict[str, object]:
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    plus_topology = _topology(pes, plus)
    minus_topology = _topology(pes, minus)
    same_topology = plus_topology == central == minus_topology
    audit = {
        "step_angstrom": step,
        "plus_topology": plus_topology,
        "minus_topology": minus_topology,
        "same_topology": same_topology,
    }
    if not same_topology:
        return audit
    v2_finite_difference = -(pes.get_forces(plus) - pes.get_forces(minus)).reshape(
        -1
    ) / (2 * step)
    native_finite_difference = -(
        reference.get_forces(plus) - reference.get_forces(minus)
    ).reshape(-1) / (2 * step)
    audit.update(
        v2_same_graph_hvp_eV_A2=v2_finite_difference.tolist(),
        v2_same_graph_maximum_error_eV_A2=float(
            np.max(np.abs(v2_finite_difference - hvp))
        ),
        native_pyddx_hvp_eV_A2=native_finite_difference.tolist(),
        native_pyddx_maximum_error_eV_A2=float(
            np.max(np.abs(native_finite_difference - hvp))
        ),
    )
    return audit


def _validation(pes, atoms, args, protocol) -> dict[str, object]:
    result = pes.evaluate_hessian(atoms)
    payload = _serialize_hessian(result)
    payload["source_witness_pass"] = common._source_witnesses_pass(
        payload["diagnostics"], 1.0e-12
    )
    direction = _direction(len(atoms))
    hvp = pes.hessian_vector_product(atoms, direction).reshape(-1)
    payload["hvp_eV_A2"] = hvp.tolist()
    payload["hvp_vs_full_error_eV_A2"] = float(
        np.max(np.abs(hvp - result.hessian_eV_per_A2 @ direction.reshape(-1)))
    )
    payload["translation_null_mode_max_abs_eV_A2"] = []
    for axis in range(3):
        translation = np.zeros((len(atoms), 3))
        translation[:, axis] = 1.0
        payload["translation_null_mode_max_abs_eV_A2"].append(
            float(np.max(np.abs(result.hessian_eV_per_A2 @ translation.reshape(-1))))
        )
    payload["net_force_max_abs_eV_A"] = float(
        np.max(np.abs(result.forces_eV_per_A.sum(axis=0)))
    )
    reference = _native_reference(atoms, args, pes.solvent)
    reference_forces = reference.evaluate_forces(atoms)
    payload["native_pyddx_reference"] = {
        "energy_eV": reference_forces.central_state.total_energy_eV,
        "forces_eV_A": reference_forces.total_forces_eV_per_A.tolist(),
        "energy_error_eV": abs(
            result.energy_eV - reference_forces.central_state.total_energy_eV
        ),
        "force_error_eV_A": float(
            np.max(
                np.abs(result.forces_eV_per_A - reference_forces.total_forces_eV_per_A)
            )
        ),
        "implementation": "unmodified-native-pyddx-reference",
    }
    central = _topology(pes, atoms)
    audits = [
        _finite_difference_audit(
            pes=pes,
            reference=reference,
            atoms=atoms,
            direction=direction,
            hvp=hvp,
            step=step,
            central=central,
        )
        for step in AUDIT_STEPS
    ]
    payload["force_difference_audits"] = audits
    payload["negative_topology_rejection_pass"] = audits[0]["same_topology"] is False
    fine = audits[1:]
    fine_accepted = all(
        audit.get("same_topology")
        and audit.get("v2_same_graph_maximum_error_eV_A2", float("inf"))
        <= protocol["validation_limits"]["force_fd_eV_A2"]
        for audit in fine
    )
    agreement = (
        float(
            np.max(
                np.abs(
                    np.asarray(fine[1]["v2_same_graph_hvp_eV_A2"])
                    - np.asarray(fine[0]["v2_same_graph_hvp_eV_A2"])
                )
            )
        )
        if fine_accepted
        else None
    )
    payload["fine_step_agreement_eV_A2"] = agreement
    limits = protocol["validation_limits"]
    accuracy = protocol["accuracy_limits"]
    payload["validation_pass"] = bool(
        payload["hvp_vs_full_error_eV_A2"] <= limits["hvp_vs_full_eV_A2"]
        and payload["raw_antisymmetry_eV_A2"] <= limits["raw_antisymmetry_eV_A2"]
        and max(payload["translation_null_mode_max_abs_eV_A2"])
        <= limits["translation_null_mode_eV_A2"]
        and payload["net_force_max_abs_eV_A"] <= limits["net_force_eV_A"]
        and payload["native_pyddx_reference"]["energy_error_eV"]
        <= accuracy["energy_eV"]
        and payload["native_pyddx_reference"]["force_error_eV_A"]
        <= accuracy["force_eV_A"]
        and payload["source_witness_pass"]
        and payload["negative_topology_rejection_pass"]
        and fine_accepted
        and agreement is not None
        and agreement <= limits["fine_step_agreement_eV_A2"]
    )
    return payload


def _row_path(args) -> Path:
    return (
        args.output_dir
        / "rows"
        / args.backend
        / args.device.replace(":", "-")
        / args.case
        / f"{args.mode}.json"
    )


def _run_locked(args) -> None:
    import torch

    row_path = _row_path(args)
    if row_path.exists():
        raise FileExistsError(
            f"Preserve prior evidence; refusing overwrite: {row_path}"
        )
    row = {
        "schema": "ddpcm-native-fp64-v2-row-v1",
        "protocol_sha256": sha(args.output_dir / "protocol.json"),
        "backend": args.backend,
        "device": args.device,
        "case": args.case,
        "mode": args.mode,
        "completed": False,
        "benchmark_lock": getattr(args, "_lock_evidence", None),
        "runtime_environment": runtime_environment(),
        "source_manifest_sha256": canonical_sha(source_manifest()),
        "checkpoint_sha256": sha(args.checkpoint),
        "source_checks": [],
    }
    begun = time.perf_counter()
    try:
        protocol = verify_protocol(args)
        case = _case(args.case)
        if (
            args.mode in {"hvp", "hessian"}
            and args.case not in protocol["performance_cases"]
        ):
            raise ValueError("Performance is restricted to water and methane.")
        if args.mode == "validation" and not (
            args.backend == "response"
            and args.case == protocol["native_validation_case"]
        ):
            raise ValueError("Native validation is response-only acetone-10.")
        atoms = _atoms(case)
        pes = _build(args.backend, case, args)
        row["atom_count"] = len(atoms)
        row["geometry_sha256"] = case["geometry_sha256"]
        row["identities"] = _component_identities(pes)
        row["source_identity"] = _source_identity(pes, atoms)
        row["resource_preflight"] = _resource_preflight(pes, args.backend)
        row["source_checks"].append(
            {
                "stage": "after-build",
                "source_manifest_sha256": canonical_sha(
                    verify_protocol(args)["source_manifest"]
                ),
            }
        )
        if args.mode == "validation":
            row["result"] = _validation(pes, atoms, args, protocol)
            row["source_checks"].append(
                {
                    "stage": "after-validation",
                    "source_manifest_sha256": canonical_sha(
                        verify_protocol(args)["source_manifest"]
                    ),
                }
            )
        else:
            direction = _direction(len(atoms))

            def evaluate():
                if args.mode == "hessian":
                    return pes.evaluate_hessian(atoms)
                return pes.hessian_vector_product(atoms, direction)

            warm = evaluate()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            row["warmup_completed"] = True
            row["source_checks"].append(
                {
                    "stage": "after-warmup",
                    "source_manifest_sha256": canonical_sha(
                        verify_protocol(args)["source_manifest"]
                    ),
                }
            )
            del warm
            row["measurements"] = []
            for index in range(REPETITIONS):
                gc.collect()
                if args.device.startswith("cuda"):
                    guard_before = common.assert_gpu_idle_for_timing()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                with common.PeakMonitor() as monitor:
                    start = time.perf_counter()
                    output = evaluate()
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    elapsed = time.perf_counter() - start
                measurement = {
                    "index": index,
                    "elapsed_seconds": elapsed,
                    **monitor.result(),
                    "result": (
                        _serialize_hessian(output)
                        if args.mode == "hessian"
                        else {"hvp_eV_A2": output.tolist()}
                    ),
                }
                if args.device.startswith("cuda"):
                    measurement["cuda_peak_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated(args.device)
                    )
                    measurement["cuda_process_guard_before"] = guard_before
                    measurement["cuda_process_guard_after"] = (
                        common.assert_gpu_idle_for_timing()
                    )
                row["measurements"].append(measurement)
                row["source_checks"].append(
                    {
                        "stage": f"after-repetition-{index}",
                        "source_manifest_sha256": canonical_sha(
                            verify_protocol(args)["source_manifest"]
                        ),
                    }
                )
                del output
        row["completed"] = True
    except Exception as error:
        row["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    row["total_seconds"] = time.perf_counter() - begun
    write_json_new(row_path, row)
    print(f"saved {row_path} completed={row['completed']}", flush=True)
    if not row["completed"]:
        raise SystemExit(1)
    if row.get("result", {}).get("validation_pass") is False:
        raise SystemExit(2)


def run(args) -> None:
    with common.benchmark_lock() as lock_evidence:
        args._lock_evidence = lock_evidence
        _run_locked(args)


def required_row_identities(protocol) -> set[tuple[str, str, str, str]]:
    identities = set()
    for device in protocol["requested_devices"]:
        for case in protocol["performance_cases"]:
            for mode in ("hvp", "hessian"):
                for backend in ("oracle", "response"):
                    identities.add((backend, device, case, mode))
        identities.add(
            ("response", device, protocol["native_validation_case"], "validation")
        )
    return identities


def _read_rows(output_dir: Path):
    records = {}
    for path in sorted((output_dir / "rows").glob("**/*.json")):
        record = json.loads(path.read_text())
        identity = tuple(
            record.get(key) for key in ("backend", "device", "case", "mode")
        )
        if identity in records:
            raise RuntimeError(f"Duplicate result row identity: {identity}")
        records[identity] = record
    return records


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_row_binding(record, protocol, protocol_sha: str) -> None:
    if not record.get("completed") or record.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Incomplete row or protocol binding failure.")
    if record.get("runtime_environment") != protocol["runtime_environment"]:
        raise RuntimeError("Row runtime-environment binding failure.")
    manifest_sha = canonical_sha(protocol["source_manifest"])
    if record.get("source_manifest_sha256") != manifest_sha:
        raise RuntimeError("Row source-manifest binding failure.")
    if record.get("checkpoint_sha256") != protocol["checkpoint_sha256"]:
        raise RuntimeError("Row checkpoint binding failure.")
    case = _case(record["case"])
    if record.get("geometry_sha256") != case["geometry_sha256"]:
        raise RuntimeError("Row geometry binding failure.")
    expected_stages = (
        {"after-build", "after-validation"}
        if record["mode"] == "validation"
        else {
            "after-build",
            "after-warmup",
            "after-repetition-0",
            "after-repetition-1",
            "after-repetition-2",
        }
    )
    checks = record.get("source_checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(expected_stages)
        or {check.get("stage") for check in checks} != expected_stages
    ):
        raise RuntimeError("Row source-check stages are incomplete.")
    if any(check.get("source_manifest_sha256") != manifest_sha for check in checks):
        raise RuntimeError("Row source-check binding failure.")
    source_identity = record.get("source_identity")
    if not isinstance(source_identity, dict) or not all(
        _is_sha256(source_identity.get(key))
        for key in ("active_sha256", "embedded_sha256", "vacuum_sha256")
    ):
        raise RuntimeError("Row source identity is missing or invalid.")
    identities = record.get("identities")
    if not isinstance(identities, dict):
        raise RuntimeError("Row component identities are missing.")
    runtime = identities.get("fp64_runtime")
    required = protocol["required_identity"]
    if not isinstance(runtime, dict) or any(
        runtime.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("Row native-FP64 runtime identity mismatch.")
    if (
        runtime.get("torch_version") != protocol["runtime_environment"]["torch_version"]
        or runtime.get("mace_torch_version")
        != protocol["runtime_environment"]["versions"]["mace-torch"]
    ):
        raise RuntimeError("Row model-library runtime version mismatch.")
    components = identities.get("component_configuration_sha256")
    if not isinstance(components, dict) or not all(
        _is_sha256(components.get(key)) for key in ("model", "ddpcm", "cds")
    ):
        raise RuntimeError("Row component configuration identities are invalid.")


def _series(record, memory_key: str):
    measurements = record.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != REPETITIONS:
        raise RuntimeError("Missing fixed timing repetitions.")
    if [item.get("index") for item in measurements] != list(range(REPETITIONS)):
        raise RuntimeError("Invalid fixed repetition indexes.")
    for item in measurements:
        if not all(
            isinstance(item.get(key), (int, float))
            and not isinstance(item[key], bool)
            and np.isfinite(item[key])
            and item[key] > 0
            for key in ("elapsed_seconds", memory_key)
        ):
            raise RuntimeError("Timing/memory values must be finite and positive.")
    return measurements


def performance_statistics(
    *, oracle_times, response_times, oracle_memory, response_memory, mode, policy
) -> dict[str, object]:
    time_oracle_median, time_oracle_mad = common._median_and_mad(oracle_times)
    time_response_median, time_response_mad = common._median_and_mad(response_times)
    memory_oracle_median, memory_oracle_mad = common._median_and_mad(oracle_memory)
    memory_response_median, memory_response_mad = common._median_and_mad(
        response_memory
    )
    time_ratio = time_response_median / time_oracle_median
    memory_ratio = memory_response_median / memory_oracle_median
    time_nonoverlap = max(response_times) < min(oracle_times)
    memory_nonoverlap = max(response_memory) < min(oracle_memory)
    memory_limit = policy[
        "maximum_hvp_memory_ratio" if mode == "hvp" else "maximum_hessian_memory_ratio"
    ]
    passed = bool(
        time_ratio <= policy["maximum_median_time_ratio"]
        and time_nonoverlap
        and memory_ratio <= memory_limit
        and memory_nonoverlap
    )
    return {
        "raw_times_seconds": {
            "oracle": list(oracle_times),
            "response": list(response_times),
        },
        "raw_memory_bytes": {
            "oracle": list(oracle_memory),
            "response": list(response_memory),
        },
        "observed_time_ranges_seconds": {
            "oracle": [min(oracle_times), max(oracle_times)],
            "response": [min(response_times), max(response_times)],
        },
        "observed_memory_ranges_bytes": {
            "oracle": [min(oracle_memory), max(oracle_memory)],
            "response": [min(response_memory), max(response_memory)],
        },
        "median_time_seconds": {
            "oracle": time_oracle_median,
            "response": time_response_median,
        },
        "time_mad_seconds": {
            "oracle": time_oracle_mad,
            "response": time_response_mad,
        },
        "median_memory_bytes": {
            "oracle": memory_oracle_median,
            "response": memory_response_median,
        },
        "memory_mad_bytes": {
            "oracle": memory_oracle_mad,
            "response": memory_response_mad,
        },
        "median_time_ratio_response_over_oracle": time_ratio,
        "median_memory_ratio_response_over_oracle": memory_ratio,
        "time_ranges_nonoverlap": time_nonoverlap,
        "memory_ranges_nonoverlap": memory_nonoverlap,
        "performance_conditions_pass": passed,
    }


def summarize(args) -> None:
    protocol = verify_protocol(args)
    expected = required_row_identities(protocol)
    records = _read_rows(args.output_dir)
    missing = expected - set(records)
    unexpected = set(records) - expected
    if missing:
        raise RuntimeError(f"Missing preregistered rows: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected result row identities: {sorted(unexpected)}")
    protocol_sha = sha(args.output_dir / "protocol.json")
    for record in records.values():
        _validate_row_binding(record, protocol, protocol_sha)
    comparisons = []
    numerical_checks_pass = True
    performance_conditions_pass = True
    for device in protocol["requested_devices"]:
        memory_key = (
            "cuda_peak_allocated_bytes"
            if device.startswith("cuda")
            else "uss_sampled_peak_bytes"
        )
        for case in protocol["performance_cases"]:
            for backend in ("oracle", "response"):
                first = records[(backend, device, case, "hvp")]["identities"]
                second = records[(backend, device, case, "hessian")]["identities"]
                for key in (
                    "provider_id",
                    "profile_id",
                    "scalar_contract_id",
                    "pes_configuration_sha256",
                    "component_configuration_sha256",
                    "physical_contract_sha256",
                    "fp64_runtime",
                ):
                    if first.get(key) != second.get(key):
                        raise RuntimeError(
                            f"Frozen {backend} construction identity drifted across modes."
                        )
            for mode in ("hvp", "hessian"):
                oracle = records[("oracle", device, case, mode)]
                response = records[("response", device, case, mode)]
                if (
                    oracle["identities"]["physical_contract_sha256"]
                    != response["identities"]["physical_contract_sha256"]
                ):
                    raise RuntimeError("Oracle/response physical contract mismatch.")
                if oracle["source_identity"] != response["source_identity"]:
                    raise RuntimeError("Oracle/response source identity mismatch.")
                oracle_series = _series(oracle, memory_key)
                response_series = _series(response, memory_key)
                keys = (
                    ("hvp_eV_A2",)
                    if mode == "hvp"
                    else ("energy_eV", "forces_eV_A", "hessian_eV_A2")
                )
                errors = {}
                for key in keys:
                    errors[key] = max(
                        common._maximum_error(
                            oracle_item["result"][key],
                            response_item["result"][key],
                            key,
                        )
                        for oracle_item, response_item in zip(
                            oracle_series, response_series
                        )
                    )
                time_oracle = [item["elapsed_seconds"] for item in oracle_series]
                time_response = [item["elapsed_seconds"] for item in response_series]
                memory_oracle = [item[memory_key] for item in oracle_series]
                memory_response = [item[memory_key] for item in response_series]
                performance = performance_statistics(
                    oracle_times=time_oracle,
                    response_times=time_response,
                    oracle_memory=memory_oracle,
                    response_memory=memory_response,
                    mode=mode,
                    policy=protocol["performance"],
                )
                accuracy = protocol["accuracy_limits"]
                accuracy_pass = all(
                    errors[key]
                    <= accuracy[
                        {
                            "energy_eV": "energy_eV",
                            "forces_eV_A": "force_eV_A",
                            "hessian_eV_A2": "hessian_eV_A2",
                            "hvp_eV_A2": "hvp_eV_A2",
                        }[key]
                    ]
                    for key in keys
                )
                if mode == "hessian":
                    source_witness_match = all(
                        _is_sha256(oracle_item["result"]["diagnostics"].get(key))
                        and oracle_item["result"]["diagnostics"].get(key)
                        == response_item["result"]["diagnostics"].get(key)
                        for oracle_item, response_item in zip(
                            oracle_series, response_series
                        )
                        for key in (
                            "source_active_sha256",
                            "source_embedded_sha256",
                        )
                    )
                    accuracy_pass &= source_witness_match
                else:
                    source_witness_match = True
                resource_envelope_pass = all(
                    item[memory_key]
                    <= record["resource_preflight"]["estimated_total_bytes"]
                    for record, series in (
                        (oracle, oracle_series),
                        (response, response_series),
                    )
                    for item in series
                )
                accuracy_pass &= resource_envelope_pass
                performance_pass = performance["performance_conditions_pass"]
                numerical_checks_pass &= accuracy_pass
                performance_conditions_pass &= performance_pass
                comparisons.append(
                    {
                        "device": device,
                        "case": case,
                        "mode": mode,
                        "maximum_errors": errors,
                        "accuracy_pass": accuracy_pass,
                        "source_witness_match": source_witness_match,
                        "resource_envelope_pass": resource_envelope_pass,
                        **performance,
                        "physical_contract_sha256": oracle["identities"][
                            "physical_contract_sha256"
                        ],
                    }
                )
    validations = {
        device: records[
            ("response", device, protocol["native_validation_case"], "validation")
        ]["result"]
        for device in protocol["requested_devices"]
    }
    numerical_checks_pass &= all(
        result.get("validation_pass") is True for result in validations.values()
    )
    summary = {
        "schema": "ddpcm-native-fp64-v2-summary-v1",
        "protocol_sha256": protocol_sha,
        "comparisons": comparisons,
        "native_validations": validations,
        "numerical_checks_pass": numerical_checks_pass,
        "performance_conditions_pass": performance_conditions_pass,
        "implementation_pass": False,
        "final_review_passed": False,
        "stage1_pass": False,
        "scientific_admitted": False,
        "release_admitted": False,
    }
    write_json_new(args.output_dir / "summary.json", summary)
    if not numerical_checks_pass or not performance_conditions_pass:
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-POLAR-1-M.model",
    )
    parser.add_argument("--backend", choices=("oracle", "response"), default="response")
    parser.add_argument(
        "--case", choices=("water", "methane", "acetone-10"), default="water"
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--mode", choices=("hvp", "hessian", "validation"), default="hessian"
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    initialize_process_fp64()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
