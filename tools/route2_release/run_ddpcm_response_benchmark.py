#!/usr/bin/env python3
"""Frozen same-scalar response qualification, with private resumable outputs.

Prepare once after source stabilization; every run checks that same manifest.
Timing uses isolated backend processes, one warmup and three observations.
Validation is outside timed windows. Finite differences are audits only.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import fcntl
import gc
import hashlib
from importlib.metadata import version
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/route2/evidence"
PANEL = EVIDENCE / "route2-ddpcm-response-performance-geometries-v1.json"
BASELINE = EVIDENCE / "route2-pure-macepolar-torch-v3-source-manifest.json"
BASELINE_SHA = "527b7aebf3623c6cc0e7ada61fa1aca508d29b2c772c2dd908f3fdffadfa6d99"
PANEL_SHA = "a1b4adee9471884ebac8c55e1b93fc837e95eccd65878df324e1b30ec1b904d2"
CHECKPOINT_SHA = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
NEW_SOURCES = (
    "maple/solvation/continuum/solid_harmonic_response.py",
    "maple/solvation/continuum/response_topology.py",
    "maple/solvation/continuum/response_tensor_binding.py",
    "maple/solvation/continuum/ddpcm_response_operators.py",
    "maple/solvation/continuum/ddpcm_response.py",
    "maple/solvation/experimental/mace_polar_response.py",
    "maple/solvation/experimental/mace_polar_response_core.py",
    "maple/solvation/derivatives/response.py",
    "tools/route2_release/run_ddpcm_response_benchmark.py",
)
REPETITIONS = 3
DIRECTION_SEED = 20260922
AUDIT_STEPS = (2e-5, 1e-5, 5e-6)
SOURCE_WITNESS_FLOOR = 1e-12
LOCK_PATH = Path.home() / ".cache/maple/ddpcm-response-benchmark.lock"


def protocol_policy():
    return {
        "schema": "ddpcm-exact-response-performance-v1",
        "status": "frozen-before-qualification",
        "warmups": 1,
        "repetitions": REPETITIONS,
        "direction_seed": DIRECTION_SEED,
        "audit_steps_angstrom": list(AUDIT_STEPS),
        "source_witness_floor": SOURCE_WITNESS_FLOOR,
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
            "legacy_force_fd_eV_A2": 1e-4,
            "fine_step_agreement_eV_A2": 1e-4,
        },
        "performance": {
            "maximum_median_time_ratio": 0.9,
            "maximum_hvp_memory_ratio": 0.95,
            "maximum_hessian_memory_ratio": 0.8,
            "require_observed_ranges_nonoverlap": True,
            "outlier_removal": False,
        },
        "memory_metric": {
            "cpu": "uss_sampled_peak_bytes",
            "cuda": "cuda_peak_allocated_bytes",
        },
        "requested_devices": ["cpu", "cuda:0"],
        "continuum_memory_required_cases": ["water", "methane", "acetone-10"],
        "resource_policy": {
            "derivative_order": 2,
            "continuum_limit_bytes": 4_000_000_000,
            "full_pes_observed_metric_must_not_exceed_estimated_total": True,
            "continuum_incremental_metric_must_not_exceed_estimate": True,
        },
        "physical_configuration": {
            "lmax": 15,
            "n_lebedev": 1202,
            "eta": 0.1,
            "dtype": "float64",
            "fit_performed": False,
        },
        "scientific_admitted": False,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def source_manifest():
    baseline = json.loads(BASELINE.read_text())
    if len(baseline) != 352 or canonical_sha(baseline) != BASELINE_SHA:
        raise RuntimeError("Reviewed baseline inventory changed.")
    for name, digest in baseline.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Reviewed baseline source changed: {name}")
    if sha(PANEL) != PANEL_SHA:
        raise RuntimeError("Prospective geometry panel changed.")
    return {
        "baseline_source_sha256": BASELINE_SHA,
        "candidate_files": {name: sha(ROOT / name) for name in NEW_SOURCES},
    }


def environment():
    import torch

    numpy_configuration = io.StringIO()
    with contextlib.redirect_stdout(numpy_configuration):
        np.show_config()
    return {
        "torch_default_dtype": str(torch.get_default_dtype()),
        "versions": {
            name: version(name)
            for name in ("torch", "numpy", "ase", "pyscf", "pyddx", "mace-torch")
        },
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "numpy_blas": {"show_config": numpy_configuration.getvalue()},
    }


def hardware_context():
    try:
        nvidia_smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "output", "") or str(error)
        raise RuntimeError(
            f"nvidia-smi is required for frozen CPU/CUDA qualification: {detail}"
        ) from error
    import torch

    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_version": torch.version.cuda,
        "nvidia_smi_gpu_inventory": nvidia_smi,
    }


def assert_gpu_idle_for_timing():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "output", "") or str(error)
        raise RuntimeError(
            f"nvidia-smi compute-process guard unavailable: {detail}"
        ) from error
    try:
        pids = {int(line.strip()) for line in output.splitlines() if line.strip()}
    except ValueError as error:
        raise RuntimeError(
            f"Unparseable nvidia-smi compute-process output: {output!r}"
        ) from error
    foreign = sorted(pids - {os.getpid()})
    if foreign:
        raise RuntimeError(
            f"GPU timing blocked by foreign compute processes: {foreign}"
        )
    return {
        "checked_unix_seconds": time.time(),
        "current_pid": os.getpid(),
        "observed_compute_pids": sorted(pids),
        "foreign_compute_pids": [],
    }


@contextlib.contextmanager
def benchmark_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another host benchmark holds the qualification lock: {LOCK_PATH}"
            ) from error
        evidence = {
            "path": str(LOCK_PATH),
            "pid": os.getpid(),
            "nonblocking_exclusive": True,
            "acquired_unix_seconds": time.time(),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(evidence, sort_keys=True) + "\n")
        handle.flush()
        try:
            yield evidence
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare(args):
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(
            "Protocol already frozen; choose a fresh output directory."
        )
    if sha(args.checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Official checkpoint changed.")
    payload = {
        **protocol_policy(),
        "source_manifest": source_manifest(),
        "environment": environment(),
        "hardware_context": hardware_context(),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "panel_sha256": PANEL_SHA,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=ROOT)
        ).hexdigest(),
    }
    write_json(protocol_path, payload)
    print(f"prepared {protocol_path} sha256={sha(protocol_path)}", flush=True)


def verify(args):
    path = args.output_dir / "protocol.json"
    protocol = json.loads(path.read_text())
    policy = protocol_policy()
    frozen_policy = {name: protocol.get(name) for name in policy}
    if frozen_policy != policy:
        raise RuntimeError("Frozen benchmark policy drift.")
    if (
        protocol["source_manifest"] != source_manifest()
        or protocol["environment"] != environment()
        or protocol["hardware_context"] != hardware_context()
    ):
        raise RuntimeError(
            "Frozen source/runtime drift; preserve this run and prepare a new version."
        )
    if sha(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint drift.")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    diff_hash = hashlib.sha256(
        subprocess.check_output(["git", "diff"], cwd=ROOT)
    ).hexdigest()
    if head != protocol["git_head"] or diff_hash != protocol["tracked_diff_sha256"]:
        raise RuntimeError("Frozen git revision or tracked diff drift.")
    return protocol


def memory_read():
    fields = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            if name in {"Rss", "Private_Clean", "Private_Dirty"}:
                fields[name] = int(value.split()[0]) * 1024
    return fields["Rss"], fields["Private_Clean"] + fields["Private_Dirty"]


class PeakMonitor:
    def __init__(self):
        self.start_rss, self.start_uss = memory_read()
        self.rss, self.uss = self.start_rss, self.start_uss
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while True:
            rss, uss = memory_read()
            self.rss, self.uss = max(rss, self.rss), max(uss, self.uss)
            self.samples += 1
            if self.stop_event.wait(0.01):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        rss, uss = memory_read()
        self.rss, self.uss = max(rss, self.rss), max(uss, self.uss)
        self.stop_event.set()
        self.thread.join()

    def result(self):
        return {
            "rss_start_bytes": self.start_rss,
            "uss_start_bytes": self.start_uss,
            "rss_sampled_peak_bytes": self.rss,
            "uss_sampled_peak_bytes": self.uss,
            "samples": self.samples,
            "sample_period_seconds": 0.01,
        }


def plain(value):
    from maple.solvation.derivatives.analytic import _plain_json

    return _plain_json(value)


def build_pes(args, case):
    if args.backend == "reference":
        from maple.solvation.experimental.mace_polar_torch import (
            build_smd_mace_polar_torch_pes as builder,
        )
    else:
        from maple.solvation.experimental.mace_polar_response import (
            build_smd_mace_polar_response_pes as builder,
        )
    return builder(
        case["symbols"],
        solvent=case["solvent"],
        device=args.device,
        checkpoint_path=args.checkpoint,
    )


def direction(atoms):
    vector = np.random.default_rng(DIRECTION_SEED).normal(size=(len(atoms), 3))
    return vector / np.linalg.norm(vector)


def serialize_hessian(result):
    return {
        "energy_eV": result.energy_eV,
        "forces_eV_A": result.forces_eV_per_A.tolist(),
        "hessian_eV_A2": result.hessian_eV_per_A2.tolist(),
        "raw_antisymmetry_eV_A2": result.maximum_antisymmetry_eV_per_A2,
        "evaluation_sha256": result.evaluation_sha256,
        "configuration_sha256": result.configuration_sha256,
        "derivative_method": result.derivative_method,
        "components_eV": dict(result.component_energies_eV),
        "diagnostics": plain(result.diagnostics),
    }


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_witnesses_pass(diagnostics, floor):
    witnesses = diagnostics.get("source_witnesses")
    if not isinstance(witnesses, dict):
        return False
    if (
        witnesses.get("schema") != "macepolar-response-source-witness-v1"
        or witnesses.get("active_columns") != [0, 2, 3, 4]
        or witnesses.get("inactive_columns") != [1, 5, 6, 7]
        or witnesses.get("inactive_nonzero_count") != 0
        or witnesses.get("inactive_exact_zero") is not True
        or witnesses.get("charge_atol_e") != 1e-8
        or witnesses.get("neutral_charge_pass") is not True
        or witnesses.get("probe_policy") != "normalized-linspace-minus1-plus1-v1"
        or not _is_sha256(witnesses.get("probe_sha256"))
        or not _is_sha256(diagnostics.get("source_active_sha256"))
        or not _is_sha256(diagnostics.get("source_embedded_sha256"))
    ):
        return False
    charge_sum = witnesses.get("charge_sum_e")
    if (
        not isinstance(charge_sum, (int, float))
        or isinstance(charge_sum, bool)
        or not np.isfinite(charge_sum)
        or abs(charge_sum) > witnesses["charge_atol_e"]
    ):
        return False
    nonzero_fields = (
        "source_jacobian_frobenius_norm",
        "source_jvp_norm",
        "source_vjp_norm",
        "mixed_R_source_max_abs",
        "weighted_source_curvature_hvp_norm",
    )
    return all(
        isinstance(witnesses.get(name), (int, float))
        and not isinstance(witnesses[name], bool)
        and np.isfinite(witnesses[name])
        and witnesses[name] > floor
        for name in nonzero_fields
    )


def topology(pes, atoms):
    import torch

    positions = torch.tensor(atoms.positions, dtype=torch.float64, device=pes.device)
    if hasattr(pes.continuum, "_operator"):
        polar = pes.continuum._operator._geometry(positions).topology.topology_sha256
    else:
        from tools.route2_release.run_pure_mace_polar_torch_canary import _topology

        return _topology(pes, atoms)
    cds = pes.solvent_term.evaluate_torch(positions).diagnostics.dareal.topology_sha256
    model = pes.model.topology_diagnostics(atoms)["pair_mask_order_sha256"]
    return {"ddpcm": polar, "cds": cds, "model": model}


def validation(pes, atoms, args, protocol):
    result = pes.evaluate_hessian(atoms)
    payload = serialize_hessian(result)
    payload["source_witness_pass"] = _source_witnesses_pass(
        payload["diagnostics"], protocol["source_witness_floor"]
    )
    v = direction(atoms)
    hv = pes.hessian_vector_product(atoms, v).ravel()
    payload["hvp_eV_A2"] = hv.tolist()
    payload["hvp_vs_full_error_eV_A2"] = float(
        np.max(abs(hv - result.hessian_eV_per_A2 @ v.ravel()))
    )
    translations = []
    for axis in range(3):
        translation = np.zeros((len(atoms), 3), dtype=float)
        translation[:, axis] = 1.0
        translations.append(
            float(np.max(np.abs(result.hessian_eV_per_A2 @ translation.reshape(-1))))
        )
    payload["translation_null_mode_max_abs_eV_A2"] = translations
    payload["net_force_max_abs_eV_A"] = float(
        np.max(np.abs(np.sum(result.forces_eV_per_A, axis=0)))
    )
    from maple.solvation.models.mace_polar import (
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.experimental.mace_polar_frozen_ddx import (
        build_smd_mace_polar_frozen_point_ddx_pes,
    )

    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device, checkpoint_path=args.checkpoint
    )
    old = build_smd_mace_polar_frozen_point_ddx_pes(
        model, tuple(atoms.get_chemical_symbols()), solvent=pes.solvent
    )
    old_forces = old.evaluate_forces(atoms)
    payload["legacy_reference"] = {
        "energy_eV": old_forces.central_state.total_energy_eV,
        "forces_eV_A": old_forces.total_forces_eV_per_A.tolist(),
        "energy_error_eV": abs(
            result.energy_eV - old_forces.central_state.total_energy_eV
        ),
        "force_error_eV_A": float(
            np.max(abs(result.forces_eV_per_A - old_forces.total_forces_eV_per_A))
        ),
    }
    central = topology(pes, atoms)
    audits = []
    for step in AUDIT_STEPS:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions += step * v
        minus.positions -= step * v
        same = topology(pes, plus) == central == topology(pes, minus)
        audit = {"step_angstrom": step, "same_topology": same}
        if same:
            independent = -(old.get_forces(plus) - old.get_forces(minus)).ravel() / (
                2 * step
            )
            audit.update(
                hvp_eV_A2=independent.tolist(),
                maximum_error_eV_A2=float(np.max(abs(independent - hv))),
            )
        audits.append(audit)
    payload["legacy_force_difference_audit"] = audits
    fine_ok = all(
        a["same_topology"]
        and a.get("maximum_error_eV_A2", float("inf"))
        <= protocol["validation_limits"]["legacy_force_fd_eV_A2"]
        for a in audits[-2:]
    )
    fine_agreement = None
    if fine_ok:
        fine_agreement = float(
            np.max(
                abs(
                    np.asarray(audits[-1]["hvp_eV_A2"])
                    - np.asarray(audits[-2]["hvp_eV_A2"])
                )
            )
        )
        fine_ok = (
            fine_agreement <= protocol["validation_limits"]["fine_step_agreement_eV_A2"]
        )
    payload["fine_step_agreement_eV_A2"] = fine_agreement
    payload["validation_pass"] = (
        payload["hvp_vs_full_error_eV_A2"]
        <= protocol["validation_limits"]["hvp_vs_full_eV_A2"]
        and payload["raw_antisymmetry_eV_A2"]
        <= protocol["validation_limits"]["raw_antisymmetry_eV_A2"]
        and payload["legacy_reference"]["energy_error_eV"]
        <= protocol["accuracy_limits"]["energy_eV"]
        and payload["legacy_reference"]["force_error_eV_A"]
        <= protocol["accuracy_limits"]["force_eV_A"]
        and max(payload["translation_null_mode_max_abs_eV_A2"])
        <= protocol["validation_limits"]["translation_null_mode_eV_A2"]
        and payload["net_force_max_abs_eV_A"]
        <= protocol["validation_limits"]["net_force_eV_A"]
        and payload["source_witness_pass"]
        and fine_ok
    )
    return payload


def _cuda_measurement_guard(device):
    if not device.startswith("cuda"):
        return None
    return assert_gpu_idle_for_timing()


def _record_source_check(row, args, stage):
    """Bind warmup and each measured observation to the frozen protocol."""
    protocol = verify(args)
    manifest = protocol.get("source_manifest") if isinstance(protocol, dict) else None
    row.setdefault("source_checks", []).append(
        {
            "stage": stage,
            "checked_unix_seconds": time.time(),
            "source_manifest_sha256": (
                canonical_sha(manifest) if manifest is not None else None
            ),
        }
    )


def _continuum_memory_inputs(pes, atoms):
    """Derive the sealed source before the caller releases the model and PES."""
    resources = pes._preflight(2)
    positions = pes._positions(atoms, requires_grad=True)
    _, _, source = pes._model_and_source(atoms, positions)
    return (
        resources,
        pes.continuum,
        positions.detach().clone(),
        source.detach().clone(),
    )


def _measure_continuum_memory(resources, continuum, positions, source, args):
    """Measure only a fresh order-two continuum state after model/PES release."""
    import torch

    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        cuda_start_allocated = torch.cuda.memory_allocated(args.device)
        cuda_start_reserved = torch.cuda.memory_reserved(args.device)
    before = _cuda_measurement_guard(args.device)
    with PeakMonitor() as monitor:
        state = continuum._linearize(positions, source, derivative_order=2)
        partial = state.hessian_partial()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
    after = _cuda_measurement_guard(args.device)
    memory = monitor.result()
    observed = {
        **memory,
        "cpu_uss_incremental_peak_bytes": max(
            0, memory["uss_sampled_peak_bytes"] - memory["uss_start_bytes"]
        ),
        "resource_preflight": resources,
        "continuum_estimate_bytes": resources["continuum"]["conservative_peak_bytes"],
        "claim_boundary": (
            "incremental process/GPU allocated memory only; not absolute continuum "
            "memory and not an allocator guarantee"
        ),
        "hessian_shape": list(partial.shape),
        "cuda_process_guard_before": before,
        "cuda_process_guard_after": after,
    }
    if args.device.startswith("cuda"):
        observed.update(
            cuda_start_allocated_bytes=cuda_start_allocated,
            cuda_start_reserved_bytes=cuda_start_reserved,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
        )
        observed["cuda_allocated_incremental_peak_bytes"] = max(
            0,
            observed["cuda_peak_allocated_bytes"]
            - observed["cuda_start_allocated_bytes"],
        )
    incremental = (
        observed["cuda_allocated_incremental_peak_bytes"]
        if args.device.startswith("cuda")
        else observed["cpu_uss_incremental_peak_bytes"]
    )
    observed["incremental_peak_bytes"] = incremental
    observed["resource_envelope_pass"] = (
        incremental <= observed["continuum_estimate_bytes"]
    )
    return observed


def _run_locked(args):
    import torch
    from ase import Atoms

    cases = json.loads(PANEL.read_text())["cases"]
    case = next(c for c in cases if c["name"] == args.case)
    row_path = (
        args.output_dir
        / "rows"
        / args.backend
        / args.device.replace(":", "-")
        / args.case
        / f"{args.mode}.json"
    )
    if row_path.exists():
        raise FileExistsError(f"Preserve prior result: {row_path}")
    row = {
        "protocol_sha256": sha(args.output_dir / "protocol.json"),
        "case": args.case,
        "backend": args.backend,
        "device": args.device,
        "mode": args.mode,
        "atom_count": len(case["symbols"]),
        "completed": False,
        "benchmark_lock": getattr(args, "_lock_evidence", None),
    }
    start = time.perf_counter()
    try:
        protocol = verify(args)
        if args.mode == "resource":
            if args.backend != "response":
                raise ValueError("Resource-only rows describe the response policy.")
            from maple.solvation.continuum.ddpcm_response import TorchDDPCMResponse

            estimate = TorchDDPCMResponse._resource_estimate(
                len(case["symbols"]),
                15,
                1202,
                2,
                protocol["resource_policy"]["continuum_limit_bytes"],
                "cuda" if args.device.startswith("cuda") else "cpu",
            )
            row.update(
                resource=asdict(estimate),
                within_limit=estimate.within_limit,
                dense_allocation_attempted=False,
                execution_claim=False,
                completed=True,
            )
        else:
            if case["positions_angstrom"] is None:
                raise ValueError("Composition-only case permits resource mode only.")
            atoms = Atoms(case["symbols"], positions=case["positions_angstrom"])
            atoms.info.update(charge=case["charge"], mult=case["multiplicity"])
            geometry_payload = {
                "symbols": case["symbols"],
                "positions_angstrom": case["positions_angstrom"],
                "charge": case["charge"],
                "multiplicity": case["multiplicity"],
            }
            if canonical_sha(geometry_payload) != case["geometry_sha256"]:
                raise RuntimeError("Prospective geometry binding failed.")
            pes = build_pes(args, case)
            row["load_seconds"] = time.perf_counter() - start
            row["geometry_sha256"] = case["geometry_sha256"]
            if args.mode == "validation":
                if args.backend == "response":
                    try:
                        row["resource_preflight"] = pes._preflight(2)
                    except MemoryError as error:
                        row["resource_boundary"] = {
                            "preflight_rejected": True,
                            "derivative_order": 2,
                            "message": str(error),
                        }
                        raise
                row["result"] = validation(pes, atoms, args, protocol)
            elif args.mode == "continuum-memory":
                if args.backend != "response":
                    raise ValueError("Continuum-memory mode requires response backend.")
                inputs = _continuum_memory_inputs(pes, atoms)
                del pes
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                row["result"] = _measure_continuum_memory(*inputs, args)
            else:
                if args.backend == "response":
                    row["resource_preflight"] = pes._preflight(2)

                def evaluate():
                    if args.mode == "hessian":
                        return pes.evaluate_hessian(atoms)
                    return pes.hessian_vector_product(atoms, direction(atoms))

                warm = time.perf_counter()
                output = evaluate()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                row["warmup_seconds"] = time.perf_counter() - warm
                del output
                _record_source_check(row, args, "after-warmup")
                row["measurements"] = []
                for index in range(REPETITIONS):
                    gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(args.device)
                        torch.cuda.reset_peak_memory_stats(args.device)
                        cuda_start_allocated = torch.cuda.memory_allocated(args.device)
                        cuda_start_reserved = torch.cuda.memory_reserved(args.device)
                    guard_before = _cuda_measurement_guard(args.device)
                    with PeakMonitor() as monitor:
                        begun = time.perf_counter()
                        output = evaluate()
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize(args.device)
                        elapsed = time.perf_counter() - begun
                    guard_after = _cuda_measurement_guard(args.device)
                    observed = {
                        "index": index,
                        "elapsed_seconds": elapsed,
                        **monitor.result(),
                        "cuda_process_guard_before": guard_before,
                        "cuda_process_guard_after": guard_after,
                    }
                    if args.device.startswith("cuda"):
                        observed["cuda_start_allocated_bytes"] = cuda_start_allocated
                        observed["cuda_start_reserved_bytes"] = cuda_start_reserved
                        observed["cuda_peak_allocated_bytes"] = (
                            torch.cuda.max_memory_allocated(args.device)
                        )
                        observed["cuda_peak_reserved_bytes"] = (
                            torch.cuda.max_memory_reserved(args.device)
                        )
                    observed["result"] = (
                        serialize_hessian(output)
                        if args.mode == "hessian"
                        else {"hvp_eV_A2": output.tolist()}
                    )
                    row["measurements"].append(observed)
                    _record_source_check(row, args, f"after-repetition-{index}")
                    row["total_seconds"] = time.perf_counter() - start
                    write_json(row_path, row)
                    print(
                        f"{args.backend} {args.case} {args.mode} {args.device} repetition {index+1}: {elapsed:.3f}s",
                        flush=True,
                    )
                    if index == REPETITIONS - 1:
                        row["result"] = observed["result"]
                    del output
            row["completed"] = True
        verify(args)
    except Exception as error:
        row["completed"] = False
        row["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    row["total_seconds"] = time.perf_counter() - start
    write_json(row_path, row)
    print(f"saved {row_path} completed={row['completed']}", flush=True)
    if "failure" in row:
        print(row["failure"]["traceback"], flush=True)
        raise SystemExit(1)
    if row.get("result", {}).get("validation_pass") is False:
        raise SystemExit(2)


def run(args):
    # The lock is host-wide and covers model loading, warmup and every repetition.
    with benchmark_lock() as lock_evidence:
        args._lock_evidence = lock_evidence
        return _run_locked(args)


def _required_row_identities(protocol):
    identities = set()
    for device in protocol["requested_devices"]:
        for case in ("water", "methane"):
            for mode in ("hvp", "hessian"):
                for backend in ("reference", "response"):
                    identities.add((backend, device, case, mode))
        for case in ("water", "methane", "acetone-10", "hexane-20"):
            identities.add(("response", device, case, "validation"))
        for case in protocol["continuum_memory_required_cases"]:
            identities.add(("response", device, case, "continuum-memory"))
        for case in ("hexadecane-50", "alkylamine-100"):
            identities.add(("response", device, case, "resource"))
    return identities


def _validated_measurements(record, memory_key):
    measurements = record.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != REPETITIONS:
        raise RuntimeError("Missing preregistered timing repetitions.")
    indexes = [measurement.get("index") for measurement in measurements]
    if indexes != list(range(REPETITIONS)):
        raise RuntimeError("Invalid preregistered repetition indexes.")
    for measurement in measurements:
        for key in ("elapsed_seconds", memory_key):
            value = measurement.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise RuntimeError(
                    f"Timing and memory measurements must be finite and positive: {key}"
                )
    return measurements


def _maximum_error(reference, candidate, label):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Numerical result shape mismatch for {label}.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise RuntimeError(f"Numerical result must be finite for {label}.")
    return float(np.max(np.abs(reference - candidate)))


def _median_and_mad(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return median, float(np.median(np.abs(values - median)))


def _validate_result_series(measurements, mode, atom_count):
    keys = (
        ("hvp_eV_A2",)
        if mode == "hvp"
        else (
            "forces_eV_A",
            "hessian_eV_A2",
        )
    )
    for key in keys:
        arrays = [
            np.asarray(measurement["result"][key], dtype=float)
            for measurement in measurements
        ]
        if (
            any(array.size == 0 for array in arrays)
            or len({array.shape for array in arrays}) != 1
        ):
            raise RuntimeError(
                f"Numerical result shape mismatch across repetitions: {key}"
            )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError(f"Numerical result must be finite for {key}.")
        expected = {
            "hvp_eV_A2": (atom_count, 3),
            "forces_eV_A": (atom_count, 3),
            "hessian_eV_A2": (3 * atom_count, 3 * atom_count),
        }[key]
        if arrays[0].shape != expected:
            raise RuntimeError(
                f"Numerical result shape mismatch for {key}: expected {expected}."
            )
    if mode == "hessian":
        energies = [measurement["result"]["energy_eV"] for measurement in measurements]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in energies
        ):
            raise RuntimeError("Numerical result must be finite for energy_eV.")


def summarize(args):
    protocol = verify(args)
    protocol_hash = sha(args.output_dir / "protocol.json")
    rows = {}
    for path in (args.output_dir / "rows").rglob("*.json"):
        record = json.loads(path.read_text())
        if record["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Result protocol binding failed: {path}")
        identity = tuple(record[key] for key in ("backend", "device", "case", "mode"))
        if identity in rows:
            raise RuntimeError(f"Duplicate result row identity: {identity}")
        rows[identity] = record
    missing = sorted(_required_row_identities(protocol) - rows.keys())
    if missing:
        raise RuntimeError(f"Missing preregistered rows: {missing}")
    unexpected = sorted(rows.keys() - _required_row_identities(protocol))
    if unexpected:
        raise RuntimeError(f"Unexpected result row identities: {unexpected}")
    report = {
        "protocol_sha256": protocol_hash,
        "comparisons": [],
        "validation": [],
        "resource_screens": [],
        "continuum_memory": [],
        "implementation_pass": False,
        "performance_pass": False,
        "stage1_pass_before_final_review": False,
        "final_review_passed": False,
        "scientific_admitted": False,
    }
    for device in protocol["requested_devices"]:
        memory_key = protocol["memory_metric"][
            "cuda" if device.startswith("cuda") else "cpu"
        ]
        for case in ("water", "methane"):
            for mode in ("hvp", "hessian"):
                comparison = {
                    "device": device,
                    "case": case,
                    "mode": mode,
                    "complete": False,
                    "accuracy_pass": False,
                    "performance_pass": False,
                }
                old = rows.get(("reference", device, case, mode))
                new = rows.get(("response", device, case, mode))
                if old and new and old["completed"] and new["completed"]:
                    om = _validated_measurements(old, memory_key)
                    nm = _validated_measurements(new, memory_key)
                    atom_counts = {old.get("atom_count"), new.get("atom_count")}
                    if len(atom_counts) != 1 or next(iter(atom_counts)) not in {
                        3,
                        5,
                    }:
                        raise RuntimeError("Timing row atom-count binding failed.")
                    atom_count = next(iter(atom_counts))
                    _validate_result_series([*om, *nm], mode, atom_count)
                    ot, nt = [r["elapsed_seconds"] for r in om], [
                        r["elapsed_seconds"] for r in nm
                    ]
                    ob, nb = [r[memory_key] for r in om], [r[memory_key] for r in nm]
                    old_time_median, old_time_mad = _median_and_mad(ot)
                    new_time_median, new_time_mad = _median_and_mad(nt)
                    old_memory_median, old_memory_mad = _median_and_mad(ob)
                    new_memory_median, new_memory_mad = _median_and_mad(nb)
                    t_ratio = new_time_median / old_time_median
                    m_ratio = new_memory_median / old_memory_median
                    errors = []
                    for a, b in zip(om, nm, strict=True):
                        a, b = a["result"], b["result"]
                        if mode == "hvp":
                            errors.append(
                                {
                                    "hvp_eV_A2": _maximum_error(
                                        a["hvp_eV_A2"],
                                        b["hvp_eV_A2"],
                                        "hvp_eV_A2",
                                    )
                                }
                            )
                        else:
                            energies = (a["energy_eV"], b["energy_eV"])
                            if not all(np.isfinite(value) for value in energies):
                                raise RuntimeError(
                                    "Numerical result must be finite for energy_eV."
                                )
                            errors.append(
                                {
                                    "energy_eV": abs(a["energy_eV"] - b["energy_eV"]),
                                    "force_eV_A": _maximum_error(
                                        a["forces_eV_A"],
                                        b["forces_eV_A"],
                                        "forces_eV_A",
                                    ),
                                    "hessian_eV_A2": _maximum_error(
                                        a["hessian_eV_A2"],
                                        b["hessian_eV_A2"],
                                        "hessian_eV_A2",
                                    ),
                                }
                            )
                    accuracy = all(
                        value <= protocol["accuracy_limits"][key]
                        for e in errors
                        for key, value in e.items()
                    )
                    memory_limit = protocol["performance"][
                        f"maximum_{mode}_memory_ratio"
                    ]
                    resource_envelope_pass = all(
                        measurement[memory_key]
                        <= new["resource_preflight"]["estimated_total_bytes"]
                        for measurement in nm
                    )
                    performance = (
                        t_ratio <= protocol["performance"]["maximum_median_time_ratio"]
                        and m_ratio <= memory_limit
                        and max(nt) < min(ot)
                        and max(nb) < min(ob)
                        and resource_envelope_pass
                    )
                    comparison.update(
                        complete=True,
                        accuracy_pass=accuracy,
                        performance_pass=performance,
                        errors_per_repetition=errors,
                        median_time_ratio=t_ratio,
                        median_memory_ratio=m_ratio,
                        old_time_median_seconds=old_time_median,
                        old_time_mad_seconds=old_time_mad,
                        new_time_median_seconds=new_time_median,
                        new_time_mad_seconds=new_time_mad,
                        old_memory_median_bytes=old_memory_median,
                        old_memory_mad_bytes=old_memory_mad,
                        new_memory_median_bytes=new_memory_median,
                        new_memory_mad_bytes=new_memory_mad,
                        repetition_order=list(range(REPETITIONS)),
                        memory_metric=memory_key,
                        resource_envelope_pass=resource_envelope_pass,
                        estimated_total_bytes=new["resource_preflight"][
                            "estimated_total_bytes"
                        ],
                        old_seconds=ot,
                        new_seconds=nt,
                        old_peak_bytes=ob,
                        new_peak_bytes=nb,
                    )
                report["comparisons"].append(comparison)
        for case in ("water", "methane", "acetone-10", "hexane-20"):
            item = rows.get(("response", device, case, "validation"))
            report["validation"].append(
                {
                    "device": device,
                    "case": case,
                    "attempted": item is not None,
                    "completed": bool(item and item["completed"]),
                    "passed": bool(
                        item and item.get("result", {}).get("validation_pass")
                    ),
                    "failure": item.get("failure") if item else None,
                    "resource_boundary": (
                        item.get("resource_boundary") if item else None
                    ),
                }
            )
        for case in ("hexadecane-50", "alkylamine-100"):
            item = rows.get(("response", device, case, "resource"))
            resource_record = item.get("resource", {}) if item else {}
            expected_atoms = 50 if case == "hexadecane-50" else 100
            resource_valid = bool(
                item
                and item["completed"]
                and item.get("atom_count") == expected_atoms
                and resource_record.get("atom_count") == expected_atoms
                and resource_record.get("derivative_order") == 2
                and resource_record.get("conservative_peak_bytes", 0) > 0
                and (
                    (
                        device == "cpu"
                        and resource_record.get("host_peak_bytes")
                        == resource_record.get("conservative_peak_bytes")
                        and resource_record.get("device_peak_bytes") == 0
                    )
                    or (
                        device.startswith("cuda")
                        and resource_record.get("device_peak_bytes", 0) > 0
                        and resource_record.get("host_peak_bytes", 0) >= 0
                        and resource_record.get("host_peak_bytes", 0)
                        + resource_record.get("device_peak_bytes", 0)
                        == resource_record.get("conservative_peak_bytes")
                    )
                )
            )
            report["resource_screens"].append(
                {
                    "device": device,
                    "case": case,
                    "completed": bool(item and item["completed"]),
                    "resource": resource_record or None,
                    "resource_record_valid": resource_valid,
                    "within_limit": item.get("within_limit") if item else None,
                }
            )
        for case in protocol["continuum_memory_required_cases"]:
            item = rows.get(("response", device, case, "continuum-memory"))
            result = item.get("result", {}) if item else {}
            report["continuum_memory"].append(
                {
                    "device": device,
                    "case": case,
                    "completed": bool(item and item["completed"]),
                    "incremental_peak_bytes": result.get("incremental_peak_bytes"),
                    "continuum_estimate_bytes": result.get("continuum_estimate_bytes"),
                    "resource_envelope_pass": bool(
                        item
                        and item["completed"]
                        and result.get("resource_envelope_pass")
                    ),
                    "claim_boundary": result.get("claim_boundary"),
                    "failure": item.get("failure") if item else None,
                }
            )
    report["implementation_pass"] = all(
        x["accuracy_pass"] for x in report["comparisons"]
    )
    report["performance_pass"] = all(
        x["performance_pass"] for x in report["comparisons"]
    )
    mandatory = all(
        x["passed"] for x in report["validation"] if x["case"] != "hexane-20"
    )
    attempted20 = all(
        (x["completed"] and x["passed"])
        or (
            not x["completed"]
            and x["failure"] is not None
            and x["failure"].get("type") == "MemoryError"
            and (x.get("resource_boundary") or {}).get("preflight_rejected") is True
        )
        for x in report["validation"]
        if x["case"] == "hexane-20"
    )
    resource = all(x["resource_record_valid"] for x in report["resource_screens"])
    continuum_resource = all(
        x["resource_envelope_pass"] for x in report["continuum_memory"]
    )
    report["stage1_pass_before_final_review"] = (
        report["implementation_pass"]
        and report["performance_pass"]
        and mandatory
        and attempted20
        and resource
        and continuum_resource
    )
    write_json(args.output_dir / "summary.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "comparisons",
                    "validation",
                    "resource_screens",
                    "continuum_memory",
                }
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-POLAR-1-M.model",
    )
    parser.add_argument(
        "--backend", choices=("reference", "response"), default="response"
    )
    parser.add_argument("--case", default="water")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--mode",
        choices=("hvp", "hessian", "validation", "resource", "continuum-memory"),
        default="hessian",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
