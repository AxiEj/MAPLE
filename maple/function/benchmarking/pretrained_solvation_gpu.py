"""Diagnostic-only raw pretrained-solvation panel driver.

Run with ``python -m maple.function.benchmarking.pretrained_solvation_gpu``.
This module gathers admission evidence; it never constructs a production gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .solvation_gpu import (
    FROZEN_PANEL_SHA256,
    collect_numerical_panel,
    load_frozen_panel,
)

CALCULATOR_ROOT = Path(__file__).resolve().parents[1] / "calculator"
DEFAULT_SOURCE_ROOT = Path.home() / ".cache/maple/anysolv/anysolv"
AIMNET2_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: {path}: expected {expected}, got {actual}"
        )
    return {"path": str(Path(path).resolve()), "sha256": actual}


def _sync(execution):
    if execution.device != "cpu":
        import torch

        torch.cuda.synchronize(execution.device)


def _memory(execution):
    if execution.device == "cpu":
        return {"device": "cpu", "cuda_memory": "inapplicable"}
    import torch

    free, total = torch.cuda.mem_get_info(execution.device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(execution.device),
        "reserved_bytes": torch.cuda.memory_reserved(execution.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(execution.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(execution.device),
        "free_bytes": free,
        "total_bytes": total,
    }


def _reset_peak(execution):
    if execution.device != "cpu":
        import torch

        # PyTorch's allocator statistics are unavailable until lazy CUDA init.
        # Device availability alone does not initialize the caching allocator.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(execution.device)


def _release(execution):
    gc.collect()
    if execution.device != "cpu":
        import torch

        torch.cuda.empty_cache()


def _state_hash(model):
    """CPU byte hash outside measured inference, including integer buffers."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


class _Owners:
    def __init__(self, execution):
        self.execution = execution
        self.events = []

    def _new_event(self, signature, started):
        _sync(self.execution)
        event = {
            "signature": repr(signature),
            "load_and_transfer_seconds": time.perf_counter() - started,
            "memory_after_initialization": _memory(self.execution),
            "first_calls": [],
        }
        self.events.append(event)
        return event

    def _call(self, event, variant, evaluate):
        # Only cold variants synchronize/time here; the harness owns warm timing.
        if any(row["variant"] == variant for row in event["first_calls"]):
            return evaluate()
        _sync(self.execution)
        started = time.perf_counter()
        row = {
            "variant": variant,
            "status": "failed",
            "timing_scope": "first_call_including_export_compile_and_sentinel_when_optimized",
        }
        try:
            result = evaluate()
            row["status"] = "success"
            return result
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            _sync(self.execution)
            row.update(
                seconds=time.perf_counter() - started, memory=_memory(self.execution)
            )
            event["first_calls"].append(row)


class AnySolvDiagnostic(_Owners):
    def __init__(
        self,
        execution,
        source_root=None,
        *,
        runtime_factory=None,
        checkpoint=None,
        checkpoint_sha256=None,
        validate_source=None,
    ):
        super().__init__(execution)
        from maple.function.calculator.extra_correction.implicit import anysolv as api

        self.source_root = (
            Path(source_root or DEFAULT_SOURCE_ROOT).expanduser().resolve()
        )
        self.source_identity = (validate_source or api._validate_source)(
            self.source_root
        )
        self.checkpoint = Path(checkpoint or api.ANYSOLV_MOE_DEFAULT_PATH)
        # Raw runtime does not hash weights: verify before any construction.
        self.checkpoint_identity = verify_checkpoint(
            self.checkpoint, checkpoint_sha256 or api.ANYSOLV_MOE_CHECKPOINT_SHA256
        )
        self.factory = runtime_factory or api._OfficialAnySolvRuntime
        self.runtime = None
        self.signature = None
        self.failed = None
        self.source_revision = api.ANYSOLV_SOURCE_REVISION

    def _archive(self):
        if self.runtime is not None:
            self.events[-1].update(
                compilation_active=bool(
                    getattr(self.runtime, "compilation_active", False)
                ),
                backend_executions=getattr(
                    self.runtime, "_compiled_backend_executions", 0
                ),
                memory_before_release=_memory(self.execution),
            )

    def __call__(self, numbers, positions, solvent, need_forces):
        if need_forces:
            raise ValueError("AnySolv diagnostic exposes scalar corrections only.")
        signature = (tuple(sorted(np.asarray(numbers).tolist())), solvent)
        if self.failed is not None:
            raise RuntimeError(f"AnySolv owner failed; retries disabled: {self.failed}")
        if self.runtime is None or (
            self.execution.inference == "optimized" and signature != self.signature
        ):
            self._archive()
            self.runtime = None
            _release(self.execution)
            if self.events:
                self.events[-1]["memory_after_release"] = _memory(self.execution)
            _reset_peak(self.execution)
            started = time.perf_counter()
            try:
                with self.execution.precision_context():
                    self.runtime = self.factory(
                        source_root=self.source_root,
                        checkpoint=self.checkpoint,
                        checkpoint_sha256=self.checkpoint_identity["sha256"],
                        device=self.execution.device,
                        dtype=self.execution.dtype,
                        inference=self.execution.inference,
                    )
            except Exception as exc:
                self.failed = str(exc)
                self._new_event(signature, started).update(
                    initialization_error=str(exc)
                )
                raise
            self.signature = signature
            self._new_event(signature, started)
        try:

            def evaluate():
                energy, _ = self.runtime.evaluate(
                    numbers,
                    positions,
                    charge=0,
                    multiplicity=1,
                    solvent=solvent,
                    scalar_only=True,
                )
                return {"energy_ev": float(energy)}

            return self._call(self.events[-1], "scalar", evaluate)
        except Exception as exc:
            self.failed = str(exc)
            raise

    def diagnostics(self):
        self._archive()
        return {
            "owners": self.events,
            "resident_runtime_count": int(self.runtime is not None),
            "checkpoint": self.checkpoint_identity,
            "source_revision": self.source_revision,
            "source_identity": self.source_identity,
        }


class ConSolvDiagnostic(_Owners):
    def __init__(
        self, execution, *, loader=None, compiled_factory=None, model_path=None
    ):
        super().__init__(execution)
        from maple.function.calculator.consolv._torch_model import (
            load_consolv_torch_checkpoint,
            CHECKPOINT_IDENTITIES,
            SOURCE_REVISION,
        )
        from maple.function.calculator.consolv._consolv_calculator import (
            CONSOLV_DEFAULT_MODEL_PATH,
            CONSOLV_SOLVENTS,
        )
        from maple.function.calculator.consolv._compiled import ConSolvCompiledGraph

        self.solvents = CONSOLV_SOLVENTS
        self.compiled_factory = compiled_factory or ConSolvCompiledGraph
        self.models, self.current, self.failed = {}, {}, {}
        self.checkpoints, self.state_before = {}, {}
        self.source_revision = SOURCE_REVISION
        path = Path(model_path or CONSOLV_DEFAULT_MODEL_PATH)
        for state, checkpoint in (
            ("solution", path),
            ("vacuum", path.with_name("consolv-vacuum-torch-v1.pt")),
        ):
            self.checkpoints[state] = verify_checkpoint(
                checkpoint, CHECKPOINT_IDENTITIES[state]
            )
            started = time.perf_counter()
            with execution.precision_context():
                model = (loader or load_consolv_torch_checkpoint)(
                    checkpoint,
                    device=execution.device,
                    dtype=execution.torch_dtype,
                    expected_state=state,
                    inference=execution.inference,
                )
                execution.check_precision(model=model)
            self.models[state] = model
            self._new_event((state, "weights_load_once"), started)
            self.state_before[state] = _state_hash(model)

    def _archive(self, state):
        if state not in self.current:
            return
        _, event = self.current[state]
        owner = self.models[state]._compiled_energy
        if owner is not None:
            event["export_records"] = list(owner.export_records)
            event["variants"] = [
                {"requires_grad": bool(grad), **dict(counters)}
                for grad, (_, counters) in owner._variants.items()
            ]
        event["memory_before_release"] = _memory(self.execution)

    def __call__(self, numbers, positions, solvent, need_forces):
        state = "vacuum" if solvent == "vacuum" else "solution"
        try:
            return self._evaluate(numbers, positions, solvent, need_forces)
        except Exception as exc:
            self.failed[state] = str(exc)
            raise

    def _evaluate(self, numbers, positions, solvent, need_forces):
        import torch

        state = "vacuum" if solvent == "vacuum" else "solution"
        if state in self.failed:
            raise RuntimeError(
                f"ConSolv owner failed; retries disabled: {self.failed[state]}"
            )
        model = self.models[state]
        index = 0 if state == "vacuum" else self.solvents[solvent]
        with self.execution.precision_context():
            n = torch.as_tensor(numbers, dtype=torch.long, device=self.execution.device)
            r = torch.as_tensor(
                positions,
                dtype=self.execution.torch_dtype,
                device=self.execution.device,
            )
            self.execution.check_precision(inputs=(n, r), model=model)
            signature = (
                len(numbers),
                index,
                self.execution.device,
                self.execution.dtype,
            )
            if self.execution.inference == "optimized":
                sender, _, _ = model._valid_edges(r)
                signature += (sender.numel(),)
            if state not in self.current or self.current[state][0] != signature:
                self._archive(state)
                model._compiled_energy = None
                _release(self.execution)
                if state in self.current:
                    self.current[state][1]["memory_after_release"] = _memory(
                        self.execution
                    )
                _reset_peak(self.execution)
                started = time.perf_counter()
                if self.execution.inference == "optimized":
                    model._compiled_energy = self.compiled_factory(model)
                self.current[state] = (
                    signature,
                    self._new_event((state, signature), started),
                )

            def evaluate():
                if need_forces:
                    energy, force = model.energy_and_forces_ev(n, r, index)
                    result = {
                        "energy_ev": float(energy.detach()),
                        "forces_ev_per_angstrom": force.detach().cpu().numpy(),
                    }
                else:
                    result = {"energy_ev": float(model.energy_ev(n, r, index).detach())}
                self.execution.check_precision(inputs=(n, r), model=model)
                return result

            try:
                return self._call(
                    self.current[state][1],
                    "forces" if need_forces else "energy",
                    evaluate,
                )
            except Exception as exc:
                self.failed[state] = str(exc)
                raise

    def diagnostics(self):
        for state in self.current:
            self._archive(state)
        after = {state: _state_hash(model) for state, model in self.models.items()}
        return {
            "owners": self.events,
            "checkpoints": self.checkpoints,
            "source_revision": self.source_revision,
            "state_sha256_before": self.state_before,
            "state_sha256_after": after,
            "source_state_unchanged": self.state_before == after,
            "resident_weight_models": len(self.models),
            "current_compiled_owners": sum(
                model._compiled_energy is not None for model in self.models.values()
            ),
        }


def cast_floating_data(data, dtype):
    """Preserve integer neighbor/index tensors while matching native gas precision."""
    import torch

    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype) if data.is_floating_point() else data
    if isinstance(data, dict):
        return {key: cast_floating_data(value, dtype) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(cast_floating_data(value, dtype) for value in data)
    return data


def collect_gas_base(
    execution, panel, scalar_records, *, calculator_factory=None, checkpoint=None
):
    from ase import Atoms
    import torch
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    identity = verify_checkpoint(
        checkpoint or CALCULATOR_ROOT / "model/aimnet2.pt", AIMNET2_SHA256
    )
    initialization_error = None
    try:
        with execution.precision_context():
            calc = (calculator_factory or AIMNet2Calculator)(
                device=torch.device(execution.device),
                model_path=identity["path"],
                coulomb_method="simple",
            )
            calc.model.to(dtype=execution.torch_dtype)
    except Exception as exc:
        initialization_error = str(exc)
    cases = [case for case in panel["scalar_cases"] if case["route"] == "anysolv"]
    gas, gas_records = {}, []
    for case in cases:
        geometry = case["geometry_sha256"]
        if geometry in gas:
            continue
        record = {
            "geometry_sha256": geometry,
            "case_id": case["case_id"],
            "status": "failed",
        }
        try:
            if initialization_error is not None:
                raise RuntimeError(f"Gas initialization failed: {initialization_error}")
            with execution.precision_context():
                atoms = Atoms(
                    numbers=case["numbers"], positions=case["positions_angstrom"]
                )
                atoms.info.update(charge=0, mult=1)
                positions = torch.tensor(
                    case["positions_angstrom"],
                    dtype=execution.torch_dtype,
                    device=execution.device,
                )
                data = cast_floating_data(
                    calc._build_data(positions, atoms), execution.torch_dtype
                )
                execution.check_precision(inputs=data, model=calc.model)
                energy = calc._forward_energy(data)
                execution.check_precision(inputs=data, model=calc.model)
                value = float(energy.detach())
                if not math.isfinite(value):
                    raise ValueError("Gas energy must be finite.")
                record.update(status="success", gas_ev=value)
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
        gas[geometry] = record
        gas_records.append(record)
    corrections = {record["case_id"]: record for record in scalar_records}
    combined = []
    for case in cases:
        base = gas[case["geometry_sha256"]]
        correction = corrections[case["case_id"]]
        row = {
            "case_id": case["case_id"],
            "geometry_sha256": case["geometry_sha256"],
            "atom_count": len(case["numbers"]),
            "status": "failed",
        }
        if base["status"] == correction["status"] == "success":
            row.update(
                status="success",
                gas_ev=base["gas_ev"],
                correction_ev=correction["energy_ev"],
                combined_ev=base["gas_ev"] + correction["energy_ev"],
            )
        else:
            row.update(error="Gas or correction evaluation failed.")
        combined.append(row)
    return {
        "checkpoint": identity,
        "gas_reference_compute_dtype": execution.dtype,
        "gas_records": gas_records,
        "combined_records": combined,
    }


def source_files(route, source_root=None, withgas=False):
    files = [
        Path(__file__),
        Path(__file__).with_name("solvation_gpu.py"),
        CALCULATOR_ROOT / "solvation_execution.py",
    ]
    if route == "anysolv":
        from maple.function.calculator.extra_correction.implicit.anysolv import (
            ANYSOLV_SOURCE_ARTIFACTS,
        )

        files.append(CALCULATOR_ROOT / "extra_correction/implicit/anysolv.py")
        files.extend(
            Path(source_root or DEFAULT_SOURCE_ROOT) / relative
            for relative in ANYSOLV_SOURCE_ARTIFACTS
        )
    else:
        files.extend(
            CALCULATOR_ROOT / "consolv" / name
            for name in ("_torch_model.py", "_compiled.py", "_consolv_calculator.py")
        )
    if withgas:
        files.append(CALCULATOR_ROOT / "aimnet/_aimnet2_calculator.py")
    return files


def _hash_sources(files):
    return {
        str(path.resolve()): file_sha256(path) if path.is_file() else None
        for path in files
    }


def run_diagnostic(
    route, execution, panel, source_root=None, withgas=False, *, adapter_factory=None
):
    if route not in {"anysolv", "consolv"}:
        raise ValueError("route must be anysolv or consolv")
    if withgas and route != "anysolv":
        raise ValueError("--with-gas-base requires anysolv")
    files = source_files(route, source_root, withgas)
    before = _hash_sources(files)
    started = time.perf_counter()
    _reset_peak(execution)
    initialization_error = None
    adapter = None
    try:
        factory = adapter_factory or (
            AnySolvDiagnostic if route == "anysolv" else ConSolvDiagnostic
        )
        adapter = (
            factory(execution, source_root)
            if route == "anysolv"
            else factory(execution)
        )
        evaluate = adapter
    except Exception as exc:
        initialization_error = {"error_type": type(exc).__name__, "error": str(exc)}

        def evaluate(*args):
            raise RuntimeError(
                f"Diagnostic initialization failed: {initialization_error}"
            )

    initialized = time.perf_counter() - started
    report = collect_numerical_panel(
        evaluate,
        panel,
        route,
        inference_mode=execution.inference,
        synchronize=lambda: _sync(execution),
        include_timing=True,
    )
    report.update(
        diagnostic=True,
        no_loss=False,
        execution=execution.provenance,
        initialization_seconds=initialized,
        initialization_error=initialization_error,
    )
    with execution.precision_context():
        report["runtime"] = execution.runtime_context()
        report["effective_precision"] = execution.check_precision()
    if adapter is not None:
        report["adapter"] = adapter.diagnostics()
    if withgas:
        try:
            report["gas_base"] = collect_gas_base(
                execution, panel, report["scalar_records"]
            )
        except Exception as exc:
            report["gas_base"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    if route == "consolv":
        report.update(
            force_reference="cpu64_central_fd",
            same_dtype_fd_classification="diagnostic_failed_resolution_limited_for_fp32; raw errors retained; independent CPU64 audit required",
        )
    report["memory"] = _memory(execution)
    report["source_sha256_before"] = before
    report["source_sha256_after"] = _hash_sources(files)
    report["source_drift"] = before != report["source_sha256_after"]
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("--route", choices=("anysolv", "consolv"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--inference", choices=("reference", "optimized"), default="reference"
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--with-gas-base", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.with_gas_base and args.route != "anysolv":
        parser.error("--with-gas-base requires --route anysolv")
    panel = load_frozen_panel(args.panel, FROZEN_PANEL_SHA256)
    # Exclusive creation reserves the receipt before any model work.
    with args.output.open("x") as stream:
        try:
            from maple.function.calculator.solvation_execution import SolvationExecution

            execution = SolvationExecution(args.device, args.dtype, args.inference)
            report = run_diagnostic(
                args.route, execution, panel, args.source_root, args.with_gas_base
            )
        except Exception as exc:
            report = {
                "diagnostic": True,
                "route": args.route,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return int(
        report.get("status") == "failed"
        or report.get("source_drift", False)
        or report.get("counts", {}).get("failed_records", 0) > 0
        or report.get("gas_base", {}).get("status") == "failed"
        or any(
            row["status"] != "success"
            for row in report.get("gas_base", {}).get("combined_records", [])
        )
        or report.get("timing", {}).get("status") == "failed"
        or report.get("adapter", {}).get("source_state_unchanged") is False
    )


if __name__ == "__main__":
    raise SystemExit(main())
