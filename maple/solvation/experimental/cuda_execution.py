"""Fixed serial inference policy for the pure CUDA-v2 scalar.

Deterministic reductions prevent repeated-source bit drift. Disabling JIT
executor optimization also prevents first-call profiling/fusion transitions
from changing the arithmetic during strict replay. Neither guard is weakened.
Torch's process-wide flags are restored even when inference fails; concurrent
foreign Torch calls are outside this explicitly serial execution contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from threading import RLock

CUDA_EXECUTION_POLICY = "deterministic-no-jit-optimization-cublas4096-v1"
_WORKSPACE = ":4096:8"
_LOCK = RLock()


def prepare_cuda_execution():
    import torch

    with _LOCK:
        configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured is None:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization "
                    "for the pure CUDA-v2 execution policy."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _WORKSPACE
        elif configured != _WORKSPACE:
            raise RuntimeError("Pure CUDA-v2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")


def execution_policy_metadata():
    import hashlib

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _WORKSPACE:
        raise RuntimeError("Pure CUDA-v2 workspace configuration drifted.")
    return {
        "policy": CUDA_EXECUTION_POLICY,
        "cublas_workspace": _WORKSPACE,
        "deterministic_algorithms": True,
        "jit_executor_optimization": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


@contextmanager
def cuda_model_execution():
    import torch

    with _LOCK:
        execution_policy_metadata()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            with torch.jit.optimized_execution(False):
                yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
