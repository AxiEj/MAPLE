"""Shared, fail-closed PyTorch device resolution."""

from __future__ import annotations

import re
from typing import Any


def _load_torch():
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "PyTorch is required to resolve an accelerator device."
        ) from exc
    return torch


def resolve_torch_device(device_name: Any) -> Any:
    """Resolve MAPLE device syntax; only ``auto`` may fall back to CPU."""
    torch = _load_torch()
    if device_name is None:
        return torch.device("cpu")
    if not isinstance(device_name, str):
        raise TypeError(
            f"Invalid device {device_name!r}; expected auto, cpu, cuda[:N], "
            "gpu[N], mps, or xpu[:N]."
        )

    name = device_name.strip().lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cuda_match = re.fullmatch(r"(?:cuda|gpu)(?::?(\d+))?", name)
    if cuda_match:
        index = int(cuda_match.group(1) or 0)
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device '{device_name}' was explicitly requested, but CUDA is not available."
            )
        count = torch.cuda.device_count()
        if index >= count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; PyTorch reports {count} device(s)."
            )
        return torch.device(f"cuda:{index}")

    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError(
                "MPS device was explicitly requested, but MPS is not available."
            )
        return torch.device("mps")

    xpu_match = re.fullmatch(r"xpu(?::(\d+))?", name)
    if xpu_match:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise ValueError(
                f"XPU device '{device_name}' was explicitly requested, but XPU is not available."
            )
        index = int(xpu_match.group(1) or 0)
        count = xpu.device_count()
        if index >= count:
            raise ValueError(
                f"XPU device index {index} is unavailable; PyTorch reports {count} device(s)."
            )
        return torch.device(f"xpu:{index}")

    raise ValueError(
        f"Invalid device '{device_name}'; expected auto, cpu, cuda[:N], "
        "gpu[N], mps, or xpu[:N]."
    )
