"""OpenMM execution selection, independent of MLIP and solvent physics.

An MLIP device is only a preference for ``platform=auto``. Explicit solvent
platforms/indices override it. Native context failures never trigger a retry
on CPU. This module does not install plugins or change process-wide defaults.
"""

from __future__ import annotations

import re
import sys
from typing import Any

CPU_PROPERTIES = {"Threads": "1", "DeterministicForces": "true"}
GPU_PLATFORMS = frozenset({"CUDA", "HIP", "OpenCL"})


def _indices(value: Any, name: str, *, multiple: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{name} must be a nonnegative integer device index")
    text = str(value).strip()
    pattern = r"\d+(?:\s*,\s*\d+)*" if multiple else r"\d+"
    if re.fullmatch(pattern, text) is None:
        raise ValueError(f"Invalid {name}: {value!r}")
    values = [str(int(part.strip())) for part in text.split(",")]
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate device indices")
    return ",".join(values)


def resolve_openmm_platform(
    platform_name: str,
    *,
    model_device: Any = None,
    precision: str | None = None,
    device_index: Any = None,
    opencl_platform_index: Any = None,
) -> tuple[Any, dict[str, str], dict[str, Any]]:
    """Resolve a native platform and only its supported context properties.

    CUDA uses the CUDA MLIP ordinal; PyTorch ROCm builds use the same ``cuda``
    spelling but select HIP. OpenCL indices refer to its own device namespace
    and are not inferred from CUDA/MPS/XPU ordinals. GPU default precision is
    double for stable solvent derivatives; explicit single/mixed is retained
    and reported, not silently promoted midway through a calculation.
    """
    from openmm import Platform

    available = {
        Platform.getPlatform(i).getName().lower(): Platform.getPlatform(i).getName()
        for i in range(Platform.getNumPlatforms())
    }
    requested = str(platform_name).strip()
    key = requested.lower()
    model = str(model_device or "cpu").lower()
    preferred = "CPU"
    if model.startswith(("cuda", "gpu")):
        torch_version = getattr(sys.modules.get("torch"), "version", None)
        preferred = "HIP" if getattr(torch_version, "hip", None) else "CUDA"
    elif model.startswith(("mps", "xpu")):
        # These runtimes do not share a device namespace with OpenCL. Let the
        # caller select OpenCL and its platform/device explicitly instead.
        preferred = "unmapped_accelerator"

    reason = "explicit_platform"
    if key == "auto":
        if preferred.lower() in available:
            key = preferred.lower()
            reason = "model_device_preference"
        else:
            key = "cpu"
            reason = "requested_accelerator_unavailable"
    if key not in available:
        raise ValueError(
            f"OpenMM platform {platform_name!r} is unavailable; available: "
            + ", ".join(available.values())
            + ". Install a matching native plugin/driver or select an available platform explicitly."
        )
    selected = available[key]
    index_source = "explicit" if device_index is not None else "native_default"
    platform = Platform.getPlatformByName(selected)
    properties: dict[str, str] = {}
    requested_precision = None if precision is None else str(precision).strip().lower()
    if requested_precision not in {None, "single", "mixed", "double"}:
        raise ValueError("OpenMM precision must be single, mixed or double")
    if selected in GPU_PLATFORMS:
        properties["Precision"] = requested_precision or "double"
        if (
            device_index is None
            and selected in {"CUDA", "HIP"}
            and selected == preferred
            and model.startswith(("cuda", "gpu"))
        ):
            match = re.fullmatch(r"(?:cuda|gpu)(?::?(\d+))?", model)
            if match is not None:
                device_index = match.group(1) or "0"
                index_source = "model_device"
        if device_index is not None:
            properties["DeviceIndex"] = _indices(
                device_index, "device_index", multiple=True
            )
        if selected in {"CUDA", "HIP"}:
            properties["DeterministicForces"] = "true"
    elif selected == "CPU":
        if requested_precision is not None:
            raise ValueError(
                "OpenMM CPU has no selectable precision; use Reference or a GPU platform for double"
            )
        properties.update(CPU_PROPERTIES)
    elif requested_precision not in {None, "double"}:
        raise ValueError("OpenMM Reference has fixed double precision")

    if device_index is not None and selected not in GPU_PLATFORMS:
        raise ValueError("device_index applies only to CUDA, HIP or OpenCL")
    if opencl_platform_index is not None:
        if selected != "OpenCL":
            raise ValueError("opencl_platform_index requires platform=OpenCL")
        properties["OpenCLPlatformIndex"] = _indices(
            opencl_platform_index, "opencl_platform_index"
        )
    unsupported = set(properties) - set(platform.getPropertyNames())
    if unsupported:
        raise ValueError(
            f"OpenMM {selected} does not expose requested properties: {sorted(unsupported)}"
        )
    record = {
        "requested_platform": requested,
        "resolved_platform": selected,
        "model_device_preference": model,
        "requested_precision": requested_precision,
        "selection_reason": reason,
        "gpu_requested_but_unavailable": reason == "requested_accelerator_unavailable",
        "precision_policy": properties.get(
            "Precision", "double" if selected == "Reference" else "CPU-native"
        ),
        "available_platforms": list(available.values()),
        "device_index_source": index_source
        if selected in GPU_PLATFORMS
        else "not_applicable",
    }
    return platform, properties, record
