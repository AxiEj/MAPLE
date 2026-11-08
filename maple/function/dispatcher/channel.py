"""Closed request classification and preflight for the common Dispatcher path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, cast

from maple.function.calculator.route2 import (
    is_pure_mace_polar_workflow_calculator,
)
from maple.function.route2_smd_profiles import (
    FrequencyMethodRequest,
    Route2SMDProfileSpec,
    Route2WorkflowCapability,
    normalize_frequency_method_request,
    route2_workflow_capability_for_params,
)

from .pure_nonmd_status import (
    calculator_provenance,
    is_pure_nonmd_v2,
    is_pure_torch_calculator,
    is_pure_torch_v4,
)
from .sp.result import normalize_v4_sp_request


class _DispatchImage(Protocol):
    calc: object
    info: dict[str, object]
    constraints: object

    def get_chemical_symbols(self) -> list[str]: ...

    def get_masses(self) -> Sequence[float]: ...


class _Route2Calculator(Protocol):
    profile_spec: Route2SMDProfileSpec

    def _validate_atoms(self, atoms: _DispatchImage) -> tuple[object, ...]: ...


class CanonicalFamily(Enum):
    ORDINARY = "ordinary"
    DDX_V1 = "ddx_v1"
    DDX_V2 = "ddx_v2"
    TORCH_V3 = "torch_v3"
    TORCH_V4 = "torch_v4"


class DispatchTask(Enum):
    SP = "sp"
    OPT = "opt"
    SCAN = "scan"
    FREQ = "freq"
    TS = "ts"
    IRC = "irc"
    MD = "md"


class SPShape(Enum):
    LEGACY = "legacy"
    SINGLE = "single"
    BATCH = "batch"


class ThresholdPolicy(Enum):
    COMMON = "common"
    V4_EXPLICIT_OVER_LEVEL = "v4_explicit_over_level"


class PreflightMode(Enum):
    LEGACY = "legacy"
    DDX_V2 = "ddx_v2"
    TORCH_V3 = "torch_v3"
    TORCH_V4 = "torch_v4"


class ReportingMode(Enum):
    NONE = "none"
    NONMD_V2 = "nonmd_v2"
    WORKFLOW = "workflow"


@dataclass(frozen=True, slots=True)
class FamilyPolicy:
    family: CanonicalFamily
    preflight: PreflightMode
    reporting: ReportingMode


@dataclass(frozen=True, slots=True)
class NormalizedDispatchRequest:
    task: DispatchTask
    method: str
    frequency_request: FrequencyMethodRequest | None


@dataclass(frozen=True, slots=True)
class ResolvedDispatchRequest:
    family_policy: FamilyPolicy
    normalized_request: NormalizedDispatchRequest
    v4_capability: Route2WorkflowCapability | None
    sp_shape: SPShape
    threshold_policy: ThresholdPolicy


_FAMILY_POLICIES = MappingProxyType({
    CanonicalFamily.ORDINARY: FamilyPolicy(
        CanonicalFamily.ORDINARY, PreflightMode.LEGACY, ReportingMode.NONE
    ),
    CanonicalFamily.DDX_V1: FamilyPolicy(
        CanonicalFamily.DDX_V1, PreflightMode.LEGACY, ReportingMode.NONE
    ),
    CanonicalFamily.DDX_V2: FamilyPolicy(
        CanonicalFamily.DDX_V2, PreflightMode.DDX_V2, ReportingMode.NONMD_V2
    ),
    CanonicalFamily.TORCH_V3: FamilyPolicy(
        CanonicalFamily.TORCH_V3, PreflightMode.TORCH_V3, ReportingMode.NONE
    ),
    CanonicalFamily.TORCH_V4: FamilyPolicy(
        CanonicalFamily.TORCH_V4, PreflightMode.TORCH_V4, ReportingMode.WORKFLOW
    ),
})


def _images(atoms: object) -> tuple[_DispatchImage, ...]:
    multiatoms = getattr(atoms, "multiatoms", None)
    if multiatoms is not None:
        return cast(tuple[_DispatchImage, ...], tuple(multiatoms))
    if isinstance(atoms, (list, tuple)):
        return cast(tuple[_DispatchImage, ...], tuple(atoms))
    return (cast(_DispatchImage, atoms),)


def _all_workflow_ddx(images: tuple[_DispatchImage, ...]) -> bool:
    return bool(images) and all(
        is_pure_mace_polar_workflow_calculator(
            getattr(image, "calc", None)
        )
        for image in images
    )


def resolve_dispatch_request(
    atoms: object,
    jobtype: str,
    params: Mapping[str, object],
) -> ResolvedDispatchRequest:
    if "channel" in params:
        raise ValueError("Dispatcher channel selection is internal.")
    try:
        task = DispatchTask(str(jobtype).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported dispatch task: {jobtype}") from exc

    images = _images(atoms)
    v2_flags = tuple(is_pure_nonmd_v2(image) for image in images)
    torch_flags = tuple(is_pure_torch_calculator(image) for image in images)
    v4_flags = tuple(is_pure_torch_v4(image) for image in images)
    if any(v2_flags) and not all(v2_flags):
        raise ValueError(
            "pure non-MD v2 jobs require every image to use a canonical "
            "pure non-MD calculator."
        )
    if any(torch_flags) and not all(torch_flags):
        raise ValueError(
            "pure Torch jobs require every image to use the canonical "
            "analytic Torch calculator."
        )

    if v4_flags and all(v4_flags):
        family = CanonicalFamily.TORCH_V4
    elif torch_flags and all(torch_flags):
        family = CanonicalFamily.TORCH_V3
    elif v2_flags and all(v2_flags):
        family = CanonicalFamily.DDX_V2
    elif _all_workflow_ddx(images):
        family = CanonicalFamily.DDX_V1
    else:
        family = CanonicalFamily.ORDINARY

    if task is DispatchTask.FREQ and "deprecated_alias" in params:
        raise ValueError(
            "deprecated_alias is derived internally and cannot be supplied."
        )
    frequency_request = (
        normalize_frequency_method_request(
            params.get("method"), pure_torch_v4=family is CanonicalFamily.TORCH_V4
        )
        if task is DispatchTask.FREQ
        else None
    )
    method = str(
        params.get("ensemble")
        if task is DispatchTask.MD
        else (
            frequency_request.canonical_method
            if frequency_request is not None
            else params.get("method") or task.value
        )
    )

    capability = None
    sp_shape = SPShape.LEGACY
    if family is CanonicalFamily.TORCH_V4:
        calculator = getattr(images[0], "calc", None)
        raw_calculator = getattr(calculator, "raw_calculator", calculator)
        if raw_calculator is None:
            raise ValueError("Pure Torch V4 requires a canonical calculator.")
        route2_calculator = cast(_Route2Calculator, raw_calculator)
        if task is DispatchTask.SP:
            sp_request = normalize_v4_sp_request(atoms, params)
            sp_shape = SPShape.BATCH if sp_request.returns_batch else SPShape.SINGLE
        capability = route2_workflow_capability_for_params(
            route2_calculator.profile_spec, task.value, params
        )

    threshold_policy = (
        ThresholdPolicy.V4_EXPLICIT_OVER_LEVEL
        if family is CanonicalFamily.TORCH_V4 and task is DispatchTask.IRC
        else ThresholdPolicy.COMMON
    )
    return ResolvedDispatchRequest(
        family_policy=_FAMILY_POLICIES[family],
        normalized_request=NormalizedDispatchRequest(
            task=task,
            method=method,
            frequency_request=frequency_request,
        ),
        v4_capability=capability,
        sp_shape=sp_shape,
        threshold_policy=threshold_policy,
    )


def _validate_homogeneous_route2_images(
    images: tuple[_DispatchImage, ...]
) -> None:
    physical_identities = {
        (
            tuple(image.get_chemical_symbols()),
            tuple(float(value) for value in image.get_masses()),
            image.info.get("charge"),
            image.info.get("mult"),
        )
        for image in images
    }
    if len(physical_identities) != 1:
        raise ValueError(
            "Route 2 workflow images must share symbols, masses, charge, and multiplicity."
        )
    configurations = set()
    for image in images:
        calculator = getattr(image, "calc", None)
        calculator = getattr(calculator, "raw_calculator", calculator)
        if calculator is None:
            raise ValueError("Route 2 workflow image has no calculator.")
        validation = cast(_Route2Calculator, calculator)._validate_atoms(image)
        configurations.add(validation[1])
    if len(configurations) != 1:
        raise ValueError("Route 2 workflow images must share one PES configuration.")
    identities = {
        tuple(calculator_provenance(image).values()) for image in images
    }
    if len(identities) != 1:
        raise ValueError(
            "Route 2 workflow images must share profile, scalar, and device."
        )


def preflight_dispatch_request(
    request: ResolvedDispatchRequest,
    atoms: object,
    params: Mapping[str, object],
) -> None:
    images = _images(atoms)
    task = request.normalized_request.task
    mode = request.family_policy.preflight
    if mode is PreflightMode.LEGACY:
        return
    if mode is PreflightMode.TORCH_V3:
        if len(images) != 1:
            raise ValueError("pure Torch v3 does not support path or multi-image jobs.")
        if task not in {DispatchTask.SP, DispatchTask.OPT, DispatchTask.FREQ}:
            raise ValueError(
                "pure Torch v3 supports only SP, OPT, and FREQ; "
                "TS, IRC, MD, and path workflows remain closed."
            )
        if task is DispatchTask.FREQ and str(params.get("method", "mw")).lower() != "mw":
            raise ValueError("pure Torch v3 frequency supports only method=mw.")
        return

    _validate_homogeneous_route2_images(images)
    if mode is PreflightMode.DDX_V2:
        if task is DispatchTask.MD:
            raise ValueError("MD is not supported by pure non-MD v2 profiles.")
        if task not in {
            DispatchTask.SP, DispatchTask.OPT, DispatchTask.SCAN,
            DispatchTask.FREQ, DispatchTask.TS, DispatchTask.IRC,
        }:
            raise ValueError(f"Unsupported pure non-MD v2 task: {task.value}")
        if (
            task is DispatchTask.SCAN
            and str(params.get("mode", "relaxed")).lower() == "relaxed"
            and str(params.get("method", "lbfgs")).lower() == "rfo"
        ):
            raise ValueError(
                "relaxed RFO scan is unsupported because a constrained "
                "Hessian projection is not implemented."
            )
        if task is DispatchTask.FREQ and str(params.get("method", "mw")).lower() != "mw":
            raise ValueError(
                "pure non-MD v2 frequency supports only mass-weighted method=mw."
            )
        return

    capability = request.v4_capability
    if capability is None or capability.execution_class == "closed":
        raise ValueError("Pure Torch V4 workflow capability is closed.")
    if len(images) != 1 and capability.image_policy != "multi":
        raise ValueError("Pure Torch V4 task requires one molecular structure.")
    constrained = any(
        tuple(getattr(image, "constraints", ()) or ()) for image in images
    )
    if constrained and not capability.constraints_allowed:
        raise ValueError(
            "Pure Torch V4 Hessian/HVP workflows do not support atom constraints."
        )


__all__ = [
    "CanonicalFamily",
    "DispatchTask",
    "FamilyPolicy",
    "NormalizedDispatchRequest",
    "PreflightMode",
    "ReportingMode",
    "ResolvedDispatchRequest",
    "SPShape",
    "ThresholdPolicy",
    "preflight_dispatch_request",
    "resolve_dispatch_request",
]
