"""Versioned boundary around the private OpenMM APIs used by Route 1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path


VERIFIED_OPENMM_VERSION = "8.5.2"
VERIFIED_CUSTOMGBFORCES_SHA256 = (
    "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
)
VERIFIED_OBC2_EXPRESSION_PROFILE = "openmm-8.5.2-obc2-ace-v1"
REQUIRED_CUSTOM_GB_CLASSES = frozenset(
    {
        "GBSAHCTForce",
        "GBSAOBC1Force",
        "GBSAOBC2Force",
        "GBSAGBnForce",
        "GBSAGBn2Force",
    }
)
REQUIRED_LCPO_ATTRIBUTES = frozenset(
    {
        "LCPO_PARAMETERS",
        "addLCPOForce",
        "getLCPOParamsTopology",
    }
)


def openmm_version() -> str:
    try:
        return importlib.metadata.version("openmm")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def require_verified_openmm() -> str:
    version = openmm_version()
    if version != VERIFIED_OPENMM_VERSION:
        raise ImportError(
            "Route 1 uses private OpenMM GB/LCPO modules and is verified only "
            f"against OpenMM {VERIFIED_OPENMM_VERSION}; observed {version!r}. "
            "Install the pinned 'maple[implicit-gb]' dependency and rerun the "
            "provider parity suite before admitting another version."
        )
    return version


def customgbforces_module():
    require_verified_openmm()
    try:
        from openmm.app.internal import customgbforces
    except ImportError as exc:
        raise ImportError(
            "The verified OpenMM installation does not expose the required "
            "private customgbforces module."
        ) from exc
    missing = sorted(
        name for name in REQUIRED_CUSTOM_GB_CLASSES if not hasattr(customgbforces, name)
    )
    if missing:
        raise ImportError(
            "The verified OpenMM private customgbforces contract is incomplete: "
            + ", ".join(missing)
            + "."
        )
    return customgbforces


def require_verified_obc2_expression_source() -> dict[str, str]:
    """Pin the private source transcribed by the Torch OBC-II backend.

    Ordinary OpenMM execution remains version-gated as before.  The stronger
    content check is performed only when the independent Torch derivative
    backend is requested, because that backend promises expression identity
    with this exact upstream source file.
    """
    customgbforces = customgbforces_module()
    source = inspect.getsourcefile(customgbforces)
    if source is None:
        raise ImportError(
            "Cannot locate OpenMM customgbforces.py for Torch OBC-II parity."
        )
    path = Path(source).resolve()
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != VERIFIED_CUSTOMGBFORCES_SHA256:
        raise ImportError(
            "Torch OBC-II is verified only against customgbforces.py SHA256 "
            f"{VERIFIED_CUSTOMGBFORCES_SHA256}; observed {observed}."
        )
    return {
        "openmm_version": require_verified_openmm(),
        "source_path": str(path),
        "source_sha256": observed,
        "expression_profile": VERIFIED_OBC2_EXPRESSION_PROFILE,
    }


def lcpo_module():
    require_verified_openmm()
    try:
        from openmm.app.internal import lcpo
    except ImportError as exc:
        raise ImportError(
            "The verified OpenMM installation does not expose the required "
            "private LCPO module."
        ) from exc
    missing = sorted(
        name for name in REQUIRED_LCPO_ATTRIBUTES if not hasattr(lcpo, name)
    )
    if missing:
        raise ImportError(
            "The verified OpenMM private LCPO contract is incomplete: "
            + ", ".join(missing)
            + "."
        )
    return lcpo


def private_api_provenance() -> dict[str, object]:
    return {
        "adapter": "maple-openmm-private-api-compat",
        "verified_openmm_version": VERIFIED_OPENMM_VERSION,
        "observed_openmm_version": require_verified_openmm(),
        "private_modules": [
            "openmm.app.internal.customgbforces",
            "openmm.app.internal.lcpo",
        ],
        "unverified_versions_fail_closed": True,
    }
