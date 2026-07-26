"""Versioned boundary around the private OpenMM APIs used by Route 1."""

from __future__ import annotations

import importlib.metadata


VERIFIED_OPENMM_VERSION = "8.5.2"
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
