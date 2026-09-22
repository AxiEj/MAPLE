"""Immutable parameters shared by equivalent OBC-II execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .openmm_compat import (
    VERIFIED_CUSTOMGBFORCES_SHA256,
    VERIFIED_OBC2_EXPRESSION_PROFILE,
    VERIFIED_OPENMM_VERSION,
)

OBC_OFFSET_NM = 0.009
SUPPORTED_NONPOLAR = frozenset({"ace", "none"})


def _owned_read_only(values: np.ndarray) -> np.ndarray:
    owned = np.array(values, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _deep_freeze(value: Any) -> Any:
    """Return an immutable, ownership-independent provenance value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return _owned_read_only(value)
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class OBC2Parameters:
    """Fully finalized OBC-II particle parameters.

    Arrays are owned copies and read-only.  ``offset_radii_nm`` and
    ``scaled_offset_radii_nm`` are the values actually installed in OpenMM
    after :class:`CustomAmberGBForceBase` applies its offset and screening
    transformations.
    """

    charges: np.ndarray
    provider_parameters: np.ndarray
    offset_radii_nm: np.ndarray
    scaled_offset_radii_nm: np.ndarray
    nonpolar: str
    provenance: Mapping[str, Any]
    solvent_dielectric: float = 78.5
    solute_dielectric: float = 1.0


def build_obc2_parameters(
    charges,
    radius_result,
    *,
    nonpolar: str,
) -> OBC2Parameters:
    """Finalize radius-provider output for both OpenMM and Torch OBC-II.

    The transformation is copied from the pinned OpenMM 8.5.2
    ``CustomAmberGBForceBase.addParticle`` contract: subtract 0.009 nm from
    each intrinsic radius, then multiply the screening factor by that offset
    radius.
    """
    nonpolar = str(nonpolar).lower()
    if nonpolar not in SUPPORTED_NONPOLAR:
        raise ValueError("OBC-II Torch nonpolar must be 'ace' or 'none'.")

    charge_values = np.asarray(charges, dtype=np.float64)
    provider = np.asarray(radius_result.provider_parameters, dtype=np.float64)
    if charge_values.ndim != 1 or charge_values.size == 0:
        raise ValueError("OBC-II requires a non-empty one-dimensional charge array.")
    if provider.shape != (charge_values.size, 2):
        raise ValueError(
            "OBC-II radius provider must return one radius and screening factor "
            "per charge."
        )
    if not np.isfinite(charge_values).all() or not np.isfinite(provider).all():
        raise ValueError("OBC-II charges and radius parameters must be finite.")
    if np.any(provider[:, 0] <= OBC_OFFSET_NM):
        raise ValueError(
            "OBC-II intrinsic radii must be positive and greater than the "
            f"{OBC_OFFSET_NM:g} nm offset."
        )
    if np.any(provider[:, 1] <= 0.0):
        raise ValueError("OBC-II screening factors must be positive.")

    offset_radii = provider[:, 0] - OBC_OFFSET_NM
    scaled_offset_radii = provider[:, 1] * offset_radii
    provenance = _deep_freeze(
        {
            "category": "obc2-parameters",
            "radius_provider": radius_result.provenance,
            "radius_profile": radius_result.profile,
            "nonpolar": nonpolar,
            "offset_nm": OBC_OFFSET_NM,
            "source": (
                f"OpenMM {VERIFIED_OPENMM_VERSION} "
                "openmm.app.internal.customgbforces; "
                f"SHA256 {VERIFIED_CUSTOMGBFORCES_SHA256}"
            ),
            "expression_profile": VERIFIED_OBC2_EXPRESSION_PROFILE,
        }
    )
    return OBC2Parameters(
        charges=_owned_read_only(charge_values),
        provider_parameters=_owned_read_only(provider),
        offset_radii_nm=_owned_read_only(offset_radii),
        scaled_offset_radii_nm=_owned_read_only(scaled_offset_radii),
        nonpolar=nonpolar,
        provenance=provenance,
    )
