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
OBC_SOLVENT_DIELECTRIC = 78.5
OBC_SOLUTE_DIELECTRIC = 1.0
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
    solvent_dielectric: float = OBC_SOLVENT_DIELECTRIC
    solute_dielectric: float = OBC_SOLUTE_DIELECTRIC

    def __post_init__(self) -> None:
        """Own, freeze, and validate every finalized backend input."""
        if not isinstance(self.provenance, Mapping):
            raise TypeError("OBC-II provenance must be a mapping.")
        if not isinstance(self.nonpolar, str):
            raise TypeError("OBC-II nonpolar must be an explicit string.")

        for name in (
            "charges",
            "provider_parameters",
            "offset_radii_nm",
            "scaled_offset_radii_nm",
        ):
            try:
                owned = _owned_read_only(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"OBC-II {name} must be a numeric array.") from exc
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "nonpolar", self.nonpolar.lower())
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))
        self._validate()

    def _validate(self) -> None:
        """Apply the single semantic boundary for finalized OBC-II inputs."""
        count = self.charges.size
        if self.charges.ndim != 1 or count == 0:
            raise ValueError(
                "OBC-II requires a non-empty one-dimensional charge array."
            )
        if self.provider_parameters.shape != (count, 2):
            raise ValueError(
                "OBC-II radius provider must return one radius and screening "
                "factor per charge."
            )
        if self.offset_radii_nm.shape != (count,) or (
            self.scaled_offset_radii_nm.shape != (count,)
        ):
            raise ValueError("OBC-II finalized radius arrays must match the charges.")
        arrays = (
            self.charges,
            self.provider_parameters,
            self.offset_radii_nm,
            self.scaled_offset_radii_nm,
        )
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("OBC-II charges and radius parameters must be finite.")
        if np.any(self.provider_parameters[:, 0] <= OBC_OFFSET_NM):
            raise ValueError(
                "OBC-II intrinsic radii must be positive and greater than the "
                f"{OBC_OFFSET_NM:g} nm offset."
            )
        if np.any(self.provider_parameters[:, 1] <= 0.0):
            raise ValueError("OBC-II screening factors must be positive.")

        expected_offset = self.provider_parameters[:, 0] - OBC_OFFSET_NM
        expected_scaled = self.provider_parameters[:, 1] * expected_offset
        if not np.allclose(
            self.offset_radii_nm, expected_offset, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("OBC-II offset radii do not match provider radii.")
        if not np.allclose(
            self.scaled_offset_radii_nm, expected_scaled, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                "OBC-II scaled offset radii do not match screening factors."
            )
        if self.nonpolar not in SUPPORTED_NONPOLAR:
            raise ValueError("OBC-II Torch nonpolar must be 'ace' or 'none'.")
        for name in ("solvent_dielectric", "solute_dielectric"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"OBC-II {name} must be finite and positive.")
        if (
            self.solvent_dielectric != OBC_SOLVENT_DIELECTRIC
            or self.solute_dielectric != OBC_SOLUTE_DIELECTRIC
        ):
            raise ValueError(
                "OBC-II uses the verified dielectric profile "
                f"{OBC_SOLVENT_DIELECTRIC:g}/{OBC_SOLUTE_DIELECTRIC:g}."
            )


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
    if not isinstance(nonpolar, str):
        raise TypeError("OBC-II nonpolar must be an explicit string.")
    nonpolar = nonpolar.lower()
    charge_values = np.asarray(charges, dtype=np.float64)
    provider = np.asarray(radius_result.provider_parameters, dtype=np.float64)

    try:
        offset_radii = provider[:, 0] - OBC_OFFSET_NM
        scaled_offset_radii = provider[:, 1] * offset_radii
    except IndexError as exc:
        raise ValueError(
            "OBC-II radius provider must return radius/screening pairs."
        ) from exc
    provenance = {
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
    return OBC2Parameters(
        charges=charge_values,
        provider_parameters=provider,
        offset_radii_nm=offset_radii,
        scaled_offset_radii_nm=scaled_offset_radii,
        nonpolar=nonpolar,
        provenance=provenance,
    )
