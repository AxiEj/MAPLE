"""Closed numerical profiles for the three-site Gaussian-CHA R6 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from .torch_continuum_chagb_domain import CertifiedLocalPatchScope


GAUSSIAN_CHA_R6_V1_PROFILE_ID = "gaussian-cha-r6-v1"
GAUSSIAN_CHA_R6_V2_PROFILE_ID = (
    "gaussian-cha-r6-derivative-v2-numerical-profile-" "20261001.2-direct-complement"
)


@dataclass(frozen=True)
class R6BackendDiagnostics:
    profile_id: str
    backend_function: str
    order: int


@dataclass(frozen=True)
class R6BackendEvaluation:
    inverse_born_per_angstrom: Tensor
    diagnostics: R6BackendDiagnostics


@dataclass(frozen=True)
class GaussianChaNumericalProfile:
    """One of exactly two admitted numerical representations."""

    profile_id: str

    def __post_init__(self) -> None:
        if self.profile_id not in {
            GAUSSIAN_CHA_R6_V1_PROFILE_ID,
            GAUSSIAN_CHA_R6_V2_PROFILE_ID,
        }:
            raise ValueError(
                f"Unknown Gaussian-CHA numerical profile: {self.profile_id!r}"
            )

    def evaluate_inverse_born(
        self,
        positions_angstrom: Tensor,
        intrinsic_radii_angstrom: Tensor,
        domain: CertifiedLocalPatchScope,
        order: int,
    ) -> R6BackendEvaluation:
        if self.profile_id == GAUSSIAN_CHA_R6_V1_PROFILE_ID:
            from .torch_continuum_chagb import _r6_inverse_born

            inverse_born = _r6_inverse_born(
                positions_angstrom, intrinsic_radii_angstrom, domain, order
            )
            backend = "torch_continuum_chagb._r6_inverse_born"
        elif self.profile_id == GAUSSIAN_CHA_R6_V2_PROFILE_ID:
            from .torch_continuum_r6_derivative_v2 import _r6_inverse_born_v2

            inverse_born = _r6_inverse_born_v2(
                positions_angstrom, intrinsic_radii_angstrom, domain, order
            )
            backend = "torch_continuum_r6_derivative_v2._r6_inverse_born_v2"
        else:  # Construction is closed, but retain a fail-closed invariant.
            raise ValueError(
                f"Unknown Gaussian-CHA numerical profile: {self.profile_id!r}"
            )
        return R6BackendEvaluation(
            inverse_born_per_angstrom=inverse_born,
            diagnostics=R6BackendDiagnostics(
                profile_id=self.profile_id,
                backend_function=backend,
                order=order,
            ),
        )


_V1_PROFILE = GaussianChaNumericalProfile(GAUSSIAN_CHA_R6_V1_PROFILE_ID)
_V2_PROFILE = GaussianChaNumericalProfile(GAUSSIAN_CHA_R6_V2_PROFILE_ID)


def resolve_gaussian_cha_profile(profile_id: str) -> GaussianChaNumericalProfile:
    """Resolve an exact closed profile ID before importing either backend."""

    if not isinstance(profile_id, str):
        raise TypeError("Gaussian-CHA numerical profile ID must be a string.")
    if profile_id == GAUSSIAN_CHA_R6_V1_PROFILE_ID:
        return _V1_PROFILE
    if profile_id == GAUSSIAN_CHA_R6_V2_PROFILE_ID:
        return _V2_PROFILE
    raise ValueError(f"Unknown Gaussian-CHA numerical profile: {profile_id!r}")
