"""Fail-closed domain certificate for a single covering dispersion SAS ball."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .torch_continuum_dispersion import DISPERSION_PROBE_ANGSTROM

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class SingleCoverSASCertificate:
    """Proof record for one sphere strictly covering every other SAS ball."""

    status: str
    reason: str
    covering_index: int
    sas_radii_angstrom: Tensor
    containment_margins_angstrom: Tensor
    minimum_margin_angstrom: Tensor
    dtype: str
    device: str


@dataclass(frozen=True)
class SingleCoverSASRejection:
    """Typed rejection returned when the exact specialization is inadmissible."""

    status: str
    reason: str
    detail: str


def _reject(reason: str, detail: str) -> SingleCoverSASRejection:
    return SingleCoverSASRejection(status="rejected", reason=reason, detail=detail)


def certify_single_covering_dispersion_sas(
    positions_angstrom: Tensor,
    lj_rmin_angstrom: Tensor,
) -> SingleCoverSASCertificate | SingleCoverSASRejection:
    """Certify a unique strict cover using the live dispersion SAS geometry.

    The open-domain guard is ``64*eps*max(1, R_cover, distance + R_other)``.
    The function never accepts a supplied or cached certificate.
    """
    import torch

    if (
        not isinstance(positions_angstrom, torch.Tensor)
        or positions_angstrom.dtype != torch.float64
        or positions_angstrom.device.type != "cpu"
        or positions_angstrom.ndim != 2
        or positions_angstrom.shape[1:] != (3,)
        or len(positions_angstrom) == 0
    ):
        return _reject(
            "INVALID_POSITIONS", "positions must be nonempty CPU float64 [N,3]"
        )
    if (
        not isinstance(lj_rmin_angstrom, torch.Tensor)
        or lj_rmin_angstrom.dtype != torch.float64
        or lj_rmin_angstrom.device != positions_angstrom.device
        or lj_rmin_angstrom.shape != (len(positions_angstrom),)
    ):
        return _reject("INVALID_RMIN", "rmin must be same-device CPU float64 [N]")
    if not bool(
        torch.isfinite(positions_angstrom).all()
        and torch.isfinite(lj_rmin_angstrom).all()
    ):
        return _reject("NONFINITE_INPUT", "positions and rmin must be finite")
    if not bool((lj_rmin_angstrom > 0.0).all()):
        return _reject("NONPOSITIVE_RMIN", "every rmin value must be positive")

    sas_radii = lj_rmin_angstrom + DISPERSION_PROBE_ANGSTROM
    epsilon = torch.finfo(torch.float64).eps
    candidates: list[tuple[int, Tensor]] = []
    guard_limited = False
    for cover_index in range(len(positions_angstrom)):
        other_indices = [
            index for index in range(len(positions_angstrom)) if index != cover_index
        ]
        if not other_indices:
            empty = sas_radii.new_empty((0,))
            candidates.append((cover_index, empty))
            continue
        displacement = (
            positions_angstrom[other_indices] - positions_angstrom[cover_index]
        )
        distances = torch.linalg.vector_norm(displacement, dim=1)
        extents = distances + sas_radii[other_indices]
        margins = sas_radii[cover_index] - extents
        scales = torch.maximum(
            torch.ones_like(extents),
            torch.maximum(sas_radii[cover_index].expand_as(extents), extents),
        )
        guards = 64.0 * epsilon * scales
        if bool((margins > guards).all()):
            candidates.append((cover_index, margins))
        elif bool((margins >= -guards).all()):
            guard_limited = True

    if len(candidates) == 0:
        reason = "CONTAINMENT_GUARD" if guard_limited else "NO_UNIQUE_COVER"
        return _reject(
            reason, "no sphere strictly covers all other dispersion SAS balls"
        )
    if len(candidates) != 1:
        return _reject(
            "AMBIGUOUS_COVER",
            "more than one sphere satisfies the strict cover predicate",
        )

    covering_index, margins = candidates[0]
    minimum_margin = (
        margins.min()
        if len(margins)
        else torch.tensor(float("inf"), dtype=torch.float64, device="cpu")
    )
    return SingleCoverSASCertificate(
        status="certified",
        reason="UNIQUE_STRICT_COVER",
        covering_index=covering_index,
        sas_radii_angstrom=sas_radii,
        containment_margins_angstrom=margins,
        minimum_margin_angstrom=minimum_margin,
        dtype=str(positions_angstrom.dtype),
        device=str(positions_angstrom.device),
    )
