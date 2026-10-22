"""Stable FP32 partition of unity for the ddPCM cavity surface.

The ddX switch is a quintic smoothstep.  In float32, evaluating the switch
close to one and then forming ``1 - switch`` loses the small exposed weight.
This module evaluates the small member of the complementary pair directly
and classifies support from radial ratios, rather than rounded switch values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

FP32_PARTITION_POLICY = "ddx-v0.8.0-fp32-complementary-quintic-partition-v1"


@dataclass(frozen=True, slots=True)
class FP32PartitionDiagnostics:
    contract: str
    support_sha256: str
    plateau_one_entry_count: int
    interior_entry_count: int
    plateau_zero_entry_count: int
    active_node_count: int
    buried_node_count: int
    buried_plateau_node_count: int
    transition_node_count: int
    single_interior_node_count: int
    multiple_interior_node_count: int
    minimum_switch_margin: float | None
    minimum_active_f_margin: float | None
    warnings: tuple[str, ...]
    hard_failures: tuple[str, ...]


def _hash_support(*tensors: Any) -> str:
    digest = hashlib.sha256(FP32_PARTITION_POLICY.encode())
    for tensor in tensors:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _smoothstep5(value: Any) -> Any:
    return value**3 * (value * (6.0 * value - 15.0) + 10.0)


def _support_from_centered_ratio(
    ratios: Any, owner_mask: Any, *, eta: float
) -> tuple[Any, Any, Any, Any]:
    """Classify radial support from one centered FP32 coordinate."""

    q = (ratios - 1.0) / eta
    nonself = ~owner_mask
    plateau_one = (q <= -0.5) & nonself
    plateau_zero = (q >= 0.5) & nonself
    interior = ~(plateau_one | plateau_zero | owner_mask)
    return q, plateau_one, interior, plateau_zero


def stable_fp32_partition(
    ratios: Any, owner_mask: Any, *, eta: float
) -> tuple[Any, Any, Any]:
    """Return ``(chi, f, u)`` using a cancellation-safe quintic pair.

    Plateau/interior support is decided from ``ratios``.  For an interior
    switch, the polynomial closer to zero is evaluated directly and its
    partner is formed by one subtraction.  This is the same quintic function,
    not clipping or a tolerance-based change to the scalar.
    """

    torch = __import__("torch")
    if ratios.dtype is not torch.float32:
        raise TypeError("stable_fp32_partition requires float32 ratios.")
    q, plateau_one, interior, plateau_zero = _support_from_centered_ratio(
        ratios, owner_mask, eta=eta
    )
    raw_z = 0.5 - q
    raw_complement = 0.5 + q
    # Evaluate the polynomial only on its mathematical [0, 1] support.  The
    # value is selected by exact radial branch, not by numerical clipping.
    z = torch.where(
        interior,
        raw_z,
        torch.where(plateau_one, torch.ones_like(raw_z), torch.zeros_like(raw_z)),
    )
    complement_argument = torch.where(
        interior,
        raw_complement,
        torch.where(
            plateau_one,
            torch.zeros_like(raw_complement),
            torch.ones_like(raw_complement),
        ),
    )
    direct = _smoothstep5(z)
    complementary = _smoothstep5(complement_argument)
    low_half = z <= 0.5
    interior_chi = torch.where(low_half, direct, 1.0 - complementary)
    interior_complement = torch.where(low_half, 1.0 - direct, complementary)

    chi = torch.where(
        plateau_one,
        torch.ones_like(ratios),
        torch.where(interior, interior_chi, torch.zeros_like(ratios)),
    )
    fi = chi.sum(dim=-1)
    one_count = plateau_one.sum(dim=-1)
    interior_count = interior.sum(dim=-1)
    single_interior = (one_count == 0) & (interior_count == 1)
    no_cover = (one_count == 0) & (interior_count == 0)
    stable_complement = torch.where(
        interior, interior_complement, torch.zeros_like(ratios)
    ).sum(dim=-1)
    ui = torch.where(
        one_count > 0,
        torch.zeros_like(fi),
        torch.where(
            no_cover,
            torch.ones_like(fi),
            torch.where(
                single_interior,
                stable_complement,
                torch.clamp(1.0 - fi, min=0.0),
            ),
        ),
    )
    return chi, fi, ui


def fp32_partition_normalizer(
    ratios: Any, owner_mask: Any, fi: Any, *, eta: float
) -> Any:
    """Return the ddX overlap normalizer from structural radial support."""

    torch = __import__("torch")
    _, plateau_one, interior, _ = _support_from_centered_ratio(
        ratios, owner_mask, eta=eta
    )
    plateau_one_count = plateau_one.sum(dim=-1)
    interior_count = interior.sum(dim=-1)
    upper_branch = (plateau_one_count > 0) | ((interior_count > 1) & (fi >= 1.0))
    return torch.where(upper_branch, fi, torch.ones_like(fi))


def assess_fp32_partition(
    ratios: Any,
    owner_mask: Any,
    chi: Any,
    fi: Any,
    ui: Any,
    *,
    eta: float,
    margin: float,
) -> FP32PartitionDiagnostics:
    """Classify structural support and retain only genuine clamp ambiguity.

    Quintic endpoints are C2 and a single interior contribution has a proven
    lower branch with its directly evaluated complement.  A node with several
    interior contributions near ``f=1`` has no such one-sided proof and remains
    a hard failure.
    """

    torch = __import__("torch")
    q, plateau_one, interior, plateau_zero = _support_from_centered_ratio(
        ratios, owner_mask, eta=eta
    )
    nonself = ~owner_mask
    selected = nonself.expand_as(ratios)
    endpoint_distance = eta * torch.minimum((q + 0.5).abs(), (q - 0.5).abs())
    relevant_distance = endpoint_distance[selected]
    minimum_switch = (
        float(relevant_distance.min().detach().cpu())
        if relevant_distance.numel()
        else None
    )
    one_count = plateau_one.sum(dim=-1)
    interior_count = interior.sum(dim=-1)
    transition = interior_count > 0
    active_margins = (fi[transition] - 1.0).abs()
    minimum_active = (
        float(active_margins.min().detach().cpu()) if active_margins.numel() else None
    )
    near_one = (fi - 1.0).abs() <= margin
    ambiguous = (interior_count > 1) & near_one & (one_count == 0)

    warning_codes = []
    if minimum_switch is not None and minimum_switch <= margin:
        warning_codes.append("fp32_c2_switch_endpoint_margin")
    if bool(((interior_count == 1) & near_one & (one_count == 0)).any().detach().cpu()):
        warning_codes.append("fp32_single_interior_complementary_branch")
    if bool(((one_count > 0) & transition & near_one).any().detach().cpu()):
        warning_codes.append("fp32_plateau_one_buried_branch")
    failures = (
        ("topology_branch_margin",) if bool(ambiguous.any().detach().cpu()) else ()
    )
    support = torch.where(
        plateau_one,
        torch.full_like(ratios, 2, dtype=torch.int8),
        torch.where(
            interior,
            torch.ones_like(ratios, dtype=torch.int8),
            torch.zeros_like(ratios, dtype=torch.int8),
        ),
    )
    return FP32PartitionDiagnostics(
        contract=FP32_PARTITION_POLICY,
        support_sha256=_hash_support(support, ui > 0.0, fi > 1.0),
        plateau_one_entry_count=int(plateau_one.sum().detach().cpu()),
        interior_entry_count=int(interior.sum().detach().cpu()),
        plateau_zero_entry_count=int(plateau_zero.sum().detach().cpu()),
        active_node_count=int((ui > 0.0).sum().detach().cpu()),
        buried_node_count=int((ui == 0.0).sum().detach().cpu()),
        buried_plateau_node_count=int(((fi == 1.0) & ~transition).sum().detach().cpu()),
        transition_node_count=int(transition.sum().detach().cpu()),
        single_interior_node_count=int((interior_count == 1).sum().detach().cpu()),
        multiple_interior_node_count=int((interior_count > 1).sum().detach().cpu()),
        minimum_switch_margin=minimum_switch,
        minimum_active_f_margin=minimum_active,
        warnings=tuple(warning_codes),
        hard_failures=failures,
    )


__all__ = [
    "FP32_PARTITION_POLICY",
    "FP32PartitionDiagnostics",
    "assess_fp32_partition",
    "fp32_partition_normalizer",
    "stable_fp32_partition",
]
