"""Local one-sided branch proofs for the unchanged ddPCM scalar.

This response-only policy does not smooth or clip the exposure. It refines
the legacy certificate only where the actual radial support proves a side
of f=1. Genuine multi-interior ambiguity and all endpoint margins still fail.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

RESPONSE_TOPOLOGY_POLICY = "ddx-v0.8.0-response-one-sided-fixed-branch-v1"


@dataclass(frozen=True, slots=True)
class OneSidedNodeProof:
    owner: int
    lebedev_index: int
    f: float
    side: str
    proof: str
    plateau_one_count: int
    interior_count: int
    plateau_zero_count: int
    nonowner_radial_branches: tuple[int, ...]
    minimum_switch_margin: float


@dataclass(frozen=True, slots=True)
class ResponseTopologyCertificate:
    contract: str
    topology_sha256: str
    positions_sha256: str
    active_node_count: int
    buried_plateau_count: int
    transition_node_count: int
    minimum_switch_margin: float | None
    minimum_active_f_margin: float | None
    one_sided_nodes: tuple[OneSidedNodeProof, ...]


def _hash_arrays(label, arrays):
    digest = hashlib.sha256(label.encode())
    for tensor in arrays:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def certify_response_topology(positions, ratios, chi, fi, ui, *, eta, margin):
    """Certify actual switch support, never rounded switch-value categories.

    Codes are 0=plateau zero, 1=interior, 2=plateau one. Owner entries are
    excluded from proofs. A computed equality with any interior contribution
    is rejected rather than choosing a different Torch clamp derivative.
    """
    count = len(positions)
    nonself = ~torch.eye(count, dtype=torch.bool, device=fi.device)[:, None, :]
    selected = nonself.expand_as(ratios)
    if (
        not bool(torch.isfinite(ratios[selected]).all())
        or bool((ratios[selected] <= 0).any())
        or not bool(torch.isfinite(chi).all())
        or bool(((chi < 0) | (chi > 1)).any())
        or not bool(torch.isfinite(fi).all())
        or not torch.equal(fi, chi.sum(dim=-1))
        or not torch.equal(ui, torch.clamp(1 - fi, min=0))
    ):
        raise RuntimeError("Invalid radial values for ddPCM response topology proof.")
    lower, upper = 1 - eta / 2, 1 + eta / 2
    endpoint_distance = torch.minimum((ratios - lower).abs(), (ratios - upper).abs())
    distances = endpoint_distance[selected]
    minimum_switch = float(distances.min()) if distances.numel() else None
    if minimum_switch is not None and minimum_switch <= margin:
        raise RuntimeError(
            "ddPCM response switch endpoint is within the certified margin."
        )
    one = (ratios < lower) & nonself
    interior = (ratios > lower) & (ratios < upper) & nonself
    zero = (ratios > upper) & nonself
    if bool(((one & (chi != 1)) | (zero & (chi != 0))).any()):
        raise RuntimeError("ddPCM radial support and plateau values disagree.")
    n_one, n_interior = one.sum(-1), interior.sum(-1)
    transition = n_interior > 0
    near = (fi - 1).abs() <= margin
    if bool((near & transition & (fi == 1)).any()):
        raise RuntimeError(
            "ddPCM response computed equality has an active interior switch."
        )
    upper_proof = (n_one == 1) & (fi > 1)
    lower_proof = (n_one == 0) & (n_interior == 1) & (fi < 1)
    if bool((near & transition & ~(upper_proof | lower_proof)).any()):
        raise RuntimeError("ddPCM response has an unproved near-one branch crossing.")
    support = torch.where(one, 2, torch.where(interior, 1, 0)).to(torch.int8)
    proofs = []
    for i, node in (near & transition).nonzero().tolist():
        is_upper = bool(upper_proof[i, node])
        mask = selected[i, node]
        proofs.append(
            OneSidedNodeProof(
                owner=i,
                lebedev_index=node,
                f=float(fi[i, node]),
                side="upper" if is_upper else "lower",
                proof="stable-plateau-one" if is_upper else "single-interior",
                plateau_one_count=int(n_one[i, node]),
                interior_count=int(n_interior[i, node]),
                plateau_zero_count=int(zero[i, node].sum()),
                nonowner_radial_branches=tuple(int(v) for v in support[i, node, mask]),
                minimum_switch_margin=float(endpoint_distance[i, node, mask].min()),
            )
        )
    active_margins = (fi[transition] - 1).abs()
    return ResponseTopologyCertificate(
        contract=RESPONSE_TOPOLOGY_POLICY,
        topology_sha256=_hash_arrays(
            RESPONSE_TOPOLOGY_POLICY, (support, ui > 0, fi > 1)
        ),
        positions_sha256=_hash_arrays("positions-angstrom-v1", (positions,)),
        active_node_count=int((ui > 0).sum()),
        buried_plateau_count=int(((fi == 1) & ~transition).sum()),
        transition_node_count=int(transition.sum()),
        minimum_switch_margin=minimum_switch,
        minimum_active_f_margin=(
            float(active_margins.min()) if active_margins.numel() else None
        ),
        one_sided_nodes=tuple(proofs),
    )
