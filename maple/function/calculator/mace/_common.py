"""Shared primitives for the hand-wrapped MACE backends.

These helpers used to be copy-pasted verbatim across ``_mace_calculator.py``,
``_mace_general_calculator.py`` and ``_macepol_calculator.py``. They are kept
here so the three no-PBC MACE wrappers share one radius-graph / one-hot / dtype
implementation. The full per-model ``data_dict``/input tuples still live in each
backend because the traced forward signatures differ (20-field dict vs 6/12
positional tensors).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Union

import torch

from ..analytic_protocol import (
    NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
    NEIGHBOR_CUTOFF_GUARD_POLICY,
)


@dataclass(frozen=True)
class GuardedRadiusGraph:
    """One fixed no-PBC edge set for a single analytic derivative operation."""

    edge_index: torch.Tensor
    shifts: torch.Tensor
    minimum_cutoff_gap_angstrom: float | None


# Element-symbol → atomic-number map. Superset of what the individual MACE
# wrappers historically carried (MACE-POLAR additionally handled Br/I).
_SYMBOL2Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "I": 53,
}


def symbols_to_Z(symbols: Sequence[Union[str, int]]) -> list:
    """Convert element symbols (or pass-through ints) to atomic numbers."""
    out = []
    for s in symbols:
        if isinstance(s, int):
            out.append(int(s))
        else:
            z = _SYMBOL2Z.get(str(s))
            if z is None:
                raise ValueError(f"Unknown element symbol: {s}")
            out.append(z)
    return out


def one_hot_node_attrs(
    Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64
) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with the model table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(
            f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}"
        )
    return eq.to(dtype)


def radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct an O(N^2) radius graph without periodic boundaries.

    Returns ``(edge_index, shifts)`` where ``shifts`` is all-zero (no PBC).
    """
    N = positions.size(0)
    rij = positions[:, None, :] - positions[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=positions.device)
    mask.fill_diagonal_(False)
    mask &= (d2 <= (r_max + 1e-12) ** 2)
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    src = torch.cat([iu, ju], dim=0)
    dst = torch.cat([ju, iu], dim=0)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    shifts = torch.zeros((edge_index.size(1), 3), dtype=positions.dtype, device=positions.device)
    return edge_index, shifts


def guarded_radius_graph_no_pbc(
    positions: torch.Tensor,
    r_max: float,
    *,
    margin_angstrom: float = NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM,
) -> GuardedRadiusGraph:
    """Build one fixed graph and reject a second-AD cutoff discontinuity.

    The ordinary :func:`radius_graph_no_pbc` path intentionally remains
    unchanged.  This guard is used only by new analytic Hessian/HVP operations.
    Pair distances are inspected without attaching the discrete graph decision
    to autograd, then the exact legacy graph builder supplies the edge tensors.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Analytic MACE positions must have shape (N, 3).")
    if not torch.isfinite(positions).all():
        raise ValueError("Analytic MACE coordinates must be finite.")
    cutoff = float(r_max)
    margin = float(margin_angstrom)
    if not (cutoff > 0.0 and math.isfinite(cutoff)):
        raise ValueError("Analytic MACE radius-graph cutoff must be finite and positive.")
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError(
            "Analytic MACE cutoff-guard margin must be finite and non-negative."
        )

    atom_count = int(positions.shape[0])
    minimum_gap = float('inf')
    if atom_count > 1:
        pair_indices = torch.triu_indices(
            atom_count,
            atom_count,
            offset=1,
            device=positions.device,
        )
        pair_vectors = (
            positions[pair_indices[0]] - positions[pair_indices[1]]
        ).detach()
        distances = torch.linalg.vector_norm(pair_vectors, dim=1)
        if not torch.isfinite(distances).all():
            raise ValueError("Analytic MACE pair distances must be finite.")
        if torch.any(distances == 0):
            raise ValueError(
                f"{NEIGHBOR_CUTOFF_GUARD_POLICY} rejects coincident atom pairs."
            )
        gaps = torch.abs(distances - cutoff)
        minimum_gap = float(torch.min(gaps).detach().cpu())
        if minimum_gap <= margin + 1.0e-12:
            raise ValueError(
                f"{NEIGHBOR_CUTOFF_GUARD_POLICY} rejects a geometry with "
                f"minimum |r_ij-r_cut|={minimum_gap:.17g} Angstrom at or below "
                f"the frozen {margin:.17g} Angstrom margin."
            )

    edge_index, shifts = radius_graph_no_pbc(positions, cutoff)
    return GuardedRadiusGraph(
        edge_index=edge_index,
        shifts=shifts,
        minimum_cutoff_gap_angstrom=minimum_gap,
    )


def model_float_dtype(model, default=torch.float64) -> torch.dtype:
    """Infer the scripted/traced model's floating dtype for tensor inputs."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return default
