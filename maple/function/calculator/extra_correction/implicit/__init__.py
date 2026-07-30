"""Evidence-gated implicit-solvation providers."""

from __future__ import annotations

from .charges import ChargeResult, QEqGTO, prepare_charges
from .correction import ImplicitSolvationCorrection
from .gnnis import GNNISReferenceBackend, build_gnnis_reference_adapter
from .smd import SMDImplicitSolvation
from .smd_cds import smd_water_cds
from .result import SolvationResult
from .topology import CanonicalTopology, TopologyProvider, canonicalize_topology

__all__ = [
    "ChargeResult",
    "AniSolvCompactBackend",
    "GNNISReferenceBackend",
    "ImplicitSolvationCorrection",
    "build_gnnis_reference_adapter",
    "QEqGTO",
    "TopologyProvider",
    "CanonicalTopology",
    "canonicalize_topology",
    "SMDImplicitSolvation",
    "SolvationResult",
    "prepare_charges",
    "smd_water_cds",
]


def __getattr__(name: str):
    """Avoid importing additive backends while ``backends`` imports SolvationResult."""

    if name == "AniSolvCompactBackend":
        from .anisolv import AniSolvCompactBackend

        return AniSolvCompactBackend
    raise AttributeError(name)
