"""Optional implicit-solvent correction exports.

Importing the package must stay dependency-light because Python executes this
module before any legacy ``extra_correction.implicit`` submodule. Torch-backed
GBSA/QEq implementations are therefore loaded only when their public names are
actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .charge.qeq import QEqTorch as QEqTorch
    from .solvent.gbsa.gbsa import GBSA as GBSA


def __getattr__(name: str):
    if name == "GBSA":
        from .solvent.gbsa.gbsa import GBSA

        return GBSA
    if name == "QEqTorch":
        from .charge.qeq import QEqTorch

        return QEqTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GBSA", "QEqTorch"]
