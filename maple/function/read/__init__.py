"""Lazy public exports for MAPLE input and explicit-solvent helpers.

Pure format readers such as ``filereader.mol2_reader`` must remain usable in
the dependency-light Route-2 core.  Python executes this package initializer
before loading those submodules, so Torch-backed input orchestration and the
explicit-solvent stack are imported only when their public symbols are
requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_reader import InputReader as InputReader
    from .post_process.explicit_solvent.solvate import ExplicitSolv as ExplicitSolv


def __getattr__(name: str):
    if name == "InputReader":
        from .input_reader import InputReader

        return InputReader
    if name == "ExplicitSolv":
        from .post_process.explicit_solvent.solvate import ExplicitSolv

        return ExplicitSolv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExplicitSolv", "InputReader"]
