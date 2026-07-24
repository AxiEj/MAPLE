"""Shared version boundary for optional PySCF Route-2 research components."""

from __future__ import annotations


TESTED_PYSCF_VERSION = "2.13.1"


def require_tested_pyscf_version(
    version: object,
    *,
    feature: str,
) -> str:
    """Return the normalized version or fail before using private APIs."""

    normalized = str(version)
    if normalized != TESTED_PYSCF_VERSION:
        raise RuntimeError(
            f"{feature} is tested only with PySCF {TESTED_PYSCF_VERSION}; "
            f"received {normalized}."
        )
    return normalized
