"""Shared helpers for official MACE-family PBC adapters."""

from typing import Optional


def extract_mace_r_max(calc) -> Optional[float]:
    """Return a MACE-family cutoff as float when available."""
    models = getattr(calc, "models", None)
    if not models:
        r_max = getattr(calc, "r_max", None)
    else:
        r_max = getattr(models[0], "r_max", None)

    if r_max is None:
        return None
    try:
        return float(r_max.item() if hasattr(r_max, "item") else r_max)
    except Exception:
        return None
