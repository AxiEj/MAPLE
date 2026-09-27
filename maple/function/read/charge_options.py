"""Pure fixed-charge option identity shared by input and runtime boundaries."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

CHARGE_OPTION_KEYS = frozenset(
    {"source", "method", "mode", "label", "geometry", "executable", "timeout"}
)


def normalize_charge_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return owned canonical options, rejecting ignored or contradictory labels.

    This validates configuration only: molecular identity, charge sums and
    provider output checks remain in the charge preparation layer.
    """
    if not isinstance(options, Mapping):
        raise ValueError("#charge options must be a mapping.")
    charge = dict(options)
    unknown = set(charge) - CHARGE_OPTION_KEYS
    if unknown:
        raise ValueError(f"Unsupported #charge options: {sorted(map(str, unknown))}.")
    source = str(charge.get("source", "")).lower()
    if source not in {"mol2", "maple"}:
        raise ValueError("#charge source must be 'mol2' or 'maple'.")
    method = charge.get("method")
    mode = str(charge.get("mode", "fixed")).lower()
    geometry = str(charge.get("geometry", "keep")).lower()
    if mode not in {"fixed", "polarizable"}:
        raise ValueError("#charge supports mode=fixed only.")
    normalized_method = "" if method is None else str(method).lower()
    if mode == "polarizable" or normalized_method in {
        "qeq",
        "qeq-gto",
        "cqeq",
        "cqeq-gto",
    }:
        raise ValueError(
            "QEq/CQEq charge models are disabled; use fixed MOL2 charges or "
            "MAPLE AM1-BCC/ABCG2."
        )
    if geometry not in {"keep", "provider"}:
        raise ValueError("#charge geometry must be 'keep' or 'provider'.")
    if source == "mol2":
        if "method" in charge:
            raise ValueError(
                "#charge(source=mol2) reads fixed charges and does not accept method=."
            )
        if geometry != "keep":
            raise ValueError(
                "#charge(source=mol2) does not run a geometry provider; use geometry=keep."
            )
        ignored = sorted({"executable", "timeout"}.intersection(charge))
        if ignored:
            raise ValueError(
                "#charge(source=mol2) does not run a charge executable; remove "
                + ", ".join(ignored)
                + "."
            )
    else:
        normalized_method = "am1bcc" if method is None else normalized_method
        if normalized_method not in {"am1bcc", "abcg2"}:
            raise ValueError("MAPLE charge method must be am1bcc or abcg2.")
        charge["method"] = normalized_method
        if "label" in charge:
            raise ValueError(
                "#charge label is only valid for source=mol2 fixed-charge provenance."
            )
    if "timeout" in charge and (
        isinstance(charge["timeout"], bool)
        or not isinstance(charge["timeout"], (int, float))
        or not math.isfinite(charge["timeout"])
        or charge["timeout"] <= 0
    ):
        raise ValueError("#charge timeout must be a positive finite number of seconds.")
    for key in ("label", "executable"):
        if key in charge and (
            not isinstance(charge[key], str) or not charge[key].strip()
        ):
            raise ValueError(f"#charge {key} must be a non-empty string.")
    charge.update(source=source, mode=mode, geometry=geometry)
    return charge
