from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-structured-solvent-admission-v1.json"
)


def test_periodic_coulomb_control_is_admitted_only_as_a_neutral_math_gate():
    protocol = json.loads(ADMISSION.read_text(encoding="utf-8"))
    foundation = protocol["implemented_foundation"]
    control = foundation["periodic_coulomb_long_range_control"]

    assert control["module"].endswith("route2_v0_periodic_coulomb")
    assert "same scalar" in control["capability"]
    assert "finite-difference" in control["capability"]
    assert "non-neutral" in control["charged_branch_boundary"]
    assert (
        "does not interpolate Cvv short-range"
        in control["not_a_physical_liquid_backend"]
    )
    assert any(
        "neutral source-SMEAR periodic Poisson long-range scalar" in item
        for item in protocol["validation_sequence"]
    )
    assert any(
        "short-range direct-correlation interpolation" in item
        for item in foundation["not_implemented"]
    )
