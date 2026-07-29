from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-structured-solvent-admission-v1.json"
)


def test_rism_reciprocal_control_is_a_preregistered_math_gate_only():
    protocol = json.loads(ADMISSION.read_text(encoding="utf-8"))
    foundation = protocol["implemented_foundation"]
    control = foundation["rism_short_range_reciprocal_control"]

    assert control["module"].endswith("route2_v0_rism_reciprocal")
    assert "radial-Nyquist" in control["capability"]
    assert "HNC scalar/gradient" in control["capability"]
    assert "frozen before any target-solute calculation" in control["tail_policy"]
    assert (
        "not a registered real-solvent mapping"
        in control["not_a_physical_liquid_backend"]
    )
    assert any(
        "source-defined short-range radial transform" in item
        for item in protocol["validation_sequence"]
    )
    assert any(
        "production-scale grid-convergence" in item
        for item in foundation["not_implemented"]
    )
