from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_permanent_reference_identifiability_audit_fails_closed():
    audit = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-permanent-reference-identifiability-audit-v1.json"
        ).read_text(encoding="utf-8")
    )
    theory = (
        ROOT / "docs/implicit-solvation/ROUTE2_V0_PERMANENT_REFERENCE_IDENTIFIABILITY.md"
    ).read_text(encoding="utf-8")
    register = (
        ROOT / "docs/implicit-solvation/ROUTE2_V0_EXPLORATION_REGISTER.md"
    ).read_text(encoding="utf-8")

    assert audit["status"] == (
        "current-response-assets-rejected-as-a-complete-permanent-reference"
    )
    assert all(value is False for value in audit["hard_constraints"].values())
    assert "neutral transition densities" in audit["mathematical_finding"][
        "transition_space_boundary"
    ]
    assert "not be coupled to a physical continuum" in audit["decision_rule"]
    assert "Reference-shift theorem" in theory
    assert "Each mode integrates to zero." in theory
    assert (
        "must not be silently imported as a V0 molecular solution"
        in " ".join(theory.split())
    )
    assert "V0 permanent-reference identifiability audit" in register
    assert (
        "same-basis structural KKT and permanent-reference identifiability gate"
        in register
    )
