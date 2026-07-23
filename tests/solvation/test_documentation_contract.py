from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_documentation_does_not_claim_a_certified_current_default():
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")

    assert "## Current default:" not in benchmark
    assert "temporary experimental default" in benchmark
    assert "current AM1-BCC/OBC-II/ACE default" not in validation


def test_qeq_documentation_freezes_it_as_explicit_experimental_only():
    overview = (
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(overview.split())

    assert "QEq/CQEq are frozen experimental research profiles" in overview
    assert "never selected as defaults or provider fallbacks" in normalized


def test_route2_documentation_matches_the_public_fail_closed_contract():
    overview = (
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md"
    ).read_text(encoding="utf-8")
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")

    assert "#model=macepol-m" in overview
    assert "#sp\n" in overview
    assert "method=smd,response=scf,standard_state=1m,experimental=true" in overview
    assert "backend=mock" not in overview
    assert "coarse-grained net charge density" in overview
    assert "not a thermochemical Gibbs free energy" in overview
    assert "E_{\\mathrm{MACE,intrinsic}}" in formulas
    assert "\\frac12" in formulas
    assert "route2-protocol.json" in benchmark
    assert "No Route-2 FreeSolv accuracy artifact is frozen yet" in benchmark
