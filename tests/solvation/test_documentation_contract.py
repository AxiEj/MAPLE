from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]






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
    assert "Research/Innovation Route" in formulas
    assert "complete MAPLE solution-phase PES" in overview
    assert "Force derivative: next primary milestone" in formulas
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    assert "current PCMSolver C ABI has no force endpoint" in validation
    roadmap = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE2_FORCE_ROADMAP.md"
    ).read_text(encoding="utf-8")
    assert "Stationarity is not established" in roadmap
    assert "current CDS area is not differentiable" in roadmap
    assert "Warning-triggered cavity switching is not a PES rule" in roadmap
    assert "route2-protocol.json" in benchmark
    assert "does not define Route 2" in benchmark
    assert "No Route-2 FreeSolv accuracy artifact is frozen yet" in benchmark
