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
    normalized_validation = " ".join(validation.split())
    assert "current PCMSolver C ABI has no force endpoint" in validation
    roadmap = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE2_FORCE_ROADMAP.md"
    ).read_text(encoding="utf-8")
    assert "Stationarity is not established" in roadmap
    assert "Route 2 requires an adjoint fixed-point derivative" in roadmap
    assert "polar_output_torch()" in roadmap
    assert "current CDS area is not differentiable" in roadmap
    assert "Warning-triggered cavity switching is not a PES rule" in roadmap
    assert "cavity_policy=fixed-stability-branch" in overview
    assert "geometry-dependent **policy selection**" in overview
    assert "engineering stability hyperparameters" in formulas
    assert "not SMD/PCM constants" in formulas
    assert "PCMSolver--GePol profile remains energy-only" in roadmap
    assert "Never combine energy from one cavity/operator definition" in roadmap
    assert "direct vector-Jacobian products" in roadmap
    assert "fixed_field_forces_ev_per_angstrom" in formulas
    assert "This is not a total solvent force" in roadmap
    assert "UnmixedDensityResidualLinearization" in formulas
    assert "neutral density tangent space" in roadmap
    assert "solve_adjoint()" in formulas
    assert "FixedCavityPCMReactionFieldLinearMap" in formulas
    assert "FixedCavityPCMReactionFieldLinearMap.position_vjp()" in formulas
    assert "intrinsic_energy_field_gradient()" in formulas
    assert "density_position_vjp()" in formulas
    assert "density_to_external_field_order()" in formulas
    assert "fixed_cavity_energy_density_gradient()" in formulas
    assert "fixed_surface_solvation_coordinate_gradient()" in formulas
    assert "aqueous_atomic_surface_tension_position_vjp()" in formulas
    assert (
        "smd_water_cds_fibonacci_swig_inspired_position_gradient()"
        in formulas
    )
    assert (
        "Neither finite quadrature is exactly rotation invariant"
        in formulas
    )
    assert "PySCF 2.13.1" in formulas
    assert "not established as equivalent" in formulas
    assert "5.84--9.73%" in validation
    assert "not established as SWIG-equivalent" in normalized_validation
    assert "MATRIXSYMM=TRUE" in roadmap
    assert "physical energy-gradient right-hand side" in roadmap
    assert "route2-protocol.json" in benchmark
    assert "does not define Route 2" in benchmark
    assert "No Route-2 FreeSolv accuracy artifact is frozen yet" in benchmark
