from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_route2_pyddx_optional_runtime_extra_is_exactly_pinned():
    """Keep the user-installable optional runtime aligned with import guards."""

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "route2-pyddx = [" in pyproject
    assert '"pyddx==0.8.0"' in pyproject
    assert '"pyscf==2.13.1"' in pyproject






def test_route2_documentation_matches_the_public_fail_closed_contract():
    fidelity_baseline = json.loads(
        (
            REPOSITORY_ROOT
            / "docs/implicit-solvation/benchmarks/route2-qm-fidelity-v1.json"
        ).read_text(encoding="utf-8")
    )
    fixed_qm_mae = fidelity_baseline["fixed_conformer_panel"]["summary"][
        "route2_vs_qm"
    ]["mae_kcal_mol"]
    flexible_qm_mae = fidelity_baseline["flexible_conformer_panel"]["summary"][
        "mean_absolute_route2_minus_qm_kcal_mol"
    ]
    overview = (
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md"
    ).read_text(encoding="utf-8")
    normalized_overview = " ".join(overview.split())
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    normalized_formulas = " ".join(formulas.split())
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    normalized_benchmark = " ".join(benchmark.split())

    assert "#model=macepol-m" in overview
    assert "#sp\n" in overview
    assert (
        "method=smd,provider=pcmsolver,profile=smd-iefpcm,"
        "response=scf,standard_state=1m,experimental=true"
    ) in overview
    assert "backend=mock" not in overview
    assert "coarse-grained net charge density" in overview
    assert "not a thermochemical Gibbs free energy" in overview
    assert "immutable **leaf** components" in overview
    assert "must never be summed over" in normalized_overview
    assert "E_{\\mathrm{MACE,intrinsic}}" in formulas
    assert "\\frac12" in formulas
    assert "Research/Innovation Route" in formulas
    assert "complete MAPLE solution-phase PES" in overview
    assert "Research derivative evidence: retained; public force capability disabled" in formulas
    assert "#sp(verbose=1)" in overview
    assert "provider=pyddx" in overview
    assert "profile=smd-ddpcm-l15-n1202-v1" in overview
    assert (
        "profile=smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1"
        in overview
    )
    assert "use_pbc_evaluator=True" in overview
    assert "fixed 40 Å cubic helper box" in overview
    assert "not proof that it is equivalent" in overview
    assert "pyddx==0.8.0" in overview
    assert "pyscf==2.13.1" in overview
    assert "single-point derivative evidence" in overview
    assert "not a universal grid" in formulas
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation.split())
    assert "current PCMSolver C ABI has no force endpoint" in validation
    roadmap = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE2_FORCE_ROADMAP.md"
    ).read_text(encoding="utf-8")
    normalized_roadmap = " ".join(roadmap.split())
    assert "Stationarity is not established" in roadmap
    assert "Route 2 requires an adjoint fixed-point derivative" in roadmap
    assert "polar_output_torch()" in roadmap
    assert "current CDS area on the default PCMSolver profile is not" in roadmap
    assert "Warning-triggered cavity switching is not a PES rule" in roadmap
    assert "cavity_policy=fixed-stability-branch" in overview
    assert "geometry-dependent **policy selection**" in overview
    assert "`PEDRA.OUT` warning lines are" in overview
    assert "Complete same-energy coordinate VJP" in overview
    assert "PCMSolver--GePol | yes | no; fails closed" in overview
    assert (
        "pyddx ddPCM `l15/n1202` + PySCF SMD CDS | yes | "
        "research evidence only | energy only"
    ) in overview
    assert "profile=smd-ddpcm-l15-n1202-multisolv-v1" in overview
    assert "strict_original_smd_equivalence=false" in overview
    assert "SMD-CDS-augmented MACE-POLAR/ddPCM hybrid" in overview
    assert "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2" in validation
    assert "No immutable multi-solvent chemical-accuracy artifact" in validation
    assert "not a multi-solvent accuracy estimate" in normalized_validation
    assert "route2-ddpcm-result.json" in overview
    assert "route2-ddpcm-state.npz" in overview
    assert "route2-public-result-ledger.json" in overview
    assert "engineering stability hyperparameters" in formulas
    assert "not SMD/PCM constants" in formulas
    assert "0.9999" in formulas
    assert "0.0840" in formulas
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
    assert "FullReactionFieldPositionDerivative" in formulas
    assert "continuum_coupled_solvation_coordinate_gradient()" in formulas
    assert "same reaction-field object" in formulas
    assert "ExternalMEPCavityResponse" in formulas
    assert "SurfaceChargeState" in formulas
    assert "ExternalMEPCavityOperatorDerivative" in formulas
    assert "continuum_operator_position_vjp()" in formulas
    assert "polarization_operator_position_gradient()" in formulas
    assert "q_{\\mathrm{sym}}" in formulas
    assert "not production-ready" in roadmap
    assert "54cd781" in formulas
    assert "3bd6a31" in roadmap
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
    assert "full-continuum assembly boundary" in roadmap
    assert "current PCMSolver-backed map" in roadmap
    assert "three-orientation" in validation
    assert "Order 47" in validation
    assert "rotation-covariant discretization" in roadmap
    assert "pyscf_smd_cds()" in formulas
    assert "pyscf_smd_water_cds()" in formulas
    assert "pyscf.solvent.smd.get_cds_legacy" in validation
    assert "position gradient, not force" in formulas
    assert "\\mathcal L^{(\\mathrm{recip},40)}" in formulas
    assert "coordinate-transform pullback" in formulas
    assert "both sides of a Cartesian Hessian" in formulas
    assert "arbitrary box lengths fail closed" in formulas
    assert "loader-only manifest is not retained" in roadmap
    assert "CDS-only component gate" in validation
    assert "assemble_total_solvation_coordinate_gradient()" in formulas
    assert "no gas-force argument" in formulas
    assert "algebra/interface gate only" in validation
    assert "2744038" in validation
    assert "Both failed attempts remain" in validation
    assert "dense order-47 continuum" in validation
    assert "discrete-energy rotation anisotropy" in formulas
    assert "post-hoc torque projection is forbidden" in normalized_formulas
    assert "order 53" in validation
    assert "no order-59 run" in normalized_validation
    assert "three-case MAE worsened" in normalized_validation
    assert "0.7248%" in validation
    assert "public correction force differed" in normalized_validation
    assert "0.0018627013 kcal/mol" in validation
    assert "10^{-10}" in validation
    assert "clean tracked working tree" in validation
    assert "_macepol_long_range.py" in roadmap
    assert "4.11\\times10^{-11}" in validation
    assert "forbidden provider-warning count was zero" in normalized_validation
    assert "not a PCMSolver cavity warning" in normalized_validation
    assert "short NVE conservation" in overview
    assert "route2-qm-fidelity-v1.json" in overview
    assert "polarization errors currently cancel" in normalized_overview
    assert "route2-protocol.json" in benchmark
    assert (
        "three same-geometry Route-2/QM/experiment comparisons"
        in normalized_benchmark
    )
    assert "not hardware-normalized speedups" in normalized_benchmark
    assert "does not define Route 2" in benchmark
    assert "No full-population Route-2 FreeSolv accuracy artifact is frozen" in benchmark
    assert "route2-direct-pcm-freesolv12-ddpcm-ddcosmo-v2-execution-173d5fdf.json" in benchmark
    assert "route2-live-multisolv-ddpcm-ddcosmo-v2-execution-1bbef280.json" in benchmark
    assert (
        "route2-v0-freesolv12-zero-field-mace-static-surface-mep-"
        "execution-edfae78e.json"
    ) in benchmark
    assert "The force adjoint is **not** an additional" in formulas
    assert "QM fidelity, component cancellation" in formulas
    assert f"{fixed_qm_mae:.4f} kcal/mol" in formulas
    assert f"{flexible_qm_mae:.4f} kcal/mol" in formulas
    assert "not a discretization-identical operator reference" in formulas
    assert "not an HFE-accuracy estimate" in formulas
    assert (
        "does not prove that all AO-density sharp-cavity PCM"
        in normalized_formulas
    )
    assert (
        "All three fixed-conformer energies have been reconfirmed"
        in normalized_overview
    )
    assert "source-geometry analytic-force canary" in normalized_overview
    assert "finite difference is `0.7570030893 eV/angstrom`" in normalized_overview
    assert "0.0451054120 eV/rad" in normalized_overview
    assert "0.0450446898 eV/rad" in normalized_overview
    assert "0.0451675753 eV/rad" in normalized_overview
    assert (
        "pre-registered smooth second-order refinement test **fails**"
        in normalized_overview
    )
    assert "Frozen density carries `73.8%`" in normalized_overview
    assert (
        "active-set-associated explicit continuum geometry response"
        in normalized_overview
    )
    assert "does not authorize a production change" in normalized_overview
    assert "without rerunning either energy" in normalized_overview
    assert "0.04520643472551838 eV/rad" in formulas
    assert "0.04510541195057784 eV/rad" in formulas
    assert "0.04504468982867865 eV/rad" in formulas
    assert "0.04516757532134049 eV/rad" in formulas
    assert "no scientific energy evaluation was rerun" in normalized_formulas
    assert "evidence-generating execution heads" in benchmark
    assert "runtime-equivalence reference head" in benchmark
    assert "pre-registered `1.5--2.5` smooth second-order gate therefore fails" in (
        normalized_benchmark
    )
    assert "predeclared three-step validation" in validation
    assert "cannot retroactively overwrite" in normalized_validation
    assert (
        "pre-registered asymptotic-order validation nevertheless has "
        "`status=fail`"
        in normalized_validation
    )
    assert "self-consistent electrostatic coupling block" in normalized_validation
    assert "oriented loop force work" in normalized_validation
    assert "Broader flexible-geometry continuity" in roadmap
    assert "historical local two-torsion closed loop" in normalized_roadmap
    assert (
        "The completed pre-registered diagnosis"
        in normalized_roadmap
    )
    assert "all 18 validity gates pass" in normalized_roadmap
    assert (
        "active-set-associated-explicit-continuum-geometry-response"
        in normalized_roadmap
    )
    assert "upstream-backed smooth cavity/operator profile" in normalized_roadmap
    assert (
        "upstream-backed smooth cavity/operator feasibility profile"
        in normalized_benchmark
    )


def test_route2_continuum_engine_stage_preserves_scientific_boundary():
    stage = (
        REPOSITORY_ROOT
        / "docs"
        / "implicit-solvation"
        / "ROUTE2_CONTINUUM_ENGINE.md"
    ).read_text(encoding="utf-8")
    normalized_stage = " ".join(stage.lower().split())

    assert "zero-behaviour-change" in stage
    assert "unmixed" in stage
    assert "This stage must not" in stage
    assert "add or accept a new `provider`" in stage
    assert "does not import pyddx" in normalized_stage
    assert "separately pre-registered" in stage
