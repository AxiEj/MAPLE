from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_documentation_declares_am1bcc_development_default_without_certification():
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")

    assert "## Route 1 development default: AM1-BCC/OBC-II/ACE" in benchmark
    assert "`#charge(source=maple)` defaults to AM1-BCC" in overview
    assert "ABCG2 requires explicit `method=abcg2` selection" in overview
    assert "# Route 1: additive fixed-charge PB/GB implicit solvation" in overview
    assert (
        "No gas-phase MM energy, retraining, hydration-label residual model" in overview
    )
    assert "Conformer-aware calculations are an optional evaluation layer" in overview
    assert "AM1-BCC/OBC-II/ACE is the forward development default" in validation
    assert "not a scientific certification" in validation


def test_qeq_documentation_freezes_it_as_explicit_experimental_only():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(overview.split())

    assert (
        "Fixed QEq-GTO and polarizable CQEq-GTO/GB are frozen experimental "
        "research controls"
    ) in normalized
    assert "never selected as defaults or provider fallbacks" in normalized
    assert "not a Route 1 fixed-charge product profile" in normalized


def test_route1_product_spec_separates_provider_accuracy_and_speed_claims():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(specification.split())

    assert (
        "Radius and nonpolar providers are first-class runtime objects" in specification
    )
    assert "Fixed-geometry SP composition" in specification
    assert "Provider fidelity" in specification
    assert "Solvent overhead" in specification
    assert "reject both a global “faster than MM” claim" in normalized
    assert "Route 1 does not retrain the gas MLIP" in normalized
    assert "MACE-OFF23m (`float64`) and AIMNet2 (`float32`)" in specification
    assert "not universal MLIP" in normalized
    assert (
        "**Route name:** Additive fixed-charge PB/GB implicit solvation"
        in specification
    )
    assert "**Product role:** Baseline/Product Route" in specification
    assert (
        "No chemistry-specific learned correction is part of Route 1" in specification
    )
    assert "Explicit-inner/implicit-outer composition" in specification
    assert "## Canonical acceptance contract" in specification
    assert "mandatory product goal" in normalized
    assert "fast SP, OPT, and SCAN/PES" in normalized
    assert "provider parity, complete-potential force checks, FreeSolv evaluation" in (
        normalized
    )
    assert "opt-in extensions over that same potential" in normalized
    assert "Innovation is not required to modify the PB/GB equations" in normalized


def test_route1_docs_pin_chagb_derivative_rejection_and_ar6_boundary():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((specification + validation + benchmark).split())

    assert "route1-chagb-derivative-capability-audit-2026-07-25.json" in normalized
    assert "0b35bfeb96026ffa4e5876391a0828f39b3cfc8d" in normalized
    assert "nonzero numerical" in normalized
    assert "zero analytical" in normalized
    assert "neither an `igb=9` branch nor AR6 topology" in normalized
    assert "SP-only fail-closed policy" in normalized


def test_route1_docs_reject_apbs_spline_and_apolar_force_candidates():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((overview + specification + validation + benchmark).split())

    assert (
        "route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json" in normalized
    )
    assert (
        "route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json" in normalized
    )
    assert "Forces *must* be calculated with spline-based surfaces!" in normalized
    assert "0.483/2.579 kJ/mol/A" in normalized
    assert "1.114/4.871 kJ/mol/A" in normalized
    assert "generic mbondi2" in normalized
    assert "APBS remains SP-only" in normalized


def test_chagb_runtime_docs_keep_accuracy_profile_sp_only_and_nondefault():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((overview + specification + validation).split())

    assert "provider=ambertools,model=chagb" in normalized
    assert "explicit AM1-BCC SP-only" in normalized
    assert "95 of the 526 development cases" in normalized
    assert "gas and bonded MM energies are unused" in normalized
    assert "does not replace the force-capable OBC-II/ACE default" in normalized


def test_provider_parity_docs_separate_polar_coverage_from_lcpo_applicability():
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((benchmark + validation).split())

    assert "58 supported polar records" in normalized
    assert "39 complete LCPO records" in normalized
    assert "two sulfur/GBn2 slots fail closed" in normalized
    assert "0.161904 kcal/mol" in normalized
    assert "fails closed for Br/I" in normalized
    assert "polar-only parity targets" in normalized


def test_route1_docs_reject_force_inconsistent_amber_pb_exact_difference():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    plain_specification = specification.replace("**", "")
    normalized_validation = " ".join(validation.split())

    assert "Amber PB exact-difference force probe" in specification
    assert (
        "does not, however, restore a conservative solvent correction"
        in plain_specification
    )
    assert "0.0750 to 3.4200 kcal/mol/A" in normalized_validation


def test_route1_docs_record_rejected_analytical_candidate_screen():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(benchmark.split())
    normalized_specification = " ".join(specification.split())
    normalized_overview = " ".join(overview.split())

    assert "Remaining analytical candidate screen" in benchmark
    assert "0.00235 kcal/mol" in normalized
    assert "454/526" in normalized
    assert "1.800792 -> 2.250243" in normalized
    assert "-0.449450 kcal/mol" in normalized
    assert "285/454 cases worsen" in normalized
    assert "No remaining maintained analytical candidate" in specification
    assert (
        "post-hoc development screen, not an independent or label-blind confirmation"
        in normalized_specification
    )
    assert (
        "post-hoc development screen, not an independent or label-blind confirmation"
        in normalized_overview
    )


def test_route1_docs_record_real_dispatcher_evidence_and_confirmation_leak():
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized_benchmark = " ".join(benchmark.split())
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")

    assert "Actual MAPLE SP/OPT/SCAN/MD task matrix" in benchmark
    assert "real `engine`/dispatcher jobs" in benchmark
    assert "MACE-OFF23m, AIMNet2, and ANI2x" in benchmark
    assert "All twelve jobs" in benchmark
    assert "non-periodic NVE/NVT MD" in overview
    assert "10.1063/1.1740409" in formulas
    assert "10.1063/1.2978177" in formulas
    assert "does not yet expose this aggregation" in formulas
    assert "not an unopened or label-sealed confirmation set" in normalized_benchmark
    assert "116 untouched confirmation records" not in benchmark
    assert "label-exposed held-out-by-computation evidence" in validation
    assert "1.301/1.846/6.376" in validation
    assert "0.490 kcal/mol" in validation


def test_route1_product_spec_freezes_ensemble_estimator_and_standard_state():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((specification + formulas).split())

    assert "## Ensemble free-energy contract" in specification
    assert "multi-window MBAR" in specification
    assert "endpoint Zwanzig FEP" in specification
    assert "gas 1 M to ideal-dilute solution 1 M" in specification
    assert "does not silently add" in normalized
    assert "first off-diagonal of an adjacent pair is below `0.03`" in normalized
    assert "10.1021/jp0764384" in formulas
    assert "10.1021/acs.jctc.5b00784" in formulas
    assert "does not yet expose this aggregation" in formulas
    assert "MACE-OFF23m, AIMNet2, and ANI2x" in specification
    assert "0.047-0.083 kcal/mol" in specification


def test_route1_docs_keep_iwm_gb_and_agbnp_outside_product_runtime():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    normalized_specification = " ".join(specification.split())

    assert "IWM-GB is the strongest recent small-molecule endpoint" in specification
    assert "not independent confirmation evidence or an OPT/SCAN provider" in (
        normalized_specification
    )
    assert "External provider feasibility audit" in benchmark
    assert "The finite AGBNP1 energy is an ABI smoke only" in benchmark
    assert "avoids a speculative backend registry or typing layer" in benchmark


def test_route1_docs_freeze_gbr6_and_two_level_probes_as_negative_evidence():
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((specification + benchmark + validation).split())

    assert "GBr6 physical-provider screen" in specification
    assert "526/526 label-free polar-energy records" in benchmark
    assert "GBr6/cavity-dispersion | 2.257 | 3.451" in specification
    assert "no runtime provider was added" in validation
    assert "Two-level OBC-II OPT -> CHA-GB final-SP diagnostic" in benchmark
    assert "all six absolute errors worsen" in normalized
    assert "both potentials and the missing final-SP force disclosed" in normalized


def test_route1_docs_bound_prebuilt_inner_outer_to_a_cluster_potential():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(
        (overview + specification + formulas + validation + benchmark).split()
    )

    assert "## Prebuilt explicit inner / implicit outer" in overview
    assert "inner=prebuilt" in normalized
    assert "fixed-shell cluster-continuum configurational potential" in normalized
    assert "cluster_continuum_correction_hartree" in normalized
    assert "absolute_solvation_free_energy_claim=false" in normalized
    assert "OpenMM GB with ACE or LCPO" in normalized
    assert "stable per-atom identity array" in normalized
    assert "same-element reordering" in normalized
    assert "10.1021/jp802665d" in formulas
    assert "10.1021/jp809712y" in formulas
    assert "10.1039/D0CP02768E" in formulas
    assert "all nine jobs" in normalized.lower()
    assert "must not be compared directly with FreeSolv" in normalized


def test_registered_mlip_route1_capability_gate_is_documented():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    authoring = (REPOSITORY_ROOT / "maple/function/calculator/AUTHORING.md").read_text(
        encoding="utf-8"
    )
    normalized_specification = " ".join(specification.split())

    assert "Registration alone is not treated as compatibility" in overview
    assert "registration alone is insufficient" in specification
    assert (
        "undeclared checkpoint element domain is not certified"
        in normalized_specification
    )
    assert "SUPPORTS_IMPLICIT_SOLVATION" in authoring
    assert "MAPLE cannot certify their element domain" in authoring


def test_implicit_frequency_docs_require_the_complete_numerical_hessian():
    overview = (REPOSITORY_ROOT / "docs/implicit-solvation/README.md").read_text(
        encoding="utf-8"
    )
    specification = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md"
    ).read_text(encoding="utf-8")
    formulas = (
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")
    authoring = (REPOSITORY_ROOT / "maple/function/calculator/AUTHORING.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(
        (overview + specification + formulas + validation + authoring).split()
    )

    assert "#model=ani2x(hessian=numerical)" in overview
    assert "complete reported MLIP-plus-GB force" in normalized
    assert "two force evaluations per movable Cartesian" in normalized
    assert "analytic gas-MLIP Hessian" in normalized
    assert "Energy-only solvent providers cannot use numerical Hessians" in authoring
    assert "not an absolute hydration free energy" in normalized
    assert "solution-standard-state Gibbs energy" in normalized
    assert "Constrained FREQ" in normalized
    assert "active-coordinate Hessian" in normalized
    assert "MACE-OFF23m, AIMNet2, and ANI2x" in validation
