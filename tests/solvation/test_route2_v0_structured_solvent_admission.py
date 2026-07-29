from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADMISSION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-structured-solvent-admission-v1.json"
)


def _protocol() -> dict:
    return json.loads(ADMISSION.read_text(encoding="utf-8"))


def test_structured_solvent_admission_locks_the_no_training_boundary():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-structured-solvent-admission-v1"
    assert protocol["status"] == (
        "preregistered-before-physical-liquid-functional-execution"
    )
    assert protocol["construction"]["name"] == (
        "route2-v0-fixed-density-structured-solvent-v1"
    )

    assert protocol["hard_constraints"] == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "density_projection": False,
        "response_tempering_or_eigenvalue_clipping": False,
        "error_driven_cavity_or_dispersion_adjustment": False,
        "per_record_closure_or_model_selection": False,
        "smd_cds_in_no_fit_core": False,
        "amber_gaff_or_am1bcc_solute_substitution": False,
        "legacy_public_route_changed": False,
    }
    foundation = protocol["implemented_foundation"]
    assert foundation["module"].endswith("route2_v0_structured_solvent")
    assert "MACE Gaussian l<=1 electrostatic potential" in foundation["capability"]
    assert any("short-range" in item for item in foundation["not_implemented"])
    assert "chemistry benchmark or accuracy claim" in foundation["not_implemented"]
    promolecular = foundation["independent_promolecular_input"]
    assert promolecular["artifact"] == "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1"
    assert "H, C, N, O, S, and Cl" in promolecular["capability"]
    assert "energy-only molecular Thomas-Fermi control" in promolecular[
        "not_coupled"
    ]
    pauli = foundation["frozen_density_pauli_overlap_control"]
    assert pauli["module"].endswith("route2_v0_frozen_density_embedding")
    assert "parameter-free Thomas-Fermi nonadditive kinetic scalar" in pauli[
        "capability"
    ]
    assert "does not relabel the MACE Gaussian coefficients" in pauli[
        "source_boundary"
    ]
    assert "C1 but not C2" in pauli["smoothness_boundary"]
    assert "total solvation free energy" in pauli["not_a_physical_liquid_backend"]
    assert "accuracy result" in pauli["not_a_physical_liquid_backend"]
    molecular = foundation["molecular_promolecular_external_potential_control"]
    assert molecular["module"].endswith("route2_v0_molecular_external_potential")
    assert "whole-molecule nonnegative promolecular solvent density" in molecular[
        "capability"
    ]
    assert "not as a sum of sitewise nonadditive scalars" in molecular[
        "source_boundary"
    ]
    assert "content-addressed" in molecular["source_boundary"]
    assert "C1 but not C2" in molecular["smoothness_boundary"]
    assert "total solvation free energy" in molecular["not_a_physical_liquid_backend"]
    assert "accuracy result" in molecular["not_a_physical_liquid_backend"]
    ideal_gas = foundation["molecular_ideal_gas_variational_control"]
    assert ideal_gas["module"].endswith("route2_v0_molecular_ideal_gas")
    assert "8*pi^2 orientation measure" in ideal_gas["capability"]
    assert "analytic stationary density" in ideal_gas["capability"]
    assert "exactly equal" in ideal_gas["source_boundary"]
    assert "rho_bulk/(8*pi^2)" in ideal_gas["source_boundary"]
    assert "total solvation free energy" in ideal_gas[
        "not_a_physical_liquid_backend"
    ]
    molecular_hnc = foundation["molecular_site_hnc_variational_bridge_control"]
    assert molecular_hnc["module"].endswith("route2_v0_molecular_site_hnc")
    assert "periodic-cell times 8*pi^2 measure" in molecular_hnc["capability"]
    assert "projection adjoint" in molecular_hnc["capability"]
    assert "exactly equal" in molecular_hnc["source_boundary"]
    assert "raw 1D-RISM Cvv table is rejected" in molecular_hnc["source_boundary"]
    assert "total solvation free energy" in molecular_hnc[
        "not_a_physical_liquid_backend"
    ]
    hnc = foundation["site_hnc_variational_reference"]
    assert hnc["module"].endswith("route2_v0_site_hnc")
    assert "synthetic-only periodic multi-site HNC" in hnc["capability"]
    assert "MACE Gaussian grid potential" in hnc["mace_coupling_boundary"]
    assert "No frozen physical direct correlations" in hnc["not_a_physical_solvent_asset"]
    rism = foundation["rism_bulk_direct_correlation_parser"]
    assert rism["module"].endswith("route2_v0_rism_bulk")
    assert "bulk-only 1D-RISM XVV metadata" in rism["capability"]
    assert rism["control_artifact"] == "route2-v0-rism1d-cspce-bulk-control-v1.json"
    assert "cannot be wrapped" in rism["not_a_cartesian_liquid_backend"]


def test_structured_solvent_admission_preserves_the_mace_source_boundary():
    protocol = _protocol()
    construction = protocol["construction"]
    equivalence = protocol["source_equivalence_contract"]

    assert "unmodified zero-reaction-field MACE-POLAR" in construction[
        "electronic_state"
    ]
    assert "gaussian_multipole_potential" in construction["electrostatic_source"]
    assert "nonnegative promolecular" in construction["short_range_source"]
    assert "does not replace phi_c0^G" in construction["short_range_source"]
    assert "MACE-native all-space grid potential" in equivalence[
        "required_grid_interface"
    ]
    assert "forbidden" in equivalence["ambertools_stock_endpoint"]
    assert "GAFF, AM1-BCC" in equivalence["ambertools_stock_endpoint"]
    assert "separate research control only" in equivalence["route1_all_site_rism"]
    assert "forbidden" in equivalence["smd_cds_combination"]


def test_structured_solvent_admission_requires_a_frozen_liquid_asset_and_review():
    protocol = _protocol()
    liquid = protocol["structured_solvent_contract"]
    solvent = protocol["solvent_asset_contract"]
    benchmark = protocol["benchmark_and_review_contract"]

    assert liquid["primary_closure"].startswith("Kovalenko-Hirata")
    assert "thermodynamic pressure correction" in liquid["pressure_correction"]
    assert "diagnostic-only" in liquid["pc_plus_policy"]
    assert "discrete energy/derivative control only" in liquid["reference_kernel"]
    assert "matching input/model provenance" in liquid["rism_bulk_correlation_asset"]
    assert "multiplicity-weighted neutral" in liquid["rism_bulk_correlation_asset"]
    assert (
        "forbidden from a finite Cartesian FFT" in liquid["rism_bulk_correlation_asset"]
    )
    assert solvent["minimum_registered_solvent_count"] == 11
    assert len(solvent["registered_default_solvents"]) == 11
    assert "methanol" in solvent["registered_default_solvents"]
    assert any("bulk susceptibility" in item for item in solvent["required_before_target_scoring"])
    assert "insufficient" in solvent["custom_solvent_rule"]
    assert benchmark["target_solvation_labels_read_before_implementation"] is False
    assert "external final blind dataset" in benchmark["external_blind_manifest"]
    assert "strictly below 1.5 kcal/mol" in benchmark["accuracy_gate"]
    assert "independently review source equivalence" in benchmark["independent_review"]


def test_structured_solvent_admission_keeps_nonlocal_dielectric_electrostatic_only():
    protocol = _protocol()
    solvent = protocol["solvent_asset_contract"]
    foundation = protocol["implemented_foundation"]
    validation = protocol["validation_sequence"]

    assert "k->0 limit" in solvent["nonlocal_dielectric_response_rule"]
    assert "cannot define a molecular-liquid" in solvent[
        "nonlocal_dielectric_response_rule"
    ]
    dielectric = foundation["nonlocal_dielectric_electrostatic_control"]
    assert dielectric["module"].endswith("route2_v0_nonlocal_dielectric")
    assert "epsilon(k) >= 1" in dielectric["capability"]
    assert "scalar dielectric constant" in dielectric["custom_solvent_boundary"]
    assert "non-neutral source is rejected" in dielectric["charged_branch_boundary"]
    assert "total solvation free energy or accuracy result" in dielectric[
        "not_a_physical_liquid_backend"
    ]
    assert any("full nonlocal dielectric spectrum" in item for item in validation)
    assert any("Thomas-Fermi nonadditive kinetic Pauli scalar" in item for item in validation)
    assert any(
        "whole-molecule rather than sitewise Thomas-Fermi density" in item
        for item in validation
    )
    assert any(
        "exact molecular ideal-gas configuration scalar" in item
        for item in validation
    )
    assert any("projected molecular-site HNC bridge" in item for item in validation)
    rism_kernel = foundation["rism_energy_conjugate_periodic_kernel_control"]
    assert rism_kernel["module"].endswith("route2_v0_rism_energy_conjugate")
    assert "short-range-plus-long-range scalar" in rism_kernel["capability"]
    assert "Amber QV is converted only by its length unit" in rism_kernel["capability"]
    assert "must match exactly" in rism_kernel["source_boundary"]
    assert "never passed directly" in rism_kernel["source_boundary"]
    assert "total solvation free energy" in rism_kernel[
        "not_a_physical_liquid_backend"
    ]
    assert any("energy-conjugate RISM periodic-kernel assembly" in item for item in validation)
