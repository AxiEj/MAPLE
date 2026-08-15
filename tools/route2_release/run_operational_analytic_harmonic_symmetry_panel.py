#!/usr/bin/env python3
"""Run closed-loop gates for the disabled analytic-harmonic operational PES."""

from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)

from operational_analytic_harmonic_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    EXPOSURE_LMAX,
    EXPOSURE_RADIAL_QUADRATURE_ORDER,
    GREEN_RADIAL_QUADRATURE_ORDER,
    PROFILE_ID,
    SCALAR_ID,
    SOURCE_RADIAL_QUADRATURE_ORDER,
    SURFACE_LMAX,
    TRANSITION_WIDTH_ANGSTROM2,
    build_system_with_model,
    cold_warm_record,
    harmonic_domain_record,
    identity_record,
    root_context,
)
from run_fixedbox590_symmetry_panel import _parse_args, run_symmetry_panel

SCHEMA_VERSION = "route2-operational-analytic-harmonic-symmetry-panel-shard-v1"
CONTRACT_VERSION = "route2-operational-analytic-harmonic-symmetry-loop-panel-v1"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/pes_validation.py",
    "maple/solvation/release/symmetry_panel.py",
    "tools/route2_release/panel_continuum_identity.py",
    "tools/route2_release/run_fixedbox590_pes_panel.py",
    "tools/route2_release/run_fixedbox590_symmetry_panel.py",
    "tools/route2_release/run_operational_analytic_harmonic_symmetry_panel.py",
)


def main() -> None:
    run_symmetry_panel(
        _parse_args(),
        schema_version=SCHEMA_VERSION,
        contract_version=CONTRACT_VERSION,
        required_source_paths=REQUIRED_SOURCE_PATHS,
        model_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
        system_builder=build_system_with_model,
        root_context_builder=root_context,
        cold_warm_record_builder=cold_warm_record,
        identity_record_builder=identity_record,
        domain_record_builder=harmonic_domain_record,
        box_length=0,
        output_marker="ROUTE2_OPERATIONAL_ANALYTIC_HARMONIC_SYMMETRY_PANEL_SHARD",
        artifact_kind=(
            "disabled-operational-analytic-harmonic-symmetry-loop-panel-shard"
        ),
        contract_metadata={
            "profile_id": PROFILE_ID,
            "scalar_id": SCALAR_ID,
            "harmonic_configuration": {
                "surface_lmax": SURFACE_LMAX,
                "exposure_lmax": EXPOSURE_LMAX,
                "transition_width_A2": TRANSITION_WIDTH_ANGSTROM2,
                "exposure_radial_quadrature_order": (EXPOSURE_RADIAL_QUADRATURE_ORDER),
                "source_radial_quadrature_order": SOURCE_RADIAL_QUADRATURE_ORDER,
                "green_radial_quadrature_order": GREEN_RADIAL_QUADRATURE_ORDER,
                "laboratory_fixed_surface_grid": False,
            },
            "evidence_reuse": "geometry-and-gates-only; no-legacy-numerical-values",
        },
    )


if __name__ == "__main__":
    main()
