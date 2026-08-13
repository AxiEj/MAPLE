#!/usr/bin/env python3
"""Run unchanged symmetry/loop gates for the disabled pair-frame candidate."""

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.release import PAIRFRAME110_SYMMETRY_PANEL_CONTRACT_VERSION

from pairframe110_water_common import (
    PAIRFRAME110_BOX_LENGTH_A,
    PAIRFRAME110_REQUIRED_SOURCE_PATHS,
    build_system_with_model,
    root_context,
)
from run_fixedbox590_symmetry_panel import _parse_args, run_symmetry_panel

SCHEMA_VERSION = "route2-pairframe110-symmetry-panel-shard-v1"
REQUIRED_SOURCE_PATHS = PAIRFRAME110_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/symmetry_panel.py",
    "tools/route2_release/run_fixedbox590_symmetry_panel.py",
    "tools/route2_release/run_pairframe110_symmetry_panel.py",
)


def main() -> None:
    run_symmetry_panel(
        _parse_args(),
        schema_version=SCHEMA_VERSION,
        contract_version=PAIRFRAME110_SYMMETRY_PANEL_CONTRACT_VERSION,
        required_source_paths=REQUIRED_SOURCE_PATHS,
        model_evaluator_profile=MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[40],
        system_builder=build_system_with_model,
        root_context_builder=root_context,
        box_length=PAIRFRAME110_BOX_LENGTH_A,
        output_marker="ROUTE2_PAIRFRAME110_SYMMETRY_PANEL_SHARD",
        artifact_kind="disabled-real-stack-pairframe110-symmetry-loop-panel-shard",
    )


if __name__ == "__main__":
    main()
