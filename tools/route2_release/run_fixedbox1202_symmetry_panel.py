#!/usr/bin/env python3
"""Run the unchanged symmetry/loop gates for the disabled CPCM1202 profile."""

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.release import FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION

from fixedbox1202_common import (
    FIXEDBOX1202_BOX_LENGTH_A,
    FIXEDBOX1202_REQUIRED_SOURCE_PATHS,
    build_fixedbox1202_system_with_model,
)
from run_fixedbox590_symmetry_panel import _parse_args, run_symmetry_panel


def main() -> None:
    run_symmetry_panel(
        _parse_args(),
        schema_version="route2-fixedbox1202-symmetry-panel-shard-v1",
        contract_version=FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION,
        required_source_paths=FIXEDBOX1202_REQUIRED_SOURCE_PATHS,
        model_evaluator_profile=MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[48],
        system_builder=build_fixedbox1202_system_with_model,
        box_length=FIXEDBOX1202_BOX_LENGTH_A,
        output_marker="ROUTE2_FIXEDBOX1202_SYMMETRY_PANEL_SHARD",
        artifact_kind="disabled-real-stack-fixedbox1202-symmetry-loop-panel-shard",
    )


if __name__ == "__main__":
    main()
