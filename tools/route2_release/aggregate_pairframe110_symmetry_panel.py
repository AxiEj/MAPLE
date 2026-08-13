#!/usr/bin/env python3
"""Independently aggregate disabled pair-frame symmetry/loop shards."""

from maple.solvation.api.profiles import (
    DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
)
from maple.solvation.release import PAIRFRAME110_SYMMETRY_PANEL_CONTRACT_VERSION

from aggregate_fixedbox590_symmetry_panel import (
    _parse_args,
    aggregate_symmetry_panel,
)

SHARD_SCHEMA_VERSION = "route2-pairframe110-symmetry-panel-shard-v1"


def main() -> None:
    aggregate_symmetry_panel(
        _parse_args(),
        schema_version="route2-pairframe110-symmetry-panel-aggregate-v1",
        shard_schema_version=SHARD_SCHEMA_VERSION,
        contract_version=PAIRFRAME110_SYMMETRY_PANEL_CONTRACT_VERSION,
        expected_profile_id=DIAGNOSTIC_PAIR_FRAME_CPCM_RADIAL_GTO_PROFILE_V1,
        verifier_required_paths=(
            "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
            "tools/route2_release/aggregate_pairframe110_symmetry_panel.py",
            "maple/solvation/release/symmetry_panel.py",
            "maple/solvation/release/pes_validation.py",
        ),
        output_marker="ROUTE2_PAIRFRAME110_SYMMETRY_PANEL_AGGREGATE",
    )


if __name__ == "__main__":
    main()
