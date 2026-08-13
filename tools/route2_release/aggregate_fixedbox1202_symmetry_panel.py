#!/usr/bin/env python3
"""Independently aggregate unchanged symmetry/loop gates for CPCM1202."""

from maple.solvation.api.profiles import (
    DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1,
)
from maple.solvation.release import FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION

from aggregate_fixedbox590_symmetry_panel import (
    _parse_args,
    aggregate_symmetry_panel,
)


def main() -> None:
    aggregate_symmetry_panel(
        _parse_args(),
        schema_version="route2-fixedbox1202-symmetry-panel-aggregate-v1",
        shard_schema_version="route2-fixedbox1202-symmetry-panel-shard-v1",
        contract_version=FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION,
        expected_profile_id=(
            DIAGNOSTIC_FIXED_BOX48_CPCM_1202_RADIAL_GTO_PROFILE_V1
        ),
        verifier_required_paths=(
            "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
            "tools/route2_release/aggregate_fixedbox1202_symmetry_panel.py",
            "maple/solvation/release/symmetry_panel.py",
            "maple/solvation/release/pes_validation.py",
        ),
        output_marker="ROUTE2_FIXEDBOX1202_SYMMETRY_PANEL_AGGREGATE",
    )


if __name__ == "__main__":
    main()
