#!/usr/bin/env python3
"""Independently aggregate analytic-harmonic distorted/path PES shards."""

from aggregate_fixedbox590_pes_panel import _parse_args, aggregate_pes_panel
from operational_analytic_harmonic_common import PROFILE_ID, SCALAR_ID
from run_operational_analytic_harmonic_pes_panel import (
    CONTRACT_VERSION,
    SCHEMA_VERSION as SHARD_SCHEMA_VERSION,
)

SCHEMA_VERSION = "route2-operational-analytic-harmonic-pes-panel-aggregate-v1"
REQUIRED_SOURCE_PATHS = (
    "tools/route2_release/aggregate_fixedbox590_pes_panel.py",
    "tools/route2_release/aggregate_operational_analytic_harmonic_pes_panel.py",
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/pes_validation.py",
)


def main() -> None:
    aggregate_pes_panel(
        _parse_args(),
        schema_version=SCHEMA_VERSION,
        shard_schema_version=SHARD_SCHEMA_VERSION,
        contract_version=CONTRACT_VERSION,
        expected_profile_id=PROFILE_ID,
        expected_scalar_id=SCALAR_ID,
        require_extended_gates=True,
        verifier_required_paths=REQUIRED_SOURCE_PATHS,
        output_marker="ROUTE2_OPERATIONAL_ANALYTIC_HARMONIC_PES_PANEL_AGGREGATE",
    )


if __name__ == "__main__":
    main()
