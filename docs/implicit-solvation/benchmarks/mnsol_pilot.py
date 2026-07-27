#!/usr/bin/env python3
"""Deterministic, experiment-blind selection for the bounded MNSol pilot.

The tracked selection manifest contains only opaque integrity identifiers and
aggregate candidate counts.  Solute names, formulas, coordinates, database
entry numbers, and experimental values remain in the user-supplied MNSol
distribution and are never emitted by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

from benchmark_core import canonical_json_bytes, write_json_atomic
from mnsol_dataset import (
    MNSolDataset,
    MNSolEligibleRecord,
    MNSolProtocol,
    eligible_mnsol_records,
    load_mnsol_protocol,
    load_mnsol_v2012,
)

PILOT_ID = "maple-route2-mnsol-aimnet2-multisolvent-pilot-v1"
PILOT_SELECTION_SEED = "maple-route2-mnsol-aimnet2-pilot-v1"
PILOT_MAX_ATOM_COUNT = 20
PILOT_RECORDS_PER_SOLVENT = 1
PILOT_PARTITION_PREFERENCE = ("confirmation", "development")
PILOT_REQUIRE_DISTINCT_GEOMETRIES = True
PILOT_SCORE_INPUT = (
    "selection_seed\\0canonical_solvent\\0geometry_handle\\0entry_number"
)
_OPAQUE_RECORD_DOMAIN = b"maple-mnsol-pilot-record-v1\0"


@dataclass(frozen=True)
class MNSolPilotSelection:
    """One private in-memory row selected by the public opaque manifest."""

    canonical_solvent: str
    eligible_record: MNSolEligibleRecord
    opaque_record_id: str
    selection_score_sha256: str


def _opaque_record_id(record: MNSolEligibleRecord) -> str:
    digest = hashlib.sha256()
    digest.update(_OPAQUE_RECORD_DOMAIN)
    digest.update(record.record.raw_row_sha256.encode("ascii"))
    return digest.hexdigest()


def _selection_score(record: MNSolEligibleRecord) -> str:
    key = (
        f"{PILOT_SELECTION_SEED}\0"
        f"{record.canonical_solvent}\0"
        f"{record.record.geometry_handle}\0"
        f"{record.record.entry_number}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_mnsol_pilot(
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
) -> tuple[MNSolPilotSelection, ...]:
    """Select one small, preferably confirmation, unique row per solvent.

    Selection never uses ``delta_g_kcal_mol`` or any model output.  The
    20-atom ceiling keeps this first diagnostic bounded; later validation must
    use the complete preregistered MNSol partitions rather than this pilot.
    """

    eligible = eligible_mnsol_records(dataset, protocol)
    solvent_order = tuple(
        item.canonical_name
        for item in protocol.panel
        if item.neutral_absolute_validation
    )
    if len(solvent_order) != 10:
        raise ValueError(
            "The bounded MNSol pilot requires exactly ten validation solvents."
        )

    selected: list[MNSolPilotSelection] = []
    used_geometry_handles: set[str] = set()
    for solvent in solvent_order:
        candidates = [
            item
            for item in eligible
            if item.canonical_solvent == solvent
            and len(item.geometry.atomic_numbers) <= PILOT_MAX_ATOM_COUNT
        ]
        chosen: MNSolEligibleRecord | None = None
        chosen_score = ""
        for partition in PILOT_PARTITION_PREFERENCE:
            partition_candidates = [
                item
                for item in candidates
                if item.partition == partition
                and (
                    not PILOT_REQUIRE_DISTINCT_GEOMETRIES
                    or item.record.geometry_handle not in used_geometry_handles
                )
            ]
            if partition_candidates:
                chosen = min(partition_candidates, key=_selection_score)
                chosen_score = _selection_score(chosen)
                break
        if chosen is None:
            raise ValueError(
                "No MNSol pilot candidate satisfies the frozen selection "
                f"policy for solvent={solvent}."
            )
        used_geometry_handles.add(chosen.record.geometry_handle)
        selected.append(
            MNSolPilotSelection(
                canonical_solvent=solvent,
                eligible_record=chosen,
                opaque_record_id=_opaque_record_id(chosen),
                selection_score_sha256=chosen_score,
            )
        )
    return tuple(selected)


def build_mnsol_pilot_selection_manifest(
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
) -> dict[str, object]:
    """Build the deterministic, redistribution-safe preregistration artifact."""

    selection = select_mnsol_pilot(dataset, protocol)
    eligible = eligible_mnsol_records(dataset, protocol)
    bounded = [
        item
        for item in eligible
        if len(item.geometry.atomic_numbers) <= PILOT_MAX_ATOM_COUNT
    ]
    candidate_counts: dict[str, dict[str, int]] = {}
    for solvent in (item.canonical_solvent for item in selection):
        counts = Counter(
            item.partition for item in bounded if item.canonical_solvent == solvent
        )
        candidate_counts[solvent] = {
            name: counts.get(name, 0) for name in PILOT_PARTITION_PREFERENCE
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "route2-mnsol-pilot-selection-v1",
        "pilot_id": PILOT_ID,
        "selection_status": "preregistered-before-model-run",
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
        },
        "selection_policy": {
            "selection_seed": PILOT_SELECTION_SEED,
            "records_per_solvent": PILOT_RECORDS_PER_SOLVENT,
            "max_atom_count": PILOT_MAX_ATOM_COUNT,
            "partition_preference": list(PILOT_PARTITION_PREFERENCE),
            "require_distinct_geometry_handles": (PILOT_REQUIRE_DISTINCT_GEOMETRIES),
            "score": "sha256-lexicographic-minimum",
            "score_input": PILOT_SCORE_INPUT,
            "experimental_value_used_for_selection": False,
            "model_output_used_for_selection": False,
        },
        "candidate_counts_after_domain_and_atom_ceiling": candidate_counts,
        "selected_records": [
            {
                "canonical_solvent": item.canonical_solvent,
                "partition": item.eligible_record.partition,
                "opaque_record_id": item.opaque_record_id,
                "geometry_sha256": item.eligible_record.geometry.sha256,
                "atom_count": len(item.eligible_record.geometry.atomic_numbers),
            }
            for item in selection
        ],
        "redistribution_guard": {
            "raw_rows_emitted": False,
            "entry_numbers_emitted": False,
            "geometry_handles_emitted": False,
            "solute_names_emitted": False,
            "formulas_emitted": False,
            "coordinates_emitted": False,
            "experimental_values_emitted": False,
        },
        "claim_boundary": (
            "This artifact freezes an experiment-blind ten-record pilot "
            "selection only. It contains no MNSol row-level scientific data "
            "and reports no AIMNet2, MACE-POLAR, continuum, SMD-CDS, force, "
            "accuracy, or PES result."
        ),
    }
    manifest["selection_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def validate_frozen_mnsol_pilot_selection(
    manifest: Mapping[str, object],
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
) -> tuple[MNSolPilotSelection, ...]:
    """Recompute the pilot and reject any manifest or dataset drift."""

    expected = build_mnsol_pilot_selection_manifest(dataset, protocol)
    if dict(manifest) != expected:
        raise ValueError(
            "Frozen MNSol pilot selection does not match the current pinned "
            "dataset, protocol, and experiment-blind selection algorithm."
        )
    return select_mnsol_pilot(dataset, protocol)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the bounded experiment-blind MNSol pilot."
    )
    parser.add_argument("action", choices=("freeze",))
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    manifest = build_mnsol_pilot_selection_manifest(dataset, protocol)
    write_json_atomic(args.output, manifest)
    print(
        f"Frozen {len(manifest['selected_records'])} MNSol pilot records "
        "without using experiment values or model outputs for selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
