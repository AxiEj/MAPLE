#!/usr/bin/env python3
"""Freeze and validate complete, shardable MNSol partition selections.

Unlike the ten-record pilot, a partition selection does not choose a favorable
subset: every eligible row in the requested frozen partition is retained.
Records are merely ordered, without experiment values or model outputs, so
bounded ``--record-index`` runs can accumulate into the complete evaluation.

The manifest also marks every row sharing a solute geometry with the earlier
pilot.  This prevents the already-inspected pilot geometries from being
mistaken for an untouched confirmation set.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
from mnsol_pilot import (
    _opaque_record_id,
    validate_frozen_mnsol_pilot_selection,
)

PARTITION_ARTIFACT = "route2-mnsol-partition-selection-v1"
PARTITION_SELECTION_SEED = "maple-route2-mnsol-complete-partition-v1"
PARTITION_ORDERING = "no-prior-pilot-overlap-first-solvent-round-robin-sha256-v1"
SUPPORTED_PARTITIONS = frozenset({"development", "confirmation"})


@dataclass(frozen=True)
class MNSolPartitionSelection:
    """One private in-memory row in a complete partition selection."""

    canonical_solvent: str
    eligible_record: MNSolEligibleRecord
    opaque_record_id: str
    selection_score_sha256: str
    prior_pilot_geometry_overlap: bool


def _selection_score(
    record: MNSolEligibleRecord,
    *,
    partition: str,
) -> str:
    key = (
        f"{PARTITION_SELECTION_SEED}\0"
        f"{partition}\0"
        f"{record.canonical_solvent}\0"
        f"{record.record.geometry_handle}\0"
        f"{record.record.entry_number}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _solvent_order(protocol: MNSolProtocol) -> tuple[str, ...]:
    return tuple(
        item.canonical_name
        for item in protocol.panel
        if item.neutral_absolute_validation
    )


def _round_robin(
    records: Sequence[MNSolPartitionSelection],
    solvent_order: Sequence[str],
) -> list[MNSolPartitionSelection]:
    grouped = {
        solvent: sorted(
            (record for record in records if record.canonical_solvent == solvent),
            key=lambda record: record.selection_score_sha256,
        )
        for solvent in solvent_order
    }
    ordered: list[MNSolPartitionSelection] = []
    maximum = max((len(values) for values in grouped.values()), default=0)
    for ordinal in range(maximum):
        for solvent in solvent_order:
            values = grouped[solvent]
            if ordinal < len(values):
                ordered.append(values[ordinal])
    return ordered


def select_mnsol_partition(
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
    pilot_manifest: Mapping[str, object],
    *,
    partition: str,
) -> tuple[MNSolPartitionSelection, ...]:
    """Return every eligible partition row in deterministic audit order."""

    if partition not in SUPPORTED_PARTITIONS:
        raise ValueError(
            "MNSol partition selection requires development or confirmation."
        )
    pilot = validate_frozen_mnsol_pilot_selection(
        pilot_manifest,
        dataset,
        protocol,
    )
    pilot_handles = {item.eligible_record.record.geometry_handle for item in pilot}
    selected = [
        MNSolPartitionSelection(
            canonical_solvent=item.canonical_solvent,
            eligible_record=item,
            opaque_record_id=_opaque_record_id(item),
            selection_score_sha256=_selection_score(
                item,
                partition=partition,
            ),
            prior_pilot_geometry_overlap=(item.record.geometry_handle in pilot_handles),
        )
        for item in eligible_mnsol_records(dataset, protocol)
        if item.partition == partition
    ]
    solvent_order = _solvent_order(protocol)
    no_pilot_overlap = _round_robin(
        [item for item in selected if not item.prior_pilot_geometry_overlap],
        solvent_order,
    )
    overlap = _round_robin(
        [item for item in selected if item.prior_pilot_geometry_overlap],
        solvent_order,
    )
    ordered = tuple(no_pilot_overlap + overlap)
    if len({item.opaque_record_id for item in ordered}) != len(ordered):
        raise RuntimeError("MNSol partition selection produced duplicate rows.")
    return ordered


def build_mnsol_partition_selection_manifest(
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
    pilot_manifest: Mapping[str, object],
    *,
    partition: str,
) -> dict[str, object]:
    """Build an aggregate-safe manifest for a complete MNSol partition."""

    selection = select_mnsol_partition(
        dataset,
        protocol,
        pilot_manifest,
        partition=partition,
    )
    overlap = [item for item in selection if item.prior_pilot_geometry_overlap]
    counts = Counter(item.canonical_solvent for item in selection)
    no_pilot_overlap_counts = Counter(
        item.canonical_solvent
        for item in selection
        if not item.prior_pilot_geometry_overlap
    )
    overlap_counts = Counter(
        item.canonical_solvent
        for item in selection
        if item.prior_pilot_geometry_overlap
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": PARTITION_ARTIFACT,
        "selection_status": "frozen-before-partition-run",
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "partition": partition,
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
        },
        "prior_pilot_selection_fingerprint": pilot_manifest["selection_fingerprint"],
        "record_count": len(selection),
        "unique_geometry_count": len(
            {item.eligible_record.record.geometry_handle for item in selection}
        ),
        "solvent_record_counts": {
            solvent: counts.get(solvent, 0) for solvent in _solvent_order(protocol)
        },
        "prior_pilot_overlap": {
            "classification_key": "FileHandle/solute geometry",
            "record_count": len(overlap),
            "unique_geometry_count": len(
                {item.eligible_record.record.geometry_handle for item in overlap}
            ),
            "no_prior_pilot_geometry_overlap_record_count": (
                len(selection) - len(overlap)
            ),
            "no_prior_pilot_geometry_overlap_solvent_record_counts": {
                solvent: no_pilot_overlap_counts.get(solvent, 0)
                for solvent in _solvent_order(protocol)
            },
            "overlap_solvent_record_counts": {
                solvent: overlap_counts.get(solvent, 0)
                for solvent in _solvent_order(protocol)
            },
        },
        "ordering": {
            "strategy": PARTITION_ORDERING,
            "selection_seed": PARTITION_SELECTION_SEED,
            "strata": [
                "no-prior-pilot-geometry-overlap",
                "prior-pilot-geometry-overlap",
            ],
            "solvent_order": list(_solvent_order(protocol)),
            "within_solvent_score": (
                "sha256(selection_seed\\0partition\\0canonical_solvent"
                "\\0geometry_handle\\0entry_number)"
            ),
            "uses_experimental_values": False,
            "uses_model_outputs": False,
            "excludes_records": False,
        },
        "selected_records": [
            {
                "selection_index": index,
                "canonical_solvent": item.canonical_solvent,
                "partition": item.eligible_record.partition,
                "opaque_record_id": item.opaque_record_id,
                "geometry_sha256": item.eligible_record.geometry.sha256,
                "atom_count": len(item.eligible_record.geometry.atomic_numbers),
                "prior_pilot_geometry_overlap": (item.prior_pilot_geometry_overlap),
            }
            for index, item in enumerate(selection)
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
            "This manifest retains every eligible row in the frozen partition "
            "and only defines an experiment-blind execution order. Rows that "
            "share a solute geometry with the earlier ten-record pilot are "
            "explicitly marked and cannot be presented as untouched "
            "confirmation evidence. The manifest contains no row-level "
            "experimental values or model outputs."
        ),
    }
    manifest["selection_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def validate_frozen_mnsol_partition_selection(
    manifest: Mapping[str, object],
    dataset: MNSolDataset,
    protocol: MNSolProtocol,
    pilot_manifest: Mapping[str, object],
) -> tuple[MNSolPartitionSelection, ...]:
    """Recompute a partition manifest and reject any data or policy drift."""

    if manifest.get("artifact") != PARTITION_ARTIFACT:
        raise ValueError("Unsupported MNSol partition selection artifact.")
    partition = str(manifest.get("partition", ""))
    expected = build_mnsol_partition_selection_manifest(
        dataset,
        protocol,
        pilot_manifest,
        partition=partition,
    )
    if dict(manifest) != expected:
        raise ValueError(
            "Frozen MNSol partition selection does not match the pinned "
            "dataset, protocol, pilot-overlap classification, and ordering."
        )
    return select_mnsol_partition(
        dataset,
        protocol,
        pilot_manifest,
        partition=partition,
    )


def indexed_partition_record(
    selection: Sequence[object],
    record_index: int | None,
) -> list[tuple[int, object]]:
    """Require one explicit bounded record from a potentially large partition."""

    if record_index is None:
        raise ValueError(
            "Complete MNSol partition runs require an explicit --record-index; "
            "iterate bounded records or use a separately reviewed orchestrator."
        )
    if not 0 <= record_index < len(selection):
        raise ValueError(f"--record-index must lie in [0, {len(selection) - 1}].")
    return [(record_index, selection[record_index])]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a complete, shardable MNSol partition selection."
    )
    parser.add_argument("action", choices=("freeze",))
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--pilot-selection", required=True)
    parser.add_argument(
        "--partition",
        choices=sorted(SUPPORTED_PARTITIONS),
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol = load_mnsol_protocol(args.protocol)
    dataset = load_mnsol_v2012(args.source, protocol)
    pilot_manifest = json.loads(Path(args.pilot_selection).read_text(encoding="utf-8"))
    manifest = build_mnsol_partition_selection_manifest(
        dataset,
        protocol,
        pilot_manifest,
        partition=args.partition,
    )
    write_json_atomic(args.output, manifest)
    print(
        f"Frozen {manifest['record_count']} complete MNSol "
        f"{args.partition} records without using experiment values or model "
        "outputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
