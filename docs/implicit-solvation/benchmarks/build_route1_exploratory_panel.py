#!/usr/bin/env python3
"""Build a deterministic, label-free MNSol exploratory source panel.

This prepares source identities only.  It performs no energy calculation,
scoring, fitting, or accuracy qualification.  Row-level output is restricted
to the repository-local ``.omx`` directory because MNSol is license controlled.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]
import build_route1_mnsol_label_free_manifest as manifest_builder  # pyright: ignore[reportImplicitRelativeImport]

RECORD_FIELDS = {
    "data_row_ordinal",
    "file_handle",
    "charge",
    "solvent_mnsol_name",
    "solvent_canonical_name",
    "process_type",
    "source_table_bound_record_identity_sha256",
    "label_independent_record_identity_sha256",
    "geometry_member",
    "geometry_sha256",
    "atom_count",
    "element_sequence_sha256",
    "multiplicity",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch


def _positive_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("requested_per_solvent must be a positive integer.")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX_SHA256(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def _validated_manifest(manifest: object) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise TypeError("Each source manifest must be a JSON object.")
    artifact = cast(dict[str, Any], manifest)
    manifest_builder._forbid_value_fields(artifact)
    if artifact.get("artifact_type") != "route1-mnsol-label-free-record-manifest-v1":
        raise ValueError("Unexpected source-manifest artifact type.")
    if artifact.get("content_sha256") != core.artifact_content_sha256(artifact):
        raise ValueError("Source-manifest content seal is invalid.")

    selection = artifact.get("selection")
    if not isinstance(selection, dict):
        raise TypeError("Source-manifest selection must be an object.")
    canonical_solvent = selection.get("canonical_solvent")
    mnsol_solvent = selection.get("mnsol_solvent")
    if not isinstance(canonical_solvent, str) or not canonical_solvent:
        raise ValueError("Source-manifest canonical solvent is invalid.")
    if not isinstance(mnsol_solvent, str) or not mnsol_solvent:
        raise ValueError("Source-manifest MNSol solvent is invalid.")
    if selection.get("charge") != "0" or selection.get("process_type") != "abs":
        raise ValueError("Source manifest is not neutral absolute-solvation data.")

    records = artifact.get("records")
    count = artifact.get("record_count")
    if (
        not isinstance(records, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count != len(records)
        or selection.get("expected_complete_denominator") != count
    ):
        raise ValueError("Source-manifest record count is invalid.")

    handles: set[str] = set()
    geometries: set[str] = set()
    identities: set[str] = set()
    for index, record_obj in enumerate(records):
        if not isinstance(record_obj, dict) or set(record_obj) != RECORD_FIELDS:
            raise ValueError(f"Source-manifest record {index} schema is invalid.")
        record = cast(dict[str, Any], record_obj)
        handle = record.get("file_handle")
        geometry_sha256 = _validate_sha256(
            record.get("geometry_sha256"), f"records[{index}].geometry_sha256"
        )
        identity_sha256 = _validate_sha256(
            record.get("label_independent_record_identity_sha256"),
            f"records[{index}].label_independent_record_identity_sha256",
        )
        _validate_sha256(
            record.get("source_table_bound_record_identity_sha256"),
            f"records[{index}].source_table_bound_record_identity_sha256",
        )
        _validate_sha256(
            record.get("element_sequence_sha256"),
            f"records[{index}].element_sequence_sha256",
        )
        if not isinstance(handle, str) or not handle:
            raise ValueError(
                f"Source-manifest record {index} has an invalid FileHandle."
            )
        if (
            record.get("charge") != "0"
            or record.get("process_type") != "abs"
            or record.get("solvent_canonical_name") != canonical_solvent
            or record.get("solvent_mnsol_name") != mnsol_solvent
        ):
            raise ValueError("Source-manifest record changed the selected scope.")
        if (
            handle in handles
            or geometry_sha256 in geometries
            or identity_sha256 in identities
        ):
            raise ValueError(
                "Source manifest contains duplicate solutes or identities."
            )
        handles.add(handle)
        geometries.add(geometry_sha256)
        identities.add(identity_sha256)

    label_independent_record_set = [
        {
            key: value
            for key, value in cast(dict[str, Any], record).items()
            if key != "source_table_bound_record_identity_sha256"
        }
        for record in records
    ]
    expected_record_set_sha256 = core.sha256_bytes(
        core.canonical_json_bytes(label_independent_record_set)
    )
    if artifact.get("record_set_sha256") != expected_record_set_sha256:
        raise ValueError("Source-manifest record-set seal is invalid.")
    return artifact


def assemble_panel(
    manifests: list[dict[str, Any]],
    *,
    requested_per_solvent: int = 20,
    canonical_solvent_order: list[str] | None = None,
) -> dict[str, Any]:
    """Validate sealed manifests and select stable-hash-first source records."""
    requested = _positive_count(requested_per_solvent)
    validated = [_validated_manifest(manifest) for manifest in manifests]
    by_solvent: dict[str, dict[str, Any]] = {}
    for manifest in validated:
        name = cast(str, manifest["selection"]["canonical_solvent"])
        if name in by_solvent:
            raise ValueError(f"Duplicate source manifest for solvent {name!r}.")
        by_solvent[name] = manifest
    if not by_solvent:
        raise ValueError("At least one source manifest is required.")

    order = canonical_solvent_order or sorted(by_solvent)
    if len(order) != len(set(order)) or set(order) != set(by_solvent):
        raise ValueError("Canonical solvent order does not match source manifests.")

    first = by_solvent[order[0]]
    common_fields = ("protocol_id", "protocol_fingerprint", "route1_boundary")
    for manifest in validated:
        if any(manifest.get(field) != first.get(field) for field in common_fields):
            raise ValueError("Source manifests do not share one frozen protocol.")
        if manifest.get("source_audit") != first.get("source_audit"):
            raise ValueError("Source manifests do not share one source audit.")
        if manifest.get("source_archive") != first.get("source_archive"):
            raise ValueError("Source manifests do not share one source archive.")
        if manifest.get("source_table") != first.get("source_table"):
            raise ValueError("Source manifests do not share one source table.")

    solvent_panels: list[dict[str, Any]] = []
    total_available = 0
    total_selected = 0
    for canonical_solvent in order:
        manifest = by_solvent[canonical_solvent]
        records = cast(list[dict[str, Any]], manifest["records"])
        selected = sorted(
            records,
            key=lambda record: (
                record["label_independent_record_identity_sha256"],
                record["file_handle"],
            ),
        )[:requested]
        available = len(records)
        selected_count = len(selected)
        shortfall = requested - selected_count
        total_available += available
        total_selected += selected_count
        solvent_panels.append(
            {
                "canonical_solvent": canonical_solvent,
                "mnsol_solvent": manifest["selection"]["mnsol_solvent"],
                "requested_count": requested,
                "available_count": available,
                "selected_count": selected_count,
                "shortfall_count": shortfall,
                "source_manifest": {
                    "content_sha256": manifest["content_sha256"],
                    "record_set_sha256": manifest["record_set_sha256"],
                },
                "selected_record_set_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(selected)
                ),
                "records": selected,
            }
        )

    solvent_count = len(solvent_panels)
    full_count = sum(panel["shortfall_count"] == 0 for panel in solvent_panels)
    artifact: dict[str, Any] = {
        "artifact_type": "route1-mnsol-exploratory-panel-v1",
        "claim_scope": (
            "Deterministic, label-free, local source selection only; no energy, "
            "score, fit, endpoint, task-support, or accuracy-completion claim."
        ),
        "protocol_id": first["protocol_id"],
        "protocol_fingerprint": first["protocol_fingerprint"],
        "route1_boundary": first["route1_boundary"],
        "source_archive": first["source_archive"],
        "source_table": first["source_table"],
        "selection_policy": {
            "eligible_charge": "0",
            "eligible_process_type": "abs",
            "requested_per_solvent": requested,
            "record_order": (
                "ascending_label_independent_record_identity_sha256_then_file_handle"
            ),
            "solvent_order": "frozen_source_protocol_panel_order",
            "replacement_after_evaluation_allowed": False,
        },
        "counts": {
            "solvent_count": solvent_count,
            "requested_count": requested * solvent_count,
            "available_count": total_available,
            "selected_count": total_selected,
            "shortfall_count": requested * solvent_count - total_selected,
            "full_solvent_count": full_count,
            "shortfall_solvent_count": solvent_count - full_count,
        },
        "containment": {
            "experimental_value_columns_decoded": [],
            "experimental_values_interpreted": False,
            "experimental_values_used": False,
            "fit_or_tuning_performed": False,
            "intended_storage": "local_omx_only_not_for_redistribution",
        },
        "solvents": solvent_panels,
    }
    manifest_builder._forbid_value_fields(artifact)
    return core.seal_artifact(artifact)


def build_panel(
    *,
    protocol_path: str | Path,
    source_audit_path: str | Path,
    archive_path: str | Path,
    requested_per_solvent: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all frozen per-solvent manifests and their exploratory selection."""
    protocol, _fingerprint = manifest_builder.source_audit._load_protocol(protocol_path)
    order = [entry["canonical_name"] for entry in protocol["panel"]]
    manifests = [
        manifest_builder.build_manifest(
            protocol_path=protocol_path,
            source_audit_path=source_audit_path,
            archive_path=archive_path,
            canonical_solvent=canonical_solvent,
        )
        for canonical_solvent in order
    ]
    return manifests, assemble_panel(
        manifests,
        requested_per_solvent=requested_per_solvent,
        canonical_solvent_order=order,
    )


def _local_output_directory(path: Path) -> Path:
    marker = manifest_builder._local_omx_output_path(path / "panel.json")
    return marker.parent


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--requested-per-solvent", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = _local_output_directory(args.output_dir)
    manifests, panel = build_panel(
        protocol_path=args.protocol,
        source_audit_path=args.source_audit,
        archive_path=args.archive,
        requested_per_solvent=args.requested_per_solvent,
    )
    manifest_dir = output_dir / "manifests"
    for index, manifest in enumerate(manifests, start=1):
        name = cast(str, manifest["selection"]["canonical_solvent"])
        core.write_json_atomic(
            manifest_dir / f"{index:02d}-{_slug(name)}.json", manifest
        )

    payload = {key: value for key, value in panel.items() if key != "content_sha256"}
    payload["command_provenance"] = core.command_provenance(
        __file__,
        {
            "protocol": manifest_builder._repo_reference(args.protocol),
            "source_audit": args.source_audit.name,
            "source_audit_path_recorded": False,
            "archive_name": args.archive.name,
            "archive_sha256": core.sha256_file(args.archive),
            "archive_path_recorded": False,
            "requested_per_solvent": args.requested_per_solvent,
            "output": manifest_builder._repo_reference(output_dir / "panel.json"),
        },
        repository_root=REPOSITORY_ROOT,
    )
    core.write_json_atomic(output_dir / "panel.json", core.seal_artifact(payload))


if __name__ == "__main__":
    main()
