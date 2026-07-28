#!/usr/bin/env python3
"""Validate and aggregate preregistered openCOSMO-RS MNSol fragments.

This companion never launches ORCA.  It independently rebinds every private
fragment to the frozen MNSol selection and PCM-family baseline, re-renders each
input, reparses each main output, and rechecks all child outputs and immutable
asset hashes before emitting a complete aggregate-only public artifact.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ase.data import chemical_symbols

from benchmark_core import sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import (
    MNSolPilotSelection,
    validate_frozen_mnsol_pilot_selection,
)
import run_mnsol_opencosmors24a_fixed_geometry as runner
from maple.function.cosmo_rs import (
    OpenCOSMORS24aInputBundle,
    parse_orca_opencosmors_solvation_output,
    render_orca_opencosmors24a_input,
    validate_orca_opencosmors_completion,
)

PUBLIC_ARTIFACT_PATH = (
    BENCHMARK_DIR / "route2-mnsol-opencosmors24a-fixed-geometry-v1.json"
)
_NORMAL_TERMINATION = "****ORCA TERMINATED NORMALLY****"
_FAILURE_MARKERS = ("error termination", "unable to open file")
_NUMERIC_TOLERANCE = 1.0e-12


def _validate_fragment_header(
    fragment: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    protocol_fingerprint: str,
    selection_fingerprint: str,
    dataset_hashes: Mapping[str, str | None],
    baseline_sha256: str,
) -> None:
    expected = {
        "artifact": runner.ARTIFACT_NAME,
        "schema_version": 1,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "status": "complete",
        "complete_panel": False,
        "preregistration_sha256": runner.PREREGISTRATION_SHA256,
        "protocol_fingerprint": protocol_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "dataset": dict(dataset_hashes),
        "comparison_baseline_sha256": baseline_sha256,
    }
    for field, value in expected.items():
        if fragment.get(field) != value:
            raise ValueError(f"Private fragment field {field!r} drifted.")
    head = fragment.get("execution_git_head")
    if (
        not isinstance(head, str)
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise ValueError("Private fragment execution Git head is invalid.")
    runtime = fragment.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Private fragment runtime metadata is missing.")
    runtime_expected = {
        "orca_sha256": runner.ORCA_SHA256,
        "orca_version": runner.ORCA_VERSION,
        "opencosmors_sha256": runner.OPEN_COSMORS_SHA256,
        "nprocs": preregistration["runtime"]["nprocs"],
        "maxcore_mb": preregistration["runtime"]["maxcore_mb"],
        "timeout_seconds_per_record": preregistration["runtime"][
            "timeout_seconds_per_record"
        ],
    }
    for field, value in runtime_expected.items():
        if runtime.get(field) != value:
            raise ValueError(f"Private fragment runtime field {field!r} drifted.")


def _assert_close(observed: object, expected: float, *, label: str) -> None:
    value = float(observed)
    if not math.isfinite(value) or not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=_NUMERIC_TOLERANCE,
    ):
        raise ValueError(f"{label} drifted.")


def _validate_record_assets(
    record: Mapping[str, Any],
    selected: MNSolPilotSelection,
    baseline_record: Mapping[str, Any],
    *,
    selection_index: int,
    preregistration: Mapping[str, Any],
) -> None:
    runner._validate_record_binding(
        record,
        selected,
        selection_index=selection_index,
    )
    solvent_alias = preregistration["method"]["solvent_aliases"][
        selected.canonical_solvent
    ]
    if record.get("opencosmors_solvent_alias") != solvent_alias:
        raise ValueError("openCOSMO-RS solvent alias drifted.")

    geometry = selected.eligible_record.geometry
    symbols = tuple(chemical_symbols[number] for number in geometry.atomic_numbers)
    stem = f"record-{selection_index:02d}"
    manifest = record.get("input_bundle")
    if not isinstance(manifest, Mapping):
        raise ValueError("Private record omitted its input bundle.")
    gas_path = (
        Path(manifest["assets"]["solute_gas_output"]["path"]).expanduser().resolve()
    )
    workdir = runner._require_private_path(
        gas_path.parent,
        label="ORCA fragment work directory",
    )
    bundle = OpenCOSMORS24aInputBundle.from_orca_run(workdir, stem)
    if bundle.as_manifest() != dict(manifest):
        raise ValueError("Reconstructed openCOSMO-RS input bundle drifted.")

    input_path = workdir / f"{stem}.inp"
    main_output_path = workdir / f"{stem}.out"
    stderr_path = workdir / f"{stem}.stderr"
    expected_input = render_orca_opencosmors24a_input(
        symbols,
        geometry.coordinates_angstrom,
        solvent_alias=solvent_alias,
        maxcore_mb=preregistration["runtime"]["maxcore_mb"],
        nprocs=1,
    )
    if input_path.read_text(encoding="utf-8") != expected_input:
        raise ValueError("Stored ORCA input differs from deterministic rendering.")
    for path, field in (
        (input_path, "input_sha256"),
        (main_output_path, "main_output_sha256"),
        (stderr_path, "stderr_sha256"),
    ):
        if sha256_file(path) != record.get(field):
            raise ValueError(f"Private record {field!r} drifted.")

    main_output = main_output_path.read_text(encoding="utf-8", errors="replace")
    validate_orca_opencosmors_completion(main_output)
    parsed = parse_orca_opencosmors_solvation_output(main_output, inputs=bundle)
    prediction = float(parsed.delta_g_solvation_kcal_mol)
    experiment = float(selected.eligible_record.record.delta_g_kcal_mol)
    signed_error = prediction - experiment
    _assert_close(
        record["opencosmors_delta_g_kcal_mol"],
        prediction,
        label="openCOSMO-RS prediction",
    )
    _assert_close(
        record["signed_error_kcal_mol"],
        signed_error,
        label="openCOSMO-RS signed error",
    )
    _assert_close(
        record["absolute_error_kcal_mol"],
        abs(signed_error),
        label="openCOSMO-RS absolute error",
    )
    wall_seconds = float(record["wall_seconds"])
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise ValueError("Private record wall time must be finite and positive.")

    expected_children = {
        f"{stem}.solute_vac.lastout",
        f"{stem}.solute_cpcm.lastout",
        f"{stem}.solvent_cpcm.lastout",
    }
    observed_children = {path.name for path in workdir.glob("*.lastout")}
    if observed_children != expected_children:
        raise ValueError("ORCA child-output set drifted.")
    child_hashes = record.get("child_output_sha256")
    if not isinstance(child_hashes, Mapping) or set(child_hashes) != expected_children:
        raise ValueError("Private child-output hash ledger drifted.")
    for child_name in expected_children:
        child = workdir / child_name
        output = child.read_text(encoding="utf-8", errors="replace")
        if output.upper().count(_NORMAL_TERMINATION) != 1:
            raise ValueError(f"{child_name} did not terminate normally once.")
        lowered = output.lower()
        if any(marker in lowered for marker in _FAILURE_MARKERS):
            raise ValueError(f"{child_name} contains an ORCA failure marker.")
        if sha256_file(child) != child_hashes[child_name]:
            raise ValueError(f"{child_name} hash drifted.")

    comparisons = record.get("baseline_comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(
        runner.BASELINE_METHOD_IDS
    ):
        raise ValueError("Private baseline-comparison method set drifted.")
    for method_id in runner.BASELINE_METHOD_IDS:
        baseline = baseline_record["methods"][method_id]
        baseline_prediction = float(baseline["total_solvation_kcal_mol"])
        baseline_absolute_error = float(baseline["absolute_error_kcal_mol"])
        comparison = comparisons[method_id]
        _assert_close(
            comparison["baseline_total_solvation_kcal_mol"],
            baseline_prediction,
            label=f"{method_id} baseline prediction",
        )
        _assert_close(
            comparison["baseline_absolute_error_kcal_mol"],
            baseline_absolute_error,
            label=f"{method_id} baseline absolute error",
        )
        _assert_close(
            comparison["opencosmors_minus_baseline_prediction_kcal_mol"],
            prediction - baseline_prediction,
            label=f"{method_id} paired prediction difference",
        )
        _assert_close(
            comparison["opencosmors_minus_baseline_absolute_error_kcal_mol"],
            abs(signed_error) - baseline_absolute_error,
            label=f"{method_id} paired absolute-error difference",
        )


def _load_fragments(
    paths: Sequence[Path],
    *,
    preregistration: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    selection: Sequence[MNSolPilotSelection],
    protocol_fingerprint: str,
    dataset_hashes: Mapping[str, str | None],
    baseline_path: Path,
    baseline_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, object]], str]:
    baseline_sha256 = sha256_file(baseline_path)
    indexed: dict[int, dict[str, Any]] = {}
    provenance = []
    execution_heads = set()
    for path in paths:
        private_path = runner._require_private_path(
            path,
            label="openCOSMO-RS private fragment",
        )
        fragment = runner._load_json_mapping(
            private_path,
            label="openCOSMO-RS private fragment",
        )
        _validate_fragment_header(
            fragment,
            preregistration=preregistration,
            protocol_fingerprint=protocol_fingerprint,
            selection_fingerprint=selection_manifest["selection_fingerprint"],
            dataset_hashes=dataset_hashes,
            baseline_sha256=baseline_sha256,
        )
        records = fragment.get("records")
        if (
            not isinstance(records, list)
            or fragment.get("completed_record_count") != len(records)
            or not records
        ):
            raise ValueError("Private fragment record count drifted.")
        fragment_indices = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Private fragment records must be JSON objects.")
            index = record.get("selection_index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(selection)
                or index in indexed
            ):
                raise ValueError("Private fragment selection indices are invalid.")
            _validate_record_assets(
                record,
                selection[index],
                baseline_records[index],
                selection_index=index,
                preregistration=preregistration,
            )
            indexed[index] = record
            fragment_indices.append(index)
        execution_heads.add(fragment["execution_git_head"])
        provenance.append(
            {
                "sha256": sha256_file(private_path),
                "execution_git_head": fragment["execution_git_head"],
                "record_count": len(records),
                "selection_indices": sorted(fragment_indices),
            }
        )
    if set(indexed) != set(range(runner.FULL_PANEL_RECORD_COUNT)):
        raise ValueError(
            "Private fragments do not cover the complete ten-record panel."
        )
    if len(execution_heads) != 1:
        raise ValueError("Private fragments were produced by different code commits.")
    records = [indexed[index] for index in range(runner.FULL_PANEL_RECORD_COUNT)]
    return records, provenance, execution_heads.pop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=runner.PREREGISTRATION_PATH,
    )
    parser.add_argument(
        "--fragment",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every private fragment included in the aggregation.",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument(
        "--public-output",
        type=Path,
        default=PUBLIC_ARTIFACT_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preregistration = runner._validate_preregistration(args.preregistration)
    aggregation_git_head = runner._execution_git_head()
    if sha256_file(args.protocol) != runner.PROTOCOL_ARTIFACT_SHA256:
        raise ValueError("MNSol protocol artifact hash drifted.")
    if sha256_file(args.selection) != runner.SELECTION_ARTIFACT_SHA256:
        raise ValueError("MNSol selection artifact hash drifted.")
    protocol = load_mnsol_protocol(args.protocol)
    if protocol.fingerprint != runner.PROTOCOL_FINGERPRINT:
        raise ValueError("MNSol protocol fingerprint drifted.")
    dataset = load_mnsol_v2012(args.source, protocol)
    selection_manifest = runner._load_json_mapping(
        args.selection,
        label="MNSol pilot selection",
    )
    selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest,
        dataset,
        protocol,
    )
    dataset_hashes = runner._dataset_hashes(dataset)
    baseline_records = runner._load_baseline_records(
        args.baseline_artifact,
        preregistration=preregistration,
        selection_manifest=selection_manifest,
        selection=selection,
        protocol_fingerprint=protocol.fingerprint,
        dataset_hashes=dataset_hashes,
    )
    experimental_checks = runner._validate_experimental_selection(
        selection,
        temperature_k=protocol.temperature_k,
        standard_state=protocol.standard_state,
    )
    records, fragment_provenance, record_execution_head = _load_fragments(
        args.fragment,
        preregistration=preregistration,
        selection_manifest=selection_manifest,
        selection=selection,
        protocol_fingerprint=protocol.fingerprint,
        dataset_hashes=dataset_hashes,
        baseline_path=args.baseline_artifact,
        baseline_records=baseline_records,
    )
    private_output = runner._require_private_path(
        args.private_output,
        label="Combined row-level openCOSMO-RS output",
    )
    public_output = args.public_output.expanduser().resolve()
    if public_output != PUBLIC_ARTIFACT_PATH.resolve():
        raise ValueError(
            "Public aggregate path must match the versioned artifact path."
        )

    aggregate_metrics = runner._aggregate_metrics(records)
    paired_comparisons = runner._paired_comparisons(records)
    runtime = dict(
        runner._load_json_mapping(
            args.fragment[0],
            label="openCOSMO-RS private fragment",
        )["runtime"]
    )
    private = runner._private_artifact(
        execution_git_head=record_execution_head,
        preregistration=preregistration,
        protocol_fingerprint=protocol.fingerprint,
        selection_manifest=selection_manifest,
        dataset_hashes=dataset_hashes,
        records=records,
        status="complete",
        complete_panel=True,
        baseline_path=args.baseline_artifact,
        runtime=runtime,
    )
    private.update(
        {
            "aggregation_git_head": aggregation_git_head,
            "fragment_provenance": fragment_provenance,
            "aggregate_metrics": aggregate_metrics,
            "paired_method_comparisons": paired_comparisons,
            "summed_record_wall_seconds": aggregate_metrics["total_wall_seconds"],
        }
    )
    write_json_atomic(private_output, private)

    public = runner._public_artifact(
        execution_git_head=record_execution_head,
        preregistration=preregistration,
        protocol=protocol,
        selection_manifest=selection_manifest,
        dataset=dataset,
        experimental_checks=experimental_checks,
        records=records,
        complete_panel=True,
        aggregate_metrics=aggregate_metrics,
        paired_comparisons=paired_comparisons,
        actual_total_wall_seconds=aggregate_metrics["total_wall_seconds"],
        runtime=runtime,
    )
    public.update(
        {
            "aggregation_git_head": aggregation_git_head,
            "record_execution_git_head": record_execution_head,
            "fragment_provenance": fragment_provenance,
            "aggregation_contract": (
                "Ten independently validated private records were merged "
                "without rerunning ORCA; only aggregate statistics are public."
            ),
        }
    )
    public["timing_seconds"] = {
        "summed_record_wall": aggregate_metrics["total_wall_seconds"],
        "aggregation_overhead_included": False,
        "status": "metadata-only-serial-fixed-order-not-a-speed-ranking",
    }
    relative = (
        "docs/implicit-solvation/benchmarks/"
        "aggregate_mnsol_opencosmors24a_fixed_geometry.py"
    )
    public["source_files_sha256"][relative] = sha256_file(REPO_ROOT / relative)
    write_json_atomic(public_output, public)
    print(
        f"Validated and aggregated {len(records)} records into "
        f"{private_output} and {public_output}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
