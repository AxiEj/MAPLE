"""Source-backed derivation of private label-free MNSol inputs."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence, cast

from tools.route2_release._secure_artifacts import captured_named_file


class _Geometry(Protocol):
    atomic_numbers: Sequence[int]
    coordinates_angstrom: Sequence[Sequence[float]]
    multiplicity: int
    sha256: str


class _Record(Protocol):
    charge: int


class _Eligible(Protocol):
    geometry: _Geometry
    record: _Record
    partition: str


class _Selected(Protocol):
    canonical_solvent: str
    eligible_record: _Eligible
    opaque_record_id: str


class _Protocol(Protocol):
    protocol_id: str
    fingerprint: str
    temperature_k: float
    standard_state: str


class _Dataset(Protocol):
    table_sha256: str
    normalized_bundle_sha256: str


def _benchmark_modules(benchmark_directory: Path) -> tuple[Any, Any]:
    benchmark = str(benchmark_directory)
    if benchmark not in sys.path:
        sys.path.insert(0, benchmark)
    return (
        importlib.import_module("mnsol_dataset"),
        importlib.import_module("mnsol_pilot"),
    )


def derive_mnsol10_label_free_inputs(
    *,
    source_bytes: bytes,
    protocol_path: Path,
    selection_payload: Mapping[str, object],
    benchmark_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Rebuild the exact ten private inputs from captured distribution bytes."""

    dataset_module, pilot_module = _benchmark_modules(benchmark_directory)
    accuracy = importlib.import_module("maple.solvation.release.accuracy_admission")
    protocol = cast(_Protocol, dataset_module.load_mnsol_protocol(protocol_path))
    with captured_named_file(
        source_bytes, suffix=".zip", role="MNSol distribution"
    ) as source_path:
        dataset = cast(_Dataset, dataset_module.load_mnsol_v2012(source_path, protocol))
    selection = cast(
        Sequence[_Selected],
        pilot_module.validate_frozen_mnsol_pilot_selection(
            selection_payload, dataset, protocol
        ),
    )
    if len(selection) != 10:
        raise ValueError("frozen MNSol-10 selection is incomplete")
    records: list[dict[str, object]] = []
    for index, selected in enumerate(selection):
        eligible = selected.eligible_record
        geometry = eligible.geometry
        numbers = [int(value) for value in geometry.atomic_numbers]
        positions = [
            [float(component) for component in row]
            for row in geometry.coordinates_angstrom
        ]
        record: dict[str, object] = {
            "selection_index": index,
            "opaque_record_id": str(selected.opaque_record_id),
            "canonical_solvent": str(selected.canonical_solvent),
            "partition": str(eligible.partition),
            "geometry_sha256": str(geometry.sha256),
            "normalized_geometry_sha256": accuracy.normalized_geometry_sha256(
                atomic_numbers=numbers,
                positions_angstrom=positions,
                charge=int(eligible.record.charge),
                multiplicity=int(geometry.multiplicity),
            ),
            "atom_count": len(numbers),
            "atomic_numbers": numbers,
            "positions_angstrom": positions,
            "charge": int(eligible.record.charge),
            "multiplicity": int(geometry.multiplicity),
        }
        records.append(record)
    dataset_identity = {
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
        "temperature_k": float(protocol.temperature_k),
        "standard_state": protocol.standard_state,
    }
    return dataset_identity, records


__all__ = ["derive_mnsol10_label_free_inputs"]
