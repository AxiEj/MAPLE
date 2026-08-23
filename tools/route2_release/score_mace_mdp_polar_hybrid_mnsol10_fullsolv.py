#!/usr/bin/env python3
"""Mechanically score one frozen label-free hybrid MNSol-10 prediction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._secure_artifacts import (  # noqa: E402
    CapturedFile,
    SecureArtifactError,
    StabilityGuard,
    capture_clean_repository,
    capture_file,
    capture_repo_files,
    captured_named_file,
    load_json_bytes,
    publish_json_noreplace,
)
from tools.route2_release._hybrid_mnsol10_chain import (
    capture_prediction_chain,
)  # noqa: E402

BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
PROTOCOL = BENCHMARK_DIR / "route2-mnsol-protocol-v1.json"
SELECTION = BENCHMARK_DIR / "route2-mnsol-pilot-selection-v1.json"
RUN_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/runs"
SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
INPUT_BUNDLE = REPO_ROOT / ".omx/route2/hybrid-mnsol10/label-free-input-v1.json"
ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-score-v1"


class _Geometry(Protocol):
    atomic_numbers: Sequence[int]
    coordinates_angstrom: Sequence[Sequence[float]]
    multiplicity: int
    sha256: str


class _DatabaseRecord(Protocol):
    charge: int
    delta_g_kcal_mol: float


class _Eligible(Protocol):
    geometry: _Geometry
    record: _DatabaseRecord
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
    source_artifact_sha256: str | None


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SecureArtifactError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecureArtifactError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise SecureArtifactError(f"{name} must be finite")
    return result


def _load_captured_module(capture: CapturedFile, *, module_name: str) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(capture.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(capture.data, str(capture.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _benchmark_modules() -> tuple[Any, Any]:
    path = str(BENCHMARK_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("mnsol_dataset"), importlib.import_module(
        "mnsol_pilot"
    )


def _require_model_free_process() -> None:
    forbidden = (
        "maple.solvation.models",
        "maple.solvation.continuum",
        "maple.solvation.coupling",
        "mace",
        "graph_longrange",
        "torch",
    )
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    if loaded:
        raise SecureArtifactError(
            "mechanical scorer process contains model/continuum modules: "
            + ", ".join(loaded[:12])
        )


def score(
    *, source_path: str | Path, prediction_path: str | Path
) -> tuple[dict[str, object], StabilityGuard, Path]:
    _require_model_free_process()
    repository = capture_clean_repository(REPO_ROOT)
    preliminary_prediction = capture_file(
        prediction_path, role="prediction terminal preflight"
    )
    preliminary_payload = load_json_bytes(
        preliminary_prediction.data, role="prediction terminal preflight"
    )
    execution_id = str(preliminary_payload.get("execution_id"))
    preliminary_seal = capture_file(
        SEAL_DIRECTORY / f"{execution_id}.json",
        role="computation seal preflight",
    )
    seal_payload = load_json_bytes(
        preliminary_seal.data, role="computation seal preflight"
    )
    source_hashes = _mapping(
        seal_payload.get("source_files_sha256"), name="seal source ledger"
    )
    sources = capture_repo_files(repository, source_hashes.keys())
    accuracy = _load_captured_module(
        sources["maple/solvation/release/accuracy_admission.py"],
        module_name="_maple_route2_captured_accuracy_scorer",
    )
    chain = capture_prediction_chain(
        repo_root=REPO_ROOT,
        prediction_path=prediction_path,
        seal_directory=SEAL_DIRECTORY,
        run_directory=RUN_DIRECTORY,
        input_bundle_path=INPUT_BUNDLE,
        accuracy=accuracy,
    )
    source_capture = capture_file(source_path, role="MNSol distribution")
    stability = StabilityGuard(
        chain.repository,
        (*chain.stability.captures, ("MNSol distribution", source_capture)),
    )
    prediction = chain.prediction
    if prediction.get("status") != "complete":
        raise SecureArtifactError("a failed prediction terminal cannot be scored")
    execution_id = chain.execution_id

    protocol_capture = chain.source_captures[
        "docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json"
    ]
    selection_capture = chain.source_captures[
        "docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json"
    ]
    selection_payload = load_json_bytes(selection_capture.data, role="MNSol selection")
    dataset_module, pilot_module = _benchmark_modules()
    protocol = cast(
        _Protocol, dataset_module.load_mnsol_protocol(protocol_capture.path)
    )
    with captured_named_file(
        source_capture.data,
        suffix=".zip",
        role="MNSol distribution",
    ) as source:
        dataset = cast(_Dataset, dataset_module.load_mnsol_v2012(source, protocol))
    selected = cast(
        Sequence[_Selected],
        pilot_module.validate_frozen_mnsol_pilot_selection(
            selection_payload, dataset, protocol
        ),
    )
    prediction_records = prediction.get("records")
    if not isinstance(prediction_records, list) or len(prediction_records) != 10:
        raise SecureArtifactError("prediction row ledger is incomplete")
    scored_records: list[dict[str, object]] = []
    for index, (raw_prediction, selected_record) in enumerate(
        zip(prediction_records, selected, strict=True)
    ):
        predicted = _mapping(raw_prediction, name=f"prediction record {index}")
        eligible = selected_record.eligible_record
        normalized_geometry = accuracy.normalized_geometry_sha256(
            atomic_numbers=eligible.geometry.atomic_numbers,
            positions_angstrom=eligible.geometry.coordinates_angstrom,
            charge=eligible.record.charge,
            multiplicity=eligible.geometry.multiplicity,
        )
        expected_identity = {
            "selection_index": index,
            "opaque_record_id": selected_record.opaque_record_id,
            "canonical_solvent": selected_record.canonical_solvent,
            "partition": eligible.partition,
            "geometry_sha256": eligible.geometry.sha256,
            "normalized_geometry_sha256": normalized_geometry,
            "atom_count": len(eligible.geometry.atomic_numbers),
        }
        if any(predicted.get(key) != value for key, value in expected_identity.items()):
            raise SecureArtifactError(
                f"prediction identity differs from MNSol source at index {index}"
            )
        experimental = float(eligible.record.delta_g_kcal_mol)
        predicted_value = _finite(
            predicted["predicted_delta_g_kcal_mol"], name="predicted DeltaG"
        )
        signed = predicted_value - experimental
        scored_records.append(
            {
                **dict(predicted),
                "experimental_delta_g_kcal_mol": experimental,
                "signed_error_kcal_mol": signed,
                "absolute_error_kcal_mol": abs(signed),
            }
        )
    if dataset.source_artifact_sha256 is None:
        raise SecureArtifactError("MNSol archive source digest is unavailable")
    source_validation = _mapping(
        chain.seal.get("source_validation"), name="seal source validation"
    )
    if (
        source_capture.sha256 != source_validation.get("mnsol_source_file_sha256")
        or dataset.source_artifact_sha256 != source_capture.sha256
    ):
        raise SecureArtifactError("scorer MNSol archive differs from sealed bytes")
    dataset_binding = {
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "protocol_sha256": protocol_capture.sha256,
        "selection_sha256": selection_capture.sha256,
        "selection_fingerprint": selection_payload["selection_fingerprint"],
        "table_sha256": dataset.table_sha256,
        "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
        "source_artifact_sha256": dataset.source_artifact_sha256,
        "temperature_k": float(protocol.temperature_k),
        "standard_state": protocol.standard_state,
    }
    prediction_binding = {
        "attempt_slot_id": prediction["attempt_slot_id"],
        "execution_id": execution_id,
        "file_sha256": chain.prediction_capture.sha256,
        "content_sha256": prediction["content_sha256"],
        "measurement_sha256": prediction["measurement_sha256"],
    }
    scorer_source = chain.source_captures[
        "tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py"
    ]
    bundle = accuracy.build_scored_bundle(
        artifact_id=ARTIFACT_ID,
        scored_at_utc=datetime.now(timezone.utc).isoformat(),
        claim_boundary=prediction["claim_boundary"],
        profile_id=prediction["profile_id"],
        scalar_id=prediction["scalar_id"],
        prediction=prediction_binding,
        dataset=dataset_binding,
        scored_records=scored_records,
        scorer_source_sha256=scorer_source.sha256,
        expected_partition_counts={"confirmation": 8, "development": 2},
    )
    _require_model_free_process()
    stability.assert_stable()
    output = RUN_DIRECTORY / execution_id / "private-score.json"
    return bundle, stability, output


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        bundle, stability, output = score(
            source_path=args.source, prediction_path=args.prediction
        )
        publish_json_noreplace(
            output,
            bundle,
            root=REPO_ROOT,
            stability=stability.assert_stable,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "aggregate_metrics": bundle["aggregate_metrics"],
                "gates": bundle["gates"],
                "output": os.fspath(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
