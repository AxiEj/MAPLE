#!/usr/bin/env python3
"""Run the sealed, label-free hybrid MNSol-10 prediction process.

The process opens one private geometry/solvent bundle and never opens the
MNSol distribution or an experimental target.  A content-addressed execution
claim is published before model evaluation, so alternate output filenames
cannot bypass the one-time contract.  The only terminal is either ten complete
label-free predictions or one typed failure retaining all completed records.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import ModuleType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._secure_artifacts import (  # noqa: E402
    CapturedFile,
    RepositoryIdentity,
    StabilityGuard,
    canonical_custody_home,
    canonical_sha256,
    capture_clean_repository,
    capture_file,
    capture_repo_files,
    claim_execution,
    ensure_host_user_custody_directory,
    host_user_custody_contract,
    load_json_bytes,
    linux_boot_id,
    linux_process_start_ticks,
    publish_json_noreplace,
    secure_publication_contract,
    source_sha256s,
)
from tools.route2_release._hybrid_mnsol10_chain import (  # noqa: E402
    ATTEMPT_SLOT_CLAIM_PATH,
    build_attempt_slot_identity,
)

BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
PREREGISTRATION = BENCHMARK_DIR / (
    "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
)
INPUT_BUNDLE = REPO_ROOT / ".omx/route2/hybrid-mnsol10/label-free-input-v1.json"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACE-POLAR-1-M.model"
SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
RUN_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/runs"
ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-mnsol10-label-free-predictions-v1"
REQUIRED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}
CAPABILITIES_CLOSED = {tier: False for tier in ("E", "F", "H", "V", "M")}


class PredictionRunnerError(ValueError):
    """Raised when a sealed prediction cannot execute unambiguously."""


class ScientificExecutionError(RuntimeError):
    failure_stage = "science-execution"


class ProvenanceStabilityError(ScientificExecutionError):
    failure_stage = "provenance-stability"


class CandidateRecordValidationError(ScientificExecutionError):
    failure_stage = "candidate-record-validation"


class RecordGateFailure(ScientificExecutionError):
    failure_stage = "record-gate"

    def __init__(self, record: dict[str, object]) -> None:
        self.record = record
        super().__init__(
            "frozen root/charge gate failed for selection index "
            f"{record['selection_index']}"
        )


class _HybridModel(Protocol):
    provenance_sha256: str

    def prepare(self, geometry: object) -> object: ...

    def configuration_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class PredictionInput:
    selection_index: int
    opaque_record_id: str
    canonical_solvent: str
    partition: str
    geometry_sha256: str
    normalized_geometry_sha256: str
    atomic_numbers: tuple[int, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    prior_pilot_geometry_overlap: bool | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PredictionInput":
        numbers_raw = value.get("atomic_numbers")
        positions_raw = value.get("positions_angstrom")
        if not isinstance(numbers_raw, list) or not isinstance(positions_raw, list):
            raise PredictionRunnerError("input atomic numbers/positions are invalid")
        numbers = tuple(
            _integer(item, name="atomic number", minimum=1) for item in numbers_raw
        )
        positions: list[tuple[float, float, float]] = []
        for row in positions_raw:
            if not isinstance(row, list) or len(row) != 3:
                raise PredictionRunnerError("input positions must be atom-by-3")
            values = tuple(
                _finite(component, name="position component") for component in row
            )
            positions.append(cast(tuple[float, float, float], values))
        overlap = value.get("prior_pilot_geometry_overlap")
        if overlap is not None and type(overlap) is not bool:
            raise PredictionRunnerError("prior-pilot overlap must be boolean")
        result = cls(
            selection_index=_integer(value["selection_index"], name="selection_index"),
            opaque_record_id=_text(value["opaque_record_id"], name="opaque_record_id"),
            canonical_solvent=_text(
                value["canonical_solvent"], name="canonical_solvent"
            ),
            partition=_text(value["partition"], name="partition"),
            geometry_sha256=_text(value["geometry_sha256"], name="geometry_sha256"),
            normalized_geometry_sha256=_text(
                value["normalized_geometry_sha256"],
                name="normalized_geometry_sha256",
            ),
            atomic_numbers=numbers,
            positions_angstrom=tuple(positions),
            charge=_integer(value["charge"], name="charge"),
            multiplicity=_integer(
                value["multiplicity"], name="multiplicity", minimum=1
            ),
            prior_pilot_geometry_overlap=cast(bool | None, overlap),
        )
        if len(result.atomic_numbers) != len(result.positions_angstrom):
            raise PredictionRunnerError("input atom count is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class ContinuumContract:
    transition_width_angstrom2: float
    surface_lmax: int
    partition_lmax: int
    partition_radial_quadrature_order: int
    source_radial_quadrature_order: int
    double_layer_radial_quadrature_order: int
    receiver_radial_quadrature_order: int
    device: str


@dataclass(frozen=True, slots=True)
class RootContract:
    method: Literal["picard", "anderson"]
    tolerance: float
    max_iterations: int
    damping: float
    history: int
    wide_seed_base: int
    wide_initial_amplitude: float
    maximum_actual_residual_norm: float
    maximum_induced_source_absolute_difference_e: float
    maximum_native_field_absolute_difference_eV_per_e: float
    maximum_continuum_energy_absolute_difference_eV: float
    maximum_permanent_charge_error_e: float
    maximum_induced_charge_error_e: float
    maximum_combined_charge_error_e: float


@dataclass(frozen=True, slots=True)
class ScientificContract:
    profile_id: str
    scalar_id: str
    continuum: ContinuumContract
    root: RootContract

    @classmethod
    def from_preregistration(
        cls, payload: Mapping[str, object]
    ) -> "ScientificContract":
        target = _mapping(payload.get("target_identity"), name="target_identity")
        continuum = _mapping(
            payload.get("continuum_contract"), name="continuum_contract"
        )
        root = _mapping(payload.get("root_contract"), name="root_contract")
        method_raw = root.get("method")
        if method_raw not in {"picard", "anderson"}:
            raise PredictionRunnerError("root method is unsupported")
        method = cast(Literal["picard", "anderson"], method_raw)
        return cls(
            profile_id=_text(target["profile_id"], name="profile_id"),
            scalar_id=_text(target["scalar_id"], name="scalar_id"),
            continuum=ContinuumContract(
                transition_width_angstrom2=_finite(
                    continuum["transition_width_angstrom2"],
                    name="transition_width_angstrom2",
                ),
                surface_lmax=_integer(
                    continuum["surface_lmax"], name="surface_lmax", minimum=1
                ),
                partition_lmax=_integer(
                    continuum["partition_lmax"], name="partition_lmax", minimum=1
                ),
                partition_radial_quadrature_order=_integer(
                    continuum["partition_radial_quadrature_order"],
                    name="partition_radial_quadrature_order",
                    minimum=1,
                ),
                source_radial_quadrature_order=_integer(
                    continuum["source_radial_quadrature_order"],
                    name="source_radial_quadrature_order",
                    minimum=1,
                ),
                double_layer_radial_quadrature_order=_integer(
                    continuum["double_layer_radial_quadrature_order"],
                    name="double_layer_radial_quadrature_order",
                    minimum=1,
                ),
                receiver_radial_quadrature_order=_integer(
                    continuum["receiver_radial_quadrature_order"],
                    name="receiver_radial_quadrature_order",
                    minimum=1,
                ),
                device=_text(continuum["device"], name="continuum device"),
            ),
            root=RootContract(
                method=method,
                tolerance=_finite(root["tolerance"], name="root tolerance"),
                max_iterations=_integer(
                    root["max_iterations"], name="max_iterations", minimum=1
                ),
                damping=_finite(root["damping"], name="root damping"),
                history=_integer(root["history"], name="root history", minimum=1),
                wide_seed_base=_integer(root["wide_seed_base"], name="wide_seed_base"),
                wide_initial_amplitude=_finite(
                    root["wide_initial_amplitude"], name="wide_initial_amplitude"
                ),
                maximum_actual_residual_norm=_finite(
                    root["maximum_actual_residual_norm"],
                    name="maximum_actual_residual_norm",
                ),
                maximum_induced_source_absolute_difference_e=_finite(
                    root["maximum_induced_source_absolute_difference_e"],
                    name="maximum_induced_source_absolute_difference_e",
                ),
                maximum_native_field_absolute_difference_eV_per_e=_finite(
                    root["maximum_native_field_absolute_difference_eV_per_e"],
                    name="maximum_native_field_absolute_difference_eV_per_e",
                ),
                maximum_continuum_energy_absolute_difference_eV=_finite(
                    root["maximum_continuum_energy_absolute_difference_eV"],
                    name="maximum_continuum_energy_absolute_difference_eV",
                ),
                maximum_permanent_charge_error_e=_finite(
                    root["maximum_permanent_charge_error_e"],
                    name="maximum_permanent_charge_error_e",
                ),
                maximum_induced_charge_error_e=_finite(
                    root["maximum_induced_charge_error_e"],
                    name="maximum_induced_charge_error_e",
                ),
                maximum_combined_charge_error_e=_finite(
                    root["maximum_combined_charge_error_e"],
                    name="maximum_combined_charge_error_e",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class SealIdentity:
    raw: Mapping[str, object]
    execution_id: str
    attempt_slot_id: str
    attempt_slot_identity: Mapping[str, object]
    content_sha256: str
    profile_id: str
    scalar_id: str
    claim_boundary: str
    source_files_sha256: Mapping[str, str]
    source_validation: Mapping[str, object]
    input_file_sha256: str
    input_content_sha256: str
    mdp_checkpoint_sha256: str
    polar_checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    repository: RepositoryIdentity
    seal: SealIdentity
    seal_capture: CapturedFile
    input_capture: CapturedFile
    preregistration_capture: CapturedFile
    mdp_checkpoint_capture: CapturedFile
    polar_checkpoint_capture: CapturedFile
    source_captures: Mapping[str, CapturedFile]
    stability: StabilityGuard
    preregistration: Mapping[str, object]
    input_bundle: Mapping[str, object]
    accuracy: ModuleType


@dataclass(slots=True)
class ScienceProgress:
    records: list[dict[str, object]]
    runtime: dict[str, object]
    timing_seconds: dict[str, float]
    current_index: int | None = None
    current_record_id: str | None = None


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PredictionRunnerError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PredictionRunnerError(f"{name} must be nonempty text")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PredictionRunnerError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionRunnerError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise PredictionRunnerError(f"{name} must be finite")
    return result


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PredictionRunnerError(f"{name} must be a lowercase SHA256 digest")
    return value


def _self_digest(payload: Mapping[str, object], *, name: str) -> str:
    expected = _digest(payload.get("content_sha256"), name=f"{name}.content_sha256")
    content = dict(payload)
    del content["content_sha256"]
    actual = canonical_sha256(content)
    if actual != expected:
        raise PredictionRunnerError(f"{name} content digest mismatch")
    return actual


def _require_fresh_deterministic_process() -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT}
    if actual != REQUIRED_ENVIRONMENT:
        raise PredictionRunnerError(
            f"deterministic environment must be set before startup: {actual!r}"
        )
    forbidden = (
        "numpy",
        "scipy",
        "torch",
        "mace",
        "graph_longrange",
        "ase",
        "maple.solvation",
        "maple.function.calculator.mace",
    )
    preloaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    if preloaded:
        raise PredictionRunnerError(
            "scientific modules were loaded before environment validation: "
            + ", ".join(preloaded[:12])
        )
    return dict(REQUIRED_ENVIRONMENT)


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


def _parse_seal(payload: Mapping[str, object]) -> SealIdentity:
    _self_digest(payload, name="computation seal")
    if payload.get("schema_id") != "maple-route2-accuracy-computation-seal-v1":
        raise PredictionRunnerError("unsupported computation seal")
    execution_id = _digest(payload.get("execution_id"), name="execution_id")
    attempt_slot_id = _digest(payload.get("attempt_slot_id"), name="attempt_slot_id")
    attempt_slot_identity = _mapping(
        payload.get("attempt_slot_identity"), name="attempt_slot_identity"
    )
    if canonical_sha256(attempt_slot_identity) != attempt_slot_id:
        raise PredictionRunnerError("attempt-slot identity digest mismatch")
    identity = _mapping(payload.get("execution_identity"), name="execution_identity")
    if canonical_sha256(identity) != execution_id:
        raise PredictionRunnerError("execution identity digest mismatch")
    sources_raw = _mapping(
        payload.get("source_files_sha256"), name="source_files_sha256"
    )
    sources = {
        str(name): _digest(value, name=f"source_files_sha256[{name!r}]")
        for name, value in sources_raw.items()
    }
    input_bundle = _mapping(payload.get("input_bundle"), name="input_bundle")
    checkpoints = _mapping(payload.get("checkpoints"), name="checkpoints")
    source_validation = _mapping(
        payload.get("source_validation"), name="source_validation"
    )
    return SealIdentity(
        raw=payload,
        execution_id=execution_id,
        attempt_slot_id=attempt_slot_id,
        attempt_slot_identity=attempt_slot_identity,
        content_sha256=_digest(
            payload.get("content_sha256"), name="seal.content_sha256"
        ),
        profile_id=str(payload["profile_id"]),
        scalar_id=str(payload["scalar_id"]),
        claim_boundary=str(payload["claim_boundary"]),
        source_files_sha256=sources,
        source_validation=source_validation,
        input_file_sha256=_digest(
            input_bundle.get("file_sha256"), name="input_bundle.file_sha256"
        ),
        input_content_sha256=_digest(
            input_bundle.get("content_sha256"),
            name="input_bundle.content_sha256",
        ),
        mdp_checkpoint_sha256=_digest(
            checkpoints.get("mace_mdp_sha256"), name="mace_mdp_sha256"
        ),
        polar_checkpoint_sha256=_digest(
            checkpoints.get("mace_polar_sha256"), name="mace_polar_sha256"
        ),
    )


def _validate_seal_bindings(
    prepared: PreparedExecution,
    *,
    required_environment: Mapping[str, str],
) -> None:
    seal = prepared.seal
    raw = seal.raw
    git = _mapping(raw.get("git"), name="seal.git")
    if {
        "head": git.get("head"),
        "tree": git.get("tree"),
        "clean": git.get("clean"),
    } != prepared.repository.as_dict():
        raise PredictionRunnerError("seal Git identity differs from worktree")
    if raw.get("required_environment") != dict(required_environment):
        raise PredictionRunnerError("seal environment differs from startup")
    if raw.get("publication_contract") != secure_publication_contract():
        raise PredictionRunnerError("secure publication contract drifted")
    if raw.get("custody_contract") != host_user_custody_contract():
        raise PredictionRunnerError("host-user custody contract drifted")
    preregistration = _mapping(raw.get("preregistration"), name="seal.preregistration")
    if preregistration.get("path") != PREREGISTRATION.relative_to(REPO_ROOT).as_posix():
        raise PredictionRunnerError("seal preregistration path is not canonical")
    if preregistration.get("sha256") != prepared.preregistration_capture.sha256:
        raise PredictionRunnerError("seal preregistration digest drifted")
    if prepared.input_capture.sha256 != seal.input_file_sha256:
        raise PredictionRunnerError("seal input-bundle file digest drifted")
    if prepared.mdp_checkpoint_capture.sha256 != seal.mdp_checkpoint_sha256:
        raise PredictionRunnerError("seal MACE-MDP checkpoint digest drifted")
    if prepared.polar_checkpoint_capture.sha256 != seal.polar_checkpoint_sha256:
        raise PredictionRunnerError("seal MACE-POLAR checkpoint digest drifted")
    actual_sources = source_sha256s(prepared.source_captures)
    if actual_sources != dict(seal.source_files_sha256):
        raise PredictionRunnerError("seal source ledger differs from worktree")
    pro_contract = _mapping(
        prepared.preregistration.get("verified_pro_audit"),
        name="verified_pro_audit",
    )
    pro_path = pro_contract.get("path")
    if not isinstance(pro_path, str):
        raise PredictionRunnerError("verified Pro audit path is invalid")
    pro_capture = prepared.source_captures.get(pro_path)
    if (
        pro_capture is None
        or pro_capture.sha256 != pro_contract.get("file_sha256")
        or raw.get("verified_pro_audit_sha256") != pro_capture.sha256
        or pro_contract.get("final_verdict") != "APPROVE"
    ):
        raise PredictionRunnerError("verified Pro approval binding drifted")
    input_bundle = prepared.input_bundle
    input_git = _mapping(input_bundle.get("git"), name="input bundle git")
    if dict(input_git) != prepared.repository.as_dict():
        raise PredictionRunnerError("input bundle Git identity differs from worktree")
    protocol_capture = prepared.source_captures.get(
        "docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json"
    )
    selection_capture = prepared.source_captures.get(
        "docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json"
    )
    if protocol_capture is None or selection_capture is None:
        raise PredictionRunnerError("seal source ledger omitted dataset manifests")
    if (
        raw.get("protocol_sha256") != protocol_capture.sha256
        or raw.get("selection_sha256") != selection_capture.sha256
    ):
        raise PredictionRunnerError("seal dataset-manifest digests drifted")
    selection_payload = load_json_bytes(
        selection_capture.data, role="captured MNSol selection"
    )
    dataset_contract = _mapping(
        prepared.preregistration.get("dataset_contract"), name="dataset_contract"
    )
    if {
        "preregistration_sha256": input_bundle.get("preregistration_sha256"),
        "protocol_sha256": input_bundle.get("protocol_sha256"),
        "selection_sha256": input_bundle.get("selection_sha256"),
        "selection_fingerprint": input_bundle.get("selection_fingerprint"),
    } != {
        "preregistration_sha256": prepared.preregistration_capture.sha256,
        "protocol_sha256": protocol_capture.sha256,
        "selection_sha256": selection_capture.sha256,
        "selection_fingerprint": selection_payload.get("selection_fingerprint"),
    }:
        raise PredictionRunnerError("input bundle manifest bindings drifted")
    if {
        "protocol_sha256": dataset_contract.get("protocol_sha256"),
        "selection_sha256": dataset_contract.get("selection_sha256"),
        "selection_fingerprint": dataset_contract.get("selection_fingerprint"),
        "record_count": dataset_contract.get("record_count"),
    } != {
        "protocol_sha256": protocol_capture.sha256,
        "selection_sha256": selection_capture.sha256,
        "selection_fingerprint": selection_payload.get("selection_fingerprint"),
        "record_count": 10,
    }:
        raise PredictionRunnerError("preregistered dataset bindings drifted")
    input_records = input_bundle.get("records")
    manifest_records = selection_payload.get("selected_records")
    if not isinstance(input_records, list) or not isinstance(manifest_records, list):
        raise PredictionRunnerError("input/selection records are invalid")
    public_projection = [
        {
            "canonical_solvent": record.get("canonical_solvent"),
            "partition": record.get("partition"),
            "opaque_record_id": record.get("opaque_record_id"),
            "geometry_sha256": record.get("geometry_sha256"),
            "atom_count": record.get("atom_count"),
        }
        for record in input_records
        if isinstance(record, Mapping)
    ]
    if public_projection != manifest_records:
        raise PredictionRunnerError("input records differ from frozen selection")
    source_validation = seal.source_validation
    if source_validation.get("dataset") != input_bundle.get(
        "dataset"
    ) or source_validation.get("derived_records_sha256") != canonical_sha256(
        input_records
    ):
        raise PredictionRunnerError("source-backed input attestation drifted")
    expected_attempt_slot = build_attempt_slot_identity(
        preregistration=prepared.preregistration,
        source_validation=source_validation,
        protocol_sha256=protocol_capture.sha256,
        selection_sha256=selection_capture.sha256,
        mace_mdp_checkpoint_sha256=prepared.mdp_checkpoint_capture.sha256,
        mace_polar_checkpoint_sha256=prepared.polar_checkpoint_capture.sha256,
    )
    if (
        expected_attempt_slot != seal.attempt_slot_identity
        or canonical_sha256(expected_attempt_slot) != seal.attempt_slot_id
    ):
        raise PredictionRunnerError("scientific attempt-slot identity drifted")
    core = {
        "contract": "route2-hybrid-mnsol10-execution-identity-v1",
        "attempt_slot_id": seal.attempt_slot_id,
        "git_head": prepared.repository.head,
        "git_tree": prepared.repository.tree,
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "preregistration_sha256": prepared.preregistration_capture.sha256,
        "protocol_sha256": raw["protocol_sha256"],
        "selection_sha256": raw["selection_sha256"],
        "input_bundle_sha256": prepared.input_capture.sha256,
        "input_content_sha256": seal.input_content_sha256,
        "mace_mdp_checkpoint_sha256": prepared.mdp_checkpoint_capture.sha256,
        "mace_polar_checkpoint_sha256": prepared.polar_checkpoint_capture.sha256,
        "source_files_sha256": actual_sources,
        "source_validation": dict(source_validation),
        "verified_pro_audit_sha256": raw["verified_pro_audit_sha256"],
        "required_environment": dict(required_environment),
        "publication_contract": secure_publication_contract(),
        "custody_contract": host_user_custody_contract(),
    }
    if _mapping(raw.get("execution_identity"), name="execution_identity") != core:
        raise PredictionRunnerError("seal execution identity fields drifted")


def _prepare_execution(
    *,
    seal_path: str | Path,
    mdp_checkpoint_path: str | Path,
    polar_checkpoint_path: str | Path,
    required_environment: Mapping[str, str],
) -> PreparedExecution:
    repository = capture_clean_repository(REPO_ROOT)
    seal_capture = capture_file(seal_path, role="computation seal")
    seal_payload = load_json_bytes(seal_capture.data, role="computation seal")
    seal = _parse_seal(seal_payload)
    expected_seal_path = (SEAL_DIRECTORY / f"{seal.execution_id}.json").resolve()
    if seal_capture.path != expected_seal_path:
        raise PredictionRunnerError("computation seal path is not canonical")
    source_captures = capture_repo_files(repository, seal.source_files_sha256.keys())
    input_capture = capture_file(INPUT_BUNDLE, role="label-free input bundle")
    preregistration_capture = capture_file(PREREGISTRATION, role="preregistration")
    mdp_capture = capture_file(mdp_checkpoint_path, role="MACE-MDP checkpoint")
    polar_capture = capture_file(polar_checkpoint_path, role="MACE-POLAR checkpoint")
    stability = StabilityGuard(
        repository,
        tuple(
            [(f"source {name}", capture) for name, capture in source_captures.items()]
            + [
                ("computation seal", seal_capture),
                ("label-free input bundle", input_capture),
                ("preregistration", preregistration_capture),
                ("MACE-MDP checkpoint", mdp_capture),
                ("MACE-POLAR checkpoint", polar_capture),
            ]
        ),
    )
    accuracy_capture = source_captures.get(
        "maple/solvation/release/accuracy_admission.py"
    )
    if accuracy_capture is None:
        raise PredictionRunnerError("seal omitted accuracy contract source")
    accuracy = _load_captured_module(
        accuracy_capture,
        module_name="_maple_route2_captured_accuracy_admission",
    )
    input_bundle = load_json_bytes(input_capture.data, role="label-free input bundle")
    accuracy.validate_label_free_input_bundle(input_bundle, expected_record_count=10)
    preregistration = load_json_bytes(
        preregistration_capture.data, role="preregistration"
    )
    prepared = PreparedExecution(
        repository=repository,
        seal=seal,
        seal_capture=seal_capture,
        input_capture=input_capture,
        preregistration_capture=preregistration_capture,
        mdp_checkpoint_capture=mdp_capture,
        polar_checkpoint_capture=polar_capture,
        source_captures=source_captures,
        stability=stability,
        preregistration=preregistration,
        input_bundle=input_bundle,
        accuracy=accuracy,
    )
    _validate_seal_bindings(prepared, required_environment=required_environment)
    if input_bundle.get("content_sha256") != seal.input_content_sha256:
        raise PredictionRunnerError("label-free input content digest drifted")
    stability.assert_stable()
    return prepared


def _validate_runtime_contract(
    prepared: PreparedExecution, contract: ScientificContract
) -> None:
    from maple.solvation.api.identities import (
        MACE_MDP_POLAR_HYBRID_HARMONIC_DDPCM_COUPLING_ID,
    )
    from maple.solvation.api.profiles import get_solvation_profile
    from maple.solvation.api.scalar_registry import get_scalar_definition
    from maple.solvation.continuum.harmonic_ddpcm_hybrid import (
        HARMONIC_DDPCM_HYBRID_COUPLING_ID,
    )
    from maple.solvation.solvent_terms import PYSCF_SMD_CDS_PROVIDER_ID

    if (
        contract.profile_id != prepared.seal.profile_id
        or contract.scalar_id != prepared.seal.scalar_id
    ):
        raise PredictionRunnerError("preregistered target differs from seal")
    profile = get_solvation_profile(contract.profile_id)
    scalar = get_scalar_definition(contract.scalar_id)
    if (
        profile.scalar_id != scalar.scalar_id
        or profile.coupling_id != MACE_MDP_POLAR_HYBRID_HARMONIC_DDPCM_COUPLING_ID
        or HARMONIC_DDPCM_HYBRID_COUPLING_ID != profile.coupling_id
        or profile.nonpolar_profile != "pyscf-smd-cds-v1"
        or scalar.implementation_entry_point
        != (
            "maple.solvation.coupling.additive_solvent_ledgers:"
            "AdditiveSolventOperationalLedger"
        )
        or profile.enabled
        or scalar.enabled
    ):
        raise PredictionRunnerError("registered total-SMD identity drifted")
    nonpolar = _mapping(
        prepared.preregistration.get("nonpolar_contract"),
        name="nonpolar_contract",
    )
    if nonpolar.get("provider_id") != PYSCF_SMD_CDS_PROVIDER_ID:
        raise PredictionRunnerError("registered SMD-CDS provider drifted")
    continuum = contract.continuum
    if continuum != ContinuumContract(
        transition_width_angstrom2=0.08,
        surface_lmax=3,
        partition_lmax=6,
        partition_radial_quadrature_order=96,
        source_radial_quadrature_order=128,
        double_layer_radial_quadrature_order=128,
        receiver_radial_quadrature_order=128,
        device="cpu",
    ):
        raise PredictionRunnerError("frozen continuum contract drifted")
    validation = prepared.accuracy.HYBRID_MNSOL_PREDICTION_VALIDATION
    root = contract.root
    thresholds = {
        "maximum_actual_residual_norm": root.maximum_actual_residual_norm,
        "maximum_induced_source_absolute_difference_e": (
            root.maximum_induced_source_absolute_difference_e
        ),
        "maximum_native_field_absolute_difference_eV_per_e": (
            root.maximum_native_field_absolute_difference_eV_per_e
        ),
        "maximum_continuum_energy_absolute_difference_eV": (
            root.maximum_continuum_energy_absolute_difference_eV
        ),
        "maximum_permanent_charge_error_e": root.maximum_permanent_charge_error_e,
        "maximum_induced_charge_error_e": root.maximum_induced_charge_error_e,
        "maximum_combined_charge_error_e": root.maximum_combined_charge_error_e,
    }
    if any(getattr(validation, name) != value for name, value in thresholds.items()):
        raise PredictionRunnerError("prediction validator thresholds drifted")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _configure_runtime(
    required_environment: Mapping[str, str],
) -> tuple[Any, Any, dict[str, object]]:
    import numpy as np
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ScientificExecutionError(
            "preregistered hybrid MNSol-10 runtime requires exactly one CUDA GPU"
        )
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": dict(required_environment),
        "packages": {
            name: _package_version(name)
            for name in (
                "ase",
                "mace-torch",
                "graph-longrange",
                "numpy",
                "pyscf",
                "scipy",
                "torch",
            )
        },
        "numpy_show_config": configuration.getvalue().strip(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    return np, torch, runtime


def _assert_loaded_repo_modules_bound(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "maple"
            or name.startswith("maple.")
            or name == "tools"
            or name.startswith("tools.")
        ):
            continue
        origins: list[Path] = []
        raw_file = getattr(module, "__file__", None)
        if raw_file is not None:
            origins.append(Path(raw_file).resolve())
        raw_paths = getattr(module, "__path__", ())
        origins.extend(Path(item).resolve() for item in raw_paths)
        for origin in origins:
            try:
                origin.relative_to(root)
            except ValueError as exc:
                raise ScientificExecutionError(
                    f"loaded repository module {name} resolves outside clean tree"
                ) from exc


def _atoms(input_record: PredictionInput) -> Any:
    from ase import Atoms

    return Atoms(
        numbers=input_record.atomic_numbers,
        positions=input_record.positions_angstrom,
        info={"charge": input_record.charge, "mult": input_record.multiplicity},
    )


def _evaluate_record(
    *,
    input_record: PredictionInput,
    contract: ScientificContract,
    preregistration_sha256: str,
    hybrid: _HybridModel,
    np: Any,
    torch: Any,
) -> dict[str, object]:
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        HARTREE_TO_KCAL_MOL,
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.api.units import HARTREE_TO_EV
    from maple.solvation.continuum.harmonic_ddpcm_functional import (
        SmoothPartitionHarmonicDDPCMFunctionalCandidate,
    )
    from maple.solvation.continuum.harmonic_ddpcm_hybrid import (
        build_harmonic_ddpcm_hybrid_snapshot,
    )
    from maple.solvation.coupling.additive_solvent_ledgers import (
        AdditiveSolventOperationalLedger,
    )
    from maple.solvation.coupling.fixed_point import FixedPointOptions
    from maple.solvation.coupling.operator import canonical_metadata_sha256
    from maple.solvation.coupling.permanent_induced_ledgers import (
        HybridHarmonicDDPCMPhi0Ledger,
    )
    from maple.solvation.coupling.permanent_induced_state import (
        PermanentInducedOperationalStateEquation,
    )
    from maple.solvation.coupling.separated_fixed_point import (
        solve_separated_fixed_point,
    )
    from maple.solvation.coupling.spaces import (
        ATOMIC_L1_SOURCE_SPACE,
        AffineChargeCoordinates,
    )
    from maple.solvation.release.admission import canonical_source_monopole_sum_v2
    from maple.solvation.solvent_terms import PySCFSMDCDSTerm

    atoms = _atoms(input_record)
    if input_record.charge != 0 or input_record.multiplicity != 1:
        raise ScientificExecutionError("prediction input must be a neutral singlet")
    solvent_spec = route2_solvent_spec(input_record.canonical_solvent)
    symbols = tuple(atoms.get_chemical_symbols())
    radii = tuple(
        float(value)
        for value in smd_coulomb_radii(symbols, solvent=input_record.canonical_solvent)
    )
    continuum = contract.continuum
    functional = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=input_record.atomic_numbers,
        radii_angstrom=radii,
        transition_width_angstrom2=continuum.transition_width_angstrom2,
        surface_lmax=continuum.surface_lmax,
        partition_lmax=continuum.partition_lmax,
        partition_radial_quadrature_order=(continuum.partition_radial_quadrature_order),
        source_radial_quadrature_order=continuum.source_radial_quadrature_order,
        double_layer_radial_quadrature_order=(
            continuum.double_layer_radial_quadrature_order
        ),
        dielectric=float(solvent_spec.descriptors.dielectric),
        dtype=torch.float64,
        device=continuum.device,
    )
    anchor = hybrid.prepare(atoms)
    continuum_snapshot = build_harmonic_ddpcm_hybrid_snapshot(
        functional,
        atoms,
        receiver_radial_quadrature_order=(continuum.receiver_radial_quadrature_order),
    )
    coordinates = AffineChargeCoordinates(
        atom_count=len(atoms),
        total_charge=0.0,
        source_space=ATOMIC_L1_SOURCE_SPACE,
    )
    equation = PermanentInducedOperationalStateEquation(
        cast(Any, coordinates),
        cast(Any, hybrid),
        cast(Any, anchor),
        continuum_snapshot,
    )
    root = contract.root
    options = FixedPointOptions(
        method=root.method,
        tolerance=root.tolerance,
        max_iterations=root.max_iterations,
        damping=root.damping,
        history=root.history,
    )
    context = (
        f"hybrid-mnsol10-{input_record.selection_index}-"
        f"{input_record.opaque_record_id}"
    )
    cold = solve_separated_fixed_point(
        equation, atoms, root_context_id=context, options=options
    )
    seed = root.wide_seed_base + input_record.selection_index
    wide_initial = cold.y_array() + root.wide_initial_amplitude * np.random.default_rng(
        seed
    ).normal(size=cold.y_array().shape)
    wide = solve_separated_fixed_point(
        equation,
        atoms,
        root_context_id=context,
        initial_y=wide_initial,
        options=options,
    )
    solvent_term = PySCFSMDCDSTerm(symbols, input_record.canonical_solvent)
    base_ledger = HybridHarmonicDDPCMPhi0Ledger(equation)
    total_ledger = AdditiveSolventOperationalLedger(base_ledger, solvent_term)
    cold_evaluation = total_ledger.evaluate_root(
        atoms, cold.y_array(), root_tolerance=root.tolerance
    )
    wide_evaluation = total_ledger.evaluate_root(
        atoms, wide.y_array(), root_tolerance=root.tolerance
    )
    cold_energy = total_ledger.solvation_energy_eV(cold_evaluation)
    wide_energy = total_ledger.solvation_energy_eV(wide_evaluation)
    source_difference = float(np.max(np.abs(cold.source_array() - wide.source_array())))
    field_difference = float(np.max(np.abs(cold.field_array() - wide.field_array())))
    energy_difference = abs(cold_energy - wide_energy)
    residual_maximum = max(
        float(cold.actual_unmixed_residual_norm),
        float(wide.actual_unmixed_residual_norm),
    )
    permanent_charge = canonical_source_monopole_sum_v2(
        cast(Any, anchor).permanent_source4
    )
    induced_charge = canonical_source_monopole_sum_v2(cold.source_array())
    combined_charge = permanent_charge + induced_charge
    gates = {
        "cold_wide_source": (
            source_difference <= root.maximum_induced_source_absolute_difference_e
        ),
        "cold_wide_field": (
            field_difference <= root.maximum_native_field_absolute_difference_eV_per_e
        ),
        "cold_wide_energy": (
            energy_difference <= root.maximum_continuum_energy_absolute_difference_eV
        ),
        "actual_residuals": residual_maximum <= root.maximum_actual_residual_norm,
        "permanent_charge": (
            abs(permanent_charge) <= root.maximum_permanent_charge_error_e
        ),
        "induced_charge": (abs(induced_charge) <= root.maximum_induced_charge_error_e),
        "combined_charge": (
            abs(combined_charge) <= root.maximum_combined_charge_error_e
        ),
    }
    components = dict(cold_evaluation.components_eV)
    vacuum_energy = float(components["macepolar_vacuum_energy"])
    polarization_energy = float(
        components["point_permanent_gaussian_induced_smooth_harmonic_ddpcm_energy"]
    )
    cds_energy = float(components["pyscf_smd_cds_energy"])
    if not math.isclose(
        cold_evaluation.total_energy_eV - vacuum_energy,
        cold_energy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ScientificExecutionError(
            "solution total minus vacuum does not equal predicted DeltaG_solv"
        )
    ev_to_kcal_mol = HARTREE_TO_KCAL_MOL / HARTREE_TO_EV
    predicted = cold_energy * ev_to_kcal_mol
    configuration_sha256 = canonical_metadata_sha256(
        {
            "preregistration_sha256": preregistration_sha256,
            "hybrid_configuration_sha256": hybrid.configuration_sha256(),
            "continuum_configuration_sha256": (
                continuum_snapshot.configuration_sha256()
            ),
            "equation_sha256": equation.fingerprint_sha256(),
            "base_ledger_sha256": base_ledger.configuration_sha256(),
            "total_ledger_sha256": total_ledger.configuration_sha256(),
            "solvent_term_configuration_sha256": (solvent_term.configuration_sha256()),
        }
    )
    state_sha256 = canonical_metadata_sha256(
        {
            "cold_root_sha256": cold.root_hash,
            "wide_root_sha256": wide.root_hash,
            "solution_total_energy_eV": cold_evaluation.total_energy_eV,
            "vacuum_energy_eV": vacuum_energy,
            "predicted_delta_g_eV": cold_energy,
            "total_ledger_sha256": cold_evaluation.ledger_sha256,
            "predicted_delta_g_kcal_mol": predicted,
        }
    )
    record: dict[str, object] = {
        "selection_index": input_record.selection_index,
        "opaque_record_id": input_record.opaque_record_id,
        "canonical_solvent": input_record.canonical_solvent,
        "partition": input_record.partition,
        "geometry_sha256": input_record.geometry_sha256,
        "normalized_geometry_sha256": input_record.normalized_geometry_sha256,
        "atom_count": len(atoms),
        "solution_total_energy_eV": float(cold_evaluation.total_energy_eV),
        "vacuum_energy_eV": vacuum_energy,
        "predicted_delta_g_eV": cold_energy,
        "predicted_delta_g_kcal_mol": predicted,
        "polarization_kcal_mol": polarization_energy * ev_to_kcal_mol,
        "cds_kcal_mol": cds_energy * ev_to_kcal_mol,
        "permanent_charge_e": permanent_charge,
        "induced_charge_e": induced_charge,
        "combined_charge_e": combined_charge,
        "cold_actual_residual_norm": float(cold.actual_unmixed_residual_norm),
        "wide_actual_residual_norm": float(wide.actual_unmixed_residual_norm),
        "cold_iterations": len(cold.iterations) - 1,
        "wide_iterations": len(wide.iterations) - 1,
        "induced_source_max_abs_difference_e": source_difference,
        "native_field_max_abs_difference_eV_per_e": field_difference,
        "continuum_energy_abs_difference_eV": energy_difference,
        "gates": gates,
        "configuration_sha256": configuration_sha256,
        "continuum_configuration_sha256": (continuum_snapshot.configuration_sha256()),
        "continuum_provenance_sha256": continuum_snapshot.provenance_sha256,
        "equation_sha256": equation.fingerprint_sha256(),
        "cold_root_sha256": cold.root_hash,
        "wide_root_sha256": wide.root_hash,
        "base_ledger_sha256": base_ledger.configuration_sha256(),
        "total_ledger_sha256": total_ledger.configuration_sha256(),
        "solvent_term_configuration_sha256": (solvent_term.configuration_sha256()),
        "state_sha256": state_sha256,
    }
    if input_record.prior_pilot_geometry_overlap is not None:
        record["prior_pilot_geometry_overlap"] = (
            input_record.prior_pilot_geometry_overlap
        )
    return record


def _load_hybrid(
    *, prepared: PreparedExecution
) -> tuple[_HybridModel, dict[str, object]]:
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_anchored_mace_polar_hybrid,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )

    mdp = build_mace_mdp_moment_adapter(
        checkpoint_bytes=prepared.mdp_checkpoint_capture.data, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_bytes=prepared.polar_checkpoint_capture.data,
        device="cuda",
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    if mdp.checkpoint_sha256 != prepared.seal.mdp_checkpoint_sha256:
        raise ScientificExecutionError("loaded MACE-MDP checkpoint identity drifted")
    if response.checkpoint_sha256 != prepared.seal.polar_checkpoint_sha256:
        raise ScientificExecutionError("loaded MACE-POLAR checkpoint identity drifted")
    hybrid = cast(
        _HybridModel,
        build_mace_mdp_anchored_mace_polar_hybrid(permanent=mdp, response=response),
    )
    return hybrid, {
        "mace_mdp_sha256": mdp.checkpoint_sha256,
        "mace_mdp_device": "cpu",
        "mace_polar_sha256": response.checkpoint_sha256,
        "mace_polar_device": "cuda",
        "hybrid_configuration_sha256": hybrid.configuration_sha256(),
        "hybrid_provenance_sha256": hybrid.provenance_sha256,
    }


ScienceFactory = Callable[
    [PreparedExecution, ScientificContract, ScienceProgress],
    None,
]


def _run_science(
    prepared: PreparedExecution,
    contract: ScientificContract,
    progress: ScienceProgress,
) -> None:
    np, torch, runtime = _configure_runtime(REQUIRED_ENVIRONMENT)
    progress.runtime.update(runtime)
    _assert_loaded_repo_modules_bound(prepared.repository.root)
    model_started = time.perf_counter()
    hybrid, model_identity = _load_hybrid(prepared=prepared)
    model_load_seconds = time.perf_counter() - model_started
    progress.runtime["model_identity"] = model_identity
    progress.timing_seconds["model_load"] = model_load_seconds
    raw_records = prepared.input_bundle.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 10:
        raise ScientificExecutionError("label-free input bundle is incomplete")
    inputs = [
        PredictionInput.from_mapping(_mapping(item, name="prediction input"))
        for item in raw_records
    ]
    record_seconds: list[float] = []
    for input_record in inputs:
        progress.current_index = input_record.selection_index
        progress.current_record_id = input_record.opaque_record_id
        print(
            f"[{input_record.selection_index + 1}/10] "
            f"solvent={input_record.canonical_solvent} "
            f"atoms={len(input_record.atomic_numbers)}",
            flush=True,
        )
        started = time.perf_counter()
        record = _evaluate_record(
            input_record=input_record,
            contract=contract,
            preregistration_sha256=prepared.preregistration_capture.sha256,
            hybrid=hybrid,
            np=np,
            torch=torch,
        )
        prepared.accuracy.validate_prediction_record(
            record, expected_index=input_record.selection_index
        )
        record_seconds.append(time.perf_counter() - started)
        progress.records.append(record)
        progress.timing_seconds["record_total"] = sum(record_seconds)
        print(
            json.dumps(
                {
                    "selection_index": record["selection_index"],
                    "canonical_solvent": record["canonical_solvent"],
                    "gates": record["gates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gates = _mapping(record["gates"], name="record gates")
        if not all(value is True for value in gates.values()):
            raise RecordGateFailure(record)
    progress.current_index = None
    progress.current_record_id = None


def _terminal_base(
    *,
    prepared: PreparedExecution,
    process_started_at_utc: str,
    finished_at_utc: str,
    records: Sequence[Mapping[str, object]],
    runtime: Mapping[str, object],
    timing_seconds: Mapping[str, float],
) -> dict[str, object]:
    return {
        "schema_id": "maple-route2-accuracy-prediction-terminal-v1",
        "artifact_id": ARTIFACT_ID,
        "attempt_slot_id": prepared.seal.attempt_slot_id,
        "execution_id": prepared.seal.execution_id,
        "started_at_utc": process_started_at_utc,
        "finished_at_utc": finished_at_utc,
        "claim_boundary": prepared.seal.claim_boundary,
        "profile_id": prepared.seal.profile_id,
        "scalar_id": prepared.seal.scalar_id,
        "git": prepared.repository.as_dict(),
        "seal": {
            "file_sha256": prepared.seal_capture.sha256,
            "content_sha256": prepared.seal.content_sha256,
        },
        "input_bundle": {
            "file_sha256": prepared.input_capture.sha256,
            "content_sha256": prepared.seal.input_content_sha256,
        },
        "checkpoints": {
            "mace_mdp_sha256": prepared.mdp_checkpoint_capture.sha256,
            "mace_polar_sha256": prepared.polar_checkpoint_capture.sha256,
        },
        "source_files_sha256": dict(prepared.seal.source_files_sha256),
        "runtime": dict(runtime),
        "record_count_expected": 10,
        "records": [dict(record) for record in records],
        "timing_seconds": dict(timing_seconds),
        "capabilities": dict(CAPABILITIES_CLOSED),
    }


def _typed_failure(
    *,
    error: BaseException,
    failed_index: int | None,
    failed_record_id: str | None,
) -> dict[str, object]:
    message = str(error).strip() or type(error).__name__
    return {
        "stage": str(getattr(error, "failure_stage", "science-execution")),
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message,
        "failed_selection_index": failed_index,
        "failed_opaque_record_id": failed_record_id,
    }


def _validated_record_prefix(
    accuracy: ModuleType,
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], int | None, str | None, BaseException | None]:
    prefix: list[dict[str, object]] = []
    for index, record in enumerate(records):
        try:
            accuracy.validate_prediction_record(record, expected_index=index)
        except BaseException as error:
            record_id = record.get("opaque_record_id")
            return (
                prefix,
                index,
                record_id if isinstance(record_id, str) else None,
                error,
            )
        prefix.append(dict(record))
    return prefix, None, None, None


def execute(
    *,
    seal_path: str | Path,
    mdp_checkpoint_path: str | Path = DEFAULT_MDP_CHECKPOINT,
    polar_checkpoint_path: str | Path = DEFAULT_POLAR_CHECKPOINT,
    science_factory: ScienceFactory = _run_science,
) -> tuple[Path, Mapping[str, object]]:
    required_environment = _require_fresh_deterministic_process()
    prepared = _prepare_execution(
        seal_path=seal_path,
        mdp_checkpoint_path=mdp_checkpoint_path,
        polar_checkpoint_path=polar_checkpoint_path,
        required_environment=required_environment,
    )
    contract = ScientificContract.from_preregistration(prepared.preregistration)
    _validate_runtime_contract(prepared, contract)
    prepared.stability.assert_stable()
    process_uuid = str(uuid4())
    process_id = os.getpid()
    process_start_ticks = linux_process_start_ticks(process_id)
    boot_id = linux_boot_id()
    process_started_at_utc = datetime.now(timezone.utc).isoformat()
    run_directory = RUN_DIRECTORY / prepared.seal.execution_id
    claim_path = ATTEMPT_SLOT_CLAIM_PATH
    terminal_path = run_directory / "prediction-terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        raise PredictionRunnerError(
            "canonical prediction terminal already exists before stage claim"
        )
    ensure_host_user_custody_directory(claim_path.parent)
    claim_execution(
        claim_path,
        {
            "schema_id": "maple-route2-accuracy-execution-claim-v1",
            "attempt_slot_id": prepared.seal.attempt_slot_id,
            "execution_id": prepared.seal.execution_id,
            "seal_file_sha256": prepared.seal_capture.sha256,
            "seal_content_sha256": prepared.seal.content_sha256,
            "preregistration_sha256": prepared.preregistration_capture.sha256,
            "source_files_sha256": dict(prepared.seal.source_files_sha256),
            "checkpoints": {
                "mace_mdp_sha256": prepared.mdp_checkpoint_capture.sha256,
                "mace_polar_sha256": prepared.polar_checkpoint_capture.sha256,
            },
            "input_bundle": {
                "file_sha256": prepared.input_capture.sha256,
                "content_sha256": prepared.seal.input_content_sha256,
            },
            "required_environment": required_environment,
            "publication_contract": secure_publication_contract(),
            "custody_contract": host_user_custody_contract(),
            "process_uuid": process_uuid,
            "boot_id": boot_id,
            "process_id": process_id,
            "process_start_ticks": process_start_ticks,
            "process_started_at_utc": process_started_at_utc,
            "terminal_path": terminal_path.relative_to(REPO_ROOT).as_posix(),
            "rule": (
                "one scientific attempt slot permits exactly one execution and "
                "never reopens"
            ),
        },
        root=canonical_custody_home(),
        stability=prepared.stability.assert_stable,
    )
    started = time.perf_counter()
    progress = ScienceProgress(
        records=[],
        runtime={
            "environment": required_environment,
            "process_uuid": process_uuid,
            "boot_id": boot_id,
            "process_id": process_id,
            "process_start_ticks": process_start_ticks,
        },
        timing_seconds={},
    )
    claim_capture: CapturedFile | None = None

    def claimed_execution_stability() -> None:
        prepared.stability.assert_stable()
        if claim_capture is not None:
            claim_capture.assert_stable(role="stage claim")

    publication_stability: Callable[[], None] = claimed_execution_stability
    try:
        claim_capture = capture_file(claim_path, role="execution claim")
        science_factory(prepared, contract, progress)
        prepared.stability.assert_stable()
        terminal = _terminal_base(
            prepared=prepared,
            process_started_at_utc=process_started_at_utc,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            records=progress.records,
            runtime=progress.runtime,
            timing_seconds={
                **progress.timing_seconds,
                "total_wall": time.perf_counter() - started,
            },
        )
        terminal["status"] = "complete"
        terminal["measurement_sha256"] = (
            prepared.accuracy.prediction_measurement_sha256(progress.records)
        )
        terminal = prepared.accuracy.with_content_sha256(terminal)
        prepared.accuracy.validate_prediction_terminal(
            terminal, expected_record_count=10
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        prefix, invalid_index, invalid_record_id, validation_error = (
            _validated_record_prefix(prepared.accuracy, progress.records)
        )
        if validation_error is not None:
            progress.records = prefix
            progress.current_index = invalid_index
            progress.current_record_id = invalid_record_id
            error = CandidateRecordValidationError(
                "candidate record failed mechanical validation: " f"{validation_error}"
            )
        try:
            prepared.stability.assert_stable()
        except BaseException as stability_error:
            error = ProvenanceStabilityError(
                f"provenance changed after claim: {stability_error}"
            )

            def captured_failure_stability() -> None:
                prepared.seal_capture.assert_stable(role="computation seal")
                if claim_capture is not None:
                    claim_capture.assert_stable(role="execution claim")

            publication_stability = captured_failure_stability
        terminal = _terminal_base(
            prepared=prepared,
            process_started_at_utc=process_started_at_utc,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            records=progress.records,
            runtime=progress.runtime,
            timing_seconds={
                **progress.timing_seconds,
                "total_wall": time.perf_counter() - started,
            },
        )
        terminal["status"] = "failure"
        terminal["failure"] = _typed_failure(
            error=error,
            failed_index=progress.current_index,
            failed_record_id=progress.current_record_id,
        )
        terminal = prepared.accuracy.with_content_sha256(terminal)
        prepared.accuracy.validate_prediction_terminal(
            terminal, expected_record_count=10
        )
    publish_json_noreplace(
        terminal_path,
        terminal,
        root=REPO_ROOT,
        stability=publication_stability,
    )
    return terminal_path, terminal


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        output, terminal = execute(
            seal_path=args.seal,
            mdp_checkpoint_path=args.mdp_checkpoint,
            polar_checkpoint_path=args.polar_checkpoint,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "execution_id": terminal["execution_id"],
                "status": terminal["status"],
                "record_count": len(cast(Sequence[object], terminal["records"])),
                "output": os.fspath(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
