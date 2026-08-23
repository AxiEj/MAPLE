"""Independent seal/claim/input/prediction chain verification for MNSol-10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, cast

from tools.route2_release._secure_artifacts import (
    CapturedFile,
    RepositoryIdentity,
    SecureArtifactError,
    StabilityGuard,
    canonical_custody_home,
    canonical_sha256,
    capture_clean_repository,
    capture_file,
    capture_repo_files,
    host_user_custody_contract,
    load_json_bytes,
    secure_publication_contract,
    source_sha256s,
)

ATTEMPT_SLOT_STAGE_ID = "maple.route2.mnsol10-known-regression.attempt-slot.v1"
ATTEMPT_SLOT_EVIDENCE_CLASS = "exposure-aware-known-panel-regression"
STAGE_CLAIM_REGISTRY = canonical_custody_home() / ".cache/maple-route2-stage-claims-v1"
ATTEMPT_SLOT_CLAIM_PATH = (
    STAGE_CLAIM_REGISTRY / ATTEMPT_SLOT_STAGE_ID / "execution-claim.json"
)


def build_attempt_slot_identity(
    *,
    preregistration: Mapping[str, object],
    source_validation: Mapping[str, object],
    protocol_sha256: str,
    selection_sha256: str,
    mace_mdp_checkpoint_sha256: str,
    mace_polar_checkpoint_sha256: str,
) -> dict[str, object]:
    """Build the immutable scientific slot, intentionally excluding code/Git."""

    sections = {
        name: preregistration[name]
        for name in (
            "protocol_classification",
            "target_identity",
            "dataset_contract",
            "model_contract",
            "continuum_contract",
            "root_contract",
            "nonpolar_contract",
            "decision_rule",
            "post_execution_prohibitions",
        )
    }
    execution = _mapping(
        preregistration.get("execution_contract"), name="execution_contract"
    )
    sections["execution_attempt_contract"] = {
        "required_environment": execution.get("required_environment"),
        "evidence_class": execution.get("evidence_class"),
    }
    target = _mapping(preregistration.get("target_identity"), name="target_identity")
    dataset = _mapping(preregistration.get("dataset_contract"), name="dataset_contract")
    return {
        "contract": "maple-route2-scientific-attempt-slot-identity-v1",
        "stage_id": ATTEMPT_SLOT_STAGE_ID,
        "evidence_class": ATTEMPT_SLOT_EVIDENCE_CLASS,
        "mnsol_source_file_sha256": _digest(
            source_validation.get("mnsol_source_file_sha256"),
            name="mnsol_source_file_sha256",
        ),
        "dataset_identity": source_validation.get("dataset"),
        "derived_records_sha256": _digest(
            source_validation.get("derived_records_sha256"),
            name="derived_records_sha256",
        ),
        "protocol_sha256": _digest(protocol_sha256, name="protocol_sha256"),
        "selection_sha256": _digest(selection_sha256, name="selection_sha256"),
        "selection_fingerprint": dataset.get("selection_fingerprint"),
        "profile_id": target.get("profile_id"),
        "scalar_id": target.get("scalar_id"),
        "state_equation_id": target.get("state_equation_id"),
        "mace_mdp_checkpoint_sha256": _digest(
            mace_mdp_checkpoint_sha256, name="mace_mdp_checkpoint_sha256"
        ),
        "mace_polar_checkpoint_sha256": _digest(
            mace_polar_checkpoint_sha256, name="mace_polar_checkpoint_sha256"
        ),
        "scientific_contract_sha256": canonical_sha256(sections),
    }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SecureArtifactError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecureArtifactError(f"{name} must be a lowercase SHA256 digest")
    return value


def _self_digest(payload: Mapping[str, object], *, name: str) -> str:
    expected = _digest(payload.get("content_sha256"), name=f"{name}.content_sha256")
    content = dict(payload)
    del content["content_sha256"]
    actual = canonical_sha256(content)
    if actual != expected:
        raise SecureArtifactError(f"{name} content digest mismatch")
    return actual


@dataclass(frozen=True, slots=True)
class PredictionChain:
    repository: RepositoryIdentity
    attempt_slot_id: str
    execution_id: str
    seal_capture: CapturedFile
    claim_capture: CapturedFile
    input_capture: CapturedFile
    prediction_capture: CapturedFile
    source_captures: Mapping[str, CapturedFile]
    seal: Mapping[str, object]
    claim: Mapping[str, object]
    input_bundle: Mapping[str, object]
    prediction: Mapping[str, object]
    stability: StabilityGuard


def capture_prediction_chain(
    *,
    repo_root: Path,
    prediction_path: str | Path,
    seal_directory: Path,
    run_directory: Path,
    input_bundle_path: Path,
    accuracy: ModuleType,
) -> PredictionChain:
    """Capture and verify the complete prediction lineage on a clean tree."""

    repository = capture_clean_repository(repo_root)
    prediction_capture = capture_file(prediction_path, role="prediction terminal")
    prediction = load_json_bytes(prediction_capture.data, role="prediction terminal")
    execution_id = _digest(prediction.get("execution_id"), name="execution_id")
    expected_prediction = (
        run_directory / execution_id / "prediction-terminal.json"
    ).resolve()
    if prediction_capture.path != expected_prediction:
        raise SecureArtifactError("prediction terminal path is not canonical")
    seal_capture = capture_file(
        seal_directory / f"{execution_id}.json", role="computation seal"
    )
    input_capture = capture_file(input_bundle_path, role="label-free input bundle")
    seal = load_json_bytes(seal_capture.data, role="computation seal")
    input_bundle = load_json_bytes(input_capture.data, role="label-free input bundle")
    if seal.get("schema_id") != "maple-route2-accuracy-computation-seal-v1":
        raise SecureArtifactError("unsupported computation seal schema")
    _self_digest(seal, name="computation seal")
    if seal.get("publication_contract") != secure_publication_contract():
        raise SecureArtifactError("seal secure publication contract drifted")
    if seal.get("custody_contract") != host_user_custody_contract():
        raise SecureArtifactError("seal host-user custody contract drifted")
    attempt_slot_id = _digest(seal.get("attempt_slot_id"), name="seal.attempt_slot_id")
    attempt_slot_identity = _mapping(
        seal.get("attempt_slot_identity"), name="seal.attempt_slot_identity"
    )
    if canonical_sha256(attempt_slot_identity) != attempt_slot_id:
        raise SecureArtifactError("seal attempt-slot identity digest mismatch")
    claim_capture = capture_file(
        ATTEMPT_SLOT_CLAIM_PATH,
        role="execution claim",
    )
    claim = load_json_bytes(claim_capture.data, role="execution claim")
    if _digest(seal.get("execution_id"), name="seal.execution_id") != execution_id:
        raise SecureArtifactError("seal execution identity differs from prediction")
    execution_identity = _mapping(
        seal.get("execution_identity"), name="seal.execution_identity"
    )
    if canonical_sha256(execution_identity) != execution_id:
        raise SecureArtifactError("seal execution identity digest mismatch")
    source_hashes_raw = _mapping(
        seal.get("source_files_sha256"), name="seal.source_files_sha256"
    )
    expected_source_hashes = {
        str(name): _digest(value, name=f"seal source {name}")
        for name, value in source_hashes_raw.items()
    }
    source_captures = capture_repo_files(repository, expected_source_hashes)
    if source_sha256s(source_captures) != expected_source_hashes:
        raise SecureArtifactError("seal source ledger differs from clean tree")
    preregistration_capture = source_captures.get(
        "docs/implicit-solvation/benchmarks/"
        "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
    )
    if preregistration_capture is None:
        raise SecureArtifactError("seal source ledger omitted preregistration")
    preregistration = load_json_bytes(
        preregistration_capture.data, role="captured preregistration"
    )
    source_validation = _mapping(
        seal.get("source_validation"), name="seal.source_validation"
    )
    seal_checkpoints_for_slot = _mapping(
        seal.get("checkpoints"), name="seal.checkpoints"
    )
    expected_attempt_slot_identity = build_attempt_slot_identity(
        preregistration=preregistration,
        source_validation=source_validation,
        protocol_sha256=_digest(
            seal.get("protocol_sha256"), name="seal.protocol_sha256"
        ),
        selection_sha256=_digest(
            seal.get("selection_sha256"), name="seal.selection_sha256"
        ),
        mace_mdp_checkpoint_sha256=_digest(
            seal_checkpoints_for_slot.get("mace_mdp_sha256"),
            name="seal.mace_mdp_sha256",
        ),
        mace_polar_checkpoint_sha256=_digest(
            seal_checkpoints_for_slot.get("mace_polar_sha256"),
            name="seal.mace_polar_sha256",
        ),
    )
    if expected_attempt_slot_identity != attempt_slot_identity:
        raise SecureArtifactError("seal scientific attempt-slot identity drifted")
    git = _mapping(seal.get("git"), name="seal.git")
    if dict(git) != repository.as_dict():
        raise SecureArtifactError("seal Git identity differs from clean tree")
    seal_input = _mapping(seal.get("input_bundle"), name="seal.input_bundle")
    if seal_input.get("file_sha256") != input_capture.sha256 or seal_input.get(
        "content_sha256"
    ) != input_bundle.get("content_sha256"):
        raise SecureArtifactError("seal input binding differs from captured bundle")
    accuracy.validate_label_free_input_bundle(input_bundle, expected_record_count=10)
    accuracy.validate_prediction_terminal(prediction, expected_record_count=10)
    prediction_seal = _mapping(prediction.get("seal"), name="prediction.seal")
    prediction_input = _mapping(
        prediction.get("input_bundle"), name="prediction.input_bundle"
    )
    if {
        "attempt_slot_id": prediction.get("attempt_slot_id"),
        "execution_id": prediction.get("execution_id"),
        "profile_id": prediction.get("profile_id"),
        "scalar_id": prediction.get("scalar_id"),
        "claim_boundary": prediction.get("claim_boundary"),
        "seal_file_sha256": prediction_seal.get("file_sha256"),
        "seal_content_sha256": prediction_seal.get("content_sha256"),
        "input_file_sha256": prediction_input.get("file_sha256"),
        "input_content_sha256": prediction_input.get("content_sha256"),
    } != {
        "attempt_slot_id": seal.get("attempt_slot_id"),
        "execution_id": seal.get("execution_id"),
        "profile_id": seal.get("profile_id"),
        "scalar_id": seal.get("scalar_id"),
        "claim_boundary": seal.get("claim_boundary"),
        "seal_file_sha256": seal_capture.sha256,
        "seal_content_sha256": seal.get("content_sha256"),
        "input_file_sha256": input_capture.sha256,
        "input_content_sha256": input_bundle.get("content_sha256"),
    }:
        raise SecureArtifactError("prediction identity differs from seal/input")
    prediction_checkpoints = _mapping(
        prediction.get("checkpoints"), name="prediction.checkpoints"
    )
    seal_checkpoints = _mapping(seal.get("checkpoints"), name="seal.checkpoints")
    if {
        "mace_mdp_sha256": prediction_checkpoints.get("mace_mdp_sha256"),
        "mace_polar_sha256": prediction_checkpoints.get("mace_polar_sha256"),
    } != dict(seal_checkpoints):
        raise SecureArtifactError("prediction checkpoint identities differ from seal")
    if prediction.get("source_files_sha256") != expected_source_hashes:
        raise SecureArtifactError("prediction source ledger differs from seal")
    prediction_git = _mapping(prediction.get("git"), name="prediction.git")
    if dict(prediction_git) != repository.as_dict():
        raise SecureArtifactError("prediction Git identity differs from clean tree")
    prediction_runtime = _mapping(prediction.get("runtime"), name="prediction.runtime")
    expected_claim_keys = {
        "schema_id",
        "attempt_slot_id",
        "execution_id",
        "seal_file_sha256",
        "seal_content_sha256",
        "preregistration_sha256",
        "source_files_sha256",
        "checkpoints",
        "input_bundle",
        "required_environment",
        "publication_contract",
        "custody_contract",
        "process_uuid",
        "boot_id",
        "process_id",
        "process_start_ticks",
        "process_started_at_utc",
        "terminal_path",
        "rule",
    }
    if set(claim) != expected_claim_keys:
        raise SecureArtifactError("execution claim schema drifted")
    if {
        "schema_id": claim.get("schema_id"),
        "attempt_slot_id": claim.get("attempt_slot_id"),
        "execution_id": claim.get("execution_id"),
        "seal_file_sha256": claim.get("seal_file_sha256"),
        "seal_content_sha256": claim.get("seal_content_sha256"),
        "preregistration_sha256": claim.get("preregistration_sha256"),
        "source_files_sha256": claim.get("source_files_sha256"),
        "checkpoints": claim.get("checkpoints"),
        "input_bundle": claim.get("input_bundle"),
        "required_environment": claim.get("required_environment"),
        "publication_contract": claim.get("publication_contract"),
        "custody_contract": claim.get("custody_contract"),
        "terminal_path": claim.get("terminal_path"),
        "rule": claim.get("rule"),
    } != {
        "schema_id": "maple-route2-accuracy-execution-claim-v1",
        "attempt_slot_id": attempt_slot_id,
        "execution_id": execution_id,
        "seal_file_sha256": seal_capture.sha256,
        "seal_content_sha256": seal.get("content_sha256"),
        "preregistration_sha256": preregistration_capture.sha256,
        "source_files_sha256": expected_source_hashes,
        "checkpoints": dict(seal_checkpoints_for_slot),
        "input_bundle": {
            "file_sha256": input_capture.sha256,
            "content_sha256": input_bundle.get("content_sha256"),
        },
        "required_environment": seal.get("required_environment"),
        "publication_contract": secure_publication_contract(),
        "custody_contract": host_user_custody_contract(),
        "terminal_path": expected_prediction.relative_to(repo_root).as_posix(),
        "rule": (
            "one scientific attempt slot permits exactly one execution and never reopens"
        ),
    }:
        raise SecureArtifactError("execution claim differs from seal/terminal")
    if (
        not isinstance(claim.get("process_uuid"), str)
        or not isinstance(claim.get("process_started_at_utc"), str)
        or not isinstance(claim.get("boot_id"), str)
    ):
        raise SecureArtifactError("execution claim process identity is invalid")
    if (
        type(claim.get("process_id")) is not int
        or type(claim.get("process_start_ticks")) is not int
    ):
        raise SecureArtifactError("execution claim Linux process identity is invalid")
    if (
        claim.get("process_uuid") != prediction_runtime.get("process_uuid")
        or claim.get("boot_id") != prediction_runtime.get("boot_id")
        or claim.get("process_id") != prediction_runtime.get("process_id")
        or claim.get("process_start_ticks")
        != prediction_runtime.get("process_start_ticks")
        or claim.get("process_started_at_utc") != prediction.get("started_at_utc")
    ):
        raise SecureArtifactError(
            "prediction process identity differs from the one-time claim"
        )
    stability = StabilityGuard(
        repository,
        tuple(
            [(f"source {name}", capture) for name, capture in source_captures.items()]
            + [
                ("computation seal", seal_capture),
                ("execution claim", claim_capture),
                ("label-free input bundle", input_capture),
                ("prediction terminal", prediction_capture),
            ]
        ),
    )
    stability.assert_stable()
    return PredictionChain(
        repository=repository,
        attempt_slot_id=attempt_slot_id,
        execution_id=execution_id,
        seal_capture=seal_capture,
        claim_capture=claim_capture,
        input_capture=input_capture,
        prediction_capture=prediction_capture,
        source_captures=source_captures,
        seal=seal,
        claim=claim,
        input_bundle=input_bundle,
        prediction=prediction,
        stability=stability,
    )


__all__ = [
    "ATTEMPT_SLOT_EVIDENCE_CLASS",
    "ATTEMPT_SLOT_CLAIM_PATH",
    "ATTEMPT_SLOT_STAGE_ID",
    "STAGE_CLAIM_REGISTRY",
    "PredictionChain",
    "build_attempt_slot_identity",
    "capture_prediction_chain",
]
