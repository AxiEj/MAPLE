#!/usr/bin/env python3
"""Mechanically validate and aggregate two H1 v2 E/F revalidation records.

This program emits a content-addressed *candidate overlay artifact*.  It never
edits a repository or registry, and successful aggregation is not itself a
capability-admission decision.

All checks finish on one Linux anonymous ``O_TMPFILE`` inode before one atomic
``linkat(AT_EMPTY_PATH)`` publication.  That link is the terminal commit point:
mutation afterward is external tampering that cannot be policed atomically by
this CLI.  Downstream consumers must single-read, hash, and self-validate the
artifact just as this program does for every input.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from typing import Mapping

sys.dont_write_bytecode = True

from maple.solvation.api import capabilities as capabilities_module  # noqa: E402
from maple.solvation.release import admission as admission_module  # noqa: E402
from maple.solvation.release import evidence as evidence_module  # noqa: E402
from maple.solvation.release.admission import (  # noqa: E402
    ADMISSION_OVERLAY_V2_SCHEMA,
    AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA,
    AdmissionOverlayV2,
    AggregateAdmissionInputsV2,
    ComputationSealV2,
    EXPOSURE_AWARE_PROTOCOL_LABEL,
    ReplicateAdmissionRecordV2,
    verified_pro_schema_amendment_contract,
)
from maple.solvation.release.evidence import (  # noqa: E402
    RepositorySnapshot,
    canonical_json_sha256,
)

SCHEMA_ID = "maple-route2-h1-ef-revalidation-aggregate-artifact-v2"
PREREGISTRATION_SCHEMA = (
    "route2-exposure-aware-retrospective-ef-revalidation-protocol-v2"
)
AGGREGATOR_ID = (
    "tools/route2_release/aggregate_mace_mdp_polar_harmonic_ef_revalidation.py"
)
EF_CAPABILITIES = {"E": True, "F": True, "H": False, "V": False, "M": False}
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
PREREGISTRATION_ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-harmonic-ef-revalidation-v2"
PREREGISTRATION_STATUS = (
    "prepared-exposure-aware-retrospective-revalidation-awaiting-clean-commit-"
    "content-addressed-freeze-before-h1-seal"
)
PREREGISTRATION_CLASSIFICATION = {
    "label": EXPOSURE_AWARE_PROTOCOL_LABEL,
    "is_blind_preregistration": False,
    "is_retrospective_revalidation": True,
    "freeze_boundary": (
        "This tracked draft is prepared but not frozen. Only a future clean "
        "committed protocol byte SHA256 bound by the computation seal freezes "
        "the protocol and creates the H1 identity; no cold replay, aggregate "
        "result, or admission overlay may precede that seal."
    ),
    "purpose": (
        "Prepare retrospective revalidation for the pre-H1 development checkout "
        "after acknowledged exposure to the H0/4cf8db40 outcomes; historical "
        "passes are disclosed evidence, not transferable admission."
    ),
    "is_frozen": False,
    "h1_identity_exists": False,
}
IDENTITY_LIFECYCLE = (
    "Reserved contract values only; these fields do not constitute an H1 "
    "identity until a clean committed protocol byte SHA256 is bound by a valid "
    "computation seal."
)
FROZEN_EXECUTION_SEQUENCE = [
    "protocol",
    "computation_seal",
    "cold_replicate_a",
    "cold_replicate_b",
    "mechanical_aggregator",
    "data_only_admission_overlay",
]
SEQUENCE_INVARIANTS = {
    "protocol_precedes_computation_seal": True,
    "computation_seal_precedes_both_cold_runs": True,
    "cold_runs_are_independent_processes": True,
    "mechanical_aggregator_has_no_discretionary_thresholds": True,
    "overlay_is_data_only_and_cannot_change_science": True,
    "failure_retains_negative_evidence": True,
    "failure_capabilities": NO_CAPABILITIES,
}
POST_FREEZE_PROHIBITIONS = {
    "fit": False,
    "calibration": False,
    "case_selection": False,
    "threshold_changes": False,
    "panel_changes": False,
    "geometry_changes": False,
    "scientific_setting_changes": False,
    "decision_rule_changes": False,
    "post_run_protocol_changes": False,
    "policy": (
        "No fit, calibration, case selection, threshold relaxation, panel or "
        "geometry substitution, scientific-setting change, decision-rule change, "
        "or other post-run protocol change is allowed. Any such change creates "
        "a new protocol identity and requires a new seal and fresh cold runs."
    ),
}
EXECUTION_STATE = {
    "computation_seal": None,
    "cold_replicate_a": None,
    "cold_replicate_b": None,
    "aggregate_result": None,
    "admission_overlay": None,
    "capabilities_currently_admitted_by_this_protocol": NO_CAPABILITIES,
}
CLAIM_BOUNDARY_TEXT = {
    "candidate_admission": (
        "Only the exact experimental electrostatic scalar E and its runtime-error-"
        "bounded fourth-order Richardson numerical scalar-gradient F may become "
        "true after the complete v2 sequence passes."
    ),
    "protocol_does_not_admit": (
        "Preparing this protocol admits no capability. Historical H0 passes admit "
        "no capability for the pre-H1 development checkout. Only a future clean "
        "committed protocol SHA256 bound by a valid computation seal, two sealed "
        "cold runs, mechanical aggregation, and a valid data-only overlay may "
        "activate E/F for the resulting H1 identity."
    ),
}


class AggregationInputError(ValueError):
    """Raised when aggregation cannot prove a coherent, immutable input set."""


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_Identity":
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_DirectoryIdentity":
        return cls(value.st_dev, value.st_ino, value.st_mode)


@dataclass(frozen=True, slots=True)
class _Capture:
    role: str
    path: Path
    identity: _Identity
    data: bytes
    sha256: str
    payload: object

    def record(self, root: Path) -> dict[str, object]:
        try:
            displayed_path = self.path.relative_to(root).as_posix()
            location = "repository"
        except ValueError:
            displayed_path = str(self.path)
            location = "external"
        return {
            "role": self.role,
            "path": displayed_path,
            "location": location,
            "sha256": self.sha256,
            "size_bytes": len(self.data),
        }

    def assert_stable(self) -> None:
        current = _path_identity(self.path, role=self.role)
        if current != self.identity:
            raise AggregationInputError(f"{self.role} changed during aggregation")


@dataclass(frozen=True, slots=True)
class _Stability:
    repository: RepositorySnapshot
    captures: tuple[_Capture, ...]

    def assert_stable(self) -> None:
        try:
            self.repository.assert_unchanged()
        except (OSError, RuntimeError) as exc:
            raise AggregationInputError(
                f"repository changed during aggregation: {exc}"
            ) from exc
        for capture in self.captures:
            capture.assert_stable()


def _after_input_capture() -> None:
    """Deterministic test hook for input races."""


def _after_anonymous_write(descriptor: int) -> None:
    """Deterministic test hook after durable anonymous-inode creation."""


def _after_anonymous_verification(descriptor: int) -> None:
    """Deterministic test hook before the final pre-commit checks."""


def _run_git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise AggregationInputError(
            f"Git command {arguments!r} failed: {diagnostic.strip()}"
        )
    return result.stdout


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _reject_symlink_components(path: Path, *, role: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            value = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AggregationInputError(f"cannot inspect {role} path: {exc}") from exc
        if stat.S_ISLNK(value.st_mode):
            raise AggregationInputError(f"{role} path must not contain symbolic links")


def _path_identity(path: Path, *, role: str) -> _Identity:
    _reject_symlink_components(path, role=role)
    try:
        identity = _Identity.from_stat(path.stat())
    except OSError as exc:
        raise AggregationInputError(f"cannot inspect {role}: {exc}") from exc
    if not stat.S_ISREG(identity.mode):
        raise AggregationInputError(f"{role} must be a regular file")
    if identity.links != 1:
        raise AggregationInputError(f"{role} must not be a hard-linked file")
    return identity


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AggregationInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> object:
    raise AggregationInputError(f"non-finite JSON number is forbidden: {value}")


def _read_json_once(path: str | Path, *, role: str) -> _Capture:
    capture = _read_file_once(path, role=role)
    try:
        payload = json.loads(
            capture.data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_invalid_json_constant,
        )
    except AggregationInputError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationInputError(f"cannot parse {role}: {exc}") from exc
    return _Capture(
        role=role,
        path=capture.path,
        identity=capture.identity,
        data=capture.data,
        sha256=capture.sha256,
        payload=payload,
    )


def _mapping(payload: object, *, role: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise AggregationInputError(f"{role} JSON must contain an object")
    return payload


def _repo_relative(path: Path, root: Path, *, role: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AggregationInputError(f"{role} must be inside the repository") from exc
    text = relative.as_posix()
    parsed = PurePosixPath(text)
    if (
        not text
        or parsed.is_absolute()
        or any(part in ("", ".", "..") for part in parsed.parts)
    ):
        raise AggregationInputError(f"{role} has an invalid repository-relative path")
    return text


def _require_external(capture: _Capture, root: Path) -> None:
    try:
        capture.path.relative_to(root)
    except ValueError:
        return
    raise AggregationInputError(f"{capture.role} must be external to the repository")


def _validate_preregistration(
    capture: _Capture, *, repository: RepositorySnapshot, seal: ComputationSealV2
) -> None:
    relative = _repo_relative(capture.path, repository.root, role=capture.role)
    tracked = set(
        _run_git(repository.root, "ls-tree", "-r", "--name-only", repository.head)
        .decode("utf-8")
        .splitlines()
    )
    if relative not in tracked:
        raise AggregationInputError(
            "preregistration must be committed at repository HEAD"
        )
    committed = _run_git(repository.root, "show", f"{repository.head}:{relative}")
    if committed != capture.data:
        raise AggregationInputError("preregistration bytes differ from committed HEAD")
    payload = _mapping(capture.payload, role=capture.role)
    _validate_preregistration_contract(payload, seal=seal)
    if capture.sha256 != seal.preregistration_sha256:
        raise AggregationInputError("preregistration file SHA256 differs from the seal")


def _validate_preregistration_contract(
    payload: Mapping[str, object], *, seal: ComputationSealV2
) -> None:
    if payload.get("artifact_id") != PREREGISTRATION_ARTIFACT_ID:
        raise AggregationInputError(
            "preregistration artifact_id is not the H1 v2 protocol"
        )
    if payload["artifact_id"] != seal.preregistration_id:
        raise AggregationInputError("preregistration artifact_id differs from the seal")
    expected_identity = {
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "claim_boundary_id": seal.claim_boundary_id,
        "lifecycle": IDENTITY_LIFECYCLE,
    }
    expected_protocol = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": PREREGISTRATION_STATUS,
        "protocol_classification": PREREGISTRATION_CLASSIFICATION,
        "prospective_h1_identity_contract": expected_identity,
        "target_capabilities": EF_CAPABILITIES,
        "prospective_h1_scientific_settings": seal.as_dict()["scientific_settings"],
        "runtime_guards": list(seal.runtime_guards),
        "domain_guards": list(seal.domain_guards),
        "frozen_execution_sequence": FROZEN_EXECUTION_SEQUENCE,
        "sequence_invariants": SEQUENCE_INVARIANTS,
        "post_freeze_prohibitions": POST_FREEZE_PROHIBITIONS,
        "execution_state": EXECUTION_STATE,
        "verified_pro_schema_amendment": verified_pro_schema_amendment_contract(),
    }
    for field, expected in expected_protocol.items():
        if payload.get(field) != expected:
            raise AggregationInputError(
                f"preregistration {field} differs from the sealed v2 contract"
            )
    amendment = verified_pro_schema_amendment_contract()
    assets = dict(seal.asset_sha256s)
    if assets["verified_pro_audit"] != amendment["verified_pro_audit"][
        "raw_file_sha256"
    ]:
        raise AggregationInputError(
            "verified-Pro audit asset differs from the protocol amendment"
        )
    expected_claim = {
        "id": seal.claim_boundary_id,
        **CLAIM_BOUNDARY_TEXT,
        "non_admissions": list(seal.non_admissions),
    }
    if payload.get("claim_boundary") != expected_claim:
        raise AggregationInputError(
            "preregistration claim boundary differs from the computation seal"
        )
    parent_bindings = payload.get("parent_bindings")
    if not isinstance(parent_bindings, Mapping):
        raise AggregationInputError("preregistration parent_bindings must be an object")
    v1_preregistration = parent_bindings.get("v1_preregistration")
    v1_parent_panel = parent_bindings.get("v1_parent_panel")
    if not isinstance(v1_preregistration, Mapping) or not isinstance(
        v1_parent_panel, Mapping
    ):
        raise AggregationInputError("preregistration parent bindings are incomplete")
    assets = dict(seal.asset_sha256s)
    if assets["v1_preregistration"] != v1_preregistration.get("sha256"):
        raise AggregationInputError(
            "v1_preregistration asset differs from the protocol parent binding"
        )
    if assets["parent_panel"] != v1_parent_panel.get("sha256"):
        raise AggregationInputError(
            "parent_panel asset differs from the protocol parent binding"
        )


def _execution_source_relatives(repository: RepositorySnapshot) -> tuple[str, ...]:
    execution_root = Path(__file__).resolve(strict=True).parents[2]
    if execution_root != repository.root:
        raise AggregationInputError(
            "running aggregator does not originate from the sealed repository root"
        )
    paths = (
        Path(__file__).resolve(strict=True),
        Path(admission_module.__file__).resolve(strict=True),
        Path(evidence_module.__file__).resolve(strict=True),
        Path(capabilities_module.__file__).resolve(strict=True),
    )
    relatives: list[str] = []
    for path in paths:
        try:
            relatives.append(path.relative_to(repository.root).as_posix())
        except ValueError as exc:
            raise AggregationInputError(
                f"executed validation module is outside the sealed repository: {path}"
            ) from exc
    return tuple(relatives)


def _capture_and_validate_sources(
    repository: RepositorySnapshot, seal: ComputationSealV2
) -> tuple[_Capture, ...]:
    ledger = dict(seal.computation_source_sha256s)
    if seal.aggregator_id != AGGREGATOR_ID:
        raise AggregationInputError(
            "seal aggregator_id is not this mechanical aggregator"
        )
    if ledger.get(seal.aggregator_id) != seal.aggregator_sha256:
        raise AggregationInputError(
            "seal aggregator identity is not bound to its source ledger"
        )
    if ledger.get(seal.runner_id) != seal.runner_sha256:
        raise AggregationInputError(
            "seal runner identity is not bound to its source ledger"
        )
    execution_sources = _execution_source_relatives(repository)
    missing_execution_sources = sorted(set(execution_sources) - set(ledger))
    if missing_execution_sources:
        raise AggregationInputError(
            "seal source ledger omits executed validation modules: "
            + ", ".join(missing_execution_sources)
        )

    captures: list[_Capture] = []
    for relative, expected in sorted(ledger.items()):
        committed = _run_git(repository.root, "show", f"{repository.head}:{relative}")
        if hashlib.sha256(committed).hexdigest() != expected:
            raise AggregationInputError(
                f"seal source ledger differs from committed HEAD: {relative!r}"
            )
        capture = _read_file_once(repository.root / relative, role=f"source {relative}")
        if capture.data != committed or capture.sha256 != expected:
            raise AggregationInputError(
                f"current source bytes differ from committed HEAD: {relative!r}"
            )
        captures.append(capture)
    return tuple(captures)


def _read_file_once(path: str | Path, *, role: str) -> _Capture:
    """Read an arbitrary regular file once with the same coherence contract."""
    lexical = _absolute_lexical(path)
    _reject_symlink_components(lexical, role=role)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        before = _Identity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before.links != 1:
            raise AggregationInputError(
                f"{role} must be a non-hard-linked regular file"
            )
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = _Identity.from_stat(os.fstat(descriptor))
    except AggregationInputError:
        raise
    except OSError as exc:
        raise AggregationInputError(f"cannot read {role}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    data = b"".join(blocks)
    if before != after or len(data) != before.size:
        raise AggregationInputError(f"{role} changed while being read")
    resolved = lexical.resolve(strict=True)
    if _Identity.from_stat(resolved.stat()) != before:
        raise AggregationInputError(f"{role} path identity changed while being read")
    return _Capture(
        role, resolved, before, data, hashlib.sha256(data).hexdigest(), None
    )


def _typed_aggregate(
    seal: ComputationSealV2,
    replicate_a: ReplicateAdmissionRecordV2,
    replicate_b: ReplicateAdmissionRecordV2,
) -> AggregateAdmissionInputsV2:
    content: dict[str, object] = {
        "schema_id": AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA,
        "aggregate_id": f"{seal.artifact_id}-replicated-ef-inputs",
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "capabilities": EF_CAPABILITIES,
        "runtime_guards": list(seal.runtime_guards),
        "domain_guards": list(seal.domain_guards),
        "claim_boundary_id": seal.claim_boundary_id,
        "non_admissions": list(seal.non_admissions),
        "replicates": [replicate_a.as_dict(), replicate_b.as_dict()],
        "aggregate_gate_results": {
            name: True for name in admission_module.AGGREGATE_GATE_NAMES
        },
    }
    content["aggregate_sha256"] = canonical_json_sha256(content)
    return AggregateAdmissionInputsV2.from_mapping(content, seal=seal)


def _typed_overlay(
    seal: ComputationSealV2, aggregate: AggregateAdmissionInputsV2
) -> AdmissionOverlayV2:
    content: dict[str, object] = {
        "schema_id": ADMISSION_OVERLAY_V2_SCHEMA,
        "overlay_id": f"{seal.artifact_id}-candidate-ef-overlay",
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "aggregate_id": aggregate.aggregate_id,
        "aggregate_sha256": aggregate.aggregate_sha256,
        "profile_id": aggregate.profile_id,
        "scalar_id": aggregate.scalar_id,
        "state_id": aggregate.state_id,
        "capabilities": EF_CAPABILITIES,
        "runtime_guards": list(aggregate.runtime_guards),
        "domain_guards": list(aggregate.domain_guards),
        "claim_boundary_id": aggregate.claim_boundary_id,
        "non_admissions": list(aggregate.non_admissions),
    }
    content["content_sha256"] = canonical_json_sha256(content)
    return AdmissionOverlayV2.from_mapping(content, seal=seal, aggregate=aggregate)


def _validate_process_start_order(started_a: str, started_b: str) -> None:
    parsed_a = datetime.fromisoformat(started_a.replace("Z", "+00:00"))
    parsed_b = datetime.fromisoformat(started_b.replace("Z", "+00:00"))
    if parsed_a >= parsed_b:
        raise AggregationInputError(
            "replicate a process must start strictly before replicate b"
        )


def _prepare(
    repo_root: str | Path,
    preregistration_path: str | Path,
    seal_path: str | Path,
    replicate_a_path: str | Path,
    replicate_b_path: str | Path,
) -> tuple[dict[str, object], _Stability]:
    try:
        repository = RepositorySnapshot.capture(repo_root, require_clean=True)
    except (OSError, RuntimeError) as exc:
        raise AggregationInputError(f"invalid clean repository: {exc}") from exc

    preregistration = _read_json_once(preregistration_path, role="preregistration")
    seal_capture = _read_json_once(seal_path, role="computation seal")
    replicate_a_capture = _read_json_once(replicate_a_path, role="replicate a")
    replicate_b_capture = _read_json_once(replicate_b_path, role="replicate b")
    for external in (seal_capture, replicate_a_capture, replicate_b_capture):
        _require_external(external, repository.root)
    identities = {
        (capture.identity.device, capture.identity.inode)
        for capture in (
            preregistration,
            seal_capture,
            replicate_a_capture,
            replicate_b_capture,
        )
    }
    if len(identities) != 4:
        raise AggregationInputError(
            "all input JSON files must have distinct identities"
        )

    seal = ComputationSealV2.from_mapping(
        _mapping(seal_capture.payload, role=seal_capture.role)
    )
    if seal.git_head != repository.head or seal.git_tree != repository.tree:
        raise AggregationInputError(
            "seal Git HEAD/tree differs from the clean repository"
        )
    _validate_preregistration(preregistration, repository=repository, seal=seal)
    source_captures = _capture_and_validate_sources(repository, seal)

    replicate_a = ReplicateAdmissionRecordV2.from_mapping(
        _mapping(replicate_a_capture.payload, role=replicate_a_capture.role), seal=seal
    )
    replicate_b = ReplicateAdmissionRecordV2.from_mapping(
        _mapping(replicate_b_capture.payload, role=replicate_b_capture.role), seal=seal
    )
    if replicate_a.label != "a" or replicate_b.label != "b":
        raise AggregationInputError(
            "replicate files must carry labels a and b respectively"
        )
    _validate_process_start_order(
        replicate_a.process_started_at_utc,
        replicate_b.process_started_at_utc,
    )
    aggregate = _typed_aggregate(seal, replicate_a, replicate_b)
    overlay = _typed_overlay(seal, aggregate)
    if aggregate.as_dict()["capabilities"] != EF_CAPABILITIES:
        raise AggregationInputError(
            "aggregate capability boundary widened unexpectedly"
        )
    if overlay.as_dict()["capabilities"] != EF_CAPABILITIES:
        raise AggregationInputError("overlay capability boundary widened unexpectedly")

    input_captures = (
        preregistration,
        seal_capture,
        replicate_a_capture,
        replicate_b_capture,
    )
    artifact: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "artifact_id": f"{seal.artifact_id}-mechanical-aggregation",
        "status": "validated-candidate-overlay-not-admitted",
        "repository": {
            "head": repository.head,
            "tree": repository.tree,
            "clean": True,
        },
        "aggregate_inputs": aggregate.as_dict(),
        "admission_overlay": overlay.as_dict(),
        "input_files": [capture.record(repository.root) for capture in input_captures],
        "registry_mutation_performed": False,
        "capability_admission_claimed": False,
        "post_commit_integrity_boundary": (
            "linkat(AT_EMPTY_PATH) of the verified O_TMPFILE inode is the terminal "
            "commit point; every "
            "downstream consumer must single-read, hash, and self-validate this "
            "content-addressed artifact before use"
        ),
    }
    artifact["content_sha256"] = canonical_json_sha256(artifact)
    stability = _Stability(repository, (*input_captures, *source_captures))
    return artifact, stability


def aggregate_revalidation(
    repo_root: str | Path,
    preregistration_path: str | Path,
    seal_path: str | Path,
    replicate_a_path: str | Path,
    replicate_b_path: str | Path,
) -> dict[str, object]:
    artifact, stability = _prepare(
        repo_root,
        preregistration_path,
        seal_path,
        replicate_a_path,
        replicate_b_path,
    )
    _after_input_capture()
    stability.assert_stable()
    return artifact


def _directory_identity(descriptor: int) -> _DirectoryIdentity:
    identity = _DirectoryIdentity.from_stat(os.fstat(descriptor))
    if not stat.S_ISDIR(identity.mode):
        raise AggregationInputError("output parent descriptor is not a directory")
    return identity


def _output_is_absent(descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise AggregationInputError(f"cannot inspect output entry: {exc}") from exc
    return False


def _open_output_parent(
    path: str | Path, *, root: Path
) -> tuple[Path, str, int, _DirectoryIdentity]:
    lexical = _absolute_lexical(path)
    _reject_symlink_components(lexical, role="output")
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as exc:
        raise AggregationInputError(f"output parent does not exist: {exc}") from exc
    output = parent / lexical.name
    if not parent.is_dir():
        raise AggregationInputError("output parent must be a directory")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise AggregationInputError("output must be external to the repository")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise AggregationInputError(
            f"cannot open output parent directory: {exc}"
        ) from exc
    try:
        identity = _directory_identity(descriptor)
        if _DirectoryIdentity.from_stat(parent.stat()) != identity:
            raise AggregationInputError("output parent changed while being opened")
        if not _output_is_absent(descriptor, output.name):
            raise AggregationInputError(
                "output already exists; refusing to overwrite it"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return parent, output.name, descriptor, identity


def _assert_output_parent_stable(
    parent: Path, descriptor: int, identity: _DirectoryIdentity
) -> None:
    _reject_symlink_components(parent, role="output parent")
    try:
        descriptor_now = _directory_identity(descriptor)
        path_now = _DirectoryIdentity.from_stat(parent.stat())
    except OSError as exc:
        raise AggregationInputError(f"output parent changed: {exc}") from exc
    if descriptor_now != identity or path_now != identity:
        raise AggregationInputError(
            "output parent directory changed during publication"
        )


def _verify_anonymous_descriptor(descriptor: int, expected: bytes) -> _Identity:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = _Identity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before.links != 0:
            raise AggregationInputError(
                "anonymous output must be an unlinked regular file"
            )
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = _Identity.from_stat(os.fstat(descriptor))
    except AggregationInputError:
        raise
    except OSError as exc:
        raise AggregationInputError(f"cannot verify anonymous output: {exc}") from exc
    actual = b"".join(blocks)
    if before != after or actual != expected:
        raise AggregationInputError("anonymous output bytes changed or differ")
    if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
        raise AggregationInputError("anonymous output SHA256 differs")
    return after


def _open_anonymous_output(directory_descriptor: int) -> int:
    temporary_flag = getattr(os, "O_TMPFILE", None)
    if not isinstance(temporary_flag, int) or temporary_flag == 0:
        raise AggregationInputError("Linux O_TMPFILE is unavailable")
    flags = temporary_flag | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(".", flags, 0o600, dir_fd=directory_descriptor)
    except OSError as exc:
        raise AggregationInputError(
            f"output filesystem does not support secure O_TMPFILE creation: {exc}"
        ) from exc
    identity = _Identity.from_stat(os.fstat(descriptor))
    if not stat.S_ISREG(identity.mode) or identity.links != 0:
        os.close(descriptor)
        raise AggregationInputError(
            "O_TMPFILE did not create an anonymous regular inode"
        )
    return descriptor


def _link_anonymous_noreplace(
    temporary_descriptor: int,
    directory_descriptor: int,
    output_name: str,
) -> None:
    """Atomically publish the open inode, using Linux's documented fallback.

    ``open(2)`` documents linking an ``O_TMPFILE`` inode either with
    ``linkat(..., AT_EMPTY_PATH)`` or, when that operation is unavailable to an
    unprivileged process, through ``/proc/self/fd/<fd>`` with
    ``AT_SYMLINK_FOLLOW``.  Both forms link the exact open process-owned inode,
    and ordinary hard-link semantics fail with ``EEXIST`` rather than replace.
    """

    direct_error = _call_linkat(
        temporary_descriptor,
        b"",
        directory_descriptor,
        os.fsencode(output_name),
        0x1000,  # AT_EMPTY_PATH
    )
    if direct_error is None:
        _verify_linked_inode(temporary_descriptor, directory_descriptor, output_name)
        return
    if direct_error == errno.EEXIST:
        raise AggregationInputError("output appeared during publication")
    if direct_error not in (errno.ENOENT, errno.EPERM, errno.EOPNOTSUPP):
        raise AggregationInputError(
            "atomic anonymous-inode publication failed: " + os.strerror(direct_error)
        )

    proc_path = f"/proc/self/fd/{temporary_descriptor}"
    try:
        proc_identity = _Identity.from_stat(os.stat(proc_path))
        descriptor_identity = _Identity.from_stat(os.fstat(temporary_descriptor))
    except OSError as exc:
        raise AggregationInputError(
            f"secure /proc/self/fd fallback is unavailable: {exc}"
        ) from exc
    if proc_identity != descriptor_identity:
        raise AggregationInputError(
            "/proc/self/fd fallback does not identify the anonymous output inode"
        )
    fallback_error = _call_linkat(
        -100,  # AT_FDCWD
        os.fsencode(proc_path),
        directory_descriptor,
        os.fsencode(output_name),
        0x400,  # AT_SYMLINK_FOLLOW
    )
    if fallback_error is None:
        _verify_linked_inode(temporary_descriptor, directory_descriptor, output_name)
        return
    if fallback_error == errno.EEXIST:
        raise AggregationInputError("output appeared during publication")
    raise AggregationInputError(
        "secure /proc/self/fd anonymous-inode publication failed: "
        + os.strerror(fallback_error)
    )


def _call_linkat(
    old_directory_descriptor: int,
    old_path: bytes,
    new_directory_descriptor: int,
    new_path: bytes,
    flags: int,
) -> int | None:
    """Return ``None`` on linkat success, otherwise the captured errno."""

    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise AggregationInputError(
            "atomic anonymous-inode publication requires linkat"
        )
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    result = linkat(
        old_directory_descriptor,
        old_path,
        new_directory_descriptor,
        new_path,
        flags,
    )
    if result == 0:
        return None
    return ctypes.get_errno()


def _verify_linked_inode(
    temporary_descriptor: int,
    directory_descriptor: int,
    output_name: str,
) -> None:
    try:
        descriptor_identity = _Identity.from_stat(os.fstat(temporary_descriptor))
        output_identity = _Identity.from_stat(
            os.stat(
                output_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        )
    except OSError as exc:
        raise AggregationInputError(
            f"cannot verify published anonymous inode: {exc}"
        ) from exc
    if descriptor_identity != output_identity:
        raise AggregationInputError(
            "published output inode differs from the anonymous descriptor"
        )


def _publish_external_output(
    path: str | Path,
    data: bytes,
    *,
    stability: _Stability,
) -> None:
    parent, output_name, directory_descriptor, parent_identity = _open_output_parent(
        path, root=stability.repository.root
    )
    anonymous_descriptor: int | None = None
    try:
        anonymous_descriptor = _open_anonymous_output(directory_descriptor)
        view = memoryview(data)
        while view:
            written = os.write(anonymous_descriptor, view)
            if written <= 0:
                raise AggregationInputError("cannot write complete anonymous output")
            view = view[written:]
        os.fsync(anonymous_descriptor)
        _after_anonymous_write(anonymous_descriptor)
        anonymous_identity = _verify_anonymous_descriptor(anonymous_descriptor, data)
        _after_anonymous_verification(anonymous_descriptor)
        stability.assert_stable()
        _assert_output_parent_stable(parent, directory_descriptor, parent_identity)
        if _Identity.from_stat(os.fstat(anonymous_descriptor)) != anonymous_identity:
            raise AggregationInputError(
                "anonymous output inode changed after byte verification"
            )
        if not _output_is_absent(directory_descriptor, output_name):
            raise AggregationInputError("output appeared during publication")
        _link_anonymous_noreplace(
            anonymous_descriptor, directory_descriptor, output_name
        )
        os.fsync(directory_descriptor)
    finally:
        if anonymous_descriptor is not None:
            os.close(anonymous_descriptor)
        os.close(directory_descriptor)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--replicate-a", type=Path, required=True)
    parser.add_argument("--replicate-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifact, stability = _prepare(
            args.repo_root,
            args.preregistration,
            args.seal,
            args.replicate_a,
            args.replicate_b,
        )
        _after_input_capture()
        stability.assert_stable()
        encoded = (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _publish_external_output(args.output, encoded, stability=stability)
    except (AggregationInputError, TypeError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
