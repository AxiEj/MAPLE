#!/usr/bin/env python3
"""Close an orphaned hybrid MNSol-10 stage claim as aborted-unresolved.

Recovery is model-free and anchored only to the durable computation seal and
host-user-global stage claim.  It deliberately does not require the post-crash
worktree, checkpoints, input bundle, or scientific providers to remain present.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._hybrid_mnsol10_chain import (  # noqa: E402
    ATTEMPT_SLOT_CLAIM_PATH,
)
from tools.route2_release._secure_artifacts import (  # noqa: E402
    SecureArtifactError,
    canonical_sha256,
    capture_file,
    host_user_custody_contract,
    linux_boot_id,
    linux_process_start_ticks,
    load_json_bytes,
    publish_json_noreplace,
    secure_publication_contract,
)

SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
RUN_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/runs"
ACCURACY_SOURCE = "maple/solvation/release/accuracy_admission.py"
CLAIM_RULE = (
    "one scientific attempt slot permits exactly one execution and never reopens"
)
CAPABILITIES_CLOSED = {tier: False for tier in ("E", "F", "H", "V", "M")}


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


def _self_digest(payload: Mapping[str, object], *, name: str) -> None:
    expected = _digest(payload.get("content_sha256"), name=f"{name}.content_sha256")
    content = dict(payload)
    del content["content_sha256"]
    if canonical_sha256(content) != expected:
        raise SecureArtifactError(f"{name} content digest mismatch")


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
            "orphan recovery process contains model/continuum modules: "
            + ", ".join(loaded[:12])
        )


def _git_blob(head: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "show", f"{head}:{relative_path}"),
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise SecureArtifactError(
            f"cannot recover sealed source {relative_path}: {detail.strip()}"
        )
    return result.stdout


def _load_accuracy_contract(seal: Mapping[str, object]) -> ModuleType:
    git = _mapping(seal.get("git"), name="seal.git")
    head = git.get("head")
    if not isinstance(head, str) or len(head) != 40:
        raise SecureArtifactError("seal Git head is invalid")
    source_hashes = _mapping(
        seal.get("source_files_sha256"), name="seal.source_files_sha256"
    )
    expected = _digest(source_hashes.get(ACCURACY_SOURCE), name="accuracy source")
    source = _git_blob(head, ACCURACY_SOURCE)
    if hashlib.sha256(source).hexdigest() != expected:
        raise SecureArtifactError("sealed accuracy contract source digest drifted")
    module = ModuleType("_maple_route2_recovered_accuracy_contract")
    module.__file__ = f"git:{head}:{ACCURACY_SOURCE}"
    module.__package__ = ""
    sys.modules[module.__name__] = module
    try:
        exec(
            compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__
        )
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def _claim_process_is_alive(claim: Mapping[str, object]) -> bool:
    claimed_boot_id = claim.get("boot_id")
    process_id = claim.get("process_id")
    start_ticks = claim.get("process_start_ticks")
    if not isinstance(claimed_boot_id, str):
        raise SecureArtifactError("claim Linux boot identity is invalid")
    if type(process_id) is not int or type(start_ticks) is not int:
        raise SecureArtifactError("claim Linux process identity is invalid")
    if linux_boot_id() != claimed_boot_id:
        return False
    if not Path(f"/proc/{process_id}/stat").exists():
        return False
    try:
        current = linux_process_start_ticks(process_id)
    except SecureArtifactError as exc:
        raise SecureArtifactError(
            "cannot safely inspect the claimed Linux process"
        ) from exc
    return current == start_ticks


def recover(seal_path: str | Path) -> tuple[Path, Mapping[str, object]]:
    _require_model_free_process()
    seal_capture = capture_file(seal_path, role="computation seal")
    seal = load_json_bytes(seal_capture.data, role="computation seal")
    if seal.get("schema_id") != "maple-route2-accuracy-computation-seal-v1":
        raise SecureArtifactError("unsupported computation seal")
    _self_digest(seal, name="computation seal")
    execution_id = _digest(seal.get("execution_id"), name="execution_id")
    attempt_slot_id = _digest(seal.get("attempt_slot_id"), name="attempt_slot_id")
    attempt_identity = _mapping(
        seal.get("attempt_slot_identity"), name="attempt_slot_identity"
    )
    execution_identity = _mapping(
        seal.get("execution_identity"), name="execution_identity"
    )
    if (
        canonical_sha256(attempt_identity) != attempt_slot_id
        or canonical_sha256(execution_identity) != execution_id
    ):
        raise SecureArtifactError("seal execution/attempt identity digest mismatch")
    if seal_capture.path != (SEAL_DIRECTORY / f"{execution_id}.json").resolve():
        raise SecureArtifactError("computation seal path is not canonical")
    if seal.get("publication_contract") != secure_publication_contract():
        raise SecureArtifactError("seal publication contract drifted")
    if seal.get("custody_contract") != host_user_custody_contract():
        raise SecureArtifactError("seal custody contract drifted")
    terminal_path = RUN_DIRECTORY / execution_id / "prediction-terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        raise SecureArtifactError("attempt already has a prediction terminal")
    claim_capture = capture_file(ATTEMPT_SLOT_CLAIM_PATH, role="stage claim")
    claim = load_json_bytes(claim_capture.data, role="stage claim")
    seal_preregistration = _mapping(
        seal.get("preregistration"), name="seal.preregistration"
    )
    seal_input = _mapping(seal.get("input_bundle"), name="seal.input_bundle")
    seal_checkpoints = _mapping(seal.get("checkpoints"), name="seal.checkpoints")
    expected_keys = {
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
    if set(claim) != expected_keys or {
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
        "preregistration_sha256": seal_preregistration.get("sha256"),
        "source_files_sha256": seal.get("source_files_sha256"),
        "checkpoints": dict(seal_checkpoints),
        "input_bundle": {
            "file_sha256": seal_input.get("file_sha256"),
            "content_sha256": seal_input.get("content_sha256"),
        },
        "required_environment": seal.get("required_environment"),
        "publication_contract": secure_publication_contract(),
        "custody_contract": host_user_custody_contract(),
        "terminal_path": terminal_path.relative_to(REPO_ROOT).as_posix(),
        "rule": CLAIM_RULE,
    }:
        raise SecureArtifactError("stage claim differs from durable seal")
    if not isinstance(claim.get("process_uuid"), str) or not isinstance(
        claim.get("process_started_at_utc"), str
    ):
        raise SecureArtifactError("stage claim process identity is invalid")
    if _claim_process_is_alive(claim):
        raise SecureArtifactError("claimed prediction process is still alive")
    accuracy = _load_accuracy_contract(seal)
    terminal: dict[str, object] = {
        "schema_id": "maple-route2-accuracy-prediction-terminal-v1",
        "artifact_id": (
            "route2-mace-mdp-polar-hybrid-mnsol10-label-free-predictions-v1"
        ),
        "status": "failure",
        "attempt_slot_id": attempt_slot_id,
        "execution_id": execution_id,
        "started_at_utc": claim["process_started_at_utc"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": seal["claim_boundary"],
        "profile_id": seal["profile_id"],
        "scalar_id": seal["scalar_id"],
        "git": seal["git"],
        "seal": {
            "file_sha256": seal_capture.sha256,
            "content_sha256": seal["content_sha256"],
        },
        "input_bundle": {
            "file_sha256": seal_input["file_sha256"],
            "content_sha256": seal_input["content_sha256"],
        },
        "checkpoints": dict(seal_checkpoints),
        "source_files_sha256": seal["source_files_sha256"],
        "runtime": {
            "environment": claim["required_environment"],
            "process_uuid": claim["process_uuid"],
            "boot_id": claim["boot_id"],
            "process_id": claim["process_id"],
            "process_start_ticks": claim["process_start_ticks"],
            "recovery": {
                "contract": "maple-route2-orphaned-attempt-recovery-v1",
                "terminal_type": "aborted-unresolved",
                "scientific_result": "unavailable",
                "attempt_slot_closed": True,
            },
        },
        "record_count_expected": 10,
        "records": [],
        "timing_seconds": {"total_wall": 0.0},
        "capabilities": CAPABILITIES_CLOSED,
        "failure": {
            "stage": "aborted-unresolved",
            "error_type": "maple.route2.release.AbortedUnresolvedAttempt",
            "message": (
                "durable stage claim exists, the recorded process is dead, and no "
                "valid prediction terminal was published"
            ),
            "failed_selection_index": None,
            "failed_opaque_record_id": None,
        },
    }
    terminal = accuracy.with_content_sha256(terminal)
    accuracy.validate_prediction_terminal(terminal, expected_record_count=10)
    _require_model_free_process()

    def captured_stability() -> None:
        seal_capture.assert_stable(role="computation seal")
        claim_capture.assert_stable(role="stage claim")

    publish_json_noreplace(
        terminal_path,
        terminal,
        root=REPO_ROOT,
        stability=captured_stability,
    )
    return terminal_path, terminal


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        output, terminal = recover(args.seal)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"closed attempt_slot_id={terminal['attempt_slot_id']} as "
        f"aborted-unresolved at {os.fspath(output)}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
