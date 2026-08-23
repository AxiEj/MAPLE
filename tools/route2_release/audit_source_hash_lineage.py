#!/usr/bin/env python3
"""Read-only audit of an evidence artifact's repository source lineage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat as stat_module
import sys
from typing import Mapping

# An audit must not change the checkout merely by importing its hash helpers.
sys.dont_write_bytecode = True

from maple.solvation.release import (  # noqa: E402
    RepositorySnapshot,
    canonical_json_sha256,
)
from maple.solvation.release.evidence import sha256_file  # noqa: E402


SCHEMA_VERSION = "route2-source-hash-lineage-audit-v1"
DEFAULT_SOURCE_HASH_FIELD = "source_files_sha256"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AuditInputError(ValueError):
    """Raised when an audit request or evidence ledger is malformed."""


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, stat: os.stat_result) -> "_FileIdentity":
        return cls(
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class _SourceCapture:
    path: Path
    identity: _FileIdentity
    sha256: str


@dataclass(frozen=True, slots=True)
class _AuditStability:
    snapshot: RepositorySnapshot
    evidence_path: Path
    evidence_identity: _FileIdentity
    sources: tuple[_SourceCapture, ...]
    missing_sources: tuple[Path, ...]

    def assert_stable(self) -> None:
        try:
            self.snapshot.assert_unchanged()
            evidence_now = _FileIdentity.from_stat(self.evidence_path.stat())
        except (OSError, RuntimeError) as exc:
            raise AuditInputError(
                f"audit inputs changed during capture: {exc}"
            ) from exc
        if evidence_now != self.evidence_identity:
            raise AuditInputError("evidence artifact changed during audit")
        for source in self.sources:
            current = _stable_source_hash(self.snapshot.root, source.path)
            if current.identity != source.identity or current.sha256 != source.sha256:
                relative = source.path.relative_to(self.snapshot.root).as_posix()
                raise AuditInputError(
                    f"source file changed during audit: {relative!r}"
                )
        for missing in self.missing_sources:
            _non_symlink_component_identities(self.snapshot.root, missing)
            if missing.exists() or missing.is_symlink():
                relative = missing.relative_to(self.snapshot.root).as_posix()
                raise AuditInputError(
                    f"missing source path changed during audit: {relative!r}"
                )


def _after_initial_capture() -> None:
    """Deterministic test hook for changes between capture and validation."""


def _after_output_write() -> None:
    """Deterministic test hook for changes immediately after output."""


def _repository_relative_path(raw: object) -> PurePosixPath:
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or "\0" in raw
        or (len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":")
    ):
        raise AuditInputError(
            "source paths must be nonempty repository-relative POSIX paths"
        )
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise AuditInputError(f"invalid repository-relative source path: {raw!r}")
    return path


def _source_ledger(payload: object, field: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise AuditInputError("evidence JSON must contain an object")
    if not isinstance(field, str) or not field:
        raise AuditInputError("source-hash field must be a nonempty string")
    raw_ledger = payload.get(field)
    if not isinstance(raw_ledger, Mapping) or not raw_ledger:
        raise AuditInputError(f"evidence field {field!r} must be a nonempty mapping")

    ledger: dict[str, str] = {}
    for raw_path, raw_digest in raw_ledger.items():
        path = _repository_relative_path(raw_path).as_posix()
        if not isinstance(raw_digest, str) or _SHA256.fullmatch(raw_digest) is None:
            raise AuditInputError(
                f"source digest for {path!r} must be a lowercase SHA256"
            )
        ledger[path] = raw_digest
    return ledger


def _artifact_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _read_evidence_once(path: Path) -> tuple[object, str, _FileIdentity]:
    try:
        with path.open("rb") as handle:
            before = _FileIdentity.from_stat(os.fstat(handle.fileno()))
            data = handle.read()
            after = _FileIdentity.from_stat(os.fstat(handle.fileno()))
    except OSError as exc:
        raise AuditInputError(f"cannot read evidence JSON: {exc}") from exc
    if before != after or len(data) != before.size:
        raise AuditInputError("evidence artifact changed while being read")
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"cannot parse evidence JSON: {exc}") from exc
    return payload, hashlib.sha256(data).hexdigest(), before


def _non_symlink_component_identities(
    repository_root: Path, path: Path
) -> tuple[_FileIdentity, ...]:
    try:
        relative = path.relative_to(repository_root)
    except ValueError as exc:
        raise AuditInputError(f"source path is outside repository: {path}") from exc
    identities: list[_FileIdentity] = []
    current = repository_root
    for component in relative.parts:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AuditInputError(
                f"cannot inspect source path component {current}: {exc}"
            ) from exc
        if stat_module.S_ISLNK(component_stat.st_mode):
            raise AuditInputError(
                f"source ledger paths must not contain symbolic links: {current}"
            )
        identities.append(_FileIdentity.from_stat(component_stat))
    return tuple(identities)


def _stable_source_hash(
    repository_root: Path, path: Path
) -> _SourceCapture:
    try:
        components_before = _non_symlink_component_identities(repository_root, path)
        before = _FileIdentity.from_stat(path.stat())
        digest = sha256_file(path)
        after = _FileIdentity.from_stat(path.stat())
        components_after = _non_symlink_component_identities(repository_root, path)
    except (OSError, ValueError) as exc:
        raise AuditInputError(f"cannot hash source file {path}: {exc}") from exc
    if before != after or components_before != components_after:
        raise AuditInputError(f"source file changed while being hashed: {path}")
    return _SourceCapture(path, before, digest)


def _prepare_audit(
    repository_root: str | Path,
    evidence_path: str | Path,
    *,
    source_hash_field: str,
) -> tuple[dict[str, object], _AuditStability]:
    """Capture an audit report and the identities needed for final validation."""


    try:
        snapshot = RepositorySnapshot.capture(repository_root, require_clean=False)
    except (OSError, RuntimeError) as exc:
        raise AuditInputError(f"invalid Git repository: {exc}") from exc

    try:
        evidence = Path(evidence_path).expanduser().resolve(strict=True)
        if not evidence.is_file():
            raise AuditInputError("evidence path must name a regular file")
    except AuditInputError:
        raise
    except OSError as exc:
        raise AuditInputError(f"cannot resolve evidence JSON: {exc}") from exc

    payload, evidence_sha256, evidence_identity = _read_evidence_once(evidence)

    ledger = _source_ledger(payload, source_hash_field)
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise AuditInputError("evidence artifact_id must be a nonempty string")

    matching: list[str] = []
    drifted: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    source_captures: list[_SourceCapture] = []
    missing_sources: list[Path] = []
    for relative, expected in sorted(ledger.items()):
        candidate = snapshot.root / relative
        _non_symlink_component_identities(snapshot.root, candidate)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            missing.append({"path": relative, "expected_sha256": expected})
            missing_sources.append(candidate)
            continue
        except OSError as exc:
            raise AuditInputError(
                f"cannot resolve source path {relative!r}: {exc}"
            ) from exc
        try:
            resolved.relative_to(snapshot.root)
        except ValueError as exc:
            raise AuditInputError(
                f"source path resolves outside repository: {relative!r}"
            ) from exc
        if not resolved.is_file():
            missing.append({"path": relative, "expected_sha256": expected})
            missing_sources.append(candidate)
            continue
        capture = _stable_source_hash(snapshot.root, candidate)
        source_captures.append(capture)
        actual = capture.sha256
        if actual == expected:
            matching.append(relative)
        else:
            drifted.append(
                {
                    "path": relative,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_artifact": {
            "artifact_id": artifact_id,
            "path": _artifact_path(evidence, snapshot.root),
            "sha256": evidence_sha256,
            "source_hash_field": source_hash_field,
        },
        "repository": {
            "head": snapshot.head,
            "tree": snapshot.tree,
            "status_porcelain": snapshot.status_porcelain,
            "dirty": not snapshot.clean,
        },
        "counts": {
            "total": len(ledger),
            "matching": len(matching),
            "drifted": len(drifted),
            "missing": len(missing),
        },
        "all_match": not drifted and not missing,
        "matching": matching,
        "drifted": drifted,
        "missing": missing,
    }
    result["report_sha256"] = canonical_json_sha256(result)
    stability = _AuditStability(
        snapshot,
        evidence,
        evidence_identity,
        tuple(source_captures),
        tuple(missing_sources),
    )
    return result, stability


def audit_source_hash_lineage(
    repository_root: str | Path,
    evidence_path: str | Path,
    *,
    source_hash_field: str = DEFAULT_SOURCE_HASH_FIELD,
) -> dict[str, object]:
    """Compare an evidence source ledger with current repository file bytes."""

    result, stability = _prepare_audit(
        repository_root,
        evidence_path,
        source_hash_field=source_hash_field,
    )
    _after_initial_capture()
    stability.assert_stable()
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--source-hash-field",
        default=DEFAULT_SOURCE_HASH_FIELD,
        help=f"evidence mapping field (default: {DEFAULT_SOURCE_HASH_FIELD})",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _repository_file_identities(root: Path) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    walk_error: OSError | None = None

    def record_error(exc: OSError) -> None:
        nonlocal walk_error
        walk_error = exc

    for directory, _, names in os.walk(
        root, followlinks=False, onerror=record_error
    ):
        for name in names:
            try:
                stat = os.stat(Path(directory) / name, follow_symlinks=False)
            except OSError as exc:
                raise AuditInputError(
                    f"cannot inspect repository file for output aliases: {exc}"
                ) from exc
            identities.add((stat.st_dev, stat.st_ino))
    if walk_error is not None:
        raise AuditInputError(
            f"cannot inspect repository for output aliases: {walk_error}"
        )
    return identities


def _external_output(
    path: Path,
    repository_root: Path,
    evidence_path: Path,
) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(path)))
    if lexical.is_symlink():
        raise AuditInputError("audit output must not be a symbolic link")
    resolved = lexical.parent.resolve(strict=True) / lexical.name
    root = repository_root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise AuditInputError("audit output must be outside the repository")
    if not resolved.parent.is_dir():
        raise AuditInputError("audit output parent directory does not exist")
    if resolved == evidence_path:
        raise AuditInputError("audit output must not replace the evidence artifact")
    if resolved.exists():
        try:
            if resolved.samefile(evidence_path):
                raise AuditInputError(
                    "audit output must not alias the evidence artifact"
                )
            output_stat = resolved.stat()
        except OSError as exc:
            raise AuditInputError(f"cannot inspect audit output: {exc}") from exc
        if (output_stat.st_dev, output_stat.st_ino) in _repository_file_identities(
            root
        ):
            raise AuditInputError("audit output must not alias a repository file")
        if not resolved.is_file():
            raise AuditInputError("existing audit output must be a regular file")
    return resolved


def _atomic_write_output(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        for _ in range(100):
            candidate = path.with_name(
                f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
            )
            try:
                descriptor = os.open(candidate, flags, 0o600)
            except FileExistsError:
                continue
            temporary = candidate
            break
        if descriptor is None or temporary is None:
            raise AuditInputError("cannot allocate fresh audit output temporary")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result, stability = _prepare_audit(
            args.repo_root,
            args.evidence,
            source_hash_field=args.source_hash_field,
        )
        _after_initial_capture()
        stability.assert_stable()
        if args.output is not None:
            output = _external_output(
                args.output,
                stability.snapshot.root,
                stability.evidence_path,
            )
            content = (
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            _atomic_write_output(output, content)
            _after_output_write()
            stability.assert_stable()
    except (AuditInputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
