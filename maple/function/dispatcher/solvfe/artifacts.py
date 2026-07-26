from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .protocol import canonical_sha256


class RestartHashMismatch(RuntimeError):
    pass


class ConcurrentArtifactWrite(RuntimeError):
    pass


class ArtifactStore:
    _HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
    _ATTEMPT_PATTERN = re.compile(r"^[0-9a-f]{32}$")
    _STAGE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]*$")

    def __init__(self, root: str | Path, *, run_hash: str) -> None:
        self.root = Path(root)
        self.run_hash = run_hash
        self.stages_dir = self.root / "stages"
        self.attempt_id = uuid.uuid4().hex
        self._thread_lock = threading.RLock()
        self._lock_handle = None
        self._lock_depth = 0

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self._thread_lock:
            if self._lock_handle is not None:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return

            self.root.mkdir(parents=True, exist_ok=True)
            lock_path = self.root / ".artifact-store.lock"
            handle = lock_path.open("a+", encoding="utf-8")
            try:
                try:
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    raise ConcurrentArtifactWrite(
                        "ARTIFACT_WRITER_ACTIVE: another process or store "
                        f"instance is writing Route A run {self.run_hash}."
                    ) from exc
                self._lock_handle = handle
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    self._lock_handle = None
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @contextmanager
    def exclusive_session(self) -> Iterator["ArtifactStore"]:
        """Hold the run-directory writer lock for one complete workflow."""

        with self._exclusive_lock():
            yield self

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def initialize(self, manifest: dict[str, Any]) -> None:
        with self._exclusive_lock():
            manifest_path = self.root / "manifest.json"
            if manifest_path.exists():
                existing = self._read_json(manifest_path)
                if (
                    existing.get("run_hash") != self.run_hash
                    or canonical_sha256(existing) != canonical_sha256(manifest)
                ):
                    raise RestartHashMismatch(
                        "RESTART_HASH_MISMATCH: existing Route A manifest does "
                        "not match the requested run."
                    )
            else:
                self._atomic_json(manifest_path, manifest)
            ledger_path = self.root / "stage-ledger.json"
            if not ledger_path.exists():
                self._atomic_json(
                    ledger_path,
                    {
                        "schema_version": 2,
                        "run_hash": self.run_hash,
                        "generation": 0,
                        "stages": {},
                    },
                )
            else:
                self._validated_ledger(ledger_path)

    def write_json(self, relative_path: str | Path, value: object) -> None:
        with self._exclusive_lock():
            self._atomic_json(self._safe_relative_path(relative_path), value)

    def _safe_relative_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("Artifact paths must be relative to the run root.")
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Artifact paths must remain inside the run root.")
        return candidate

    @classmethod
    def _normalized_stage(cls, stage: str) -> str:
        normalized = str(stage).upper()
        if not cls._STAGE_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Artifact stage names must match [A-Z][A-Z0-9_-]*."
            )
        return normalized

    def _validated_ledger(self, path: Path) -> dict[str, Any]:
        ledger = self._read_json(path)
        generation = ledger.get("generation")
        if (
            ledger.get("schema_version") != 2
            or ledger.get("run_hash") != self.run_hash
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or not isinstance(ledger.get("stages"), dict)
        ):
            raise RestartHashMismatch(
                "RESTART_HASH_MISMATCH: Route A stage ledger identity or "
                "generation is invalid."
            )
        record_generations: list[int] = []
        for stage, record in ledger["stages"].items():
            if (
                not isinstance(stage, str)
                or not self._STAGE_PATTERN.fullmatch(stage)
                or not isinstance(record, dict)
                or set(record)
                != {
                    "status",
                    "input_hash",
                    "output_hash",
                    "output_path",
                    "generation",
                    "attempt_id",
                }
            ):
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: Route A stage record structure "
                    "is invalid."
                )
            record_generation = record.get("generation")
            expected_output_path = f"stages/{stage.lower()}.json"
            if (
                record.get("status") != "completed"
                or not isinstance(record.get("input_hash"), str)
                or not self._HASH_PATTERN.fullmatch(record["input_hash"])
                or not isinstance(record.get("output_hash"), str)
                or not self._HASH_PATTERN.fullmatch(record["output_hash"])
                or record.get("output_path") != expected_output_path
                or isinstance(record_generation, bool)
                or not isinstance(record_generation, int)
                or not 1 <= record_generation <= generation
                or not isinstance(record.get("attempt_id"), str)
                or not self._ATTEMPT_PATTERN.fullmatch(record["attempt_id"])
            ):
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: Route A stage audit record is "
                    "invalid."
                )
            record_generations.append(record_generation)
        if (
            len(record_generations) != len(set(record_generations))
            or generation != (max(record_generations, default=0))
        ):
            raise RestartHashMismatch(
                "RESTART_HASH_MISMATCH: Route A stage generations are "
                "inconsistent."
            )
        return ledger

    def complete_stage(
        self,
        stage: str,
        *,
        input_hash: str,
        output: dict[str, Any],
    ) -> str:
        with self._exclusive_lock():
            normalized_stage = self._normalized_stage(stage)
            if not self._HASH_PATTERN.fullmatch(str(input_hash)):
                raise ValueError(
                    "Artifact stage input_hash must be lowercase SHA-256."
                )
            output_hash = canonical_sha256(output)
            output_path = self._safe_relative_path(
                Path("stages") / f"{normalized_stage.lower()}.json"
            )
            self._atomic_json(output_path, output)
            ledger_path = self.root / "stage-ledger.json"
            ledger = self._validated_ledger(ledger_path)
            next_generation = int(ledger["generation"]) + 1
            ledger["generation"] = next_generation
            ledger["stages"][normalized_stage] = {
                "status": "completed",
                "input_hash": input_hash,
                "output_hash": output_hash,
                "output_path": output_path.relative_to(self.root).as_posix(),
                "generation": next_generation,
                "attempt_id": self.attempt_id,
            }
            self._atomic_json(ledger_path, ledger)
            return output_hash

    def resume_stage(
        self,
        stage: str,
        *,
        input_hash: str,
    ) -> dict[str, Any] | None:
        with self._exclusive_lock():
            ledger_path = self.root / "stage-ledger.json"
            if not ledger_path.exists():
                return None
            normalized_stage = self._normalized_stage(stage)
            if not self._HASH_PATTERN.fullmatch(str(input_hash)):
                raise ValueError(
                    "Artifact stage input_hash must be lowercase SHA-256."
                )
            record = self._validated_ledger(ledger_path)["stages"].get(
                normalized_stage
            )
            if record is None:
                return None
            if record.get("input_hash") != input_hash:
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: stage "
                    f"{normalized_stage} input changed."
                )
            output_path = self._safe_relative_path(record["output_path"])
            output = self._read_json(output_path)
            if canonical_sha256(output) != record.get("output_hash"):
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: stage "
                    f"{normalized_stage} output is corrupt."
                )
            return output
