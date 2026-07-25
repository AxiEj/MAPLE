from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .protocol import canonical_sha256


class RestartHashMismatch(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path, *, run_hash: str) -> None:
        self.root = Path(root)
        self.run_hash = run_hash
        self.stages_dir = self.root / "stages"

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
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = self._read_json(manifest_path)
            if (
                existing.get("run_hash") != self.run_hash
                or canonical_sha256(existing) != canonical_sha256(manifest)
            ):
                raise RestartHashMismatch(
                    "RESTART_HASH_MISMATCH: existing Route A manifest does not "
                    "match the requested run."
                )
        else:
            self._atomic_json(manifest_path, manifest)
        ledger_path = self.root / "stage-ledger.json"
        if not ledger_path.exists():
            self._atomic_json(
                ledger_path,
                {
                    "schema_version": 1,
                    "run_hash": self.run_hash,
                    "stages": {},
                },
            )

    def write_json(self, relative_path: str | Path, value: object) -> None:
        self._atomic_json(self.root / relative_path, value)

    def complete_stage(
        self,
        stage: str,
        *,
        input_hash: str,
        output: dict[str, Any],
    ) -> str:
        normalized_stage = stage.upper()
        output_hash = canonical_sha256(output)
        output_path = self.stages_dir / f"{normalized_stage.lower()}.json"
        self._atomic_json(output_path, output)
        ledger_path = self.root / "stage-ledger.json"
        ledger = self._read_json(ledger_path)
        ledger["stages"][normalized_stage] = {
            "status": "completed",
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output_path": output_path.relative_to(self.root).as_posix(),
        }
        self._atomic_json(ledger_path, ledger)
        return output_hash

    def resume_stage(
        self,
        stage: str,
        *,
        input_hash: str,
    ) -> dict[str, Any] | None:
        ledger_path = self.root / "stage-ledger.json"
        if not ledger_path.exists():
            return None
        normalized_stage = stage.upper()
        record = self._read_json(ledger_path).get("stages", {}).get(
            normalized_stage
        )
        if record is None:
            return None
        if record.get("input_hash") != input_hash:
            raise RestartHashMismatch(
                f"RESTART_HASH_MISMATCH: stage {normalized_stage} input changed."
            )
        output_path = self.root / record["output_path"]
        output = self._read_json(output_path)
        if canonical_sha256(output) != record.get("output_hash"):
            raise RestartHashMismatch(
                f"RESTART_HASH_MISMATCH: stage {normalized_stage} output is corrupt."
            )
        return output
