from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


class ProtocolIntegrityError(ValueError):
    """Raised when a Route A protocol or one of its bound artifacts drifts."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolIntegrityError(
            f"Cannot load Route A protocol artifact '{path}': {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProtocolIntegrityError(
            f"Route A protocol artifact '{path}' must contain a JSON object."
        )
    return value


def _find_project_root(protocol_path: Path) -> Path:
    for parent in (protocol_path.parent, *protocol_path.parents):
        if (parent / "maple").is_dir() and (parent / "docs").is_dir():
            return parent
    raise ProtocolIntegrityError(
        "Cannot locate the MAPLE project root for Route A protocol validation."
    )


@dataclass(frozen=True)
class RouteAProtocol:
    path: Path
    project_root: Path
    data: Mapping[str, Any]
    content_hash: str
    verified_artifact_count: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_artifacts: bool = True,
    ) -> "RouteAProtocol":
        protocol_path = Path(path).expanduser().resolve()
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else _find_project_root(protocol_path)
        )
        data = _load_json(protocol_path)

        declared_protocol_hash = data.get("protocol_sha256")
        payload = copy.deepcopy(data)
        payload.pop("protocol_sha256", None)
        actual_protocol_hash = canonical_sha256(payload)
        if declared_protocol_hash != actual_protocol_hash:
            raise ProtocolIntegrityError(
                "Route A protocol self hash mismatch: "
                f"declared={declared_protocol_hash}, actual={actual_protocol_hash}."
            )

        schema_ref = data.get("protocol_schema")
        if not isinstance(schema_ref, dict):
            raise ProtocolIntegrityError(
                "Route A protocol is missing protocol_schema metadata."
            )
        schema_path = root / str(schema_ref.get("path", ""))
        schema = _load_json(schema_path)
        actual_schema_hash = canonical_sha256(schema)
        if schema_ref.get("sha256") != actual_schema_hash:
            raise ProtocolIntegrityError(
                "Route A protocol schema hash mismatch."
            )
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(data)
        except Exception as exc:
            raise ProtocolIntegrityError(
                f"Route A protocol schema validation failed: {exc}"
            ) from exc

        tail_contract = copy.deepcopy(data.get("tail_envelope_contract", {}))
        declared_tail_hash = tail_contract.pop("contract_sha256", None)
        if declared_tail_hash != canonical_sha256(tail_contract):
            raise ProtocolIntegrityError(
                "Route A tail-envelope contract hash mismatch."
            )

        verified = 0
        if verify_artifacts:
            artifact_refs = data.get("artifact_references")
            if not isinstance(artifact_refs, dict):
                raise ProtocolIntegrityError(
                    "Route A protocol artifact_references must be an object."
                )
            for name, entry in artifact_refs.items():
                if not isinstance(entry, dict):
                    raise ProtocolIntegrityError(
                        f"Route A artifact reference '{name}' must be an object."
                    )
                artifact_path = root / str(entry.get("path", ""))
                kind = entry.get("hash_kind")
                if kind == "canonical_sha256":
                    actual = canonical_sha256(_load_json(artifact_path))
                elif kind == "sha256":
                    try:
                        actual = raw_sha256(artifact_path)
                    except OSError as exc:
                        raise ProtocolIntegrityError(
                            f"Cannot read Route A artifact '{artifact_path}': {exc}"
                        ) from exc
                else:
                    raise ProtocolIntegrityError(
                        f"Unsupported hash_kind '{kind}' for Route A artifact '{name}'."
                    )
                if entry.get("sha256") != actual:
                    raise ProtocolIntegrityError(
                        f"Route A artifact hash mismatch for '{name}'."
                    )
                verified += 1

        return cls(
            path=protocol_path,
            project_root=root,
            data=copy.deepcopy(data),
            content_hash=actual_protocol_hash,
            verified_artifact_count=verified,
        )
