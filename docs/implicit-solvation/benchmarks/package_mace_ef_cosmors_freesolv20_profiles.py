#!/usr/bin/env python3
"""Package exact v3 SCF/frozen profiles into a deterministic replay bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from run_mace_ef_cosmors_freesolv20 import (  # noqa: E402
    _bind_source_files,
    _git_blob_sha256,
    _repository_relative,
    _resolve_artifact_path,
    _sha256_file,
    _write_json_atomic,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v3.json"
FROZEN_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-frozen-source-ablation-v2.json"
CROSS_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-response-cross-v1.json"
ARCHIVE_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
MANIFEST_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.json"


def _assert_source_compatible(artifact: dict[str, object], commit: str) -> None:
    hashes = artifact.get("source_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Input artifact lacks committed source-file binding.")
    for relative, expected in hashes.items():
        if _git_blob_sha256(commit, str(relative)) != expected:
            raise RuntimeError(
                f"Input artifact source {relative!r} is incompatible with {commit}."
            )


def _checked_path(spec: dict[str, object]) -> Path:
    path = _resolve_artifact_path(spec["path"])
    if _sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"Replay asset bytes drifted: {path}")
    return path


def _write_deterministic_tar_gz(entries: dict[str, Path], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for name in sorted(entries):
                    data = entries[name].read_bytes()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    temporary.replace(output)


def main() -> int:
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    cross = json.loads(CROSS_PATH.read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (primary, frozen, cross)):
        raise ValueError(
            "Replay bundle requires complete primary, frozen, and cross artifacts."
        )
    execution_git_head, source_files_sha256 = _bind_source_files((Path(__file__),))
    for artifact in (primary, frozen, cross):
        _assert_source_compatible(artifact, execution_git_head)
    if frozen["primary_artifact"]["sha256"] != _sha256_file(PRIMARY_PATH):
        raise RuntimeError("Frozen control is not bound to the packaged primary.")
    if cross["primary_artifact"]["sha256"] != _sha256_file(PRIMARY_PATH):
        raise RuntimeError("Response-cross is not bound to the packaged primary.")
    if cross["frozen_control_artifact"]["sha256"] != _sha256_file(FROZEN_PATH):
        raise RuntimeError(
            "Response-cross is not bound to the packaged frozen control."
        )

    entries: dict[str, Path] = {
        "scf/water.torch-cosmors.json": _checked_path(
            {
                "path": primary["water_profile"]["path"],
                "sha256": primary["water_profile"]["sha256"],
            }
        ),
        "geometries/water.xyz": _checked_path(
            {
                "path": primary["water_profile"]["xyz_path"],
                "sha256": primary["water_profile"]["xyz_sha256"],
            }
        ),
        "frozen/water.torch-cosmors.json": _checked_path(
            {
                "path": frozen["water_frozen_source"]["profile_path"],
                "sha256": frozen["water_frozen_source"]["profile_sha256"],
            }
        ),
    }
    frozen_by_id = {record["compound_id"]: record for record in frozen["records"]}
    for primary_record in primary["records"]:
        compound_id = primary_record["compound_id"]
        frozen_record = frozen_by_id[compound_id]
        entries[f"geometries/{compound_id}.xyz"] = _checked_path(
            primary_record["generated_xyz"]
        )
        entries[f"scf/{compound_id}.torch-cosmors.json"] = _checked_path(
            primary_record["mace_ef_profile"]
        )
        entries[f"frozen/{compound_id}.torch-cosmors.json"] = _checked_path(
            frozen_record["frozen_profile"]
        )

    _write_deterministic_tar_gz(entries, ARCHIVE_PATH)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-freesolv20-profile-bundle-v1",
        "status": "complete",
        "scientific_result": False,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "execution_git_head": execution_git_head,
        "source_files_sha256": source_files_sha256,
        "input_artifacts": {
            "primary": {
                "path": _repository_relative(PRIMARY_PATH),
                "sha256": _sha256_file(PRIMARY_PATH),
            },
            "frozen_control": {
                "path": _repository_relative(FROZEN_PATH),
                "sha256": _sha256_file(FROZEN_PATH),
            },
            "response_cross": {
                "path": _repository_relative(CROSS_PATH),
                "sha256": _sha256_file(CROSS_PATH),
            },
        },
        "archive": {
            "path": _repository_relative(ARCHIVE_PATH),
            "sha256": _sha256_file(ARCHIVE_PATH),
            "format": "deterministic-ustar-gzip-mtime-zero",
            "member_count": len(entries),
        },
        "member_sha256": {
            name: _sha256_file(path) for name, path in sorted(entries.items())
        },
        "claim_boundary": "Exact geometry and sigma-profile replay assets only; the bundle adds no scientific admission or new numerical result.",
    }
    _write_json_atomic(MANIFEST_PATH, manifest)
    print(json.dumps(manifest["archive"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
