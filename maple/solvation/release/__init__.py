"""Release-evidence helpers for the conservative Route-2 rebuild.

The release package never admits a scientific capability.  It only captures
and validates source/runtime identity plus raw diagnostic measurements used by
the authoritative registries.
"""

from .evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

__all__ = [
    "RepositorySnapshot",
    "canonical_json_sha256",
    "checkpoint_record",
    "collect_loaded_repository_sources",
    "committed_source_hashes",
    "runtime_record",
    "sha256_file",
    "write_external_json_artifact",
]
