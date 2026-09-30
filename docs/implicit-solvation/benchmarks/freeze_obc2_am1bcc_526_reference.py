#!/usr/bin/env python3
"""Freeze the label-denied AM1-BCC/OBC-II OpenMM Reference oracle.

This Stage-0 generator is intentionally separate from the historical FreeSolv
runner.  It reads only the tracked label-free member/charge manifest, the pinned
method protocol, manifest-bound MOL2 files, and raw bytes needed for SHA-256
identity checks.  Historical result JSON, prepared data, summaries, database
contents, and score artifacts are never parsed.

The generated 526-case artifact is an energy/component oracle for ``ace`` and
``none``.  The companion 12-case fixture additionally freezes topology/radius
parameters and the established two-geometry E/F matrix.  Neither artifact reads
or reports experimental values and neither is an accuracy benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_core as core  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

BASE_HEAD = "14664ed85159da6b0c67e5deea972e2ec3e75c5e"
RECORDED_DATE = "2026-09-27"
GENERATOR_REL = (
    "docs/implicit-solvation/benchmarks/" "freeze_obc2_am1bcc_526_reference.py"
)
OUTPUT_REL = (
    "docs/implicit-solvation/benchmarks/"
    "route1-obc2-am1bcc-526-reference-2026-09-27.json"
)
PORTABLE_REL = "tests/solvation/data/obc2_openmm_8_5_2_reference_v1.json"
STAGE0_TEST_REL = "tests/solvation/test_obc2_stage0_oracle_freezer.py"
PROTOCOL_REL = "docs/implicit-solvation/benchmarks/protocol.json"
MANIFEST_REL = "docs/implicit-solvation/benchmarks/chagb_nonpolar_source_manifest.json"
CORPUS_ROOT_REL = ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
VENDORED_OPENMM_SOURCE_REL = (
    ".omx/vendor-audits/gbmv2/openmm-src/wrappers/python/"
    "openmm/app/internal/customgbforces.py"
)

EXPECTED_PROTOCOL_SHA256 = (
    "0aa134a94189583deccd027b428895a21a4244489711670fa4973b0d38513a1d"
)
EXPECTED_PROTOCOL_FINGERPRINT = (
    "76ed1ea6483678d113353d7371ca1e015c780a38937fa5333d7c40ec9db8c676"
)
EXPECTED_DATASET_COMMIT = "6c7d19b4b565537365ffd22006aa2cd4643200c6"
EXPECTED_MANIFEST_SHA256 = (
    "84cd6569296ab4f0ae8410f3f03728fd04c358980936b811be57e8ff2c2508dc"
)
EXPECTED_SAFE_MEMBER_BINDING_SHA256 = (
    "ae7e13f6e36855a0fa6b7f2d3bae4733ad58bf02afa5d89cdb57cf4529dddbef"
)
EXPECTED_OPENMM_VERSION = "8.5.2"
EXPECTED_CUSTOMGBFORCES_SHA256 = (
    "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
)
FORMULA_SOURCE_PROFILE = "openmm-8.5.2-obc2-mbondi2-v1"

DATASET_BYTE_PINS = {
    "dataset/database.txt": (
        "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260"
    ),
    "dataset/database.json": (
        "9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a"
    ),
    "dataset/mol2files_gaff.tar.gz": (
        "15ece6114442a8eb5e720c0ad25bb48e4a6a35392da36f72b80a6f2f415fd133"
    ),
}

HISTORICAL_BYTE_PINS = {
    "docs/implicit-solvation/benchmarks/"
    "route1-torch-obc2-analytic-derivatives-2026-09-22.json": (
        "961ba66a97a7b9a0c6d80640409bfb1218de496d42757c06e8dfd2e1e81947f5"
    ),
    "docs/implicit-solvation/benchmarks/"
    "route1-torch-obc2-derivative-performance-2026-09-22.json": (
        "49f009bfe8921a9609b6487e363ac61a77dcc03854a19c4c02beeeb07ae36e6f"
    ),
    "docs/implicit-solvation/benchmarks/"
    "route1-torch-post-audit-revalidation-2026-09-27.json": (
        "5cfe3a3aaa5303a2d371fa3945eb5cbb8e7ab1bcc2a2d903d42cc5fcc2047989"
    ),
    "docs/implicit-solvation/benchmarks/freesolv-development-2026-07-23.json": (
        "911e53feee177ffe1e92112b0a723793748850dfe470b6e4d08f8611ef3f6855"
    ),
}

PRODUCTION_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/common.py",
    "maple/function/calculator/extra_correction/implicit/nonpolar.py",
    "maple/function/calculator/extra_correction/implicit/obc2_parameters.py",
    "maple/function/calculator/extra_correction/implicit/openmm_compat.py",
    "maple/function/calculator/extra_correction/implicit/openmm_execution.py",
    "maple/function/calculator/extra_correction/implicit/openmm_gb.py",
    "maple/function/calculator/extra_correction/implicit/radii.py",
    "maple/function/calculator/extra_correction/implicit/result.py",
    "maple/function/calculator/extra_correction/implicit/torch_obc2.py",
    "maple/function/read/filereader/mol2_reader.py",
)

FROZEN_CASES = (
    "benzene",
    "methanol",
    "bromobenzene",
    "methyl-hexanoate",
    "iodobenzene",
    "glucose",
    "4-nitroaniline",
    "dimethyl-sulfide",
    "fluorobenzene",
    "nitralin",
    "aniline",
    "chlorobenzene",
)
PROFILE_ORDER = ("ace", "none")
GEOMETRY_ORDER = ("base", "displaced")

MANIFEST_TOP_KEYS = frozenset(
    {
        "baseline_model",
        "case_count",
        "charge_method",
        "records",
        "schema_version",
        "source_partition",
        "topology_atom_types",
    }
)
MANIFEST_ROW_KEYS = frozenset(
    {
        "am1bcc_charges_e",
        "compound_id",
        "source_mol2_relative_path",
        "source_mol2_sha256",
        "source_record_relative_path",
        "source_record_sha256",
    }
)

# These tokens are rejected as output *field names*.  OpenMM's computational
# platform name "Reference" is intentionally allowed; reference-label fields are not.
FORBIDDEN_OUTPUT_KEY_TOKENS = (
    "experimental",
    "score",
    "signed_error",
    "absolute_error",
    "mae",
    "rmse",
    "residual",
    "fit",
    "target_value",
)

SEMANTIC_JSON_ALLOWLIST = frozenset({PROTOCOL_REL, MANIFEST_REL})
OWNED_STAGE0_PATHS = frozenset(
    {GENERATOR_REL, OUTPUT_REL, PORTABLE_REL, STAGE0_TEST_REL}
)

FORBIDDEN_SEMANTIC_KEY_MARKERS = (
    b'"experimental_',
    b'"signed_error',
    b'"absolute_error',
    b'"mean_absolute_error',
    b'"root_mean_square_error',
    b'"score',
)

# Existing comparison limits are recorded, never selected by this generator.
FROZEN_COMPARISON_CONTRACT = {
    "energy_abs_hartree": 2.0e-10,
    "energy_relative": 1.0e-9,
    "component_abs_hartree": 2.0e-10,
    "force_max_hartree_per_angstrom": 2.0e-7,
    "force_rms_hartree_per_angstrom": 5.0e-8,
}


@dataclass(frozen=True)
class MemberBinding:
    compound_id: str
    mol2_relative_path: str
    mol2_sha256: str
    record_relative_path: str
    record_sha256: str
    charges_e: tuple[float, ...]
    charge_vector_sha256: str
    atom_count: int


class SemanticDecoderGuard:
    """Block semantic JSON decoding of known sensitive Stage-0 inputs.

    Raw byte hashing is deliberately outside this guard.  The guard is scoped to
    ``json.load``/``json.loads`` during the Stage-0 process, detects both named
    paths and byte payloads (including BytesIO/read-bytes bypasses), and leaves
    unrelated OpenMM/package metadata alone.
    """

    def __init__(self, bindings: Iterable[MemberBinding]):
        corpus_root = (ROOT / CORPUS_ROOT_REL).resolve()
        self._forbidden_digests = {binding.record_sha256 for binding in bindings}
        self._forbidden_paths = {
            (corpus_root / binding.record_relative_path).resolve()
            for binding in bindings
        }
        self._forbidden_paths.update(
            {
                (corpus_root / "prepared.json").resolve(),
                (corpus_root / "development-summary-repeat.json").resolve(),
                (corpus_root / "dataset/database.json").resolve(),
                (
                    ROOT
                    / "docs/implicit-solvation/benchmarks/freesolv-development-2026-07-23.json"
                ).resolve(),
                (
                    ROOT
                    / "docs/implicit-solvation/benchmarks/route1-freesolv-reserve-score-2026-07-25.json"
                ).resolve(),
                (
                    ROOT
                    / "docs/implicit-solvation/benchmarks/route1-freesolv-reserve-score-sealed-v1-2026-07-25.json"
                ).resolve(),
            }
        )
        self.blocked: list[dict[str, str]] = []
        self._original_load = json.load
        self._original_loads = json.loads

    @staticmethod
    def _payload_bytes(payload: str | bytes | bytearray) -> bytes:
        if isinstance(payload, str):
            return payload.encode("utf-8")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        raise TypeError(
            f"json.loads input has unsupported type {type(payload).__name__}"
        )

    def _deny(self, *, api: str, digest: str, role: str) -> None:
        self.blocked.append({"api": api, "sha256": digest, "role": role})
        raise PermissionError(
            f"Stage-0 semantic decoder denied {role} via {api}; raw SHA256 is allowed"
        )

    def guarded_loads(self, payload, *args, **kwargs):
        raw = self._payload_bytes(payload)
        digest = core.sha256_bytes(raw)
        if digest in self._forbidden_digests:
            self._deny(api="json.loads", digest=digest, role="historical-member-bytes")
        lowered = raw.lower()
        if any(marker in lowered for marker in FORBIDDEN_SEMANTIC_KEY_MARKERS):
            self._deny(api="json.loads", digest=digest, role="sensitive-field-payload")
        return self._original_loads(payload, *args, **kwargs)

    def guarded_load(self, handle, *args, **kwargs):
        name = getattr(handle, "name", None)
        if name is not None:
            try:
                path = Path(name).resolve()
            except (OSError, TypeError, ValueError):
                path = None
            if path is not None and path in self._forbidden_paths:
                digest = core.sha256_file(path)
                self._deny(
                    api="json.load", digest=digest, role="forbidden-semantic-path"
                )
        payload = handle.read()
        return self.guarded_loads(payload, *args, **kwargs)

    @contextmanager
    def active(self):
        previous_load = json.load
        previous_loads = json.loads
        json.load = self.guarded_load  # type: ignore[assignment]
        json.loads = self.guarded_loads  # type: ignore[assignment]
        try:
            yield self
        finally:
            json.load = previous_load  # type: ignore[assignment]
            json.loads = previous_loads  # type: ignore[assignment]

    def reset(self) -> None:
        self.blocked.clear()


def _canonical_no_newline(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _require_file_hash(path: Path, expected: str, *, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    observed = core.sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA256 mismatch: expected {expected}, observed {observed}: {path}"
        )
    return observed


def _semantic_load_json(relative_path: str) -> Any:
    if relative_path not in SEMANTIC_JSON_ALLOWLIST:
        raise PermissionError(
            f"Stage-0 semantic JSON read is not allowlisted: {relative_path}"
        )
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _assert_output_key_policy(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_OUTPUT_KEY_TOKENS):
                raise ValueError(f"Forbidden Stage-0 output field {path}.{key}")
            _assert_output_key_policy(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_output_key_policy(item, f"{path}[{index}]")


def _status_path(line: str) -> str:
    raw = line[3:]
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip().strip('"')


def _assert_stage0_worktree() -> dict[str, str]:
    head = _git("rev-parse", "HEAD")
    if head != BASE_HEAD:
        raise ValueError(f"Stage 0 requires HEAD {BASE_HEAD}; observed {head}")
    if subprocess.run(
        ["git", "diff", "--quiet", BASE_HEAD, "--", "maple"], cwd=ROOT
    ).returncode:
        raise ValueError(
            "Tracked production files under maple/ differ from baseline HEAD"
        )
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", BASE_HEAD, "--", "maple"],
        cwd=ROOT,
    ).returncode:
        raise ValueError(
            "Staged production files under maple/ differ from baseline HEAD"
        )

    concurrent_tests: dict[str, str] = {}
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    for line in status.splitlines():
        path = _status_path(line)
        if path in OWNED_STAGE0_PATHS:
            continue
        if path.startswith("tests/"):
            candidate = ROOT / path
            if candidate.is_file():
                concurrent_tests[path] = core.sha256_file(candidate)
            continue
        raise ValueError(f"Unexpected non-Stage-0 worktree change: {line}")
    return concurrent_tests


def _snapshot_production_sources() -> dict[str, Any]:
    tracked = [
        line
        for line in _git("ls-files", "maple").splitlines()
        if line and (ROOT / line).is_file()
    ]
    hashes = {path: core.sha256_file(ROOT / path) for path in sorted(tracked)}
    return {
        "file_count": len(hashes),
        "aggregate_sha256": core.sha256_bytes(core.canonical_json_bytes(hashes)),
        "files_sha256": hashes,
    }


def _snapshot_fixed_bytes() -> dict[str, str]:
    expected = {
        PROTOCOL_REL: EXPECTED_PROTOCOL_SHA256,
        MANIFEST_REL: EXPECTED_MANIFEST_SHA256,
        VENDORED_OPENMM_SOURCE_REL: EXPECTED_CUSTOMGBFORCES_SHA256,
        **HISTORICAL_BYTE_PINS,
    }
    observed = {}
    for relative_path, digest in sorted(expected.items()):
        observed[relative_path] = _require_file_hash(
            ROOT / relative_path, digest, role="frozen input/evidence"
        )
    corpus_root = ROOT / CORPUS_ROOT_REL
    for relative_path, digest in sorted(DATASET_BYTE_PINS.items()):
        key = f"{CORPUS_ROOT_REL}/{relative_path}"
        observed[key] = _require_file_hash(
            corpus_root / relative_path, digest, role="raw-hash-only dataset identity"
        )
    return observed


def _normalize_bonds(metadata: Mapping[str, Any]) -> list[list[int | str]]:
    return [
        [int(left), int(right), str(bond_type)]
        for left, right, bond_type in metadata.get("bonds", [])
    ]


def _topology_payload(atoms) -> dict[str, Any]:
    metadata = atoms.info.get("mol2")
    if not isinstance(metadata, dict):
        raise ValueError("Stage-0 OBC oracle requires frozen MOL2 metadata")
    payload = {
        "atomic_numbers": [int(value) for value in atoms.get_atomic_numbers()],
        "symbols": list(atoms.get_chemical_symbols()),
        "atom_ids": [int(value) for value in metadata.get("atom_ids", [])],
        "atom_names": [str(value) for value in metadata.get("atom_names", [])],
        "atom_types": [str(value) for value in metadata.get("atom_types", [])],
        "bonds": _normalize_bonds(metadata),
        "component_ids": [int(value) for value in metadata.get("component_ids", [])],
        "component_count": int(metadata.get("component_count", -1)),
        "mol2_identity_sha256": str(metadata.get("identity_sha256", "")),
    }
    atom_count = len(atoms)
    if any(
        len(payload[key]) != atom_count
        for key in (
            "atomic_numbers",
            "symbols",
            "atom_ids",
            "atom_names",
            "atom_types",
            "component_ids",
        )
    ):
        raise ValueError("MOL2 topology arrays do not match the atom count")
    if payload["component_count"] != 1 or set(payload["component_ids"]) != {0}:
        raise ValueError("Stage-0 corpus requires one connected component")
    payload["topology_sha256"] = core.sha256_bytes(core.canonical_json_bytes(payload))
    return payload


def _coordinate_payload(atoms) -> dict[str, Any]:
    coordinates = np.asarray(atoms.get_positions(), dtype=np.float64)
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be a finite (N,3) float64 array")
    values = coordinates.tolist()
    return {
        "coordinates_angstrom": values,
        "coordinate_sha256": core.sha256_bytes(core.canonical_json_bytes(values)),
    }


def _load_manifest_members() -> tuple[dict[str, Any], list[MemberBinding], str]:
    manifest_path = ROOT / MANIFEST_REL
    raw = manifest_path.read_bytes()
    if core.sha256_bytes(raw) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Canonical label-free manifest bytes changed")
    lowered = raw.lower()
    for token in (b'"experimental', b'"signed_error', b'"absolute_error', b'"score'):
        if token in lowered:
            raise ValueError(
                f"Canonical manifest contains forbidden field token {token!r}"
            )
    manifest = _semantic_load_json(MANIFEST_REL)
    if set(manifest) != MANIFEST_TOP_KEYS:
        raise ValueError("Canonical manifest top-level schema changed")
    expected_header = {
        "schema_version": 1,
        "case_count": 526,
        "charge_method": "am1bcc",
        "baseline_model": "obc2/ace",
        "source_partition": "development",
        "topology_atom_types": "FreeSolv-GAFF",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Canonical manifest field {key!r} changed")

    corpus_root = (ROOT / CORPUS_ROOT_REL).resolve()
    bindings: list[MemberBinding] = []
    aggregate_rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for row in manifest["records"]:
        if set(row) != MANIFEST_ROW_KEYS:
            raise ValueError("Canonical manifest member schema changed")
        compound_id = str(row["compound_id"])
        ids.append(compound_id)
        mol2_relative = Path(str(row["source_mol2_relative_path"]))
        record_relative = Path(str(row["source_record_relative_path"]))
        if mol2_relative != Path("dataset/mol2files_gaff") / f"{compound_id}.mol2":
            raise ValueError(f"Unexpected MOL2 member path for {compound_id}")
        if record_relative != (
            Path("records/development") / f"{compound_id}__am1bcc__obc2.json"
        ):
            raise ValueError(f"Unexpected historical member path for {compound_id}")
        mol2_path = (corpus_root / mol2_relative).resolve()
        record_path = (corpus_root / record_relative).resolve()
        for path in (mol2_path, record_path):
            try:
                path.relative_to(corpus_root)
            except ValueError as exc:
                raise ValueError(f"Corpus member escapes the root: {path}") from exc
        _require_file_hash(
            mol2_path, str(row["source_mol2_sha256"]), role="manifest-bound MOL2"
        )
        # Hash bytes only.  Never parse the historical record JSON.
        _require_file_hash(
            record_path,
            str(row["source_record_sha256"]),
            role="raw-hash-only historical record",
        )
        atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
        charges = tuple(float(value) for value in row["am1bcc_charges_e"])
        if len(charges) != len(atoms) or not all(
            math.isfinite(value) for value in charges
        ):
            raise ValueError(f"Invalid ordered AM1-BCC vector for {compound_id}")
        if abs(sum(charges)) > 1.0e-8:
            raise ValueError(f"AM1-BCC vector is not neutral for {compound_id}")
        charge_hash = core.sha256_bytes(_canonical_no_newline(list(charges)))
        binding = MemberBinding(
            compound_id=compound_id,
            mol2_relative_path=mol2_relative.as_posix(),
            mol2_sha256=str(row["source_mol2_sha256"]),
            record_relative_path=record_relative.as_posix(),
            record_sha256=str(row["source_record_sha256"]),
            charges_e=charges,
            charge_vector_sha256=charge_hash,
            atom_count=len(atoms),
        )
        bindings.append(binding)
        aggregate_rows.append(
            {
                "compound_id": compound_id,
                "source_mol2_relative_path": binding.mol2_relative_path,
                "source_mol2_sha256": binding.mol2_sha256,
                "source_record_relative_path": binding.record_relative_path,
                "source_record_sha256": binding.record_sha256,
                "am1bcc_charge_vector_sha256": charge_hash,
                "atom_count": binding.atom_count,
            }
        )
    if ids != sorted(ids) or len(ids) != len(set(ids)) or len(ids) != 526:
        raise ValueError("Canonical manifest IDs must be 526 sorted unique values")
    aggregate = core.sha256_bytes(_canonical_no_newline(aggregate_rows))
    if aggregate != EXPECTED_SAFE_MEMBER_BINDING_SHA256:
        raise ValueError(
            "Safe ordered member binding changed: "
            f"expected {EXPECTED_SAFE_MEMBER_BINDING_SHA256}, observed {aggregate}"
        )
    return manifest, bindings, aggregate


def _validate_protocol() -> tuple[dict[str, Any], str]:
    _require_file_hash(
        ROOT / PROTOCOL_REL, EXPECTED_PROTOCOL_SHA256, role="base method protocol"
    )
    protocol = _semantic_load_json(PROTOCOL_REL)
    fingerprint = core.sha256_bytes(core.canonical_json_bytes(protocol))
    if fingerprint != EXPECTED_PROTOCOL_FINGERPRINT:
        raise ValueError("Base protocol canonical fingerprint changed")
    if protocol.get("protocol_id") != "maple-neutral-water-freesolv-v1":
        raise ValueError("Unexpected base protocol ID")
    if protocol.get("dataset", {}).get("commit") != EXPECTED_DATASET_COMMIT:
        raise ValueError("Pinned FreeSolv dataset commit changed")
    methods = protocol.get("methods", {})
    if (
        methods.get("nonpolar") != "ace"
        or methods.get("openmm_platform") != "Reference"
        or "am1bcc" not in methods.get("charge_methods", [])
        or "obc2" not in methods.get("gb_models", [])
        or methods.get("charge_geometry_policy") != "keep-reference-geometry-fixed"
    ):
        raise ValueError("Base AM1-BCC/OBC-II/ACE method identity changed")
    return protocol, fingerprint


def _preflight() -> dict[str, Any]:
    concurrent_tests = _assert_stage0_worktree()
    production_sources = _snapshot_production_sources()
    fixed_bytes = _snapshot_fixed_bytes()
    protocol, fingerprint = _validate_protocol()
    manifest, bindings, aggregate = _load_manifest_members()
    snapshot = {
        "git_head": BASE_HEAD,
        "generator_sha256": core.sha256_file(ROOT / GENERATOR_REL),
        "production_sources": production_sources,
        "fixed_bytes_sha256": fixed_bytes,
        "protocol_fingerprint": fingerprint,
        "dataset_commit": protocol["dataset"]["commit"],
        "manifest_case_count": int(manifest["case_count"]),
        "safe_member_binding_sha256": aggregate,
        "stage0_test_sha256": (
            core.sha256_file(ROOT / STAGE0_TEST_REL)
            if (ROOT / STAGE0_TEST_REL).is_file()
            else None
        ),
        "concurrent_noninput_test_files_sha256": concurrent_tests,
    }
    snapshot["source_freeze_sha256"] = core.sha256_bytes(
        core.canonical_json_bytes(snapshot)
    )
    return {"snapshot": snapshot, "bindings": bindings}


def _exercise_decoder_guard(
    guard: SemanticDecoderGuard, bindings: list[MemberBinding]
) -> dict[str, Any]:
    """Run real bypass probes before numerical work and return their receipts."""
    first = bindings[0]
    record_path = ROOT / CORPUS_ROOT_REL / first.record_relative_path
    raw_hash = core.sha256_file(record_path)
    if raw_hash != first.record_sha256:
        raise ValueError("Raw historical-record hash probe changed")

    guard.reset()
    with guard.active():
        try:
            json.loads(record_path.read_bytes())
        except PermissionError:
            pass
        else:
            raise AssertionError("json.loads historical-byte bypass was not blocked")
        with record_path.open("r", encoding="utf-8") as handle:
            try:
                json.load(handle)
            except PermissionError:
                pass
            else:
                raise AssertionError("json.load historical-path bypass was not blocked")
        # A known-safe tracked manifest remains decodable while the guard is active.
        safe = json.loads((ROOT / MANIFEST_REL).read_bytes())
        if int(safe.get("case_count", -1)) != 526:
            raise AssertionError("Guard corrupted safe-manifest decoding")

    probes = list(guard.blocked)
    if len(probes) != 2 or {item["api"] for item in probes} != {
        "json.load",
        "json.loads",
    }:
        raise AssertionError("Semantic decoder guard probe receipt is incomplete")
    guard.reset()
    return {
        "enforced": True,
        "forbidden_member_digest_count": len(guard._forbidden_digests),
        "blocked_probe_count": len(probes),
        "blocked_probes": probes,
        "raw_hash_probe_allowed": True,
        "safe_manifest_decode_allowed": True,
        "runtime_violation_count": 0,
    }


def _openmm_identity() -> dict[str, Any]:
    version = importlib.metadata.version("openmm")
    if version != EXPECTED_OPENMM_VERSION:
        raise ImportError(
            f"Stage-0 oracle requires OpenMM {EXPECTED_OPENMM_VERSION}; observed {version}"
        )
    from maple.function.calculator.extra_correction.implicit.openmm_compat import (
        require_verified_obc2_expression_source,
    )
    from openmm import Platform

    compatibility = require_verified_obc2_expression_source()
    source_path = Path(compatibility["source_path"]).resolve()
    source_sha = core.sha256_file(source_path)
    if source_sha != EXPECTED_CUSTOMGBFORCES_SHA256:
        raise ImportError("Installed OpenMM customgbforces.py bytes changed")
    available = [
        Platform.getPlatform(index).getName()
        for index in range(Platform.getNumPlatforms())
    ]
    if "Reference" not in available:
        raise RuntimeError("OpenMM Reference platform is unavailable")
    try:
        installed_relative = source_path.relative_to(
            Path(sys.prefix).resolve()
        ).as_posix()
    except ValueError:
        installed_relative = source_path.name
    return {
        "provider": "openmm",
        "version": version,
        "platform": "Reference",
        "available_platforms": available,
        "customgbforces_module": "openmm.app.internal.customgbforces",
        "customgbforces_installed_relative_path": installed_relative,
        "customgbforces_sha256": source_sha,
        "vendored_audit_source": VENDORED_OPENMM_SOURCE_REL,
        "vendored_audit_source_sha256": _require_file_hash(
            ROOT / VENDORED_OPENMM_SOURCE_REL,
            EXPECTED_CUSTOMGBFORCES_SHA256,
            role="frozen upstream OpenMM audit source",
        ),
        "formula_source_profile": FORMULA_SOURCE_PROFILE,
    }


@contextmanager
def _network_denied():
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def denied(*_args, **_kwargs):
        raise RuntimeError("Network access is forbidden during Stage-0 oracle freezing")

    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.create_connection = denied  # type: ignore[assignment]
    previous = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MAPLE_OFFLINE")
    }
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MAPLE_OFFLINE="1")
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parameter_payload(provider) -> dict[str, Any]:
    parameters = provider.obc2_parameters
    if parameters is None:
        raise RuntimeError("OpenMM oracle did not expose finalized OBC-II parameters")
    raw = np.asarray(parameters.provider_parameters, dtype=np.float64)
    finalized = np.asarray(
        [
            list(provider._total.force.getParticleParameters(index))
            for index in range(provider._total.force.getNumParticles())
        ],
        dtype=np.float64,
    )
    if finalized.shape != (len(parameters.charges), 3):
        raise ValueError("Unexpected finalized OpenMM OBC-II particle shape")
    np.testing.assert_array_equal(finalized[:, 0], parameters.charges)
    np.testing.assert_allclose(
        finalized[:, 1], parameters.offset_radii_nm, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        finalized[:, 2],
        parameters.scaled_offset_radii_nm,
        rtol=0.0,
        atol=1.0e-15,
    )
    payload = {
        "radius_profile": provider.radius_result.profile,
        "provider_parameters_radius_nm_screen": raw.tolist(),
        "intrinsic_radii_angstrom": np.asarray(
            provider.radius_result.radii_angstrom, dtype=np.float64
        ).tolist(),
        "screen_factors": raw[:, 1].tolist(),
        "offset_radii_nm": np.asarray(
            parameters.offset_radii_nm, dtype=np.float64
        ).tolist(),
        "scaled_offset_radii_nm": np.asarray(
            parameters.scaled_offset_radii_nm, dtype=np.float64
        ).tolist(),
        "finalized_particle_charge_offset_scaled": finalized.tolist(),
        "offset_nm": 0.009,
        "solute_dielectric": float(parameters.solute_dielectric),
        "solvent_dielectric": float(parameters.solvent_dielectric),
    }
    arrays = (
        raw,
        np.asarray(provider.radius_result.radii_angstrom),
        np.asarray(parameters.offset_radii_nm),
        np.asarray(parameters.scaled_offset_radii_nm),
        finalized,
    )
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("OpenMM OBC-II parameter oracle contains nonfinite values")
    payload["parameter_payload_sha256"] = core.sha256_bytes(
        core.canonical_json_bytes(payload)
    )
    return payload


def _scalar_payload(result, *, include_forces: bool) -> dict[str, Any]:
    polar = float(result.components_hartree["polar"])
    nonpolar = float(result.components_hartree["nonpolar"])
    total = float(result.energy_hartree)
    if not all(math.isfinite(value) for value in (polar, nonpolar, total)):
        raise ValueError("OpenMM OBC-II oracle produced a nonfinite energy")
    if (
        abs((polar + nonpolar) - total)
        > FROZEN_COMPARISON_CONTRACT["component_abs_hartree"]
    ):
        raise ValueError("OpenMM OBC-II component closure exceeds the existing gate")
    payload: dict[str, Any] = {
        "polar_hartree": polar,
        "nonpolar_hartree": nonpolar,
        "total_hartree": total,
    }
    if include_forces:
        forces = np.asarray(result.forces_hartree_per_angstrom, dtype=np.float64)
        if (
            forces.ndim != 2
            or forces.shape[1:] != (3,)
            or not np.isfinite(forces).all()
        ):
            raise ValueError("OpenMM OBC-II oracle produced invalid forces")
        payload["forces_hartree_per_angstrom"] = forces.tolist()
    return payload


def _profile_record(atoms, charges: Iterable[float], nonpolar: str, *, forces: bool):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    provider = OpenMMGB(
        atoms,
        np.asarray(list(charges), dtype=np.float64),
        model="obc2",
        nonpolar=nonpolar,
        platform="Reference",
    )
    if provider.platform != "Reference":
        raise RuntimeError(
            f"Requested Reference but OpenMM selected {provider.platform}"
        )
    parameters = _parameter_payload(provider)
    result = provider.evaluate(atoms, need_forces=forces)
    scalar = _scalar_payload(result, include_forces=forces)
    if nonpolar == "none" and scalar["nonpolar_hartree"] != 0.0:
        raise ValueError("OBC-II/none oracle must return exact zero nonpolar energy")
    profile = {
        "nonpolar": nonpolar,
        "scalar_profile": f"openmm-8.5.2-obc2-{nonpolar}-reference-v1",
        "formula_source_profile": FORMULA_SOURCE_PROFILE,
        "parameters": parameters,
        **scalar,
    }
    if nonpolar == "none" and "ace" in profile["scalar_profile"]:
        raise AssertionError("OBC-II/none record is mislabeled as ACE")
    return profile


def _record_digest(record: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in record.items() if key != "record_content_sha256"
    }
    return core.sha256_bytes(core.canonical_json_bytes(payload))


def _freeze_526(bindings: list[MemberBinding]) -> list[dict[str, Any]]:
    corpus_root = ROOT / CORPUS_ROOT_REL
    records: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings, start=1):
        path = corpus_root / binding.mol2_relative_path
        atoms = MOL2Reader(str(path), charge=0, mult=1)
        atoms.set_initial_charges(np.asarray(binding.charges_e, dtype=np.float64))
        topology = _topology_payload(atoms)
        coordinates = _coordinate_payload(atoms)
        common = {
            "compound_id": binding.compound_id,
            "source_mol2_relative_path": binding.mol2_relative_path,
            "source_mol2_sha256": binding.mol2_sha256,
            "historical_record_relative_path": binding.record_relative_path,
            "historical_record_sha256": binding.record_sha256,
            "atom_count": binding.atom_count,
            "topology": topology,
            **coordinates,
            "ordered_am1bcc_charges_e": list(binding.charges_e),
            "charge_vector_sha256": binding.charge_vector_sha256,
        }
        parameters_by_profile: dict[str, str] = {}
        polar_by_profile: dict[str, float] = {}
        for nonpolar in PROFILE_ORDER:
            profile = _profile_record(atoms, binding.charges_e, nonpolar, forces=False)
            parameters_by_profile[nonpolar] = profile["parameters"][
                "parameter_payload_sha256"
            ]
            polar_by_profile[nonpolar] = profile["polar_hartree"]
            record = {**common, **profile}
            record["input_identity_sha256"] = core.sha256_bytes(
                core.canonical_json_bytes(common)
            )
            record["record_content_sha256"] = _record_digest(record)
            records.append(record)
        if parameters_by_profile["ace"] != parameters_by_profile["none"]:
            raise ValueError(
                f"ACE/NONE parameter identity differs for {binding.compound_id}"
            )
        if abs(polar_by_profile["ace"] - polar_by_profile["none"]) > (
            FROZEN_COMPARISON_CONTRACT["component_abs_hartree"]
        ):
            raise ValueError(
                f"ACE/NONE polar identity exceeds existing gate for {binding.compound_id}"
            )
        if index == 1 or index % 25 == 0 or index == len(bindings):
            print(f"[stage0] 526 oracle {index}/{len(bindings)}", flush=True)
    return records


def _portable_case_path(case: str) -> Path:
    return (
        ROOT
        / "tests/solvation/data/amber_gb_reference/audit"
        / case
        / "normalized.mol2"
    )


def _freeze_portable_12() -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for index, case in enumerate(FROZEN_CASES):
        path = _portable_case_path(case)
        relative_path = _relative(path)
        source_hash = core.sha256_file(path)
        source_hashes[relative_path] = source_hash
        atoms = MOL2Reader(str(path), charge=0, mult=1)
        charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
        if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
            raise ValueError(f"Invalid portable-corpus charges for {case}")
        charge_values = charges.tolist()
        charge_hash = core.sha256_bytes(_canonical_no_newline(charge_values))
        topology = _topology_payload(atoms)
        displaced = atoms.copy()
        displaced.positions += np.random.default_rng(20260922 + index).normal(
            scale=2.0e-4, size=(len(atoms), 3)
        )
        profile_parameters: dict[str, str] = {}
        for nonpolar in PROFILE_ORDER:
            # Build one provider per exact profile, then evaluate the two frozen
            # geometries through that same immutable topology/parameter context.
            from maple.function.calculator.extra_correction.implicit.openmm_gb import (
                OpenMMGB,
            )

            provider = OpenMMGB(
                atoms,
                charges,
                model="obc2",
                nonpolar=nonpolar,
                platform="Reference",
            )
            if provider.platform != "Reference":
                raise RuntimeError("Portable corpus did not use OpenMM Reference")
            parameters = _parameter_payload(provider)
            profile_parameters[nonpolar] = parameters["parameter_payload_sha256"]
            for geometry_name, geometry in (("base", atoms), ("displaced", displaced)):
                common = {
                    "case": case,
                    "geometry": geometry_name,
                    "nonpolar": nonpolar,
                    "source_mol2_relative_path": relative_path,
                    "source_mol2_sha256": source_hash,
                    "atom_count": len(atoms),
                    "topology": topology,
                    **_coordinate_payload(geometry),
                    "ordered_charges_e": charge_values,
                    "charge_vector_sha256": charge_hash,
                    "formula_source_profile": FORMULA_SOURCE_PROFILE,
                    "scalar_profile": (f"openmm-8.5.2-obc2-{nonpolar}-reference-v1"),
                    "parameters": parameters,
                }
                result = provider.evaluate(geometry, need_forces=True)
                scalar = _scalar_payload(result, include_forces=True)
                if nonpolar == "none" and scalar["nonpolar_hartree"] != 0.0:
                    raise ValueError("Portable OBC-II/none nonpolar energy is not zero")
                record = {**common, **scalar}
                record["input_identity_sha256"] = core.sha256_bytes(
                    core.canonical_json_bytes(common)
                )
                record["record_content_sha256"] = _record_digest(record)
                records.append(record)
        if profile_parameters["ace"] != profile_parameters["none"]:
            raise ValueError(f"Portable ACE/NONE parameters differ for {case}")
        print(f"[stage0] portable E/F {index + 1}/{len(FROZEN_CASES)}", flush=True)
    return records, source_hashes


def _access_policy() -> dict[str, Any]:
    return {
        "semantic_json_allowlist": sorted(SEMANTIC_JSON_ALLOWLIST),
        "semantic_mol2_inputs": [
            f"{CORPUS_ROOT_REL}/dataset/mol2files_gaff/<manifest-bound-id>.mol2",
            "tests/solvation/data/amber_gb_reference/audit/"
            "<frozen-case>/normalized.mol2",
        ],
        "raw_hash_only_inputs": [
            f"{CORPUS_ROOT_REL}/records/development/" "<id>__am1bcc__obc2.json",
            f"{CORPUS_ROOT_REL}/dataset/database.txt",
            f"{CORPUS_ROOT_REL}/dataset/database.json",
            f"{CORPUS_ROOT_REL}/dataset/mol2files_gaff.tar.gz",
            *sorted(HISTORICAL_BYTE_PINS),
        ],
        "denied_semantic_globs": [
            "**/prepared.json",
            "**/*summary*.json",
            "**/*reserve-score*.json",
            "**/records/development/*__am1bcc__obc2.json",
            "**/database.txt",
            "**/database.json",
        ],
        "network": "denied",
        "charge_generation": "denied-reuse-manifest-vector",
        "qm_generation": "denied",
        "historical_artifact_rewrite": "denied",
    }


def _base_artifact(
    *,
    artifact_type: str,
    source_snapshot: Mapping[str, Any],
    openmm_identity: Mapping[str, Any],
    decoder_guard_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "recorded_date": RECORDED_DATE,
        "claim_boundary": (
            "Pre-change equation-oracle values only; no fitting, ranking, "
            "hydration-accuracy, force-field retuning, or capability admission."
        ),
        "data_access_policy": _access_policy(),
        "semantic_decoder_guard": dict(decoder_guard_receipt),
        "source_freeze": dict(source_snapshot),
        "oracle_identity": dict(openmm_identity),
        "comparison_contract": dict(FROZEN_COMPARISON_CONTRACT),
    }


def _build_artifacts(
    preflight: Mapping[str, Any],
    openmm_identity: Mapping[str, Any],
    decoder_guard_receipt: Mapping[str, Any],
):
    portable_records, portable_sources = _freeze_portable_12()
    portable = _base_artifact(
        artifact_type="route1-obc2-openmm-8.5.2-portable-reference-v1",
        source_snapshot=preflight["snapshot"],
        openmm_identity=openmm_identity,
        decoder_guard_receipt=decoder_guard_receipt,
    )
    portable.update(
        case_count=len(FROZEN_CASES),
        cases=list(FROZEN_CASES),
        geometry_order=list(GEOMETRY_ORDER),
        profile_order=list(PROFILE_ORDER),
        comparison_count=len(portable_records),
        source_mol2_sha256=portable_sources,
        records=portable_records,
    )
    core.seal_artifact(portable)
    portable_file_sha = core.sha256_bytes(core.canonical_json_bytes(portable))

    oracle_records = _freeze_526(list(preflight["bindings"]))
    main = _base_artifact(
        artifact_type="route1-obc2-am1bcc-526-reference-v1",
        source_snapshot=preflight["snapshot"],
        openmm_identity=openmm_identity,
        decoder_guard_receipt=decoder_guard_receipt,
    )
    main.update(
        input_identity={
            "protocol": PROTOCOL_REL,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "protocol_fingerprint": EXPECTED_PROTOCOL_FINGERPRINT,
            "dataset_commit": EXPECTED_DATASET_COMMIT,
            "manifest": MANIFEST_REL,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "safe_member_binding_sha256": EXPECTED_SAFE_MEMBER_BINDING_SHA256,
            "corpus_root_role": "ignored-mandatory-source",
            "case_count": 526,
            "geometry": "one-pinned-freesolv-gaff-mol2-geometry",
            "charge_method": "frozen-ordered-am1bcc",
        },
        profile_order=list(PROFILE_ORDER),
        case_count=526,
        profile_count=2,
        record_count=len(oracle_records),
        portable_fixture={
            "path": PORTABLE_REL,
            "file_sha256": portable_file_sha,
            "content_sha256": portable["content_sha256"],
            "comparison_count": portable["comparison_count"],
        },
        records=oracle_records,
    )
    core.seal_artifact(main)
    return portable, main


def _is_finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_is_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _validate_records(records: list[dict[str, Any]], *, forces: bool) -> None:
    for record in records:
        if record.get("record_content_sha256") != _record_digest(record):
            raise ValueError("Stage-0 record content hash mismatch")
        if not _is_finite_tree(record):
            raise ValueError("Stage-0 record contains nonfinite numerical data")
        atom_count = int(record["atom_count"])
        parameters = record["parameters"]
        arrays = (
            parameters["provider_parameters_radius_nm_screen"],
            parameters["intrinsic_radii_angstrom"],
            parameters["screen_factors"],
            parameters["offset_radii_nm"],
            parameters["scaled_offset_radii_nm"],
            parameters["finalized_particle_charge_offset_scaled"],
        )
        if any(len(values) != atom_count for values in arrays):
            raise ValueError("Stage-0 parameter array length mismatch")
        if (
            len(
                record["ordered_charges_e"]
                if "ordered_charges_e" in record
                else record["ordered_am1bcc_charges_e"]
            )
            != atom_count
        ):
            raise ValueError("Stage-0 charge vector length mismatch")
        if len(record["coordinates_angstrom"]) != atom_count:
            raise ValueError("Stage-0 coordinate length mismatch")
        closure = abs(
            record["polar_hartree"]
            + record["nonpolar_hartree"]
            - record["total_hartree"]
        )
        if closure > FROZEN_COMPARISON_CONTRACT["component_abs_hartree"]:
            raise ValueError("Stage-0 component closure exceeds existing contract")
        if record["nonpolar"] == "none":
            if record["nonpolar_hartree"] != 0.0:
                raise ValueError("Stage-0 NONE record has a nonzero nonpolar term")
            if "ace" in record["scalar_profile"]:
                raise ValueError("Stage-0 NONE record has an ACE scalar label")
        if forces:
            force_values = record.get("forces_hartree_per_angstrom")
            if not isinstance(force_values, list) or len(force_values) != atom_count:
                raise ValueError("Portable Stage-0 force shape mismatch")
            if any(len(row) != 3 for row in force_values):
                raise ValueError("Portable Stage-0 force row shape mismatch")


def _validate_portable(portable: Mapping[str, Any]) -> None:
    _assert_output_key_policy(portable)
    if core.artifact_content_sha256(portable) != portable.get("content_sha256"):
        raise ValueError("Portable fixture self-hash mismatch")
    if portable.get("case_count") != 12 or portable.get("comparison_count") != 48:
        raise ValueError("Portable fixture must contain 12 cases and 48 comparisons")
    if tuple(portable.get("cases", ())) != FROZEN_CASES:
        raise ValueError("Portable fixture case order changed")
    records = list(portable.get("records", ()))
    expected = {
        (case, profile, geometry)
        for case in FROZEN_CASES
        for profile in PROFILE_ORDER
        for geometry in GEOMETRY_ORDER
    }
    observed = {(row["case"], row["nonpolar"], row["geometry"]) for row in records}
    if observed != expected or len(records) != len(expected):
        raise ValueError("Portable fixture matrix is incomplete or duplicated")
    _validate_records(records, forces=True)


def _validate_main(main: Mapping[str, Any], portable: Mapping[str, Any]) -> None:
    _assert_output_key_policy(main)
    if core.artifact_content_sha256(main) != main.get("content_sha256"):
        raise ValueError("526 oracle self-hash mismatch")
    if (
        main.get("case_count") != 526
        or main.get("profile_count") != 2
        or main.get("record_count") != 1052
    ):
        raise ValueError("526 oracle matrix shape changed")
    records = list(main.get("records", ()))
    ids = [row["compound_id"] for row in records]
    if len(set(ids)) != 526 or any(
        ids.count(compound_id) != 2 for compound_id in set(ids)
    ):
        raise ValueError("526 oracle case/profile membership is incomplete")
    for compound_id in set(ids):
        if {
            row["nonpolar"] for row in records if row["compound_id"] == compound_id
        } != set(PROFILE_ORDER):
            raise ValueError(
                f"526 oracle profile coverage is incomplete for {compound_id}"
            )
    _validate_records(records, forces=False)
    expected_fixture_sha = core.sha256_bytes(core.canonical_json_bytes(portable))
    if main["portable_fixture"]["file_sha256"] != expected_fixture_sha:
        raise ValueError("526 oracle does not bind the portable fixture bytes")


def _atomic_write_pair(portable: Mapping[str, Any], main: Mapping[str, Any]) -> None:
    destinations = ((ROOT / PORTABLE_REL, portable), (ROOT / OUTPUT_REL, main))
    for destination, _value in destinations:
        if destination.exists():
            raise FileExistsError(
                f"Stage-0 outputs are immutable; refusing to overwrite {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for destination, value in destinations:
            payload = core.canonical_json_bytes(value)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths.append((Path(handle.name), destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in temporary_paths:
            temporary.unlink(missing_ok=True)


def _load_and_validate_outputs() -> dict[str, Any]:
    portable_path = ROOT / PORTABLE_REL
    output_path = ROOT / OUTPUT_REL
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    main = json.loads(output_path.read_text(encoding="utf-8"))
    _validate_portable(portable)
    _validate_main(main, portable)
    if main["source_freeze"]["generator_sha256"] != core.sha256_file(
        ROOT / GENERATOR_REL
    ):
        raise ValueError("Generator bytes changed after Stage-0 freeze")
    return {
        "portable_path": PORTABLE_REL,
        "portable_file_sha256": core.sha256_file(portable_path),
        "portable_content_sha256": portable["content_sha256"],
        "portable_comparison_count": portable["comparison_count"],
        "output_path": OUTPUT_REL,
        "output_file_sha256": core.sha256_file(output_path),
        "output_content_sha256": main["content_sha256"],
        "output_record_count": main["record_count"],
        "source_freeze_sha256": main["source_freeze"]["source_freeze_sha256"],
    }


def _self_test() -> None:
    benign = {"records": [{"polar_hartree": -1.0, "total_hartree": -1.0}]}
    _assert_output_key_policy(benign)
    for forbidden in ("experimental_value", "score", "absolute_error", "mae"):
        try:
            _assert_output_key_policy({forbidden: 1.0})
        except ValueError:
            pass
        else:
            raise AssertionError(f"Forbidden key self-test failed: {forbidden}")
    try:
        _semantic_load_json("docs/implicit-solvation/benchmarks/prepared.json")
    except PermissionError:
        pass
    else:
        raise AssertionError("Semantic-read allowlist self-test failed")
    if core.sha256_bytes(_canonical_no_newline({"b": 2, "a": 1})) != (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    ):
        raise AssertionError("Canonical JSON self-test failed")
    print("stage0-self-test=passed")


def _print_preflight(
    preflight: Mapping[str, Any],
    openmm_identity: Mapping[str, Any],
    decoder_guard_receipt: Mapping[str, Any],
):
    print(
        json.dumps(
            {
                "status": "SOURCE_FROZEN_READY",
                "generator": GENERATOR_REL,
                "generator_sha256": preflight["snapshot"]["generator_sha256"],
                "output": OUTPUT_REL,
                "portable_fixture": PORTABLE_REL,
                "git_head": BASE_HEAD,
                "source_freeze_sha256": preflight["snapshot"]["source_freeze_sha256"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "safe_member_binding_sha256": EXPECTED_SAFE_MEMBER_BINDING_SHA256,
                "case_count": len(preflight["bindings"]),
                "profiles": list(PROFILE_ORDER),
                "openmm": openmm_identity,
                "semantic_decoder_guard": dict(decoder_guard_receipt),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _generate() -> None:
    preflight = _preflight()
    openmm_identity = _openmm_identity()
    guard = SemanticDecoderGuard(preflight["bindings"])
    guard_receipt = _exercise_decoder_guard(guard, list(preflight["bindings"]))
    _print_preflight(preflight, openmm_identity, guard_receipt)
    source_before = {
        "production": _snapshot_production_sources(),
        "fixed": _snapshot_fixed_bytes(),
    }
    with _network_denied(), guard.active():
        portable, main = _build_artifacts(preflight, openmm_identity, guard_receipt)
    if guard.blocked:
        raise RuntimeError(
            "A forbidden semantic decoder attempt occurred during freezing"
        )
    _validate_portable(portable)
    _validate_main(main, portable)
    source_after = {
        "production": _snapshot_production_sources(),
        "fixed": _snapshot_fixed_bytes(),
    }
    if source_before != source_after:
        raise RuntimeError(
            "Stage-0 sources or historical evidence changed during execution"
        )
    _atomic_write_pair(portable, main)
    result = _load_and_validate_outputs()
    print(json.dumps({"status": "SEALED", **result}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.validate_only:
        print(json.dumps(_load_and_validate_outputs(), indent=2, sort_keys=True))
        return
    if args.preflight_only:
        preflight = _preflight()
        identity = _openmm_identity()
        guard = SemanticDecoderGuard(preflight["bindings"])
        receipt = _exercise_decoder_guard(guard, list(preflight["bindings"]))
        _print_preflight(preflight, identity, receipt)
        return
    _generate()


if __name__ == "__main__":
    main()
