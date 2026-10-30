"""Thin local-private adapter for the installed FP32 exact CHA SP engine.

This module contains only input/identity validation and result conversion.  The
private numerical implementation is supplied by the separately installed
``maple_cha_exact_private`` distribution and is never loaded from the checkout.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np
import torch

from maple.function.calculator.analytic_protocol import CheckpointFileIdentity
from maple.function.device import resolve_torch_device
from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    mol2_identity_sha256,
)

from .charges import _molecular_charge_and_mult
from .result import SolvationResult

KCAL_PER_MOL_PER_HARTREE = 627.5094740631
_ENGINE_MODULE = "maple_cha_exact_private"
_ENGINE_DISTRIBUTION = "maple-cha-exact-private"
_ENGINE_VERSION = "0.4.0"
_API_VERSION = 4
_ARITHMETIC_PROFILE = "torch-fp32-sp-v1"
_SHA256_LENGTH = 64
_MAX_PREPARED_TOPOLOGY_BYTES = 16 * 1024 * 1024
_RELEASE_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "private_cha_torch_fp32_release_v1.json"
)
# Root qualification sets this only after the final private/public source freeze.
_PINNED_RELEASE_FILE_SHA256: str | None = '373666b7fdcbecaaece15b283b740fe5c80890044131d5050f717f2904920231'


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _release_aggregate_payload(release: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "api_version": release["api_version"],
        "distribution": release["distribution"],
        "maple_distribution": release["maple_distribution"],
        "arithmetic_profile": release["arithmetic_profile"],
        "model_profile_sha256": release["model_profile_sha256"],
        "runtime": release["runtime"],
        "runtime_contract": release["runtime_contract"],
        "private_source_sha256": release["private_source_sha256"],
        "private_provenance_sha256": release["private_provenance_sha256"],
        "maple_public_source": release["maple_public_source"],
    }


def _validate_release_contract(
    release: Mapping[str, Any], *, file_sha256: str, expected_file_sha256: str
) -> dict[str, Any]:
    if file_sha256 != expected_file_sha256:
        raise RuntimeError("Private CHA pinned release JSON file hash changed.")
    required = {
        "schema_version",
        "release_id",
        "api_version",
        "distribution",
        "maple_distribution",
        "arithmetic_profile",
        "model_profile_sha256",
        "runtime",
        "runtime_contract",
        "private_source_sha256",
        "private_provenance_sha256",
        "maple_public_source",
        "approved_source_aggregate_sha256",
        "content_sha256",
    }
    if set(release) != required:
        raise RuntimeError("Private CHA release JSON has the wrong exact schema.")
    if release["schema_version"] != 1 or release["release_id"] != (
        "private-cha-torch-fp32-release-v1"
    ):
        raise RuntimeError("Private CHA release JSON identity is invalid.")
    without_content = dict(release)
    recorded_content = _validate_sha256(
        without_content.pop("content_sha256"), name="private CHA release content hash"
    )
    if _sha256_bytes(_canonical_json(without_content)) != recorded_content:
        raise RuntimeError("Private CHA release JSON content seal is invalid.")
    aggregate = _sha256_bytes(_canonical_json(_release_aggregate_payload(release)))
    if aggregate != release["approved_source_aggregate_sha256"]:
        raise RuntimeError("Private CHA approved source aggregate is invalid.")
    if release["api_version"] != _API_VERSION:
        raise RuntimeError("Private CHA pinned release requires API_VERSION=4.")
    if release["arithmetic_profile"] != _ARITHMETIC_PROFILE:
        raise RuntimeError(
            "Private CHA pinned release has the wrong arithmetic profile."
        )
    if release["distribution"] != {
        "name": _ENGINE_DISTRIBUTION,
        "version": _ENGINE_VERSION,
        "package": _ENGINE_MODULE,
    }:
        raise RuntimeError("Private CHA pinned distribution identity is invalid.")
    if release["maple_distribution"] != {"name": "maple"}:
        raise RuntimeError("Private CHA pinned MAPLE distribution identity is invalid.")
    _validate_sha256(
        release["model_profile_sha256"], name="private CHA pinned model profile"
    )
    runtime = release["runtime"]
    if set(runtime) != {"torch_version", "numpy_version", "python_major_minor"}:
        raise RuntimeError("Private CHA pinned runtime identity is invalid.")
    if not all(isinstance(value, str) and value for value in runtime.values()):
        raise RuntimeError("Private CHA pinned runtime versions must be strings.")
    if release["runtime_contract"] != {
        "supported_device_types": ["cpu", "cuda"],
        "default_device": "cpu",
        "floating_dtype": "float32",
        "supports_forces": False,
        "failure_fallback": False,
    }:
        raise RuntimeError("Private CHA pinned Torch runtime contract is invalid.")
    private_sources = release["private_source_sha256"]
    if not isinstance(private_sources, Mapping) or not private_sources:
        raise RuntimeError("Private CHA pinned private-source inventory is empty.")
    for name, digest in private_sources.items():
        if Path(str(name)).as_posix() != str(name) or not str(name).endswith(".py"):
            raise RuntimeError("Private CHA pinned private-source path is invalid.")
        _validate_sha256(digest, name=f"private CHA pinned source hash {name}")
    provenance = release["private_provenance_sha256"]
    required_provenance = {
        "share/doc/maple-cha-exact-private/NOTICE.md",
        "share/doc/maple-cha-exact-private/API.md",
        "share/doc/maple-cha-exact-private/BASELINE.json",
        "share/doc/maple-cha-exact-private/SOURCE_MAP.json",
    }
    if not isinstance(provenance, Mapping) or not required_provenance.issubset(
        provenance
    ):
        raise RuntimeError("Private CHA pinned provenance inventory is incomplete.")
    license_paths = {
        str(name)
        for name in provenance
        if str(name).startswith("share/licenses/maple-cha-exact-private/")
    }
    if not license_paths:
        raise RuntimeError("Private CHA pinned license inventory is empty.")
    for name, digest in provenance.items():
        if Path(str(name)).as_posix() != str(name) or Path(str(name)).is_absolute():
            raise RuntimeError("Private CHA pinned provenance path is invalid.")
        _validate_sha256(digest, name=f"private CHA pinned provenance hash {name}")
    public = release["maple_public_source"]
    if set(public) != {
        "maple_public_polar",
        "maple_public_cavity",
        "maple_public_dense_budget",
    }:
        raise RuntimeError("Private CHA pinned MAPLE source inventory is invalid.")
    for name, record in public.items():
        if set(record) != {"module", "relative_path", "sha256"}:
            raise RuntimeError(f"Private CHA pinned MAPLE source {name!r} is invalid.")
        _validate_sha256(record["sha256"], name=f"pinned MAPLE source hash {name}")
    return dict(release)


def _load_release_contract() -> tuple[dict[str, Any], CheckpointFileIdentity]:
    if _PINNED_RELEASE_FILE_SHA256 is None:
        raise RuntimeError(
            "The private CHA FP32 release manifest is not sealed; local activation "
            "remains blocked until root qualification freezes its exact hash."
        )
    try:
        payload = _RELEASE_DATA_PATH.read_bytes()
        release = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot load the pinned private CHA release JSON.") from exc
    if not isinstance(release, Mapping):
        raise RuntimeError("Private CHA release JSON must contain one object.")
    validated = _validate_release_contract(
        release,
        file_sha256=_sha256_bytes(payload),
        expected_file_sha256=_PINNED_RELEASE_FILE_SHA256,
    )
    bound = CheckpointFileIdentity.bind(_RELEASE_DATA_PATH)
    if bound.sha256 != _PINNED_RELEASE_FILE_SHA256:
        raise RuntimeError("Private CHA release JSON changed while it was bound.")
    return validated, bound


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    raise ValueError(f"Private CHA engine value is missing {name!r}.")


def _validate_sha256(value: Any, *, name: str) -> str:
    text = str(value).lower()
    if len(text) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return text


def _serialized_fp32_scalar(record: Any, name: str) -> np.float32:
    value = _value(record, name)
    if type(value) is not float:
        raise RuntimeError(
            f"Private CHA result field {name!r} must be a Python-float serialization "
            "of one finalized Torch-FP32 scalar."
        )
    converted = np.float32(value)
    if not math.isfinite(value) or float(converted) != value:
        raise RuntimeError(
            f"Private CHA result field {name!r} is not exactly representable in binary32."
        )
    return converted


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Private CHA result field {name!r} is invalid.")
    return value


def _optional_nonnegative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name=name)


def _runtime_prefix() -> Path:
    return Path(sys.prefix).resolve()


def _runtime_versions() -> dict[str, str]:
    torch = importlib.import_module("torch")
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"{label} is not inside the installed runtime prefix."
        ) from exc


def _source_directory_snapshot(path: Path) -> tuple[Any, frozenset[tuple[str, str]]]:
    observed = os.lstat(path)
    if not stat.S_ISDIR(observed.st_mode):
        raise RuntimeError("Private CHA bound source directory is not a directory.")
    entries: set[tuple[str, str]] = set()
    with os.scandir(path) as iterator:
        for entry in iterator:
            if entry.name == "__pycache__":
                continue
            if entry.is_symlink():
                kind = "symlink"
            elif entry.is_dir(follow_symlinks=False):
                kind = "directory"
            elif entry.is_file(follow_symlinks=False):
                kind = "python" if entry.name.endswith(".py") else "file"
            else:
                continue
            entries.add((entry.name, kind))
    token = (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )
    return token, frozenset(entries)


def _module_token(module: Any) -> tuple[Any, ...]:
    spec = getattr(module, "__spec__", None)
    search = getattr(module, "__path__", None)
    return (
        getattr(module, "__file__", None),
        getattr(spec, "origin", None),
        None if search is None else tuple(search),
    )


def _anchor_token(path: Path) -> tuple[int, int, int]:
    observed = os.lstat(path)
    return int(observed.st_dev), int(observed.st_ino), int(observed.st_mode)


def _file_metadata_token(path: Path) -> tuple[int, int, int, int, int, int]:
    return _stat_metadata_token(os.lstat(path))


def _stat_metadata_token(observed) -> tuple[int, int, int, int, int, int]:
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
        int(observed.st_mode),
    )


def _read_bounded_stable_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be one regular file.")
            if before.st_size > max_bytes:
                raise ValueError(
                    f"{label} exceeds the {max_bytes}-byte resource limit."
                )
            payload = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        current = os.stat(path)
    except OSError as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte resource limit.")
    if (
        _stat_metadata_token(before) != _stat_metadata_token(after)
        or _stat_metadata_token(after) != _stat_metadata_token(current)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was being read.")
    return payload


def _hash_stable_file(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be one regular file.")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = os.stat(path)
    except OSError as exc:
        raise ValueError(f"Cannot hash {label}: {path}") from exc
    if _stat_metadata_token(before) != _stat_metadata_token(
        after
    ) or _stat_metadata_token(after) != _stat_metadata_token(current):
        raise ValueError(f"{label} changed while it was being hashed.")
    return digest.hexdigest()


def _bind_anchors(paths) -> dict[str, tuple[int, int, int]]:
    anchors: dict[str, tuple[int, int, int]] = {}
    for path in paths:
        current = Path(path).resolve().parent
        for entry in (current, *current.parents):
            anchors.setdefault(str(entry), _anchor_token(entry))
    return anchors


def _require_anchors(anchors: Mapping[str, tuple[int, int, int]]) -> None:
    for path, expected in anchors.items():
        if _anchor_token(Path(path)) != expected:
            raise RuntimeError("Private CHA bound path ancestry changed after load.")


class _RuntimeSourceBinding:
    def __init__(self, *, files, directories, modules, anchors=None):
        self.files = tuple(files)
        self.file_metadata = {
            file.path: _file_metadata_token(Path(file.path)) for file in self.files
        }
        self.directories = dict(directories)
        self.modules = dict(modules)
        self.anchors = dict(anchors or {})

    def require_unchanged(self) -> None:
        try:
            for file in self.files:
                file.require_unchanged()
                if (
                    _file_metadata_token(Path(file.path))
                    != self.file_metadata[file.path]
                ):
                    raise RuntimeError(
                        "Private CHA bound source file metadata changed after load."
                    )
            _require_anchors(self.anchors)
            for path, expected in self.directories.items():
                if _source_directory_snapshot(Path(path)) != expected:
                    raise RuntimeError(
                        "Private CHA installed source inventory changed after load."
                    )
            executing_private = {
                name
                for name, module in tuple(sys.modules.items())
                if module is not None
                and (name == _ENGINE_MODULE or name.startswith(_ENGINE_MODULE + "."))
            }
            bound_private = {
                name
                for name in self.modules
                if name == _ENGINE_MODULE or name.startswith(_ENGINE_MODULE + ".")
            }
            if executing_private != bound_private:
                raise RuntimeError(
                    "Private CHA executing module inventory changed after load."
                )
            for name, (module, token) in self.modules.items():
                if (
                    sys.modules.get(name) is not module
                    or _module_token(module) != token
                ):
                    raise RuntimeError(
                        "Private CHA executing module identity changed after load."
                    )
        except OSError as exc:
            raise RuntimeError(
                "Private CHA installed source identity is no longer readable."
            ) from exc


def _distribution_owned_paths(distribution) -> set[Path]:
    files = distribution.files
    if files is None:
        raise RuntimeError("Installed distribution does not expose a file inventory.")
    return {Path(distribution.locate_file(item)).resolve() for item in files}


def _require_noneditable_distribution(distribution, *, label: str) -> None:
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return
    try:
        record = json.loads(direct_url)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} direct_url metadata is invalid.") from exc
    if bool(record.get("dir_info", {}).get("editable")):
        raise RuntimeError(f"{label} must be a non-editable installed distribution.")


def _bind_file(path: Path, expected_sha256: str, owned: set[Path], *, label: str):
    resolved = path.resolve()
    if resolved not in owned:
        raise RuntimeError(f"{label} is not owned by its installed distribution.")
    bound = CheckpointFileIdentity.bind(resolved)
    if bound.sha256 != expected_sha256:
        raise RuntimeError(f"{label} does not match the independently pinned source.")
    return bound


def _validate_engine_contract(
    engine: Any, identity: Mapping[str, Any], release: Mapping[str, Any]
) -> None:
    if getattr(engine, "API_VERSION", None) != _API_VERSION:
        raise RuntimeError("Private CHA engine requires API_VERSION=4.")
    if getattr(engine, "ARITHMETIC_PROFILE", None) != _ARITHMETIC_PROFILE:
        raise RuntimeError(
            "Private CHA engine arithmetic profile must be torch-fp32-sp-v1."
        )
    for entrypoint in (
        "load_topology",
        "bind_coordinates",
        "evaluate_exact_sp",
        "engine_identity",
    ):
        if not callable(getattr(engine, entrypoint, None)):
            raise RuntimeError(
                f"Private CHA API v4 entry point {entrypoint} is unavailable."
            )
    model_profile = str(getattr(engine, "MODEL_PROFILE_SHA256", ""))
    if model_profile != release["model_profile_sha256"]:
        raise RuntimeError(
            "Private CHA engine model profile does not match the independent release pin."
        )
    if str(identity.get("distribution_name", "")) != _ENGINE_DISTRIBUTION:
        raise RuntimeError(
            f"Private CHA engine distribution must be {_ENGINE_DISTRIBUTION}."
        )
    if identity.get("expected_distribution_version") != _ENGINE_VERSION:
        raise RuntimeError(f"Private CHA engine version must be {_ENGINE_VERSION}.")
    if identity.get("installed_distribution_version") != _ENGINE_VERSION:
        raise RuntimeError(f"Private CHA engine version must be {_ENGINE_VERSION}.")
    if identity.get("api_version") != _API_VERSION:
        raise RuntimeError("Private CHA engine identity requires API_VERSION=4.")
    if identity.get("arithmetic_profile") != _ARITHMETIC_PROFILE:
        raise RuntimeError(
            "Private CHA engine identity has the wrong arithmetic profile."
        )
    if identity.get("model_profile_sha256") != release["model_profile_sha256"]:
        raise RuntimeError(
            "Private CHA engine identity has the wrong model profile hash."
        )
    if _runtime_versions() != release["runtime"]:
        raise RuntimeError(
            "Private CHA runtime does not match the independently qualified versions."
        )
    private_hashes = identity.get("private_source_sha256")
    if private_hashes != release["private_source_sha256"]:
        raise RuntimeError(
            "Private CHA engine source inventory does not match the independent release pin."
        )
    installed_provenance = identity.get("installed_provenance_sha256")
    if not isinstance(installed_provenance, Mapping) or set(
        installed_provenance
    ) != set(release["private_provenance_sha256"]):
        raise RuntimeError(
            "Private CHA installed provenance inventory has the wrong exact file set."
        )
    for name, expected_sha256 in release["private_provenance_sha256"].items():
        record = installed_provenance[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"origin", "sha256"}
            or record["sha256"] != expected_sha256
        ):
            raise RuntimeError(
                "Private CHA installed provenance does not match the independent release pin."
            )
    public_hashes = identity.get("maple_public_source")
    pinned_public = release["maple_public_source"]
    if not isinstance(public_hashes, Mapping) or set(public_hashes) != set(
        pinned_public
    ):
        raise RuntimeError(
            "Private CHA engine MAPLE source inventory has the wrong exact source set."
        )
    for name, expected in pinned_public.items():
        record = public_hashes[name]
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != expected["sha256"]
        ):
            raise RuntimeError(
                "Private CHA engine MAPLE source hashes do not match the independent release pin."
            )


def _bind_installed_runtime(
    engine: Any, release: Mapping[str, Any]
) -> tuple[dict[str, Any], _RuntimeSourceBinding]:
    identity = engine.engine_identity()
    if not isinstance(identity, Mapping):
        raise RuntimeError("Private CHA engine_identity() must return a mapping.")
    _validate_engine_contract(engine, identity, release)
    prefix = _runtime_prefix()
    try:
        private_dist = metadata.distribution(_ENGINE_DISTRIBUTION)
        maple_dist = metadata.distribution(release["maple_distribution"]["name"])
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Private CHA and MAPLE must be installed distributions."
        ) from exc
    if private_dist.version != _ENGINE_VERSION:
        raise RuntimeError("Installed private CHA distribution has the wrong version.")
    _require_noneditable_distribution(private_dist, label="Private CHA")
    _require_noneditable_distribution(maple_dist, label="MAPLE")
    private_owned = _distribution_owned_paths(private_dist)
    maple_owned = _distribution_owned_paths(maple_dist)
    private_root = Path(private_dist.locate_file(_ENGINE_MODULE)).resolve()
    _require_within(private_root, prefix, label="Private CHA package")
    distribution_root = Path(private_dist.locate_file("")).resolve()
    _require_within(distribution_root, prefix, label="Private CHA distribution")
    if Path(
        str(identity.get("installed_distribution_metadata_origin", ""))
    ).resolve() != (distribution_root):
        raise RuntimeError("Private CHA distribution metadata origin is inconsistent.")

    expected_private = release["private_source_sha256"]
    actual_private = {
        path.relative_to(private_root).as_posix()
        for path in private_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if actual_private != set(expected_private):
        raise RuntimeError(
            "Private CHA installed source inventory is incomplete or has extras."
        )
    files = [
        _bind_file(
            private_root / relative,
            expected_private[relative],
            private_owned,
            label=f"Private CHA source {relative}",
        )
        for relative in sorted(expected_private)
    ]
    installed_provenance = identity["installed_provenance_sha256"]
    distribution_provenance = set()
    for item in private_dist.files or ():
        located = Path(private_dist.locate_file(item)).resolve()
        try:
            relative = located.relative_to(prefix).as_posix()
        except ValueError:
            continue
        if relative.startswith("share/doc/maple-cha-exact-private/") or (
            relative.startswith("share/licenses/maple-cha-exact-private/")
        ):
            distribution_provenance.add(relative)
    if distribution_provenance != set(release["private_provenance_sha256"]):
        raise RuntimeError(
            "Private CHA distribution provenance inventory is incomplete or has extras."
        )
    for relative, expected_sha256 in sorted(
        release["private_provenance_sha256"].items()
    ):
        expected_path = (prefix / relative).resolve()
        claimed_path = Path(str(installed_provenance[relative]["origin"])).resolve()
        if claimed_path != expected_path:
            raise RuntimeError(
                "Private CHA installed provenance origin is not prefix-relative as pinned."
            )
        _require_within(
            expected_path, prefix, label=f"Private CHA provenance {relative}"
        )
        files.append(
            _bind_file(
                expected_path,
                expected_sha256,
                private_owned,
                label=f"Private CHA provenance {relative}",
            )
        )
    record_items = [
        item
        for item in private_dist.files or ()
        if Path(str(item)).name == "RECORD"
        and Path(str(item)).parent.name.endswith(".dist-info")
    ]
    if len(record_items) != 1:
        raise RuntimeError("Private CHA installed distribution requires one RECORD.")
    record_path = Path(private_dist.locate_file(record_items[0])).resolve()
    _require_within(record_path, prefix, label="Private CHA RECORD")
    if record_path not in private_owned:
        raise RuntimeError("Private CHA RECORD is not distribution-owned.")
    files.append(CheckpointFileIdentity.bind(record_path))
    directories = {
        str(path): _source_directory_snapshot(path)
        for path in sorted(
            {
                private_root,
                *(
                    path
                    for path in private_root.rglob("*")
                    if path.is_dir() and path.name != "__pycache__"
                ),
            }
        )
    }

    module_origin = Path(str(identity.get("module_origin", ""))).resolve()
    if module_origin != private_root:
        raise RuntimeError(
            "Private CHA claimed module origin is not its installed package."
        )
    modules: dict[str, tuple[Any, tuple[Any, ...]]] = {}
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name == _ENGINE_MODULE or name.startswith(_ENGINE_MODULE + ".")
        ):
            continue
        filename = getattr(module, "__file__", None)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if (
            not filename
            or not origin
            or Path(filename).resolve() != Path(origin).resolve()
        ):
            raise RuntimeError(
                "Private CHA executing module origin is missing or inconsistent."
            )
        path = Path(filename).resolve()
        _require_within(path, private_root, label=f"Private CHA module {name}")
        if path not in private_owned:
            raise RuntimeError(
                "Private CHA executing module is not distribution-owned."
            )
        modules[name] = (module, _module_token(module))
    if _ENGINE_MODULE not in modules:
        raise RuntimeError(
            "Private CHA package module is not bound to the installation."
        )

    fixed_public_paths = {
        "maple_public_polar": (
            "maple.function.calculator.extra_correction.implicit.torch_chagb",
            "maple/function/calculator/extra_correction/implicit/torch_chagb.py",
        ),
        "maple_public_cavity": (
            "maple.function.calculator.extra_correction.implicit.torch_pbsa_exact_cavity",
            "maple/function/calculator/extra_correction/implicit/torch_pbsa_exact_cavity.py",
        ),
        "maple_public_dense_budget": (
            "maple.function.calculator.extra_correction.implicit.torch_dense_budget",
            "maple/function/calculator/extra_correction/implicit/torch_dense_budget.py",
        ),
    }
    for name, (module_name, relative) in fixed_public_paths.items():
        expected = release["maple_public_source"][name]
        if expected["module"] != module_name or expected["relative_path"] != relative:
            raise RuntimeError(
                "Private CHA pinned MAPLE module/path identity is invalid."
            )
        path = Path(maple_dist.locate_file(relative)).resolve()
        _require_within(path, prefix, label=f"MAPLE source {name}")
        files.append(
            _bind_file(
                path, expected["sha256"], maple_owned, label=f"MAPLE source {name}"
            )
        )
        module = importlib.import_module(module_name)
        filename = Path(str(getattr(module, "__file__", ""))).resolve()
        origin = Path(
            str(getattr(getattr(module, "__spec__", None), "origin", ""))
        ).resolve()
        if filename != path or origin != path:
            raise RuntimeError(
                "Executing MAPLE source is not the installed pinned module."
            )
        claimed = identity["maple_public_source"][name]
        if Path(str(claimed.get("origin", ""))).resolve() != path:
            raise RuntimeError("Private CHA claimed MAPLE origin is inconsistent.")
        modules[module_name] = (module, _module_token(module))

    binding = _RuntimeSourceBinding(
        files=tuple(files),
        directories=directories,
        modules=modules,
        anchors=_bind_anchors(Path(file.path) for file in files),
    )
    binding.require_unchanged()
    return dict(identity), binding


def _import_engine() -> Any:
    try:
        return importlib.import_module(_ENGINE_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            "The local-private CHA backend requires the separately installed "
            "maple-cha-exact-private==0.4.0 package; no AmberTools fallback is used."
        ) from exc


def _resolve_private_torch_device(device: Any) -> torch.device:
    """Return one canonical CPU/CUDA device without re-resolving canonical input."""
    if isinstance(device, torch.device):
        resolved = device
    elif device is None or isinstance(device, str):
        resolved = resolve_torch_device(device)
    else:
        raise TypeError("private-torch-cha device must be a torch.device, string, or None.")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(
            "private-torch-cha supports resolved CPU and CUDA devices only; "
            f"got {resolved}."
        )
    if resolved.type == "cuda" and resolved.index is None:
        raise ValueError("private-torch-cha requires a canonical indexed CUDA device.")
    if resolved.type == "cpu" and resolved.index is not None:
        raise ValueError("private-torch-cha requires the canonical CPU device.")
    return resolved


class PrivateTorchChaGB:
    """Prepared-topology Torch-FP32 single-point CHA/cavity/dispersion provider."""

    supported_properties = {"energy"}

    def __init__(
        self,
        atoms,
        charge_result,
        *,
        prepared_topology: str | Path,
        prepared_topology_sha256: str,
        device: Any = None,
        dtype: str | None = "float32",
    ):
        _molecular_charge_and_mult(atoms)
        self._initialize_loaded(
            prepared_topology=prepared_topology,
            prepared_topology_sha256=prepared_topology_sha256,
            device=device,
            dtype=dtype,
        )
        self._bind_source_identity(atoms, charge_result)

    def _initialize_loaded(
        self,
        *,
        prepared_topology: str | Path,
        prepared_topology_sha256: str,
        device: Any = None,
        dtype: str | None = "float32",
    ) -> None:
        selected_device = _resolve_private_torch_device(device)
        if str(dtype or "float32").lower() != "float32":
            raise ValueError("private-torch-cha requires dtype=float32.")
        release, release_file_identity = _load_release_contract()
        self._topology_path = Path(prepared_topology).expanduser().resolve()
        self._topology_file_sha256 = _validate_sha256(
            prepared_topology_sha256, name="prepared_topology_sha256"
        )
        topology_bytes = self._read_bound_topology_bytes()
        try:
            mapping = json.loads(topology_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("prepared_topology must be a UTF-8 JSON object.") from exc
        if not isinstance(mapping, dict):
            raise ValueError("prepared_topology must contain one JSON object.")
        content_sha256 = _validate_sha256(
            mapping.get("content_sha256", ""), name="prepared topology content_sha256"
        )

        engine = _import_engine()
        identity, runtime_binding = _bind_installed_runtime(engine, release)
        policy_type = getattr(engine, "ExecutionPolicy", None)
        if not callable(policy_type):
            raise RuntimeError("Private CHA API v4 ExecutionPolicy is unavailable.")
        execution_policy = policy_type()
        topology = engine.load_topology(
            mapping,
            expected_content_sha256=content_sha256,
            device=selected_device,
            execution_policy=execution_policy,
        )
        self._engine = engine
        self._execution_policy = execution_policy
        self._device = selected_device
        self._device_text = str(selected_device)
        self.platform = selected_device.type.upper()
        self._release = release
        self._release_file_identity = release_file_identity
        self._release_file_metadata = (
            None
            if release_file_identity is None
            else _file_metadata_token(Path(release_file_identity.path))
        )
        self._runtime_binding = runtime_binding
        self._engine_identity = identity
        self._topology = topology
        self._source_topology_content_sha256 = content_sha256
        self._topology_content_sha256 = _validate_sha256(
            _value(topology, "content_sha256"),
            name="private CHA FP32 topology content hash",
        )
        self._model_profile_sha256 = str(engine.MODEL_PROFILE_SHA256)
        self._topology_file_identity = CheckpointFileIdentity.bind(self._topology_path)
        if self._topology_file_identity.sha256 != self._topology_file_sha256:
            raise ValueError("prepared topology file changed while it was bound.")

    def _read_bound_topology_bytes(self) -> bytes:
        payload = _read_bounded_stable_file(
            self._topology_path,
            max_bytes=_MAX_PREPARED_TOPOLOGY_BYTES,
            label="prepared_topology JSON",
        )
        actual = _sha256_bytes(payload)
        if actual != self._topology_file_sha256:
            raise ValueError(
                "prepared topology file changed or does not match "
                "prepared_topology_sha256."
            )
        return payload

    def _bind_source_identity(self, atoms, charge_result) -> None:
        if str(getattr(charge_result, "mode", "fixed")).lower() != "fixed":
            raise ValueError("private-torch-cha requires one fixed ChargeResult.")
        metadata = atoms.info.get("mol2")
        if not isinstance(metadata, Mapping):
            raise ValueError("private-torch-cha requires a charged typed MOL2 input.")
        path_text = metadata.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError("private-torch-cha source MOL2 metadata path is invalid.")
        metadata_identity = mol2_identity_sha256(metadata)
        if metadata.get(MOL2_IDENTITY_SHA256_KEY) != metadata_identity:
            raise ValueError(
                "private-torch-cha source MOL2 metadata identity is invalid."
            )
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise ValueError("private-torch-cha source MOL2 path is missing.")
        source_sha256 = _hash_stable_file(path, label="source MOL2")
        expected_source_sha256 = str(_value(self._topology, "source_mol2_sha256"))
        if source_sha256 != expected_source_sha256:
            raise ValueError(
                "Prepared topology source MOL2 hash does not match the live MOL2."
            )

        atom_ids = tuple(int(value) for value in metadata.get("atom_ids", ()))
        atom_id_array = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
        if atom_id_array is None:
            raise ValueError("private-torch-cha requires the live MOL2 atom-ID array.")
        atom_id_array = np.asarray(atom_id_array)
        if (
            atom_id_array.shape != (len(atoms),)
            or not np.issubdtype(atom_id_array.dtype, np.integer)
            or len(np.unique(atom_id_array)) != len(atoms)
        ):
            raise ValueError("private-torch-cha live MOL2 atom-ID array is invalid.")
        live_atom_ids = tuple(int(value) for value in atom_id_array)
        if live_atom_ids != atom_ids:
            raise ValueError(
                "Live MOL2 atom-ID array does not match MOL2 metadata order."
            )
        names = tuple(str(value) for value in metadata.get("atom_names", ()))
        atom_types = tuple(str(value) for value in metadata.get("atom_types", ()))
        numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        expected = {
            "atom IDs": tuple(int(v) for v in _value(self._topology, "atom_ids")),
            "atom names": tuple(str(v) for v in _value(self._topology, "atom_names")),
            "GAFF types": tuple(str(v) for v in _value(self._topology, "gaff2_types")),
            "atomic numbers": tuple(
                int(v) for v in _value(self._topology, "atomic_numbers")
            ),
        }
        actual = {
            "atom IDs": atom_ids,
            "atom names": names,
            "GAFF types": atom_types,
            "atomic numbers": numbers,
        }
        for label in expected:
            if actual[label] != expected[label]:
                raise ValueError(f"Live MOL2 {label} do not match prepared topology.")

        source_charges = np.asarray(
            _value(self._topology, "source_charges_e"), dtype=np.float64
        )
        charges = np.asarray(charge_result.charges)
        if charges.shape != source_charges.shape or not np.array_equal(
            charges, source_charges
        ):
            raise ValueError(
                "ChargeResult charges do not exactly match prepared topology source charges."
            )
        effective_charges = np.asarray(
            _value(self._topology, "effective_charges_e"), dtype=np.float64
        )
        if effective_charges.shape != source_charges.shape or not np.all(
            np.isfinite(effective_charges)
        ):
            raise ValueError("Prepared topology effective charges are invalid.")
        effective_fp32 = np.asarray(effective_charges, dtype=np.float32, order="C")
        cast_delta = np.abs(effective_fp32.astype(np.float64) - effective_charges)

        self._source_mol2_path = path
        self._source_mol2_path_text = path_text
        self._source_mol2_metadata_identity = metadata_identity
        self._source_mol2_sha256 = source_sha256
        self._source_mol2_file_identity = CheckpointFileIdentity.bind(path)
        if self._source_mol2_file_identity.sha256 != source_sha256:
            raise ValueError("Bound source MOL2 changed while it was bound.")
        self._input_anchors = _bind_anchors(
            (self._topology_path, self._source_mol2_path)
        )
        self._input_file_metadata = {
            str(self._topology_path): _file_metadata_token(self._topology_path),
            str(self._source_mol2_path): _file_metadata_token(self._source_mol2_path),
        }
        self._frozen_atom_ids = atom_ids
        self._frozen_atom_id_array = live_atom_ids
        self._frozen_names = names
        self._frozen_types = atom_types
        self._frozen_numbers = numbers
        self._source_charges = source_charges.copy()
        self._source_charges_sha256 = str(
            _value(self._topology, "source_charges_sha256")
        )
        self._effective_charges_sha256 = str(
            _value(self._topology, "effective_charges_sha256")
        )
        self._effective_charges_fp32_sha256 = _sha256_bytes(effective_fp32.tobytes())
        self._effective_charges_fp32_max_cast_delta_e = float(
            cast_delta.max(initial=0.0)
        )
        compute_hashes = dict(_value(self._topology, "compute_tensor_sha256"))
        if (
            compute_hashes.get("effective_charges_e")
            != self._effective_charges_fp32_sha256
        ):
            raise ValueError(
                "Private CHA effective-charge FP32 tensor identity is inconsistent."
            )
        for name, digest in compute_hashes.items():
            _validate_sha256(digest, name=f"private CHA compute tensor hash for {name}")
        self._compute_tensor_sha256 = compute_hashes
        self._maximum_float32_cast_delta = float(
            _value(self._topology, "maximum_float32_cast_delta")
        )

    def _validate_live_identity(self, atoms) -> np.ndarray:
        _molecular_charge_and_mult(atoms)
        try:
            self._topology_file_identity.require_unchanged()
        except ValueError as exc:
            raise ValueError(
                "Bound prepared topology file changed after load."
            ) from exc
        try:
            self._source_mol2_file_identity.require_unchanged()
        except ValueError as exc:
            raise ValueError("Bound source MOL2 file changed after load.") from exc
        if self._release_file_identity is not None:
            self._release_file_identity.require_unchanged()
            if _file_metadata_token(Path(self._release_file_identity.path)) != (
                self._release_file_metadata
            ):
                raise RuntimeError(
                    "Private CHA release file metadata changed after load."
                )
        for path, expected in self._input_file_metadata.items():
            if _file_metadata_token(Path(path)) != expected:
                raise ValueError(
                    "Bound private CHA input file metadata changed after load."
                )
        _require_anchors(self._input_anchors)
        self._runtime_binding.require_unchanged()
        metadata = atoms.info.get("mol2")
        if not isinstance(metadata, Mapping):
            raise ValueError("Bound atom identity requires the original MOL2 metadata.")
        current_path_text = metadata.get("path")
        if (
            not isinstance(current_path_text, str)
            or current_path_text != self._source_mol2_path_text
            or Path(current_path_text).expanduser().resolve() != self._source_mol2_path
        ):
            raise ValueError("Bound source MOL2 metadata path changed after load.")
        current_metadata_identity = mol2_identity_sha256(metadata)
        if (
            metadata.get(MOL2_IDENTITY_SHA256_KEY) != current_metadata_identity
            or current_metadata_identity != self._source_mol2_metadata_identity
        ):
            raise ValueError("Bound source MOL2 metadata identity changed after load.")
        current = (
            tuple(int(v) for v in metadata.get("atom_ids", ())),
            tuple(str(v) for v in metadata.get("atom_names", ())),
            tuple(str(v) for v in metadata.get("atom_types", ())),
            tuple(int(v) for v in atoms.get_atomic_numbers()),
        )
        frozen = (
            self._frozen_atom_ids,
            self._frozen_names,
            self._frozen_types,
            self._frozen_numbers,
        )
        if current != frozen:
            raise ValueError("Bound atom identity or atom order changed.")
        atom_id_array = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
        if atom_id_array is None:
            raise ValueError("Bound MOL2 atom-ID array is missing.")
        atom_id_array = np.asarray(atom_id_array)
        if (
            atom_id_array.shape != (len(self._frozen_atom_id_array),)
            or not np.issubdtype(atom_id_array.dtype, np.integer)
            or len(np.unique(atom_id_array)) != len(atom_id_array)
            or tuple(int(value) for value in atom_id_array)
            != self._frozen_atom_id_array
        ):
            raise ValueError("Bound MOL2 atom-ID array or atom order changed.")
        charges = np.asarray(atoms.get_initial_charges())
        if charges.shape != self._source_charges.shape or not np.array_equal(
            charges, self._source_charges
        ):
            raise ValueError("Bound source charge vector changed.")
        positions = np.asarray(atoms.get_positions())
        if positions.shape != (len(self._source_charges), 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("Live coordinates must be a finite (N, 3) array.")
        if not np.issubdtype(positions.dtype, np.floating):
            raise ValueError("Live coordinates must use a floating dtype.")
        return np.array(positions, copy=True, order="C")

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "runtime_provider": "private-torch-cha",
            "engine": dict(self._engine_identity),
            "release_pin": {
                "release_id": self._release["release_id"],
                "content_sha256": self._release["content_sha256"],
                "approved_source_aggregate_sha256": self._release[
                    "approved_source_aggregate_sha256"
                ],
                "file_sha256": (
                    None
                    if self._release_file_identity is None
                    else self._release_file_identity.sha256
                ),
            },
            "arithmetic_profile": _ARITHMETIC_PROFILE,
            "model_profile_sha256": self._model_profile_sha256,
            "topology": {
                "path": str(self._topology_path),
                "file_sha256": self._topology_file_sha256,
                "content_sha256": self._topology_content_sha256,
                "source_content_sha256": self._source_topology_content_sha256,
                "source_mol2_sha256": self._source_mol2_sha256,
                "source_charges_sha256": self._source_charges_sha256,
                "effective_charges_sha256": self._effective_charges_sha256,
                "effective_charges_fp32_sha256": self._effective_charges_fp32_sha256,
                "effective_charges_fp32_max_cast_delta_e": (
                    self._effective_charges_fp32_max_cast_delta_e
                ),
                "effective_charges_fp32_serialization": (
                    "numpy-float32-c-order-native-bytes"
                ),
                "compute_tensor_sha256": dict(self._compute_tensor_sha256),
                "maximum_float32_cast_delta": self._maximum_float32_cast_delta,
            },
            "device": self._device_text,
            "platform": self.platform,
            "precision": "Torch binary32 core; Python floats are final serialization only",
            "supports_forces": False,
            "release_status": "local-private-license-blocked",
        }

    def evaluate(
        self, atoms, *, need_forces: bool = False, calculator=None
    ) -> SolvationResult:
        del calculator
        if need_forces:
            raise NotImplementedError("private-torch-cha is single-point energy-only.")
        positions = self._validate_live_identity(atoms)
        before = positions.copy()
        coordinates = self._engine.bind_coordinates(
            self._topology,
            positions,
            device=self._device,
            execution_policy=self._execution_policy,
        )
        if not np.array_equal(positions, before):
            raise RuntimeError("Private CHA coordinate binding mutated live positions.")
        coordinate_sha256 = _validate_sha256(
            _value(coordinates, "content_sha256"),
            name="private CHA live coordinate hash",
        )
        expected_tokens = tuple(
            f"{float(value):12.7f}" for value in positions.reshape(-1)
        )
        returned_tokens = tuple(
            str(value) for value in _value(coordinates, "coordinate_tokens_f12_7")
        )
        if returned_tokens != expected_tokens:
            raise RuntimeError(
                "Private CHA coordinate binding did not preserve fresh canonical F12.7 tokens."
            )
        coordinate_bytes_sha256 = _validate_sha256(
            _value(coordinates, "coordinate_bytes_sha256"),
            name="private CHA in-memory coordinate serialization hash",
        )
        if coordinate_bytes_sha256 != _sha256_bytes(
            "".join(expected_tokens).encode("ascii")
        ):
            raise RuntimeError(
                "Private CHA F12.7 coordinate byte hash is inconsistent."
            )
        if _value(coordinates, "topology_sha256") != self._topology_content_sha256:
            raise RuntimeError(
                "Private CHA coordinates are bound to the wrong topology."
            )
        coordinate_tensor = _value(coordinates, "positions_f32")
        if (
            str(getattr(coordinate_tensor, "dtype", "")) != "torch.float32"
            or getattr(coordinate_tensor, "device", None) != self._device
            or bool(getattr(coordinate_tensor, "requires_grad", True))
        ):
            raise RuntimeError(
                "Private CHA prepared coordinates are not sealed Torch FP32 on the selected device."
            )
        expected_coordinate_array = np.asarray(
            expected_tokens, dtype=np.float32
        ).reshape(positions.shape)
        raw_positions_sha256 = _sha256_bytes(positions.tobytes())
        raw64 = positions.astype(np.float64)
        token64 = np.asarray(expected_tokens, dtype=np.float64).reshape(positions.shape)
        direct32 = positions.astype(np.float32).astype(np.float64)
        returned32_as64 = expected_coordinate_array.astype(np.float64)
        f12_7_max_delta = float(np.max(np.abs(token64 - raw64), initial=0.0))
        token_to_returned_float32_max_delta = float(
            np.max(np.abs(returned32_as64 - token64), initial=0.0)
        )
        raw_to_returned_float32_max_delta = float(
            np.max(np.abs(returned32_as64 - raw64), initial=0.0)
        )
        direct_raw_to_float32_max_delta = float(
            np.max(np.abs(direct32 - raw64), initial=0.0)
        )
        result = self._engine.evaluate_exact_sp(
            self._topology,
            coordinates,
            execution_policy=self._execution_policy,
        )
        if bool(_value(result, "supports_forces")):
            raise RuntimeError("Private CHA engine must report supports_forces=False.")
        if _value(result, "profile_sha256") != self._model_profile_sha256:
            raise RuntimeError("Private CHA result has the wrong model profile hash.")
        if _value(result, "topology_sha256") != self._topology_content_sha256:
            raise RuntimeError("Private CHA result has the wrong topology identity.")
        if _value(result, "coordinate_sha256") != coordinate_sha256:
            raise RuntimeError(
                "Private CHA result has the wrong live-coordinate identity."
            )
        if str(_value(result, "device")) != self._device_text:
            raise RuntimeError("Private CHA result reports the wrong Torch device.")
        if _value(result, "platform") != self.platform:
            raise RuntimeError("Private CHA result reports the wrong execution platform.")
        if _value(result, "runtime") != "torch":
            raise RuntimeError("Private CHA result reports the wrong execution runtime.")
        if _value(result, "dtype") != "torch.float32":
            raise RuntimeError("Private CHA result reports a non-FP32 dtype.")
        workspace_bytes = _nonnegative_int(
            _value(result, "workspace_bytes"), name="workspace_bytes"
        )
        peak_allocated_bytes = _optional_nonnegative_int(
            _value(result, "peak_allocated_bytes"), name="peak_allocated_bytes"
        )
        peak_reserved_bytes = _optional_nonnegative_int(
            _value(result, "peak_reserved_bytes"), name="peak_reserved_bytes"
        )
        if self._device.type == "cpu":
            if peak_allocated_bytes is not None or peak_reserved_bytes is not None:
                raise RuntimeError(
                    "Private CHA CPU result must report null CUDA peak-memory fields."
                )
        elif peak_allocated_bytes is None or peak_reserved_bytes is None:
            raise RuntimeError(
                "Private CHA CUDA result requires exact peak-memory byte counts."
            )

        electrostatic_size = _serialized_fp32_scalar(
            result, "electrostatic_size_angstrom"
        )
        switch = _serialized_fp32_scalar(result, "electrostatic_size_switch_angstrom")
        if switch.tobytes() != np.float32(10.0).tobytes():
            raise RuntimeError(
                "Private CHA result has the wrong electrostatic size switch."
            )
        selected_branch = str(_value(result, "selected_size_branch"))
        expected_branch = (
            "below-switch"
            if bool(electrostatic_size < switch)
            else "at-or-above-switch"
        )
        if selected_branch != expected_branch:
            raise RuntimeError(
                "Private CHA result has an inconsistent size-switch branch."
            )
        result_content_sha256 = _validate_sha256(
            _value(result, "content_sha256"), name="private CHA result content hash"
        )
        polar = _serialized_fp32_scalar(result, "polar_kcal_mol")
        cavity = _serialized_fp32_scalar(result, "cavity_kcal_mol")
        dispersion = _serialized_fp32_scalar(result, "dispersion_kcal_mol")
        total = _serialized_fp32_scalar(result, "total_kcal_mol")
        closed_total = np.float32(np.float32(polar + cavity) + dispersion)
        if closed_total.tobytes() != total.tobytes():
            raise RuntimeError("Private CHA FP32 component closure failed.")
        conversion = np.float32(KCAL_PER_MOL_PER_HARTREE)
        polar_hartree = np.float32(polar / conversion)
        cavity_hartree = np.float32(cavity / conversion)
        dispersion_hartree = np.float32(dispersion / conversion)
        nonpolar_hartree = np.float32(cavity_hartree + dispersion_hartree)
        component_total_hartree = np.float32(polar_hartree + nonpolar_hartree)
        source_total_hartree = np.float32(total / conversion)
        scaling_delta_hartree = abs(
            float(component_total_hartree) - float(source_total_hartree)
        )
        unit_roundoff = float(np.finfo(np.float32).eps) / 2.0
        gamma_five = (5.0 * unit_roundoff) / (1.0 - 5.0 * unit_roundoff)
        scaling_bound_hartree = gamma_five * (
            abs(float(polar_hartree))
            + abs(float(cavity_hartree))
            + abs(float(dispersion_hartree))
            + abs(float(source_total_hartree))
        )
        if scaling_delta_hartree > scaling_bound_hartree:
            raise RuntimeError(
                "Private CHA FP32 kcal/mol-to-Hartree rounding is unbounded."
            )
        provenance = {
            **self.provenance,
            "coordinates": {
                "coordinate_sha256": coordinate_sha256,
                "coordinate_serialization_sha256": coordinate_bytes_sha256,
                "raw_positions_sha256": raw_positions_sha256,
                "raw_positions_dtype": positions.dtype.str,
                "raw_positions_shape": list(positions.shape),
                "f12_7_max_quantization_delta_angstrom": f12_7_max_delta,
                "f12_7_decimal_to_returned_float32_max_cast_delta_angstrom": (
                    token_to_returned_float32_max_delta
                ),
                "raw_to_returned_float32_max_delta_angstrom": (
                    raw_to_returned_float32_max_delta
                ),
                "direct_raw_to_float32_diagnostic_max_cast_delta_angstrom": (
                    direct_raw_to_float32_max_delta
                ),
                "serialization": "fresh in-memory F12.7",
            },
            "result_content_sha256": result_content_sha256,
            "runtime": {
                "device": self._device_text,
                "dtype": "torch.float32",
                "workspace_bytes": workspace_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
            },
            "readback_ledger": tuple(_value(result, "readback_records")),
            "readback_bytes": int(_value(result, "readback_bytes")),
            "private_result_source_sha256": tuple(_value(result, "source_sha256")),
            "electrostatic_size_angstrom": float(electrostatic_size),
            "electrostatic_size_switch_angstrom": float(switch),
            "selected_size_branch": selected_branch,
            "size_switch_scope": (
                "known nonsmooth-boundary diagnostic; no arbitrary-geometry parity claim"
            ),
            "component_closure": "exact Torch-FP32 kcal/mol before serialization",
            "reported_formula": "polar + cavity + dispersion",
            "serialized_kcal_mol": {
                "polar": float(polar),
                "cavity": float(cavity),
                "dispersion": float(dispersion),
                "total": float(total),
            },
            "hartree_fp32_sum_order": (
                "nonpolar=f32(cavity+dispersion); total=f32(polar+nonpolar)"
            ),
            "component_reconstructed_total_hartree": float(component_total_hartree),
            "source_total_scaled_hartree": float(source_total_hartree),
            "unit_conversion_rounding_delta_hartree": scaling_delta_hartree,
            "unit_conversion_rounding_bound_hartree": scaling_bound_hartree,
        }
        return SolvationResult(
            energy_hartree=float(source_total_hartree),
            components_hartree={
                "polar": float(polar_hartree),
                "nonpolar": float(nonpolar_hartree),
                "cavity": float(cavity_hartree),
                "dispersion": float(dispersion_hartree),
            },
            provenance=provenance,
        )


__all__ = ["KCAL_PER_MOL_PER_HARTREE", "PrivateTorchChaGB"]
