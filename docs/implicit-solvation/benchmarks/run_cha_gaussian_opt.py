#!/usr/bin/env python3
"""Pinned Gaussian-CHA validation and real MAPLE optimization runner.

The production path intentionally uses ``SetCalculator.set_calculator`` and
the unmodified MAPLE ``Optimization`` dispatcher.  Small dependency-injection
seams exist only so the transactional observer can be unit tested without a
model checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, NamedTuple, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

ROOT = Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[3]
V1_PROFILE_ID = "gaussian-cha-r6-v1"
V2_PROFILE_ID = (
    "gaussian-cha-r6-derivative-v2-numerical-profile-" "20261001.2-direct-complement"
)
KCAL_PER_HARTREE = 627.5094740631
FRESH_REPLAY_ENERGY_TOLERANCE_HARTREE = 1.0e-10
FRESH_REPLAY_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM = 1.0e-8
EXPECTED_MODEL_METADATA = {
    "ambient_default_dtype": "torch.float32",
    "parameter_count": 29,
    "parameter_dtypes": ["torch.float64"],
    "floating_buffer_count": 48,
    "floating_buffer_dtypes": ["torch.float64"],
    "int64_buffer_count": 2,
    "other_buffer_dtypes": [],
    "r_max_angstrom": 5.0,
}
BENCHMARK_SOURCE_MODULES = (
    "docs/implicit-solvation/benchmarks/cha_gaussian_reference.py",
    "docs/implicit-solvation/benchmarks/cha_continuum_reference.py",
    "docs/implicit-solvation/benchmarks/prepare_cha_continuum_water.py",
    "docs/implicit-solvation/benchmarks/validate_cha_continuum_branch_study.py",
    "docs/implicit-solvation/benchmarks/run_cha_gaussian_opt.py",
    "docs/implicit-solvation/benchmarks/cha_gaussian_opt_protocol.json",
)


class GaussianChaCalculatorFactory(NamedTuple):
    numerical_profile_id: str
    worktree: Path
    root: Path
    context_id: str

    def __call__(
        self, atoms: Atoms, topology: dict[str, Any], sigma_e: float, output_log: Path
    ) -> Calculator:
        return build_composed_calculator(
            atoms,
            topology,
            sigma_e,
            output_log,
            numerical_profile_id=self.numerical_profile_id,
            expected_worktree=self.worktree,
            root=self.root,
        )


class GaussianChaRunContext(NamedTuple):
    context_id: str
    worktree: Path
    root: Path
    evidence_root: Path
    output_root: Path
    protocol_template: Path
    protocol_template_sha256: str
    approved_plan: Path
    approved_plan_sha256: str
    approved_handoff: Path
    approved_handoff_sha256: str
    external_evidence_sha256: tuple[tuple[Path, str], ...]
    numerical_profile_id: str
    calculator_factory: GaussianChaCalculatorFactory


DEFAULT_V1_CONTEXT = GaussianChaRunContext(
    context_id="gaussian-cha-r6-v1-replay-in-v2-worktree",
    worktree=WORKTREE,
    root=ROOT,
    evidence_root=ROOT / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001",
    output_root=(
        ROOT
        / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001/v1-profile-replay"
    ),
    protocol_template=SCRIPT.with_name("cha_gaussian_opt_protocol.json"),
    protocol_template_sha256=(
        "d70486de0e0c492a2f7231bd122920b77478eda0e1abd35e1f3a491ef1d55a0f"
    ),
    approved_plan=ROOT / ".omx/plans/route1-cha-gaussian-opt-v1-20261001.md",
    approved_plan_sha256=(
        "74d4aac2a5275d38599985398e45bb1e8d79b0dfd1fc2e3ae5a33a9d8f401838"
    ),
    approved_handoff=ROOT / ".omx/plans/route1-cha-gaussian-opt-v1-handoff.json",
    approved_handoff_sha256=(
        "18db0fac5297fa945dc9243ac896e1939438fa367c969f3b85bb224927764571"
    ),
    external_evidence_sha256=(),
    numerical_profile_id=V1_PROFILE_ID,
    calculator_factory=GaussianChaCalculatorFactory(
        V1_PROFILE_ID, WORKTREE, ROOT, "gaussian-cha-r6-v1-replay-in-v2-worktree"
    ),
)

V2_RUN_CONTEXT = GaussianChaRunContext(
    context_id="gaussian-cha-r6-derivative-v2-campaign",
    worktree=WORKTREE,
    root=ROOT,
    evidence_root=ROOT / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001",
    output_root=(ROOT / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001"),
    protocol_template=SCRIPT.with_name("cha_gaussian_r6_v2_protocol.json"),
    protocol_template_sha256=(
        "de61f1c624a8a5bad70cfcbc181c01524b620f9c92696bd29ef82bae9d551fe8"
    ),
    approved_plan=ROOT / ".omx/plans/route1-cha-r6-derivative-repair-v2-20261001.md",
    approved_plan_sha256=(
        "caa2a31a0bb895f4ef143d7632a2de45ab7e7892acef50de968d10ecfcaa81d9"
    ),
    approved_handoff=ROOT
    / ".omx/plans/route1-cha-r6-derivative-repair-v2-handoff.json",
    approved_handoff_sha256=(
        "6e784f68dc2c912e405f301a9e48b756909fbcc40b244d2196524f0a58a6df7a"
    ),
    external_evidence_sha256=(
        (
            ROOT
            / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001/planning/cap-classifier-preregistration-v2.json",
            "94fc9258dbd39b80821c046098fce6db1fe2aed4b4797037e41bd40e60cbd665",
        ),
        (
            ROOT
            / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001/planning/CAP_CLASSIFIER_AND_CONDITIONING_V2.md",
            "3bf44a4f05f2284fa9db9da37198b2d6771b43d9878419baa410fe3808f07e86",
        ),
        (
            ROOT
            / ".omx/benchmarks/route1-cha-r6-derivative-repair-v2-20261001/planning/planning-validation-v2.json",
            "1b69569ad29c7788a65cf2fbe009442b97c69cd9e553b72041f510f1935dcbab",
        ),
        (
            ROOT
            / ".omx/benchmarks/route1-cha-gaussian-opt-v1-20261001/FAILED_V1_FINAL_EVIDENCE_MANIFEST.json",
            "db9d862e3438cacf884ee9654aa0f33ab7cc58cab067462a3445e3f82be400f8",
        ),
        (
            ROOT
            / ".omx/benchmarks/route1-cha-gaussian-opt-v1-20261001/campaign-v2-20261001T110415Z/failed-v1-candidate-source.tar.gz",
            "2b1f93af9c10242d75c3382d7f3ed717c716d5945d122a8ba103b3d3cbfcf982",
        ),
    ),
    numerical_profile_id=V2_PROFILE_ID,
    calculator_factory=GaussianChaCalculatorFactory(
        V2_PROFILE_ID, WORKTREE, ROOT, "gaussian-cha-r6-derivative-v2-campaign"
    ),
)

# Legacy names remain read-only aliases for tests and importers; execution uses context.
EXPECTED_WORKTREE = DEFAULT_V1_CONTEXT.worktree
OUTPUT_ROOT = DEFAULT_V1_CONTEXT.output_root
PROTOCOL_TEMPLATE = DEFAULT_V1_CONTEXT.protocol_template
PROTOCOL_TEMPLATE_SHA256 = DEFAULT_V1_CONTEXT.protocol_template_sha256
PLAN = DEFAULT_V1_CONTEXT.approved_plan
PLAN_SHA256 = DEFAULT_V1_CONTEXT.approved_plan_sha256


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingProxyType):
        return {key: json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _error_safe_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"tag": "NONFINITE_FLOAT", "value": "NaN"}
        if math.isinf(number):
            return {
                "tag": "NONFINITE_FLOAT",
                "value": "+Infinity" if number > 0.0 else "-Infinity",
            }
        return number
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, dict):
        return {str(key): _error_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_error_safe_value(item) for item in value]
    return str(value)


def _serialize_exception_details(exc: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {}
    reason = getattr(exc, "reason", None)
    if reason is not None:
        details["reason"] = str(getattr(reason, "value", reason))
    raw_margins = getattr(exc, "raw_margins", None)
    if raw_margins is not None:
        details["raw_margins"] = {
            str(key): _error_safe_value(value) for key, value in raw_margins
        }
    for name in (
        "owner_index",
        "receiver_index",
        "cap_index",
        "node_index",
        "order",
    ):
        if hasattr(exc, name):
            details[name] = _error_safe_value(getattr(exc, name))
    return details


def serialize_exception(exc: Exception) -> dict[str, Any]:
    """Serialize an error without allowing diagnostics to mask the original."""

    try:
        message = str(exc)
    except Exception as message_exc:
        message = f"<message unavailable: {type(message_exc).__name__}>"
    result: dict[str, Any] = {"type": type(exc).__name__, "message": message}
    try:
        result.update(_serialize_exception_details(exc))
    except Exception as serializer_exc:
        result["serializer_error"] = {
            "type": type(serializer_exc).__name__,
            "message": str(serializer_exc),
        }
    return result


def read_hashed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def read_externally_pinned_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError(f"{label} external SHA256 pin differs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def content_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    )


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = content_sha256(result)
    return result


def verify_seal(value: dict[str, Any], label: str) -> None:
    digest = value.get("content_sha256")
    if not isinstance(digest, str) or digest != content_sha256(value):
        raise ValueError(f"{label} content seal differs")


def write_json_new_atomic(path: Path, value: Any) -> None:
    """Atomically create *path* without replacing any existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def coordinate_sha256(positions: Any) -> str:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("coordinates must be finite float64 shape (N,3)")
    return sha256_bytes(array.astype("<f8", copy=False).tobytes(order="C"))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_context(context: GaussianChaRunContext) -> None:
    factory = context.calculator_factory
    if (
        factory.numerical_profile_id != context.numerical_profile_id
        or factory.worktree != context.worktree
        or factory.root != context.root
        or factory.context_id != context.context_id
    ):
        raise ValueError("calculator factory ownership differs from execution context")


def _assert_runtime_origin_paths(
    expected_worktree: Path, root: Path, *, require_cwd: bool
) -> dict[str, str]:
    if WORKTREE != expected_worktree or not _inside(SCRIPT, expected_worktree):
        raise RuntimeError("runner is not executing from the pinned candidate worktree")
    if require_cwd and Path.cwd().resolve() != expected_worktree:
        raise RuntimeError("cwd must be the pinned candidate worktree")
    candidate = str(expected_worktree)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    if candidate in sys.path:
        sys.path.remove(candidate)
    sys.path.insert(0, candidate)
    import maple
    from maple.function.calculator.set_calculator import SetCalculator

    origins = {
        "cwd": str(Path.cwd().resolve()),
        "maple": str(Path(maple.__file__).resolve()),
        "SetCalculator": str(
            Path(inspect.getsourcefile(SetCalculator) or "").resolve()
        ),
    }
    for name, path in origins.items():
        if name != "cwd" and not _inside(Path(path), expected_worktree):
            raise RuntimeError(
                f"{name} resolved outside the candidate worktree: {path}"
            )
    return origins


def assert_runtime_origins(
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT, *, require_cwd: bool = True
) -> dict[str, str]:
    _validate_context(context)
    return _assert_runtime_origin_paths(
        context.worktree, context.root, require_cwd=require_cwd
    )


def load_protocol_template(
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> tuple[dict[str, Any], str]:
    protocol, digest = read_hashed_json(context.protocol_template, "protocol template")
    if digest != context.protocol_template_sha256:
        raise ValueError("external protocol template SHA256 changed")
    if "base_protocol" in protocol:
        wrapper_keys = {
            "schema_version",
            "protocol_id",
            "numerical_profile_id",
            "base_protocol",
            "base_protocol_sha256",
            "approved_plan_sha256",
            "handoff_sha256",
            "planning_artifacts",
            "scientific_identity",
        }
        if set(protocol) != wrapper_keys:
            raise ValueError("v2 wrapper protocol keys differ from the closed schema")
        if protocol.get("numerical_profile_id") != context.numerical_profile_id:
            raise ValueError("wrapper protocol numerical profile differs from context")
        base_path = (context.worktree / str(protocol["base_protocol"])).resolve()
        if not _inside(base_path, context.worktree):
            raise ValueError("base protocol escapes the candidate worktree")
        base, base_digest = read_hashed_json(base_path, "base protocol")
        if base_digest != protocol.get("base_protocol_sha256"):
            raise ValueError("base protocol SHA256 changed")
        if protocol.get("approved_plan_sha256") != context.approved_plan_sha256:
            raise ValueError("wrapper protocol plan pin differs from context")
        if protocol.get("handoff_sha256") != context.approved_handoff_sha256:
            raise ValueError("wrapper protocol handoff pin differs from context")
        planning = protocol.get("planning_artifacts")
        if planning != {
            "cap_classifier_preregistration_sha256": (
                "94fc9258dbd39b80821c046098fce6db1fe2aed4b4797037e41bd40e60cbd665"
            ),
            "cap_classifier_markdown_sha256": (
                "3bf44a4f05f2284fa9db9da37198b2d6771b43d9878419baa410fe3808f07e86"
            ),
        }:
            raise ValueError("v2 wrapper planning-artifact pins differ")
        base_protocol_id = base.get("protocol_id")
        wrapper_protocol_id = protocol.get("protocol_id")
        protocol = dict(base)
        protocol["base_protocol_id"] = base_protocol_id
        protocol["protocol_id"] = wrapper_protocol_id
        protocol["execution"] = dict(base["execution"])
        protocol["execution"][
            "campaign_output"
        ] = "unique direct child campaign-v2-YYYYMMDDTHHMMSSZ"
        protocol["numerical_profile_id"] = context.numerical_profile_id
        protocol["context_protocol_sha256"] = digest
        return protocol, digest
    allowed = {
        "schema_version",
        "protocol_id",
        "model_identity",
        "claim_boundary",
        "inputs",
        "precision",
        "scalar",
        "crossing_validation",
        "combined_fd_validation",
        "optimizer",
        "execution",
        "success_rule",
        "citation",
    }
    unknown = set(protocol) - allowed
    if unknown or set(protocol) != allowed:
        raise ValueError(f"protocol keys differ from frozen schema: {sorted(unknown)}")
    return protocol, digest


def _root_path(
    relative: str, context: GaussianChaRunContext = DEFAULT_V1_CONTEXT
) -> Path:
    path = (context.root / relative).resolve()
    if not _inside(path, context.root):
        raise ValueError(f"input path escapes root: {relative}")
    return path


def _read_once(path: Path, cache: dict[Path, bytes]) -> bytes:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = resolved.read_bytes()
    return cache[resolved]


def _source_paths(
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> list[Path]:
    paths = list((context.worktree / "maple").rglob("*.py"))
    paths.extend(context.worktree / relative for relative in BENCHMARK_SOURCE_MODULES)
    paths.extend(
        (
            context.protocol_template,
            SCRIPT.with_name("audit_cha_gaussian_terminal.py"),
            SCRIPT.with_name("run_cha_gaussian_r6_v2.py"),
        )
    )
    return sorted({path.resolve() for path in paths})


def _source_snapshot(
    cache: dict[Path, bytes] | None = None,
    *,
    verified_protocol_sha256: str | None = None,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, str]:
    byte_cache = {} if cache is None else cache
    snapshot = {}
    protocol_sha256 = (
        context.protocol_template_sha256
        if verified_protocol_sha256 is None
        else verified_protocol_sha256
    )
    for path in _source_paths(context):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = str(path.relative_to(context.worktree))
        snapshot[relative] = (
            protocol_sha256
            if path == context.protocol_template.resolve()
            else sha256_bytes(_read_once(path, byte_cache))
        )
    return snapshot


def _input_snapshot(
    protocol: dict[str, Any],
    cache: dict[Path, bytes] | None = None,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, str]:
    byte_cache = {} if cache is None else cache
    inputs = protocol["inputs"]
    paths = {
        "center_source": _root_path(inputs["center_source"], context),
        "topology": _root_path(inputs["topology"], context),
        "preparation_record": _root_path(inputs["preparation_record"], context),
        "gas_preflight": _root_path(inputs["gas_preflight"], context),
        "checkpoint": context.worktree / inputs["checkpoint"],
    }
    return {
        name: sha256_bytes(_read_once(path, byte_cache)) for name, path in paths.items()
    }


def _protected_snapshot(
    protocol: dict[str, Any],
    cache: dict[Path, bytes] | None = None,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, str]:
    byte_cache = {} if cache is None else cache
    inputs = protocol["inputs"]
    manifest_path = _root_path(inputs["protected_manifest"], context)
    manifest_raw = _read_once(manifest_path, byte_cache)
    digest = sha256_bytes(manifest_raw)
    manifest = json.loads(manifest_raw)
    if digest != inputs["protected_manifest_sha256"]:
        raise ValueError("protected source manifest SHA256 changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != inputs["protected_file_count"]:
        raise ValueError("protected source manifest does not enumerate 24 files")
    observed = {}
    for relative, expected in sorted(files.items()):
        path = context.worktree / relative
        actual = sha256_bytes(_read_once(path, byte_cache))
        if actual != expected:
            raise ValueError(f"protected source changed: {relative}")
        observed[relative] = actual
    return observed


def build_row_inventory(center_source: dict[str, Any], protocol: dict[str, Any]):
    rows = center_source.get("rows")
    expected_scales = protocol["inputs"]["center_scales"]
    widths = protocol["inputs"]["sigma_e"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("center source must contain five rows")
    inventory = []
    for center_index, (row, expected_scale) in enumerate(zip(rows, expected_scales)):
        if float(row["scale"]) != float(expected_scale):
            raise ValueError("center ordering or scale changed")
        positions = np.asarray(row["positions_angstrom"], dtype=np.float64)
        if positions.shape != (3, 3) or not np.isfinite(positions).all():
            raise ValueError("every center must contain three finite XYZ rows")
        external_event = row.get("row_status") == "EVENT"
        if external_event != (float(expected_scale) == 1.0):
            raise ValueError("frozen external EVENT annotations changed")
        for width_index, sigma_e in enumerate(widths):
            inventory.append(
                {
                    "row_id": f"center-{center_index:02d}-sigma-{width_index:02d}",
                    "center_index": center_index,
                    "center_scale": float(expected_scale),
                    "sigma_e": float(sigma_e),
                    "positions_angstrom": positions.tolist(),
                    "positions_sha256": coordinate_sha256(positions),
                    "source_row_status": row["row_status"],
                    "source_external_event_annotation": external_event,
                }
            )
    if len(inventory) != 15:
        raise AssertionError("frozen matrix must contain 15 rows")
    return inventory


def _phase_evidence(
    protocol: dict[str, Any],
    verified_protocol_sha256: str | None = None,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, Any]:
    cache: dict[Path, bytes] = {}
    inputs = protocol["inputs"]
    center_raw = _read_once(_root_path(inputs["center_source"], context), cache)
    topology_raw = _read_once(_root_path(inputs["topology"], context), cache)
    preparation_raw = _read_once(
        _root_path(inputs["preparation_record"], context), cache
    )
    gas_preflight_raw = _read_once(_root_path(inputs["gas_preflight"], context), cache)
    plan_raw = _read_once(context.approved_plan, cache)
    handoff_raw = _read_once(context.approved_handoff, cache)
    center_digest = sha256_bytes(center_raw)
    topology_digest = sha256_bytes(topology_raw)
    preparation_digest = sha256_bytes(preparation_raw)
    gas_preflight_digest = sha256_bytes(gas_preflight_raw)
    if center_digest != inputs["center_source_sha256"]:
        raise ValueError("center source SHA256 changed")
    if topology_digest != inputs["topology_file_sha256"]:
        raise ValueError("topology file SHA256 changed")
    if preparation_digest != inputs["preparation_record_sha256"]:
        raise ValueError("preparation record SHA256 changed")
    if gas_preflight_digest != inputs["gas_preflight_sha256"]:
        raise ValueError("gas preflight SHA256 changed")
    if sha256_bytes(plan_raw) != context.approved_plan_sha256:
        raise ValueError("approved plan SHA256 changed")
    if sha256_bytes(handoff_raw) != context.approved_handoff_sha256:
        raise ValueError("approved handoff SHA256 changed")
    external_evidence = {}
    for path, expected in context.external_evidence_sha256:
        observed = sha256_bytes(_read_once(path, cache))
        if observed != expected:
            raise ValueError(f"external evidence SHA256 changed: {path}")
        external_evidence[str(path)] = observed
    center = json.loads(center_raw)
    topology = json.loads(topology_raw)
    preparation = json.loads(preparation_raw)
    gas_preflight = json.loads(gas_preflight_raw)
    if not all(
        isinstance(value, dict)
        for value in (center, topology, preparation, gas_preflight)
    ):
        raise ValueError("frozen inputs must be JSON objects")
    if topology.get("content_sha256") != inputs["topology_content_sha256"]:
        raise ValueError("topology content identity changed")
    input_sha256 = _input_snapshot(protocol, cache, context)
    if input_sha256["checkpoint"] != inputs["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 changed")
    if preparation.get("topology_content_sha256") not in (
        None,
        inputs["topology_content_sha256"],
    ):
        raise ValueError("preparation record topology identity changed")
    for key in (
        "checkpoint_sha256",
        "parameters",
        "floating_buffers",
        "int64_buffers",
        "default_dtype",
        "calculator_dtype",
        "rmax_angstrom",
    ):
        expected = {
            "checkpoint_sha256": inputs["checkpoint_sha256"],
            "parameters": 29,
            "floating_buffers": 48,
            "int64_buffers": 2,
            "default_dtype": "torch.float32",
            "calculator_dtype": "torch.float64",
            "rmax_angstrom": 5.0,
        }[key]
        if gas_preflight.get(key) != expected:
            raise ValueError(f"gas preflight mismatch: {key}")
    if gas_preflight.get("network_and_subprocess_disabled") is not True:
        raise ValueError("gas preflight did not disable network/subprocess")
    source = _source_snapshot(
        cache,
        verified_protocol_sha256=verified_protocol_sha256,
        context=context,
    )
    protected = _protected_snapshot(protocol, cache, context)
    identity_payload = {
        "context_id": context.context_id,
        "external_evidence_sha256": external_evidence,
        "protocol_template_sha256": context.protocol_template_sha256,
        "approved_plan_sha256": context.approved_plan_sha256,
        "approved_handoff_sha256": context.approved_handoff_sha256,
        "protocol_id": protocol["protocol_id"],
        "numerical_profile_id": context.numerical_profile_id,
        "input_sha256": input_sha256,
        "source": source,
        "protected": protected,
    }
    return {
        "center": center,
        "topology": topology,
        "input_sha256": input_sha256,
        "source": source,
        "protected": protected,
        "source_identity_sha256": sha256_bytes(canonical_bytes(identity_payload)),
        "numerical_profile_id": context.numerical_profile_id,
        "context_id": context.context_id,
        "external_evidence_sha256": external_evidence,
        "protocol_template_sha256": context.protocol_template_sha256,
        "approved_plan_sha256": context.approved_plan_sha256,
        "approved_handoff_sha256": context.approved_handoff_sha256,
        "protocol_id": protocol["protocol_id"],
    }


def preregister(
    output: Path, context: GaussianChaRunContext = DEFAULT_V1_CONTEXT
) -> dict[str, Any]:
    origins = assert_runtime_origins(context)
    protocol, protocol_digest = load_protocol_template(context)
    evidence = _phase_evidence(protocol, protocol_digest, context)
    record = seal(
        {
            "schema_version": 1,
            "phase": "PREREGISTERED",
            "protocol_template_sha256": protocol_digest,
            "approved_plan_sha256": context.approved_plan_sha256,
            "approved_handoff_sha256": context.approved_handoff_sha256,
            "runtime_origins": origins,
            "input_sha256": evidence["input_sha256"],
            "source_before": evidence["source"],
            "protected_before": evidence["protected"],
            "source_identity_sha256": evidence["source_identity_sha256"],
            "external_evidence_sha256": evidence["external_evidence_sha256"],
            "numerical_profile_id": evidence["numerical_profile_id"],
            "context_id": evidence["context_id"],
            "protocol_id": evidence["protocol_id"],
            "rows": build_row_inventory(evidence["center"], protocol),
            "prohibitions": dict(protocol["execution"]),
        }
    )
    write_json_new_atomic(output / "preregistration.json", record)
    return record


def load_preregistration(
    output: Path,
    expected_preregistration_sha256: str,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, protocol_digest = load_protocol_template(context)
    prereg, observed_preregistration_sha256 = read_externally_pinned_json(
        output / "preregistration.json",
        expected_preregistration_sha256,
        "preregistration",
    )
    verify_seal(prereg, "preregistration")
    if prereg.get("phase") != "PREREGISTERED":
        raise ValueError("invalid preregistration phase")
    if prereg.get("protocol_template_sha256") != protocol_digest:
        raise ValueError("preregistration protocol pin differs")
    if prereg.get("approved_plan_sha256") != context.approved_plan_sha256:
        raise ValueError("preregistration plan pin differs")
    if prereg.get("approved_handoff_sha256") != context.approved_handoff_sha256:
        raise ValueError("preregistration handoff pin differs")
    if prereg.get("context_id") != context.context_id:
        raise ValueError("preregistration execution context differs")
    if prereg.get("numerical_profile_id") != context.numerical_profile_id:
        raise ValueError("preregistration numerical profile differs")
    if prereg.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("preregistration protocol ID differs")
    evidence = _phase_evidence(protocol, protocol_digest, context)
    if prereg.get("source_before") != evidence["source"]:
        raise ValueError("candidate source changed after preregistration")
    if prereg.get("protected_before") != evidence["protected"]:
        raise ValueError("protected source changed after preregistration")
    if prereg.get("input_sha256") != evidence["input_sha256"]:
        raise ValueError("frozen input changed after preregistration")
    if prereg.get("source_identity_sha256") != evidence["source_identity_sha256"]:
        raise ValueError("source identity differs from preregistration")
    if prereg.get("external_evidence_sha256") != evidence["external_evidence_sha256"]:
        raise ValueError("external evidence differs from preregistration")
    reconstructed = build_row_inventory(evidence["center"], protocol)
    if canonical_bytes(prereg.get("rows")) != canonical_bytes(reconstructed):
        raise ValueError("preregistered row roster differs from frozen source")
    evidence["preregistration_sha256"] = observed_preregistration_sha256
    return protocol, prereg, evidence


class ObserverCaps:
    def __init__(self, calls: int = 512, unique_force_coordinates: int = 258):
        self.calls = int(calls)
        self.unique_force_coordinates = int(unique_force_coordinates)


class ObservationBudgetExceeded(RuntimeError):
    pass


def _extract_composed_frame(atoms: Atoms, results: dict[str, Any]):
    try:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
        forces = np.asarray(results["forces"], dtype=np.float64).copy()
        energy = float(results["energy"])
        solvation = results["solvation"]
        if not isinstance(solvation, dict):
            raise TypeError("solvation ledger must be a dictionary")
        components = solvation["components_hartree"]
        if not isinstance(components, dict) or set(components) != {
            "polar",
            "cavity",
            "dispersion",
        }:
            raise ValueError("solvation components are incomplete")
        solvent_energy = float(solvation["energy_hartree"])
        gas_energy = float(solvation["gas_energy_hartree"])
        combined_energy = float(solvation["combined_energy_hartree"])
        energy_values = [
            energy,
            solvent_energy,
            gas_energy,
            combined_energy,
            *(float(value) for value in components.values()),
        ]
        valid = bool(
            positions.shape == (3, 3)
            and forces.shape == (3, 3)
            and np.isfinite(positions).all()
            and np.isfinite(forces).all()
            and np.isfinite(energy_values).all()
            and abs(combined_energy - energy) <= 1.0e-12
            and abs(gas_energy + solvent_energy - combined_energy) <= 1.0e-10
            and abs(sum(float(value) for value in components.values()) - solvent_energy)
            <= 1.0e-10
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("composed frame ledger is invalid") from exc
    if not valid:
        raise FloatingPointError(
            "successful composed frame must be finite and consistent"
        )
    return positions, forces, energy


class ComposedRecordingCalculator(Calculator):
    """Transparent ASE wrapper recording only successful composed force calls."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        composed: Calculator,
        *,
        caps: ObserverCaps,
        expected_solvation_profile: tuple[str, str, float, int] | None = None,
    ):
        super().__init__()
        self.composed = composed
        self.caps = caps
        self.call_count = 0
        self.frames: list[dict[str, Any]] = []
        self._force_hashes: set[str] = set()
        self.expected_solvation_profile = expected_solvation_profile

    @property
    def unique_force_coordinate_count(self) -> int:
        return len(self._force_hashes)

    @property
    def last_force_positions(self) -> np.ndarray | None:
        if not self.frames:
            return None
        return np.asarray(self.frames[-1]["positions_angstrom"], dtype=np.float64)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms is None:
            raise ValueError("observer requires explicit Atoms")
        self.call_count += 1
        if self.call_count > self.caps.calls:
            raise ObservationBudgetExceeded("observer calculator-call cap exceeded")
        super().calculate(atoms, properties, system_changes)
        try:
            self.composed.calculate(atoms, properties, system_changes)
        except Exception:
            self.results = {}
            self.atoms = None
            raise
        copied = {}
        for key, value in self.composed.results.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.results = copied
        if "forces" not in self.results:
            return
        try:
            positions, forces, energy = _extract_composed_frame(atoms, self.results)
            if self.expected_solvation_profile is not None:
                profile_id, topology_sha256, sigma_e, order = (
                    self.expected_solvation_profile
                )
                _validate_solvation_profile(
                    self.results["solvation"],
                    profile_id,
                    topology_sha256,
                    sigma_e,
                    order,
                )
        except FloatingPointError:
            self.results = {}
            self.atoms = None
            raise
        except ValueError:
            self.results = {}
            self.atoms = None
            raise
        digest = coordinate_sha256(positions)
        prospective = len(self._force_hashes | {digest})
        if prospective > self.caps.unique_force_coordinates:
            self.results = {}
            self.atoms = None
            raise ObservationBudgetExceeded("unique force-coordinate cap exceeded")
        self._force_hashes.add(digest)
        self.frames.append(
            {
                "frame_index": len(self.frames),
                "calculator_call": self.call_count,
                "coordinate_sha256": digest,
                "positions_angstrom": positions.tolist(),
                "combined_energy_hartree": energy,
                "combined_forces_hartree_per_angstrom": forces.tolist(),
                "solvation": json_safe(self.results.get("solvation")),
            }
        )


def _set_thresholds(atoms: Atoms, thresholds: dict[str, float]) -> None:
    setattr(atoms, "f_max_th", thresholds["f_max_hartree_per_angstrom"])
    setattr(atoms, "f_rms_th", thresholds["f_rms_hartree_per_angstrom"])
    setattr(atoms, "dp_max_th", thresholds["dp_max_angstrom"])
    setattr(atoms, "dp_rms_th", thresholds["dp_rms_angstrom"])


def convergence_metrics(atoms: Atoms, forces: np.ndarray | None = None):
    if forces is None:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return {
        "f_max_hartree_per_angstrom": float(np.max(np.abs(forces))),
        "f_rms_hartree_per_angstrom": float(np.sqrt(np.mean(np.square(forces)))),
        "dp_max_angstrom": (
            None
            if getattr(atoms, "max_dp", None) is None
            else float(getattr(atoms, "max_dp"))
        ),
        "dp_rms_angstrom": (
            None
            if getattr(atoms, "rms_dp", None) is None
            else float(getattr(atoms, "rms_dp"))
        ),
    }


def metrics_pass(
    metrics: dict[str, float | None], thresholds: dict[str, float]
) -> bool:
    for key, limit in thresholds.items():
        value = metrics.get(key)
        if value is None or not math.isfinite(value) or value < 0.0 or value > limit:
            return False
    return True


def _import_owned(name: str, context: GaussianChaRunContext = DEFAULT_V1_CONTEXT):
    return _import_owned_from(name, context.worktree)


def _import_owned_from(name: str, expected_worktree: Path):
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"{name} has no filesystem origin")
    origin = Path(module_file).resolve()
    if not _inside(origin, expected_worktree):
        raise RuntimeError(f"{name} resolved outside candidate worktree")
    return module


def _typed_topology(topology: dict[str, Any]):
    inputs = importlib.import_module(
        "maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs"
    )
    return inputs.ContinuumChaTopology.from_mapping(
        topology, expected_content_sha256=topology["content_sha256"]
    )


def _is_domain_exception(exc: Exception) -> bool:
    from maple.function.calculator.extra_correction.implicit.torch_chagb_gaussian import (
        GaussianChaSizeDomainError,
    )
    from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb import (
        ContinuumChaDomainError,
    )
    from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_contact_v2 import (
        ContactV2DomainError,
    )

    return isinstance(
        exc,
        (ContinuumChaDomainError, GaussianChaSizeDomainError, ContactV2DomainError),
    )


def _assert_model_metadata(calculator: Calculator) -> dict[str, Any]:
    import torch

    concrete = cast(Any, calculator)
    model = concrete.model
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    floating_buffers = [value for value in buffers if value.is_floating_point()]
    int64_buffers = [value for value in buffers if value.dtype == torch.int64]
    metadata = {
        "ambient_default_dtype": str(torch.get_default_dtype()),
        "parameter_count": len(parameters),
        "parameter_dtypes": sorted({str(value.dtype) for value in parameters}),
        "floating_buffer_count": len(floating_buffers),
        "floating_buffer_dtypes": sorted(
            {str(value.dtype) for value in floating_buffers}
        ),
        "int64_buffer_count": len(int64_buffers),
        "other_buffer_dtypes": sorted(
            {
                str(value.dtype)
                for value in buffers
                if not value.is_floating_point() and value.dtype != torch.int64
            }
        ),
        "r_max_angstrom": float(concrete.r_max),
    }
    if metadata != EXPECTED_MODEL_METADATA:
        raise ValueError(f"loaded MACE precision identity differs: {metadata}")
    return metadata


def build_composed_calculator(
    atoms: Atoms,
    topology: dict[str, Any],
    sigma_e: float,
    output_log: Path,
    *,
    numerical_profile_id: str = V1_PROFILE_ID,
    expected_worktree: Path = WORKTREE,
    root: Path = ROOT,
) -> Calculator:
    from maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles import (
        resolve_gaussian_cha_profile,
    )

    numerical_profile = resolve_gaussian_cha_profile(numerical_profile_id)
    _assert_runtime_origin_paths(expected_worktree, root, require_cwd=True)
    os.environ["MAPLE_OFFLINE"] = "1"
    from maple.function.calculator.set_calculator import SetCalculator

    correction_module = _import_owned_from(
        "maple.function.calculator.extra_correction.implicit.gaussian_cha_correction",
        expected_worktree,
    )
    checkpoint = expected_worktree / "maple/function/calculator/model/maceoff23m.pt"
    if sha256_file(checkpoint) != (
        "ac172fdf9b5173fef4c64667739dbd06f230b0167b4ae2c67e8c08033254c9ee"
    ):
        raise ValueError("gas checkpoint SHA256 changed before model load")
    mace_source = (
        expected_worktree / "maple/function/calculator/mace/_mace_calculator.py"
    )
    if sha256_file(mace_source) != (
        "3423624cd58ba41abf567efe38a38386e9d4b062b2b3e8973f60aa15c534b73e"
    ):
        raise ValueError("MACE calculator source SHA256 changed before model load")
    calculator = SetCalculator(
        device="cpu",
        model="maceoff23m",
        output=str(output_log),
        atoms=atoms,
        implicit="None",
        solvent="None",
        model_options={"model_path": str(checkpoint)},
    ).set_calculator()
    calculator.gaussian_opt_model_metadata = _assert_model_metadata(calculator)
    correction = correction_module.GaussianChaCorrection(
        atoms,
        _typed_topology(topology),
        expected_topology_sha256=topology["content_sha256"],
        sigma_e=sigma_e,
        order=64,
        numerical_profile_id=numerical_profile.profile_id,
    )
    if correction.numerical_profile_id != numerical_profile.profile_id:
        raise ValueError("constructed correction numerical profile differs")
    calculator.solvent_correction = correction
    return calculator


def _fresh_final(
    atoms: Atoms,
    topology: dict[str, Any],
    sigma_e: float,
    calculator_factory: Callable[..., Calculator],
    log: Path,
) -> dict[str, Any]:
    fresh_atoms = atoms.copy()
    calculator = calculator_factory(fresh_atoms, topology, sigma_e, log)
    fresh_atoms.calc = calculator
    forces = np.asarray(fresh_atoms.get_forces(), dtype=np.float64)
    energy = float(fresh_atoms.get_potential_energy(force_consistent=True))
    solvation = json_safe(calculator.results.get("solvation"))
    factory_profile_id = getattr(calculator_factory, "numerical_profile_id", None)
    if isinstance(factory_profile_id, str):
        _validate_solvation_profile(
            solvation,
            factory_profile_id,
            topology["content_sha256"],
            sigma_e,
            64,
        )
    return {
        "tag": "FRESH_FINAL_UNCACHED",
        "positions_angstrom": fresh_atoms.get_positions().tolist(),
        "coordinate_sha256": coordinate_sha256(fresh_atoms.get_positions()),
        "combined_energy_hartree": energy,
        "combined_forces_hartree_per_angstrom": forces.tolist(),
        "f_max_hartree_per_angstrom": float(np.max(np.abs(forces))),
        "f_rms_hartree_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "solvation": solvation,
    }


def _safe_positions(positions: Any) -> tuple[list[list[float | None]], str]:
    array = np.asarray(positions, dtype=np.float64)
    safe = [
        [float(value) if math.isfinite(float(value)) else None for value in row]
        for row in array
    ]
    return safe, repr(array.tolist())


def _energy_ledger(record: dict[str, Any]) -> dict[str, float]:
    solvation = record.get("solvation")
    if not isinstance(solvation, dict):
        raise ValueError("composed record lacks solvation ledger")
    components = solvation.get("components_hartree")
    if not isinstance(components, dict) or set(components) != {
        "polar",
        "cavity",
        "dispersion",
    }:
        raise ValueError("composed record lacks exact solvent components")
    ledger = {
        "combined": float(record["combined_energy_hartree"]),
        "gas": float(solvation["gas_energy_hartree"]),
        "solvent": float(solvation["energy_hartree"]),
        **{f"component_{key}": float(value) for key, value in components.items()},
    }
    if not all(math.isfinite(value) for value in ledger.values()):
        raise ValueError("composed energy ledger is nonfinite")
    return ledger


def _validate_solvation_profile(
    solvation: dict[str, Any],
    expected_profile_id: str,
    expected_topology_sha256: str,
    expected_sigma_e: float,
    expected_order: int,
) -> None:
    provenance = solvation.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("solvation provenance is missing")
    backend = provenance.get("r6_backend_diagnostics")
    expected_backend = (
        "torch_continuum_chagb._r6_inverse_born"
        if expected_profile_id == V1_PROFILE_ID
        else "torch_continuum_r6_derivative_v2._r6_inverse_born_v2"
    )
    if not isinstance(backend, dict) or (
        provenance.get("numerical_profile_id") != expected_profile_id
        or provenance.get("topology_sha256") != expected_topology_sha256
        or float(provenance.get("sigma_e", math.nan)) != float(expected_sigma_e)
        or provenance.get("quadrature_order") != expected_order
        or backend.get("profile_id") != expected_profile_id
        or backend.get("backend_function") != expected_backend
        or backend.get("order") != expected_order
    ):
        raise ValueError("solvation provenance differs from execution context")


def compare_fresh_replay(
    retained_frame: dict[str, Any] | None, fresh: dict[str, Any]
) -> dict[str, Any]:
    if retained_frame is None:
        return {"passed": False, "reason": "NO_RETAINED_COMPOSED_FORCE_FRAME"}
    retained_forces = np.asarray(
        retained_frame["combined_forces_hartree_per_angstrom"], dtype=np.float64
    )
    fresh_forces = np.asarray(
        fresh["combined_forces_hartree_per_angstrom"], dtype=np.float64
    )
    retained_ledger = _energy_ledger(retained_frame)
    fresh_ledger = _energy_ledger(fresh)
    energy_differences = {
        key: abs(fresh_ledger[key] - retained_ledger[key]) for key in retained_ledger
    }
    retained_coordinate_valid = coordinate_sha256(
        retained_frame["positions_angstrom"]
    ) == retained_frame.get("coordinate_sha256")
    fresh_coordinate_valid = coordinate_sha256(
        fresh["positions_angstrom"]
    ) == fresh.get("coordinate_sha256")
    coordinate_match = (
        retained_coordinate_valid
        and fresh_coordinate_valid
        and retained_frame.get("coordinate_sha256") == fresh.get("coordinate_sha256")
    )
    force_difference = (
        math.inf
        if retained_forces.shape != (3, 3) or fresh_forces.shape != (3, 3)
        else float(np.max(np.abs(fresh_forces - retained_forces)))
    )
    passed = bool(
        coordinate_match
        and max(energy_differences.values()) <= FRESH_REPLAY_ENERGY_TOLERANCE_HARTREE
        and force_difference <= FRESH_REPLAY_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM
    )
    return {
        "passed": passed,
        "coordinate_match": coordinate_match,
        "retained_coordinate_hash_valid": retained_coordinate_valid,
        "fresh_coordinate_hash_valid": fresh_coordinate_valid,
        "energy_abs_differences_hartree": energy_differences,
        "maximum_force_abs_difference_hartree_per_angstrom": force_difference,
        "energy_tolerance_hartree": FRESH_REPLAY_ENERGY_TOLERANCE_HARTREE,
        "force_tolerance_hartree_per_angstrom": FRESH_REPLAY_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM,
    }


def _runtime_record() -> dict[str, Any]:
    import ase
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ase": ase.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_default_dtype": str(torch.get_default_dtype()),
    }


def _safe_runtime_record() -> dict[str, Any]:
    try:
        return _runtime_record()
    except Exception as exc:
        return serialize_exception(exc)


def _load_optimizer_implementation() -> tuple[Callable[..., Any], dict[str, Any]]:
    from maple.function.dispatcher.optimization.optimization import Optimization
    from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS

    method_source_name = inspect.getsourcefile(LBFGS)
    if method_source_name is None:
        raise RuntimeError("LBFGS implementation has no filesystem source")
    method_source = Path(method_source_name).resolve()
    return Optimization, {
        "class": LBFGS.__qualname__,
        "module": LBFGS.__module__,
        "source": str(method_source),
        "source_sha256": sha256_file(method_source),
    }


def execute_opt_row(
    row: dict[str, Any],
    topology: dict[str, Any],
    protocol: dict[str, Any],
    row_dir: Path,
    *,
    calculator_factory: Callable[..., Calculator] = build_composed_calculator,
    optimization_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one real row transaction and always attempt one fresh final."""
    method_implementation: dict[str, Any] | None = None
    atoms = Atoms(
        numbers=topology["atomic_numbers"],
        positions=np.asarray(row["positions_angstrom"], dtype=np.float64),
    )
    thresholds = protocol["optimizer"]["thresholds"]
    _set_thresholds(atoms, thresholds)
    original = atoms.get_positions().copy()
    rejected = None
    classification_error = None
    failure: Exception | None = None
    observer = None
    parameters = {
        "method": "lbfgs",
        "memory": protocol["optimizer"]["memory"],
        "curvature": protocol["optimizer"]["curvature"],
        "max_step": protocol["optimizer"]["max_step_angstrom"],
        "max_iter": protocol["optimizer"]["max_iterations"],
        "verbose": 1,
    }
    try:
        if optimization_factory is None:
            optimization_factory, method_implementation = (
                _load_optimizer_implementation()
            )
        composed = calculator_factory(
            atoms, topology, row["sigma_e"], row_dir / "calculator.log"
        )
        caps = protocol["optimizer"]["per_row_caps"]
        factory_profile_id = getattr(calculator_factory, "numerical_profile_id", None)
        expected_solvation_profile = (
            None
            if not isinstance(factory_profile_id, str)
            else (
                factory_profile_id,
                topology["content_sha256"],
                float(row["sigma_e"]),
                64,
            )
        )
        observer = ComposedRecordingCalculator(
            composed,
            caps=ObserverCaps(
                calls=int(caps["observer_calculator_calls"]),
                unique_force_coordinates=int(
                    caps["unique_successful_force_coordinate_hashes"]
                ),
            ),
            expected_solvation_profile=expected_solvation_profile,
        )
        atoms.calc = observer
        if optimization_factory is None:
            raise RuntimeError("optimizer implementation did not load")
        optimized = optimization_factory(
            parameters, str(row_dir / "optimization.out"), atoms
        ).run()
        retained_metrics = convergence_metrics(optimized)
        retained_pass = metrics_pass(retained_metrics, thresholds)
        provisional_status = "CONVERGED" if retained_pass else "MAX_ITER"
    except Exception as exc:  # row boundary deliberately retains all failures
        rejected_positions, rejected_raw = _safe_positions(atoms.get_positions())
        rejected = {
            **serialize_exception(exc),
            "positions_angstrom": rejected_positions,
            "raw_positions_repr": rejected_raw,
            "coordinate_sha256": (
                coordinate_sha256(atoms.get_positions())
                if np.isfinite(atoms.get_positions()).all()
                else None
            ),
        }
        failure = exc
        restored = observer.last_force_positions if observer is not None else None
        atoms.set_positions(original if restored is None else restored)
        atoms.calc = None
        retained_metrics = None
        try:
            domain_rejected = _is_domain_exception(exc)
        except Exception as classifier_exc:
            classification_error = {
                "type": type(classifier_exc).__name__,
                "message": str(classifier_exc),
            }
            domain_rejected = False
        provisional_status = "DOMAIN_REJECTED" if domain_rejected else "RUNTIME_ERROR"
    # Never reuse the optimization calculator/history for terminal proof.
    atoms.calc = None
    try:
        fresh = _fresh_final(
            atoms,
            topology,
            row["sigma_e"],
            calculator_factory,
            row_dir / "fresh-final-calculator.log",
        )
        fresh_pass = bool(
            math.isfinite(float(fresh["f_max_hartree_per_angstrom"]))
            and math.isfinite(float(fresh["f_rms_hartree_per_angstrom"]))
            and float(fresh["f_max_hartree_per_angstrom"]) >= 0.0
            and float(fresh["f_rms_hartree_per_angstrom"]) >= 0.0
            and fresh["f_max_hartree_per_angstrom"]
            <= thresholds["f_max_hartree_per_angstrom"]
            and fresh["f_rms_hartree_per_angstrom"]
            <= thresholds["f_rms_hartree_per_angstrom"]
        )
    except Exception as exc:
        try:
            fresh_domain_rejected = _is_domain_exception(exc)
        except Exception as classifier_exc:
            fresh_domain_rejected = False
            fresh_classification_error = serialize_exception(classifier_exc)
        else:
            fresh_classification_error = None
        fresh = {
            "tag": "FRESH_FINAL_UNCACHED",
            "status": ("DOMAIN_REJECTED" if fresh_domain_rejected else "RUNTIME_ERROR"),
            **serialize_exception(exc),
            "classification_error": fresh_classification_error,
        }
        fresh_pass = False
        if failure is None:
            provisional_status = (
                "DOMAIN_REJECTED" if fresh_domain_rejected else "RUNTIME_ERROR"
            )
    try:
        replay = (
            compare_fresh_replay(
                (
                    None
                    if observer is None or not observer.frames
                    else observer.frames[-1]
                ),
                fresh,
            )
            if "combined_forces_hartree_per_angstrom" in fresh
            else {"passed": False, "reason": "FRESH_FINAL_FAILED"}
        )
    except (KeyError, TypeError, ValueError) as exc:
        replay = {
            "passed": False,
            "reason": "FRESH_REPLAY_INVALID",
            **serialize_exception(exc),
        }
    status = (
        "CONVERGED"
        if provisional_status == "CONVERGED" and fresh_pass and replay["passed"]
        else (
            "RUNTIME_ERROR"
            if provisional_status == "CONVERGED" and not replay["passed"]
            else (
                "MAX_ITER" if provisional_status == "CONVERGED" else provisional_status
            )
        )
    )
    return {
        **{
            key: row[key]
            for key in ("row_id", "center_index", "center_scale", "sigma_e")
        },
        "status": status,
        "retained_positions_angstrom": atoms.get_positions().tolist(),
        "retained_coordinate_sha256": coordinate_sha256(atoms.get_positions()),
        "retained_metrics": retained_metrics,
        "observer_calculator_calls": 0 if observer is None else observer.call_count,
        "unique_successful_force_coordinate_hashes": (
            0 if observer is None else observer.unique_force_coordinate_count
        ),
        "successful_force_frames": [] if observer is None else observer.frames,
        "rejected_evaluation": rejected,
        "classification_error": classification_error,
        "fresh_final": fresh,
        "fresh_final_pass": fresh_pass,
        "fresh_replay": replay,
        "optimizer_execution": {
            "class": getattr(
                optimization_factory,
                "__qualname__",
                type(optimization_factory).__qualname__,
            ),
            "module": getattr(
                optimization_factory,
                "__module__",
                type(optimization_factory).__module__,
            ),
            "requested_parameters": parameters,
            "thresholds": dict(thresholds),
            "method_implementation": method_implementation,
        },
        "runtime": _safe_runtime_record(),
        "gas_model_metadata": (
            None
            if observer is None
            else json_safe(
                getattr(observer.composed, "gaussian_opt_model_metadata", None)
            )
        ),
        "returned_atoms_is_not_convergence_evidence": True,
    }


def five_point_derivative(energies: Iterable[float], h: float) -> float:
    em2, em1, ep1, ep2 = [float(value) for value in energies]
    return (em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * float(h))


def summarize_matrix(rows: list[dict[str, Any]], required: int = 15):
    success = sum(row.get("status") == "CONVERGED" for row in rows)
    fresh = sum(bool(row.get("fresh_final_pass")) for row in rows)
    complete = len(rows) == required and success == required and fresh == required
    return {
        "row_count": len(rows),
        "converged_count": success,
        "fresh_final_pass_count": fresh,
        "status": "VALIDATED" if complete else "GAUSSIAN_OPT_INCOMPLETE",
    }


def new_attempt_directory(work_root: Path, row_id: str) -> Path:
    """Retain interrupted attempts and allocate the next immutable directory."""
    row_root = work_root / row_id
    row_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10_000):
        candidate = row_root / f"attempt-{index:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"attempt directory budget exhausted for {row_id}")


def _bind_receipt(
    payload: dict[str, Any],
    evidence: dict[str, Any],
    *,
    receipt_kind: str,
    case_id: str,
    case_identity: dict[str, Any],
) -> dict[str, Any]:
    return seal(
        {
            **payload,
            "receipt_kind": receipt_kind,
            "case_id": case_id,
            "case_identity": case_identity,
            "preregistration_sha256": evidence["preregistration_sha256"],
            "source_identity_sha256": evidence["source_identity_sha256"],
            "numerical_profile_id": evidence["numerical_profile_id"],
            "context_id": evidence["context_id"],
            "protocol_template_sha256": evidence["protocol_template_sha256"],
            "approved_plan_sha256": evidence["approved_plan_sha256"],
            "approved_handoff_sha256": evidence["approved_handoff_sha256"],
            "protocol_id": evidence["protocol_id"],
        }
    )


def _verify_receipt_binding(
    receipt: dict[str, Any],
    evidence: dict[str, Any],
    *,
    receipt_kind: str,
    case_id: str,
    case_identity: dict[str, Any],
) -> None:
    verify_seal(receipt, f"{receipt_kind} receipt {case_id}")
    expected = {
        "receipt_kind": receipt_kind,
        "case_id": case_id,
        "case_identity": case_identity,
        "preregistration_sha256": evidence["preregistration_sha256"],
        "source_identity_sha256": evidence["source_identity_sha256"],
        "numerical_profile_id": evidence["numerical_profile_id"],
        "context_id": evidence["context_id"],
        "protocol_template_sha256": evidence["protocol_template_sha256"],
        "approved_plan_sha256": evidence["approved_plan_sha256"],
        "approved_handoff_sha256": evidence["approved_handoff_sha256"],
        "protocol_id": evidence["protocol_id"],
    }
    for key, value in expected.items():
        if canonical_bytes(receipt.get(key)) != canonical_bytes(value):
            raise ValueError(f"{receipt_kind} receipt binding differs: {case_id} {key}")


def _assert_phase_stable(
    protocol: dict[str, Any],
    evidence_before: dict[str, Any],
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> None:
    _, protocol_digest = load_protocol_template(context)
    evidence_after = _phase_evidence(protocol, protocol_digest, context)
    if (
        evidence_after["source_identity_sha256"]
        != evidence_before["source_identity_sha256"]
    ):
        raise ValueError("input/source/protected snapshot changed during phase")


def run_opt_matrix(
    output: Path,
    expected_preregistration_sha256: str,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, Any]:
    protocol, prereg, evidence = load_preregistration(
        output, expected_preregistration_sha256, context
    )
    topology = evidence["topology"]
    rows_dir = output / "rows"
    collected = []
    for row in prereg["rows"]:
        row_path = rows_dir / f"{row['row_id']}.json"
        if row_path.exists():
            existing, _ = read_hashed_json(row_path, "existing row receipt")
            _verify_receipt_binding(
                existing,
                evidence,
                receipt_kind="OPT_ROW",
                case_id=row["row_id"],
                case_identity=row,
            )
            collected.append(existing)
            continue
        row_dir = new_attempt_directory(output / "work", row["row_id"])
        receipt = _bind_receipt(
            execute_opt_row(
                row,
                topology,
                protocol,
                row_dir,
                calculator_factory=context.calculator_factory,
            ),
            evidence,
            receipt_kind="OPT_ROW",
            case_id=row["row_id"],
            case_identity=row,
        )
        write_json_new_atomic(row_path, receipt)
        collected.append(receipt)
        progress_path = output / "progress" / f"{len(collected):02d}.json"
        write_json_new_atomic(
            progress_path,
            _bind_receipt(
                summarize_matrix(collected),
                evidence,
                receipt_kind="OPT_PROGRESS",
                case_id=f"rows-{len(collected):02d}",
                case_identity={"row_ids": [item["row_id"] for item in collected]},
            ),
        )
    summary = _bind_receipt(
        recompute_opt_summary(
            collected,
            protocol,
            prereg["rows"],
            evidence["source"],
            evidence["topology"]["content_sha256"],
            context.numerical_profile_id,
        ),
        evidence,
        receipt_kind="OPT_SUMMARY",
        case_id="optimization-summary",
        case_identity={"row_ids": [row["row_id"] for row in prereg["rows"]]},
    )
    summary["rows"] = [row["row_id"] for row in collected]
    summary["content_sha256"] = content_sha256(summary)
    _assert_phase_stable(protocol, evidence, context)
    write_json_new_atomic(output / "optimization-summary.json", summary)
    return summary


def _evaluate_total(atoms: Atoms) -> tuple[float, np.ndarray]:
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    energy = float(atoms.get_potential_energy(force_consistent=True))
    return energy, forces


def _combined_fd_row(
    row: dict[str, Any],
    topology: dict[str, Any],
    protocol: dict[str, Any],
    log: Path,
    calculator_factory: Callable[..., Calculator] = build_composed_calculator,
) -> dict[str, Any]:
    steps = protocol["combined_fd_validation"]["fd_steps_angstrom"]
    log.parent.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(
        numbers=topology["atomic_numbers"], positions=row["positions_angstrom"]
    )
    atoms.calc = calculator_factory(atoms, topology, row["sigma_e"], log)
    base = atoms.get_positions().copy()
    _, analytic = _evaluate_total(atoms)
    factory_profile_id = getattr(calculator_factory, "numerical_profile_id", None)
    if isinstance(factory_profile_id, str):
        _validate_solvation_profile(
            atoms.calc.results["solvation"],
            factory_profile_id,
            topology["content_sha256"],
            float(row["sigma_e"]),
            64,
        )
    comparisons = []
    for flat in range(base.size):
        atom, axis = divmod(flat, 3)
        estimates = []
        stencils = []
        for h in steps:
            samples = []
            energies = []
            for multiplier in (-2.0, -1.0, 1.0, 2.0):
                displaced = base.copy()
                displaced[atom, axis] += multiplier * h
                atoms.set_positions(displaced)
                energy = float(atoms.get_potential_energy(force_consistent=True))
                energies.append(energy)
                samples.append(
                    {
                        "multiplier": multiplier,
                        "positions_angstrom": displaced.tolist(),
                        "coordinate_sha256": coordinate_sha256(displaced),
                        "combined_energy_hartree": energy,
                    }
                )
            estimates.append(-five_point_derivative(energies, h))
            stencils.append({"step_angstrom": h, "samples": samples})
        comparisons.append(
            {
                "atom_index": atom,
                "axis": axis,
                "analytic_force_hartree_per_angstrom": float(analytic[atom, axis]),
                "fd_force_hartree_per_angstrom": estimates,
                "stencils": stencils,
                "finest_error": float(estimates[-1] - analytic[atom, axis]),
                "finest_step_disagreement": float(estimates[-1] - estimates[-2]),
            }
        )
    errors = np.asarray([item["finest_error"] for item in comparisons])
    disagreement = max(abs(item["finest_step_disagreement"]) for item in comparisons)
    gates = protocol["combined_fd_validation"]
    record = {
        "row_id": row["row_id"],
        "source_positions_angstrom": base.tolist(),
        "source_positions_sha256": coordinate_sha256(base),
        "analytic_forces_hartree_per_angstrom": analytic.tolist(),
        "comparisons": comparisons,
        "maximum_absolute_error_hartree_per_angstrom": float(np.max(np.abs(errors))),
        "rms_error_hartree_per_angstrom": float(np.sqrt(np.mean(errors**2))),
        "maximum_finest_step_disagreement_hartree_per_angstrom": float(disagreement),
    }
    record["passed"] = bool(
        record["maximum_absolute_error_hartree_per_angstrom"]
        <= gates["force_max_abs_hartree_per_angstrom"]
        and record["rms_error_hartree_per_angstrom"]
        <= gates["force_rms_hartree_per_angstrom"]
        and disagreement <= gates["finest_step_disagreement_max_hartree_per_angstrom"]
    )
    return record


def run_combined_fd(
    output: Path,
    expected_preregistration_sha256: str,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, Any]:
    protocol, prereg, evidence = load_preregistration(
        output, expected_preregistration_sha256, context
    )
    topology = evidence["topology"]
    records = []
    for row in prereg["rows"]:
        path = output / "combined-fd" / f"{row['row_id']}.json"
        if path.exists():
            existing, _ = read_hashed_json(path, "combined FD receipt")
            _verify_receipt_binding(
                existing,
                evidence,
                receipt_kind="COMBINED_FD",
                case_id=row["row_id"],
                case_identity=row,
            )
            records.append(existing)
            continue
        try:
            payload = _combined_fd_row(
                row,
                topology,
                protocol,
                output / "combined-fd" / f"{row['row_id']}.log",
                context.calculator_factory,
            )
        except Exception as exc:
            payload = {
                "row_id": row["row_id"],
                "passed": False,
                **serialize_exception(exc),
            }
        record = _bind_receipt(
            payload,
            evidence,
            receipt_kind="COMBINED_FD",
            case_id=row["row_id"],
            case_identity=row,
        )
        write_json_new_atomic(path, record)
        records.append(record)
    summary = recompute_combined_fd_summary(records, protocol, prereg["rows"])
    summary = _bind_receipt(
        summary,
        evidence,
        receipt_kind="COMBINED_FD_SUMMARY",
        case_id="combined-fd-summary",
        case_identity={"row_ids": [row["row_id"] for row in prereg["rows"]]},
    )
    _assert_phase_stable(protocol, evidence, context)
    write_json_new_atomic(output / "combined-fd-summary.json", summary)
    return summary


def _reference_geometry_metadata(geometry: Any) -> dict[str, Any]:
    metadata = {
        "r6_levels": [
            {
                "azimuth_order": int(level.azimuth_order),
                "inverse_cube_quad_error_estimate_per_angstrom3": float(
                    level.inverse_cube_quad_error_estimate_per_angstrom3
                ),
                "meridian_evaluations": int(level.meridian_evaluations),
                "inverse_cube_per_angstrom3": json_safe(
                    level.inverse_cube_per_angstrom3
                ),
                "inverse_born_per_angstrom": json_safe(level.inverse_born_per_angstrom),
                "gauss_closure_vector_angstrom2": json_safe(
                    level.gauss_closure_vector_angstrom2
                ),
            }
            for level in geometry.r6_levels
        ],
        "r6_diagnostics": json_safe(geometry.r6_diagnostics),
        "nonpolar_diagnostics": json_safe(geometry.diagnostics),
        "epsabs": float(geometry.diagnostics["epsabs"]),
        "epsrel": float(geometry.diagnostics["epsrel"]),
        "cavity_volume_angstrom3": float(geometry.cavity_volume_angstrom3),
        "cavity_kcal_mol": float(geometry.cavity_kcal_mol),
        "dispersion_kcal_mol": float(geometry.dispersion_kcal_mol),
        "coordinates_sha256": geometry.coordinates_sha256,
        "parameters_sha256": geometry.parameters_sha256,
        "identity_sha256": geometry.identity_sha256,
        "payload_sha256": geometry.payload_sha256,
    }
    return seal(metadata)


def _validate_reference_geometry_metadata(
    metadata: dict[str, Any],
    expected_orders: tuple[int, ...],
    expected_epsabs: float,
    expected_epsrel: float,
) -> bool:
    try:
        verify_seal(metadata, "reference geometry metadata")
        levels = metadata["r6_levels"]
        diagnostics = metadata["nonpolar_diagnostics"]
        valid = bool(
            [int(level["azimuth_order"]) for level in levels] == list(expected_orders)
            and float(metadata["epsabs"]) == float(expected_epsabs)
            and float(metadata["epsrel"]) == float(expected_epsrel)
            and isinstance(metadata["r6_diagnostics"], dict)
            and bool(metadata["r6_diagnostics"])
            and isinstance(diagnostics, dict)
            and diagnostics["azimuth_orders"] == list(expected_orders)
            and float(diagnostics["epsabs"]) == float(expected_epsabs)
            and float(diagnostics["epsrel"]) == float(expected_epsrel)
            and diagnostics["error_estimates_are_rigorous_bounds"] is False
            and diagnostics["mixed_vector_quad_errors_are_scalar_uncertainties"]
            is False
            and "cavity_mixed_vector_quad_error" in diagnostics
            and "dispersion_mixed_vector_quad_error" in diagnostics
            and np.isfinite(
                np.asarray(
                    diagnostics["cavity_mixed_vector_quad_error"], dtype=np.float64
                )
            ).all()
            and np.isfinite(
                np.asarray(
                    diagnostics["dispersion_mixed_vector_quad_error"],
                    dtype=np.float64,
                )
            ).all()
        )
        for level in levels:
            valid = bool(
                valid
                and int(level["meridian_evaluations"]) > 0
                and math.isfinite(
                    float(level["inverse_cube_quad_error_estimate_per_angstrom3"])
                )
                and float(level["inverse_cube_quad_error_estimate_per_angstrom3"])
                >= 0.0
                and all(
                    np.asarray(level[key], dtype=np.float64).shape == (3,)
                    and np.isfinite(level[key]).all()
                    for key in (
                        "inverse_cube_per_angstrom3",
                        "inverse_born_per_angstrom",
                        "gauss_closure_vector_angstrom2",
                    )
                )
            )
        return valid
    except (KeyError, TypeError, ValueError):
        return False


def run_crossing_reference(
    output: Path,
    expected_preregistration_sha256: str,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, Any]:
    """Run independent solvent reference/FD checks at the frozen event crossing."""
    protocol, prereg, evidence = load_preregistration(
        output, expected_preregistration_sha256, context
    )
    topology = evidence["topology"]
    oracle = _import_owned(
        "docs.implicit-solvation.benchmarks.cha_gaussian_reference", context
    )
    correction_module = _import_owned(
        "maple.function.calculator.extra_correction.implicit.gaussian_cha_correction",
        context,
    )
    event = next(row for row in prereg["rows"] if row["center_scale"] == 1.0)
    base = np.asarray(event["positions_angstrom"], dtype=np.float64)
    offsets = np.linspace(-0.002, 0.002, 41, dtype=np.float64)
    steps = protocol["crossing_validation"]["fd_steps_angstrom"]
    coordinates = [("scan", 0, 0, float(offset)) for offset in offsets]
    coordinates.extend(
        ("event-cartesian", flat // 3, flat % 3, 0.0) for flat in range(9)
    )
    parameter_identity = sha256_bytes(
        canonical_bytes(
            {
                key: topology[key]
                for key in (
                    "effective_charges_e",
                    "cha_radii_angstrom",
                    "lj_rmin_angstrom",
                    "lj_epsilon_kcal_mol",
                )
            }
        )
    )
    geometry_cache: dict[tuple[bytes, int, str, str], Any] = {}

    def geometry_for(positions: np.ndarray, order: int):
        exact = np.asarray(positions, dtype="<f8").tobytes(order="C")
        key = (exact, order, parameter_identity, evidence["source_identity_sha256"])
        if key not in geometry_cache:
            geometry_cache[key] = oracle.prepare_gaussian_reference_geometry(
                positions,
                topology["effective_charges_e"],
                topology["cha_radii_angstrom"],
                topology["lj_rmin_angstrom"],
                topology["lj_epsilon_kcal_mol"],
                azimuth_orders=(order,),
                epsabs=protocol["scalar"]["reference_epsabs"],
                epsrel=protocol["scalar"]["reference_epsrel"],
            )
        return geometry_cache[key]

    records = []
    expected_ids = []
    for sigma_e in protocol["inputs"]["sigma_e"]:
        for kind, atom, axis, offset in coordinates:
            record_id = f"sigma-{sigma_e:g}-{kind}-{atom}-{axis}-{offset:+.8f}"
            expected_ids.append(record_id)
            center = base.copy()
            center[0, 0] += offset
            case_identity = {
                "record_id": record_id,
                "sigma_e": sigma_e,
                "kind": kind,
                "atom_index": atom,
                "axis": axis,
                "offset_angstrom": offset,
                "source_event_positions_angstrom": base.tolist(),
                "positions_angstrom": center.tolist(),
                "positions_sha256": coordinate_sha256(center),
            }
            path = output / "crossing-reference" / f"{record_id}.json"
            if path.exists():
                existing, _ = read_hashed_json(path, "crossing receipt")
                _verify_receipt_binding(
                    existing,
                    evidence,
                    receipt_kind="CROSSING_REFERENCE",
                    case_id=record_id,
                    case_identity=case_identity,
                )
                records.append(existing)
                continue
            try:
                atoms = Atoms(numbers=topology["atomic_numbers"], positions=center)
                correction = correction_module.GaussianChaCorrection(
                    atoms,
                    _typed_topology(topology),
                    expected_topology_sha256=topology["content_sha256"],
                    sigma_e=sigma_e,
                    order=64,
                    numerical_profile_id=context.numerical_profile_id,
                )
                production = correction.evaluate(atoms, need_forces=True)
                _validate_solvation_profile(
                    {"provenance": production.provenance},
                    context.numerical_profile_id,
                    topology["content_sha256"],
                    float(sigma_e),
                    64,
                )
                geometry_by_order = {
                    str(order): geometry_for(center, order)
                    for order in protocol["scalar"]["reference_quadrature_orders"]
                }
                reference_by_order = {
                    order: oracle.gaussian_cha_from_reference_geometry(
                        geometry, sigma_e=sigma_e
                    )
                    for order, geometry in geometry_by_order.items()
                }
                stencils = []
                estimates = []
                for h in steps:
                    samples = []
                    energies = []
                    for multiplier in (-2.0, -1.0, 1.0, 2.0):
                        displaced = center.copy()
                        displaced[atom, axis] += multiplier * h
                        result = oracle.gaussian_cha_from_reference_geometry(
                            geometry_for(displaced, 128), sigma_e=sigma_e
                        )
                        energy = float(result.energies_kcal_mol["total"])
                        energies.append(energy)
                        samples.append(
                            {
                                "multiplier": multiplier,
                                "positions_angstrom": displaced.tolist(),
                                "coordinate_sha256": coordinate_sha256(displaced),
                                "reference_total_kcal_mol": energy,
                            }
                        )
                    estimates.append(-five_point_derivative(energies, h))
                    stencils.append({"step_angstrom": h, "samples": samples})
                production_components = {
                    key: float(value) * KCAL_PER_HARTREE
                    for key, value in production.components_hartree.items()
                }
                production_components["total"] = (
                    float(production.energy_hartree) * KCAL_PER_HARTREE
                )
                reference_components = {
                    order: {
                        key: float(value.energies_kcal_mol[key])
                        for key in ("polar", "cavity", "dispersion", "total")
                    }
                    for order, value in reference_by_order.items()
                }
                energy_errors = {
                    order: {
                        key: production_components[key] - components[key]
                        for key in production_components
                    }
                    for order, components in reference_components.items()
                }
                prod_force = (
                    float(production.forces_hartree_per_angstrom[atom, axis])
                    * KCAL_PER_HARTREE
                )
                payload = {
                    "record_id": record_id,
                    "status": "COMPLETED",
                    "production_components_kcal_mol": production_components,
                    "reference_components_kcal_mol_by_order": reference_components,
                    "reference_geometry_metadata_by_order": {
                        order: _reference_geometry_metadata(geometry)
                        for order, geometry in geometry_by_order.items()
                    },
                    "energy_errors_kcal_mol_by_order": energy_errors,
                    "production_force_kcal_mol_per_angstrom": prod_force,
                    "reference_fd_force_kcal_mol_per_angstrom": estimates,
                    "reference_force_method": "five-point energy FD; nonrigorous independent error estimate",
                    "reference_fd_stencils": stencils,
                    "force_error_kcal_mol_per_angstrom": prod_force - estimates[-1],
                    "finest_step_disagreement_kcal_mol_per_angstrom": estimates[-1]
                    - estimates[-2],
                    "production_provenance": json_safe(production.provenance),
                    "reference_128_diagnostics": {
                        "smoothed_signs": json_safe(
                            reference_by_order["128"].smoothed_signs
                        ),
                        "weighted_signs_over_sigma": json_safe(
                            reference_by_order["128"].weighted_signs_over_sigma
                        ),
                        "gaussian_erfc_tails": json_safe(
                            reference_by_order["128"].gaussian_erfc_tails
                        ),
                        "born_radii_angstrom": json_safe(
                            reference_by_order["128"].born_radii_angstrom
                        ),
                        "electrostatic_size_angstrom": float(
                            reference_by_order["128"].electrostatic_size_angstrom
                        ),
                    },
                }
            except Exception as exc:
                payload = {
                    "record_id": record_id,
                    "status": "ERROR",
                    **serialize_exception(exc),
                }
            record = _bind_receipt(
                payload,
                evidence,
                receipt_kind="CROSSING_REFERENCE",
                case_id=record_id,
                case_identity=case_identity,
            )
            write_json_new_atomic(path, record)
            records.append(record)
    summary = recompute_crossing_summary(records, protocol, expected_ids)
    summary["geometry_cache"] = {
        "entry_count": len(geometry_cache),
        "key_contract": "exact float64 coordinate bytes + order + width-independent parameter identity + source identity",
        "shared_across_sigma_only_for_identical_coordinates": True,
        "runtime_born_frozen": False,
    }
    summary = _bind_receipt(
        summary,
        evidence,
        receipt_kind="CROSSING_SUMMARY",
        case_id="crossing-reference-summary",
        case_identity={"record_ids": expected_ids},
    )
    _assert_phase_stable(protocol, evidence, context)
    write_json_new_atomic(output / "crossing-reference-summary.json", summary)
    return summary


def _rederive_crossing_record(
    record: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        identity = record["case_identity"]
        center = np.asarray(identity["positions_angstrom"], dtype=np.float64)
        atom = int(identity["atom_index"])
        axis = int(identity["axis"])
        production = record["production_components_kcal_mol"]
        reference = record["reference_components_kcal_mol_by_order"]
        reference_metadata = record["reference_geometry_metadata_by_order"]
        component_keys = {"polar", "cavity", "dispersion", "total"}
        orders = [
            str(value) for value in protocol["scalar"]["reference_quadrature_orders"]
        ]
        if (
            set(production) != component_keys
            or set(reference) != set(orders)
            or set(reference_metadata) != set(orders)
        ):
            return None
        if any(set(reference[order]) != component_keys for order in orders):
            return None
        if any(
            not _validate_reference_geometry_metadata(
                reference_metadata[order],
                (int(order),),
                protocol["scalar"]["reference_epsabs"],
                protocol["scalar"]["reference_epsrel"],
            )
            for order in orders
        ):
            return None
        energy_errors = {
            order: {
                key: float(production[key]) - float(reference[order][key])
                for key in component_keys
            }
            for order in orders
        }
        estimates = []
        stencils = record["reference_fd_stencils"]
        for expected_h, stencil in zip(
            protocol["crossing_validation"]["fd_steps_angstrom"], stencils
        ):
            if float(stencil["step_angstrom"]) != float(expected_h):
                return None
            samples = stencil["samples"]
            if [sample["multiplier"] for sample in samples] != [-2.0, -1.0, 1.0, 2.0]:
                return None
            energies = []
            for sample in samples:
                expected_positions = center.copy()
                expected_positions[atom, axis] += float(sample["multiplier"]) * float(
                    expected_h
                )
                observed = np.asarray(sample["positions_angstrom"], dtype=np.float64)
                if not np.array_equal(observed, expected_positions):
                    return None
                if coordinate_sha256(observed) != sample["coordinate_sha256"]:
                    return None
                energies.append(float(sample["reference_total_kcal_mol"]))
            estimates.append(-five_point_derivative(energies, expected_h))
        if len(stencils) != 3 or not all(math.isfinite(value) for value in estimates):
            return None
        result = dict(record)
        result["energy_errors_kcal_mol_by_order"] = energy_errors
        result["reference_fd_force_kcal_mol_per_angstrom"] = estimates
        result["force_error_kcal_mol_per_angstrom"] = (
            float(record["production_force_kcal_mol_per_angstrom"]) - estimates[-1]
        )
        result["finest_step_disagreement_kcal_mol_per_angstrom"] = (
            estimates[-1] - estimates[-2]
        )
        return result
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def recompute_crossing_summary(
    records: list[dict[str, Any]], protocol: dict[str, Any], expected_ids: list[str]
) -> dict[str, Any]:
    gate = protocol["crossing_validation"]
    ids = [record.get("record_id") for record in records]
    inventory_ok = ids == expected_ids and len(set(ids)) == len(expected_ids)
    completed = []
    for record in records:
        if record.get("status") == "COMPLETED":
            derived = _rederive_crossing_record(record, protocol)
            if derived is not None:
                completed.append(derived)
    energy_values = [
        abs(error)
        for record in completed
        for by_component in record["energy_errors_kcal_mol_by_order"].values()
        for error in by_component.values()
    ]
    groups = []
    for sigma_e in protocol["inputs"]["sigma_e"]:
        for kind in ("scan", "event-cartesian"):
            group = [
                record
                for record in completed
                if float(record["case_identity"]["sigma_e"]) == float(sigma_e)
                and record["case_identity"]["kind"] == kind
            ]
            force = np.asarray(
                [record["force_error_kcal_mol_per_angstrom"] for record in group]
            )
            disagreement = np.asarray(
                [
                    record["finest_step_disagreement_kcal_mol_per_angstrom"]
                    for record in group
                ]
            )
            metrics = {
                "sigma_e": sigma_e,
                "kind": kind,
                "count": len(group),
                "maximum_absolute_force_error_kcal_mol_per_angstrom": (
                    None if not len(group) else float(np.max(np.abs(force)))
                ),
                "rms_force_error_kcal_mol_per_angstrom": (
                    None if not len(group) else float(np.sqrt(np.mean(force**2)))
                ),
                "maximum_finest_step_disagreement_kcal_mol_per_angstrom": (
                    None if not len(group) else float(np.max(np.abs(disagreement)))
                ),
            }
            metrics["passed"] = bool(
                len(group) == (41 if kind == "scan" else 9)
                and metrics["maximum_absolute_force_error_kcal_mol_per_angstrom"]
                <= gate["force_max_abs_kcal_mol_per_angstrom"]
                and metrics["rms_force_error_kcal_mol_per_angstrom"]
                <= gate["force_rms_kcal_mol_per_angstrom"]
                and metrics["maximum_finest_step_disagreement_kcal_mol_per_angstrom"]
                <= gate["finest_step_disagreement_max_kcal_mol_per_angstrom"]
            )
            groups.append(metrics)
    by_coordinate: dict[tuple[Any, ...], list[float]] = {}
    for record in completed:
        identity = record["case_identity"]
        key = (
            identity["kind"],
            identity["atom_index"],
            identity["axis"],
            identity["offset_angstrom"],
        )
        by_coordinate.setdefault(key, []).append(
            record["production_components_kcal_mol"]["total"]
        )
    maximum_energy_error = None if not energy_values else max(energy_values)
    return {
        "record_count": len(records),
        "completed_count": len(completed),
        "inventory_exact": inventory_ok,
        "maximum_absolute_component_energy_error_kcal_mol_all_orders": maximum_energy_error,
        "force_groups": groups,
        "maximum_width_energy_span_kcal_mol_diagnostic_only": (
            None
            if not by_coordinate
            else float(max(max(v) - min(v) for v in by_coordinate.values()))
        ),
        "width_selection_performed": False,
        "passed": bool(
            inventory_ok
            and len(completed) == 150
            and maximum_energy_error is not None
            and maximum_energy_error <= gate["energy_max_abs_kcal_mol"]
            and all(group["passed"] for group in groups)
        ),
    }


def recompute_combined_fd_summary(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = protocol["combined_fd_validation"]
    exact_ids = [record.get("row_id") for record in records] == [
        row["row_id"] for row in expected_rows
    ] and len({record.get("row_id") for record in records}) == 15
    expected_by_id = {row["row_id"]: row for row in expected_rows}
    row_results = []
    for record in records:
        try:
            analytic = np.asarray(
                record["analytic_forces_hartree_per_angstrom"], dtype=np.float64
            )
            comparisons = record["comparisons"]
            expected = expected_by_id[record["row_id"]]
            base = np.asarray(expected["positions_angstrom"], dtype=np.float64)
            errors = []
            disagreements = []
            valid = analytic.shape == (3, 3) and len(comparisons) == 9
            for flat, comparison in enumerate(comparisons):
                atom = int(comparison["atom_index"])
                axis = int(comparison["axis"])
                valid = valid and (atom, axis) == divmod(flat, 3)
                estimates = []
                stencils = comparison["stencils"]
                valid = valid and len(stencils) == 3
                for expected_h, stencil in zip(gates["fd_steps_angstrom"], stencils):
                    samples = stencil["samples"]
                    valid = valid and float(stencil["step_angstrom"]) == float(
                        expected_h
                    )
                    valid = valid and [sample["multiplier"] for sample in samples] == [
                        -2.0,
                        -1.0,
                        1.0,
                        2.0,
                    ]
                    valid = valid and all(
                        coordinate_sha256(sample["positions_angstrom"])
                        == sample["coordinate_sha256"]
                        for sample in samples
                    )
                    for sample in samples:
                        expected_positions = base.copy()
                        expected_positions[atom, axis] += float(
                            sample["multiplier"]
                        ) * float(expected_h)
                        valid = valid and np.array_equal(
                            np.asarray(sample["positions_angstrom"], dtype=np.float64),
                            expected_positions,
                        )
                    estimates.append(
                        -five_point_derivative(
                            [sample["combined_energy_hartree"] for sample in samples],
                            expected_h,
                        )
                    )
                errors.append(estimates[-1] - analytic[atom, axis])
                disagreements.append(estimates[-1] - estimates[-2])
            errors_array = np.asarray(errors, dtype=np.float64)
            maximum = float(np.max(np.abs(errors_array)))
            rms = float(np.sqrt(np.mean(errors_array**2)))
            finest = float(np.max(np.abs(disagreements)))
            passed = bool(
                valid
                and np.isfinite(analytic).all()
                and maximum <= gates["force_max_abs_hartree_per_angstrom"]
                and rms <= gates["force_rms_hartree_per_angstrom"]
                and finest <= gates["finest_step_disagreement_max_hartree_per_angstrom"]
            )
            row_results.append(
                {
                    "row_id": record["row_id"],
                    "maximum_absolute_error_hartree_per_angstrom": maximum,
                    "rms_error_hartree_per_angstrom": rms,
                    "maximum_finest_step_disagreement_hartree_per_angstrom": finest,
                    "passed": passed,
                }
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            row_results.append(
                {
                    "row_id": record.get("row_id"),
                    "passed": False,
                    **serialize_exception(exc),
                }
            )
    return {
        "row_count": len(records),
        "inventory_exact": exact_ids,
        "rows": row_results,
        "all_passed": bool(
            exact_ids
            and len(row_results) == 15
            and all(row["passed"] for row in row_results)
        ),
    }


def recompute_opt_summary(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    source_snapshot: dict[str, str],
    expected_topology_sha256: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    thresholds = protocol["optimizer"]["thresholds"]
    caps = protocol["optimizer"]["per_row_caps"]
    expected_parameters = {
        "method": "lbfgs",
        "memory": protocol["optimizer"]["memory"],
        "curvature": protocol["optimizer"]["curvature"],
        "max_step": protocol["optimizer"]["max_step_angstrom"],
        "max_iter": protocol["optimizer"]["max_iterations"],
        "verbose": 1,
    }
    exact_ids = [row.get("row_id") for row in rows] == [
        row["row_id"] for row in expected_rows
    ] and len({row.get("row_id") for row in rows}) == 15
    results = []
    for row in rows:
        try:
            frames = row["successful_force_frames"]
            for frame in frames:
                _validate_solvation_profile(
                    frame["solvation"],
                    expected_profile_id,
                    expected_topology_sha256,
                    float(row["sigma_e"]),
                    64,
                )
            valid_frames = all(
                np.asarray(frame["positions_angstrom"]).shape == (3, 3)
                and np.asarray(frame["combined_forces_hartree_per_angstrom"]).shape
                == (3, 3)
                and np.isfinite(frame["positions_angstrom"]).all()
                and np.isfinite(frame["combined_forces_hartree_per_angstrom"]).all()
                and coordinate_sha256(frame["positions_angstrom"])
                == frame["coordinate_sha256"]
                and all(
                    math.isfinite(value) for value in _energy_ledger(frame).values()
                )
                for frame in frames
            )
            metrics = row["retained_metrics"]
            metrics_ok = isinstance(metrics, dict) and metrics_pass(metrics, thresholds)
            fresh = row["fresh_final"]
            _validate_solvation_profile(
                fresh["solvation"],
                expected_profile_id,
                expected_topology_sha256,
                float(row["sigma_e"]),
                64,
            )
            fresh_forces = np.asarray(
                fresh["combined_forces_hartree_per_angstrom"], dtype=np.float64
            )
            fresh_force_ok = bool(
                fresh_forces.shape == (3, 3)
                and np.isfinite(fresh_forces).all()
                and float(np.max(np.abs(fresh_forces)))
                <= thresholds["f_max_hartree_per_angstrom"]
                and float(np.sqrt(np.mean(fresh_forces**2)))
                <= thresholds["f_rms_hartree_per_angstrom"]
                and coordinate_sha256(fresh["positions_angstrom"])
                == fresh["coordinate_sha256"]
                and all(
                    math.isfinite(value) for value in _energy_ledger(fresh).values()
                )
            )
            replay = compare_fresh_replay(frames[-1] if frames else None, fresh)
            optimizer = row["optimizer_execution"]
            method = optimizer["method_implementation"]
            method_relative = (
                "maple/function/dispatcher/optimization/algorithm/LBFGS.py"
            )
            optimizer_ok = bool(
                optimizer["module"]
                == "maple.function.dispatcher.optimization.optimization"
                and optimizer["class"] == "Optimization"
                and optimizer["requested_parameters"] == expected_parameters
                and optimizer["thresholds"] == thresholds
                and method["module"]
                == "maple.function.dispatcher.optimization.algorithm.LBFGS"
                and method["class"] == "LBFGS"
                and Path(method["source"]).resolve()
                == (WORKTREE / method_relative).resolve()
                and method["source_sha256"] == source_snapshot[method_relative]
            )
            counts_ok = bool(
                int(row["observer_calculator_calls"])
                <= caps["observer_calculator_calls"]
                and int(row["observer_calculator_calls"]) >= len(frames)
                and int(row["unique_successful_force_coordinate_hashes"])
                <= caps["unique_successful_force_coordinate_hashes"]
                and len({frame["coordinate_sha256"] for frame in frames})
                == int(row["unique_successful_force_coordinate_hashes"])
            )
            runtime = row["runtime"]
            runtime_ok = bool(
                runtime["torch_num_threads"] == 1
                and runtime["thread_environment"]
                == {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                }
                and runtime["torch_default_dtype"] == "torch.float32"
            )
            passed = bool(
                row.get("status") == "CONVERGED"
                and metrics_ok
                and fresh_force_ok
                and replay["passed"]
                and valid_frames
                and counts_ok
                and optimizer_ok
                and runtime_ok
                and row.get("gas_model_metadata") == EXPECTED_MODEL_METADATA
            )
            results.append({"row_id": row["row_id"], "passed": passed})
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            results.append(
                {
                    "row_id": row.get("row_id"),
                    "passed": False,
                    **serialize_exception(exc),
                }
            )
    converged = sum(result["passed"] for result in results)
    return {
        "row_count": len(rows),
        "inventory_exact": exact_ids,
        "rows": results,
        "converged_count": converged,
        "status": (
            "VALIDATED" if exact_ids and converged == 15 else "GAUSSIAN_OPT_INCOMPLETE"
        ),
    }


def run(
    output: Path,
    expected_preregistration_sha256: str,
    phase: str = "all",
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
) -> dict[str, Any]:
    assert_runtime_origins(context)
    if phase not in {"crossing", "combined-fd", "opt", "all"}:
        raise ValueError(f"unknown run phase: {phase!r}")
    results = {}
    if phase in ("crossing", "all"):
        results["crossing"] = run_crossing_reference(
            output, expected_preregistration_sha256, context
        )
    if phase in ("combined-fd", "all"):
        results["combined_fd"] = run_combined_fd(
            output, expected_preregistration_sha256, context
        )
    if phase in ("opt", "all"):
        results["optimization"] = run_opt_matrix(
            output, expected_preregistration_sha256, context
        )
    phase_passed = bool(
        results
        and all(
            (
                payload.get("passed") is True
                if name == "crossing"
                else (
                    payload.get("all_passed") is True
                    if name == "combined_fd"
                    else payload.get("status") == "VALIDATED"
                )
            )
            for name, payload in results.items()
        )
    )
    results["phase"] = phase
    results["context_id"] = context.context_id
    results["numerical_profile_id"] = context.numerical_profile_id
    results["status"] = "PHASE_PASSED" if phase_passed else "GAUSSIAN_OPT_INCOMPLETE"
    return results


def validate(
    output: Path,
    expected_preregistration_sha256: str,
    context: GaussianChaRunContext = DEFAULT_V1_CONTEXT,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    protocol, prereg, evidence = load_preregistration(
        output, expected_preregistration_sha256, context
    )
    expected_rows = prereg["rows"]
    row_receipts = []
    combined_receipts = []
    for expected in expected_rows:
        row, _ = read_hashed_json(
            output / "rows" / f"{expected['row_id']}.json", "OPT row receipt"
        )
        _verify_receipt_binding(
            row,
            evidence,
            receipt_kind="OPT_ROW",
            case_id=expected["row_id"],
            case_identity=expected,
        )
        row_receipts.append(row)
        combined, _ = read_hashed_json(
            output / "combined-fd" / f"{expected['row_id']}.json",
            "combined FD receipt",
        )
        _verify_receipt_binding(
            combined,
            evidence,
            receipt_kind="COMBINED_FD",
            case_id=expected["row_id"],
            case_identity=expected,
        )
        combined_receipts.append(combined)
    event = next(row for row in expected_rows if row["center_scale"] == 1.0)
    base = np.asarray(event["positions_angstrom"], dtype=np.float64)
    crossing_receipts = []
    expected_crossing_ids = []
    for sigma_e in protocol["inputs"]["sigma_e"]:
        coordinate_cases = [
            ("scan", 0, 0, float(offset))
            for offset in np.linspace(-0.002, 0.002, 41, dtype=np.float64)
        ]
        coordinate_cases.extend(
            ("event-cartesian", flat // 3, flat % 3, 0.0) for flat in range(9)
        )
        for kind, atom, axis, offset in coordinate_cases:
            record_id = f"sigma-{sigma_e:g}-{kind}-{atom}-{axis}-{offset:+.8f}"
            expected_crossing_ids.append(record_id)
            positions = base.copy()
            positions[0, 0] += offset
            identity = {
                "record_id": record_id,
                "sigma_e": sigma_e,
                "kind": kind,
                "atom_index": atom,
                "axis": axis,
                "offset_angstrom": offset,
                "source_event_positions_angstrom": base.tolist(),
                "positions_angstrom": positions.tolist(),
                "positions_sha256": coordinate_sha256(positions),
            }
            receipt, _ = read_hashed_json(
                output / "crossing-reference" / f"{record_id}.json",
                "crossing receipt",
            )
            _verify_receipt_binding(
                receipt,
                evidence,
                receipt_kind="CROSSING_REFERENCE",
                case_id=record_id,
                case_identity=identity,
            )
            crossing_receipts.append(receipt)
    crossing = recompute_crossing_summary(
        crossing_receipts, protocol, expected_crossing_ids
    )
    combined = recompute_combined_fd_summary(combined_receipts, protocol, expected_rows)
    optimization = recompute_opt_summary(
        row_receipts,
        protocol,
        expected_rows,
        evidence["source"],
        evidence["topology"]["content_sha256"],
        context.numerical_profile_id,
    )
    _, post_protocol_digest = load_protocol_template(context)
    after = _phase_evidence(protocol, post_protocol_digest, context)
    snapshot_stable = (
        after["source_identity_sha256"] == evidence["source_identity_sha256"]
    )
    passed = bool(
        crossing["passed"]
        and combined["all_passed"]
        and optimization["status"] == "VALIDATED"
        and snapshot_stable
    )
    result = _bind_receipt(
        {
            "status": "VALIDATED" if passed else "GAUSSIAN_OPT_INCOMPLETE",
            "crossing_reference_recomputed": crossing,
            "combined_fd_recomputed": combined,
            "optimization_recomputed": optimization,
            "pre_post_phase_snapshot_stable": snapshot_stable,
            "claim_boundary": protocol["claim_boundary"],
        },
        evidence,
        receipt_kind="FINAL_VALIDATION",
        case_id="validation",
        case_identity={
            "row_ids": [row["row_id"] for row in expected_rows],
            "crossing_record_ids": expected_crossing_ids,
        },
    )
    if write_output:
        write_json_new_atomic(output / "validation.json", result)
    return result


def _output_allowed(output: Path, context: GaussianChaRunContext) -> bool:
    resolved = output.resolve()
    if context.numerical_profile_id == V2_PROFILE_ID:
        return bool(
            resolved.parent == context.output_root.resolve()
            and re.fullmatch(r"campaign-v2-\d{8}T\d{6}Z", resolved.name)
        )
    return resolved == context.output_root.resolve() or _inside(
        resolved, context.output_root
    )


def _context_for_profile(profile_id: str) -> GaussianChaRunContext:
    if profile_id == V1_PROFILE_ID:
        return DEFAULT_V1_CONTEXT
    if profile_id == V2_PROFILE_ID:
        return V2_RUN_CONTEXT
    raise ValueError(f"Unknown Gaussian-CHA numerical profile: {profile_id!r}")


def main(
    argv: list[str] | None = None, context: GaussianChaRunContext | None = None
) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--profile")
        if name in ("run", "validate"):
            command.add_argument("--expected-preregistration-sha256", required=True)
        if name == "run":
            command.add_argument(
                "--phase",
                choices=("crossing", "combined-fd", "opt", "all"),
                default="all",
            )
    args = parser.parse_args(argv)
    requested_profile = args.profile or (
        DEFAULT_V1_CONTEXT.numerical_profile_id
        if context is None
        else context.numerical_profile_id
    )
    try:
        selected_context = _context_for_profile(requested_profile)
    except ValueError as exc:
        parser.error(str(exc))
    if context is not None and selected_context != context:
        parser.error("--profile differs from the entrypoint execution context")
    if not _output_allowed(args.output, selected_context):
        parser.error(
            f"--output must be {selected_context.output_root} or a descendant campaign directory"
        )
    if args.command == "preregister":
        result = preregister(args.output, selected_context)
    elif args.command == "run":
        result = run(
            args.output,
            args.expected_preregistration_sha256,
            args.phase,
            selected_context,
        )
    else:
        result = validate(
            args.output, args.expected_preregistration_sha256, selected_context
        )
    print(json.dumps(json_safe(result), sort_keys=True))
    if args.command == "preregister":
        succeeded = result.get("phase") == "PREREGISTERED"
    elif args.command == "run":
        succeeded = result.get("status") == "PHASE_PASSED"
    else:
        succeeded = result.get("status") == "VALIDATED"
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
