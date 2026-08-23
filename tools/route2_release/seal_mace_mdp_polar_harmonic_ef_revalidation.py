#!/usr/bin/env python3
"""Create the prospective H1 v2 E/F computation seal.

The seal is deliberately narrower than an admission.  It freezes one clean
Git tree, the prepared exposure-aware protocol, executable Python sources,
model assets, provider identities, scientific settings, and the deterministic
CUDA runtime.  It never edits the repository and publishes one external JSON
file without replacement.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Mapping
from uuid import UUID

sys.dont_write_bytecode = True

SEAL_GENERATOR_ID = (
    "tools/route2_release/seal_mace_mdp_polar_harmonic_ef_revalidation.py"
)
RUNNER_ID = "tools/route2_release/run_mace_mdp_polar_harmonic_ef_revalidation.py"
AGGREGATOR_ID = (
    "tools/route2_release/aggregate_mace_mdp_polar_harmonic_ef_revalidation.py"
)
SMD_CDS_SOURCE_ID = "maple/function/calculator/extra_correction/implicit/smd_cds.py"
CAVITY_IMPLEMENTATION_SOURCE_PREFIXES = (
    "maple/solvation/continuum/",
    "maple/solvation/solvent_terms.py",
    SMD_CDS_SOURCE_ID,
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
PREREGISTRATION_SCHEMA = (
    "route2-exposure-aware-retrospective-ef-revalidation-protocol-v2"
)
PREREGISTRATION_ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-harmonic-ef-revalidation-v2"
PREREGISTRATION_STATUS = (
    "prepared-exposure-aware-retrospective-revalidation-awaiting-clean-commit-"
    "content-addressed-freeze-before-h1-seal"
)
SEAL_ARTIFACT_ID = (
    "route2-mace-mdp-polar-hybrid-harmonic-ef-revalidation-h1-v2-computation-seal"
)
EXPECTED_CAPABILITIES = {"E": True, "F": True, "H": False, "V": False, "M": False}
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
EXPECTED_PROTOCOL_CLASSIFICATION = {
    "label": "route2-h1-exposure-aware-ef-revalidation-v2",
    "is_blind_preregistration": False,
    "is_retrospective_revalidation": True,
    "freeze_boundary": (
        "This tracked draft is prepared but not frozen. Only a future clean "
        "committed protocol byte SHA256 bound by the computation seal freezes "
        "the protocol and creates the H1 identity; no cold replay, aggregate "
        "result, or admission overlay may precede that seal."
    ),
    "purpose": (
        "Prepare retrospective revalidation for the pre-H1 development checkout "
        "after acknowledged exposure to the H0/4cf8db40 outcomes; historical "
        "passes are disclosed evidence, not transferable admission."
    ),
    "is_frozen": False,
    "h1_identity_exists": False,
}
EXPECTED_IDENTITY_LIFECYCLE = (
    "Reserved contract values only; these fields do not constitute an H1 "
    "identity until a clean committed protocol byte SHA256 is bound by a valid "
    "computation seal."
)
EXPECTED_SEQUENCE = [
    "protocol",
    "computation_seal",
    "cold_replicate_a",
    "cold_replicate_b",
    "mechanical_aggregator",
    "data_only_admission_overlay",
]
EXPECTED_SEQUENCE_INVARIANTS = {
    "protocol_precedes_computation_seal": True,
    "computation_seal_precedes_both_cold_runs": True,
    "cold_runs_are_independent_processes": True,
    "mechanical_aggregator_has_no_discretionary_thresholds": True,
    "overlay_is_data_only_and_cannot_change_science": True,
    "failure_retains_negative_evidence": True,
    "failure_capabilities": NO_CAPABILITIES,
}
EXPECTED_POST_FREEZE_PROHIBITIONS = {
    "fit": False,
    "calibration": False,
    "case_selection": False,
    "threshold_changes": False,
    "panel_changes": False,
    "geometry_changes": False,
    "scientific_setting_changes": False,
    "decision_rule_changes": False,
    "post_run_protocol_changes": False,
    "policy": (
        "No fit, calibration, case selection, threshold relaxation, panel or "
        "geometry substitution, scientific-setting change, decision-rule change, "
        "or other post-run protocol change is allowed. Any such change creates "
        "a new protocol identity and requires a new seal and fresh cold runs."
    ),
}
EXPECTED_EXECUTION_STATE = {
    "computation_seal": None,
    "cold_replicate_a": None,
    "cold_replicate_b": None,
    "aggregate_result": None,
    "admission_overlay": None,
    "capabilities_currently_admitted_by_this_protocol": NO_CAPABILITIES,
}
EXPECTED_CLAIM_TEXT = {
    "candidate_admission": (
        "Only the exact experimental electrostatic scalar E and its runtime-error-"
        "bounded fourth-order Richardson numerical scalar-gradient F may become "
        "true after the complete v2 sequence passes."
    ),
    "protocol_does_not_admit": (
        "Preparing this protocol admits no capability. Historical H0 passes admit "
        "no capability for the pre-H1 development checkout. Only a future clean "
        "committed protocol SHA256 bound by a valid computation seal, two sealed "
        "cold runs, mechanical aggregation, and a valid data-only overlay may "
        "activate E/F for the resulting H1 identity."
    ),
}
EXPECTED_VERIFIED_PRO_AMENDMENT = {
    "amendment_id": "route2-rich-v2-source-space-and-h0-archive-amendment-20260823",
    "reason": (
        "Historical H0 rich leaves are non-identifiable; verified Pro approved an "
        "archive-only v1 compatibility gate and required machine-checkable rich-v2 "
        "representation contracts before H1 execution."
    ),
    "verified_pro_audit": {
        "relative_path": "docs/route2/evidence/rich-v2-source-space-pro-q3-audit.json",
        "raw_file_sha256": (
            "024e8ab3617f34f48e35fc1b88e2257ee741fac49ba0b84d9cc62f99f298d436"
        ),
        "audit_sha256": (
            "d9e1023e5c098456107a74cb665cfac893c6314787d197ba9bb994e80015f229"
        ),
        "verdict": "APPROVE",
        "terminal_marker": "MAPLE RICH V2 SOURCE SPACE AUDIT COMPLETE",
    },
    "required_contract_refinements": [
        "bind-source-and-receiver-space-convention-digests",
        "type-audit-coefficient-sum-as-nonoperational-and-forbid-dispatch",
        "kernel-tag-permanent-point-and-induced-gto-branches-separately",
        "retain-polar-final-and-zero-reference-arrays-and-derive-induced",
        "limit-replay-claim-to-endpoint-defined-gates",
        "bind-state-specific-vacuum-energy-and-add-exactly-once",
        "separate-state-content-from-solve-occurrence-identity",
        "use-acyclic-hash-dag-and-recompute-prepared-inputs-from-captured-objects",
    ],
    "historical_h0_archive_policy": {
        "real_h0_v1_bytes_and_digest": "pin-exactly-as-observed-archive-evidence",
        "historical_rich_preimage": "non-identifiable-do-not-fabricate",
        "synthetic_rich_golden": "adapter-compatibility-only",
        "archive_gate": "independent-of-h1-admission",
        "admission_transfer_to_h1": False,
    },
    "h1_admission_requirement": (
        "two-independent-clean-sealed-replays-plus-mechanical-aggregation-and-"
        "data-only-overlay"
    ),
    "capability_effect_before_h1_overlay": "none",
}
EXPECTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_id",
        "schema_version",
        "status",
        "protocol_classification",
        "prospective_h1_identity_contract",
        "target_capabilities",
        "parent_bindings",
        "exposure_inventory",
        "historical_lineage_audit",
        "inherited_v1_contract",
        "prospective_h1_scientific_settings",
        "runtime_guards",
        "domain_guards",
        "claim_boundary",
        "frozen_execution_sequence",
        "sequence_invariants",
        "post_freeze_prohibitions",
        "execution_state",
        "verified_pro_schema_amendment",
    }
)
EXPECTED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SealInputError(ValueError):
    """Raised when the H1 identity cannot be frozen without ambiguity."""


@dataclass(frozen=True, slots=True)
class _FileCapture:
    path: Path
    identity: tuple[int, int, int, int, int, int, int]
    data: bytes
    sha256: str

    def assert_stable(self, *, role: str) -> None:
        if _file_identity(self.path, role=role) != self.identity:
            raise SealInputError(f"{role} changed while the computation seal was built")


@dataclass(frozen=True, slots=True)
class _SealStability:
    root: Path
    head: str
    tree: str
    captures: tuple[tuple[str, _FileCapture], ...]

    def assert_stable(self) -> None:
        if _git(self.root, "rev-parse", "HEAD").decode().strip() != self.head:
            raise SealInputError("repository HEAD changed while sealing")
        if _git(self.root, "rev-parse", "HEAD^{tree}").decode().strip() != self.tree:
            raise SealInputError("repository tree changed while sealing")
        if _git(self.root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise SealInputError("repository changed while sealing")
        for role, capture in self.captures:
            capture.assert_stable(role=role)


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments), capture_output=True, check=False
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise SealInputError(f"Git {arguments!r} failed: {detail.strip()}")
    return result.stdout


def _reject_symlinks(path: Path, *, role: str) -> None:
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            value = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise SealInputError(f"cannot inspect {role}: {exc}") from exc
        if stat.S_ISLNK(value.st_mode):
            raise SealInputError(f"{role} path must not contain symbolic links")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_identity(
    path: Path, *, role: str
) -> tuple[int, int, int, int, int, int, int]:
    _reject_symlinks(path, role=role)
    try:
        result = path.stat()
    except OSError as exc:
        raise SealInputError(f"cannot inspect {role}: {exc}") from exc
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise SealInputError(f"{role} must be a non-hard-linked regular file")
    return _identity(result)


def _capture(path: str | Path, *, role: str) -> _FileCapture:
    lexical = Path(os.path.abspath(os.path.expanduser(path)))
    _reject_symlinks(lexical, role=role)
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = _identity(os.fstat(descriptor))
        if not stat.S_ISREG(before[2]) or before[3] != 1:
            raise SealInputError(f"{role} must be a non-hard-linked regular file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = _identity(os.fstat(descriptor))
    except SealInputError:
        raise
    except OSError as exc:
        raise SealInputError(f"cannot read {role}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    data = b"".join(blocks)
    if before != after or len(data) != before[4]:
        raise SealInputError(f"{role} changed while being read")
    resolved = lexical.resolve(strict=True)
    if _file_identity(resolved, role=role) != before:
        raise SealInputError(f"{role} path changed while being read")
    return _FileCapture(resolved, before, data, hashlib.sha256(data).hexdigest())


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SealInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(capture: _FileCapture, *, role: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            capture.data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SealInputError(f"non-finite JSON number in {role}: {value}")
            ),
        )
    except SealInputError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SealInputError(f"cannot parse {role}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SealInputError(f"{role} must contain one JSON object")
    return value


def _repo_relative(path: Path, root: Path, *, role: str) -> str:
    try:
        text = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SealInputError(f"{role} must be inside the repository") from exc
    parsed = PurePosixPath(text)
    if (
        not text
        or parsed.is_absolute()
        or any(p in {"", ".", ".."} for p in parsed.parts)
    ):
        raise SealInputError(f"{role} has a non-canonical repository path")
    return text


def _clean_repository(root: str | Path) -> tuple[Path, str, str, bytes]:
    lexical = Path(os.path.abspath(os.path.expanduser(root)))
    _reject_symlinks(lexical, role="repository root")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise SealInputError("repository root must be a directory")
    top = Path(
        _git(resolved, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if top != resolved:
        raise SealInputError("--repo-root must be the exact Git worktree root")
    status = _git(resolved, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SealInputError("computation sealing requires a clean Git worktree")
    head = _git(resolved, "rev-parse", "HEAD").decode().strip()
    tree = _git(resolved, "rev-parse", "HEAD^{tree}").decode().strip()
    return resolved, head, tree, status


def _running_repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _require_running_root(root: Path) -> None:
    if _running_repository_root() != root:
        raise SealInputError(
            "running seal generator does not originate from --repo-root"
        )


def _committed_capture(
    root: Path,
    head: str,
    path: str | Path,
    *,
    role: str,
    expected_relative: str | None = None,
) -> tuple[str, _FileCapture]:
    capture = _capture(path, role=role)
    relative = _repo_relative(capture.path, root, role=role)
    if expected_relative is not None and relative != expected_relative:
        raise SealInputError(f"{role} must be {expected_relative}")
    tracked = set(
        _git(root, "ls-tree", "-r", "--name-only", head).decode().splitlines()
    )
    if relative not in tracked:
        raise SealInputError(f"{role} must be committed at clean HEAD")
    committed = _git(root, "show", f"{head}:{relative}")
    if committed != capture.data:
        raise SealInputError(f"{role} bytes differ from clean HEAD")
    return relative, capture


def _digest(value: object, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SealInputError(f"{role} must be a lowercase SHA256")
    return value


def _require_fresh_execution_process() -> None:
    """Reject scientific runtimes imported before deterministic setup."""

    if os.environ.get("PYTHONPYCACHEPREFIX") is not None or sys.pycache_prefix:
        raise SealInputError(
            "PYTHONPYCACHEPREFIX and -X pycache_prefix are forbidden for a "
            "source-bound computation seal"
        )

    forbidden = (
        "numpy",
        "torch",
        "mace",
        "graph_longrange",
        "maple.function.calculator.mace",
        "maple.solvation.models",
        "maple.solvation.continuum",
        "maple.solvation.experimental",
    )
    preloaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    if preloaded:
        raise SealInputError(
            "scientific runtime/provider modules were preloaded before deterministic "
            "configuration: " + ", ".join(preloaded[:12])
        )


def _module_python_origin(module: ModuleType) -> Path | None:
    file_value = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_value = getattr(spec, "origin", None)
    candidates = [value for value in (file_value, spec_value) if isinstance(value, str)]
    python_candidates: list[Path] = []
    for value in candidates:
        if value.endswith(".py"):
            source = value
        elif value.endswith(".pyc"):
            try:
                source = importlib.util.source_from_cache(value)
            except ValueError as exc:
                raise SealInputError(
                    "loaded module has a noncanonical pyc origin"
                ) from exc
        else:
            continue
        python_candidates.append(Path(source).resolve(strict=True))
    if not python_candidates:
        return None
    if len(set(python_candidates)) != 1:
        raise SealInputError("loaded module __file__ and spec origin disagree")
    return python_candidates[0]


def _assert_no_unsealed_bytecode(module: ModuleType, *, module_name: str) -> None:
    """Reject executed or eligible bytecode outside the source-byte ledger."""

    spec = getattr(module, "__spec__", None)
    origins = (
        ("__file__", getattr(module, "__file__", None), True),
        ("__spec__.origin", getattr(spec, "origin", None), True),
        ("__cached__", getattr(module, "__cached__", None), False),
    )
    for label, value, executed_origin in origins:
        if not isinstance(value, str) or not value.endswith(".pyc"):
            continue
        candidate = Path(value).resolve(strict=False)
        if not executed_origin and not candidate.exists():
            continue
        raise SealInputError(
            f"unsealed bytecode is not admissible for loaded module "
            f"{module_name}: {label}={candidate}"
        )


def _assert_namespace_package_bound(
    root: Path,
    module_name: str,
    module: ModuleType,
    source_ledger: Mapping[str, str],
) -> None:
    """Accept only a source-bearing namespace package rooted exactly in the repo."""

    paths = getattr(module, "__path__", None)
    if paths is None:
        raise SealInputError(
            f"loaded MAPLE/tool module has no repository Python source: {module_name}"
        )
    try:
        resolved_paths = tuple(Path(value).resolve(strict=True) for value in paths)
    except (TypeError, OSError) as exc:
        raise SealInputError(f"loaded namespace package path is invalid: {module_name}") from exc
    expected = (root / PurePosixPath(*module_name.split("."))).resolve(strict=True)
    if not resolved_paths or any(path != expected for path in resolved_paths):
        raise SealInputError(
            f"loaded namespace package escapes its exact repository path: {module_name}"
        )
    _reject_symlinks(expected, role=f"loaded namespace package {module_name}")
    if (expected / "__init__.py").exists():
        raise SealInputError(
            f"loaded source package unexpectedly lacks an origin: {module_name}"
        )
    relative = _repo_relative(expected, root, role=f"loaded namespace package {module_name}")
    prefix = relative + "/"
    if not any(
        path.startswith(prefix) and path.endswith(".py") for path in source_ledger
    ):
        raise SealInputError(
            f"loaded namespace package has no tracked Python children: {module_name}"
        )


def _assert_loaded_repo_modules_bound(
    root: Path, head: str, source_ledger: Mapping[str, str]
) -> None:
    """Bind every executed repository-local MAPLE/tool Python module."""

    for name, module in sorted(sys.modules.items()):
        if not (
            name == "maple"
            or name.startswith("maple.")
            or name == "tools"
            or name.startswith("tools.")
        ):
            continue
        if not isinstance(module, ModuleType):
            continue
        origin = _module_python_origin(module)
        if origin is None:
            if (
                name == "maple"
                or name.startswith("maple.")
                or name.startswith("tools.route2_release")
            ):
                _assert_namespace_package_bound(
                    root,
                    name,
                    module,
                    source_ledger,
                )
                continue
            continue
        relative = _repo_relative(origin, root, role=f"loaded module {name}")
        _assert_no_unsealed_bytecode(module, module_name=name)
        expected = source_ledger.get(relative)
        if expected is None:
            raise SealInputError(
                f"loaded module is not tracked in source ledger: {name}"
            )
        capture = _capture(origin, role=f"loaded module {name}")
        committed = _git(root, "show", f"{head}:{relative}")
        if capture.data != committed or capture.sha256 != expected:
            raise SealInputError(f"loaded module differs from committed source: {name}")


def _validate_preregistration(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    if frozenset(payload) != EXPECTED_TOP_LEVEL_FIELDS:
        raise SealInputError(
            "preregistration top-level shape is not the current v2 protocol"
        )
    expected = {
        "artifact_id": PREREGISTRATION_ARTIFACT_ID,
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": PREREGISTRATION_STATUS,
        "target_capabilities": EXPECTED_CAPABILITIES,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SealInputError(
                f"preregistration {key} is not the current v2 contract"
            )
    if payload.get("protocol_classification") != EXPECTED_PROTOCOL_CLASSIFICATION:
        raise SealInputError("preregistration exposure classification is invalid")
    identity = payload.get("prospective_h1_identity_contract")
    science = payload.get("prospective_h1_scientific_settings")
    claim = payload.get("claim_boundary")
    parents = payload.get("parent_bindings")
    if not all(
        isinstance(item, Mapping) for item in (identity, science, claim, parents)
    ):
        raise SealInputError(
            "preregistration identity/science/claim/parent sections are invalid"
        )
    if identity.get("lifecycle") != EXPECTED_IDENTITY_LIFECYCLE:
        raise SealInputError(
            "preregistration prospective identity lifecycle is invalid"
        )
    if payload.get("frozen_execution_sequence") != EXPECTED_SEQUENCE:
        raise SealInputError("preregistration frozen execution sequence is invalid")
    if payload.get("sequence_invariants") != EXPECTED_SEQUENCE_INVARIANTS:
        raise SealInputError("preregistration sequence invariants are invalid")
    if payload.get("post_freeze_prohibitions") != EXPECTED_POST_FREEZE_PROHIBITIONS:
        raise SealInputError("preregistration post-freeze prohibitions are invalid")
    if payload.get("execution_state") != EXPECTED_EXECUTION_STATE:
        raise SealInputError("preregistration execution state is not pre-seal")
    if payload.get("verified_pro_schema_amendment") != EXPECTED_VERIFIED_PRO_AMENDMENT:
        raise SealInputError("preregistration verified-Pro amendment is invalid")
    if any(claim.get(key) != value for key, value in EXPECTED_CLAIM_TEXT.items()):
        raise SealInputError(
            "preregistration claim text is not the frozen E/F boundary"
        )
    parent_panel = parents.get("v1_parent_panel")
    inherited = parents.get("v1_preregistration")
    if not isinstance(parent_panel, Mapping) or not isinstance(inherited, Mapping):
        raise SealInputError("preregistration parent bindings are incomplete")
    panel_sha = _digest(parent_panel.get("sha256"), role="parent panel binding")
    inherited_sha = _digest(inherited.get("sha256"), role="v1 preregistration binding")
    runtime = payload.get("runtime_guards")
    domain = payload.get("domain_guards")
    non_admissions = claim.get("non_admissions")
    if not all(isinstance(item, list) for item in (runtime, domain, non_admissions)):
        raise SealInputError("preregistration guards/non-admissions are invalid")
    # ComputationSealV2 below is the authoritative exact science validator.
    return dict(science), panel_sha, inherited_sha


def _reviewed_relative(value: object, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise SealInputError(f"{role} path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SealInputError(f"{role} path is not canonical repository-relative")
    return value


def _validate_verified_pro_amendment_artifacts(
    root: Path, head: str, amendment: Mapping[str, object]
) -> dict[str, str]:

    if amendment != EXPECTED_VERIFIED_PRO_AMENDMENT:
        raise SealInputError("verified-Pro amendment differs from the frozen contract")
    binding = amendment["verified_pro_audit"]
    if not isinstance(binding, Mapping):
        raise SealInputError("verified-Pro audit binding is invalid")
    audit_relative = _reviewed_relative(
        binding["relative_path"], role="verified-Pro audit manifest"
    )
    _, audit_capture = _committed_capture(
        root,
        head,
        root / audit_relative,
        role="verified-Pro audit manifest",
        expected_relative=audit_relative,
    )
    if audit_capture.sha256 != binding["raw_file_sha256"]:
        raise SealInputError("verified-Pro audit manifest bytes differ from binding")
    audit = _read_json(audit_capture, role="verified-Pro audit manifest")
    audit_payload = dict(audit)
    internal_sha256 = audit_payload.pop("audit_sha256", None)
    if internal_sha256 != binding["audit_sha256"] or _canonical_hash(
        audit_payload
    ) != internal_sha256:
        raise SealInputError("verified-Pro audit manifest self-hash is invalid")
    response = audit.get("response")
    prompt = audit.get("prompt")
    mode = audit.get("mode_verification")
    if not all(isinstance(item, Mapping) for item in (response, prompt, mode)):
        raise SealInputError("verified-Pro audit document bindings are incomplete")
    response_relative = _reviewed_relative(
        response["relative_path"], role="verified-Pro response"
    )
    prompt_relative = _reviewed_relative(
        prompt["relative_path"], role="verified-Pro prompt"
    )
    mode_relative = _reviewed_relative(
        mode["relative_path"], role="verified-Pro mode verification"
    )
    captures = {
        "response": _committed_capture(
            root,
            head,
            root / response_relative,
            role="verified-Pro response",
            expected_relative=response_relative,
        )[1],
        "prompt": _committed_capture(
            root,
            head,
            root / prompt_relative,
            role="verified-Pro prompt",
            expected_relative=prompt_relative,
        )[1],
        "mode": _committed_capture(
            root,
            head,
            root / mode_relative,
            role="verified-Pro mode verification",
            expected_relative=mode_relative,
        )[1],
    }
    for role, capture in captures.items():
        record = {"response": response, "prompt": prompt, "mode": mode}[role]
        if capture.sha256 != record["sha256"]:
            raise SealInputError(f"verified-Pro {role} bytes differ from audit binding")
    response_text = captures["response"].data.decode("utf-8")
    mode_text = captures["mode"].data.decode("utf-8")
    if (
        response.get("verdict") != "APPROVE"
        or "**APPROVE**" not in response_text
        or str(binding["terminal_marker"]) not in response_text
    ):
        raise SealInputError("verified-Pro response lacks its approved terminal record")
    for exact_line in ("composer=Pro", "power=Pro, 5 of 5."):
        if exact_line not in mode_text.splitlines():
            raise SealInputError("verified-Pro mode verification is incomplete")
    return {
        "verified_pro_audit": audit_capture.sha256,
        "verified_pro_prompt": captures["prompt"].sha256,
        "verified_pro_response": captures["response"].sha256,
        "verified_pro_mode_verification": captures["mode"].sha256,
    }


def _selected_gpu_identity(torch_module: object) -> dict[str, object]:
    cuda = getattr(torch_module, "cuda")
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,name,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode:
        raise SealInputError(
            f"nvidia-smi identity query failed: {query.stderr.strip()}"
        )
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    if not rows:
        raise SealInputError("nvidia-smi returned no GPU identities")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 4:
        raise SealInputError("nvidia-smi returned an unexpected GPU identity row")
    smi_uuid, driver_version, smi_name, smi_capability = fields
    properties = cuda.get_device_properties(0)
    torch_uuid = getattr(properties, "uuid", None)
    if torch_uuid is None:
        uuid_method = getattr(cuda, "get_device_uuid", None)
        torch_uuid = uuid_method(0) if callable(uuid_method) else None
    torch_uuid = str(torch_uuid)
    torch_uuid_suffix = torch_uuid[4:] if torch_uuid.startswith("GPU-") else torch_uuid
    try:
        torch_uuid = "GPU-" + str(UUID(torch_uuid_suffix))
    except ValueError as exc:
        raise SealInputError("Torch device UUID is not canonical") from exc
    torch_name = cuda.get_device_name(0)
    torch_capability = ".".join(str(value) for value in cuda.get_device_capability(0))
    torch_sm_count = int(properties.multi_processor_count)
    if (
        torch_uuid != smi_uuid
        or torch_name != smi_name
        or torch_capability != smi_capability
    ):
        raise SealInputError("nvidia-smi selected GPU differs from Torch device 0")
    return {
        "driver_version": driver_version,
        "device_uuid": torch_uuid,
        "device_capability": torch_capability,
        "device_multiprocessor_count": torch_sm_count,
    }


def _configure_and_capture_runtime() -> dict[str, object]:
    for key, expected in EXPECTED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise SealInputError(
                f"environment {key} must equal {expected!r} before imports"
            )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise SealInputError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    seed = os.environ.get("PYTHONHASHSEED")
    if seed is None or re.fullmatch(r"[0-9]+", seed) is None:
        raise SealInputError("PYTHONHASHSEED must be an explicit decimal integer")

    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise SealInputError("the H1 v2 runtime requires CUDA")
    if torch.cuda.device_count() != 1:
        raise SealInputError("CUDA_VISIBLE_DEVICES=0 must expose exactly one Torch GPU")
    gpu = _selected_gpu_identity(torch)

    from maple.solvation.release.evidence import runtime_record

    raw = runtime_record()
    numpy_record = raw.get("numpy")
    torch_record = raw.get("torch")
    if not isinstance(numpy_record, Mapping) or not isinstance(torch_record, Mapping):
        raise SealInputError("runtime_record did not capture NumPy and Torch")
    torch_record = {
        **torch_record,
        "interop_threads": torch.get_num_interop_threads(),
        "driver_version": gpu["driver_version"],
        "device_uuids": [gpu["device_uuid"]],
        "device_capabilities": [gpu["device_capability"]],
        "device_multiprocessor_counts": [gpu["device_multiprocessor_count"]],
    }
    fingerprint = {
        "python": str(raw["python"]).split()[0],
        "implementation": raw["implementation"],
        "platform": raw["platform"],
        "machine": raw["machine"],
        "cpu_model": raw["cpu_model"],
        "packages": raw["packages"],
        "numpy": dict(numpy_record),
        "torch": torch_record,
        "environment": raw["environment"],
        "execution_device": "cuda",
        "execution_dtype": "float64",
    }
    return fingerprint


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _relevant_source_ledger(
    source_ledger: Mapping[str, str], prefixes: tuple[str, ...]
) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(source_ledger.items())
        if key.startswith(prefixes)
    }


def _cavity_system_manifest(
    *,
    system_id: str,
    atomic_numbers: tuple[int, ...],
    chemical_symbols: tuple[str, ...],
    positions_angstrom: tuple[tuple[float, ...], ...],
    radii_angstrom: tuple[float, ...],
    radii_provider_id: str,
    source_asset: Mapping[str, str],
    legacy_projection_artifact_sha256: str | None = None,
    legacy_projection_radii_angstrom: tuple[float, ...] | None = None,
    legacy_projection_maximum_absolute_difference_angstrom: float | None = None,
) -> dict[str, object]:
    geometry_sha256 = _canonical_hash(
        {
            "atomic_numbers": list(atomic_numbers),
            "positions_angstrom": [list(row) for row in positions_angstrom],
        }
    )
    canonical_input = {
        "contract": "route2-h1-v2-canonical-cavity-input-v1",
        "system_id": system_id,
        "radii_provider_id": radii_provider_id,
        "atomic_numbers": list(atomic_numbers),
        "chemical_symbols": list(chemical_symbols),
        "geometry_sha256": geometry_sha256,
        "radii_angstrom": list(radii_angstrom),
        "source_asset": dict(source_asset),
    }
    return {
        "system_id": system_id,
        "atomic_numbers": list(atomic_numbers),
        "radii_angstrom": list(radii_angstrom),
        "radii_provider_id": radii_provider_id,
        "geometry_sha256": geometry_sha256,
        "canonical_input": canonical_input,
        "input_sha256": _canonical_hash(canonical_input),
        "legacy_projection_artifact_sha256": legacy_projection_artifact_sha256,
        "legacy_projection_radii_angstrom": (
            list(legacy_projection_radii_angstrom)
            if legacy_projection_radii_angstrom is not None
            else None
        ),
        "legacy_projection_maximum_absolute_difference_angstrom": (
            legacy_projection_maximum_absolute_difference_angstrom
        ),
        "legacy_projection_role": (
            "provenance-only-not-the-h1-cavity-input"
            if legacy_projection_artifact_sha256 is not None
            else None
        ),
    }


def _prepared_input_manifest(
    *,
    compound_id: str,
    name: str,
    benzene_atomic_numbers: tuple[int, ...],
    benzene_positions_angstrom: tuple[tuple[float, ...], ...],
    benzene_radii_angstrom: tuple[float, ...],
    benzene_mol2_sha256: str,
    benzene_projection_sha256: str,
    predecessor_dof: tuple[int, int],
    water_positions_angstrom: tuple[tuple[float, ...], ...],
    water_radii_angstrom: tuple[float, ...],
    v1_preregistration_sha256: str,
    parent_panel_sha256: str,
    force_panel_contract: Mapping[str, object],
    force_panel_contract_sha256: str,
) -> dict[str, object]:
    """Return the exact JSON preimage sealed for prepared H1 inputs."""

    return {
        "benzene": {
            "compound_id": compound_id,
            "name": name,
            "atomic_numbers": list(benzene_atomic_numbers),
            "positions_angstrom": [list(row) for row in benzene_positions_angstrom],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": list(benzene_radii_angstrom),
            "mol2_sha256": benzene_mol2_sha256,
            "projection_result_sha256": benzene_projection_sha256,
            "predecessor_dof": list(predecessor_dof),
        },
        "water": {
            "atomic_numbers": [8, 1, 1],
            "positions_angstrom": [list(row) for row in water_positions_angstrom],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": list(water_radii_angstrom),
        },
        "v1_preregistration_sha256": v1_preregistration_sha256,
        "parent_panel_sha256": parent_panel_sha256,
        "force_panel_contract": json.loads(
            json.dumps(force_panel_contract, sort_keys=True, allow_nan=False)
        ),
        "force_panel_contract_sha256": force_panel_contract_sha256,
    }


def _provider_configuration_factory(
    mace_mdp_checkpoint_bytes: bytes,
    mace_polar_checkpoint_bytes: bytes,
    expected_mace_mdp_sha256: str,
    expected_mace_polar_sha256: str,
    benzene_mol2: _FileCapture,
    benzene_projection: _FileCapture,
    force_panel_contract: Mapping[str, object],
    v1_preregistration_sha256: str,
    parent_panel_sha256: str,
    scientific_settings: Mapping[str, object],
    source_ledger: Mapping[str, str],
) -> dict[str, str]:
    """Build exact checkpoint and frozen-panel provider manifests."""
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.function.read.filereader.mol2_reader import mol2_atoms_from_bytes
    from maple.solvation.continuum.harmonic_torch_functional import (
        SmoothWeightedHarmonicGalerkinFunctionalCandidate,
    )
    from maple.solvation.coupling.metrics import (
        atomic_l1_source_convention_contract_sha256,
    )
    from maple.solvation.coupling.separated_operators import (
        mace_polar_native_field_convention_contract_sha256,
    )
    from maple.solvation.derivatives import RichardsonScalarForce
    from maple.solvation.experimental.mace_mdp_polar_harmonic import (
        MACE_MDPPolarHybridSmoothHarmonicPES,
        SCALAR_ID,
    )
    from maple.solvation.release.harmonic_ef_measurement import (
        H1PreparedHarmonicEFPESV2,
        PreparedHarmonicEFBenzeneInputV2,
        PreparedHarmonicEFWaterInputV2,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_anchored_mace_polar_hybrid,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )
    import numpy as np
    import torch

    permanent = build_mace_mdp_moment_adapter(
        checkpoint_bytes=mace_mdp_checkpoint_bytes, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_bytes=mace_polar_checkpoint_bytes,
        device="cuda",
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=permanent, response=response
    )
    source_receiver = scientific_settings["source_receiver"]
    expected_spaces = {
        "source_space_id": hybrid.source_space.scalar_id,
        "source_space_contract_sha256": (
            atomic_l1_source_convention_contract_sha256()
        ),
        "receiver_space_id": hybrid.receiver_space.space_id,
        "receiver_space_contract_sha256": (
            mace_polar_native_field_convention_contract_sha256()
        ),
    }
    if any(source_receiver.get(key) != value for key, value in expected_spaces.items()):
        raise SealInputError(
            "preregistered source/receiver convention does not match the captured "
            "providers"
        )
    if permanent.checkpoint_sha256 != expected_mace_mdp_sha256:
        raise SealInputError("MACE-MDP provider reports a different checkpoint SHA256")
    radial_checkpoint = getattr(
        getattr(radial, "provenance", None), "checkpoint_sha256", None
    )
    if radial_checkpoint != expected_mace_polar_sha256:
        raise SealInputError(
            "MACE-POLAR provider reports a different checkpoint SHA256"
        )

    benzene = mol2_atoms_from_bytes(
        benzene_mol2.data,
        source_id=f"captured-sha256:{benzene_mol2.sha256}",
        charge=0,
        mult=1,
    )
    if not np.all(np.isfinite(np.asarray(benzene.positions, dtype=float))):
        raise SealInputError("captured benzene MOL2 coordinates are non-finite")
    projection_payload = _read_json(
        benzene_projection, role="benzene frozen projection result"
    )
    projection_inputs = projection_payload.get("inputs")
    parsed_pcm = (
        projection_inputs.get("parsed_pcm_input")
        if isinstance(projection_inputs, Mapping)
        else None
    )
    if not isinstance(parsed_pcm, Mapping):
        raise SealInputError("benzene projection omits parsed PCM input")
    benzene_radii = np.asarray(parsed_pcm.get("cavity_radii_angstrom"), dtype=float)
    if benzene_radii.shape != (len(benzene),) or not np.all(np.isfinite(benzene_radii)):
        raise SealInputError("benzene projection cavity radii are invalid")
    canonical_benzene_radii = np.asarray(
        smd_water_coulomb_radii(benzene.get_chemical_symbols()), dtype=float
    )
    if canonical_benzene_radii.shape != benzene_radii.shape:
        raise SealInputError("canonical benzene cavity radii shape is invalid")
    water_positions = np.asarray(
        force_panel_contract["water_geometry_angstrom"], dtype=float
    )
    if water_positions.shape != (3, 3) or not np.all(np.isfinite(water_positions)):
        raise SealInputError("preregistered water geometry must be finite (3,3)")
    water_radii = tuple(
        float(value) for value in smd_water_coulomb_radii(("O", "H", "H"))
    )
    systems = (
        (
            "benzene",
            tuple(int(value) for value in benzene.numbers),
            tuple(str(value) for value in benzene.get_chemical_symbols()),
            tuple(tuple(float(value) for value in row) for row in benzene.positions),
            tuple(float(value) for value in canonical_benzene_radii),
            {
                "role": "benzene_mol2",
                "sha256": benzene_mol2.sha256,
            },
        ),
        (
            "water",
            (8, 1, 1),
            ("O", "H", "H"),
            tuple(tuple(float(value) for value in row) for row in water_positions),
            water_radii,
            {
                "role": "preregistered_water_geometry",
                "sha256": _canonical_hash(
                    {
                        "atomic_numbers": [8, 1, 1],
                        "positions_angstrom": [list(row) for row in water_positions],
                    }
                ),
            },
        ),
    )
    continuum = scientific_settings["continuum"]
    manifests: list[dict[str, object]] = []
    cavity_manifests: list[dict[str, object]] = []
    for system_id, numbers, symbols, positions, radii, source_asset in systems:
        functional = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
            atomic_numbers=numbers,
            radii_angstrom=radii,
            transition_width_angstrom2=continuum["transition_width_angstrom2"],
            surface_lmax=continuum["surface_lmax"],
            exposure_lmax=continuum["exposure_lmax"],
            exposure_radial_quadrature_order=continuum[
                "exposure_radial_quadrature_order"
            ],
            source_radial_quadrature_order=continuum["source_radial_quadrature_order"],
            green_radial_quadrature_order=continuum["green_radial_quadrature_order"],
            dtype=torch.float64,
            device="cuda",
            scalar_id=SCALAR_ID,
        )
        manifests.append(
            {
                "system_id": system_id,
                "provider_id": functional.provider_id,
                "scalar_id": functional.scalar_id,
                "continuum_profile_id": functional.continuum_profile_id,
                "cavity_profile_id": functional.cavity_profile_id,
                "atomic_numbers": list(numbers),
                "radii_angstrom": list(radii),
                "configuration_sha256": functional.configuration_sha256(),
                "provenance_sha256": functional.provenance_sha256,
                "topology_sha256": functional.topology_sha256(),
            }
        )
        cavity_manifests.append(
            _cavity_system_manifest(
                system_id=system_id,
                atomic_numbers=numbers,
                chemical_symbols=symbols,
                positions_angstrom=positions,
                radii_angstrom=radii,
                radii_provider_id=scientific_settings["cavity"]["radii_provider_id"],
                source_asset=source_asset,
                legacy_projection_artifact_sha256=(
                    benzene_projection.sha256 if system_id == "benzene" else None
                ),
                legacy_projection_radii_angstrom=(
                    tuple(float(value) for value in benzene_radii)
                    if system_id == "benzene"
                    else None
                ),
                legacy_projection_maximum_absolute_difference_angstrom=(
                    float(np.max(np.abs(benzene_radii - canonical_benzene_radii)))
                    if system_id == "benzene"
                    else None
                ),
            )
        )
    implementation_sources = _relevant_source_ledger(
        source_ledger,
        CAVITY_IMPLEMENTATION_SOURCE_PREFIXES,
    )
    configuration_manifest = {
        "contract": "route2-h1-v2-frozen-panel-continuum-manifest-v1",
        "scientific_settings": scientific_settings["continuum"],
        "source_receiver_settings": scientific_settings["source_receiver"],
        "systems": manifests,
    }
    continuum_configuration_sha = _canonical_hash(configuration_manifest)
    force_panel_contract_sha = _canonical_hash(force_panel_contract)
    mol2_metadata = benzene.info.get("mol2")
    benzene_name = (
        mol2_metadata.get("name") if isinstance(mol2_metadata, Mapping) else None
    )
    if benzene_name != "benzene":
        raise SealInputError("captured benzene MOL2 name is not exact benzene")
    predecessor = force_panel_contract.get("gepol_regression_cartesian_dof")
    if predecessor != [0, 0]:
        raise SealInputError("force-panel predecessor DOF is not [0, 0]")
    prepared_manifest = _prepared_input_manifest(
        compound_id="mobley_3053621",
        name=benzene_name,
        benzene_atomic_numbers=tuple(int(value) for value in benzene.numbers),
        benzene_positions_angstrom=tuple(
            tuple(float(value) for value in row) for row in benzene.positions
        ),
        benzene_radii_angstrom=tuple(float(value) for value in canonical_benzene_radii),
        benzene_mol2_sha256=benzene_mol2.sha256,
        benzene_projection_sha256=benzene_projection.sha256,
        predecessor_dof=(0, 0),
        water_positions_angstrom=tuple(
            tuple(float(value) for value in row) for row in water_positions
        ),
        water_radii_angstrom=water_radii,
        v1_preregistration_sha256=v1_preregistration_sha256,
        parent_panel_sha256=parent_panel_sha256,
        force_panel_contract=force_panel_contract,
        force_panel_contract_sha256=force_panel_contract_sha,
    )
    continuum_settings = scientific_settings["continuum"]
    force_settings = scientific_settings["force_stencil"]

    def prepared_pes(numbers: tuple[int, ...], radii: tuple[float, ...]):
        return MACE_MDPPolarHybridSmoothHarmonicPES(
            hybrid=hybrid,
            atomic_numbers=numbers,
            cavity_radii_angstrom=radii,
            dtype=torch.float64,
            device="cuda",
            transition_width_angstrom2=float(
                continuum_settings["transition_width_angstrom2"]
            ),
            surface_lmax=int(continuum_settings["surface_lmax"]),
            exposure_lmax=int(continuum_settings["exposure_lmax"]),
            exposure_radial_quadrature_order=int(
                continuum_settings["exposure_radial_quadrature_order"]
            ),
            source_radial_quadrature_order=int(
                continuum_settings["source_radial_quadrature_order"]
            ),
            green_radial_quadrature_order=int(
                continuum_settings["green_radial_quadrature_order"]
            ),
            force_backend=RichardsonScalarForce(
                coarse_step_angstrom=float(force_settings["coarse_step_angstrom"]),
                maximum_error_eV_per_A=float(
                    force_settings["maximum_local_error_ev_per_angstrom"]
                ),
            ),
        )

    benzene_pes = prepared_pes(
        tuple(int(value) for value in benzene.numbers),
        tuple(float(value) for value in canonical_benzene_radii),
    )
    water_pes = prepared_pes(
        (8, 1, 1),
        tuple(float(value) for value in water_radii),
    )
    benzene_prepared = PreparedHarmonicEFBenzeneInputV2(
        compound_id="mobley_3053621",
        name=benzene_name,
        atomic_numbers=tuple(int(value) for value in benzene.numbers),
        positions_angstrom=tuple(
            tuple(float(value) for value in row) for row in benzene.positions
        ),
        charge=0,
        multiplicity=1,
        cavity_radii_angstrom=tuple(float(value) for value in canonical_benzene_radii),
        mol2_sha256=benzene_mol2.sha256,
        projection_result_sha256=benzene_projection.sha256,
        predecessor_dof=(0, 0),
    )
    water_prepared = PreparedHarmonicEFWaterInputV2(
        atomic_numbers=(8, 1, 1),
        positions_angstrom=tuple(
            tuple(float(value) for value in row) for row in water_positions
        ),
        charge=0,
        multiplicity=1,
        cavity_radii_angstrom=tuple(float(value) for value in water_radii),
    )
    benzene_h1_pes = H1PreparedHarmonicEFPESV2(
        benzene_pes, benzene_prepared, system_role="benzene"
    )
    water_h1_pes = H1PreparedHarmonicEFPESV2(
        water_pes, water_prepared, system_role="water"
    )
    return {
        "hybrid_configuration": hybrid.configuration_sha256(),
        "hybrid_provenance": hybrid.provenance_sha256,
        "continuum_configuration": continuum_configuration_sha,
        "continuum_provenance": _canonical_hash(
            {
                "contract": "route2-h1-v2-frozen-panel-continuum-provenance-v1",
                "configuration_sha256": continuum_configuration_sha,
                "systems": [
                    {
                        "system_id": item["system_id"],
                        "provider_id": item["provider_id"],
                        "provenance_sha256": item["provenance_sha256"],
                    }
                    for item in manifests
                ],
                "implementation_source_sha256s": implementation_sources,
            }
        ),
        "cavity_configuration": _canonical_hash(
            {
                "contract": "route2-h1-v2-frozen-panel-cavity-manifest-v1",
                "cavity": scientific_settings["cavity"],
                "systems": cavity_manifests,
                "implementation_source_sha256s": implementation_sources,
            }
        ),
        "prepared_input_manifest": _canonical_hash(prepared_manifest),
        "force_panel_contract": force_panel_contract_sha,
        "benzene_pes_configuration": benzene_h1_pes.configuration_sha256(),
        "water_pes_configuration": water_h1_pes.configuration_sha256(),
    }


def _tracked_python_ledger(
    root: Path, head: str
) -> tuple[dict[str, str], tuple[tuple[str, _FileCapture], ...]]:
    names = _git(root, "ls-tree", "-r", "--name-only", head).decode().splitlines()
    python_names = sorted(name for name in names if name.endswith(".py"))
    if not python_names:
        raise SealInputError("clean HEAD contains no tracked Python sources")
    ledger: dict[str, str] = {}
    captures: list[tuple[str, _FileCapture]] = []
    for relative in python_names:
        committed = _git(root, "show", f"{head}:{relative}")
        capture = _capture(root / relative, role=f"tracked source {relative}")
        if capture.data != committed:
            raise SealInputError(f"tracked Python source differs from HEAD: {relative}")
        ledger[relative] = capture.sha256
        captures.append((relative, capture))
    return ledger, tuple(captures)


def _same_inode(paths: list[_FileCapture]) -> None:
    identities = {(item.identity[0], item.identity[1]) for item in paths}
    if len(identities) != len(paths):
        raise SealInputError("all seal inputs must have distinct file identities")


def _prepare_computation_seal(
    repo_root: str | Path,
    preregistration_path: str | Path,
    runner_path: str | Path,
    aggregator_path: str | Path,
    mace_mdp_checkpoint_path: str | Path,
    mace_polar_checkpoint_path: str | Path,
    parent_panel_path: str | Path,
    asset_root_path: str | Path,
) -> tuple[dict[str, object], _SealStability]:
    _require_fresh_execution_process()
    root, head, tree, _ = _clean_repository(repo_root)
    _require_running_root(root)
    _, prereg = _committed_capture(
        root, head, preregistration_path, role="preregistration"
    )
    runner_relative, runner = _committed_capture(
        root, head, runner_path, role="runner", expected_relative=RUNNER_ID
    )
    aggregator_relative, aggregator = _committed_capture(
        root, head, aggregator_path, role="aggregator", expected_relative=AGGREGATOR_ID
    )
    prereg_payload = _read_json(prereg, role="preregistration")
    science, parent_panel_sha, v1_prereg_sha = _validate_preregistration(prereg_payload)
    pro_assets = _validate_verified_pro_amendment_artifacts(
        root,
        head,
        prereg_payload["verified_pro_schema_amendment"],
    )

    v1_binding = prereg_payload["parent_bindings"]["v1_preregistration"]
    v1_relative = v1_binding.get("relative_path")
    if not isinstance(v1_relative, str):
        raise SealInputError("v1 preregistration relative_path is invalid")
    _, v1_prereg = _committed_capture(
        root,
        head,
        root / v1_relative,
        role="v1 preregistration",
        expected_relative=v1_relative,
    )
    if v1_prereg.sha256 != v1_prereg_sha:
        raise SealInputError("v1 preregistration bytes differ from prospective binding")

    mdp = _capture(mace_mdp_checkpoint_path, role="MACE-MDP checkpoint")
    polar = _capture(mace_polar_checkpoint_path, role="MACE-POLAR checkpoint")
    panel_relative, panel = _committed_capture(
        root, head, parent_panel_path, role="parent panel"
    )
    if panel.sha256 != parent_panel_sha:
        raise SealInputError("parent panel bytes differ from the preregistered SHA256")
    panel_binding = prereg_payload["parent_bindings"]["v1_parent_panel"]
    if panel_binding.get("relative_path") != panel_relative:
        raise SealInputError("parent panel path differs from the preregistered path")
    panel_payload = _read_json(panel, role="parent panel")
    records = panel_payload.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not isinstance(records[0], Mapping)
    ):
        raise SealInputError("parent panel must contain an ordered first record")
    first_record = records[0]
    expected_order = panel_binding.get("record_order")
    expected_compound = prereg_payload["inherited_v1_contract"]["force_panel"].get(
        "gepol_regression_compound_id"
    )
    if (
        not isinstance(expected_order, list)
        or not expected_order
        or first_record.get("compound_id") != expected_order[0]
        or first_record.get("compound_id") != expected_compound
    ):
        raise SealInputError("parent panel first record is not the frozen benzene case")
    mol2_relative = first_record.get("mol2_path")
    if not isinstance(mol2_relative, str):
        raise SealInputError("parent panel benzene mol2_path is invalid")
    mol2_posix = PurePosixPath(mol2_relative)
    if mol2_posix.is_absolute() or any(
        part in {"", ".", ".."} for part in mol2_posix.parts
    ):
        raise SealInputError("parent panel benzene mol2_path is non-canonical")
    asset_root = Path(os.path.abspath(os.path.expanduser(asset_root_path)))
    _reject_symlinks(asset_root, role="asset root")
    asset_root = asset_root.resolve(strict=True)
    if not asset_root.is_dir():
        raise SealInputError("asset root must be a directory")
    benzene_mol2 = _capture(asset_root / mol2_relative, role="benzene MOL2")
    try:
        benzene_mol2.path.relative_to(asset_root)
    except ValueError as exc:
        raise SealInputError("benzene MOL2 escapes asset root") from exc
    expected_mol2_sha = _digest(
        first_record.get("mol2_sha256"), role="benzene MOL2 binding"
    )
    if benzene_mol2.sha256 != expected_mol2_sha:
        raise SealInputError("benzene MOL2 bytes differ from parent panel binding")
    projection_path = (
        asset_root
        / PANEL_RELATIVE_ROOT
        / str(first_record["compound_id"])
        / CUTOFF_DIRECTORY
        / "result.json"
    )
    benzene_projection = _capture(
        projection_path, role="benzene frozen projection result"
    )
    try:
        benzene_projection.path.relative_to(asset_root)
    except ValueError as exc:
        raise SealInputError("benzene projection result escapes asset root") from exc
    _same_inode(
        [
            prereg,
            v1_prereg,
            runner,
            aggregator,
            mdp,
            polar,
            panel,
            benzene_mol2,
            benzene_projection,
        ]
    )

    frozen_runtime = prereg_payload["inherited_v1_contract"]["frozen_runtime_contract"]
    expected_mdp = _digest(
        frozen_runtime.get("mace_mdp_checkpoint_sha256"),
        role="MACE-MDP preregistration binding",
    )
    expected_polar = _digest(
        frozen_runtime.get("mace_polar_checkpoint_sha256"),
        role="MACE-POLAR preregistration binding",
    )
    if mdp.sha256 != expected_mdp or polar.sha256 != expected_polar:
        raise SealInputError("checkpoint bytes differ from preregistered model assets")

    runtime = _configure_and_capture_runtime()
    source_ledger, source_captures = _tracked_python_ledger(root, head)
    for relative, capture in (
        (runner_relative, runner),
        (aggregator_relative, aggregator),
    ):
        if source_ledger.get(relative) != capture.sha256:
            raise SealInputError(f"execution source is absent from ledger: {relative}")
    if (
        source_ledger.get(SEAL_GENERATOR_ID)
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise SealInputError("running seal generator is not committed generator bytes")
    required_modules = {
        "maple/solvation/release/admission.py",
        "maple/solvation/release/evidence.py",
        "maple/solvation/api/capabilities.py",
    }
    if not required_modules <= set(source_ledger):
        raise SealInputError(
            "source ledger omits admission/evidence/capability modules"
        )

    provider_hashes = _provider_configuration_factory(
        mdp.data,
        polar.data,
        mdp.sha256,
        polar.sha256,
        benzene_mol2,
        benzene_projection,
        prereg_payload["inherited_v1_contract"]["force_panel"],
        v1_prereg.sha256,
        panel.sha256,
        science,
        source_ledger,
    )

    from maple.solvation.release.admission import (
        CLAIM_BOUNDARY_ID,
        COMPUTATION_SEAL_V2_SCHEMA,
        EXPOSURE_AWARE_PROTOCOL_LABEL,
        H1_V2_PROFILE_ID,
        H1_V2_SCALAR_ID,
        H1_V2_STATE_ID,
        ComputationSealV2,
    )
    from maple.solvation.release.evidence import canonical_json_sha256

    identity = prereg_payload["prospective_h1_identity_contract"]
    if {
        "profile_id": identity.get("profile_id"),
        "scalar_id": identity.get("scalar_id"),
        "state_id": identity.get("state_id"),
        "claim_boundary_id": identity.get("claim_boundary_id"),
    } != {
        "profile_id": H1_V2_PROFILE_ID,
        "scalar_id": H1_V2_SCALAR_ID,
        "state_id": H1_V2_STATE_ID,
        "claim_boundary_id": CLAIM_BOUNDARY_ID,
    }:
        raise SealInputError("preregistration prospective H1 identity is invalid")
    claim = prereg_payload["claim_boundary"]
    payload: dict[str, object] = {
        "schema_id": COMPUTATION_SEAL_V2_SCHEMA,
        "artifact_id": SEAL_ARTIFACT_ID,
        "profile_id": H1_V2_PROFILE_ID,
        "scalar_id": H1_V2_SCALAR_ID,
        "state_id": H1_V2_STATE_ID,
        "git_head": head,
        "git_tree": tree,
        "git_clean": True,
        "preregistration_id": PREREGISTRATION_ARTIFACT_ID,
        "preregistration_sha256": prereg.sha256,
        "runner_id": RUNNER_ID,
        "runner_sha256": runner.sha256,
        "aggregator_id": AGGREGATOR_ID,
        "aggregator_sha256": aggregator.sha256,
        "computation_source_sha256s": source_ledger,
        "asset_sha256s": {
            "benzene_mol2": benzene_mol2.sha256,
            "benzene_projection_result": benzene_projection.sha256,
            "parent_panel": panel.sha256,
            "v1_preregistration": v1_prereg.sha256,
            **pro_assets,
        },
        "checkpoint_sha256s": {
            "mace_mdp_checkpoint": mdp.sha256,
            "mace_polar_checkpoint": polar.sha256,
        },
        "provider_configuration_sha256s": provider_hashes,
        "scientific_settings": science,
        "runtime_fingerprint": runtime,
        "runtime_guards": prereg_payload["runtime_guards"],
        "domain_guards": prereg_payload["domain_guards"],
        "exposure_aware_protocol_label": EXPOSURE_AWARE_PROTOCOL_LABEL,
        "claim_boundary_id": CLAIM_BOUNDARY_ID,
        "non_admissions": claim["non_admissions"],
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    seal = ComputationSealV2.from_mapping(payload)
    _assert_loaded_repo_modules_bound(root, head, source_ledger)
    captures = (
        ("preregistration", prereg),
        ("v1 preregistration", v1_prereg),
        ("runner", runner),
        ("aggregator", aggregator),
        ("MACE-MDP checkpoint", mdp),
        ("MACE-POLAR checkpoint", polar),
        ("parent panel", panel),
        ("benzene MOL2", benzene_mol2),
        ("benzene frozen projection result", benzene_projection),
        *(
            (f"tracked source {relative}", capture)
            for relative, capture in source_captures
        ),
    )
    stability = _SealStability(root, head, tree, captures)
    stability.assert_stable()
    return seal.as_dict(), stability


def build_computation_seal(
    repo_root: str | Path,
    preregistration_path: str | Path,
    runner_path: str | Path,
    aggregator_path: str | Path,
    mace_mdp_checkpoint_path: str | Path,
    mace_polar_checkpoint_path: str | Path,
    parent_panel_path: str | Path,
    asset_root_path: str | Path,
) -> dict[str, object]:
    payload, stability = _prepare_computation_seal(
        repo_root,
        preregistration_path,
        runner_path,
        aggregator_path,
        mace_mdp_checkpoint_path,
        mace_polar_checkpoint_path,
        parent_panel_path,
        asset_root_path,
    )
    stability.assert_stable()
    return payload


def _external_output(
    output: str | Path, root: Path
) -> tuple[Path, int, tuple[int, int, int]]:
    lexical = Path(os.path.abspath(os.path.expanduser(output)))
    _reject_symlinks(lexical, role="output")
    try:
        lexical.relative_to(root)
    except ValueError:
        pass
    else:
        raise SealInputError("output must be external to the repository")
    parent = lexical.parent.resolve(strict=True)
    if not parent.is_dir() or lexical.name in {"", ".", ".."}:
        raise SealInputError("output parent/name is invalid")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        value = os.fstat(descriptor)
        identity = (value.st_dev, value.st_ino, value.st_mode)
        lexical_parent = os.stat(lexical.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(lexical_parent.st_mode)
            or (lexical_parent.st_dev, lexical_parent.st_ino, lexical_parent.st_mode)
            != identity
        ):
            raise SealInputError("output parent changed while being opened")
        try:
            os.stat(lexical.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SealInputError(
                "output already exists; computation seals are never overwritten"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return lexical, descriptor, identity


def _assert_output_parent_stable(
    output: Path,
    directory_fd: int,
    directory_identity: tuple[int, int, int],
) -> None:
    """Bind the original lexical parent path to the retained directory FD."""

    _reject_symlinks(output.parent, role="output parent")
    try:
        descriptor_value = os.fstat(directory_fd)
        path_value = os.stat(output.parent, follow_symlinks=False)
    except OSError as exc:
        raise SealInputError(
            f"output parent changed during publication: {exc}"
        ) from exc
    descriptor_identity = (
        descriptor_value.st_dev,
        descriptor_value.st_ino,
        descriptor_value.st_mode,
    )
    path_identity = (path_value.st_dev, path_value.st_ino, path_value.st_mode)
    if (
        not stat.S_ISDIR(path_value.st_mode)
        or descriptor_identity != directory_identity
        or path_identity != directory_identity
    ):
        raise SealInputError("output parent directory changed during publication")


def _assert_output_absent(output_name: str, directory_fd: int) -> None:
    try:
        os.stat(output_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SealInputError(f"cannot inspect output entry: {exc}") from exc
    raise SealInputError("output appeared during publication")


def _call_linkat(
    oldfd: int, old: bytes, newfd: int, new: bytes, flags: int
) -> int | None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise SealInputError("secure publication requires linkat")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    return None if linkat(oldfd, old, newfd, new, flags) == 0 else ctypes.get_errno()


def _publish_external_json(
    output: Path,
    directory_fd: int,
    directory_identity: tuple[int, int, int],
    payload: Mapping[str, object],
    stability: _SealStability,
) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flag = getattr(os, "O_TMPFILE", None)
    if not isinstance(flag, int) or not flag:
        raise SealInputError("Linux O_TMPFILE is unavailable")
    temporary = os.open(
        ".", flag | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=directory_fd
    )
    try:
        anonymous_before = os.fstat(temporary)
        if not stat.S_ISREG(anonymous_before.st_mode) or anonymous_before.st_nlink != 0:
            raise SealInputError("O_TMPFILE did not create an anonymous regular inode")
        view = memoryview(data)
        while view:
            written = os.write(temporary, view)
            if written <= 0:
                raise SealInputError("anonymous output write made no progress")
            view = view[written:]
        os.fsync(temporary)
        os.lseek(temporary, 0, os.SEEK_SET)
        if os.read(temporary, len(data) + 1) != data:
            raise SealInputError("anonymous output verification failed")
        anonymous_verified = os.fstat(temporary)
        if (
            anonymous_verified.st_dev != anonymous_before.st_dev
            or anonymous_verified.st_ino != anonymous_before.st_ino
            or anonymous_verified.st_mode != anonymous_before.st_mode
            or anonymous_verified.st_nlink != 0
            or anonymous_verified.st_size != len(data)
        ):
            raise SealInputError(
                "anonymous output identity changed during verification"
            )
        _before_publish()
        stability.assert_stable()
        _assert_output_parent_stable(output, directory_fd, directory_identity)
        _assert_output_absent(output.name, directory_fd)
        error = _call_linkat(
            temporary, b"", directory_fd, os.fsencode(output.name), 0x1000
        )
        if error in {errno.ENOENT, errno.EPERM, errno.EOPNOTSUPP}:
            proc_path = f"/proc/self/fd/{temporary}"
            proc_value = os.stat(proc_path)
            descriptor_value = os.fstat(temporary)
            if (proc_value.st_dev, proc_value.st_ino) != (
                descriptor_value.st_dev,
                descriptor_value.st_ino,
            ):
                raise SealInputError("/proc/self/fd does not identify anonymous output")
            error = _call_linkat(
                -100,
                os.fsencode(proc_path),
                directory_fd,
                os.fsencode(output.name),
                0x400,
            )
        if error is not None:
            if error == errno.EEXIST:
                raise SealInputError("output appeared during publication")
            raise SealInputError(
                f"anonymous output publication failed: {os.strerror(error)}"
            )
        linked = os.stat(output.name, dir_fd=directory_fd, follow_symlinks=False)
        current = os.fstat(temporary)
        if (linked.st_dev, linked.st_ino) != (current.st_dev, current.st_ino):
            raise SealInputError("published output is not the verified anonymous inode")
        os.fsync(directory_fd)
    finally:
        os.close(temporary)


def _before_publish() -> None:
    """Test hook immediately before the no-replace publication commit point."""


def seal_revalidation(
    repo_root: str | Path,
    preregistration_path: str | Path,
    runner_path: str | Path,
    aggregator_path: str | Path,
    mace_mdp_checkpoint_path: str | Path,
    mace_polar_checkpoint_path: str | Path,
    parent_panel_path: str | Path,
    asset_root_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    root = Path(os.path.abspath(os.path.expanduser(repo_root))).resolve(strict=True)
    output, directory_fd, directory_identity = _external_output(output_path, root)
    try:
        payload, stability = _prepare_computation_seal(
            root,
            preregistration_path,
            runner_path,
            aggregator_path,
            mace_mdp_checkpoint_path,
            mace_polar_checkpoint_path,
            parent_panel_path,
            asset_root_path,
        )
        _publish_external_json(
            output, directory_fd, directory_identity, payload, stability
        )
    finally:
        os.close(directory_fd)
    return payload


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--aggregator", type=Path, required=True)
    parser.add_argument("--mace-mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--mace-polar-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-panel", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _arguments(argv)
        payload = seal_revalidation(
            args.repo_root,
            args.preregistration,
            args.runner,
            args.aggregator,
            args.mace_mdp_checkpoint,
            args.mace_polar_checkpoint,
            args.parent_panel,
            args.asset_root,
            args.output,
        )
    except (OSError, SealInputError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(payload["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
