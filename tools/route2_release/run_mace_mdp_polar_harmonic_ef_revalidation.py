#!/usr/bin/env python3
"""Secure process/capture/publication boundary for H1 harmonic E/F replay.

The scientific candidate and failure schemas are injected deliberately.  This
module does not invent admission fields while the shared typed APIs are being
landed.  Identity-forming publication reuses the computation-sealer's strict
single-read capture and anonymous-inode no-replace commit implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping
from uuid import uuid4
import argparse
import hashlib
import stat

ScienceFactory = Callable[
    ..., tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]
]

_SECURITY_MODULE = None


def _security_module():
    """Load only the verified sibling sealer, never a PYTHONPATH package."""

    global _SECURITY_MODULE
    if _SECURITY_MODULE is not None:
        return _SECURITY_MODULE
    path = (
        Path(__file__).resolve(strict=True).parent
        / "seal_mace_mdp_polar_harmonic_ef_revalidation.py"
    )
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RunnerInputError("sibling sealer must be a non-hard-linked source")
        blocks = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise RunnerInputError(
            f"cannot capture sibling sealer source: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RunnerInputError("sibling sealer changed while source was captured")
    source = b"".join(blocks)
    if len(source) != before.st_size:
        raise RunnerInputError("sibling sealer source capture is incomplete")
    name = "_maple_route2_h1_sealer_security"
    from types import ModuleType

    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__cached__ = None
    module._captured_source_sha256 = hashlib.sha256(source).hexdigest()
    sys.modules[name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    if Path(module.__file__).resolve(strict=True) != path:
        raise RunnerInputError("loaded sealer security module path changed")
    _SECURITY_MODULE = module
    return module


def _admission_module():
    return importlib.import_module("maple.solvation.release.admission")


class RunnerInputError(ValueError):
    """Raised when replay identity or terminal publication is unsafe."""


class ScienceExecutionError(RuntimeError):
    failure_stage = "science-execution"


_CLAIMED_PROCESS_IDENTITIES: set[tuple[str, str, str]] = set()


def claim_fresh_process_identity(
    *, label: str, process_uuid: str, process_started_at_utc: str
) -> None:
    """Prevent A/B or retry reuse inside one interpreter process."""

    if label not in {"a", "b"}:
        raise RunnerInputError("replicate label must be a or b")
    if not process_uuid or not process_started_at_utc:
        raise RunnerInputError("process UUID/start time must be nonempty")
    if _CLAIMED_PROCESS_IDENTITIES:
        raise RunnerInputError("one fresh process may execute only one replicate")
    _CLAIMED_PROCESS_IDENTITIES.add((label, process_uuid, process_started_at_utc))


def require_fresh_execution_process() -> None:
    """Run before importing NumPy, Torch, MACE, or MAPLE scientific modules."""

    if os.environ.get("PYTHONPYCACHEPREFIX") is not None or sys.pycache_prefix:
        raise RunnerInputError("pycache prefixing is forbidden for H1 replay")
    forbidden = (
        "numpy",
        "torch",
        "mace",
        "graph_longrange",
        "maple.function.calculator.mace",
        "maple.solvation.models",
        "maple.solvation.continuum",
        "maple.solvation.experimental",
        "maple.solvation.release.admission",
        "maple.solvation.release.evidence",
        "maple.solvation.api",
    )
    preloaded = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    if preloaded:
        raise RunnerInputError(
            "scientific modules were preloaded before deterministic setup: "
            + ", ".join(preloaded[:12])
        )


def clean_repository_identity(repo_root: str | Path) -> tuple[Path, str, str]:
    """Return the exact clean worktree root, HEAD and tree."""

    security = _security_module()
    try:
        root, head, tree, _ = security._clean_repository(repo_root)
    except security.SealInputError as error:
        raise RunnerInputError(str(error)) from error
    return root, head, tree


def anchor_running_repository(root: Path) -> None:
    """Require this runner and import resolution to originate from repo_root."""

    running = Path(__file__).resolve(strict=True).parents[2]
    if running != root:
        raise RunnerInputError("executing runner root differs from --repo-root")
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "maple"
            or name.startswith("maple.")
            or name == "tools"
            or name.startswith("tools.")
        ):
            continue
        origins = []
        raw_origin = getattr(module, "__file__", None)
        if raw_origin is not None:
            origins.append(Path(raw_origin).resolve())
        raw_paths = getattr(module, "__path__", ())
        origins.extend(Path(item).resolve() for item in raw_paths)
        if not origins:
            raise RunnerInputError(f"preloaded {name} has no verifiable repo origin")
        for origin in origins:
            try:
                origin.relative_to(root)
            except ValueError as error:
                raise RunnerInputError(
                    f"preloaded {name} resolves outside verified repo_root"
                ) from error
    filtered = []
    for entry in sys.path:
        try:
            if Path(entry or os.getcwd()).resolve() == root:
                continue
        except OSError:
            pass
        filtered.append(entry)
    sys.path[:] = [str(root), *filtered]
    spec = importlib.util.find_spec("maple")
    if spec is None or spec.origin is None:
        raise RunnerInputError("MAPLE import cannot be resolved from repo_root")
    try:
        Path(spec.origin).resolve().relative_to(root)
    except ValueError as error:
        raise RunnerInputError("MAPLE import resolves outside repo_root") from error


def configure_and_capture_runtime() -> Mapping[str, object]:
    """Apply and capture the sealer's exact deterministic runtime contract."""

    security = _security_module()
    try:
        return security._configure_and_capture_runtime()
    except (security.SealInputError, OSError) as error:
        raise RunnerInputError(str(error)) from error


def capture_single_read_inputs(
    paths: Mapping[str, str | Path],
) -> dict[str, object]:
    """Capture every input once and reject symlink/hardlink/inode aliases."""

    if not isinstance(paths, Mapping) or not paths:
        raise RunnerInputError("capture paths must be a nonempty role mapping")
    security = _security_module()
    captures = {}
    try:
        for role, path in paths.items():
            if not isinstance(role, str) or not role:
                raise RunnerInputError("capture roles must be nonempty strings")
            captures[role] = security._capture(path, role=role)
        security._same_inode(list(captures.values()))
    except security.SealInputError as error:
        raise RunnerInputError(str(error)) from error
    return captures


def read_captured_json(capture: object, *, role: str) -> Mapping[str, object]:
    security = _security_module()
    try:
        return security._read_json(capture, role=role)
    except security.SealInputError as error:
        raise RunnerInputError(str(error)) from error


@dataclass(frozen=True, slots=True)
class RunnerStability:
    """Captured input stability plus optional clean Git identity."""

    captures: tuple[tuple[str, object], ...]
    repo_root: Path | None = None
    git_head: str | None = None
    git_tree: str | None = None
    source_ledger: tuple[tuple[str, str], ...] = ()

    def assert_modules_bound(self) -> None:
        if self.repo_root is None or self.git_head is None or not self.source_ledger:
            return
        security = _security_module()
        try:
            security._assert_loaded_repo_modules_bound(
                self.repo_root, self.git_head, dict(self.source_ledger)
            )
        except security.SealInputError as error:
            raise RunnerInputError(str(error)) from error

    def assert_stable(self) -> None:
        security = _security_module()
        try:
            if self.repo_root is not None:
                root, head, tree, _ = security._clean_repository(self.repo_root)
                if (
                    root != self.repo_root
                    or head != self.git_head
                    or tree != self.git_tree
                ):
                    raise RunnerInputError("repository identity changed during replay")
            for role, capture in self.captures:
                capture.assert_stable(role=role)
            self.assert_modules_bound()
        except security.SealInputError as error:
            raise RunnerInputError(str(error)) from error


def select_terminal_artifact(
    *,
    terminal_active: bool,
    candidate_factory: Callable[[], Mapping[str, object]],
    failure_factory: Callable[[str, Exception | None], Mapping[str, object]],
) -> tuple[str, Mapping[str, object]]:
    """Choose success/failure without knowing either shared artifact schema."""

    if type(terminal_active) is not bool:
        raise TypeError("terminal_active must be exactly bool")
    try:
        candidate = candidate_factory()
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate_factory must return a mapping")
    except Exception as error:
        stage = getattr(error, "failure_stage", "candidate-validation")
        failure = failure_factory(stage, error)
        if not isinstance(failure, Mapping):
            raise TypeError("failure_factory must return a mapping")
        return "failure", failure
    if terminal_active:
        failure = failure_factory("rich-v2-terminal", None)
        if not isinstance(failure, Mapping):
            raise TypeError("failure_factory must return a mapping")
        return "failure", failure
    return "success", candidate


def build_typed_replicate_candidate(
    *,
    seal: object,
    label: str,
    process_uuid: str,
    process_started_at_utc: str,
    prepared_inputs: Mapping[str, object],
    prepared_providers: Mapping[str, object],
    recording: Mapping[str, object],
    candidate_observer: Callable[[str, Mapping[str, bool]], None] | None = None,
) -> Mapping[str, object]:
    """Build the exact shared candidate and typed replicate record."""

    admission = _admission_module()
    candidate, gates = admission.build_rich_harmonic_ef_measurement_candidate_v2(
        seal=seal,
        prepared_inputs=prepared_inputs,
        prepared_providers=prepared_providers,
        recording=recording,
    )
    measurement_sha256 = admission.canonical_json_sha256(candidate)
    if candidate_observer is not None:
        candidate_observer(measurement_sha256, gates)
    content = {
        "schema_id": admission.REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
        "artifact_id": f"route2-h1-harmonic-ef-replicate-{label}-v2",
        "label": label,
        "process_uuid": process_uuid,
        "process_started_at_utc": process_started_at_utc,
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "git_head": seal.git_head,
        "git_tree": seal.git_tree,
        "source_ledger_sha256": seal.source_ledger_sha256,
        "asset_ledger_sha256": seal.asset_ledger_sha256,
        "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
        "measurement_schema_id": admission.RICH_MEASUREMENT_SCHEMA_ID,
        "measurement_sha256": measurement_sha256,
        "measurement": candidate,
        "gate_results": gates,
    }
    payload = {
        **content,
        "artifact_sha256": admission.canonical_json_sha256(content),
    }
    return admission.ReplicateAdmissionRecordV2.from_mapping(
        payload, seal=seal
    ).as_dict()


def build_typed_execution_failure(
    *,
    seal: object,
    label: str,
    process_uuid: str,
    process_started_at_utc: str,
    stage: str,
    error: Exception | None,
    available_partial_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    """Build the exact shared fail-closed execution-failure artifact."""

    admission = _admission_module()
    actual_error = error or RuntimeError("explicit fail-closed terminal selected")
    full_message = str(actual_error) or type(actual_error).__name__
    displayed_message = full_message[:4096]
    content = {
        "schema_id": admission.EXECUTION_FAILURE_V2_SCHEMA,
        "artifact_id": f"route2-h1-harmonic-ef-execution-failure-{label}-v2",
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "label": label,
        "process_uuid": process_uuid,
        "process_started_at_utc": process_started_at_utc,
        "stage": stage,
        "exception_type": type(actual_error).__name__,
        "exception_message": displayed_message,
        "exception_message_sha256": admission.canonical_json_sha256(full_message),
        "exception_message_truncated": len(full_message) > len(displayed_message),
        "available_partial_evidence": json.loads(
            json.dumps(available_partial_evidence, sort_keys=True, allow_nan=False)
        ),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "claim_boundary_id": seal.claim_boundary_id,
        "non_admissions": list(seal.non_admissions),
    }
    payload = {
        **content,
        "artifact_sha256": admission.canonical_json_sha256(content),
    }
    return admission.ExecutionFailureV2.from_mapping(payload, seal=seal).as_dict()


def publish_external_artifact(
    *,
    repo_root: str | Path,
    output_path: str | Path,
    payload: Mapping[str, object],
    stability: RunnerStability,
) -> Path:
    """Atomically no-replace publish one verified JSON artifact outside the repo."""

    if not isinstance(stability, RunnerStability):
        raise TypeError("stability must be RunnerStability")
    security = _security_module()
    root = Path(repo_root).resolve(strict=True)
    directory_fd = None
    try:
        output, directory_fd, directory_identity = security._external_output(
            output_path, root
        )
        security._publish_external_json(
            output, directory_fd, directory_identity, payload, stability
        )
        return output
    except (security.SealInputError, RunnerInputError, OSError) as error:
        raise RunnerInputError(str(error)) from error
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def orchestrate_terminal_publication(
    *,
    repo_root: str | Path,
    success_path: str | Path,
    failure_path: str | Path,
    terminal_active: bool,
    candidate_factory: Callable[[], Mapping[str, object]],
    failure_factory: Callable[[str, Exception | None], Mapping[str, object]],
    stability: RunnerStability,
) -> tuple[str, Path, Mapping[str, object]]:
    """Publish only the selected explicit terminal path."""

    if Path(os.path.abspath(success_path)) == Path(os.path.abspath(failure_path)):
        raise RunnerInputError("success and failure outputs must be distinct paths")
    kind, payload = select_terminal_artifact(
        terminal_active=terminal_active,
        candidate_factory=candidate_factory,
        failure_factory=failure_factory,
    )
    output = failure_path if kind == "failure" else success_path
    published = publish_external_artifact(
        repo_root=repo_root,
        output_path=output,
        payload=payload,
        stability=stability,
    )
    return kind, published, payload


def _execute_science(
    *, seal: object, captures: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    """Rebuild prepared providers/inputs and execute the exact 141-event tree."""

    security = _security_module()
    from maple.function.read.filereader.mol2_reader import mol2_atoms_from_bytes
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )

    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.derivatives import RichardsonScalarForce
    from maple.solvation.experimental.mace_mdp_polar_harmonic import (
        MACE_MDPPolarHybridSmoothHarmonicPES,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_anchored_mace_polar_hybrid,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.release.harmonic_ef_measurement import (
        H1PreparedHarmonicEFPESV2,
        PreparedHarmonicEFBenzeneInputV2,
        PreparedHarmonicEFInputsV2,
        PreparedHarmonicEFWaterInputV2,
        run_rich_harmonic_ef_raw_measurement_v2,
    )
    import numpy as np
    import torch

    seal_payload = seal.as_dict()
    settings = seal_payload["scientific_settings"]
    v1 = read_captured_json(captures["v1_preregistration"], role="v1 preregistration")
    force_panel = v1["force_panel"]
    benzene = mol2_atoms_from_bytes(
        captures["benzene_mol2"].data,
        source_id=f"captured-sha256:{captures['benzene_mol2'].sha256}",
        charge=0,
        mult=1,
    )
    projection = read_captured_json(
        captures["benzene_projection"], role="benzene projection"
    )
    parsed = projection["inputs"]["parsed_pcm_input"]
    projection_radii = np.asarray(parsed["cavity_radii_angstrom"], dtype=float)
    if projection_radii.shape != (len(benzene),) or np.any(projection_radii <= 0.0):
        raise RunnerInputError("captured benzene projection radii are invalid")
    benzene_radii = tuple(
        float(value)
        for value in smd_water_coulomb_radii(benzene.get_chemical_symbols())
    )
    water_radii = tuple(
        float(value) for value in smd_water_coulomb_radii(("O", "H", "H"))
    )
    prepared_benzene = PreparedHarmonicEFBenzeneInputV2(
        compound_id=force_panel["gepol_regression_compound_id"],
        name="benzene",
        atomic_numbers=tuple(int(value) for value in benzene.numbers),
        positions_angstrom=tuple(
            tuple(float(value) for value in row) for row in benzene.positions
        ),
        charge=0,
        multiplicity=1,
        cavity_radii_angstrom=benzene_radii,
        mol2_sha256=captures["benzene_mol2"].sha256,
        projection_result_sha256=captures["benzene_projection"].sha256,
        predecessor_dof=tuple(force_panel["gepol_regression_cartesian_dof"]),
    )
    prepared_water = PreparedHarmonicEFWaterInputV2(
        atomic_numbers=(8, 1, 1),
        positions_angstrom=tuple(
            tuple(float(value) for value in row)
            for row in force_panel["water_geometry_angstrom"]
        ),
        charge=0,
        multiplicity=1,
        cavity_radii_angstrom=water_radii,
    )
    prepared_inputs = PreparedHarmonicEFInputsV2(
        benzene=prepared_benzene,
        water=prepared_water,
        v1_preregistration_sha256=captures["v1_preregistration"].sha256,
        parent_panel_sha256=captures["parent_panel"].sha256,
        force_panel_contract=force_panel,
        force_panel_contract_sha256=security._canonical_hash(force_panel),
    )
    captures["mace_mdp_checkpoint"].assert_stable(role="MACE-MDP checkpoint")
    captures["mace_polar_checkpoint"].assert_stable(role="MACE-POLAR checkpoint")
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_bytes=captures["mace_mdp_checkpoint"].data, device="cpu"
    )
    polar_kwargs = {
        "device": "cuda",
        "long_range_evaluator_profile": (
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    }
    if (
        "checkpoint_bytes"
        in inspect.signature(
            build_official_mace_polar_1_m_radial_gto_adapter
        ).parameters
    ):
        polar_kwargs["checkpoint_bytes"] = captures["mace_polar_checkpoint"].data
    else:
        raise RunnerInputError(
            "MACE-POLAR checkpoint_bytes API is unavailable; path reopen forbidden"
        )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(**polar_kwargs)
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    continuum = settings["continuum"]

    def pes(prepared):
        return MACE_MDPPolarHybridSmoothHarmonicPES(
            hybrid=hybrid,
            atomic_numbers=prepared.atomic_numbers,
            cavity_radii_angstrom=prepared.cavity_radii_angstrom,
            dtype=torch.float64,
            device="cuda",
            transition_width_angstrom2=continuum["transition_width_angstrom2"],
            surface_lmax=continuum["surface_lmax"],
            exposure_lmax=continuum["exposure_lmax"],
            exposure_radial_quadrature_order=continuum[
                "exposure_radial_quadrature_order"
            ],
            source_radial_quadrature_order=continuum["source_radial_quadrature_order"],
            green_radial_quadrature_order=continuum["green_radial_quadrature_order"],
            force_backend=RichardsonScalarForce(
                coarse_step_angstrom=settings["force_stencil"]["coarse_step_angstrom"],
                maximum_error_eV_per_A=settings["force_stencil"][
                    "maximum_local_error_ev_per_angstrom"
                ],
            ),
        )

    benzene_h1 = H1PreparedHarmonicEFPESV2(
        pes(prepared_benzene), prepared_benzene, system_role="benzene"
    )
    water_h1 = H1PreparedHarmonicEFPESV2(
        pes(prepared_water), prepared_water, system_role="water"
    )
    captures["mace_mdp_checkpoint"].assert_stable(role="MACE-MDP checkpoint")
    captures["mace_polar_checkpoint"].assert_stable(role="MACE-POLAR checkpoint")
    prepared_providers = {
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "checkpoint_sha256s": dict(seal.checkpoint_sha256s),
        "provider_configuration_sha256s": dict(seal.provider_configuration_sha256s),
    }
    expected = prepared_providers["provider_configuration_sha256s"]
    if expected["benzene_pes_configuration"] != benzene_h1.configuration_sha256():
        raise RunnerInputError("benzene H1 PES configuration differs from seal")
    if expected["water_pes_configuration"] != water_h1.configuration_sha256():
        raise RunnerInputError("water H1 PES configuration differs from seal")
    raw = run_rich_harmonic_ef_raw_measurement_v2(
        prepared_inputs,
        benzene_h1,
        water_h1,
        prepared_provider_contract=prepared_providers,
    )
    return (
        security._prepared_input_manifest(
            compound_id=prepared_benzene.compound_id,
            name=prepared_benzene.name,
            benzene_atomic_numbers=prepared_benzene.atomic_numbers,
            benzene_positions_angstrom=prepared_benzene.positions_angstrom,
            benzene_radii_angstrom=prepared_benzene.cavity_radii_angstrom,
            benzene_mol2_sha256=prepared_benzene.mol2_sha256,
            benzene_projection_sha256=prepared_benzene.projection_result_sha256,
            predecessor_dof=prepared_benzene.predecessor_dof,
            water_positions_angstrom=prepared_water.positions_angstrom,
            water_radii_angstrom=prepared_water.cavity_radii_angstrom,
            v1_preregistration_sha256=prepared_inputs.v1_preregistration_sha256,
            parent_panel_sha256=prepared_inputs.parent_panel_sha256,
            force_panel_contract=force_panel,
            force_panel_contract_sha256=prepared_inputs.force_panel_contract_sha256,
        ),
        prepared_providers,
        raw,
    )


def prepare_sealed_replay_inputs(
    *, repo_root: str | Path, paths: Mapping[str, str | Path]
) -> tuple[object, dict[str, object], RunnerStability, Mapping[str, object]]:
    """Capture and cross-bind every sealed replay input and deterministic runtime."""

    required = {
        "seal",
        "preregistration",
        "v1_preregistration",
        "parent_panel",
        "benzene_mol2",
        "benzene_projection",
        "mace_mdp_checkpoint",
        "mace_polar_checkpoint",
        "runner",
    }
    if set(paths) != required:
        raise RunnerInputError("sealed replay capture roles are missing or unknown")
    root, head, tree = clean_repository_identity(repo_root)
    anchor_running_repository(root)
    captures = capture_single_read_inputs(paths)
    raw_seal = read_captured_json(captures["seal"], role="computation seal")
    runtime = configure_and_capture_runtime()
    if runtime != raw_seal.get("runtime_fingerprint"):
        raise RunnerInputError("runtime fingerprint differs from raw computation seal")
    security = _security_module()
    source_ledger, source_captures = security._tracked_python_ledger(root, head)
    security_relative = Path(security.__file__).resolve().relative_to(root).as_posix()
    if source_ledger.get(security_relative) != security._captured_source_sha256:
        raise RunnerInputError("captured sealer source differs from committed ledger")
    admission = _admission_module()
    try:
        security._assert_loaded_repo_modules_bound(root, head, source_ledger)
    except security.SealInputError as error:
        raise RunnerInputError(str(error)) from error
    seal = admission.ComputationSealV2.from_mapping(raw_seal)
    if seal.git_head != head or seal.git_tree != tree:
        raise RunnerInputError("computation seal Git identity differs from worktree")
    runner_relative = captures["runner"].path.relative_to(root).as_posix()
    if (
        runner_relative != seal.runner_id
        or captures["runner"].sha256 != seal.runner_sha256
    ):
        raise RunnerInputError("runner path/hash differs from computation seal")
    if captures["preregistration"].sha256 != seal.preregistration_sha256:
        raise RunnerInputError("preregistration hash differs from computation seal")
    assets = dict(seal.asset_sha256s)
    checkpoints = dict(seal.checkpoint_sha256s)
    expected_captures = {
        "v1_preregistration": assets["v1_preregistration"],
        "parent_panel": assets["parent_panel"],
        "benzene_mol2": assets["benzene_mol2"],
        "benzene_projection": assets["benzene_projection_result"],
        "mace_mdp_checkpoint": checkpoints["mace_mdp_checkpoint"],
        "mace_polar_checkpoint": checkpoints["mace_polar_checkpoint"],
    }
    for role, expected in expected_captures.items():
        if captures[role].sha256 != expected:
            raise RunnerInputError(f"{role} hash differs from computation seal")
    if source_ledger != dict(seal.computation_source_sha256s):
        raise RunnerInputError("tracked Python source ledger differs from seal")
    stability = RunnerStability(
        tuple(captures.items())
        + tuple(
            (f"tracked source {name}", capture) for name, capture in source_captures
        ),
        root,
        head,
        tree,
        tuple(sorted(source_ledger.items())),
    )
    stability.assert_stable()
    return seal, captures, stability, runtime


def execute_revalidation(
    *,
    repo_root: str | Path,
    paths: Mapping[str, str | Path],
    label: str,
    process_uuid: str,
    process_started_at_utc: str,
    success_path: str | Path,
    failure_path: str | Path,
    science_factory: ScienceFactory = _execute_science,
) -> tuple[str, Path, Mapping[str, object]]:
    """Execute once; current shared terminal automatically selects typed failure."""

    if Path(os.path.abspath(success_path)) == Path(os.path.abspath(failure_path)):
        raise RunnerInputError("success and failure outputs must be distinct paths")
    claim_fresh_process_identity(
        label=label,
        process_uuid=process_uuid,
        process_started_at_utc=process_started_at_utc,
    )
    seal, captures, stability, runtime = prepare_sealed_replay_inputs(
        repo_root=repo_root, paths=paths
    )
    partial: dict[str, object] = {
        "captured_sha256s": {
            role: capture.sha256 for role, capture in captures.items()
        },
        "runtime_fingerprint_sha256": _admission_module().canonical_json_sha256(
            runtime
        ),
        "recording_sha256": None,
        "state_leaf_count": None,
        "solve_event_count": None,
        "system_keys": None,
        "system_event_ranges": None,
        "candidate_measurement_sha256": None,
        "derived_gate_results": None,
    }

    def candidate_factory():
        try:
            prepared_inputs, prepared_providers, recording = science_factory(
                seal=seal, captures=captures
            )
        except Exception as error:
            raise ScienceExecutionError(str(error)) from error
        stability.assert_modules_bound()
        systems = recording.get("systems")
        partial.update(
            {
                "recording_sha256": _admission_module().canonical_json_sha256(
                    recording
                ),
                "state_leaf_count": len(recording.get("state_leaves", {})),
                "solve_event_count": len(recording.get("solve_events", [])),
                "system_keys": (
                    sorted(systems) if isinstance(systems, Mapping) else None
                ),
                "system_event_ranges": (
                    {
                        key: systems[key].get("event_range")
                        for key in sorted(systems)
                        if isinstance(systems[key], Mapping)
                    }
                    if isinstance(systems, Mapping)
                    else None
                ),
            }
        )

        def observe_candidate(
            measurement_sha256: str, gates: Mapping[str, bool]
        ) -> None:
            partial["candidate_measurement_sha256"] = measurement_sha256
            partial["derived_gate_results"] = dict(gates)

        return build_typed_replicate_candidate(
            seal=seal,
            label=label,
            process_uuid=process_uuid,
            process_started_at_utc=process_started_at_utc,
            prepared_inputs=prepared_inputs,
            prepared_providers=prepared_providers,
            recording=recording,
            candidate_observer=observe_candidate,
        )

    def failure_factory(stage: str, error: Exception | None):
        return build_typed_execution_failure(
            seal=seal,
            label=label,
            process_uuid=process_uuid,
            process_started_at_utc=process_started_at_utc,
            stage=stage,
            error=error,
            available_partial_evidence=partial,
        )

    return orchestrate_terminal_publication(
        repo_root=repo_root,
        success_path=success_path,
        failure_path=failure_path,
        terminal_active=False,
        candidate_factory=candidate_factory,
        failure_factory=failure_factory,
        stability=stability,
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--v1-preregistration", type=Path, required=True)
    parser.add_argument("--parent-panel", type=Path, required=True)
    parser.add_argument("--benzene-mol2", type=Path, required=True)
    parser.add_argument("--benzene-projection", type=Path, required=True)
    parser.add_argument("--mace-mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--mace-polar-checkpoint", type=Path, required=True)
    parser.add_argument("--label", choices=("a", "b"), required=True)
    parser.add_argument("--success-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        require_fresh_execution_process()
        args = _arguments(argv)
        process_uuid = str(uuid4())
        process_started = datetime.now(timezone.utc).isoformat()
        kind, output, payload = execute_revalidation(
            repo_root=args.repo_root,
            paths={
                "seal": args.seal,
                "preregistration": args.preregistration,
                "v1_preregistration": args.v1_preregistration,
                "parent_panel": args.parent_panel,
                "benzene_mol2": args.benzene_mol2,
                "benzene_projection": args.benzene_projection,
                "mace_mdp_checkpoint": args.mace_mdp_checkpoint,
                "mace_polar_checkpoint": args.mace_polar_checkpoint,
                "runner": Path(__file__).resolve(strict=True),
            },
            label=args.label,
            process_uuid=process_uuid,
            process_started_at_utc=process_started,
            success_path=args.success_output,
            failure_path=args.failure_output,
        )
    except (RunnerInputError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        "ROUTE2_HARMONIC_EF_REVALIDATION="
        + json.dumps(
            {
                "kind": kind,
                "output": str(output),
                "artifact_id": payload.get("artifact_id"),
                "artifact_sha256": payload.get("artifact_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0 if kind == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
