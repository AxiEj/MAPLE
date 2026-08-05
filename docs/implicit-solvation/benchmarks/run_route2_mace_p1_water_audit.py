#!/usr/bin/env python3
"""Run the P-minus-1 water/MACE-POLAR/ddPCM P1 evidence audit.

The runner is deliberately energy-only and electrostatics-only.  It records a
strict outer fixed-point history and a complete scalar ledger, then classifies
the inner continuum solve from evidence the public provider actually exposes.
It does not run P2/P3 when P1 remains blocked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

_CPP_RUNTIME_REEXEC_MARKER = "MAPLE_ROUTE2_P1_CPP_RUNTIME_REEXEC"


def _ensure_process_cpp_runtime() -> None:
    """Preload the active environment C++ runtime before binary imports."""

    candidate = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not candidate.is_file():
        raise RuntimeError(
            "The MACE-POLAR P1 audit requires libstdc++.so.6 in the active "
            f"environment; missing: {candidate}"
        )
    resolved = str(candidate.resolve())
    current = os.environ.get("LD_PRELOAD", "")
    entries = {
        str(Path(entry).expanduser().resolve())
        for entry in current.replace(":", " ").split()
        if entry
    }
    if resolved in entries:
        return
    if os.environ.get(_CPP_RUNTIME_REEXEC_MARKER) == "1":
        raise RuntimeError("Unable to preload the active C++ runtime.")
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = resolved if not current else resolved + ":" + current
    environment[_CPP_RUNTIME_REEXEC_MARKER] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    _ensure_process_cpp_runtime()


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import ase
from ase.build import molecule
import numpy as np
import pyddx
import torch

from maple.function.calculator.calculator_base import (
    _IMPLICIT_SOLVENT_FACTORY_TOKEN,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2SCFConvergenceError,
)
from maple.function.calculator.extra_correction.implicit.route2_field_state import (
    LocalReactionField,
)
from maple.function.calculator.extra_correction.implicit.route2_mace_p0 import (
    load_mace_p0_certificate,
    sha256_file,
)
from maple.function.calculator.extra_correction.implicit.route2_p1_audit import (
    EnergySourceConjugacyEvidence,
    audit_plugin_energy_source_conjugacy,
    build_operational_energy_audit,
)
from maple.function.calculator.extra_correction.implicit.route2_plugin import (
    DEFAULT_ROUTE2_PLUGIN_REGISTRY,
    LegacyAtomicL1BridgeSpec,
    adapt_atomic_l1_plugin_to_legacy_engine,
    admit_plugin,
)
from maple.function.calculator.extra_correction.implicit.route2_pminus1 import (
    P_MINUS_1_WATER_CONTINUUM_SPEC,
    ElectrostaticsOnlyCDSResult,
)
from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator
from maple.function.route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1

ARTIFACT_ID = "route2-mace-p1-water-operational-audit-v1"
SCHEMA_VERSION = 1
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/run_route2_mace_p1_water_audit.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "maple/function/calculator/extra_correction/implicit/route2_p1_audit.py",
    "maple/function/calculator/extra_correction/implicit/route2_pminus1.py",
    "maple/function/calculator/extra_correction/implicit/route2_engine.py",
    "maple/function/calculator/extra_correction/implicit/route2_plugin_contracts.py",
    "maple/function/calculator/extra_correction/implicit/route2_plugin_legacy.py",
    "maple/function/calculator/extra_correction/implicit/route2_plugin_spaces.py",
    "maple/function/calculator/extra_correction/implicit/pyddx_pcm_response.py",
)
CONJUGACY_STEPS = (2.0e-3, 1.0e-3, 5.0e-4)
CONJUGACY_ABSOLUTE_TOLERANCE_EV = 1.0e-7
CONJUGACY_RELATIVE_TOLERANCE = 1.0e-4
CONJUGACY_DIRECTION = "scale-converged-local-reaction-field-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _geometry_sha256(atoms: ase.Atoms) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _conjugacy_probes(plugin, atoms, coupled) -> list[dict[str, object]]:
    base = LocalReactionField.from_nodewise_jet(coupled.reaction_field_values_ev)
    direction = np.array(
        coupled.reaction_field_values_ev,
        dtype=float,
        copy=True,
    )
    if not np.all(np.isfinite(direction)) or not np.any(direction):
        raise RuntimeError("P1 conjugacy direction is zero or non-finite.")
    return [
        audit_plugin_energy_source_conjugacy(
            plugin,
            atoms,
            base,
            direction,
            finite_difference_step=step,
            absolute_tolerance_ev=CONJUGACY_ABSOLUTE_TOLERANCE_EV,
            relative_tolerance=CONJUGACY_RELATIVE_TOLERANCE,
        ).as_dict()
        for step in CONJUGACY_STEPS
    ]


def _failed_scf_payload(
    *,
    args: argparse.Namespace,
    atoms: ase.Atoms,
    calculator: MACEPolCalculator,
    error: Route2SCFConvergenceError,
    started: float,
) -> dict[str, object]:
    return {
        "artifact": ARTIFACT_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "claim_boundary": (
            "This failed P1 execution is numerical diagnostics only. It is not "
            "P1 admission, P2/P3 evidence, a force/PES result, or an accuracy result."
        ),
        "execution": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git(
                "status", "--porcelain", "--untracked-files=all"
            ).splitlines(),
            "source_binding": "content-hash-bound-working-tree",
        },
        "repository_source_files_sha256": _source_hashes(),
        "checkpoint": dict(calculator.mace_polar_checkpoint_provenance or {}),
        "protocol": {
            "geometry_sha256": _geometry_sha256(atoms),
            "continuum": P_MINUS_1_WATER_CONTINUUM_SPEC.as_provenance(),
            "device": args.device,
        },
        "outer_scf_failure": {
            "message": str(error),
            "history": list(error.history),
        },
        "decision": {
            "p1_complete": False,
            "blockers": ["outer-scf-did-not-converge"],
            "p2_p3_status": "blocked-by-p1",
        },
        "runtime": {
            "python": platform.python_version(),
            "pyddx": pyddx.__version__,
            "pyddx_module": pyddx.__file__,
            "torch": torch.__version__,
            "device": args.device,
            "timing_seconds": {"total": time.perf_counter() - started},
        },
    }


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    started = time.perf_counter()
    atoms = molecule("H2O")
    atoms.info.update(charge=0, mult=1)
    certificate = load_mace_p0_certificate(REPO_ROOT)
    _synchronize(args.device)
    load_started = time.perf_counter()
    calculator = MACEPolCalculator(
        device=args.device,
        model="macepolm",
        implicit="smd",
        solvent="water",
        long_range_evaluator_profile=certificate.model_field_evaluator,
        route2_mace_geometry_frame_policy=certificate.coordinate_frame_policy,
        _implicit_solvent_factory_token=_IMPLICIT_SOLVENT_FACTORY_TOKEN,
    )
    _synchronize(args.device)
    model_load_seconds = time.perf_counter() - load_started
    plugin = DEFAULT_ROUTE2_PLUGIN_REGISTRY.create(
        "mace-polar-1",
        calculator,
        p0_certificate=certificate,
    )
    admission = admit_plugin(
        plugin,
        atomic_numbers=np.asarray(atoms.numbers, dtype=int),
        total_charge_e=0.0,
    )
    adapter = adapt_atomic_l1_plugin_to_legacy_engine(
        plugin,
        spec=LegacyAtomicL1BridgeSpec(
            model_family=plugin.plugin_id,
            field_evaluator=plugin.field_dual_space.name,
            profile_binding=admission.profile.name,
        ),
    )
    reaction_fields: list[Any] = []

    def reaction_field_factory(current_atoms):
        reaction_field = P_MINUS_1_WATER_CONTINUUM_SPEC.build_reaction_field(
            current_atoms
        )
        reaction_fields.append(reaction_field)
        return reaction_field

    engine = Route2ContinuumEngine(
        reaction_field_factory=reaction_field_factory,
        cds_evaluator=lambda _atoms: ElectrostaticsOnlyCDSResult(),
        settings=P_MINUS_1_WATER_CONTINUUM_SPEC.engine_settings(),
    )
    _synchronize(args.device)
    gas_started = time.perf_counter()
    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    _synchronize(args.device)
    gas_seconds = time.perf_counter() - gas_started
    _synchronize(args.device)
    scf_started = time.perf_counter()
    try:
        coupled = engine.solve_coupled_state(
            atoms,
            adapter,
            gas_state,
            provider_cache_signature=(
                ARTIFACT_ID,
                P_MINUS_1_WATER_CONTINUUM_SPEC.cavity_sha256(atoms),
            ),
            electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        )
    except Route2SCFConvergenceError as error:
        artifact = _failed_scf_payload(
            args=args,
            atoms=atoms,
            calculator=calculator,
            error=error,
            started=started,
        )
        _write_exclusive_json(output, artifact)
        print(output)
        print(json.dumps(artifact["decision"], sort_keys=True))
        return 0
    _synchronize(args.device)
    scf_seconds = time.perf_counter() - scf_started

    _synchronize(args.device)
    conjugacy_started = time.perf_counter()
    conjugacy_probes = _conjugacy_probes(plugin, atoms, coupled)
    _synchronize(args.device)
    conjugacy_seconds = time.perf_counter() - conjugacy_started
    selected_conjugacy = EnergySourceConjugacyEvidence(**conjugacy_probes[1])
    audit = build_operational_energy_audit(
        gas_state,
        coupled,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        standard_state_energy_hartree=0.0,
        energy_source_conjugacy=selected_conjugacy,
        pairing_identity_tolerance_ev=(
            P_MINUS_1_WATER_CONTINUUM_SPEC.engine_settings().energy_identity_tolerance_ev
        ),
    )
    git_status = _git("status", "--porcelain", "--untracked-files=all")
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "claim_boundary": (
            "This one-water, neutral-H/O, fixed-cavity, electrostatics-only "
            "MACE-POLAR/ddPCM run audits outer fixed-point residuals, one "
            "operational scalar ledger, and finite-field conjugacy. It is not "
            "P2/P3, a common functional, force/PES evidence, chemical accuracy, "
            "or arbitrary element/charge/solvent evidence."
        ),
        "execution": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_status_porcelain": git_status.splitlines(),
            "source_binding": (
                "clean-head-plus-source-hashes"
                if not git_status
                else "content-hash-bound-working-tree"
            ),
        },
        "repository_source_files_sha256": _source_hashes(),
        "checkpoint": dict(calculator.mace_polar_checkpoint_provenance or {}),
        "p0_certificate": {
            "artifact_id": certificate.artifact_id,
            "artifact_path": certificate.artifact_path,
            "artifact_sha256": certificate.artifact_sha256,
            "field_convention": certificate.field_convention,
            "blocked_routes": list(certificate.blocked_routes),
        },
        "protocol": {
            "molecule": "water",
            "geometry_source": "ASE G2 molecule collection",
            "geometry_sha256": _geometry_sha256(atoms),
            "atomic_numbers": np.asarray(atoms.numbers, dtype=int).tolist(),
            "positions_angstrom": np.asarray(atoms.positions, dtype=float).tolist(),
            "total_charge_e": 0.0,
            "continuum": P_MINUS_1_WATER_CONTINUUM_SPEC.as_provenance(),
            "cavity_sha256": P_MINUS_1_WATER_CONTINUUM_SPEC.cavity_sha256(atoms),
            "electrostatic_energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
            "standard_state_energy_hartree": 0.0,
            "response_mode": "scf",
            "scientific_route": "response-only",
            "energy_source_conjugacy_direction": CONJUGACY_DIRECTION,
            "force_requested": False,
            "p2_p3_execution_policy": "do-not-run-unless-p1-complete",
        },
        "gas_state": {
            "model_energy_ev": float(gas_state.energy_ev),
            "source_total_charge_e": float(
                np.sum(gas_state.density_coefficients[:, 0])
            ),
        },
        "outer_scf": {
            "history": list(coupled.history),
            "convergence": dict(coupled.scf_convergence),
        },
        "energy_source_conjugacy_step_stability": conjugacy_probes,
        "operational_audit": audit.as_dict(),
        "decision": {
            "p1_complete": audit.p1_complete,
            "blockers": list(audit.blockers),
            "energy_interpretation": audit.energy_interpretation,
            "inner_continuum_residual_status": audit.continuum_solve.status,
            "p2_p3_status": ("eligible" if audit.p1_complete else "blocked-by-p1"),
            "operational_force_status": "blocked-by-p0-and-profile",
            "variational_status": "blocked-by-p0-and-missing-functional",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "mace_torch": importlib.metadata.version("mace-torch"),
            "pyddx": pyddx.__version__,
            "pyddx_module": pyddx.__file__,
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name()
                if args.device == "cuda"
                else platform.processor()
            ),
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "timing_seconds": {
                "model_load": model_load_seconds,
                "gas_state": gas_seconds,
                "scf": scf_seconds,
                "energy_source_conjugacy": conjugacy_seconds,
                "total": time.perf_counter() - started,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(output)
    print(json.dumps(artifact["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
