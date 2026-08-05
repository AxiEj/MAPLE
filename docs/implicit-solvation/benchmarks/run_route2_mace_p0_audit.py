#!/usr/bin/env python3
"""Audit the MACE-POLAR P0 field contract without a continuum dependency.

The runner compares the official graph-level homogeneous-field interface with
MAPLE's nodewise ``[phi, grad(phi)]`` projection, then audits the exact
P-minus-1 coordinate-frame policy, batch isolation, energy/dipole conjugacy,
induced response, and an analytic Born sign canary.  It changes no model
weights and invokes no PCM, SMD/CDS, fit, or experimental solvation label.

Only the interface/symmetry gates can certify a field convention.  Failed
energy-conjugacy or response-reciprocity diagnostics remain in the artifact
and keep force and variational routes blocked.
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

_CPP_RUNTIME_REEXEC_MARKER = "MAPLE_ROUTE2_P0_CPP_RUNTIME_REEXEC"


def _ensure_process_cpp_runtime() -> None:
    """Preload the active environment's C++ runtime before binary imports."""

    candidate = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not candidate.is_file():
        raise RuntimeError(
            "The MACE-POLAR P0 audit requires libstdc++.so.6 in the active "
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
import torch

from maple.function.calculator.calculator_base import (
    _IMPLICIT_SOLVENT_FACTORY_TOKEN,
)
from maple.function.calculator.extra_correction.implicit.route2_mace_p0 import (
    MACE_P0_ARTIFACT_ID,
    MACE_P0_SCHEMA_VERSION,
    born_point_charge_sign_canary,
    composite_sha256,
    runtime_inference_file_hashes,
    sha256_file,
)
from maple.function.calculator.mace._macepol_calculator import (
    MACEPolCalculator,
)

RUNNER_RELATIVE_PATH = "docs/implicit-solvation/benchmarks/run_route2_mace_p0_audit.py"
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "docs/implicit-solvation/benchmarks/route2_mace_p0_live.py",
    ("maple/function/calculator/extra_correction/implicit/" "route2_mace_p0.py"),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_thermodynamic_diagnostics.py"
    ),
    ("maple/function/calculator/extra_correction/implicit/" "electrostatic_pairing.py"),
    ("maple/function/calculator/extra_correction/implicit/" "gto_density.py"),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_jgp94_mace_frame.py"
    ),
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/mace/_macepol_long_range.py",
)


from route2_mace_p0_live import (
    COORDINATE_FRAME_POLICY,
    MODEL_FIELD_EVALUATOR,
    RANDOM_SEED,
    _autograd_energy_dipole_conjugacy,
    _batch_isolation,
    _canonical_rotation_covariance,
    _canonical_translation_gauge,
    _induced_dipole_response,
    _source_response,
    _sync,
    _uniform_field_interface,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _geometry_sha256(atoms: ase.Atoms) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.asarray(atoms.positions, dtype=np.float64).tobytes())
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
    _sync(args.device)
    load_started = time.perf_counter()
    calculator = MACEPolCalculator(
        device=args.device,
        model="macepolm",
        implicit="smd",
        solvent="water",
        long_range_evaluator_profile=MODEL_FIELD_EVALUATOR,
        route2_mace_geometry_frame_policy=COORDINATE_FRAME_POLICY,
        _implicit_solvent_factory_token=_IMPLICIT_SOLVENT_FACTORY_TOKEN,
    )
    _sync(args.device)
    model_load_seconds = time.perf_counter() - load_started

    gates: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for name, evaluator in (
        ("uniform_field_interface", _uniform_field_interface),
        ("canonical_translation_gauge", _canonical_translation_gauge),
        ("canonical_rotation_covariance", _canonical_rotation_covariance),
        ("batch_isolation", _batch_isolation),
        ("autograd_energy_dipole_conjugacy", _autograd_energy_dipole_conjugacy),
        ("induced_dipole_response", _induced_dipole_response),
        ("source_response", _source_response),
    ):
        _sync(args.device)
        gate_started = time.perf_counter()
        gates[name] = evaluator(calculator, atoms)
        _sync(args.device)
        timings[name] = time.perf_counter() - gate_started
    gates["born_sign"] = born_point_charge_sign_canary()

    convention = gates["uniform_field_interface"]["selected_field_convention"]
    interface_certificate_passed = all(
        gates[name]["passed"]
        for name in (
            "uniform_field_interface",
            "canonical_translation_gauge",
            "canonical_rotation_covariance",
            "batch_isolation",
            "born_sign",
        )
    )
    nonvariational_diagnostics_passed = all(
        gates[name]["passed"]
        for name in (
            "autograd_energy_dipole_conjugacy",
            "induced_dipole_response",
            "source_response",
        )
    )
    if not interface_certificate_passed:
        admitted_route = "none"
    else:
        admitted_route = "response-only"

    checkpoint = dict(calculator.mace_polar_checkpoint_provenance or {})
    runtime_hashes = runtime_inference_file_hashes(calculator)
    git_status = _git("status", "--porcelain", "--untracked-files=all")
    artifact = {
        "artifact": MACE_P0_ARTIFACT_ID,
        "schema_version": MACE_P0_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "claim_boundary": (
            "This one-water, gas-phase MACE-POLAR interface audit can establish "
            "only the field sign/order, the exact named coordinate/evaluator "
            "policy, batch isolation, and reported derivative diagnostics for "
            "the hash-bound checkpoint and code. It invokes no continuum, CDS, "
            "solvation free energy, force/PES release, optimizer-quality claim, "
            "experimental label, fit, or arbitrary-element/charge certification."
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
        "runtime_inference_files_sha256": runtime_hashes,
        "inference_code_sha256": composite_sha256(runtime_hashes),
        "checkpoint": checkpoint,
        "protocol": {
            "molecule": "water",
            "geometry_source": "ASE G2 molecule collection",
            "geometry_sha256": _geometry_sha256(atoms),
            "atomic_numbers": np.asarray(atoms.numbers, dtype=int).tolist(),
            "positions_angstrom": np.asarray(atoms.positions, dtype=float).tolist(),
            "audited_atomic_numbers": sorted(set(map(int, atoms.numbers))),
            "audited_total_charge_domain_e": [0.0, 0.0],
            "coordinate_frame_policy": COORDINATE_FRAME_POLICY,
            "model_field_evaluator": MODEL_FIELD_EVALUATOR,
            "field_interface": "nodewise-[phi,grad(phi)]-v1",
            "uniform_field_gauge": "barycenter-zero-test-gauge",
            "random_seed": RANDOM_SEED,
            "threshold_status": (
                "versioned implementation thresholds; not a blinded benchmark"
            ),
        },
        "gates": gates,
        "optimizer_coverage": {
            "status": "unattestable",
            "reason": (
                "The released checkpoint and inference package do not bind a "
                "complete training run, optimizer parameter inventory, and step "
                "history from which coverage can be proved."
            ),
            "training_quality_conclusion": None,
        },
        "decision": {
            "field_interface_certificate_passed": interface_certificate_passed,
            "field_convention": convention,
            "physical_electric_field_relation": gates["uniform_field_interface"][
                "physical_electric_field_relation"
            ],
            "admitted_route": admitted_route,
            "blocked_routes": ["operational-force", "variational"],
            "scf_status": "requires-P1-residual-and-ledger-audit",
            "energy_semantics": (
                "response-conditioned-operational-prediction"
                if not gates["autograd_energy_dipole_conjugacy"]["passed"]
                else "conjugacy-canary-passed-but-no-common-functional-proved"
            ),
            "nonvariational_diagnostics_passed": (nonvariational_diagnostics_passed),
            "optimizer_coverage": "unattestable",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "mace_torch": importlib.metadata.version("mace-torch"),
            "graph_longrange": importlib.metadata.version("graph-longrange"),
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name()
                if args.device == "cuda"
                else platform.processor()
            ),
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "timing_seconds": {
                "model_load": model_load_seconds,
                **timings,
                "total": time.perf_counter() - started,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(output)
    print(json.dumps(artifact["decision"], sort_keys=True))
    return 0 if interface_certificate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
