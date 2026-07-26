#!/usr/bin/env python3
"""Run one bounded MACE-POLAR local-field thermodynamic diagnostic.

The canary uses MAPLE's public MACE-POLAR/pyddx/ddPCM construction and the
shared fixed-point engine.  It then interrogates only the learned local
potential/gradient response at the converged field.  No diagnostic changes the
SCF root, scalar solvation energy, production force, or public profile.

This runner deliberately evaluates one small water geometry.  It is not an
accuracy benchmark and cannot certify the exact-GTO energy-only profile, a
solution-phase PES, optimization, transition states, scans, or dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


_CPP_RUNTIME_REEXEC_MARKER = "MAPLE_ROUTE2_CPP_RUNTIME_REEXEC"


def _ensure_process_cpp_runtime() -> None:
    """Load one modern C++ runtime before optional binary extensions.

    The validated pyddx wheel resolves ``libstdc++.so.6`` from the host while
    the CUDA/PyTorch and SciPy stacks require the newer runtime in the active
    environment.  Loading both SONAME-compatible libraries in the opposite
    order can segfault before Python can raise an exception.  Re-exec once,
    before importing any third-party extension, with the active environment's
    runtime preloaded.
    """

    candidate = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not candidate.is_file():
        raise RuntimeError(
            "The Route-2 canary requires libstdc++.so.6 in the active "
            f"Python environment; missing: {candidate}"
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
        raise RuntimeError(
            "The Route-2 canary failed to preload the active environment's "
            "C++ runtime."
        )
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = (
        resolved if not current else resolved + ":" + current
    )
    environment[_CPP_RUNTIME_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )


if __name__ == "__main__":
    _ensure_process_cpp_runtime()


import ase
from ase.build import molecule
import numpy as np
# Load PyTorch's C++ runtime before the separately built pyddx extension.  The
# validated Route-2 launchers use this order to avoid a process-wide libstdc++
# collision between the CUDA model stack and the optional continuum runtime.
import torch
import pyddx

from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    SCF_DENSITY_TOLERANCE,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_thermodynamic_diagnostics import (
    energy_density_conjugacy_diagnostic,
    field_loop_work_diagnostic,
    response_reciprocity_diagnostic,
    response_stability_diagnostic,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.route2_smd_profiles import DDPCM_SMD_PROFILE


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ID = "route2-mace-local-field-thermodynamic-canary-v1"
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_mace_thermodynamic_canary.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "maple/function/calculator/set_calculator.py",
    "maple/function/read/command_control.py",
    "maple/function/route2_smd_profiles.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    (
        "maple/function/calculator/extra_correction/implicit/"
        "ddpcm_smd.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "electrostatic_pairing.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_engine.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_response.py"
    ),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_thermodynamic_diagnostics.py"
    ),
)
PUBLIC_SETTINGS = (
    "#model=macepol-m",
    "#sp(verbose=1)",
    (
        "#solv(implicit=water,method=smd,provider=pyddx,"
        f"profile={DDPCM_SMD_PROFILE},response=scf,"
        "standard_state=1m,experimental=true)"
    ),
)
RANDOM_SEED = 20260727
RECIPROCITY_DIRECTION_PAIRS = 3
STABILITY_EIGENVALUE_TOLERANCE_EV = 1.0e-10
LINEARIZATION_ADJOINT_TOLERANCE_EV = 1.0e-10
LOOP_STEP_SCALES = (1.0e-3, 5.0e-4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(
        np.asarray(atoms.get_positions(), dtype=np.float64).tobytes()
    )
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The thermodynamic canary requires a clean tracked source "
            f"checkout; git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _scaled_unit_direction(
    rng: np.random.Generator,
    atom_count: int,
    *,
    potential_scale_ev: float,
    gradient_scale_ev_per_angstrom: float,
) -> np.ndarray:
    raw = rng.normal(size=4 * atom_count)
    raw /= np.linalg.norm(raw)
    direction = raw.reshape(atom_count, 4)
    direction[:, 0] *= potential_scale_ev
    direction[:, 1:] *= gradient_scale_ev_per_angstrom
    return direction


def _orthogonal_loop_directions(
    rng: np.random.Generator,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = rng.normal(size=4 * atom_count)
    first /= np.linalg.norm(first)
    second = rng.normal(size=4 * atom_count)
    second -= float(np.dot(second, first)) * first
    second /= np.linalg.norm(second)
    return (
        first.reshape(atom_count, 4),
        second.reshape(atom_count, 4),
    )


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
    git_head = _require_clean_source()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    total_started = time.perf_counter()
    atoms = molecule("H2O")
    atoms.info.update(
        charge=0,
        mult=1,
        mol2={"atom_types": ["o", "h", "h"]},
    )
    parameters = CommandControl.from_settings(
        list(PUBLIC_SETTINGS)
    ).as_dict()
    maple_output = work_dir / "maple.out"

    _sync(args.device)
    started = time.perf_counter()
    calculator = SetCalculator(
        args.device,
        parameters["model"],
        str(maple_output),
        atoms=atoms,
        d4=bool(parameters.get("d4", False)),
        implicit=parameters["solv"]["method"],
        solvent=parameters["solv"]["implicit"],
        model_options=parameters.get("model_options"),
        solvation_options=parameters["solv"],
        charge_options=parameters.get("charge") or {},
    ).set_calculator()
    _sync(args.device)
    model_load_seconds = time.perf_counter() - started

    atoms.calc = calculator
    _sync(args.device)
    started = time.perf_counter()
    combined_energy_hartree = float(atoms.get_potential_energy())
    _sync(args.device)
    public_root_seconds = time.perf_counter() - started

    correction = getattr(calculator, "solvent_correction", None)
    provider = getattr(correction, "provider", None)
    if (
        provider is None
        or getattr(provider, "provider", None) != "pyddx"
        or getattr(provider, "profile", None) != DDPCM_SMD_PROFILE
    ):
        raise RuntimeError(
            "The public route did not construct the pyddx/ddPCM provider."
        )
    if provider.profile_spec.reaction_field_projector != "local-jet":
        raise RuntimeError(
            "This diagnostic is valid only for the default differentiable "
            "local potential/gradient interface."
        )

    audit_dir = Path(correction.audit_dir)
    audit_json_path = audit_dir / "route2-ddpcm-result.json"
    audit_state_path = audit_dir / "route2-ddpcm-state.npz"
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    with np.load(audit_state_path) as state:
        root_density = np.asarray(
            state["density_coefficients"],
            dtype=float,
        )
        field = np.asarray(
            state["reaction_field_values_ev"],
            dtype=float,
        )
    _sync(args.device)
    started = time.perf_counter()
    response_state, _ = calculator.polar_state(
        atoms,
        node_potential_ev=field[:, 0],
        node_gradient_ev_per_angstrom=field[:, 1:],
    )
    _sync(args.device)
    same_field_response_seconds = time.perf_counter() - started
    response_density = np.asarray(
        response_state.density_coefficients,
        dtype=float,
    )
    root_response_residual_inf = float(
        np.max(np.abs(response_density - root_density))
    )
    if root_response_residual_inf > max(
        10.0 * SCF_DENSITY_TOLERANCE,
        1.0e-10,
    ):
        raise RuntimeError(
            "The fresh same-field MACE response does not reproduce the "
            "public coupled root "
            f"(residual={root_response_residual_inf:.3e} e)."
        )

    _sync(args.device)
    started = time.perf_counter()
    intrinsic_gradient = calculator.intrinsic_energy_field_gradient(
        atoms,
        node_potential_ev=field[:, 0],
        node_gradient_ev_per_angstrom=field[:, 1:],
    )
    _sync(args.device)
    intrinsic_gradient_seconds = time.perf_counter() - started

    started = time.perf_counter()
    density_response = calculator.linearize_density_response(
        atoms,
        node_potential_ev=field[:, 0],
        node_gradient_ev_per_angstrom=field[:, 1:],
    )
    linearization_setup_seconds = time.perf_counter() - started

    _sync(args.device)
    started = time.perf_counter()
    conjugacy = energy_density_conjugacy_diagnostic(
        reaction_field_values_ev=field,
        response_density_coefficients=response_density,
        intrinsic_energy_field_gradient=intrinsic_gradient,
        density_response=density_response,
    )
    _sync(args.device)
    conjugacy_seconds = time.perf_counter() - started

    rng = np.random.default_rng(RANDOM_SEED)
    reciprocal_records = []
    _sync(args.device)
    started = time.perf_counter()
    for index in range(RECIPROCITY_DIRECTION_PAIRS):
        first = _scaled_unit_direction(
            rng,
            len(atoms),
            potential_scale_ev=1.0,
            gradient_scale_ev_per_angstrom=1.0,
        )
        second = _scaled_unit_direction(
            rng,
            len(atoms),
            potential_scale_ev=1.0,
            gradient_scale_ev_per_angstrom=1.0,
        )
        second_response = density_response.jvp(second)
        first_density_cotangent = external_field_to_density_order(first)
        first_field_cotangent = density_response.vjp(
            first_density_cotangent
        )
        adjoint_forward = float(
            np.vdot(first_density_cotangent, second_response)
        )
        adjoint_reverse = float(
            np.vdot(first_field_cotangent, second)
        )
        adjoint_absolute = abs(adjoint_forward - adjoint_reverse)
        adjoint_scale = abs(adjoint_forward) + abs(adjoint_reverse)
        adjoint_relative = (
            0.0
            if adjoint_scale <= 1.0e-30
            else adjoint_absolute / adjoint_scale
        )
        if adjoint_absolute > LINEARIZATION_ADJOINT_TOLERANCE_EV:
            raise RuntimeError(
                "The MACE response JVP/VJP failed its implementation "
                f"adjoint check ({adjoint_absolute:.3e} eV)."
            )
        reciprocal_records.append(
            {
                "pair_index": index,
                "first_field_direction": first.tolist(),
                "second_field_direction": second.tolist(),
                "linearization_adjoint_dot_test": {
                    "forward_dot_ev": adjoint_forward,
                    "reverse_dot_ev": adjoint_reverse,
                    "absolute_error_ev": adjoint_absolute,
                    "relative_error": adjoint_relative,
                    "tolerance_ev": (
                        LINEARIZATION_ADJOINT_TOLERANCE_EV
                    ),
                    "passed": True,
                },
                **asdict(
                    response_reciprocity_diagnostic(
                        density_response,
                        first,
                        second,
                    )
                ),
            }
        )
    _sync(args.device)
    reciprocity_seconds = time.perf_counter() - started

    _sync(args.device)
    started = time.perf_counter()
    stability = response_stability_diagnostic(
        density_response,
        atom_count=len(atoms),
        potential_direction_scale_ev=1.0,
        gradient_direction_scale_ev_per_angstrom=1.0,
        eigenvalue_tolerance_ev=STABILITY_EIGENVALUE_TOLERANCE_EV,
    )
    _sync(args.device)
    stability_seconds = time.perf_counter() - started

    loop_first, loop_second = _orthogonal_loop_directions(rng, len(atoms))
    density_cache = {
        np.ascontiguousarray(field).tobytes(): response_density.copy()
    }
    fresh_loop_evaluations = 0

    def density_evaluator(field_values: np.ndarray) -> np.ndarray:
        nonlocal fresh_loop_evaluations
        values = np.ascontiguousarray(field_values, dtype=float)
        key = values.tobytes()
        cached = density_cache.get(key)
        if cached is not None:
            return cached.copy()
        state, _ = calculator.polar_state(
            atoms,
            node_potential_ev=values[:, 0],
            node_gradient_ev_per_angstrom=values[:, 1:],
        )
        density = np.asarray(state.density_coefficients, dtype=float)
        density_cache[key] = density.copy()
        fresh_loop_evaluations += 1
        return density

    loop_records = []
    loop_seconds = []
    for step_scale in LOOP_STEP_SCALES:
        first_step = step_scale * loop_first
        second_step = step_scale * loop_second
        _sync(args.device)
        started = time.perf_counter()
        diagnostic = field_loop_work_diagnostic(
            density_evaluator,
            base_field=field,
            first_field_step=first_step,
            second_field_step=second_step,
        )
        _sync(args.device)
        elapsed = time.perf_counter() - started
        loop_seconds.append(elapsed)
        loop_records.append(
            {
                "potential_step_scale_ev": step_scale,
                "gradient_step_scale_ev_per_angstrom": step_scale,
                "first_field_step": first_step.tolist(),
                "second_field_step": second_step.tolist(),
                **asdict(diagnostic),
            }
        )

    solvation = calculator.results.get("solvation")
    if not isinstance(solvation, dict):
        raise RuntimeError("The public Route-2 finalizer omitted solvation.")
    total_seconds = time.perf_counter() - total_started
    checkpoint = dict(
        getattr(calculator, "mace_polar_checkpoint_provenance", {}) or {}
    )
    artifact = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "scientific_identity": {
            "solute_response_model": "official MACE-POLAR-1-M",
            "density_interpretation": (
                "coarse-grained residual/net-charge l<=1 coefficients"
            ),
            "solute_source": "point-multipole-l1",
            "continuum": "pyddx ddPCM",
            "reaction_field_projector": "local-jet",
            "field_interface": "local-potential-gradient-v1",
            "cds_present_in_public_energy": True,
            "cds_used_in_diagnostics": False,
            "strict_original_smd_equivalence": False,
        },
        "claim_boundary": (
            "This one-water canary discriminates two energy-density "
            "conjugacy identities and measures local-field response "
            "reciprocity, a small-system susceptibility spectrum, and "
            "field-loop work. It changes no production energy or force and "
            "sets no scientific pass threshold. It does not validate the "
            "exact-GTO energy-only response, full electron-density PCM, "
            "original SMD equivalence, hydration accuracy, coordinate-force "
            "smoothness, a solution-phase PES, OPT, TS, scan, MD, or NVE."
        ),
        "diagnostic_protocol": {
            "random_seed": RANDOM_SEED,
            "reciprocity_direction_pairs": (
                RECIPROCITY_DIRECTION_PAIRS
            ),
            "reciprocity_potential_direction_scale_ev": 1.0,
            "reciprocity_gradient_direction_scale_ev_per_angstrom": 1.0,
            "stability_potential_direction_scale_ev": 1.0,
            "stability_gradient_direction_scale_ev_per_angstrom": 1.0,
            "stability_eigenvalue_tolerance_ev": (
                STABILITY_EIGENVALUE_TOLERANCE_EV
            ),
            "linearization_adjoint_tolerance_ev": (
                LINEARIZATION_ADJOINT_TOLERANCE_EV
            ),
            "loop_step_scales": list(LOOP_STEP_SCALES),
            "loop_quadrature": "endpoint-trapezoid-v1",
            "pass_thresholds": None,
        },
        "public_settings": list(PUBLIC_SETTINGS),
        "profile": {
            "name": provider.profile,
            "provenance": provider.provenance,
        },
        "molecule": {
            "name": "water",
            "formula": atoms.get_chemical_formula(),
            "atom_count": len(atoms),
            "geometry_source": "ASE G2 molecule collection",
            "geometry_sha256": _geometry_sha256(atoms),
            "positions_angstrom": np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).tolist(),
        },
        "checkpoint": checkpoint,
        "same_root": {
            "scf_iterations": int(audit["scf"]["iterations"]),
            "unmixed_density_residual_inf_e": (
                root_response_residual_inf
            ),
            "half_coupling_identity_error_ev": (
                float(audit["polarization_energy_identity_error_ev"])
            ),
            "root_density_monopole_sum_e": float(
                np.sum(root_density[:, 0])
            ),
            "response_density_monopole_sum_e": float(
                np.sum(response_density[:, 0])
            ),
            "history": audit["scf"]["history"],
            "public_audit": {
                "result_sha256": _sha256(audit_json_path),
                "state_sha256": _sha256(audit_state_path),
            },
        },
        "public_energy_hartree": {
            "gas": float(solvation["gas_energy_hartree"]),
            "delta_g_solv": float(solvation["delta_g_solv_hartree"]),
            "combined": combined_energy_hartree,
        },
        "field_summary": {
            "potential_min_ev": float(np.min(field[:, 0])),
            "potential_max_ev": float(np.max(field[:, 0])),
            "gradient_l2_ev_per_angstrom": float(
                np.linalg.norm(field[:, 1:])
            ),
        },
        "state_arrays": {
            "root_density_coefficients": root_density.tolist(),
            "response_density_coefficients": response_density.tolist(),
            "reaction_field_values_ev": field.tolist(),
            "intrinsic_energy_field_gradient": np.asarray(
                intrinsic_gradient,
                dtype=float,
            ).tolist(),
        },
        "diagnostics": {
            "energy_density_conjugacy": asdict(conjugacy),
            "response_reciprocity": reciprocal_records,
            "response_stability": asdict(stability),
            "field_loop_work_refinement": loop_records,
            "field_loop_fresh_model_evaluations": (
                fresh_loop_evaluations
            ),
            "interpretation_policy": (
                "Report both candidate defects and all raw diagnostics. "
                "The numerically lower defect is not a pass or proof of a "
                "variational free-energy functional."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "pyddx": pyddx.__version__,
            "pyscf": importlib.metadata.version("pyscf"),
            "mace_torch": importlib.metadata.version("mace-torch"),
            "graph_longrange": importlib.metadata.version(
                "graph-longrange"
            ),
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name()
                if args.device == "cuda"
                else platform.processor()
            ),
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "pyddx_extension": {
                "path": str(Path(pyddx.__file__).resolve()),
                "sha256": _sha256(Path(pyddx.__file__).resolve()),
            },
            "timing_seconds": {
                "model_load": model_load_seconds,
                "public_coupled_root_and_energy": public_root_seconds,
                "same_field_response_state": (
                    same_field_response_seconds
                ),
                "intrinsic_energy_field_gradient": (
                    intrinsic_gradient_seconds
                ),
                "density_linearization_setup": (
                    linearization_setup_seconds
                ),
                "energy_density_conjugacy": conjugacy_seconds,
                "reciprocity_three_pairs": reciprocity_seconds,
                "stability_twelve_jvps": stability_seconds,
                "field_loop_work_by_step": loop_seconds,
                "total": total_seconds,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
