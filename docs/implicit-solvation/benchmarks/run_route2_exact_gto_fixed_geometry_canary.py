#!/usr/bin/env python3
"""Run one source-bound exact-GTO fixed-geometry derivative canary.

The public MACE-POLAR/PCMSolver/IEFPCM exact-GTO profile first produces one
same-root acetone state.  The canary then reopens the identical fixed cavity
and composes only the rectangular continuum feature map, learned density
response, physical density-space energy right-hand side, and matrix-free
adjoint.  It never requests a coordinate derivative and cannot certify a
solution-phase PES, force, optimization, transition state, scan, or dynamics.
"""

from __future__ import annotations

import argparse
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
from typing import Any

import ase
from ase.units import Bohr, Hartree
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maple.function.calculator.extra_correction.implicit.continuum_response import (  # noqa: E402
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_field_projection import (  # noqa: E402
    ExactGTOFieldProjector,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (  # noqa: E402
    PCMSolverSession,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (  # noqa: E402
    fixed_cavity_model_feature_energy_density_gradient,
)
from maple.function.calculator.extra_correction.implicit.route2_feature_response import (  # noqa: E402
    FeatureDrivenUnmixedDensityResidualLinearization,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (  # noqa: E402
    FixedCavityPCMReactionFieldLinearMap,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (  # noqa: E402
    project_neutral_density_tangent,
    solve_adjoint,
)
from maple.function.calculator.extra_correction.implicit.smd import (  # noqa: E402
    PCM_WARNING_MARKER,
    _capture_process_stderr,
    _pcmsolver_audit_working_directory,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.command_control import CommandControl  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
from maple.function.route2_smd_profiles import (  # noqa: E402
    PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
)

ARTIFACT_ID = "route2-mace-exact-gto-fixed-geometry-canary-v1"
COMPOUND_ID = "mobley_3867265"
RANDOM_SEED = 20260727
FINITE_DIFFERENCE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)
IMPLEMENTATION_DOT_TOLERANCE = 1.0e-9
RUNNER_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "run_route2_exact_gto_fixed_geometry_canary.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    "maple/function/calculator/set_calculator.py",
    "maple/function/read/command_control.py",
    "maple/function/route2_smd_profiles.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    ("maple/function/calculator/extra_correction/implicit/" "continuum_response.py"),
    ("maple/function/calculator/extra_correction/implicit/" "electrostatic_pairing.py"),
    ("maple/function/calculator/extra_correction/implicit/" "gto_density.py"),
    ("maple/function/calculator/extra_correction/implicit/" "gto_field_projection.py"),
    ("maple/function/calculator/extra_correction/implicit/" "pcmsolver.py"),
    ("maple/function/calculator/extra_correction/implicit/" "route2_derivative.py"),
    ("maple/function/calculator/extra_correction/implicit/" "route2_engine.py"),
    (
        "maple/function/calculator/extra_correction/implicit/"
        "route2_feature_response.py"
    ),
    ("maple/function/calculator/extra_correction/implicit/" "route2_pcm_response.py"),
    ("maple/function/calculator/extra_correction/implicit/" "route2_response.py"),
    "maple/function/calculator/extra_correction/implicit/smd.py",
)
PUBLIC_SETTINGS = (
    "#model=macepol-m",
    "#sp(verbose=1)",
    (
        "#solv(implicit=water,method=smd,provider=pcmsolver,"
        f"profile={PCMSOLVER_EXACT_GTO_FIELD_PROFILE},response=scf,"
        "standard_state=1m,cavity_policy=fixed-stability-branch,"
        "experimental=true)"
    ),
)
PREPARED_PATH = REPO_ROOT / ".omx/benchmarks/route2-macepolar-smd/prepared.json"


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


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _require_clean_source() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "The exact-GTO canary requires a clean tracked checkout; "
            f"git reported:\n{status}"
        )
    for relative in SOURCE_RELATIVE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    return _git("rev-parse", "HEAD")


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS
    }


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Canary direction normalization failed.")
    return array / norm


def _dot_record(left: float, right: float) -> dict[str, float]:
    absolute = abs(left - right)
    scale = abs(left) + abs(right)
    return {
        "forward": left,
        "reverse": right,
        "absolute_error": absolute,
        "relative_error": 0.0 if scale <= 1.0e-30 else absolute / scale,
    }


def _assert_dot_test(name: str, record: dict[str, float]) -> None:
    if record["absolute_error"] > IMPLEMENTATION_DOT_TOLERANCE:
        raise RuntimeError(
            f"{name} failed its discrete transpose test "
            f"({record['absolute_error']:.3e})."
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


def _fixed_geometry_energy_ev(
    calculator,
    atoms,
    reaction_field: FixedCavityPCMReactionFieldLinearMap,
    density: np.ndarray,
) -> float:
    features = reaction_field.model_feature_jvp(density)
    state, _ = calculator.polar_state(
        atoms,
        model_field_features=features,
    )
    polarization_ev = reaction_field.scf_polarization_energy_hartree(density) * Hartree
    return float(state.energy_ev) + float(polarization_ev)


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
    if not PREPARED_PATH.is_file():
        raise FileNotFoundError(
            "The pinned Route-2 FreeSolv preparation is missing: " f"{PREPARED_PATH}"
        )

    prepared = json.loads(PREPARED_PATH.read_text(encoding="utf-8"))
    candidate = next(
        item for item in prepared["candidates"] if item["compound_id"] == COMPOUND_ID
    )
    mol2_path = (
        REPO_ROOT
        / ".omx/benchmarks/route2-macepolar-smd"
        / str(candidate["mol2_relative_path"])
    )
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    settings = CommandControl.from_settings(list(PUBLIC_SETTINGS)).as_dict()
    maple_output = work_dir / "maple.out"

    total_started = time.perf_counter()
    _sync(args.device)
    started = time.perf_counter()
    calculator = SetCalculator(
        args.device,
        settings["model"],
        str(maple_output),
        atoms=atoms,
        d4=bool(settings.get("d4", False)),
        implicit=settings["solv"]["method"],
        solvent=settings["solv"]["implicit"],
        model_options=settings.get("model_options"),
        solvation_options=settings["solv"],
        charge_options=settings.get("charge") or {},
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
        or getattr(provider, "provider", None) != "pcmsolver"
        or getattr(provider, "profile", None) != PCMSOLVER_EXACT_GTO_FIELD_PROFILE
    ):
        raise RuntimeError(
            "The public route did not construct the exact-GTO PCMSolver " "provider."
        )
    audit_dir = Path(correction.audit_dir)
    audit_json_path = audit_dir / "route2-result.json"
    audit_state_path = audit_dir / "route2-state.npz"
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    with np.load(audit_state_path) as state:
        root_density = np.asarray(
            state["root_density_coefficients"],
            dtype=float,
        )
        response_density = np.asarray(
            state["response_density_coefficients"],
            dtype=float,
        )
        public_field = np.asarray(
            state["reaction_field_values_ev"],
            dtype=float,
        )
        public_features = np.asarray(
            state["model_field_features"],
            dtype=float,
        )
        public_cavity_centers = np.asarray(
            state["cavity_centers_bohr"],
            dtype=float,
        )
        public_cavity_areas = np.asarray(
            state["cavity_areas_bohr2"],
            dtype=float,
        )
    parsed_input = getattr(provider, "_parsed_pcm_input_path", None)
    if not isinstance(parsed_input, Path) or not parsed_input.is_file():
        raise RuntimeError(
            "The public PCMSolver evaluation did not retain its parsed input."
        )

    diagnostic_dir = work_dir / "fixed-geometry-continuum"
    diagnostic_stderr = diagnostic_dir / "pcmsolver-stderr.log"
    diagnostic_dir.mkdir()
    diagnostic_started = time.perf_counter()
    with _pcmsolver_audit_working_directory(diagnostic_dir):
        with _capture_process_stderr(diagnostic_stderr):
            with PCMSolverSession(
                np.asarray(atoms.numbers, dtype=float),
                np.asarray(atoms.get_positions(), dtype=float) / Bohr,
                parsed_input,
            ) as session:
                cavity_centers = session.cavity_centers_bohr
                cavity_areas = session.cavity_areas_bohr2
                continuum_response = PCMSolverExternalMEPCavityResponse(
                    session,
                    cavity_radii_angstrom=provider.coulomb_radii_angstrom,
                )
                reaction_field = FixedCavityPCMReactionFieldLinearMap(
                    continuum_response,
                    np.asarray(atoms.get_positions(), dtype=float),
                    model_field_projector=ExactGTOFieldProjector(
                        calculator.route2_gto_field_projection_spec()
                    ),
                    model_field_gauge=provider.profile_spec.model_field_gauge,
                )
                drive = reaction_field.apply_scf_drive(root_density)
                if drive.model_field_features is None:
                    raise RuntimeError(
                        "The exact-GTO continuum drive omitted model features."
                    )
                features = np.asarray(
                    drive.model_field_features,
                    dtype=float,
                )
                field_values = np.asarray(
                    drive.density_dual_field_ev,
                    dtype=float,
                )

                _sync(args.device)
                started = time.perf_counter()
                fresh_state, _ = calculator.polar_state(
                    atoms,
                    model_field_features=features,
                )
                intrinsic_gradient = calculator.intrinsic_energy_model_feature_gradient(
                    atoms,
                    model_field_features=features,
                )
                density_response = calculator.linearize_density_response_features(
                    atoms,
                    model_field_features=features,
                )
                _sync(args.device)
                model_derivative_setup_seconds = time.perf_counter() - started

                rng = np.random.default_rng(RANDOM_SEED)
                density_direction = _normalized(
                    project_neutral_density_tangent(rng.normal(size=root_density.shape))
                )
                density_cotangent = _normalized(
                    project_neutral_density_tangent(rng.normal(size=root_density.shape))
                )
                feature_direction = _normalized(rng.normal(size=features.shape))
                feature_cotangent = _normalized(rng.normal(size=features.shape))

                continuum_jvp = reaction_field.model_feature_jvp(density_direction)
                continuum_vjp = reaction_field.model_feature_vjp(feature_cotangent)
                continuum_dot = _dot_record(
                    float(np.vdot(feature_cotangent, continuum_jvp)),
                    float(np.vdot(continuum_vjp, density_direction)),
                )
                _assert_dot_test("Continuum exact-GTO JVP/VJP", continuum_dot)

                model_jvp = density_response.jvp(feature_direction)
                model_vjp = density_response.vjp(density_cotangent)
                model_dot = _dot_record(
                    float(np.vdot(density_cotangent, model_jvp)),
                    float(np.vdot(model_vjp, feature_direction)),
                )
                _assert_dot_test("MACE feature JVP/VJP", model_dot)

                residual = FeatureDrivenUnmixedDensityResidualLinearization(
                    atom_count=len(atoms),
                    model_feature_map=reaction_field,
                    density_response=density_response,
                )
                residual_jvp = residual.jvp(density_direction)
                residual_vjp = residual.vjp(density_cotangent)
                residual_dot = _dot_record(
                    float(np.vdot(density_cotangent, residual_jvp)),
                    float(np.vdot(residual_vjp, density_direction)),
                )
                _assert_dot_test(
                    "Composed exact-GTO residual JVP/VJP",
                    residual_dot,
                )

                physical_rhs = fixed_cavity_model_feature_energy_density_gradient(
                    reaction_field,
                    reaction_field_values=field_values,
                    intrinsic_energy_feature_gradient=(intrinsic_gradient),
                )
                adjoint = solve_adjoint(
                    residual,  # type: ignore[arg-type]
                    physical_rhs,
                    relative_tolerance=1.0e-8,
                    absolute_tolerance=1.0e-10,
                    max_iterations=100,
                )

                analytic_directional = float(np.vdot(physical_rhs, density_direction))
                finite_differences = []
                _sync(args.device)
                started = time.perf_counter()
                for step in FINITE_DIFFERENCE_STEPS:
                    plus = _fixed_geometry_energy_ev(
                        calculator,
                        atoms,
                        reaction_field,
                        root_density + step * density_direction,
                    )
                    minus = _fixed_geometry_energy_ev(
                        calculator,
                        atoms,
                        reaction_field,
                        root_density - step * density_direction,
                    )
                    numerical = (plus - minus) / (2.0 * step)
                    absolute_error = abs(numerical - analytic_directional)
                    scale = abs(numerical) + abs(analytic_directional)
                    finite_differences.append(
                        {
                            "step_density_coefficient": step,
                            "central_difference_ev": numerical,
                            "analytic_ev": analytic_directional,
                            "absolute_error_ev": absolute_error,
                            "relative_error": (
                                0.0 if scale <= 1.0e-30 else absolute_error / scale
                            ),
                        }
                    )
                _sync(args.device)
                finite_difference_seconds = time.perf_counter() - started
                pcmsolver_library = session.library_source
    diagnostic_seconds = time.perf_counter() - diagnostic_started

    native_stderr = diagnostic_stderr.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if PCM_WARNING_MARKER in native_stderr:
        raise RuntimeError(
            "The reopened fixed-cavity diagnostic emitted a native "
            "PCMSolver warning."
        )
    if not np.array_equal(cavity_centers, public_cavity_centers):
        raise RuntimeError(
            "The reopened PCMSolver cavity centers differ from the public root."
        )
    if not np.array_equal(cavity_areas, public_cavity_areas):
        raise RuntimeError(
            "The reopened PCMSolver cavity areas differ from the public root."
        )

    fresh_density = np.asarray(
        fresh_state.density_coefficients,
        dtype=float,
    )
    solvation = calculator.results.get("solvation")
    if not isinstance(solvation, dict):
        raise RuntimeError("The public finalizer omitted Route-2 solvation.")
    checkpoint = dict(getattr(calculator, "mace_polar_checkpoint_provenance", {}) or {})
    total_seconds = time.perf_counter() - total_started
    artifact: dict[str, Any] = {
        "artifact": ARTIFACT_ID,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "execution_git_head": git_head,
        "source_files_sha256": _source_hashes(),
        "scientific_identity": {
            "solute_response_model": "official MACE-POLAR-1-M",
            "density_interpretation": (
                "coarse-grained residual/net-charge l<=1 coefficients"
            ),
            "solute_source": "point-multipole-l1",
            "continuum": "PCMSolver IEFPCM",
            "reaction_field_projector": "exact-gto-v1",
            "model_field_gauge": "atomic-center-mean-zero-v1",
            "nonpolar": "native water SMD-CDS in public energy only",
            "strict_original_smd_equivalence": False,
        },
        "claim_boundary": (
            "This one-acetone canary checks only fixed-geometry derivative "
            "algebra for the exact-GTO model drive. It changes no production "
            "energy or force and sets no scientific pass threshold. It does "
            "not provide an exact-GTO coordinate or gauge derivative, full "
            "electron-density PCM, original SMD equivalence, hydration "
            "accuracy, a smooth solution-phase PES, OPT, TS, scan, MD, or "
            "NVE certification."
        ),
        "diagnostic_protocol": {
            "random_seed": RANDOM_SEED,
            "finite_difference_steps_density_coefficient": list(
                FINITE_DIFFERENCE_STEPS
            ),
            "implementation_dot_tolerance": (IMPLEMENTATION_DOT_TOLERANCE),
            "scientific_pass_thresholds": None,
            "coordinate_derivative_requested": False,
        },
        "public_settings": list(PUBLIC_SETTINGS),
        "molecule": {
            "compound_id": COMPOUND_ID,
            "name": str(candidate["name"]),
            "partition": str(candidate["partition"]),
            "experimental_kcal_mol": float(candidate["experimental_kcal_mol"]),
            "experimental_uncertainty_kcal_mol": float(
                candidate["experimental_uncertainty_kcal_mol"]
            ),
            "mol2_sha256": _sha256(mol2_path),
            "prepared_manifest_sha256": _sha256(PREPARED_PATH),
            "atom_count": len(atoms),
        },
        "checkpoint": checkpoint,
        "same_root": {
            "scf_iterations": int(audit["iterations"]),
            "stored_unmixed_density_residual_inf_e": float(
                np.max(np.abs(response_density - root_density))
            ),
            "fresh_unmixed_density_residual_inf_e": float(
                np.max(np.abs(fresh_density - root_density))
            ),
            "root_density_monopole_sum_e": float(np.sum(root_density[:, 0])),
            "response_density_monopole_sum_e": float(np.sum(response_density[:, 0])),
            "public_to_reopened_field_max_abs_ev": float(
                np.max(np.abs(public_field - field_values))
            ),
            "public_to_reopened_feature_max_abs": float(
                np.max(np.abs(public_features - features))
            ),
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
        "fixed_geometry_state": {
            "density_shape": list(root_density.shape),
            "model_feature_shape": list(features.shape),
            "cavity_size": int(cavity_centers.shape[0]),
            "cavity_centers_sha256": _sha256_array(cavity_centers),
            "cavity_areas_sha256": _sha256_array(cavity_areas),
            "physical_rhs_l2_ev": float(np.linalg.norm(physical_rhs)),
            "physical_rhs_monopole_sum_ev": float(np.sum(physical_rhs[:, 0])),
            "adjoint_solution_l2": float(np.linalg.norm(adjoint.solution)),
        },
        "diagnostics": {
            "continuum_feature_jvp_vjp": continuum_dot,
            "mace_feature_density_jvp_vjp": model_dot,
            "composed_residual_jvp_vjp": residual_dot,
            "physical_rhs_directional_finite_difference": (finite_differences),
            "adjoint": {
                "method": adjoint.method,
                "residual_callback_count": (adjoint.residual_callback_count),
                "operator_applications": adjoint.operator_applications,
                "residual_norm": adjoint.residual_norm,
                "relative_residual": adjoint.relative_residual,
                "restart_size": adjoint.restart_size,
                "maximum_inner_iterations": (adjoint.maximum_inner_iterations),
            },
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
            "pcmsolver_library": pcmsolver_library,
            "pcmsolver_library_environment": os.environ.get("PCMSOLVER_LIBRARY"),
            "pcmsolver_python_path_environment": os.environ.get(
                "PCMSOLVER_PYTHON_PATH"
            ),
            "timing_seconds": {
                "model_load": model_load_seconds,
                "public_coupled_root_and_energy": public_root_seconds,
                "model_derivative_setup": model_derivative_setup_seconds,
                "fixed_geometry_continuum_and_derivatives": (diagnostic_seconds),
                "directional_finite_differences": (finite_difference_seconds),
                "total": total_seconds,
            },
        },
    }
    _write_exclusive_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
