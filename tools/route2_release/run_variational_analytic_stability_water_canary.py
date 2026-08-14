#!/usr/bin/env python3
"""Run the disabled analytic-model local stability/multi-start water canary."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from ase import Atoms

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1,
)
from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
)
from maple.solvation.continuum import SmoothWeightedHarmonicGalerkinFunctionalCandidate
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.variational_state import (
    build_variational_common_functional,
)
from maple.solvation.models import (
    MACEPolarVariationalFieldEnergy,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    VariationalStabilityThresholds,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    dense_matrix_from_action,
    runtime_record,
    variational_stability_diagnostic,
    write_external_json_artifact,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)

SCHEMA_VERSION = (
    "route2-variational-analytic-gaussian-multipole-water-local-stability-canary-v1"
)
ARTIFACT_KIND = (
    "disabled-real-checkpoint-analytic-model-local-stability-multistart-canary"
)
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
RANDOM_SEEDS = (20260816, 20260817)
ROOT_CONTEXT_PREFIX = "route2-variational-analytic-stability-water-v1"
ROOT_OPTIONS = FixedPointOptions(
    method="anderson",
    tolerance=2.0e-10,
    max_iterations=120,
    damping=0.5,
    history=6,
)
ROOT_SOURCE_RELATIVE_TOLERANCE = 1.0e-8
ROOT_FIELD_RELATIVE_TOLERANCE = 1.0e-8
ROOT_REDUCED_RELATIVE_TOLERANCE = 1.0e-8
ROOT_ENERGY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
LINEARIZATION_RELATIVE_TOLERANCE = 1.0e-10
STABILITY_THRESHOLDS = VariationalStabilityThresholds(
    symmetry_relative=1.0e-10,
    curvature_sign_relative=1.0e-8,
    absolute_eigenvalue=1.0e-12,
    model_invertibility_relative=1.0e-6,
    feedback_imaginary_relative=1.0e-10,
    feedback_spectral_radius_maximum=0.95,
    residual_minimum_singular_relative=1.0e-3,
    combined_positive_relative=1.0e-6,
)
SURFACE_LMAX = 1
EXPOSURE_LMAX = 2
TRANSITION_WIDTH_ANGSTROM2 = 0.18
EXPOSURE_RADIAL_QUADRATURE_ORDER = 32
SOURCE_RADIAL_QUADRATURE_ORDER = 32
GREEN_RADIAL_QUADRATURE_ORDER = 32
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/continuum/functional.py",
    "maple/solvation/continuum/harmonic_torch_functional.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/coupling/variational_adapters.py",
    "maple/solvation/coupling/variational_state.py",
    "maple/solvation/models/field_energy.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_variational.py",
    "maple/solvation/models/runtime/analytic_gaussian_multipole.py",
    "maple/solvation/release/variational_stability.py",
    "tools/route2_release/run_variational_analytic_stability_water_canary.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one disabled real-checkpoint local stability and multi-start "
            "canary; this does not admit Route-2 E/F/H/V/M."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def _configure_determinism(torch) -> None:
    torch.manual_seed(RANDOM_SEEDS[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEEDS[0])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _build_common(atoms: Atoms, checkpoint: Path, device: str):
    base = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    model = MACEPolarVariationalFieldEnergy(base)
    continuum = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=tuple(
            float(value)
            for value in smd_water_coulomb_radii(atoms.get_chemical_symbols())
        ),
        transition_width_angstrom2=TRANSITION_WIDTH_ANGSTROM2,
        surface_lmax=SURFACE_LMAX,
        exposure_lmax=EXPOSURE_LMAX,
        exposure_radial_quadrature_order=EXPOSURE_RADIAL_QUADRATURE_ORDER,
        source_radial_quadrature_order=SOURCE_RADIAL_QUADRATURE_ORDER,
        green_radial_quadrature_order=GREEN_RADIAL_QUADRATURE_ORDER,
        dtype=base._calculator.dtype,
        device=base._calculator.device,
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
    )
    common = build_variational_common_functional(
        model,
        continuum,
        atoms,
        scalar_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
        profile_id=(
            VARIATIONAL_MACEPOLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_ENERGYGRADIENT_SMOOTH_HARMONIC_GALERKIN_CPCM_PROFILE_V1
        ),
    )
    return base, model, continuum, common


def _normalized_random(dimension: int, seed: int, scale: float) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=dimension)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("deterministic multi-start vector is singular.")
    return float(scale) * values / norm


def _starts(dimension: int) -> dict[str, np.ndarray]:
    axis = np.zeros(dimension)
    axis[0] = 0.25
    random_a = _normalized_random(dimension, RANDOM_SEEDS[0], 0.25)
    random_b = _normalized_random(dimension, RANDOM_SEEDS[1], 0.50)
    return {
        "zero": np.zeros(dimension),
        "axis_positive_0p25": axis,
        "random_20260816_positive_0p25": random_a,
        "random_20260816_negative_0p25": -random_a,
        "random_20260817_positive_0p50": random_b,
    }


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second)) / max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )


def _multi_start_record(common, atoms: Atoms):
    starts = _starts(common.equation.reduced_dimension)
    records: dict[str, dict[str, object]] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    timings: dict[str, float] = {}
    for label, initial in starts.items():
        started = time.perf_counter()
        state = common.solve_state(
            atoms,
            root_context_id=f"{ROOT_CONTEXT_PREFIX}/{label}",
            initial_y=initial,
            options=ROOT_OPTIONS,
            require_convergence=False,
        )
        timings[label] = time.perf_counter() - started
        scalar = common.evaluate(atoms, state.y)
        y = state.y_array()
        source = state.source_array()
        field = state.field_array()
        arrays[label] = (y, source, field, scalar.total_energy_eV)
        records[label] = {
            "initial_sha256": canonical_json_sha256(initial.tolist()),
            "initial_norm": float(np.linalg.norm(initial)),
            "converged": bool(state.converged),
            "iterations": len(state.iterations) - 1,
            "actual_unmixed_residual_norm": state.actual_unmixed_residual_norm,
            "root_context_id": state.root_context_id,
            "root_hash": state.root_hash,
            "y": y.tolist(),
            "source": source.tolist(),
            "field": field.tolist(),
            "total_energy_eV": scalar.total_energy_eV,
        }

    pairwise: list[dict[str, object]] = []
    for first, second in itertools.combinations(records, 2):
        y_first, source_first, field_first, energy_first = arrays[first]
        y_second, source_second, field_second, energy_second = arrays[second]
        pairwise.append(
            {
                "first": first,
                "second": second,
                "reduced_relative_difference": _relative_difference(y_first, y_second),
                "source_relative_difference": _relative_difference(
                    source_first, source_second
                ),
                "field_relative_difference": _relative_difference(
                    field_first, field_second
                ),
                "energy_absolute_difference_eV": abs(energy_first - energy_second),
            }
        )
    maxima = {
        "reduced_relative_difference": max(
            record["reduced_relative_difference"] for record in pairwise
        ),
        "source_relative_difference": max(
            record["source_relative_difference"] for record in pairwise
        ),
        "field_relative_difference": max(
            record["field_relative_difference"] for record in pairwise
        ),
        "energy_absolute_difference_eV": max(
            record["energy_absolute_difference_eV"] for record in pairwise
        ),
    }
    gate = bool(
        all(record["converged"] for record in records.values())
        and all(
            record["actual_unmixed_residual_norm"] <= ROOT_OPTIONS.tolerance
            for record in records.values()
        )
        and maxima["reduced_relative_difference"] <= ROOT_REDUCED_RELATIVE_TOLERANCE
        and maxima["source_relative_difference"] <= ROOT_SOURCE_RELATIVE_TOLERANCE
        and maxima["field_relative_difference"] <= ROOT_FIELD_RELATIVE_TOLERANCE
        and maxima["energy_absolute_difference_eV"] <= ROOT_ENERGY_ABSOLUTE_TOLERANCE_EV
    )
    return (
        {
            "starts": records,
            "pairwise": pairwise,
            "maxima": maxima,
            "thresholds": {
                "reduced_relative": ROOT_REDUCED_RELATIVE_TOLERANCE,
                "source_relative": ROOT_SOURCE_RELATIVE_TOLERANCE,
                "field_relative": ROOT_FIELD_RELATIVE_TOLERANCE,
                "energy_absolute_eV": ROOT_ENERGY_ABSOLUTE_TOLERANCE_EV,
                "actual_unmixed_residual": ROOT_OPTIONS.tolerance,
            },
            "gate_passed": gate,
        },
        arrays["zero"],
        timings,
    )


def _stability_record(common, atoms: Atoms, primary):
    y, source, field, _ = primary
    coordinates = common.equation.coordinates
    reduced_field, gauge = common.model.duality_map.decompose_field(
        field,
        atom_count=len(atoms),
        total_charge=common.total_charge,
    )
    if not math.isfinite(gauge):
        raise RuntimeError("stability canary produced a non-finite gauge potential.")
    if common.total_charge != 0.0:
        raise RuntimeError("the preregistered water stability canary must be neutral.")
    dimension = coordinates.reduced_dimension
    sign = common.conjugacy_sign

    started = time.perf_counter()
    model_response = dense_matrix_from_action(
        lambda direction: sign
        * common.model.field_hvp(
            atoms,
            reduced_field,
            direction,
            total_charge=common.total_charge,
        ),
        dimension,
        name="model reduced susceptibility",
    )
    model_seconds = time.perf_counter() - started

    started = time.perf_counter()
    continuum_hessian = dense_matrix_from_action(
        lambda direction: coordinates.reduce_source_cotangent(
            common.continuum.source_hvp(
                atoms,
                source,
                coordinates.expand_direction(direction),
            )
        ),
        dimension,
        name="continuum reduced Hessian",
    )
    continuum_seconds = time.perf_counter() - started

    started = time.perf_counter()
    equation_jacobian = dense_matrix_from_action(
        lambda direction: common.equation.jvp(atoms, y, direction),
        dimension,
        name="state residual Jacobian",
    )
    equation_seconds = time.perf_counter() - started
    predicted = np.eye(dimension) - model_response @ continuum_hessian
    absolute = float(np.linalg.norm(equation_jacobian - predicted, ord="fro"))
    relative = absolute / max(
        float(np.linalg.norm(equation_jacobian, ord="fro")),
        float(np.linalg.norm(predicted, ord="fro")),
        np.finfo(float).tiny,
    )
    stability = variational_stability_diagnostic(
        model_response,
        continuum_hessian,
        thresholds=STABILITY_THRESHOLDS,
    )
    return (
        {
            "gauge_reduction": {
                "identity": "u=W xi+kappa g",
                "constant_potential_eV_per_e": float(gauge),
                "total_charge_e": common.total_charge,
                "constant_potential_coupling_energy_eV": float(
                    common.total_charge * gauge
                ),
                "reduced_fixed_charge_tangent_used": True,
            },
            "state_residual_factorization": {
                "identity": "J_r=I-J_M H_G",
                "absolute_frobenius_error": absolute,
                "relative_frobenius_error": relative,
                "threshold": LINEARIZATION_RELATIVE_TOLERANCE,
                "gate_passed": relative <= LINEARIZATION_RELATIVE_TOLERANCE,
            },
            "local_stability": stability,
        },
        {
            "model_dense_hvp_seconds": model_seconds,
            "continuum_dense_hvp_seconds": continuum_seconds,
            "equation_dense_jvp_seconds": equation_seconds,
        },
    )


def _measurement_record(
    *,
    protocol: dict[str, object],
    identity: dict[str, object],
    geometry: dict[str, object],
    multi_start: dict[str, object],
    stability: dict[str, object],
    decision: dict[str, object],
) -> dict[str, object]:
    """Return only deterministic scientific measurements, never wall time."""

    return {
        "protocol": protocol,
        "identity": identity,
        "geometry": geometry,
        "multi_start": multi_start,
        "stability": stability,
        "decision": decision,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    _configure_determinism(torch)
    atoms = _water()
    base, model, continuum, common = _build_common(atoms, checkpoint, args.device)
    multi_start, primary, solve_timings = _multi_start_record(common, atoms)
    stability, derivative_timings = _stability_record(common, atoms, primary)
    local_decisions = stability["local_stability"]["decisions"]
    decision = {
        "multi_start_root_gate_passed": bool(multi_start["gate_passed"]),
        "state_residual_factorization_gate_passed": bool(
            stability["state_residual_factorization"]["gate_passed"]
        ),
        "all_local_stability_gates_passed": bool(
            local_decisions["all_local_stability_gates_passed"]
        ),
        "one_water_local_stability_canary_passed": bool(
            multi_start["gate_passed"]
            and stability["state_residual_factorization"]["gate_passed"]
            and local_decisions["all_local_stability_gates_passed"]
        ),
        "global_passivity_proven": False,
        "global_root_uniqueness_proven": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }
    identity = {
        "scalar_id": common.scalar_id,
        "profile_id": common.profile_id,
        "state_equation_id": common.equation.state_equation_id,
        "common_functional_sha256": common.fingerprint_sha256(),
        "state_equation_sha256": common.equation.fingerprint_sha256(),
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_configuration_sha256": model.configuration_sha256(),
        "model_provenance_sha256": model.provenance_sha256,
        "continuum_provider_id": continuum.provider_id,
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "long_range_evaluator_profile": (
            base.release_contract.long_range_evaluator_profile
        ),
        "duality_map_sha256": model.duality_map.configuration_sha256(),
        "conjugacy_sign": common.conjugacy_sign,
    }
    geometry = {
        "formula": atoms.get_chemical_formula(),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "charge": 0,
        "multiplicity": 1,
        "cavity_radii_A": list(continuum.radii_angstrom),
    }
    protocol = {
        "random_seeds": list(RANDOM_SEEDS),
        "root_context_prefix": ROOT_CONTEXT_PREFIX,
        "root_options": {
            "method": ROOT_OPTIONS.method,
            "tolerance": ROOT_OPTIONS.tolerance,
            "max_iterations": ROOT_OPTIONS.max_iterations,
            "damping": ROOT_OPTIONS.damping,
            "history": ROOT_OPTIONS.history,
        },
        "start_contract": {
            "zero": 0.0,
            "axis_positive_0p25": 0.25,
            "random_20260816_positive_0p25": 0.25,
            "random_20260816_negative_0p25": 0.25,
            "random_20260817_positive_0p50": 0.50,
        },
        "harmonic_configuration": {
            "surface_lmax": SURFACE_LMAX,
            "exposure_lmax": EXPOSURE_LMAX,
            "transition_width_A2": TRANSITION_WIDTH_ANGSTROM2,
            "exposure_radial_quadrature_order": EXPOSURE_RADIAL_QUADRATURE_ORDER,
            "source_radial_quadrature_order": SOURCE_RADIAL_QUADRATURE_ORDER,
            "green_radial_quadrature_order": GREEN_RADIAL_QUADRATURE_ORDER,
        },
        "stability_thresholds": STABILITY_THRESHOLDS.as_dict(),
        "linearization_relative_tolerance": LINEARIZATION_RELATIVE_TOLERANCE,
    }
    measured = _measurement_record(
        protocol=protocol,
        identity=identity,
        geometry=geometry,
        multi_start=multi_start,
        stability=stability,
        decision=decision,
    )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": (
            "local-stability-canary-passed-not-admitted"
            if decision["one_water_local_stability_canary_passed"]
            else "local-stability-canary-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water local diagnostic checks five declared root starts, "
            "one model susceptibility, one continuum reduced Hessian, one "
            "residual factorization, and one combined Hessian. Passing cannot "
            "prove global passivity, root uniqueness, physical accuracy, PES "
            "smoothness, or Route-2 E/F/H/V/M admission."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": model.dtype,
        **measured,
        "measurement_sha256": canonical_json_sha256(measured),
        "solve_timings_seconds": solve_timings,
        "derivative_timings_seconds": derivative_timings,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_VARIATIONAL_ANALYTIC_STABILITY_WATER_CANARY="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": decision,
                "local_stability_decisions": local_decisions,
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
