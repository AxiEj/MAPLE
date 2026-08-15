#!/usr/bin/env python3
"""Run the final coupling-active-space audit for original MACE-POLAR source.

The canary keeps the checkpoint's source4 and native receiver8 spaces distinct,
projects both through the actual smooth harmonic boundary operators, and tests
the remaining quotient-space integrability loophole.  A coupled-curl failure
is terminal even while the intrinsic-energy external-enthalpy semantics remain
unverified.  No capability is admitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

from ase import Atoms
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    get_scalar_definition,
)
from maple.solvation.continuum import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.coupling.separated_operators import (
    build_mace_polar_harmonic_separated_snapshot,
)
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    NativeSemanticsCanary,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    analyze_coupled_conjugacy,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-coupled-conjugacy-terminal-v1"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
FIELD_SCALE = 5.0e-3
RANDOM_SEED = 20260815
SURFACE_LMAX = 1
EXPOSURE_LMAX = 2
TRANSITION_WIDTH_ANGSTROM2 = 0.18
RADIAL_QUADRATURE_ORDER = 32
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/continuum/harmonic_torch_functional.py",
    "maple/solvation/continuum/harmonic_torch_primitives.py",
    "maple/solvation/coupling/separated_operators.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/coupled_conjugacy.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/extra_correction/implicit/gto_field_projection.py",
    "tools/route2_release/run_mace_coupled_conjugacy_terminal.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "H2O",
        positions=np.asarray(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9266, 0.0000],
            ],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def _normalized(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("deterministic audit direction is singular.")
    return result / norm


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _build(atoms: Atoms, checkpoint: Path, device: str):
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    radii = tuple(
        float(value) for value in smd_water_coulomb_radii(atoms.get_chemical_symbols())
    )
    continuum = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=radii,
        transition_width_angstrom2=TRANSITION_WIDTH_ANGSTROM2,
        surface_lmax=SURFACE_LMAX,
        exposure_lmax=EXPOSURE_LMAX,
        exposure_radial_quadrature_order=RADIAL_QUADRATURE_ORDER,
        source_radial_quadrature_order=RADIAL_QUADRATURE_ORDER,
        green_radial_quadrature_order=RADIAL_QUADRATURE_ORDER,
        dtype=radial.dtype,
        device=radial.device,
        scalar_id=(
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ),
    )
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    snapshot = build_mace_polar_harmonic_separated_snapshot(continuum, atoms)
    return radial, electronic, continuum, snapshot


def _state_record(electronic, snapshot, atoms: Atoms, field: np.ndarray, rng):
    source = electronic.evaluate_source(atoms, field).reshape(-1)
    gradient = electronic.intrinsic_energy_field_gradient(atoms, field).reshape(-1)
    jacobian = electronic.dense_source_jacobian(atoms, field)

    direction = _normalized(rng.normal(size=field.shape))
    cotangent = _normalized(rng.normal(size=source.shape)).reshape(len(atoms), 4)
    jvp = electronic.field_jvp(atoms, field, direction).reshape(-1)
    vjp = electronic.field_vjp(atoms, field, cotangent).reshape(-1)
    dense_jvp = jacobian @ direction.reshape(-1)
    jvp_dense_error = float(np.linalg.norm(jvp - dense_jvp))
    dot_left = float(np.vdot(jvp, cotangent.reshape(-1)))
    dot_right = float(np.vdot(direction.reshape(-1), vjp))
    dot_error = abs(dot_left - dot_right)
    if jvp_dense_error > 2.0e-10 or dot_error > 2.0e-10:
        raise RuntimeError("separated source Jacobian implementation check failed.")

    charge_covector = np.zeros(source.size)
    charge_covector[::4] = 1.0
    charge_response = charge_covector @ jacobian
    charge_response_norm = float(np.linalg.norm(charge_response))

    signs = {
        str(sign): analyze_coupled_conjugacy(
            source_to_boundary=snapshot.source_to_boundary,
            boundary_to_native_field=snapshot.boundary_to_native_field,
            original_source=source,
            intrinsic_energy_native_field_gradient=gradient,
            source_native_field_jacobian=jacobian,
            boundary_metric=snapshot.surface_operator,
            energy_gradient_sign=sign,
            energy_semantics_verified=False,
        ).as_dict()
        for sign in (-1, 1)
    }
    return {
        "field_norm": float(np.linalg.norm(field)),
        "source_norm": float(np.linalg.norm(source)),
        "intrinsic_energy_gradient_norm": float(np.linalg.norm(gradient)),
        "fixed_charge_response_l2": charge_response_norm,
        "dense_jvp_l2_error": jvp_dense_error,
        "jvp_vjp_dot_absolute_error": dot_error,
        "signs": signs,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    definition = get_scalar_definition(
        OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    )
    if definition.enabled or definition.admitted_capabilities.enabled_tiers:
        raise RuntimeError("the separated operational scalar must remain disabled.")
    started = time.perf_counter()
    import torch

    _configure_determinism(torch)
    atoms = _water()
    radial, electronic, continuum, snapshot = _build(atoms, checkpoint, args.device)
    canary = NativeSemanticsCanary.from_adapter(radial)
    if not canary.separated_operational_contract_complete:
        raise RuntimeError("live checkpoint failed the source4/receiver8 canary.")

    rng = np.random.default_rng(RANDOM_SEED)
    boundary_direction = _normalized(rng.normal(size=snapshot.boundary_dimension))
    active_field = snapshot.native_field_from_boundary(boundary_direction)
    active_field *= FIELD_SCALE / float(np.linalg.norm(active_field))
    states = {
        "zero-field": _state_record(
            electronic, snapshot, atoms, np.zeros_like(active_field), rng
        ),
        "continuum-active-nonzero-field": _state_record(
            electronic, snapshot, atoms, active_field, rng
        ),
    }
    curl_failure = any(
        not bool(record["signs"]["1"]["coupled_reciprocity_passed"])
        for record in states.values()
    )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-coupling-active-space-conjugacy-terminal-audit",
        "status": (
            "terminal-coupled-curl-no-go"
            if curl_failure
            else "no-coupled-curl-witness-in-two-tested-states"
        ),
        "claim_boundary": (
            "A coupled-curl failure is a terminal counterexample to an original-"
            "source potential on the tested continuum-active space. Direct "
            "energy/source defects are recorded but not interpreted because the "
            "intrinsic hook is not yet verified as complete external enthalpy. "
            "No energy, force, variational, Hessian, or MD capability is admitted."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": radial.dtype,
        "native_semantics": canary.as_dict(),
        "continuum": {
            "provider_id": continuum.provider_id,
            "continuum_profile_id": continuum.continuum_profile_id,
            "cavity_profile_id": continuum.cavity_profile_id,
            "configuration_sha256": continuum.configuration_sha256(),
            "provenance_sha256": continuum.provenance_sha256,
            "snapshot_configuration_sha256": snapshot.configuration_sha256(),
            "snapshot_provenance_sha256": snapshot.provenance_sha256,
            "boundary_dimension": snapshot.boundary_dimension,
            "source_dimension": snapshot.source_dimension,
            "native_field_dimension": snapshot.receiver_dimension,
        },
        "protocol": {
            "random_seed": RANDOM_SEED,
            "field_scale": FIELD_SCALE,
            "states": list(states),
            "energy_gradient_signs": [-1, 1],
            "energy_semantics_verified": False,
            "boundary_metric": "harmonic surface operator A",
        },
        "states": states,
        "decision": {
            "coupled_curl_terminal_no_go": curl_failure,
            "direct_identity_terminal_decision": "blocked-unverified-energy-semantics",
            "original_checkpoint_tier_v_admitted": False,
            "operational_route_closed": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "native_semantics": payload["native_semantics"],
            "continuum": payload["continuum"],
            "protocol": payload["protocol"],
            "states": payload["states"],
            "decision": payload["decision"],
        }
    )
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_COUPLED_CONJUGACY_TERMINAL="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": payload["decision"],
                "capabilities": payload["capabilities"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
