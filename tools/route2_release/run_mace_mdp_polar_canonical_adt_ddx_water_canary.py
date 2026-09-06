#!/usr/bin/env python3
"""Run the target-free real-checkpoint canonical-ADT hybrid water canary.

The canary binds the unchanged official MACE-MDP and MACE-POLAR checkpoints to
the canonical atomic-density-translation response and the existing separated
ddX continuum.  It reads no experimental solvation target and admits no public
capability.  Its only purpose is to decide whether this zero-training response
profile is structurally sound enough to freeze before an independent accuracy
panel is opened.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.continuum import (
    SeparatedSourceDDXBackend,
    prepare_canonical_adt_separated_ddx,
)
from maple.solvation.experimental.mace_mdp_polar_adt_ddx import (
    CANONICAL_ADT_HYBRID_DDX_PROFILE_ID,
    MACE_MDPPolarCanonicalADTDDXEnergy,
    POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID,
)
from maple.solvation.models import (
    MACE_MDPPermanentSourceAdapter,
    MACEPolarOriginalSourceNativeFieldAdapter,
    MACEPolarZeroFieldPointPermanentSource,
    build_mace_mdp_moment_adapter,
    build_mdp_polar_canonical_adt_response,
    build_mdp_polar_role_separated_adt_response,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    canonical_json_sha256,
    checkpoint_record,
    runtime_record,
    symmetry_panel_rotations,
)

SCHEMA_VERSION = "route2-mace-mdp-polar-canonical-adt-ddx-water-canary-v2"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACEPOLAR1Mmodel"
RANDOM_SEED = 20260818

JVP_VJP_RELATIVE_TOLERANCE = 1.0e-9
FINITE_FIELD_FINEST_MAX_ERROR = 2.0e-7
ROTATION_ENERGY_ATOL_EV = 1.0e-5
ROTATION_SOURCE_RELATIVE_TOLERANCE = 3.0e-5
ROTATION_FIELD_RELATIVE_TOLERANCE = 3.0e-4
PASSIVITY_EIGENVALUE_FLOOR = 1.0e-8
DDX_LMAX = 8
DDX_N_LEBEDEV = 1202
GRID_CONVERGENCE_N_LEBEDEV = (194, 302, 434, 590, 770, 974, 1202)
LMAX_CONVERGENCE = (4, 6, 8, 10, 12, 15)
LMAX_ENERGY_ATOL_EV = 4.34e-4  # 0.0100 kcal/mol, target independent.
LMAX_FIELD_RELATIVE_TOLERANCE = 1.0e-3

NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
SOURCE_FILES = (
    "maple/solvation/continuum/separated_source_adt_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/atomic_displacement_lift.py",
    "maple/solvation/coupling/adt_radial_shape.py",
    "maple/solvation/experimental/mace_mdp_polar_adt_ddx.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_adt.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar_separated.py",
    "tools/route2_release/run_mace_mdp_polar_canonical_adt_ddx_water_canary.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _checkpoint_record(path: Path, *, role: str) -> dict[str, object]:
    record = checkpoint_record(path)
    record["role"] = role
    return record


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399872, 0.927297, 0.0],
            ],
            dtype=float,
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def _configure_torch() -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("torch")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def _relative_error(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(
            np.linalg.norm(left_array),
            np.linalg.norm(right_array),
            np.finfo(float).tiny,
        )
    )


def _rotate_raw_l1(values: object, rotation: object) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    transform = np.asarray(rotation, dtype=np.float64)
    result = np.array(source, copy=True)
    result[:, (3, 1, 2)] = source[:, (3, 1, 2)] @ transform.T
    return result


def _rotate_native_field(values: object, rotation: object) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    transform = np.asarray(rotation, dtype=np.float64)
    result = np.array(field, copy=True)
    for channels in ((4, 2, 3), (7, 5, 6)):
        result[:, channels] = field[:, channels] @ transform.T
    return result


def _build_evaluator(
    atoms: Atoms,
    *,
    permanent: object,
    response: object,
    lmax: int = DDX_LMAX,
) -> MACE_MDPPolarCanonicalADTDDXEnergy:
    continuum = SeparatedSourceDDXBackend(
        tuple(atoms.get_chemical_symbols()),
        smd_water_coulomb_radii(atoms.get_chemical_symbols()),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=lmax,
        n_lebedev=DDX_N_LEBEDEV,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
    )
    return MACE_MDPPolarCanonicalADTDDXEnergy(
        atoms,
        permanent=permanent,
        response=response,
        continuum=continuum,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    parser.add_argument(
        "--polar-device",
        choices=("cpu", "cuda"),
        default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--permanent-source-profile",
        choices=("mdp-latent-point", "polar-zero-point-mdp-alpha"),
        default="mdp-latent-point",
        help=(
            "Select the separately identified permanent-source hypothesis. "
            "The polar-zero option keeps MDP only as the molecular-alpha provider."
        ),
    )
    parser.add_argument(
        "--ddx-lmax",
        type=int,
        choices=LMAX_CONVERGENCE,
        default=DDX_LMAX,
        help="Selected ddX harmonic order; audited against the frozen lmax scan.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    torch = _configure_torch()
    repo = Path(__file__).resolve().parents[2]
    atoms = _water()
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    if args.polar_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The preregistered CUDA MACE-POLAR path is unavailable.")

    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.polar_device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    base_response = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    if args.permanent_source_profile == "mdp-latent-point":
        response = build_mdp_polar_canonical_adt_response(mdp=mdp, base=base_response)
        permanent = MACE_MDPPermanentSourceAdapter(mdp)
        expected_profile_id = CANONICAL_ADT_HYBRID_DDX_PROFILE_ID
        permanent_source_description = "MACE-MDP latent point q/p"
    else:
        response = build_mdp_polar_role_separated_adt_response(
            mdp=mdp,
            base=base_response,
            source_root=repo,
        )
        permanent = MACEPolarZeroFieldPointPermanentSource(response)
        expected_profile_id = POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
        permanent_source_description = (
            "MACE-POLAR zero-field point q/p; MDP supplies molecular alpha only"
        )
    evaluator = _build_evaluator(
        atoms,
        permanent=permanent,
        response=response,
        lmax=args.ddx_lmax,
    )
    if evaluator.profile_id != expected_profile_id:
        raise RuntimeError("water canary selected an unexpected profile identity.")

    state = evaluator.solve(atoms)
    replay = evaluator.solve(atoms)
    certificate = evaluator.local_root_certificate(atoms, state)
    passivity = evaluator.molecular_response_passivity_eigenvalues(atoms)

    lmax_states = {args.ddx_lmax: state}
    for lmax in LMAX_CONVERGENCE:
        if lmax == args.ddx_lmax:
            continue
        lmax_states[lmax] = _build_evaluator(
            atoms,
            permanent=permanent,
            response=response,
            lmax=lmax,
        ).solve(atoms)
    lmax_reference = lmax_states[LMAX_CONVERGENCE[-1]]
    lmax_convergence = []
    for lmax in LMAX_CONVERGENCE:
        current = lmax_states[lmax]
        lmax_convergence.append(
            {
                "lmax": lmax,
                "polarization_energy_eV": current.polarization_energy_ev,
                "energy_absolute_vs_lmax15_eV": abs(
                    current.polarization_energy_ev
                    - lmax_reference.polarization_energy_ev
                ),
                "native_field_relative_vs_lmax15": _relative_error(
                    current.native_field8,
                    lmax_reference.native_field8,
                ),
                "maximum_root_iterations": max(
                    item.iterations for item in current.root_starts
                ),
                "primal_residual_eV": current.primal_residual_ev,
            }
        )
    selected_lmax = next(
        item for item in lmax_convergence if item["lmax"] == args.ddx_lmax
    )

    rng = np.random.default_rng(RANDOM_SEED)
    direction = rng.normal(size=state.native_field8.shape)
    cotangent = rng.normal(size=state.native_field8.shape)
    jvp = evaluator.state_map_jvp(atoms, state.native_field8, direction)
    vjp = evaluator.state_map_vjp(atoms, state.native_field8, cotangent)
    dot_left = float(np.vdot(cotangent, jvp))
    dot_right = float(np.vdot(vjp, direction))
    dot_absolute = abs(dot_left - dot_right)
    dot_relative = dot_absolute / max(
        abs(dot_left), abs(dot_right), np.finfo(float).tiny
    )

    finite_field_steps = (2.0e-3, 1.0e-3, 5.0e-4)
    finite_field_errors = []
    for step in finite_field_steps:
        plus = evaluator.state_map(atoms, state.native_field8 + step * direction)[0]
        minus = evaluator.state_map(atoms, state.native_field8 - step * direction)[0]
        finite_field_errors.append(
            float(np.max(np.abs((plus - minus) / (2.0 * step) - jvp)))
        )

    rotation = symmetry_panel_rotations("canonical-adt-water-v1")[0]
    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T + np.asarray([0.71, -0.43, 0.29])
    moved.info.update(atoms.info)
    moved_evaluator = _build_evaluator(
        moved,
        permanent=permanent,
        response=response,
        lmax=args.ddx_lmax,
    )
    moved_state = moved_evaluator.solve(moved)
    rotation_errors = {
        "energy_absolute_eV": abs(moved_state.total_energy_ev - state.total_energy_ev),
        "polarization_energy_absolute_eV": abs(
            moved_state.polarization_energy_ev - state.polarization_energy_ev
        ),
        "permanent_source_relative": _relative_error(
            moved_state.permanent_source4,
            _rotate_raw_l1(state.permanent_source4, rotation),
        ),
        "radial_residual_source_relative": _relative_error(
            moved_state.radial_residual_source4,
            _rotate_raw_l1(state.radial_residual_source4, rotation),
        ),
        "adt_atomic_dipoles_relative": _relative_error(
            moved_state.adt_atomic_dipoles_eangstrom,
            state.adt_atomic_dipoles_eangstrom @ rotation.T,
        ),
        "native_field_relative": _relative_error(
            moved_state.native_field8,
            _rotate_native_field(state.native_field8, rotation),
        ),
    }

    grid_convergence = []
    rotated_permanent = _rotate_raw_l1(state.permanent_source4, rotation)
    rotated_radial = _rotate_raw_l1(state.radial_residual_source4, rotation)
    rotated_adt = state.adt_atomic_dipoles_eangstrom @ rotation.T
    for n_lebedev in GRID_CONVERGENCE_N_LEBEDEV:
        base_backend = SeparatedSourceDDXBackend(
            tuple(atoms.get_chemical_symbols()),
            smd_water_coulomb_radii(atoms.get_chemical_symbols()),
            continuum_model="pcm",
            dielectric=78.39,
            lmax=args.ddx_lmax,
            n_lebedev=n_lebedev,
            solver_tolerance=1.0e-12,
            eta=0.1,
            n_proc=1,
        )
        moved_backend = SeparatedSourceDDXBackend(
            tuple(moved.get_chemical_symbols()),
            smd_water_coulomb_radii(moved.get_chemical_symbols()),
            continuum_model="pcm",
            dielectric=78.39,
            lmax=args.ddx_lmax,
            n_lebedev=n_lebedev,
            solver_tolerance=1.0e-12,
            eta=0.1,
            n_proc=1,
        )
        base_direct_sum = prepare_canonical_adt_separated_ddx(
            base_backend.prepare(atoms, state.permanent_source4),
            evaluator.direct_sum.adt_lift,
        )
        moved_direct_sum = prepare_canonical_adt_separated_ddx(
            moved_backend.prepare(moved, rotated_permanent),
            moved_evaluator.direct_sum.adt_lift,
        )
        base_fixed = base_direct_sum.solve(
            state.radial_residual_source4,
            state.adt_atomic_dipoles_eangstrom,
        )
        moved_fixed = moved_direct_sum.solve(rotated_radial, rotated_adt)
        grid_convergence.append(
            {
                "n_lebedev": n_lebedev,
                "fixed_source_energy_absolute_eV": abs(
                    moved_fixed.polarization_energy_ev
                    - base_fixed.polarization_energy_ev
                ),
                "fixed_source_field_relative": _relative_error(
                    moved_fixed.model_field8,
                    _rotate_native_field(base_fixed.model_field8, rotation),
                ),
            }
        )

    gates = {
        "five_start_root_residual": max(
            item.residual_norm_ev for item in state.root_starts
        )
        < 1.0e-10,
        "five_start_same_energy": (
            max(item.polarization_energy_ev for item in state.root_starts)
            - min(item.polarization_energy_ev for item in state.root_starts)
        )
        < 2.0e-10,
        "exact_cold_replay": replay.root_sha256 == state.root_sha256,
        "jvp_vjp_dot": dot_relative <= JVP_VJP_RELATIVE_TOLERANCE,
        "finite_field_second_order": (
            finite_field_errors[0] > finite_field_errors[1] > finite_field_errors[2]
            and finite_field_errors[2] <= FINITE_FIELD_FINEST_MAX_ERROR
        ),
        "uniform_response_passive": (
            float(np.min(passivity)) >= PASSIVITY_EIGENVALUE_FLOOR
        ),
        "local_root_nonsingular": certificate.local_implicit_branch_certified,
        "local_contraction": certificate.contraction_at_state,
        "local_strong_monotonicity": certificate.strong_monotonicity_at_state,
        "rigid_energy_covariance": (
            rotation_errors["energy_absolute_eV"] <= ROTATION_ENERGY_ATOL_EV
        ),
        "rigid_source_covariance": max(
            rotation_errors["permanent_source_relative"],
            rotation_errors["radial_residual_source_relative"],
            rotation_errors["adt_atomic_dipoles_relative"],
        )
        <= ROTATION_SOURCE_RELATIVE_TOLERANCE,
        "rigid_field_covariance": (
            rotation_errors["native_field_relative"]
            <= ROTATION_FIELD_RELATIVE_TOLERANCE
        ),
        "target_free_grid_convergence": (
            grid_convergence[-1]["fixed_source_energy_absolute_eV"]
            <= ROTATION_ENERGY_ATOL_EV
            and grid_convergence[-1]["fixed_source_field_relative"]
            <= ROTATION_FIELD_RELATIVE_TOLERANCE
            and grid_convergence[-1]["fixed_source_energy_absolute_eV"]
            < grid_convergence[0]["fixed_source_energy_absolute_eV"] / 10.0
            and grid_convergence[-1]["fixed_source_field_relative"]
            < grid_convergence[0]["fixed_source_field_relative"] / 5.0
        ),
        "target_free_lmax_convergence": (
            selected_lmax["energy_absolute_vs_lmax15_eV"] <= LMAX_ENERGY_ATOL_EV
            and selected_lmax["native_field_relative_vs_lmax15"]
            <= LMAX_FIELD_RELATIVE_TOLERANCE
        ),
        "no_global_uniqueness_overclaim": (
            not certificate.global_unique_root_certified
            and not certificate.domain_uniform_bound_certified
        ),
    }
    status = "pass-target-free-prequalification" if all(gates.values()) else "fail"

    source_hashes = {
        name: _sha256((repo / name).resolve(strict=True)) for name in SOURCE_FILES
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "target-free-real-checkpoint-canonical-adt-hybrid-canary",
        "status": status,
        "claim_boundary": (
            "This water canary reads no experimental solvation target. It tests "
            "the exact frozen response/root/covariance implementation only. A pass "
            "permits freezing the candidate before an independent accuracy panel; "
            "it admits no E/F/H/V/M, global root uniqueness, force, or Tier V claim."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution": {
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_head_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
            "working_tree_status": _git(repo, "status", "--porcelain=v1"),
            "source_files_sha256": source_hashes,
        },
        "configuration": {
            "profile_id": expected_profile_id,
            "evaluator_configuration_sha256": evaluator.configuration_sha256(),
            "moved_evaluator_configuration_sha256": (
                moved_evaluator.configuration_sha256()
            ),
            "response_configuration_sha256": response.configuration_sha256(),
            "continuum_configuration_sha256": (
                evaluator.direct_sum.configuration_sha256()
            ),
            "mdp_checkpoint": _checkpoint_record(
                mdp_checkpoint,
                role="mace-mdp-official-checkpoint",
            ),
            "polar_checkpoint": _checkpoint_record(
                polar_checkpoint,
                role="mace-polar-1-m-official-checkpoint",
            ),
            "mdp_device": "cpu",
            "polar_device": args.polar_device,
            "permanent_source_profile": args.permanent_source_profile,
            "permanent_source": permanent_source_description,
            "radial_response": "MACE-POLAR nonuniform residual 1.5-A Gaussian q/p",
            "uniform_response": (
                "canonical free-atom-density ADT lift of public MACE-MDP molecular alpha"
            ),
            "continuum": "shared pyddx ddPCM coefficient state",
            "dielectric": 78.39,
            "lmax": args.ddx_lmax,
            "lmax_selection": {
                "basis": (
                    "target-free water convergence against lmax=15 before the "
                    "canonical-ADT accuracy panel"
                ),
                "energy_tolerance_eV": LMAX_ENERGY_ATOL_EV,
                "field_relative_tolerance": LMAX_FIELD_RELATIVE_TOLERANCE,
            },
            "n_lebedev": DDX_N_LEBEDEV,
            "n_lebedev_selection": (
                "target-free rigid-covariance convergence; no solvation target read"
            ),
        },
        "root": {
            "root_sha256": state.root_sha256,
            "primal_residual_eV": state.primal_residual_ev,
            "polarization_energy_eV": state.polarization_energy_ev,
            "total_energy_eV": state.total_energy_ev,
            "starts": [
                {
                    "label": item.label,
                    "iterations": item.iterations,
                    "residual_norm_eV": item.residual_norm_ev,
                    "polarization_energy_eV": item.polarization_energy_ev,
                    "record_sha256": item.record_sha256,
                }
                for item in state.root_starts
            ],
            "cold_replay_root_sha256": replay.root_sha256,
        },
        "linearization": {
            "jvp_vjp_dot_left": dot_left,
            "jvp_vjp_dot_right": dot_right,
            "jvp_vjp_absolute_defect": dot_absolute,
            "jvp_vjp_relative_defect": dot_relative,
            "finite_field_steps": list(finite_field_steps),
            "finite_field_maximum_errors": finite_field_errors,
        },
        "passivity": {
            "molecular_alpha_eigenvalues": passivity.tolist(),
            "claim_boundary": (
                "This certifies only the selected three-dimensional uniform ADT "
                "response, not global passivity of every nonlinear field direction."
            ),
        },
        "local_root_certificate": certificate.as_dict(),
        "rigid_covariance": {
            "rotation_matrix": rotation.tolist(),
            "translation_angstrom": [0.71, -0.43, 0.29],
            **rotation_errors,
            "fixed_source_grid_convergence": grid_convergence,
            "claim_boundary": (
                "The finite ddX laboratory quadrature is tested numerically and is "
                "not relabelled as a structural SO(3) proof."
            ),
        },
        "lmax_convergence": lmax_convergence,
        "gates": gates,
        "decision": {
            "target_free_prequalification_passed": all(gates.values()),
            "accuracy_panel_permitted_next": all(gates.values()),
            "accuracy_claim_made": False,
            "force_admitted": False,
            "tier_v_admitted": False,
            "global_unique_root_certified": False,
        },
        "runtime": {
            **runtime_record(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "packages": {
                "ase": _version("ase"),
                "mace-torch": _version("mace-torch"),
                "numpy": _version("numpy"),
                "pyddx": _version("pyddx"),
                "scipy": _version("scipy"),
                "torch": _version("torch"),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "configuration": payload["configuration"],
            "root": payload["root"],
            "linearization": payload["linearization"],
            "passivity": payload["passivity"],
            "local_root_certificate": payload["local_root_certificate"],
            "rigid_covariance": payload["rigid_covariance"],
            "lmax_convergence": payload["lmax_convergence"],
            "gates": payload["gates"],
            "decision": payload["decision"],
        }
    )
    payload["artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    args = _parse_args()
    payload = run(args)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite canary artifact: {output}") from exc
    print(
        "ROUTE2_CANONICAL_ADT_DDX_WATER_CANARY="
        + json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload["status"],
                "measurement_sha256": payload["measurement_sha256"],
                "artifact_sha256": payload["artifact_sha256"],
                "gates": payload["gates"],
            },
            sort_keys=True,
        )
    )
    if not all(payload["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
