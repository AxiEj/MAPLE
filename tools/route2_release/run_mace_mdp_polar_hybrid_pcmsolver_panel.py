#!/usr/bin/env python3
"""Run the preregistered hybrid energy/numerical-force admission panel.

The source seen by PCMSolver is deliberately heterogeneous:

* the MACE-MDP zero-field permanent ``(q,p)`` source is evaluated as exterior
  point multipoles;
* only ``MACE-POLAR(R,u) - MACE-POLAR(R,0)`` is evaluated with the checkpoint's
  1.5-A Gaussian source basis;
* the PCMSolver surface charge is returned to MACE-POLAR through its audited
  two-width 1.5/3.0-A receiver.

The force is the fourth-order Richardson gradient of that same scalar.  Every
stencil point rebuilds PCMSolver and resolves the two-start root.  Accuracy
values are retained as context but are not an admission threshold.  Two
independent clean processes must reproduce the same measurement digest before
public experimental E/F can be enabled.  Analytic derivatives, complete
solvation, solvent transfer, Hessians, MD, and Tier V remain excluded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

from ase.units import Bohr
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (
    PCMSolverSession,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1,
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1,
)
from maple.solvation.experimental.mace_mdp_polar_pcmsolver import (
    MACE_MDPPolarHybridPCMSolverEnergy,
    MACE_MDPPolarHybridPCMSolverPES,
    NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
    NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
)
from maple.solvation.models import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    MACE_MDP_POLAR_HYBRID_PROFILE_ID,
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-mdp-polar-hybrid-energy-force-admission-v1"
EXPECTED_MACE_POLAR_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
PARENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
ADMISSION_PREREGISTRATION_RELATIVE_PATH = (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-numerical-force-admission-v1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
KCAL_PER_HARTREE = 627.5094740631
ROOT_TOLERANCE_EV = 1.0e-10
MAX_ROOT_ITERATIONS = 40
ROOT_REPLAY_ATOL_EV = 2.0e-9
TOTAL_CHARGE_ATOL_E = 1.0e-8
GEOMETRY_ATOL_ANGSTROM = 1.0e-8
SURFACE_REPLAY_ATOL_BOHR = 1.0e-12
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/solvation/coupling/exact_gto.py",
    "maple/solvation/experimental/mace_mdp_polar_pcmsolver.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/evidence.py",
    "tools/route2_release/run_mace_mdp_polar_hybrid_pcmsolver_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument(
        "--mace-mdp-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-MDP.model",
    )
    parser.add_argument(
        "--mace-polar-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACEPOLAR1Mmodel",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _rebased_asset_path(asset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Frozen asset path must be a non-empty string.")
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    try:
        marker = path.parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"Cannot rebase frozen asset path {raw_path!r}.") from exc
    rebased = asset_root.joinpath(*path.parts[marker:]).resolve()
    if not rebased.is_file():
        raise FileNotFoundError(rebased)
    return rebased


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260815)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _molecular_dipole(source4: np.ndarray, positions: np.ndarray) -> np.ndarray:
    # Raw real-l=1 order [m=0,m=1,m=-1] maps to Cartesian [y,z,x].
    return np.sum(source4[:, 0, None] * positions + source4[:, (3, 1, 2)], axis=0)


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _record_case(
    *,
    record: dict[str, object],
    asset_root: Path,
    hybrid: object,
    pcmsolver_library: Path,
    force_panel: dict[str, object],
) -> dict[str, object]:
    compound_id = str(record["compound_id"])
    panel_root = asset_root / PANEL_RELATIVE_ROOT / compound_id
    result_path = panel_root / CUTOFF_DIRECTORY / "result.json"
    work = panel_root / CUTOFF_DIRECTORY / "work"
    result = _load_json(result_path, name=f"{compound_id} projection result")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != compound_id:
        raise RuntimeError(f"{compound_id} result input identity is invalid.")

    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name=f"{compound_id} MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    qm_path = work / "qm-surface-mep.npz"
    surface_path = work / "surface.npz"
    with np.load(qm_path) as qm:
        qm_potential = np.asarray(qm["surface_potential_hartree_per_e"], dtype=float)
        qm_positions = np.asarray(qm["atom_positions_angstrom"], dtype=float)
        qm_numbers = np.asarray(qm["atomic_numbers"], dtype=int)
        qm_dipole = np.asarray(qm["molecular_dipole_e_angstrom"], dtype=float)
        qm_total_charge = float(qm["total_charge_e"])
    if not np.array_equal(qm_numbers, atoms.numbers):
        raise RuntimeError(f"{compound_id} QM and MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(qm_positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} QM/MOL2 geometry mismatch.")

    pcm_record = inputs.get("parsed_pcm_input")
    if not isinstance(pcm_record, dict):
        raise RuntimeError(f"{compound_id} result omits its PCM input record.")
    pcm_input = _rebased_asset_path(asset_root, pcm_record.get("path"))
    _validated_sha(pcm_input, pcm_record.get("sha256"), name=f"{compound_id} PCM input")
    radii = np.asarray(pcm_record.get("cavity_radii_angstrom"), dtype=float)
    if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
        raise RuntimeError(f"{compound_id} PCM radii are invalid.")

    anchor = hybrid.prepare(atoms)
    frozen_surface = np.asarray(
        np.load(surface_path)["surface_points_bohr"], dtype=float
    )
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"route2-hybrid-{compound_id}-") as tmp:
        try:
            os.chdir(tmp)
            with PCMSolverSession(
                np.asarray(atoms.numbers, dtype=float),
                atoms.positions / Bohr,
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                points = np.asarray(session.cavity_centers_bohr, dtype=float)
                surface_error = float(np.max(np.abs(points - frozen_surface)))
                if surface_error > SURFACE_REPLAY_ATOL_BOHR:
                    raise RuntimeError(f"{compound_id} PCMSolver surface drifted.")
                response = PCMSolverExternalMEPCavityResponse(
                    session, cavity_radii_angstrom=radii
                )
                induced_operator = AtomCenteredL1GTOBasis((1.5,)).surface_operator(
                    points, atoms.positions
                )
                permanent_potential = point_multipole_potential(
                    points, atoms.positions, anchor.permanent_source4
                )
                polar_zero_potential = induced_operator @ (
                    anchor.response_zero_source4.reshape(-1)
                )

                def surface_energy(potential: np.ndarray) -> tuple[np.ndarray, float]:
                    charge = np.asarray(
                        response.apply_energy_conjugate(potential), dtype=float
                    )
                    return charge, 0.5 * float(np.vdot(potential, charge))

                qm_charge, qm_energy = surface_energy(qm_potential)
                mdp_charge, mdp_energy = surface_energy(permanent_potential)
                _polar_charge, polar_energy = surface_energy(polar_zero_potential)
                energy_evaluator = MACE_MDPPolarHybridPCMSolverEnergy(
                    atoms,
                    hybrid=hybrid,
                    response=response,
                )
                energy_state = energy_evaluator.solve(atoms)
                areas = np.asarray(session.cavity_areas_bohr2, dtype=float)
        finally:
            os.chdir(previous)

    pes = MACE_MDPPolarHybridPCMSolverPES(
        hybrid=hybrid,
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=radii,
        parsed_input_path=pcm_input,
        pcmsolver_library_path=pcmsolver_library,
    )
    full_force_compound_id = str(force_panel["full_cartesian_compound_id"])
    if compound_id == full_force_compound_id:
        force_evaluation = pes.numerical_force(
            atoms,
            central_state=energy_state,
        )
        force_values = np.asarray(force_evaluation.forces_eV_per_A, dtype=float)
        force_errors = np.asarray(
            force_evaluation.error_estimates_eV_per_A, dtype=float
        )
        force_record: dict[str, object] = {
            "scope": "full-cartesian",
            "forces_eV_per_A": force_values.tolist(),
            "error_estimates_eV_per_A": force_errors.tolist(),
            "maximum_error_estimate_eV_per_A": (
                force_evaluation.maximum_error_estimate_eV_per_A
            ),
            "evaluation_sha256": force_evaluation.evaluation_sha256,
            "net_force_norm_eV_per_A": float(
                np.linalg.norm(np.sum(force_values, axis=0))
            ),
            "finite": bool(
                np.all(np.isfinite(force_values)) and np.all(np.isfinite(force_errors))
            ),
        }

        direction_seed = int(force_panel["independent_direction_seed"])
        direction_step = float(
            force_panel["independent_directional_check_step_angstrom"]
        )
        direction = np.random.default_rng(direction_seed).normal(
            size=atoms.positions.shape
        )
        direction /= np.linalg.norm(direction)
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions = np.asarray(atoms.positions) + direction_step * direction
        minus.positions = np.asarray(atoms.positions) - direction_step * direction
        plus_sample = pes.sample(plus)
        minus_sample = pes.sample(minus)
        central_topology = energy_state.cavity_topology_id
        if (
            plus_sample.topology_id != central_topology
            or minus_sample.topology_id != central_topology
        ):
            raise RuntimeError(
                f"{compound_id} independent directional stencil changed topology."
            )
        directional_energy_derivative = (
            plus_sample.energy_eV - minus_sample.energy_eV
        ) / (2.0 * direction_step)
        directional_force_projection = -float(np.vdot(force_values, direction))
        directional_error = abs(
            directional_energy_derivative - directional_force_projection
        )
        force_record["independent_directional_check"] = {
            "seed": direction_seed,
            "step_angstrom": direction_step,
            "energy_derivative_eV_per_A": directional_energy_derivative,
            "negative_force_projection_eV_per_A": directional_force_projection,
            "absolute_error_eV_per_A": directional_error,
            "plus_state_sha256": plus_sample.state_sha256,
            "minus_state_sha256": minus_sample.state_sha256,
        }
    else:
        sampled_dof = force_panel["sampled_cartesian_dof_for_other_records"]
        if (
            not isinstance(sampled_dof, list)
            or len(sampled_dof) != 2
            or any(type(value) is not int for value in sampled_dof)
        ):
            raise RuntimeError("Preregistered sampled Cartesian DOF is invalid.")
        component = pes.numerical_force_component(
            atoms,
            atom_index=sampled_dof[0],
            axis_index=sampled_dof[1],
            central_state=energy_state,
        )
        force_record = {
            "scope": "sampled-cartesian-component",
            "atom_index": component.atom_index,
            "axis_index": component.axis_index,
            "force_eV_per_A": component.force_eV_per_A,
            "error_estimate_eV_per_A": component.error_estimate_eV_per_A,
            "displaced_state_sha256": list(component.displaced_state_sha256),
            "finite": bool(
                np.isfinite(component.force_eV_per_A)
                and np.isfinite(component.error_estimate_eV_per_A)
            ),
        }

    field_replay_error = energy_state.replay_field_max_abs_difference_ev
    energy_replay_error = energy_state.replay_energy_abs_difference_ev / 27.211386245988
    if field_replay_error > ROOT_REPLAY_ATOL_EV:
        raise RuntimeError(f"{compound_id} hybrid multi-start roots disagree.")
    hybrid_potential = np.asarray(
        energy_state.surface_potential_hartree_per_e, dtype=float
    )
    potential_error = hybrid_potential - qm_potential
    area_relative_mep = float(
        np.sqrt(
            np.sum(areas * potential_error * potential_error)
            / np.sum(areas * qm_potential * qm_potential)
        )
    )
    total_source = np.asarray(energy_state.total_source4, dtype=float)
    hybrid_dipole = _molecular_dipole(total_source, atoms.positions)
    hybrid_energy = energy_state.polarization_energy_ev / 27.211386245988
    total_charge = float(np.sum(total_source[:, 0]))
    target_from_prior = float(
        result["basis_results"]["one_radial"]["target_polarization_energy_hartree"]
    )
    if abs(qm_energy - target_from_prior) > 2.0e-13:
        raise RuntimeError(f"{compound_id} target polarization energy drifted.")
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "class": str(record["class"]),
        "atom_count": len(atoms),
        "geometry_max_abs_error_angstrom": geometry_error,
        "surface_replay_max_abs_error_bohr": surface_error,
        "qm_total_charge_e": qm_total_charge,
        "hybrid_total_charge_e": total_charge,
        "total_charge_absolute_error_e": abs(total_charge - qm_total_charge),
        "qm_dipole_e_angstrom": qm_dipole.tolist(),
        "hybrid_dipole_e_angstrom": hybrid_dipole.tolist(),
        "hybrid_dipole_relative_l2_error": _relative_l2(hybrid_dipole, qm_dipole),
        "hybrid_surface_mep_area_weighted_relative_l2_error": area_relative_mep,
        "qm_polarization_energy_kcal_per_mol": qm_energy * KCAL_PER_HARTREE,
        "mace_mdp_fixed_polarization_energy_kcal_per_mol": (
            mdp_energy * KCAL_PER_HARTREE
        ),
        "mace_polar_fixed_polarization_energy_kcal_per_mol": (
            polar_energy * KCAL_PER_HARTREE
        ),
        "hybrid_polarization_energy_kcal_per_mol": (hybrid_energy * KCAL_PER_HARTREE),
        "mace_mdp_fixed_absolute_error_kcal_per_mol": (
            abs(mdp_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "mace_polar_fixed_absolute_error_kcal_per_mol": (
            abs(polar_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "hybrid_absolute_error_kcal_per_mol": (
            abs(hybrid_energy - qm_energy) * KCAL_PER_HARTREE
        ),
        "hybrid_improvement_vs_mdp_kcal_per_mol": (
            (abs(mdp_energy - qm_energy) - abs(hybrid_energy - qm_energy))
            * KCAL_PER_HARTREE
        ),
        "macepolar_vacuum_energy_ev": energy_state.vacuum_energy_ev,
        "experimental_total_energy_ev": energy_state.total_energy_ev,
        "energy_root_sha256": energy_state.root_sha256,
        "energy_evaluator_configuration_sha256": (
            energy_state.evaluator_configuration_sha256
        ),
        "half_coupling_gate_passed": True,
        "finite_scalar_leaves_gate_passed": bool(
            np.isfinite(energy_state.vacuum_energy_ev)
            and np.isfinite(energy_state.polarization_energy_ev)
            and np.isfinite(energy_state.total_energy_ev)
        ),
        "cavity_topology_id": energy_state.cavity_topology_id,
        "force": force_record,
        "induced_source_l2_norm": float(np.linalg.norm(energy_state.induced_source4)),
        "cold_root_iterations": energy_state.cold_iterations,
        "wide_root_iterations": energy_state.wide_iterations,
        "cold_final_residual_ev": energy_state.primal_residual_ev,
        "wide_final_residual_ev": energy_state.primal_residual_ev,
        "multi_start_field_max_abs_difference_ev": field_replay_error,
        "multi_start_energy_abs_difference_hartree": energy_replay_error,
        "charge_gate_passed": bool(
            abs(total_charge - qm_total_charge) <= TOTAL_CHARGE_ATOL_E
        ),
        "root_gate_passed": bool(
            energy_state.primal_residual_ev < ROOT_TOLERANCE_EV
            and field_replay_error <= ROOT_REPLAY_ATOL_EV
        ),
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
            "qm_surface_mep": sha256_file(qm_path),
            "surface_points": sha256_file(surface_path),
            "pcm_input": sha256_file(pcm_input),
        },
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver_library = args.pcmsolver_library.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mace_mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.mace_polar_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(mdp_checkpoint) != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-MDP checkpoint does not match the frozen model.")
    if sha256_file(polar_checkpoint) != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-POLAR checkpoint does not match the frozen model.")

    admission_preregistration_path = (
        repository.root / ADMISSION_PREREGISTRATION_RELATIVE_PATH
    )
    admission_preregistration = _load_json(
        admission_preregistration_path,
        name="hybrid experimental-energy admission preregistration",
    )
    if (
        admission_preregistration.get("target_profile_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
        or admission_preregistration.get("target_scalar_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1
        or admission_preregistration.get("status") != "preregistered-before-execution"
    ):
        raise RuntimeError("Hybrid energy/force admission preregistration is invalid.")
    target_capabilities = admission_preregistration.get("target_capabilities")
    if target_capabilities != {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }:
        raise RuntimeError("Hybrid admission target capabilities drifted.")
    frozen_runtime = admission_preregistration.get("frozen_runtime_contract")
    force_panel = admission_preregistration.get("force_panel")
    if not isinstance(frozen_runtime, dict) or not isinstance(force_panel, dict):
        raise RuntimeError("Hybrid admission omits its force contract.")
    if (
        float(frozen_runtime.get("force_coarse_step_angstrom", np.nan))
        != NUMERICAL_FORCE_COARSE_STEP_ANGSTROM
        or float(
            frozen_runtime.get(
                "force_maximum_local_error_estimate_ev_per_angstrom", np.nan
            )
        )
        != NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
    ):
        raise RuntimeError(
            "Hybrid numerical-force implementation drifted from preregistration."
        )
    parent_record = admission_preregistration.get("parent_panel")
    if not isinstance(parent_record, dict):
        raise RuntimeError("Hybrid admission preregistration omits its parent panel.")
    preregistration_path = repository.root / PARENT_PREREGISTRATION_RELATIVE_PATH
    if str(parent_record.get("relative_path")) != PARENT_PREREGISTRATION_RELATIVE_PATH:
        raise RuntimeError("Hybrid admission parent-panel path drifted.")
    preregistration = _load_json(
        preregistration_path, name="parent PCM preregistration"
    )
    _validated_sha(
        preregistration_path,
        parent_record.get("sha256"),
        name="parent PCM preregistration",
    )
    execution_contract = preregistration.get("execution_contract")
    records = preregistration.get("records")
    if not isinstance(execution_contract, dict) or not isinstance(records, list):
        raise RuntimeError("Parent preregistration omits its panel contract.")
    _validated_sha(
        pcmsolver_library,
        execution_contract.get("pcmsolver_library_sha256"),
        name="PCMSolver shared library",
    )
    if len(records) != 4 or not all(isinstance(record, dict) for record in records):
        raise RuntimeError("Expected the frozen four-record PCM panel.")

    import torch

    _configure_determinism(torch)
    mdp = build_mace_mdp_moment_adapter(checkpoint_path=mdp_checkpoint, device="cpu")
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    case_records = [
        _record_case(
            record=record,
            asset_root=asset_root,
            hybrid=hybrid,
            pcmsolver_library=pcmsolver_library,
            force_panel=force_panel,
        )
        for record in sorted(records, key=lambda value: str(value["compound_id"]))
    ]
    mdp_errors = [
        float(record["mace_mdp_fixed_absolute_error_kcal_per_mol"])
        for record in case_records
    ]
    polar_errors = [
        float(record["mace_polar_fixed_absolute_error_kcal_per_mol"])
        for record in case_records
    ]
    hybrid_errors = [
        float(record["hybrid_absolute_error_kcal_per_mol"]) for record in case_records
    ]
    improved = [
        float(record["hybrid_improvement_vs_mdp_kcal_per_mol"]) > 0.0
        for record in case_records
    ]
    force_records = [record["force"] for record in case_records]
    if not all(isinstance(record, dict) for record in force_records):
        raise RuntimeError("Force panel records are invalid.")
    maximum_force_error = max(
        float(
            record.get(
                "maximum_error_estimate_eV_per_A",
                record.get("error_estimate_eV_per_A", np.inf),
            )
        )
        for record in force_records
    )
    full_force_records = [
        record for record in force_records if record.get("scope") == "full-cartesian"
    ]
    if len(full_force_records) != 1:
        raise RuntimeError(
            "Admission requires exactly one full Cartesian force record."
        )
    full_force_record = full_force_records[0]
    directional_record = full_force_record.get("independent_directional_check")
    if not isinstance(directional_record, dict):
        raise RuntimeError("Full force record omits its independent direction.")
    directional_error = float(directional_record["absolute_error_eV_per_A"])
    net_force_norm = float(full_force_record["net_force_norm_eV_per_A"])
    directional_limit = float(
        force_panel["maximum_directional_derivative_abs_error_ev_per_angstrom"]
    )
    net_force_limit = float(force_panel["maximum_net_force_norm_ev_per_angstrom"])
    aggregate = {
        "record_count": len(case_records),
        "root_and_charge_pass_count": sum(
            bool(record["root_gate_passed"] and record["charge_gate_passed"])
            for record in case_records
        ),
        "hybrid_improved_case_count": sum(improved),
        "hybrid_worsened_case_count": len(improved) - sum(improved),
        "mace_mdp_fixed_mean_absolute_error_kcal_per_mol": float(np.mean(mdp_errors)),
        "mace_mdp_fixed_maximum_absolute_error_kcal_per_mol": max(mdp_errors),
        "mace_polar_fixed_mean_absolute_error_kcal_per_mol": float(
            np.mean(polar_errors)
        ),
        "hybrid_mean_absolute_error_kcal_per_mol": float(np.mean(hybrid_errors)),
        "hybrid_maximum_absolute_error_kcal_per_mol": max(hybrid_errors),
        "hybrid_case_count_below_1_kcal_per_mol": sum(
            error < 1.0 for error in hybrid_errors
        ),
        "maximum_multi_start_field_difference_ev": max(
            float(record["multi_start_field_max_abs_difference_ev"])
            for record in case_records
        ),
        "maximum_final_root_residual_ev": max(
            float(record["cold_final_residual_ev"]) for record in case_records
        ),
        "maximum_total_charge_absolute_error_e": max(
            float(record["total_charge_absolute_error_e"]) for record in case_records
        ),
        "half_coupling_pass_count": sum(
            bool(record["half_coupling_gate_passed"]) for record in case_records
        ),
        "finite_scalar_leaves_pass_count": sum(
            bool(record["finite_scalar_leaves_gate_passed"]) for record in case_records
        ),
        "force_record_count": len(force_records),
        "finite_force_record_count": sum(
            bool(record["finite"]) for record in force_records
        ),
        "maximum_richardson_force_error_estimate_eV_per_A": (maximum_force_error),
        "acetone_independent_directional_derivative_absolute_error_eV_per_A": (
            directional_error
        ),
        "acetone_net_force_norm_eV_per_A": net_force_norm,
    }
    energy_force_execution_gate_passed = bool(
        aggregate["record_count"] == 4
        and aggregate["root_and_charge_pass_count"] == 4
        and aggregate["half_coupling_pass_count"] == 4
        and aggregate["finite_scalar_leaves_pass_count"] == 4
        and aggregate["force_record_count"] == 4
        and aggregate["finite_force_record_count"] == 4
        and maximum_force_error <= NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
        and directional_error <= directional_limit
        and net_force_norm <= net_force_limit
        and aggregate["maximum_multi_start_field_difference_ev"] <= ROOT_REPLAY_ATOL_EV
        and aggregate["maximum_final_root_residual_ev"] < ROOT_TOLERANCE_EV
        and aggregate["maximum_total_charge_absolute_error_e"] <= TOTAL_CHARGE_ATOL_E
    )
    measurement = {
        "protocol": {
            "profile_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
            ),
            "scalar_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_V1
            ),
            "model_profile_id": MACE_MDP_POLAR_HYBRID_PROFILE_ID,
            "permanent_source": "frozen MACE-MDP atom-centred point q/p",
            "induced_source": ("MACE-POLAR(R,u)-MACE-POLAR(R,0), 1.5-A Gaussian q/p"),
            "receiver": "MACE-POLAR physical radial 1.5/3.0-A field8",
            "continuum": preregistration["method"]["continuum"],
            "solvent_scope": "water only; frozen epsilon/cavity/input per case",
            "root_method": "undamped deterministic Picard",
            "root_tolerance_ev": ROOT_TOLERANCE_EV,
            "maximum_root_iterations": MAX_ROOT_ITERATIONS,
            "second_start": "twice the permanent-source reaction field",
            "force_derivative": (
                "fourth-order Richardson central derivative of the complete "
                "re-solved scalar"
            ),
            "force_coarse_step_angstrom": (NUMERICAL_FORCE_COARSE_STEP_ANGSTROM),
            "force_fine_step_angstrom": (0.5 * NUMERICAL_FORCE_COARSE_STEP_ANGSTROM),
            "force_maximum_error_eV_per_A": (NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM),
            "fit_or_calibration": False,
            "formal_preregistration": True,
            "admission_preregistration_id": admission_preregistration["artifact_id"],
            "accuracy_threshold": None,
        },
        "records": case_records,
        "aggregate": aggregate,
        "decision": {
            "hybrid_feasibility_demonstrated": bool(
                aggregate["root_and_charge_pass_count"] == len(case_records)
            ),
            "experimental_energy_force_execution_gate_passed": (
                energy_force_execution_gate_passed
            ),
            "uniform_accuracy_improvement_demonstrated": bool(all(improved)),
            "chemical_accuracy_admitted": False,
            "multi_solvent_transfer_demonstrated": False,
            "numerical_conservative_force_candidate": (
                energy_force_execution_gate_passed
            ),
            "tier_v_available": False,
            "public_capability_admitted": False,
        },
    }
    measurement_sha256 = canonical_json_sha256(measurement)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "preregistered-experimental-energy-force-admission",
        "status": (
            "candidate-energy-force-gate-passed-awaiting-independent-replay"
            if energy_force_execution_gate_passed
            else "candidate-energy-force-gate-failed"
        ),
        "claim_boundary": (
            "A replicated pass can admit execution of one explicitly named "
            "experimental electrostatic scalar and its error-bounded numerical "
            "scalar-gradient force. It does not admit chemical accuracy, complete "
            "solvation free energy, named-solvent transfer, analytic force, "
            "Hessian/frequency, MD, or Tier V."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "admission_preregistration": {
                "path": str(admission_preregistration_path),
                "sha256": sha256_file(admission_preregistration_path),
                "artifact_id": admission_preregistration["artifact_id"],
            },
            "parent_preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "artifact_id": preregistration["artifact_id"],
            },
            "mace_mdp_checkpoint": checkpoint_record(mdp_checkpoint),
            "mace_polar_checkpoint": checkpoint_record(polar_checkpoint),
            "pcmsolver_library": {
                "path": str(pcmsolver_library),
                "sha256": sha256_file(pcmsolver_library),
            },
        },
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": str(radial.dtype).replace("torch.", ""),
        **measurement,
        "measurement_sha256": measurement_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_MDP_POLAR_HYBRID="
        + json.dumps(
            {
                "artifact": artifact,
                "aggregate": aggregate,
                "decision": payload["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
