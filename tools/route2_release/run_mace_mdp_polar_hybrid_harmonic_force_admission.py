#!/usr/bin/env python3
"""Run the frozen smooth-harmonic hybrid E/F admission panel.

This runner evaluates exactly one disabled operational scalar:

``E = E_vac^MACE-POLAR - 1/2 b(u*)^T A_harm(R)^-1 b(u*)``.

The MACE-MDP permanent point multipoles and the MACE-POLAR induced Gaussian
multipoles share one fixed-dimensional smooth harmonic-Galerkin continuum.  A
complete re-solve is performed at every finite-difference point.  Passing this
panel can support only experimental electrostatic energy and its numerical
scalar-gradient force.  It cannot admit chemical accuracy, complete solvation
free energy, analytic derivatives, Hessians, frequencies, MD, or Tier V.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
)
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarForceComponentEvaluation,
    RichardsonScalarForceEvaluation,
    ScalarEnergySample,
)
from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    MACE_MDPPolarHybridSmoothHarmonicPES,
    NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
    NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
    ROOT_REPLAY_ENERGY_ATOL_EV,
    ROOT_REPLAY_FIELD_ATOL_EV,
    ROOT_TOLERANCE_EV,
)
from maple.solvation.models import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
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

SCHEMA_VERSION = "route2-mace-mdp-polar-hybrid-harmonic-force-admission-v1"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
)
PARENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
EXPECTED_MACE_POLAR_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
TOTAL_CHARGE_ATOL_E = 1.0e-8
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/harmonic_point_source.py",
    "maple/solvation/continuum/harmonic_torch_functional.py",
    "maple/solvation/continuum/harmonic_torch_primitives.py",
    "maple/solvation/derivatives/scalar_finite_difference.py",
    "maple/solvation/experimental/mace_mdp_polar_harmonic.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/models/runtime/analytic_gaussian_multipole.py",
    "maple/solvation/release/evidence.py",
    "tools/route2_release/run_mace_mdp_polar_hybrid_harmonic_force_admission.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
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
    parser.add_argument("--replicate-label", choices=("a", "b"), required=True)
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


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _copy_with_positions(geometry: Atoms, positions: np.ndarray) -> Atoms:
    result = geometry.copy()
    values = np.asarray(positions, dtype=float)
    if values.shape != geometry.positions.shape or not np.all(np.isfinite(values)):
        raise ValueError("replacement positions must be finite with geometry shape.")
    result.positions = values
    return result


def _proper_rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15, rtol=0.0):
        raise RuntimeError("deterministic rotation lost orthogonality.")
    return rotation


def _checkpoint_with_role(path: Path, role: str) -> dict[str, object]:
    record = checkpoint_record(path)
    record["role"] = role
    return record


@dataclass(slots=True)
class _RecordingSampler:
    """Collect fail-closed root metrics without caching any stencil point."""

    pes: MACE_MDPPolarHybridSmoothHarmonicPES
    target_charge_e: float
    state_records: list[dict[str, object]]

    def __init__(
        self, pes: MACE_MDPPolarHybridSmoothHarmonicPES, *, target_charge_e: float
    ) -> None:
        self.pes = pes
        self.target_charge_e = float(target_charge_e)
        self.state_records = []

    @property
    def provider_id(self) -> str:
        return self.pes.provider_id

    def configuration_sha256(self) -> str:
        return self.pes.configuration_sha256()

    def solve(self, geometry: Atoms):
        state = self.pes.solve(geometry)
        charge_error = abs(
            float(np.sum(np.asarray(state.total_source4)[:, 0])) - self.target_charge_e
        )
        if charge_error > TOTAL_CHARGE_ATOL_E:
            raise RuntimeError("recorded harmonic root violates total charge.")
        self.state_records.append(
            {
                "root_sha256": state.root_sha256,
                "geometry_sha256": state.geometry_sha256,
                "topology_id": state.coefficient_topology_id,
                "energy_eV": state.total_energy_ev,
                "polarization_energy_eV": state.polarization_energy_ev,
                "primal_residual_eV": state.primal_residual_ev,
                "replay_field_max_abs_difference_eV": (
                    state.replay_field_max_abs_difference_ev
                ),
                "replay_energy_abs_difference_eV": (
                    state.replay_energy_abs_difference_ev
                ),
                "total_charge_absolute_error_e": charge_error,
                "cold_iterations": state.cold_iterations,
                "wide_iterations": state.wide_iterations,
            }
        )
        return state

    def sample(self, geometry: Atoms) -> ScalarEnergySample:
        state = self.solve(geometry)
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=state.coefficient_topology_id,
        )

    def summary(self) -> dict[str, object]:
        if not self.state_records:
            raise RuntimeError("root recorder contains no states.")
        return {
            "state_count": len(self.state_records),
            "maximum_primal_residual_eV": max(
                float(record["primal_residual_eV"]) for record in self.state_records
            ),
            "maximum_replay_field_difference_eV": max(
                float(record["replay_field_max_abs_difference_eV"])
                for record in self.state_records
            ),
            "maximum_replay_energy_difference_eV": max(
                float(record["replay_energy_abs_difference_eV"])
                for record in self.state_records
            ),
            "maximum_total_charge_error_e": max(
                float(record["total_charge_absolute_error_e"])
                for record in self.state_records
            ),
            "maximum_cold_iterations": max(
                int(record["cold_iterations"]) for record in self.state_records
            ),
            "maximum_wide_iterations": max(
                int(record["wide_iterations"]) for record in self.state_records
            ),
            "topology_ids": sorted(
                {str(record["topology_id"]) for record in self.state_records}
            ),
        }


def _force_record(evaluation: RichardsonScalarForceEvaluation) -> dict[str, object]:
    forces = np.asarray(evaluation.forces_eV_per_A, dtype=float)
    errors = np.asarray(evaluation.error_estimates_eV_per_A, dtype=float)
    return {
        "evaluation_sha256": evaluation.evaluation_sha256,
        "forces_eV_per_A": forces.tolist(),
        "error_estimates_eV_per_A": errors.tolist(),
        "maximum_error_estimate_eV_per_A": (evaluation.maximum_error_estimate_eV_per_A),
        "net_force_norm_eV_per_A": float(np.linalg.norm(np.sum(forces, axis=0))),
        "finite": bool(np.all(np.isfinite(forces)) and np.all(np.isfinite(errors))),
    }


def _component_record(
    evaluation: RichardsonScalarForceComponentEvaluation,
) -> dict[str, object]:
    return {
        "atom_index": evaluation.atom_index,
        "axis_index": evaluation.axis_index,
        "force_eV_per_A": evaluation.force_eV_per_A,
        "error_estimate_eV_per_A": evaluation.error_estimate_eV_per_A,
        "displaced_state_sha256": list(evaluation.displaced_state_sha256),
    }


def _displaced_component_sample(
    sampler: _RecordingSampler,
    geometry: Atoms,
    *,
    atom: int,
    axis: int,
    delta: float,
) -> ScalarEnergySample:
    displaced = geometry.copy()
    displaced.positions[atom, axis] += float(delta)
    return sampler.sample(displaced)


def _component_convergence_record(
    sampler: _RecordingSampler,
    geometry: Atoms,
    *,
    atom: int,
    axis: int,
    coarse_step: float,
    h4_step: float,
    maximum_local_error: float,
    maximum_h4_difference: float,
) -> dict[str, object]:
    center = sampler.sample(geometry)
    samples: dict[str, ScalarEnergySample] = {}
    for label, delta in (
        ("plus_h", coarse_step),
        ("minus_h", -coarse_step),
        ("plus_h2", 0.5 * coarse_step),
        ("minus_h2", -0.5 * coarse_step),
        ("plus_h4", h4_step),
        ("minus_h4", -h4_step),
    ):
        sample = _displaced_component_sample(
            sampler, geometry, atom=atom, axis=axis, delta=delta
        )
        if sample.topology_id != center.topology_id:
            raise RuntimeError("component convergence stencil changed topology.")
        samples[label] = sample
    derivative_h = (samples["plus_h"].energy_eV - samples["minus_h"].energy_eV) / (
        2.0 * coarse_step
    )
    derivative_h2 = (
        samples["plus_h2"].energy_eV - samples["minus_h2"].energy_eV
    ) / coarse_step
    derivative_h4 = (samples["plus_h4"].energy_eV - samples["minus_h4"].energy_eV) / (
        2.0 * h4_step
    )
    richardson_derivative = (4.0 * derivative_h2 - derivative_h) / 3.0
    local_error = abs(richardson_derivative - derivative_h2)
    h4_difference = abs(richardson_derivative - derivative_h4)
    return {
        "atom_index": atom,
        "axis_index": axis,
        "coarse_step_angstrom": coarse_step,
        "fine_step_angstrom": 0.5 * coarse_step,
        "independent_step_angstrom": h4_step,
        "energy_derivative_h_eV_per_A": derivative_h,
        "energy_derivative_h2_eV_per_A": derivative_h2,
        "energy_derivative_h4_eV_per_A": derivative_h4,
        "richardson_energy_derivative_eV_per_A": richardson_derivative,
        "richardson_force_eV_per_A": -richardson_derivative,
        "local_error_estimate_eV_per_A": local_error,
        "h4_difference_eV_per_A": h4_difference,
        "maximum_local_error_eV_per_A": maximum_local_error,
        "maximum_h4_difference_eV_per_A": maximum_h4_difference,
        "topology_id": center.topology_id,
        "center_state_sha256": center.state_sha256,
        "displaced_state_sha256": {
            label: sample.state_sha256 for label, sample in samples.items()
        },
        "gate_passed": bool(
            local_error <= maximum_local_error
            and h4_difference <= maximum_h4_difference
        ),
    }


def _directional_record(
    sampler: _RecordingSampler,
    geometry: Atoms,
    forces: np.ndarray,
    *,
    seed: int,
    step: float,
    maximum_error: float,
) -> dict[str, object]:
    direction = np.random.default_rng(seed).normal(size=geometry.positions.shape)
    direction /= np.linalg.norm(direction)
    plus = _copy_with_positions(geometry, geometry.positions + step * direction)
    minus = _copy_with_positions(geometry, geometry.positions - step * direction)
    plus_sample = sampler.sample(plus)
    minus_sample = sampler.sample(minus)
    if plus_sample.topology_id != minus_sample.topology_id:
        raise RuntimeError("independent directional stencil changed topology.")
    energy_derivative = (plus_sample.energy_eV - minus_sample.energy_eV) / (2.0 * step)
    force_projection = -float(np.vdot(forces, direction))
    error = abs(energy_derivative - force_projection)
    return {
        "seed": seed,
        "step_angstrom": step,
        "direction": direction.tolist(),
        "energy_derivative_eV_per_A": energy_derivative,
        "negative_force_projection_eV_per_A": force_projection,
        "absolute_error_eV_per_A": error,
        "maximum_error_eV_per_A": maximum_error,
        "plus_state_sha256": plus_sample.state_sha256,
        "minus_state_sha256": minus_sample.state_sha256,
        "gate_passed": bool(error <= maximum_error),
    }


def _closed_loop_record(
    sampler: _RecordingSampler,
    backend: RichardsonScalarForce,
    geometry: Atoms,
    *,
    dofs: tuple[tuple[int, int], tuple[int, int]],
    half_width: float,
    maximum_work: float,
) -> dict[str, object]:
    (atom_x, axis_x), (atom_y, axis_y) = dofs
    edge_contract = (
        ("bottom", 0.0, -half_width, atom_x, axis_x, 2.0 * half_width),
        ("right", half_width, 0.0, atom_y, axis_y, 2.0 * half_width),
        ("top", 0.0, half_width, atom_x, axis_x, -2.0 * half_width),
        ("left", -half_width, 0.0, atom_y, axis_y, -2.0 * half_width),
    )
    edges = []
    work = 0.0
    numerical_error_bound = 0.0
    for label, x, y, atom, axis, displacement in edge_contract:
        midpoint = geometry.copy()
        midpoint.positions[atom_x, axis_x] += x
        midpoint.positions[atom_y, axis_y] += y
        component = backend.evaluate_component(
            sampler, midpoint, atom_index=atom, axis_index=axis
        )
        contribution = component.force_eV_per_A * displacement
        error_bound = component.error_estimate_eV_per_A * abs(displacement)
        work += contribution
        numerical_error_bound += error_bound
        edges.append(
            {
                "label": label,
                "midpoint_offsets_angstrom": [x, y],
                "displacement_angstrom": displacement,
                "work_eV": contribution,
                "work_error_bound_eV": error_bound,
                "force_component": _component_record(component),
            }
        )
    guarded_work = abs(work) + numerical_error_bound
    return {
        "cartesian_dofs": [list(dof) for dof in dofs],
        "half_width_angstrom": half_width,
        "orientation": "counterclockwise",
        "edges": edges,
        "closed_loop_work_eV": work,
        "absolute_work_eV": abs(work),
        "numerical_work_error_bound_eV": numerical_error_bound,
        "guarded_absolute_work_eV": guarded_work,
        "maximum_guarded_absolute_work_eV": maximum_work,
        "gate_passed": bool(guarded_work <= maximum_work),
    }


def _build_pes(
    hybrid: object, radial: object, atoms: Atoms, radii: np.ndarray, contract
):
    return MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=hybrid,
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=radii,
        dtype=radial.dtype,
        device=radial.device,
        transition_width_angstrom2=float(contract["transition_width_angstrom2"]),
        surface_lmax=int(contract["surface_lmax"]),
        exposure_lmax=int(contract["exposure_lmax"]),
        exposure_radial_quadrature_order=int(
            contract["exposure_radial_quadrature_order"]
        ),
        source_radial_quadrature_order=int(contract["source_radial_quadrature_order"]),
        green_radial_quadrature_order=int(contract["green_radial_quadrature_order"]),
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=float(contract["force_coarse_step_angstrom"]),
            maximum_error_eV_per_A=float(
                contract["force_maximum_local_error_estimate_ev_per_angstrom"]
            ),
        ),
    )


def _benzene_record(
    *,
    asset_root: Path,
    parent: dict[str, object],
    hybrid: object,
    radial: object,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
) -> dict[str, object]:
    records = parent.get("records")
    if not isinstance(records, list):
        raise RuntimeError("parent preregistration omits records.")
    compound_id = str(force_panel["gepol_regression_compound_id"])
    selected = [
        record
        for record in records
        if isinstance(record, dict) and record.get("compound_id") == compound_id
    ]
    if len(selected) != 1:
        raise RuntimeError("benzene regression record is not unique.")
    record = selected[0]
    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name="benzene MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    result_path = (
        asset_root
        / PANEL_RELATIVE_ROOT
        / compound_id
        / CUTOFF_DIRECTORY
        / "result.json"
    )
    result = _load_json(result_path, name="benzene frozen projection result")
    inputs = result.get("inputs")
    parsed = inputs.get("parsed_pcm_input") if isinstance(inputs, dict) else None
    if not isinstance(parsed, dict):
        raise RuntimeError("benzene result omits frozen cavity radii.")
    radii = np.asarray(parsed.get("cavity_radii_angstrom"), dtype=float)
    if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
        raise RuntimeError("benzene frozen cavity radii are invalid.")
    pes = _build_pes(hybrid, radial, atoms, radii, runtime_contract)
    sampler = _RecordingSampler(pes, target_charge_e=0.0)
    dof = force_panel["gepol_regression_cartesian_dof"]
    if not isinstance(dof, list) or len(dof) != 2:
        raise RuntimeError("benzene regression Cartesian DOF is invalid.")
    convergence = _component_convergence_record(
        sampler,
        atoms,
        atom=int(dof[0]),
        axis=int(dof[1]),
        coarse_step=float(runtime_contract["force_coarse_step_angstrom"]),
        h4_step=float(runtime_contract["independent_force_step_angstrom"]),
        maximum_local_error=float(
            runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"]
        ),
        maximum_h4_difference=float(
            force_panel["maximum_benzene_h4_difference_ev_per_angstrom"]
        ),
    )
    center = sampler.solve(atoms)
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "atom_count": len(atoms),
        "cavity_radii_angstrom": radii.tolist(),
        "center_energy_eV": center.total_energy_ev,
        "center_polarization_energy_eV": center.polarization_energy_ev,
        "center_root_sha256": center.root_sha256,
        "coefficient_topology_id": center.coefficient_topology_id,
        "force_convergence": convergence,
        "root_summary": sampler.summary(),
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
        },
        "gate_passed": bool(convergence["gate_passed"]),
    }


def _water_record(
    *,
    hybrid: object,
    radial: object,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
) -> dict[str, object]:
    positions = np.asarray(force_panel["water_geometry_angstrom"], dtype=float)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("frozen water geometry is invalid.")
    atoms = Atoms(numbers=[8, 1, 1], positions=positions)
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    radii = np.asarray(
        smd_water_coulomb_radii(atoms.get_chemical_symbols()), dtype=float
    )
    pes = _build_pes(hybrid, radial, atoms, radii, runtime_contract)
    sampler = _RecordingSampler(pes, target_charge_e=0.0)
    backend = pes.force_backend
    center = sampler.solve(atoms)
    center_sample = ScalarEnergySample(
        energy_eV=center.total_energy_ev,
        state_sha256=center.root_sha256,
        topology_id=center.coefficient_topology_id,
    )
    center_force = backend.evaluate(sampler, atoms, central_sample=center_sample)
    center_force_record = _force_record(center_force)
    forces = np.asarray(center_force.forces_eV_per_A, dtype=float)
    direction = _directional_record(
        sampler,
        atoms,
        forces,
        seed=int(force_panel["independent_direction_seed"]),
        step=float(runtime_contract["independent_force_step_angstrom"]),
        maximum_error=float(
            force_panel["maximum_independent_directional_error_ev_per_angstrom"]
        ),
    )

    translation = np.asarray(force_panel["translation_angstrom"], dtype=float)
    translated = _copy_with_positions(atoms, atoms.positions + translation)
    translated_state = sampler.solve(translated)
    translated_force = backend.evaluate(
        sampler,
        translated,
        central_sample=ScalarEnergySample(
            energy_eV=translated_state.total_energy_ev,
            state_sha256=translated_state.root_sha256,
            topology_id=translated_state.coefficient_topology_id,
        ),
    )
    translated_forces = np.asarray(translated_force.forces_eV_per_A, dtype=float)
    translation_force_delta = translated_forces - forces
    translation_force_relative = float(
        np.linalg.norm(translation_force_delta)
        / max(np.linalg.norm(forces), np.finfo(float).tiny)
    )
    translation_force_absolute = float(np.max(np.abs(translation_force_delta)))
    translation_energy_error = abs(
        translated_state.total_energy_ev - center.total_energy_ev
    )
    translation_record = {
        "translation_angstrom": translation.tolist(),
        "energy_absolute_error_eV": translation_energy_error,
        "force_relative_error": translation_force_relative,
        "force_maximum_absolute_error_eV_per_A": translation_force_absolute,
        "translated_force": _force_record(translated_force),
        "translated_root_sha256": translated_state.root_sha256,
        "gate_passed": bool(
            translation_energy_error
            <= float(force_panel["maximum_translation_energy_error_ev"])
            and translation_force_relative
            <= float(force_panel["maximum_translation_force_relative_error"])
            and translation_force_absolute
            <= float(
                force_panel["maximum_translation_force_absolute_error_ev_per_angstrom"]
            )
        ),
    }

    rotation = _proper_rotation(int(force_panel["rotation_seed"]))
    centroid = np.mean(atoms.positions, axis=0)
    rotated_positions = (atoms.positions - centroid) @ rotation.T + centroid
    rotated = _copy_with_positions(atoms, rotated_positions)
    rotated_state = sampler.solve(rotated)
    rotated_force = backend.evaluate(
        sampler,
        rotated,
        central_sample=ScalarEnergySample(
            energy_eV=rotated_state.total_energy_ev,
            state_sha256=rotated_state.root_sha256,
            topology_id=rotated_state.coefficient_topology_id,
        ),
    )
    rotated_forces = np.asarray(rotated_force.forces_eV_per_A, dtype=float)
    expected_rotated_forces = forces @ rotation.T
    rotation_force_delta = rotated_forces - expected_rotated_forces
    rotation_force_relative = float(
        np.linalg.norm(rotation_force_delta)
        / max(np.linalg.norm(expected_rotated_forces), np.finfo(float).tiny)
    )
    rotation_force_absolute = float(np.max(np.abs(rotation_force_delta)))
    rotation_energy_error = abs(rotated_state.total_energy_ev - center.total_energy_ev)
    rotation_record = {
        "seed": int(force_panel["rotation_seed"]),
        "rotation_matrix": rotation.tolist(),
        "energy_absolute_error_eV": rotation_energy_error,
        "force_relative_error": rotation_force_relative,
        "force_maximum_absolute_error_eV_per_A": rotation_force_absolute,
        "rotated_force": _force_record(rotated_force),
        "rotated_root_sha256": rotated_state.root_sha256,
        "gate_passed": bool(
            rotation_energy_error
            <= float(force_panel["maximum_rotation_energy_error_ev"])
            and rotation_force_relative
            <= float(force_panel["maximum_rotation_force_relative_error"])
            and rotation_force_absolute
            <= float(
                force_panel["maximum_rotation_force_absolute_error_ev_per_angstrom"]
            )
        ),
    }

    raw_dofs = force_panel["closed_loop_cartesian_dofs"]
    if (
        not isinstance(raw_dofs, list)
        or len(raw_dofs) != 2
        or any(not isinstance(dof, list) or len(dof) != 2 for dof in raw_dofs)
    ):
        raise RuntimeError("closed-loop Cartesian DOFs are invalid.")
    dofs = tuple((int(dof[0]), int(dof[1])) for dof in raw_dofs)
    loop = _closed_loop_record(
        sampler,
        backend,
        atoms,
        dofs=dofs,  # type: ignore[arg-type]
        half_width=float(force_panel["closed_loop_half_width_angstrom"]),
        maximum_work=float(force_panel["maximum_closed_loop_work_abs_ev"]),
    )
    root_summary = sampler.summary()
    root_gate = bool(
        float(root_summary["maximum_primal_residual_eV"]) < ROOT_TOLERANCE_EV
        and float(root_summary["maximum_replay_field_difference_eV"])
        <= ROOT_REPLAY_FIELD_ATOL_EV
        and float(root_summary["maximum_replay_energy_difference_eV"])
        <= ROOT_REPLAY_ENERGY_ATOL_EV
        and float(root_summary["maximum_total_charge_error_e"]) <= TOTAL_CHARGE_ATOL_E
        and len(root_summary["topology_ids"]) == 1
    )
    return {
        "system": "water",
        "geometry_angstrom": positions.tolist(),
        "cavity_radii_angstrom": radii.tolist(),
        "center_energy_eV": center.total_energy_ev,
        "center_polarization_energy_eV": center.polarization_energy_ev,
        "center_root_sha256": center.root_sha256,
        "coefficient_topology_id": center.coefficient_topology_id,
        "center_force": center_force_record,
        "independent_h4_directional_check": direction,
        "translation": translation_record,
        "rotation": rotation_record,
        "closed_loop": loop,
        "root_summary": root_summary,
        "root_gate_passed": root_gate,
        "gate_passed": bool(
            center_force_record["finite"]
            and float(center_force_record["maximum_error_estimate_eV_per_A"])
            <= NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
            and float(center_force_record["net_force_norm_eV_per_A"])
            <= float(force_panel["maximum_net_force_norm_ev_per_angstrom"])
            and direction["gate_passed"]
            and translation_record["gate_passed"]
            and rotation_record["gate_passed"]
            and loop["gate_passed"]
            and root_gate
        ),
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mace_mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.mace_polar_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(mdp_checkpoint) != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-MDP checkpoint does not match the frozen model.")
    if sha256_file(polar_checkpoint) != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-POLAR checkpoint does not match the frozen model.")

    preregistration_path = repository.root / PREREGISTRATION_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, name="smooth-harmonic hybrid force preregistration"
    )
    if (
        preregistration.get("target_profile_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
        or preregistration.get("target_scalar_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        or preregistration.get("status") != "frozen-before-full-admission-execution"
    ):
        raise RuntimeError("smooth-harmonic force preregistration identity is invalid.")
    if preregistration.get("target_capabilities") != {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }:
        raise RuntimeError("smooth-harmonic target capabilities drifted.")
    runtime_contract = preregistration.get("frozen_runtime_contract")
    force_panel = preregistration.get("force_panel")
    parent_binding = preregistration.get("parent_panel")
    if not all(
        isinstance(value, dict)
        for value in (runtime_contract, force_panel, parent_binding)
    ):
        raise RuntimeError("smooth-harmonic preregistration is incomplete.")
    assert isinstance(runtime_contract, dict)
    assert isinstance(force_panel, dict)
    assert isinstance(parent_binding, dict)
    if (
        float(runtime_contract["force_coarse_step_angstrom"])
        != NUMERICAL_FORCE_COARSE_STEP_ANGSTROM
        or float(runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"])
        != NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
    ):
        raise RuntimeError("force implementation drifted from preregistration.")
    if (
        runtime_contract.get("mace_mdp_checkpoint_sha256")
        != MACE_MDP_EXPECTED_CHECKPOINT_SHA256
        or runtime_contract.get("mace_polar_checkpoint_sha256")
        != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256
        or runtime_contract.get("mace_polar_long_range_evaluator")
        != MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
    ):
        raise RuntimeError("checkpoint/evaluator preregistration drifted.")

    parent_path = repository.root / PARENT_PREREGISTRATION_RELATIVE_PATH
    if parent_binding.get("relative_path") != PARENT_PREREGISTRATION_RELATIVE_PATH:
        raise RuntimeError("parent preregistration path drifted.")
    _validated_sha(
        parent_path,
        parent_binding.get("sha256"),
        name="parent PCM preregistration",
    )
    parent = _load_json(parent_path, name="parent PCM preregistration")

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
    benzene = _benzene_record(
        asset_root=asset_root,
        parent=parent,
        hybrid=hybrid,
        radial=radial,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    water = _water_record(
        hybrid=hybrid,
        radial=radial,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    maximum_local_error = max(
        float(benzene["force_convergence"]["local_error_estimate_eV_per_A"]),
        float(water["center_force"]["maximum_error_estimate_eV_per_A"]),
        float(
            water["translation"]["translated_force"]["maximum_error_estimate_eV_per_A"]
        ),
        float(water["rotation"]["rotated_force"]["maximum_error_estimate_eV_per_A"]),
        max(
            float(edge["force_component"]["error_estimate_eV_per_A"])
            for edge in water["closed_loop"]["edges"]
        ),
    )
    execution_gate = bool(
        benzene["gate_passed"]
        and water["gate_passed"]
        and maximum_local_error
        <= float(runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"])
    )
    measurement = {
        "protocol": {
            "profile_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
            ),
            "scalar_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
            ),
            "preregistration_artifact_id": preregistration["artifact_id"],
            "preregistration_sha256": sha256_file(preregistration_path),
            "parent_preregistration_artifact_id": parent["artifact_id"],
            "parent_preregistration_sha256": sha256_file(parent_path),
            "frozen_runtime_contract": runtime_contract,
            "force_panel": force_panel,
            "fit_calibration_or_case_selection": False,
        },
        "benzene_gepol_regression": benzene,
        "water_force_symmetry_loop": water,
        "aggregate": {
            "maximum_local_richardson_error_eV_per_A": maximum_local_error,
            "benzene_h4_difference_eV_per_A": float(
                benzene["force_convergence"]["h4_difference_eV_per_A"]
            ),
            "water_independent_directional_error_eV_per_A": float(
                water["independent_h4_directional_check"]["absolute_error_eV_per_A"]
            ),
            "water_translation_energy_error_eV": float(
                water["translation"]["energy_absolute_error_eV"]
            ),
            "water_translation_force_relative_error": float(
                water["translation"]["force_relative_error"]
            ),
            "water_rotation_energy_error_eV": float(
                water["rotation"]["energy_absolute_error_eV"]
            ),
            "water_rotation_force_relative_error": float(
                water["rotation"]["force_relative_error"]
            ),
            "water_closed_loop_guarded_absolute_work_eV": float(
                water["closed_loop"]["guarded_absolute_work_eV"]
            ),
            "all_execution_gates_passed": execution_gate,
        },
        "decision": {
            "candidate_energy_force_gate_passed": execution_gate,
            "awaiting_independent_replay": execution_gate,
            "public_capability_admitted": False,
            "chemical_accuracy_admitted": False,
            "complete_solvation_free_energy_admitted": False,
            "analytic_force_admitted": False,
            "hessian_frequency_md_admitted": False,
            "tier_v_admitted": False,
        },
    }
    measurement_sha256 = canonical_json_sha256(measurement)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload = {
        "artifact_id": (
            "route2-mace-mdp-polar-hybrid-harmonic-force-admission-"
            f"replicate-{args.replicate_label}-v1"
        ),
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "preregistered-experimental-energy-force-admission-run",
        "status": (
            "candidate-energy-force-gate-passed-awaiting-independent-replay"
            if execution_gate
            else "candidate-energy-force-gate-failed"
        ),
        "claim_boundary": preregistration["claim_boundary"],
        "capabilities": NO_CAPABILITIES,
        "replicate_label": args.replicate_label,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "artifact_id": preregistration["artifact_id"],
            },
            "parent_preregistration": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "artifact_id": parent["artifact_id"],
            },
            "mace_mdp_checkpoint": _checkpoint_with_role(
                mdp_checkpoint, "mace-mdp-official-checkpoint"
            ),
            "mace_polar_checkpoint": _checkpoint_with_role(
                polar_checkpoint, "mace-polar-1-m-official-checkpoint"
            ),
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
        "ROUTE2_MACE_MDP_POLAR_HARMONIC_FORCE="
        + json.dumps(
            {
                "artifact": artifact,
                "aggregate": measurement["aggregate"],
                "decision": measurement["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
