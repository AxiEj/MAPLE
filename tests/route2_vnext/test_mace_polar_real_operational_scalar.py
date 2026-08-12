from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.units import kcal, mol

from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.api.profiles import (
    OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.continuum import (
    build_water_radial_gto_cpcm_backend,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_REAL_MACEPOL") != "1",
    reason="set MAPLE_ROUTE2_REAL_MACEPOL=1 in the pinned real-checkpoint job",
)

_LEGACY_POINT_SOURCE_GTO_RECEIVER_ELECTROSTATIC_KCAL_MOL = -1.1554017711374178
_LEGACY_POINT_SOURCE_GTO_RECEIVER_CDS_KCAL_MOL = 2.764796099834587
_LEGACY_POINT_SOURCE_GTO_RECEIVER_TOTAL_KCAL_MOL = 1.609394328697169
_VNEXT_LOCAL_JET_ELECTROSTATIC_KCAL_MOL = -1.483425546707858
_FREESOLV_METHANE_TOTAL_KCAL_MOL = 2.0


def _root_context(geometry: Atoms, suffix: str) -> str:
    digest = hashlib.sha256(np.asarray(geometry.positions).tobytes()).hexdigest()
    return f"real-water-radial-cpcm-194/{suffix}/{digest}"


def _rotation() -> np.ndarray:
    axis = np.asarray([1.0, -2.0, 3.0], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotate_radial_blocks(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    rotated = np.array(values, dtype=float, copy=True)
    for raw_indices in ((2, 3, 4), (5, 6, 7)):
        raw = rotated[:, raw_indices]
        cartesian = raw[:, (2, 0, 1)]
        rotated_cartesian = cartesian @ rotation.T
        rotated[:, raw_indices] = rotated_cartesian[:, (1, 2, 0)]
    return rotated


def _radial_system(atoms: Atoms):
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
        checkpoint_path=os.environ.get("ROUTE2_MACE_CHECKPOINT"),
    )
    continuum = build_water_radial_gto_cpcm_backend(atoms.get_chemical_symbols())
    coordinates = LinearChargeCoordinates(
        len(atoms),
        total_charge=0.0,
        source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        component_scales=(1.0,) * 8,
    )
    equation = ReducedStateEquation(
        coordinates,
        ElectronicResponseEquationAdapter(model, MACE_POLAR_RADIAL_GTO_COUPLING_ID),
        continuum,
    )
    scalar = OperationalElectrostaticScalar(
        equation,
        VacuumScalarEquationAdapter(model),
        profile_id=OPERATIONAL_CPCM_RADIAL_GTO_ELECTROSTATIC_PROFILE_V1,
    )
    return model, continuum, coordinates, equation, scalar


def test_real_water_194_point_same_scalar_directional_and_symmetry_audit():
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]]
        ),
        info={"charge": 0, "mult": 1},
    )
    model, continuum, coordinates, equation, scalar = _radial_system(atoms)
    profile = get_solvation_profile(scalar.profile_id)
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()

    primal_options = FixedPointOptions(
        tolerance=1.0e-12,
        max_iterations=100,
        damping=0.7,
        history=6,
    )

    def solve(geometry: Atoms, *, initial_y=None, suffix: str):
        return solve_fixed_point(
            equation,
            geometry,
            scalar_id=scalar.scalar_id,
            profile_id=scalar.profile_id,
            scalar_binding=scalar,
            root_context_id=_root_context(geometry, suffix),
            initial_y=initial_y,
            options=primal_options,
        )

    cold = solve(atoms, suffix="base")
    warm = solve(
        atoms,
        initial_y=np.asarray(cold.y) + 1.0e-6,
        suffix="base",
    )
    assert roots_numerically_equivalent(cold, warm)
    assert cold.actual_unmixed_residual_norm <= primal_options.tolerance
    assert warm.actual_unmixed_residual_norm <= primal_options.tolerance
    source_difference = float(np.linalg.norm(cold.source_array() - warm.source_array()))
    energy_difference = abs(
        scalar.evaluate(atoms, cold.y).total_energy
        - scalar.evaluate(atoms, warm.y).total_energy
    )
    assert source_difference <= 1.0e-8
    assert energy_difference <= 1.0e-8

    gradient = scalar.implicit_gradient(
        atoms,
        cold,
        adjoint_options=AdjointOptions(
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=500,
        ),
    )
    assert gradient.adjoint.converged
    assert gradient.adjoint.true_residual_norm <= gradient.adjoint.acceptance_tolerance
    rng = np.random.default_rng(20260813)
    direction = rng.normal(size=(len(atoms), 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = float(
        np.vdot(
            np.asarray(gradient.total_coordinate_gradient).reshape(len(atoms), 3),
            direction,
        )
    )

    measurements = []
    topology_hashes = set()
    for step in (4.0e-4, 2.0e-4, 1.0e-4):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        plus_state = solve(plus, initial_y=np.asarray(cold.y), suffix=f"plus-{step}")
        minus_state = solve(minus, initial_y=np.asarray(cold.y), suffix=f"minus-{step}")
        plus_energy = scalar.evaluate(plus, plus_state.y).total_energy
        minus_energy = scalar.evaluate(minus, minus_state.y).total_energy
        finite_difference = (plus_energy - minus_energy) / (2.0 * step)
        absolute_error = abs(analytic - finite_difference)
        relative_error = absolute_error / max(abs(finite_difference), 1.0e-15)
        assert absolute_error <= 5.0e-4
        assert relative_error <= 2.0e-3
        assert plus_state.actual_unmixed_residual_norm <= primal_options.tolerance
        assert minus_state.actual_unmixed_residual_norm <= primal_options.tolerance
        plus_continuum = continuum.build_state(plus, plus_state.source)
        minus_continuum = continuum.build_state(minus, minus_state.source)
        topology_hashes.update(
            (
                plus_continuum.surface.topology_hash,
                minus_continuum.surface.topology_hash,
            )
        )
        measurements.append(
            {
                "step_A": step,
                "analytic_eV_per_A": analytic,
                "finite_difference_eV_per_A": finite_difference,
                "absolute_error_eV_per_A": absolute_error,
                "relative_error": relative_error,
                "plus_primal_residual": plus_state.actual_unmixed_residual_norm,
                "minus_primal_residual": minus_state.actual_unmixed_residual_norm,
            }
        )
    base_continuum = continuum.build_state(atoms, cold.source)
    topology_hashes.add(base_continuum.surface.topology_hash)
    assert len(topology_hashes) == 1
    assert base_continuum.surface.candidate_count == 194 * len(atoms)

    base_energy = gradient.scalar.total_energy
    base_forces = np.asarray(gradient.forces).reshape(len(atoms), 3)
    net_force_norm = float(np.linalg.norm(np.sum(base_forces, axis=0)))
    centred_positions = atoms.positions - np.mean(atoms.positions, axis=0)
    torque_norm = float(
        np.linalg.norm(np.sum(np.cross(centred_positions, base_forces), axis=0))
    )
    net_force_gate_pass = net_force_norm <= 1.0e-5
    torque_gate_pass = torque_norm <= 1.0e-4

    translated = atoms.copy()
    translated.positions += np.asarray([1.7, -0.8, 0.5])
    translated_state = solve(
        translated, initial_y=np.asarray(cold.y), suffix="translated"
    )
    translated_gradient = scalar.implicit_gradient(
        translated,
        translated_state,
        adjoint_options=AdjointOptions(
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=500,
        ),
    )
    translated_forces = np.asarray(translated_gradient.forces).reshape(len(atoms), 3)
    translation_energy_abs = abs(translated_gradient.scalar.total_energy - base_energy)
    translation_force_relative = float(
        np.linalg.norm(translated_forces - base_forces)
        / max(np.linalg.norm(base_forces), 1.0)
    )
    translated_net_force_norm = float(np.linalg.norm(np.sum(translated_forces, axis=0)))
    translation_energy_gate_pass = translation_energy_abs <= 1.0e-6
    translation_force_gate_pass = translation_force_relative <= 1.0e-4
    translated_net_force_gate_pass = translated_net_force_norm <= 1.0e-5
    topology_hashes.add(
        continuum.build_state(translated, translated_state.source).surface.topology_hash
    )

    rotation = _rotation()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated_state = solve(rotated, suffix="rotated")
    rotated_gradient = scalar.implicit_gradient(
        rotated,
        rotated_state,
        adjoint_options=AdjointOptions(
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=500,
        ),
    )
    rotated_forces = np.asarray(rotated_gradient.forces).reshape(len(atoms), 3)
    base_direct_forces = -np.asarray(
        gradient.scalar.direct_coordinate_gradient
    ).reshape(len(atoms), 3)
    rotated_direct_forces = -np.asarray(
        rotated_gradient.scalar.direct_coordinate_gradient
    ).reshape(len(atoms), 3)
    base_adjoint_forces = base_forces - base_direct_forces
    rotated_adjoint_forces = rotated_forces - rotated_direct_forces
    base_vacuum_forces = -np.asarray(scalar.vacuum.coordinate_gradient(atoms)).reshape(
        len(atoms), 3
    )
    rotated_vacuum_forces = -np.asarray(
        scalar.vacuum.coordinate_gradient(rotated)
    ).reshape(len(atoms), 3)
    base_continuum_direct_forces = base_direct_forces - base_vacuum_forces
    rotated_continuum_direct_forces = rotated_direct_forces - rotated_vacuum_forces

    def covariance_error(rotated_values, base_values):
        return float(
            np.linalg.norm(rotated_values - base_values @ rotation.T)
            / max(np.linalg.norm(base_values), 1.0)
        )

    component_rotation_errors = {
        "vacuum_force": covariance_error(rotated_vacuum_forces, base_vacuum_forces),
        "continuum_direct_force": covariance_error(
            rotated_continuum_direct_forces, base_continuum_direct_forces
        ),
        "outer_adjoint_force": covariance_error(
            rotated_adjoint_forces, base_adjoint_forces
        ),
        "source": float(
            np.linalg.norm(
                rotated_state.source_array()
                - _rotate_radial_blocks(cold.source_array(), rotation)
            )
            / max(np.linalg.norm(cold.source_array()), 1.0)
        ),
    }
    rotation_energy_abs = abs(rotated_gradient.scalar.total_energy - base_energy)
    rotation_force_relative = float(
        np.linalg.norm(rotated_forces - base_forces @ rotation.T)
        / max(np.linalg.norm(base_forces), 1.0)
    )
    rotated_net_force_norm = float(np.linalg.norm(np.sum(rotated_forces, axis=0)))
    rotation_energy_gate_pass = rotation_energy_abs <= 1.0e-6
    rotation_force_gate_pass = rotation_force_relative <= 1.0e-4
    rotated_net_force_gate_pass = rotated_net_force_norm <= 1.0e-5
    topology_hashes.add(
        continuum.build_state(rotated, rotated_state.source).surface.topology_hash
    )
    assert len(topology_hashes) == 1
    symmetry_gates = {
        "net_force": net_force_gate_pass,
        "torque": torque_gate_pass,
        "translation_energy": translation_energy_gate_pass,
        "translation_force": translation_force_gate_pass,
        "translated_net_force": translated_net_force_gate_pass,
        "rotation_energy": rotation_energy_gate_pass,
        "rotation_force": rotation_force_gate_pass,
        "rotated_net_force": rotated_net_force_gate_pass,
    }
    # This is an audit, not an admission test.  Any newly complete gate set must
    # trigger a deliberate evidence/admission review rather than silently
    # enabling Tier F.  The current expected negative is the torque gate.
    assert not all(symmetry_gates.values())

    print(
        "ROUTE2_REAL_WATER_SAME_SCALAR_CANARY="
        + json.dumps(
            {
                "checkpoint_sha256": model.provenance.checkpoint_sha256,
                "profile_id": scalar.profile_id,
                "scalar_id": scalar.scalar_id,
                "state_equation_id": equation.state_equation_id,
                "profile_admitted": False,
                "surface_points_per_atom": 194,
                "surface_size": base_continuum.surface.candidate_count,
                "topology_hash": base_continuum.surface.topology_hash,
                "cold_warm_source_l2": source_difference,
                "cold_warm_energy_abs_eV": energy_difference,
                "primal_residual": cold.actual_unmixed_residual_norm,
                "adjoint_residual": gradient.adjoint.true_residual_norm,
                "vacuum_energy_eV": gradient.scalar.vacuum_energy,
                "continuum_energy_eV": gradient.scalar.continuum_energy,
                "total_energy_eV": gradient.scalar.total_energy,
                "force_norm_eV_per_A": float(np.linalg.norm(gradient.forces)),
                "net_force_norm_eV_per_A": net_force_norm,
                "torque_norm_eV": torque_norm,
                "translation_energy_abs_eV": translation_energy_abs,
                "translation_force_relative": translation_force_relative,
                "translated_net_force_norm_eV_per_A": (translated_net_force_norm),
                "rotation_energy_abs_eV": rotation_energy_abs,
                "rotation_force_relative": rotation_force_relative,
                "rotation_component_relative_errors": component_rotation_errors,
                "rotated_net_force_norm_eV_per_A": rotated_net_force_norm,
                "symmetry_gates": symmetry_gates,
                "directional_measurements": measurements,
                "claim_boundary": (
                    "one real water canary; not Tier-F admission, not a PES panel, "
                    "not a complete solvation free energy"
                ),
            },
            sort_keys=True,
        )
    )


def test_real_methane_194_point_electrostatic_component_comparison():
    """Record component-level negative evidence; do not score an incomplete total."""

    mol2_path = os.environ.get("MAPLE_ROUTE2_METHANE_MOL2")
    if not mol2_path or not os.path.isfile(mol2_path):
        pytest.fail(
            "The advertised real-stack methane audit requires "
            "MAPLE_ROUTE2_METHANE_MOL2 to name the pinned mobley_9055303 MOL2."
        )
    atoms = MOL2Reader(mol2_path, charge=0, mult=1)
    model, continuum, coordinates, equation, scalar = _radial_system(atoms)
    state = solve_fixed_point(
        equation,
        atoms,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=_root_context(atoms, "methane-component-audit"),
        options=FixedPointOptions(
            tolerance=1.0e-12,
            max_iterations=100,
            damping=0.7,
            history=6,
        ),
    )
    evaluated = scalar.evaluate(atoms, state.y)
    zero_field = np.zeros((len(atoms), 8))
    gas_source = coordinates.project_affine(
        np.asarray(
            model.evaluate_source(
                atoms, zero_field, need_fixed_field_forces=False
            ).source
        )
    )
    frozen_electrostatic_eV = continuum.energy(atoms, gas_source)
    kcal_mol_per_eV = 1.0 / (kcal / mol)
    electrostatic_kcal_mol = evaluated.continuum_energy * kcal_mol_per_eV
    frozen_electrostatic_kcal_mol = frozen_electrostatic_eV * kcal_mol_per_eV
    legacy_magnitude_ratio = abs(electrostatic_kcal_mol) / abs(
        _LEGACY_POINT_SOURCE_GTO_RECEIVER_ELECTROSTATIC_KCAL_MOL
    )
    local_jet_magnitude_ratio = abs(electrostatic_kcal_mol) / abs(
        _VNEXT_LOCAL_JET_ELECTROSTATIC_KCAL_MOL
    )

    assert state.actual_unmixed_residual_norm <= 1.0e-12
    assert get_solvation_profile(scalar.profile_id).enabled is False
    # The fixed point is not the cause of the magnitude collapse: it changes
    # the frozen-gas-source component by much less than the old/new gap.
    assert abs(electrostatic_kcal_mol - frozen_electrostatic_kcal_mol) <= 1.0e-3
    # Preserve the observed negative component evidence.  The legacy number is
    # a point-source/GTO-receiver profile, not a parity oracle for this operator.
    assert legacy_magnitude_ratio < 0.10
    assert local_jet_magnitude_ratio < 0.10

    print(
        "ROUTE2_REAL_METHANE_COMPONENT_AUDIT="
        + json.dumps(
            {
                "checkpoint_sha256": model.provenance.checkpoint_sha256,
                "compound_id": "mobley_9055303",
                "profile_id": scalar.profile_id,
                "scalar_id": scalar.scalar_id,
                "profile_admitted": False,
                "primal_residual": state.actual_unmixed_residual_norm,
                "iterations": len(state.iterations),
                "surface_points_per_atom": 194,
                "vacuum_energy_eV": evaluated.vacuum_energy,
                "radial_gto_electrostatic_eV": evaluated.continuum_energy,
                "radial_gto_electrostatic_kcal_mol": electrostatic_kcal_mol,
                "frozen_gas_source_electrostatic_kcal_mol": (
                    frozen_electrostatic_kcal_mol
                ),
                "root_minus_gas_source_l2": float(
                    np.linalg.norm(state.source_array() - gas_source)
                ),
                "legacy_point_source_gto_receiver_electrostatic_kcal_mol": (
                    _LEGACY_POINT_SOURCE_GTO_RECEIVER_ELECTROSTATIC_KCAL_MOL
                ),
                "legacy_electrostatic_magnitude_ratio": legacy_magnitude_ratio,
                "vnext_local_jet_electrostatic_kcal_mol": (
                    _VNEXT_LOCAL_JET_ELECTROSTATIC_KCAL_MOL
                ),
                "local_jet_electrostatic_magnitude_ratio": (local_jet_magnitude_ratio),
                "legacy_incompatible_cds_kcal_mol": (
                    _LEGACY_POINT_SOURCE_GTO_RECEIVER_CDS_KCAL_MOL
                ),
                "legacy_complete_ledger_kcal_mol": (
                    _LEGACY_POINT_SOURCE_GTO_RECEIVER_TOTAL_KCAL_MOL
                ),
                "experimental_total_kcal_mol": _FREESOLV_METHANE_TOTAL_KCAL_MOL,
                "claim_boundary": (
                    "new result is electrostatics-only; old comparison uses a "
                    "nonconjugate point source/GTO receiver and incompatible CDS; "
                    "the new component cannot be scored against the experimental total"
                ),
            },
            sort_keys=True,
        )
    )
