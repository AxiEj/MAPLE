from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.route2_v0_frozen_density_embedding import (
    Route2V0FrozenDensityPauliOverlap,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_molecular_external_potential import (
    Route2V0FrozenMaceGaussianSource,
    Route2V0MolecularConfigurations,
    Route2V0MolecularSolventReference,
    V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION,
    V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY,
    V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE,
    evaluate_route2_v0_molecular_external_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_promolecular_density import (
    Route2V0PromolecularDensityTable,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_structured_solvent import (
    RegularCartesianGrid,
)


def _table() -> Route2V0PromolecularDensityTable:
    radial_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    return Route2V0PromolecularDensityTable(
        radial_grid_bohr=radial_grid,
        densities_by_atomic_number={
            1: np.array([1.0, 0.7, 0.3, 0.08, 0.0]),
            8: np.array([3.0, 1.4, 0.5, 0.12, 0.0]),
        },
        table_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )


def _grid(shift_bohr: np.ndarray | None = None) -> RegularCartesianGrid:
    origin = np.array([-3.0, -3.0, -3.0])
    if shift_bohr is not None:
        origin = origin + shift_bohr
    return RegularCartesianGrid(
        origin_bohr=origin,
        spacing_bohr=np.ones(3),
        shape=(7, 7, 7),
    )


def _solvent(
    *,
    reference_positions_bohr: np.ndarray | None = None,
) -> Route2V0MolecularSolventReference:
    return Route2V0MolecularSolventReference(
        atomic_numbers=np.array([1, 8]),
        site_charges_e=np.array([0.4, -0.4]),
        reference_positions_bohr=(
            np.array([[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0]])
            if reference_positions_bohr is None
            else reference_positions_bohr
        ),
        provenance_label="synthetic neutral H--O molecular reference for scalar tests",
    )


def _configurations(
    *,
    translations_bohr: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> Route2V0MolecularConfigurations:
    return Route2V0MolecularConfigurations(
        translations_bohr=(
            np.array([[0.2, 0.1, -0.1], [6.0, 0.0, 0.0]])
            if translations_bohr is None
            else translations_bohr
        ),
        rotations=(
            np.repeat(np.eye(3)[None], 2, axis=0) if rotations is None else rotations
        ),
    )


def _mace_source(
    *,
    positions_angstrom: np.ndarray | None = None,
) -> Route2V0FrozenMaceGaussianSource:
    return Route2V0FrozenMaceGaussianSource(
        density_coefficients=np.array([[0.5, 0.0, 0.0, 0.0]]),
        atom_positions_angstrom=(
            np.zeros((1, 3)) if positions_angstrom is None else positions_angstrom
        ),
        target_total_charge_e=0.5,
    )


def _evaluate(
    *,
    grid: RegularCartesianGrid | None = None,
    solute_positions_angstrom: np.ndarray | None = None,
    mace_source: Route2V0FrozenMaceGaussianSource | None = None,
    solvent: Route2V0MolecularSolventReference | None = None,
    configurations: Route2V0MolecularConfigurations | None = None,
):
    positions = (
        np.zeros((1, 3))
        if solute_positions_angstrom is None
        else solute_positions_angstrom
    )
    return evaluate_route2_v0_molecular_external_potential(
        integration_grid=_grid() if grid is None else grid,
        promolecular_table=_table(),
        solute_atomic_numbers=np.array([1]),
        solute_atom_positions_angstrom=positions,
        mace_source=_mace_source() if mace_source is None else mace_source,
        solvent=_solvent() if solvent is None else solvent,
        configurations=_configurations() if configurations is None else configurations,
    )


def test_molecular_external_potential_is_one_pauli_plus_one_mace_electrostatic_scalar():
    state = _evaluate()
    grid = state.integration_grid
    table = state.promolecular_table
    sites = state.configurations.site_positions_bohr(state.solvent)
    direct_solvent_density = table.evaluate(
        grid.points_bohr(),
        state.solvent.atomic_numbers,
        sites[0] * Bohr,
    ).reshape(grid.shape)
    direct_pauli = Route2V0FrozenDensityPauliOverlap(
        grid=grid,
        solute_electron_density_e_per_bohr3=state.solute_reference_density_e_per_bohr3,
        solvent_electron_density_e_per_bohr3=direct_solvent_density,
    ).nonadditive_kinetic_energy_hartree()
    direct_electrostatic = np.dot(
        state.solvent.site_charges_e,
        state.mace_source.potential_hartree_per_e(sites[0]),
    )

    assert state.construction == V0_MOLECULAR_EXTERNAL_POTENTIAL_CONSTRUCTION
    assert state.interaction_scope == V0_MOLECULAR_EXTERNAL_POTENTIAL_SCOPE
    assert (
        state.coordinate_derivative_policy
        == V0_MOLECULAR_EXTERNAL_POTENTIAL_COORDINATE_POLICY
    )
    assert state.pauli_repulsion_hartree[0] == pytest.approx(direct_pauli)
    assert state.pauli_repulsion_hartree[0] >= 0.0
    assert state.pauli_repulsion_hartree[1] == pytest.approx(0.0, abs=1.0e-15)
    assert state.electrostatic_energy_hartree[0] == pytest.approx(direct_electrostatic)
    np.testing.assert_allclose(
        state.external_potential_hartree,
        state.pauli_repulsion_hartree + state.electrostatic_energy_hartree,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert not state.external_potential_hartree.flags.writeable


def test_molecular_external_potential_is_covariant_under_joint_translation():
    original = _evaluate()
    shift_bohr = np.array([0.35, -0.55, 0.25])
    shifted_positions_angstrom = np.array([shift_bohr * Bohr])
    shifted = _evaluate(
        grid=_grid(shift_bohr),
        solute_positions_angstrom=shifted_positions_angstrom,
        mace_source=_mace_source(positions_angstrom=shifted_positions_angstrom),
        configurations=_configurations(
            translations_bohr=original.configurations.translations_bohr + shift_bohr
        ),
    )

    np.testing.assert_allclose(
        shifted.pauli_repulsion_hartree,
        original.pauli_repulsion_hartree,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        shifted.electrostatic_energy_hartree,
        original.electrostatic_energy_hartree,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_molecular_external_potential_is_invariant_to_molecular_reference_origin():
    original = _evaluate()
    reference_shift = np.array([0.2, -0.1, 0.15])
    rotations = np.array(
        [
            np.eye(3),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    configurations = _configurations(rotations=rotations)
    original_oriented = _evaluate(configurations=configurations)
    shifted_reference = _solvent(
        reference_positions_bohr=original_oriented.solvent.reference_positions_bohr
        + reference_shift
    )
    shifted_configurations = _configurations(
        translations_bohr=configurations.translations_bohr
        - np.einsum("nij,j->ni", rotations, reference_shift),
        rotations=rotations,
    )
    shifted = _evaluate(
        solvent=shifted_reference,
        configurations=shifted_configurations,
    )

    np.testing.assert_allclose(
        shifted.external_potential_hartree,
        original_oriented.external_potential_hartree,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_molecular_external_potential_rejects_invalid_source_or_geometry_shortcuts():
    with pytest.raises(ValueError, match="molecular total charge"):
        Route2V0MolecularSolventReference(
            atomic_numbers=np.array([1]),
            site_charges_e=np.array([0.2]),
            reference_positions_bohr=np.zeros((1, 3)),
            provenance_label="invalid charged neutral solvent",
        )
    with pytest.raises(ValueError, match="proper orthogonal"):
        Route2V0MolecularConfigurations(
            translations_bohr=np.zeros((1, 3)),
            rotations=np.array([np.diag([1.0, 1.0, -1.0])]),
        )
    with pytest.raises(ValueError, match="total-charge constraint"):
        Route2V0FrozenMaceGaussianSource(
            density_coefficients=np.array([[0.5, 0.0, 0.0, 0.0]]),
            atom_positions_angstrom=np.zeros((1, 3)),
        )
    unsupported = Route2V0MolecularSolventReference(
        atomic_numbers=np.array([17]),
        site_charges_e=np.array([0.0]),
        reference_positions_bohr=np.zeros((1, 3)),
        provenance_label="unsupported atom test",
    )
    with pytest.raises(ValueError, match="no frozen reference"):
        _evaluate(solvent=unsupported)


def test_molecular_external_potential_result_rejects_a_source_from_another_geometry():
    state = _evaluate()
    mismatched_source = _mace_source(positions_angstrom=np.array([[0.1, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="source geometry"):
        replace(state, mace_source=mismatched_source)
