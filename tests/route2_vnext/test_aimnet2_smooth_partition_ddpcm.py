from __future__ import annotations

import math

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1,
)
from maple.solvation.continuum.aimnet2_smooth_partition_ddpcm import (
    AIMNET2_SMOOTH_PARTITION_DOUBLE_LAYER_RADIAL_ORDER,
    AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
    AIMNET2_SMOOTH_PARTITION_PARTITION_RADIAL_ORDER,
    AIMNET2_SMOOTH_PARTITION_SOURCE_RADIAL_ORDER,
    AIMNET2_SMOOTH_PARTITION_SURFACE_LMAX,
    AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2,
    build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate,
)
from maple.solvation.continuum.atomic_l1_pyddx import AtomicL1PyDDXPCMBackend
from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SMOOTH_PARTITION_HARMONIC_DDPCM_PROVIDER_ID,
    SMOOTH_PARTITION_HARMONIC_DDPCM_SCALAR_ID,
)


def _candidate(symbols: tuple[str, ...], *, solvent: str = "water"):
    torch = pytest.importorskip("torch")
    return build_aimnet2_frozen_charge_smooth_partition_ddpcm_candidate(
        symbols,
        solvent=solvent,
        dtype=torch.float64,
        device="cpu",
    )


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
    )


def test_candidate_has_distinct_aimnet2_multisolvent_identity_and_no_admission():
    water = _candidate(("O", "H", "H"), solvent="water")
    methanol = _candidate(("O", "H", "H"), solvent="methanol")

    assert water.provider_id != SMOOTH_PARTITION_HARMONIC_DDPCM_PROVIDER_ID
    assert water.scalar_id != SMOOTH_PARTITION_HARMONIC_DDPCM_SCALAR_ID
    assert water.scalar_id == (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1
    )
    assert water.solvent == "water"
    assert methanol.solvent == "methanol"
    assert water.dielectric == pytest.approx(78.355)
    assert methanol.dielectric == pytest.approx(32.613)
    assert water.configuration_sha256() != methanol.configuration_sha256()
    assert len(water.configuration_sha256()) == 64
    assert len(water.provenance_sha256) == 64
    assert water.capabilities.enabled_tiers == ()
    assert water.reciprocal is True
    assert water.laboratory_fixed_surface_grid is False
    assert water.continuum_field_supplied_to_aimnet2 is False
    assert water.electronic_scf_iteration is False
    assert water.aimnet2_source_evaluation == "one-shot-per-geometry"
    assert water.surface_lmax == AIMNET2_SMOOTH_PARTITION_SURFACE_LMAX
    assert water.partition_lmax == AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX
    assert (
        water.partition_radial_quadrature_order
        == AIMNET2_SMOOTH_PARTITION_PARTITION_RADIAL_ORDER
    )
    assert (
        water.source_radial_quadrature_order
        == AIMNET2_SMOOTH_PARTITION_SOURCE_RADIAL_ORDER
    )
    assert (
        water.double_layer_radial_quadrature_order
        == AIMNET2_SMOOTH_PARTITION_DOUBLE_LAYER_RADIAL_ORDER
    )
    assert water.transition_width_angstrom2 == pytest.approx(
        AIMNET2_SMOOTH_PARTITION_TRANSITION_WIDTH_ANGSTROM2
    )
    provenance = dict(water.runtime_provenance())
    assert provenance["aimnet2_source_evaluation"] == "one-shot-per-geometry"
    assert provenance["continuum_field_supplied_to_aimnet2"] is False
    assert provenance["electronic_scf_iteration"] is False
    assert provenance["active_coefficient_deletion"] is False
    with pytest.raises(AttributeError):
        water._solvent = "methanol"


def test_water_point_monopoles_match_high_resolution_pyddx_ddpcm():
    pytest.importorskip("pyddx")
    atoms = _water()
    source = np.asarray(
        [[-0.70, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0]]
    )
    candidate = _candidate(tuple(atoms.get_chemical_symbols()))
    reference = AtomicL1PyDDXPCMBackend(
        atoms,
        candidate.radii_angstrom,
        dielectric=candidate.dielectric,
        lmax=15,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
    )
    candidate_energy = candidate.energy_eV(atoms, source)
    reference_energy = reference.energy(atoms, source)

    assert abs(candidate_energy - reference_energy) < 6.0e-4


def test_rigid_motion_and_coordinate_derivative_share_the_same_scalar():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.13, -0.08]])
    source = np.asarray([[0.30, 0.0, 0.0, 0.0], [-0.30, 0.0, 0.0, 0.0]])
    candidate = _candidate(("H", "H"))

    axis = np.asarray([0.3, -0.4, 0.8])
    axis /= np.linalg.norm(axis)
    angle = 0.71
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T + np.asarray([1.2, -0.5, 0.7])
    energy = candidate.energy_eV(atoms, source)
    moved_energy = candidate.energy_eV(moved, source)
    gradient = candidate.coordinate_partial(atoms, source)

    direction = np.asarray([[0.2, -0.3, 0.1], [-0.2, 0.3, -0.1]])
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    finite_difference = (
        candidate.energy_eV(plus, source) - candidate.energy_eV(minus, source)
    ) / (2.0 * step)
    analytic = float(np.vdot(gradient, direction))

    assert moved_energy == pytest.approx(energy, rel=0.0, abs=2.0e-12)
    assert np.allclose(np.sum(gradient, axis=0), 0.0, rtol=0.0, atol=2.0e-12)
    assert analytic == pytest.approx(finite_difference, rel=0.0, abs=2.0e-7)
    topology = candidate.topology_state(atoms)
    assert topology["coefficient_count"] == 2 * (candidate.surface_lmax + 1) ** 2
    assert topology["active_coefficient_deletion"] is False
    assert topology["laboratory_fixed_surface_grid"] is False


def test_candidate_rejects_non_monopole_source_components():
    atoms = _water()
    source = np.asarray(
        [[-0.70, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0]]
    )
    source[1, 2] = 1.0e-15

    with pytest.raises(ValueError, match="point monopoles only"):
        _candidate(tuple(atoms.get_chemical_symbols())).energy_eV(atoms, source)
