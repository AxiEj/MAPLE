from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms
from scipy.integrate import quad

from maple.function.dispatcher.solvfe.association_conditioning import (
    SoftMembershipSurfaceRestraint,
)
from maple.function.dispatcher.solvfe.membership import SoftCutoffMembership
from maple.function.dispatcher.solvfe.packing_conditioning import (
    GeometryConditionedCavity,
)


def _membership() -> SoftCutoffMembership:
    return SoftCutoffMembership(
        vdw_radii_angstrom={6: 1.5},
        lambda_s_angstrom=1.0,
        softness_angstrom=0.1,
        surface_smoothing_angstrom=0.05,
        shell_boundary_id="smooth-union-solute-vdw-surface-v3",
    )


def _restraint() -> SoftMembershipSurfaceRestraint:
    return SoftMembershipSurfaceRestraint(
        solute_indices=(0,),
        water_oxygen_indices=(1,),
        water_atom_indices=(1, 2, 3),
        membership=_membership(),
        temperature_k=298.15,
        conditioning_measure_id="fixed-solute-soft-membership-v3",
        solute_measure_hash="1" * 64,
        solute_atom_map_hash="2" * 64,
    )


def _system(distance: float) -> Atoms:
    return Atoms(
        "COH2",
        positions=[
            [0.0, 0.0, 0.0],
            [distance, 0.0, 0.0],
            [distance + 0.96, 0.0, 0.0],
            [distance - 0.24, 0.93, 0.0],
        ],
    )


def test_association_field_is_exact_soft_membership_weight():
    restraint = _restraint()
    # d_surface = 2.5 - 1.5 = lambda_s, hence b = 1/2.
    details = restraint.evaluate_details(_system(2.5))

    assert details.memberships == pytest.approx([0.5])
    assert details.log_member_weight == pytest.approx(-math.log(2.0))
    assert details.energy_ev == pytest.approx(
        -details.log_member_weight
        / restraint.beta_ev_inverse
    )
    np.testing.assert_allclose(
        details.forces_ev_per_angstrom.sum(axis=0),
        0.0,
        atol=1.0e-12,
    )
    assert details.forces_ev_per_angstrom[1, 0] < 0.0
    energy, forces = restraint.evaluate(_system(2.5))
    assert energy == details.energy_ev
    np.testing.assert_allclose(forces, details.forces_ev_per_angstrom)


@pytest.mark.parametrize("atom_index", [0, 1])
def test_association_field_force_matches_finite_difference(atom_index):
    restraint = _restraint()
    atoms = _system(2.5)
    step = 1.0e-6
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[atom_index, 0] += step
    minus.positions[atom_index, 0] -= step
    finite_difference_force = -(
        restraint.evaluate(plus)[0] - restraint.evaluate(minus)[0]
    ) / (2.0 * step)

    analytic = restraint.evaluate(atoms)[1][atom_index, 0]

    assert analytic == pytest.approx(finite_difference_force, abs=1.0e-8)


def test_association_field_force_is_smooth_at_two_center_seam():
    restraint = SoftMembershipSurfaceRestraint(
        solute_indices=(0, 1),
        water_oxygen_indices=(2,),
        water_atom_indices=(2, 3, 4),
        membership=_membership(),
        temperature_k=298.15,
        conditioning_measure_id="fixed-solute-soft-membership-v3",
        solute_measure_hash="1" * 64,
        solute_atom_map_hash="2" * 64,
    )
    atoms = Atoms(
        "CCOH2",
        positions=[
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    step = 1.0e-6

    center = restraint.evaluate_details(atoms)
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[2, 0] += step
    minus.positions[2, 0] -= step
    plus_details = restraint.evaluate_details(plus)
    minus_details = restraint.evaluate_details(minus)
    finite_difference = -(
        plus_details.energy_ev - minus_details.energy_ev
    ) / (2.0 * step)

    assert center.forces_ev_per_angstrom[2, 0] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert center.forces_ev_per_angstrom[2, 0] == pytest.approx(
        finite_difference,
        abs=1.0e-10,
    )
    assert plus_details.forces_ev_per_angstrom[2, 0] == pytest.approx(
        -minus_details.forces_ev_per_angstrom[2, 0],
        abs=1.0e-12,
    )
    assert abs(plus_details.forces_ev_per_angstrom[2, 0]) < 1.0e-4
    np.testing.assert_allclose(
        center.forces_ev_per_angstrom.sum(axis=0),
        0.0,
        atol=1.0e-12,
    )


def test_association_and_packing_can_bind_same_membership_hash():
    restraint = _restraint()
    cavity = GeometryConditionedCavity(
        solute_indices=(0,),
        membership=_membership(),
        temperature_k=298.15,
        conditioning_measure_id="product-soft-packing-v3",
        solute_measure_hash="1" * 64,
        solute_atom_map_hash="2" * 64,
        active_occupancy_max=1,
    )

    assert restraint.membership_definition_hash == _membership().content_hash
    assert restraint.membership_surface_hash == cavity.membership_surface_hash
    assert restraint.observation_volume_hash == cavity.observation_volume_hash
    assert restraint.boundary_adapter_hash != cavity.boundary_adapter_hash
    assert restraint.content_hash
    assert restraint.observation_volume_hash


def test_soft_effective_volume_matches_radial_quadrature():
    restraint = _restraint()
    estimate = restraint.effective_translational_volume(
        _system(2.5),
        sobol_power=15,
        replicates=4,
        seed=20260725,
        tail_log_tolerance=32.0,
    )
    hard_radius = 1.5 + 1.0
    softness = 0.1
    analytic = 4.0 * math.pi * quad(
        lambda radius: (
            radius**2
            / (1.0 + math.exp((radius - hard_radius) / softness))
        ),
        0.0,
        hard_radius + 40.0 * softness,
        epsabs=1.0e-10,
    )[0]

    assert estimate.volume_angstrom3 == pytest.approx(
        analytic,
        rel=3.0e-3,
    )
    assert estimate.tail_upper_bound_angstrom3 < 1.0e-9
    assert estimate.samples_per_replicate == 2**15


def test_effective_volume_box_bounds_smooth_union_tail():
    restraint = SoftMembershipSurfaceRestraint(
        solute_indices=(0, 1),
        water_oxygen_indices=(2,),
        water_atom_indices=(2, 3, 4),
        membership=_membership(),
        temperature_k=298.15,
        conditioning_measure_id="fixed-solute-soft-membership-v3",
        solute_measure_hash="1" * 64,
        solute_atom_map_hash="2" * 64,
    )
    atoms = Atoms(
        "CCOH2",
        positions=[
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    tail_log_tolerance = 4.0

    estimate = restraint.effective_translational_volume(
        atoms,
        sobol_power=4,
        replicates=2,
        tail_log_tolerance=tail_log_tolerance,
    )

    surface_shift = (
        _membership().surface_smoothing_angstrom * math.log(2.0)
    )
    reference_radius = 1.5 + 1.0 + surface_shift
    tail_extent = tail_log_tolerance * 0.1
    lower, upper = estimate.integration_box_angstrom
    assert lower[0] == pytest.approx(-2.0 - reference_radius - tail_extent)
    assert upper[0] == pytest.approx(2.0 + reference_radius + tail_extent)


def test_association_rejects_periodic_cluster_measure():
    atoms = _system(2.5)
    atoms.set_cell(np.eye(3) * 20.0)
    atoms.set_pbc(True)

    with pytest.raises(ValueError, match="nonperiodic"):
        _restraint().evaluate(atoms)
