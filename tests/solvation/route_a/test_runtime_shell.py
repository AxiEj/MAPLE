from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB

from maple.function.dispatcher.solvfe.shell import (
    FlatBottomSurfaceRestraint,
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


def _restraint() -> FlatBottomSurfaceRestraint:
    return FlatBottomSurfaceRestraint(
        solute_indices=(0,),
        water_oxygen_indices=(1,),
        vdw_radii_angstrom={6: 1.70},
        lambda_s_angstrom=1.50,
        force_constant_ev_per_angstrom2=2.0,
        shell_boundary_id="toy-shell",
        measure_id="nonperiodic-solute-com-reduced-v1",
    )


def test_flat_bottom_surface_wall_is_zero_inside_the_shell():
    energy, forces = _restraint().evaluate(_system(3.0))

    assert energy == 0.0
    assert np.array_equal(forces, np.zeros((4, 3)))


def test_flat_bottom_surface_wall_force_matches_finite_difference():
    restraint = _restraint()
    atoms = _system(3.5)
    energy, forces = restraint.evaluate(atoms)
    step = 1.0e-6
    plus = atoms.copy()
    plus.positions[1, 0] += step
    minus = atoms.copy()
    minus.positions[1, 0] -= step
    finite_difference = (
        restraint.evaluate(plus)[0] - restraint.evaluate(minus)[0]
    ) / (2.0 * step)

    # d_s = 3.5 - 1.7 = 1.8, so the 0.3 A excess costs 0.09 eV.
    assert energy == pytest.approx(0.09)
    assert forces[1, 0] == pytest.approx(-finite_difference, abs=1.0e-9)
    assert forces[0, 0] == pytest.approx(-forces[1, 0])
    assert np.sum(forces, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-12)


def test_protocol_wall_parameters_and_hash_are_deterministic():
    first = FlatBottomSurfaceRestraint.from_protocol(
        solute_indices=(0,),
        water_oxygen_indices=(1,),
        vdw_radii_angstrom={6: 1.70},
        lambda_s_angstrom=1.50,
        shell_boundary_id="toy-shell",
        measure_id="nonperiodic-solute-com-reduced-v1",
        temperature_k=298.15,
        buffer_height_kbt=10.0,
        buffer_width_angstrom=0.5,
    )
    second = FlatBottomSurfaceRestraint.from_protocol(
        solute_indices=(0,),
        water_oxygen_indices=(1,),
        vdw_radii_angstrom={6: 1.70},
        lambda_s_angstrom=1.50,
        shell_boundary_id="toy-shell",
        measure_id="nonperiodic-solute-com-reduced-v1",
        temperature_k=298.15,
        buffer_height_kbt=10.0,
        buffer_width_angstrom=0.5,
    )

    assert first.force_constant_ev_per_angstrom2 > 2.0
    assert first.content_hash == second.content_hash


def test_sobol_effective_volume_recovers_single_sphere_soft_tail():
    restraint = _restraint()
    atoms = _system(3.0)
    beta = 1.0 / (kB * 298.15)
    estimate = restraint.effective_translational_volume(
        atoms,
        beta_ev_inverse=beta,
        sobol_power=15,
        replicates=4,
        seed=20260725,
    )
    flat_radius = 1.70 + 1.50
    alpha = 0.5 * beta * 2.0
    analytic = 4.0 * math.pi * (
        flat_radius**3 / 3.0
        + flat_radius**2 * math.sqrt(math.pi) / (2.0 * math.sqrt(alpha))
        + flat_radius / alpha
        + math.sqrt(math.pi) / (4.0 * alpha**1.5)
    )

    assert estimate.volume_angstrom3 == pytest.approx(
        analytic,
        rel=2.0e-3,
    )
    assert estimate.tail_upper_bound_angstrom3 < 1.0e-30
    assert estimate.samples_per_replicate == 2**15
