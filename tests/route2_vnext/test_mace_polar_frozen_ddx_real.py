from __future__ import annotations

import os

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.experimental.mace_polar_frozen_ddx import (
    build_water_mace_polar_frozen_ddx_pes,
)
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter

pytestmark = pytest.mark.skipif(
    os.environ.get("MAPLE_ROUTE2_REAL_MACEPOL") != "1",
    reason="set MAPLE_ROUTE2_REAL_MACEPOL=1 in the pinned real-checkpoint job",
)


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]]
        ),
        info={"charge": 0, "mult": 1},
    )


def test_real_water_pure_frozen_ddx_chain_rule_matches_the_total_scalar():
    atoms = _water()
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"),
        checkpoint_path=os.environ.get("ROUTE2_MACE_CHECKPOINT"),
    )
    # A smaller smooth CDS grid keeps this derivative canary focused.  It is
    # not an accuracy artifact and cannot be mixed with the production grid.
    pes = build_water_mace_polar_frozen_ddx_pes(
        model, atoms.get_chemical_symbols(), cds_grid_points=302
    )
    state = pes.solve(atoms)
    evaluated = pes.evaluate_forces(atoms, central_state=state)
    direction = np.random.default_rng(20260816).normal(size=(3, 3))
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = -float(np.vdot(evaluated.total_forces_eV_per_A, direction))
    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        finite_difference = (
            pes.get_potential_energy(plus) - pes.get_potential_energy(minus)
        ) / (2.0 * step)
        errors.append(abs(finite_difference - analytic))
    assert min(errors) < 2.0e-6
    assert state.source_total_charge_e == pytest.approx(0.0, abs=2.0e-12)
    np.testing.assert_allclose(
        np.sum(evaluated.total_forces_eV_per_A, axis=0), 0.0, atol=3.0e-11
    )
