from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.source import (
    PointChargeL0Source,
    solve_fixed_charge_continuum,
)


class _ReciprocalReactionField:
    def __init__(self, scale=-0.25, energy_offset_hartree=0.0):
        self.scale = float(scale)
        self.energy_offset_hartree = float(energy_offset_hartree)

    def apply_scf(self, multipoles):
        values = np.asarray(multipoles, dtype=float)
        field = np.zeros_like(values)
        field[:, 0] = self.scale * values[:, 0]
        return field

    def scf_polarization_energy_hartree(self, multipoles):
        values = np.asarray(multipoles, dtype=float)
        coupling_ev = self.scale * float(np.dot(values[:, 0], values[:, 0]))
        return 0.5 * coupling_ev / Hartree + self.energy_offset_hartree


def test_point_charge_source_embeds_only_the_monopole_channel():
    source = PointChargeL0Source(
        charges_e=np.asarray([-0.8, 0.4, 0.4]),
        declared_total_charge_e=0.0,
        source_model="aimnet2-nqe",
    )

    multipoles = source.continuum_multipole_coefficients

    assert multipoles.shape == (3, 4)
    np.testing.assert_allclose(multipoles[:, 0], source.charges_e)
    np.testing.assert_array_equal(multipoles[:, 1:], 0.0)
    assert multipoles.flags.writeable is False
    assert source.provenance == {
        "solute_source": "point-charge-l0",
        "source_model": "aimnet2-nqe",
        "polarization_response": "fixed",
        "declared_total_charge_e": 0.0,
    }


def test_point_charge_surface_potential_matches_direct_coulomb_sum():
    positions_angstrom = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    points_bohr = np.asarray(
        [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
    )
    source = PointChargeL0Source(
        charges_e=np.asarray([-0.3, 0.3]),
        declared_total_charge_e=0.0,
        source_model="aimnet2-nqe",
    )

    actual = source.surface_potential_hartree_per_e(
        points_bohr,
        positions_angstrom,
    )
    positions_bohr = positions_angstrom / Bohr
    expected = sum(
        charge / np.linalg.norm(points_bohr - center, axis=1)
        for charge, center in zip(
            source.charges_e,
            positions_bohr,
            strict=True,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)


def test_fixed_charge_continuum_state_enforces_the_half_coupling_identity():
    source = PointChargeL0Source(
        charges_e=np.asarray([-0.8, 0.4, 0.4]),
        declared_total_charge_e=0.0,
        source_model="aimnet2-nqe",
    )

    state = solve_fixed_charge_continuum(
        _ReciprocalReactionField(),
        source,
    )

    assert state.solute_source == "point-charge-l0"
    assert state.polarization_response == "fixed"
    assert state.energy_identity_error_ev == pytest.approx(0.0, abs=1.0e-15)
    assert state.reaction_field_values_ev.flags.writeable is False
    assert state.continuum_multipole_coefficients.flags.writeable is False


def test_fixed_charge_continuum_state_rejects_an_inconsistent_energy_ledger():
    source = PointChargeL0Source(
        charges_e=np.asarray([-0.8, 0.4, 0.4]),
        declared_total_charge_e=0.0,
        source_model="aimnet2-nqe",
    )

    with pytest.raises(RuntimeError, match="half-coupling identity"):
        solve_fixed_charge_continuum(
            _ReciprocalReactionField(energy_offset_hartree=1.0e-4),
            source,
            energy_identity_tolerance_ev=1.0e-8,
        )
