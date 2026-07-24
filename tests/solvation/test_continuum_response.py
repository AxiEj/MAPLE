from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    PCMSolverExternalMEPCavityResponse,
    SurfaceChargeState,
)


class _FakePCMSolverSession:
    def __init__(self, *, symmetric: bool = True, energy_offset: float = 0.0):
        self.atomic_numbers = np.asarray([6.0, 8.0])
        self.coordinates_bohr = np.asarray(
            [[-1.2, 0.1, 0.0], [1.0, -0.2, 0.3]]
        )
        self.cavity_centers_bohr = np.asarray(
            [
                [-3.0, 0.0, 0.2],
                [2.7, 0.4, -0.1],
                [0.1, -2.8, 0.7],
            ]
        )
        self.cavity_areas_bohr2 = np.asarray([0.7, 0.8, 0.9])
        self.response_operator_is_symmetric = symmetric
        self._response_matrix = np.asarray(
            [
                [-0.31, 0.04, -0.02],
                [0.04, -0.27, 0.03],
                [-0.02, 0.03, -0.22],
            ]
        )
        self._energy_offset = float(energy_offset)

    def compute_asc(self, mep: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(mep, dtype=float)

    def solve(self, mep: np.ndarray) -> dict[str, np.ndarray | float]:
        potential = np.asarray(mep, dtype=float)
        asc = self.compute_asc(potential)
        return {
            "asc": asc,
            "polarization_energy": (
                0.5 * float(np.dot(potential, asc)) + self._energy_offset
            ),
        }


def test_surface_charge_state_supports_nonsymmetric_iefpcm_energy_pairing():
    potential = np.asarray([0.4, -0.2, 0.1])
    direct = np.asarray([-0.12, 0.03, -0.01])
    adjoint = np.asarray([-0.10, 0.01, -0.03])
    conjugate = 0.5 * (direct + adjoint)

    state = SurfaceChargeState(
        surface_potential_hartree_per_e=potential,
        direct_surface_charge_e=direct,
        adjoint_surface_charge_e=adjoint,
        energy_conjugate_surface_charge_e=conjugate,
        polarization_energy_hartree=0.5 * float(np.dot(potential, conjugate)),
    )

    np.testing.assert_allclose(
        state.energy_conjugate_surface_charge_e,
        0.5
        * (
            state.direct_surface_charge_e
            + state.adjoint_surface_charge_e
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert state.polarization_energy_hartree == pytest.approx(
        0.5
        * float(
            np.dot(
                state.surface_potential_hartree_per_e,
                state.energy_conjugate_surface_charge_e,
            )
        ),
        abs=1.0e-15,
    )


def test_surface_charge_state_rejects_inconsistent_conjugate_or_energy():
    potential = np.asarray([0.4, -0.2])
    direct = np.asarray([-0.12, 0.03])
    adjoint = np.asarray([-0.10, 0.01])

    with pytest.raises(ValueError, match="arithmetic mean"):
        SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=direct,
            adjoint_surface_charge_e=adjoint,
            energy_conjugate_surface_charge_e=direct,
            polarization_energy_hartree=0.5 * float(np.dot(potential, direct)),
        )

    conjugate = 0.5 * (direct + adjoint)
    with pytest.raises(ValueError, match=r"0\.5\*dot"):
        SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=direct,
            adjoint_surface_charge_e=adjoint,
            energy_conjugate_surface_charge_e=conjugate,
            polarization_energy_hartree=1.0,
        )


def test_pcmsolver_external_mep_response_locks_per_atom_cavity_and_energy():
    session = _FakePCMSolverSession()
    response = PCMSolverExternalMEPCavityResponse(
        session,
        cavity_radii_angstrom=np.asarray([1.85, 1.70]),
    )
    potential = np.asarray([0.4, -0.2, 0.1])

    state = response.solve(potential)

    assert response.contract_version == EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    np.testing.assert_allclose(response.atomic_numbers, session.atomic_numbers)
    np.testing.assert_allclose(
        response.reference_positions_bohr,
        session.coordinates_bohr,
    )
    np.testing.assert_allclose(response.cavity_radii_angstrom, [1.85, 1.70])
    np.testing.assert_allclose(
        response.surface_points_bohr,
        session.cavity_centers_bohr,
    )
    np.testing.assert_allclose(
        response.surface_areas_bohr2,
        session.cavity_areas_bohr2,
    )
    np.testing.assert_allclose(
        response.apply_energy_conjugate(potential),
        session.compute_asc(potential),
    )
    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        state.adjoint_surface_charge_e,
    )
    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        state.energy_conjugate_surface_charge_e,
    )


def test_pcmsolver_external_mep_response_fails_closed_on_contract_violations():
    with pytest.raises(ValueError, match="MATRIXSYMM=TRUE"):
        PCMSolverExternalMEPCavityResponse(
            _FakePCMSolverSession(symmetric=False),
            cavity_radii_angstrom=np.asarray([1.85, 1.70]),
        )

    with pytest.raises(ValueError, match="cavity_radii_angstrom"):
        PCMSolverExternalMEPCavityResponse(
            _FakePCMSolverSession(),
            cavity_radii_angstrom=np.asarray([1.85]),
        )

    response = PCMSolverExternalMEPCavityResponse(
        _FakePCMSolverSession(energy_offset=1.0e-4),
        cavity_radii_angstrom=np.asarray([1.85, 1.70]),
    )
    with pytest.raises(ValueError, match=r"0\.5\*dot"):
        response.solve(np.asarray([0.4, -0.2, 0.1]))
