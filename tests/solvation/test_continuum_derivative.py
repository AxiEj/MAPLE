from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_derivative import (
    EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION,
    continuum_operator_position_vjp,
    polarization_operator_position_gradient,
)
from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    SurfaceChargeState,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    FixedCavityPCMReactionFieldLinearMap,
)


class _CoordinateDependentIEFPCMResponse:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    operator_derivative_contract_version = (
        EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
    )
    energy_response_is_reciprocal = True

    def __init__(self, coordinate_angstrom: float):
        self.coordinate_angstrom = float(coordinate_angstrom)
        self._atomic_numbers = np.asarray([6.0, 8.0])
        self._reference_positions_bohr = np.asarray(
            [[-1.0, 0.0, 0.0], [1.1, 0.2, -0.1]]
        )
        self._cavity_radii_angstrom = np.asarray([1.85, 1.70])
        self._surface_points_bohr = np.asarray(
            [
                [-3.0, 0.1, 0.2],
                [2.8, 0.4, -0.2],
                [0.2, -2.7, 0.8],
            ]
        )
        self._surface_areas_bohr2 = np.asarray([0.7, 0.8, 0.9])
        self._k0 = np.asarray(
            [
                [1.50, 0.08, -0.03],
                [-0.02, 1.35, 0.06],
                [0.04, -0.01, 1.20],
            ]
        )
        self._dk = np.asarray(
            [
                [0.025, -0.006, 0.003],
                [0.004, -0.018, 0.005],
                [-0.002, 0.003, 0.014],
            ]
        )
        self._r0 = np.asarray(
            [
                [-0.42, 0.05, -0.01],
                [0.02, -0.37, 0.04],
                [-0.03, 0.01, -0.31],
            ]
        )
        self._dr = np.asarray(
            [
                [-0.012, 0.004, -0.002],
                [0.003, 0.009, -0.001],
                [0.002, -0.003, -0.007],
            ]
        )

    @property
    def atomic_numbers(self):
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self):
        return self._reference_positions_bohr.copy()

    @property
    def cavity_radii_angstrom(self):
        return self._cavity_radii_angstrom.copy()

    @property
    def surface_points_bohr(self):
        return self._surface_points_bohr.copy()

    @property
    def surface_areas_bohr2(self):
        return self._surface_areas_bohr2.copy()

    @property
    def _k(self):
        return self._k0 + self.coordinate_angstrom * self._dk

    @property
    def _r(self):
        return self._r0 + self.coordinate_angstrom * self._dr

    @property
    def _direct_operator(self):
        return np.linalg.solve(self._k, self._r)

    @property
    def _energy_conjugate_operator(self):
        direct = self._direct_operator
        return 0.5 * (direct + direct.T)

    def apply_energy_conjugate(self, surface_potential_hartree_per_e):
        potential = np.asarray(surface_potential_hartree_per_e, dtype=float)
        return self._energy_conjugate_operator @ potential

    def solve(self, surface_potential_hartree_per_e):
        potential = np.asarray(surface_potential_hartree_per_e, dtype=float)
        direct = self._direct_operator @ potential
        adjoint = self._direct_operator.T @ potential
        conjugate = 0.5 * (direct + adjoint)
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=direct,
            adjoint_surface_charge_e=adjoint,
            energy_conjugate_surface_charge_e=conjugate,
            polarization_energy_hartree=0.5
            * float(np.dot(potential, conjugate)),
        )

    def operator_position_vjp(
        self,
        left_surface_potential_hartree_per_e,
        right_surface_potential_hartree_per_e,
    ):
        left = np.asarray(
            left_surface_potential_hartree_per_e,
            dtype=float,
        )
        right = np.asarray(
            right_surface_potential_hartree_per_e,
            dtype=float,
        )
        direct = self._direct_operator
        direct_derivative = np.linalg.solve(
            self._k,
            self._dr - self._dk @ direct,
        )
        conjugate_derivative = 0.5 * (
            direct_derivative + direct_derivative.T
        )
        result = np.zeros((self._atomic_numbers.size, 3), dtype=float)
        result[0, 0] = float(left @ conjugate_derivative @ right)
        return result


def test_polarization_operator_gradient_matches_nonsymmetric_iefpcm_fd():
    coordinate = 0.17
    potential = np.asarray([0.31, -0.22, 0.09])
    response = _CoordinateDependentIEFPCMResponse(coordinate)
    state = response.solve(potential)

    analytic = polarization_operator_position_gradient(response, state)

    def energy_at(value):
        displaced = _CoordinateDependentIEFPCMResponse(value)
        return displaced.solve(potential).polarization_energy_hartree

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            energy_at(coordinate + step) - energy_at(coordinate - step)
        ) / (2.0 * step)
        assert analytic[0, 0] == pytest.approx(
            finite_difference,
            rel=2.0e-10,
            abs=2.0e-12,
        )
    np.testing.assert_allclose(
        analytic[1:, :],
        np.zeros((1, 3)),
        rtol=0.0,
        atol=0.0,
    )


def test_operator_bilinear_vjp_is_symmetric_and_matches_fd():
    coordinate = -0.11
    left = np.asarray([-0.14, 0.27, 0.06])
    right = np.asarray([0.23, -0.08, 0.19])
    response = _CoordinateDependentIEFPCMResponse(coordinate)

    analytic = continuum_operator_position_vjp(
        response,
        left,
        right,
    )
    swapped = continuum_operator_position_vjp(
        response,
        right,
        left,
    )
    np.testing.assert_allclose(analytic, swapped, rtol=1.0e-13, atol=1.0e-14)

    def pairing_at(value):
        displaced = _CoordinateDependentIEFPCMResponse(value)
        return float(
            left @ displaced._energy_conjugate_operator @ right
        )

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            pairing_at(coordinate + step) - pairing_at(coordinate - step)
        ) / (2.0 * step)
        assert analytic[0, 0] == pytest.approx(
            finite_difference,
            rel=2.0e-10,
            abs=2.0e-12,
        )


def test_fixed_cavity_map_operator_vjp_matches_field_pairing_fd():
    coordinate = 0.07
    response = _CoordinateDependentIEFPCMResponse(coordinate)
    positions_angstrom = (
        response.reference_positions_bohr * Bohr
    )
    density = np.asarray(
        [[0.16, -0.05, 0.02, 0.04], [-0.16, 0.03, -0.06, 0.01]]
    )
    field_cotangent = np.asarray(
        [[0.21, -0.04, 0.08, 0.03], [-0.09, 0.02, 0.05, -0.07]]
    )
    reaction_field = FixedCavityPCMReactionFieldLinearMap(
        response,
        positions_angstrom,
        geometry_tolerance_angstrom=1.0e-10,
    )

    analytic = reaction_field.continuum_operator_position_vjp(
        density,
        field_cotangent,
    )

    def pairing_at(value):
        displaced_response = _CoordinateDependentIEFPCMResponse(value)
        displaced_map = FixedCavityPCMReactionFieldLinearMap(
            displaced_response,
            positions_angstrom,
            geometry_tolerance_angstrom=1.0e-10,
        )
        return float(
            np.vdot(
                field_cotangent,
                displaced_map.apply(density),
            )
        )

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            pairing_at(coordinate + step) - pairing_at(coordinate - step)
        ) / (2.0 * step)
        assert analytic[0, 0] == pytest.approx(
            finite_difference,
            rel=3.0e-10,
            abs=3.0e-11,
        )


def test_operator_gradient_fails_closed_without_derivative_contract():
    response = SimpleNamespace(
        contract_version=EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
        energy_response_is_reciprocal=True,
        atomic_numbers=np.asarray([1.0]),
        surface_points_bohr=np.asarray([[2.0, 0.0, 0.0]]),
    )

    with pytest.raises(NotImplementedError, match="operator derivative"):
        continuum_operator_position_vjp(
            response,
            np.asarray([0.2]),
            np.asarray([0.2]),
        )


def test_operator_gradient_rejects_bad_version_or_output():
    response = _CoordinateDependentIEFPCMResponse(0.0)
    response.operator_derivative_contract_version = 2
    with pytest.raises(ValueError, match="contract version"):
        continuum_operator_position_vjp(
            response,
            np.asarray([0.2, -0.1, 0.3]),
            np.asarray([0.2, -0.1, 0.3]),
        )

    response.operator_derivative_contract_version = (
        EXTERNAL_MEP_OPERATOR_DERIVATIVE_CONTRACT_VERSION
    )
    response.operator_position_vjp = lambda left, right: np.zeros((1, 3))
    with pytest.raises(ValueError, match="finite with shape"):
        continuum_operator_position_vjp(
            response,
            np.asarray([0.2, -0.1, 0.3]),
            np.asarray([0.2, -0.1, 0.3]),
        )
