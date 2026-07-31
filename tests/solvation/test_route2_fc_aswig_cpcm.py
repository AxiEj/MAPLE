from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_cpcm import (
    FixedTopologyAmplitudeSWIGCPCMResponse,
)
from maple.function.calculator.extra_correction.implicit.route2_fixed_topology_surface import (
    amplitude_switch,
)


_SIX_POINT_SPHERE = np.asarray(
    [
        [1.0, 0.0, 0.0, 1.0 / 6.0],
        [-1.0, 0.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0, 0.0, 1.0 / 6.0],
        [0.0, -1.0, 0.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0, 1.0 / 6.0],
        [0.0, 0.0, -1.0, 1.0 / 6.0],
    ],
    dtype=float,
)
_SWITCHING_CONSTANT = 4.84566077868


def _response(distance_angstrom: float) -> FixedTopologyAmplitudeSWIGCPCMResponse:
    return FixedTopologyAmplitudeSWIGCPCMResponse(
        ("H", "H"),
        np.asarray([[0.0, 0.0, 0.0], [distance_angstrom, 0.0, 0.0]]),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        _unit_sphere=_SIX_POINT_SPHERE,
        _switching_constant=_SWITCHING_CONSTANT,
    )


def test_amplitude_switch_is_compact_and_has_c3_endpoint_flatness():
    values = amplitude_switch(np.asarray([-2.0, 0.0, 0.25, 1.0, 2.0]))
    np.testing.assert_allclose(values, [0.0, 0.0, 0.070556640625, 1.0, 1.0])

    # The one-sided finite-difference derivatives at both compact endpoints
    # rapidly vanish.  This guards against accidentally reverting to sqrt of
    # PySCF's cubic switch, whose second derivative is singular at zero.
    for endpoint, direction in ((0.0, 1.0), (1.0, -1.0)):
        h = 1.0e-4
        derivative = (
            amplitude_switch(np.asarray([endpoint + direction * h]))[0]
            - amplitude_switch(np.asarray([endpoint]))[0]
        ) / h
        assert abs(derivative) < 1.0e-8

    # The mathematical polynomial is bounded.  This additionally guards the
    # ULP-scale overshoot near t=1 that previously rejected a high-order grid.
    dense = amplitude_switch(np.linspace(-1.0, 2.0, 100_003))
    assert np.all(dense >= 0.0)
    assert np.all(dense <= 1.0)


def test_fixed_topology_retains_buried_nodes_and_regularizes_their_charge():
    response = _response(1.0)
    # Two nodes are exactly buried at this crossing geometry but the candidate
    # dimension remains 2 atoms * 6 directions.
    assert response.surface_size == 12
    amplitudes = response.exposure_amplitudes
    assert np.count_nonzero(amplitudes == 0.0) >= 2
    state = response.solve_amplitude_state(np.linspace(-0.2, 0.3, 12))
    np.testing.assert_array_equal(
        state.physical_surface_charge_e[amplitudes == 0.0],
        np.zeros(np.count_nonzero(amplitudes == 0.0)),
    )
    # The amplitude variable itself is also zero because a buried candidate has
    # finite diagonal curvature and a zero right-hand side.
    np.testing.assert_array_equal(
        state.amplitude_charge_e[amplitudes == 0.0],
        np.zeros(np.count_nonzero(amplitudes == 0.0)),
    )
    eigenvalues = np.linalg.eigvalsh(response.surface_hessian)
    assert eigenvalues[0] > 0.0


def test_amplitude_cpcm_is_reciprocal_and_obeys_the_half_coupling_identity():
    response = _response(1.0)
    left = np.linspace(-0.3, 0.4, response.surface_size)
    right = np.linspace(0.25, -0.2, response.surface_size)
    q_left = response.apply_energy_conjugate(left)
    q_right = response.apply_energy_conjugate(right)
    assert float(np.dot(left, q_right)) == pytest.approx(
        float(np.dot(right, q_left)), rel=1.0e-13, abs=1.0e-13
    )
    state = response.solve(right)
    assert state.direct_surface_charge_e is not state.adjoint_surface_charge_e
    np.testing.assert_allclose(
        state.direct_surface_charge_e,
        state.adjoint_surface_charge_e,
        rtol=0.0,
        atol=0.0,
    )
    assert state.polarization_energy_hartree == pytest.approx(
        0.5 * float(np.dot(right, state.energy_conjugate_surface_charge_e)),
        rel=1.0e-13,
        abs=1.0e-13,
    )


def test_single_atom_fully_exposed_limit_is_the_unmasked_gaussian_cpcm_matrix():
    response = FixedTopologyAmplitudeSWIGCPCMResponse(
        ("H",),
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1.1]),
        dielectric=78.39,
        _unit_sphere=_SIX_POINT_SPHERE,
        _switching_constant=_SWITCHING_CONSTANT,
    )
    np.testing.assert_array_equal(response.exposure_amplitudes, np.ones(6))
    hessian = response.surface_hessian
    diagonal = np.diag(hessian)
    assert np.all(diagonal > 0.0)
    assert np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-14)
    potential = np.linspace(-0.2, 0.2, 6)
    expected = -((78.39 - 1.0) / 78.39) * np.linalg.solve(hessian, potential)
    np.testing.assert_allclose(
        response.apply_energy_conjugate(potential),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_crossing_scan_has_fixed_cardinality_and_no_inverse_step_difference_blowup():
    potential = np.linspace(-0.4, 0.7, 12)

    def energy(distance: float) -> float:
        response = _response(distance)
        assert response.surface_size == 12
        return response.solve(potential).polarization_energy_hartree

    # This scan crosses the buried/exposed transition for the axial nodes.  A
    # hard active-set removal would add an O(1/h) term to the central quotient;
    # this C3 amplitude construction instead exhibits the expected shrinking
    # differences as h is halved.
    center = 2.5
    steps = np.asarray([0.10, 0.05, 0.025, 0.0125, 0.00625])
    derivatives = np.asarray(
        [(energy(center + h) - energy(center - h)) / (2.0 * h) for h in steps]
    )
    deltas = np.abs(np.diff(derivatives))
    assert np.all(deltas[1:] < deltas[:-1])
    ratios = deltas[:-1] / deltas[1:]
    assert np.all(ratios[-2:] > 3.0)


def test_fixed_topology_response_integrates_with_the_existing_energy_only_map():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    response = _response(1.0)
    reaction_map = response.reaction_field_linear_map(positions)
    density = np.asarray(
        [[0.3, 0.02, 0.01, -0.01], [-0.3, 0.01, -0.02, 0.03]]
    )
    field = reaction_map.apply_scf(density)
    provider_energy_hartree = reaction_map.scf_polarization_energy_hartree(density)
    paired_energy_ev = 0.5 * MACE_POLAR_L1_PAIRING.pair(density, field)
    assert paired_energy_ev == pytest.approx(
        provider_energy_hartree * Hartree,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    provenance = response.runtime_provenance
    assert provenance["force_capability"] == "energy-only-no-coordinate-vjp-yet"
    assert provenance["upstream_equivalence"].startswith("new-discretization")
    assert math.isfinite(provider_energy_hartree)
