from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_displacement_frozen_kkt as frozen_kkt,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_displacement_kkt as reduced_kkt,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_displacement_response as atomic_displacement_response,
)
from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)


ROOT = Path(__file__).resolve().parents[2]


class _LinearReciprocalContinuum:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._atomic_numbers = np.ones(self.atom_count, dtype=int)
        self._positions_bohr = (
            np.asarray(positions_angstrom, dtype=float)
            / atomic_displacement_response.BOHR_ANGSTROM
        )
        self._surface_points_bohr = np.asarray(surface_points_bohr, dtype=float)
        self._response_matrix = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers.copy()

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return self._positions_bohr.copy()

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return self._surface_points_bohr.copy()

    def apply_energy_conjugate(self, potential: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(potential, dtype=float)


class _AffineContinuum(_LinearReciprocalContinuum):
    """Deliberately violates linearity while preserving the reduced curvature."""

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        super().__init__(positions_angstrom, surface_points_bohr, response_matrix)
        self._offset = np.asarray(offset, dtype=float)

    def apply_energy_conjugate(self, potential: np.ndarray) -> np.ndarray:
        return super().apply_energy_conjugate(potential) + self._offset


def _table() -> atomic_displacement_response.Route2V0AtomicDisplacementResponseTable:
    return atomic_displacement_response.Route2V0AtomicDisplacementResponseTable(
        radial_grid_bohr=np.asarray([0.0, 0.5, 2.0, 8.0]),
        enclosed_electrons_by_atomic_number={
            1: np.asarray([0.0, 0.3, 0.9, 1.0]),
        },
        table_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.9, -0.2, 0.4]])
    numbers = np.asarray([1, 1])
    weights = np.stack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    points = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [-2.5, 1.5, 0.7],
            [1.1, -2.7, 1.8],
            [-1.2, -0.8, -2.5],
        ]
    )
    return positions, numbers, weights, points


def _coupling() -> frozen_kkt.AtomicDisplacementDirectSumSurfaceCoupling:
    positions, numbers, weights, points = _geometry()
    permanent = frozen_kkt.PointMultipoleSurfaceCoupling(
        atom_positions_angstrom=positions,
        surface_points_bohr=points,
    )
    induced = reduced_kkt.AtomicDisplacementSurfaceCoupling(
        response_table=_table(),
        atomic_numbers=numbers,
        atom_positions_angstrom=positions,
        atomic_dipole_weights=weights,
        surface_points_bohr=points,
    )
    return frozen_kkt.AtomicDisplacementDirectSumSurfaceCoupling(
        permanent=permanent,
        induced=induced,
    )


def _density() -> np.ndarray:
    return np.asarray(
        [[0.17, 0.11, -0.02, 0.03], [-0.17, -0.04, 0.08, -0.01]],
        dtype=float,
    )


def _continuum() -> _LinearReciprocalContinuum:
    positions, _, _, points = _geometry()
    return _LinearReciprocalContinuum(
        positions,
        points,
        response_matrix=-0.03 * np.eye(len(points)),
    )


def test_direct_sum_uses_exact_point_and_adt_transposes():
    coupling = _coupling()
    density = _density()
    dipole = np.asarray([0.6, -0.3, 0.2])
    charge = np.asarray([0.1, -0.2, 0.4, -0.1, 0.3, -0.2])
    positions, _, _, points = _geometry()

    np.testing.assert_allclose(
        coupling.permanent.surface_potential(density),
        point_multipole_potential(points, positions, density),
        rtol=0.0,
        atol=1.0e-15,
    )
    density_dual, dipole_dual = coupling.surface_to_dual(charge)
    assert float(coupling.surface_potential(density, dipole) @ charge) == pytest.approx(
        float(density.reshape(-1) @ density_dual.reshape(-1) + dipole @ dipole_dual),
        abs=1.0e-14,
    )


def test_frozen_kkt_preregistration_locks_its_no_projection_boundary():
    protocol = json.loads(
        (
            ROOT
            / "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-displacement-frozen-kkt-prereg-v1.json"
        ).read_text(encoding="utf-8")
    )
    theory = (
        ROOT
        / "docs/implicit-solvation/"
        "ROUTE2_V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_THEORY.md"
    ).read_text(encoding="utf-8")

    assert protocol["status"] == (
        "structural-kernel-implemented-before-physical-continuum-binding"
    )
    assert protocol["construction"]["name"] == (
        "route2-v0-atomic-displacement-frozen-kkt-v1"
    )
    hard_constraints = protocol["hard_constraints"]
    assert hard_constraints["post_training"] is False
    assert hard_constraints["fine_tuning"] is False
    assert hard_constraints["experimental_solvation_fit"] is False
    assert hard_constraints["gto_projection_of_atomic_displacement_source"] is False
    assert hard_constraints["physical_continuum_execution"] is False
    forbidden = " ".join(protocol["forbidden_shortcuts"])
    assert "FreeSolv" in forbidden
    assert "MNSol" in forbidden
    assert "GTO radial channel" in forbidden
    assert "distributional source space" in theory
    assert "full stationary electronic density functional" in theory


def test_frozen_kkt_stationarizes_adt_without_updating_the_permanent_source():
    coupling = _coupling()
    state = frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
        coupling=coupling,
        continuum=_continuum(),
        frozen_density_coefficients=_density(),
        polarizability_bohr3=2.0 * np.eye(3),
    )

    assert np.linalg.norm(state.induced_dipole_ebohr) > 1.0e-8
    assert state.direct_sum_duality_error_hartree < 1.0e-14
    assert state.continuum_direct_sum_antisymmetry_norm < 1.0e-14
    assert (
        state.continuum_direct_sum_antisymmetry_norm
        <= state.continuum_direct_sum_reciprocity_tolerance
    )
    assert state.continuum_linearity_error_e < 1.0e-14
    assert state.stationarity_residual_inf < 1.0e-12
    np.testing.assert_allclose(
        state.total_surface_charge_e,
        state.frozen_surface_charge_e + state.induced_surface_charge_e,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert state.stationary_total_energy_hartree == pytest.approx(
        state.electronic_induction_energy_hartree
        + state.total_continuum_energy_hartree,
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        state.external_dipole_response_bohr3,
        state.external_dipole_response_bohr3.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.max(np.linalg.eigvalsh(state.external_dipole_response_bohr3)) <= 1.0e-12


def test_frozen_source_scaling_obeys_the_stationary_envelope_identity():
    coupling = _coupling()
    continuum = _continuum()
    density = _density()
    base = frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
        coupling=coupling,
        continuum=continuum,
        frozen_density_coefficients=density,
        polarizability_bohr3=2.0 * np.eye(3),
    )
    step = 1.0e-6
    plus = frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
        coupling=coupling,
        continuum=continuum,
        frozen_density_coefficients=(1.0 + step) * density,
        polarizability_bohr3=2.0 * np.eye(3),
    )
    minus = frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
        coupling=coupling,
        continuum=continuum,
        frozen_density_coefficients=(1.0 - step) * density,
        polarizability_bohr3=2.0 * np.eye(3),
    )

    finite_difference = (
        plus.stationary_total_energy_hartree - minus.stationary_total_energy_hartree
    ) / (2.0 * step)
    analytic = float(
        base.frozen_surface_potential_hartree_per_e @ base.total_surface_charge_e
    )
    assert finite_difference == pytest.approx(analytic, rel=2.0e-8, abs=2.0e-10)


def test_direct_sum_rejects_crossed_permanent_and_induced_surfaces():
    positions, numbers, weights, points = _geometry()
    permanent = frozen_kkt.PointMultipoleSurfaceCoupling(
        atom_positions_angstrom=positions,
        surface_points_bohr=points + np.asarray([0.0, 0.0, 1.0e-7]),
    )
    induced = reduced_kkt.AtomicDisplacementSurfaceCoupling(
        response_table=_table(),
        atomic_numbers=numbers,
        atom_positions_angstrom=positions,
        atomic_dipole_weights=weights,
        surface_points_bohr=points,
    )

    with pytest.raises(ValueError, match="share one geometry and surface"):
        frozen_kkt.AtomicDisplacementDirectSumSurfaceCoupling(
            permanent=permanent,
            induced=induced,
        )


def test_frozen_kkt_rejects_charge_drift_without_projecting_the_source():
    density = _density()
    density[0, 0] += 1.0e-6

    with pytest.raises(ValueError, match="violates its total-charge constraint"):
        frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
            coupling=_coupling(),
            continuum=_continuum(),
            frozen_density_coefficients=density,
            polarizability_bohr3=2.0 * np.eye(3),
        )


def test_frozen_kkt_rejects_an_affine_response_even_if_the_reduced_hessian_is_ok():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    # Construct an offset in null(B.T), so the reduced Hessian/reciprocity gate
    # cannot see it. The full direct-sum linearity gate must still reject it.
    _, _, right_vectors = np.linalg.svd(
        coupling.induced.surface_operator_hartree_per_e_per_ebohr.T,
        full_matrices=True,
    )
    offset = 0.01 * right_vectors[-1]
    continuum = _AffineContinuum(
        positions,
        points,
        response_matrix=-0.03 * np.eye(len(points)),
        offset=offset,
    )

    with pytest.raises(RuntimeError, match="not linear"):
        frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
            coupling=coupling,
            continuum=continuum,
            frozen_density_coefficients=_density(),
            polarizability_bohr3=2.0 * np.eye(3),
        )


def test_frozen_kkt_rejects_cross_source_nonreciprocity_hidden_from_induction():
    coupling = _coupling()
    positions, _, _, points = _geometry()
    induced_operator = (
        coupling.induced.surface_operator_hartree_per_e_per_ebohr
    )
    _, singular_values, right_vectors = np.linalg.svd(
        induced_operator.T,
        full_matrices=True,
    )
    rank = int(np.count_nonzero(singular_values > 1.0e-12))
    null_vectors = right_vectors[rank:]
    permanent_operator = (
        coupling.permanent.surface_operator_hartree_per_e_per_coefficient
    )
    scores = np.asarray(
        [np.linalg.norm(vector @ permanent_operator) for vector in null_vectors]
    )
    null_vector = null_vectors[int(np.argmax(scores))]
    assert float(np.max(scores)) > 1.0e-8
    induced_vector = induced_operator[:, 0]
    antisymmetric_part = 0.01 * (
        np.outer(null_vector, induced_vector)
        - np.outer(induced_vector, null_vector)
    )
    continuum = _LinearReciprocalContinuum(
        positions,
        points,
        response_matrix=-0.03 * np.eye(len(points)) + antisymmetric_part,
    )

    with pytest.raises(
        RuntimeError,
        match="not reciprocal in the direct-sum source space",
    ):
        frozen_kkt.solve_route2_v0_atomic_displacement_frozen_kkt(
            coupling=coupling,
            continuum=continuum,
            frozen_density_coefficients=_density(),
            polarizability_bohr3=2.0 * np.eye(3),
        )
