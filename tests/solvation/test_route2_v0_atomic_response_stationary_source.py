from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_independent_particle_response as atomic_ip,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_response_stationary_source as stationary_source,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_response_kernel as response_kernel,
)

ROOT = Path(__file__).resolve().parents[2]


def _response() -> atomic_ip.Route2V0AtomicIndependentParticleResponse:
    coefficients = np.eye(4)
    energies = np.asarray([-1.0, 0.0, 0.0, 0.0])
    occupations = np.asarray([2.0, 0.0, 0.0, 0.0])
    position = np.zeros((3, 4, 4))
    for axis in range(3):
        position[axis, 0, axis + 1] = 0.5
        position[axis, axis + 1, 0] = 0.5
    return atomic_ip.build_route2_v0_atomic_independent_particle_response(
        atomic_number=1,
        symbol="H",
        basis="synthetic",
        spin_2s=0,
        mo_coefficients=coefficients,
        orbital_energies_hartree=energies,
        orbital_occupations=occupations,
        overlap_matrix=np.eye(4),
        position_integrals_ebohr=position,
    )


def _table() -> atomic_ip.Route2V0AtomicIndependentParticleResponseTable:
    return atomic_ip.Route2V0AtomicIndependentParticleResponseTable(
        responses_by_atomic_number={1: _response()},
        table_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )


def _completion() -> response_kernel.Route2V0ResponseKernelCompletion:
    baseline = _table().assemble(np.asarray([1, 1]))
    return response_kernel.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=np.vstack(
            (0.5 * np.eye(3), 0.5 * np.eye(3))
        ),
        molecular_polarizability_bohr3=2.0 * np.eye(3),
        charge_constraint_vector=None,
    )


def _solve(dual: np.ndarray):
    return stationary_source.solve_route2_v0_atomic_response_stationary_state(
        completion=_completion(),
        reference_coefficient_dual_hartree=dual,
    )


def test_atomic_response_stationary_source_has_one_support_constrained_scalar():
    completion = _completion()
    dual = np.asarray([0.1, -0.3, 0.2, 0.4, -0.7, 0.9])
    state = _solve(dual)

    assert (
        state.construction
        == stationary_source.V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION
    )
    np.testing.assert_allclose(
        state.permanent_response_coefficients,
        -completion.response_covariance_coefficient_dual @ dual,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        completion.electronic_curvature_coefficient_dual
        @ state.permanent_response_coefficients
        + state.projected_reference_coefficient_dual_hartree,
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        completion.response_support_constraints @ state.permanent_response_coefficients,
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert state.stationarity_residual_inf_hartree < 1.0e-14
    assert state.support_constraint_residual_inf < 1.0e-14
    assert state.support_minimum_curvature_hartree > 0.0
    assert state.stationary_energy_identity_error_hartree < 1.0e-14


def test_atomic_response_stationary_scalar_obeys_the_dual_envelope_identity():
    dual = np.asarray([0.1, -0.3, 0.2, 0.4, -0.7, 0.9])
    state = _solve(dual)
    direction = np.asarray([-0.2, 0.3, 0.1, -0.4, 0.2, 0.6])
    step = 1.0e-6
    plus = _solve(dual + step * direction)
    minus = _solve(dual - step * direction)

    finite_difference = (
        plus.electronic_correction_energy_hartree
        - minus.electronic_correction_energy_hartree
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(
        float(state.permanent_response_coefficients @ direction),
        abs=1.0e-9,
    )


def test_atomic_response_stationary_source_rejects_a_wrong_dual_dimension():
    with pytest.raises(ValueError, match="shape"):
        _solve(np.ones(5))


def test_permanent_density_audit_retains_the_ao_carrier_spectrum_as_a_diagnostic():
    audit = stationary_source.Route2V0AtomicResponsePermanentDensityAudit(
        density_matrix=np.eye(2),
        electron_count_e=2.0,
        reference_electron_count_e=2.0,
        electron_count_error_e=0.0,
        minimum_ao_metric_density_eigenvalue=-0.2,
        maximum_ao_metric_density_eigenvalue=2.2,
        density_symmetry_error=0.0,
    )

    assert audit.minimum_ao_metric_density_eigenvalue == pytest.approx(-0.2)
    assert audit.maximum_ao_metric_density_eigenvalue == pytest.approx(2.2)


@pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None,
    reason="The repository's default test environment intentionally omits PySCF.",
)
def test_pyscf_atomic_reference_keeps_atomic_metric_count_and_same_source_dual():
    table = atomic_ip.load_route2_v0_atomic_independent_particle_response_table(
        table_path=(
            ROOT / "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.npz"
        ),
        manifest_path=(
            ROOT / "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1.json"
        ),
    )
    baseline = table.assemble(np.asarray([1, 1]))
    atom_sum = np.hstack((np.eye(3), np.eye(3)))
    molecular_polarizability = (
        atom_sum
        @ baseline.atom_dipole_map_coefficient_to_ebohr
        @ baseline.baseline_response_covariance_coefficient_dual
        @ baseline.atom_dipole_map_coefficient_to_ebohr.T
        @ atom_sum.T
    )
    completion = response_kernel.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=np.vstack(
            (0.5 * np.eye(3), 0.5 * np.eye(3))
        ),
        molecular_polarizability_bohr3=molecular_polarizability,
        charge_constraint_vector=None,
    )
    reference = stationary_source.build_route2_v0_atomic_response_pyscf_reference(
        response_table=table,
        atomic_numbers=np.asarray([1, 1]),
        atom_positions_angstrom=np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
    )
    state = stationary_source.solve_route2_v0_atomic_response_stationary_state(
        completion=completion,
        reference_coefficient_dual_hartree=reference.reference_coefficient_dual_hartree(),
    )
    audit = reference.permanent_density_audit(state)
    density_grid = reference.density_positivity_grid_bohr()
    density = reference.electron_number_density_e_per_bohr3(
        audit,
        density_grid[:5],
    )
    potential = reference.total_permanent_potential_hartree_per_e(
        audit,
        np.asarray([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
    )

    assert np.max(reference.atomic_mo_metric_errors) < 1.0e-12
    assert audit.electron_count_error_e < 1.0e-12
    assert density_grid.shape[1] == 3
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(potential))
