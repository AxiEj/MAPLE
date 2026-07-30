from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import (
    route2_v0_atomic_independent_particle_response as atomic_ip,
)
from maple.function.calculator.extra_correction.implicit import (
    route2_v0_response_kernel as response_kernel,
)


def _response(atomic_number: int = 1) -> atomic_ip.Route2V0AtomicIndependentParticleResponse:
    coefficient_count = 4
    coefficients = np.eye(coefficient_count)
    energies = np.asarray([-1.0, 0.0, 0.0, 0.0])
    occupations = np.asarray([2.0, 0.0, 0.0, 0.0])
    overlap = np.eye(coefficient_count)
    position = np.zeros((3, coefficient_count, coefficient_count))
    for axis in range(3):
        position[axis, 0, axis + 1] = 0.5
        position[axis, axis + 1, 0] = 0.5
    return atomic_ip.build_route2_v0_atomic_independent_particle_response(
        atomic_number=atomic_number,
        symbol="H" if atomic_number == 1 else "He",
        basis="synthetic",
        spin_2s=0,
        mo_coefficients=coefficients,
        orbital_energies_hartree=energies,
        orbital_occupations=occupations,
        overlap_matrix=overlap,
        position_integrals_ebohr=position,
    )


def test_atomic_independent_particle_response_is_neutral_psd_and_isotropic():
    response = _response()

    assert response.transition_count == 3
    np.testing.assert_allclose(
        response.transition_weights_hartree_inverse,
        np.ones(3),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response.transition_charges_e,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response.atomic_polarizability_bohr3,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response.atom_dipole_map_coefficient_to_ebohr,
        -np.eye(3),
        rtol=0.0,
        atol=1.0e-12,
    )
    transitions = response.transition_density_matrices()
    assert transitions.shape == (3, 4, 4)
    np.testing.assert_allclose(transitions, transitions.transpose(0, 2, 1))


def test_atomic_independent_particle_table_direct_sum_completes_mace_moments():
    response_h = _response(1)
    response_he = _response(2)
    table = atomic_ip.Route2V0AtomicIndependentParticleResponseTable(
        responses_by_atomic_number={1: response_h, 2: response_he},
        table_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    baseline = table.assemble(np.asarray([1, 2]))
    partition = np.vstack((0.4 * np.eye(3), 0.6 * np.eye(3)))
    polarizability = np.asarray(
        [[2.0, 0.1, 0.0], [0.1, 1.5, 0.2], [0.0, 0.2, 1.2]]
    )
    state = response_kernel.complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline.baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=(
            baseline.atom_dipole_map_coefficient_to_ebohr
        ),
        atomic_dipole_partition_molecular_to_ebohr=partition,
        molecular_polarizability_bohr3=polarizability,
    )

    assert state.charge_constraint_vector is None
    np.testing.assert_allclose(
        baseline.transition_charges_e,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    atom_sum = np.hstack((np.eye(3), np.eye(3)))
    np.testing.assert_allclose(
        atom_sum
        @ baseline.atom_dipole_map_coefficient_to_ebohr
        @ state.response_covariance_coefficient_dual
        @ baseline.atom_dipole_map_coefficient_to_ebohr.T
        @ atom_sum.T,
        polarizability,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert state.completed_minimum_eigenvalue >= -1.0e-12


def test_atomic_independent_particle_response_rejects_nonorthonormal_or_nonneutral_source():
    coefficients = np.eye(4)
    coefficients[0, 0] = 2.0
    with pytest.raises(ValueError, match="overlap-orthonormal"):
        atomic_ip.build_route2_v0_atomic_independent_particle_response(
            atomic_number=1,
            symbol="H",
            basis="synthetic",
            spin_2s=0,
            mo_coefficients=coefficients,
            orbital_energies_hartree=np.asarray([-1.0, 0.0, 0.0, 0.0]),
            orbital_occupations=np.asarray([2.0, 0.0, 0.0, 0.0]),
            overlap_matrix=np.eye(4),
            position_integrals_ebohr=np.zeros((3, 4, 4)),
        )

    response = _response()
    with pytest.raises(ValueError, match="must be neutral"):
        atomic_ip.Route2V0AtomicIndependentParticleResponse(
            atomic_number=response.atomic_number,
            symbol=response.symbol,
            basis=response.basis,
            spin_2s=response.spin_2s,
            mo_coefficients=response.mo_coefficients,
            orbital_energies_hartree=response.orbital_energies_hartree,
            orbital_occupations=response.orbital_occupations,
            transition_lower_indices=response.transition_lower_indices,
            transition_upper_indices=response.transition_upper_indices,
            transition_weights_hartree_inverse=response.transition_weights_hartree_inverse,
            transition_dipoles_ebohr=response.transition_dipoles_ebohr,
            transition_charges_e=np.asarray([1.0e-4, 0.0, 0.0]),
        )


def test_atomic_independent_particle_table_loader_checks_every_array_hash(tmp_path):
    response = _response()
    table_path = tmp_path / "atomic-ip.npz"
    np.savez(
        table_path,
        atomic_numbers=np.asarray([1], dtype=np.int64),
        mo_coefficients_Z1=response.mo_coefficients,
        orbital_energies_hartree_Z1=response.orbital_energies_hartree,
        orbital_occupations_Z1=response.orbital_occupations,
        transition_lower_indices_Z1=response.transition_lower_indices,
        transition_upper_indices_Z1=response.transition_upper_indices,
        transition_weights_hartree_inverse_Z1=response.transition_weights_hartree_inverse,
        transition_dipoles_ebohr_Z1=response.transition_dipoles_ebohr,
        transition_charges_e_Z1=response.transition_charges_e,
    )

    def sha256(path):
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    def array_hash(values):
        array = np.ascontiguousarray(np.asarray(values))
        return hashlib.sha256(array.view(np.uint8)).hexdigest()

    manifest_path = tmp_path / "atomic-ip.json"
    manifest = {
        "artifact": atomic_ip.V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_ARTIFACT,
        "status": "pass",
        "table": {"sha256": sha256(table_path)},
        "generation_contract": {
            "basis": "synthetic",
            "numerical_relative_tolerance": 1.0e-10,
        },
        "results": [
            {
                "atomic_number": 1,
                "symbol": "H",
                "spin_2s": 0,
                "array_sha256": {
                    "mo_coefficients": array_hash(response.mo_coefficients),
                    "orbital_energies_hartree": array_hash(
                        response.orbital_energies_hartree
                    ),
                    "orbital_occupations": array_hash(response.orbital_occupations),
                    "transition_lower_indices": array_hash(
                        response.transition_lower_indices
                    ),
                    "transition_upper_indices": array_hash(
                        response.transition_upper_indices
                    ),
                    "transition_weights_hartree_inverse": array_hash(
                        response.transition_weights_hartree_inverse
                    ),
                    "transition_dipoles_ebohr": array_hash(
                        response.transition_dipoles_ebohr
                    ),
                    "transition_charges_e": array_hash(response.transition_charges_e),
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    table = atomic_ip.load_route2_v0_atomic_independent_particle_response_table(
        table_path=table_path,
        manifest_path=manifest_path,
    )
    assert table.supported_atomic_numbers == (1,)

    manifest["results"][0]["array_sha256"]["transition_charges_e"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree with manifest"):
        atomic_ip.load_route2_v0_atomic_independent_particle_response_table(
            table_path=table_path,
            manifest_path=manifest_path,
        )
