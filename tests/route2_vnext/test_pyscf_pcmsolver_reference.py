from __future__ import annotations

from contextlib import contextmanager
import subprocess
import sys

import numpy as np
import pytest

from maple.solvation.reference.pyscf_pcmsolver import (
    AOInverseDistanceIntegralCache,
    PCMSolverSCFSolvent,
    array_sha256,
    closed_shell_density_from_orbitals,
    response_symmetry_defect,
    solvent_energy_directional_derivative_error,
)


class _FakeMolecule:
    natm = 2

    def __init__(self) -> None:
        self._origin = np.zeros(3)

    def nao_nr(self) -> int:
        return 2

    def atom_coords(self) -> np.ndarray:
        return np.asarray([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])

    def atom_charges(self) -> np.ndarray:
        return np.asarray([1.0, 1.0])

    @contextmanager
    def with_rinv_origin(self, origin):
        previous = self._origin.copy()
        self._origin = np.asarray(origin, dtype=float)
        try:
            yield self
        finally:
            self._origin = previous

    def intor(self, name: str) -> np.ndarray:
        assert name == "int1e_rinv"
        x, y, z = self._origin
        return np.asarray(
            [
                [1.2 + 0.04 * x + 0.02 * z, 0.13 + 0.01 * y],
                [0.13 + 0.01 * y, 0.9 - 0.03 * x + 0.01 * z],
            ]
        )


class _FakeSession:
    cavity_centers_bohr = np.asarray([[1.5, 0.1, 0.2], [-1.2, 0.3, -0.4]])
    response_operator_is_symmetric = True
    operator = np.asarray([[-0.35, 0.04], [0.04, -0.22]])

    def compute_asc(self, mep, **_kwargs):
        return self.operator @ np.asarray(mep, dtype=float)

    def solve(self, mep, **_kwargs):
        values = np.asarray(mep, dtype=float)
        asc = self.compute_asc(values)
        return {
            "asc": asc,
            "polarization_energy": 0.5 * float(values @ asc),
        }


def _density() -> np.ndarray:
    return np.asarray([[0.72, 0.08], [0.08, 0.61]])


def test_reference_module_import_does_not_require_pyscf() -> None:
    code = r"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise RuntimeError("pyscf import forbidden")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import maple.solvation.reference.pyscf_pcmsolver
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "ok"


def test_closed_shell_density_is_metric_bound_and_hashed() -> None:
    coefficients = np.eye(2)
    occupations = np.asarray([2.0, 0.0])
    density, record = closed_shell_density_from_orbitals(
        coefficients, occupations, np.eye(2)
    )
    np.testing.assert_array_equal(density, np.diag([2.0, 0.0]))
    assert record["electron_count_e"] == 2.0
    assert record["density_binding_residual_e"] == 0.0
    assert record["density_sha256"] == array_sha256(density)
    with pytest.raises(ValueError, match="0/2"):
        closed_shell_density_from_orbitals(
            coefficients, np.asarray([1.0, 1.0]), np.eye(2)
        )


def test_memory_and_memmap_integral_caches_are_identical(tmp_path) -> None:
    molecule = _FakeMolecule()
    session = _FakeSession()
    memory = AOInverseDistanceIntegralCache(
        molecule, session.cavity_centers_bohr, batch_size=1
    )
    mapped = AOInverseDistanceIntegralCache(
        molecule,
        session.cavity_centers_bohr,
        path=tmp_path / "rinv.npy",
        batch_size=2,
    )
    np.testing.assert_allclose(
        memory.electronic_surface_potential(_density()),
        mapped.electronic_surface_potential(_density()),
        atol=0.0,
        rtol=0.0,
    )
    charges = np.asarray([-0.2, 0.1])
    np.testing.assert_allclose(
        memory.reaction_potential_matrix(charges),
        mapped.reaction_potential_matrix(charges),
        atol=0.0,
        rtol=0.0,
    )
    assert memory.storage_kind == "memory"
    assert mapped.storage_kind == "npy-memmap"
    assert memory.bytes == mapped.bytes == 2 * 2 * 2 * 8


def test_solvent_energy_and_reaction_operator_are_exact_derivatives() -> None:
    molecule = _FakeMolecule()
    session = _FakeSession()
    cache = AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr)
    solvent = PCMSolverSCFSolvent(molecule, session, cache)
    response = solvent.evaluate_density(_density())
    assert response.half_coupling_residual_hartree < 1.0e-15
    assert response.polarization_energy_hartree < 0.0
    assert response.total_surface_mep_hartree_per_e.flags.writeable is False
    assert response.apparent_surface_charge_e.flags.writeable is False
    assert response.reaction_potential_ao_hartree.flags.writeable is False
    with pytest.raises(ValueError):
        response.total_surface_mep_hartree_per_e[0] = 0.0
    direction = np.asarray([[0.1, -0.2], [-0.2, -0.1]])
    audit = solvent_energy_directional_derivative_error(
        solvent, density=_density(), direction=direction, step=1.0e-4
    )
    assert audit["absolute_error_hartree"] < 1.0e-11


def test_linear_response_potential_matches_finite_difference() -> None:
    molecule = _FakeMolecule()
    session = _FakeSession()
    cache = AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr)
    solvent = PCMSolverSCFSolvent(molecule, session, cache)
    direction = np.asarray([[0.13, 0.04], [0.04, -0.08]])
    step = 1.0e-5
    plus = solvent.evaluate_density(_density() + step * direction)
    minus = solvent.evaluate_density(_density() - step * direction)
    finite_difference = (
        plus.reaction_potential_ao_hartree - minus.reaction_potential_ao_hartree
    ) / (2.0 * step)
    np.testing.assert_allclose(
        solvent.response_potential(direction),
        finite_difference,
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_response_symmetry_canary_uses_the_actual_surface_operator() -> None:
    audit = response_symmetry_defect(_FakeSession(), surface_size=2)
    assert audit["absolute_defect"] < 1.0e-15
    assert audit["relative_defect"] < 1.0e-15


def test_nonsymmetric_response_is_rejected_before_scf() -> None:
    molecule = _FakeMolecule()
    session = _FakeSession()
    session.response_operator_is_symmetric = False
    cache = AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr)
    with pytest.raises(ValueError, match="MATRIXSYMM=True"):
        PCMSolverSCFSolvent(molecule, session, cache)


def test_surface_cache_identity_and_shape_are_fail_closed(tmp_path) -> None:
    molecule = _FakeMolecule()
    session = _FakeSession()
    cache = AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr)
    shifted = _FakeSession()
    shifted.cavity_centers_bohr = session.cavity_centers_bohr + 1.0e-12
    with pytest.raises(ValueError, match="surface differ"):
        PCMSolverSCFSolvent(molecule, shifted, cache)
    with pytest.raises(ValueError, match="shape"):
        AOInverseDistanceIntegralCache(molecule, np.zeros((2, 2)))
    path = tmp_path / "cache.npy"
    AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr, path=path)
    with pytest.raises(FileExistsError):
        AOInverseDistanceIntegralCache(molecule, session.cavity_centers_bohr, path=path)
