from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import gto, scf
from pyscf.dft import numint

from maple.solvation.reference.pyscf_density_levelset import (
    PySCFAODensityLevelSet,
)


def _hydrogen_level_set(
    *,
    n_iso_e_per_bohr3: float = 1.0e-3,
    checkpoint_sha256: str = "1" * 64,
) -> PySCFAODensityLevelSet:
    molecule = gto.M(
        atom="H 0 0 -0.35; H 0 0 0.35",
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1.0e-12)
    return PySCFAODensityLevelSet(
        molecule,
        mean_field.mo_coeff,
        mean_field.mo_occ,
        n_iso_e_per_bohr3=n_iso_e_per_bohr3,
        checkpoint_sha256=checkpoint_sha256,
    )


def test_qm_density_value_matches_pyscf_reference() -> None:
    level_set = _hydrogen_level_set()
    points = np.asarray([[2.1, -0.3, 0.2], [-1.7, 0.4, -0.1]])
    density, _, _, _ = level_set.evaluate_density(points)
    ao = numint.eval_ao(level_set.molecule, points)
    reference = numint.eval_rho(level_set.molecule, ao, level_set.density_matrix)
    np.testing.assert_allclose(density, reference, rtol=2.0e-13, atol=2.0e-14)


def test_qm_density_spatial_jets_match_central_differences() -> None:
    level_set = _hydrogen_level_set()
    point = np.asarray([[2.1, -0.3, 0.2]])
    _, gradient, hessian, third = level_set.evaluate_density(point)
    step = 2.0e-4

    gradient_fd = np.empty(3)
    hessian_fd = np.empty((3, 3))
    third_fd = np.empty((3, 3, 3))
    for axis in range(3):
        displacement = np.zeros_like(point)
        displacement[0, axis] = step
        plus = level_set.evaluate_density(point + displacement)
        minus = level_set.evaluate_density(point - displacement)
        gradient_fd[axis] = (plus[0][0] - minus[0][0]) / (2.0 * step)
        hessian_fd[:, axis] = (plus[1][0] - minus[1][0]) / (2.0 * step)
        third_fd[:, :, axis] = (plus[2][0] - minus[2][0]) / (2.0 * step)

    np.testing.assert_allclose(gradient[0], gradient_fd, rtol=2.0e-7, atol=2.0e-10)
    np.testing.assert_allclose(hessian[0], hessian_fd, rtol=2.0e-7, atol=2.0e-10)
    np.testing.assert_allclose(third[0], third_fd, rtol=3.0e-6, atol=2.0e-9)


def test_qm_density_state_identity_binds_isovalue_and_checkpoint() -> None:
    first = _hydrogen_level_set()
    second = _hydrogen_level_set(
        n_iso_e_per_bohr3=2.0e-3,
        checkpoint_sha256="2" * 64,
    )
    assert first.state_sha256 != second.state_sha256


def test_qm_density_response_derivatives_fail_closed() -> None:
    level_set = _hydrogen_level_set()
    with pytest.raises(NotImplementedError, match="source JVP"):
        level_set.source_jvp(np.zeros((1, 3)), np.zeros((2, 4)))
    with pytest.raises(NotImplementedError, match="source VJP"):
        level_set.source_vjp(np.zeros((1, 3)), object())
    with pytest.raises(NotImplementedError, match="nuclear response"):
        level_set.nuclear_vjp(np.zeros((1, 3)), object())
