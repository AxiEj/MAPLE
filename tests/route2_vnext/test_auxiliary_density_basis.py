from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.reference.auxiliary_density_basis import (
    coulomb_fit_auxiliary_density_in_basis,
)


def _h2_density():
    pyscf = pytest.importorskip("pyscf")
    molecule = pyscf.gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="def2-svp",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = pyscf.scf.RHF(molecule).run()
    assert mean_field.converged
    return molecule, np.asarray(mean_field.make_rdm1(), dtype=np.float64)


def test_explicit_even_tempered_basis_preserves_density_moments() -> None:
    molecule, density = _h2_density()
    from pyscf import df

    basis = df.aug_etb(molecule, beta=2.0)

    auxiliary, projection = coulomb_fit_auxiliary_density_in_basis(
        molecule,
        density,
        auxiliary_basis=basis,
        moment_grid_level=2,
    )

    assert auxiliary.nao_nr() == projection.constrained_coefficients.size
    assert projection.constraint_rank == 4
    assert projection.normalized_restricted_minimum_eigenvalue > 0.0
    assert projection.normalized_restricted_condition_number >= 1.0
    assert projection.constrained_moment_max_absolute_error < 1.0e-12
    assert projection.projected_stationarity_max_absolute_error < 1.0e-10
    np.testing.assert_allclose(
        projection.constraint_matrix @ projection.constrained_coefficients,
        projection.target_moments,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_explicit_auxiliary_basis_is_required() -> None:
    molecule, density = _h2_density()

    with pytest.raises(ValueError, match="nonempty element mapping"):
        coulomb_fit_auxiliary_density_in_basis(
            molecule,
            density,
            auxiliary_basis={},
        )
