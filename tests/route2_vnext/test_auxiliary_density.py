from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.reference import (
    AuxiliaryDensityProjection,
    coulomb_fit_auxiliary_density,
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


def test_coulomb_fit_preserves_electron_count_and_first_moment() -> None:
    molecule, density = _h2_density()

    auxiliary, projection = coulomb_fit_auxiliary_density(
        molecule,
        density,
        moment_grid_level=2,
    )

    assert isinstance(projection, AuxiliaryDensityProjection)
    assert projection.metric_rank == auxiliary.nao_nr()
    assert projection.raw_moment_max_absolute_error > 1.0e-5
    assert projection.constrained_moment_max_absolute_error < 1.0e-12
    np.testing.assert_allclose(
        projection.constraint_matrix @ projection.constrained_coefficients,
        projection.target_moments,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert projection.target_moments[0] == pytest.approx(2.0, abs=1.0e-12)
    assert projection.raw_coefficients.flags.writeable is False
    assert projection.constrained_coefficients.flags.writeable is False


def test_coulomb_fit_rejects_density_with_wrong_shape() -> None:
    molecule, _density = _h2_density()

    with pytest.raises(ValueError, match="AO square matrix"):
        coulomb_fit_auxiliary_density(molecule, np.zeros((2, 3)))
