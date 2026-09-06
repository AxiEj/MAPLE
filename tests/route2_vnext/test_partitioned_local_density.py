from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    load_atomic_reference_density_asset,
)
from maple.solvation.coupling.stockholder_partition import (
    PromolecularStockholderPartition,
)
from maple.solvation.reference.partitioned_local_density import (
    fit_partitioned_local_density,
)


ROOT = Path(__file__).resolve().parents[2]


def _partition() -> PromolecularStockholderPartition:
    asset = load_atomic_reference_density_asset(
        table_path=ROOT / (
            "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.npz"
        ),
        manifest_path=ROOT / (
            "docs/implicit-solvation/benchmarks/"
            "route2-rhodrop-atomic-reference-gaussian-mixture-v1.json"
        ),
    )
    return PromolecularStockholderPartition(asset)


def test_partitioned_local_fit_closes_exact_global_moments() -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df

    molecule = pyscf.gto.M(
        atom="O 0 0 0; H 0.9572 0 0; H -0.2399872 0.927297 0",
        basis="def2-svp",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )
    mean_field = pyscf.scf.RHF(molecule).run()
    assert mean_field.converged

    auxiliary, fitted = fit_partitioned_local_density(
        molecule,
        mean_field.make_rdm1(),
        partition=_partition(),
        auxiliary_basis=df.aug_etb(molecule, beta=2.0),
        grid_level=2,
        batch_size=1000,
    )

    assert fitted.auxiliary_dimension == auxiliary.nao_nr()
    assert fitted.constraint_max_absolute_residual < 1.0e-10
    assert fitted.constraint_gram_condition_number < 1.0e10
    assert max(fitted.local_metric_condition_numbers) < 1.0e10
    assert float(np.sum(fitted.local_target_electron_counts)) == pytest.approx(
        fitted.grid_electron_count,
        rel=0.0,
        abs=1.0e-11,
    )
    assert fitted.grid_electron_count == pytest.approx(
        fitted.exact_electron_count,
        rel=0.0,
        abs=2.0e-5,
    )
