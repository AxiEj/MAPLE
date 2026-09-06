from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.reference.atomic_auxiliary_whitening import (
    build_atomic_coulomb_whitening,
)


@pytest.mark.parametrize(("symbol", "spin"), (("H", 1), ("C", 0), ("O", 0)))
def test_etb2_atomic_whitening_is_exact_and_rotation_blocked(
    symbol: str,
    spin: int,
) -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df

    molecule = pyscf.gto.M(
        atom=f"{symbol} 0 0 0",
        basis="def2-tzvpd",
        spin=spin,
        verbose=0,
    )
    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=df.aug_etb(molecule, beta=2.0),
    )

    whitening = build_atomic_coulomb_whitening(auxiliary)
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)

    assert whitening.dimension == auxiliary.nao_nr()
    assert whitening.full_identity_max_absolute_error < 1.0e-8
    assert whitening.forbidden_angular_coupling_max_absolute_value < 1.0e-10
    np.testing.assert_allclose(
        whitening.transform.T @ metric @ whitening.transform,
        np.eye(auxiliary.nao_nr()),
        rtol=0.0,
        atol=1.0e-8,
    )
    assert all(
        block.normalized_condition_number < 1.0e8 for block in whitening.blocks
    )


def test_atomic_whitening_rejects_a_multi_atom_basis() -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df

    molecule = pyscf.gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="def2-svp",
        spin=0,
        verbose=0,
    )
    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=df.aug_etb(molecule, beta=2.0),
    )

    with pytest.raises(ValueError, match="one atom"):
        build_atomic_coulomb_whitening(auxiliary)


def test_atomic_whitening_preserves_source_field_duality() -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df

    molecule = pyscf.gto.M(
        atom="O 0 0 0",
        basis="def2-tzvpd",
        spin=0,
        verbose=0,
    )
    auxiliary = df.addons.make_auxmol(
        molecule,
        auxbasis=df.aug_etb(molecule, beta=2.0),
    )
    whitening = build_atomic_coulomb_whitening(auxiliary)
    generator = np.random.default_rng(20260824)
    whitened_source = generator.normal(size=whitening.dimension)
    original_field = generator.normal(size=whitening.dimension)

    original_source = whitening.transform @ whitened_source
    whitened_field = whitening.transform.T @ original_field

    assert float(np.vdot(original_source, original_field)) == pytest.approx(
        float(np.vdot(whitened_source, whitened_field)),
        rel=2.0e-13,
        abs=2.0e-13,
    )
