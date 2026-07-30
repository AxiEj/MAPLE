"""Frozen MACE-MDP induced-moment maps in the Route-2 GTO pairing.

This module has one deliberately narrow role: given an already-audited
atom-resolved partition of a *molecular* induced dipole, express that neutral
induced moment in the one-radial ``l<=1`` Gaussian coefficient basis used by
the fixed Route-2 GTO Galerkin primitive.  It neither creates an electronic
energy nor selects a radial width.  The caller must explicitly provide the
width and independently validate the resulting source before coupling it to a
continuum.

The MACE raw ``l=1`` convention is retained exactly.  Coefficients are
dipole moments in ``e Angstrom``; molecular induced dipoles accepted by this
module are in ``e bohr`` so they can be paired directly with an atomic-unit
polarizability tensor.
"""

from __future__ import annotations

import numpy as np
from ase.units import Bohr

from .gto_density import external_field_to_density_order
from .gto_galerkin import AtomCenteredL1GTOBasis


def _finite_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
    ):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    return array


def atomic_partition_to_l1_gto_coefficients(
    atomic_dipole_weights: np.ndarray,
    molecular_induced_dipole_ebohr: np.ndarray,
) -> np.ndarray:
    """Map an identity-bound atomic partition to neutral raw-GTO coefficients.

    ``atomic_dipole_weights[a]`` maps the Cartesian molecular induced dipole
    to atom ``a``.  The result has one radial channel and raw MACE ordering
    ``[q, l1_0, l1_1, l1_2]``.  It contains no induced monopoles, so exact
    charge conservation is structural rather than penalty-enforced.

    This function intentionally does not check whether the weights sum to the
    identity: that is an audit of the frozen upstream decomposition, not a
    property that may be repaired here.  Callers that require a molecular
    source must check the induced moment identity explicitly.
    """

    weights = _finite_array(atomic_dipole_weights, name="atomic_dipole_weights")
    if weights.ndim != 3 or weights.shape[1:] != (3, 3) or weights.shape[0] == 0:
        raise ValueError(
            "atomic_dipole_weights must be finite with shape (n_atoms, 3, 3)."
        )
    molecular_dipole = _finite_array(
        molecular_induced_dipole_ebohr,
        name="molecular_induced_dipole_ebohr",
        shape=(3,),
    )

    atomic_dipoles_eangstrom = np.einsum(
        "aij,j->ai", weights, molecular_dipole, optimize=True
    ) * Bohr
    external_cartesian = np.concatenate(
        (
            np.zeros((weights.shape[0], 1), dtype=float),
            atomic_dipoles_eangstrom,
        ),
        axis=1,
    )
    coefficients = external_field_to_density_order(external_cartesian)[:, None, :]
    if coefficients.shape != (weights.shape[0], 1, 4):
        raise RuntimeError("MACE-MDP induced source has an invalid GTO shape.")
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("MACE-MDP induced source contains non-finite values.")
    return coefficients


def induced_dipole_source_map(
    atomic_dipole_weights: np.ndarray,
) -> np.ndarray:
    """Return ``D`` with ``coefficients = D @ p`` for ``p`` in ``e bohr``.

    The returned shape is ``(n_atoms, 1, 4, 3)``.  It is built only by applying
    :func:`atomic_partition_to_l1_gto_coefficients` to Cartesian unit vectors,
    so its convention cannot drift from the direct source construction.
    """

    weights = _finite_array(atomic_dipole_weights, name="atomic_dipole_weights")
    if weights.ndim != 3 or weights.shape[1:] != (3, 3) or weights.shape[0] == 0:
        raise ValueError(
            "atomic_dipole_weights must be finite with shape (n_atoms, 3, 3)."
        )
    columns = [
        atomic_partition_to_l1_gto_coefficients(weights, np.eye(3)[index])
        for index in range(3)
    ]
    source_map = np.stack(columns, axis=-1)
    if source_map.shape != (weights.shape[0], 1, 4, 3):
        raise RuntimeError("MACE-MDP induced source map has an invalid shape.")
    return source_map


def induced_source_charge_and_dipole(
    atom_positions_angstrom: np.ndarray,
    coefficients: np.ndarray,
    *,
    sigma_angstrom: float,
) -> np.ndarray:
    """Return total charge and Cartesian dipole in ``[e, e Angstrom]``.

    This uses the same one-radial GTO convention that will generate the source
    potential.  It is useful for exposing the exact charge/moment identities
    before any continuum coupling is attempted.
    """

    positions = _finite_array(
        atom_positions_angstrom,
        name="atom_positions_angstrom",
    )
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 3:
        raise ValueError(
            "atom_positions_angstrom must be finite with shape (n_atoms, 3)."
        )
    basis = AtomCenteredL1GTOBasis((float(sigma_angstrom),))
    source = basis.validate_coefficients(
        coefficients,
        atom_count=positions.shape[0],
        name="induced_l1_gto_coefficients",
    )
    constraints = basis.molecular_charge_dipole_constraints(positions)
    return constraints @ source.reshape(-1)


__all__ = [
    "atomic_partition_to_l1_gto_coefficients",
    "induced_dipole_source_map",
    "induced_source_charge_and_dipole",
]
