"""Degree-of-freedom policy helpers and the linear-molecule test.

Depends on :mod:`.units`; re-exported by ``...md.utils``.  The
operator-aware *resolution* of init vs runtime DOF lives in ``semantics.py``;
these are the lower-level building blocks it composes.
"""

from typing import Dict, Optional

import numpy as np
from ase import Atoms

from .units import AMU_TO_AU, ANGSTROM_TO_BOHR


def is_linear_molecule(atoms: Atoms, tol: float = 1e-8) -> bool:
    """Return True if a non-periodic system is effectively linear."""
    if any(atoms.pbc):
        return False

    n_atoms = len(atoms)
    if n_atoms <= 1:
        return False
    if n_atoms == 2:
        return True

    positions = atoms.get_positions()
    masses = atoms.get_masses()
    total_mass = np.sum(masses)
    if total_mass <= 0:
        return False

    com = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    centered = positions - com
    if np.linalg.matrix_rank(centered, tol=tol) <= 1:
        return True

    centered_bohr = centered * ANGSTROM_TO_BOHR
    masses_au = masses * AMU_TO_AU
    inertia = np.zeros((3, 3))
    for mi, ri in zip(masses_au, centered_bohr):
        inertia += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    eigvals = np.sort(np.linalg.eigvalsh(inertia))
    return bool(eigvals[0] < tol * max(eigvals[-1], 1.0))


def get_initialization_dof_policy(
    atoms: Atoms,
    remove_com: bool = True,
    remove_angular: bool = False,
) -> Dict[str, object]:
    """Return initialization DOF policy for velocity generation.

    `remove_com` and `remove_angular` are parallel initialization settings.
    `remove_com` controls initialization-only COM removal. `remove_angular`
    controls initialization-only angular projection and always includes COM
    removal first.
    """
    warnings_list = []
    is_pbc = any(atoms.pbc)
    angular_active = bool(remove_angular and not is_pbc)
    if remove_angular and is_pbc:
        warnings_list.append(
            "remove_angular is ignored for periodic systems because global rigid-body rotation is not well-defined under PBC."
        )

    linear_active = bool(remove_com or angular_active)
    rotational_removed = 0
    if angular_active:
        rotational_removed = 2 if is_linear_molecule(atoms) else 3

    return {
        "atoms": atoms,
        "is_pbc": is_pbc,
        "angular_active": angular_active,
        "linear_active": linear_active,
        "rotational_dof_removed": rotational_removed,
        "warnings": warnings_list,
    }


def get_runtime_dof_policy(
    atoms: Atoms,
    remove_com_every: int = 0,
    remove_angular_every: int = 0,
) -> Dict[str, object]:
    """Return runtime DOF policy for temperature control and logging."""
    warnings_list = []
    is_pbc = any(atoms.pbc)
    angular_requested = remove_angular_every > 0
    linear_requested = remove_com_every > 0

    if is_pbc and angular_requested:
        warnings_list.append(
            "remove_angular_every is ignored for periodic systems because global rigid-body rotation is not well-defined under PBC. remove_com_every remains an independent optional runtime COM-drift removal under PBC."
        )
        angular_requested = False
    if is_pbc and linear_requested:
        warnings_list.append(
            "remove_com_every under PBC removes total momentum during dynamics. "
            "This can affect diffusion, velocity autocorrelation, and other transport analyses; "
            "set remove_com_every=0 for transport/strict dynamics."
        )

    angular_active = bool(angular_requested)
    linear_active = bool(angular_active or linear_requested)
    rotational_removed = 0
    if angular_active:
        rotational_removed = 2 if is_linear_molecule(atoms) else 3

    return {
        "atoms": atoms,
        "is_pbc": is_pbc,
        "angular_active": angular_active,
        "linear_active": linear_active,
        "rotational_dof_removed": rotational_removed,
        "warnings": warnings_list,
    }


def get_n_dof_from_policy(policy: Dict[str, object], n_atoms: Optional[int] = None) -> int:
    """Convert a DOF policy dictionary into an active N_dof count."""
    if n_atoms is None:
        atoms = policy.get("atoms")
        if atoms is None:
            raise ValueError("n_atoms is required when policy does not include atoms")
        n_atoms = len(atoms)

    n_dof = 3 * n_atoms
    if policy.get("linear_active", False):
        n_dof -= 3
    n_dof -= int(policy.get("rotational_dof_removed", 0))
    return max(n_dof, 1)


def describe_dof_policy(policy: Dict[str, object]) -> str:
    """Return a short human-readable DOF description for logs/summaries."""
    if policy.get("is_pbc", False):
        if policy.get("linear_active", False):
            return "PBC: 3N - 3 (runtime COM removal)"
        return "PBC: 3N"

    rotational = int(policy.get("rotational_dof_removed", 0))
    if policy.get("angular_active", False):
        return f"isolated: 3N - 3 - {rotational} (runtime angular removal)"
    if policy.get("linear_active", False):
        return "isolated: 3N - 3 (runtime COM removal)"
    return "isolated: 3N"
