"""Runtime COM/angular projection, velocity-representation conversions, momenta.

Depends on :mod:`.units`; re-exported by ``...md.utils``.
"""

from typing import Optional

import numpy as np
from ase import Atoms

from .units import (
    AMU_TO_AU,
    ANGSTROM_TO_BOHR,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    _VALID_VELOCITY_REPRESENTATIONS,
)


def remove_rigid_body_rotation(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """Project out rigid-body rotation in the center-of-mass frame."""
    if any(atoms.pbc) or len(atoms) <= 1:
        return velocities.copy()

    masses = atoms.get_masses() * AMU_TO_AU
    positions_au = atoms.get_positions() * ANGSTROM_TO_BOHR
    total_mass = np.sum(masses)
    com = np.sum(masses[:, np.newaxis] * positions_au, axis=0) / total_mass
    r = positions_au - com

    L = np.sum(masses[:, np.newaxis] * np.cross(r, velocities), axis=0)
    I = np.zeros((3, 3))
    for mi, ri in zip(masses, r):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))

    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        omega = np.linalg.lstsq(I, L, rcond=None)[0]

    return velocities - np.cross(omega, r)


def apply_runtime_motion_projection(
    atoms: Atoms,
    velocities: np.ndarray,
    step: int,
    remove_com_every: int = 0,
    remove_angular_every: int = 0,
) -> tuple[np.ndarray, str]:
    """Apply runtime COM/angular projection according to the configured cadence.

    `remove_com_every` and `remove_angular_every` are parallel settings, not
    enable/disable toggles. The former controls runtime COM removal only; the
    latter controls runtime angular projection. Under PBC, runtime COM removal
    may still be applied as an optional numerical COM-drift control, whereas
    runtime angular projection is ignored because global rigid-body rotation is
    not well-defined. If an angular projection fires, it always includes COM
    removal first and therefore supersedes COM-only removal for that step.
    """
    out = velocities.copy()
    if any(atoms.pbc):
        if remove_com_every and step % remove_com_every == 0:
            return remove_center_of_mass_motion(atoms, out), "com"
        return out, "none"

    if remove_angular_every and step % remove_angular_every == 0:
        out = remove_center_of_mass_motion(atoms, out)
        out = remove_rigid_body_rotation(atoms, out)
        return out, "angular"

    if remove_com_every and step % remove_com_every == 0:
        return remove_center_of_mass_motion(atoms, out), "com"

    return out, "none"


def remove_center_of_mass_motion(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Remove center of mass translational motion.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities
        Shape: (N_atoms, 3)

    Returns
    -------
    np.ndarray
        Velocities with COM motion removed
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    total_mass = np.sum(masses)

    velocities_corrected = velocities - total_momentum / total_mass
    return velocities_corrected


def normalize_velocity_representation(representation: Optional[str]) -> str:
    """Return a supported velocity representation label."""
    if representation in _VALID_VELOCITY_REPRESENTATIONS:
        return representation
    return VELOCITY_REPR_STANDARD


def get_atoms_velocity_representation(atoms: Atoms) -> str:
    """Read velocity representation metadata from ``atoms.info``."""
    return normalize_velocity_representation(atoms.info.get("velocity_representation"))


def set_atoms_velocity_representation(atoms: Atoms, representation: str) -> str:
    """Store normalized velocity representation metadata on ``atoms.info``."""
    normalized = normalize_velocity_representation(representation)
    atoms.info["velocity_representation"] = normalized
    return normalized


def standard_to_lfmiddle_carried(
    atoms: Atoms,
    velocities: np.ndarray,
    forces: np.ndarray,
    timestep_au: float,
) -> np.ndarray:
    """
    Convert standard velocities into the carried LF-Middle velocities.

    Zhang et al. (JPCA 2019, Eq. 16/19) formulate LF-Middle with leapfrog
    carried momentum/velocity as the internal state. For a fresh start from a
    standard velocity defined at the same coordinates, the corresponding
    carried velocity is obtained by a backward half-kick:

        v_carried = v_standard - 0.5 * (F / m) * dt

    Parameters
    ----------
    atoms : Atoms
        Atomic system.
    velocities : np.ndarray
        Standard velocities in atomic units.
    forces : np.ndarray
        Forces in atomic units (Ha/Bohr).
    timestep_au : float
        Timestep in atomic units.
    """
    masses = atoms.get_masses() * AMU_TO_AU
    return velocities - 0.5 * timestep_au * forces / masses[:, np.newaxis]


def lfmiddle_carried_to_standard(
    atoms: Atoms,
    velocities: np.ndarray,
    forces: np.ndarray,
    timestep_au: float,
) -> np.ndarray:
    """
    Convert carried LF-Middle velocities back into standard velocities.

    This is the inverse of ``standard_to_lfmiddle_carried()`` at the same
    coordinates/forces:

        v_standard = v_carried + 0.5 * (F / m) * dt
    """
    masses = atoms.get_masses() * AMU_TO_AU
    return velocities + 0.5 * timestep_au * forces / masses[:, np.newaxis]


def normalize_velocities_to_standard(
    atoms: Atoms,
    velocities: np.ndarray,
    representation: str,
    forces: np.ndarray,
    timestep_au: Optional[float],
) -> tuple[np.ndarray, str]:
    """Normalize restart/checkpoint velocities to the standard representation.

    The single gate every ensemble (NVE/NVT/NPT) routes its loaded velocities
    through.  A ``standard`` checkpoint is returned unchanged; an LF-Middle
    ``lfmiddle_carried`` checkpoint is converted back with the source-geometry
    ``forces`` and the *source* ``timestep_au`` recorded in the RST (Zhang et al.,
    JPCA 2019, Eq. 16/19).  A missing source timestep or an unrecognized
    representation is rejected, not guessed — so a half-kicked carried velocity can
    never be silently consumed as a full-step velocity (the NVE cross-ensemble
    hazard).  After this call the caller is free to re-derive a carried velocity for
    a Langevin integrator at the *current* timestep via
    :func:`standard_to_lfmiddle_carried`.
    """
    if representation == VELOCITY_REPR_STANDARD:
        return velocities, VELOCITY_REPR_STANDARD
    if representation == VELOCITY_REPR_LFMIDDLE_CARRIED:
        if timestep_au is None or not np.isfinite(timestep_au):
            raise ValueError(
                "Cannot convert lfmiddle_carried checkpoint velocities to standard "
                "without the source timestep from the RST checkpoint; refusing to guess."
            )
        standard = lfmiddle_carried_to_standard(atoms, velocities, forces, timestep_au)
        return standard, VELOCITY_REPR_STANDARD
    raise ValueError(
        f"Unrecognized velocity_representation {representation!r} in checkpoint; "
        f"expected one of {sorted(_VALID_VELOCITY_REPRESENTATIONS)}."
    )


def calculate_momentum(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """
    Calculate total momentum.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities

    Returns
    -------
    np.ndarray
        Total momentum vector (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    return total_momentum


def calculate_angular_momentum(
    atoms: Atoms,
    velocities: np.ndarray,
    origin: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate total angular momentum in atomic units.

    L = sum_i (r_i - origin) × (m_i * v_i)

    All quantities are converted to atomic units before computation:
    positions Å → Bohr, masses amu → a.u., velocities already in a.u.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Atomic velocities in atomic units (Bohr/a.u. time)
    origin : np.ndarray, optional
        Reference point in Å (default: center of mass).
        Converted to Bohr internally.

    Returns
    -------
    np.ndarray
        Angular momentum vector in atomic units (3,)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    positions = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

    if origin is None:
        # Center of mass in Bohr
        origin = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    else:
        origin = np.asarray(origin) * ANGSTROM_TO_BOHR

    angular_momentum = np.zeros(3)
    for mass, pos, vel in zip(masses, positions, velocities):
        r = pos - origin
        p = mass * vel
        angular_momentum += np.cross(r, p)

    return angular_momentum
