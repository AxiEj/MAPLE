"""Maxwell-Boltzmann velocity initialization and temperature rescaling.

Depends on :mod:`.units`, :mod:`.dof`, and :mod:`.thermo`; re-exported by
``...md.utils``.
"""

from typing import Optional

import numpy as np
from ase import Atoms

from .dof import get_initialization_dof_policy, get_n_dof_from_policy
from .motion_projection import remove_center_of_mass_motion, remove_rigid_body_rotation
from .thermo import calculate_temperature
from .units import AMU_TO_AU, ANGSTROM_TO_BOHR, KELVIN_TO_HARTREE


def initialize_velocities(
    atoms: Atoms,
    temperature: float,
    remove_com: bool = True,
    remove_rotation: bool = False,
    remove_angular: Optional[bool] = None,
    target_n_dof: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Initialize velocities from a Maxwell-Boltzmann distribution.

    For each atom i with mass m_i at temperature T:
        v_i ~ N(0, sqrt(k_B * T / m_i))

    Initialization projection and runtime projection are intentionally distinct:
    `remove_com` / `remove_angular` act only on the initial velocity draw,
    whereas `remove_com_every` / `remove_angular_every` act during dynamics in
    the ensemble loops. To keep initialization and runtime thermodynamic targets
    consistent, callers should pass `target_n_dof` from the runtime DOF policy.
    If `target_n_dof` is omitted, a legacy fallback based on the initialization
    projection is used.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system.
    temperature : float
        Target temperature in Kelvin.
    remove_com : bool, default=True
        Initialization-only COM removal.
    remove_rotation : bool, default=False
        Legacy alias path used to support older call sites. Prefer
        `remove_angular`, which represents initialization-only COM + rigid-body
        rotation projection.
    remove_angular : bool, optional
        Initialization-only angular projection. When true, COM removal is always
        applied first and rigid-body rotation is projected out for non-periodic
        systems.
    target_n_dof : int, optional
        DOF used for the final temperature rescaling. In migrated MD paths this
        should come from the runtime policy so initialization and thermostat
        targets remain consistent.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Velocities in atomic units with shape ``(N_atoms, 3)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if remove_angular is None:
        remove_angular = bool(remove_rotation)
    if remove_angular:
        remove_com = True
        remove_rotation = True

    kT = temperature * KELVIN_TO_HARTREE
    masses = atoms.get_masses() * AMU_TO_AU
    n_atoms = len(atoms)

    # Generate random velocities from standard normal distribution
    velocities = rng.standard_normal(size=(n_atoms, 3))

    # Scale each atom's velocity by sqrt(kT/m)
    for i, mass in enumerate(masses):
        sigma = np.sqrt(kT / mass)
        velocities[i] *= sigma

    # Remove center of mass motion, then rescale to restore target temperature.
    # COM removal reduces the number of active DOF by 3, which lowers the
    # instantaneous kinetic energy below the target; rescaling corrects this.
    # Ref: Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.2
    if remove_com:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass

    # Remove overall rigid-body rotation (non-PBC only).
    #
    # Method (Shirts 2013, §2):
    #   1. Compute angular momentum L = Σ r_i × (m_i v_i)  in the COM frame.
    #   2. Compute the inertia tensor I = Σ m_i (|r_i|² E − r_i ⊗ r_i).
    #   3. Solve ω = I⁻¹ L  for the rigid-body angular velocity.
    #   4. Subtract the rigid-rotation contribution: v_i -= ω × r_i.
    #
    # This is a linear projection onto the subspace orthogonal to the three
    # infinitesimal rotation generators; internal DOF are exactly preserved.
    if remove_rotation and not any(atoms.pbc):
        positions_au = atoms.get_positions() * ANGSTROM_TO_BOHR  # Å → Bohr

        # Step 1 — COM frame positions
        total_mass = np.sum(masses)
        com = np.sum(masses[:, np.newaxis] * positions_au, axis=0) / total_mass
        r = positions_au - com  # (N, 3)

        # Step 2 — Angular momentum
        L = np.sum(
            masses[:, np.newaxis] * np.cross(r, velocities),
            axis=0
        )  # (3,)

        # Step 3 — Inertia tensor
        I = np.zeros((3, 3))
        for mi, ri in zip(masses, r):
            I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))

        # Step 4 — Solve for ω; use pseudoinverse to handle near-singular I
        # (e.g. linear molecules where one principal moment is ~0)
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            omega = np.linalg.lstsq(I, L, rcond=None)[0]

        # Step 5 — Subtract rigid rotation from each atom
        velocities -= np.cross(omega, r)  # v_i -= ω × r_i

    # Rescale to exact target temperature using the runtime DOF policy.
    # Initialization projection (`remove_com` / `remove_angular`) and runtime
    # projection (`remove_com_every` / `remove_angular_every`) are parallel
    # concepts. To keep initialization and thermostat targets consistent, the
    # target DOF for velocity scaling is provided explicitly by the caller and
    # should match the runtime policy. Fall back to the initialization policy only
    # for legacy callers that do not pass target_n_dof.
    if target_n_dof is None:
        init_policy = get_initialization_dof_policy(
            atoms,
            remove_com=bool(remove_com),
            remove_angular=bool(remove_angular),
        )
        n_dof = get_n_dof_from_policy(init_policy, n_atoms=n_atoms)
    else:
        n_dof = target_n_dof
    current_ke2 = np.sum(masses[:, np.newaxis] * velocities**2)  # 2*KE
    if n_dof > 0 and current_ke2 > 0:
        actual_temp = current_ke2 / (n_dof * KELVIN_TO_HARTREE)
        velocities *= np.sqrt(temperature / actual_temp)

    return velocities


def _coerce_velocity_array(atoms: Atoms, velocities: np.ndarray) -> np.ndarray:
    """Return a finite ``(N, 3)`` velocity array copy in atomic units."""
    array = np.asarray(velocities, dtype=float)
    expected_shape = (len(atoms), 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"Input velocities must have shape {expected_shape}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array)).tolist()
        raise ValueError(f"Input velocities must be finite; bad indices={bad}.")
    return array.copy()


def condition_input_velocities(
    atoms: Atoms,
    velocities: np.ndarray,
    temperature: float,
    remove_com: bool = True,
    remove_rotation: bool = False,
    remove_angular: Optional[bool] = None,
    target_n_dof: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project and rescale user-supplied velocities as an initialization draw.

    ``init_velocities=True`` means the fresh-run velocity state must satisfy the
    same initialization policy whether it came from a Maxwell-Boltzmann draw or
    from ``atoms.arrays["velocities"]``.  This helper applies the same
    initialization-only COM / angular projection as :func:`initialize_velocities`
    and then rescales the projected velocities to the requested target
    temperature in the initialization DOF basis.

    Callers that explicitly set ``init_velocities=False`` should *not* use this
    helper: those velocities are intentionally consumed as provided, and the DOF
    resolver treats them as unprojected input unless runtime projection is
    requested.
    """
    out = _coerce_velocity_array(atoms, velocities)

    if remove_angular is None:
        remove_angular = bool(remove_rotation)
    if remove_angular:
        remove_com = True
        remove_rotation = True

    if target_n_dof is None:
        init_policy = get_initialization_dof_policy(
            atoms,
            remove_com=bool(remove_com),
            remove_angular=bool(remove_angular),
        )
        n_dof = get_n_dof_from_policy(init_policy, n_atoms=len(atoms))
    else:
        n_dof = target_n_dof

    temperature_before = calculate_temperature(atoms, out, n_dof=n_dof)

    projected_com = False
    projected_angular = False
    if remove_com:
        out = remove_center_of_mass_motion(atoms, out)
        projected_com = True
    if remove_rotation and not any(atoms.pbc):
        out = remove_rigid_body_rotation(atoms, out)
        projected_angular = True

    masses = atoms.get_masses() * AMU_TO_AU
    current_ke2 = np.sum(masses[:, np.newaxis] * out**2)
    rescaled = False
    if n_dof > 0 and current_ke2 > 0:
        actual_temp = current_ke2 / (n_dof * KELVIN_TO_HARTREE)
        out *= np.sqrt(temperature / actual_temp)
        rescaled = True

    temperature_after = calculate_temperature(atoms, out, n_dof=n_dof)
    return out, {
        "n_dof": n_dof,
        "temperature_before": temperature_before,
        "temperature_after": temperature_after,
        "projected_com": projected_com,
        "projected_angular": projected_angular,
        "rescaled": rescaled,
    }


def scale_velocities_to_temperature(
    atoms: Atoms,
    velocities: np.ndarray,
    target_temperature: float
) -> np.ndarray:
    """
    Scale velocities to match target temperature.

    Useful for initialization or re-thermalization.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic system
    velocities : np.ndarray
        Current velocities
    target_temperature : float
        Target temperature in Kelvin

    Returns
    -------
    np.ndarray
        Scaled velocities
    """
    current_temp = calculate_temperature(atoms, velocities)

    if current_temp < 1e-10:  # Avoid division by zero
        return velocities

    scale_factor = np.sqrt(target_temperature / current_temp)
    return velocities * scale_factor
