# -*- coding: utf-8 -*-
"""
NEB implementation with:
- Kabsch alignment (reports RMSD)
- IDPP interpolation for robust initial path
- Improved tangent NEB forces (true_perp + spring_parallel)
- L-BFGS optimizer on projected NEB forces
- ORCA-style dynamic spring constants
"""
from __future__ import annotations
from contextlib import ExitStack
import os
import stat
import tempfile
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import List, Literal, Tuple, Optional

import numpy as np
from ase import Atoms

from .logger import log_info
from .path_optimizer import LBFGSDriver
from ...jobABC import JobABC

from maple.function.dispatcher.numerical_state import (
    scientific_atoms_transaction,
    to_numpy_f64,
    vec1d,
)
from maple.function.dispatcher.result_contracts import (
    PRFOResult,
    PathStageResult,
    RunStatus,
)
from maple.function.utility import Molecules

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Rigid-body least-squares alignment (Kabsch):
    Align Q onto P. Both are (N,3). Returns: Q_aligned, rmsd, R, t
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3)")

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V @ Wt) < 0.0)
    if d:
        V[:, -1] *= -1.0
    R = V @ Wt
    t = Pc - Qc @ R

    Q_aligned = Q @ R + t
    diff = P - Q_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / P.shape[0]))
    return Q_aligned, rmsd, R, t

def write_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None):
    """
    Write a multi-frame XYZ trajectory. If energies given, write in comment line.
    """
    with open(filename, "w") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None:
                f.write(f"Image {i}  Energy = {energies[i]:.8f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _write_xyz_atomic(
    filename: str,
    images: List[Atoms],
    energies: Optional[List[float]] = None,
) -> None:
    """Publish a complete XYZ without exposing a partial current artifact."""
    directory = os.path.dirname(os.path.abspath(filename))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(filename) + ".",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        write_xyz(temporary, images, energies=energies)
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _retire_owned_nebts_output(output: str) -> Optional[str]:
    """Atomically archive this NEB job's current certified-TS artifact."""
    base, _ = os.path.splitext(os.path.abspath(output))
    current = base + "_nebts_ts.xyz"
    if not os.path.lexists(current):
        return None
    mode = os.lstat(current).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError("Refusing to retire a symlink or non-file NEB-TS output.")
    archive_index = 1
    while True:
        archive = f"{base}_nebts_ts.previous-{archive_index}.xyz"
        if not os.path.lexists(archive):
            break
        archive_index += 1
    os.replace(current, archive)
    return archive

def write_all_images_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None, iteration: int = 0):
    """
    Append all current images to an xyz trajectory file (for NEB debug).
    Each block corresponds to one iteration of NEB optimization.
    """
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    with open(filename, "a") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            E = None if energies is None else energies[i]
            f.write(f"{len(symbols)}\n")
            if E is not None:
                f.write(f"Iter {iteration} Image {i}  Energy = {E:.10f}\n")
            else:
                f.write(f"Iter {iteration} Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")






# =============================================================================
# ------------------------------- IDPP ----------------------------------------
# =============================================================================

@dataclass
class IDPPParams:
    max_steps: int = 300
    step_size: float = 0.01       # Cartesian step size for IDPP relaxation
    r_cut: float = 12.0           # cutoff for pair list (Å)
    eps: float = 1e-8             # to avoid 1/0
    w_power: float = 2.0          # weight ~ (1 / d_target^w_power)^2 (classical choice: 2-4)

def _pair_indices(n_atoms: int) -> np.ndarray:
    """Return upper-triangular (i<j) index pairs as (M,2)."""
    idx = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            idx.append((i, j))
    return np.asarray(idx, dtype=np.int32)

def _idpp_targets(P0: np.ndarray, P1: np.ndarray, n_inner: int) -> List[np.ndarray]:
    """
    Build target 1/dist arrays for each internal image by linear interpolation
    between endpoints pair distances.
    Returns list of arrays (M,), M = Npair.
    """
    n_atoms = P0.shape[0]
    pairs = _pair_indices(n_atoms)
    Rij0 = np.linalg.norm(P0[pairs[:, 0]] - P0[pairs[:, 1]], axis=1)
    Rij1 = np.linalg.norm(P1[pairs[:, 0]] - P1[pairs[:, 1]], axis=1)
    inv0 = 1.0 / np.maximum(Rij0, 1e-8)
    inv1 = 1.0 / np.maximum(Rij1, 1e-8)

    targets = []
    for k in range(1, n_inner + 1):
        lam = k / (n_inner + 1)
        inv_tgt = (1.0 - lam) * inv0 + lam * inv1
        targets.append(inv_tgt)
    return targets

# =============================================================================
# ------------------------------- NEB Core ------------------------------------
# =============================================================================

@dataclass
class NEBParams:
    n_images: int = 10                     # total images excluding endpoints
    k_min: float = 0.03                    # minimum spring constant
    k_max: float = 0.3                     # maximum spring constant
    use_dynamic_k: bool = True             # enable ORCA-style dynamic spring constants
    k_decay: float = 0.5                   # decay factor for dynamic k (ORCA default: 0.5)
    max_iter: int = 500 #256
    lbfgs_m: int = 20                      # memory size for L-BFGS
    step0: float = 2e-2                    # initial step length on search direction
    # IDPP control
    ifidpp: int = 1                     # 1: use IDPP for initial path, 0: linear interp.
    # Convergence on projected forces
    neb_f_max_th: float = 9.5e-3           # max(|Fp|) threshold
    neb_f_rms_th: float = 5e-3             # RMS(Fp) threshold
    initial_opt: bool = False              # do initial relaxation of endpoints
    refine: Optional[str] = None           # 'cineb' or 'nebts' or None
    # CINEB-specific
    cineb_f_max_th: float = 1e-2 #3e-03          # max(|Fp|) threshold for CINEB
    cineb_f_rms_th: float = 1e-2 #2e-03          # RMS(Fp) threshold for CINEB
    cilbfgs_m: int = 20                    # memory size for L-BFGS in CINEB
    cistep0: float = 5e-3                  # initial step length for CINEB
    cineb_max_iter: Optional[int] = None   # independent CI budget; None preserves max_iter
    # Optional maximum per-atom physical CI step length (Angstrom).
    ci_displacement_tol: Optional[float] = None
    prfo_max_iter: int = 256               # independent PRFO budget for NEB-TS
    prfo_project_rigid_modes: bool = False # explicit PRFO optimization subspace


def _validate_ci_displacement_tol(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("ci_displacement_tol must be finite and nonnegative or None.")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError("ci_displacement_tol must be finite and nonnegative or None.")
    return converted


def compute_dynamic_k(energies: List[float], k_min: float, k_max: float, k_decay: float = 0.5) -> List[float]:
    """
    Compute ORCA-style dynamic spring constants for each internal image.
    
    The spring constant for image i is computed as:
    k_i = k_max - (k_max - k_min) * exp(-decay * ΔE_i / ΔE_max)
    
    where:
    - ΔE_i = max(E_i - E_ref, 0) with E_ref = max(E_{i-1}, E_{i+1})
    - ΔE_max = max energy barrier along the path
    
    This ensures:
    - Images near the barrier get weaker springs (closer to k_min)
    - Images far from the barrier get stronger springs (closer to k_max)
    
    Parameters
    ----------
    energies : List[float]
        Energies of all images (including endpoints)
    k_min : float
        Minimum spring constant (for barrier region)
    k_max : float
        Maximum spring constant (for flat regions)
    k_decay : float
        Decay factor controlling transition sharpness (ORCA default: 0.5)
    
    Returns
    -------
    List[float]
        Spring constants for each image (length = len(energies))
        Endpoints get k_max by convention (not used in force calculation)
    """
    n_img = len(energies)
    k_springs = [k_max] * n_img  # initialize with k_max
    
    if n_img < 3:
        return k_springs
    
    # Compute energy barriers for each internal image
    delta_E = []
    for i in range(1, n_img - 1):
        # Local reference: max of neighboring energies
        E_ref_local = max(energies[i-1], energies[i+1])
        # Energy rise above local reference
        dE = max(energies[i] - E_ref_local, 0.0)
        delta_E.append(dE)
    
    # Find maximum barrier
    delta_E_max = max(delta_E) if delta_E else 1.0
    
    # Avoid division by zero
    if delta_E_max < 1e-10:
        # Essentially flat path, use k_max everywhere
        return k_springs
    
    # Compute dynamic k for each internal image
    for idx, dE in enumerate(delta_E):
        i = idx + 1  # actual image index
        # Normalized energy barrier
        dE_norm = dE / delta_E_max
        # ORCA formula: k decreases exponentially as we approach the barrier
        k_i = k_max - (k_max - k_min) * np.exp(-k_decay * dE_norm)
        k_springs[i] = k_i
    
    return k_springs


def improved_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Energy-weighted (Henkelman-Jonsson) improved tangent.
    Always returns a normalized (3N,) vector.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)
    Em, Ep = Em1, Ep1

    # decide direction
    if Ep > E and E > Em:
        t = dp
    elif Ep < E and E < Em:
        t = dm
    else:
        dE_plus = abs(Ep - E)
        dE_minus = abs(Em - E)
        dE_max = max(dE_plus, dE_minus)
        dE_min = min(dE_plus, dE_minus)
        if Ep > Em:
            t = dE_max * dp + dE_min * dm
        else:
            t = dE_min * dp + dE_max * dm

    # normalize
    norm = np.linalg.norm(t)

    if norm < 1e-16:
        # fallback 1
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            raise ValueError("NEB tangent is degenerate; no path direction exists.")
    
    return t / norm

def cineb_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Upwinding tangent for CINEB (Henkelman-Jonsson style).
    Returns a normalized (3N,) vector.

    This function is separate from `improved_tangent` so that:
    - NEB keeps using the original `improved_tangent`.
    - CINEB can use a robust upwinding scheme without touching NEB behavior.
    """
    dm = vec1d(R - Rm1)
    dp = vec1d(Rp1 - R)

    prev_energy = Em1
    ith_energy  = E
    next_energy = Ep1

    # Energy differences relative to image i
    dE_plus  = abs(next_energy - ith_energy)
    dE_minus = abs(prev_energy - ith_energy)

    # Strict uphill: E_{i-1} < E_i < E_{i+1} -> use forward tangent
    if (next_energy > ith_energy) and (ith_energy > prev_energy):
        t = dp

    # Strict downhill: E_{i-1} > E_i > E_{i+1} -> use backward tangent
    elif (next_energy < ith_energy) and (ith_energy < prev_energy):
        t = dm

    else:
        # Around maxima or minima: use energy-weighted combination
        delta_max = max(dE_plus, dE_minus)
        delta_min = min(dE_plus, dE_minus)

        if next_energy >= prev_energy:
            # Next image has higher energy
            t = delta_max * dp + delta_min * dm
        else:
            # Previous image has higher energy
            t = delta_min * dp + delta_max * dm

    norm = np.linalg.norm(t)
    if norm < 1e-16:
        # Degenerate case, fall back to simple sum
        t = dp + dm
        norm = np.linalg.norm(t)
        if norm < 1e-16:
            raise ValueError("CINEB tangent is degenerate; no path direction exists.")

    return t / norm


def neb_forces(images: List[Atoms], energies: List[float], 
               k_spring: float = None, k_springs: List[float] = None,
               use_dynamic_k: bool = False, k_min: float = 0.03, 
               k_max: float = 0.3, k_decay: float = 0.5,
               raw_forces: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], float, int]:
    """
    Compute NEB projected forces for internal images:
    F_NEB = F_true_perp + F_spring_parallel  (per image)
    
    Parameters
    ----------
    images : List[Atoms]
        All images including endpoints
    energies : List[float]
        Energies of all images
    k_spring : float, optional
        Single spring constant (used if k_springs is None and use_dynamic_k=False)
    k_springs : List[float], optional
        Pre-computed spring constants for each image
    use_dynamic_k : bool
        If True and k_springs is None, compute dynamic spring constants
    k_min, k_max, k_decay : float
        Parameters for dynamic spring constant calculation
    
    Returns
    -------
    forces_proj : List[np.ndarray]
        Projected forces (cartesian shape (N,3)) for each image
    max_fp : float
        Maximum force component among all internal images
    hei_idx : int
        Index of highest energy image
    """
    n_img = len(images)
    forces_proj = [None] * n_img
    max_fp = 0.0
    hei_idx = 1

    # get raw forces and flatten
    if raw_forces is None:
        raw_forces = [to_numpy_f64(at.get_forces()) for at in images]
    else:
        raw_forces = [to_numpy_f64(force) for force in raw_forces]
    coords = [to_numpy_f64(at.get_positions()) for at in images]
    Es = [float(e) for e in energies]

    # determine HEI
    inner_indices = list(range(1, n_img - 1))
    hei_idx = max(inner_indices, key=lambda i: Es[i])

    # Determine spring constants
    if k_springs is None:
        if use_dynamic_k:
            k_springs = compute_dynamic_k(Es, k_min, k_max, k_decay)
        else:
            # Use single spring constant for all images
            k_springs = [k_spring if k_spring is not None else k_max] * n_img

    # loop over internal images
    for i in inner_indices:
        Rm1 = coords[i - 1]; R = coords[i]; Rp1 = coords[i + 1]
        Em1 = Es[i - 1];     E = Es[i];     Ep1 = Es[i + 1]

        # tangent
        tau = improved_tangent(Rm1, R, Rp1, Em1, E, Ep1)  # (3N,)
        tau_resh = tau.reshape(-1, 3)

        # true force (calculator gives forces = -grad V); NEB uses gradient convention:
        # F_true = -∇V, but our "forces" are already that.
        F_true = vec1d(raw_forces[i])

        # remove tangent component: F_true_perp = F_true - (F_true·tau) tau
        c = float(np.dot(F_true, tau))
        F_true_perp = F_true - c * tau
        F_true_perp = F_true_perp.reshape(-1, 3)

        # spring force along tangent with dynamic k
        k_i = k_springs[i]
        d_next = float(np.linalg.norm(Rp1 - R))
        d_prev = float(np.linalg.norm(R - Rm1))
        F_spring_par = k_i * (d_next - d_prev) * tau_resh

        # total projected
        Fp = F_true_perp + F_spring_par
        forces_proj[i] = Fp

        max_fp = max(max_fp, float(np.abs(Fp).max()))

    # endpoints fixed: zero projected forces
    forces_proj[0] = np.zeros_like(coords[0])
    forces_proj[-1] = np.zeros_like(coords[-1])

    return forces_proj, max_fp, hei_idx

def rms_force(forces_list: List[np.ndarray]) -> float:
    """RMS of projected forces over all DOFs of internal images."""
    arrs = []
    for i, F in enumerate(forces_list):
        if i == 0 or i == len(forces_list) - 1:
            continue
        arrs.append(F.reshape(-1))
    if not arrs:
        return 0.0
    v = np.concatenate(arrs)
    return float(np.sqrt(np.mean(v * v)))


def _pack_internal(images: List[Atoms]) -> np.ndarray:
    coordinates = [
        to_numpy_f64(image.get_positions()).reshape(-1)
        for image in images[1:-1]
    ]
    return np.concatenate(coordinates) if coordinates else np.zeros(0, dtype=np.float64)


def _unpack_internal(coordinates: np.ndarray, images: List[Atoms]) -> None:
    offset = 0
    for image in images[1:-1]:
        size = len(image) * 3
        image.set_positions(coordinates[offset : offset + size].reshape(-1, 3))
        offset += size


def _evaluate_neb_images(images: List[Atoms]):
    from maple.function.dispatcher.implicit_path import (
        evaluate_path_images,
        uses_admitted_implicit_path,
    )

    with ExitStack() as stack:
        for index, image in enumerate(images):
            stack.enter_context(
                scientific_atoms_transaction(
                    image,
                    operation=f"NEB image {index} energy/force",
                )
            )
        if uses_admitted_implicit_path(images):
            energies, forces = evaluate_path_images(images, label="NEB")
        else:
            energies = [
                image.get_potential_energy(force_consistent=True) for image in images
            ]
            forces = [image.get_forces() for image in images]
    return (
        [float(to_numpy_f64(energy)) for energy in energies],
        [np.array(to_numpy_f64(force), copy=True) for force in forces],
    )


def _cineb_forces(images, energies, raw_forces, k_spring, climbing_index):
    coordinates = [to_numpy_f64(image.get_positions()) for image in images]
    projected = [np.zeros_like(coordinates[0]) for _ in images]
    for index in range(1, len(images) - 1):
        tangent = improved_tangent(
            coordinates[index - 1],
            coordinates[index],
            coordinates[index + 1],
            energies[index - 1],
            energies[index],
            energies[index + 1],
        )
        true_force = vec1d(raw_forces[index])
        if index == climbing_index:
            # Henkelman-Jonsson climbing image: F_CI = F - 2(F dot tau)tau.
            projected[index] = (
                true_force - 2.0 * np.dot(true_force, tangent) * tangent
            ).reshape(-1, 3)
            continue
        perpendicular = true_force - np.dot(true_force, tangent) * tangent
        next_distance = float(
            np.linalg.norm(coordinates[index + 1] - coordinates[index])
        )
        previous_distance = float(
            np.linalg.norm(coordinates[index] - coordinates[index - 1])
        )
        spring = k_spring * (next_distance - previous_distance) * tangent
        projected[index] = perpendicular.reshape(-1, 3) + spring.reshape(-1, 3)
    return projected


def run_neb_stage(
    images: List[Atoms],
    *,
    params: NEBParams,
    stage_kind: Literal["base", "ci"],
    output: str,
) -> PathStageResult:
    """Optimize one already-built base or climbing-image NEB stage."""
    if stage_kind not in {"base", "ci"}:
        raise ValueError("stage_kind must be 'base' or 'ci'.")
    if len(images) < 3:
        raise ValueError("An NEB stage requires at least one internal image.")
    ci_displacement_tol = _validate_ci_displacement_tol(
        params.ci_displacement_tol
    )

    if stage_kind == "base":
        maximum_iterations = int(params.max_iter)
        driver = LBFGSDriver(
            m=params.lbfgs_m,
            curvature=70.0,
            maxstep=params.step0,
        )
    else:
        maximum_iterations = int(
            params.max_iter
            if params.cineb_max_iter is None
            else params.cineb_max_iter
        )
        driver = LBFGSDriver(
            m=params.cilbfgs_m,
            curvature=70.0,
            maxstep=params.cistep0,
        )
        driver.fmax_reg = params.neb_f_max_th
        driver.frms_reg = params.neb_f_rms_th
        driver.fmax_ci = params.cineb_f_max_th
        driver.frms_ci = params.cineb_f_rms_th
    if maximum_iterations < 0:
        raise ValueError("NEB stage iteration budgets must be nonnegative.")

    climbing_index = None
    previous_ci_positions = None
    ci_max_displacement = 0.0

    def evaluate(coordinates):
        nonlocal climbing_index, previous_ci_positions, ci_max_displacement
        _unpack_internal(coordinates, images)
        energies, raw_forces = _evaluate_neb_images(images)
        if stage_kind == "base":
            projected, _, highest_index = neb_forces(
                images,
                energies,
                use_dynamic_k=params.use_dynamic_k,
                k_min=params.k_min,
                k_max=params.k_max,
                k_decay=params.k_decay,
                raw_forces=raw_forces,
            )
        else:
            if climbing_index is None:
                climbing_index = max(
                    range(1, len(images) - 1), key=lambda index: energies[index]
                )
            current_ci_positions = to_numpy_f64(
                images[climbing_index].get_positions()
            )
            if previous_ci_positions is not None:
                ci_max_displacement = float(
                    np.linalg.norm(
                        current_ci_positions - previous_ci_positions,
                        axis=1,
                    ).max(initial=0.0)
                )
            previous_ci_positions = np.array(current_ci_positions, copy=True)
            highest_index = climbing_index
            projected = _cineb_forces(
                images,
                energies,
                raw_forces,
                params.k_max,
                climbing_index,
            )
        gradient = np.concatenate(
            [(-projected[index]).reshape(-1) for index in range(1, len(images) - 1)]
        )
        return gradient, energies, raw_forces, projected, highest_index

    coordinates = _pack_internal(images)
    gradient, energies, raw_forces, projected, highest_index = evaluate(coordinates)

    def is_converged():
        if stage_kind == "base":
            return driver.should_stop(
                gradient,
                params.neb_f_max_th,
                params.neb_f_rms_th,
            )
        regular = [
            projected[index].reshape(-1)
            for index in range(1, len(images) - 1)
            if index != climbing_index
        ]
        regular_forces = (
            np.concatenate(regular) if regular else np.zeros(0, dtype=np.float64)
        )
        force_converged = driver.ci_should_stop(
            regular_forces,
            raw_forces[climbing_index].reshape(-1),
        )
        displacement_converged = (
            ci_displacement_tol is None
            or ci_max_displacement <= ci_displacement_tol
        )
        return force_converged and displacement_converged

    converged = is_converged()
    iterations = 0
    trajectory_suffix = "_image_traj.xyz" if stage_kind == "base" else "_cineb_traj.xyz"
    trajectory = os.path.splitext(output)[0] + trajectory_suffix
    write_all_images_xyz(trajectory, images, energies=energies, iteration=0)
    while iterations < maximum_iterations and not converged:
        step = driver.step_limit(driver.two_loop(gradient))
        new_coordinates = coordinates + step
        (
            new_gradient,
            energies,
            raw_forces,
            projected,
            highest_index,
        ) = evaluate(new_coordinates)
        driver.update(new_coordinates - coordinates, new_gradient - gradient)
        coordinates = new_coordinates
        gradient = new_gradient
        iterations += 1
        converged = is_converged()
        write_all_images_xyz(
            trajectory,
            images,
            energies=energies,
            iteration=iterations,
        )

    result = PathStageResult.capture(
        status=(RunStatus.CONVERGED if converged else RunStatus.BOUNDED_NONCONVERGED),
        images=images,
        energies_hartree=energies,
        forces_hartree_per_angstrom=raw_forces,
        budgets={
            "maximum_iterations": maximum_iterations,
            "used_iterations": iterations,
        },
        termination_reason="force_converged" if converged else "maximum_iterations",
        candidate=images[highest_index],
    )
    object.__setattr__(
        result,
        "_stage_diagnostics",
        MappingProxyType(
            {
                "climbing_image_index": climbing_index,
                "ci_max_displacement_angstrom": (
                    ci_max_displacement if stage_kind == "ci" else None
                ),
            }
        ),
    )
    return result

# =============================================================================
# ------------------------------- NEB Class -----------------------------------
# =============================================================================

class NEB(JobABC):
    def __init__(self,
                output: str,
                atoms_or_molecules,
                paras: Optional[dict] = None):
        super().__init__(output)

        # Handle Molecules input
        if isinstance(atoms_or_molecules, Molecules):
            self.input_images = atoms_or_molecules.multiatoms
            self.atoms_R = None
            self.atoms_P = None
        else:
            raise ValueError("Please provide Molecules object containing all images")

        threshold_names = ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
        self._prfo_thresholds = {}
        for name in threshold_names:
            endpoint_values = [
                getattr(endpoint, name, None)
                for endpoint in (self.input_images[0], self.input_images[-1])
            ]
            if (endpoint_values[0] is None) != (endpoint_values[1] is None):
                raise ValueError(
                    f"NEB endpoint thresholds disagree for {name}: one is missing."
                )
            if endpoint_values[0] is not None:
                if endpoint_values[0] != endpoint_values[1]:
                    raise ValueError(
                        f"NEB endpoint thresholds disagree for {name}: "
                        f"{endpoint_values[0]!r} != {endpoint_values[1]!r}."
                    )
                self._prfo_thresholds[name] = endpoint_values[0]

        # Initialize params from paras dict
        self.params = self._init_params(NEBParams, paras, ("neb", "NEB", "ts"))

        # Safety: minimal guard
        if self.params.n_images < 1:
            raise ValueError("n_images must be >= 1")
        if self.params.cineb_max_iter is not None and (
            type(self.params.cineb_max_iter) is not int
            or self.params.cineb_max_iter < 0
        ):
            raise ValueError("cineb_max_iter must be a nonnegative integer or None.")
        self.params.ci_displacement_tol = _validate_ci_displacement_tol(
            self.params.ci_displacement_tol
        )
        if type(self.params.prfo_max_iter) is not int or self.params.prfo_max_iter < 0:
            raise ValueError("prfo_max_iter must be a nonnegative integer.")
        if type(self.params.prfo_project_rigid_modes) is not bool:
            raise TypeError("prfo_project_rigid_modes must be a boolean.")
        if self.params.refine not in {None, "cineb", "nebts"}:
            raise ValueError("NEB refine must be None, 'cineb', or 'nebts'.")

        self.params.n_images = int(self.params.n_images) + 2  # total images including endpoints
        self.nebts_retired_output = None

    @staticmethod
    def _evaluate_images(images: List[Atoms]) -> Tuple[List[float], List[np.ndarray]]:
        return _evaluate_neb_images(images)


    def optimize_endpoints(self, atoms_R: Atoms, atoms_P: Atoms, 
                        f_max_th=1.0e-3, f_rms_th=5.0e-4, max_iter=200):
        """
        Optimize the reactant and product endpoints before NEB if initial_opt=True.
        After optimization, write the minimized structures to XYZ files.
        """
        def single_point_optimize(atoms: Atoms):
            driver = LBFGSDriver(m=self.params.lbfgs_m, curvature=70.0, maxstep=self.params.step0)
            iteration = 0
            g = -to_numpy_f64(atoms.get_forces()).reshape(-1)
            x = to_numpy_f64(atoms.get_positions()).reshape(-1)

            while iteration < max_iter and not driver.should_stop(g, f_max_th, f_rms_th):
                p = driver.two_loop(g)
                p = driver.step_limit(p)
                x_new = x + p

                # update positions and forces
                atoms.set_positions(x_new.reshape(-1, 3))
                g_new = -to_numpy_f64(atoms.get_forces()).reshape(-1)
                driver.update(x_new - x, g_new - g)

                x = x_new
                g = g_new
                iteration += 1

            return atoms

        log_info(["\nInitial endpoint optimization started...\n"], self.output)
        atoms_R = single_point_optimize(atoms_R)
        atoms_P = single_point_optimize(atoms_P)
        log_info(["Initial endpoint optimization completed.\n"], self.output)

        # --- write optimized endpoints to XYZ ---
        base, ext = os.path.splitext(self.output)
        reactant_path = base + "_reactant_min.xyz"
        product_path  = base + "_product_min.xyz"
        write_xyz(reactant_path, [atoms_R], energies=[atoms_R.get_potential_energy(force_consistent=True)])
        write_xyz(product_path,  [atoms_P], energies=[atoms_P.get_potential_energy(force_consistent=True)])

        log_info([f"Optimized reactant written to: {reactant_path}\n"], self.output)
        log_info([f"Optimized product written to:  {product_path}\n"], self.output)

        return atoms_R, atoms_P

    def atoms_to_xyz(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"

    def _align_path(self, images: List[Atoms], ref_mode: str = "reactant"):
        """
        Rigid alignment for all internal images each iteration.
        ref_mode = "reactant"  → use images[0] as reference
                 = "centroid"  → use centroid of all images as reference
        Returns a list of RMSDs for internal images.
        """
        # choose reference coordinates
        if ref_mode == "reactant":
            ref = to_numpy_f64(images[0].get_positions())
        elif ref_mode == "centroid":
            coords = [to_numpy_f64(img.get_positions()) for img in images]
            ref = np.mean(coords, axis=0)
        else:
            ref = to_numpy_f64(images[0].get_positions())

        rmsds = []
        # align internal images only (1..N-2)
        for i in range(1, len(images) - 1):
            Q = to_numpy_f64(images[i].get_positions())
            Q_aligned, rmsd, _, _ = kabsch_align(ref, Q)
            images[i].set_positions(Q_aligned)
            rmsds.append(rmsd)
        return rmsds


    def get_energies(self, imgs): 
        return [float(at.get_potential_energy(force_consistent=True)) for at in imgs]
    
    def _compute_distances(self, images: List[Atoms]) -> List[float]:
        """
        Compute straight-line distances between consecutive images.
        Returns list of distances with length = len(images) - 1
        """
        distances = []
        for i in range(len(images) - 1):
            pos1 = to_numpy_f64(images[i].get_positions())
            pos2 = to_numpy_f64(images[i+1].get_positions())
            dist = np.linalg.norm(pos2 - pos1)
            distances.append(dist)
        return distances

    def _determine_insertion_plan(self, distances: List[float], n_to_insert: int) -> List[Tuple[int, int]]:
        """
        Determine where to insert new images based on distances.
        
        Parameters
        ----------
        distances : List[float]
            Distances between consecutive images
        n_to_insert : int
            Total number of images to insert
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (segment_index, count) indicating how many images to insert after segment_index
            
        Example: if distances = [4.0, 2.0, 1.0] and n_to_insert = 3
                largest gaps are at index 0 (4.0) and 1 (2.0)
                return [(0, 2), (1, 1)] means insert 2 after segment 0, 1 after segment 1
        """
        # Create list of (index, distance) and sort by distance (descending)
        indexed_distances = [(i, d) for i, d in enumerate(distances)]
        indexed_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute insertions proportionally to gap size
        insertion_counts = [0] * len(distances)
        
        for k in range(n_to_insert):
            # Insert in the largest remaining gap
            # Find segment with largest distance/insertions ratio
            max_ratio = -1
            max_idx = 0
            for i, d in enumerate(distances):
                ratio = d / (insertion_counts[i] + 1)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_idx = i
            insertion_counts[max_idx] += 1
        
        # Convert to list of (index, count) tuples
        plan = [(i, count) for i, count in enumerate(insertion_counts) if count > 0]
        return plan

    def _insert_images_by_plan(self, images: List[Atoms], plan: List[Tuple[int, int]]) -> List[Atoms]:
        """
        Insert interpolated images according to the insertion plan.
        
        Parameters
        ----------
        images : List[Atoms]
            Current list of images
        plan : List[Tuple[int, int]]
            Insertion plan from _determine_insertion_plan
        
        Returns
        -------
        List[Atoms]
            New list with interpolated images inserted
        """
        # Sort plan by index (descending) to insert from back to front
        plan_sorted = sorted(plan, key=lambda x: x[0], reverse=True)
        
        new_images = images.copy()
        
        for seg_idx, count in plan_sorted:
            # Insert 'count' images between new_images[seg_idx] and new_images[seg_idx+1]
            pos1 = to_numpy_f64(new_images[seg_idx].get_positions())
            pos2 = to_numpy_f64(new_images[seg_idx + 1].get_positions())
            
            inserted = []
            for k in range(1, count + 1):
                # Linear interpolation
                lam = k / (count + 1)
                new_pos = (1.0 - lam) * pos1 + lam * pos2
                
                # Copy atoms object and set new positions
                new_atom = new_images[seg_idx].copy()
                new_atom.set_positions(new_pos)
                new_atom.calc = new_images[seg_idx].calc  # Inherit calculator
                inserted.append(new_atom)
            
            # Insert all new images after seg_idx
            for idx, img in enumerate(inserted):
                new_images.insert(seg_idx + 1 + idx, img)
        
        return new_images
    
    # ---------------- Climbing Image NEB (CINEB) -------------------------------
    def cineb_forces(
        self,
        images: List[Atoms],
        energies: List[float],
        k_spring: float,
        raw_forces: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], float, int]:
        if raw_forces is None:
            _, raw_forces = self._evaluate_images(images)
        inner_indices = range(1, len(images) - 1)
        fixed_index = getattr(self, "_cineb_fixed_hei", None)
        climbing_index = (
            fixed_index
            if fixed_index in inner_indices
            else max(inner_indices, key=lambda index: energies[index])
        )
        projected = _cineb_forces(
            images,
            energies,
            raw_forces,
            k_spring,
            climbing_index,
        )
        nonclimbing = [
            np.abs(projected[index]).max()
            for index in inner_indices
            if index != climbing_index
        ]
        max_projected = float(max(nonclimbing, default=0.0))
        return projected, max_projected, climbing_index


    def _run_cineb_stage(self, images: List[Atoms]) -> PathStageResult:
        stage = run_neb_stage(
            images,
            params=self.params,
            stage_kind="ci",
            output=self.output,
        )
        climbing_index = stage._stage_diagnostics["climbing_image_index"]
        if climbing_index not in range(1, len(images) - 1):
            raise RuntimeError("CI stage did not report a valid internal climbing index.")
        projected = _cineb_forces(
            images,
            list(stage.energies_hartree),
            list(stage.forces_hartree_per_angstrom),
            self.params.k_max,
            climbing_index,
        )
        self._ci_projected_forces = tuple(
            np.array(force, copy=True) for force in projected
        )
        self._ci_climbing_index = climbing_index
        self._ci_candidate = images[climbing_index]
        base, _ = os.path.splitext(self.output)
        write_xyz(base + "_cineb_mep.xyz", images, energies=list(stage.energies_hartree))
        write_xyz(
            base + "_cineb_hei.xyz",
            [images[climbing_index]],
            energies=[stage.energies_hartree[climbing_index]],
        )
        return stage

    def _run_nebts_stage(self, ci_stage: PathStageResult) -> PathStageResult:
        """Refine a converged CI candidate without changing the MEP images."""
        from .PRFO import PRFO

        self._retire_nebts_current()
        ts_guess = self._ci_candidate.copy()
        ts_guess.calc = self._ci_candidate.calc
        for name, value in self._prfo_thresholds.items():
            setattr(ts_guess, name, value)

        prfo_parameters = {
            "prfo": {
                "max_iter": int(self.params.prfo_max_iter),
                "project_rigid_modes": bool(self.params.prfo_project_rigid_modes),
            }
        }
        prfo_result = PRFO(
            output=self.output,
            atoms=ts_guess,
            paras=prfo_parameters,
        ).run()
        if not isinstance(prfo_result, PRFOResult):
            raise TypeError("NEB-TS requires PRFO.run() to return PRFOResult.")
        self.prfo_result = prfo_result

        if prfo_result.certified_ts:
            base, _ = os.path.splitext(self.output)
            write_xyz(
                base + "_nebts_mep.xyz",
                list(ci_stage.images),
                energies=list(ci_stage.energies_hartree),
            )
            _write_xyz_atomic(
                base + "_nebts_ts.xyz",
                [prfo_result.atoms],
                energies=[prfo_result.energy_hartree],
            )

        return PathStageResult.capture(
            status=prfo_result.status,
            images=ci_stage.images,
            energies_hartree=ci_stage.energies_hartree,
            forces_hartree_per_angstrom=ci_stage.forces_hartree_per_angstrom,
            budgets={
                "maximum_iterations": int(self.params.prfo_max_iter),
                "used_iterations": prfo_result.iterations,
            },
            termination_reason=prfo_result.termination_reason,
            candidate=prfo_result.atoms,
            certification=prfo_result.certification,
        )

    def _retire_nebts_current(self) -> None:
        if self.params.refine != "nebts":
            return
        retired_output = _retire_owned_nebts_output(self.output)
        if retired_output is not None:
            self.nebts_retired_output = retired_output
            log_info(
                [
                    "Retired previous certified NEB-TS output to: ",
                    retired_output,
                    "\n",
                ],
                self.output,
            )

    def restart_run(self, images: List[Atoms], energies=None) -> PathStageResult:
        """Run CI-NEB and optional certified-only PRFO refinement."""
        del energies
        self._retire_nebts_current()
        self.ci_stage_result = self._run_cineb_stage(images)
        self.nebts_stage_result = None
        if self.ci_stage_result.status is not RunStatus.CONVERGED:
            return self.ci_stage_result
        if self.params.refine != "nebts":
            return self.ci_stage_result
        self.nebts_stage_result = self._run_nebts_stage(self.ci_stage_result)
        return self.nebts_stage_result


    # ==============================================================
    # IDPP smoothing for initial path
    # ==============================================================
    def _run_idpp_smoothing(self, images, idpp_params=IDPPParams()):
        """
        Strong IDPP smoothing (BNEB-style):
        - LBFGS optimization instead of GD
        - damping + curvature correction
        - step clipping
        - projected force formulation identical to pysisyphus/BNEB

        This function preserves your image list + ASE Atoms interface.
        """

        # ----- extract coords -----
        coords = [to_numpy_f64(img.get_positions()) for img in images]
        R0 = coords[0]
        R1 = coords[-1]

        n_img = len(images)
        n_inner = n_img - 2
        n_atoms = R0.shape[0]

        # pair list (upper-triangle)
        pairs = _pair_indices(n_atoms)
        n_pair = pairs.shape[0]

        # ----- IDPP target 1/dist -----
        targets = _idpp_targets(R0, R1, n_inner)  # list length = n_inner

        # ----- flatten internal images -----
        x = np.concatenate([coords[i].reshape(-1) for i in range(1, n_img - 1)])

        # ----- LBFGS state -----
        m = 7
        S = []
        Y = []
        rho = []
        maxstep = 0.1
        curvature = 1.0

        def lbfgs_direction(g):
            """Two-loop recursion."""
            q = g.copy()
            alpha = []

            for s, y, r in reversed(list(zip(S, Y, rho))):
                a = r * np.dot(s, q)
                alpha.append(a)
                q -= a * y

            if len(Y) > 0:
                gamma = np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-20)
            else:
                gamma = 1.0 / curvature

            z = gamma * q

            for (s, y, r), a in zip(zip(S, Y, rho), reversed(alpha)):
                b = r * np.dot(y, z)
                z += s * (a - b)

            return -z

        def compute_grad(x):
            """Compute IDPP gradient for all internal images."""
            grad = np.zeros_like(x)

            offset = 0
            for k in range(n_inner):
                Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)

                # pair distances
                Rij = Xi[pairs[:, 0]] - Xi[pairs[:, 1]]      # (M,3)
                dij = np.linalg.norm(Rij, axis=1) + 1e-12
                inv = 1.0 / dij

                diff = inv - targets[k]                     # (M,)
                dE_dd = -2.0 * diff / (dij**2)              # derivative wrt dij

                g_pair = (dE_dd[:, None] * (Rij / dij[:, None]))  # (M,3)

                # accumulate forces
                g_atoms = np.zeros_like(Xi)
                for p in range(n_pair):
                    i, j = pairs[p]
                    g_atoms[i] +=  g_pair[p]
                    g_atoms[j] += -g_pair[p]

                grad[offset:offset + n_atoms * 3] = g_atoms.reshape(-1)
                offset += n_atoms * 3

            return grad

        # ----- optimization loop -----
        grad = compute_grad(x)

        for _ in range(idpp_params.max_steps):
            g = grad.copy()
            step = lbfgs_direction(g)

            # step clipping
            max_disp = np.max(np.abs(step))
            if max_disp > maxstep:
                step *= maxstep / max_disp

            x_new = x + step
            grad_new = compute_grad(x_new)

            # LBFGS update
            s = x_new - x
            y = grad_new - grad

            sy = np.dot(s, y)
            if sy > 1e-12:
                if len(S) == m:
                    S.pop(0); Y.pop(0); rho.pop(0)
                S.append(s); Y.append(y); rho.append(1.0 / sy)

            x = x_new
            grad = grad_new

            # stopping based on RMS gradient
            if np.sqrt(np.mean(grad * grad)) < idpp_params.eps:
                break

        # ----- write back optimized coords -----
        offset = 0
        for i in range(1, n_img - 1):
            Xi = x[offset:offset + n_atoms * 3].reshape(n_atoms, 3)
            images[i].set_positions(Xi)
            offset += n_atoms * 3

        return images



    # ------------------------------- main flow --------------------------------

    def run(self):
        # ===================================================================
        # Step 0: Process input images and determine if interpolation needed
        # ===================================================================
        n_input = len(self.input_images)
        n_required = self.params.n_images  # This is already n_images + 2
        
        log_info([
            f"\n{'='*70}\n",
            "NEB Initialization\n",
            f"{'='*70}\n",
            f"Input images:    {n_input}\n",
            f"Required images: {n_required} (including endpoints)\n"
        ], self.output)
        
        # Check input validity
        if n_input < 2:
            raise ValueError(f"Need at least 2 images (reactant + product), got {n_input}")
        
        if n_input > n_required:
            raise ValueError(
                f"Too many input images: got {n_input}, but n_images={self.params.n_images-2} "
                f"requires exactly {n_required} images (including endpoints). "
                f"Please reduce input images or increase n_images parameter."
            )

        self._retire_nebts_current()
        
        # Assign endpoints
        self.atoms_R = self.input_images[0]
        self.atoms_P = self.input_images[-1]
        
        # ===================================================================
        # CRITICAL: Align all input images BEFORE any interpolation/processing
        # This ensures interpolation happens in the correct coordinate system
        # ===================================================================
        log_info(["\nAligning input images to reactant reference...\n"], self.output)
        
        # Align product endpoint to reactant
        R_ref = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd_product, _, _ = kabsch_align(R_ref, P)
        self.atoms_P.set_positions(P_aligned)
        self.input_images[-1] = self.atoms_P  # Update in input list
        
        log_info([f"Product aligned to reactant. RMSD: {rmsd_product:.6f} Angstrom\n"], self.output)
        
        # Align all intermediate input images (if any exist)
        if n_input > 2:
            log_info([f"Aligning {n_input - 2} intermediate input image(s)...\n"], self.output)
            for i in range(1, n_input - 1):
                Q = to_numpy_f64(self.input_images[i].get_positions())
                Q_aligned, rmsd_i, _, _ = kabsch_align(R_ref, Q)
                self.input_images[i].set_positions(Q_aligned)
                log_info([f"  Image {i} aligned. RMSD: {rmsd_i:.6f} Angstrom\n"], self.output)
        
        log_info(["Input alignment completed.\n"], self.output)
        
        # ===================================================================
        # Determine if interpolation is needed
        # ===================================================================
        
        # Case 1: Exact number of images - use directly (already aligned)
        if n_input == n_required:
            log_info(["\nExact number of images provided. Using input directly.\n"], self.output)
            images = self.input_images
            
        # Case 2: Need to insert additional images via interpolation
        else:
            n_to_insert = n_required - n_input
            log_info([
                f"\nNeed to insert {n_to_insert} additional image(s).\n",
                "Analyzing distances to determine optimal insertion points...\n"
            ], self.output)
            
            # Compute straight-line distances between consecutive input images
            distances = self._compute_distances(self.input_images)
            
            log_info(["\nDistances between consecutive input images:\n"], self.output)
            for i, d in enumerate(distances):
                log_info([f"  Image {i} -> {i+1}: {d:.4f} Angstrom\n"], self.output)
            
            # Determine optimal insertion plan (prioritize largest gaps)
            insertion_plan = self._determine_insertion_plan(distances, n_to_insert)
            
            log_info(["\nInsertion plan:\n"], self.output)
            for seg_idx, count in sorted(insertion_plan, key=lambda x: x[0]):
                log_info([f"  Insert {count} image(s) between image {seg_idx} and {seg_idx+1}\n"], self.output)
            
            # Perform linear interpolation according to plan
            images = self._insert_images_by_plan(self.input_images, insertion_plan)
            
            log_info([f"\nTotal images after insertion: {len(images)}\n"], self.output)
            
            # Apply IDPP smoothing to refine interpolated path (optional)
            if self.params.ifidpp == 1:
                log_info(["\nApplying IDPP smoothing to interpolated images...\n"], self.output)
                images = self._run_idpp_smoothing(images)
                log_info(["IDPP smoothing completed.\n"], self.output)
            else:
                log_info(["\nIDPP smoothing disabled (ifidpp=0). Using linear interpolation only.\n"], self.output)
        
        # ===================================================================
        # CRITICAL: Re-align entire path after interpolation/IDPP
        # This ensures the complete path is properly aligned before NEB starts
        # ===================================================================
        log_info(["\nPerforming final Kabsch alignment of all images to reactant...\n"], self.output)
        self._align_path(images, ref_mode="reactant")
        log_info(["Final alignment completed.\n"], self.output)
        

        # ===================================================================
        # Step 1: Optional endpoint optimization
        # ===================================================================
        if self.params.initial_opt:
            self.atoms_R, self.atoms_P = self.optimize_endpoints(
                images[0], images[-1],
                f_max_th=self.params.neb_f_max_th,
                f_rms_th=self.params.neb_f_rms_th,
                max_iter=self.params.max_iter
            )
            images[0] = self.atoms_R
            images[-1] = self.atoms_P
        
        # ===================================================================
        # Step 2: Optimize one prepared base-stage image list
        # ===================================================================
        for image in images:
            if image.calc is None:
                image.calc = self.atoms_R.calc

        base_stage = run_neb_stage(
            images,
            params=self.params,
            stage_kind="base",
            output=self.output,
        )
        Es = list(base_stage.energies_hartree)
        raw_forces = list(base_stage.forces_hartree_per_angstrom)
        iteration = base_stage.budgets["used_iterations"]
        converged = base_stage.status is RunStatus.CONVERGED
        Fp_list, maxfp, hei = neb_forces(
            images,
            Es,
            use_dynamic_k=self.params.use_dynamic_k,
            k_min=self.params.k_min,
            k_max=self.params.k_max,
            k_decay=self.params.k_decay,
            raw_forces=raw_forces,
        )
        rmsfp = rms_force(Fp_list)
        self._k_springs_history = [
            compute_dynamic_k(
                Es,
                self.params.k_min,
                self.params.k_max,
                self.params.k_decay,
            )
        ] if self.params.use_dynamic_k else []
        traj_file = os.path.splitext(self.output)[0] + "_image_traj.xyz"

        def forces_info(forces):
            norms = np.linalg.norm(forces, axis=1)
            return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))

        maxF_R, rmsF_R = forces_info(raw_forces[0])
        maxF_P, rmsF_P = forces_info(raw_forces[-1])
        log_info([
            "\nProperties of fixed NEB end points:\n",
            f"Reactant E={Es[0]: .6f} Eh RMS(F)={rmsF_R: .6f} MAX(|F|)={maxF_R: .6f}\n",
            f"Product  E={Es[-1]: .6f} Eh RMS(F)={rmsF_P: .6f} MAX(|F|)={maxF_P: .6f}\n",
        ], self.output)
        if converged:
            log_info([f"\nNEB force criterion met after {iteration} iterations.\n"], self.output)
        else:
            log_info(["\nNEB optimization reached maximum iterations.\n"], self.output)


        # ---------------------------------------------------------------
        #  Final path summary
        # ---------------------------------------------------------------
        base, _ = os.path.splitext(self.output)
        mep_path = base + "_mep.xyz"
        hip_path = base + "_hei.xyz"
        write_xyz(mep_path, images, energies=Es)
        write_xyz(hip_path, [images[hei]], energies=[Es[hei]])

        # Compute straight-line distances
        distances = [np.linalg.norm(images[i+1].get_positions() - images[i].get_positions()) for i in range(len(images)-1)]

        kcal_per_Eh = 627.509
        summary = [
            "\n---------------------------------------------------------------\n",
            "                         PATH SUMMARY              \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom.\n\n",
            "Image Dist.(Ang.)    E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n"
        ]

        for i, (E, Fp) in enumerate(zip(Es, Fp_list)):
            dE = (E - Es[0]) * kcal_per_Eh
            dist = 0.0 if i == 0 else distances[i-1]
            maxF = np.max(np.linalg.norm(Fp, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(Fp, axis=1) ** 2))
            marker = " <= HEI" if i == hei else ""
            summary.append(f"{i:3d} {dist:9.3f} {E:13.5f} {dE:11.2f} {maxF:11.5f} {rmsF:10.5f}{marker}\n")

        summary.append("\nStraight line distance between images along the path:\n")
        for i, d in enumerate(distances):
            summary.append(f"        D({i:2d}-{i+1:2d}) = {d:7.4f} Ang.\n")

        summary.append(
            "\n---------------------------------------------------------------\n"
            "           INFORMATION ABOUT HIGHEST ENERGY IMAGE\n"
            "---------------------------------------------------------------\n"
            f"Highest energy image                      ....  {hei}\n"
            f"Energy                                    ....  {Es[hei]: .8f} Eh\n"
            f"Max. abs. force                           ....  {np.max(np.linalg.norm(Fp_list[hei], axis=1)) : .4e} Eh/Angstrom\n"
            "\n-----------------------------------------\n"
            "  HIGHEST ENERGY IMAGE (ANGSTROEM)\n"
            "-----------------------------------------\n"
        )
        summary.append(self.atoms_to_xyz(images[hei]))
        log_info(summary, self.output)

        log_info([
            f"\nWrote MEP to: {mep_path}\n",
            f"Wrote HEI to: {hip_path}\n",
            f"Wrote trajectory to: {traj_file}\n"
        ], self.output)
        
        
        self.base_stage_result = base_stage

        refinement_stage = None
        self.ci_stage_result = None
        self.nebts_stage_result = None
        self.prfo_result = None
        if self.params.refine == 'cineb' or self.params.refine == 'nebts':
            if not converged:
                log_info([
                    "\nNEB did not converge. Skipping CINEB refinement.\n",
                    "You may try to increase max_iter or check the initial path.\n"
                ], self.output)
            else:
                refinement_stage = self.restart_run(images, energies=Es)

        final_stage = (
            refinement_stage
            if isinstance(refinement_stage, PathStageResult)
            else base_stage
        )
        if self.ci_stage_result is not None:
            Es = list(self.ci_stage_result.energies_hartree)
            Fp_list = list(self._ci_projected_forces)
            maxfp = max(
                float(np.max(np.abs(force)))
                for force in Fp_list[1:-1]
            )
            rmsfp = rms_force(Fp_list)
        return {
            "status": final_stage.status.value,
            "converged": final_stage.status is RunStatus.CONVERGED,
            "termination_reason": final_stage.termination_reason,
            "iterations": iteration,
            "images": images,
            "energies": list(Es),
            "projected_forces": Fp_list,
            "max_projected_force": float(maxfp),
            "rms_projected_force": float(rmsfp),
            "prfo_result": self.prfo_result,
            "stage_result": final_stage,
            "base_stage": base_stage,
            "ci_stage": self.ci_stage_result,
            "nebts_stage": self.nebts_stage_result,
            "candidate": final_stage.candidate,
            "certification": final_stage.certification,
        }
