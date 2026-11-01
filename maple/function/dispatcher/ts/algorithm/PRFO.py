# -*- coding: utf-8 -*-
"""
PRFO (Partitioned Rational Function Optimization) implementation for TS search.
Features:
- Dual-shift PRFO with trust region (RS-PRFO)
- Mass-weighted coordinates for step computation
- Trust radius adaptation based on model agreement
- Mode-following for transition state optimization
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC
from ...curvature import (
    CurvatureValidationError,
    bofill_update,
    build_mass_weighted_mode_basis,
    resolve_hessian_precision_profile,
    validate_cartesian_hessian,
)
from ...result_contracts import (
    PRFOResult,
    RunStatus,
    SaddleCertification,
    geometry_sha256,
)

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

class NumericalStateError(ValueError):
    """Expected refusal for non-real or non-finite numerical state."""

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            raise NumericalStateError("Complex numerical state cannot be converted to float64.")
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            if np.iscomplexobj(arr):
                raise NumericalStateError("Complex numerical state cannot be converted to float64.")
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise NumericalStateError("Complex numerical state cannot be converted to float64.")
    return np.asarray(raw, dtype=np.float64)

def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def write_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, 
              iteration: Optional[int] = None):
    """
    Write a single geometry to XYZ file.
    
    Parameters
    ----------
    filename : str
        Output file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int, optional
        Iteration number to include in comment line
    """
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        
        # Build comment line
        comment_parts = []
        if iteration is not None:
            comment_parts.append(f"Iteration {iteration}")
        if energy is not None:
            comment_parts.append(f"Energy = {energy:.10f}")
        
        if comment_parts:
            f.write("  ".join(comment_parts) + "\n")
        else:
            f.write("TS optimization\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def write_xyz_atomic(filename: str, atoms: Atoms, **kwargs) -> None:
    temporary = filename + ".tmp"
    try:
        write_xyz(temporary, atoms, **kwargs)
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

def append_xyz_trajectory(filename: str, atoms: Atoms, energy: Optional[float] = None,
                         iteration: int = 0):
    """
    Append a geometry to XYZ trajectory file.
    
    Parameters
    ----------
    filename : str
        Output trajectory file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int
        Iteration number
    """
    # Remove file if first iteration
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "a") as f:
        f.write(f"{len(symbols)}\n")
        
        if energy is not None:
            f.write(f"Iteration {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iteration {iteration}\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------- PRFO Core -----------------------------------
# =============================================================================

def prfo_step(H, g, is_ts=False, target_mode=None, trust_radius=0.2,
              evals_eps=1e-10, mu_margin=1e-8, max_bisect_it=60,
              pre_eig=None):
    """
    Dual-shift PRFO trust-region step in the SAME coordinates as H,g (e.g., MW coords).

    Parameters
    ----------
    H : (n,n) array-like
        Hessian matrix
    g : (n,) array-like
        Gradient vector (NOT forces)
    is_ts : bool
        TS mode if True (enables partition into uphill/downhill subspaces)
    target_mode : int or None
        Uphill mode index in TS mode (defaults to most negative)
    trust_radius : float
        Trust radius bound measured in coordinates of H,g
    evals_eps : float
        Eigenvalue regularization threshold
    mu_margin : float
        Safety margin for bisection boundary
    max_bisect_it : int
        Maximum bisection iterations
    pre_eig : optional tuple
        Pre-computed eigendecomposition (w, V, g_proj) to reuse

    Returns
    -------
    s : (n,) array
        Step in the SAME coordinates as H,g
    """
    H = to_numpy_f64(H)
    if H.ndim == 3 and H.shape[0] == 1:
        H = H[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square 2D, got shape={H.shape}")
    n = H.shape[0]

    g = vec1d(g, n)

    # Eigendecomposition (or reuse)
    if pre_eig is not None:
        w, V, gp = pre_eig
        w = vec1d(w, n)
        V = to_numpy_f64(V)
        gp = vec1d(gp, n)
        if V.shape != (n, n):
            raise ValueError("pre_eig V has wrong shape.")
    else:
        w, V = np.linalg.eigh(H)
        gp = V.T @ g

    # Gentle regularization of tiny eigenvalues (keep sign if nonzero)
    tiny = (np.abs(w) < evals_eps)
    w = np.where(tiny & (w == 0.0), evals_eps, w)
    w = np.where(tiny & (w != 0.0), np.sign(w) * evals_eps, w)

    # Partition into uphill (minus set) and downhill (plus set)
    if is_ts:
        if target_mode is None:
            neg_idx = int(np.argmin(w))
            if w[neg_idx] < -1e-6:
                j = neg_idx
            else:
                j = int(np.argmax(np.abs(gp)))
        else:
            j = int(target_mode)
        minus_idx = np.array([j], dtype=int)
        plus_mask = np.ones(n, dtype=bool)
        plus_mask[minus_idx] = False
        plus_idx = np.where(plus_mask)[0]
    else:
        minus_idx = np.array([], dtype=int)
        plus_idx = np.arange(n, dtype=int)

    # Helper to compute unconstrained step & norm^2 for a subspace with sigma flip
    def unconstrained_component(idx, sigma_sign):
        """
        Compute unconstrained step components (μ=0) in eigen-basis.
        
        Parameters
        ----------
        idx : array of int
            Indices into eigen-basis for this subspace
        sigma_sign : float
            +1 for downhill, -1 for uphill (signature flip)
        
        Returns
        -------
        s_unc : array
            Unconstrained step components in eigen-basis on idx
        norm2_unc : float
            Squared norm of unconstrained step on idx
        w_tilde : array
            Flipped curvatures for bisection use
        num : array
            Flipped gradient components for bisection use
        """
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64), 0.0, np.zeros(0), np.zeros(0)
        w_sub = w[idx]
        gp_sub = gp[idx]
        w_tilde = sigma_sign * w_sub
        num = sigma_sign * gp_sub
        denom0 = np.where(np.abs(w_tilde) < evals_eps, 
                         np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = -num / denom0
        return s_unc, float(np.dot(s_unc, s_unc)), w_tilde, num

    # Uphill (minus) uses sigma = -1; Downhill (plus) uses sigma = +1
    s_unc_minus, norm2_unc_minus, wtil_minus, num_minus = \
        unconstrained_component(minus_idx, -1.0)
    s_unc_plus, norm2_unc_plus, wtil_plus, num_plus = \
        unconstrained_component(plus_idx, +1.0)

    # Total unconstrained norm^2 after partition
    total_unc = norm2_unc_minus + norm2_unc_plus
    R2 = trust_radius * trust_radius

    # Edge case: if total_unc already within R^2
    if total_unc <= R2:
        s_p = np.zeros(n, dtype=np.float64)
        if minus_idx.size:
            s_p[minus_idx] = s_unc_minus
        if plus_idx.size:
            s_p[plus_idx] = s_unc_plus
        return V @ s_p

    # Allocate trust radius between two subspaces
    if total_unc > 0.0:
        alpha = norm2_unc_minus / total_unc
    else:
        alpha = 0.5
    alpha_min, alpha_max = 0.05, 0.95
    alpha = float(np.clip(alpha, alpha_min, alpha_max))

    R2_minus = alpha * R2
    R2_plus = (1.0 - alpha) * R2

    # Solve two scalar bisections independently
    def solve_mu_for_norm2(target_norm2, w_tilde, num):
        """
        Find μ such that sum_i (num_i/(w_tilde_i - μ))^2 = target_norm2.
        Domain: μ < min(w_tilde). Monotonically increasing in this interval.
        """
        if num.size == 0:
            return 0.0, np.zeros(0, dtype=np.float64)

        # Unconstrained norm^2 (μ=0)
        denom0 = np.where(np.abs(w_tilde) < evals_eps,
                         np.sign(w_tilde) * evals_eps, w_tilde)
        s_unc = -num / denom0
        norm2_unc = float(np.dot(s_unc, s_unc))
        if norm2_unc <= target_norm2:
            return 0.0, s_unc

        # Bisection for μ
        def F(mu):
            denom = w_tilde - mu
            denom = np.where(np.abs(denom) < evals_eps,
                           np.sign(denom) * evals_eps, denom)
            return np.sum((num / denom) ** 2)

        wt_min = float(np.min(w_tilde))
        b = wt_min - mu_margin
        Fb = F(b)
        if (not np.isfinite(Fb)) or (Fb > 1e300):
            b = wt_min - 1e-4
            Fb = F(b)

        # Find a < b with F(a) < target_norm2
        a = b - 1.0
        Fa = F(a)
        it = 0
        while Fa > target_norm2 and it < 60:
            a -= max(1.0, abs(a) * 0.5)
            Fa = F(a)
            it += 1

        lo, hi = a, b
        for _ in range(max_bisect_it):
            mid = 0.5 * (lo + hi)
            Fm = F(mid)
            if Fm > target_norm2:
                hi = mid
            else:
                lo = mid
            if abs(Fm - target_norm2) <= 1e-12 * max(1.0, target_norm2) or \
               abs(hi - lo) < 1e-12:
                break
        mu_star = 0.5 * (lo + hi)

        denom = w_tilde - mu_star
        denom = np.where(np.abs(denom) < evals_eps,
                        np.sign(denom) * evals_eps, denom)
        s_part = -num / denom
        return mu_star, s_part

    # Solve for uphill and downhill parts
    mu_minus, s_part_minus = solve_mu_for_norm2(R2_minus, wtil_minus, num_minus)
    mu_plus, s_part_plus = solve_mu_for_norm2(R2_plus, wtil_plus, num_plus)

    # Assemble full eigen-basis step and rotate back
    s_p = np.zeros(n, dtype=np.float64)
    if minus_idx.size:
        s_p[minus_idx] = s_part_minus
    if plus_idx.size:
        s_p[plus_idx] = s_part_plus

    return V @ s_p

def calculate_Hessian(atoms: Atoms):
    """
    Calculate Hessian matrix from calculator.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with calculator attached
    
    Returns
    -------
    H : array
        Hessian matrix in float64
    """
    calc = atoms.calc
    before_numbers = atoms.get_atomic_numbers().copy()
    before_positions = atoms.get_positions().copy()
    before_cell = np.asarray(atoms.cell).copy()
    before_pbc = np.asarray(atoms.pbc).copy()
    try:
        hessian = np.asarray(calc.get_hessian(atoms))
    except ValueError as exc:
        message = str(exc)
        if "Hessian" in message and ("real" in message or "finite" in message):
            raise CurvatureValidationError(message) from exc
        raise
    mutated = (
        not np.array_equal(atoms.get_atomic_numbers(), before_numbers)
        or not np.array_equal(atoms.get_positions(), before_positions)
        or not np.array_equal(np.asarray(atoms.cell), before_cell)
        or not np.array_equal(np.asarray(atoms.pbc), before_pbc)
    )
    if mutated:
        atoms.set_atomic_numbers(before_numbers)
        atoms.set_positions(before_positions)
        atoms.set_cell(before_cell)
        atoms.set_pbc(before_pbc)
        raise CurvatureValidationError(
            "Hessian provider mutated the geometry during evaluation."
        )
    return hessian

def _bfgs_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BFGS update of Hessian approximation."""
    H = to_numpy_f64(H)
    s = vec1d(s)
    y = vec1d(y, s.size)
    ys = float(y.dot(s))
    if ys <= 1e-12:
        return H
    Hs = H.dot(s)
    sHs = float(s.dot(Hs))
    if sHs <= 1e-12:
        return H
    H_new = H + np.outer(y, y) / ys - np.outer(Hs, Hs) / sHs
    return 0.5 * (H_new + H_new.T)


# =============================================================================
# ------------------------------- PRFO Parameters -----------------------------
# =============================================================================

@dataclass
class PRFOParams:
    """Parameters for PRFO transition state search."""
    
    # Optimization control
    max_iter: int = 256                    # Maximum optimization iterations
    
    # Trust region parameters
    trust_radius: float = 0.2              # Initial trust radius (in MW coords)
    trust_min: float = 1e-3                # Minimum trust radius
    trust_max: float = 1.0                 # Maximum trust radius
    
    # Trust region adaptation thresholds
    eta_shrink: float = 0.75               # If rho < eta_shrink -> reject & shrink
    eta_expand: float = 1.75               # If rho > eta_expand and on boundary -> expand
    
    # PRFO step parameters
    evals_eps: float = 1e-10               # Eigenvalue regularization threshold
    mu_margin: float = 1e-8                # Safety margin for bisection
    max_bisect_it: int = 60                # Maximum bisection iterations
    recalc: int = 1                        # Exact Hessian recalculation interval
    hessian_update: str = "bofill"         # Working Hessian update: bofill or bfgs
    # Explicit because non-periodic does not imply lab-frame invariance.
    project_rigid_modes: bool = False       # Optimize an isolated molecule in 3N-rank(rigid)
    
    # Convergence thresholds (should be set from atoms object)
    f_max_th: float = 9.5e-3               # Maximum force threshold (Eh/Angstrom)
    f_rms_th: float = 5e-3                 # RMS force threshold (Eh/Angstrom)
    dp_max_th: float = 1.8e-3              # Maximum displacement threshold (Angstrom)
    dp_rms_th: float = 1.2e-3              # RMS displacement threshold (Angstrom)

# =============================================================================
# ------------------------------- PRFO Class ----------------------------------
# =============================================================================

class PRFO(JobABC):
    """
    Transition state search using Dual-Shift PRFO with trust region adaptation.

    The optimization is performed in mass-weighted coordinates for the step
    computation and trust-region enforcement, while geometry updates are done
    in Cartesian coordinates.
    """

    def __init__(self,
                 output: str,
                 atoms: Atoms,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms

        # Initialize params from paras dict
        self.params = self._init_params(PRFOParams, paras, ("prfo", "PRFO", "ts"))
        self.params.recalc = max(1, int(self.params.recalc))
        self.params.hessian_update = str(self.params.hessian_update).lower()
        if self.params.hessian_update not in {"bofill", "bfgs"}:
            raise ValueError("Hessian update method must be 'bofill' or 'bfgs'.")
        if not isinstance(self.params.project_rigid_modes, (bool, np.bool_)):
            raise TypeError("project_rigid_modes must be a boolean.")
        self.last_saddle_curvature = None

        # Override convergence thresholds from atoms if available
        for attr in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
            if hasattr(atoms, attr):
                setattr(self.params, attr, getattr(atoms, attr))

        # Mode tracking
        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None
    
    def atoms_to_xyz(self, atoms: Atoms) -> str:
        """Convert Atoms object to XYZ format string."""
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"
    
    def check_convergence(self, atoms: Atoms) -> bool:
        """
        Check if optimization has converged based on forces and displacements.
        
        Parameters
        ----------
        atoms : Atoms
            Current geometry with convergence metrics attached
        
        Returns
        -------
        bool
            True if all convergence criteria are met
        """
        return (atoms.max_f <= self.params.f_max_th and
                atoms.rms_f <= self.params.f_rms_th and
                atoms.max_dp <= self.params.dp_max_th and
                atoms.rms_dp <= self.params.dp_rms_th)
    
    def check_saddle_curvature(self, atoms: Atoms) -> SaddleCertification:
        """Require a fresh index-one Hessian before reporting TS convergence.

        Use the same mass metric and explicitly selected optimization subspace
        as the search. Rigid projection is not silently enabled for external
        fields or periodic systems. Constrained saddle certification is not
        implemented; force/displacement convergence cannot substitute for it.
        """
        self.last_saddle_curvature = None
        if bool(getattr(atoms, "constraints", None)):
            raise CurvatureValidationError(
                "PRFO saddle certification does not support constraints."
            )
        hessian_function = getattr(atoms.calc, "get_hessian", None)
        if not callable(hessian_function):
            raise CurvatureValidationError(
                "PRFO saddle certification requires get_hessian."
            )
        before_geometry = geometry_sha256(atoms)
        raw_hessian = calculate_Hessian(atoms)
        self.log_numerical_hessian_diagnostics(atoms.calc)
        profile = resolve_hessian_precision_profile(
            atoms,
            purpose="prfo-final-certification",
        )
        validated = validate_cartesian_hessian(atoms, raw_hessian, profile)
        modes = build_mass_weighted_mode_basis(
            atoms,
            validated,
            project_rigid_modes=self.params.project_rigid_modes,
        )
        frequencies = modes.frequencies_cm1
        negative_count = int(np.count_nonzero(frequencies < -10.0))
        self.last_saddle_curvature = {
            "source": "fresh-final-hessian",
            "positions_angstrom": atoms.get_positions().tolist(),
            "project_rigid_modes": bool(self.params.project_rigid_modes),
            "rigid_rank": modes.rigid_rank,
            "frequencies_cm1": frequencies.tolist(),
            "material_negative_threshold_cm1": -10.0,
            "negative_mode_count": negative_count,
            "raw_hessian_sha256": validated.raw_sha256,
            "raw_maximum_skew": validated.raw_maximum_skew,
            "raw_relative_skew": validated.raw_relative_skew,
            "precision_profile": profile.identity,
        }
        log_info(
            [f"\nFinal saddle curvature: {negative_count} material negative modes "
             "(required: exactly one below -10 cm^-1).\n"],
            self.output,
        )
        return SaddleCertification(
            geometry_sha256=before_geometry,
            raw_hessian_sha256=validated.raw_sha256,
            precision_profile=profile.identity,
            material_negative_mode_count=negative_count,
            force_displacement_converged=True,
            diagnostics=self.last_saddle_curvature,
            refusal_reason=(
                None if negative_count == 1 else "fresh_final_hessian_not_index_one"
            ),
        )

    def _refused_result(
        self,
        *,
        atoms: Atoms,
        iteration: int,
        reason: str,
        energy=None,
        forces=None,
        certification: SaddleCertification | None = None,
    ) -> PRFOResult:
        return PRFOResult.capture(
            status=RunStatus.REFUSED,
            atoms=atoms,
            iterations=iteration,
            max_iterations=self.params.max_iter,
            force_displacement_converged=bool(
                certification and certification.force_displacement_converged
            ),
            termination_reason=reason,
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces,
            certification=certification,
            output_paths={},
        )

    def _final_energy_forces(self, atoms: Atoms):
        before = geometry_sha256(atoms)
        energy = to_numpy_f64(atoms.get_potential_energy(force_consistent=True))
        forces = to_numpy_f64(atoms.get_forces())
        if geometry_sha256(atoms) != before:
            raise CurvatureValidationError(
                "Energy/force provider mutated the final geometry during evaluation."
            )
        if np.ndim(energy) != 0 or not np.isfinite(float(energy)):
            raise CurvatureValidationError("Final energy must be finite.")
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise CurvatureValidationError("Final forces must be finite with shape (N, 3).")
        return float(energy), forces

    def log_iteration(self, iteration: int, atoms: Atoms, E: float,
                     model_change: float, actual_change: float,
                     rho: Optional[float], trust_radius: float,
                     norm_mw: float, on_boundary: bool):
        """
        Log detailed information for current iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        atoms : Atoms
            Current geometry
        E : float
            Current energy
        model_change : float
            Predicted energy change by model
        actual_change : float
            Actual energy change
        rho : float or None
            Model agreement ratio
        trust_radius : float
            Current trust radius
        norm_mw : float
            Step norm in MW coordinates
        on_boundary : bool
            Whether step is on trust radius boundary
        """
        iter_str = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_str.center(70)}\n\n']
        
        # Coordinates section
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70 + '\n')
        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            coord = atom.position
            info_message.append(
                f"{atom_index:<4} {element_type:<2} "
                f"{coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n"
            )
        
        # Energy and convergence
        info_message.append(
            f"\n\nEnergy:                {E:>12.6f} "
            "Convergence criteria  Is converged \n"
        )
        
        # Force convergence
        if atoms.max_f > self.params.f_max_th:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_f > self.params.f_rms_th:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                Yes\n"
            )
        
        # Displacement convergence
        if atoms.max_dp > self.params.dp_max_th:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_dp > self.params.dp_rms_th:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                Yes\n"
            )
        
        # Trust region info
        rho_str = f"{rho:.3f}" if rho is not None else "nan"
        info_message.append(
            f"\nModel change: {model_change: .6e}  "
            f"Actual change: {actual_change: .6e}  "
            f"rho: {rho_str}\n"
        )
        info_message.append(
            f"Trust radius (MW): {trust_radius: .6f}  "
            f"Step norm (MW): {norm_mw: .6f}  "
            f"On boundary: {on_boundary}\n"
        )
        
        log_info(info_message, self.output)

    def update_mode_tracking(
        self,
        w_mw: np.ndarray,
        V_mw: np.ndarray,
        gp_mw: np.ndarray,
        mode_basis_mw: np.ndarray | None = None,
    ) -> int:
        """
        Update mode-following for transition state search.
        
        Parameters
        ----------
        w_mw : array
            Eigenvalues in mass-weighted coordinates
        V_mw : array
            Eigenvectors in mass-weighted coordinates
        gp_mw : array
            Projected gradient in eigen-basis
        mode_basis_mw : array, optional
            Orthonormal map from the current optimization coordinates into the
            ambient mass-weighted Cartesian space. Required for invariant mode
            tracking when the internal-coordinate complement changes.
        
        Returns
        -------
        int
            Index of tracked mode
        """
        if mode_basis_mw is None:
            ambient_modes_mw = V_mw
        else:
            mode_basis_mw = to_numpy_f64(mode_basis_mw)
            if mode_basis_mw.ndim != 2 or mode_basis_mw.shape[1] != V_mw.shape[0]:
                raise ValueError("mode_basis_mw is incompatible with eigenvectors.")
            ambient_modes_mw = mode_basis_mw @ V_mw

        if self.tracked_mode_vec_mw is None:
            # Initialize: track most negative mode or largest gradient
            neg_idx = int(np.argmin(w_mw))
            if w_mw[neg_idx] < -1e-6:
                self.tracked_mode_idx = neg_idx
            else:
                self.tracked_mode_idx = int(np.argmax(np.abs(gp_mw)))
            self.tracked_mode_vec_mw = ambient_modes_mw[:, self.tracked_mode_idx].copy()
        else:
            # Follow the physical ambient-space mode.  Reduced internal bases
            # rotate as the molecular geometry moves, so comparing their raw
            # coefficient vectors would not be coordinate invariant.
            overlaps = np.abs(ambient_modes_mw.T @ self.tracked_mode_vec_mw)
            self.tracked_mode_idx = int(np.argmax(overlaps))
            
            # Keep consistent sign to avoid flips
            selected_mode = ambient_modes_mw[:, self.tracked_mode_idx]
            sign_align = np.sign(np.dot(selected_mode, self.tracked_mode_vec_mw))
            if sign_align == 0.0:
                sign_align = 1.0
            self.tracked_mode_vec_mw = selected_mode * sign_align
        
        return self.tracked_mode_idx
    
    def run(self) -> PRFOResult:
        """
        Run PRFO transition state optimization.
        
        Returns
        -------
        PRFOResult
            Typed terminal state with certification evidence.
        """
        sys.setrecursionlimit(1000)
        
        atoms = self.atoms
        calculator = atoms.calc
        if getattr(calculator, "hessian", None) == "numerical":
            prepare = getattr(calculator, "prepare_numerical_derivatives", None)
            if callable(prepare):
                prepare()
        trust_radius = self.params.trust_radius
        
        iteration = 0
        
        # Setup trajectory file
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_prfo_traj.xyz"
        ts_file = base + "_prfo_ts.xyz"
        candidate_file = base + "_prfo_candidate.xyz"
        if os.path.exists(ts_file):
            os.remove(ts_file)
        
        # Log header
        info_message = [
            "\nStarting Transition State Search (TS) with RS-PRFO...\n",
            (
                f"Hessian recalc interval: {self.params.recalc}; "
                f"update method: {self.params.hessian_update}\n"
            ),
            (
                f"Trust radius adaptation: eta_shrink={self.params.eta_shrink}, "
                f"eta_expand={self.params.eta_expand}\n"
            ),
            (
                "Rigid-mode projection: "
                f"{'enabled (explicit isolated-molecule mode)' if self.params.project_rigid_modes else 'disabled'}\n"
            ),
            (
                "Convergence thresholds: "
                f"f_max={self.params.f_max_th:.6f}, "
                f"f_rms={self.params.f_rms_th:.6f}, "
                f"dp_max={self.params.dp_max_th:.6f}, "
                f"dp_rms={self.params.dp_rms_th:.6f}\n"
            ),
        ]
        log_info(info_message, self.output)

        H_work = None
        force_displacement_converged = False
        curvature_checked = False
        
        # Main optimization loop
        while iteration < self.params.max_iter:
            # Get current geometry and energy/forces
            X = atoms.get_positions().reshape(-1, 3)
            try:
                E_old = to_numpy_f64(
                    atoms.get_potential_energy(force_consistent=True)
                )
                F_cart = to_numpy_f64(atoms.get_forces())
            except NumericalStateError as exc:
                return self._refused_result(
                    atoms=atoms,
                    iteration=iteration,
                    reason=str(exc),
                )
            if np.ndim(E_old) != 0 or not np.isfinite(float(E_old)):
                return self._refused_result(
                    atoms=atoms,
                    iteration=iteration,
                    reason="Final energy must be finite.",
                )
            if F_cart.shape != (len(atoms), 3) or not np.isfinite(F_cart).all():
                return self._refused_result(
                    atoms=atoms,
                    iteration=iteration,
                    reason="Final forces must be finite with shape (N, 3).",
                    energy=float(E_old),
                )
            g_cart = vec1d(-F_cart)
            
            need_recalc = (
                H_work is None
                or (iteration % self.params.recalc == 0)
            )
            if need_recalc:
                try:
                    raw_hessian = calculate_Hessian(atoms)
                    self.log_numerical_hessian_diagnostics(calculator)
                    profile = resolve_hessian_precision_profile(
                        atoms,
                        purpose="prfo-search",
                    )
                    validated = validate_cartesian_hessian(atoms, raw_hessian, profile)
                except CurvatureValidationError as exc:
                    return self._refused_result(
                        atoms=atoms,
                        iteration=iteration,
                        reason=str(exc),
                        energy=float(E_old),
                        forces=F_cart,
                    )
                modes = build_mass_weighted_mode_basis(
                    atoms,
                    validated,
                    project_rigid_modes=self.params.project_rigid_modes,
                )
                H_cart = validated.physical_hessian
                H_work = H_cart.copy()
            else:
                H_cart = H_work
                profile = resolve_hessian_precision_profile(
                    atoms,
                    purpose="prfo-updated-search",
                )
                validated = validate_cartesian_hessian(atoms, H_cart, profile)
                modes = build_mass_weighted_mode_basis(
                    atoms,
                    validated,
                    project_rigid_modes=self.params.project_rigid_modes,
                )
            
            n3 = H_cart.shape[0]
            if g_cart.size != n3:
                raise ValueError(
                    f"Gradient size {g_cart.size} != Hessian dim {n3}"
                )
            
            # Eigenvalues for logging
            eigvals_log, _ = np.linalg.eigh(validated.physical_hessian)
            eigvals_log = np.real(eigvals_log).astype(np.float64).squeeze()
            
            # Mass-weighting
            D = modes.inverse_sqrt_mass
            
            g_mw = vec1d(D * g_cart, n3)
            H_mw = modes.weighted_hessian

            optimization_basis_mw = None
            if self.params.project_rigid_modes:
                optimization_basis_mw = modes.basis
                if optimization_basis_mw.shape[1] == 0:
                    raise ValueError(
                        "PRFO rigid-mode projection left no internal optimization "
                        "coordinates."
                    )
                g_opt_mw = vec1d(optimization_basis_mw.T @ g_mw)
                H_opt_mw = optimization_basis_mw.T @ H_mw @ optimization_basis_mw
                H_opt_mw = 0.5 * (H_opt_mw + H_opt_mw.T)
            else:
                # Preserve the historical full-space path exactly and avoid
                # cubic identity-matrix products when projection is disabled.
                g_opt_mw = g_mw
                H_opt_mw = H_mw
            
            # Eigendecomposition in the selected MW optimization coordinates.
            # With projection disabled this is exactly the historical full 3N
            # problem.  With projection enabled no artificial rigid-mode
            # eigenvalues enter mode following or the trust-region step.
            w_mw, V_mw = np.linalg.eigh(H_opt_mw)
            n_opt = H_opt_mw.shape[0]
            gp_mw = vec1d(V_mw.T @ g_opt_mw, n_opt)
            
            # Regularize tiny eigenvalues
            tiny = (np.abs(w_mw) < 1e-10)
            w_mw = np.where(tiny & (w_mw == 0.0), 1e-10, w_mw)
            w_mw = np.where(tiny & (w_mw != 0.0),
                          np.sign(w_mw) * 1e-10, w_mw)
            
            # Update mode tracking
            tracked_mode_idx = self.update_mode_tracking(
                w_mw,
                V_mw,
                gp_mw,
                mode_basis_mw=optimization_basis_mw,
            )
            
            # RS loop: try step with current trust_radius, accept/reject by rho
            accepted = False
            max_attempts = 8
            attempts = 0
            
            while not accepted and attempts < max_attempts:
                attempts += 1
                
                # Compute PRFO step in MW coords
                s_opt_mw = prfo_step(
                    H=H_opt_mw,
                    g=g_opt_mw,
                    is_ts=True,
                    target_mode=tracked_mode_idx,
                    trust_radius=trust_radius,
                    evals_eps=self.params.evals_eps,
                    mu_margin=self.params.mu_margin,
                    max_bisect_it=self.params.max_bisect_it,
                    pre_eig=(w_mw, V_mw, gp_mw)
                )
                if optimization_basis_mw is None:
                    s_mw = vec1d(s_opt_mw, n3)
                else:
                    s_mw = vec1d(optimization_basis_mw @ s_opt_mw, n3)
                norm_mw = float(np.linalg.norm(s_mw))
                on_boundary = (abs(norm_mw - trust_radius) <=
                             1e-6 * max(1.0, trust_radius))
                
                # Back to Cartesian
                s_cart = vec1d(D * s_mw, n3)
                
                # Model agreement for trust-radius adaptation
                Hs = H_cart @ s_cart
                model_change = float(g_cart.dot(s_cart) + 0.5 * s_cart.dot(Hs))
                
                # Trial geometry
                X_new = X.reshape(-1, 3) + s_cart.reshape(-1, 3)
                atoms.set_positions(X_new)
                
                # New energy
                E_new = to_numpy_f64(
                    atoms.get_potential_energy(force_consistent=True)
                )
                
                actual_change = float(E_new - E_old)
                rho = None
                if abs(model_change) > 1e-16:
                    rho = actual_change / model_change
                
                # Accept/reject decision
                bad_model = (rho is None or rho < self.params.eta_shrink or
                           not np.isfinite(rho))
                
                if bad_model and trust_radius > self.params.trust_min * (1.0 + 1e-12):
                    # Reject: rollback geometry, shrink radius, retry
                    atoms.set_positions(X)
                    trust_radius = max(self.params.trust_min,
                                     0.5 * trust_radius)
                    continue
                else:
                    # Accept the step
                    accepted = True
                    
                    # Radius adaptation after acceptance
                    if (rho is not None and rho > self.params.eta_expand and
                        on_boundary):
                        trust_radius = min(self.params.trust_max,
                                         2.0 * trust_radius)
                    
                    # Get new forces for convergence check
                    F_new = to_numpy_f64(atoms.get_forces())
                    g_new_cart = vec1d(-F_new, n3)
                    if not need_recalc:
                        y_cart = g_new_cart - g_cart
                        if self.params.hessian_update == "bfgs":
                            H_work = _bfgs_update(H_work, s_cart, y_cart)
                        else:
                            H_work = bofill_update(H_work, s_cart, y_cart)
                    
                    # Compute convergence metrics (per DOF RMS)
                    dof = s_cart.size
                    atoms.max_dp = abs(s_cart).max()
                    atoms.rms_dp = np.sqrt((s_cart**2).sum() / dof)
                    atoms.max_f = abs(F_new).max()
                    atoms.rms_f = np.sqrt((F_new**2).sum() / dof)
                    
                    # Log iteration
                    self.log_iteration(iteration + 1, atoms, E_new,
                                     model_change, actual_change, rho,
                                     trust_radius, norm_mw, on_boundary)
                    
                    # Write to trajectory
                    append_xyz_trajectory(traj_file, atoms, energy=E_new,
                                        iteration=iteration + 1)
                    
                    # Check convergence
                    force_displacement_converged = self.check_convergence(atoms)
                    if force_displacement_converged:
                        curvature_checked = True
                    if force_displacement_converged:
                        try:
                            final_energy, final_forces = self._final_energy_forces(atoms)
                            certification = self.check_saddle_curvature(atoms)
                        except CurvatureValidationError as exc:
                            return self._refused_result(
                                atoms=atoms,
                                iteration=iteration + 1,
                                reason=str(exc),
                            )
                    if force_displacement_converged and certification.certified_ts:
                        info_message = [
                            '\n\n' + '-' * 70 + '\n',
                            f'{"Normal Termination".center(70)}\n\n'
                        ]
                        log_info(info_message, self.output)
                        
                        result = PRFOResult.capture(
                            status=RunStatus.CONVERGED,
                            atoms=atoms,
                            iterations=iteration + 1,
                            max_iterations=self.params.max_iter,
                            force_displacement_converged=True,
                            termination_reason="force_displacement_and_index_one",
                            energy_hartree=final_energy,
                            forces_hartree_per_angstrom=final_forces,
                            certification=certification,
                            output_paths={"trajectory": traj_file, "certified_ts": ts_file},
                        )
                        write_xyz_atomic(
                            ts_file,
                            result.atoms,
                            energy=result.energy_hartree,
                            iteration=iteration + 1,
                        )
                        return result
                    if force_displacement_converged:
                        result = PRFOResult.capture(
                            status=RunStatus.REFUSED,
                            atoms=atoms,
                            iterations=iteration + 1,
                            max_iterations=self.params.max_iter,
                            force_displacement_converged=True,
                            termination_reason="fresh_final_hessian_not_index_one",
                            energy_hartree=final_energy,
                            forces_hartree_per_angstrom=final_forces,
                            certification=certification,
                            output_paths={"trajectory": traj_file, "candidate": candidate_file},
                        )
                        write_xyz_atomic(
                            candidate_file,
                            result.atoms,
                            energy=result.energy_hartree,
                            iteration=iteration + 1,
                        )
                        return result
            
            iteration += 1
        
        # Maximum iterations reached
        log_info([f'\n\n{"Maximum Iterations Reached".center(70)}\n\n'],
                self.output)
        
        # Preserve a candidate without naming it as a certified transition state.
        try:
            E_final, F_final = self._final_energy_forces(atoms)
        except CurvatureValidationError as exc:
            return self._refused_result(
                atoms=atoms,
                iteration=iteration,
                reason=str(exc),
            )
        
        log_info([
            f"\nWrote trajectory to: {traj_file}\n",
            f"Wrote uncertified candidate structure to: {candidate_file}\n"
        ], self.output)
        result = PRFOResult.capture(
            status=RunStatus.BOUNDED_NONCONVERGED,
            atoms=atoms,
            iterations=iteration,
            max_iterations=self.params.max_iter,
            force_displacement_converged=force_displacement_converged,
            termination_reason="maximum_iterations",
            energy_hartree=E_final,
            forces_hartree_per_angstrom=F_final,
            certification=None,
            output_paths={"trajectory": traj_file, "candidate": candidate_file},
        )
        write_xyz_atomic(
            candidate_file,
            result.atoms,
            energy=result.energy_hartree,
            iteration=iteration,
        )
        return result
