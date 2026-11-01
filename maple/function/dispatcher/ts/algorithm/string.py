# -*- coding: utf-8 -*-
"""
GSM (Growing String Method, engineering version) with:
- Kabsch alignment (reports RMSD)
- Two-ended adaptive growth + constrained mini-relax (LBFGS, projected onto tau-orthogonal subspace)
- Energy-weighted tangent for robust direction estimation
- Dynamic growth policy (pause side / refine endpoint / adapt step)
- Merge when close; re-sample to fixed n_images (default 9 incl. endpoints)
- 2007 full-force String evolution followed by equal-arc reparameterization
- Fixed-tangent CI-String with split reparameterization around the CI anchor
- Optional PRFO TS refinement only after converged CI-String

The ordinary and climbing stages follow E, Ren, and Vanden-Eijnden,
J. Chem. Phys. 126, 164103 (2007), Eqs. (6), (10)-(13), and (22).

Units:
- Energies in Eh
- Forces in Eh/Angstrom
- Distances in Angstrom
"""

from __future__ import annotations
from contextlib import ExitStack
import os
from dataclasses import dataclass, fields
from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms

# You already have these utilities / mixins in your codebase:
from .logger import log_info
from .path_optimizer import LBFGSDriver
from ...curvature import CurvatureValidationError
from ...jobABC import JobABC
from ...numerical_state import (
    NumericalStateError,
    scientific_atoms_transaction,
    to_numpy_f64,
)
from ...result_contracts import PathStageResult, PRFOResult, RunStatus


# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Rigid-body least-squares alignment (Kabsch):
    Align Q onto P. Both are (N,3). Returns: Q_aligned, rmsd, R, t
    """
    P = np.asarray(P, dtype=np.float64); Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3)")
    Pc = P.mean(axis=0); Qc = Q.mean(axis=0)
    P0 = P - Pc; Q0 = Q - Qc
    C = Q0.T @ P0
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(V @ Wt) < 0.0:
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


def inherit_attrs(src: Atoms, dst: Atoms):
    """
    Ensure new images/TS carry calculator and threshold attributes.
    Called immediately after we create/copy a new Atoms.
    """
    if getattr(src, "calc", None) is not None:
        dst.calc = src.calc
    # user thresholds set by Dispatcher.set_threshold()
    for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def atoms_to_xyz_block(atoms: Atoms) -> str:
    """Pretty XYZ (Angstrom) block used in logs."""
    lines = []
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    for s, (x, y, z) in zip(syms, pos):
        lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
    return "\n".join(lines) + "\n"


def get_energies(images: List[Atoms]) -> List[float]:
    return [float(at.get_potential_energy(force_consistent=True)) for at in images]


def rms_force_perp(Fp_list: List[np.ndarray]) -> float:
    """RMS of projected-perp forces over all DOFs of internal images."""
    arrs = []
    for i, Fp in enumerate(Fp_list):
        if i == 0 or i == len(Fp_list) - 1:
            continue
        arrs.append(Fp.reshape(-1))
    if not arrs:
        return 0.0
    v = np.concatenate(arrs)
    return float(np.sqrt(np.mean(v * v)))


def project_out_rigidbody_forces(forces: np.ndarray,
                                 positions: np.ndarray,
                                 masses: np.ndarray,
                                 tol: float = 1e-10) -> np.ndarray:
    """
    Remove 6 rigid-body modes (3 translations + 3 rotations) from forces
    using mass-weighted Eckart projection. Safe for near-linear molecules:
    rank-deficiency is handled via SVD cutoff.

    Parameters
    ----------
    forces : (N,3) array, true Cartesian forces (Eh/Ang)
    positions : (N,3) array, current Cartesian coords (Ang)
    masses : (N,) array, atomic masses (amu)
    tol : float, relative SVD cutoff

    Returns
    -------
    F_proj : (N,3) array, forces with rigid-body components removed
    """
    N = positions.shape[0]
    sqrtm = np.sqrt(masses).reshape(-1, 1)

    R_cm = (positions * masses.reshape(-1,1)).sum(axis=0) / masses.sum()
    r = positions - R_cm

    B = np.zeros((3 * N, 6), dtype=np.float64)
    # Translations
    for a in range(N):
        s = float(sqrtm[a, 0]); i3 = 3*a
        B[i3+0, 0] = s; B[i3+1, 1] = s; B[i3+2, 2] = s
    # Rotations (about x,y,z): omega × r (mass-weighted)
    for a in range(N):
        s = float(sqrtm[a, 0]); x,y,z = r[a]; i3 = 3*a
        B[i3+0, 3] = 0.0; B[i3+1, 3] = -s*z; B[i3+2, 3] = s*y
        B[i3+0, 4] = s*z; B[i3+1, 4] = 0.0; B[i3+2, 4] = -s*x
        B[i3+0, 5] = -s*y; B[i3+1, 5] = s*x; B[i3+2, 5] = 0.0

    Fmw = (forces * sqrtm).reshape(-1)
    U, S, VT = np.linalg.svd(B, full_matrices=False)
    if S.size == 0 or S.max() <= 0.0:
        return forces.copy()
    rank = int(np.sum(S > tol * S.max()))
    Q = U[:, :rank]
    Fmw_proj = Fmw - Q @ (Q.T @ Fmw)
    return (Fmw_proj.reshape(-1, 3) / sqrtm)


def align_path_inplace(images, mode: str = "chain"):
    """
    Align all internal images to suppress rigid rotations along the path.

    mode="chain":  image i aligned to (i-1), propagates smoothly.
    mode="first":  every internal image aligned to image 0.

    In-place: modifies images[i].positions
    """
    assert len(images) >= 2
    if mode not in ("chain", "first"):
        mode = "chain"
    if mode == "first":
        ref = images[0].get_positions()
        for i in range(1, len(images) - 1):
            Pi = images[i].get_positions()
            aligned, _, _, _ = kabsch_align(ref, Pi)
            images[i].set_positions(aligned)
    else:
        ref = images[0].get_positions()
        for i in range(1, len(images) - 1):
            Pi = images[i].get_positions()
            aligned, _, _, _ = kabsch_align(ref, Pi)
            images[i].set_positions(aligned)
            ref = aligned


def linear_reparam(images: List[Atoms]):
    """
    Reparameterize a path to equal arclength by piecewise-linear resampling.
    Uses the CURRENT (already aligned) coordinates. Endpoints fixed.

    In-place on images: updates internal images' positions.
    """
    M = len(images)
    if M <= 2:
        return
    X = [img.get_positions().copy() for img in images]
    s = [0.0]
    for i in range(1, M):
        ds = np.linalg.norm(X[i] - X[i-1])
        s.append(s[-1] + ds)
    L = s[-1]
    if L <= 1e-15:
        return
    s_targets = np.linspace(0.0, L, M)
    for k in range(1, M - 1):
        sk = s_targets[k]
        j = np.searchsorted(s, sk) - 1
        if j < 0: j = 0
        if j >= M - 1: j = M - 2
        if s[j+1] - s[j] < 1e-16:
            lam = 0.0
        else:
            lam = (sk - s[j]) / (s[j+1] - s[j])
        Xk = (1.0 - lam) * X[j] + lam * X[j+1]
        images[k].set_positions(Xk)


# =============================================================================
# ------------------------------- GSM Params ----------------------------------
# =============================================================================

@dataclass
class GSMParams:
    # ---- Growth (engineering) ----
    grow_step: float = 0.12               # initial step length (Ang)
    grow_step_min: float = 0.02           # minimal step to accept (Ang)
    grow_backtrack: float = 0.5           # backtracking factor for step
    grow_relax_iters: int = 10            # mini-relax LBFGS iterations (orthogonal)
    grow_relax_tol: float = 2.5e-3        # |grad_perp| threshold for local convergence
    grow_energy_tol: float = 1.0e-3       # forbid large uphill during prediction (Eh)
    pause_imbalance: int = 2              # if |nL - nR| > this, pause longer side
    refine_end_every: int = 3             # every K growth iters, re-opt last image (orthogonal)

    # merge
    merge_threshold: float = 0.5         # mean per-atom distance (Ang) to merge

    # target path points after merge (incl. endpoints)
    n_images: int = 9

    # ---- 2007 full-force String relaxation ----
    max_iter_relax: int = 300
    string_f_max_th: float = 5.0e-3
    string_f_rms_th: float = 1.0e-3
    lbfgs_memory: int = 7
    lbfgs_curvature: float = 70.0
    lbfgs_max_step: float = 0.2
    reparam_every: int = 1

    # ---- CI-STRING ----
    refine: Optional[str] = None          # 'cistring' or 'stringts' or None
    max_iter_cistring: int = 300
    cistring_f_max_th: float = 4.0e-3
    cistring_f_rms_th: float = 2.0e-3

    # ---- Optional PRFO after a converged CI-String ----
    prfo_params: Optional[dict] = None
    project_rigid_modes: bool = False

    # ---- Safety / debug ----
    max_iter_grow: int = 100              # total growth rounds (each ideally adds 2 nodes)
    log_every: int = 1                    # growth log cadence

    # guiding policy (safer growth)
    guide_conn_min: float = 0.3   # ≥ this weight for connector even when far apart
    guide_d_switch: float = 4.0    # distance scale for mixing connector vs. tangent (Å)
    guide_inc_trigger: int = 2     # force a connector-pull after this many worsening steps


STRING_PARAMETER_METADATA = {
    "all_fields": frozenset(field.name for field in fields(GSMParams)),
    "field_defaults": {
        field.name: getattr(GSMParams(), field.name) for field in fields(GSMParams)
    },
    "stage_fields": {
        "growth": {
            "grow_step", "grow_step_min", "grow_backtrack", "grow_relax_iters",
            "grow_relax_tol", "grow_energy_tol", "pause_imbalance",
            "refine_end_every", "merge_threshold", "n_images", "max_iter_grow",
            "log_every", "guide_conn_min", "guide_d_switch", "guide_inc_trigger",
        },
        "ordinary": {
            "max_iter_relax", "string_f_max_th", "string_f_rms_th",
            "lbfgs_memory", "lbfgs_curvature", "lbfgs_max_step", "reparam_every",
        },
        "ci": {
            "max_iter_cistring", "cistring_f_max_th", "cistring_f_rms_th",
        },
        "ts": {"prfo_params", "project_rigid_modes"},
    },
    "inactive_rules": {
        "refine=None": "ordinary, CI, and TS fields are inactive",
        "refine=cistring": "TS fields are inactive",
        "refine=stringts": "all String stages are active",
    },
    "cross_field_rules": (
        "refine must be None, cistring, or stringts",
        "refined strings require n_images >= 3",
        "2007 String refinement requires reparam_every == 1",
        "project_rigid_modes is explicit and cannot also occur in prfo_params",
        "PRFO starts only after a converged CI-String stage",
    ),
}


@dataclass(frozen=True)
class StringResiduals:
    regular_max: float
    regular_rms: float
    ci_max: Optional[float] = None
    ci_rms: Optional[float] = None



# =============================================================================
# --------------------------------- GSM Class ---------------------------------
# =============================================================================

class GSM(JobABC):
    """
    Engineering GSM:
      - Adaptive predictor (step backtracking on uphill / |F_perp|)
      - Constrained mini-relax in tau-orthogonal subspace (LBFGS)
      - Energy-weighted tangent
      - Dynamic growth policy
      - Full physical-force String relaxation followed by reparameterization
      - Fixed-tangent climbing image with split left/right reparameterization
    """

    def __init__(self, output: str, atoms_R: Atoms, atoms_P: Atoms,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms_R = atoms_R
        self.atoms_P = atoms_P
        self.atoms_R.calc = atoms_R.calc
        self.atoms_P.calc = atoms_P.calc

        # Initialize params from paras dict
        self.params = self._init_params(GSMParams, paras, ("gsm", "GSM", "string", "STRING", "ts"))
        self._validate_params()

    def _validate_params(self):
        p = self.params
        if p.refine is not None:
            p.refine = str(p.refine).lower()
        if p.refine not in {None, "cistring", "stringts"}:
            raise ValueError("refine must be None, 'cistring', or 'stringts'.")
        if p.refine is not None and p.reparam_every != 1:
            raise ValueError(
                "reparam_every must be 1 for the audited 2007 String split iteration."
            )
        if p.refine is not None and p.n_images < 3:
            raise ValueError("CI-String refinement requires n_images >= 3.")
        for name in (
            "grow_relax_iters", "pause_imbalance", "refine_end_every",
            "max_iter_grow", "max_iter_relax", "max_iter_cistring",
            "guide_inc_trigger",
        ):
            value = getattr(p, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if type(p.log_every) is not int or p.log_every < 1:
            raise ValueError("log_every must be a positive integer.")
        if type(p.n_images) is not int or p.n_images < 2:
            raise ValueError("n_images must be an integer >= 2.")
        if type(p.lbfgs_memory) is not int or p.lbfgs_memory < 0:
            raise ValueError("lbfgs_memory must be a non-negative integer.")
        if type(p.reparam_every) is not int or p.reparam_every < 1:
            raise ValueError("reparam_every must be a positive integer.")
        for name in (
            "grow_step", "grow_step_min", "grow_relax_tol", "grow_energy_tol",
            "merge_threshold", "guide_d_switch",
            "string_f_max_th", "string_f_rms_th", "cistring_f_max_th",
            "cistring_f_rms_th", "lbfgs_curvature", "lbfgs_max_step",
        ):
            raw_value = getattr(p, name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric.")
            value = to_numpy_f64(raw_value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            setattr(p, name, value)
        if isinstance(p.grow_backtrack, (bool, np.bool_)):
            raise TypeError("grow_backtrack must be numeric.")
        p.grow_backtrack = to_numpy_f64(p.grow_backtrack)
        if not 0.0 < p.grow_backtrack < 1.0:
            raise ValueError("grow_backtrack must be between 0 and 1.")
        if isinstance(p.guide_conn_min, (bool, np.bool_)):
            raise TypeError("guide_conn_min must be numeric.")
        p.guide_conn_min = to_numpy_f64(p.guide_conn_min)
        if not 0.0 <= p.guide_conn_min <= 1.0:
            raise ValueError("guide_conn_min must be between 0 and 1 inclusive.")
        if p.grow_step_min > p.grow_step:
            raise ValueError("grow_step_min must not exceed grow_step.")
        if not isinstance(p.project_rigid_modes, (bool, np.bool_)):
            raise TypeError("project_rigid_modes must be a boolean.")
        if p.prfo_params is None:
            p.prfo_params = {}
        if not isinstance(p.prfo_params, dict):
            raise TypeError("prfo_params must be a dictionary.")
        normalized_prfo = {}
        origins = {}
        for key, value in p.prfo_params.items():
            if not isinstance(key, str):
                raise TypeError("prfo_params keys must be strings.")
            normalized = key.lower()
            if normalized in origins:
                raise ValueError(
                    "Conflicting PRFO parameter spellings: "
                    f"{origins[normalized]!r} and {key!r}."
                )
            origins[normalized] = key
            normalized_prfo[normalized] = value
        p.prfo_params = normalized_prfo
        if "project_rigid_modes" in p.prfo_params:
            raise ValueError(
                "project_rigid_modes must be supplied explicitly as a String parameter, "
                "not duplicated in prfo_params."
            )
        if p.prfo_params:
            from .PRFO import PRFOParams

            known = {field.name.lower() for field in fields(PRFOParams)}
            unknown = sorted(
                str(key) for key in p.prfo_params
                if str(key).lower() not in known
            )
            if unknown:
                raise ValueError(f"unknown PRFO parameter: {', '.join(unknown)}")
            if p.refine != "stringts":
                raise ValueError("prfo_params are active only when refine='stringts'.")

    # --------------------------- helpers (geometry) ---------------------------

    @staticmethod
    def _copy_with_inherit(src: Atoms) -> Atoms:
        out = src.copy()
        inherit_attrs(src, out)
        if out.calc is None and src.calc is not None:
            out.calc = src.calc
        return out

    @staticmethod
    def _evaluate_images(images: List[Atoms]) -> Tuple[List[float], List[np.ndarray]]:
        from maple.function.dispatcher.implicit_path import evaluate_path_images
        with ExitStack() as stack:
            for image in images:
                stack.enter_context(
                    scientific_atoms_transaction(image, operation="String energy/force")
                )
            energies, forces = evaluate_path_images(images, label="String")
        return (
            [to_numpy_f64(energy) for energy in energies],
            [to_numpy_f64(force) for force in forces],
        )

    @classmethod
    def _evaluate_image(cls, image: Atoms) -> Tuple[float, np.ndarray]:
        energies, forces = cls._evaluate_images([image])
        return float(energies[0]), to_numpy_f64(forces[0])

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return v * 0.0
        return v / n

    @staticmethod
    def _min_image_distance(A: Atoms, B: Atoms) -> float:
        """Mean per-atom distance between two images (Ang)."""
        pa = to_numpy_f64(A.get_positions()); pb = to_numpy_f64(B.get_positions())
        return float(np.linalg.norm((pa - pb), axis=1).mean())

    @classmethod
    def _copy_images(cls, images: List[Atoms]) -> List[Atoms]:
        return [cls._copy_with_inherit(image) for image in images]

    @staticmethod
    def _positions_matrix(images: List[Atoms]) -> np.ndarray:
        return np.array(
            [to_numpy_f64(image.get_positions()).reshape(-1) for image in images],
            dtype=np.float64,
        )

    @staticmethod
    def _set_positions_matrix(images: List[Atoms], positions: np.ndarray) -> None:
        for image, value in zip(images, positions):
            image.set_positions(value.reshape(-1, 3))

    @staticmethod
    def _equal_arc_positions(positions: np.ndarray) -> np.ndarray:
        q = to_numpy_f64(positions)
        if q.ndim != 2 or len(q) < 2:
            raise ValueError("String positions must have shape (images>=2, coordinates).")
        segment = np.linalg.norm(np.diff(q, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment)))
        if cumulative[-1] <= 1.0e-15:
            return q.copy()
        keep = np.concatenate(([True], np.diff(cumulative) > 0.0))
        target = np.linspace(0.0, cumulative[-1], len(q))
        result = np.empty_like(q)
        for axis in range(q.shape[1]):
            result[:, axis] = np.interp(
                target,
                cumulative[keep],
                q[keep, axis],
            )
        result[0] = q[0]
        result[-1] = q[-1]
        return result

    @classmethod
    def _equal_arc_reparameterize(cls, images: List[Atoms]) -> None:
        cls._set_positions_matrix(
            images,
            cls._equal_arc_positions(cls._positions_matrix(images)),
        )

    @classmethod
    def _split_reparameterize(cls, images: List[Atoms], ci_index: int) -> None:
        if not 0 < ci_index < len(images) - 1:
            raise ValueError("CI index must identify an interior String image.")
        positions = cls._positions_matrix(images)
        left = positions[0].copy()
        ci = positions[ci_index].copy()
        right = positions[-1].copy()
        positions[: ci_index + 1] = cls._equal_arc_positions(
            positions[: ci_index + 1]
        )
        positions[ci_index:] = cls._equal_arc_positions(positions[ci_index:])
        positions[0] = left
        positions[ci_index] = ci
        positions[-1] = right
        cls._set_positions_matrix(images, positions)

    @classmethod
    def _path_tangent(cls, images: List[Atoms], index: int) -> np.ndarray:
        positions = cls._positions_matrix(images)
        tangent = positions[index + 1] - positions[index - 1]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-15:
            raise NumericalStateError("String path tangent is degenerate.")
        return tangent / norm

    @staticmethod
    def _climbing_force(force: np.ndarray, tau0: np.ndarray) -> np.ndarray:
        """Return Eq. (22) force F - 2(F dot tau0)tau0 for frozen tau0."""
        physical = to_numpy_f64(force).reshape(-1)
        tangent = to_numpy_f64(tau0).reshape(-1)
        return physical - 2.0 * float(np.dot(physical, tangent)) * tangent

    @classmethod
    def _string_residuals(
        cls,
        images: List[Atoms],
        forces: List[np.ndarray],
        *,
        ci_index: Optional[int] = None,
    ) -> StringResiduals:
        regular = []
        for index in range(1, len(images) - 1):
            if index == ci_index:
                continue
            physical = to_numpy_f64(forces[index]).reshape(-1)
            tangent = cls._path_tangent(images, index)
            regular.append(physical - float(np.dot(physical, tangent)) * tangent)
        regular_flat = (
            np.concatenate(regular)
            if regular
            else np.zeros(0, dtype=np.float64)
        )
        regular_max = (
            float(np.max(np.abs(regular_flat))) if regular_flat.size else 0.0
        )
        regular_rms = (
            float(np.sqrt(np.mean(regular_flat**2))) if regular_flat.size else 0.0
        )
        if ci_index is None:
            return StringResiduals(regular_max, regular_rms)
        ci_force = to_numpy_f64(forces[ci_index]).reshape(-1)
        return StringResiduals(
            regular_max,
            regular_rms,
            float(np.max(np.abs(ci_force))),
            float(np.sqrt(np.mean(ci_force**2))),
        )

    def _capture_path_stage(
        self,
        *,
        status: RunStatus,
        images: List[Atoms],
        energies,
        forces,
        maximum_iterations: int,
        used_iterations: int,
        termination_reason: str,
        candidate: Optional[Atoms] = None,
    ) -> PathStageResult:
        return PathStageResult.capture(
            status=status,
            images=images,
            energies_hartree=energies,
            forces_hartree_per_angstrom=forces,
            budgets={
                "maximum_iterations": maximum_iterations,
                "used_iterations": used_iterations,
            },
            termination_reason=termination_reason,
            candidate=candidate,
        )

    def relax_string(self, source_images: List[Atoms]) -> PathStageResult:
        """Run the 2007 full-force evolution and equal-arc split iteration."""
        images = self._copy_images(list(source_images))
        drivers = [
            LBFGSDriver(
                m=self.params.lbfgs_memory,
                curvature=self.params.lbfgs_curvature,
                maxstep=self.params.lbfgs_max_step,
                step_limit_in_place=False,
            )
            for _ in images[1:-1]
        ]
        used = 0
        try:
            energies, forces = self._evaluate_images(images)
            residual = self._string_residuals(images, forces)
            converged = (
                residual.regular_max < self.params.string_f_max_th
                and residual.regular_rms < self.params.string_f_rms_th
            )
            while not converged and used < self.params.max_iter_relax:
                old_positions = self._positions_matrix(images)
                old_gradients = [
                    -to_numpy_f64(force).reshape(-1) for force in forces[1:-1]
                ]
                evolved = old_positions.copy()
                for offset, (driver, gradient) in enumerate(
                    zip(drivers, old_gradients), start=1
                ):
                    evolved[offset] += driver.step_limit(driver.two_loop(gradient))
                self._set_positions_matrix(images, evolved)
                self._equal_arc_reparameterize(images)
                energies, forces = self._evaluate_images(images)
                new_positions = self._positions_matrix(images)
                new_gradients = [
                    -to_numpy_f64(force).reshape(-1) for force in forces[1:-1]
                ]
                for offset, (driver, old_gradient, new_gradient) in enumerate(
                    zip(drivers, old_gradients, new_gradients), start=1
                ):
                    driver.update(
                        new_positions[offset] - old_positions[offset],
                        new_gradient - old_gradient,
                    )
                used += 1
                residual = self._string_residuals(images, forces)
                converged = (
                    residual.regular_max < self.params.string_f_max_th
                    and residual.regular_rms < self.params.string_f_rms_th
                )
        except (NumericalStateError, CurvatureValidationError) as exc:
            return self._capture_path_stage(
                status=RunStatus.REFUSED,
                images=images,
                energies=None,
                forces=None,
                maximum_iterations=self.params.max_iter_relax,
                used_iterations=used,
                termination_reason=str(exc),
            )
        return self._capture_path_stage(
            status=(RunStatus.CONVERGED if converged else RunStatus.BOUNDED_NONCONVERGED),
            images=images,
            energies=energies,
            forces=forces,
            maximum_iterations=self.params.max_iter_relax,
            used_iterations=used,
            termination_reason=(
                "string_converged" if converged else "string_iteration_limit"
            ),
        )

    def relax_cistring(
        self,
        source_images: List[Atoms],
        *,
        ci_index: int,
    ) -> PathStageResult:
        """Run Eq. (22) with one frozen tangent and split anchored reparameterization."""
        images = self._copy_images(list(source_images))
        if not 0 < ci_index < len(images) - 1:
            raise ValueError("CI index must identify an interior String image.")
        try:
            tau0 = self._path_tangent(images, ci_index).copy()
        except (NumericalStateError, CurvatureValidationError) as exc:
            return self._capture_path_stage(
                status=RunStatus.REFUSED,
                images=images,
                energies=None,
                forces=None,
                maximum_iterations=self.params.max_iter_cistring,
                used_iterations=0,
                termination_reason=str(exc),
                candidate=images[ci_index],
            )
        drivers = [
            LBFGSDriver(
                m=self.params.lbfgs_memory,
                curvature=self.params.lbfgs_curvature,
                maxstep=self.params.lbfgs_max_step,
                step_limit_in_place=False,
            )
            for _ in images[1:-1]
        ]
        used = 0
        try:
            energies, forces = self._evaluate_images(images)
            residual = self._string_residuals(images, forces, ci_index=ci_index)
            converged = (
                residual.regular_max < self.params.string_f_max_th
                and residual.regular_rms < self.params.string_f_rms_th
                and residual.ci_max < self.params.cistring_f_max_th
                and residual.ci_rms < self.params.cistring_f_rms_th
            )
            while not converged and used < self.params.max_iter_cistring:
                old_positions = self._positions_matrix(images)
                old_gradients = []
                for index in range(1, len(images) - 1):
                    physical = to_numpy_f64(forces[index]).reshape(-1)
                    effective = (
                        self._climbing_force(physical, tau0)
                        if index == ci_index
                        else physical
                    )
                    old_gradients.append(-effective)
                evolved = old_positions.copy()
                for offset, (driver, gradient) in enumerate(
                    zip(drivers, old_gradients), start=1
                ):
                    evolved[offset] += driver.step_limit(driver.two_loop(gradient))
                self._set_positions_matrix(images, evolved)
                self._split_reparameterize(images, ci_index)
                energies, forces = self._evaluate_images(images)
                new_positions = self._positions_matrix(images)
                new_gradients = []
                for index in range(1, len(images) - 1):
                    physical = to_numpy_f64(forces[index]).reshape(-1)
                    effective = (
                        self._climbing_force(physical, tau0)
                        if index == ci_index
                        else physical
                    )
                    new_gradients.append(-effective)
                for offset, (driver, old_gradient, new_gradient) in enumerate(
                    zip(drivers, old_gradients, new_gradients), start=1
                ):
                    driver.update(
                        new_positions[offset] - old_positions[offset],
                        new_gradient - old_gradient,
                    )
                used += 1
                residual = self._string_residuals(images, forces, ci_index=ci_index)
                converged = (
                    residual.regular_max < self.params.string_f_max_th
                    and residual.regular_rms < self.params.string_f_rms_th
                    and residual.ci_max < self.params.cistring_f_max_th
                    and residual.ci_rms < self.params.cistring_f_rms_th
                )
        except (NumericalStateError, CurvatureValidationError) as exc:
            return self._capture_path_stage(
                status=RunStatus.REFUSED,
                images=images,
                energies=None,
                forces=None,
                maximum_iterations=self.params.max_iter_cistring,
                used_iterations=used,
                termination_reason=str(exc),
                candidate=images[ci_index],
            )
        return self._capture_path_stage(
            status=(RunStatus.CONVERGED if converged else RunStatus.BOUNDED_NONCONVERGED),
            images=images,
            energies=energies,
            forces=forces,
            maximum_iterations=self.params.max_iter_cistring,
            used_iterations=used,
            termination_reason=(
                "cistring_converged" if converged else "cistring_iteration_limit"
            ),
            candidate=images[ci_index],
        )

    @staticmethod
    def _energy_weighted_tangent(images: List[Atoms], energies: List[float], i: int) -> np.ndarray:
        """Energy-weighted tangent for interior index i; returns flattened 3N unit vector."""
        Rm1 = to_numpy_f64(images[i - 1].get_positions())
        R   = to_numpy_f64(images[i].get_positions())
        Rp1 = to_numpy_f64(images[i + 1].get_positions())
        Em1 = energies[i - 1]; E = energies[i]; Ep1 = energies[i + 1]
        t = (Ep1 - E) * (Rp1 - R) + (E - Em1) * (R - Rm1)
        t = t.reshape(-1)
        n = np.linalg.norm(t)
        if n < 1e-14:
            t = (Rp1 - R + R - Rm1).reshape(-1)
            n = np.linalg.norm(t)
            if n < 1e-14:
                t = np.zeros_like(t); t[0] = 1.0; n = 1.0
        return t / n

    def _endpoint_tangent3(self, prev_img: Atoms, last_img: Atoms, fallback_dir3: np.ndarray) -> np.ndarray:
        """Estimate a full Cartesian 3N endpoint tangent."""
        R0 = to_numpy_f64(prev_img.get_positions())
        R1 = to_numpy_f64(last_img.get_positions())
        d = R1 - R0
        if np.linalg.norm(d) < 1e-12:
            return self._unit(fallback_dir3)
        return self._unit(d.reshape(-1))

    def _endpoint_direct_step(self, at: Atoms, dir3: np.ndarray, step: float):
        """Move an endpoint in full Cartesian configuration space."""
        pos = to_numpy_f64(at.get_positions())
        nrm = np.linalg.norm(dir3)
        if nrm < 1e-15:
            return
        disp = (dir3 / nrm).reshape(pos.shape) * float(step)
        at.set_positions(pos + disp)


    # ------------------- constrained mini-relax on a single image --------------

    def _mini_relax_perp_lbfgs(self, img: Atoms, tau3: np.ndarray, max_iter: int, gtol: float):
        """
        L-BFGS mini-optimization that only moves in the hyperplane orthogonal to tau3.
        Gradient is dE/dx (projected), i.e., g = -F_perp.
        """
        pos = to_numpy_f64(img.get_positions()).reshape(-1)
        masses = to_numpy_f64(img.get_masses())
        tau3N = self._unit(np.asarray(tau3, dtype=np.float64).reshape(-1))

        def proj_perp(vec):
            return vec - float(np.dot(vec, tau3N)) * tau3N

        def energy_and_grad(xvec):
            img.set_positions(xvec.reshape(-1,3))
            E, F = self._evaluate_image(img)
            F = F.reshape(-1,3)
            F = project_out_rigidbody_forces(F, xvec.reshape(-1,3), masses).reshape(-1)
            g = -F
            g = proj_perp(g)
            return E, g

        lb = LBFGSDriver(
            m=5,
            curvature=70.0,
            maxstep=0.1,
            step_limit_in_place=False,
        )
        x = pos.copy()
        _, g = energy_and_grad(x)
        k = 0
        while k < max_iter and not lb.should_stop(g, fmax_th=gtol, frms_th=gtol*0.5):
            p = lb.two_loop(g)            # descent direction in subspace
            p = proj_perp(p)              # safety: ensure perpendicular
            p = lb.step_limit(p)
            x_new = x + p
            _, g_new = energy_and_grad(x_new)
            s = x_new - x
            y = g_new - g
            lb.update(s, y)
            x, g = x_new, g_new
            k += 1
        img.set_positions(x.reshape(-1,3))

    # ---------------------------- adaptive predictor ---------------------------

    def _predict_and_relax_one(self, last_img: Atoms, dir3: np.ndarray,
                               E_last: float, label: str) -> Tuple[Atoms, float, float]:
        """
        Adaptive predictor (step backtracking on uphill / |F_perp|) + mini-relax (LBFGS).
        Returns: (new_img, step_used, max|F_perp| after relax)
        """
        p = self.params
        ds = p.grow_step
        dir3 = self._unit(dir3)
        masses_last = to_numpy_f64(last_img.get_masses())

        while ds >= p.grow_step_min:
            # predict
            trial = self._copy_with_inherit(last_img)
            pos_last = to_numpy_f64(last_img.get_positions())
            trial.set_positions(pos_last + ds * dir3.reshape(pos_last.shape))

            # quick check at predicted point
            pos  = to_numpy_f64(trial.get_positions())
            E_pred, Fraw = self._evaluate_image(trial)
            Fraw = Fraw.reshape(-1,3)
            Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
            tau3N = dir3
            F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
            maxFperp_pred = float(np.max(np.abs(F_perp)))

            uphill = (E_pred - E_last) > p.grow_energy_tol

            # backtrack on step if energy is too uphill or force is too large
            if (uphill or maxFperp_pred > 10.0 * p.grow_relax_tol) and ds > p.grow_step_min:
                ds *= p.grow_backtrack
                continue

            # orthogonal mini-relax by L-BFGS
            self._mini_relax_perp_lbfgs(trial, dir3, max_iter=p.grow_relax_iters, gtol=p.grow_relax_tol)

            # recompute max|F_perp| after relax
            pos  = to_numpy_f64(trial.get_positions())
            _, Fraw = self._evaluate_image(trial)
            Fraw = Fraw.reshape(-1,3)
            Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
            F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
            maxFperp_final = float(np.max(np.abs(F_perp)))
            return trial, ds, maxFperp_final

        # fallback minimal step
        trial = self._copy_with_inherit(last_img)
        pos_last = to_numpy_f64(last_img.get_positions())
        trial.set_positions(pos_last + p.grow_step_min * dir3.reshape(pos_last.shape))
        self._mini_relax_perp_lbfgs(trial, dir3, max_iter=p.grow_relax_iters, gtol=p.grow_relax_tol)
        pos  = to_numpy_f64(trial.get_positions())
        _, Fraw = self._evaluate_image(trial)
        Fraw = Fraw.reshape(-1,3)
        Frb  = project_out_rigidbody_forces(Fraw, pos, masses_last).reshape(-1)
        tau3N = dir3
        F_perp = Frb - float(np.dot(Frb, tau3N)) * tau3N
        return trial, p.grow_step_min, float(np.max(np.abs(F_perp)))

    # ---------------------------- restart / STRING-TS --------------------------

    def restart_run(
        self,
        ci_stage: PathStageResult,
        hei_idx: int,
        base_prefix: str,
    ) -> PRFOResult:
        """Run PRFO only from a converged CI-String snapshot."""
        if not isinstance(ci_stage, PathStageResult):
            raise TypeError("String-TS requires a PathStageResult from CI-String.")
        if ci_stage.status is not RunStatus.CONVERGED:
            raise ValueError("String-TS requires a converged CI-String stage.")
        if (
            ci_stage.energies_hartree is None
            or ci_stage.forces_hartree_per_angstrom is None
        ):
            raise ValueError("String-TS requires final CI-String energy and forces.")
        if not 0 < hei_idx < len(ci_stage.images) - 1:
            raise ValueError("String-TS candidate index must be an interior image.")

        from .PRFO import PRFO

        images = self._copy_images(list(ci_stage.images))
        ts_guess = self._copy_with_inherit(images[hei_idx])
        prfo_parameters = dict(self.params.prfo_params)
        prfo_parameters["project_rigid_modes"] = bool(self.params.project_rigid_modes)
        prfo = PRFO(
            output=self.output,
            atoms=ts_guess,
            paras={"prfo": prfo_parameters},
        )
        prfo_result = self._run_subjob(prfo)
        if not isinstance(prfo_result, PRFOResult):
            raise TypeError("PRFO must return a typed PRFOResult.")

        energies = np.asarray(ci_stage.energies_hartree, dtype=np.float64)
        forces = list(ci_stage.forces_hartree_per_angstrom)
        residuals = []
        for index in range(len(images)):
            physical = to_numpy_f64(forces[index]).reshape(-1)
            if index in {0, len(images) - 1}:
                projected = physical
            else:
                tangent = self._path_tangent(images, index)
                projected = physical - float(np.dot(physical, tangent)) * tangent
            residuals.append(
                (
                    float(np.max(np.abs(projected))),
                    float(np.sqrt(np.mean(projected**2))),
                )
            )

        candidate = prfo_result.atoms
        candidate_energy = prfo_result.energy_hartree
        candidate_forces = prfo_result.forces_hartree_per_angstrom
        if candidate_energy is None or candidate_forces is None:
            candidate_energy, candidate_forces = self._evaluate_image(candidate)
        candidate_force = to_numpy_f64(candidate_forces).reshape(-1)
        candidate_max = float(np.max(np.abs(candidate_force)))
        candidate_rms = float(np.sqrt(np.mean(candidate_force**2)))

        mep_path = base_prefix + "_stringts_mep.xyz"
        write_xyz(mep_path, images, energies=list(energies))
        if prfo_result.certified_ts:
            structure_path = base_prefix + "_stringts_ts.xyz"
            structure_label = "certified TS"
        else:
            structure_path = base_prefix + "_stringts_candidate.xyz"
            structure_label = "uncertified candidate"
        write_xyz(structure_path, [candidate], energies=[candidate_energy])

        kcal_per_Eh = 627.509
        log_info([
            "\n---------------------------------------------------------------\n",
            "                    PATH SUMMARY FOR String-TS             \n",
            "---------------------------------------------------------------\n",
            "All forces in Eh/Angstrom. Candidate/TS forces are physical forces.\n\n",
            "Image     E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n",
        ], self.output)
        for index, energy in enumerate(energies):
            marker = " <= CI" if index == hei_idx else ""
            maximum, rms = residuals[index]
            log_info([
                f"{index:4d} {energy:12.5f} {(energy - energies[0]) * kcal_per_Eh:11.2f} "
                f"{maximum:11.5f} {rms:10.5f}{marker}\n"
            ], self.output)
        log_info([
            f"  TS candidate {candidate_energy:12.5f} "
            f"{(candidate_energy - energies[0]) * kcal_per_Eh:11.2f} "
            f"{candidate_max:11.5f} {candidate_rms:10.5f} <= {structure_label}\n",
            f"\nWrote converged CI-String MEP to: {mep_path}\n",
            f"Wrote separate {structure_label} structure to: {structure_path}\n",
        ], self.output)
        return prfo_result

    # ------------------------------- main flow --------------------------------

    def run(self):
        """
        Main GSM workflow:
        1) Two-ended adaptive growth with stabilizers (connector floor, monotonic guard, forced pull).
        2) Merge L + reversed R and equal-arc reparameterization to fixed n_images.
        3) Return the path at refine=None, or refine the HEI only when requested.
        """
        def forces_info_from_array(forces):
            F = to_numpy_f64(forces)
            maxF = np.max(np.linalg.norm(F, axis=1))
            rmsF = np.sqrt(np.mean(np.linalg.norm(F, axis=1) ** 2))
            return maxF, rmsF

        # --- Report endpoints
        endpoint_energies, endpoint_forces = self._evaluate_images([self.atoms_R, self.atoms_P])
        E_R, E_P = endpoint_energies
        maxF_R, rmsF_R = forces_info_from_array(endpoint_forces[0])
        maxF_P, rmsF_P = forces_info_from_array(endpoint_forces[1])

        log_info([
            "\nProperties of fixed STRING end points:\n",
            "               Reactant:\n",
            f"                         E               ....   {E_R: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_R: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_R: .6f} Eh/Angstrom\n",
            "               Product:\n",
            f"                         E               ....   {E_P: .6f} Eh\n",
            f"                         RMS(F)          ....   {rmsF_P: .6f} Eh/Angstrom\n",
            f"                         MAX(|F|)        ....   {maxF_P: .6f} Eh/Angstrom\n",
        ], self.output)

        # --- Kabsch alignment of endpoints
        R = to_numpy_f64(self.atoms_R.get_positions())
        P = to_numpy_f64(self.atoms_P.get_positions())
        P_aligned, rmsd, _, _ = kabsch_align(R, P)
        self.atoms_P.set_positions(P_aligned)
        log_info([f"\nAlignment done. RMSD: {rmsd:.6f} (Angstrom)\n"], self.output)

        base, _ = os.path.splitext(self.output)

        # --- Initialize two growth fronts
        L: List[Atoms] = [self._copy_with_inherit(self.atoms_R)]
        R_: List[Atoms] = [self._copy_with_inherit(self.atoms_P)]
        E_L_last = self._evaluate_image(L[-1])[0]
        E_R_last = self._evaluate_image(R_[-1])[0]

        # Initial directions retain the complete non-rigid endpoint displacement.
        connector = (P_aligned - R).reshape(-1)
        if np.linalg.norm(connector) < 1e-14:
            images = [self._copy_with_inherit(self.atoms_R), self._copy_with_inherit(self.atoms_P)]
            energies, forces = self._evaluate_images(images)
            stage = self._capture_path_stage(
                status=RunStatus.BOUNDED_NONCONVERGED,
                images=images,
                energies=energies,
                forces=forces,
                maximum_iterations=self.params.max_iter_grow,
                used_iterations=0,
                termination_reason="degenerate_guide",
            )
            return {
                "status": "bounded_nonconverged", "converged": False,
                "termination_reason": "degenerate_guide", "growth_iterations": 0,
                "images": list(stage.images),
                "energies": list(stage.energies_hartree),
                "highest_energy_image": int(np.argmax(energies)),
                "prfo_result": None,
                "growth_stage": stage,
                "string_stage": None,
                "cistring_stage": None,
                "stage_statuses": {"growth": stage.status.value},
                "refinement_skipped_reason": "growth_not_merged",
            }
        dirL = self._unit(connector); dirR = -dirL
        if self.params.max_iter_grow == 0:
            images = []
            for fraction in np.linspace(0.0, 1.0, self.params.n_images):
                image = self._copy_with_inherit(self.atoms_R)
                image.set_positions((1.0 - fraction) * R + fraction * P_aligned)
                images.append(image)
            energies, forces = self._evaluate_images(images)
            stage = self._capture_path_stage(
                status=RunStatus.BOUNDED_NONCONVERGED,
                images=images,
                energies=energies,
                forces=forces,
                maximum_iterations=0,
                used_iterations=0,
                termination_reason="growth_limit",
                candidate=images[int(np.argmax(energies))],
            )
            return {
                "status": "bounded_nonconverged", "converged": False,
                "termination_reason": "growth_limit", "growth_iterations": 0,
                "images": list(stage.images),
                "energies": list(stage.energies_hartree),
                "highest_energy_image": int(np.argmax(energies)),
                "prfo_result": None,
                "growth_stage": stage,
                "string_stage": None,
                "cistring_stage": None,
                "stage_statuses": {"growth": stage.status.value},
                "refinement_skipped_reason": "growth_not_merged",
            }

        # The first bilateral growth consumes the first growth round.
        newL, dsL, fpL = self._predict_and_relax_one(L[-1], dirL, E_L_last, "L")
        L.append(newL); E_L_last = self._evaluate_image(newL)[0]
        newR, dsR, fpR = self._predict_and_relax_one(R_[-1], dirR, E_R_last, "R")
        R_.append(newR); E_R_last = self._evaluate_image(newR)[0]

        # Growth log header
        log_info([
            "\nStarting GSM growth (two-ended, adaptive):\n",
            "GrowIter   d_min(Ang)      thr   |L|  |R|   dsL    dsR   max|F_perp|_L  max|F_perp|_R\n"
        ], self.output)

        # --- Growth policy parameters
        dmin_tol        = 0.05                           # allow ≤ +5% worsening per accepted step
        near_cutoff     = 2.0                            # in Å, shrink step when close
        max_retry_side  = 3                              # per-side backtracking trials
        max_retry_iter  = 3                              # global shrunken retries if worsening

        # NEW stabilizers (configurable through GSMParams)
        conn_min        = getattr(self.params, "guide_conn_min", 0.30)
        d_switch        = getattr(self.params, "guide_d_switch", 4.0)
        inc_streak      = 0
        inc_streak_tr   = getattr(self.params, "guide_inc_trigger", 2)

        it = 1
        d_min = self._min_image_distance(L[-1], R_[-1])
        terminal_reason = "merged_path" if d_min <= self.params.merge_threshold else "growth_limit"
        while it < self.params.max_iter_grow and d_min > self.params.merge_threshold:
            it += 1
            d_min_old = self._min_image_distance(L[-1], R_[-1])

            # Imbalance gating: pause the longer side
            grow_left = True; grow_right = True
            if len(L) - len(R_) > self.params.pause_imbalance:
                grow_left = False
            if len(R_) - len(L) > self.params.pause_imbalance:
                grow_right = False

            # Periodic endpoint mini-relax in the plane normal to the local direction
            if self.params.refine_end_every > 0 and it > 0 and (it % self.params.refine_end_every == 0):
                if len(L) >= 2:
                    E_prev = self._evaluate_image(L[-2])[0]
                    E_last = self._evaluate_image(L[-1])[0]
                    v = to_numpy_f64(L[-1].get_positions()) - to_numpy_f64(L[-2].get_positions())
                    dir3 = self._unit((E_last - E_prev) * v.reshape(-1))
                    self._mini_relax_perp_lbfgs(L[-1], dir3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                    E_L_last = self._evaluate_image(L[-1])[0]
                if len(R_) >= 2:
                    E_prev = self._evaluate_image(R_[-2])[0]
                    E_last = self._evaluate_image(R_[-1])[0]
                    v = to_numpy_f64(R_[-1].get_positions()) - to_numpy_f64(R_[-2].get_positions())
                    dir3 = self._unit((E_last - E_prev) * v.reshape(-1))
                    self._mini_relax_perp_lbfgs(R_[-1], dir3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                    E_R_last = self._evaluate_image(R_[-1])[0]

            # --- Blended directions: energy-weighted endpoint tangent + connector (with floor)
            connL3 = self._unit((to_numpy_f64(R_[-1].get_positions()) - to_numpy_f64(L[-1].get_positions())).reshape(-1))
            connR3 = self._unit((to_numpy_f64(L[-1].get_positions()) - to_numpy_f64(R_[-1].get_positions())).reshape(-1))

            def ew_tangent_end(chain):
                """Energy-weighted endpoint tangent as a 3D guide vector."""
                if len(chain) < 2:
                    return None
                v = to_numpy_f64(chain[-1].get_positions()) - to_numpy_f64(chain[-2].get_positions())
                if np.linalg.norm(v) < 1e-14:
                    return None
                E_prev = self._evaluate_image(chain[-2])[0]
                E_last = self._evaluate_image(chain[-1])[0]
                return self._unit((E_last - E_prev) * v.reshape(-1))

            tauL3 = ew_tangent_end(L);  tauL3 = connL3 if tauL3 is None else tauL3
            tauR3 = ew_tangent_end(R_); tauR3 = connR3 if tauR3 is None else tauR3

            # connector weight never below conn_min; decays with distance d_min_old
            beta_conn = max(conn_min, max(0.0, 1.0 - d_min_old / d_switch))
            dirL3 = self._unit((1.0 - beta_conn) * tauL3 + beta_conn * connL3)
            dirR3 = self._unit((1.0 - beta_conn) * tauR3 + beta_conn * connR3)
            if np.linalg.norm(dirL3) < 1e-14 or np.linalg.norm(dirR3) < 1e-14:
                terminal_reason = "degenerate_guide"
                break

            def preset_step(dmin, base_step):
                """Shrink step when close to merge."""
                if dmin < near_cutoff:
                    return max(0.1 * dmin, self.params.grow_step_min)
                return base_step

            # --- Generate candidates (per-side backtracking; do NOT append yet)
            base_step_saved = self.params.grow_step

            candL, dsL, fpL = None, 0.0, 0.0
            if grow_left and len(L) >= 2:
                self.params.grow_step = preset_step(d_min_old, base_step_saved)
                retry = 0
                while True:
                    testL, ds_try, fp_try = self._predict_and_relax_one(L[-1], dirL3, self._evaluate_image(L[-1])[0], "L")
                    dL_only = self._min_image_distance(testL, R_[-1])
                    if (dL_only <= d_min_old * (1.0 + dmin_tol)) or (ds_try <= self.params.grow_step_min) or (retry >= max_retry_side):
                        candL, dsL, fpL = testL, ds_try, fp_try
                        break
                    self.params.grow_step = max(self.params.grow_step * self.params.grow_backtrack, self.params.grow_step_min)
                    retry += 1

            candR, dsR, fpR = None, 0.0, 0.0
            if grow_right and len(R_) >= 2:
                self.params.grow_step = preset_step(d_min_old, base_step_saved)
                retry = 0
                while True:
                    testR, ds_try, fp_try = self._predict_and_relax_one(R_[-1], dirR3, self._evaluate_image(R_[-1])[0], "R")
                    dR_only = self._min_image_distance(L[-1], testR)
                    if (dR_only <= d_min_old * (1.0 + dmin_tol)) or (ds_try <= self.params.grow_step_min) or (retry >= max_retry_side):
                        candR, dsR, fpR = testR, ds_try, fp_try
                        break
                    self.params.grow_step = max(self.params.grow_step * self.params.grow_backtrack, self.params.grow_step_min)
                    retry += 1

            self.params.grow_step = base_step_saved  # restore

            # --- Choose among {L-only, R-only, BOTH}
            options = []
            if candL is not None:
                options.append(("L",    self._min_image_distance(candL, R_[-1])))
            if candR is not None:
                options.append(("R",    self._min_image_distance(L[-1], candR)))
            if (candL is not None) and (candR is not None):
                options.append(("BOTH", self._min_image_distance(candL, candR)))

            if not options:
                log_info(["\nNo feasible growth candidates; stopping growth.\n"], self.output)
                terminal_reason = "no_feasible_growth"
                break

            options.sort(key=lambda x: x[1])  # prefer smaller d_min
            chosen = options[0]
            d_new  = chosen[1]

            # --- Monotonic guard: if still worse, shrink globally and retry a few times
            if d_new > d_min_old * (1.0 + dmin_tol):
                improved = False
                base_step_saved = self.params.grow_step
                for _ in range(max_retry_iter):
                    self.params.grow_step = max(self.params.grow_step * 0.5, self.params.grow_step_min)
                    # Rebuild candidates with smaller step
                    if grow_left and len(L) >= 2:
                        testL, ds_try, fp_try = self._predict_and_relax_one(L[-1], dirL3, self._evaluate_image(L[-1])[0], "L")
                        dL_only = self._min_image_distance(testL, R_[-1])
                        if dL_only <= d_min_old * (1.0 + dmin_tol):
                            candL, dsL, fpL = testL, ds_try, fp_try
                    if grow_right and len(R_) >= 2:
                        testR, ds_try, fp_try = self._predict_and_relax_one(R_[-1], dirR3, self._evaluate_image(R_[-1])[0], "R")
                        dR_only = self._min_image_distance(L[-1], testR)
                        if dR_only <= d_min_old * (1.0 + dmin_tol):
                            candR, dsR, fpR = testR, ds_try, fp_try
                    # Re-evaluate
                    options = []
                    if candL is not None: options.append(("L",    self._min_image_distance(candL, R_[-1])))
                    if candR is not None: options.append(("R",    self._min_image_distance(L[-1], candR)))
                    if (candL is not None) and (candR is not None):
                        options.append(("BOTH", self._min_image_distance(candL, candR)))
                    if options:
                        options.sort(key=lambda x: x[1])
                        if options[0][1] <= d_min_old * (1.0 + dmin_tol):
                            chosen = options[0]; d_new = chosen[1]; improved = True; break
                self.params.grow_step = base_step_saved

                # --- Forced connector pull if repeated worsening
                if not improved:
                    inc_streak += 1
                    if inc_streak >= inc_streak_tr:
                        inc_streak = 0
                        pull = max(self.params.grow_step_min, min(0.15 * d_min_old, 0.5))  # Å
                        self._endpoint_direct_step(L[-1], connL3, pull)
                        self._mini_relax_perp_lbfgs(L[-1], connL3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                        self._endpoint_direct_step(R_[-1], connR3, pull)
                        self._mini_relax_perp_lbfgs(R_[-1], connR3, max_iter=self.params.grow_relax_iters, gtol=self.params.grow_relax_tol)
                        # Local alignment for stability
                        if len(L) >= 2: align_path_inplace(L[-2:],  mode="chain")
                        if len(R_)>= 2: align_path_inplace(R_[-2:], mode="chain")
                        d_now = self._min_image_distance(L[-1], R_[-1])
                        d_min = d_now
                        if it % self.params.log_every == 0:
                            log_info([f"{it:>8d} {d_now:12.4f} {self.params.merge_threshold:7.3f} "
                                    f"{len(L):5d} {len(R_):5d} {0.0:6.3f} {0.0:6.3f} {0.0:13.5f} {0.0:13.5f}\n"], self.output)
                        if d_now <= self.params.merge_threshold:
                            terminal_reason = "merged_path"
                            break
                        continue
                else:
                    inc_streak = 0
            else:
                inc_streak = 0

            # --- Commit chosen candidates
            if chosen[0] == "L":
                L.append(candL); E_L_last = self._evaluate_image(candL)[0]
            elif chosen[0] == "R":
                R_.append(candR); E_R_last = self._evaluate_image(candR)[0]
            else:
                L.append(candL); E_L_last = self._evaluate_image(candL)[0]
                R_.append(candR); E_R_last = self._evaluate_image(candR)[0]

            # Local alignment for each side
            if len(L) >= 3: align_path_inplace(L[-3:], mode="chain")
            if len(R_) >= 3: align_path_inplace(R_[-3:], mode="chain")

            # Log one line
            d_min = self._min_image_distance(L[-1], R_[-1])
            if it % self.params.log_every == 0:
                log_info([f"{it:>8d} {d_min:12.4f} {self.params.merge_threshold:7.3f} "
                        f"{len(L):5d} {len(R_):5d} {dsL:6.3f} {dsR:6.3f} {fpL:13.5f} {fpR:13.5f}\n"], self.output)

            # Merge check
            if d_min <= self.params.merge_threshold:
                terminal_reason = "merged_path"
                break

        if terminal_reason == "growth_limit" and it >= self.params.max_iter_grow:
            log_info([
                "\nGSM growth reached its maximum iterations. The bounded candidate "
                "path will be reported without String refinement.\n"
            ], self.output)

        # --- Merge & equal-arc reparameterization to fixed n_images
        def concat_and_resample(Ls: List[Atoms], Rs: List[Atoms], n_images: int) -> List[Atoms]:
            """Concatenate L side + reversed R side, then resample to n_images by arclength."""
            path = [Ls[i] for i in range(len(Ls))] + [Rs[i] for i in range(len(Rs) - 2, -1, -1)]
            images = [Ls[0].copy()]
            inherit_attrs(Ls[0], images[0])
            for k in range(1, n_images - 1):
                images.append(Ls[0].copy()); inherit_attrs(Ls[0], images[k])
            images.append(Rs[0].copy()); inherit_attrs(Rs[0], images[-1])

            X = [to_numpy_f64(img.get_positions()) for img in path]
            s = [0.0]
            for i in range(1, len(X)):
                s.append(s[-1] + np.linalg.norm(X[i] - X[i-1]))
            Ltot = s[-1] if len(s) else 0.0
            if Ltot < 1e-15:
                for k in range(1, n_images - 1):
                    images[k].set_positions(X[0])
                return images

            s_targets = np.linspace(0.0, Ltot, n_images)
            for k in range(1, n_images - 1):
                sk = s_targets[k]
                j = np.searchsorted(s, sk) - 1
                j = max(0, min(j, len(X) - 2))
                lam = 0.0 if (s[j+1] - s[j] < 1e-16) else (sk - s[j]) / (s[j+1] - s[j])
                pos = (1.0 - lam) * X[j] + lam * X[j+1]
                images[k].set_positions(pos)
            return images

        images = concat_and_resample(L, R_, self.params.n_images)
        for img in images:
            if img.calc is None:
                img.calc = self.atoms_R.calc
        align_path_inplace(images, mode="chain")
        linear_reparam(images)

        # Dump growth-final equal-arc path & HEI
        grow_final = base + "_gsm_candidate_path.xyz"
        Es, Fs = self._evaluate_images(images)
        write_xyz(grow_final, images, energies=Es)
        hei = max(range(1, len(images) - 1), key=lambda i: Es[i]) if len(images) > 2 else 0
        candidate_path = base + "_gsm_equal_arc_candidate.xyz"
        hei_path = base + "_gsm_hei.xyz"
        write_xyz(candidate_path, images, energies=Es)
        write_xyz(hei_path, [images[hei]], energies=[Es[hei]])

        log_info([
            f"\nWrote GSM candidate path to: {grow_final}\n",
            f"Wrote equal-arc candidate path to: {candidate_path}\n",
            f"Wrote HEI to:                 {hei_path}\n"
        ], self.output)

        d_min = self._min_image_distance(L[-1], R_[-1])
        merged = d_min <= self.params.merge_threshold
        if merged:
            terminal_reason = "merged_path"
        growth_stage = self._capture_path_stage(
            status=(RunStatus.CONVERGED if merged else RunStatus.BOUNDED_NONCONVERGED),
            images=images,
            energies=Es,
            forces=Fs,
            maximum_iterations=self.params.max_iter_grow,
            used_iterations=it,
            termination_reason=terminal_reason,
            candidate=images[hei],
        )

        result = {
            "status": "path_generated" if merged else "bounded_nonconverged",
            "converged": False,
            "termination_reason": terminal_reason,
            "growth_iterations": it,
            "images": list(growth_stage.images),
            "energies": list(growth_stage.energies_hartree),
            "highest_energy_image": hei,
            "prfo_result": None,
            "growth_stage": growth_stage,
            "string_stage": None,
            "cistring_stage": None,
            "stage_statuses": {"growth": growth_stage.status.value},
            "refinement_skipped_reason": None,
        }

        if not merged:
            result["refinement_skipped_reason"] = "growth_not_merged"
            return result

        # refine=None retains the established growth-only public behavior.
        if self.params.refine is None:
            result["refinement_skipped_reason"] = "refine_none"
            return result

        ordinary_stage = self.relax_string(list(growth_stage.images))
        result["string_stage"] = ordinary_stage
        result["stage_statuses"]["ordinary"] = ordinary_stage.status.value
        result["images"] = list(ordinary_stage.images)
        result["energies"] = (
            None
            if ordinary_stage.energies_hartree is None
            else list(ordinary_stage.energies_hartree)
        )
        if ordinary_stage.status is not RunStatus.CONVERGED:
            result["status"] = ordinary_stage.status.value
            result["termination_reason"] = ordinary_stage.termination_reason
            result["refinement_skipped_reason"] = "ordinary_string_not_converged"
            return result

        hei = int(np.argmax(ordinary_stage.energies_hartree[1:-1])) + 1
        ci_stage = self.relax_cistring(list(ordinary_stage.images), ci_index=hei)
        result["cistring_stage"] = ci_stage
        result["stage_statuses"]["ci"] = ci_stage.status.value
        result["images"] = list(ci_stage.images)
        result["energies"] = (
            None
            if ci_stage.energies_hartree is None
            else list(ci_stage.energies_hartree)
        )
        result["highest_energy_image"] = hei
        result["status"] = ci_stage.status.value
        result["converged"] = ci_stage.status is RunStatus.CONVERGED
        result["termination_reason"] = ci_stage.termination_reason
        if ci_stage.status is not RunStatus.CONVERGED:
            result["refinement_skipped_reason"] = "cistring_not_converged"
            return result

        ci_mep = base + "_cistring_mep.xyz"
        ci_image = base + "_cistring_ci.xyz"
        write_xyz(ci_mep, list(ci_stage.images), energies=list(ci_stage.energies_hartree))
        write_xyz(ci_image, [ci_stage.images[hei]], energies=[ci_stage.energies_hartree[hei]])
        if self.params.refine == "cistring":
            return result

        prfo_result = self.restart_run(ci_stage, hei, base)
        result["prfo_result"] = prfo_result
        result["stage_statuses"]["prfo"] = prfo_result.status.value
        result["status"] = prfo_result.status.value
        result["converged"] = prfo_result.certified_ts
        result["termination_reason"] = prfo_result.termination_reason
        if not prfo_result.certified_ts:
            result["refinement_skipped_reason"] = "prfo_not_certified"
        return result

