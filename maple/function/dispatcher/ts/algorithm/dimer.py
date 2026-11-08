# -*- coding: utf-8 -*-
"""
Dimer implementation with:
- Minimum-mode following (rotation: minimize kappa = n^T H n)
- Translation with parallel flip once kappa < 0
- Optional HVP (autograd) callback to replace finite-difference (no Δ tuning)
- Trust-radius / max-step control
- Detailed human-readable logging and XYZ outputs per iteration
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC

from .dimer_translation import (
    TRANSLATION_POLICY,
    DimerTranslationAttempt,
    effective_translation_force,
    full_force_norm,
    propose_translation,
    validate_translation_controls,
)


def _stationary_success_reason(atoms) -> str:
    from ...pure_nonmd_status import is_pure_torch_v4

    return (
        "computed_index_one_stationary_point"
        if is_pure_torch_v4(atoms) else "resolved_index_one_saddle"
    )

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

def vec1d(x, n_expected=None) -> np.ndarray:
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v

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
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def write_all_images_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, iteration: int = 0):
    """
    Append current single structure to an xyz trajectory file (for Dimer debug).
    Each block corresponds to one iteration of Dimer optimization.
    """
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    with open(filename, "a") as f:
        f.write(f"{len(symbols)}\n")
        if energy is not None:
            f.write(f"Iter {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iter {iteration}\n")
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------ Dimer Params ---------------------------------
# =============================================================================

@dataclass
class DimerParams:
    # Rotation / curvature estimation
    use_hvp: bool = False               # if True, call hvp_fn(atoms, n_flat) for Hn
    delta: float = 0.005                 # Angstrom; used only if not use_hvp
    rot_max_iter: int = 5               # rotation inner iterations per outer step
    rot_alpha: float = 0.5              # rotation step factor on F_rot (unitless); small ~ (0.1~1)
    rot_f_max_th: float = 1.0e-3        # convergence threshold on max|F_rot| (Eh/Ang)
    rot_f_rms_th: float = 5.0e-4        # convergence threshold on RMS(F_rot)

    # Translation / trust region
    step0: float = 0.2                  # initial step scaling on search direction
    step_max: float = 0.15              # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15          # same role as max_step; kept both for clarity

    # Convergence (translation forces)
    f_max_th: float = 5.0e-3            # max(|F_trans|) Eh/Ang
    f_rms_th: float = 1.0e-3            # RMS(F_trans) Eh/Ang
    kappa_to_flip: float = 0.0          # if kappa < this value, flip parallel component

    # Iterations
    max_iter: int = 200

    # Metric / projections
    use_mass_weight: bool = False       # if True, use M-metric for dot/proj (mass weighted)
    remove_rigid: bool = True           # remove global translation/rotation from n (molecular)

    # Initialization of n
    n_init: str = "random"              # "random" | "force" | "given"
    n_given: Optional[np.ndarray] = None
    random_seed: Optional[int] = None

    # Outputs
    save_traj: bool = True
    save_metrics: bool = True


@dataclass(frozen=True, slots=True)
class DimerRotationResult:
    direction: np.ndarray
    h_direction: np.ndarray
    curvature: float
    max_residual: float
    rms_residual: float
    converged: bool
    hvp_count: int
    subspace_dimension: int
    rigid_rank: int
    projected_antisymmetry: float
    force: np.ndarray
    energy: float
    positions_angstrom: np.ndarray
    termination_reason: str

    def __post_init__(self):
        for name in (
            "direction", "h_direction", "force", "positions_angstrom"
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        for name in (
            "curvature", "max_residual", "rms_residual",
            "projected_antisymmetry", "energy",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"Dimer rotation {name} must be finite.")
        if not self.termination_reason:
            raise ValueError("Dimer rotation termination reason is required.")


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Dimer rotation values must be finite.")
    result.setflags(write=False)
    return result


def _euclidean_rigid_basis(atoms: Atoms) -> np.ndarray:
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    centered = positions - positions.mean(axis=0)
    vectors = []
    for axis in np.eye(3):
        vectors.append(np.tile(axis, (len(atoms), 1)).reshape(-1))
    for axis in np.eye(3):
        vectors.append(np.cross(centered, axis).reshape(-1))
    matrix = np.column_stack(vectors)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    rank = int(np.count_nonzero(singular_values > tolerance))
    return _readonly(u[:, :rank])


def _project(vector: np.ndarray, rigid_basis: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    return value - rigid_basis @ (rigid_basis.T @ value)


def _two_pass_orthogonalize(vector, basis):
    result = np.array(vector, dtype=np.float64, copy=True)
    norm_before = float(np.linalg.norm(result))
    for _ in range(2):
        for column in basis:
            result -= float(np.dot(column, result)) * column
    return result, norm_before, float(np.linalg.norm(result))

# =============================================================================
# ------------------------------ Metric helpers -------------------------------
# =============================================================================

def _get_metric(atoms: Atoms, use_mass_weight: bool):
    if not use_mass_weight:
        return None
    m = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)  # (N,1)
    M = np.repeat(m, 3, axis=1).reshape(-1)              # (3N,)
    return M                                             # diagonal metric entries

def _dot(v, w, M=None):
    """Metric dot: if M is None -> Euclidean; else mass-weighted sum(M * v * w)."""
    if M is None:
        return float(np.dot(v, w))
    return float(np.dot(M * v, w))

def _norm(v, M=None):
    val = _dot(v, v, M=M)
    return math.sqrt(max(val, 0.0))

def _proj_parallel(F, n, M=None):
    """Return parallel component of F along n (unit under metric)."""
    # assume n is normalized in metric (n^T M n = 1) if M given
    c = _dot(F, n, M=M)
    return c * n

def _proj_perp(F, n, M=None):
    return F - _proj_parallel(F, n, M=M)

def _normalize(n, M=None, eps=1e-20):
    dn = _norm(n, M=M)
    if dn < eps:
        raise ValueError("Zero-length direction encountered during normalization.")
    return n / dn

def _remove_rigid_body_components(n, atoms: Atoms, M=None):
    """
    Remove global translation & rotation components from direction n.
    Only meaningful for molecules (non-periodic). For simplicity:
    - Remove translation: subtract mean per axis (mass-weighted average if M given)
    - Remove rotation: project out 3 rotational modes around COM using cross r x axis
      (simple approximate projector; robust enough for search direction).
    """
    # translation
    N = len(atoms)
    X = to_numpy_f64(atoms.get_positions()).reshape(-1, 3)
    v = n.reshape(-1, 3).copy()
    if M is None:
        t = v.mean(axis=0, keepdims=True)
    else:
        mw = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)
        t = (mw * v).sum(axis=0, keepdims=True) / (mw.sum() + 1e-20)
    v -= t

    # rotation (approx): project out components proportional to r x omega, omega = basis unit vectors
    # center positions
    rc = X.mean(axis=0)
    r = X - rc
    # three axes basis
    axes = np.eye(3)
    for k in range(3):
        rot_mode = np.cross(r, axes[k])  # (N,3)
        # project each component of v onto rot_mode
        a = (v * rot_mode).sum() / ((rot_mode * rot_mode).sum() + 1e-20)
        v -= a * rot_mode
    return v.reshape(-1)

# =============================================================================
# ------------------------------ Dimer class ----------------------------------
# =============================================================================

class Dimer(JobABC):
    """
    Dimer saddle search with optional HVP (autograd) backend.

    hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]]
        If provided and DimerParams.use_hvp=True, returns H @ n (shape (3N,))
        Else: we use finite-difference via two force calls at R ± Δ n.
    """
    def __init__(self,
                 output: str,
                 atoms_init: Atoms,
                 paras: Optional[dict] = None,
                 hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]] = None):
        super().__init__(output)

        self.atoms = atoms_init
        if self.atoms.calc is None:
            raise ValueError("atoms_init must have a working calculator set (atoms.calc).")
        from ...legacy_units import require_legacy_hartree_workflow_view
        require_legacy_hartree_workflow_view(self.atoms)

        # Initialize params from paras dict
        self.params = self._init_params(DimerParams, paras, ("dimer", "DIMER", "ts"))

        from ...pure_nonmd_status import is_pure_torch_v4
        if is_pure_torch_v4(self.atoms):
            if hvp_fn is not None:
                raise ValueError(
                    "Pure Torch V4 Dimer uses only the canonical calculator HVP tuple."
                )
            self.params.use_hvp = True
            if self.params.use_mass_weight:
                raise ValueError(
                    "Pure Torch V4 Dimer mass weighting is not defined for the "
                    "current translation metric."
                )
            if type(self.params.rot_max_iter) is not int or self.params.rot_max_iter < 1:
                raise ValueError("rot_max_iter must be a positive integer.")
            validate_translation_controls(
                beta=self.params.step0,
                step_max=self.params.step_max,
                trust_radius=self.params.trust_radius,
                max_iter=self.params.max_iter,
            )

        self.hvp_fn = hvp_fn if self.params.use_hvp else None

        # metric
        self.M = _get_metric(self.atoms, self.params.use_mass_weight)

        # init direction n
        self.n = self._init_direction()

        # trust region bookkeeping
        self.alpha = float(self.params.step0)

    def _v4_rotation(self, seed: np.ndarray) -> DimerRotationResult:
        """Bounded residual-expanded Rayleigh-Ritz rotation for one geometry."""

        rigid = (
            _euclidean_rigid_basis(self.atoms)
            if self.params.remove_rigid
            else np.zeros((3 * len(self.atoms), 0), dtype=np.float64)
        )
        projected = _project(seed, rigid)
        norm = float(np.linalg.norm(projected))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise ValueError("Dimer seed has no non-rigid component.")
        basis: list[np.ndarray] = [projected / norm]
        h_columns: list[np.ndarray] = []
        force = energy = None
        last = None
        calculator = self.atoms.calc
        assert calculator is not None
        for _ in range(self.params.rot_max_iter):
            vector = basis[len(h_columns)]
            h_direction, force_value, energy_value = calculator.get_hvp(
                self.atoms, vector
            )
            h_columns.append(_project(vec1d(h_direction, seed.size), rigid))
            force = vec1d(force_value, seed.size)
            energy = float(to_numpy_f64(energy_value))
            V = np.column_stack(basis)
            W = np.column_stack(h_columns)
            raw = V.T @ W
            if not np.isfinite(raw).all():
                raise RuntimeError("Dimer projected HVP operator is nonfinite.")
            antisymmetry = float(np.max(np.abs(raw - raw.T), initial=0.0))
            # A Cartesian componentwise skew budget does not transport to this
            # basis. Convergence uses the unsymmetrized full-space action below.
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (raw + raw.T))
            coefficients = eigenvectors[:, 0]
            raw_direction = V @ coefficients
            direction_norm = float(np.linalg.norm(raw_direction))
            if direction_norm <= np.finfo(np.float64).tiny:
                raise RuntimeError("Dimer Ritz vector has zero norm.")
            direction = raw_direction / direction_norm
            h_ritz = (W @ coefficients) / direction_norm
            curvature = float(eigenvalues[0])
            residual = _project(h_ritz - curvature * direction, rigid)
            max_residual = float(np.max(np.abs(residual)))
            rms_residual = float(np.sqrt(np.mean(np.square(residual))))
            converged = bool(
                max_residual <= self.params.rot_f_max_th
                and rms_residual <= self.params.rot_f_rms_th
            )
            reason = (
                "rotation_residual_converged" if converged
                else "rotation_iteration_budget_exhausted"
            )
            last = DimerRotationResult(
                direction=_readonly(direction),
                h_direction=_readonly(h_ritz),
                curvature=curvature,
                max_residual=max_residual,
                rms_residual=rms_residual,
                converged=converged,
                hvp_count=len(h_columns),
                subspace_dimension=len(basis),
                rigid_rank=rigid.shape[1],
                projected_antisymmetry=antisymmetry,
                force=_readonly(force),
                energy=energy,
                positions_angstrom=_readonly(self.atoms.get_positions()),
                termination_reason=reason,
            )
            if converged:
                return last
            expanded, before, after = _two_pass_orthogonalize(residual, basis)
            breakdown = (
                after
                <= np.finfo(np.float64).eps
                * max(3 * len(self.atoms), len(basis))
                * before
            )
            if breakdown:
                return DimerRotationResult(
                    **{
                        field: getattr(last, field)
                        for field in last.__dataclass_fields__
                        if field != "termination_reason"
                    },
                    termination_reason="rotation_subspace_breakdown",
                )
            basis.append(expanded / after)
        assert last is not None
        return last

    @staticmethod
    def _rotation_record(rotation: DimerRotationResult) -> dict:
        if not (
            np.isfinite(rotation.energy)
            and np.isfinite(rotation.curvature)
            and np.isfinite(rotation.force).all()
        ):
            raise ValueError("Dimer rotation receipt observables must be finite.")
        return {
            "direction": rotation.direction.tolist(),
            "energy_hartree": rotation.energy,
            "forces_Ha_per_A": rotation.force.reshape(-1, 3).tolist(),
            "curvature_Ha_per_A2": rotation.curvature,
            "max_residual_Ha_per_A2": rotation.max_residual,
            "rms_residual_Ha_per_A2": rotation.rms_residual,
            "converged": rotation.converged,
            "termination_reason": rotation.termination_reason,
            "hvp_count": rotation.hvp_count,
            "subspace_dimension": rotation.subspace_dimension,
            "rigid_rank": rotation.rigid_rank,
            "projected_antisymmetry_Ha_per_A2": (
                rotation.projected_antisymmetry
            ),
            "projected_antisymmetry_role": "diagnostic_only_no_transported_bound",
            "positions_angstrom": rotation.positions_angstrom.tolist(),
        }

    # ------------------------------ init n -----------------------------------

    def _init_direction(self) -> np.ndarray:
        N = len(self.atoms)
        D = 3 * N
        p = self.params

        if p.n_init.lower() == "given" and (p.n_given is not None):
            n = vec1d(p.n_given, D)
        elif p.n_init.lower() == "force":
            F = -vec1d(self.atoms.get_forces(), D)  # gradient = -F; here use force itself
            n = F
        else:  # random
            seed = p.random_seed
            if seed is None:
                seed = os.environ.get("MAPLE_NONMD_SEED")
            rng = np.random.default_rng(None if seed is None else int(seed))
            n = rng.normal(size=D)

        if p.remove_rigid:
            n = _remove_rigid_body_components(n, self.atoms, M=self.M)
        n = _normalize(n, M=self.M)
        return n

    # ----------------------- curvature & rotation helpers ---------------------

    def _finite_diff_Hn_and_F(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (Hn, F_avg, F_diff) with central difference using forces at R ± Δ n.
        """
        p = self.params
        Δ = float(p.delta)

        # save & perturb positions
        X = x_flat.reshape(-1, 3)
        # +Δ
        self.atoms.set_positions((X + Δ * n.reshape(-1, 3)))
        F_plus = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # -Δ
        self.atoms.set_positions((X - Δ * n.reshape(-1, 3)))
        F_minus = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # restore
        self.atoms.set_positions(X)

        F_avg = 0.5 * (F_plus + F_minus)
        F_diff = 0.5 * (F_plus - F_minus)
        Hn = -(F_plus - F_minus) / (2.0 * Δ)  # central difference for Hn
        return Hn, F_avg, F_diff

    def _hvp_Hn_and_F(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (Hn, F) using autograd HVP for Hn, and single force at R for F.
        """
        # force at current R
        F = to_numpy_f64(self.atoms.get_forces()).reshape(-1)
        # Hn via callback
        if self.hvp_fn is None:
            raise RuntimeError("use_hvp=True but hvp_fn is not provided.")
        Hn = vec1d(self.hvp_fn(self.atoms, n), n.size)
        return Hn, F

    def _rotate_minimize_kappa(self, x_flat: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Do up to rot_max_iter steps of rotation to minimize kappa = n^T H n.
        Returns (n_new, max|F_rot|, rms(F_rot)).
        """
        from ...pure_nonmd_status import is_pure_torch_v4

        if is_pure_torch_v4(self.atoms):
            result = self._v4_rotation(n)
            return result.direction, result.max_residual, result.rms_residual
        p = self.params
        n_cur = n.copy()

        for _ in range(p.rot_max_iter):
            if p.use_hvp:
                if self.hvp_fn is not None:
                    Hn, F = self._hvp_Hn_and_F(x_flat, n_cur)
                else:
                    Hn, F, _ = self.atoms.calc.get_hvp(self.atoms, n_cur)
                    Hn = vec1d(Hn, n_cur.size)
                    F = vec1d(F, n_cur.size)
            else:
                Hn, F_avg, _ = self._finite_diff_Hn_and_F(x_flat, n_cur)
                F = F_avg  # keep F_avg for later translation when using FD

            # rotation force: (I - nn^T) Hn
            F_par = _proj_parallel(Hn, n_cur, M=self.M)
            F_rot = Hn - F_par

            max_frot = float(np.max(np.abs(F_rot)))
            rms_frot = float(math.sqrt(np.mean(F_rot * F_rot)))

            # convergence of rotation
            if (max_frot < p.rot_f_max_th) and (rms_frot < p.rot_f_rms_th):
                return n_cur, max_frot, rms_frot

            # gradient descent on kappa: n <- n - α * F_rot (and renormalize)
            n_next = n_cur - p.rot_alpha * F_rot
            if self.params.remove_rigid:
                n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
            n_next = _normalize(n_next, M=self.M)
            n_cur = n_next

        # after max rot steps return last
        return n_cur, max_frot, rms_frot

    # ------------------------------- main flow --------------------------------

    def atoms_to_xyz_block(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for idx, (s, (x, y, z)) in enumerate(zip(syms, pos)):
            lines.append(f"{idx:<4d}{s:>2s}{x:18.4f}{y:18.4f}{z:18.4f}")
        return "\n".join(lines) + "\n"

    def _run_v4(self):
        """Run the V4 Dimer with authoritative post-translation evaluations."""

        p = self.params
        base, _ = os.path.splitext(self.output)
        trajectory = base + "_dimer_traj.xyz"
        candidate = base + "_dimer_candidate.xyz"
        for name, default in (
            ("f_max_th", p.f_max_th), ("f_rms_th", p.f_rms_th),
            ("dp_max_th", 1.8e-3), ("dp_rms_th", 1.2e-3),
        ):
            if not hasattr(self.atoms, name):
                setattr(self.atoms, name, default)

        beta = float(p.step0)
        cap = min(float(p.trust_radius), float(p.step_max))
        completed_translations = 0
        attempted_translations = 0
        accepted_translations = 0
        final_attempt: DimerTranslationAttempt | None = None
        initial_positions = np.array(self.atoms.get_positions(), copy=True)
        initial_direction = np.array(self.n, copy=True)
        try:
            rotation = self._v4_rotation(initial_direction)
        except Exception as exc:
            self.atoms.set_positions(initial_positions)
            self.n = initial_direction
            from ...pure_nonmd_status import NonMDWorkflowFailure, make_status

            failure_metrics: dict[str, object] = {
                "translation_policy": TRANSLATION_POLICY,
                "attempted_translations": attempted_translations,
                "accepted_translations": accepted_translations,
                "completed_translations": completed_translations,
                "final_attempt": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_message": str(exc),
            }
            raise NonMDWorkflowFailure(
                make_status(
                    workflow="ts",
                    method="dimer",
                    converged=False,
                    termination_reason="initial_rotation_evaluation_failed",
                    termination_class="execution_failure",
                    iterations=completed_translations,
                    final_metrics=failure_metrics,
                    atoms=self.atoms,
                )
            ) from exc
        self.n = np.array(rotation.direction, copy=True)
        log_info(
            [
                "\nStarting Pure Torch V4 Dimer numerical search.\n",
                "Rotation uses bounded residual-expanded Rayleigh-Ritz HVPs.\n",
            ],
            self.output,
        )
        displacement = np.zeros_like(rotation.force)
        converged = False
        terminal_reason: str | None = None

        def metrics(current, applied):
            forces = current.force.reshape(-1, 3)
            steps = applied.reshape(-1, 3)
            return {
                "max_f": float(np.max(np.linalg.norm(forces, axis=1))),
                "rms_f": float(np.sqrt(np.mean(np.sum(np.square(forces), axis=1)))),
                "max_dp": float(np.max(np.linalg.norm(steps, axis=1))),
                "rms_dp": float(np.sqrt(np.mean(np.sum(np.square(steps), axis=1)))),
            }

        def gates(current, values):
            return bool(
                values["max_f"] <= getattr(self.atoms, "f_max_th")
                and values["rms_f"] <= getattr(self.atoms, "f_rms_th")
                and values["max_dp"] <= getattr(self.atoms, "dp_max_th")
                and values["rms_dp"] <= getattr(self.atoms, "dp_rms_th")
                and current.converged
            )

        values = metrics(rotation, displacement)
        converged = gates(rotation, values)
        while not converged and accepted_translations < p.max_iter:
            direction = rotation.direction
            attempted_translations += 1
            ordinal = attempted_translations
            current_rotation = rotation
            previous = np.array(rotation.positions_angstrom, copy=True)
            current_force_norm = full_force_norm(rotation.force)
            translation_force = effective_translation_force(
                rotation.force,
                direction,
                curvature=rotation.curvature,
                kappa_to_flip=p.kappa_to_flip,
            )
            try:
                proposal = propose_translation(
                    previous,
                    translation_force,
                    beta_A2_per_Ha=beta,
                    cap_angstrom=cap,
                )
            except FloatingPointError as exc:
                final_attempt = DimerTranslationAttempt(
                    ordinal=ordinal,
                    stage="nonfinite_proposal",
                    current_rotation=self._rotation_record(current_rotation),
                    current_positions_angstrom=previous,
                    proposed_positions_angstrom=None,
                    translation_force_Ha_per_A=translation_force,
                    beta_A2_per_Ha=beta,
                    raw_step_angstrom=None,
                    applied_step_angstrom=None,
                    evaluated_displacement_angstrom=None,
                    cap_angstrom=cap,
                    clipped=None,
                    current_force_norm_Ha_per_A=current_force_norm,
                    trial_force_norm_Ha_per_A=None,
                    trial_rotation=None,
                    accepted=False,
                    reason="nonfinite_translation_proposal",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                from ...pure_nonmd_status import NonMDWorkflowFailure, make_status

                failure_metrics: dict[str, object] = dict(values)
                failure_metrics.update(
                    rotation=self._rotation_record(rotation),
                    translation_policy=TRANSLATION_POLICY,
                    attempted_translations=attempted_translations,
                    accepted_translations=accepted_translations,
                    completed_translations=completed_translations,
                    final_attempt=final_attempt.to_record(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise NonMDWorkflowFailure(
                    make_status(
                        workflow="ts", method="dimer", converged=False,
                        termination_reason="nonfinite_translation_proposal",
                        termination_class="execution_failure",
                        iterations=completed_translations,
                        final_metrics=failure_metrics,
                        atoms=self.atoms,
                    )
                ) from exc
            if not np.any(proposal.applied_step_angstrom):
                final_attempt = DimerTranslationAttempt(
                    ordinal=ordinal,
                    stage="zero_direction",
                    current_rotation=self._rotation_record(current_rotation),
                    current_positions_angstrom=previous,
                    proposed_positions_angstrom=proposal.proposed_positions_angstrom,
                    translation_force_Ha_per_A=translation_force,
                    beta_A2_per_Ha=beta,
                    raw_step_angstrom=proposal.raw_step_angstrom,
                    applied_step_angstrom=proposal.applied_step_angstrom,
                    evaluated_displacement_angstrom=None,
                    cap_angstrom=cap,
                    clipped=proposal.clipped,
                    current_force_norm_Ha_per_A=current_force_norm,
                    trial_force_norm_Ha_per_A=None,
                    trial_rotation=None,
                    accepted=False,
                    reason="zero_translation_direction",
                )
                terminal_reason = "zero_translation_direction"
                break
            self.atoms.set_positions(
                proposal.proposed_positions_angstrom.reshape(-1, 3)
            )
            try:
                trial_rotation = self._v4_rotation(direction)
            except Exception as exc:
                self.atoms.set_positions(previous)
                self.n = np.array(current_rotation.direction, copy=True)
                from ...pure_nonmd_status import NonMDWorkflowFailure, make_status

                final_attempt = DimerTranslationAttempt(
                    ordinal=ordinal,
                    stage="evaluation_failed",
                    current_rotation=self._rotation_record(current_rotation),
                    current_positions_angstrom=previous,
                    proposed_positions_angstrom=proposal.proposed_positions_angstrom,
                    translation_force_Ha_per_A=translation_force,
                    beta_A2_per_Ha=beta,
                    raw_step_angstrom=proposal.raw_step_angstrom,
                    applied_step_angstrom=proposal.applied_step_angstrom,
                    evaluated_displacement_angstrom=None,
                    cap_angstrom=cap,
                    clipped=proposal.clipped,
                    current_force_norm_Ha_per_A=current_force_norm,
                    trial_force_norm_Ha_per_A=None,
                    trial_rotation=None,
                    accepted=False,
                    reason="rotation_evaluation_failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                failure_metrics: dict[str, object] = dict(values)
                failure_metrics.update(
                    rotation=self._rotation_record(rotation),
                    translation_policy=TRANSLATION_POLICY,
                    attempted_translations=attempted_translations,
                    accepted_translations=accepted_translations,
                    completed_translations=completed_translations,
                    final_attempt=final_attempt.to_record(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise NonMDWorkflowFailure(
                    make_status(
                        workflow="ts",
                        method="dimer",
                        converged=False,
                        termination_reason="rotation_evaluation_failed",
                        termination_class="execution_failure",
                        iterations=completed_translations,
                        final_metrics=failure_metrics,
                        atoms=self.atoms,
                    )
                ) from exc
            evaluated_displacement = (
                trial_rotation.positions_angstrom.reshape(-1) - previous.reshape(-1)
            )
            trial_values = metrics(trial_rotation, evaluated_displacement)
            trial_force_norm = full_force_norm(trial_rotation.force)
            trial_gates_pass = gates(trial_rotation, trial_values)
            accepted = bool(
                trial_gates_pass or trial_force_norm < current_force_norm
            )
            final_attempt = DimerTranslationAttempt(
                ordinal=ordinal,
                stage="evaluated",
                current_rotation=self._rotation_record(current_rotation),
                current_positions_angstrom=previous,
                proposed_positions_angstrom=proposal.proposed_positions_angstrom,
                translation_force_Ha_per_A=translation_force,
                beta_A2_per_Ha=beta,
                raw_step_angstrom=proposal.raw_step_angstrom,
                applied_step_angstrom=proposal.applied_step_angstrom,
                evaluated_displacement_angstrom=evaluated_displacement,
                cap_angstrom=cap,
                clipped=proposal.clipped,
                current_force_norm_Ha_per_A=current_force_norm,
                trial_force_norm_Ha_per_A=trial_force_norm,
                trial_rotation=self._rotation_record(trial_rotation),
                accepted=accepted,
                reason=(
                    "local_convergence_gates_satisfied"
                    if trial_gates_pass
                    else (
                        "full_force_norm_strictly_decreased"
                        if accepted else "translation_force_merit_not_decreased"
                    )
                ),
            )
            if not accepted:
                self.atoms.set_positions(previous)
                self.n = np.array(current_rotation.direction, copy=True)
                terminal_reason = "translation_force_merit_not_decreased"
                break
            rotation = trial_rotation
            displacement = evaluated_displacement
            self.atoms.set_positions(rotation.positions_angstrom)
            self.n = np.array(rotation.direction, copy=True)
            values = trial_values
            accepted_translations += 1
            completed_translations = accepted_translations
            converged = trial_gates_pass
            log_info(
                [
                    f"Dimer translation {completed_translations}: "
                    f"maxF={values['max_f']:.6e} rmsF={values['rms_f']:.6e} "
                    f"maxR={rotation.max_residual:.6e} "
                    f"rmsR={rotation.rms_residual:.6e}\n"
                ],
                self.output,
            )
            if p.save_traj:
                write_all_images_xyz(
                    trajectory, self.atoms, energy=rotation.energy,
                    iteration=completed_translations,
                )

        self.n = np.array(rotation.direction, copy=True)
        for name, value in values.items():
            setattr(self.atoms, name, value)
        write_xyz(candidate, [self.atoms], energies=[rotation.energy])
        log_info(
            [
                "\n---------------------------------------------------------------\n",
                "                 INFORMATION ABOUT SADDLE CANDIDATE\n",
                "---------------------------------------------------------------\n",
                "Scientific stationary-point certification is unavailable; "
                "this is a numerical candidate pending the computed Hessian "
                "postcondition.\n",
                f"Wrote candidate to: {candidate}\n",
            ],
            self.output,
        )
        from ...pure_nonmd_status import (
            add_v4_outcome_metrics,
            make_status,
        )

        final_metrics: dict[str, object] = dict(values)
        final_metrics["rotation"] = self._rotation_record(rotation)
        final_metrics["translation_policy"] = TRANSLATION_POLICY
        final_metrics["attempted_translations"] = attempted_translations
        final_metrics["accepted_translations"] = accepted_translations
        final_metrics["completed_translations"] = completed_translations
        final_metrics["final_attempt"] = (
            None if final_attempt is None else final_attempt.to_record()
        )
        add_v4_outcome_metrics(
            self.atoms, final_metrics, optimizer_converged=converged
        )
        if not converged:
            return make_status(
                workflow="ts", method="dimer", converged=False,
                termination_reason=(
                    terminal_reason or "maximum_iterations_reached"
                ),
                termination_class="bounded_nonconvergence",
                iterations=completed_translations,
                final_metrics=final_metrics,
                atoms=self.atoms,
            )
        try:
            from .PRFO import _pure_frozen_ts_postcondition

            diagnostic, _ = _pure_frozen_ts_postcondition(self.atoms)
            final_metrics.update(diagnostic)
        except Exception as exc:
            final_metrics["postcondition_error"] = str(exc)
            return make_status(
                workflow="ts", method="dimer", converged=False,
                termination_reason="index_one_validation_failed",
                termination_class="validation_failure",
                iterations=completed_translations,
                final_metrics=final_metrics,
                atoms=self.atoms,
            )
        success = bool(diagnostic["workflow_success"])
        return make_status(
            workflow="ts", method="dimer", converged=success,
            termination_reason=(
                _stationary_success_reason(self.atoms) if success
                else "index_one_validation_failed"
            ),
            termination_class="converged" if success else "validation_failure",
            iterations=completed_translations,
            final_metrics=final_metrics,
            atoms=self.atoms,
        )

    def run(self):
        """
        Main Dimer optimization loop with autograd-based Hessian-vector product (Hn).
        - Rotation uses atoms.calc.get_hvp(atoms, n) to compute Hn (no finite difference)
        - Translation also uses get_hvp to obtain (Hn, forces, energy) in one pass
        - No duplicate energy/force evaluations
        - Logs curvature, rotational force, mode flip, 4 PRFO-style criteria, and trust region
        """
        from ...pure_nonmd_status import is_pure_torch_v4
        if is_pure_torch_v4(self.atoms):
            return self._run_v4()

        import torch

        p = self.params
        base, _ = os.path.splitext(self.output)
        from ...pure_nonmd_status import is_pure_torch_v4
        v4_candidate = is_pure_torch_v4(self.atoms)
        traj_file = base + "_dimer_traj.xyz"
        ts_file = base + (
            "_dimer_candidate.xyz" if v4_candidate else "_dimer_ts.xyz"
        )
        kcal_per_Eh = 627.509

        # ------------------ init direction & step size ------------------
        n = self.n.copy()        # initial dimer orientation (assumed normalized & rigid-body removed if requested)
        alpha = float(self.alpha)

        def evaluate_hn_force_energy(n_vec: np.ndarray):
            if p.use_hvp:
                if self.hvp_fn is not None:
                    hn = self.hvp_fn(self.atoms, n_vec)
                    force = self.atoms.get_forces().reshape(-1)
                    energy = self.atoms.get_potential_energy(force_consistent=True)
                    return (torch.as_tensor(hn), torch.as_tensor(force),
                            torch.as_tensor(energy))
                return self.atoms.calc.get_hvp(self.atoms, n_vec)
            x_flat = self.atoms.get_positions().reshape(-1)
            hn, force, _ = self._finite_diff_Hn_and_F(x_flat, n_vec)
            energy = self.atoms.get_potential_energy(force_consistent=True)
            return (torch.as_tensor(hn), torch.as_tensor(force),
                    torch.as_tensor(energy))

        # ------------------ initial eval via autograd HVP ------------------
        # get forces & energy once (Hn unused for initial report)
        _, forces_t, energy_t = evaluate_hn_force_energy(n)
        forces_np = forces_t.detach().cpu().numpy()
        E0 = float(energy_t.detach().cpu().item())
        maxF0 = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
        rmsF0  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

        log_info([
            "\n---------------------------------------------------------------\n",
            "                       DIMER INITIAL STATE\n",
            "---------------------------------------------------------------\n",
            f"Energy (initial)                        ....  {E0: .8f} Eh\n",
            f"RMS(|F|)                                ....  {rmsF0: .6f} Eh/Angstrom\n",
            f"MAX(|F|)                                ....  {maxF0: .6f} Eh/Angstrom\n",
            "\nINITIAL COORDINATES (ANGSTROEM):\n",
            self.atoms_to_xyz_block(self.atoms),
        ], self.output)

        # ------------------ ensure PRFO-style thresholds exist ------------------
        if not hasattr(self.atoms, "f_max_th"):  self.atoms.f_max_th  = p.f_max_th
        if not hasattr(self.atoms, "f_rms_th"):  self.atoms.f_rms_th  = p.f_rms_th
        if not hasattr(self.atoms, "dp_max_th"): self.atoms.dp_max_th = 1.8e-3
        if not hasattr(self.atoms, "dp_rms_th"): self.atoms.dp_rms_th = 1.2e-3

        log_info([
            "\n----------------------------------------------------------------------\n",
            "                           Dimer Iterations                           \n",
            "----------------------------------------------------------------------\n"
        ], self.output)

        # ------------------ rotation helper (HVP-based) ------------------
        def rotate_minimize_kappa(n_vec: np.ndarray):
            """
            Up to rot_max_iter inner steps to reduce kappa, using autograd Hn.
            Returns: n_new (np.ndarray), max|F_rot| (float), RMS(F_rot) (float)
            """
            n_curr = n_vec.copy()
            max_frot = np.inf
            rms_frot = np.inf

            for _ in range(p.rot_max_iter):
                # Hn from autograd; discard forces/energy here
                Hn_t, _, _ = evaluate_hn_force_energy(n_curr)
                dev, dty = Hn_t.device, Hn_t.dtype
                n_th = torch.tensor(n_curr, device=dev, dtype=dty)

                # rotational force: (I - n n^T) Hn
                Hn_par = torch.dot(Hn_t, n_th) * n_th
                F_rot_t = Hn_t - Hn_par

                # metrics
                max_frot = float(torch.max(torch.abs(F_rot_t)).detach().cpu().item())
                rms_frot = float(torch.sqrt(torch.mean(F_rot_t * F_rot_t)).detach().cpu().item())

                # convergence of rotation
                if (max_frot <= p.rot_f_max_th) and (rms_frot <= p.rot_f_rms_th):
                    break

                # gradient descent on kappa in orientation space, then re-normalize (and rigid-body remove if requested)
                n_next = (n_th - p.rot_alpha * F_rot_t).detach().cpu().numpy()
                if p.remove_rigid:
                    n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
                n_next = _normalize(n_next, M=self.M)
                n_curr = n_next

            return n_curr, max_frot, rms_frot

        # ======================= main iteration loop =======================
        converged = False
        for it in range(1, p.max_iter + 1):

            # (1) rotation step using autograd HVP
            n, max_frot, rms_frot = rotate_minimize_kappa(n)

            # (2) translation-side evaluation in one pass
            Hn_t, forces_t, energy_t = evaluate_hn_force_energy(n)
            forces_np = forces_t.detach().cpu().numpy()
            E = float(energy_t.detach().cpu().item())

            # curvature kappa = n^T H n
            dev, dty = Hn_t.device, Hn_t.dtype
            n_th = torch.tensor(n, device=dev, dtype=dty)
            kappa = float(torch.dot(n_th, Hn_t).detach().cpu().item())

            # project forces parallel / perpendicular to n (torch tensors)
            Fpar_t  = torch.dot(forces_t, n_th) * n_th
            Fperp_t = forces_t - Fpar_t
            mode_flip = (kappa < p.kappa_to_flip)
            Ftrans_t = (Fperp_t - Fpar_t) if mode_flip else Fperp_t

            # (3) trust-region step (convert only the step to numpy)
            step_vec = (alpha * Ftrans_t).detach().cpu().numpy()
            step_norm_inf = float(np.max(np.abs(step_vec)))
            on_boundary = False
            max_allow = min(p.trust_radius, p.step_max)
            if step_norm_inf > max_allow:
                step_vec *= (max_allow / (step_norm_inf + 1e-20))
                on_boundary = True

            new_positions = self.atoms.get_positions() + step_vec.reshape(-1, 3)
            self.atoms.set_positions(new_positions)
            alpha = (max(0.5 * alpha, 0.1 * p.step0) if on_boundary
                    else min(1.2 * alpha, p.step_max))

            # (4) PRFO-style metrics
            max_dp = float(np.max(np.linalg.norm(step_vec.reshape(-1, 3), axis=1)))
            rms_dp = float(np.sqrt(np.mean(np.linalg.norm(step_vec.reshape(-1, 3), axis=1) ** 2)))
            max_f  = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
            rms_f  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

            self.atoms.max_f  = max_f
            self.atoms.rms_f  = rms_f
            self.atoms.max_dp = max_dp
            self.atoms.rms_dp = rms_dp

            # (5) report one iteration block
            coords_block = self.atoms_to_xyz_block(self.atoms)
            info = [
                "\n----------------------------------------------------------------------\n",
                f"                             Iteration: {it:<3d}                              \n\n",
                "                             Coordinates                               \n",
                "----------------------------------------------------------------------\n",
                coords_block,
                "\n",
                f"Energy:                  {E: .6f} Convergence criteria  Is converged \n",
                f"Maximum Force:         {max_f:>12.6f} {self.atoms.f_max_th:>12.6f}                {'Yes' if max_f <= self.atoms.f_max_th else 'No'}\n",
                f"RMS Force:             {rms_f:>12.6f} {self.atoms.f_rms_th:>12.6f}                {'Yes' if rms_f <= self.atoms.f_rms_th else 'No'}\n",
                f"Maximum Displacement:  {max_dp:>12.6f} {self.atoms.dp_max_th:>12.6f}                {'Yes' if max_dp <= self.atoms.dp_max_th else 'No'}\n",
                f"RMS Displacement:      {rms_dp:>12.6f} {self.atoms.dp_rms_th:>12.6f}                {'Yes' if rms_dp <= self.atoms.dp_rms_th else 'No'}\n",
                f"\nTrust radius (MW): {max_allow: .6f}  Step norm (MW): {step_norm_inf: .6f}  On boundary: {on_boundary}\n"
                f"Curvature (kappa):      {kappa:>12.6f} Eh/Å²\n",
                f"Max Rotational Force:   {max_frot:>12.6f} Eh/Angstrom\n",
                f"RMS Rotational Force:   {rms_frot:>12.6f} Eh/Angstrom\n",
                f"Mode Flip:              {'Yes' if mode_flip else 'No'}\n\n",
            ]
            log_info(info, self.output)

            if p.save_traj:
                # Your local helper supports single Atoms; keeping your current call signature
                write_all_images_xyz(traj_file, self.atoms, energy=E, iteration=it)

            # (6) convergence: all 4 PRFO-style criteria
            if (max_f <= self.atoms.f_max_th and
                rms_f <= self.atoms.f_rms_th and
                max_dp <= self.atoms.dp_max_th and
                rms_dp <= self.atoms.dp_rms_th):
                log_info([f"\nDimer optimization converged at iteration {it}.\n"], self.output)
                converged = True
                break

        # ------------------ final write & brief summary ------------------
        _, _, E_final_t = evaluate_hn_force_energy(n)
        E_final = float(E_final_t.detach().cpu().item())
        write_xyz(ts_file, [self.atoms], energies=[E_final])

        log_info([
            "\n---------------------------------------------------------------\n",
            (
                "                 INFORMATION ABOUT SADDLE CANDIDATE            \n"
                if v4_candidate
                else "                 INFORMATION ABOUT SADDLE POINT                \n"
            ),
            "---------------------------------------------------------------\n",
            f"Energy (final)                           ....  {E_final: .8f} Eh\n",
            "\n-----------------------------------------\n",
            "  SADDLE GUESS (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz_block(self.atoms),
            f"\nWrote Dimer trajectory to: {traj_file}\n",
            (
                "Scientific stationary-point certification is unavailable; "
                "this structure remains an uncertified candidate; certification "
                "requires a separate postcondition.\n"
                if v4_candidate else ""
            ),
            f"Wrote {'candidate' if v4_candidate else 'TS guess'} to: {ts_file}\n"
        ], self.output)

        from ...pure_nonmd_status import (
            add_v4_outcome_metrics, is_route2_workflow_status_enabled,
            make_status, optimization_metrics,
        )
        if not is_route2_workflow_status_enabled(self.atoms):
            return self.atoms
        final_metrics = optimization_metrics(self.atoms)
        add_v4_outcome_metrics(
            self.atoms, final_metrics, optimizer_converged=bool(converged)
        )
        if not converged:
            return make_status(
                workflow="ts", method="dimer", converged=False,
                termination_reason="maximum_iterations_reached",
                iterations=p.max_iter, final_metrics=final_metrics,
                atoms=self.atoms,
            )
        try:
            from .PRFO import _pure_frozen_ts_postcondition
            diagnostic, _ = _pure_frozen_ts_postcondition(self.atoms)
            final_metrics.update(diagnostic)
        except Exception as exc:
            final_metrics["postcondition_error"] = str(exc)
            return make_status(
                workflow="ts", method="dimer", converged=False,
                termination_reason="index_one_validation_failed",
                termination_class="validation_failure",
                iterations=it, final_metrics=final_metrics, atoms=self.atoms,
            )
        success = bool(diagnostic["workflow_success"])
        if diagnostic.get("numerical_stationary_point_validated") is not None:
            log_info(
                [
                    "Stored-operator numerical index-one assessment: "
                    f"{str(success).lower()}. Independent Hessian uncertainty "
                    "and scientific certification remain unavailable.\n"
                ],
                self.output,
            )
        return make_status(
            workflow="ts", method="dimer", converged=success,
            termination_reason=(
                _stationary_success_reason(self.atoms) if success
                else "index_one_validation_failed"
            ),
            termination_class="converged" if success else "validation_failure",
            iterations=it, final_metrics=final_metrics, atoms=self.atoms,
        )
