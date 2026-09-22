"""
Dimer implementation with:
- Minimum-mode following (rotation: minimize kappa = n^T H n)
- Translation with parallel flip once kappa < 0
- Optional HVP (autograd) callback to replace finite-difference (no Δ tuning)
- Trust-radius / max-step control
- Detailed human-readable logging and XYZ outputs per iteration
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from ase import Atoms

from ...jobABC import JobABC
from .logger import log_info

FloatArray = npt.NDArray[np.float64]

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x: Any) -> FloatArray:
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
        return arr.astype(np.float64, copy=False)
    if np.isscalar(x):
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)

def vec1d(x: Any, n_expected: int | None = None) -> FloatArray:
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v

def write_xyz(
    filename: str,
    images: Sequence[Atoms],
    energies: Sequence[float] | None = None,
):
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
            f.writelines(
                f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n"
                for s, (x, y, z) in zip(symbols, pos)
            )

def write_all_images_xyz(
    filename: str,
    atoms: Atoms,
    energy: float | None = None,
    iteration: int = 0,
):
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
        f.writelines(
            f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n"
            for s, (x, y, z) in zip(symbols, pos)
        )

# =============================================================================
# ------------------------------ Dimer Params ---------------------------------
# =============================================================================

@dataclass
class DimerParams:
    # Rotation / curvature estimation
    # None preserves the conservative default (gas HVP, implicit FD).  True
    # requires a separately admitted composed gas+solvent HVP provider.
    use_hvp: bool | None = None
    delta: float = 0.005                 # Angstrom; used only if not use_hvp
    rot_max_iter: int = 5               # rotation inner iterations per outer step
    rot_alpha: float = 0.5              # rotation step factor on F_rot (unitless); small ~ (0.1~1)
    rot_f_max_th: float = 1.0e-3        # convergence threshold on max|F_rot| (Eh/Ang^2)
    rot_f_rms_th: float = 5.0e-4        # convergence threshold on RMS(F_rot)

    # Translation / trust region
    step0: float = 0.2                  # initial step scaling on search direction
    step_max: float = 0.15              # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15          # same role as max_step; kept both for clarity

    # Convergence (translation forces)
    f_max_th: float = 5.0e-3            # max(|F_trans|) Eh/Ang
    f_rms_th: float = 1.0e-3            # RMS(F_trans) Eh/Ang
    dp_max_th: float = 1.8e-3           # max Cartesian displacement (Ang)
    dp_rms_th: float = 1.2e-3           # RMS Cartesian displacement (Ang)
    kappa_to_flip: float = 0.0          # if kappa < this value, flip parallel component

    # Iterations
    max_iter: int = 200

    # Metric / projections
    use_mass_weight: bool = False       # if True, use M-metric for dot/proj (mass weighted)
    remove_rigid: bool = True           # remove global translation/rotation from n (molecular)

    # Initialization of n
    n_init: str = "random"              # "random" | "force" | "given"
    n_given: FloatArray | None = None

    # Outputs
    save_traj: bool = True
    save_metrics: bool = True

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

    Derivative values follow MAPLE's CalcABC convention: energy in Hartree,
    force in Hartree/Angstrom, and Hn in Hartree/Angstrom**2. This optimizer
    does not reinterpret the native eV units of arbitrary ASE calculators.

    hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]]
        In HVP mode, returns H @ n in Hartree/Angstrom**2 (shape (3N,)).
        With no callback, HVP mode uses ``atoms.calc.get_hvp``. Finite-
        difference mode uses complete MAPLE-reported forces at R +/- delta*n.
    """
    def __init__(self,
                 output: str,
                 atoms_init: Atoms,
                 paras: dict[str, Any] | None = None,
                 hvp_fn: Callable[[Atoms, FloatArray], object] | None = None):
        super().__init__(output)

        self.atoms = atoms_init
        if self.atoms.calc is None:
            raise ValueError("atoms_init must have a working calculator set (atoms.calc).")
        # MAPLE calculators extend ASE's calculator protocol with get_hvp.
        self.calculator: Any = self.atoms.calc

        # Initialize params from paras dict
        self.params = self._init_params(
            DimerParams, paras or {}, ("dimer", "DIMER", "ts")
        )
        self._validate_params()

        implicit = getattr(self.calculator, "solvent_correction", None) is not None
        if self.params.use_hvp is True and implicit:
            correction = self.calculator.solvent_correction
            if not (
                getattr(
                    self.calculator,
                    "analytic_implicit_derivatives_admitted",
                    False,
                )
                is True
                and getattr(
                    correction, "analytic_task_derivatives_admitted", False
                )
                is True
                and callable(
                    getattr(correction, "get_directional_derivatives", None)
                )
            ):
                raise ValueError(
                    "Dimer implicit-solvent HVP requires an admitted solvent "
                    "directional-derivative backend; use use_hvp=false for the "
                    "complete-force finite-difference fallback."
                )
        self.derivative_mode = (
            "finite_difference"
            if self.params.use_hvp is False or (self.params.use_hvp is None and implicit)
            else "hvp"
        )
        if self.derivative_mode == "finite_difference" and self.atoms.constraints:
            raise NotImplementedError(
                "Dimer finite-difference curvature does not yet support ASE constraints; "
                "unconstrained side geometries are required for correct derivatives."
            )
        if self.derivative_mode == "finite_difference":
            supplied = self._lower_keys(
                self._select_subdict(paras or {}, ("dimer", "DIMER", "ts"))
            )
            prepare = getattr(self.calculator, "prepare_numerical_derivatives", None)
            recommended_step = None
            if callable(prepare):
                recommended_step = prepare()
            correction = getattr(self.calculator, "solvent_correction", None)
            resolve_outer_step = getattr(
                correction, "resolve_outer_curvature_step", None
            )
            if callable(resolve_outer_step):
                self.params.delta = resolve_outer_step(
                    task_delta=(self.params.delta if "delta" in supplied else None),
                    backend_hint=recommended_step,
                    task="dimer",
                )
                self._validate_params()
            elif recommended_step is not None and "delta" not in supplied:
                if isinstance(recommended_step, (bool, np.bool_)) or not isinstance(
                    recommended_step, (int, float, np.integer, np.floating)
                ):
                    raise ValueError("Backend numerical derivative step must be numeric.")
                self.params.delta = float(recommended_step)
                self._validate_params()
        self.hvp_fn = hvp_fn if self.derivative_mode == "hvp" else None

        # metric
        self.M = _get_metric(self.atoms, self.params.use_mass_weight)

        # init direction n
        self.n = self._init_direction()

        # trust region bookkeeping
        self.alpha = float(self.params.step0)

    def _validate_params(self) -> None:
        p = self.params
        if p.use_hvp is not None and not isinstance(p.use_hvp, bool):
            raise ValueError("use_hvp must be None, True, or False.")
        if (
            isinstance(p.delta, (bool, np.bool_))
            or not isinstance(p.delta, (int, float, np.integer, np.floating))
            or not np.isfinite(p.delta)
            or p.delta <= 0.0
        ):
            raise ValueError("Dimer delta must be finite and positive.")
        for name in ("rot_max_iter", "max_iter"):
            value = getattr(p, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"Dimer {name} must be a positive integer.")
        for name in (
            "rot_alpha",
            "rot_f_max_th",
            "rot_f_rms_th",
            "step0",
            "step_max",
            "trust_radius",
            "f_max_th",
            "f_rms_th",
            "dp_max_th",
            "dp_rms_th",
        ):
            value = getattr(p, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"Dimer {name} must be finite and positive.")
        if (
            isinstance(p.kappa_to_flip, (bool, np.bool_))
            or not isinstance(
                p.kappa_to_flip, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(p.kappa_to_flip)
        ):
            raise ValueError("Dimer kappa_to_flip must be finite.")
        if not isinstance(p.n_init, str) or p.n_init.lower() not in {
            "random",
            "force",
            "given",
        }:
            raise ValueError("Dimer n_init must be 'random', 'force', or 'given'.")
        if str(p.n_init).lower() == "given" and p.n_given is None:
            raise ValueError("Dimer n_given is required when n_init='given'.")

    # ------------------------------ init n -----------------------------------

    def _init_direction(self) -> FloatArray:
        N = len(self.atoms)
        D = 3 * N
        p = self.params

        if p.n_init.lower() == "given" and (p.n_given is not None):
            n = vec1d(p.n_given, D)
        elif p.n_init.lower() == "force":
            F = -vec1d(self.atoms.get_forces(), D)  # gradient = -F; here use force itself
            n = F
        else:  # random
            rng = np.random.default_rng()
            n = rng.normal(size=D)

        if p.remove_rigid:
            n = _remove_rigid_body_components(n, self.atoms, M=self.M)
        n = _normalize(n, M=self.M)
        return n

    # ----------------------- curvature & rotation helpers ---------------------

    @staticmethod
    def _require_finite(
        value: Any, *, label: str, size: int | None = None
    ) -> FloatArray:
        array = vec1d(value, size) if size is not None else to_numpy_f64(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Dimer {label} contains non-finite values.")
        return array

    def _center_energy_and_forces(
        self, x_flat: FloatArray
    ) -> tuple[FloatArray, float]:
        """Evaluate MAPLE force (Eh/A) and energy (Eh) at one center geometry."""
        center = vec1d(x_flat, 3 * len(self.atoms)).reshape(-1, 3)
        self.atoms.set_positions(center)
        forces = self._require_finite(
            self.atoms.get_forces(), label="center force", size=center.size
        )
        energy = float(to_numpy_f64(self.atoms.get_potential_energy()).item())
        if not np.isfinite(energy):
            raise ValueError("Dimer center energy contains non-finite values.")
        return forces, energy

    def _finite_difference_hn(
        self, x_flat: FloatArray, n: FloatArray
    ) -> FloatArray:
        """Return complete-force central-difference Hn in Eh/A^2.

        The calculator is evaluated at ``R +/- delta*n`` and is restored to the
        exact center geometry even when a side evaluation fails.
        """
        center = vec1d(x_flat, n.size).reshape(-1, 3).copy()
        direction = vec1d(n, center.size).reshape(-1, 3)
        delta = float(self.params.delta)
        try:
            self.atoms.set_positions(center + delta * direction)
            force_plus = self._require_finite(
                self.atoms.get_forces(),
                label="finite-difference force (+delta)",
                size=center.size,
            )
            self.atoms.set_positions(center - delta * direction)
            force_minus = self._require_finite(
                self.atoms.get_forces(),
                label="finite-difference force (-delta)",
                size=center.size,
            )
        finally:
            self.atoms.set_positions(center)
        return -(force_plus - force_minus) / (2.0 * delta)

    def _evaluate_derivatives(
        self, x_flat: FloatArray, n: FloatArray
    ) -> tuple[FloatArray, FloatArray, float]:
        """Return ``(Hn, force, energy)`` at one center geometry.

        Units are Eh/A^2, Eh/A, and Eh. Gas-phase auto mode preserves the
        calculator's legacy ``get_hvp`` path. Implicit-solvent auto mode uses
        central differences of the complete reported force, so the solvent
        curvature is included rather than silently substituted with gas HVP.
        """
        center = vec1d(x_flat, n.size).copy()
        direction = vec1d(n, center.size)
        self.atoms.set_positions(center.reshape(-1, 3))

        if self.derivative_mode == "finite_difference":
            correction = getattr(self.calculator, "solvent_correction", None)
            preflight = getattr(
                correction, "preflight_directional_outer_displacements", None
            )
            if callable(preflight):
                preflight(self.atoms, direction.reshape(-1, 3), self.params.delta)
            reserve_operation = getattr(
                getattr(correction, "provider", None),
                "reserve_force_operation",
                None,
            )
            operation_context = (
                reserve_operation(
                    atom_count=len(self.atoms),
                    force_call_count=3,
                    operation="dimer-derivative-set",
                )
                if callable(reserve_operation)
                else nullcontext()
            )
            with operation_context:
                hn = self._finite_difference_hn(center, direction)
                forces, energy = self._center_energy_and_forces(center)
            return (
                self._require_finite(hn, label="curvature", size=center.size),
                forces,
                energy,
            )

        if self.hvp_fn is not None:
            forces, energy = self._center_energy_and_forces(center)
            try:
                hn_raw = self.hvp_fn(self.atoms, direction)
            finally:
                self.atoms.set_positions(center.reshape(-1, 3))
            hn = self._require_finite(hn_raw, label="callback HVP", size=center.size)
            return hn, forces, energy

        # Do not catch backend HVP errors: a gas HVP failure must remain visible,
        # never trigger a hidden finite-difference fallback.
        get_hvp: Any = getattr(self.calculator, "get_hvp", None)
        if not callable(get_hvp):
            raise NotImplementedError(
                f"{type(self.calculator).__name__} does not implement get_hvp."
            )
        hvp_result: Any = get_hvp(self.atoms, direction)
        hn_raw, forces_raw, energy_raw = hvp_result
        hn = self._require_finite(hn_raw, label="HVP", size=center.size)
        forces = self._require_finite(forces_raw, label="HVP force", size=center.size)
        energy = float(to_numpy_f64(energy_raw).item())
        if not np.isfinite(energy):
            raise ValueError("Dimer HVP energy contains non-finite values.")
        return hn, forces, energy

    def _rotate_minimize_kappa(
        self, x_flat: FloatArray, n: FloatArray
    ) -> tuple[FloatArray, float, float]:
        """
        Do up to rot_max_iter steps of rotation to minimize kappa = n^T H n.
        Returns (n_new, max|F_rot|, rms(F_rot)).
        """
        p = self.params
        n_cur = n.copy()
        max_frot = math.inf
        rms_frot = math.inf

        for _ in range(p.rot_max_iter):
            Hn, _, _ = self._evaluate_derivatives(x_flat, n_cur)

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

    def run(self):
        """Run one minimum-mode-following loop with the selected derivative mode."""
        p = self.params
        base, _ = os.path.splitext(self.output)
        traj_file = base + "_dimer_traj.xyz"
        ts_file = base + "_dimer_ts.xyz"
        if p.save_traj and os.path.exists(traj_file):
            os.remove(traj_file)

        n = self.n.copy()
        alpha = float(self.alpha)
        x0 = to_numpy_f64(self.atoms.get_positions()).reshape(-1)
        _, forces_np, E0 = self._evaluate_derivatives(x0, n)
        maxF0 = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
        rmsF0 = float(
            np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2))
        )

        log_info(
            [
                "\n---------------------------------------------------------------\n",
                "                       DIMER INITIAL STATE\n",
                "---------------------------------------------------------------\n",
                f"Energy (initial)                        ....  {E0: .8f} Eh\n",
                f"RMS(|F|)                                ....  {rmsF0: .6f} Eh/Angstrom\n",
                f"MAX(|F|)                                ....  {maxF0: .6f} Eh/Angstrom\n",
                f"Derivative mode                         ....  {self.derivative_mode}\n",
                (
                    f"Finite-difference step                   ....  {p.delta:g} Angstrom\n"
                    if self.derivative_mode == "finite_difference" else ""
                ),
                "\nINITIAL COORDINATES (ANGSTROEM):\n",
                self.atoms_to_xyz_block(self.atoms),
            ],
            self.output,
        )

        f_max_th = float(getattr(self.atoms, "f_max_th", p.f_max_th))
        self.log_inference_precision(self.calculator)
        f_rms_th = float(getattr(self.atoms, "f_rms_th", p.f_rms_th))
        dp_max_th = float(getattr(self.atoms, "dp_max_th", p.dp_max_th))
        dp_rms_th = float(getattr(self.atoms, "dp_rms_th", p.dp_rms_th))

        log_info([
            "\n----------------------------------------------------------------------\n",
            "                           Dimer Iterations                           \n",
            "----------------------------------------------------------------------\n"
        ], self.output)

        converged = False
        for it in range(1, p.max_iter + 1):
            center = to_numpy_f64(self.atoms.get_positions()).reshape(-1)
            n, _, _ = self._rotate_minimize_kappa(center, n)
            Hn, forces_np, E = self._evaluate_derivatives(center, n)
            kappa = _dot(n, Hn, M=self.M)
            rotation_residual = _proj_perp(Hn, n, M=self.M)
            max_frot = float(np.max(np.abs(rotation_residual)))
            rms_frot = float(
                math.sqrt(np.mean(rotation_residual * rotation_residual))
            )

            Fpar = _proj_parallel(forces_np, n, M=self.M)
            Fperp = forces_np - Fpar
            mode_flip = kappa < p.kappa_to_flip
            Ftrans = (Fperp - Fpar) if mode_flip else Fperp

            step_vec = alpha * Ftrans
            step_norm_inf = float(np.max(np.abs(step_vec)))
            on_boundary = False
            max_allow = min(p.trust_radius, p.step_max)
            if step_norm_inf > max_allow:
                step_vec *= (max_allow / (step_norm_inf + 1e-20))
                on_boundary = True

            max_dp = float(np.max(np.linalg.norm(step_vec.reshape(-1, 3), axis=1)))
            rms_dp = float(
                np.sqrt(
                    np.mean(np.linalg.norm(step_vec.reshape(-1, 3), axis=1) ** 2)
                )
            )
            max_f = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
            rms_f = float(
                np.sqrt(
                    np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)
                )
            )

            for metric_name, metric_value in (
                ("max_f", max_f),
                ("rms_f", rms_f),
                ("max_dp", max_dp),
                ("rms_dp", rms_dp),
            ):
                setattr(self.atoms, metric_name, metric_value)

            negative_mode = kappa < 0.0
            rotation_converged = (
                max_frot <= p.rot_f_max_th and rms_frot <= p.rot_f_rms_th
            )
            converged = (
                negative_mode
                and rotation_converged
                and max_f <= f_max_th
                and rms_f <= f_rms_th
                and max_dp <= dp_max_th
                and rms_dp <= dp_rms_th
            )

            # Coordinates, energy, and force metrics all describe the same center.
            coords_block = self.atoms_to_xyz_block(self.atoms)
            info = [
                "\n----------------------------------------------------------------------\n",
                f"                             Iteration: {it:<3d}                              \n\n",
                "                             Coordinates                               \n",
                "----------------------------------------------------------------------\n",
                coords_block,
                "\n",
                f"Energy:                  {E: .6f} Convergence criteria  Is converged \n",
                f"Maximum Force:         {max_f:>12.6f} {f_max_th:>12.6f}                {'Yes' if max_f <= f_max_th else 'No'}\n",
                f"RMS Force:             {rms_f:>12.6f} {f_rms_th:>12.6f}                {'Yes' if rms_f <= f_rms_th else 'No'}\n",
                f"Maximum Displacement:  {max_dp:>12.6f} {dp_max_th:>12.6f}                {'Yes' if max_dp <= dp_max_th else 'No'}\n",
                f"RMS Displacement:      {rms_dp:>12.6f} {dp_rms_th:>12.6f}                {'Yes' if rms_dp <= dp_rms_th else 'No'}\n",
                (
                    f"\nTrust radius (MW): {max_allow: .6f}  "
                    f"Step norm (MW): {step_norm_inf: .6f}  "
                    f"On boundary: {on_boundary}\n"
                    f"Curvature (kappa):      {kappa:>12.6f} Eh/Å²\n"
                ),
                f"Max Rotation Residual:  {max_frot:>12.6f} Eh/Å²\n",
                f"RMS Rotation Residual:  {rms_frot:>12.6f} Eh/Å²\n",
                f"Negative Mode:          {'Yes' if negative_mode else 'No'}\n",
                f"Mode Flip:              {'Yes' if mode_flip else 'No'}\n\n",
            ]
            log_info(info, self.output)

            if p.save_traj:
                write_all_images_xyz(traj_file, self.atoms, energy=E, iteration=it)

            if converged:
                log_info(
                    [
                        f"\nDimer converged TS candidate at iteration {it}.\n",
                        "Post-search frequency/IRC validation remains required.\n",
                    ],
                    self.output,
                )
                break

            self.atoms.set_positions(
                center.reshape(-1, 3) + step_vec.reshape(-1, 3)
            )
            alpha = (
                max(0.5 * alpha, 0.1 * p.step0)
                if on_boundary
                else min(1.2 * alpha, p.step_max)
            )

        if not converged:
            log_info(
                [
                    f"\nMaximum Iterations Reached ({p.max_iter}).\n",
                    "Dimer did not converge to a TS candidate.\n",
                ],
                self.output,
            )

        final_center = to_numpy_f64(self.atoms.get_positions()).reshape(-1)
        _, _, E_final = self._evaluate_derivatives(final_center, n)
        write_xyz(ts_file, [self.atoms], energies=[E_final])

        trajectory_message = (
            f"\nWrote Dimer trajectory to: {traj_file}\n" if p.save_traj else "\n"
        )
        log_info(
            [
                "\n---------------------------------------------------------------\n",
                "                    DIMER FINAL CANDIDATE                      \n",
                "---------------------------------------------------------------\n",
                f"Energy (final)                           ....  {E_final: .8f} Eh\n",
                "\n-----------------------------------------\n",
                "  FINAL GEOMETRY (NOT A CERTIFIED TS)\n",
                "-----------------------------------------\n",
                self.atoms_to_xyz_block(self.atoms),
                trajectory_message,
                f"Wrote Dimer final candidate to: {ts_file}\n",
            ],
            self.output,
        )
