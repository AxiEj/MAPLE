# -*- coding: utf-8 -*-
"""
Intrinsic Reaction Coordinate (IRC) integrator using Hessian-based Predictor-Corrector(HPC):

- Mass-weighted coordinates and Hessian
- Predictor step via local quadratic approximation
- DWI surface fitting and mBS (modified Bulirsch–Stoer) corrector integration
- Hessian updated by BFGS/Bofill (optional periodic full recalculation)
- Forward and backward paths from the TS geometry, starting along the
  lowest negative eigenmode of the mass-weighted Hessian
- Deterministic endpoint-to-TS-to-endpoint merge with the exact validated TS
  inserted once and the lower endpoint energy used only as the dE reference

Units:
- Cartesian coordinates: Å
- Forces: Eh/Å
- Energies: Eh
- Mass-weighted coordinates: sqrt(amu) * Å (via masses_D)
"""

from collections import deque
from typing import List, Optional, Dict, Tuple

import numpy as np
from ase import Atoms

from ..path import (
    assemble_irc_path,
    enforce_irc_path_admission,
    finalize_irc_branch,
    irc_force_criteria_satisfied,
    make_irc_record,
    render_irc_path_summary,
    write_irc_trajectories,
)
from ..parameters import (
    HPCParams,
    apply_irc_parameter_overrides,
    select_irc_parameter_overrides,
    validate_irc_params,
)
from ..preflight import validate_irc_transition_state
from .logger import log_info

# =============================== Utilities ===============================
BOHR_TO_ANG = 0.529177210903
KCAL_PER_EH = 627.509474


def to_f64(x):
    """Convert input to float64 numpy array or scalar."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def v1(x, n: Optional[int] = None) -> np.ndarray:
    """Flatten to 1D float64 array, optionally enforcing size."""
    v = to_f64(x).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"Expected size {n}, got {v.size}")
    return v


def masses_D(atoms: Atoms) -> np.ndarray:
    """
    Return diagonal scaling vector D (length 3N) for mass-weighted quantities.

    Convention here (consistent with earlier code):

        q_mw = q_cart / D       where D = 1/sqrt(m_i)

    so that q_mw has units sqrt(amu) * Å, and the mass-weighted Hessian is

        H_mw = D * H_cart * D

    Gradients in MW coordinates follow

        g_mw = D * g_cart
    """
    m = to_f64(atoms.get_masses())
    m = np.where(m > 0.0, m, 1.0)
    return v1(1.0 / np.sqrt(np.repeat(m, 3)))


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    n = _norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _taylor_energy(energy: float, gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> float:
    """Second-order Taylor energy expansion."""
    return float(energy + step @ gradient + 0.5 * step @ hessian @ step)


def _taylor_grad(gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Gradient of the second-order Taylor expansion."""
    return gradient + hessian @ step


class DWI:
    """
    Distance-weighted interpolation (DWI) of a local quadratic PES.
    Keeps the most recent two points (coords, energy, gradient, hessian).
    """

    def __init__(self, n: int = 4, maxlen: int = 2):
        self.n = int(n)
        if self.n <= 0 or (self.n % 2) != 0:
            raise ValueError("DWI n must be a positive even integer.")
        self.maxlen = int(maxlen)
        if self.maxlen != 2:
            raise ValueError("DWI currently supports maxlen=2 only.")

        self.coords = deque(maxlen=self.maxlen)
        self.energies = deque(maxlen=self.maxlen)
        self.gradients = deque(maxlen=self.maxlen)
        self.hessians = deque(maxlen=self.maxlen)

    def update(self, coords: np.ndarray, energy: float, gradient: np.ndarray, hessian: np.ndarray) -> None:
        self.coords.append(coords)
        self.energies.append(float(energy))
        self.gradients.append(gradient)
        self.hessians.append(hessian)

        if not (len(self.coords) == len(self.energies) == len(self.gradients) == len(self.hessians)):
            raise RuntimeError("DWI internal buffers are inconsistent.")

    def interpolate(self, at_coords: np.ndarray, gradient: bool = False):
        """Distance-weighted interpolation of energy (and optionally gradient)."""
        if len(self.coords) < 2:
            raise RuntimeError("DWI requires two points before interpolation.")

        c1, c2 = self.coords
        dx1 = at_coords - c1
        dx2 = at_coords - c2

        dx1_norm = _norm(dx1)
        dx2_norm = _norm(dx2)
        dx1_norm_n = dx1_norm ** self.n
        dx2_norm_n = dx2_norm ** self.n

        denom = dx1_norm_n + dx2_norm_n
        if denom <= 0.0:
            raise RuntimeError("DWI interpolation denominator is zero.")

        w1 = dx2_norm_n / denom
        w2 = dx1_norm_n / denom

        e1, e2 = self.energies
        g1, g2 = self.gradients
        h1, h2 = self.hessians

        t1 = _taylor_energy(e1, g1, h1, dx1)
        t2 = _taylor_energy(e2, g2, h2, dx2)
        e_dwi = w1 * t1 + w2 * t2

        if not gradient:
            return e_dwi

        t1_grad = _taylor_grad(g1, h1, dx1)
        t2_grad = _taylor_grad(g2, h2, dx2)

        n_2 = self.n // 2
        dx1_norm_n_grad = 2 * n_2 * (dx1_norm ** (2 * n_2 - 2)) * dx1
        dx2_norm_n_grad = 2 * n_2 * (dx2_norm ** (2 * n_2 - 2)) * dx2
        w1_grad = (dx2_norm_n_grad * dx1_norm_n - dx1_norm_n_grad * dx2_norm_n) / (denom ** 2)
        w2_grad = -w1_grad

        g_dwi = w1_grad * t1 + w1 * t1_grad + w2_grad * t2 + w2 * t2_grad
        return e_dwi, g_dwi


# ================================== HPC ===================================
class HPC:
    """
    Hessian-based Predictor-Corrector(HPC):

    - Works in mass-weighted coordinates.
    - Each macro step uses LQA predictor propagation followed by mBS
      corrector integration on a DWI-fitted surface.
    - Uses BFGS/Bofill updates of the mass-weighted Hessian, with optional
      periodic full recalculation.
    - Integrates forward and backward from the TS along the lowest
      negative eigenmode of H_mw, then merges paths.
    """

    def __init__(self, atoms: Atoms, output: str,
                 params: Optional[HPCParams] = None,
                 paras: Optional[dict] = None):
        self.atoms = atoms
        self.output = output
        self.p = params if params is not None else HPCParams()

        if isinstance(paras, dict):
            apply_irc_parameter_overrides(
                self.p,
                select_irc_parameter_overrides(paras, "hpc"),
            )

        # Internal state for HPC integration
        self._D: Optional[np.ndarray] = None  # mass-weight scaling vector
        self._step_len_mw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw: float = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self.mw_coords: Optional[np.ndarray] = None
        self.mw_hessian: Optional[np.ndarray] = None
        self.prev_coords: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.micro_counter: int = 0
        self._dwi: Optional[DWI] = None

    # ------------------------------ Public API ------------------------------
    def run(self) -> Dict[str, any]:
        """
        Compute forward and backward HPC IRC paths and produce a merged summary.

        Returns:
            {
                "forward": {...},
                "backward": {...},
                "summary": {...}
            }
        """
        validate_irc_params(self.p, "hpc")
        # Prepare mass weights once at TS geometry
        self._D = masses_D(self.atoms)
        self._step_len_mw = float(self.p.step_length_bohr * BOHR_TO_ANG)
        self._step_len_umw = float(self.p.step_length_bohr * BOHR_TO_ANG)

        # Validate the first-order saddle and select its projected reaction mode.
        H_cart_ts = self._get_hessian_cart()
        F_ts_cart = to_f64(self.atoms.get_forces()).reshape(-1)
        preflight = validate_irc_transition_state(
            self.atoms,
            H_cart_ts,
            F_ts_cart.reshape(-1, 3),
            self.p,
        )
        eigval = preflight.negative_eigenvalue_hartree_per_A2_amu
        v_neg_mw = preflight.negative_mode_mass_weighted

        log_info(
            [
                "\n[INFO] HPC-IRC: Validated one projected imaginary mode "
                f"at {preflight.assessment.imaginary_frequency_cm1:.6f} cm^-1 "
                f"(λ = {eigval:.6e} Eh/(Angstrom^2*amu)); "
                f"max |F| = {preflight.maximum_force_eV_per_A:.3e} eV/Angstrom; "
                f"rigid residual = {preflight.rigid_residual_cm1:.3e} cm^-1.\n"
            ],
            self.output,
        )

        # Exact validated TS record; it must appear once in the full path.
        E_ts = float(self.atoms.get_potential_energy(force_consistent=True))
        R_ts = self.atoms.get_positions().copy()
        R_ts_cart = R_ts.reshape(-1)
        transition_state_record = make_irc_record(
            energy_hartree=E_ts,
            forces_hartree_per_A=F_ts_cart,
            positions_angstrom=R_ts,
            point_kind="transition_state",
        )

        # Forward and backward HPC-IRC
        forward_log = self._one_side(
            forward=True,
            sign=+1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )
        backward_log = self._one_side(
            forward=False,
            sign=-1.0,
            q_ts_cart=R_ts_cart,
            v_neg_mw=v_neg_mw,
            E_ts=E_ts,
        )

        merged = assemble_irc_path(
            method_label="HPC",
            forward=forward_log,
            backward=backward_log,
            transition_state_record=transition_state_record,
            path_energy_tolerance_hartree=self.p.path_energy_tolerance_hartree,
        )
        log_info(render_irc_path_summary(merged), self.output)

        if self.p.write_traj:
            paths = write_irc_trajectories(
                template_atoms=self.atoms,
                output=self.output,
                forward=forward_log,
                backward=backward_log,
                summary=merged,
            )
            log_info(
                [
                    f"\n[INFO] HPC-IRC forward trajectory written to: {paths['forward']}\n",
                    f"[INFO] HPC-IRC backward trajectory written to: {paths['backward']}\n",
                    f"[INFO] HPC-IRC full trajectory written to: {paths['full']}\n",
                ],
                self.output,
            )

        enforce_irc_path_admission(
            merged,
            require_converged_endpoints=self.p.require_converged_endpoints,
        )

        return {"forward": forward_log, "backward": backward_log, "summary": merged}

    # --------------------------- Low-level helpers --------------------------
    def _cart_from_mw(self, q_mw: np.ndarray) -> np.ndarray:
        """Convert mass-weighted coordinates back to Cartesian (Å)."""
        return (q_mw * self._D).reshape(-1)

    def _mw_from_cart(self, q_cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates (Å) to mass-weighted coordinates."""
        return (q_cart / self._D).reshape(-1)

    def _unweight_len(self, dq_mw: np.ndarray) -> float:
        """Return unweighted length of a MW displacement."""
        return _norm(dq_mw * self._D)

    def _scale_mw_step(self, direction_mw: np.ndarray, step_umw: float) -> np.ndarray:
        """Scale a MW direction to a target unweighted step length."""
        denom = self._unweight_len(direction_mw)
        if denom <= 1e-16:
            return np.zeros_like(direction_mw)
        return direction_mw * (step_umw / denom)

    def _unweight_grad(self, g_mw: np.ndarray) -> np.ndarray:
        """Convert MW gradient to Cartesian gradient."""
        return g_mw / self._D

    def _get_conv_fact(self, g_mw: np.ndarray, min_fact: float = 2.0) -> float:
        """Estimate MW-to-unweighted conversion factor for step length."""
        norm_mw = _norm(g_mw)
        if norm_mw <= 1e-16:
            return min_fact
        norm_cart = _norm(self._unweight_grad(g_mw))
        return max(min_fact, norm_cart / norm_mw)

    def _get_hessian_cart(self) -> np.ndarray:
        """
        Get Cartesian Hessian (3N x 3N) from the calculator.

        Assumes atoms.calc implements get_hessian(atoms) and returns 3N x 3N
        or shape (1, 3N, 3N).
        """
        H = self.atoms.calc.get_hessian(self.atoms)
        H = to_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square 2D, got shape {H.shape}")
        return H

    def _energy_forces_from_mw(self, q_mw: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Set positions from MW coordinates, then return (E, F_cart).

        E: Eh
        F_cart: (3N,) Eh/Å
        """
        q_cart = self._cart_from_mw(q_mw)
        self.atoms.set_positions(q_cart.reshape(-1, 3))
        E = float(self.atoms.get_potential_energy(force_consistent=True))
        F_cart = to_f64(self.atoms.get_forces()).reshape(-1)
        return E, F_cart

    def _gradient_mw_from_forces(self, F_cart: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian forces (Eh/Å) to MW gradient:

            g_cart = dE/dR = -F_cart
            g_mw = D * g_cart
        """
        g_cart = -F_cart.reshape(-1)
        return self._D * g_cart

    @staticmethod
    def _perp_component(vec: np.ndarray, perp_to: np.ndarray) -> np.ndarray:
        """Return component of vec perpendicular to perp_to."""
        denom = float(np.dot(perp_to, perp_to))
        if denom <= 0.0:
            return np.zeros_like(vec)
        return vec - np.dot(perp_to, vec) * perp_to / denom

    @staticmethod
    def _bfgs_update(H: np.ndarray,
                     s: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
        """
        Symmetric BFGS update:

            H_{k+1} = H_k + (y y^T)/(y·s) - (H s s^T H)/(s^T H s)

        Safeguards if denominators are too small or curvature condition fails.
        """
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            return H

        Hy = H.dot(s)
        sTHs = float(np.dot(s, Hy))
        if sTHs <= 1e-12:
            return H

        term1 = np.outer(y, y) / ys
        term2 = np.outer(Hy, Hy) / sTHs
        return H + term1 - term2

    @staticmethod
    def _bofill_update(H: np.ndarray,
                       s: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
        """
        Bofill update:
        Combines Murtagh-Sargent and Powell-symmetric-Broyden updates using the Bofill mixing factor.
        """
        dx, dg = s, y

        z = dg - H.dot(dx)
        # MS
        ms = np.outer(z, z) / z.dot(dx)
        # PSB
        dx2 = dx.dot(dx)
        psb = ((np.outer(dx, z) + np.outer(z, dx)) / dx2) - (z.dot(dx)) * np.outer(dx, dx) / (dx2 * dx2)
        # Bofill mixing
        mix = (z.dot(dx) ** 2) / (z.dot(z) * dx2)

        dH = mix * ms + (1.0 - mix) * psb
        return H + dH

    @staticmethod
    def _newton_1d(on_sphere,
                   lambda_0: float,
                   maxiter: int = 50,
                   tol: float = 1e-10) -> float:
        """
        Simple 1D Newton root finder with finite-difference derivative.
        Used instead of scipy.optimize.newton to avoid extra dependency.
        """
        lam = float(lambda_0)
        for _ in range(maxiter):
            f = float(on_sphere(lam))
            if abs(f) < tol:
                break
            h = 1e-4 * max(1.0, abs(lam))
            f1 = float(on_sphere(lam + h))
            df = (f1 - f) / h
            if abs(df) < 1e-16:
                lam *= 0.5
                continue
            lam -= f / df
        return lam

    # --------------------------- Micro-step (HPC) ----------------------------
    def _micro_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one HPC step using LQA predictor + mBS corrector.

        Updates:
            self.mw_coords, self.mw_hessian, self.prev_coords,
            self.prev_grad, self.micro_counter

        Returns:
            dx: MW displacement for this step
            g_pred: predicted MW gradient (for diagnostics)
        """
        init_mw = self.mw_coords.copy()

        # Current energy / gradient
        e_curr, F_cart = self._energy_forces_from_mw(init_mw)
        g_curr = self._gradient_mw_from_forces(F_cart)
        if _norm(g_curr) < 1e-12:
            return np.zeros_like(g_curr), g_curr

        # Update Hessian (or optionally recalc)
        recalc = (
            self.p.hessian_recalc is not None
            and self.p.hessian_recalc > 0
            and (self.micro_counter % self.p.hessian_recalc == 0)
        )
        if recalc and self.micro_counter > 0:
            H_cart = self._get_hessian_cart()
            self.mw_hessian = (self._D[:, None] * H_cart) * self._D[None, :]
        elif (self.prev_coords is not None) and (self.prev_grad is not None):
            gradient_diff = g_curr - self.prev_grad
            coords_diff = init_mw - self.prev_coords
            if str(self.p.hessian_update).lower() == "bofill":
                self.mw_hessian = self._bofill_update(self.mw_hessian, coords_diff, gradient_diff)
            else:
                self.mw_hessian = self._bfgs_update(self.mw_hessian, coords_diff, gradient_diff)

        # Store for next update
        self.prev_coords = init_mw
        self.prev_grad = g_curr

        # Update DWI with current point (skip the very first step)
        if self._dwi is not None and self.micro_counter > 0:
            self._dwi.update(init_mw.copy(), e_curr, g_curr, self.mw_hessian.copy())

        # Predictor: LQA propagation using current Hessian
        eigvals, eigvecs = np.linalg.eigh(self.mw_hessian)
        mask = np.abs(eigvals) > 1e-8
        if not np.any(mask):
            return np.zeros_like(g_curr), g_curr

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        g_star = eigvecs.T.dot(g_curr)
        g_norm = _norm(g_curr)
        if g_norm < 1e-12:
            return np.zeros_like(g_curr), g_curr

        dt = self._step_len_mw / (float(self.p.euler_n) * g_norm)
        t = dt
        cur_length = 0.0
        for _ in range(int(self.p.euler_n)):
            dsdt = np.sqrt(np.sum((g_star ** 2) * np.exp(-2.0 * eigvals * t)))
            cur_length += dsdt * dt
            if cur_length >= self._step_len_mw:
                break
            t += dt

        alphas = (np.exp(-eigvals * t) - 1.0) / eigvals
        dx_pred = eigvecs.dot(alphas * g_star)
        pred_mw = init_mw + dx_pred

        # Evaluate predicted point and update Hessian
        E_pred, F_pred = self._energy_forces_from_mw(pred_mw)
        g_pred = self._gradient_mw_from_forces(F_pred)

        dg_pred = g_pred - g_curr
        if str(self.p.hessian_update).lower() == "bofill":
            self.mw_hessian = self._bofill_update(self.mw_hessian, dx_pred, dg_pred)
        else:
            self.mw_hessian = self._bfgs_update(self.mw_hessian, dx_pred, dg_pred)

        # Update DWI with predicted point
        if self._dwi is not None:
            self._dwi.update(pred_mw.copy(), E_pred, g_pred, self.mw_hessian.copy())

        # Corrector: mBS integration on DWI surface
        if self._dwi is not None:
            corr_mw = self._corrector_step(init_mw, self._step_len_umw, self._dwi)
        else:
            corr_mw = pred_mw

        self.mw_coords = corr_mw
        self.micro_counter += 1

        dx = self.mw_coords - init_mw
        return dx, g_pred

    def _corrector_step(self, init_mw: np.ndarray, step_length: float, dwi: DWI) -> np.ndarray:
        """mBS (modified Bulirsch–Stoer) corrector integration on DWI PES."""
        errors = []
        richardson: Dict[Tuple[int, int], np.ndarray] = {}

        for k in range(int(self.p.mbs_max_k)):
            points = int(self.p.mbs_points) * (2 ** k)
            corr_step = step_length / (points - 1)
            cur_coords = init_mw.copy()
            k_coords: List[np.ndarray] = []
            cur_length = 0.0

            while True:
                k_coords.append(cur_coords.copy())
                if abs(step_length - cur_length) < 0.5 * corr_step:
                    break

                _, gradient = dwi.interpolate(cur_coords, gradient=True)
                grad_norm = _norm(gradient)
                if grad_norm < 1e-12:
                    break
                cur_coords = cur_coords + corr_step * (-gradient / grad_norm)
                cur_length = self._unweight_len(cur_coords - init_mw)

                # Oscillation check
                if len(k_coords) > 1:
                    prev_coords = k_coords[-2]
                    if _norm(cur_coords - prev_coords) <= corr_step:
                        return prev_coords

            richardson[(k, 0)] = cur_coords

            for j in range(1, k + 1):
                richardson[(k, j)] = (
                    (2 ** j) * richardson[(k, j - 1)] - richardson[(k - 1, j - 1)]
                ) / (2 ** j - 1)

            if k > 0:
                error = float(np.sqrt(np.mean((richardson[(k, k)] - richardson[(k, k - 1)]) ** 2)))
                errors.append(error)
                if error <= float(self.p.mbs_tol):
                    break
        else:
            raise RuntimeError("mBS Richardson extrapolation did not converge.")

        return richardson[(k, k)]

    # --------------------------- One direction ------------------------------
    def _one_side(self,
                  forward: bool,
                  sign: float,
                  q_ts_cart: np.ndarray,
                  v_neg_mw: np.ndarray,
                  E_ts: float) -> Dict[str, any]:
        """
        Single-sided HPC IRC integration.

        Args:
            forward: True for forward path, False for backward.
            sign: +1.0 or -1.0 to choose direction along the negative mode.
            q_ts_cart: TS Cartesian coordinates (flattened, Å).
            v_neg_mw: selected negative eigenmode in MW basis (3N, normalized later).
            E_ts: TS reference energy (Eh).
        """
        p = self.p
        title = "FORWARD HPC-IRC" if forward else "BACKWARD HPC-IRC"

        self._print_header(title)
        self._print_conv_thresholds(p.f_max_th, p.f_rms_th)

        # Initial MW coordinates at TS
        q_ts_mw = self._mw_from_cart(q_ts_cart)

        # Initial displacement along negative mode in MW
        v_dir = _unit(v_neg_mw) * sign
        q0_mw = q_ts_mw + self._scale_mw_step(v_dir, 0.5 * self._step_len_mw)

        # Gradient at TS (for initial Hessian update)
        _, F_ts_cart = self._energy_forces_from_mw(q_ts_mw)
        g_ts_mw = self._gradient_mw_from_forces(F_ts_cart)

        # Initial Hessian at TS (MW)
        H_ts_cart = self._get_hessian_cart()
        H0_mw = (self._D[:, None] * H_ts_cart) * self._D[None, :]

        # Initial energy / forces / gradient at displaced point
        E0, F0_cart = self._energy_forces_from_mw(q0_mw)
        g0_mw = self._gradient_mw_from_forces(F0_cart)

        maxF0 = float(np.max(np.abs(F0_cart)))
        rmsF0 = float(np.sqrt(np.mean(F0_cart ** 2)))

        # Hessian update using TS -> displaced point
        if str(self.p.hessian_update).lower() == "bofill":
            H0_mw = self._bofill_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)
        else:
            H0_mw = self._bfgs_update(H0_mw, q0_mw - q_ts_mw, g0_mw - g_ts_mw)

        # Initialize HPC state
        self.mw_coords = q0_mw.copy()
        self.mw_hessian = H0_mw.copy()
        self.prev_coords = None
        self.prev_grad = None
        self.micro_counter = 0
        self._dwi = DWI(n=self.p.dwi_n, maxlen=2)
        self._dwi.update(q0_mw.copy(), E0, g0_mw, self.mw_hessian.copy())

        # Iteration 0 logging
        if p.print_each:
            self._print_iter_line(0, E0, (E0 - E_ts) * KCAL_PER_EH, maxF0, rmsF0)

        records: List[Dict[str, any]] = []
        records.append(
            {
                "E": E0,
                "maxG": maxF0,
                "rmsG": rmsF0,
                "x": self.atoms.get_positions().copy(),
            }
        )

        # Macro steps
        initial_converged = irc_force_criteria_satisfied(
            maximum_force_hartree_per_A=maxF0,
            rms_force_hartree_per_A=rmsF0,
            maximum_force_threshold_hartree_per_A=p.f_max_th,
            rms_force_threshold_hartree_per_A=p.f_rms_th,
        )
        termination_reason = (
            "force_converged" if initial_converged else "maximum_steps"
        )
        if initial_converged:
            self._print_hurray()
        iterations_attempted = 0
        macro_steps = range(0) if initial_converged else range(1, p.max_steps + 1)
        for it in macro_steps:
            iterations_attempted = it
            dx, _ = self._micro_step()
            if _norm(dx) <= 1e-12:
                termination_reason = "step_too_small"
                log_info(
                    [f"[INFO] {title}: step too small at step {it}, stopping.\n"],
                    self.output,
                )
                break

            # Energy / forces at new point
            E_new, F_new = self._energy_forces_from_mw(self.mw_coords)
            maxF = float(np.max(np.abs(F_new)))
            rmsF = float(np.sqrt(np.mean(F_new ** 2)))

            if p.print_each:
                self._print_iter_line(it, E_new, (E_new - E_ts) * KCAL_PER_EH, maxF, rmsF)

            records.append(
                {
                    "E": E_new,
                    "maxG": maxF,
                    "rmsG": rmsF,
                    "x": self.atoms.get_positions().copy(),
                }
            )

            # Convergence in terms of Cartesian forces
            if irc_force_criteria_satisfied(
                maximum_force_hartree_per_A=maxF,
                rms_force_hartree_per_A=rmsF,
                maximum_force_threshold_hartree_per_A=p.f_max_th,
                rms_force_threshold_hartree_per_A=p.f_rms_th,
            ):
                termination_reason = "force_converged"
                self._print_hurray()
                break

        return finalize_irc_branch(
            title=title,
            direction="forward" if forward else "backward",
            records=records,
            transition_state_energy_hartree=E_ts,
            termination_reason=termination_reason,
            iterations_attempted=iterations_attempted,
            maximum_force_threshold_hartree_per_A=p.f_max_th,
            rms_force_threshold_hartree_per_A=p.f_rms_th,
        )

    # ----------------------------- Printing ------------------------------
    def _print_header(self, title: str):
        log_info(
            [
                f"\n         {'*' * 61}\n",
                f"         *{title.center(59)}*\n",
                f"         {'*' * 61}\n\n",
            ],
            self.output,
        )

    def _print_conv_thresholds(self, f_max_th: float, f_rms_th: float):
        log_info(
            [
                "Iteration    E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G) \n",
                f"Convergence thresholds                {f_max_th:0.6f}  {f_rms_th:0.6f}\n",
            ],
            self.output,
        )

    def _print_iter_line(self,
                         i: int,
                         E: float,
                         dE_kcal: float,
                         maxF: float,
                         rmsF: float):
        log_info(
            [f"{i:5d}  {E:14.6f}  {dE_kcal:12.6f}    {maxF:8.6f}  {rmsF:8.6f}\n"],
            self.output,
        )

    def _print_hurray(self):
        log_info(
            [
                "\n                      *************************************************\n",
                "                      ***          THE HPC-IRC HAS CONVERGED        ***\n",
                "                      *************************************************\n\n",
            ],
            self.output,
        )
