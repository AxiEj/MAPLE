"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it corrects the Berendsen
barostat by adding a stochastic term to the cell update.

Algorithm — reversible λ = √V integrator (Bernetti & Bussi, 2020, Eq. 7):
    Stochastic cell rescaling can be propagated either as the volume logarithm
    ε = log(V/V0) (a simple Euler scheme, NOT time-reversible) or as the
    square-root-volume variable λ = √V.  We propagate λ, the reversible form the
    paper recommends for production: its noise amplitude is *constant* (it does
    not depend on V), which removes the multiplicative-noise discretization bias
    of the ε form and lets the run track an effective-energy diagnostic whose
    drift monitors integration quality (the NPT analogue of NVE energy drift).

    Scheme boundary (honest): when used through the production NPT driver
    (v-rescale + c-rescale), this implements the paper's "reversible Euler
    integrator" (their Table I): propagate √V by a finite-difference of Eq. 7,
    then perform a full Velocity Verlet step after recomputing forces at the
    volume-changed geometry.  It is NOT the paper's symmetric "Trotter
    integrator", which interleaves the volume move with velocity Verlet; that is
    a heavier scheme and is not used here.

        dλ = -(β λ)/(2 τ_P) · (P_0 - P_int - k_B T/(2V)) dt
           + sqrt(k_B T β / (2 τ_P)) · dW

    The -k_B T/(2V) term is the Itô correction from the V → √V change of
    variable (it follows exactly from applying Itô's lemma to the ε-form SDE;
    verified analytically).  ``W ~ N(0, 1)`` is the discrete Wiener increment.

    Per scheduled barostat propagation (every N_P MD steps in the production
    driver) λ is re-derived from the current volume, advanced by the equation
    above over N_P·dt, and the cell + positions are scaled isotropically by
    μ = (V_new/V)^{1/3} = (λ_new/λ)^{2/3}; velocities are returned as ``v / μ``
    (Bernetti & Bussi "Formulation A", scaled momenta).  Because λ is recomputed
    from the actual volume each step, the per-step stability clamp on μ cannot
    make the strain variable drift away from the true log-volume.

Effective-energy diagnostic:
    The energy the barostat injects on scheduled N_P propagation steps (the
    change in K + U + P_0·V it causes) is accumulated by the NPT driver into the
    same external-work ledger as the thermostat.  MAPLE reports
    H̃ = K + U + P_0·V − Σ ΔW_ext as an effective-energy diagnostic; its residual
    drift is an integration-quality check, not a standalone proof that the NPT
    ensemble implementation is production-ready.

Scope / honesty:
    - Production isotropic stochastic pressure coupling: it generates genuine
      volume fluctuations (unlike Berendsen) and is the reference-aligned
      stochastic isotropic NPT path that must pass MAPLE's production acceptance
      gates before production use.
    - Isotropic (hydrostatic) scaling ONLY.  This is not a Parrinello-Rahman /
      MTTK / Nosé-Hoover anisotropic-cell barostat: it scales the cell by a
      single scalar μ and cannot relax non-hydrostatic stress, cell shape, or
      lattice angles.  Do not use it where anisotropic cell response matters
      (e.g. solids under shear or non-cubic stress); that is a roadmap item.
    - Pressure is computed from the virial theorem. In this implementation,
      calculator stress is treated as the configurational/virial contribution,
      and the kinetic term is computed explicitly from current velocities.
      If stress is unavailable, NPT fails instead of using a kinetic-only fallback.

Reference:
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020); arXiv:2006.09250.
"""

from typing import Optional

import numpy as np
from ase import Atoms

from ..utils import (
    KELVIN_TO_HARTREE,
    EV_PER_ANG3_TO_BAR,
    DEFAULT_COMPRESSIBILITY,
    compute_instantaneous_pressure,
)


class MDBarostatClampError(RuntimeError):
    """Raised when the c-rescale per-step stability clamp fires in a production run.

    A fired clamp means the per-step volume ratio left the Berendsen-style bound, so the
    stochastic-cell-rescaling ensemble is truncated and no longer samples NPT.  The NPT
    driver raises this immediately (rather than after the run) unless the run opted into
    ``allow_barostat_clamp`` for an EXPERIMENTAL equilibration.
    """


class CRescaleBarostat:
    """
    Stochastic cell rescaling barostat (C-rescale), reversible λ = √V form.

    Isotropically rescales cell and atomic positions by advancing the
    square-root-volume variable λ = √V (Bernetti & Bussi 2020), the
    reversible integrator with constant noise amplitude.
    """

    def __init__(
        self,
        atoms: Atoms,
        pressure: float,
        temperature: float,
        tau_p: float,
        timestep: float,
        compressibility: float = DEFAULT_COMPRESSIBILITY,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system (must have a periodic cell)
        pressure : float
            Target pressure in bar
        temperature : float
            Target temperature in Kelvin (needed for the noise term)
        tau_p : float
            Pressure relaxation time in fs.
            Larger τ_P = weaker coupling.  Recommended: 2000–5000 fs for MLP.
        timestep : float
            MD timestep in fs
        compressibility : float
            Isothermal compressibility in 1/bar (default: water ~4.5e-5)
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.atoms = atoms
        self.pressure_target = pressure          # bar
        self.temperature = temperature           # K
        self.tau_p = tau_p                       # fs
        self.timestep = timestep                 # fs
        self.compressibility = compressibility   # 1/bar
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-step stability-clamp accounting (WS-C).  The μ clamp is a Berendsen-style
        # guard; if it ever fires the stochastic-cell-rescaling ensemble is truncated, so
        # the NPT driver treats a clamp as fatal in production and records the count and
        # the largest log-volume excursion it had to clip in the run manifest.
        self.last_clamped = False
        self.clamp_count = 0
        self.max_abs_log_excursion = 0.0

        # Reversible λ = √V integrator prefactors (Bernetti & Bussi 2020):
        #   dλ = _lam_det_prefactor · λ · (P_int + k_BT/(2V) − P_0)   [√Å³]
        #      + _lam_noise_prefactor · W                            [√Å³, V-independent]
        # k_B T in eV; β re-expressed in Å³/eV so the pressure terms cancel to bar.
        HARTREE_TO_EV = 27.211386245988
        kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV      # eV
        beta_ang3_per_ev = compressibility * EV_PER_ANG3_TO_BAR      # Å³/eV

        # Deterministic: β·dt/(2 τ_P) [1/bar]; × λ [√Å³] × ΔP [bar] → √Å³.
        self._lam_det_prefactor = compressibility * timestep / (2.0 * tau_p)
        # Stochastic: sqrt(k_B T β dt / (2 τ_P)) [√Å³]; CONSTANT (no 1/√V) — the
        # reversibility advantage of the √V form over the ε (log-volume) form.
        self._lam_noise_prefactor = np.sqrt(
            kT_ev * beta_ang3_per_ev * timestep / (2.0 * tau_p)
        )
        # Half of k_B T in eV; the Itô correction term k_BT/(2V) is formed in bar
        # inside apply() as (_half_kT_ev / V) * EV_PER_ANG3_TO_BAR.
        self._half_kT_ev = 0.5 * kT_ev

    def get_pressure(self, velocities: np.ndarray) -> float:
        """
        Compute instantaneous pressure in bar via the virial theorem.

        P = (2*KE + W) / (3*V)

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        float
            Instantaneous pressure in bar
        """
        return compute_instantaneous_pressure(self.atoms, velocities)

    def apply(
        self,
        velocities: np.ndarray,
        pressure_velocities: Optional[np.ndarray] = None,
        timestep_multiplier: float = 1.0,
    ) -> tuple[float, np.ndarray]:
        """
        Apply one C-rescale barostat step: stochastically rescale the cell.

        Advances λ = √V by the reversible Bernetti & Bussi update

            dλ = (β·Δt/2τ_P)·λ·(P_int + k_BT/(2V) − P_0)
               + sqrt(k_BT·β·Δt/2τ_P)·W

        where ``Δt = timestep_multiplier × timestep``.  ``timestep_multiplier``
        is the Bernetti-Bussi ``N_P`` multiple-time-step stride: if the barostat
        is only propagated every ``N_P`` MD steps, the SDE is advanced over the
        whole ``N_P·dt`` interval rather than doing a one-step update after simply
        skipping the intervening pressure-control steps.

        then scales cell and positions isotropically by μ = (λ_new/λ)^{2/3} and
        returns a new ``velocities / μ`` array (the input array is not modified
        in-place).  The NPT driver recomputes forces at the rescaled geometry,
        completing the reversible step.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units
        pressure_velocities : np.ndarray, optional
            Velocities to use for the kinetic pressure term when they differ
            from the propagated velocity state (for example LF-Middle carried
            velocities in Langevin paths).
        timestep_multiplier : float, optional
            Multiple of the base MD timestep used for this barostat propagation.
            For the Reversible-Euler ``N_P`` scheme this is exactly ``N_P`` on a
            barostat step.  Must be positive.

        Returns
        -------
        tuple[float, np.ndarray]
            ``(pressure, rescaled_velocities)`` where pressure is the
            instantaneous pre-rescaling pressure in bar.
        """
        try:
            timestep_multiplier = float(timestep_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"C-rescale timestep_multiplier must be a positive finite number, got {timestep_multiplier!r}."
            ) from exc
        if timestep_multiplier <= 0.0 or not np.isfinite(timestep_multiplier):
            raise ValueError(
                f"C-rescale timestep_multiplier must be a positive finite number, got {timestep_multiplier!r}."
            )

        pressure_input = velocities if pressure_velocities is None else pressure_velocities
        pressure = self.get_pressure(pressure_input)
        volume = self.atoms.get_volume()   # Å³
        if volume <= 0.0 or not np.isfinite(volume):
            raise ValueError(f"C-rescale requires a finite positive cell volume, got {volume!r}.")

        # Reversible λ = √V update (Bernetti & Bussi 2020, Eq. 7):
        #   dλ = (β·Δt/2τ_P)·λ·(P_int + k_BT/(2V) − P_0) + sqrt(k_BT β Δt/2τ_P)·W
        # with Δt = timestep_multiplier × base timestep.  This is the N_P
        # multiple-time-step propagation interval, not a skipped single-step update.
        # The k_BT/(2V) term is the Itô correction from the V → √V change of
        # variable, formed in bar to combine with the pressures.  Sign: P_int >
        # P_0 expands the cell, P_int < P_0 shrinks it.
        lam = np.sqrt(volume)
        kT_over_2v_bar = (self._half_kT_ev / volume) * EV_PER_ANG3_TO_BAR
        d_lam_det = (
            self._lam_det_prefactor * timestep_multiplier * lam
            * (pressure + kT_over_2v_bar - self.pressure_target)
        )
        w = self.rng.standard_normal()
        d_lam_stoch = self._lam_noise_prefactor * np.sqrt(timestep_multiplier) * w
        lam_new = lam + d_lam_det + d_lam_stoch

        if (not np.isfinite(lam_new)) or lam_new <= 0.0:
            self.last_clamped = True
            self.clamp_count += 1
            self.max_abs_log_excursion = float("inf")
            raise MDBarostatClampError(
                "C-rescale sqrt-volume lambda became non-positive or non-finite "
                f"(lambda={lam:.8e}, lambda_new={lam_new:.8e}, volume={volume:.8e} A^3, "
                f"pressure={pressure:.8e} bar). A λ sign crossing is not a physical "
                "volume state and must not be reflected into a positive volume ratio."
            )

        # Isotropic length scale μ = (V_new/V)^{1/3} = (λ_new/λ)^{2/3}, clamped to
        # the Berendsen-style per-step [0.5, 2.0] bound for stability.
        raw_ratio = float((lam_new / lam) ** 2)
        vol_ratio = min(max(raw_ratio, 0.125), 8.0)
        # Record whether the stability bound actually clipped this step.  raw_ratio is a
        # square, hence strictly positive, so the log excursion is always well defined.
        self.last_clamped = raw_ratio < 0.125 or raw_ratio > 8.0
        if self.last_clamped:
            self.clamp_count += 1
            self.max_abs_log_excursion = max(
                self.max_abs_log_excursion,
                abs(float(np.log(raw_ratio)) - float(np.log(vol_ratio))),
            )
        mu = vol_ratio ** (1.0 / 3.0)

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell() * mu, scale_atoms=True)

        return pressure, velocities / mu
