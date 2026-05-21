"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it corrects the Berendsen
barostat by adding a stochastic term to the cell update.

Algorithm — reversible λ = √V integrator (Bernetti & Bussi, 2020, §II.B):
    Stochastic cell rescaling can be propagated either as the volume logarithm
    ε = log(V/V0) (a simple Euler scheme, NOT time-reversible) or as the
    square-root-volume variable λ = √V.  We propagate λ, the reversible form the
    paper recommends for production: its noise amplitude is *constant* (it does
    not depend on V), which removes the multiplicative-noise discretization bias
    of the ε form and lets the run define a conserved "effective energy" whose
    drift diagnoses integration quality (the NPT analogue of NVE energy drift).

        dλ = -(β λ)/(2 τ_P) · (P_0 - P_int - k_B T/(2V)) dt
           + sqrt(k_B T β / (2 τ_P)) · dW

    The -k_B T/(2V) term is the Itô correction from the V → √V change of
    variable (it follows exactly from applying Itô's lemma to the ε-form SDE;
    verified analytically).  ``W ~ N(0, 1)`` is the discrete Wiener increment.

    Per step λ is re-derived from the current volume, advanced by the equation
    above, and the cell + positions are scaled isotropically by
    μ = (V_new/V)^{1/3} = (λ_new/λ)^{2/3}; velocities are returned as ``v / μ``
    (Bernetti & Bussi "Formulation A", scaled momenta).  Because λ is recomputed
    from the actual volume each step, the per-step stability clamp on μ cannot
    make the strain variable drift away from the true log-volume.

Effective-energy monitoring:
    The energy the barostat injects each step (the change in K + U + P_0·V it
    causes) is accumulated by the NPT driver into the same external-work ledger
    as the thermostat, so the reported conserved quantity
    H̃ = K + U + P_0·V − Σ ΔW_ext is constant under exact dynamics and its
    residual drift is the integrator-quality diagnostic.

Scope / honesty:
    - Production isotropic stochastic pressure coupling: it generates genuine
      volume fluctuations (unlike Berendsen) and, in the reversible λ form with
      effective-energy monitoring, is suitable for production NPT averages.
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


class CRescaleBarostat:
    """
    Stochastic cell rescaling barostat (C-rescale), reversible λ = √V form.

    Isotropically rescales cell and atomic positions by advancing the
    square-root-volume variable λ = √V (Bernetti & Bussi 2020, §II.B), the
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

        # Reversible λ = √V integrator prefactors (Bernetti & Bussi 2020, §II.B):
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
    ) -> tuple[float, np.ndarray]:
        """
        Apply one C-rescale barostat step: stochastically rescale the cell.

        Advances λ = √V by the reversible Bernetti & Bussi update

            dλ = (β·dt/2τ_P)·λ·(P_int + k_BT/(2V) − P_0)
               + sqrt(k_BT·β·dt/2τ_P)·W

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

        Returns
        -------
        tuple[float, np.ndarray]
            ``(pressure, rescaled_velocities)`` where pressure is the
            instantaneous pre-rescaling pressure in bar.
        """
        pressure_input = velocities if pressure_velocities is None else pressure_velocities
        pressure = self.get_pressure(pressure_input)
        volume = self.atoms.get_volume()   # Å³
        if volume <= 0.0 or not np.isfinite(volume):
            raise ValueError(f"C-rescale requires a finite positive cell volume, got {volume!r}.")

        # Reversible λ = √V update (Bernetti & Bussi 2020, §II.B):
        #   dλ = (β·dt/2τ_P)·λ·(P_int + k_BT/(2V) − P_0) + sqrt(k_BT β dt/2τ_P)·W
        # The k_BT/(2V) term is the Itô correction from the V → √V change of
        # variable, formed in bar to combine with the pressures.  Sign: P_int >
        # P_0 expands the cell, P_int < P_0 shrinks it.
        lam = np.sqrt(volume)
        kT_over_2v_bar = (self._half_kT_ev / volume) * EV_PER_ANG3_TO_BAR
        d_lam_det = (
            self._lam_det_prefactor * lam
            * (pressure + kT_over_2v_bar - self.pressure_target)
        )
        w = self.rng.standard_normal()
        d_lam_stoch = self._lam_noise_prefactor * w
        lam_new = lam + d_lam_det + d_lam_stoch

        # Isotropic length scale μ = (V_new/V)^{1/3} = (λ_new/λ)^{2/3}, clamped to
        # the Berendsen-style per-step [0.5, 2.0] bound for stability.  The
        # squared ratio keeps μ real and positive even for a pathological step,
        # and λ is re-derived from the actual volume next step so the clamp never
        # makes the strain variable drift from the true log-volume.
        vol_ratio = float(np.clip((lam_new / lam) ** 2, 0.125, 8.0))
        mu = vol_ratio ** (1.0 / 3.0)

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell() * mu, scale_atoms=True)

        return pressure, velocities / mu
