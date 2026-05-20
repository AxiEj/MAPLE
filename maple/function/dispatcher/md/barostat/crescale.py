"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it corrects the Berendsen
barostat by adding a stochastic term to the cell update.

Algorithm (Bernetti & Bussi, 2020):
    The isotropic strain ε = log(V/V0) is advanced stochastically:

        dε = β * (dt/τ_P) * (P - P_target)
           + sqrt(2 * k_B * T * β * dt / (V * τ_P)) * W

    where W ~ N(0, 1) is the discrete Wiener increment. This is the
    log-volume/strain form used by stochastic cell rescaling; the old
    first-order volume-fraction update is not used here.

    Positions and cell are scaled isotropically by μ = exp(dε / 3).
    Velocities are returned as ``v / μ`` following the Trotter-splitting
    correction described by Bernetti & Bussi.

Notes:
    - Includes stochastic volume fluctuations, unlike plain Berendsen.
    - Isotropic scaling only; anisotropic tensors not yet supported.
    - Pressure is computed from the virial theorem. In this implementation,
      calculator stress is treated as the configurational/virial contribution,
      and the kinetic term is computed explicitly from current velocities.
      If stress is unavailable, NPT fails instead of using a kinetic-only fallback.

Reference:
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
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
    Stochastic cell rescaling barostat (C-rescale).

    Isotropically rescales cell and atomic positions by advancing the
    log-volume strain variable ε = log(V/V0).
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

        # Deterministic prefactor: β * dt / τ_P  (dimensionless)
        self._det_prefactor = compressibility * timestep / tau_p

        # Stochastic noise prefactor (dimensionless, multiplied by 1/√V later):
        #   dε_noise = sqrt(2 k_B T β dt / (τ_P V)) * W
        # We precompute sqrt(2 k_B T β dt / τ_P) in units of √Å³:
        #   k_B T in eV = T * KELVIN_TO_HARTREE * HARTREE_TO_EV
        #   β in Å³/eV  = compressibility * EV_PER_ANG3_TO_BAR
        #   → product: [eV * Å³/eV * 1] = Å³  ✓
        HARTREE_TO_EV = 27.211386245988
        kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV      # eV
        beta_ang3_per_ev = compressibility * EV_PER_ANG3_TO_BAR      # Å³/eV
        self._noise_prefactor = np.sqrt(
            2.0 * kT_ev * beta_ang3_per_ev * (timestep / tau_p)
        )   # units: √Å³

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
        Apply one C-rescale barostat step: stochastically rescale cell.

        The log-volume strain increment has both a deterministic
        Berendsen-like part and a stochastic part:

            dε = β*(dt/τ_P)*(P - P_target)  +  noise * W / sqrt(V)

        Cell and positions are scaled isotropically by μ = exp(dε/3).
        Velocities are returned as a new ``velocities / μ`` array; the input
        array is not modified in-place.

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

        # Deterministic strain part (Berendsen-like): β*(dt/τ_P)*(P - P_target)
        # so that P < P_target shrinks the cell and P > P_target expands it.
        d_epsilon_det = self._det_prefactor * (pressure - self.pressure_target)

        # Stochastic part: _noise_prefactor [√Å³] / sqrt(V [Å³]) * W
        #                = sqrt(2 k_B T β dt / (τ_P V)) * W  (dimensionless strain)
        w = self.rng.standard_normal()
        d_epsilon_stoch = self._noise_prefactor / np.sqrt(volume) * w

        # New log-volume increment.  Clamp to the same per-step position
        # scaling bounds used by Berendsen, but apply the bound in log-space so
        # the C-rescale variable remains ε = log(V/V0).
        d_epsilon = d_epsilon_det + d_epsilon_stoch
        d_epsilon = float(np.clip(d_epsilon, 3.0 * np.log(0.5), 3.0 * np.log(2.0)))
        mu = float(np.exp(d_epsilon / 3.0))

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell() * mu, scale_atoms=True)

        return pressure, velocities / mu
