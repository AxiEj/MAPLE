"""Physical constants, unit conversions, and the force-unit helper for MD.

Leaf module: depends only on numpy/ASE.  ``maple.function.dispatcher.md.utils``
re-exports every name here for backward compatibility.
"""

import numpy as np
from ase import Atoms


# ========== Physical Constants and Unit Conversions ==========

# Temperature conversions
KELVIN_TO_HARTREE = 3.1668114e-6  # k_B in Hartree/K
HARTREE_TO_KELVIN = 1.0 / KELVIN_TO_HARTREE

# Mass conversions
AMU_TO_AU = 1822.888486209  # atomic mass unit to atomic units

# Time conversions
FS_TO_AU = 41.341374575751  # femtoseconds to atomic units
AU_TO_FS = 1.0 / FS_TO_AU

# Energy conversions
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Length conversions
# NIST CODATA 2018: 1 Bohr = 0.529177210903 Å (exact to 12 sig. fig.)
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Force conversions
# MAPLE calculators (AIMNet2, MACE, UMA) return forces in Ha/Å.
# The MD integrator needs Ha/Bohr (atomic units).
# Ha/Å → Ha/Bohr:  F[Ha/Bohr] = F[Ha/Å] × (dr_Å / dr_Bohr) = F[Ha/Å] × BOHR_TO_ANGSTROM
# (1 Bohr = 0.5292 Å, so force per Bohr is smaller than force per Å)
HA_PER_ANG_TO_AU = BOHR_TO_ANGSTROM  # Ha/Å → Ha/Bohr ≈ 0.5292

# Legacy alias kept for backward compatibility (was used when forces were assumed eV/Å)
EV_PER_ANG_TO_AU = 1.0 / (27.211386245988 * BOHR_TO_ANGSTROM)  # ≈ 0.019447

# Pressure unit conversions
# Derivation: 1 eV = 1.6021766208e-19 J, 1 Å³ = 1e-30 m³ → 1 eV/Å³ = 1.6021766208e11 Pa = 1.6021766208e6 bar
EV_PER_ANG3_TO_BAR = 1.6021766208e-19 / 1e-30 * 1e-5   # eV/Å³ → bar
BAR_TO_EV_PER_ANG3 = 1.0 / EV_PER_ANG3_TO_BAR          # bar → eV/Å³

# Kinetic energy conversion for pressure calculation
# 1 amu·Å²/fs² = 1.66054e-27 kg × 1e-20 m² / 1e-30 s² = 1.66054e-17 J
# → 1.66054e-17 / 1.60218e-19 eV ≈ 103.6427 eV
AMU_ANG2_PER_FS2_TO_EV = 1.03642695e+2

# Default isothermal compressibility (liquid water at 300 K, 1 bar)
# Reference: CRC Handbook of Chemistry and Physics
DEFAULT_COMPRESSIBILITY = 4.5e-5   # 1/bar

# Velocity representation metadata
VELOCITY_REPR_STANDARD = "standard"
VELOCITY_REPR_LFMIDDLE_CARRIED = "lfmiddle_carried"
_VALID_VELOCITY_REPRESENTATIONS = {
    VELOCITY_REPR_STANDARD,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
}


def forces_au(atoms: Atoms) -> np.ndarray:
    """Return calculator forces in atomic units (Ha/Bohr).

    Single definition of the ``get_forces() * HA_PER_ANG_TO_AU`` conversion that
    the integrator and the ensemble loops use, so the conversion constant is not
    duplicated at every force read.  This only reads (and caches via ASE) forces;
    it does not alter integration semantics.
    """
    return np.asarray(atoms.get_forces(), dtype=float) * HA_PER_ANG_TO_AU
