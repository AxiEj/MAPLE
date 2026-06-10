"""Shared ASE unit conversion constants for MAPLE calculators.

CODATA recommended value, 2018 / 2022. NIST CUU page:
https://physics.nist.gov/cgi-bin/cuu/Value?hrev
1 Hartree = 27.211386245988(53) eV.
"""

EV2HARTREE = 1.0 / 27.211386245988

# ASE calculator stress is passed through in its native Voigt convention:
# eV / Å^3, with ASE's sign convention.  MAPLE converts energy/forces from eV
# to Hartree units, but pressure code consumes stress in this explicit ASE unit.
ASE_STRESS_UNIT = "eV/A^3"

# MAPLE MD internal energy/force unit contract.  Every MAPLE-native calculator
# returns energy in Hartree (converted via EV2HARTREE, or Hartree-native for ANI)
# and forces in Hartree/Å (the autograd gradient of the Hartree energy w.r.t.
# positions in Å).  The MD layer therefore consumes get_potential_energy() and
# get_forces() as Ha / Ha·Å⁻¹ without re-checking the source each step.  These are
# the values declared on CalcABC and enforced at MD admission.
MAPLE_ENERGY_UNIT = "Ha"
MAPLE_FORCE_UNIT = "Ha/A"

# Whitelisted source units an explicit ASE adapter may declare and convert FROM.
# Free-form unit strings are rejected: an adapter must convert a known source unit
# into the MAPLE contract, never assert an arbitrary label.
SUPPORTED_ENERGY_UNITS = ("eV", "Ha")
SUPPORTED_FORCE_UNITS = ("eV/A", "Ha/A")
SUPPORTED_STRESS_UNITS = ("eV/A^3",)
