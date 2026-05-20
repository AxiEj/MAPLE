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
