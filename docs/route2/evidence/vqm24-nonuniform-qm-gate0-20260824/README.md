# VQM24 localized-field QM Gate-0

This is the first independent observable-training datum for the strongly convex
latent-source hybrid. The geometry is the first frozen VQM24 training record
(CH6N4). No VQM24 energy, PCM quantity, or solvation target was used.

Protocol:

- gas reference: RKS omegaB97M-V/def2-TZVPD density fitting;
- semilocal grid level 3, nonlocal 50x194 SG1;
- four geometry-only exterior point-charge modes;
- amplitudes +/-3e-4 and +/-1e-3 e;
- exterior shell from the frozen promolecular 1e-4 e/bohr^3 atomic radii plus
  1.0 angstrom clearance; no solvent or cavity.

Result: **pass-numerical-gate0**.

- gas SCF: 16 cycles, 61.99 s;
- 16 perturbed SCFs: all converged in 6--8 cycles;
- maximum electron-count error: 4.97e-14 e;
- two-step induced-MEP relative difference: 5.97e-4;
- two-step induced-dipole relative difference: 5.57e-4;
- all four central field-energy curvatures are negative.

The record supplies field-conditioned energies, induced exterior MEPs, and
induced dipoles for training. It does not fit a model or admit a capability.
