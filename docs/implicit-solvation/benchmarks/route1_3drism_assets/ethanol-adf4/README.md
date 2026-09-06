# Ethanol ADF4 3D-RISM asset

This directory freezes one **research-only**, four-site united-atom ethanol solvent
susceptibility for the Route 1 non-water numerical pilot. It is not a runtime
solvent provider and carries no solvation-accuracy claim.

## Molecular model

The H/O/CH2/CH3 site parameters and geometry are re-expressed from Table 7 of
the [SCM ADF 3D-RISM documentation](https://www.scm.com/doc/ADF/Input/3D-RISM.html).
That table is a parameter source only and is not a runtime dependency.

The documented point charges _q<sub>e</sub>_, Lennard-Jones energies ε, and
conventional Lennard-Jones diameters σ are converted to Amber MDL fields by

\[
q_{\mathrm{MDL}} = 18.2223 q_e,
\qquad
(R_{\min}/2)_{\mathrm{MDL}} = 2^{-5/6}\sigma,
\qquad
\epsilon_{\mathrm{MDL}} = \epsilon.
\]

## Thermodynamic state

- temperature: 298.15 K;
- molar density: 17.0499 mol/L (rounded). This is derived from a source density of
  0.78546 g/cm<sup>3</sup> at 298.15 K and the conventional ethanol molecular mass
  46.06844 g/mol; literature values around this condition can differ by a few
  1e-4 g/cm<sup>3</sup>. The explicit MDL
  site masses in this manifest are 1.008 + 16.000 + 14.026 + 15.034 = 46.068
  using the AmberTools 26 `parm10.dat` atomic-mass convention
  (`C=12.01`, `H=1.008`, `O=16.00`; source SHA-256 pinned in the manifest).
  These are the model masses used for compatibility checks, while 46.06844 is
  the conventional molar mass used for reporting 17.0499 M.
- static dielectric constant: 24.35 at 25 °C;
- 1D theory/closure: DRISM/KH;
- radial grid: 4096 points at 0.025 Å;
- convergence tolerance: \(10^{-8}\).

Primary references for this candidate:
- [ADF Table 7 solvent model source](https://www.scm.com/doc/ADF/Input/3D-RISM.html)
- [density source (Calvar et al., 2010)](https://doi.org/10.1021/je900998f)
- [dielectric ThermoML entry](https://trc.nist.gov/ThermoML/10.1021/je060248p.html)

Amber field mapping note: Table 7 also reports `Weight=47.07`, but this file uses the explicit
four-site masses `1.008, 16.000, 14.026, 15.034` (sum 46.068) in the manifest and conversion workflow.

## Generation and containment

`ethanol_adf4_kh.xvv` was generated from the committed MDL and input with the
AmberTools 26 `rism1d` executable whose SHA-256 is
`7c5c87d61ead9802c7a2df1a71a4d528d4da2db357e654d5514c603fd4c6918d`.
The primary RISM solve converged at residual
`9.5735929061237814e-09` after 137 iterations; the temperature-derivative solve
converged at `9.5695710460183020e-09` after 79 iterations.
The sealed `generation_evidence.json` cross-hashes the MDL, input, XVV, and
committed `rism1d` output that records both convergence sequences. The asset
manifest and downstream pilot protocol pin the evidence JSON and stdout hashes.

No MNSol, FreeSolv, or other experimental solvation label was opened during
model construction, susceptibility generation, or numerical threshold selection.
Passing the asset audit proves only cross-file physical-input consistency. Passing
the downstream single-solvent pilot proves only numerical convergence and
sensitivity/repeatability gates for one fixed solute.
