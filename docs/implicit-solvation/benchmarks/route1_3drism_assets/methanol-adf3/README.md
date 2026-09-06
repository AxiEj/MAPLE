# Methanol ADF3 1D-RISM asset

This directory freezes one **research-only**, three-site methanol solvent
susceptibility for the Route 1 non-water numerical pilot. It is not a runtime
solvent provider and carries no solvation-accuracy claim.

## Molecular model

The H/O/CH3 site parameters and geometry are re-expressed from Table 6 of the
[SCM ADF 3D-RISM documentation](https://www.scm.com/doc/ADF/Input/3D-RISM.html).
That table describes a literature-compiled parameter set; ADF is a provenance
source only and is not a dependency of this asset.

The documented point charges \(q_e\), conventional Lennard-Jones diameters
\(\sigma\), and energies \(\epsilon\) were converted to Amber MDL fields by

\[
q_{\mathrm{MDL}} = 18.2223 q_e,
\qquad
(R_{\min}/2)_{\mathrm{MDL}} = 2^{-5/6}\sigma,
\qquad
\epsilon_{\mathrm{MDL}} = \epsilon .
\]

These are file-format conversions, not fitted solvent parameters.

## Thermodynamic state

- temperature: 298.15 K;
- molar density: 24.550 mol/L, from Goodwin's NIST equation-of-state table at
  298.15 K;
- static dielectric constant: 32.63 at 25 °C, from NBS Circular 514;
- 1D theory/closure: DRISM/KH;
- radial grid: 4096 points at 0.025 Å;
- convergence tolerance: \(10^{-8}\).

Primary state references:

- R. D. Goodwin, *Methanol Thermodynamic Properties from 176 to 673 K at
  Pressures to 700 Bar*, J. Phys. Chem. Ref. Data 16 (1987),
  [NIST reprint](https://srd.nist.gov/jpcrdreprint/1.555786.pdf).
- A. A. Maryott and E. R. Smith, *Table of Dielectric Constants of Pure
  Liquids*, NBS Circular 514,
  [NIST scan](https://nvlpubs.nist.gov/nistpubs/Legacy/circ/nbscircular514.pdf).

## Generation and containment

`methanol_adf3_kh.xvv` was generated from the committed MDL and input with the
AmberTools 26 `rism1d` executable whose SHA-256 is
`7c5c87d61ead9802c7a2df1a71a4d528d4da2db357e654d5514c603fd4c6918d`.
The primary RISM solve converged at residual
`4.1372146223171028e-09` after 90 iterations; the subsequent temperature-
derivative solve converged at `4.8501235289513124e-09` after 66 iterations.
The sealed `generation_evidence.json` cross-hashes the MDL, input, XVV, and
committed `rism1d` output that records both convergence sequences.  The asset
manifest and downstream pilot protocol also pin the evidence JSON and stdout
hashes, and the generic audit parses the terminal residuals from the bound
stdout.  This is a cross-file generation record, not an executable replay or
byte-for-byte regeneration-parity claim; the XVV `DATE` header is generated.

No MNSol, FreeSolv, or other experimental solvation label was opened during
model construction, susceptibility generation, or numerical threshold
selection. Passing the asset audit proves only cross-file physical-input
consistency. Passing the downstream single-solvent pilot proves only numerical
convergence and discretization/repeatability gates for one fixed solute.
