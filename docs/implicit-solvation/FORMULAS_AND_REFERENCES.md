# Formula and reference ledger

This file is the human-review ledger for MAPLE's route-1 PB/GB and route-2
MACE-POLAR/SMD implementations. Equations implemented directly in MAPLE are
listed explicitly; upstream continuum and charge-generation algorithms are
called through their documented provider interfaces rather than reimplemented
from memory.

## Fixed-charge composition

MAPLE adds only the solvent correction to the MLIP gas potential:

\[
E_{\mathrm{solution}}(R)=E_{\mathrm{MLIP,gas}}(R)
 +\Delta G_{\mathrm{polar}}(R,q)+\Delta G_{\mathrm{nonpolar}}(R).
\]

No OpenMM, APBS, or Amber gas-phase molecular-mechanics energy is added.

## QEq-GTO

MAPLE's `qeq-gto` profile uses the original QEq atomic parameters and
hydrogen self-consistency, with a specified single-Gaussian approximation to
the original Slater charge densities.  For each atom,

\[
\zeta_i^0=\lambda\frac{2n_i+1}{2R_i^{\mathrm{bohr}}},
\qquad \lambda=0.5,
\]

where (R_i) is the UFF covalent radius and (n_i) is the principal
quantum number.  The fitted Gaussian exponent and two-center integral are

\[
\alpha_i=c_{n_i}\frac{\zeta_i^2}{n_i},\qquad
\beta_{ij}=\sqrt{\frac{2\alpha_i\alpha_j}{\alpha_i+\alpha_j}},
\]

\[
J_{ij}^{\mathrm{GTO}}(r)=E_h
\frac{\operatorname{erf}(\beta_{ij}r^{\mathrm{bohr}})}{r^{\mathrm{bohr}}}.
\]

The fitting coefficients are (c_1=0.270917), (c_2=0.098800),
(c_3=0.055600), (c_4=0.039100), and (c_5=0.029600).  The runtime
domain is H, C, N, O, F, P, S, Cl, Br, and I.

Hydrogen is not frozen at its neutral-atom parameters.  Following equations
20--21 of Rappé and Goddard,

\[
\zeta_H(q_H)=\zeta_H^0+q_H,
\qquad
J_{HH}(q_H)=J_{HH}^0\left(1+\frac{q_H}{\zeta_H^0}\right).
\]

The corresponding published electrostatic energy is

\[
E_{\mathrm{QEq}}=
\sum_{i\notin H}\left(\chi_i^0q_i+\frac12J_{ii}^0q_i^2\right)
+\sum_{i\in H}\left[\chi_i^0q_i+\frac12J_{ii}^0
\left(1+\frac{q_i}{\zeta_H^0}\right)q_i^2\right]
+\sum_{i<j}q_iq_jJ_{ij},
\qquad \sum_iq_i=Q.
\]

Every SCF iteration rebuilds the hydrogen diagonal and every H--X/H--H pair
integral, solves the charge-constrained KKT system, then mixes the new charges
with damping 0.4.  MAPLE fails closed outside the published
(-1<q_H<1) range, on non-convergence, or when the charge/KKT residual gates
fail.  This repairs both simplifications explicitly documented by Open Babel;
Caltech `cheq` 0.5.1 supplies the permissively licensed GTO mapping and
parameter cross-check, but its fixed pair matrix is not used as the complete
hydrogen-SCF algorithm.

This is a formula-locked **QEq-GTO profile**, not the exact STO numerical
model reported in the 1991 tables.  MAPLE does not relabel its GTO regression
charges as the original paper's STO charge benchmarks.

The original QEq SCF neglects the full charge derivative of the
charge-dependent hydrogen exponent.  It is therefore not a strict stationary
point of the displayed QEq energy and is used only for `mode=fixed`.

For `mode=polarizable`, MAPLE implements the consistent-QEq derivative from
Ogawa et al.  For a hydrogen (i), this adds the derivative of the cubic
one-center term and every hydrogen-dependent pair integral:

\[
\frac{\partial E}{\partial q_i}=\chi_i^0+J_{ii}^0q_i
+\frac{3J_{ii}^0}{2\zeta_H^0}q_i^2
+\sum_{j\ne i}\left(J_{ij}+q_i\frac{\partial J_{ij}}{\partial q_i}\right)q_j.
\]

The GB polar contribution adds (K_{GB}q).  MAPLE solves the resulting
nonlinear constrained minimum with an analytic charge gradient, then requires
charge conservation, a KKT residual below `2e-6 eV`, and a nonnegative
projected charge-Hessian eigenvalue.  Only this CQEq path uses the envelope
theorem for polarizable forces.  This is a mathematically consistent but still
experimental continuum coupling; its parameters were not jointly trained with
the Amber GB radii or nonpolar term.

References:

- A. K. Rappé and W. A. Goddard III, “Charge equilibration for molecular
  dynamics simulations,” *J. Phys. Chem.* **95** (1991),
  DOI `10.1021/j100161a070`.
- J. Chen and T. J. Martínez, “Charge conservation in electronegativity
  equalization and its implications for the electrostatic properties of
  fluctuating-charge models,” *J. Chem. Phys.* **131**, 044114 (2009),
  DOI `10.1063/1.3183167`.
- T. Ogawa et al., “Consistent Charge Equilibration Method Combined with
  Universal Force Field: Application to Amino Acid Molecules,”
  *Chem-Bio Informatics Journal* **3**, 78--85 (2003),
  DOI `10.1273/cbij.3.78`, for the explicit consistency problem and full
  charge derivatives required by a variational QEq force model.
- L. G. Dias, K. Shimizu, J. P. S. Farah, and H. Chaimovich, “A simple method
  for the fast calculation of charge redistribution of solutes in an implicit
  solvent model,” *Chemical Physics* **282** (2002), DOI
  `10.1016/S0301-0104(02)00717-6`, for the self-consistent QEq/continuum
  precedent.
- Caltech `cheq` 0.5.1, MIT license, commit
  `f71eeb4ab3790557fb244a8f09fa60e255ee6274`, for the UFF-radius parameter
  table and the explicitly versioned STO-to-GTO fitting coefficients.
- Open Babel `qeq.cpp` at commit
  `0e94434fa75c9f61095023e3c12e0d5f2ac035ff`, reviewed as a rejected
  simplification because its own documentation states that charge-dependent
  hydrogen hardness and screening are omitted.

## Amber GB models

MAPLE calls OpenMM's upstream `CustomAmberGBForceBase` implementations and
their `getStandardParameters(topology)` radius assignment.  It does not copy the
HCT/OBC/GBn descreening equations.

The primary hydration benchmark uses OpenMM's ACE term, exactly as implemented
by `customgbforces._createEnergyTerms`:

\[
G_{np}^{\mathrm{ACE}}=\sum_i 28.3919551\,(r_i+0.14)^2
\left(\frac{r_i}{B_i}\right)^6,
\]

with OpenMM's internal nanometre/kJ-mol unit convention.  This term is traced to
Schaefer and Karplus, “A Comprehensive Analytical Treatment of Continuum
Electrostatics,” *J. Phys. Chem.* (1996), DOI `10.1021/jp9521621`.

ACE is **not** relabelled as Amber `gbsa=1`.  Amber documents `gbsa=1` as LCPO
with

\[
G_{np}^{\mathrm{LCPO}}=\gamma A_{\mathrm{LCPO}},\qquad
\gamma=0.005\ \mathrm{kcal\,mol^{-1}\,\mathring A^{-2}}.
\]

Therefore independent complete-energy parity uses OpenMM 8.5's separate
`LCPOForce`, while experimental-accuracy ranking continues to use the
predeclared ACE profile.  Mixing those two nonpolar definitions would be a
model change, not a numerical tolerance.

OpenMM 8.5.2's generic `GBSAGBn2Force` table has no Amber-specific phosphorus
alpha/beta/gamma entry.  MAPLE consequently rejects GBn2 for P-containing
molecules instead of silently accepting OpenMM's generic default.  Other GB
models still run, and the full benchmark records that method-specific failure
in its original denominator.

- HCT: Hawkins, Cramer, Truhlar, “Pairwise solute descreening of solute charges
  from a dielectric medium,” *Chem. Phys. Lett.* (1995),
  DOI `10.1016/0009-2614(95)01082-K`.
- OBC-I/OBC-II: Onufriev, Bashford, Case, “Exploring protein native states and
  large-scale conformational changes with a modified generalized Born model,”
  *Proteins* (2004),
  DOI `10.1002/prot.20033`.
- GBn: Mongan et al., “Generalized Born model with a simple, robust molecular
  volume correction,” *J. Chem. Theory Comput.* (2007),
  DOI `10.1021/ct600085e`.
- GBn2: Nguyen et al., “Improved Generalized Born Solvent Model Parameters for
  Protein Simulations,” *J. Chem. Theory Comput.* (2013),
  DOI `10.1021/ct3010485`.
- LCPO: Weiser, Shenkin, Still, “Approximate atomic surfaces from linear
  combinations of pairwise overlaps (LCPO),” *J. Comput. Chem.* (1999),
  DOI `10.1002/(SICI)1096-987X(19990130)20:2<217::AID-JCC4>3.0.CO;2-A`.

## Atomic charge providers

- AM1-BCC is invoked through AmberTools Antechamber `-c bcc`; see Jakalian et
  al., “Fast, efficient generation of high-quality atomic charges. AM1-BCC
  model: II. Parameterization and validation,” *J. Comput. Chem.* (2002),
  DOI `10.1002/jcc.10128`.
- ABCG2 is invoked through AmberTools24+ Antechamber `-c abcg2`; see “ABCG2: A
  Milestone Charge Model for Accurate Solvation Free Energy Calculation,”
  *J. Chem. Theory Comput.* (2025), DOI `10.1021/acs.jctc.5c00038`.
- Antechamber's finite-precision text output can have a small residual from the
  declared integer charge.  MAPLE follows the published FESetup procedure and
  distributes only residuals no larger than `0.01 e` uniformly over all atoms,
  preserving equivalence while recording the provider values, correction, and
  used values in a normalization audit.  Larger residuals fail closed.  See
  Loeffler et al., “FESetup: Automating Setup for Alchemical Free Energy
  Simulations,” *J. Chem. Inf. Model.* (2015), DOI
  `10.1021/acs.jcim.5b00368`.
- RESP and RESP2 are not regenerated by MAPLE route 1; their charges are read
  from MOL2 without modification. RESP: “A well-behaved electrostatic potential
  based method using charge restraints for deriving atomic charges: the RESP
  model,” DOI `10.1021/j100142a004`; RESP2: “Non-bonded force field model with
  advanced restrained electrostatic potential charges (RESP2),” DOI
  `10.1038/s42004-020-0291-4`.

## LPB

The APBS adapter follows the official solvation-energy construction: one LPBE
calculation at solvent dielectric 78.5, one reference calculation at dielectric
1.0, followed by `print elecEnergy solv - ref end`.  The APBS APOLAR block
provides the required nonpolar term.  The locked generic profile reduces the
APBS nonpolar equation

\[
G_{np}=\gamma A+pV+\bar\rho\sum_i\int u_i^{att}\theta\,dy
\]

to `gamma*A` by setting `p=0` and `bar(rho)=0`; `gamma=0.105 kJ mol^-1 A^-2`
is the documented APBS/iAPBS default used by the profile.

The canonical provider control is the official radius-3-A, charge-+1 Born ion.
For zero ionic strength its analytic polar transfer energy is

\[
\Delta_pG_{\mathrm{Born}}=
\frac{q^2}{8\pi\epsilon_0a}
\left(\frac{1}{\epsilon_{out}}-\frac{1}{\epsilon_{in}}\right).
\]

The official APBS settings (`97^3`, `0.33 A`, `pdie=1`, `sdie=78.54`, LPBE,
`srfm=mol`, `srad=1.4`, `chgm=spl2`) report `-229.59 kJ/mol`; the analytic
formula gives `-230.62 kJ/mol`.  MAPLE stores the PQR, input, stdout, stderr,
binary hash, and parsed value.  Neutral methanol and aniline controls additionally
measure grid sensitivity from `0.50` through approximately `0.1667 A` at an
approximately fixed 32-A physical grid length.

- Baker et al., “Electrostatics of nanosystems: Application to microtubules and
  the ribosome,” *PNAS* (2001), DOI `10.1073/pnas.181342398`.
- Jurrus et al., “Improvements to the APBS biomolecular solvation software
  suite,” *Protein Science* (2018), DOI `10.1002/pro.3280`.
- Wagoner and Baker, “Assessing implicit models for nonpolar mean solvation
  forces: the importance of dispersion and volume terms,” *PNAS* (2006),
  DOI `10.1073/pnas.0600118103`.

The ABCG2-specific PBSA profile is not approximated.  Its defining reference is
Sun et al., “Development and test of highly accurate endpoint free energy
methods. 1: Evaluation of ABCG2 charge model on solvation free energy prediction
and optimization of atom radii suitable for more accurate solvation free energy
prediction by the PBSA method,” *J. Comput. Chem.* (2023), DOI
`10.1002/jcc.27089`; execution stays blocked until its exact optimized radii and
nonpolar parameter files are obtained from an authoritative, redistributable
source and pass human review.

## Upstream source snapshots reviewed

- Caltech `cheq` 0.5.1 source and parameter table at commit
  `f71eeb4ab3790557fb244a8f09fa60e255ee6274`, especially
  `src/shielding/gto.rs`, `src/solver/implementation.rs`, and
  `resources/qeq.data.toml`.
- Open Babel QEq source and data at commit
  `0e94434fa75c9f61095023e3c12e0d5f2ac035ff`, used to identify and reject the
  fixed-hydrogen shortcut rather than as MAPLE's current numerical profile.
- APBS input/output and APOLAR execution paths at commit
  `4613d0d547c3c71df8815dcb85e9e19abf61822c`, especially
  `examples/solv`, `examples/born`, and `src/routines.c`.
- OpenMM 8.5 `openmm.app.internal.customgbforces` implementations and their
  `getStandardParameters(topology)` assignments for all five GB models.
