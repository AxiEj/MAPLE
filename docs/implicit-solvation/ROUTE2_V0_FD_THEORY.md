# Route-2 V0-FD: fixed-density variational-solvent research contract

## Status and scope

This document defines the only no-training successor that is admissible after
the frozen field-space scalar V0 falsifier.  It is **not** a public profile,
chemical-accuracy result, force/PES implementation, or replacement for the
legacy Route-2 fixed point.

The failed construction was a field-conditioned MACE scalar whose gradient was
used as a new density.  It passed its local finite-difference implementation
checks but failed the nonzero-field charge, reciprocity, and passivity gates.
Those failures are evidence about the frozen checkpoint, not numerical SCF
tolerance.  The associated artifact is
[`route2-v0-scalar-response-water-v1.json`](benchmarks/route2-v0-scalar-response-water-v1.json).

V0-FD instead freezes the MACE-POLAR state at zero reaction field and varies
only the solvent state.  It preserves the MACE gas branch and requires no
post-training, fine-tuning, MAP/UQ calibration, density mixing, response
tempering, or experimental-solvation fitting.

The machine-readable commitment is
[`route2-v0-fd-multisolvent-prereg-v1.json`](benchmarks/route2-v0-fd-multisolvent-prereg-v1.json).

## 1. Fixed-density variational construction

At nuclear geometry \(\mathbf R\), let

\[
c_0(\mathbf R)=c_{\mathrm{MACE}}(\mathbf R,0),
\qquad
u^\mathsf Tc_0=Q,
\]

be the unmodified, zero-field MACE-POLAR \(l\leq1\) density coefficients.
They are a frozen solute source, not a variational electronic density and not
an object to be projected after inference.  A runtime charge violation is a
fail-closed error.

For one declared source basis, let \(B_{\mathbf R}\) map \(c_0\) to a
continuum boundary potential and let \(A_{\mathbf R}\) be the symmetric
operator defining the continuum energy.  A Galerkin variational backend
exposes this matrix directly.  For a discrete IEFPCM backend with distinct
direct and transpose solves, it is represented only through its certified
energy-conjugate response \(q_{\mathrm{energy}}\), never by pretending that
the direct operator itself is symmetric.  The solvent functional is

\[
\mathcal L_{\mathrm{solv}}(\mathbf R,c_0,\sigma)
=
\frac12\sigma^\mathsf TA_{\mathbf R}\sigma
+\sigma^\mathsf TB_{\mathbf R}c_0.
\]

Its stationary solvent state satisfies

\[
A_{\mathbf R}\sigma+B_{\mathbf R}c_0=0.
\]

Eliminating \(\sigma\) gives

\[
P_{\mathbf R}=-B_{\mathbf R}^\mathsf TA_{\mathbf R}^{-1}B_{\mathbf R},
\qquad
G_{\mathrm{el}}(\mathbf R,c_0)
=\frac12c_0^\mathsf TP_{\mathbf R}c_0.
\]

Equivalently, with \(v=Bc_0\) and the energy-conjugate continuum charge
\(q_{\mathrm{energy}}\),

\[
G_{\mathrm{el}}
=\frac12v^\mathsf Tq_{\mathrm{energy}}(v).
\]

This is the sole scalar evaluated by
`route2_v0_frozen_density.py`.  Its contract requires an energy-conjugate
continuum response, retains direct/adjoint/energy charges for audit, and does
not accept a field-conditioned MACE state.

With the same point-\(l\leq1\) source, cavity, and PCM operator, this
electrostatic term is intentionally identical to the already-existing legacy
`response=frozen` electrostatic branch.  V0-FD does **not** claim a new
accuracy gain merely by renaming that scalar.  Its purpose is to make the
fixed-density variational boundary explicit, so that any later cavity,
nonpolar, or structured-solvent contribution can be evaluated against one
auditable energy rather than against the rejected learned fixed point.

The eventual total fixed-density V0 free energy is

\[
G_{\mathrm{V0-FD}}
=E_{\mathrm{MACE,gas}}(\mathbf R)
+G_{\mathrm{el}}(\mathbf R,c_0)
+G_{\mathrm{np}}(\mathbf R; s)
+\Delta G^{\circ}(T,p).
\]

Here \(G_{\mathrm{np}}\) is deliberately unresolved rather than silently
identified with the existing fitted SMD CDS term.  The current SMD CDS is a
useful explicit empirical comparator, but not evidence for a no-fit core.

### Consequences

1. The continuum response is reciprocal because it is derived from one scalar
   quadratic form.  No MACE field-response Jacobian is used.
2. The construction contains solvent polarization but **not** solute induced
   polarization.  It must be reported as frozen-density solvation, never as a
   self-consistent electronic state.
3. The legacy ML--PCM fixed point remains available only as a nonvariational
   diagnostic.  It is not altered, retuned, or used to repair V0-FD.

## 2. Coordinate derivatives and smoothness

At a stationary \(\sigma^*\), the envelope theorem removes
\(d\sigma^*/d\mathbf R\), but it does not remove the coordinate dependence of
the frozen MACE source:

\[
\frac{dG_{\mathrm{el}}}{d\mathbf R}
=
\frac12\sigma^{*\mathsf T}
\frac{\partial A_{\mathbf R}}{\partial\mathbf R}\sigma^*
+\sigma^{*\mathsf T}
\frac{\partial B_{\mathbf R}}{\partial\mathbf R}c_0
+\sigma^{*\mathsf T}B_{\mathbf R}
\frac{dc_0}{d\mathbf R}.
\]

Therefore a conservative V0-FD force requires all three terms above plus
\(\partial_{\mathbf R}G_{\mathrm{np}}\) and the gas MACE force.  It is not
permitted to reuse the legacy fixed-point adjoint or to project net force and
torque after the fact.

Existing pyddx and finite Lebedev SWIG/ISWIG experiments are retained as
energy/derivative controls, not as a smooth V0-FD PES provider: their active
sets or orientation gates failed.  A future force backend must expose a
single smooth cavity/operator scalar before an OPT, scan, MD, or NVE claim.

## 3. Why dielectric constant alone cannot define a custom total solvent

For a homogeneous liquid, the static dielectric constrains only the
long-wavelength longitudinal susceptibility,

\[
\epsilon(0)=1+4\pi\chi_L(k\!\to\!0).
\]

The solvation free energy depends on the finite-wavevector response,
excluded-volume work, dispersion, and (for associating liquids) orientational
and hydrogen-bond structure.  Hence two liquids can have the same
\(\epsilon(0)\) but distinct \(\chi(k)\), number density, surface tension,
polarizability, and \(\Delta G_{\mathrm{solv}}\).  A dielectric-only custom
input is mathematically underdetermined for a total free energy.

V0-FD therefore distinguishes two contracts:

| Input | Allowed result | Forbidden claim |
|---|---|---|
| \(\epsilon(0)\) and geometry | electrostatics-only diagnostic | total solvation free energy |
| \(T,p,\epsilon(0),\epsilon_\infty,\rho\), independently sourced cavity and dispersion data | declared continuum total-free-energy candidate | accuracy certification before blind tests |
| molecular site model plus bulk susceptibility/direct correlations | MDFT/RISM structured-solvent candidate | equivalence to dielectric-only PCM |

This is stricter than accepting a Gaussian-like solvent name with only a
dielectric.  It prevents hidden empirical parameters from entering through a
custom-solvent shortcut.

## 4. Nonpolar and structured-solvent candidates

### 4.1 Density-derived cavity plus weighted-density nonpolar free energy

Sundararaman, Gunceler, and Arias derived a weighted-density cavity functional
and pair-potential dispersion treatment intended to replace atom-surface
tensions.  It is the closest mathematical candidate for a V0-FD nonpolar
term because it separates cavity formation, dispersion, and dielectric
response.  It is **not yet adopted** here for two reasons:

1. MACE-POLAR's current \(l\leq1\) coefficients are signed residual
   multipoles, not a positive all-electron density.  They cannot be used as an
   isodensity cavity without inventing electron density.
2. The published approximation retains a solvent-dependent dispersion scale.
   V0-FD may use it only if that quantity is fixed from independently sourced
   solvent physics before any target solvation calculation, never fitted to
   MNSol/FreeSolv.

A future implementation may use a separately validated promolecular reference
density for the cavity while retaining MACE \(c_0\) only for electrostatics.
That distinction must remain explicit in the source provenance.

### 4.2 Molecular density functional theory / 3D-RISM

The persistent polar-protic and nonpolar/halogen outliers point to missing
short-range solvent structure, not merely a different linear PCM equation.
MDFT or 3D-RISM can define a solvent free-energy functional over solvent site
densities,

\[
\Omega[\{n_\alpha\};c_0]
=F_{\mathrm{liq}}[\{n_\alpha\}]
+\sum_\alpha\int n_\alpha(\mathbf r)
u_\alpha(\mathbf r;c_0)\,d\mathbf r,
\]

and obtain the solvation free energy by minimization.  This retains molecular
solvent structure and can support a custom solvent if its site model, bulk
correlations, temperature, pressure, and provenance are supplied.  It is an
independent continuum-backend candidate, not an equation switch, and must be
pre-registered before chemistry evaluation.

### 4.3 Explicit exclusions

- Selecting IEFPCM/CPCM/COSMO per record is forbidden.  The existing same-
  source ten-solvent panel found only small equation differences and does not
  establish a chemical-accuracy winner.
- COSMO-RS/openCOSMO-RS remains an external comparator: its current parameter
  family was fitted to solvation data and cannot certify a no-fit V0-FD core.
- Post-hoc charge projection, response symmetrization, eigenvalue clipping,
  response scaling, cavity-radius tuning, and error regressions are not
  physical candidate families.

## 5. Accuracy evidence and falsification gates

The present small panels already forbid a success claim:

- the ten-record MNSol fixed \(l\leq1\) ablation has maximum absolute error
  \(1.6454\ \mathrm{kcal\ mol^{-1}}\);
- the same-profile SCF member reaches \(3.1905\ \mathrm{kcal\ mol^{-1}}\);
- the distinct FreeSolv exact-GTO SCF development panel reaches
  \(3.2473\ \mathrm{kcal\ mol^{-1}}\).

These are different benchmark identities and are not ranked against one
another.  They do establish that a panel MAE below \(1\) does not meet the
requested all-record maximum-error condition.

The required progression is therefore:

1. pass structural energy, charge, pairing, and smoothness gates;
2. freeze a replicated **11-solvent** development design before looking at
   V0-FD scores.  The current MNSol-v2012 neutral-absolute partition has only
   ten solvent strata because it has no neutral absolute methanol rows in the
   frozen Route-2 domain; a separately versioned methanol extension is a
   prerequisite, not a gap to hide behind aggregate metrics;
3. require every development record to satisfy
   \(|\Delta G_{\mathrm{calc}}-\Delta G_{\mathrm{exp}}|<1.5\) kcal/mol;
4. freeze a disjoint multi-solvent confirmation set and impose the same
   per-record condition without changing the method;
5. freeze an external final blind dataset absent from construction,
   development, and confirmation; require the same condition there;
6. treat all-record \(<1.0\) kcal/mol as the stricter final target.

No aggregate MAE, a lucky subset, or an experiment-selected model choice can
substitute for these gates.

## 6. Literature boundary

1. I. Batatia *et al.*, *MACE-POLAR-1: A Polarisable Electrostatic Foundation
   Model for Molecular Chemistry* (2026),
   [arXiv:2602.19411](https://arxiv.org/abs/2602.19411).  The current model is
   described using a non-self-consistent field formalism; that does not supply
   a variational electronic functional for the present coupling.
2. W. J. Baldwin *et al.*, *Design Space of Self-Consistent Electrostatic
   Machine Learning Interatomic Potentials* (2026),
   [arXiv:2603.14700](https://arxiv.org/abs/2603.14700).  It distinguishes
   energy-functional and fixed-point electrostatic MLIP constructions.
3. S. Petrosyan, J.-F. Briere, D. Roundy, and T. A. Arias, *Joint
   density-functional theory for electronic structure of solvated systems*,
   [arXiv:cond-mat/0606817](https://arxiv.org/abs/cond-mat/0606817).  This is
   the variational reference model; it requires an electronic density
   functional not present in the frozen checkpoint.
4. R. Sundararaman, D. Gunceler, and T. A. Arias, *Weighted-density
   functionals for cavity formation and dispersion energies in continuum
   solvation models*, *J. Chem. Phys.* **141**, 134105 (2014),
   [DOI:10.1063/1.4896827](https://doi.org/10.1063/1.4896827).
5. A. W. Lange and J. M. Herbert, *Polarizable Continuum Reaction-Field
   Solvation Models Affording Smooth Potential Energy Surfaces*, *J. Phys.
   Chem. Lett.* **1**, 556--561 (2010),
   [DOI:10.1021/jz900282c](https://doi.org/10.1021/jz900282c).
6. E. L. Ratkova, D. S. Palmer, and M. V. Fedorov, *Solvation Thermodynamics
   of Organic Molecules by the Molecular Integral Equation Theory: Approaching
   Chemical Accuracy*, *Chem. Rev.* **115**,
   6312--6356 (2015),
   [DOI:10.1021/cr5000283](https://doi.org/10.1021/cr5000283).
7. A. V. Marenich *et al.*, *Minnesota Solvation Database -- version 2012*,
   [DOI:10.13020/3eks-j059](https://doi.org/10.13020/3eks-j059).  MNSol
   supplies experimental coverage across many solvents, but may never be used
   to fit the V0-FD functional.
