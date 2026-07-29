# Route-2 V0-FD: fixed-density variational-solvent research contract

## Status and scope

This document defines the first separately preregistered no-training successor
that is admissible after the frozen field-space scalar V0 falsifier.  It is
**not** a public profile,
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
The separately preregistered structured-solvent admission boundary is
[`route2-v0-structured-solvent-admission-v1.json`](benchmarks/route2-v0-structured-solvent-admission-v1.json).

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

#### 4.2.1 V0-FD-S: admissible fixed-density structured-solvent branch

The first admissible no-training branch is named **V0-FD-S**.  It freezes the
same MACE-POLAR zero-field source \(c_0\) as V0-FD, but replaces the
macroscopic cavity/PCM approximation with a molecular liquid functional.  It
does **not** update \(c_0\), introduce a learned electronic response, or add
the SMD CDS term.

The electrostatic solute potential must be generated directly from the
unmodified MACE Gaussian multipoles over the whole solvent grid,

\[
\phi^G_{c_0}(\mathbf r)
=\sum_a\left[q_a g_{\sigma_a}(\mathbf r-\mathbf R_a)
+\boldsymbol\mu_a\!\cdot\!\nabla
g_{\sigma_a}(\mathbf r-\mathbf R_a)\right],
\]

using the declared all-space Gaussian convention in
`gaussian_multipole_potential`.  The current point-multipole exterior field
is not an interchangeable source for this purpose.  A distinct, positive,
independently sourced promolecular reference density

\[
n_{\mathrm{ref}}(\mathbf r;\mathbf R)
=\sum_a n_a^{\mathrm{ref}}(|\mathbf r-\mathbf R_a|)\geq0
\]

may define short-range exclusion/repulsion, but it must never be relabelled
as a MACE electron density or used to replace the MACE electrostatic source.
For solvent site \(\alpha\), the frozen solute--solvent interaction is then

\[
u_{\alpha s}(\mathbf r)
=z_{\alpha s}\phi^G_{c_0}(\mathbf r)
+u^{\mathrm{sr}}_{\alpha s}[n_{\mathrm{ref}}](\mathbf r).
\]

For example, a 3D-RISM/MDFT implementation may combine the bulk solvent
susceptibility \(\chi_{\alpha\gamma,s}\) with this potential through

\[
h_{\gamma s}=\sum_\alpha c_{\alpha s}*\chi_{\alpha\gamma,s},
\qquad
d_{\alpha s}=-\beta u_{\alpha s}+h_{\alpha s}-c_{\alpha s},
\]

and the Kovalenko--Hirata closure

\[
g_{\alpha s}=
\begin{cases}
\exp(d_{\alpha s}),&d_{\alpha s}\leq0,\\
1+d_{\alpha s},&d_{\alpha s}>0.
\end{cases}
\]

The corresponding closed KH excess-free-energy convention is

\[
\beta\mu_{\mathrm{KH}}^{\mathrm{ex}}
=\sum_\alpha\rho_{\alpha s}\!\int\!d\mathbf r
\left[
\tfrac12h_{\alpha s}^2\Theta(-h_{\alpha s})
-c_{\alpha s}-\tfrac12h_{\alpha s}c_{\alpha s}
\right].
\]

If the frozen bulk asset supplies a thermodynamic pressure and partial molar
volume from the *same* liquid functional, the primary correction convention
is the corresponding thermodynamic pressure correction,

\[
G_{\mathrm{V0-FD-S}}
=E_{\mathrm{MACE,gas}}+\mu_{\mathrm{KH}}^{\mathrm{ex}}
-P_s\bar V+\Delta G^\circ.
\]

The sign, standard state, and definitions of \(P_s\) and \(\bar V\) must be
versioned with the solvent asset before a score is read.  PC+ and any other
empirical extension may be retained only as a labelled diagnostic; no
per-record selection between PC, PC+, closures, or liquid models is allowed.

#### 4.2.2 Source equivalence is a hard boundary

Stock AmberTools `rism3d.snglpnt` consumes a PDB, an AMBER `prmtop`, and an
XVV bulk-solvent file.  A GAFF/AM1-BCC `prmtop` therefore changes the solute
Hamiltonian and cannot be silently used as a Route-2 V0-FD-S endpoint.  The
existing Route-1 all-site 3D-RISM pilot is valuable as a separately scoped
control, but is not source-equivalent to the frozen MACE Gaussian source.

V0-FD-S needs a MACE-native grid-potential adapter and an independently
provenanced positive short-range source before a structured-solvent
calculation may be run.  Reusing an Amber bulk susceptibility is potentially
admissible only when its solvent provenance is frozen and its interface acts
on the MACE-native potential; the AMBER solute charges/Lennard-Jones terms
must not enter by fallback.  This is deliberately a fail-closed precondition,
not a request to tune a GAFF mapping.

The molecular functional already contains molecular exclusion and dispersion
physics through \(u^{\mathrm{sr}}\) and its liquid free energy.  Adding
SMD-CDS to it would double count those effects and would reintroduce an
experiment-parameterized comparator into the no-fit core.

#### 4.2.3 Implemented MACE-native electrostatic grid primitive

`route2_v0_structured_solvent.py` now implements the first, deliberately
narrow V0-FD-S primitive.  For a regular Bohr grid

\[
\mathbf r_{ijk}=\mathbf r_0+(i\Delta x,j\Delta y,k\Delta z),
\]

it evaluates and preserves the exact MACE Gaussian source,

\[
\Phi_{ijk}=\phi^G_{c_0}(\mathbf r_{ijk}),
\qquad
u_{\alpha s}^{\mathrm{el}}(\mathbf r_{ijk})
=z_{\alpha s}\Phi_{ijk}.
\]

The source is checked for the declared total charge, stores immutable values
in an explicit \((x,y,z)\) C-order layout, remains finite at atomic centres,
and has translation-covariance tests.  The grid width is fixed at the
checkpoint-native MACE Gaussian width; changing it is a different source
model, not a numerical grid-refinement parameter.

This primitive does **not** expose \(u^{\mathrm{sr}}\), a liquid-state
minimizer, an excess chemical potential, a force, or an accuracy result.  It
therefore cannot yet be called a 3D-RISM/MDFT calculation.  That separation is
also consistent with an existing density-coupled 3D-RISM design: AMS documents
direct use of a fitted solute electron density for electrostatics while still
requiring declared solute--solvent Lennard-Jones inputs for the short-range
interaction.  Direct density electrostatics does not make short-range
repulsion/dispersion disappear.

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
8. S. M. Kast *et al.*, *Molecular Solvation in 3D-RISM: A New Method for
   Calculating Solvation Free Energies*, [PMC2861832](https://pmc.ncbi.nlm.nih.gov/articles/PMC2861832/).
   This provides the KH closure and closed excess-chemical-potential context;
   it does not make a GAFF/AM1-BCC solute source equivalent to MACE-POLAR.
9. D. S. Palmer *et al.*, *The Amber molecular dynamics package*,
   [PMC10598796](https://pmc.ncbi.nlm.nih.gov/articles/PMC10598796/).  Its
   3D-RISM workflow illustrates the distinct AMBER topology and bulk-solvent
   asset contract that V0-FD-S must not substitute for the MACE source.
10. SCM, *3D-RISM: 3D Reference Interaction Site Model*,
    [ADF 2026.1 documentation](https://www.scm.com/doc/ADF/Input/3D-RISM.html).
    It documents density-based electrostatic coupling together with separate
    short-range Lennard-Jones inputs, motivating the explicit source split
    retained here.
