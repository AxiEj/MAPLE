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

### 3.1 What the pure-liquid scalar state fixes -- and what it cannot fix

The restriction is not merely a provenance preference.  In the scalar
one-component reduction, the compressibility sum rule gives

\[
S_{NN}(0)=\rho_m k_{\rm B}T\kappa_T
=\frac{1}{1-\rho_m\widehat c_{NN}(0)},
\qquad
\widehat c_{NN}(0)
=\frac{1-S_{NN}(0)^{-1}}{\rho_m}.
\]

Thus \(T,\rho_m,\kappa_T\) determine **one projected zero mode** of the
number-number direct correlation.  The static dielectric similarly determines
only a long-wavelength polarization projection,
\(\epsilon_s=1+4\pi\chi_L(k\!\to\!0)\), under a declared macroscopic
boundary convention.  The optical dielectric adds another scalar limit.  A
surface tension is one planar-interface integral constraint.  None of these
quantities supplies the site/orientation-resolved finite-wavevector kernel
\(C_{ab}(\mathbf k)\).

The non-uniqueness has a direct constructive form.  If
\(C_{ab}^{(0)}(\mathbf k)\) is a stable reciprocal kernel, then for any real
symmetric site matrix \(A_{ab}\), length \(\ell>0\), and sufficiently small
\(t\),

\[
\delta C_{ab}(\mathbf k)
=t A_{ab} k^2 e^{-(k\ell)^2},
\qquad
C_{ab}^{(t)}=C_{ab}^{(0)}+\delta C_{ab},
\]

is real, even, reciprocal, and has \(\delta C_{ab}(0)=0\).  It leaves the
above zero-mode constraints unchanged while changing the finite-\(k\)
quadratic liquid term sampled by any cavity or solute perturbation with
nonzero Fourier support.  Stability persists for small enough \(t\) by
continuity.  Therefore no finite list of macroscopic scalars can identify the
molecular liquid functional needed for a total solvation free energy.

To make this boundary executable,
`route2_v0_bulk_liquid_state_source.py` accepts a **source-only** pure-liquid
state certificate only when all of the following are independently
content-addressed and bound to the exact molecular-model digest:

- \(T,p,\rho_m,\epsilon_s,\epsilon_\infty,\kappa_T,\gamma\);
- one source locator and origin for every individual property; and
- explicit false no-training, no-fine-tuning, no-solvation-fit, no-MAP/UQ, and
  no-target-label flags.

It exposes the scalar compressibility zero mode only and can cross-check the
three quantities a 1D-RISM input actually contains
(\(T,\rho_m,\epsilon_s\)).  It deliberately has no conversion from these
scalars to \(C_{ab}(k)\), no closure selection, no short-range potential, and
no `total_free_energy` operation.  The machine-readable contract is
[`route2-v0-bulk-liquid-state-prereg-v1.json`](benchmarks/route2-v0-bulk-liquid-state-prereg-v1.json).
It begins with **zero** physical state records: this is a guard against
inventing density/dielectric values for the checked-in chloroform and
dichloromethane molecular models, not an asset-admission claim.

Molecular DFT explicitly requires pure-solvent direct correlations (or an
alternative declared liquid functional), and JDFTx likewise distinguishes a
classical-DFT liquid functional from linear/nonlinear dielectric modes.
Accordingly, a complete finite-\(k\) susceptibility can at most enter the
separately labelled electrostatic control until it is joined to one stationary
molecular liquid scalar with the short-range, bridge, and standard-state
terms.

### 3.2 State-domain separation: fixed-charge molecular RISM versus a nonlocal dielectric control

The full source-only record above retains the optical dielectric
\(\epsilon_\infty\) because the separately labelled nonlocal-dielectric
continuum control represents a macroscopic electronic-response limit.  That
is **not** a requirement that can be imported into a fixed-charge molecular
RISM model.  A fixed-charge site model has no independent solvent-electronic
polarization degree of freedom: the bulk `rism1d` state it can cross-check is
only

\[
(T,\rho_m,\epsilon_s),
\]

alongside independently recorded \(p,\kappa_T,\gamma\).  In particular, adding
\(\epsilon_\infty\) to an Amber-style fixed-charge RISM input would not create
a missing electronic mode, would not identify a closure, and would not turn
scalar anchors into \(C_{ab}(k)\).

`route2_v0_molecular_rism_state_source.py` therefore enforces a distinct,
strict six-field source contract

\[
(T,p,\rho_m,\epsilon_s,\kappa_T,\gamma),
\]

with no optical-dielectric field.  It may test the same number-channel zero
mode

\[
S_{NN}(0)=\rho_m k_{\rm B}T\kappa_T,
\]

and the `rism1d` \(T,\rho_m,\epsilon_s\) serialization only.  It cannot load,
derive, or infer a finite-\(k\) correlation, orientational response,
solute--solvent short-range interaction, bridge, stationary molecular scalar,
or total free energy.  The initial
[`route2-v0-molecular-rism-state-prereg-v1.json`](benchmarks/route2-v0-molecular-rism-state-prereg-v1.json)
contains one **source-only** dichloromethane record bound to its exact
SCM-table-derived AMBER MDL.  It is not a converged AmberTools asset and does
not change the zero-admitted-asset or zero-accuracy-result status.

This separation prevents two opposite category errors: treating a continuum
optical limit as a fixed-charge RISM input, or treating a RISM static
dielectric as a complete continuum spectrum.  Both branches remain
fail-closed until their own finite-wavevector/functional requirements are met.

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

#### 4.2.4 Independent promolecular reference-density pre-registration

The next required input is not an absolute value, clipping, or reinterpretation
of the signed MACE coefficients.  It is a separately generated positive
promolecular reference density,

\[
n_{\mathrm{ref}}(\mathbf r;\mathbf R)
=\sum_a n^{\mathrm{atom}}_{Z_a}(|\mathbf r-\mathbf R_a|),
\qquad n^{\mathrm{atom}}_{Z_a}\geq0,
\]

which may later enter only a separately admitted short-range coupling
functional.  In particular, \(n_{\mathrm{ref}}\) never replaces
\(\phi^G_{c_0}\) in the electrostatic channel.

[`route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json`](benchmarks/route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json)
freezes a spherically averaged isolated-atom HF/def2-TZVPD construction for
H, C, N, O, S, and Cl: the elements required by the eleven registered default
solvents.  Isolated-atom superpositions are the conventional definition of a
promolecular density, but they are a reference input rather than a claim that
the molecular density is spherical or that its short-range interaction has
already been determined.  The generated
[`route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json`](benchmarks/route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json)
records a passing source-bound table: each of the six neutral atom densities
is nonnegative, spherically averaged, radially normalized to its electron
count within the preregistered tolerance, and negligible at the frozen radial
boundary.  `route2_v0_promolecular_density.py` verifies the table and per-
element hashes before it evaluates the superposition, and fails closed for
unsupported elements.

The resulting table is still not a density-overlap free-energy model.  A
hand-chosen overlap coefficient would be a new empirical potential, and is
not admitted.  The next physical decision is the coupling functional itself,
not a numerical rescaling of this reference density.

Section 4.2.11 supplies one deliberately generic, parameter-free
frozen-density kinetic-overlap **control** for that decision.  It consumes two
explicit electron-reference-density grids; it does not silently couple the
current solute-only promolecular table to an invented solvent density or
declare a physical \(u^{\mathrm{sr}}\).

#### 4.2.5 Synthetic variational site-HNC reference kernel

Before a real molecular-solvent asset is admitted, the code now contains a
strictly **synthetic-only** multi-site HNC reference functional.  It exists to
lock the discrete scalar/gradient relation that the eventual liquid backend
must preserve; it is not a physical solvent model or a solvation endpoint.
For periodic site densities \(\rho_a(\mathbf r)\), bulk number densities
\(\rho_a^b\), external site potentials \(u_a\), and a frozen direct
correlation \(c_{ab}\), its declared grand-potential difference is

\[
\begin{aligned}
\Omega_{\mathrm{HNC}}[\rho;u]
={}&k_{\mathrm B}T\sum_a\int
\left[\rho_a\ln\frac{\rho_a}{\rho_a^b}
-\left(\rho_a-\rho_a^b\right)\right]d\mathbf r\\
&-\frac{k_{\mathrm B}T}{2}\sum_{ab}\iint
\delta\rho_a(\mathbf r)c_{ab}(\mathbf r-\mathbf r')
\delta\rho_b(\mathbf r')d\mathbf r\,d\mathbf r'
+\sum_a\int\rho_a(\mathbf r)u_a(\mathbf r)d\mathbf r,
\end{aligned}
\]

where \(\delta\rho_a=\rho_a-\rho_a^b\).  The frozen asset must obey the
reciprocal discrete pairing condition

\[
c_{ab}(\mathbf r)=c_{ba}(-\mathbf r).
\]

That condition makes the quadratic term a scalar rather than merely an
iteration map.  The Euler equation is then the derivative of exactly the same
stored scalar,

\[
\frac{1}{k_{\mathrm B}T}
\frac{\delta\Omega_{\mathrm{HNC}}}{\delta\rho_a}
=\ln\frac{\rho_a}{\rho_a^b}
-\sum_b c_{ab}*\delta\rho_b
+\beta u_a=0.
\]

`route2_v0_site_hnc.py` evaluates this expression with the declared Cartesian
cell volume in both the scalar and its FFT convolution.  It rejects a
nonreciprocal direct-correlation array, keeps the physical residual separate
from optional Picard mixing, and tests the finite-difference derivative, the
reciprocal convolution pairing, and the ideal-gas stationary solution.  The
MACE Gaussian grid potential can enter only as the electrostatic summand of
\(u_a\); no charge/Lennard-Jones substitute is introduced.

This is the mathematical form used by HNC molecular density functional theory:
the standard functional is minimized with respect to solvent density and its
excess term is a direct-correlation quadratic form.  The present site-only
kernel deliberately omits orientational degrees of freedom, bridge terms, a
short-range \(u^{\mathrm{sr}}\), a physical \(c_{ab}\) asset, and a
thermodynamic pressure convention.  In particular, HNC pressure can be badly
overestimated, so this reference does **not** choose HNC as the primary
free-energy closure or authorize a chemistry score.  It is a falsifiable
energy/derivative control while the frozen KH/MDFT physical asset contract is
being built.

#### 4.2.6 Bulk RISM asset and Coulomb-tail boundary

A physical RISM asset must be generated by a **bulk-only** 1D-RISM solve and
must contain both its `.xvv` susceptibility metadata and its radial `.cvv`
direct-correlation table.  These are solvent-state assets: they are independent
of the solute and may be reused only at their frozen temperature, density,
dielectric convention, site model, closure, and solver version.  An AMBER
`prmtop`, AM1-BCC charge, or GAFF Lennard-Jones term for the *solute* is not an
input to this asset path.  The `.xvv` site multiplicities and `QV` charges
must also be multiplicity-weighted neutral; a charged or partially specified
bulk asset is rejected rather than neutralized numerically.

The checked-in dichloromethane state record now also has one negative,
source-frozen numerical feasibility audit:
[`route2-v0-dcm-kh-source-frozen-feasibility-audit-v1.json`](benchmarks/route2-v0-dcm-kh-source-frozen-feasibility-audit-v1.json).
For the exact SCM all-atom model, source-bound scalar state, and predeclared
AmberTools `DRISM`/`KH` input, `rism1d` entered a printed residual three-cycle
after step 27 and never emitted `.xvv` or `.cvv`.  The audit therefore rejects
that exact nonconverged numerical candidate as a finite-\(k\) asset.  It is
not a verdict on the physical validity of the SCM model, a closure comparison,
or permission to search mixing/closure settings from solvation errors.  Any
later generator candidate must be source-complete and preregistered before its
execution; no target-solvation quantity may choose it.

The direct correlation cannot be naively interpolated from a finite radial
table onto a periodic Cartesian grid.  In the native `QV` plus source-defined
`SMEAR` convention stored in an Amber `.xvv` file, the raw site direct
correlation has the analytic smooth decomposition

\[
c^{\mathrm{raw}}_{ab}(r)
=c^{\mathrm{sr}}_{ab}(r)
-q^{\mathrm{QV}}_a q^{\mathrm{QV}}_b
\frac{\operatorname{erf}(r/\eta)}{r},
\qquad r\ge0.
\]

Here \(q^{\mathrm{QV}}\) and \(r\) use the source file's
\(\sqrt{k_{\mathrm B}T\,\mathrm{\AA}}\) and Angstrom conventions, and
\(\eta\) is Amber's positive `SMEAR` parameter, so the right-hand side is
dimensionless.  Its source-defined origin is finite,

\[
\lim_{r\to0}\frac{\operatorname{erf}(r/\eta)}r
=\frac2{\sqrt\pi\eta}.
\]

At large distance it approaches the usual bare
\(-q^{\mathrm{QV}}_a q^{\mathrm{QV}}_b/r\) asymptote, which remains an
independent bulk-control diagnostic.  This is not a fit: it is the exact
Coulomb asymptotic convention of the bulk RISM model with its declared
short/long-range split.  Substitution into the HNC quadratic term gives the
required energy partition,

\[
\begin{aligned}
-\frac{k_{\mathrm B}T}{2}
\sum_{ab}\delta\rho_a*c^{\mathrm{raw}}_{ab}*\delta\rho_b
={}&-\frac{k_{\mathrm B}T}{2}
\sum_{ab}\delta\rho_a*c^{\mathrm{sr}}_{ab}*\delta\rho_b\\
&+\frac{k_{\mathrm B}T}{2}
\sum_{ab}\delta\rho_a*
q^{\mathrm{QV}}_a q^{\mathrm{QV}}_b
\frac{\operatorname{erf}(r/\eta)}{r}*\delta\rho_b.
\end{aligned}
\]

The second term requires one declared periodic Coulomb/Ewald or Poisson
operator that uses the **same** source `SMEAR`.  Simply wrapping the raw
radial table with an FFT changes that operator.  Conversely, the smooth split
does not require origin imputation: it uses the stored raw `r=0` value plus
the analytic \(2/(\sqrt\pi\eta)\) limit.  Therefore raw `.cvv` data is *not*
admitted to `Route2V0SiteHNCFunctional` yet; the finite short-range remainder
must pass the separate reciprocal Cartesian control and physical-grid
refinement proof before it can be coupled to the long-range scalar.

`route2_v0_rism_bulk.py` now parses the 1D-RISM `.xvv` metadata and `.cvv`
site-pair table, requires exact site-pair coverage and matching radial grid,
checks site reciprocity, and exposes the full radial \(c^{\mathrm{sr}}\)
remainder, including the analytically determined source origin, plus a
measurable Coulomb-tail residual.  It does not run 3D-RISM, accept a solute
topology, construct a Cartesian kernel,
or report a solvation energy.  A local cSPC/E bulk-only control confirms that
the installed 1D-RISM producer emits both files and follows this tail identity;
[`route2-v0-rism1d-cspce-bulk-control-v1.json`](benchmarks/route2-v0-rism1d-cspce-bulk-control-v1.json)
records its source hashes and structural checks.  It remains a water-only
parser control.  The separate schema-v2 source-complete water candidate in
Section 4.2.9 adds molecular/source binding, but neither object is one of the
required eleven **production-admitted physical** solvent assets.

#### 4.2.7 Neutral periodic Poisson long-range control

The analytic split now has one explicit, independent long-range operator.
`route2_v0_periodic_coulomb.py` interprets a Cartesian grid as one periodic
cell of side lengths \(L_i=N_i\Delta_i\) in Bohr.  It converts the radial
length factor and the source `SMEAR` in Amber's native QV convention only,

\[
\widetilde q_a=\frac{q^{\mathrm{QV}}_a}{\sqrt{a_0/\mathrm{\AA}}},
\qquad
\widetilde\eta=\frac{\eta}{a_0/\mathrm{\AA}},
\]

so that

\[
q^{\mathrm{QV}}_a q^{\mathrm{QV}}_b
\frac{\operatorname{erf}(r_{\mathrm{\AA}}/\eta)}{r_{\mathrm{\AA}}}
=\widetilde q_a\widetilde q_b
\frac{\operatorname{erf}(r_{a_0}/\widetilde\eta)}{r_{a_0}}.
\]

No elementary-charge reinterpretation or adjustable electrostatic prefactor
is introduced: the QV product is already the dimensionless
\(-\beta u^{\mathrm{lr}}\) tail convention of the RISM direct correlation.
For a site-density difference the operator forms

\[
\rho_Q(\mathbf r)=\sum_a\widetilde q_a\,\delta\rho_a(\mathbf r),
\qquad
\widehat V_Q(\mathbf k)=
\begin{cases}
4\pi e^{-\widetilde\eta^2|\mathbf k|^2/4}
\widehat\rho_Q(\mathbf k)/|\mathbf k|^2,&\mathbf k\ne0,\\
0,&\mathbf k=0.
\end{cases}
\]

The declared dimensionless scalar and its site derivative are therefore

\[
\mathcal E_{\mathrm{lr}}
=\frac12\int\rho_Q(\mathbf r)V_Q(\mathbf r)d\mathbf r,
\qquad
\frac{\delta\mathcal E_{\mathrm{lr}}}
{\delta\,\delta\rho_a(\mathbf r)}
=\widetilde q_aV_Q(\mathbf r).
\]

When this term is eventually joined to the HNC scalar, it contributes
\(+k_{\mathrm B}T\mathcal E_{\mathrm{lr}}\), exactly the second term of the
split in Section 4.2.6.  The code evaluates both expressions through the same
zero-average, source-smeared FFT Poisson inverse; it does not assemble an
energy from one kernel and a residual from another.  The unsmeared
\(\widetilde\eta=0\) path is retained only as a mathematical periodic-control
limit and is not the physical path for an Amber `.cvv` asset.

The omitted zero Fourier mode is a gauge only for a neutral field.  The
operator therefore rejects a non-neutral \(\rho_Q\) rather than silently
adding a uniform compensating background.  A charged-solute branch needs a
separately preregistered Ewald-background, finite-size, and standard-state
free-energy convention; it cannot inherit this neutral control by accident.
The current tests lock an exact reciprocal Fourier mode, zero-mean gauge,
reciprocal pairing, translation covariance, finite-difference scalar/gradient
agreement, and fixed-cell grid refinement.  They do **not** yet interpolate
an actual source-provenanced \(c^{\mathrm{sr}}\) asset onto a production
Cartesian grid, attach a physical liquid functional, or report a
solvent/accuracy result.

#### 4.2.8 Reciprocal short-range radial-transform control

The finite source-SMEAR remainder now has a second, separate mathematical
control in `route2_v0_rism_reciprocal.py`.  It does not wrap radial samples
onto a Cartesian minimum-image grid or interpolate a new origin.  Instead it
uses the source radial grid directly in the spherical transform

\[
\widetilde c^{\mathrm{sr}}_{ab}(k)
=4\pi\int_0^{R_{\max}}r^2
c^{\mathrm{sr}}_{ab}(r)\operatorname{sinc}(kr)\,dr.
\]

For the Cartesian FFT convention used by the HNC reference, the periodic grid
kernel \(c^P\) is normalized by

\[
\Delta V\,\operatorname{FFT}[c^P_{ab}](\mathbf k)
=\widetilde c^{\mathrm{sr}}_{ab}(|\mathbf k|).
\]

The same discrete object therefore appears in the quadratic scalar and in its
density derivative; radial site symmetry gives
\(c^P_{ab}(\mathbf r)=c^P_{ba}(-\mathbf r)\) by construction.  The control
uses source-grid trapezoidal quadrature, not a fit.  It rejects a source grid
without the analytic \(r=0\) value, a Cartesian reciprocal grid beyond the
source radial Nyquist limit, or a source-tail maximum above a caller-declared
and preregistered numerical tolerance.  That tolerance is a representation
gate fixed before any solvation result, never an error-selected solvent
parameter.

Synthetic Gaussian data verifies the transform normalization at every FFT
mode, the reciprocal real-space pairing, and the finite-difference derivative
after the generated kernel is passed through the HNC scalar.  This is still a
small-grid reference control: it does not register a physical solvent,
establish a production-grid convergence threshold, define
\(u^{\mathrm{sr}}\), run a molecular liquid solve, or make a thermodynamic or
accuracy claim.

#### 4.2.9 Content-addressed frozen solvent-asset boundary

The next physical input is a *solvent-side* asset, not a new solute charge
model and not a trainable correction.  Before such an asset exists,
`route2_v0_all_atom_solvent_model_source.py` may freeze a narrower
**source-only** all-atom molecular record: real-element mass identity,
explicit rigid Cartesian sites, charges, \(\sigma\), \(\epsilon\), a neutral
charge sum, source molecular-weight agreement, an HTTPS document hash/locator,
and a deterministic AMBER-MDL serialization.  This rejects united atoms,
virtual sites, target-label provenance, post-training, and fine-tuning.  It
does **not** select density, dielectric constant, closure, a 1D-RISM result,
or a liquid functional.  Thus the checked-in chloroform and dichloromethane
records are source evidence only, not assets in the physical 11-solvent panel.

`route2_v0_solvent_asset.py` accepts a registry only when each solvent manifest
hash-locks all of the following:

- the molecular site-model source and the bulk 1D-RISM input;
- matching `.xvv` and `.cvv` files, which are reparsed using the native-QV,
  source-`SMEAR` checks above;
- the thermodynamic-output source and a strict solvent-side short-range
  certificate that binds the model, input, XVV, Cvv, thermodynamic output,
  and provenance hashes;
- a provenance statement explicitly declaring that MNSol, FreeSolv,
  development, confirmation, and blind target labels were not used; and
- temperature, pressure, closure, standard-state, pressure,
  partial-molar-volume, sign, and Coulomb-tail conventions.

Changing any file causes a SHA-256 failure; a missing field, a state mismatch,
an unregistered Coulomb tail, a path outside the asset root, or a target-label
declaration other than explicit `false` also fails closed.  The registry does
not infer an absent solvent from a dielectric constant, substitute a GAFF or
AM1-BCC **solute** topology, or choose a surrogate from an observed error.
Before future target-solute scoring, its 11 pre-registered solvent IDs must be
present as one frozen panel rather than selected record by record.

The provenance document is itself parsed rather than treated as an opaque
note.  It must record exactly two source-identical `rism1d` executions.  Both
runs must reproduce Cvv and the thermodynamic self-test exactly, reproduce XVV
after normalizing only the `DATE` value in its first `%VERSION` line, and expose
complete primary and temperature-derivative residual sequences whose final
values pass the tolerance declared in the frozen input.  The two raw XVV and
transcript hashes must differ, while all non-`DATE` XVV bytes, Cvv, and
thermodynamic output reproduce exactly.  Each raw first line is retained, so
every run's XVV hash remains reconstructible from that line and the frozen
body.  Package, binary, command, literature, source hashes, package-relative
site-model path, and the no-training/no-target policy are part of the same
content-addressed certificate.  This establishes deterministic source
generation; it does not establish a converged three-dimensional liquid or an
accuracy result.

The source audit treats the AMBER MDL, its `rism1d` namelists, and the
generated XVV metadata as three serializations of one molecular liquid.
Atom/site counts, multiplicities, masses, charges, Lennard--Jones parameters,
labelled rigid geometry, density, dielectric, temperature, grid, source
`SMEAR`, closure, generation controls (`SELFTEST`, `OUTLIST`, `MAXSTEP`), and
model filename must agree.  The canonical manifest-to-MDL boundary is
chirality preserving and therefore requires congruence under a **proper**
rotation.  The MDL-to-XVV boundary instead compares the complete labelled
pairwise-distance matrix: AmberTools may independently choose principal-axis
signs and serialize an equivalent reflected coordinate frame.  Thus an
arbitrary reflected manifest still fails, while an AmberTools axis-sign
reflection in generated XVV coordinates does not falsely reject the same
molecular source.

AMBER's decimal MDL charge serialization may leave a residual of the order of
its final printed digit.  Such a residual is first required to pass a fixed
source-rounding gate and is then removed by the unique Euclidean projection
onto exact neutrality,

\[
\boxed{
\widetilde{\mathbf q}
=
\mathbf q^{\rm serial}
-\mathbf 1
\frac{\mathbf 1^T\mathbf q^{\rm serial}}{N}
},
\qquad
\mathbf 1^T\widetilde{\mathbf q}=0.
\]

The projected vector, the unmodified serialized vector, and the algebraic rule
are all retained.  After evaluating the orthogonal projection, its last
component absorbs only the floating-point summation residue needed to serialize
an exactly zero sum.  This prevents an \(O(10^{-9}e)\) text-rounding residue
from entering a periodic zero mode; it is a fixed constraint projection, not a
solvation-error fit or a charge-model adjustment.

[`route2-v0-solvent-asset-inventory-v1.json`](benchmarks/route2-v0-solvent-asset-inventory-v1.json)
records the current deliberately incomplete state.  The checkout may contain
source-complete cSPC/E **candidates** with distinct closure roles, while
**zero** physical default-solvent assets remain admitted.  Consequently
total-free-energy execution remains rejected.  This registry is provenance
infrastructure only: it neither defines
the MACE-native \(u^{\mathrm{sr}}\), maps a production short-range kernel,
minimizes a liquid functional, nor changes an accuracy number.


#### 4.2.10 Frozen nonlocal-dielectric electrostatic control

The next zero-training control is deliberately narrower than a molecular
liquid functional.  For a fixed, neutral solute charge density
\(\rho_0(\mathbf r)\) on a periodic Cartesian grid, and for a separately
frozen solvent response spectrum \(\epsilon_{\mathcal S}(\mathbf k)\), define

\[
\widehat V_{\mathrm{reac}}(\mathbf k)=
\begin{cases}
\dfrac{4\pi}{k^2}
\left[\epsilon_{\mathcal S}(\mathbf k)^{-1}-1\right]
\widehat\rho_0(\mathbf k), & \mathbf k\ne0,\\
0, & \mathbf k=0,
\end{cases}
\qquad
G_{\mathrm{pol}}^{(0)}[\rho_0]
=\frac12\int\rho_0(\mathbf r)V_{\mathrm{reac}}(\mathbf r)d\mathbf r.
\]

The admitted numerical conditions are not fit parameters:

\[
\epsilon_{\mathcal S}(\mathbf k)\in\mathbb R,\qquad
\epsilon_{\mathcal S}(-\mathbf k)=\epsilon_{\mathcal S}(\mathbf k),\qquad
\epsilon_{\mathcal S}(\mathbf k)\ge1,\qquad
\int\rho_0=0.
\]

They make the discrete FFT operator self-adjoint and passive:

\[
\frac{\delta G_{\mathrm{pol}}^{(0)}}{\delta\rho_0}
=V_{\mathrm{reac}},\qquad
\langle\rho_1,V_{\mathrm{reac}}[\rho_2]\rangle
=\langle\rho_2,V_{\mathrm{reac}}[\rho_1]\rangle,\qquad
G_{\mathrm{pol}}^{(0)}\le0.
\]

`route2_v0_nonlocal_dielectric.py` implements exactly this one scalar and its
one derivative.  Its tests lock an exact reciprocal Fourier mode,
finite-difference differentiation, reciprocal pairing, passivity,
translation covariance, the zero-mode gauge, and rejection of non-even,
active, scalar-spectrum, or non-neutral inputs.  It has no trainable weights
and cannot inspect experimental solvation records.

This is **not** an alternative name for the RISM long-range control in Section
4.2.7.  That term is a positive source-SMEAR Poisson contribution needed to
reconstruct the RISM direct-correlation split.  The present term is the
negative vacuum-subtracted reaction component of a frozen dielectric response.
They may be joined only by a later, explicitly derived common liquid
functional; adding their scalars ad hoc would double-count or change the
reference convention.

Nor does a reported macroscopic dielectric constant identify this response:

\[
\epsilon_s=\lim_{k\to0}\epsilon_{\mathcal S}(\mathbf k)
\not\Rightarrow
\bigl\{\epsilon_{\mathcal S}(\mathbf k\ne0),
 u^{\mathrm{sr}}_{\alpha},G_{\mathrm{cav}},G_{\mathrm{disp}}\bigr\}.
\]

Therefore a custom solvent supplied only as a name or scalar \(\epsilon_s\)
may run this class solely as an explicitly labelled continuum-electrostatics
diagnostic, never as a molecular-liquid or total-solvation result.  A future
physical branch must freeze the full reciprocal response (or an equivalent
molecular susceptibility) together with the site model, short-range
interaction, cavity/non-electrostatic terms, thermodynamic convention, and
provenance required by Section 4.2.9.  The current inventory contains none of
those physical 11-solvent assets.  This control also rejects charged solutes;
a charged periodic branch requires a separately preregistered background,
finite-size, and standard-state free-energy derivation.

#### 4.2.11 Parameter-free frozen-density Pauli-overlap control

The first short-range **scalar control** is the Thomas--Fermi nonadditive
kinetic energy from frozen-density embedding.  For separately declared,
nonnegative solute and solvent *electron* reference densities on one common
grid, it is

\[
T_{\mathrm{TF}}^{\mathrm{nad}}[n_{\mathrm{sol}},n_{\mathrm{liq}}]
=C_{\mathrm{TF}}\int d\mathbf r\,
\left[
(n_{\mathrm{sol}}+n_{\mathrm{liq}})^{5/3}
-n_{\mathrm{sol}}^{5/3}-n_{\mathrm{liq}}^{5/3}
\right],
\qquad
C_{\mathrm{TF}}=\frac3{10}(3\pi^2)^{2/3}.
\]

This coefficient is fixed by the atomic-unit Thomas--Fermi functional; it is
not an overlap radius, a solvent scale, or a fitted replacement for the
missing liquid physics.  Convexity gives

\[
T_{\mathrm{TF}}^{\mathrm{nad}}\ge0,
\]

with equality for pointwise-disjoint density support.  The two exact discrete
functional derivatives are

\[
\frac{\delta T_{\mathrm{TF}}^{\mathrm{nad}}}{\delta n_{\mathrm{sol}}}
=\frac53C_{\mathrm{TF}}
\left[(n_{\mathrm{sol}}+n_{\mathrm{liq}})^{2/3}
-n_{\mathrm{sol}}^{2/3}\right],
\]

and the corresponding solvent expression under
\(n_{\mathrm{sol}}\leftrightarrow n_{\mathrm{liq}}\).  Thus this is one
energy/derivative-conjugate Pauli-repulsion primitive rather than a
post-hoc exclusion energy.

`route2_v0_frozen_density_embedding.py` implements this scalar, both
functional derivatives, immutable nonnegative inputs, and construction/scope
tags.  Its tests lock nonnegativity, the exact disjoint-support limit,
exchange symmetry, finite-difference derivatives, translation covariance,
and rejection of invalid densities.  It has no trainable weights and does
not inspect target solvation labels.

This still does **not** define the V0-FD-S short-range site potential.  The
checked-in promolecular table is a solute-side reference only, and the
checkout has no frozen solvent electron-density asset, solvent embedding
construction, dispersion term, or common liquid free-energy functional.
Moreover, the Thomas--Fermi integrand is \(C^1\) but not \(C^2\) at zero
density.  It is therefore deliberately excluded from Newton/Hessian and
production-PES claims until a smooth source-provenanced functional and its
stationary liquid state are derived together.

#### 4.2.12 Molecular-orientation external-potential control

A molecular solvent cannot obtain its short-range contribution by adding a
separate Thomas--Fermi nonadditive scalar for each site: the density
nonadditivity is nonlinear.  For a declared rigid solvent reference

\[
\mathcal S=\{Z_\alpha,z_\alpha,\mathbf r_\alpha\}_{\alpha=1}^{N_s}
\]

and a configuration \((\mathbf X,\mathbf\Omega)\), define its global site
positions and one **whole-molecule** promolecular reference density as

\[
\mathbf x_\alpha=\mathbf X+\mathbf\Omega\mathbf r_\alpha,
\qquad
n_{\mathcal S}^{\mathrm{ref}}
(\mathbf r;\mathbf X,\mathbf\Omega)
=\sum_{\alpha=1}^{N_s}
n_{Z_\alpha}^{\mathrm{ref}}(\mathbf r-\mathbf x_\alpha).
\]

The configuration-wise V0 control is then the one declared scalar

\[
u_{\mathcal S}^{\mathrm{V0}}(\mathbf X,\mathbf\Omega)
=\sum_{\alpha=1}^{N_s}z_\alpha
\phi^G_{c_0}(\mathbf x_\alpha)
+T_{\mathrm{TF}}^{\mathrm{nad}}
\!\left[n_{\mathrm{ref}},
n_{\mathcal S}^{\mathrm{ref}}(\mathbf X,\mathbf\Omega)\right].
\]

The first term uses the unmodified all-space MACE Gaussian potential; the
second uses the independent positive reference densities from Section 4.2.4.
The latter is specifically **not**
\(\sum_\alpha T_{\mathrm{TF}}^{\mathrm{nad}}
[n_{\mathrm{ref}},n_{Z_\alpha}^{\mathrm{ref}}]\).  The site charges,
geometry, and their total-charge convention must be content-addressed by a
future physical solvent asset before they may be used for a liquid result.
Thus this introduces no new fit, overlap radius, density relabelling, or
MACE field update; it does not assert that the promolecule is MACE density or
that it contains a bonded molecular electron density.

`route2_v0_molecular_external_potential.py` implements this direct discrete
control.  Its checks require a proper rotation, a declared molecular total
charge, the same nuclear geometry for the MACE Gaussian and solute
promolecule, and an exact Pauli-plus-electrostatic ledger.  Its tests lock the
disjoint-support zero, joint-translation covariance, and invariance to a
change of molecular reference origin.  It is deliberately configuration-wise:
it provides neither an MDFT/3D-RISM excess functional, orientational measure,
bulk correlation, dispersion term, liquid minimization, standard-state term,
force, PES, nor solvation/accuracy result.  Linear radial interpolation and
the \(C^1\)-but-not-\(C^2\) Thomas--Fermi integrand keep it energy-only until
one source-provenanced stationary molecular-liquid functional is derived.

#### 4.2.13 Exact rigid-molecule ideal-gas configuration functional

The configuration energy in Section 4.2.12 is not itself a liquid free
energy.  Let
\(\Gamma=(\mathbf X,\mathbf\Omega)\) denote the translation and rigid
orientation of one solvent molecule, and use the unnormalised Haar measure

\[
\int d\mathbf\Omega=8\pi^2.
\]

For bulk molecular number density \(\rho_b\), the uniform
configuration-density convention is

\[
\nu_b=\frac{\rho_b}{8\pi^2}.
\]

The exact ideal-gas grand-potential difference for the external energy
\(u_{\mathcal S}^{\mathrm{V0}}(\Gamma)\) is

\[
\Delta\Omega_{\mathrm{id}}[\nu;u]
=\int d^3X\,d\mathbf\Omega\,
\left\{
k_BT\left[
\nu\ln\frac{\nu}{\nu_b}-(\nu-\nu_b)
\right]
+\nu u
\right\}.
\]

It has the exact variational derivative and stationary state

\[
\frac{\delta\Delta\Omega_{\mathrm{id}}}{\delta\nu}
=k_BT\ln\frac{\nu}{\nu_b}+u=0,
\qquad
\nu^*(\Gamma)=\nu_b\exp[-\beta u(\Gamma)],
\]

with stationary scalar

\[
\Delta\Omega_{\mathrm{id}}^*
=-k_BT\nu_b
\int d^3X\,d\mathbf\Omega\,
\left[\exp[-\beta u(\Gamma)]-1\right].
\]

`route2_v0_molecular_ideal_gas.py` discretizes this **one scalar** with fixed
positive configuration-space weights \(w_i\) for
\(d^3X\,d\mathbf\Omega\).  It locks the \(8\pi^2\) orientation convention,
the exact configuration grid shared with the MACE-plus-TF external potential,
the analytic stationary density, and the finite-difference gradient identity.
There is no mixing parameter in this minimizer.

This still intentionally omits liquid physics.  A physical molecular branch
must add one source-provenanced excess functional with its bulk tangent,

\[
\Delta\Omega[\nu;u]
=\Delta\Omega_{\mathrm{id}}[\nu;u]
+F_{\mathrm{ex}}[\nu]-F_{\mathrm{ex}}[\nu_b]
-\int d^3X\,d\mathbf\Omega\,
\left.
\frac{\delta F_{\mathrm{ex}}}{\delta\nu}
\right|_{\nu_b}
(\nu-\nu_b).
\]

The final term is required to make the declared bulk state stationary at
\(u=0\); omitting it would silently change the chemical-potential convention.
Until a frozen molecular correlation/EOS/dispersion asset defines
\(F_{\mathrm{ex}}\), the ideal result is not a liquid calculation, a
standard-state correction, a force, or a solvation/accuracy result.

#### 4.2.14 Projected molecular-site HNC scalar control

The ideal molecular density \(\nu(\mathbf X,\mathbf\Omega)\) and the
site-density HNC reference in Section 4.2 use different state spaces.  They
must not be added as unrelated corrections.  On a declared periodic Cartesian
cell, define one fixed discrete map from molecular configurations \(i\) to
site densities \((a,g)\),

\[
n_{a g}
=\frac{1}{\Delta V}
\sum_i w_i A_{a g i}\nu_i,
\]

where \(w_i\) integrates \(d^3X\,d\mathbf\Omega\), \(A_{agi}\geq0\)
is a dimensionless site-occupancy weight, and \(\Delta V\) is a Cartesian
grid-voxel volume.  For distinct site type \(a\) with molecular multiplicity
\(m_a\), the map is admitted only when

\[
\sum_g A_{agi}=m_a,
\qquad
\sum_i w_i=V\,8\pi^2,
\qquad
\rho_a^b=m_a\rho_b.
\]

It then maps the uniform configuration reference
\(\nu_b=\rho_b/(8\pi^2)\) exactly to the declared HNC bulk density.  Its
discrete adjoint is fixed by the pairing identity

\[
\Delta V\sum_{ag} v_{ag}\,\delta n_{ag}
=\sum_i w_i
\left[\sum_{ag} A_{agi}v_{ag}\right]\delta\nu_i.
\]

For one reciprocal Cartesian direct-correlation kernel, this permits the
single projected scalar

\[
\Delta\Omega_{\mathrm{mHNC}}[\nu;u]
=\Delta\Omega_{\mathrm{id}}[\nu;u]
-\frac{k_BT}{2}\Delta V
\sum_{ag}\delta n_{ag}
\left(c*\delta n\right)_{ag},
\qquad
\delta n=n-n_b.
\]

Its exact dimensionless configuration derivative is

\[
\beta\frac{\delta\Delta\Omega_{\mathrm{mHNC}}}{\delta\nu_i}
=\ln\frac{\nu_i}{\nu_b}+\beta u_i
-\sum_{ag}A_{agi}\left(c*\delta n\right)_{ag}.
\]

The same scalar fixes the local stability operator.  For a signed molecular
configuration-density tangent \(d\),

\[
(\mathcal H_{\nu}d)_i
=\frac{d_i}{\nu_i}
-\left[P^T C_{\mathrm{RISM}}P d\right]_i,
\qquad
\delta^2\Omega[d,d]
=k_BT\sum_i w_i d_i(\mathcal H_{\nu}d)_i.
\]

Reciprocity of the fixed occupancy adjoint and of the RISM kernel gives
\(\sum_iw_i a_i(\mathcal H_\nu b)_i=\sum_iw_i b_i(\mathcal H_\nu a)_i\).  Therefore a
converged HNC residual is admitted as a stable liquid state only after the
relevant allowed tangent space has positive quadratic form; a Picard spectral
radius or mixing factor is not a thermodynamic stability criterion.
`route2_v0_molecular_site_hnc.py` exposes the Hessian-vector product and the
quadratic form, and locks their finite-difference scalar and reciprocity
identities.  It does not add a fitted response term.

Thus a numerical Picard factor can only accelerate the update; it cannot
alter the stationary equation or be selected from solvation errors.
`route2_v0_molecular_site_hnc.py` locks the common Cartesian grid, exact
periodic-cell orientation measure, molecular-site multiplicities, uniform
bulk map, projection adjoint, scalar/gradient identity, and zero-correlation
ideal-gas limit.

This is deliberately only a bridge control.  The current `Route2V0SiteHNCAsset`
is synthetic; a raw 1D-RISM \(C_{vv}\) table remains inadmissible here until
its source-SMEAR Coulomb tail is assembled through the exact energy-conjugate
operator in Section 4.2.15.  No frozen physical molecular correlation,
solvent-side short-range source, pressure/standard-state convention, real
liquid solve, force, PES, or accuracy result follows from this scalar.

#### 4.2.14a Closure identity: HNC scalar versus PSE-\(n\) bulk source

The symbol \(C_{\mathrm{RISM}}\) in Section 4.2.14 is not an arbitrary
site--site direct correlation.  The quadratic excess term and its displayed
derivative are specifically the **HNC** functional, so the source must be
\(C_{\mathrm{HNC}}\).  Matching solvent name, temperature, site model,
charges, radial grid, or Coulomb split cannot make a source produced by a
different closure an HNC functional.

For example, with the usual PSE-\(n\) closure variable \(\Xi_\alpha\), the
positive branch is instead truncated,

\[
g_\alpha(\mathbf r)=
\begin{cases}
\exp\!\left[\Xi_\alpha(\mathbf r)\right],
    & \Xi_\alpha\leq0,\\
\displaystyle\sum_{j=0}^{n}
\frac{\Xi_\alpha(\mathbf r)^j}{j!},
    & \Xi_\alpha>0.
\end{cases}
\]

Its path-independent excess-chemical-potential expression has the associated
closure-specific positive-\(\Xi\) correction proportional to
\(\Theta(\Xi_\alpha)\Xi_\alpha^{n+1}/(n+1)!\), rather than the HNC
quadratic scalar alone [6, 28].  Thus a PSE3 `Cvv` is not rejected because it
is numerically inaccurate; it is rejected because silently placing it in
\(\Delta\Omega_{\mathrm{mHNC}}\) would report the derivative of one closure
and the free energy of another.

Accordingly, `route2_v0_mace_cluster_rism_bridge.py` requires the frozen
asset to declare `closure == "HNC"` **before** it builds either the reciprocal
kernel or the MACE/RISM bridge.  The checked-in cSPC/E PSE3 source is retained
only as a source-complete control for a future, separately preregistered
PSE-\(n\) configuration-space scalar.  The closure-aligned cSPC/E HNC source
is likewise only source-complete: it has not passed production
Cartesian/orientation convergence, physical liquid/EOS/standard-state, force,
eleven-solvent, or chemistry gates.  The source choice follows the scalar
identity before reading any target-solvation result; no closure search or
error-selected substitution is allowed.

#### 4.2.15 Energy-conjugate periodic RISM kernel control

Amber's source convention is not a fitted decomposition.  After the declared
short-range transform, its direct correlation is

\[
C^{\mathrm{full}}_{ab}(\mathbf k)
=C^{\mathrm{sr}}_{ab}(\mathbf k)
-q_aq_bG_\eta(\mathbf k),
\]

where \(q_a\) is Amber's native QV scale converted only from Angstrom to
Bohr length units and \(G_\eta\) is the zero-average periodic
\(4\pi e^{-\eta^2k^2/4}/k^2\) multiplier using that same source SMEAR.
The matching excess scalar is

\[
\Delta F_{\mathrm{ex}}
=-\frac{k_BT}{2}\Delta V
\sum_{ag}\delta n_{ag}
\left(C^{\mathrm{sr}}*\delta n\right)_{ag}
+\frac{k_BT}{2}\Delta V
\sum_g\rho_Q(\mathbf r_g)V_Q(\mathbf r_g),
\]

with \(\rho_Q=\sum_aq_a\delta n_a\).  Its site derivative is

\[
\beta\frac{\delta\Delta F_{\mathrm{ex}}}{\delta n_a}
=-\left(C^{\mathrm{sr}}*\delta n\right)_a+q_aV_Q.
\]

`route2_v0_rism_energy_conjugate.py` constructs the equivalent full periodic
direct correlation and proves that its existing HNC convolution, split scalar,
and split derivative agree.  It rejects a mismatched grid or SMEAR and makes
the resulting full `Route2V0SiteHNCAsset` available to the molecular
projection in Section 4.2.14.  Thus no Coulomb prefactor, tail smoothing, or
long-range coefficient is selected from solvation labels.

This is still only a mathematical assembly gate.  The source-complete cSPC/E
PSE3 and HNC candidates do not supply production-grid certification, a complete
MACE-native short-range liquid potential, force certification, or the other ten
pre-registered solvent candidates.  The current molecular-HNC bridge accepts
only the HNC candidate; a real endpoint still needs those gates, the
asset-bound MACE-cluster-to-liquid connector in Section 4.2.17, and one frozen
same-functional thermodynamic/standard-state convention.

#### 4.2.16 Zero-field MACE molecular external-potential control

The current MACE-POLAR field-response map cannot supply a variational
electronic state.  That observation does **not** invalidate the model's
ordinary zero-field energy, which is already one scalar of nuclear geometry.
For a neutral singlet solute \(A\) and a rigid neutral-singlet solvent molecule
\(B_\Gamma\) at configuration
\(\Gamma=(\mathbf X,\Omega)\), define instead the complete zero-field molecular
external potential

\[
u_{\mathrm{MACE}}(A,B_\Gamma)
=E^0_{\mathrm{MACE}}(A\cup B_\Gamma)
-E^0_{\mathrm{MACE}}(A)
-E^0_{\mathrm{MACE}}(B_\Gamma).
\]

All three evaluations use the same content-addressed official MACE-POLAR-1-M
checkpoint, its float64 Route-2 profile, and its default molecular real-space
long-range evaluator.  No local reaction potential, field feature, induced
density, or response iterate is passed to any evaluation.  Thus the isolated
fragment references cancel identically; no atom-charge, Lennard-Jones,
Thomas--Fermi, damping, switching, or solvation-label coefficient is added.

This object has the exact force ledger

\[
\mathbf F_A^{\mathrm{int}}
=\mathbf F_{A\cup B_\Gamma,A}^{0}
-\mathbf F_A^{0},
\qquad
\mathbf F_{B}^{\mathrm{int}}
=\mathbf F_{A\cup B_\Gamma,B}^{0}
-\mathbf F_{B_\Gamma}^{0},
\]

so \(\mathbf F_A^{\mathrm{int}}=-\partial_Au_{\mathrm{MACE}}\) whenever the
zero-field MACE force is the derivative of its reported scalar.  If this
external potential later enters the stationary molecular functional of Section
4.2.14, envelope differentiation gives the solvation-force contribution

\[
\mathbf F_A^{\mathrm{liq}}
=\sum_iw_i\nu_i^*\mathbf F_A^{\mathrm{int}}(\Gamma_i),
\]

without differentiating a liquid response iterate.  This is precisely why the
source is a safer V0 route than trying to interpret the nonreciprocal
field-conditioned density as a physical charge response.

The route2_v0_mace_cluster_external_potential.py module implements this
three-energy scalar and, when requested, the matching three-force difference.
It fails closed for a changed checkpoint, an unsupported runtime profile,
periodic, charged, or open-shell fragments, and any attempt to call a
field-response interface.  The resulting \(u_{\mathrm{MACE}}\) must be used
as one whole molecular external potential; adding the older
MACE-Gaussian-plus-Thomas--Fermi control would double count a distinct source
definition.

Both source definitions now implement one narrow molecular-external-potential
contract.  Consequently the MACE cluster vector can enter the exact ideal
configuration scalar and the projected molecular-site HNC scalar without an
adapter that changes its energy.  The zero-correlation limit recovers the same
analytic ideal stationary density, while the nonzero HNC derivative remains
the adjoint projection of the same configuration-space scalar.  This is an
interface result, not a liquid-physics claim: it prevents accidental mixing of
two external-potential ledgers while retaining one variational liquid route.

The source is not yet a total liquid theory.  Pairing an MACE solute--solvent
external potential with a frozen solvent-side RISM correlation defines a
declared **hybrid reference functional**, not an assertion that both terms came
from one microscopic force field.  Before physical execution, the frozen
solvent asset must bind its molecular geometry, bulk \(C_{vv}\), closure,
thermodynamic convention, and explicit cross-model reference statement to the
same source ledger.  The current checkout has no such 11-solvent bundle, no
production molecular quadrature/grid certification, no physical
closure/EOS certificate or standard-state term, and no benchmark result.  The
same-functional discrete pressure identity below does not fill those physical
asset gates.  This therefore remains a zero-training external-potential
control, not a claimed accuracy improvement.

#### 4.2.17 Asset-bound MACE-cluster/RISM molecular-HNC bridge

The previous controls establish the two terms of a prospective liquid scalar,
but neither a solvent name nor a raw `Cvv` table proves that they describe the
same declared liquid convention.  The bridge therefore admits only

\[
\boxed{
\Omega_{\mathrm{hyb}}[\nu;\mathbf R]
=\Omega_{\mathrm{id}}[\nu;u_{\mathrm{MACE}}(\mathbf R,\Gamma)]
+F_{\mathrm{ex}}^{\mathrm{RISM}}[P\nu]
}
\]

when all source identities are already frozen.  Here \(P\) is the same
configuration-to-site projection used in Section 4.2.14; it is not an
energy-changing adapter.  The RISM term is built only by splitting the
hash-locked radial source at its native SMEAR, transforming the declared
short-range remainder subject to its frozen tail and radial-Nyquist gates, and
joining it to the matching zero-average periodic smeared Coulomb operator.
Thus the molecular stationarity equation is the derivative of one scalar,

\[
\frac{\beta}{w_i}\frac{\partial\Omega_{\mathrm{hyb}}}{\partial\nu_i}
=\log\!\frac{\nu_i}{\nu_\mathrm{bulk}/8\pi^2}
+\beta u_{\mathrm{MACE},i}
-\left[P^T C_{\mathrm{RISM}}(P\nu-n_\mathrm{bulk})\right]_i=0.
\]

`route2_v0_mace_cluster_rism_bridge.py` creates this assembly.  It rejects a
changed solvent-side hash, a source whose declared closure is not HNC, a
reciprocal kernel derived from another radial asset or tail convention, an
external grid different from the RISM grid, or a MACE solvent geometry/charge
record that differs from the **canonical
molecular reference stored in the frozen asset**.  Schema-v2 validation binds
that record to the exact `site_model` digest and requires its named per-atom
RISM site map to reproduce the XVV multiplicities before the bridge is even
constructed.  The MACE vector itself carries the exact official checkpoint,
runtime profile, and long-range-evaluator provenance on every configuration,
so a hand-assembled three-energy ledger cannot impersonate this source.

This is a hash-locked source declaration, not a claim that MAPLE can parse
every upstream molecular-model syntax or infer a geometry from a solvent name.
The physical asset must provide a human-auditable transcription from the hashed
model source to the canonical rigid geometry, charges, and named site types;
the manifest records that transcription and rejects a mismatch rather than
allowing bridge callers to hand-supply it.  This is intentionally stricter than
a dielectric-only custom-solvent input: a macroscopic \(\epsilon\) alone fixes
neither molecular geometry, finite-\(k\) correlation, nor the
non-electrostatic liquid functional.

At a stationary \(\nu^*(\mathbf R)\), the frozen-asset control also has an
unambiguous **external MACE envelope-force** identity.  Since the fixed RISM
asset, Cartesian grid, configuration quadrature, and projection do not depend
on a solute Cartesian coordinate \(R_I\),

\[
\frac{d\Omega_{\mathrm{hyb}}}{dR_I}
=
\left.\frac{\partial\Omega_{\mathrm{hyb}}}{\partial R_I}\right|_{\nu^*}
=
\sum_i w_i\nu_i^*\frac{\partial u_{\mathrm{MACE},i}}{\partial R_I},
\qquad
\boxed{
F_I^{\mathrm{MACE-env}}
=\sum_i w_i\nu_i^* f_{I,i}^{\mathrm{MACE}}
}.
\]

`stationary_mace_solute_force_ev_per_angstrom` implements exactly that finite
sum only after checking the HNC residual and the availability of the
configuration-wise three-energy MACE forces.  No \(d\nu^*/d\mathbf R\) or
fixed-point adjoint appears: its coefficient is the verified stationary
residual.  This is deliberately not yet a complete physical liquid force.
Any moving cavity, coordinate-dependent solvent functional, or production
quadrature requires its own explicit derivative in the same scalar before a
PES claim is allowed.

The bridge is therefore a structural common-energy result, not a liquid
endpoint.  It has no registered physical 11-solvent asset, production
orientation/grid certificate, independently certified physical closure/EOS,
standard-state term, complete solvation force, or chemistry score.  In
particular, the synthetic test asset proves only source binding,
scalar/derivative pairing, the fixed-asset envelope identity, and the
same-functional thermodynamic identities below; it may not be relabelled as a
cSPC/E or general-water solvation prediction.

#### 4.2.18 Same-functional pressure and fixed-solute ensemble identity

The stationary value of the molecular HNC scalar is a grand-potential
difference \(\Delta\Omega[\nu^*]\).  A pressure term may not be copied from a
different site-density 3D-RISM functional or read from the external
\(1\ {\rm bar}\) state label.  It must be derived from the scalar actually
minimized.

Let the periodic cell volume be \(V=N_g\Delta v\), the bulk molecular number
density be \(\rho_b\), the uniform configuration density be
\(\nu_b=\rho_b/(8\pi^2)\), and the bulk density of site type \(a\) be
\(n_a^b=m_a\rho_b\).  Emptying this exact molecular functional gives its
vacuum-limit pressure:

\[
\begin{aligned}
P_F V
&=
\lim_{\epsilon\rightarrow0^+}
\left[
\Delta\Omega_{\rm id}[\epsilon\nu_b;0]
-\frac{k_BT}{2}\Delta v
\sum_{ag}\delta n_{ag}
(C*\delta n)_{ag}
\right] \\
&=
k_BT\rho_bV
-\frac{k_BT}{2}\Delta v
\sum_{ag}(-n_a^b)
\left[C*(-n^b)\right]_{ag}.
\end{aligned}
\]

For a translationally invariant continuum kernel this is equivalently

\[
\boxed{
P_F
=k_BT\left[
\rho_b
-\frac12\sum_{ab}n_a^b n_b^b\widehat c_{ab}(0)
\right].
}
\]

The first term occurs **once per solvent molecule** because the variational
state is \(\nu(\mathbf X,\mathbf\Omega)\).  The
\((n_s+1)\rho_bk_BT/2\) ideal term derived for the distinct stock
site-density 3D-RISM functional is therefore not interchangeable with this
one, even when the frozen correlation table originated in a RISM solve.
MAPLE evaluates the vacuum-limit quadratic form through the same discrete
convolution used in \(\Delta\Omega\), avoiding a second FFT normalization or
\(k=0\) transcription.

At the stationary state,

\[
N^*=\sum_iw_i\nu_i^*,
\qquad
N_b=\rho_bV,
\qquad
\boxed{
\bar V_F=\frac{N_b-N^*}{\rho_b}.
}
\]

The fixed-solute \(\mu VT\rightarrow NPT\) ensemble identity associated with
this approximate functional is then

\[
\boxed{
\Delta G_F^{\rm fixed}
=\Delta\Omega[\nu^*]-P_F\bar V_F.
}
\]

`route2_v0_molecular_thermodynamics.py` implements these equations, and
`stationary_fixed_solute_thermodynamics` binds them to the exact
asset-verified MACE/RISM bridge.  Both reject a nonstationary state or a state
from another projection.  The result contains no fitted coefficient.

This classification is strict:

- PC+ adds an additional ideal-density volume term; the primary derivation
  later described that microscopic-solute term as an empirical adjustment.
  It is not present here.
- UC, MILC, and linear partial-volume regressions use fitted coefficients and
  are excluded.
- the external physical-pressure work
  \(P_{\rm ext}\bar V_{\rm phys}\), mobile-solute translational convention,
  and gas/solution standard-state conversion are separate thermodynamic
  categories.  None is silently set from `pressure_bar` or folded into this
  object.
- a physical result still requires cell/quadrature convergence and an
  all-atom frozen solvent asset.  The exact discrete identity does not prove
  the hybrid functional chemically accurate.

#### 4.2.19 Pure-solvent weighted-density bridge: repair the HNC cavity *before* stationarity

The HNC reference is useful because its scalar, derivative, and Hessian are
explicit, but a quadratic expansion about the liquid bulk state has a known
cavity/pressure defect.  In particular, an HNC-only liquid can assign a very
large vacuum-limit pressure and therefore overestimate the work of making a
hydrophobic cavity.  This is a structural candidate for the observed
large-error tail; it is not evidence that a radius, a half factor, or a
solvation-label correction should be tuned.

The admissible V0 repair is a **bridge term inside the scalar before the
liquid is minimized**.  It follows the weighted-density molecular-DFT form of
Gageat *et al.* rather than an a-posteriori PC/PC+ adjustment.  Let $D$ be a
second, source-bound projection from the configuration density to molecular
centres:


\[
\rho_g=(D\nu)_g
=\frac{1}{\Delta v}\sum_iw_iD_{gi}\nu_i,
\qquad
\sum_gD_{gi}=1,
\qquad
D\nu_b=\rho_b.
\]

The exact discrete pairing is

\[
\Delta v\sum_gq_g(Dd)_g
=\sum_iw_i(D^Tq)_i d_i.
\]

It is intentionally a molecular-centre map, not an average of RISM site
densities: site multiplicity and displaced atomic sites do not define a
molecular cavity field.  Let $K$ be a real, nonnegative, normalized, and
periodic-even kernel,

\[
\Delta v\sum_gK_g=1,
\qquad K_g=K_{-g},
\qquad \bar\rho=K*\rho.
\]

The bridge scalar is then

\[
\boxed{
F_{\rm B}[\nu]
=\Delta v\sum_g\left[
A_s(\bar\rho_g-\rho_b)^3
+B_s\bar\rho_g^2(\bar\rho_g-\rho_b)^4
\right].
}
\]

Its cubic coefficient is **not free**.  Let $P_{\rm HNC}$ be the
vacuum-limit pressure evaluated by Section 4.2.18 from the exact molecular
HNC convolution, with its one-molecule ideal term, and let $P_s$ be the
independently frozen pressure of the pure liquid state.  The empty-state
condition for the augmented scalar gives

\[
\boxed{
A_s=\frac{P_{\rm HNC}-P_s}{\rho_b^3}.
}
\]

Consequently

\[
\frac{\Omega_{\rm HNC+B}[0;0]}{V}
=P_{\rm HNC}-A_s\rho_b^3=P_s.
\]

This is an identity of the scalar actually minimized.  It is neither the
stock site-3D-RISM pressure formula nor a post-hoc volume correction.  The
positive quartic coefficient $B_s$ controls the barrier between liquid and
gas-like densities; its value and $K$ are admissible only through a
content-addressed **pure-solvent** planar-interface certificate which uses
the frozen bulk correlation and independently sourced surface tension.  No
MNSol, FreeSolv, development, confirmation, or blind solvation value may
enter that certificate.  Thus matching a bulk pressure/surface tension is a
predeclared thermodynamic boundary condition, not target-solvation fitting.

The quartic anchor has a direct uniqueness certificate rather than a
trial-and-error search.  On a predeclared bracket, let
\(\nu^*_{B_s}\) be the selected stationary planar-interface profile of the
same frozen scalar and define the excess grand potential per transverse area

\[
\gamma(B_s)
=\frac{\Omega_{B_s}[\nu^*_{B_s}]-\Omega_{\rm bulk}}{A}.
\]

Provided the selected interface branch remains stationary on that bracket,
the envelope theorem gives

\[
\boxed{
\frac{d\gamma}{dB_s}
=\frac1A\int d^3r\,
\bar\rho_{B_s}^{*2}(\mathbf r)
\bigl(\bar\rho^*_{B_s}(\mathbf r)-\rho_b\bigr)^4
\ge0.
}
\]

The bulk liquid and gas contributions vanish because the quartic bridge is
zero at both \(\bar\rho=\rho_b\) and \(\bar\rho=0\).  For a nontrivial,
resolved interface the integrand is positive on a set of nonzero measure, so
the derivative is strictly positive and the surface-tension equation has at
most one root on that stationary branch.  A valid certificate must therefore
freeze \(B_{\rm low}\), \(B_{\rm high}\), the two endpoint surface tensions,
the independent target surface tension, planar residuals, transverse-area and
interface-count conventions, grid refinement, and the declared root residual;
it must establish

\[
\gamma(B_{\rm low})\le\gamma_{\rm target}
\le\gamma(B_{\rm high}).
\]

This is a pure-liquid thermodynamic boundary-value inversion, not a
solvation-label fit: a solute cavity, MNSol/FreeSolv record, development,
confirmation, or blind error may not select \(B_s\), \(K\), or the bracket.

For clarity, write $\Delta\bar\rho=\bar\rho-\rho_b$.  The local first and second
derivatives are

\[
\begin{aligned}
b'(\bar\rho)
&=3A_s\Delta\bar\rho^2
+B_s\left(2\bar\rho\Delta\bar\rho^4
+4\bar\rho^2\Delta\bar\rho^3\right),\\
b''(\bar\rho)
&=6A_s\Delta\bar\rho
+B_s\left(
2\Delta\bar\rho^4
+16\bar\rho\Delta\bar\rho^3
+12\bar\rho^2\Delta\bar\rho^2
\right).
\end{aligned}
\]

Because both $D$ and $K$ retain their exact adjoints, the bridge supplies

\[
\boxed{
\frac{\beta}{w_i}\frac{\partial F_{\rm B}}{\partial\nu_i}
=\beta\left[D^TK\,b'(KD\nu)\right]_i
}
\]

and, for any signed configuration-density direction $d$, the Hessian action

\[
\boxed{
\delta\!\left(
\frac{\beta}{w_i}\frac{\partial F_{\rm B}}{\partial\nu_i}
\right)[d]
=\beta\left[
D^TK\left\{b''(KD\nu)\,KDd\right\}
\right]_i.
}
\]

The action is self-adjoint in the same quadrature pairing; therefore the
HNC-plus-bridge response remains reciprocal by construction.  The production
code in `route2_v0_molecular_weighted_density_bridge.py` validates the centre
map, the kernel normalization/evenness, the pressure-derived $A_s$, the
source certificate digest, finite-difference gradient/Hessian identities, and
the same-functional pressure identity.  Its Picard mixing remains numerical
only.

This control does **not** establish a chemical result yet.  A physical bridge
asset still needs a frozen all-atom solvent model, source-provenanced
correlation, pressure, surface tension, kernel/certificate, production
orientation/grid convergence, full MACE-cluster external potential, standard
state, force terms, and the preregistered 11-solvent/blind validation sequence.
If $D$, $K$, or any bridge anchor moves with a solute coordinate, its
explicit derivative must be added to the same scalar before any force claim.

##### 4.2.19a Certificate binding is a numerical reproducibility gate, not a fitted term

The words "content-addressed certificate" have a precise implementation
meaning.  For the declared pressure $p_s$ in bar and independently sourced
surface tension $\gamma_s^{\rm SI}$ in N/m, the certificate stores their
unambiguous atomic-unit forms,

\[
P_s=\frac{10^5p_s}
{E_h/(a_0\times10^{-10}\,{\rm m})^3},
\qquad
\gamma_s=\frac{\gamma_s^{\rm SI}}
{E_h/(a_0\times10^{-10}\,{\rm m})^2}.
\]

It rejects a record if either serialized atomic-unit value differs from this
conversion, if $P_s\ge P_{\rm HNC}$, or if

\[
A_s\ne\frac{P_{\rm HNC}-P_s}{\rho_b^3}.
\]

This prevents a nominally physical pressure or a surface tension from becoming
a hidden second fitting knob.  The certificate also stores the complete
stationary planar bracket

\[
(B_{\rm low},B_*,B_{\rm high};\;
\gamma_{\rm low},\gamma_*,\gamma_{\rm high};\;
r_{\rm low},r_*,r_{\rm high}),
\]

and accepts it only when

\[
B_{\rm low}\le B_*\le B_{\rm high},\quad
\gamma_{\rm low}\le\gamma_s\le\gamma_{\rm high},\quad
|\gamma_*-\gamma_s|\le\tau_\gamma,\quad
\max_jr_j\le\tau_{\rm stat}.
\]

It additionally retains transverse area, interface count, coarse/fine grid
surface tensions and their convergence tolerance.  Thus the monotonic-envelope
argument above is tied to a numerically stationary, resolved branch rather
than to a bare scalar $B_s$ copied into a configuration file.

The discrete operators are not identified by a mutable Python object or by a
lossy text dump.  Their values are hashed as little-endian C-order float64
arrays together with the periodic Cartesian layout, shape, origin, spacing,
and construction identifier:

\[
h_D=H\!\left({\tt grid},D\right),\qquad
h_K=H\!\left({\tt grid},K\right).
\]

Before constructing a source-bound bridge asset, the implementation requires
the exact equality of the HNC closure, solvent/model identifiers, temperature,
pressure, molecular bulk density, HNC vacuum-limit pressure, all seven frozen
solvent-source hashes, $h_D$, and $h_K$.  Equivalently, the executable scalar
is admitted only under

\[
\mathcal C_s\equiv
(\mathcal S_s,\rho_b,P_{\rm HNC},P_s,\gamma_s,A_s,B_s,h_D,h_K)
\]

matching the live asset and operators componentwise.  Any changed source,
operator, closure, pressure, or planar root fails closed.  The loaded
certificate is re-hashed and re-parsed at physical admission, so a changed JSON
file or a manually assembled certificate-shaped object cannot stand in for the
content-addressed evidence.  It also carries an explicit evidence scope:
`synthetic-control` remains useful only for scalar/derivative tests, whereas
`physical-pure-liquid-admission` is required by any physical endpoint and is
rejected if the frozen solvent provenance itself explicitly lists a physical
liquid as not claimed.  Thus source binding alone cannot silently upgrade a
test fixture into liquid physics.  The parser also
requires explicit `false` declarations for post-training, fine-tuning,
experimental-solvation fitting, MAP/UQ calibration, and error-driven
cavity/dispersion adjustment, together with the full excluded
MNSol/FreeSolv/development/confirmation/blind label set.

This contract is implemented by
`route2_v0_pure_solvent_bridge_certificate.py` and by
`Route2V0MolecularWeightedDensityBridgeAsset.from_source_bound_pure_solvent_certificate`.
It does **not** make a synthetic certificate physical, prove an HNC liquid
model accurate, or relax any later force, multi-solvent, historical-outlier, or
blind-data gate.

##### 4.2.19b The literature-exact cubic Gaussian WDA is a separate V0 branch

The cubic-plus-quartic bridge above follows the earlier pure-liquid
pressure/surface-tension construction.  It must not be relabelled as the
later simple molecular-WDA form merely because both are variational weighted
density bridges.  Borgis *et al.*'s 2021 construction instead takes the
lowest nonzero angular-independent bridge order,

\[
F_{\rm B}^{\rm cWDA}[\nu]
=k_BT\rho_b\Delta v\sum_g
a_s\left(\frac{(K_\sigma D\nu)_g-\rho_b}{\rho_b}\right)^3
=\Delta v\sum_gA_s(\bar\rho_g-\rho_b)^3,
\]

with a Gaussian coarse graining

\[
K_\sigma(\mathbf r)
=(2\pi\sigma_s^2)^{-3/2}
\exp\!\left(-\frac{|\mathbf r|^2}{2\sigma_s^2}\right).
\]

Its pure-liquid number-channel compressibility fixes the dimensionless cubic
coefficient rather than a solute benchmark:

\[
S_{NN}(0)=\rho_b k_BT\chi_T,
\qquad
\boxed{a_s=\frac12\left(1+\frac{1}{S_{NN}(0)}\right).}
\]

For the exact molecular-HNC zero mode, this supplies a second derivation of
the vacuum pressure,

\[
P_{\rm HNC}=\rho_bk_BT a_s,
\qquad
\boxed{
A_s=\frac{P_{\rm HNC}-P_s}{\rho_b^3}
=\frac{\rho_bk_BT a_s-P_s}{\rho_b^3}.
}
\]

At liquid--vapour coexistence \(P_s=0\), this reduces to the literature
cubic-WDA expression.  For a declared finite physical pressure, the two
forms must still agree; otherwise the scalar's actual HNC zero mode and the
independently sourced compressibility describe different liquids and the
candidate fails closed.  The finite-grid implementation uses a positive
periodic Gaussian image sum and normalizes it in the same \(\Delta v\)
pairing as the HNC convolution.  Its only local derivatives are

\[
b'(\bar\rho)=3A_s(\bar\rho-\rho_b)^2,
\qquad
b''(\bar\rho)=6A_s(\bar\rho-\rho_b),
\]

so the same \(D^TK_\sigma\) gradient and self-adjoint Hessian proof in
Section 4.2.19 applies without response symmetrization.

`Route2V0MolecularWeightedDensityBridgeAsset.from_molecular_cubic_wda_anchors`
implements this branch.  It rejects a non-Gaussian live kernel, a nonzero
quartic coefficient, an absent compressibility, or disagreement between the
compressibility zero mode and the exact molecular-HNC pressure.  It therefore
is a real common-scalar control, but **not yet a physical liquid endpoint**:
the present direct constructor has no parsed source-bound cubic-WDA
certificate, and cannot pass `require_physical_pure_solvent_asset()`.

Physical admission must later bind, before any target-solute result, one exact
all-atom solvent model, finite-wavevector/orientational correlation,
independent \(\chi_T\), \(P_s\), and \(\gamma_s\), a stationary planar
surface-tension determination of \(\sigma_s\), and the content digests of
the Gaussian kernel and centre projection.  The one-component
compressibility number alone still cannot reconstruct the molecular
direct-correlation tensor.  No MNSol, FreeSolv, development, confirmation, or
blind solvation value may select \(\sigma_s\), \(A_s\), or the underlying
liquid source.

The 2021 paper also explored a separate solute--water electrostatic bridge
whose coefficient was set from the bulk-water excess chemical potential.  It
is **not** silently adopted here: it is a different solute-density-coupled
functional term, and the paper itself reports limitations for ionic solvation.
V0 needs a distinct source-bound excess-chemical-potential and multi-solvent
falsification contract before that term can enter a physical endpoint.  This
keeps the current branch free of a disguised response correction.

The paper's reported comparison is against simulations with its own frozen
water model and geometry convention.  It is useful evidence that the
pure-liquid cubic-WDA structure is worth testing, but it is not evidence that
Route-2 has met the immutable historical ten-record maximum-error gate, the
11-solvent gate, or the independent experimental blind gate.

#### 4.2.20 Unified no-training V0 variational and stability theorem

The preceding modules form one admissible mathematical trunk only when they
are read as derivatives and decompositions of the **same** frozen scalar.  For
one hash-bound solvent asset, fixed Cartesian grid, rigid-configuration
quadrature with positive weights \(w_i\), projection \(P\), and zero-field MACE
molecular external potential, define

\[
\boxed{
\Omega_{\rm V0,HNC}[\nu;\mathbf R]
=
\Omega_{\rm id}
[\nu;u_{\rm MACE}(\mathbf R,\Gamma)]
+F_{\rm ex}^{\rm RISM}[P\nu].
}
\]

The source files and their certificates are admissibility metadata for this
definition; they are not additional energy terms.  Here
\(\nu_i>0\) is the molecular configuration density and
\(\nu_b=\rho_b/(8\pi^2)\) is its uniform bulk value, with
\(n_b=P\nu_b\) the corresponding site-density field.  The projection and its
adjoint use the exact discrete pairing
\(\Delta V\langle v,Pd\rangle_{\rm site}
=\sum_i w_i\,(P^Tv)_i\,d_i\).  With \(\delta n=P\nu-n_b\), the source-`SMEAR`
decomposition is

\[
F_{\rm ex}^{\rm RISM}
=-\frac{k_BT}{2}\Delta V
\sum_{ag}\delta n_{ag}(C^{\rm sr}*\delta n)_{ag}
+\frac{k_BT}{2}\Delta V
\sum_g\rho_Q(\mathbf r_g)V_Q(\mathbf r_g),
\qquad
\rho_Q=\sum_a q_a\delta n_a .
\]

The exact configuration derivative is therefore

\[
\boxed{
\frac{\beta}{w_i}
\frac{\partial\Omega_{\rm V0,HNC}}{\partial\nu_i}
=
\log\!\frac{\nu_i}{\nu_b}
+\beta u_{{\rm MACE},i}
-\left[P^TC_{\rm RISM}(P\nu-n_b)\right]_i
=0.
}
\]

This equation, rather than a mixing update, defines the liquid state.  At a
verified stationary solution \(\nu^*(\mathbf R)\), the envelope theorem gives

\[
\boxed{
\frac{d\Omega_{\rm V0,HNC}[\nu^*(\mathbf R);\mathbf R]}{dR_I}
=
\sum_i w_i\nu_i^*
\frac{\partial u_{{\rm MACE},i}}{\partial R_I}.
}
\]

Equivalently, when
\(f^{\rm MACE}_{I,i}=-\partial u_{{\rm MACE},i}/\partial R_I\), the nuclear
force is
\[
\mathbf F_I
=\sum_iw_i\nu_i^*\mathbf f^{\rm MACE}_{I,i}.
\]

No \(d\nu^*/dR_I\) term may be restored as a separate correction: its
coefficient is precisely the stationary residual.  Conversely, any
coordinate-dependent grid, cavity, quadrature, projection, or solvent
functional must contribute its explicit derivative to this same scalar before
a force or PES claim is permitted.

Stationarity is necessary but not sufficient.  On the declared admissible
tangent space \(\mathcal T\), the same scalar fixes

\[
\boxed{
\delta^2\Omega_{\rm V0,HNC}[d,d]
=k_BT\sum_iw_i d_i
\left[
\frac{d_i}{\nu_i}
-\left(P^TC_{\rm RISM}Pd\right)_i
\right],
\qquad d\in\mathcal T .
}
\]

The V0 state is admitted as a locally stable minimum only when this quadratic
form is positive for every nonzero allowed tangent and the Hessian pairing is
reciprocal.  Residual convergence, a favorable Picard spectral radius, or a
chosen mixing coefficient cannot replace this thermodynamic gate.

The weighted-density extension is not a separate ledger.  It replaces the
base scalar by

\[
\Omega_{\rm V0,B}[\nu;\mathbf R]
=\Omega_{\rm V0,HNC}[\nu;\mathbf R]+F_{\rm B}[D\nu],
\]

and replaces the HNC gradient and Hessian by the exact additions in Section
4.2.19.  In particular, for $d\in\mathcal T$,

\[
\delta^2\Omega_{\rm V0,B}[d,d]
=\delta^2\Omega_{\rm V0,HNC}[d,d]
+\Delta v\sum_g
\left[KDd\right]_g^2 b''\!\left([KD\nu]_g\right).
\]

The final term is the scalar second variation of the bridge, so it cannot be
replaced by a response filter or an error-selected post-correction.  The
same envelope-force statement holds only while $D,K,A_s,B_s$, and their
source assets are fixed with respect to the solute coordinate; otherwise the
explicit derivative of each coordinate-dependent term belongs in the common
scalar force.

This theorem is deliberately conditional.  It proves the internal
common-energy, envelope-force, and stability identities of a frozen
no-training construction.  It does not prove that a source-complete candidate
is a production liquid asset, that the 11-solvent panel exists, or that any
experimental maximum-error target has been met.  The checked-in bridge exposes
the exact Hessian action and quadratic form.  Its
``stationary_hessian_stability_certificate`` first rechecks every frozen asset
hash, the exact bridge projection, and the scalar state/residual before it
invokes this controlled diagnostic.  A physical asset still needs a
grid/orientation-converged minimum-eigenvalue certificate before this stability
gate is considered passed.

#### 4.2.21 Controlled finite-dimensional Hessian certificate

For a declared positive configuration quadrature W = diag(w_i) and the exact
dimensionless Hessian action H already exposed by the molecular HNC or
HNC-plus-bridge scalar, the small-grid diagnostic in
route2_v0_molecular_stability.py assembles the weighted matrix

A = W^(1/2) H W^(-1/2).

Because H maps a density direction to a dimensionless gradient, A and its
spectrum have Bohr^3 units. Accordingly, the numeric 1 in the tolerance below
means 1 Bohr^3 in this atomic-unit representation.

The current molecular liquid scalar is grand-canonical, so this controlled
certificate acts on the full configuration-density space. A future canonical
fixed-number branch must register an explicit constrained tangent projector; it
must not silently discard a mode here.

For x = W^(1/2) d, the same scalar has second variation
kBT * x.T A x. The diagnostic therefore first measures the raw reciprocity
residual

rho_A = ||A - A.T||_F / max(1, ||A||_F).

It rejects the action when rho_A is larger than 128 * eps_machine * n. It does
not symmetrise A before this gate. Only after that rejection test does it use
the raw accepted matrix with eigvalsh and classify its minimum eigenvalue using

tau_A = 128 * eps_machine * n * max(1, ||A||_2).

A minimum eigenvalue above tau_A is positive-definite on that declared finite
grid; one below -tau_A records a negative mode; the remaining interval is
numerically singular. The matrix, spectrum, weights, density, raw
antisymmetric residual (Bohr^3), and two numerical tolerances are frozen in
the returned record. Eigenvalue clipping, response damping, density projection,
and selection from a solvation error are not available operations.

This is deliberately capped at 256 configuration degrees of freedom. It is a
controlled-discretisation certificate, not a production-grid claim: before a
physical liquid state passes the stability gate, it still needs an independently
verified stationary residual, source-complete liquid asset, and
grid/orientation-converged minimum-eigenvalue evidence. The diagnostic does
not read the field-conditioned MACE response or introduce post-training,
fine-tuning, experimental fitting, or calibration.

#### 4.2.22 Deterministic full-\(SO(3)\) orientation quadrature

The molecular density is a function of the full rigid orientation, not a
single preferred solvent pose.  To make the existing unnormalised convention

\[
\int_{SO(3)}d\Omega=8\pi^2
\]

executable without a random sample, V0 now fixes the Euler parameterisation

\[
R(\alpha,x,\gamma)
=R_z(\alpha)R_y(\arccos x)R_z(\gamma),
\qquad
\alpha,\gamma\in[0,2\pi),\quad x\in[-1,1],
\]

for which the Haar integral is

\[
\int_{SO(3)}f(R)d\Omega
=\int_0^{2\pi}d\alpha
 \int_{-1}^{1}dx
 \int_0^{2\pi}d\gamma\;f(R(\alpha,x,\gamma)).
\]

For a declared integer \(n\), the `Route2V0EulerSO3Quadrature` rule uses the
\(n\) Gauss--Legendre nodes \((x_p,w_p)\) and \(2n\) equally spaced values of
each periodic angle.  Its \(4n^3\) orientations have weights

\[
\omega_{pab}=w_p\left(\frac{2\pi}{2n}\right)^2,
\qquad
\sum_{pab}\omega_{pab}=8\pi^2.
\]

The product rule has the documented Wigner-rank bandlimit \(2n-1\); this is
the rigid-molecular Euler/Legendre construction used by Sundararaman and Arias
for molecular classical DFT.  The newer orientation-average analysis by Blech
*et al.* likewise emphasizes that quadrature choice and convergence must be
resolved for the interaction being averaged, rather than inferred from one
convenient orientation.  [Sundararaman and Arias, *Comput. Phys. Commun.*
**185**, 818 (2014)](https://doi.org/10.1016/j.cpc.2013.11.013), [Blech *et
al.*, *J. Chem. Phys.* **161**, 131501 (2024)](https://doi.org/10.1063/5.0230569).

`Route2V0CartesianEulerProductQuadrature` takes its exact Cartesian product
with the declared periodic grid, so every configuration weight is
\(\Delta V\omega_{pab}\) and the total measure is
\(V\,8\pi^2\).  The implementation verifies proper rotations, positive weights,
and the exact phase-space convention consumed by the existing ideal/HNC scalar;
its nontrivial-order regression controls verify the first and second Haar
moments.  It neither
uses a random seed nor applies an automatic water or solvent point-group
quotient: a symmetry reduction requires a separately frozen proof that the
specific molecular source and all energy terms share that symmetry.

This is a refinement construction, not a convergence assertion.  Before any
physical target-solute calculation, the frozen source asset, closure, solute
set, and standard-state ledger must be held fixed while independently refining
both \(n\) and the Cartesian grid.  The stationary free energy, site-density
observables, minimum-curvature evidence, and any permitted envelope force must
converge across that preregistered sequence and across rigid solute
orientations.  No orientation order, symmetry quotient, or stopping threshold
may be selected from a solvation error.  The current explicit product is a
controlled reference representation; production-scale use still requires a
separately validated matrix-free implementation that preserves the same
quadrature and adjoint identities.


### 4.3 Separate auxiliary-QM liquid-difference route

V0-FD deliberately freezes the MACE source and varies only the solvent.  A
separate no-training route, V0-AQ, retains the MACE gas potential but obtains
an **auxiliary** electronic response from a joint electronic--liquid scalar
and adds only its gas-to-liquid difference.  It does not relabel the auxiliary
density as MACE density, repair the legacy MACE field map, or bypass the
physical liquid-asset gate.  Its exact subtraction ledger, stationary
electronic-continuum control, force boundary, and liquid-functional admission
requirements are defined in
[`ROUTE2_V0_AQ_THEORY.md`](ROUTE2_V0_AQ_THEORY.md).  V0-AQ-E remains a
fixed-geometry electronic-continuum control; a physical V0-AQ-L total free
energy still requires one frozen molecular-liquid functional and its solvent
assets.

### 4.4 Explicit exclusions

- Selecting IEFPCM/CPCM/COSMO per record is forbidden.  The existing same-
  source ten-solvent panel found only small equation differences and does not
  establish a chemical-accuracy winner.
- COSMO-RS/openCOSMO-RS remains an external comparator: its current parameter
  family was fitted to solvation data and cannot certify a no-fit V0-FD core.
- Target-observed or response-output charge projection, response
  symmetrization, eigenvalue clipping, response scaling, cavity-radius tuning,
  and error regressions are not physical candidate families.  The fixed
  source-rounding neutrality projection in Section 4.2.9 is a preregistered
  algebraic zero-mode constraint and is not selected from a solvation result.
- Post-training and fine-tuning are deferred outside V0.  A future learned
  electronic-functional head must be a separately preregistered route, not a
  fallback used to rescue a failed V0 structural or benchmark gate.

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

### 5.1 Mandatory recovery of the historical \(7.0414\) kcal/mol outlier panel

The later ten-record MNSol and ten-record exact-GTO FreeSolv diagnostics do
**not** have the same membership as the original ten-class FreeSolv screen.
That original screen's fixed GTO/QEq--GBn2 member reached
\(7.041442082076966\) kcal/mol absolute error on ethyl acetate
(`mobley_6973347`).  Its retired GTO/QEq/GBn2 calculation is not a V0
candidate or a parameter source, but its ten experimental records and exact
FreeSolv MOL2 conformers are an irreplaceable regression set.  They are frozen
with the historical Git commit/blob/content hashes in
[`route2-v0-historical-freesolv10-regression-v1.json`](benchmarks/route2-v0-historical-freesolv10-regression-v1.json).

A candidate V0 result must run the validator
[`route2_v0_historical_freesolv10.py`](benchmarks/route2_v0_historical_freesolv10.py)
on **all and only** these ten records.  It passes that gate only if every
recomputed absolute error is strictly below \(1.5\) kcal/mol.  The validator
rejects a missing, extra, duplicate, non-finite, or substituted record and
recomputes every error from the locked experimental value; a reported MAE,
RMSE, or a favorable subset can never override an outlier.  This water
regression is mandatory in addition to—not instead of—the 11-solvent,
confirmation, and blind panels.  Passing it alone remains insufficient for a
physical or broad-accuracy claim.

The required progression is therefore:

1. pass structural energy, charge, pairing, and smoothness gates;
2. pass the immutable historical ten-record FreeSolv regression, including the
   ethyl-acetate historical outlier, with every record strictly below
   \(1.5\) kcal/mol;
3. freeze a replicated **11-solvent** development design before looking at
   V0-FD scores.  The current MNSol-v2012 neutral-absolute partition has only
   ten solvent strata because it has no neutral absolute methanol rows in the
   frozen Route-2 domain; a separately versioned methanol extension is a
   prerequisite, not a gap to hide behind aggregate metrics;
4. require every development record to satisfy
   \(|\Delta G_{\mathrm{calc}}-\Delta G_{\mathrm{exp}}|<1.5\) kcal/mol;
5. freeze a disjoint multi-solvent confirmation set and impose the same
   per-record condition without changing the method;
6. freeze an external final blind dataset absent from construction,
   development, and confirmation; require the same condition there;
7. treat all-record \(<1.0\) kcal/mol as the stricter final target.

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
11. PySCF developers, *Atomic Hartree--Fock module*,
    [PySCF API documentation](https://pyscf.org/pyscf_api_docs/pyscf.scf.html).
    `AtomSphAverageRHF` supplies the declared spherical fractional-occupation
    isolated-atom construction used solely for the frozen reference table.
12. T. A. Manz and D. S. Sholl, *Polarized Protein-Specific Charges from
    Atoms-in-Molecule Electron Density Partitioning*,
    [PMC3719162](https://pmc.ncbi.nlm.nih.gov/articles/PMC3719162/).  It
    describes the promolecular density as a superposition of reference atomic
    densities; that convention does not make it an actual molecular density.
13. L. Ding, M. Levesque, D. Borgis, and L. Belloni, *Efficient molecular
    density functional theory using generalized spherical harmonics
    expansions*, *J. Chem. Phys.* **147**, 094107 (2017),
    [arXiv:1707.01385](https://arxiv.org/abs/1707.01385).  It states the HNC
    density functional and its variational minimization; the reduced
    site-only reference here is not claimed to reproduce its full molecular
    orientational model.
14. J. Johnson *et al.*, *Small Molecule Hydration Energy and Entropy from
    3D-RISM*, [PMC5118872](https://pmc.ncbi.nlm.nih.gov/articles/PMC5118872/).
    It documents the large HNC pressure contribution motivating a separately
    declared pressure convention rather than an error-selected correction.
15. D. A. Case *et al.*, *Amber 2021 Reference Manual*,
    [PDF](https://supercrispr.github.io/file/Amber/manual_21.pdf).  Its RISM
    file-format section identifies `.xvv` as the reusable bulk susceptibility
    asset and documents the radial site--site correlation outputs; this does
    not authorize its solute `prmtop` workflow for Route-2 V0-FD-S.
16. A. P. Lyubartsev *et al.*, *Simple electrolyte solutions: Comparison of
    DRISM and molecular dynamics results for alkali halide solutions*,
    [PMC3568087](https://pmc.ncbi.nlm.nih.gov/articles/PMC3568087/).  Its
    long-range appendix writes the direct-correlation Coulomb asymptote as
    \(-\beta u\), motivating the analytic tail split rather than a finite-box
    radial wrap.
17. T. A. Wesolowski, *Frozen-Density Embedding Strategy for Multilevel
    Simulations of Electronic Structure*, *Chem. Rev.* **115**, 5891--5928
    (2015), [DOI:10.1021/cr500502v](https://doi.org/10.1021/cr500502v).
    It supplies the frozen-density embedding boundary; the present
    Thomas--Fermi term remains a deliberately limited frozen-density control.
18. S. Zhao, R. Ramirez, R. Vuilleumier, and D. Borgis, *Molecular density
    functional theory of solvation: from polar solvents to water*, *J. Chem.
    Phys.* **134**, 194102 (2011),
    [PMID:21599039](https://pubmed.ncbi.nlm.nih.gov/21599039/).  Its
    position-and-orientation dependent molecular functional motivates the
    configuration convention above, not a claim that the present direct
    quadrature control is a molecular-liquid free-energy calculation.
19. R. Sundararaman and T. A. Arias, *Efficient classical density-functional
    theories of rigid-molecular fluids and a simplified free energy functional
    for liquid water*, [arXiv:1302.0026](https://arxiv.org/abs/1302.0026).
    It motivates a configuration-space formulation for rigid molecular fluids;
    the present ideal term is only the exact common starting point before a
    source-provenanced excess functional is admitted.
20. V. P. Sergiievskyi, G. Jeanmairet, M. Levesque, and D. Borgis, *Fast
    Computation of Solvation Free Energies with Molecular Density Functional
    Theory: Thermodynamic-Ensemble Partial Molar Volume Corrections*,
    *J. Phys. Chem. Lett.* **5**, 1935--1942 (2014),
    [DOI:10.1021/jz500428s](https://doi.org/10.1021/jz500428s).
    It derives the molecular-HRF pressure from the vacuum functional and the
    particle-deficit partial molar volume used in Section 4.2.18.
21. V. P. Sergiievskyi, G. Jeanmairet, M. Levesque, and D. Borgis,
    *Solvation free-energy pressure corrections in the Three Dimensional
    Reference Interaction Site Model*, *J. Chem. Phys.* **143**, 184116
    (2015), [DOI:10.1063/1.4935065](https://doi.org/10.1063/1.4935065).
    It derives the different site-3D-RISM pressure, showing why functional
    identity matters, and distinguishes the rigorous macroscopic PC term from
    the additional microscopic PC+ adjustment.
22. C. Gageat, L. Belloni, D. Borgis, and M. Levesque, *Bridge functional for
    the molecular density functional theory with consistent pressure and
    surface tension and its importance for solvation in water*,
    [arXiv:1709.10139](https://arxiv.org/abs/1709.10139).  It motivates a
    coarse-grained weighted-density bridge inserted before minimisation,
    derives a cubic coexistence constraint from the HNC pressure, and anchors
    the remaining barrier against a pure-liquid surface tension rather than a
    molecular solvation-error regression.
23. G. Jeanmairet, *A molecular density functional theory to study solvation in
    water*, [arXiv:1408.7008](https://arxiv.org/abs/1408.7008).  It states the
    pure-solvent direct-correlation requirement of a molecular density
    functional; macroscopic dielectric data are not a replacement for that
    input.
24. JDFTx developers, [fluid](https://jdftx.org/CommandFluid.html) and
    [fluid-solvent](https://jdftx.org/CommandFluidSolvent.html) documentation.
    The interface distinguishes linear/nonlinear dielectric modes from a
    `ClassicalDFT` fluid with an explicitly selected excess functional,
    matching the V0 separation between an electrostatic control and a
    molecular-liquid endpoint.
25. D. Borgis, S. Luukkonen, L. Belloni, and G. Jeanmairet, *Simple
    Parameter-Free Bridge Functionals for Molecular Density Functional Theory.
    Application to Hydrophobic Solvation*, *J. Phys. Chem. B* **124**,
    6885--6893 (2020),
    [DOI:10.1021/acs.jpcb.0c04496](https://doi.org/10.1021/acs.jpcb.0c04496).
    Its weighted-density bridge fixes the water variables from pure-liquid
    pressure, compressibility, and liquid--gas surface tension rather than
    solvation labels; it is a water-model precedent, not a V0 accuracy claim.
26. D. Borgis, S. Luukkonen, L. Belloni, and G. Jeanmairet, *Accurate
    Prediction of Hydration Free Energies and Solvation Structures Using
    Molecular Density Functional Theory with a Simple Bridge Functional*,
    *J. Chem. Phys.* **155**, 024117 (2021),
    [DOI:10.1063/5.0057506](https://doi.org/10.1063/5.0057506).
    Its hydration comparisons are against a specific frozen water model and
    geometry convention; they do not establish the required multi-solvent
    experimental maximum-error gate for Route-2 V0.
27. G. Jeanmairet, M. Levesque, and D. Borgis, *Tackling Solvent Effects by
    Coupling Electronic and Molecular Density Functional Theory*, *J. Chem.
    Theory Comput.* **16**, 7123--7134 (2020),
    [DOI:10.1021/acs.jctc.0c00729](https://doi.org/10.1021/acs.jctc.0c00729).
    Its jointly stationary electronic and molecular densities illustrate the
    common-functional direction, but cannot be attributed to the frozen,
    nonvariational MACE response retained in V0.
28. S. M. Kast, *Free Energies from Integral Equation Theories: Enforcing
    Path Independence*, *Phys. Rev. E* **67**, 041203 (2003),
    [DOI:10.1103/PhysRevE.67.041203](https://doi.org/10.1103/PhysRevE.67.041203).
    It distinguishes closure-consistent free-energy expressions; it does not
    authorize inserting a PSE-\(n\) correlation into the HNC scalar above.
