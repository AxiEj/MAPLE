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
parser control, not one of the required eleven frozen solvent assets.

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
model and not a trainable correction.  `route2_v0_solvent_asset.py` accepts a
registry only when each solvent manifest hash-locks all of the following:

- the molecular site-model source and the bulk 1D-RISM input;
- matching `.xvv` and `.cvv` files, which are reparsed using the native-QV,
  source-`SMEAR` checks above;
- the thermodynamic-output source and the separately sourced short-range
  interaction record;
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

[`route2-v0-solvent-asset-inventory-v1.json`](benchmarks/route2-v0-solvent-asset-inventory-v1.json)
records the current deliberately incomplete state: the local cSPC/E data is a
bulk/parser control, while **zero** physical default-solvent assets are
registered.  Consequently total-free-energy execution remains rejected.  This
new registry is provenance infrastructure only: it neither defines the
MACE-native \(u^{\mathrm{sr}}\), maps a production short-range kernel,
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
- Post-hoc charge projection, response symmetrization, eigenvalue clipping,
  response scaling, cavity-radius tuning, and error regressions are not
  physical candidate families.
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
