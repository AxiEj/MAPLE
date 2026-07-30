# Route-2 V0-AQ: auxiliary quantum--liquid variational-difference contract

## Status and scope

V0-AQ is a no-training **reference architecture** for restoring a stationary
electronic response without claiming that the frozen MACE-POLAR field update
is an energy gradient.  It leaves the gas MACE model untouched and introduces
a separate, declared auxiliary electronic--liquid calculation only through a
gas-to-liquid free-energy difference.  If that auxiliary state requires an
SCF/QM calculation at every geometry, it is not a production Route-2 speed
endpoint; it is an offline physical oracle and a specification for a future
pretrained variational electronic model.

It is not a public Route-2 profile, a replacement for V0-FD, a total-solvation
result, a force/PES implementation, an accuracy result, or a justification to
ship a QM-speed method as an MLIP.  In particular, the first V0-AQ-E control
below is a fixed-geometry electronic-continuum control: it contains neither
an empirical SMD CDS term nor a complete liquid non-electrostatic free energy.
It cannot be compared with experimental solvation free energies.

The machine-readable commitment is
[`route2-v0-auxiliary-qm-liquid-prereg-v1.json`](benchmarks/route2-v0-auxiliary-qm-liquid-prereg-v1.json).

## 1. One scalar ledger without changing the MACE gas branch

At a nuclear geometry \(\mathbf R\), retain the unmodified MACE gas energy

\[
E_{\mathrm{MACE,gas}}(\mathbf R).
\]

Let \(n\) denote an **auxiliary** electronic density.  It is not a MACE
density, is not projected from MACE coefficients, and never drives a
field-conditioned MACE inference.  The corresponding auxiliary gas reference
is

\[
A_{\mathrm{aux}}^{\mathrm{gas}}(\mathbf R)
=\min_{n\in\mathcal N_e} A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R].
\]

For a physical liquid state with solvent degrees of freedom \(N\), define one
joint scalar

\[
A_{\mathrm{aux}}^{\mathrm{sol}}(\mathbf R;s)
=\operatorname*{stat}_{n,N,\lambda}
\left[
 A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R]
+\Phi_s[N]
+U_s[n,N;\mathbf R]
+\lambda\,C[n,N]
\right].
\]

The only admissible composite ledger is therefore

\[
\boxed{
G_{\mathrm{V0\text{-}AQ}}(\mathbf R;s)
=E_{\mathrm{MACE,gas}}(\mathbf R)
+\left[A_{\mathrm{aux}}^{\mathrm{sol}}(\mathbf R;s)
-A_{\mathrm{aux}}^{\mathrm{gas}}(\mathbf R)\right]
+\Delta G_s^\circ .
}
\]

The gas subtraction is mandatory.  Adding a solvated auxiliary electronic
energy directly to \(E_{\mathrm{MACE,gas}}\) would count a second gas solute
energy; adding a PCM half-coupling or a CDS term outside this same auxiliary
difference would create a second, nonconjugate ledger.  \(\Delta G_s^\circ\)
is permitted only when it is part of a declared, physical liquid
standard-state convention.  It is absent from the V0-AQ-E control.

This construction is deliberately an *auxiliary solvation difference*, not a
claim that MACE and the auxiliary quantum method describe the same electronic
state.  It preserves the MACE gas potential exactly while making the added
response the derivative of a scalar by construction.

## 2. The first bounded control: V0-AQ-E

V0-AQ-E uses a self-consistent auxiliary QM--PCM scalar only to test the
electronic/continuum stationarity boundary at fixed geometry.  It is **not a
total solvation free energy**.  For an
energy-conjugate PCM discretization with apparent surface charge \(\sigma\),
one representative Lagrangian is

\[
\mathcal L_{\mathrm{aux,PCM}}[n,\sigma]
=A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R]
+\frac12\sigma^\mathsf T A_{\mathbf R}\sigma
+\sigma^\mathsf T B_{\mathbf R}n.
\]

Its stationary equations are

\[
\frac{\delta A_{\mathrm{aux}}^{\mathrm{gas}}}{\delta n}
+B_{\mathbf R}^\mathsf T\sigma=0,
\qquad
A_{\mathbf R}\sigma+B_{\mathbf R}n=0.
\]

The correct V0-AQ-E correction is the difference between the fully
self-consistent stationary total energies,

\[
\Delta A_{\mathrm{aux}}^{\mathrm{PCM}}
=A_{\mathrm{aux}}^{\mathrm{PCM\text{-}SCF}}
-A_{\mathrm{aux}}^{\mathrm{gas\text{-}SCF}},
\]

not an isolated implementation-specific ``PCM energy`` field and not a
frozen-density half-coupling estimate.  Thus the electron density and surface
charge are both stationary in the same scalar.  This directly avoids the
nonreciprocal learned fixed-point response diagnosed for legacy Route 2.

V0-AQ-E has the following hard limits:

* it is an electrostatic fixed-geometry control, not a total solvation free
  energy or a solvent-ranking calculation;
* it does not use SMD radii, SMD CDS, target solvation data, or any
  post-result radius/scale selection;
* a finite tessellation may still be unsuitable for a production PES.  An
  analytic gradient at an isolated geometry does not certify smooth
  rotation/translation, scan, optimization, MD, or NVE behavior;
* no auxiliary density may be relabelled as a MACE density or be sent into a
  MACE field-feature interface.

The first execution is deliberately a source-bound water control, frozen in
[`route2-v0-aq-water-pcm-prereg-v1.json`](benchmarks/route2-v0-aq-water-pcm-prereg-v1.json)
and run by
[`run_route2_v0_auxiliary_qm_pcm_water.py`](benchmarks/run_route2_v0_auxiliary_qm_pcm_water.py).
It records the gas and PCM-SCF **total** stationary energies and their
analytic nuclear gradients, then takes their difference.  It does not call
MACE, inspect a solvation label, or certify a force/PES; the separate V0-AQ
ledger owns the future addition to an unchanged MACE gas term.

The exact frozen water control was rejected before its PCM-SCF phase because
the gas auxiliary-QM translation-gradient gate failed.  The failure record
[`route2-v0-aq-water-pcm-control-v1.json`](benchmarks/route2-v0-aq-water-pcm-control-v1.json)
binds the original runner, geometry, runtime, threshold, and fail-closed
error.  No grid, radius, tolerance, score, or MACE component was changed to
rescue it.  This rejects that finite-grid control; it neither establishes a
MACE-response defect nor validates or invalidates the physical V0-AQ-L branch.

## 3. Main implicit-continuum completion: V0-AQ-C

The V0 main line does **not** require an explicit molecular-liquid trajectory,
GROMACS, 3D-RISM, or MDFT.  The required replacement for the rejected learned
fixed point can instead be a diffuse, variational continuum coupled to the
auxiliary electronic state.  This is called V0-AQ-C below.  It is a mathematical
architecture, not a claim that the present code already supplies all of its
physical solvent inputs.

Let \(m(\mathbf r)\in[0,1]\) be a smooth solvent-occupancy field, with
\(m=0\) inside the solute and \(m=1\) in bulk solvent.  Let
\(\rho[n;\mathbf R]\) be the total auxiliary charge density and define

\[
\mathscr E_{\epsilon}[\rho,\phi]
=-\frac{1}{8\pi}\int
\epsilon(\mathbf r)|\nabla\phi(\mathbf r)|^2\,d\mathbf r
+\int\rho(\mathbf r)\phi(\mathbf r)\,d\mathbf r.
\]

The sign is intentional: \(\phi\) is a stationary saddle variable.  Its
Euler equation is

\[
-\nabla\!\cdot\!\bigl[\epsilon(\mathbf r)\nabla\phi(\mathbf r)\bigr]
=4\pi\rho(\mathbf r),
\]

and at the stationary solution
\(\mathscr E_{\epsilon}^*=\tfrac12\int\rho\phi\).  Because the auxiliary
gas functional already contains vacuum Coulomb energy, the continuum must add
**the reaction difference**, rather than a second total electrostatic energy.
A valid one-scalar formulation is therefore

\[
\begin{aligned}
\mathcal L_{s}^{\rm C}[n,m,\phi_s,\phi_0;\mathbf R]
={}&A_{\rm aux}^{\rm gas}[n;\mathbf R]
+\mathscr E_{\epsilon_s(m)}[\rho[n;\mathbf R],\phi_s]
-\mathscr E_{1}[\rho[n;\mathbf R],\phi_0]\\
&+\Phi_s[m;n,\mathbf R],\\
\epsilon_s(m)={}&1+(\epsilon_{s,0}-1)m.
\end{aligned}
\]

Here \(\phi_0\) is not an independent physical gas calculation: it is the
vacuum stationary field required to subtract the vacuum Coulomb part already
present in \(A_{\rm aux}^{\rm gas}\).  The electronic/continuum correction is

\[
\Delta A_{\rm aux,s}^{\rm C}(\mathbf R)
=\operatorname*{stat}_{n,m,\phi_s,\phi_0}
\mathcal L_s^{\rm C}
-\min_n A_{\rm aux}^{\rm gas}[n;\mathbf R].
\]

The Route-2 ledger remains

\[
G_{\rm V0-AQ-C}(\mathbf R;s)
=E_{\rm MACE,gas}(\mathbf R)
+\Delta A_{\rm aux,s}^{\rm C}(\mathbf R)
+\Delta G_s^\circ.
\]

Thus MACE remains the untouched gas potential, while the added solvent
response is a stationary scalar.  Neither \(n\) nor \(m\) is a MACE density,
and neither is allowed to enter MACE field features.

### 3.1 What the solvent functional must contain

\(\Phi_s\) is where a total implicit-solvent model either remains physical or
quietly becomes a fitted correction.  It must contain the *pre-minimisation*
liquid/cavity, short-range repulsion, dispersion, and standard-state physics:

\[
\Phi_s[m;n,\mathbf R]
=\Phi_{\rm bulk,cav,s}[m]
+\Phi_{\rm sr,s}[m;n,\mathbf R]
+\Phi_{\rm disp,s}[m;n,\mathbf R]
+\Phi_{\rm std,s}[m].
\]

A term such as \(pV+\gamma A+\kappa C+\bar\kappa X\) is permissible only as
a declared approximation to \(\Phi_{\rm bulk,cav,s}\), with its dividing
surface and every coefficient sourced from a pure-liquid model or independent
bulk/interfacial measurement **before** target solvation values are read.
Appending a surface-area correction, a dispersion scale, SMD-CDS, or a
standard-state offset after solving the electrostatic state breaks this
contract.  For molecular-sized cavities, omitting curvature or dispersion is
an approximation to be tested, not a justification for target scoring.

The stationary equations make the coupling explicit:

\[
\frac{\delta\mathcal L_s^{\rm C}}{\delta n}=0,\qquad
\frac{\delta\mathcal L_s^{\rm C}}{\delta m}=0,\qquad
\frac{\delta\mathcal L_s^{\rm C}}{\delta\phi_s}=0,\qquad
\frac{\delta\mathcal L_s^{\rm C}}{\delta\phi_0}=0.
\]

In particular, \(\delta\mathcal L/\delta m\) contains the dielectric-field
term \(-\frac{\epsilon_{s,0}-1}{8\pi}|\nabla\phi_s|^2\) and the derivative of
the same nonpolar functional.  The cavity cannot be moved independently after
an electrostatic result has been observed.

### 3.2 Custom-solvent contract

A scalar dielectric constant is enough only for a *linear-electrostatic
control*.  A custom **total** V0-AQ-C solvent needs a frozen provenance record
for at least

\[
(T,p,\rho_{\rm bulk},\epsilon_0,\epsilon_\infty,
\chi_s(k)\ \text{or a declared local limit},
\Phi_{\rm bulk,cav,s},\Phi_{\rm sr,s},\Phi_{\rm disp,s},
\Delta G_s^\circ).
\]

The input may be supplied through independently measured bulk/interfacial
properties or an independently parameterised pure-solvent theory, but never
by changing a cavity radius, surface coefficient, dispersion scale, or
finite-\(k\) length after inspecting a solvation error.  This is the strict
version of the user-facing “Gaussian-style custom solvent” requirement: users
may specify physical solvent properties, but insufficient properties fail
closed rather than silently selecting a proxy solvent.

### 3.3 Literature boundary and immediate implication

Joint density-functional theory provides the relevant variational template:
it joins electronic and liquid density functionals in one free-energy
principle without fitting solvation data in its basic construction.
SaLSA shows that nonlocal dielectric response can be derived without empirical
*dielectric* parameters, but its published total model still fitted a
solvent-dependent dispersion scale; it may not be imported as a no-fit V0
total endpoint.  The weighted-density cavity work likewise supplies useful
physical cavity ideas, but its published practical model retains a density
threshold and dispersion scale fitted to solvation data.  These are design
references, not licence to copy their fitted constants.

The immediate consequence is narrow: a self-consistent auxiliary QM--PCM
state can be used only as an electronic/continuum **control** until a smooth,
source-complete \(\Phi_s\) is available.  The preserved PBE0/RHF ddCOSMO
preflight record distinguishes numerical stationarity from chemical accuracy;
it does not authorize a FreeSolv or multi-solvent score.

### 3.4 Implemented fixed-occupancy reaction block

[`route2_v0_diffuse_continuum.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_diffuse_continuum.py)
now implements the deliberately narrow electrostatic part of this scalar on a
periodic Cartesian grid.  Given a **supplied** smooth occupancy \(m\), it uses
the same finite-volume operator for both fields,

\[
A_\epsilon\phi_\epsilon=4\pi\rho,\qquad
A_\epsilon=-\nabla_h\!\cdot\!\left(\epsilon_f\nabla_h\right),\qquad
\epsilon_f=\tfrac12(\epsilon_i+\epsilon_j),
\]

and reports only

\[
G_{\rm reac,h}=\tfrac12\Delta V\,\rho^\mathsf T
(\phi_\epsilon-\phi_1).
\]

Its returned reaction potential is the density derivative of that exact
discrete scalar.  Its returned \(m\)-derivative is the envelope derivative of
the same scalar; it contains no response solve or separately chosen force
term.  A non-neutral periodic density fails closed rather than receiving an
unphysical neutralizing background.

The accompanying
[`test_route2_v0_diffuse_continuum.py`](../../tests/solvation/test_route2_v0_diffuse_continuum.py)
locks the structural claims: homogeneous-dielectric scaling against the same
discrete vacuum operator, reciprocal reaction pairings, passive reaction
energy, density finite-difference conjugacy, and occupancy-envelope
finite-difference conjugacy.  This is **not** a total V0-AQ-C calculation:
\(m\) is not yet stationary, there is no auxiliary electronic minimization,
and \(\Phi_s\) is absent.  Accordingly the implementation cannot emit a
solvation free energy, force certificate, or accuracy value.

### 3.4A Open-boundary isolated-source reaction block

The periodic control is deliberately not the direct endpoint for a finite AO
molecular density: it requires neutral charge on the discrete torus and must
reject any finite-box electron-count residual instead of adding an unphysical
background.  The separate
[`ROUTE2_V0_OPEN_BOUNDARY_CONTINUUM_THEORY.md`](ROUTE2_V0_OPEN_BOUNDARY_CONTINUUM_THEORY.md)
and
[`route2_v0_open_diffuse_continuum.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_open_diffuse_continuum.py)
therefore implement a cell-centred zero-Dirichlet-face finite-volume operator.
The matching non-wrapping cubic B-spline nuclear map is in
[`route2_v0_open_bspline.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_open_bspline.py): a nuclear support that reaches the
box edge fails and requires a larger buffer rather than periodic wrapping.
It is symmetric positive definite for any source charge, reports the same
vacuum-subtracted reaction scalar, and includes the required exterior-face
terms in the occupancy envelope derivative.  Its tests prove scaling,
reciprocity, passivity, and both finite-difference identities for a
deliberately non-neutral source, then prove that its electron reaction
potential pulls back to the AO density dual with the matching central
AO-density finite difference.

This removes a **boundary-condition mismatch**, not an accuracy obstacle: the
finite box, AO density count, solvent kernel/cavity, nonpolar scalar, and
stationary electronic functional each remain independent convergence or
physics gates.  In particular, no density is normalised and no periodic
neutralising background is introduced.

### 3.5 Nonlocal density-defined cavity closure

There is a second mathematically valid way to close the cavity without adding
an independently moved radius field: make it a differentiable functional of
the **auxiliary** electron density.  Let \(n_s^{(0)}\) be the spherical,
isolated-electron-density kernel of solvent \(s\), let \(K_s\) denote its
convolution, and let \(\bar q_s>0\) be a frozen electron-density-overlap
threshold.  The iso-density-product closure is

\[
q_s[n](\mathbf r)=(K_sn)(\mathbf r)
=\int n_s^{(0)}(\mathbf r-\mathbf r')n(\mathbf r')\,d\mathbf r',
\qquad
m_s[n](\mathbf r)
=\frac12\operatorname{erfc}
\!\left[\log\frac{q_s[n](\mathbf r)}{\bar q_s}\right].
\]

It is a *composed scalar functional*, not a post-processed cavity.  For the
electrostatic reaction difference,

\[
G_{\rm reac}[n]
=\frac12\int[\rho_{\rm nuc}-n]
\bigl(\phi_{\epsilon(m_s[n])}-\phi_1\bigr)\,d\mathbf r,
\]

the exact chain rule is

\[
\boxed{
\frac{\delta G_{\rm reac}}{\delta n}
=-\phi_{\rm reac}
+K_s^\dagger\!\left[
h'(K_sn)\frac{\delta G_{\rm reac}}{\delta m}
\right],\qquad
h'(q)=-\frac{e^{-[\log(q/\bar q_s)]^2}}{\sqrt\pi\,q},
\quad h'(0)=0.
}
\]

The first term is the ordinary electronic charge response and the second is
the cavity response.  Omitting either one would again give a response that is
not the derivative of the reported scalar.  This closure differs from the
joint \(m\)-stationarity option above only in variable choice: \(m\) is
eliminated analytically as \(m_s[n]\), so its derivative must be retained in
the electronic Euler equation.

[`route2_v0_iso_density_cavity.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_iso_density_cavity.py)
implements this exact periodic-grid composition and the adjoint \(K_s^\dagger\).
Its tests verify the cavity VJP and the *full* central finite-difference
derivative of \(G_{\rm reac}[\rho_{\rm nuc}-n,m_s[n]]\), including the
nonzero cavity chain term.  The test kernel is synthetic by design.  No
physical solvent kernel or threshold is silently supplied, and the object
refuses promotion to a total-solvent asset.

The density-overlap idea is physically motivated by the non-bonded Pauli
overlap construction in SaLSA.  It is admissible here only when the isolated
solvent kernel and \(\bar q_s\) have an independently recorded pre-label
provenance (for example an ab-initio isolated-solvent density plus a frozen
non-bonded-contact rule).  The published SaLSA **dispersion** scale remains
excluded; neither its fitted scale nor its published RMS can become a V0
parameter or accuracy claim.  The recent solvent-aware SCCS work independently
confirms why the nonlocal convolution and its full functional derivative are
needed to prevent unphysical solvent islands, but its published density
thresholds are likewise not imported as defaults.

### 3.5A Exact auxiliary AO-to-grid source/dual bridge

The preceding density functional is useful only if a stationary auxiliary
electronic state can enter it without changing representation by an
untracked interpolation.  Let (D_{mu\nu}) be a real spin-summed AO density
matrix, (S_{mu\nu}) the AO overlap of the **same** electronic method, and
(\chi_\mu(\mathbf r_g)) its AO values at the nodes of a Cartesian grid of
cell volume (\Delta V).  Define the discrete density projection

\[
(\mathcal P D)_g
=n_D(\mathbf r_g)
=\sum_{\mu\nu}
\chi_\mu(\mathbf r_g)D_{\mu\nu}\chi_\nu(\mathbf r_g).
\]

For a grid potential (u_g=\delta\mathcal A/\delta n_g), the unique AO
pullback in the declared trace pairing is

\[
\boxed{
\bigl(\mathcal P^\ast u\bigr)_{\mu\nu}
=\Delta V\sum_g
\chi_\mu(\mathbf r_g)u_g\chi_\nu(\mathbf r_g).
}
\]

It obeys the exact discrete identity

\[
\boxed{
\Delta V\sum_g(\mathcal P D)_g u_g
=\operatorname{Tr}\!\left[D\,\mathcal P^\ast u\right].
}
\]

This is not a numerical convenience: if the electron-density reaction
derivative from Section 3.5 is

\[
u_{\rm reac}
=-\phi_{\rm reac}
+K_s^\dagger\!\left[
h'(K_sn)\frac{\delta G_{\rm reac}}{\delta m}
\right],
\]

then the reaction contribution to the auxiliary AO Euler/Fock matrix must be

\[
F_{\rm reac}=\mathcal P^\ast u_{\rm reac}.
\]

Thus a future stationary auxiliary electronic functional
(A_{\rm aux}^{\rm gas}[D;\mathbf R]) and the V0-AQ-C reaction scalar obey

\[
\frac{\partial}{\partial D}
\left[
A_{\rm aux}^{\rm gas}[D;\mathbf R]
+G_{\rm reac}[\rho_{\rm nuc}-\mathcal P D]
\right]
=\frac{\partial A_{\rm aux}^{\rm gas}}{\partial D}
+\mathcal P^\ast u_{\rm reac},
\]

up to the electronic-method's occupancy/orthonormality constraints.  No
separate field interpolation, density rescaling, or post-hoc reaction Fock
term is admissible.

[`route2_v0_auxiliary_ao_grid.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_auxiliary_ao_grid.py)
implements (\mathcal P), (\mathcal P^\ast), representation fingerprints,
and both AO and finite-grid electron counts.  The count mismatch

\[
\Delta N_h=\Delta V\sum_g n_D(\mathbf r_g)-\operatorname{Tr}[DS]
\]

is reported rather than normalized away; it is a grid/domain convergence gate
for any physical calculation.  Its dedicated test combines this exact
pullback with the full nonlocal-cavity reaction derivative and verifies the
finite-difference identity in AO-density directions.  The test uses a
synthetic density and solvent kernel only.  It establishes neither a physical
RHF/DFT source, a solvent asset, a complete nonpolar scalar, forces, nor an
accuracy result.

### 3.6 No-label weighted-density cavitation scalar

The missing \(\Phi_{\rm cav,s}\) must not be replaced by an atom-surface
tension or a fitted area coefficient.  A directly relevant implicit-continuum
construction is the weighted-density cavity functional of Sundararaman,
Gunceler, and Arias.  It constrains a fourth-order local free-energy density
by the small-cavity, small-droplet, and planar-interface limits of the **pure
solvent**.  Let \(s_{\rm c}\) be a solvent-*centre* occupancy, let \(W_s\) be
the normalized even nearest-neighbour shell average at the solvent van der
Waals diameter, and write

\[
\bar s_{\rm c}=W_ss_{\rm c},\qquad
\Gamma_s=\log\!\frac{N_sT_s}{p_{\rm vap,s}}-1,
\qquad
A_s=\frac{\sigma_s}{N_sT_sR_{\rm vdW,s}}
-\frac{1+\Gamma_s}{6}.
\]

Then the scalar is

\[
\boxed{
G_{\rm cav,s}[s_{\rm c}]
=\int\!\left\{
p_s(1-\bar s_{\rm c})
+N_sT_s\bar s_{\rm c}(1-\bar s_{\rm c})
\left[
\bar s_{\rm c}+(1-\bar s_{\rm c})\Gamma_s
+15\bar s_{\rm c}(1-\bar s_{\rm c})A_s
\right]\right\}\,d\mathbf r .
}
\]

Its inputs are \((T_s,p_s,N_s,p_{\rm vap,s},\sigma_s,R_{\rm vdW,s})\), all
of which must come from independently sourced pure-liquid or equation-of-state
evidence.  There is no solvation-label coefficient in this term.  Its exact
functional derivative is also closed:

\[
\frac{\delta G_{\rm cav,s}}{\delta s_{\rm c}}
=W_s^\dagger f_s'(\bar s_{\rm c}),
\]

where, with \(x=\bar s_{\rm c}\), \(q=x(1-x)\), and
\(B=x+(1-x)\Gamma_s+15qA_s\),

\[
f_s'(x)=-p_s+N_sT_s
\left[(1-2x)B+q\{1-\Gamma_s+15(1-2x)A_s\}\right].
\]

[`route2_v0_weighted_density_cavity.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_weighted_density_cavity.py)
implements this scalar and its exact periodic discrete VJP; its dedicated test
locks the empty/bulk limits, central finite-difference derivative, translation
covariance, self-adjoint shell pairing, immutability, and crossed-state
rejection.  This is a genuine \(\Phi_{\rm cav}\) candidate in the **implicit**
V0-AQ-C route; it is not the archived molecular-HNC/WDA bridge and needs no
GROMACS, 3D-RISM, or explicit-liquid trajectory.

It is still deliberately below total admission.  The current
iso-density-product occupancy is an electrostatic-cavity candidate, whereas
the weighted-density scalar requires a source-derived solvent-centre
occupancy.  Setting the two fields equal would silently discard the distinct
electrostatic/solvent-centre separation that the construction is meant to
represent.  In addition, a physical source record must bind the shell kernel,
\(p_{\rm vap}\), and \(R_{\rm vdW}\) to the same solvent state; the Pauli and
dispersion scalar remain absent.  The independent standard-state conversion is
specified below, but does not make this cavitation component total.
Consequently this component cannot yet be added to a reaction energy, called a
total \(\Delta G_{\rm solv}\), or scored against FreeSolv/MNSol.

### 3.7 Exact standard-state scalar

The standard-state term is neither a solvent-specific fitted offset nor a
post-solve calibration.  Once the stationary auxiliary difference is declared
as an infinite-dilution excess free energy, conversion from an ideal-gas
pressure standard \(p^\circ\) to an ideal-solution concentration standard
\(c^\circ\) is the exact thermodynamic scalar

\[
\boxed{
\Delta G^\circ_{p\to c}(T)
=RT\log\!\left(\frac{c^\circ RT}{p^\circ}\right).
}
\]

The concentration is in mol/m\(^3\), so the logarithm is dimensionless.  For
the usual \(1\ \mathrm{atm}\to1\ \mathrm{mol\,L^{-1}}\) convention at
298.15 K, the term is positive (about \(1.893\ \mathrm{kcal\,mol^{-1}}\)).
It can be placed inside \(\Phi_{\rm std,s}\) before the stationary notation
because it is constant in \(n,m,\phi_s,\phi_0\); its nuclear-coordinate
derivative is exactly zero.  It may be used only in the declared direction.
Negating it, changing its pressure/concentration convention after an error is
known, or using it to compensate a missing nonpolar term is prohibited.

[`route2_v0_standard_state.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_standard_state.py)
computes this SI-defined scalar and its temperature derivative without a
solvent name, solvation label, radius, or adjustable coefficient.  This closes
only the standard-state *formula*: a total V0-AQ-C asset must still bind its
auxiliary excess-free-energy reference to the same \((p^\circ,c^\circ,T)\)
convention and supply the missing short-range/dispersion physics.

## 4. Archived molecular-liquid completion: V0-AQ-L

The full V0-AQ-L branch replaces the PCM control with a molecular liquid
functional \(\Phi_s[N]+U_s[n,N]\) that contains the solvent structure,
short-range interaction, and thermodynamic convention in the same scalar.
It needs frozen, independently sourced solvent assets before any target
solute is evaluated.  Required inputs include a molecular site model, bulk
state and equation of state, liquid correlation or classical-DFT functional,
short-range/dispersion physics, and a standard-state convention.  A solvent
name or scalar dielectric is insufficient.

There are two admissible upstream families, subject to their own independently
parameterized liquid assets:

1. **Joint/classical DFT.**  Joint density-functional theory couples an
   electronic functional and a classical liquid functional in one variational
   free energy.  JDFTx documents a `ClassicalDFT` fluid mode and named
   `fluid-solvent` assets, but those assets are inputs to freeze before
   scoring, not free parameters to tune after observing errors.

   The upstream name catalogue is not itself an eleven-solvent result.  Its
   documented `H2O`, `Methanol`, `Ethanol`, `CH3CN`, `DMSO`, `DMF`, `THF`,
   `CHCl3`, and `CH2Cl2` entries overlap nine Route-2 default strata, but it
   has no named `toluene` or `hexane` entry.  Thus even a source-pinned JDFTx
   installation would still require two independently sourced custom liquid
   assets and the same physical-liquid admission gates for all eleven.  The
   documented `FittedCorrelations` functional and property overrides such as
   `epsBulk`, `epsInf`, `pMol`, `Pvap`, `sigmaBulk`, `Rvdw`, and `Res` do not
   authorize a no-fit custom solvent: the former requires a separate
   source-policy review, and the latter do not determine a molecular excess
   functional, short-range interaction, or standard-state ledger.

   In particular, if \(\mathcal A_J^{\mathrm{sol}}[n,N;\mathbf R]\) is a
   JDFTx joint electronic--liquid scalar, then

   \[
   E_{\mathrm{MACE,gas}}(\mathbf R)
   +\min_{n,N}\mathcal A_J^{\mathrm{sol}}[n,N;\mathbf R]
   \]

   would double-count an unrelated auxiliary gas solute energy.  The only
   admissible ledger is the already declared stationary difference

   \[
   E_{\mathrm{MACE,gas}}(\mathbf R)
   +\left[
   \min_{n,N}\mathcal A_J^{\mathrm{sol}}[n,N;\mathbf R]
   -\min_n\mathcal A_J^{\mathrm{gas}}[n;\mathbf R]
   \right]
   +\Delta G_s^\circ.
   \]

   It neither identifies the auxiliary density with MACE coefficients nor
   feeds it into MACE field features.  The current host has no `jdftx` or
   `jdftx_gpu` executable; this remains an audited upstream possibility, not
   an executable backend.  [JDFTx `fluid`](https://jdftx.org/CommandFluid.html),
   [JDFTx `fluid-solvent`](https://jdftx.org/CommandFluidSolvent.html).
2. **QM--3D-RISM-SCF/MDFT.**  A molecular liquid density or RISM site-density
   functional can be jointly stationary with an auxiliary QM state.  The
   actual MACE source may not be substituted by GAFF/AM1-BCC topology charges;
   if a MACE-native interaction cannot be derived from one scalar, the
   auxiliary QM state remains the solute source for this branch.

Neither family is automatically ``parameter free``: solvent force-field,
correlation, and cavity/dispersion inputs must be recorded as independent
physical assets.  SaLSA is useful evidence that a nonlocal dielectric response
can be derived without empirical dielectric parameters, but its published
solvation model retains a fitted dispersion contribution and is therefore not
a strictly zero-fit total-free-energy endpoint here.

## 5. Conserved derivatives and acceptance gates

At a stationary auxiliary gas/continuum or gas/liquid state, the envelope theorem gives

\[
\frac{dG_{\mathrm{V0\text{-}AQ}}}{d\mathbf R}
=\frac{\partial E_{\mathrm{MACE,gas}}}{\partial\mathbf R}
+\left.
\frac{\partial\mathcal L_{\mathrm{aux}}^{\mathrm{sol}}}
{\partial\mathbf R}\right|_*
-\left.
\frac{\partial A_{\mathrm{aux}}^{\mathrm{gas}}}
{\partial\mathbf R}\right|_* .
\]

No response derivative from the legacy MACE field fixed point appears in this
expression.  Before any force claim, the candidate must prove that all terms
are derivatives of the same auxiliary scalar and pass the independent
smooth-cavity and central-difference gates.  V0-AQ-E is explicitly below that
threshold.

The prerequisite order is:

1. source-bind an auxiliary QM gas/solvent pair with the same functional,
   basis, geometry, charge/spin, numerical grid, and converged stationary
   states;
2. verify the exact gas subtraction and the absence of an extra half-coupling,
   CDS, fitted response scale, or MACE-field update;
3. for V0-AQ-C, register complete source-bound continuum/nonpolar assets for
   at least the frozen 11-solvent set before inspecting target-solvation labels;
   the archived V0-AQ-L route has the stricter molecular-liquid asset contract;
4. prove scalar/gradient/force gates, then freeze development, disjoint
   confirmation, and external-blind records; and
5. reject any failed structural or per-record accuracy gate without tuning,
   retraining, fine-tuning, response clipping, or per-record model selection.

## 6. Literature and upstream boundary

1. S. Petrosyan *et al.*, *Joint density-functional theory for electronic
   structure of solvated systems*,
   [arXiv:cond-mat/0606817](https://arxiv.org/abs/cond-mat/0606817).  It gives
   the joint electron--liquid variational principle used as the V0-AQ-L
   structural reference.
2. JDFTx, [`fluid`](https://jdftx.org/CommandFluid.html) and
   [`fluid-solvent`](https://jdftx.org/CommandFluidSolvent.html) documentation.
   The upstream interface distinguishes ClassicalDFT from continuum modes and
   lists named solvent assets plus explicit physical-property overrides; it
   does not authorize error-driven property selection.
3. N. Minezawa and S. Kato, *Efficient implementation of 3D-RISM-SCF*,
   *J. Chem. Phys.* **126**, 054511 (2007),
   [DOI:10.1063/1.2431809](https://doi.org/10.1063/1.2431809).  It reports
   direct electron-density-to-grid coupling and first derivatives of the
   free energy, motivating the V0-AQ-L scalar/force requirement.
4. R. Sundararaman *et al.*, *Spicing up continuum solvation models with
   SaLSA*, *J. Chem. Phys.* **142**, 054102 (2015),
   [arXiv:1410.2273](https://arxiv.org/abs/1410.2273).  Its dielectric response
   is parameter-free, while the reported full solvation model retains a
   dispersion fit; this is why it is not a no-fit total endpoint here.
5. R. Sundararaman, D. Gunceler, and T. A. Arias, *Weighted-density
   functionals for cavity formation and dispersion energies in continuum
   solvation models*, *J. Chem. Phys.* **141**, 134102 (2014),
   [arXiv:1407.4011](https://arxiv.org/abs/1407.4011).  Its physically motivated
   cavity construction is useful design evidence, but the published practical
   model retains fitted density-threshold and dispersion-scale parameters and
   cannot be copied into the no-fit V0 endpoint.
6. [PySCF solvent documentation](https://pyscf.org/user/solvent.html) and
   [ddCOSMO gradient API](https://pyscf.org/pyscf_api_docs/pyscf.solvent.grad.html).
   PySCF documents stationary PCM/ddCOSMO energy and analytic gradient
   machinery; this supports the numerical-control boundary only, not a smooth
   production cavity or an accuracy claim.
