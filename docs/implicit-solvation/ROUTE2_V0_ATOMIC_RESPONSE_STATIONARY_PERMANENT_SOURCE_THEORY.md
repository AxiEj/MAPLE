# Route-2 V0-ARSP: stationary permanent source from frozen atomic response

## Status and claim boundary

This document defines **V0-ARSP** (atomic-response stationary permanent
source), a no-training candidate for the missing permanent electronic source
in the V0-AIPR coefficient/dual space.  It is deliberately narrower than a
Route-2 solvation method.  It creates neither a continuum calculation, cavity,
nonpolar term, force/PES result, runtime result, nor experimental solvation
prediction.

The prior GFN2 MOLDEN candidate has already been rejected by its frozen
static-MEP gate.  That rejection is retained; V0-ARSP does not reuse its
orbitals, effective cores, response, or thresholds.  The only reusable inputs
are the independently frozen spherical atomic-HF table, the already admitted
MACE-MDP **moment** covariance/partition, and exact AO Coulomb integrals.

The next admissible result is a preregistered one-acetone *gas-phase* static
MEP falsifier.  V1 preserved a helper output but stopped in its parent
field-name mapping before a final registered gate verdict; that sealed output
is not used to select the candidate or thresholds.  V2 changes only that
name-to-field mapping and reruns the unchanged frozen source.  A pass remains
only a source admission to a broad QM physics panel and same-basis
continuum/KKT gate.  It cannot be presented as a solvation-accuracy result.

## 1. Why this construction is needed

The response completion supplies a reciprocal passive tangent relation

\[
\delta x=-C_{\mathbf R}f,
\qquad C_{\mathbf R}\succeq0,
\]

but the reference-shift theorem shows that it cannot choose a permanent
source.  In particular, merely adding the frozen atomic-HF densities gives a
promolecule, which is a source control rather than a stationary molecular
state.  The permanent correction must instead be selected by an explicit
scalar in the same neutral transition-density coordinates as the response.

The construction below is the second-order, stationary auxiliary-model route
from the permanent-reference identifiability audit.  It resembles the
well-known logic of a second-order density-fluctuation expansion, but it does
**not** import SCC-DFTB's parameterized repulsive energy or any charge-fit
model.  The historical SCC-DFTB derivation motivates the distinction between a
scalar density expansion and an arbitrary charge update
([Elstner *et al.*](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.58.7260/fulltext)).

## 2. Frozen all-electron reference and neutral coordinates

For each atom \(A\), the hash-bound atomic-HF table contains canonical orbitals
\(\phi_{Ap}\), spin-summed occupations \(n_{Ap}\), and all neutral transition
densities \(\tau_{Am}\).  Its frozen all-electron reference density is

\[
n_{A}^{0}(\mathbf r)
 =\sum_p n_{Ap}\lvert\phi_{Ap}(\mathbf r)\rvert^2,
\qquad
D_A^0=C_A\operatorname{diag}(n_A)C_A^\mathsf T.
\]

At a molecular geometry, assemble its direct sum

\[
n_{\rm ref}(\mathbf r;\mathbf R)
 =\sum_A n_A^0(\mathbf r-\mathbf R_A).
\]

It has the exact neutral-atom electron count but is **not** yet called the
molecular state.  The transition coordinates preserve that count exactly:

\[
n(\mathbf r;\mathbf R,x)
=n_{\rm ref}(\mathbf r;\mathbf R)
 +\sum_m x_m\tau_m(\mathbf r),
\qquad
\int\tau_m(\mathbf r)d\mathbf r=0.
\]

The pre-existing V0-RK completion supplies the symmetric covariance \(C\) and
its Moore--Penrose curvature \(C^+\) on the explicitly declared support
\(\mathcal S=\operatorname{Ran}C\).  Every null direction remains an exact
constraint; no artificial hardness, eigenvalue clipping, or monopole has been
added.

## 3. A single stationary permanent-source scalar

Let the electronic potential energy acting on an electron density change on
atom \(A\), due to all **other** frozen atoms, be

\[
u^{\rm other}_A(\mathbf r)
=-\sum_{B\ne A}\frac{Z_B}{\lvert\mathbf r-\mathbf R_B\rvert}
 +\sum_{B\ne A}\int
 \frac{n_B^0(\mathbf r')}{\lvert\mathbf r-\mathbf r'\rvert}d\mathbf r'.
\]

Its exact AO-integral dual is

\[
b_m(\mathbf R)=\int\tau_m(\mathbf r)u_A^{\rm other}(\mathbf r)d\mathbf r.
\]

The candidate electronic scalar is

\[
F_{\rm ARSP}(\mathbf R,x)
=E_{\rm ref}(\mathbf R)
 +\frac12x^\mathsf TC_{\mathbf R}^{+}x
 +x^\mathsf Tb_{\mathbf R},
\qquad x\in\mathcal S.
\]

Here \(E_{\rm ref}\) is a geometry-dependent but \(x\)-independent frozen
atomic reference ledger (atomic-HF self energies plus its declared exact
interatomic electrostatic ledger).  It does not affect the stationary density
at a fixed geometry.  Future force work must expose its coordinate derivative;
this document makes **no** force claim.

The Euler equation is

\[
C^+x+P_Cb=0,
\quad\Longrightarrow\quad
\boxed{x_0=-Cb},
\]

where \(P_C\) is the response-support projector.  The permanent candidate is
therefore

\[
n_0(\mathbf r;\mathbf R)=n_{\rm ref}(\mathbf r;\mathbf R)
+\sum_m(x_0)_m\tau_m(\mathbf r).
\]

This is not a Harris-like nonselfconsistent energy evaluation: the correction
is the unique stationary point of the declared positive-curvature scalar in
its exact support.  Its on-shell correction energy is checked both as

\[
\frac12x_0^\mathsf TC^+x_0+x_0^\mathsf Tb
=-\frac12b^\mathsf TCb.
\]

The map \(b\mapsto x_0\) has derivative \(-C\), so it preserves the existing
reciprocal/passive tangent response by construction.

## 4. Exact source ledger and mandatory state gates

The permanent exterior potential uses one all-electron ledger:

\[
V_0(\mathbf s)=
\sum_A\frac{Z_A}{\lvert\mathbf s-\mathbf R_A\rvert}
-\operatorname{Tr}\left[D_0 I(\mathbf s)\right],
\qquad
I_{\mu\nu}(\mathbf s)=
\int\frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}
{\lvert\mathbf r-\mathbf s\rvert}d\mathbf r.
\]

No effective core, fitted atom charge, Gaussian width, or independent source
is spliced into this ledger.  Before this potential may be interpreted, the
candidate must pass all of the following without an edit:

1. global AO carrier count and every atom-local frozen-MO metric;
2. response-support stationarity, exact support constraints, and the
   closed-form stationary-energy identity;
3. total electron-number conservation;
4. nonnegative represented electron-number density on the fixed, source-blind
   Cartesian density grid (and no density renormalization); and
5. the full frozen 516-point static QM MEP and permanent-dipole falsifier.

The AO product coefficient matrix is a carrier for the coarse electron-number
density, not a claimed ensemble one-particle density matrix after the
response-kernel completion.  A first-order occupied--virtual density tangent
can have a negative AO-metric eigenvalue even while the represented real-space
density is positive; treating that carrier spectrum as a Pauli test would
therefore reject a valid density representation for the wrong reason.  The
physical gate is instead the pointwise \(n(\mathbf r)\ge0\) source test on a
preregistered grid.  A material negative density is rejected rather than
projected, clipped, mixed, damped, or renormalized.

## 5. Explicit limitations

All AIPR transition modes are neutral.  V0-ARSP consequently cannot represent
net interatomic charge transfer.  We will not conceal that limitation by
adding QEq sites, fitted hardnesses, learned monopoles, or error-selected
charge-transfer coordinates.  If the fixed source fails its QM MEP/occupancy
or later multi-functional-group physics gates, it is rejected as a V0
candidate.  A future, separately registered route could study ab-initio
ionization/electron-affinity charge-transfer modes, but that is neither this
model nor a permitted retrofit.

Likewise, the source uses a completed frozen MACE-MDP **polarizability** only
as an observed moment constraint.  MACE-MDP remains neither an electronic
energy nor a force model.  No raw MACE density is relabelled as an electron
density.

## 6. Future common scalar (not executed here)

Only after the source passes its physical gates can it be coupled to an
energy-conjugate continuum in the same density/source pairing:

\[
\mathcal G(\mathbf R,x,\sigma)=F_{\rm ARSP}(\mathbf R,x)
+\frac12\sigma^\mathsf TA_{\mathbf R}\sigma
+\sigma^\mathsf TB_{\mathbf R}
\left[n_{\rm ref}+T x\right].
\]

The joint Euler equations then have a symmetric KKT form.  This is the same
variational-polarization principle needed for consistent energies and
responses; a polarizable-embedding treatment is physically meaningful only
when its induced variables are stationary in the energy that is differentiated
([Goletto *et al.*](https://pubs.acs.org/doi/10.1021/acs.accounts.0c00662)).

That continuum step remains prohibited until the fixed permanent source,
broad QM physics panel, smooth cavity, coordinate derivative, and force/PES
gates have independently passed.
