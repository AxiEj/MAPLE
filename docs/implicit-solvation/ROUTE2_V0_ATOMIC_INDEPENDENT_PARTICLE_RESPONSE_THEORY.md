# Route-2 V0-RK: frozen atomic independent-particle response baseline

## Status and non-claim

`route2_v0_atomic_independent_particle_response.py` defines the algebra and
the preregistered generator for an offline atom-response asset.  It is not yet
a molecular response model, a continuum binding, a force/PES result, a
solvation calculation, or an accuracy result.  In particular, an
independent-particle response of isolated spherical atoms omits bonding,
charge transfer, molecular screening, and all solvent short-range physics.

The scope is narrower and useful: it supplies a physically sourced,
high-dimensional, charge-neutral candidate for the missing V0-RK baseline
(C_0).  The frozen MACE-MDP checkpoint supplies only the molecular
three-dimensional moment response; it does not get relabelled as a density or
used as an energy/force model.

The immutable source policy is in
[`route2-v0-atomic-independent-particle-hf-def2-tzvpd-prereg-v1.json`](benchmarks/route2-v0-atomic-independent-particle-hf-def2-tzvpd-prereg-v1.json).

## 1. Atomic transition-density coefficient/dual space

For one neutral spherical atomic Hartree--Fock calculation, let canonical
spatial orbitals, energies, and spin-summed occupations be

\[
\{\phi_p,\epsilon_p,n_p\}_{p=1}^{K},
\qquad \epsilon_1\le\cdots\le\epsilon_K.
\]

For every pair (i<a) with (n_i>n_a) and
(epsilon_a>epsilon_i), define the real symmetric transition density

\[
\tau_{ia}(\mathbf r)
=\phi_i(\mathbf r)\phi_a(\mathbf r)
+\phi_a(\mathbf r)\phi_i(\mathbf r),
\]

and its static independent-particle coefficient

\[
w_{ia}=
\frac{n_i-n_a}{2(\epsilon_a-\epsilon_i)}>0.
\]

There is no learned, fitted, or error-selected transition cutoff: the frozen
numerical rule retains every pair with positive occupation difference and a
nonzero canonical gap.  Fractional occupations of `AtomSphAverageRHF` are
preserved, so the open-shell free atoms retain a spherical density rather than
being assigned an arbitrary Cartesian open-shell orientation.

For a scalar external potential (v\), the declared source/dual pairing is

\[
f_{ia}=\int \tau_{ia}(\mathbf r)v(\mathbf r)\,d\mathbf r,
\qquad
x_{ia}=-w_{ia}f_{ia}.
\]

Consequently the frozen atomic baseline covariance is diagonal,

\[
C_{0,ia,jb}=w_{ia}\,\delta_{ij}\delta_{ab}\succeq0.
\]

This is the static independent-particle (irreducible) response, not a claim
of TDHF, RPA, or exact molecular susceptibility.  The transition-density
representation follows the standard occupied--virtual response construction;
response/polarizability densities retain spatial information that a molecular
polarizability alone loses.  See the response-density discussion in
[Zhao *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10601476/) and the
linear-response distinction between irreducible and screened polarizability in
[Sundararaman *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9813915/).

## 2. Neutrality, passivity, and the atomic moment map

Orbital orthogonality gives each mode exactly zero integrated electron number:

\[
q_{ia}=\int\tau_{ia}(\mathbf r)d\mathbf r
=2\langle\phi_i\mid\phi_a\rangle=0.
\]

Thus the induced coordinate space is **intrinsically neutral**.  It must not
be enlarged with a dummy monopole merely to satisfy an implementation
interface.  `complete_route2_v0_response_kernel` now accepts
`charge_constraint_vector=None` for this case; it returns every kernel-null
mode as an exact KKT constraint, rather than duplicating a nonexistent charge
row or assigning a large artificial hardness.

Let the first moment of the electron-number transition density be

\[
d_{ia,k}=\int r_k\tau_{ia}(\mathbf r)d\mathbf r.
\]

The physical electronic dipole map is (A=-d^\mathsf T), because electron
charge is negative.  Therefore

\[
\alpha_A=A C_0 A^\mathsf T
=d^\mathsf T C_0d\succ0.
\]

For the frozen spherical atoms, (alpha_A) must be proportional to the
identity to numerical tolerance.  The generator fails closed if any retained
mode has nonzero charge, if (C_0) loses positivity, or if this isotropy gate
fails.  No eigenvalue clipping, radial width selection, or post-hoc response
rescaling is allowed.

## 3. Molecular direct sum and V0-RK completion

For a molecule containing atoms (A=1,\ldots,N), concatenate their frozen
mode coordinates:

\[
x=(x^{(1)},\ldots,x^{(N)}),
\qquad
C_0=\operatorname{diag}(C_0^{(1)},\ldots,C_0^{(N)}),
\]

with a block-diagonal atom-dipole map

\[
A=\operatorname{diag}(A^{(1)},\ldots,A^{(N)}).
\]

The direct sum preserves neutrality and makes
(S_0=A C_0A^\mathsf T\succ0) testable.  It does **not** pretend that the
direct sum already has the correct molecular polarizability.  The latter is
the one item frozen MACE-MDP has demonstrated at the molecular level.

With its audited atom partition (W\),

\[
GW=I_3,
\qquad \Gamma=W\alpha_\theta W^\mathsf T,
\]

V0-RK applies the exact covariance replacement

\[
\begin{aligned}
L&=C_0A^\mathsf T(AC_0A^\mathsf T)^{-1},\\
C&=C_0-L(AC_0A^\mathsf T)L^\mathsf T+L\Gamma L^\mathsf T.
\end{aligned}
\]

It follows without fitting that

\[
C\succeq0,
\qquad AC A^\mathsf T=\Gamma,
\qquad (GA)C(GA)^\mathsf T=\alpha_\theta.
\]

Hence the atom-HF transition densities fix the full radial and hidden-mode
baseline, while the unmodified MACE tensor fixes only its observed molecular
moment projection.  A singular (S_0) is rejected, not pseudoinverted.

## 4. One scalar and the source potential

On (operatorname{Ran}C), the electronic induction scalar is

\[
F_{\rm AIPR}(x;\mathbf R)
=\frac12 x^\mathsf TC^+x,
\qquad x\in\operatorname{Ran}C.
\]

For an external dual (f), its constrained Euler equation is

\[
C^+x+\Pi_Cf=0,
\qquad x=-Cf,
\]

which guarantees reciprocal, passive response in this declared dual space.
There is no fixed-point mixer and no operational energy derivative.

At an exterior point (mathbf s), the induced *electronic* potential is

\[
V_e(\mathbf s)
=-\sum_{A,ia\in A}x_{ia}
\int\frac{\tau_{ia}(\mathbf r)}{|\mathbf r-\mathbf s|}d\mathbf r.
\]

The source falsifier evaluates this integral with the same atom-centred GTO
AO basis used to construct the modes.  This oracle evaluation is appropriate
for a gas-phase spatial-response gate.  It is not yet the required
production source-to-continuum implementation or a runtime-speed result.

With a later reciprocal physical continuum (P_{\mathbf R}), a permanent
same-basis source (c_0), and every support constraint (N^\mathsf Tx=0),
the only admissible future electronic--continuum scalar is

\[
\mathcal G(x;\mathbf R)=E_{\rm gas}(\mathbf R)
+\frac12x^\mathsf TC^+x
+\frac12(c_0+x)^\mathsf TP_{\mathbf R}(c_0+x),
\]

not a separately assembled model-energy change plus a PCM half coupling.

## 5. Falsification order

Before any experimental solvation value is read, this candidate must pass:

1. deterministic atom-asset provenance, orbital orthonormality, mode
   neutrality, atomic PSD/isotropy, and a repeat table hash;
2. the frozen MACE partition identity, V0-RK PSD/moment/support identities,
   and a same-basis acetone QM induced-MEP falsifier;
3. the predeclared twelve-record, ten-actual-functional-group QM density,
   MEP, induced-dipole, reaction-potential, rigid-motion, and coordinate
   derivative gates;
4. an independently source-bound smooth cavity, reciprocal continuum,
   nonpolar ledger, KKT/envelope-force/PES proof, and runtime gate; then only
5. the immutable all-record experimental and disjoint blind panels.

A passing single acetone source test would only advance step 2.  It cannot
erase the historical (7.0414420821\) kcal/mol ethyl-acetate outlier or claim
that every record is below (1.5\) kcal/mol.

## 6. Relation to density-defined continuum work

The construction solves the **solute response** underdetermination; it does
not manufacture a physical custom solvent from a dielectric constant.  JDFT
provides the appropriate common-scalar architecture for electron--liquid
coupling, while its practical liquid assets remain separate requirements.
[Petrosyan *et al.*](https://arxiv.org/abs/cond-mat/0606817) formulate that
joint variational principle.  SaLSA derives a nonlocal dielectric response
without empirical dielectric-response parameters, but its published total
model includes a fitted dispersion contribution and cannot be imported as a
V0 total free-energy ledger.
[Sundararaman *et al.*](https://arxiv.org/abs/1410.2273)

Similarly, density-overlap cavities are a promising way to remove
error-selected atomic radii, but a source-pinned cavity/nonpolar functional
and its coordinate derivative are independent gates.  They must not be
replaced by an iso-density threshold or solvent radius chosen after a
solvation error is known.
