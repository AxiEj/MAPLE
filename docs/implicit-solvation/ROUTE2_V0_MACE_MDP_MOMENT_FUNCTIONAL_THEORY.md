# Route-2 V0 frozen MACE-MDP moment-functional branch

## Status and boundary

This document records the coefficient-only V0 branch opened by the frozen
MACE-MDP acetone screen. Its inherited one-radial Gaussian induced-density
realization has been **rejected before PCM**. A distinct physical free-atom
translation-tangent source now passes one acetone gas-phase QM-MEP falsifier.
It has both an induced-only reduced KKT gate and a frozen-permanent-source
direct-sum KKT gate. The latter provides an exact permanent/induced *surface*
source dual without forcing the radial tangent into a GTO width. It still has
no stationary permanent electronic density, physical continuum binding, force,
or broad physics certificate. Consequently this remains a **theory and
admission contract**, not an admitted electronic implementation, a public
Route-2 calculator, a total-solvation result, or an accuracy claim.

The sealed artifact
[`route2-v0-mace-mdp-acetone-response-v1.json`](benchmarks/route2-v0-mace-mdp-acetone-response-v1.json)
used the official frozen MACE-MDP checkpoint without a continuum, an
experimental solvation label, training, fine tuning, fitting, rescaling, or
eigenvalue repair. Against the already frozen acetone
omegaB97M-V/def2-TZVPD finite-field reference, its polarizability passed:

\[
\begin{array}{rl}
\|\alpha_{\rm MDP}-\alpha_{\rm QM}\|_F/\|\alpha_{\rm QM}\|_F
  &=0.0038678622 < 0.20,\\
\operatorname{tr}(\alpha_{\rm MDP})/\operatorname{tr}(\alpha_{\rm QM})
  &=0.9987591052,\\
\max_i |a_i-a_i^{\rm QM}|/|a_i^{\rm QM}|
  &=0.0041058654 < 0.30,\\
\lambda_{\min}(\alpha_{\rm MDP}) &=34.64389157\ a_0^3>0.
\end{array}
\]

The raw tensor antisymmetry was only
\(2.4676\times10^{-17}\), below the frozen \(10^{-10}\) gate. This permits a
machine-precision symmetric representation of that **already symmetric**
property, not a response symmetrization repair. The source result does **not**
make MACE-MDP an energy or force model. It only admits its frozen
\(\alpha_\theta(\mathbf R)\) as a coefficient of the scalar below.

The direct MACE-MDP dipole is not selected as a permanent PCM source. Its
acetone agreement (relative error \(0.001903\)) is a non-gating interface
diagnostic only. A permanent source must remain an independently auditable,
energy-conjugate density representation.

The subsequent sealed
[`route2-v0-mace-mdp-atomic-map-acetone-v1.json`](benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json)
audit resolved one part of the \(D\) problem without choosing a width or a
cavity.  The official frozen readouts reconstruct their public molecular
properties to machine precision: atomic dipole closure is
\(6.41\times10^{-17}\), atomic polarizability closure is
\(1.07\times10^{-16}\), and the Cartesian partition
\(W_a=\alpha_a\alpha^{-1}\) satisfies \(\sum_aW_a=I\) to
\(4.06\times10^{-17}\).  Thus \(p_a=W_ap\) is an identity-bound atomic
induced-dipole partition.  It does **not** yet specify how those atom moments
become a radial GTO density or an electrostatic source.

That missing radial information is now a measured, rather than merely
theoretical, obstruction. The preregistered one-radial \(l=1\) GTO map using
the inherited 1.5-\(\mathring{\rm A}\) width passed all algebraic identities
(charge, induced-dipole, source-potential, source-duality, and atomic-partition
errors were at most \(2.26\times10^{-16}\)). It nevertheless failed against
the frozen acetone QM finite-field exterior potential **in vacuum, before any
cavity or solvent calculation**:

\[
\frac{\|\partial_E V_{D_1}-\partial_E V_{\rm QM}\|_F}
     {\|\partial_E V_{\rm QM}\|_F}=0.4597926523>0.20,
\qquad
\max_{\hat E}\operatorname{relerr}(V_{D_1},V_{\rm QM})
=0.4965624773>0.30.
\]

The independent induced-dipole tensor still agrees at \(0.0038678622\), so
the failure is specifically a spatial-response failure, not a polarizability
or source-duality failure. The immutable execution record is
[route2-v0-mace-mdp-induced-source-acetone-v1.json](benchmarks/route2-v0-mace-mdp-induced-source-acetone-v1.json);
its QM-dipole bookkeeping erratum is preserved separately and does not change
either failed MEP gate. The rejected map must not enter a KKT solve, a PCM
calculation, or any experimental accuracy panel.

## V0-ADT: a source-provenanced radial successor

The new
[`route2-v0-atomic-displacement-source-acetone-v1.json`](benchmarks/route2-v0-atomic-displacement-source-acetone-v1.json)
does not select a Gaussian width. For each supported atom it starts from a
frozen, independently generated spherical free-atom HF electron density
\(n_Z(r)\), tabulates

\[
N_Z(r)=4\pi\int_0^r n_Z(s)s^2\,ds,
\]

and uses the exact infinitesimal electron-translation tangent for the frozen
atom dipole \(p_a=W_ap\):

\[
V_{\rm ADT}(\mathbf r)=
\sum_a
\frac{p_a\cdot(\mathbf r-\mathbf R_a)}
     {Z_a|\mathbf r-\mathbf R_a|^3}
N_{Z_a}(|\mathbf r-\mathbf R_a|).
\]

Two preregistered one-thread generations gave a byte-identical radial asset.
At every one of the frozen 516 exterior acetone QM-MEP points, its field
response passes the already registered gates:

\[
\operatorname{relerr}_F(V_{\rm ADT},V_{\rm QM})=0.1484314509<0.20,
\qquad
\max_{\hat E}\operatorname{relerr}=0.1611208415<0.30.
\]

The molecular induced-dipole mismatch is still
\(0.0038678622<0.20\). This is a positive **source-level** result: replacing
the guessed radial shape resolves the observed acetone near-field failure
without fitting or retraining. It is not a density-basis or scalar result.
The table is not yet a GTO coefficient vector and has not been tested on the
frozen twelve-record physics panel. The new reduced source-space kernel exposes
an exact surface transpose only for the **induced** rank-three map. A separate
direct-sum construction now supplies a frozen permanent source dual without a
GTO projection, but neither construction supplies a full stationary electronic
functional. It must not enter a physical PCM/total-KKT calculation, force, or
experimental-accuracy work yet.

## Reduced induced-only source-space KKT gate

The real-space V0-ADT map can now be evaluated directly at a declared
continuum surface. Let \(B_{\mathbf R}\in\mathbb R^{m\times3}\) contain its
three Cartesian unit-dipole surface-potential columns and let
\(Q_{\mathbf R}\) be the continuum's energy-conjugate response. Then the
reduced scalar

\[
g_{\mathbf R}(p;f)=\tfrac12p^\mathsf T\alpha_\theta^{-1}p
+\tfrac12(B_{\mathbf R}p)^\mathsf TQ_{\mathbf R}(B_{\mathbf R}p)
+p^\mathsf Tf
\]

has the stationary Hessian
\(K_p=\alpha_\theta^{-1}+B_{\mathbf R}^\mathsf TQ_{\mathbf R}B_{\mathbf R}\).
The implementation enforces that the source pullback is exactly
\(B_{\mathbf R}^{\mathsf T}\), that \(K_p\succ0\), and that the induced
response \(-K_p^{-1}\) is reciprocal and passive. This is a strict lower
structural gate: its unit controls use a synthetic reciprocal continuum and it
contains no permanent solute source, gas energy, nonpolar term, coordinate
derivative, or experimental solvation value. The complete derivation and
boundaries are in
[`ROUTE2_V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_THEORY.md`](ROUTE2_V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_THEORY.md).

## Frozen-permanent-source direct-sum KKT gate

The no-training V0-FD source and V0-ADT tangent can be joined without a
Gaussian projection.  The direct-sum surface map is

\[
v=S_{\mathbf R}c_0+B_{\mathbf R}p,
\]

where \(c_0\) is the unchanged zero-field MACE point-multipole source and
\(p\) is the V0-ADT induced dipole.  Both `S` and `B` are evaluated directly,
and their transposes are the only allowed surface-charge duals.  With \(c_0\)
frozen, the scalar

\[
g(p;c_0)=\tfrac12p^\mathsf T\alpha_\theta^{-1}p+
\tfrac12(S c_0+B p)^\mathsf TQ(S c_0+B p)
\]

gives one stationary induced response and one energy ledger. The structural
implementation verifies direct-sum duality, full \([S\;B]\)-restricted
continuum reciprocity and homogeneous linearity, KKT stationarity, passivity,
and a frozen-source envelope identity.
It is not a claim that \(c_0\) minimizes an electronic functional.  Full
details and non-claims are in
[`ROUTE2_V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_THEORY.md`](ROUTE2_V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_THEORY.md).

## Conditional rank-three scalar

The following rank-three construction explains the target *single-GTO-basis*
common-energy requirement, but it is **not admitted**: V0-ADT has only a
real-space source potential, not the required one-basis \(D_{\mathbf R}\) and
transpose. The direct-sum frozen-source gate above is an exact alternative
surface representation, not a replacement for this full electronic-density
functional. A full response-kernel successor will use
\(\delta c=-C_{\mathbf R}f\) rather than infer the entire response from three
columns of \(D_{\mathbf R}\).

Let \(c_0(\mathbf R)\) be a frozen gas-phase density coefficient vector in a
single GTO source/dual basis. It must already satisfy
\(q^\mathsf Tc_0=Q\). Let \(p\in\mathbb R^3\) be an induced molecular dipole
in \(e\,a_0\), and let \(D_{\mathbf R}\) map that physical dipole into the same
GTO coefficient basis:

\[
\delta c=D_{\mathbf R}p,
\qquad q^\mathsf TD_{\mathbf R}=0,
\qquad M_{\mathbf R}D_{\mathbf R}=I_3.
\]

Here \(M_{\mathbf R}\) is the molecular dipole moment map expressed in
\(e\,a_0\). The existing
`AtomCenteredL1GTOBasis.molecular_charge_dipole_constraints` supplies its
charge-plus-dipole rows in \(e\,\mathring{\rm A}\); the conversion to the
\(e\,a_0\) convention must be explicit in the future map constructor. It
may not be hidden in a field or continuum adapter.

Let \(B_{\mathbf R}\) map these **same** coefficients to the continuum
surface potential, and let \(A_{\mathbf R}\) be the reciprocal surface-energy
matrix. With the MACE-MDP tensor converted from
\(e\,\mathring{\rm A}^2/{\rm V}\) to \(a_0^3\), define

\[
\boxed{
\begin{aligned}
\mathcal L(\mathbf R,p,\sigma)
={}&E_{\rm gas}(\mathbf R)
+\tfrac12 p^\mathsf T\alpha_\theta(\mathbf R)^{-1}p\\
&+\tfrac12\sigma^\mathsf TA_{\mathbf R}\sigma
+\sigma^\mathsf TB_{\mathbf R}
  \bigl[c_0(\mathbf R)+D_{\mathbf R}p\bigr].
\end{aligned}}
\]

`E_gas` must itself be a frozen energy/force-capable gas model; MACE-MDP does
not supply it. This is therefore a newly specified hybrid scalar, with every
coefficient fixed before solvation labels are read. It is not a claim that
either upstream network individually supplied this energy.

The stationary equations are

\[
\begin{aligned}
\alpha_\theta^{-1}p+D_{\mathbf R}^\mathsf TB_{\mathbf R}^\mathsf T\sigma&=0,\\
A_{\mathbf R}\sigma+B_{\mathbf R}(c_0+D_{\mathbf R}p)&=0.
\end{aligned}
\]

Eliminating \(\sigma\) gives

\[
P_{\mathbf R}=-B_{\mathbf R}^\mathsf TA_{\mathbf R}^{-1}B_{\mathbf R},
\qquad
\mathcal G(\mathbf R,p)=E_{\rm gas}
+\tfrac12p^\mathsf T\alpha_\theta^{-1}p
+\tfrac12(c_0+D p)^\mathsf TP_{\mathbf R}(c_0+D p).
\]

Thus the required local stability condition is not an SCF mixing radius but

\[
\boxed{K_p=\alpha_\theta^{-1}+D_{\mathbf R}^\mathsf TP_{\mathbf R}D_{\mathbf R}
\succ0.}
\]

If it holds, response, polarization work, and forces come from the same
scalar. With an external coefficient-dual potential \(f\), add
\((c_0+Dp)^\mathsf Tf\) before taking stationarity. The induced response is
then a derivative of \(\mathcal G\), so reciprocity follows from the scalar;
it is not imposed on a learned fixed-point Jacobian.

## Non-negotiable source-map rule

The **same-basis radial density representer** in \(D_{\mathbf R}\) remains
unresolved. V0-ADT fixes a physical real-space radial translation tangent;
the inherited one-radial Gaussian has been rejected and the atomic Cartesian
partition is no longer guessed. A point dipole, another guessed Gaussian
width, an error-selected atomic partition, a higher-multipole add-on, or a
projection that changes after viewing solvation errors is forbidden.

This is an identifiability boundary, not a request to search widths. The
MACE-MDP tensor fixes only the three-dimensional projected response
\(M_{\mathbf R}C_{\mathbf R}M_{\mathbf R}^{\mathsf T}=\alpha_\theta\), where
\(C_{\mathbf R}\) is a full induced-density susceptibility. It does not fix
the high-dimensional null space of \(M_{\mathbf R}\), which controls the
near-field potential. The one-radial failure is the executable counterexample:
matching the complete molecular dipole response does not identify a correct
response density. V0-ADT shows that independently sourced radial physics can
resolve this ambiguity on acetone; it must now either acquire an exact
source/dual basis and pass broad physics, or be rejected. A separately
source-provenanced full response kernel remains the fallback, not an invitation
to add an error-selected moment-preserving representer.

Any future rank-three representer must reproduce the frozen
\(p_a=W_ap\) partition above under a uniform field and provide a
source-provenanced, geometry-differentiable GTO map satisfying all of the
following before it is connected to a PCM. A full-kernel successor must
instead prove the corresponding charge neutrality, source duality, covariance,
and uniform-field partition identities for \(C_{\mathbf R}\).

1. **Charge and moment identity:** \(q^\mathsf TD=0\) and
   \(M_{\mathbf R}D=I_3\) at float64 precision.
2. **Energy duality:** surface coupling is evaluated only as
   \(\sigma^\mathsf TB Dp=p^\mathsf TD^\mathsf TB^\mathsf T\sigma\), using the
   same `B` and transpose as the permanent source.
3. **Euclidean covariance:** translation leaves the induced neutral density's
   dipole invariant, and a rigid rotation covaries both \(D\) and \(p\).
4. **No hidden response correction:** a failed raw MACE-MDP tensor or source
   cannot be made eligible by clipping, shifting, rescaling, changing a
   width, adding moments, or a source-map choice.
5. **Physical validation:** the representer's vacuum potential, reaction
   potential, and induced density/moment must be compared with preregistered
   QM quantities before it is allowed to influence a solvation result.

The existing same-basis GTO Galerkin operator is useful because it already
enforces \(\sigma^\mathsf TBc=c^\mathsf TB^\mathsf T\sigma\). It is not by
itself a valid \(D\): its current MACE density embedding preserves a frozen
`l<=1` density but does not prescribe how a molecular induced dipole is
distributed across that density.

## Full response-kernel fallback

The no-training fallback does not invent a radial GTO representer. It starts
only from an independently source-bound full neutral response covariance
\(C_0\) in the eventual source/dual basis, then replaces its atom-dipole
covariance by the frozen MACE-MDP moment covariance
\(W\alpha_\theta W^\mathsf T\) through an exact Schur-complement identity.
The resulting \(C\succeq0\) has an electronic scalar
\(\frac12\delta c^\mathsf TC^+\delta c\) on its declared support and returns
kernel-null modes as constraints rather than adding an arbitrary hardness.
It is implemented as a synthetic structural gate only; no physical \(C_0\)
asset is checked in. The derivation, limitations, and source requirements are
in [`ROUTE2_V0_RESPONSE_KERNEL_THEORY.md`](ROUTE2_V0_RESPONSE_KERNEL_THEORY.md).

## Force and performance gates

At a joint stationary state, the force must be the envelope derivative of the
same scalar:

\[
-\frac{d\mathcal L}{dR_I}
=-\left.\frac{\partial\mathcal L}{\partial R_I}\right|_{p^*,\sigma^*}.
\]

It therefore includes explicit derivatives of \(E_{\rm gas}\),
\(\alpha_\theta^{-1}\), \(c_0\), \(D\), \(B\), and \(A\). The first
coefficient-derivative gate has passed: the sealed all-Cartesian MACE-MDP
artifact reports \(2.28\times10^{-8}\) relative error for \(d\mu/dR\) and
\(3.98\times10^{-8}\) for \(d\alpha/dR\) against a fixed \(10^{-4}\,
\mathring{\rm A}\) central difference, with translation and tensor-symmetry
identities at machine precision. This makes the frozen \(\alpha_\theta\)
coordinate derivative available as one explicit term; it does not eliminate
any other term in the envelope derivative. The following are still mandatory
before a force or PES claim:

* KKT residual, charge, source-duality, reciprocal/passive response, and
  \(K_p\) positivity checks;
* multi-direction Cartesian force finite differences with second-order
  refinement, rigid-motion covariance, and a coordinate-loop-work check;
* a smooth-cavity/grid-refinement check on each supported continuum backend.

The coefficient screen took 4.342 s including model startup versus 855.736 s
for the archived fixed-geometry QM finite-field reference (ratio 0.00507).
That is encouraging only for this isolated coefficient evaluation. It is not
an end-to-end implicit-solvation speed claim. A complete Route-2V0
implementation must measure cold and warm calculation cost, continuum solve,
and force cost against the same QM task before it can claim a runtime
advantage.

## Ordered admission path to accuracy

1. Bind the direct-sum V0-ADT frozen-source gate to a physical reciprocal
   continuum and test it over the frozen QM physics set. The gate has no
   source-bound cavity and is not a total-solvation representation. The
   rejected one-radial map is not a starting point for a parameter sweep and
   may not be used to choose a projection.
2. Either preserve the explicitly frozen permanent source as a bounded V0
   approximation or construct a full non-arbitrary stationary electronic
   density functional; do not silently relabel the direct sum as the latter.
   Then demonstrate the joint scalar, \(K_p\succ0\), one energy ledger, and
   envelope forces.
3. Extend the **physics** validation over the frozen twelve-record,
   ten-actual-functional-group geometries with QM response/MEP/PCM component
   references. This is not an experimental accuracy panel and must not read
   solvation labels.
4. Freeze the whole scalar and only then run the immutable historical
   FreeSolv10 and twelve-record/ten-actual-functional-group experimental gates,
   followed by the required 11-solvent, confirmation, and blind sets. Every
   record must satisfy the user's maximum-error rule; no MAE-only escape is
   valid.

Until all four steps pass, this branch is **not eligible** to report Route-2
solvation accuracy.
