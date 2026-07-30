# Route-2 V0 frozen MACE-MDP moment-functional branch

## Status and boundary

This document defines the only new V0 electronic branch admitted by the
frozen MACE-MDP acetone coefficient screen. It is a **theory and admission
contract**, not a public Route-2 calculator, a total-solvation result, or an
accuracy claim.

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
\(6.41	imes10^{-17}\), atomic polarizability closure is
\(1.07	imes10^{-16}\), and the Cartesian partition
\(W_a=lpha_alpha^{-1}\) satisfies \(\sum_aW_a=I\) to
\(4.06	imes10^{-17}\).  Thus \(p_a=W_ap\) is an identity-bound atomic
induced-dipole partition.  It does **not** yet specify how those atom moments
become a radial GTO density or an electrostatic source.

## One explicit scalar

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

The **radial density representer** in \(D_{\mathbf R}\) remains unresolved;
the atomic Cartesian partition is no longer guessed.  A point dipole, a
guessed Gaussian width, an error-selected atomic partition, or a projection
that changes after viewing solvation errors is forbidden. Any implementation
must use the frozen \(p_a=W_ap\) partition above and provide a
source-provenanced, geometry-differentiable GTO representer satisfying all of
the following before it is connected to a PCM:

1. **Charge and moment identity:** \(q^\mathsf TD=0\) and
   \(M_{\mathbf R}D=I_3\) at float64 precision.
2. **Energy duality:** surface coupling is evaluated only as
   \(\sigma^\mathsf TB Dp=p^\mathsf TD^\mathsf TB^\mathsf T\sigma\), using the
   same `B` and transpose as the permanent source.
3. **Euclidean covariance:** translation leaves the induced neutral density's
   dipole invariant, and a rigid rotation covaries both \(D\) and \(p\).
4. **No hidden response correction:** a failed raw MACE-MDP tensor cannot be
   made eligible by clipping, shifting, rescaling, or a source-map choice.
5. **Physical validation:** the representer's vacuum potential, reaction
   potential, and induced density/moment must be compared with preregistered
   QM quantities before it is allowed to influence a solvation result.

The existing same-basis GTO Galerkin operator is useful because it already
enforces \(\sigma^\mathsf TBc=c^\mathsf TB^\mathsf T\sigma\). It is not by
itself a valid \(D\): its current MACE density embedding preserves a frozen
`l<=1` density but does not prescribe how a molecular induced dipole is
distributed across that density.

## Force and performance gates

At a joint stationary state, the force must be the envelope derivative of the
same scalar:

\[
-\frac{d\mathcal L}{dR_I}
=-\left.\frac{\partial\mathcal L}{\partial R_I}\right|_{p^*,\sigma^*}.
\]

It therefore includes explicit derivatives of \(E_{\rm gas}\),
\(\alpha_\theta^{-1}\), \(c_0\), \(D\), \(B\), and \(A\). The following are
mandatory before a force or PES claim:

* frozen-model coordinate derivatives of \(\alpha_\theta\) checked against
  central differences;
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

1. Pre-register and implement a non-arbitrary \(D_{\mathbf R}\) with the
   identities above; do not select it from experimental solvation errors.
2. Couple it to the existing reciprocal GTO/continuum primitive and demonstrate
   \(K_p\succ0\), one energy ledger, and envelope forces.
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
