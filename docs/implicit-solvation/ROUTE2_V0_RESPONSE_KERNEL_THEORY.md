# Route-2 V0-RK: constrained full density-response kernel

## Status and boundary

`route2_v0_response_kernel.py` implements two **algebraic** V0-RK
completions of a physical baseline density-response kernel: the historical
stacked-atomic completion is retained as a rank-three structural control, and
the molecular-moment completion constrains only the molecular three-component
moment. Neither has a checked-in physical kernel asset, source-bound GTO
basis, continuum/cavity, force calculation, or QM or experimental solvation
run. Its unit tests use only synthetic matrices. Therefore neither is a
**physical response-kernel asset**, a solvation method, or an accuracy result.

The machine-readable boundary for the historical stacked-atomic control is
[`route2-v0-response-kernel-completion-prereg-v1.json`](benchmarks/route2-v0-response-kernel-completion-prereg-v1.json).

The purpose is narrow but necessary. A molecular polarizability is a
three-dimensional moment of the true susceptibility; it does not identify the
near-field response density. V0-RK preserves an independently sourced
high-dimensional baseline response and imposes the frozen MACE-MDP moment only
as an exact low-rank constraint. It neither chooses a radial width nor fits a
solvation error.

## 1. Pairing and inputs

At fixed nuclear geometry, let \(x=\delta c\in\mathbb R^m\) be a neutral
induced density coefficient vector and \(f\) its declared coefficient-dual
external potential. The baseline physical response is written

\[
x=-C_0 f,
\qquad C_0=C_0^\mathsf T\succeq0,
\qquad C_0q=0,
\]

where \(q\) is the total-charge row. The last identity ensures that every
induced density is neutral without a projection or charge repair.

Let \(A\in\mathbb R^{3N\times m}\) map \(x\) to stacked atom-resolved dipoles
in one explicit \(e\,a_0\) convention. Let

\[
G=[I_3\;I_3\;\cdots\;I_3]\in\mathbb R^{3\times 3N}
\]

sum those atomic dipoles. The frozen audited MACE partition is

\[
W\in\mathbb R^{3N\times3},
\qquad GW=I_3,
\]

and its frozen MACE-MDP molecular polarizability is
\(\alpha_\theta=\alpha_\theta^\mathsf T\succ0\). The desired atomic and
molecular moment covariances are

\[
\Gamma=W\alpha_\theta W^\mathsf T,
\qquad
M=GA,
\qquad
MC M^\mathsf T=\alpha_\theta.
\]

All maps must use the same source/dual convention as the eventual continuum.
`A` and `W` cannot be inferred from a continuum grid, a response fit, or a
solvation result.

## 2. Stacked-atomic Schur-complement structural control

Require the source-bound baseline atom covariance

\[
S_0=AC_0A^\mathsf T\succ0.
\]

This strict condition is intentional: a singular \(S_0\) means that the
baseline does not resolve every atom-dipole degree of freedom being replaced.
Using a pseudoinverse there would silently invent a response in an unsupported
direction, so V0-RK rejects it.

Define the lifting map

\[
L=C_0A^\mathsf TS_0^{-1},
\qquad AL=I_{3N},
\]

and split the baseline response into the conditional atom-dipole-null part

\[
C_\perp=C_0-LS_0L^\mathsf T\succeq0.
\]

The completed response is

\[
\boxed{
C=C_\perp+L\Gamma L^\mathsf T
=C_0-C_0A^\mathsf TS_0^{-1}AC_0
+C_0A^\mathsf TS_0^{-1}\Gamma S_0^{-1}AC_0.}
\]

This is a constrained covariance replacement, not a learned correction. Its
properties follow directly:

\[
\begin{aligned}
C&\succeq0,\\
ACA^\mathsf T&=\Gamma,\\
MC M^\mathsf T
&=G\Gamma G^\mathsf T
=\alpha_\theta,\\
Cq&=0.
\end{aligned}
\]

The implementation checks each identity numerically before returning. It does
not symmetrize a material antisymmetry, clip an eigenvalue, rescale
\(\alpha_\theta\), or add a penalty to force a failed identity.

However, \(\operatorname{rank}(\Gamma)\leq3\).  Since

\[
AC=\Gamma L^\mathsf T,
\]

the atom-resolved response of this completion lies in the three-dimensional
range of \(W\), even under a nonuniform field.  It can preserve the audited
uniform-field molecular polarizability, but it must not be treated as a
general atom-resolved response.  The code therefore retains it as
`complete_route2_v0_response_kernel` for a structural control only.

## 3. Molecular-moment completion candidate

Let \(T=GA\in\mathbb R^{3\times m}\) be the total molecular-dipole map.  The
alternative completion replaces only the covariance observed by the frozen
MACE molecular polarizability:

\[
S_{\rm mol}=TC_0T^\mathsf T,
\qquad
L_{\rm mol}=C_0T^\mathsf TS_{\rm mol}^{-1},
\]

\[
\boxed{
C_{\rm mol}=C_0-L_{\rm mol}S_{\rm mol}L_{\rm mol}^\mathsf T
+L_{\rm mol}\alpha_\theta L_{\rm mol}^\mathsf T.}
\]

It exactly enforces
\(TC_{\rm mol}T^\mathsf T=\alpha_\theta\), preserves PSD and charge
neutrality, and retains the baseline conditional covariance outside the total
molecular-moment subspace.  It makes no unsupported claim that the full
\(3N\times3N\) atom-dipole covariance equals \(W\alpha_\theta W^\mathsf T\).
Thus it avoids the rank-three collapse but does not create molecular bonding
response or charge transfer.  Its first frozen acetone QM-MEP oracle is a
structural/source observation only; it cannot select a kernel by a solvation
error or establish transferability.  Broad nonuniform finite-field QM gates
remain required.

## 4. One electronic scalar on the response support

For the positive support of \(C\), let \(C^+\) be the Moore--Penrose inverse
and \(\Pi_C=CC^+\) the support projector. The gas electronic scalar is

\[
F_{\rm RK}(x;\mathbf R)
=E_{\rm gas}(\mathbf R)+\frac12x^\mathsf TC^+x,
\qquad x\in\operatorname{Ran}C.
\]

Its Euler equation under an external dual is

\[
C^+x+\Pi_Cf=0,
\qquad x=-Cf.
\]

Thus the response is automatically reciprocal and passive in the declared
coefficient/dual pairing:

\[
\frac{\partial x}{\partial f}=-C
=\left(\frac{\partial x}{\partial f}\right)^\mathsf T\preceq0.
\]

The generic KKT already imposes the total-charge row \(q^\mathsf{T}x=0\).
The kernel module therefore returns only the remaining exact zero-mode rows
\(N^\mathsf{T}x=0\), chosen so that the rows of
\(\begin{bmatrix}q^\mathsf{T}\\N^\mathsf{T}\end{bmatrix}\) span
\(\ker C\) without a duplicate charge constraint. They are never assigned an
arbitrary large curvature.
This output can later enter a common scalar only after a reciprocal physical
continuum and a separately provenance-bound permanent source have passed their
independent gates.  A frozen permanent source need not be represented in the
induced \(x\) coefficient basis: it may enter directly as a total surface
potential \(v_0\), provided \(B\) and \(B^\mathsf T\) remain exact duals in
the induced coefficient/receiver pairing.  This narrower frozen-source
statement is not a claim of a complete stationary permanent electronic state;
see [ROUTE2_V0_FROZEN_SOURCE_KKT_BOUNDARY.md](ROUTE2_V0_FROZEN_SOURCE_KKT_BOUNDARY.md).

With a reciprocal continuum response \(Q=Q^\mathsf T\), the future common scalar is

\[
\mathcal G(x;\mathbf R)=E_{\rm gas}(\mathbf R)
+\frac12x^\mathsf TC^+x
+\frac12(v_0+B_{\mathbf R}x)^\mathsf TQ_{\mathbf R}(v_0+B_{\mathbf R}x),
\quad x\in\operatorname{Ran}C,\quad q^\mathsf Tx=0.
\]

Its stability condition is the positive definiteness of
\(\Pi_C(C^++B^\mathsf TQB)\Pi_C\) on the allowed response support, not an SCF mixing
spectral radius.

## 5. What a physical source must prove

An admissible \(C_0\) must be fixed **before** target solvation labels are
read and must carry all of the following:

1. a source document and immutable digest for its electronic-response data,
   basis, nuclear geometry convention, and coefficient/dual pairing;
2. charge-neutrality, positive-semidefinite, atom-covariance, source/receiver
   duality, and rigid-motion certificates;
3. preregistered gas-phase QM finite-field density, MEP, induced-dipole, and
   reaction-potential gates over the twelve-record, ten-actual-functional-group
   physics panel;
4. a production runtime path. Per-molecule runtime QM response may qualify an
   oracle but cannot be called a faster Route-2 endpoint unless the exact
   end-to-end runtime gate is met;
5. exact coordinate derivatives of \(C_0,A,W,\alpha_\theta\), the induced
   source map, the permanent surface potential \(v_0\), and the continuum for
   envelope-force and PES certification.

No currently checked-in artifact meets these requirements. In particular,
free-atom V0-ADT is a successful **rank-three source** canary, not the full
positive-semidefinite \(C_0\) required here.

## 6. Relation to literature and custom solvents

The response-kernel construction addresses the **solute** side. It does not
make a solvent model physical by itself. The 2026 self-consistent-MLIP design
space explicitly distinguishes energy-functional and fixed-point
architectures; V0-RK follows the former by defining response as a Hessian
inverse rather than a learned iteration. [Baldwin *et al.*](https://arxiv.org/abs/2603.14700)

On the solvent side, JDFT/PCM literature supports a common scalar and
nonlocal response, but not a dielectric-only accuracy shortcut:

* [Universal iso-density PCM](https://arxiv.org/abs/1403.6465) derives
  multi-solvent parameters from single-molecule calculations plus bulk data;
  it is a source/provenance pattern, not permission to copy a threshold.
* [SaLSA](https://arxiv.org/abs/1410.2273) derives nonlocal dielectric response
  from a liquid functional, but its published total model includes a fitted
  dispersion contribution and therefore cannot be imported as a V0 total
  ledger.
* The official [JDFTx `fluid-solvent` documentation](https://jdftx.org/CommandFluidSolvent.html)
  exposes custom-solvent inputs including density/concentration, static and
  optical dielectric response, molecular dipole, vapour pressure, surface
  tension, and radii. This confirms that \(\epsilon_0\) alone is insufficient
  for the requested custom-solvent total free energy.

Accordingly, V0-RK remains below the physical-continuum and accuracy gates
until the required independent solute and solvent source assets exist.
