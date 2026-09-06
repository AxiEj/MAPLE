# Decision

The queued seven-combination decomposition is worth running, but it is **not by itself a causal attribution experiment**. It will tell you how the three *already-converged, frozen* source distributions interact through the ddPCM quadratic form. It cannot, without matched component references, decide whether a large term reflects a bad source, a bad response, a bad cavity, omitted electronic polarization work, or legitimate interaction between two components.

The arithmetic (UG) split is an exact direct sum only in the **native receiver-coordinate space**. It is neither guaranteed to be a direct sum of physical source distributions nor orthogonal under the continuum reaction-field energy. A geometry-dependent physical projector is mathematically legitimate and target-free, but it is a **new solvent/cavity-dependent model**, not a harmless coordinate change. It should not be implemented merely because (C_{ra}) is large or because the three experimental sentinels look bad.

The decisive next implementation should be a **matched-QM permanent/induced cavity-boundary potential-response test**, not a projector and not another 505-record run.

---

# 0. Separate the fixed-point operator from the energy operator

The symbol (K_R) is currently doing two conceptually different jobs:

[
u=T_R s,
\qquad
s=p+r(u)+a(u),
]

where (T_R) maps a physical source to the native eight-channel receiver field, and

[
G=\frac12\langle s,H_Rs\rangle
]

where (H_R) maps a source to the reaction potential dual to that source.

Unless the native receiver features are exactly the dual coordinates used in the source–potential pairing, (T_R) and (H_R) are not the same matrix. This matters because (T_R) need not be square or self-adjoint, whereas only the symmetric part of (H_R) enters a quadratic energy.

For a passive linear dielectric, define the continuum-active source metric

[
W_R=-\operatorname{sym}H_R
==========================

-\frac12(H_R+H_R^\dagger)
\succeq 0.
]

Then

[
G=-\frac12|s|*{W_R}^{2},
\qquad
|x|*{W_R}^{2}:=\langle x,W_Rx\rangle .
]

The standard PCM electrostatic energy is the half-work (E_s=\frac12\langle \rho,\psi_r\rangle); domain-decomposition PCM implementations likewise differentiate the discrete energy through primal and adjoint solves. 

Because the branches use different representations—point multipoles, Gaussian multipoles, and finite-density ADT tangents—one should more explicitly write physical source embeddings

[
s=B_pp+B_rr+B_aa
]

and block metrics

[
W_{ij}=B_i^\dagger W_RB_j.
]

Raw Euclidean norms of (p,r,a) are not comparable. Their coefficients have different radial shapes and, potentially, different normalizations. The meaningful common norms are the continuum-active (W_R) norms, boundary-potential norms, or induced surface-charge norms.

A second crucial limitation is that (G) is stationary with respect to the **linear continuum variables**, not with respect to the induced ML sources. In a conventional variational polarizable model one would have something like

[
\mathcal E(x)
=============

\Phi_{\rm int}(x)
+
\frac12
\langle p+B x,H_R(p+Bx)\rangle ,
]

where (\Phi_{\rm int}(x)\ge 0) is the internal electronic distortion or polarization cost. Your operational ledger contains only the second term. Because no common source–energy functional exists, there is no identity showing that the omitted (\Phi_{\rm int}) is already encoded in (G).

Therefore:

[
E_r,\ E_a,\ C_{pr},\ldots
]

are **continuum half-work components of frozen sources**, not electronic distortion energies and not thermodynamic induction energies.

This does not invalidate the operational scalar. It sharply limits the physical labels that may be attached to its components.

---

# 1. Exact meaning of the six terms and the response Jacobian

## 1.1 Frozen-root quadratic decomposition

At the converged root (u^\star), set

[
s_p=p,\qquad
s_r=r(u^\star),\qquad
s_a=a(u^\star).
]

For each frozen source,

[
E_i
===

-\frac12|s_i|_{W_R}^{2}
\le 0.
]

For two branches,

[
\begin{aligned}
C_{ij}
&=
E_{i+j}-E_i-E_j\
&=
-\langle s_i,W_Rs_j\rangle .
\end{aligned}
]

Consequently,

[
G
=

E_p+E_r+E_a+C_{pr}+C_{pa}+C_{ra}.
]

This gives several exact internal checks.

First, passivity requires

[
E_p,E_r,E_a\le 0
]

up to solver/discretization error. A positive self term is not evidence of an unusual molecule; it indicates a sign, pairing, source-map, or operator problem.

Second, Cauchy–Schwarz requires

[
|C_{ij}|
\le
2\sqrt{(-E_i)(-E_j)}.
]

Define the continuum-active correlation

[
\gamma_{ij}
===========

\frac{-C_{ij}}
{2\sqrt{(-E_i)(-E_j)}}.
]

Then, on the active quotient,

[
-1\le \gamma_{ij}\le 1.
]

Its interpretation is:

[
\gamma_{ij}\simeq +1:
\quad
\text{sources produce nearly parallel reaction potentials;}
]

[
\gamma_{ij}\simeq -1:
\quad
\text{sources are nearly cancelling;}
]

[
\gamma_{ij}\simeq 0:
\quad
\text{energy-orthogonal for this particular state.}
]

A convenient symmetric allocation of the total half-work is

[
L_i
===

# E_i+\frac12\sum_{j\ne i}C_{ij}

-\frac12\langle s_i,W_Rs\rangle ,
]

with

[
L_p+L_r+L_a=G.
]

An (L_i) can be positive even though (E_i\le0), because another branch may oppose it. (L_i) is therefore an allocation, not an intrinsic energy.

Most importantly, the decomposition is evaluated at the **full hybrid root**. The residual (r(u^\star)) already contains response to (p), (a), and itself. Thus:

* (E_r) is not the result of solving a residual-only model.
* (E_a) is not an ADT-only induction energy.
* (C_{ra}) is not automatically “double-counting energy.”
* Re-solving each subset would produce causal counterfactuals, but then the six-term closure would no longer be an exact quadratic identity.

## 1.2 The causal object is the fixed-point Jacobian

Let

[
J_\star
=======

D_uM_{\rm POLAR}(u^\star),
\qquad
J_0
===

D_uM_{\rm POLAR}(0),
\qquad
P=UG.
]

The residual tangent at the operating point is

[
R_\star
=======

# D_ur(u^\star)

J_\star-J_0P.
]

Let the linear ADT source map be

[
A_0e
====

P_{\rm ADT}[-\alpha_{\rm MDP}e],
]

so

[
A_\star
=======

# D_ua(u^\star)

A_0G.
]

The receiver-space loop Jacobian is

[
\mathcal J_u
============

T_R(R_\star+A_\star).
]

For a perturbation in the permanent source or an externally applied receiver field (h),

[
(I-\mathcal J_u),\delta u
=========================

T_R\delta p+\delta h,
]

hence

[
\delta u
========

(I-\mathcal J_u)^{-1}
(T_R\delta p+\delta h).
]

The associated source-space loop is

[
\mathcal L_s
============

(R_\star+A_\star)T_R.
]

The nonzero eigenvalues of (\mathcal J_u=T_RS_\star) and (\mathcal L_s=S_\star T_R) coincide. The source-space form is preferable for physical conditioning because it can be whitened by (W_R):

[
\widetilde{\mathcal L}_s
========================

W_R^{1/2}\mathcal L_sW_R^{-1/2}
]

on the continuum-active quotient.

The relevant amplification is not merely (\sigma_{\max}(\mathcal J_u)). It is

[
\mathcal A_{\rm loop}
=====================

\left|
(I-\widetilde{\mathcal L}_s)^{-1}
\right|
=======

\frac{1}
{\sigma_{\min}(I-\widetilde{\mathcal L}_s)}.
]

This distinction matters for non-normal maps. One can have

[
\rho(\mathcal L_s)<1
]

but a very large resolvent because (I-\mathcal L_s) is nearly singular in a non-normal direction.

Raw Euclidean singular values in the eight-channel native receiver space are coordinate dependent. The two radial channels and scalar/vector channels need not share a physically meaningful Euclidean scale. Report either the source-metric-whitened loop above or a separately justified receiver Gram matrix.

For a near-critical right/left eigenpair,

[
\mathcal J_uv=\lambda v,
\qquad
w^\dagger\mathcal J_u=\lambda w^\dagger,
\qquad
w^\dagger v=1,
]

the branch contributions are

[
\lambda_r=w^\dagger T_RR_\star v,
\qquad
\lambda_a=w^\dagger T_RA_\star v,
\qquad
\lambda=\lambda_r+\lambda_a.
]

This is far more discriminating than a total singular value.

## 1.3 Exact diagnostic patterns

### A. Bad permanent MDP near-field source

A bad (p) does not enter the Jacobian directly:

[
D_up=0.
]

It changes the fixed-point operating field (u^\star), and therefore can move the nonlinear residual into a high-gain region, but it is not itself a feedback tangent.

The characteristic pattern would be:

[
|E_p|\ \text{or}\ |p|_{W_R}
\quad\text{abnormal relative to a matched permanent source,}
]

while

[
R_\star,\ A_\star,\
\sigma_{\min}(I-\widetilde{\mathcal L}_s)
]

remain ordinary. The induced branches may then show large (C_{pr}) and (C_{pa}) simply because they are responding correctly to an incorrect permanent field.

What proves A is not a large (E_p), but:

[
V_{\partial\Omega}[p_{\rm MDP}]
-------------------------------

V_{\partial\Omega}[\rho^{\rm QM}_{0}]
]

being large in the same frozen cavity, or a large difference in their ddPCM half-works.

Without a matched permanent density or boundary-potential reference, A and D are not identifiable from (E_p).

### B. Excessive MACE-POLAR residual response

The strong pattern is:

[
-E_r\ \text{large},
\qquad
C_{pr}\ll0,
\qquad
C_{ra}\ll0,
]

together with a large open-loop residual tangent

[
|W_R^{1/2}R_\star h|
]

under a fixed imposed physical field (h), and a dominant near-critical modal contribution

[
\lambda_r\simeq \lambda,
\qquad
|\lambda_a|\ll|\lambda_r|.
]

The discriminating separation from nonlinear feedback is:

* **Intrinsic residual excess:** (R_\star h) is already too large against matched-QM response while the resolvent is benign.
* **Feedback excess:** the open-loop (R_\star h) is plausible, but multiplication by ((I-\mathcal J_u)^{-1}) makes the closed-loop response excessive.

A large negative (E_r) alone does not prove excessive response because the hybrid has no explicit positive cost for creating (r).

### C. Residual/ADT overlap or double counting

A state-level overlap signature is

[
\gamma_{ra}\simeq +1
]

with both (-E_r) and (-E_a) substantial. Equivalently,

[
C_{ra}
\simeq
-2\sqrt{(-E_r)(-E_a)}.
]

That means that the two frozen sources generate nearly the same continuum-active reaction potential. It still does not prove that both sources represent the same physical susceptibility.

The tangent-level test is stronger. Let

[
\mathcal S_r=\operatorname{ran}(W_R^{1/2}R_\star),
\qquad
\mathcal S_a=\operatorname{ran}(W_R^{1/2}A_\star).
]

Compute their principal angles after removing exact null/gauge modes. Double-response evidence is:

1. one or more principal-angle cosines close to one;
2. both branches contribute positively to the same near-unit loop mode,
   [
   \lambda_r>0,\quad\lambda_a>0,\quad
   \lambda_r+\lambda_a\simeq1;
   ]
3. the combined induced dipole or cavity-boundary response exceeds matched QM, although the two individual responses are individually plausible.

There is also an exact nonlinear leakage observable. Since

[
R_\star U
=========

(J_\star-J_0)U,
]

the POLAR residual re-acquires a uniform-field tangent whenever the MACE-POLAR Jacobian changes between zero field and (u^\star). Thus the zero-field subtraction guarantees removal of the uniform tangent only at (u=0). If

[
|(J_\star-J_0)U|_{W_R}
]

is large, the residual and ADT branches overlap even on the nominally protected uniform subspace at the operating root.

### D. Cavity or continuum-operator problem

A numerical/operator defect has exact internal symptoms:

[
E_i>0,
\qquad
|\gamma_{ij}|>1,
]

failure of seven-way closure, adjoint inconsistency, lack of discretization convergence, or failure on analytic spherical-cavity point-charge/dipole tests.

A physically inadequate but mathematically passive cavity need not show any of those symptoms. All self terms can be negative, all Cauchy bounds can pass, and the result can still be chemically wrong.

A cavity problem becomes identifiable only by source swapping:

* hold the physical source fixed and change only the independently validated continuum realization;
* or put a matched QM source into the frozen ddPCM cavity and see whether the abnormal stabilization remains.

If a matched QM source produces the same anomalous size/conjugation trend in ddPCM, the source branches are exonerated and D becomes plausible.

### E. Nonlinear fixed-point feedback amplification

The decisive patterns are:

[
\sigma_{\min}(I-\widetilde{\mathcal L}_s)\ll1,
]

large resolvent norm, strong superlinearity along a coupling path, root sensitivity, or forward/reverse hysteresis.

A near-(+1) eigenvalue is the dangerous positive-feedback case:

[
\lambda\simeq +1
\quad\Longrightarrow\quad
(I-\mathcal J_u)^{-1}
\ \text{large}.
]

A near-(-1) eigenvalue may make simple fixed-point iteration oscillatory, but it does not by itself create a large static response because

[
(1-(-1))^{-1}=\frac12.
]

Therefore “largest singular value near one” is not enough. Record:

[
\rho(\mathcal J_u),\quad
\sigma_{\min}(I-\widetilde{\mathcal L}_s),\quad
|(I-\widetilde{\mathcal L}_s)^{-1}|,
]

and branch-resolved eigenmode contributions.

A standard diagnostic coupling parameter is

[
u_\lambda
=========

T_R\left[p+\lambda{r(u_\lambda)+a(u_\lambda)}\right],
\qquad
0\le\lambda\le1.
]

Then

[
\frac{du_\lambda}{d\lambda}
===========================

\left[
I-\lambda T_R(R_\lambda+A_\lambda)
\right]^{-1}
T_R[r(u_\lambda)+a(u_\lambda)].
]

Collapse of the smallest singular value as (\lambda\to1), accompanied by rapidly growing source norms, is direct evidence of feedback amplification. Damping and smeared-dipole constructions are standard precisely because unconstrained interactive polarizability models can develop divergent polarization. ([Groningen Research Portal][1])

---

# 2. Is the arithmetic (UG) split sufficient?

## 2.1 It is an exact algebraic direct sum in receiver space

Let

[
P=UG,\qquad Q=I-P,
\qquad GU=I_3.
]

Then

[
P^2=UGUG=UG=P,
]

and

[
GQ=G-GUG=0.
]

Therefore

[
\operatorname{ran}P=\operatorname{ran}U,
\qquad
\operatorname{ran}Q=\ker G,
]

and every native field has the unique decomposition

[
u=UGu+Qu.
]

Moreover,

[
\operatorname{ran}U\cap\ker G={0},
]

because (Ue\in\ker G) implies

[
e=GUe=0.
]

Thus

[
\mathcal U
==========

\operatorname{ran}U
\oplus
\ker G.
]

That is a correct algebraic statement.

## 2.2 It does not imply a physical source direct sum

At zero field,

[
\Delta M(u)
===========

J_0u+\frac12M^{(2)}[u,u]+\cdots .
]

The split gives

[
r(u)
====

J_0Qu
+
\frac12M^{(2)}[u,u]
+\cdots ,
]

and

[
a(u)=A_0Gu.
]

The linear source subspaces are therefore

[
\mathcal C_r=J_0(\ker G),
\qquad
\mathcal C_a=A_0\mathbb R^3.
]

Nothing in (GU=I) implies

[
\mathcal C_r\cap\mathcal C_a={0}.
]

There can exist (q\in\ker G) and (e\ne0) such that

[
J_0q=A_0e.
]

Even when the intersection is zero, the two spaces need not be orthogonal under (W_R).

The linearized residual/ADT cross work is

[
\begin{aligned}
-\langle J_0Qu,W_RA_0Gu\rangle
&=
-u^\dagger
Q^\dagger J_0^\dagger W_RA_0G
u.
\end{aligned}
]

There is no algebraic identity making

[
A_0^\dagger W_RJ_0Q
]

zero.

Hence an arithmetic decomposition can produce a systematic negative (C_{ra}) whenever the continuum fields sampled by the molecule make the two source images coherently aligned.

## 2.3 Nonlinear overlap remains even on exact uniform fields

For a purely uniform native input (u=Ue),

[
Qu=0,
]

but

[
r(Ue)
=====

# M(Ue)-M(0)-J_0Ue

\frac12M^{(2)}[Ue,Ue]+\cdots .
]

Thus:

* the **linear zero-field molecular polarizability** is replaced by ADT;
* finite-field nonlinear MACE-POLAR uniform response remains;
* mixed uniform/nonuniform Hessian terms also remain.

The current split therefore does not define a globally uniform/nonuniform decomposition of the nonlinear response.

## 2.4 Orthogonality is not automatically the physical truth

A nonzero (C_{ra}) can be entirely legitimate. Two physically distinct pieces of an induced density generally interact electrostatically. Forcing

[
C_{ra}=0
]

by construction may erase real cooperative induction.

The physically relevant question is not “are the branches orthogonal?” but:

> Do the branches describe complementary parts of the matched-QM response, or do they each reproduce the same response mode?

That requires a matched induced-density or boundary-response reference.

A geometry-dependent (W_R)-based projector is therefore:

* mathematically well-defined under rank conditions;
* target-free in the narrow sense that it uses no solvation labels;
* a legitimate new operational model if frozen in advance;
* **not** evidence-free, because choosing it after observing these three experimental errors would be target-informed architecture selection;
* solvent/cavity dependent if (W_R) contains the named solvent dielectric and cavity.

It is not a disguised numerical fit, but it is not a mere change of coordinates either.

---

# 3. Minimal physically motivated projector

Let

[
J=J_0,
\qquad
B=JU,
\qquad
A=A_0,
]

where (B) is the MACE-POLAR uniform-field source tangent and (A) is the ADT source tangent with the desired MDP molecular polarizability.

All formulas below are on the continuum-active quotient after exact gauge removal.

## 3.1 Receiver-space metric projector

Given a positive receiver metric (M), the (M)-orthogonal projection onto (\operatorname{ran}U) is

[
G_M
===

(U^\dagger MU)^{-1}U^\dagger M,
]

[
P_M=UG_M,
\qquad
Q_M=I-P_M.
]

A natural pullback metric from the MACE tangent and continuum energy is

[
M_J=J^\dagger W_RJ.
]

Then

[
B^\dagger W_RJQ_M=0.
]

This makes the residual MACE response orthogonal to the **MACE-POLAR uniform tangent** (B), not necessarily to ADT:

[
A^\dagger W_RJQ_M
]

need not vanish.

Therefore a receiver-space orthogonal projector is sufficient only if

[
\operatorname{ran}(W_R^{1/2}A)
==============================

\operatorname{ran}(W_R^{1/2}B).
]

That equivalence should be checked using continuum-active principal angles.

## 3.2 Source-response tangent projector

Define the cross Gram matrix

[
\Gamma
======

# A^\dagger W_RB

A^\dagger W_RJU.
]

If (\Gamma) is invertible, define

[
\Pi_{B|A}
=========

B\Gamma^{-1}A^\dagger W_R.
]

Then

[
\Pi_{B|A}^{2}=\Pi_{B|A},
]

[
\operatorname{ran}\Pi_{B|A}
===========================

\operatorname{ran}B,
]

and

[
A^\dagger W_R(I-\Pi_{B|A})=0.
]

It removes the MACE uniform tangent and leaves the residual source (W_R)-orthogonal to ADT.

This is generally an **oblique** projector. It becomes the ordinary (W_R)-orthogonal projector only when

[
\operatorname{ran}A=\operatorname{ran}B
]

in the active metric. Otherwise it can increase norms and worsen conditioning.

## 3.3 Minimal continuum-active chart projector

Within the existing architecture

[
r=M-M_0-JUGu,
\qquad
a=AGu,
]

the minimal modification is to replace (G) by

[
\boxed{
G_K
===

\left(A^\dagger W_RJU\right)^{-1}
A^\dagger W_RJ
}
]

provided the (3\times3) cross Gram matrix is invertible.

It satisfies

[
G_KU=I_3,
]

and, with

[
Q_K=I-UG_K,
]

[
A^\dagger W_RJQ_K=0.
]

The linearized induced source is

[
\chi_K
======

# JQ_K+AG_K

J+(A-JU)G_K.
]

On a uniform field,

[
\chi_KU=A,
]

so the MDP molecular polarizability closure is retained.

This is the unique chart of the form (P=UG_K) satisfying both

[
G_KU=I
]

and

[
A^\dagger W_RJ(I-UG_K)=0.
]

It is equivalent to the source tangent projector because

[
JQ_K
====

(I-\Pi_{B|A})J.
]

### Required conditions

**Invertibility.**
The cross Gram

[
\Gamma=A^\dagger W_RJU
]

must have rank three. If a Cartesian direction is continuum-inactive or the two uniform source subspaces are orthogonal in one direction, there is no valid three-dimensional projector. That is a fail-closed result, not a reason to regularize.

**Gauge treatment.**
If (W_R) is semidefinite, first quotient exact continuum-null modes. A Moore–Penrose inverse is legitimate only for a known, fixed physical gauge nullspace with constant rank. It must not be used to rescue an empirically rank-deficient projector.

**Conditioning.**
An invertible but nearly singular (\Gamma) produces a large oblique projector. That is evidence that ADT and the removed POLAR uniform source do not represent the same physical response subspace. The correct decision is then termination of the hybrid split, not clipping the singular values.

**SO(3) covariance.**
Suppose under a rotation (Q),

[
U(QR)
=====

\mathcal R_U(Q)U(R)Q^\dagger,
]

[
J(QR)
=====

\mathcal R_C(Q)J(R)\mathcal R_U(Q)^\dagger,
]

[
A(QR)
=====

\mathcal R_C(Q)A(R)Q^\dagger,
]

and (W_R) transforms covariantly. Then

[
\Gamma(QR)=Q\Gamma(R)Q^\dagger,
]

[
G_K(QR)
=======

QG_K(R)\mathcal R_U(Q)^\dagger,
]

and

[
P_K(QR)
=======

\mathcal R_U(Q)P_K(R)\mathcal R_U(Q)^\dagger.
]

Thus the exact construction is SO(3)-covariant. Any finite-grid covariance error in (W_R), the source maps, or the receiver map is inherited by the projector.

**Differentiability.**
(G_K(R)) is differentiable if (J(R)), (A(R)), and (W_R(R)) are differentiable and

[
\sigma_{\min}\Gamma(R)
]

stays bounded away from zero. The inverse derivative is

[
d\Gamma^{-1}
============

-\Gamma^{-1}(d\Gamma)\Gamma^{-1}.
]

Rank crossings or non-smooth cavity masks destroy global differentiability.

**Passivity.**
The passive continuum still guarantees

[
\frac12s^\dagger H_Rs\le0
]

for each frozen source. It does not guarantee that the projected induced response is a passive static susceptibility. Since (G_K) is generally oblique, verify separately:

[
\alpha_{\rm eff}
================

\alpha_{\rm eff}^\dagger
]

within numerical tolerance and

[
\operatorname{sym}\alpha_{\rm eff}\succeq0
]

for uniform fields, together with nonnegative induced work for the physical nonuniform probe fields.

### Does it preserve the original response on (\ker G)?

No, unless it is the original (G).

Suppose (G_K) and (G) are both left inverses of (U) and one requires

[
G_Kq=0
\qquad
\text{for every }q\in\ker G.
]

Then

[
\ker G\subseteq\ker G_K.
]

Both kernels have codimension three, so they are equal. For any

[
u=Ue+q,\qquad q\in\ker G,
]

one then has

[
G_Ku=e=Gu.
]

Therefore

[
G_K=G.
]

So any nontrivial physical chart necessarily changes the MACE-POLAR response for at least some fields that the old arithmetic chart called “nonuniform.”

### Projector decision

A physical projector is warranted only if all of the following target-free observations occur:

1. (A) and (JU) represent the same three-dimensional continuum-active physical response subspace;
2. the arithmetic residual has substantial (W_R)-overlap with (A);
3. matched-QM response shows that the combined hybrid response is excessive while the two parent responses are individually plausible;
4. (\Gamma) is well-conditioned and rank three;
5. the projected response remains reciprocal, passive, SO(3)-covariant, and fixed-point stable.

A large (C_{ra}) alone is not enough.

---

# 4. Can ADT allocation cause size- or conjugation-dependent overpolarization?

Yes. Exact positive molecular polarizability does not determine the spatial induced density.

Let (d) denote atomic translation amplitudes, let (Q_{\rm FA}\succ0) be the free-atom self-work metric, and let (C_\mu d=\mu_{\rm ind}) impose the molecular induced dipole. The ADT minimizer is

[
d^\star
=======

Q_{\rm FA}^{-1}C_\mu^\dagger
\left(
C_\mu Q_{\rm FA}^{-1}C_\mu^\dagger
\right)^{-1}
\mu_{\rm ind}.
]

For

[
\mu_{\rm ind}=-\alpha_{\rm MDP}e,
]

this gives the ADT source map (A).

The minimization is canonical under (Q_{\rm FA}), but ddPCM sees (W_R), not (Q_{\rm FA}).

Let another allocation be

[
\widetilde A=A+N
]

with

[
C_\mu N=0.
]

Then (A) and (\widetilde A) have exactly the same molecular polarizability, but

[
\begin{aligned}
\widetilde A^\dagger W_R\widetilde A
------------------------------------

A^\dagger W_RA
&=
A^\dagger W_RN
+
N^\dagger W_RA
+
N^\dagger W_RN.
\end{aligned}
]

There is no reason for this difference to vanish.

The nullspace

[
\ker C_\mu
]

contains all intramolecular redistributions whose total dipole is zero. Its dimension grows with molecular size. Those modes can be highly active at the cavity boundary even though they do not alter the molecular dipole. A large conjugated molecule therefore provides many more ways for an allocation to have the correct total (\alpha) but an excessive cavity-active near field.

There are three separate ADT failure possibilities:

1. **The MDP total (\alpha) itself is too large.**
2. **The total (\alpha) is accurate, but the ADT spatial allocation is wrong.**
3. **The allocation is reasonable for uniform fields, but the arithmetic (Gu) extracts an excessive pseudo-uniform component from a nonuniform reaction field.**

These should not be conflated.

## Target-free falsification observable

The decisive observable is not an atomwise polarizability partition. It is the finite-field induced potential on the cavity boundary.

For three weak Cartesian fields (e_k), obtain matched-QM induced densities

[
\delta\rho^{\rm QM}_k
]

and corresponding boundary potentials. Compare them to

[
s^{\rm ADT}_k=Ae_k
]

after confirming that both have the same total induced dipole.

Use the continuum-active discrepancy

[
\epsilon_{\rm ADT}^{2}
======================

\frac{
\sum_{k=1}^{3}
|
s^{\rm ADT}*k-\delta\rho^{\rm QM}*k
|*{W_R}^{2}
}{
\sum*{k=1}^{3}
|
\delta\rho^{\rm QM}*k
|*{W_R}^{2}
}.
]

Equivalently, compare the reaction-response tensors

[
A^\dagger W_RA
]

and

[
\chi_{\rm QM}^\dagger W_R\chi_{\rm QM}.
]

If the molecular (\alpha) agrees but the large conjugated sentinel has a much larger ADT cavity-active response than QM, the allocation is falsified.

## Atom-resolved leverage is a screening diagnostic, not truth

For a field (e), write

[
Ae=\sum_A s_A(e).
]

Define atomwise active self-leverage

[
\ell_A(e)
=========

|s_A(e)|_{W_R}^{2}
]

and participation ratio

[
{\rm PR}(e)
===========

\frac{
\left(\sum_A\ell_A(e)\right)^2
}{
\sum_A\ell_A(e)^2
}.
]

Also examine the pair Gram matrix

[
K_{AB}(e)
=========

\langle s_A(e),W_Rs_B(e)\rangle .
]

A collapse in PR, extreme surface-exposed atomic leverage, or superlinear growth of the dominant eigenvalue of (K_{AB}) with conjugated size would support an ADT localization problem. It does not by itself prove one.

An MDP atomwise (\alpha) decomposition may be used as an internal sensitivity comparison, but not as reference truth unless that decomposition is separately gauge-fixed and physically supervised. Agreement between two latent atomic partitions is not validation.

---

# 5. What the sign-changing hybrid-minus-native discrepancy proves

The required multiplicative factors to make hybrid (G) equal the native electrostatic values are

[
s_{\rm urea}
============

# \frac{-16.0861}{-11.6836}

1.3768,
]

[
s_{\rm uracil}
==============

# \frac{-18.8704}{-16.8012}

1.1232,
]

[
s_{\rm tetra}
=============

# \frac{-20.8691}{-28.7509}

0.7259.
]

To match experiment after retaining stock CDS, the required factors are

[
1.4792,\qquad
1.2977,\qquad
0.4891.
]

Therefore the three records prove that no single positive multiplicative factor applied directly to the reported hybrid electrostatic scalar can exactly align all three with either comparator.

They also prove that a correction which moves every molecule in the same signed direction cannot improve all three simultaneously:

* urea and uracil need more negative electrostatics relative to the native electrostatic lane;
* the large sentinel needs much less negative electrostatics;
* a universally positive added distortion worsens the first two relative to the native electrostatic comparison while helping the third;
* removing positive CDS helps some totals but makes the already over-stabilized sentinel worse.

Under a fixed-source passive linear continuum, a scalar increase in source amplitude or dielectric screening moves all electrostatic half-works in the same stabilizing direction, so that class is directly contradicted.

What is **not** proved:

1. A scalar inserted inside the nonlinear fixed point is not rigorously ruled out solely by these signs unless monotonicity of every converged branch with respect to that scalar is established.
2. A single physical algorithmic change can still produce chemistry-dependent shifts of either sign.
3. The discrepancy does not identify (p), (r), (a), the cavity, or feedback.
4. It does not prove double counting.
5. It does not establish a population-level size or conjugation trend.
6. The native lane is a different continuum/electronic model and is itself badly wrong for the large sentinel, so hybrid-minus-native is a mechanistic discrepancy, not an accuracy residual.
7. Three selected sentinels cannot support a prevalence claim.

The strongest justified inference is:

> The failure is heterogeneous and cannot be repaired by any one-dimensional, common-direction electrostatic adjustment.

---

# 6. Ranked next three no-fit experiments

## 1. Matched-QM permanent and induced cavity-boundary response test — implement this next

At each frozen geometry and the identical cavity, compute:

[
V_{\partial\Omega}^{\rm QM,0}
]

from the vacuum QM density, and finite-field induced densities for:

* the three weak uniform Cartesian fields;
* the frozen converged hybrid reaction potential (v_R^\star), applied as an external perturbation with (\pm\epsilon v_R^\star).

Compare directly on the cavity boundary:

[
V_{\partial\Omega}[p_{\rm MDP}]
\quad\text{vs}\quad
V_{\partial\Omega}^{\rm QM,0},
]

and

[
V_{\partial\Omega}[r(h)+a(h)]
\quad\text{vs}\quad
\delta V_{\partial\Omega}^{\rm QM}[h].
]

Do not fit atom-centred charges to the QM density. Compare the physical potentials and their (W_R) half-works.

Freeze an uncertainty envelope from basis/method convergence or from two accepted reference electronic-structure levels before inspecting hybrid discrepancies:

[
\tau_{\rm ref}
==============

\text{reference-to-reference boundary-response spread}.
]

**Pass:** both the permanent-source and induced-response discrepancies are no larger than (\tau_{\rm ref}), and the induced response is reciprocal and passive.

**Fail and terminate:**

* permanent boundary potential fails (\Rightarrow) terminate the present MDP point-source hybrid;
* residual response fails (\Rightarrow) terminate the MACE-POLAR residual branch;
* ADT response fails despite accurate total (\alpha) (\Rightarrow) terminate ADT;
* individual (r) and (a) responses are plausible but their sum fails, with high continuum-active overlap (\Rightarrow) only then admit a projector as a new target-free candidate.

No additional 505-scale run is justified before this test.

## 2. Closed-loop continuation, reciprocity, and stability test

Use

[
u_\lambda
=========

T_R[p+\lambda(r(u_\lambda)+a(u_\lambda))]
]

for a fixed grid of (\lambda\in[0,1]), continuing both forward and backward. At every point record:

[
\rho(\mathcal J_\lambda),
]

[
\sigma_{\min}
\left(
I-\widetilde{\mathcal L}_{s,\lambda}
\right),
]

[
\left|
\left(
I-\widetilde{\mathcal L}_{s,\lambda}
\right)^{-1}
\right|,
]

branch-resolved eigenmode contributions, and finite-difference versus implicit response.

**Pass:** one reversible branch, forward/reverse agreement to numerical tolerance, no root jump, no negative eigenvalue of the symmetric static molecular susceptibility, implicit and finite-difference responses agree, and the loop remains subcritical:

[
\rho(\mathcal J_\lambda)<1
\quad
\text{for all }\lambda.
]

**Fail and terminate:** hysteresis, multiple roots, singular (I-\mathcal J), nonpassive response, derivative failure, or a near-unit mode whose amplification explains the sentinel. Terminate the closed-loop operational hybrid rather than treating solver convergence as physical validation.

## 3. Fixed-source independent continuum/cavity source-swap test

Use sources that contain no ML ambiguity:

1. analytic centered and off-centre charge/dipole sources in spherical cavities;
2. the matched-QM permanent and induced densities from Experiment 1;
3. the identical molecular cavity, with systematic (l_{\max})/Lebedev convergence.

Compare ddPCM against an independent standard PCM/BEM or Poisson solver using the same physical source and, as closely as possible, the same cavity definition.

Record energy, surface charge, boundary potential, and returned receiver field.

**Pass:** analytic cases converge to their known solutions and molecular fixed-source energies/fields agree with the independent solver within the independently estimated discretization error.

**Fail and terminate:** source-independent discrepancies, large grid/orientation drift, unstable cavity convergence, or a molecular anomaly that remains when the ML source is replaced by QM. Terminate the current continuum coupling/cavity implementation; do not compensate through source changes.

---

# 7. Finite decision tree

```text
Run frozen-root seven-combination decomposition.
|
|-- Closure, E_i <= 0, or Cauchy bound fails
|     -> numerical pairing/source-map/operator defect
|     -> TERMINATE current implementation.
|
`-- Algebraic checks pass
      |
      |-- Run matched-QM boundary source/response test.
      |     |
      |     |-- p fails
      |     |     -> MDP atomwise point source is not PCM-transferable
      |     |     -> TERMINATE frozen hybrid.
      |     |
      |     |-- r fails under fixed imposed fields
      |     |     -> POLAR residual is intrinsically non-transferable/excessive
      |     |     -> TERMINATE frozen hybrid.
      |     |
      |     |-- ADT fails with accurate molecular alpha
      |     |     -> spatial ADT allocation is wrong
      |     |     -> TERMINATE frozen hybrid.
      |     |
      |     |-- r and a individually pass, sum fails, overlap diagnostics high
      |     |     -> projector is scientifically admissible as a new candidate
      |     |     -> require rank-3, well-conditioned continuum Gram
      |     |     -> otherwise TERMINATE.
      |     |
      |     `-- permanent and induced boundary responses pass
      |           -> continue.
      |
      |-- Run coupling continuation/stability test.
      |     |
      |     |-- nonpassive, hysteretic, near-singular, or derivative-inconsistent
      |     |     -> nonlinear feedback failure
      |     |     -> TERMINATE closed-loop hybrid.
      |     |
      |     `-- stable and reciprocal
      |           -> continue.
      |
      |-- Run independent fixed-source continuum/cavity test.
      |     |
      |     |-- fails
      |     |     -> cavity/operator implementation failure
      |     |     -> TERMINATE continuum coupling.
      |     |
      |     `-- passes
      |           -> sources, response, feedback, and operator have passed
      |           -> if sentinel free energies remain poor, the operational
      |              half-work is not a transferable solvation free-energy scalar
      |           -> TERMINATE product lane before another 505-scale run.
```

---

# 8. Critical challenge to the hybrid premise

Yes: the hybrid may be combining quantities that are useful inside their parent models but are not separately transferable PCM source/response objects.

By the frozen premises:

* MACE-MDP is trained for total molecular dipoles and polarizabilities, not for a unique atomwise electrostatic density.
* Its atom-centred (q/p) decomposition can reproduce the molecular moment while having an unconstrained near-field gauge.
* MACE-POLAR’s multipoles are an internal coarse-grained density used together with its local energy, learned non-local energy, Coulomb term, and fixed number of field updates.
* Subtracting the zero-field source and a tangent, placing the remainder in a different continuum, and iterating it to a new fixed point are all out-of-parent operations.

The published MACE-POLAR architecture is explicitly non-self-consistent: it performs two learned field updates rather than iterating to convergence. Its external-field dipoles and polarizabilities are an unsupervised extrapolation from energy/force training, and the reported polarizability errors are materially larger than the corresponding reference-method errors. ([arXiv][2]) Its final multipoles contribute to a model that also contains local and learned non-local energy terms, so extracting the multipoles alone removes parent-model compensation by construction. ([arXiv][2])

This creates three structural risks:

[
\text{accurate total dipole}
\centernot\implies
\text{accurate cavity-boundary potential},
]

[
\text{useful internal density feature}
\centernot\implies
\text{standalone linear-response source},
]

and

[
\text{correct molecular }\alpha
\centernot\implies
\text{correct distributed PCM response}.
]

The earliest decisive test is therefore the operator-valued boundary-response comparison

[
\rho^{\rm QM}*0
\longmapsto
V*{\partial\Omega}^{\rm QM,0},
]

and

[
v_{\rm app}
\longmapsto
\delta V_{\partial\Omega}^{\rm QM}[v_{\rm app}]
]

against

[
p_{\rm MDP}
\longmapsto
V_{\partial\Omega}[p_{\rm MDP}]
]

and

[
v_{\rm app}
\longmapsto
V_{\partial\Omega}[r(v_{\rm app})+a(v_{\rm app})].
]

This comparison:

* uses no experimental solvation energies;
* avoids arbitrary atomic charge or atomic polarizability partitions;
* tests exactly what PCM sees;
* separates permanent-source transfer from induced-response transfer;
* can be performed before any additional 505-record run.

If the permanent or induced cavity-boundary map fails, neither an energy decomposition nor an orthogonal projector rescues the premise. A projector can separate two source subspaces; it cannot turn a non-transferable latent source into the correct QM boundary response.

The present three-sentinel evidence is enough to **block further scale-up**, but not enough to choose a projector. The queued decomposition should be interpreted as an internal geometry-specific attribution. The next decisive implementation is the matched-QM boundary-response experiment. A physical projector becomes admissible only under the narrow “each parent response passes, their sum fails through demonstrable continuum-active redundancy” outcome.

HYBRID-BRANCH-ATTRIBUTION-DECISION

[1]: https://research.rug.nl/en/publications/molecular-polarizabilities-calculated-with-a-modified-dipole-inte "https://research.rug.nl/en/publications/molecular-polarizabilities-calculated-with-a-modified-dipole-inte"
[2]: https://arxiv.org/html/2602.19411v1 "https://arxiv.org/html/2602.19411v1"
