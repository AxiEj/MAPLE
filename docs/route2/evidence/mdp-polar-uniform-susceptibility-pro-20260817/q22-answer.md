# Terminal verdict

[
\boxed{
\text{The original }B_PA G\text{ lane is mathematically closed.}
}
]

[
\boxed{
\textbf{Implement Candidate ADT with one fixed, source-physical minimum-norm lift.}
}
]

More precisely, implement the construction defined below using the **block-diagonal free-atom Coulomb self-work metric of the frozen ADT translation tangents**. Treat it as one last **operational zero-training** candidate, not as Tier V. Do not implement Candidate D as the production construction: its atomwise (\alpha_A) partition is a checkpoint-defined latent gauge, not a molecular observable or a uniquely determined source response.

If the ADT construction fails any of the frozen rank, passivity, fixed-point, finite-field, or force gates below, the zero-training source-composition lane terminates. Only then should the project open a separately named frozen-backbone source/scalar head trained on independent QM electrostatic data. Candidate D should not be revived after seeing those failures.

---

# 1. Exact no-go theorem for (K=B_PA G)

Let

[
K=B_PA G,
\qquad
GU_0=I_3.
]

Then the corrected zero-field molecular susceptibility is

[
\begin{aligned}
C(J_P+K)U_0
&=CJ_PU_0+CB_PA,GU_0\
&=C_P+C_PA\
&=C_P(I+A).
\end{aligned}
]

Therefore

[
\operatorname{range}!\left(C(J_P+K)U_0\right)
\subseteq \operatorname{range}(C_P),
]

and

[
\operatorname{rank}!\left(C(J_P+K)U_0\right)
\le \operatorname{rank}(C_P).
]

If

[
\operatorname{rank}(C_P)<\operatorname{rank}(\alpha_D),
]

then

[
C_P(I+A)=-\alpha_D
]

has no solution.

For water,

[
\operatorname{rank}(C_P)=2,
\qquad
\operatorname{rank}(\alpha_D)=3,
]

so every correction confined to the original uniform POLAR source span is incapable of producing the out-of-plane molecular response. This is an exact algebraic obstruction.

## Least-squares lower bound

Let (P_{\mathcal R}) be the orthogonal projector onto (\operatorname{range}(C_P)). Then

[
\min_X|C_PX+\alpha_D|_F
=======================

|(I-P_{\mathcal R})\alpha_D|_F.
]

The Moore–Penrose solution

[
X_\star=-C_P^+\alpha_D
]

attains this minimum, but it cannot remove the component of (\alpha_D) lying outside (\operatorname{range}(C_P)). In the reported case,

[
\frac{
\min_X|C_PX+\alpha_D|_F
}{
|\alpha_D|_F
}
=0.55974756.
]

That is not a conditioning estimate. It is the irreducible approximation error of the restricted model class.

## Why the prohibited repairs do not help

### Moore–Penrose inverse

A pseudoinverse supplies the least-squares projection

[
C_PC_P^+\alpha_D=P_{\mathcal R}\alpha_D,
]

not (\alpha_D). It silently changes the required response to its rank-two projection.

Near an exactly planar geometry, a regularized attempt to invert the third singular direction either:

* leaves a finite residual; or
* produces coefficients growing like (1/\sigma_3).

The second behavior has no smooth planar limit and would make geometry derivatives and forces singular.

### Eigenvalue or singular-value clipping

Clipping changes either (C_P), (\alpha_D), or the requested response subspace. It therefore changes the estimand and violates the frozen no-scaling/no-clipping rule. It cannot turn a rank-two image into an exact rank-three image without adding a new source direction.

### Geometry-dependent rank completion

If the completion remains inside (\operatorname{range}(B_P)), its molecular image remains inside (\operatorname{range}(C_P)), so it cannot increase the rank.

If it adds directions outside (\operatorname{range}(B_P)), it is no longer a (B_PA G) construction. It is a new source lift.

Moreover, a completion built from a near-null singular vector has difficult symmetry limits: at planar geometries the normal direction is symmetry-protected, while near degeneracies its sign and basis choice can become discontinuous. A carefully designed equivariant completion is mathematically possible, but it is precisely a new model outside the rejected restricted topology and is forbidden here if selected geometry by geometry.

The rotated water result confirms that the null direction is covariant and physical to the architecture; it does not rescue the rank-deficient map.

---

# 2. Candidate D is well-defined but not a canonical physical source response

Candidate D defines

[
(B_Dg)*{q_A}=0,
\qquad
(B_Dg)*{p_A}=-\alpha_Ag,
]

so that

[
CB_D=-\sum_A\alpha_A=-\alpha_D.
]

Consequently,

[
M_D(u)=M_P(u)+(B_D-B_P)Gu
]

has

[
D_uM_D(0)U_0=B_D
]

and exact molecular closure.

That algebra is correct. The problem is identifiability of the distributed source.

## The atomwise-partition gauge

For any set of tensors (\Gamma_A) satisfying

[
\sum_A\Gamma_A=0,
]

the transformed partition

[
\alpha_A'=\alpha_A+\Gamma_A
]

has the same supervised molecular tensor:

[
\sum_A\alpha_A'=\alpha_D.
]

But its source response changes by

[
(\delta p_A)'-(\delta p_A)
=-\Gamma_Ag.
]

A continuum operator generally resolves the atomwise location of these dipoles, so

[
\mathcal B_RB_D'
\ne
\mathcal B_RB_D
]

even though

[
CB_D'=CB_D.
]

Thus the molecular observable (\alpha_D) does not determine the source distribution. The checkpoint exposes one particular gauge, but another parameterization with the same public molecular predictions could expose another.

Candidate D is therefore:

* deterministic for the frozen checkpoint;
* algebraically valid;
* potentially usable as an engineering constitutive map;
* **not a canonical consequence of the supervised molecular polarizability**.

The fact that the decomposition comes from a neural-network checkpoint does not turn its atomwise pieces into observables.

## Exact conditions for Candidate D

Let (Q_c:\mathcal C\to\mathbb R) extract total source charge, and let (W) be the physical source–receiver work map.

### Charge conservation

Necessary and sufficient for the linear correction is

[
Q_cB_D=0.
]

Candidate D satisfies this structurally because (dq_A=0), provided the raw real-(l=1) conversion does not leak dipole coefficients into monopoles.

For the complete response one also needs

[
Q_cJ_P(u)=0
]

throughout the operational field domain.

### Translation invariance

Sufficient conditions are

[
\alpha_A(R+a)=\alpha_A(R)
]

for every global translation (a), and

[
Q_cB_D=0.
]

The molecular dipole changes under an origin displacement (a) by a term proportional to induced total charge. Hence zero induced charge is what removes hidden origin dependence.

### SO(3) covariance

For every (Q\in SO(3)),

[
\alpha_A(QR)=Q\alpha_A(R)Q^T,
]

and the raw real-(l=1) conversion must intertwine the Cartesian and source representations exactly. Then

[
B_D(QR)Q
========

\rho_c(Q)B_D(R).
]

The receiver chart must satisfy

[
G(QR)\rho_u(Q)=QG(R).
]

These imply covariance of the corrected source.

### Permutation covariance

For an allowed atom permutation (\pi),

[
\alpha_{\pi(A)}(\pi R)=\alpha_A(R),
]

with corresponding source and receiver permutation representations. Atom sorting or a noncovariant output order would invalidate this.

### Smoothness

For conservative forces, each (\alpha_A(R)) must be at least (C^1) over the claimed geometry domain, and the source conversion must be (C^1). Hessians require (C^2).

A smooth molecular sum does not imply smooth atomwise terms. The individual decomposition must be audited directly.

### Uniform reciprocity and passivity

On truly uniform fields,

[
U_0^TW^TB_D
===========

# CB_D

-\alpha_D.
]

Since (\alpha_D=\alpha_D^T\succ0), Candidate D has the correct symmetric passive molecular response on the three-dimensional uniform subspace.

This conclusion requires only the molecular sum. It does not require each (\alpha_A) to be symmetric.

### Atomwise energy interpretation

A separate atomic quadratic energy

[
\Psi_A(g)=-\frac12g^T\alpha_Ag
]

generates the claimed atomic dipole

[
-\alpha_Ag
]

if and only if

[
\alpha_A=\alpha_A^T.
]

It is atomwise passive if additionally

[
\alpha_A\succeq0.
]

If the individual tensors are nonsymmetric, the antisymmetric atomic response cannot be generated by an atomic scalar, even though the antisymmetric pieces cancel in the molecular sum.

### Full native-space work interpretation

For the linear source law

[
c_D(u)=B_DGu,
]

a quadratic scalar in the full native receiver space exists if and only if

[
\boxed{
W^TB_DG=(W^TB_DG)^T.
}
]

Passivity additionally requires, with the frozen sign convention,

[
\boxed{
\operatorname{sym}(W^TB_DG)\preceq0
}
]

on the gauge-free, continuum-accessible receiver subspace.

Neither property follows from

[
\sum_A\alpha_A=\alpha_D.
]

Even individually symmetric positive-semidefinite (\alpha_A) do not establish full-space integrability when (G) is not work-dual to (B_D).

## Decision on Candidate D

Candidate D is **not structurally nonsensical**, but it uses an unvalidated latent gauge to make a chemically consequential atomwise source prediction. For a project explicitly seeking a zero-training **physical construction**, that is insufficient.

It should not be the terminal production candidate.

---

# 3. Candidate ADT and the missing canonical object

Let

[
\mathcal T_R:\mathbb R^{3N}\to\mathcal C
]

be the frozen ADT map from atomic density-translation coordinates (d) to source changes.

Assume first that the ADT primitive conserves charge exactly:

[
Q_c\mathcal T_R=0.
]

Define its molecular-dipole map

[
D_R=C_R\mathcal T_R:
\mathbb R^{3N}\to\mathbb R^3.
]

To convert a desired molecular induced dipole (z\in\mathbb R^3) into atomic translations, one needs a right inverse

[
P_R:\mathbb R^3\to\mathbb R^{3N},
\qquad
D_RP_R=I_3.
]

Then

[
L_R=\mathcal T_RP_R
]

satisfies

[
C_RL_R=I_3,
\qquad
Q_cL_R=0.
]

The MDP response can then be lifted as

[
\boxed{
B_{\rm ADT}(R)=-L_R\alpha_D(R).
}
]

This gives

[
CB_{\rm ADT}=-\alpha_D.
]

## Why (\alpha_D) alone does not determine (P_R)

If (P_0) is one right inverse, then

[
P=P_0+N,
\qquad
D_RN=0,
]

is another. The null-space term changes the atomwise translations and source distribution without changing the molecular dipole.

The missing object is therefore:

[
\boxed{
\text{a fixed physical metric on the ADT translation coordinates,}
}
]

or, equivalently, a prospectively frozen source-lift variational principle.

Without such a metric, “atomic weights” are precisely the unspecified null-space choice.

## Canonical ADT metric for this asset

Use the free-atom Coulomb self-work of the same frozen translation tangents.

For atom (A), let

[
\mathcal T_A:\mathbb R^3\to\mathcal C_A
]

be its frozen free-atom translation tangent, and let (V_A) be the exact vacuum Coulomb self-energy metric of that atom’s frozen source representation. Define

[
H_A=\mathcal T_A^TV_A\mathcal T_A.
]

For a spherical free atom,

[
H_A=\kappa_{Z_A}I_3,
\qquad
\kappa_{Z_A}>0.
]

Freeze

[
\boxed{
H_{\rm ADT}
===========

\bigoplus_A H_A.
}
]

This metric:

* is derived from the same source-bound free-atom asset;
* has physical energy units;
* does not mix raw charge and dipole coordinates by an arbitrary Euclidean convention;
* introduces no fitted atomic weights;
* is fixed by element and source definition, not by geometry-specific rank choices;
* is rotationally and permutation covariant.

The full molecular Coulomb Gram matrix would be a different model. It should not be substituted after inspecting results.

## Unique minimum-(H_{\rm ADT}) lift

Require

[
H_{\rm ADT}\succ0,
\qquad
\operatorname{rank}(D_R)=3.
]

Define

[
S_R
===

D_RH_{\rm ADT}^{-1}D_R^T.
]

Then (S_R\succ0), and

[
\boxed{
P_R
===

H_{\rm ADT}^{-1}D_R^T
S_R^{-1}.
}
]

It satisfies

[
D_RP_R=I_3.
]

For every requested molecular dipole (z),

[
d_\star=P_Rz
]

is the unique solution of

[
\min_d\frac12d^TH_{\rm ADT}d
\quad
\text{subject to}
\quad
D_Rd=z.
]

This follows from the KKT equations

[
H_{\rm ADT}d-D_R^T\lambda=0,
\qquad
D_Rd=z.
]

No pseudoinverse and no regularization are used. If (S_R) is singular or fails the frozen conditioning gate, the construction fails closed.

The source lift is

[
\boxed{
L_R
===

\mathcal T_RH_{\rm ADT}^{-1}D_R^T
\left(
D_RH_{\rm ADT}^{-1}D_R^T
\right)^{-1}.
}
]

Finally,

[
\boxed{
B_{\rm ADT}=-L_R\alpha_D.
}
]

This is the missing canonical construction.

## Covariance of the lift

Let (\rho_d(Q)=I_N\otimes Q). Sufficient conditions are

[
\mathcal T_{QR}
===============

\rho_c(Q)\mathcal T_R\rho_d(Q)^T,
]

[
C_{QR}\rho_c(Q)=QC_R,
]

[
H_{\rm ADT}(QR)
===============

\rho_d(Q)H_{\rm ADT}(R)\rho_d(Q)^T.
]

Then

[
D_{QR}=QD_R\rho_d(Q)^T,
]

[
P_{QR}=\rho_d(Q)P_RQ^T,
]

and

[
L_{QR}=\rho_c(Q)L_RQ^T.
]

The analogous relations hold for atom permutations. Charge neutrality of (\mathcal T_R) makes (D_R), (P_R), and (L_R) origin independent.

## What the acetone canary establishes

The acetone MEP numbers are finite evidence that the underlying ADT source primitive can approximate one QM response pattern. They do not establish:

* rank-three liftability for all geometries;
* correctness of the new minimum-(H_{\rm ADT}) allocation;
* water behavior;
* finite-field stability;
* chemical accuracy.

They may not be used to choose among alternative metrics now. The metric above is justified by the frozen free-atom source physics, not by the canary score.

---

# 4. Exact tangent replacement and double counting

Define

[
B_A:=B_{\rm ADT}=-L\alpha_D,
\qquad
\Delta B=B_A-B_P.
]

The recommended operational source law is

[
\boxed{
M_A(R,u)
========

M_P(R,u)
+
\Delta B(R)G(R)u.
}
]

Expand the native model at zero field:

[
M_P(u)
======

M_P(0)
+
J_0u
+
\frac12D_u^2M_P(0)[u,u]
+\cdots,
]

where

[
B_P=J_0U_0.
]

For a uniform input (u=U_0g),

[
\begin{aligned}
D_uM_A(0)U_0g
&=
J_0U_0g+(B_A-B_P)GU_0g\
&=
B_Pg+(B_A-B_P)g\
&=
B_Ag.
\end{aligned}
]

Therefore

[
C,D_uM_A(0)U_0=-\alpha_D.
]

For (n\in\ker G),

[
D_uM_A(0)n=J_0n.
]

And for every (k\ge2),

[
D_u^kM_A(u)=D_u^kM_P(u).
]

Thus subtracting (B_PGu) and adding (B_AGu) is an **exact replacement of the selected linear tangent**. It is not linear-response double counting.

What remains is a modeling assumption:

* the native nonlinear POLAR response is retained;
* the native nonuniform response is retained;
* only the arithmetic uniform linear coordinate is replaced.

The retained higher-order terms may still produce nonlinear uniform–nonuniform couplings. They are not duplicated by the linear ADT term, but their physical compatibility with the replacement must be tested at finite field.

By contrast,

[
M_P(u)+B_AGu
]

without subtracting (B_PGu) would have molecular uniform susceptibility

[
C_P-\alpha_D
]

and would double-count the uniform linear response.

## Strict work-generated ADT state

The source lift (L) has the work-dual field coordinate

[
G_L=L^TW,
]

because

[
G_LU_0
======

# L^TWU_0

# (CL)^T

I.
]

An explicit induced molecular dipole state has energy

[
\Psi_z(z;u)
===========

\frac12z^T\alpha_D^{-1}z
+
z^TL^TWu.
]

Stationarity gives

[
z_\ast=-\alpha_DL^TWu
]

and source

[
c_z=Lz_\ast
===========

-L\alpha_DL^TWu.
]

Its work Jacobian is

[
W^TD_uc_z
=========

-W^TL\alpha_DL^TW,
]

which is symmetric negative semidefinite.

That construction is genuinely reciprocal and passive. But retaining the native nonlinear/nonuniform response around it generally destroys the common-functional conclusion unless the retained residual independently passes work reciprocity.

The fully work-generated replacement would be

[
M_{\rm strict}(u)
=================

M_P(0)-L\alpha_DL^TWu,
]

which discards the native induced response. That is not the selected implementation here. The selected (M_A) remains an operational constitutive response because it uses the frozen arithmetic chart (G) and retains the native nonlinear residual.

---

# 5. Exact implementation derivatives for the ADT candidate

To avoid symbol collision, write the continuum equations as

[
\mathcal A_R\sigma=\mathcal B_Rc,
\qquad
u=\mathcal L_R\sigma.
]

Let

[
J_M(R,u)=D_uM_A(R,u).
]

## Source JVP

For a receiver perturbation (v),

[
\boxed{
J_Mv
====

J_P(R,u)v+\Delta B,Gv.
}
]

## Source VJP

For a source cotangent (\bar c),

[
\boxed{
J_M^T\bar c
===========

J_P(R,u)^T\bar c
+
G^T\Delta B^T\bar c.
}
]

The JVP and VJP must be implemented from the same algebra, not through independently coded approximations.

## Geometry derivative at fixed (u)

Let (d) denote a nuclear-coordinate differential. Then

[
dM_A\big|_u
===========

dM_P\big|_u
+
(d\Delta B)Gu
+
\Delta B(dG)u.
]

Since

[
\Delta B=-L\alpha_D-B_P,
]

[
\boxed{
d\Delta B
=========

-(dL)\alpha_D
-L(d\alpha_D)
-dB_P.
}
]

With

[
B_P=J_0U_0,
\qquad
J_0=D_uM_P(R,0),
]

[
\boxed{
dB_P=(dJ_0)U_0+J_0(dU_0).
}
]

The term (dJ_0) is the mixed geometry–field derivative

[
D_RD_uM_P(R,0).
]

Frozen model weights do not make this derivative zero.

For the ADT lift,

[
L=\mathcal TP,
]

so

[
dL=(d\mathcal T)P+\mathcal T(dP).
]

Define

[
D=C\mathcal T,
\qquad
Y=H^{-1}D^T,
\qquad
S=DY,
\qquad
P=YS^{-1}.
]

Then

[
dD=(dC)\mathcal T+C(d\mathcal T),
]

[
dY
==

## H^{-1}(dD)^T

H^{-1}(dH)Y,
]

[
dS=(dD)Y+D(dY),
]

and

[
\boxed{
dP
==

## (dY)S^{-1}

P(dS)S^{-1}.
}
]

For the frozen block free-atom metric proposed above,

[
dH=0
]

at fixed chemical identities. It should nevertheless remain an explicit zero in the implementation contract rather than being omitted accidentally.

## Coordinate VJP bookkeeping

Let

[
h=Gu.
]

For a source cotangent (\bar c),

[
\bar c^T(dM_A)
==============

\bar c^T(dM_P)
+
\operatorname{tr}
\left[
(\bar c h^T)^T d\Delta B
\right]
+
\operatorname{tr}
\left[
\bigl((\Delta B^T\bar c)u^T\bigr)^TdG
\right].
]

Thus the immediate reverse-mode cotangents are

[
\bar{\Delta B}=\bar c h^T,
]

[
\bar G=(\Delta B^T\bar c)u^T,
]

[
\bar L=-\bar{\Delta B}\alpha_D^T,
]

[
\bar\alpha_D=-L^T\bar{\Delta B},
]

[
\bar B_P=-\bar{\Delta B}.
]

These must be propagated through (L=\mathcal TP), the fixed-metric constrained solve for (P), (D=C\mathcal T), (B_P=J_0U_0), and all analytic ADT coordinate derivatives.

## Coupled root Jacobian

Use the block residual

[
F(R,\sigma,u)
=============

\begin{pmatrix}
\mathcal A_R\sigma-\mathcal B_RM_A(R,u)\
u-\mathcal L_R\sigma
\end{pmatrix}
=0.
]

Its state Jacobian is

[
\boxed{
F_{(\sigma,u)}
==============

\begin{pmatrix}
\mathcal A_R & -\mathcal B_RJ_M\
-\mathcal L_R & I
\end{pmatrix}.
}
]

After eliminating (\sigma), define

[
\mathcal P_R
============

\mathcal L_R\mathcal A_R^{-1}\mathcal B_R.
]

The receiver fixed-point residual is

[
H(R,u)=u-\mathcal P_RM_A(R,u),
]

with

[
\boxed{
H_u=I-\mathcal P_RJ_M.
}
]

The corresponding VJP is

[
H_u^Tw
======

## w

J_M^T
\mathcal B_R^T
\mathcal A_R^{-T}
\mathcal L_R^T w.
]

## Conservative operational force

Let the prospectively frozen scalar ledger, after substituting (c=M_A(R,u)), be

[
\widehat\Phi(R,\sigma,u).
]

Solve

[
F_{(\sigma,u)}^T
\begin{pmatrix}
\lambda_\sigma\
\lambda_u
\end{pmatrix}
=============

\begin{pmatrix}
\widehat\Phi_\sigma^T\
\widehat\Phi_u^T
\end{pmatrix}.
]

At fixed state variables,

[
d_RF_1
======

## (d\mathcal A)\sigma

## (d\mathcal B)c

\mathcal B,d_RM_A\big|_u,
]

[
d_RF_2=-(d\mathcal L)\sigma.
]

Therefore

[
\boxed{
\begin{aligned}
\frac{d\Phi_{\rm op}}{dR}
={}&
\widehat\Phi_R
-\lambda_\sigma^T(d\mathcal A)\sigma
+\lambda_\sigma^T(d\mathcal B)c\
&+\lambda_\sigma^T\mathcal B,d_RM_A\big|_u
+\lambda_u^T(d\mathcal L)\sigma.
\end{aligned}
}
]

All cavity, continuum, source, receiver, ADT-lift, MDP-(\alpha_D), POLAR mixed-derivative, and fixed-point terms are mandatory.

An operational scalar can be conservative when the selected root is unique, differentiable, and has a nonsingular Jacobian, provided the force differentiates that exact scalar including the implicit root response.  The adjoint formula establishes the derivative of the declared scalar; it does not turn a nonintegrable electronic constitutive map into a common variational functional. 

---

# 6. Terminal target-free admission gates

The following gates must be frozen before any solvation-accuracy panel is opened.

## A. Structural ADT-lift gate

Require for every geometry in the prospectively frozen geometry audit:

[
Q_c\mathcal T=0
]

to numerical precision,

[
H_{\rm ADT}\succ0,
]

[
S=D H_{\rm ADT}^{-1}D^T\succ0,
]

and

[
\frac{\lambda_{\min}(S)}
{\lambda_{\max}(S)}
\ge10^{-8}.
]

Also require

[
|DP-I|_2\le10^{-12},
]

[
|CL-I|_2\le10^{-12},
]

[
|Q_cL|_2\le10^{-12}.
]

No pseudoinverse, ridge term, singular-value truncation, or geometry-local basis deletion is permitted. Failure is terminal for ADT.

## B. Uniform susceptibility closure

Require

[
\boxed{
\frac{
|C,D_uM_A(0)U_0+\alpha_D|_F
}{
|\alpha_D|_F
}
\le10^{-12}.
}
]

Because closure is algebraic, failure at this scale is an implementation error, not a model-accuracy issue.

## C. Gauge and charge

For the constant-potential gauge vector (z_{\rm gauge}),

[
Gz_{\rm gauge}=0,
]

[
|D_uM_A(u)z_{\rm gauge}|\le10^{-12}
]

throughout the finite-field audit.

Require total charge invariance

[
|Q_c[M_A(u)-M_A(0)]|
\le10^{-10}\ e
]

for every audited field.

## D. SO(3), translation, and permutation

For deterministic random proper rotations and allowed atom permutations, require:

* source covariance relative defect (\le10^{-9});
* molecular susceptibility covariance defect (\le10^{-10});
* coordinate-JVP covariance defect (\le10^{-7});
* origin translation changes no induced molecular response beyond (10^{-10}).

The molecular-plane normal must rotate covariantly without any geometry-dependent sign repair.

## E. JVP/VJP consistency

For deterministic hashed vectors (v) and (\bar c),

[
\frac{
|\bar c^T(J_Mv)-v^T(J_M^T\bar c)|
}{
\max(1,|\bar c^TJ_Mv|,|v^TJ_M^T\bar c|)
}
\le10^{-10}.
]

Compare source JVPs with centered finite differences at two steps. Require relative convergence to (\le10^{-6}).

Perform the same checks for:

* (\mathcal T);
* (P);
* (L);
* (B_{\rm ADT});
* (B_P);
* the full continuum fixed-point residual.

## F. Passivity and reciprocity status

Let (S_u) be the frozen public normalization that converts native receiver features to dimensionless coordinates. Define

[
\widehat H_W(u)
===============

S_u^{-T}W^TJ_M(u)S_u^{-1}.
]

For the operational candidate, global work reciprocity is not assumed. Record

[
r_{\rm curl}(u)
===============

\frac{
|\operatorname{skew}\widehat H_W(u)|_2
}{
\max(1,|\operatorname{sym}\widehat H_W(u)|_2)
}.
]

This is a reportable nonvariational diagnostic, not automatically a rejection criterion under claim B.

Passivity of the symmetric response is mandatory:

[
\boxed{
\lambda_{\max}
\left[
\operatorname{sym}\widehat H_W(u)
\right]
\le
10^{-8}
\max
\left(
1,
|\operatorname{sym}\widehat H_W(u)|_2
\right)
}
]

on the gauge-free, continuum-accessible field subspace.

Uniformly,

[
U_0^TW^TJ_M(0)U_0=-\alpha_D\prec0
]

must hold exactly.

If the full symmetric response has a materially positive mode, the model is an active static constitutive law and is rejected even as operational. No clipping is allowed.

Because global reciprocity is absent, the frozen scalar ledger may not contain a polarization term obtained by treating

[
\int c^TW,du
]

as path independent.

## G. ADT tangent-support gate

For every audited field, form

[
z=-\alpha_DGu,
\qquad
d=Pz.
]

Using the exact translated free-atom density/source asset, compare the exact finite translation with its tangent:

[
r_{\rm ADT}(d)
==============

\frac{
|\Delta c_{\rm exact}(d)-\mathcal Td|*{V}
}{
\max(|\Delta c*{\rm exact}(d)|_V,\epsilon)
}.
]

Require

[
r_{\rm ADT}(d)\le10^{-2}
]

throughout the frozen operational field envelope.

This is a support test for the linear translation tangent, not a fit. If it fails, no damping, displacement clipping, or response scaling may be introduced.

## H. Finite-field audit

Freeze the field envelope before inspecting solvation accuracy. It must include:

* both signs along all three eigenvectors of (\alpha_D);
* the actual target-free ddPCM field norms from the intended geometry domain;
* at least a (25%) prospective norm margin;
* deterministic nonuniform continuum modes in (\ker G);
* mixed fields
  [
  u=tU_0g+r n,\qquad Gn=0;
  ]
* duplicated-radial-block difference modes.

At every point require:

* finite, smooth source response;
* charge and gauge invariance;
* JVP/VJP closure;
* no passive-sign reversal;
* no discontinuous atomwise source redistribution;
* ADT tangent-support pass;
* no source norm outside the prospectively frozen source envelope.

The retained nonlinear POLAR derivatives must be checked explicitly on mixed uniform/nonuniform fields.

## I. Five-start coupled-root audit

For each geometry solve exactly the same final equations from:

1. (u=0);
2. the permanent/zero-field-source continuum field;
3. the converged uncorrected MACE-POLAR root;
4. an overpolarized start obtained by doubling the native induced increment about the permanent-source field;
5. a reversed induced start obtained by negating that increment.

All five must converge to the same root, with:

[
|H(u_\ast)|\le10^{-10},
]

relative root difference (\le10^{-8}), and ledger difference at the frozen numerical floor.

Let

[
\widehat H_u=S_uH_uS_u^{-1}.
]

Require

[
\boxed{
\frac{\sigma_{\min}(\widehat H_u)}
{\max(1,|\widehat H_u|_2)}
\ge10^{-6}.
}
]

For the frozen finite-field audit nodes, additionally require numerical strong monotonicity

[
\boxed{
\lambda_{\min}
\left[
\operatorname{sym}(\widehat H_u)
\right]
\ge10^{-5}.
}
]

An analytic domain-wide lower bound would constitute structural evidence. Passing a finite grid plus the five starts is finite numerical evidence only. Any competing root, hysteresis, fold, or unresolved near-singularity rejects the candidate before the accuracy panel.

## J. Coordinate and force finite differences

At fixed (u), compare analytic coordinate derivatives of

* (\mathcal T);
* (D);
* (P);
* (L);
* (\alpha_D);
* (B_{\rm ADT});
* (B_P);
* (M_A);

against centered finite differences at two step sizes. Require relative convergence to (\le10^{-6}).

For the complete implicitly solved scalar, require analytic-force agreement with central finite differences to

[
10^{-5}
]

relative, or a prospectively frozen absolute numerical floor.

Also require translational and rotational force sum rules.

A force-closure pass proves conservativity of the declared operational ledger. It does not establish Tier V or chemical accuracy.

---

# 7. Structural proof, numerical evidence, and chemical calibration

## Structural proof

The following are theorem-level conclusions:

* (B_PA G) cannot repair the water response rank.
* A pseudoinverse cannot remove the missing range component.
* Candidate D’s atomwise source is not determined by (\alpha_D).
* The fixed-metric ADT right inverse is unique when (H_{\rm ADT}\succ0) and (D) has full row rank.
* Subtracting (B_PGu) makes the additive construction an exact uniform-tangent replacement.
* The stated adjoint differentiates the declared scalar if the selected root is nonsingular and differentiable.

## Finite numerical evidence

The following require testing:

* rank and conditioning of the ADT dipole map over chemistry;
* covariance and smoothness;
* finite-field tangent support;
* passivity of the retained POLAR-plus-ADT response;
* root uniqueness and conditioning;
* analytic-force closure.

The acetone canary is one item of finite evidence, not a structural certificate.

## Chemical calibration

Only after every target-free gate passes may one prospectively frozen solvation-accuracy panel be opened.

That panel can test the total operational model. It cannot validate:

* an atomwise polarizability decomposition;
* a Tier-V/common-functional interpretation;
* path-independent source work;
* uniqueness of internal component energies.

No post-panel comparison between Candidate D and ADT is permitted.

---

# Final recommendation

Implement exactly

[
\boxed{
\begin{aligned}
H_{\rm ADT}
&=\bigoplus_A\mathcal T_A^TV_A\mathcal T_A,[2mm]
D&=C\mathcal T,[2mm]
P&=H_{\rm ADT}^{-1}D^T
\left(DH_{\rm ADT}^{-1}D^T\right)^{-1},[2mm]
L&=\mathcal TP,[2mm]
B_{\rm ADT}&=-L\alpha_D,[2mm]
M_A(u)
&=
M_P(u)
+
\left(B_{\rm ADT}-J_0U_0\right)Gu.
\end{aligned}
}
]

This is **Candidate ADT with a fully specified canonical source lift**.

It is admitted only to one terminal target-independent audit. It is not yet admitted to a solvation panel and is not Tier V. If it fails any frozen gate, declare the zero-training construction lane closed and proceed directly to the separately named frozen-backbone QM-electrostatic source/scalar head.
