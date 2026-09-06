# Terminal recommendation

**Advance the proposed v3 decomposition to a target-independent operator prequalification, but not directly to the 505-case energy run. Retire the MACE-MDP latent atomwise (q/p) from the production permanent-source branch.** Retain MACE-MDP only for its supervised molecular observables, principally (\alpha_{\mathrm{MDP}}), plus optional diagnostics involving its total dipole.

This is a conditional architecture decision, not a claim that the v3 source is a uniquely correct electronic density. Officially, MACE-MDP is a molecular dipole/polarizability model, whereas MACE-POLAR uses a non-self-consistent update of Gaussian multipole densities. MACE-POLAR was trained through energies and forces rather than atomwise charge labels, and its own documentation warns that partial atomic charges and dipoles are not unique. ([GitHub][1])

---

## Notation and the sign convention that must be frozen

At a fixed geometry (R), write

[
m_0=M_P(R,0),\qquad
J_0=D_uM_P(R,0),
]

[
g=G_Ru,\qquad
\Pi_R=U_RG_R,
]

and define

[
n(R,u)
======

M_P(R,u)-m_0-J_0\Pi_Ru .
]

Let (P_R=P_{\mathrm{ADT}}(R)), and introduce a sign (s_F) by

[
a(R,u)=s_F P_R\alpha_R G_Ru,
\qquad
\alpha_R=\alpha_{\mathrm{MDP}}(R).
]

The proposed formula corresponds to (s_F=-1).

That sign is correct under the following convention:

[
v_g(\mathbf r)=g\cdot(\mathbf r-\mathbf r_0),\qquad
\mathbf E=-\nabla v_g=-g,
]

and therefore

[
\delta\boldsymbol\mu=\alpha,\delta\mathbf E
=-\alpha,\delta g.
]

Thus (a=-P\alpha g).

If, instead, (G_Ru) is literally the physical electric field (\mathbf E), then the physically correct expression is

[
a=+P\alpha \mathbf E.
]

The identity (G_RU_R=I_3) alone does **not** determine this sign. It must be fixed by an audited energy/field convention, for example

[
\frac{\partial E}{\partial g}=+\mu
\quad\text{for }v_g=g\cdot r,
]

or equivalently

[
\frac{\partial E}{\partial \mathbf E}=-\mu
\quad\text{for }v_{\mathbf E}=-\mathbf E\cdot r.
]

---

# A. Uniform-response double counting

## Theorem A1: exact linear cancellation

Assume:

[
G_RU_R=I_3,
]

(M_P(R,u)) is Fréchet differentiable at (u=0), and the same (J_0), (U_R), and (G_R) are used in both the subtraction and the fixed-point implementation.

Then

[
\Pi_R^2
=======

# U_RG_RU_RG_R

# U_R(G_RU_R)G_R

\Pi_R.
]

Hence (\Pi_R) is a generally oblique projector onto (\operatorname{im}U_R), and

[
u=U_Rg+(I-\Pi_R)u,
\qquad
G_R(I-\Pi_R)u=0.
]

The source residual obeys

[
n(R,0)=0
]

and

[
D_un(R,0)
=========

# J_0-J_0\Pi_R

J_0(I-\Pi_R).
]

For every uniform-field direction (h\in\mathbb R^3),

[
D_un(R,0)[U_Rh]
===============

# J_0(I-\Pi_R)U_Rh

# J_0\left(U_R-U_RG_RU_R\right)h

0.

]

This is coefficientwise cancellation in the complete (4N)-dimensional MACE-POLAR source space. It is stronger than merely cancelling the molecular dipole.

Consequently, if the permanent branch is field independent,

[
D_uc_{\mathrm{perm}}(R)[U_Rh]=0,
]

and the total uniform linear response is solely

[
D_ua(R,0)[U_Rh]
===============

s_FP_R\alpha_Rh.
]

If the ADT source satisfies

[
Q_{\mathrm{ADT}}P_R=0,
\qquad
\mathcal D_{\mathrm{ADT},R}P_R=I_3,
]

where (Q_{\mathrm{ADT}}) is total charge and (\mathcal D_{\mathrm{ADT},R}) is the molecular-dipole map, then

[
Q_{\mathrm{ind}}'=0
]

and

[
D_u\boldsymbol\mu_{\mathrm{ind}}(R,0)[U_Rh]
===========================================

s_F\alpha_Rh.
]

Under the potential-gradient convention (s_F=-1), this becomes

[
D_u\boldsymbol\mu_{\mathrm{ind}}[U_Rh]
======================================

# -\alpha_Rh

\alpha_R\mathbf E.
]

Therefore, **the decomposition is exactly free of uniform-response double counting at linear order**, conditional on the identities above.

## Charge identities

Let (Q_G) be the total-charge functional on the MACE-POLAR coefficient space. The MACE-POLAR charge constraint must give

[
Q_GM_P(R,u)=Q_{\mathrm{mol}}
\quad\text{for all relevant }u.
]

Then

[
Q_GJ_0=0
]

and

[
Q_Gn(R,u)=0.
]

The point decoder must preserve the same total charge,

[
Q_{\mathrm{pt}}m_0=Q_{\mathrm{mol}},
]

and the ADT branch must remain neutral. These identities prevent a solvent field from spuriously changing the molecular charge.

## Important limitation: cancellation is only linear about (u=0)

For a pure uniform perturbation (u=tU_Rh),

[
n(R,tU_Rh)
==========

\frac{t^2}{2}
D^2_{uu}M_P(R,0)[U_Rh,U_Rh]
+O(t^3).
]

Thus MACE-POLAR nonlinear uniform response remains in the residual at second and higher order. That is not linear double counting, because the MDP branch contributes only the linear tensor (\alpha_{\mathrm{MDP}}), but it must be described honestly as

[
\text{MDP linear uniform response}
+
\text{POLAR nonlinear residual response}.
]

## Important limitation: the subtraction removes more than a dipole

The operation

[
J_0U_Rg
]

is a full (4N)-component source tangent. It can contain continuum-visible atomic charge redistribution and globally translated higher multipoles in addition to its molecular induced dipole.

Subtracting it removes **all** of those linear uniform-field components. MDP supplies only the three-dimensional constraint

[
\delta\mu=\alpha_{\mathrm{MDP}}\mathbf E.
]

ADT then selects one particular atom-centred realization of that molecular dipole. Therefore:

* double counting is eliminated;
* but continuum-visible uniform-field information in
  (\ker \mathcal D_R) is discarded and replaced by the ADT physical prior.

That is a physical approximation, not a mathematical defect.

## Chart dependence

“Nonuniform” here means belonging to

[
\ker G_R,
]

not orthogonality in a Coulomb, Euclidean, or response metric. Different left inverses (G_R) can produce different complements and hence different finite-field models. Therefore (U_R,G_R) must be frozen by physical affine-potential semantics, not selected because they produce convenient downstream results.

For rotational objectivity they must obey, for every rotation (Q),

[
D_U(Q)U_R=U_{QR}Q,
]

[
G_{QR}D_U(Q)=QG_R,
]

which implies

[
\Pi_{QR}D_U(Q)
==============

D_U(Q)\Pi_R.
]

Origin changes in an affine potential add a constant-potential mode. At least at linear order, the source must be invariant to that mode:

[
J_0U_{\mathrm{const}}=0.
]

Otherwise the split depends on the arbitrary origin used to write (v_g).

## Counterexamples

If (G_RU_R\neq I), exact cancellation fails. In the scalar example

[
U=1,\qquad G=1+\varepsilon,\qquad J_0=1,
]

one obtains

[
D_un(0)[U]
==========

# 1-(1+\varepsilon)

-\varepsilon\neq0.
]

The MACE-POLAR uniform tangent then survives and is added to ADT.

Separately, suppose (g) is the physical electric field and (\alpha>0), but the implementation nevertheless uses (-\alpha g). The predicted dipole is anti-aligned with the electric field. Algebraic double counting is still absent, but the physical response sign is wrong.

---

# B. Point permanent source plus Gaussian response

## Conditional proposition B1: mathematical validity

Define the point permanent distribution by

[
\rho_{\mathrm{pt}}(\mathbf r)
=============================

\sum_A
\left[
q_A\delta(\mathbf r-\mathbf R_A)
--------------------------------

\mathbf p_A\cdot\nabla
\delta(\mathbf r-\mathbf R_A)
\right].
]

Its potential away from the source centres is

[
V_{\mathrm{pt}}(\mathbf s)
==========================

\sum_A
\left[
\frac{q_A}{|\mathbf s-\mathbf R_A|}
+
\frac{\mathbf p_A\cdot(\mathbf s-\mathbf R_A)}
{|\mathbf s-\mathbf R_A|^3}
\right].
]

The Gaussian increment and ADT source are separate smooth distributions. The complete operational source is

[
\rho_{\mathrm{op}}(R,u)
=======================

S_{\mathrm{pt},R}m_0
+
S_{\mathrm G,R}n(R,u)
+
S_{\mathrm A,R}a(R,u).
]

This is a perfectly legitimate mixed-basis distribution. It need not belong to one radial family.

At (u=0),

[
n(R,0)=0,\qquad a(R,0)=0,
]

and therefore

[
\rho_{\mathrm{op}}(R,0)
=======================

S_{\mathrm{pt},R}m_0.
]

As (u\to0),

[
\rho_{\mathrm{op}}(R,u)
\longrightarrow
S_{\mathrm{pt},R}m_0
]

in the distributional topology and, provided all source centres remain a positive distance from the exposed cavity boundary, in the boundary-potential norm relevant to ddPCM.

Therefore the mixed point/Gaussian construction does **not** itself create:

* a discontinuity at (u=0);
* a cusp in the field response;
* or a nonconservative coordinate derivative.

It defines a **role-separated operational density**.

## What would create nonconservativity

Nonconservativity would arise if one:

* omitted the implicit fixed-point response;
* omitted derivatives of point/Gaussian/ADT basis centres;
* omitted (\partial_RJ_0), (\partial_RP_{\mathrm{ADT}}), or (\partial_R\alpha);
* used a discontinuous cavity or geometry-dependent clipping;
* or allowed the selected fixed-point branch to jump.

The radial mismatch itself is not the cause.

## Physical calibration is a different question

The point decoding is not the checkpoint-native MACE-POLAR density representation. The MACE-POLAR architecture deliberately uses broad Gaussian multipoles as a coarse-grained long-range density, with shorter-range physics assigned to the local energy model. ([arXiv][2])

Consequently, using (m_0) as point multipoles:

1. preserves its total charge and first moment if the coefficient normalization is correct;
2. changes the near-field potential and charge-penetration behavior;
3. removes Gaussian leakage from the permanent source;
4. introduces singular point-source behavior not used in the checkpoint’s internal Coulomb calculation.

Your two disjoint MBIS/ddPCM source panels provide legitimate target-independent evidence for this particular point decoder. They do not prove that it is a unique physical density.

There is an additional conceptual point: if the 1.5-Å Gaussian or ADT density extends into the continuum region, the boundary-value calculation remains operationally well defined, but it is no longer literally the textbook PCM problem of a free charge density strictly confined inside the cavity. That is a physical interpretation issue, not a differentiability failure.

---

# C. Exact JVP, VJP, and nuclear derivative

## Continuum definitions

Let

[
B_{\mathrm p}(R),\quad
B_{\mathrm g}(R),\quad
B_{\mathrm a}(R)
]

map the three source categories to the common ddPCM right-hand side. Define

[
b(R,u)
======

B_{\mathrm p}m_0
+
B_{\mathrm g}n(R,u)
+
B_{\mathrm a}a(R,u).
]

Let

[
A_R\sigma=b(R,u),
\qquad
u=L_R\sigma.
]

Set

[
S_R=A_R^{-1}
]

and use the reduced fixed-point residual

[
r(R,u)
======

u-L_RS_Rb(R,u).
]

The selected state satisfies

[
r(R,u^*)=0.
]

Domain-decomposition continuum methods have established energy and first-derivative formulations, but the nonvariational source update here requires the additional implicit-response terms derived below. ([PubMed][3])

---

## Fixed-geometry source JVP

At fixed (R), define

[
J(u)=D_uM_P(R,u).
]

For a direction (\dot u),

[
\dot n
======

\left[J(u)-J_0\Pi_R\right]\dot u,
]

[
\dot a
======

s_FP_R\alpha_RG_R\dot u.
]

Hence

[
K_R(u)
:=
D_ub(R,u)
=========

B_{\mathrm g}
\left[J(u)-J_0\Pi_R\right]
+
s_FB_{\mathrm a}P_R\alpha_RG_R.
]

For the proposed potential-gradient convention (s_F=-1),

[
K_R(u)
======

B_{\mathrm g}
\left[J(u)-J_0U_RG_R\right]
---------------------------

B_{\mathrm a}P_R\alpha_RG_R.
]

The continuum and residual JVPs are

[
\dot\sigma
==========

S_RK_R(u)\dot u,
]

[
D_uT(R,u)[\dot u]
=================

L_RS_RK_R(u)\dot u,
]

[
D_ur(R,u)[\dot u]
=================

\left[I-L_RS_RK_R(u)\right]\dot u.
]

Equivalently, using the surface-state residual

[
F(R,\sigma)
===========

A_R\sigma-b(R,L_R\sigma),
]

one has

[
D_\sigma F[\dot\sigma]
======================

\left[A_R-K_R(u)L_R\right]\dot\sigma.
]

---

## Fixed-geometry VJP

For a continuum-RHS cotangent (y), set

[
w_{\mathrm g}=B_{\mathrm g}^{T}y,
\qquad
w_{\mathrm a}=B_{\mathrm a}^{T}y.
]

Then

[
K_R(u)^Ty
=========

## J(u)^Tw_{\mathrm g}

\Pi_R^TJ_0^Tw_{\mathrm g}
+
s_FG_R^T\alpha_R^TP_R^Tw_{\mathrm a}.
]

Since

[
\Pi_R^T=G_R^TU_R^T,
]

this can be evaluated without forming (J_0) or (\Pi_R) explicitly.

For (s_F=-1),

[
K_R(u)^Ty
=========

## J(u)^TB_{\mathrm g}^Ty

## G_R^TU_R^TJ_0^TB_{\mathrm g}^Ty

G_R^T\alpha_R^TP_R^TB_{\mathrm a}^Ty.
]

The fixed-point residual VJP is

[
D_ur(R,u)^Ty
============

## y

K_R(u)^TA_R^{-T}L_R^Ty.
]

The surface-state residual VJP is

[
D_\sigma F(R,\sigma)^Ty
=======================

## A_R^Ty

L_R^TK_R(u)^Ty.
]

These formulas give an exact matrix-free implementation contract:

1. apply (L_R^T);
2. solve one (A_R^T) system;
3. apply the branch source VJPs;
4. combine with the identity term.

---

## Nuclear partial derivative at fixed (u)

Let a dot denote a directional nuclear derivative along (\dot R), while holding the coordinate vector (u) fixed.

First,

[
\dot m_0
========

D_RM_P(R,0)[\dot R].
]

The derivative of the zero-field Jacobian is the mixed derivative

[
\dot J_0[h]
===========

D^2_{Ru}M_P(R,0)[\dot R,h].
]

This mixed Hessian-vector product is mandatory. It cannot be replaced by treating (J_0) as a frozen numerical matrix.

The projector derivative is

[
\dot\Pi_R
=========

\dot U_RG_R+U_R\dot G_R.
]

Therefore

[
\boxed{
\dot n\big|_u
=============

## D_RM_P(R,u)[\dot R]

## D_RM_P(R,0)[\dot R]

## \dot J_0[\Pi_Ru]

J_0\dot\Pi_Ru
}
]

or, expanded,

[
\dot n\big|_u
=============

## M_R(R,u)[\dot R]

## M_R(R,0)[\dot R]

## \dot J_0[U_RG_Ru]

## J_0\dot U_RG_Ru

J_0U_R\dot G_Ru.
]

For the ADT branch,

[
\boxed{
\dot a\big|_u
=============

s_F
\left[
\dot P_R,\alpha_RG_Ru
+
P_R\dot\alpha_R,G_Ru
+
P_R\alpha_R\dot G_Ru
\right].
}
]

The complete source-RHS derivative is

[
\boxed{
\begin{aligned}
\dot b\big|*u={}&
\dot B*{\mathrm p}m_0
+
B_{\mathrm p}\dot m_0\
&+
\dot B_{\mathrm g}n
+
B_{\mathrm g}\dot n\big|*u\
&+
\dot B*{\mathrm a}a
+
B_{\mathrm a}\dot a\big|_u.
\end{aligned}
}
]

Here (\dot B_{\mathrm p}), (\dot B_{\mathrm g}), and (\dot B_{\mathrm a}) include both cavity changes and derivatives of the respective point, Gaussian, and translated-density source potentials.

At fixed (u),

[
\dot\sigma\big|_u
=================

S_R\left(
\dot b\big|_u-\dot A_R\sigma
\right),
]

and

[
\boxed{
\dot r\big|_u
=============

## -\dot L_R\sigma

L_RS_R
\left(
\dot b\big|_u-\dot A_R\sigma
\right).
}
]

---

## Exact ADT allocation derivative

If ADT is defined by minimum work

[
\min_x\frac12x^TH_Rx
\quad\text{subject to}\quad
C_Rx=p_{\mathrm{mol}},
]

then

[
P_R
===

H_R^{-1}C_R^T
\left(
C_RH_R^{-1}C_R^T
\right)^{-1}.
]

Define

[
X_R=H_R^{-1},
\qquad
S_R^{\mathrm{ADT}}
==================

C_RX_RC_R^T.
]

Then

[
\dot X_R
========

-X_R\dot H_RX_R,
]

[
\dot S_R^{\mathrm{ADT}}
=======================

\dot C_RX_RC_R^T
+
C_R\dot X_RC_R^T
+
C_RX_R\dot C_R^T,
]

and

[
\boxed{
\begin{aligned}
\dot P_R={}&
\dot X_RC_R^T(S_R^{\mathrm{ADT}})^{-1}
+
X_R\dot C_R^T(S_R^{\mathrm{ADT}})^{-1}\
&-
X_RC_R^T(S_R^{\mathrm{ADT}})^{-1}
\dot S_R^{\mathrm{ADT}}
(S_R^{\mathrm{ADT}})^{-1}.
\end{aligned}
}
]

For a pure sum-of-atomic-dipoles constraint, (C_R) is usually constant, but the Coulomb work matrix (H_R) remains geometry dependent.

---

## Adjoint derivative of the explicit operational scalar

Assume the registered ddPCM scalar is

[
\Gamma(R,u)
===========

-\frac{f}{2}
b(R,u)^TA_R^{-1}b(R,u).
]

Define

[
\sigma=A_R^{-1}b,
\qquad
\tau=A_R^{-T}b.
]

No symmetry of (A_R) is required for the following formulas.

The field partial is

[
\boxed{
\Gamma_u
========

-\frac f2
K_R(u)^T(\sigma+\tau).
}
]

The fixed-(u) nuclear partial is

[
\boxed{
\dot\Gamma\big|_u
=================

-\frac{\dot f}{2}b^T\sigma
-\frac f2(\sigma+\tau)^T\dot b\big|_u
+\frac f2\tau^T\dot A_R\sigma.
}
]

At fixed solvent, (\dot f=0).

If (A_R=A_R^T), then (\tau=\sigma), and these simplify to

[
\Gamma_u=-fK_R(u)^T\sigma,
]

[
\dot\Gamma\big|_u
=================

-f\sigma^T\dot b\big|_u
+
\frac f2\sigma^T\dot A_R\sigma.
]

Let

[
H_u
===

# D_ur(R,u^*)

I-L_RA_R^{-1}K_R(u^*).
]

Solve the adjoint equation

[
\boxed{
H_u^T\lambda=\Gamma_u.
}
]

Then the exact directional derivative of

[
E(R)=E_{\mathrm{vac}}(R)+\Gamma(R,u^*(R))
]

is

[
\boxed{
\dot E
======

\dot E_{\mathrm{vac}}
+
\dot\Gamma\big|_u
-----------------

\lambda^T\dot r\big|_u.
}
]

This is the conservative derivative of the explicitly declared branch scalar. It does not require a common MACE/PCM variational functional.

It does require:

[
\det H_u\neq0
]

and a locally single-valued (C^1) root branch. If the root switches branches, the scalar can become nondifferentiable or branch dependent.

## Complete nuclear-coordinate ledger

A production force must include all of the following:

1. (D_RE_{\mathrm{vac}}).
2. (D_RM_P(R,u)) at the converged nonzero field.
3. (D_RM_P(R,0)).
4. (D_RD_uM_P(R,0)), the mixed MACE-POLAR derivative.
5. (D_R\alpha_{\mathrm{MDP}}(R)).
6. (D_RU_R) and (D_RG_R), including any centroid, origin, local-frame, or feature-normalization dependence.
7. (D_RP_{\mathrm{ADT}}(R)), including its work metric and constraint solve.
8. Motion of the ADT free-atom translation profiles, even if their radial shapes are element-fixed.
9. Point-source coefficient and centre derivatives in (B_{\mathrm p}).
10. Gaussian source-centre and basis derivatives in (B_{\mathrm g}).
11. ADT boundary-potential derivatives in (B_{\mathrm a}).
12. The complete cavity/operator derivative (\dot A_R): sphere centres, exposure functions, overlap maps, quadrature points and weights, normals, translations, and any geometry-dependent radii.
13. The complete two-width receiver derivative (\dot L_R).
14. Any geometry-dependent left/right metric or dielectric prefactor used in the actual ddPCM energy expression.
15. The implicit root response through the adjoint (\lambda).
16. The derivative of any recentering or rigid-frame operation applied before either checkpoint.
17. Smooth neighbor-list/cutoff and cavity-topology behavior.

Omitting any one of items 2–15 generally gives a force that is not the derivative of the registered scalar.

---

# D. Whether to retain the MDP latent permanent source

## Theorem D1: supervised molecular closure does not identify a PCM source

Let (C) map an atomwise latent MDP decomposition (c) to its supervised molecular dipole:

[
Cc=\mu_{\mathrm{MDP}}.
]

Any

[
k\in\ker C
]

is invisible to the molecular dipole loss. If polarizability supervision is also included, there remains a large atomwise gauge unless the complete source response was directly constrained.

Let (B) map the latent point source to the ddPCM right-hand side. For symmetric positive-definite (A),

[
G(c)
====

-\frac f2(Bc)^TA^{-1}(Bc).
]

For another equally dipole-consistent partition (c+k),

[
\begin{aligned}
G(c+k)-G(c)
={}&
-f(Bc)^TA^{-1}(Bk)\
&-
\frac f2(Bk)^TA^{-1}(Bk).
\end{aligned}
]

Thus PCM invariance under the latent gauge would require

[
\ker C\subseteq\ker B.
]

That is generically false: redistributions with zero total molecular dipole can produce substantial boundary MEPs and reaction energies.

## Physical inference from your frozen records

Your two molecule-disjoint source panels show that this is not merely a formal possibility:

* MDP latent point source: (8.19) kcal/mol fixed-source MAE;
* POLAR zero-field point source: (1.346) and (1.461) kcal/mol;
* molecular-level closure changes the prospective result by only (2.23%).

The permanent error is therefore predominantly in continuum-visible spatial structure, not in the three-dimensional molecular dipole.

There is **no scientifically defensible production role** for the MDP latent atomwise (q/p) after these results. Its only legitimate remaining uses are:

* frozen negative control;
* historical v2 reproduction;
* source-ablation analysis;
* debugging and provenance.

The supervised MDP total dipole may remain a diagnostic. The supervised (\alpha_{\mathrm{MDP}}) has a legitimate role in the uniform response branch. Neither justifies retaining the unsupervised atomwise partition.

MACE-POLAR atomwise partials should also not be advertised as unique chemical atomic charges. The justified object is the complete frozen source-plus-decoder field that passed the target-independent source tests, not the semantic uniqueness of each atomic number. ([MACE Documentation][4])

---

# E. Smallest defensible prequalification before the 505 run

The smallest adequate contract has two parts:

1. all 60 prospective source cases for cheap static checks;
2. a 12-case coupled mechanism panel for roots, derivatives, rotations, and tails.

No solvation target is needed.

## Selection of the 12 coupled cases

Select sequentially, skipping duplicates and taking the next ranked case in the same category:

1. the four largest prospective
   (|\delta b|_{A^{-1}}) source errors;
2. the two smallest source-to-exposed-boundary clearances, normalized by the 1.5-Å width;
3. the two largest (|\alpha_{\mathrm{MDP}}|_2);
4. the largest polarizability anisotropy;
5. one cation and one anion or zwitterionic/charge-separated case;
6. one near-symmetric, low-dipole cancellation case.

Use a frozen molecule identifier as the tie-break. This includes the (4.53) kcal/mol maximum and the prospective upper tail without selecting on solvation energy.

If the 505 set uses materially different cavity/operator families, select geometry–operator pairs so that the largest frozen estimate of

[
|L_RA_R^{-1}B_R|
]

and the minimum cavity clearance are represented.

---

## Gate E1: algebraic closure on all 60 cases

Check

[
G_RU_R=I_3,\qquad
\Pi_R^2=\Pi_R,
]

[
Q_{\mathrm{pt}}m_0=Q_{\mathrm{mol}},
]

[
Q_Gn(R,u)=0,
]

[
Q_{\mathrm{ADT}}P_Rp=0,
]

[
\mathcal D_{\mathrm{ADT},R}P_Rp=p.
]

Also check that point and Gaussian coefficient interpretations preserve the same first two molecular moments:

[
Q_{\mathrm{pt}}c=Q_Gc,
]

[
\mathcal D_{\mathrm{pt},R}c
===========================

\mathcal D_{\mathrm G,R}c.
]

The ADT work matrix and its constrained Schur complement must be positive definite and full rank. No pseudoinverse, eigenvalue clipping, or active-subspace deletion is allowed.

For deterministic float64 evaluation, identity residuals should scale as

[
O!\left(
\widehat\kappa,\epsilon_{\mathrm{mach}}
+
\tau_{\mathrm{linear}}
\right),
]

not as a fixed chemistry-dependent error. A suitable preregistered numerical envelope is

[
100,\widehat\kappa,\epsilon_{\mathrm{mach}}
+
10,\tau_{\mathrm{linear}},
]

with all condition estimates and solver residuals reported.

Also require

[
\alpha_R=\alpha_R^T
]

to numerical precision and no physically significant negative eigenvalue. A negative static polarizability eigenvalue is a fail, not a value to clip.

---

## Gate E2: fixed-source MEP and reaction norm

At (u=0), v3 must reduce exactly to the standalone MACE-POLAR zero-field point source. Therefore the assembled implementation must reproduce the already frozen prospective record:

[
\mathrm{MAE}=1.461,\qquad
q_{95}=3.67,\qquad
\max=4.53\ {\rm kcal/mol},
]

up to continuum-solver and reduction error.

In addition, record the boundary MEP error

[
|\delta V|_W
============

\left(
\delta V^TW_R\delta V
\right)^{1/2}
]

and, in an SPD continuum representation,

[
|\delta b|_{A^{-1}}
===================

\left(
\delta b^TA^{-1}\delta b
\right)^{1/2}.
]

For

[
G(b)=-\frac f2b^TA^{-1}b,
\qquad
b=b_{\mathrm{ref}}+\delta b,
]

one has the exact bound

[
|\Delta G|
\le
f|b_{\mathrm{ref}}|*{A^{-1}}
|\delta b|*{A^{-1}}
+
\frac f2|\delta b|_{A^{-1}}^2.
]

Thus fixed-source energy error is expected to scale linearly with the (A^{-1}) source error at small error, with a quadratic remainder. This supplies the correct continuum-visible norm without inventing a hydration-energy threshold.

If the implemented v3 source does not reproduce its standalone source record, the architecture fails before any closed-loop calculation.

---

## Gate E3: JVP/VJP and mixed-derivative audit

For each of the 12 cases, test at both (u=0) and (u=u^*):

* three independent native-field directions;
* three independent cotangent directions;
* three internal nuclear-coordinate directions.

For every operator (J_0), (J(u)), (K_R(u)), and (r_u), verify the adjoint identity

[
y^TJx=x^TJ^Ty.
]

The normalized defect should scale as

[
O(\widehat\kappa\epsilon_{\mathrm{mach}})
]

plus the linear-solve residual.

Use a centered finite-difference sweep, not a single chosen step:

[
\frac{F(x+hv)-F(x-hv)}{2h}.
]

The error should show

[
O(h^2)+O(\epsilon_{\mathrm{mach}}/h)
]

behavior: a second-order regime followed by a roundoff regime.

Explicitly test the mixed derivative

[
D_RD_uM_P(R,0)[\dot R,h]
]

against centered differences of the zero-field JVP. This is the term most likely to be accidentally omitted.

Finally, compare the complete adjoint force against centered finite differences of the complete operational scalar. The convergent regime must again be second order.

---

## Gate E4: uniform-tangent cancellation and sign

At (u=0), verify directly

[
\left[J_0-J_0U_RG_R\right]U_R=0
]

at the numerical-identity scale.

For finite amplitudes (t),

[
n(R,tU_Rh)=O(t^2).
]

Consequently,

[
\frac{|n(R,tU_Rh)|}
{|n(R,\tfrac12tU_Rh)|}
\longrightarrow4
]

before roundoff.

The induced molecular dipole must satisfy

[
\mu_{\mathrm{ind}}(tU_Rh)
=========================

s_Ft\alpha_Rh+O(t^2).
]

This single test simultaneously certifies:

* subtraction of the complete MACE-POLAR uniform tangent;
* the ADT dipole closure;
* the (G/U) convention;
* and the physical sign.

Also shift the origin used for the affine potential. The density response and induced dipole must remain invariant to the resulting constant-potential mode.

---

## Gate E5: certified local root

At each coupled test case compute

[
H_*
===

I-L_RA_R^{-1}K_R(u_*).
]

A small residual alone is not sufficient.

At minimum, obtain a certified lower bound

[
s_{\mathrm{lb}}
\le
\sigma_{\min}(H_*),
\qquad
s_{\mathrm{lb}}>0.
]

Then the local linearized state uncertainty is bounded by

[
|\delta u|
\lesssim
\frac{|r(u_*)|}{s_{\mathrm{lb}}}.
]

For an actual local uniqueness certificate, bound the Jacobian Lipschitz constant (L_H) on a ball of radius (\rho) around the computed state. It is sufficient that

[
\frac{|r(u_*)|}{s_{\mathrm{lb}}}
+
\frac{L_H}{2s_{\mathrm{lb}}}\rho^2
\le\rho
]

and

[
\frac{L_H\rho}{s_{\mathrm{lb}}}<1.
]

These inequalities make the Newton map a contraction on that ball and certify one exact root in it.

If production uses plain Picard iteration, additionally require

[
\rho!\left(L_RA_R^{-1}K_R(u_*)\right)<1.
]

For Newton or another root solver, Picard contractivity is not necessary, but invertibility of (H_*) is.

Forward/reverse continuation and multiple deterministic starting points should recover the same root. That is branch evidence, not a substitute for the local certificate.

---

## Gate E6: rotations and rigid motions

For three fixed generic rotations per case, test

[
M_P(QR,D_U(Q)u)
===============

D_C(Q)M_P(R,u),
]

[
\alpha_{\mathrm{MDP}}(QR)
=========================

Q\alpha_{\mathrm{MDP}}(R)Q^T,
]

[
P_{\mathrm{ADT}}(QR)Q
=====================

D_A(Q)P_{\mathrm{ADT}}(R),
]

and the (U/G) intertwining identities given above.

At the closed-loop level verify

[
u^*(QR)=D_U(Q)u^*(R),
]

[
E(QR)=E(R),
]

[
F(QR)=QF(R).
]

The exact MACE/ADT/operator components should be at the condition-scaled floating-point floor. The complete ddPCM result may retain a finite angular-discretization floor. Measure that floor independently by rotating a fixed exact source through the same continuum implementation. The hybrid excess must be no larger than the propagated continuum floor, and one higher-order audit discretization must reduce the continuum component rather than reveal a plateau.

Also perform one rigid translation on a charged and a neutral case to expose affine-potential-origin and permanent-dipole bookkeeping errors.

---

## Gate E7: adversarial chemistry and tails

The (4.53) kcal/mol source-tail case and the prospective (q_{95}) cases must be carried through:

* source assembly;
* root certificate;
* field-response tests;
* force finite differences;
* rotations;
* small internal coordinate displacements.

The panel must include high polarizability, high anisotropy, charge separation, small cavity clearance, and near-symmetry. Unsupported elements, charge states, or spins are a hard fail. They must not be hidden by substituting MDP latent sources.

No full 505 run is justified if any structural identity, derivative identity, root certificate, or objectivity gate fails.

---

# F. Missing (l\ge2) and penetration information

## Identifiability theorem F1

Let (\mathcal O) denote all frozen observables used by v3:

* one atom-centred (l\le1), one-radial-block source;
* molecular dipole;
* molecular polarizability;
* and the response of those outputs to the available native (l\le1) field features.

For continuum recovery to be identifiable, the exact continuum source map (B\rho) would have to factor through those observables:

[
B\rho=D(\mathcal O(\rho))
]

for some uniquely defined decoder (D).

Consider an atom-centred perturbation

[
\delta\rho_A(\mathbf r)
=======================

f(r_A)Y_{2m}(\widehat{\mathbf r_A}).
]

By angular orthogonality it can be chosen so that

[
\delta Q=0,\qquad
\delta\mu=0,
]

and it has zero projection into every atom-centred (l\le1) output channel. It does not alter the supplied molecular polarizability constraint either if introduced as a field-independent static perturbation.

Nevertheless, its electrostatic potential on a generic molecular cavity is nonzero:

[
B_R\delta\rho_A\neq0.
]

Therefore (\rho) and (\rho+\delta\rho_A) produce the same available (l\le1) observables but different continuum right-hand sides. Hence no decoder from those observables can recover the missing quadrupolar source for both densities.

The radial argument is analogous. One can choose a spherically symmetric radial perturbation with:

[
\int\delta\rho,d\mathbf r=0,
]

zero overlap with the one frozen radial source degree of freedom, and zero low moments, while changing the enclosed charge as a function of radius. At boundary points sampling the penetration region,

[
\delta V(\mathbf s)\neq0.
]

Thus the one-width source does not identify radial penetration either.

## What is already present

Displaced atom-centred monopoles and dipoles do generate nonzero **global** molecular multipoles of arbitrarily high order when translated to a common origin. ADT allocations do the same.

Those translated moments are already determined by (q_A,p_A,R_A). The no-go concerns:

* independent atom-centred quadrupoles and higher multipoles;
* anisotropic short-range density structure;
* additional radial degrees of freedom;
* and penetration information not fixed by the selected point/Gaussian/ADT profiles.

## Why the richer receiver does not solve the source problem

MACE-POLAR explicitly permits a potential-feature basis richer than its source-density basis. A richer receiver tells the model more about the applied potential; it does not manufacture missing output coefficients in the source density. ([arXiv][2])

Likewise, the local backbone’s hidden (l\le3) features do not provide an identified quadrupole source. Their channels have latent basis/gauge freedom, and no frozen supervised readout maps them to a physical (l=2) density.

Differentiating a conditioned MACE energy with respect to the native eight-channel field would produce a covector in that finite receiver space. It might define a new operational continuum coupling through (L_R^T), but:

* it still contains no independent atom-centred (l\ge2) channel;
* interpreting its radial covector as a density requires a metric/Riesz map or decoder;
* and it would be a different architecture, not recovery of the missing density used by v3.

Point decoding, ADT translation, free-atom profiles, and analytic multipole translation all impose additional priors. They do not recover information contained in the checkpoint.

## Physical inference

The (4.53) kcal/mol tail may be caused by (l\le1) truncation, penetration, source partitioning, point decoding, domain extrapolation, or some combination. The scalar tail value does not identify which missing density component should be added.

Therefore, if v3 fails after the frozen prequalification and the one permitted 505 evaluation, there is no identified zero-training route that can honestly claim to reconstruct the absent continuum-visible information.

The next scientifically defensible model would require new QM-electrostatic supervision, such as:

* a multi-radial, (l\ge2) density or multipole head;
* a molecular/cavity MEP head trained on geometry-independent electrostatic grids;
* or a field-conditioned electrostatic scalar whose derivative with respect to a sufficiently complete external potential is the conjugate density.

That training can remain entirely free of experimental or solvation-energy targets. It is not a residual correction; it supplies an observable that the present checkpoints do not identify.

**Final adjudication:** v3 is mathematically well-defined, exactly removes the MACE-POLAR uniform tangent at linear order, and is the strongest available zero-training architecture. Its point permanent source and mixed source regularizations are operationally valid but physically calibrated rather than uniquely derived. The MDP latent permanent source has no remaining production justification. Proceed only through the stated target-independent prequalification; a failure attributable to unresolved near-field information closes the zero-training route rather than licensing another latent decoder.

POLAR-ZERO ADT HYBRID DECISION

[1]: https://github.com/Nilsgoe/MACE-MDP "GitHub - Nilsgoe/MACE-MDP: MACE-MDP is a machine-learning model for predicting molecular dipole moments and fully anisotropic polarizability tensors for organic systems. · GitHub"
[2]: https://arxiv.org/html/2602.19411v1 "https://arxiv.org/html/2602.19411v1"
[3]: https://pubmed.ncbi.nlm.nih.gov/26584117/?utm_source=chatgpt.com "Fast Domain Decomposition Algorithm for Continuum ..."
[4]: https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html "Electrostatic MACE — mace 0.3.13 documentation"
