# Terminal decision

| Claim/model                                                                                                         | Decision                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A. Unchanged MACE-POLAR response plus a work-generated additive correction as Tier-V/common-functional response** | **Terminal NO-GO.** The observed (0.77%-2.49%) molecular skew is already a structural contradiction.                                                                                                                           |
| **(M_{\rm avg}) as an explicitly nonvariational operational constitutive map**                                      | **Not structurally rejected. GO only to a frozen target-independent finite-field/root/force audit. Do not open the accuracy panel yet.**                                                                                       |
| **Nearest genuinely work-generated zero-training fallback**                                                         | Replace the native induced response by an explicit three-dimensional variational dipole state lifted through the POLAR tangent span. Retaining the native response while adding that state double-counts the uniform response. |

The (1.13-1.22) chart-duality defect does **not** prove that (W) is wrong. It proves that the arithmetic direct-sum complement and the work-dual complement are very different.

---

## 1. What the skew terminally rejects

Let (\mathsf U:\mathbb R^3\to\mathcal U) denote the affine embedding, reserving (U) conceptually for the receiver space. The exact affine work identity is

[
W\mathsf U=C^T.
]

For a work-generated response, there must locally exist a scalar (\Psi(u)) satisfying

[
\nabla_u\Psi(u)=W^TM(u).
]

At zero field this requires

[
H_P:=W^TJ
]

to be symmetric on the gauge-free accessible receiver space. Its restriction to affine fields is

[
\mathsf U^T H_P\mathsf U
========================

# \mathsf U^TW^TJ\mathsf U

# C J\mathsf U

C_P.
]

Therefore,

[
W^TJ=(W^TJ)^T
\quad\Longrightarrow\quad
C_P=C_P^T.
]

The measured

[
\frac{|\operatorname{skew}C_P|_2}
{|\operatorname{sym}C_P|_2}
=0.00770\text{--}0.02490
]

is many orders above a numerical reciprocity residual. One counterexample in the claimed domain is sufficient to reject a globally work-generated unchanged response.

There is an even sharper obstruction. Let (K) be any additive correction whose own work Jacobian is reciprocal:

[
W^TK=(W^TK)^T.
]

Then

[
C K\mathsf U
============

\mathsf U^TW^TK\mathsf U
]

must be symmetric. Consequently,

[
\operatorname{skew}!\left(C(J+K)\mathsf U\right)
================================================

\operatorname{skew}(C_P).
]

So:

> **No additive work-reciprocal correction can cancel the native molecular skew while retaining the native (J).**

This is why the Q20 work-dual correction can change only the symmetric part. Its failure to cancel the skew is not a weakness of its implementation; it is mathematically unavoidable.

For (M_{\rm avg}),

[
K_{\rm avg}
===========

B(X-I)G_{\rm avg},
\qquad
X=C_P^{-1}(-\alpha_D),
]

and

[
C K_{\rm avg}\mathsf U
======================

-\alpha_D-C_P.
]

Hence

[
\operatorname{skew}(C K_{\rm avg}\mathsf U)
===========================================

-\operatorname{skew}(C_P).
]

Its correction is therefore necessarily nonreciprocal under (W). That conclusion does not depend on the numerical size of the arithmetic-chart defect.

### Does that also reject the operational model?

**No.**

A constitutive fixed-point map does not have to be the gradient of the independently declared scalar ledger. One may define

[
H(R,u)=0,
\qquad
\Phi_{\rm op}(R)=\Phi(R,u_\ast(R)),
]

and obtain a conservative force for (\Phi_{\rm op}) through the complete implicit derivative, even when (M(u)) itself is nonintegrable. The required conditions are unique root selection, nonsingular root Jacobian, differentiability, and differentiation of the same declared scalar including all implicit response.  The corresponding adjoint derivative is conservative because it differentiates an explicit scalar composed with the selected root; that still does not make the response variational or establish a common functional. 

Thus (M_{\rm avg}) is scientifically admissible under claim B provided it is described narrowly:

* it is an operational source constitutive law;
* it imposes the supervised molecular uniform susceptibility exactly;
* it uses the POLAR uniform tangent span as an atomwise topology prior;
* its source work is path-dependent in general;
* no electronic polarization energy may be inferred by integrating (M_{\rm avg}) against (u), unless a particular integration path is explicitly part of the separately frozen ledger;
* its atomwise source correction is not thereby established as a physical observable.

The (35.6%-58.4%) tangent change is substantial and makes a finite-field audit mandatory, but it is not a category error.

---

## 2. Correct interpretation of the (1.13-1.22) duality defect

Define the arithmetic oblique projector

[
P_{\rm avg}=\mathsf U G_{\rm avg},
\qquad
N_{\rm avg}=I-P_{\rm avg}.
]

Because (G_{\rm avg}\mathsf U=I),

[
G_{\rm avg}N_{\rm avg}=0,
]

so (N_{\rm avg}) maps into the chosen arithmetic complement.

Using

[
C_P^T
=====

B^TW\mathsf U,
]

we obtain

[
\begin{aligned}
C_P^TG_{\rm avg}-B^TW
&=
B^TW\mathsf U G_{\rm avg}-B^TW\
&=
-B^TW(I-\mathsf U G_{\rm avg})\
&=
-B^TW N_{\rm avg}.
\end{aligned}
]

Therefore the reported defect is exactly

[
\boxed{
\frac{|B^TW N_{\rm avg}|}
{|B^TW|}.
}
]

It measures how strongly the uniform source tangent (B) does physical work against receiver directions that the arithmetic chart assigns to its zero-uniform-coordinate complement.

It is **not** a test of whether (W) is the correct physical pairing.

The work-dual chart is

[
G_{\rm work}
============

C_P^{-T}B^TW,
]

and

[
G_{\rm work}-G_{\rm avg}
========================

C_P^{-T}B^TW N_{\rm avg}.
]

Thus the large defect says that the two complements are strongly oblique relative to one another.

### Why this is expected in the 4-source/8-receiver architecture

Consider a gradient pattern that is positive in one radial block and negative in the duplicated block. Equal arithmetic averaging can give

[
G_{\rm avg}u=0,
]

while (W), which pairs through the learned-source embedding in the first (1.5)-Å block, can still give

[
B^TWu\ne0.
]

Such a direction is “nonuniform” in the arithmetic direct-sum sense but is not work-orthogonal to the uniform induced-source tangent.

A rectangular pairing is normal here. (W) does not need to be square, invertible, or a metric. It is a bilinear source–receiver pairing between intentionally different representations.

Evidence that (W) is wrong would instead be:

[
W\mathsf U\ne C^T,
]

failure of direct source–affine-field work,

[
s^TW\mathsf Ug\ne(Cs)^Tg,
]

or disagreement with the declared public radial embedding. The defect involving (G_{\rm avg}) cannot establish any of those failures.

Indeed, the observed

[
|G_{\rm work}\mathsf U-I|\le1.39\times10^{-15}
]

supports the work identity on the response span (B).

### Which chart is better for claim B?

For a nonvariational operational construction, (G_{\rm avg}) has a legitimate advantage:

* MDP supplies information only about truly affine uniform fields;
* (G_{\rm avg}) extracts exactly that architectural coordinate;
* the correction vanishes on the declared arithmetic complement.

By contrast, (G_{\rm work}) extends the MDP correction to nonuniform receiver patterns whenever they have work overlap with (B). That is appropriate for a work-based model, but it is a stronger extrapolation of the molecular (\alpha_D) into nonuniform fields.

So the large defect does not make (G_{\rm avg}) invalid. It shows that the choice between (G_{\rm avg}) and (G_{\rm work}) is scientifically consequential and claim-dependent.

---

## 3. Does symmetric (\alpha_D) justify canceling the molecular skew?

### At the molecular uniform-response level: yes

The full supervised tensor (\alpha_D), not symmetry alone, supplies the desired molecular response

[
\frac{\partial\mu}{\partial g}
==============================

-\alpha_D.
]

Its symmetry says the physical static molecular susceptibility has no antisymmetric component under the frozen semantics. Therefore it is legitimate to treat the observed skew of (C_P) as model error in that molecular observable and replace it operationally.

That does **not** establish that a particular atomwise correction is uniquely physical. The atomwise distribution becomes unique only after imposing the POLAR-span and chart constraints.

### Uniqueness theorem for (K_{\rm avg})

Let (G=G_{\rm avg}). Assume:

[
G\mathsf U=I,
\qquad
C_P=CB\ \text{nonsingular}.
]

Suppose (K) satisfies:

1. (C(J+K)\mathsf U=-\alpha_D);
2. (\operatorname{range}(K)\subseteq\operatorname{range}(B));
3. (K\ker G=0);
4. the correction is linear in (u), so all pure field derivatives of order (n\ge2) remain unchanged.

Because (G\mathsf U=I), every receiver decomposes as

[
u=\mathsf UGu+\left(u-\mathsf UGu\right),
]

with the second term in (\ker G). Hence

[
Ku=K\mathsf UGu.
]

Define (L=K\mathsf U). Then

[
K=LG.
]

The range condition and full column rank of (B) imply

[
L=BA
]

for a unique (3\times3) matrix (A). Molecular closure requires

[
C_P+CBA
=======

# C_P+C_PA

-\alpha_D.
]

Thus

[
\boxed{
A=C_P^{-1}(-\alpha_D-C_P)=X-I
}
]

and

[
\boxed{
K_{\rm avg}=B(X-I)G_{\rm avg}.
}
]

So under the stated constraints the correction is **unique**, not merely minimum-norm.

Charge conservation follows automatically when the uniform POLAR tangent is charge neutral. If (\ell_Q^T) extracts total charge and

[
\ell_Q^TB=0,
]

then

[
\ell_Q^TK_{\rm avg}=0.
]

The correction is linear in (u), so

[
D_u^n(K_{\rm avg}u)=0,\qquad n\ge2.
]

Geometry dependence of (B,X,G_{\rm avg}) creates geometry–field mixed derivatives, but it does not alter second or higher **pure field** derivatives.

### Where a metric would be needed

No metric is needed under the complete constraints above.

If the range-(B) constraint is relaxed, there are infinitely many source lifts because one can add source directions invisible to (C). Then “minimum change” requires a prospectively fixed positive source metric (H_s). The rectangular work pairing (W) is not itself a norm.

A dimensionally valid choice is the Coulomb self-energy Gram matrix of the same (1.5)-Å Gaussian monopole/dipole source basis. That metric places (q) and (p) in one physical energy norm without treating their raw numerical coordinates as commensurate.

For (K=LG), define

[
\widehat C=
\begin{pmatrix}
C\
\ell_Q^T
\end{pmatrix},
\qquad
\widehat\Delta=
\begin{pmatrix}
-\alpha_D-C_P\
0
\end{pmatrix}.
]

The minimum-(H_s)-norm lift is

[
\boxed{
L_\star
=======

H_s^{-1}\widehat C^T
\left(
\widehat C H_s^{-1}\widehat C^T
\right)^{-1}
\widehat\Delta,
\qquad
K_\star=L_\star G,
}
]

assuming the constrained Gram matrix is nonsingular.

That broader minimum-norm solution is unnecessary here because the POLAR-span restriction already fixes (K) uniquely.

### What remains unobserved

The missing observable is not an antisymmetric molecular polarizability. The physical molecular target has none.

What remains unobserved is the distributed source response under nonuniform fields. Choosing (\operatorname{range}(B)) and (G_{\rm avg}) is a zero-training structural prior for that distribution, not an experimentally established atomwise decomposition.

---

## 4. Explicit three-dimensional induced-dipole fallback

### Work-dual source lift

Let (z\in\mathbb R^3) represent the total induced molecular dipole. Require

[
CL=I,
\qquad
\ell_Q^TL=0,
\qquad
\operatorname{range}(L)\subseteq\operatorname{range}(B).
]

Writing (L=BQ), the condition (CL=I) gives

[
C_PQ=I.
]

Therefore

[
\boxed{
L=BC_P^{-1}.
}
]

Again, this is unique under the topology constraint; a minimum norm is not needed.

Its work-conjugate generalized field coordinate is

[
\boxed{
G_L=L^TW=C_P^{-T}B^TW=G_{\rm work}.
}
]

For an affine field,

[
G_L\mathsf Ug=g,
]

because (CL=I) and (W\mathsf U=C^T).

Define

[
E_{\rm eff}(u)=-G_{\rm work}u.
]

The quadratic dipole energy is

[
\Psi_z(z;u)
===========

\frac12z^T\alpha_D^{-1}z
-z^TE_{\rm eff}(u)
==================

\frac12z^T\alpha_D^{-1}z
+z^TG_{\rm work}u.
]

Stationarity gives

[
\boxed{
z_\ast(u)=-\alpha_DG_{\rm work}u.
}
]

Lift it into source space:

[
s_z(u)=Lz_\ast(u)
=================

-L\alpha_D L^TWu.
]

Then

[
C,D_us_z,\mathsf U=-\alpha_D
]

and

[
W^TD_us_z
=========

-W^TL\alpha_D L^TW,
]

which is symmetric negative semidefinite. Thus this induced-dipole subsystem is exactly reciprocal and passive.

### Strict zero-training response

The clean strict alternative is

[
\boxed{
M_z(u)
======

## M_{\rm POLAR}(0)

L\alpha_D L^TWu.
}
]

It has:

* the original zero-field POLAR source;
* exact molecular (\alpha_D);
* charge conservation;
* induced sources entirely in the POLAR uniform tangent span;
* a work-generated quadratic response;
* no MDP atomwise latent quantities.

Its cost is substantial: it discards POLAR's native nonlinear and native nonuniform induced response. That is the price of obtaining a genuinely work-generated zero-training response from the available molecular observable.

The (z) subsystem can participate in a common functional with ddPCM only if the PCM source/receiver coupling, signs, cavity dependence, and all remaining energy terms use the same adjoint pairing. The existence of (\Psi_z) alone does not prove Tier V for the complete model.

### Why simply adding (z) double-counts

If one writes

[
M_{\rm POLAR}(u)+s_z(u),
]

then on affine fields the molecular susceptibility becomes

[
C_P-\alpha_D,
]

not (-\alpha_D). Native and MDP response have both been included.

One might instead try to make (z) a correction state. The needed correction polarizability in physical-field coordinates is

[
\beta
=====

\alpha_D+\operatorname{sym}C_P.
]

A passive quadratic correction exists only if

[
\beta\succeq0.
]

It changes the symmetric molecular response while leaving the native skew. This is exactly the interpretation of the Q20 work-dual correction:

[
D=-\beta=-\alpha_D-\operatorname{sym}C_P.
]

So the work-dual candidate is equivalent to adding a reciprocal correction dipole state, provided (\beta) is positive semidefinite. Its passivity has not been established merely by the reported reciprocity residual.

To cancel the full skew while retaining native response, the correction would require

[
\beta_{\rm full}=\alpha_D+C_P,
]

which is nonsymmetric. No ordinary quadratic equilibrium dipole energy can generate it.

### Retaining nonlinear POLAR response without double counting

One can remove the native uniform tangent and retain the nonlinear residual:

[
\boxed{
M_{\rm res+z}(u)
================

## M_{\rm POLAR}(u)

## B G_{\rm work}u

L\alpha_DG_{\rm work}u.
}
]

Since (L=BC_P^{-1}),

[
M_{\rm res+z}(u)
================

M_{\rm POLAR}(u)
+
B\left(C_P^{-1}(-\alpha_D)-I\right)G_{\rm work}u.
]

This keeps all second and higher POLAR field derivatives and replaces only the zero-field uniform tangent. However, the residual

[
M_{\rm POLAR}(u)-BG_{\rm work}u
]

is not thereby made work-integrable. Therefore this remains an operational model unless its residual response separately passes full reciprocity.

Using (G_{\rm avg}) instead in both the subtraction and (z)-equation gives

[
M_{\rm POLAR}(u)-BG_{\rm avg}u
-L\alpha_DG_{\rm avg}u
======================

M_{\rm avg}(u).
]

But then (E_{\rm eff}=-G_{\rm avg}u) is not the work-conjugate field of the lifted source, because

[
L^TW=G_{\rm work}\ne G_{\rm avg}.
]

An explicit (z) variable with (G_{\rm avg}) is therefore only an algebraic repackaging of (M_{\rm avg}); it does not provide new variational justification.

### Coupling to nonuniform ddPCM fields

For the work-generated (z) state,

[
E_{\rm eff}(u)=-L^TWu.
]

A nonuniform ddPCM field can therefore drive (z) whenever it has nonzero work overlap with the lifted molecular-dipole source pattern. Components in

[
\ker(L^TW)=\ker G_{\rm work}
]

do not drive (z).

This is not a claim that a nonuniform field is literally uniform. It is the unique generalized field coordinate conjugate to the chosen molecular dipole lift.

The coupled (z)-PCM equations can be written as

[
u=\mathcal P_R!\left[s_0+Lz\right],
]

[
\alpha_D^{-1}z+L^TWu=0.
]

Their block Jacobian is

[
\begin{pmatrix}
I & -\mathcal P_RL\
L^TW & \alpha_D^{-1}
\end{pmatrix},
]

up to the frozen continuum sign convention. This system has a much cleaner passivity and root-stability interpretation than the nonreciprocal tangent, although it still requires a numerical stability audit.

### Fallback decision

* If the project accepts claim B, **do not replace (M_{\rm avg}) merely to make the internal (z) notation look variational**.
* If a work-generated response is mandatory, implement **(M_z), the z-only replacement**, not “native POLAR plus (z).”
* Retaining a native residual is permissible only as an explicitly operational extension.

---

## 5. Exact rule for opening the audits and panel

## A. Structural invalidity: reject before any finite-field audit

Reject the candidate immediately if any of the following fails:

[
G_{\rm avg}\mathsf U=I,
\qquad
W\mathsf U=C^T,
]

constant-potential gauge invariance,

[
Jz_{\rm gauge}=0,
\qquad
G_{\rm avg}z_{\rm gauge}=0,
]

uniform charge conservation,

[
\ell_Q^TB=0,
]

nonsingularity and smooth geometry dependence of (C_P),

SO(3), translation, and permutation covariance of (B,C_P,X,G_{\rm avg}),

or semantic identity of (\alpha_D) with the desired static clamped-nuclei molecular response.

The reported skew and (G_{\rm avg})-duality defect do **not** belong in this structural-rejection list for claim B.

They do terminally reject claim A for the unchanged native response.

## B. Operational usefulness: frozen target-independent audit

(M_{\rm avg}) may now enter exactly one preregistered audit. It must include:

### Algebra and symmetry

Require, over the complete target-free geometry domain,

[
|G_{\rm avg}\mathsf U-I|\le10^{-10},
]

[
\frac{|C(J+K_{\rm avg})\mathsf U+\alpha_D|}
{|\alpha_D|}
\le10^{-10},
]

charge-response error (\le10^{-10}), and rotation/permutation/translation covariance defects (\le10^{-8}).

### Finite-field domain

Freeze a field envelope before looking at solvation accuracy. It must contain:

* actual target-free ddPCM fields from the intended geometry domain;
* uniform rays along the principal directions of (\alpha_D);
* arithmetic-complement modes (n\in\ker G_{\rm avg});
* especially radial-block difference modes;
* mixed fields
  [
  u=t\mathsf Ug+r n.
  ]

The mixed fields are essential because the measured duality defect predicts the largest nonreciprocal cross effects there.

### Passivity without imposing reciprocity

For claim B, full symmetry of (W^TD_uM_{\rm avg}) is **not** an admission condition.

But its symmetric part must remain passive on the gauge-free, continuum-accessible quotient:

[
-\operatorname{sym}!\left(
W^TD_uM_{\rm avg}(u)
\right)
\succeq
-\tau I,
]

with a frozen relative tolerance such as

[
\tau=
10^{-8}
\max!\left(
1,
|\operatorname{sym}(W^TD_uM_{\rm avg})|_2
\right).
]

At minimum, the uniform molecular differential response must satisfy

[
-\operatorname{sym}!\left(
C D_uM_{\rm avg}(u)\mathsf U
\right)\succ0
]

throughout the operational field envelope.

The antisymmetric/curl part must be recorded, including mixed rectangular loop work, but it is not by itself a rejection criterion for claim B. It becomes a rejection criterion only if the declared ledger uses source work as though it were path-independent.

### Response magnitude and checkpoint support

Because the zero-field tangent changes by (0.356-0.584), require:

* no discontinuity or source-norm blow-up;
* no finite-field sign reversal of the passive molecular response;
* corrected source increments within a prospectively fixed native source-response support envelope;
* no clipping, damping, or field scaling introduced after seeing the audit.

If the checkpoint’s training-field support is unavailable, the claimed deployment domain must be restricted to the prospectively audited ddPCM field envelope.

### Coupled fixed-point stability

For

[
H(R,u)=u-\mathcal F_R[M_{\rm avg}(R,u)],
]

require:

[
|H(R,u_\ast)|\le10^{-10},
]

[
\frac{\sigma_{\min}(H_u)}
{\max(1,|H_u|_2)}
\ge10^{-6},
]

and the same root from zero, native-response, underpolarized, overpolarized, and reverse-continuation starts.

Any competing root, hysteresis, fold, or terminal near-singularity rejects the candidate before the accuracy panel.

### Scalar and force closure

Exactly one scalar ledger must be frozen before target access. Its force must include derivatives of

[
B=J\mathsf U,
\qquad
C_P=CB,
\qquad
X=C_P^{-1}(-\alpha_D),
\qquad
K_{\rm avg}=B(X-I)G_{\rm avg},
]

including

[
dJ,\ d\mathsf U,\ dC,\ d\alpha_D,\ dC_P^{-1},
\ dX,\ dG_{\rm avg},
]

as well as the complete continuum and root adjoint.

Require central finite-difference agreement of the complete scalar and analytic force at approximately (10^{-5}) relative, or a prospectively frozen absolute numerical floor.

A conservative force proves only that the declared ledger was differentiated correctly. It does not repair source-work nonreciprocity.

## C. Chemical validation

Only if every operational gate passes may one frozen accuracy panel be opened.

That panel may answer:

> Does this prospectively frozen operational source map and scalar ledger improve total solution-state predictions?

It may not answer:

* whether (M_{\rm avg}) is work-generated;
* whether its atomwise source is physically unique;
* whether (G_{\rm avg}) is the work-dual chart;
* whether internal electrostatic components have a common-functional meaning.

No post-panel selection is permitted among (M_{\rm avg}), (M_{\rm work}), and (M_z). If (M_{\rm avg}) fails the panel, a newly implemented (M_z) model requires a newly sealed validation lane; it cannot be chosen using the already opened results.

# Final fork

1. **Close claim A now for the unchanged MACE-POLAR response.** The molecular skew is a direct structural counterexample to work reciprocity, and no reciprocal additive correction can remove it.
2. **Do not close claim B merely because (G_{\rm avg}) is not work-dual.** (M_{\rm avg}) is a coherent, unique, zero-training operational tangent replacement.
3. **Authorize (M_{\rm avg}) only for a terminal target-independent finite-field/root/force audit.** The current seven-point zero-field results are not enough to open the accuracy panel.
4. **If that audit passes, open one frozen accuracy panel under an operational-only claim.**
5. **If work generation is required, or if the operational audit fails, implement the z-only response**
   [
   M_z(u)=M_{\rm POLAR}(0)-L\alpha_DL^TWu,
   \qquad
   L=BC_P^{-1},
   ]
   and audit it as a new zero-training model.

So the exact present decision is:

[
\boxed{
\text{Tier V: NO-GO}
\qquad
\text{(M_{\rm avg}) operational audit: GO}
\qquad
\text{accuracy panel now: NO-GO}.
}
]
