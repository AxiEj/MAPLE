## Terminal conclusion

The completed uniform-field audit proves one exact statement on the **uniform-field subspace**:

[
E_{\mathrm{up}}(R,\mathbf F)-E_{\mathrm{raw}}(R,J_R\mathbf F)
=============================================================

+\mathbf F\cdot\boldsymbol\mu_{\mathrm{POLAR}}(R,J_R\mathbf F),
]

under the audited sign convention.

It does **not** uniquely define an explicit-work functional for a general nonuniform checkpoint field (u=L_R\sigma). It does not identify a pairing between (u\in U_8) and either (c_p\in C_p) or (\Delta c_g\in C_g). It also does not convert the nonintegrable four-channel source into an energy gradient.

From the currently audited objects alone:

[
\Phi_0=E_{\mathrm{vac}}+G_{\mathrm{cont}}
]

and

[
\Phi_{\mathrm{raw}}
===================

E_{\mathrm{vac}}
+
\left[E_{\mathrm{raw}}(u^\star)-E_{\mathrm{raw}}(0)\right]
+
G_{\mathrm{cont}}
=================

E_{\mathrm{raw}}(u^\star)+G_{\mathrm{cont}}
]

are both mathematically well-defined operational scalars. Neither is presently identifiable as the unique physically complete electronic-plus-continuum ledger. Both can nevertheless define conservative operational PESs through a block implicit adjoint, provided the operational root is unique, differentiable, and numerically certified.

The one immediate experiment should be a **nonuniform primitive-work and ddPCM-reciprocity replay on the full continuum-active field space**. No development solvation target should be opened before that experiment terminates.

---

# 1. Uniform (+\mathbf E\cdot\boldsymbol\mu) does not determine nonuniform work

Let

[
C=C_p\oplus C_g
]

be the direct-sum hybrid source space, let (U=U_8) be the native field space, and let

[
J_R:\mathbb R^3\longrightarrow U
]

embed a uniform electric field into the native eight-channel representation. Write

[
U_{\mathrm{uni}}=\operatorname{range}J_R.
]

The audit determines the explicit upstream work only on (U_{\mathrm{uni}}), and specifically for the upstream MACE-POLAR source, not automatically for the hybrid source (c_p\oplus\Delta c_g).

## 1.1 Abstract extension nonuniqueness

Suppose some candidate functional (W_R(c,u)) agrees with the uniform audit:

[
W_R(c,J_R\mathbf F)
===================

+\mathbf F\cdot\boldsymbol\mu(c)
\quad
\text{for all }\mathbf F.
]

Assume (U_{\mathrm{uni}}) is a proper subspace of the continuum-active field space. Choose any nonzero functional

[
\ell\in U^*
]

such that

[
\ell(J_R\mathbf F)=0
\quad
\text{for all }\mathbf F.
]

For any nonzero scalar functional (h:C\to\mathbb R), define

[
W_R'(c,u)
=========

W_R(c,u)+h(c)\ell(u).
]

Then

[
W_R'(c,J_R\mathbf F)
====================

W_R(c,J_R\mathbf F)
]

for every audited uniform field, but generally

[
W_R'(c,u)\ne W_R(c,u)
]

for nonuniform (u).

Thus the uniform identity determines only the restriction

[
W_R\big|*{C\times U*{\mathrm{uni}}},
]

not an extension to (C\times\operatorname{range}L_R).

Uniqueness would follow only under an additional statement such as

[
\operatorname{range}L_R\subseteq U_{\mathrm{uni}},
]

which is not true for a general molecular reaction potential, or from an independently declared full-space work pairing.

The audit is actually weaker still: it samples the response graph

[
\left{
\big(c_{\mathrm{POLAR}}(J_R\mathbf F),J_R\mathbf F\big)
\right},
]

rather than all (c\in C) and all (u\in U_{\mathrm{uni}}). Arbitrary terms that vanish on this response graph remain invisible.

---

## 1.2 Physical real-space density–potential pairing

If one possesses both:

1. a declared real-space source density
   [
   \rho_c(\mathbf r),
   ]
2. a declared real-space external potential
   [
   \varphi_u(\mathbf r),
   ]

then the physical external work is identifiable as

[
W_{\mathrm{real}}(c,u)
======================

\int \rho_c(\mathbf r)\varphi_u(\mathbf r),d\mathbf r,
]

with the overall sign fixed by the audited convention.

Uniform potentials span only affine functions,

[
\varphi_{\mathrm{uni}}(\mathbf r)
=================================

a+\mathbf f\cdot\mathbf r.
]

Their pairings determine only the total charge and dipole:

[
\int\rho(\mathbf r)\varphi_{\mathrm{uni}}(\mathbf r),d\mathbf r
===============================================================

aQ+\mathbf f\cdot\boldsymbol\mu.
]

Choose any nonzero smooth (\delta\rho) satisfying

[
\int\delta\rho,d\mathbf r=0,
\qquad
\int\mathbf r,\delta\rho,d\mathbf r=0.
]

For example, a quadrupolar density perturbation has these properties. Then

[
\int\delta\rho,\varphi_{\mathrm{uni}},d\mathbf r=0
]

for every uniform field, while for a generic nonuniform potential,

[
\int\delta\rho,\varphi_{\mathrm{nonuni}},d\mathbf r\ne0.
]

Therefore exact uniform-field work identifies charge and dipole work, not the coupling to a general reaction potential.

In the present model, point and Gaussian source primitives are declared, but (u\in U_8) is not itself a real-space potential. A map

[
u\longmapsto\varphi_u
]

or retention of the full continuum state (\sigma\mapsto\varphi_\sigma) is still required.

---

## 1.3 Native feature-space pairing

A native feature-space work requires linear maps

[
D_{p,R}:C_p\longrightarrow U^*,
\qquad
D_{g,R}:C_g\longrightarrow U^*,
]

so that

[
W_{\mathrm{native}}
===================

\left\langle
D_{p,R}c_p+D_{g,R}\Delta c_g,,
u
\right\rangle_{U^*,U}.
]

The uniform audit constrains only

[
J_R^*D_{g,R}
]

on the upstream MACE-POLAR source. It does not determine (D_{g,R}) on the annihilator of (U_{\mathrm{uni}}), and it says nothing by itself about (D_{p,R}).

Moreover, the uniform audit concerns the full upstream MACE-POLAR density

[
c_g^0+\Delta c_g,
]

whereas the hybrid permanent source is (c_p). Even on the uniform-field subspace, the audited upstream permanent dipole need not equal the MACE-MDP permanent point-source dipole.

---

## 1.4 Path integration of the nonintegrable source

Suppose a DualityMap (D_g) were declared. The source would define a one-form

[
\alpha_u(\delta u)
==================

\left\langle
D_g\Delta c_g(R,u),\delta u
\right\rangle.
]

A scalar source work would require this one-form to be exact. A necessary local condition is symmetry of its derivative:

[
D\alpha_u(\delta u_1,\delta u_2)
================================

D\alpha_u(\delta u_2,\delta u_1).
]

Equivalently,

[
\begin{aligned}
&
\left\langle
D_g D_u\Delta c_g[\delta u_1],
\delta u_2
\right\rangle
\
&\qquad =
\left\langle
D_g D_u\Delta c_g[\delta u_2],
\delta u_1
\right\rangle.
\end{aligned}
]

The frozen continuum-active curl defect rejects this condition for the exposed source.

One may choose a path (\gamma) and define

[
W_\gamma(u)
===========

\int_\gamma \alpha,
]

but then generally

[
W_{\gamma_1}(u)\ne W_{\gamma_2}(u).
]

A radial convention,

[
W_{\mathrm{rad}}(u)
===================

\int_0^1
\left\langle
D_g\Delta c_g(R,\lambda u),u
\right\rangle d\lambda,
]

would therefore be an additional operational model choice, not a quantity determined by the source.

This must be distinguished from the passed charging identity for (E_{\mathrm{raw}}):

[
E_{\mathrm{raw}}(R,u)-E_{\mathrm{raw}}(R,0)
===========================================

\int_0^1
\left\langle
\nabla_uE_{\mathrm{raw}}(R,\lambda u),u
\right\rangle d\lambda.
]

That identity is valid because (\nabla_uE_{\mathrm{raw}}) is an exact gradient. It does not imply that the exposed source equals that gradient.

---

# 2. Which (c)-(u) contractions are legitimate?

## 2.1 (\langle c_p+\Delta c_g,u\rangle)

This expression is not presently type-correct.

The two source terms belong to different spaces:

[
c_p\in C_p,
\qquad
\Delta c_g\in C_g.
]

At most one may write the direct-sum element

[
c_p\oplus\Delta c_g\in C_p\oplus C_g.
]

A scalar pairing then requires

[
D_p\oplus D_g:
C_p\oplus C_g\longrightarrow U^*.
]

The apparent similarity of the monopole/dipole coefficient layouts does not supply this map. Point and Gaussian multipoles have different real-space distributions and different source operators (B_p) and (B_g).

---

## 2.2 (\langle\Delta c_g,u\rangle)

This is also undefined without a map

[
D_g:C_g\to U^*.
]

The dimensions, units, radial widths, basis normalization, potential gauge, and Cartesian dipole convention must all be part of that map.

The fact that (\Delta c_g) preserves total charge only gives one gauge property:

[
W_g(\Delta c_g,\varphi+\kappa)
==============================

W_g(\Delta c_g,\varphi),
]

because

[
\Delta Q_g=0.
]

It does not define the nonconstant-potential coupling.

---

## 2.3 The exact factorization criterion

The continuum source work is naturally defined in boundary space. With declared pairings, it has the form

[
W_s(c_s,\sigma)
===============

\langle B_s c_s,\sigma\rangle,
\qquad s\in{p,g}.
]

A native-field pairing represents that same work if and only if there exists (D_s:C_s\to U^*) satisfying

[
\langle D_sc_s,L\sigma\rangle
=============================

\langle B_sc_s,\sigma\rangle
]

for every (c_s,\sigma). Equivalently,

[
\boxed{
L^*D_s=B_s.
}
]

This equation uses adjoints in the declared boundary and native-field metrics; it is not necessarily a Euclidean matrix transpose.

A solution exists if and only if

[
\operatorname{range}B_s
\subseteq
\operatorname{range}L^*.
]

In finite dimensions this is equivalent to

[
\boxed{
\ker L\subseteq\ker B_s^*.
}
]

The obstruction is direct. If (\eta\in\ker L), then

[
L(\sigma+\eta)=L\sigma,
]

but

[
W_s(c_s,\sigma+\eta)-W_s(c_s,\sigma)
====================================

\langle c_s,B_s^*\eta\rangle.
]

If (B_s^*\eta\ne0), the same native field (u) corresponds to different continuum source work. No function of (u) alone can then represent that work.

A necessary rank condition is

[
\operatorname{rank}B_s\le \operatorname{rank}L.
]

It is necessary but not sufficient.

---

## 2.4 Gauge obstruction

A constant potential shift obeys

[
W(c,\varphi+\kappa)
===================

W(c,\varphi)+Q(c)\kappa.
]

Therefore:

* the zero-charge induced increment must annihilate the constant-potential direction;
* a charged permanent source must transform by (Q_p\kappa);
* if (U_8) quotients out the constant-potential direction, a charged-source work cannot be represented solely in (U_8);
* if the constant direction is retained, its normalization must be fixed.

The uniform electric-field audit does not test the constant-potential gauge.

There is also an origin issue. For nonzero total charge, the dipole changes under a shift of origin. The charge and dipole work must transform together. A bare (+\mathbf E\cdot\boldsymbol\mu) identity does not establish this covariance for the hybrid point/Gaussian representation.

---

## 2.5 (\langle c_E,u\rangle)

If

[
c_E(R,u):=\nabla_uE_{\mathrm{raw}}(R,u)\in U^*,
]

then

[
\langle c_E,u\rangle
]

is mathematically type-correct. No additional DualityMap is needed because (c_E) is already in the dual of (U).

It is nevertheless not generally an external-work functional.

The exact scalar difference is

[
\boxed{
E_{\mathrm{raw}}(u)-E_{\mathrm{raw}}(0)
=======================================

\int_0^1
\langle c_E(\lambda u),u\rangle,d\lambda.
}
]

The endpoint contraction

[
\langle c_E(u),u\rangle
]

equals the scalar difference only under a special homogeneity condition. For example, if

[
E_{\mathrm{raw}}(u)-E_{\mathrm{raw}}(0)
]

is homogeneous of degree (k), Euler’s theorem gives

[
\langle c_E(u),u\rangle
=======================

k\left[E_{\mathrm{raw}}(u)-E_{\mathrm{raw}}(0)\right].
]

For a quadratic response, the endpoint contraction is twice the energy change.

Furthermore, (c_E) is a native energy gradient. It is not automatically:

* a point/Gaussian source;
* a ddPCM boundary RHS;
* a physical real-space density;
* or the exposed MACE-POLAR source.

Mapping (c_E) back to (C_p) or (C_g) through a pseudoinverse would require another nonunique model choice.

---

## 2.6 Double-counting condition

Let (W_h) be the signed full interaction work between the hybrid source and the reaction potential. With the standard on-shell half-work convention,

[
G_{\mathrm{cont}}=\frac12W_h.
]

The environment self-energy is then

[
C_{\mathrm{env}}=-\frac12W_h=-G_{\mathrm{cont}}.
]

Suppose an electronic graph scalar decomposes as

[
E_{\mathrm{graph}}
==================

E_{\mathrm{int}}+W_0,
]

where (W_0) is the exact explicit work already included in that graph. The non-double-counting total is

[
\begin{aligned}
\Phi
&=
E_{\mathrm{int}}+W_h+C_{\mathrm{env}}
\
&=
E_{\mathrm{graph}}-W_0+W_h-G_{\mathrm{cont}}
\
&=
\boxed{
E_{\mathrm{graph}}-W_0+G_{\mathrm{cont}}.
}
\end{aligned}
]

Therefore:

* if (W_0=0),
  [
  \Phi=E_{\mathrm{graph}}+G_{\mathrm{cont}};
  ]
* if (W_0=W_h),
  [
  \Phi=E_{\mathrm{graph}}-G_{\mathrm{cont}};
  ]
* if only part of the work is included, that exact part must be subtracted.

An unaudited term such as (\langle\Delta c_g,u\rangle) does not determine (W_0). Adding it can double-count the full interaction, half of it, or a feature-space quantity unrelated to physical work.

---

# 3. Scalars defined by the current objects

Strictly speaking, infinitely many algebraic functions of existing scalars can be written. The primitive scalar generators currently available are

[
E_{\mathrm{vac}}(R),
]

[
\Delta E_{\mathrm{raw}}(R)
==========================

E_{\mathrm{raw}}(R,u^\star)-E_{\mathrm{raw}}(R,0),
]

and

[
G_{\mathrm{cont}}(R,\sigma^\star).
]

Under the scientific restrictions that the vacuum anchor is included once, the native ddPCM energy is included once with its existing sign, and no unaudited coefficient or work correction is introduced, there are exactly two admissible total operational ledgers.

## 3.1 Continuum-only ledger

[
\boxed{
\Phi_0
======

E_{\mathrm{vac}}+G_{\mathrm{cont}}.
}
]

It may claim:

* a vacuum electronic baseline;
* plus native ddPCM electrostatic energy evaluated at the separated operational root;
* a well-defined scalar composition if the root is single-valued.

It may not claim:

* inclusion of field-conditioned electronic distortion cost;
* derivation from a joint stationary electronic-continuum functional;
* that the source is conjugate to the electronic energy;
* a unique physical polarizable-solute free-energy decomposition.

The fact that the source changes self-consistently does not itself insert its electronic deformation cost into (\Phi_0).

---

## 3.2 Raw-field ledger

Because

[
E_{\mathrm{raw}}(R,0)=E_{\mathrm{vac}}(R),
]

the proposed raw ledger is

[
\boxed{
\Phi_{\mathrm{raw}}
===================

E_{\mathrm{raw}}(R,u^\star)+G_{\mathrm{cont}}(R,\sigma^\star).
}
]

It may claim:

* a completely specified operational scalar;
* inclusion of the checkpoint’s raw field-conditioned scalar change;
* a conservative operational PES after complete implicit differentiation;
* exact reproducibility under the frozen graph and root convention.

Without an arbitrary-nonuniform work audit, it may not claim:

* that (\Delta E_{\mathrm{raw}}) is the physical electronic distortion energy;
* that (E_{\mathrm{raw}}) excludes all nonuniform explicit work;
* that (G_{\mathrm{cont}}) is combined with it without double-counting;
* Tier V stationarity;
* source–energy conjugacy.

---

## 3.3 Component scalars

The following are independently well-defined:

[
\Delta E_{\mathrm{raw}},
\qquad
G_{\mathrm{cont}},
\qquad
E_{\mathrm{raw}}(R,u^\star).
]

Their interpretations remain component-level:

* (\Delta E_{\mathrm{raw}}) is a checkpoint graph response scalar;
* (G_{\mathrm{cont}}) is the native ddPCM electrostatic component for the operational source;
* neither alone is a complete solvation energy.

The charging representation

[
\Delta E_{\mathrm{raw}}
=======================

\int_0^1
\left\langle
\nabla_uE_{\mathrm{raw}}(R,\lambda u^\star),
u^\star
\right\rangle d\lambda
]

is not a third ledger. It is an exact alternative evaluation of the same (\Delta E_{\mathrm{raw}}).

---

## 3.4 Uniform-field-only upstream scalar

For (u=J_R\mathbf F), the audited scalar

[
E_{\mathrm{raw}}(R,J_R\mathbf F)
+
\mathbf F\cdot\boldsymbol\mu_{\mathrm{POLAR}}
]

is defined.

It is not a ledger for the ddPCM state unless the ddPCM reaction potential is uniform, and it is not automatically the work of the hybrid MDP-plus-increment source.

---

## 3.5 Nonpolar extension

For separately versioned nonpolar scalar (G_{\mathrm{np}}^{(s,\nu)}(R)), the two operational totals are

[
\Phi_0^{(s,\nu)}
================

\Phi_0^{(s)}+G_{\mathrm{np}}^{(s,\nu)},
]

and

[
\Phi_{\mathrm{raw}}^{(s,\nu)}
=============================

\Phi_{\mathrm{raw}}^{(s)}+G_{\mathrm{np}}^{(s,\nu)}.
]

This adds no information about electrostatic work semantics.

There is no presently justified third ledger containing a (c)-(u) endpoint contraction.

---

# 4. Minimal additional object that resolves the ledger

## 4.1 Mathematically minimal sufficient object

The minimal missing object is a content-addressed, differentiable **WorkCoupling** on the actual continuum-active state:

[
\mathcal W_R:
(C_p\oplus C_g)\times\Sigma_R
\longrightarrow\mathbb R,
]

or, if it genuinely factors through the native field,

[
\mathcal W_R:
(C_p\oplus C_g)\times U_{8,R}
\longrightarrow\mathbb R.
]

For a physical interpretation it must be derived from declared primitives:

[
\mathcal W_R(c_p,\Delta c_g,\sigma)
===================================

\int
\left[
\rho_{p,R}(c_p)
+
\delta\rho_{g,R}(\Delta c_g)
\right]
\varphi_{R,\sigma},d\mathbf r.
]

It must satisfy four conditions.

### Upstream nonuniform-work identity

For the full upstream MACE-POLAR source

[
c_g^{\mathrm{full}}(u)=c_g^0+\Delta c_g(u),
]

require

[
\boxed{
E_{\mathrm{up}}^{\mathrm{arb}}(R,\varphi_\sigma)
------------------------------------------------

# E_{\mathrm{raw}}(R,L_R\sigma)

\mathcal W_{g,R}
\left(c_g^{\mathrm{full}}(L_R\sigma),\sigma\right).
}
]

This must hold for arbitrary continuum-active nonuniform potentials, not only uniform fields.

### Hybrid continuum half-work identity

Require

[
\boxed{
2G_{\mathrm{cont}}
==================

\mathcal W_{p,R}(c_p,\sigma)
+
\mathcal W_{g,R}(\Delta c_g,\sigma).
}
]

The signs must be the native ddPCM signs.

### Gauge covariance

For a constant-potential shift (\kappa),

[
\mathcal W_p(c_p,\varphi+\kappa)
--------------------------------

# \mathcal W_p(c_p,\varphi)

Q_p\kappa,
]

and

[
\mathcal W_g(\Delta c_g,\varphi+\kappa)
---------------------------------------

# \mathcal W_g(\Delta c_g,\varphi)

0.

]

### Native factorization, if a (c)-(u) expression is claimed

There must exist (D_p,D_g) satisfying

[
L_R^*D_{p,R}=B_{p,R},
\qquad
L_R^*D_{g,R}=B_{g,R}.
]

A real-space (\sigma)-dependent work may remain valid even if this last factorization fails. In that case no expression (\langle Dc,u\rangle) is permitted.

A bare content-addressed DualityMap without primitive density or upstream provenance defines an operational convention. It does not by itself establish physical work.

---

## 4.2 Ranked options

### 1. Checkpoint-native arbitrary-potential coupling with primitive density

**Decisiveness:** highest for the current checkpoint.
**Cost:** lowest if the upstream arbitrary-potential work node already exists or can be exposed without changing the model.

Required output:

[
E_{\mathrm{up}}^{\mathrm{arb}}
==============================

E_{\mathrm{raw}}
+
\int\rho_{\mathrm{POLAR}}\varphi_{\mathrm{ext}}.
]

This directly resolves whether the uniform identity extends to the nonuniform reaction-potential class.

### 2. Content-addressed DualityMap derived from primitive point/Gaussian densities

**Decisiveness:** mathematically sufficient for source–field work after compatibility is proved.
**Cost:** moderate.

It must be constructed analytically from the fixed source distributions and native potential features, not calibrated from solvation energies. It must pass

[
L^*D_p=B_p,\qquad L^*D_g=B_g.
]

Without an upstream graph decomposition, it identifies continuum work but not necessarily which work is already present in (E_{\mathrm{raw}}).

### 3. Matched QM density-functional evaluations

**Decisiveness:** strong for validating distortion and continuum components on selected systems.
**Cost:** high.

These can determine whether (\Delta E_{\mathrm{raw}}) behaves like the vacuum-to-solution electronic distortion cost, but finite QM cases do not define a universal native-field work functional.

### 4. New scalar-first field head

**Decisiveness:** structurally complete as a successor.
**Cost:** highest.

Define a solvent-independent scalar (H_\theta(R,u)) from upstream electronic field-response data and derive its conjugate source from

[
\nabla_uH_\theta.
]

This resolves integrability by construction. It is a successor model, not an interpretation of the present nonintegrable source.

---

# 5. Immediate experiment: nonuniform primitive-work and reciprocity replay

Execute exactly one experiment:

[
\boxed{
\text{Full continuum-active nonuniform WorkCoupling replay.}
}
]

It must use no experimental hydration or solvation target.

## 5.1 Test states

Use the frozen fixed-source geometries already employed for the QM/PCM gate, plus one-centre analytic monopole and dipole cases.

For each geometry and at least two materially different dielectric/cavity instances, include:

1. the converged operational state (\sigma^\star);
2. canonical manufactured boundary coefficient directions;
3. their signed amplitudes;
4. deterministic linear combinations that exercise nonuniform angular structure.

The operator identities below must be checked on the full discrete spaces, so numerical sampling is supplementary rather than the proof of factorization.

For each test state define

[
\varphi_k=\mathcal V_R\sigma_k,
\qquad
u_k=L_R\sigma_k.
]

Here (\mathcal V_R\sigma) is the actual real-space reaction potential represented by the ddPCM boundary state.

---

## 5.2 Required identities

### Source replay

The official arbitrary-potential path and direct native injection must produce the same MACE-POLAR source:

[
\boxed{
c_{\mathrm{POLAR}}^{\mathrm{up}}(R,\varphi_k)
=============================================

c_{\mathrm{POLAR}}^{\mathrm{direct}}(R,u_k).
}
]

This extends the already passed uniform source replay.

### Upstream explicit-work identity

Using the full upstream MACE-POLAR primitive Gaussian density,

[
\rho_g^{\mathrm{full}}
======================

\rho_g(c_g^0+\Delta c_g),
]

require

[
\boxed{
r_{E,k}
=======

## E_{\mathrm{up}}^{\mathrm{arb}}(R,\varphi_k)

## E_{\mathrm{raw}}(R,u_k)

# \int \rho_g^{\mathrm{full}}(\mathbf r)\varphi_k(\mathbf r),d\mathbf r

0.

}
]

The integral and sign must use the audited convention.

### Hybrid ddPCM half-work identity

For the actual hybrid source,

[
\rho_h
======

\rho_p(c_p)+\delta\rho_g(\Delta c_g),
]

require at the operational solution

[
\boxed{
r_{G}
=====

## 2G_{\mathrm{cont}}

# \int\rho_h(\mathbf r)\varphi_{\sigma^\star}(\mathbf r),d\mathbf r

0.

}
]

This verifies that the primitive source work is the same work whose half appears in the ddPCM ledger.

### Native factorization

Construct analytic candidate maps (D_p,D_g) from the primitive density–potential coupling and test

[
\boxed{
R_p=L_R^*D_{p,R}-B_{p,R}=0,
}
]

[
\boxed{
R_g=L_R^*D_{g,R}-B_{g,R}=0.
}
]

If these fail, the real-space (\sigma)-dependent work may still exist, but no native endpoint pairing with (u) is legitimate.

### Gauge identities

For a declared constant-potential direction (g_0),

[
\boxed{
\mathcal W_g(\Delta c_g,u+\kappa g_0)
-------------------------------------

# \mathcal W_g(\Delta c_g,u)

0,
}
]

and

[
\boxed{
\mathcal W_p(c_p,u+\kappa g_0)
------------------------------

# \mathcal W_p(c_p,u)

Q_p\kappa.
}
]

### Directional derivative identity

For deterministic nonuniform (\delta\sigma), with

[
\delta u=L_R\delta\sigma,
\qquad
\delta\varphi=\mathcal V_R\delta\sigma,
]

require

[
\boxed{
D_\sigma
\left[
E_{\mathrm{up}}^{\mathrm{arb}}
------------------------------

E_{\mathrm{raw}}
\right][\delta\sigma]
=====================

D_\sigma
\left[
\int\rho_g^{\mathrm{full}}\varphi_\sigma
\right][\delta\sigma].
}
]

The right side includes both the explicit potential variation and the induced change of the MACE-POLAR density.

---

## 5.3 Required artifacts

Freeze and content-address:

* checkpoint and source-code hashes;
* the official arbitrary-potential work path;
* the direct native-field path;
* definitions of all eight native channels;
* (A_R,B_{p,R},B_{g,R},L_R) and their metrics;
* primitive point and Gaussian density definitions;
* Gaussian widths and normalization;
* dipole sign and Cartesian convention;
* potential gauge and origin;
* reaction-potential evaluator (\mathcal V_R);
* candidate (D_p,D_g);
* all test (\sigma_k,\varphi_k,u_k);
* source coefficient outputs;
* raw, upstream, work, and ddPCM energies;
* operator residuals;
* AD and finite-difference derivative records;
* solver residual and conditioning certificates;
* floating-point precision and library versions.

---

## 5.4 Pass/fail thresholds

No fixed chemical tolerance is permitted.

For every numerical identity, construct an uncertainty interval from independently bounded errors.

For the upstream work residual,

[
|r_{E,k}|
\le
\delta E_{\mathrm{up},k}
+
\delta E_{\mathrm{raw},k}
+
\delta W_{g,k}
+
\delta_{\mathrm{round},k}.
]

For the continuum work residual,

[
|r_G|
\le
2\delta G_{\mathrm{cont}}
+
\delta W_p
+
\delta W_g
+
\delta_{\mathrm{round}}.
]

The bounds must come from:

* explicit solver residual bounds;
* precision refinement;
* analytic integration error or outward-rounded quadrature intervals;
* condition-number propagation where applicable;
* finite-difference truncation and roundoff bounds.

Pass only if zero lies in the resulting residual interval and the interval contracts under increased precision or numerical refinement.

For the operator identities, use outward-rounded high-precision matrix arithmetic. Pass only if every component of

[
L^*D_s-B_s
]

has an interval containing zero. If full interval evaluation is unavailable, use a rigorously propagated norm bound

[
|L^*D_s-B_s|
\le \Delta_{\mathrm{op},s},
]

where (\Delta_{\mathrm{op},s}) is computed from the arithmetic and integration errors, not selected as an acceptance parameter.

An unresolved residual is a failure, not a pass.

Every geometry, field direction, amplitude, and solvent instance must pass individually. No averaging is allowed.

---

## 5.5 Terminal implications

### Outcome A: all real-space and upstream-work identities pass

Then the uniform work audit has been extended to the actual nonuniform reaction-potential class.

If the graph audit also proves that the displayed primitive work is the only explicit external-potential term excluded from (E_{\mathrm{raw}}), then

[
\boxed{
\Phi_{\mathrm{raw}}
===================

E_{\mathrm{raw}}(R,u^\star)
+
G_{\mathrm{cont}}(R,\sigma^\star)
}
]

is admitted as the non-double-counting **operational** electronic-plus-continuum scalar.

The derivation is

[
E_{\mathrm{total}}
==================

E_{\mathrm{raw}}
+
W_h
+
C_{\mathrm{env}},
]

with

[
C_{\mathrm{env}}=-\frac12W_h,
\qquad
G_{\mathrm{cont}}=\frac12W_h,
]

so

[
E_{\mathrm{total}}
==================

E_{\mathrm{raw}}+G_{\mathrm{cont}}.
]

This still does not make the operational root stationary with respect to the electronic source, because the source remains nonintegrable.

### Outcome B: real-space identities pass but native factorization fails

A physically declared work exists as a function of ((c,\sigma)), but not as a function of ((c,u)) alone.

Then:

* (\Phi_{\mathrm{raw}}) may still be admitted through the real-space state-space derivation;
* no term (\langle c_p+\Delta c_g,u\rangle), (\langle\Delta c_g,u\rangle), or similar may appear;
* forces must retain the full (\sigma)-dependence and corresponding VJPs.

### Outcome C: upstream nonuniform explicit-work identity fails

Then the exact uniform (+\mathbf E\cdot\boldsymbol\mu) relation is a special-case branch identity and supplies no nonuniform work semantics.

Consequences:

* no explicit nonuniform work correction is admitted;
* (\Phi_0) and (\Phi_{\mathrm{raw}}) remain operational scalars only;
* neither may be labelled the uniquely correct electronic-plus-continuum ledger;
* no new development-panel comparison should be used to select between them;
* the smallest structurally clean successor is scalar-first.

### Outcome D: primitive work and (2G_{\mathrm{cont}}) disagree

Then the point/Gaussian primitive source used for the proposed work is not the same energy source represented by the ddPCM ledger, or the sign/pairing implementation is inconsistent.

This rejects a physically unified hybrid ledger. It is not repaired by adding a native endpoint contraction.

### Outcome E: Gaussian work passes but permanent work fails

Then the MACE-POLAR work semantics are established, but the hybrid MACE-MDP permanent replacement is not.

The full hybrid ledger remains nonidentifiable. The Gaussian result cannot be extrapolated to the point permanent source.

---

# 6. What nonidentifiability closes

The correct selection is:

[
\boxed{\text{(a) It closes the quantitative physical interpretation, not all operational PES use or the entire route.}}
]

More precisely, current nonidentifiability closes:

* the claim of a uniquely derived non-double-counting electronic-plus-continuum ledger;
* the claim that a native (c)-(u) endpoint contraction represents physical work;
* the claim that (\Delta E_{\mathrm{raw}}) is already proven to be electronic distortion energy;
* strict Tier V for the present exposed source.

Strict Tier V is independently excluded by the continuum-active curl defect. A common stationary scalar would require the source update to be conjugate to the electronic scalar in the relevant duality. It is not.

Nonidentifiability does **not** close an operational conservative PES. Let

[
r_1(R,\sigma,u)
===============

## A_R\sigma

## B_{p,R}c_p(R)

B_{g,R}\Delta c_g(R,u),
]

[
r_2(R,\sigma,u)
===============

u-L_R\sigma.
]

Set

[
z=(\sigma,u),
\qquad
r=(r_1,r_2).
]

The state Jacobian is

[
J_z
===

\begin{pmatrix}
A_R & -B_{g,R}D_u\Delta c_g\
-L_R & I
\end{pmatrix}.
]

For either chosen scalar (\Phi_0) or (\Phi_{\mathrm{raw}}), solve

[
\boxed{
J_z^T\lambda=\nabla_z\Phi.
}
]

Then

[
\boxed{
\frac{d\bar\Phi}{dR}
====================

## \partial_R\Phi

\lambda^T\partial_Rr.
}
]

This produces the derivative of the chosen scalar even though the root is not a stationary point of that scalar.

Operational conservative-PES use requires:

1. a unique selected root;
2. nonsingular (J_z);
3. no branch hysteresis or discontinuous root switching;
4. all direct and implicit coordinate derivatives;
5. directional-derivative agreement for the same scalar.

Failure of these conditions would close operational PES use. Source–ledger nonidentifiability alone does not.

It also does not logically close the whole MDP+POLAR route. It restricts that route to an explicitly operational status unless the nonuniform work experiment passes or a scalar-first successor replaces the nonintegrable source.

---

# 7. Multiple solvents, coordinate VJPs, and nonpolar separation

For solvent (s), retain

[
A_R^{(s)}\sigma^{(s)}
=====================

B_{p,R}^{(s)}c_p
+
B_{g,R}^{(s)}\Delta c_g,
]

[
u^{(s)}
=======

L_R^{(s)}\sigma^{(s)}.
]

The electronic graph and any primitive electronic WorkCoupling must remain solvent-independent. Solvent dependence may enter through:

* dielectric response;
* SMD Coulomb radii;
* (A_R^{(s)});
* (B_{p,R}^{(s)});
* (B_{g,R}^{(s)});
* (L_R^{(s)});
* the resulting reaction potential.

The nonuniform work and half-work identities must pass separately for every admitted solvent. A water-only identity does not establish a universal ledger.

The force implementation must include coordinate VJPs through

[
A_R^{(s)},\quad
B_{p,R}^{(s)},\quad
B_{g,R}^{(s)},\quad
L_R^{(s)},\quad
c_p(R),\quad
\Delta c_g(R,u),
]

and, if an explicit primitive WorkCoupling is used in a scalar, through its density, potential, origin, gauge, and geometry maps.

Keep the nonpolar term separately versioned:

[
G_{\mathrm{total}}^{(s,\nu)}
============================

\bar\Phi^{(s)}
+
G_{\mathrm{np}}^{(s,\nu)}.
]

It must not:

* enter the polarization root;
* alter the WorkCoupling;
* compensate a failed electrostatic identity;
* or influence selection of the electrostatic ledger.

## Final decision

* The uniform (+\mathbf E\cdot\boldsymbol\mu) identity does not uniquely determine nonuniform work.
* No endpoint pairing involving (c_p), (\Delta c_g), or their sum is currently admitted in (U_8).
* (\langle\nabla_uE_{\mathrm{raw}},u\rangle) is type-correct but is not generally an energy difference or physical source work.
* (\Phi_0) and (\Phi_{\mathrm{raw}}) are the only scientifically admissible operational total ledgers constructed from the current scalar objects without introducing a new convention.
* Execute the full nonuniform primitive-work and ddPCM-reciprocity replay immediately.
* A complete pass admits (\Phi_{\mathrm{raw}}) as the non-double-counting operational ledger, but not as Tier V.
* A failure leaves both ledgers operational only and requires a scalar-first successor for a structurally unified electronic-continuum model.
* The present nonidentifiability closes the unique physical interpretation, not conservative operational differentiation and not the entire MDP+POLAR research route.

HYBRID NONUNIFORM WORK IDENTIFIABILITY DECISION
