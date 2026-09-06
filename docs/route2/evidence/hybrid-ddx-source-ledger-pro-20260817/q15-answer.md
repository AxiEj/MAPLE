## Decisive disposition

1. **(\Phi_1^\Delta) is not yet admissible as written.** The term (\tfrac12\sigma^TA_R\sigma) is a legitimate continuum self-energy only if the particular (A_R) is a symmetric Hessian in a declared metric. Native ddPCM commonly exposes a primal operator plus a separate energy pairing; the quadratic form must therefore be proved, not inferred.

2. Even after that proof, (\Phi_1^\Delta) is correct only if (E_{\rm cond}(R,u)-E_{\rm cond}(R,0)) contains **exactly the full external work of the hybrid source**
   [
   c_h(u)=c_p+\Delta c(u).
   ]
   The published MACE-POLAR construction includes an explicit applied-potential term for its own final Gaussian density, not for the MACE-MDP permanent point source. Unless a replay proves equality, a nonzero work-replacement term is mandatory. ([arXiv][1])

3. **(\Phi_0) is a well-defined operational scalar but is not, by standard variational accounting, a complete polarizable-solute energy.** Self-consistency of the source does not automatically pay the electronic distortion cost.

4. The failed permanent-source gate closes the claim that unchanged MACE-MDP (q/p) is a quantitatively certified standalone PCM source. It does **not logically close the entire operational hybrid**, but the current route remains an uncertified research model until its energy semantics, matched QM/PCM decomposition, root differentiability, and forces pass.

5. The immediate next experiment is the **native-field (E_{\rm cond}) semantics and charging replay**, frozen before any new water-error calculation. No further (q/p)-only radial repair is identifiable.

---

# 1. Primitive continuum accounting and every non-double-counting ledger

Define

[
b_h(R,u)
========

B_{p,R}c_p(R)+B_{g,R}\Delta c(R,u),
]

with operational root

[
A_R\sigma=b_h(R,u),
\qquad
u=L_R\sigma.
]

Let (W_h(R,u,\sigma)) denote the **full signed interaction work** between the hybrid solute source and the reaction potential:

[
W_h
===

\int \rho_h(\mathbf r),W_\sigma(\mathbf r),d\mathbf r.
]

For a stable dielectric response this is normally negative. The on-shell PCM electrostatic energy is

[
G_{\rm ddPCM}=\frac12 W_h.
]

This half-work relation, rather than (\tfrac12\sigma^TA\sigma), is the representation-independent statement. ddX defines the PCM energy as one half of the source–reaction-potential interaction and presents the discrete ddPCM operator and energy pairing separately. ([DD Solvation][2])

## 1.1 When is (\tfrac12\sigma^TA\sigma) valid?

Suppose there exists an SPD metric (M_R) such that

[
M_RA_R=A_R^TM_R
]

and (M_RA_R) is positive definite. Suppose also that the source coupling is represented by

[
W_h=-\sigma^TM_Rb_h.
]

Then the primitive continuum functional is

[
\mathcal J_R(\sigma;c_h)
========================

\frac12\sigma^TM_RA_R\sigma
-\sigma^TM_Rb_h.
]

Its stationarity condition is

[
\nabla_\sigma\mathcal J_R
=========================

M_R(A_R\sigma-b_h)=0.
]

At the solution,

[
W_h=-\sigma^TM_RA_R\sigma,
]

and

[
G_{\rm ddPCM}
=============

-\frac12\sigma^TM_RA_R\sigma.
]

Therefore the positive environment self-energy is

[
C_{\rm env}
===========

# \frac12\sigma^TM_RA_R\sigma

-G_{\rm ddPCM}.
]

The proposed expression (\tfrac12\sigma^TA_R\sigma) is the special case (M_R=I).

If no such metric has been established, the gradient of

[
\frac12\sigma^TA_R\sigma
]

is

[
\frac12(A_R+A_R^T)\sigma,
]

not (A_R\sigma). In that case the correct force machinery is a primal-adjoint Lagrangian using the native ddPCM energy pairing, not an invented Euclidean quadratic form.

A generic discrete Lagrangian is

[
\mathscr L_{\rm PCM}(x,y;c)
===========================

G_{\rm native}(x,c)
+
y^T!\left(A_Rx-B_Rc\right),
]

with the adjoint equation obtained from

[
\nabla_x\mathscr L_{\rm PCM}=0.
]

All ledger formulas below therefore use (G_{\rm ddPCM}) and (W_h=2G_{\rm ddPCM}). They do not require (A_R) to be a Euclidean Hessian.

---

## 1.2 Electronic notation

Let

[
c_{\rm P}^0(R)=M_{\rm POLAR}(R,0),
]

[
c_{\rm P}(R,u)
==============

c_{\rm P}^0(R)+\Delta c(R,u).
]

Define

[
\Delta E_{\rm cond}(R,u)
========================

E_{\rm cond}(R,u)-E_{\rm cond}(R,0).
]

Let (W_{\rm P}(R,u)) be the exact external-potential work already included in (E_{\rm cond}). It must be extracted from the actual checkpoint graph, not guessed from the API name.

The published architecture does two relevant things:

* the applied potential enters the field features used by the multipole updates and learned nonlocal energy;
* the total electrostatic energy contains an explicit
  [
  E_{\rm app}=\int\rho(\mathbf r)v_{\rm app}(\mathbf r),d\mathbf r
  ]
  for the final MACE-POLAR density. ([arXiv][1])

Consequently, subtracting only a visibly named (E_{\rm app}) term does not prove that every remaining field-dependent graph contribution is an internal electronic energy.

---

## 1.3 Ledger A: continuum-only operational scalar

The current ledger is

[
\boxed{
\Phi_0(R)=E_{\rm vac}(R)+G_{\rm ddPCM}(R,\sigma^\star)
}
]

or, when reporting only the electrostatic solvation correction,

[
\boxed{\phi_0=G_{\rm ddPCM}.}
]

This is internally unambiguous. It makes no claim that electronic deformation has been paid.

---

## 1.4 Ledger B: (E_{\rm cond}) is internal electronic energy only

Suppose

[
E_{\rm cond}(R,u)=I_{\rm el}(R,u)
]

contains no source–external-field interaction work.

Then the electronic distortion cost is

[
\Delta I_{\rm el}
=================

\Delta E_{\rm cond}.
]

The non-double-counting total scalar is

[
\boxed{
\Phi_{\rm int}
==============

E_{\rm vac}
+
\Delta E_{\rm cond}
+
G_{\rm ddPCM}.
}
]

If the metric-Hessian representation above exists,

[
G_{\rm ddPCM}
=============

-\frac12\sigma^TM_RA_R\sigma,
]

so

[
\Phi_{\rm int}
==============

E_{\rm vac}
+\Delta E_{\rm cond}
-\frac12\sigma^TM_RA_R\sigma.
]

Thus the candidate’s **positive** quadratic term would have the wrong sign in the internal-only case.

---

## 1.5 Ledger C: (E_{\rm cond}) is a full external enthalpy

Suppose

[
E_{\rm cond}(R,u)
=================

I_{\rm el}(R,u)+W_{\rm P}(R,u),
]

with (W_{\rm P}(R,0)=0). Then

[
\Delta I_{\rm el}
=================

\Delta E_{\rm cond}-W_{\rm P}.
]

The correct scalar is

[
\boxed{
\Phi_{\rm enth}
===============

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

W_{\rm P}
+
G_{\rm ddPCM}.
}
]

Using (W_h=2G_{\rm ddPCM}), this can be written as

[
\boxed{
\Phi_{\rm enth}
===============

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

G_{\rm ddPCM}
+
\left(W_h-W_{\rm P}\right).
}
]

Therefore the proposed ledger

[
E_{\rm vac}+\Delta E_{\rm cond}
+\frac12\sigma^TA_R\sigma
]

is valid only if both conditions hold:

[
\frac12\sigma^TA_R\sigma=-G_{\rm ddPCM},
]

and

[
W_{\rm P}=W_h.
]

### The permanent-source mismatch

Under the likely graph interpretation,

[
W_{\rm P}
=========

W_g(c_{\rm P}^0+\Delta c,u),
]

while the hybrid source work is

[
W_h
===

W_p(c_p,u)+W_g(\Delta c,u).
]

Hence

[
\boxed{
W_h-W_{\rm P}
=============

W_p(c_p,u)-W_g(c_{\rm P}^0,u).
}
]

There is no reason for this term to vanish merely because the induced increments are identical.

This is the central double-counting issue: (E_{\rm cond}) can include the reaction-field work of the **zero-field MACE-POLAR Gaussian source**, whereas the continuum equation uses the **MACE-MDP point source** as its permanent component.

Even after adding this exact correction, the result remains operational. The electronic distortion term is the cost of deforming MACE-POLAR’s internal baseline state, while the permanent continuum source is supplied by MACE-MDP. They are not a single electronic state functional.

---

## 1.6 Ledger D: graph with an identifiable explicit work term

Suppose code provenance establishes

[
E_{\rm cond}
============

E_{\rm rem}
+
W_{\rm explicit},
]

where (W_{\rm explicit}) may include a charge-potential or dipole-field contraction.

If—and only if—(E_{\rm rem}) is shown to be internal-only, then

[
\boxed{
\Phi_{\rm split}
================

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

W_{\rm explicit}
+
G_{\rm ddPCM}.
}
]

For a uniform electric field (\mathbf F), with

[
v_{\rm app}(\mathbf r)
======================

-\mathbf F\cdot(\mathbf r-\mathbf r_0)+V_0,
]

the work must reduce to

[
W_{\rm explicit}
================

Q_{\rm tot}V_0-\boldsymbol\mu\cdot\mathbf F.
]

An explicit dipole term alone does not account for arbitrary native (1.5/3.0)-Å field channels unless its generalization to those channels is frozen and proven.

If (E_{\rm rem}) still depends directly on field features in a way not generated by an internal-state stationary principle, there is no unique physical “distortion” interpretation. It can be retained only as a declared operational scalar.

---

## 1.7 Path-defined source-work ledger

Because the exposed MACE-POLAR update has nonzero continuum-active curl, one may mathematically define, for a chosen path (\gamma),

[
\Delta H_\gamma(u)
==================

\int_\gamma c_{\rm P}(v)^T\mathsf Q_R,dv.
]

Then, for matched endpoint work,

[
\Phi_\gamma
===========

E_{\rm vac}+\Delta H_\gamma-G_{\rm ddPCM}.
]

But

[
\Delta H_{\gamma_1}\ne\Delta H_{\gamma_2}
]

for some paths whenever

[
\oint c_{\rm P}(u)^T\mathsf Q_R,du\ne0.
]

A straight-line path is therefore a convention, not a recovered electronic functional. Given the measured curl defect, this option is **not admitted as a physical repair**.

---

# 2. Is (\Phi_0) missing electronic distortion cost?

## Yes under the standard primitive interpretation

Consider a linear induced coordinate (d) with internal cost

[
I(d)=\frac12d^THd,
\qquad H\succ0,
]

and a reaction operator (R\succeq0). For total source (c_p+d),

[
G(c_p+d)
========

-\frac12(c_p+d)^TR(c_p+d).
]

The joint scalar is

[
F(d)
====

\frac12d^THd
-\frac12(c_p+d)^TR(c_p+d).
]

Stationarity gives

[
Hd=R(c_p+d)=u.
]

At that stationary point,

[
F(d^\star)
==========

G(c_p+d^\star)
+
\frac12(d^\star)^Tu.
]

Thus the continuum half-work evaluated on the polarized source omits

[
\boxed{
\Delta I_{\rm dist}
===================

\frac12(d^\star)^Tu
}
]

unless (d^\star=0).

Self-consistency changes the source entering (G); it does not make the positive distortion term disappear.

## What cannot be inferred here

The current (\Phi_0) might empirically approximate the **sum** of distortion and continuum stabilization because the learned response map and source errors happen to compensate. But that would be an operational approximation, not a consequence of PCM theory.

For the cost to be rigorously “implicitly encoded,” one would need either:

1. a common reduced scalar obtained by eliminating electronic variables from a stationary electronic-continuum functional; or
2. an explicitly fixed charging-path functional whose path dependence is accepted as part of the model definition.

The measured source curl defect rules out the first interpretation for the exposed MACE-POLAR source on the continuum-active field subspace. Therefore:

[
\boxed{
\Phi_0\text{ is not physically complete by derivation, although it remains a valid declared operational scalar.}
}
]

---

# 3. Finite (E_{\rm cond}) semantics and charging canaries

Let (\mathsf Q_R) be the frozen source–native-field work pairing:

[
w(c,u)=c^T\mathsf Q_Ru.
]

Its sign, units, channel order, Gaussian normalization, and potential gauge must be fixed by analytic one-source tests before using the equations below.

All finite-difference tests should be run in deterministic arithmetic. Use a three-level central-difference ladder (h,h/2,h/4), require the expected second-order convergence before roundoff, and compare AD against the Richardson-extrapolated value. Integral identities should be evaluated with nested quadrature, for example 16- and 32-point Gauss–Legendre; pass only when the identity residual is bounded by the measured discretization, root-solve, and quadrature error. No chemical-error tolerance enters these canaries.

## Canary 1: zero-field identity

Require

[
E_{\rm cond}^{\rm adapter}(R,0)
===============================

E_{\rm POLAR}^{\rm official}(R,0),
]

[
M_{\rm POLAR}^{\rm adapter}(R,0)
================================

M_{\rm POLAR}^{\rm official}(R,0),
]

[
\Delta c(R,0)=0,
]

and

[
\sum_i\Delta q_i(R,u)=0
]

for every tested (u).

Also require the zero-field geometry derivative to replay:

[
\nabla_RE_{\rm cond}^{\rm adapter}(R,0)
=======================================

\nabla_RE_{\rm POLAR}^{\rm official}(R,0).
]

Failure means that the candidate energy difference does not even share the intended vacuum anchor.

---

## Canary 2: official uniform-field replay

For each Cartesian field (\mathbf F), construct

[
v_{\rm app}(\mathbf r)
======================

-\mathbf F\cdot(\mathbf r-\mathbf r_0).
]

Generate the native field tensor (u_{\mathbf F}) using the exact frozen projection convention.

Require simultaneous replay of

[
E_{\rm adapter}(R,u_{\mathbf F})
================================

E_{\rm official}(R,\mathbf F),
]

[
c_{\rm adapter}(R,u_{\mathbf F})
================================

c_{\rm official}(R,\mathbf F),
]

and the corresponding forces.

The published MACE-POLAR convention computes field-response dipoles from

[
\boldsymbol\mu_E
================

-\frac{\partial E}{\partial\mathbf F}.
]

([arXiv][1])

Therefore require AD and FD agreement:

[
-\frac{\partial E_{\rm cond}}{\partial F_\alpha}
================================================

-\lim_{h\to0}
\frac{
E_{\rm cond}(\mathbf F+h\mathbf e_\alpha)
-----------------------------------------

E_{\rm cond}(\mathbf F-h\mathbf e_\alpha)
}{2h}.
]

Separately calculate the source dipole

[
\boldsymbol\mu_{\rm src}(c)
===========================

\sum_i q_i(\mathbf R_i-\mathbf r_0)
+
\sum_i\mathbf p_i,
]

with the exact basis normalization.

A full-enthalpy interpretation tied to the exposed source requires

[
\boxed{
\boldsymbol\mu_E=\boldsymbol\mu_{\rm src}.
}
]

Uniform-field agreement alone is not enough to establish arbitrary native-field conjugacy, but disagreement already rejects that claim.

---

## Canary 3: fixed-source dipole/work sign and gauge

Detach the source coefficients from the graph and test the work contraction itself.

For a fixed (c),

[
w(c,u_{\mathbf F})
==================

## Q_{\rm tot}V_0

\boldsymbol\mu_{\rm src}(c)\cdot\mathbf F.
]

Require

[
\left.\frac{\partial w}{\partial \mathbf F}\right|_c
====================================================

-\boldsymbol\mu_{\rm src}(c),
]

and

[
w(c,u+\delta V_0g_0)-w(c,u)
===========================

Q_{\rm tot}\delta V_0.
]

This freezes:

* the potential sign;
* dipole axis ordering;
* origin dependence;
* constant-potential gauge;
* charge normalization;
* energy units.

Failure stops all ledger derivation.

---

## Canary 4: arbitrary continuum-active AD versus FD

Use field directions in

[
\mathcal U_R=\operatorname{range}L_R,
]

not arbitrary unrelated tensors. For deterministic (\delta u=L_R\delta\sigma), require

[
\nabla_uE_{\rm cond}(R,u)^T\delta u
===================================

\lim_{h\to0}
\frac{
E_{\rm cond}(R,u+h\delta u)
---------------------------

E_{\rm cond}(R,u-h\delta u)
}{2h}.
]

This validates the scalar graph and adapter.

It does not establish that the exposed source is its gradient. That stronger full-enthalpy claim requires

[
\boxed{
\nabla_uE_{\rm cond}(R,u)
=========================

\mathsf Q_R^Tc_{\rm P}(R,u)
}
]

on the complete continuum-active subspace.

Because the measured reduced curl defect is approximately (0.208), this identity cannot hold globally for the existing exposed source unless the previously measured source and the energy-conjugate source are different objects.

### Internal-only necessary identity

Under an ideal internal functional (I(c(u))) coupled bilinearly through (c^T\mathsf Q_Ru), stationarity gives

[
\boxed{
\nabla_uE_{\rm int}(u)
======================

*

[D_uc(u)]^T\mathsf Q_Ru.
}
]

This is necessary, not sufficient. It should be applied to the residual graph energy after any exact explicit-work term is subtracted.

---

## Canary 5: straight-line (\lambda)-charging identity

For a fixed native field (u_\star), define

[
u_\lambda=\lambda u_\star,\qquad 0\le\lambda\le1.
]

Let

[
s_E(\lambda)
============

\nabla_uE_{\rm cond}(u_\lambda)^Tu_\star,
]

and

[
s_M(\lambda)
============

c_{\rm P}(u_\lambda)^T\mathsf Q_Ru_\star.
]

The fundamental theorem for the scalar graph requires

[
\boxed{
E_{\rm cond}(u_\star)-E_{\rm cond}(0)
=====================================

\int_0^1s_E(\lambda),d\lambda.
}
]

This must pass regardless of physical semantics.

### Full external enthalpy tied to the exposed source

Require pointwise

[
s_E(\lambda)=s_M(\lambda)
]

and hence

[
\boxed{
\Delta E_{\rm cond}
===================

\int_0^1s_M(\lambda),d\lambda.
}
]

### Internal-only energy

For a stationary internal energy under bilinear external work,

[
\frac{dE_{\rm int}(u_\lambda)}{d\lambda}
========================================

-\lambda\frac{ds_M}{d\lambda},
]

and therefore

[
\boxed{
\Delta E_{\rm int}
==================

## \int_0^1s_M(\lambda),d\lambda

s_M(1).
}
]

### Explicit-work split

If (W_{\rm explicit}(u)) is known, define

[
E_{\rm rem}(u)=E_{\rm cond}(u)-W_{\rm explicit}(u)
]

and apply the internal-only identity to (E_{\rm rem}).

If neither the full-enthalpy nor internal-only identity passes, (E_{\rm cond}) is a hybrid operational graph with no physically unique decomposition.

---

## Canary 6: closed-loop integrability

For two continuum-active directions (u_1,u_2), evaluate the source work around the rectangle

[
0\rightarrow u_1\rightarrow u_1+u_2
\rightarrow u_2\rightarrow0.
]

A source conjugate to a scalar must satisfy

[
\boxed{
\oint c_{\rm P}(u)^T\mathsf Q_R,du=0.
}
]

The already measured nonzero curl defect rejects this identity for the exposed source. It does not reject (E_{\rm cond}) as a scalar; it proves that

[
\mathsf Q_R^Tc_{\rm P}(u)
\ne
\nabla_uE_{\rm cond}(u)
]

somewhere in the active subspace.

---

# 4. Matched QM/PCM decomposition of distortion and stabilization

Fix all of the following before calculation:

* nuclear geometry (R);
* electron number and spin;
* electronic method, basis, integration grid, and numerical thresholds;
* SMD intrinsic radii and exact cavity construction;
* solvent dielectric;
* PCM equation;
* nuclear and electronic source conventions;
* PCM discretization convergence;
* no CDS, cavitation, dispersion, geometry relaxation, ZPE, thermal correction, or standard-state term.

Let (P) be the QM density matrix and let

[
E_0[P;R]
]

be the vacuum electronic functional, including the same nuclear repulsion and all method-specific terms.

Let

[
G_s[P;R]
========

\frac12
\int \rho_{\rm tot}[P](\mathbf r)
W_s[P](\mathbf r),d\mathbf r
]

be the same-equation PCM electrostatic energy.

Define the vacuum density

[
P_0
===

\arg\min_P E_0[P;R],
]

and the PCM-relaxed density

[
P_s
===

\arg\min_P
\left{
E_0[P;R]+G_s[P;R]
\right}.
]

Then the exact fixed-geometry decomposition is:

### Electronic distortion cost

[
\boxed{
D_{\rm QM}
==========

E_0[P_s;R]-E_0[P_0;R].
}
]

### Continuum stabilization of the relaxed density

[
\boxed{
S_{\rm QM}
==========

G_s[P_s;R].
}
]

### Total electrostatic solution shift

[
\boxed{
T_{\rm QM}
==========

# D_{\rm QM}+S_{\rm QM}

E_0[P_s;R]+G_s[P_s;R]-E_0[P_0;R].
}
]

Also compute the frozen-density PCM term

[
S_{\rm frozen}=G_s[P_0;R],
]

and the electronic-relaxation contribution

[
\Delta_{\rm relax}
==================

T_{\rm QM}-S_{\rm frozen}.
]

## Evaluating the vacuum functional on the PCM-relaxed density

After converging the PCM SCF:

1. save the converged density matrix (P_s);
2. remove the PCM reaction operator and all PCM energy terms;
3. **do not reoptimize the orbitals or density**;
4. evaluate the vacuum functional once on (P_s):
   [
   E_0[P_s;R].
   ]

For DFT this requires recomputing the one-electron, Coulomb, exact-exchange where applicable, XC, dispersion if it is part of the vacuum model, and nuclear terms using the same basis and grids. The identity

[
E_{\rm QM+PCM}[P_s]
===================

E_0[P_s]+G_s[P_s]
]

must be replayed independently.

## ML comparison

For the operational hybrid calculate

[
S_{\rm ML}=G_{\rm ddPCM}[c_p+\Delta c(u^\star)].
]

For a candidate electronic distortion ledger calculate

[
D_{\rm ML}
]

according to the semantics established in Section 3.

Then compare separately:

[
e_D=D_{\rm ML}-D_{\rm QM},
]

[
e_S=S_{\rm ML}-S_{\rm QM},
]

[
e_T=(D_{\rm ML}+S_{\rm ML})-T_{\rm QM}.
]

### Decision logic

* (\Phi_0) represents only (S_{\rm ML}). It may not be relabelled as (T_{\rm QM}) merely because its sum error happens to be small.
* A corrected (\Phi_1^\Delta) is admissible only if its (D_{\rm ML}) separately represents (D_{\rm QM}), and its total represents (T_{\rm QM}).
* Improvement arising from cancellation between a poor (D_{\rm ML}) and poor (S_{\rm ML}) is not evidence for the proposed physical decomposition.
* The comparison must be repeated for more than one dielectric. Otherwise a water-specific cancellation cannot be distinguished from a solvent-independent electronic term.

This matched decomposition—not experimental hydration free energy—is the correct experiment for deciding whether the missing scalar is electronic distortion.

---

# 5. Identifiability of a (q/p)-only permanent-source repair

## The claim is true in the identifiability sense

Monopoles and dipoles do not uniquely determine a molecular near-field density.

For example, around one center define a smooth approximation to

[
\delta\rho(\mathbf r)
=====================

q\big[
\delta(\mathbf r-a\mathbf e_x)
+\delta(\mathbf r+a\mathbf e_x)
-\delta(\mathbf r-a\mathbf e_y)
-\delta(\mathbf r+a\mathbf e_y)
\big].
]

It satisfies

[
\int\delta\rho(\mathbf r),d\mathbf r=0,
]

and

[
\int\mathbf r,\delta\rho(\mathbf r),d\mathbf r=0,
]

but it has a nonzero quadrupole tensor and a nonzero potential on a generic molecular cavity.

Therefore (\rho) and (\rho+\delta\rho) have identical (q/p) data but different:

* cavity MEPs;
* apparent surface charges;
* PCM energies;
* reaction fields.

Consequently no map

[
(q_i,\mathbf p_i)_{i=1}^N
\longmapsto
B_pc_p
]

is physically identified by (q/p) alone.

A deterministic radial embedding can be **chosen**, but it cannot be inferred as uniquely correct. The failed universal Gaussian embedding closes that particular prior. It does not mathematically prove that every possible radial convention is numerically poor; it proves that choosing another one would require new independent information.

## Legitimate extra information

A successor source can become identifiable from any of the following upstream observables:

1. **Higher multipoles**, at least (l\ge2), with frozen normalization and centers.
2. **Atom-centred density-expansion coefficients** in a specified radial basis.
3. **QM cavity or surface MEP values** over prospectively fixed geometries and cavities.
4. **QM external-field energies and derivatives** spanning the native continuum-active field subspace.
5. **A scalar-first field enthalpy head**, whose field gradient defines the source.
6. A direct **boundary-potential head** validated on electronic MEP data.

These are not ad hoc solvation corrections because they target upstream electronic observables, are independent of experimental hydration energies, and can be fixed before any solvent-performance evaluation.

---

# 6. What the failed permanent-source gate closes

The 3/4 result under the preregistered all-case (1) kcal/mol rule closes:

[
\boxed{
\text{“Unchanged MACE-MDP point }q/p\text{ is a quantitatively certified standalone PCM source.”}
}
]

It also forbids claiming that the later hybrid diagnostic retroactively passed that gate. The induced response may compensate a permanent-source error, and a four-case mean does not establish that the permanent representation became correct.

It does **not logically prove** that the operational composition

[
c_p+\Delta c
]

has no utility. The route may remain an operational research model if all of the following are obtained:

1. **Energy semantics:** one and only one ledger is selected by the canaries above.
2. **Matched component evidence:** (D_{\rm ML}), (S_{\rm ML}), and their sum are compared with the same-equation QM/PCM decomposition.
3. **Hybrid-source evidence:** total cavity MEP and PCM energy of (c_p+\Delta c) are evaluated prospectively, rather than inferring success from separate source components.
4. **Root regularity:** the operational root is single-valued over the admitted geometry domain and its block Jacobian is nonsingular.
5. **Force evidence:** analytic implicit-adjoint derivatives agree with directional derivatives of the same scalar.
6. **Solvent transfer:** the same electronic model and ledger work across multiple solvents, with solvent identity entering only through the continuum/cavity definition.
7. **No target-driven selection:** the water development MAE is recomputed only after the upstream ledger and source convention are frozen.

Until then, the correct status is:

[
\boxed{
\text{operational prototype, not quantitatively certified electrostatic source or unified solvation PES.}
}
]

---

# 7. Ranked next experiments and the single immediate experiment

## Rank 1 — Native-field (E_{\rm cond}) semantics and charging replay

**Highest information gain.** It determines whether any distortion ledger is algebraically defined, whether the proposed positive continuum self-energy has the right role, and whether an energy-conjugate scalar-first source already exists in the checkpoint.

## Rank 2 — Matched QM/PCM distortion–stabilization decomposition

This directly tests whether the missing electronic distortion term is materially present and whether the corrected ledger represents the proper component rather than merely shifting the total.

## Rank 3 — Scalar-first source observability test

Construct the energy-conjugate source, where possible,

[
\mathsf Q_R^Tc_E(R,u)
=====================

\nabla_uE_{\rm cond}(R,u),
]

and compare its surface MEP and fixed-source PCM energy with QM electronic references.

If (\mathsf Q_R^T) is not square, require

[
\nabla_uE_{\rm cond}
\in
\operatorname{range}(\mathsf Q_R^T).
]

Then the minimum-norm coordinate solution is not enough by itself; the source space and its physical density representation must be uniquely declared. A nonzero orthogonal residual

[
\left|
(I-\Pi_{\operatorname{range}\mathsf Q_R^T})
\nabla_uE_{\rm cond}
\right|
\ne0
]

means the present (q/p) source space cannot represent the scalar’s conjugate response.

---

## Immediate experiment: execute Rank 1 only

### Inputs

Use:

* the existing four frozen QM/PCM gate geometries;
* analytic isolated monopole and dipole systems;
* each geometry’s current operational reaction field (u^\star=L_R\sigma^\star);
* two deterministic additional continuum-active directions (L_R\delta\sigma_k) generated from a frozen hash-derived seed;
* uniform (x,y,z) fields;
* no experimental solvation values.

### Required calculations

For each geometry and field direction:

1. zero-field official replay;
2. uniform-field official replay;
3. exact graph-component output:
   [
   E_{\rm local},\ E_{\rm nonlocal},\ E_{\rm Hartree},\ E_{\rm app},
   ]
   where exposed;
4. source (c_{\rm P}(u));
5. (\nabla_uE_{\rm cond}) by AD;
6. central finite differences in the same native field coordinates;
7. (\lambda)-path evaluations at the 16- and 32-point Gauss–Legendre nodes;
8. exact endpoint work (W_{\rm P});
9. hybrid work
   [
   W_h=2G_{\rm ddPCM};
   ]
10. the work mismatch
    [
    \Delta W_{\rm perm}=W_h-W_{\rm P}.
    ]

### Frozen artifacts

The replay package must bind:

* checkpoint and code hashes;
* field channel order;
* source coefficient order;
* Gaussian widths and normalizations;
* charge/dipole Cartesian convention;
* potential sign and constant-potential gauge;
* units;
* explicit-work implementation;
* dtype and runtime;
* deterministic field directions;
* AD, FD, and quadrature outputs;
* ddPCM root residuals and native energy pairing.

### Stop criteria and conclusions

#### Outcome A: full enthalpy, hybrid work exactly matched

If

[
\nabla_uE_{\rm cond}
====================

\mathsf Q_R^Tc_h(u)
]

on the active space and

[
W_{\rm P}=W_h,
]

then freeze

[
\boxed{
\Phi
====

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

G_{\rm ddPCM}.
}
]

Use (\tfrac12\sigma^TA\sigma) only after proving it equals (-G_{\rm ddPCM}).

This outcome is unlikely for the current exposed source because of the frozen curl defect, but the canary must decide it rather than assumption.

#### Outcome B: full enthalpy of the POLAR source, not the hybrid source

If

[
\nabla_uE_{\rm cond}
====================

\mathsf Q_R^Tc_{\rm P}(u)
]

but

[
W_{\rm P}\ne W_h,
]

freeze

[
\boxed{
\Phi
====

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

W_{\rm P}
+
G_{\rm ddPCM}.
}
]

Equivalently,

[
\Phi
====

E_{\rm vac}
+
\Delta E_{\rm cond}
-------------------

G_{\rm ddPCM}
+
(W_h-W_{\rm P}).
]

This proves that the original (\Phi_1^\Delta) omitted the permanent-source work replacement.

#### Outcome C: internal-only energy

If the full-enthalpy identity fails but the internal-energy derivative and charging identities pass after removing exact explicit work, freeze

[
\boxed{
\Phi
====

E_{\rm vac}
+
\Delta E_{\rm int}
+
G_{\rm ddPCM}.
}
]

#### Outcome D: hybrid graph with no valid decomposition

If:

* AD does not replay FD;
* official uniform fields do not replay;
* neither charging identity passes;
* the field gradient lies outside the dual source range; or
* the explicit-work subtraction leaves unclassified direct field dependence,

then:

[
\boxed{
E_{\rm cond}\text{ must not be used to modify }\Phi_0.
}
]

No water MAE is computed for alternative ledgers.

### Governance after a passing outcome

Even after A, B, or C, first perform the matched QM/PCM decomposition in Section 4. Only after that component gate is frozen and passed may the newly fixed ledger be evaluated once on the 505-development and water panels.

The current evidence does not identify a guaranteed (0.2) kcal/mol no-training improvement. The distortion ledger is a scientifically motivated hypothesis, not an assured correction.

---

## Smallest scalar-first successor if Outcome D occurs

The smallest legitimate successor is a solvent-independent scalar

[
H_\theta(R,u)
]

trained on upstream QM energies under arbitrary external potentials spanning the native continuum-active field space—not on experimental solvation targets.

Define its conjugate source by

[
\boxed{
\mathsf Q_R^Tc_\theta(R,u)
==========================

\nabla_uH_\theta(R,u).
}
]

Then

[
c_{\theta,p}(R)=c_\theta(R,0),
]

and

[
\Delta c_\theta(R,u)
====================

c_\theta(R,u)-c_\theta(R,0).
]

This construction gives exact integrability by definition.

Charge conservation is imposed through potential-gauge covariance. For the constant-potential field direction (g_0),

[
H_\theta(R,u+\alpha g_0)
========================

H_\theta(R,u)+Q_{\rm tot}\alpha,
]

which implies the correct total charge in the conjugate source.

To solve the permanent near-field problem rather than merely the induction problem, the scalar-first source must replace the **whole continuum-active permanent-plus-induced source**, or be accompanied by independently identified permanent density coefficients. Retaining uncertified MDP (q/p) while replacing only (\Delta c) does not resolve (B_p)’s nonidentifiability.

---

# 8. Multiple solvents and analytic conservative forces

For solvent (s), write the operational residual as

[
r_1^{(s)}
=========

## A_R^{(s)}\sigma

## B_{p,R}^{(s)}c_p(R)

B_{g,R}^{(s)}\Delta c(R,u),
]

[
r_2^{(s)}
=========

u-L_R^{(s)}\sigma.
]

Let

[
z=(\sigma,u),
\qquad
M_u=D_u\Delta c.
]

The block state Jacobian is

[
J_z^{(s)}
=========

\begin{pmatrix}
A_R^{(s)} & -B_{g,R}^{(s)}M_u\
-L_R^{(s)} & I
\end{pmatrix}.
]

For any chosen scalar ledger (\Phi^{(s)}(R,z)), solve the adjoint system

[
\boxed{
\left(J_z^{(s)}\right)^T\lambda
===============================

\nabla_z\Phi^{(s)}.
}
]

Then the total coordinate derivative is

[
\boxed{
\frac{d\bar\Phi^{(s)}}{dR}
==========================

## \partial_R\Phi^{(s)}

\lambda^T\partial_Rr^{(s)}.
}
]

The term (\partial_Rr^{(s)}) contains the required coordinate VJPs through:

[
A_R^{(s)},\quad
B_{p,R}^{(s)},\quad
B_{g,R}^{(s)},\quad
L_R^{(s)},\quad
c_p(R),\quad
\Delta c(R,u).
]

A nonvariational operational root does not prevent conservative forces of the **chosen scalar composition**, provided:

1. the root is unique on the admitted branch;
2. (J_z^{(s)}) is nonsingular;
3. the branch varies continuously with (R);
4. all direct and implicit derivatives are included;
5. forces are verified against directional derivatives of that same scalar.

If root selection exhibits hysteresis, branch switching, or singular (J_z), the PES claim fails regardless of the scalar formula.

## Solvent modularity

The electronic source/enthalpy model should receive no solvent label. Solvent dependence enters through

[
A_R^{(s)},\ B_R^{(s)},\ L_R^{(s)},
]

the dielectric, and the solvent-specific radii. This permits the same electronic semantics and adjoint structure across solvents.

## CDS/nonpolar separation

Keep

[
G_{\rm nonpolar}^{(s)}(R)
]

as a separately versioned scalar:

[
E_{\rm total}^{(s)}
===================

\bar\Phi^{(s)}(R)
+
G_{\rm nonpolar}^{(s)}(R).
]

Its value, parameters, geometry derivatives, and provenance must remain independent of the continuum-source ledger decision. It must not enter the operational polarization root or compensate for a rejected electrostatic component.

---

# Terminal decision

* **Do not deploy (\Phi_1^\Delta) as written.**
* **Do not modify (B_p) again from (q/p) alone.**
* **Do not close the entire hybrid research route solely because the permanent standalone gate failed.**
* **Do close the quantitative-certification claim for unchanged MDP point (q/p).**
* **Execute the (E_{\rm cond}) native-field semantics and charging replay now.**
* If it uniquely establishes an internal, full-enthalpy, or exact split-work ledger, test that one ledger against the matched QM/PCM distortion–stabilization decomposition before any water-panel reevaluation.
* If it does not, retain (\Phi_0) only as an operational baseline and move to the scalar-first conjugate-source successor; there is no identifiable no-training (q/p)-only correction capable of being scientifically justified.

HYBRID DDX SOURCE-LEDGER TERMINAL DECISION

[1]: https://arxiv.org/html/2602.19411v1 "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry"
[2]: https://ddsolvation.github.io/ddX/md_docs_theory.html "ddx: Theory"
