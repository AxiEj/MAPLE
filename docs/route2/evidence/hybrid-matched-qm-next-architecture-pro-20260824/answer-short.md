## 1. Zero-training loophole for the unchanged source

**No—assuming the stated acetone comparison has already excluded representation and implementation mistakes.** The unchanged

[
\text{MDP point }q/p
+\text{POLAR nonlinear residual}
+\text{ADT uniform response}
]

is no longer defensible as a generally quantitative electrostatic source.

For a fixed cavity and PCM discretization, the apparent surface charge is a linear function of the solute boundary potential,

[
\sigma_C=K_C,v_C,
]

and the corresponding polarization energy is a quadratic pairing of (v_C) and (\sigma_C). Therefore, once the cavity and PCM operator are identical, failure of (v_C) is a source failure, not something that the continuum can repair. ([Q-Chem Manual][1])

The decisive mathematical point is that molecular polarizability is only a three-by-three contraction of the full density-response kernel:

[
\alpha_{ij}
===========

\int
(r_i-O_i)
\frac{\partial \rho(\mathbf r)}{\partial F_j}
,d\mathbf r .
]

It does **not** determine the induced density (\partial\rho(\mathbf r)/\partial F_j), much less its potential over a finite nonspherical cavity. Infinitely many induced densities have the same total induced dipole and therefore the same (\boldsymbol\alpha), while differing in quadrupolar, octupolar, radial, and charge-flow structure. Those null-space components generally produce different boundary potentials.

Your acetone result isolates exactly this non-identifiability:

* the molecular (\boldsymbol\alpha) is correct to (0.32%-0.42%);
* the ADT boundary potentials remain wrong by (35%-69%);
* the errors are (7.49)-(13.85) times the frozen numerical/basis uncertainty.

Therefore the defect is not principally the magnitude of the uniform molecular response. It is the **spatial realization of that response**.

The permanent branch independently fails: (31.3%) area-potential error and (35.0%) passive-continuum error. The molecule-held-out MBIS (q/p) head reducing ddPCM MAE from (8.19) to (2.53) kcal/mol shows that learning helps, but its failure also supports the conclusion that atom-centered (l\leq1) is not a sufficient source class.

The 505-result is corroborative rather than the proof: ADT improves MAE by only

[
1.696313-1.638962=0.057351\ {\rm kcal/mol},
]

or (3.38%), while leaving water at (2.011536) and a (14.688878) kcal/mol maximum.

The remaining caveats are narrow:

1. A wrong spherical-component ordering, normalization, Gaussian width, dipole origin, field sign, or source-to-PCMSolver map must still be corrected if present. That is an implementation correction, not an architectural loophole.
2. Acetone is one molecule. It does not establish the error distribution over all chemistry, but one valid in-domain counterexample is sufficient to falsify universal adequacy of the unchanged architecture.
3. An exact invertible coordinate transformation cannot help: if (c'=Tc) and the physical kernel is transformed consistently as (B'=BT^{-1}), then (B'c'=Bc). A transformation that changes the boundary potential is a new source model, not a re-expression.
4. Another geometry-only allocation of the same molecular (\boldsymbol\alpha) would be a new unvalidated physical prior. There is no target-free criterion that identifies it as the correct allocation from (\boldsymbol\alpha) alone.

So the zero-training route is exhausted for the **unchanged** source.

---

## 2. Minimal new component with both backbones frozen

**Choose C: one cavity-independent scalar electrostatic generating-functional head over a nonuniform potential basis.**

This is a strengthened version of B. A B-type head that accepts only a uniform three-vector field is insufficient, because it can learn only the same three-dimensional response subspace already shown to be non-identifying.

The new learned object should be

[
\mathcal G_\theta(R,\mathbf u),
]

where:

* (R) is the geometry;
* frozen MACE-MDP and MACE-POLAR equivariant features are inputs;
* (\mathbf u) represents a **general nonuniform external electrostatic potential** in a fixed atom-centered basis;
* the source basis is a compact Gaussian/density basis, with (l\leq3) as the smallest serious candidate and a small fixed radial ladder;
* no cavity coordinates, PCM surface points, solvent labels, or SMD quantities enter the head.

The permanent and induced sources are then exact derivatives of one scalar:

[
c^{(0)}_\theta(R)
=================

-\left.\nabla_{\mathbf u}\mathcal G_\theta(R,\mathbf u)\right|_{\mathbf u=0},
]

[
\Delta c_\theta(R,\mathbf u)
============================

-\nabla_{\mathbf u}\mathcal G_\theta(R,\mathbf u)
-c^{(0)}_\theta(R).
]

Thus:

* permanent source remains identifiable as the zero-field derivative;
* induced response remains identifiable as the zero-anchored increment;
* the continuum remains a separate operator;
* stock SMD-CDS remains a separate ledger term.

A scalar-first construction makes the response Jacobian a Hessian, so reciprocity follows by construction. Field-aware MACE models already use this principle for molecular/material response: a scalar electric enthalpy is differentiated to obtain polarization and polarizability, preserving Maxwell reciprocity. ([arXiv][2])

Why not A as written? Two independently trained permanent and induced source heads can fit the same QM data, but an unconstrained induced head need not be integrable, reciprocal, or passive. Once A’s induced head is required to be the derivative of a scalar functional, A has effectively become the proposed C, except with unnecessary duplication.

The operational MDP latent (q/p) readout should therefore be retired as the physical permanent source. The frozen MDP backbone, molecular dipole, and molecular polarizability remain valuable. Likewise, the MACE-POLAR backbone can remain as a feature generator. MACE-POLAR itself already represents long-range charge density through atom-centered Gaussian multipoles and projects general electrostatic potentials into atom-centered field features, so the proposed representation is compatible with its physical vocabulary without treating its existing multipoles as validated PCM sources. ([arXiv][3])

Crucially, **(l\leq3) is a candidate basis, not a proven sufficient basis**. It must first pass an oracle QM projection gate. If the best possible constrained (l\leq3) projection fails, add radial channels or increase angular order; do not compensate through cavity-dependent transformations.

---

## 3. Exact charge, MDP dipole, and MDP polarizability constraints

Use a fixed atom-centered source basis

[
\rho_\theta(\mathbf r;R,\mathbf u)
==================================

\sum_{k=1}^{M}c_k(R,\mathbf u),
b_k(\mathbf r;R).
]

Assume first that the source and potential coordinates have been biorthogonalized so their coupling is the Euclidean pairing. In a nonorthogonal implementation, every expression below simply acquires a fixed Gram matrix (G).

Define the charge and dipole moment maps

[
t_k=\int b_k(\mathbf r;R),d\mathbf r,
]

[
D_{ak}(R)
=========

\int (r_a-O_a)b_k(\mathbf r;R),d\mathbf r,
]

and

[
A(R)=
\begin{bmatrix}
t^\mathsf T\
D(R)
\end{bmatrix},
\qquad
b(R)=
\begin{bmatrix}
Q_{\rm exact}\
\boldsymbol\mu_{\rm MDP}(R)
\end{bmatrix}.
]

The permanent source must satisfy the hard equality

[
A(R)c^{(0)}_\theta(R)=b(R).
]

For normalized atom-centered monopoles and dipoles this reduces to the familiar identities

[
Q=\sum_A q_A,
]

[
\boldsymbol\mu_O
================

\sum_A
\left[
q_A(\mathbf R_A-\mathbf O)
+\mathbf p_A
\right].
]

Quadrupoles and octupoles do not alter these two global constraints.

A general hard construction is

[
c^{(0)}_\theta
==============

c_{\rm part}(Q,\boldsymbol\mu_{\rm MDP})
+
N_{Q\mu}(R)z_\theta(R),
]

where

[
A,c_{\rm part}=b,
\qquad
A,N_{Q\mu}=0.
]

Equivalently, from a raw prediction (\widetilde c),

[
c^{(0)}_\theta
==============

\widetilde c+
W^{-1}A^\mathsf T
\left(AW^{-1}A^\mathsf T\right)^\dagger
\left(b-A\widetilde c\right),
]

where (W) is a fixed, cavity-independent source metric such as a source-overlap or Coulomb metric. This projection removes only four global moment errors. It does not select the remaining density using a PCM cavity.

For charged molecules, (\boldsymbol\mu) must be referenced to one deterministic equivariant origin (O(R)), and the MDP dipole must be transformed to that same origin:

[
\boldsymbol\mu_{O'}
===================

## \boldsymbol\mu_O

Q(O'-O).
]

### Charge-conserving induced response

For every external potential,

[
t^\mathsf T\Delta c_\theta(R,\mathbf u)=0.
]

This should be structural, not a penalty. Let (N_Q) span the charge-neutral source subspace, (t^\mathsf TN_Q=0), and write

[
\Delta c_\theta=N_Q,y_\theta.
]

For a scalar functional, the same result follows by making the nonlinear response functional depend only on charge-neutral potential coordinates.

### Exact molecular polarizability

Let (U(R)\in\mathbb R^{M\times3}) embed a uniform electric field into the generalized potential coordinates. Choose the dual coordinates so that

[
\boldsymbol\mu=U^\mathsf Tc.
]

Define

[
J_0(R)
======

\left.
\frac{\partial c_\theta}{\partial\mathbf u}
\right|_{\mathbf u=0}.
]

The exact MDP polarizability constraint is

[
U^\mathsf T J_0 U
=================

\boldsymbol\alpha_{\rm MDP}.
]

This fixes only the uniform-field compression of (J_0), not its spatial response.

A constructive passive parameterization is available. Choose matrices (L) and (N) satisfying

[
t^\mathsf TL=0,\qquad U^\mathsf TL=I_3,
]

[
t^\mathsf TN=0,\qquad U^\mathsf TN=0.
]

Then define

[
J_0=
\begin{bmatrix}L&N\end{bmatrix}
H
\begin{bmatrix}L&N\end{bmatrix}^{\mathsf T},
]

with

[
H=
\begin{bmatrix}
\alpha_{\rm MDP}
&
\alpha_{\rm MDP}^{1/2}K^\mathsf T[2mm]
K\alpha_{\rm MDP}^{1/2}
&
KK^\mathsf T+RR^\mathsf T
\end{bmatrix}.
]

This construction gives, exactly,

[
t^\mathsf TJ_0=0,
\qquad
U^\mathsf TJ_0U=\alpha_{\rm MDP},
\qquad
J_0\succeq0.
]

It also retains learnable uniform–nonuniform cross-response through (K). It is therefore not ADT: the spatial lifting is learned from QM response data under exact moment constraints.

Write the generating functional as

[
\mathcal G_\theta(R,\mathbf u)
==============================

-c^{(0)\mathsf T}*\theta\mathbf u
-\Psi*\theta(R,\mathbf u),
]

with

[
\Psi_\theta(R,0)=0,
\qquad
\nabla_{\mathbf u}\Psi_\theta(R,0)=0,
\qquad
\nabla_{\mathbf u}^2\Psi_\theta(R,0)=J_0.
]

The total response should be convex in (\mathbf u=-v_{\rm ext}) over its declared field domain. This is the sign-reversed form of the concavity of ground-state energy with respect to external potential. ([arXiv][4])

No eigenvalue clipping is permitted. If a frozen MDP polarizability has a materially antisymmetric part or a negative physical eigenvalue beyond numerical tolerance, that geometry fails. Numerical antisymmetric roundoff may be removed by the identity ((\alpha+\alpha^\mathsf T)/2), but negative eigenvalues must not be repaired by geometry-dependent clipping.

MACE-MDP is explicitly a molecular dipole/polarizability model rather than an energy or force model, so using its molecular outputs as exact global constraints is consistent with its trained role; treating unsupervised latent atomic (q/p) as uniquely physical is not. ([MACE Documentation][5])

---

## 4. Independent QM supervision and finite gates

### QM supervision

Freeze one gas-phase reference level before opening model results. The existing (\omega)B97M-V/def2-TZVPD reference is suitable.

For every training geometry, supervise:

1. Zero-field total electrostatic potential on generic exterior probe points.
2. The constrained projection of the zero-field density or residual charge density into the chosen compact basis.
3. Response-energy differences
   [
   E_{\rm QM}(R,\mathbf u)-E_{\rm QM}(R,0).
   ]
4. Induced densities and induced exterior potentials
   [
   \Delta\rho_{\rm QM}(\mathbf u),
   \qquad
   \Delta v_{\rm QM}(\mathbf u).
   ]
5. Uniform-field first derivatives and polarizabilities.
6. Nonuniform response to cavity-independent perturbations.

Use three uniform-field directions and eight fixed nonuniform directions per geometry. The nonuniform directions should be fixed combinations of atom-centered Gaussian/harmonic potentials spanning (l=0,\ldots,3) and the available radial channels, projected to remove constant-potential and uniform-field components. Evaluate each direction at

[
+h,\quad-h,\quad+h/2,\quad-h/2.
]

Choose (h) once from finite-field convergence and SCF stability on a small numerical calibration set, then freeze it before training. Richardson differences between (h) and (h/2) define the finite-field uncertainty.

No PCM cavity should appear in the training inputs or losses. PCM surfaces are used only for independent validation of the exterior potential.

Use five molecule-held-out folds. All conformers, charge states used as the same molecular identity, and all field perturbations of an identity remain in one fold. Final qualification uses a sealed panel of 48 molecular identities and two geometries per identity, stratified by formal charge, size, flexibility, and heteroatom chemistry.

### Boundary-potential metrics

For cavity points (s_i) with area weights (a_i), define

[
e_A(v)
======

\frac{
\left[\sum_i a_i(v_i-v_i^{\rm QM})^2\right]^{1/2}
}{
\left[\sum_i a_i(v_i^{\rm QM})^2\right]^{1/2}
}.
]

For the frozen passive PCM reaction-energy metric (W_C\succeq0), define

[
e_C(v)
======

\frac{
\left[(v-v^{\rm QM})^\mathsf TW_C(v-v^{\rm QM})\right]^{1/2}
}{
\left[v^{{\rm QM}\mathsf T}W_Cv^{\rm QM}\right]^{1/2}
}.
]

Apply these separately to:

* permanent (v^{(0)});
* total induced (\Delta v);
* each uniform direction;
* each nonuniform direction.

Low-signal records with QM norm below five times numerical uncertainty are judged by absolute error in units of that uncertainty, not unstable relative errors.

Validate on the nominal PCMSolver cavity used in the acetone comparison and on fixed (0.95) and (1.05) radial dilations. This is a transfer test; none of these cavities enters the head.

### Representation gate before head training

The best constrained projection of the QM source into the candidate basis must satisfy, on a separate 24-geometry basis-qualification panel:

[
\operatorname{mean}(e_A)\leq0.05,
\qquad
\operatorname{mean}(e_C)\leq0.05,
]

[
P_{90}(e_A)\leq0.10,
\qquad
P_{90}(e_C)\leq0.10.
]

Failure means the basis itself is inadequate. Increase radial resolution or angular order. Do not train a head that cannot pass its own oracle projection.

### Learned permanent and induced potential gates

Permanent and induced predictions must each satisfy, in **every** molecule-held-out fold and on the sealed panel:

[
\operatorname{macro\ mean}(e_A)\leq0.10,
\qquad
\operatorname{macro\ mean}(e_C)\leq0.10,
]

[
P_{90}(e_A),P_{90}(e_C)\leq0.20,
]

[
\max(e_A,e_C)\leq0.25.
]

Additionally:

* area-weighted correlation must be at least (0.98) for at least (90%) of high-signal records;
* at least (90%) of records must lie within three times their combined finite-field, basis, and cavity-grid uncertainty;
* normalized signed passive-continuum bias must be at most (0.05);
* every formal-charge stratum and every (x/y/z) uniform direction must pass separately;
* the upper 95% molecule-bootstrap confidence bound on each macro mean must remain below (0.10).

These numerical thresholds are qualification decisions, not mathematical constants and not claims about eventual solvation MAE.

### Reciprocity

In dual coordinates,

[
J(\mathbf u)
============

# \frac{\partial c}{\partial\mathbf u}

\nabla_{\mathbf u}^2\Psi
]

must be symmetric. In native coordinates with source/potential metric (G), test (GJ).

Require

[
\frac{|GJ-(GJ)^\mathsf T|_F}{|GJ|_F}
\leq10^{-8}
]

for analytic derivatives and at most (10^{-4}) against independent finite-difference mixed derivatives. A scalar head should satisfy the first essentially by construction; failure indicates a broken source/field pairing or derivative path.

### Passivity and closed-loop stability

At zero field and every finite validation field,

[
\lambda_{\min}
\left[
\operatorname{sym}
\left(G^{1/2}JG^{-1/2}\right)
\right]
\geq
-10^{-6}
\max(1,\lambda_{\max}).
]

Also test the finite-field monotonicity inequality

[
(\mathbf u_1-\mathbf u_2)^\mathsf G
\left[c(\mathbf u_1)-c(\mathbf u_2)\right]
\geq -3\sigma_{\rm num}
]

for every paired field state.

For each validation cavity and dielectric, require a closed-loop linear-response margin

[
\lambda_{\max}
\left(
J^{1/2}K_CJ^{1/2}
\right)
\leq0.95
]

in the sign convention where positive (K_C) reinforces polarization. No damping, clipping, or source scaling is allowed to turn a failure into a pass.

### Rotations

For 24 fixed random Haar rotations per sealed geometry, require:

[
\frac{|\mathcal G(QR,Qu)-\mathcal G(R,u)|}
{\max(1,|\mathcal G|)}
\leq10^{-10},
]

[
\frac{|c(QR,Qu)-D(Q)c(R,u)|}
{\max(|c|,\epsilon)}
\leq10^{-7},
]

[
\frac{|J(QR,Qu)-D(Q)J(R,u)D(Q)^\mathsf T|_F}
{\max(|J|_F,\epsilon)}
\leq10^{-6}.
]

Boundary potentials evaluated at correspondingly rotated cavity points must agree to (10^{-7}) relative. One systematic rotation failure is an architectural failure, not statistical noise.

---

## 5. Whether the original MACE-POLAR nonlinear residual can remain

**It should not remain as a directly additive physical source.**

The acetone experiment does not directly falsify the nonuniform nonlinear residual, because it primarily falsifies the ADT uniform allocation. Therefore the residual’s physical accuracy remains unknown. However, direct addition of a frozen vector-valued residual to the new source would forfeit the scalar-head guarantee of reciprocity and passivity.

The original residual may remain only as a frozen **input feature** to the scalar functional. Its output must not be summed directly into the PCM source.

One finite terminal experiment is sufficient:

1. Train two otherwise capacity-matched scalar heads on identical QM folds:
   [
   H_0:\ \text{MDP/POLAR backbone features without the old residual output},
   ]
   [
   H_1:\ \text{same, plus the frozen zero-anchored POLAR residual as a feature}.
   ]
2. Evaluate once on the sealed 48-identity, 96-geometry panel using the three uniform and eight nonuniform directions at (\pm h,\pm h/2).
3. Retain the residual feature only if (H_1):

   * lowers both induced (e_A) and (e_C) by at least (15%) relative to (H_0);
   * has a paired molecule-bootstrap 95% lower confidence bound of at least (5%) improvement for both metrics;
   * passes every absolute potential, reciprocity, passivity, rotation, and closed-loop gate;
   * causes no formal-charge class or (P_{90}) tail to deteriorate by more than (0.02).

Otherwise, remove it permanently. No additional architecture modification follows that one-use decision.

This strict test is warranted because MACE-POLAR’s external-field response is an extrapolation from energy/force training rather than a response-supervised capability, and its published molecular polarizability performance is materially less accurate than its dipole performance. ([arXiv][3])

Even if the feature wins, the **new scalar functional**, not the original residual multipoles, remains the physical response source.

---

## 6. Theorem, evidence-supported inference, and unknown

**Theorem-level statements**

* For a fixed PCM cavity/operator, the solute boundary potential determines the induced surface response.
* Exact (Q), (\boldsymbol\mu), and (\boldsymbol\alpha) do not uniquely determine the permanent or induced boundary potential.
* An exact coordinate transformation of a source cannot alter its physical potential.
* A twice-differentiable scalar response functional gives reciprocal mixed response derivatives.
* Convexity of (\Psi), equivalently concavity of electronic energy in the external potential, gives differential passivity.
* The null-space constructions above impose charge, dipole, and polarizability exactly without using a cavity.

**Evidence-supported architectural inference**

* The unchanged MDP-point + POLAR-residual + ADT source is falsified as a generally quantitative PCM source.
* The excellent acetone molecular polarizability together with poor ADT boundary potentials identifies spatial allocation, not overall polarizability magnitude, as the principal uniform-response defect.
* The permanent MDP latent (q/p) source is independently inadequate.
* The failed molecule-held-out MBIS (q/p) result supports moving beyond (l\leq1).
* The minimal defensible repair is a QM-supervised, cavity-independent, scalar nonuniform-potential functional using the frozen backbones.

**Unknown**

* Whether (l\leq3) with the first chosen radial basis will pass the oracle projection gate.
* Whether frozen MDP/POLAR features contain enough information for molecule-held-out spatial-density response.
* Whether the old POLAR nonlinear residual contains useful information after scalarization.
* How much the new source will improve the 505 MAE, water MAE, or (14.69) kcal/mol maximum.
* Whether remaining cavity, CDS, reference-functional, standard-state, or dataset errors will dominate after the electrostatic source is repaired.
* Whether the final model will reach (1) kcal/mol MAE. Nothing in the present evidence establishes or guarantees that outcome.

[1]: https://manual.q-chem.com/5.2/Ch12.S2.SS2.html "12.2.2 Polarizable Continuum Models‣ 12.2 Chemical Solvent Models ‣ Chapter 12 Molecules in Complex Environments: Solvent Models, QM/MM and QM/EFP Features, Density Embedding ‣ Q-Chem 5.2 User’s Manual"
[2]: https://arxiv.org/abs/2508.17870 "[2508.17870] General Learning of the Electric Response of Inorganic Materials"
[3]: https://arxiv.org/html/2602.19411v1 "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry"
[4]: https://arxiv.org/html/2204.12216v1 "Lieb variation principle in density-functional theory"
[5]: https://mace-docs.readthedocs.io/en/latest/guide/polarizability.html "Dipole Moments and Polarizabilities with MACE — mace 0.3.13 documentation"
