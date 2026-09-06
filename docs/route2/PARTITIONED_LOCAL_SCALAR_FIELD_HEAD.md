# Partitioned local scalar field head

Status: **stockholder coefficient-label variant rejected; observable-supervised
scalar functional retained as the next hybrid mainline**

## Why this architecture is needed

The completed source audits establish both sides of the representation problem:

- the original MACE-MDP point source and MACE-POLAR residual/ADT response do
  not reproduce the matched-QM cavity MEP or nonuniform response;
- standard atom-centred auxiliary bases reproduce fixed-source PCM energies
  extremely accurately, but their globally fitted molecular coefficient
  vectors are nonidentifiable on aromatic overlap patterns, even after the one
  allowed fixed local whitening attempt.

The failed global coefficients must not become ML labels. The continuous
density/MEP span remains useful only after a unique smooth local ownership rule
is introduced.

## Fixed local ownership

For geometry `R`, define a positive rotational scalar

\[
p_A(\mathbf r;R)=
\exp[-\alpha_{Z_A}|\mathbf r-\mathbf R_A|^2]
\]

and the normalized partition

\[
w_A(\mathbf r;R)=
\frac{p_A(\mathbf r;R)}{\sum_Bp_B(\mathbf r;R)}.
\]

The element widths are a versioned physical asset and may not depend on a PCM
cavity, solvent, model output, or solvation target. Evaluation uses a log-sum-
exp implementation. Consequently

\[
0<w_A<1,\qquad \sum_Aw_A=1,
\]

and the partition is translation covariant, rotation invariant, permutation
equivariant, and smooth for noncoincident nuclei.

The diagnostic reference atomic fields are defined, rather than inferred, by

\[
\rho_A^{\rm ref}(\mathbf r)=w_A(\mathbf r;R)\rho_{\rm QM}(\mathbf r).
\]

The sum of the atomic reference fields is exactly the QM density before local
basis projection. However, the preregistered 12-case test showed that fitting
these components independently loses important inter-centre interference:
the reaction-metric upper-bound mean/max became 0.707/1.011 kcal/mol. Therefore
the partition is retained only as smooth ownership/feature infrastructure.
Partitioned local coefficients are forbidden as training labels.

The trainable architecture does not infer, fit, or supervise any global or
partitioned density coefficient vector. It learns the scalar directly from
independent observable targets.

## Scalar-first electronic contract

Let `B_R z` be the physical source represented in the fixed internal radial-
spherical span and let `eta_R(v)=B_R^*v` be the exactly dual external-potential
projection. The final trainable object is

\[
A_\theta(R,z)=
\Psi_\theta^{\rm cvx}(h_R,z)-b_\theta(h_R)^Tz
+\frac{\mu}{2}z^TDz,
\qquad \mu>0,\quad D\succ0,
\]

with strong convexity on the fixed-charge tangent space. The observable field
scalar is the zero-anchored constrained dual

\[
\mathcal E_\theta(R,v)=E_{\rm vac}^{\rm MDP}(R)
+\min_{a^Tz=Q}[A_\theta(R,z)-\eta_R(v)^Tz]
-\min_{a^Tz=Q}A_\theta(R,z).
\]

The geometry features come from the frozen MACE-MDP backbone. Frozen zero-field
MACE-POLAR latent features may be supplied only as optional inputs in a
capacity-matched ablation. Arbitrary field-conditioned hidden activations, the
old four-channel source, and ADT may not bypass the conjugate coupling.

If `z*(R,v)` is the unique minimizer, then one derivative gives both sources:

\[
s_{\rm perm}=B_Rz^*(R,0),\qquad
s_{\rm ind}=B_R[z^*(R,v)-z^*(R,0)],
\]

and the tangent response is positive semidefinite and reciprocal. The strongly
convex latent energy chooses a unique internal gauge even though the entries of
`z` are not observables. No molecular Coulomb inverse or coefficient target is
used.

`QuadraticLatentSourceFunctional` is the checked algebraic oracle for the first
quadratic instance. The existing `FieldEnergyFunctional` derivative ownership
and passive training contracts remain the production boundary. Model-specific
code may provide scalar/convex tensors only; it may not implement an independent
source, JVP, VJP, HVP, or coordinate derivative.

Total charge and constant-potential gauge are enforced by the existing fixed-
charge affine chart. Molecular dipole and polarizability are supervised QM
observables, not frozen MACE-MDP constraints. This distinction is essential for
a field-responsive model.

## Required target-free QM data

Training and validation may use only matched electronic observables:

1. zero-field QM density and exterior MEP;
2. field-conditioned electronic energies;
3. uniform-field dipoles and polarizabilities;
4. nonuniform point-charge perturbation energies and induced MEPs;
5. the stockholder partition only as a deterministic feature/diagnostic; its
   rejected local coefficient projection is not a target.

Experimental solvation energies, CDS residuals, FreeSolv/MNSol targets, and the
sealed confirmation partition are forbidden during architecture selection or
training. They remain downstream full-ledger evaluation only.

## Admission sequence

1. **Independent QM observables:** zero-field MEP/dipole plus uniform and
   genuinely nonuniform field energies and responses on grouped train,
   validation, and untouched blind identities.
2. **Scalar derivative:** energy/source directional derivatives and mixed
   coordinate-field derivatives from one graph.
3. **Passivity and root:** response spectrum, combined continuum Hessian, and
   cold multi-start root identity.
4. **Matched continuum:** fixed-source component and self-consistent
   electrostatic energy against the same cavity/operator.
5. **PES:** Cartesian finite differences, distortions, closed-loop work,
   rotation/translation/permutation covariance, and topology guards.
6. **Full ledger:** only after the electrostatic component passes, add one
   separately versioned differentiable nonpolar/standard-state scalar and open
   untouched solvation targets.

No intermediate result admits MAPLE `E/F/H/V/M` production capability.

## Literature alignment

- SALTED learns atom-centred radial/spherical density and density-response
  fields with the corresponding two-centre metric in the loss rather than
  treating raw coefficient RMSE as the physical objective:
  <https://salted.readthedocs.io/en/latest/theory/>.
- The 2025 symmetry-adapted vector-field model directly targets electron-density
  response under electric fields: <https://arxiv.org/abs/2501.11019>.
- The 2026 self-consistent electrostatic MLIP design-space analysis motivates a
  shared density representation and scalar energy rather than an appended
  response correction: <https://arxiv.org/abs/2603.14700>.
- Recent E(3)-equivariant DFT acceleration work demonstrates species-dependent
  auxiliary-density coefficient heads, while MAPLE deliberately replaces the
  globally nonidentifiable coefficient label with the fixed local partition:
  <https://arxiv.org/abs/2509.25724>.
