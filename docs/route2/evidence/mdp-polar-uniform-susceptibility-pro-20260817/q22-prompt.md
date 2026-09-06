# Terminal zero-training review after a real rank-deficiency counterexample

We are improving a frozen-checkpoint MACE-MDP + MACE-POLAR implicit-solvation
hybrid.  No fitting, calibration, experimental solvation labels, response
scaling, eigenvalue clipping, geometry-dependent active subspaces, or
pseudoinverse regularization is allowed.  Post-training is a final fallback
only after zero-training physical constructions reach a terminal decision.

## Spaces and the rejected constrained construction

Let

- `u in U` be the native MACE-POLAR 8N potential-feature space;
- `c in C` be the atomwise 4N monopole/dipole source space;
- `U0: R^3 -> U` embed a uniform affine potential gradient;
- `G: U -> R^3` be the fixed arithmetic native-gradient chart with `G U0=I`;
- `J_P = d M_P / d u |_{u=0}` be the original MACE-POLAR source Jacobian;
- `B_P = J_P U0`;
- `C: C -> R^3` map atomwise charges/dipoles to the molecular dipole;
- `C_P = C B_P`;
- `alpha_D` be the symmetric positive-definite public MACE-MDP molecular
  polarizability.

The previous proposal restricted the additive correction to the original
POLAR uniform source span:

```text
K = B_P A G,
M_corr(u) = M_P(u) + K u,
C (J_P + K) U0 = C_P (I + A) = -alpha_D.
```

On the real official checkpoints for water, the target-free audit gives

```text
C_P = [[-0.0464805119,  0.0164364497, 0],
       [ 0.0164364497, -0.0379667587, 0],
       [ 0,             0,            0]]
singular values(C_P) = [0.0592023831, 0.0252448875, 0]
rank(C_P) = 2
rank(alpha_D) = 3
minimum relative Frobenius residual of C_P X = -alpha_D = 0.55974756
```

After an arbitrary proper rotation and translation, the singular values are
`[0.0592023830, 0.0252448875, 1.51e-17]`, the right null vector rotates exactly
with the molecular-plane normal, and the same relative lower bound remains.
The production constructor fails closed.  This is a structural counterexample,
not a conditioning accident.

## Two possible zero-training relaxations

### Candidate D: MDP atomwise-polarizability tangent

MACE-MDP exposes atomwise 3x3 tensors `alpha_A` satisfying exactly

```text
sum_A alpha_A = alpha_D.
```

They are the checkpoint's internal atom partition of its public molecular
polarizability, but only the molecular sum is physically supervised/validated;
the individual tensors need not be symmetric or positive semidefinite.

For `g = grad(phi) = -E`, define a source tangent `B_D` by

```text
dq_A = 0,
dp_A = -alpha_A g,
```

converted exactly to the repository's raw real-l=1 order.  Then

```text
C B_D = -sum_A alpha_A = -alpha_D.
```

Use the additive replacement

```text
M_D(u) = M_P(u) + (B_D - B_P) G u.
```

It closes the full molecular susceptibility even for planar molecules, kills
`ker(G)`, preserves total charge, and retains all nonlinear and nonuniform
MACE-POLAR response outside the replaced uniform linear tangent.  It leaves the
original POLAR uniform source span.

### Candidate ADT: source-bound free-atom density-translation tangent

The repository already contains a target-independent V0 atomic-displacement
translation-tangent asset derived from frozen free-atom HF densities.  It maps
an atom displacement into the analytic source change from translating that
atom's reference electron density; its source normalization and coordinate
derivatives are analytic/content-addressed.  A prior one-molecule acetone
QM-surface-MEP canary reported Frobenius error 0.1484, worst directional error
0.1611, and molecular-dipole mismatch 0.00387.  Denote a properly source-bound,
uniform-field tangent constructed from this primitive by `B_ADT`.  The analogous
replacement would be

```text
M_ADT(u) = M_P(u) + (B_ADT - B_P) G u.
```

No experimental solvation target is used.  However the exact mapping from the
MDP molecular polarizability to atomic density translations, and the risk of
double-counting the retained MACE-POLAR response, must be justified rather than
invented.

## Questions requiring a terminal mathematical/physical decision

1. Prove the no-go theorem for every correction `K=B_P A G` when
   `rank(C_P)<rank(alpha_D)`.  State precisely why a Moore-Penrose inverse,
   clipping, or a geometry-dependent rank completion cannot meet the stated
   constraints.

2. Is Candidate D a canonical zero-training relaxation, or is the atomwise
   partition a gauge/latent decomposition whose use as real induced atomic
   dipoles is underdetermined?  Give necessary and sufficient or clearly
   sufficient conditions on `{alpha_A}` for charge conservation, translation
   invariance, SO(3) and permutation covariance, reciprocity, passivity,
   smoothness, and a physically meaningful work/energy interpretation.

3. Compare Candidate D with Candidate ADT.  Can `B_ADT` be combined with the
   MDP molecular `alpha_D` without fitted atomic weights?  If not, identify the
   missing mathematical object.  Does either additive formula double-count the
   original POLAR response, or does subtracting `B_P G u` make it an exact
   tangent replacement?

4. The operational continuum uses separate spaces and operators

```text
c = M(R,u),
A_R sigma = B_R c,
u = L_R sigma.
```

For the best admissible zero-training candidate, derive the exact source JVP,
VJP, coordinate VJP terms, coupled root Jacobian, and a conservative
operational force for a frozen scalar ledger.  Distinguish this from strict
Tier-V/common-functional admission.

5. Give exact target-free rejection/admission gates before any accuracy panel:
   uniform susceptibility closure, gauge/charge, SO(3), JVP/VJP, passivity or
   its honest absence, five-start root audit, minimum singular value/strong
   monotonicity, finite-field behavior, and coordinate finite differences.

6. Choose exactly one terminal recommendation:

   - implement Candidate D;
   - implement Candidate ADT (with the missing canonical construction stated);
   - implement a different zero-training construction with an explicit formula;
   - or declare a zero-training no-go and move only then to a separately named
     frozen-backbone scalar/source head trained on independent QM electrostatic
     data.

Do not recommend empirical scaling, fitted weights, target-selected kernels,
geometry-local spectral repairs, or changing signs by trial and error.  Clearly
separate theorem, sufficient assumptions, numerical gates, and scientific
inference.
