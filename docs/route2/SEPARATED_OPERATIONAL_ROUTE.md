# Route 2 separated operational branch

## Decision

The original MACE-POLAR checkpoint is retained only as an **operational
response model**. Its four source coefficients and eight receiver features are
different mathematical spaces:

```text
C4 --B(R)--> Sigma*    A(R) sigma = B(R)c    U8 <-L(R)-- Sigma
                         c = M_theta(R, u)
```

The generic state kernel eliminates `sigma` but preserves those categorical
boundaries:

```text
y = T_plus M_theta(R, L A^-1 B(c_ref + T y)).
```

`L=B*` is **not** an operational requirement. The present smooth harmonic
construction happens to derive both from one analytic eight-channel Gaussian
kernel, with the original source embedded by an explicit content-addressed
`4 -> 8` matrix. A future field-responsive MLIP may supply different source
and receiver representations by implementing the same separated protocol.

## Checkpoint-native facts

The official local checkpoint, not a paper recollection or a hard-coded
default, is authoritative. The current canary reads and hashes:

- source widths, angular order, and normalization;
- receiver widths, angular order, normalization, and layout;
- the live upstream projection matrix;
- field-feature transform and normalization;
- spin-channel count and the two recursive response updates;
- self-interaction and local-electron-energy switches.

For the currently cached official checkpoint the source width is `1.5 A` and
the receiver widths are `1.5 A` and `3.0 A`. This branch therefore rejects the
unverified `1.5/2.0 A` recollection for this artifact. The canary intentionally
leaves the applied-field energy sign and complete external-enthalpy semantics
unverified.

## Frozen ledgers, not a unique energy

A fixed point selects a state, not an energy. Two distinct, disabled scalar
IDs are registered:

```text
Phi0 = E_vac(R) - 1/2 (Bc)^T A^-1(Bc)
Phi1 = E_intrinsic_theta(R, L sigma) + 1/2 sigma^T A sigma
```

`Phi0` is the existing vacuum-plus-continuum ledger and explicitly omits an
internal solute polarization cost. `Phi1` is an external-enthalpy candidate;
it does not make the original source the derivative of the checkpoint energy.
Neither is selected as physically correct until matched energy, force,
surface-MEP, and continuum-component references are run. Both remain absent
from the public profile registry and admit no E/F/H/V/M capability.

## Terminal strict-variational audit

There is one remaining mathematically legitimate quotient-space check for the
unchanged checkpoint:

```text
B M_theta(R,u) + s L^T grad_u E_intrinsic(R,u) = 0
B M_u(R,u) L is self-adjoint on the boundary coefficient chart.
```

A coupled-curl failure at one admitted state is terminal. A direct
energy/source failure becomes terminal only after Gate 1 has verified that the
consumed scalar is the complete external enthalpy and has frozen its sign; the
current intrinsic-energy hook does not yet meet that prerequisite. A finite
pass is only local numerical evidence, never a global Tier-V proof.
Geometry-dependent projections, spectral clipping, path integrals,
residual-squared energies, and solver damping are not accepted repairs. A fixed
quadratic correction would be a separately named new model and is not
implemented here.

## Physical and root gates

Source validation is deliberately decomposed into charge, molecular dipole,
traceless quadrupole, far/surface MEP, and matched fixed-source PCM energy.
No reference is inferred by the code, and no total-solvation comparison may be
mixed with an electrostatic-only component.

For `r(y)=y-F(y)`, the root diagnostic records:

- `sigma_min(I-J_F)` for a local implicit branch;
- a weighted contraction norm;
- a weighted strong-monotonicity margin.

Anderson, DIIS, damping, a small final residual, and a pointwise Jacobian do
not prove global uniqueness. A residual-to-root error bound is returned only
with an externally evidenced invariant domain and a uniform domain bound.

## Claim boundary

This change establishes modular contracts, hashes, disabled ledgers, and
diagnostic mathematics. It does **not** establish:

- a unified variational functional for the original checkpoint;
- source/MEP accuracy against QM;
- a preferred `Phi0` or `Phi1` energy ledger;
- a globally unique smooth root;
- distorted-geometry PES, Cartesian finite differences, or closed-loop work;
- a public conservative force or any solvation-accuracy claim.

Those gates must be rerun under this exact separated scalar/state identity;
evidence from the earlier eight-dimensional embedded-source profile cannot be
silently inherited.
