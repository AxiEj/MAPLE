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
unverified `1.5/2.0 A` recollection for this artifact. The structural canary
alone intentionally leaves the applied-field energy sign and complete
external-enthalpy semantics unverified. A separate content-addressed
`FieldSemanticsManifest` binds the checkpoint, adapter configuration and
provenance, source/native-field
spaces, pairing metric, channel order, radial widths, units, real-harmonic
convention, coordinate transform, origin policy, spin-channel factor,
work-term inclusion, and supporting measurement SHA-256 values. Unknown
sign/work/origin fields remain explicit `None`/`unverified`; they cannot be
inferred from an `external_field` variable name.

The clean two-process checkpoint replay at commit `3014f1a6` resolves this
specific branch question negatively. Upstream uniform-field and MAPLE native
injection produce identical source and dipole, but upstream minus native raw
energy equals the explicit `+E dot mu` work to `6.63e-14 eV`. The energy
difference is `3.853499335e-3 eV`, the fixed-field force difference is
`3.533242862e-3 eV/A`, and the directional-derivative difference is
`0.4928416604 e A`. The native raw graph passes its own reverse/forward AD,
finite-difference, zero-field, and charging checks; it is simply a different
energy branch that omits the upstream explicit work. Both runs reproduce
measurement SHA-256
`e77ab89951e8eefec3480571fcc8df03026090f06aa15b817c464d033f230439`.
Evidence is retained under
[`evidence/mace-field-semantics-3014f1a6/`](evidence/mace-field-semantics-3014f1a6/README.md).

## Frozen ledgers, not a unique energy

A fixed point selects a state, not an energy. Two distinct, disabled scalar
IDs are registered:

```text
Phi0 = E_vac(R) - 1/2 (Bc)^T A^-1(Bc)
Phi1Delta = E_vac(R)
            + [E_conditioned_raw(R, L sigma)-E_conditioned_raw(R, 0)]
            + 1/2 sigma^T A sigma
```

`Phi0` is the existing vacuum-plus-continuum ledger and explicitly omits an
internal solute polarization cost. `Phi1Delta` is vacuum-normalized, but the
replay above proves that its consumed native-injection branch is not the
upstream complete external enthalpy. It is therefore retained only as a
disabled diagnostic/ablation scalar, not as an admitted external-enthalpy
ledger. It also does not make the original source the derivative of the
checkpoint energy. `Phi0` is not selected as physically correct until matched
source, surface-MEP, continuum-component, and energy references pass. Both
remain absent from the public profile registry and admit no E/F/H/V/M
capability.

## Terminal strict-variational audit

The only mathematically legitimate quotient-space check for the unchanged
checkpoint was:

```text
B M_theta(R,u) + s L^T grad_u E_conditioned_raw(R,u) = 0
B M_u(R,u) L is self-adjoint on the boundary coefficient chart.
```

A coupled-curl failure at one admitted state is terminal. The direct
energy/source residual is not reinterpreted as a full-enthalpy residual,
because the field-semantics replay proves that the consumed raw branch omits
the upstream explicit work. A finite pass would only be local numerical
evidence, never a global Tier-V proof.
Geometry-dependent projections, spectral clipping, path integrals,
residual-squared energies, and solver damping are not accepted repairs. A fixed
quadratic correction would be a separately named new model and is not
implemented here.

The clean official-checkpoint water audit at commit `0246e887` now closes this
last loophole negatively. The relative coupled-curl defects are
`0.2081259719` at zero field and `0.2085410719` at a deterministic
continuum-active nonzero field, against `1e-9`; two processes reproduce the
same scientific measurement digest. Fixed-charge response leakage is below
`8.7e-17`. Evidence is retained under
[`evidence/mace-coupled-conjugacy-terminal-0246e887/`](evidence/mace-coupled-conjugacy-terminal-0246e887/README.md).
This terminal result applies to unchanged-checkpoint Tier V, not to the
operational state/ledger research branch.

## Physical and root gates

Source validation is deliberately decomposed into charge, molecular dipole,
traceless quadrupole, far/surface MEP, and matched fixed-source PCM energy.
The decisive continuum-active comparison consumes a QM-projected boundary RHS
directly, not another fitted atom-charge model. For symmetric positive-definite
`A`, it reports

```text
d = sqrt((b_ML-b_QM)^T A^-1 (b_ML-b_QM))
|G_ML-G_QM| <= ||b_QM||_(A^-1) d + 1/2 d^2
```

and derives the accepted `d` from a preregistered fixed-source energy budget.
Every record binds geometry, continuum configuration/provenance, topology,
cavity, and reference artifact SHA. No reference is inferred by the code, and
no total-solvation comparison may be mixed with an electrostatic-only
component.

Symmetry and positive definiteness of `A` make this metric legal but do not by
themselves validate the continuum as a quantitative reference. A case can pass
only when a separate validation artifact is bound to the exact geometry,
continuum configuration/provenance, topology, and cavity identity. An
internally converged but unvalidated continuum may report diagnostics, but it
cannot admit or reject the source.

The clean four-case audit at head `1d40c93b` therefore uses the already frozen
QM MEPs and intrinsic PCMSolver IEFPCM cavities rather than treating the new
harmonic candidate as its own reference. Both executions reproduce
measurement SHA-256
`ff40c7c426cbcb59ab28378f29a1ed5675da98d4063fba20423b0247d997649b`.
Total charge passes in all records, and molecular dipoles are much closer than
the near-field observables, but area-weighted surface-MEP relative errors are
`0.628` to `1.034`. The fixed-source polarization-energy absolute errors are
`2.824` to `11.828 kcal/mol`; all four exceed the inherited strict
`1 kcal/mol` budget. Evidence is retained under
[`evidence/mace-original-source-pcmsolver-four-1d40c93b/`](evidence/mace-original-source-pcmsolver-four-1d40c93b/README.md).

This closes the unchanged original four-channel source as a full quantitative
PCM source for the separated profile. Consequently neither `Phi0` nor
`Phi1Delta` proceeds to ledger, force, PES, or public admission on this source
identity.

A later preregistered four-case checkpoint audit passes charge, dipole,
traceless-quadrupole, and all three far-field MEP shell gates. Both clean runs
reproduce measurement SHA-256
`dc9cce0f1ced9c34af1ab67bdae329d4521f2e67b780a1adcad6b7dd645d97d8`;
the raw evidence is
[`evidence/mace-original-source-farfield-four-9c1d4fb9/`](evidence/mace-original-source-farfield-four-9c1d4fb9/README.md).
This satisfies a necessary low-multipole/far-field precondition and therefore
authorizes exactly one separately named, fixed, symmetry-preserving radial-
embedding research experiment. It does **not** prove that the near-field defect
is radial-only; the new profile must still pass held-out cavity-surface MEP and
fixed-source PCM-energy gates before any ledger is evaluated.

The preregistered experiment has now been executed twice at clean head
`7c5fc16b`. A universal `sigma=0.75 Angstrom` embedding passes its training and
held-out distant-surface MEP gates, but fails every matched intrinsic-cavity
PCMSolver energy case: absolute errors are `1.829` to `4.888 kcal/mol`.
Evidence is retained under
[`evidence/mace-fixed-radial-source-terminal-7c5fc16b/`](evidence/mace-fixed-radial-source-terminal-7c5fc16b/README.md).
Per the frozen termination rule, no further radial patch is authorized and no
ledger or force work proceeds on either the original or repaired source.

For `r(y)=y-F(y)`, the root diagnostic records:

- `sigma_min(I-J_F)` for a local implicit branch;
- a weighted contraction norm;
- a weighted strong-monotonicity margin.

Anderson, DIIS, damping, a small final residual, and a pointwise Jacobian do
not prove global uniqueness. A residual-to-root error bound is returned only
with an externally evidenced invariant domain and a uniform domain bound.

## Claim boundary

This change establishes modular contracts, hashes, and two terminal negative
decisions for the unchanged checkpoint: original-source Tier V fails the
coupled-curl gate, and the same source fails the matched quantitative PCM
source gate. It does **not** establish:

- a unified variational functional for the original checkpoint;
- a validated repaired source/MEP model (the single authorized fixed-radial
  experiment has now failed its matched PCMSolver energy gate);
- an admitted `Phi0` or `Phi1Delta` ledger (`Phi1Delta` additionally fails
  complete-enthalpy semantics for the current native-injection branch);
- a globally unique smooth root;
- distorted-geometry PES, Cartesian finite differences, or closed-loop work;
- a public conservative force or any solvation-accuracy claim.

The current separated source identity terminates here. The authorized radial
embedding, any scalar-first field head, or any independent variational
polarization model must receive a new profile/hash and rerun its own gates;
earlier force or ledger evidence cannot be silently inherited.
