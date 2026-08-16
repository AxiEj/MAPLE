# Route 2 — MACE-MDP + MACE-POLAR Hybrid

## Ownership

This workspace advances only the hybrid Route-2 profile:

```text
MACE-MDP permanent q/p
+ MACE-POLAR zero-anchored induced response
+ continuum electrostatics
+ a separately declared solvent term
```

Pure MACE-POLAR is developed in a separate workspace. Evidence, scalar IDs,
source definitions, checkpoints, and admission decisions from the two routes
must not be mixed.

## Frozen scientific target

The primary quantitative target is:

```text
mean absolute error <= 1.5 kcal/mol
```

on the preregistered frozen 505-record MNSol development panel, using the
exact profile bound by
`route2-hybrid-smd-development-prereg-v3`. The 148-record confirmation
partition remains sealed. The v3 result is a frozen baseline: it cannot be
retroactively altered by fitting, calibration, checkpoint selection, or
method selection.

The running evidence source is read-only:

```text
source snapshot: ~/.local/share/maple/route2/hybrid-smd-development-v3/source
records:         ~/.local/share/maple/route2/hybrid-smd-development-v3/records
```

This workspace was copied from that snapshot. It may evolve only under new,
explicitly versioned profiles; it must not alter the running snapshot or
retroactively change its preregistered method.

## Chrome Pro mathematical advisor

The retained Windows Chrome session is an approved autonomous external-advisor
surface for this hybrid route. When a material scientific decision depends on
advanced mathematics or remains unresolved after a concrete local derivation,
the agent should use that session to ask the genuine **Pro** model rather than
silently guessing or reducing the claim. Appropriate topics include:

- variational and stationary-functional constructions;
- implicit-function, adjoint, HVP, and mixed-derivative derivations;
- reciprocity, gauge, passivity, conditioning, and root-uniqueness arguments;
- equivariant continuum discretization and cavity regularity;
- identifiability of electrostatic and CDS components;
- adversarial review of a proposed scientific admission proof.

Before submitting, the UI must visibly identify the selected model as `Pro`
(the prior validated surface displayed `Pro, 5 of 5`), not `Medium` or an
unverified default. The prompt, selected-model evidence, answer, timestamps,
and SHA256 digests should be retained in a repository evidence note, following
[`docs/route2/evidence/HARMONIC_GREEN_OPERATOR_PRO_AUDIT_2026-08-14.md`](docs/route2/evidence/HARMONIC_GREEN_OPERATOR_PRO_AUDIT_2026-08-14.md).

Pro output is an external mathematical review, not authoritative implementation
or admission evidence. Every accepted result must be independently rederived,
translated into an explicit repository contract, and verified by analytic
identities, finite differences, or other appropriate tests. Routine coding
questions should remain local; Pro should be used promptly when the uncertainty
is genuinely mathematical or could change the scientific model.

For this goal, Pro review is a required second-opinion checkpoint rather than a
one-off escalation. Unless a single completed answer explicitly closes the next
checkpoint too, ask a new, focused Pro question before freezing each of:

1. the target-blind M3 statistical subspace and its grouped validation rule;
2. the final positive-parent, SO(3)-structured smooth CDS area and its force
   regularity contract;
3. the source/continuum/energy-ledger interpretation and matched QM/PCM
   component admission;
4. the complete block/reduced implicit-force derivation and root-well-posedness
   certificate; and
5. the final public scientific admission proof.

Questions should combine tightly related mathematics, but a still-running query
must never be duplicated or overlapped. Poll sparsely, retain the complete
answer and its evidence hashes, then critically rederive and test it before the
next checkpoint is submitted. Passing Pro review never substitutes for local
proof, chemical reference data, or profile-bound validation.

## Prospective profile-bound CDS lane

Stock SMD-CDS transferability is a hypothesis tested by the frozen v3
baseline, not an invariant. SMD electrostatic and CDS decompositions are
model-dependent, so a separately versioned effective CDS may be developed
after the complete v3 component matrix is frozen.

The first such candidate is water-only:

```text
MAPLE-CDS-W1
= low-dimensional linear aqueous SMD surface-tension basis
+ a frozen differentiable area definition
+ frozen MACE-MDP + MACE-POLAR hybrid electrostatics v3
```

It may fit only the residual between the frozen electrostatic prediction and
the experimental total under the frozen standard-state convention. It must
not alter source scales, Gaussian widths, dielectric factors, cavity radii,
checkpoints, or the electrostatic ledger. It is an effective model-dependent
term, not an experimentally observable standalone CDS component.

The 505 records remain the development pool; no retroactive blind subset may
be carved out after partial execution. The existing 148-record confirmation
manifest remains the only current MNSol confirmation partition. A water-only
candidate and a future multi-solvent candidate require different profile IDs
and admission decisions. See
[`docs/route2/MAPLE_CDS_W1.md`](docs/route2/MAPLE_CDS_W1.md).

### Frozen CDS terminal result

The final positive-parent PH1.0 CDS lane was preregistered before target
joining and then executed exactly once.  All ten numerical minimizer
certificates passed, but the conjunctive development gate failed:

```text
water grouped-OOF exact-minimizer MAE upper bound: 1.6969972 kcal/mol  FAIL
mixed-505 exact-minimizer MAE upper bound:          1.4546582 kcal/mol  PASS
required:                                           both <= 1.5 kcal/mol
```

Consequently, no all-306 deployment fit exists and no further CDS feature,
loss, mode, intercept, regularizer, fold, or threshold variant is permitted.
This closes only the attempt to repair the current electrostatic profile by
increasing CDS capacity; it does not close the hybrid route.

The next mainline is a target-independent heterogeneous-electrostatics audit:

```text
MDP permanent source = point monopoles/dipoles
POLAR induced source = exact zero-anchored 1.5 A Gaussian increments
continuum             = shared-cavity harmonic coefficient operator
```

Analytic single-sphere identities and frozen-source permanent-only,
induced-only, and direct-sum comparisons against PCMSolver must pass before a
new accuracy profile is frozen.  The existing PH1 OOF CDS predictions may be
carried only as the immutable CDS term of that comparison; they may not be
refitted.  The terminal result and post-failure Pro review are recorded in
`docs/route2/evidence/POSITIVE_PH1_TERMINAL_FAILURE_2026-08-17.md` and
`docs/route2/evidence/POSITIVE_PH1_POSTFAILURE_PRO_AUDIT_2026-08-17.md`.

## Required MAPLE capability surface

The final hybrid route must expose one content-addressed scalar per solvent and
derive all supported quantities from that same scalar:

- energy;
- conservative force;
- molecular virial/stress where defined;
- Hessian-vector products;
- Cartesian Hessian and frequencies;
- stable OPT/TS/IRC and, after independent validation, MD.

The implementation must remain modular in three independent spaces:

1. permanent and induced MLIP source models;
2. source-to-continuum and continuum-to-native-field operators;
3. additive solvent free-energy terms.

Source/receiver dimensions need not be artificially identical. Every operator,
checkpoint, solvent definition, cavity, unit convention, derivative route, and
runtime must be bound by stable provenance.

## Admission boundaries

Passing the 505 energy target establishes only development-set full-solvation
energy accuracy for the frozen profile. It does not by itself admit forces,
Hessians, optimization, frequencies, or MD.

Those capabilities require, on the same exact profile:

- cold/warm replay and root residual checks;
- analytic-force versus multi-step scalar finite differences on distorted
  geometries;
- translation, rotation, permutation, and torque covariance;
- closed-loop work and branch/topology guards;
- HVP/Hessian symmetry and finite-difference convergence;
- downstream OPT/FREQ/NVE validation appropriate to the claimed domain.

Strict common-functional Tier V is a separate claim. The operational hybrid
route may provide a conservative composite PES through an explicitly frozen
ledger and complete implicit differentiation without claiming that the
original MACE-MDP/MACE-POLAR response equations are stationary equations of a
single electronic functional.
