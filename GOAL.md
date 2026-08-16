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
`route2-hybrid-smd-development-prereg-v2`. The 148-record confirmation
partition remains sealed. No fitting, calibration, checkpoint selection, or
method selection is permitted after observing this development result.

The running evidence source is read-only:

```text
source snapshot: /tmp/maple-route2-v5-sourcebound.7btWng
records:         /tmp/maple-route2-hybrid-smd-development-v2
```

This workspace was copied from that snapshot. It may evolve only under new,
explicitly versioned profiles; it must not alter the running snapshot or
retroactively change its preregistered method.

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
