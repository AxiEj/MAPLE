# Operational analytic-harmonic Tier-F validation contract

This document freezes the non-rigid validation framework for the disabled
operational energy ledger

```text
route2-operational-macepolar-analytic-gaussian-multipole-
smoothharmonicgalerkin-cpcm-v1
```

The fixed point selects a state; it does not uniquely determine an energy.
This profile explicitly freezes the ledger currently implemented as vacuum
energy plus the harmonic continuum half-coupling.  Conservative derivatives of
that ledger do not, by themselves, prove that it is the physically correct
solvation free energy.

This document does not enable `E/F/H/V/M`.  It reuses the old panel's geometry asset and
mathematical gate definitions only.  No energy, force, root, residual,
topology, or pass/fail value from a fixed-box/laboratory-grid profile is
transferable to this profile.

## Frozen identity

- analytic isotropic Gaussian-multipole MACE evaluator;
- original learned four-channel source embedded only in the first radial block;
- smooth weighted-overlap harmonic-Galerkin electrostatic scalar;
- no laboratory-fixed surface grid;
- no CDS/nonpolar term;
- no field-conditioned MACE energy difference or solute internal-polarization
  closure term;
- official `polar-1-m` checkpoint, float64;
- all public capability flags remain false during evidence generation.

The existing twenty-molecule asset remains frozen at SHA-256
`ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`.

## Panel A: distorted and path directional derivatives

Run all twenty molecules at reference, deterministic bond compression, and
deterministic bond stretch, plus the frozen trans-butane torsion and hydrogen
peroxide O--O path.  Every direction uses central steps

```text
4e-4, 2e-4, 1e-4 Angstrom
```

and the existing absolute/relative gates (`5e-4 eV/Angstrom`, `2e-3` away from
the `1e-3 eV/Angstrom` floor).  Every base and path geometry additionally
records:

- roots from both `+1e-3` and `-1e-3` reduced-coordinate starts;
- source, field, energy, primal, and adjoint replay values;
- exact continuum half-coupling and zero missing-radial-block checks;
- distinct-centre, weighted-basis rank, minimum singular-value, minimum
  surface-eigenvalue, and condition-number margins.

The independent aggregator recomputes every derivative and gate from raw
energies, forces, geometries, roots, and margins.

## Panel B: full Cartesian finite differences

At all twenty frozen reference geometries, evaluate all 465 Cartesian force
components at the same three steps.  Preserve the existing RMS
`5e-4 eV/Angstrom`, maximum-component `2e-3 eV/Angstrom`, and
second-order-or-low-plateau convergence contract.  The same multistart,
scalar-identity, residual, and domain-margin gates apply.

## Panel C: bidirectional closed-loop work

For all twenty reference geometries, use the existing two deterministic
orthogonal internal directions, amplitudes `(0.02, 0.02) Angstrom`, and four
subdivisions per edge.  Recompute cold forward/reverse and path-warm
forward/reverse Simpson work, root repeatability, residuals, and domain margins.
Rigid values are recorded by this runner but the already completed dedicated
rigid panel remains the authoritative rigid-symmetry evidence.

## Admission boundary

The runners and independent aggregators are frozen now, but the expensive real
checkpoint campaign is downstream of the native-semantics, separated
source-to-boundary/continuum-to-native-field, source/MEP, energy-ledger, and
root-well-posedness terminal audits.  Those audits may reject or supersede this
ledger without invalidating the reusable geometry and raw-recomputation
framework.

Passing these three panels establishes sampled same-scalar derivative,
root-branch, loop-work, and domain evidence for this exact profile.  It does
not by itself establish:

- matched electrostatic component accuracy;
- total experimental solvation free energy;
- public calculator/workflow integration;
- Hessian, FREQ, TS, HVP, NVE, or MD;
- strict common-functional Tier V;
- uniqueness or physical correctness of this operational energy ledger;
- global uniqueness outside the sampled and guarded domain.

Tier V remains a separate disabled research route.  Its current negative
evidence is not a prerequisite for this operational Tier-F validation.
