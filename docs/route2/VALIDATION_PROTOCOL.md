# Route 2 validation protocol

Thresholds below are preregistered for conservative-vNext. A changed threshold
requires a new admission-contract version before blind evaluation.

## Algebraic float64 gates

| check | threshold |
| --- | ---: |
| source/receiver dot product | relative error `<= 1e-10` |
| JVP/VJP transpose | relative error `<= 1e-9` |
| synthetic implicit adjoint vs independent AD/FD | relative error `<= 1e-8` |
| normalized charge constraint | `<= 1e-12` |
| reciprocal linear C-PCM half coupling | relative error `<= 1e-10` |
| synthetic raw Hessian antisymmetry | relative Frobenius `<= 1e-8` |

## Real-stack same-scalar force gates

Use float64, at least three displacement sizes, multiple directions,
geometries, and orientations. Directional derivatives require relative error
`<= 2e-3` away from zero and absolute projected-force error
`<= 5e-4 eV/Angstrom`. Cartesian tests require RMS
`<= 5e-4 eV/Angstrom`, maximum `<= 2e-3 eV/Angstrom`, and no systematic
first-order/non-convergent step trend.

The primal/adjoint residual contribution to force uncertainty must be below ten
percent of the force-FD tolerance.

Current executed real-water evidence is intentionally below admission scope:
one geometry, one direction, and three step sizes passed the directional gate,
but the same calculation failed the rotation-force and torque thresholds.
Consequently it is local derivative evidence only; component Cartesian FD,
multi-geometry/orientation, loop-work, PES-panel, Hessian, and NVE gates remain
required.

## Symmetry, root, topology, and path gates

- rigid-translation energy change `<= 1e-6 eV`;
- net force after translation `<= 1e-5 eV/Angstrom`;
- rotation-covariance relative force error `<= 1e-4`;
- torque residual `<= 1e-4 eV`;
- cold/warm energy difference `<= 1e-8 eV`;
- normalized cold/warm source difference `<= 1e-8`;
- identical topology and ownership hash along the admitted path;
- forward/reverse, cold/warm closed-loop work
  `<= max(1e-5 eV, 1e-3 sum(abs(F dot dR)))`.

Root uniqueness is tested with declared multi-start seeds and the actual
unmixed dimensionless residual. Damping/DIIS convergence alone is not a root
uniqueness proof.

## Hessian and frequency gates

The raw force-FD Hessian is retained. Before optional symmetrization,

```text
||H-H.T||_F / max(||H||_F, 1 eV/Angstrom^2) <= 1e-3
```

The defect must decrease or plateau under three-step refinement. Projected
translations/rotations must be within 10 cm-1 of zero for an unconstrained
small nonlinear molecule. A minimum has no meaningful mode below -20 cm-1. A
TS has exactly one robust mode below -50 cm-1 with the intended reaction-vector
overlap, reproducible at two accepted displacement sizes.

## Evidence interpretation

- synthetic tests establish algebra, not chemical accuracy;
- one-geometry FD establishes a local derivative, not a PES;
- a short NVE test is a diagnostic, not Tier M admission;
- totals cannot hide wrong components;
- compare solvation components only under explicit source/receiver, cavity,
  continuum-equation, and nonpolar identities; an electrostatics-only value
  cannot be scored against an experimental total solvation free energy;
- missing optional runtimes are skips only in a generic job and failures in a
  job advertising that runtime;
- every artifact binds exact source, model, continuum, cavity, runtime, command,
  raw values, and warnings.
