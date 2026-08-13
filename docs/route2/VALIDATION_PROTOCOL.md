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
the original molecular-realspace/CPCM194 candidate passed one directional test
but failed rotation/torque.  The separately versioned fixed-box40/CPCM590
diagnostic then passed all nine Cartesian components at three steps and six
rigid orientations on one equilibrium water geometry, followed by its
seven-geometry force/path audit and the equilibrium 32/40/48/56-Angstrom box
tail audit. It remains single-molecule evidence only; the required
multi-molecule PES panel, multi-geometry box convergence, component physics,
Hessian, and NVE gates remain open. Exact measurements and reproducible
commands are recorded in `FIXED_BOX590_WATER_DIAGNOSTIC.md`.

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

The first fixed-box590 distorted-water path contract is frozen in
`run_fixedbox590_water_path_diagnostic.py` before executing the real stack. It
contains equilibrium, symmetric compression/stretch, asymmetric stretch, two
bends, and one combined distortion. At every geometry the three deterministic
translation-free water coordinates are checked with `4e-4`, `2e-4`, and
`1e-4 Angstrom` central differences. A separate symmetric-stretch/bend
rectangle uses four equal subintervals per edge and composite Simpson
integration; it is repeated cold and sequential-warm in both directions. This
is a single-molecule path gate, not the required 20-molecule PES panel.

The clean-commit execution at `241e98b7` passed all of those preregistered
single-water path gates. Its worst directional errors were `1.84e-5 eV/A`
absolute and `3.75e-4` relative; cold/warm forward/reverse loop work had
magnitude `7.22e-6 eV`, and the topology hash was constant. Raw evidence is in
`evidence/fixedbox590-water-path-241e98b7/`. This does not change the remaining
multi-molecule, box-convergence, component-physics, Hessian, or NVE gates.

## Fixed-box operator-convergence gate

The fixed-box reciprocal evaluator is a numerical model operator and must not
inherit the 40-Angstrom choice without convergence evidence. The first
preregistered family is `32/40/48/56 Angstrom`; every size has a distinct
evaluator ID, model profile, provider/configuration hash, and disabled
solvation profile. The 56-Angstrom calculation is the finite reference and the
48-to-56 tail must satisfy, on the same geometry/root/scalar:

- total-energy and continuum-component changes `<= 1e-4 eV`;
- force RMS change `<= 1e-4 eV/Angstrom`;
- force maximum change `<= 5e-4 eV/Angstrom`;
- normalized source change `<= 1e-5`;
- primal residual `<= 1e-12` for every size;
- adjoint true residual `<= 1e-10` for every size.

These thresholds and sizes are frozen before the clean real-stack run. Passing
one equilibrium-water comparison only admits the box choice for continued
diagnostics; it is not Tier E/F or multi-geometry/multi-molecule convergence.

The clean `a7fdf2fa` execution passed all five tail gates: 48-to-56 changes
were `2.87123e-6 eV` total energy, `2.38313e-8 eV` continuum energy,
`6.78548e-7 eV/Angstrom` force RMS, `1.24714e-6 eV/Angstrom` force maximum,
and `8.27721e-7` relative source. The 40-to-56 total-energy and force-RMS
changes were `2.14385e-7 eV` and `3.98917e-7 eV/Angstrom`, respectively. The
raw artifact is in `evidence/fixedbox590-water-box-convergence-a7fdf2fa/`;
capabilities remain closed because this is one molecule and one geometry.

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
