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

## Harmonic-Galerkin structural gates

The disabled coefficient-space continuum reference has additional gates that
must not be replaced by random-rotation evidence alone:

- every per-atom space contains complete `l=0..L` irreps with fixed dimension;
- the exposure product map is rectangular with product bandwidth
  `P >= L + L_e`; incomplete bandwidth fails closed;
- the pair-axis Coulomb block commutes with its complete `SO(2)` stabilizer and
  is independent of the transverse section used to map `+z` to the pair axis;
- `K(QR)=D_P K(R) D_P.T`, `E(QR)D_L=D_P E(R)`, and
  `S(QR)D_c=D_L S(R)` hold to floating-point/invariant-quadrature error;
- `A=E.T K E` is symmetric, raw `K` is positive definite, `E` has full column
  rank, and `A` is positive definite without eigenvalue clipping;
- source/receiver is exactly the same stored `S/S.T` pair;
- the differentiable candidate independently matches NumPy `E`, `K`, `V`,
  `A`, and `S`, while coordinate partials and mixed pullbacks come only from
  the sealed stationary scalar and match multi-step central differences;
- rotation tests include an exactly polar pair axis so a pair-section branch
  cannot silently erase transverse derivatives;
- coincident centres, full-burial rank loss, nonfinite state, excessive
  condition number, or content-hash drift fail closed;
- nested, intersecting, tangent, and separated regimes retain fixed dimensions;
- the exact shell `l=0` block is `C1` but has the expected second-derivative
  jump at tangency; no Tier H/FREQ domain may cross such an event without a
  separately versioned smoother physical kernel.

Increasing harmonic order or invariant radial quadrature may improve physical
accuracy. It must not systematically improve rotation covariance; such a trend
would indicate that a laboratory-frame approximation has leaked into the
mathematical operator. See `HARMONIC_GALERKIN_TIER_V.md`.

## Real-stack same-scalar force gates

Use float64, at least three displacement sizes, multiple directions,
geometries, and orientations. Directional derivatives require relative error
`<= 2e-3` away from zero and absolute projected-force error
`<= 5e-4 eV/Angstrom`. Cartesian tests require RMS
`<= 5e-4 eV/Angstrom`, maximum `<= 2e-3 eV/Angstrom`, and no systematic
first-order/non-convergent step trend.

The primal/adjoint residual contribution to force uncertainty must be below ten
percent of the force-FD tolerance.

The disabled changed-source strict-variational candidate has one separate
real-checkpoint implementation canary at clean head `576550e9`. It solves the
common state for equilibrium water, replays the root cold/warm, and evaluates
one translation-free stationary-envelope direction at `5e-4`, `2e-4`, and
`1e-4 Angstrom`. All three local comparisons pass the directional thresholds,
and a second cold process reproduces measurement SHA-256
`62d43c63868270fc74254cf0ddbc182b33af0d158c3fbca1ef6493cdf52ee9fc`.
The raw record is
`evidence/variational-common-water-576550e9/`. It is explicitly below
admission scope: it changes the source identity, samples only one
geometry/direction, and uses a laboratory-fixed Lebedev continuum without a
structural global `SO(3)` guarantee. It therefore admits no `E/F/H/V/M` tier.

The separately identified analytic Gaussian-multipole model evaluator is
tracked by a second, still-disabled protocol. At clean head `50809803`, the
analytic-model plus smooth-harmonic common scalar converged one water state,
replayed its scientific measurement exactly in a second process, passed the
same three envelope steps, and passed one proper-rotation check. The complete
stationary-scalar rotation errors were `2.799424692057073e-9 eV` in energy and
`8.198584915195558e-8` relative in the coordinate gradient. The raw record is
`evidence/variational-analytic-harmonic-water-50809803/`. Because this changes
the inference/model identity and covers only one geometry, direction, and
rotation, it also admits no `E/F/H/V/M` tier.

The independently versioned protocol is frozen in
`RESIDUAL_FORCE_GATE.md`. It uses three primal tolerances and three adjoint
tolerance pairs, separates the two residual spaces, requires contraction or a
declared numerical plateau, applies a factor-two tail estimate, and gates both
RMS and maximum estimated error at `5e-5 eV/Angstrom`. It is an empirical
a-posteriori residual-refinement estimate, not a rigorous analytic upper bound.
The clean-head 20-molecule execution at `9918dea6` passed: worst estimated RMS
and component errors were `3.70243e-13` and `1.26565e-12 eV/Angstrom`, versus
the frozen `5e-5 eV/Angstrom` budget. Raw evidence is in
`evidence/fixedbox590-residual-force-9918dea6/`. This closes only the empirical
residual-contribution clause; other Tier-F gates remain open.

Current executed real-stack evidence is intentionally below admission scope:
the original molecular-realspace/CPCM194 candidate passed one directional test
but failed rotation/torque.  The separately versioned fixed-box40/CPCM590
diagnostic then passed all nine Cartesian components at three steps and six
rigid orientations on one equilibrium water geometry, followed by its
seven-geometry force/path audit and the equilibrium 32/40/48/56-Angstrom box
tail audit. It subsequently passed the preregistered 20-molecule,
60-base-geometry directional panel and 11 additional torsion/stretch path
points on clean head `f7f68165`. Exact measurements and reproducible commands
are recorded in `FIXED_BOX590_WATER_DIAGNOSTIC.md`, `PES_PANEL.md`, and the
corresponding evidence bundles.

The independently aggregated reference-geometry Cartesian panel subsequently
passed at clean head `abb34a05`: 20 molecules, 465 components, and 1395
component/step comparisons gave maximum RMS and component errors of
`5.89334e-6` and `1.85578e-5 eV/Angstrom`. Every molecule had an observed
first-to-last central-difference order of at least `1.94289`; maximum primal
and adjoint residuals were `9.99155e-13` and `2.46413e-12`. Raw source-bound
evidence is in `evidence/fixedbox590-cartesian-panel-abb34a05/`.

These results close the frozen directional and component-resolved derivative
thresholds, not Tier F. The subsequent residual-refinement panel at
`9918dea6` also passed its empirical residual-contribution gate. All-panel
symmetry/loop scope, matched component physics, multi-geometry box convergence,
public workflow integration, Hessian, and NVE gates remain open.

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
`evidence/fixedbox590-water-path-241e98b7/`. Later multi-molecule directional
and reference-geometry Cartesian panels also passed, but all-panel symmetry
and closed-loop coverage, box convergence beyond one equilibrium geometry,
component physics, Hessian, and NVE remain open.

The first all-panel extension is now preregistered in `SYMMETRY_PANEL.md`:
every one of the 20 reference molecules receives three frozen rotations, one
rigid translation, one identical-atom permutation, and one bidirectional
cold/warm 17-point loop. The real panel has not yet been executed; all
capabilities stay closed regardless of its eventual result.

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
