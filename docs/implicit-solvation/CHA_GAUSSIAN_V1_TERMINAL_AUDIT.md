# Gaussian CHA v1: executed, not force-qualified

## Outcome (2026-10-01)

**`GAUSSIAN_OPT_INCOMPLETE` — no scientific or production admission.**
The runner defects recorded in the earlier progress checkpoint were repaired
and regression-tested. The pinned CPU-float64, three-site water campaign then
ran through the existing MACE calculator and MAPLE LBFGS implementation.

| Check | Passing cases |
|---|---:|
| Real optimization convergence and uncached final replay | 15/15 |
| Independent crossing/event energy and force checks | 150/150 |
| Combined energy/force consistency at initial geometries | 15/15 |
| Independent solvent energy/force checks at final geometries | 4/15 |
| Combined energy/force consistency at final geometries | 2/15 |
| All final-geometry gates | **2/15** |

The largest terminal solvent energy difference is `4.06449e-8 kcal/mol`,
but the largest solvent force difference is `6.36189e-4 kcal/mol/Angstrom`.
The maximum combined-force discrepancy is `1.01136e-6 Hartree/Angstrom`.
Energy agreement and optimizer convergence therefore do not establish force
qualification. No width, geometry, failure, or fixed acceptance gate was removed.

## Diagnosed numerical defect

Focused component probes isolate the dominant failure to the R6 atomic-contact
integral. Its global latitude breakpoints change ordering or coalesce near
equal O-H distances; independently quadratured moving panels then produce an
artificial finite-quadrature force change. Tiny steps confined to one panel
regime agree with automatic differentiation: the Born graph is live, but the
numerical integration is not sufficiently smooth across that chart change.
The corresponding torus, cavity, and dispersion controls agree with their
own energy derivatives near `1e-11` in their respective units.

Any repair requires a separately identified numerical revision and a fresh
campaign. Raising the production order, relaxing force gates, changing the
Gaussian width, freezing Born radii, or substituting runtime finite-difference
forces would not resolve this qualification failure. The v1 scalar and all
negative results are preserved.

## Evidence boundaries

The local negative-evidence package binds 252 files, including all 15 terminal
receipts and a byte-verified 178-file source archive. Its manifest SHA-256 is
`db9d862e3438cacf884ee9654aa0f33ab7cc58cab067462a3445e3f82be400f8`.
Private inputs, raw artifacts, and model weights are not distributed here.

For the evaluated campaign, each process records one narrowly constrained
`ldconfig -p` during Torch
bootstrap. During scientific execution all subprocesses are denied; network
and OpenMM access remain denied throughout. No prohibited attempts occurred.
This is not a whole-process zero-subprocess claim. An earlier attempt that
failed during bootstrap before any OPT is retained separately.

Fresh full solvation tests: 1714 passed, 37 failed, 10 skipped, one error.
The 38 failure/error IDs match the preserved baseline, with no new failures.
These tests do not override the failed numerical gates above. Ordinary `.inp`
selection, general molecular support, physical-accuracy claims, and FREQ/TS/MD
remain outside the qualified scope.
