# Route 2 Goal: pure MACE-POLAR exact-GTO accuracy

> **Current branch direction (2026-09-20): withdraw MACE-MDP/MACE-POLAR hybrid
> development. This worktree now targets the accuracy of the pure MACE-POLAR
> exact-GTO solvation profile.**

## Scientific identity

The primary profile is the existing, explicitly named exact-GTO route:

```text
macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1
```

Its identity is:

- official, unmodified `MACE-POLAR-1-M` checkpoint in float64;
- no MACE-MDP checkpoint, permanent source, energy, or response contribution;
- PCMSolver IEFPCM with the runtime-verified intrinsic SMD electrostatic cavity;
- the checkpoint-native finite-width exact-GTO reaction-field projection;
- the registered MACE-POLAR operational SCF ledger;
- a separately identified SMD-CDS nonpolar term and the frozen standard-state
  convention.

The current profile intentionally has a point-`l<=1` cavity source and an
exact-GTO receiver. It is therefore an operational, source/receiver-
nonconjugate, energy-only fixed point. Accuracy work must not relabel it as a
common variational functional or conservative solution-phase PES.

## Hybrid withdrawal boundary

MACE-MDP/MACE-POLAR hybrid profiles, campaigns, and admission work are no
longer part of this worktree's active goal. Existing negative results and
historical evidence remain immutable provenance records; they must not be
deleted or reused as pure exact-GTO evidence. All hybrid registry entries
remain disabled. No new hybrid benchmark, repair, admission, or capability
claim is authorized on this branch.

The pre-pivot state is recoverable from the local archive branch
`archive/route2-conservative-vnext-hybrid-20260920`; its uncommitted MNSol148
work is retained in the named 2026-09-20 hybrid archive stash.

## Frozen baselines

The currently reproducible FreeSolv-10 development result for the primary
exact-GTO profile is:

| metric | value (`kcal/mol`) |
| --- | ---: |
| MAE | `0.9152863783` |
| RMSE | `1.3054578142` |
| mean signed error | `-0.2943811841` |
| maximum absolute error | `3.2473279957` |

The maximum-error failure is acetic acid. Exact GTO improves 7/10 paired
records relative to local-jet SCF but is not a universal improvement. The
authoritative record is
`docs/implicit-solvation/benchmarks/route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json`.

The separately named frozen radial-GTO/ddPCM MNSol-10 profile is not the same
scientific identity. Its retained MAE is `2.4925344349 kcal/mol`; it must not
be combined with the PCMSolver exact-GTO result or presented as the baseline
above.

## Accuracy-improvement protocol

1. Freeze checkpoint, exact-GTO projector, PCMSolver equation/cavity,
   geometries/conformers, standard state, and dataset membership before model
   selection.
2. Diagnose polar/electronic and CDS/nonpolar components separately. A lower
   total error does not excuse a worse electrostatic component.
3. Use development data only for selection. Keep confirmation labels sealed
   until the candidate identity and scoring protocol are frozen.
4. If the SMD-CDS term is reparameterized, fit the SMD CDS coefficients under
   a new method identity for this fixed exact-GTO polar term. Do not use a
   global residual shift or response scale.
5. Deduplicate FreeSolv and MNSol by chemical identity rather than geometry
   hash, and report overlap explicitly.
6. Report MAE, RMSE, mean signed error, maximum absolute error, per-record
   errors, provider failures, and component errors. Do not report only an
   aggregate that hides outliers.
7. Preserve failed candidates and negative records. Do not retune radii,
   Gaussian widths, thresholds, source normalization, or benchmark membership
   after reading the target results.

## Acceptance boundary

Accuracy improvement requires a prospectively frozen, identity-bound
development result that reduces the exact-GTO error distribution without
changing the polar identity, followed by independent confirmation. The
existing `MAE <= 1.5 kcal/mol` requirement remains necessary, while the present
`3.2473 kcal/mol` maximum-error failure must be addressed and disclosed.

Passing an accuracy panel admits neither forces, Hessians, frequencies,
optimization, TS/IRC, MD/NVE, nor a strict variational claim. Each capability
requires its own same-scalar derivative and provenance evidence.

## Engineering constraints

- Prefer existing MAPLE source/field/continuum/ledger boundaries and delete or
  archive hybrid-only wiring rather than adding a parallel stack.
- Add no dependency and change no official checkpoint.
- Keep exact-GTO, local-jet, point-`l<=1`, radial-GTO, and continuum backends as
  separate identities.
- Use `PYTHONPATH="$PWD"` and single-thread BLAS/OpenMP settings for
  source-sensitive validation.
- Keep temporary scientific diagnostics and private row-level results outside
  the tracked tree unless a persistent artifact is explicitly required.
