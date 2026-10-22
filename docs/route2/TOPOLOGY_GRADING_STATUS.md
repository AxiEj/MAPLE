# FP32 topology grading: warnings without skipping proven smooth cases

## Scope and implementation

Defaults remain FP32 / lmax 5 / Lebedev 194. This change does not disable
all topology protection or relax a residual/conditioning tolerance.

- Stable paired quintics use one centered coordinate q=(ratio-1)/eta for
  support classification, polynomial arguments, normalization and diagnostics.
- A single interior contribution with no plateau-one contribution has a proven
  lower branch: its small exposed weight is evaluated directly and its overlap
  normalizer is exactly one, even if chi rounds to one.
- A plateau-one contribution proves the buried upper branch. Smooth quintic
  endpoints match through second derivatives and are warnings, not hard stops.
- True multi-interior near-f=1 ambiguity remains a hard error. Invalid tensors,
  coincident singularities, resource limits, bad matrix rank/conditioning,
  non-finite results, and failed solve residuals retain their existing checks.
- FP64 retains the legacy switch/clamp arithmetic and policy.
- Diagnostic warning codes are emitted as RuntimeWarning and retained in
  `diagnostics()['topology_warnings']` / `fp32_partition` and sweep records.

Core files: `maple/solvation/continuum/fp32_partition.py` and `torch_ddpcm.py`.
Policy ID: `ddx-v0.8.0-fp32-complementary-quintic-partition-v1`.

## Exact-input before/after results

Water solute in acetonitrile; same representable coordinates and source-model
checkpoint, eta, lmax and grid; no coordinate finite-difference derivatives.
The original center, +1e-5 A and +1e-4 A critical-direction points were replayed
on CPU FP32, CUDA FP32 and a same-grid CPU FP64 reference. All nine post-fix
runs use strict checks; no diagnostic topology bypass is used.

Maximum solvation-Hessian component difference from the matched FP64 result:

| Configuration | Before (eV/A²) | After (eV/A²) |
|---|---:|---:|
| CPU FP32 | 0.00545890 | 0.0000137904 |
| CUDA FP32 | 0.00546492 | 0.0000098960 |

The formerly rejected center now passes with a warning. Ordinary public CPU
and CUDA acetonitrile `.inp` single-point/force calls also complete. This is
numerical agreement, not experimental accuracy or global Hessian qualification.

## Full solvent coverage replay

- CPU: 179/179 strict energy/force successes, zero skips, 10 warning cases.
- CUDA: 179/179 strict energy/force successes, zero skips, 10 warning cases.
- 358/358 requested cases are retained; 8/8 selected HVP checks pass per device.
- Published source-bound engineering and multisolvent artifacts were genuinely
  rerun on current source. Prior rejecting artifacts are preserved byte-for-byte
  in `docs/route2/evidence/history/`; unfinished source/worktree evidence is
  recoverable under `.omx/checkpoints/before-topology-grading-20261017T123136Z`.

Local replay and before/after accounting:
`.omx/research/topology-grading-20261017T123445Z/`.
This integration does not extend the admitted tasks to MD/TS/IRC or establish
experimental solvation accuracy. The historical v3 source-bound qualification
mismatch remains a separate, previously disclosed boundary.

## Final validation

- Targeted implementation, legacy FP64, topology, solvent, CDS and artifact
  contracts: **714 passed, 1 skipped** (one existing deprecation warning).
- Core four-case source-bound E/F/H/HVP replay: 4/4 passed.
- Complete cached-component solvent replay: 358/358 passed; no survivor filtering.
- Three exact-input MeCN points x three precision/device configurations:
  9/9 strict accepted, all finite and within the existing Hessian symmetry budget.
- Ordinary MeCN CPU/CUDA `.inp` replays: both exited successfully, retaining
  the nonblocking RuntimeWarning.
- AST parse, targeted pyflakes and `git diff --check`: passed.
- Current published artifacts have zero source-manifest mismatches.
