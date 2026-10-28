# Route 2 performance diagnostic — 2026-10-28

## Status and scope

This records the read-only measurements made before the next optimization.
It is a diagnostic checkpoint, not an implemented speedup, a release admission,
or evidence that a different numerical backend is interchangeable everywhere.

Measured checkout: `d6563c6cdb40b5a028d7876910328a9a468ab8d2`.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU. Runtime: PyTorch 2.12.0+cu130,
one CPU thread, FP32, lmax 5, 194 Lebedev points, frozen response, water solvent.
The real model was resident; each public PES `evaluate_forces` call computed
fresh energy and forces without a final-result cache. Loading was excluded.
These measurements are not paired with the older ASE-facade benchmarks.

The probes were executed inline and their stdout is retained in the conversation
tool record. This document transcribes aggregate results; no standalone raw-run
artifact or reproducible harness was saved for these exploratory measurements.
Future acceptance requires a durable, source-bound harness and fresh results.

## Current energy/force baseline

Three warmups preceded five synchronized, uninstrumented calls per geometry.
Stage instrumentation used three separate calls; nested stage medians are not
additive and instrumentation can perturb scheduling. Instrumented energy and
force differences against the warm reference were exactly zero in these cases.

| Atoms | Uninstrumented median ms | MACE forward ms | ddPCM forward ms | Of which two full SVD checks ms | CDS forward ms | Total reverse ms |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 291.74 | 69.18 | 85.58 | 62.13 | 26.43 | 87.88 |
| 20 | 749.34 | 75.16 | 192.13 | 162.88 | 181.14 | 290.87 |
| 35 | 1336.34 | 74.04 | 385.55 | 323.75 | 374.24 | 520.02 |

Peak allocated CUDA memory during the uninstrumented calls was approximately
265, 416 and 749 MiB respectively; this is not a complete host/device resource
admission. At 35 atoms, the two dense solves took about 7.06 ms, PES configuration
checks 7.54 ms, and six graph-connectivity checks together 0.98 ms.

A separate retained-graph reverse diagnostic at 35 atoms gave median times
of 570.50 ms for the total, 66.62 ms for vacuum, 133.47 ms for ddPCM, and
382.27 ms for CDS. These component times must not be added: ddPCM reverse
includes the shared MACE source graph. They identify CDS as a substantial
reverse-path cost, not an exact attribution of the original whole-call timing.

## Full-spectrum SVD placement probe

The same captured CUDA FP32 `L` and `R_epsilon` matrices were evaluated by
CUDA `svdvals(driver="gesvd")` and CPU FP32 `svdvals`. Each case used five
alternating-order pairs after warmup. CPU timing includes matrix device-to-host
transfer; both timings include the returned singular values on the host.
No production method was replaced by the CPU candidate in this probe.

| Atoms | Matrix dimension | Two CUDA SVDs median ms | Two CPU SVDs plus transfer median ms |
|---:|---:|---:|---:|
| 9 | 324 | 60.48 | 8.29 |
| 20 | 720 | 159.03 | 68.42 |
| 35 | 1260 | 346.77 | 360.15 |

All six matrices had matching numerical rank and pass/fail classification.
Singular values were not bitwise identical. No near-threshold matrix corpus,
parameter-reconstruction suite, or candidate end-to-end qualification was run.
CPU placement is therefore only a small/medium-matrix hypothesis; an unconditional
CPU switch is not supported. The rank tolerance, reciprocal-condition threshold,
both direct solves, residual checks, and fail-before-solve ordering must remain.
Any placement change must have an explicit numerical execution identity rather
than continuing to claim the existing requested-device-only SVD policy.

## Next bounded work

1. Test a separately identified small-matrix full-SVD placement candidate with
   a predeclared end-to-end, classification, derivative and resource gate.
2. Treat CDS graph/continuous-kernel reduction as a separate potential larger
   improvement. Do not combine it with SVD work or repeat the rejected
   call-local pair-geometry reuse design.
3. Preserve all physical tunables and CPU/FP64 behavior. No approximate SVD,
   tolerance relaxation, parameter fitting, persistent geometry/result cache,
   dependency installation, or Route 4 change is justified by these probes.

Relevant contracts: `continuum/torch_ddpcm.py` implements detached full-spectrum
rank/condition checks; `surfaces/legacy_dareal.py` implements analytic areas;
`experimental/mace_polar_torch.py` owns the one-forward derivative boundary.
All paths are under `maple/solvation/`.

## Measured source identity

| File under maple/solvation | SHA-256 |
|---|---|
| experimental/mace_polar_torch.py | `280f03d620ff9b3cdea5b4ed0afa887aa7188c5d98373a5b1412e7f9bc69ff95` |
| continuum/torch_ddpcm.py | `95a861c5c3dfd4498cce8ada63b6a2bf62904b7045d745f3902cac27943a0f2e` |
| nonpolar/legacy_smd_cds.py | `670a362f0d00447fee62c689ab41a9016a60899a974f960880e797dfa6eeab53` |
| surfaces/legacy_dareal.py | `3cb9edaf61ffbb79b8cc9feefa1ade35dc96100dd0440be33fd020e5097e4240` |

The source hashes matched before and after measurement. No production source,
scientific parameter, or existing receipt was modified.
