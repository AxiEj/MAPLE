# Route 2 scan execution contract v1

This contract freezes execution requirements, not new scientific parameters.
It applies to future scan engines. Creating this document does **not** claim that
all six engineering changes have already been implemented or admitted.

## Immutable scientific identity

- Official frozen MACE-POLAR source/checkpoint; no source renormalization.
- Original ddPCM equations, two solves, rank/conditioning, residual, finite-value,
  source and resource gates. Existing private scalar topology policy is unchanged;
  this is not force/Hessian admission.
- FP32, lmax=5, Lebedev194, eta=0.1; no autocast, TF32, reduced precision or
  relaxed solver tolerance to make GPU timings attractive.
- A: all ten element Coulomb radii = 1.1*R_SMD(original solvent)+0.10 Angstrom.
- B: all ten element Coulomb radii = 0.9*R_SMD(original solvent)+0.05 Angstrom.
- Oxygen is frozen with the ORIGINAL catalog acidity before any fitted alpha/beta.
- Fitted nonaqueous alpha/beta affect CDS only. Water is unchanged. CDS coefficients,
  SASA radii/probe, special environments and branch-boundary checks stay unchanged.
- No new parameter selection, confirmation scoring, overwrite of old energy files,
  or full-population rerun is authorized by an engineering benchmark.

## Q1. Dynamic persistent CPU workers

- At most 10 workers; Torch, BLAS, OpenMP and inter-op CPU thread counts are 1.
- Use completion-driven dispatch, not a fixed batch barrier.
- A task owns an exact geometry/source/actual-radius family and all its compatible
  solvent responses. Group by physical numerical identity, not just A/B label.
- Keep dielectric-independent geometry/operators/psi/L checks shared inside the task.
- The coordinator claims numerical keys as pending/running/committed to prevent
  concurrent cache misses from recomputing the same key.
- Task manifests enumerate exact expected state keys. Counts and receipt validation
  are generated from these manifests, not hard-coded four-shard assumptions.
- Resume validates immutable existing records and computes missing states only;
  preserve failures and interrupted attempts. No survivor-only MAE.

## Q2. Memory admission

- 10 is a concurrency ceiling, not an instruction to bypass resource rejection.
- Admit a task only when its conservative workspace estimate plus measured worker,
  source/index cache, parent and pending-result memory fit the available budget.
- Record observed process/tree peak RSS and, for CUDA, allocator allocated/reserved
  peaks and initial free memory. Retain original resource preflight as well.
- Do not materialize the complete historical JSON index independently in 10 workers;
  use immutable task subsets, shared read-only data, or coordinator lookup.

## I1. Immutable index versions

- Coordinator alone registers validated completed outputs and handles conflicts.
- A dispatched task binds an immutable version/path and content SHA. Never overwrite
  that file while it is in use. New tasks may use a newer published version.
- Cache identity includes geometry, Q/M, exact source FP32 bytes, actual radii,
  continuum configuration and numerical policy. A candidate label is insufficient.
- Snapshot, used record/source files and implementations remain content-bound;
  stale/conflicting/tampered inputs fail explicitly. No permissive fallback.
- Keep verified sources/index entries resident with explicit lifetime/invalidation.
  Rebuilding or rereading the entire history per task is prohibited.

## C1. CDS geometry reuse

- Cache geometry-only distances, COT and all special environment features once for
  identical symbols/positions/SASA radii/probe/dtype/device/configuration.
- Different alpha/beta update effective coefficients, beta-squared term and final
  reductions only. Do not interpolate energies or drop a special environment term.
- Preserve original arithmetic order where required for CPU bitwise energy parity.
- Water uses verified immutable results. Shards receive only their required records.

## G1. GPU experiment / promotion boundary

- CPU-only in the old workspace is an implementation restriction, not evidence that
  a GPU is unsuitable. CUDA prototypes live outside frozen source files.
- Batch independent atom-pair harmonic/projection work before drawing conclusions
  from Python-loop-heavy CUDA code. Keep original SVD/solve/check policies.
- A benchmark must distinguish a naive port, a pair-batched prototype, and production.
  No GPU prototype becomes a formal scan provider merely because a timing passes.
- Before timing, freeze panel, branches, solvents, dtype, reuse scope, repetitions,
  memory cap and numerical comparison tolerances. Include failures in the report.
- CPU workspace-copy parity target is bitwise. Cross-device/prototype energy
  tolerance for this small probe: absolute difference <=0.001 kcal/mol per state;
  **this comparison tolerance does not replace any original numerical gate**.
- Report discrete topology/allowed-gate differences explicitly; any such difference
  prevents automatic promotion. Do not discard discrepant states from the panel.
- CUDA timings synchronize at the measured boundaries. Report cold subprocess wall
  and warm full computation separately. Reset peak allocator counters for each job.
- Force actual fresh calculations in isolated outputs. Source/geometry caches are
  named explicitly; final-energy cache hits cannot count as computation speedup.
- A single CPU worker vs one GPU process is not a comparison with a 10-worker CPU
  pool. A PCM-only acceleration is not a complete pipeline acceleration.

## E1. Small-sample acceptance and delivery

1. Lock original behavior on a bounded panel before editing frozen behavior.
2. Validate energy/diagnostics, exact failure gates, closed caches and crash recovery.
3. Measure identical state populations and report initialization, input/cache,
   transfer, configuration, numerical computation, checks/materialization and report.
4. Record peak memory and worker busy/idle utilization. State what was NOT tested.
5. Preserve source/output SHA evidence. No repeated AI polling, repeated full audit,
   mass rerun, model fitting, or production switch without additional authorization.

Version reference: private experiment
`.omx/research/scan-engineering-gpu-20261020/`.
