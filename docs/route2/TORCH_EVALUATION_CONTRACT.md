# Pure-Torch Route 2: evaluation and result ownership

## Responsibility boundaries

```text
ASE request                       Torch PES                     Scientific components
energy only ──────────────────→ scalar forward ──────────────→ MACE + ddPCM + CDS

forces (with central energy) ──→ one forward + one reverse ───→ same scalar/components
                                      │
                                      ▼
                            AnalyticForceEvaluation
                            detached, immutable, identity-bound
                                      │
                                      ▼
                            ASE validates and commits results
```

- Scientific components own the equations and numerical/topology checks.
- `MACEPolarTorchPES.evaluate_forces()` owns one graph for energy and forces.
- `AnalyticForceEvaluation` owns detached immutable values and their identity.
- `PureMACEPolarTorchCalculator` owns ASE result caching and invalidation.

No persistent Torch graph, geometry-plan cache, alternate engine or compatibility
fallback is introduced. The result object is transient; ASE retains only its
standard detached `energy`, `free_energy` and `forces` results.

## Force result identity

The result in `maple.solvation.derivatives.analytic` contains:

- energy, forces and component energies;
- geometry SHA and full model-input SHA (including canonical charge/multiplicity);
- component configuration, scalar, provider, profile and device identity;
- computation dtype and registered verification-evidence identity;
- a force-specific derivative-policy SHA and deterministic result-content SHA.

NumPy record storage is immutable FP64, matching the analytic-result convention.
This is a storage choice, not recovered numerical precision. The standalone
`get_forces()` API still returns a defensive FP32 array for V4 and FP64 for V3.
`AnalyticHessianEvaluation` and its historical digest are unchanged.

## Atomic ASE handoff

1. Clear request results and the pending cache key.
2. Validate the initial model input and configuration.
3. For a force request, call `evaluate_forces()` once.
4. Verify the record's model-input/configuration/provider/profile/scalar/device
   and geometry identity.
5. Revalidate the complete input/configuration key after evaluation.
6. Commit detached ASE results only after all checks pass.

A component failure, forged identity or input/configuration mutation produces no
partial result and no new cache key. Unchanged subsequent energy/force reads use
ASE's existing cache; geometry, charge, spin and configuration changes invalidate it.

Energy-only requests retain the scalar path and do not compute gradients. This
change does not speed up the previously reported **energy-only** latency by
combining it with another property: it removes duplication specifically within
**energy + force** requests, including verbose SP and force-driven optimization.

## Verification and timing

The paired benchmark alternates the frozen two-forward ASE facade and the current
one-forward facade over one resident real PES. Every request freshly evaluates
MACE + ddPCM + stock SMD-CDS. Loading is excluded, CUDA is synchronized, and the
full-forward counter is observed independently.

Current aggregate timings and evidence:
[single-evaluation result](evidence/route2-single-evaluation-performance-20261025.json).
Measured on 2026-10-25, RTX 4060 Laptop, for **energy + forces**:

| Atoms | Two-forward facade (ms) | One-forward facade (ms) | Speedup |
|---:|---:|---:|---:|
| 9 | 738.4 | 417.3 | 1.77x |
| 20 | 1601.9 | 933.5 | 1.72x |
| 35 | 2874.8 | 1720.6 | 1.67x |

All three paired energy and force differences were exactly zero. Final validation
passed 317 targeted tests, four CPU/CUDA derivative cases, and 358 catalog
component cases. A real verbose CUDA SP entry executed one full forward. A real
V3 CPU-FP64 water replay preserved energy/forces exactly. Previous source-bound
receipts were archived before current-source replacements were published.

These three fixed size examples do not establish full-dataset throughput or
universal GPU superiority to CPU. No scientific parameters, tolerances, checkpoint,
dtype, grid, radii or fitted coefficients changed.

The prior CDS 300-geometry accounting remains a **component-level** result:
230 successes and 70 original strict rejections. Its MACE/ddPCM/CDS implementation
hashes are unchanged by this execution-boundary work; it is not relabeled as a
new 300-record end-to-end force validation.

## Explicit exclusions

- HVP fusion is a separate follow-up. Its current facade still performs repeated
  central observations; this first-derivative change does not claim to fix it.
- DAREAL topology/geometry restructuring is not mixed into this change.
- No TS/IRC/MD, new scientific/release admission, new dependency or Route-4 change.
- Static LSP/type-check tooling is unavailable in this runtime; targeted runtime,
  formatting, Pyflakes and AST checks do not substitute for a claimed full typecheck.
