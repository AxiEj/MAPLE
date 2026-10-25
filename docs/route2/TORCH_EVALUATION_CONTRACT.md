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

HVP + forces + energy ────────→ one shared forward
                               ├─ original central-force reverse
                               └─ HVP gradient + directional reverse
                                      │
                                      ▼
                            AnalyticHVPEvaluation
                            composed with the force record
                                      │
                                      ▼
                            validated legacy tuple (not cached)
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

- The first-derivative timing table above remains its historical operation scope.
  The subsequent direct-HVP tuple optimization is described below; the two speed
  factors must not be multiplied or compared as identical workloads.
- DAREAL topology/geometry restructuring is not mixed into this change.
- No TS/IRC/MD, new scientific/release admission, new dependency or Route-4 change.
- Static LSP/type-check tooling is unavailable in this runtime; targeted runtime,
  formatting, Pyflakes and AST checks do not substitute for a claimed full typecheck.

## Direct HVP tuple: one graph, original reverse arithmetic

`MACEPolarTorchPES.evaluate_hvp(atoms, direction)` returns an immutable
`AnalyticHVPEvaluation`. It composes the existing force record with the actual
computation-rounded direction, HVP array and an HVP-specific content digest.
Scientific identity fields are not duplicated in a second hierarchy.

The scalar forward is shared, but two distinct reverse-mode requests are not
conflated. Reusing the `create_graph=True` gradient as the central force produced
a reproducible FP32 difference (maximum `2.133846e-5 eV/Å`) during development.
That candidate was rejected without changing tolerances. The composed path
preserves the original central-force reverse with `create_graph=False`, then
performs the graph-building gradient and directional reverse for HVP:

| Operation | Full forwards | Reverse-mode sequence |
|---|---:|---|
| Energy + forces | 1 | `[False]` |
| Standalone `hessian_vector_product()` | 1 | `[True, False]` |
| Composed `get_hvp()` tuple | 1 (formerly 3) | `[False, True, False]` |

Here the booleans denote `create_graph`. The molecular Hessian is not materialized.
The existing second-order resource preflight still runs before graph construction.
A private transient tensor record joins the shared helper to its two return paths;
it is never retained on the PES or calculator.

Both the direct PES and facade snapshot the request direction as independent FP64
data before computation. The public return rechecks its original shape and values
after result materialization. The facade additionally verifies the recorded probe
against the request rounded to the declared computation dtype, the full nested
force identity and the HVP content digest. Post-construction corruption is rejected.

The standalone API retains native writable FP32/FP64 NumPy output. The facade
retains its flat `(Hn, forces, energy)` CPU Torch-FP64 tuple, with defensive copies
and explicit CPU placement even under a non-CPU Torch default device. HVP success
and failure leave ASE and Hessian caches unchanged. No fallback, new persistent
cache, numerical threshold, model parameter, Force/Hessian digest or scientific
admission is introduced.

Current HVP timing and complete validation evidence:
[HVP single-graph result](evidence/route2-hvp-single-graph-performance-20261025.json).

Warm interleaved RTX 4060 measurements on 2026-10-25, three repeats per path,
returning **HVP + forces + energy**:

| Atoms | Three forwards (ms) | One shared forward (ms) | Speedup |
|---:|---:|---:|---:|
| 9 | 1488.0 | 893.8 | 1.66x |
| 20 | 3897.9 | 2613.1 | 1.49x |
| 35 | 7294.8 | 4669.4 | 1.56x |

All paired HVP, force and energy differences were exactly zero. Final validation
passed 446 targeted tests, 4/4 public derivative cases, and 358/358 catalog
component cases. The public verbose SP entry still uses one forward. A separate
warm CPU-FP64 comparison was also exactly equal. An earlier cold-versus-warm
exact check differed by `8.88e-16`; that diagnostic remains preserved and is not
used as the matched warm acceptance result. Static LSP/type checking remains
unavailable. These small-panel timings are not a whole-dataset throughput claim.
