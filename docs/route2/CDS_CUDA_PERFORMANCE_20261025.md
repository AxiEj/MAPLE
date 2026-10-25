# Route 2 CDS/DAREAL CUDA batching — 2026-10-25

## Result

The CUDA-FP32 CDS path now batches geometry and coordination switches instead of
performing a host round trip for each scalar. The original equations, coefficients,
radii, topology traversal, guard thresholds and continuous first/second AD are retained.
CPU and FP64 remain on the reference arithmetic. No fallback or persistent geometry
cache is introduced, and no AnySolv/Route-4 implementation was changed.

## Real public prediction timing

RTX 4060 Laptop; FP32/lmax5/Lebedev194; one CPU thread. Each call freshly computes
MACE source + ddPCM + stock SMD-CDS. Models are resident; load/init and final-result
caches are excluded. Three synchronized samples per variant are interleaved in one
process. Reference uses the frozen pre-CDS-batching implementation with the same
already-optimized ddPCM. These are small-panel descriptive results, not full throughput.

| Atoms | Before CDS batching (ms) | After (ms) | Speedup |
|---:|---:|---:|---:|
| 9 | 343.0 | 282.5 | 1.21x |
| 20 | 1792.3 | 640.8 | 2.80x |
| 35 | 3768.5 | 1313.7 | 2.87x |

## What was removed

Production changes are confined to `maple/solvation/surfaces/legacy_dareal.py`
and `maple/solvation/nonpolar/legacy_smd_cds.py`. The two new CUDA regression
files live under `tests/route2_vnext/`; documentation and source-bound receipts
were updated separately. No dependency was added.

- DAREAL prepares overlaps, cap tests, bounded intersection visibility and boundary
  angles on the GPU in stages. Bulk host snapshots carry branch metadata only.
- Python still executes the original ordered topology/polygon walk. Continuous point
  and angle tensors stay on-device and connected to coordinates/radii through AD.
- Tensions reuse one batched distance snapshot. Host-double cutoff selection and the
  sequential FP32 `(distance - reference) - width` expression are preserved exactly.
- An independent review caught an initial cutoff-rounding mismatch. It was repaired
  with a regression test before the final 300-geometry audit; the interrupted earlier
  audit is preserved privately and is not used as completion evidence.

In the instrumented 35-atom example, counted CDS host-read calls dropped from 20,850
to 976 (95.3% fewer). DAREAL itself changed from
2512.1 to 358.6 ms; tensions from
247.8 to 37.8 ms.
This is a single instrumented diagnostic, distinct from the repeated public timing table.

## Numerical and execution verification

- Full 300-geometry comparison: **230 matched successes + 70 unchanged rejections**.
  Zero status or topology mismatches; maximum stock-CDS energy difference
  `9.53674316e-06 kcal/mol`, within the frozen `1e-4` check.
  No experimental labels were used and this is not a fitted-A MAE evaluation.
- Original rejections remain: 56 four-sphere events, 12 other strict topology events,
  one cap-containment ambiguity and one unsupported-Si input (70 total).
- Old/new full-pipeline force comparisons cover 9/20/35 atoms; HVP comparisons cover
  9/20 atoms. The 35-atom HVP is not claimed.
- **241 targeted tests passed**, including chunk boundaries (>128 cap pairs),
  coordinate/radius gradients and second AD, cutoff rounding, and negative guards.
- Fresh public CPU/CUDA water/hexane E/F/Hessian/HVP: **4/4 passed**.
- Fresh complete solvent catalog component E/F: **358/358 passed**, with one geometry
  and cached model source per device; this is engineering coverage, not chemical accuracy.
- Actual CUDA `.inp` SP entry completed on the FP32/5/194 profile.
- Public current-source receipts were regenerated; their previous bytes were archived.
- Black, Pyflakes, AST and whitespace checks pass for the scoped production/new tests.
- Independent code review found no code/security defects (COMMENT); architecture is
  WATCH. LSP/static-type tooling is unavailable, so formal merge approval is not claimed.

## Scope and remaining limits

This is engineering optimization, not new scientific/release admission. No radii, fitted
A/B coefficients, MACE checkpoint, grid, dtype or guard threshold changed. No confirmation
labels were opened. Dense ddPCM resource limits and strict DAREAL rejections remain.
Small single structures are not guaranteed faster on GPU than CPU. No commit or push
was performed.

Use the existing `#device=cuda:0` pure-Torch v4 entry:

```bash
python -m maple.main examples/solvation/pure_torch/sp-cuda-default-fp32.inp result.out
```

[Aggregate evidence](evidence/route2-cds-cuda-fp32-performance-20261025.json).
Restricted geometry rows and logs remain under `.omx/research/cds-cuda-batch-20261024/`.
