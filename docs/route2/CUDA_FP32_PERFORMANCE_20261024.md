# Route 2 CUDA FP32 acceleration — 2026-10-24

This is the retained ddPCM-stage result. The subsequent CDS-stage optimization
and current timings are in [CDS CUDA performance](CDS_CUDA_PERFORMANCE_20261025.md).

Only the current Route-2 worktree changed. No AnySolv/Route-4 implementation is included.

## Change

CUDA FP32 now uses bounded batched atom-pair assembly and exact QR-based full-spectrum
SVD diagnostics (`gesvd`). CPU/FP64 operators, checkpoint, source, A/B radii, CDS coefficients,
two direct solves and numerical thresholds remain unchanged. No TF32, AMP or fallback.

## Warm public CUDA calls

Fresh MACE + ddPCM + stock SMD-CDS per call on RTX 4060 Laptop. Each row alternates
the retained historical controls and current controls in one resident process, three
synchronized samples per path. Model loading is excluded. These are not fitted-A timings.

| Atoms | Historical GPU (s) | Current GPU (s) | Speedup |
|---:|---:|---:|---:|
| 9 | 0.7132 | 0.3878 | 1.84x |
| 20 | 3.5961 | 1.8989 | 1.89x |
| 35 | 8.7062 | 3.5034 | 2.49x |

This is a small descriptive panel, not proof of full-dataset throughput or universal GPU
superiority to CPU. Initial separate CPU/GPU measurements still found the 20/35-atom CPU
calls faster. Small kernels, synchronization and the legacy CDS path remain costs.

The original six-geometry public panel completed 4/6 both before and after; the same
62-atom dense-resource rejection and 14-atom DAREAL topology rejection remain. Neither
guard was relaxed. The successful subset is not reported as full-panel admission.

## Numerical preservation

- Fixed A: 300/300 already-open development water scalar states replayed on CUDA.
- Maximum CPU/CUDA polar difference: 1.16833841e-05 kcal/mol, within the retained 1e-4 check.
- Topology differences: 0; numerical-gate differences: 0.
- Fixed-A MAE: CPU 1.0280555499; CUDA 1.0280554757 kcal/mol.
- Frozen source and fitted CDS were reused. This is component parity, not complete inference timing.
- Four original derivative-topology-unqualified scalar states remain unqualified.

## Verification and scope

- 143 targeted regression tests passed, including real CUDA and unchanged FP64 baselines.
- Four fresh public CPU/CUDA water/hexane E/F/Hessian/HVP cases passed.
- All 358 CPU/CUDA catalog component E/F cases passed; this is engineering coverage only.
- The real CUDA `.inp` SP entry passed. Unavailable devices fail; no automatic CPU fallback.
- Previous public evidence bytes were archived before genuine current-source replay was published.
- Production/new-test Black, Pyflakes, AST and whitespace checks pass. The legacy precision
  test file retains pre-existing Black formatting differences (also present at HEAD).
- No confirmation access, new fit, scientific/release admission, Git commit or push.
- Independent code review found no code/security defects; its recommendation is COMMENT
  because LSP/static-type tooling is unavailable. Architecture status is WATCH for the
  limited performance/scaling scope, with no architectural blocker. No merge approval is claimed.

## Run

```bash
python -m maple.main examples/solvation/pure_torch/sp-cuda-default-fp32.inp result.out
```

Use `#device=cuda:0` with the default pure-Torch v4 profile. CPU/FP64 remain explicit alternatives.

[Aggregate evidence](evidence/route2-cuda-fp32-performance-20261024.json).
Restricted row-level inputs/results and timing logs remain under `.omx/research/route2-cuda-20261024/`.
