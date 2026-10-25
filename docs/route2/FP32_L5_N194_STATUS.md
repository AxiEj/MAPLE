# FP32 / lmax 5 / Lebedev 194 — integrated experimental default

The pure Torch MACE-POLAR workflow now selects FP32 / lmax 5 / Lebedev 194
when profile is omitted. Omitting response selects frozen response. Other
providers retain their existing defaults and explicit historical v3 profiles
retain FP64 / 15 / 1202. No private research directory is needed at runtime.

## Precision and numerical policy

- MACE parameters, inputs and returned energy/source: FP32; unmodified upstream
  explicit FP64 reduction operations are retained. This is not an all-op FP32 claim.
- ddPCM, analytic legacy SMD CDS, DAREAL and coordinate differentiation: FP32.
- Residual default: 100 eps32 = 1.1920928955078125e-5 for the existing scaled
  residual metric; not a normwise backward-error theorem.
- FP32 topology uses stable complementary quintics and ratio-derived structural
  branches. Proven C2/one-sided boundaries warn and continue; unresolved
  multi-interior crossings still reject. FP32 uses its explicit dtype-aware
  topology margins; no additional ad-hoc tolerance widening, radius jitter,
  output clipping or fallback is introduced.
- Charge auditing uses a scale-aware summation-roundoff allowance without
  modifying or renormalizing predicted charges.
- Hessian antisymmetry budget: max(1e-4, 100 eps32 max|H|), eV/Angstrom^2.
  The raw Hessian is not symmetrized. This is an engineering consistency budget,
  not an accuracy guarantee. FP64's former absolute budget is preserved.
- Analytic result objects distinguish FP32 computation from normalized FP64
  array storage; storage conversion does not recover precision.

## Current verification

- Single-evaluation execution boundary (2026-10-25): force-capable ASE requests
  use one MACE+ddPCM+CDS forward, down from two, and atomically receive energy
  and forces from one immutable first-derivative result. Energy-only execution,
  standalone force dtypes, historical Hessian results and scientific component
  source bytes are preserved. See [evaluation ownership](TORCH_EVALUATION_CONTRACT.md)
  and its source-bound aggregate evidence for current timing and exact test counts.
- CDS/DAREAL CUDA batching (2026-10-25): 300 fixed development geometries were
  compared against the frozen previous CUDA implementation. All outcomes matched:
  230 successful pairs and 70 unchanged strict-SMD rejections, zero topology
  differences; maximum CDS-energy difference was `9.536743e-6 kcal/mol`.
  This is stock-CDS numerical parity, not a fitted-A accuracy evaluation.
  Fresh public water/hexane energy/force/Hessian/HVP replay passed 4/4 cases;
  full-catalog CPU/CUDA component replay passed 358/358. Previous evidence bytes
  were archived before publishing the new source-bound receipts. No thresholds,
  fit coefficients, radii, confirmation access or derivative scope were changed.
- CUDA operator update (2026-10-24): bounded pair batching and exact QR-based
  full-spectrum SVD checks are active only on CUDA FP32. Fresh replay passed
  all four CPU/CUDA water/hexane derivative cases and all 358 CPU/CUDA catalog
  component cases. Previous receipt bytes were archived under `evidence/history/`
  before the current-source results replaced them. CPU/FP64 execution paths and
  numerical thresholds were retained. The 300 already-open fixed-A water scalar
  records also completed on CUDA with no topology/gate differences; this audit
  reuses the frozen MACE source and fitted CDS, not end-to-end prediction timing.
- Commit-time verification (2026-10-22): 577 changed/new public regression tests
  passed, 1 skipped. Genuine current-source replay passed all four CPU/CUDA
  water/hexane derivative cases and all 358 CPU/CUDA full-catalog component cases.
  Previous evidence bytes were archived before publication of the new receipts.
- Initial FP32 implementation/regression suite: 319 passed, 1 skipped;
  multisolvency regression and coverage are reported in `MULTISOLVENT_V4_STATUS.md`.
- Actual public `.inp`: default CPU SP, default CUDA SP, CPU LBFGS optimization
  and CUDA MW-FREQ completed. The small water optimization converged at iteration 1.
- Water solute in both water and hexane: CPU/CUDA strict energy, forces, full
  analytic Hessian and analytic HVP replayed without numerical-force fallback.
- New portable source-bound engineering evidence:
  `evidence/route2-pure-macepolar-torch-fp32-v4-engineering-tests.json`.
- Real run logs: `.omx/research/fp32-default-v4-20261017/` (local only).
- Multisolvent extension: 179 named published SMD parameter sets are available.
  After structural topology grading, each CPU/CUDA sweep has 179 strict E/F
  successes and zero skips; the former 10 topology rejections now carry
  explicit warnings. Prior artifacts remain preserved in evidence/history/.
- The original four-case v4 artifact was archived byte-for-byte under
  `evidence/history/` before genuine current-source replay replaced it.
- Static validation: AST parse, targeted pyflakes and `git diff --check` pass.

Reproduce the bounded evidence in a compatible configured environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m tools.route2_release.run_pure_mace_polar_fp32_smoke --output new-evidence.json
```

The runner refuses to overwrite an existing artifact. It tests both CPU and
CUDA and retains failed cases rather than reporting only survivors.

## Remaining boundaries

This is engineering integration, not new whole-dataset scientific admission.
Supported experimental scope is closed-shell SP/OPT/MW-FREQ, including explicit
integer molecular charges on V4. Charge conservation targets the input charge,
not zero; historical V3 and other providers remain neutral-only. Ionic execution
does not establish ionic-solvation accuracy. TS/IRC/MD remain closed.
V4 now accepts all 179 named SMD parameterizations; historical
v3 remains water/hexane only. See `SMD_SOLVENT_CATALOG.md` and
`MULTISOLVENT_V4_STATUS.md` for parameter coverage and explicit topology failures. Public default example:
`examples/solvation/pure_torch/sp-cpu-default-fp32.inp`.

The historical v3 source-bound evidence guard is still stale: 1 failed, 3 passed.
`torch_ddpcm.py` already mismatched that frozen manifest at pre-edit HEAD;
this integration changes additional shared source files. The old manifests
were not rewritten to masquerade as validation of new code. Shared-file changes
also change native-FP64 component identity; old evidence must not be attributed
to this checkout. Existing FP64 numerical regressions pass; source-bound
qualification and numerical parity are distinct claims.
