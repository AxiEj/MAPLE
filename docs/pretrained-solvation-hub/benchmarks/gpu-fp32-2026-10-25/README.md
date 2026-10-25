# CUDA / FP32 numerical execution evidence (2026-10-25)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB. Runtime: Torch 2.12.0+cu130,
e3nn 0.4.4, Triton 3.7.0. Production parameters, operations, energy accumulation
and forces use FP32. Owned IEEE contexts disable TF32/autocast/fast math and
restore caller state. CPU64 is a separately declared numerical audit reference.

## Fresh-inference timing

One 9-atom ethanol geometry in ethanol solvent, 3 warmups and 10 distinct fresh
coordinate inputs, CUDA-synchronized medians. Startup/checkpoint/first compile
are excluded below and retained separately in receipts. The composition row
uses the actual gas-plus-correction production calculate path, not correction-only timing.

| Quantity | CUDA reference | CUDA optimized | GPU speedup |
|---|---:|---:|---:|
| AIMNet2 + AnySolv, combined scalar energy | 63.71 ms | 46.58 ms | 1.37x |
| ConSolv, native energy + forces | 28.02 ms | 19.79 ms | 1.42x |

These are small-molecule measurements, not large-system throughput promises.
ConSolv CPU32 measured 17.96 ms on this case, so GPU optimized is not claimed
to beat CPU here. Identical-input repeated calls are not cache-hit evidence.
Compilation can take tens of seconds or minutes on first use.

## Numerical qualification

- AnySolv: 18 solvent cases plus 6 exact vacuum zeros, CPU64/CPU32/CUDA32
  reference and optimized, exact source/checkpoint pins.
- ConSolv: 12 solution + 6 vacuum, 234 Cartesian FD components, 18 rigid
  transforms and two 10-step force-driven probes; all runtime denominators retained.
- The separately versioned v3 force audit compares FP32 analytic forces with
  CPU64 central FD (h=0.002 A, bound 0.002 eV/A), plus direct CPU64/CPU32/CUDA32
  analytic-force parity (0.0001 eV/A). Original same-FP32 absolute-energy FD
  failures are retained as resolution-limited diagnostics, never retroactive passes.
- Scoped context/dtype is owned by **both** AIMNet2 gas and AnySolv correction.
  Hostile caller TF32/autocast tests restore caller state; public CPU64
  composition matches the independently promoted gas+correction reference.
- Bounded numerical qualification is not strict scientific no-loss, chemical
  accuracy, thermodynamic Gibbs/free-energy, OPT, MD or Hessian admission.

Machine-readable artifacts, manifests, frozen public panel, unchanged v3 method,
SHA-bound implementation addendum and complete historical failure archive are
packaged under `maple/function/calculator/model_cards/evidence/gpu-fp32-2026-10-25/`.
No model weights are redistributed.

## Reproduction

Use `python -m maple.function.benchmarking.pretrained_solvation_gpu --help` for
raw label-free diagnostics and `python -m maple.function.benchmarking.solvation_gpu_evidence --help`
for the tested receipt validator/publisher. `measure_composite.py` records the
separate production gas-plus-correction sequence. `publish_evidence.py` is only
a compatibility CLI for the packaged publisher. Publishing rejects swapped
route/panel/device/dtype/mode/checkpoint/source identities, missing counts,
incorrect quantity archetypes, forged errors and non-improving optimized timings.

Public supported inputs are in `examples/pretrained_solvation/`; use the
project's compatible Torch environment and trusted legacy-constant loading
setting described in `CONSOLV_ANYSOLV.md`. GPU evidence is exact hardware/runtime
and source bound; another environment needs its own qualified evidence.
