# TS-PRFO precision-preserving acceleration notes

Success gate: a speed result is valid only when the same model and same analytic
Hessian definition still reach PRFO normal termination and final TS validation
reports exactly one non-trivial imaginary frequency.

This branch ships **two distinct categories of acceleration** that should not
be conflated:

1. **Batch evaluator acceleration (always-on, physics-preserving).**
   `calculate_many` / `PathEvaluator` / `FDHessianEvaluator` and the per-backend
   `(B, N, 3)` / mol_idx / disconnected-graph batch paths.  These only change
   *how* energies and forces are dispatched to the same model; the NEB
   tangent, spring force, CINEB climbing image, PRFO equations, analytic
   Hessian definition, and convergence criteria are byte-for-byte unchanged.
   No user opt-in is required; auto chunk sizing degrades gracefully and
   PBC/solvent paths fail closed to sequential evaluation.

2. **Expert-gated algorithmic acceleration (off by default).**
   PRFO `expert_prfo_hessian_recalc>1` + Bofill quasi-Newton secant update, and the
   NEB-TS multi-candidate hand-off.  These change the TS optimization
   policy, not the batch evaluator.  `hessian_recalc=1` (the default) keeps
   the historical exact-Hessian-every-step behavior; the Bofill path is only
   taken when the user explicitly requests the expert PRFO strategy gate.
   Treat published references
   (ORCA `Recalc_Hess`, pysisyphus `hessian_recalc`, Bofill 1994) as the
   provenance, and require model-specific TS acceptance evidence before
   defaulting any of these on.

## Evidence and allowed acceleration classes

- NEB/CINEB core force definitions are kept unchanged. The path evaluator only
  changes how energies and true forces are collected. This matches standard
  NEB descriptions where each image uses the real force perpendicular to the
  tangent and the spring force parallel to the tangent, with CI-NEB removing the
  spring force on the climbing image and reversing the parallel true-force
  component. References:
  https://docs.onetep.org/nudged-elastic-band.html
  https://ajjackson.gitlab.io/ase/ase/neb.html
- PyTorch `torch.autograd.grad(..., is_grads_batched=True)` can batch a group of
  vector-Jacobian products through the vmap backend. This is an exact autograd
  Hessian assembly optimization when supported by the model's operators. Source:
  https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad.html
- PyTorch `torch.func.hessian` is the official high-level Hessian transform, but
  operator coverage can fail; unsupported TorchScript stacks must stay on the
  row-wise exact autograd loop.
- PRFO / eigenvector-following TS optimizers may use Hessian update strategies
  such as Bofill's mixture of Murtagh-Sargent/SR1 and Powell symmetric Broyden,
  but that is an algorithmic change and needs a separate literature-mapped
  design plus end-to-end TS validation before production use. geomeTRIC's TS
  notes also emphasize close TS guesses, final one-imaginary-mode validation,
  RS-P-RFO, conservative trust radii, and Bofill Hessian updates:
  https://geometric.readthedocs.io/en/latest/transition.html
- Classical TS optimizers do not require an exact Hessian at every PRFO step.
  ORCA exposes `Recalc_Hess` (for example every 5 cycles) and uses Bofill as
  the TS default update; pysisyphus exposes `hessian_recalc` and recommends
  Bofill for Hessian-based TS optimizers; the original Bofill update paper is
  "Updated Hessian matrix and the restricted step method for locating
  transition structures" (J. Comput. Chem. 1994, DOI:
  10.1002/jcc.540150102). MAPLE therefore adds the same pattern as an opt-in
  PRFO policy while keeping exact-Hessian-every-step as the default:
  https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/geomopt.html
  https://pysisyphus.readthedocs.io/en/latest/tsoptimization.html
- UMA true batch should follow FAIR-Chem's native pattern: convert ASE `Atoms`
  to `AtomicData`, combine with `atomicdata_list_to_batch`, run one predictor
  call, then split system-level energies and atom-level forces by the batch
  index. Source:
  https://fair-chem.github.io/core/common_tasks/batch_inference.html
- FAIR-Chem documents `turbo` inference as optimized for speed but requiring
  fixed atomic composition, so MAPLE should keep it opt-in rather than default:
  https://fair-chem.github.io/autoapi/core/calculate/ase_calculator/index.html
- AIMNet2's public package advertises energies, forces, stress, and Hessians,
  with GPU/compile support; MAPLE's safe route is exact autograd Hessian
  acceleration and a loop fallback, not finite-difference substitution:
  https://pypi.org/project/aimnet/
- MACE is PyTorch-based, documents an ASE calculator and analytical Hessian
  surface, and PyTorch Geometric's standard batching model is mini-batches of
  many small graphs. MAPLE's MACE true-batch force path therefore preserves the
  original no-PBC graph semantics while concatenating disconnected graphs, and
  keeps analytic Hessian assembly unchanged until a backend-native Hessian batch
  is proven equivalent. Upstream MACE also publishes MACE-OFF and MACE-OMOL
  foundation loaders, so MAPLE can use official raw checkpoints instead of
  inventing a local wrapper when MAPLE's HF bundle lacks that variant:
  https://mace-docs.readthedocs.io/en/latest/
  https://pytorch-geometric.readthedocs.io/en/1.7.2/index.html
  https://github.com/ACEsuit/mace
- TorchANI 2.0 reports that ANI performance bottlenecks are AEV, neighbor-list,
  backward, and double-backward kernels; their successful high-speed route is
  CUDA kernels for forward/backward/double-backward, not replacing exact
  Hessians with finite differences. MAPLE therefore keeps current ANI Hessians
  on exact autograd and accelerates safe outer workloads (path E/F batches and
  same-composition FD batches) until a backend-native CUDA AEV route is adopted:
  https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6890d92523be8e43d6b9bbba/original/torch-ani-2-0-an-extensible-high-performance-library-for-the-design-training-and-use-of-nn-i-ps.pdf


## Memory-safe batch-size control

Users can cap model-level batch chunks in input files without changing task
syntax:

```text
#model=uma(task=omol,batch_size=2)
#freq
```

or equivalently:

```text
#model=aimnet2
#batch_size=2
#ts(method=neb)
```

The parser stores this as `model_options["batch_size"]`; calculator
initialization attaches `batch_size`, `path_batch_size`, `fd_batch_size`, and
`hessian_batch_size` to the calculator. Evaluation surfaces that support
chunking consume it: PathEvaluator for NEB/CINEB path snapshots,
FDHessianEvaluator for numerical Hessians such as UMA frequency Hessians, exact
analytic-Hessian row-block VJPs for ANI/AIMNet2/MACE/MACEPol where applicable,
and the finite-difference HVP fallback. Tasks that do not support batching
ignore the attribute. The knob is therefore an OOM guard, not a request to
change physics or Hessian precision.

Do not specify both `#batch_size=...` and `#model(...batch_size=...)` with
different values. MAPLE now fails fast on conflicting values rather than
silently choosing one, because this knob is often used for OOM triage.

For analytic Hessians, `batch_size` means "maximum Hessian rows per exact
autograd VJP block". `batch_size=1` is the lowest-memory exact row loop; larger
values can improve speed when the backend supports batched VJPs, while keeping
the same Hessian definition.

The generic auto-sizer estimates one structure's memory cost in isolation, so
calculators whose batch path concatenates structures into a single dense
neighbor mask need an additional cap.  AIMNet2 in particular constructs
``nblist_dense_padded_multi`` over the concatenated coordinate stack, which is
``O((sum N_i)^2)`` in memory rather than ``sum O(N_i^2)``.
``_AutoBatchSizer._apply_math_cap`` therefore uses explicit calculator
capability metadata instead of backend-name strings. AIMNet2 auto-chunks are
capped at 8 for path E/F, FD Hessian, and HVP. UMA auto-chunks are also capped
conservatively at 8 because FAIR-Chem graph memory depends on atom count, edge
count, task head, and predictor settings. ANI/MACE keep their path-throughput
cap while leaving FD/HVP unconstrained unless a backend capability cap says
otherwise. Users wanting larger AIMNet2/UMA batches must set ``batch_size``
explicitly after local parity/OOM testing.

## Expert PRFO Hessian recalculation interval

By default MAPLE keeps the previous precision-first behavior:

```text
#ts(method=prfo)
```

means exact Hessian at every PRFO outer step, followed by the existing final
one-imaginary-mode validation. To trade fewer expensive Hessian builds for the
standard TS quasi-Newton update path used by mature optimizers, use the expert
PRFO strategy knob:

```text
#ts(method=prfo,expert_prfo_hessian_recalc=5,expert_prfo_hessian_update=bofill)
```

or for path methods that hand off to PRFO refinement:

```text
#ts(method=neb,refine=nebts,expert_prfo_hessian_recalc=5,expert_prfo_hessian_update=bofill)
#ts(method=string,refine=stringts,expert_prfo_hessian_recalc=5,expert_prfo_hessian_update=bofill)
```

Semantics:

- `hessian_recalc=1` (default): exact Hessian every PRFO step.
- `expert_prfo_hessian_recalc=N>1`: exact Hessian initially and every N accepted PRFO
  steps; accepted intermediate steps use a symmetric Bofill update from the
  accepted mass-weighted displacement and gradient change, matching the
  coordinate system used by MAPLE's PRFO step.
- Compatibility aliases `hessian_recalc=N>1` / `hessian_update=bofill` require
  `allow_prfo_hessian_update=true`; without that explicit gate PRFO raises.
- Any rejected PRFO trial forces an exact Hessian refresh on the next outer
  step, because rejection indicates that the local quadratic model was poor.
- Degenerate Bofill updates (for example negligible step or no new secant
  information) are skipped and also force the next step back to an exact
  Hessian.
- Final TS validation still computes the validation Hessian normally; the
  feature changes the optimization Hessian schedule, not the acceptance gate.

This knob is algorithmic acceleration, not a change of the model, force, or
Hessian definition. For publication-grade runs, record both the value and the
final frequency validation; if the path is fragile, lower the interval or use
the default exact-every-step policy. It is intended first for close NEBTS /
STRING-TS handoff geometries; direct PRFO from a loose guess can still require
`hessian_recalc=1` because the Cartesian/MW optimizer lacks the redundant
internal-coordinate safeguards used by some quantum-chemistry optimizers.

Similarly, PRFO refuses `hessian=numerical` when a backend advertises an
analytic Hessian. Expert debugging can set
`expert_prfo_allow_numerical_hessian=true`; this is a precision downgrade for
diagnostics, not batch acceleration, and final TS validation still prefers the
analytic Hessian when available.

## Current model decisions

- ANI: keep analytic Hessian. `batch_size` now caps exact autograd Hessian
  row-blocks; unsupported TorchScript batched-VJP operators fall back to the
  exact row loop rather than finite differences. ANI acceleration is applied to
  safe outer layers: same-composition `calculate_many` and NEB/CINEB path E/F
  batching.
- AIMNet2: use exact batched VJP analytic Hessian with `batch_size` row-block
  capping and exact row-loop fallback.
- MACE/MACEGeneral/MACEPol: keep analytic Hessians exact while allowing
  `batch_size` to cap autograd Hessian row-blocks where the generic path is
  used. Energy+forces calls now reuse one model forward where the traced model
  supports autograd, avoiding a redundant E-only pass without changing the
  Hessian or force definition. MACE-OFF23S/L and MACE-OMOL now use the official
  upstream raw MACE checkpoints when MAPLE-local TorchScript files are absent;
  graph-level charge/spin/head fields are populated to match upstream ASE
  calculator semantics.
- UMA: no analytic Hessian is exposed; FD batch acceleration is allowed only with
  strict final one-imaginary-mode validation and model-specific acceptance cases.

## Current validation snapshot

The `/tmp/...` paths below are historical local-run notes from 2026-05-26. They
are useful for debugging provenance but are **not** production evidence unless
the same checks are reproduced by CI artifacts. The release-blocking CI surface
is `.github/workflows/real-backend-smoke.yml`, which runs
`MAPLE_REAL_BACKEND_SMOKE=1 MAPLE_REAL_BACKEND_REQUIRED=1 python -m pytest -q
maple/function/calculator/test_real_backend_hessian_smoke.py` on a provisioned
`self-hosted` runner with model weights and backend caches.

Representative PRFO runs from existing NEBTS-quality small-molecule guesses pass
the strict gate (Normal Termination + exactly one imaginary frequency):

| backend | source geometry | wall evidence |
| --- | --- | --- |
| ANI-1xnr | `example/freq/mw/inp1_nebts_ts.xyz` | `/tmp/maple-ts-prfo-acceptance-ani-nebts`, MAPLE wall 3.812 s |
| AIMNet2 | `example/ts/neb/inp2_nebts_ts.xyz` | `/tmp/maple-ts-prfo-acceptance-aimnet2-nebts`, MAPLE wall 1.429 s |
| MACE-OFF23M | `example/ts/neb/inp2_nebts_ts.xyz` | `/tmp/maple-ts-prfo-acceptance-mace-nebts`, MAPLE wall 9.858 s |
| MACE-Polar-S | `example/ts/neb/inp2_nebts_ts.xyz` | `/tmp/maple-ts-prfo-acceptance-macepol-nebts`, MAPLE wall 22.312 s |
| UMA-S-1p1 | `example/ts/neb/inp2_nebts_ts.xyz` | `/tmp/maple-ts-prfo-acceptance-uma-nebts`, MAPLE wall 10.848 s |

Expanded direct-PRFO matrix from `example/ts/neb/inp2_nebts_ts.xyz`:
ANI-1xnr, AIMNet2, AIMNet2-NSE, MACE-OFF23M, MACE-Polar-S, and UMA-S-1p1
pass the strict gate in `/tmp/maple-ts-prfo-acceptance-direct-matrix-20260526`.
EGRET now runs after fixing checkpoint dtype handling (`float32`), but this
shared AIMNet2/MACE TS guess relaxes to a zero-imaginary EGRET minimum; EGRET
therefore needs a model-specific TS guess before it can be counted.

EGRET same-model NEB(refine=nebts) -> PRFO now passes the strict gate after the
dtype fix:
`/tmp/maple-ts-prfo-acceptance-egret-own-nebts`, prep wall 31.947 s, PRFO MAPLE
wall 8.435 s, one imaginary frequency (-712.26 cm^-1).

Same-model NEB(refine=nebts) -> PRFO now uses a single path energy+force batch
per NEB/CINEB geometry where the calculator supports `calculate_many`. This is
an evaluation-layer change only; NEB tangents, spring forces, CINEB climbing
force, PRFO, and convergence criteria are unchanged.

Current path-batch acceptance evidence:

Final evaluator pass on the current candidate-handoff code: `python -m pytest -q`, real-backend smoke, GPU path benchmark, and strict same-model NEBTS->PRFO acceptance all passed on 2026-05-26. Benchmark JSON: `/tmp/maple_path_batch_benchmark_current_20260526.json`; current core-family acceptance JSON: `/tmp/maple_core_current_20260526.json`.

All rows are same-model `NEB(refine=nebts) -> PRFO` runs from
`example/ts/neb/inp2.inp`, with strict success requiring Normal Termination and
exactly one non-trivial imaginary frequency.  The CINEB handoff uses the final
highest-energy internal image, not necessarily the image that was fixed as the
climbing image at CINEB start.  This is an evaluation/handoff correction only:
NEB/CINEB forces, tangents, springs, PRFO equations, and thresholds are
unchanged.

| backend | run root | NEBTS prep wall | PRFO MAPLE wall | lowest frequency |
| --- | --- | ---: | ---: | ---: |
| ANI-1xnr | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/ani` | 8.925 s | 4.771 s | -2355.69 cm^-1 |
| AIMNet2 | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/aimnet2` | 12.636 s | 1.298 s | -2786.99 cm^-1 |
| AIMNet2-NSE | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/aimnet2nse` | 7.510 s | 1.447 s | -3638.48 cm^-1 |
| MACE-OFF23M | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/mace` | 20.140 s | 5.756 s | -2925.33 cm^-1 |
| MACE-OMOL | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/maceomol` | 34.399 s | 10.156 s | -2390.71 cm^-1 |
| MACE-Polar-S | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/macepol` | 58.049 s | 18.486 s | -2378.07 cm^-1 |
| UMA-S-1p1 | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/uma` | 29.836 s | 12.164 s | -2394.77 cm^-1 |
| EGRET | `/tmp/maple-ts-prfo-acceptance-core-current-20260526195458/egret` | 14.468 s | 8.445 s | -712.27 cm^-1 |

ANI-2x failure mode and fix: before the final-highest handoff correction, the
CINEB run fixed image 4 as the climbing image, but the final path had image 5 as
the highest-energy internal image.  PRFO from stale image 4 converged to a
stationary point with zero non-trivial imaginary modes and was correctly
rejected.  PRFO from final image 5 passes the strict gate in
`/tmp/maple-ts-prfo-acceptance-ani2x-finalhei-20260526190712/ani2x`: NEBTS prep
39.088 s, PRFO MAPLE wall 3.792 s, lowest frequency -2013.89 cm^-1.

Additional installed checkpoint variants pass the same strict gate in the
current candidate-handoff run
`/tmp/maple-ts-prfo-acceptance-all-variants-candidates-20260526192745`; parsed
summary JSON: `/tmp/maple_all_variants_candidates_no_umam_20260526.json`.  The
user explicitly stopped the long UMA-M-1p1 rerun, so that variant is not counted
in this current-candidate table; a prior pre-candidate run had passed
UMA-M-1p1, but it remains outside this final-current evidence set.

| model variant | NEBTS MAPLE wall | PRFO MAPLE wall | lowest frequency |
| --- | ---: | ---: | ---: |
| ANI-1x | 9.787 s | 4.047 s | -2088.60 cm^-1 |
| ANI-1ccx | 9.140 s | 3.860 s | -1987.29 cm^-1 |
| ANI-2x | 48.718 s | 4.008 s | -840.78 cm^-1 |
| AIMNet2-NSE | 5.346 s | 1.275 s | -3638.48 cm^-1 |
| MACE-OFF23S | 9.853 s | 5.546 s | -2703.50 cm^-1 |
| MACE-OFF23L | 64.160 s | 12.160 s | -1070.13 cm^-1 |
| MACE-Polar-M | 96.047 s | 28.715 s | -2284.44 cm^-1 |
| MACE-Polar-L | 296.280 s | 34.576 s | -2359.92 cm^-1 |
| UMA-S-1p2 | 62.744 s | 22.140 s | -2532.92 cm^-1 |

ANI-2x candidate-handoff detail: the final path's top three barrier images all
relaxed to zero-imaginary stationary points under PRFO; the handoff fallback
then tried image 3 and accepted it only after strict PRFO validation reported
Normal Termination plus one imaginary mode.  This preserves NEB/CINEB/PRFO
equations and changes only the robust selection of the TS refinement starting
geometry.

Previously missing listed model entry now covered:

| model name | current availability evidence |
| --- | --- |
| MACE-OMOL | upstream MACE 0.3.16 exposes `mace_omol(..., return_raw_model=True)` and cached `/home/axie/.cache/mace/MACE-omol-0-extra-large-1024.model`; same-model NEBTS->PRFO passes strict gate |

MACE-OFF23S/L and MACE-OMOL note: MAPLE's Hugging Face model bundle currently
lists only `maceoff23m.pt`, but the upstream MACE package publishes official
small/large MACE-OFF and MACE-OMOL checkpoints. MAPLE now uses those upstream
raw models when no local TorchScript checkpoint is present, while preserving
the MAPLE TorchScript medium path for `maceoff23m` and any local traced
`maceomol.pt` if one is installed.

ANI path E/F micro-benchmark on 12 same-composition 7-atom images:
legacy sequential energy+forces median 0.376 s; path batch median 0.0210 s
after warmup; max energy difference 4.47e-8 Eh and max force difference
4.47e-8 Eh/Angstrom.

Expanded GPU path energy+force micro-benchmark on 12 same-composition 7-atom
images. "Legacy" is the old NEB evaluator shape: all image energies, then all
image forces. "Path batch" is one `calculate_many(..., ("energy", "forces"))`
snapshot. Model initialization is excluded; first-call batch compilation/warmup
outliers are excluded by the median:

| backend | legacy median | path-batch median | speedup | max energy diff | max force diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| ANI-1xnr | 0.547 s | 0.0258 s | 21.2x | 4.47e-8 Eh | 5.59e-8 Eh/Ang |
| AIMNet2 | 0.408 s | 0.0470 s | 8.7x | 5.09e-8 Eh | 7.26e-8 Eh/Ang |
| AIMNet2-NSE | 0.515 s | 0.0453 s | 11.4x | 2.75e-8 Eh | 1.32e-7 Eh/Ang |
| MACE-OFF23M | 0.732 s | 0.211 s | 3.5x | 0.0 Eh | 1.06e-16 Eh/Ang |
| MACE-OMOL | 1.762 s | 0.406 s | 4.3x | 0.0 Eh | 1.01e-16 Eh/Ang |
| MACE-Polar-S | 1.181 s | 0.721 s | 1.6x | 0.0 Eh | 3.35e-8 Eh/Ang |
| UMA-S-1p1 | 2.319 s | 0.224 s | 10.4x | 8.28e-9 Eh | 5.33e-8 Eh/Ang |
| EGRET | 0.333 s | 0.0384 s | 8.7x | 0.0 Eh | 6.61e-8 Eh/Ang |

This benchmark is reproducible with:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/path_batch_benchmark.py --device cuda:0 --reps 3 --warmups 2
```

ANI direct PRFO note: the stale repository example output
`example/ts/prfo/inp1.out` predates strict TS-mode validation.  A current
ANI-1xnr direct rerun of `example/ts/prfo/inp1.inp` passes strict validation in
`/tmp/maple-ts-prfo-acceptance-ani-direct-current-20260526181441/ani_direct`: 20
iterations, Normal Termination, one imaginary mode, lowest frequency
-339.99 cm^-1.  Direct PRFO starts remain geometry-sensitive; the production
acceptance gate is same-model NEBTS -> PRFO plus final one-imaginary-mode
validation.

The acceptance tool now has `--nebts-first` for full same-model
NEB(refine=nebts) -> PRFO checks. A short `--neb-max-iter 1` smoke verifies that
the tool fails closed when NEBTS preparation does not produce a TS geometry.
It also has `--neb-param key=value` so stability experiments can be recorded
without editing examples. The earlier AIMNet2 live NEBTS stall was reproduced
on the non-path-batched evaluator; the path-batched evaluator now passes the
strict gate without changing the NEB/CINEB/PRFO equations.
