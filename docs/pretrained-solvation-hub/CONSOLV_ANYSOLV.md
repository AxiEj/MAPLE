# ConSolv / AnySolv public usage

Both routes use MAPLE's calculator registry, shared Hartree result boundary,
and SP dispatcher. Install compatible PyTorch and model-specific optional
dependencies separately; importing generic MAPLE requires neither runtime.
No model download, training, device/dtype substitution, or compact fallback
occurs on these routes.

| Route | Public header | Quantity and admitted tasks |
| --- | --- | --- |
| ConSolv | `#model=consolv(native_solvent=ethanol)` | Native solution potential; CUDA FP32 by default, `#sp`, including `#sp(verbose=1)` gradients |
| AnySolv MoE | `#model=aimnet2` plus the directive below | Gas potential plus one geometry-level correction; CUDA FP32 by default, `#sp(verbose=0)` only |

```text
#model=aimnet2
#device=cuda:0
#sp(verbose=0)
#solv(implicit=water,method=anysolv,experimental=true,source_root=/home/axie/.cache/maple/anysolv/anysolv)

0 1
O 0 0 0
H 0.758602 0 0.504284
H -0.758602 0 0.504284
```

For ConSolv, replace the model header, use `#sp(verbose=1)` if desired, and
omit `#solv`. ConSolv rejects a second solvent correction and D4. AnySolv
requires an explicit pinned source checkout and a built-in vacuum CalcABC
base; UMA's separate interface is unsupported. Charged/open-shell correction
composition also requires a base that represents charge/multiplicity.

Examples in [`examples/pretrained_solvation`](../../examples/pretrained_solvation)
run with `python -m maple.main examples/pretrained_solvation/consolv_ethanol.inp`
or `python -m maple.main examples/pretrained_solvation/anysolv_water.inp`.
Set `source_root` to your verified checkout location. The local upstream Jd
constant uses a legacy Torch serialization API; this installation's public
replay uses `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for that hash-verified constant.
The MoE checkpoint itself is explicitly loaded with `weights_only=True`.

## Execution controls

Only these two routes default to `cuda:0` and `float32` when omitted. Other MAPLE
models keep their existing defaults. Missing/requested CUDA raises an error; no
CPU fallback or implicit device/dtype substitution occurs. Explicit `#device=cpu`
is supported, still FP32 unless `dtype=float64` is selected. FP64 is an audit
reference on CPU, not the default production precision. The pinned AIMNet2+AnySolv
composition propagates the selected dtype to both gas and correction components.

- ConSolv controls belong to `#model=consolv(native_solvent=ethanol,dtype=float32,inference=optimized)`.
- AnySolv controls belong to `#solv(...,dtype=float32,inference=optimized)`.
- `inference=reference` is default (eager CUDA, **not** a CPU selector).
- `inference=optimized` requires separately hash-bound numerical, fresh-inference
  speed, and GPU-memory evidence. Compilation can make the first call much slower.

AnySolv uses upstream merged block-GEMM/Triton plus owned Inductor execution.
Its compile scope is `upstream_hybrid`, not a claim of a fully compiled backbone.
ConSolv uses strict stateless FX export and fullgraph static Inductor programs.
ConSolv optimized owners freeze atom/edge counts, solvent, device and dtype;
changed signatures require a new calculator. Weights are explicit graph inputs,
not model constants retained by compiler caches. Both energy and AD variants
require actual compiled backend execution; eager fallback is an error.

CUDA FP32 uses scoped IEEE controls: no TF32, AMP, FP16/BF16 or fast-math.
The owned context restores caller settings on success and failure. GPU cards bind
exact checkpoints, implementation sources, runtime/hardware and numerical modes.
This is `bounded_numerical_fp32`, with `no_loss_parity_verified=false`: it is
not a claim of unchanged experimental/literature accuracy. The supported CUDA
AnySolv composition is the exact pinned AIMNet2 scalar base plus the correction,
not blanket GPU admission of standalone AIMNet2 or other gas calculators.

### Force-audit precision

ConSolv production weights, network operations, scalar reduction and gradients
remain FP32. At small displacements, finite differences of its returned FP32
absolute energies are resolution-limited; the original same-FP32 FD failures
are retained, not relabeled as passes. The separately versioned force audit uses
CPU-FP64 central differences as a higher-precision numerical reference and also
requires direct CPU64/CPU32/CUDA32 analytic-force parity. It does not enable
optimization, MD, or thermodynamic free-energy calculations.

Reproduce label-free numerical diagnostics (not a production calculator):

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
python -m maple.function.benchmarking.pretrained_solvation_gpu \
  --diagnostic --route consolv --device cuda:0 --dtype float32 \
  --inference optimized \
  --panel maple/function/calculator/model_cards/evidence/gpu-fp32-2026-10-25/frozen-panel-v2.json \
  --output /tmp/consolv-gpu-audit.json
```

The output path must not already exist. CPU64 is used only for the separate audit
run; original checkpoints and task gates are unchanged.

Default paths are relative to `maple/function/calculator/`:

| Model | Default path | SHA256 |
| --- | --- | --- |
| AnySolv MoE | `model/anysolv/model_moe.pt` | `77b7d883ac1a01fc13d78b8cf5530e5709af679e6fbdec53cb086ae0a90bd251` |
| ConSolv solution | `model/consolv/consolv-torch-v1.pt` | `534114cd0a45d97308eb28506ccfe731df143ae5fb0c6d611b6c6c0cb1656767` |
| ConSolv paired vacuum diagnostic | `model/consolv/consolv-vacuum-torch-v1.pt` | `d0369560cd34312604d8f4df3a002c7544c8867236ce50a6808c75794a5195f0` |

Only omission of `model_path` selects the default. Empty, null, or non-path
values are rejected. Optional overrides are `model_path=/absolute/file.pt`
inside `#model=consolv(...)` or the AnySolv `#solv(...)` directive. Hashes
remain mandatory; solution/vacuum identities cannot be interchanged.
Solvent, checkpoint, source, and device are constructor configuration; create
a new instance to change them. ASE positions and electronic-state metadata
continue to invalidate cached results.

AnySolv owns its resident model, loaded directly by the pinned official
`load_model` with explicit device/dtype/inference controls. A private weak cache shares that
unmerged reference resident across solvents using verified source/checkpoint digests and exact
runtime controls. Optimized MoE residents are owner-local and freeze atom count,
composition, charge, multiplicity, and solvent; a changed signature requires a
new calculator, never implicit remerging or an unbounded resident cache. The upstream pathname cache is not consumed or modified;
evaluation does not rehash the checkpoint. Composite calculators invalidate
charge/multiplicity changes and require the potential's separate capabilities.
Direct AnySolv evaluation rejects native, alchemical, and property-only contexts.
ConSolv also rejects an attached correction before inference or cache reuse.

The low-level loader is an internal diagnostic in `consolv._torch_model`, not
a package export or an admitted calculation API. It validates explicit CPU/CUDA
and FP32/FP64 controls before file access and keeps explicit `expected_state="vacuum"` for
paired-endpoint diagnostics. Use `ConSolvCalculator` for supported calculations.

ConSolv accepts one connected, nonperiodic, neutral closed-shell molecule
containing H/C/N/O/F/P/S/Cl/Br/I and the frozen 66 nonaqueous solvents.
Water, acetonitrile, and DMSO are outside this solvent set. AnySolv accepts
only its frozen 36 trained/validated solvent names. Its atomic-number capacity
1–99, charge capacity −100–100, and multiplicity capacity 1–100 describe input
embeddings, not a validated chemistry domain. A separated two-water scalar
probe ([frozen receipt](benchmarks/anysolv-multifragment-2026-10-24.json))
establishes mechanical multifragment execution only; the validated
multifragment domain remains unknown. Neither route supports PBC.
OPT/scan/TS/IRC/FREQ/MD remain disabled; GPU acceleration does not open derivative tasks for AnySolv.

AnySolv returns `quantity_kind=geometry_level_energy_correction` with gas,
correction, and combined energies in Hartree. It exposes no forces and no
`delta_g_solv_hartree`. ConSolv returns native solution energy/gradients.
Neither fixed-geometry result is a sampled or harmonic Gibbs free energy.
ASE's `free_energy` property is an energy alias, not a thermochemical claim.

Source identities are [AnySolv 1778e1f](https://github.com/Ant-on-knee/anysolv/tree/1778e1f18384e5f59de1475182765733a7401e39)
and [ConSolv 06801dd5](https://github.com/tummfm/ConSolv/tree/06801dd5f53182ed86c31eee1fa03cc011c05b5a).
ConSolv uses the independently implemented MACE_CONSOL graph with 60 source
tensors / 768338 parameters; it requires no production JAX runtime. Conversion
receipts in shared model storage retain the reconstructed audit-reference
identity, rather than claiming unmodified official-runtime parity. ConSolv
source has no root license grant; model/data are CC-BY under Zenodo 22956180.
AnySolv code is MIT and the authorized MoE checkpoint uses FAIR Chemistry
License v1. Weights are external assets and are not packaged.

The [frozen MoE force-rejection evidence](benchmarks/anysolv-moe-mechanics-2026-10-23/README.md)
is retained byte-for-byte. Real-asset tests are explicit:
`MAPLE_TEST_ANYSOLV_MOE=1 MAPLE_RUN_CONSOLV_REAL_ASSET_TESTS=1 python -m pytest -q tests/solvation/test_anysolv_backend.py tests/solvation/test_consolv_calculator.py tests/solvation/test_consolv_torch_model.py`.
