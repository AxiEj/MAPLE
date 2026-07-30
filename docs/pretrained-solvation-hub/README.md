# Pretrained solvation-aware model hub

This directory records the scientific and deployment contracts for Route 4.
Route 4 is a registry of pretrained potentials, solvent backends, and
free-energy protocol adapters.  It is not a claim that an MLIP alone computes
a binding free energy.

## Free-energy boundary

A binding free-energy calculation requires a potential, solvent environment,
thermodynamic path, sampling, restraints/standard-state corrections, and an
estimator.  MAPLE therefore distinguishes:

1. ordinary potential-energy surfaces;
2. native solution-phase potentials;
3. additive solvent PMFs;
4. alchemical free-energy protocol backends; and
5. scalar property predictors.

Only the first four may participate in calculator or thermodynamic-protocol
execution, and each task is gated by a machine-readable capability card.
Scalar predictors may have dedicated, auditable property adapters, but are not
ASE calculators.

ASE `results["free_energy"]` remains an energy-like calculator result.  It is
not thermochemical Gibbs energy, absolute solvation free energy, or binding
free energy.  Thermodynamic results must be emitted by a dedicated protocol
with sampling, estimator, standard-state, and uncertainty metadata.

## Experimental-accuracy evidence boundary

An experimental accuracy metric is forbidden unless its immutable panel
contains at least 10 records spanning at least 10 distinct, predeclared
functional groups.  Every record must bind its functional-group assignment,
the canonical solute graph, the SHA256 and locator of the exact molecular
payload used for prediction, experimental provenance, experimental value,
prediction, signed error, and absolute error.  For SDF, MOL2, or coordinate
inputs, the hash must cover the exact geometry-bearing record rather than a
separate free-form SMILES.  The featurization boundary must create a
`MolecularInputReceipt` by reading the actual file or ZIP member; the receipt
derives both the payload SHA256 and canonical graph from those same bytes and
cannot be constructed from result-table metadata alone.  The model input must
come through the runner-owned `VerifiedMolecularInput.read()` handle.
`run_accuracy_panel()` accepts neither labels, arbitrary callables, nor a
caller-assembled prediction table.  Record-level experimental values, units,
provenance, source artifact SHA256, and source locators are derived from an
actual JSON artifact by `BenchmarkExperimentalReference.from_json_file()` and
frozen in the panel identity; free metadata construction is rejected.  The
runner re-reads that artifact before prediction.  The identity also selects an
adapter registration from a source-controlled registry; no production adapter
is registered until its model-specific featurization path is reviewed.  The
runner computes the adapter fingerprint itself from live `predict()` code,
canonical JSON configuration, and reverified adapter-code, featurizer-code,
checkpoint, and dependency-lock artifacts.  Prediction runs in a fresh
bubblewrap-confined Python interpreter that re-verifies those artifacts and
imports the adapter from disk rather than reusing parent-process code or
monkeypatches.  The child receives only verified molecular-input bytes plus a
label-free `AccuracyPredictionContext`; no input source path, receipt,
benchmark identity, experimental artifact, or value enters the sandbox.  Its
mount namespace exposes the Python runtime, MAPLE implementation tree, and
exact declared adapter artifacts, but not benchmark-data directories; absence
of bubblewrap is fail-closed for formal accuracy.  The parent collects and
converts every prediction before re-reading experimental references for
scoring.  The worker rejects an adapter that does not consume the handle,
recomputes the implementation fingerprint after execution, and returns only
its sealed prediction transcript and fingerprint.
Paired comparison re-executes both registered adapters rather than trusting a
caller-provided result object.  The receipt's graph, payload hash, locator, and
format must equal the frozen record-input manifest.
The complete panel must succeed, and the summary must report maximum absolute
error as well as MAE/RMSE.  Functional-group assignments, exact record inputs,
and the taxonomy are part of the benchmark fingerprint; they may not be added
or changed after inspecting model outputs.

The allowed structural families are frozen in
`functional-group-taxonomy-v1.json` (SHA256
`46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36`;
semantic fingerprint
`4928108edfe8529a9302484cca01d60860bb5bc53458db668c79994c91f65db2`).
The artifact binds RDKit SMARTS definitions, source hashes, and a deterministic
primary-family precedence.  Coverage counts distinct predeclared **primary
functional-group families** only.  Chemical classes, aromaticity, ring
systems, charge classes, and fine-grained subtypes cannot be counted as extra
functional groups.  Each assignment also binds the source structure identifier
and assignment evidence into the panel fingerprint, and must match the same
record's canonical structure in the hash-bound molecular-input manifest.

A panel below either minimum, or one whose functional-group coverage was not
predeclared, is runtime/interface smoke only.  It may report execution
coverage and failures, but it may not emit accuracy metrics, rank models,
support a literature-accuracy claim, unlock model admission, or be
generalized to broader chemistry.  Ten groups is a minimum evidence floor,
not proof of universal chemical coverage.

## No-loss GPU acceleration boundary

GPU execution is an acceleration target, not a license to change the
scientific calculation.  A GPU path remains disabled until it uses the same
checkpoint, equations, cutoffs, solvent state, lambda schedule, estimator,
convergence criteria, and accepted scalar precision as its reference path.
MAPLE must not silently enable float16, bfloat16, TF32, fast-math,
reduced-precision matrix multiplication, looser neighbor lists, shorter
sampling, or relaxed convergence.

Enablement requires a frozen paired audit of energy, forces, virial, and any
lambda derivatives on identical configurations, followed by a paired
free-energy/uncertainty audit.  Any experimental-accuracy or
identity-matched literature comparison used for GPU admission must itself
pass the 10-record/10-functional-group gate above and contain per-record
errors.  A small numerical CPU/GPU difference is diagnostic and does not mean
that the MLIP lacks GPU support.  A smaller panel can never unlock an
accelerator.  GPU availability or speed alone is not evidence of chemical
parity.  The GPU route must not worsen coverage, maximum absolute error, or
identity-matched literature-panel metrics; otherwise MAPLE keeps the validated
reference route and records accelerator admission as pending or rejected.

This boundary is enforced in the registered MAPLE calculator path and in the
direct Route 4 adapters.  A CUDA, MPS, XPU, HIP, ROCm, integer, `auto`, or
generic GPU request is rejected before checkpoint resolution when there is no
model card or when its no-loss evidence is incomplete.  A verified card must
name every task exposed by the model capabilities and every inference mode;
partial task admission is deliberately forbidden because a direct calculator
can expose several properties from one instance.  In particular, any public
force API is mechanically reusable by an external MD driver, so MD/trajectory
parity remains mandatory even when the product model card forbids MD.
Product-level forbidden tasks do not waive accelerator parity coverage.  The
card must bind those paths to a repository-local frozen parity artifact by its
full SHA256.  The file must still exist and match that digest at execution
time.  No current Route 4 model card has passed this gate, so none is silently
accelerated from hardware availability alone.

## Matched-QM end-to-end performance boundary

Accuracy is evaluated first.  A candidate is timed only after both it and a
pinned QM baseline pass the same formal maximum-error limit on the same record
IDs, molecular-input hashes, and experimental references.  The timed
candidate and QM runs must use one identical explicit `float32` or `float64`
policy, and each timed policy must match the corresponding accuracy-accepted
policy; reduced precision cannot buy admission.

The hard observable is the median wall time of at least three complete task
repeats, including input preparation, geometry or sampling, model/QM
execution, analysis, and artifact writing:

```text
speedup = median(QM full-task seconds) / median(candidate full-task seconds)
```

`speedup` must be strictly greater than one.  A tie or a candidate slower than
QM has no production advantage and is rejected.  Cold-start and warm-state
timings are retained as diagnostics only.
`evaluate_matched_qm_performance()` is a rejection-capable policy evaluator:
it rejects caller-supplied summary mappings, reruns both registered adapters
from source-verifying `BenchmarkIdentity` objects, and refuses mismatched task
identities, input hashes, experimental values, functional-group coverage,
scopes, a precision policy that differs between candidate and QM, insufficient
full-task repeats, or either method's failed accuracy gate.  Caller-supplied
seconds, scope labels, precision strings, and hardware text can never approve
production performance.
A positive policy result remains blocked until a model-specific trusted runner
emits hash-bound receipts for the exact candidate and QM commands, methods,
checkpoints, record order/input hashes, environment, hardware, precision, and
every full-task component.  No current Route 4 candidate has such a receipt.

The frozen artifact is not an opaque approval token.  Its schema binds the
exact model version and every checkpoint SHA256, an immutable comparison-panel
ID, repository-local manifest path, and SHA256, the complete
task/inference-mode matrix, matching concrete `float32` or `float64`
reference/accelerator dtypes, and the required matrix multiplication
precision.  MAPLE loads that manifest at execution and recomputes its digest;
matching strings in the card and parity JSON are insufficient.  The manifest
must also parse as the admitted JSON schema and bind the exact model,
checkpoints, task/mode matrix, non-empty unique record identities, panel IDs,
configuration/reference hashes, and per-observable comparison counts.  Those
counts must equal the parity results.  Every manifest record also binds the
raw reference and accelerator output digests for every required observable;
MAPLE recomputes those digests, per-element comparisons, counts, maxima, and
pass verdicts instead of trusting declared aggregates.  The version-controlled
model card—not the evidence file—predeclares each observable's unit,
comparison meaning, and maximum threshold; the artifact may be stricter but
cannot choose a looser tolerance.
Coverage must be exactly equal, while every individual identity-matched case,
the maximum error, and the literature-panel metrics have a zero-degradation
threshold.  Energy, force, Hessian, trajectory, virial, lambda-derivative, and
free-energy records are required whenever the model exposes them.

At execution time the adapter must also supply its current hardware/runtime
fingerprint: backend, package versions, effective dtype,
matrix-multiplication precision, and
explicit false values for float16, bfloat16, TF32, autocast, fast-math,
reduced matmul, shortened sampling, and relaxed convergence.  A copied result
from another accelerator or a runtime that cannot report those live controls
cannot unlock the route.  Current adapters intentionally remain non-runnable
on accelerators until both a frozen parity panel and that live fingerprint
hook exist; this is pending admission, not a speed claim.

## Incremental architecture

The existing `register_calculator`/`SetCalculator`/`CalcABC` path remains the
potential registry and composition boundary.  Route 4 adds:

- explicit model capabilities and immutable provenance cards;
- a formal solvent-backend protocol over the existing GB/PB/SMD dispatcher;
- fail-closed model/task/solvent combination validation;
- a lightweight topology provider; and
- dedicated protocol adapters for models such as LSNN.

Route 4 does not duplicate the Route 3 `#solvfe` sampling engine and does not add a public `#bindfe` task.

## Initial engineering integration lanes

This milestone implements architecture, pinned provenance, adapters, and
fail-closed unit contracts.  It does **not** complete the four scientific
verticals: no real-checkpoint NVE certification, FreeSolv/MNSol accuracy run,
SAMPL host--guest calculation, or protein--ligand free-energy calculation is
claimed here.

| Lane | Backend | Engineering status |
| --- | --- | --- |
| Conservative PES | MACE-OFF24(M), AceFF 2.0 | Adapter SP/force/numerical-Hessian contracts are unit-tested; live-checkpoint MD/NVE certification remains pending |
| Explicit-system soft-core PES | MACE-OFF23-SC | The exact public checkpoint runs through the official MACE float64 runtime on CPU for energy/forces/PBC/multifragment systems. This is the full-coupling PES endpoint only; the separate OpenMM-ML/replica-exchange/MBAR protocol remains unavailable |
| Native solution PES | AIMNet2-CPCMS v2 | Neutral-singlet mechanics probe only; MD is disabled until real-checkpoint stability evidence and solvent/license/reference metadata close |
| Additive/reference solvent PMF | OpenFF + GNNIS; AniSolv compact | GNNIS remains a sealed reference Hamiltonian. AniSolv compact is a pinned 21-solvent scalar single-point correction whose upstream force path is rejected; both forbid absolute solvation free energy |
| Alchemical solvation | LSNN-v1 | Scaffold-only water TI/MBAR protocol contract; default execution remains disabled until the model domain and runtime are audited |
| Direct scalar property prediction | C3Net; CIGIN; SolProp-mix Exp (blocked); AtomicESE (audit only) | C3Net/CIGIN run through separate property-only adapters. SolProp-mix is an exact pinned pure-and-mixture dGsolv ensemble whose original upstream runtime has unavailable dependencies. AtomicESE is a dedicated scalar multi-organic-solvent candidate represented only by a packaged-binary release audit; none exposes a PES, forces, ASE calculator, or acceptance-panel claim |

### Direct-native route versus protocol controls

`aimnet2-cpcms-v2` is the Route 4 direct-native implicit-solvent integration:
it evaluates a single solution-phase energy surface from coordinates, elements,
and molecular charge.  It must not receive a second GB/PCM/GNNIS correction.

LSNN-v1 is **not** an interchangeable native solution-phase MLIP.  Its
alchemical workflow has the composed form
`U_base(R) + W_LSNN(R, lambda)` and therefore requires a separately chosen
vacuum/base Hamiltonian.  Any LSNN run belongs to the alchemical protocol
control lane; it must not be used as evidence that the direct-native AIMNet2
adapter has been validated, and it is never selected as the default Route 4
calculator.

GNNIS is an additive solvent PMF physically, but the executable
`gnnis-reference` adapter seals it to the upstream OpenFF-2.0.0 vacuum
Hamiltonian.  Its capability card therefore treats the resulting composition
as a native/reference potential so a second solvent term cannot be added.
The pinned upstream helper auto-selects CUDA whenever PyTorch reports it
visible.  Until that behavior has a frozen no-loss parity artifact, MAPLE
rejects accelerator requests and permits the original helper only when CUDA is
not visible and the resulting OpenMM context proves a CPU or Reference
platform.

AniSolv compact is an additive geometry-level `E_solv - E_gas` correction. It
is executable only with its pinned public compact checkpoint, its explicit
21-solvent whitelist, explicit neutral-singlet metadata, and a non-periodic
base potential. Its exact zero vacuum output anchors the paired gas/solution
potential; it does **not** make ASE `free_energy` a thermodynamic quantity or
certify absolute solvation free energy, MD, charged/open-shell chemistry, or
descriptor-based solvent extrapolation.

The ordinary MAPLE energy entry point is deliberately sealed rather than a
generic modifier: `#model=anisolv-uma(base_model_path=/absolute/path/to/verified-uma-s-1p2.pt,solvation_model_path=/absolute/path/to/model1_compact.pt)`
with `#solv(implicit=<one of the 21 names>,method=anisolv,experimental=true)`.
The upstream UMA release is access-controlled, so MAPLE does not download or
bypass it: `base_model_path` must be an already-authorized local compatibility
checkpoint with the pinned SHA256.  The command also checks the public AniSolv
compact SHA256 before loading, rejects custom base substitutions/D4/PBC/charged
or open-shell inputs, and permits scalar `#sp(verbose=0)` only.  Exact-upstream
AniSolv forces are not rotation-covariant at Euler-chart pole orientations, so
MAPLE rejects gradients, optimization, scans, TS/IRC, frequencies/Hessians,
MD, and absolute free-energy tasks.

MACE-OFF23-SC is a distinct explicit-system lane.  MAPLE now binds the exact
public `MACE-OFF23-SC_swa.model` checkpoint and exposes its unchanged
full-coupling float64 MACE energy/force surface for neutral-singlet,
supported-element systems, including fully periodic multifragment boxes.
This does not reproduce `mace-md`: the adapter exposes no
`lambda_interpolate`, replica exchange, MBAR estimator, uncertainty, or
absolute solvation-free-energy result.  It must not be substituted for the
unreleased MACE-OFF24-SC identity behind the later paper tables.

The official 192-atom water box was also evaluated on the same checkpoint and
dtype on CPU and CUDA with reduced-precision paths disabled.  The frozen
negative artifact
`benchmarks/mace-off23-sc-gpu-no-loss-audit-2026-07-30.json`
records a nonzero `2.6193447411060333e-10 eV` energy difference and a
`1.0769163338864018e-14 eV/angstrom` maximum force-component difference.
It also confirms that the official CUDA runtime executes and was about 2.9
times faster than CPU on that input, so MACE-OFF23-SC is not GPU-unsupported.
The one-box diagnostic evaluates no experimental labels and is not an accuracy
panel.  MAPLE therefore keeps production CUDA selection fail-closed pending a
predeclared experimental comparison with at least ten records and ten distinct
primary functional groups showing no chemical-accuracy degradation.  This is
an accuracy-admission block, not a runtime-support block.

### Reproducible AniSolv compact upstream-sample audit

`run_anisolv_compact_sample_audit.py` binds the exact clean upstream revision,
compact checkpoint, source tree, sample definitions, geometries, and
experimental labels.  Execution is materialized from the verified Git tree
into an isolated temporary import root, so ignored bytecode and sibling
modules cannot enter the result.  It then calls the official `predict_solvation_energy`
API for the six water examples from `H2O_single_point.py`.  The audit is a
deterministic **CPU reference**: float32 `default` inference, one thread,
highest matmul precision, and TF32, autocast, reduced-precision reductions,
compilation, and the upstream `fast`/`fast_gpu` modes all disabled.

```bash
python docs/pretrained-solvation-hub/run_anisolv_compact_sample_audit.py \
  --upstream-root /absolute/path/to/clean/anisolv \
  --output /absolute/path/to/anisolv-compact-sample-audit.json
```

The frozen artifact is
`benchmarks/anisolv-compact-upstream-sample-audit-2026-07-30.json`
(SHA256
`9018364bef21b7d672f0a1404954b26702a096d0be4da266af719195349b07f6`).
The six-row output is explicitly ineligible for accuracy evaluation: it has
fewer than ten records and no predeclared ten-functional-group taxonomy.
Consequently `metrics = null`; the artifact does not calculate, freeze, or
display experimental errors, MAE, RMSE, or maximum error.

It is water-only, its exact FreeSolv/MNSol record IDs and training overlap are
unavailable, and it compares a geometry-level correction directly with
experimental hydration free energies without a complete sampled
thermodynamic cycle or standard-state ledger.  The artifact includes
row-complete CPU correction forces only as identity diagnostics; they are not
admitted scientific outputs.  No GPU run was performed, no CPU/GPU
scalar-energy parity was demonstrated, and the AniSolv GPU gate remains
closed.

### Reproducible AniSolv compact no-loss GPU rejection audit

`run_anisolv_compact_gpu_no_loss_audit.py` performs the separate physical-GPU
check required before any accelerator admission.  It reuses the exact pinned
six-example water identities and source-isolated official runtime, warms each
model lane, and then executes three deterministic repetitions on CPU and an
NVIDIA GeForce RTX 4060 Laptop GPU.  CPU and CUDA use the same float32 or
float64 dtype and the upstream `default` inference mode.  Matmul precision is
`highest`; TF32, autocast, float16/bfloat16 paths, reduced-precision
reductions, compilation, and fast modes are disabled.  Literal scalar equality
is mandatory on this diagnostic.  Experimental accuracy is not evaluated on
the ineligible six-row panel; any future approval would separately require a
formal panel satisfying the 10-record/10-functional-group gate.

```bash
python docs/pretrained-solvation-hub/run_anisolv_compact_gpu_no_loss_audit.py \
  --upstream-root /absolute/path/to/clean/anisolv \
  --output /absolute/path/to/anisolv-compact-gpu-no-loss-audit.json
```

The frozen negative artifact is
`benchmarks/anisolv-compact-gpu-no-loss-audit-2026-07-30.json` (SHA256
`5164e7ff38b9533ed9dd06d6f64e5ef53c555deb3fd346bf648a0d6b8d99462b`).
Float64 has five nonidentical values out of six and a maximum CPU/CUDA energy
difference of `1.110223e-16 eV`.  Float32 has six nonidentical values and a
maximum energy difference of `4.398637e-8 eV`.  No experimental error is
computed on this undersized panel.  The frozen six-record timing diagnostic
also did not show a speedup, but timing is deliberately not an admission
observable and could not override scientific failure if it had.

The artifact verdict is therefore `fail_no_loss`.  Its six-row panel is also
below the accuracy gate and cannot unlock an accelerator even if literal
parity had passed.  A smaller panel may reject an accelerator by exposing a
nonzero numerical mismatch, but it can never unlock one or support an
experimental-accuracy conclusion.  It is not a parity approval token: the
AniSolv model card retains null evidence fields, no allowed GPU task or
inference mode, and `no_loss_parity_verified = false`.  Machine-rounding scale
is still nonzero under the strict zero-loss contract, so MAPLE continues using
the validated CPU route rather than silently accepting a tolerance.

### Reproducible AniSolv compact force-consistency audit

`run_anisolv_compact_force_consistency_audit.py` evaluates exact-upstream
autograd forces against central finite differences on the sample H2O and
CH3OH geometries.  It uses the same verified Git-archive source isolation,
deterministic CPU float64 `default` inference, a `1e-5 angstrom` displacement,
and a fixed proper rotation.  All reduced-precision and fast modes remain
disabled.

The frozen artifact is
`benchmarks/anisolv-compact-force-consistency-audit-2026-07-30.json`
(SHA256
`44b095002282b80f5f4e9f8a16bbccded0f9bf105e020b8b8f243900c1edfb35`).
The original geometries have maximum force discrepancies of
`0.004636657 eV/angstrom` (H2O) and `0.005704464 eV/angstrom` (CH3OH), above
the `1e-6 eV/angstrom` gate.  After the fixed rigid rotation the panel maximum
is `1.086984e-9 eV/angstrom`, while the scalar-energy rotation drift is exactly
zero.  This orientation-specific failure rejects upstream AniSolv forces and
every force-derived MAPLE task; it does not reject the separately gated scalar
single-point correction.

MACE-POLAR-1 already uses the upstream `mace_polar()` loader and is audited,
not reimplemented.  The public MACE-OFF23-SC checkpoint is pinned in this
branch only as a full-coupling explicit-system PES endpoint
(`solvation_mode=none`); its separate optional OpenMM-ML/OpenMMTools
replica-exchange alchemical bridge remains pending.  The published
MACE-OFF24-SC result is not assigned to that public OFF23-SC checkpoint.

### Reproducible MACE-OFF24-SC supplied-result audit

`run_mace_off24_sc_supplied_result_audit.py` verifies the exact official
supporting CSV and PDF, extracts the two article panels, and recomputes their
rounded-table statistics without loading a model.  The source files are
CC BY-NC 4.0 and are not redistributed here; the frozen JSON contains artifact
identity, aggregate metrics, record-set digests, and the maximum-error record,
but no row-level labels or predictions.

```bash
python docs/pretrained-solvation-hub/run_mace_off24_sc_supplied_result_audit.py \
  --csv /absolute/path/to/ja5c10940_si_002.csv \
  --pdf /absolute/path/to/ja5c10940_si_001.pdf \
  --output /absolute/path/to/mace-off24-sc-supplied-results.json
```

From the official rounded values, the selected 36-record FreeSolv hydration
panel has `MAE = 0.763889`, `RMSE = 0.834389`, and maximum absolute error
`= 1.600000 kcal/mol`; it therefore exceeds the strict `< 1.5 kcal/mol`
maximum-error target.  The separate 10-record MNSol octanol panel has
`MAE = 0.474000`, `RMSE = 0.587929`, and maximum absolute error
`= 0.930000 kcal/mol`.  These two different panel identities are reported
separately and are never pooled or ranked as one benchmark.

The rounded hydration CSV does not reproduce the article Table 1 aggregates
of `MAE = 0.69` and `RMSE = 0.80`: its recomputed-minus-reported gaps are
`0.073889` and `0.034389 kcal/mol`.  Both exceed a conservative
`0.03 kcal/mol` combined displayed-value bound, so rounding alone cannot
explain the mismatch; the underlying source-analysis discrepancy remains
unknown.  More importantly, the exact MACE-OFF24-SC checkpoint is not publicly
pinned: the clean `mace-md` protocol source points to the distinct public
MACE-OFF23-SC release.  This artifact is consequently static
negative-admission evidence, not checkpoint inference, an alchemical protocol
rerun, a strict holdout, or a Route 4 runtime result.

### FeNNix-Bio1: public GPU/Lambda source, paper FE integration blocked

FeNNix-Bio1 closes the artifact-identity gate that MACE-OFF24-SC leaves open.
The official FeNNol-PMC revision
`83f299b81c1d62e2a15c892280559a7c0cc2fac3` publishes the base small and
medium checkpoints:

- `fennix-bio1S.fnx`: 29,772,732 bytes, SHA256
  `82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07`;
- `fennix-bio1M.fnx`: 38,124,300 bytes, SHA256
  `5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a`.

They are ASL academic-only, non-commercial artifacts.  The separate upstream
`v1.0-finetuneIons` files are different identities and are not silently
substituted.  The LGPL-3.0 FeNNol runtime at
`d62b8740343b803a2b864140ec79e347f8ba034e` provides an ASE calculator with
energy, energy-derived force, and stress outputs.  Its graph, repulsion, and
charge-equilibration code also accepts the alchemical group and lambda inputs
described by the article.

Tinker-HP revision
`384bb7d451f85f2ea825dabd888e64ad4448674b` closes a narrower source-code
question: `v1.3/GPU/source/mlinterface.py` (16,624 bytes, SHA256
`4437e0b93a55156d554ba13a3d7c974a4c537a0792b15c3ec3cac7d2d8758967`)
loads a FeNNol `.fnx` model, adds `alch_elambda` and `alch_vlambda` gradient
keys, passes the alchemical group and ligand-charge state, and returns both
lambda derivatives to Tinker-HP.  The same tree contains generic Deep-HP and
Lambda-ABF/Colvars documentation.  Thus a public combined FeNNol plus
GPU/Lambda-ABF source path exists; it is no longer correct to describe those
operations as available only in separate projects.

That source tree still does not package a named FeNNix-Bio1/FreeSolv input
bundle or row-complete paper ledger.  Its shipped Deep-HP example selects
`ani2x.fnx`, not either Bio1 checkpoint.  The static, dependency-free audit in
`run_fennix_bio1_release_audit.py` freezes these exact source and checkpoint
identities in
`benchmarks/fennix-bio1-release-audit-2026-07-30.json`; it performs no
inference or protocol run.

```bash
python docs/pretrained-solvation-hub/run_fennix_bio1_release_audit.py \
  --fennol-root /absolute/path/to/FeNNol \
  --tinker-hp-root /absolute/path/to/tinker-hp \
  --small-checkpoint /absolute/path/to/fennix-bio1S.fnx \
  --medium-checkpoint /absolute/path/to/fennix-bio1M.fnx \
  --output /absolute/path/to/fennix-bio1-release-audit.json
```

The v4 paper reports explicit-water Lambda-ABF hydration results for 625
retained FreeSolv molecules using the medium checkpoint: `MAE = 0.70` and
`RMSE = 0.98 kcal/mol`.  Its 452-record H/C/N/O/S/P subset is reported at
`MAE = 0.59` and `RMSE = 0.82 kcal/mol`.  Those values are promising
literature evidence, not an acceptance result: no row-complete public
prediction table or maximum error is supplied, training overlap is unknown,
and the paper-specific treatment separately equilibrates solute and solvent
charges during alchemical decoupling.  The reported solvent is water; an
ordinary explicit-system MLIP is not by itself a validated multi-solvent
solvation backend.

No single installed MAPLE environment contains the unchanged stack.  The
`maple-resolv` environment has CPU-only JAX 0.4.23 but lacks FeNNol, ASE, and
OpenMM; the OpenMM-bearing environments lack JAX and FeNNol.  Tinker-HP's
Deep-HP build also requires its original GPU toolchain and Python bridge
dependencies, while the required NVIDIA HPC compiler is absent.  Without
authorization MAPLE does not install or splice those environments, port the
JAX runtime, or expose a placeholder calculator.

The pinned Tinker bridge requests JAX's `highest` matrix-multiplication
precision, but passes coordinates and lambda values as float32 and exposes no
bridge-level precision switch.  FeNNol's generic ASE/MD paths separately
provide float64 controls.  None of this proves GPU/reference equivalence:
energy, forces, virial, lambda derivatives, trajectories, final free energy,
and uncertainty have not been compared locally.  Under the no-loss rule the
GPU path therefore remains blocked rather than being accepted from source
availability alone.  FeNNix-Bio1 remains a high-priority, checkpoint-pinned
water-alchemical candidate whose ordinary mechanics and exact end-to-end
free-energy protocol still require original-runtime execution, no-loss GPU
parity, and task-by-task validation.

C3Net is deliberately a fifth, property-only lane.  Its dedicated
`C3NetPropertyAdapter` runs an unmodified, hash-pinned official source checkout
in an isolated subprocess, accepts one supplied neutral single-fragment
three-dimensional SDF record or one to five records of that same solute, and
one of its 103 upstream solvent identifiers.  The multi-record form returns
the upstream predictions' arithmetic mean in `kcal/mol`.  It has no
energy surface, forces, optimization, frequency, transition-state, IRC, MD, or
ASE `#model` interface.  Its published random pair split is useful upstream
reproduction evidence only: absent a complete record-level training manifest,
C3Net results remain `property_prediction` with `overlap_unknown`, not an
absolute-solvation backend or a sealed independent experimental score.  The
paper describes 5,718 solvation-free-energy values spanning 890 solutes and
103 solvents and a random 80/20 split, but the only checked-in split ledger
located in the official repository contains 41 training and 10 validation
indices and is not the paper-scale manifest.  Without canonical record IDs for
the complete checkpoint training pool, no public experimental panel can be
proven never-used by record-level intersection.  The current C3Net pilots are
therefore development-only negative evidence and must never be relabelled as a
strict extrapolation holdout.

CIGIN is a separate, exact-checkpoint property sidecar.  Its original source
directly accepts one solute SMILES and one solvent SMILES; MAPLE canonicalizes
and restricts both to neutral, radical-free, single-fragment molecules in the
upstream feature domain, then invokes the clean pinned source checkout in an
isolated **CPU-only** child process.  CPU forcing is an explicit upstream
compatibility boundary: the released CUDA branch combines CPU interaction maps
with CUDA tensors, and MAPLE does not patch or replace it.  The CIGIN authors
report a `0.91 kcal/mol` FreeSolv RMSE, but the source only identifies FreeSolv
as training provenance and publishes no complete record-level training ledger
or certified multi-solvent coverage.  Thus its syntactic solvent-SMILES input
is not a multi-solvent admission claim, its result remains
`property_prediction` with `overlap_unknown`, and it has no PES, force,
geometry, conformer, optimizer, or absolute-solvation meaning.  A CIGIN score
must never be ranked against C3Net or Route 2 unless every panel-identity
field is equal.

### Reproducible C3Net development pilot

The fixed ten-record, water-only FreeSolv development pilot is captured in
`benchmarks/c3net-freesolv10-2026-07-30.json`.  It generates one deterministic
RDKit ETKDGv3 + UFF conformer per fixed pilot SMILES, which is an explicit
single-conformer property protocol rather than the paper's multi-conformer
random split.  Re-run it only against a clean official checkout:

```bash
python docs/pretrained-solvation-hub/run_c3net_freesolv10_pilot.py \
  --source-root /absolute/path/to/C3Net \
  --output /absolute/path/to/c3net-freesolv10.json
```

The immutable input predeclares ten chemistry classes, but classes such as
alkane and aromatic hydrocarbon are not a validated ten-functional-group
taxonomy.  The artifact is therefore runtime/interface smoke with
`metrics = null` and no per-record experimental errors.  It is water-only,
development-only, and `overlap_unknown`, so it cannot support accuracy,
ranking, generalization, selection, confirmation, or independent
extrapolation.

The separate `benchmarks/c3net-dgsolvdb1-11solvent-2026-07-30.json` artifact
uses the public dGsolvDB1 archive (298 K, 1M/1M) and an output-blind rule of
two neutral, supported records in each of 11 exact C3Net solvents.  Each record
uses five ETKDGv3 + UFF conformers and the paper's arithmetic-mean inference
convention.  Because its record-level functional groups were not predeclared,
the frozen 22/22 output is now explicitly runtime/interface smoke:
`metrics = null`, no per-record experimental errors are emitted as accuracy
evidence, and it cannot support a precision, ranking, admission, or
generalization claim.  Its multi-solvent execution coverage remains useful,
but `overlap_unknown` also prevents any strict-holdout interpretation.

```bash
python docs/pretrained-solvation-hub/run_c3net_dgsolvdb1_multisolv_pilot.py \
  --source-root /absolute/path/to/C3Net \
  --dataset-archive /absolute/path/to/Solvation_data-1.0.0.zip \
  --output /absolute/path/to/c3net-dgsolvdb1.json
```

### Formal C3Net/FlexiSol development accuracy result

`run_c3net_flexisol_formal_accuracy.py` is the first C3Net panel allowed to
report experimental errors.  It verifies the pinned FlexiSol Git object,
filters only neutral single-fragment C3Net-domain solutes, applies the frozen
SMARTS taxonomy before model execution, and selects one minimum-SHA256 record
for every eligible primary group without reading the experimental value or
model output.  The resulting panel has 19 records and 19 distinct primary
functional groups.  Its selected records span water, octanol, and hexadecane;
this development panel is still `overlap_unknown` and is not the final
independent extrapolation set.

The registered adapter is path-independent and binds its runner, adapter,
C3Net property wrapper, full-history Git bundle, checkpoint, embedding, and
official historical environment lock.  Formal execution separately binds and
validates the observed executable runtime lock
`c3net-formal-runtime-lock-2026-07-30.json` (SHA256
`3db89924dae337c6cb5de7cabc67f87b7c32ef6a05d5a7321927ff443fc342d6`);
this explicitly records the current Python, Torch, NumPy, RDKit, platform, and
build fingerprints rather than pretending that the upstream Python 3.7 /
Torch 1.7 environment was reproduced.  Any runtime drift fails closed.  The
parent process also binds the panel runner and functional-group taxonomy by
SHA256 without mounting either label-bearing selection surface into the
label-free prediction sandbox.  Conformer generation is a deterministic,
output-blind adaptation of the pinned upstream structure: 50
`EmbedMultipleConfs` candidates using the observed runtime defaults except for
the deterministic seed, default UFF relaxation, hydrogen-stripped alignment,
greedy heavy-atom RMSD deduplication at `0.5 Angstrom`, and at most five
retained conformers.  It is not described as a byte-exact historical replay
because upstream used an unseeded generator under RDKit 2022.3.5.

The label-free worker receives exact SDF bytes containing the actual one to
five retained conformers plus only registered implementation paths inside
bubblewrap.  C3Net inference is CPU deterministic, preserves explicitly
declared stereochemistry across conformers, uses `model.eval()`, and averages
the numerically ordered raw checkpoint tensor predictions with
order-independent `math.fsum`; it does not re-ingest the upstream
three-decimal text report.  Every UFF status and energy is retained.  Under
the upstream default 200-iteration limit, 83/950 candidate relaxations and
9/80 retained conformers reported non-convergence; MAPLE records rather than
hides this limitation and does not silently alter the upstream-structured
protocol after seeing the labels.

```bash
PYTHONPATH=. python \
  docs/pretrained-solvation-hub/run_c3net_flexisol_formal_accuracy.py \
  --flexisol-source-root /absolute/path/to/flexisol \
  --c3net-source-root /absolute/path/to/C3Net \
  --c3net-source-bundle /absolute/path/to/c3net-full-history.bundle \
  --output-directory /absolute/path/to/formal-output
```

The frozen result is
`benchmarks/c3net-flexisol-formal-2026-07-30/result.json` (SHA256
`fdd2f535b2e8be04788323283812f20c817e3641d783339f6b80b3c92db3884e`).
Two fresh isolated runs emitted byte-identical 21-file artifact trees.  The
complete file manifest and both equal tree digests are retained in
`benchmarks/c3net-flexisol-formal-2026-07-30-reproducibility.json` (SHA256
`d54f7b09848330bd51d9263355b7fdd7f08c284f02306bf1d9c666b80bf82a7c`).
The different filesystem artifact-write spans in that receipt are positive
reproducibility diagnostics and lower bounds only, not wall-clock or matched-QM
performance evidence.  C3Net obtained MAE
`2.148012556778758 kcal/mol`, RMSE `2.9857217677885237 kcal/mol`, and maximum
absolute error `8.459287071228028 kcal/mol`; the maximum-error record is the ester
`flexisol-ester-08a31fdfb3b2581f` (`-5.6` experimental versus
`-14.059287071228027` predicted).  It therefore fails the strict
`<1.5 kcal/mol` maximum-error gate.  The matched-QM performance gate is
consequently **not eligible**, rather than being run and used to hide an
accuracy failure.

### Reproducible CIGIN development pilot

`run_cigin_freesolv10_pilot.py` evaluates the same immutable ten-record
FreeSolv input with **a different model/prediction identity**: direct canonical
SMILES pairs, no supplied geometry, no conformer generation, and a CIGIN
checkpoint whose record-level training accounting is unavailable.  It is
therefore development-only `property_prediction` evidence with known source
dataset provenance but `overlap_unknown` records, never a strict holdout,
acceptance result, or cross-model ranking.  Re-run it only against a clean
official checkout:

```bash
python docs/pretrained-solvation-hub/run_cigin_freesolv10_pilot.py \
  --source-root /absolute/path/to/CIGIN \
  --output /absolute/path/to/cigin-freesolv10.json
```

The frozen artifact is
`benchmarks/cigin-freesolv10-2026-07-30.json`.  Like the C3Net artifact, its
ten chemistry classes are not a validated ten-functional-group taxonomy, so
it is runtime/interface smoke with `metrics = null` and no experimental error
calculation.  It cannot support accuracy or generalization.  It is also not
comparable to C3Net despite shared records: their geometry/conformer and
model/prediction identities differ, and CIGIN's FreeSolv source-dataset
provenance makes a strict holdout impossible.

### AtomicESE: audit-only scalar multi-organic-solvent candidate

The official 2025 article
([DOI 10.1002/jcc.70104](https://doi.org/10.1002/jcc.70104)) describes
AtomicESE as a dedicated scalar predictor for molecules and ions in organic
solvents. Its `13 x 30 x 28 x 1` atomic network has 1,258 weights and 59
biases, or 1,317 adjustable parameters, and sums atomic contributions into a
reported standard solvation free energy. The article/SI page advertises
`jcc70104-sup-0001-supinfo.zip` as containing scaling factors, weights, biases,
and AtomicESE-calculated solvation-free-energy predictions. Those are
publication statements, not locally verified artifacts: the SI ZIP bytes are
currently unavailable to this audit because the publisher returns HTTP 403,
so MAPLE has no SI hash, parsed weight ledger, or prediction-table audit.

`run_atomicese_release_audit.py` instead verifies the much narrower official
Git release at revision
`31e643c7e8974497c78fc2fb6f3c17778d61fa10`, tree
`e08ce8bfaf73ca9c35c2b85929067e639d7bf048`. In that exact pinned official Git
tree only, the release contains packaged x86_64 Linux and Windows executables, a
README, and one input example; it contains no repository license file, source
code, separate checkpoint or checkpoint ledger, or record-level training/split
ledger. This makes no absence claim about the article or currently unavailable
supporting-information bytes. The frozen release/runtime result is
`benchmarks/atomicese-release-audit-2026-07-31.json` (SHA256
`efae3cd8355d6221a3814e6f1a8abdc06adb108d95d084f0589dbe3d0bfa708a`).
Its sandboxed repeated acetone sample establishes only deterministic scalar
CLI behavior. **Binary sample timing is not speed evidence.**

AtomicESE therefore has no MAPLE integration, model card, adapter, accuracy
panel, GPU admission, or matched-QM timing. The exact quantitative standard
state and record-level training/split membership remain unresolved. No
documented or verified packaged GPU path exists, and GPU capability remains
unverified: absence of selected GPU/CUDA ASCII markers does not prove the
executables are CPU-only. GPU admission is false. The candidate
is rejected and blocked unless a formal accuracy panel first passes with at
least 10 records spanning at least 10 distinct, predeclared primary functional
groups. Only after that pass may the same task at the same accepted precision
be timed end to end for at least three complete repeats. Its hard performance
gate is
`median(QM full-task seconds) / median(AtomicESE full-task seconds) > 1`;
equality or a slower candidate is rejection.

### MoletoSolv: literature baseline, not a pretrained release

MoletoSolv reports neutral, anionic, and cationic multi-solvent scalar
ΔGsolv MAEs of `0.44`, `1.72`, and `1.60 kcal/mol`, respectively.  Its public
revision `04e6229ec414dcb66fa46c1e26f89f3ea91ba5e2` is not a drop-in pretrained
release: it contains source, datasets, 92 solvent descriptors, solute
geometries, and precomputed predictions, but no serialized estimator, scaler,
feature-selection artifact, tag, or release.  The code creates/loads local
`joblib` artifacts, and the neutral training CSV is empty at that revision.
MAPLE therefore will not reconstruct or train the missing artifacts.  With no
PES/forces, maximum-error table, immutable split ledger, independent holdout,
standard-state contract, or no-loss GPU parity evidence, MoletoSolv remains a
scalar literature baseline only.

### SolProp-mix Exp: immutable official mixture-property candidate, blocked

The exact official SolProp revision
`80043ce09eb8802517c35b59254f8e9c181f2dac` and the complete Zenodo `Files.zip`
archive (288,947,625 bytes; SHA256
`670915e5bf86d5457bc2f43301e5a02e77a66b529535fcd0eddfadeb72ed73a5`) identify
the authors' requested `SolPropmixExp` ten-member ensemble.  The clean
upstream README limits it to neutral supported solutes and nonionic liquids,
and identifies pure plus mixed-solvent solvation-free-energy predictions.  The
paper's MolPool construction treats solvent composition as input; the authors
report `0.25 kcal/mol` MAE and `0.37 kcal/mol` RMSE for nonaqueous mixtures.
That result is publication evidence only, not a MAPLE score or an independent
panel.

`run_solpropmix_candidate_holdout_audit.py` now freezes a narrower,
non-scoring candidate ledger from only `Not used - From 3-comp VLE` and
`Not used - From 4-comp VLE`.  It verifies the exact outer archive, workbook,
worksheet relationship map, worksheet bytes, and Dortmund InChI dictionary.
It then canonicalizes only solute/mixture/condition/citation identities:
the 30,184 rows contain 293 solutes, 279 solvents, and 28,551 unique physical
mixtures.  Declared components at exactly zero mole fraction remain nominal
source-cardinality provenance but are omitted from physical mixture identity.
The experimental `Gsolv` cells are counted structurally but their values are
never decoded, embedded, compared, or used for model selection.  Three
four-component records require an exact Decimal correction of `1e-16` to
remove XLSX binary-serialization excess above unit solvent composition; no
experimental value or model result is altered.

```bash
python docs/pretrained-solvation-hub/run_solpropmix_candidate_holdout_audit.py \
  --archive /absolute/path/to/Files.zip \
  --output /absolute/path/to/solpropmix-candidate-audit.json
```

The frozen audit remains negative final-holdout evidence: 43 duplicate
physical-record identities and 48 rows with incomplete core citation
provenance are present,
and canonical joins against FreeSolv, MNSol, dGsolvDB, CombiSolv, Solv@TUM,
SolProp, QM-derived corpora, and every candidate model family are pending.
The artifact now also freezes the paper's dataset boundary: the released
workflow pretrains on CombiSolv-QM plus BinarySolv-QM, fine-tunes on
CombiSolv-Exp, and tests on BinarySolv-Exp plus TernarySolv-Exp using
solute-level splitting.  The paper reports that those published experimental
test solutes were excluded from CombiSolv-Exp but still have some overlap with
the synthetic COSMO-RS corpora.  That statement applies to the named published
test sets only; it does not clear either `Not used` worksheet.
The upstream `Not used` worksheet names therefore establish only release
provenance, not absence from all training/test histories.  No SolProp-mix
label may be scored until that overlap ledger is complete, and this static
audit supplies no GPU precision or performance evidence.

MAPLE has not installed or substituted anything.  The original entry point
currently fails before inference because its required `Tap` and
`memory_profiler` packages are absent, while model construction requires
absent `DGL`; `tensorboardX` and `CoolProp` are also listed official
requirements but are absent.  The candidate consequently has no exposed
adapter.  The separate public G-NequIP SMD-water checkpoint is likewise kept
out of the calculator registry: its uninstalled original NequIP loader and a
missing common gas/solution energy gauge make single-point subtraction
non-thermodynamic.  If authorization later permits installation of the original runtime,
any sidecar must keep the official ten final weights unchanged and record
explicit mixture identities, composition, temperature/standard-state meaning,
training-overlap audit, and an independent experimental panel.  It is never a
PES, force, calculator, geometry, MD, or sampled absolute-solvation-protocol
backend.

### Reproducible G-NequIP supplied-result audit

`run_gnequip_smdw_supplied_result_audit.py` verifies the small official Zenodo
result archive rather than loading a checkpoint.  It recomputes the reported
FreeSolv experimental-panel statistics from the supplied gas/SMD NNP energy
tables: `MAE = 1.082689`, `RMSE = 1.425792`, and maximum absolute error
`= 5.445611 kcal/mol` across the fixed 388-record intersection.

```bash
python docs/pretrained-solvation-hub/run_gnequip_smdw_supplied_result_audit.py \
  --archive /absolute/path/to/solvation_free_energies.zip \
  --output /absolute/path/to/gnequip-smdw-supplied-audit.json
```

This is deliberately not a MAPLE model-inference or absolute-solvation result:
the upstream archive defines its number by subtracting independently trained
gas and SMD-water single-point models on separate geometries, supplies no
thermodynamic estimator or standard-state field, is water only, and has
`overlap_unknown` training records.  Its bundled plotting script uses stale
archive member names and a bootstrap-count setting inconsistent with the README,
so this audit verifies central statistics only.  The frozen result is negative
admission evidence, not a route around those conditions.

### ML-for-charges PBE0-ESP: tracked exclusion, not an implicit-solvent backend

The public MIT repository
[`mathilfiker/ml_for_charges`](https://github.com/mathilfiker/ml_for_charges)
contains a pretrained XGBoost atomic-charge regressor at source revision
`e8407cc7e500d89cf66ed0673d6cd7421ab5637d`.  Its tracked model artifact is
`charge_prediction/model/xgb_chgs.json` (91,849,931 bytes; upstream Git blob
`8d8ae6e35496ed5587685fa3a24c9a598dec2565`), and its unchanged example uses
MACE-OFF23-large descriptors.  This proves a legitimate pretrained release;
it does **not** make it a Route 4 candidate.

The output is gas-phase PBE0-D3(BJ)/def2-TZVP-quality atomic charges, with no
solvent or mixture identity, energy, force, paired endpoint, or free-energy
estimator.  The paper's reported 30-molecule FreeSolv hydration result instead
comes from a separate explicit-water alchemical MD workflow and must not be
treated as a MAPLE score or compared to a Route 4 panel.  The original loader
also requires `xgboost`, absent in all installed MAPLE environments; MAPLE
will neither install it nor reimplement the source path without explicit
authorization.  It remains excluded from Route 4 and could only be assessed
later as a fixed charge provider in a separately authorized Route 1
explicit-water protocol with an independent experimental identity.

### FlexiSol: pinned external-confirmation candidate, not a final holdout

The official FlexiSol repository is pinned at revision
`7b44798f26c888ef541faa6143a813136921483f` and tree
`8cf4879bc1435ca1061acfffd0daf10784445984`, after the authors' documented
hexadecane-reference correction.  Its experimental tables contain 530
ΔGsolv records over seven pure solvents and 294 logK records over six solvent
pairs.  The seven pure solvents do not satisfy Route 4's required ten-solvent
coverage, and the official method registry already contains `directml` and
`cigin`; model-specific training/test intersections remain unknown.

`run_flexisol_release_audit.py` verifies the exact Git blobs, CSV schemas,
finite reference fields, solvent/source counts, correction notice, registry,
and license.  It reads no model predictions, computes no model errors, embeds
no records, and performs no model selection:

```bash
python docs/pretrained-solvation-hub/run_flexisol_release_audit.py \
  --source-root /absolute/path/to/flexisol \
  --output /absolute/path/to/flexisol-release-audit.json
```

The frozen summary is
`benchmarks/flexisol-release-audit-2026-07-30.json` (SHA256
`cf2cd04845390e7b7574232b43207f352cc8e32c58469a799d8f21592bbd5ca4`).
FlexiSol may later serve as external confirmation only after exact
candidate-specific overlap accounting.  It is not acceptance evidence, a
never-used final blind panel, or a substitute for the required independent
experimental extrapolation set.

## Benchmark rules

- FreeSolv and MNSol are independent panels and are never ranked against one
  another.
- A paired comparison may differ in potential/model identity, but requires
  identical records, geometry/conformer policy, solvent protocol/backend,
  cavity/PMF, sampling, estimator, standard-state, and experimental provenance.
  For a mixture, the record identity also contains the exact component multiset
  and positive mole fractions summing to one; reordering components is the same
  physical state, while changing a fraction is a different panel.
- Training overlap defaults to `overlap_unknown`.  Absence of evidence is not a
  strict holdout.
- The frozen MNSol split is referenced by
  `mnsol-partition-reference.json`.  MNSol row-level data and labels are not
  redistributed.
- Exact public checkpoint identities, sizes, hashes, unit contracts, and
  license unknowns are frozen in `upstream-artifacts.json`.
- Numerical target bands such as 1.0 or 1.5 kcal/mol are internal research
  targets, not universal literature-derived integration gates.

## Upstream sources

- AIMNet2-CPCMS artifact context: <https://github.com/isayevlab/LoQI>
- GNNIS: <https://github.com/rinikerlab/GNNImplicitSolvent>
- AniSolv compact: <https://github.com/Ant-on-knee/anisolv> and <https://huggingface.co/antonknee/anisolv>
- C3Net: <https://github.com/SehanLee/C3Net> and <https://arxiv.org/abs/2309.15334>
- CIGIN: <https://github.com/devalab/CIGIN> and <https://hai.iiit.ac.in/d4.html>
- MoletoSolv: <https://github.com/lyjia-bnu/MoletoSolv> and <https://doi.org/10.1021/acs.jpcb.5c01669>
- SolProp-mix: <https://gitlab.kuleuven.be/creas/vermeiregroup/solprop>, <https://zenodo.org/records/14238055>, and <https://arxiv.org/abs/2412.01982>
- FlexiSol: <https://github.com/grimme-lab/flexisol> and <https://doi.org/10.1039/D5SC06406F>
- G-NequIP: <https://github.com/otayfuroglu/deepPotential> and <https://zenodo.org/records/20690503>
- ML-for-charges PBE0-ESP: <https://github.com/mathilfiker/ml_for_charges>, <https://arxiv.org/abs/2512.13579>, and <https://doi.org/10.5281/zenodo.17790331>
- Organic_MPNICE HFE protocol: <https://doi.org/10.1021/acs.jctc.5c02019> and <https://github.com/leifjacobson/MLFF_HFE>
- ABCG2/GAFF2 explicit-solvent control: <https://doi.org/10.1021/acs.jctc.5c00038> and <https://github.com/junmwang/abcg2>
- LSNN-v1: <https://github.com/Popov-Lab-UNC/LSNN-v1>
- MACE-OFF: <https://github.com/ACEsuit/mace-off>
- MACE-OFF24-SC article and protocol: <https://doi.org/10.1021/jacs.5c10940> and <https://github.com/jharrymoore/mace-md>
- FeNNix-Bio1 model and runtime: <https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/FENNIX-BIO1>, <https://github.com/FeNNol-tools/FeNNol>, and <https://doi.org/10.26434/chemrxiv-2025-f1hgn-v4>
- AceFF 2.0: <https://huggingface.co/Acellera/AceFF-2.0>
- OpenMM-ML sampling bridge: <https://github.com/openmm/openmm-ml>
