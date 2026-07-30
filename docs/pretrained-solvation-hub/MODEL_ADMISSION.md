# Route 4 model-admission decision record

This record turns the Route 4 objective into a proof obligation.  A candidate
is not admitted merely because it reports a low scalar error: MAPLE must be
able to execute the exact released potential and the requested thermodynamic
quantity must be defined by that potential.

## Non-negotiable mathematical boundary

For an implicit-solvent potential `U_solv(R, S)` and its matching vacuum
potential `U_vac(R)`, the configured solvation free energy is a partition
function ratio, not a single-point energy difference:

```text
Delta G_solv(S) = -beta^-1 log < exp[-beta (U_solv(R, S) - U_vac(R))] >_vac
```

Equivalently, an alchemical implementation must define one continuous,
conservative family `U(R, lambda, S)` with the exact endpoint identities
`U(R, 0, S) = U_vac(R)` and `U(R, 1, S) = U_solv(R, S)`, and evaluate a
converged TI/BAR/MBAR estimator.  If a learned force field only determines
forces, `U'(R, S) = U(R, S) + C(molecule, S)` leaves forces unchanged but
changes `Delta G_solv`; it is therefore not an absolute-solvation backend.

This is why GNNIS remains a sealed multi-solvent reference Hamiltonian rather
than an absolute-solvation calculator.  It also rules out empirical offsets,
label calibration, refitting, and treating a composition of an arbitrary MAPLE
MLIP with a learned solvent PMF as an absolute-solvation calculation.

## Admission gates

Every candidate must pass all gates before it can become an executable Route 4
backend:

1. **Released identity:** official inference code and already-trained weights
   are public, with a fixed revision, artifact URL, SHA256, license, and unit
   contract.  MAPLE never reconstructs, trains, fine-tunes, or substitutes a
   checkpoint.
2. **Thermodynamic gauge:** the released model supplies the endpoint energy
   reference above, or a lambda-conditioned energy with audited
   `dU/dlambda`.  Force matching alone is insufficient.
3. **Solvent state:** every selectable solvent has an upstream identifier and
   published meaning.  A native solution potential cannot receive a second
   GB/PB/PCM/GNNIS correction.
4. **MAPLE task contract:** SP/OPT/scan/TS/IRC need conservative
   energy-derived forces; frequency needs a valid solution-PMF Hessian;
   MD/free-energy use needs deterministic runtime, sampled-state diagnostics,
   and a converged estimator.  Each unsupported task fails closed.
5. **Evidence identity:** every reported metric binds exact records, solvent,
   temperature, standard state, protonation/tautomer/conformer policy,
   potential, cavity/PMF, sampling, estimator, and experimental provenance.
   A mixture record additionally binds its full solvent-component multiset and
   strictly positive mole fractions summing to one; component order is
   canonicalized, but a fraction change changes the identity fingerprint.
   Training overlap is `overlap_unknown` until a complete record manifest
   proves otherwise.  An `absolute_solvation_free_energy` panel additionally
   requires a quantitative standard state, named sampling protocol, and
   TI/BAR/MBAR/FEP (or an explicitly named harmonic) estimator; a single-point
   energy difference is a geometry-level PMF observable, not a free-energy
   benchmark.
6. **Acceptance sequence:** reproduce an upstream-comparable panel without
   labels in the execution path; then run a chemistry-stratified multi-solvent
   panel; then lock an independent experimental extrapolation panel never used
   for model selection.  An accuracy panel must contain at least 10 records
   spanning at least 10 distinct, predeclared functional groups.  Its immutable
   identity binds the exact frozen taxonomy manifest, its structural SMARTS,
   primary-family precedence, source structure identifier, and exactly one
   record-level assignment set per record.  It also binds a canonical solute
   graph plus the SHA256, locator, and format of the exact per-record molecular
   input.  For SDF, MOL2, or coordinate inputs, that hash covers the actual
   geometry-bearing record, not an unrelated free-form SMILES.  A
   `MolecularInputReceipt` is created at the featurization boundary by reading
   that real file or ZIP member; it derives the graph and SHA256 from the same
   bytes.  Formal scoring is possible only through `run_accuracy_panel()`.
   Every experimental value, `kcal/mol` unit, record-level provenance, source
   artifact SHA256, and source-row locator must be parsed from an actual JSON
   artifact by `BenchmarkExperimentalReference.from_json_file()` before being
   frozen into the identity; free metadata construction is rejected.  The
   identity selects a source-controlled adapter registration; arbitrary
   callables and caller-assembled predictions are rejected.  The runner
   recomputes a composite fingerprint from live `predict()` code, canonical
   configuration, and reverified adapter-code, featurizer-code, checkpoint, and
   dependency-lock artifacts.  It starts a fresh bubblewrap-confined Python
   interpreter, re-verifies those artifacts there, and imports the adapter from
   disk so parent-process stack state, monkeypatches, labels, and benchmark-data
   directories cannot enter prediction.  The child receives verified
   molecular-input bytes, never their source path or receipt, plus a label-free
   `AccuracyPredictionContext` containing scientific protocol fields.  Its
   mount namespace exposes only the runtime implementation trees and exact
   declared model artifacts; if bubblewrap is unavailable, formal accuracy
   fails closed.  Every prediction is collected and converted before the
   parent re-reads experimental references for scoring, and the implementation
   fingerprint is recomputed after execution.  Paired comparison re-executes
   both registered adapters instead of trusting a supplied result object.
   Scoring re-reads both the molecular and experimental sources and compares
   their receipts against the frozen manifest; duplicated result-table strings
   are not accepted as proof.
   The canonical graph must equal the SMARTS-verified assignment.  Coverage
   counts distinct primary
   functional-group families only; chemical classes, aromaticity, ring systems,
   charge classes, and subtypes cannot inflate the count.  Every record must
   then report its experimental value, prediction, signed error, and absolute
   error.  Complete prediction coverage is mandatory, and the panel reports
   maximum absolute error as well as MAE/RMSE.  The current taxonomy artifact is
   `functional-group-taxonomy-v1.json`, SHA256
   `46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36`.
   A smaller or unclassified panel is runtime/interface smoke only: it cannot
   support accuracy, ranking, generalization, model admission, or GPU
   enablement.  Ten groups is a minimum floor, not universal chemical coverage,
   and no result is compared across panels with different identities.
7. **No-loss acceleration:** a GPU route must preserve the exact checkpoint,
   equations, cutoffs, solvent and lambda state, estimator, convergence
   criteria, and accepted scalar precision.  Float16, bfloat16, TF32,
   fast-math, reduced matrix multiplication, shortened sampling, or relaxed
   convergence are forbidden substitutes.  Before enablement, a frozen paired
   reference/GPU audit must establish energy, force, virial, lambda-derivative,
   free-energy, uncertainty, coverage, and maximum-error behavior.  Numerical
   CPU/GPU nonidentity is diagnostic and does not mean that the MLIP lacks GPU
   support; final accelerator admission requires a formal experimental panel
   with at least 10 records and 10 distinct primary functional groups showing
   no chemical-accuracy degradation.  Hardware availability or throughput
   alone is never admission evidence, and a sub-threshold panel can never
   unlock an accelerator.
8. **Matched-QM performance:** speed is evaluated only after both the candidate
   and its pinned QM baseline pass the same formal accuracy limit on the same
   record IDs, molecular-input hashes, and experimental references.  The timed
   precision must equal the accuracy-accepted precision.  Candidate and QM
   timing scopes must both include input preparation, geometry or sampling,
   prediction/QM execution, analysis, and artifact writing.  At least three
   complete repeats are required and the hard gate is
   `median(QM full-task seconds) / median(candidate full-task seconds) > 1`.
   Equality or a slower candidate is a production rejection.  Cold-start and
   warm-state timings are reported as diagnostics but cannot override the
   full-task gate and may be omitted.  Candidate and QM must use the same
   explicit precision policy accepted by their formal accuracy runs.  The
   generic evaluator rejects caller-supplied summary mappings and reruns both
   registered adapters from source-verifying benchmark identities, but it
   still cannot approve caller-supplied timing inputs: positive admission
   additionally requires
   model-specific, hash-bound timing receipts proving the exact candidate/QM
   commands, methods, checkpoints, inputs and order, environment, hardware,
   precision, and complete task components.  Self-reported arrays, scope
   labels, precision strings, or hardware text are not admission evidence.

A GPU parity artifact is executable admission evidence only when its full
SHA256 matches the model card and its schema binds the exact model/checkpoint
identities, immutable comparison-panel ID/SHA256, complete task/mode matrix,
equal concrete scalar dtype, matrix-multiplication precision, disabled
reduced-precision controls, hardware, software versions, observables, and a
passing no-loss verdict.  The comparison panel must be a repository-relative
manifest artifact that still exists and whose SHA256 is recomputed at runtime;
card/JSON strings alone cannot establish panel identity.  Its admitted JSON
schema must contain the exact model/checkpoints, task/mode matrix, non-empty
unique record identities, panel IDs, configuration/reference SHA256s, and
per-observable counts that agree with the parity artifact.  Partial task
admission is intentionally disallowed:
the evidence must cover every mechanically exposed path.  A public force API
can be reused by an external MD driver, so its trajectory parity remains
mandatory even when the product card forbids MD; forbidden tasks do not waive
accelerator parity coverage.  Observable units, comparison semantics, and
thresholds are fixed by the version-controlled model card; the evidence may
use a stricter threshold but cannot select a looser one.  Coverage is exact,
and per-case accuracy, maximum error, and identity-matched literature metrics
permit zero degradation.

The running adapter must supply the same hardware/runtime fingerprint,
effective dtype, matrix-multiplication precision, every disabled precision
control, an explicit execution task, and an explicit inference mode.  Unknown
devices, integer CUDA aliases, missing cards/tasks/modes, opaque files,
self-selected tolerances, and copied evidence from a different runtime all
fail closed.  Until an adapter supplies this live fingerprint, a future
`verified` card remains deliberately non-runnable rather than silently
accelerated.

A direct scalar predictor can satisfy the released-identity and solvent-state
checks without becoming a Route 4 potential backend.  It must be kept in the
`property_prediction` benchmark quantity, expose no PES or force task, and
cannot use a low property error to waive the thermodynamic-gauge, sampling, or
independent-holdout requirements above.

## Current frontier candidate: ConSolv

The ConSolv preprint is the strongest *scientific* match found so far: one
solvent-conditional MACE-style implicit potential, 66 non-aqueous solvents,
and a top-down free-energy objective between its vacuum and solution
potentials.  The paper describes `U_vac`, `U_solv`, and a differentiable
Zwanzig/BAR path, so it directly addresses the energy-gauge defect that
disqualifies force-only models.  Its reported panels are useful targets, not
certification for MAPLE: the internal Solv@TUM split is not a strict holdout;
the filtered external CombiSolv panel has different scope; and the published
figures do not provide a MAPLE-executable artifact or an independent sealed
final panel.

As of 2026-07-30, ConSolv **fails gate 1**.  The authors state that code and
data will be released on GitHub upon manuscript acceptance; no official
runtime, pretrained weights, architecture configuration, solvent-descriptor
bundle, or checksum has been released.  Route 4 must therefore retain it as
`paper_only` and must not reverse engineer, retrain, or mock its interface.

Primary sources:

- Zhang and Zavadlav, *ConSolv: Solvent-Conditional Machine Learning Implicit
  Solvent Potential*, arXiv:2606.24983 (v1, 2026-06-23),
  <https://arxiv.org/abs/2606.24983>.
- The source manuscript's “Data and Software Availability” section: code and
  data are deferred until acceptance (same archival source).

## Executable controls and their limits

| candidate | public pretrained artifact | solvent scope | free-energy status | Route 4 disposition |
| --- | --- | --- | --- | --- |
| GNNIS reference | yes | 39 upstream identifiers | force-matched PMF; energy gauge for absolute transfer is not established | sealed OpenFF+GNNIS solution-PMF control only |
| GNNImplicitSolvent Chem. Sci. 2024 | yes, distinct water checkpoint | water only | learned solution PMF for sampling; no released gas-to-water standard-state transfer protocol | separate water-PMF/MD audit candidate; it must not replace the 39-solvent GNNIS control |
| G-NequIP SMD-water | yes, tracked water-SMD checkpoint | water SMD only | geometry-level water energy; separate gas and water models have no released common gauge or free-energy protocol | pinned water-mechanics research control only; it cannot be converted to ΔG by subtracting model outputs |
| MACE-OFF24-SC | **no exact public checkpoint**; official result tables and protocol source are public | article panels in water and 1-octanol | the article defines a rigorous replica-exchange alchemical protocol, but the exact potential identity cannot be executed from the public release | aggregate-only static supporting-result audit; never substitute MACE-OFF23-SC or ordinary MACE-OFF24 |
| FeNNix-Bio1 | yes, base S/M checkpoints | explicit-water HFE is published; no validated nonaqueous selector or panel | Lambda-ABF and alchemical graph/charge operations are public, but the exact end-to-end paper protocol is not packaged as a MAPLE-ready workflow | checkpoint-pinned, high-priority water candidate blocked by absent original runtime and protocol validation |
| ML-for-charges PBE0-ESP | yes, XGBoost charge regressor | no solvent input; water only in the published downstream hydration protocol | atomic-charge parameters plus a separate explicit-water alchemical workflow; no potential, endpoint pair, or estimator | excluded from Route 4; an authorization-gated future Route 1 charge-provider study only |
| Organic_MPNICE + MLFF_HFE | no public Organic_MPNICE checkpoint; public generic HFE protocol only | water only | published explicit-water FEP/replica-exchange protocol, but the released source requires a user-provided lambda-aware TorchScript model | excluded from Route 4; no model may be substituted for the proprietary release |
| ABCG2/GAFF2 | yes, official AmberTools fixed BCC parameters | water and diverse neutral organic-solvent protocols reported | explicit-solvent gas/solution alchemical simulation is possible, but the charge model is not an MLIP or implicit potential | direct runnable Route 1 physics control only; never a Route 4 substitution |
| AniSolv compact | yes | 21 explicitly validated identifiers | geometry-level scalar `E_solv - E_gas` correction and exact zero vacuum gate; the upstream force path fails rotation-covariance/finite-difference consistency and no MAPLE thermodynamic protocol is published | sealed single-point energy control only |
| LSNN-v1 | yes, but upstream runtime/domain is incomplete | water only | lambda derivatives target the correct class of quantity, but the MAPLE compatibility pilot is not upstream reproduction and fails its pilot maximum-error gate | water-only research control, disabled by default |
| AIMNet2-CPCMS v2 | yes | fixed THF metadata | native solution PES; no released gas/solution pair or FE protocol | neutral-singlet mechanics probe only |
| C3Net | yes | 103 upstream solvent identifiers | direct scalar solvation-property prediction; no energy surface, endpoint pair, or sampled estimator | SDF-only property-prediction sidecar, never an ordinary calculator or absolute-solvation backend |
| CIGIN | yes | one syntactically accepted solvent SMILES; complete multi-solvent training coverage is unverified | direct scalar solvation-property prediction; no energy surface, endpoint pair, or sampled estimator | CPU-only SMILES property-prediction sidecar, never an ordinary calculator or absolute-solvation backend |
| MoletoSolv | **no serialized trained estimators/scalers**; source, data, and precomputed predictions only | 92 public solvent descriptors, while the ionic code path hard-codes four solvents | descriptor-based direct scalar ΔGsolv regression; no PES, forces, endpoint pair, or sampled estimator | literature baseline only; unchanged use would require prohibited local training/artifact reconstruction |
| SolProp_ML Gsolv | yes, ten-member Zenodo bundle | neutral pair input; CombiSolv-Exp reports 291 solvents but exact checkpoint record membership is unavailable | direct scalar gas-liquid dGsolv at 298 K / 1 M; no energy surface, endpoint pair, or sampled estimator | artifact-pinned candidate only: current unmodified official runtime has unavailable dependencies, so no adapter is exposed |
| SolProp-mix Exp | yes, selected ten-member Zenodo ensemble | neutral solutes/nonionic liquids; pure, binary, and ternary mixtures are in the published task, but MAPLE coverage is unverified | direct infinite-dilution dGsolv scalar prediction; no energy surface, force field, endpoint pair, or sampled estimator | artifact-pinned mixture-property candidate only: current unmodified official runtime has unavailable dependencies, so no adapter is exposed |
| Schake GNN v2 | yes, protein-backbone checkpoint | no solvent selector; only CA/C/N backbone representation coupled to GBn2 | learned structural correction for protein conformational landscapes; no direct solvation estimator | artifact-pinned protein research control only: upstream dependencies are unavailable and its own README says it is not production-ready |
| ConSolv | **no** | 66 reported non-aqueous solvents | published mathematical form is suitable | watchlist only until official code and weights exist |
| TWIN | **no** | water | reported implicit-water potential, but its v1 manuscript defers simulation data and code until publication | watchlist only; no runtime may be reconstructed |

### GNNImplicitSolvent Chem. Sci. 2024 is a separate water control

The official `Chem._Sci.,_2024` tag resolves to
`b05075cbcd8cf0ee92e48605587619f7640b0a65` and includes the already trained
`MachineLearning/trained_models/GNN.model` artifact (247,228 bytes;
SHA256 `393917c43ba5f527eba8a3e8cc3b4f548e234e22eb4f19b46ca6bea6aefbad06`).
The upstream README identifies this as a pre-trained GNN for implicit-water
simulation.  It is not byte- or runtime-identical to the 2025 39-solvent
GNNIS checkpoint already sealed in MAPLE, so a silent checkpoint substitution
would invalidate the existing model card and benchmark identities.

The candidate is scientifically useful only as a narrow water solution-PMF/MD
control.  Its released simulation command accepts a solute SMILES and model
path but no solvent selector; the branch's public description and its model
configuration identify water.  It supplies neither a released standard-state
gas-to-water transfer protocol nor a converged absolute-solvation estimator,
so it cannot satisfy the Route 4 absolute-solvation gate.  Moreover, the
unmodified upstream runtime requires the `torch_geometric` family and its
example emits an OpenMM-Torch artifact on CUDA.  Those packages are absent in
the current environment; MAPLE therefore records the exact candidate but does
not patch the upstream source, substitute a CPU implementation, or expose a
placeholder calculator.  This is a capability boundary, not a reason to
relax the multi-solvent main route.

### G-NequIP SMD-water: released water checkpoint, no transferable ΔG path

The current official `deepPotential` source at
`c6d43de9630c98486ff7ed904c4c54cdeb7e14ad` tracks both
`all_NNP_MODELS/G_NequIP_smdW.pth` (11,436,716 bytes; SHA256
`5615c0002a24ee3485368b59721198b2ca2e0432c1c6f7b73b5a31f6fe662484`) and a
separate gas-phase `G_NequIP.pth`.  The associated Zenodo archive explicitly
labels this as an implicit **aqueous SMD** evaluation, not as a
solvent-conditioned or multi-solvent potential.  It is therefore a real public
water mechanics control rather than paper-only evidence.

MAPLE also pins the small official `solvation_free_energies.zip` result archive
(474,332 bytes; SHA256
`76522218fcadc921706ba489ab0b7ff1c5e28b335029f07ac7c5ce5877dae055`).  Its
388-record intersection of supplied FreeSolv experimental, gas-NNP, and
SMD-NNP tables reproduces the source's experimental-panel central values:
`MAE = 1.082689`, `RMSE = 1.425792`, and `R = 0.913146 kcal/mol`-scale
comparison.  The audited static maximum absolute error is `5.445611 kcal/mol`.
This verifies only supplied tables, not the model runtime, and it does not
establish uniform accuracy or an independent holdout.  The bundled plotting
script references legacy filenames that are absent from this archive and its
default bootstrap count differs from the README, so MAPLE reproduces central
statistics directly from the pinned member tables without claiming the source
confidence intervals were rerun.

It nevertheless fails the Route 4 thermodynamic gate.  The source itself
estimates `Delta G_solv = E_SMD_NNP - E_gas_NNP` from single points on separate
supplied geometries.  Two independently trained energy models can differ by
arbitrary molecule-dependent constants; that subtraction does not define a
solvation free energy without a common gauge.  The source supplies neither a
common lambda-conditioned Hamiltonian nor a sampled estimator, standard-state
convention, convergence diagnostic, or record-complete training ledger.  In
addition, the original loader imports `nequip.ase`, while the upstream
environment requests `nequip==0.11.1` and that package is absent locally.
MAPLE neither installs it without authorization nor rewrites the loader.
G-NequIP therefore remains a pinned, blocked, water-only mechanics control—
never a multi-solvent model or absolute solvation backend.

### MACE-OFF24-SC: rigorous article protocol, unreleased checkpoint

Moore, Cole, and Csányi report a theoretically rigorous explicit-solvent
replica-exchange alchemical protocol for a soft-core-equipped MACE-OFF24-SC
potential.  The exact model would be a scientifically relevant Route 4
protocol backend: it defines a continuous lambda Hamiltonian, samples 16
replicas, and evaluates hydration and octanol solvation free energies rather
than subtracting unrelated single-point energy gauges.

Gate 1 nevertheless remains closed.  The clean public `mace-md` source at
`c3056287622ba18f9b905e9df45affdff46cb147` supplies the protocol machinery but
points to the distinct public MACE-OFF23-SC release.  It does not pin the
MACE-OFF24-SC checkpoint used in the article.  The ordinary MACE-OFF24 models
and MACE-OFF23-SC are different potential identities and cannot be substituted.
The current environment also lacks `openmmtools`, but installing that dependency
would not repair the missing checkpoint identity.

The official supporting CSV (2,650 bytes; SHA256
`0b44cb14b4d1c97413c7f6fad45655ddf2b15b44370fa0f89b049eb9af19ff6a`)
and PDF (1,704,085 bytes; SHA256
`7c675c6657a3b5b92a317ea8a3c17241f5738cc44113440caefe882227b535b1`)
are separately pinned under their Figshare article/file identities.  Both are
CC BY-NC 4.0, so MAPLE freezes aggregate results and record-set digests rather
than redistributing their row-level labels and predictions.

The exact rounded 36-record FreeSolv table recomputes to `MAE = 0.763889`,
`RMSE = 0.834389`, and maximum absolute error `= 1.600000 kcal/mol`; the latter
fails the strict `< 1.5 kcal/mol` target.  The separate 10-record MNSol octanol
table recomputes to `MAE = 0.474000`, `RMSE = 0.587929`, and maximum absolute
error `= 0.930000 kcal/mol`.  These panel identities are not pooled.  The
rounded hydration CSV also does not reproduce the article Table 1
`MAE = 0.69` and `RMSE = 0.80`.  Its recomputed-minus-reported gaps are
`0.073889` and `0.034389 kcal/mol`, both larger than a conservative
`0.03 kcal/mol` combined displayed-value bound.  Rounding alone is therefore
mathematically insufficient; the underlying source-analysis discrepancy
remains unknown.

`run_mace_off24_sc_supplied_result_audit.py` therefore records a static
official-table audit only.  It performs no checkpoint inference or alchemical
rerun, proves no training isolation or independent extrapolation, and cannot
admit MACE-OFF24-SC into the runtime registry.

### FeNNix-Bio1: public weights and GPU/Lambda source, execution still blocked

FeNNix-Bio1 is the strongest newly verified public water-alchemical candidate.
The official FeNNol-PMC revision
`83f299b81c1d62e2a15c892280559a7c0cc2fac3` publishes two base checkpoints:
the 29,772,732-byte small model with SHA256
`82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07`
and the 38,124,300-byte medium model with SHA256
`5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a`.
The model artifacts use the academic-only, non-commercial ASL.  Separate
`finetuneIons` checkpoints exist upstream, but they are distinct potential
identities and cannot replace these base weights silently.

The LGPL-3.0 FeNNol source at
`d62b8740343b803a2b864140ec79e347f8ba034e` loads the `.fnx` artifacts and
provides an ASE energy/force/stress calculator.  The same original source
contains the published alchemical group operations: lambda-scaled graph
switches, softcore distances and repulsion, and separate solute/solvent charge
equilibration.  Tinker-HP revision
`384bb7d451f85f2ea825dabd888e64ad4448674b` supplies the combined original
GPU bridge: its pinned `v1.3/GPU/source/mlinterface.py` loads arbitrary FeNNol
`.fnx` files, requests gradients with respect to `alch_elambda` and
`alch_vlambda`, passes the alchemical group and ligand-charge inputs, and
returns both derivatives.  Generic Lambda-ABF/Colvars documentation is in the
same exact source tree.  This is real endpoint and estimator plumbing rather
than a scalar property regressor.

The combined source does not, however, establish the exact paper run.  No path
named for FeNNix-Bio1 or FreeSolv exists in the exact Tinker-HP tree, and the
shipped Deep-HP example selects `ani2x.fnx`.  That path-name audit is narrow:
it does not claim that no external, author-held, or generically named artifact
exists.  What remains unverified as one reproducible bundle is the paper's
system building, lambda schedule, sampling controls, convergence evidence,
standard-state treatment, Bio1 checkpoint binding, and row ledger.

For the medium model, the v4 article starts from 642 FreeSolv molecules,
discards 17 systems that react or change protonation during decoupling, and
reports 625 retained water HFEs at `MAE = 0.70`, `RMSE = 0.98 kcal/mol`.
The 452-record H/C/N/O/S/P subset is reported at `MAE = 0.59`,
`RMSE = 0.82 kcal/mol`.  The halogen cases require the article's separate
solute/solvent charge equilibration and retain larger error.  No public
row-complete prediction table establishes the maximum absolute error or a
strict independent holdout, so the average values cannot satisfy Route 4
acceptance.  The paper also mentions a 37-record comparison with
MACE-OFF24-SC, but it is not the same identity as the official 36-record
FreeSolv supporting table audited above and is not cross-ranked here.

No single existing MAPLE environment contains the unchanged stack:
`maple-resolv` has CPU-only JAX 0.4.23 but not FeNNol, ASE, or OpenMM, while
the OpenMM-bearing environments have no JAX or FeNNol.  The original Deep-HP
GPU build also needs FeNNol's bridge dependencies and the Tinker-HP GPU
compiler toolchain; the required NVIDIA HPC compiler is not installed.  MAPLE
therefore pins both weights and both runtime revisions without installing
dependencies, splicing incompatible environments, porting the JAX runtime, or
adding a placeholder calculator.

The Tinker bridge statically sets JAX matrix multiplication to `highest`, but
passes coordinates and lambda variables as float32 and exposes no bridge-level
precision switch.  FeNNol's generic ASE/MD code separately offers float64
controls and warns that default float32 matrix multiplication may invoke
float16 operations.  Static source inspection cannot prove no-loss GPU
behavior.  No paired reference/GPU energies, forces, virials, lambda
derivatives, trajectories, free energies, or uncertainties have been
generated; consequently the accelerator is explicitly blocked by gate 7.
Ordinary SP/OPT/frequency/TS/IRC/MD and water-HFE tasks remain disabled until
the unchanged runtime can be executed, GPU parity is proven without reduced
precision or looser convergence, and each task contract is validated.  No
nonaqueous or mixed-solvent literature panel currently justifies a
multi-solvent claim.

`run_fennix_bio1_release_audit.py` regenerates the identity-only
`benchmarks/fennix-bio1-release-audit-2026-07-30.json` from the two checkpoint
files and exact FeNNol/Tinker-HP Git trees.  The artifact records the public
GPU/Lambda source as present while keeping protocol execution, GPU parity,
maximum error, strict holdout, multi-solvent evidence, and Route 4 acceptance
false.

### ML-for-charges PBE0-ESP: public pretrained charge regressor, not Route 4

Hilfiker *et al.* release their code under MIT at
`mathilfiker/ml_for_charges` revision
`e8407cc7e500d89cf66ed0673d6cd7421ab5637d`, including the trained
`charge_prediction/model/xgb_chgs.json` artifact (91,849,931 bytes; upstream
Git blob `8d8ae6e35496ed5587685fa3a24c9a598dec2565`).  The predictor uses
MACE-OFF23-large atomic descriptors to emit PBE0-D3(BJ)/def2-TZVP-quality
atomic charges.  That establishes public source-and-artifact provenance, but
does not clear gate 1: the upstream Git object is content-addressed while a
separate SHA256 has not been frozen here, because this is a charge
parameterization rather than a solvent-conditioned energy model.

The unmodified inference example imports `MACECalculator`, `xgboost`, and
ASE, then explicitly loads `MACE-OFF23_large.model`.  MAPLE currently has the
MACE runtime and that named MACE model file, but no installed environment has
`xgboost`; it does not install the missing official dependency, rewrite the
loader, or regenerate the trained regressor.  This runtime gap is recorded
rather than worked around.

More importantly, the model accepts no solvent identity or composition.  Its
reported absolute hydration calculation applies the predicted charges inside a
separate gas-to-water explicit-solvent alchemical MD protocol.  The authors'
30-molecule FreeSolv panel reports `RMSE = 1.69 kcal/mol` for the
Boltzmann-percentile charge procedure versus `3.05 kcal/mol` for AM1-BCC, but
that is their protocol result—not a MAPLE reproduction, a strict holdout, or
a multi-solvent comparison.  It exposes neither Route 4 `U_solv(R, S)` nor a
matching `U_vac(R)`, force task, lambda path, or estimator.  The candidate is
therefore explicitly excluded from Route 4.  If separately authorized, it may
be assessed only as an immutable charge provider in a Route 1 explicit-water
study with its own fully recorded protocol and independent experimental panel;
it must never be fitted or used to calibrate outcomes.

### Organic_MPNICE plus MLFF_HFE: published protocol, unavailable model

Xie *et al.* report sub-kcal/mol average hydration error for 59 organic
molecules using the proprietary `Organic_MPNICE` MLFF in an explicit-water
FEP/replica-exchange protocol.  The article identifies that MLFF as part of
the Schrödinger Suite 2025-3 release.  The authors' public
[`MLFF_HFE`](https://github.com/leifjacobson/MLFF_HFE) repository at
`872ed7b5f5dcf45696090604383d777acbfd8d70` supplies a generic OpenMM-Torch
protocol, not that model: its command requires a user-supplied `model.pt`
whose force module implements the prescribed `lambda_lj`, `lambda_nn`, and
solute-temperature inputs.  It tracks no Organic_MPNICE or alternate
pretrained model weight.

The protocol is useful evidence that a rigorous explicit-water route can be
constructed when a suitable lambda-aware MLFF is released, but it is water
only and has no solvent selector or Route 4 implicit `U_solv(R, S)`.  Its
unmodified package also needs `openmmtools`, which is absent from all MAPLE
environments (although OpenMM, OpenMM-Torch, and torch are present in the
main environment).  MAPLE therefore neither installs it nor inserts a MACE,
AniSolv, or other unrelated checkpoint into the user-model interface.  The
repository's top-level license file is CC-BY-4.0 while its `pyproject.toml`
declares MIT, an additional source-identity ambiguity to resolve before any
reuse.  This remains publication/protocol provenance only, never a MAPLE
runtime, a Route 4 candidate, or an acceptance score.

### ABCG2/GAFF2: strong explicit-solvent control, deliberately outside Route 4

ABCG2 is not an MLIP: it is the authors' published AM1-BCC-GAFF2 correction
set, distributed through official AmberTools as the immutable Antechamber
method `-c abcg2`.  The local `maple-ambertools` environment uses AmberTools
26.0 and contains `BCCPARM_ABCG2.DAT` (15,057 bytes; SHA256
`7412ef2c6283aa96a12ab44a881a2e755bbdaa4318a55bec5cebc7895fe8a94f`) and
`ATOMTYPE_ABCG2.DEF` (11,131 bytes; SHA256
`7fb77a85268789c59995540ecadc0322d67fb6de8c9dd3e59e8fc460c483b6f4`).  Two
independent unmodified `antechamber -c abcg2 -at gaff2` runs on neutral
methanol produced byte-identical MOL2 output with zero net charge.  This
verifies only the official charge-assignment entry point; it is neither an
accuracy benchmark nor a solvation calculation.

The literature is unusually strong as a *physics control*: the 2025 study
reports `RMSE = 0.99 kcal/mol` over 642 FreeSolv solutes and `0.89 kcal/mol`
over 2,068 MNSol pairs, while the original multi-solvent paper reports
`MUE = 0.51` and `RMSE = 0.65 kcal/mol` for 895 neutral organic
solvent--solute pairs.  Those claims depend on separately built GAFF2 solvent
and solute topologies plus gas/solution alchemical simulations, and their
published panels have no MAPLE-held-out record ledger.  They are evidence to
reproduce under a separate Route 1 protocol—not a MAPLE score and never a
license to pool FreeSolv with MNSol.

ABCG2 has no solvent selector, implicit `U_solv(R, S)`, force/PES model, or
trained MLIP checkpoint.  Route 4 therefore must not expose it as `#model`,
reuse its reported errors for a Route 4 candidate, fit further BCC terms, or
use it to calibrate another route.  It remains the cleanest no-retraining,
official-runtime baseline if a future explicitly authorized Route 1
multi-solvent TI/BAR/MBAR study is separately planned and identity-locked.

AniSolv compact passes the released-identity gate: its source revision,
bundled checkpoint, SHA256, MIT status, 21-solvent whitelist, exact vacuum
gate, and scalar single-point runtime are pinned.  Its own model card describes
a geometry-level additive correction, not a converged partition-function
estimator with standard-state metadata.  A source-bound CPU float64 central-
difference audit found orientation-specific upstream force failures for the
exact sample water and methanol geometries.  The same geometries pass after a
fixed rigid rotation while their energies remain invariant, so MAPLE exposes
the correction only for scalar single-point energy and rejects every force-
derived task.  The row-complete evidence is frozen in
`benchmarks/anisolv-compact-force-consistency-audit-2026-07-30.json`.  The
ordinary-calculator path is the sealed
`anisolv-uma` composition: a user-authorized, hash-pinned UMA-S-1P2
compatibility checkpoint plus the exact public compact AniSolv checkpoint.
UMA access is gated upstream, so MAPLE neither downloads nor bypasses that
gate.  It cannot emit gradients or an absolute solvation free energy, run
optimization/scan/TS/IRC/frequency/MD, accept charged/open-shell inputs, select
additional solvent descriptor entries, or substitute either checkpoint.

The exact compact checkpoint and official source revision have been freshly
executed for every one of those 21 enabled identifiers on the same neutral
methanol geometry, with finite scalar energy in each case.  Upstream force
arrays from that smoke run are not admitted.  This closes only the narrow
runtime-coverage claim for the geometry-level scalar term; it supplies no
experimental ΔG values, sampling diagnostics, or accuracy claim and therefore
does not change its admission disposition.

The upstream examples do not close this remaining gate.  MAPLE now reproduces
the exact six rows from `H2O_single_point.py` in
`benchmarks/anisolv-compact-upstream-sample-audit-2026-07-30.json`, using the
pinned public checkpoint and official source on deterministic CPU float32
`default` inference with all reduced-precision and fast modes disabled.  The
six-row panel is below the mandatory ten-record/ten-functional-group accuracy
floor, so it emits no Route 4 accuracy metrics and does not calculate
experimental error, MAE, RMSE, or maximum error.  This is intentionally
non-accuracy development evidence:
the sample multiplies a raw geometry-level correction by an energy conversion
factor and prints it beside experimental hydration free energies, while exact
record IDs, training overlap, a sampled thermodynamic cycle, and standard-state
metadata are unavailable.  Its row-complete CPU forces are a reference only;
the separate force-consistency audit rejects them for MAPLE execution.  No GPU
scalar-energy parity has been run or admitted.

The separate `H2O_dGsolv.py` is a one-water harmonic-cycle example with a
separately obtained UMA base potential, but it consumes the rejected upstream
force path for optimization and vibrations.  It also has no released multi-
solvent benchmark identity, sampled-state/convergence diagnostics, standard-
state correction, or independent external confirmation panel.  Neither
upstream example is an acceptance benchmark or independent extrapolation
panel.

The decision remains deliberately conservative: current Route 4 does **not**
have a single released pretrained backend that proves every requested
combination of absolute solvation energy, multi-solvent coverage, and the full
MAPLE task surface.  A future official ConSolv release must still be audited
against the seven gates above rather than reconstructed or trained locally.

### C3Net: released multi-solvent property sidecar, not a potential

C3Net supplies official source and pretrained artifacts for a direct scalar
prediction across 103 named solvents.  MAPLE pins the official source revision,
both required artifact hashes, the upstream solvent registry, and its one- to
five-record same-solute inference path.  The multi-record path preserves the
paper's arithmetic-mean convention; it is not a configurational free-energy
estimator.  Its compatibility shim is process-local:
the unmodified 2023 upstream dataset loader uses NumPy's removed `np.int`
alias, whose historical value is the native Python `int` on the supported
runtime.  MAPLE neither changes the source tree nor the checkpoint.

The paper reports `0.270 kcal/mol` MAE for a multi-conformer random external
validation split (`0.349 kcal/mol` for a single conformation), but the released
evidence does not provide complete record-level training accounting for the
fixed checkpoint.  Specifically, the article describes 5,718 solvation-free
energies for 890 solutes in 103 solvents and a random 80/20 split, whereas the
only public `Train_validation_idx_1.txt` located in the official repository
contains 41 training and 10 validation indices.  The repository also does not
publish a paper-scale canonical record ledger.  Those figures are therefore
neither a MAPLE validation nor an acceptance comparison.

This is a proof boundary, not merely missing convenience metadata.  A strict
never-used panel requires an exact set-intersection test against canonical
record identities from the complete checkpoint training and validation pool.
Without that set, FreeSolv, Solv@TUM, dGsolvDB, or any other public
experimental corpus can overlap invisibly and cannot be certified independent
by assumption.  C3Net may provide a separately labelled
`property_prediction` development baseline once an immutable panel identity is
recorded; it cannot be selected as a sampled, gauge-defined absolute-solvation
calculation, a solvent-aware MAPLE PES, or a strict extrapolation score.

MAPLE's fixed water-only FreeSolv-10 development pilot is recorded in
`benchmarks/c3net-freesolv10-2026-07-30.json` with a distinct property-prediction
identity, deterministic ETKDGv3 + UFF single-conformer policy, and
`overlap_unknown` leakage audit.  Its ten records bind ten chemistry classes,
not ten audited functional groups.  It is therefore runtime/interface smoke
with `metrics = null`, not an accuracy result.  The panel is water-only and
the C3Net training ledger is incomplete; it must never be generalized or
compared to a Route 2 FreeSolv calculation identity.

A broader frozen development panel is also recorded in
`benchmarks/c3net-dgsolvdb1-11solvent-2026-07-30.json`.  It uses 22 direct
experimental dGsolvDB1 records across acetone, acetonitrile, benzene, carbon
tetrachloride, chloroform, dichloromethane, dimethylsulfoxide, ethanol,
methanol, tetrahydrofuran, and toluene.  Its selection is output-blind, each
solute receives five RDKit ETKDGv3 + UFF conformers, and the final scalar is
the paper's arithmetic conformer mean.  No functional-group taxonomy was
predeclared for these rows, so the frozen result is now runtime/interface
smoke with `metrics = null`, not a precision panel.  Its complete 22/22
execution coverage is useful only for runtime compatibility; it cannot support
accuracy, ranking, admission, or generalization.

### CIGIN: FreeSolv-trained scalar sidecar, not verified multi-solvent coverage

The official CIGIN source at `9990f826f1a17590f1925dfd470423633275840c` and
its `weights/cigin.tar` checkpoint (SHA256
`79f07c6475f8adfce6913b61a7e1e911d5a049f031954aeeb1eae986d2ce4cce`) are
public and reproducibly executable as a direct solute-SMILES/solvent-SMILES
scalar predictor.  The model authors identify the training source as FreeSolv
and report `0.91 kcal/mol` RMSE.  They do not publish a complete checkpoint
training-record manifest, however, and the released artifact contains no
auditable multi-solvent coverage ledger.  A solvent SMILES accepted by the
network is syntactic input support, not evidence that it belongs to a trained
or validated solvent domain.

MAPLE therefore pins the clean original source and checkpoint, rejects charged,
radical, multi-fragment, unsupported-element, and degree-greater-than-four
inputs before the subprocess, and records only a direct `kcal/mol`
`property_prediction`.  The child process forces CPU before imports because
the unmodified upstream CUDA path allocates interaction tensors on CPU while
its learned tensors are on CUDA.  This is an explicit runtime precondition,
not a source patch or a replacement implementation.  CIGIN exposes no PES,
forces, conformer policy, optimizer, standard-state convention, endpoint pair,
or free-energy estimator.  It is forbidden from `#model`, geometry tasks,
absolute-solvation ledgers, and any multi-solvent extrapolation claim.

MAPLE's fixed water-only FreeSolv-10 development artifact is
`benchmarks/cigin-freesolv10-2026-07-30.json`.  Its ten predeclared chemistry
classes are not an audited ten-functional-group taxonomy, so it is
runtime/interface smoke with `metrics = null` and no experimental error
calculation; no accuracy or generalization claim follows.
Because FreeSolv is named training provenance but record membership is
unresolved, the leakage audit is `overlap_unknown`, not a strict holdout.  It
is development-only negative evidence and may not be compared to C3Net or
Route 2 values without a fully matched identity.

### AtomicESE: packaged scalar release, audit only

The official AtomicESE article
([DOI 10.1002/jcc.70104](https://doi.org/10.1002/jcc.70104)) describes a
dedicated scalar model for solvation free energies of molecules and ions in
organic solvents. It specifies a `13 x 30 x 28 x 1` atomic network with 1,258
weights and 59 biases (1,317 adjustable parameters) and advertises Supporting
Information containing scaling factors, weights, biases, and calculated
predictions. The advertised `jcc70104-sup-0001-supinfo.zip` bytes are currently
unavailable to the audit because the publisher returns HTTP 403. The article
and SI description thus cannot substitute for a hash-pinned checkpoint or
record-level prediction ledger.

The separate exact-release audit pins official Git revision
`31e643c7e8974497c78fc2fb6f3c17778d61fa10` and tree
`e08ce8bfaf73ca9c35c2b85929067e639d7bf048`. In that exact pinned official Git
tree only, AtomicESE is distributed as packaged x86_64 Linux/Windows
executables plus a README and one sample input. No repository license file,
source, separate checkpoint, checkpoint ledger, or record-level training/split
membership ledger is present in that tree; this makes no absence claim about
the article or currently unavailable supporting-information bytes. Its
sandboxed repeated sample verifies a scalar CLI boundary, not chemical
accuracy or performance. Binary sample timing is not speed evidence.

The audit found no selected GPU/CUDA ASCII markers and verified no GPU
runtime. That is not proof of CPU-only capability. AtomicESE has no documented
or verified packaged GPU path, GPU capability remains unverified, and GPU
admission is false.

AtomicESE is consequently an audit-only dedicated scalar multi-organic-solvent
candidate, not an MLIP, PES, force model, calculator, or sampling protocol. It
has no MAPLE integration, model card, adapter, accuracy panel, GPU admission,
or matched-QM timing. Its quantitative standard state and record-level split
membership remain unresolved. It is rejected and blocked until a formal
accuracy panel passes at least 10 records spanning at least 10 distinct,
predeclared primary functional groups. Only then may a same-task,
same-precision end-to-end matched-QM benchmark run at least three complete
repeats; the mandatory gate is
`median(QM full-task seconds) / median(AtomicESE full-task seconds) > 1`.
Equality or a slower candidate remains rejected.

### MoletoSolv: relevant scalar literature, no pretrained release

The current official MoletoSolv source revision
`04e6229ec414dcb66fa46c1e26f89f3ea91ba5e2` supports a scientifically relevant
descriptor-based study of neutral, anionic, and cationic solvation in aqueous
and nonaqueous solvents.  The paper reports MAE values of `0.44`, `1.72`, and
`1.60 kcal/mol` for those three charge classes.  The public solvent descriptor
table contains 92 solvents, although the released ionic code path hard-codes
only water, acetonitrile, DMSO, and methanol.

This is not an already-trained runtime release.  The official tree contains
source, datasets, solute geometries, and a precomputed prediction table, but no
serialized estimator, scaler, feature-selection artifact, tag, or release.
The code creates and later loads local `joblib` artifacts, while the neutral
`model/neutral/dataset.csv` at the pinned revision is empty.  Executing new
predictions would therefore require local training or reconstruction, both
forbidden on this route.  The public sources also provide no immutable
record-level split ledger, maximum-error table, independent experimental
holdout, production standard-state contract, or no-loss GPU parity evidence.
MoletoSolv remains a literature baseline only: it is not a pretrained
property sidecar, MLIP/PES, force model, or calculator candidate.

### SolProp_ML Gsolv: authentic multi-solvent model bundle, runtime not admitted

SolProp_ML is the strongest currently located *property-only* multi-solvent
candidate: its official Zenodo v1.2 archive is 268,574,239 bytes with SHA256
`f66bb046bc1d5b8471d36436e485d0cb9d95fb7111aa2db4deb85fdbbea6b766`, and
contains ten `Gsolv/model_Gsolv_{0..9}.pt` members totalling 36,539,307 bytes.
The associated source revision is
`d345646c827b4e577e1dc485ae76d7b6d7f266c3`.  The archive's own README labels
the model a transfer-learned direct gas-liquid dGsolv predictor at 298 K and
1 M, with CombiSolv-Exp describing 291 neutral solvents.  This is a genuine
pretrained artifact, but not a potential-energy surface or an absolute-
solvation workflow.

MAPLE intentionally does not expose it yet.  The clean upstream high-level
path imports `tensorboardX` and `chemprop_solvation` even for Gsolv-only
loading; neither is installed in this environment, and the public
`calculate_solubility` entry point also imports `CoolProp`.  Installing those
dependencies is forbidden without explicit authorization, while copying only
the loading fragments would be a source reimplementation.  Moreover, the
published provenance includes FreeSolv, CompSol, and Abraham records but no
complete exact-ensemble training ledger, so it cannot provide an independent
holdout or a current acceptance score.  It remains an identity-pinned blocked
candidate rather than a shortcut around either restriction.

### SolProp-mix Exp: released mixture dGsolv ensemble, runtime not admitted

The current official SolProp source revision
`80043ce09eb8802517c35b59254f8e9c181f2dac` instructs users to copy exactly the
`SolPropmixExp` folder from the authors' Zenodo record into the original
runtime.  MAPLE has verified the record's complete `Files.zip` archive
(288,947,625 bytes; SHA256
`670915e5bf86d5457bc2f43301e5a02e77a66b529535fcd0eddfadeb72ed73a5`) and its
selected ten-member ensemble (36,170,142 bytes).  This is an authentic,
already-trained direct scalar dGsolv artifact for neutral supported solutes in
nonionic liquid pure or mixed solvents.  It is a stronger task match than
pure-solvent pair predictors because its mathematical input includes solvent
identity and composition, but it is still not a molecular potential, force
model, endpoint pair, or sampled free-energy estimator.

The paper reports a `0.25 kcal/mol` MAE and `0.37 kcal/mol` RMSE for
nonaqueous mixed-solvent predictions.  Those are author-level results, not a
MAPLE reproduction: the corresponding record identities, exact training
overlap, and a never-used experimental extrapolation panel have not been
established here.  The official final weights embody the authors' historical
transfer-learning and fine-tuning procedure; MAPLE may only execute that
unchanged released ensemble, never fit, fine-tune, calibrate, or otherwise
modify it.

The pinned static candidate audit covers only the workbook worksheets
`Not used - From 3-comp VLE` and `Not used - From 4-comp VLE`.  It verifies
30,184 release rows and 279 declared solvent identities without decoding any
experimental `Gsolv` value or reading model predictions.  This is not an
independence certificate: the release contains 43 duplicate physical-record
identities, 48 rows with incomplete core citation provenance, and no complete
join against the candidate training/test families.  Zero-fraction declared
components remain nominal source-cardinality provenance but are omitted from
the physical mixture identity, exposing one duplicate across the two source
worksheets.  Until the cross-dataset joins are complete, the only allowed
status is candidate-only with overlap pending.

The paper-level boundary is now frozen separately from that candidate claim.
Its final workflow pretrains on CombiSolv-QM plus BinarySolv-QM, fine-tunes on
CombiSolv-Exp, and tests on BinarySolv-Exp plus TernarySolv-Exp with
solute-level splitting.  The reported experimental-test solutes are excluded
from CombiSolv-Exp but retain some overlap with the synthetic COSMO-RS
families.  Those exclusions do not mention or certify the two `Not used`
worksheets, so exact-record and solute-structure joins remain mandatory before
any label is consumed.

The current environment cannot execute the clean upstream path.  Importing
its `PredictArgs` aborts before prediction because `Tap` is absent; importing
its original `load_checkpoint` aborts because `memory_profiler` is absent;
model construction also depends on absent `DGL`.  The official requirements
further name `tensorboardX` and `CoolProp`, which are also absent.  Installing
or substituting those dependencies requires explicit authorization, and
copying a reduced loader would be a prohibited reimplementation.  Therefore
SolProp-mix remains a precisely pinned, blocked property-sidecar candidate:
it is never `#model`, a geometry/PES/MD task, or an
absolute-solvation-protocol ledger backend.

### Schake GNN v2: released protein correction, not a general solvation backend

The current official Schake repository has superseded the preprint's
future-release statement: commit `62d8e2381edb2a84822f33530ed9364ea2543cd0`
contains `Schake/v2/Schake_trained_weights.pt` (212,619 bytes; SHA256
`1be539766161b396f2dbaeabf87cf26b6f98b8c91ac26a47d5263e385ecec8b4`) plus a
JIT-tracing and OpenMM-Torch example.  This is a valuable strictly pretrained
protein task-control candidate, but its physical contract is narrow: backbone
CA/C/N only, with ff14SBonlysc and GBn2 plus the paper's fixed gamma `0.175`.
It has no solvent identity, direct solute--solvent dGsolv, standard-state
conversion, or small-molecule calculator contract.

The unmodified source imports `torch_scatter` and `torch_geometric`, which are
absent from all installed MAPLE environments.  Installing them without
authorization or copying the model around those imports would violate the
dependency and provenance rules.  The upstream README also explicitly labels
the model not production-ready.  Therefore Schake remains a pinned protein
research control; it may not replace the general Route 4 backend, support
multi-solvent extrapolation, or enter any absolute-solvation ledger.

## Research sweep and route assignment

The screen below separates a candidate's scientific promise from what MAPLE
may execute today.  It deliberately assigns models to their narrowest justified
route instead of treating a reported scalar benchmark as a universal
calculator/free-energy capability.

| candidate | strongest supported capability | missing gate | MAPLE assignment |
| --- | --- | --- | --- |
| ReSolv | a sampled, water-only hydration free-energy path with an implicit ML potential | multi-solvent release and a general calculator contract | Route 3 `#solvfe` only; never an ordinary Route 4 `#model` backend |
| GNNIS | 39 solvent-conditioned, upstream OpenFF+GNNIS solution Hamiltonians for mechanics/MD | a transferable absolute-energy gauge and a record-isolated solvation-free-energy panel | sealed Route 4 reference-Hamiltonian control, not an arbitrary MLIP correction |
| G-NequIP SMD-water | public SMD-water geometry-level checkpoint with a separate gas checkpoint | water-only scope, unavailable original NequIP runtime, and no common energy gauge, thermodynamic path, or independent ledger | blocked water mechanics control; never ΔG from single-point model subtraction or a multi-solvent backend |
| MACE-OFF23-SC | exact public soft-core checkpoint; live official-MACE float64 CPU energy/force execution for full-coupling explicit systems | original OpenMM-ML/OpenMMTools lambda protocol, replica exchange, MBAR/uncertainty, independent 10-group experimental panel, and literal CPU/GPU parity | CPU-only explicit-system PES endpoint; never an absolute-solvation result or a substitute for MACE-OFF24-SC |
| MACE-OFF24-SC | rigorous explicit-solvent alchemical protocol and official rounded water/octanol result tables | exact public checkpoint identity, original runtime reproduction, record-isolated panel, and hydration maximum-error gate | static negative-admission evidence only; never substitute MACE-OFF23-SC |
| FeNNix-Bio1 | public base S/M energy-and-force checkpoints plus original alchemical graph and charge operations | installed unchanged runtime, exact paper protocol bundle, maximum-error evidence, independent ledger, and nonaqueous validation | checkpoint-pinned high-priority water candidate; no placeholder calculator or multi-solvent claim |
| ML-for-charges PBE0-ESP | pretrained atomic charge prediction from MACE-OFF23-large descriptors | unavailable unmodified XGBoost runtime, no solvent input/PES/endpoints, and a separate explicit-water alchemical protocol | excluded from Route 4; a future authorization-gated Route 1 charge-provider audit only, with no fitting |
| Organic_MPNICE + MLFF_HFE | published water-HFE FEP/replica-exchange protocol plus a proprietary pretrained MLFF result | no public Organic_MPNICE checkpoint, user-supplied lambda-aware TorchScript requirement, missing `openmmtools`, and no solvent selector | excluded from Route 4; neither a surrogate checkpoint nor a reconstructed protocol may be used |
| ABCG2/GAFF2 | official AmberTools fixed-charge assignment, smoke-verified with no local fitting | a separately constructed explicit solvent/gas TI/BAR/MBAR protocol, record ledger, and independent panel; it is not an MLIP | Route 1 explicit-solvent physics control only; never Route 4 `#model` or a cross-panel score |
| AniSolv compact | 21 solvent-conditioned scalar geometry-level corrections with a public compact checkpoint | rotation-covariant conservative forces, a sampled thermodynamic endpoint, standard state, and independent experimental panel | sealed `anisolv-uma` single-point energy control only |
| C3Net | direct scalar predictions for 103 exact upstream solvent identifiers from a supplied SDF geometry | force/PES task contract, thermodynamic endpoint pair, and complete record-level training accounting | property-prediction sidecar only; never `#model` or an absolute-solvation ledger backend |
| CIGIN | direct scalar prediction from one canonical solute SMILES and one canonical solvent SMILES | force/PES task contract, thermodynamic endpoint pair, and complete record-level training/multi-solvent accounting | CPU-only property-prediction sidecar; never `#model`, an absolute-solvation ledger backend, or a multi-solvent-coverage claim |
| MoletoSolv | published neutral/anion/cation descriptor-regression study plus public data and precomputed predictions | no serialized pretrained estimators/scalers, incomplete neutral data asset, no maximum-error/holdout/GPU-parity evidence, and no PES/force contract | literature baseline only; local artifact reconstruction or training is prohibited |
| SolProp_ML Gsolv | ten-member direct scalar pair-prediction ensemble from a pinned official bundle | uninstalled original runtime dependencies, force/PES task contract, and complete record-level training accounting | blocked property-only candidate; never `#model` or an absolute-solvation ledger backend |
| SolProp-mix Exp | selected ten-member official direct dGsolv ensemble for pure and mixed solvents | uninstalled original runtime dependencies, force/PES task contract, and record-isolated benchmark identity | blocked mixture-property candidate; never `#model`, a geometry/MD path, or an absolute-solvation-protocol ledger backend |
| Schake GNN v2 | pretrained backbone CA/C/N structural correction coupled to GBn2 for protein landscapes | uninstalled original runtime dependencies, no solvent selector, and no small-molecule/PES/absolute-solvation contract | blocked protein research control only; never the general Route 4 backend |
| ConSolv | a solvent-conditioned 66-solvent potential trained through a free-energy path | official inference runtime, weights, descriptor package, and artifact identity | watchlist only; audit immediately if the authors release artifacts |
| TWIN | a water implicit potential spanning drug-like molecules, peptides, and proteins | public runtime/weights and non-water coverage | water-only watchlist; no reconstructed adapter |

The non-negotiable test is always endpoint identity.  A result may enter the
absolute-solvation ledger only when the exact released potential supplies both
endpoints, the protocol samples the required ensemble, and the estimator plus
standard state are stored in the benchmark identity.  This is why neither the
AniSolv scalar correction nor a GNNIS energy trace is allowed to stand in for
an experimental solvation free energy.

### GNNIS original-runtime audit: identity verified, current environment blocked

The official source was freshly cloned at
`5f849bab475570aeaf129ec2e5672f4afbef6bbe`, with a clean tree and an
origin URL matching the pinned repository.  The existing local `GNN.pt` has
the model-card SHA256 `304a6cb2e1f804d30dcc4e1b135be1aa768074b4cb268ba26ee34acaa513e5f6`.
However, importing the unchanged upstream
`Simulation.helper_functions.get_gnn_sim` fails before simulation because
`openmoltools` is absent.  The same source imports `openff.toolkit` and the
GNN layers import `torch_geometric`; all three are absent from every installed
MAPLE environment.

MAPLE therefore does not claim a live 39-solvent original-runtime execution
from its factory/unit contracts.  It will neither install those dependencies
without authorization nor copy/rewrite the source around them.  The pinned
reference adapter, solvent registry, capability boundaries, and checksum still
protect the intended composition, but an actual upstream-runtime multi-solvent
energy/force certificate remains open until the official dependency set is
available unchanged.

The same exact helper chooses CUDA with
`device = "cuda" if torch.cuda.is_available() else "cpu"` before constructing
its GNN force.
Consequently a nominal MAPLE CPU request does not control the upstream device.
MAPLE now blocks that original runtime whenever CUDA is visible and verifies
that any allowed CPU-only construction actually created an OpenMM `CPU` or
`Reference` context.  GNNIS acceleration remains disabled until its own
energy/force/Hessian/MD parity artifact passes gate 7.

### GNNIS upstream hydration-helper audit

The pinned GNNIS revision contains `Simulation/Hydration_Free_Energy.py`, an
older two-state MBAR helper that reads separately supplied vacuum and GNN
trajectories and evaluates the four endpoint energy arrays.  It is useful
upstream provenance, but it is not an admissible MAPLE ΔG protocol: its own
`calculate_hydration_free_energies()` warns that correlated samples should be
subsampled, fixes `kT = 2.479` without an explicit temperature identity, and
does not define a gas-to-solution standard-state conversion, solvent-specific
endpoint convention, uncertainty/convergence criterion, or record-level
training-overlap status.  MAPLE therefore preserves GNNIS as a multi-solvent
mechanics/MD reference Hamiltonian and requires a new audited protocol artifact
before any GNNIS result can enter the absolute-solvation ledger.  Reimplementing
or calibrating that helper around labels would violate the no-fitting rule.
