# Test Specification — Independent OBC-II/ACE Analytic Multi-MLIP

**Parent plan:** `.omx/plans/route1-independent-obc2-analytic-mlip-20260927.md`  
**Status:** initial test-first specification; consensus gate pending.

## A. Proof obligations

1. Independent provider requires no OpenMM import, initialization, metadata, files, or distribution.
2. Independent topology/radius/screen preparation is identical to the pinned OpenMM 8.5.2 OBC-II/mbondi2 source for the supported domain.
3. OBC-II/ACE|none E/F/H/HVP comes from the unchanged exact Torch scalar; FD is validation-only.
4. Every qualified gas backend differentiates its actual production scalar and composes the same E/F/H/HVP identity.
5. Parser/factory/package changes preserve old OpenMM and non-OBC providers.
6. Claims and artifacts distinguish qualification, failure, unavailability, feasibility block, convergence, performance, and scientific admission.
7. Existing `implicit`, `implicit-gb`, `implicit-ddlpb`, and `implicit-free-energy` extras remain byte/meaning compatible; independence uses a new distinct extra.
8. Formula/source manifest identity is separate from scalar-cell `nonpolar=ace|none` identity.

## B. Required test modules

### New focused suites

- `test_independent_obc2_topology.py`: immutable snapshot, dense components, bond validation, hydrogen environments, topology hash.
- `test_independent_obc2_profile.py`: all mbondi2/screen rule branches, canonical source payload, upstream frozen-source extractor parity, license/provenance.
- `test_independent_obc2_provider.py`: E/F/H/HVP API, bound-atom guards, profile rejections, device/dtype metadata, no fallback.
- `test_independent_obc2_no_openmm.py`: clean wheel environment plus import/distribution blockers and real task paths.
- `test_independent_obc2_mlip_matrix.py`: named gas cells, same-center scalar closure, composite derivatives, cell statuses.
- `test_uma_analytic_feasibility.py`: source-bound go/no-go; only enables analytic tests after genuine same-scalar second AD is proven.

### Existing suites to extend, not replace

- `test_obc2_parameter_contract.py`
- `test_torch_obc2_parity.py`
- `test_torch_obc2_derivatives.py`
- `test_implicit_analytic_derivatives.py`
- `test_command_contract.py`
- `test_direct_solvent_contract.py`
- `test_implicit_frequency.py`
- `test_implicit_ts.py`
- `test_dimer_derivatives.py`
- `test_torch_obc2_artifacts.py`
- existing OpenMM provider and non-OBC regression suites.

## C. Frozen numerical gates

| Check | Gate |
|---|---:|
| OBC total/component energy vs OpenMM Reference | abs `<=2e-10 Ha`, relative `<=1e-9` |
| Component sum closure | `<=2e-10 Ha` (retain tighter current closure where already tested) |
| OBC force max / RMS | `<=2e-7` / `<=5e-8 Ha/A` |
| Hessian asymmetry | `<=1e-10 Ha/A^2` |
| direct HVP vs `H@v` | `atol=rtol=1e-9` |
| translation leakage | max `<=1e-8` |
| solvent Hessian vs OpenMM Reference force FD | RMS `<=5e-6`, max `<=2e-5 Ha/A^2` |
| exact bundle composition after declared conversion | elementwise `<=1e-12` for float64 mock/ANI closure; native-dtype backend gates predeclared per cell |

No gate may be loosened after observing failures merely to admit a cell.

### C.1 Effective AD dtype decision and composite gates

Before backend tests, classify each exact cell from active computation:

- `D64`: all coordinate-dependent network parameters, coordinates, scalar-forward intermediates, gradients, and second-AD operations are float64.
- `N32`: any such network/AD computation is float32 or mixed float32/float64.
- fp16, bf16, AMP/autocast, and TF32 are not admissible analytic profiles in this plan.

NumPy/output casting does not affect classification. Record actual parameter, coordinate, forward, intermediate-gradient, and second-AD dtypes plus all inference/autocast/TF32 flags. The public E/F comparison and analytic E/F/H/HVP path must use the same checkpoint, device, precision, gas options, and neighbor topology.

| Check | D64 | N32 |
|---|---:|---:|
| Center E closure | `abs<=2e-10+1e-9*abs(Eref) Ha` | absolute `<=2e-5 Ha`, no relative escape |
| Center F max/RMS | `2e-7 / 5e-8 Ha/A` | `2e-6 / 5e-7 Ha/A` |
| Raw H symmetry | `<=1e-10 Ha/A^2` | `<=2e-6*max(1,maxabs(H)) Ha/A^2` |
| Direct HVP vs raw `H@v` | per-entry `atol=rtol=1e-9` | per-entry `atol=5e-6, rtol=5e-5` |
| Complete H vs force FD RMS/max | `5e-6 / 2e-5 Ha/A^2` | `5e-5 / 2e-4 Ha/A^2` |
| H translation | `max(1e-8,1e-8*||H||)` | `max(1e-5,1e-5*||H||)` |
| Fixed FD steps | `1e-3,5e-4,2.5e-4 A` | `2e-3,1e-3,5e-4 A` |

Log all three FD results. Both preselected finest steps independently pass RMS/max gates, and their pairwise matrix difference passes RMS/max `<=2x` the corresponding gate. No best-step selection. HVP vectors are unit-L2 normalized and their seed/vectors are frozen before testing. Raw H symmetry and `H@v` are evaluated before any symmetrization. Threshold failure remains `FAILED`.

The OBC solvent always retains the original 48 E/F and solvent H/HVP thresholds independent of D64/N32. These composite budgets acknowledge native float32 engineering roundoff; they are not physical-accuracy guarantees or post-result fitted tolerances. Any float64 promotion is a separately declared same-checkpoint cell with measured native/promoted gas differences.

## D. No-OpenMM harness

The authoritative independence test must:

1. build wheel/sdist and install the independent extra into a fresh environment;
2. prove OpenMM is absent from dependency metadata;
3. assert `find_spec("openmm") is None` and `distribution("openmm")` raises;
4. install a meta-path blocker that raises on any `openmm` import and a metadata spy that raises on any OpenMM query;
5. disable network access, then initialize charges/topology/radii/provider and run E/F/H/HVP;
6. run parser/factory SP, OPT, FREQ, PRFO, and Dimer `use_hvp=true` paths;
7. assert no numerical derivative helper, OpenMM adapter, or fallback was touched.

A development-environment monkeypatch test supplements but does not replace the fresh-install proof.

The packaging portion also asserts the four legacy implicit extras remain unchanged in name, dependency pins, and meaning, while the new independent extra alone omits OpenMM.

## E. Expanded integration matrix

- Preserve the 12-molecule x two-geometry x `ace|none` 48-cell E/F oracle at frozen thresholds.
- Bind the 526 replay only to `.omx/benchmarks/neutral-water-freesolv-route1-20260723` through tracked `chagb_nonpolar_source_manifest.json` SHA `84cd6569296ab4f0ae8410f3f03728fd04c358980936b811be57e8ff2c2508dc` and ordered safe-tuple SHA `ae7e13f6e36855a0fa6b7f2d3bae4733ad58bf02afa5d89cdb57cf4529dddbef`. Assert 526 unique sorted IDs, exact AM1-BCC/OBC-II member suffixes, MOL2/record hashes, ordered charge hashes/counts, and fixed geometry.
- Preflight the pinned dataset identity bytes `database.txt` SHA `2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260`, `database.json` SHA `9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a`, and `mol2files_gaff.tar.gz` SHA `15ece6114442a8eb5e720c0ad25bb48e4a6a35392da36f72b80a6f2f415fd133` by raw bytes only; they are never semantic label inputs.
- Before production edits, generate and seal a fresh offline OpenMM 8.5.2 Reference oracle for both ACE and NONE. Predeclare identity, coordinate/charge/radius/screen, `polar_hartree`, `nonpolar_hartree`, and `total_hartree` fields. NONE must be generated, never inferred; historical ACE record values are not the portable oracle.
- Use the predeclared tracked generator `freeze_obc2_am1bcc_526_reference.py` and output `route1-obc2-am1bcc-526-reference-2026-09-27.json`; both are hashed before implementation.
- Label-denial spies forbid parsing old source-record JSON, `prepared.json`, summaries, database, scores, errors, or experimental fields; raw byte hashing of manifest-bound records is allowed. Output-schema tests reject label/score/error/fit/MAE fields.
- Freeze portable per-atom topology/radius/screen fixtures and optionally regenerate them only in the separate pinned-OpenMM oracle environment.
- Run the label-blind 526-molecule fresh-baseline/new OBC prediction replay without experimental labels, fitting, or rescoring; keep historical artifacts byte-identical.
- Materialize full solvent Hessians for water, methanol, methyl hexanoate, and one preregistered largest/crowded case for both profiles and base/displaced geometries.
- Run direct HVP for all 12 corpus molecules, both profiles/geometries, and at least three deterministic normalized directions; compare against `H@v` wherever full H exists.
- Run multi-step centered-force-FD validation for the four full-H cases, while runtime analytic spies prove FD is unreachable.
- Validate each composite MLIP cell against its exact public gas E/F scalar and identity tuple; a float32/native result converted to NumPy float64 remains a float32/native computation in provenance.
- Validate separate `ace` and `none` scalar-cell identities against the same neutral formula/source manifest; reject any `none` record labeled as an ACE profile.
- For H/D topology fixtures, reproduce upstream valid-graph `bondeds[0]` results and explicitly reject missing or multiply ambiguous connectivity instead of choosing a different first neighbor.
- For every named cached MACE OFF23 S/M/L, OFF24-M, OMOL-XL, and POLAR S/M/L checkpoint, verify actual format/provenance before loading and finish with `QUALIFIED`, `FAILED`, `UNAVAILABLE`, or `FEASIBILITY_BLOCKED` plus a reason. An executed parity, AD, or workflow failure remains `FAILED`, not `FEASIBILITY_BLOCKED`. Spies reject rename/conversion/path substitution.
- Apply the frozen D64/N32 gate without switching gas precision between public E/F, analytic H, and direct HVP. Runtime FD helpers are patched to fail and HVP spies reject dense-H construction.
- UMA S-1p1 Stage 0 binds original `/home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/uma-s-1p1.pt` SHA `07068e9c76702ca173d13155095f2117c1b327ec228557e64cd2709c777b824a`, FAIR-Chem registry source hashes, current wrapper SHA `db16ceda235459912780aa85f46d3848dc92ba9ef39e3a54b1d17a8988bbd092`, transform slice SHA `0c427b17ded483338d236a7a93794f6873cdc494cbcbca15d3c6f61cfe12b9c5`, every config mutation, content-addressed output SHA, Python/Torch/FAIR-Chem versions, `task=omol`, `inference=default`, selected head/dataset, reference-energy identity, effective base dtype, source-unchanged proof, and energy-gradient/force-head graph semantics. S-1p2 original SHA `ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b` and M-1p1 original SHA `c30034edbf2e127f703f814cacb632661767da99b3e71c5b2ee5290510a52d68` are explicit separate rows and require their own receipts; M-1p1 device rows also require memory/resource preflight and need not qualify on every device. Fixed-name cache files without sidecars are rejected as provenance.
- For every UMA compatibility receipt, compare canonical tensor-key lists, shapes, dtypes, and per-tensor hashes or a canonical aggregate tensor-content hash before/after; require exact equality of the learned weight payload and a complete, separate config diff. If archive/ZIP bytes are nondeterministic, semantic tensor equality plus the complete config diff remains mandatory before AD.

## F. Per-cell record schema

Each matrix record includes:

- status and exact reason;
- repository SHA/diff hash and environment identity;
- backend/model/checkpoint path+SHA;
- device/dtype, D4, charge/multiplicity, inference/task settings;
- neighbor topology/cutoff identity and discontinuity margin;
- OBC profile/source/preparation identity and topology hash;
- center gas public-path versus derivative-path E/F deltas;
- solvent and composite E/F/H/HVP metrics;
- workflow commands/results;
- all failures/skips without successful-subset scoring.

## G. Anti-fallback spies

For every analytic workflow, fail if execution reaches:

- `numerical_hessian_from_atoms`;
- `NumericalForceProvider`;
- force-difference HVP/Dimer branches;
- OpenMM import/provider construction on `provider=torch`;
- full Hessian construction during direct HVP;
- caught AD exceptions followed by a numerical/gas-only result.
- eager Hessian construction during E/F or direct-HVP operations;
- a Dimer `get_hvp` result that does not preserve `(Hn, forces, energy)` from the same composed scalar identity and center.
- post-result dtype reclassification, FD-step selection, or Hessian averaging before the raw asymmetry gate.

## H. Workflow admission

For every admitted representative cell:

1. use the frozen water MOL2 fixture for every step; do not replace it after observing a failure;
2. real SP through parser/factory;
3. actual OPT with fixed `max_steps=200` and final `max|F|<=5e-4 Ha/A`;
4. FREQ from the complete analytic Hessian;
5. one-step P-RFO with analytic-path spies;
6. one-step Dimer using direct HVP and no dense Hessian.

The P-RFO/Dimer checks prove execution-path integrity only; they do not claim globally converged transition states.

## I. Artifact integrity

- Hash historical artifacts before and after; bytes must be identical.
- New artifacts are content-hashed and bind scripts, implementation, tests, checkpoint assets, source manifest, and environment.
- New evidence names the exact old claim it supersedes; it does not replace or refresh old SHA fields.
- A final Astra max review consumes both successes and failures and emits the scientific verdict separately from engineering completion.
- The Stage-0 526 generator, allowed-input manifest, source/profile identities, and oracle artifact are sealed before source edits; any forbidden label-path open fails the run.
- UMA receipts bind original SHA, transform code/config SHA and every mutation, canonical tensor-key/shape/dtype/content equality, output SHA or documented container nondeterminism, task/head/inference/dtype/reference configuration, environment, and graph-feasibility verdict. Missing binding terminates `FEASIBILITY_BLOCKED` before AD.
