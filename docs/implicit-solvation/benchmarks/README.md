# Route-2 secondary energy diagnostic

## Route 2: MACE-POLAR-1-M/SMD/IEFPCM

[`route2-protocol.json`](route2-protocol.json) is a reproducible FreeSolv
diagnostic for the Route-2 energy proof-of-concept. It does not define Route 2,
certify analytic forces, or establish a complete solution-phase PES.
It pins the same FreeSolv v0.52 source hashes as route 1 while defining a
separate split seed, model/continuum/CDS composition, domain, standard state,
runtime measurement, and confirmation gates. The ten structures previously
inspected during route-1 development remain development-only; this prevents
prior human review from leaking into Route-2 confirmation.

Canonical commands:

```bash
PROTOCOL=docs/implicit-solvation/benchmarks/route2-protocol.json
WORK=.omx/benchmarks/route2-macepolar-smd

python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  prepare --protocol "$PROTOCOL" --work-dir "$WORK"
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  run-supervised --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --device cuda
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  summarize --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --output "$WORK/development-summary.json"
```

`run-supervised` keeps the normal calculator-reuse path inside one worker, but
restarts that worker if PCMSolver terminates the process on a fatal cavity
check. It freezes the affected compound as an audited provider failure and
continues; unidentified worker exits are not relabelled. Long runs can also be
split into deterministic, disjoint shards that share the same work directory:

```bash
for SHARD in 0 1 2 3; do
  python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
    run-supervised --protocol "$PROTOCOL" --work-dir "$WORK" \
    --partition development --device cuda \
    --shard-count 4 --shard-index "$SHARD" &
done
wait
```

The runner parses the exact public three-line Route-2 input, initializes the
official MACE-POLAR-1-M calculator once through `SetCalculator`, and evaluates
each molecule through `ImplicitSolvationCorrection` plus the common calculator
finalizer rather than calling the SMD provider directly. Model-load time is
excluded. Each atomic record stores the public input, gas/solvation/combined
energy identity, non-thermochemical ASE `free_energy` marker, and the complete
PCMSolver audit directory. The summary reports MSE/MAE/RMSE/max error/failure
rate, deterministic bootstrap intervals, chemistry strata, component energies,
and the median/p90 ratio of integrated Route-2 wall time to gas MACE wall time.
A `--max-compounds` run is a smoke only and cannot be summarized as a complete
partition.

Before confirmation, freeze the exact reviewed proposal and pass rule:

```bash
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  freeze-confirmation --protocol "$PROTOCOL" --work-dir "$WORK" \
  --proposed-default 'macepol-m/smd-iefpcm-water/scf' \
  --pass-rule 'MAE <= 1.5 kcal/mol; failure_rate == 0; median total/gas <= 2.0'
```

The protocol also records separate open Dip146 (`MAE <= 0.25 D`) and HR46
(`MAE <= 2.0 A^3`) response gates. Those datasets are not silently folded into
FreeSolv, because hydration error alone cannot certify the learned
density/polarizability mechanism.

No Route-2 FreeSolv accuracy artifact is frozen yet. The tracked
[`route2-pcmsolver-exact-gto-acetone-v1.json`](route2-pcmsolver-exact-gto-acetone-v1.json)
is only a one-molecule development canary: it records the failed raw
constant-potential gauge probe, the atomic-centre-mean repair, and a matched
gauge local-jet/exact-GTO comparison. The current artifact was regenerated on
a clean committed source snapshot after independent review, and records the
complete MACE-POLAR-1-M checkpoint hash, exact graph-longrange layout, and
live checkpoint-matrix fingerprint. That closes execution provenance only;
it does not replace a development
partition or confirmation set. The other real-water and static SMD-CDS
controls in `VALIDATION_STATUS.md` prove execution, units, reciprocity, and
convergence only. Until development and one-shot confirmation pass, the
public input continues to require `experimental=true`; no constant shift,
fine-tuning, or confirmation-set refit is allowed.

[`route2-qm-fidelity-v1.json`](route2-qm-fidelity-v1.json) is a separate frozen
bounded pilot: three same-geometry Route-2/QM/experiment comparisons plus one
locked four-geometry electronic conformer panel. Its tests recompute energy
decomposition, MAE/RMSE, and ensemble statistics. It does not replace the
unrun FreeSolv confirmation partition, and its observed CUDA-versus-CPU timing
ratios are not hardware-normalized speedups. The recorded `.omx` source paths
and SHA256 values are witnesses to retained local raw evidence, not tracked CI
inputs. The JSON records evidence-generating execution heads separately from
the runtime-equivalence reference head; ancestry alone is not treated as
evidence that a result was rerun. All three fixed-conformer energies have clean
canaries at `5746f24`, `f3e9892`, and `e34abc5`, with no tracked
`maple/` runtime-source change among those heads. One
2-acetoxyethyl-acetate source-geometry analytic-force canary at `d72dfba`
reproduces the historical correction force within `2.66e-14 eV/angstrom`.
Its direct one-component `5e-4`-angstrom finite-difference pair was rerun
against the same runtime and differs from the analytic force by
`3.084e-6 eV/angstrom`; both displaced energies converge in 18 iterations
without PCMSolver or legacy `primary` warnings. One pre-registered central
C--C torsion pair at \(\pm0.5^\circ\) was then rerun at `d72dfba`. Its analytic
and central-difference generalized forces are `0.0452064347` and
`0.0451054120 eV/rad`, differing by `1.01023e-4 eV/rad` (`0.2235%`); both
energy-only points converge in 18 iterations without either warning class.
A separately locked \(\pm1.0^\circ\) pair on the same coordinate gives
`0.0450446898 eV/rad`, an absolute error of `1.61745e-4 eV/rad` (`0.3578%`).
Refining to \(0.5^\circ\) reduces the absolute error by
`6.07221e-5 eV/rad`, to `0.62458` of the coarse-step error. Its two roots also
converge in 18 iterations with no PCMSolver or legacy `primary` warnings; the
`151.35 s` wall time is not used as a speed claim. The original
\(0.5^\circ\) runner wrote both immutable point records before a NumPy-Boolean
JSON serialization error, so a read-only finalizer validated those records and
no scientific energy was rerun. The two steps establish a bounded refinement
trend, not an asymptotic convergence order.

The third pre-registered \(\pm0.25^\circ\) pair gives
`0.0451675753 eV/rad`; its analytic discrepancy falls to
`3.88594e-5 eV/rad` (`0.0860%`). Both roots converge in 18 iterations with no
PCMSolver or legacy `primary` warnings. However, the \(0.5^\circ\)-to-
\(0.25^\circ\) drift is `6.21634e-5 eV/rad`, not smaller than the preceding
`6.07221e-5 eV/rad`, and the observed central-difference order is `-0.0338`.
The pre-registered `1.5--2.5` smooth second-order gate therefore fails. The
runner's exit 1 records that scientific failure after both results were
written; it is not an execution failure. The `148.50 s` wall time is not a
speed claim.

Component decomposition gives order `2.008` for gas MACE and `2.001` for CDS,
but only `0.544` for solvent-intrinsic MACE and `0.212` for PCM polarization.
The failing behavior is localized to the self-consistent electrostatic
coupling block.

The completed pre-registered root-cause diagnostic reuses all six immutable
torsion states and performs exactly one independent \(+0.25^\circ\)
energy-only root, with no new geometry, finite-difference step, or force
evaluation. Energy reproduces to `2.22e-16 eV` and density to `2.22e-16 e`;
all 18 locked validity gates pass, and the PCMSolver and legacy `primary`
warning counts are zero.
Formal PySCF-1202-Lebedev mapping finds 19 changed active points on the
minus-side fine interval and 27 on the plus side. Frozen density carries
`73.8%` of the fine-drift L1 norm; fixed-center-density PCM and
reaction-map-through-MACE carry `67.6%` and `31.9%` of that block's
subcomponent L1 norm. The bounded classification is
`active-set-associated-explicit-continuum-geometry-response`.

This is an association, not an operator-only/cavity-only causal split, and it
does not authorize a production change. The four-geometry flexible panel,
closed-loop panel, and second-molecule force evidence remain historical. The
JSON freezes the reviewed numerical ledger, including the failed
pre-registered gate, the subsequent valid diagnostic, and arithmetic checks.

The converged MLIP--PCM/SMD total derivative is now implemented for the
explicit pyddx/PySCF single-point candidate. The next primary milestone is a
separately pre-registered upstream-backed smooth cavity/operator feasibility
profile with one same-energy forward/adjoint/coordinate-VJP/CDS definition,
followed by local smooth-force and translation/rotation canaries. The existing
optional PySCF SWIG control is not automatically promotable because its fixed
laboratory-frame quadrature has already shown nonmonotonic rotation residuals.
Do not add another finer pyddx point, retune the failed gate, rerun the bounded
closed loop, or advance to a second flexible molecule or short-NVE
conservation before the smooth-provider gate passes.
Running more FreeSolv records must not displace those PES gates, while the
bounded QM-fidelity panel must not be promoted into a broad accuracy claim.

The first feasibility protocol is
`../ROUTE2_PROVIDER_CANARY.md`, with tracked runner
`run_route2_jgp94_ddpcm_canary.py`. It is intentionally smaller than that
eventual provider milestone: one frozen acetone density, three prescribed
rigid orientations, six total ddPCM scalar cases, and no MACE/CDS/force call.
It tests only whether the JGP94 nuclear-charge standard frame removes the
laboratory-grid scalar rotation span while preserving the identity-orientation
energy within `0.01 kcal/mol`.

The one-shot run at `da89ab8` passed every frozen gate: the laboratory-frame
span was `1.6528e-4 kcal/mol`, the molecule-frame span was
`4.8644e-14 kcal/mol`, and the identity-profile shift was
`1.4473e-4 kcal/mol`. The exact result is tracked as
`route2-ddpcm-ri-jgp94-acetone-v1.json` with SHA256
`f1b7b01eb80b42489c1306b2006f919e5cc79bc295f73c84824626abccbeeeb7`.
This pass authorizes only a separately locked analytic frame-VJP canary.

That next protocol is `../ROUTE2_FRAME_VJP_CANARY.md`, with pure transform/VJP
module `route2_jgp94_frame.py` and one-shot runner
`run_route2_jgp94_frame_vjp_canary.py`. Its zero-solve preflight passes all 30
coordinate and 30 dipole derivatives of the frame algebra, but rejects the
provider experiment before a lock: the eight prescribed displaced cavities
produce four active-set signatures and 4992/4993 active pairs. The exact
preflight is tracked as `route2-jgp94-frame-vjp-preflight-v1.json`, SHA256
`eefad3ce832bf8ffc1c2db9d29663e151da83e1558a3489e4193a360eaaeafb2`.
No continuum state, MACE call, ML--SCF root, CDS evaluation, finite-difference
energy, retry, or public-profile change occurred.

The next bounded upstream control tested PySCF 2.13.1 ISWIG without integrating
another provider. The pre-registration
`route2-pyscf-iswig-discriminator-prereg-v1.json` locked methanol and acetone,
three rigid orientations, order 47, the SWIG control, the fixed density
archives, and the exact 12-solve budget before execution. The exact executed
runner is retained as `run_route2_pyscf_iswig_discriminator.py` with the
SHA256 recorded in `route2-pyscf-iswig-discriminator-v1.json`; it is an
archived research runner, not a public provider. The result also distinguishes
the pre-registered ddPCM scientific-profile context from the actual
`smd-iefpcm-gaff2-o` radius selector used by the runner. At the execution head
both select the same strict GAFF/GAFF2 carbonyl-oxygen radii, but only the
actual selector is treated as executed provenance.
ISWIG failed without a retry: its rotation spans were `1.6858x` and `1.2634x`
the corresponding SWIG spans, and the surviving parent counts changed across
orientations for both molecules. All scalar/reciprocity checks remained
finite. The tracked result
`route2-pyscf-iswig-discriminator-v1.json` therefore rejects ISWIG as the
selected next Route-2 provider candidate; it does not authorize a gradient,
ML--SCF, CDS, PES, or production-provider claim.

The proposed four-molecule same-geometry accuracy protocol was independently
reviewed before execution and withdrawn without running any of its five new
single points. Its immutable preregistration remains
`route2-four-molecule-accuracy-prereg-v1.json`; the separate
`route2-four-molecule-accuracy-prereg-v1-disposition.json` records the missing
per-molecule Route-1 geometry/source-record, QM-artifact, and FreeSolv
dataset/uncertainty provenance. A broader replacement must be registered as a
new protocol with those fields locked before any calculation. The rejected
protocol must not be repaired in place or used as evidence for accuracy.

## MNSol-v2012 multi-solvent dataset protocol

[`route2-mnsol-protocol-v1.json`](route2-mnsol-protocol-v1.json) freezes a
separate, provider-independent MNSol-v2012 ingestion contract. MNSol contains
row-level material that MAPLE does not redistribute, so the adapter never
downloads the database and never writes raw rows to its output. It accepts
only a user-supplied archive or extracted directory, applies bounded archive
reads, verifies both the pinned `MNSol_alldata.txt` hash and the normalized
table-plus-geometry bundle hash, reconciles all 3037 rows with all 790 fixed
M06-2X/MG3S gas-phase geometries, and emits aggregate coverage and provenance:

```bash
python docs/implicit-solvation/benchmarks/mnsol_dataset.py inspect \
  --source .omx/datasets/mnsol-v2012/MNSolDatabase_v2012.zip \
  --protocol docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json \
  --output .omx/benchmarks/route2-mnsol-v2012-coverage.json
```

The initial neutral absolute panel has ten experimentally populated solvents:
water, ethanol, acetonitrile, dimethyl sulfoxide, dimethylformamide,
tetrahydrofuran, chloroform, dichloromethane, toluene, and hexane. Methanol
remains a supported runtime solvent, but MNSol-v2012 has no neutral absolute
methanol rows; its 80 entries are ionic and therefore outside the current
public Route-2 domain. The protocol keeps absolute gas-to-solution free
energies separate from water-to-organic transfer free energies and groups
every row sharing one `FileHandle` into the same partition, preventing the
same solute from leaking across solvents.

The Route-2 domain audit also applies the frozen single-covalent-component
criterion. Consequently, the disconnected MNSol water-dimer record is reported
as an aggregate exclusion rather than being mislabeled as a single-molecule
Route-2 candidate.

This stage is dataset infrastructure only. It does not run MACE-POLAR, expose
AIMNet charges, add ddCOSMO, or compare accuracy. AIMNet will first be treated
as a distinct fixed point-charge baseline; COSMO-RS remains a separate
sigma-profile/statistical-thermodynamic model family rather than a continuum
solver switch. The later comparison matrix belongs to benchmark strategy, not
the MNSol data contract:

- solute-source axis: self-consistent MACE-POLAR coarse residual
  point-\(l\le1\) multipoles versus an AIMNet2 point-\(l=0\) fixed-charge
  baseline;
- continuum-equation axis: PCMSolver IEFPCM as an energy-only oracle, pyddx
  ddPCM as the current single-point research-force candidate, and pyddx
  ddCOSMO as a separate planned family;
- liquid-thermodynamics axis: COSMO-RS only through a separate sigma-profile
  implementation and a matching partition/transfer benchmark, never by
  relabelling a COSMO boundary solver.

## AIMNet2 fixed point-charge baseline

[`route2-aimnet2-point-charge-ddpcm-canary-v1.json`](route2-aimnet2-point-charge-ddpcm-canary-v1.json)
is the first bounded AIMNet2 source-interface result. AIMNet2 predicts
atom-centred point charges and applies Neural Charge Equilibration after each
charge update to enforce the requested molecular total; the learned charges
are trained against DFT electrostatic observables
([Chemical Science 2025](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc08572h)).
The official model guide also identifies the standard AIMNet2 models as
gas-phase molecular models without implicit solvation
([AIMNetCentral](https://isayevlab.github.io/aimnetcentral/models/guide/)).
MAPLE therefore exposes these charges as a separate fixed
`point-charge-l0` source and does not insert AIMNet2 into the MACE-POLAR
mutual-polarization fixed-point engine.

The tracked runner is
`run_aimnet2_point_charge_canary.py`. It pins the local checkpoint by SHA256,
does not redistribute it, uses four small ASE G2 geometries, and runs a reduced
`lmax=7`, 302-point pyddx/ddPCM water canary without CDS:

```bash
python docs/implicit-solvation/benchmarks/run_aimnet2_point_charge_canary.py \
  --checkpoint /path/to/aimnet2.pt \
  --output route2-aimnet2-point-charge-ddpcm-canary-v1.json
```

For checkpoint
`85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`,
the largest raw total-charge residue is `8.4043e-6 e` for methane and is
removed by an explicitly audited uniform float-residue projection. The four
half-coupling identity errors range from `1.28e-15` to `8.59e-13 eV`.
On the recorded single-thread CPU run, median AIMNet2 charge inference is
`0.0036--0.0041 s`; the reduced ddPCM solve is `0.0164--0.1170 s`.

The corresponding continuum-only polarization energies are:

| molecule | fixed-charge ddPCM polarization (kcal/mol) |
|---|---:|
| water | -6.6851 |
| ammonia | -3.6138 |
| methane | -0.1152 |
| acetone | -4.9187 |

These are interface canaries, not hydration-free-energy predictions: no
solute polarization response, SMD-CDS term, experimental comparison, force,
or solution-phase PES is included. The artifact SHA256 is
`126f6f08329f6553654defa6fc59dd059c60c6d9e87ca3f1c307764e8c4335c5`.

## MACE-POLAR local-field thermodynamic diagnostic

[`route2-mace-local-field-thermodynamic-canary-v1.json`](route2-mace-local-field-thermodynamic-canary-v1.json)
is a one-water, no-threshold mechanism diagnostic. It constructs the public
`MACE-POLAR-1-M/point-l1/local-jet/pyddx-ddPCM/PySCF-SMD-CDS` profile through
`CommandControl` and `SetCalculator`, reuses its shared-engine coupled root,
then interrogates only the learned local potential/gradient response. CDS is
present in the public energy but is not part of any field-response diagnostic.

The root converged in 9 iterations with an unmixed density residual of
`1.8633e-12 e`; its half-coupling identity error was `8.16e-15 eV`. Three
independent implementation-level JVP/VJP dot tests closed within
`2.11e-12 eV`, ruling out a broken automatic transpose as the source of the
physical diagnostics.

Neither tested energy interpretation is close to its exact identity. The
maximum separately normalized monopole/dipole defect is `0.6641` for the
intrinsic-energy candidate and `0.9995` for the already-coupled-energy
candidate. “Intrinsic” is only the numerically smaller defect; it is not a
pass. Three response-reciprocity probes have absolute discrepancies from
`0.00288` to `0.00465 eV`. In the explicit 12-dimensional scaled response
matrix, the symmetric part has 5 negative, 4 near-zero, and 3 positive modes,
with a largest positive eigenvalue of `0.00856 eV`; the antisymmetric
Frobenius norm is `0.01715 eV`.

The field-space rectangular loop gives `-4.1610e-9 eV` at step scale
`1e-3` and `-1.0402e-9 eV` at `5e-4`. Dividing by the squared step gives
`-0.00416103` and `-0.00416091 eV`, respectively. This local refinement is
consistent with nonzero circulation for the tested water state and directions,
not merely a constant absolute quadrature residue. It is still one molecule,
one local-jet interface, and one direction pair; it does not quantify broad
chemical-space behavior.

The rebound CUDA process took `11.47 s` on an RTX 4060 Laptop GPU:
`4.41 s` model load, `3.60 s` public root/energy, `0.15 s` intrinsic field
gradient, `1.18 s` for three reciprocity pairs plus their adjoint checks, and
`1.21 s` for the 12 JVP stability matrix. These are run metadata, not portable
speed claims.

The result therefore strengthens the conservative wording already used by
Route 2: the matrix-free adjoint differentiates the scalar surrogate actually
implemented, but the tested learned local-field response does not support a
claim that this scalar is a jointly variational physical free-energy
functional. The continuum density-to-feature forward/adjoint and model-side
rectangular feature JVP/VJP are now composed in fixed-geometry
diagnostic infrastructure. This local-jet artifact is not retroactively an
exact-GTO result, and the exact-GTO profile remains outside the production
force path because its matching coordinate/gauge derivative does not exist.
The artifact SHA256 is
`518dc61f20df790ce8b110548e8fff40926e2d5a9d0f7ffcfefa898706fc65e6`.
