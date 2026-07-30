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

### Mandatory historical FreeSolv-10 regression for V0

The original ten-class FreeSolv screen is not allowed to disappear merely
because later MNSol, FreeSolv, or 11-solvent panels have different membership.
Its historical GTO/QEq `GBn2` result reached **7.041442082076966 kcal/mol** on
ethyl acetate. That retired method is not reused, but the exact ten
experimental records and hash-bound MOL2 conformers are frozen in
[`route2-v0-historical-freesolv10-regression-v1.json`](route2-v0-historical-freesolv10-regression-v1.json).

For a V0 result, all ten original records must be supplied to
[`route2_v0_historical_freesolv10.py`](route2_v0_historical_freesolv10.py), and
each recomputed absolute error must be **strictly less than 1.5 kcal/mol**.
Missing, extra, duplicate, substituted, or error-selected records fail; MAE or
RMSE cannot override one outlier. This mandatory water regression is in
addition to—not a substitute for—the pre-registered 11-solvent development,
confirmation, and final-blind gates.

```bash
python docs/implicit-solvation/benchmarks/route2_v0_historical_freesolv10.py \
  --predictions "$WORK/v0-historical-freesolv10-predictions.json" \
  --output "$WORK/v0-historical-freesolv10-evaluation.json"
```

The prediction input is a JSON object containing a `predictions` list, each
with only `compound_id` and `predicted_kcal_mol`; the validator obtains
experimental values only from the immutable manifest and returns nonzero on a
failed strict gate.

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
  --pass-rule 'all records: max_absolute_error < 1.5 kcal/mol; failure_rate == 0; median total/gas <= 2.0'
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

## Same-basis GTO--PCM representation transfer

[`route2-gto-pcm-energy-projection-four-prereg-v1.json`](route2-gto-pcm-energy-projection-four-prereg-v1.json)
freezes a separate representation-only experiment over benzene, acetone,
acetic acid, and 2-acetoxyethyl acetate. It compares one and two radial
\(l\leq1\) GTO bases by projecting the same fixed-geometry
\(\omega\)B97M-V/def2-TZVPD cavity MEP in the PCMSolver IEFPCM energy norm.
Source and receiver use the same GTO basis, the total charge and molecular
dipole are constrained exactly, and every raw canary binds the preregistration,
QM checkpoint, MOL2 geometry, intrinsic probe-zero cavity, executable, and
source hashes.

The completed aggregate is
[`route2-gto-pcm-energy-projection-four-v1.json`](route2-gto-pcm-energy-projection-four-v1.json).
The second radial channel improves all four records at the preregistered
`1e-4` cutoff and lowers mean representation error from `1.2365` to
`0.7536 kcal/mol`. The flexible diester remains at `1.5372 kcal/mol`, so the
frozen sub-`1 kcal/mol` maximum-error gate **fails**. Lower cutoffs reduce the
two-radial mean error but expose near-null modes:

| relative cutoff | two-radial mean error / kcal mol\(^{-1}\) | maximum retained condition number | maximum coefficient L2 norm |
| ---: | ---: | ---: | ---: |
| `1e-4` | 0.7536 | \(9.87\times10^3\) | 6.49 |
| `1e-6` | 0.2998 | \(9.43\times10^5\) | 65.72 |
| `1e-8` | 0.1224 | \(9.33\times10^7\) | 243.9 |
| `1e-10` | 0.1043 | \(9.92\times10^9\) | 963.5 |
| `1e-12` | 0.0855 | \(9.88\times10^{11}\) | 16541.3 |

This is therefore a complete negative preregistered result, not a reason to
select the lowest cutoff. It establishes a representation
accuracy--conditioning tradeoff only. It does not validate MACE-POLAR
density--energy conjugacy, projected coefficients as ML targets, experimental
solvation accuracy, a variational free energy, forces, or a solution-phase
PES.

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

[`route2-mnsol-pilot-selection-v1.json`](route2-mnsol-pilot-selection-v1.json)
freezes the first bounded ten-solvent pilot **before** any model or error is
evaluated. It takes one record per validation solvent, prefers the confirmation
partition, requires distinct solute geometries, caps this first diagnostic at
20 atoms, and chooses the lexicographically smallest seeded SHA256 score. The
score uses only solvent identity, geometry handle, and database entry number;
it does not use the experimental free energy or any model output. The tracked
manifest contains only opaque row and geometry hashes plus aggregate candidate
counts—no names, formulas, coordinates, entry numbers, or experimental values.

[`route2-mnsol-confirmation-selection-v1.json`](route2-mnsol-confirmation-selection-v1.json)
freezes the complete 148-row confirmation partition without selecting a
favorable subset. It contains 83 unique solute geometries. Because the earlier
ten-record pilot was already inspected, 46 confirmation rows sharing eight
pilot solute geometries are explicitly marked as prior-pilot overlap; the
other 102 rows are ordered first but are not described as a fully untouched
ten-solvent set. Within each overlap stratum, records are interleaved by
solvent and ordered by a seeded hash of solvent identity, geometry handle, and
entry number. Experimental free energies and model outputs do not enter that
order.

Both multisolvent runners accept this partition manifest only with the frozen
pilot manifest and one explicit `--record-index`. They refuse an implicit
148-record run, require both derived outputs below `.omx`, and mark each shard
`do_not_commit`. This keeps execution bounded and prevents a single row from
being presented as population accuracy:

```bash
python docs/implicit-solvation/benchmarks/run_mnsol_macepolar_multisolvent_pilot.py \
  --source .omx/datasets/mnsol-v2012/MNSolDatabase_v2012.zip \
  --protocol docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json \
  --selection docs/implicit-solvation/benchmarks/route2-mnsol-confirmation-selection-v1.json \
  --pilot-selection docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json \
  --record-index 0 \
  --private-output .omx/benchmarks/mnsol-confirmation-000-private.json \
  --public-output .omx/benchmarks/mnsol-confirmation-000-summary.json \
  --work-dir .omx/benchmarks/mnsol-confirmation-000-work
```

The corresponding runner is
`run_mnsol_aimnet2_multisolvent_pilot.py`. It refuses a dirty checkout, verifies
the frozen selection against the user-supplied database, writes the row-level
result only below `.omx`, and emits a separate aggregate-only public summary:

```bash
PYTHONPATH=/path/to/pyddx-and-pyscf/site-packages \
python docs/implicit-solvation/benchmarks/run_mnsol_aimnet2_multisolvent_pilot.py \
  --source .omx/datasets/mnsol-v2012/MNSolDatabase_v2012.zip \
  --protocol docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json \
  --selection docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json \
  --checkpoint /path/to/aimnet2.pt \
  --private-output .omx/benchmarks/route2-mnsol-aimnet2-private-v1.json \
  --public-output docs/implicit-solvation/benchmarks/route2-mnsol-aimnet2-multisolvent-pilot-v1.json
```

The pilot uses fixed AIMNet2 NQE point charges, the production-resolution
`lmax=15`/1202-point discretization for pyddx ddPCM and scaled ddCOSMO,
solvent-specific SMD Coulomb radii, and the official PySCF 2.13.1 SMD-CDS
entrypoint. The multisolvent profile supplies shared radii/descriptors/CDS;
ddCOSMO remains an explicitly named equation override rather than pretending
to be the ddPCM profile. The two predictions are
\(U_\mathrm{pol}+G_\mathrm{CDS}\); neither contains mutual
AIMNet2–continuum polarization or a solute internal-response term. No
1-atm-to-1-M correction is added because MNSol absolute values already use the
1-M ideal-gas to 1-M ideal-solution convention.

This preregistration is selection infrastructure, not an accuracy result.
AIMNet is treated as a distinct fixed point-charge baseline. A separately
named pyddx ddCOSMO equation adapter exists, but it is not C-PCM. A separately
named PySCF SWIG/C-PCM research operator is available for bounded equation
canaries; it is not a public profile and has not passed a broad chemical
benchmark. COSMO-RS remains a separate sigma-profile/statistical-thermodynamic
model family rather than a continuum solver switch. The later comparison
matrix belongs to benchmark strategy, not the MNSol data contract:

- solute-source axis: self-consistent MACE-POLAR coarse residual
  point-\(l\le1\) multipoles versus an AIMNet2 point-\(l=0\) fixed-charge
  baseline;
- continuum-equation axis: PCMSolver IEFPCM as an energy-only oracle, pyddx
  ddPCM as the current single-point research-force candidate, and pyddx
  ddCOSMO as a separate optional equation family with host-applied
  \((\epsilon-1)/\epsilon\) scaling for pyddx 0.8.0;
- liquid-thermodynamics axis: COSMO-RS only through a separate sigma-profile
  implementation and a matching partition/transfer benchmark, never by
  relabelling a COSMO boundary solver.

## Two-record PCM-family equation screen

A bounded private MNSol water canary changed the continuum equation while
holding the frozen gas-phase MACE-POLAR \(l\leq1\) source and each record's
SMD-CDS value fixed. The three PySCF 2.13.1 SWIG comparisons used exactly the
same order-17 surface; the pyddx 0.8.0 comparison used the same
`lmax=15`/1202-point ddX discretization. Aggregate-only results are:

| equation arm | two-record MAE / kcal mol\(^{-1}\) | mean wall time / s |
|---|---:|---:|
| PySCF SWIG IEFPCM | 2.3352 | 0.2250 |
| PySCF SWIG C-PCM | 2.2838 | 0.0774 |
| PySCF SWIG COSMO | 2.3516 | 0.0717 |
| pyddx ddPCM | 2.2756 | 2.9677 |
| pyddx scaled ddCOSMO | 2.2212 | 0.6762 |

The same-surface PySCF equation changes move either prediction by only
`0.0160--0.0588 kcal/mol`; scaled ddCOSMO moves the two pyddx predictions by
`-0.0452` and `-0.0635 kcal/mol`. Every half-coupling identity error is below
`1.9e-14 eV`. The fixed, non-randomized execution order makes the timing column
runtime metadata rather than a speed ranking. PySCF-versus-pyddx differences
also change discretization and backend implementation, so only within-backend
comparisons isolate the equation.

The private row-level artifact has SHA256
`de912de10d71401229434381d5dea429486327acfba5838269956259fe742c3a`
and remains below `.omx` because it contains user-supplied MNSol records. With
only two development points and an equation effect below
`0.064 kcal/mol`, this screen alone did not justify a broader claim. A later
separately preregistered source-by-equation panel, described below, answered
the broader AIMNet2/MACE and ten-solvent comparison question without altering
this historical screen.

## MNSol fixed-source PySCF PCM-family panel

[`route2-mnsol-fixed-source-pyscf-pcm-family-prereg-v1.json`](route2-mnsol-fixed-source-pyscf-pcm-family-prereg-v1.json)
froze a same-surface comparison before any new PySCF equation result was
observed. It reuses the exact ten-record AIMNet2/MACE source artifact, then
changes only the PySCF 2.13.1 SWIG boundary equation among IEFPCM, C-PCM, and
COSMO. Every record recomputes and verifies the MNSol identity, geometry,
solvent-specific SMD Coulomb radii, dielectric, and SMD-CDS value. AIMNet2
contributes fixed monopoles; MACE-POLAR contributes fixed atom-centred
\(l\leq1\) coarse residual-charge multipoles. Neither source is polarized by
the continuum in this experiment.

The clean-head run at `da9d3ba` completed all 60 fixed-source evaluations. All
three equations used exactly the same SWIG points and areas for each record,
and the maximum half-coupling identity error was
\(1.67\times10^{-16}\) eV. The aggregate-only result is
[`route2-mnsol-fixed-source-pyscf-pcm-family-v1.json`](route2-mnsol-fixed-source-pyscf-pcm-family-v1.json):

| frozen source and equation | MAE | RMSE | mean signed error | maximum absolute error |
|---|---:|---:|---:|---:|
| AIMNet2 \(l=0\) + IEFPCM | 1.1297 | 1.3547 | +1.0387 | 2.2727 |
| AIMNet2 \(l=0\) + C-PCM | **1.0396** | **1.2953** | +0.8905 | **2.2434** |
| AIMNet2 \(l=0\) + COSMO | 1.1142 | 1.3519 | +1.0249 | 2.2855 |
| MACE \(l\leq1\) + IEFPCM | 0.8780 | 1.0021 | +0.3555 | 1.6458 |
| MACE \(l\leq1\) + C-PCM | 0.9127 | 1.0291 | +0.1674 | **1.6456** |
| MACE \(l\leq1\) + COSMO | **0.8738** | **1.0004** | +0.3490 | 1.6457 |

All errors are in kcal/mol. C-PCM lowers the AIMNet2 MAE by
`0.0901 kcal/mol` relative to IEFPCM and has lower absolute error on 8/10
rows. For fixed MACE \(l\leq1\), COSMO lowers the IEFPCM MAE by only
`0.0042 kcal/mol`; each equation wins five paired rows. The MACE source lowers
the same-equation MAE relative to AIMNet2 by `0.1269--0.2517 kcal/mol`, but
the paired win count ranges from 5/10 to 7/10.

These are descriptive results from one different molecule in each of ten
different solvents and ten post-selection functional-group labels. They do
not separate solvent from chemistry, establish per-solvent generalization, or
support changing a production default. The `5.27 s` total and per-method
timings exclude frozen AIMNet2/MACE inference and were not randomized.
The standard ORCA/openCOSMO-RS reference is not included because it requires a
separate quantum-chemical sigma-profile and statistical-thermodynamic workflow
rather than another switch on the same PCM surface. A later experimental
MLIP-surface bridge is a separately named out-of-parameterization arm, not a
reinterpretation of this same-surface equation panel. The public artifact
SHA256 is
`d387adb6d3e2e1fa082685948911eb45fbf60ba8ff5cc3c313d758df46d8179f`;
the private row-level artifact remains below `.omx` with SHA256
`7ea80dbf4f9b7e75106b41f3f7f1d42f7b9d38bbded12da1e581c6cf5a696e5f`.

## COSMO-RS QM reference and MLIP-surface research boundary

COSMO-RS is intentionally absent from the Route-2 profile registry. The
version-locked boundary in `maple.function.cosmo_rs` accepts the three assets
required by the ORCA/openCOSMO-RS 24a workflow: the solute gas output, solute
perfect-conductor surface, and solvent perfect-conductor surface. It records
their SHA256 hashes and enforces BP86/def2-TZVPD, ORCA 6, 298.15 K, neutral
closed-shell systems, parameterized COSMO-RS radii, and the
`0.01 angstrom squared` surface-segment cutoff.

This boundary parses and audits externally generated `dGsolv`, renders a
neutral-singlet fixed-geometry ORCA input, and binds the three assets from one
audited ORCA run. It is the **QM oracle arm**. It does not add ORCA as a MAPLE
dependency or expose a public calculator.

`maple.function.mlip_cosmo_rs` is the separate experimental integration arm.
It takes neutral AIMNet2 monopoles or MACE-POLAR \(l\leq1\) coefficients,
constructs a PySCF SWIG perfect-conductor surface, solves its screening charge,
applies the 24a minimum-segment rule, and renders the upstream surface-file
contract consumed by openCOSMO-RS. Thus the solute chain is MLIP
\(\rightarrow\) implicit conductor \(\rightarrow\) sigma profile
\(\rightarrow\) liquid thermodynamics, with no solute QM calculation.

This does not make the result strict openCOSMO-RS 24a: the parameterization was
fitted to BP86/def2-TZVPD profiles, and a precomputed QM-derived solvent profile
may still be reused. The first bridge is fixed-source only; a self-consistent
MACE--conductor profile and an MLIP-generated solvent-profile arm remain open
gates. The upstream method-mismatch warning is scientific evidence and must
not be hidden by falsely labelling the surface as BP86/def2-TZVPD.

The preregistered external benchmark
[`route2-mnsol-opencosmors24a-fixed-geometry-prereg-v1.json`](route2-mnsol-opencosmors24a-fixed-geometry-prereg-v1.json)
locks ORCA `6.1.0-f.0`, the bundled openCOSMO-RS executable, serial execution,
ten solvent aliases, one fixed MNSol geometry per record, and comparison
against the already-frozen six-method PCM-family artifact. Its first execution
is deliberately limited to selection indices `1`, `6`, and `9`; all row-level
outputs and ORCA work files remain below `.omx`.

This is not the full published openCOSMO-RS 24a conformer/geometry workflow.
It is also a training-domain reproduction diagnostic rather than an
independent confirmation: the published 24a parameterization used the
Marenich/Minnesota solvation-free-energy data in fitting. The runner therefore
reports the overlap explicitly and forbids a blind-generalization claim.

The ten records were then executed incrementally from clean scientific-code
commit `6421f18`, not as one uninspected batch. A separately committed
aggregator re-rendered every input, rebound every QC asset, reparsed every main
output, checked all 30 child outputs, and reconstructed every comparison
against the frozen PCM-family artifact before emitting
[`route2-mnsol-opencosmors24a-fixed-geometry-v1.json`](route2-mnsol-opencosmors24a-fixed-geometry-v1.json).
All 10 main and 30 child calculations terminated normally.

| method | MAE | RMSE | mean signed error | maximum absolute error |
|---|---:|---:|---:|---:|
| ORCA/openCOSMO-RS 24a, fixed MNSol geometry | **0.6460** | **0.8038** | +0.5792 | **1.5081** |
| AIMNet2 fixed \(l=0\) + IEFPCM | 1.1297 | 1.3547 | +1.0387 | 2.2727 |
| AIMNet2 fixed \(l=0\) + C-PCM | 1.0396 | 1.2953 | +0.8905 | 2.2434 |
| AIMNet2 fixed \(l=0\) + COSMO | 1.1142 | 1.3519 | +1.0249 | 2.2855 |
| MACE fixed \(l\leq1\) + IEFPCM | 0.8780 | 1.0021 | +0.3555 | 1.6458 |
| MACE fixed \(l\leq1\) + C-PCM | 0.9127 | 1.0291 | +0.1674 | 1.6456 |
| MACE fixed \(l\leq1\) + COSMO | 0.8738 | 1.0004 | +0.3490 | 1.6457 |

All values are in kcal/mol. openCOSMO-RS has the lower absolute error on
`8/10`, `6/10`, and `7/10` rows relative to the AIMNet2 IEFPCM, C-PCM, and
COSMO arms, and on `7/10` rows relative to each MACE arm. Its summed serial
wall time is `258.03 s` (`25.80 s/record`). The fixed-source continuum timings
exclude AIMNet2/MACE inference and are therefore not an end-to-end speed
comparison.

The lower error is descriptive only. openCOSMO-RS was fitted using the
Minnesota/Marenich data, whereas the fixed-source arms were not fitted in this
experiment, and the one-row-per-solvent design still confounds solvent and
chemistry. No default is changed. The public artifact SHA256 is
`13c78bdaf72a61b830a5535882ada87fd32acf196a4aa78a8f9e840a84531125`;
the complete private row-level artifact remains below `.omx` with SHA256
`f9110f74767b06ce77a67e707656fa60d2717ed449ecab5ebcb6082f2a750148`.

## MNSol ten-solvent AIMNet2 pilot

[`route2-mnsol-aimnet2-multisolvent-pilot-v1.json`](route2-mnsol-aimnet2-multisolvent-pilot-v1.json)
is the aggregate-only result from the preregistered ten-record selection. It
was executed from clean commit `7bc6164`; rerunning after the audit-only runner
update reproduced every scientific scalar from the first run exactly. Eight
records are in the confirmation partition and two are deterministic
development fallbacks needed to preserve distinct solute geometries.

| fixed-charge method | MAE | RMSE | mean signed error | maximum absolute error |
|---|---:|---:|---:|---:|
| AIMNet2 + ddPCM + SMD-CDS | 1.1431 | 1.3516 | +1.0333 | 2.2322 |
| AIMNet2 + scaled ddCOSMO + SMD-CDS | 1.0121 | 1.2743 | +0.8300 | 2.1933 |

All values are in kcal/mol. Scaled ddCOSMO has the lower absolute error on
8/10 selected rows and is on average `0.2033 kcal/mol` more negative than
ddPCM. That is a small-pilot observation, not evidence that ddCOSMO is
generally more accurate: the ten rows represent ten different solvents, the
chemical sample is intentionally capped at 20 atoms, and there is no
per-solvent replication.

The maximum raw AIMNet2 total-charge residue is `1.49e-7 e`; the explicit
float-residue projection closes the largest final charge sum to
`1.11e-16 e`. Maximum half-coupling errors are `3.12e-14 eV` for ddPCM and
`2.09e-14 eV` for ddCOSMO. The mean component ledger is:

| method | mean \(U_\mathrm{pol}\) | mean \(G_\mathrm{CDS}\) | mean prediction | mean experiment |
|---|---:|---:|---:|---:|
| ddPCM | -2.4715 | -1.3122 | -3.7837 | -4.8170 |
| ddCOSMO | -2.6748 | -1.3122 | -3.9870 | -4.8170 |

The internal single-process wall time was `28.92 s`. Shared AIMNet2 charge and
SMD-CDS work totaled `0.0775 s` and `0.0623 s`; ddPCM build/solve totals were
`2.435/19.968 s`, while ddCOSMO build/solve totals were `0.0297/6.301 s`.
These timings are metadata only: the fixed ddPCM-then-ddCOSMO order was not
randomized and process-level cache effects make them unsuitable as a method
speed ranking.

The tracked artifact contains no row-level MNSol values. The detailed
experiment/component table remains only in the ignored `.omx` output for the
local database holder. The public artifact SHA256 is
`68319cc76c9f6f8ceb479625dd3f2a841cd592609b0a564d8978f1e783521d11`.
This pilot does not validate MACE-POLAR, mutual polarization, C-PCM, COSMO-RS,
forces, or a solution-phase PES, and it is not a substitute for the complete
MNSol development/confirmation partitions.

## MNSol fixed-versus-polarizable response ablation

[`route2-mnsol-macepolar-response-ablation-v1.json`](route2-mnsol-macepolar-response-ablation-v1.json)
uses the same preregistered ten records to separate source representation from
ML response. The ten reference values are neutral absolute experimental
MNSol-v2012 free energies at 298 K in the 1-M ideal-gas to 1-M ideal-solution
standard state: one record is from subset `[a]` and nine are from subset `[g]`.
The runner verified the selected rows directly against the pinned
`MNSol_alldata.txt` hash
`6dba4397764d1ca665c5dac653b9963bd64f784c353311a72c15e42897b90156`.
The public artifact remains aggregate-only; the selected names, entry numbers,
coordinates, and experimental row values remain in the ignored private
`.omx` result.

Every method uses the same pyddx ddPCM equation, solvent-specific PySCF SMD
Coulomb radii, and PySCF 2.13.1 SMD-CDS term. Only the learned source and
response treatment change:

| method | MAE | RMSE | mean signed error | maximum absolute error | mean time (s) |
|---|---:|---:|---:|---:|---:|
| AIMNet2 fixed \(l=0\) charges | 1.1431 | 1.3516 | +1.0333 | 2.2322 | 2.379 |
| MACE fixed \(l=0\) monopoles | 3.3946 | 3.7781 | +3.3946 | 5.9762 | 2.684 |
| MACE fixed \(l\le1\) multipoles | **0.8635** | **0.9902** | +0.3430 | **1.6454** | 2.715 |
| MACE one-response diagnostic | 1.0392 | 1.1906 | +0.7306 | 1.9386 | 3.048 |
| MACE same-root \(l\le1\) SCF | 0.9905 | 1.3225 | -0.4703 | 3.1905 | 26.247 |

Energies and errors are in kcal/mol. The fixed MACE monopoles alone are not a
competitive electrostatic source: restoring the learned atomic dipoles lowers
the MAE by `2.5311 kcal/mol` and gives the best descriptive aggregate in this
panel. Relative to that fixed \(l\le1\) source, the converged SCF makes the
mean prediction `0.8132 kcal/mol` more negative, lowers the absolute error on
4/10 records, raises it on 6/10, and raises the MAE by `0.1270 kcal/mol`.
The one-response diagnostic lies between the two energy treatments in the
aggregate but is explicitly not a fixed point.

This does not show that fixed electrostatics is intrinsically more accurate
than mutual polarization. It shows that the current local-jet
MACE-POLAR/\(l\le1\)-multipole + ddPCM + SMD-CDS combination does not turn its
additional response into a robust ten-point accuracy gain. The fixed
\(l\le1\) result can benefit from cancellation between gas-phase multipoles,
cavity/CDS parameters, and missing response, while the SCF can expose
representation or calibration error by adding an average
`+0.6889 kcal/mol` solute response and changing the mean continuum
polarization from `-3.1618` to `-4.6639 kcal/mol`. Exact-GTO field projection,
variational-response diagnostics, and replicated per-solvent chemistry remain
open gates before any general accuracy conclusion.

The actual single-process wall time was `342.93 s`. Per-method times are
phase-summed estimated standalone costs with shared phases charged to each
method; execution order was not randomized, so they are diagnostic metadata,
not a hardware speed ranking. The public artifact SHA256 is
`2d7e7d2ea5356ad97344e77316d1c4c30dae3cfa2642892ff5b1c93cd8168c52`.

### Frozen-partition two-member aggregation

`aggregate_mnsol_response_partition.py` is the fail-closed,
development-only finalizer for the next full-partition milestone. Its source
shards must all come from the current response-ablation runner with
`--maximum-response-stage scf`, one common clean execution commit, the frozen
development selection, official unfine-tuned MACE-POLAR-1-M checkpoint, and
the fixed ddPCM profile. The source runner evaluates its complete five-method
audit schema; the finalizer validates that schema and every component/error
identity, then retains only `mace_fixed_l1` and `mace_scf_l1`.

After every index in one partition has completed, aggregate the private shards
with repeated `--private-shard` arguments:

```bash
SHARD_ARGS=()
for SHARD in .omx/benchmarks/mnsol-development-*/private.json; do
  SHARD_ARGS+=(--private-shard "$SHARD")
done

python docs/implicit-solvation/benchmarks/aggregate_mnsol_response_partition.py \
  --source .omx/datasets/mnsol-v2012/MNSolDatabase_v2012.zip \
  --protocol docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json \
  --selection .omx/benchmarks/route2-mnsol-development-selection-v1.private.json \
  --pilot-selection docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json \
  "${SHARD_ARGS[@]}" \
  --private-output .omx/benchmarks/route2-mnsol-development-two-member-v1.private.json \
  --public-output docs/implicit-solvation/benchmarks/route2-mnsol-development-two-member-v1.json
```

The private output must stay below `.omx/benchmarks`; the public output is
aggregate-only and strips local checkpoint paths. Missing/duplicate indices,
mixed or nonexistent commits, fingerprints, input-shard hashes, continuum
profiles, checkpoints, prior-pilot overlap flags, or ledger identities abort
aggregation. Existing isolated shards that predate the required continuum
metadata are intentionally incompatible.

For `mace_scf_l1`, every completed shard must carry dimension-separated SCF
convergence evidence. A nominal row records separate monopole and dipole
residuals and no fallback evidence. A finite-resolution row for
`smd-ddpcm-l15-n1202-multisolv-v1` now uses response-ablation artifact/schema
v5 and the explicit `route2-scf-convergence-evidence-v4` contract. The
finite-resolution policy itself remains `finite-resolution-stagnation-v2`, but
the integrated solver policy is `safeguarded-anderson-v2`: the accepted-state
actual-residual objective is `Phi = max(monopole/tau_mono, dipole/tau_dipole)`
with the nominal channel tolerances in the denominator, and any Anderson
arrival with `Phi_k > 2 Phi_anchor` is rejected, rolled back to the prior
accepted anchor, advanced by exactly one Picard step, and restarted with an
accepted-only Anderson history. Rejected attempts still count toward the
existing 100-attempt bound, but they are excluded from Anderson samples,
best-state selection, and finite-resolution windows.

Accordingly, a finite-resolution row must additionally record the earliest
online seven-step accepted Anderson window satisfying every predicate, the
frozen enumerated runtime identity gate (including `torch==2.12.0+cu130` and
the nested pyddx relative-iterate tolerance `1e-14`), the
accepted-parent lineage, the solver epoch, and three fresh reaction-map
reevaluations at the retained density. These are not independent cold SCF
starts. Under `finite-resolution-stagnation-v2`, the three fresh-cold field
arrays must have identical canonical little-endian float64 digests, and the
three fresh-cold response arrays must have identical canonical little-endian
float64 digests. The online warm candidate is compared against each
fresh-cold replay with bounded deltas only: reaction-potential and
reaction-gradient component deltas must stay within the existing
`1e-10 eV/e` and `1e-10 eV/(e angstrom)` gates, while response monopole and
dipole deltas must stay within the nominal `2e-12 e` and
`2e-12 e angstrom` gates. Residual ceilings, the
intrinsic/PCM/electrostatic ledger spans, and the `2e-10 eV` half-coupling
identity gate are unchanged. The shard records four per-evaluation hashes
(online warm plus three fresh-cold) for fields and for responses, plus the
four maximum online-to-cold ULP metrics, as diagnostics only. This
finite-resolution branch is energy-only; force requests require nominal
convergence. Aggregation rejects ULP values outside the finite float64 range,
delta/ULP zero-status contradictions, and identical online/cold hashes paired
with nonzero delta or ULP metrics.

Historical index-004 evidence remains preserved but uncounted. The retained
private trajectory and earliest-window map-reevaluation diagnostics are
`.omx/diagnostics/mnsol-development-index004-trajectory-4d09ba74-root-v1/trajectory.json`
and
`.omx/diagnostics/mnsol-development-index004-iter023-cold-map-4d09ba74-root-v1/diagnostic.json`;
they keep the iteration-23 candidate from the 17--23 window, the legacy mixed
raw-\(l\le1\) component infinity norm `2.6204635683590993e-11` (not a
channel-separated v2 quantity), the
`4.526934382909076e-14 eV` half-coupling identity error, and zero locked
`1e-12` ledger span. The older post-hoc best-state map reevaluation,
`.omx/diagnostics/mnsol-development-index004-cold-map-4d09ba74-root-v1/diagnostic.json`,
remains failed and uncounted; it is a diagnostic witness, not an aggregatable
row result.

The clean byte-identity failure that motivated replay v2 is also preserved as
non-countable evidence in
`.omx/diagnostics/mnsol-development-index004-map-replay-norm-965aeb8-v2/`.
That diagnostic-only run showed six false actual `np.array_equal` checks while
keeping the measured deltas small. Online-warm versus fresh-cold fields had
max-absolute/L2/max-relative differences of `1.936228954946273e-12` /
`7.901838936066065e-12` / `4.286571482853837e-10`; responses had
`7.577272143066693e-15` / `2.9938859233415605e-14` /
`8.560585688915213e-13`. The same run retained
`maximum_monopole_residual_e = 1.1122637533222957e-11`,
`maximum_dipole_residual_e_angstrom = 2.6204635683590993e-11`,
`intrinsic_ledger_span_ev = 0.0`,
`pcm_ledger_span_ev = 9.71445146547012e-14`,
`electrostatic_ledger_span_ev = 9.71445146547012e-14`, and
`maximum_polarization_identity_error_ev = 4.526934382909076e-14`.
That replay-v2 rerun is still not countable.

New index-180 rollback evidence is preserved separately as uncounted runtime
monkeypatch diagnostics. The archived best failed root in
`.omx/diagnostics/index180-picard-from-best-16bf09b-v2/diagnostic.json`
started from residual `3.1794e-12`; a full Picard trial rose to `6.937e-11`;
and a damped `alpha=0.25` trial reached `2.4996e-12`, still not nominal. The
rollback diagnostic in `.omx/diagnostics/index180-rollback-picard-16bf09b-v1/`
then reached attempt 33 under the unchanged finite-resolution-v2 policy with
final monopole residual `1.1485e-12`, final dipole residual `2.36193e-12`,
and four rejected growth ratios `21.88`, `27.57`, `27.87`, and `28.69`.
Its one-record shard reported absolute error `0.3790 kcal/mol`, but that value
is diagnostic only and is not an accuracy certification. Because the run used a
runtime monkeypatch instead of the prospectively integrated public solver, it
cannot count toward the exact two-member matrix, the frozen 505-row
development partition, or the sealed 148-row confirmation partition. The
prospective integrated rerun is still pending.

No complete 505-row development or 148-row confirmation artifact has been
produced yet. This benchmark lane remains the exact two-member matrix only;
it does not authorize fitting, UQ calibration, response tempering, density
mixing retuning, or any public-API claim. This artifact rejects confirmation
selections outright; a separately reviewed, hash-bound freeze/unseal gate is
required before the sealed confirmation partition can be evaluated.

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
`a058ed4a74f4533048c22a231995f78d28e6b743eb62f298a8cfdcc064b29bfb`;
the file now also records its original execution commit, without changing any
scientific value.

## AIMNet2 ddPCM/ddCOSMO equation-axis canary

[`route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1.json`](route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1.json)
changes only the pyddx continuum equation for fixed AIMNet2 charges. Water and
acetone use identical ASE geometry, `point-charge-l0` source, SMD water radii,
dielectric, `lmax=7`, 302-point grid, and solver tolerance. The ddCOSMO host
factor is explicitly \((78.39-1)/78.39=0.9872432708253603\), because pyddx
0.8.0 returns unscaled COSMO energies and derivatives.

| molecule | ddPCM \(U_\mathrm{pol}\) | ddCOSMO \(U_\mathrm{pol}\) | COSMO - PCM |
|---|---:|---:|---:|
| water | -6.6851 | -6.7223 | -0.0373 kcal/mol |
| acetone | -4.9187 | -4.9472 | -0.0285 kcal/mol |

All four half-coupling errors are below `8.59e-13 eV`. The recorded internal
build/solve times are `0.0013/0.0127 s` (water ddPCM),
`0.0004/0.0038 s` (water ddCOSMO), `0.0029/0.1167 s` (acetone ddPCM), and
`0.0005/0.0342 s` (acetone ddCOSMO). They were obtained in one fixed
ddPCM-then-ddCOSMO process, so cache/order effects make them metadata rather
than a method-speed ranking.

The runner is
`run_aimnet2_continuum_equation_canary.py`; it refuses a dirty checkout and
binds the result to execution commit `6b3ba1e`. This remains a continuum-only
control. It includes no CDS, experimental value, total solvation free energy,
self-consistent AIMNet2 polarization, force, C-PCM, or COSMO-RS calculation.
The artifact SHA256 is
`0daf325fe24f384d3f29c61d14216acccb7e7099114aaa6ee41abb254c738847`.

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

## Matrix-free root-local feedback-gain canary

[`route2-matrix-free-feedback-gain-two-state-v1.json`](route2-matrix-free-feedback-gain-two-state-v1.json)
anonymizes two fixed neutral MNSol-derived states and compares the
matrix-free largest-feedback-singular-value estimate against the complete
neutral-space dense oracle. No row identity, geometry, measured value,
prediction, or chemistry-error record is published.

| state | neutral dimension | dense \(\sigma_{\max}\) | matrix-free estimate | absolute difference | triplet residual | JVP/VJP calls | dense time | matrix-free time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| state-01 | 31 | 0.3195933458 | 0.3195933457 | \(1.02\times10^{-10}\) | \(1.55\times10^{-10}\) | 13/12 | 39.49 s | 35.74 s |
| state-02 | 75 | 0.4167834048 | 0.4167834047 | \(5.41\times10^{-11}\) | \(3.46\times10^{-10}\) | 23/22 | 447.10 s | 269.61 s |

Both iterative estimates are below one at their tested roots and match the
dense oracle within \(1.1\times10^{-10}\), but neither value is a certified
upper bound on a nonlinear neighborhood. They therefore establish numerical
agreement and local cost only—not a Banach contraction certificate, global
fixed-point uniqueness, variational identity, thermodynamic passivity, or
chemical-accuracy result. The 75-dimensional estimate still took 269.61 s, so
the diagnostic remains outside default SCF and broad panels.

The execution occurred at tracked head `ca5aed3` while the diagnostic source
was still uncommitted. The exact diagnostic and synthetic-test byte hashes
match their subsequent landed commit `f833d8e` and the current frozen source.
The private execution summaries did not record the model-checkpoint hash, so
this is source-bound evidence rather than a checkpoint-bound rerun. Timings
exclude model loading and fixed-state setup and are single-host metadata, not
a portable speed ranking. Artifact SHA256:
`7683c93ae373341de0dcd8563e48a092a9ff231965c326d68cc21de1830dae48`.

## MACE-POLAR exact-GTO fixed-geometry derivative canary

[`route2-mace-exact-gto-fixed-geometry-canary-v1.json`](route2-mace-exact-gto-fixed-geometry-canary-v1.json)
is a source-bound one-acetone implementation canary for the exact-GTO model
drive. It constructs the public
`MACE-POLAR-1-M/point-l1/exact-gto/PCMSolver-IEFPCM/native-SMD-CDS` root,
reopens the identical fixed cavity, and composes the continuum
density-to-feature map with the learned feature-to-density response. It
requests no coordinate derivative and changes neither the production energy
nor the disabled force path.

The 13-iteration root has a `5.2759e-6 e` unmixed residual. Reopening the
516-point cavity reproduces the reaction field and model feature tensor
exactly. Continuum, learned-model, and composed-residual JVP/VJP identities
close to `5.00e-16`, `4.09e-13`, and `2.46e-12`; the physical adjoint reaches
`1.96e-9` relative residual in five callbacks and seven operator
applications. Its neutral-direction energy derivative has `1.75e-7` relative
central-difference error at the best tested coefficient step, `1e-3`.
The two smaller steps are less accurate, consistent with finite-precision and
solver cancellation rather than a demonstrated asymptotic refinement order.

The acetone hydration value, `-4.6836 kcal/mol`, differs from the development
record `-3.80 +/- 0.60 kcal/mol` by `0.8836 kcal/mol`; it is recorded only for
provenance. There are zero native `PCMSolver warning.` markers and one retained
PEDRA poor-tessellation warning in each public/reopened solve. The local
internal total was `8.49 s` (`10.50 s` process wall time), which is run
metadata rather than a portable performance result.

This artifact establishes only fixed-geometry derivative algebra for the
exact-GTO receiver. It sets no scientific pass threshold and supplies no
coordinate/gauge derivative, thermodynamic-conjugacy result, full
electron-density PCM, original-SMD equivalence, hydration certification,
smooth PES, OPT, TS, scan, MD, or NVE claim. Its SHA256 is
`8c63c7a85f8b100a0165ba5d0eb4626250f326aaf85ed1d3c13ee57552a6d491`.

## Intrinsic-cavity exact-GTO ten-class development panel

[`route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json`](route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json)
freezes the first pre-registered cross-functional-group check after replacing
the unintended PCMSolver water probe with the SMD intrinsic electrostatic
cavity. All three arms use the same point-\(l\le1\) source, SMD radii,
zero-probe/no-added-sphere cavity, explicit `78.355` dielectric, IEFPCM
equation, and native water SMD-CDS term. The arms are gas-density fixed
\(l\le1\), self-consistent local jet, and self-consistent exact receiver GTO.

The ten neutral FreeSolv v0.52 records cover ketone, alcohol, aromatic
hydrocarbon, amide, carboxylic acid, cyclic diether, nitro, phenol, alkyl
chloride, and ester. Their experimental values and uncertainties are checked
against the pinned upstream `database.txt` SHA-256
`2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260`;
each row retains its original experimental reference and row fingerprint.
Selection was frozen before the panel and did not use candidate errors.

| Method | MAE | RMSE | Max error | Mean wall time |
| --- | ---: | ---: | ---: | ---: |
| MACE fixed \(l\le1\) | 1.084 | 1.294 | 2.647 | 0.420 s |
| MACE SCF local jet | 1.752 | 2.516 | 5.744 | 2.130 s |
| MACE SCF exact GTO | **0.915** | 1.305 | 3.247 | 1.599 s |

Exact GTO has lower absolute error on 7/10 rows and lowers acetone from
`2.963` to `0.933 kcal/mol` relative to local jet. It also materially improves
acetamide, acetic acid, and methyl acetate. It is not uniformly better:
methanol, phenol, and chloroethane favor local jet, while acetic acid remains a
`3.247 kcal/mol` outlier. Consequently the pre-registered panel-MAE,
control-MAE, acetone, and native-warning gates pass, but the `<2 kcal/mol`
maximum-error gate fails. The candidate is retained as a non-default,
energy-only experimental profile; neither the old profile nor any global
default is replaced.

The timing order was counterbalanced by record and model load was excluded,
but each arm was run only once on this host, so the timing is diagnostic rather
than a portable benchmark. This development panel does not certify FreeSolv
population accuracy, an unseen confirmation set, training-set exclusion,
variational thermodynamics, forces, or a smooth solution-phase PES. Artifact
SHA-256:
`d20469a091b84523d3b7417dc3a90696a72c31124ff6e18a54a33e1e62b71d90`.

Schema v2 also freezes both panel runners and the pre-registered selection
under
[`reproducers/route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1/`](reproducers/route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1/).
The artifact records immutable Git blob identities for the source bytes that
produced the panel, the complete MACE-POLAR checkpoint, PCMSolver library and
parser identities, runtime package versions, and a MOL2 plus canonical
structure fingerprint for every geometry.  Tests intentionally validate those
immutable blobs rather than equating a historical result with the current
working-tree implementation.

## Same-basis GTO/PCM energy-norm acetone canary

[`route2-gto-pcm-energy-projection-acetone-v1.json`](route2-gto-pcm-energy-projection-acetone-v1.json)
locks the first fixed-geometry reference for a same-source/same-receiver
finite-width \(l\le1\) GTO continuum operator. The frozen acetone
\(\omega\)B97M-V/def2-TZVPD gas checkpoint supplies the AO density directly
through its stored restricted orbitals and occupations; no separately supplied
density matrix can be substituted. The checkpoint geometry matches the
FreeSolv MOL2 within \(1.1\times10^{-10}\) angstrom, and the reconstructed
density has zero electron-count binding residual.

Both basis arms use the same 516-point intrinsic PCMSolver IEFPCM cavity, SMD
radii, zero probe, no added spheres, and energy-conjugate surface response.
The one-radial \(1.5\)-angstrom basis has 40 coefficients; adding a
\(3.0\)-angstrom radial channel gives 80:

| GTO radial widths | QM PCM polarization | Fitted polarization | Absolute projection error |
| --- | ---: | ---: | ---: |
| 1.5 angstrom | -6.64334 | -6.23222 | 0.41112 kcal/mol |
| 1.5, 3.0 angstrom | -6.64334 | -6.57660 | **0.06674 kcal/mol** |

The surface/source reciprocity residual is below
\(1.8\times10^{-16}\) hartree, and both constrained projections pass their
charge/dipole, tangent-optimality, semidefinite, and shifted affine
Pythagorean gates. This establishes a useful representation result only. It
does not connect the new coefficient space to the current MACE-POLAR
checkpoint, prove density--energy conjugacy, certify hydration free-energy
accuracy, or authorize forces, a solution-phase PES, optimization, TS, scan,
frequency, or MD use. Artifact SHA-256:
`9f5d13432ee5a3759bcc45ec6b0e6e845d3eb977d16cd7f3578c9813f3660823`.
