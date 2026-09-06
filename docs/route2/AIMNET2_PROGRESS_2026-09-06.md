# AIMNet2 frozen-charge progress — 2026-09-06

## Delivery and scientific boundary

This update publishes the runtime-v4 source-response ledger, source-domain
prequalification runner/protocol, multi-solvent same-scalar force
prequalification runner/reducer/protocol, and regression tests. These are
research infrastructure, not a completed source-domain or force admission.
Public E/F/H/V/M and OPT/FREQ/TS/IRC/MD capabilities remain closed.

Ordinary-vs-decomposed energy, charges, and charge-position VJP remain hard
parity gates. The ordinary embedded DFT-D3 intrinsic-gradient difference is
explicitly report-only; repeat decomposed energy/charge/gradient/VJP checks
are hard gates. Repeatability is not a substitute for the pending independent
same-scalar finite-difference qualification.

Pre-push review also added regression coverage and corrected three engineering
edge cases: one geometry can complete multiple missing elements without
redundant selection; failed force finalization preserves its negative artifact
but exits nonzero; zero-endpoint convergence orders remain JSON-safe rather
than embedding infinities in hash-bound evidence. No numerical acceptance
threshold or public capability was relaxed by these corrections.

Private MNSol rows, local runtime evidence, and session files remain under
ignored `.omx/`. No checkpoint, licensed dataset, or row-level results are
redistributed. No new QM calculation was performed for the accuracy review.

## Frozen full-panel accuracy (not a rerun of this source tree)

The retained [full-panel report](../implicit-solvation/benchmarks/route2-mnsol-aimnet2-smooth-partition-ddpcm-full-v1.json)
uses MNSol 2012, **not FreeSolv**: 395 solute names/geometry handles, one fixed
gas-phase geometry each, 653 experimental solute–solvent records, 10 solvents.
This is a sparse experimental panel, not all 395 x 10 combinations. The
domain is neutral singlets; no geometry optimization or conformer averaging
was performed. The standard state is 1 M ideal gas to 1 M ideal solution at
298 K.

| Solvent | Records |
| --- | ---: |
| Water | 387 |
| Chloroform | 109 |
| Hexane | 59 |
| Toluene | 51 |
| Dichloromethane | 11 |
| Ethanol | 8 |
| Acetonitrile | 7 |
| Dimethylsulfoxide | 7 |
| Dimethylformamide | 7 |
| Tetrahydrofuran | 7 |

All 653 eligible records completed. They comprise 505 development and 148
confirmation records; the full-panel metric is not an untouched-test metric.

| Full-panel quantity | kcal/mol |
| --- | ---: |
| Smooth ddPCM MAE | 2.5817042931 |
| Mean signed error (prediction minus experiment) | 2.5426064847 |
| RMSE | 3.3525667445 |
| Maximum absolute error | 10.5342609462 |
| Same-source/CDS high-resolution ddPCM reference MAE | 2.5442693466 |
| Mean absolute smooth-versus-reference difference | 0.0618835476 |

Small average reference differences do not prove order convergence. A retained
post-hoc worst-parity diagnostic changed by 0.7052698965 kcal/mol between the
l4/l8 and l5/l10 configurations. That single-record diagnostic is not an
admission gate and does not justify accuracy-driven order selection.

## Reused matched QM diagnostic: ten records only

Existing artifacts on `research/route2-salted-v1` and `implicitsolv-route2`
provide a common ten-record panel with PySCF 2.13.1 SWIG PCM equations, shared
SMD Coulomb radii, and identical SMD-CDS terms. The QM source is frozen
gas-phase omegaB97M-V/def2-SVPD DF-RKS density. Its RI arm uses the same QM
density in a charge-constrained def2-SVP-JKFIT auxiliary basis, **not a learned
SALTED prediction**. Neither QM arm includes solvent-induced electronic SCF.

The reused artifact identifiers and exact input-file SHA256 witnesses are:

- `route2-mnsol-frozen-full-density-pcm-family-v2`, execution `f9c354f6ace13abaac1c18e39a6f4361eedf60be`:
  `c4325ffc4ba299fac091d00d7f1c13c95bfe94b8fb210dcb0104b59831e3efb2`.
- `route2-mnsol-fixed-source-pyscf-pcm-family-v1`, execution `da9d3ba112b7ba2e1f0a681c9b28b3aa32f19c39`:
  `7ea80dbf4f9b7e75106b41f3f7f1d42f7b9d38bbded12da1e581c6cf5a696e5f`.
- Current branch's retained smooth-ddPCM private panel, execution `c578fda2b974dd44833b8f0813bd689751f47d4d`:
  `6844978ae5f0bcd9e05949903ea66e1976a32dca6eb5ce0e7f2b92d17bbf4ae6`.

The read-only reuse checked baseline hash binding, dataset and selection
fingerprints, all ten record/geometry/experiment/partition identities, shared
CDS values and surface sizes. Row-derived MAE, bias, RMSE, and maximum errors
were recomputed and matched the frozen aggregates within 1e-12 kcal/mol.

| Equation | AIMNet2 point-charge MAE | QM AO-density MAE | QM RI-density MAE |
| --- | ---: | ---: | ---: |
| IEFPCM | 1.1297309378 | 0.4866923161 | 0.4790567962 |
| CPCM | 1.0395967316 | 0.5914300991 | 0.6152723812 |
| COSMO | 1.1142242850 | 0.4836313776 | 0.4760219640 |

All values above are kcal/mol on the same ten records. Under IEFPCM the
maximum absolute error changes from 2.2727316058 to 0.9683820959 kcal/mol;
QM AO has lower absolute error on seven of ten records.

The [frozen pilot selection](../implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json)
chooses one distinct geometry per solvent, with a 20-atom ceiling, using a
pre-model-run hash rule rather than experimental values or model outputs.
The ten selected geometries actually have 5–15 atoms. Water has 10% weight
in this panel versus 387/653 in the full panel. Restricting the existing
smooth-ddPCM results to these ten records already yields MAE 1.1616462604.
Thus the apparent improvement from 2.58 to about 1.13 is predominantly a
panel-composition difference, not a solver improvement. No cross-panel
accuracy ranking or full-panel QM accuracy is established.

## Interpretation and remaining work

The matched diagnostic supports investigating the electrostatic source, but
changes both the source predictor and its representation. It does not isolate
AIMNet2 charge prediction error from point-charge representation loss, nor
prove that missing solvent polarization dominates. No validated large
accuracy improvement is currently demonstrated under unchanged checkpoint,
no-training, frozen-point-charge constraints. No fitting, charge scaling,
radius tuning, or production QM substitution was introduced.

Source-byte changes invalidate reuse of old artifacts as current admission
evidence. Source-domain and frozen full-Cartesian two-cold-process force
qualification still require fresh, clean-commit, hash-bound execution.
Publishing engineering progress does not open scientific capabilities.

## Pre-push engineering verification

After the review corrections, the expanded regression command was:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/route2_vnext tests/calculator/test_aimnet2_charge_state.py tests/solvation/test_benchmark_core.py tests/solvation/test_benchmark_protocol.py
```

Result: **788 passed, 10 skipped, 18 Torch deprecation warnings**, in 197.83 s.
The final targeted suite also passed (106 passed, 4 skipped). Black with
Python-3.10 target, Pyflakes, compilation, protocol JSON parsing, CLI help
smokes, and `git diff --check` passed. Regression tests first reproduced the
three corrected edge cases before their implementation changes.

The initial expanded command lacked `PYTHONPATH` and produced 14 subprocess
import failures: the shared editable installation points to a different MAPLE
worktree. Explicit per-command source selection fixed all 14; no shared
environment installation was changed.

Static typing is **not clean**. An ad-hoc cached Pyright check reported 37
diagnostics across the seven production/tool modules; reviewer coverage of
all twelve changed Python files reported 93, including dynamic test code.
These include benchmark import-path resolution and object/protocol typing
debt. No reviewed diagnostic established another runtime defect, but this is
not a type-clean approval. The independent review disposition is COMMENT,
not scientific admission. Real-checkpoint/domain/full-Cartesian cold replay
qualification was not executed for this publication request.
