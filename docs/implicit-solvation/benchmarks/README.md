# Implicit-solvation benchmarks

## Route 1: FreeSolv ten-molecule pilot

## Reproducible full-corpus harness

The pilot below is preserved as development-only evidence.  The certification
runner is now defined by [`protocol.json`](protocol.json), which pins FreeSolv
v0.52 commit `6c7d19b4b565537365ffd22006aa2cd4643200c6` and SHA256 hashes for
`database.txt`, `database.json`, and `mol2files_gaff.tar.gz`.  Preparation
currently reconciles all 642 records: 526 deterministic development records and
116 untouched confirmation records, with all ten previously inspected pilot
records forced into development.

Canonical commands:

```bash
PROTOCOL=docs/implicit-solvation/benchmarks/protocol.json
WORK=.omx/benchmarks/neutral-water-freesolv

python docs/implicit-solvation/benchmarks/run_freesolv.py \
  prepare --protocol "$PROTOCOL" --work-dir "$WORK"
python docs/implicit-solvation/benchmarks/run_freesolv.py \
  run --protocol "$PROTOCOL" --work-dir "$WORK" --partition development \
  --jobs 4
python docs/implicit-solvation/benchmarks/run_freesolv.py \
  summarize --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --output "$WORK/development-summary.json"
```

The runner writes one atomic JSON record per molecule/charge/GB attempt, resumes
without overwriting completed records, retains provider failures in the original
denominator, and refuses incomplete or protocol-mismatched summaries.  Summary
JSON contains deterministic MSE, MAE, RMSE, maximum error, failure rate,
bootstrap confidence intervals, and predeclared functional-group, element,
size, heteroatom, and flexibility-proxy strata.

`--jobs` parallelizes independent molecules while retaining the same atomic,
per-attempt records.  Choose the count for the available memory and CPU; the
default remains one.

### Completed development partition (2026-07-23)

The frozen 526-molecule development partition has been run with AmberTools
26.0 and OpenMM 8.5.2.  The exact summary, all predeclared strata, failure
records, and 5,260 per-attempt hashes are stored in
[`freesolv-development-2026-07-23.json`](freesolv-development-2026-07-23.json).

| charge / GB | n | MSE | MAE | RMSE | maximum absolute error | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| ABCG2 / HCT | 526 | -1.215 | 2.044 | 3.185 | 20.038 | 0 |
| ABCG2 / OBC-I | 526 | -0.728 | 1.778 | 2.526 | 11.205 | 0 |
| ABCG2 / OBC-II | 526 | -0.715 | **1.652** | **2.358** | 10.429 | 0 |
| ABCG2 / GBn | 526 | -0.586 | 1.760 | 2.398 | 11.080 | 0 |
| ABCG2 / GBn2 | 515 | -1.083 | 1.894 | 2.797 | 15.235 | 0.0209 |
| AM1-BCC / HCT | 526 | -0.361 | 1.855 | 2.769 | 13.127 | 0 |
| AM1-BCC / OBC-I | 526 | 0.068 | 1.930 | 2.728 | 14.561 | 0 |
| AM1-BCC / OBC-II | 526 | 0.044 | 1.760 | 2.537 | 13.550 | 0 |
| AM1-BCC / GBn | 526 | 0.081 | 2.028 | 2.983 | 15.285 | 0 |
| AM1-BCC / GBn2 | 515 | -0.301 | 1.894 | 2.827 | 20.863 | 0.0209 |

All energies are in `kcal/mol`.  The 22 failures are the two charge methods
for GBn2 on the 11 phosphorus-containing development molecules.  They are the
predeclared OpenMM phosphorus fail-closed boundary, not missing records.
ABCG2/OBC-II has the lowest development MAE, but this does not freeze a product
default: conformer sensitivity and human review remain open, and the
confirmation partition has not been opened.

Confirmation is one-shot.  The proposed default and numerical pass rule must be
frozen before any confirmation calculation:

```bash
python docs/implicit-solvation/benchmarks/run_freesolv.py \
  freeze-confirmation --protocol "$PROTOCOL" --work-dir "$WORK" \
  --proposed-default 'REQUIRES HUMAN REVIEW' \
  --pass-rule 'REQUIRES HUMAN REVIEW'
```

Do not substitute placeholder text when opening confirmation.  Once written,
the lock is immutable; a failed confirmation cannot be used for tuning on the
same corpus.

Independent provider evidence uses
[`run_provider_parity.py`](run_provider_parity.py).  Amber-GB references, APBS
Born-ion/grid cases, and their raw audit inputs/outputs are now present under
`tests/solvation/data/`.  The Amber corpus contains five neutral molecules
covering C/H/O/N/F/S and all five GB profiles; the APBS corpus contains the
official Born ion plus methanol and aniline grid sweeps.  Numerical tolerances
remain a separate human-review gate.  The repository includes only a
`provider_parity_tolerances.proposed.json` proposal, and verification fails
closed until an explicitly reviewed file is frozen as
`provider_parity_tolerances.json` with `review_status=human-reviewed-frozen`:

```bash
python docs/implicit-solvation/benchmarks/run_provider_parity.py \
  amber-gb --protocol "$PROTOCOL" --output-dir "$WORK/amber-gb-parity"
python docs/implicit-solvation/benchmarks/run_provider_parity.py \
  apbs-grid --protocol "$PROTOCOL" --output-dir "$WORK/apbs-grid"
python docs/implicit-solvation/benchmarks/run_provider_parity.py \
  verify --protocol "$PROTOCOL" --artifact-dir "$WORK"
```

Amber `gbsa=1` is LCPO, so the parity runner compares polar GB, LCPO nonpolar,
and GB+LCPO totals and forces.  It does not mistake OpenMM's ACE accuracy profile
for Amber LCPO.  On the current frozen five-case corpus, the largest absolute
Amber/OpenMM difference is `0.011194 kcal/mol` for energy and
`0.045856 kcal/mol/A` for force, both from dimethyl sulfide/GBn2; LCPO-only
energies agree to floating-point precision.  These are provider-parity numbers,
not hydration-accuracy statistics.

APBS 1.4.1 reproduces the official `-229.59 kJ/mol` Born-ion result to
`0.000505 kcal/mol` after unit conversion.  The neutral grid sweeps show a
largest successive-finest change of `0.113892 kcal/mol`, demonstrating that
the molecular-surface grid sequence is numerically noisier than the canonical
ion.  The proposed `0.15 kcal/mol` grid threshold records that observation; it
does not claim monotone continuum convergence or chemical accuracy.

Dataset interpretation follows Mobley and Guthrie, *J. Comput.-Aided Mol.
Des.* 2014, DOI `10.1007/s10822-014-9747-x`, and the FreeSolv v0.5 update by
Duarte Ramos Matos et al., *J. Chem. Eng. Data* 2017, DOI
`10.1021/acs.jced.7b00104`.  Exact charge/GB literature names are recorded in
`protocol.json` for human review.

This pilot fixed ten chemical classes before calculation: alkane, aromatic
hydrocarbon, alcohol, ether, ketone, ester, nitrile, aromatic amine,
haloalkane, and sulfoxide.  Experimental values and GAFF MOL2 geometries come
from FreeSolv v0.52 at commit
`6c7d19b4b565537365ffd22006aa2cd4643200c6`.

Each molecule was charged independently with AmberTools 26.0 Antechamber using
AM1-BCC or ABCG2.  MAPLE then evaluated HCT, OBC-I, OBC-II, GBn, and GBn2 in
water with the ACE nonpolar term using OpenMM 8.4.0.post2 on the Reference
platform.  The calculation is a single-FreeSolv-geometry GBSA estimate, not a
conformational or finite-temperature free-energy average.

## Temporary experimental default: AM1-BCC/OBC-II/ACE

| class | molecule | experiment | prediction | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.000 | 1.564 | -0.436 |
| aromatic hydrocarbon | benzene | -0.900 | -1.263 | -0.363 |
| alcohol | methanol | -5.100 | -4.651 | 0.449 |
| ether | methoxymethane | -1.910 | -1.389 | 0.521 |
| ketone | acetone | -3.800 | -5.052 | -1.252 |
| ester | ethyl acetate | -2.940 | -4.796 | -1.856 |
| nitrile | acetonitrile | -3.880 | -4.535 | -0.655 |
| aromatic amine | aniline | -5.490 | -5.783 | -0.293 |
| haloalkane | chloroethane | -0.630 | -0.043 | 0.587 |
| sulfoxide | methylsulfinylmethane | -9.280 | -8.895 | 0.385 |

All energies are in `kcal/mol`.  Aggregate default metrics are MSE `-0.291`,
MAE `0.680`, RMSE `0.826`, and maximum absolute error `1.856 kcal/mol`.

## Corrected QEq-GTO diagnostic

The QEq rows use the same ten MOL2 geometries and the same OpenMM/ACE protocol,
but generate fixed charges internally with the repaired full hydrogen SCF.  At
each iteration MAPLE updates both the hydrogen idempotential and the hydrogen
screening exponent used by every H-containing GTO pair integral.  The largest
absolute atomic charge across the pilot is `0.490 e`, versus up to `3.704 e`
from the removed Open Babel-style simplification.

For the default OBC-II comparison:

| class | molecule | experiment | corrected QEq-GTO | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.000 | 1.523 | -0.477 |
| aromatic hydrocarbon | benzene | -0.900 | 0.997 | +1.897 |
| alcohol | methanol | -5.100 | -3.768 | +1.332 |
| ether | methoxymethane | -1.910 | -4.678 | -2.768 |
| ketone | acetone | -3.800 | -5.965 | -2.165 |
| ester | ethyl acetate | -2.940 | -8.249 | -5.309 |
| nitrile | acetonitrile | -3.880 | -0.414 | +3.466 |
| aromatic amine | aniline | -5.490 | 0.149 | +5.639 |
| haloalkane | chloroethane | -0.630 | -7.062 | -6.432 |
| sulfoxide | methylsulfinylmethane | -9.280 | -8.881 | +0.399 |

The corrected model is numerically stable, but its OBC-II MAE/RMSE are
`2.988/3.626 kcal/mol`, still much worse than AM1-BCC (`0.680/0.826`) and
ABCG2 (`0.963/1.233`).  This is expected evidence of a parameter-pairing
problem: the QEq charge model and Amber GB radii/nonpolar terms were not jointly
fit.  It is not evidence that the old multi-electron charges were acceptable.
The full per-atom charges, SCF/KKT diagnostics, components, and all five GB
models are frozen in
[`freesolv10-qeq-gto-full-h-2026-07-22.json`](freesolv10-qeq-gto-full-h-2026-07-22.json).

## Polarizable CQEq-GTO diagnostic

The polarizable run does not reuse the non-variational original-QEq fixed
point.  It minimizes the consistent-QEq (CQEq) energy in vacuum and minimizes
CQEq plus the GB polar quadratic form in solution.  Across all 50
molecule/model calculations, the largest solution-phase atomic charge is
`0.894 e`, the largest constrained KKT residual is `1.37e-6 eV`, and the
smallest projected charge-Hessian eigenvalue is `+1.720 eV`.  The coupled
solutions are therefore finite constrained local minima rather than the
multi-electron runaway produced by the removed simplified implementation.

The accuracy result is nevertheless negative: the best model is HCT, with
MAE/RMSE `5.719/7.748 kcal/mol`; OBC-II gives `7.219/9.654 kcal/mol`.  Allowing
the charges to relax therefore worsens this pilot relative to fixed QEq-GTO,
AM1-BCC, and ABCG2.  The implementation is retained as an experimental,
force-consistent profile, not as an accuracy-certified default, because the
QEq/CQEq parameters and Amber GB radii/nonpolar parameters were not jointly
fit.  The complete charges, KKT diagnostics, projected Hessian checks, energy
components, and all five GB models are frozen in
[`freesolv10-cqeq-gto-polarizable-2026-07-22.json`](freesolv10-cqeq-gto-polarizable-2026-07-22.json).

This small pilot must not be used to select or refit a model.  The full
supported FreeSolv benchmark and conformer/sampling analysis remain open.
