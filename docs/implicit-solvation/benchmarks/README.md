# Implicit-solvation benchmarks

## Route 2: MACE-POLAR-1-M/SMD/IEFPCM

[`route2-protocol.json`](route2-protocol.json) is the public Route-2 standard.
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
  run --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --device cpu
python docs/implicit-solvation/benchmarks/run_route2_freesolv.py \
  summarize --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --output "$WORK/development-summary.json"
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

No Route-2 FreeSolv accuracy artifact is frozen yet. The real water and static
SMD-CDS controls in `VALIDATION_STATUS.md` prove execution, units, reciprocity,
and convergence only. Until development and one-shot confirmation pass, the
public input continues to require `experimental=true`; no constant shift,
fine-tuning, or confirmation-set refit is allowed.

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
  run --protocol "$PROTOCOL" --work-dir "$WORK" --partition development
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

## Aggregate comparison

| charge | GB | MSE | MAE | RMSE | max absolute error |
|---|---|---:|---:|---:|---:|
| AM1-BCC | HCT | -0.288 | 0.741 | 0.864 | 1.525 |
| AM1-BCC | OBC-I | -0.292 | **0.663** | 0.845 | 1.936 |
| AM1-BCC | OBC-II | -0.291 | 0.680 | **0.826** | 1.856 |
| AM1-BCC | GBn | -0.345 | 0.666 | 0.923 | 2.150 |
| AM1-BCC | GBn2 | -0.290 | 0.792 | 1.121 | 2.926 |
| ABCG2 | HCT | -0.825 | 1.026 | 1.469 | 3.226 |
| ABCG2 | OBC-I | -0.845 | 1.028 | 1.277 | 3.073 |
| ABCG2 | OBC-II | -0.822 | **0.963** | **1.233** | 3.066 |
| ABCG2 | GBn | -0.784 | 1.185 | 1.388 | 3.119 |
| ABCG2 | GBn2 | -0.795 | 1.020 | 1.417 | 3.426 |
| corrected QEq-GTO | HCT | -0.167 | **2.876** | **3.416** | 5.956 |
| corrected QEq-GTO | OBC-I | -0.439 | 3.088 | 3.779 | 6.452 |
| corrected QEq-GTO | OBC-II | -0.442 | 2.988 | 3.626 | 6.432 |
| corrected QEq-GTO | GBn | -0.459 | 3.259 | 3.945 | 6.452 |
| corrected QEq-GTO | GBn2 | -0.266 | 2.893 | 3.578 | 7.041 |
| polarizable CQEq-GTO | HCT | -4.578 | **5.719** | **7.748** | 14.676 |
| polarizable CQEq-GTO | OBC-I | -7.151 | 8.159 | 10.942 | 21.036 |
| polarizable CQEq-GTO | OBC-II | -6.209 | 7.219 | 9.654 | 17.841 |
| polarizable CQEq-GTO | GBn | -6.446 | 7.586 | 9.966 | 18.461 |
| polarizable CQEq-GTO | GBn2 | -6.207 | 7.267 | 10.198 | 23.619 |

The complete AM1-BCC/ABCG2 machine-readable records, source-member hashes,
energy components, charge sums, and per-atom conservation corrections are in
[`freesolv10-2026-07-22.json`](freesolv10-2026-07-22.json).

## Literature-tuned EE/GB diagnostic: ESE-GB-DNN

ESE-GB-DNN is the closest published correction to the failed generic QEq/Amber
GB pairing found so far.  It is a complete, jointly trained predictor rather
than a charge generator: geometry and total charge produce
coordination-dependent electronegativity-equalization (EE) charges, GB-derived
features, surface/volume features, and finally a dense-neural-network
prediction of the complete solvation free energy.  Its printed charges must
therefore **not** be detached and presented as an independently validated
OpenMM-GB charge model.

The upstream Linux executable at commit
`5b8cdf1f52e52b5bad5351df48a30df00b9a00bc` (SHA256
`18701e01b0a17edffb80462375faf08d800502589230ed31747a7e2b4bf0b317`)
was run externally on exactly the same ten FreeSolv geometries:

| class | molecule | experiment | ESE-GB-DNN | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.00 | 1.14 | -0.86 |
| aromatic hydrocarbon | benzene | -0.90 | -0.49 | +0.41 |
| alcohol | methanol | -5.10 | -4.31 | +0.79 |
| ether | methoxymethane | -1.91 | -0.93 | +0.98 |
| ketone | acetone | -3.80 | -2.67 | +1.13 |
| ester | ethyl acetate | -2.94 | -3.61 | -0.67 |
| nitrile | acetonitrile | -3.88 | -3.28 | +0.60 |
| aromatic amine | aniline | -5.49 | -6.61 | -1.12 |
| haloalkane | chloroethane | -0.63 | -0.30 | +0.33 |
| sulfoxide | methylsulfinylmethane | -9.28 | -5.75 | +3.53 |

The ten-molecule MSE/MAE/RMSE are `+0.512/1.042/1.357 kcal/mol`.  This is a
large improvement over the corrected generic QEq-GTO/HCT pairing
(`2.876/3.416` MAE/RMSE), but is not better than this pilot's AM1-BCC or ABCG2
rows.  DMSO is the dominant outlier; excluding it gives MAE/RMSE
`0.766/0.813 kcal/mol`, which is diagnostic only and not a permitted benchmark
selection.

The same executable completed all 642 neutral FreeSolv v0.52 records with
MSE/MAE/RMSE `-0.025/0.775/1.116 kcal/mol`.  This full-set result is recorded as
a **screening reproduction, not an independent generalization statistic**:
the paper's aqueous training data combine CombiSolv-QM with a random half of
MNSol, so overlap with FreeSolv is possible and has not been resolved molecule
by molecule.  The paper's explicitly independent Mobley-141 result is RMSE
`1.30 kcal/mol`.  A newer, substantially harder FlexiSol evaluation reports
ESE-GB-DNN MAE/RMSE `3.3/5.1 kcal/mol` on 519 retained solvation-energy records
under its GFN2-xTB ensemble protocol, demonstrating that small rigid-molecule
accuracy does not transfer unchanged to flexible, chemically complex systems.

Machine-readable records:

- [`freesolv10-ese-gb-dnn-2026-07-22.json`](freesolv10-ese-gb-dnn-2026-07-22.json)
- [`freesolv642-ese-gb-dnn-screening-2026-07-22.json`](freesolv642-ese-gb-dnn-screening-2026-07-22.json)

No ESE executable is stored or downloaded by MAPLE.  The upstream repository
publishes binaries but no source code, tests, or explicit software license;
the current evidence therefore permits a research benchmark with a
user-supplied executable, not redistribution or a production provider.  The
program returns only a scalar energy (printed to `0.01 kcal/mol`) and provides
no forces, gradients, or MD interface, so it cannot satisfy MAPLE's OPT/PES
contract.

References: Vyboishchikov, “Predicting Solvation Free Energies Using
Electronegativity-Equalization Atomic Charges and a Dense Neural Network: A
Generalized-Born Approach,” *JCTC* (2023), DOI
`10.1021/acs.jctc.3c00858`; Wittmann, Selzer, and Grimme, “A diverse and
chemically relevant solvation model benchmark set with flexible molecules and
conformer ensembles,” *Chemical Science* (2025), DOI `10.1039/D5SC06406F`.

## Open EEQ control: Kallisto and Jazzy

Kallisto provides a modern electronegativity-equilibration (EEQ) model whose
parameters were fitted to PBE0/def2-TZVP Hirshfeld charges.  Jazzy combines
those charges with published polar, apolar, and interaction terms fitted to
experimental hydration free energies.  Unlike ESE-GB-DNN, both Jazzy 0.1.4 and
Kallisto 1.0.10 are available as source under Apache-2.0, so this is the
cleanest permissively licensed control for separating two questions:

1. does a modern gas-phase EEQ charge model become accurate when inserted into
   MAPLE's Amber GB profiles; and
2. does a complete hydration-fitted model built on those same EEQ charges do
   better?

For the first question, the Kallisto charges were computed at each submitted
FreeSolv geometry, mapped back to MOL2 atom order, frozen, and evaluated with
the same five MAPLE OpenMM/ACE GB models:

| charge | GB | MSE | MAE | RMSE | max absolute error |
|---|---|---:|---:|---:|---:|
| Kallisto EEQ | HCT | +0.118 | **1.869** | **2.218** | 3.595 |
| Kallisto EEQ | OBC-I | -0.027 | 2.000 | 2.324 | 3.976 |
| Kallisto EEQ | OBC-II | -0.070 | 1.956 | 2.281 | 3.830 |
| Kallisto EEQ | GBn | -0.029 | 2.242 | 2.592 | 4.423 |
| Kallisto EEQ | GBn2 | -0.060 | 2.207 | 2.620 | 5.195 |

This is substantially better than corrected QEq-GTO/HCT
(`2.876/3.416 kcal/mol` MAE/RMSE), but it remains worse than AM1-BCC and ABCG2
on the same ten-molecule protocol.  Kallisto was fitted to electronic charges,
not jointly to Amber radii and ACE, so these rows are a diagnostic cross-pair,
not a new certified MAPLE charge profile.

For the second question, the complete Jazzy hydration model was evaluated on
the same ten coordinates without embedding or force-field minimization:

| class | molecule | experiment | Jazzy | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.00 | 0.874 | -1.126 |
| aromatic hydrocarbon | benzene | -0.90 | -1.178 | -0.278 |
| alcohol | methanol | -5.10 | -3.854 | +1.246 |
| ether | methoxymethane | -1.91 | -2.998 | -1.088 |
| ketone | acetone | -3.80 | -3.467 | +0.333 |
| ester | ethyl acetate | -2.94 | -4.632 | -1.692 |
| nitrile | acetonitrile | -3.88 | -3.363 | +0.517 |
| aromatic amine | aniline | -5.49 | -5.981 | -0.491 |
| haloalkane | chloroethane | -0.63 | -0.959 | -0.329 |
| sulfoxide | methylsulfinylmethane | -9.28 | -2.998 | +6.282 |

The ten-molecule MSE/MAE/RMSE are `+0.337/1.338/2.171 kcal/mol`.  Seven of the
ten are exact structural matches to the Gerber table used to fit Jazzy, and
DMSO exposes a `+6.282 kcal/mol` functional-group failure.  This pilot is
therefore not an independent accuracy result.

On all 642 FreeSolv records, the same geometry-preserving protocol completed
without runtime failure.  A conservative overlap audit used canonical
isomeric and non-isomeric SMILES plus full and connectivity-block InChIKeys:

| relation to Jazzy's Gerber fitting table | N | MSE | MAE | RMSE | max absolute error |
|---|---:|---:|---:|---:|---:|
| all records | 642 | -0.608 | 1.552 | 2.433 | 12.756 |
| exact or connectivity-level overlap | 245 | -0.035 | 0.819 | 1.114 | 4.682 |
| strict identifier non-overlap | 397 | -0.961 | 2.004 | 2.968 | 12.756 |

“Strict identifier non-overlap” means only that none of the four recorded
identifiers matched the official Gerber fitting file; it does not prove the
absence of close-analogue leakage.  The large overlap/non-overlap gap is still
sufficient to reject Jazzy as a general replacement for MAPLE's fixed-charge
GB route.  Its published external Guthrie validation reports MAE/RMSE
`5.89/9.19 kJ/mol` (about `1.41/2.20 kcal/mol`).

Machine-readable records:

- [`freesolv10-kallisto-eeq-openmm-gb-2026-07-22.json`](freesolv10-kallisto-eeq-openmm-gb-2026-07-22.json)
- [`freesolv10-jazzy-eeq-2026-07-22.json`](freesolv10-jazzy-eeq-2026-07-22.json)
- [`freesolv642-jazzy-eeq-screening-2026-07-22.json`](freesolv642-jazzy-eeq-screening-2026-07-22.json)

The compatibility environment used Jazzy 0.1.4/Kallisto 1.0.10 with RDKit
2024.9.2 and NumPy 2.3.5, rather than Jazzy's RDKit 2024.3.1 pin and Kallisto's
declared NumPy `<2` bound; the artifact records that deviation explicitly.
Jazzy exposes scalar hydration predictions but no force/gradient API.  Its
license is acceptable, but its accuracy and capabilities do not pass MAPLE's
OPT/PES production gate.  MAPLE therefore records it as a research control and
does not add a runtime provider.

Reference: Ghiandoni and Caldeweyher, “Fast calculation of hydrogen-bond
strengths and free energy of hydration of small molecules,” *Scientific
Reports* **13**, 4143 (2023), DOI `10.1038/s41598-023-30089-x`.

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

## Decision after the tuned-EE audit

1. **Keep AM1-BCC/OBC-II/ACE as the temporary experimental default.**  It has
   the best same-protocol ten-molecule RMSE (`0.826 kcal/mol`).
2. **Keep ABCG2 as a supported alternative, not as proof of GB superiority.**
   Its strongest published full-FreeSolv result (`0.57/0.99 kcal/mol`
   MUE/RMSE) comes from GAFF2, explicit TIP3P water, MD, and alchemical free
   energies, not this single-geometry implicit-GB protocol.
3. **Keep native QEq-GTO/CQEq experimental and nondefault.**  Correct
   mathematics did not make the unmatched QEq/Amber-GB parameter combination
   accurate.
4. **Do not extract ESE-GB-DNN charges into OpenMM GB.**  Its improvement comes
   from the jointly trained EE/GB-feature/surface/DNN model.  The current
   binary license and lack of forces prevent production integration.
5. **Do not add Jazzy/Kallisto as a production solvation provider.**  The open
   implementation is valuable as a control, but the strict-nonoverlap error
   and missing gradients fail the scientific and OPT/PES gates.

The positive result is narrower but useful: literature tuning can make an EE
family model roughly competitive for ordinary small neutral molecules.  The
negative result is equally important: no audited candidate yet combines that
accuracy with permissive redistribution, force continuity, OPT/PES support,
and demonstrated transferability to flexible out-of-domain chemistry.
