# Route 1 exploratory status — 2026-09-05

## Current policy

For new Route 1 exploratory pilots, experimental error is now **descriptive**.
There is no mandatory per-record or aggregate experimental-accuracy threshold:
the machine-readable field
`experimental_accuracy_threshold_kcal_mol` is `null`. MAE, RMSE, bias,
maximum absolute error, and complete success/failure denominators must still be
reported, but none of those values alone promotes or rejects a method.

This prospective policy is frozen in
[`benchmarks/route1_exploratory_policy_v1.json`](benchmarks/route1_exploratory_policy_v1.json).
It supersedes the old `<1.5` and `<1.0 kcal/mol` targets only for new
exploratory decisions made on or after 2026-09-05. Historical protocols,
artifacts, scores, and the decision rules under which they were recorded are
not edited, resealed, or retrospectively relabelled.

Relaxing the experimental-error target does **not** relax numerical or
provenance gates. Finite values, exact scalar-energy/force consistency,
step-size convergence, all-`3N` finite differences, Hessian validity, genuine
TS semantics, manifest seals, hashes, and complete failure accounting remain
required.

## SP / OPT / FREQ / TS status

The requested task scope is SP, OPT, FREQ, and TS. It is a goal, not a claim of
current delivery for the high-accuracy endpoint:

| Endpoint | SP | OPT | FREQ | TS |
|---|---:|---:|---:|---:|
| AM1-BCC/CHA-GB/PBSA cavity-dispersion | available | **not delivered** | **not delivered** | **not delivered** |

The current CHA-GB/PBSA provider is scalar-energy-only. A current-source,
benchmark-only all-`3N` centered-difference probe at `0.001`, `0.003`, and
`0.01 A` found adjacent-step force RMS differences of `3.6007` and `1.5543
kcal/mol/A` (maxima `11.8` and `4.35 kcal/mol/A`). That is not a
step-converged force result, so no force API or OPT/FREQ/TS capability is
promoted. A TS result will require a consistent composed force and Hessian,
convergence to a saddle, and exactly one imaginary vibrational mode; accepting
a task keyword or dispatching a job would not be sufficient.

Existing OBC-II/ACE task plumbing is a different, explicitly identified
force-capable water baseline. It is not silently substituted for CHA-GB/PBSA
and does not inherit the latter's fixed-geometry accuracy measurements.

## Fifteen-solvent, 20-distinct-solute pilot

The label-free MNSol source audit contains 15 requested solvents. Applying the
rule `min(20, eligible distinct FileHandles)` produces **222 selectable pairs
out of 300 requested**: eight solvents have all 20 and seven have explicit
shortfalls.

| solvent | eligible | requested | selectable | shortfall |
|---|---:|---:|---:|---:|
| water | 390 | 20 | 20 | 0 |
| ethanol | 8 | 20 | 8 | 12 |
| acetonitrile | 7 | 20 | 7 | 13 |
| dimethylsulfoxide | 7 | 20 | 7 | 13 |
| dimethylformamide | 7 | 20 | 7 | 13 |
| tetrahydrofuran | 7 | 20 | 7 | 13 |
| chloroform | 109 | 20 | 20 | 0 |
| dichloromethane | 11 | 20 | 11 | 9 |
| toluene | 51 | 20 | 20 | 0 |
| hexane | 59 | 20 | 20 | 0 |
| cyclohexane | 92 | 20 | 20 | 0 |
| diethyl ether | 72 | 20 | 20 | 0 |
| 1-octanol | 247 | 20 | 20 | 0 |
| ethyl acetate | 24 | 20 | 20 | 0 |
| nitrobenzene | 15 | 20 | 15 | 5 |

This is source-panel preparation only. **No new solvent energies or accuracy
scores have been produced.** Selection is label-independent and distinct by
canonical `FileHandle`; short solvents are not padded with duplicates, ions,
transfer records, or replacements chosen after evaluation.

The current product parameters are water-only. Nonwater calculations remain
blocked until each solvent has a frozen physical model and parameter set,
thermodynamic state, standard-state convention, and provenance. Reusing water
parameters under another solvent name, or changing only a dielectric constant,
would not constitute a valid multi-solvent test.

## Completion ledger

- **Completed:** prospective no-fixed-error-threshold policy; retained
  numerical/provenance gates; deterministic 15-solvent source-panel target and
  explicit 222/300 coverage accounting.
- **Not completed:** CHA-GB/PBSA forces; its OPT, FREQ, or TS tasks; nonwater
  physical providers; 222 energy calculations; multi-solvent scoring.
