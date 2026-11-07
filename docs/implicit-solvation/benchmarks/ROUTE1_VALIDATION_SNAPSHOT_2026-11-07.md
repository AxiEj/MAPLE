# Route-1 validation snapshot, 2026-11-07

This is a research reference snapshot, not a production-admission notice.
The validated implementation is on `feature/route1-advanced-paths-20261101`.
Its reviewed source is `c7f3c17ccb5b42f64d6c07b7090055dec646fd65` and its data-only delivery is `17080aac539e3c0239cb38147960698c2ab35b4c`.
Publishing this report does not merge that implementation into `implicitsolv-route1` or qualify unrelated working-tree changes.

## Local advanced-task acceptance

The accepted matrix contains 70 rows: 48 positive cases, 20 negative contracts and two CHA single-point cases.
It combines 64 unchanged original passes with six new passes under the prospectively approved amendment.
The original campaign remains 64 passes and six failures. No old failure was overwritten or relabelled.

Both water fixtures used native MAPLE input, engine, OPT/L-BFGS, SP and FREQ. L-BFGS converged in seven steps.
The native MAPLE criteria and physical parameters were unchanged.
The protocol no longer imposed the auxiliary `1e-5` atom-force-norm criterion, as explicitly approved before the amended run.
All 82 preservation checks passed. The 173 installed MAPLE files and 65 distribution versions retained their reviewed identities.

These are bounded NH3/H2O canaries, not general reaction-search evidence.
The NEB and CI segments were preconverged zero-update cases; P-RFO refinement took one or two steps.
The water P-RFO negative control was a one-step certification check.
The public native FREQ return did not expose a structured raw Hessian.
No charged, open-shell, periodic, large-molecule, MD or general chemical-accuracy qualification is implied.

The machine-readable [local acceptance summary](route1_advanced_local_acceptance_2026-11-06.json) binds the retained evidence hashes.

## Matched ABCG2 and AM1-BCC CHA accuracy

The panel contains the same 526 neutral FreeSolv development molecules in water, with fixed original geometries.
Both arms use native MAPLE SP calls and the same private Torch FP32 CUDA CHA-GB/ALPB plus cavity-dispersion chain.
Only the fixed charge vector changes. Coordinates and non-charge topology agree for every pair.
The cavity and dispersion terms are bitwise equal for all 526 pairs.
The scored observable is the native serialized `EGB + ECAVITY + EDISPER`, not the gas-plus-solvent MLIP total.

All values below are in kcal/mol. Bias means prediction minus experiment.

| Charge model | N | MAE | RMSE | Bias | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| AM1-BCC | 526 | 1.321854 | 1.854206 | -0.159727 | 9.416506 |
| ABCG2 | 526 | 1.204749 | 1.670036 | -0.573645 | 8.455666 |

ABCG2 reduces this panel's MAE by 8.86% and RMSE by 9.93%.
It improves 291 cases, worsens 178 and ties 57. Its mean signed bias is more negative.
These differences do not establish statistical significance or superiority outside this panel.

The final matrix has 1,052 successful records: 216 reused successes, 555 recovery successes and 281 continuation successes.
The earlier 65 CUDA-workspace failures and one interrupted attempt remain preserved separately.
Previously accepted SP records in this energy matrix were reused without recomputation. No source, physical parameter or numerical threshold was changed to improve accuracy.

The [accuracy summary](route1_abcg2_cha_freesolv526_2026-11-07.json) includes the original bootstrap intervals, scope and provenance.
The [per-molecule predictions](route1_abcg2_cha_freesolv526_predictions_2026-11-07.csv) include geometry, charge and record hashes.
Reference labels are not redistributed. Their pinned database hash is recorded in the summary.
The [existing physical protocol](chagb_nonpolar_protocol.json) remains the parameter reference.

## Interpretation and access boundaries

This is development-only candidate evidence, not a blind confirmation.
A review-side broad hash scan may have read database bytes before the energy seal, without parsing or displaying label values.
That deviation is retained. A separate allowlisted review checked the full matrix before the scorer parsed labels.
The deviation does not alter the fixed calculations, but it prevents a whole-workflow non-access claim.

Private Torch CHA remains prepared-topology SP energy-only. These results do not open forces, OPT, FREQ, TS, paths or MD for that provider.
The OBC-II task qualifications do not transfer to CHA or arbitrary model/device/precision combinations.
The private engine and model weights are not included. The public adapter alone is not a complete energy-reproduction environment.
No default model, runtime gate or production-admission state changes in this data publication.
