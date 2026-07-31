# SolProp-mix QMExp v1.1 broad audit (2026-07-31)

This is a **development/non-blind** audit of the released `SolPropmixQMExp`
ensemble. Public training overlap is unknown. It is not a final blind holdout and
it does not admit a PES, forces, optimization, frequency, MD, or free-energy
protocol.

## Main supported lane

The main evidence is a deterministic, model-output-blind uniform sample of 1,000
supported rows from the v1.1 nonaqueous pure-solvent and binary-solvent sheets.
The selection implementation reads only input columns, freezes exact sheet row
indices, and only then reads experimental labels from the frozen rows. The panel
contains 15 primary functional groups. `rows.csv` records, for every row, exact
`sheet!ExcelRow`, input-cell addresses, experiment, CPU/GPU prediction, signed
error, absolute error, and CPU/GPU delta.

Strict CPU float64 results (kcal/mol):

| n | groups | MAE | RMSE | MaxAE |
|---:|---:|---:|---:|---:|
| 1000 | 15 | 0.2600320951491671 | 0.3768962032607486 | 1.8977049801370343 |

The frozen admission thresholds are MAE <= 0.25 and RMSE <= 0.37. Both fail.
Therefore matched-QM speed testing is ineligible; wall-clock measurements cannot
be used to admit this route.

## GPU precision boundary

Both devices used promoted float64 with deterministic algorithms and TF32, AMP,
FP16, BF16, and reduced-precision reductions disabled. GPU and CPU ensemble
predictions differ bitwise on 496/1000 rows. GPU absolute error is exactly worse
on 218/1000 rows, although zero rows worsen by more than the separate 5e-12
roundoff diagnostic. Thus strict zero-loss is **false** and GPU admission is
**false**. The 5e-12 diagnostic does not override the exact criterion.

## Adapter crosscheck and identities

`adapter-crosscheck.json` compares three pure and three binary rows on CPU and
GPU against the final adapter, including every ensemble member. Every ensemble
and per-model delta is <= 5e-12. `result.json` binds the Git revision/tree,
v1.1 workbook, ten checkpoint hashes, four supplemental-module hashes, taxonomy,
runner, adapter, and worker. Cache locations are execution details, not artifact
identity.

## Ternary diagnostic

`ternary-rows.csv` is a separate 100-row **undocumented dynamic-slot diagnostic**.
It is excluded from the main lane and all admission decisions. CPU MAE/RMSE/MaxAE
are 0.18568599085530849 / 0.2630909605347411 / 1.426683398055653 kcal/mol.
Mean/max absolute live-vs-workbook deltas are
0.06220292400566668 / 0.38707916302832324 kcal/mol.

## Files

- `panel-inputs.json`: label-free main-panel model inputs.
- `experimental-references.json`: separately stored labels.
- `rows.csv`: independently recomputable 1,000-row main evidence.
- `ternary-rows.csv`: separately scoped 100-row diagnostic.
- `adapter-crosscheck.json`: 3 pure + 3 binary final-adapter checks on both devices.
- `result.json`: compact identities, aggregate results, gates, and file hashes.
- `../../run_solpropmix_qmexp_broad_audit.py`: rerunner accepting explicit official
  v1.1 workbook, source, supplemental-code, weights, Python, and output paths.

The canonical result intentionally excludes wall-clock times, absolute cache
paths, stream hashes, and redundant per-model row JSON.
