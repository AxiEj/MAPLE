# SolProp-mix QMExp v1.1 release audit — 2026-07-31

**Decision: negative; not admitted.** The evidence is pinned to the latest official Zenodo v1.1 data release, record `15587866`, and keeps three claims separate:

1. The v1.1 workbook's published `SolProp-mix_QM_Exp_GsolvT` columns reproduce their aggregate metrics when streamed over all 22,564 nonaqueous-binary, 30,030 all-binary, and 4,242 all-ternary rows. Columns are found by exact header name, so the newly inserted sigma columns cannot silently shift provenance.
2. The v1.1 workbook is broadly much closer to live execution of the ten released QMExp weights on the unchanged frozen 10-group chemical panel, but it is not an exact oracle: mean/max absolute live-to-workbook delta are `0.023138089950940088` / `0.2105751123873958` kcal/mol. Nine of ten workbook predictions changed from v1.0; the chemistry and experimental values did not.
3. Promoted-float64 CPU/GPU execution agrees within the diagnostic `5e-12` kcal/mol roundoff tolerance (maximum per-model delta `2.6645352591003757e-15` kcal/mol), but it fails the user's strict zero-loss rule: 3/10 rows have an exact floating-point error worsening, 4/10 ensemble predictions are not bitwise identical, and 55/100 individual model predictions differ at roundoff scale. GPU admission therefore remains false. Accuracy also fails: CPU MAE/RMSE/MaxAE are `0.5098988620625835` / `0.7100644511098742` / `1.6089863795855495` kcal/mol.

`panel-inputs.json` is label-free and is the only input permitted to the model worker. `selection-provenance.json` freezes the historical row identities and explicitly records that output-blind human selection is an author attestation, not something the frozen files can independently prove. `experimental-references.json` contains labels, v1.1 workbook predictions, exact sheet/row/cell/header provenance, and literature citations; it must never be supplied to inference. `v1.0-v1.1-exact-chemical-mapping.json` makes the “v1.1 is closer” comparison recomputable: 9/10 mapped rows moved closer to live released-weight execution, with mean absolute live delta falling from `0.20559967169078072` to `0.023138089950940088` kcal/mol. `raw-live-execution.json` is the deterministic inference artifact produced by `--execute-live`; only the three volatile worker stdout/stderr receipt keys are omitted, and this normalization rule plus the artifact SHA-256 and execution origin are bound in canonical `result.json`. `rows.csv` is regenerated from that canonical result. `runtime-diagnostic.json` contains volatile wall-clock observations only; no matched-QM comparison was performed and those timings are not admission evidence.

The official public prediction defaults document one solute plus at most two solvents. The adapter's three-solvent row uses a code-compatible dynamic fourth molecular slot, but that path is not explicitly documented by the official public entrypoints and is therefore not treated as an officially certified ternary API.

Regenerate `result.json` and `rows.csv` directly from exact official assets with:

```bash
python docs/pretrained-solvation-hub/run_solpropmix_qmexp_release_audit.py \
  --workbook /path/to/solprop-mix_v1.1.xlsx \
  --execute-live \
  --source-root /path/to/exact/pinned/solprop-git-checkout \
  --static-root /path/to/extracted/SolProp_ML-StaticCodeGsolv \
  --weights-root /path/to/extracted/ModelWeights/SolPropmixQMExp \
  --python-executable /path/to/compatible/python
```

The canonical result deliberately omits volatile worker stdout/stderr hashes; those stream diagnostics do not define model identity and can change between otherwise equivalent reruns. The source, supplemental modules, ten checkpoints, adapter, worker, taxonomy, workbook, and panel remain hash-pinned in `result.json`.

`--live-results` is intentionally **not** a canonical regeneration path. It accepts a trusted imported fixture only for parser/diagnostic work, records the exact input-file SHA-256 and `imported_trusted_fixture` origin, forces every scientific/live-execution/admission claim false, writes `imported-fixture-result.json` plus `imported-fixture-rows.csv`, and refuses to target this canonical benchmark directory. Coordinated edits to ensemble members, their means, and the joined CPU/GPU values can therefore remain internally consistent but cannot be labeled `live_released_weight_execution` or replace canonical evidence.
