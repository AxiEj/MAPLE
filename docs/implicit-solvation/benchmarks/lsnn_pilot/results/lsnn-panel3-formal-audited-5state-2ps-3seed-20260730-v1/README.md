# Formal LSNN compatibility pilot result

This directory freezes the actual output of a three-molecule FreeSolv
compatibility experiment. The scientific identity is:

```text
LSNN-v1 conservative energy reconstruction
+ FreeSolv archived GAFF/AM1-BCC vacuum Hamiltonian
+ OpenMM MD
+ PyMBAR 4
```

It is not an upstream LSNN reproduction, a strict independent holdout, or a
binding-free-energy benchmark.

## Protocol

- FreeSolv revision: `6c7d19b4b565537365ffd22006aa2cd4643200c6`
- LSNN-v1 revision: `1768d068dcb1ea65e8585af3f4a0cbf4047d9125`
- temperature: 300 K
- lambda states: `(0,0)`, `(0.5,0)`, `(1,0)`, `(1,0.5)`, `(1,1)`
- equilibration: 200 steps/state at 1 fs
- production: 2000 steps/state at 1 fs
- sampling: every 10 steps
- seeds: `20260729`, `20260730`, `20260731`
- OpenMM platform: CPU with 10 threads

The recorded command was:

```bash
OPENMM_CPU_THREADS=10 \
/home/axie/.cache/maple-envs/lsnn-pilot-venv/bin/python \
  docs/implicit-solvation/benchmarks/lsnn_pilot/run_lsnn_pilot.py \
  --upstream-repo /home/axie/.cache/maple-benchmarks/LSNN-v1-1768d068 \
  --freesolv-repo /home/axie/MAPLE/MAPLE-implicitsolv-route1/.omx/vendor-audits/freesolv-current-20260726-v1 \
  --work-dir /home/axie/.cache/maple-benchmarks/runs/lsnn-panel3-formal-audited-5state-2ps-3seed-20260730-v1 \
  --seeds 20260729,20260730,20260731 \
  --lambda-values 0,0.5,1 \
  --temperature-kelvin 300 \
  --equilibration-steps 200 \
  --production-steps 2000 \
  --sample-interval 10 \
  --timestep-fs 1 \
  --platform CPU
```

## Experiment comparison

| FreeSolv ID | molecule | prediction (kcal/mol) | experiment (kcal/mol) | signed error | absolute error | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `mobley_9055303` | methane | 1.0521 +/- 0.0034 | 2.00 +/- 0.20 | -0.9479 | 0.9479 | ok |
| `mobley_1636752` | methanol | -3.4771 +/- 0.0179 | -5.10 +/- 0.60 | +1.6229 | 1.6229 | ok |
| `mobley_3053621` | benzene | -1.0318 +/- 0.0042 | -0.90 +/- 0.20 | -0.1318 | 0.1318 | unconverged |

The prediction uncertainty combines within-seed MBAR variance and between-seed
SEM as a sampling-only heuristic. It is not an estimate of model error.

Benzene failed `minimum_decorrelated_frames_per_window >= 20` in seeds
`20260730` and `20260731`. The formal metrics therefore cover two of three
molecules:

```text
coverage                         2/3
conditional MAE                  1.2854339823 kcal/mol
conditional RMSE                 1.3290022542 kcal/mol
conditional maximum abs. error   1.6229343065 kcal/mol
```

**Acceptance verdict: fail.** Methanol exceeds the proposed maximum absolute
error limit of 1.5 kcal/mol. The all-three descriptive MAE is
0.9008953280 kcal/mol, but it includes an unconverged result and is not a valid
formal accuracy claim.

## Reproducibility evidence

- `summary.json`: full protocol, pins, diagnostics, and claims
- `records.csv`: compact result table
- `cases/*/prediction-label-free.json`: predictions sealed before labels
- `cases/*/raw-u-kln-seed-*.npz`: all nine raw energy tensors
- `recompute.json`: independent PyMBAR recomputation
- `environment.lock.json`: runtime package versions
- `run.log`, `time.txt`, `launcher-environment.txt`: launcher evidence
- `SHA256SUMS`: hashes of every frozen evidence file except itself

The frozen-artifact recomputation result is:

```text
verified=true
raw_artifact_count=9
mismatch_count=0
```

## Claim boundary

- Exact LSNN training membership is not public, so leakage status is unknown.
- The vacuum Hamiltonian is GAFF/AM1-BCC rather than upstream OpenFF.
- Forces are the conservative gradient of the reconstructed full scalar energy;
  upstream detaches its electrostatic term when returning explicit forces.
- The panel has only three neutral molecules in water and one failed protocol
  gate.
- No fitting, calibration, or label-based adjustment was performed.
- No solute binding free energy was calculated.
