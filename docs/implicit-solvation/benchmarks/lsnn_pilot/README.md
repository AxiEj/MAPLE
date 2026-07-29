# LSNN-v1 × FreeSolv compatibility pilot

This directory runs a real, pinned LSNN-v1 state dict through OpenMM molecular
dynamics and PyMBAR. It is a real public-weight compatibility experiment, not
an upstream reproduction or production MAPLE backend.

## Why a reconstruction is required

At upstream revision
`1768d068dcb1ea65e8585af3f4a0cbf4047d9125`, the public LSNN-v1 repository:

- publishes the model state dict;
- publishes the model graph and intended OpenMM/MBAR workflow;
- does not publish the `environment.yml` named in its README;
- has multiple executable errors in `solv.py` and `methods/LSNN.py`.

`lsnn_model.py` therefore reconstructs the public MIT-licensed energy graph
using plain PyTorch operations. It preserves the state-dict keys and energy
equations but intentionally uses the conservative negative gradient of the
full energy. The pinned upstream graph detaches its electrostatic energy before
returning explicit forces, so the two force conventions are not identical.
The upstream copyright and MIT permission notice are retained verbatim in
[`LICENSE.LSNN-v1`](LICENSE.LSNN-v1).
The production `lsnn-v1` model card and audited-domain gate remain unchanged.

The vacuum Hamiltonian is also different: this pilot uses FreeSolv's archived
GAFF/AM1-BCC `prmtop` files, whereas upstream creates an OpenFF system through
`SMIRNOFFTemplateGenerator`. Results must therefore be labelled
`LSNN-conservative + GAFF compatibility`, not LSNN paper reproduction.

## Fixed pilot

The panel is declared before execution:

- methane (`mobley_9055303`);
- methanol (`mobley_1636752`);
- benzene (`mobley_3053621`).

Inputs and experimental labels are pinned to FreeSolv revision
`6c7d19b4b565537365ffd22006aa2cd4643200c6`. GAFF/AM1-BCC topology and
coordinates come from its `amber.tar.gz`. The panel contains no experimental
values; the script loads and verifies `database.txt` only after all label-free
predictions have been sealed.

The default compatibility protocol uses a denser five-state path, a conservative
1 fs timestep for the unconstrained archived GAFF systems, 0.2 ps
equilibration, and 1.0 ps production per window. The formal run recorded below
explicitly increases production to 2.0 ps per window:

```text
(lambda_sterics, lambda_electrostatics)
(0, 0) -> (0.5, 0) -> (1, 0) -> (1, 0.5) -> (1, 1)
```

Three independent seeds are run. Samples are decorrelated per window and
analysed with PyMBAR 4. A result is excluded from accuracy metrics if
decorrelation falls back, a window retains fewer than 20 frames, adjacent MBAR
overlap is below 0.03, replicate SD exceeds 0.5 kcal/mol, or the OpenMM-Torch
force fails its finite-difference check.

## Isolated environment

The execution used a dedicated virtual environment rather than adding LSNN
dependencies to MAPLE's core package:

```bash
python -m venv ~/.cache/maple-envs/lsnn-pilot-venv
~/.cache/maple-envs/lsnn-pilot-venv/bin/pip install \
  'numpy<2' 'torch==2.9.1' 'openmm==8.5.2' \
  'openmmtorch==1.5' 'pymbar==4.0.3'
```

## Command

```bash
~/.cache/maple-envs/lsnn-pilot-venv/bin/python \
  docs/implicit-solvation/benchmarks/lsnn_pilot/run_lsnn_pilot.py \
  --upstream-repo ~/.cache/maple-benchmarks/LSNN-v1-1768d068 \
  --freesolv-repo /path/to/FreeSolv-at-pinned-revision \
  --work-dir .omx/benchmarks/lsnn-v1-freesolv-real-pilot-20260729
```

The output includes:

- `summary.json`;
- `records.csv`;
- label-free per-molecule prediction records;
- model/data/runtime SHA256 and environment lock;
- MBAR uncertainty, state counts, overlap matrix, and decorrelation diagnostics.

## Actual formal run

The frozen run
[`lsnn-panel3-formal-audited-5state-2ps-3seed-20260730-v1`](results/lsnn-panel3-formal-audited-5state-2ps-3seed-20260730-v1/)
used five lambda states, 0.2 ps equilibration and 2.0 ps production per state,
10-step sampling, and three unique seeds at 300 K. It executed public weights,
OpenMM MD, and PyMBAR rather than substituting literature values.

| molecule | prediction (kcal/mol) | experiment (kcal/mol) | signed error | protocol status |
| --- | ---: | ---: | ---: | --- |
| methane | 1.0521 +/- 0.0034 | 2.00 +/- 0.20 | -0.9479 | ok |
| methanol | -3.4771 +/- 0.0179 | -5.10 +/- 0.60 | +1.6229 | ok |
| benzene | -1.0318 +/- 0.0042 | -0.90 +/- 0.20 | -0.1318 | unconverged |

The prediction uncertainty is a sampling-only heuristic, not a calibrated
model-error interval. Benzene failed because two seeds retained fewer than 20
decorrelated frames in at least one state. It is therefore excluded from the
formal aggregate:

- protocol coverage: 2/3;
- conditional MAE: 1.2854 kcal/mol;
- conditional RMSE: 1.3290 kcal/mol;
- conditional maximum absolute error: 1.6229 kcal/mol.

The formal pilot **fails** the proposed maximum-error acceptance threshold of
1.5 kcal/mol because of methanol. Including the unconverged benzene estimate
would give a descriptive three-point MAE of 0.9009 kcal/mol, but that is not an
accepted panel metric and must not be used to claim sub-kcal/mol accuracy.

All nine raw `u_kln` artifacts are frozen with the result. Independent PyMBAR
recomputation reports `verified=true`, `raw_artifact_count=9`, and
`mismatch_count=0`. The runner returned exit status 1 as intended because the
panel did not pass all quality gates.

## Claim boundary

This is an actual public-weight MD/MBAR experiment, but it is not a strict
external holdout. The LSNN paper says FreeSolv similarity influenced its test
split, while the public repository does not expose an exact membership
manifest. It also does not reproduce upstream's vacuum Hamiltonian or force
convention. No fitting, calibration, or experimental-label adjustment is
performed here. This run tests hydration free energy for three neutral small
molecules only; it does not test binding free energy, ions, nonaqueous solvents,
proteins, or production convergence.
