# Frozen AIMNet2 charge MNSol pilot: current-head replay

This bundle retains two independent clean-process replays, from current commit
`b758aede9d52693c2e45412c5f4582eee4ef0300`, of the preregistered ten-record
MNSol-v2012 AIMNet2 pilot. Both public artifacts are aggregate-only and contain
no redistributed MNSol rows or coordinates. The private row-level outputs stay
under `.omx` and are not committed.

## Scientific identity

- model: the same unchanged AIMNet2 wB97M-D3 checkpoint used by the current
  frozen-charge Route-2 work, SHA-256
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`;
- source: one zero-field AIMNet2 NQE point-charge evaluation per fixed MNSol
  geometry; no continuum field is supplied to AIMNet2 and no electronic SCF is
  performed;
- energy: fixed-source pyddx ddPCM or separately named scaled ddCOSMO
  polarization plus PySCF 2.13.1 SMD-CDS;
- continuum accuracy numerics: solvent-specific dielectric and SMD Coulomb
  radii, `lmax=15`, 1202 Lebedev points, `eta=0.1`, solver tolerance `1e-12`;
- data: ten experiment-blind, preregistered neutral absolute MNSol-v2012 rows,
  one per represented solvent, with eight confirmation and two development
  records;
- standard state: MNSol's 1-M ideal-gas to 1-M ideal-solution convention; no
  additional standard-state correction.

The two executions reproduce identical scientific aggregates and charge-
quality measures. Timing and full-file hashes differ as expected.

| equation | MAE | RMSE | max absolute error | mean signed error |
|---|---:|---:|---:|---:|
| ddPCM | 1.1430625 | 1.3515630 | 2.2322398 | +1.0333127 |
| scaled ddCOSMO | 1.0120976 | 1.2743164 | 2.1932768 | +0.8300035 |

All error values are in kcal/mol. Relative to the earlier tracked AIMNet2
execution at `7bc61643`, every displayed aggregate is reproduced within
`2.1e-13 kcal/mol`; the current ddPCM and ddCOSMO half-coupling identity errors
are at most `3.42e-15 eV`.

For descriptive context on the same frozen ten-record MNSol selection, the
tracked frozen-source MACE-POLAR arm reports ddPCM MAE `0.8635007 kcal/mol` and
ddCOSMO MAE `0.9154388 kcal/mol`. Its self-consistent direct-PCM arm reports
larger errors on this pilot. These ten points do not establish a source ranking
or solvent generalization.

## Exact execution

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python docs/implicit-solvation/benchmarks/run_mnsol_aimnet2_multisolvent_pilot.py \
  --source /path/to/user-supplied/MNSolDatabase_v2012.zip \
  --protocol docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json \
  --selection docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json \
  --checkpoint /path/to/aimnet2.pt \
  --private-output .omx/benchmarks/aimnet2-private.json \
  --public-output /tmp/aimnet2-public.json
```

## Claim boundary

This is an early chemical-accuracy diagnostic for a **complete reported
solvation ledger** (`fixed electrostatics + SMD-CDS`) on ten fixed MNSol
geometries. It is not the same scalar as the current
`harmonic-ddpcm-water` PES candidate, whose registered energy intentionally
has `G_np=0` and lower-order differentiable cavity numerics. Therefore this
bundle does not validate that candidate's force, PES, or energy accuracy. It
also does not certify the full MNSol population, per-solvent generalization,
conformer/geometry policy, original-SMD equivalence, fixed-R mutual
polarization, OPT, FREQ, TS, IRC, MD, or any public capability.
