# Frozen AIMNet2 water-ddPCM daily-task diagnostics (`76d4d097`)

This evidence set exercises the exact candidate scalar

`E_AIMNet2(R) + G_ddPCM,water(R, q_NQE(R))`

for one water molecule in three diagnostic workflows: a bidirectional closed
force-work loop, complete weak-scalar Hessian-vector products (HVPs), and a
guarded stationary solve followed by a dense Cartesian Hessian/frequency
analysis. AIMNet2 is evaluated once per geometry. It receives no continuum
field and performs no electronic SCF or outer charge fixed-point iteration.
The continuum arm is finite-dielectric water (`epsilon = 78.355`) and
`G_np = 0` exactly.

## Source and checkpoint binding

- Execution commit: `76d4d0974abf7a631d783f65723f87af1af88498`
- Execution tree: `58218130c36191cd2a5002ca41f96bfaa9bf1322`
- AIMNet2 checkpoint SHA256: `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- Runtime: source-bound reconstructed float64 AIMNet2, CPU
- Both processes reported a clean worktree and the same source/checkpoint identities.

## Results

| Diagnostic | Measurement SHA256 | Selected result | Status |
| --- | --- | --- | --- |
| Bidirectional water loop | `4c70022f570e2736e4b2f16312009971058461cf2a78776e3ba45e01738cd26f` | forward + reverse work = `-1.100465205072787e-17 eV`; minimum point/source-shell margin = `0.20407200707293016 A` | all diagnostic gates passed, not admitted |
| Complete water HVP | `60159b0b31ec0ae1ae308dd03175121d9c467078882965cbf9a72670d4d39dfb` | bilinear symmetry error = `7.105427357601002e-15 eV/A^2`; smallest-step total-HVP FD error = `3.294758659925606e-05 eV/A^2` | all diagnostic gates passed, not admitted |
| Stationary dense Hessian/frequency | `2cb5aceee9d22c84bfe458bf9e9967d5c5cb61c9536f90a450246d97730c258c` | stationary Cartesian max gradient = `6.744916582722416e-15 eV/A`; Hessian symmetry error = `1.126172990825879e-14 eV/A^2`; smallest-step FD Frobenius error = `3.524831549078756e-05 eV/A^2` | all diagnostic gates passed, not admitted |

The HVP central-difference errors show approximately quadratic refinement
(ratios near `0.25`). The dense-Hessian FD Frobenius errors are
`[2.2565065e-3, 5.6400865e-4, 1.4099480e-4, 3.5248315e-5] eV/A^2`.
All three translational HVPs are exactly zero in the local HVP canary. At the
stationary geometry, all translational and rotational Hessian residuals are
below `1.5e-14 eV/A^2`.

The stationary search uses the same frozen internal-coordinate bounds and
MINPACK hybrid root thresholds as the conductor-reference diagnostic. For the
finite-dielectric arm only, a componentwise open-bound `tanh`
parameterization keeps every root trial strictly inside those bounds. It
converged in 18 function evaluations to
`(r_OH1, r_OH2, angle_HOH) = (1.0207730339378311 A,
1.0207730339378314 A, 2.00873796263625 rad)`.

The reported frequencies (`1663.93`, `2672.83`, and `2900.47 cm^-1`) are
implementation diagnostics for this incomplete electrostatic scalar. They are
not experimental predictions or chemical-accuracy evidence.

## Cold replay

Each workflow was rerun in a second fresh Python process. Primary and replay
artifacts have identical `measurement_sha256` values and identical measured
records/summaries. Whole-file hashes differ only because exact output paths and
wall-clock runtime are intentionally outside the measurement hash.

## Reproduction

Run from a clean checkout of the execution commit:

```bash
export MAPLE_ROUTE2_AIMNET2_CHECKPOINT=/absolute/path/to/aimnet2.pt

python tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py \
  --checkpoint "$MAPLE_ROUTE2_AIMNET2_CHECKPOINT" --device cpu \
  --continuum harmonic-ddpcm-water --output /tmp/water-loop.json

python tools/route2_release/run_aimnet2_geometry_mediated_hvp.py \
  --checkpoint "$MAPLE_ROUTE2_AIMNET2_CHECKPOINT" --device cpu \
  --continuum harmonic-ddpcm-water --output /tmp/hvp.json

python tools/route2_release/run_aimnet2_geometry_mediated_frequency.py \
  --checkpoint "$MAPLE_ROUTE2_AIMNET2_CHECKPOINT" --device cpu \
  --continuum harmonic-ddpcm-water --output /tmp/frequency.json
```

## Claim boundary

This is water-only, fixed-graph/fixed-cavity-stratum, diagnostic evidence. It
does not establish global C1/C2 regularity, a complete nonpolar/solvation
free energy, fixed-geometry mutual polarization, broad-molecule force
admission, chemical accuracy, or public E/F/H/V/M, OPT, FREQ/TS/IRC, or MD
support. The separate 17-molecule PES panel still has five topology/event
failures, so all public capability flags remain false.
