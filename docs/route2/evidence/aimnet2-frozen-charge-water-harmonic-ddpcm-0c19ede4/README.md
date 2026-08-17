# Water-bound frozen-charge AIMNet2 / harmonic-ddPCM canary

This bundle retains two independent clean-process executions of the exact
registered candidate
`route2-profile-candidate-aimnet2-frozen-charge-water-smoothharmonicgalerkin-ddpcm-electrostatic-v1`
at source commit `0c19ede4b4354e2c75deeea98ea27ba19c374846` and tree
`111417ed58787c775bce00b3c1405a4b48db58eb`.

## Bound protocol

- Local AIMNet2 wB97M-D3 checkpoint SHA256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`.
- Source-bound reconstructed Python runtime, float64 on CPU; checkpoint weights
  are unchanged.
- Model profile `aimnet2-polarizable-v1`; here the name identifies the NQE
  checkpoint family. Charges are evaluated once per geometry, no continuum
  field is supplied to AIMNet2, and no electronic SCF is run.
- Water dielectric `78.355`; Coulomb radii from
  `smd-ddpcm-l15-n1202-multisolv-v1` (`O=1.52 A`, `H=1.20 A`).
- Smooth harmonic finite-dielectric ddPCM with transition width `0.18 A^2`,
  surface/exposure orders `1/2`, and order-32 exposure and Green/double-layer
  radial quadratures.
- Electrostatic scalar only. CDS/nonpolar and standard-state components are
  absent.

Exact command shape:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/route2_release/run_aimnet2_geometry_mediated_canary.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --aimnet-runtime reconstructed-python-float64 \
  --continuum harmonic-ddpcm-water --device cpu \
  --output /tmp/aimnet2-frozen-charge-water-ddpcm-runN.json
```

## Replayed result

Both processes reproduce scientific measurement SHA256
`29ad72cd84a137b6bc7b9079ce6979000e7dab58983e2c41d6d2e3ce82fa08e5`.
The continuum contribution is `-0.49914801549225324 eV`. The adaptive
analytic/numerical directional error is `1.8257224045914455e-6 eV/A`; the
terminal full-Cartesian maximum/RMS errors are
`2.826476747208595e-5 / 1.5826934760464932e-5 eV/A`, with observed order
`1.9945636040785466`. Three rigid rotations have zero recorded energy drift
and maximum force-covariance relative error `5.5951040041648464e-11`.

All recorded deterministic replay, metric reciprocity, charge-gauge,
primal/adjoint stationarity, condition, half-coupling, adaptive directional,
full-Cartesian, hard-neighbor, cavity-event, rigid-rotation, net-force, and
torque gates pass. The largest condition number is `109.61213248462516` and
the largest recorded relative residual is `1.0735508334481661e-15`.

## Claim boundary

This is positive local one-water evidence for the exact solvent-bound frozen
charge identity. It does **not** establish chemical accuracy, a broad smooth
PES/domain, CDS/nonpolar completeness, global differentiability, production
optimization/frequency/TS/IRC/MD readiness, or fixed-geometry electronic mutual
polarization. `E/F/H/V/M` and every public task capability remain false.

The equation follows the finite-dielectric ddPCM operator documented by ddX:
<https://ddsolvation.github.io/ddX/md_docs_theory.html>. AIMNet2 model scope is
reported in <https://doi.org/10.1039/D4SC08572H>; neither reference by itself
admits this MAPLE composition.
