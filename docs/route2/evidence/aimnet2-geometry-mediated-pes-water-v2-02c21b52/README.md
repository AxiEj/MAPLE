# AIMNet2 geometry-mediated float64 water PES shard v2

This bundle retains two independent clean-process executions of water shard 0
under the strengthened v2 distorted-geometry contract. Unlike the earlier v1
bundle, v2 independently binds unordered sphere-pair topology and distance to
internal/external tangency, and the pure reducer recomputes metric reciprocity,
apply/adjoint identities, charge-direction finite differences, and
charge-gauge response from raw operands.

Each process evaluates the frozen reference, bond-compressed, and bond-stretched
geometries; three translation-free internal directions per geometry; and
central differences at `4e-4`, `2e-4`, and `1e-4 Angstrom`. All public
capabilities remain disabled.

## Bound identity

- execution commit: `02c21b52ad3bd2d7539640ed4863e4c1d022563f`
- execution tree: `834430397c1ac9c446aa4a76b9088490a931df6a`
- contract: `route2-aimnet2-geometry-mediated-pes-shard-contract-v2`
- molecule/shard: `water`, index `0` of the exact 17-shard H/C/N/O contract
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: reconstructed official `aimnet==0.2.0` Python graph, unchanged
  checkpoint weights, CPU float64
- continuum: fixed-dimensional smooth-harmonic point-monopole conductor
  reference; no finite-dielectric solvent identity
- bound committed Python sources: `114`
- scientific measurement SHA-256 (both runs):
  `3357780ea571d3920fe0637ecbf2d92ce2fd852646b7b0f2c0831ba789f13570`
- primary JSON SHA-256:
  `5f34132c0d8fb182590cca382b6dfc982983acbbc29249d8a7566f204dcfe877`
- replay JSON SHA-256:
  `2fcb25c9b618d21add18d54bd2e375d6cd638d2c3bd3a5cbf6a35b8c07db119b`

The raw records, derived summary, and scientific measurement SHA are exactly
identical across the two processes.

## Exact executions

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --molecule-index 0 --device cpu \
  --output /tmp/maple-route2-aimnet2-pes-water-v2.1nhl10/measurements.json

python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --molecule-index 0 --device cpu \
  --output /tmp/maple-route2-aimnet2-pes-water-v2.1nhl10/cold-replay.json
```

## Result

All v2 diagnostic gates pass over three geometries, nine direction records,
27 central differences, and 54 displaced scalar evaluations:

- maximum directional absolute error: `5.866424167422224e-05 eV/Angstrom`;
- minimum hard-neighbor cutoff margin: `3.428049281108132 Angstrom`;
- minimum point/source-shell event margin: `0.11773221132261891 Angstrom`;
- minimum sphere-tangency event margin: `0.5351277072462719 Angstrom`;
- maximum stationary surface condition number: `150.593715831781`;
- maximum metric reciprocity absolute error: `2.6645352591003757e-15 eV`;
- maximum apply/adjoint absolute error: `2.220446049250313e-15 eV`;
- maximum charge-direction finite-difference absolute error:
  `2.1084523016412504e-12 eV/e`;
- maximum charge-gauge coordinate VJP norm: exactly zero.

## Claim boundary

This supersedes no historical file: the v1 water bundle remains immutable
source-bound evidence for its older contract, while this bundle is the first
retained v2 shard. It is one of seventeen required H/C/N/O shards and does not
establish the complete panel, a global `C1` proof, finite-dielectric solvent
physics, chemical accuracy, fixed-geometry mutual polarization, HVP/Hessian,
or workflow safety.

All public gates remain closed: `E/F/H/V/M = false`, `OPT = false`,
`FREQ/TS/IRC = false`, and `MD = false`.
