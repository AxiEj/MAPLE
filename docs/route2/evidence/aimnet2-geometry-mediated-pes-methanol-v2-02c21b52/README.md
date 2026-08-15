# AIMNet2 geometry-mediated float64 methanol PES shard v2: negative event gate

This bundle retains two independent clean-process executions of methanol shard
1 under the strengthened v2 distorted-geometry contract. It is intentionally
preserved as negative admission evidence. No radius, geometry, finite-
difference step, guard, or threshold was changed after observing the result.

Each process evaluates the frozen reference, bond-compressed, and bond-stretched
geometries; three translation-free internal directions per geometry; and
central differences at `4e-4`, `2e-4`, and `1e-4 Angstrom`. The reducer derives
all numerical and topology decisions from raw records.

## Bound identity

- execution commit: `02c21b52ad3bd2d7539640ed4863e4c1d022563f`
- execution tree: `834430397c1ac9c446aa4a76b9088490a931df6a`
- contract: `route2-aimnet2-geometry-mediated-pes-shard-contract-v2`
- molecule/shard: `methanol`, index `1` of the exact 17-shard H/C/N/O contract
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: reconstructed official `aimnet==0.2.0` Python graph, unchanged
  checkpoint weights, CPU float64
- continuum: fixed-dimensional smooth-harmonic point-monopole conductor
  reference; no finite-dielectric solvent identity
- bound committed Python sources: `114`
- scientific measurement SHA-256 (both runs):
  `9f0b43679a1cbdadcbae2daccf10b62bfab7cad5987db71c853a5e4f75933f20`
- primary JSON SHA-256:
  `9aa6f7c96cae674d2a612e3c536fffde6edfb2209279493440ba6e7b3c4a4912`
- replay JSON SHA-256:
  `2f27779b37de05cd899d24e5ffb90b912e09815027809ff8c04193aa1e6bf78f`

The raw records, derived negative summary, and scientific measurement SHA are
exactly identical across the two processes.

## Exact executions

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --molecule-index 1 --device cpu \
  --output /tmp/maple-route2-aimnet2-pes-methanol-v2.zbA5UF/measurements.json

python tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --molecule-index 1 --device cpu \
  --output /tmp/maple-route2-aimnet2-pes-methanol-v2.zbA5UF/cold-replay.json
```

## Result and diagnosis

The shard correctly reports `diagnostic-gates-failed-not-admitted`:

- deterministic replays, stationary solves, metric reciprocity,
  apply/adjoint checks, charge-direction finite differences, charge-gauge VJP,
  hard-neighbor guards, sphere-tangency guards, and exact stencil topology all
  pass;
- every one of the 27 derivative samples passes its numerical error test, and
  every directional convergence summary passes;
- the frozen bond-stretched geometry has minimum point/source-shell event
  margin `0.015511399564898554 Angstrom`, below the preregistered
  `0.02 Angstrom` guard;
- consequently the three bond-stretched directional records fail closed on
  their continuum event guard even though their numerical derivative gates
  pass;
- minimum sphere-tangency margin is `0.04978002374103285 Angstrom`, so the
  independent sphere event gate is not the blocker;
- maximum directional absolute error is
  `5.690032751104468e-05 eV/Angstrom`;
- maximum stationary surface condition number is `604.244462442586`;
- maximum metric reciprocity absolute error is
  `1.1102230246251565e-15 eV`;
- maximum charge-direction finite-difference absolute error is
  `1.887073830530994e-12 eV/e`;
- maximum charge-gauge coordinate VJP norm is exactly zero.

This is an event-distance admission failure, not evidence of a mismatched
analytic force. The fixed tiny finite-difference stencils remain in one sampled
stratum, but the wider safety margin required for downstream geometry tasks is
not available at that frozen configuration.

## Claim boundary

This negative shard blocks the current v2 full-panel pass for this continuum
profile. It does not justify lowering the guard, tuning radii, deleting the
frozen distorted geometry, or claiming that the underlying local scalar or
numerical derivative is invalid. A scientifically different smooth source-map
or cavity definition would require a new versioned profile and fresh evidence.

All public gates remain closed: `E/F/H/V/M = false`, `OPT = false`,
`FREQ/TS/IRC = false`, and `MD = false`.
