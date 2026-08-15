# AIMNet2 geometry-mediated water loop and event audit

This bundle retains two independent clean-process executions of the frozen
water loop for the disabled, reconstructed-float64 AIMNet2 plus smooth-harmonic
conductor-reference scalar. Each process evaluates all 17 forward and all 17
reverse points independently. There is no warm state, outer charge root,
fixed-geometry electronic response, graph feedback, nonpolar term, or public
ASE calculator.

The contract uses two frozen translation-free orthonormal internal directions,
amplitudes `(0.02, 0.02) Angstrom`, and four Simpson-compatible subdivisions per
rectangular edge. The pure release reducer reconstructs the force-work
integrals, same-coordinate replays, stationary-solve diagnostics, metric
reciprocity, apply/adjoint identities, charge-direction finite differences,
charge-gauge VJP, and every straight-segment topology/event certificate from
raw operands. A stored pass boolean cannot substitute for those measurements.

## Bound identity

- execution commit: `e0a347f5f4e5a93f9b238333964273a6f19cbf02`
- execution tree: `db21006bd6996a64ee5bb945eafbe8a57e45ffc7`
- contract: `route2-aimnet2-geometry-mediated-water-loop-contract-v1`
- scalar profile:
  `route2-profile-diagnostic-aimnet2-geometry-mediated-smoothharmonicgalerkin-cpcm-electrostatic-v1`
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: reconstructed official `aimnet==0.2.0` Python graph, unchanged
  checkpoint weights, CPU float64
- continuum: fixed-dimensional smooth-harmonic point-monopole conductor
  reference; no finite-dielectric solvent identity
- bound committed Python sources: `114`
- scientific measurement SHA-256 (both runs):
  `5e389819e022dda7a2466d9e472486f45cc9bd6715c9735efefa0bd0b0dad0e3`
- primary JSON SHA-256:
  `ecb496767dde7562b35e8b175ff311dc17ed3b59a6ca54c9ad4dc5cab3bc1bb5`
- replay JSON SHA-256:
  `d7e8ac83bca638f4dd09f50132c000a3d97f58007c5cb952e5ffe0bae8da15cb`

The complete JSON files differ only in execution metadata such as output path
and wall-clock time. Their raw forward records, raw reverse records, derived
summary, and scientific measurement SHA are exactly identical.

## Exact executions

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --device cpu \
  --output /tmp/maple-route2-aimnet2-water-loop-recip.qyzsSl/measurements.json

python tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --device cpu \
  --output /tmp/maple-route2-aimnet2-water-loop-recip.qyzsSl/cold-replay.json
```

## Result

Every diagnostic gate passed in both processes:

- forward Simpson work: `0.00014855129592190874 eV`;
- reverse Simpson work: `-0.0001485512959219093 eV`;
- forward-plus-reverse work: `-5.692061405548898e-19 eV`;
- sum of absolute edge work per traversal: `0.21023531545951998 eV`;
- preregistered work threshold per traversal: `0.00021023531545952 eV`;
- maximum same-coordinate energy, source, and force replay errors: exactly
  zero;
- maximum stationary absolute residual: `5.497741648040659e-15 eV/e`;
- maximum stationary relative residual: `1.0711567553273667e-15`;
- maximum stationary surface condition number: `113.03902157292583`;
- maximum metric reciprocity absolute error: `3.1086244689504383e-15 eV`;
- maximum metric reciprocity relative error: `3.1244452797106744e-15`;
- maximum apply/adjoint absolute error: `3.1086244689504383e-15 eV`;
- maximum apply/adjoint relative error: `3.0639139441879957e-15`;
- maximum charge-direction finite-difference absolute error:
  `4.294398170401337e-12 eV/e`;
- maximum charge-direction finite-difference relative error:
  `2.685225389303748e-11`;
- maximum source-gradient half-coupling error:
  `9.650801630464006e-16 eV/source-unit`;
- maximum charge-gauge coordinate VJP norm: exactly zero;
- minimum endpoint hard-neighbor margin: `3.4355787226946504 Angstrom`;
- minimum endpoint point/source-shell margin: `0.20407200707293016 Angstrom`;
- minimum endpoint sphere-tangency margin: `0.6216205409966026 Angstrom`;
- minimum certified straight-segment hard-neighbor lower bound:
  `3.423985104581346 Angstrom`;
- minimum certified straight-segment point/source-shell lower bound:
  `0.1924783889596255 Angstrom`;
- minimum certified straight-segment sphere-tangency lower bound:
  `0.610026922883298 Angstrom`.

The nonzero Simpson loop work lies inside the frozen relative work gate; it is
not reported as exact zero. The independent reverse traversal supplies a
separate near-machine-precision antisymmetry check.

## Claim boundary

This is positive, source-bound, water-only local evidence for one explicit
`R -> q_AIMNet2(R)` composite scalar on one certified model/cavity stratum. It
does not prove global `C1` regularity, the complete v2 H/C/N/O panel, a physical
finite-dielectric solvent, chemical accuracy, fixed-geometry mutual
polarization, Hessian/HVP correctness, optimizer safety outside the tested
segments, or long-time dynamics.

All public capability and workflow gates remain closed:
`E/F/H/V/M = false`, `OPT = false`, `FREQ/TS/IRC = false`, and `MD = false`.
The earlier temporary run made before raw-reciprocity recomputation was added
is not retained and is not evidence.
