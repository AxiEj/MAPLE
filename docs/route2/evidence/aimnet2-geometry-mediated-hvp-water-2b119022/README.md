# AIMNet2 geometry-mediated complete-HVP water canary

This bundle retains two independent clean-process executions of the frozen
water HVP contract for the disabled reconstructed-float64 AIMNet2 plus
smooth-harmonic point-monopole conductor-reference scalar. The pure release
reducer reconstructs every numerical conclusion from raw sources, gradients,
VJPs, HVP components, finite-difference endpoints, and topology records; it
does not trust a stored pass boolean.

The implemented weak-scalar action is

```text
H_E h
+ (G_RR h + G_Rc J_c h)
+ J_c^T (G_cR h + G_cc J_c h)
+ D_R[J_c^T v][h]
```

where the final model term holds the center reaction-potential cotangent `v`
fixed. There is no outer charge fixed point, latent polarization variable,
PCM-to-graph feedback, nonpolar/CDS term, or public calculator.

## Bound identity

- execution commit: `2b11902290ed9ccf884410cd30773fa0bb9abcf7`
- execution tree: `1dd9124f395e4b2fdd9484ea5a416a0cbb47f564`
- contract: `route2-aimnet2-geometry-mediated-hvp-water-contract-v1`
- scalar profile:
  `route2-profile-diagnostic-aimnet2-geometry-mediated-smoothharmonicgalerkin-cpcm-electrostatic-v1`
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- runtime: source-bound `aimnet==0.2.0` Python reconstruction, unchanged
  checkpoint weights, CPU float64; upstream DFT-D3 is reused with its
  differentiable `hessian=True` path in a separately parity-gated graph
- continuum: fixed-dimensional smooth-harmonic point-monopole conductor
  reference; no finite-dielectric solvent identity
- bound committed Python sources: `115`
- scientific measurement SHA-256 (both runs):
  `01009ee3f5f829da5906cf051f4bc6b110d8c28300ff31467a6dc663943b88a6`

The complete JSON files differ only in execution metadata such as output path
and wall-clock time. Their protocol, identities, admission boundary, center
record, all five HVP direction records, all six finite-difference endpoints,
derived summary, and scientific measurement SHA are exactly identical.

## Exact executions

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_hvp.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --device cpu \
  --output /tmp/maple-route2-hvp-2b119022-a.lOAZM9/measurements.json

python tools/route2_release/run_aimnet2_geometry_mediated_hvp.py \
  --checkpoint /home/axie/MAPLE/MAPLE-fork/maple/function/calculator/model/aimnet2.pt \
  --device cpu \
  --output /tmp/maple-route2-hvp-2b119022-b.desHFn/cold-replay.json
```

## Result

Every preregistered diagnostic gate passed in both processes.

### Same-graph and first-order prerequisites

- ordinary/decomposed energy parity error: `4.306457412894815e-10 eV`;
- charge parity and fixed-cotangent charge-VJP parity errors: exactly zero;
- intrinsic-gradient parity error: `1.4042003115832813e-9 eV/Angstrom`;
- maximum charge-tangent residual: `2.7755575615628914e-17 e/Angstrom`;
- center stationary residual: `2.8867763194343613e-15 eV/e`;
- center surface condition number: `107.08433946218663`;
- center reciprocity absolute error: `1.9984014443252818e-15 eV`;
- center charge-direction FD error: `2.1084523016412504e-12 eV/e`;
- center charge-gauge VJP norm: exactly zero.

### Complete HVP ledger

For the primary internal direction, the component norms are:

| component | norm |
| --- | ---: |
| intrinsic `H_E h` | `34.13111580104285 eV/Angstrom^2` |
| continuum position block | `4.179189867015273 eV/Angstrom^2` |
| source-response pullback | `0.28487009736981256 eV/Angstrom^2` |
| fixed-cotangent contracted charge Hessian | `2.1310336906441183 eV/Angstrom^2` |
| complete HVP | `32.35690698108939 eV/Angstrom^2` |

The stored component ledger closes exactly. The source cotangent differs from
the audited reaction potential by at most `2.220446049250313e-16 eV/e`.

### Central-difference closure

| step (Angstrom) | charge JVP error (e/Angstrom) | contracted charge-Hessian error (eV/Angstrom^2) | complete HVP error (eV/Angstrom^2) |
| ---: | ---: | ---: | ---: |
| `4e-4` | `3.524794331122906e-7` | `8.140312848327802e-6` | `5.355905410854544e-4` |
| `2e-4` | `8.81192833351706e-8` | `2.035084244973546e-6` | `1.4003029983263116e-4` |
| `1e-4` | `2.203106592738926e-8` | `5.08769444399133e-7` | `3.391279860294818e-5` |

Successive error ratios are approximately `0.25` for the charge JVP and
contracted charge Hessian and `0.261`, then `0.242`, for the complete HVP,
consistent with the expected second-order central-difference regime.

The two frozen internal directions give

```text
u^T H v = 14.294605083745521 eV/Angstrom^2
v^T H u = 14.294605083745525 eV/Angstrom^2
absolute difference = 3.552713678800501e-15 eV/Angstrom^2
```

All three normalized rigid translations have exactly zero charge JVP and
exactly zero complete HVP. Every stencil endpoint retained the same AIMNet2
neighbor and continuum topology, and every center-to-endpoint segment passed
the conservative neighbor, point/source-shell, and sphere-tangency guards.

## Claim boundary

This is positive implementation evidence for one water geometry, two internal
directions, three translational directions, and one certified graph/cavity
stratum of the explicit `R -> q_AIMNet2(R)` conductor-reference scalar. It is
not a global `C2` proof, a full H/C/N/O Hessian panel, a stationary-point
frequency calculation, finite-dielectric or nonpolar validation, a solvent
accuracy result, fixed-geometry mutual polarization, or proof that all
cutoff/cavity neighborhoods are smooth.

The broader current-profile Tier-F/domain panel remains negative because the
bond-stretched methanol point/source-shell margin is below its guard. This
water HVP canary cannot override that failure. All public capability and
workflow gates remain closed: `E/F/H/V/M = false`, public ASE HVP is false,
`OPT = false`, `FREQ/TS/IRC = false`, and `MD = false`.
