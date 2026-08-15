# AIMNet2 geometry-mediated float64 water PES shard

This bundle retains two clean-process executions of the preregistered water
shard for the disabled AIMNet2 geometry-mediated smooth-harmonic conductor
reference.  The shard reuses MAPLE's frozen PES asset and contains the exact
reference, bond-compressed, and bond-stretched geometries, three frozen
translation-free internal directions per geometry, and three decreasing
central-difference steps per direction.

## Bound identity

- execution commit: `5244de8c274fca4d44938f327b04716bfca24faa`
- execution tree: `6055f79c1e06e2defefb277646a81b0b9f5fcce9`
- local checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `water`, index `0` of the exact 17-shard H/C/N/O contract
- measurement SHA-256 (both runs):
  `f0d8c0c3ec329ae4ff76ba2b2bb57974ee30721d447ec775b6aad8ed80b88ab7`
- primary JSON SHA-256:
  `1bb9a80758beaab32fdd04b6525f2cff85e62e548fdc44774ce8ec14e6750252`
- replay JSON SHA-256:
  `95b267961505aef87236df8ba0388f330a4b4c676fee0a1b58354be80c39c9e6`
- bound committed Python sources: `113`
- bound `aimnet==0.2.0` runtime files: `17`

The complete files differ only in execution metadata, output path, and
wall-clock time.  Their raw scientific measurement, derived summary, decision,
and `measurement_sha256` are identical.

## Result

All local diagnostic gates passed for the three geometries, nine directional
records, twenty-seven central differences, and fifty-four displaced scalar
energies:

- maximum directional absolute error:
  `5.866424167422224e-05 eV/Angstrom`;
- maximum applicable directional relative error:
  `1.8409686918823584e-04`;
- maximum stationary surface condition number: `150.593715831781`;
- maximum stationary absolute residual:
  `5.403808139617256e-15 eV/e`;
- maximum stationary relative residual: `1.00315871428364e-15`;
- minimum AIMNet2 hard-neighbor cutoff margin:
  `3.428049281108132 Angstrom`;
- minimum point/source-shell event margin:
  `0.11773221132261891 Angstrom`;
- maximum bilinear reciprocity absolute error:
  `2.6645352591003757e-15 eV`;
- maximum charge-direction finite-difference absolute error:
  `2.1084523016412504e-12 eV/e`;
- charge-gauge coordinate VJP norm: exactly zero.

The reference, compressed, and stretched total scalar energies are,
respectively, `-2081.586346969525`, `-2081.0676721434875`, and
`-2081.475982012667 eV`.  These values are ledger evidence, not accuracy or
relative-stability claims.

## Claim boundary

This is one of seventeen required H/C/N/O shards.  It does not establish the
complete chemistry panel, a finite-dielectric solvent model, chemical
accuracy, global `C1` regularity, a closed-loop work gate, explicit
cutoff/tangency trial-step safety, fixed-geometry mutual polarization, or any
Hessian/HVP/NVE result.  All public capability tiers and workflows remain
closed: `E/F/H/V/M = false`, `OPT = false`, `FREQ/TS/IRC = false`, and
`MD = false`.
