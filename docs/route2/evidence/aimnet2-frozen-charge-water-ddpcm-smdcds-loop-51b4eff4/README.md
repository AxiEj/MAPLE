# Frozen AIMNet2 water ddPCM + SMD-CDS total-force loop (`51b4eff4`)

This bundle retains two independent clean-process executions of the disabled
aqueous candidate scalar

`E_AIMNet2(R) + G_harmonic-ddPCM,water(R, q_NQE(R)) + G_PySCF-SMD-CDS,water(R)`.

AIMNet2 is evaluated exactly once at each geometry. It receives no continuum
field and performs no electronic SCF or outer charge fixed-point iteration.
The electrostatic derivative includes the complete geometry-dependent AIMNet2
charge chain rule. The nonpolar contribution is the energy and analytic
coordinate gradient returned by PySCF 2.13.1
`pyscf.solvent.smd.get_cds_legacy` for the same geometry.

## Bound identity

- execution commit: `51b4eff47ca95fc114f420466b4a7f0e250e9c50`
- execution tree: `1244222d4275b4c47ae4ee41dafedb4ee472c9d7`
- contract:
  `route2-aimnet2-geometry-mediated-water-ddpcm-smdcds-loop-contract-v1`
- scalar:
  `route2-candidate-aimnet2-frozen-charge-water-smoothharmonicgalerkin-ddpcm-pyscf-smdcds-v1`
- profile:
  `route2-profile-candidate-aimnet2-frozen-charge-water-smoothharmonicgalerkin-ddpcm-pyscf-smdcds-v1`
- scalar fingerprint:
  `4783776d873fe70f687db21f749b0a5aac7c02a6c8413da094483db965c4795c`
- nonpolar configuration:
  `0ab1a3b4bd2f73826d7726acc6bddb7ed1240a491707845868c8164c0bd4b72e`
- AIMNet2 checkpoint SHA-256:
  `85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d`
- scientific measurement SHA-256, identical in both processes:
  `2bba943688afccc89a16b012eb6763437a835af6ccd060266d9382bb73397aaf`
- bound committed Python sources: `136`

Both processes reported a clean worktree and the same source, checkpoint,
scalar, continuum, and nonpolar identities.

## Result

Every registered water-loop diagnostic passed in both processes:

- forward Simpson work: `0.00014572688007338797 eV`;
- reverse Simpson work: `-0.00014572688007338883 eV`;
- forward-plus-reverse work: `-8.673617379884035e-19 eV`;
- sum of absolute edge work: `0.2019811732893414 eV`;
- frozen relative work threshold: `0.0002019811732893414 eV`;
- maximum same-coordinate energy, source, and total-force replay errors:
  exactly zero;
- maximum harmonic-ddPCM stationarity absolute residual:
  `2.0887116361891608e-14`;
- maximum stationarity/surface condition number: `113.03902157292583`;
- maximum metric-reciprocity absolute error:
  `5.551115123125783e-15 eV`;
- maximum apply/adjoint absolute error: `1.7763568394002505e-15 eV`;
- maximum charge-direction finite-difference absolute error:
  `2.6767477123712524e-12 eV/e`;
- maximum source-gradient half-coupling error: exactly zero;
- maximum charge-gauge coordinate VJP norm: exactly zero;
- minimum endpoint hard-neighbor margin: `3.4355787226946504 Angstrom`;
- minimum endpoint point/source-shell margin: `0.20407200707293016 Angstrom`;
- minimum endpoint sphere-tangency margin: `0.6216205409966026 Angstrom`.

The total gradient ledger was recomputed at every point as the sum of the
electrostatic total gradient and the PySCF SMD-CDS gradient before any
force-work gate was evaluated. The independent reverse process reproduced the
complete forward and reverse records and derived summary exactly. Whole-file
hashes differ only because output paths and wall-clock runtimes are outside the
scientific measurement hash.

## Reproduction

Run twice from a clean checkout of the execution commit:

```bash
python tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py \
  --checkpoint /absolute/path/to/aimnet2.pt --device cpu \
  --continuum harmonic-ddpcm-water \
  --nonpolar pyscf-smd-cds-water \
  --output /tmp/aimnet2-water-ddpcm-smdcds-loop.json
```

## Claim boundary

This is positive, source-bound, water-only local evidence for one complete
vacuum + electrostatic + SMD-CDS scalar on one certified topology stratum. It
does not establish strict original-SMD electrostatic equivalence, the complete
518-record MNSol accuracy matrix, global `C1`/`C2` regularity, a same-scalar
SMD-CDS Hessian, or broad-molecule force admission. The separate 17-molecule
PES panel still has five topology/event failures.

All public capabilities and workflows remain closed:
`E/F/H/V/M = false`, `OPT = false`, `FREQ/TS/IRC = false`, and `MD = false`.
