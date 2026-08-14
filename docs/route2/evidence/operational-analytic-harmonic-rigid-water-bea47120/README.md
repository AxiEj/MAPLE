# Operational analytic-harmonic water rigid-panel shard

This bundle retains two clean-process executions of the preregistered water
shard for the disabled operational analytic/original-source harmonic scalar.
Unlike the earlier water force-directional canary, this uses the same frozen
three-rotation, translation, and identical-atom permutation contract as the
remaining 20-molecule rigid panel.

## Bound identity

- execution commit: `bea4712067e9b0eb32d3b9476c6d839c64122c32`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `water`, indices `[0, 1)`
- measurement SHA-256 (both runs):
  `d8802924939ad40814eb81986aef4e76a76ba03fde88880597b5724b8f45480c`
- primary JSON SHA-256:
  `b66e9fe9333c9b75aab1ec43e2111f6f1cf2d4c87a8e39a41ea24f1639c03ed0`
- replay JSON SHA-256:
  `0864b3743323bfcfa891a8c9ae16bb533c7996543c5b0221a6759fe649e5cd41`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.545566114861273e-13`;
- adjoint true residual: `2.5836982065329233e-15`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum:
  `3.469446951953614e-18`;
- net-force norm: `2.7755575615628914e-17 eV/Angstrom`;
- torque norm: `1.640687091180359e-19 eV`;
- translation energy and force errors: exactly zero;
- identical-H permutation energy error: exactly zero;
- identical-H permutation force relative error:
  `5.1343598882883267e-17`.

Across three preregistered proper rotations:

- maximum energy error: `2.0313564164098352e-9 eV`;
- maximum force-covariance relative error: `5.7058772273700395e-8`;
- maximum source-covariance relative error: `1.9864466151380228e-9`;
- maximum field-covariance relative error: `4.337470603753039e-9`;
- maximum primal residual: `4.368923247736517e-13`;
- maximum adjoint residual: `2.6042185552828183e-15`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This replaces neither the earlier water force-directional evidence nor any
missing distorted-geometry, closed-loop, physical-component, accuracy,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability tiers
remain closed: `E/F/H/V/M = false`.
