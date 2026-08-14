# Operational analytic-harmonic pyridine rigid-panel shard

This bundle retains two clean-process executions of the preregistered
pyridine shard for the disabled operational analytic/original-source harmonic
scalar. It extends contiguous frozen rigid-panel coverage to `[0,15)` without
changing any tolerance after observing the result.

## Bound identity

- execution commit: `ce23da3ed4feca01a26f9463e9722ce010969375`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `pyridine`, indices `[14, 15)`
- measurement SHA-256 (both runs):
  `4003107b64c1a58917e526f18c390fa0e96936d6100e0dbd450439e1ebc4628a`
- primary JSON SHA-256:
  `9f8f8520959f7d6d11be885767cf84430b83b936d1e930ac40db14043df3c19f`
- replay JSON SHA-256:
  `662ff35556195055df46737f798dff5ad76fe9b876b77be088e05ed55b1afb82`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.9378840510672893e-13`;
- adjoint true residual: `7.073052528271538e-13`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum: exactly `0.0`;
- net-force norm: `5.003707553108401e-17 eV/Angstrom`;
- torque norm: `4.164030699072683e-17 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `2.919865947705001e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `8.380828986531585e-16`.

Across three preregistered proper rotations:

- maximum energy error: `4.036155587527901e-8 eV`;
- maximum force-covariance relative error: `6.934036634982884e-8`;
- maximum source-covariance relative error: `4.392705955992085e-9`;
- maximum field-covariance relative error: `1.2134327982950777e-8`;
- maximum primal residual: `9.853824852595837e-13`;
- maximum adjoint residual: `7.073065314385768e-13`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium pyridine geometry, one translation, one identical-atom
permutation, and three rotations. It is not a Cartesian force-FD, closed-loop,
distorted-geometry, physical-component, solvation-accuracy, PES/domain,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability tiers
remain closed: `E/F/H/V/M = false`.
