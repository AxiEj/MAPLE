# Operational analytic-harmonic ethylamine rigid-panel shard

This bundle retains two clean-process executions of the preregistered
ethylamine shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,14)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `4b40b81a18c74902f35a96f417d43a45b645d8d1`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `ethylamine`, indices `[13, 14)`
- measurement SHA-256 (both runs):
  `d7e08bb0797e20192e095c22f346efb9d25dcd8cbd3cf87b3717dd0eaafb0cb8`
- primary JSON SHA-256:
  `76de00b473b1bfde479e991e103e624e34e03f6849a1926c6d4fa9aff39a7fbf`
- replay JSON SHA-256:
  `95f426d318d8eb03cccc8a62edef359398ce05482961e90294b0a83a5d837576`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.790884527310545e-13`;
- adjoint true residual: `1.8370342085260074e-12`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error: exactly `0.0 eV`;
- missing second-radial source block maximum:
  `1.3010426069826053e-18`;
- net-force norm: `1.0031442548020353e-16 eV/Angstrom`;
- torque norm: `1.013358119639238e-9 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.758998379830466e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `6.332652078404679e-16`.

Across three preregistered proper rotations:

- maximum energy error: `3.653894964372739e-9 eV`;
- maximum force-covariance relative error: `4.700291638874309e-8`;
- maximum source-covariance relative error: `2.742671304428915e-9`;
- maximum field-covariance relative error: `5.8245504543087704e-9`;
- maximum primal residual: `7.806175177769732e-13`;
- maximum adjoint residual: `1.83704191645252e-12`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium ethylamine geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
