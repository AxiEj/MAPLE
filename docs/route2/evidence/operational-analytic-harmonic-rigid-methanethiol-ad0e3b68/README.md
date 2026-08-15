# Operational analytic-harmonic methanethiol rigid-panel shard

This bundle retains two clean-process executions of the preregistered
methanethiol shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,19)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `ad0e3b687294cea4d0a5539664d219a3c228566a`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `methanethiol`, indices `[18, 19)`
- measurement SHA-256 (both runs):
  `8ca3a93c8b2c3283bb3ba85b4eb035c20e4671ee728d5e8e591f6da1e3663026`
- primary JSON SHA-256:
  `b98b00fb7dd91977e26b1842badc052e988d64abcba0a3a9f0d7051a782d6568`
- replay JSON SHA-256:
  `4f973bb80d19f501faec094df56ea328eacb02bce61cf57208e050b45559e0bf`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `8.555208516750313e-13`;
- adjoint true residual: `3.460065862286548e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum:
  `8.673617379884035e-19`;
- net-force norm: `9.412373634839028e-17 eV/Angstrom`;
- torque norm: `1.1401168793778726e-9 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.3372954724572458e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `1.0111562748009299e-15`.

Across three preregistered proper rotations:

- maximum energy error: `6.270056474022567e-9 eV`;
- maximum force-covariance relative error: `2.3666944175900336e-8`;
- maximum source-covariance relative error: `2.8887724513946564e-9`;
- maximum field-covariance relative error: `1.0805136148919826e-8`;
- maximum primal residual: `8.555274399411498e-13`;
- maximum adjoint residual: `3.4603541777744226e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium methanethiol geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
