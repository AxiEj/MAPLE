# Operational analytic-harmonic acetamide rigid-panel shard

This bundle retains two clean-process executions of the preregistered
acetamide shard for the disabled operational analytic/original-source harmonic
scalar. It extends contiguous frozen rigid-panel coverage to `[0,13)` without
changing any tolerance after observing the result.

## Bound identity

- execution commit: `ab1c779946a12240eda1b54a81ef88aa05a9827d`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `acetamide`, indices `[12, 13)`
- measurement SHA-256 (both runs):
  `3327e6e070d3ab83dd0e68c8461d6f09088d2f2fa7c423906dd03a4cd8941de7`
- primary JSON SHA-256:
  `b0a25bc0e88191873955c91735b2719a0443f7a6eb5a4fbc630637bade0abedb`
- replay JSON SHA-256:
  `78defc71db3f3e29ba905a86738ad5113091e5f83449b991c8d5d1b8ab80f5e6`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.917926943572601e-13`;
- adjoint true residual: `3.889382009680901e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.3877787807814457e-17 eV`;
- missing second-radial source block maximum:
  `2.0816681711721685e-17`;
- net-force norm: `3.5571561302006606e-16 eV/Angstrom`;
- torque norm: `2.546785668698071e-9 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.8578445752105446e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `8.663386032678559e-16`.

Across three preregistered proper rotations:

- maximum energy error: `3.928107616957277e-9 eV`;
- maximum force-covariance relative error: `4.6556222143172366e-8`;
- maximum source-covariance relative error: `4.5519884567843026e-9`;
- maximum field-covariance relative error: `4.265275526136837e-9`;
- maximum primal residual: `9.98767393738792e-13`;
- maximum adjoint residual: `3.89157848698684e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium acetamide geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
