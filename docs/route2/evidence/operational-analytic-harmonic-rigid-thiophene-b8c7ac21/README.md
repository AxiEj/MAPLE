# Operational analytic-harmonic thiophene rigid-panel shard

This bundle retains two clean-process executions of the preregistered
thiophene shard for the disabled operational analytic/original-source harmonic
scalar. It extends contiguous frozen rigid-panel coverage to `[0,18)` without
changing any tolerance after observing the result.

## Bound identity

- execution commit: `b8c7ac21aee8c0d8301bdc6ad0ddbe2f6cc38d84`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `thiophene`, indices `[17, 18)`
- measurement SHA-256 (both runs):
  `db104947eb42b0802723a101a3fd02dace2f57002b9075c6f748052d7e7763f7`
- primary JSON SHA-256:
  `675cf5e72104f391cab65bb1ea4c46d294971b0d821174cce97cd8d994e63149`
- replay JSON SHA-256:
  `b3d2f614b883843d6597be4d9ea8d84a2a19bed221fe47229371269979ca6f22`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `8.072873845760511e-13`;
- adjoint true residual: `1.029613925086936e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.734723475976807e-18 eV`;
- missing second-radial source block maximum:
  `3.469446951953614e-18`;
- net-force norm: `1.9478403400799703e-16 eV/Angstrom`;
- torque norm: `2.220452966656844e-16 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.2259064177750026e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `1.0278464083922914e-15`.

Across three preregistered proper rotations:

- maximum energy error: `2.561137080192566e-8 eV`;
- maximum force-covariance relative error: `4.0470329051080785e-8`;
- maximum source-covariance relative error: `7.683791133801178e-9`;
- maximum field-covariance relative error: `1.1699202622620264e-8`;
- maximum primal residual: `8.072902860752413e-13`;
- maximum adjoint residual: `1.0305199637545521e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium thiophene geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
