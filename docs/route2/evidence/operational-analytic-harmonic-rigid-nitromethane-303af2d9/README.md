# Operational analytic-harmonic nitromethane rigid-panel shard

This bundle retains two clean-process executions of the preregistered
nitromethane shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,16)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `303af2d9ce16a886c8464a3108cb2cdef7beb088`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `nitromethane`, indices `[15, 16)`
- measurement SHA-256 (both runs):
  `f62fa0b0d22a0f710dc95ece7299bbdde437a715e6e10745e9d9d61bb63b468c`
- primary JSON SHA-256:
  `d41e37c9df7bebffe01a8f8eeb30e3a7e77b2cd6e347ec3b3d5bbbe670be0ebc`
- replay JSON SHA-256:
  `00746cd754d1ee918dc322304684b97a81e33c76972695c64ac8232e59a8c908`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.905594062507608e-13`;
- adjoint true residual: `1.1266763949449021e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.3877787807814457e-17 eV`;
- missing second-radial source block maximum:
  `8.673617379884035e-18`;
- net-force norm: `3.1401849173675503e-16 eV/Angstrom`;
- torque norm: `5.659227253495905e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.9033702791110927e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `3.06448456827688e-16`.

Across three preregistered proper rotations:

- maximum energy error: `1.0765688784886152e-8 eV`;
- maximum force-covariance relative error: `1.5400011762629733e-8`;
- maximum source-covariance relative error: `9.046633806516271e-9`;
- maximum field-covariance relative error: `1.5063971145521345e-8`;
- maximum primal residual: `5.984306387368631e-13`;
- maximum adjoint residual: `1.129549340608394e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium nitromethane geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
