# Operational analytic-harmonic acetaldehyde rigid-panel shard

This bundle retains two clean-process executions of the preregistered
acetaldehyde shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,12)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `857af1b80be12eb5dfd3ea90ee0036dad682e285`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `acetaldehyde`, indices `[11, 12)`
- measurement SHA-256 (both runs):
  `76ae1cb0abbec4aa64ba9f492aed8c4a2bf4fe65f3af051aac78afd9897f45a6`
- primary JSON SHA-256:
  `7753fc64f27684f49f6cbaafc4427988aa65e56e22c601ce7c9cf88c21bc225d`
- replay JSON SHA-256:
  `50dd905cfedce199e4ebaf844d37383eea8fa71a5909c27b317facc44e2a6c0e`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.6717873413406096e-13`;
- adjoint true residual: `3.9450254736576947e-13`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.3877787807814457e-17 eV`;
- missing second-radial source block maximum:
  `2.7755575615628914e-17`;
- net-force norm: `2.155529167242746e-16 eV/Angstrom`;
- torque norm: `4.2566807023636265e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `2.2079166899244158e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `5.292398143191071e-16`.

Across three preregistered proper rotations:

- maximum energy error: `5.841684469487518e-9 eV`;
- maximum force-covariance relative error: `2.376488400904626e-8`;
- maximum source-covariance relative error: `3.844067558132801e-9`;
- maximum field-covariance relative error: `6.277857652878454e-9`;
- maximum primal residual: `8.447192019404236e-13`;
- maximum adjoint residual: `3.9455838807674034e-13`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium acetaldehyde geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
