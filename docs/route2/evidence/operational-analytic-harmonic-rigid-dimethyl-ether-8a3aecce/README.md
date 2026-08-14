# Operational analytic-harmonic dimethyl-ether rigid-panel shard

This bundle retains two clean-process executions of the preregistered
dimethyl-ether shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,9)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `8a3aecce257695b27c390d36af057ac52c36427a`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `dimethyl-ether`, indices `[8, 9)`
- measurement SHA-256 (both runs):
  `23f177cb0a70fbd513b68951231205e241d2a92eecb2b0457cee7a1804caad5d`
- primary JSON SHA-256:
  `a5b0e23b3786a927b1c9128b6870e1d0f3bb4d5b2579461f5d6974023a9b9f47`
- replay JSON SHA-256:
  `53a98de9c148242b3af235c4c4dcb52af6a4a5d682354a83e465b2daa688cc9c`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `9.15765466694941e-13`;
- adjoint true residual: `5.744402887847373e-13`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error: exactly `0.0 eV`;
- missing second-radial source block maximum:
  `2.710505431213761e-19`;
- net-force norm: `9.916793725054143e-17 eV/Angstrom`;
- torque norm: `7.632783294297951e-17 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.5305684963551143e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `1.0983903693555078e-15`.

Across three preregistered proper rotations:

- maximum energy error: `2.2464519133791327e-9 eV`;
- maximum force-covariance relative error: `3.798391250201632e-8`;
- maximum source-covariance relative error: `2.706560324580968e-9`;
- maximum field-covariance relative error: `4.651522667379831e-9`;
- maximum primal residual: `9.158608446094616e-13`;
- maximum adjoint residual: `5.744529907367299e-13`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium dimethyl-ether geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
