# Operational analytic-harmonic acetonitrile rigid-symmetry canary

This bundle retains two clean-process executions of the preregistered
acetonitrile shard for the disabled operational analytic/original-source
harmonic scalar. It is the fifth equilibrium molecule executed from the frozen
20-molecule rigid-symmetry contract; tolerances were not changed after seeing
the result.

## Bound identity

- execution commit: `cf43e050f2c7abfa7f864a516dbe5fe71fac2bf5`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `acetonitrile`, indices `[4, 5)`
- measurement SHA-256 (both runs):
  `2f01bb45ffa23f1dba09ee15123e71dbae0f742b607ff231758423487cfde616`
- primary JSON SHA-256:
  `3b3c168582f5da7f6d2b06dc19c995477a24802e9e073c38732fb2d765910c52`
- replay JSON SHA-256:
  `7d822548a0aadc09846a9c782d5bda7a21963b4461b18bc3470cbe0814fa7eaf`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.5044820892005636e-13`;
- adjoint true residual: `9.575193078134051e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.3877787807814457e-17 eV`;
- missing second-radial source block maximum:
  `2.168404344971009e-18`;
- net-force norm: `5.83568815493317e-16 eV/Angstrom`;
- torque norm: `2.1959850604666625e-11 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `2.8926200964921166e-15 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `2.1502325476490493e-16`.

Across three preregistered proper rotations:

- maximum energy error: `1.7384991224389523e-8 eV`;
- maximum force-covariance relative error: `1.7558870386345935e-8`;
- maximum source-covariance relative error: `3.395228677265498e-9`;
- maximum field-covariance relative error: `8.92400703128298e-9`;
- maximum primal residual: `7.579717824709418e-13`;
- maximum adjoint residual: `9.575338705685621e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium acetonitrile geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
