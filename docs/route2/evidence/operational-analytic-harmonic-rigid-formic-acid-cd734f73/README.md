# Operational analytic-harmonic formic-acid rigid-panel shard

This bundle retains two clean-process executions of the preregistered
formic-acid shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,10)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `cd734f732f655012ca6595d0f99570b5531af601`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `formic-acid`, indices `[9, 10)`
- measurement SHA-256 (both runs):
  `5440c145c1e616d5ca180ed67d51662c1f47e3d430d6bfaa1e9585ecf6c266a5`
- primary JSON SHA-256:
  `69e5c2d7a6907ba0130e6f737679487e53f429715d84f5304a4078948f554b1d`
- replay JSON SHA-256:
  `b079741a8b9bbd9ad90e6dec015b0783ca7a819239b4fdcdce343792bdc1ca6f`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.248654039937734e-13`;
- adjoint true residual: `1.633111005463644e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum:
  `6.071532165918825e-18`;
- net-force norm: `1.3947004136484323e-16 eV/Angstrom`;
- torque norm: `9.421059210534821e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `2.418321514127715e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `5.411771650428679e-16`.

Across three preregistered proper rotations:

- maximum energy error: `1.4257238944992423e-8 eV`;
- maximum force-covariance relative error: `3.227108575652941e-8`;
- maximum source-covariance relative error: `5.5699893264746404e-9`;
- maximum field-covariance relative error: `8.798010362147867e-9`;
- maximum primal residual: `8.202151676028382e-13`;
- maximum adjoint residual: `1.635558017206375e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium formic-acid geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
