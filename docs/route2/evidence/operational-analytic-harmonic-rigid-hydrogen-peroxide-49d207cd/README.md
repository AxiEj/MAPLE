# Operational analytic-harmonic hydrogen-peroxide rigid-panel shard

This bundle retains two clean-process executions of the preregistered
hydrogen-peroxide shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,17)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `49d207cd358715b236e228c7f631999d57faa7b3`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `hydrogen-peroxide`, indices `[16, 17)`
- measurement SHA-256 (both runs):
  `99236f3f0ad6736214ac96507d7e7149d9eb10b9695f76d82a0514cfc5383918`
- primary JSON SHA-256:
  `e2df917438d353d42ec70eb5d07806e8e48b7f48c41c500b8e3d135b3ec1856c`
- replay JSON SHA-256:
  `d390165a4c2f6a45280d9ddb42f8c28d8e560645aee9b997602a511fd902505a`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.426970530221418e-13`;
- adjoint true residual: `8.382968886509482e-17`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error: exactly `0.0 eV`;
- missing second-radial source block maximum:
  `5.204170427930421e-18`;
- net-force norm: `1.0838898772828417e-16 eV/Angstrom`;
- torque norm: `1.933854021363146e-9 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `8.515428013942885e-15 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `1.0916488435622867e-15`.

Across three preregistered proper rotations:

- maximum energy error: `1.1331394489388913e-8 eV`;
- maximum force-covariance relative error: `4.2576773532666994e-8`;
- maximum source-covariance relative error: `4.201384421908143e-9`;
- maximum field-covariance relative error: `1.145032651465579e-8`;
- maximum primal residual: `9.215063587897527e-13`;
- maximum adjoint residual: `2.9528253631857205e-16`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium hydrogen-peroxide geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
