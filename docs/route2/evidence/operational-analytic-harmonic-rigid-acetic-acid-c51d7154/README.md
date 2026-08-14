# Operational analytic-harmonic acetic-acid rigid-panel shard

This bundle retains two clean-process executions of the preregistered
acetic-acid shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,11)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `c51d7154c282c5946a6515ef818cb90f6fdf148d`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `acetic-acid`, indices `[10, 11)`
- measurement SHA-256 (both runs):
  `efc0a12bf086eaac2cc6485379bc1843cf826614b3f1caee574c88d14f4153fa`
- primary JSON SHA-256:
  `926e81d1927d2a119105a4daba13affcb056d13a5263d08ee94e113bbefe390a`
- replay JSON SHA-256:
  `e19b8096d82e2583ef6a1d1431b35df5b9ce84dfb04b5bacd536913e44179663`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.706440107656876e-13`;
- adjoint true residual: `2.244359988468331e-12`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum:
  `1.3877787807814457e-17`;
- net-force norm: `3.8867485032918187e-16 eV/Angstrom`;
- torque norm: `8.727288683396673e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `3.101288114909473e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `3.3776985607183714e-16`.

Across three preregistered proper rotations:

- maximum energy error: `1.2804775906261057e-8 eV`;
- maximum force-covariance relative error: `2.5678550041588242e-8`;
- maximum source-covariance relative error: `4.762861913842296e-9`;
- maximum field-covariance relative error: `1.4546627392879143e-8`;
- maximum primal residual: `8.505056956620726e-13`;
- maximum adjoint residual: `2.2443887620914156e-12`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium acetic-acid geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
