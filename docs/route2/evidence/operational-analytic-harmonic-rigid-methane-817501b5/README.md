# Operational analytic-harmonic methane rigid-panel shard

This bundle retains two clean-process executions of the preregistered methane
shard for the disabled operational analytic/original-source harmonic scalar.
It extends contiguous frozen rigid-panel coverage to `[0,7)` without changing
any tolerance after observing the result.

## Bound identity

- execution commit: `817501b5455f6218a66bba4d84b36d36c6deeec2`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `methane`, indices `[6, 7)`
- measurement SHA-256 (both runs):
  `c0977ba027f2a94ef2e71d2db8c9384937c4df0ac9c2d48ea2c531bb77b382c6`
- primary JSON SHA-256:
  `5a38c726d3a1aa35b9e842f3c2621fefc8719c684dd7d85e05dabd0a251f3364`
- replay JSON SHA-256:
  `be084ac90709648d6258100a65b85687f3443838a1cd2cfd1921e9eee0769119`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `5.803213416501237e-13`;
- adjoint true residual: `5.977609313801794e-17`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `4.336808689942018e-19 eV`;
- missing second-radial source block maximum:
  `1.734723475976807e-18`;
- net-force norm: `1.5515838457795457e-17 eV/Angstrom`;
- torque norm: `7.77341958247138e-17 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `2.5483834132006434e-15 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `3.882467425083545e-16`.

Across three preregistered proper rotations:

- maximum energy error: `3.5061020753346384e-10 eV`;
- maximum force-covariance relative error: `8.977094991729323e-9`;
- maximum source-covariance relative error: `1.8199643211398086e-9`;
- maximum field-covariance relative error: `7.235201996874498e-9`;
- maximum primal residual: `6.862769658388697e-13`;
- maximum adjoint residual: `8.423647284431995e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium methane geometry, one translation, one identical-atom
permutation, and three rotations. It is not a Cartesian force-FD, closed-loop,
distorted-geometry, physical-component, solvation-accuracy, PES/domain,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability tiers
remain closed: `E/F/H/V/M = false`.
