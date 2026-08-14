# Operational analytic-harmonic trans-butane rigid-panel shard

This bundle retains two clean-process executions of the preregistered
trans-butane shard for the disabled operational analytic/original-source
harmonic scalar. It extends contiguous frozen rigid-panel coverage to
`[0,8)` without changing any tolerance after observing the result.

## Bound identity

- execution commit: `77abe42b0622ea9f829ce00de213c89a0491cbb7`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `trans-butane`, indices `[7, 8)`
- measurement SHA-256 (both runs):
  `44102bdffb58e49e95694fe2e7993f661b2178aa23136c92208f2f8b818a3734`
- primary JSON SHA-256:
  `7e68af6f9f342103d04cfe9712aea92743b3f3bd7c49d7e4a122e1ba090c6ac4`
- replay JSON SHA-256:
  `02985a242f0093d56420b97342eab3099821a95dffcef06cc220a7ada092e441`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `7.652889002253314e-13`;
- adjoint true residual: `2.1753222515410796e-12`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.734723475976807e-18 eV`;
- missing second-radial source block maximum:
  `1.3010426069826053e-18`;
- net-force norm: `5.815852873680767e-17 eV/Angstrom`;
- torque norm: `1.819463801844281e-9 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.5878248913511165e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `6.819242177825913e-16`.

Across three preregistered proper rotations:

- maximum energy error: `2.6966517907567322e-9 eV`;
- maximum force-covariance relative error: `3.591746985795435e-8`;
- maximum source-covariance relative error: `2.899118004565727e-9`;
- maximum field-covariance relative error: `7.601105449126282e-9`;
- maximum primal residual: `7.65300346861557e-13`;
- maximum adjoint residual: `2.1753307182778503e-12`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium trans-butane geometry, one translation, one
identical-atom permutation, and three rotations. It is not a Cartesian
force-FD, closed-loop, distorted-geometry, physical-component,
solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result.
All public capability tiers remain closed: `E/F/H/V/M = false`.
