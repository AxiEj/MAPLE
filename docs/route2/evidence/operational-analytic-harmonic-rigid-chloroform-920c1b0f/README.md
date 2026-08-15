# Operational analytic-harmonic chloroform rigid-panel shard

This bundle retains two clean-process executions of the preregistered
chloroform shard for the disabled operational analytic/original-source
harmonic scalar. It completes contiguous frozen equilibrium rigid-panel
coverage over all twenty registered molecules, `[0,20)`, without changing any
tolerance after observing the result.

## Bound identity

- execution commit: `920c1b0f6474c0d201e02549288c3e45437a5907`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `chloroform`, indices `[19, 20)`
- measurement SHA-256 (both runs):
  `99020fbf5d3a893fd3820e682004aaaf4ea0fb7cade9bb7df74550aa9ba921dd`
- primary JSON SHA-256:
  `6b7b0f050e78d0bddf5b5419240f964daa6548c7984d01cddbc6d02b57872062`
- replay JSON SHA-256:
  `7ce5e5e4fb3b8a06334675173653eed8f3101054f734deafa0567b49f471c564`

The complete files differ only in command/output metadata and wall-clock
runtime. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `5.341749550461511e-13`;
- adjoint true residual: `6.164335186956882e-14`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error: exactly `0.0 eV`;
- missing second-radial source block maximum: exactly `0.0`;
- net-force norm: `1.993045536433795e-17 eV/Angstrom`;
- torque norm: `2.0201465500489977e-11 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.1127089385332281e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `7.327572826701294e-16`.

Across three preregistered proper rotations:

- maximum energy error: `3.2669049687683582e-9 eV`;
- maximum force-covariance relative error: `2.1676643473015314e-8`;
- maximum source-covariance relative error: `5.141216133607704e-9`;
- maximum field-covariance relative error: `5.0708117686345365e-8`;
- maximum primal residual: `9.019087642060526e-13`;
- maximum adjoint residual: `6.174559832555903e-14`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly. Together with the nineteen preceding
shards, this completes the preregistered twenty-molecule equilibrium rigid
panel. Across the complete panel the maximum recorded rotation-energy error is
`4.2611645767465234e-8 eV` (benzene) and the maximum relative force-covariance
error is `6.934036634982884e-8` (pyridine), both well below the frozen local
thresholds.

## Claim boundary

This closes only equilibrium rigid translation/rotation/permutation and replay
coverage for the twenty frozen geometries. It is not a Cartesian force-FD,
closed-loop, distorted-geometry, physical-component, solvation-accuracy,
PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability
tiers remain closed: `E/F/H/V/M = false`. The result supports the new
coefficient-space replacement; it does not repair or admit the legacy finite
laboratory-grid route.
