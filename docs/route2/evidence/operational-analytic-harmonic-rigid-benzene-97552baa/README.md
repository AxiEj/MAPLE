# Operational analytic-harmonic benzene rigid-panel shard

This bundle retains two clean-process executions of the preregistered benzene
shard for the disabled operational analytic/original-source harmonic scalar.
It extends the contiguous frozen rigid-panel coverage through index 5 without
changing any tolerance after observing the result.

## Bound identity

- execution commit: `97552baab245acb35c6ae5a051a7c3386ea72b41`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `benzene`, indices `[5, 6)`
- measurement SHA-256 (both runs):
  `8f16ef863f5299eb3c7d4ace639c42d804058749740fe6713ced7477cff2f527`
- primary JSON SHA-256:
  `23e6527467ceb8e629a226aff335d9b52e75a386ec25328fe37412d4320e2e01`
- replay JSON SHA-256:
  `1df965f66790ffe35749156f620c3696c231a4e6f92808386d798dad4427ed9e`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 22 Anderson iterations, actual unmixed residual
  `7.908002301122923e-13`;
- adjoint true residual: `8.328419637025845e-13`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.734723475976807e-18 eV`;
- missing second-radial source block maximum:
  `6.505213034913027e-19`;
- net-force norm: `1.6653345369377348e-16 eV/Angstrom`;
- torque norm: `9.990986502170242e-17 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `4.1672459090050264e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `9.683011557945513e-16`.

Across three preregistered proper rotations:

- maximum energy error: `4.2611645767465234e-8 eV`;
- maximum force-covariance relative error: `5.356119997094117e-8`;
- maximum source-covariance relative error: `5.310399867986426e-9`;
- maximum field-covariance relative error: `2.866431904466529e-9`;
- maximum primal residual: `7.908674848785054e-13`;
- maximum adjoint residual: `8.56703328304397e-13`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium benzene geometry, one translation, one identical-atom
permutation, and three rotations. It is not a Cartesian force-FD, closed-loop,
distorted-geometry, physical-component, solvation-accuracy, PES/domain,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability tiers
remain closed: `E/F/H/V/M = false`.
