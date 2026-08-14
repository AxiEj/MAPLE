# Operational analytic-harmonic acetone rigid-symmetry canary

This bundle retains two clean-process executions of the preregistered acetone
shard for the disabled operational analytic/original-source harmonic scalar.
It is the fourth equilibrium molecule executed from the frozen 20-molecule
rigid-symmetry contract; tolerances were not changed after seeing the result.

## Bound identity

- execution commit: `371a2b605b09bd88286555360b72a5c34baf4f8f`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `acetone`, indices `[3, 4)`
- scalar:
  `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`
- measurement SHA-256 (both runs):
  `38efae6ff40ac225b6f1992bcfb00a62249ac695a607bbdb35a7915b3212d387`
- primary JSON SHA-256:
  `faec8db8b875ffad20001d74bb02842add026919be4f77336db442567b5947c5`
- replay JSON SHA-256:
  `5d42af54f3cec4504f373f70c9d725243c8521c122aab94da31de74dcd1355c1`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `4.0773181220552224e-13`;
- adjoint true residual: `8.13212155028115e-16`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error: exactly `0.0 eV`;
- missing second-radial source block maximum:
  `1.3877787807814457e-17`;
- net-force norm: `3.291406376790657e-16 eV/Angstrom`;
- torque norm: `2.223966914868445e-16 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.0212924774920328e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `4.2759257641501914e-16`.

Across three preregistered proper rotations:

- maximum energy error: `6.936716090422124e-9 eV`;
- maximum force-covariance relative error: `2.0250564002461316e-8`;
- maximum source-covariance relative error: `6.218784064316717e-9`;
- maximum field-covariance relative error: `3.212899614059117e-8`;
- maximum primal residual: `4.1500973478519087e-13`;
- maximum adjoint residual: `8.625906011240059e-16`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium acetone geometry, one translation, one identical-atom
permutation, and three rotations. It is not a Cartesian force-FD, closed-loop,
distorted-geometry, physical-component, solvation-accuracy, PES/domain,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. All public capability tiers
remain closed: `E/F/H/V/M = false`.
