# Operational analytic-harmonic ethanol rigid-symmetry canary

This bundle retains two clean-process executions of the preregistered ethanol
shard for the disabled operational analytic/original-source harmonic scalar.
It extends the same frozen 20-molecule rigid-symmetry contract previously run
for water and methanol; it does not alter tolerances after seeing the result.

## Bound identity

- execution commit: `538f9f4d350c4f2f0ed0519a59dbae4858df75a7`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen 20-molecule geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `ethanol`, indices `[2, 3)`
- scalar:
  `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`
- source identity: original four-channel learned density head, embedded only in
  the first radial block
- measurement SHA-256 (both runs):
  `575420ae56c17f8dc37d28536ba86c9cba0f13559552f6afdc7934c4a2945937`
- primary JSON SHA-256:
  `8e23ff06d4571c3b72ebb12e49ef8be3808a12cf697da3ce40f9018d0a2a1030`
- replay JSON SHA-256:
  `b259a5fd0d0fbe7f239467fb42180be8519b0c0c68146e230ccffbb580c79696`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Executed command

```bash
python tools/route2_release/run_operational_analytic_harmonic_rigid_panel.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --molecule-start 2 \
  --molecule-stop 3 \
  --max-iterations 160 \
  --output /tmp/route2-operational-harmonic-rigid-ethanol-538f9f4d-run1.json
```

The replay changed only the output path.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.3676767929859406e-13`;
- adjoint true residual: `3.548825119053039e-13`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `1.3877787807814457e-17 eV`;
- missing second-radial source block maximum: exactly `0.0`;
- net-force norm: `5.815852873680767e-17 eV/Angstrom`;
- torque norm: `2.3096049378557194e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `1.1801983333760825e-14 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `4.981661729142136e-16`.

Across three preregistered proper rotations:

- maximum energy error: `1.9072103896178305e-9 eV`;
- maximum force-covariance relative error: `4.7353129839838976e-8`;
- maximum source-covariance relative error: `3.2020656091357397e-9`;
- maximum field-covariance relative error: `2.3729140073528977e-9`;
- maximum primal residual: `7.856985340851677e-13`;
- maximum adjoint residual: `3.548967219884553e-13`.

Every frozen rigid-symmetry gate passed. Both clean processes reproduce the
same scientific measurement exactly.

## Claim boundary

This is one equilibrium ethanol geometry, one translation, one identical-atom
permutation, and three rotations. It is not a Cartesian force-FD, closed-loop,
distorted-geometry, physical-component, solvation-accuracy, PES/domain,
Hessian/FREQ/TS/HVP/NVE/MD, or release result. It does not prove global
floating-point rotation covariance or chemical accuracy.

All public capability tiers remain closed:

```text
E = false
F = false
H = false
V = false
M = false
```
