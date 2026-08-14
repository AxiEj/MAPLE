# Operational analytic-harmonic methanol rigid-symmetry canary

This bundle retains two clean-process executions of the preregistered methanol
shard for the disabled operational analytic/original-source harmonic scalar.
It directly revisits the molecule for which the higher-order fixed-grid
CPCM1202 profile had still failed its rotation energy and force gates.

## Bound identity

- execution commit: `93c985980756a590b761839043e38cb9f1107df7`
- official checkpoint SHA-256:
  `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`
- frozen 20-molecule geometry asset SHA-256:
  `ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3`
- molecule/shard: `methanol`, indices `[1, 2)`
- scalar:
  `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`
- source identity: original four-channel learned density head, embedded only in
  the first radial block
- measurement SHA-256 (both runs):
  `2bd5b02b6376faa39789cd9b081985f6805903a9ce227044842bf57af666ccfe`
- primary JSON SHA-256:
  `d9022f228f0838c732e1f33fb85972d0380465984d4cb62787265b4953f64334`
- replay JSON SHA-256:
  `e2a5f3c8b7656bb77f60d78eb7f1f31aeeac71c69b5a093b1ef1ad66b5ff5647`

The complete files differ only in execution metadata, output path, warnings,
and wall-clock time. Their contract, molecule measurement, decision, and
`measurement_sha256` are identical.

## Executed command

```bash
python tools/route2_release/run_operational_analytic_harmonic_rigid_panel.py \
  --checkpoint /home/axie/.cache/mace/MACEPOLAR1Mmodel \
  --device cuda \
  --molecule-start 1 \
  --molecule-stop 2 \
  --max-iterations 160 \
  --output /tmp/route2-operational-harmonic-rigid-methanol-93c98598-run1.json
```

The replay changed only the output path.

## Result

- reference root: 23 Anderson iterations, actual unmixed residual
  `3.1258015520278824e-13`;
- adjoint true residual: `6.934609168311564e-16`;
- cold/warm source, field, and scalar differences: exactly zero;
- continuum half-coupling identity error:
  `3.469446951953614e-18 eV`;
- missing second-radial source block maximum:
  `5.692061405548898e-19`;
- net-force norm: `1.3455014484288012e-16 eV/Angstrom`;
- torque norm: `5.441308376097357e-10 eV`;
- translation energy error: exactly `0.0 eV`;
- translation force difference:
  `5.702249140124941e-15 eV/Angstrom`;
- identical-atom permutation energy error: exactly `0.0 eV`;
- identical-atom permutation force relative error:
  `5.953525952799556e-16`.

Across three preregistered proper rotations:

- maximum energy error: `2.8617250791285187e-9 eV`;
- maximum force-covariance relative error: `5.000913075374153e-8`;
- maximum source-covariance relative error: `2.592598170552474e-9`;
- maximum field-covariance relative error: `4.269092690650618e-9`;
- maximum primal residual: `5.6774011402604e-13`;
- maximum adjoint residual: `7.006299331053965e-16`.

Every frozen rigid-symmetry gate passed.

## Comparison with the failed CPCM1202 grid route

The retained fixed-grid CPCM1202 methanol evidence reported:

- maximum rotation energy error: `5.079472884972347e-6 eV`;
- maximum rotation force-covariance relative error:
  `1.7880747343495252e-4`;
- maximum source-covariance relative error: `1.6934783599427846e-6`.

The new measured maxima are smaller by factors of approximately `1775`,
`3575`, and `653`, respectively. This is not a same-profile numerical tweak:
the new scalar uses a separately identified analytic isotropic MACE evaluator
and a fixed-dimensional coefficient-space harmonic continuum. Therefore the
result supports the replacement route; it does not rehabilitate the finite
laboratory-fixed Lebedev/SWiG implementation.

## Claim boundary

This is one equilibrium methanol geometry, one translation, one identical-atom
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
