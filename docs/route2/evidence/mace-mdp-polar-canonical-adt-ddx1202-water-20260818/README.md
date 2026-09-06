# Canonical-ADT MACE-MDP + MACE-POLAR ddX target-free canary

This directory freezes the **target-free** prequalification of the operational
hybrid profile
`route2-research-mace-mdp-point-polar-residual-adt-ddx-operational-v1`.
The calculation read no experimental solvation target and performed no fitting,
calibration, post-training, source scaling, or CDS adjustment.

The exact source categories are kept as a direct sum:

1. frozen official MACE-MDP permanent point monopoles/dipoles;
2. frozen official MACE-POLAR nonuniform zero-anchored 1.5 A Gaussian residual;
3. the canonical free-atom-density atomic-displacement-translation lift of the
   same official MACE-MDP molecular polarizability.

All three branches share one pyddx PCM coefficient solve. MACE-MDP ran on CPU;
MACE-POLAR ran on CUDA. The selected continuum settings are `lmax=8` and
`n_lebedev=1202`. Both were chosen without solvation targets:

- `n_lebedev=1202` reduced the fixed-source rigid-rotation energy drift from
  `9.3051e-5` to `2.7713e-6 eV` relative to the 194-point rule;
- `lmax=8` differs from the target-free `lmax=15` water reference by
  `1.37793e-4 eV` (`0.00318 kcal/mol`) and by `4.89496e-4` in relative native
  field norm, below the predeclared `0.01 kcal/mol` / `1e-3` budgets.

All fifteen prequalification gates passed. Key measurements include:

- five deterministic starts: same root, maximum 21 iterations;
- primal residual: `6.53645e-11 eV`;
- JVP/VJP relative dot defect: `1.60067e-12`;
- finest centered finite-field error: `1.81526e-8` with clean second-order decay;
- local residual-Jacobian minimum singular value: `0.943680`;
- local contraction norm: `0.0653324`;
- canonical uniform-response eigenvalues:
  `0.0976066, 0.0999690, 0.1043815`;
- combined rotation/translation energy drift: `2.79385e-6 eV`;
- combined native-field covariance error: `2.20163e-5` relative.

The finite ddX laboratory quadrature remains a numerical approximation rather
than a structural SO(3) proof. This artifact permits freezing the exact profile
before its 505-development accuracy run. It makes **no accuracy, global-root,
force, Tier-V, or public E/F/H/V/M claim**.

## Reproduction

```bash
python tools/route2_release/run_mace_mdp_polar_canonical_adt_ddx_water_canary.py \
  --polar-device cuda \
  --output /tmp/route2-canonical-adt-ddx1202-water-canary-v1-run3.json
```

The measurement JSON binds the exact command, Git identity, dirty-worktree
status, checkpoint SHA-256 values, loaded implementation source hashes, runtime,
and all numerical gates. Its internal artifact SHA-256 is
`ee004c30376024df8e516caa2621572fa24e750b8efdc3c18d203ef7b2b76b5a`.
