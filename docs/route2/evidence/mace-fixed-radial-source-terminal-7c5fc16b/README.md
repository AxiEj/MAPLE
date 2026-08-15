# Fixed radial-source terminal audit

This directory retains two clean executions at Git head
`7c5fc16b5bcb98221a97d5663329941ee74fe4f3` of the candidate grid, split,
selection rule, and gates frozen in
`mace-original-source-fixed-radial-embedding-v1.json` before any candidate
result was read.  Both executions reproduce scientific measurement SHA-256
`77cd7ab8a845820a78d0236395411ace5c68fd3568c27e999e0a15705291f7bc`.

The runner uses the current analytic-Gaussian evaluator profile.  The older
twelve-molecule artifact used the upstream molecular-realspace evaluator, so
its MACE predictions are not treated as the same source identity.  Only its
frozen QM MEP, geometry, and geometry-only probe-surface assets are reused.

## Preregistered model change

The learned four-channel source is unchanged.  Every atom-centred monopole and
dipole is represented by one universal convex mixture of normalized Gaussian
radial primitives.  The same mixture is used for every element, atom, `l`, and
`m`; weights sum to one, so charge and dipole moments are preserved exactly.
No held-out MEP, PCMSolver energy, experimental solvation label, or CDS/SMD
term enters candidate selection.

Eight frozen surface-MEP records select among 21 candidates.  Four chemically
diverse MEP records are held out from selection.  The selected candidate is a
single `sigma=0.75 Angstrom` normalized Gaussian:

```text
configuration SHA-256 2477a79a7331d3b01f56697aeeac6f7e2c559bedb13e7aeb26175bee634394f3
training objective   0.5277461282 -> 0.05623146425
relative improvement 0.8934497834
held-out objective   0.1754427659 -> 0.02034296929
held-out ratio       0.1159521692
```

Every held-out weighted RMSE improves relative to the original 1.5-A Gaussian;
the selected/baseline ratios are `0.3283` (benzene), `0.3895` (acetone),
`0.3699` (acetonitrile), and `0.3479` (acetic acid).  The preregistered static
MEP gate therefore passes.

## Decisive matched PCMSolver result

The selected candidate is then evaluated, without refitting, against the four
frozen total-QM MEPs on their identical intrinsic PCMSolver cavities:

| compound | QM PCM (kcal/mol) | selected source PCM (kcal/mol) | absolute error (kcal/mol) | cavity MEP relative error |
| --- | ---: | ---: | ---: | ---: |
| acetic acid | -10.983432 | -7.392472 | 3.590961 | 0.303757 |
| benzene | -3.136303 | -1.307773 | 1.828530 | 0.797055 |
| 2-acetoxyethyl acetate | -12.747357 | -7.859652 | 4.887705 | 0.415215 |
| acetone | -6.643339 | -4.535586 | 2.107753 | 0.381104 |

All four exceed the inherited, preregistered `1 kcal/mol` fixed-source budget.
Thus the apparent improvement on the more distant geometry-only surface does
not transfer to the actual cavity-near-field electrostatic component.

## Terminal decision

The single authorized fixed-radial experiment **fails**.  The repository must
not add element-specific, geometry-dependent, cavity-dependent, or
energy-fitted radial patches.  The unchanged source and this radial repair may
select neither `Phi0` nor `Phi1Delta`, and no E/F/H/V/M capability is admitted.

Per the frozen termination rule, Route 2 now transitions to a separately
identified scalar-first field head or an independent variational polarization
model.  Existing vacuum MACE can remain frozen, but the polarization/source
model needs its own scalar, provenance, training/reference data, and gates.
