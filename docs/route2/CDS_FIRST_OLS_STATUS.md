# First neutral CDS least-squares fit

This is an **exploratory parameter candidate**, not a new production default,
an independent confirmation result, or admission of forces/Hessians.

## Frozen scientific configuration

- Branch A, all ten supported elements including oxygen:
  intrinsic Coulomb radius `1.1 * original-solvent SMD radius + 0.10 Angstrom`.
  Oxygen retains the original catalog acidity when assigning that radius.
- Nonwater CDS descriptors: `alpha * 0.8`, `beta * 1.1`.
- SASA sphere radius: `0.95 * Bondi radius + 0.4 Angstrom`; additional offset zero.
- Original MACE-POLAR frozen source, ddPCM equations and numerical checks;
  physical evaluation remains FP32, lmax 5, Lebedev 194.
- Water has 18 independent coefficients and no molecular surface-tension term.
  Nonwater has 36 coefficients including four molecular terms.

The fitted coefficients are based on the current nonzero **SM8 Table 6** slots.
They are not the original SMD paper's final 37-coefficient parametrization.

## Actual fitting procedure

For fixed geometry, radii, COT environments and solvent descriptors, the CDS
energy is linear in the coefficient vector: `G_CDS = X @ theta`. The target is
experimental total solvation free energy minus the **fixed MAPLE PCM** term,
not DFT-SMD training labels. Transfer rows use organic-minus-water for both
the feature vector and the PCM offset, with the existing standard-state policy.

Only 1,661 neutral training records enter the ordinary, unweighted least-squares
objective: 249 water absolute, 1,329 nonwater absolute, and 83 transfer records.
The 305 validation records are scored after solving and do not enter column
scaling, numerical-rank decisions, or coefficient estimation. That validation
set was already inspected during earlier exploratory tuning; it is not blind.
Confirmation records remain sealed.

The 1,661 by 54 design has 52 covered columns and rank 52. The nonwater O--O
alpha and beta columns have zero training coverage and retain their original
values. All original structural-zero slots remain fixed. There is no intercept,
regularization, positivity restriction, parameter pruning, or radius refit.

The solver uses FP64 column-RMS scaling and a prior-centered linear correction.
For this full-rank supported matrix, the starting coefficients do not impose a
penalty or alter the unique least-squares solution. Rank-deficient inputs are
explicitly reported as nonunique, with minimum RMS-scaled change from the prior
used only as the representative convention. Numerical uniqueness does not
establish statistical significance or stability of sparsely supported terms.

Public solver: [`cds_lstsq.py`](../../tools/route2_release/cds_lstsq.py).
Synthetic regression tests: [`test_cds_lstsq.py`](../../tests/route2_vnext/test_cds_lstsq.py).

## Aggregate results

All errors are kcal/mol relative to experiment. Energy scoring replays the
original ordered FP32 CDS and component-composition arithmetic, not just the
FP64 design-matrix prediction.

| Population | N | Baseline MAE | Fitted MAE | Fitted RMSE |
|---|---:|---:|---:|---:|
| Water absolute | 300 | 1.262626 | 1.028056 | 1.473890 |
| Nonwater absolute | 1549 | 0.716143 | 0.551427 | 0.780236 |
| Water-to-nonwater transfer | 117 | 0.746144 | 0.778632 | 1.064957 |
| All neutral development | 1966 | 0.801319 | 0.637679 | 0.937351 |
| Training | 1661 | 0.803271 | 0.626201 | 0.924730 |
| Validation | 305 | 0.790687 | 0.700187 | 1.003301 |

Transfer MAE worsened: ordinary least squares does not minimize each subgroup's
MAE. Water parameters are separate but coupled to nonwater parameters through
the transfer rows. Some coefficients have only 1--14 supporting training rows;
their fitted values must not be treated as validated transferable parameters.

Baseline replay matched the existing FP32 energy records bitwise. Maximum
linear-design versus FP32 replay differences were below `2.4e-6 kcal/mol`.
This checks implementation consistency, not physical grid convergence.
The measured fit execution, including input validation, feature preparation,
scoring and reporting, took about 25.7 seconds with at most ten memory-admitted,
single-threaded workers; the linear solve itself took about 0.009 seconds.
These are historical timings with existing PCM/source caches, not a fresh
MACE/PCM throughput benchmark.

## Publication and reproducibility boundary

The [aggregate evidence and candidate coefficients](evidence/neutral-A-cds-first-ols-20261022.json)
contain no experimental rows, per-molecule predictions, geometries, model
weights, source arrays or private paths. The executed private solver snapshot
is retained unchanged; the public solver is a byte-identical reusable copy.
The private dataset assembly, SHA-bound manifests, geometry receipts and
per-record evaluations are not redistributed. Consequently, the public summary
and synthetic tests alone cannot reproduce the complete dataset fit.

No fitted coefficient set has been automatically selected or installed as a
calculator default. B, ions, confirmation, forces and Hessians were not fitted
or newly qualified in this experiment.
