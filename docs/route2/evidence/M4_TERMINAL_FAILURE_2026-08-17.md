# M4-PH1.0 terminal development failure — 2026-08-17

## Frozen candidate

`M4-PH1.0-v1` was preregistered before its first target-visible fit at source
HEAD `02ecdb27d54de4be9425262e09e72e2595651af5`.  It retained the M3
target-blind three-dimensional stock-area subspace and exact-family ten-fold
OOF split, but replaced nonunique LAD with a strictly convex pseudo-Huber loss
at fixed `delta=1.0 kcal/mol`.

The preregistration froze two independent requirements that matter here:

1. both exact-Hessian `trust-exact` starts must report successful convergence;
2. their held-out predictions must agree within `1e-8 kcal/mol`.

It also froze that any failed fold certificate terminates this stock-area 3D
lane.  The solver, tolerance, transition scale, and rule may not be changed
after the failure.

## First target-visible execution

The first execution stopped on fold 0 before producing an OOF prediction or
MAE:

```text
M4FitError: Pseudo-Huber zero-start solve failed:
A bad approximation caused failure to predict improvement.
```

The exact external artifacts are:

```text
preregistration file SHA256:
0e844d937cd43eb864966d28d6468599607cd27ad9cafb889dcfce75b2211444

terminal failure file SHA256:
36de2acfbd3389ef117ed15266d56789104c796c85cef52d663c4ad728469d8b

terminal failure self SHA256:
a12a22425e40b26dc1637713e8c20690901aa1737b1f5aa139690fc945a17495

stderr SHA256:
dee07e5ca794d4a09230c4ffb078caf0234f4cb28b555c03b2536e33e7435e8f
```

The failure artifact records:

```text
status = development-fail-numerical-certificate
failed_fold = 0
failed_start = zero-start
failed_certificate = scipy-trust-exact-success-status
oof_prediction_completed = false
oof_mae_computed = false
confirmation_partition_opened = false
threshold_solver_or_rule_changed_after_failure = false
```

## Failure-after diagnostic only

After the terminal outcome had been frozen, one read-only fold-0 diagnostic
replayed the optimizer endpoints without changing or overriding the M4 result.
The training design had 275 rows, standardized condition number `4.6275`, and
singular values

```text
[1.3882920816, 0.9912821228, 0.3000080820].
```

Both starts reached the same strictly convex basin but both returned SciPy
status 2.  The zero-start endpoint had gradient infinity norm `1.109e-12`;
the OLS-start endpoint had `8.844e-11`.  Their normalized objectives differed
by `2.22e-16`, while their held-out predictions differed by
`3.745e-8 kcal/mol`, above the frozen `1e-8` replay threshold.

These values explain the numerical failure but do not convert it into a pass.
No M4 accuracy result exists.  In particular, the earlier M2 water MAE of
`1.626595 kcal/mol` remains the last completed result for this diagnostic
stock-area family and still misses the `1.5 kcal/mol` development gate.

## Scientific consequence

M4 does not show that pseudo-Huber lacks an exact unique minimizer.  It shows
that the preregistered floating-point certificate for this candidate was not
met.  Replacing the optimizer or loosening replay tolerances would be a new
post-result candidate and would violate the explicit terminal policy for the
stock-area 3D lane.

The production smooth-harmonic same-cavity area is a different physical
feature definition.  It may be evaluated only under a new target-blind
preregistration and cannot inherit an M4 pass, coefficient, or accuracy claim.
Confirmation remains sealed.
