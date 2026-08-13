# Route 2 residual-refinement force-error gate

This document preregisters the first real-stack estimate of the force error
left by incomplete primal and adjoint solves. It does **not** enable a
capability and it does not alter the canonical scalar.

## Scope and scalar

The frozen panel is the 20 reference geometries in
`fixedbox590_pes_panel_v1.json`, under the disabled profile
`route2-profile-diagnostic-fixedbox40-cpcm590-radialgto-electrostatic-v1`.
Every force differentiates only

```text
E_op(R) = E_vac(R) + 0.5 <c*(R), P_R c*(R)>_Q
G_np = 0
```

No CDS/nonpolar term, standard-state term, field-conditioned MACE energy
difference, or experimental solvation free energy enters this gate.

## Frozen two-axis refinement

Primal and adjoint residuals are not combined into one arbitrary mixed-unit
norm. For each molecule the runner evaluates two separate sequences:

1. **adjoint refinement:** reuse one exact release-primal state at requested
   tolerance `1e-12`; solve the adjoint at `(rtol, atol)` values
   `(1e-11, 1e-13)`, `(1e-12, 1e-14)`, and `(1e-13, 1e-15)`;
2. **primal refinement:** solve roots at requested tolerances `1e-12`, `1e-13`,
   and `1e-14`; evaluate every force with the tightest adjoint settings
   `(1e-13, 1e-15)`.

The tight-adjoint/release-primal force is one bitwise-identical bridge shared
by both sequences. Measured residual ceilings are, separately:

| level | primal norm ceiling | adjoint norm ceiling |
| ---: | ---: | ---: |
| 0 | `1e-12` | `1e-10` |
| 1 | `1e-13` | `1e-11` |
| 2 | `1e-14` | `1e-12` |

The contract is
`route2-fixedbox590-residual-refinement-force-error-contract-v1`.

## Estimator and frozen gate

For each refinement axis, let `d01` and `d12` be the componentwise absolute
force changes. Outside a preregistered numerical plateau of `1e-8 eV/A`, the
maximum refinement ratio `q=max(d12)/max(d01)` must be at most `0.5`. The
componentwise contribution estimate is

```text
2 * (d01 + d12 / (1 - q))
```

where `2` is a frozen safety factor. At the numerical plateau, both observed
differences are retained and the estimate is `2 * (d01 + d12)`; no apparent
convergence order is manufactured from roundoff. The total estimate is the
componentwise sum of the independently estimated primal and adjoint
contributions.

The total must dominate the directly observed change from the release force
to the tight-primal/tight-adjoint reference force. Both its RMS and maximum
component must be no more than `5e-5 eV/A`, which is ten percent of the
preregistered `5e-4 eV/A` Cartesian RMS tolerance and is stricter than ten
percent of the separate `2e-3 eV/A` component-maximum tolerance.

This is a conservative, empirical a-posteriori **residual-refinement
estimate**, not a rigorous analytic upper bound. A noncontracting sequence,
missing molecule, wrong tolerance, mismatched bridge force, residual-ceiling
violation, or over-budget estimate fails closed.

## Execution sequence

The contract and runner must first be committed on a clean tree. Then execute
one source-bound shard per molecule and aggregate only on the exact tested Git
tree:

```bash
python tools/route2_release/run_fixedbox590_residual_force_panel.py \
  --molecule-start 0 --molecule-stop 1 \
  --output /absolute/path/residual-force-shard-00-01.json

python tools/route2_release/aggregate_fixedbox590_residual_force_panel.py \
  --shard /absolute/path/residual-force-shard-00-01.json \
  ... \
  --output /absolute/path/residual-force-aggregate.json
```

Thresholds must not be changed after the real panel is observed. Regardless of
the outcome, E/F/H/V/M remain false until every other admission gate passes.
