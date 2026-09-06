# Partitioned local-density PCMSolver gate

A frozen positive promolecular Gaussian-mixture stockholder partition assigned
the QM density to unique atom-local ETB beta 2.0 targets. Each target was fitted
only in its one-centre overlap metric, followed by a stable four-constraint
block-diagonal correction.

The development result is **FAIL**:

- mean reaction-metric upper bound: 0.707210 kcal/mol (required <=0.25);
- maximum bound: 1.010777 kcal/mol (required <=0.50);
- mean actual energy error: 0.093641 kcal/mol (diagnostic only);
- maximum local condition: 5.51e4;
- exact constraint residual: 8.53e-14;
- grid count/moment convergence also missed their strict thresholds on the
  larger sulfur-containing cases.

Thus unique atom-local coefficient labels are numerically stable but destroy
important inter-centre density interference. The partition widths, grid gates,
local basis, and thresholds may not be tuned to rescue this experiment. The
stockholder partition remains valid smooth ownership/feature infrastructure,
but these fitted coefficients are rejected as physical scalar-head labels.

No experimental solvation target was read, no model was trained, and no MAPLE
capability is admitted.
