# Auxiliary-density PCMSolver reaction-metric gate

This gate was prospectively frozen after the exterior-MEP v1 failure and a
genuine Chrome Pro 5/5 review. It retained all twelve molecules and compared
the direct frozen QM AO MEP with the constrained `df.make_auxbasis` MEP through
the same symmetric intrinsic-SMD-water PCMSolver operator.

The locked result is **FAIL**:

- mean reaction-metric energy-error upper bound: 0.330061 kcal/mol
  (required <= 0.25);
- maximum bound: 0.480214 kcal/mol (required <= 0.50);
- mean actual fixed-source energy error: 0.127611 kcal/mol (diagnostic only);
- maximum actual error: 0.165494 kcal/mol;
- maximum reciprocity defect: 4.02e-16;
- maximum moment-constraint residual: 4.32e-12.

Methane is no longer the limiting case: its operator-level bound is 0.09950
kcal/mol. The earlier 19.63% relative-MEP failure was therefore a low-signal
normalization problem, but that does not rescue the basis because the
prospectively locked mean reaction-metric gate fails.

`df.make_auxbasis` is terminated for the final scalar-head representation. No
threshold or case was changed, no experimental solvation target was read, and
no capability is admitted. A distinct literature-backed, systematically
convergent auxiliary basis must pass a new prospective gate before training.

The first invocation imported the separate pure-MACE-POLAR worktree through an
ambient `PYTHONPATH` and stopped before any case evaluation or result write.
The retained execution used an explicit hybrid-worktree `PYTHONPATH` and the
same unchanged preregistration.
