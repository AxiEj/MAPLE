# VQM24 A3 O2 optimization closure

The exact analytic derivative backend matches the scalar-autograd source to
`1.11e-16`, the full parameter gradient to `3.73e-17` absolute / `2.68e-14`
relative, the energy/work identity to `1.39e-17`, and exact induced charge to
`2.78e-17 e`.  Reduced gradcheck/gradgradcheck and rotation/edge-orientation
unit tests also pass.

One preregistered full-batch strong-Wolfe L-BFGS continuation reduced the total
training objective from `0.0169507` to `0.0116846` over 500 outer steps and
1567 objective evaluations.  It did not satisfy the frozen plateau rule: the
last three 50-step blocks still improved the best objective by 1.08%, 1.44%,
and 1.25%, each above 0.25%.  Therefore the run terminates as
`fail-a3-o2-no-plateau`; audit gates were not reopened, A4 is not authorized by
this candidate contract, and validation/blind formulas remain sealed.
