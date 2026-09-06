# Local decision after the P13 post-pilot Pro review

## Accepted and independently verified

1. The analytic response is an exact derivative backend for the same
   `-1/2 ||C eta||^2` scalar, not a second source model.  Repository tests now
   cover source equality, energy/work identity, parameter and coordinate VJPs,
   gradcheck, gradgradcheck, charge, passivity, rotation and edge orientation.
2. The one-time O2 continuation kept the A3 topology, targets, weights and L2
   regularization fixed.  Full-batch strong-Wolfe L-BFGS was bounded at 500
   outer steps / 5000 objective evaluations with a three-block plateau rule.
3. The 0.002 Ha/e absolute response-MEP gate remains unchanged.  No downstream
   closed-loop error budget currently justifies revising it.
4. A point-q/p permanent source plus a smoother induced P13 source is a coherent
   direct sum only if the induced field is the exact adjoint coordinate and the
   mixed permanent/induced continuum pairing, stationary scalar identity and
   closed-loop Hessian margin all pass.

## Actual disposition

O2 passed its analytic-backend equivalence checks but hit the 500-step cap
without plateau.  The last three 50-step blocks still improved the best total
objective by 1.08%, 1.44% and 1.25%, well above the frozen 0.25% threshold.
Under the preregistered contract this exact A3 candidate is terminal:

- its audit metrics are not reopened;
- it is not selected for grouped CV;
- A4 is not authorized by this run;
- radial-l2 cross-factor candidate C remains forbidden;
- validation and blind formulas remain sealed.

This is not proof that passive P13 response is impossible.  It shows that the
current nonconvex coefficient-head candidate did not reach a defensible
optimization closure within its precommitted budget.  The four point-charge
modes also identify susceptibility action on rank at most four per molecule,
so increasing head capacity would be poorly diagnosed.  The next scientific
branch must improve response identifiability with additional target-independent
input modes (and a separately frozen generator/cost gate), or replace the
nonconvex coefficient parameterization with a globally certifiable one.  It may
not continue the same A3 checkpoint under a larger post-hoc step cap.
