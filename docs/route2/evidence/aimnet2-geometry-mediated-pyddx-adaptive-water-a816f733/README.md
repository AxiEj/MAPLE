# AIMNet2 geometry-mediated pyddx adaptive water canary

This directory retains two independent clean-tree executions of commit
`a816f73365623800a8f1c68058ee34b9fe982eaf` with the unchanged SHA256-bound
AIMNet2 checkpoint, the source-bound reconstructed Python float64 runtime,
`pyddx==0.8.0`, and `scipy==1.17.1`.

Both executions reproduce the exact scientific measurement hash
`6fad81d35b8da5926e0f9117baed0f6e88d295729fd3ab3d56df9016a595061c`.
Runtime duration and command/output-path metadata are intentionally excluded
from that measurement hash. The artifact retains raw operands for the
adaptive, frozen-step directional, full Cartesian, reciprocity, and rigid
rotation reducers.

## Result

- deterministic center replay: passed exactly;
- registered-metric reciprocity/apply-adjoint/charge-direction FD/gauge audit:
  passed;
- pinned adaptive directional derivative: converged in two iterations and seven
  unique scalar samples; analytic-vs-numerical error
  `3.190291369137288e-07 eV/angstrom`;
- exact ddX sphere/Lebedev active-set reconstruction: passed at every sampled
  geometry;
- center conservative distance-to-active-set-event lower bound:
  `5.826261813812086e-09 angstrom`, far below the frozen `0.02 angstrom` guard;
- adaptive samples do not remain in one cavity active-set stratum;
- fixed three-step directional and complete Cartesian numerical convergence
  pass numerically, but their overall gates fail on the active-set clearance;
- frozen rigid rotations change the laboratory-frame cavity active set and fail
  energy/force covariance;
- pyddx exposes the requested solve tolerance but no measured post-solve
  algebraic residual.

Therefore all E/F/H/V/M capabilities and OPT/FREQ/TS/IRC/MD remain false. This
is negative admission evidence for the finite-grid pyddx branch, not a failure
of the weak composite-scalar theorem inside one fixed smooth stratum, and not
fixed-geometry mutual polarization.

The active-set reconstruction follows the union-of-spheres characteristic
function in the [ddX theory documentation](https://ddsolvation.github.io/ddX/md_docs_theory.html)
and the version-pinned ddX 0.8.0 implementation. The adaptive trace delegates
to [`scipy.differentiate.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html)
and retains every raw abscissa for replay.
