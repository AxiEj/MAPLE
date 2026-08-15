# IRC starting-point contract

The `gs`, `lqa`, `hpc`, and `eulerpc` examples all start from the retained
ANI-1xnr transition-state geometry in `../ts/neb/inp1_nebts_ts.xyz`.

Before any IRC step, MAPLE now requires the input geometry to pass the shared
molecular first-order-saddle preflight:

```text
#irc(method=gs,target_mode=1,stationarity_tolerance_ev_per_a=1e-3,hessian_symmetry_relative_tolerance=1e-6,rigid_mode_tolerance_cm1=5,transition_state_imaginary_threshold_cm1=50)
```

The initial forces and Hessian arrive through MAPLE's private Hartree job view,
are converted back to public ASE units, and are checked with the same
mass-metric rigid/vibrational decomposition used by public FREQ. The start must
have exactly one robust imaginary vibrational mode; the projected eigenvector
defines the forward/backward initial directions.

This is a starting-point admission only. Converged trajectories still require
step-size/convergence studies, endpoint optimization, and an independent check
that the two branches connect the intended minima. Old generated `.out` and
trajectory files under the method directories predate this shared preflight
and are not current validation artifacts.

See [the stationary-point validation contract](../../docs/STATIONARY_POINT_VALIDATION.md)
and the [ORCA IRC reference](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/irc.html).
