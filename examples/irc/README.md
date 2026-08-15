# Generic IRC examples and admission boundary

These examples exercise MAPLE's **generic molecular IRC infrastructure**. They
are not part of the AIMNet2/pyddx Route-2 evidence set, and success here cannot
open any Route-2 OPT, FREQ/TS/IRC, or MD gate.

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

After preflight, the common path contract inserts the exact TS once, writes a
deterministic endpoint-to-TS-to-endpoint full trajectory, records why each leg
stopped, and by default rejects a run unless both maximum and RMS force
thresholds pass. The retained defaults are `max_steps=50`,
`f_max_th=2e-3 Hartree/Angstrom`, and
`f_rms_th=5e-4 Hartree/Angstrom`; the explicit fail-closed TS/path energy
tolerance is `path_energy_tolerance_hartree=1e-7`.

This still does not prove endpoint minimum identity, chemical connectivity, or
step-size convergence. Old generated `.out` and trajectory files under the
method directories predate the shared starting-point and path contracts and
are not current validation artifacts.

See [the stationary-point validation contract](../../docs/STATIONARY_POINT_VALIDATION.md)
and the [IRC path and termination contract](../../docs/IRC_PATH_VALIDATION.md).
