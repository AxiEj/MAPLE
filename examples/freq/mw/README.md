# Molecular FREQ template

The public FREQ workflow accepts a stationary, unconstrained, non-periodic
molecule and a calculator that implements a public ASE-unit Cartesian Hessian
(`eV/angstrom^2`). Run geometry optimization first, then use its converged
structure in a separate input.

For a minimum with ideal-gas RRHO thermochemistry:

```text
#model=<hessian-capable molecular model>
#freq(method=mw,stationary_point=minimum,temperature=298.15,pressure_kpa=101.325,symmetry_number=1,ilowfreq=0)
#device=<cpu-or-gpu>

XYZ <charge> <multiplicity> /absolute/path/to/converged_minimum.xyz
```

The default gates require a maximum force component no larger than
`1e-3 eV/angstrom`, a raw Hessian symmetry defect below the declared relative
tolerance `1e-6`, and a raw rigid-subspace Hessian residual no larger than the
frequency equivalent of `5 cm^-1`. Set the physical rotational symmetry number
explicitly for entropy work.

For first-order-saddle Hessian validation without thermochemistry:

```text
#model=<hessian-capable molecular model>
#freq(method=mw,stationary_point=transition_state,transition_state_imaginary_threshold_cm1=50)
#device=<cpu-or-gpu>

XYZ <charge> <multiplicity> /absolute/path/to/converged_transition_state.xyz
```

The transition-state contract requires one robust imaginary vibrational mode
and positive curvature in every other vibrational direction. The threshold is
a reported numerical admission guard rather than a universal physical
definition. Inspect the mode and run IRC at the same model level before making
a reaction-connectivity claim. MAPLE does not yet expose transition-state
thermochemistry.

Historical files in this directory admitted non-mass-weighted modes or applied
unvalidated low-frequency corrections. Those inputs and their generated
outputs were removed rather than retained as apparently supported examples.

References:

- [ASE vibrational modes](https://wiki.fysik.dtu.dk/ase/ase/vibrations/modes.html)
- [ASE ideal-gas thermochemistry](https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html)
- [MAPLE stationary-point validation](../../../docs/STATIONARY_POINT_VALIDATION.md)
