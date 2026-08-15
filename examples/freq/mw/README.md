# Molecular FREQ template

The public FREQ workflow accepts only a stationary, unconstrained,
non-periodic molecular **minimum** and a calculator that implements a public
ASE-unit Cartesian Hessian (`eV/angstrom^2`). Run geometry optimization first,
then use its converged structure in a separate input:

```text
#model=<hessian-capable molecular model>
#freq(method=mw,temperature=298.15,pressure_kpa=101.325,symmetry_number=1,ilowfreq=0)
#device=<cpu-or-gpu>

XYZ <charge> <multiplicity> /absolute/path/to/converged_minimum.xyz
```

The default gates require a maximum force component no larger than
`1e-3 eV/angstrom`, a raw Hessian symmetry defect below the declared relative
tolerance `1e-6`, and a raw rigid-subspace Hessian residual no larger than the
frequency equivalent of `5 cm^-1`. Set the physical rotational symmetry number
explicitly for entropy work.

Historical files in this directory used a transition-state geometry, admitted
non-mass-weighted modes, or applied unvalidated low-frequency corrections.
Those inputs and their generated outputs were removed rather than retained as
apparently supported examples. Transition-state mode validation and
transition-state thermochemistry require separate task contracts.

References:

- [ASE vibrational modes](https://wiki.fysik.dtu.dk/ase/ase/vibrations/modes.html)
- [ASE ideal-gas thermochemistry](https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html)
