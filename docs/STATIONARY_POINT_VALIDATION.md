# Molecular stationary-point validation

MAPLE's public molecular FREQ path classifies a stationary geometry only after
the raw Cartesian Hessian has passed the ASE-unit, symmetry, mass-metric, and
rigid-subspace checks implemented in
`maple/function/dispatcher/frequency/normal_modes.py`.

## Mathematical contract

Let \(H_v\) be the Hessian restricted to the mass-weighted vibrational
subspace. The two admitted targets are deliberately exclusive:

- `stationary_point=minimum`: every eigenvalue of \(H_v\) is positive;
- `stationary_point=transition_state`: exactly one eigenvalue of \(H_v\) is
  negative and every remaining eigenvalue is positive.

The signed harmonic wavenumbers reported by MAPLE preserve that inertia. A
negative wavenumber is not converted into a positive vibration unless the user
explicitly requests the legacy numerical-noise policy
`treat_imag_as_real=true`. That policy is minimum-only and cannot be combined
with transition-state validation. Both the main report and verbosity-10
summary record the policy threshold and every raw negative mode that was
reinterpreted, so this correction is never silent.

For a transition state, the default
`transition_state_imaginary_threshold_cm1=50` requires the unique negative mode
to be at or below \(-50\ \mathrm{cm}^{-1}\). A negative mode between zero and
that guard is rejected as ambiguous. This magnitude is a conservative MAPLE
admission guard, not a universal physical definition of a transition state;
changing it changes the reported acceptance policy.

## What the result proves

A passing transition-state FREQ result proves only a stationary,
first-order-saddle **Hessian signature** for the calculator and geometry that
were evaluated. It does not prove that the imaginary eigenvector represents
the intended chemical rearrangement. Inspect the eigenvector and run an IRC at
the same calculator/model level to establish downhill connectivity in both
directions.

The current thermochemistry implementation is the ideal-gas,
rigid-rotor/harmonic-oscillator model for minima. MAPLE therefore withholds ZPE,
enthalpy, entropy, and Gibbs corrections in `transition_state` mode rather than
silently deleting the imaginary mode or presenting minimum RRHO as
transition-state theory.

## Input examples

Minimum plus RRHO:

```text
#freq(method=mw,stationary_point=minimum,temperature=298.15,pressure_kpa=101.325,symmetry_number=1,ilowfreq=0)
```

First-order-saddle signature without thermochemistry:

```text
#freq(method=mw,stationary_point=transition_state,transition_state_imaginary_threshold_cm1=50)
```

Both modes retain the existing force-stationarity, Hessian-symmetry, and rigid
invariance gates. Periodic phonons and constrained-coordinate Hessians remain
outside this molecular contract.

An opt-in real-checkpoint canary replays the retained ANI-1xnr transition-state
geometry:

```bash
MAPLE_REAL_ANI1XNR_CHECKPOINT=/absolute/path/to/ani1xnr.pt \
  pytest -q tests/test_frequency_real_ani1xnr.py
```

The checkpoint is supplied externally rather than redistributed by this test.

## References

- The ORCA 6.1 transition-state manual defines the target Hessian as having
  exactly one negative eigenvalue and recommends checking for a single
  imaginary mode after optimization:
  <https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations_TS.html>
- The ORCA 6.1 IRC manual describes IRC as the minimum-energy path from a
  transition state to its downhill-nearest intermediates:
  <https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/irc.html>
- Fukui's reaction-path formulation is the primary historical source:
  K. Fukui, *The Path of Chemical Reactions - The IRC Approach*, Accounts of
  Chemical Research **14** (1981) 363-368,
  <https://doi.org/10.1021/ar00072a001>.

These references support the first-order-saddle and IRC-connectivity
contracts. They do not prescribe MAPLE's configurable \(50\ \mathrm{cm}^{-1}\)
numerical admission guard.
