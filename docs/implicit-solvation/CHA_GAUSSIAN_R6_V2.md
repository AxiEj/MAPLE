# Gaussian CHA: R6 contact numerical profile v2

## Scope and result

The bounded **CPU float64, three-site water, programmatic MACE + MAPLE LBFGS**
campaign passed the frozen numerical gates. This is not general CHA production
admission, physical-accuracy certification, or ordinary `.inp` integration.

The panel is five fixed water starting geometries at three mandatory diagnostic
widths (`0.001`, `0.003`, `0.01` electron): 15 optimization cases, not 15 molecules.
No width was selected or calibrated, and none of the original gates was relaxed.

| Evidence | Result |
|---|---:|
| Original terminal coordinates, production orders 64/96/128 | 45/45 pass |
| Fresh legacy-profile control at the original 15 terminals | Exact reproduction; old pass counts 4/15 solvent, 2/15 combined, 2/15 both retained |
| Real MAPLE LBFGS and uncached final replay | 15/15 pass |
| Independent crossing/event checks | 150/150 pass |
| Combined energy/force consistency at starting geometries | 15/15 pass |
| Independent solvent energy/force checks at new terminals | 15/15 pass |
| Combined energy/force consistency at new terminals | 15/15 pass |

At the new terminal coordinates, the maximum solvent energy discrepancy is
`4.06884e-8 kcal/mol`, the maximum solvent force discrepancy is
`8.70279e-9 kcal/mol/Angstrom`, and the maximum combined-force discrepancy is
`5.33887e-9 Hartree/Angstrom`. These measure numerical consistency/reference
agreement, **not experimental solvation accuracy**. Combined total-energy finite
differences test derivative consistency; they are not an independent gas model.
The independent solvent reference uses NumPy/SciPy; its uncertainty estimates
remain explicitly nonrigorous. Reference FD always uses order 128, distinct
from production-order refinement.

## What changed

Gaussian smoothing removed the old weighted-charge sign jump, but the v1 R6
contact quadrature still developed artificial force changes when global
latitude breakpoints exchanged order near equal O-H distances.

V2 replaces only that contact integration representation:

- Basis-free analytic full-azimuth moments and fixed local-cosine quadrature.
- Receiver-specific direct cap complements, avoiding large full-minus-cap
  cancellation.
- Explicit no-cap, single/nested, disjoint, and full-coverage classification.
- Typed failures for unresolved geometry or conditioning, with indexed raw
  diagnostics; no numerical-force or lower-accuracy fallback.

The live R6-to-Born-to-Gaussian-polar graph, fixed charges/radii, torus terms,
Gaussian equations, cavity/dispersion, gas checkpoint, optimizer, and production
order 64 are unchanged. Forces differentiate the same complete energy scalar.
Finite differences are validation-only.

`torch_continuum_r6_contact_v2.py` owns the contact rule;
`torch_continuum_r6_derivative_v2.py` assembles the existing certified surface;
`gaussian_cha_profiles.py` selects exactly two immutable numerical profiles.
The scalar, correction, and campaign machinery share explicit profile/context
interfaces rather than duplicated implementations or rebound module globals.

## Explicit programmatic selection

Given already prepared `atoms`, immutable `topology`, and its external content
pin, select the new representation explicitly:

```python
from maple.function.calculator.extra_correction.implicit.gaussian_cha_correction import (
    GaussianChaCorrection,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles import (
    GAUSSIAN_CHA_R6_V2_PROFILE_ID,
)

correction = GaussianChaCorrection(
    atoms,
    topology,
    expected_topology_sha256=topology_pin,
    sigma_e=chosen_diagnostic_width,
    order=64,
    numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
)
gas_calculator.solvent_correction = correction
```

The default remains v1 for historical reproducibility; that profile's failed
terminal qualification is not repaired or relabeled. The frozen v1 worktree,
source archive, and 252-file negative-evidence package are preserved.

## Remaining boundaries

- Exactly-three-site certified point domain and electrostatic size below 9.5 A;
  no arbitrary-molecule or globally smooth-SES claim.
- No public selector, FREQ/TS/MD, GPU qualification, calibrated width, or
  experimental-accuracy claim.
- Campaign inputs, model weights, and raw private evidence are not distributed
  by these source additions; a fresh clone is not a self-contained campaign.
- Each of the 12 guarded processes recorded exactly one `ldconfig -p` during
  Torch bootstrap (12 total). All scientific-phase subprocesses and all
  network/OpenMM access are denied.
- Final full solvation suite: 1786 passed, 36 failed, one error, 13 skipped.
  No new failure IDs; three optional-panel tests have reduced asset coverage
  in the isolated worktree, including one historical failure now skipped.

The terminal meta-validation file has SHA-256
`d73827194d6bd5364f14072f3c7267a688acc5d69397df43157e7115f85a22f5`.
Use the complete source/evidence manifests with that receipt, not test counts
or optimizer convergence alone, when assessing the bounded result.
