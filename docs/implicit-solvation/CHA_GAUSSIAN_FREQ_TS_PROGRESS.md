# Gaussian CHA: analytic FREQ and TS-interface development

**Status: implemented and exercised; independent qualification is incomplete.**

The user-authorized scope is the existing CPU-float64, fixed-charge, three-site
water panel and programmatic interfaces. There is no supplied reaction or TS
guess. Nothing here admits ordinary `.inp` selection, arbitrary molecules,
physical accuracy, a genuine chemical transition state, IRC, MD, or GPU use.

## Implementation

- `GaussianChaAnalyticCorrection` inherits the existing energy/force evaluation
  unchanged. Dense Hessians and direct HVPs differentiate the same full scalar.
- `MACEChaAnalyticCalculator` is unregistered and binds the exact local
  MACE-OFF23-M checkpoint, CPU-native FP64 model state, and correction identity.
  It composes gas and solvent derivatives exactly once. Runtime finite
  differences are not a fallback, and HVP does not construct a dense Hessian.
  The supported experimental seams are `get_hessian()` and `get_hvp()`;
  inherited ASE `calculate(properties=['hessian'])` behavior is unchanged and
  is not claimed as supported with this correction.
- Historical Gaussian E/F, MACE core, OPT worktree and negative evidence remain
  unchanged. Runtime compatibility flags are not scientific qualification.
- The TS dispatcher now reports the actual derivative route rather than always
  describing numerical Hessians or finite-difference curvature.

## A real PRFO defect reproduced and fixed

The first real-water replay produced three positive internal modes, yet PRFO
reported `Normal Termination`: its convergence test checked only forces and
displacements. The failed development receipt is retained.

PRFO now recomputes a **fresh final Hessian** before accepting convergence and
requires exactly one material negative mode, below -10 cm^-1, in the same
mass-weighted optimization subspace selected for the search. It does not silently
enable rigid projection for laboratory-frame or periodic potentials. Nonfinite
or materially asymmetric Hessians and unsupported constrained certification
fail explicitly. A small force alone is not TS convergence.

The existing `_prfo_ts.xyz` filename can still contain an unconverged candidate;
the filename or returned `Atoms` object is not a convergence certificate.

## Actual development replay

Five frozen optimized water geometries at each of three mandatory diagnostic
widths give **15 cases, not 15 molecules**. All 15 completed:

1. Existing mass-weighted FREQ workflow, with six projected rigid modes and three
   positive internal modes.
2. One PRFO iteration through the analytic composed-Hessian interface, correctly
   not reporting the water minimum as a converged TS.
3. One Dimer iteration using direct analytic HVP, not reporting a converged TS.

The replay poisoned runtime numerical-Hessian and force-pair-FD entry points;
the Dimer phase also poisoned dense-Hessian entry points. Raw outputs and the
exact source snapshot are retained in the local experiment evidence directory.

### Width sensitivity is scientifically important

The internal frequency ranges over the five starting geometries are:

| sigma (electron) | Lowest mode (cm^-1) | Middle mode (cm^-1) | Highest mode (cm^-1) |
|---:|---:|---:|---:|
| 0.001 | 2780.8-2781.7 | 10226.0-10251.8 | 18702.0-18751.4 |
| 0.003 | 2696.0-2697.5 | 7064.8-7071.0 | 12546.4-12559.7 |
| 0.010 | 2265.9-2266.5 | 4726.2-4726.8 | 6768.2-6771.9 |

These are model diagnostics, **not experimental frequency accuracy**. The strong
width dependence must not be hidden by selecting a preferred width after seeing
the results. No width, charge, radius, checkpoint, quadrature order, or accuracy
tolerance was fitted or relaxed.

The widths have different optimized geometries. A separate three-case
same-geometry decomposition gives maximum gas-only projected frequencies of
3803.6, 3867.7 and 3976.4 cm^-1, versus composed maxima of 18733.4, 12551.8 and
6771.3 cm^-1. Gas and solvent Hessians sum to the composed Hessian to within
1e-12 Hartree/Angstrom^2. This points to strong added solvent curvature; the
separate component spectra are not stationary gas/solvent normal-mode results.

## Remaining verification

Fresh regression coverage was run in two disjoint processes, with the combined
case inventory checked against the full suite: **1874 passed, 38 failed, one
error, 13 skipped**. All 81 added cases pass. Relative to the baseline, the only
two additional failing cases are the historical source-binding checks below;
the original 36 failures and one error remain. New Python files are Black-clean;
changed Python files pass Pyflakes. Pyright reports the same 69 pre-existing
PRFO diagnostics, with an identical rule/message inventory and none in the new
files. This is deliberately not described as an all-green suite.

- The complete independent Hessian/directional-curvature campaign and resource
  qualification have not run. The 15 workflow replays do not replace them.
- Independent review was interrupted by native subagent usage limits. The
  leader's subsequent PRFO, counter-tracing, and receipt-validation changes
  still require independent review; this is not a merge-ready verdict.
- The PRFO and TS-log fixes invalidate two historical **current-source** binding
  checks. Their old audit hashes have not been rewritten or their tests skipped.
  Fresh versioned revalidation is required before historical qualification can
  be attached to the changed source.
- New numerical-validation machinery is development work, not a qualification
  receipt. Its source/input bindings, strict option schemas, actual graph-call
  counting, and resource checks must be reviewed before a final campaign.
- The historical full suite has known failures and missing-asset skips. Report
  fresh counts and exact failure-ID comparisons, not a blanket green-suite claim.

No commit of this development stage or new push is implied by this document.
