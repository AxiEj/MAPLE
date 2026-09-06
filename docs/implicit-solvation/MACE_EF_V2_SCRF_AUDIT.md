# Centered-V V2: full conductor SCRF and local-response audit

## Outcome

The corrected-input candidate converges for both fixed water and methanol
molecular geometries. Multiple prescribed initial sources reach the same
root within the frozen tolerances. The supported joint minimum is locally
positive and the fixed-point residual is not ill-conditioned in the declared
metric.

**This is not a physical passivity certificate.** Methanol has reproducible
positive electronic-response modes at its converged root. The largest was
independently confirmed through an actual continuum-generated perturbation.
It lies below the inherited numerical threshold `0.01`, demonstrating why a
numeric `pass` flag must not be read as proof of nonpositive physical curvature.
No threshold was changed and no mode was clipped.

## Inputs and implementation boundary

- Same candidate SHA-256:
  `0cac472aa566363a00f5f3fbc41e7fb8bc5e1f5bb8aa8cba654ebac3fe4dc902`.
- A distinct immutable checkpoint spec is passed directly to the existing
  evaluator. It is **not** installed in the global registry or public profiles.
- The qualified parent receipt is checked for program SHA, tensor-state digest,
  source-only member change and parent identity before model construction.
- Primary roots use the unchanged `MACEPolarEFSegmentCOSMOCoupling.evaluate`.
  The separate seed harness reuses the stock safeguarded-Anderson step and
  validators; it does not pretend an initial guess is an energy gradient.
- Mixing `0.5`, maximum 100 iterations, raw source tolerance `5e-5`, and
  charge tolerance `2e-5 e` are fixed before execution.
- The existing conductor cavity/radii/grid configurations are matched exactly.
  Coordinates follow the stock core's float32-roundtrip canonicalization and
  are recorded. These are **two solute molecules under conductor COSMO**, not
  water-solvent versus methanol-solvent dielectric calculations.
- No model parameters, candidate program, cavity, solver settings, old
  artifacts or production source files were modified.

[Base preregistration](benchmarks/mace-ef-v2-scrf-prereg-v1.json),
[driver](benchmarks/audit_mace_ef_v2_scrf.py),
[raw result](benchmarks/mace-ef-v2-scrf-water-methanol-v1.json).

## Convergence and energy ledger

| Quantity | Water molecule | Methanol molecule |
| --- | ---: | ---: |
| Stock-core iterations | 26 | 25 |
| Seed-harness iteration range | 24–28 | 21–27 |
| Fresh stock source residual | 4.64e-5 | 4.63e-5 |
| Maximum seed/root source difference | 1.31e-5 | 1.12e-5 |
| Maximum seed/root total-energy difference, eV | 7.25e-5 | 4.34e-5 |
| Stock half-coupling identity error, eV | 4.44e-16 | 1.11e-16 |
| Stock field replay difference | 0 | 0 |

Seeds are the gas energy-conjugate source, 0.5 and 1.5 times that source,
and the gas source plus a prescribed neutral-tangent perturbation. All four
converge; this probes the selected basin, not global uniqueness. The existing
Anderson safeguards frequently fall back to mixed Picard because of their
charge-drift/coefficient checks. Those decisions are logged and were not altered.

The ledger is the existing common scalar

\[
\mathcal L=E_\theta(R,f)+U_{\rm COSMO}(R,c)-\langle c,f\rangle.
\]

The source is the full energy derivative, not the auxiliary density head.
Both fresh source residuals and mapped-source field residuals are recorded.
These conductor quantities are not complete hydration/COSMO-RS free energies.

## Local stability versus electronic passivity

The fixed metric uses charge scale `1 e`, dipole scale `1 e Angstrom`, and
energy scale `1 eV`. The existing fixed-charge Helmert basis removes only the
constant-charge tangent/gauge direction. Every raw model state is separately
checked for charge closure; no physical output source is projected or repaired.

For dimensionless reduced continuum K and electronic response H, the audit
records raw `J=H K`, an independent finite-difference map Jacobian, and

\[
T=-K_s,\qquad C=I+T^{1/2}H_sT^{1/2}
\]

on the positive continuum-response support. This tests a reduced local minimum;
the unreduced common scalar is a saddle and is not required to be positive
definite. Raw matrices/antisymmetry are saved before interpretation, including
partial matrices if a diagnostic fails.

| Finest-step diagnostic | Water | Methanol |
| --- | ---: | ---: |
| Continuum support dimension / neutral dimension | 11 / 11 | 23 / 23 |
| Minimum normalized joint curvature | 0.5019 | 0.5769 |
| Unmixed feedback spectral radius | 0.4981 | 0.4231 |
| Smallest singular value of I−J | 0.4907 | 0.5407 |
| Residual 2-norm condition number | 2.17 | 2.00 |

Three frozen field/source step sizes were used. Both cases pass the registered
matrix reciprocity, step-resolution, chain-rule comparison, root-gauge and
positive-joint-curvature checks. These results do not establish response
accuracy or stability over other geometries or field domains.

## Confirmed remaining methanol charge-response mode

The pilot showed four positive eigenvalues in methanol's full neutral-input
electronic Hessian. A separate
[confirmation preregistration](benchmarks/mace-ef-v2-scrf-mode-confirmation-prereg-v1.json)
was written after that observation, without changing the base gates or model.
For each molecule the largest symmetric electronic mode on the continuum
support is selected. Its source preimage is solved for, then the **actual**
continuum is applied to the positive/negative source perturbations.

For methanol:

| Dimensionless step | Measured electronic directional curvature |
| --- | ---: |
| 0.002 | +0.008158 |
| 0.001 | +0.008134 |
| 0.0005 | +0.008142 |

The field-mode realization error is about `6.5e-14`; continuum linearity errors
are below `1e-15`. Repeated center derivatives differ by about `9.4e-9`, far
smaller than the observed directional derivative separation. All participating
electronic states and derivatives are retained.

In the preregistered metric, about 99.9998% of the direction's squared norm is
in the **scalar-potential coordinates**, not the field-gradient coordinates.
This localizes the remaining issue to a potential/charge-response direction;
it does not by itself identify which energy-network term causes it. The water
control remains negative along its selected mode.

Thus the old `0.01` numerical gate is passed, but methanol is not strictly
concave along this tested physical direction. Neither stable SCRF nor a
well-conditioned root removes that distinction. The result does not prove
that a chemical error target is impossible; no FreeSolv labels were used here.

## Reproduction and next boundary

Use the existing CUDA project environment and a new output path:

```bash
PYTHONNOUSERSITE=1 python docs/implicit-solvation/benchmarks/audit_mace_ef_v2_scrf.py --checkpoint .omx/artifacts/mace-ef-v2-trace-inputs-20260905/final/centered_v.pt --input-audit docs/implicit-solvation/benchmarks/mace-ef-v2-trace-input-ablation-v1.json --preregistration docs/implicit-solvation/benchmarks/mace-ef-v2-scrf-prereg-v1.json --confirmation docs/implicit-solvation/benchmarks/mace-ef-v2-scrf-mode-confirmation-prereg-v1.json --output /tmp/mace-ef-v2-scrf-replay.json
```

This working-tree diagnostic is source-hash-bound, not a clean admission replay.
The full FreeSolv-20/COSMO-RS paired panel has not been rerun. Public/scientific
admission remains closed; no force or PES claim is added. A further energy-term
analysis of the potential-dominated positive mode is warranted before calling
the electronic response repaired. Do not change mixing, discard that mode,
or reinterpret the inherited tolerance to hide it.

Final targeted verification: **143 passed**, including 40 SCRF-specific CPU
and frozen-evidence tests plus the prior CUDA/model/COSMO suites. Black,
Pyflakes, Python compilation, CLI help smoke and `git diff --check` passed.
Dedicated LSP/type checking was unavailable. Reviewer findings on the parent
program identity and failure-evidence preservation were fixed and covered by
regression tests before the final measured bytes were frozen. Earlier dirty
files, old artifacts, the candidate checkpoint and branch HEAD were verified
unchanged. No commit or push had been performed at measurement time; later
publication does not alter that execution provenance or admission scope.
