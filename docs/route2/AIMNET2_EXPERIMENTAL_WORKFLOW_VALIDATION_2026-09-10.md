# AIMNet2 experimental workflow validation — 2026-09-10

## Result and boundary

SP, OPT/LBFGS, FREQ/MW and TS/PRFO all completed the actual public input ->
factory -> dispatcher path using the same frozen local checkpoint, CPU,
water solvent, and the independently preregistered numerical-Hessian v2.
These are **experimental workflow canaries**, not population accuracy,
solvent-generalization, production admission, or formal E/F/H/V/M evidence.
The scalar and profile admission registries remain unchanged and disabled.

| Task | Real canary | Outcome | Total E/F calls | Hessian calls |
| --- | --- | --- | ---: | ---: |
| SP | Water | Direct-factory E/F exactly identical | 1 | 0 |
| OPT | Water | 8 iterations; final max force 6.78575852e-07 eV/A; RMS 3.70333767e-07 eV/A | 10 | 0 |
| FREQ | Optimized water | Minimum; three positive vibrational modes | 75 | 1 |
| TS | Fixed planar NH3 starting geometry | 5 PRFO iterations; final first-order saddle | 203 | 2 |

- Water vibrations (cm-1): 1596.901432, 3805.483090, 3961.617353.
- NH3 saddle vibrations (cm-1): -1064.791359, 1376.349294, 1376.349294, 3605.175632, 3726.929582, 3726.929582.
- Final TS max force: 2.37807817e-07 eV/A; RMS: 1.1890377e-07 eV/A.
- Rigid-body residual equivalents: water 0.410138 cm-1; TS 0.253978 cm-1, both below the unchanged 5 cm-1 limit.
- The final spectra and stationary-point indices were independently recomputed
  from the saved returned Hessians, not accepted solely from cached status flags.
- Real negative CLI checks: water OPT with max_iter=1 exited 1 after 3 E/F calls;
  FREQ on unoptimized water exited 1 before any Hessian. Neither was labeled success.

## Numerical-method lineage

V1's fixed planar-NH3 initial Hessian remains a failed result:
Richardson estimate 1.18586649e-4 eV/A2 > 5e-5. Its full raw vectors/matrices,
identity, and failure status were retained. V2 recomputed a full independent
four-step ladder, used fourth-order Richardson cancellation, and passed its
prospectively fixed ratio/error/antisymmetry/replay/work gates. No model,
checkpoint, radius, charge scaling, solvent parameter, or accuracy threshold
was fitted or changed. See the [protocol and usage guide](AIMNET2_EXPERIMENTAL_WORKFLOWS.md).

## Engineering verification

- Final post-review regression: **2358 passed, 15 skipped, 19 warnings**.
  The skipped tests are recorded as skipped, not as real-checkpoint passes.
- Four new production modules: cached Pyright **0 errors, 0 warnings**.
- New Python files: Black passes; all changed Python files: Pyflakes and
  compilation pass; git diff --check passes.
- Independent source review found no remaining critical/high defect.
- Known gaps: pre-existing whole-file formatting/type debt remains in legacy
  modules; dynamic test modules are not Pyright-clean. This is not a claim of
  whole-repository type cleanliness. The V2 task-summary refinement field is
  null; its linked Hessian sidecar retains the full richardson_refinement data.
- The hash-bound local checkpoint's upstream release-origin provenance remains
  unresolved; no upstream-origin or broad scientific certification is asserted.

## Provenance and changed files

Validation base Git HEAD: `278191b400856ee36dfa78a2da2eca7f34d43589`.
Validation ran against the local diff before publication. The final real-task
source snapshot and every input/checkpoint hash are retained below. All
recorded source hashes matched the validated source after the runs; publication
does not change the experimental/scientific claim boundary.

Private development evidence (ignored by Git):
- `.omx/validation/aimnet2-experimental-20260910/cli-v2-final/manifest.json`
- `.omx/validation/aimnet2-experimental-20260910/cli-v2-final/source_snapshot/`
- `.omx/validation/aimnet2-experimental-20260910/cli-v2-final/*.hessian-*.json`
- `.omx/validation/aimnet2-experimental-20260910/cli-v1/ts.out.hessian-0001.json`
- `.omx/validation/aimnet2-experimental-20260910/expanded-regression-final-rms.log`
- `.omx/validation/aimnet2-experimental-20260910/completion-evidence.json`

Changed source files (all SHA-bound in the final private manifest):
- `maple/function/aimnet2_experimental.py`
- `maple/function/calculator/set_calculator.py`
- `maple/function/dispatcher/aimnet2_experimental.py`
- `maple/function/dispatcher/dispatcher.py`
- `maple/function/read/command_control.py`
- `maple/function/read/input_reader.py`
- `maple/solvation/coupling/aimnet2_experimental_ase.py`
- `maple/solvation/coupling/aimnet2_experimental_ase_v2.py`

New regression files cover v1/v2 numerical Hessians, parser/factory routing,
workflow status and raw failure retention. README, this report, the workflow
guide, and four example inputs provide the usage surface. Reuse was limited
to existing scalar construction, topology guards, legacy unit conversion,
LBFGS/PRFO, and the shared mass-weighted frequency kernel; no new dependency
or alternate physical model was added.
