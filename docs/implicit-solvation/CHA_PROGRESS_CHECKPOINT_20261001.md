# CHA progress checkpoint — 2026-10-01

**Unfinished research source checkpoint, not an enabled or qualified CHA OPT
release.** This feature-branch snapshot preserves the current implementation
before further runner repairs and real optimization validation.

## Included work

- The unregistered continuous R6/CHA foundation, independent reference, and
  bounded three-site coordinate-precondition study. These retain the original
  hard-sign model and its unresolved sign-event evidence.
- The separately identified `chagb-r6-pbsa-gaussian-sign-v1` scalar, analytic
  Torch forces, explicit programmatic correction, and independent NumPy/SciPy
  reference. Gaussian widths are diagnostic, not calibrated defaults.
- A **work-in-progress** preregistered MACE/LBFGS validation runner and tests.
  Its planned matrix contains 15 starts: five water geometries at three
  Gaussian widths. Unit tests are not evidence that these optimizations ran.

The scalar/correction and reference passed their bounded code reviews. The
runner still has a **REQUEST CHANGES** verdict, with these known issues:

1. `run()` returns phase summaries without a top-level status; the CLI defaults
   that missing status to `VALIDATED`, allowing a failed phase to exit zero.
   Phase success must be distinguished from full-campaign validation.
2. A fresh combined-force finite-difference phase does not create the log's
   parent directory before calculator construction, so its first row can fail.
3. Crossing receipts retain reference energies at orders 64/96/128 but omit
   their quadrature/error diagnostics and integration tolerances. Those must
   be retained and independently revalidated.

Do not rely on this runner's exit code or summary as a qualification receipt
until these issues are repaired and regression-tested. No real Gaussian
crossing scan, combined-force finite-difference campaign, or 15-start OPT
campaign has been completed at this checkpoint. A separate cached-model gas
preflight is only a prerequisite, not solvent or optimization validation.

## Checkpoint verification

- Fresh targeted run of all 17 added test files: **222 passed**, with two
  expected Torch anomaly-detection warnings.
- All 10 added Gaussian Python files: Black check, Pyflakes, Pyright, and
  bytecode compilation passed (Pyright: zero diagnostics).
- The 24 previously frozen foundation/precondition files and historical CHA
  polar module were checked against their preserved content hashes: unchanged.

These are scoped checks, not a clean full-suite claim. The earlier full
solvation-suite run retained 37 failures and one error matching the baseline;
it was not rerun for this push. Eight previously recorded type diagnostics in
restored older tests also remain outside the clean Gaussian-only typecheck.

## Source and capability boundaries

This commit contains source, tests, documentation, and a protocol only. It
does not include private source-derived GPL/LGPL exact-SP implementations,
vendor source/binaries, model weights, prepared private inputs, or raw local
evidence under `.omx`. Publishing this research source checkpoint is not an
assertion of scientific qualification or permission to redistribute those
separate private artifacts. Existing exact-SP and continuum identities remain
distinct; the historical `torch_chagb.py` is unchanged.

The protocol deliberately pins local inputs, source manifests, and cached
weights that are **not shipped here**. The campaign is therefore not a
self-contained fresh-clone reproduction package. It must reject missing or
mismatched assets rather than download, recharge, substitute, or relax pins.

No public provider, ordinary `.inp` selector, optimizer implementation, or
capability registry is changed. General molecular support, globally smooth
surfaces, physical-accuracy claims, FREQ/TS/MD, and production admission remain
outside the validated scope. Next work is runner repair, followed by the
frozen real-execution and independent energy/force validation protocol.
