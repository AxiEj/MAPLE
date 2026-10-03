# Gaussian-CHA research checkpoint — 2026-10-03

This branch is a **research checkpoint**, not a release or a merge-ready claim.
Publishing the source does not enable a public calculator or `.inp` option.

## What this checkpoint contains

- The preserved Gaussian-CHA v1 terminal-force audit and bounded R6-v2 repair.
- Opt-in analytic Hessian/direct-HVP facades and FREQ/PRFO/Dimer development
  interfaces. PRFO requires a fresh final index-one curvature check before
  reporting saddle convergence.
- A separately validated exact single-cover SAS dispersion kernel. It uses a
  live domain certificate and the stable all-inner/upper-cap crossing formula.
  It has **not** been integrated into the complete Gaussian-CHA profile.

The standalone kernel passed two fresh, byte-identical formal executions:
15 fixed water cases, 25 synthetic anchors, 15 certificate fixtures and one
additional stability regression. These are numerical consistency checks,
not experimental frequency accuracy or a real chemical transition state.
Runtime derivatives remain analytic/autograd; finite differences are validation
only. Unsupported single-cover geometries are rejected, without fallback.

## Known unresolved work

The existing composed v2 second-derivative campaign remains **15/15 unresolved**.
All rows fail its dispersion rotation checks; the narrow-width rows also fail
its original force-difference convergence checks. Those failures were preserved,
not converted to passes. The new standalone kernel does not retroactively
qualify that composed campaign.

The next stages are a separately reviewed profile integration, fresh composed
E/F and all 15 OPT revalidations, then FREQ/PRFO/Dimer revalidation. No general
molecule, true-TS, physical-accuracy, GPU/MD/IRC or public-input admission is made.

Historical full-suite failures remain documented in
`CHA_GAUSSIAN_FREQ_TS_PROGRESS.md`; no historical source-binding receipt was
rewritten to make the suite green. That document is a preserved, earlier
checkpoint, including its then-pending review status. The newer standalone
result is described in `CHA_SINGLE_SAS_DISPERSION_V3_PROGRESS.md`. Statements
about no commit/push in those frozen reports describe their pre-publication
snapshots, not the status of this publication checkpoint.

## Fresh pre-publication checks

A broad targeted run from the isolated v3 worktree returned **152 passed,
5 failed, 14 setup errors, 1 deselected**. Most failures/errors are inherited
MACE tests whose exact local checkpoint is not installed in this worktree.
The copied v2 FREQ/TS runner also remains bound to its original private
`cha-freq-ts-v1-20261002` worktree, so its current-source test rejects the v3
location. These limitations are recorded rather than repaired during a push
or hidden by changing source identities. The deselected live v2 derivative
qualification assertion is already covered by the retained 15/15 unresolved
campaign, not replaced by a passing result.

The standalone kernel/legacy regression set was rerun in v3: **35 passed**.
The older facade/interface set was rerun in its original pinned FREQ worktree:
**136 passed, 1 known numerical qualification assertion deselected**. The
independent standalone-validator adversarial suite was rerun: **44 passed**.
These scoped results do not waive the v3 relocation failures or establish an
all-green full suite.

## Reproducibility boundaries

The source and regression tests, including frozen water regression fixtures,
are committed. Model weights, `.omx` runtime state, complete private campaign
directories and scratch files are deliberately not committed.
Some composed-workflow tests need the original local pinned inputs and paths.
They do not download inputs. The live v2 derivative qualification assertion is
expected to fail where those inputs are installed; it is not a CI-green claim.

The standalone source tests can run without a model checkpoint:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/solvation/test_torch_continuum_dispersion_domain.py \
  tests/solvation/test_torch_single_sas_dispersion_v3.py
```

Local source/evidence archives retain the complete earlier snapshots. Their
readback verified 1,521 historical evidence members and all 191 candidate files
without drift. Reference hashes:

- Standalone final summary: `f5d85c8c751079f34e0f77b4497ef187874f65f9e0a63b9eeae671580f0b5dd6`
- Final evidence manifest: `67e6aa47466d65c3caf3cc6d1c729067c83e261ff680384989194f1a0f32f32d`
- Pre-publication source archive: `af6586d0581a01192c2941966fd3f3c38e71c29f9f40a7558149e67a093728b5`
