# Route 2 provenance contract

Every vNext scientific artifact must bind the computation to its exact source,
model, continuum, cavity, and runtime. A boolean `passed` field without the raw
measurements and identities below is not release evidence.

Required fields:

- Git HEAD, tree, dirty state, source-file SHA256 map, and exact command;
- scalar ID, state-equation ID, profile ID, and admitted capability tier;
- checkpoint path role and SHA256, upstream model version/commit, inference
  implementation digest, optimizer-parameter-group audit status;
- continuum backend/version/build/library SHA256 and cavity/profile digest;
- Python and package versions, compiler, BLAS, hardware, thread settings,
  device, dtype, and seed;
- primal/adjoint residuals, root identity, topology/ownership identity;
- raw gate measurements, pass/fail result, warnings, and output-file hashes.

The Phase 0 evidence bundle under `evidence/baseline-15777aad/` demonstrates
the storage format and records negative evidence as well as passing tests. It
is a baseline artifact, not a Route 2 release certificate.

`tools/route2_release/run_fixedbox590_water_pes_diagnostic.py` is the first
vNext real-PES evidence capture command.  It requires a clean checkout, hashes
the committed bytes of all loaded repository Python sources, binds the official
checkpoint and provider identities, writes outside the checkout, and checks
that HEAD/tree/status remain unchanged before and after capture.  Its output is
explicitly diagnostic and cannot grant a registry capability.

## CI and branch protection

`route2-core.yml` is the required lightweight job. It runs without scientific
native runtimes and verifies the baseline evidence/compile boundary. As the new
package grows, its dependency-free algebra and contract tests belong in this
job.

`route2-real-stack.yml` is a manually triggered, self-hosted job. A runner that
claims the `route2-real-stack` label must provide the declared Torch/MACE,
PySCF/pyddx, PCMSolver, MOIST, and checkpoint assets. The workflow validates
those prerequisites before tests; missing dependencies fail rather than turn
into a green skip.

Repository workflows cannot enable GitHub branch protection themselves. A
maintainer must require `Route 2 core / core` on the vNext branch. Real-stack
jobs become required only when a stable, access-controlled runner is available;
the release manifest must still show that the corresponding job executed.
