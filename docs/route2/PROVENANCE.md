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

`tools/route2_release/run_fixedbox590_water_path_diagnostic.py` applies the
same fail-closed source/checkpoint/runtime binding to the preregistered
distorted-water and bidirectional-loop panel. It records every geometry,
energy, force, source/root, topology hash, residual, warning, and raw
finite-difference/line-integral measurement. Its output also remains external
to the checkout and capability-neutral.

## CI and branch protection

`route2-core.yml` is the required lightweight job. It installs only the pinned
ASE/NumPy/SciPy/pytest environment, explicitly fails if Torch, MACE, PySCF,
pyddx, AIMNet, or FAIR-Chem is unexpectedly importable, and verifies the
baseline evidence/compile boundary plus all vNext dependency-light tests. It
uploads the raw pytest log, sorted `pip freeze --all`, runtime manifest, and
their checksums for every run, including failures. As the new package grows,
its dependency-free algebra and contract tests belong in this job.

`route2-pyddx.yml` is the public optional-runtime job. It creates a fresh venv
from the selected Python 3.11 runtime and installs the exact PySCF
2.13.1/pyddx 0.8.0 stack from
`requirements/route2-pyddx-ci-py311.txt` while proving that unrelated
Torch/MACE runtimes remain absent. Its 29 version-locked canaries cover the
real PySCF solvent/cavity paths, real pyddx energy and coordinate derivatives,
thread equivalence, and the vNext radial/pair-frame continuum contracts. The
job parses its JUnit report and fails unless all 29 tests execute with zero
failures, errors, or skips; missing optional dependencies therefore cannot
produce a green check. It retains the JUnit and pytest logs, exact runtime,
sorted package inventory, and checksums as a source-bound artifact.

`route2-real-stack.yml` is a manually triggered, self-hosted job. A runner that
claims the `route2-real-stack` label must provide the declared Torch/MACE,
PySCF/pyddx, PCMSolver, MOIST, and checkpoint assets. The workflow validates
those prerequisites before tests; missing dependencies fail rather than turn
into a green skip.

Repository workflows cannot enable GitHub branch protection themselves. A
maintainer must require `Route 2 core / core` and
`Route 2 PySCF and pyddx / optional-runtime` on the vNext branch. Real-stack
jobs become required only when a stable, access-controlled runner is available;
the release manifest must still show that the corresponding job executed.
