# FP32 V4 workflow execution status

This change connects the existing MAPLE task implementations to the default
pure-Torch V4 profile. FP32, lmax 5, Lebedev 194, the scalar equation, fitted
parameters, radii, and numerical tolerances are unchanged.

## Current task boundary

The 2026-11-07 task-opening evidence was frozen at `nonmd-source7`. The
2026-11-08 common-channel migration below changes framework ownership, not
task admission. This is not complete support for every MAPLE task.
Heterogeneous SP collections and the registered
constraint manifold remain pending. MD implementation is frozen until the
non-MD work is complete.

| Family | Experimental execution | Remaining boundary |
| --- | --- | --- |
| SP | Single structure and same-identity trajectory | Trajectory records retain each frame's quantities and identity. Heterogeneous `mode=collection` remains closed. |
| OPT | LBFGS, RFO, SD, CG, SDCG | A step cap is not convergence. Constrained RFO is closed. |
| SCAN | Rigid; relaxed LBFGS, SD, CG, SDCG | Relaxed RFO requires an unavailable constrained Hessian. |
| FREQ | `mw`, `cartesian-curvature`, and `both`; `nonmw` is a deprecated Cartesian-curvature alias | Physical frequencies require mass weighting. Cartesian curvature has no thermochemistry. `both` uses one Hessian. |
| Paths | NEB, CI-NEB, String, CI-String, AutoNEB | Path convergence does not certify a transition state. |
| TS | P-RFO, Dimer, NEBTS, StringTS | Numerical success requires force convergence and computed index one. Bounded failure remains non-success. Scientific certification is separate. Mass-weighted Dimer remains closed. |
| IRC | Automatic GS, HPC, EulerPC, LQA; explicit-mode exploratory runs | Automatic success requires a computed index-one seed, both force-converged branches, and computed endpoint minima. Explicit `mode_index` remains exploratory. |
| MD | Nonperiodic NVE; Langevin and v-rescale NVT | NPT, PBC, constrained MD, and restart/load-state remain closed. |

The molecular domain remains closed-shell and nonperiodic. Historical V3 and
non-MD V2 contracts remain separate. Scientific and release admission remain
false.

## Common calculator and Dispatcher channel, 2026-11-08

Both Route-2 facades now directly inherit `CalcABC`. A completed calculation
enters the real parent `calculate()` and publishes its public results through
the common `_finalize_results()` exactly once. Complete total-PES calculators
cannot also attach an additive solvent correction. Cached reads check that
invariant too. Failed publication clears partial results without committing
prospective cache state.

All existing calculator families use one Dispatcher sequence:

```text
resolve request -> preflight -> prepare thresholds -> common unit view
                -> one task executor -> reporting
```

The closed family table preserves existing admission and output policies.
Ordinary calculators, DDX V1, and Torch V3 retain their raw task returns.
DDX V2 retains its non-MD status record. Torch V4 retains its workflow outcome.
Only the internal resolver supplies the request consumed by Dispatcher.
Constructing a frozen request data object grants no execution authority.
Backend-specific Hessian and HVP calculations remain in the calculators.

The [migration evidence](evidence/calcabc-common-channel-20261108.json) records
two fresh-process CPU SP runs through the public CLI, using ordinary
MACE-POLAR and Route-2 V4 with an existing local checkpoint. Both use FP32.
Each run enters the parent, finalizer, and task executor once. Energy and force
publication match the same-call backend values exactly. This does not compare
gas and solution energies as if they were the same PES.

Final focused regression reports 666 passed, one pre-existing failure, one
skip, and one exact CUDA-availability deselection. The pre-existing failure
is the broad optional-runtime import assertion in `test_model_contracts.py`.
Its test and all scanned model files are unchanged. Pyflakes reports no issues
in the seven implementation/test files. The fixed Pyright comparison reports
197 current diagnostics against 199 baseline diagnostics, with none introduced.
Both independent code and architectural reviews passed.

All 124 solvation source files, 19 MD source files, and both frozen MD outcome
fixtures remain unchanged. No model parameter, numerical tolerance, precision
default, or scientific admission changed. No new GPU run, parameter fit, QM
calculation, or experimental-data accuracy evaluation was performed.

## Frequency quantities

`cartesian-curvature` reports all `3N` eigenvalues of the symmetrized Cartesian
Hessian in `eV/angstrom^2`, its eigenvectors, and computed-operator QA. It does
not report wavenumbers, ZPE, entropy, or Gibbs energy. Failed operator QA is a
validation failure, not task success. Unknown independent Hessian uncertainty
remains unavailable.

`both` preserves the physical MW lifecycle, including the verbosity-10 summary.
Its physical record binds the masses, Hessian identity, temperature, pressure,
low-frequency settings, mode basis, and normalization. The ideal-gas RRHO
correction on a solution PES is not a complete solution free energy.

## Continuation verification

The [source7 evidence summary](evidence/route2-nonmd-source7-20261107.json)
records these checks. All four real frequency runs use the existing small
water fixture on CPU. Each run evaluates one analytic Hessian. Same-operator
checks do not establish physical accuracy or an independent derivative-error
bound.

| Check | Result and source |
| --- | --- |
| Root focused regression | 152 passed on source7 |
| Frequency and reader compatibility | 105 passed, one inherited message-assertion failure on source7 |
| Real frequency entry points | `mw`, `cartesian-curvature`, `nonmw`, `both`: four passed on source7 |
| Frequency static comparison | 69 baseline diagnostics, 32 candidate diagnostics, zero introduced in the seven changed files |
| Preservation | 19 MD source files and 127 scientific files unchanged across the frequency change |
| Native-default Dimer | Passed in 34 accepted translations on source5. The original frozen 20-step challenge remains failed. |
| Automatic HPC and EulerPC | Both real cases passed on source3, including both branches and computed endpoint minima. |
| Trajectory SP | All three same-call records preserved exactly on source6. The original cross-call repeated-A force equality check remains failed and open. |

The observed repeated-A force difference is `1.1641532182693481e-10 eV/angstrom`.
This is not an error bound, an ULP claim, or scientific certification. Results
from source3, source5, and source6 are not relabeled as source7 reruns. A final
same-source task matrix remains pending. No additional GPU time was used in
this continuation.

## Structure

- CPU and CUDA V4 share one immutable workflow policy.
- The existing Dispatcher owns the ASE-to-Hartree unit boundary.
- `WorkflowOutcome` separates completion, task success, optimizer convergence,
  path convergence, and scientific certification.
- Final geometry comes from the algorithm result, not the original input.
  Immutable snapshots preserve all final path frames.
- MD records constant-space progress, a time-zero energy reference, and the
  last valid geometry and velocity. It does not duplicate the full trajectory
  in the status JSON.
- Dimer consumes one canonical analytic HVP/force/energy tuple. It does not
  substitute finite differences or recompute the returned energy and forces.

## Historical 2026-11-06 source-bound verification

The earlier workflow-opening phase ended at its source epoch 4. The following
receipts describe that historical source, not the current continuation.
Earlier failed attempts remain local audit records.

| Check | Final result |
| --- | --- |
| CPU real CLI matrix | 28 requested, 28 attempted, 28 expected-outcome checks passed; 16 task successes |
| CUDA real CLI sentinels | 8 requested, 8 attempted, 8 expected-outcome checks passed; 4 task successes |
| CPU/CUDA derivative replay | 4 of 4 passed |
| Named-solvent component replay | 358 of 358 passed; one water geometry, 179 parameter sets, two devices |
| CPU regression partition | 3,297 passed, 8 failed, 13 skipped, 80 deselected; 3,398 collected |
| GPU regression partition | 80 passed, 0 failed, 0 skipped, 3,318 deselected; 3,398 collected |
| Baseline failure selection | 10 selected: 8 failed, 2 passed, 0 skipped; commit `dc66007a` |
| Scientific kernel preservation | All `maple/solvation` source bytes unchanged |
| Fixed water/ammonia CPU E/F/H comparison | Maximum differences were exactly zero for each observable |

Expected-outcome checks include truthful bounded nonconvergence and uncertified
candidates. They do not mean that all tasks converged or gained scientific
certification. The component catalog is not an experimental accuracy dataset.

The CPU and GPU selectors are complementary, giving 3,377 passing tests,
8 failing tests, and 13 skipped tests across the selected suite. All eight
failing test IDs also fail in the baseline selection. The matched-QM baseline
archive lacks an ignored asset and fails earlier; a separate base/current SHA
comparison confirms that its frozen-source mismatch predates this change.
The audit does not claim that all eight traceback texts are identical.

For water MD at 8 steps of 0.1 fs, maximum NVE total-energy drift was
`3.929214e-4 eV`. Maximum v-rescale conserved-energy drift was
`2.916018e-4 eV`. Both passed the frozen `0.0104005644 eV` engineering limit.
The limit is the maximum of the preregistered absolute and relative components;
it is not a long-time MD stability guarantee. Repeating Langevin with the same
seed reproduced the recorded thermodynamic rows, final geometry, and final
velocity exactly. Langevin checks use synchronized thermodynamic quantities;
the original MAPLE carried-velocity output convention remains explicitly
distinguished.

An independent harmonic-potential check exercised the real unit adapter and
Velocity-Verlet integrator, including a central-difference force check and
time reversal. An unadapted negative control failed the unit-composition check.
These are software and numerical checks, not MLIP accuracy measurements.

The original four-case and component-catalog receipts were archived byte for
byte under `evidence/history/` before genuine current-source replays replaced
them. No manifest was edited to masquerade as a rerun. Historical V3,
original-352, and matched-QM evidence was not requalified.

## Resource and validation limits

All GPU attempts, including superseded and interrupted runs, consumed
`382.405218 s` of the authorized `600 s` wall-time budget. This is resource
accounting, not a performance benchmark. A host-capacity guard interrupted one
concurrent regression attempt after a transient low C-drive free-space reading.
Serial reruns completed. The final GPU regression used a 2 GiB Torch CUDA
allocator limit. The interrupted attempts remain in the audit record.

AST and baseline-relative Pyflakes checks supplement the tests. LSP, mypy,
pyright, and ruff were unavailable. The eight baseline failures are listed in
the aggregate evidence rather than hidden or repaired outside this task.

## Evidence and reproduction

- [Portable aggregate and complete task outcomes](evidence/route2-pure-macepolar-torch-v4-workflows-20261106.json)
- [Frozen task matrix](preregistrations/pure-mace-polar-fp32-v4-workflows.json)
- Local raw inputs, logs, resource receipts, replay scripts, and script hashes:
  `.omx/research/route2-fp32-workflows-20261106/runtime/`.

The local scripts are this-machine verification artifacts, not a new MAPLE
task framework. Runtime logs were stored on the Windows C drive and POSIX test
temporaries in RAM. No model weights or environments were copied, and no
workspace cleanup, commit, or push was performed.
