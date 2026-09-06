# Hybrid ddX experimental daily workflows — 2026-09-05

This delivery opens **experimental callable** SP, first-order OPT,
mass-weighted vibrational-only FREQ, and E/F-only NEB/CINEB under
`macemdppolarhybridddx`. It does not admit production E/F/H/TS, chemical
accuracy, solution thermochemistry, or a certified transition state.
The old `macemdppolarhybrid` harmonic admission is not changed.

## Implemented boundaries

- The `1.5 kcal/mol` target does not block experimental availability. Frozen
  accuracy results, checkpoints and solvation source code were not modified.
- Neutral singlet/nonperiodic metadata is checked at evaluation and derivative
  boundaries, including metadata changes after ASE geometry-cache hits.
- FREQ has no gas thermochemistry, preserves small signed modes with a
  near-zero uncertainty label, uses one ASE/legacy unit conversion and
  mass-weighted/rank-aware rigid-mode projection.
- The cm^-1 conversion is derived from the module's frozen SI constants,
  consistent with [NIST CODATA 2018](https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2018.txt).
  Independent H2/HCl reduced-mass and nonlinear-mode tests verify it.
- Richardson Hessians retain topology/error/antisymmetry guards with at most
  six topology-preserving step halvings. Partial topology observation remains
  explicitly reported; hidden SMD surface topology is not claimed validated.
- NEB/CINEB uses one content-bound PES and one unit boundary; endpoints must
  satisfy fixed component max/RMS force limits. Failed initial optimization
  produces explicitly unconverged files, not minimum files. Nonconverged paths
  are not labeled TS candidates; even converged candidates are not certified
  first-order saddles. PRFO/NEBTS, Dimer and IRC remain closed.
- Full input-reader regressions cover the documented job headers. Parser
  globals now accept `model`, and FREQ defaults no longer overwrite an explicit
  preceding CUDA device directive.

## Fresh evidence

| check | result | evidence |
| --- | --- | --- |
| targeted regression + input/SMD/COSMO boundary suites | **266 passed**, one pre-existing lib2to3 warning | `final-targeted-tests.log` |
| real water SP | pass | `real-workflows.json` |
| real water first-order OPT | pass; maximum force `2.1750510644e-4 eV/Angstrom` under unchanged tight settings | `real-workflows.json`, `water-opt.out` |
| first real water H/FREQ | failed topology guard, retained as negative evidence | `real-workflows.json` |
| bounded adaptive real water H | pass; 3 halvings, coarse/fine `0.00025/0.000125 Angstrom` | `real-frequency-adaptive.json` |
| H Richardson error / antisymmetry | `1.2267777393e-5` / `2.6375730044e-8 eV/Angstrom^2` | same artifact |
| final-source vibrational transform/report replay | pass; six near-zero modes, three positive modes at `1626.3656`, `3798.5872`, `3866.3890 cm^-1`; no thermochemistry | `final-frequency-replay.json`, `water-final-freq.out`, `water-final-freq.sum` |
| actual NEB/CINEB algorithms on manufactured conservative potentials | pass, including a displaced triatomic path requiring 6 iterations, direct/Dispatcher unit equivalence, no H/HVP requests | `tests/test_hybrid_neb_workflow.py` in repository |

## Exact provenance and scope

The expensive real Hessian process loaded an earlier frequency module while
the final frequency conversion/report repairs were being completed. **Its
source hashes are retained unchanged.** The final frequency result therefore
uses explicitly split evidence:

1. A real checkpoint/solvation evaluation produced the guarded Hessian saved
   as `water-adaptive-hessian-eV-per-A2.npy`, SHA256
   `e046364444a6de930096f4cb865c3e8ccba0645fb59b491ffa368a0e8f07f336`.
2. `replay_final_frequency.py` feeds that recorded real matrix, with its
   geometry and diagnostics, through the frozen final frequency code.
   It is a transform/report replay, **not a second checkpoint evaluation or a
   final-tree complete end-to-end run**.

The calculator bytes at the real-H run are archived as
`calculator-at-hessian-run.py`; their SHA exactly matches the run record.
They differ from the final calculator only by the later stationary-endpoint
marker, which does not change the Hessian computation. All solvation-core
source hashes in `real-workflows.json` still matched at final verification.
`final-source-manifest.json` binds the final changed implementation and tests.

The recorded water frequencies are model predictions, not experimental
frequency accuracy measurements. No real reactive-model path, barrier
accuracy, or TS/IRC confirmation has been established by this bundle.
Accuracy/source-research work such as CPKS coverage and new heads remains a
separate follow-up; it was not used to tune this delivery.

## Review and verification limits

Independent code review found no remaining code-change findings after repair.
The formal verdict is `COMMENT` because LSP diagnostics are unavailable.
Architecture status is `WATCH` for the declared experimental evidence limits,
not a production-approval claim. See `independent-review.json`.
Source compilation, scoped Pyflakes and diff checks passed; historical
frequency/NEB static warnings were not silently reclassified as clean.
No dependency installation, checkpoint modification, commit or push occurred.

Scripts and measurements are retained for reproduction. Copy this bundle to
a **fresh output directory** before running a verifier; do not overwrite these
frozen records. Model/runtime pins are recorded in `real-workflows.json`.
