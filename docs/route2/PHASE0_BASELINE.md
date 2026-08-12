# Route 2 conservative-vNext: Phase 0 baseline

This document freezes the starting point for the conservative Route 2 rebuild.
It is an inventory and reproducibility record, not evidence that the legacy
implementation satisfies the vNext conservative-PES contract.

## Source identity

| field | value |
| --- | --- |
| repository | `AxiEj/MAPLE` |
| audited baseline | `15777aadf92e8a14419e5ff5d8ac6b3cc17aa482` |
| audited tree | `e97904ce23cce34a65353d56419f076fa238a1e9` |
| working branch | `refactor/route2-conservative-vnext` |
| baseline worktree | clean before and after the test run |

The branch was created directly from the audited baseline. The dirty
`implicitsolv-route2` worktree was not reset, copied, staged, or modified.

## Executed baseline

All commands below were executed in the baseline worktree on 2026-08-13
(Asia/Singapore):

```bash
test "$(git rev-parse HEAD)" = \
  15777aadf92e8a14419e5ff5d8ac6b3cc17aa482
test -z "$(git status --porcelain=v1)"
python -VV
python -m pip check
python -m pip freeze --all | sort
python -m pytest --collect-only -q
python -m pytest -q -rs
test -z "$(git status --porcelain=v1)"
git diff --check
```

Results:

| check | result |
| --- | --- |
| collection | `1445 tests collected in 2.10s` |
| full pytest | `1430 passed, 15 skipped in 50.09s` |
| PySCF-dependent skips | 6 |
| pyddx-dependent skips | 4 |
| MOIST-dependent skips | 5 |
| `git diff --check` | pass |
| `pip check` | **fail**, shared-host conflicts recorded verbatim |

`pip check` is deliberately reported as a failure. The shared development
environment contains unrelated packages with incompatible requirements; the
passing pytest result must not be described as a clean locked-environment
result. See `evidence/baseline-15777aad/pip-check.log`.

A second probe installed only MAPLE's declared core dependencies into a fresh
virtual environment. Full collection then failed with 128 import errors because
legacy package initializers and tests import Torch unconditionally. That probe
is a dependency-boundary/CI defect, not an optional-dependency test pass.

## Runtime scope

The executed full suite had Python 3.11.14, ASE 3.27.0, NumPy 2.4.6, SciPy
1.17.1, pytest 9.0.2, Torch 2.12.0+cu130, and mace-torch 0.3.16. PySCF,
pyddx, PCMSolver, and MOIST were not available. Exact package and platform
records are stored in `evidence/baseline-15777aad/`.

Tests that skip because a scientific runtime is absent are not evidence for
that runtime. In particular, this baseline does not prove:

- a PCMSolver or ddX real-stack calculation;
- a MOIST/\(\rho\)-DROP coordinate derivative;
- real-checkpoint conservative-force admission;
- chemical accuracy, OPT, NEB, TS, FREQ, or MD readiness.

## Existing Route 2 inventory

The machine-readable inventory is
`evidence/baseline-15777aad/inventory.json`. At the audited baseline it records:

- 102 production files selected by the Route 2/source/receiver/GTO inventory;
- 150 tests whose filename contains `route2`;
- the checked-in Route 2 benchmark/evidence assets and their SHA256 digests;
- the 2,605-line legacy `route2_engine.py` orchestration module.

The inventory is intentionally broader than the future production package. It
preserves scientific history so migration can reuse valid components rather
than rewrite or silently lose them.

## Baseline capability disposition

The legacy profile registry contains 21 public profile specifications:

- 20 declare `experimental-energy-only`;
- one declares a bounded experimental energy-and-force capability;
- none establish a strict common variational electronic functional.

These are legacy declarations. **No legacy profile is automatically admitted
to a vNext E/F/H/V/M tier.** The vNext matrix therefore starts with every tier
false and requires new same-scalar evidence. The full snapshot is in
`evidence/baseline-15777aad/capability-matrix.json` and summarized in
`CAPABILITY_MATRIX.md`.

## Phase 0 conclusion

The source baseline and existing unit tests are reproducible in the captured
host environment. The environment itself is not dependency-clean, optional
real stacks are missing, public ASE units need correction, non-solvation job
coverage is narrow, and no vNext profile is admitted. Those negative facts are
part of the frozen baseline and may not be rewritten by later passing tests.
