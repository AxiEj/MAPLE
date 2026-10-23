# Retire continuous CHA research; retain exact SP and OBC-II

## Scope

The active goals are faithful independent CHA single-point energy and independent
OBC-II composition across MAPLE tasks. This cleanup does not implement new task
support, activate the independent OBC-II candidate, or change a scientific gate.
Existing OpenMM, AmberTools, APBS and ddLPB runtime behavior remains unchanged.

Six unregistered continuous-CHA prototype modules are removed from the active
package: `continuum_chagb_inputs`, `torch_continuum_sav`,
`torch_continuum_dispersion`, `torch_sphere_union_geometry`,
`sphere_union_volume`, and `sphere_union_dispersion`. Their six exclusive test
modules and the continuous-nonpolar probe plus its exclusive test are removed
with them. There are no compatibility stubs, fallback paths or new dependencies.

The exact-SP dependencies `torch_chagb.py` and `torch_pbsa_exact_cavity.py`, shared
resource budgets, OBC-II implementation and all task/provider code are retained.
The private prepared-input exact-SP implementation stays private; this cleanup
does not grant distribution permission or expose a new public provider.

## Evidence identity

Historical JSON reports and their numerical results are not rewritten. The
2026-09-27 source ledger remains an archival audit; its three explicitly retired
source entries remain recorded, while tests require those files to be absent.
All retained ledger entries still require their exact live source hashes.

`route1-retained-solvent-revalidation-2026-10-23.json` is a separately executed,
source-bound receipt. It supersedes the current-source binding of the immutable
2026-09-27 receipt and retains the original 2026-09-22 validation lineage. It
replays the same 48 native OBC-II energy/force cases and the retained regressions,
including actual engine analytic FREQ/PRFO/Dimer paths. It does not establish
experimental accuracy, CHA derivatives or all-model OBC-II task qualification.

The producing runner snapshots its consumer test but does not execute that test
inside the producing run. The receipt itself is not part of its own source hash
set. The consumer checks the completed receipt and current source afterward.

## Research preservation

The already-published `feature/route1-cha-dispersion-v3-20261003` checkpoint
`76de6c48` remains preserved. Its ancestors make the older local CHA branch
pointers redundant. Four retired local worktrees and four local branch pointers
are removed only after complete file/symlink archive verification. Remote
branches are not deleted.

Local recovery evidence is under
`.omx/checkpoints/route1-cleanup-20261023/archives/RESTORE.md`. It includes private
and ignored files, not just Git-tracked code. This directory is not shipped with
the package; a Git push alone is not a backup of private evidence or model files.

The independent OBC-II/shared-layer snapshot `11cc18b8`, its detached validation
worktrees and evidence remain intact. The general PRFO final-curvature repair is
retained in `76de6c48` for separate review; it is not discarded or merged here.
