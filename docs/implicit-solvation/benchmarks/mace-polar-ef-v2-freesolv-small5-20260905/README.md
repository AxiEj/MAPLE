# V2 fixed-geometry hydration diagnostic, 2026-09-05

**Not scientific admission.** Five preselected calculations were attempted:
three succeeded and two failed. Successful-subset MAE = 8.3180780967 and
RMSE = 10.4165466323 kcal/mol. No full-five MAE/RMSE exists.

| Compound ID | Molecule | Experiment | Estimate | Signed error |
| --- | --- | ---: | ---: | ---: |
| mobley_9055303 | methane | 2.00 | failed: SCF nonconvergence | unavailable |
| mobley_1636752 | methanol | -5.10 | -12.1572397188 | -7.0572397188 |
| mobley_7532833 | acetonitrile | -3.88 | -2.5327358852 | 1.3472641148 |
| mobley_2198613 | chloroethane | -0.63 | failed: source/target-shell clearance | unavailable |
| mobley_3034976 | acetic acid | -6.69 | -23.2397304565 | -16.5497304565 |

All values are kcal/mol; signed error = estimate - experiment. The estimate
is the current fixed-conformer SMD electrostatic-plus-CDS result, not a
sampled ensemble free energy. These five small molecules are not a
representative full-FreeSolv test or a replacement for the parent 12-record
gate. No parameter fitting or conformer optimization was performed.

## Contents and provenance

- `preregistered-panel.json`: original, unmodified pre-run lock; exact source
  hashes, checkpoint digest, profile, geometry policy, inputs, dependency
  versions, experimental records, and dataset identity.
- `parent-panel.json`: original parent manifest; selection is its five
  smallest `(natoms, compound_id)` records, independent of outcomes.
- `results.json`: original complete accounting and successful-subset metrics.
- `raw/<compound_id>/`: original input, output and console log, plus the
  original coupling JSON for each successful calculation.
- `archive-sha256.json`: byte hashes of the original archived evidence.
- `run_panel.py`, `runtime-origin.txt`: original execution script and recorded
  import origin. Historical absolute local paths are retained, not rewritten.

The original run used a frozen 370-file source snapshot, neutral charge,
singlet multiplicity 1, CUDA 0, and the unchanged multi-solvent derivative
diagnostic profile. Each record had a fixed 1200-second harness limit; the
two failures exited with exceptions rather than timing out. All successful
outputs retain `scientifically_valid=false`. A molecule-specific passivity
check can pass (acetonitrile did) without admitting the model/profile.

This is an audit archive, **not a self-contained executable environment**.
The private checkpoint, full source snapshot, deposited MOL2 files, and
runtime dependencies are not redistributed here. They remain in the original
local `.omx/validation/v2-freesolv-small5-20260905/` layout. Replaying the
original script requires that layout or a separately documented relocation;
validate every locked source/checkpoint/input/geometry hash first. The script
itself checks runtime source/checkpoint/input/geometry hashes but does not
rehash dataset/parent files or recompute selection during execution. An
independent post-run audit verified those as well and recomputed all metrics.
Archive validation is not fresh scientific validation of a later checkout.

Before publication on 2026-09-06, the five V2 test modules, three related
ledger/provider/contract modules, and the new archive module passed
**211 tests in 139.34 seconds**. Python compilation passed. No configured
lint/type checker was available. Original output files retain their exact
whitespace (including the coordinate-heading trailing spaces); do not
reformat these hash-bound raw records.

## Experimental reference

The experimental column, not the historical GAFF calculated column, comes
from [MobleyLab FreeSolv v0.52 at the pinned commit](https://github.com/MobleyLab/FreeSolv/tree/6c7d19b4b565537365ffd22006aa2cd4643200c6).
Per-record primary references and uncertainties are preserved in the lock.
The original 1 M ideal-gas to 1 M aqueous convention is used; no 1-atm shift
is added. Deposited GAFF MOL2 files are fixed reproducible conformers, not
experimental geometries or conformer ensembles. Neutral acetic acid is
compared with its neutral-species record, not an ionized aqueous ensemble.
