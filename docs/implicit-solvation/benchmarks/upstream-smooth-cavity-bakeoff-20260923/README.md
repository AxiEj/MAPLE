# Upstream smooth-cavity research bake-off (2026-09-23)

This is an **archived, non-admitted research result**, not a replacement for
MAPLE's reviewed Torch v3 scalar and not a physical-solvation-accuracy claim.
The frozen MACE-POLAR checkpoint was not changed, no parameters were fitted,
and no production source was edited. The active baseline is Git
`6ce902c5aafb61299934f75ebe51563fc9bf67c8`. Its source-manifest digest
matches the prior ten-molecule v3 panel:
`527b7aebf3623c6cc0e7ada61fa1aca508d29b2c772c2dd908f3fdffadfa6d99`.

## Read this first

- [Complete report](upstream-swig-bakeoff-20260923/REPORT.md) — model identities,
  ten-species results, numerical caveats and explicit non-admission.
- [SWIG/ISWIG frozen protocol](upstream-swig-bakeoff-20260923/protocol.json) and
  [raw results](upstream-swig-bakeoff-20260923/results.json).
- [MOIST corrected v2 protocol](upstream-moist-area-v2-20260923/moist-protocol.json)
  and [raw results](upstream-moist-area-v2-20260923/moist-results.json).
- [MOIST v1 invalidation](upstream-swig-bakeoff-20260923/MOIST_V1_INVALID.md):
  the original wrapper transposed coordinates; both its failed and misleading
  values are preserved but **must not be used**.
- [Original v3 ten-molecule baseline](ten-molecule-panel-20260923/report.md),
  its frozen protocol/source manifest and all ten raw `result.json` files are
  included for comparison. HCN remains fail-closed in Torch v3.

PySCF 2.13.1 supplied the upstream SWIG/ISWIG C-PCM surfaces and Gaussian
interaction matrix; current MOIST was pinned to commit
`a0116e8ee369c546d48558e5e389e6f20bdff2e9` and used only through its
public C API for SvdW-DROP per-atom areas with the unchanged SMD CDS radii.
MOIST's tagged `v0.6.0-alpha.1` could not be built unchanged because its
`jonquil.wrap` referenced a removed `recursion` branch; the tested main commit
was built in an isolated `/tmp` directory. Neither external library nor the
MACE checkpoint is copied into this bundle.

## Reproduction contract

The Python files here are **byte-identical frozen source snapshots**. Their
root calculation assumes their original private artifact depth; run them from
a fresh copy under `.omx/research/pure-torch-analytic-20260922/`, not directly
from this documentation directory. Copy the included `ten-molecule-panel-20260923`
there as the baseline. For the SWIG replay, copy only `bakeoff.py` and
`protocol.json` into a new sibling research directory at that depth, then run
`bakeoff.py run`; retain the original frozen result separately. For the MOIST
replay, restore the original `upstream-swig-bakeoff-20260923` directory (its
invalid-v1 protocol/result hashes are intentionally part of the v2 contract),
copy only the v2 `moist_area.py` and `moist-protocol.json` into a fresh sibling
at the same depth, then run `moist_area.py run`.

The checkpoint must be available at
`~/.cache/mace/MACE-POLAR-1-M.model` with SHA256
`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.
Run SWIG with the same Python environment (`pyscf==2.13.1`, Torch float64),
`PYTHONPATH="$PWD"`, and the conda environment's `lib` first in
`LD_LIBRARY_PATH`. For MOIST, clone the pinned commit to
`/tmp/maple-moist-svdw-drop-main-20260923`, build its Python extension in
`build-maple` with Meson debug settings, then prepend
`build-maple/python` to `PYTHONPATH` and `build-maple` to `LD_LIBRARY_PATH`.
The exact build path and Python 3.11 extension suffix are environment-specific;
this is research reproducibility evidence, not a deployable package recipe.

Both executed batch processes exited 0. Existing targeted tests returned
`30 passed in 2.49s`; the two frozen research scripts passed Python 3.11
bytecode compilation, Black and Pyflakes. All JSON and Python files listed in
`SHA256SUMS` are immutable archived evidence; replay outputs belong in a new
directory.

## Scientific boundary

The table values are fixed-geometry `E_polar + E_CDS` model energies in water,
not measured solvation free energies. There was no MNSol/FreeSolv/Solv@TUM/
CompSol/FlexiSol three-fit/two-sealed-test experiment, no demonstrated
non-inferiority, no full-total-PES analytic Hessian, and no final Astra max
approval of a smoothed model. The current v3 baseline retains its existing
approval and branch-limited identity; this bundle opens no new capability.
