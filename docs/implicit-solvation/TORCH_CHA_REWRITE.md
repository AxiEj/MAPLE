# Source-faithful Torch CHA-GB rewrite: staged evidence

## Active scope decision (2026-09-24)

The target is now an **energy-only, Amber-numerics-faithful Torch single-point
endpoint**. OBC-II/ACE remains the OPT force route; this CHA effort does not
seek analytic forces, Hessians, FREQ, TS, or MD. Exact scalar parity requires
the legacy R6 discrete grid and PBSA cavity/dispersion sampling, not the
alternative continuous surface/volume quadratures. The latter uncommitted
research slice is preserved in a verified local archive, not integrated into
this target. The exact-SP execution contract is
`.omx/plans/route1-torch-amber-exact-sp.md`; the pre-existing M1 algebra
evidence below remains valid but cannot by itself establish whole-endpoint
parity. No new hydration MAE or speedup is claimed.

**License correction:** no additional written exception was supplied, but
GPL-covered code can be modified for private use under the GPL itself. A
source-derived Torch translation may therefore be explored in a separately
identified local research area with upstream notices/provenance; it must not
be silently presented as an independently licensed MAPLE module or pushed as
part of the current combined distribution. The project's `LICENSE` adds
academic-only/noncommercial wording to BSD-3 text, which raises a separate
compatibility issue for distributing GPL-derived combined work. Resolve that
boundary with file-level license review, rights-holder exception, or an
authorized project-license change before integration/release. The already
source-exposed research is **not** a clean-room implementation. None of these
license questions changes the scientific requirement for exact scalar parity.

After scalar parity and final implementation review, the requested accuracy
check is a paired FreeSolv comparison plus **MNSol water-only** neutral
absolute-solvation subset, not all MNSol solvents. It will hold each method's
geometry, charge/topology and standard-state convention fixed, retain every
failure and the cross-dataset overlap, and report the two datasets separately.
The current MNSol source audit identifies 390 candidate neutral water rows;
the count of rows with reusable AM1-BCC/GAFF2 inputs is not yet established.
Under the existing no-new-QM boundary, unsupported rows must be reported as
coverage gaps, not silently given new SQM charges or omitted from a claimed
full-set MAE. Exact replacement is not intended to lower the old model's MAE.

### Exact-SP component status

`torch_pbsa_exact_cavity.py` independently counts the fixed 0.5-Å integer
lattice inside the union of effective `rmin+1.3 Å` spheres and applies the
frozen PBSA SAV coefficient/offset. It is **energy-only and unregistered**;
the caller must pass coordinates after the existing Amber 12.7f serialization
boundary. Against two deterministic full-precision native component replays,
its cavity energy matches all 30 frozen panel molecules. A separate 21-case
coordinate-translation canary and 60 panel-molecule translated/perturbed
coordinate canaries also match native cavity energies and voxel counts.
The frozen 30 cases have identical CPU/CUDA voxel counts and energies on the
locally available CUDA device, and a methyl-hexanoate canary retains the
native cavity step rather than smoothing it away. Raw, pre-serialization
coordinates and gradient requests fail closed.
These are component parity tests, not evidence for exact R6, dispersion, a
complete SP provider, MNSol coverage, or experimental accuracy. No Amber
source body was translated into this kernel.

In separately identified, ignored local **GPL-2.0-or-later research** (not in
the MAPLE package), a source-derived Torch R6 accumulator now consumes
native owner-labelled boundary edges supplied by an instrumented local
GBNSR6 build. Its unshifted inverse Born values match the native trace on
methane, benzene, and a larger flexible ester within `1e-12 Å⁻¹`, including
reentrant-probe-owned edges. This checks the edge-to-R6 accumulation stage
only. The isolated local translation also reproduces native **pre-contact
SAS/VDW grid marking and contact reclassification** for methane, benzene,
and acetone, including thousands of class changes. Probe reentry, final
dielectric-boundary owners/fractions, dispersion, and the full EGB endpoint
were still absent at that stage. The 2026-09-26 continuation below adds only
the reentry stage; there is still no public Torch CHA method. The local
translation retains upstream license/provenance and is not authorized for
the current MAPLE distribution merely by passing numeric tests.

### 2026-09-26: reentrant-probe stage F and rewrite audit

The local `ses_grid_stage_f.py` consumes a supplied stage-E map and ordered
probe-center table. It reproduces stage-F class changes, nearest probe
ownership (including the native `1e-9` tie threshold), and directional
intersection candidates. Methane, benzene, acetone and the larger flexible
ester all pass comparison of **every** class/owner/edge-owner grid entry;
the maximum floating-field difference is `3.55e-15` in grid-index units.
Two native fixture runs have identical stage input/output bytes and retain
the full-precision reference EGB. The frozen comparison is local artifact
`gpl_research/stage-f-comparison-v1.json` under the exact-SP benchmark folder.

This is a CPU reference-order implementation, not an optimized Torch runtime:
the four stage-F evaluations took approximately `0.85 / 15.31 / 10.23 /
35.92 s` in this run. Probe-center generation, final boundary-edge selection
and fractions, dispersion, and whole-endpoint composition remain incomplete.
Native-injected intermediate parity cannot establish Amber-free execution.

All earlier Torch rewrites (admitted OBC derivatives, CHA supplied-radius
algebra, cavity and superseded continuum prototypes as well as local SES/R6)
have received separate interim code and architecture review. The overall
verdict on 2026-09-26 was **REQUEST CHANGES**, not approval. It identified
direct-API identity, analytic-device/D4 admission, constructor ownership,
mutable quadrature caches and unbounded work; these code-contract findings
are addressed by the dated repair below. Missing exact-SP input/assembly
contracts remain open. Ordinary OpenMM OPT is distinct from Torch analytic
workflow admission.
The confirmed cavity index-overflow, pair-work/memory and native field-width
defects were repaired with failing-before regression tests and a narrow
independent recheck (24 cavity/contract tests passed). Stage F now has an
explicit stage-E input type, CPU allocation independent of ambient Torch
defaults, and strict native trace/protocol checks. Complete per-stage
provenance binding still requires work. A portable cavity-only native oracle
has since been added; it does not cover the other exact-SP stages.

### 2026-09-27: audit-driven contract repairs

This repair preserves the existing physical equations, fitted constants,
provider defaults and historical numerical artifacts. It does not register
a Torch CHA endpoint or authorize a new scientific claim.

- The direct correction API validates water/model/profile/nonpolar identity
  and rejects unsupported options and null identity selectors. Provider,
  charge and inner-mode configuration gates precede mutation of caller atoms.
  Charge identity normalization is shared with CLI parsing and charge
  preparation, rather than copied into another provider-specific validator.
- Finalized OBC parameters own their arrays and nested provenance, validate
  derived radii, and enforce the same `78.5/1.0` dielectric profile as the
  verified OpenMM expression. Archived continuum inputs now validate their
  full semantic contract on direct construction as well as artifact loading;
  they are still a different model, not exact-SP inputs.
- Analytic workflow admission is limited to the qualified CPU/no-D4 cell;
  selecting OpenMM Reference alone does not qualify Torch CUDA derivatives.
  Existing OpenMM native-force OPT is unchanged.
- Cached quadrature rules return owned tensors. Archived SAV rejects
  insufficiently bounded geometry/integration work, invalid tolerances and
  stalled intervals. OBC/CHA dense pair graphs and OBC full Hessians use
  explicit work and memory-planning budgets before dense allocation. These
  estimates are not measured performance or proven process-memory bounds.
- `tests/solvation/data/pbsa_cavity_oracle_v1.json` holds six pinned numeric
  native cavity references (three molecules plus translated methane), with
  input/output/binary provenance but no Amber source or experimental labels.
  These portable component tests need neither Amber nor ignored `.omx` data.

The old OBC validation and source-compatibility artifacts remain byte-identical.
A separately dated `route1-torch-post-audit-revalidation-2026-09-27.json`
records fresh native energy/force comparisons and derivative/workflow tests
against frozen implementation, input and test bytes. The new compatibility
ledger documents historical source drift only: it cannot authorize execution
under a changed historical protocol. Validation outcomes must be read from
the new artifact, not inferred from the old scores.

Final local repair verification: **48/48** native OBC comparisons and **396**
targeted regressions pass against the frozen source; the complete local
`tests/solvation` rerun passes **1513 tests** (zero skips/failures). Maximum
energy/force-component differences on that OBC corpus are
`1.44e-15 Hartree` / `4.17e-17 Hartree/Å`. Independent code/spec/security and
architecture reviewers cleared these repairs. Pyflakes, compilation, new-file
formatting and wheel build pass; no static type checker was available. This
is a local supported-environment result, not a clean-clone/full-CHA claim.

That **contract-repair validation** did not cover the then-unfinished exact-SP
chain. The private continuation below now has a complete pinned-panel scalar
path; this does not retroactively alter the public repair artifact or its
1513-test result. No FreeSolv/MNSol-water rescore is reported here.

Current test counts are not an approval of scientific generality or
production readiness. The earlier requested **Astra max final review** has
since approved only the private prepared-input CPU float64 energy-only
milestone below, not public admission or experimental accuracy.

### 2026-09-27: private native-free full-chain candidate

An ignored, source-derived GPL/LGPL research assembly now accepts only the
frozen prepared topology and F12.7 coordinates. It recomputes level-one grid,
ordered arcs/probes, SES stages D/E/F, final `ipb=2` boundary, R6 inverse Born,
CHA polar, PBSA integer-voxel cavity, finite-surface dispersion and total.
All candidate evaluations run with empty `PATH` and file-open/process denial;
native oracle records are read only afterward for comparison. The frozen
30-ID panel passes **30/30** with maximum absolute component/total differences
of `8.53e-14 / 0 / 1.52e-12 / 1.61e-12 kcal/mol`, respectively, versus the
pre-registered `1e-6` per-field gate. Source hashes stayed fixed throughout,
and 30/30 guard contexts recorded zero violations. The complete retained
pre-format private report is
`.omx/benchmarks/route1-torch-amber-exact-sp/exact_chain_panel30_v2.json`
(file SHA256
`614a06dbdf3174ce8711041f3878bfaea6b0b52118141c99eaeff7625db35662`).
The first full report remains preserved at 29/30; a tested native-roundoff
raster fix closed that one failure without changing fitted parameters or
clipping the square root. The focused private GPL/LGPL suite passes 155 tests.

After formatting only 10 newly modified private files, independent AST hashes
are unchanged for all 10 and all 122 frozen prepared-input/original
`prmtop/inpcrd` file hashes remain identical. Black, Pyflakes and compileall
now pass on all 36 private/evidence Python files. The fresh guarded
`exact_chain_panel30_v4.json` (SHA256
`a27625f2788bf113f2285cb23b0af1b5c4913a24fa0e9a8e5a20b13cf2a44200`)
again passes 30/30 and 120/120 strict fields with zero violations; every
candidate energy is binary64-identical to v2. The 155-test private suite and
same-ID coordinate-recomputation witness also pass against the new source
seal. Earlier reports and their source snapshots remain unmodified. A
read-only Astra max incremental review confirmed that the scoped private
approval carries over to the v4 post-Black source freeze; its record is
`.omx/benchmarks/route1-torch-amber-exact-sp/ASTRA_POST_BLACK_FINAL_REVIEW_2026-09-27.md`.

The final Astra max verdict is **APPROVE solely for this private prepared-input
CPU float64 energy-only milestone**, with no unresolved scientific blocker
inside that scope; its retained local record is
`.omx/benchmarks/route1-torch-amber-exact-sp/ASTRA_FINAL_REVIEW_2026-09-27.md`.
This is **not** a public MAPLE provider, licensed combined distribution,
arbitrary-geometry qualification, force/OPT capability, GPU performance, or
FreeSolv/MNSol-water predictive-accuracy result. The separate frozen dataset
comparisons still require explicit prepared-input coverage and protocol gates.
The production
AmberTools CHA endpoint and OBC-II/ACE force route remain unchanged.

This is a **separate research implementation**, not a new MAPLE solvent
provider. The existing `AmberToolsChaGB` scalar remains the only exposed
CHA-GB/PBSA endpoint, supports SP only, and continues to return exactly
`EGB + ECAVITY + EDISPER`. Nothing here changes fixed charges, GAFF2
typing, molecular inputs, fitted parameters, hydration predictions, or the
OBC-II force-capable runtime.

## Current milestone and API

`implicit/torch_chagb.py` currently implements only the zero-salt CHA-GB/ALPB
**polar algebra downstream of NSR6 inverse Born radii**. It consumes four
explicit same-device float64 tensors: Cartesian positions [Å], fixed charges
[electron], `effective_cha_radii_angstrom`, and
`unshifted_inverse_born_per_angstrom` [Å⁻¹]. Its result is in kcal/mol and
retains tensor graphs for those inputs. The inverse Born input is **before**
the CHA size shift; a printed post-shift `rinv` is not a valid substitute.
The supplied radii are **after** the native `cha_rad` mapping and `+Rs`, not
the Bondi values in the original topology.

Actual native input processing also reduces the requested 1.4-Å GBNSR6
probe by `Rs=0.52 Å`, giving 0.88 Å within the CHA factor. Charges entering
the native energy equation are the supplied electron charges multiplied by
18.2223. These conventions are part of the target endpoint and not
tunable Torch options.

**No complete coordinate force exists in this milestone.** Supplying Born
radii from an external executable severs their geometry derivative; the
algebra's autograd output is then only a partial derivative. The module is
intentionally unregistered, has no `SolvationResult`, no `get_forces`, and
is not imported by `ImplicitSolvationCorrection`.

## Critical fidelity boundaries

1. CHA effective-charge sign changes and the electrostatic-size threshold at
   10 Å are hard branches. This implementation preserves them and rejects
   differentiation near those surfaces. It never replaces `sign` with `tanh`.
2. Native GBNSR6 constructs a discrete SES/grid and numerically integrates
   R6 radii. The requested `space=0.3 Å, arcres=0.2 Å` results in **effective**
   `arcres=0.15 Å`. A supplied Born tensor does not implement or validate this
   coordinate chain.
3. The selected PBSA `inp=2, use_sav=1` cavity uses a 0.5-Å integer voxel
   occupancy count, not a continuous sphere-union volume. The existing
   continuous sphere-union prototype is a distinct numerical model and
   cannot be silently substituted. A label-free live canary on pinned
   methyl hexanoate displaced atom-1 x by −0.005 Å and −0.001 Å from its
   reference position and obtained
   cavity values of 20.9437 and 20.9485 kcal/mol, respectively, while
   displacements −0.001/0/+0.001 Å shared the latter rounded value. This
   corroborates the occupancy step; it is not a derivative or experimental-
   accuracy result.
   PBSA dispersion likewise uses its own finite surface sampling and GAFF2
   Lennard-Jones mixing, not ACE.
4. The full native source trace and extracted-equation oracle are built only
   from the verified local AmberTools26 RC7 source in two clean scratch trees.
   The exact 64-file GBNSR6 source inventory is checked against the tracked
   SHA256 manifest, and all six included header files are pinned separately;
   extra files or symlinks are rejected before a forced source build. Object,
   binary, static-link input and resolved dynamic-library hashes are retained.
   The full trace reproduces the previously recorded reference EGB of
   −6.282267527560081 kcal/mol. The extracted equation differs from that
   full trace by only 4.44×10⁻¹⁵ kcal/mol; Torch differs from the nine
   frozen oracle cases by at most 2.85×10⁻¹⁴ kcal/mol in any polar energy
   component and 1.78×10⁻¹⁴ in recorded algebra intermediates. These are
   algebra results, not full-endpoint parity, derivative qualification, or
   experimental accuracy. The committed numeric fixture contains **no
   upstream source or binary**.
5. The exact selected cavity scalar has integer-occupancy steps. At those
   crossings a finite classical force does not exist. Torch automatic
   differentiation cannot turn an exactly matching discontinuous energy
   into a globally smooth force. Any smooth substitute would define a new
   endpoint and requires an explicit scientific decision and fresh accuracy
   validation, not a hidden implementation detail.

The relevant numerical definitions are source-pinned in the local verified
AT26 RC7 `egb.F90`, `gb_read.F90`, `pb_init.F90`, `sa_driver.F90`, and
`NSR6routines.F90`. The independent public AmberClassic snapshot is
[`0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`](https://github.com/Amber-MD/AmberClassic/tree/0b35bfeb96026ffa4e5876391a0828f39b3cfc8d).
GBNSR6 carries GPL-2.0-or-later terms; the locally inspected PBSA tree has
an LGPL-3.0 notice. Formula-level independent implementation is not a
declaration that translating and redistributing either native code body
under MAPLE's existing license is automatically permitted. Licence review
is required before any such source translation is distributed.

## Current energy-only gates (not completed by the polar algebra)

| Gate | Required evidence | Status |
| --- | --- | --- |
| CHA polar algebra | Same native units, parameters and branches, supplied unshifted inverse Born values | Existing M1 parity; included in the scoped private full chain, not a public provider |
| Native SES/R6 geometry | Coordinates to grid/arc construction, all map stages, final boundary edges and inverse radii | Private native-free 30/30 scalar chain; arbitrary geometries not qualified |
| PBSA cavity scalar | Native finite-lattice voxel count with prepared radii and coordinate serialization | Bounded component parity; independent public kernel remains unregistered |
| PBSA dispersion scalar | Native finite-surface sigma-split summation, GAFF2 mixing and sampling | Private source-derived finite implementation passes the pinned panel; not publicly distributable here |
| Full Torch single point | All components composed on the same pinned inputs without native executable or intermediate-state dependency | Private energy-only 30/30, zero runtime guard violations; not registered |
| Scientific comparison and review | Frozen 30-case complete parity, FreeSolv/MNSol-water coverage and paired score, final Astra max review | Private panel and scoped Astra review complete; dataset input coverage and accuracy comparison pending |
| Public integration/release | File-level source/license boundary, capability contract, resource and runtime validation | Pending; no Torch CHA provider enabled |

The historic FreeSolv 526/116 MAE does not transfer merely because a tensor
routine has been written. Former M2–M4 derivative/OPT milestones are
**superseded**, not current blockers to the selected single-point target.
CHA OPT/TS/FREQ/MD are outside this rewrite; existing OBC force workflows
remain unchanged. Passing every scalar gate will not admit CHA derivatives.
