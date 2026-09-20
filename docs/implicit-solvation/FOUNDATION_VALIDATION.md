# Route 1 numerical foundations — 2026-09-13

The first foundation tier exercises **E, F, H, actual OPT convergence and a
first-order saddle**, not additional task keywords. It does not certify
experimental solvation energies, arbitrary reaction chemistry or every solvent.
Experimental SP/OPT/FREQ/TS availability is not withheld for marginal accuracy
or benchmark-equivalence improvements.

## Fixed scope and reproducibility

- Six neutral molecules: water, ammonia, methanol, ethane, fluoromethane,
  benzene. Water and methanol are OPT→FREQ minimum anchors.
- ANI2x, one checkpoint, explicitly float64 throughout this validation;
  ordinary SP/OPT defaults are unchanged. AM1-BCC is prepared once and frozen,
  with identical molecular inputs, charges and mbondi2 radii across endpoints.
- FAST remains OBC-II/ACE on OpenMM CPU. The second endpoint is **polar-only**
  ddLPB with κ = 0.1 Å⁻¹, lmax 9, 302 Lebedev points, η = 0.1 and solver
  tolerance 10⁻¹⁰. It is not full hydration free energy or an ALPB substitute.
- Full Cartesian E→F differences use 0.0002/0.0001 Å. Raw, unsymmetrized
  F→H differences use 0.0005/0.00025 Å. Replays and source/input/charge/radius
  identities are checked. Both raw matrices survive even when a check fails.
- Independent stationary force limits are maximum 2.5×10⁻⁴ and RMS
  1.5×10⁻⁴ Ha/Å. Significant imaginary modes are below −30 cm⁻¹; smaller
  negative/soft modes are retained rather than deleted.

The prospective protocol, raw inputs, source fingerprints and all attempted
runs are under `.omx/benchmarks/route1-foundation-20260913/`; the initial plan
is `.omx/plans/route1-foundation-20260913.md`. These local measurements are
not distributed as historical benchmark replacements.

### Input preparation boundary

The original high-precision RDKit inputs failed the existing AM1-BCC
`geometry=keep` serialization gate: the external tool rounded coordinates.
Those six failures are retained. A separate, declared input case rounds the
**reference geometry before charge generation** to 0.001 Å. All six of these
inputs prepared successfully; no charge/geometry gate was relaxed.
`input-lineage-reconstruction.json` records post-preparation byte-exact
reconstruction of both input sets (ETKDGv3 seed 20260913, RDKit 2024.09.2,
UFF converged), not fabricated original status logs. Arbitrary-precision
external charge preparation is not claimed repaired.

## Complete six-molecule panel

All **12/12** endpoint/molecule cases executed; none were dropped. Energy/force
finiteness, coordinate replay and fixed-input identities pass in all 12.
"E→F" below requires both tested steps to pass maximum and RMS error limits;
"H" requires raw asymmetry and raw step-refinement checks. It is a numerical
test outcome, not a new task-availability switch.

| Molecule | OBC2/ACE E→F | OBC2/ACE H | ddLPB E→F | ddLPB H |
|---|---|---|---|---|
| Water | Fail | Pass | Pass | Pass |
| Ammonia | Fail | Pass | Pass | Pass |
| Methanol | Fail | Pass | Pass | Pass |
| Ethane | Pass | Fail | Pass | Fail |
| Fluoromethane | Pass | Pass | Pass | Pass |
| Benzene | Fail at the smaller step | Pass | Pass | Pass |

Thus all numerical panel checks pass for **1/6 CPU OBC2 cases and 5/6 ddLPB
cases (6/12 combined)**, not 12/12. All **four** water/methanol OPT→FREQ
minimum anchors independently pass force, step, raw-H and minimum-spectrum
checks. The CPU precision and ethane gas-Hessian findings are explained below.
Native results are `panel-v3/summary.json`; `panel-v3-postflight.json` verifies
the full denominator, paired charges/radii/inputs, one live checkpoint,
endpoint settings, source freeze, input reconstruction and raw artifact hashes.
Evidence integrity passes; full numerical-panel acceptance remains false.

## Actual transition-state results

P-RFO previously chased noisy rigid rotations in its full 3N eigensystem.
The repair adds explicit `project_rigid_modes=true`: optimize and follow
modes in the internal mass-weighted subspace, while retaining full Cartesian
forces for convergence. Default full-space behavior is preserved for
laboratory-frame-dependent potentials. Both endpoint runs use the same
original planar NH₃ start and controls, not endpoint-specific tiny trust radii.

| Check | OBC-II/ACE | ddLPB |
|---|---:|---:|
| Converged P-RFO iterations | 3 | 3 |
| Final maximum force, Ha/Å | 9.83×10⁻⁷ | 2.76×10⁻⁵ |
| Significant imaginary modes | 1 | 1 |
| Imaginary frequency at the two H steps, cm⁻¹ | −989.649 / −989.651 | −1008.077 / −1008.079 |
| Both downhill OPT branches converged | Yes, 25/25 iterations | Yes, 26/30 iterations |
| Lower-energy, intact, opposite pyramidal geometries | Yes | Yes |
| Strict preregistered endpoint-energy equivalence | Pass | **Fail** |

Both negative modes match ammonia inversion and remain stable under Hessian
step refinement. Charges, radii, checkpoint and frozen execution sources
match. These are equivalent pyramidal configurations of NH₃ under fixed atom
labels, not different reaction products or evidence for bond-making accuracy.

The ddLPB downhill energies differ by **7.0677×10⁻⁶ Ha = 0.004435 kcal/mol**,
above the locked 10⁻⁶-Ha equivalence threshold. Geometric equivalence checks
pass. The complete strict TS benchmark therefore remains **not accepted**;
the genuine first-order saddle and successful optimizations remain positive
functional evidence. No threshold is relaxed and this small residual is not
used to close experimental TS or launch an indefinite refinement campaign.

## CPU finite-difference precision is a separate issue

Float64 ANI does not make OpenMM's CPU CustomGB kernel double precision.
For the same water input, CPU E→F finite differences have maximum errors
4.4×10⁻⁵ / 9.3×10⁻⁵ Ha/Å at the two small steps. A **separate Reference-platform
diagnostic**, with the same model and parameters, gives 2.46×10⁻⁸ /
5.80×10⁻⁹ Ha/Å. Direct CPU versus Reference differences are only
5.13×10⁻⁹ Ha in E and 4.57×10⁻⁹ Ha/Å in F.

The upstream CPU CustomGB implementation uses float coordinates/intermediates;
Reference provides a double-precision comparison. The source cited is
OpenMM 8.5.0; runtime 8.5.2 properties were also checked. CPU has no
`Precision=double` property. See the
[official CPU CustomGB source](https://github.com/openmm/openmm/blob/b55e60882dffcddb2532cceda4201c0fdc9ec2d5/platforms/cpu/src/CpuCustomGBForce.cpp)
and [CPU platform documentation](https://docs.openmm.org/latest/developerguide/05_cpu_platform.html).
The [platform-properties reference](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html#cpu-platform)
documents CPU controls separately from the GPU `Precision` property.
The CPU panel failures remain failures; the Reference diagnostic does not
replace them, alter the FAST default or independently establish accuracy.

Ethane also fails the locked Hessian step-refinement check on **both**
endpoints: the ddLPB-composed maximum raw change is 0.00310647 Ha/Å² against
a 0.00253033 budget. A same-checkpoint, same-input gas-only diagnostic reproduces
0.00310647 Ha/Å²; the polar component isolated by subtraction changes by only
0.00008505 Ha/Å². This localizes the dominant refinement failure to the ANI2x
gas contribution, not missing ddLPB force support. It does not establish the
underlying network-level cause or qualify a different Hessian step. No model,
PES, difference step or threshold is silently changed to obtain a pass.

## Reusable runners

Use the existing environment with pinned providers and a prepared six-record
input manifest. Output directories must be new. Single-thread settings avoid
inherited MKL oversubscription; they do not change the potential parameters.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export PYTHONPATH="$PWD"
python docs/implicit-solvation/benchmarks/run_route1_foundation_panel.py \
  --inputs-dir /path/to/inputs-precision-v2 --output-dir /path/to/new-panel
python docs/implicit-solvation/benchmarks/run_route1_stationary_validation.py \
  --input /path/to/inputs-precision-v2/ammonia/fixed.mol2 \
  --work-dir /path/to/new-stationary-evidence
```

`check_route1_foundation_evidence.py` adds independent cross-case/checkpoint,
fixed-input-lineage and raw-array postflight checks without changing native
records. Partial endpoint selections cannot become full-panel acceptance.

## Remaining boundaries

- CHA-GB/ALPB + PBSA and APBS remain energy-only; this work does not supply
  their missing derivatives.
- Non-water coverage, 20-per-solvent measurements, broader reaction chemistry,
  MD/path qualification and experimental energy accuracy remain separate work.
- The PRFO file retains whole-file typing debt, including inference diagnostics
  around the new ndarray helper. Numerical behavior passed independent review;
  new benchmark/helper files have clean scoped typing. This is not a claim
  that repository-wide static analysis is clean.
- Prior failed, interrupted and superseded attempts are retained. Historical
  numerical JSON is not recomputed/resealed; the compatibility ledger is
  archival source bookkeeping only. No commit or push is part of this run.

## Regression evidence

The full suite reports **1030 passed and 2 unchanged historical reserve-file
hash failures** (`full-suite-final.log`). The subsequent focused suite, including
the additional parser/provenance guards, passes **38 tests**. New benchmark and
test surfaces pass Ruff, scoped Pyright and compilation. The 143 historical
numerical/protocol JSON files remain byte-identical. Full-suite runs during
an intermediate unsealed compatibility-ledger edit are retained as superseded
verification attempts; the final sealed-ledger run has only the two known
reserve failures. The scientific numerical artifacts were never resealed.
