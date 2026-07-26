# Implicit-solvation benchmarks

## Route 1: FreeSolv ten-molecule pilot

## Reproducible full-corpus harness

The pilot below is preserved as development-only evidence.  The certification
runner is now defined by [`protocol.json`](protocol.json), which pins FreeSolv
v0.52 commit `6c7d19b4b565537365ffd22006aa2cd4643200c6` and SHA256 hashes for
`database.txt`, `database.json`, and `mol2files_gaff.tar.gz`.  Preparation
currently reconciles all 642 records: 526 deterministic development records and
116 reserved records, with all ten previously inspected pilot records forced
into development. The legacy `prepared.json` includes experimental labels for
both partitions. The 116 records have now been calculated behind the separate
sealed boundary documented below, but they are still not an unopened or
label-sealed confirmation set; a future certification claim needs a genuinely
independent external or label-sealed evaluation.

Canonical commands:

```bash
PROTOCOL=docs/implicit-solvation/benchmarks/protocol.json
WORK=.omx/benchmarks/neutral-water-freesolv

python docs/implicit-solvation/benchmarks/run_freesolv.py \
  prepare --protocol "$PROTOCOL" --work-dir "$WORK"
python docs/implicit-solvation/benchmarks/run_freesolv.py \
  run --protocol "$PROTOCOL" --work-dir "$WORK" --partition development \
  --jobs 4
python docs/implicit-solvation/benchmarks/run_freesolv.py \
  summarize --protocol "$PROTOCOL" --work-dir "$WORK" \
  --partition development --output "$WORK/development-summary.json"
```

The runner writes one atomic JSON record per molecule/charge/GB attempt, resumes
without overwriting completed records, retains provider failures in the original
denominator, and refuses incomplete or protocol-mismatched summaries.  Summary
JSON contains deterministic MSE, MAE, RMSE, maximum error, failure rate,
bootstrap confidence intervals, and predeclared functional-group, element,
size, heteroatom, and flexibility-proxy strata.

`--jobs` parallelizes independent molecules while retaining the same atomic,
per-attempt records.  Choose the count for the available memory and CPU; the
default remains one.

### Completed development partition (2026-07-23)

The frozen 526-molecule development partition has been run with AmberTools
26.0 and OpenMM 8.5.2.  The exact summary, all predeclared strata, failure
records, and 5,260 per-attempt hashes are stored in
[`freesolv-development-2026-07-23.json`](freesolv-development-2026-07-23.json).

| charge / GB | n | MSE | MAE | RMSE | maximum absolute error | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| ABCG2 / HCT | 526 | -1.215 | 2.044 | 3.185 | 20.038 | 0 |
| ABCG2 / OBC-I | 526 | -0.728 | 1.778 | 2.526 | 11.205 | 0 |
| ABCG2 / OBC-II | 526 | -0.715 | **1.652** | **2.358** | 10.429 | 0 |
| ABCG2 / GBn | 526 | -0.586 | 1.760 | 2.398 | 11.080 | 0 |
| ABCG2 / GBn2 | 515 | -1.083 | 1.894 | 2.797 | 15.235 | 0.0209 |
| AM1-BCC / HCT | 526 | -0.361 | 1.855 | 2.769 | 13.127 | 0 |
| AM1-BCC / OBC-I | 526 | 0.068 | 1.930 | 2.728 | 14.561 | 0 |
| AM1-BCC / OBC-II | 526 | 0.044 | 1.760 | 2.537 | 13.550 | 0 |
| AM1-BCC / GBn | 526 | 0.081 | 2.028 | 2.983 | 15.285 | 0 |
| AM1-BCC / GBn2 | 515 | -0.301 | 1.894 | 2.827 | 20.863 | 0.0209 |

All energies are in `kcal/mol`.  The 22 failures are the two charge methods
for GBn2 on the 11 phosphorus-containing development molecules.  They are the
predeclared OpenMM phosphorus fail-closed boundary, not missing records. This
frozen 2026-07-23 matrix predates the later sulfur-parity finding; the current
runtime also rejects S-containing GBn2 rather than using OpenMM's inaccurate
generic sulfur branch.
ABCG2/OBC-II has the lowest development MAE, but this does not freeze a product
default: conformer sensitivity and human review remain open, and the
reserved partition has not been calculated.

### Label-blind CHA-GB and nonpolar comparison (2026-07-24)

[`chagb_nonpolar_protocol.json`](chagb_nonpolar_protocol.json) freezes the two
requested AM1-BCC development endpoints without using FreeSolv labels during
energy execution:

1. GBNSR6 CHA-GB `EGB` plus its `0.005 kcal mol^-1 A^-2` surface term.
2. The same CHA-GB `EGB` plus PBSA `inp=2` `ECAVITY + EDISPER`.

The label-free
[`chagb_nonpolar_source_manifest.json`](chagb_nonpolar_source_manifest.json)
pins all 526 source-record hashes, original FreeSolv GAFF MOL2 hashes, and
AM1-BCC charge vectors. The energy records contain no experimental field.
Only the separate summary phase opens the experimental values. No residual
model, experimental refit, or label-selected chemistry switch is present.

| AM1-BCC endpoint | n | MSE | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|---:|---:|
| OBC-II + ACE baseline | 526 | 0.044 | 1.760 | 2.537 | 13.550 |
| CHA-GB + surface tension | 526 | -0.123 | 1.449 | 1.975 | 9.600 |
| CHA-GB + cavity/dispersion | 526 | -0.160 | **1.322** | **1.854** | **9.416** |

Relative to OBC-II/ACE, the paired MAE gain is
`0.311 kcal/mol` for CHA-GB/surface (95% bootstrap CI
`[0.197, 0.429]`) and `0.438 kcal/mol` for
CHA-GB/cavity-dispersion (CI `[0.321, 0.559]`). The second endpoint improves
320 cases and worsens 206. Replacing the simple surface term by explicit
cavity/dispersion gives a further `0.128 kcal/mol` mean MAE gain
(CI `[0.054, 0.199]`).

The gain is not uniform. Cavity/dispersion is especially helpful for
halogen-containing molecules (`2.148 -> 1.232 kcal/mol`) and hydrocarbons
(`0.611 -> 0.486`), but is worse than the simple CHA surface term for the
one-heteroatom stratum (`0.939 -> 1.110`). A per-chemistry selector inferred
from those labels would be a new fitted model and is deliberately not added.

The direct Antechamber GAFF2 atom-type output was rejected for this benchmark:
AmberTools 26 `parmchk2` produced nondeterministic completion of novel types
such as `n8`, changing the PBSA dispersion energy. The frozen workflow instead
preserves the original FreeSolv GAFF atom types, replaces only their charge
column with the AM1-BCC vector, and obtains the dispersion nonbonded
parameters from `leaprc.gaff2`. Bonded `parmchk2` terms are not part of any
reported solvation component. Two independent 526-case runs produced
bit-for-bit identical raw records and the same summary SHA256
`a842604d423dd01833b7e4a2b2d92ce0944e0d0e6f2c2e1d883edeaf2cf7308a`.

The exact metrics, strata, bootstrap results, and record hashes are in
[`freesolv-am1bcc-chagb-nonpolar-2026-07-24.json`](freesolv-am1bcc-chagb-nonpolar-2026-07-24.json).
Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_chagb_nonpolar.py run \
  --workers 4
python docs/implicit-solvation/benchmarks/run_chagb_nonpolar.py summarize
```

This is still a fixed-geometry development benchmark. Neither endpoint is
wired into the MAPLE runtime default, and no force/optimization or
finite-temperature conformer gate has been passed for the new composition.
The legacy reserved partition is evaluated separately below. Its earlier label
exposure means that result is held-out by computation, not label-sealed
confirmation evidence.

### Factorial component-attribution audit (2026-07-25)

[`route1_component_attribution_protocol.json`](route1_component_attribution_protocol.json)
freezes a post hoc \(2\times2\) analysis of the component records above. The
script recombines already generated OBC-II/CHA-GB polar and ACE/PBSA
cavity-dispersion energies; it performs no new physical calculation, fit,
residual correction, chemistry selector, or runtime change. Experimental
labels are used only to score the frozen development endpoints:

| frozen AM1-BCC endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| OBC-II/PBSA cavity-dispersion | 2.103 | 3.011 | 15.784 |
| CHA-GB/ACE | 1.819 | 2.412 | 10.036 |
| CHA-GB/GBNSR6 surface term | 1.449 | 1.975 | 9.600 |
| CHA-GB/PBSA cavity-dispersion | 1.322 | 1.854 | 9.416 |
| fine-grid APBS/ACE | 1.629 | 2.462 | 14.116 |
| fine-grid APBS/PBSA cavity-dispersion | 2.161 | 3.143 | 16.310 |

All values are kcal/mol. Against OBC-II/ACE, the isolated CHA-GB polar swap
has paired MAE gain `-0.059` with 95% interval `[-0.208, 0.094]`; the isolated
PBSA nonpolar swap has gain `-0.343` with interval `[-0.467, -0.216]`.
The joint swap alone passes, with gain `0.438` and interval `[0.319, 0.559]`.
The endpoint Shapley allocation is `0.361/0.077 kcal/mol` for the
polar/nonpolar swaps and closes to the joint gain, but its
`0.840 kcal/mol` interaction is large and the allocation does not prove
microscopic causality. The result is evidence for a coupled parameterization,
not permission to transplant either component by itself. Thus neither
isolated component swap is promoted.

The sealed artifact
[`route1-chagb-component-attribution-2026-07-25.json`](route1-chagb-component-attribution-2026-07-25.json)
contains 526 source hashes, all component energies, all crossed predictions,
bootstrap intervals, and the fail-closed decision. Reproduce it with:

```bash
python docs/implicit-solvation/benchmarks/run_route1_component_attribution.py
```

### Pre-registered FreeSolv reserve (2026-07-25)

[`route1_freesolv_reserve_protocol.json`](route1_freesolv_reserve_protocol.json)
freezes two endpoints, all 116 source hashes, full coverage, paired materiality
and uncertainty gates, and a no-fit/no-selection/no-post-score-tuning policy.
[`route1_freesolv_reserve_source_manifest.json`](route1_freesolv_reserve_source_manifest.json)
contains no experimental fields. The energy phase generates AM1-BCC charges,
OBC-II/ACE energies, and CHA-GB/PBSA cavity-dispersion energies, then seals all
116 records before the scoring phase may read the pinned legacy labels:

```bash
python docs/implicit-solvation/benchmarks/run_route1_freesolv_reserve.py \
  prepare-manifest
# Confirm the printed manifest SHA256 equals the value frozen in the protocol.
python docs/implicit-solvation/benchmarks/run_route1_freesolv_reserve.py run
python docs/implicit-solvation/benchmarks/run_route1_freesolv_reserve.py score
```

The frozen results are:

| endpoint | coverage | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|---:|
| AM1-BCC/OBC-II/ACE | 116/116 | 1.791 | 2.464 | 9.622 |
| AM1-BCC/CHA-GB/PBSA cavity-dispersion | 116/116 | 1.301 | 1.846 | 6.376 |

All errors are kcal/mol. The paired MAE gain is `0.490`, with 95% bootstrap
interval `[0.215, 0.782]`; all pre-registered overall retention gates pass.
This retains CHA-GB/PBSA as the explicit SP-only fixed-geometry accuracy
profile but does not change the OBC-II/ACE default or establish forces,
optimization, conformer sampling, MLIP accuracy, or independent confirmation.
The label-free energy and score artifacts are
[`route1-freesolv-reserve-energy-2026-07-25.json`](route1-freesolv-reserve-energy-2026-07-25.json)
and
[`route1-freesolv-reserve-score-2026-07-25.json`](route1-freesolv-reserve-score-2026-07-25.json).

### Amber PB `inp=2` exact-difference force rejection (2026-07-24)

[`amber-pb-inp2-force-probe-methyl-hexanoate-2026-07-24.json`](amber-pb-inp2-force-probe-methyl-hexanoate-2026-07-24.json)
tests the otherwise attractive internal correction

```text
DeltaG_candidate(R) = E_MM,PB(inp=2)(R) - E_MM,vacuum(R)
```

on the same 23-atom methyl-hexanoate topology used by the performance trace.
The subtraction removes the MM gas term from the reported Route 1 potential,
but its force difference fails centered finite differences of that exact
energy difference. At `0.003 A`, the largest of four sampled Cartesian errors
is `3.4200 kcal/mol/A`; at `0.01 A`, it is still
`1.8108 kcal/mol/A`. This is a local rejection probe rather than a broad PB
force benchmark, but one failed conservative derivative is sufficient to
exclude the candidate from OPT and relaxed SCAN. The artifact also stores the
bonded and direct/1-4 electrostatic and van der Waals terms from both contexts;
their largest difference is `5.33e-15 kcal/mol`, directly auditing the intended
gas-MM cancellation.

Reproduction in the pinned AmberTools environment:

```bash
/home/axie/miniconda3/envs/maple-ambertools/bin/python \
  docs/implicit-solvation/benchmarks/run_amber_pb_force_probe.py
```

The official Amber25 manual independently states that GBNSR6 cannot yet be
used in dynamics and documents the standalone PB force example with nonpolar
interactions disabled. Thus the high-accuracy CHA-GB/PBSA endpoint remains
single-point energy-only rather than a derivative-capable OPT/SCAN provider.

[`route1-chagb-derivative-capability-audit-2026-07-25.json`](route1-chagb-derivative-capability-audit-2026-07-25.json)
strengthens that boundary with pinned source plus a real `debugf` run. At
AmberClassic commit `0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`, the driver
passes a force array into `egb`, but the GB/CHA-GB equation routines accumulate
energy without accepting that array. The cavity derivative call is inactive;
only partial dispersion force assignments remain. AmberTools 26 reports
nonzero numerical force components beside zero analytical components for the
audited methyl-hexanoate atom. The same audit finds no `igb=9` or AR6 topology
consumer in current `msander`, so the dormant AR6 topology writer is not a
maintained Route 1 force provider.

Reproduce the capability audit with:

```bash
/home/axie/miniconda3/envs/maple-ambertools/bin/python \
  docs/implicit-solvation/benchmarks/run_chagb_derivative_capability_audit.py \
  --amberclassic-source /path/to/pinned/AmberClassic
```

[`route1-chagb-runtime-provider-parity-2026-07-25.json`](route1-chagb-runtime-provider-parity-2026-07-25.json)
records the explicit runtime promotion boundary. Across all 526 source cases,
the AM1-BCC normalized charge vector and geometry are unchanged, but
Antechamber `-at gaff2` retyping differs from the frozen input atom types in
95 cases. The runtime therefore preserves the submitted GAFF/GAFF2 types and
fails if `parmchk2` requests any `NONBON` override. On the live methyl
hexanoate parity case, runtime `EGB`, `ECAVITY`, `EDISPER`, and total match the
frozen record within `6e-13 kcal/mol`. The provider is explicit SP-only,
retains executable hashes and inputs/outputs, and reads no gas or bonded MM
energy. A real CPU MACE-OFF23m SP closes
`E_MLIP,gas + DeltaG_solv` exactly and returns no solvent force. This is one
named-adapter composition smoke, not universal-MLIP evidence. OBC-II/ACE
remains the force-capable default.

### Remaining analytical candidate screen (2026-07-24)

[`route1-force-consistent-candidate-screen-2026-07-24.json`](route1-force-consistent-candidate-screen-2026-07-24.json)
reuses the frozen 526-case AM1-BCC development evidence to screen every
remaining maintained analytical shortcut before adding another product option:

| candidate | accuracy/applicability result | decision |
|---|---|---|
| neutral ALPB + OBC-II/ACE | MAE `1.760351 -> 1.758003`; paired gain `0.00235 kcal/mol`, CI `[-0.00274, 0.00758]`; 335/526 cases worsen | no material gain |
| GBn2 + ACE | on the same 515 cases, MAE is `1.893983` versus OBC-II's `1.661487`; 11 phosphorus cases fail closed | worse and incomplete |
| LCPO nonpolar | parameters available for 454/526 cases; on that identical subset MAE `1.800792 -> 2.250243`, paired ACE-minus-LCPO gain `-0.449450 kcal/mol`, CI `[-0.581143, -0.318252]`; 285/454 cases worsen | worse and incomplete |

The frozen candidate screen predates the expanded provider-parity corpus.
Current GBn2 runtime applicability is narrower: both P- and S-containing
inputs fail closed.

For neutral solutes, the ALPB \(Q^2/A\) term is zero, so its polar energy and
force are simply the original GB polar energy and force multiplied by the
literature-fixed factor `0.9927734691`. This makes the force path honest but
does not make the accuracy gain meaningful. LCPO coverage is tested directly
against the upstream OpenMM parameter assignment; missing combinations are not
silently substituted. The supported-subset comparison recomputes LCPO from
the frozen coordinates and AM1-BCC charges, retains the same OBC-II polar
energy exactly, and stores all 454 paired predictions in the self-hashed
artifact.

The source development records already contain experimental labels, so this is
open development screening, not label-blind confirmation. No constants are fit
and no residual model is introduced.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_force_consistent_candidate_screen.py
```

### External provider feasibility audit (2026-07-24)

[`route1-provider-feasibility-2026-07-24.json`](route1-provider-feasibility-2026-07-24.json)
freezes why no literature candidate was added merely because it reports better
endpoint accuracy or contains an OpenMM force class:

| candidate | positive evidence | blocking product gate |
| --- | --- | --- |
| IWM-GB | published test RMSE `0.87-0.95 kcal/mol` on 85 rigid neutral H/C/N/O molecules | experimental-label parameter optimization, restricted domain, no audited released conservative-force API |
| OpenMM AGBNP | AGBNP1 Reference source builds against local OpenMM 8.5.2 and returns a finite two-particle energy | AGBNP2 incomplete, no maintained generic small-molecule typing, no Route 1 force/coverage/accuracy gate |
| OpenMM AGBNP3 | force interface exposes radius, dispersion, cavity, and hydrogen-bond inputs | 2015-era code and Desmond DMS/OPLS-specific example typing; adopting it would require a new MAPLE typing model |

The finite AGBNP1 energy is an ABI smoke only. It is not provider parity,
finite-difference force validation, or FreeSolv accuracy evidence. The audit
therefore keeps OpenMM Amber GB plus declared ACE/LCPO applicability as the
maintained runtime and avoids a speculative backend registry or typing layer.
The artifact freezes protocol id `route1-provider-feasibility-v1`, primary
paper/source/build fingerprints, and a self-consistent content hash; temporary
build paths themselves are deliberately omitted.

### GBMV2 physical-provider feasibility audit (2026-07-25)

[`route1-gbmv2-feasibility-audit-2026-07-25.json`](route1-gbmv2-feasibility-audit-2026-07-25.json)
freezes a promising physical-provider result without pretending the runtime is
available:

| evidence | result | boundary |
|---|---|---|
| 499 neutral molecules, GAFF/AM1-BCC, vacuum and GBMV2 10.5 ns trajectories, BAR | GBMV2 AUE/RMSE `1.14/1.60 kcal/mol`; 84% within 2 kcal/mol | SASA coefficient selected using experimental hydration labels; historical protocol |
| coefficient selected on 82 CGENFF compounds, evaluated on 375 other compounds | GAFF/AM1-BCC GBMV2 AUE `1.24 kcal/mol`, \(R^2=0.758\) | not the current MAPLE FreeSolv split or provider |
| CHARMM GBMV2 analytical method II and 2020 GPU implementation | polar and SASA first derivatives; protein CPU/GPU parity | registered CHARMM/pyCHARMM or CHARMM/OpenMM plugin, not public OpenMM core |
| local environment | OpenMM 8.5.2 exposes OBC and AMOEBA GK, but no GBMV2 symbol; no `charmm` executable | no provider parity, all-\(3N\) force, license, or small-molecule throughput test |

No GBMV2 runtime provider, dependency, FreeSolv selector, or paper-based
reimplementation was added. The machine-readable decision is
`scientifically_promising_runtime_blocked`, with explicit gates for reopening
the candidate when a runnable and deployable upstream implementation exists.

### FACTS/GBSW physical-provider feasibility audit (2026-07-25)

[`route1-facts-gbsw-feasibility-audit-2026-07-25.json`](route1-facts-gbsw-feasibility-audit-2026-07-25.json)
freezes two additional physical candidates without relabeling their published
comparators:

| model | AM1-BCC hydration evidence | force/speed boundary | admission |
|---|---|---|---|
| GBSW | optimized 499-case AUE/RMSE `1.20/1.52`; coefficient-selected-on-82 then 375-case AUE `1.33 kcal/mol` | full solvation forces and smooth boundary; about `4x` slower than vacuum, `2--3x` faster only than GBMV | secondary watch item; registered CHARMM GBSW plugin unavailable locally |
| FACTS | optimized 499-case AUE/RMSE `1.25/1.80`; 375-case AUE `1.42 kcal/mol` | analytical, MD-capable; about `4x` slower than vacuum; protein-derived radius parameters with `TAVW` interpolation | not selected behind GBMV2/GBSW |

No FACTS or GBSW runtime provider, dependency, FreeSolv selector, or
paper-based reimplementation was added. Public OpenMM has neither named
implementation. The artifact records `GBMV2 -> GBSW -> FACTS` as the current
physical-provider watch order and requires an upstream runtime, fixed
small-molecule parameter profile, all-\(3N\) force parity, label-separated
accuracy, deployment rights, and measured multi-MLIP throughput before
reopening admission.

### SLIC/CDC physical-provider feasibility audit (2026-07-25)

[`route1-slic-cdc-feasibility-audit-2026-07-25.json`](route1-slic-cdc-feasibility-audit-2026-07-25.json)
freezes a high-accuracy physical lead and the reasons it is not a current
provider:

| evidence | result | boundary |
|---|---|---|
| 2022 SLIC/CDC molecular model | SLIC electrostatics plus cavity, atom-typed dispersion, combinatorial, and hydrogen-bond terms; Mobley AM1-BCC inputs | 38 fitted physical-model parameters; experimental totals and explicit-solvent components are used |
| training disclosure | prose says 65 neutral compounds; Table S4 lists 63 names | not label-blind |
| independent SI Table S10 reanalysis | 494 rows, 492 normalized unique names; MAE/RMSE `0.813/1.152 kcal/mol` | differs from footer `0.69/0.98`; `1.152` matches the preprint's `1.15` |
| historical rows after excluding the 63 listed training names | 431 rows, MAE/RMSE `0.826/1.190 kcal/mol` | not a prospective MAPLE reserve |
| public implementation | Figshare supplies one SI PDF; PBJ supplies related SLIC plus simple SASA, not complete SLIC/CDC | no complete executable 2022 upstream; no complete atom-resolved polar-plus-CDC force |

The source-level reproduction must also reconcile the article's 500-solute
claim, Appendix E's 502-corpus statement, the 494 tabulated rows, and the two
RMSE summaries. PBJ is not the published SLIC/CDC model: its nonpolar term is
not complete CDC, no nonpolar force is exposed, and its final solvation force
is a total three-vector rather than one force per atom. The older MATLAB
repository targets ion/solvent parameter optimization and has no top-level
license.

The frozen Table S10 statistics are source-bound rather than copied from the
audit JSON. Reproduce
[`route1-slic-cdc-si-table-replay-2026-07-25.json`](route1-slic-cdc-si-table-replay-2026-07-25.json)
from the retained, SHA-256-pinned published SI with:

```bash
python docs/implicit-solvation/benchmarks/replay_slic_cdc_si_table.py \
  --pdf .omx/research/slic-cdc-paper-20260725/ct2c00248_si_001.pdf \
  --output docs/implicit-solvation/benchmarks/route1-slic-cdc-si-table-replay-2026-07-25.json
```

The script parses Table S4's 21-by-3 name grid from the layout-preserving
extraction and freezes the ordered 63-name payload hash. It parses Table S10
from both `-layout` and `-raw` modes, requires identical row records, and
recomputes every reported metric. These are two extraction modes from the same
PDF and Poppler executable, not independent data sources. The replay checks
transcription and arithmetic; because it uses the paper's published table, it
is not an independent confirmation of model accuracy.

No SLIC/CDC runtime provider, dependency, FreeSolv selector, speed claim, or
paper-based implementation was added. The candidate is an energy-only final-SP
watch item. Reopening requires a maintained complete upstream, exact typing and
parameter provenance, resolution of the public-data discrepancies, prospective
label-separated accuracy, all-\(3N\) conservative forces before OPT/SCAN/MD,
and measured multi-MLIP performance.

### Learned-solvent product-boundary audit (2026-07-25)

[`route1-learned-solvent-boundary-audit-2026-07-25.json`](route1-learned-solvent-boundary-audit-2026-07-25.json)
freezes why technical feasibility does not admit GNNIS or QM-GNNIS to Route 1:

| candidate | positive evidence | blocking Route 1 gate |
|---|---|---|
| GNNIS | solvent-only scalar; polar/nonpolar closure; scalar/autograd-gradient construction; released checkpoint with permissive MIT-family text; predefined charges or native AM1-BCC | trained with force-only MSE against explicit-solvent mean forces minus vacuum OpenFF forces; learned Born-radius and SASA terms |
| QM-GNNIS | no new QM/MM or experimental training; differentiable energy/gradient/Hessian transfer; published relative conformer/NMR/IR evidence | released delta class implements `G_GNNIS-G_GB-Neck2`; published runtime is Torch delta + ORCA/CPCM, while applying it to arbitrary gas MLIPs would be both a learned correction and an unvalidated transfer |

The isolated methyl-hexanoate smoke gives
`-28.891659 + 4.725903 = -24.165756 kJ/mol` and reproduces one Cartesian
force to `0.20%` relative error. That result is deliberately recorded as
runtime-feasibility evidence only. It proves local scalar-gradient agreement,
not global smoothness, broad force accuracy, MD admission, or absolute
hydration accuracy, and it cannot override the product contract.
No dependency, runtime provider, experimental-label selector, or FreeSolv
screen was added. The artifact pins both source commits, the GNN checkpoint,
licenses, paper extraction, audit script, and all relevant SHA-256 values.
The repository page classifies the GNNIS license as MIT-0 while `setup.py`
uses the generic MIT classifier; the audit records both labels and treats the
hashed license text, rather than either shorthand, as authoritative.

### GBr6 full development energy screen (2026-07-24)

[`gbr6_screen_protocol.json`](gbr6_screen_protocol.json) pins the public GPL
archive and Fortran source hashes, build command, dielectric conditions,
fixed-AM1-BCC charges, and OpenMM `gbn-bondi` radius assignment. The GBr6
website describes first derivatives, but the distributed `GBr6_v1.f` program
only parses coordinates/charges/radii and prints an energy; no released force
or gradient interface was identified. The energy was therefore screened before
considering a MAPLE derivative implementation.

[`route1-gbr6-energy-screen-2026-07-24.json`](route1-gbr6-energy-screen-2026-07-24.json)
contains 526/526 label-free polar-energy records. Only after that artifact was
sealed were development labels used to produce
[`route1-gbr6-development-summary-2026-07-24.json`](route1-gbr6-development-summary-2026-07-24.json):

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| AM1-BCC/OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| AM1-BCC/CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |
| AM1-BCC/GBr6/cavity-dispersion | 2.257 | 3.451 | 20.820 |

All values are kcal/mol. GBr6 worsens mean absolute error by
`0.496 kcal/mol` relative to OBC-II/ACE (paired-gain CI
`[-0.681, -0.317]`) and by `0.935 kcal/mol` relative to CHA-GB
(CI `[-1.115, -0.758]`). It is rejected without fitting, and no runtime
provider or derivative code is added.

Reproduction:

```bash
GBR6_TMP=$(mktemp -d)
curl -L https://sourceforge.net/projects/gbr6/files/latest/download \
  -o "$GBR6_TMP/GBr6.tgz"
tar -xzf "$GBR6_TMP/GBr6.tgz" -C "$GBR6_TMP"
gfortran -O2 -std=legacy \
  -o "$GBR6_TMP/GBr6_v1/GBr6_v1.x" \
  "$GBR6_TMP/GBr6_v1/GBr6_v1.f"

python docs/implicit-solvation/benchmarks/run_gbr6_screen.py energy \
  --source-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --source-archive "$GBR6_TMP/GBr6.tgz" \
  --fortran-source "$GBR6_TMP/GBr6_v1/GBr6_v1.f" \
  --executable "$GBR6_TMP/GBr6_v1/GBr6_v1.x" \
  --output docs/implicit-solvation/benchmarks/route1-gbr6-energy-screen-2026-07-24.json

python docs/implicit-solvation/benchmarks/run_gbr6_screen.py summarize \
  --energy-artifact docs/implicit-solvation/benchmarks/route1-gbr6-energy-screen-2026-07-24.json \
  --source-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --high-level-record-dir .omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724/records \
  --output docs/implicit-solvation/benchmarks/route1-gbr6-development-summary-2026-07-24.json
```

### ddX/ddPCM derivative and development screen (2026-07-25)

[`ddx_pcm_protocol.json`](ddx_pcm_protocol.json) pins `pyddx 0.8.0`, its
LGPL-3.0 PyPI source archive hash, the Route 1 no-MM/no-residual/no-retraining
boundary, ddPCM numerical settings, fixed AM1-BCC charges, the pre-existing
ACE nonpolar component, label-use separation, and product admission gates.
`pyddx` is built only in an isolated audit environment and is not declared as a
MAPLE dependency.

The force artifact
[`route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json`](route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json)
contains both required native ddX derivative terms, the explicit sign/unit
conversion, all analytical forces, and centered finite differences for all 69
Cartesian components at `0.003 A` and `0.010 A`. The mbondi2 candidate's
primary-step RMSE/maximum are `0.000404/0.001504 kJ/mol/A`; its net force is
zero to numerical precision. The same artifact records the preselected
label-free 20-case convergence result:

| ddPCM candidate vs reference | MAE | P90 absolute | maximum absolute |
|---|---:|---:|---:|
| `lmax=7,nLebedev=194` vs `lmax=13,nLebedev=590` | 0.0618 | 0.1312 | 0.1820 |

All convergence values are kcal/mol and pass the frozen `0.15` P90 / `0.25`
maximum gate.

The label-blind energy artifact
[`route1-ddx-ddpcm-energy-screen-2026-07-25.json`](route1-ddx-ddpcm-energy-screen-2026-07-25.json)
contains 526/526 successful records. Development labels are opened only by the
separate summary
[`route1-ddx-ddpcm-development-summary-2026-07-25.json`](route1-ddx-ddpcm-development-summary-2026-07-25.json):

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| AM1-BCC/OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| AM1-BCC/fine APBS-mol/ACE | 1.629 | 2.462 | 14.116 |
| AM1-BCC/CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |
| AM1-BCC/ddPCM/mbondi2/ACE | 1.782 | 2.881 | 18.813 |
| AM1-BCC/ddPCM/(mbondi2+1.4 A)/ACE | 5.829 | 6.706 | 26.717 |

All values are kcal/mol. The mbondi2 candidate changes paired MAE relative to
OBC-II/ACE by `-0.022 kcal/mol`, with 95% interval `[-0.143, 0.098]`; it also
has worse RMSE and outliers. Its local `~0.53 s` polar-force call is roughly
`360x` the frozen `1.47 ms` warm OpenMM correction median on the same
23-atom molecule. Force and numerical gates pass, but accuracy, performance,
confirmation, and therefore product admission fail. No provider/default is
changed.

Reproduction from the repository root:

```bash
DDX_AUDIT=$(mktemp -d)
python -m venv --system-site-packages "$DDX_AUDIT/venv"
"$DDX_AUDIT/venv/bin/python" -m pip install pybind11==3.0.1
# Download pyddx-0.8.0.tar.gz from the protocol-pinned PyPI release and verify:
echo "31a1ddfe72105a0a6843ef2bcda76763cb99bab56d85d844c08e078156201f08  pyddx-0.8.0.tar.gz" \
  | sha256sum -c -
"$DDX_AUDIT/venv/bin/python" -m pip install --no-build-isolation \
  ./pyddx-0.8.0.tar.gz

"$DDX_AUDIT/venv/bin/python" \
  docs/implicit-solvation/benchmarks/run_ddx_pcm_force_probe.py --workers 6
"$DDX_AUDIT/venv/bin/python" \
  docs/implicit-solvation/benchmarks/run_ddx_pcm_screen.py energy --workers 6
python docs/implicit-solvation/benchmarks/run_ddx_pcm_screen.py score
```

### MLSES PB surface feasibility boundary (2026-07-26)

[`route1-mlses-pb-feasibility-probe-2026-07-26.json`](route1-mlses-pb-feasibility-probe-2026-07-26.json)
pins the local AmberTools/PBSA executable and topology hashes, the current
AmberTools 26 manual, the executed GENIUSES paper
(DOI `10.1021/acs.jpclett.3c02176`), and its predecessor MLSES paper
(DOI `10.1021/acs.jctc.1c00492`). AmberTools maps `sasopt=3, mlses_opt=0`
to GENIUSES; the original MLSES classifier is not the runtime executed here.
Both learn classical solvent-excluded-surface geometry rather than a
hydration-energy residual, so the model class remains inside the Route 1
physical boundary in principle.

The amended v2 probe first scans every `ENEOPT=1..4` /
`FRCOPT=1..5` input pairing under the fixed linear-PB setup. It identifies five
runtime-accepted force pairings: `1/1`, `2/2`, `2/3`, `2/4`, and `3/2`.
The force matrix then crosses those five pairings with both a coarse `0.50 A`
grid and the manual-style `0.25 A` grid. Every classical-SES control writes a
nonempty atom-force file. GENIUSES provides no atom-resolved MLSES force:
`1/1` terminates by signal, while the other four pairings abort during
dielectric-boundary force projection. Because no candidate force vector
exists, the independent finite-difference gate is not eligible and is not
executed.

Three standalone energy-only process repeats per surface and grid also show no
local small-molecule speed advantage:

| grid | classical SES median | GENIUSES median | classical/GENIUSES | GENIUSES minus classical PB energy |
|---|---:|---:|---:|---:|
| `0.50 A`, fill `1.25` | 0.12 s | 0.24 s | 0.500 | -0.0046 kcal/mol |
| `0.25 A`, fill `2.00` | 3.16 s | 3.79 s | 0.834 | +0.0912 kcal/mol |

These timings cover one 23-atom molecule on one CPU build and are not a
large-system or GPU performance claim. They nevertheless close the local
product-admission question: no MLSES runtime provider, dependency, FreeSolv
screen, or default change is introduced. Noncanonical binaries, topologies, or
repeat counts must use `--allow-other-inputs` together with an explicit
noncanonical `--output`; they cannot overwrite the canonical artifact, and
their decision is derived from their own force and timing observations.

Reproduction with the artifact-pinned local inputs:

```bash
python docs/implicit-solvation/benchmarks/run_mlses_pb_feasibility_probe.py
```

### Actual MAPLE SP/OPT/SCAN/MD task matrix (2026-07-25)

[`route1-task-matrix-methyl-hexanoate-2026-07-24.json`](route1-task-matrix-methyl-hexanoate-2026-07-24.json)
records twelve real `engine`/dispatcher jobs on one normalized 23-atom AM1-BCC
MOL2: SP, a two-iteration MAPLE LBFGS OPT, a three-point rigid MAPLE SCAN, and
a deterministic four-step NVT trajectory for MACE-OFF23m, AIMNet2, and ANI2x.
This differs from the earlier compatibility trace's manually displaced
geometry smoke: the SCAN records here are produced by MAPLE's `Scan`
dispatcher and its `*_scan_final.xyz` stream, while MD uses MAPLE's NVT
dispatcher and writes the normal thermo, trajectory, summary, final-geometry,
and restart outputs.

All twelve jobs retain the `results["solvation"]` decomposition and audit
manifest. The largest additive energy-closure residual is below
`3e-14 Hartree`; all three OPT jobs lower the same combined potential; all
three SCAN trajectories contain the requested three points; all three NVT jobs
write four finite thermodynamic rows and four coordinate frames; and the
fixed-geometry solvent energy has zero spread across the three gas models.
These results prove the named task/interface plumbing only. Four MD steps are
not equilibration or sampling evidence, and the artifact does not certify
broad optimization/scan/MD stability or hydration accuracy.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_route1_task_matrix.py \
  --mol2 .omx/benchmarks/route1-performance-20260724/mobley_1017962/m.mol2 \
  --charge-manifest docs/implicit-solvation/benchmarks/chagb_nonpolar_source_manifest.json \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model maceoff23m --model aimnet2 --model ani2x --device gpu0 \
  --openmm-platform Reference --opt-max-iter 2 \
  --scan-atoms 8 9 --scan-step 0.02 --scan-steps 2 \
  --md-steps 4 --md-timestep 0.05 --md-temperature 298.15 \
  --md-random-seed 20260725 \
  --work-dir .omx/benchmarks/route1-task-matrix-20260724/final4-md \
  --output docs/implicit-solvation/benchmarks/route1-task-matrix-methyl-hexanoate-2026-07-24.json
```

### Prebuilt explicit-inner / implicit-outer smoke (2026-07-24)

[`prebuilt_inner_outer_protocol.json`](prebuilt_inner_outer_protocol.json)
freezes the first runtime slice for a user-supplied solute-plus-inner-solvent
cluster. It requires a neutral closed-shell multicomponent MOL2, complete
bonds, fixed charges for every cluster atom, and
`#solv(implicit=water,inner=prebuilt,...)`. The coordinate-only automatic
explicit-solvent generator is not reused.

[`route1-prebuilt-inner-outer-smoke-2026-07-24.json`](route1-prebuilt-inner-outer-smoke-2026-07-24.json)
records nine real MAPLE engine/dispatcher jobs on one illustrative
methanol-plus-water cluster: SP, a two-iteration LBFGS OPT, and a three-point
rigid SCAN for MACE-OFF23m, AIMNet2, and ANI2x. Every job passes its frozen
topology, energy-decomposition, task-output, and audit checks. One
combined-potential force component per model also agrees with a centered
`0.003 A` finite difference: the absolute errors are `2.16e-5`, `4.17e-6`,
and `3.47e-5 Hartree/A`, all below the frozen `5e-5 Hartree/A` tolerance. At
the common SP geometry, all three models receive exactly the same
`-0.028742631009 Hartree` fixed-charge outer correction.

The artifact deliberately contains
`cluster_continuum_correction_hartree` and no `delta_g_solv_hartree`.
It identifies the result as a fixed-shell cluster-continuum configurational
potential and sets `absolute_solvation_free_energy_claim=false`. No FreeSolv
labels, gas-phase MM energy, fit, residual, or retraining are used. This smoke
does not calculate cluster formation/occupancy, standard-state,
solvent-cluster reference, or cluster ensemble terms and is therefore not a
hydration-accuracy benchmark.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_prebuilt_inner_outer_smoke.py \
  --protocol docs/implicit-solvation/benchmarks/prebuilt_inner_outer_protocol.json \
  --mol2 tests/solvation/data/prebuilt_inner_outer/methanol-water.mol2 \
  --work-dir .omx/benchmarks/route1-prebuilt-inner-outer-20260724 \
  --output docs/implicit-solvation/benchmarks/route1-prebuilt-inner-outer-smoke-2026-07-24.json \
  --model maceoff23m --model aimnet2 --model ani2x \
  --device gpu0 --openmm-platform Reference
```

### Automatic explicit-inner / implicit-outer feasibility audit (2026-07-25)

[`route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json`](route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json)
separates an inner-shell structure generator, a fixed-shell potential, and a
complete absolute-transfer free-energy cycle.

| candidate | useful capability | blocking boundary |
|---|---|---|
| Route 1 `inner=prebuilt` | multi-MLIP SP/OPT/SCAN energy and conservative-force composition on a supplied cluster | no shell occupancy, pure-solvent reference, standard-state, or cluster ensemble |
| CREST QCG | solute-solvent and reference-solvent ensembles plus conformational and thermochemical free-energy terms | current QCG source directly uses xTB/GFN-family methods; its separate generic CREST backend is not connected to QCG |
| FEBISS | GIST-based solvent-site ranking and selected microsolvated structures from a supplied trajectory | proposal preprocessor only; not an absolute-transfer free-energy engine |
| ensemble cluster-continuum literature | establishes conformer/cluster-size convergence and uncertainty requirements | evaluated primarily for ions; no arbitrary-MLIP neutral FreeSolv product validation |

The current CREST main source was frozen at commit
`cfdc301f759686b0fd66ced63b5ddbd6c693fa4f`. Its five QCG source files contain
no generic-backend call or calculation object; QCG instead constructs direct
xTB single-point, optimization, and `--hess`/`--ohess` commands. The audited
FEBISS `2.0` source was frozen at commit
`4ca103d1a57b0c179d8c4bce066371ad76ad00f7`.

No automatic explicit-inner runtime, dependency, FreeSolv selector, accuracy
claim, or faster-than-MM claim was added. The only admissible future experiment
is a complete atom-conserving cycle in which one target MLIP evaluates the
solute, pure solvent, and all clusters; proposal-to-target overlap,
thermochemistry, standard states, conformer/coordination/cluster-size
convergence, neutral aqueous accuracy, and end-to-end cost must all pass
prospectively frozen gates.

### Frozen conformer sensitivity (2026-07-23)

The independent
[`conformer_protocol.json`](conformer_protocol.json) freezes 20 development-only
cases by SHA-256 rank within the predeclared flexibility strata: four rigid,
eight limited, and eight flexible molecules.  CREST 3.0.2 iMTD-GC/quick with
GFN2-xTB 6.7.1 and ALPB(water) generated 1,294 low-energy conformers; all 20
generator cases completed.  The exact method/case statistics and record hashes
are in
[`freesolv-conformer-sensitivity-2026-07-23.json`](freesolv-conformer-sensitivity-2026-07-23.json).

| fixed-charge method | all-case median range | all-case p90 range | all-case maximum range | flexible-case p90 range | maximum displacement from reference |
|---|---:|---:|---:|---:|---:|
| ABCG2 / OBC-II / ACE | 0.317 | 1.791 | 5.328 | 4.277 | 6.083 |
| AM1-BCC / OBC-II / ACE | 0.295 | 1.390 | 5.956 | 4.278 | 5.184 |

All values are `kcal/mol` changes in the fixed-charge GB/ACE correction.  The
rigid controls each produced one conformer; the flexible cases produced a
median of 134.5 and as many as 309 conformers.  The large flexible-tail changes
show that the single FreeSolv geometry cannot be silently interpreted as a
population-averaged hydration free energy.  No CREST or arithmetic weighting
is applied, because that would require a separately frozen partition-function
and conformational-entropy protocol.

### MLIP discrete-conformer weighting diagnostic (2026-07-24)

The development-only
[`mlip_conformer_weighting_protocol.json`](mlip_conformer_weighting_protocol.json)
pins MACE-OFF23 medium checkpoint
`ac172fdf9b5173fef4c64667739dbd06f230b0167b4ae2c67e8c08033254c9ee`
and combines its gas-phase relative single-point energies with the existing
ABCG2/OBC-II/ACE conformer corrections through a discrete partition-function
ratio. The exact result and per-case record hashes are in
[`freesolv-mlip-conformer-weighting-2026-07-24.json`](freesolv-mlip-conformer-weighting-2026-07-24.json).

The 1,294 frozen CREST conformers plus four non-duplicate FreeSolv reference
geometries produced 1,298 discrete states. On the 20 selected development
cases, the reference-geometry MAE/RMSE of `1.983/2.317 kcal/mol` changed to
`1.799/2.120 kcal/mol`; 10 cases improved, 6 worsened, and the four rigid
single-state controls were unchanged. The 8 flexible cases changed from
`1.763/2.167` to `1.433/1.724 kcal/mol`.

This apparent improvement is not yet robust evidence of a general MLIP
accuracy gain. The largest shift is alachlor (`+5.063 kcal/mol` relative to
the reference endpoint); leave-one-out removal of that case reduces the
overall MAE improvement from `0.184` to `0.059 kcal/mol`. The calculation is
also a single-point, equal-basin discrete-state approximation on conformers
generated with GFN2-xTB/ALPB(water), not MACE-relaxed gas/solution ensembles or
lambda-window MBAR.

On an RTX 4060 Laptop GPU, all 1,298 state evaluations took `59.304 s`
(`21.887 states/s`). Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_conformer_weighting.py run \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_weighting_protocol.json \
  --source-output-dir .omx/benchmarks/conformer-sensitivity-route1-20260723-v2 \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output-dir .omx/benchmarks/mlip-conformer-weighting-route1-20260724 \
  --device cuda

python docs/implicit-solvation/benchmarks/run_mlip_conformer_weighting.py summarize \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_weighting_protocol.json \
  --output-dir .omx/benchmarks/mlip-conformer-weighting-route1-20260724 \
  --output docs/implicit-solvation/benchmarks/freesolv-mlip-conformer-weighting-2026-07-24.json
```

### MLIP gas/solution relaxation diagnostic (2026-07-24)

[`mlip_conformer_relaxation_protocol.json`](mlip_conformer_relaxation_protocol.json)
freezes six stratified development cases and up to three starts per case: the
exact FreeSolv reference state plus the gas- and solution-dominant states from
the first MLIP diagnostic, with duplicate indices removed. Each start is
minimized both with gas-phase MACE-OFF23 medium and with
MACE+fixed-ABCG2/OBC-II/ACE. ASE LBFGS uses `fmax=0.03 eV/A`,
`maxstep=0.1 A`, and at most 300 steps; a tested unit bridge converts MAPLE's
Hartree and Hartree/A outputs to ASE eV units.

The exact summary and per-case record hashes are in
[`freesolv-mlip-conformer-relaxation-2026-07-24.json`](freesolv-mlip-conformer-relaxation-2026-07-24.json).
All 13 gas and 13 solution branches converged.

| case | experiment | reference endpoint | discrete MLIP weighting | relaxed transfer | gas reorganization | solvent term at solution minimum |
|---|---:|---:|---:|---:|---:|---:|
| alachlor | -8.210 | -12.016 | -6.953 | -6.885 | +1.106 | -7.992 |
| P/S flexible case | -10.030 | -12.290 | -12.358 | -12.178 | +0.232 | -12.410 |
| pentylbenzene | -0.230 | +1.574 | +1.415 | +1.464 | +0.005 | +1.459 |
| pentan-1-ol | -4.570 | -3.917 | -3.970 | -4.176 | +0.101 | -4.277 |
| isobutyl nitrate | -1.880 | -3.784 | -3.486 | -3.556 | +0.057 | -3.613 |
| nitromethane | -4.020 | -8.579 | -8.579 | -8.027 | +0.073 | -8.100 |

All values are `kcal/mol`. On this deliberately small diagnostic subset,
reference-endpoint, discrete-weighting, and relaxed-transfer MAE/RMSE are
`2.498/2.819`, `1.999/2.360`, and `1.874/2.170`, respectively. Relaxation
improves only 3/6 cases relative to discrete weighting; its MAE gain is
`0.125 kcal/mol`, or `0.164 kcal/mol` after removing alachlor. Thus the extra
MLIP force information contributes a measurable but modest conformer term and
is not merely duplicating the endpoint GB calculation.

The strongest mechanistic signal is alachlor: the selected gas and solution
minima differ by `1.243 A` aligned heavy-atom RMSD, and the solution conformation
costs `1.106 kcal/mol` on the gas MACE surface. The relaxed result remains close
to the first-stage weighted value, so the earlier large alachlor correction is
not eliminated by relaxing the xTB geometries. Conversely, nitromethane still
has a `4.007 kcal/mol` absolute error despite having only one rigid start. That
residual cannot be fixed by better intramolecular conformer energies; it points
to the charge/GB/nonpolar thermodynamic model.

This is a separately minimized **0 K potential-energy diagnostic**, not a
finite-temperature hydration free energy. The P/S case retains a
`2.582 kcal/mol` solution-minimum span across its two starts, and alachlor
retains a `3.043 kcal/mol` gas-minimum span across three starts. Broader
MACE-relaxed basin coverage is therefore required before a frozen
lambda-window sampling/MBAR calculation.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_conformer_relaxation.py run \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_relaxation_protocol.json \
  --weighting-output-dir .omx/benchmarks/mlip-conformer-weighting-route1-20260724 \
  --conformer-output-dir .omx/benchmarks/conformer-sensitivity-route1-20260723-v2 \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output-dir .omx/benchmarks/mlip-conformer-relaxation-route1-20260724 \
  --device cuda

python docs/implicit-solvation/benchmarks/run_mlip_conformer_relaxation.py summarize \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_relaxation_protocol.json \
  --output-dir .omx/benchmarks/mlip-conformer-relaxation-route1-20260724 \
  --output docs/implicit-solvation/benchmarks/freesolv-mlip-conformer-relaxation-2026-07-24.json
```

### Forward-default AM1-BCC conformer repeats (2026-07-24)

The historical ABCG2 artifacts above remain immutable. New forward work uses
AM1-BCC by default, so the same frozen conformers and MACE checkpoint were
re-evaluated under
[`mlip_conformer_weighting_am1bcc_protocol.json`](mlip_conformer_weighting_am1bcc_protocol.json)
and
[`mlip_conformer_relaxation_am1bcc_protocol.json`](mlip_conformer_relaxation_am1bcc_protocol.json).
Exact summaries are stored in
[`freesolv-mlip-conformer-weighting-am1bcc-2026-07-24.json`](freesolv-mlip-conformer-weighting-am1bcc-2026-07-24.json)
and
[`freesolv-mlip-conformer-relaxation-am1bcc-2026-07-24.json`](freesolv-mlip-conformer-relaxation-am1bcc-2026-07-24.json).

The result does not justify immediately scaling MBAR:

| AM1-BCC diagnostic | baseline MAE/RMSE | MLIP result MAE/RMSE | outcome |
|---|---:|---:|---|
| 20-case discrete conformer weighting | 2.227 / 2.829 | 2.291 / 2.925 | worse |
| 6-case separately relaxed transfer | 2.582 / 3.510 | 2.834 / 3.760 | worse |

All 24 AM1-BCC relaxation branches converged. The negative result is therefore
not caused by failed optimization. AM1-BCC substantially improves some rigid
charge failures such as nitromethane, but the endpoint and conformer errors are
chemistry dependent; the large P/S case remains badly over-stabilized.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_conformer_weighting.py run \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_weighting_am1bcc_protocol.json \
  --source-output-dir .omx/benchmarks/conformer-sensitivity-route1-20260723-v2 \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output-dir .omx/benchmarks/mlip-conformer-weighting-am1bcc-route1-20260724 \
  --device cuda

python docs/implicit-solvation/benchmarks/run_mlip_conformer_weighting.py summarize \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_weighting_am1bcc_protocol.json \
  --output-dir .omx/benchmarks/mlip-conformer-weighting-am1bcc-route1-20260724 \
  --output docs/implicit-solvation/benchmarks/freesolv-mlip-conformer-weighting-am1bcc-2026-07-24.json

python docs/implicit-solvation/benchmarks/run_mlip_conformer_relaxation.py run \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_relaxation_am1bcc_protocol.json \
  --weighting-output-dir .omx/benchmarks/mlip-conformer-weighting-am1bcc-route1-20260724 \
  --conformer-output-dir .omx/benchmarks/conformer-sensitivity-route1-20260723-v2 \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output-dir .omx/benchmarks/mlip-conformer-relaxation-am1bcc-route1-20260724 \
  --device cuda

python docs/implicit-solvation/benchmarks/run_mlip_conformer_relaxation.py summarize \
  --protocol docs/implicit-solvation/benchmarks/mlip_conformer_relaxation_am1bcc_protocol.json \
  --output-dir .omx/benchmarks/mlip-conformer-relaxation-am1bcc-route1-20260724 \
  --output docs/implicit-solvation/benchmarks/freesolv-mlip-conformer-relaxation-am1bcc-2026-07-24.json
```

### Reusable discrete-conformer core replay (2026-07-25)

The discrete partition-ratio estimator has been moved out of the
MACE-specific benchmark runner into
`maple.function.free_energy.analyze_discrete_conformer_ensemble`. The core
accepts only same-state gas energies, fixed-charge solvent corrections,
temperature, and optional state identifiers. It neither loads an MLIP nor
generates conformers, so any registered gas MLIP can supply the energy array
without changing the estimator.

The self-hashed
[`route1-discrete-conformer-core-replay-2026-07-25.json`](route1-discrete-conformer-core-replay-2026-07-25.json)
replays both historical charge-method record sets through the extracted core:

| historical MACE-OFF23m method | records | states | default weight diagnostic passes | overlap minimum / median / maximum |
|---|---:|---:|---:|---:|
| ABCG2 / OBC-II | 20 | 1,298 | 14/20 | 0.475648 / 0.946189 / 1.000000 |
| AM1-BCC / OBC-II | 20 | 1,298 | 14/20 | 0.338870 / 0.942454 / 1.000000 |

All 11 fields shared by the historical implementation and new core reproduce
with exactly zero numerical difference across all 40 records and 2,596 states.
The diagnostic gate requires at least two states, gas and solution effective
counts of at least two, maximum weights no greater than `0.95`, overlap of at
least `0.10`, and the endpoint bound. It detects obvious state-set
concentration but cannot certify basin coverage.

The historical input records contain experimental values because they predate
the extracted core; the core signature and replay metrics do not access those
fields. This replay is consequently extraction/parity evidence, not a residual
fit. It is also MACE-only named evidence: MLIP-agnostic input semantics do not
by themselves establish multi-MLIP chemical accuracy. Equal discrete-state
measure omits basin volumes, vibrational free energies, uncertainty,
standard-state handling, and equilibrated overlap, so the public `#solvfe`
surface remains closed.

Reproduction from the retained historical record directories:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_conformer_weighting.py audit-core \
  --record-dir .omx/benchmarks/mlip-conformer-weighting-route1-20260724/records \
  --record-dir .omx/benchmarks/mlip-conformer-weighting-am1bcc-route1-20260724/records \
  --output docs/implicit-solvation/benchmarks/route1-discrete-conformer-core-replay-2026-07-25.json
```

### Prospective common-state multi-MLIP discrete diagnostic (2026-07-25)

[`multi_mlip_discrete_conformer_protocol.json`](multi_mlip_discrete_conformer_protocol.json)
freezes MACE-OFF23m, AIMNet2, and ANI2x checkpoints before the run. MAPLE now
extracts a common `atomic_numbers` table from each loaded checkpoint and
rejects unsupported input elements before the first model evaluation. The
protocol retains every original conformer case inside the intersection
`H/C/N/O/F/S/Cl`: 19 cases and 1,270 states remain, while the single
phosphorus case is excluded by model compatibility rather than by a label.

The label-free
[`multi_mlip_discrete_conformer_source_manifest.json`](multi_mlip_discrete_conformer_source_manifest.json)
pins the normalized `float64` coordinate arrays, original CREST/MOL2/record
hashes, the historical MACE energy arrays, and one identical
AM1-BCC/OBC-II/ACE correction array per case. The state archives occupy about
`0.91 MB` under
[`route1-multi-mlip-discrete-conformer-raw/`](route1-multi-mlip-discrete-conformer-raw/).
No conformer is regenerated and no experimental value appears in the prepared
manifest or energy artifact.

All 57 model/case evaluations complete. Each model traverses the full state set
twice on the same RTX 4060 Laptop GPU:

| gas MLIP | primary states/s | weight-diagnostic passes | median gas-vs-solution overlap |
|---|---:|---:|---:|
| MACE-OFF23m | 20.29 | 13/19 | 0.9443 |
| AIMNet2 | 40.56 | 4/19 | 0.9963 |
| ANI2x | 44.49 | 12/19 | 0.9471 |

The low AIMNet2 pass count is not a contradiction with its high
gas-vs-solution overlap: many AIMNet2 gas and solution distributions are
similarly concentrated on very few submitted states, so the effective-count
and maximum-weight gates fail.

Across-model sensitivity separates agreement of the final partition ratio from
agreement of the underlying populations:

| MLIP pair | median / p90 / maximum absolute correction difference | median gas / solution weight overlap |
|---|---:|---:|
| MACE-OFF23m / AIMNet2 | 0.01187 / 0.10509 / 1.16717 | 0.2522 / 0.2564 |
| MACE-OFF23m / ANI2x | 0.01291 / 0.11783 / 0.24303 | 0.8022 / 0.7922 |
| AIMNet2 / ANI2x | 0.03523 / 0.11461 / 1.41020 | 0.2141 / 0.2013 |

Differences are in `kcal/mol`. Across all three models, the per-case correction
range has median/p90 `0.03597/0.11969 kcal/mol`; alachlor is the
`1.41020 kcal/mol` maximum. Similar final values therefore cannot be used as
evidence that the MLIPs sampled the same conformer populations.

The predeclared `1e-6 kcal/mol` repeat-relative-energy gate is deliberately
left failed. Float32 CUDA backends reach `7.59e-5 kcal/mol` in relative state
energy, although the maximum propagated correction difference is only
`3.93e-6 kcal/mol`; the gate was not relaxed after inspection. MACE reproduces
the old relative energies within `7.14e-10 kcal/mol`. The sealed label-free
artifact is
[`route1-multi-mlip-discrete-conformer-2026-07-25.json`](route1-multi-mlip-discrete-conformer-2026-07-25.json).

Only after sealing that artifact was the development score written to
[`route1-multi-mlip-discrete-conformer-score-2026-07-25.json`](route1-multi-mlip-discrete-conformer-score-2026-07-25.json):

| endpoint | MAE | RMSE | paired MAE gain vs fixed geometry | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| fixed AM1-BCC/OBC-II/ACE geometry | 1.930 | 2.273 | — | — |
| AIMNet2 discrete weighting | 1.910 | 2.246 | +0.020 | -0.024 to +0.063 |
| MACE-OFF23m discrete weighting | 1.969 | 2.297 | -0.038 | -0.191 to +0.062 |
| ANI2x discrete weighting | 1.993 | 2.324 | -0.063 | -0.240 to +0.051 |

Every interval crosses zero. This is multi-MLIP compatibility and sensitivity
evidence, not an accuracy improvement, model-selection rule, equilibrated
hydration free energy, or reason to open `#solvfe`.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_multi_mlip_discrete_conformers.py prepare \
  --protocol docs/implicit-solvation/benchmarks/multi_mlip_discrete_conformer_protocol.json \
  --conformer-record-dir .omx/benchmarks/conformer-sensitivity-route1-20260723-v2/records \
  --weighting-record-dir .omx/benchmarks/mlip-conformer-weighting-am1bcc-route1-20260724/records \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --raw-dir docs/implicit-solvation/benchmarks/route1-multi-mlip-discrete-conformer-raw \
  --output docs/implicit-solvation/benchmarks/multi_mlip_discrete_conformer_source_manifest.json

python docs/implicit-solvation/benchmarks/run_multi_mlip_discrete_conformers.py run \
  --protocol docs/implicit-solvation/benchmarks/multi_mlip_discrete_conformer_protocol.json \
  --manifest docs/implicit-solvation/benchmarks/multi_mlip_discrete_conformer_source_manifest.json \
  --work-dir .omx/benchmarks/multi-mlip-discrete-conformer-route1-20260725 \
  --output docs/implicit-solvation/benchmarks/route1-multi-mlip-discrete-conformer-2026-07-25.json

python docs/implicit-solvation/benchmarks/run_multi_mlip_discrete_conformers.py score \
  --protocol docs/implicit-solvation/benchmarks/multi_mlip_discrete_conformer_protocol.json \
  --energy-artifact docs/implicit-solvation/benchmarks/route1-multi-mlip-discrete-conformer-2026-07-25.json \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output docs/implicit-solvation/benchmarks/route1-multi-mlip-discrete-conformer-score-2026-07-25.json
```

### Native multi-MLIP conformer batch parity (2026-07-25)

[`multi_mlip_conformer_batch_parity_protocol.json`](multi_mlip_conformer_batch_parity_protocol.json)
reuses the sealed 19-case/1,270-state label-free manifest and freezes one
common CUDA batch size (`16`), four balanced serial/batch repeats, a
`0.001 kcal/mol` numerical-equivalence limit, and a prospective `1.25x`
per-model material-speedup floor. A disclosed one-case pilot selected the
batch size; the relative-energy, propagated-correction, and speed gates were
fixed before the initial full run. Review of that run then exposed a missing
absolute-energy admission gate plus warmup, provenance, and fail-closed
decision defects. Protocol v2 adds the same `0.001 kcal/mol` tolerance
converted to Hartree for absolute energies and repairs those contracts before
the confirmatory rerun. A preserved
[`v2 review run`](route1-multi-mlip-conformer-batch-parity-v2-review-run-2026-07-25.json)
then showed a MACE ratio-of-medians above `1.25x`, while the paired repeats
disagreed at `1.060x` and `1.619x`. Protocol v3 therefore requires every one
of four balanced paired repeats to clear the unchanged speed floor. A later
provenance refresh overlapped another CUDA workload and produced a 174 s MACE
batch repeat, so it was discarded. Protocol v5 added fail-closed GPU endpoint
screens: any competing GPU compute process at preflight/postflight, or a
preflight one-minute host load above `0.25` per logical CPU, aborts the
benchmark without writing an artifact. Protocol v6 makes the evidence boundary
explicit: endpoint snapshots do not prove whole-run GPU exclusivity, continuous
monitoring was not performed, and timings remain backend-specific conditional
evidence. v2 through v6 are review-amended confirmation, not wholly pre-data
protocols.

The common `CalcABC.calculate_many` API returns a `BatchResult`; its base
implementation is sequential. Three named adapters provide model-native paths:
MACE-OFF23m builds a disconnected molecular graph, AIMNet2 masks dense
neighbors by `mol_idx`, and ANI2x groups identical element orderings. The
fixed-topology `evaluate_gas_conformer_energies` helper chunks these calls and
rejects an attached solvent correction, so this benchmark changes only
gas-energy evaluation and never adds a gas MM term or a learned residual.

The sealed artifact
[`route1-multi-mlip-conformer-batch-parity-2026-07-25.json`](route1-multi-mlip-conformer-batch-parity-2026-07-25.json)
reports:

| gas MLIP | serial states/s | batch states/s | median speedup | max relative-energy difference | max discrete-correction difference |
|---|---:|---:|---:|---:|---:|
| MACE-OFF23m | 16.01 | 15.82 | 0.99x | `7.13e-10` | `2.19e-11` |
| AIMNet2 | 16.61 | 23.64 | 1.42x | `2.86e-4` | `3.48e-5` |
| ANI2x | 12.48 | 154.63 | 12.39x | `7.48e-5` | `2.09e-6` |

Relative-energy and discrete-correction differences are in `kcal/mol`; the
separate absolute-energy gate is stored in Hartree. All numerical gates pass,
so the common batch interface is admitted. MACE has paired repeats below the
unchanged `1.25x` floor (`0.616x` minimum); AIMNet2 and ANI2x pass all four
repeats (`1.283x` and `11.298x` minima). The universal batch acceleration
claim is therefore forbidden, while the named AIMNet2 and ANI2x
material-speedup claims are admitted. These ratios compare each MLIP with its
own serial path; they are not speed-versus-MM or solvent-accuracy evidence.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_multi_mlip_conformer_batch_parity.py \
  --protocol docs/implicit-solvation/benchmarks/multi_mlip_conformer_batch_parity_protocol.json \
  --work-dir .omx/benchmarks/route1-multi-mlip-conformer-batch-parity-20260725 \
  --output docs/implicit-solvation/benchmarks/route1-multi-mlip-conformer-batch-parity-2026-07-25.json
```

### Phase-specific multi-MLIP selected-minimum RRHO pilot (v6, falsified)

[`multi_mlip_phase_specific_qrrho_protocol.json`](multi_mlip_phase_specific_qrrho_protocol.json)
freezes a development-only falsification pilot. It does not
claim an exact hydration partition function. The label-blind source rule targets
state counts `[1,2,8,20,40,150]`, selects without replacement by nearest
log-state-count under a 23-atom ceiling, and tie-breaks by atom count and
compound ID. This yields six cases (221 frozen source states) and 18
model/case records across MACE-OFF23m, AIMNet2, and ANI2x.

For every source state the runner evaluates the gas MLIP twice, adds the same
frozen AM1-BCC/OBC-II/ACE correction for solution ranking, and selects up to
three heavy-atom-RMSD-diverse seeds independently per phase. Gas and solution
are optimized on `E_MLIP,gas` and `E_MLIP,gas + W_AM1-BCC/OBC-II/ACE`,
respectively. Every selected branch must converge and then resolve as one
valid unique minimum or an explicitly recorded duplicate; failed or
nonconverged branches are never silently removed.

The v6 primary is **local harmonic RRHO** (`ilowfreq=0`). Grimme entropy-only
and Otlyotov--Minenkov qRRHO at 50, 100, and 150 cm-1 are sensitivity-only:
explicitly summing selected minima and also assigning free-rotor entropy to
the same torsional coordinate can double count configurational entropy. Any
selected negative vibrational frequency invalidates the minimum; no negative
mode is replaced by its absolute value. The runner freezes the `Imin/Imax <
0.01` linearity rule, requires the projected rigid complement to remain below
`1 cm-1`, and preregisters a `0.001/0.002/0.004 angstrom` Hessian-displacement
preflight on both the rigid nitromethane plumbing anchor and flexible
1-acetoxyethyl acetate. This targeted pair does not prove numerical stability
for every flexible molecule. Its 0.002-angstrom reference is required to be
the exact primary Hessian record, not merely another internally consistent
Hessian at the same nominal displacement.

At identical 1 M gas and solution conventions, common center-of-mass
translation, momentum factors, and the global orientation-group volume cancel,
so no approximately `1.89 kcal/mol` 1-atm-to-1-M correction is added. The
geometry-dependent local `Q_rot,j/sigma_rot,j` is nevertheless retained in each
basin, with per-case rotational symmetry numbers frozen separately from
conformer degeneracy. This makes the estimator a **selected-minimum local-RRHO
surrogate**, not the exact Cartesian endpoint ratio. Unit weights are a
lower-information selected-basin approximation; duplicate arrival counts are
not degeneracies or basin volumes.

The shared v6 semantic validator is used by resume, durable sealing, and
post-seal scoring. Records and every primary or sensitivity Hessian are copied
from `.omx` scratch into repository-contained, content-addressed evidence before
labels can be opened. All 18 records must pass. A pass only permits a separately
preregistered larger seed-budget and held-out study; it does not open public
`#solvfe`, prove conformer completeness, or validate an absolute hydration
method.

The pilot stopped on its first record, ANI2x/nitromethane. Gas and solution
LBFGS optimizations converged below `0.009 eV/A`, but the frozen `0.002 A`
central-difference Hessians had raw asymmetries of
`1.214e-3/1.130e-3 Hartree/A^2`, more than one thousand times the `1e-6`
gate. The record therefore failed before thermochemistry or label scoring.
Post-failure label-blind diagnostics found valid stationary spectra after
symmetrization and only `1.10/1.06 cm-1` maximum selected-mode RMS changes
across `0.001/0.002/0.004 A`. This separates a non-dtype-aware raw-asymmetry
gate from optimizer failure, but cannot rescue v6.

The exact failure record, optimization logs/trajectories, frozen gates, and
post-failure diagnostic are sealed in
[`route1-multi-mlip-phase-specific-selected-minimum-rrho-v6-failure-2026-07-25.json`](route1-multi-mlip-phase-specific-selected-minimum-rrho-v6-failure-2026-07-25.json).
V6 must not be resumed, have its failure record replaced, be sealed, or be
scored. Any successor first needs a separately versioned label-blind
numerical-Hessian qualification.

### V8 numerical qualification and rigid three-MLIP preflight

V8 is a separate protocol amendment, not a v6 rescue. The durable
[`route1-hessian-numerical-qualification-2026-07-25.json`](route1-hessian-numerical-qualification-2026-07-25.json)
uses no experimental labels and covers all three MLIPs, gas and solution, and
the `0.001/0.002/0.004 A` displacement set on nitromethane. The observed
maxima were `1.765e-3 Hartree/A^2` raw absolute asymmetry, `7.368e-4`
relative Frobenius asymmetry, and `1.097 cm-1` selected-mode RMS shift. V8
therefore uses a broad `0.02 Hartree/A^2` nonconservative-force sanity ceiling,
a scale-aware `0.002` relative Frobenius gate, and retains the independent
`25 cm-1` displacement-frequency gate.

AIMNet2 also exposed a separate scientific issue: LBFGS met the `0.01 eV/A`
force threshold at a roughly `-61 cm-1` first-order saddle. V8 preregisters
one recovery cycle: displace both signs of the most negative selected mode with
a maximum per-atom displacement of `0.5 A`, reoptimize the unchanged phase
potential, accept the lower converged result only when it lowers the energy by
at least `0.001 kcal/mol`, then repeat all Hessian, frequency, and deduplication
checks. MACE-OFF23m and ANI2x required no recovery.

The first v7 three-model preflight succeeded but was durably invalidated after
a behavior-preserving evidence-path type guard changed the frozen runner
bytes. Its audit is
[`route1-multi-mlip-phase-specific-selected-minimum-rrho-v7-preflight-invalidated-2026-07-25.json`](route1-multi-mlip-phase-specific-selected-minimum-rrho-v7-preflight-invalidated-2026-07-25.json);
its records are not reused. The independently frozen v8 rerun completed all
three nitromethane records in `41.55 s` wall time with `2.10 GB` maximum RSS.
Every final selected frequency is positive and every numerical gate passes.
This is 3/18 label-free records, not a sealed aggregate or an accuracy result.
The interim records and raw Hessians are durably mirrored in
[`route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-rigid-preflight-2026-07-25.json`](route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-rigid-preflight-2026-07-25.json);
that audit does not substitute for the required 18-record aggregate seal.
The next gate is the flexible `mobley_4463913` three-MLIP preflight.

The active frozen protocol remains independently validatable:

```bash
python docs/implicit-solvation/benchmarks/run_multi_mlip_phase_specific_qrrho.py validate

python -m pytest -q \
  tests/solvation/test_multi_mlip_phase_specific_qrrho_protocol.py
```

### Global multi-MLIP optimizer qualification v2 (2026-07-26)

The flexible v8 preflight failed before Hessian construction when one selected
AIMNet2 gas branch exhausted the frozen 500-step LBFGS budget at
`0.138705 eV/A`. Its self-hashed failure artifact forbids resume, record
replacement, aggregate sealing, and label scoring:
[`route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-flexible-preflight-failure-2026-07-25.json`](route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-flexible-preflight-failure-2026-07-25.json).

The separately frozen optimizer qualification asks only whether one
model-neutral policy can relax the exact three selected flexible-case source
states in gas and AM1-BCC/OBC-II/ACE solution for MACE-OFF23m, AIMNet2, and
ANI2x. It reads no hydration labels and permits neither per-model nor
per-phase selection. V1 was interrupted after bounding optimizer steps but not
line-search calculator evaluations; its partial records are retained solely
for interruption audit and are not reused. V2 reruns every branch from its
exact source coordinates and adds one common 1,000-evaluation ceiling.

The sealed result is
[`route1-multi-mlip-optimizer-qualification-v2-2026-07-26.json`](route1-multi-mlip-optimizer-qualification-v2-2026-07-26.json):

| global candidate | branches passing | calculator evaluations | successful-branch optimizer steps | wall time |
|---|---:|---:|---:|---:|
| BFGSLineSearch | 12/18 | 8,180 | 640 | 309.31 s |
| LBFGSLineSearch | 12/18 | 8,502 | 845 | 309.88 s |
| FIRE2/ABC | 12/18 | 7,601 | 6,134 | 365.36 s |

The frozen v2 exception schema records calculator evaluations and elapsed time
but omits `optimizer.nsteps` on failed branches. The displayed step column is
therefore the sum for successful branches only, not a total-cost metric; the
failed-branch logs contain additional steps. This omission does not affect the
`12/18` denominator or null global-policy decision. The frozen candidate
`basis` strings also say that the v8 500-step ceiling is retained, but the
normative and executed v2 gate is `maximum_steps=1000`. That wording is a
protocol-provenance erratum and is not repaired in place because changing the
frozen protocol bytes would invalidate its fingerprint and sealed records.

Both line-search policies fail all six AIMNet2 branches through evaluation
exhaustion or another frozen convergence failure. FIRE2/ABC reaches the force
threshold for five of those branches but violates the predeclared
non-increasing-energy gate, and the sixth exhausts the evaluation ceiling.
Thus no candidate passes every branch, `selected_global_policy` is null, and
the terminal status is `failed-closed-no-global-policy`.

This result closes the v9 escalation. It is not a hydration-accuracy result,
does not invalidate the Route 1 SP/OPT/SCAN composition, and does not license
per-model optimizer tuning, v8 state reuse, Hessian continuation, or label
scoring. Validation:

```bash
python docs/implicit-solvation/benchmarks/run_multi_mlip_optimizer_qualification.py validate
python -m pytest -q tests/solvation/test_multi_mlip_optimizer_qualification.py
```

### Common-state CHA-GB/PBSA discrete-conformer screen (2026-07-25)

[`multi_mlip_chagb_discrete_conformer_protocol.json`](multi_mlip_chagb_discrete_conformer_protocol.json)
prospectively asks whether the more accurate fixed-charge solvent provider and
the already frozen MLIP-dependent conformer weights reinforce each other
strongly enough to justify expensive long-trajectory sampling. The target
remains exactly

```text
E_solution(R) =
    E_MLIP,gas(R)
  + G_CHA-GB(R, q_AM1-BCC)
  + G_PBSA,cavity+dispersion(R)
```

The energy phase reads no experimental labels, reuses the sealed gas MLIP
energies without modification, and evaluates the high solvent endpoint twice
for every one of the 1,270 common states. It makes 2,540 provider calls for
57 model/case records. The repeats take `72.63` and `72.05 s` locally
(`145.51 s` total), and both the endpoint arrays and propagated discrete free
energies agree exactly. The sealed label-free artifact is
[`route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json`](route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json).

Only after sealing that artifact does the score phase compare against
development labels:

| gas MLIP | fixed high-endpoint MAE | discrete high-endpoint MAE | paired MAE gain vs fixed | 95% interval | improved / unchanged / worsened | weight passes |
|---|---:|---:|---:|---:|---:|---:|
| AIMNet2 | 1.700 | 1.699 | +0.001 | -0.148 to +0.173 | 6 / 4 / 9 | 4/19 |
| ANI2x | 1.700 | 1.720 | -0.020 | -0.087 to +0.047 | 7 / 4 / 8 | 12/19 |
| MACE-OFF23m | 1.700 | 1.709 | -0.009 | -0.073 to +0.056 | 7 / 4 / 8 | 13/19 |

All values are kcal/mol. The high discrete MAEs are `0.211-0.273 kcal/mol`
better than their low OBC-II/ACE discrete counterparts, but this is the
fixed-charge provider change: no model improves materially or significantly
over the same high endpoint at the reference geometry. Fewer than the frozen
60% of cases improve, AIMNet2 also fails the single-case-concentration gate,
and no model passes every discrete-weight diagnostic. The score artifact
[`route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json`](route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json)
therefore records `long_sampling_candidate_not_supported`. This is a
prospective negative allocation decision, not a hydration-free-energy or
universal no-conformer-effect claim.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_multi_mlip_chagb_discrete_conformers.py energy
python docs/implicit-solvation/benchmarks/run_multi_mlip_chagb_discrete_conformers.py score
```

### Two-level OBC-II OPT -> CHA-GB final-SP diagnostic (2026-07-24)

[`two_level_final_sp_protocol.json`](two_level_final_sp_protocol.json) freezes a
strict two-potential construction:

```text
DeltaE_2L =
  min_i [E_MLIP,gas(R_solution,low,i) + G_high(R_solution,low,i)]
  - min_j E_MLIP,gas(R_gas,low,j)
```

The low potential is MACE-OFF23m plus fixed-AM1-BCC/OBC-II/ACE and supplies
consistent optimization forces. The high solvent endpoint is
fixed-AM1-BCC/CHA-GB/PBSA cavity-dispersion and supplies final single-point
energies only. Every unique converged solution minimum from the pre-existing
six-case AM1-BCC relaxation diagnostic is evaluated, and the high-level total
potential reranks those candidates. Both potentials and the absent final-SP
force claim remain explicit.

The source selection and input records are label-exposed, but no label enters
the energy or reranking rule and no parameter is fit. The result is negative:

| six-case endpoint | MAE | RMSE |
|---|---:|---:|
| low-level separately relaxed transfer | 2.834 | 3.760 |
| fixed-reference-geometry high-level endpoint | 1.773 | 2.319 |
| high-level SP at low-level-selected minimum | 1.871 | 2.393 |
| high-level-reranked two-level endpoint | 1.901 | 2.403 |

All values are kcal/mol. High-level final SP usefully repairs the low-level
solvent endpoint, but it does not beat the relevant high-level fixed-geometry
baseline: all six absolute errors worsen. Reranking changes one of six minima
and worsens rather than improves the aggregate. The raw 12-single-point trace
and scored summary are
[`route1-two-level-final-sp-rerank-2026-07-24.json`](route1-two-level-final-sp-rerank-2026-07-24.json)
and
[`route1-two-level-final-sp-rerank-summary-2026-07-24.json`](route1-two-level-final-sp-rerank-summary-2026-07-24.json).
This workflow is not promoted to the product runtime.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_two_level_final_sp.py energy \
  --relaxation-record-dir .omx/benchmarks/mlip-conformer-relaxation-am1bcc-route1-20260724/records \
  --base-work-dir .omx/benchmarks/neutral-water-freesolv-route1-20260723 \
  --output docs/implicit-solvation/benchmarks/route1-two-level-final-sp-rerank-2026-07-24.json

python docs/implicit-solvation/benchmarks/run_two_level_final_sp.py summarize \
  --energy-artifact docs/implicit-solvation/benchmarks/route1-two-level-final-sp-rerank-2026-07-24.json \
  --relaxation-record-dir .omx/benchmarks/mlip-conformer-relaxation-am1bcc-route1-20260724/records \
  --high-level-record-dir .omx/benchmarks/am1bcc-chagb-nonpolar-route1-20260724/records \
  --output docs/implicit-solvation/benchmarks/route1-two-level-final-sp-rerank-summary-2026-07-24.json
```

### Energy-only MLIP + CHA-GB Metropolis/TI probe (2026-07-24)

[`mlip_chagb_metropolis_protocol.json`](mlip_chagb_metropolis_protocol.json)
freezes a force-free conformer-aware integration pattern:

```text
U_lambda(R) = U_MLIP,gas(R) + lambda * W_CHA-GB+cavity/dispersion(R)
Delta G      = integral_0^1 <W>_lambda dlambda
```

Two deterministic chains traverse five lambda windows in opposite directions.
Every acceptance decision uses the MACE-OFF23m gas energy plus the exact
frozen AM1-BCC/GBNSR6/PBSA solvent energy. Symmetric torsion and Cartesian
Metropolis proposals avoid requesting the unavailable GBNSR6/PBSA target
force; no MM population, residual fit, or experimental label enters sampling.

All three predeclared development cases pass the acceptance and chain
agreement gates, but accuracy becomes worse:

| case | flexibility | fixed endpoint | sampled TI | sampling shift | chain gap |
|---|---|---:|---:|---:|---:|
| nitromethane | rigid | -2.489 | -2.292 | +0.197 | 0.046 |
| propionic acid | limited | -8.211 | -8.413 | -0.202 | 0.008 |
| methyl hexanoate | flexible | -4.109 | -4.177 | -0.068 | 0.024 |

All values are kcal/mol. Aggregate MAE/RMSE changes from `1.634/1.636` to
`1.790/1.793`, and all three absolute errors worsen. The result establishes
the provider/MLIP integration mechanism, not an accuracy gain, and it is not
wired into the product default. A second full run reproduced every scientific
record exactly after excluding only provider and wall-clock timings; the
summary records `independent_reproduction.verified=true`.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_chagb_metropolis.py run \
  --device cuda
python docs/implicit-solvation/benchmarks/run_mlip_chagb_metropolis.py summarize \
  --reproduction-work-dir \
  .omx/benchmarks/mlip-chagb-metropolis-route1-20260724-repro2
```

### Force-consistent multi-MLIP OBC-II/ACE TI probe (2026-07-25)

[`mlip_obc2_ti_protocol.json`](mlip_obc2_ti_protocol.json) freezes the
corresponding derivative-capable experiment. MACE-OFF23m, AIMNet2, and ANI2x
each sample the same three development molecules with fixed AM1-BCC charges,
OpenMM OBC-II/ACE, five lambda windows, and two opposite traversal orders:

```text
U_lambda(R) = U_MLIP,gas(R) + lambda * W_OBC-II/ACE(R)
Delta G_TI   = integral_0^1 <W_OBC-II/ACE>_lambda dlambda
```

Every window uses the exact scaled energy and force of this expression. Each
raw model/case record is label-free, self-hashed, and stores 40 production
samples per window with the full five-state reduced-potential vector. The nine
raw records were sealed before the development labels were read. Endpoint FEP
ESS and the two-chain TI gap are diagnostics only; the short run does not
pretend to prove equilibration.

The three-case result is again negative:

| gas MLIP | fixed-geometry MAE/RMSE | sampled TI MAE/RMSE | MAE change |
|---|---:|---:|---:|
| MACE-OFF23m | 2.150/2.428 | 2.218/2.532 | +0.067 |
| AIMNet2 | 2.150/2.428 | 2.198/2.508 | +0.047 |
| ANI2x | 2.150/2.428 | 2.234/2.575 | +0.083 |

All values are kcal/mol; positive MAE change is worse. The largest two-chain
gap is `0.136 kcal/mol`, and the minimum endpoint-FEP ESS fraction is `0.685`.
Those narrow diagnostics do not rescue the result: every record remains
fail-closed because a 30-fs window cannot establish equilibration. The
important physical result is consistent across all three gas MLIPs: short
conformational sampling changes the endpoint by roughly `0.1-0.25 kcal/mol`,
while the fixed-charge OBC-II/ACE endpoint error remains around
`2.2 kcal/mol` on these cases. The MLIP supplies a real ensemble term, but it
does not repair the dominant solvent-provider error.

The self-hashed summary is
[`route1-multi-mlip-obc2-ti-2026-07-25.json`](route1-multi-mlip-obc2-ti-2026-07-25.json).
Its published raw-record directory includes the three normalized MOL2 inputs
and all 90 hash-linked window trajectories using repository-relative paths, so
the archived provenance does not depend on the original local `.omx` paths.
MAPLE does not ship a handwritten MBAR implementation.

#### Upstream PyMBAR analysis

The optional `implicit-free-energy` extra now supplies PyMBAR. The reusable
`maple.function.free_energy.analyze_mbar` API validates the state/replicate
matrix, delegates equilibration detection and decorrelation to
`pymbar.timeseries`, runs upstream MBAR, and returns overlap, uncertainty,
effective sample counts, and explicit gates.

The label-free source manifest
[`mlip_obc2_mbar_source_manifest.json`](mlip_obc2_mbar_source_manifest.json)
pins all nine raw records. The analysis protocol
[`mlip_obc2_mbar_protocol.json`](mlip_obc2_mbar_protocol.json) prevents the
post hoc analysis from promoting the short parent screen. Its sealed result is
[`route1-multi-mlip-obc2-mbar-2026-07-25.json`](route1-multi-mlip-obc2-mbar-2026-07-25.json).

Across all nine model/case combinations:

- minimum directional adjacent overlap: `0.145` (`9/9` pass `0.03`);
- maximum absolute MBAR/TI difference: `0.112 kcal/mol` (`9/9` pass `0.25`);
- maximum independent-repeat difference: `0.239 kcal/mol`
  (`9/9` pass `0.25`);
- PyMBAR weight-based effective count gate: `9/9` pass 50;
- per-state decorrelated sample gate: `0/9` pass 20 because only `9-19`
  samples survive;
- equilibrium/conformer mixing: not established.

The analysis is therefore `diagnostic_only_not_promotable`. After that
label-free artifact was sealed, a separate scoring phase produced
[`route1-multi-mlip-obc2-mbar-score-2026-07-25.json`](route1-multi-mlip-obc2-mbar-score-2026-07-25.json):

| gas MLIP | fixed-geometry MAE | MBAR MAE | MAE change |
|---|---:|---:|---:|
| AIMNet2 | 2.150 | 2.207 | +0.056 |
| ANI2x | 2.150 | 2.212 | +0.061 |
| MACE-OFF23m | 2.150 | 2.219 | +0.069 |

All values are kcal/mol; positive change is worse. This confirms that the
correct estimator does not turn the short ensemble term into an accuracy gain.

#### OBC-II/ACE to energy-only CHA-GB/PBSA endpoint correction

[`chagb_endpoint_perturbation_protocol.json`](chagb_endpoint_perturbation_protocol.json)
freezes a label-separated provider correction over the same Route 1 target
family:

```text
U_low(R)  = E_MLIP,gas(R) + W_OBC-II/ACE(R)
U_high(R) = E_MLIP,gas(R) + W_CHA-GB/PBSA(R)

DeltaG_high_solv =
    DeltaG_low_solv
  - RT ln <exp[-beta*(W_CHA-GB/PBSA-W_OBC-II/ACE)]>_low
```

The selected gas MLIP is identical at both endpoints, so the gas correction is
zero. The MLIP cancels from each same-geometry endpoint difference but still
controls the sampled low-endpoint population. No gas-phase MM energy,
hydration-label residual, retraining, or model-specific solvent term is used.

The sealed label-free energy artifact is
[`route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json`](route1-multi-mlip-chagb-endpoint-perturbation-2026-07-25.json).
It contains 3 MLIPs x 3 cases x 2 chains x 40 frames: 720
CHA-GB/PBSA evaluations. The provider evaluations take `21.64 s` locally
(`0.030 s/frame`). This is incremental endpoint cost, not a comparison with
bare MM.

Observed one-sided weights do not collapse:

- minimum combined effective-sample fraction: `0.911`;
- maximum normalized weight: `0.136`;
- maximum independent-chain estimate range: `0.418 kcal/mol`.

However, automated equilibration/decorrelation retains only `4-13` samples per
chain, against the frozen minimum of 20. The combined selected counts are only
`10-21`, and the source windows are 30 fs. Thus `0/9` numerical gates and
`0/9` scientific gates pass even though the observed weights look favorable.

Only after sealing the energy artifact, the separate score artifact
[`route1-multi-mlip-chagb-endpoint-perturbation-score-2026-07-25.json`](route1-multi-mlip-chagb-endpoint-perturbation-score-2026-07-25.json)
was generated:

| gas MLIP | fixed-geometry OBC-II/ACE MAE | low OBC-II/ACE MBAR MAE | corrected CHA-GB/PBSA MAE |
|---|---:|---:|---:|
| AIMNet2 | 2.150 | 2.207 | 1.725 |
| ANI2x | 2.150 | 2.212 | 1.646 |
| MACE-OFF23m | 2.150 | 2.219 | 1.777 |

All values are kcal/mol on only three development cases. Propionic acid
supplies nearly all of the apparent gain; methyl hexanoate worsens for all
three MLIPs. The experiment is therefore
`development_score_not_promotable`. It motivated the broader 19-case screen
above, whose frozen decision is `long_sampling_candidate_not_supported`; it
does not support parameter adjustment or a product claim.

Reproduction:

```bash
python -m pip install '.[implicit-free-energy]'
python docs/implicit-solvation/benchmarks/run_chagb_endpoint_perturbation.py energy
python docs/implicit-solvation/benchmarks/run_chagb_endpoint_perturbation.py score
```

#### MM/GB reference sampling to sparse MLIP target energies

[`mlip_mm_reference_reweighting_protocol.json`](mlip_mm_reference_reweighting_protocol.json)
freezes an indirect free-energy experiment that preserves Route 1's target:

```text
U_target,gas(R)      = U_MLIP,gas(R)
U_target,solution(R) = U_MLIP,gas(R) + W_OBC-II/ACE(R)

DeltaG_target_solv =
    DeltaG_reference_solv
  + DeltaF_reference->target,solution
  - DeltaF_reference->target,gas
```

GAFF2 gas and GAFF2/OBC-II/ACE are sampling references only. The accelerated
candidate evaluates MLIP energies on the reference configurations and uses no
MLIP forces. Direct MLIP endpoint trajectories from the prior label-free
multi-window diagnostic are consumed only for bidirectional validation.
PyMBAR MBAR is the primary indirect estimator; BAR is a two-state cross-check;
single-direction EXP is the accelerated candidate. The MM gas energy never
appears in the target potential or final formula.

The 3x3 diagnostic is negative:

| quantity | observed result |
|---|---:|
| sparse target energy evaluations | 400 per model/case; 3600 total |
| target force evaluations in accelerated candidate | 0 |
| maximum OBC-II/ACE provider mismatch | `1.83e-5 kcal/mol` |
| minimum gas / solution BAR overlap | `0.015 / 0.057` |
| minimum gas / solution directional MBAR overlap | `0.005 / 0.016` |
| minimum gas / solution forward ESS fraction | `0.074 / 0.021` |
| records passing all MBAR/BAR solver-convergence checks | `7/9` |
| gas / solution forward-reverse agreement passes | `0/9 / 0/9` |
| maximum accelerated-vs-bidirectional difference | `2.514 kcal/mol` |
| maximum bidirectional-vs-direct-target MBAR difference | `1.378 kcal/mol` |
| complete numerical-gate passes | `0/9` |

The reported BAR and MBAR point estimates agree within the frozen
`0.005 kcal/mol` energy cross-check for every two-state analysis. That
agreement does not rescue two gas-correction BAR-overlap calculations for
which PyMBAR reports final solver nonconvergence. BAR overlap and the minimum
directional entry of the MBAR overlap matrix are different diagnostics; both
expose inadequate phase-space support here. The failure is not a mismatched
solvent provider or a hidden target formula.

No cycle-level uncertainty is reported. The reference, gas-correction, and
solution-correction terms reuse configurations, so adding their component
uncertainties in quadrature would ignore covariance. The raw artifact retains
the component uncertainties; a production interval would require a joint
chain-aware resampling analysis.

Development scoring was performed only after the label-free artifact was
sealed:

| gas MLIP | fixed geometry MAE | direct target MBAR MAE | reference-only MAE | bidirectional indirect MAE |
|---|---:|---:|---:|---:|
| AIMNet2 | 2.150 | 2.207 | 3.999 | 2.749 |
| ANI2x | 2.150 | 2.212 | 3.514 | 2.805 |
| MACE-OFF23m | 2.150 | 2.219 | 3.025 | 2.442 |

All values are kcal/mol. The reference-only shortcut worsens every MLIP and is
not promoted. The result is consistent with published indirect-free-energy
work: the construction is formally model-agnostic but efficiency is controlled
by reference-target overlap. A further experiment would need intermediate
reference-to-target Hamiltonians or validated nonequilibrium switching.

Artifacts:

- [`route1-mm-reference-reweighting-2026-07-25.json`](route1-mm-reference-reweighting-2026-07-25.json)
- [`route1-mm-reference-reweighting-score-2026-07-25.json`](route1-mm-reference-reweighting-score-2026-07-25.json)
- [`route1-mm-reference-reweighting-raw/manifest.json`](route1-mm-reference-reweighting-raw/manifest.json)

Reproduction:

```bash
python -m pip install '.[implicit-free-energy]'
PYMBAR_DISABLE_JAX=TRUE \
python docs/implicit-solvation/benchmarks/run_mlip_mm_reference_reweighting.py \
  --device gpu0
python \
  docs/implicit-solvation/benchmarks/score_mlip_mm_reference_reweighting.py
```

#### MM/GB-to-MLIP/GB bidirectional nonequilibrium switching

[`mlip_mm_nonequilibrium_switching_protocol.json`](mlip_mm_nonequilibrium_switching_protocol.json)
freezes the next label-free bridge rather than repairing the failed endpoint
estimate post hoc. It uses the same Route 1 target and indirect cycle as the
reference-reweighting experiment, but propagates a linear Hamiltonian:

```text
U_phase,lambda(R) =
    (1-lambda) U_reference,phase(R)
  + lambda [U_target,phase(R) - C_model,case]
```

`C_model,case` is calculated once at a reference gas anchor and applied
unchanged in gas and solution. It shifts no force and cancels exactly from the
cycle; phase-specific alignment is rejected. The public switching engine
evaluates each endpoint force once per MD step, reuses them to mix the current
lambda force, propagates with MAPLE LFMiddle Langevin dynamics, and accumulates
work after propagation at the new coordinate. PyMBAR BAR is the primary
bidirectional estimate and EXP is retained in each direction.

The diagnostic uses:

- MACE-OFF23m, AIMNet2, and ANI2x;
- nitromethane, propionic acid, and methyl hexanoate;
- fixed AM1-BCC charges and the same OBC-II/ACE provider in reference and
  target solution Hamiltonians;
- two endpoint chains and four frozen starts from each chain, for eight work
  values per direction;
- `0.25 fs` integration, with `5 fs` and `20 fs` switches.

The complete label-free result is
[`route1-mm-mlip-nonequilibrium-switching-2026-07-25.json`](route1-mm-mlip-nonequilibrium-switching-2026-07-25.json):

| quantity | observed result |
|---|---:|
| model/case records | 9 |
| target energy+force evaluations | 3264 per record; 29376 total |
| minimum gas / solution BAR overlap | `0.087 / 0.072` |
| maximum 20-fs indirect-vs-direct-target MBAR difference | `0.492 kcal/mol` |
| mean absolute 20-fs indirect-vs-direct-target MBAR difference | `0.246 kcal/mol` |
| maximum 5-fs-to-20-fs cycle change | `1.364 kcal/mol` |
| complete numerical-gate passes | `0/9` |
| endpoint equilibrium / independent starts established | no / no |

The overlap floor alone is not the failure. Individual gas and solution work
legs retain forward/reverse, uncertainty, replicate, and switch-length
failures. Their difference can therefore appear close to the short direct
target control through correlated or accidental cancellation; the cycle is
not promoted when its legs fail.

Only after the label-free result and raw hash manifest were sealed,
[`score_mlip_mm_nonequilibrium_switching.py`](score_mlip_mm_nonequilibrium_switching.py)
read the development labels:

| gas MLIP | fixed geometry MAE | reference MM MAE | endpoint indirect MAE | 20-fs switching MAE | direct target MBAR MAE |
|---|---:|---:|---:|---:|---:|
| AIMNet2 | 2.150 | 2.362 | 2.749 | 2.281 | 2.207 |
| ANI2x | 2.150 | 2.362 | 2.805 | 2.468 | 2.212 |
| MACE-OFF23m | 2.150 | 2.362 | 2.442 | 2.242 | 2.219 |
| all nine records | 2.150 | 2.362 | 2.665 | 2.330 | 2.213 |

All values are kcal/mol. Switching is a better estimator than the failed
endpoint correction for every gas MLIP, but it worsens the experimental MAE
relative to fixed geometry for every gas MLIP. That is a useful negative
separation of concerns: MLIP/MM switching can recover more of the selected
MLIP ensemble without correcting the OBC-II/ACE endpoint physics.

The run consumed `29,376` target force evaluations plus the same number of
reference-force evaluations. By comparison, the frozen short direct-target
protocol nominally contains `3,010` target force evaluations and `400`
target-energy analyses per model/case. No acceleration is established.
Published solvation work used 300 switches of 5 ps rather than eight switches
per direction of 5-20 fs; moving to that regime would be orders of magnitude
more expensive and would still estimate the same fixed-charge solvent target.
The generic switching primitive remains available for other overlap-limited
applications, but this branch is not a Route 1 default.

Artifacts:

- [`route1-mm-mlip-nonequilibrium-switching-2026-07-25.json`](route1-mm-mlip-nonequilibrium-switching-2026-07-25.json)
- [`route1-mm-mlip-nonequilibrium-switching-score-2026-07-25.json`](route1-mm-mlip-nonequilibrium-switching-score-2026-07-25.json)
- [`route1-mm-mlip-nonequilibrium-switching-raw/manifest.json`](route1-mm-mlip-nonequilibrium-switching-raw/manifest.json)

Reproduction:

```bash
python -m pip install '.[implicit-free-energy]'
PYMBAR_DISABLE_JAX=TRUE \
python \
  docs/implicit-solvation/benchmarks/run_mlip_mm_nonequilibrium_switching.py \
  --device gpu0 --resume-existing
python \
  docs/implicit-solvation/benchmarks/score_mlip_mm_nonequilibrium_switching.py
```

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_mlip_obc2_ti.py \
  --protocol docs/implicit-solvation/benchmarks/mlip_obc2_ti_protocol.json \
  --base-case-dir \
    .omx/benchmarks/mlip-chagb-metropolis-route1-20260724/cases \
  --label-root \
    .omx/benchmarks/neutral-water-freesolv-route1-20260723/records/development \
  --work-dir .omx/benchmarks/mlip-obc2-ti-route1-20260725-smoke-ani \
  --output \
    docs/implicit-solvation/benchmarks/route1-multi-mlip-obc2-ti-2026-07-25.json \
  --publish-raw-dir \
    docs/implicit-solvation/benchmarks/route1-multi-mlip-obc2-ti-raw \
  --device gpu0 --resume-existing

python -m pip install '.[implicit-free-energy]'
PYMBAR_DISABLE_JAX=TRUE \
python docs/implicit-solvation/benchmarks/run_mlip_obc2_mbar.py
python docs/implicit-solvation/benchmarks/score_mlip_obc2_mbar.py
```

Reproduction commands:

```bash
BASE=.omx/benchmarks/neutral-water-freesolv
CONF=.omx/benchmarks/conformer-sensitivity

python docs/implicit-solvation/benchmarks/run_conformer_sensitivity.py run \
  --protocol docs/implicit-solvation/benchmarks/conformer_protocol.json \
  --base-work-dir "$BASE" --output-dir "$CONF" --jobs 4
python docs/implicit-solvation/benchmarks/run_conformer_sensitivity.py summarize \
  --protocol docs/implicit-solvation/benchmarks/conformer_protocol.json \
  --output-dir "$CONF" \
  --output "$CONF/conformer-sensitivity-summary.json"
```

The runner requires `crest_conformers.xyz`; a CREST return code of zero alone
is not accepted as success.  This catches a CREST 3.0.2 CLI edge case where
explicit `--cross/--hflip` stopped after ZSORT without producing an ensemble.

The legacy protocol describes a one-shot confirmation command, but that path
is not certification-eligible: preparation already placed the reserved
experimental labels in `prepared.json`, and the old lock accepts an
uninterpreted free-text pass rule. Do not run `freeze-confirmation` or the
reserved calculation as a label-blind gate. A replacement must freeze a
structured candidate and executable numerical rule, record a label-free energy
run first, and only then allow a separate scorer to read the sealed labels. A
failed replacement confirmation must not trigger tuning on the same corpus.

Independent provider evidence uses
[`run_provider_parity.py`](run_provider_parity.py).  Amber-GB references, APBS
Born-ion/grid cases, and their raw audit inputs/outputs are now present under
`tests/solvation/data/`.  The expanded Amber corpus contains 12 neutral
molecules covering C/H/O/N/F/S/Cl/Br/I, including a flexible ester, glucose,
nitro chemistry, and a multifunctional sulfone. It defines 60 molecule/model
slots across all five GB profiles: 58 supported polar records and two
sulfur/GBn2 slots fail closed as predeclared applicability observations.
Complete LCPO comparison contains 39 records: seven cases cover all five
models and dimethyl sulfide covers its four supported models. The APBS corpus
contains the official Born ion plus methanol and aniline grid sweeps.
The unchanged pre-expansion numerical bounds and expanded corpus have now
passed independent architecture/science review. An adversarial code review
then demonstrated that the original verifier could accept a shortened matrix,
invented `expected-unavailable` records, or forged precomputed differences.
The reviewed bounds were frozen only after the verifier was changed to:

- bind the frozen file to the exact proposal, observations, raw artifacts, and
  reference manifests by SHA-256;
- rebuild the complete case/model/grid matrices from the reference manifests;
- permit only the reviewed unsupported records and require their declared
  exception/reason signatures;
- independently recompute all energy differences, force arrays, max/RMS
  metrics, component closures, and APBS kcal/kJ conversion.

The frozen contract is
`provider_parity_tolerances.json` with
`review_status=independently-reviewed-frozen`. Verification of that immutable
evidence uses the pinned raw artifacts:

```bash
PROTOCOL=docs/implicit-solvation/benchmarks/protocol.json
WORK=.omx/verification/provider-parity-reviewed
mkdir -p "$WORK/amber-gb-parity" "$WORK/apbs-grid"
cp tests/solvation/data/provider_parity_raw/amber-openmm-results.json \
  "$WORK/amber-gb-parity/results.json"
cp tests/solvation/data/provider_parity_raw/apbs-grid-results.json \
  "$WORK/apbs-grid/results.json"
python docs/implicit-solvation/benchmarks/run_provider_parity.py \
  verify --protocol "$PROTOCOL" --artifact-dir "$WORK"
```

That frozen verification currently passes 355 independently reconstructed
checks. A new provider run has a new artifact hash and intentionally cannot
reuse this reviewed contract: run `amber-gb`, `apbs-grid`, and `observations`
into a new directory, then create and review a new proposal/evidence chain.
This prevents a fresh or modified corpus from silently inheriting old bounds.

Amber `gbsa=1` is LCPO, so the parity runner compares polar GB, LCPO nonpolar,
and GB+LCPO totals and forces only where both providers implement the same
LCPO parameter semantics. It does not mistake OpenMM's ACE accuracy profile
for Amber LCPO. Across the 58 supported polar records, the largest absolute
Amber/OpenMM difference is `0.001350 kcal/mol` for energy and
`0.001009 kcal/mol/A` for force, both from nitralin/GBn. The 39 complete LCPO
records agree to floating-point precision.

The expanded corpus found two provider-translation defects rather than
relaxing the comparison tolerance. OpenMM's connectivity-only `mbondi3`
helper assigned a carboxylate-like radius to a neutral ester carbonyl oxygen;
the MAPLE radius provider now preserves Amber's generic small-molecule
`1.50 A` oxygen radius and records the adjustment. OpenMM's topology-only LCPO
classifier similarly treated nitro oxygen as carboxylate oxygen; MAPLE now
uses the submitted GAFF `o`/`o2` atom types to select the corresponding
upstream LCPO parameter row and records any adjustment. Both fixes are
provider translation rules, not FreeSolv-fitted corrections.

Cl/Br/I expose an upstream LCPO applicability boundary rather than a polar-GB
failure. Amber `gbsa=1` applies carbon surface-area parameters to those three
halogens. OpenMM instead uses a chlorine-specific term, producing a
`0.161904 kcal/mol` chlorobenzene nonpolar difference, and fails closed because
it has no bromine or iodine LCPO parameters. Therefore chlorobenzene,
bromobenzene, and iodobenzene are explicitly polar-only parity targets; their
LCPO support behavior is audited but excluded from complete LCPO parity.
Nitralin is likewise polar-only because OpenMM has no LCPO row for sulfur with
four heavy-atom bonds.

GBn2 has a separate sulfur applicability boundary. Amber's signed Taylor
descreening branch retains a close-pair contribution for sulfur's negative
screening radius, while OpenMM's generic closed form zeros that branch.
Consequently dimethyl sulfide and nitralin/GBn2 are recorded as
`expected-unavailable`; they do not enter the energy/force tolerance. These
are provider-parity and applicability results, not hydration-accuracy
statistics.

APBS 1.4.1 reproduces the official `-229.59 kJ/mol` Born-ion result to
`0.000505 kcal/mol` after unit conversion.  The neutral grid sweeps show a
largest successive-finest change of `0.113892 kcal/mol`, demonstrating that
the molecular-surface grid sequence is numerically noisier than the canonical
ion.  The proposed `0.15 kcal/mol` grid threshold records that observation; it
does not claim monotone continuum convergence or chemical accuracy.

Dataset interpretation follows Mobley and Guthrie, *J. Comput.-Aided Mol.
Des.* 2014, DOI `10.1007/s10822-014-9747-x`, and the FreeSolv v0.5 update by
Duarte Ramos Matos et al., *J. Chem. Eng. Data* 2017, DOI
`10.1021/acs.jced.7b00104`.  Exact charge/GB literature names are recorded in
`protocol.json` for human review.

This pilot fixed ten chemical classes before calculation: alkane, aromatic
hydrocarbon, alcohol, ether, ketone, ester, nitrile, aromatic amine,
haloalkane, and sulfoxide.  Experimental values and GAFF MOL2 geometries come
from FreeSolv v0.52 at commit
`6c7d19b4b565537365ffd22006aa2cd4643200c6`.

Each molecule was charged independently with AmberTools 26.0 Antechamber using
AM1-BCC or ABCG2.  MAPLE then evaluated HCT, OBC-I, OBC-II, GBn, and GBn2 in
water with the ACE nonpolar term using OpenMM 8.4.0.post2 on the Reference
platform.  The calculation is a single-FreeSolv-geometry GBSA estimate, not a
conformational or finite-temperature free-energy average.

## Route 1 development default: AM1-BCC/OBC-II/ACE

Forward Route 1 development uses AM1-BCC/OBC-II/ACE unless a protocol
explicitly selects another charge/model pair. ABCG2 remains available only by
explicit selection. This policy does not retroactively alter the frozen
development matrix or the historical ABCG2 MLIP diagnostics above, and it is
not confirmation certification.

| class | molecule | experiment | prediction | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.000 | 1.564 | -0.436 |
| aromatic hydrocarbon | benzene | -0.900 | -1.263 | -0.363 |
| alcohol | methanol | -5.100 | -4.651 | 0.449 |
| ether | methoxymethane | -1.910 | -1.389 | 0.521 |
| ketone | acetone | -3.800 | -5.052 | -1.252 |
| ester | ethyl acetate | -2.940 | -4.796 | -1.856 |
| nitrile | acetonitrile | -3.880 | -4.535 | -0.655 |
| aromatic amine | aniline | -5.490 | -5.783 | -0.293 |
| haloalkane | chloroethane | -0.630 | -0.043 | 0.587 |
| sulfoxide | methylsulfinylmethane | -9.280 | -8.895 | 0.385 |

All energies are in `kcal/mol`.  Aggregate default metrics are MSE `-0.291`,
MAE `0.680`, RMSE `0.826`, and maximum absolute error `1.856 kcal/mol`.

## Corrected QEq-GTO diagnostic

The QEq rows use the same ten MOL2 geometries and the same OpenMM/ACE protocol,
but generate fixed charges internally with the repaired full hydrogen SCF.  At
each iteration MAPLE updates both the hydrogen idempotential and the hydrogen
screening exponent used by every H-containing GTO pair integral.  The largest
absolute atomic charge across the pilot is `0.490 e`, versus up to `3.704 e`
from the removed Open Babel-style simplification.

For the default OBC-II comparison:

| class | molecule | experiment | corrected QEq-GTO | signed error |
|---|---|---:|---:|---:|
| alkane | methane | 2.000 | 1.523 | -0.477 |
| aromatic hydrocarbon | benzene | -0.900 | 0.997 | +1.897 |
| alcohol | methanol | -5.100 | -3.768 | +1.332 |
| ether | methoxymethane | -1.910 | -4.678 | -2.768 |
| ketone | acetone | -3.800 | -5.965 | -2.165 |
| ester | ethyl acetate | -2.940 | -8.249 | -5.309 |
| nitrile | acetonitrile | -3.880 | -0.414 | +3.466 |
| aromatic amine | aniline | -5.490 | 0.149 | +5.639 |
| haloalkane | chloroethane | -0.630 | -7.062 | -6.432 |
| sulfoxide | methylsulfinylmethane | -9.280 | -8.881 | +0.399 |

The corrected model is numerically stable, but its OBC-II MAE/RMSE are
`2.988/3.626 kcal/mol`, still much worse than AM1-BCC (`0.680/0.826`) and
ABCG2 (`0.963/1.233`).  This is expected evidence of a parameter-pairing
problem: the QEq charge model and Amber GB radii/nonpolar terms were not jointly
fit.  It is not evidence that the old multi-electron charges were acceptable.
The full per-atom charges, SCF/KKT diagnostics, components, and all five GB
models are frozen in
[`freesolv10-qeq-gto-full-h-2026-07-22.json`](freesolv10-qeq-gto-full-h-2026-07-22.json).

## Polarizable CQEq-GTO diagnostic

The polarizable run does not reuse the non-variational original-QEq fixed
point.  It minimizes the consistent-QEq (CQEq) energy in vacuum and minimizes
CQEq plus the GB polar quadratic form in solution.  Across all 50
molecule/model calculations, the largest solution-phase atomic charge is
`0.894 e`, the largest constrained KKT residual is `1.37e-6 eV`, and the
smallest projected charge-Hessian eigenvalue is `+1.720 eV`.  The coupled
solutions are therefore finite constrained local minima rather than the
multi-electron runaway produced by the removed simplified implementation.

The accuracy result is nevertheless negative: the best model is HCT, with
MAE/RMSE `5.719/7.748 kcal/mol`; OBC-II gives `7.219/9.654 kcal/mol`.  Allowing
the charges to relax therefore worsens this pilot relative to fixed QEq-GTO,
AM1-BCC, and ABCG2.  The implementation is retained as an experimental,
force-consistent profile, not as an accuracy-certified default, because the
QEq/CQEq parameters and Amber GB radii/nonpolar parameters were not jointly
fit.  The complete charges, KKT diagnostics, projected Hessian checks, energy
components, and all five GB models are frozen in
[`freesolv10-cqeq-gto-polarizable-2026-07-22.json`](freesolv10-cqeq-gto-polarizable-2026-07-22.json).

This small pilot must not be used to select or refit a model.  The full
supported FreeSolv benchmark and conformer/sampling analysis remain open.

## APBS molecular-surface PB plus ACE screen

[`apbs_ace_protocol.json`](apbs_ace_protocol.json) prospectively freezes a
development-only, fixed-AM1-BCC endpoint:

```text
APBS molecular-surface LPBE polar + OpenMM OBC-II/mbondi2 ACE nonpolar
```

The label-free manifest contains geometry and charge hashes plus the already
decomposed ACE component; the energy phase never opens its label-bearing source
record.  APBS APOLAR is evaluated only as a control and is not added together
with ACE.  No gas-phase MM energy, fitted selector, residual model, or MLIP
retraining enters either endpoint.

All 526 energy jobs and the preselected 20-case fine-grid jobs completed with
zero failures.  Development scoring gives:

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| AM1-BCC/OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| AM1-BCC/APBS-mol/APOLAR | 4.415 | 4.984 | 15.880 |
| AM1-BCC/APBS-mol/ACE | 1.658 | 2.535 | 14.747 |
| AM1-BCC/CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |

PB/ACE gains `0.102 kcal/mol` MAE over OBC-II/ACE, with a paired 95% bootstrap
interval `[0.016, 0.191]`, but misses the frozen `0.15 kcal/mol` materiality
gate.  Changing the APBS polar grid from `97^3 @ 0.33 A` to
`129^3 @ 0.25 A` changes the 20 preselected cases by up to
`0.671 kcal/mol` (P90 `0.455`), failing the `0.25/0.15` numerical thresholds.
The deterministic decision is `rejected_for_runtime_promotion`.

Because the parent failure mixed materiality and grid effects, a second
label-free protocol evaluated `129^3 @ 0.25 A` against
`161^3 @ 0.20 A` on the same 20 preselected cases.  That fine pair passes:
maximum/P90 differences are `0.175/0.148 kcal/mol`.  A separately frozen
526-case `129^3` screen then gives PB/ACE MAE/RMSE `1.629/2.462 kcal/mol` and
a paired gain over OBC-II/ACE of `0.131 kcal/mol` (95% interval
`[0.050, 0.217]`).  It still misses the prospective `0.15` materiality gate,
so numerical convergence does not change the rejection.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_apbs_ace_screen.py run --workers 6
python docs/implicit-solvation/benchmarks/run_apbs_ace_screen.py grid --workers 6
python docs/implicit-solvation/benchmarks/run_apbs_ace_screen.py summarize

python docs/implicit-solvation/benchmarks/run_apbs_mol_grid_followup.py all \
  --workers 6

python docs/implicit-solvation/benchmarks/run_apbs_ace_screen.py all \
  --protocol docs/implicit-solvation/benchmarks/apbs_ace_fine_protocol.json \
  --work-dir .omx/benchmarks/am1bcc-apbs-ace-fine-route1-20260725 \
  --summary docs/implicit-solvation/benchmarks/freesolv-am1bcc-apbs-ace-fine-2026-07-25.json \
  --workers 6
```

The frozen summary is
[`freesolv-am1bcc-apbs-ace-2026-07-25.json`](freesolv-am1bcc-apbs-ace-2026-07-25.json).
The fine-grid artifacts are
[`apbs-mol-grid-followup-2026-07-25.json`](apbs-mol-grid-followup-2026-07-25.json)
and
[`freesolv-am1bcc-apbs-ace-fine-2026-07-25.json`](freesolv-am1bcc-apbs-ace-fine-2026-07-25.json).
The exploratory aniline observation preceded the full screen, so this remains
development evidence even though the energy execution itself is label-free.

## APBS SPL4 polar-force and APOLAR force probes

The official APBS interface exposes `calcforce comps` and permits
`print elecForce solv - ref end`, but the product SP endpoint uses
`srfm=mol`. The pinned APBS 3.4.1 Linux executable aborts that force request
with `Forces *must* be calculated with spline-based surfaces!`. The isolated
probe therefore changes only the dielectric surface to `srfm=spl4`, keeps
fixed AM1-BCC charges and mbondi2 radii, and excludes every nonpolar term.

[`route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json`](route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json)
contains 554 APBS jobs: one analytical-force call per grid plus centered
energies for all 69 Cartesian components, two displacement sizes
(`0.003 A`, `0.010 A`), and two grids. Results are in `kJ/mol/A`:

| grid | FD step | force/FD RMSE | maximum error | net-force norm |
|---|---:|---:|---:|---:|
| `129^3 @ 0.25 A` | 0.003 | 0.396 | 1.779 | 4.659 |
| `129^3 @ 0.25 A` | 0.010 | 0.395 | 1.773 | 4.659 |
| `161^3 @ 0.20 A` | 0.003 | 0.199 | 1.103 | 2.336 |
| `161^3 @ 0.20 A` | 0.010 | 0.198 | 1.098 | 2.336 |

The polar energy changes by only `0.073 kJ/mol` between grids, but the force
RMSE/maximum change by `0.483/2.579 kJ/mol/A`. Thus energy convergence does
not establish gradient convergence. APBS documentation also states that
spline surfaces require substantial force-field reparameterization; the
generic mbondi2 assignment has not undergone that calibration.

[`route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json`](route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json)
separately checks the locked `gamma*SASA` nonpolar component with 277 jobs.
At the APBS internal and external matching displacement of `0.05 A`, the
calculation-block total force has RMSE/maximum
`1.114/4.871 kJ/mol/A` against the reported energy and net-force norm
`3.694 kJ/mol/A`. The `PRINT APOL` vector is an unscaled surface component;
the calculation total equals approximately `-gamma` times that vector, but
the scaled result still fails the external energy derivative.

Both probes read no hydration labels and use no gas-phase MM energy, fit,
residual, or MLIP retraining. They reject APBS force promotion before any
SPL4 FreeSolv or speed campaign. Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_apbs_spline_force_probe.py \
  --workers 4
python docs/implicit-solvation/benchmarks/run_apbs_apolar_force_probe.py \
  --workers 4
```

## Route 1 multi-MLIP compatibility trace

[`run_route1_compatibility.py`](run_route1_compatibility.py) constructs Route 1
through the public `SetCalculator` path for MACE-OFF23m and AIMNet2. It verifies
the frozen AM1-BCC vector against its source manifest, then records gas,
solvent, and combined energies and forces, all-`3N`-component finite
differences of the combined potential, a two-step combined BFGS smoke, and
three manually displaced combined-energy points.

The frozen local trace,
[`route1-compatibility-methyl-hexanoate-2026-07-24.json`](route1-compatibility-methyl-hexanoate-2026-07-24.json),
passes every predeclared gate. Energy closure is below `1.6e-15 Ha`; the
maximum repeated-forward force-closure errors are below `7.5e-17 Ha/A` for
MACE-OFF23m and `2.1e-8 Ha/A` for float32 AIMNet2. At a predeclared `0.003 A`
finite-difference step, all 69 Cartesian components are evaluated and the
maximum errors are `1.31e-5` and `3.29e-5 Ha/A`, respectively. The
fixed-geometry solvent-energy spread between gas models is zero.

This is adapter and composition evidence for two named checkpoints on one
molecule. It is not a universal MLIP accuracy result or an OPT/SCAN stability
certification. The three displaced points are explicitly not a test of MAPLE's
SCAN dispatcher.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_route1_compatibility.py \
  --mol2 .omx/benchmarks/route1-performance-20260724/mobley_1017962/m.mol2 \
  --charge-manifest docs/implicit-solvation/benchmarks/chagb_nonpolar_source_manifest.json \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model maceoff23m --model aimnet2 --device cuda \
  --openmm-platform Reference --fd-step 0.003 --opt-steps 2 \
  --scan-displacement 0.03 \
  --work-dir .omx/benchmarks/route1-compatibility-20260724-single-context \
  --output docs/implicit-solvation/benchmarks/route1-compatibility-methyl-hexanoate-2026-07-24.json
```

## Route 1 performance diagnostic

[`run_route1_performance.py`](run_route1_performance.py) measures warm,
in-process energy+force calls for:

1. the gas MLIP,
2. the solvent correction alone,
3. the combined Route 1 potential, and
4. a named GAFF2 plus OBC-II MM baseline.

The first local trace,
[`route1-performance-methyl-hexanoate-2026-07-24.json`](route1-performance-methyl-hexanoate-2026-07-24.json),
uses 50 samples after six warmups on a 23-atom development molecule. With
MACE-OFF23m on an RTX 4060 Laptop GPU, OpenMM 8.5.2 Reference OBC-II/ACE adds
an observed `2.28%` to the interleaved paired gas-MLIP median; the
correction-only median is `2.12%` of the gas median. Its raw MM ratio is
not a local comparison because the MLIP runs on the GPU while the named MM
baseline runs on an OpenMM CPU backend.

The second local trace,
[`route1-performance-methyl-hexanoate-cpu-2026-07-24.json`](route1-performance-methyl-hexanoate-cpu-2026-07-24.json),
uses the same host with PyTorch restricted to one CPU thread and the
single-thread OpenMM Reference platform. The paired combined-minus-gas
observation is `0.45%`, while the correction-only median is `0.48%` of the gas
median; the combined Route 1 median is about `8749x` the named GAFF2/OBC-II
median for this small molecule.

Two additional traces repeat the protocol with ANI2x:

- [`route1-performance-methyl-hexanoate-ani2x-2026-07-24.json`](route1-performance-methyl-hexanoate-ani2x-2026-07-24.json)
  records a `0.27%` paired combined-minus-gas observation and a `4.93%`
  correction-only fraction on GPU/Reference. Its mixed-backend MM ratio is not
  a local comparison.
- [`route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json`](route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json)
  records `14.96%` paired and `10.41%` correction-only under the one-thread
  CPU/Reference policy; combined ANI2x+GB is about `444x` the named
  GAFF2/OBC-II median.

Reproduction:

```bash
MOL2=.omx/benchmarks/neutral-water-freesolv-route1-20260723/dataset/mol2files_gaff/mobley_1017962.mol2
MANIFEST=docs/implicit-solvation/benchmarks/chagb_nonpolar_source_manifest.json
PRMTOP=.omx/benchmarks/route1-performance-20260724/mobley_1017962/obc2.prmtop
RUNNER=docs/implicit-solvation/benchmarks/run_route1_performance.py

python "$RUNNER" \
  --mol2 "$MOL2" --charge-manifest "$MANIFEST" --prmtop "$PRMTOP" \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model maceoff23m --device cuda \
  --openmm-platform Reference --openmm-platform CUDA \
  --samples 50 --warmups 6 --log /tmp/maple-route1-performance-mace-gpu.log \
  --output docs/implicit-solvation/benchmarks/route1-performance-methyl-hexanoate-2026-07-24.json

python "$RUNNER" \
  --mol2 "$MOL2" --charge-manifest "$MANIFEST" --prmtop "$PRMTOP" \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model maceoff23m --device cpu --torch-threads 1 \
  --openmm-platform Reference \
  --samples 50 --warmups 6 --log /tmp/maple-route1-performance-mace-cpu.log \
  --output docs/implicit-solvation/benchmarks/route1-performance-methyl-hexanoate-cpu-2026-07-24.json

python "$RUNNER" \
  --mol2 "$MOL2" --charge-manifest "$MANIFEST" --prmtop "$PRMTOP" \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model ani2x --device cuda \
  --openmm-platform Reference --openmm-platform CUDA \
  --samples 50 --warmups 6 --log /tmp/maple-route1-performance-ani2x-gpu.log \
  --output docs/implicit-solvation/benchmarks/route1-performance-methyl-hexanoate-ani2x-2026-07-24.json

python "$RUNNER" \
  --mol2 "$MOL2" --charge-manifest "$MANIFEST" --prmtop "$PRMTOP" \
  --compound-id mobley_1017962 --molecule-name "methyl hexanoate" \
  --model ani2x --device cpu --torch-threads 1 \
  --openmm-platform Reference \
  --samples 50 --warmups 6 --log /tmp/maple-route1-performance-ani2x-cpu.log \
  --output docs/implicit-solvation/benchmarks/route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json
```

The compatibility trace, task matrix, and four performance traces each embed a
machine-readable `command_provenance` record with the runner hash, normalized
arguments and argument hash, Python executable, and selected resource
environment variables. Each also carries a self-consistent `content_sha256`
computed without its own hash field. Tests recompute both fingerprints instead
of trusting the recorded strings.

This is a single-molecule engineering diagnostic. It supports neither a global
“faster than MM” claim, a universal “negligible solvent overhead” claim, nor
an OPT/SCAN throughput claim. Charge generation and model/context startup are
excluded, gas and combined calls use interleaved independent calculator
instances, no uncertainty interval is reported, the absolute inputs make these
locally traceable rather than checkout-portable artifacts, and the installed
OpenMM build has no CUDA platform. OpenMM Reference is a
correctness-oriented fixed backend in these traces, not a production MM
throughput baseline; its two same-host, one-thread ratios are local
counterexamples only.

## OpenMM product-platform audit

[`run_openmm_platform_audit.py`](run_openmm_platform_audit.py) compares the
single-thread deterministic CPU product path with the independent Reference
control on the pinned 12-molecule Amber/OpenMM corpus. It evaluates all five GB
models with polar-only and LCPO paths, verifies expected applicability
failures, rebuilds independent CPU contexts to test exact repeatability, and
times warm OBC-II/ACE energy+force calls.

The sealed
[`route1-openmm-platform-audit-2026-07-25.json`](route1-openmm-platform-audit-2026-07-25.json)
contains 60 polar slots: 58 succeed and the two sulfur/GBn2 slots remain
expected-unavailable. Forty-four LCPO paths succeed and 16 retain their frozen
applicability boundary. Across supported paths, CPU-versus-Reference
energy/component and force maxima are `9.602e-6 kcal/mol` and
`1.235e-5 kcal/mol/A`; independently built CPU contexts repeat exactly. Warm
OBC-II/ACE energy+force correction calls are locally `2.40x` to `3.03x`
faster than Reference across the 12 molecules, with median `2.79x`.

Reproduction:

```bash
python docs/implicit-solvation/benchmarks/run_openmm_platform_audit.py \
  --manifest tests/solvation/data/amber_gb_reference/manifest.json \
  --warmups 10 --samples 100 \
  --output docs/implicit-solvation/benchmarks/route1-openmm-platform-audit-2026-07-25.json
```

This audit changes only the OpenMM execution backend. It neither changes the
GB/ACE functional nor improves chemical accuracy, and its local correction
speedup is not a claim that MLIP plus implicit solvent is faster than bare MM.
The canonical Amber provider-parity protocol remains explicitly pinned to
Reference.

The ACE runtime behind these refreshed traces performs one OpenMM state
evaluation per correction call. It scales only the upstream ACE energy term by
a unit global parameter, reads the corresponding energy derivative as the
nonpolar component, and obtains the polar component by subtraction from the
same total energy. On the water unit case, tests compare that polar value with
an independent polar-only context for HCT, OBC-I, OBC-II, GBn, and GBn2 at
`2e-10 Hartree` tolerance on the product CPU backend. The frozen compatibility
and task-matrix traces
record `energy_force_evaluations_per_call=1`. LCPO remains a separate-force,
two-context decomposition.
