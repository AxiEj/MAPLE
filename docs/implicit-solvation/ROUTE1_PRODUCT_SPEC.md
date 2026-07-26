# Route 1 product specification

**Route name:** Additive fixed-charge PB/GB implicit solvation

**Product role:** Baseline/Product Route

## Purpose

Route 1 is MAPLE's baseline/product path for adding a fixed-charge continuum
solvent functional to a registered gas-phase MLIP:

\[
E_{\mathrm{solution}}(R)=E_{\mathrm{MLIP,gas}}(R)
 +G_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
 +G_{\mathrm{nonpolar}}(R).
\]

The mandatory product goal is a small, auditable composition layer for fast
SP, OPT, and SCAN/PES. Route 1 does not retrain the gas MLIP, fit a residual
to hydration labels, or add a gas-phase MM energy.

This additive identity is authoritative. Innovation is not required to modify
the PB/GB equations themselves: it may instead come from the common MLIP
interface, provider abstractions, provenance, force-consistent composition,
conformer-aware evaluation, compatibility checks, or explicit-inner/
implicit-outer integration. Those product innovations must still preserve the
displayed fixed-charge formula.

Explicit numerical FREQ, conformer/ensemble evaluation, conservative
non-periodic sampling, and explicit-inner/implicit-outer composition are
opt-in extensions over that same potential. They may strengthen Route 1, but
they do not redefine its formula or gate the identity of the baseline
SP/OPT/SCAN product.

No chemistry-specific learned correction is part of Route 1. Improvements must
come from the physical fixed-charge solvent functional or from the product
composition around it, not from learning the remaining error against FreeSolv.

The intended innovation surfaces are:

- one composition interface across independently registered gas-phase MLIPs;
- charge, radius, polar, and nonpolar provider abstractions with locked physical
  pairings;
- complete audit/provenance records and automatic checks for declared element
  domains plus task/derivative capabilities; an undeclared checkpoint element
  domain is not certified by the shared preflight;
- energy/force-consistent SP, OPT, SCAN, explicit numerical FREQ, and NVE/NVT
  MD integration;
- one return-value batch contract for gas energies/forces, with a conservative
  sequential fallback and backend-native acceleration only after numerical
  parity is demonstrated;
- conformer-aware evaluation when the sampling protocol itself is validated;
- Explicit-inner/implicit-outer composition behind the same additive boundary.

## Canonical acceptance contract

Route 1 is accepted as the baseline/product route only when all of the
following remain true:

1. the gas term comes exclusively from the selected registered gas-phase MLIP;
2. the solvent term is the sum of fixed-charge polar and geometry-dependent
   nonpolar providers, with no gas-phase MM term;
3. `#charge(source=maple)` defaults to AM1-BCC, while ABCG2 and research charge
   profiles require explicit selection;
4. the common composition interface supports SP, OPT, and SCAN/PES whenever
   the selected gas and solvent providers supply the required derivatives,
   and permits an explicit numerical FREQ only when the complete composed
   force is available;
5. provider parity, complete-potential force checks, FreeSolv evaluation, and
   provenance/applicability accounting remain separate validation axes;
6. no residual model, hydration-label fit, MLIP retraining, or hidden
   MLIP-specific solvent correction is introduced.

Named MLIP traces demonstrate adapter compatibility only for those checkpoints.
The architecture is MLIP-agnostic, but registration alone is insufficient:
the backend must inherit the common CalcABC composition path or explicitly
declare and implement the equivalent Route 1 capability. Unsupported plugins
fail before construction. Chemical accuracy is not assumed to transfer
universally between gas models.

## Provider boundaries

| component | required contract | current product implementation |
|---|---|---|
| gas model | MAPLE calculator registration, energy, and task-required derivatives | any registered molecular MLIP that passes the compatibility gate |
| charge | one finite charge per atom, topology/order audit, declared lifecycle | MOL2 fixed charge or AmberTools AM1-BCC; ABCG2 is explicit-only |
| radius | one positive radius per atom plus provider-native GB parameters and provenance | locked upstream OpenMM Amber assignments |
| polar | additive energy and an honest derivative capability declaration | OpenMM Amber GB; APBS LPB and AmberTools CHA-GB are energy-only |
| nonpolar | additive energy, derivative capability, parameters, implementation provenance | OpenMM ACE/LCPO/none, locked APBS SASA, or locked PBSA cavity/dispersion |

Radius and nonpolar providers are first-class runtime objects. A provider record
states its category, name, profile, implementation, parameters, and component
properties. Physical pairings remain locked: abstraction does not make
incompatible charge/radius/nonpolar parameter sets interchangeable.

The audit manifest records charge, radius, polar, and nonpolar provenance under
one composition boundary.

## Defaults and applicability

- `#charge(source=maple)` means fixed AM1-BCC with `geometry=keep`.
- ABCG2 and fixed QEq-GTO charge diagnostics require explicit selection and are
  never automatic fallbacks. Polarizable CQEq-GTO/GB is labeled a separate
  research control and is not a Route 1 fixed-charge product profile.
- OBC-II/ACE remains the forward development default. A literature candidate
  does not replace it without the frozen benchmark and confirmation process.
- The product OpenMM platform default is single-thread CPU with deterministic
  forces. `platform=Reference` remains the explicit independent
  correctness/parity control; platform selection never changes the solvent
  functional.
- AM1-BCC/CHA-GB/PBSA cavity-dispersion is an explicit SP-only accuracy
  provider. It does not replace the force-capable OBC-II/ACE default.
- The default declared domain is a neutral, closed-shell, connected organic
  molecule in water with explicit MOL2 hydrogens and connectivity.
- `inner=prebuilt` is an experimental multicomponent-MOL2 extension. It uses
  fixed user-supplied charges for the entire cluster and does not alter the
  default single-solute charge-generation path. It is restricted to OpenMM GB
  plus ACE/LCPO and binds the frozen charge/radius/topology state to a stable
  per-atom identity array so same-element reordering also fails closed.
- “Any MLIP” means adapter/interface compatibility, not universal chemical
  accuracy. Each MLIP receives its own element-domain, energy/force, task, and
  performance record.

## Task and derivative contract

| provider composition | SP | OPT | SCAN/PES | FREQ | MD | policy |
|---|---:|---:|---:|---:|---:|---|
| fixed-charge OpenMM GB + ACE/LCPO | energy + gradient | yes | yes | explicit numerical complete-force Hessian | non-periodic NVE/NVT | combined conservative potential |
| prebuilt cluster + OpenMM GB + ACE/LCPO | energy + gradient | yes | yes | numerical fixed-shell curvature only | no | whole-cluster fixed-shell potential; occupancy-constrained MD not yet defined; no absolute `DeltaG_solv` claim |
| automatic QCG/FEBISS explicit-inner cycle | unavailable | no | no | no | no | source-audited research path only; no arbitrary-MLIP QCG backend, complete neutral FreeSolv validation, or end-to-end speed evidence |
| APBS LPB + APOLAR | energy only | no | no | no | no | fail closed on forces |
| APBS SPL4 LPB force probe | rejected benchmark | no | no | no | no | molecular surface aborts; SPL4 polar and APOLAR force gates fail |
| AmberTools GENIUSES/MLSES PB surface probe | rejected benchmark | no | no | no | no | no atom-resolved MLSES force; no local small-molecule speedup; no runtime provider |
| external ddX/ddPCM audit | benchmark only | no | no | no | no | polar derivative passes, but accuracy and performance gates fail; no dependency/provider added |
| CHARMM GBMV2/SA source audit | unavailable locally | no | no | no | no | scientifically promising physical candidate; registered CHARMM runtime and deployable provider path not available for parity/force validation |
| SLIC/CDC source audit | unavailable locally | no | no | no | no | promising AM1-BCC-compatible physical energy model; complete 2022 upstream and atom-resolved polar-plus-nonpolar force are unavailable |
| AmberTools CHA-GB + PBSA cavity/dispersion | energy only | no | no | no | no | explicit AM1-BCC SP provider; preserves input GAFF/GAFF2 types and fails closed on forces |
| low-level OBC-II/ACE OPT -> CHA-GB final SP | benchmark only | low-level OPT only | no | no | no | both potentials exposed; no final-SP force claim |
| Amber PB exact difference | benchmark only | no | no | no | no | MM gas term cancels, but the returned force is not the derivative of the endpoint |

OPT and relaxed SCAN must differentiate the same combined potential reported as
the energy. MAPLE must never return gas-only MLIP forces beside a solvent-added
energy.

Implicit-solvent FREQ follows the same rule at second-derivative order. It is
accepted only with explicit `hessian=numerical`, which central-differences the
complete combined force, and only with the mass-weighted `method=mw` frequency
path and `ilowfreq` in `0..3`. Non-mass-weighted and unimplemented dual-mode
requests fail during input validation. A gas-backend analytic Hessian is
rejected because it omits the solvent curvature. The resulting local Hessian
does not itself supply gas/solution conformer populations, standard-state
conversion, or an absolute solvation free energy; the existing RRHO
translational/rotational terms remain ideal-gas quantities and are labeled
accordingly. `ilowfreq=2` is the Grimme entropy-only interpolation using a
finite effective inertia; `ilowfreq=3` is the Otlyotov--Minenkov extension that
also interpolates the complete per-mode vibrational internal energy, with ZPE
inside rather than outside the interpolation. These implementations are
formula-level capabilities, not evidence that qRRHO improves hydration
accuracy. Constrained FREQ is not admitted until MAPLE implements a
reduced active-coordinate Hessian, mass matrix, and rigid-body projection;
full-space projection of `FixAtoms` or `FixInternals` Hessians is not physical.
These guards are enforced again at the computation API, which requires
`mode=fixed`, solvent force support, calculator Route 1 composition capability,
`hessian=numerical`, and non-periodic atoms. Text parsing is not the sole
scientific boundary.

The preregistered
`maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v6` benchmark
composes those formula-level capabilities into separate gas and solution
selected-minimum local-RRHO surrogates for MACE-OFF23m, AIMNet2, and ANI2x. It
was a six-case development falsification pilot. Cases were chosen label-blindly
against target source-state counts under a 23-atom cost ceiling. Harmonic RRHO
was primary; Grimme/Otlyotov--Minenkov qRRHO variants were sensitivity-only
because explicit minima plus rotor interpolation can double count torsional
entropy.

The runner now rejects any selected optimization branch that fails or does not
converge, any unique minimum with a selected negative frequency, excessive
projected rigid-mode leakage, and the preregistered Hessian-displacement
preflight on a rigid anchor plus one flexible case. Rotational symmetry
numbers are frozen per case and remain
separate from conformer degeneracy or duplicate arrival count. Its label-free
runner and post-seal scorer are not public MAPLE task dispatchers. Neither
preregistration nor a development pass establishes basin measures,
conformer completeness, external accuracy, or eligibility for a public
absolute-solvation-free-energy task.

The first v6 record, ANI2x on nitromethane, closed the pilot before scoring.
Both gas and solution optimizations converged below `0.009 eV/A`, but their
raw central-difference Hessian asymmetries were approximately
`1.21e-3/1.13e-3 Hartree/A^2`, over one thousand times the frozen `1e-6`
gate. The exact failure record and optimization trajectories are durable.
A post-failure, label-blind diagnosis found stationary symmetrized spectra and
only `1.10/1.06 cm-1` maximum selected-mode RMS changes across
`0.001/0.002/0.004 A`; that diagnoses a non-dtype-aware raw-asymmetry gate but
cannot rescue or rerun v6. Any successor requires a separately versioned
numerical-Hessian qualification before a new scientific pilot.

The v8 successor supplied that qualification without reading experimental
labels. It combined a broad `0.02 Hartree/Angstrom^2` raw absolute sanity
ceiling with a scale-aware `0.002` Frobenius asymmetry gate and retained the
independent `25 cm-1` displacement-frequency criterion. A force-converged
selected negative mode triggered one deterministic plus/minus normal-mode
displacement and reoptimization cycle; it was never made positive by taking
an absolute value. The exact v8 nitromethane preflight passed for
MACE-OFF23m, AIMNet2, and ANI2x, with AIMNet2 requiring saddle recovery in
both phases. The next flexible preflight then failed closed before any
Hessian or label scoring: one selected AIMNet2 gas branch exhausted the
frozen 500-step LBFGS budget at `0.138705 eV/A`.

A separately frozen, label-blind optimizer qualification subsequently compared
one common BFGSLineSearch, LBFGSLineSearch, and FIRE2/ABC policy on all
`3 MLIPs x 2 phases x 3 source states`. V2 reran from the exact source
coordinates and limited every branch to 1,000 calculator evaluations; no v1
partial or v8 optimized state was reused. Every candidate passed only `12/18`
branches. The two line-search families exhausted the evaluation ceiling or
otherwise failed on AIMNet2, while FIRE2/ABC converged five AIMNet2 branches
but failed their frozen no-final-energy-increase gate and exhausted the ceiling
on the sixth. Therefore no single model-neutral optimizer policy qualifies,
no v9 successor may be created, and no aggregate or hydration-accuracy score
may be opened. This is a negative optimizer-robustness result, not a failure
of the additive energy formula and not permission for per-model optimizer
tuning.

### Explicit CHA-GB single-point profile

The higher-accuracy development endpoint is exposed only through the locked
single-point contract:

```text
#sp
#charge(source=maple,method=am1bcc,geometry=keep,executable=/path/to/ambertools/bin/antechamber)
#solv(implicit=water,method=gb,provider=ambertools,model=chagb,profile=chagb-bondi-pbsa-inp2,nonpolar=cavity-dispersion,executable=/path/to/ambertools/bin/gbnsr6,experimental=true)
```

The submitted MOL2 must already carry GAFF/GAFF2-compatible atom types. The
runtime preserves those input types and connectivity and replaces only the
coordinates and frozen AM1-BCC charge column. `parmchk2` must report no
`NONBON` overrides; otherwise the provider fails closed. This is necessary
because Antechamber retyping differs from the frozen input types in 95 of the
526 development cases. GBNSR6 execution is serialized both within a process
and across POSIX processes because concurrent AmberTools 26 jobs showed
allocator failures.

The provider parses only GBNSR6 `EGB` and PBSA `ECAVITY + EDISPER`. Bonded
terms and gas-phase MM energy are never read into the result. A live runtime
parity case reproduces all frozen components within
`6e-13 kcal/mol`; the 526-case topology audit and executable hashes are stored
in
[`route1-chagb-runtime-provider-parity-2026-07-25.json`](benchmarks/route1-chagb-runtime-provider-parity-2026-07-25.json).
The same artifact records a real CPU MACE-OFF23m SP with exact additive closure
and only `energy/free_energy/solvation` results, confirming the named adapter
without opening a universal-MLIP claim.
This promotes an auditable SP implementation, not the endpoint to the default
and not a force, OPT, SCAN, sampling, or confirmation claim.

The derivative boundary is independently frozen in
[`route1-chagb-derivative-capability-audit-2026-07-25.json`](benchmarks/route1-chagb-derivative-capability-audit-2026-07-25.json).
At pinned AmberClassic commit
`0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`, the driver passes a force array
into `egb`, but `gb_equation` and `chagb_equation` accumulate only energy and
have no force-array argument. The active cavity path computes energy from
`prtsas`/`prtsav` while its derivative call is commented; only part of the
dispersion path writes force components. AmberTools 26 `debugf` then reports
nonzero numerical components beside zero analytical components for the audited
methyl-hexanoate atom. This is direct rejection evidence, not merely an
inference from a manual or from the presence of an MD/minimization driver.

## What the MLIP contributes

1. **Fixed-geometry SP composition**: the MLIP supplies the solution potential's
   gas term, but it cancels from
   \(E_{\mathrm{solution}}(R)-E_{\mathrm{MLIP,gas}}(R)\). A fixed-geometry
   FreeSolv error therefore evaluates the solvent endpoint, not MLIP accuracy.
2. **OPT, relaxed SCAN, and MD**: MLIP and solvent forces jointly choose the
   geometry or sampled distribution and the gas reorganization cost. These are
   the product tasks where the MLIP can change a solvent-sensitive prediction.
3. **Conformer integration**: relative MLIP conformer energies can enter a
   population integral. Current discrete, relaxed, and Metropolis/TI studies
   are development diagnostics; their AM1-BCC evidence is negative and they
   are not defaults. A two-level low-force OPT/high-energy final-SP diagnostic
   likewise exposes a legitimate MLIP gas reorganization term, but it worsens
   the six-case fixed-geometry high-level baseline and is not promoted.
   The force-consistent OBC-II/ACE TI probe also gives the same qualitative
   result for MACE-OFF23m, AIMNet2, and ANI2x: on three development cases,
   sampling changes the endpoint by only about `0.1-0.25 kcal/mol` and worsens
   MAE by `0.047-0.083 kcal/mol`. This proves that the MLIP can contribute a
   genuine model-dependent ensemble term; it also shows that the present
   solvent-provider error dominates that term.

The reusable discrete analysis entry point is
`maple.function.free_energy.analyze_discrete_conformer_ensemble`. It consumes
same-state gas-phase MLIP energies and fixed-charge solvent corrections in
`kcal/mol`; it does not select a gas model, generate conformers, optimize
structures, read experimental labels, or add a gas-phase MM term. Consequently
the estimator is compatible with energy arrays from any registered MLIP while
conformer generation and completeness remain explicit protocol
responsibilities.

The sealed
[`route1-discrete-conformer-core-replay-2026-07-25.json`](benchmarks/route1-discrete-conformer-core-replay-2026-07-25.json)
passes 40 historical MACE-OFF23m records (2,596 states: 20 AM1-BCC/OBC-II and
20 ABCG2/OBC-II cases) through that core. All 11 common legacy fields reproduce
exactly, but only 14/20 cases in each charge set pass the default weight
diagnostic. This establishes an extracted MLIP-agnostic interface and catches
obvious weight-concentration failures. Because the named replay is MACE-only
and the discrete states omit basin measures, conformer completeness, and
uncertainty, it neither demonstrates multi-MLIP scientific accuracy nor opens
the public `#solvfe` task.

A prospectively frozen follow-up now provides named multi-MLIP evidence without
changing the estimator or solvent endpoint. MACE-OFF23m, AIMNet2, and ANI2x
each evaluate the same 1,270 states from 19 common-domain cases with the same
fixed AM1-BCC/OBC-II/ACE correction. MAPLE obtains the supported atomic-number
table from each loaded checkpoint and rejects unsupported input elements before
the first energy evaluation; the phosphorus case is therefore excluded by the
predeclared intersection rule rather than by an experimental label.

Across those 19 cases, the median and p90 ranges of the final discrete
correction across the three MLIPs are only `0.03597` and
`0.11969 kcal/mol`, but alachlor reaches `1.41020 kcal/mol`. More importantly,
the weight gate passes only `13/19` MACE, `4/19` AIMNet2, and `12/19` ANI2x
records; AIMNet2 and ANI2x have median gas/solution weight overlaps of only
`0.2141/0.2013`. This proves that the common interface exposes genuine
model-dependent conformer populations. It also requires Route 1 to fail closed
instead of treating similar final corrections as proof of population
agreement.

The label-free artifact was sealed before separate development scoring.
Fixed-geometry MAE is `1.930 kcal/mol`; AIMNet2, MACE-OFF23m, and ANI2x
discrete results are `1.910`, `1.969`, and `1.993 kcal/mol`. AIMNet2's apparent
`0.020 kcal/mol` gain and both degradations have paired bootstrap intervals
that cross zero. No model is selected, no accuracy promotion is allowed, and
the public `#solvfe` task remains closed.

The common calculator protocol now also exposes
`calculate_many(atoms_list, properties) -> BatchResult`. Every `CalcABC`
backend and the shipped third-party UMA wrapper have a result-driven sequential
fallback. MACE-OFF23m constructs one disconnected no-PBC graph, AIMNet2 uses
one same-molecule-masked `mol_idx` batch, and ANI2x groups states by identical
element order. PBC, D4, and attached implicit-solvent cases fall back to the
validated single-structure path; batched Hessians are rejected. The public
`evaluate_gas_conformer_energies` helper chunks one fixed-topology conformer
array and rejects a calculator with an attached solvent correction, preventing
double counting before the fixed \(W_i\) array is applied.

A review-amended 19-case/1,270-state CUDA confirmation at common batch size 16
passes the original `0.001 kcal/mol` relative-energy and propagated
discrete-correction limits plus the v2 absolute-energy gate using the same
tolerance converted to Hartree. That absolute gate and runner provenance,
warmup, and decision repairs were added after review of the initial full run,
so v2 is not presented as wholly pre-data. A preserved v2 review run then
showed that MACE's ratio-of-medians could pass while its two paired repeats
disagreed (`1.060x` versus `1.619x`). Protocol v3 therefore uses four balanced
repeats and requires every paired repeat to clear the unchanged `1.25x` floor;
after a discarded provenance refresh was visibly contaminated by concurrent
GPU work, v5 added preflight/postflight GPU snapshots and a preflight
one-minute host load no greater than `0.25` per logical CPU. Protocol v6
clarifies that these are endpoint screens, not proof of whole-run exclusivity:
continuous monitoring and whole-run exclusivity are explicitly false in the
artifact. It writes no artifact on an endpoint or host-preflight failure and is
likewise disclosed as review-amended. Median
serial-to-batch speedups are
`0.99x`, `1.42x`, and `12.39x`, respectively. The interface is therefore
numerically admissible, but the all-repeat `1.25x` speed gate fails for MACE
(minimum paired repeat `0.616x`); AIMNet2 and ANI2x pass all four repeats
(minima `1.283x` and `11.298x`). Route 1 may report the observed named timings
and the admitted AIMNet2/ANI2x accelerations; it may not claim a universal
material batch speedup, stable MACE promotion, improved solvent accuracy, or
superiority to bare MM. Evidence:
[`route1-multi-mlip-conformer-batch-parity-2026-07-25.json`](benchmarks/route1-multi-mlip-conformer-batch-parity-2026-07-25.json).

The current executable compatibility trace uses the same fixed AM1-BCC
OBC-II/ACE correction through two independent registered gas backends:
MACE-OFF23m (`float64`) and AIMNet2 (`float32`). Both close the additive energy
and force decomposition, pass an all-`3N`-component finite-difference check of
the combined potential at `0.003 A`, and lower the combined potential in a
two-step BFGS smoke. Three manually displaced geometries also return finite
combined energies; those points do not execute the MAPLE SCAN dispatcher. The
solvent energy is identical across the two gas models at the common geometry.
This proves the composition boundary for those named adapters; it is not
universal MLIP chemical-accuracy evidence.

An independent task-matrix trace now executes MAPLE's actual `engine` and
dispatcher for MACE-OFF23m, AIMNet2, and ANI2x. Each backend completes SP, a
two-iteration MAPLE LBFGS OPT, a three-point rigid MAPLE SCAN, and a
deterministic four-step NVT trajectory from the same normalized AM1-BCC MOL2.
All twelve jobs retain the structured solvation result and audit manifest,
close the energy decomposition, and create the task-specific output. This
closes the named-adapter task-plumbing gap; the deliberately short jobs do not
certify broad optimization, scan, or MD stability and are not equilibrated
free-energy sampling.

The composed expression is a configurational potential. It is not called a
hydration free energy unless a protocol performs and validates the required
configuration integral and standard-state convention.

## Ensemble free-energy contract

Route 1 uses the same fixed-charge additive potential at every alchemical
window:

\[
U_\lambda(R)=U_{\mathrm{MLIP,gas}}(R)+\lambda W(R),
\qquad
W(R)=G_{\mathrm{polar}}(R,q_{\mathrm{fixed}})
     +G_{\mathrm{nonpolar}}(R),
\qquad 0\leq\lambda\leq1.
\]

The charge vector, atom order, radius profile, polar provider, nonpolar
provider, and provider parameters are frozen across all windows. Only the
dimensionless coupling parameter changes. The target quantity for one
connected neutral solute is

\[
\Delta G_{\mathrm{solv,model}}^{*,1\mathrm{M}\rightarrow1\mathrm{M}}
=-k_BT\ln\frac{Z_1}{Z_0}.
\]

Using the same molecular coordinate measure at both endpoints makes
center-of-mass translation, the global orientation-group volume, momenta, and
the MLIP energy zero cancel. The reported convention is gas 1 M to ideal-dilute solution 1 M;
MAPLE does not silently add the approximately 1.89-kcal/mol 1-atm-to-1-M
conversion. A different requested standard state must be represented by a
separate, explicit correction and provenance field.
This exact statement does not remove the geometry-dependent
\(Q_{\mathrm{rot},j}/\sigma_{\mathrm{rot},j}\) carried by each local RRHO
basin; retaining it defines a selected-minimum local surrogate, not an exact
basin integral.

The production analysis hierarchy is:

1. **multi-window MBAR** is the primary estimator because it uses all sampled
   states and exposes the overlap matrix;
2. **TI**, integrating
   \(\langle\partial U_\lambda/\partial\lambda\rangle_\lambda
   =\langle W\rangle_\lambda\), is an independently reported cross-check;
3. **endpoint Zwanzig FEP** is diagnostic-only and may be reported only when
   its effective sample size and endpoint overlap pass the same fail-closed
   policy.

Sampling and analysis must remain separate from experimental scoring.
Production records must contain the complete reduced-potential matrix, lambda
schedule, random seeds, equilibration cutoff, statistical inefficiency,
effective independent sample counts, MBAR overlap matrix, uncertainty,
TI/MBAR agreement, provider manifest, and hashes of the analyzed trajectories.
FreeSolv labels may be opened only after those records and the scoring
partition are frozen.

An ensemble result fails closed rather than returning a promotable free energy
if any of the following is true:

- a window is not equilibrated or has fewer effectively independent samples
  than the prospectively frozen protocol requires;
- the MBAR overlap matrix is not connected at least through adjacent windows,
  or either first off-diagonal of an adjacent pair is below `0.03`;
- independent repeats, forward/reverse accumulation, or leave-one-window-out
  analysis exceed their predeclared uncertainty-aware tolerances;
- MBAR and TI disagree beyond the predeclared combined statistical and
  integration uncertainty;
- any window changes the fixed charge/provider lifecycle or evaluates a
  different physical endpoint.

The `0.03` overlap value is a caution boundary from published analysis
guidance, not a substitute for uncertainty or convergence analysis. Window
placement must be refined when it is missed.

The short prospective multi-MLIP TI screen is a fail-closed development
screen. A reusable MAPLE analysis API now delegates equilibration detection,
decorrelation, MBAR, effective sample counts, and overlap to upstream PyMBAR
through the optional `implicit-free-energy` extra. Post hoc analysis of all
nine label-free MACE-OFF23m, AIMNet2, and ANI2x records finds a minimum
directional adjacent overlap of `0.145`, maximum absolute MBAR/TI difference of
`0.112 kcal/mol`, and maximum independent-repeat difference of
`0.239 kcal/mol`. Those narrow diagnostics pass, but only `9-19` decorrelated
samples remain per state against the diagnostic minimum of 20, and the 30-fs
windows cannot establish equilibrium or conformer mixing. Consequently no
record passes the complete gate and the end-to-end public solvation-free-energy
task remains closed. A four-step MD smoke or a 30-fs lambda window is not a
hydration-free-energy product result.

The sealed MBAR records were scored only in a separate development phase.
Relative to the common `2.150 kcal/mol` fixed-geometry three-case MAE, MBAR
gives `2.207`, `2.212`, and `2.219 kcal/mol` for AIMNet2, ANI2x, and
MACE-OFF23m. Thus exposing the correct estimator confirms the previous
qualitative TI result; it does not create an accuracy gain.

### Energy-only high-endpoint correction boundary

Route 1 may reuse an equilibrated force-capable solution ensemble to evaluate
a more accurate energy-only fixed-charge solvent provider:

\[
U_{\mathrm{low}}(R)=E_{\mathrm{MLIP,gas}}(R)+W_{\mathrm{OBC2/ACE}}(R),
\]

\[
U_{\mathrm{high}}(R)
=E_{\mathrm{MLIP,gas}}(R)+W_{\mathrm{CHA\mbox{-}GB/PBSA}}(R).
\]

The exact one-sided correction, when the low ensemble is equilibrated and has
target support, is

\[
\Delta F_{\mathrm{low}\rightarrow\mathrm{high}}^{\mathrm{solution}}
=-RT\ln\left\langle
\exp\{-\beta(W_{\mathrm{CHA\mbox{-}GB/PBSA}}
-W_{\mathrm{OBC2/ACE}})\}
\right\rangle_{\mathrm{low}},
\]

\[
\Delta G_{\mathrm{solv}}^{\mathrm{high}}
=\Delta G_{\mathrm{solv}}^{\mathrm{low}}
+\Delta F_{\mathrm{low}\rightarrow\mathrm{high}}^{\mathrm{solution}}.
\]

The gas correction is exactly zero because both solvent endpoints retain the
same selected gas MLIP. The gas-MLIP term cancels from each same-geometry
endpoint energy difference, but it still determines the sampled solution
population. No MM gas energy, residual model, label fit, retraining, or
MLIP-specific solvent term is permitted.

The frozen 3-MLIP x 3-case pilot used the existing 30-fs OBC-II/ACE
lambda-one trajectories and 720 label-free CHA-GB/PBSA energy evaluations.
The additional high-endpoint work took `21.64 s` locally. The observed
one-sided weights are numerically benign (`minimum ESS fraction=0.911`,
`maximum normalized weight=0.136`), but PyMBAR equilibration/decorrelation
retains only `4-13` samples per replicate. Therefore `0/9` records pass the
predeclared numerical gate and `0/9` pass the scientific gate.

Separate development scoring gives corrected three-case MAEs of `1.725`,
`1.646`, and `1.777 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m, versus the
common `2.150 kcal/mol` fixed-geometry OBC-II/ACE baseline. This signal is
dominated by one propionic-acid case; methyl hexanoate worsens for all three
MLIPs. It motivated the broader screen below, but never justified promotion.
It is not a speed comparison with bare MM, a target-ensemble validation, or a
public solvation-free-energy result.

### 19-case common-state high-endpoint screen

The broader prospective screen reuses the frozen 19-case, 1,270-state
common-coordinate matrix for MACE-OFF23m, AIMNet2, and ANI2x, leaves every gas
MLIP energy unchanged, and replaces only the solvent correction by
fixed-AM1-BCC/CHA-GB/PBSA cavity-dispersion. The label-free energy phase makes
2,540 high-endpoint evaluations, two per state, in `145.51 s` locally
(`72.63/72.05 s` by repeat). Both the solvent arrays and the propagated
discrete free energies repeat exactly.

The common fixed-geometry CHA-GB/PBSA MAE is `1.700 kcal/mol`. Applying the
same finite-state partition ratio with the high endpoint gives
`1.699`, `1.720`, and `1.709 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m.
The paired MAE gains relative to fixed high-endpoint geometry are
`+0.001`, `-0.020`, and `-0.009 kcal/mol`, with 95% bootstrap intervals
`[-0.148, 0.173]`, `[-0.087, 0.047]`, and `[-0.073, 0.056]`. Only `6/19`,
`7/19`, and `7/19` cases improve, and high-endpoint weight diagnostics pass
only `4/19`, `12/19`, and `13/19` cases.

Every frozen long-sampling signal gate therefore fails. The sealed decision is
`long_sampling_candidate_not_supported` in
[`route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json),
linked to the label-free
[`route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json).
The improvement relative to the low OBC-II/ACE discrete results comes from
changing the fixed-charge solvent provider; MLIP-dependent finite-state
weighting adds no demonstrated accuracy beyond the high-endpoint reference
geometry. Route 1 should not spend a large MD budget on this construction
without a new, independently justified sampling hypothesis.

### Reference-potential acceleration boundary

Route 1 may use a cheap MM/GB Hamiltonian to generate candidate configurations
without adding that MM energy to the target potential. The exact indirect
cycle is

\[
\Delta G_{\mathrm{solv}}^{\mathrm{target}}
=\Delta G_{\mathrm{solv}}^{\mathrm{reference}}
+\Delta F_{\mathrm{reference}\rightarrow\mathrm{target}}^{\mathrm{solution}}
-\Delta F_{\mathrm{reference}\rightarrow\mathrm{target}}^{\mathrm{gas}}.
\]

Here the target remains exactly
\(U_{\mathrm{MLIP,gas}}+G_{\mathrm{polar}}+G_{\mathrm{nonpolar}}\).
The MM/GB energy is a sampling reference only. Its arbitrary energy zero and
the large absolute MM-to-MLIP corrections cancel between gas and solution.
Reference-only EXP requires sparse target **energies**, not target forces;
bidirectional MBAR is the primary validation estimator and two-state BAR is an
independent cross-check. Poor overlap must fail closed or trigger additional
reference-to-target states/nonequilibrium switching.

Cycle components share sampled configurations. Their individual uncertainty
estimates must not be combined as though independent; until a joint
chain-aware covariance or resampling analysis exists, Route 1 reports the
component uncertainties and no aggregate cycle confidence interval.

The first frozen three-MLIP, three-molecule diagnostic does not pass that gate:

- OBC-II/ACE solvent terms from the reference and target implementations agree
  within `1.83e-5 kcal/mol`, proving that the experiment changes the sampled
  gas Hamiltonian rather than the solvent provider;
- each accelerated record uses 400 sparse MLIP energy evaluations and zero
  MLIP force evaluations, but the observed timing is hardware-specific and is
  not a comparison with a bare MM force call;
- minimum gas/solution BAR overlaps are `0.015/0.057`, minimum directional
  MBAR overlap entries are `0.005/0.016`, the minimum reference-only
  gas/solution ESS fractions are `0.074/0.021`, and all nine gas and all nine
  solution forward/reverse agreement checks fail;
- only `7/9` records pass every explicit MBAR/BAR solver-convergence check;
  the two failures preserve PyMBAR's final nonconvergence messages in the raw
  artifact and fail closed;
- the accelerated cycle differs from bidirectional MBAR by as much as
  `2.514 kcal/mol`, while bidirectional indirect MBAR differs from the direct
  target multistate MBAR by as much as `1.378 kcal/mol`;
- separate development scoring gives reference-only MAEs of `3.999`, `3.514`,
  and `3.025 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m, versus the common
  `2.150 kcal/mol` fixed-geometry baseline.

This is a valid generic extension mechanism but a negative endpoint-FEP
result. It is not part of the baseline product workflow. The next scientifically
admissible escalation is prospectively frozen intermediate Hamiltonians or
nonequilibrium switching, not a hydration-label residual or a hidden
MLIP-specific correction.

### Nonequilibrium reference-to-target bridge diagnostic

The remaining force-based bridge was also tested without changing the target
Hamiltonian. In each phase, MAPLE propagated a linear mixed potential

\[
U_{p,\lambda}(R)=(1-\lambda)U_p^{\mathrm{reference}}(R)
+\lambda[U_p^{\mathrm{target}}(R)-C],
\qquad p\in\{\mathrm{gas},\mathrm{solution}\},
\]

where the same model/case constant \(C\) was used in gas and solution. It
changes neither force and cancels exactly from the indirect cycle. The target
remained the selected gas MLIP plus AM1-BCC/OBC-II/ACE in solution; the GAFF2
energy remained a switching/reference Hamiltonian only.

The prospective 3x3 diagnostic used MACE-OFF23m, AIMNet2, and ANI2x; rigid,
limited, and flexible development molecules; eight forward and eight reverse
work values per phase; and 5-fs and 20-fs switches. PyMBAR BAR was the
bidirectional estimator and directional EXP was retained as a diagnostic.
The 20-fs indirect cycles approach the deliberately short direct-target MBAR
control better than endpoint reweighting, but no record passes the complete
numerical gate:

- minimum gas/solution BAR overlaps are `0.087/0.072`;
- maximum 20-fs indirect-versus-direct-target MBAR difference is
  `0.492 kcal/mol`;
- maximum 5-fs-to-20-fs cycle change is `1.364 kcal/mol`;
- the gas and solution legs retain large direction and uncertainty failures,
  so apparent cycle agreement cannot be credited to convergence;
- all `9/9` records fail at least one numerical gate and all endpoint
  equilibrium/independence claims remain false.

Separate development scoring gives:

| gas MLIP | fixed geometry MAE | endpoint indirect MAE | 20-fs switching MAE | direct target MBAR MAE |
|---|---:|---:|---:|---:|
| AIMNet2 | 2.150 | 2.749 | 2.281 | 2.207 |
| ANI2x | 2.150 | 2.805 | 2.468 | 2.212 |
| MACE-OFF23m | 2.150 | 2.442 | 2.242 | 2.219 |

All values are kcal/mol. Switching improves on the failed endpoint correction
for every MLIP but worsens the fixed-geometry experimental MAE for every
MLIP. This distinction is essential: the bridge can estimate the selected
MLIP ensemble more faithfully without repairing the fixed-charge solvent
functional.

The diagnostic consumes `29,376` target energy+force evaluations
(`3,264` per model/case) plus the same number of reference evaluations.
The frozen direct-target short diagnostic nominally uses `3,010` target
energy+force evaluations plus `400` target energy-only analyses per
model/case. The switching experiment therefore does not establish an
acceleration even against that deliberately short direct control. Published
solvation work uses hundreds of switches several picoseconds long; escalating
this 20-fs probe to that regime would increase cost by orders of magnitude and
still would not change OBC-II/ACE's physical bias. Nonequilibrium switching
remains a reusable, model-agnostic analysis/bridge primitive, not a Route 1
default or an accuracy/speed claim.

### Explicit-inner / implicit-outer contract

The minimal general runtime extension is intentionally prebuilt rather than an
automatic water-shell generator:

```text
#charge(source=mol2,label=prebuilt-fixed-cluster)
#solv(implicit=water,inner=prebuilt,method=gb,experimental=true)
```

The supplied neutral closed-shell MOL2 contains the solute and at least one
disconnected inner-solvent component, explicit bonds, and one fixed charge per
atom. The graph components become separate synthetic continuum-provider
residues. The gas MLIP evaluates the full cluster, and the outer continuum
acts on that same full coordinate and charge vector:

\[
E_{\mathrm{cluster,outer}}(R)=
E_{\mathrm{MLIP,gas}}(\mathrm{cluster};R)
+G_{\mathrm{polar,outer}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{nonpolar,outer}}(R).
\]

The returned quantity is named a **fixed-shell cluster-continuum
configurational potential**. Its structured output contains
`cluster_continuum_correction_hartree`, not `delta_g_solv_hartree`, and the
audit manifest sets `absolute_solvation_free_energy_claim=false`. “Fixed
shell” freezes the component membership and coordination number \(n\), not
the atomic coordinates varied by OPT or SCAN.

For a cluster cycle with solute \(A\) and \(n\) waters, an absolute 1 M
solvation free energy instead has the form

\[
\Delta G_{\mathrm{solv}}^*(A)=
\Delta G_{\mathrm{clust,g}}^*(A(\mathrm{H_2O})_n)
+\Delta G_{\mathrm{solv}}^*(A(\mathrm{H_2O})_n)
-\Delta G_{\mathrm{solv}}^*((\mathrm{H_2O})_n)
-RT\ln([\mathrm{H_2O}]/n).
\]

Route 1's prebuilt runtime does not compute the cluster-formation/occupancy
term, solvent-cluster reference, standard-state conversion, or cluster
conformer/coordination-number ensemble. Neutral solutes avoid the single-ion
absolute-reference convention but not those four requirements. Consequently,
the mode cannot be scored against FreeSolv as a solute hydration prediction.

The existing coordinate-only `explicit=water` generator is not silently
composed with `implicit=water`; that would lack a complete charge/topology and
inner-shell sampling contract. A future automatic generator must first supply
those artifacts and a separate thermodynamic-cycle workflow.

### Automatic explicit-inner / implicit-outer feasibility boundary

The strongest maintained upstream scaffold found for closing the missing
thermodynamic cycle is CREST Quantum Cluster Growth (QCG). Unlike the current
prebuilt Route 1 mode, QCG generates solute-solvent and reference
pure-solvent ensembles and evaluates a supermolecular free-energy difference
with conformational and thermochemical contributions. This is the correct
class of workflow to investigate; a shell builder by itself is not.

The current upstream nevertheless does not satisfy Route 1's arbitrary-MLIP
contract. QCG is published and documented around GFN-xTB/GFN-FF. The audited
CREST source directly launches xTB for QCG single points, optimizations, and
Hessians, and its QCG ensemble/frequency selectors accept only GFN-family
levels. CREST has a `generic` energy/gradient subprocess for other
calculator-driven workflows, but QCG does not call it. Therefore CREST's
generic backend is not wired into QCG; exposing a generic backend elsewhere
in CREST cannot be treated as end-to-end multi-MLIP validation.

FEBISS is an optional proposal preprocessor. It consumes an already generated
explicit-solvent trajectory and topology, performs GIST analysis, ranks solvent
sites, and writes a selected microsolvated structure. It does not compute the
complete solute transfer free energy and cannot supply QCG's reference-solvent,
standard-state, target-ensemble, or cluster-size terms.

Even a future backend adapter is not automatically universal. One target MLIP
must cover the solute, solvent, solute-solvent clusters, and pure-solvent
clusters with a consistent size-extensive energy; its intermolecular domain,
elements, charge/spin, forces, and any required Hessian must be validated.
Atomic reference-energy zeros can cancel in the atom-conserving
\(A(S)_n-(S)_n-A\) cycle, but that cancellation does not validate
solute-solvent interactions, ensemble overlap, thermochemistry, or the outer
continuum. If a cheaper QCG/xTB or MM ensemble proposes states for a target
MLIP, MBAR or nonequilibrium switching must fail closed on poor overlap rather
than silently reweight a nonrepresentative ensemble.

The present decision is
`no_admissible_neutral_small_molecule_product_path`. No automatic inner-shell
runtime, dependency, or FreeSolv selector is added. The existing
`inner=prebuilt` mode remains an experimental fixed-shell potential only. A
research path may reopen when a maintained full-cycle arbitrary-backend
adapter exists, every admitted MLIP passes the cluster-domain gates,
conformer/coordination/cluster-size convergence and uncertainty are explicit,
a prospective neutral aqueous benchmark improves Route 1 accuracy, and
end-to-end cost beats a like-for-like MM workflow. The frozen decision record
is
[`route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json`](benchmarks/route1-explicit-inner-implicit-outer-feasibility-audit-2026-07-25.json).

## Scientific gates

The product evidence is deliberately split:

1. **Provider fidelity**: energy/force parity against an independent named
   upstream implementation.
2. **Composition consistency**: finite-difference checks on the complete
   MLIP-plus-solvent force and failure on unsupported derivatives.
3. **Task stability**: OPT/SCAN convergence, constraints, restarts, and
   perturbed geometries for each declared MLIP/provider pairing.
4. **Experimental hydration accuracy**: one frozen candidate followed by a
   label-sealed holdout or external evaluation. The legacy 116-record FreeSolv
   partition has now been evaluated with energies sealed before scoring, but
   its labels were written into the earlier preparation manifest before
   candidate freeze. It is therefore useful held-out-by-computation evidence,
   not the stricter independent or label-sealed confirmation.
5. **Applicability**: failures and element/size/functional-group strata are
   retained, not removed from the denominator.

Engineering tests and development-set improvements do not close the scientific
confirmation gate.

### Amber PB exact-difference force probe

The tempting internal construction

\[
\Delta G_{\mathrm{candidate}}(R)=
E_{\mathrm{MM,PB(inp=2)}}(R)-E_{\mathrm{MM,vacuum}}(R)
\]

does remove the gas-phase MM energy algebraically before the correction is
added to an MLIP. It therefore does not violate the Route 1 energy formula by
itself. It does **not**, however, restore a conservative solvent correction.

The local AmberTools probe on methyl hexanoate subtracts the corresponding
vacuum force from the `sander` PB `inp=2` force and compares that result with
centered finite differences of the same energy difference. At the Route 1
`0.003 A` force-check step, the four sampled Cartesian errors are `0.0750`,
`2.5697`, `2.0494`, and `3.4200 kcal/mol/A`. Increasing the step to `0.01 A`
does not repair the path: the maximum error remains `1.8108 kcal/mol/A`.
The stored bond, angle, dihedral, direct and 1-4 electrostatic, and van der
Waals terms cancel to within `5.4e-15 kcal/mol`, so this failure is not a
leftover gas-MM energy.
Consequently this composition is retained only as a single-point reference and
is rejected for OPT and relaxed SCAN. The executable trace is
`benchmarks/amber-pb-inp2-force-probe-methyl-hexanoate-2026-07-24.json`.

### Maintained analytical candidates already in the product stack

Among the analytical alternatives already available through the current
OpenMM provider stack, none passes both the material-accuracy and
applicability gates:

- neutral-water ALPB reduces the AM1-BCC/OBC-II/ACE development MAE by only
  `0.00235 kcal/mol`; the paired bootstrap interval crosses zero;
- the frozen July 24 GBn2 screen is force-capable but has a worse development
  MAE even on its identical 515-case success subset (`1.8940` versus OBC-II's
  `1.6615 kcal/mol`) and records 11 phosphorus failures. The later independent
  parity expansion found that OpenMM's generic GBn2 expression also disagrees
  with Amber's signed sulfur descreening branch, so the current runtime fails
  closed for both P- and S-containing GBn2 inputs rather than preserving the
  older screen's sulfur fallback;
- OpenMM's LCPO implementation is force-capable where parameterized, but its
  upstream topology table rejects 72 of the 526 development molecules. On the
  identical 454-case supported subset, OBC-II/LCPO worsens MAE from ACE's
  `1.801` to `2.250 kcal/mol`; the paired ACE-minus-LCPO gain is
  `-0.449 kcal/mol` with 95% interval `[-0.581, -0.318]`, and 285/454 cases
  worsen. The OBC-II polar component agrees exactly, so the degradation is
  isolated to the nonpolar replacement. This is a post-hoc development screen,
  not an independent or label-blind confirmation.

The deterministic evidence is
`benchmarks/route1-force-consistent-candidate-screen-2026-07-24.json`. None is
promoted merely because it has analytical forces. The next accuracy advance
must be an independently justified conservative provider or an explicitly
two-level workflow that keeps final-SP and optimization potentials distinct;
it cannot be a hidden parameter fallback. The later external ddX audit below
also fails product admission, but for accuracy and speed rather than derivative
correctness.

### GBMV2 physical-provider feasibility boundary

GBMV2 is the strongest remaining **physical**, non-learned candidate identified
by this audit. It has the required additive form

\[
E_{\mathrm{solution}}(R)
=E_{\mathrm{MLIP,gas}}(R)
+G_{\mathrm{GBMV2}}(R,q_{\mathrm{fixed}},r)
+\gamma\,\mathrm{SASA}(R)+\beta,
\]

and does not require a gas-phase MM term, MLIP retraining, or a learned
hydration residual. Official CHARMM documentation describes analytical method
II as the preferred dynamics variant, exposes user-supplied radii, and provides
first derivatives for the polar and SASA terms.

The small-molecule evidence is directly relevant to the default charge policy.
Knight and Brooks evaluated 499 neutral organic molecules with GAFF/AM1-BCC,
10.5 ns vacuum and GBMV2 trajectories, and BAR. With a surface coefficient
selected from experimental hydration labels, GBMV2 reported AUE/RMSE
`1.14/1.60 kcal/mol`; 84% and 94% of the compounds were within 2 and
3 kcal/mol. A later 457-molecule comparison selected the GAFF/AM1-BCC GBMV2
surface coefficient on 82 CGENFF compounds and reported AUE `1.24 kcal/mol`
and \(R^2=0.758\) on the other 375 compounds. This is compelling historical
multi-conformer evidence, but it is not a same-protocol comparison with the
current MAPLE FreeSolv corpus and the nonpolar coefficient is label-exposed.

The blocking gate is implementation availability, not the Route 1 formula.
The current academic CHARMM distribution is available by registration for
academic, government, and nonprofit use and supplies the source plus
pyCHARMM shared library. The GPU paper describes a CHARMM/OpenMM plugin, but
the audited public OpenMM source has no GBMV/GBMV2/GBSW class, and the current
official CHARMM/OpenMM documentation names GBSW rather than GBMV2 among the
distributed plugins. No local `charmm` executable or GBMV2 OpenMM symbol is
available, so energy parity, all-\(3N\) force finite differences, deployment
rights, and small-molecule speed cannot be tested here.

Route 1 therefore records GBMV2 as
`scientifically_promising_runtime_blocked`. It does **not** reimplement the
CHARMM algorithm from the papers and adds no provider, dependency, or FreeSolv
selector. Admission may reopen only for a registered local CHARMM/pyCHARMM
runtime or a maintained redistributable upstream implementation that accepts
the declared fixed AM1-BCC charges/radii and passes provider parity, complete
force, label-separated accuracy, and multi-MLIP throughput gates. The frozen
source and decision record is
[`route1-gbmv2-feasibility-audit-2026-07-25.json`](benchmarks/route1-gbmv2-feasibility-audit-2026-07-25.json).

### FACTS/GBSW physical-provider feasibility boundary

FACTS and GBSW also have the required physical additive structure and require
neither a gas-phase MM energy nor a learned solvent residual. Historical
GAFF/AM1-BCC hydration studies are substantially better than the observed
3--5 kcal/mol concern: on 499 neutral molecules, optimized-profile GBSW and
FACTS reported AUE/RMSE `1.20/1.52` and `1.25/1.80 kcal/mol`, respectively.
The nonpolar coefficients were selected with experimental hydration labels,
so these values motivate an implementation audit rather than certify MAPLE.
When coefficients were selected on 82 CGENFF compounds, the other 375
GAFF/AM1-BCC compounds gave AUE \(1.33\) for GBSW and \(1.42\) for FACTS,
compared with \(1.24\ \mathrm{kcal/mol}\) for GBMV2.

GBSW exposes the complete electrostatic-plus-nonpolar solvation energy and
forces. Its smoothed boundary targets numerically stable electrostatic forces,
and official examples cover minimization and dynamics. This advantage does
not establish the requested faster-than-MM claim: current CHARMM documentation
says GBSW is **about four times slower than vacuum**, while its reported
`2--3x` speedup is only relative to GBMV. It also requires an explicit,
audited radius vector; the protein-specific radii and CMAP must not be silently
transferred to arbitrary small molecules.

FACTS is an analytical, MD-capable GB/SASA construction, but its speed claim is
also about four times slower than vacuum in the original paper (`3--5x` in
current peptide/protein documentation), not faster than bare MM. More
importantly, the official parameters were derived only for protein atoms.
Unknown small-molecule radii are handled by `TAVW` interpolation, and the
published surveys used interpolation or extrapolation where parameters were
missing. Together with its weaker historical accuracy, this prevents treating
FACTS as a universal provider merely because it is cheaper than GBMV/GBSW.

The implementation boundary is the same registered distribution. CHARMM c50b2
ships a GBSW OpenMM plugin with a CHARMM build, but public OpenMM contains no
named GBSW or FACTS implementation; the local OpenMM 8.5.2 installation exposes
neither symbol and no local `charmm` executable exists. Route 1 therefore adds
no provider or paper-based reimplementation. The physical watch priority is
`GBMV2 -> GBSW -> FACTS`: GBSW is a secondary force-stability candidate;
FACTS remains unselected until a deployable upstream and prospectively valid
small-molecule parameters exist. The frozen evidence and decision are
[`route1-facts-gbsw-feasibility-audit-2026-07-25.json`](benchmarks/route1-facts-gbsw-feasibility-audit-2026-07-25.json).

### SLIC/CDC physical-provider feasibility boundary

SLIC/CDC is the strongest newly audited **energy-accuracy lead** that still
preserves the Route 1 additive boundary. It combines nonlinear SLIC continuum
electrostatics with physical/statistical cavity, atom-typed dispersion,
combinatorial-mixing, and hydrogen-bond terms. The 2022 study used the Mobley
neutral-molecule structures and fixed AM1-BCC charges, so it does not add a
gas-phase MM energy, retrain the gas MLIP, or introduce a learned hydration
residual.

It is nevertheless a fitted physical model, not a parameter-free formula. The
published aqueous SLIC/CDC profile has 38 fitted physical-model parameters,
uses experimental total hydration data and explicit-solvent component data,
and says that 65 neutral compounds were used for training. The accompanying
Table S4 actually lists 63 names. Therefore the paper's sub-kcal average error
is relevant evidence, but not a label-blind product certification.

The same public SI PDF was processed with Poppler's layout and raw extraction
modes. Both modes yield the same 494 numerical rows in Table S10, representing
492 normalized unique names; Table S4's 63 training names are parsed from its
21-by-3 layout grid. Directly recomputing the first two columns gives MAE/RMSE
`0.813/1.152 kcal/mol`, not the SI footer's `0.69/0.98`; the
recomputed RMSE agrees with the ChemRxiv preprint's `1.15 kcal/mol`. Removing
the 63 explicitly listed training names leaves 431 historical rows with
MAE/RMSE `0.826/1.190 kcal/mol`. That is encouraging, but it is only a
post-publication table holdout, **not a prospective MAPLE reserve**. The
abstract's 500-solute claim, Appendix E's 502-corpus statement, the 494
tabulated rows, and the two RMSE summaries must be reconciled by a source-level
reproduction before any headline number becomes a Route 1 gate.

No complete executable implementation of the published 2022 SLIC/CDC model was
identified. The journal Figshare record contains only the SI PDF. PBJ is a
useful MIT-licensed related SLIC boundary-element code, but **PBJ is not the
published SLIC/CDC model**: it exposes three SLIC shape parameters rather than
the fitted molecular SLIC profile, supplies a simple SASA nonpolar energy
rather than complete CDC terms, has no nonpolar force, and reports only a
three-component total solvation-force vector rather than a complete per-atom
force array. Its README still calls force calculation under development and
its tests do not exercise SLIC forces. The older Bitbucket MATLAB source is an
ion/solvent parameter-optimization code, not the 2022 neutral-molecule model,
and has no top-level license.

Route 1 therefore records SLIC/CDC as
`scientifically_promising_no_complete_upstream_provider`. It is retained as an
energy-only final-SP watch item, while the force-capable physical watch order
remains `GBMV2 -> GBSW -> FACTS`. No provider, dependency, FreeSolv selector,
speed claim, or paper-derived implementation was added. Reopening requires a
maintained complete upstream, exact typing and parameter provenance, resolution
of the public-table discrepancies, a prospectively frozen independent
FreeSolv gate, all-\(3N\) conservative forces before OPT/SCAN/MD, and measured
multi-MLIP performance. The frozen source and decision record is
[`route1-slic-cdc-feasibility-audit-2026-07-25.json`](benchmarks/route1-slic-cdc-feasibility-audit-2026-07-25.json).

### Learned-solvent boundary: GNNIS and QM-GNNIS

GNNIS is technically attractive but does not satisfy the Baseline/Product
Route contract. The released scalar contains only the implicit-solvent term:
the upstream OpenFF vacuum force field is constructed separately, and forces
are obtained as the negative automatic derivative of the same reported
solvent energy. A local 23-atom audit therefore evaluates finite polar and
nonpolar components with exact energy closure and reproduces one coordinate
force by centered finite difference to `0.20%` relative error. This confirms
scalar-gradient consistency at the probed geometry and, together with the
separate upstream vacuum construction, shows that the observed scalar contains
no gas-MM term. It is not a global smoothness, force-conservativity, or MD
admission test.

The actual blocking fact is the training definition. Upstream source constructs
the target as the mean explicit-solvent solute force minus the vacuum OpenFF
solute force and trains with force-only mean-squared error. The GNN then scales
GB-Neck2 effective Born radii and supplies a learned SASA contribution. It is
therefore a chemistry-dependent learned solvent functional, not an unmodified
fixed-charge physical PB/GB provider. Force-only training also does not by
itself establish the absolute molecular energy gauge needed to reinterpret the
raw scalar as a validated FreeSolv hydration free energy; this Route 1 audit
does not claim an exhaustive negative result about every upstream GNNIS
free-energy workflow.

QM-GNNIS does not remove that boundary. Its published transfer is explicitly

\[
\Delta\Delta G_{\mathrm{corr}}
\approx G_{\mathrm{GNNIS}}-G_{\mathrm{GB\mbox{-}Neck2}},
\]

combined with a continuum QM model as

\[
G_{\mathrm{QM\mbox{-}GNNIS}}
=E_{\mathrm{QM,CPCM}}
+\left(G_{\mathrm{GNNIS}}-G_{\mathrm{GB\mbox{-}Neck2}}\right),
\]

with a separately reported vibrational term for conformer free energies. The
released `GNN3_Multisolvent_embedding_run_multiple_Delta` class implements that
learned difference. It requires no new QM or experimental training and is
scientifically relevant for relative conformer/NMR/IR applications, but it is
still an explicit learned delta correction. Transferring the same term to an
arbitrary gas-phase MLIP would additionally be an unvalidated inference from
the QM study. Published QM-GNNIS is also not itself a direct MLIP solvent
provider: its released ASE composition sums the Torch delta with ORCA/CPCM,
and the paper frames the result as an emulation of QM/MM with electrostatic
embedding and a nonpolarizable MM solvent. It therefore differs categorically
from Route 1's fixed-charge PB/GB-only solvent term even before considering a
new MLIP transfer.

Consequently neither GNNIS nor QM-GNNIS is integrated, screened against
FreeSolv for provider selection, or described as Route 1 innovation. They
belong to a separately named learned-solvent research route if that boundary
is ever opened. The source commits, model/license/paper hashes, exact
classification, and limited feasibility smoke are frozen in
[`route1-learned-solvent-boundary-audit-2026-07-25.json`](benchmarks/route1-learned-solvent-boundary-audit-2026-07-25.json).
Technical feasibility is deliberately not used to weaken acceptance item 6.

### APBS molecular-surface PB + ACE screen

The remaining PB composition was tested without changing the Route 1 formula:

\[
G_{\mathrm{PB+ACE}}(R,q_{\mathrm{fixed}})
=G_{\mathrm{APBS,LPBE}}^{\mathrm{polar}}(R,q_{\mathrm{fixed}})
+G_{\mathrm{OpenMM,OBCII}}^{\mathrm{ACE}}(R).
\]

The APBS APOLAR term is discarded, so exactly one nonpolar component is added.
The ACE component is taken from the same mbondi2/OBC-II implementation used by
the baseline; no OpenMM gas-phase MM energy is evaluated or subtracted.  The
energy phase used the frozen AM1-BCC vectors and 526 source geometries without
opening experimental labels.  The candidate was commissioned after an
exploratory aniline observation, so even a positive development result would
not have been independent confirmation.

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| APBS molecular-surface LPB/APOLAR | 4.415 | 4.984 | 15.880 |
| APBS molecular-surface LPB/ACE | 1.658 | 2.535 | 14.747 |
| CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |

All values are kcal/mol.  PB/ACE improves paired MAE over OBC-II/ACE by
`0.102 kcal/mol`, with a 95% bootstrap interval `[0.016, 0.191]`, but misses
the prospectively frozen `0.15 kcal/mol` materiality threshold.  Its RMSE is
effectively unchanged.  The label-free 20-case grid check is more decisive:
changing from `97^3 @ 0.33 A` to `129^3 @ 0.25 A` changes the polar energy by
up to `0.671 kcal/mol`, with P90 `0.455 kcal/mol`; the frozen thresholds were
`0.25` and `0.15 kcal/mol`.  Because it fails both the material-gain and grid
gates and remains energy-only, the composition is rejected for runtime
promotion.  The protocol, label-free source manifest, raw records, and summary
are `benchmarks/apbs_ace_protocol.json`,
`benchmarks/apbs_ace_source_manifest.json`, and
`benchmarks/freesolv-am1bcc-apbs-ace-2026-07-25.json`.

The label-free numerical follow-up then compared the two finer approximately
32-A grids on the same preselected 20 cases.  `129^3 @ 0.25 A` to
`161^3 @ 0.20 A` gives maximum/P90 changes `0.175/0.148 kcal/mol`, so the
fine-grid pair passes the numerical gate.  A separately frozen full
`129^3` development screen gives:

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| fine-grid APBS molecular-surface LPB/ACE | 1.629 | 2.462 | 14.116 |
| CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |

Its paired MAE gain over OBC-II/ACE is `0.131 kcal/mol`, with 95% interval
`[0.050, 0.217]`.  The improvement is statistically positive on this exposed
development set but still below the predeclared `0.15 kcal/mol` product
materiality threshold.  Thus the fine grid resolves the numerical question
without changing the rejection.  Evidence is frozen in
`benchmarks/apbs_mol_grid_followup_protocol.json`,
`benchmarks/apbs-mol-grid-followup-2026-07-25.json`,
`benchmarks/apbs_ace_fine_protocol.json`, and
`benchmarks/freesolv-am1bcc-apbs-ace-fine-2026-07-25.json`.

### APBS spline-force and APOLAR derivative closure

APBS documents electrostatic force output, but its molecular-surface
(`srfm=mol`) endpoint cannot supply the required per-atom gradient. The pinned
official APBS 3.4.1 executable aborts that request with
`Forces *must* be calculated with spline-based surfaces!`. Route 1 therefore
tested `srfm=spl4` as a separate polar-only candidate without changing the
production provider.

The frozen methyl-hexanoate audit checks all 69 Cartesian components at
`0.003 A` and `0.010 A` on both `129^3 @ 0.25 A` and
`161^3 @ 0.20 A` grids. At the finer grid, the analytical-to-energy finite-
difference RMSE is `0.199 kJ/mol/A` and the maximum error is
`1.103 kJ/mol/A`. Coarse-to-fine force differences have RMSE
`0.483 kJ/mol/A` and maximum `2.579 kJ/mol/A`; the reported fine-grid net
force norm is `2.336 kJ/mol/A`. Only the polar-energy grid check passes.
Moreover, APBS documents that spline surfaces require force-field
reparameterization; the generic mbondi2 radii used by the SP provider are not
a validated SPL4 radius profile.

The nonpolar term fails independently. For the locked APBS `gamma*SASA`
profile, the calculation-block total force compared with centered energy
differences at the matching `0.05 A` displacement has RMSE
`1.114 kJ/mol/A`, maximum error `4.871 kJ/mol/A`, and net-force norm
`3.694 kJ/mol/A`. The `PRINT APOL` block is an unscaled surface component, not
the `gamma*SASA` total force; applying its observed `-gamma` scaling still does
not repair the external energy derivative.

Evidence is sealed in
`benchmarks/route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json`
and
`benchmarks/route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json`.
Both candidates are rejected before a 526-molecule SPL4 accuracy screen or
runtime integration. The existing molecular-surface APBS endpoint remains
SP-only, and OBC-II/ACE remains the force-capable product baseline.

### GBr6 physical-provider screen

GBr6 was the last lightweight analytical literature candidate tested in this
round. Its website describes an analytical \(r^6\) GB construction and says
first derivatives are available, but the pinned GPL source release contains
an energy-only Fortran program with no force or gradient output interface.
MAPLE therefore screened its polar energy before considering any derivative
implementation.

The frozen energy phase used all 526 development molecules, fixed AM1-BCC
charges, OpenMM's upstream `gbn-bondi` radii, dielectric constants 1.0/78.5,
zero salt, and the already frozen PBSA cavity/dispersion term. It read no
experimental labels and fit no parameter. After all energy records existed,
development labels were opened:

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |
| GBr6/cavity-dispersion | 2.257 | 3.451 | 20.820 |

All values are kcal/mol. Relative to OBC-II/ACE, the paired mean MAE gain is
`-0.496 kcal/mol` with a 95% bootstrap interval
`[-0.681, -0.317]`; relative to CHA-GB it is `-0.935 kcal/mol` with interval
`[-1.115, -0.758]`. GBr6 is therefore rejected on accuracy before the missing
released derivative path is relevant. The raw 526-case energy artifact,
source/archive hashes, and deterministic summary are
`benchmarks/route1-gbr6-energy-screen-2026-07-24.json` and
`benchmarks/route1-gbr6-development-summary-2026-07-24.json`.

### ddX/ddPCM conservative-provider audit

ddX is a maintained LGPL domain-decomposition implementation of ddCOSMO,
ddPCM, and ddLPB with point-charge energies and analytical coordinate
derivatives. Route 1 built `pyddx 0.8.0` from the pinned PyPI source archive in
an isolated environment; the four shipped Python test files passed `9/9`.
No MAPLE dependency or runtime provider was added. At zero ionic strength the
appropriate model is ddPCM, the \(\kappa=0\) limit of ddLPB; finite ionic
strength would require an explicit ddLPB profile rather than silently changing
the zero-salt endpoint.

The native API has a non-obvious sign boundary:
`State.solvation_force_terms + State.multipole_force_terms` matches
\(+\partial E/\partial R\), despite the `force_terms` name. The audit therefore
adds both native terms, negates their sum, and converts Hartree/Bohr to
kJ/mol/A. On methyl hexanoate, both the mbondi2 van-der-Waals cavity and the
mbondi2+1.4-A solvent-accessible cavity pass all 69 Cartesian finite-difference
checks. For the product-relevant van-der-Waals profile at the primary `0.003 A`
step, RMSE/maximum error are `0.000404/0.001504 kJ/mol/A`, and the net-force
norm is `8.3e-14 kJ/mol/A`. The preselected label-free 20-molecule
`lmax=7, nLebedev=194` versus `lmax=13, nLebedev=590` energy check also passes:
P90/maximum differences are `0.131/0.182 kcal/mol`.

Derivative correctness does not make the provider a product improvement. The
separate 526-molecule label-blind energy phase froze fixed AM1-BCC charges and
the existing ACE nonpolar component before development labels were opened:

| endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| fine-grid APBS molecular-surface LPB/ACE | 1.629 | 2.462 | 14.116 |
| CHA-GB/cavity-dispersion | 1.322 | 1.854 | 9.416 |
| ddPCM/mbondi2/ACE | 1.782 | 2.881 | 18.813 |
| ddPCM/(mbondi2+1.4 A)/ACE | 5.829 | 6.706 | 26.717 |

All values are kcal/mol. The mbondi2 candidate's paired mean MAE gain relative
to OBC-II/ACE is `-0.022 kcal/mol`; its 95% bootstrap interval
`[-0.143, 0.098]` crosses zero, while RMSE and the largest error are worse.
The expanded-radius profile is rejected decisively.

The same local 23-atom force evaluation takes about `0.53 s`, versus a frozen
warm OpenMM OBC-II/ACE correction median of `1.47 ms`: roughly `360x` slower.
It also exceeds the named MACE-OFF23m gas time by about `7.6x` on GPU and
`1.6x` on one CPU thread. These are local diagnostic ratios, not universal
throughput claims, but they rule out a fast-product promotion here.

The conclusion is deliberately fail-closed: ddX proves that a real
force-consistent external PCM path exists, but this tested pairing provides
neither a material accuracy gain nor product-appropriate speed. It remains an
external reference candidate. OBC-II/ACE remains the force-capable product
baseline, and CHA-GB/cavity-dispersion remains the explicit higher-accuracy
SP-only profile. No remaining maintained analytical candidate evaluated in
this round passes all derivative, material-accuracy, applicability,
performance, and confirmation gates. The protocol and sealed evidence are
`benchmarks/ddx_pcm_protocol.json`,
`benchmarks/route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json`,
`benchmarks/route1-ddx-ddpcm-energy-screen-2026-07-25.json`, and
`benchmarks/route1-ddx-ddpcm-development-summary-2026-07-25.json`.

### MLSES PB surface feasibility boundary

The tested AmberTools MLSES selector (`sasopt=3`, `mlses_opt=0`, `ipb=2`)
executes GENIUSES. Its primary paper (DOI `10.1021/acs.jpclett.3c02176`)
learns classical solvent-excluded-surface point-cloud geometry; the earlier
MLSES classifier (DOI `10.1021/acs.jctc.1c00492`) is a predecessor rather than
the executed runtime. Both are scientifically admissible within Route 1 as
surface-geometry surrogates because neither fits hydration labels, molecular
forces, or a chemistry-specific residual.

That admissibility does not establish a product advantage. With the maintained
local AmberTools 26 CPU build, a complete `ENEOPT=1..4` /
`FRCOPT=1..5` input-discovery matrix identifies five runtime-accepted force
pairings under the fixed linear-PB setup: `1/1`, `2/2`, `2/3`, `2/4`, and
`3/2`. Classical SES produces a nonempty atom force for all five at both
tested grids. GENIUSES produces no atom-resolved MLSES force for any of the
five: `1/1` terminates by signal, while the other four abort during force
projection. The finite-difference gate is therefore not eligible and is not
executed. Three independent energy-only process repeats on one 23-atom
molecule also show GENIUSES slower than classical SES at both grids
(classical/GENIUSES median ratios `0.500` and `0.834`), while changing the PB
reaction-field energy by `-0.0046` and `+0.0912 kcal/mol`.

Those timing ratios are explicitly local small-molecule CPU observations, not
a claim about large systems or a GPU implementation. They are nevertheless
sufficient for the present admission question: there is no MLSES runtime
provider, no dependency or FreeSolv screen is opened, and OBC-II/ACE remains
the force-capable default. The self-hashed evidence is
[`route1-mlses-pb-feasibility-probe-2026-07-26.json`](benchmarks/route1-mlses-pb-feasibility-probe-2026-07-26.json).

### Route 1 component-attribution audit

The apparent accuracy gain of the energy-only CHA-GB/PBSA endpoint was not
assigned to either component by inspection. A post hoc \(2\times2\) audit
recombined the already frozen OBC-II and CHA-GB polar terms with the already
frozen ACE and PBSA cavity/dispersion terms on all 526 development geometries.
It read labels only after every component energy was fixed, fit no parameter,
and changed no provider:

| frozen endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| OBC-II/ACE | 1.760 | 2.537 | 13.550 |
| OBC-II/PBSA cavity-dispersion | 2.103 | 3.011 | 15.784 |
| CHA-GB/ACE | 1.819 | 2.412 | 10.036 |
| CHA-GB/GBNSR6 surface term | 1.449 | 1.975 | 9.600 |
| CHA-GB/PBSA cavity-dispersion | 1.322 | 1.854 | 9.416 |
| fine-grid APBS/ACE | 1.629 | 2.462 | 14.116 |
| fine-grid APBS/PBSA cavity-dispersion | 2.161 | 3.143 | 16.310 |

All values are kcal/mol. Relative to OBC-II/ACE, the isolated CHA-GB polar
swap changes paired MAE by `-0.059 kcal/mol` with 95% interval
`[-0.208, 0.094]`, and the isolated PBSA nonpolar swap changes it by
`-0.343 kcal/mol` with interval `[-0.467, -0.216]`. Only the joint swap passes
the development gate: `+0.438 kcal/mol`, interval `[0.319, 0.559]`.

A two-factor Shapley allocation of that nonlinear endpoint MAE gain assigns
`0.361/0.077 kcal/mol` to the polar/nonpolar swaps, with a large
`0.840 kcal/mol` marginal-gain interaction. This allocation closes
algebraically but does not prove microscopic causality. The evidence instead
shows that the successful endpoint is a coupled parameterization: neither
isolated component swap is promoted, and a conservative successor must
jointly justify its polar equation, radii, cavity, and dispersion terms.
The protocol and sealed 526-record artifact are
`benchmarks/route1_component_attribution_protocol.json` and
`benchmarks/route1-chagb-component-attribution-2026-07-25.json`.

### Pre-registered 116-case reserve evaluation

The remaining legacy FreeSolv partition was evaluated without relabeling it as
blind confirmation. Its earlier `prepared.json` already exposed the labels, so
the protocol calls it a **label-exposed, held-out-by-computation reserve**.
Before any reserve energy was generated, the following were frozen:

- exactly 116 source MOL2 hashes and stratification fields in a label-free
  source manifest;
- fixed AM1-BCC/OBC-II/ACE and fixed
  AM1-BCC/CHA-GB/PBSA-cavity-dispersion endpoints;
- full 116/116 coverage for both endpoints;
- a minimum paired MAE gain of `0.15 kcal/mol`, a strictly positive lower
  bootstrap bound, and non-worsening RMSE and maximum absolute error for
  retaining the CHA-GB profile;
- no fit, residual, endpoint selection, chemistry selector, or post-score
  tuning, regardless of the result.

The energy phase read only the protocol, the label-free source manifest, and
the pinned MOL2 files. It generated AM1-BCC charges and both endpoint energies
for all 116 cases, then sealed a complete artifact containing no experimental
field. Only after that artifact passed its count and content-hash checks did
the scoring phase read the pinned legacy labels.

| frozen reserve endpoint | MAE | RMSE | maximum absolute error |
|---|---:|---:|---:|
| AM1-BCC/OBC-II/ACE | 1.791 | 2.464 | 9.622 |
| AM1-BCC/CHA-GB/PBSA cavity-dispersion | 1.301 | 1.846 | 6.376 |

All values are kcal/mol. The paired MAE gain is `0.490 kcal/mol`, with a 95%
bootstrap interval of `[0.215, 0.782]`; 77 cases improve and 39 worsen. Every
pre-registered overall gate passes. Relative to the 526-case development
results, reserve MAE changes by only `+0.031 kcal/mol` for OBC-II/ACE and
`-0.021 kcal/mol` for CHA-GB/PBSA. The largest class-level gain is for
halogen-containing molecules (`0.974 kcal/mol`); hydrocarbons are essentially
tied in MAE (`0.018 kcal/mol` gain), and the CHA endpoint is slightly worse
there in RMSE and maximum error. Thus the result supports the coupled profile
overall but does not imply universal per-class or per-molecule improvement.

The pre-registered decision is therefore:

- retain AM1-BCC/CHA-GB/PBSA cavity-dispersion as an explicit **SP-only
  fixed-geometry accuracy profile**;
- keep AM1-BCC/OBC-II/ACE as the force-capable SP/OPT/SCAN product baseline
  and default;
- infer no CHA-GB force, OPT, SCAN, sampling, or MLIP-accuracy claim from this
  fixed-geometry score;
- keep the independent label-sealed/external confirmation gate open.

The frozen protocol, source manifest, label-free energy artifact, and scored
artifact are `benchmarks/route1_freesolv_reserve_protocol.json`,
`benchmarks/route1_freesolv_reserve_source_manifest.json`,
`benchmarks/route1-freesolv-reserve-energy-2026-07-25.json`, and
`benchmarks/route1-freesolv-reserve-score-2026-07-25.json`.

### Two-level OPT -> final-SP diagnostic

The remaining compliant composition was tested explicitly rather than inferred:

\[
\Delta E_{\mathrm{2L}}=
\min_i\left[
E_{\mathrm{MLIP,gas}}(R^{\mathrm{low}}_{s,i})
+G_{\mathrm{high}}(R^{\mathrm{low}}_{s,i})
\right]
-\min_j E_{\mathrm{MLIP,gas}}(R^{\mathrm{low}}_{g,j}).
\]

Here the low potential is MACE-OFF23m plus fixed-AM1-BCC/OBC-II/ACE, which has
consistent forces, and the final-SP solvent endpoint is
fixed-AM1-BCC/CHA-GB/PBSA cavity-dispersion, which is energy-only. Every unique
converged low-level solution minimum in the pre-existing six-case development
diagnostic was evaluated and reranked with the high-level total potential.
The source selection and records are label-exposed, no parameter was tuned,
and this is not confirmation evidence.

The high-level final SP repairs much of the error of reporting the low-level
relaxed endpoint directly (`2.834 -> 1.901 kcal/mol` MAE), but that is the
wrong product comparison: the frozen high-level fixed-geometry endpoint is
already `1.773 kcal/mol`. The two-level result worsens it to
`1.901 kcal/mol`; all six absolute errors increase. High-level reranking changes
one of six selected minima and makes the aggregate result slightly worse than
retaining the low-level-selected minimum (`1.871 -> 1.901 kcal/mol`).

This proves the interface pattern, not an accuracy gain. It remains an
explicit benchmark-only workflow with both potentials and the missing
final-SP force disclosed. It is not the Route 1 default and is not eligible
for a relaxed SCAN claim. The frozen evidence is
`benchmarks/route1-two-level-final-sp-rerank-2026-07-24.json` and
`benchmarks/route1-two-level-final-sp-rerank-summary-2026-07-24.json`.

## Performance claims

Route 1 has two separate performance questions:

1. **Solvent overhead**:
   \(t_{\mathrm{MLIP+solv}}/t_{\mathrm{MLIP}}-1\).
2. **Named MM comparison**:
   \(t_{\mathrm{MLIP+solv}}/t_{\mathrm{named\ MM+solv}}\).

Neither ratio may be generalized beyond the recorded model, device, OpenMM
platform, task, molecule bin, and warm/cold policy. A raw ratio across a GPU
MLIP and CPU MM backend is recorded only as an ineligible observation.

Gas-conformer batching is a third, separate ratio:
\(t_{\mathrm{serial\ MLIP}}/t_{\mathrm{batched\ same\ MLIP}}\). On the frozen
1,270-state CUDA workload the four-repeat medians are `0.99x` for MACE-OFF23m,
`1.42x` for AIMNet2, and `12.39x` for ANI2x. This admits the numerically
equivalent common interface, but the every-repeat `1.25x` promotion gate fails
for MACE; AIMNet2 and ANI2x pass all four paired repeats. It is not a
comparison with MM and does not change the solvent functional.

The first locally traceable 23-atom warm energy+force run uses MACE-OFF23 medium
on an RTX 4060 Laptop GPU and OpenMM 8.5.2 Reference:

- OBC-II/ACE raises the interleaved paired median from `69.375` to `70.955 ms`,
  a `2.28%` observed combined-minus-gas fraction; the correction-only median is
  `2.12%` of the gas median.
- The MM ratio is not a same-host local MM observation because the MLIP and MM
  calls use different execution backends.

A separate same-host, single-thread CPU/Reference run raises the MACE median
from `323.156` to `324.624 ms`, a `0.45%` observed combined-minus-gas
fraction; the correction-only median is `0.48%` of the gas median. Its named
GAFF2/OBC-II median is `0.0371 ms`, so MLIP+GB is about `8749x` slower for
this small molecule under the recorded resource policy.

The same protocol with ANI2x measures `0.27%` paired and `4.93%`
correction-only on the GPU/Reference combination; its mixed-backend MM ratio
is not a local comparison. Under the same-host, one-thread CPU/Reference
policy, ANI2x measures `14.96%` paired and `10.41%` correction-only, while the
combined potential is about `444x` slower than the named GAFF2/OBC-II
baseline.

A separate 12-molecule OpenMM platform audit compares the product CPU
single-thread/deterministic path with Reference across all five Amber GB
models. The 60 polar slots retain the same 58 supported and two
expected-unavailable sulfur/GBn2 outcomes; the LCPO paths retain 44 supported
and 16 expected-unavailable outcomes. Across every supported polar and LCPO
calculation, the maximum CPU-versus-Reference energy/component difference is
`9.602e-6 kcal/mol` and the maximum force difference is
`1.235e-5 kcal/mol/A`. Two independently constructed CPU contexts reproduce
energy, components, and force exactly. For warm OBC-II/ACE energy+force calls
on the same 12 molecules, CPU is locally `2.40x` to `3.03x` faster than
Reference, with median speedup `2.79x`. This promotes a solvent-backend
default only: it changes neither chemical accuracy nor the Route 1 formula,
and it is not a claim that MLIP+solvent is faster than bare MM. The sealed
evidence is
`benchmarks/route1-openmm-platform-audit-2026-07-25.json`.

These 50-sample local observations have no uncertainty interval. They show
that solvent overhead is gas-model dependent rather than a universal small
constant. They reject both a global “faster than MM” claim and a universal
“negligible solvent overhead” claim: Route 1 is a quality/integration route
for MLIPs, not a way to outrun classical MM. OpenMM Reference is deliberately
fixed here as a correctness-oriented local backend; neither CPU ratio is a
production-MM or general throughput comparison. The four local traces are
`benchmarks/route1-performance-methyl-hexanoate-2026-07-24.json`,
`benchmarks/route1-performance-methyl-hexanoate-cpu-2026-07-24.json`,
`benchmarks/route1-performance-methyl-hexanoate-ani2x-2026-07-24.json`, and
`benchmarks/route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json`.

The default ACE path obtains total energy, total force, and the nonpolar
component from one OpenMM state evaluation. It multiplies only the upstream
ACE energy term by a unit global parameter and reads its exact energy
derivative; the polar component is total minus that derivative. On the water
unit case, all five Amber GB models reproduce an independent polar-only context
within `2e-10 Hartree` on the product CPU backend,
and the frozen two-model all-`3N` force checks remain below their existing
`1e-4 Hartree/A` gate. LCPO retains its two-context decomposition because it is
a separate OpenMM force.

## Literature-guided provider roadmap

- OpenMM `CustomGBForce` and its Amber HCT/OBC/GBn/GBn2 implementations remain
  the maintained force-consistent product runtime inside their audited
  applicability domains. GBn2 fails closed for P and S; the other four models
  remain available for those elements.
- AmberTools GBNSR6/CHA-GB and PBSA cavity/dispersion remain accuracy/reference
  candidates until a maintained force-consistent runtime path exists. The
  Amber manual states that GBNSR6 cannot yet be used in dynamics. A pinned
  source/runtime audit additionally shows an energy-only CHA-GB equation path,
  an inactive cavity derivative, and a numerical/analytical force mismatch.
  The AR6 source fragments can write future `igb9` topology fields, but their
  own developer note says the functionality is not used and current `msander`
  exposes neither an `igb=9` branch nor the AR6 topology fields. AR6 is
  therefore not a maintained runtime candidate for Route 1.
- APBS remains the PB reference/SP backend, not an accuracy-selected endpoint.
  Replacing its APOLAR term with ACE improves the development MAE but fails the
  predeclared materiality gate. The original `97^3` grid also fails stability;
  the converged `129^3` endpoint still gains only `0.131 kcal/mol` MAE and
  therefore remains rejected. The separate SPL4 polar-force audit fails
  all-component finite-difference, cross-grid force, and net-force gates, while
  APBS APOLAR fails its own energy-derivative gate. A different PB endpoint
  must establish a calibrated radius profile, converged energies, and a
  conservative complete polar-plus-nonpolar gradient.
- ALPB, GBn2, and LCPO remain explicit alternatives or research controls; the
  full-corpus analytical screen does not support replacing OBC-II/ACE with any
  of them as the default. The provider-parity layer preserves Amber's generic
  small-molecule ester-oxygen `mbondi3` radius and uses submitted GAFF `o`/`o2`
  types to disambiguate LCPO nitro versus carboxylate oxygen; these audited
  translations do not expand the underlying provider's element/bond-count
  domain.
- GBr6 remains a rejected development control. Its full 526-case energy screen
  is materially worse than both frozen comparators, and the released 2008
  Fortran program does not expose the first-derivative interface advertised by
  the project website.
- ddX/pyddx remains an external reference candidate, not a MAPLE dependency.
  Its complete polar derivative and zero-salt ddPCM numerical convergence
  pass, but the frozen mbondi2/ACE pairing gives `1.782/2.881 kcal/mol`
  MAE/RMSE and is roughly `360x` slower than the current local OBC-II/ACE
  correction. A revisit requires a materially different, independently
  justified charge/radius/nonpolar profile or performance regime; the failed
  pairing must not be relabeled as a product improvement.
- GBMV2/SA is retained as the highest-priority physical provider watch item.
  Historical GAFF/AM1-BCC trajectory/BAR studies report about
  `1.14-1.24 kcal/mol` AUE and the analytical model exposes first derivatives.
  The audited implementation is in registered CHARMM/pyCHARMM or a
  paper-reported CHARMM/OpenMM plugin, not public OpenMM core, and no local
  runtime exists for parity, all-\(3N\) force, license, or small-molecule
  throughput validation. Route 1 will not reimplement it from the papers.
- GBSW is the secondary physical watch item. It has complete smoothed
  solvation forces and historical GAFF/AM1-BCC AUE `1.20-1.33 kcal/mol`, but
  the runnable plugin is part of a registered CHARMM build, not public OpenMM,
  and its documented speed is about four times slower than vacuum (`2--3x`
  faster only than GBMV). FACTS ranks below GBSW because its corresponding AUE
  is `1.25-1.42 kcal/mol` and its official protein-derived radius parameters
  require `TAVW` interpolation for unknown small-molecule radii. Neither is
  integrated or reconstructed from the papers.
- SLIC/CDC is retained separately as the strongest energy-only physical
  accuracy watch item. Its 38-parameter AM1-BCC-compatible model gives
  promising historical errors, but the public 2022 release has no complete
  executable implementation or atom-resolved conservative polar-plus-CDC
  force. Independent SI-table transcription gives `0.813/1.152 kcal/mol`
  MAE/RMSE on 494 rows and `0.826/1.190` after excluding the 63 listed training
  names; these are not a prospective MAPLE reserve and expose unresolved
  count/summary discrepancies. PBJ is related SLIC software, not a drop-in
  implementation of this model. Route 1 will not reconstruct SLIC/CDC from the
  paper.
- The component-attribution result rules out a shortcut in which only the
  CHA-GB energy expression or only PBSA cavity/dispersion is transplanted into
  the OpenMM baseline. The published CHA-GB equation is analytical, but its
  successful endpoint also uses R6 surface Born radii, CHA-specific intrinsic
  radii, and a matching nonpolar parameterization. OpenMM `CustomGBForce`
  supplies automatic derivatives only for the expression actually encoded; it
  does not turn an OBC-radii/CHA hybrid into the audited GBNSR6 endpoint.
- dSASA is a relevant exact, differentiable SASA implementation and is
  integrated into Amber, but it supplies a surface-area nonpolar term rather
  than PBSA cavity plus dispersion and does not repair the missing CHA-GB polar
  derivative. No standalone, redistributable small-molecule provider API was
  identified in this audit, so Route 1 does not add it as a dependency.
- AmberTorchPB is a current accelerator-aware PB linear-system backend, not a
  ready Route 1 molecular provider. At audited commit
  `a92c90b9e57726a9816de105892dd5b2ff2aae9c`, its public wrapper accepts a
  preassembled sparse system and returns the solution; the 13-file repository
  exposes no atom-coordinate/radius/charge frontend, coordinate-force API, or
  license file. It is therefore retained as a future upstream watch item, not
  integrated or benchmarked as if it already supplied a conservative
  molecular potential.
- IWM-GB is the strongest recent small-molecule endpoint identified in this
  audit. It reports `0.87-0.95 kcal/mol` test RMSE on 85 rigid neutral H/C/N/O
  molecules, but its parameters were optimized against experimental hydration
  labels on an 88-molecule training set. The published construction also
  depends on GBNSR6 plus a NanoShaper surface mesh, and the audit found no
  released conservative-force API. It is therefore a research endpoint, not
  independent confirmation evidence or an OPT/SCAN provider.
- AGBNP physics remains relevant for first-shell, dispersion, cavity, and
  solvent-excluding-volume effects. The reviewed OpenMM plugin implements
  AGBNP1 while declaring AGBNP2 “in progress” and was last tested upstream with
  OpenMM 7.2.2. A local Reference build against OpenMM 8.5.2 returned a finite
  two-particle energy, proving only ABI feasibility. AGBNP3 additionally
  requires radius, gamma, alpha, hydrogen-bond, and connectivity typing, while
  its example path is tied to Desmond DMS/OPLS parameters. Route 1 does not
  invent a replacement typing layer, so none is adopted as a product
  dependency.

The frozen dependency audit is
`benchmarks/route1-provider-feasibility-2026-07-24.json`.

Primary references:

- Mongan et al., GBn neck correction, *J. Chem. Theory Comput.* 2007,
  DOI `10.1021/ct600085e`.
- Forouzesh, Izadi, and Onufriev, GBNSR6, *J. Chem. Inf. Model.* 2017,
  DOI `10.1021/acs.jcim.7b00192`.
- Mukhopadhyay et al., CHA-GB, *J. Chem. Theory Comput.* 2014,
  DOI `10.1021/ct4010917`.
- Gallicchio, Paris, and Levy, AGBNP2, *J. Chem. Theory Comput.* 2009,
  DOI `10.1021/ct900234u`.
- Tolokh et al., IWM-GB, *J. Phys. Chem. B* 2024,
  DOI `10.1021/acs.jpcb.4c00254`.
- Tjong and Zhou, GBr6, *J. Phys. Chem. B* 2007,
  DOI `10.1021/jp066284c`.
- ddX theory and API documentation:
  <https://ddsolvation.github.io/ddX/md_docs_theory.html>.
- Stamm et al., domain-decomposition LPB model:
  <https://arxiv.org/abs/1807.05384>.
- Herbst et al., analytical ddLPB forces:
  <https://arxiv.org/abs/2203.00552>.
- OpenMM AGBNP plugin:
  <https://github.com/Gallicchio-Lab/openmm_agbnp_plugin>.
- Amber25 Reference Manual:
  <https://ambermd.org/doc12/Amber25.pdf>.
- OpenMM `CustomGBForce` documentation:
  <https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomGBForce.html>.
- Cao et al., differentiable SASA, *J. Chem. Theory Comput.* 2024,
  DOI `10.1021/acs.jctc.3c01366`.
- Wu et al., AmberTorchPB, *J. Chem. Theory Comput.* 2026,
  DOI `10.1021/acs.jctc.6c00085`; source:
  <https://github.com/yxwu21/AmberTorchPB>.
- OpenMM-ML adapter documentation: <https://github.com/openmm/openmm-ml>.
