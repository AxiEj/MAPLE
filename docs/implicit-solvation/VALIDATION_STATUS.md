# Route 1 implicit-solvation validation status

The Route 1 product surface in this branch is the fixed-charge PB/GB route.
Explicit QEq/CQEq research controls coexist for diagnostic comparison but are
not Route 1 product profiles. All providers still require `experimental=true`
until the scientific gates below are closed.
AM1-BCC/OBC-II/ACE is the forward development default; ABCG2 remains an
explicit alternative and is never selected automatically. This user-selected
development policy is not a scientific certification, and the historical
ABCG2 benchmark artifacts below remain immutable evidence.

## Passing engineering gates

- Tripos MOL2 topology, atom order, atom types, substructure metadata, and fixed
  charges are preserved; disconnected structures and charge-sum mismatches fail
  closed. A per-atom identity array also rejects deletions, substitutions, and
  same-element reorderings after charges, radii, and topology have been frozen.
- MOL2 fixed charges, AmberTools AM1-BCC, and AmberTools ABCG2 are implemented
  with topology, mapping, command, output, and charge-residual audits.
- QEq-GTO and variational CQEq-GTO/GB pass their numerical and force-consistency
  tests, but remain explicit-only experimental controls and are never defaults or fallbacks.
- OpenMM HCT, OBC-I, OBC-II, GBn, and GBn2 execute with finite energies and
  forces. ACE, LCPO, and diagnostic polar-only paths are separated.
- Radius assignment and nonpolar construction are first-class, audited provider
  objects. The OpenMM path consumes their actual standard parameters/force
  configuration, and APBS consumes their actual PQR radii/APOLAR block.
- Registered gas backends now pass an explicit composition-capability gate
  before construction. `CalcABC` backends inherit the shared additive
  energy/force path; non-`CalcABC` plugins must declare and implement the
  equivalent capability. Registration without that declaration fails closed.
- The actual SetCalculator composition path passes a two-model compatibility
  trace for MACE-OFF23m and AIMNet2 using the same fixed AM1-BCC/OBC-II/ACE
  correction. Both close energy and force addition, pass all 69 Cartesian
  finite differences of the combined potential at the predeclared `0.003 A`
  step, and lower the combined potential in a two-step BFGS smoke. Three
  manually displaced geometries return finite combined energies but are not
  labeled as a MAPLE SCAN-task test. The identical fixed-geometry solvent term
  confirms that the gas-model adapter does not alter the
  charge/radius/nonpolar provider result.
- A separate engine-level task matrix runs the actual MAPLE dispatcher for
  MACE-OFF23m, AIMNet2, and ANI2x. All three complete SP, a two-iteration MAPLE
  LBFGS OPT, a three-point rigid MAPLE SCAN, and a deterministic four-step NVT
  trajectory with the same normalized fixed-AM1-BCC MOL2 and OBC-II/ACE
  correction. All twelve jobs write structured solvation metadata and an audit
  manifest; all three OPT energies decrease, all three SCAN trajectories
  contain the requested three points, all three MD trajectories contain four
  finite thermodynamic records, and all additive energy closures are below
  `3e-14 Hartree`. This is named-adapter task-plumbing evidence, not broad
  OPT/SCAN/MD stability, equilibrated sampling, or chemical-accuracy
  certification.
- Three real MAPLE fixed-charge OpenMM OBC-II/ACE water FREQ jobs exercise
  MACE-OFF23m, AIMNet2, and ANI2x with explicit `hessian=numerical` through the
  normal input reader, calculator factory, implicit provider, and frequency
  dispatcher. Every resulting `9 x 9` composed Hessian is finite and symmetric,
  and differs nontrivially from that model's gas-only numerical Hessian at the
  same geometry. This proves across three independently registered adapters
  that the common numerical path differentiates the complete reported
  MLIP-plus-GB force rather than returning a gas Hessian beside a solvent-added
  energy. Analytic implicit Hessians, implicit HVP, and energy-only-provider
  FREQ remain fail-closed. The output warns that the existing
  translational/rotational RRHO terms use ideal-gas pressure and are neither a
  solution-standard-state Gibbs energy nor an absolute solvation free energy.
  Independent formula regressions additionally lock the Grimme finite-inertia
  entropy interpolation and the Otlyotov--Minenkov complete vibrational
  internal-energy interpolation, including damped ZPE rather than an
  additional undamped ZPE. These are implementation-parity checks against the
  published formulas, not hydration-accuracy evidence.
- The experimental `inner=prebuilt` extension accepts a neutral
  multicomponent MOL2 only with fixed per-atom MOL2 charges. It creates one
  synthetic OpenMM residue per connected component and records component
  counts and charge sums. A separate engine-level smoke runs MACE-OFF23m,
  AIMNet2, and ANI2x through SP, two-iteration OPT, and three-point rigid SCAN
  on one methanol-plus-water cluster. All nine jobs pass finite energy,
  additive closure, structured-provenance, and task-output checks. A centered
  `0.003 A` finite difference of one combined-potential force component per
  MLIP passes the frozen `5e-5 Hartree/A` tolerance, and the fixed-geometry
  outer correction is identical across all three MLIPs. Output and manifests
  call the result a fixed-shell cluster-continuum
  configurational potential and explicitly reject an absolute solvation
  free-energy claim. This is engineering evidence only: cluster
  formation/occupancy, standard-state, solvent-cluster reference, and ensemble
  terms remain absent.
- A separate automatic explicit-inner/implicit-outer source audit closes the
  current product-admission question. CREST QCG is a complete, maintained
  supermolecular scaffold: it generates solute-solvent and reference-solvent
  ensembles and includes conformational and thermochemical terms. Its current
  QCG source is nevertheless wired directly to xTB/GFN-family single points,
  optimizations, sampling levels, and Hessians. CREST's generic backend is not
  wired into QCG, so the upstream does not establish an arbitrary-MLIP
  free-energy path. FEBISS is an optional proposal preprocessor that ranks
  solvent sites from a supplied explicit-solvent trajectory; it is not an
  absolute-solvation-free-energy engine. No neutral aqueous FreeSolv-scale
  accuracy or end-to-end multi-MLIP speed evidence closes the remaining
  compatibility, overlap, cluster-size, and standard-state gates. Consequently
  no automatic runtime or FreeSolv selector was added, and `inner=prebuilt`
  retains its fixed-shell-potential-only classification.
- The expanded Amber/OpenMM corpus covers 12 neutral molecules, all five GB
  models, and 60 molecule/model slots across C/H/O/N/F/S/Cl/Br/I. There are
  58 supported polar records and two sulfur/GBn2 slots fail closed as
  predeclared applicability observations. Across the supported records, the
  maximum polar differences are 0.001350 kcal/mol for energy and
  0.001009 kcal/mol/A for force. The 39 complete LCPO records agree to
  numerical precision; seven cases cover all five models and dimethyl sulfide
  covers its four supported models. The expansion exposed and fixed two
  general provider-translation errors: neutral ester carbonyl oxygen now keeps
  Amber's generic-MOL `mbondi3` radius, and LCPO uses exact GAFF `o`/`o2`
  typing for nitro versus carboxylate oxygen. Amber applies carbon
  surface-area parameters to Cl/Br/I, whereas OpenMM uses a chlorine-specific
  term (a 0.161904 kcal/mol chlorobenzene nonpolar difference) and fails
  closed for Br/I. Nitralin also remains polar-only because OpenMM has no LCPO
  parameter for tetravalent sulfone sulfur. Those four cases are therefore
  polar-only parity targets with their LCPO boundaries retained in the audit.
  OpenMM's generic GBn2 expression also does not reproduce Amber's signed
  sulfur near-pair descreening branch, so dimethyl sulfide and nitralin/GBn2
  fail closed rather than enter a numerical tolerance.
- A separate platform audit promotes single-thread deterministic OpenMM CPU as
  the product runtime default while retaining Reference as the explicit
  correctness/parity control. Across the same 12 molecules and five GB models,
  CPU and Reference retain identical support outcomes: 58/60 polar slots
  succeed, two sulfur/GBn2 slots are expected-unavailable, 44 LCPO paths
  succeed, and 16 are expected-unavailable. Supported energy/component
  differences are at most `9.602e-6 kcal/mol`; force differences are at most
  `1.235e-5 kcal/mol/A`; independent CPU contexts repeat all three exactly.
  Warm OBC-II/ACE energy+force calls are locally `2.40x` to `3.03x` faster
  than Reference across the 12 cases (median `2.79x`). This changes only the
  solvent execution backend: it provides no chemical-accuracy gain and no
  faster-than-MM claim.
- APBS LPB input generation, solvent/reference subtraction, APOLAR composition,
  audit persistence, provider failures, official Born-ion reproduction, and
  methanol/aniline grid sweeps are under regression tests.
- The ten-molecule FreeSolv pilot is frozen. AM1-BCC/OBC-II/ACE gives MAE
  0.680 kcal/mol; this is a pilot, not scientific certification.
- The full-corpus harness pins FreeSolv v0.52 and freezes development and
  reserved membership without opening a certification claim. The legacy
  preparation manifest contains experimental labels for both partitions.
  The 116-record reserve has now been computed behind a separate label-free
  energy boundary, but remains label-exposed held-out-by-computation evidence,
  not independent or label-sealed confirmation.
- The complete 526-molecule development matrix is frozen: 5,238/5,260 attempts
  succeeded. The only 22 failures are the predeclared GBn2 phosphorus boundary.
  ABCG2/OBC-II has the lowest development MAE at 1.652 kcal/mol; all strata,
  failures, and per-attempt hashes are retained in the development summary.
- A separate label-blind 526-molecule AM1-BCC comparison completed both
  requested physical model changes. CHA-GB/GBNSR6 plus its simple surface term
  reduces MAE/RMSE from 1.760/2.537 to 1.449/1.975 kcal/mol. Replacing that
  surface term with PBSA cavity plus dispersion further reduces them to
  1.322/1.854 kcal/mol. The paired MAE gain over OBC-II/ACE is
  0.438 kcal/mol with a 95% bootstrap interval of [0.321, 0.559].
  Energy records contain no experimental labels, and an independent full
  rerun reproduced all 526 record hashes and the complete summary bit for bit.
  This is a development endpoint comparison, not a new runtime default or
  confirmation result.
- A Route 1 component-attribution audit then crossed the frozen OBC-II and
  CHA-GB polar energies with the frozen ACE and PBSA cavity/dispersion
  energies. The isolated PBSA nonpolar swap worsens OBC-II/ACE to
  `2.103/3.011 kcal/mol` MAE/RMSE, while the isolated CHA-GB polar swap gives
  `1.819/2.412`; only the coupled CHA-GB/PBSA endpoint reaches
  `1.322/1.854`. Its paired MAE gain is `0.438 kcal/mol`, whereas the two
  isolated gains are `-0.343` and `-0.059 kcal/mol`. A descriptive endpoint
  Shapley allocation is `0.361/0.077 kcal/mol` for the polar/nonpolar swaps,
  but the large interaction means it does not prove microscopic causality.
  Therefore neither isolated component swap is promoted; the next
  force-consistent candidate must preserve a jointly justified
  polar/radius/nonpolar parameterization. The development labels were used
  only for this post hoc score, not for fitting, and confirmation remains
  closed. Evidence is frozen in
  `route1-chagb-component-attribution-2026-07-25.json`.
- A pre-registered reserve protocol then fixed the two endpoints, all 116
  source MOL2 hashes, full-coverage requirement, `0.15 kcal/mol` materiality
  threshold, positive paired-bootstrap lower bound, non-worsening RMSE/max
  requirements, and a no-fit/no-selection/no-post-score-tuning rule before
  generating any reserve energy. Both endpoints completed 116/116 cases.
  The sealed energy artifact contains no experimental field; labels were read
  only afterward. OBC-II/ACE gives MAE/RMSE/max
  `1.791/2.464/9.622 kcal/mol`; CHA-GB/PBSA cavity-dispersion gives
  `1.301/1.846/6.376`. Its paired MAE gain is `0.490 kcal/mol`, with 95%
  interval `[0.215, 0.782]`; 77 cases improve and 39 worsen. All
  pre-registered overall gates pass, so the coupled endpoint is retained as
  the explicit SP-only fixed-geometry accuracy profile. OBC-II/ACE remains the
  force-capable default, no derivative capability is inferred, no tuning is
  allowed, and the independent confirmation gate remains open. Evidence is
  frozen in `route1_freesolv_reserve_protocol.json`,
  `route1_freesolv_reserve_source_manifest.json`,
  `route1-freesolv-reserve-energy-2026-07-25.json`, and
  `route1-freesolv-reserve-score-2026-07-25.json`.
- The exact CHA-GB/cavity-dispersion endpoint is now exposed as an explicit
  AM1-BCC SP-only runtime provider. A 526-case topology audit found that
  Antechamber retyping changes 95 input atom-type vectors, so the runtime
  preserves submitted GAFF/GAFF2 atom types and fails if `parmchk2` requires a
  `NONBON` override. A live provider run reproduces frozen `EGB`, `ECAVITY`,
  `EDISPER`, and total energy within `6e-13 kcal/mol`. Audit records pin all
  four AmberTools executable hashes and state that gas and bonded MM energies
  are unused. Force requests, OPT, SCAN, MD, `inner=prebuilt`, non-AM1-BCC
  charges, and parameter mixing fail closed.
- A separate source/runtime derivative audit pins AmberClassic commit
  `0b35bfeb96026ffa4e5876391a0828f39b3cfc8d`. The force driver passes `f` into
  `egb`, but the inspected GB/CHA-GB equations accumulate energy without a
  force-array argument; the active cavity path has no derivative call, and only
  part of the dispersion path writes forces. AmberTools 26 `debugf` reports
  numerical components `(0.05786533, -0.55185254, 0.30099780)` beside
  analytical `(0, 0, 0)` for the audited atom. The same source tree labels its
  AR6 topology writer as unused future `igb9` work, while current `msander`
  contains neither an `igb=9` branch nor AR6 topology-field consumers. This
  closes AR6 as an available product candidate and independently supports the
  CHA-GB SP-only fail-closed policy; it does not evaluate accuracy.
- A separate frozen PB screen combined the APBS molecular-surface LPBE polar
  term with the existing OpenMM mbondi2/OBC-II ACE nonpolar component. Its
  label-free energy phase completed all 526 development molecules with fixed
  AM1-BCC charges and no gas-phase MM energy, fit, residual, or retraining.
  Replacing APBS APOLAR with ACE improves MAE/RMSE from `4.415/4.984` to
  `1.658/2.535 kcal/mol`; relative to OBC-II/ACE the paired MAE gain is only
  `0.102 kcal/mol`, below the frozen `0.15` materiality gate. The preselected
  20-case fine-grid check changes polar energies by up to `0.671 kcal/mol`
  (P90 `0.455`), failing the `0.25/0.15` stability thresholds. The candidate
  is rejected and no cross-provider PB/ACE runtime path was added.
- A label-free follow-up confirmed that `129^3 @ 0.25 A` and
  `161^3 @ 0.20 A` agree on the same 20 cases within maximum/P90
  `0.175/0.148 kcal/mol`. The resulting full 526-case `129^3` PB/ACE endpoint
  gives MAE/RMSE `1.629/2.462` and a paired MAE gain of
  `0.131 kcal/mol` over OBC-II/ACE (95% interval `[0.050, 0.217]`). It passes
  the fine-grid numerical gate but still misses the frozen `0.15` materiality
  gate and remains worse than CHA-GB/cavity-dispersion. It is also rejected;
  no runtime or performance claim is opened.
- The remaining APBS force path was then closed with two source-pinned,
  all-component probes on methyl hexanoate. APBS 3.4.1 aborts
  `srfm=mol` force requests, so the polar probe used `srfm=spl4`. Even at
  `161^3 @ 0.20 A`, its force-to-energy finite-difference RMSE/maximum are
  `0.199/1.103 kJ/mol/A`; coarse-to-fine force RMSE/maximum are
  `0.483/2.579 kJ/mol/A`, and the fine-grid net-force norm is
  `2.336 kJ/mol/A`. The locked APBS `gamma*SASA` total force independently
  gives matching-step RMSE/maximum `1.114/4.871 kJ/mol/A` and net-force norm
  `3.694 kJ/mol/A`. Generic mbondi2 radii are also not a validated
  spline-surface parameterization. Both numerical gates fail, no SPL4 accuracy
  screen is opened, and APBS remains SP-only.
- An AmberTools `sander` exact-difference probe tested whether the accurate
  PBSA `inp=2` endpoint could retain only
  `E_MM,PB(inp=2)-E_MM,vacuum` internally and then be added to an arbitrary gas
  MLIP. The MM gas energy cancels exactly, but the corresponding force
  difference is not the centered derivative of that energy difference. At
  `0.003 A`, four sampled Cartesian errors span 0.0750 to
  3.4200 kcal/mol/A; at `0.01 A`, the maximum remains
  1.8108 kcal/mol/A. All stored bonded and direct/1-4 nonbonded MM components
  cancel within `5.4e-15 kcal/mol`, confirming that the force failure is not an
  uncancelled gas-MM term. The candidate is therefore SP/reference-only and is
  not eligible for OPT, relaxed SCAN, or a product default.
- The remaining maintained analytical candidates were screened without fitting
  any parameter or residual. In the neutral domain, ALPB scales the OBC-II
  polar energy and force by the literature constant `0.9927734691`; MAE changes
  only from 1.760351 to 1.758003 kcal/mol, a paired gain of
  0.002348 kcal/mol with bootstrap interval `[-0.002740, 0.007582]`, while
  335/526 cases worsen. On the identical 515-case success subset, GBn2 has MAE
  1.893983 versus OBC-II's 1.661487 kcal/mol and retains the 11-case phosphorus
  failure boundary. That frozen screen predates the later independent
  sulfur-parity finding; the current runtime additionally fails closed for
  S-containing GBn2 rather than reusing the inaccurate generic branch. OpenMM
  LCPO can assign parameters to only 454/526 molecules. On those same 454
  supported cases, ACE and LCPO share an exactly matching OBC-II polar term,
  but LCPO worsens MAE from `1.801` to `2.250 kcal/mol`; the paired
  ACE-minus-LCPO gain is `-0.449 kcal/mol` with 95% interval
  `[-0.581, -0.318]`, and 285 cases worsen. These results reject all three as
  a replacement default; they do not justify a label-selected fallback.
- A source/dependency audit also rejected premature integration of IWM-GB and
  AGBNP. IWM-GB reports `0.87-0.95 kcal/mol` test RMSE on 85 rigid neutral
  H/C/N/O molecules, but uses experimental hydration labels for parameter
  optimization and has no audited released conservative-force API. The
  reviewed OpenMM AGBNP plugin implements AGBNP1, not complete AGBNP2. It
  builds against the local OpenMM 8.5.2 Reference library and returns a finite
  two-particle energy, but lacks a maintained generic small-molecule typing
  provider and has not passed Route 1 force, coverage, or accuracy gates.
  AGBNP3 would require an additional hydrogen-bond/connectivity typing layer.
  No runtime provider was added; the exact commits, binary/input hashes, and
  decisions are frozen in
  `route1-provider-feasibility-2026-07-24.json`.
- A primary-source GBMV2 audit identifies a scientifically promising physical
  path rather than another residual model. Published GAFF/AM1-BCC work used
  vacuum and GBMV2 trajectories plus BAR and reported `1.14 kcal/mol` AUE on
  499 neutral molecules; a later coefficient-selection/test separation
  reported `1.24 kcal/mol` AUE on 375 non-CGENFF compounds. GBMV2 analytical
  method II and its SASA term expose first derivatives. These historical
  results are not current-profile FreeSolv certification because the nonpolar
  coefficient used experimental labels and the protocol differs.
  The audited implementation path is registered CHARMM/pyCHARMM or a
  paper-reported CHARMM/OpenMM plugin. Public OpenMM contains no GBMV2 class
  and no local runtime was available, so provider parity, full force checks,
  redistribution, and small-molecule speed remain untested. No GBMV2 runtime
  provider or reimplementation was added; the decision is frozen in
  `route1-gbmv2-feasibility-audit-2026-07-25.json`.
- A matched FACTS/GBSW primary-source audit finds two more physical,
  AM1-BCC-compatible constructions but no deployable improvement. The 499-case
  optimized-profile AUE/RMSE values are `1.20/1.52 kcal/mol` for GBSW and
  `1.25/1.80 kcal/mol` for FACTS; the experimental-label-selected nonpolar
  coefficients prevent independent certification. On 375 compounds separated
  from coefficient selection, AUE is `1.33` for GBSW and `1.42 kcal/mol` for
  FACTS, both worse than GBMV2's `1.24`. GBSW supplies full solvation forces
  and a smooth boundary, but is about four times slower than vacuum; its
  `2--3x` speedup is only against GBMV. FACTS is also about four times slower
  than vacuum and its protein-derived radius parameters require `TAVW`
  interpolation for unknown small-molecule radii. CHARMM distributes GBSW as
  a plugin with a registered CHARMM build, not public OpenMM core; no local
  FACTS or GBSW runtime is available. No provider or paper-based
  reimplementation was added. The watch order `GBMV2 -> GBSW -> FACTS` and
  reopen gates are frozen in
  `route1-facts-gbsw-feasibility-audit-2026-07-25.json`.
- A separate SLIC/CDC source and supporting-information audit identifies a
  stronger energy-only physical lead without confusing parameter fitting with
  a learned residual. The 2022 model uses the Mobley AM1-BCC structures and
  charges and combines SLIC electrostatics with cavity, atom-typed dispersion,
  combinatorial, and hydrogen-bond terms. It has 38 fitted physical-model
  parameters and uses experimental total hydration and explicit-solvent
  component data; the prose says 65 training compounds, while Table S4 lists
  63. Independent `pdftotext -layout` and `-raw` extraction gives identical
  numerical results for 494 Table S10 rows: MAE/RMSE
  `0.813/1.152 kcal/mol`. Excluding the 63 listed training names leaves 431
  historical rows at `0.826/1.190`, but this is not a prospective MAPLE
  reserve. This source-bound replay verifies table transcription and
  arithmetic; it is not an independent confirmation of model accuracy. The
  recomputed `1.152` RMSE agrees with the preprint's `1.15`, not the SI
  footer's `0.98`; the 494 rows also do not reconcile with the article's
  500-solute or Appendix E's 502-corpus statements. The public journal record
  contains only the SI PDF. PBJ is not the published SLIC/CDC model and lacks
  a complete atom-resolved polar-plus-nonpolar force; the historical MATLAB
  source is an ion/solvent optimization code. No SLIC/CDC provider, dependency,
  paper-derived reimplementation, force claim, speed claim, or FreeSolv
  selector was added. The decision is frozen in
  `route1-slic-cdc-feasibility-audit-2026-07-25.json`.
- A separate source and one-molecule feasibility audit closes the
  learned-solvent ambiguity without weakening Route 1. GNNIS returns a
  solvent-only scalar with conservative autograd forces and the local
  polar/nonpolar energy closes exactly, but upstream trains it with force-only
  MSE against mean explicit-solvent forces minus vacuum OpenFF forces. The
  model learns both GB-Neck2 radius scaling and a SASA contribution.
  QM-GNNIS then explicitly transfers
  `G_GNNIS-G_GB-Neck2` as a delta correction on top of ORCA/CPCM rather than
  exposing a direct fixed-charge MLIP-plus-PB/GB provider. Neither path
  contains a hidden gas-MM energy, yet both violate or fall outside the
  no-learned-correction baseline. The one-coordinate finite-difference smoke
  proves only local scalar-gradient agreement, not a global force or MD gate.
  No runtime provider or FreeSolv selection experiment was opened; the
  commits, checkpoint/license/paper hashes, limited finite-difference smoke,
  and exclusion decision are frozen in
  `route1-learned-solvent-boundary-audit-2026-07-25.json`.
- GBr6 was then tested as the remaining lightweight analytical physical
  provider. A label-free energy phase completed all 526 development molecules
  with fixed AM1-BCC charges, upstream `gbn-bondi` radii, and the frozen PBSA
  cavity/dispersion term. After energies were frozen, development scoring gave
  MAE/RMSE `2.257/3.451 kcal/mol`, worse than OBC-II/ACE
  (`1.760/2.537`) and CHA-GB/cavity-dispersion (`1.322/1.854`). The paired
  MAE gains are negative with 95% intervals wholly below zero. The website
  advertises first derivatives, but the pinned GPL Fortran release exposes
  only an energy program and no force/gradient interface. GBr6 is rejected;
  no runtime provider was added.
- ddX/pyddx was audited next as a maintained, force-capable external PCM
  candidate without adding a MAPLE dependency. The isolated `pyddx 0.8.0`
  build passed `9/9` shipped Python tests. Both tested ddPCM cavity profiles
  pass all 69 methyl-hexanoate polar-force finite differences; the mbondi2
  profile gives primary-step RMSE/maximum
  `0.000404/0.001504 kJ/mol/A`. Its preselected label-free 20-case
  `lmax=7,nLebedev=194` versus `lmax=13,nLebedev=590` comparison also passes
  with P90/maximum differences `0.131/0.182 kcal/mol`.
  The separate label-blind 526-case energy phase then shows why derivative
  correctness alone is insufficient: ddPCM/mbondi2/ACE gives MAE/RMSE/maximum
  error `1.782/2.881/18.813 kcal/mol`, versus
  `1.760/2.537/13.550` for OBC-II/ACE. Its paired MAE gain is
  `-0.022 kcal/mol` with 95% interval `[-0.143, 0.098]`. The expanded-radius
  profile is much worse (`5.829/6.706 kcal/mol` MAE/RMSE), and the local
  ddPCM force call is roughly `360x` slower than the frozen warm OpenMM
  correction. Accuracy, performance, and confirmation gates fail; no runtime
  provider or project dependency was added.
- The frozen 20-case conformer-sensitivity protocol completed all generators
  and evaluated 1,294 CREST conformers. For ABCG2/OBC-II, the fixed-charge
  correction has a 1.791 kcal/mol all-case p90 range and a 4.277 kcal/mol
  flexible-case p90 range, so the single-geometry result is not relabeled as a
  population-averaged hydration free energy.
- A development-only MACE-OFF23m discrete-conformer diagnostic evaluated 1,298
  states in 59.304 seconds on an RTX 4060 Laptop GPU. Across the 20 selected
  cases, MAE/RMSE changed from 1.983/2.317 to 1.799/2.120 kcal/mol, but
  removing the most influential alachlor case reduces the MAE improvement from
  0.184 to 0.059 kcal/mol. This is not yet a converged finite-temperature or
  confirmation result.
- The discrete estimator is now extracted into the MLIP-agnostic
  `analyze_discrete_conformer_ensemble` analysis API. A sealed replay sends 40
  historical MACE-OFF23m records, 20 per charge method and 2,596 states total,
  through the new core and reproduces all 11 common legacy fields exactly.
  Only 14/20 AM1-BCC/OBC-II cases and 14/20 ABCG2/OBC-II cases pass the
  default state-count/effective-count/dominant-weight/overlap diagnostics.
  Historical files contain experimental fields, but the core and replay
  metrics do not read them. This validates extraction and fail-closed weight
  diagnostics only: the named replay is MACE-only, conformer completeness and
  basin measures remain unproved, and no public solvation-free-energy claim is
  opened.
- A prospective multi-MLIP repeat then intersects the element domains exposed
  by the actual MACE-OFF23m, AIMNet2, and ANI2x checkpoints. It retains 19
  cases and 1,270 common states; the sole phosphorus case is rejected before
  evaluation because ANI2x excludes P. Every model evaluates the identical
  state coordinates and fixed AM1-BCC/OBC-II/ACE correction twice. All 57
  model/case records complete, and the new MACE relative energies reproduce
  the historical values within `7.14e-10 kcal/mol`.
- The repeat is diagnostic rather than a pass. The predeclared
  `1e-6 kcal/mol` repeat-relative-energy gate remains failed because float32
  CUDA AIMNet2/ANI2x reaches `7.59e-5 kcal/mol`; the maximum propagated
  discrete-correction difference is `3.93e-6 kcal/mol`. The threshold was not
  changed after the result. Weight diagnostics pass only `13/19` MACE,
  `4/19` AIMNet2, and `12/19` ANI2x cases. The final correction has median/p90
  cross-model ranges `0.03597/0.11969 kcal/mol`, but alachlor reaches
  `1.41020 kcal/mol`, while AIMNet2/ANI2x median gas and solution weight
  overlaps are only `0.2141/0.2013`.
- Separate development scoring was run only after the label-free energy
  artifact was sealed. The common fixed-geometry MAE is `1.930 kcal/mol`;
  AIMNet2, MACE-OFF23m, and ANI2x discrete weighting gives
  `1.910/1.969/1.993 kcal/mol`. Each paired MAE-gain bootstrap interval crosses
  zero, so the result establishes multi-MLIP plumbing and sensitivity, not an
  accuracy gain or model selection.
- The same 19-case/1,270-state set now exercises the common return-value
  `calculate_many` protocol at a fixed CUDA batch size of 16. MACE-OFF23m uses
  disconnected graphs, AIMNet2 uses a same-molecule `mol_idx` mask, and ANI2x
  groups identical element orderings; the base class retains a sequential
  fallback. The largest serial/batch relative-energy and propagated discrete
  correction differences are `2.86e-4` and `3.48e-5 kcal/mol`, both below the
  originally prospective `0.001 kcal/mol` limits. Review of the initial run
  then added an absolute-energy gate at the same tolerance converted to
  Hartree and repaired warmup, provenance, and decision logic before the
  confirmatory v2 rerun; v2 is therefore explicitly review-amended rather than
  wholly pre-data. A preserved v2 run then found MACE paired-repeat speedups of
  `1.060x/1.619x`, even though its ratio-of-medians passed `1.25x`. Protocol v3
  therefore uses four balanced repeats and requires every paired repeat to
  clear the unchanged floor. After a contaminated refresh was discarded,
  protocol v5 added fail-closed GPU endpoint snapshots and a preflight
  one-minute host-load ceiling of `0.25` per logical CPU. Protocol v6 narrows
  the claim: those snapshots do not prove whole-run GPU exclusivity, and the
  artifact records that no continuous monitoring was performed. v6 is
  explicitly review-amended.
  Median speedups are
  `0.99x/1.42x/12.39x` for MACE-OFF23m/AIMNet2/ANI2x. Numerical
  interface admission passes, but the all-repeat `1.25x` material-speedup gate
  fails for MACE (minimum `0.616x`); AIMNet2 and ANI2x pass all repeats
  (minima `1.283x` and `11.298x`). Their named accelerations are admitted,
  never a universal speed, speed versus bare MM, or improved solvent
  accuracy claim.
- The remaining phase-specific conformer-thermochemistry hypothesis was
  recorded as the closed
  `maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v6`
  falsification pilot. No experimental label was read or scored. The
  development-only protocol label-blindly selects six cost-bounded
  cases (221 frozen states), uses three actual checkpoints and exact external
  frozen AM1-BCC/OBC-II/ACE, and optimizes gas and solution phases separately.
  Harmonic local RRHO is primary; qRRHO/free-rotor variants are sensitivity
  diagnostics because explicit minima plus rotor interpolation can double
  count torsional entropy. Every selected branch must converge and resolve as
  a valid unique minimum or explicit duplicate. Any selected negative mode,
  excessive rigid-mode leakage, or failed preregistered Hessian-displacement
  preflight on the rigid anchor plus one flexible case invalidates the
  affected model-case. The sensitivity reference is hash- and
  frequency-anchored to the primary 0.002-A Hessian. A separately frozen
  scorer refuses label access until all 18 self-hashed records pass the same
  strict semantic
  validator used by resume and sealing. Complete records and every primary or
  sensitivity Hessian are copied from `.omx` scratch into a fixed durable
  sibling under the benchmark tree. Its pre-execution
  `0.0002 kcal/mol` repeat ceiling is documented from the prior label-free
  `0.0001037760 kcal/mol` maximum; the runner revalidates that source
  artifact's file/self/content hashes, label-free boundary, and maximum. The
  ceiling was not selected from experimental labels. The first attempted
  model-case, ANI2x/nitromethane, wrote a self-hashed fail-closed record:
  both phase optimizations converged below `0.009 eV/A`, but raw Hessian
  asymmetry at the frozen `0.002 A` step was
  `1.214e-3/1.130e-3 Hartree/A^2`, exceeding the `1e-6` gate by over three
  orders of magnitude. Post-failure label-blind diagnostics show stationary
  symmetrized spectra and maximum selected-mode RMS shifts of
  `1.10/1.06 cm-1` across all three frozen displacements. This diagnoses an
  incompatible universal raw-asymmetry threshold, but it cannot rescue,
  replace, resume, seal, or score v6. The durable failure artifact is
  [`route1-multi-mlip-phase-specific-selected-minimum-rrho-v6-failure-2026-07-25.json`](benchmarks/route1-multi-mlip-phase-specific-selected-minimum-rrho-v6-failure-2026-07-25.json).
  V8 now supplies that separately versioned, label-blind numerical-Hessian
  qualification. Across MACE-OFF23m, AIMNet2, and ANI2x, both phases and
  `0.001/0.002/0.004 A`, the largest raw absolute asymmetry, relative
  Frobenius asymmetry, and selected-mode RMS shift were
  `1.765e-3 Hartree/A^2`, `7.368e-4`, and `1.097 cm-1`. This evidence selected
  a broad `0.02 Hartree/A^2` absolute sanity ceiling and `0.002` relative gate
  without opening a FreeSolv label. It also found that force-converged AIMNet2
  nitromethane was an approximately `-61 cm-1` saddle; deterministic
  plus/minus mode displacement and reoptimization produced valid gas and
  solution minima. The qualification and raw Hessians are durable in
  [`route1-hessian-numerical-qualification-2026-07-25.json`](benchmarks/route1-hessian-numerical-qualification-2026-07-25.json).
  A behavior-preserving evidence-path type guard added after the first v7
  preflight invalidated v7 under its own source-freeze rule; its exact records
  are retained but cannot be resumed, sealed, or scored. The independently
  frozen v8 rerun then passed the real nitromethane preflight for all three
  MLIPs in `41.55 s` wall time with `2.10 GB` maximum RSS. AIMNet2 triggered
  recovery in both phases; MACE-OFF23m and ANI2x did not. All final selected
  frequencies were positive, the largest displacement RMS remained
  `1.107 cm-1`, and all scale-aware Hessian gates passed. This is only the
  rigid numerical/plumbing milestone: 3/18 records exist, no aggregate is
  sealed, and no experimental label or accuracy score has been opened. The
  interim records and raw Hessians are durably mirrored in
  [`route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-rigid-preflight-2026-07-25.json`](benchmarks/route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-rigid-preflight-2026-07-25.json);
  that mirror is not the protocol's required 18-record aggregate seal.
  The subsequent three-MLIP flexible `mobley_4463913` preflight failed closed:
  one selected AIMNet2 gas branch exhausted 500 LBFGS steps with a final
  maximum force of `0.138705 eV/A`. V8 cannot be resumed, repaired, sealed, or
  scored.
- A separately frozen label-blind optimizer qualification then tested
  BFGSLineSearch, LBFGSLineSearch, and FIRE2/ABC from the exact same three
  source states in both phases for MACE-OFF23m, AIMNet2, and ANI2x. V1 had
  bounded optimizer steps but not line-search calculator evaluations and was
  interrupted; v2 reuses no partial record and adds one common 1,000-evaluation
  ceiling. Each candidate passes `12/18` branches. BFGSLineSearch and
  LBFGSLineSearch fail every AIMNet2 branch through evaluation exhaustion or
  another frozen convergence failure. FIRE2/ABC reaches the force threshold
  on five AIMNet2 branches but violates the no-final-energy-increase gate, and
  exhausts the ceiling on the sixth. The result is
  `failed-closed-no-global-policy`: no per-model/per-phase tuning, no v9, no
  Hessian or thermochemistry continuation, no label scoring, and no public
  `#solvfe` claim. This closes the proposed selected-minimum RRHO escalation
  without changing the force-capable SP/OPT/SCAN product.
- A follow-up six-case stratified MACE relaxation diagnostic ran 13 gas and 13
  MACE+ABCG2/OBC-II/ACE minimizations; all 26 branches converged below
  0.03 eV/A. Relative to the first-stage discrete weighting on the same cases,
  MAE/RMSE changed from 1.999/2.360 to 1.874/2.170 kcal/mol. Only 3/6 cases
  improved, and the improvement remains 0.164 kcal/mol with alachlor removed.
  Alachlor shows a real solvent-induced MACE conformer change
  (1.243-A heavy-atom RMSD and 1.106-kcal/mol gas reorganization), while the
  remaining maximum error of 4.007 kcal/mol shows that MLIP relaxation does not
  repair the fixed-charge GB/ACE model itself. This remains a 0 K development
  diagnostic, not a hydration free energy.
- Repeating the frozen conformer tests with the forward-default AM1-BCC pairing
  is negative evidence for a direct sampling escalation. Across the same 20
  cases and 1,298 states, discrete MACE weighting changes
  AM1-BCC/OBC-II/ACE MAE/RMSE from
  2.227/2.829 to 2.291/2.925 kcal/mol. Across the six relaxed cases,
  MAE/RMSE changes from 2.582/3.510 at the endpoint to 2.834/3.760 kcal/mol.
  All 24 optimization branches converged, so the degradation is scientific
  rather than an optimizer failure.
- A force-free, energy-exact Metropolis/TI prototype then combined the MACE
  gas potential with frozen AM1-BCC/CHA-GB/cavity-dispersion energies at five
  lambda windows. Rigid, limited-flexibility, and flexible development cases
  all passed the predeclared acceptance and two-chain agreement gates, proving
  that an energy-only solvent provider can participate in conformer-aware
  evaluation without MM populations or fabricated forces. Accuracy did not
  improve: the three-case MAE/RMSE changed from 1.634/1.636 at the fixed
  geometry to 1.790/1.793 kcal/mol, with all three cases worsening. This
  remains an optional negative diagnostic and is not a Route 1 product
  default. A second full run reproduced every scientific record exactly after
  excluding only provider and wall-clock timings.
- A separate force-consistent OBC-II/ACE TI diagnostic used the exact
  `U_MLIP,gas + lambda*W` energy and force with MACE-OFF23m, AIMNet2, and
  ANI2x. For each MLIP, two chains crossed five windows for the same rigid,
  limited, and flexible development molecules; every label-free raw record
  stores 40 production samples per window and the complete five-state reduced
  potential matrix. Three-case fixed-geometry MAE/RMSE is `2.150/2.428
  kcal/mol`; sampled TI gives `2.218/2.532`, `2.198/2.508`, and `2.234/2.575`
  for MACE, AIMNet2, and ANI2x respectively. Thus all three MLIPs worsen MAE
  by `0.047-0.083 kcal/mol`. The largest chain gap is `0.136 kcal/mol` and the
  minimum endpoint-FEP ESS fraction is `0.685`, but every result remains
  fail-closed because each window is only 30 fs; the original TI phase did not
  evaluate MBAR overlap. This is multi-MLIP evidence for a real but small
  ensemble term, not a production free-energy or accuracy gain.
- The complete stored reduced-potential matrices were subsequently analyzed
  label-free with an upstream PyMBAR-backed MAPLE API. Across all nine
  model/case records, the minimum directional adjacent overlap is `0.145`, the
  maximum absolute MBAR/TI difference is `0.112 kcal/mol`, and the maximum
  independent-repeat difference is `0.239 kcal/mol`; all pass their diagnostic
  bounds. PyMBAR's weight-based effective counts also pass 50 for every record,
  but per-state time-series decorrelation leaves only `9-19` samples, so all
  records fail the prospectively stated minimum of 20 and retain the explicit
  no-equilibrium claim. Separate development scoring changes the common
  fixed-geometry MAE of `2.150 kcal/mol` to `2.207`, `2.212`, and
  `2.219 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m. The upstream estimator
  plumbing is therefore validated, while product solvation-free-energy and
  accuracy claims remain closed.
- A label-free endpoint-provider correction then reused the same
  force-capable OBC-II/ACE lambda-one solution trajectories and evaluated the
  energy-only AM1-BCC/CHA-GB/PBSA cavity-dispersion endpoint. The exact
  one-sided Zwanzig difference contains no gas-leg correction because the
  selected gas MLIP is identical at both solvent endpoints; no MM gas energy,
  residual, label fit, or retraining enters the target. All 720 high-endpoint
  evaluations completed in `21.64 s`. The observed weight diagnostics are
  favorable (`minimum ESS fraction=0.911`, `maximum normalized weight=0.136`),
  but only `4-13` decorrelated samples survive per replicate and the parent
  30-fs trajectories cannot establish equilibrium. Consequently `0/9`
  numerical gates and `0/9` scientific gates pass. Separate development
  scoring lowers the three-case MAE from the common fixed-geometry
  `2.150 kcal/mol` to `1.725`, `1.646`, and `1.777 kcal/mol` for AIMNet2,
  ANI2x, and MACE-OFF23m. The improvement is dominated by propionic acid,
  while methyl hexanoate worsens for all three MLIPs. This motivated a broader
  prospective screen; by itself it is not a product, accuracy, equilibrium, or
  speed-versus-MM claim.
- The broader label-free screen reuses all 19 common cases and 1,270 frozen
  states, changes only the solvent endpoint to fixed-AM1-BCC/CHA-GB/PBSA, and
  evaluates it twice for every state. All 2,540 provider calls complete in
  `145.51 s`; endpoint arrays and propagated discrete free energies repeat
  exactly. The energy artifact is
  [`route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-2026-07-25.json).
- Development scoring is separately sealed in
  [`route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json`](benchmarks/route1-multi-mlip-chagb-discrete-conformer-score-2026-07-25.json).
  The common fixed high-endpoint MAE is `1.700 kcal/mol`; discrete high-
  endpoint weighting gives `1.699/1.720/1.709 kcal/mol` for AIMNet2, ANI2x,
  and MACE-OFF23m. Paired MAE-gain intervals versus fixed high-endpoint
  geometry are `[-0.148, 0.173]`, `[-0.087, 0.047]`, and
  `[-0.073, 0.056]`; only `6/19`, `7/19`, and `7/19` cases improve, while
  complete weight diagnostics pass for no model. Every frozen long-sampling
  signal gate fails, so the decision is
  `long_sampling_candidate_not_supported`. The gain over the low-endpoint
  discrete values is provider-driven; this screen does not support spending a
  large MD budget to seek an MLIP conformer-weighting accuracy gain.
- A separate label-free indirect-free-energy diagnostic uses
  GAFF2/OBC-II/ACE only as a reference sampler and keeps the target exactly
  `U_MLIP,gas + W_OBC-II/ACE`. Each of the nine model/case records evaluates
  400 reference configurations with target energies and no target forces.
  Reference and target solvent terms agree within `1.83e-5 kcal/mol`, but the
  sampling distributions do not: minimum gas/solution BAR overlaps are
  `0.015/0.057`, minimum gas/solution directional MBAR overlaps are
  `0.005/0.016`, minimum forward ESS fractions are `0.074/0.021`, and all gas
  and solution forward/reverse checks fail. Only `7/9` records pass every
  explicit MBAR/BAR solver-convergence check; the other two retain the exact
  PyMBAR final-nonconvergence messages in their raw diagnostics.
  Reference-only cycles differ
  from bidirectional MBAR by up to `2.514 kcal/mol`; bidirectional indirect
  MBAR differs from direct target MBAR by up to `1.378 kcal/mol`.
  Development scoring also worsens every three-case MLIP MAE. The mechanism
  is therefore retained as a fail-closed analysis extension, not a product
  sampler. Any retry must add prospectively frozen intermediate Hamiltonians
  or nonequilibrium switching; it must not add a label residual or put MM
  energy into the Route 1 target.
- The remaining nonequilibrium bridge was then executed prospectively with the
  same three MLIPs and cases. It used the exact linear
  GAFF2/OBC-II/ACE-to-MLIP/OBC-II/ACE Hamiltonian, a single common gas-derived
  energy offset that cancels between phases, eight work values per direction,
  and 5-fs/20-fs switching. The 20-fs cycle is much closer to the short direct
  target MBAR control than endpoint reweighting: the maximum absolute
  difference is `0.492 kcal/mol` and the nine-case mean absolute difference is
  `0.246 kcal/mol`. That apparent recovery is not convergence. The largest
  5-fs-to-20-fs cycle change remains `1.364 kcal/mol`, large directional and
  uncertainty failures remain in the individual phase legs, endpoint
  equilibrium/independence are explicitly unproven, and `0/9` complete
  numerical gate sets pass. Separate development scoring gives 20-fs MAEs of
  `2.281`, `2.468`, and `2.242 kcal/mol` for AIMNet2, ANI2x, and MACE-OFF23m,
  all worse than the common `2.150 kcal/mol` fixed-geometry baseline although
  all improve on the failed endpoint correction. The experiment uses `29,376`
  MLIP energy+force evaluations plus equal reference evaluations and does not
  beat the nominal force-call count of the deliberately short direct target
  diagnostic. It validates a generic bridge primitive, not accuracy,
  equilibrium, acceleration, or product promotion.
- A separate two-level diagnostic used the force-consistent
  MACE-OFF23m+AM1-BCC/OBC-II/ACE potential only to generate gas and solution
  minima, then evaluated every unique converged solution minimum with the
  energy-only AM1-BCC/CHA-GB/PBSA endpoint and reranked the high-level total
  energy. Across the pre-existing six label-exposed development cases, the
  high-level final SP improves over reporting the low-level relaxed endpoint
  (`2.834 -> 1.901 kcal/mol` MAE) but worsens the relevant fixed-geometry
  high-level baseline (`1.773 -> 1.901 kcal/mol`); all six cases worsen.
  Reranking changes one minimum and also worsens the low-selected two-level
  result (`1.871 -> 1.901 kcal/mol`). Both potentials and the absent final-SP
  force claim are recorded; the workflow is not promoted.
- Four locally traceable 23-atom warm energy+force runs separate solvent
  overhead from a named MM comparison. MACE-OFF23m gives 2.28% paired/2.12%
  correction-only on GPU/Reference and 0.45%/0.48% under a same-host,
  one-thread CPU/Reference policy; the latter MLIP+GB potential is about 8749x
  slower than named GAFF2/OBC-II. ANI2x gives 0.27%/4.93% on GPU/Reference
  and 14.96%/10.41% on one-thread CPU/Reference; the latter is about 444x
  slower than GAFF2/OBC-II. GPU/MM ratios are not local comparisons because
  the backends differ. OpenMM Reference is a correctness-oriented fixed
  backend here, so even the two same-host, one-thread ratios are local
  counterexamples rather than production-MM throughput comparisons. These
  observations reject both a global “faster than MM” claim and a universal
  “negligible solvent overhead” claim; all four 50-sample timing traces have no
  uncertainty interval.
- The product OpenMM backend is now separately audited rather than inferred
  from those old Reference traces. On the 12-case provider corpus,
  single-thread deterministic CPU reproduces Reference within
  `9.602e-6 kcal/mol` for energy/components and
  `1.235e-5 kcal/mol/A` for forces, repeats exactly across independent
  contexts, and is locally `2.40x` to `3.03x` faster for warm OBC-II/ACE
  energy+force evaluation (median `2.79x`). This reduces correction overhead
  only; it does not alter accuracy or overturn the explicit rejection of a
  general faster-than-MM claim.
- The default ACE component decomposition now uses one OpenMM state evaluation
  instead of separately evaluating polar and total contexts. A unit global
  parameter scales only the upstream ACE energy term, so its energy derivative
  is the exact nonpolar component and the total force is unchanged. For all
  five Amber GB models on the water unit case, the extracted polar component
  matches an independent polar-only context within `1e-12 Hartree`; the
  two-model 69-component
  finite-difference and three-model nine-task matrices remain green. LCPO
  continues to use a two-context difference.
- The **MLSES PB surface feasibility boundary** is frozen against the
  AmberTools 26 manual, the executed GENIUSES paper
  (DOI `10.1021/acs.jpclett.3c02176`), and the predecessor MLSES paper
  (DOI `10.1021/acs.jctc.1c00492`). Both learned objects approximate classical
  solvent-excluded-surface geometry rather than a hydration-label residual, so
  they are physically admissible in Route 1 in principle. In the maintained
  local CPU build, a complete `ENEOPT=1..4` / `FRCOPT=1..5` input discovery
  finds five runtime-accepted force pairings. Classical SES emits nonempty atom
  forces for all five at both grids, whereas GENIUSES emits no atom-resolved
  MLSES force for any of them: one terminates by signal and four abort during
  force projection. The finite-difference force gate is therefore not
  eligible. Three-process energy repeats on the same 23-atom case also make
  GENIUSES slower than classical SES at both grids. This is a local
  small-molecule boundary, not a universal large-system/GPU speed claim, but
  it opens no dependency, FreeSolv screen, or MLSES runtime provider.

## Open scientific gates

1. Human-review and freeze the proposed Amber/OpenMM and APBS parity bounds.
2. Obtain independent/human review of the expanded 12-molecule parity corpus.
   If broader GBn2 support is required, add an authoritative implementation
   that reproduces both phosphorus parameters and Amber's signed sulfur
   descreening branch in energy and force; current P- and S-containing GBn2
   inputs must continue to fail closed.
3. Keep the explicit AM1-BCC/CHA-GB/cavity-dispersion provider limited to the
   pre-registered **fixed-geometry SP-only accuracy profile**. Its reserve
   result and runtime parity cannot replace the force-consistent OBC-II/ACE
   OPT/SCAN path without an independently justified derivative-capable
   implementation and force gate.
4. Obtain a genuinely independent external or label-sealed evaluation. The
   116-record FreeSolv reserve has been computed behind a sealed energy/scoring
   boundary and passed its pre-registered gates, but its labels were exposed
   in the legacy preparation artifact. Do not call it blind confirmation or
   use it as the final certification claim.
5. Obtain and review the exact redistributable ABCG2-PBSA-2023 radii and
   nonpolar parameters before enabling that profile.
6. Keep PB energy-only and keep APBS/ACE out of the runtime. Its 526-case
   development accuracy improvement is too small. The original molecular-
   surface grid fails stability, while a converged finer grid still misses the
   material-gain gate. Any replacement PB endpoint must first pass an
   energy-convergence protocol, then an independent force gate.
7. Add a genuinely force-consistent runtime path for any selected
   high-accuracy GB/PB/nonpolar composition before derivative-based MLIP
   sampling; current CHA-GB, PB exact-difference, and unavailable AR6/igb9
   paths must not be reused as if force-capable. Broaden
   state coverage on a development-only subset and freeze each MLIP
   independently. IWM-GB and AGBNP remain research candidates until they supply
   a maintained generic parameter provider and pass the same derivative,
   applicability, and independent-accuracy gates. GBr6 has already failed the
   accuracy gate and must not be revisited without a materially different,
   independently justified implementation. The tested ddX/ddPCM pairing has
   valid polar derivatives but fails accuracy and performance admission; a
   revisit must change the independently justified physical pairing or
   performance regime, not relabel the failed endpoint. dSASA supplies a
   differentiable SASA term but not the coupled cavity/dispersion endpoint or
   the missing CHA-GB polar derivative. AmberTorchPB currently exposes a
   preassembled PB linear-system solver rather than a licensed molecular
   coordinate/force provider. GNNIS is conservative and solvent-only, but its
   solvent functional is trained against explicit-solvent mean-force labels;
   QM-GNNIS is explicitly a `G_GNNIS-G_GB-Neck2` learned delta transfer.
   Neither is integrated. Route 1 must not train or import any learned
   correction against experimental hydration labels or explicit-solvent force
   labels. GBMV2 is a higher-priority physical watch item, but remains blocked
   until a runnable, deployable upstream provider permits parity, all-\(3N\)
   force, accuracy, and throughput testing without a paper-based
   reimplementation. GBSW is the secondary watch item because it provides
   smoothed solvation forces but is slightly less accurate in the historical
   separated set and still about four times slower than vacuum. FACTS remains
   below both: it is less accurate and lacks prospectively complete
   small-molecule radius parameters.
8. Expand the now two-MLIP performance protocol across molecule-size bins and
   warm/cold SP, OPT, SCAN, and NVE/NVT MD before making any broader throughput
   statement.
9. Keep the frozen two-level `force-capable OPT -> high-accuracy final SP`
   probe benchmark-only. A broader attempt requires a prospectively frozen
   molecule/MLIP protocol and must still expose both potentials and provider
   manifests; do not report its final-SP energy as though optimization used
   that derivative.
10. Keep `inner=prebuilt` out of FreeSolv accuracy scoring until a separate,
    prospectively frozen cluster-continuum free-energy protocol supplies the
    missing cluster-formation/occupancy, standard-state, solvent-reference,
    and ensemble terms. Do not reinterpret the fixed-shell runtime potential
    as `Delta G_solv`.

Fresh engineering tests are evidence of implementation correctness, not a
claim of broad chemical accuracy.
