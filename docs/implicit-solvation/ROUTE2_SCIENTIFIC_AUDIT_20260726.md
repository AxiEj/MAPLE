# Route 2 scientific and production-readiness audit

**Audit date:** 2026-07-26  
**Repository:** `AxiEj/MAPLE`  
**Comparison:** `enhance` (`1477d32165cb7683520f4b07b1069ce75f29f433`) ... `implicitsolv-route2` (`d69049f8b5048c5c762d738f4151d54df822d310`)  
**Review branch:** `review/route2-scientific-contract-20260726`

## 1. Executive verdict

Route 2 is a serious research implementation rather than a cosmetic post-processing correction. Its central architecture—MACE-POLAR density response, continuum reaction map, self-consistent root, energy ledger, implicit-function adjoint, and same-provider coordinate derivative—is substantially better designed than a typical prototype.

However, the audited head must **not** be approved as a production implicit-solvent implementation. Three code-level scientific defects or contract violations are already identifiable from direct comparison with maintained reference implementations:

1. the native aqueous SMD Coulomb radii for P/S/Cl are shifted by one element;
2. the native aqueous SMD nitrogen surface-tension functional replaces the published short-range N-C3 switch with a different oxygen-environment expression;
3. the optional fixed-40-A reciprocal MACE evaluator performs coordinate centering before the upstream model creates the coordinate autograd leaf, so the documented centering Jacobian is not guaranteed to be propagated to returned coordinate derivatives.

In addition, severe PCMSolver/PEDRA tessellation warnings are currently audit-only rather than publication-fatal, and the pyddx force candidate is described as single-point-only but is not fully capability-gated at the Python/ASE boundary.

The correct current label is therefore:

> **Research/innovation implementation with a credible energy/adjoint framework, but with unresolved scientific defects and no production-qualified solution-phase PES.**

## 2. Scope and review method

The review was restricted to changes introduced relative to `enhance`. Unchanged base code was not re-reviewed except where a changed Route-2 call path depended on it.

The following production paths were reviewed at source level:

- public input validation and MOL2 ingestion;
- calculator dispatch and result composition;
- MACE-POLAR loading, local reaction-field injection, density response JVP/VJP, and coordinate VJP;
- PCMSolver C binding, input generation, cavity handling, ASC solve, and diagnostics;
- pyddx ddPCM source/multipole conversion, forward/adjoint maps, energy, and full coordinate VJP;
- PySCF SWIG/IEFPCM and SMD-CDS adapters;
- self-consistent density root, neutral-space residual, GMRES adjoint, total energy, and total correction force;
- native aqueous SMD radii, surface tensions, SASA/CDS energy, and derivative candidates;
- benchmark protocols, immutable evidence records, and public claim boundaries.

Reference comparisons used maintained source implementations rather than memory:

- PySCF PCM/SMD and analytic-gradient code;
- NWChem `mnsol.F` SMD equations;
- ddX/pyddx theory, Python interface, and force tests;
- graph_longrange GTO external/internal field projectors;
- upstream MACE-POLAR forward and force construction.

No independent real-runtime MACE/PCMSolver/pyddx/PySCF calculation was executed in the review environment. Existing tracked tests and immutable benchmark summaries were inspected, but they do not replace a clean rerun on the final corrected commit.

## 3. What is technically sound

### 3.1 Route definition

Route 2 is genuinely self-consistent. It does not merely add a fixed solvation correction after a gas-phase MLIP call. The implemented loop is

```text
c -> continuum reaction field P(c) -> MACE response M(P(c)) -> c'
```

with convergence on the learned density representation and intrinsic MACE energy.

### 3.2 Physical residual and numerical mixing

The derivative code linearizes the unmixed physical residual

```text
R(c) = Pi0 [ c - M(P(c)) ]
```

rather than differentiating the numerical mixing iteration. This is correct: mixing is a root-finding device, not part of the physical fixed-point equation.

The uniform total-charge mode is removed with an orthonormal reduced coordinate basis, so GMRES does not attempt to solve in a forbidden charge-changing direction. The post-solve residual is recomputed independently and failure is closed.

### 3.3 Energy bookkeeping

The implemented correction ledger is

```text
Delta G_solv
  = [E_MACE,intrinsic(V_reac) - E_MACE,gas]
  + U_continuum
  + G_CDS
```

and the reported combined value is

```text
E_solution = E_MACE,gas + Delta G_solv.
```

The local Route-2 adapter injects nodewise reaction-potential features while leaving the upstream graph-level uniform external field at zero. Consequently, the upstream explicit uniform-field dipole coupling is not added by this route. The continuum half-coupling is added once, and runtime reciprocity/energy identities guard against an accidental second factor of one half.

This ledger is internally coherent under the declared MACE adapter contract. It should nevertheless remain profile-versioned because a future model adapter whose returned energy already contains the full local external-potential coupling would require a different formula.

### 3.4 Same-provider derivative ownership

The force candidate correctly refuses to combine an energy from one continuum discretization with an operator derivative from another. One reaction-field object owns its forward map, adjoint, scalar polarization energy, and complete coordinate VJP. PySCF owns both SMD-CDS energy and its matching gradient in the pyddx force profile.

### 3.5 Reuse of mature software

The branch does not reimplement a complete boundary-element PCM solver. It wraps PCMSolver, pyddx, and PySCF, and uses explicit version/provenance gates. This is the correct direction for maintainability and scientific comparability.

## 4. P0 scientific defects

## P0-1. P/S/Cl SMD Coulomb radii are shifted by one element

### Current MAPLE table

`maple/function/calculator/extra_correction/implicit/smd_cds.py` currently assigns:

```text
P  = 2.47 A
S  = 2.12 A
Cl = 2.49 A
```

### Maintained SMD implementations

PySCF's SMD radius table is indexed by atomic number and gives:

```text
Si = 2.47 A
P  = 2.12 A
S  = 2.49 A
Cl = 2.38 A
Br = 2.60 A  # SMD18 revision
I  = 2.74 A
```

The MAPLE values are therefore consistent with dropping unsupported Si from a sequential list without shifting the following element labels back to their correct atomic numbers.

### Impact

This affects both public continuum providers because `route2_water_coulomb_radii()` supplies their electrostatic cavity spheres. Every P-, S-, or Cl-containing result uses the wrong cavity. The error propagates through:

- surface geometry or overlapping-sphere domain;
- continuum reaction operator;
- self-consistent reaction field and learned density;
- electrostatic solvation energy;
- pyddx coordinate derivative;
- every benchmark or application containing P/S/Cl.

This is not a harmless radius convention. The current documentation describes these values as SMD Coulomb radii, which is factually incorrect.

### Required patch

```text
P  = 2.12 A
S  = 2.49 A
Cl = 2.38 A
```

Keep Si unsupported if the selected MACE model/domain does not support it; do not shift its value onto P.

### Required regression gates

1. table-level parity with PySCF for every supported element;
2. cavity-radius audit for representative phosphate/phosphine, sulfur, and chloride organics;
3. PCMSolver and pyddx energy reruns for all affected historical records;
4. force finite differences for at least one S- and one Cl-containing molecule;
5. benchmark artifacts generated with the old radii must be marked stale rather than silently retained.

## P0-2. Native aqueous SMD nitrogen surface tension uses the wrong functional

### Current MAPLE expression

The native CDS code forms the expected coordination term

```text
t_NC = sum_C S_NC(r_NC) * environment(C)^2
```

but then forms a second term from oxygen coordination around carbon:

```text
t_nc_carbonyl = sum_C S_NC(r_NC) * oxygen_environment(C)
```

and adds

```text
84.10 * t_nc_carbonyl.
```

### Published/NWChem/PySCF expression

The aqueous SMD functional contains a separate short-range switch:

```text
S_NC3(r): r0 = 1.225 A, width = 0.065 A

t_NC3 = sum_C S_NC3(r_NC)

gamma_N = -48.22 * t_NC^1.3 + 84.10 * t_NC3.
```

The `+84.10` term is a direct N-C3 distance switch. It is not conditioned on a neighbouring oxygen and is not a carbonyl-environment term.

### Why existing tests missed it

The static native-CDS controls are water, methane, and methanol. None contains nitrogen. The analytic VJP test compares the derivative with finite differences of the same incorrect MAPLE scalar function, proving internal differentiation consistency but not literature fidelity.

### Impact

All nitrogen-containing molecules evaluated by the default PCMSolver/native-CDS path have a potentially wrong CDS term. The pyddx force candidate calls official PySCF `get_cds_legacy()` and is not affected by this particular native-functional defect.

### Required patch

- add `("N", "C3"): (1.225, 0.065)`;
- replace the oxygen-environment term with the direct N-C3 switch;
- rename the coefficient to avoid the false `carbonyl` interpretation;
- update the analytic VJP to differentiate the direct N-C3 switch;
- retain the ordinary N-C coordination term separately.

### Required regression gates

Use independent reference values, not only self-finite-difference tests. Include at least:

- HCN or acetonitrile;
- methylamine;
- formamide or another N-C-O environment;
- one geometry inside and one outside the N-C3 switching interval;
- official PySCF/NWChem native-CDS parity for a nitrogen panel.

## 5. P1 correctness and production blockers

## P1-1. Fixed-box centering Jacobian is not explicitly propagated

The reciprocal evaluator transforms coordinates as

```text
R_tilde = R - mean(R)
```

inside `MACEPolarLongRangeEvaluator.prepare_batch()`. At that point the batch coordinate tensor is not an autograd leaf requiring gradients. Upstream MACE later calls `requires_grad_(True)` on the already-centred tensor and computes forces with respect to `R_tilde`.

Therefore, the returned derivative is directly

```text
-dE/dR_tilde
```

whereas the derivative of the declared composite profile is

```text
-dE/dR = P * (-dE/dR_tilde),
P = I - 11^T/N.
```

The same transform must be applied to every coordinate cotangent, including:

- gas and fixed-field MACE force partials;
- `density_position_vjp()`;
- any future Hessian (`P H P` in Cartesian block form).

For an exactly translation-invariant numerical operator, the raw gradient may already have zero sum and the projection is numerically redundant. That empirical circumstance is not a substitute for implementing the declared chain rule. The documentation currently states that ordinary autograd supplies the projection, but the source-level graph construction does not guarantee that statement.

### Required design

The evaluator should own an explicit coordinate-transform contract:

```python
prepare_positions(R) -> R_tilde
coordinate_vjp(g_tilde) -> g_R
coordinate_hessian_pullback(H_tilde) -> H_R
```

The default real-space evaluator uses identity transforms. The fixed-box profile applies the centering projector. Do not scatter post-hoc force corrections across provider code.

### Required tests

- synthetic scalar oracle with a non-zero raw translation component;
- finite difference of `E(centre(R))` against the transformed analytic gradient;
- density-pairing VJP finite difference;
- translation closure after, not instead of, chain-rule validation;
- Hessian double-sided projection if Hessians are ever enabled for this profile.

## P1-2. Severe PEDRA tessellation warnings are not publication-fatal

The PCMSolver path separates native `PCMSolver warning.` stderr from `PEDRA.OUT` messages. This separation is useful, but the current policy permits a result to be published even when PEDRA reports messages such as a very poor tessellation being useful almost only for testing.

For a research diagnostic, retaining such a result with provenance may be acceptable. For production publication, severe cavity-quality diagnostics must fail closed. This does not require geometry-dependent retry/tuning: a fixed profile may simply reject the geometry after construction, preserving a branch-free energy definition.

### Required design

- classify exact known PEDRA diagnostics into `fatal`, `warning`, and `informational` categories;
- make poor/nonphysical tessellation and invalid added-sphere conditions fatal;
- preserve complete logs and structured codes;
- never silently retune cavity parameters after a production result fails.

## P1-3. Single-point-only status is not enforced at the lowest public boundary

The command parser blocks OPT/scan/TS/MD, but the pyddx provider advertises `forces` and accepts changed geometries. A Python user can attach the calculator directly to ASE optimizers or dynamics and bypass the CLI restriction.

Until rotation, active-set continuity, path conservativity, and short-NVE gates pass, the public provider should either:

1. reject geometry changes after construction; or
2. require an explicitly internal validation flag that is unavailable in normal user input.

A research force at one geometry is valuable, but it must not accidentally become an undocumented PES capability through ASE polymorphism.

## P1-4. PCMSolver must be process-isolated for production

The default provider loads a legacy C ABI in-process, changes process-global working directory, and redirects the process stderr file descriptor. The benchmark runner already contains a supervisor because PCMSolver can terminate a worker process rather than raise a Python exception.

Production execution should move PCMSolver into a short-lived worker or persistent isolated service with:

- timeout;
- exit-code/signal capture;
- structured request/response data;
- immutable input/output audit;
- no process-global cwd or stderr mutation in the caller.

A Python lock cannot protect against a native abort or unrelated code changing global process state.

## P1-5. Final evidence must be rerun on the corrected head

The current head added careful preregistration/disposition records but did not execute new MACE, continuum, Route-1, or QM calculations. Existing runtime-equivalence claims refer to earlier heads. After the P0 corrections, prior numerical results using native SMD radii/CDS are not evidence for the corrected model.

Required final evidence must be produced from a clean worktree/environment at the exact release commit and must archive:

- source commit;
- dependency versions and binary/library hashes;
- model checkpoint hash;
- all molecular input hashes;
- complete output/audit arrays;
- predeclared gates and untuned confirmation results.

## 6. P2 robustness and maintainability findings

### 6.1 Domain validation needs electron-parity and stronger chemistry checks

`charge=0, multiplicity=1` metadata is trusted, but electron parity is not checked. A neutral system with odd `sum(Z)` cannot be a singlet. Add a parity rule before loading the model.

The current salt/zwitterion screen recognises only a few Tripos types. This must be documented as a narrow heuristic, not a general proof that the molecule has no formal charge separation. Explicit-hydrogen and basic valence/protonation checks are also absent.

### 6.2 MOL2 element inference can misread uppercase atom names

For a lowercase GAFF/GAFF2 atom type such as `ca`, an uppercase atom name `CA` can be interpreted as calcium before the one-letter carbon fallback is considered. Element resolution should distinguish proper two-letter element notation (`Cl`, `Br`, `Si`) from uppercase force-field atom names (`CA`, `NA`, etc.).

### 6.3 Result objects should validate and freeze scientific outputs

`SolvationResult` is currently a mutable dataclass without a `__post_init__` guard. It should reject non-finite energies, invalid force shape, non-finite components, and inconsistent component closure, and should copy/freeze array and mapping payloads.

`MACEPolCalculator._polar_state_from_output()` should explicitly validate the finiteness of energy, density coefficients, and dipole—not only forces and shape.

### 6.4 Avoid exception-driven provider dispatch

The calculator finalizer retries `solvent_correction.evaluate()` without the `calculator` argument after catching `TypeError`. A genuine internal `TypeError` can therefore trigger a second evaluation, duplicate side effects, and obscure the original defect. Provider capability should be declared by an explicit protocol/signature version.

### 6.5 Consolidate duplicated validation

Route-2 restrictions are repeated in `CommandControl`, `SetCalculator`, the correction factory, and individual providers. Fail-closed duplication is safer than omission, but it creates drift risk. Introduce an immutable `Route2InputContract`/profile object validated once and rechecked by identity/version at each boundary.

### 6.6 Separate public production code from research candidates

Several experimental SWIG/Fibonacci/derivative candidates live in the ordinary package namespace. Move non-public candidates to a `research` or optional-provider namespace, keeping the production import graph small and making accidental dispatch impossible.

## 7. Requirements from the Route-2 project definition that remain open

The audited branch currently does **not** complete the broader Route-2 specification:

- water only; no methanol/ethanol, acetonitrile, or low-dielectric solvent;
- no user-defined solvent parameter object;
- no optical dielectric/refractive-index/non-equilibrium response interface;
- one SMD/PCM physical family with multiple numerical providers, rather than two clearly independent physical continuum schemes;
- no public direct-GTO-density vs point-multipole comparison;
- no ion/open-shell support;
- no certified optimization, reaction path, TS, scan, or MD;
- no short-NVE conservation test;
- no broad multi-solvent experimental benchmark;
- no broad QM-force benchmark;
- training-set exclusion for the evaluated molecules remains unconfirmed;
- no real solvent-sensitive reaction application has passed an end-to-end gate.

These are scope gaps, not reasons to discard the current architecture. They must simply not be described as completed capabilities.

## 8. Required remediation order

### Gate A — correct immutable chemistry constants/functions

1. fix P/S/Cl Coulomb radii;
2. fix the nitrogen N-C3 surface-tension term and VJP;
3. add independent PySCF/NWChem parity tests;
4. invalidate and regenerate affected evidence.

No performance or accuracy tuning should precede Gate A.

### Gate B — exact coordinate-transform and provider safety

1. add the fixed-box coordinate-transform VJP contract;
2. make severe PEDRA diagnostics fatal;
3. isolate PCMSolver in a worker process;
4. freeze scientific result objects and reject non-finite outputs;
5. enforce single-point-only status at the Python/ASE boundary.

### Gate C — real-runtime correction validation

For at least H/C/N/O/F/P/S/Cl/Br/I coverage:

- native CDS vs official PySCF/NWChem;
- PCMSolver vs pyddx/PySCF scalar comparisons with explicitly different discretizations;
- analytic correction force vs fully reconverged central differences;
- translation, rotation, and torque diagnostics;
- small displacement and torsion step-refinement;
- exact current-head environment/checkpoint provenance.

### Gate D — production PES qualification

Before enabling OPT/scan/TS/MD:

- multiple rigid orientations on multiple molecules;
- active-set/topology continuity statistics;
- multi-component Cartesian finite differences;
- closed-loop work test;
- short NVE energy drift;
- failure/restart determinism;
- no geometry-dependent provider selection.

### Gate E — scientific expansion

Only after A-D:

- add a solvent dataclass with units and provenance;
- implement at least two independent physical model profiles;
- add multiple solvents and custom parameters;
- compare point multipoles with direct GTO density projection;
- add broad experimental/QM benchmarks and solvent-sensitive reactions.

## 9. Minimum release checklist

A production tag must satisfy all of the following:

- [ ] no P0/P1 item in this audit remains open;
- [ ] all chemistry constants are checked against two independent maintained sources or the primary publication plus one implementation;
- [ ] every energy profile owns exactly one matching derivative profile;
- [ ] current-head real-runtime tests are archived and reproducible;
- [ ] broad benchmark confirmation is pre-registered and training-overlap status is reported;
- [ ] severe native diagnostics fail closed;
- [ ] user-facing capability names match actual scientific scope;
- [ ] model-weight licence is compatible with the intended deployment;
- [ ] no optimiser/MD can access an uncertified force path;
- [ ] documentation reports negative and failed gates as prominently as passing gates.

## 10. Primary source ledger

- Marenich, Cramer, Truhlar, SMD, *J. Phys. Chem. B* 2009, DOI `10.1021/jp810292n`.
- NWChem `src/solvation/mnsol.F` for aqueous SMD surface tensions and gradients.
- PySCF `pyscf/solvent/smd.py`, `smd_experiment.py`, `pcm.py`, and `solvent/grad/pcm.py`.
- Di Remigio et al., PCMSolver, *JOSS* 2019, DOI `10.21105/joss.01190`.
- ddX theory/interface and its PCM/COSMO/LPB force tests.
- Lange and Herbert, SWIG PCM discretisation, *J. Chem. Phys.* 2010, DOI `10.1063/1.3511297`.
- Upstream MACE-POLAR and graph_longrange source used by `mace-torch==0.3.16`.
- Mobley/Guthrie FreeSolv and the pinned FreeSolv 0.52 artifacts used by the branch protocol.

---

This report intentionally records both strengths and blockers. The correct next action is not a broad refactor or parameter tuning pass; it is a small, reference-driven chemistry correction, followed by exact derivative/safety gates and a clean current-head evidence regeneration.