# Exact matrix-free ddPCM roadmap toward 300 atoms

Status: conditional native draft under Architect re-review, then Critic review. The user's current execution authorization covers only reversible Gate 1-2 research inside the standalone prototype. It is not whole-target consensus, production integration, admission, release, or authorization for later gates.

## Outcome and stop condition

Build a new, independently versioned research backend that can ultimately evaluate the unchanged native-FP64 MACE-POLAR + ddPCM + CDS scalar for at least 300 atoms on the current 34 GiB RAM / 8 GiB GPU host, including energy, forces, analytic HVP, and an assembled analytic full Hessian, without relaxing the current numerical or topology contracts.

The work stops at each gate unless its evidence is complete. A 300-atom constructor, operator action, or continuum energy is **not** evidence for force, HVP, full-Hessian, chemical-domain, public-input, or scientific admission. Production integration is out of scope for this plan and requires a later reviewed plan after all research gates pass.

## Frozen identity and current evidence

- Keep tracked V2/V3 code and historical evidence byte-for-byte unchanged. The experimental source freeze is recorded at Git head `57effbb51c3442d82ca6a77d1542c490d5fee595` in `.omx/research/ddpcm-matrixfree-300-20261001T062415Z/source-freeze.json`.
- Preserve the complete scalar identity, not merely its headline discretization:
  - checkpoint SHA256 `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a` and the exact MACE native-FP64 configuration/provider metadata;
  - FP64 inputs/model/outputs/process default, graph-longrange identity, `lmax=15`, `n_lebedev=1202`, `eta=0.1`, radii, dielectric, operator ordering, and fixed-topology branch policy;
  - learned active source columns `(0, 2, 3, 4)`, inactive embedded columns `(1, 5, 6, 7)` exactly zero, unchanged learned-source embedding matrix, charge convention, and source ordering;
  - Torch native SMD-CDS provider/configuration, exact solvent, legacy DAREAL configuration/topology, units/conversion constants, component-energy decomposition, and model/ddPCM/CDS/PES/source-file configuration hashes.
- Preserve or improve the current gates; never loosen them to obtain scale:
  - energy absolute difference <= `1e-10 eV`;
  - force maximum absolute difference <= `1e-8 eV/Angstrom`;
  - Hessian and HVP maximum absolute difference <= `1e-6 eV/Angstrom^2`;
  - true original-system relative residual <= `1e-12` for every primal, transpose, tangent, or adjoint solve.
- Preserve all-node topology decisions and hashes, including inactive/buried nodes and one-sided proofs. Tiling must not silently reduce certification to active nodes.
- Current untouched dense implementation retains full operators, two full-SVD certificates, and reusable LU factors (`maple/solvation/continuum/ddpcm_response.py`). Those stability semantics may be replaced only by an explicit, independently validated certificate; removing SVD is not itself a valid optimization.
- Current research evidence:
  - tracked focused baseline: 112 tests passed (`baseline-tests.log`);
  - initial transpose implementation exposed nine failures (`iteration-1-tests.log`), providing a regression history rather than being erased;
  - corrected research prototype: 28 tests passed freshly on 2026-10-01, covering exact streamed actions/transposes at small scale, full-resolution two-atom actions, topology identity, absence of retained dense/pair-grid tensors, energy parity, and fail-closed GMRES;
  - prototype iterative results remain explicitly `scientific_admitted=False` and `stability_certified=False`.

## Shared operator-response protocol

All later implementations must expose one mathematical protocol for each `O in {L,D,A,C}` at fixed certified topology, with `R in R^(3N)` and bounded vector/RHS blocks in coefficient space:

- `apply_O(R, x) = O(R)x` and `apply_O_T(R, u) = O(R)^T u`;
- `JVP_O(R, x; v_R) = d/dt [O(R+t v_R)x] at t=0`, without constructing `dO/dR`;
- `VJP_O(R, u, x) = grad_R <u, O(R)x>`, satisfying `<u,JVP_O(x;v_R)> = <VJP_O(u,x),v_R>`;
- `directional_VJP_O(R,u,x;v_R)` for the directional derivative of the VJP, plus mixed variants when `u` or `x` has a tangent, sufficient to form analytic HVP contractions without a full operator Jacobian/Hessian.

The same apply/transpose/JVP/VJP/directional-VJP contract applies to RHS and psi as maps of coordinates and active source, with distinct coordinate/source directions and adjoints; `L`/`D` are primitive and `A`/`C` must remain algebraically derived from `D`. MACE must supply source `Jv`, source `J^T w`, and weighted source-curvature action `d[J^T w]/dR[v_R]`; native CDS must supply energy, gradient/JVP, and HVP under its unchanged DAREAL scalar.

Invariant: no protocol method may materialize or retain a `B x 3N` operator/source Jacobian, `B x B`, global `N x N x G`, global pair derivative/Hessian array, or full learned-source Jacobian. Direction/RHS block width must be explicit and preflighted against a measured resource bound.

## RALPLAN-DR

### Principles

1. Preserve scalar/model/checkpoint identity before optimizing execution.
2. Replace storage and evaluation order, not physics, precision, cutoffs, topology, or tolerances.
3. Establish exact small-system streamed derivative oracles before introducing controlled approximations such as FMM; a direct 300-atom exact full Hessian is not a prerequisite for FMM scale work.
4. Fail closed on residual, conditioning/stability, topology, resource, or provenance uncertainty.
5. Separate operator parity, iterative solution, derivatives, full PES, 300-atom engineering scale, chemical admission, and public integration.

### Top drivers

1. Eliminate `B x B` retained operators/factors and global `N x N x G` geometry that make 300 atoms impossible in current memory.
2. Retain current E/F/H/HVP accuracy and `1e-12` true-residual behavior.
3. Produce auditable evidence on the actual 34 GiB RAM / 8 GiB GPU host without disturbing historical V2/V3 capability gates.

### Options considered

| Option | Accuracy identity | 300-atom memory prospect | Main risk | Decision |
|---|---|---:|---|---|
| Exact target/source-tiled matvec + transpose, then iterative solves and analytic directional response | Highest: same quadrature and arithmetic ingredients, with only summation/order changes | Removes quadratic retained storage; bounded tiles | CPU cost remains nominally quadratic; iterative stability certificate is unresolved | **Choose as reference and first implementation lane** |
| FMM before any exact derivative oracle | Controlled approximation, not exact parity | Best asymptotic time/memory | Adds expansion-error and derivative-validation variables before a trusted reference exists | Reject; allow FMM after small exact E/F/HVP/H oracles and an explicit error budget, without requiring direct exact 300-atom full H |
| Dense micro-optimization / fewer copies / larger memory gate | Same dense math | Still `O(B^2)` storage; cannot meet host target | Can disguise peak allocation without changing scaling | Reject as 300-atom strategy; retain only for small reference oracle |

## Work plan

### 1. Freeze and harden the research boundary

**Ownership:** research harness and provenance only under `.omx/research/ddpcm-matrixfree-300-*`; no edits to tracked runtime, API registries, profiles, domain gates, or old benchmark receipts.

- Recompute tracked source hashes before and after every experimental iteration and record Git head, dirty status, Python/Torch/SciPy versions, device, default dtype, constants, and test command.
- Record and compare the complete scalar-identity ledger above, including checkpoint/configuration/component/source hashes, active/inactive source semantics, solvent/native-CDS/DAREAL identity, units, and component energies.
- Keep the experimental backend impossible to import through `maple.solvation`, calculator construction, `.inp`, or capability registries.
- Preserve failing intermediate logs and superseding test receipts; never overwrite old evidence.

**Gate 1:** source-freeze hashes match; tracked `git status --short` is empty; an import/capability negative test proves no public reachability.

### 2. Complete the exact streamed operator reference

**Ownership:** research `StreamedDDPCM` operator and operator-only tests.

- Harden exact streamed `L`, `D`, `A`, `C`, their transposes, RHS/psi, and Jacobi diagonals for vector and blocked RHS inputs.
- Keep one target row and bounded source tiles; retain only `O(NG)`, `O(tile*G*q)`, and small constant tables. Do not retain any `B x B`, `N x N x G`, pair Jacobian, or pair Hessian tensor.
- Match the complete dense `ResponseTopologyCertificate` dataclass byte-for-byte/field-for-field, including `contract`, `topology_sha256`, `positions_sha256`, all counts/margins, ordered one-sided proofs, owner exclusions, inactive masks, and branch decisions.
- Add bilinear adjoint checks `<u, Av> == <A^T u, v>` independent of dense comparison, adversarial overlapping/buried-node geometries, multiple tile sizes, multiple RHS widths, and deterministic repeat checks.
- Reject mutation of positions, symbols, radii, dielectric, discretization, topology policy/margin, source state, operator constants, preconditioner, or cached geometry/configuration binding. No cache may be reused after an identity mismatch.
- Introduce the shared response-protocol interfaces and resource preflight now, but Gate 2 admits only value/transpose/RHS/psi behavior; JVP/VJP methods remain closed until their later gates.

**Gate 2:** all tests in `ddpcm-matrixfree-300-test-spec.md` sections A-B pass at reduced and full `lmax15/n1202`; dense parity remains inside a tighter operator-level budget that is demonstrably sufficient for downstream E/F/H gates; storage instrumentation reports no forbidden retained shapes.

### 3a. Establish explicitly uncertified iterative primal/adjoint diagnostics

**Ownership:** research-only solver adapter, solver receipts, and stability experiments.

- Use a nonsymmetric Krylov method (initially restarted GMRES/FGMRES), never assume SPD or silently substitute CG.
- Check the **unpreconditioned original-system** residual after every solve; preconditioned callback residual is diagnostic only. Fail on nonfinite values, nonzero solver status, stagnation, restart/max-iteration exhaustion, or residual above `1e-12`.
- Solve and record primal `A`/`L` and transpose `L^T`/`A^T` systems; compare complete states and continuum energy with dense LU on oracle-sized cases.
- Specify whether each preconditioner is applied on the left, right, or in a flexible scheme. Define the corresponding transformed solve and recover the physical solution before checking the original residual.
- A transpose solve must use a mathematically consistent transposed preconditioner or an explicitly independent adjoint preconditioner; validate it with preconditioned and unpreconditioned bilinear identities. Never reuse a primal preconditioner for transpose merely because dimensions agree.
- Treat Jacobi as the baseline, then evaluate block-diagonal/local-sphere and reusable geometry-aware preconditioners only if they preserve the same physical state, adjoint contract, and original residual.

**Gate 3a:** dense-oracle state/energy parity and all four true residuals pass across a conditioning-stratified small panel; failures reject explicitly. Results remain `stability_certified=False`. Clearly labeled small derivative research experiments may proceed to exercise the response protocol, but no admitted energy, derivative, large-scale capability, or public path may depend on 3a.

### 3b. Define a deterministic conservative replacement stability certificate

**Ownership:** stability research and independent validation, distinct from solver tuning.

- Specify a deterministic certificate with a conservative mathematical error/conditioning bound tied to the `1e-12` solve residual and downstream E/F/H tolerances, or document that no adequate certificate has yet been found.
- Calibrate and adversarially validate it against exact dense SVD on small systems, perturbation experiments, known near-singular systems, topology-edge negatives, and derivative sensitivity.
- Randomized condition/singular-value probes may be supporting diagnostics only: they cannot alone certify stability or justify a “no false positive” guarantee.
- Any inconclusive, probabilistic-only, out-of-domain, nonfinite, or threshold-marginal result remains uncertified and fails closed.

**Gate 3b:** Architect/Critic accept the deterministic certificate and an independent verifier finds no false admission on the finite validation panel. This is bounded evidence, not a universal guarantee. Until 3b passes, all results remain uncertified even when later small derivative research tests are informative.

### 4. Add analytic directional operator/source response without global derivative tapes

**Ownership:** streamed first-directional derivatives and tangent/adjoint solve layer.

- Implement and test the shared JVP/VJP protocol for `L/D/A/C`, RHS, and psi, including JVP/VJP duality and directional derivatives of VJPs/mixed second contractions directly per target/source tile.
- Preserve the existing fixed-topology derivative convention and learned-source coordinate response. Numerical differentiation is validation-only, never the production derivative.
- Supply MACE source `Jv` and `J^T w` without a retained full source Jacobian; bind them to checkpoint/configuration/source-embedding identity. Add the native-CDS/DAREAL first-response path and its existing regression panels.
- Implement analytic energy/force through implicit differentiation using primal and transpose solves; reuse central geometry/preconditioner where mathematically valid, but bind every cached object to geometry/topology/configuration hashes.
- Do not yet claim learned-source curvature or second derivatives.

**Gate 4 research result:** continuum first derivatives and then full MACE-POLAR + continuum + CDS forces pass dense analytic parity (`E <=1e-10`, `F <=1e-8`) and topology-fixed independent finite-difference audits on oracle-sized cases; all tangent/adjoint solves pass `1e-12`; no global pair derivative tensor is retained. If Gate 3b is still open, this result is explicitly uncertified research evidence and cannot be used for admission, scale, or public capability; certified Gate 4 remains blocked on 3b.

### 5. Implement analytic HVP, then assemble the full Hessian

**Ownership:** directional second-response kernel, learned-source-curvature integration, and Hessian assembler.

- Complete the shared directional-VJP and mixed second-contraction protocol for geometry, operator, RHS/psi, native CDS/DAREAL, and MACE weighted source curvature `d[J^T w]/dR[v_R]`. Avoid materializing full operator/source Jacobians or Hessians and never replay an autograd graph through Krylov iterations.
- Validate one direction at a time first. Then support an explicit preflighted direction/RHS block width derived from tile, `N/G/q`, Krylov workspace, MACE/CDS activation, and measured RAM/GPU headroom; reuse central state/preconditioner only under immutable identity binding and without sharing mutable convergence state.
- Assemble the `3N x 3N` Hessian from analytic HVP blocks; the final 300-atom FP64 matrix is small relative to operator data, but each column remains independently residual- and provenance-bound.
- Preserve raw symmetry, translation-mode, HVP-vs-full-Hessian, topology-rejection, and independent directional audits. Do not “repair” a failing Hessian by symmetrizing before raw checks.

**Gate 5:** oracle-sized `H/HVP <=1e-6`, HVP/full-Hessian consistency, raw symmetry/translation limits, learned-source-curvature tests, and all solve residuals pass; resource traces show no forbidden quadratic retained continuum arrays. Only then proceed to 300-atom derivative scale.

### 6. Run honest scale ladders and optimize time without changing identity

**Ownership:** CPU resource harness first; GPU/MACE profiling only after CPU correctness gates.

- Use deterministic continuum-only synthetic geometries at 10/50/100/200/300 atoms to isolate scaling, plus real in-domain molecules where available. Label synthetic or >500 Da/HCO-incomplete cases engineering-only; they do not waive the current 16-500 Da, H/C/O CDS, solvent, model-domain, or public gates.
- Measure constructor peak RSS, retained bytes, one `L/D` action and transpose, iteration counts, true residuals, energy solve, force, HVP, and full Hessian as each capability exists. Record cold/warm timing separately.
- Initial safe CPU envelope: process RSS hard cap 8 GiB, no swap-driven result acceptance, one benchmark process, no dense oracle above the historical dense preflight, and a watchdog. Construction/storage checks may run through 300 atoms. For expensive exact actions, sample target rows first and project cost; run a full size only when the prior rung completed and projected wall time is within the run-specific watchdog. A timeout is recorded as a performance failure, never converted to success or a relaxed accuracy gate.
- Use exact tiling, RHS blocks, cached constants, deterministic reductions, and safe recomputation for small exact E/F/HVP/H oracles. Once those derivative oracles pass, FMM may begin as a separate experimental scale backend without waiting for a direct exact 300-atom full Hessian. Its expansion/order/near-field convergence and derivative error budget must be validated against exact small/medium cases and be small enough to preserve the unchanged downstream E/F/H gates.
- Profile the full PES memory split (MACE activations, continuum, CDS, solver workspaces, Hessian output). Apply checkpointing/offload only after second-derivative correctness tests, and never infer native FP64 from model/output dtype alone.

**Gate 6a — continuum engineering:** a 300-atom continuum operator/solve fits under 8 GiB RSS with complete provenance and residual evidence. This is not derivative or PES support.

**Gate 6b — derivative engineering:** 300-atom continuum force and HVP, then full Hessian, fit the host and pass all derivative gates.

**Gate 6c — full-PES engineering:** a 300-atom MACE-POLAR + ddPCM + CDS E/F/HVP/full-H run fits 34 GiB RAM / 8 GiB GPU and passes identity/accuracy/resource checks. This remains engineering evidence if chemistry/domain admission is not separately established.

### 7. Independent review before any tracked integration

**Ownership:** reviewer distinct from implementation owner.

- Architect reviews operator/derivative boundaries, stability-certificate replacement, caching identities, and whether exact-tiled or FMM is ready for a new versioned tracked backend.
- Critic adversarially checks hidden dense allocations, transpose mistakes, topology-mask loss, preconditioned-vs-true residual confusion, learned-source curvature omissions, raw-Hessian laundering, and engineering/admission conflation.
- Verifier reruns the frozen baseline, research suite, resource ladder, and a fresh end-to-end public negative test proving the backend remains unavailable.

**Gate 7:** reviews contain no unresolved critical/high finding, all evidence is reproducible, and a separate integration/admission plan is approved. Until then: no API registration, `.inp`, capability flag, mass-limit change, solvent/element expansion, commit, push, or release claim.

## Native agent roster and handoff

- `architect`: approve protocol boundaries, Gate 3b certificate design, FMM transition, and cache identity.
- `critic`: challenge the revised plan only after Architect re-review passes.
- `executor`: own bounded research prototype changes under `.omx/research/...`; never edit tracked runtime in this plan.
- `test-engineer`: expand red/green operator, source-adjoint, MACE/CDS, mutation, and resource regressions independently from implementation.
- `verifier`: rerun frozen hashes, tests, residual/resource evidence, and public-negative checks.
- `researcher` / `dependency-expert`: consult official primary sources for FMM/Krylov/preconditioner details only when a concrete design decision needs them.

Immediate handoff is Gate 1-2 only: the reference executor may add the already bounded source-adjoint tests, then test-engineer/verifier rerun them. No heavy 300-atom run and no native goal creation is part of this handoff; durable goal creation requires an explicit user request.

## Acceptance criteria

1. Old V2/V3 tracked hashes and historical receipts remain unchanged throughout research.
2. Exact streamed forward and transpose operators match the dense oracle at both reduced and full production discretization, preserve all-node topology identity, and retain neither `B x B` nor `N x N x G` arrays.
3. Every iterative primal/transpose/tangent/adjoint solve specifies left/right/flexible and adjoint preconditioning, reports and passes true original-system residual <= `1e-12`, and satisfies the applicable bilinear identities; all failure modes reject explicitly.
4. A deterministic, conservative, reviewed matrix-free stability certificate replaces, rather than silently removes, the dense SVD admission meaning; randomized probes remain diagnostic.
5. Energy, force, HVP, and full Hessian independently meet the existing tolerances on dense-oracle cases, including learned-source curvature and raw invariance audits.
6. The 300-atom ladder produces measured RSS/time/iteration evidence and labels each rung precisely as operator, continuum, derivative, full-PES, or not executed.
7. The full target is not declared reached until a single 300-atom full-PES case passes E/F/HVP/full analytic Hessian on the stated host; chemical/public support remains closed pending separate qualification.

## Material risks and mitigations

- **Quadratic exact-matvec time:** measure per-target cost early; optimize exact tiling first; introduce FMM only against the exact reference with convergence evidence.
- **Loss of dense SVD safety:** split useful uncertified diagnostics (3a) from deterministic certification (3b); randomized estimates never certify; include adversarial near-singular negatives.
- **Transpose or reduction-order defects:** retain the existing red-to-green transpose history, add bilinear identities and blocked-RHS/tile invariance.
- **Topology drift through streaming:** compare complete certificates/hashes over every node and reject all marginal/ambiguous geometries as before.
- **Derivative memory reappears:** enforce forbidden-shape instrumentation at every derivative order and inspect peak RSS, not only retained tensors.
- **Krylov differentiation mistakes:** differentiate the implicit equation with explicit tangent/adjoint solves; never backpropagate through iteration history.
- **Incomplete learned-source physics:** keep second derivatives closed until MACE source curvature is present and independently tested.
- **Domain inflation:** keep 300-atom synthetic/real controls engineering-only when they exceed 500 Da or current H/C/O/CDS/model qualification.

## ADR

### Decision

Use an exact streamed, matrix-free ddPCM operator as the independent reference backend; layer iterative primal/transpose solves, analytic directional derivatives, HVP/full-Hessian assembly, and only then optional FMM acceleration. Keep every capability fail-closed and outside public integration until independently reviewed.

### Drivers

- Current dense storage cannot fit 300 atoms on the host.
- Accuracy and scalar identity take precedence over speed.
- A trustworthy exact reference is required to qualify any future approximation.

### Alternatives considered

- FMM before exact derivative oracles: rejected because it combines storage redesign, approximation, derivatives, and error budgeting too early. FMM after small exact derivative oracles is allowed and does not require direct exact 300-atom full H.
- Dense micro-optimization: rejected because it does not change the prohibitive asymptotic storage.

### Why chosen

The exact streamed route removes the immediate memory blocker while retaining the same mathematical ingredients, creates a direct dense-oracle comparison at small size, and provides the only defensible reference for later FMM and derivative work.

### Consequences

- Memory can become compatible with 300 atoms before runtime does.
- Stability certification becomes a first-class research task because full SVD is no longer available.
- Full-Hessian completion remains a late milestone; operator or continuum-energy success cannot be promoted as the final capability.
- A separate, versioned production backend and new admission plan will be required after research success.

### Follow-ups

1. Native Architect re-review of this revised roadmap and test spec.
2. Native Critic review after Architect amendments.
3. Execute only the next unpassed gate; preserve receipts and stop on a failed invariant.
4. Draft a distinct tracked-integration/admission plan only after Gate 7.
