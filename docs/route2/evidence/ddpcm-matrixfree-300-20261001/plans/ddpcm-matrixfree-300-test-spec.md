# Exact matrix-free ddPCM 300-atom research test specification

This specification tests an independent research backend. Current execution authorization covers Gate 1-2 only. Passing it does not establish whole-target consensus or register/admit a public capability.

## A. Boundary and identity tests

- Assert the hashes in `.omx/research/ddpcm-matrixfree-300-20261001T062415Z/source-freeze.json` still match tracked files before and after each run.
- Assert `git status --short` remains empty for tracked files.
- Assert the complete scalar ledger: checkpoint SHA256 `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`; MACE/provider/config/graph-longrange hashes and FP64 process/input/model/output identity; `lmax15/n1202/eta0.1`, radii/dielectric/operator ordering; active source columns `(0,2,3,4)`; embedded inactive `(1,5,6,7)` exact zeros; source-embedding matrix/charge/order; native SMD-CDS provider/config; exact solvent and legacy DAREAL config/topology; units/conversion constants; component-energy decomposition; model/ddPCM/CDS/PES/source-file hashes.
- Assert import/capability/profile/`.inp` paths cannot select the research backend.
- Assert results always carry `scientific_admitted=False` and, until Gate 3b completes, `stability_certified=False`.

## B. Exact operator and topology tests

For 2-10 atom dense-oracle cases, reduced grids and at least the existing two-atom `lmax15/n1202` case:

- Compare RHS and psi; `L`, `D`, `A`, `C`; forward and transpose; single and blocked RHS; tiles `1`, a mid-size tile, `N`, and tile `>N`.
- Require operator errors tight enough to imply downstream gates; initially retain the current research bounds (`atol 2e-11`, `rtol 2e-12`) and tighten/redistribute only from measured end-to-end error, never to excuse a regression.
- Check `<u, Av>` vs `<A^T u, v>` for every operator with independent seeded vectors.
- Require complete `ResponseTopologyCertificate` dataclass equality versus dense construction: contract, topology and positions SHA256, counts, optional margins, and ordered full one-sided-proof records, in addition to exact `fi/ui/gi` and all-node masks.
- Include exposed, overlapping, buried, transition, near-margin rejected, coincident-node rejected, nonfinite, FP32, wrong-shape, and mutated-input cases. Mutate positions, symbols/radii, dielectric/discretization/margin, source state, constants, preconditioner, and cached identity one at a time; require rejection or construction-time defensive isolation as specified, never stale cache reuse.
- Inspect retained tensors and sampled peak allocations: forbid `B x B`, `N x N x G`, global pair Jacobians, and global pair Hessians. Permit explicit `N x G x 3` nodes only if measured within the resource model.
- Repeat actions to prove determinism and absence of hidden mutable-state dependence.
- Preflight RHS/direction block width and prove the reported bound includes kernel tiles, Krylov basis, output block, topology geometry, and later MACE/CDS workspace. Oversized blocks fail before allocation.

## C. Shared response-protocol tests

For each `O in {L,D,A,C}` and separately for RHS/psi coordinate and source maps:

- Compare `apply_O` and `apply_O_T` with dense actions and check `<u,Ov>=<O^T u,v>`.
- Compare `JVP_O(x;v_R)` with dense analytic contractions and topology-fixed finite difference as validation only.
- Compare `VJP_O(u,x)` with dense contraction and require `<u,JVP_O(x;v_R)>=<VJP_O(u,x),v_R>`.
- Compare the directional derivative of VJP and every mixed variant needed when `u`/`x` has a tangent with dense analytic second contractions and independent directional validation.
- Verify `A/C` responses are algebraically derived from `D` and identity terms; no separate drifting implementation.
- Test MACE source `Jv`, `J^T w`, and weighted curvature `d[J^T w]/dR[v_R]`, including active columns `(0,2,3,4)`, inactive embedded exact zeros, neutral charge, embedding hash, and checkpoint/config identity.
- Test native CDS energy/gradient/HVP and DAREAL topology/configuration against the existing native-CDS, DAREAL, dtype, oracle/benchmark, and full entry-path regression panels.
- Instrument allocations and retained tensors: forbid `B x 3N`, full learned-source Jacobian, `B x B`, `N x N x G`, and global pair first/second derivatives for every protocol method.

## D. Solver diagnostics and deterministic stability tests

- Compare matrix-free GMRES/FGMRES primal states (`A`, then `L`) and transpose adjoints (`L^T`, then `A^T`) with dense LU.
- For every solve, recompute `||Ax-b|| / max(1,||b||)` using the original unpreconditioned action and require <= `1e-12`.
- Reject nonfinite action/preconditioner values, zero diagonal, nonzero solver status, stagnation, iteration/restart exhaustion, or residual failure.
- Run no-preconditioner, Jacobi, and candidates with explicit left/right/flexible semantics on a conditioning-stratified panel; recover the physical state before original-residual comparison. Preconditioning may change time/iterations, not the accepted state beyond the downstream budget.
- For transpose systems use the mathematically transposed preconditioner or a separately specified adjoint preconditioner. Check transformed-operator bilinear identities plus original `<u,Av>=<A^T u,v>`; a primal preconditioner is not implicitly valid for the adjoint.
- Gate 3a tests diagnostic convergence/state parity only and require `stability_certified=False`; small later derivative experiments remain labeled uncertified.
- Gate 3b tests a deterministic conservative certificate against dense full-SVD values, perturbation/error bounds, derivative sensitivity, and deliberately near-singular/ill-conditioned/topology-edge negatives. Randomized estimates are diagnostic only and can never independently set `stability_certified=True`. Inconclusive/marginal/out-of-panel cases fail closed.
- Record iteration counts, restart count, action calls/time, true residuals, preconditioned trace, condition evidence, and backend/configuration hashes.

## E. Energy and force tests

- Continuum-only energy against dense response: absolute difference <= `1e-10 eV`.
- Analytic continuum force, then full MACE-POLAR + continuum + CDS force: maximum difference <= `1e-8 eV/Angstrom` against unchanged dense V2 oracle where feasible.
- Independently validate directions with topology-fixed centered finite differences at multiple steps; numerical differentiation is validation-only.
- Exercise learned source coordinate response, inactive source modes, translated/rotated geometries, and fixed topology rejection.
- Require every implicit tangent/adjoint solve true residual <= `1e-12` and no forbidden retained derivative shapes.
- Run the existing MACE native-FP64 dtype/source-witness/oracle/benchmark and native-CDS/DAREAL regressions in addition to continuum-focused tests, at minimum:
  - `tests/route2_vnext/test_mace_polar_native_fp64.py`;
  - `tests/route2_vnext/test_mace_polar_response.py`;
  - `tests/route2_vnext/test_mace_polar_response_native_fp64_v2.py` and its explicitly provisioned real-checkpoint lane;
  - `tests/route2_vnext/test_native_smd_cds.py`;
  - `tests/solvation/test_torch_legacy_dareal.py`;
  - `tests/route2_vnext/test_ddpcm_response_operators.py`;
  - `tests/route2_vnext/test_ddpcm_response.py`;
  - `tests/route2_vnext/test_ddpcm_response_benchmark.py`.
  Component energies and all configuration hashes must remain identical. A skipped credential/model-dependent lane is reported as a validation gap, not a pass.

## F. HVP and full-Hessian tests

- Test directional second contractions independently for geometry, operator, RHS/psi, CDS, and learned-source curvature before composition.
- Compare analytic HVP and full Hessian with unchanged dense V2: maximum difference <= `1e-6 eV/Angstrom^2`.
- Require HVP == `H @ v` within `1e-6` for random, translation, and targeted learned-source directions.
- Check raw Hessian symmetry and translation response before any symmetrization/projection.
- Use topology-fixed finite-difference-of-force/HVP audits at multiple steps and reject branch changes.
- Vary direction block size; results must be invariant within the same gate and must not introduce global operator derivative tensors.
- Require block-width preflight to bound continuum tiles, Krylov vectors, output directions, learned-source-curvature work, and CDS activations; measure peak RSS/GPU memory against the prediction.
- Require and record `1e-12` true residuals for every response solve used by each direction/column.

## G. Resource and scaling ladder

Cases: deterministic 10/50/100/200/300-atom continuum controls, plus real in-domain cases where available. Every record labels synthetic/real, mass/elements/solvent, and whether it is inside current scientific domain.

Measure separately:

1. constructor/static retained bytes and sampled peak RSS;
2. one target-row action and projected full-action cost;
3. complete `L/D` forward and transpose actions when watchdog permits;
4. energy solve and iteration counts;
5. force;
6. one HVP;
7. full Hessian.

Safety envelope:

- CPU first, one process, RSS hard cap 8 GiB, no accepted swap-thrashing result, no dense oracle above historical dense preflight.
- Start each rung only after the prior rung produces complete evidence.
- Use a per-run watchdog selected and recorded before launch from the prior-rung projection. Timeout/OOM/nonconvergence is a failed or not-yet-executed rung, never a pass.
- Record `/usr/bin/time -v` or equivalent peak RSS, wall/user/system time, action counts, iteration counts, retained-byte report, environment, hashes, and exit status.
- GPU/full-PES runs begin only after CPU continuum correctness; record allocated/reserved peak GPU memory and ambient default dtype as well as input/model/output dtype.
- FMM scale work may begin once small exact E/F/HVP/H derivative oracles and a component error budget pass; a direct exact 300-atom full Hessian is not required first. FMM must converge against exact small/medium actions and derivatives and preserve the unchanged final gates.

## H. Regression commands and evidence layout

Minimum fresh sequence at each gate:

```bash
git status --short
sha256sum -c <generated-source-freeze-checksums>
pytest -q .omx/research/ddpcm-matrixfree-300-20261001T062415Z/test_streamed_ddpcm.py
pytest -q tests/route2_vnext/test_ddpcm_response_operators.py \
  tests/route2_vnext/test_ddpcm_response.py \
  tests/route2_vnext/test_ddpcm_response_benchmark.py
```

Then run the gate-specific resource/derivative harness under the watchdog and capture stdout/stderr plus machine-readable JSON. Never overwrite earlier red or green receipts. A final verifier reruns the complete applicable suite from a fresh process and confirms tracked source hashes again.

## I. Failure interpretation

- Operator parity failure: fix streaming algebra/order; do not tune downstream tolerance.
- Adjoint identity failure: block all transpose, force, and derivative claims.
- Residual failure: block the corresponding state and every derivative depending on it.
- Stability-certificate uncertainty or randomized-only evidence: retain diagnostic-only status even if residual is small; this does not prohibit clearly labeled small derivative research experiments.
- Topology hash/margin difference: reject the case; do not remap masks.
- Resource timeout/OOM: report the exact last passed rung and optimize/research; do not call it 300-atom support.
- Chemistry/domain failure: report engineering scalability separately; do not alter the 500 Da/HCO/public gates in this research lane.
