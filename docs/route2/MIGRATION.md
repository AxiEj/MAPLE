# Route 2 migration ledger

This ledger prevents the rebuild from becoming a cosmetic parallel stack.

## Phase status

| phase | status | evidence |
| --- | --- | --- |
| 0: freeze and reproduce | complete | baseline `15777aad`, `1430 passed, 15 skipped`; Phase 0 evidence bundle |
| 1: contracts and units | engineering complete; unadmitted | immutable API registries; ASE public eV/eV/A boundary; full regression suite |
| 2: coupling operator | mathematical implementation complete; physical gate open | matrix-free spaces/Q; exact conjugate two-width radial-GTO B/B* and moving-node VJP; the harmonic branch has a geometry-assembled eight-channel Gaussian `S/S.T` intertwiner and rectangular exposure-product embedding; single-width/local-jet remain separate diagnostics; all tiers false |
| 3: state equation/operational scalar | kernel complete; unadmitted | constrained residual, deterministic root, implicit adjoint, exact state/provider fingerprints |
| 4: fixed-topology/union-sphere continuum | backend candidates complete; harmonic same-scalar geometry derivative candidate complete; unadmitted | legacy-parity C-PCM plus independent Torch continuum-algebra oracle; ordered-pair-frame candidate; full eight-channel ddX joint `(psi,phi)` ddPCM adapter; the fixed-dimensional harmonic branch assembles nested/intersecting/tangent Coulomb `K`, rectangular `E`, `A=E.T K E`, and `S=E.T V` with structural `SO(3)` covariance, and a separate Torch implementation differentiates the complete `R->E,K,V->G` scalar; physical calibration and all release gates remain missing |
| 5: MACE-POLAR adapter/canaries | candidates complete; original common-scalar and current molecular-realspace SO(3) routes closed negative | a clean official-checkpoint canary formally rules out retaining the original intrinsic energy and original four-channel source in one eight-channel scalar; a second clean zero-field counterexample isolates the pinned `graph_longrange` fixed-axis finite-difference feature/energy operators as non-`SO(3)`; a distinct changed-source common-state water canary passes replay and envelope FD, but every profile remains unadmitted |
| 6: force/MAPLE integration | blocked | same-scalar force is callable internally; preliminary pair-frame methanol errors meet local thresholds, but the complete clean 20-molecule PES/symmetry/loop and matched-component accuracy gates have not run; public calculator/workflow and admission gates remain closed |
| 7: Hessian/FREQ/TS | pending | no vNext Tier H profile |
| 8: CDS/multisolvent/performance | blocked by Tier F/H | not started |
| 9: legacy archive/cleanup | in progress | continuum public exports are now lazy, so dependency-light harmonic submodules do not execute legacy adapters; admitted production paths still use legacy engine assets and are not archived |
| 10: strict variational | disabled | original-energy/original-source route is formally ruled out by a real-checkpoint counterexample; the changed-source energy-gradient branch now has one reproducible real-checkpoint common-state/envelope canary but remains unadmitted |

Phase 10 now contains disabled scalar-first model and fixed-cavity continuum
engineering candidates.  The eight-channel effective source is generated from
one anchored field-energy graph, while fixed reciprocal C-PCM drive, source
HVP, and coordinate partial are generated from one Torch continuum scalar and
match the existing audited response backend.  This closes the earlier 4-to-8
rank obstruction only by changing the source/model identity and closes the
continuum half-coupling derivative identity only for the fixed-cavity candidate.
The common constrained state and stationary scalar `E-s<c,u>+sG` are now
implemented by thin scalar-derived adapters over the existing reduced solver.
Synthetic same-scalar tests with the real fixed C-PCM backend close reduced
JVP/VJP, coordinate pullback, cold/warm replay, half-coupling ledger, and a
re-solved envelope finite difference. The fixed harmonic snapshot also enters
the same kernel. At clean head `576550e9`, the official checkpoint plus the
changed eight-channel source converged one water common state, replayed it
exactly, and matched one envelope direction at all three displacement sizes;
the raw record is
[`evidence/variational-common-water-576550e9/`](evidence/variational-common-water-576550e9/README.md).
No capability tier is enabled. A later real-checkpoint common-state run with
the moving harmonic continuum converged, replayed its root, and passed the
same-scalar envelope directional check. Its harmonic continuum energy and
coordinate partial rotated at float64 roundoff, but its complete model-plus-
continuum scalar failed rotation. The subsequent model-only counterexample
under
[`evidence/mace-realspace-so3-nogo-6da676cd/`](evidence/mace-realspace-so3-nogo-6da676cd/README.md)
locates the first broken primitive in the pinned molecular-realspace MACE path,
not in the harmonic continuum.

The negative original-source decision is source/model/runtime bound under
[`evidence/mace-conjugacy-nogo-d17c35ac/`](evidence/mace-conjugacy-nogo-d17c35ac/README.md).
At both zero and deterministic nonzero field, more than `73%` of the
intrinsic-energy gradient norm survives projection into the gauge-reduced
missing source-dual subspace. A cold replay reproduced the complete protocol,
states, decision, and measurement digest exactly. This result is a formal
counterexample to the original common-scalar claim, not an admission of the
new eight-channel effective source.

The harmonic Tier-V branch additionally contains a dependency-light,
geometry-bound smooth weighted-overlap conductor reference. It replaces raw
laboratory-grid mask sampling with invariant one-dimensional pair coefficients,
exact declared-finite-band products, and a rectangular weighted basis `E`.
The physical shell Coulomb operator covers nested, intersecting, tangent, and
separated spheres; the retained matrices are `A=E.T K E` and `S=E.T V`, with
the receiver exactly `S.T`. A square exposure sandwich is locked out because
it cancels from the stationary energy whenever it is invertible. The resulting
reference is structurally rotation covariant and `C1`, but the exact shell
kernel is not generally `C2` at tangency. A sealed Torch candidate now
reassembles `E`, `K`, and `V` inside the stationary scalar and generates its
coordinate partial and mixed pullback from that graph; it matches the
independent NumPy matrices, finite differences, and rotation covariance in
synthetic tests. Combined electronic-continuum stationarity on a real
checkpoint, physical calibration, and all admission panels remain open.

## Asset mapping

| legacy asset | intended destination | rule |
| --- | --- | --- |
| `route2_plugin_spaces.py`, `electrostatic_pairing.py` | `solvation/coupling/spaces.py` | consolidate; do not create a third convention |
| `route2_plugin_contracts.py`, `route2_electronic_model.py` | `solvation/models/` | capability layers plus thin compatibility adapters |
| `route2_response.py`, `route2_fixed_point.py` | `solvation/coupling/` | separate physical residual from root algorithm |
| `route2_derivative.py` | `solvation/coupling/adjoint.py`, backend VJPs | one same-scalar derivative route |
| `route2_fixed_topology_surface.py` | `solvation/surfaces/fixed_topology.py` | preserve fixed cardinality/ownership |
| `route2_fc_aswig_cpcm.py` | `solvation/continuum/fixed_topology_cpcm.py` | reuse after parity, not duplicate |
| `gto_field_projection.py`, `gto_galerkin.py` | `solvation/coupling/exact_gto.py` | precision mainline with adjoint identity |
| `route2_force_admission.py` | `solvation/release/admission.py` | migrate gates; old certificate is not vNext admission |
| `route2_engine.py` | compatibility shell, then archive | shrink only after end-to-end parity |

Historical exploratory modules remain traceable until the new import graph is
complete. No production import may point from `maple/solvation` into the future
`research/route2_legacy` archive.

The current negative Phase-5/6 result is documented in
`MACE_POLAR_VNEXT_AUDIT.md`. It is not a solver-tolerance issue: the root and
adjoint residuals are tight, while symmetry and electrostatic-component checks
remain outside admission. Neither local-jet migration parity nor one local
same-scalar derivative promotes the radial profile.
