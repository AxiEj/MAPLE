# Route 2 migration ledger

This ledger prevents the rebuild from becoming a cosmetic parallel stack.

## Phase status

| phase | status | evidence |
| --- | --- | --- |
| 0: freeze and reproduce | complete | baseline `15777aad`, `1430 passed, 15 skipped`; Phase 0 evidence bundle |
| 1: contracts and units | engineering complete; unadmitted | immutable API registries; ASE public eV/eV/A boundary; full regression suite |
| 2: coupling operator | mathematical implementation complete; physical gate open | matrix-free spaces/Q; exact conjugate two-width radial-GTO B/B* and moving-node VJP; the harmonic branch has a geometry-assembled eight-channel Gaussian `S/S.T` intertwiner and rectangular exposure-product embedding; single-width/local-jet remain separate diagnostics; all tiers false |
| 3: state equation/operational scalar | kernel complete; unadmitted | constrained residual, deterministic root, implicit adjoint, exact state/provider fingerprints |
| 4: fixed-topology/union-sphere continuum | backend candidates complete; harmonic scalar assembly complete; unadmitted | legacy-parity C-PCM plus independent Torch continuum-algebra oracle; ordered-pair-frame candidate; full eight-channel ddX joint `(psi,phi)` ddPCM adapter; the fixed-dimensional harmonic branch now assembles nested/intersecting/tangent Coulomb `K`, rectangular `E`, `A=E.T K E`, and `S=E.T V` with structural `SO(3)` covariance; analytic coordinate derivatives, physical calibration, and all release gates remain missing |
| 5: MACE-POLAR adapter/canaries | candidates complete; admission gate open | the original 194-node and fixed-box40/CPCM590 paths fail rotation gates; CPCM1202 and finite-grid ddX also retain nonzero methanol rotation drift; a preliminary dirty-tree pair-frame/CPCM110 methanol run meets local thresholds but is not release evidence; every candidate remains unadmitted |
| 6: force/MAPLE integration | blocked | same-scalar force is callable internally; preliminary pair-frame methanol errors meet local thresholds, but the complete clean 20-molecule PES/symmetry/loop and matched-component accuracy gates have not run; public calculator/workflow and admission gates remain closed |
| 7: Hessian/FREQ/TS | pending | no vNext Tier H profile |
| 8: CDS/multisolvent/performance | blocked by Tier F/H | not started |
| 9: legacy archive/cleanup | in progress | continuum public exports are now lazy, so dependency-light harmonic submodules do not execute legacy adapters; admitted production paths still use legacy engine assets and are not archived |
| 10: strict variational | disabled | expected negative until formally proven |

Phase 10 now contains disabled scalar-first model and fixed-cavity continuum
engineering candidates.  The eight-channel effective source is generated from
one anchored field-energy graph, while fixed reciprocal C-PCM drive, source
HVP, and coordinate partial are generated from one Torch continuum scalar and
match the existing audited response backend.  This closes the earlier 4-to-8
rank obstruction only by changing the source/model identity and closes the
continuum half-coupling derivative identity only for the fixed-cavity candidate.
No combined stationary scalar or capability tier is enabled; sign/gauge,
stability/root, envelope, rotation, and release evidence remain pending.

The harmonic Tier-V branch additionally contains a dependency-light,
geometry-bound smooth weighted-overlap conductor reference. It replaces raw
laboratory-grid mask sampling with invariant one-dimensional pair coefficients,
exact declared-finite-band products, and a rectangular weighted basis `E`.
The physical shell Coulomb operator covers nested, intersecting, tangent, and
separated spheres; the retained matrices are `A=E.T K E` and `S=E.T V`, with
the receiver exactly `S.T`. A square exposure sandwich is locked out because
it cancels from the stationary energy whenever it is invertible. The resulting
reference is structurally rotation covariant and `C1`, but the exact shell
kernel is not generally `C2` at tangency. Analytic coordinate pullback,
combined electronic-continuum stationarity, physical calibration, and all
admission panels remain open.

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
