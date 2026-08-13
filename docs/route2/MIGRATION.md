# Route 2 migration ledger

This ledger prevents the rebuild from becoming a cosmetic parallel stack.

## Phase status

| phase | status | evidence |
| --- | --- | --- |
| 0: freeze and reproduce | complete | baseline `15777aad`, `1430 passed, 15 skipped`; Phase 0 evidence bundle |
| 1: contracts and units | engineering complete; unadmitted | immutable API registries; ASE public eV/eV/A boundary; full regression suite |
| 2: coupling operator | mathematical implementation complete; physical gate open | matrix-free spaces/Q; exact conjugate two-width radial-GTO B/B* and moving-node VJP; single-width/local-jet remain separate diagnostics; all tiers false |
| 3: state equation/operational scalar | kernel complete; unadmitted | constrained residual, deterministic root, implicit adjoint, exact state/provider fingerprints |
| 4: fixed-topology C-PCM | backend complete; unadmitted | legacy-parity adapter plus independent Torch continuum-algebra oracle; independent surface-primitive oracle missing |
| 5: MACE-POLAR adapter/canaries | candidates complete; full gate open | the original 194-node profile retains its rotation/torque failure; a distinct fixed-box40/590-node diagnostic passes one-water Cartesian/orientation, distorted-water path/loop, and equilibrium box-tail gates; the multi-molecule/path panel is preregistered but unexecuted and physical-component admission remains open |
| 6: force/MAPLE integration | blocked | same-scalar force is callable internally but no vNext Tier F profile passes the full real-stack PES panel |
| 7: Hessian/FREQ/TS | pending | no vNext Tier H profile |
| 8: CDS/multisolvent/performance | blocked by Tier F/H | not started |
| 9: legacy archive/cleanup | pending | production still uses legacy engine |
| 10: strict variational | disabled | expected negative until formally proven |

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
