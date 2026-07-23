# Route 1 implicit-solvation validation status

This branch contains only the fixed-charge PB/GB route. All providers still
require `experimental=true` until the scientific gates below are closed.

## Passing engineering gates

- Tripos MOL2 topology, atom order, atom types, substructure metadata, and fixed
  charges are preserved; disconnected structures and charge-sum mismatches fail closed.
- MOL2 fixed charges, AmberTools AM1-BCC, and AmberTools ABCG2 are implemented
  with topology, mapping, command, output, and charge-residual audits.
- QEq-GTO and variational CQEq-GTO/GB pass their numerical and force-consistency
  tests, but remain explicit-only experimental controls and are never defaults or fallbacks.
- OpenMM HCT, OBC-I, OBC-II, GBn, and GBn2 execute with finite energies and
  forces. ACE, LCPO, and diagnostic polar-only paths are separated.
- The five-molecule Amber/OpenMM parity corpus covers all five GB models. The
  observed maximum differences are 0.011194 kcal/mol for energy and
  0.045856 kcal/mol/A for force.
- APBS LPB input generation, solvent/reference subtraction, APOLAR composition,
  audit persistence, provider failures, official Born-ion reproduction, and
  methanol/aniline grid sweeps are under regression tests.
- The ten-molecule FreeSolv pilot is frozen. AM1-BCC/OBC-II/ACE gives MAE
  0.680 kcal/mol; this is a pilot, not scientific certification.
- The full-corpus harness pins FreeSolv v0.52 and freezes development and
  confirmation membership without opening a certification claim.
- The complete 526-molecule development matrix is frozen: 5,238/5,260 attempts
  succeeded. The only 22 failures are the predeclared GBn2 phosphorus boundary.
  ABCG2/OBC-II has the lowest development MAE at 1.652 kcal/mol; all strata,
  failures, and per-attempt hashes are retained in the development summary.
- The frozen 20-case conformer-sensitivity protocol completed all generators
  and evaluated 1,294 CREST conformers. For ABCG2/OBC-II, the fixed-charge
  correction has a 1.791 kcal/mol all-case p90 range and a 4.277 kcal/mol
  flexible-case p90 range, so the single-geometry result is not relabeled as a
  population-averaged hydration free energy.

## Open scientific gates

1. Human-review and freeze the proposed Amber/OpenMM and APBS parity bounds.
2. Expand provider parity across more atom types, larger/flexible molecules, and
   an authoritative phosphorus-aware GBn2 provider.
3. Freeze the proposed product default and pass rule before opening confirmation.
4. Run the one-shot confirmation partition and record the human certification decision.
5. Obtain and review the exact redistributable ABCG2-PBSA-2023 radii and
   nonpolar parameters before enabling that profile.
6. Keep PB energy-only until a grid-converged independent force gate passes.

Fresh engineering tests are evidence of implementation correctness, not a
claim of broad chemical accuracy.
