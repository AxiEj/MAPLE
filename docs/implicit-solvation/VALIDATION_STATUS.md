# Route-2 implicit-solvation validation status

This branch contains only official MACE-POLAR-1-M coupled to external
PCMSolver IEFPCM and MAPLE's native aqueous SMD CDS term. It remains
`experimental=true` until all scientific gates pass.

## Passing engineering gates

- The public parser is locked to `macepol-m`, neutral singlet fixed-conformer
  MOL2 input, water, PCMSolver, SMD-IEFPCM, SCF response, and 1 M to 1 M.
- The official MACE-POLAR-1-M checkpoint loads through the upstream cache with
  `mace-torch==0.3.16`; MAPLE changes no learned weight and requires float64.
- The PCMSolver v1.1.12-style C binding, matching Python parser, custom SMD
  radii, IEFPCM solve, and `0.5*dot(MEP,ASC)` convention pass fake-library and
  real-water smoke controls.
- Frozen and self-consistent response paths execute with the real
  MACE/PCMSolver stack; the water SCF smoke converges with a nonzero solute
  polarization response.
- Native aqueous SMD CDS matches static NWChem controls for water, methane, and
  methanol within the frozen 0.015 kcal/mol tolerance.
- Structured output and audit artifacts separate gas MLIP energy,
  `Delta G_solv`, and the combined result.

## Open scientific gates

1. Run the frozen development partition in `route2-protocol.json`.
2. Freeze the one-shot confirmation rule before opening confirmation.
3. Require confirmation MAE <= 1.5 kcal/mol, zero provider failures, and the
   predeclared runtime gate without post-hoc shifts or refitting.
4. Complete Dip146 dipole MAE <= 0.25 D and HR46 polarizability MAE <= 2.0 A^3.
5. Obtain an independently installed PCMSolver corpus across all supported
   elements and cavity sizes.
6. Keep forces, OPT/PES, other solvents, ions, radicals, and conformer ensembles
   outside v1 until separately planned and validated.

Fresh tests establish implementation correctness, not broad chemical accuracy.
