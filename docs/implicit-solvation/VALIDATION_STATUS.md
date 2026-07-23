# Implicit-solvation validation status

This ledger separates software completion from scientific certification.  All
PB/GB and Route-2 SMD input continues to require `experimental=true` until the
corresponding scientific gates below are closed with reproducible artifacts.

## Passing engineering gates

- Tripos MOL2 atom order, bonds, atom types, substructure metadata, and fixed
  charges are preserved; disconnected structures and charge-sum mismatches
  fail closed.
- QEq-GTO uses canonical QEq parameters plus a versioned single-Gaussian STO
  approximation, updates both hydrogen idempotential and screening exponent at
  every SCF iteration, and passes charge, KKT-residual, GTO-integral,
  full-hydrogen-SCF snapshot, and fixed-charge force tests.  The legacy
  `QEqTorch` boundary delegates to the same solver.
- QEq-GTO/CQEq result provenance now carries machine-readable
  `scientific_status=experimental`, `accuracy_certified=false`,
  `default_eligible=false`, and `selection_policy=explicit-only-no-fallback`.
  Parser and provider-failure regressions prove that QEq is neither a default
  nor a fallback from AM1-BCC/ABCG2.
- Polarizable QEq-GTO/GB switches to the nonlinear consistent-QEq derivative;
  analytic charge gradients, charge conservation, KKT residuals, projected
  local-minimum Hessians, and an end-to-end coordinate finite-difference force
  test pass.  It is engineering-valid but not accuracy-certified.
- OpenMM 8.5.2 executes HCT, OBC-I, OBC-II, GBn, and GBn2 with ACE or no
  nonpolar term on the local smoke molecule; all energies and forces are
  finite.  LCPO now uses OpenMM's separate upstream `LCPOForce` rather than
  incorrectly treating the CustomGB `SA=LCPO` marker as an energy term; its
  nonzero energy and complete-force finite difference are regression-tested.
- OpenMM 8.5.2's generic GBn2 table lacks Amber's phosphorus-specific
  alpha/beta/gamma parameters.  P-containing GBn2 inputs now fail explicitly;
  MAPLE neither invents parameters nor silently substitutes OpenMM defaults.
- The shared calculator boundary adds exactly one solvent correction and never
  adds an OpenMM/APBS gas-phase MM energy.
- APBS input generation, solvent-minus-reference parsing, polar/nonpolar
  composition, missing-executable errors, and audit-file persistence are under
  regression tests.  The optional live-provider smoke also passes with the
  locally available APBS 1.4.1 executable and preserves its full audit bundle.
- Route 2's public parser fails closed to official MACE-POLAR-1-M, MOL2
  fixed-conformer SP, neutral singlets, water, PCMSolver, SMD-IEFPCM, and the
  1 M to 1 M standard state. Route-1 `#charge`, alternate models/providers,
  PBC, forces, Hessians, D4, mock backends, and ddX are rejected.
- The official MACE-POLAR-1-M checkpoint loads through MACE's upstream cache
  with `mace-torch==0.3.16`; MAPLE changes no weight. Float64 is mandatory
  because the solvent response is a small difference between large absolute
  MLIP energies. A real water SCF smoke recovers a nonzero
  `1.1073e-5 Hartree` solute-polarization term that float32 had quantized away.
- The external PCMSolver v1.1.12-style C binding, matching official input
  parser, custom per-atom SMD radii, IEFPCM solve, `0.5*dot(MEP,ASC)` energy
  convention, and GTO/ASC reciprocity checks pass both fake-library regression
  tests and a real `libpcm.so` water smoke (204 tesserae). MAPLE does not
  vendor PCMSolver.
- Route-2 frozen and self-consistent response paths both execute with the real
  MACE/PCMSolver stack. The float64 water SCF converges in 8 iterations to
  density residual `9.32e-6 e` and energy residual `1.65e-8 eV`. These are
  implementation controls, not hydration-accuracy references.
- The native NumPy aqueous SMD CDS implementation matches static NWChem
  `mnsol.F` controls for water, methane, and methanol within the predeclared
  `0.015 kcal/mol` absolute tolerance; neither NWChem nor PySCF is a runtime
  dependency.
- Structured output separates gas MLIP energy, `Delta G_solv`, and their sum,
  records that ASE `free_energy` is not a thermochemical Gibbs free energy,
  and retains manifest, raw/parsed PCM input, arrays, SCF history, components,
  and provenance in the audit directory.
- AmberTools AM1-BCC/ABCG2 command construction, returned-MOL2 topology checks,
  charge checks, and index mapping audits are under provider-mocked regression
  tests.  Live AmberTools 26.0 calculations pass for both methods on a
  chemistry-stratified ten-molecule FreeSolv pilot.  Small Antechamber text
  output residuals are corrected by the documented FESetup procedure and fully
  audited; excessive residuals fail closed.
- The ten-molecule FreeSolv pilot is frozen under `benchmarks/`.  For the
  temporary experimental AM1-BCC/OBC-II/ACE default it gives MSE `-0.291`, MAE
  `0.680`, and RMSE `0.826 kcal/mol`.  This is a pilot, not a certification
  statistic.
- The full-corpus harness pins and hashes all FreeSolv v0.52 source artifacts,
  reconciles 642/642 structures, deterministically freezes 526 development and
  116 confirmation records, and keeps all ten inspected pilot molecules out of
  confirmation.  Atomic per-attempt records, resume behavior, failure-complete
  denominators, deterministic bootstrap summaries, stratification, and the
  immutable confirmation lock are under regression tests.  This is benchmark
  infrastructure, not chemical-accuracy certification.
- An independent AmberTools 26.0/Python-sander corpus now covers five neutral
  C/H/O/N/F/S molecules across HCT, OBC-I, OBC-II, GBn, and GBn2.  All 25
  OpenMM 8.5.2 Reference-platform comparisons complete.  Maximum absolute
  differences are `0.011194 kcal/mol` for polar/complete energy and
  `0.045856 kcal/mol/A` for polar/complete force; LCPO nonpolar energies agree
  to floating-point precision.  Raw Amber topologies, coordinates, commands,
  logs, and binary hashes are retained.  This establishes provider parity only.
- APBS 1.4.1 reproduces the official radius-3-A, charge-+1 Born-ion example at
  `-229.587890 kJ/mol`, `0.000505 kcal/mol` from the documentation's rounded
  `-229.59 kJ/mol`.  Methanol and aniline grid sweeps retain PQR/input/stdout/
  stderr/command artifacts from `0.50` through approximately `0.1667 A`; their
  largest finest-pair change is `0.113892 kcal/mol`.  The observed surface-grid
  sequence is not monotone, so this is a measured stability bound rather than a
  claim of analytic convergence.
- The repaired fixed QEq-GTO profile has a separate reproducible ten-molecule
  artifact.  Its best pilot MAE is `2.876 kcal/mol` with HCT/ACE; it removes the
  previous multi-electron charge catastrophe but is not competitive with
  AM1-BCC or ABCG2 in this unmatched Amber-GB pairing.
- The variational CQEq-GTO/GB polarizable diagnostic is also frozen.  It is
  stable (largest charge `0.894 e`) and force-consistent, but its best HCT/ACE
  MAE is `5.719 kcal/mol`; solvent polarization amplifies the unmatched-model
  error and must not be presented as an accuracy improvement.
- The literature-tuned ESE-GB-DNN executable was run as an external research
  benchmark without redistribution.  On the same ten molecules it gives
  MAE/RMSE `1.042/1.357 kcal/mol`; on all 642 FreeSolv v0.52 records it gives
  `0.775/1.116 kcal/mol`.  The latter is a screening result rather than an
  independent test because training-set overlap has not been excluded.  The
  published independent Mobley-141 RMSE is `1.30 kcal/mol`, while the newer
  flexible-molecule FlexiSol benchmark reports MAE/RMSE `3.3/5.1 kcal/mol`
  under its GFN2-xTB ensemble protocol.
- The permissively licensed Kallisto/Jazzy EEQ direction was independently
  checked rather than inferred from the ESE result.  Kallisto EEQ plus the
  same MAPLE HCT/ACE protocol improves the ten-molecule MAE/RMSE to
  `1.869/2.218 kcal/mol`, but remains worse than AM1-BCC and ABCG2.  Complete
  Jazzy gives `1.338/2.171 kcal/mol` on the ten records, but seven are exact
  training-table overlaps.  On all 642 records it gives `1.552/2.433`; the
  conservative strict-identifier-nonoverlap subset gives
  `2.004/2.968 kcal/mol`.  Its source and license are acceptable, but accuracy
  plus the absence of forces keep it out of the production OPT/PES provider
  list.

## Open scientific gates

1. Human-review and either accept or revise the proposed provider-parity bounds:
   Amber/OpenMM `0.02 kcal/mol` energy, `1e-6 kcal/mol` LCPO-only energy, and
   `0.06 kcal/mol/A` force; APBS official control `0.001 kcal/mol` and neutral
   successive-grid stability `0.15 kcal/mol`.  Until a reviewer freezes those
   values in `provider_parity_tolerances.json`, verification deliberately fails.
2. Expand the provider corpora before calling parity broad: add further atom
   types and larger/flexible molecules, and obtain an authoritative
   phosphorus-aware GBn2 provider rather than relaxing the explicit P failure.
3. Expand the frozen chemistry-stratified ten-molecule FreeSolv pilot to the
   full supported FreeSolv domain, including sampling/conformer sensitivity,
   before naming any MAPLE combination certified.
4. Obtain and human-review the exact redistributable ABCG2-PBSA-2023 optimized
   radii and refitted nonpolar parameters.  That provider remains a hard runtime
   evidence gate rather than an approximation.
5. Jointly parameterize or adopt a published, redistributable QEq-continuum
   profile before accuracy certification of `mode=polarizable`; the current
   CQEq implementation proves consistency, not transferability of the
   unpaired QEq plus Amber-GB parameters.
6. Do not integrate ESE-GB-DNN as a production MAPLE potential unless upstream
   supplies a redistribution license and a differentiable force/gradient
   interface.  Its current binary-only, scalar-energy interface is suitable
   for research benchmarking but cannot serve OPT, PES, or MD.
7. Treat Jazzy/Kallisto as a reproducible open control, not a new default.
   Reconsider only if a differentiable, force-tested version passes the same
   chemistry-stratified and flexible-molecule accuracy gates.
8. Run the pinned Route-2 development partition in
   `benchmarks/route2-protocol.json`, then freeze the one-shot confirmation
   rule before opening confirmation. Route 2 remains experimental unless the
   confirmation MAE is at most `1.5 kcal/mol`, failure rate is zero, and the
   predeclared runtime gate passes. Do not add a constant shift, refit, or tune
   against confirmation after failure.
9. Complete the separate MACE-POLAR density-response controls: Dip146 dipole
   MAE at most `0.25 D` and HR46 polarizability MAE at most `2.0 A^3`.
   FreeSolv alone cannot certify the density/response mechanism.
10. Expand beyond the fixed FreeSolv conformer only in a new protocol.
    Conformer ensembles, thermal corrections, ions, zwitterions, radicals,
    other solvents, and molecules outside 16--500 Da are explicitly outside
    Route-2 v1 and must not be inferred from a passing hydration benchmark.
11. Obtain an independently installed PCMSolver corpus across the supported
    elements and cavity sizes. The current real water smoke establishes ABI,
    units, cavity, and energy-convention plumbing, not broad provider parity.

The current test suite is therefore evidence for implementation correctness,
not a claim of chemical-accuracy certification.
