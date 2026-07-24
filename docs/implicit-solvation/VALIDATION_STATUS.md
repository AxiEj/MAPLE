# Route-2 implicit-solvation validation status

This branch contains only official MACE-POLAR-1-M coupled to external
PCMSolver IEFPCM and MAPLE's native aqueous SMD CDS term. It is a
Research/Innovation Route and currently remains an energy proof-of-concept,
not a complete solution-phase PES.

## Passing engineering gates

- The public parser is locked to `macepol-m`, neutral singlet fixed-conformer
  MOL2 input, water, PCMSolver, SMD-IEFPCM, SCF response, and 1 M to 1 M.
- The official MACE-POLAR-1-M checkpoint loads through the upstream cache with
  `mace-torch==0.3.16`; MAPLE changes no learned weight and requires float64.
- The PCMSolver v1.1.12-style C binding, matching Python parser, custom SMD
  radii, cavity-exterior point-multipole MEP, IEFPCM solve,
  `0.5*dot(MEP,ASC)` convention, and reciprocal ASC projection pass
  fake-library and real-water smoke controls.
- Frozen and self-consistent response paths execute with the real
  MACE/PCMSolver stack; the water SCF smoke converges with a nonzero solute
  polarization response.
- Native aqueous SMD CDS matches static NWChem controls for water, methane, and
  methanol within the frozen 0.015 kcal/mol tolerance.
- Structured output and audit artifacts separate gas MLIP energy,
  `Delta G_solv`, and the combined result.

## Primary next milestone: energy-consistent force

1. Keep the documented total-energy bookkeeping and differentiate the unmixed
   converged fixed-point residual with an adjoint solve. A real local-field
   derivative probe rejects the shortcut
   `dE_intrinsic/d[V,grad(V)] == [q,p]`.
2. The MACE local-field energy/density graph is exposed. Gas and polarized
   fixed-node-field intrinsic forces now match a real acetone central-difference
   component canary to \(4.37\times10^{-6}\) and
   \(3.27\times10^{-6}\) eV/angstrom. They remain MACE-side partials and must
   next be connected to the coupled adjoint.
3. The unmixed residual JVP/VJP is explicit on the neutral density tangent
   space and matches its dense synthetic operator. A real acetone MACE-response
   canary gives \(1.29\times10^{-7}\) maximum JVP finite-difference error and
   \(1.94\times10^{-12}\) absolute JVP/VJP bilinear-identity error. The fixed
   cavity PCM map/adjoint remains to be connected.
4. Fixed-density/ASC/surface point-kernel position VJPs now match central
   differences and preserve the differentiated reciprocal identity without
   dense Jacobians. This is an explicit component only, not a force capability.
5. The provider audit rejects mixing PySCF SWIG/ISWIG derivatives with the
   current PCMSolver--GePol energy; the current PCMSolver C ABI has no force endpoint.
   A separately named smooth PCM profile must be evaluated.
6. Implement the geometry-dependent SMD CDS/SASA derivative.
7. Compare the summed analytic force with central finite differences of the
   converged total energy, then enforce translation, rotation, and energy
   conservation checks.
8. Only after these gates pass, enable OPT/scan/TS/MD and call Route 2 a
   solution-phase PES.

## Secondary diagnostics

- FreeSolv fixed-conformer hydration errors remain useful for detecting gross
  energy-accounting or chemistry regressions, but expanding or tuning that
  benchmark is not the next Route-2 milestone.
- Dipole, polarizability, provider-parity, and cavity-stability controls remain
  mechanism diagnostics.
- Other solvents, ions, radicals, and conformer ensembles remain separate
  later extensions.

Fresh tests establish implementation correctness, not broad chemical accuracy.
