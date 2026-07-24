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
   \(1.94\times10^{-12}\) absolute JVP/VJP bilinear-identity error. The
   fixed-cavity PCM map reuses the energy-path MEP/ASC/back-projection, requires
   `MATRIXSYMM=TRUE`, and closes its real acetone bilinear identity to
   \(6.66\times10^{-16}\); the full residual identity closes to
   \(1.24\times10^{-11}\).
4. The neutral-subspace matrix-free GMRES adjoint solver matches a direct dense
   synthetic solution and fails closed for a singular operator. A real
   PCM-coupled acetone random right-hand side reaches \(2.36\times10^{-9}\)
   relative residual in eight callbacks and ten operator applications.
5. The physical fixed-cavity energy-gradient right-hand side is implemented as
   \(\Pi_0[\mathcal P_{\mathbf R}^*g_f+Qf]\), where \(g_f\) comes from the
   exact MACE intrinsic-energy autograd graph rather than the returned density.
   A real acetone neutral-direction canary closes the PCM energy identity to
   \(1.11\times10^{-16}\) eV, keeps all three relative finite-difference errors
   below \(6.54\times10^{-7}\), and solves the physical adjoint to
   \(7.65\times10^{-10}\) relative residual. The saved density fixed-point
   residual is \(2.59\times10^{-6}\), below the configured \(10^{-5}\)
   threshold. Warm local reruns take roughly 0.4--0.6 s for the intrinsic
   field gradient, 0.02 s for RHS assembly, and 1.4--1.5 s for the adjoint
   solve; these are local diagnostics, not portable performance claims.
6. Fixed-density and fixed-surface/operator point-kernel position VJPs match
   central differences and preserve the differentiated reciprocal identity
   without dense Jacobians. Their composed PCM field-pairing VJP matches the
   six largest real acetone components over three steps with
   \(2.40\times10^{-7}\) eV/angstrom maximum absolute and
   \(1.65\times10^{-6}\) maximum relative error. Its local analytic time was
   0.032 s versus 0.611 s for 36 scalar evaluations (18 central differences).
   This excludes tessera/operator motion and remains an explicit component,
   not a force capability or portable performance benchmark.
7. The provider audit rejects mixing PySCF SWIG/ISWIG derivatives with the
   current PCMSolver--GePol energy; the current PCMSolver C ABI has no force endpoint.
   A separately named smooth PCM profile must be evaluated.
8. Implement the geometry-dependent SMD CDS/SASA derivative.
9. Compare the summed analytic force with central finite differences of the
   converged total energy, then enforce translation, rotation, and energy
   conservation checks.
10. Only after these gates pass, enable OPT/scan/TS/MD and call Route 2 a
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
