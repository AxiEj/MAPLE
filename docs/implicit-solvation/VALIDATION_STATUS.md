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
   \(3.27\times10^{-6}\) eV/angstrom. These MACE-side partials are now connected
   to the fixed-surface coupled adjoint slice, while
   cavity/operator motion and CDS remain outside it.
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
7. The fixed-surface learned-density response is now contracted analytically.
   `density_to_external_field_order()` supplies the required
   \(Q^\mathsf Tc\), `density_position_vjp()` supplies
   \((\partial_{\mathbf R}\mathcal M|_f)^\mathsf T\lambda\), and
   `fixed_surface_solvation_coordinate_gradient()` combines those with the
   intrinsic field gradient, PCM half-coupling, field-response adjoint, and
   gas/polarized MACE force difference. A resolved-root synthetic
   implicit-function oracle passes. The two largest real fixed-field density
   position-VJP components over three steps stayed below
   \(1.60\times10^{-6}\) eV/angstrom absolute and \(2.16\times10^{-6}\)
   relative error. In the complete real fixed-surface acetone canary, all
   displaced roots were below \(2.0\times10^{-11}\); the two largest coordinate
   components over three steps stayed below \(4.02\times10^{-6}\)
   eV/angstrom absolute and \(5.00\times10^{-6}\) relative error. After the
   base root, the analytic derivative was about 15--20 times faster than twelve
   root-resolved scalar energy evaluations in local canaries; exact
   host-specific timings remain in the corresponding artifact. This still
   excludes CDS and tessera/cavity/operator motion, so it is not a total force,
   PES, or portable performance result.
8. The provider audit rejects mixing PySCF SWIG/ISWIG derivatives with the
   current PCMSolver--GePol energy; the current PCMSolver C ABI has no force endpoint.
   A separately named smooth PCM profile must be evaluated.
9. The geometry-dependent atomic-tension part of SMD CDS now has an analytic
   coordinate VJP. A synthetic H/C/N/O all-branch finite-difference oracle and
   translation check pass. On the methanol NWChem-control geometry, the six
   largest fixed-area components over three steps stayed below
   \(5.64\times10^{-10}\) hartree/angstrom absolute and
   \(9.29\times10^{-7}\) relative error; the net translation-gradient norm was
   \(1.12\times10^{-19}\) hartree/angstrom. The current hard-visibility SASA
   still lacks \(\sum_i\gamma_i\,dA_i/d\mathbf R\), so this component is not a
   complete CDS gradient and is not yet added to a published force.
10. A clean diagnostic artifact at Route-2 commit `54cd781` evaluates a
    separately named Fibonacci-grid, SWIG-inspired CDS candidate. It supplies
    an area VJP and combines it with the atomic-tension VJP. Its own discrete
    energy derivative matches methanol finite differences over five step
    sizes; the smallest-step maximum absolute discrepancy is
    \(2.31\times10^{-12}\) hartree/angstrom, and translation closure is
    numerical zero. Its 5810-point static water/methane/methanol errors are all
    below `0.001 kcal/mol`, while `smd_water_cds()` remains unchanged. An
    external PySCF 2.13.1 Lebedev-SWIG control nevertheless finds
    `5.84--9.73%` dense-grid VJP differences that do not converge
    monotonically. At 5810 points, Fibonacci and Lebedev methanol rotation
    spans are `0.00352` and `0.00260 kcal/mol`; the Fibonacci residual torque
    is \(1.67\times10^{-4}\) hartree. The candidate is not public, not
    established as SWIG-equivalent, and not a full SWIG-PCM.
11. The explicit `cavity_policy=fixed-stability-branch` public-input canary
    selects `AREA=0.28 A^2, MINRADIUS=0.30 A` before evaluation and fails
    closed instead of probing the primary branch. Acetone, methyl acetate, and
    ethoxyethane completed with zero PCMSolver warnings and 17 ML--PCM
    iterations each. Relative to the prior warning-fallback selections, the
    largest hydration-energy change was `0.000532 kcal/mol`; the three-case
    MAE was `0.451006 kcal/mol`. This only validates the policy wiring and
    local energy invariance. The parameters are engineering stability
    hyperparameters, the legacy energy default is unchanged, and GePol
    topology continuity/derivatives remain unproven.
12. The clean staged artifact at Route-2 commit `3bd6a31` used external PySCF
    2.13.1 SWIG/IEFPCM and remains an investigation rather than an adopted
    provider. Orders 17/29 failed the predeclared rigid-rotation gate; order 35
    passed a one-molecule fixed-density discriminator with a
    `0.007734 kcal/mol` span. Full order-35 acetone and ethoxyethane ML--PCM
    runs converged in 18 and 17 iterations and differed from the fixed
    PCMSolver energies by `0.042540` and `0.001338 kcal/mol`. The fixed-potential
    operator-gradient canary reached `1.53e-8` relative finite-difference error.
    This supports continued optional-provider research only; it does not
    establish a total force, production provider, chemical-space accuracy, or
    speed advantage.
13. `ExternalMEPCavityResponse` and `SurfaceChargeState` now lock the common
    external-MEP energy boundary, including per-atom radii and the
    energy-conjugate \(q_{\mathrm{sym}}\). The current energy and fixed-cavity
    derivative paths share this contract, while the public provider remains
    PCMSolver-only.
14. `ExternalMEPCavityOperatorDerivative`,
    `continuum_operator_position_vjp()`, and
    `polarization_operator_position_gradient()` now lock the operator-only
    geometry VJP \(d\langle u,Q_{\mathrm{sym}}(R)v\rangle/dR\). A synthetic
    nonsymmetric, coordinate-dependent IEFPCM \(K/R\) system passes energy,
    bilinear, symmetry, and atom-field-pairing finite differences. This is an
    algebra/interface gate only: PCMSolver still fails closed because its
    GePol/operator derivative is unavailable, and moving-surface kernel plus CDS
    terms remain absent.
15. Compare the summed analytic force with central finite differences of the
    converged total energy, then enforce translation, rotation, and energy
    conservation checks.
16. Only after these gates pass, enable OPT/scan/TS/MD and call Route 2 a
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
