# Route-2 implicit-solvation validation status

The public branch contains only official MACE-POLAR-1-M coupled to an explicit
continuum provider. PCMSolver IEFPCM plus MAPLE's native aqueous SMD CDS remains
the default energy-only proof-of-concept. The separately named pyddx ddPCM plus
PySCF SMD CDS profile is now selectable as a single-point research force
candidate. The independent PySCF SWIG/IEFPCM adapter remains private. Route 2
is still a Research/Innovation Route, not a complete solution-phase PES.

## Passing engineering gates

- The public parser is locked to `macepol-m`, neutral singlet fixed-conformer
  MOL2 input, water, SMD, SCF response, and 1 M to 1 M. PCMSolver remains the
  default provider; pyddx requires an explicit exact force-candidate profile.
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
- The explicit
  `provider=pyddx,profile=smd-ddpcm-l15-n1202-v1` path composes one same-energy
  ddPCM scalar/derivative provider with the official PySCF SMD CDS
  energy/gradient, publishes only the solvent-correction force, and lets the
  shared calculator finalizer add the gas force exactly once.
- Structured output and audit artifacts separate gas MLIP energy,
  `Delta G_solv`, and the combined result.

## Primary next milestone: PES validation after the first public force candidate

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
    ethoxyethane completed with zero native `PCMSolver warning.` stderr markers
    and 17 ML--PCM iterations each. That earlier count did not include
    `PEDRA.OUT`; audit schema 5 now reports those side-file warnings separately
    instead of calling the complete cavity output warning-free. Relative to the
    prior warning-fallback selections, the largest hydration-energy change was
    `0.000532 kcal/mol`; the three-case MAE was `0.451006 kcal/mol`. This only
    validates the policy wiring and local energy invariance. The parameters are
    engineering stability hyperparameters, the legacy energy default is
    unchanged, and tessellation quality plus GePol topology
    continuity/derivatives remain unproven. The unit audit further shows that
    `AREA=0.28 A^2` becomes about `0.9999 bohr^2`, substantially coarser than
    PCMSolver's documented `0.3 bohr^2` default.
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
    GePol/operator derivative is unavailable. The separately named optional
    PySCF smooth profile supplies its own matching moving-surface/operator terms
    only under the narrower gate in item 17; CDS remains absent from that
    operator-only result.
15. A correct-unit acetone cavity scan found no warning-free `AREA` point from
    `0.3` through `1.0 bohr^2` at `MINRADIUS=0.30 A`: the fine end retained the
    native non-positive-definite-matrix warning, while the coarse end developed
    PEDRA tessellation warnings. At fixed `AREA=1.0 bohr^2`
    (`0.280029 A^2`), raising `MINRADIUS` to `1.0 A` removed one added sphere
    and cleared both warning channels. The resulting cavity is therefore
    changed, not merely quieted. Acetone, methyl acetate, and ethoxyethane then
    completed warning-free with the same 17-iteration roots, but their
    three-case MAE worsened from `0.451006` to `0.472175 kcal/mol`; the largest
    single energy shift was `0.053160 kcal/mol`. The candidate is not adopted.
    On acetone only, seven C=O displacements from `-0.02` to `+0.02 A` retained
    508 tesserae and the same added-sphere signature with zero warnings.
    Refining the central step from `0.01` to `0.005 A` reduced the relative
    derivative disagreement to `0.7248%` for total solution energy,
    `0.0719%` for hydration energy, `0.0298%` for electrostatics, and `0.366%`
    for CDS. This passes a narrow local energy-continuity gate only; it does
    not establish force continuity, broad transferability, or a public cavity
    parameter.
16. `FullReactionFieldPositionDerivative` and
    `continuum_coupled_solvation_coordinate_gradient()` now define the
    same-object contract for the complete coordinate VJP of one continuum
    reaction map. A provider must differentiate the exact forward/adjoint map,
    including solute projection, moving surface, continuum operator, and
    reaction-field back-projection; separately injected derivative arrays are
    not accepted. Synthetic resolved-root and split-partial tests lock the
    adjoint assembly and prevent accidental addition of the fixed-surface and
    full VJPs. The public PCMSolver map lacks this contract and fails closed at
    this top-level boundary. SMD CDS remains separate, no public production
    provider implements the full contract, and Route 2 still does not expose
    forces.
17. The optional, lazily imported `PySCFSWIGIEFPCMResponse` now implements one
    same-energy smooth-continuum full-position VJP without changing the public
    PCMSolver route. It combines solute-centre and moving-surface
    point-multipole derivatives with PySCF's matching SWIG/IEFPCM operator
    derivative. Per-atom cavity radii are passed through PySCF's own
    `gen_surface()` by a version-locked atom-index view; MAPLE neither copies nor
    reimplements the SWIG construction. The adapter verifies the required
    PySCF 2.13.1 integer-index lookup at runtime. With equal radii on repeated
    elements, all audited surface arrays and atom slices were bitwise identical
    to the ordinary element-table call. A mixed-radius methyl-acetate surface
    retained distinct `o=1.70 A` and `os=1.52 A` oxygen radii.

    On fixed-density order-17 acetone (643 surface points), the three
    representative \(10^{-4}\)-angstrom relative errors were
    \(1.54\times10^{-8}\), \(6.71\times10^{-9}\), and
    \(2.00\times10^{-7}\); translation closure was below
    \(1.9\times10^{-16}\) eV/angstrom. On fixed-density methyl acetate (695
    points), the maximum error over a general component and one component on
    each oxygen was \(3.50\times10^{-8}\) eV/angstrom
    (\(1.85\times10^{-6}\) relative) at a \(3\times10^{-4}\)-angstrom step and
    \(3.82\times10^{-9}\) eV/angstrom at \(10^{-4}\) angstrom; translation
    closure was \(6.10\times10^{-16}\) eV/angstrom.

    The real float64 MACE-POLAR fixed-point adjoint now also passes narrow
    resolved-root canaries. For acetone, the two largest checked components at
    \(10^{-3}\) angstrom had absolute errors \(2.15\times10^{-6}\) and
    \(1.02\times10^{-6}\) eV/angstrom. For mixed-radius methyl acetate, the
    checked largest component differed by \(8.51\times10^{-7}\) eV/angstrom
    (\(1.14\times10^{-6}\) relative), with both the adjoint relative residual
    below \(5.0\times10^{-14}\) and translation closure below
    \(1.0\times10^{-13}\) eV/angstrom.

    A clean controlled methanol grid refinement at commit `7da54dc` retained
    derivative accuracy while exposing the finite-grid rotation/cost tradeoff.
    The checked order-17 and order-35 whole-energy derivative errors were
    \(2.64\times10^{-6}\) and \(2.68\times10^{-6}\) relative. Their
    unrotated torque norms were \(8.37\times10^{-3}\) and
    \(8.43\times10^{-4}\) eV, computed as the Euclidean norms of
    `analytic_gradient.torque_ev` in the corresponding coupled-gradient
    `results.json` artifacts. In the predeclared three-orientation gate,
    order 35 and order 41 failed the \(10^{-3}\)-eV maximum-torque criterion
    at \(2.09\times10^{-3}\) and \(1.44\times10^{-3}\) eV.
    Order 47 passed this narrow gate with a `0.001476 kcal/mol` energy span,
    `0.001678 eV/angstrom` maximum gradient-covariance error, and
    \(5.16\times10^{-4}\)-eV maximum torque. Its 2211--2245 surviving points
    produced a 54.9-second three-orientation local run and about 3.12-GiB peak
    RSS. Because covariance error was nonmonotonic from order 41 to 47, this
    does not yet establish a generally rotation-qualified order.

    At this stage these results closed only a narrow continuum-electrostatic
    coordinate-gradient slice. PySCF's private gradient intermediates remained
    locked to version 2.13.1, CDS and total-force assembly were absent, and the
    public parser remained PCMSolver-only. At least one more rigid molecule,
    denser orientations, and small-angle continuity remain open. The canaries
    load MACE through the public calculator plumbing but do not evaluate its
    attached PCMSolver correction. In the corrected order-47 artifact the
    correction is explicitly detached before PySCF-SWIG evaluation and its
    loader-only manifest is not retained; the result-level PySCF provenance
    identifies the tested continuum. Local elapsed times are diagnostic
    metadata, not portable speed benchmarks or chemical-accuracy evidence.
18. `pyscf_smd_water_cds()` now provides a separately named optional wrapper
    around PySCF 2.13.1
    `pyscf.solvent.smd.get_cds_legacy`. That one wrapper, backed by compiled
    `libsolvent`, returns both the SMD CDS energy and analytic position
    gradient; MAPLE converts the latter from hartree/bohr to hartree/angstrom
    and does not reinterpret it as a force.

    A clean one-methanol water canary at commit `b9a06f6` returned
    `2.5924801244 kcal/mol`, within \(9.42\times10^{-8}\) kcal/mol of the frozen
    NWChem reference. All 18 Cartesian components agreed with central finite
    differences to \(1.87\times10^{-12}\) hartree/angstrom maximum absolute and
    \(1.35\times10^{-9}\) relative \(L_2\) error. Translation closure was
    \(4.03\times10^{-18}\) hartree/angstrom. Three rigid orientations had a
    \(3.11\times10^{-15}\)-kcal/mol energy span,
    \(2.36\times10^{-16}\)-eV/angstrom maximum gradient-covariance error, and
    \(1.35\times10^{-16}\)-eV maximum torque.

    The first call took `0.181 s`, 36 warmed displaced calls took `0.093 s`
    wall time, warmed orientation calls were about `0.001 s` each, and the
    complete local process took `1.92 s`; the external `/usr/bin/time` process
    envelope reached about `0.92 GiB` peak RSS. This
    is a CDS-only component gate: it includes no MACE-POLAR, continuum
    electrostatics, total-force assembly, public-provider integration, broad
    chemistry, or portable performance claim. The public PCMSolver path still
    uses native `smd_water_cds()` and remains energy-only.
19. `assemble_total_solvation_coordinate_gradient()` now locks the internal
    bookkeeping equation
    \(\mathbf g_{\mathrm{solv}}=\mathbf g_{\mathrm{cont}}+
    \mathbf g_{\mathrm{CDS}}\), including the continuum eV-to-hartree
    conversion and
    \(\mathbf F_{\mathrm{solv,corr}}=-\mathbf g_{\mathrm{solv}}\).
    Component-resolved immutable arrays pass a synthetic whole-energy central
    finite-difference test. The function has no gas-force argument, so
    `CalcABC` remains the sole owner of
    \(\mathbf F_{\mathrm{solution}}=\mathbf F_{\mathrm{gas}}+
    \mathbf F_{\mathrm{solv,corr}}\).

    This is an algebra/interface gate only. It does not verify same-provider
    provenance, run MACE/PySCF, populate `SolvationResult`, or enable public
    forces.
20. A clean methanol canary under the same declared profile at commit
    `2744038` combined real
    float64 MACE-POLAR, order-47 PySCF SWIG/IEFPCM, and official PySCF SMD CDS.
    On the largest total-gradient component (C0-y), the selected
    \(3\times10^{-5}\)-angstrom central difference had continuum, CDS, and
    total absolute errors of \(2.38\times10^{-6}\),
    \(1.67\times10^{-7}\), and \(2.21\times10^{-6}\) eV/angstrom; total
    relative error was \(6.45\times10^{-6}\). The total-gradient translation
    norm was \(1.24\times10^{-14}\) eV/angstrom and torque norm was
    \(5.16\times10^{-4}\) eV. Root, adjoint, energy-identity, component, total,
    translation, and torque gates all passed.

    Step selection is part of the evidence. A \(10^{-3}\)-angstrom displacement
    changed one surviving SWIG parent count and failed closed. At
    \(10^{-4}\) angstrom the topology was stable, but the CDS central-difference
    error was \(1.86\times10^{-6}\) eV/angstrom and missed its predeclared
    \(10^{-6}\) gate. A separate cheap CDS scan showed second-order convergence
    to \(1.67\times10^{-7}\) eV/angstrom at
    \(3\times10^{-5}\) angstrom; the SWIG parent counts were also stable there.
    Both failed attempts remain in the artifact directory.

    The final local process took `48.5 s` and about `3.10 GiB` peak RSS. Its
    base CDS call took `0.0033 s`; the dense order-47 continuum response/root/
    derivative dominated. The clean log contains neither `PCMSolver warning.`
    nor `primary`, and the loader-only PCMSolver correction was detached and
    not retained.

    That canary checked one Cartesian finite-difference component on one
    molecule. At that stage it did not populate `SolvationResult` or enable
    public forces, and it did not certify chemical accuracy or portable speed.
21. The second-molecule rotation gate rejects fixed order 47 and order 53 as
    generally rotation-qualified defaults. For acetone, order 47 used 3435
    surviving points and gave a \(1.1146\times10^{-3}\)-eV base residual
    torque, while one extra orientation gave \(6.60\times10^{-4}\) eV. The
    energy span and maximum gradient-covariance error were
    `0.001513 kcal/mol` and `0.000411 eV/angstrom`, respectively.

    The analytic rigid-rotation directional derivative along the base torque
    was `0.001114604 eV/rad`. A complete self-consistent
    continuum-plus-CDS central difference at the largest topology-stable step,
    \(3\times10^{-5}\) rad, was `0.001116823 eV/rad`. The absolute difference
    was \(2.22\times10^{-6}\) eV/rad (`0.199%` relative), and the CDS
    contribution was \(5.9\times10^{-12}\) eV/rad. This is direct evidence
    that the analytic gradient differentiates the implemented energy and that
    the failed torque gate reflects discrete-energy rotation anisotropy.

    One bounded order-53 follow-up reduced the base torque to
    \(2.94\times10^{-4}\) eV but increased the same rotated-orientation torque
    to \(1.77\times10^{-3}\) eV. Its base surface contained 4244 points,
    required `38.39 s` for the analytic derivative, and peaked at
    `6.23 GiB`, versus 3435 points, `24.79 s`, and `4.74 GiB` at order 47.
    The nonmonotonic orientation result stops the order scan; no order-59 run
    was made. Neither order is adopted, the public provider remains
    energy-only, and Route 2 still has no certified solution-phase PES.

    A valid repair must alter the declared scalar discretization and carry its
    exact derivative. ISWIG retains the same laboratory-frame Lebedev
    construction and is not a fundamental rotation fix. A molecule-following
    grid requires the complete orientation-matrix derivative and explicit
    degeneracy handling; ddPCM/ddCOSMO would be a new same-energy provider.
    Post-hoc zero-force/zero-torque projection is not accepted as a conservative
    force.

    This second molecule adds full-gradient invariance diagnostics and one
    directional whole-energy derivative, not a broad orientation or
    chemical-space certification.
22. The optional `PyDDXPCMReactionFieldLinearMap` now supplies an independent
    multipole-native ddPCM scalar energy, reciprocal reaction map, adjoint, and
    complete coordinate VJP without changing the public PCMSolver path. It is
    imported lazily, fails closed unless pyddx is exactly version 0.8.0, and
    keeps the MACE raw \(l=1\) order while applying the documented
    real-spherical normalization and bohr conversion. Fake-runtime contracts
    and a real pyddx test cover order/units, energy half-coupling, reciprocity,
    a density-component finite difference, and a coordinate finite difference.

    A clean fixed-density methanol artifact at commit `facd956` used
    `lmax=15`, 770 Lebedev points per sphere, \(\eta=0.1\), zero shift, and
    \(10^{-12}\) solver tolerance. It passed with direct MEP error
    \(5.55\times10^{-17}\) hartree/e, energy-identity error
    \(2.28\times10^{-15}\) eV, reciprocity error
    \(1.92\times10^{-13}\) eV, and coordinate finite-difference error
    \(1.04\times10^{-9}\) eV/angstrom. Across three orientations, the energy
    span was `0.000207 kcal/mol`; maximum field and coordinate-gradient
    covariance errors were `0.000621` and `0.000694 eV/angstrom`.

    Base provider construction, reaction-map application, forward energy,
    separate adjoint application, and complete coordinate VJP took `0.216`,
    `0.421`, `0.188`, `0.384`, and `0.902 s` on the local host. The full
    three-orientation plus two-displacement process took `7.32 s`, peaked at
    about `743 MiB` process RSS, and recorded zero warnings. The \(4N\) source
    loop performs no additional continuum solves. This independent adapter
    also has no PCMSolver `primary` branch; a `primary` warning from the public
    route remains a warning-fallback cavity-policy event.

    This closed the fixed-density ddPCM physics backbone. The tested grid was
    not a default, and at this stage the adapter remained absent from the
    public result path; item 26 records the later explicit integration.
23. A clean one-molecule canary at baseline `309366e` connects the optional
    pyddx map to the existing real MACE-POLAR-1-M fixed point, matrix-free
    adjoint, and full continuum coordinate VJP. At `lmax=15`/770,
    `mixing=1.0` converged in 18 iterations to a
    \(6.23\times10^{-13}\) density residual. The continuum correction was
    `-0.3287750411 eV`; the energy-identity error was
    \(1.29\times10^{-14}\) eV, the adjoint relative residual
    \(5.51\times10^{-14}\), and the checked fully reconverged central
    finite-difference gradient error \(1.20\times10^{-6}\) eV/angstrom.
    Translation closure was \(1.32\times10^{-14}\) eV/angstrom.

    The base torque passed at \(9.05\times10^{-4}\) eV, but one extra
    orientation failed at \(1.31\times10^{-3}\) eV. One bounded 1202-point
    follow-up reduced the base/rotated torques to
    \(6.38\times10^{-4}\)/\(3.35\times10^{-4}\) eV and retained a
    \(9.36\times10^{-7}\)-eV/angstrom whole-energy gradient error. Its
    `0.0004776 kcal/mol` energy span and `0.0009010 eV/angstrom` covariance
    error nevertheless lie close to the predeclared `0.0005` and `0.001`
    limits. This passes one discriminator but does not establish a universal
    grid.

    The 770-point density root and analytic derivative took `17.44` and
    `17.67 s`; the 1202-point base took `23.77` and `24.83 s`. At 770 points,
    `mixing=0.5` required 48 iterations and `41.96 s`, versus 18 iterations
    and `17.67 s` for `mixing=1.0`, with converged corrections agreeing within
    \(1.12\times10^{-12}\) eV. All coupled pyddx canaries recorded zero
    PCMSolver/`primary` warnings.

    A tracked fake-runtime regression now composes
    `PyDDXPCMReactionFieldLinearMap`,
    `UnmixedDensityResidualLinearization`, the adjoint solve, and
    `continuum_coupled_solvation_coordinate_gradient()` and compares the
    result with a fully reconverged scalar-energy finite difference. This
    locks the dispatch-independent algebra but does not replace the real-MACE
    evidence.
24. A clean methanol canary at baseline `c63f78f` combines the 1202-point
    pyddx ddPCM/MACE-POLAR continuum gradient with the separately validated
    PySCF 2.13.1 water-SMD CDS energy/gradient pair. On the largest total
    component (C0-y), the predeclared \(3\times10^{-5}\)-angstrom fully
    reconverged central difference had continuum, CDS, and total absolute
    errors of \(6.00\times10^{-6}\), \(1.67\times10^{-7}\), and
    \(6.17\times10^{-6}\) eV/angstrom. The total relative error was
    \(1.80\times10^{-5}\), below but close to the predeclared
    \(2\times10^{-5}\) limit. Translation and torque norms were
    \(1.26\times10^{-14}\) eV/angstrom and \(6.38\times10^{-4}\) eV.

    The density root converged in 18 iterations to
    \(7.12\times10^{-13}\); the adjoint relative residual and
    polarization-energy identity error were \(1.83\times10^{-13}\) and
    \(1.11\times10^{-15}\) eV. Every predeclared root, adjoint, identity,
    assembly, component, total, translation, torque, and warning gate passed.

    The base root, post-root analytic derivative, and two-displacement
    fully-reconverged oracle took `14.82`, `15.73`, and `32.24 s`. The base CDS
    call took only `0.0615 s`, with about `0.0051 s` for both displaced CDS
    calls; the analytic response and two additional ML-SCF roots explain the
    longer gradient-validation command. The structured
    PCMSolver/`primary` warning count was zero. The
    `no_pcmsolver_primary_warning=true` key is a passed gate. The current
    artifact records 73 other Python warnings; earlier same-configuration,
    same-count energy canaries classify them as existing MACE/Torch
    deprecation and SWIG metadata warnings rather than a PCM cavity warning.

    This checks one total-gradient component for one molecule. Its relative
    error is close to the gate, and 1202 points remains a candidate rather
    than a default.
25. A clean acetone total-gradient canary at baseline `aae25a8` repeats the
    1202-point, `mixing=1.0` discriminator on a ten-atom second molecule. It
    first exposed a generic solver-budget bug: SciPy GMRES's default
    20-vector restart and `callback_type=pr_norm` allowed `maxiter=100` to
    mean 100 restart cycles. The corrected policy uses the complete
    39-dimensional neutral Krylov space and makes 100 a true inner-iteration
    limit.

    The original \(10^{-11}\) outer tolerance then failed closed after 100
    iterations at a \(7.76\times10^{-11}\) residual. A dense diagnostic found
    a well-conditioned operator (`cond=1.573`), while a separate probe found
    deterministic repeat differences near \(10^{-17}\) but CUDA-float64
    MACE/full-operator linear-superposition floors of
    \(2.22\times10^{-11}\)/\(5.40\times10^{-11}\). The strict target was
    below the measured operator arithmetic. At an evidence-calibrated
    \(10^{-10}\) outer tolerance, GMRES used 11 callback iterations and 14
    operator applications and reached relative residual
    \(3.65\times10^{-11}\).

    The independent \(3\times10^{-5}\)-angstrom whole-energy central
    difference gave continuum, CDS, and total absolute errors of
    \(6.46\times10^{-6}\), \(9.30\times10^{-10}\), and
    \(6.46\times10^{-6}\) eV/angstrom. The total relative error was
    \(6.97\times10^{-6}\); translation and torque norms were
    \(1.39\times10^{-14}\) eV/angstrom and \(2.38\times10^{-4}\) eV. All
    declared gates passed.

    The base root, analytic derivative, and two-displacement oracle took
    `35.97`, `35.05`, and `64.34 s`, and the full run took `149.24 s`.
    Methanol's matching phases took `14.82`, `15.73`, and `32.24 s`. The
    larger ten-sphere/39-dimensional per-iteration problem explains the
    increase; the CDS call took only `0.068 s`, and the structured
    PCMSolver/`primary` warning count was zero. The field
    `no_pcmsolver_primary_warning=true` is a passed gate.

    This adds a second molecule, not a second total-gradient orientation,
    flexible-geometry conservation, portable speed, or chemical accuracy
    certification.
26. The first public-path methanol force canary selects
    `provider=pyddx,profile=smd-ddpcm-l15-n1202-v1` through the normal parser,
    `SetCalculator`, MACE-POLAR calculator, correction factory, and shared
    result finalizer. It uses pyddx 0.8.0 and PySCF 2.13.1. The public
    correction energy was `-0.22122796846575263 eV`, differing from the prior
    independent same-profile canary by
    \(3.83\times10^{-13}\) eV. The public correction force differed from the
    negative independent total coordinate gradient by at most
    \(4.11\times10^{-11}\) eV/angstrom, and the final gas-plus-correction force
    was finite.

    The ML-SCF root converged in 16 iterations; the adjoint used 8 residual
    callbacks and 10 operator applications and reached relative residual
    \(6.33\times10^{-11}\) under the \(10^{-10}\) gate. Model loading took
    `3.35 s`, the public force evaluation `24.27 s`, and the complete process
    `29.75 s` on this host. The output retained `manifest.json`,
    `route2-ddpcm-result.json`, and `route2-ddpcm-state.npz`.

    The forbidden provider-warning count was zero. This ddPCM path never
    constructs PCMSolver and therefore has no `primary` cavity branch. The 73
    captured Python warnings are the already classified MACE/Torch conversion,
    cuequivariance-availability, and PySCF SWIG deprecation warnings—not a
    PCMSolver cavity warning. This is public wiring and same-profile
    equivalence evidence for one molecule, not chemical-accuracy, portable
    speed, rotation-continuity, or solution-phase-PES certification.
27. Only after the remaining gates pass, enable OPT/scan/TS/MD and call Route 2 a
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
