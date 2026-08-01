# Route-2 implicit-solvation validation status

The public branch contains only official MACE-POLAR-1-M coupled to an explicit
continuum provider. PCMSolver IEFPCM plus MAPLE's native aqueous SMD CDS remains
the default energy-only proof-of-concept. Separately named pyddx ddPCM plus
PySCF SMD CDS profiles retain explicit single-point derivative evidence,
including one multi-solvent parameter profile, but their public result is now
also energy-only. The independent PySCF SWIG/IEFPCM adapter remains private.
One separately versioned fixed-cardinality amplitude-CPCM profile exposes a
bounded water-only conservative force after a per-geometry fail-closed
certificate. Route 2 remains a Research/Innovation Route, not a universal
solution-phase PES or a common stationary MACE--PCM electronic functional.

**2026-08-02 public-boundary correction.** Earlier sections preserve the
historical normal-parser/ASE force-path canaries as derivative evidence. They
do not describe the current API by themselves. PCMSolver, pyddx, PySCF-SWIG,
and the FC-aSWIG `v1`/`v2` identities remain energy-only. Only
`smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-force-v3` advertises
`{"energy", "forces"}`, and it returns a force only when the local root,
adjoint, continuum identity, fixed-topology, conditioning, multi-start, and
profile-level PES gates all pass.

## Bounded FC-aSWIG force-v3 release evidence

The source-bound
`benchmarks/route2-fc-aswig-force-v3-release-evidence-v1.json` closes the
previously open narrow release gates without widening their scope:

- three acetone Cartesian components have a worst central-difference error of
  `5.420641025864159e-6 eV/angstrom`;
- rigid rotation and translation force discrepancies are
  `1.3689226743597745e-14` and `1.7371156695324036e-14 eV/angstrom` in the full
  gate suite;
- the acetone Cartesian path has a worst centered energy/force discrepancy of
  `1.3103270510803168e-5 eV/angstrom` and closed-loop work of
  `-5.480988634707378e-12 eV`, always with 860 surface candidates;
- the three `0.1/0.05/0.025 fs` NVE refinements over the same `0.2 fs` physical
  interval have maximum energy drifts of `8.9913e-7`, `2.2445e-7`, and
  `5.5886e-8 eV`, with fine/coarse ratios `0.2496` and `0.2490`;
- the independent 20-atom 2-acetoxyethyl-acetate torsion shows the registered
  second-order trend and its two-coordinate loop closes to
  `-2.8383503459790538e-12 eV`, with all 1720 candidates retained;
- a clean current-head replay passes the public
  `CommandControl -> SetCalculator -> ASE get_forces()` path and repeats the
  acetone finite-difference/rotation/translation gate.

This evidence admits a conservative derivative of the explicit direct-PCM
operational scalar only. It does not prove variational SCRF, broad chemical or
all-geometry force coverage, long-time MD, multi-solvent forces, portable
speed, or experimental solvation accuracy. Every other profile remains
energy-only.

## Frozen no-training V0 disposition

The preregistered frozen-checkpoint V0 experiment defined one anchored
field-space scalar,

\[
W_0(f)=E_{\mathrm{fc}}(f)-E_{\mathrm{fc}}(0)
       +\langle c_M(f),f\rangle-\nabla_fE_{\mathrm{fc}}(0)^\mathsf T f,
\]

and used its pairing-aware gradient as the candidate response.  A source-bound
one-water CUDA canary at `9913d43` passed the zero-field anchor
(`7.16e-15 e` maximum error), zero-field charge, and scalar-gradient
finite-difference gates.  It failed three preregistered gates:

- the base-field candidate charge is `-5.6093e-4 e`, versus the `1e-8 e`
  ceiling;
- the finite-difference neutral-response antisymmetric/symmetric Frobenius
  ratio is `5.6184e-5`, versus the `1e-5` ceiling;
- the symmetrized neutral response has
  `lambda_max=1.97098e-2 eV`, versus the `1e-6 eV` passivity ceiling.

The protocol therefore rejects this Route-2 V0 construction for the frozen
checkpoint.  No V0 continuum fixed point was solved, no production result was
changed, and the stop rule forbids falling through to V1, post-training, or
fine-tuning.  The immutable preregistration and result are
`benchmarks/route2-v0-scalar-response-water-prereg-v1.json` and
`benchmarks/route2-v0-scalar-response-water-v1.json`.

The retained cSPC/E PSE3 bulk source is separately rejected as a shortcut for
this V0 branch.  Native PSE3 has a closure-matched on-shell 3D-RISM chemical
potential, but requires site-resolved \(u_\alpha,h_\alpha,c_\alpha\) fields;
the checked-in bulk `Cvv` and whole-molecule MACE cluster potential do not
provide them.  It therefore may not be inserted into molecular HNC or added
as a post-stationary correction.  The admission audit also records that raw
published cSPC/E 3D-RISM-PSE3 statistics do not justify the immutable
all-record \(<1.5\) kcal/mol gate without excluded empirical corrections.
This is not a Route-2 accuracy result:
`benchmarks/route2-v0-pse3-native-admission-audit-v1.json`.

A distinct **V0-FD** research kernel is preregistered separately.  It freezes
the zero-field MACE density and minimizes only an energy-conjugate continuum
state; it does not revive the rejected field-conditioned V0 response and has
no chemistry, force, PES, or accuracy result yet.  Its no-fit boundary,
custom-solvent information contract, and blind-test gates are in
`ROUTE2_V0_FD_THEORY.md` and
`benchmarks/route2-v0-fd-multisolvent-prereg-v1.json`.

A separate **V0-Q** fixed-geometry KKT kernel now makes a possible
no-training induced-density functional structurally testable.  It uses a
caller-supplied symmetric gas-phase curvature in the same GTO coefficient/dual
space as the reciprocal PCM operator; it has no default curvature, no
chemistry, force, PES, or accuracy result, and cannot use the rejected learned
MACE response.  Its preregistration and theory boundary are
`ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md` and
`benchmarks/route2-v0-variational-quadratic-prereg-v1.json`.
The first frozen curvature control uses only published Rappé--Goddard QEq
hardness rows and the same MACE Gaussian monopole metric, with all induced
dipoles exactly constrained to zero.  It is a fixed-geometry structural
falsifier, not a full QEq, chemistry, force, PES, or accuracy result.
Its preregistered source-bound acetone execution passes the frozen total-charge,
reciprocal-continuum, KKT residual, stability, passive-response, and
same-basis half-coupling gates without training, fine-tuning, experimental
fitting, response clipping, or cavity selection.  The result deliberately
contains no experimental solvation value or total-solvation score; it validates
only this structural control at one fixed geometry.  Immutable evidence:
`benchmarks/route2-v0-qeq-monopole-acetone-v1.json`.

The corresponding source-frozen gas-phase QM screen is now complete.  At the
same acetone geometry, ωB97M-V/def2-TZVPD finite fields pass the registered
antisymmetry (`4.15e-7`), energy/dipole (`1.00e-4`), field-step consistency
(`2.87e-5`), and positive-eigenvalue checks.  The fixed QEq-monopole
polarizability nevertheless fails all three scientific gates:

- relative Frobenius mismatch `1.4295` (maximum `0.2`);
- trace ratio `1.9768` (allowed `0.8`--`1.2`);
- maximum principal-value relative error `2.2244` (maximum `0.3`).

The registered verdict is therefore
`reject-fixed-qeq-monopole-curvature`.  No hardness, scale, field step, cavity,
MACE weight, or experimental correction may be selected from this result.
This does not invalidate the variational KKT implementation; it means the
first no-training physical curvature is unavailable for chemistry or
multi-solvent scoring.  Immutable evidence:
`benchmarks/route2-v0-qeq-acetone-qm-field-v1.json`.


The V0 liquid controls now also expose a fail-closed dense Hessian diagnostic
for a declared finite configuration grid. It validates the supplied scalar
Hessian in the quadrature pairing and records positive, negative, or
numerically singular curvature without symmetrising or clipping it. This closes
only a controlled mathematical diagnostic: no physical liquid asset, production
grid/orientation stability certificate, force, chemistry score, or accuracy
claim has been admitted.

The weighted-density liquid branch now separately rejects the common but
invalid shortcut “vacuum-limit pressure identity = liquid--gas coexistence.”
Its homogeneous-phase gate evaluates the existing HNC-plus-bridge scalar along
the uniform configuration-density ray, then requires a zero pure-liquid
external potential, full configuration-space stationarity/uniformity, positive
liquid and gas curvatures, and equal grand-potential densities before any
planar profile may be called an interface.  The current v1 bridge certificate
is control-only and now rejects a physical-scope label; a future physical
certificate must carry this evidence plus the nested continuation/planar
protocol.  There are still **zero** physical bridge assets and no
total-solvation, multi-solvent, historical-panel, or accuracy result.

The next mathematical gate is now explicit rather than an informal quartic
fit: on a fixed pure-liquid scalar, the implementation follows the selected
finite-density stable gas branch and solves the same-scalar gas-minus-liquid
coexistence gap \(C(B_s)=0\) for the quartic bridge coefficient.  Its exact
stationary-branch envelope derivative is positive,
\(dC/dB_s=[S(\nu_g)-S(\nu_b)]/V>0\); this does **not** imply a signed
\(d\gamma/dB_s\).  A synthetic translation-invariant control verifies the
coexistence root and the derivative against a branch finite difference.
This is only inner pure-liquid mathematical evidence: it has no physical
solvent source, planar interface, surface-tension result, total-solvation
calculation, force/PES result, or accuracy claim.

The corresponding planar restriction is now implemented as a separate
synthetic control.  It embeds a \((z,\Omega)\) profile into the exact
Cartesian--full-\(SO(3)\) scalar, reduces the gradient by the matched
transverse quadrature average, and uses an equimolar molecular-count constraint
only to follow a periodic two-interface branch.  It rejects a surviving
Lagrange multiplier or transverse nonuniformity, and its local delta-kernel
control returns zero surface excess rather than inventing a positive tension.
This verifies the common-scalar planar algebra only.  It does **not** add a
physical liquid source, a source-bound physical certificate, an outer
surface-tension root, or any FreeSolv/MNSol/accuracy result.

The next nested boundary is now executable at a single declared Gaussian
kernel width through
`route2_v0_molecular_surface_tension_continuation.py`: it recomputes the
same-scalar inner coexistence root \(B_s^*(\sigma)\), freezes it, and only
then obtains the planar surface excess.  That evaluator cannot receive a
surface-tension target.  The companion outer-evidence validator reads the
target only from the frozen bridge asset, preserves all three nested points,
makes no global monotonicity/uniqueness claim, and rejects width endpoints
that are numerically the same discrete kernel.  The checked-in delta-kernel
control exercises this alias rejection; it supplies neither a resolved
physical kernel length nor an outer root.  Thus the physical-liquid,
multi-solvent, force/PES, and accuracy gates remain open.


The exact IAPWS R1-76(2014) ordinary-water surface-tension source is now
content-addressed at
`benchmarks/route2-v0-ordinary-water-surface-tension-iapws-r1-76-2014-v1.json`;
it independently reproduces $0.07199532948823899\,\mathrm{N\,m^{-1}}$ at
298 K from the declared formula.  It is intentionally source-only and not a
cSPC/E RISM state: the cSPC/E model's coincident hydrogen Lennard--Jones sites
prevent ordinary-water IAPWS thermophysics from satisfying the required
model-identity join.  Thus it provides no physical-liquid admission, outer
width/root, chemistry score, or accuracy datum.

The zero-external branch is now also a concrete fail-closed assembly rather
than an informal condition.  `route2_v0_molecular_pure_liquid_external_potential.py`
requires \(u_{
m pure}(\Gamma)=0\) exactly, while
`route2_v0_molecular_pure_liquid_hnc.py` binds an HNC frozen asset, canonical
molecular reference, full Cartesian--\(SO(3)\) quadrature, energy-conjugate
kernel, and compact \(C^2\) site map before exposing a molecular HNC scalar.
The checked-in cSPC/E HNC candidate passes only this source/control construction;
PSE3, source relabelling, and any nonzero external term are rejected.  It is
still not a physical-liquid admission or an accuracy datum because the required
independent state, bridge/planar, force, multi-solvent, and frozen-panel gates
remain open.

The independent fixed-charge molecular-RISM state can now be joined to a
frozen solvent asset only through
`route2_v0_molecular_rism_state_asset_binding.py`.  That join verifies exact
solvent/model/site-model-digest identity, pressure, the three RISM-observable
state fields, and site-density multiplicities before converting the independent
\(p,\gamma,\kappa_T,\rho_m\) anchors to atomic units.  It is explicitly
source-only: the currently checked-in DCM state record fails to join the water
cSPC/E HNC control, and no matching cSPC/E state record exists.  Therefore the
new binding contributes no physical liquid, chemistry score, historical-panel
result, or accuracy claim.

The historical FreeSolv10 evaluator makes the original chemistry-diversity
boundary executable: its ten frozen records contain ten distinct
chemical-function or scaffold classes and are compared per record to locked
experimental values.  Its methane and benzene controls are not actual
functional groups, so a second immutable FreeSolv12 evaluator embeds all ten
historical records (including the ethyl-acetate 7.041442082076966 kcal/mol
outlier) and adds amide plus carboxylic-acid representatives.  The twelve
records contain ten distinct source-labelled actual functional groups; the two
nonfunctional controls are reported but cannot count toward that minimum.  The
two evaluators reject a lower diversity count, missing or extra records,
MAE/RMSE-only reporting, threshold equality, and removal of the historical
outlier.  Consequently, small structural controls and partial numerical
regressions are never reported as accuracy evidence.

The version-locked ORCA/openCOSMO-RS 24a path is now explicitly classified as a
QM reference oracle, not as the Route-2 target. A separate experimental
`mlip_cosmo_rs` bridge can generate the **solute** perfect-conductor screening
surface from fixed AIMNet2 or MACE-POLAR \(l\leq1\) sources without solute QM.
It is out of the BP86/def2-TZVPD-fitted 24a parameterization, may reuse a
QM-derived solvent profile, has not yet closed mutual MACE--conductor
self-consistency, and is not a public calculator or accuracy-certified method.

## Passing engineering gates

- The Route-2 parser is locked to `macepol-m`, a parity-consistent neutral
  singlet fixed conformer, SMD, SCF response, and 1 M to 1 M. Canonical
  element-radius profiles accept XYZ/inline/MOL2 geometry; only GAFF/GAFF2
  radius profiles require MOL2 atom types. Experimental GBSA and gas-phase
  charge APIs remain available outside Route 2. PCMSolver water profiles remain
  water-only; pyddx requires an explicit exact profile, and the named
  `smd-ddpcm-l15-n1202-multisolv-v1` historical profile plus the paired
  direct-ledger `smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2` and
  `smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2` profiles accept the
  11 registered solvents.
- The official MACE-POLAR-1-M checkpoint loads through the upstream cache with
  `mace-torch==0.3.16`; MAPLE changes no learned weight and requires float64.
- The PCMSolver v1.1.12-style C binding, matching Python parser, custom SMD
  radii, cavity-exterior point-multipole MEP, IEFPCM solve,
  `0.5*dot(MEP,ASC)` convention, and reciprocal ASC projection pass
  fake-library and real-water smoke controls.
- Frozen and self-consistent response paths execute with the real
  MACE/PCMSolver stack; the water SCF smoke converges with a nonzero solute
  polarization response.
- The non-default `smd-iefpcm-point-l1-exact-gto-v1` profile keeps the
  historical point \(l\le1\) solute source and energy pairing but supplies the
  complete checkpoint-native receiver tensor to MACE-POLAR. It shares the
  `atomic-center-mean-zero-v1` model-field gauge with the separately versioned
  `smd-iefpcm-point-l1-local-jet-atomic-mean-v1` control; only that pair
  isolates the projector. Analytic point-ASC projection passes an independent
  three-dimensional quadrature check, the affine-field limit reproduces the
  upstream matrix, rotations transform the \(l=1\) channels covariantly, and
  both atomic-mean profiles fail closed in the legacy force path. These are
  representation/engineering gates, not broad accuracy or PES validation. In
  particular, exact GTO retains a point-multipole source but consumes a
  finite-width GTO receiver, so the public profile is explicitly
  source/receiver nonconjugate and energy-only; it is not a variational SCRF
  or common-energy force route.
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
27. A bounded acetone rotation investigation found that the default
    graph_longrange molecular real-space evaluator contributed a
    `0.0018627013 kcal/mol` full-continuum rigid-rotation span, above the
    predeclared `0.0005 kcal/mol` gate. Tuning its Cartesian finite-difference
    offsets was rejected because that changes, rather than removes, the
    \(O(h)\) anisotropy.

    The same official MACE-POLAR-1-M checkpoint was then evaluated through
    graph_longrange 0.4.0's documented forced reciprocal path in fixed
    non-periodic molecular boxes. At 20 Å the intrinsic-correction rotation
    span was `0.0001634123 kcal/mol`; at 40 Å it fell to
    \(6.0106\times10^{-6}\) kcal/mol. The corresponding density-covariance
    error fell from \(3.49\times10^{-5}\) to
    \(1.23\times10^{-6}\). Four isolated MACE evaluations took `0.523 s` in
    the 40 Å reciprocal mode versus `1.307 s` through the default molecular
    evaluator on this host.

    In the complete 40 Å MACE/ddPCM/PySCF functional, two acetone orientations
    had a total-correction span of
    \(8.3038\times10^{-5}\) kcal/mol and a maximum total-gradient covariance
    error of \(3.4356\times10^{-4}\) eV/angstrom. Both 21-iteration roots,
    adjoints, translation/torque closures, and force-assembly reconstructions
    passed. A fully reconverged \(5\times10^{-4}\)-angstrom central difference
    on the largest component gave total, continuum, and CDS errors of
    \(7.50\times10^{-7}\), \(4.69\times10^{-7}\), and
    \(2.82\times10^{-7}\) eV/angstrom. The structured
    PCMSolver/`primary` warning count was zero.

    These gates justify only the explicit, non-default
    `smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1` operator variant. They do
    not establish equivalence to the default evaluator, box convergence for
    every molecule, flexible-geometry continuity, chemical accuracy, or a
    solution-phase PES.
28. The first public-dispatch acetone run of that profile went through
    `CommandControl`, the centralized profile registry, `SetCalculator`, the
    isolated MACE evaluator adapter, the pyddx/PySCF correction provider, and
    the shared result finalizer without a process-local monkeypatch. Its
    correction energy differed from the prior independent canary by less than
    \(10^{-11}\) eV; its analytic correction force differed from the
    independent array by less than \(10^{-10}\) eV/angstrom. The root used 21
    iterations. Model-load and cold public energy-plus-force timings are
    retained as run-level diagnostics in the evidence artifact, not as
    capability gates or portable throughput claims.

    The audit records `pbc=False`, arithmetic-mean centering, a fixed 40 Å
    cubic box, `use_pbc_evaluator=True`, graph_longrange 0.4.0, and the narrow
    float32-to-float64 reciprocal-field projection bridge. The only visible
    root logger message is MACE's intentional checkpoint conversion to
    float64; no PCMSolver or `primary` warning occurred. A passing public
    artifact must record its Git commit, a clean tracked working tree, no dirty
    development override, and separate warning categories.

    The corrected evaluator now owns an explicit coordinate-transform
    derivative contract. Synthetic non-translation-invariant scalar, force,
    density-pairing, and Hessian oracles verify the centering VJP and the
    double-sided Hessian pullback against finite differences or direct matrix
    projection. Earlier fixed-box derivative artifacts remain bound to their
    execution commits and must not be used to certify the corrected pullback.
29. The frozen bounded QM-fidelity pilot
    `benchmarks/route2-qm-fidelity-v1.json` compares three identical fixed
    geometries against self-consistent PySCF 2.13.1 SMD/water at
    omegaB97M-V/def2-TZVPD. Methanol, acetone, and 2-acetoxyethyl acetate have
    absolute Route-2/QM differences of `0.6934`, `0.2918`, and
    `0.7151 kcal/mol`, respectively. The three-record Route-2/QM MAE is
    `0.5668 kcal/mol`; the Route-2/experiment MAE is `0.8388 kcal/mol`.
    Methanol uses the base `smd-ddpcm-l15-n1202-v1` profile; the other two use
    `smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1`. These are bounded
    cross-profile pilot statistics, not a single-profile or chemical-space
    confirmation.

    The frozen source records were generated at `278c312` for methanol/acetone
    and `5422a07` for the fixed 2-acetoxyethyl-acetate conformer. Clean
    detached-worktree energy canaries at `5746f24`, `f3e9892`, and `e34abc5`
    reproduce their historical corrections within \(4.55\times10^{-13}\),
    \(2.50\times10^{-16}\), and \(2.22\times10^{-16}\) eV. No tracked `maple/`
    runtime source changed among those heads. They use 16, 21, and 18 root
    iterations and emit no `primary` or solver warning. Their observed
    energy-evaluation times are `8.85`, `24.86`, and `71.91 s`, compared with
    historical `10.57`, `32.50`, and `85.55 s`; this is not a
    hardware-normalized or replicated speed result. The flexible panel,
    remaining multi-step torsion/loop and second-molecule force evidence, and
    historical timing ratios remain unaligned with the latest checked checkout
    rather than being treated as current merely through ancestry. Item 30
    records the separately rerun source force, Cartesian pair, and one-step
    central torsion.

    Component-resolved comparison exposes error cancellation. Acetone has
    Route-2-minus-QM differences of `-1.9279 kcal/mol` in
    \(\Delta E_{\mathrm{solute}}\) and `+2.2197 kcal/mol` in
    \(U_{\mathrm{pol}}\), leaving a total difference of only
    `+0.2918 kcal/mol`. The fixed 2-acetoxyethyl-acetate conformer similarly
    combines `-1.4801` and `+2.1951 kcal/mol` component differences into a
    `+0.7151 kcal/mol` total. The shared PySCF SMD CDS value is identical in
    both columns. Total-energy agreement therefore does not yet certify either
    polarization component independently.

    A locked four-geometry electronic panel for 2-acetoxyethyl acetate gives
    per-conformer Route-2/QM MAE `0.8787 kcal/mol`, maximum absolute error
    `1.0082 kcal/mol`, and bounded Route-2/QM ensemble values of `-6.5165` and
    `-7.4506 kcal/mol`. Their difference is `0.9341 kcal/mol`. Route 2 is
    arithmetically `0.1765 kcal/mol` from experiment for this one bounded panel,
    but that is not an HFE-accuracy estimate. The panel contains one original
    source geometry and three ETKDGv3/MMFF-generated screened candidates; Route
    2 or QM supplied every reported gas/aqueous energy and Boltzmann weight.
    Vibrational thermochemistry, basin degeneracy, broad sampling, solution
    relaxation, and training-set exclusion remain absent.

    Observed QM/Route-2 energy wall-time ratios are `3.05`, `5.56`, and `9.04`
    for the three fixed systems, but they compare Route-2 CUDA with PySCF
    eight-thread CPU and are not hardware-normalized speedups. A proposed
    same-cavity/ddX-settings AO-density reference was rejected for acetone. One
    omegaB97M-V/def2-SVP calculation numerically completed with the physically
    invalid `-114.1610 hartree` ddPCM correction, while the diffuse-basis
    calculation collapsed nonphysically. This does not establish that all
    AO-density sharp-cavity PCM formulations fail. Production Route-2 radii
    were not changed to rescue this invalid construction; PySCF SMD/IEFPCM
    remains a chemical-level, not operator-identical, comparison.
30. A clean source-geometry analytic-force canary executed for
    2-acetoxyethyl acetate at `d72dfba` reproduces the historical correction
    energy exactly and every correction-force component within
    \(2.66\times10^{-14}\) eV/angstrom. The 18-iteration root and
    11-application adjoint converge to a \(6.49\times10^{-11}\) relative
    residual, below the configured \(10^{-10}\) tolerance. PCMSolver and
    legacy `primary` warning counts are zero. The observed force evaluation
    took `160.91 s`; it is one run, not a performance claim.

    The fully reconverged \(5\times10^{-4}\)-angstrom displacement pair was
    then rerun against the same `d72dfba` runtime. Both displaced energies
    converged in 18 root iterations and emitted zero PCMSolver or legacy
    `primary` warnings. The central-difference force is
    `0.7570030893060409 eV/angstrom`, versus the same-execution-head analytic
    `0.7570000052313698 eV/angstrom`; the absolute and relative errors are
    \(3.0841\times10^{-6}\) eV/angstrom and \(4.0741\times10^{-6}\).
    The negative and positive energy evaluations took `70.56` and `74.21 s`,
    with `150.78 s` total process wall time; this is not a performance claim.
    Agreement with the historical `5422a07` finite-difference force is
    \(1.11\times10^{-13}\) eV/angstrom.

    One pre-registered central C--C torsion pair at \(\pm0.5^\circ\) was also
    rerun against `d72dfba`. The analytic generalized force is
    `0.04520643472551838 eV/rad`; correction energies
    `-0.33949480476764604` and `-0.340282042716652 eV` give the
    central-difference value `0.04510541195057784 eV/rad`. The absolute and
    relative errors are \(1.01023\times10^{-4}\) eV/rad and
    \(2.2347\times10^{-3}\), inside the predeclared
    \(2\times10^{-4}\) eV/rad and \(5\times10^{-3}\) limits. Both
    energy-only points converge in 18 root iterations and emit zero PCMSolver
    or legacy `primary` warnings. Their `147.73 s` total contended wall time is
    not a performance claim.

    The original runner wrote both immutable audit/manifest pairs before
    exiting on a NumPy-Boolean JSON-serialization error. A read-only finalizer
    recomputed the scalar gates from those pairs, so the two scientific energy
    evaluations were not rerun. The analytic and finite-difference generalized
    forces reproduce the historical record within
    \(1.39\times10^{-15}\) and \(3.18\times10^{-14}\) eV/rad,
    respectively.

    A separately pre-registered \(\pm1.0^\circ\) pair on the same locked
    coordinate was then run against `d72dfba`. Correction energies
    `-0.3390965192574748` and `-0.3406688755535749 eV` give
    `0.04504468982867865 eV/rad`. Its absolute and relative errors are
    \(1.61745\times10^{-4}\) eV/rad and \(3.5779\times10^{-3}\), also
    inside the locked gates. Both roots converge in 18 iterations with zero
    PCMSolver or legacy `primary` warnings. Refining from \(1.0^\circ\) to
    \(0.5^\circ\) lowers the absolute error by
    \(6.07221\times10^{-5}\) eV/rad, to `0.62458` of the coarse-step
    error. The current \(1.0^\circ\) finite difference reproduces the
    historical oracle within \(9.54\times10^{-15}\) eV/rad. Its `151.35 s`
    wall time is not a performance claim. These two current steps establish a
    bounded refinement trend, not an asymptotic convergence order.

    The third pre-registered pair at \(\pm0.25^\circ\) is now complete.
    Correction energies `-0.33969248904996924` and
    `-0.3400866505022192 eV` give
    `0.04516757532134049 eV/rad`, only
    \(3.88594\times10^{-5}\) eV/rad (\(0.0860\%\)) from the analytic
    generalized force. Both energy-only roots converge in 18 iterations;
    formula, geometry, PCMSolver-warning, and legacy-`primary` gates all pass.
    Their `148.50 s` outer wall time is not a performance claim.

    The pre-registered asymptotic-order validation nevertheless has
    `status=fail`. The finite-difference drift increases from
    \(6.07221\times10^{-5}\) to
    \(6.21634\times10^{-5}\) eV/rad after the second halving, producing
    observed order `-0.03384` rather than the locked `1.5--2.5` range. The
    runner's exit 1 records these failed gates after both immutable results
    were written; it is not an execution or serialization failure. Thresholds
    were not changed and the points were not rerun.

    Read-only component decomposition gives orders `2.008` for gas MACE and
    `2.001` for CDS, versus `0.544` for solvent-intrinsic MACE, `0.535` for
    \(\Delta E_{\mathrm{solute}}\), and `0.212` for
    \(U_{\mathrm{pol}}\). The failure is therefore localized to the
    self-consistent electrostatic coupling block, not gas MACE or CDS.

    A separately pre-registered root-cause diagnostic subsequently reused all
    six immutable torsion states and performed exactly one independent
    \(+0.25^\circ\) energy-only root. It introduced zero new geometries, zero
    finite-difference steps, and zero force evaluations. The independent root
    reproduces the stored correction energy to
    \(2.22\times10^{-16}\) eV and density to
    \(2.22\times10^{-16}\) electron, while the reaction potential and gradient agree to
    \(1.89\times10^{-15}\) eV and \(7.22\times10^{-16}\) eV/angstrom.
    All 18 locked validity gates pass; PCMSolver and legacy `primary` warning
    counts are zero.

    Canonical sphere/PySCF-1202-Lebedev witnesses show 19 changed active points
    on the minus fine interval and 27 on the plus interval. Frozen density
    contributes `73.8%` of the \(0.5^\circ\to0.25^\circ\) fine-drift L1 norm.
    Inside that term, fixed-center-density PCM contributes `67.6%` and
    reaction-map-through-MACE contributes `31.9%`; fixed-center-field
    MACE-minus-gas and CDS retain approximately second-order behavior.
    Consequently, the valid label is
    `active-set-associated-explicit-continuum-geometry-response`.

    This is a bounded association, not proof that active-set change is the
    sole cause. pyddx 0.8.0 does not expose a provider-consistent operator-only
    versus cavity-only split, so that finer claim remains prohibited. The
    diagnostic took `192.76 s` outer wall time and is explicitly not a speed
    claim. Repeated Torch/JIT/SWIG deprecation messages and one recorded
    float32-to-float64 model conversion are non-gating dependency messages;
    the result must not be summarized as having no warnings at all.

    The predeclared three-step validation for a second C--O torsion retains
    `status=fail`: its coarse-to-middle absolute error and step drift were not
    monotonic. A separate `0.125`-degree extension passes with
    \(5.68\times10^{-7}\) eV/rad absolute error, but explicitly cannot
    retroactively overwrite that failed source gate.

    An independently reconstructed local two-torsion rectangle nevertheless
    passes its scalar-energy/analytic-force consistency gates. The oriented
    loop force work is \(5.10\times10^{-8}\) eV, the largest edge
    energy/work residual is \(8.70\times10^{-8}\) eV, and the mixed generalized
    derivative mismatch is \(2.68\times10^{-3}\) eV/rad2. Every point reused
    identical cavity radii, converged its root/adjoint, and closed translation.
    A second functionally distinct flexible C/H/O molecule,
    2-propoxyethanol, also passes one center-point force/root/translation/
    rotation audit with two resolved torsional generalized forces.

    The source-geometry analytic force, one Cartesian finite-difference
    displacement pair, and one central C--C torsion at three step sizes are
    current-runtime aligned. The third pair is valid, but its pre-registered
    smooth second-order asymptotic test failed. The bounded two-dimensional
    loop and second-molecule center point remain historical. These local
    derivative checks still do not establish global flexible-geometry
    continuity, relaxed-PES behavior, QM-force fidelity, or short-NVE energy
    conservation.
31. A one-shot fixed-density acetone scalar canary at commit `da89ab8`
    evaluates the unchanged pyddx ddPCM problem in the nuclear-charge
    molecule-following frame of Johnson, Gill, and Pople. The immutable
    lock/attempt/result chain contains exactly three laboratory-frame controls
    and three molecule-frame candidates with no MACE, ML--SCF root, CDS, force,
    finite difference, retry, or production-profile change.

    All frozen gates pass. The laboratory-frame span is
    `7.1672e-6 eV` (`1.6528e-4 kcal/mol`), while the molecule-frame span is
    `2.1094e-15 eV` (`4.8644e-14 kcal/mol`). All three candidates retain 4993
    unique active sphere/Lebedev pairs and the same mapping hash. The
    identity-profile shift is `1.4473e-4 kcal/mol`, and the largest
    half-coupling identity error is `2.2760e-14 eV`. The tracked result is
    `benchmarks/route2-ddpcm-ri-jgp94-acetone-v1.json`.

    This proves only rigid-rotation covariance for one fixed density in a
    nondegenerate local frame domain. It does not validate the analytic frame
    VJP, torsional continuity, self-consistent MACE--ddPCM force, CDS
    derivative, chemical accuracy, or a public provider. A separate
    pre-registration must differentiate the center, orientation, body-frame
    positions/dipoles, source/back projection, and reaction map. Near
    eigenvalue degeneracy, axis/gauge switching, or unstable active mapping
    remains fail-closed.
32. The separately pre-registered analytic frame-VJP stage stops at its
    zero-continuum-solve preflight. Its pure NumPy oracle checks all 30 acetone
    coordinate and 30 dipole components. The three coordinate-step maximum
    errors are `9.3481e-8`, `2.3364e-8`, and `5.7252e-9`; the halving ratios are
    `0.24994` and `0.24504`; the largest dipole error is `7.4937e-12`; and
    translation closes to `1.1103e-15`. The frame response is nontrivial, with
    orientation-contribution norm `6.9426`.

    The eight prescribed pyddx cavity constructions nevertheless produce four
    active-set signatures and 4992/4993 active pairs. Two frozen gates fail:
    exact active mapping and pair count relative to the passed base scalar.
    Every displaced mapping still contains unique sphere/Lebedev pairs. No
    continuum state, MACE call, ML--SCF root, CDS term, finite-difference
    energy, lock, or attempt was evaluated. The exact preflight is tracked as
    `benchmarks/route2-jgp94-frame-vjp-preflight-v1.json`.

    This rejects the inference that the JGP94 molecule-following frame alone
    makes the current pyddx cavity a smooth geometry-dependent provider. It
    preserves the scalar rigid-rotation pass and the pure frame-VJP algebra,
    but blocks production integration and forbids replacing the prescribed
    components/steps with a post-hoc stable subset.
33. Only after a separately pre-registered smooth-cavity provider and the
    remaining end-to-end gates pass may Route 2 enable OPT/scan/TS/MD or be
    called a solution-phase PES.
34. A separately pre-registered PySCF 2.13.1 fixed-density discriminator
    evaluated exactly 12 continuum scalar cases: SWIG and ISWIG for methanol
    and acetone at three locked rigid orientations, order 47, with unchanged
    geometries, density archives, radii, dielectric, and point-multipole
    coupling. It loaded no MACE model, performed no ML--SCF root, evaluated no
    CDS term or force, and changed no production provider.

    ISWIG failed three locked gates. Its rotation span was
    `0.00076871 kcal/mol` for methanol and `0.00047738 kcal/mol` for acetone,
    versus SWIG controls of `0.00045599` and `0.00037786 kcal/mol`. The ISWIG
    to SWIG ratios were therefore `1.6858` and `1.2634`, rather than the
    required maximum `0.75`. ISWIG also changed surviving surface parent
    counts across the three orientations for both molecules. Every scalar was
    finite. The largest ISWIG half-coupling/reciprocity residual remained
    below `2.43e-17 hartree`; the largest residual across all 12 SWIG and
    ISWIG cases was `5.21e-17 hartree`.

    This rejects ISWIG as the selected next Route-2 provider candidate without
    post-result tuning. It does not show that ISWIG is generally inferior as a
    PCM discretization; it shows only that changing the switching function did
    not repair this locked finite-grid rotation/active-set discriminator. The
    frozen result is
    `benchmarks/route2-pyscf-iswig-discriminator-v1.json`. The exact executed
    runner bytes are tracked as
    `benchmarks/run_route2_pyscf_iswig_discriminator.py`; the frozen result
    records both its SHA256 and the actual `smd-iefpcm-gaff2-o` radius selector
    used by that runner.
35. The separately registered four-molecule same-geometry accuracy diagnostic
    was not executed. Independent scientific review found that its per-molecule
    Route-1 geometry/source record, existing QM artifact, and FreeSolv
    dataset/uncertainty provenance were not complete enough to audit the
    proposed gates. No Route-1 recalculation, Route-2 single point, or new QM
    reference was run; no threshold, molecule, or profile was changed after
    registration. The immutable disposition is
    `benchmarks/route2-four-molecule-accuracy-prereg-v1-disposition.json`.
    Any replacement must be newly pre-registered with a complete evidence
    chain and a staged budget rather than editing the rejected protocol.
36. The versioned
    `smd-ddpcm-l15-n1202-multisolv-v1` capability profile separates
    electrostatics, source representation, reaction-field projection,
    nonpolar model, and solvent parameters in immutable profile/solvent
    records. It uses PySCF 2.13.1 solvent descriptors, including the
    solvent-acidity-dependent SMD oxygen radius and the tested P/S/Cl mapping
    (`2.12/2.49/2.38 angstrom`). Current water profiles now use that same
    corrected atomic-number-indexed mapping; artifacts generated with the old
    shifted mapping remain commit-bound and are stale for the corrected method.

    Registry immutability, aliases, provider/profile gating, dielectric/CDS
    routing, corrected water-profile radii, and exact descriptor/radius
    agreement with the tested PySCF runtime are covered by executable tests.
    At the time this registry stage landed, no immutable multi-solvent
    chemical-accuracy artifact had been frozen.  This stage itself therefore
    establishes a capability boundary only.  Later response-bound diagnostics
    are listed below; both fail their all-record accuracy gate and do not turn
    this capability stage into a fitted result or PES gate.
37. A real MACE-POLAR-1-M checkpoint canary first exposed a model-interface
    gauge defect: adding a constant `0.25 eV` potential to neutral acetone
    changed the intrinsic energy by `3.5444e-4 eV` and the largest density
    coefficient by `3.1752e-4 e`, for both raw local-jet and raw exact-GTO
    inputs. The pre-registered `atomic-center-mean-zero-v1` repair subtracts
    one common mean point-ASC potential from the model-driving scalar channel
    only. Under the same constant-field canary, gas-versus-gauge-fixed energy
    shifts became exactly zero and the largest density shift was
    `2.22e-16 e`; local-jet and exact GTO also agreed to numerical precision.

    One separately pre-registered fixed-conformer acetone calculation then
    compared only the matched-gauge profiles. Both used the same 516-point
    cavity hash, point-\(l<=1\) source, IEFPCM equation, SMD radii, CDS term,
    SCF policy, and zero-at-infinity density-dual energy field. The
    atomic-mean local-jet control predicted `-6.6930 kcal/mol` versus the
    FreeSolv `-3.80 +/- 0.60 kcal/mol` record, an absolute error of
    `2.8930 kcal/mol`; its component ledger was `+1.1277` solute response,
    `-11.3919` PCM polarization, and `+3.5712 kcal/mol` CDS. Exact GTO
    predicted `-4.6836 kcal/mol`, an absolute error of `0.8836 kcal/mol`,
    from `+0.0786`, `-8.3333`, and `+3.5712 kcal/mol`, respectively. The
    exact-minus-local shift was `+2.0094 kcal/mol`.

    Local-jet and exact GTO converged in 18 and 13 iterations with density
    residuals `6.42e-6` and `5.28e-6 e`; their half-coupling identity errors
    were below `9.45e-17 eV`. On this host, PCMSolver initialization/response
    times were `0.146/2.460 s` and `0.145/1.603 s`. The second model load was
    cache-warm, so neither those timings nor the apparent iteration advantage
    is a portable speed claim. Native PCMSolver warning count was zero; each
    run retained one non-selection-fatal PEDRA poor-tessellation warning.

    The exact profile now fails closed unless the runtime is
    `graph-longrange==0.4.0`; the immutable result audit also fingerprints the
    ordered live-checkpoint projection matrix. This closes a feature-layout
    provenance boundary, not an accuracy or PES gate.

    The complete official MACE-POLAR-1-M checkpoint is independently bound by
    identifier, release URL, `68,133,235`-byte size, and SHA-256 before model
    loading; a matrix fingerprint alone is not treated as learned-weight
    provenance.

    The lower total error is one development-molecule observation, not an
    accuracy gate or proof of improved physics: MACE-POLAR training overlap is
    unknown, the component changes may compensate, and no force, gauge
    coordinate derivative, Gaussian solute source, variational diagnostic,
    smooth cavity, or second continuum equation was evaluated. The immutable
    evidence summary is
    `benchmarks/route2-pcmsolver-exact-gto-acetone-v1.json`.
38. A source-bound one-water canary now tests the stronger thermodynamic
    identities requested after the initial
    `dE_intrinsic/df != returned_density` probe. The public local-jet
    MACE-POLAR/ddPCM root converged in 9 iterations with
    `1.8633e-12 e` unmixed residual and `8.16e-15 eV` half-coupling error.
    Three JVP/VJP implementation dot tests closed within `2.11e-12 eV`.

    With that transpose check passing, the maximum block-normalized
    energy-density defect is `0.6641` for the intrinsic-energy identity and
    `0.9995` for the already-coupled-energy identity. Three physical
    reciprocity discrepancies span `0.00288--0.00465 eV`. The symmetric
    12-dimensional susceptibility discriminator contains 3 positive modes
    under the documented nonpositive stability convention, and the
    antisymmetric norm is `0.01715 eV`. A two-step field-loop refinement gives
    nearly constant step-squared-normalized work (`-0.00416103` and
    `-0.00416091 eV`).

    This one-state result sets no fitted pass threshold and does not prove a
    universal theorem about MACE-POLAR. It is sufficient to reject a claim
    that either tested exact conjugacy identity is established for the current
    local-jet Route-2 coupling. The implemented scalar remains differentiable
    through the fixed-point adjoint, but it must be described as a
    self-consistent differentiable surrogate rather than a demonstrated joint
    variational free-energy functional. Exact-GTO response, chemical-space
    prevalence, smooth coordinates, and PES/NVE gates remain open. The frozen
    evidence is
    `benchmarks/route2-mace-local-field-thermodynamic-canary-v1.json`.
39. A separate source-bound exact-GTO acetone canary closes the
    fixed-geometry derivative algebra without changing the public energy or
    force path. The public PCMSolver/IEFPCM root converged in 13 iterations
    with a stored unmixed density residual of `5.2759e-6 e`; reopening the
    identical 516-point cavity reproduced both the reaction field and the
    complete model-native feature tensor exactly.

    The continuum feature transpose, learned feature-to-density transpose, and
    composed unmixed-residual transpose close their bilinear identities to
    `5.00e-16`, `4.09e-13`, and `2.46e-12`, respectively. Matrix-free GMRES
    reaches `1.96e-9` relative residual in five callbacks and seven operator
    applications. The physical density-space right-hand side agrees with a
    neutral-direction central difference to `1.75e-7` relative error at the
    best tested coefficient step, `1e-3`; smaller steps are less accurate, so
    this result is not presented as an asymptotic finite-difference order.

    The public hydration result, `-4.6836 kcal/mol` versus the development
    record `-3.80 +/- 0.60 kcal/mol`, is retained only as provenance
    (`0.8836 kcal/mol` absolute error), not as an accuracy gate. Both the
    public and reopened calculations emitted zero native `PCMSolver warning.`
    markers and each retained one non-selection-fatal PEDRA
    poor-tessellation warning. The local run took `8.49 s` internally
    (`10.50 s` process wall time); this is host metadata, not a portable speed
    claim.

    This evidence proves only fixed-geometry exact-GTO derivative
    implementation closure. It supplies no coordinate or gauge derivative,
    variational/thermodynamic identity, hydration certification, smooth PES,
    OPT, TS, scan, MD, or NVE gate. The frozen evidence is
    `benchmarks/route2-mace-exact-gto-fixed-geometry-canary-v1.json`.
40. The pyddx reaction map now exposes ddPCM and ddCOSMO as one orthogonal
    equation axis while preserving the historical ddPCM wrapper. For pyddx
    0.8.0, MAPLE applies the required \((\epsilon-1)/\epsilon\) COSMO factor
    to the scalar energy, reaction field, adjoint, and coordinate VJP
    together. Contract tests cover all four quantities, and a real-runtime
    high-dielectric limit makes scaled ddCOSMO converge to ddPCM.

    A source-bound two-molecule fixed-AIMNet2 canary then changed only the
    continuum equation. At water dielectric `78.39`, water gives
    `-6.6851/-6.7223 kcal/mol` for ddPCM/ddCOSMO, and acetone gives
    `-4.9187/-4.9472 kcal/mol`; COSMO-minus-PCM is therefore `-0.0373` and
    `-0.0285 kcal/mol`. All four half-coupling errors are below
    `8.59e-13 eV`.

    The one-process internal timings are retained only as host/order metadata,
    not a speed ranking. This fixed-charge, no-CDS result is not a total
    solvation free energy, experimental accuracy test, self-consistent
    MACE/ddCOSMO result, C-PCM implementation, COSMO-RS calculation, force
    gate, or PES claim. The immutable evidence is
    `benchmarks/route2-aimnet2-ddpcm-ddcosmo-equation-canary-v1.json`.
41. A provider-independent MNSol-v2012 protocol now verifies the pinned
    3037-row table and all 790 fixed M06-2X/MG3S gas-phase geometries without
    redistributing the database. Its initial neutral, absolute,
    single-component Route-2 scope contains 653 rows across ten experimentally
    populated solvents. A separate immutable manifest selected exactly one
    at-most-20-atom, distinct-geometry row per solvent before reading any
    model output or using experimental values for selection; eight selected
    rows are confirmation records and two are deterministic development
    fallbacks.

    On that ten-record pilot, fixed AIMNet2 NQE point charges plus SMD-CDS
    give an MAE/RMSE of `1.1431/1.3516 kcal/mol` with ddPCM and
    `1.0121/1.2743 kcal/mol` with scaled ddCOSMO. Mean signed errors are
    `+1.0333` and `+0.8300 kcal/mol`, respectively. ddCOSMO has the lower
    absolute error on 8/10 rows and is on average `0.2033 kcal/mol` more
    negative. Maximum half-coupling errors remain below `3.12e-14 eV`; the
    maximum raw AIMNet2 total-charge residue is `1.49e-7 e`, and the largest
    projected final charge sum is `1.11e-16 e`.

    The source-bound clean run took `28.92 s` internally for all ten rows and
    both equations. Fixed-order method timings are retained only as metadata,
    not a speed ranking. The public artifact contains aggregate values only;
    row-level experiments, predictions, components, names, formulas,
    coordinates, and charges remain in the ignored local `.omx` artifact.

    This one-row-per-solvent diagnostic does not certify multi-solvent
    accuracy or solvent generalization. It is not self-consistent AIMNet2
    polarization, MACE-POLAR validation, C-PCM, COSMO-RS, a force gate, or a
    solution-phase PES result. Complete development/confirmation evaluation
    remains required. The immutable aggregate evidence is
    `benchmarks/route2-mnsol-aimnet2-multisolvent-pilot-v1.json`.
42. The PCMSolver route now has two explicit, non-default intrinsic-cavity
    profiles. Both replace the built-in water-solvent probe with a runtime
    verified SMD electrostatic cavity: zero probe radius, no added spheres,
    explicit `78.355` dielectric, and the canonical SMD Coulomb radii. One
    retains local jet; the other uses the checkpoint-native exact receiver-GTO
    projection. Legacy profiles and the global default are unchanged.

    A selection manifest was frozen before evaluating ten neutral FreeSolv
    v0.52 development representatives spanning ten functional-group classes.
    Experimental values and uncertainties were verified row by row against the
    pinned upstream database SHA-256
    `2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260`.
    Fixed \(l<=1\), local-jet SCF, and exact-GTO SCF give
    MAE/RMSE/max errors of `1.084/1.294/2.647`,
    `1.752/2.516/5.744`, and `0.915/1.305/3.247 kcal/mol`,
    respectively. Exact GTO wins 7/10 paired comparisons and changes the
    acetone error from `2.963` to `0.933 kcal/mol`.

    This is not a universal improvement. Relative to local jet, exact GTO
    worsens methanol by `0.761`, phenol by `1.295`, and chloroethane by
    `0.463 kcal/mol`; acetic acid remains the largest exact-GTO error at
    `3.247 kcal/mol`. Therefore the pre-registered `<1 kcal/mol` panel-MAE,
    no-worse-than-control, acetone, and zero-native-warning gates pass, but the
    `<2 kcal/mol` maximum-error gate fails and the overall promotion decision
    is **fail**. The exact-GTO candidate remains explicit, non-default, and
    energy-only. Its single-run mean wall time was `1.599 s` versus `2.130 s`
    for local jet and `0.420 s` for fixed \(l<=1\); these are host diagnostics,
    not portable speed rankings.

    The artifact is
    `benchmarks/route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json`.
    This development panel does not certify confirmation accuracy,
    training-set exclusion, original-SMD equivalence, variational
    thermodynamics, forces, or a smooth solution-phase PES.
43. The optional PySCF SWIG response layer now exposes separately named
    IEFPCM, C-PCM, and COSMO research operators without changing any public
    profile. The implementation is hard-gated to PySCF 2.13.1 and follows the
    upstream `PCM.build()` branches exactly: C-PCM uses
    \(f_\epsilon=(\epsilon-1)/\epsilon\), COSMO uses
    \(f_\epsilon=(\epsilon-1)/(\epsilon+1/2)\), and IEFPCM retains its
    nonsymmetric \(K/R\) equation with an energy-conjugate response. Synthetic
    direct/adjoint/energy/gradient tests pass for all three.

    A real water-molecule comparison against PySCF 2.13.1 produced an
    identical SWIG surface, \(K\), and \(f_\epsilon\) for all three methods;
    the largest \(R\) mismatch was \(2.78\times10^{-17}\). This is an operator
    implementation result, not a chemical-accuracy result or a public Route-2
    C-PCM profile.
44. A two-record private MNSol water screen then held frozen gas-phase
    MACE-POLAR \(l\leq1\) multipoles and SMD-CDS fixed while changing the
    continuum equation. PySCF SWIG IEFPCM, C-PCM, and COSMO give two-record
    MAEs of `2.3352`, `2.2838`, and `2.3516 kcal/mol`; pyddx ddPCM and scaled
    ddCOSMO give `2.2756` and `2.2212 kcal/mol`. The largest within-backend
    prediction shift is only `0.0635 kcal/mol`, all half-coupling errors are
    below `1.9e-14 eV`, and the three PySCF arms use identical points and
    areas. This screen alone was too small and underpowered to justify an
    equation ranking. Cross-backend values are not a pure equation comparison
    because their cavity discretizations differ. A later separately
    preregistered fixed-source panel addressed the broader ten-solvent
    comparison without changing this historical screen.

    The same stage adds a fail-closed COSMO-RS evidence boundary. It hashes the
    three ORCA/openCOSMO-RS 24a inputs, locks the audited
    BP86/def2-TZVPD/298.15 K/surface contract, and verifies the temperature and
    Hartree/kcal consistency of one external `dGsolv` result. COSMO-RS remains
    outside the Route-2 profile registry and `method=cosmo-rs` is rejected as
    a sigma-profile/statistical-thermodynamics workflow rather than a
    continuum-equation switch. A later local serial ORCA 6.1.0-f.0 runtime
    smoke reproduced the complete three-QC-job/openCOSMO-RS workflow for
    fixed-geometry acetone in water and returned `-4.176542 kcal/mol`. The
    MPI attempt failed in a child process even though the host ORCA process
    returned zero, so the new external runner is locked to one process and
    checks the main output plus every child `.lastout`. This runtime smoke is
    not an MNSol chemical-accuracy or generalization result.
45. A matrix-free feedback-gain diagnostic now reuses the unmixed neutral
    residual JVP/VJP to estimate the root-local
    \(\sigma_{\max}(\Pi_0J_{\mathcal M}J_{\mathcal P}\Pi_0)\) without
    materializing every feedback column. Two anonymous fixed neutral states
    with dimensions 31 and 75 match their dense small-system oracles within
    `1.02e-10` and `5.41e-11`; their singular-triplet relative residuals are
    below `3.47e-10`. The matrix-free calls used 13/12 and 23/22 JVP/VJP
    applications and took `35.74` and `269.61 s`, versus `39.49` and
    `447.10 s` for the complete dense oracles on the same host.

    Both root-local estimates are below one, but they are iterative values, not
    certified upper bounds over a complete invariant neighborhood. They do not
    prove Banach contractivity, nonlinear uniqueness, variational
    thermodynamics, passivity, or chemical accuracy. The diagnostic therefore
    remains research-only and is not called by default SCF or broad panels.
    The execution predated the clean implementation commit: its exact source
    hashes match landed commit `f833d8e`, but the private summaries did not
    retain a checkpoint hash. The public anonymous evidence is
    `benchmarks/route2-matrix-free-feedback-gain-two-state-v1.json`.
46. A separate preregistration froze the same ten MNSol records, their exact
    AIMNet2 monopoles and MACE-POLAR \(l\leq1\) coefficients, the
    solvent-specific SMD radii/dielectric/CDS values, PySCF 2.13.1, SWIG order
    17, and the IEFPCM/C-PCM/COSMO equation set before any new equation result
    was observed. The clean-head run at `da9d3ba` completed all 60
    fixed-source evaluations. For every record, the three equations used
    exactly identical surface points and areas; the maximum half-coupling
    identity error was `1.67e-16 eV`.

    AIMNet2 fixed \(l=0\) gives MAEs of `1.1297`, `1.0396`, and
    `1.1142 kcal/mol` for IEFPCM, C-PCM, and COSMO. MACE fixed \(l\leq1\)
    gives `0.8780`, `0.9127`, and `0.8738 kcal/mol`. C-PCM therefore has the
    lowest AIMNet2 MAE and improves 8/10 paired rows relative to IEFPCM.
    COSMO has the lowest MACE MAE, but the improvement over IEFPCM is only
    `0.0042 kcal/mol` with a 5/10 versus 5/10 paired split. Across the same
    equation, MACE lowers MAE relative to AIMNet2 by
    `0.1269--0.2517 kcal/mol`.

    This one-row-per-solvent panel does not separate solvent from chemistry,
    establish per-solvent generalization, validate a polarizable fixed point,
    or authorize a default equation. The `5.27 s` wall time excludes frozen
    source inference and is not a randomized speed ranking. COSMO-RS remains a
    separate sigma-profile/statistical-thermodynamic workflow and was not
    relabelled as another surface equation. The aggregate-only evidence is
    `benchmarks/route2-mnsol-fixed-source-pyscf-pcm-family-v1.json`.
47. The separately preregistered ORCA 6.1.0/openCOSMO-RS 24a arm then ran the
    same ten MNSol geometries incrementally from clean commit `6421f18`. Each
    record used the ORCA `COSMORS` workflow at BP86/def2-TZVPD and 298.15 K.
    All ten main jobs and all thirty gas/conductor child jobs terminated
    normally; the aggregation at `eb5b7e1` re-rendered every input, reparsed
    every result, rehashed every asset, and reconstructed every frozen
    baseline comparison without rerunning QC.

    Fixed-geometry openCOSMO-RS gives MAE/RMSE/mean-signed/max errors of
    `0.6460/0.8038/+0.5792/1.5081 kcal/mol`. This is lower than the six
    fixed-source IEFPCM/C-PCM/COSMO MAEs (`0.8738--1.1297 kcal/mol`) on this
    panel. It wins `6--8` of ten paired rows depending on the baseline. The
    summed serial ORCA wall time is `258.03 s` (`25.80 s/record`); comparison
    with the fixed-source continuum timings is not end-to-end because those
    timings exclude frozen source inference.

    This is explicitly **not** independent generalization evidence:
    openCOSMO-RS 24a was fitted using the Marenich/Minnesota solvation data.
    It also uses one fixed MNSol gas geometry rather than the complete
    published conformer and geometry protocol, and one molecule per solvent
    still confounds chemistry with solvent. COSMO-RS remains external to the
    Route-2 profile registry, no default changes, and forces/PES claims remain
    excluded. The aggregate-only evidence is
    `benchmarks/route2-mnsol-opencosmors24a-fixed-geometry-v1.json`.
48. The next no-fit MNSol milestone is now locked to exactly two same-profile
    members: gas MACE-POLAR \(l\leq1\) fixed density and the full
    self-consistent Route-2 \(l\leq1\) response. A new partition aggregator
    accepts only complete, current-schema one-record SCF shards from the
    frozen development partition and one verified execution commit. It
    hard-binds the official unfine-tuned MACE-POLAR-1-M checkpoint, ddPCM
    equation/profile, input-shard set, protocol, selection, dataset, row
    identity including prior-pilot overlap, five-method source-runner schema,
    and every retained energy/error ledger; it then emits only the two
    approved members. Public output contains no row-level MNSol values or
    local checkpoint paths and reports MAE, RMSE, maximum error, and the
    counts/fractions at or above 1.0 and 1.5 kcal/mol (the failures of the
    strict \(<1.0\) and \(<1.5\) targets).

    This is execution infrastructure, not a new accuracy result. No complete
    505-row development or 148-row confirmation matrix has been generated,
    and older isolated shards that predate the required continuum provenance
    are rejected rather than mixed into a new artifact. The confirmation
    partition is rejected by this development-only artifact and remains
    sealed.

    Separately, the data-only `route2_charging_path_diagnostics.py` module
    evaluates the
    preregistered identity
    \(\Delta E_{\mathrm{model}}+U(c_1)
      =\int_0^1U(c_\lambda)\,d\lambda\)
    on a uniform, nested \(4k+1\) coupling grid. It reports the fine/coarse
    Simpson refinement difference and Richardson quadrature-error estimate so
    a nonzero defect is not misclassified as nonvariational before numerical
    convergence. Synthetic polynomial, non-polynomial variational, and
    invalid-grid tests pass. No real MACE-POLAR charging path has yet been
    executed, the production energy is unchanged, and the thermodynamic
    release gate remains open.

49. The exact `smd-ddpcm-l15-n1202-multisolv-v1` profile now carries the
    integrated `safeguarded-anderson-v2` actual-residual rejection,
    rollback, and one-Picard-restart policy together with the unchanged,
    fail-closed, energy-only `finite-resolution-stagnation-v2` contract for an
    approximate fixed-point candidate of the frozen discrete MACE-POLAR/ddPCM
    operator. In Walker--Ni terms this remains a numerical
    Anderson/inexact-Newton acceptance rule, not a new physical free-energy or
    force statement; the variational PCM and ddPCM-force references remain the
    Lipparini et al. and Gatto et al. papers already cited in the formula
    ledger, and the newer safeguarded/restarted Anderson literature motivates
    but does not prove global convergence for this unproven Route-2 map. The
    profile now binds policy `finite-resolution-stagnation-v2`, convergence
    contract `route2-scf-convergence-evidence-v4`, and response-ablation
    runner/artifact schema v5 under the same frozen enumerated runtime identity
    gate: official unfine-tuned MACE-POLAR-1-M checkpoint,
    `mace-torch==0.3.16`, `graph-longrange==0.4.0`,
    `torch==2.12.0+cu130`, `torch.float64`, `device=cpu`, `torch_threads=1`,
    `pyddx==0.8.0`, `n_proc=1`, `lmax=15`, `n_lebedev=1202`, `eta=0.1`,
    solver tolerance `1e-14`, and hashed atomic numbers, coordinates, and
    cavity radii.

    The pyddx tolerance is a relative iterate-change threshold for the inner
    continuum solve, not an outer Route-2 density-residual tolerance. A frozen
    index-248 diagnostic at the prior `1e-12` setting reproduced the 100-attempt
    failure exactly. Keeping every outer SCF, Anderson, finite-resolution, and
    replay threshold unchanged, otherwise identical warm-start diagnostics at
    `1e-13`, `1e-14`, and `1e-15` all reached the earliest valid seven-state
    window at iterations 24--30. The `1e-14` setting is retained because it
    reduced the archived continuum linearity defect by more than one order of
    magnitude relative to `1e-13`, matched the qualitative `1e-15` convergence
    result, and avoided moving the inner solve unnecessarily close to float64
    precision. These post-failure diagnostics are numerical evidence only;
    they do not count as prospective chemistry or matrix-completion evidence.

    The unchanged nominal gate is checked first: monopole residual
    `<= 2e-12 e`, dipole residual `<= 2e-12 e angstrom`, and configured
    intrinsic-energy change `<= 1e-10 eV`. The accepted-state actual-residual
    objective is `Phi = max(monopole/tau_mono, dipole/tau_dipole)` with the
    same nominal channel tolerances in the denominator. If an evaluated
    physical trial arrives by Anderson and `Phi_k > 2 Phi_anchor` for the most
    recent accepted anchor, that trial is rejected. Rejected attempts still
    count toward the existing 100-attempt bound, but they are excluded from
    Anderson samples, best-state selection, and finite-resolution windows. The
    runtime rolls back to the prior accepted anchor, takes exactly one Picard
    step, then rebuilds Anderson from accepted-only history. Accepted-parent
    lineage and solver epochs are audited. The zero-anchor case is evaluated
    without division: zero-to-zero is not rejected, positive-over-zero is
    rejected with a `null` finite-ratio field, and non-finite JSON scalars fail
    closed.

    The fallback still accepts only the **earliest online seven-iteration
    window satisfying every predicate**; merely being the first
    Anderson/no-reset window is insufficient. Because the finite-resolution
    policy itself remains v2, its dimensions are unchanged: monopole and
    dipole residual ceilings `1e-10 e` and `1e-10 e angstrom`; per-component
    root and residual spans below their corresponding nominal `2e-12` channel
    tolerances; conjugate reaction-potential and gradient spans
    `<= 1e-10 eV/e` and `<= 1e-10 eV/(e angstrom)`; intrinsic-energy span
    `<= 1e-10 eV`. The accepted seven-state window must now also stay within
    one solver epoch, contain only accepted Anderson-arrived states, and carry
    an unbroken accepted-parent chain.

    Replay v2 still changes only the online-versus-cold acceptance rule.
    Three fresh reaction-map reevaluations at the retained density are still
    required, and these are still not independent cold SCF starts. The three
    fresh-cold field arrays must have identical canonical little-endian float64
    digests, and the three fresh-cold response arrays must have identical
    canonical little-endian float64 digests. The online warm candidate is then
    compared against each fresh-cold replay with bounded deltas rather than
    online-to-cold byte identity: reaction-potential and
    reaction-gradient component deltas `<= 1e-10 eV/e` and
    `<= 1e-10 eV/(e angstrom)`, response monopole deltas `<= 2e-12 e`, and
    response dipole deltas `<= 2e-12 e angstrom`. Residual ceilings,
    intrinsic/PCM/electrostatic ledger spans, and the `2e-10 eV`
    half-coupling identity gate are unchanged. The artifact retains four
    per-evaluation hashes (online warm plus three fresh-cold) for fields and
    responses, plus the four maximum online-to-cold ULP distances, as
    diagnostics only.

    Historical evidence is preserved. The preregistration diagnostic found the
    first eligible development-index-4 dimethylformamide window at iterations
    17--23, with the online candidate at iteration 23. The retained private
    trajectory and earliest-window replay witnesses remain
    `.omx/diagnostics/mnsol-development-index004-trajectory-4d09ba74-root-v1/trajectory.json`
    and
    `.omx/diagnostics/mnsol-development-index004-iter023-cold-map-4d09ba74-root-v1/diagnostic.json`.
    That earliest-window v1 evidence kept the residual ceiling (legacy mixed
    raw-\(l\le1\) component infinity norm `2.6204635683590993e-11`, not a
    channel-separated v2 quantity),
    half-coupling identity error (`4.526934382909076e-14 eV`), and locked
    `1e-12` intrinsic/PCM/electrostatic ledger spans at zero. The older
    post-hoc best-residual artifact,
    `.omx/diagnostics/mnsol-development-index004-cold-map-4d09ba74-root-v1/diagnostic.json`,
    remains intentionally preserved as a **failed, uncounted diagnostic**.

    The clean byte-identity failure that motivated replay v2 is still archived
    in `.omx/diagnostics/mnsol-development-index004-map-replay-norm-965aeb8-v2/`.
    In that diagnostic-only run, all six actual `np.array_equal` checks were
    false even though the measured deltas stayed small. The online-warm versus
    fresh-cold field comparisons recorded
    `max_abs = 1.936228954946273e-12`, `L2 = 7.901838936066065e-12`, and
    `max_rel = 4.286571482853837e-10`; the corresponding response comparisons
    recorded `max_abs = 7.577272143066693e-15`,
    `L2 = 2.9938859233415605e-14`, and
    `max_rel = 8.560585688915213e-13`. The same diagnostic kept
    `maximum_monopole_residual_e = 1.1122637533222957e-11`,
    `maximum_dipole_residual_e_angstrom = 2.6204635683590993e-11`,
    `intrinsic_ledger_span_ev = 0.0`,
    `pcm_ledger_span_ev = 9.71445146547012e-14`,
    `electrostatic_ledger_span_ev = 9.71445146547012e-14`, and
    `maximum_polarization_identity_error_ev = 4.526934382909076e-14`.
    This remains diagnostic evidence only.

    New index-180 evidence is also preserved as **uncounted monkeypatch-only
    diagnostics**. The archived best failed root in
    `.omx/diagnostics/index180-picard-from-best-16bf09b-v2/diagnostic.json`
    had base residual `3.1794e-12`; a full Picard trial rose to
    `6.937e-11`; and a damped `alpha=0.25` Picard trial reached
    `2.4996e-12`, still outside the nominal gate. The rollback diagnostic in
    `.omx/diagnostics/index180-rollback-picard-16bf09b-v1/` then succeeded at
    attempt 33 under the unchanged finite-resolution-v2 policy with final
    monopole residual `1.1485e-12` and dipole residual `2.36193e-12`.
    Its four rejected Anderson growth ratios were `21.88`, `27.57`, `27.87`,
    and `28.69`. The same one-record shard reported absolute error `0.3790
    kcal/mol`, but that is not an accuracy certification. Because this was a
    runtime monkeypatch diagnostic rather than a prospectively executed public
    solver run, it cannot count toward the exact two-member matrix, the frozen
    505-row development partition, or the sealed 148-row confirmation
    partition. The prospective integrated rerun is still pending.

    A future prospectively accepted result under this gate can still record
    only a repeatable finite-precision approximate fixed-point candidate under
    the frozen residual policy. Neither the preserved diagnostics nor the new
    rule certifies chemistry accuracy, forces, a smooth PES,
    solvent-population performance, any fitting/UQ/public-API claim, or a
    completed 505-row development or 148-row confirmation rerun.

## Secondary diagnostics

- A clean `e1f8acb1` direct-half-coupling MNSol pilot completed both ddPCM and
  scaled-ddCOSMO arms with the field-conditioned SCF source on ten preselected
  records in ten solvents.  The ten
  distinct functional-group labels are post-selection descriptions rather
  than selection inputs.  ddPCM gives MAE/max `1.5277/4.0953 kcal/mol` and
  ddCOSMO gives `1.8619/4.2954 kcal/mol`; both therefore fail the mandatory
  every-record `<1.5 kcal/mol` gate.  This result is retained rather than
  hidden behind the smaller MAE.
- A clean `ae427ea7` zero-field frozen-source rerun uses no field-conditioned
  MACE state and no fixed point.  On the same ten-solvent MNSol pilot, ddPCM
  gives MAE/max `0.8635/1.6454 kcal/mol` and ddCOSMO gives
  `0.9154/1.6451 kcal/mol`; both still fail.  On the immutable FreeSolv-12
  panel, the corresponding values are `1.0843/1.9910` and
  `1.0723/1.9549 kcal/mol`, with 4/12 and 3/12 records at or above `1.5`.
  This sharply reduces the historical SCF-source direct-ledger maxima of
  `7.3303/7.4479 kcal/mol` but does not pass the user's all-record gate,
  establish a common electronic functional, or open forces.
- FreeSolv fixed-conformer hydration errors remain useful for detecting gross
  energy-accounting or chemistry regressions, but expanding or tuning that
  benchmark is not the next Route-2 milestone.
- Dipole, polarizability, provider-parity, and cavity-stability controls remain
  mechanism diagnostics.
- One bounded electronic conformer panel is now recorded as a sensitivity
  diagnostic. Complete conformer thermochemistry, broad multi-solvent
  validation, ions, and radicals remain separate later extensions.

Fresh tests establish implementation correctness, not broad chemical accuracy.
