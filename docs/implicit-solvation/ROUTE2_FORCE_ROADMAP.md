# Route-2 force roadmap

## Scope

Route 2 is the Research/Innovation Route for self-consistent polarizable
MLIP--PCM/SMD coupling:

\[
\Delta G_{\mathrm{solv}}
=\Delta E_{\mathrm{solute}}+U_{\mathrm{pol}}+G_{\mathrm{CDS}}.
\]

The official MACE-POLAR-1-M checkpoint remains unchanged. FreeSolv is a
secondary energy diagnostic, not the definition of the route. The default
PCMSolver profile is a fixed-conformer energy proof-of-concept. The explicitly
named pyddx/PySCF profile is a single-point force candidate. Neither may be
described as a complete solution-phase PES.

## Current capability boundary and PES blockers

1. **Stationarity is not established, and the simple conjugacy shortcut is
   rejected.** The code converges a fixed-point map between learned density
   coefficients and PCM reaction field. A graph-preserving probe confirms that
   the intrinsic MACE energy derivative with respect to the injected local
   potential/gradient is not the returned charge/dipole density. QM-SCF
   Hellmann--Feynman cancellations therefore cannot be imported into Route 2.
   The explicit pyddx candidate therefore differentiates the unmixed
   fixed-point residual with a matrix-free adjoint rather than assuming
   stationarity.
2. **The default PCMSolver path exposes only the fixed-geometry response
   graph.**
   `polar_output_torch()` retains the graph from local potential/gradient to
   both intrinsic energy and density coefficients. A hermitivized, fixed-cavity
   PCMSolver map now supplies matrix-free density-to-field JVP/VJP operations.
   Cavity geometry and boundary operators still cross the NumPy/C boundary and
   have no coordinate derivative. The separately named pyddx map owns its
   scalar ddPCM energy, forward/adjoint maps, and complete coordinate VJP, so
   this blocker does not apply to that candidate.
3. **The PCMSolver binding is energy-only.** The v1.1.12-style C ABI loaded by
   MAPLE exports cavity centres/areas, ASC, response ASC, and polarization
   energy, but no nuclear-gradient or boundary-operator derivative endpoint.
4. **The current CDS area on the default PCMSolver profile is not
   differentiable.** The deterministic
   Shrake--Rupley implementation counts hard visible/occluded spherical points.
   Its area is piecewise constant and jumps when a grid point crosses an
   occlusion boundary. It is suitable for energy controls, not analytic force.
   The pyddx candidate instead uses the official PySCF SMD CDS scalar and its
   matching analytic gradient.
5. **Warning-triggered cavity switching is not a PES rule.** Retrying a single
   point with different GePol parameters removes known numerical warnings, but
   a geometry-dependent switch between primary and fallback cavities would
   introduce an energy/force discontinuity. Route 2 therefore keeps
   `cavity_policy=warning-fallback` as the default fixed-conformer energy
   policy and exposes a separate
   `cavity_policy=fixed-stability-branch` research policy. The latter selects
   `AREA=0.28 A^2, MINRADIUS=0.30 A` before the calculation and fails closed
   on the native `PCMSolver warning.` stderr marker, so it removes policy-level
   geometry branching. `PEDRA.OUT` warnings are retained separately and do not
   select the branch. Those two values are an engineering stability candidate,
   not theoretical constants or a certified tessellation. With
   `UNITS=ANGSTROM`, `AREA=0.28 A^2` is about `0.9999 bohr^2`, appreciably
   coarser than PCMSolver's documented `0.3 bohr^2` default. GePol topology
   continuity and boundary/operator derivatives remain unproven. A force
   implementation on this provider still needs one validated smooth cavity
   construction (or a branch frozen for the entire trajectory), not merely a
   predetermined parameter pair.
6. **The public pyddx force is not yet a PES.** Its fixed `lmax=15`/1202-point
   profile has same-energy derivative, translation, and narrow torque evidence
   on rigid methanol and acetone. The explicit k-space-40 variant additionally
   has current-runtime evidence for one flexible-molecule Cartesian component
   and one central-torsion derivative at a single step. An independently
   reconstructed local two-torsion closed loop and a second flexible
   molecule's center-point force/root audit remain historical. The predeclared
   three-step second-torsion refinement also remains failed even though a
   separate finer-step extension passes; the latter cannot overwrite the
   former. Broader flexible-geometry continuity, relaxed-path behavior,
   additional chemical classes, and short NVE conservation remain open. The
   finite laboratory-frame quadrature also leaves a small, nonmonotonic
   rotation residual, so optimization, scan, TS, and MD stay disabled.

## Required total derivative

Let \(z^*(\mathbf R)=(c^*,\sigma^*)\) denote the converged learned density
coefficients and ASC. The required solvent force is

\[
\mathbf F_A^{\mathrm{solv}}
=-\frac{dE_{\mathrm{solv}}(\mathbf R,z^*(\mathbf R))}
        {d\mathbf R_A}.
\]

If a stationary joint functional is derived and verified, some response terms
may cancel. Otherwise use explicit coupled response or implicit
differentiation. For a fixed-point residual
\(\mathcal R(z,\mathbf R)=0\), an adjoint implementation can solve

\[
\left(\frac{\partial\mathcal R}{\partial z}\right)^T\lambda
=\left(\frac{\partial E}{\partial z}\right)^T,
\qquad
\frac{dE}{d\mathbf R}
=\frac{\partial E}{\partial\mathbf R}
-\lambda^T\frac{\partial\mathcal R}{\partial\mathbf R}.
\]

This avoids back-propagating through an arbitrary number of mixed SCF
iterations while retaining the converged mutual response.

## Phase-0 evidence and derivative decision

A development probe used the final converged acetone reaction field from the
two-canary cavity-preflight run and the official MACE-POLAR-1-M checkpoint in
float64. The recomputed density agreed with the saved converged density within
\(2.59\times10^{-6}\,e\). For six selected potential/gradient components,
autograd and a central difference with step \(10^{-3}\) in the corresponding
model-native field unit agreed within \(3.92\times10^{-7}\).

That verified derivative does **not** equal the returned density:

| comparison | relative \(L_2\) difference | cosine similarity |
|---|---:|---:|
| \(\partial E_{\mathrm{intrinsic}}/\partial V\) vs. \(q\) | 1.0168 | -0.8905 |
| \(\partial E_{\mathrm{intrinsic}}/\partial\nabla V\) vs. \(\mathbf p\) | 1.1028 | -0.7788 |

The least-squares scales are also different for the scalar and vector blocks
(-0.0167 and -0.0999), so a common missing sign, unit conversion, or
Cartesian/e3nn permutation cannot explain the mismatch. One valid state is
sufficient to falsify a universal conjugacy identity; it is not sufficient to
characterize the magnitude of the mismatch over chemical space.

The force design decision is therefore:

1. retain the published Route-2 energy bookkeeping;
2. define the **unmixed** converged residual
   \[
   \mathcal R(c,\mathbf R)
   =c-\mathcal M\!\left(\mathcal P(c,\mathbf R),\mathbf R\right)=0,
   \]
   where \(\mathcal P\) is the PCM reaction-field map and \(\mathcal M\) is the
   MACE-POLAR density response;
3. differentiate that residual with an adjoint/implicit solve, independent of
   the numerical mixing used to reach the root; and
4. do not add an ad hoc local \(\langle\rho,V\rangle\) term, change checkpoint
   weights, or expose forces to make a stationary shortcut appear true.

The earlier SCF grid provides limited path-independence evidence: at
\(10^{-5}\) density/energy tolerances, mixing 0.5 versus 1.0 changed
\(\Delta G_{\mathrm{solv}}\) by at most \(1.29\times10^{-4}\) kcal/mol over
twelve development molecules. This supports treating mixing as a solver
choice, but it is not force or chemical-space certification.

## Phase-1 PCM derivative-provider audit

The provider boundary was audited before adding force code.

### PCMSolver/GePol

The current MAPLE runtime commit and the latest published PCMSolver tag
`v1.3.0` expose no nuclear-gradient endpoint in `api/pcmsolver.h`.
PCMSolver contains a dormant Fortran module,
`pedra_cavity_derivatives.F90`, for derivatives of GePol added-sphere
centres and radii. It is not listed in the PEDRA CMake sources, its import and
call are commented out, and the caller says that the derivative code was
deactivated. There is no corresponding derivative of tessera centres, areas,
normals, single/double-layer matrices, IEFPCM \(K/R\) matrices, or the final
polarization energy in the public solver path.

Therefore this is **not** a small C-ABI exposure task. Enabling the dormant
sphere routine would still omit required terms and could produce a
plausible-looking but incorrect force. The current
PCMSolver--GePol profile remains energy-only.

### PySCF SWIG/ISWIG PCM

The audited PySCF 2.14 source snapshot (`c63a953`) provides an analytic PCM
gradient implementation for its own smooth SWIG/ISWIG atom-centred surface.
Its solver response follows

\[
dE_{\mathrm{PCM}}
=\frac12 v^T K^{-1}(dR-dKq)
+q^Tdv,\qquad q=K^{-1}Rv,
\]

with explicit derivatives of the switching function, areas, and the
single/double-layer matrices. The PCM files are Apache-2.0 licensed. This
algebra is an established reference and its smooth surface is a viable
force-provider candidate.

This source observation implies a derivative-consistency requirement rather
than quoting a provider restriction: a force must differentiate the same
discrete energy. PySCF's surface points, areas, Gaussian exponents, \(S/D\)
matrices, and \(K/R\) operators differ from the GePol objects used by the
current energy. Attaching the former's derivatives to the latter's energy
cannot be assumed to give \(dE_{\mathrm{GePol}}/d\mathbf R\); it would define
a new mixed approximation requiring its own derivation and validation. MAPLE
therefore evaluates PySCF through a separately named Route-2 profile whose
energy and runtime are compared against the current profile on a small canary
set before adoption.

The audited PySCF `solvent/smd.py` and `solvent/grad/smd.py` files carry
GPL-3.0 headers. The gradient's CDS helper calls `smd.get_cds_legacy()`, which
in turn calls the compiled `libsolvent.mnsol_interface_`. MAPLE does not copy,
vendor, or automatically install that code. The optional research bridge calls
an independently installed PySCF runtime through its Python module boundary;
redistribution of such a runtime remains a deployment/license concern rather
than a reason to invent a second CDS functional. The scientific requirement is
that CDS energy and gradient come from the same selected provider.

### Decision

1. Preserve the current PCMSolver--GePol energy profile and do not advertise
   forces for it.
2. Do not vendor or reactivate the incomplete PEDRA derivative path.
3. Implement provider-neutral point-multipole and ASC kernel derivatives as
   direct vector-Jacobian products, without dense molecular Jacobians.
4. Next, evaluate a separately named smooth SWIG/ISWIG IEFPCM profile through
   an optional PySCF provider; use two canaries before any broader benchmark.
5. Never combine energy from one cavity/operator definition with derivatives
   from another.
6. For the optional PySCF profile, reuse PySCF's production SMD CDS
   energy/gradient pair rather than reconstructing a nominally similar area
   model in MAPLE.

## Implementation sequence

### Phase 0 -- lock the energy functional

1. **Done:** expose a torch-native local-field evaluation without NumPy
   detachment.
2. **Done for the stationary decision:** differentiate MACE energy with respect
   to local potential/gradient and verify autograd against finite differences.
   The density-conjugacy identity is rejected.
3. **Development evidence only:** mixing/path dependence is below the current
   energy tolerance on the twelve-molecule probe; repeat at force-gate
   geometries before production.
4. **Decision:** Route 2 requires an adjoint fixed-point derivative. Production
   force remains disabled until all later phases pass.
5. **Done for the fixed-cavity response skeleton:** the fixed-geometry MACE
   field-to-density JVP/VJP and
   `UnmixedDensityResidualLinearization` now implement
   \(\Pi_0(I-J_{\mathcal M}J_{\mathcal P})\) and its discrete adjoint on the
   neutral density tangent space. Synthetic dense-matrix comparisons and a
   JVP/VJP bilinear identity pass. On one real float64 acetone direction, the
   MACE density JVP agrees with a central field difference to
   \(1.29\times10^{-7}\) maximum absolute error, while its JVP/VJP bilinear
   identity closes to \(1.94\times10^{-12}\) absolute error.
   `FixedCavityPCMReactionFieldLinearMap` applies the same MEP--ASC--reaction
   field chain as the energy path and admits its reciprocal adjoint only for a
   parsed `MATRIXSYMM=TRUE` operator at exactly the same geometry. On the real
   saved 508-tessera acetone fallback cavity, its bilinear identity closes to
   \(6.66\times10^{-16}\) absolute error and the complete residual identity to
   \(1.24\times10^{-11}\). This remains fixed-cavity response infrastructure,
   not a force capability.
6. **Done for a diagnostic PCM-coupled adjoint solve:**
   `NeutralDensityCoordinates` uses an
   orthonormal Helmert charge basis and `solve_adjoint()` solves the residual
   VJP with matrix-free GMRES. A dense neutral-subspace system matches the
   direct solution, while a singular synthetic operator fails closed. One real
   acetone random right-hand side converges in eight callbacks and ten operator
   applications to \(2.36\times10^{-9}\) relative residual.
7. **Done for the physical energy-gradient right-hand side at fixed cavity:**
   `MACEPolCalculator.intrinsic_energy_field_gradient()` evaluates the exact
   MACE intrinsic-energy derivative \(g_f\) with respect to the external node
   field. `fixed_cavity_energy_density_gradient()` then assembles
   \[
   b=\Pi_0\left[\mathcal P_{\mathbf R}^*g_f+Qf\right].
   \]
   Synthetic neutral-direction finite differences lock the \(1/2\) PCM
   derivative, permutation, and projection. On the real saved acetone cavity,
   the energy identity closes to \(1.11\times10^{-16}\) eV, three density-space
   central differences have relative errors at most \(6.54\times10^{-7}\),
   the saved fixed-point residual is \(2.59\times10^{-6}\), below its
   \(10^{-5}\) threshold,
   and the physical-RHS adjoint reaches \(7.65\times10^{-10}\) relative
   residual in eight callbacks and ten operator applications. This is still a
   fixed-geometry/fixed-cavity density derivative, not a solvent force;
   coordinate, boundary, and CDS derivatives remain absent.

### Phase 1 -- differentiable explicit geometry terms

1. **Done for the MACE-side partial:** `PolarState` now returns the gas or
   polarized intrinsic MACE force
   \[
   -\left.\frac{\partial E_{\mathrm{MACE,intrinsic}}}
                 {\partial\mathbf R}\right|_{\{V_i,\nabla V_i\}}
   \]
   in eV/angstrom when requested. The atom-indexed potential and gradient
   samples are held fixed; their geometry response is deliberately excluded.
   A real float64 acetone canary checked nine Cartesian components in both gas
   and polarized states over \(10^{-3}\), \(3\times10^{-4}\), and
   \(10^{-4}\) angstrom central-difference steps. The best maximum absolute
   errors were \(4.37\times10^{-6}\) and \(3.27\times10^{-6}\) eV/angstrom,
   respectively. The polarized-minus-gas partial force was nonzero
   (\(L_2=8.01\times10^{-2}\) eV/angstrom). This is not a total solvent force.
2. **Done for fixed density and fixed PCM surface/operator:** analytic position
   VJPs for the point-multipole MEP and ASC back-projection kernels avoid dense
   Jacobians and pass individual central differences plus the differentiated
   reciprocal identity. `FixedCavityPCMReactionFieldLinearMap.position_vjp()`
   composes both terms for
   \(\partial_{\mathbf R}\langle w,\mathcal P_{\mathbf R}c\rangle\).
   On the real saved acetone fallback surface, the six largest components over
   three steps have at most \(2.40\times10^{-7}\) eV/angstrom absolute and
   \(1.65\times10^{-6}\) relative error. The analytic VJP took 0.032 s in the
   local canary, compared with 0.611 s for its 36 scalar evaluations
   (18 central differences). Tessera motion, PCM-operator response, and CDS are
   still missing, so this is not a total force.
3. **Done for the fixed-field MACE density coordinate VJP:**
   `MACEPolCalculator.density_position_vjp()` contracts
   \((\partial_{\mathbf R}\mathcal M|_f)^\mathsf T\lambda\) directly through
   the model autograd graph while keeping the atom-indexed potential and
   gradient samples fixed. An exact synthetic position-dependent-density model
   locks this partial, rejects a coordinate-disconnected graph, and verifies
   that temporary position-autograd state does not escape the call. On the real
   fixed-surface acetone state, its two largest components over three steps had
   at most \(1.60\times10^{-6}\) eV/angstrom absolute and
   \(2.16\times10^{-6}\) relative error. The PCM field-position chain remains a
   separate `position_vjp()` term to prevent double counting.
4. **Provider decision made:** retain PCMSolver--GePol as energy-only and
   evaluate an explicit PySCF SWIG/ISWIG profile for smooth cavity/operator
   derivatives. Do not silently approximate missing GePol terms as zero.
5. **Done for the geometry-dependent atomic-tension component:**
   `aqueous_atomic_surface_tension_position_vjp()` analytically contracts the
   published H/C/N/O switching-function response
   \(\sum_i A_i\,d\gamma_i/d\mathbf R\) without a dense Jacobian. A synthetic
   all-branch coordinate oracle and translation-invariance check pass. On the
   methanol NWChem-control geometry, the six largest fixed-area components over
   three steps had at most \(5.64\times10^{-10}\) hartree/angstrom absolute and
   \(9.29\times10^{-7}\) relative error, with a
   \(1.12\times10^{-19}\) hartree/angstrom net translation-gradient norm.
6. **Diagnostic implemented; production replacement remains open:** the clean
   artifact at Route-2 commit `54cd781` evaluates a separately named
   Fibonacci-grid, SWIG-inspired area that supplies an analytic
   \(\sum_i\gamma_i\,dA_i/d\mathbf R\) VJP. Its own discrete energy and
   derivative agree to \(2.31\times10^{-12}\) hartree/angstrom at the smallest
   methanol step, all three static NWChem controls are within
   `0.001 kcal/mol`, and the canonical hard-area energy is unchanged. However,
   an external PySCF 2.13.1 Lebedev-SWIG comparison finds dense-grid
   coordinate-VJP differences of `5.84--9.73%`, nonmonotonic with point count.
   At 5810 points the Fibonacci and Lebedev rigid-rotation spans are
   `0.00352` and `0.00260 kcal/mol`, respectively, and the Fibonacci residual
   torque is \(1.67\times10^{-4}\) hartree. The public provider does not select
   this candidate. Resolve the invariance/equivalence gates or adopt an
   independently validated analytic area provider before force publication.
7. **Policy branch removed for research canaries; derivative gate remains
   open:** `cavity_policy=fixed-stability-branch` selects the stability
   discretization before evaluation and never probes the primary cavity.
   This is branch-free at the policy level but is deliberately still marked
   `force_compatible=false`; validate GePol topology/rotation continuity and
   supply surface/operator coordinate derivatives before using it in a total
   force. A subsequent correct-unit scan found no warning-free `AREA`-only
   point on acetone. The warning-free `AREA=1.0 bohr^2`,
   `MINRADIUS=1.0 A` candidate removed an added sphere and slightly worsened the
   three-canary MAE, so it is not a public replacement. Seven acetone C=O
   displacements retained one cavity signature and showed step-refinable local
   energy derivatives, which is enough to continue code-level derivative work
   on that diagnostic branch but not enough for parameter adoption or a PES
   claim.
8. **Done for the provider-neutral energy boundary; smooth provider remains
   experimental:** `ExternalMEPCavityResponse` now makes the per-atom radii,
   reference geometry, surface, and energy-conjugate response explicit.
   `SurfaceChargeState` enforces
   \(q_{\mathrm{sym}}=(q+q^\dagger)/2\) and
   \(E_{\mathrm{pol}}=v^\mathsf Tq_{\mathrm{sym}}/2\).
   The current PCMSolver adapter admits only `MATRIXSYMM=TRUE`; the public
   parser and calculator still accept only `provider=pcmsolver`.

   A clean staged artifact at Route-2 commit `3bd6a31` used isolated PySCF
   2.13.1 SWIG/IEFPCM to test two molecules, not a broad benchmark. Orders 17
   and 29 passed energy/convergence gates but
   failed the predeclared rigid-rotation gate. On fixed ethoxyethane density,
   order 35 reduced the twelve-orientation span to `0.007734 kcal/mol`, below
   `0.01 kcal/mol`. Full order-35 ML--PCM calculations converged in 18
   iterations for acetone and 17 for ethoxyethane; their differences from the
   fixed PCMSolver profile were `0.042540` and `0.001338 kcal/mol`, respectively.
   A fixed-surface-potential operator-gradient canary reached
   `1.53e-8` relative finite-difference error and numerical translation
   closure. These results justify continued optional-provider investigation,
   but the candidate is **not production-ready**: total derivatives,
   differentiable CDS, scaling, and broad speed/accuracy evidence remain open.
   The later version-locked adapter described below resolves mixed-oxygen
   per-atom radius assignment without copying PySCF's surface algorithm. PySCF
   is not a MAPLE dependency and no speed claim follows from these local
   timings.
9. **Done for the provider-neutral operator-VJP boundary; no production
   derivative provider yet:** `ExternalMEPCavityOperatorDerivative` represents
   \(d\langle u,Q_{\mathrm{sym}}(R)v\rangle/dR\) directly, without assembling a
   dense response Jacobian. `polarization_operator_position_gradient()` applies
   the required one-half factor for
   \(E_{\mathrm{pol}}=v^\mathsf TQ_{\mathrm{sym}}v/2\), while
   `FixedCavityPCMReactionFieldLinearMap.continuum_operator_position_vjp()`
   maps an arbitrary atom-field cotangent into the corresponding left surface
   potential. A coordinate-dependent nonsymmetric synthetic \(K/R\) oracle
   passes general bilinear and energy finite differences. The real PCMSolver
   adapter has no such endpoint and intentionally raises instead of treating
   the missing GePol derivative as zero.
10. **Done for narrow optional same-energy smooth-continuum and coupled-response
    canaries; adoption remains open:** `PySCFSWIGIEFPCMResponse` lazily uses
    PySCF 2.13.1 to build one SWIG surface and the matching IEFPCM \(K/R\)
    operator.
    `AtomCenteredSurfacePCMReactionFieldLinearMap.full_position_vjp()` combines
    fixed-surface solute/back-projection kernels, rigid parent-atom motion of
    every surface node, and the same PySCF operator derivative without
    combining providers. On fixed-density order-17 acetone (643 surface
    points), three representative \(10^{-4}\)-angstrom finite differences had
    relative errors \(1.54\times10^{-8}\), \(6.71\times10^{-9}\), and
    \(2.00\times10^{-7}\), with translation closure below
    \(1.9\times10^{-16}\) eV/angstrom. The tracked adapter and an independent
    formula implementation agree to \(1.9\times10^{-15}\) relative or better.

    The version-locked adapter delegates surface generation to upstream PySCF
    using atom indices as radius-vector keys, verifies that integer lookup
    contract at runtime, and retains the real molecule for all nuclear
    bookkeeping. Repeated-element equal-radius surfaces are bitwise identical
    to the ordinary PySCF path. A 695-point methyl-acetate surface retained
    distinct `o=1.70 A` and `os=1.52 A` radii; its checked fixed-density
    components had at most \(3.50\times10^{-8}\) eV/angstrom absolute and
    \(1.85\times10^{-6}\) relative finite-difference error.

    Real resolved-root float64 MACE-POLAR adjoint canaries now pass as well.
    At a \(10^{-3}\)-angstrom step, the two largest checked acetone components
    differed from whole-energy finite differences by at most
    \(2.15\times10^{-6}\) eV/angstrom. The largest checked mixed-radius
    methyl-acetate component differed by \(8.51\times10^{-7}\) eV/angstrom
    (\(1.14\times10^{-6}\) relative); every displaced density root met the
    \(2\times10^{-12}\) tolerance. These remain two-molecule canaries: the
    private PySCF bridge is version-gated, the public parser remains
    PCMSolver-only, and CDS plus total-force assembly remain outside the result.

    A subsequent clean methanol refinement at commit `7da54dc` compared the
    complete self-consistent continuum gradient across three fixed
    orientations. Orders 35 and 41 failed the predeclared
    \(10^{-3}\)-eV torque gate at \(2.09\times10^{-3}\) and
    \(1.44\times10^{-3}\) eV. Order 47 passed with a
    `0.001476 kcal/mol` energy span, `0.001678 eV/angstrom` maximum gradient
    covariance error, and \(5.16\times10^{-4}\)-eV maximum torque. The
    order-47 run used 2211--2245 surface points, about 3.12 GiB peak RSS, and
    54.9 seconds for three orientations on the local host. The covariance
    metric was nonmonotonic between orders 41 and 47, so order 47 is only the
    first tested grid to pass this one-molecule discriminator, not an adopted
    production order or broad speed result.
11. **Done for an optional official CDS component; total assembly remains
    open:** `pyscf_smd_water_cds()` is version-locked to PySCF 2.13.1 and calls
    `pyscf.solvent.smd.get_cds_legacy`, which returns the production SMD CDS
    energy and analytic coordinate gradient together through compiled
    `libsolvent`. The adapter converts hartree/bohr to a **position gradient,
    not force**, in hartree/angstrom, freezes its result arrays/provenance, and
    fails closed for absent `libsolvent`, untested versions, invalid geometry,
    or non-finite upstream results.

    A clean one-methanol water canary at commit `b9a06f6` matched the NWChem
    static reference within \(9.42\times10^{-8}\) kcal/mol. Its complete
    18-component analytic gradient had
    \(1.87\times10^{-12}\) hartree/angstrom maximum central-difference error,
    \(1.35\times10^{-9}\) relative \(L_2\) error, numerical translation
    closure, and numerical rigid-rotation covariance/torque closure. The local
    process took 1.92 seconds and its external `/usr/bin/time` process envelope
    reached about 0.92 GiB peak RSS, but it contained CDS only. It does not
    establish public-provider integration, a total solvent
    force, chemical-space accuracy, or a portable speed advantage.

### Phase 2 -- coupled response

1. **Done:** form the unmixed converged residual for learned density plus the
   reciprocal fixed-cavity PCM response.
2. **Done:** implement Jacobian-vector and vector-Jacobian products without
   assembling a dense molecular Jacobian.
3. **Done at fixed geometry/cavity:** solve the physical energy-gradient
   adjoint equation to a tolerance tighter than the energy SCF tolerance.
4. **Done for the fixed-surface/operator coupled slice:**
   `density_to_external_field_order()` distinguishes \(Q^\mathsf Tc\) from
   \(Qc\), and `fixed_surface_solvation_coordinate_gradient()` composes
   \[
   \mathbf F_{\mathrm{gas}}-\mathbf F_{\mathrm{MACE,fixed\ field}}
   +\partial_{\mathbf R}
    \left\langle
    g_f+\tfrac12Q^\mathsf Tc+J_{\mathcal M,f}^\mathsf T\lambda,\,
    \mathcal P_{\mathbf R}c
    \right\rangle
   +(\partial_{\mathbf R}\mathcal M|_f)^\mathsf T\lambda .
   \]
   A resolved-root synthetic implicit-function oracle passes. On the real
   warning-free acetone fallback surface, every displaced root was below
   \(2.0\times10^{-11}\), and the two largest components over three steps had
   at most \(4.02\times10^{-6}\) eV/angstrom absolute and
   \(5.00\times10^{-6}\) relative error. After the base root, the analytic
   derivative was about 15--20 times faster than twelve root-resolved scalar
   energy evaluations in local canaries; exact host-specific timings remain in
   the corresponding artifact. This is still not a force because
   surface/operator motion and CDS are omitted.
5. **Done for the full-continuum assembly boundary and narrow optional
   fixed-density plus real coupled-response provider canaries; adoption
   pending:**
   `FullReactionFieldPositionDerivative` contract version 1 requires the same
   reaction-field object that supplies the forward/adjoint maps to also supply
   `full_position_vjp()`. That VJP owns the complete derivative of
   \(\langle w,\mathcal P_{\mathbf R}c\rangle\), including solute projection,
   moving-surface kernels, continuum-operator response, and reaction-field
   back-projection. `continuum_coupled_solvation_coordinate_gradient()` reuses
   the fixed-point adjoint cotangent and calls only that full VJP; it does not
   sum separately supplied partials. A resolved-root synthetic oracle and a
   split fixed-surface/full-map test pass. The current PCMSolver-backed map
   intentionally lacks this versioned contract and fails closed, so this is an
   architectural gate rather than a new public force implementation. The
   optional PySCF SWIG adapter now passes the fixed-density canaries described
   in Phase 1 and real resolved-root MACE-adjoint finite differences on acetone
   and mixed-radius methyl acetate. The canaries load MACE through existing
   public calculator plumbing but bypass the attached PCMSolver correction. In
   the corrected order-47 artifact that unused correction is detached before
   PySCF-SWIG evaluation and its loader-only manifest is not retained; the
   result-level PySCF provenance identifies the continuum actually
   differentiated. CDS is absent from these continuum result objects, while the
   separately validated optional PySCF CDS component was not evaluated inside
   them. At this stage `supported_properties` therefore remained energy-only;
   the later explicit pyddx integration is recorded in Phase 8.
6. **Done for provider-neutral bookkeeping; real validation under the same
   declared profile has begun:**
   `assemble_total_solvation_coordinate_gradient()` converts
   the continuum correction gradient from eV/angstrom to
   hartree/angstrom, adds the CDS position gradient, and returns the negative
   total as the solvent correction force. Its immutable component-resolved
   result and synthetic finite-difference tests lock the unit conversion and
   force sign. The function deliberately accepts no gas force because
   `CalcABC` owns the one and only addition of the independently returned gas
   force. It also does not select providers; callers must still prove that all
   components belong to the same declared energy profile.

   A clean one-methanol canary at commit `2744038` combined real float64
   MACE-POLAR, order-47 PySCF SWIG/IEFPCM, and official PySCF SMD CDS for the
   largest total-gradient component. The \(3\times10^{-5}\)-angstrom
   topology-stable central difference gave continuum, CDS, and total absolute
   errors of \(2.38\times10^{-6}\), \(1.67\times10^{-7}\), and
   \(2.21\times10^{-6}\) eV/angstrom; total relative error was
   \(6.45\times10^{-6}\). Translation and torque norms were
   \(1.24\times10^{-14}\) eV/angstrom and \(5.16\times10^{-4}\) eV. The clean
   run took 48.5 seconds and about 3.10 GiB peak RSS; CDS itself required only
   milliseconds, so the dense continuum dominated. The failed
   \(10^{-3}\)-angstrom topology gate and the \(10^{-4}\)-angstrom CDS
   truncation gate are retained as artifacts.
7. **Done as a rejection gate for fixed-order refinement; a replacement
   discretization remains pending:** an acetone order-47 total-gradient
   preflight used 3435 surface points. The base torque was
   \(1.1146\times10^{-3}\) eV and failed the \(10^{-3}\)-eV gate, although one
   extra orientation passed at \(6.60\times10^{-4}\) eV. Along the base
   residual-torque axis, a topology-stable \(3\times10^{-5}\)-rad central
   difference of the complete self-consistent continuum-plus-CDS energy was
   `0.001116823 eV/rad`, versus `0.001114604 eV/rad` analytically. The
   `0.199%` relative difference and essentially zero CDS contribution identify
   finite-grid discrete-energy rotation anisotropy rather than a missing
   analytic-gradient term.

   A single bounded order-53 follow-up was deliberately run instead of a
   broad order scan. Its base torque improved to \(2.94\times10^{-4}\) eV, but
   the same extra orientation worsened to \(1.77\times10^{-3}\) eV and failed.
   The base surface increased to 4244 points, the analytic derivative from
   `24.79 s` to `38.39 s`, and peak RSS from `4.74 GiB` to `6.23 GiB`.
   Therefore neither order 47 nor order 53 is a general production default,
   and order 59 was not attempted.

   The next continuum provider must use a rotation-covariant discretization
   (or an equivalently rotation-stable construction) and must own the matching
   scalar energy and analytic coordinate derivative.
   Literature-backed candidates are a molecule-following atom-centred grid
   with the complete orientation-matrix derivative, or a separately named
   ddPCM/ddCOSMO provider with its own same-energy force derivation. ISWIG
   alone is not a fundamental fix because it changes the switching function
   while retaining the laboratory-frame Lebedev construction. Post-hoc torque
   removal is nonconservative relative to the implemented scalar energy and is
   forbidden. At this phase, only after a replacement provider passed this
   gate, additional total-gradient components, and same-profile provenance
   could
   `SolvationResult.forces_hartree_per_angstrom` be populated or
   `supported_properties={"energy", "forces"}` be advertised for an explicit
   research profile. Phase 8 records that narrow publication gate. The per-atom
   radius, real-MACE fixed-point adjoint, CDS component, and algebraic total
   assembly are no longer the blockers.
8. **Done for an independent multipole-native ddPCM backbone, real
   ML-SCF/adjoint-gradient canaries, one total continuum-plus-CDS component on
   each of two molecules, and the first explicit public single-point force
   integration; PES validation remains pending:**
   `PyDDXPCMReactionFieldLinearMap` lazily and exactly version-locks pyddx
   0.8.0. It maps MACE-POLAR \(l\leq1\) atom-centred multipoles directly into
   ddX, obtains the reciprocal reaction field from one forward and one adjoint
   solve, and obtains the complete bilinear coordinate VJP from two
   same-energy ddPCM gradients. It does not reuse the SWIG surface or mix
   PCMSolver energy with ddX derivatives. The public factory selects it only
   for the exact explicit pyddx profile; PCMSolver remains the default.

   A clean fixed-density methanol canary at commit `facd956` used `lmax=15`
   and 770 Lebedev points per sphere. Direct MEP agreement was
   \(5.55\times10^{-17}\) hartree/e, the energy identity and reciprocity errors
   were \(2.28\times10^{-15}\) and \(1.92\times10^{-13}\) eV, and the checked
   coordinate derivative differed from central finite difference by
   \(1.04\times10^{-9}\) eV/angstrom. Three orientations gave a
   `0.000207 kcal/mol` energy span and `0.000694 eV/angstrom` maximum
   coordinate-gradient covariance error.

   On the local host, base construction, reaction-map application, and the
   complete coordinate VJP took `0.216`, `0.421`, and `0.902 s`. The complete
   three-orientation/two-displacement validation took `7.32 s` and emitted no
   warnings. Those figures are diagnostic only; `lmax=15`/770 is not adopted
   as a public profile. Because the adapter bypasses PCMSolver, it has no
   `primary` branch. Warnings from the default PCMSolver public path still
   describe its unchanged `warning-fallback` cavity probe, not this ddPCM
   solve.

   On clean baseline `309366e`, the same map was substituted into the existing
   real MACE-POLAR-1-M fixed point and adjoint. At `lmax=15`/770, `mixing=1.0`
   converged in 18 iterations to a \(6.23\times10^{-13}\) density residual.
   The polarization-energy identity error was \(1.29\times10^{-14}\) eV, the
   adjoint relative residual \(5.51\times10^{-14}\), and the checked
   fully-reconverged whole-energy gradient error
   \(1.20\times10^{-6}\) eV/angstrom. The density root and post-root analytic
   derivative took `17.44` and `17.67 s`.

   The 770-point base torque passed at \(9.05\times10^{-4}\) eV, but one
   additional orientation failed at \(1.31\times10^{-3}\) eV. A single
   1202-point follow-up passed both orientations at
   \(6.38\times10^{-4}\) and \(3.35\times10^{-4}\) eV, while its
   whole-energy gradient error remained
   \(9.36\times10^{-7}\) eV/angstrom. The cost rose to `23.77 s` for the base
   root and `24.83 s` for the analytic derivative. Its energy span
   (`0.0004776 kcal/mol`) and gradient covariance
   (`0.0009010 eV/angstrom`) moved close to their predeclared limits, so this
   nonmonotonic one-molecule result does not select 1202 points as a default.

   On clean baseline `c63f78f`, the separately tested PySCF 2.13.1 water-SMD
   CDS energy/gradient pair was added to the 1202-point continuum gradient.
   On the largest total-gradient component (C0-y), a predeclared
   \(3\times10^{-5}\)-angstrom fully reconverged central difference gave
   continuum, CDS, and total absolute errors of
   \(6.00\times10^{-6}\), \(1.67\times10^{-7}\), and
   \(6.17\times10^{-6}\) eV/angstrom. The total relative error was
   \(1.80\times10^{-5}\), just below the \(2\times10^{-5}\) gate.
   Translation and torque norms were \(1.26\times10^{-14}\) eV/angstrom and
   \(6.38\times10^{-4}\) eV. All predeclared algebra, convergence,
   finite-difference, translation, torque, and warning gates passed.

   The base root and analytic derivative took `14.82` and `15.73 s`; the two
   fully reconverged displaced energies took `32.24 s`. The base CDS call took
   `0.0615 s`, so the response derivative and finite-difference oracle—not
   CDS—dominate this validation command. The structured PCMSolver/`primary`
   warning count was zero; a JSON key that asserts
   `no_pcmsolver_primary_warning=true` is a passed gate rather than a warning.

   On clean baseline `aae25a8`, the same total-gradient discriminator was
   applied to ten-atom acetone. This exposed a generic SciPy GMRES budget
   error: the old 20-vector restart plus `callback_type=pr_norm` made
   `maxiter=100` permit 100 restart cycles. The corrected solve uses the full
   39-dimensional neutral Krylov space and counts at most 100 inner
   iterations. A strict \(10^{-11}\) outer target still stalled because the
   measured CUDA-float64 linear-superposition floor was
   \(5.40\times10^{-11}\), not because the explicitly assembled operator was
   ill-conditioned (`cond=1.573`) or non-repeatable.

   With an evidence-calibrated \(10^{-10}\) outer tolerance, the acetone
   adjoint converged in 11 callback iterations and 14 operator applications.
   Continuum, CDS, and total whole-energy finite-difference errors were
   \(6.46\times10^{-6}\), \(9.30\times10^{-10}\), and
   \(6.46\times10^{-6}\) eV/angstrom; the total relative error was
   \(6.97\times10^{-6}\). Translation and torque norms were
   \(1.39\times10^{-14}\) eV/angstrom and \(2.38\times10^{-4}\) eV. Every
   declared component, total, conservation, and warning gate passed.

   The acetone base root, analytic derivative, and two-displacement oracle
   took `35.97`, `35.05`, and `64.34 s`, versus methanol's `14.82`, `15.73`,
   and `32.24 s`; the complete acetone process took `149.24 s`. Both roots
   needed 18 iterations and the adjoints used a similar number of operator
   applications. The increase is therefore mainly the larger ten-sphere,
   39-dimensional per-iteration problem, not the `0.068-s` CDS term or a
   `primary` fallback. Its structured PCMSolver/`primary` warning count was
   zero.

   The public single-point methanol canary then selected
   `provider=pyddx,profile=smd-ddpcm-l15-n1202-v1` through the normal parser,
   calculator builder, correction factory, MACE-POLAR calculator, and shared
   result finalizer. Its correction energy differed from the independent
   same-profile result by \(3.83\times10^{-13}\) eV, and its correction force
   differed from the negative independent total coordinate gradient by at most
   \(4.11\times10^{-11}\) eV/angstrom. The ML-SCF root took 16 iterations; the
   adjoint took 8 residual callbacks and 10 operator applications and reached
   relative residual \(6.33\times10^{-11}\). Model load and public force
   evaluation took `3.35` and `24.27 s`; the complete process took `29.75 s`.
   The structured forbidden-provider warning count was zero.

   This public result is a single-point research force candidate, not chemical
   accuracy evidence or a MAPLE solution-phase PES. The next bounded work is
   additional rigid orientations followed by flexible-geometry continuity,
   closed-loop, and short-NVE gates without changing the named 1202-point
   profile or calibrated solver policy.

10. **Experimental reciprocal MACE evaluator is isolated behind one complete
    profile.** The default molecular real-space evaluator failed the bounded
    acetone total-energy rotation gate at `0.0018627013 kcal/mol`. The same
    unmodified checkpoint with graph_longrange's forced reciprocal evaluator
    and an explicit fixed 40 Å helper box reduced the complete ML-SCF/SMD span
    to \(8.3038\times10^{-5}\) kcal/mol. The two-orientation force-covariance
    error was \(3.4356\times10^{-4}\) eV/angstrom, and the refined
    whole-energy finite-difference force error was
    \(7.50\times10^{-7}\) eV/angstrom.

    The accepted architecture does not monkeypatch the model and does not put
    box logic in ddPCM. `route2_smd_profiles.py` binds provider, cavity, and
    evaluator; `_macepol_long_range.py` owns fixed-box centering, the
    graph_longrange 0.4.0 gate, the narrow dtype bridge, and the forward flag;
    `MACEPolCalculator._model_forward()` is the only model-call dispatch
    boundary. Existing profiles select an exact no-op evaluator policy.

    The public-path acetone energy and analytic correction force reproduce the
    prior independent evidence within \(10^{-11}\) eV and
    \(10^{-10}\) eV/angstrom, respectively. The public evidence artifact must
    also identify a clean Git commit and classify PCMSolver/`primary`,
    checkpoint-dtype, and dependency warnings separately. These gates permit
    the explicit non-default
    `smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1` research profile only.
    Arbitrary boxes/evaluators, default promotion, equivalence claims, and PES
    tasks remain rejected.

### Phase 3 -- verification gates

1. Use whole-energy central finite differences only as an oracle. Demonstrate
   step-size convergence on small rigid molecules before testing flexible
   molecules.
2. Compare component-resolved and total analytic derivatives against the
   oracle; a total-force match alone must not hide cancelling component errors.
3. Check zero net force under translation and zero net torque under rotation.
4. Check energy continuity across small geometry displacements and reject any
   cavity-topology or fallback-branch jumps.
5. The source-geometry analytic force for a fixed, unrelaxed
   2-acetoxyethyl-acetate conformer is reconfirmed at canary execution head
   `d72dfba`.
   Its direct one-component \(5\times10^{-4}\)-angstrom displacement pair is
   also reconfirmed against that runtime: the analytic/finite-difference
   mismatch is \(3.0841\times10^{-6}\) eV/angstrom, both sides converge in 18
   root iterations, and PCMSolver/legacy-`primary` warning counts are zero.
   A separately locked central C--C torsion pair at \(\pm0.5^\circ\) is now
   reconfirmed at the same execution head. Its analytic and central-difference
   generalized forces are `0.04520643472551838` and
   `0.04510541195057784 eV/rad`, an absolute error of
   \(1.01023\times10^{-4}\) eV/rad and a relative error of
   \(2.2347\times10^{-3}\). Both energy-only points converge in 18 iterations
   without either warning class. A read-only finalizer recovered the completed
   point records after a result-serialization error; no scientific evaluation
   was rerun. A second locked pair at \(\pm1.0^\circ\) gives
   `0.04504468982867865 eV/rad`, with
   \(1.61745\times10^{-4}\) eV/rad absolute error. Refining to
   \(0.5^\circ\) reduces the error by
   \(6.07221\times10^{-5}\) eV/rad, to `0.62458` of the coarse-step
   error. Both additional roots converge in 18 iterations and emit zero
   PCMSolver or legacy `primary` warnings. A third locked pair at
   \(\pm0.25^\circ\) gives `0.04516757532134049 eV/rad`, reducing the
   analytic discrepancy to \(3.88594\times10^{-5}\) eV/rad. Both new points
   converge in 18 iterations with exact formula/geometry closure and no
   PCMSolver or legacy `primary` warning.

   The three-step smooth second-order gate **fails**. The refinement drifts are
   \(6.07221\times10^{-5}\) and \(6.21634\times10^{-5}\) eV/rad, so the
   observed order is `-0.03384`, outside the locked `1.5--2.5` interval.
   Component orders are `2.008` for gas MACE and `2.001` for CDS, but only
   `0.544` for solvent-intrinsic MACE and `0.212` for PCM polarization. The
   failure is therefore in the self-consistent electrostatic coupling block,
   not in gas MACE or CDS.

   The completed pre-registered diagnosis reuses the six immutable torsion
   states and performs exactly one independent \(+0.25^\circ\) energy-only
   root. It adds no geometry, finite-difference step, or force evaluation. The
   independent result reproduces energy to
   \(2.22\times10^{-16}\) eV and density to
   \(2.22\times10^{-16}\) electron, all 18 validity gates pass, and the
   PCMSolver and legacy `primary` warning counts are zero.

   Formal mapping to the PySCF 1202-point Lebedev grid finds 19 changed cavity
   active points across the minus-side fine interval and 27 across the
   plus-side interval. Frozen density carries `73.8%` of the fine-drift L1
   norm. Inside that block, fixed-center-density PCM carries `67.6%`,
   reaction-map-through-MACE carries `31.9%`, and the fixed-center-field
   MACE-minus-gas and CDS terms retain approximately second-order behavior.
   The valid bounded label is therefore
   `active-set-associated-explicit-continuum-geometry-response`.

   This is association, not proof that an active-set change alone causes the
   failed refinement. pyddx 0.8.0 does not expose a provider-consistent
   operator-only versus cavity-only split. Do not retune this gate, add another
   finer point, alter mixing/radii, or hide the result behind a local switching
   patch.

   The next bounded stage must pre-register an upstream-backed smooth
   cavity/operator profile and require:

   1. the exact same scalar-energy, reaction-map, adjoint, coordinate-VJP, and
      CDS definitions within one provider profile;
   2. current-runtime same-geometry energy closure against its own independent
      oracle;
   3. smooth local Cartesian/torsional refinement with stable surface topology
      and explicit translation/rotation checks.

   The existing optional PySCF SWIG/IEFPCM adapter is a research control, not
   an automatic repair: its fixed laboratory-frame Lebedev quadrature already
   shows nonmonotonic rotation residuals. A molecule-following,
   derivative-complete quadrature or a published analytic-force
   domain-decomposition provider remains a candidate only after its own
   separately locked feasibility and derivative profile passes.

   The historical local two-torsion closed loop generated at `5d69ef4` must
   not be promoted or rerun as a production gate until the smooth-provider
   stage passes. The same applies to second-molecule force, broader relaxed
   paths, short NVE, optimization, scan, TS search, and MD.

## Stop condition for the force milestone

Route 2 becomes a MAPLE solution-phase PES only when one fixed public profile
has:

- a documented stationary or adjoint derivative;
- differentiable PCM and CDS terms with no omitted geometry response;
- analytic-force agreement with converged central finite differences;
- translation, rotation, continuity, and energy-conservation evidence; and
- fail-closed behavior when the selected cavity loses differentiability or
  numerical validity.
