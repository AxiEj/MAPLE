# Route-2 force roadmap

## Scope

Route 2 is the Research/Innovation Route for self-consistent polarizable
MLIP--PCM/SMD coupling:

\[
\Delta G_{\mathrm{solv}}
=\Delta E_{\mathrm{solute}}+U_{\mathrm{pol}}+G_{\mathrm{CDS}}.
\]

The official MACE-POLAR-1-M checkpoint remains unchanged. FreeSolv is a
secondary energy diagnostic, not the definition of the route. The current
implementation is a fixed-conformer energy proof-of-concept and must not be
described as a complete solution-phase PES.

## Current force blockers

1. **Stationarity is not established, and the simple conjugacy shortcut is
   rejected.** The code converges a fixed-point map between learned density
   coefficients and PCM reaction field. A graph-preserving probe confirms that
   the intrinsic MACE energy derivative with respect to the injected local
   potential/gradient is not the returned charge/dipole density. QM-SCF
   Hellmann--Feynman cancellations therefore cannot be imported into Route 2.
2. **Only the fixed-geometry response graph is exposed.**
   `polar_output_torch()` retains the graph from local potential/gradient to
   both intrinsic energy and density coefficients. A hermitivized, fixed-cavity
   PCMSolver map now supplies matrix-free density-to-field JVP/VJP operations.
   Cavity geometry and boundary operators still cross the NumPy/C boundary and
   have no coordinate derivative.
3. **The PCMSolver binding is energy-only.** The v1.1.12-style C ABI loaded by
   MAPLE exports cavity centres/areas, ASC, response ASC, and polarization
   energy, but no nuclear-gradient or boundary-operator derivative endpoint.
4. **The current CDS area is not differentiable.** The deterministic
   Shrake--Rupley implementation counts hard visible/occluded spherical points.
   Its area is piecewise constant and jumps when a grid point crosses an
   occlusion boundary. It is suitable for energy controls, not analytic force.
5. **Warning-triggered cavity switching is not a PES rule.** Retrying a single
   point with different GePol parameters removes known numerical warnings, but
   a geometry-dependent switch between primary and fallback cavities would
   introduce an energy/force discontinuity. Route 2 therefore keeps
   `cavity_policy=warning-fallback` as the default fixed-conformer energy
   policy and exposes a separate
   `cavity_policy=fixed-stability-branch` research policy. The latter selects
   `AREA=0.28 A^2, MINRADIUS=0.30 A` before the calculation and fails closed
   on any warning, so it removes policy-level geometry branching. Those two
   values are an engineering stability candidate, not theoretical constants;
   GePol topology continuity and boundary/operator derivatives remain
   unproven. A force implementation still needs one validated smooth cavity
   construction (or a branch frozen for the entire trajectory), not merely a
   predetermined parameter pair.

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
in turn calls the compiled `libsolvent.mnsol_interface_`. MAPLE will not copy
that code into its current source tree. CDS must remain an independently
selected, license-compatible differentiable provider.

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
6. **Diagnostic implemented; production replacement remains open:** a
   separately named Fibonacci-grid, SWIG-inspired area supplies an analytic
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
   force.

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
5. **Pending:** contract the adjoint with every remaining coordinate term and
   sum gas MLIP force plus all solvent derivatives into
   `SolvationResult.forces_hartree_per_angstrom`; only then advertise
   `supported_properties={"energy", "forces"}`.

### Phase 3 -- verification gates

1. Use whole-energy central finite differences only as an oracle. Demonstrate
   step-size convergence on small rigid molecules before testing flexible
   molecules.
2. Compare component-resolved and total analytic derivatives against the
   oracle; a total-force match alone must not hide cancelling component errors.
3. Check zero net force under translation and zero net torque under rotation.
4. Check energy continuity across small geometry displacements and reject any
   cavity-topology or fallback-branch jumps.
5. Run closed displacement loops and short NVE tests before enabling
   optimization, scan, TS search, or MD.

## Stop condition for the force milestone

Route 2 becomes a MAPLE solution-phase PES only when one fixed public profile
has:

- a documented stationary or adjoint derivative;
- differentiable PCM and CDS terms with no omitted geometry response;
- analytic-force agreement with converged central finite differences;
- translation, rotation, continuity, and energy-conservation evidence; and
- fail-closed behavior when the selected cavity loses differentiability or
  numerical validity.
