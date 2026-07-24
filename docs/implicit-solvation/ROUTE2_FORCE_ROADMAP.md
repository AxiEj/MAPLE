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

1. **Stationarity is not established.** The code converges a fixed-point map
   between learned density coefficients and PCM reaction field. It has not yet
   proved that the reported energy is stationary with respect to both the
   learned representation and ASC. QM-SCF Hellmann--Feynman cancellations
   therefore cannot simply be assumed.
2. **The MACE local-field path is detached.** `polar_state()` converts the PCM
   potential and gradient from NumPy into a fresh tensor. MACE can return a
   partial force at fixed local field, but the present path does not retain the
   graph needed for the field and density response.
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
   introduce an energy/force discontinuity. A force implementation needs one
   predetermined smooth cavity policy (or a branch frozen for the entire
   trajectory), not per-geometry warning selection.

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

## Implementation sequence

### Phase 0 -- lock the energy functional

1. Expose a torch-native local-field evaluation without NumPy detachment.
2. Differentiate MACE energy with respect to local potential/gradient and test
   its sign, units, Cartesian/e3nn permutation, and conjugacy to the returned
   density coefficients.
3. Verify that the reported energy is independent, within tolerance, of SCF
   mixing and convergence path.
4. Decide from those tests whether Route 2 has a stationary joint functional
   or requires an adjoint fixed-point derivative. Do not implement production
   force before this decision.

### Phase 1 -- differentiable explicit geometry terms

1. Return the gas and polarized MACE partial forces at fixed local field.
2. Add analytic derivatives for the point-multipole MEP and ASC back-projection
   kernels.
3. Select an established PCM derivative provider or extend a documented
   upstream interface to supply boundary-operator and cavity-geometry
   derivatives. Do not silently approximate these terms as zero.
4. Replace the hard-visibility CDS area with an analytic/differentiable
   SMD-compatible surface-area implementation and differentiate the published
   geometry-dependent atomic tensions.
5. Replace per-geometry warning fallback with a force-compatible cavity policy.

### Phase 2 -- coupled response

1. Form the converged residual for learned density plus PCM response.
2. Implement Jacobian-vector and vector-Jacobian products without assembling a
   dense molecular Jacobian.
3. Solve the adjoint equation to a tolerance tighter than the energy SCF
   tolerance.
4. Sum gas MLIP force and every solvent derivative into
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

