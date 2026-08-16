# Pure MACE-POLAR frozen-source ddX/SMD PES

## Scope

[`mace_polar_frozen_ddx.py`](../../maple/solvation/experimental/mace_polar_frozen_ddx.py)
is the first callable Route-2 vNext path that is both:

1. **pure MACE-POLAR** (no MACE-MDP permanent source); and
2. defined by one explicit geometry scalar with E/F/molecular-virial/HVP/H access.

It is an experimental execution surface, not a release or chemical-accuracy admission.
The callable API is intentionally open so derivative and topology failures can be
found by using the real model rather than by keeping E/F/H permanently unavailable.

## Scalar

Let `c0(R)` be the unmodified zero-field eight-channel radial-GTO source from the
official MACE-POLAR adapter.  For an additive differentiable solvent term `G_CDS`,

\[
E(\mathbf R)=E_\mathrm{vac}^{\mathrm{MACE\mbox{-}POLAR}}(\mathbf R)
+G_\mathrm{ddX}[\mathbf R,c_0(\mathbf R)]
+G_\mathrm{CDS}(\mathbf R).
\]

This route has no coupled electronic/continuum root and makes no common variational
functional claim.  It reuses:

- the official MACE-POLAR checkpoint and existing radial-GTO model adapter;
- `pyddx==0.8.0` for ddPCM energy, adjoint source derivative, and nuclear force terms;
- the existing differentiable Fibonacci/SWIG-inspired SMD-water CDS implementation;
- the existing scalar Richardson force auditor, extended with reusable HVP/Hessian
  differentiation of conservative forces.

## Complete force ledger

The implemented force is

\[
\mathbf F = \mathbf F_\mathrm{vac}
-\left.\partial_{\mathbf R}G_\mathrm{ddX}\right|_{c_0}
-\left(\frac{\partial c_0}{\partial\mathbf R}\right)^\mathsf T
  \frac{\partial G_\mathrm{ddX}}{\partial c_0}
-\partial_{\mathbf R}G_\mathrm{CDS}.
\]

The first ddX term moves the atom-centred cavity and Gaussian source centres while
holding coefficients fixed.  The second is the MACE-POLAR source-coordinate VJP
against the ddX reaction field.  Each leaf is stored separately before summation,
preventing a missing or double-counted coordinate contribution.

`RadialGTODDXBackend.fixed_source_coordinate_gradient()` now exposes the public
same-scalar ddX partial.  `build_state_with_fixed_source_coordinate_gradient()`
reuses one ddX solve for state and derivative.  Every state also carries a discrete
exposed-Lebedev-node topology hash, so numerical F/H stencils fail closed when the
**observable ddX** active cavity changes even if the exposed-node count is unchanged.
This guard says nothing about topology internal to another library unless that
library exposes a corresponding fingerprint.

## V and H definitions

- **V** means a molecular virial, not the legacy Route-2 `V=variational` tier and
  not volume-normalized periodic stress.  For declared origin `o`,
  `W = F.T @ (R-o)` and `dE/dstrain = -W`.  Raw and symmetric tensors, net force,
  origin, and antisymmetry are all retained.
- **HVP/H** are fourth-order Richardson derivatives of the same conservative
  force.  They retain step sizes, local truncation estimates, displaced force
  hashes, topology identity and observation coverage, derivative-policy SHA256,
  topology retry count, raw Hessian, symmetric Hessian, and raw antisymmetry.
  Symmetrization is therefore visible rather than hidden.

Topology observation is explicit.  The smooth in-tree CDS term reports complete
coverage.  The official PySCF SMD-CDS adapter reports its libsolvent internal
surface as unobservable because `get_cds_legacy` exposes energy and gradient but
not an active-set/surface fingerprint.  A ddX + PySCF-SMD state therefore has
partial coverage: ddX exposed-node changes remain guarded, while the internal CDS
surface does not.  Richardson F/H reject partial or unobservable coverage by
default.  An explicitly selected
`observed-components-only-experimental-v1` policy permits numerical exploration,
but the result is labelled `partial-experimental`; it is not topology fail-closed
and does not admit Hessian, frequency, optimization, or MD workflows.

## Current real-checkpoint evidence

The reproducible runner
[`run_mace_polar_frozen_ddx_water_canary.py`](../../tools/route2_release/run_mace_polar_frozen_ddx_water_canary.py)
produced
[`mace-polar-frozen-ddx-water-derivative-canary-v1.json`](evidence/mace-polar-frozen-ddx-water-derivative-canary-v1.json)
with the official MACE-POLAR-1-M checkpoint, CUDA float64 conversion,
`pyddx==0.8.0`, water ddPCM/194, and a **302-point low-cost smooth CDS canary**.

All five canary gates passed:

- total zero-field source charge: `1.39e-17 e`;
- maximum net-force component: below `1.6e-14 eV/A`;
- best analytic-force versus total-scalar directional error:
  `4.49e-7 eV/A`;
- best virial versus homogeneous-strain scalar error: `4.81e-7 eV`;
- Richardson HVP error estimate: `7.44e-5 eV/A^2`.

The raw molecular-virial antisymmetry remains measured rather than concealed; the
finite laboratory-fixed ddX grid is not structurally SO(3)-equivariant.

## Accuracy boundary

These derivative checks do **not** establish `MAE <= 1.5 kcal/mol`.  Existing
MNSol/FreeSolv pilot values bind different evaluator, continuum, cavity, CDS, or
record identities and cannot be transferred to this profile.  In particular, the
302-point smooth CDS canary is not the production accuracy identity.  A frozen
all-record benchmark must rerun this exact checkpoint/evaluator + radial-GTO ddX
+ selected CDS configuration before accuracy admission.

Energy MAE also does not bound force, virial, or Hessian errors.  Those require
their own finite-difference, covariance, topology, frequency, optimization, and MD
panels.

## Literature and upstream boundaries

- Differentiating a converged non-variational fixed point is mathematically valid
  under smoothness, local uniqueness, and nonsingular residual Jacobian conditions;
  see Christianson,
  [*Reverse accumulation and attractive fixed points*](https://doi.org/10.1080/10556789408805572)
  and [*Reverse accumulation and implicit functions*](https://doi.org/10.1080/10556789808805697).
  This frozen-source route is simpler because it has no coupled fixed point.
- ddCOSMO analytic first derivatives and adjoints are documented by Lipparini et al.,
  [JCTC 2013](https://doi.org/10.1021/ct400280b); the domain-decomposition model is
  described in [JCP 2013](https://doi.org/10.1063/1.4816767).  The public ddX 0.8
  interface supplies energy and force ingredients, not a nuclear Hessian API.
- MACE `v0.3.16` exposes PolarMACE energy/forces/stress and autograd Hessian support
  for its own model energy; see the
  [official release](https://github.com/ACEsuit/mace/releases/tag/v0.3.16) and
  [PolarMACE guide](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html).
  Those upstream derivatives do not automatically differentiate the added ddX/CDS
  scalar, which is why Route-2 owns the explicit combined chain rule and numerical
  HVP/H audit.
