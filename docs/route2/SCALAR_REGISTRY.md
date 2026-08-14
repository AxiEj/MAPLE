# Route 2 scalar registry

This document mirrors the machine-readable registry in
`maple.solvation.api.scalar_registry`. Registration defines an identity; it
does **not** admit a capability. Every current `E/F/H/V/M` capability is false
and every scalar is disabled.

`maple.solvation.api.profiles.PROFILE_REGISTRY` is the sole admission registry.
Each immutable profile binds exactly one registered scalar to that scalar's
registered state equation and provider identities. A tier can be admitted only
when both scalar and profile are enabled, the profile tier is declared by the
scalar, and non-empty evidence artifact IDs are frozen into both registrations.
All current profiles are disabled, have no capabilities, and have empty
admission evidence. Multiple profiles may share one scalar formula while
binding different coupling or physical-continuum configuration contracts.

`Route2Result` accepts only a registered `profile_id`; it derives scalar ID,
state-equation ID, and capabilities from that profile. Callers cannot attach
capabilities or override scalar/state identity. Energy and force totals are
properties computed from immutable leaf components, never constructor inputs.
The closure tolerance used by future legacy adapters is an internal, versioned,
bounded constant rather than a caller-controlled value. A disabled profile can
produce only fail-closed internal energy evidence: it cannot publish a result,
declare an admitted domain, or carry force leaves.

Public ASE units are declared centrally as energy `eV`, forces `eV/A`, and
Hessian `eV/A^2`. Historical Hartree-reporting jobs use the dispatcher-bound
non-ASE compatibility view; no ASE `Calculator.results` stores Hartree values.

## `route2-operational-cpcm-fixedtopology-electrostatic-v1`

- Formula: `E_op(R)=Phi_op(R,y*(R))`, where
  `Phi_op=E_vac(R)+1/2<c_ref(R)+T(R)y,P_R(c_ref(R)+T(R)y)>_Q` and `G_np=0`.
- Implementation entry point:
  `maple.solvation.coupling.energy:OperationalElectrostaticScalar`.
- Included: vacuum energy; fixed-topology C-PCM half-coupling electrostatics.
- Excluded: field-conditioned model energy difference; SMD/CDS nonpolar energy.
- Source: atom-centred net monopoles plus real-spherical `l=1` dipoles.
- Field: positive energy-dual field with pairing `c^T Q(R) u`.
- Continuum/cavity/nonpolar: fixed-topology linear reciprocal C-PCM v1 /
  fixed-topology amplitude-SWIG v1 / none.
- State equation: `route2-constrained-mutual-polarization-root-v1`.
- Derivative: implicit-adjoint total derivative of the same scalar along the
  unique admitted root `y*(R)`.
- Capabilities/evidence: none / none.

### Water radial-GTO operational candidate

Profile
`route2-profile-operational-cpcm-fixedtopology-radialgto-electrostatic-v1`
uses the same scalar entry point and state equation, but binds all of the
following as one immutable identity:

- the physical two-width `(1.5, 3.0 Angstrom)` radial-GTO source/field space;
- exact discrete `B/B*` coupling under its authoritative pairing;
- dimensionless reduced-coordinate contract;
- water dielectric `78.39`;
- SMD-water Coulomb radii;
- a fixed 194-point Lebedev grid per atom.

It is implemented as an internal scalar/gradient candidate, not admitted as E
or F. A real water directional derivative passed locally, but the preregistered
rotation-force and torque gates did not both pass. Methane component evidence
also shows a large electrostatic-magnitude change relative to older,
nonconjugate point-source/GTO-receiver and local-jet profiles. Therefore this
profile remains disabled and cannot be published as a solution-phase PES.

Profile
`route2-profile-diagnostic-cpcm-injectedgrid-radialgto-electrostatic-v1`
uses the same radial algebra with an explicitly unbound injected surface. It is
only for deterministic synthetic tests and cannot satisfy the water profile's
physical-configuration identity.

## `route2-diagnostic-ddx-{ddpcm,ddcosmo}-radialgto-electrostatic-v1`

Profile `route2-profile-diagnostic-ddx-ddpcm194-radialgto-electrostatic-v1`
binds pyddx/ddX `0.8.0`, ddPCM, dielectric `78.39`, SMD-water Coulomb radii,
`lmax=8`, and 194 Lebedev points per sphere.  Its full eight-channel source is
mapped by one linear operator into both ddX source inputs:

\[
  c \longmapsto (\Psi,\Phi)=(C c,B c).
\]

`B` evaluates the two finite-width Gaussian monopole/dipole blocks on the
exposed cavity nodes.  `C` maps their exact integrated monopole/dipole moments
into ddX's local multipole RHS.  Thus the two widths remain distinct in `B`
and contribute additively to their matching local coefficients in `C`; this
does not by itself certify the physical adequacy of a non-compact Gaussian
density in ddX.  The only public response is the derivative of ddX's same
half-coupling scalar,

\[
  G(c)=\tfrac12\langle\Psi,x\rangle,
  \qquad
  \nabla_c G=\tfrac12(C^T x-B^T\xi),
\]

with ddX's forward solution `x` and adjoint cavity cotangent `xi`.  The
coordinate VJP is the polarization identity applied to the analytic
fixed-source derivative of that same scalar.  No second receiver model is
used, and the second radial block is not collapsed into the old four-channel
point-multipole adapter.

Local real-runtime checks close the half-coupling identity, source directional
derivative, JVP/VJP dot product, and coordinate directional derivative.  This
does **not** admit E or F: ddX `0.8.0` exposes only requested solver tolerance,
not a post-solve algebraic residual, and its finite laboratory-frame
Lebedev/active-set discretization is not structurally rotation equivariant.
Increasing the grid to 1202 points reduced but did not eliminate an independent
methanol rotation drift.  Therefore this is a disabled prerequisite/negative
canary, not the rotation-controlled continuum requested for release.  The
ddCOSMO equation has its own disabled scalar identity and is callable under a
different provider configuration, but it has no registered vNext PES profile
in this slice.  Neither scalar repairs the original four-to-eight-channel
MACE source/energy conjugacy obstruction.

Profile
`route2-profile-diagnostic-fixedbox40-cpcm590-radialgto-electrostatic-v1`
keeps the same scalar formula but binds a distinct experimental MACE-POLAR
fixed-40-A reciprocal evaluation operator and a 590-point Lebedev grid per
atom.  It was introduced after a fixed-order scan isolated the old rotation
error to the model/cavity evaluation operators.  This identity is disabled:
its original one-water geometry/orientation/path checks, 20-molecule
directional panel, 465-component reference-geometry Cartesian panel, and 20-reference
residual-refinement panel pass. A later molecule-ID-bound water rotation canary
from the frozen all-panel symmetry contract failed: maximum energy change was
`5.1167126003e-6 eV` and maximum relative force-covariance error was
`1.0764407452e-4`, against `1e-6 eV` and `1e-4`. The negative clean-tree
artifact is retained under
`evidence/fixedbox590-symmetry-negative-f15398b0/`. This profile therefore
cannot be Tier E or F.

The otherwise identical disabled `fixedbox32`, `fixedbox48`, and `fixedbox56`
profiles exist only for a preregistered box-operator convergence audit. They
bind distinct model/evaluator identities; they are not public alternatives and
cannot be selected adaptively.

Profile
`route2-profile-diagnostic-fixedbox48-cpcm1202-radialgto-electrostatic-v1`
is a separate disabled follow-up candidate. It binds the 48-A reciprocal model
operator and the fixed Lebedev-order-59, 1202-node-per-atom continuum contract.
It does not alter, rename, or rescue the failed CPCM590 profile, retains the
same scalar formula and frozen symmetry thresholds, and has no capability or
admission evidence.

The preregistered clean run passed its 48-to-56 tail thresholds and found the
40-Angstrom scalar within `2.14385e-7 eV` and force RMS within
`3.98917e-7 eV/Angstrom` of 56 Angstrom on equilibrium water. This is not a
public profile or a multi-geometry convergence certificate.

## `route2-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1`

- Formula: the same explicit vacuum plus reciprocal C-PCM half-coupling form,
  but its state equation is driven by the separately identified exterior
  local-`l<=1`-jet coupling.
- Purpose: internal numerical comparison with the historical fixed-cavity
  Route-2 implementation.
- Excluded: checkpoint-native exact-GTO coupling, field-conditioned model
  energy difference, and every nonpolar/CDS term.
- Profile: `route2-profile-diagnostic-localjet-cpcm-fixedtopology-electrostatic-v1`.
- Coupling: `maple.route2.coupling.exterior-local-l1-jet-diagnostic.v1`.
- Capabilities/evidence: none / none. It is deliberately disabled and is not a
  released PES, force, or complete solvation-free-energy method.

## `route2-operational-cpcm-fixedtopology-smdcds-v1`

- Formula: the preceding operational scalar plus
  `G_np^SMD-CDS(R,c_ref(R)+T(R)y)`.
- Included: vacuum energy; C-PCM half coupling; fixed-topology SMD-derived CDS.
- Excluded: field-conditioned model energy difference.
- Source, field, continuum, and cavity: as above.
- Nonpolar: fixed-topology SMD-derived CDS v1.
- State equation: `route2-constrained-mutual-polarization-root-v1`.
- Derivative: implicit-adjoint total derivative; disabled pending its own
  same-scalar CDS force and smoothness gate.
- Capabilities/evidence: none / none.

## `route2-variational-common-functional-v1`

- Formula: `F_var(R,c)=Gamma_theta(R,c)+G_pcm(R,c)+G_np(R,c)`, with
  `Gamma_theta=stat_u[E_theta(R,u)-<c,u>_Q]`.
- Included: electronic Legendre functional; continuum energy; the explicitly
  declared nonpolar energy.
- Excluded: any independently assembled response correction.
- Source/field convention: as above.
- State equation: `route2-common-functional-stationarity-v1`.
- Derivative: stationary envelope derivative.
- Capabilities/evidence: none / none. In particular `V=false`; this scalar is
  disabled until every strict common-variational gate passes.

## `route2-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1`

- Target continuum scalar:

  \[
    G(c)=-\frac12(Sc)^T A^{-1}(Sc),
  \]

  on fixed complete per-atom real-spherical-harmonic coefficient blocks.
- Receiver: the exact metric adjoint of the same source operator `S`; there is
  no independently coded response receiver.
- Implemented: immutable/content-addressed coefficient snapshots, symmetric
  positive-definite stationary solve, complete finite `SO(3)` coefficient
  action, rectangular smooth-exposure product embedding `E`, physical-shell
  Coulomb `K`, `A=E.T K E`, eight-channel Gaussian `S=E.T V`, and same-scalar
  drive/HVP/JVP/VJP generation. Nested, intersecting, tangent, and separated
  spheres are covered without a laboratory grid.
- Excluded: analytic coordinate pullback, combined electronic-continuum
  stationarity, source-dependent cavity, and every nonpolar term.
- Scientific boundary: this is a regularized weighted multi-shell conductor
  reference, not the exact sharp union boundary or a production ddPCM/ddCOSMO
  backend. It is structurally rotation covariant and `C1`, but not generally
  `C2` at shell tangency; no force/Hessian/frequency capability follows.
- Profile:
  `route2-profile-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1`.
- Capabilities/evidence: none / none. The entry point remains explicitly
  disabled until model-side conjugacy and every continuum/root/coordinate/
  rotation/release gate passes.

See [HARMONIC_GALERKIN_TIER_V.md](HARMONIC_GALERKIN_TIER_V.md) for the no-grid
decision and the moving-cavity contract split.

## Capability labels

| label | meaning | Phase-1 admission |
| --- | --- | :---: |
| E | scalar energy | no |
| F | conservative force of that scalar | no |
| H | force-derived Hessian/HVP | no |
| V | strict common variational functional | no |
| M | MD release gate | no |
