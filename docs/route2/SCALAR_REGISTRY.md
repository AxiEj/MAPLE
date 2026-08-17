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

## `route2-diagnostic-aimnet2-geometry-mediated-ddx-ddpcm-electrostatic-v1`

- Exact scalar:

  \[
    E(R)=E_{\mathrm{AIMNet2}}(R)
      +\tfrac12\langle c(R),P_Rc(R)\rangle_Q,
    \qquad c(R)=[q_{\mathrm{AIMNet2}}(R),0,0,0].
  \]

- Implementation entry point:
  `maple.solvation.coupling.geometry_mediated:GeometryMediatedElectrostaticScalar`.
- State equation: direct geometry-mediated source map
  `route2-geometry-mediated-source-map-v1`; there is no fixed-geometry
  electronic self-consistency variable.
- Included: the checkpoint-bound AIMNet2 vacuum energy, model-native
  coordinate response of its NQE charges, and ddPCM electrostatics through the
  existing pyddx forward, adjoint, and coordinate-VJP implementation.
- Excluded: continuum-field-conditioned AIMNet2 energy or charge response,
  mutual electronic polarization, CDS/nonpolar terms, thermochemical and
  standard-state corrections, and every Hessian/FREQ/TS/MD claim.
- Derivative: the gradient of the same scalar, split into AIMNet2 intrinsic,
  ddPCM fixed-source coordinate, and AIMNet2 charge-position response terms.
  Fixed-geometry direct/adjoint agreement at the physical source is not treated
  as sufficient: fixed-total-charge random bilinear reciprocity, apply/adjoint
  dot products, three charge-direction finite differences, and `J_q^T 1=0`
  are checked in the registered metric before assembly.
- Profile:
  `route2-profile-diagnostic-aimnet2-geometry-mediated-ddx-ddpcm-electrostatic-v1`.
- Capability/evidence: none / negative local admission canaries. The hash-bound
  real-water metric/gauge audit passes. In the source-bound float64 arm, the
  pinned adaptive derivative converges and agrees with the analytic gradient,
  but the version-pinned ddX reconstruction gives only
  `5.826261813812086e-9 A` of center active-set clearance, sampled displacements
  change cavity stratum, and all three frozen rotations change the exposed
  laboratory-grid active set
  ([bundle](evidence/aimnet2-geometry-mediated-pyddx-adaptive-water-a816f733/README.md)).
  The legacy float32 graph separately lacks an admissible fixed-step numerical
  window. Unresolved checkpoint release identity, absent measured pyddx
  residual, omitted nonpolar physics, and absent domain/loop/HVP/NVE validation
  keep `E/F/H/V/M` false.
- Full contract and literature boundary:
  [`AIMNET2_GEOMETRY_MEDIATED.md`](AIMNET2_GEOMETRY_MEDIATED.md).

## `route2-diagnostic-aimnet2-geometry-mediated-smoothharmonicgalerkin-cpcm-electrostatic-v1`

- Exact scalar:

  \[
    E(R)=E_{\mathrm{AIMNet2}}(R)
    -\tfrac12(S_Rc(R))^T A_R^{-1}(S_Rc(R)),
    \quad A_R=E_R^TK_RE_R,\quad S_R=E_R^TV_{\rm point,R}.
  \]

- Model/state: the same unmodified, field-independent AIMNet2
  `c(R)=[q_NQE(R),0,0,0]` and direct geometry-mediated source-map identity as
  the pyddx arm; no fixed-geometry electronic variable exists.
- Continuum: the existing smooth weighted harmonic exposure, Coulomb
  single-layer, rank/SPD gates, and sealed stationary scalar, specialized only
  by the analytic point-monopole Laplace-addition-theorem source map.
- Included: geometry-dependent AIMNet2 charge chain rule, exact-adjoint
  harmonic receiver, measured dense stationarity residual, and structural
  `SO(3)` coefficient intertwiners. The research-only second-order surface
  composes a parity-gated AIMNet2 `J_q h`, intrinsic `H_E h`, fixed-cotangent
  `D_R[J_q^T v][h]`, and the sealed continuum joint `(R,c)` HVP into the
  complete weak-scalar HVP.
- Excluded: finite-dielectric solvent parameterization, CDS/nonpolar and
  standard-state terms, fixed-geometry mutual polarization, and all public
  Hessian/FREQ/TS/MD claims.
- Precision/runtime boundary: the legacy TorchScript graph hard-casts
  coordinates to float32 and is retained as a negative control. The optional
  CPU research runtime rebuilds the official `aimnet==0.2.0` wB97M-D3 Python
  architecture, verifies its source/configuration hashes, loads unchanged
  checkpoint weights, and evaluates the same input/topology semantics in
  float64. It is not a public ASE calculator; its inherited public Hessian/HVP
  methods remain disabled, while the separate Route-2 second-order response is
  diagnostic-only.
- Capability/evidence: none. Synthetic scalar/derivative/symmetry gates pass;
  the real water rigid-rotation gate passes for both precision arms. The
  float32 directional/full-Cartesian gates fail, while the float64 local
  one-water gates pass with central refinement and point/source-shell plus
  sphere-tangency event guards. A fail-closed v2 17-shard H/C/N/O
  distorted-geometry contract now
  reuses MAPLE's frozen PES asset (three variants, three directions, three
  steps per molecule) and requires all raw shards for aggregation; S/Cl
  controls remain explicitly excluded by the local checkpoint domain. The
  historical v1 water shard remains immutable
  ([bundle](evidence/aimnet2-geometry-mediated-pes-water-5244de8c/README.md));
  water passes the v2 sphere-tangency contract in two clean processes
  ([v2 bundle](evidence/aimnet2-geometry-mediated-pes-water-v2-02c21b52/README.md)).
  Methanol reproducibly fails the bond-stretched point/source-shell event
  guard despite all numerical derivative gates passing
  ([negative bundle](evidence/aimnet2-geometry-mediated-pes-methanol-v2-02c21b52/README.md));
  the complete seventeen-shard panel has now been executed twice with all
  scientific measurement hashes replaying exactly
  ([negative aggregate](evidence/aimnet2-geometry-mediated-pes-panel-v2-cd8769d4/README.md)).
  Only water and hydrogen peroxide pass every v2 gate. Four molecules fail the
  point/source-shell guard, ethylamine fails the separate sphere-tangency
  guard, and fourteen molecules lack a demonstrated refinement/plateau window
  over the frozen three small coordinate steps. The current-profile full panel
  is therefore negative without changing any radius, step, or threshold.
  A water-only bidirectional force-work and straight-segment event harness is
  retained in two clean processes with identical raw records and measurement
  SHA; its reciprocity/charge-FD operands and segment certificates are
  independently recomputed
  ([bundle](evidence/aimnet2-geometry-mediated-water-loop-e0a347f5/README.md)).
  A separate complete-HVP water canary closes the four-term ledger, three-step
  charge-JVP/contracted-Hessian/total-gradient finite differences, bilinear
  symmetry, three translations, and local event guards in two identical clean
  processes
  ([bundle](evidence/aimnet2-geometry-mediated-hvp-water-2b119022/README.md)).
  A second source-bound water bundle uses a guarded SciPy hybrid root in exact
  internal coordinates, assembles the complete `9 x 9` Cartesian Hessian,
  closes all-column total-gradient finite differences over a four-step scan,
  and verifies three translations, three stationary rotations, and a
  correctly mass-weighted three-mode vibrational subspace in two identical
  clean processes
  ([bundle](evidence/aimnet2-geometry-mediated-frequency-water-f79d5051/README.md)).
  These isolate precision and close local water diagnostics but do not supply
  broad event-free `C2`, multi-stationary-point/FREQ/TS/IRC,
  physical-solvent, workflow, or release evidence.
  It remains a conductor reference, not an admitted water ddPCM model.
- Full contract and literature boundary:
  [`AIMNET2_POINT_HARMONIC.md`](AIMNET2_POINT_HARMONIC.md).

## `route2-diagnostic-aimnet2-geometry-mediated-smoothharmonicgalerkin-ddpcm-electrostatic-v1`

- Formula: `E_AIMNet2(R)-1/2 b(R)^T x(R)`, with `b=S(R)c_A(R)`,
  `M f=b`, `R_eps phi_eps=R_inf f`, and `A x=M phi_eps`.
- `R_eps=2*pi*(eps+1)/(eps-1) M-D` and `R_inf=2*pi M-D`; `D` is the
  outward-source-normal Laplace double-layer principal-value Galerkin matrix.
- Included: explicit AIMNet2 geometry-dependent charges, finite-dielectric PCM
  electrostatics, the transpose/KKT energy cotangent, full charge-chain rule,
  and structural harmonic SO(3) intertwiners. The generally nonsymmetric
  primal apparent-charge map is retained as diagnostic evidence, not exposed
  as the provider field.
- Excluded: fixed-geometry electronic mutual polarization, uniform COSMO
  dielectric scaling, CDS/nonpolar and standard-state terms, and all public
  E/F/H/V/M or task admission.
- This identity is explicitly a parameterized, permanently disabled diagnostic.
  The dielectric value is immutable configuration/provenance, not an adjustable
  threshold. Any future admission requires a distinct solvent-bound profile and
  evidence binding the exact dielectric and full continuum configuration SHA.
- Two clean processes of the source-bound float64 AIMNet2 water canary now
  reproduce scientific SHA256
  `7ec7732e79aacfd1d9502d7c19de8fd49b0b1975b764ef1cf59a413eaaf5cbbf`.
  Its local same-scalar force, complete Cartesian refinement, rigid rotations,
  stationarity/residuals, reciprocity, gauge, and event-clearance gates pass.
  This is implementation evidence for the finite-dielectric branch, not a
  solvent-bound identity or capability admission
  ([bundle](evidence/aimnet2-geometry-mediated-harmonic-ddpcm-water-4dca73d7/README.md)).
- Full equation, literature, residual, topology, and claim boundary:
  [`AIMNET2_POINT_HARMONIC_DDPCM.md`](AIMNET2_POINT_HARMONIC_DDPCM.md).

## `route2-candidate-aimnet2-frozen-charge-water-smoothharmonicgalerkin-ddpcm-electrostatic-v1`

- Formula: the same finite-dielectric ddPCM scalar above, but with water
  `epsilon=78.355`, SMD water Coulomb radii, transition width `0.18 A^2`,
  surface/exposure orders `1/2`, and both radial quadrature orders `32` bound
  into one immutable continuum configuration contract.
- Model profile: `aimnet2-polarizable-v1`, separately versioned from MACE and
  from the earlier neutral-HCNO diagnostic. Here "polarizable" identifies the
  NQE checkpoint family; execution remains one-shot and field-independent at
  each geometry. No continuum field is supplied to AIMNet2 and there is no
  electronic SCF loop.
- Included: unchanged source-bound float64 AIMNet2 checkpoint weights,
  geometry-dependent NQE monopoles, exact water finite-dielectric harmonic
  ddPCM electrostatics, and the complete same-scalar coordinate chain rule.
- Excluded: CDS/nonpolar and standard-state terms, chemical-accuracy claims,
  and every public E/F/H/V/M or OPT/FREQ/TS/IRC/MD capability.
- This profile is registered but disabled. Two clean processes under its exact
  identity reproduce scientific SHA256
  `29ad72cd84a137b6bc7b9079ce6979000e7dab58983e2c41d6d2e3ce82fa08e5`
  and pass the local water derivative, stationarity, symmetry, topology, and
  event gates
  ([bundle](evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-0c19ede4/README.md)).
  The subsequent dual-process 17-molecule distorted-PES panel passes every
  frozen convergence check and 12/17 complete molecule gates; methanol,
  methane, dimethyl ether, and acetic acid remain below the point/source-shell
  guard, while ethylamine remains below the sphere-tangency guard
  ([bundle](evidence/aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-smoothed-97efb08e/README.md)).
  A distinct ten-record MNSol pilot provides aggregate-only early accuracy
  evidence for fixed electrostatics plus SMD-CDS, not for this `G_np=0`
  harmonic scalar
  ([bundle](evidence/aimnet2-frozen-charge-mnsol-pilot-replay-b758aede/README.md)).
  Finally, water-only loop, complete-HVP, and stationary dense-Hessian/frequency
  diagnostics each pass two clean processes under the exact finite-dielectric
  identity
  ([bundle](evidence/aimnet2-frozen-charge-water-ddpcm-daily-tasks-76d4d097/README.md)).
  These results improve implementation and task-level diagnostic coverage but
  open no capability because the full force domain remains topology-negative,
  broad event-free `C2`, nonpolar, and public workflow evidence remain absent.

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

## `route2-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`

- Operational root:

  \[
    c^*=M_{\rm orig}^{\rm analytic}(R,u^*),
    \qquad
    u^*=\nabla_c^QG_{\rm harm}(R,c^*).
  \]

- Unique scalar:

  \[
    E_{\rm op}(R)=E_{\rm vac}^{\rm analytic}(R)
      +G_{\rm harm}(R,c^*(R)),
    \qquad
    G_{\rm harm}=-\tfrac12(Sc)^TA^{-1}(Sc)
      =\tfrac12\langle c,u\rangle_Q.
  \]

- Model identity: the separately content-addressed analytic isotropic Gaussian
  multipole evaluator with the original learned four-channel density head.
  Those four outputs occupy only the `sigma=1.5` source block; the second
  radial block is not invented and the density head is not called a
  field-energy gradient.
- Continuum/cavity identity: the smooth weighted-overlap harmonic Galerkin
  scalar with fixed complete irrep dimensions and no laboratory-fixed surface
  grid. Its drive and coordinate pullbacks are generated from the same scalar.
- Excluded: the checkpoint field-conditioned energy difference, the changed
  eight-channel variational effective source, sharp-union identity, CDS, and
  every public workflow.
- Derivative: the existing operational implicit adjoint differentiates this
  exact scalar along the constrained root. The builder is deliberately named
  `build_disabled_operational_electrostatic_scalar` and rejects admitted
  registry entries.
- Profile:
  `route2-profile-operational-macepolar-analytic-gaussian-multipole-smoothharmonicgalerkin-cpcm-v1`.
- Evidence: a source-bound official-checkpoint water canary passes cold/warm
  root replay, the exact half-coupling identity, three re-solved operational
  force finite differences, translation/torque checks, and one rigid rotation.
  The harmonic continuum rotates at float64 roundoff; the complete operational
  scalar has `2.77e-9 eV` rotation-energy error and `7.86e-8` relative
  force-covariance error. Both processes reproduce the same scientific digest
  under
  [`evidence/operational-analytic-harmonic-water-fa6f0200/`](evidence/operational-analytic-harmonic-water-fa6f0200/README.md).
  Separately preregistered methanol and ethanol shards then pass cold/warm
  replay, translation, identical-atom permutation, and three rigid rotations
  each in two clean processes. Their maximum rotation-energy errors are
  `2.8617250791285187e-9 eV` and `1.9072103896178305e-9 eV`; their maximum
  relative force-covariance errors are `5.000913075374153e-8` and
  `4.7353129839838976e-8`, respectively. Evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-methanol-93c98598/`](evidence/operational-analytic-harmonic-rigid-methanol-93c98598/README.md)
  and
  [`evidence/operational-analytic-harmonic-rigid-ethanol-538f9f4d/`](evidence/operational-analytic-harmonic-rigid-ethanol-538f9f4d/README.md).
  The subsequent acetone shard passes the same local gates in two clean
  processes; evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-acetone-371a2b60/`](evidence/operational-analytic-harmonic-rigid-acetone-371a2b60/README.md).
  The acetonitrile shard is the fifth clean two-process local pass; evidence is
  retained under
  [`evidence/operational-analytic-harmonic-rigid-acetonitrile-cf43e050/`](evidence/operational-analytic-harmonic-rigid-acetonitrile-cf43e050/README.md).
  Water was also repeated under this exact three-rotation contract; together
  these records close the contiguous frozen range `[0,5)` in two clean
  processes. The water shard is retained under
  [`evidence/operational-analytic-harmonic-rigid-water-bea47120/`](evidence/operational-analytic-harmonic-rigid-water-bea47120/README.md).
  Benzene extends the contiguous two-process coverage to `[0,6)`; evidence is
  retained under
  [`evidence/operational-analytic-harmonic-rigid-benzene-97552baa/`](evidence/operational-analytic-harmonic-rigid-benzene-97552baa/README.md).
  Methane extends the contiguous two-process coverage to `[0,7)`; evidence is
  retained under
  [`evidence/operational-analytic-harmonic-rigid-methane-817501b5/`](evidence/operational-analytic-harmonic-rigid-methane-817501b5/README.md).
  Trans-butane extends the contiguous two-process coverage to `[0,8)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-trans-butane-77abe42b/`](evidence/operational-analytic-harmonic-rigid-trans-butane-77abe42b/README.md).
  Dimethyl ether extends the contiguous two-process coverage to `[0,9)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-dimethyl-ether-8a3aecce/`](evidence/operational-analytic-harmonic-rigid-dimethyl-ether-8a3aecce/README.md).
  Formic acid extends the contiguous two-process coverage to `[0,10)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-formic-acid-cd734f73/`](evidence/operational-analytic-harmonic-rigid-formic-acid-cd734f73/README.md).
  Acetic acid extends the contiguous two-process coverage to `[0,11)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-acetic-acid-c51d7154/`](evidence/operational-analytic-harmonic-rigid-acetic-acid-c51d7154/README.md).
  Acetaldehyde extends the contiguous two-process coverage to `[0,12)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-acetaldehyde-857af1b8/`](evidence/operational-analytic-harmonic-rigid-acetaldehyde-857af1b8/README.md).
  Acetamide extends the contiguous two-process coverage to `[0,13)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-acetamide-ab1c7799/`](evidence/operational-analytic-harmonic-rigid-acetamide-ab1c7799/README.md).
  Ethylamine extends the contiguous two-process coverage to `[0,14)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-ethylamine-4b40b81a/`](evidence/operational-analytic-harmonic-rigid-ethylamine-4b40b81a/README.md).
  Pyridine extends the contiguous two-process coverage to `[0,15)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-pyridine-ce23da3e/`](evidence/operational-analytic-harmonic-rigid-pyridine-ce23da3e/README.md).
  Nitromethane extends the contiguous two-process coverage to `[0,16)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-nitromethane-303af2d9/`](evidence/operational-analytic-harmonic-rigid-nitromethane-303af2d9/README.md).
  Hydrogen peroxide extends the contiguous two-process coverage to `[0,17)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-hydrogen-peroxide-49d207cd/`](evidence/operational-analytic-harmonic-rigid-hydrogen-peroxide-49d207cd/README.md).
  Thiophene extends the contiguous two-process coverage to `[0,18)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-thiophene-b8c7ac21/`](evidence/operational-analytic-harmonic-rigid-thiophene-b8c7ac21/README.md).
  Methanethiol extends the contiguous two-process coverage to `[0,19)`;
  evidence is retained under
  [`evidence/operational-analytic-harmonic-rigid-methanethiol-ad0e3b68/`](evidence/operational-analytic-harmonic-rigid-methanethiol-ad0e3b68/README.md).
  Chloroform completes the frozen equilibrium range `[0,20)`; evidence is
  retained under
  [`evidence/operational-analytic-harmonic-rigid-chloroform-920c1b0f/`](evidence/operational-analytic-harmonic-rigid-chloroform-920c1b0f/README.md).
  This supports replacement of the finite laboratory-grid route; it does not
  rehabilitate that route or establish a global/profile-wide guarantee.
- Capabilities: none. This is not physical-component, solvation-accuracy,
  PES/domain, Hessian/FREQ/MD, or release evidence; `E/F/H/V/M` remain false.

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
  disabled until every strict common-variational gate passes. For the current
  official MACE-POLAR-1-M checkpoint, the instantiation that retains both the
  original intrinsic field-conditioned energy and the original four-channel
  source is formally ruled out by the source-bound real-checkpoint
  counterexample in
  [`evidence/mace-conjugacy-nogo-d17c35ac/`](evidence/mace-conjugacy-nogo-d17c35ac/README.md).
  This negative result does not transfer capability to a changed-source model.

## `route2-variational-macepolar-energygradient-fixedcavity-cpcm-v1`

- Stationary scalar:

  \[
    L_s(R,c,u)=E_{\rm anc}(R,u)-s\langle c,u\rangle_Q+sG_{\rm cpcm}(R,c),
  \]

  with `G_np=0`, a complete eight-channel energy-gradient effective source,
  and the fixed reciprocal C-PCM drive generated from the continuum scalar.
- Implemented: immutable scalar-to-response adapters; the constrained
  fixed-charge state `c=M_E(R,u)`, `u=grad_Q G(R,c)`; exact reduced JVP/VJP and
  coordinate pullbacks; deterministic cold/warm roots; stationary ledger; and
  a fixed-chart envelope-gradient diagnostic. Synthetic integration tests use a
  same-scalar electronic oracle plus the real fixed C-PCM backend. A separate
  official-checkpoint water canary now converges the changed-source common
  state and matches one stationary-envelope coordinate direction at three
  displacement sizes; its source/runtime-bound record is
  [`evidence/variational-common-water-576550e9/`](evidence/variational-common-water-576550e9/README.md).
- Excluded: the original four-channel density head as variational source,
  source-dependent cavities, nonpolar terms, and every public workflow.
- Capabilities/admission evidence: none / none. The linked canary is diagnostic
  implementation evidence only. It covers one geometry and one direction using
  a laboratory-grid continuum without structural global `SO(3)` covariance.
  Passivity, root uniqueness over a declared real-model domain,
  combined-Hessian stability, harmonic coordinate assembly, rotation, full
  release panels, and chemical validation remain open. Therefore `E/F/H/V/M`
  all remain false.

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
  action, and same-scalar drive/HVP/JVP/VJP generation from externally frozen
  `A` and `S` matrices.
- Implemented in addition: the same disabled common-stationarity kernel can
  eliminate the electronic/continuum state for a fixed external harmonic
  snapshot. This is covered only by a synthetic electronic oracle.
- Excluded: geometry-dependent harmonic assembly and its coordinate pullback,
  source-dependent cavity, and every nonpolar term.
- Scientific boundary: this fixed external coefficient snapshot is a common-
  state plumbing target. Its zero continuum coordinate partial is not a moving-
  cavity force.
- Profile:
  `route2-profile-variational-macepolar-energygradient-fixedcavity-harmonicgalerkin-cpcm-v1`.
- Capabilities/evidence: none / none. The entry point remains explicitly
  disabled. Its complete eight-channel energy-gradient effective source is a
  new model identity rather than the original four-channel density head. It
  still requires every real-checkpoint, physical, rotation, and release gate.

## `route2-variational-macepolar-energygradient-smoothharmonicgalerkin-cpcm-v1`

- Stationary continuum scalar:

  \[
    G(R,c)=-\frac12(S(R)c)^T A(R)^{-1}(S(R)c),
    \quad A=E^TKE,\quad S=E^TV.
  \]

- Cavity/operator identity: smooth weighted-overlap harmonic coefficients,
  rectangular product embedding `E`, physical-shell Coulomb `K`, and complete
  two-width eight-channel Gaussian source map `V`. The receiver is generated
  only from the same scalar.
- Derivative route: `harmonic_torch_functional` reassembles the complete
  `R->E,K,V->G` path inside one sealed Torch graph. Source response, fixed-
  source coordinate partial, and mixed drive/coordinate pullback cannot be
  overridden independently.
- Executed synthetic evidence: matrix-by-matrix parity with the independent
  NumPy/SciPy reference; source and coordinate directional derivatives; mixed
  pullback; rotation/translation/permutation covariance; exact polar-axis
  derivative regression; and one re-solved synthetic common-state envelope
  finite difference.
- Scientific boundary: this is a regularized weighted multi-shell conductor
  candidate, not the exact sharp union boundary or production ddPCM/ddCOSMO.
  It is `C1` but not generally `C2` at shell tangency.
- Profile:
  `route2-profile-variational-macepolar-energygradient-smoothharmonicgalerkin-cpcm-v1`.
- Capabilities/evidence: none / negative rotation evidence. The original
  molecular-realspace checkpoint profile passes one root/replay/envelope water
  canary, and its harmonic continuum rotates at roundoff, but the full scalar
  fails because the pinned fixed-axis model evaluator is not an exact `SO(3)`
  intertwiner. Passivity, uniqueness, combined-Hessian/domain gates, physical
  calibration, and all release panels also remain open.

## `route2-variational-macepolar-analytic-gaussian-multipole-energygradient-smoothharmonicgalerkin-cpcm-v1`

- Model scalar: separately identified anchored MACE-POLAR field-energy graph
  using analytic isotropic Gaussian `l<=1` molecular real-space feature and
  energy primitives. Checkpoint bytes are unchanged, but inference is changed.
- Model source: complete eight-channel derivative of that same scalar. The
  original four-channel density head is only the zero-field anchor and a
  diagnostic observable.
- Continuum scalar: the same smooth weighted harmonic-Galerkin
  `G(R,c)=-1/2 (S(R)c)^T A(R)^-1(S(R)c)` implementation, explicitly bound to
  this combined scalar identity in its content hash.
- Structural scope: the replaced long-range pair primitives and the harmonic
  continuum coefficient assembly are exact finite `SO(3)` intertwiners in
  exact arithmetic. The complete checkpoint graph is not globally certified by
  this statement.
- Executed evidence: clean official-checkpoint water root/cold replay, three
  re-solved envelope finite differences, and one rigid rotation; two processes
  reproduce scientific measurement SHA-256
  `7ce9e9c07f40552ea513e0bbd4f29f4e4a5aa6887d7b5888c5647756525fe500`.
  See
  [`evidence/variational-analytic-harmonic-water-50809803/`](evidence/variational-analytic-harmonic-water-50809803/README.md).
- Stability evidence: five fixed starts reproduce one local water root and
  `J_r=I-J_M H_G`, but the same-scalar model susceptibility is indefinite and
  singular. Passivity, local invertibility, feedback-sign, and combined-Hessian
  gates fail reproducibly. See
  [`evidence/variational-analytic-stability-water-604ecfa2/`](evidence/variational-analytic-stability-water-604ecfa2/README.md).
- Profile:
  `route2-profile-variational-macepolar-analytic-gaussian-multipole-energygradient-smoothharmonicgalerkin-cpcm-v1`.
- Capabilities/admission evidence: none / negative Tier-V stability evidence.
  The implementation canary is not checkpoint parity or physical validation;
  the later local spectrum is an explicit no-go for the current candidate on
  its declared reduced space, not a global operational-PES or accuracy result.
  `E/F/H/V/M` remain false.

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
