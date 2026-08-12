# Route 2 scalar registry

This document mirrors the machine-readable registry in
`maple.solvation.api.scalar_registry`. Registration defines an identity; it
does **not** admit a capability. In Phase 1 every `E/F/H/V/M` capability is
false and every scalar is disabled.

`maple.solvation.api.profiles.PROFILE_REGISTRY` is the sole admission registry.
Each immutable profile binds exactly one registered scalar to that scalar's
registered state equation and provider identities. A tier can be admitted only
when both scalar and profile are enabled, the profile tier is declared by the
scalar, and non-empty evidence artifact IDs are frozen into both registrations.
The three initial profiles are disabled, have no capabilities, and have empty
admission evidence.

`Route2Result` accepts only a registered `profile_id`; it derives scalar ID,
state-equation ID, and capabilities from that profile. Callers cannot attach
capabilities or override scalar/state identity. Energy and force totals are
properties computed from immutable leaf components, never constructor inputs.
The closure tolerance used by future legacy adapters is an internal, versioned,
bounded constant rather than a caller-controlled value. A disabled profile can
produce only fail-closed internal energy evidence: it cannot publish a result,
declare an admitted domain, or carry force leaves.

Public ASE units are declared centrally as energy `eV`, forces `eV/A`, and
Hessian `eV/A^2`. The legacy calculator path is unchanged in this phase.

## `route2-operational-cpcm-fixedtopology-electrostatic-v1`

- Formula: `E_op(R)=Phi_op(R,y*(R))`, where
  `Phi_op=E_vac(R)+1/2<c_ref(R)+T(R)y,P_R(c_ref(R)+T(R)y)>_Q` and `G_np=0`.
- Implementation entry point: reserved as
  `maple.solvation.coupling.energy:not-implemented-phase1`.
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

## Capability labels

| label | meaning | Phase-1 admission |
| --- | --- | :---: |
| E | scalar energy | no |
| F | conservative force of that scalar | no |
| H | force-derived Hessian/HVP | no |
| V | strict common variational functional | no |
| M | MD release gate | no |
