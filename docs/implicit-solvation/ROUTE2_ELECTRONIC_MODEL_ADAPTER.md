# Route-2 electronic-model adapter contract

> **M0 successor:** the capability-layered contract in
> [`ROUTE2_MLIP_PLUGIN_CONTRACT.md`](ROUTE2_MLIP_PLUGIN_CONTRACT.md) is the
> active boundary for new MLIPs.  This document describes the retained legacy
> all-method adapter used by already released profiles.  New plug-ins must not
> copy its unsupported-method pattern.

## Status and scope

The released Route-2 scientific profiles remain bound to the official
MACE-POLAR-1-M checkpoint.  This adapter boundary does **not** make another
MLIP part of those profiles, and it does not change their energies, SCF
equations, force admission, or validation status.

Its purpose is architectural: the continuum providers and the common SCF /
adjoint engine now depend on an explicit electronic-model protocol rather
than on a concrete MACE calculator class or scattered method-name checks.  A
second field-aware MLIP can therefore be integrated by adding one adapter and
one new versioned scientific profile, without modifying PCM, COSMO, CDS, SCF,
or adjoint code.

## Four independent layers

1. **Electronic-model adapter**
   converts a model's native field input and charge/density output into a
   declared Route-2 source/dual space.
2. **Source-space contract**
   owns shape, units, total-charge component, component order, and the exact
   electrostatic pairing.
3. **Continuum provider**
   maps that source to a reaction field and owns its scalar energy, adjoint,
   geometry derivative, cavity, and solvent parameters.
4. **Versioned profile**
   binds one model family, source space, field evaluator, continuum equation,
   cavity, nonpolar functional, energy ledger, solvent set, and release
   status.

The engine composes these layers; it does not infer them from class names.

## Response strategy is an independent axis

The adapter, continuum equation, and electronic response strategy are not the
same abstraction:

* `response=frozen` requests only `cached_state()` at zero external field,
  sends that declared source to the continuum once, and never calls
  `evaluate_state()` with a reaction-field drive;
* `response=scf` additionally requires a compatible state projector and
  repeatedly calls the adapter with the continuum reaction field until the
  unmixed physical fixed-point residual converges;
* a future variational/KKT adapter may expose a stationary electronic
  functional through a separately versioned capability rather than
  impersonating either mode.

Response admission is centralized beside the immutable profile registry and
is reused by input parsing, calculator construction, and providers.  A new
MLIP adapter therefore does not require response-specific conditionals inside
ddPCM, ddCOSMO, CDS, or the shared engine.  Conversely, merely implementing an
adapter never authorizes a scientific profile: its source space, energy
semantics, capabilities, and validation evidence must still be bound by a new
versioned profile.

## Current canonical interoperability space

The only currently implemented common source space is

```text
maple-atomic-net-monopole-real-spherical-l1-v1
```

with one `(n_atoms, 4)` block per geometry:

- component 0: net atomic monopole in `e`;
- components 1--3: atom-centred real-spherical `l=1` moments in `e Angstrom`;
- dual field: atom-centred potential and Cartesian gradient in conjugate eV
  units;
- constrained coordinate: the sum of component 0;
- pairing and component permutation: the immutable
  `ElectrostaticPairing` stored by the source space.

An MLIP with Cartesian dipoles, partial charges only, or a different native
field convention must convert at its adapter boundary.  Native tensors never
leak into the continuum engine.

This is deliberately **not** a claim that arbitrary GTO densities,
quadrupoles, grids, or charge-transfer variables already work.  Those require
a separately named source-space implementation plus continuum forward,
adjoint, constraint, and derivative operators.  They must not be smuggled
through the four-component array.

## Adapter protocol

An adapter exposes an immutable `Route2ElectronicModelDescriptor`, a stable
`cache_identity`, and these operations:

```python
class Route2ElectronicModel(Protocol):
    descriptor: Route2ElectronicModelDescriptor
    cache_identity: int

    def cached_state(self, atoms, *, require_forces=False): ...
    def evaluate_state(self, atoms, drive, *, compute_forces=False): ...
    def preprojected_field_projector(self): ...
    def linearize_source_response(self, atoms, drive): ...
    def field_conditioned_energy_field_gradient(self, atoms, drive): ...
    def source_position_vjp(self, atoms, drive, *, source_cotangent): ...
```

Methods that are unsupported remain present but fail closed.  The descriptor
declares which methods are scientifically available; profile validation
checks the declaration before evaluation.

`AtomicL1CalculatorAdapter` is the compatibility adapter for the established
MACE-POLAR method surface.  It is the only production module that knows names
such as `polar_state()` and `intrinsic_energy_field_gradient()`.  A new MLIP
may instead return its own adapter from
`route2_electronic_model_adapter()` and use completely different native API
names.

There is intentionally no fallback that accepts any object merely because it
has a method called `polar_state`.  Accidental duck typing would allow an
uncertified calculator to impersonate a scientific profile.

## Capability matrix

| Route-2 operation | Required adapter capability |
|---|---|
| frozen source energy | gas electronic state only |
| local-field one-shot / SCF energy | `"local-jet"` in `state_projectors` |
| preprojected-feature one-shot / SCF energy | `"exact-gto-v1"` in `state_projectors` plus adapter-owned projector |
| any solution-phase force | `gas_forces=True` |
| direct PCM half-coupling SCF force | projector in `response_projectors` and `position_vjp_projectors` |
| legacy field-conditioned-energy + PCM force | all direct-ledger capabilities plus the projector in `fixed_field_force_projectors` and `energy_gradient_projectors` |

The direct PCM ledger requests neither the model-energy field gradient nor a
field-conditioned fixed-field force because both terms are absent from its
scalar.  It still requires the gas force for the total MAPLE force and the
response/position VJPs needed to differentiate the fixed point.  This is a
ledger-specific capability decision, not a silent approximation.

## Adding another field-aware MLIP

1. Write a narrow adapter that converts the model's native state to the
   canonical source space and converts `ReactionFieldDrive` to its native
   external-field input.
2. Declare exact model family, profile binding, field evaluator, energy
   semantics, capabilities, model/checkpoint provenance, and source space in
   an immutable descriptor.
3. Add a new `Route2SMDProfileSpec`.  Do not reuse or impersonate a MACE-bound
   profile; the profile is the frozen scientific identity.
4. Add adapter-level source/field order, charge closure, units, cache identity,
   and capability-failure tests.
5. Run continuum pairing, SCF residual, energy-ledger, response reciprocity,
   force, and chemistry-validation gates appropriate to that new profile.

The minimal non-MACE integration test in
`tests/solvation/test_route2_electronic_model.py` proves that a model with an
unrelated native API can drive the common engine through an explicit adapter.
It is synthetic architecture evidence, not chemical-accuracy evidence.

## Scientific boundary

Modularity does not repair physics.  In particular, adapting a learned
fixed-point response does not prove that it is the gradient of the model's
reported energy.  MACE-POLAR's current descriptor therefore declares

```text
field-conditioned-operational-energy-not-density-variational-v1
```

rather than a common variational electronic functional.  Reciprocity,
passivity, common-energy/KKT, continuum smoothness, force/PES, runtime, and
experimental-accuracy gates remain separate and must be reported separately.
