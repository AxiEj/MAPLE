# Route 2 conservative-vNext architecture

The rebuild preserves MAPLE's input, calculator registry, dispatchers, and
ordinary gas-phase calculators. It replaces the Route 2 physics kernel behind
a compatibility boundary.

## Target package

```text
maple/solvation/
  api/          immutable public contracts, registries, units, results
  models/       vacuum/source/response/variational capability adapters
  coupling/     spaces, Q pairing, B/B*, state equation, solver, adjoint, scalar
  continuum/    backend protocol and C-PCM/ddX/PCMSolver implementations
  surfaces/     fixed-topology and source-dependent surface providers
  nonpolar/     separately admitted nonpolar scalars and derivatives
  hessian/      conservative-force finite-difference Hessian and HVP
  release/      gates, evidence, manifests
  ase_calculator.py
```

Dependencies point inward to `api`; model, continuum, surface, and nonpolar
implementations do not inspect each other's names. The coupling/energy layer
composes protocols. The ASE calculator is the outer adapter. Production code
never imports from `research/`.

## Responsibility boundaries

- `api`: identity, types, units, capabilities, validation; no solver imports.
- `models`: model-native inference and derivatives; no continuum branching.
- `continuum`: continuum state, scalar, field, JVP/VJP, coordinate VJP; no
  model-name branching.
- `surfaces`: topology/geometry state and derivatives; no fixed-point logic.
- `coupling`: the single source/receiver operator, constrained coordinates,
  physical residual, numerical root solver, adjoint, and scalar composition.
- `release`: evaluates evidence; it cannot manufacture provider capability.

## Migration rule

The legacy `route2_engine.py` remains a compatibility shell while behavior is
migrated. Correct assets are wrapped or moved in small commits:

1. canonicalize the plugin source/dual-space and pairing contracts;
2. migrate provider-independent residual/adjoint mathematics;
3. wrap the fixed-topology amplitude-SWIG C-PCM implementation;
4. migrate the exact-GTO coupling path;
5. attach the MACE-POLAR adapter and real-checkpoint canaries;
6. switch the compatibility shell to the new kernel;
7. archive unused experiments under `research/route2_legacy/` only after
   parity and import-graph checks pass.

No new module may become a second monolithic engine. Files above roughly 600
lines require a documented single-responsibility reason.

## Public unit boundary

ASE-facing results use eV, eV/Angstrom, and eV/Angstrom squared. Internal
Hartree/Bohr kernels cross explicit conversion functions. MAPLE's historical
Hartree-formatted reports may retain their text convention only by converting
the ASE values at the dispatcher/report boundary; they may not rely on an ASE
calculator violating ASE units.

## Capability flow

A callable method is not a capability. A profile starts with no E/F/H/V/M
tier. Provider contracts describe what can be computed; release evidence admits
a bounded profile; the registry exposes only that intersection. Unsupported
direct calls fail at the provider/adapter boundary as well as the CLI.

Profile identity also includes the continuum's physical-configuration contract,
not only a backend class name. The water radial-GTO candidate binds dielectric
`78.39`, SMD-water Coulomb radii, and 194 fixed Lebedev directions per atom.
Injected grids or altered dielectric/radii/order receive a separate diagnostic
identity and cannot enter that profile.

## Strict rotation-controlled continuum branch

The Tier-V continuum target is a fixed-dimensional per-atom spherical-harmonic
Galerkin representation with covariant operator assembly and one stationary
scalar. A finite laboratory-frame point mask is not a structural `SO(3)`
guarantee, regardless of Lebedev order. The disabled reference now composes a
rectangular smooth-exposure product embedding `E`, the full-shell Coulomb
operator `K`, `A=E.T K E`, and the eight-channel Gaussian `S=E.T V` with the
receiver exactly `S.T`. The exact shell kernel is `C1` but not generally `C2`
at tangency, and no coordinate pullback or capability is enabled. The full
construction and its fail-closed boundaries are documented in
[HARMONIC_GALERKIN_TIER_V.md](HARMONIC_GALERKIN_TIER_V.md).
