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

The direct AIMNet2 geometry-mediated diagnostic is a separate state identity
from the mutual-polarization root. It composes a field-independent
`c(R)=[q_NQE(R),0,0,0]` model adapter with either the existing pyddx
atomic-`l<=1` map, the fixed-dimensional smooth-harmonic point-monopole
conductor, or its finite-dielectric double-layer PCM sibling, and differentiates
one registered scalar. The harmonic providers reuse the common `E/K/V`,
exposure, SO(3), and sealed-functional machinery rather than duplicating a
continuum engine. Neither branch routes an AIMNet2 source through the
MACE-POLAR fixed-point model or implies electronic mutual polarization. See
[AIMNET2_GEOMETRY_MEDIATED.md](AIMNET2_GEOMETRY_MEDIATED.md) and
[AIMNET2_POINT_HARMONIC.md](AIMNET2_POINT_HARMONIC.md). The distinct
finite-dielectric equations and admission boundary are documented in
[AIMNET2_POINT_HARMONIC_DDPCM.md](AIMNET2_POINT_HARMONIC_DDPCM.md).

Its second-order diagnostic preserves the same ownership. The AIMNet2 adapter
owns only `J_q h`, `H_E h`, and fixed-cotangent `D_R[J_q^T v][h]`; the sealed
continuum scalar owns one joint `(R,c)` HVP; the coupling layer owns the
four-term weak-scalar composition; and `release` recomputes evidence from raw
operands. The pyddx arm has no sealed joint-HVP contract and fails closed.

Stationary-water Hessian evidence keeps numerical roles separate rather than
creating another engine. `geometry_mediated_stationary.py` owns only the exact
water internal-coordinate map and root-trace/event reduction;
`geometry_mediated_frequency_records.py` validates raw center/HVP/FD records;
`geometry_mediated_frequency.py` owns dense-Hessian, rigid-mode, and
normal-mode gates; and the runner only orchestrates the real stack. The
calculator-independent
`maple/function/dispatcher/frequency/normal_modes.py` supplies the reusable,
unit-explicit `eV/angstrom^2` mass-weighting and rigid/vibrational subspaces.
The public molecular FREQ driver now reuses that same pure kernel and validates
its raw ASE-unit Hessian boundary. Its reusable stationary-point assessor keeps
minimum and first-order-saddle inertia signatures mutually exclusive and
withholds transition-state thermochemistry. The legacy GS/LQA/HPC/EulerPC IRC
drivers also reuse that projected first-order-saddle preflight at their initial
geometry, but neither this starting gate nor their gas-phase task wiring admits
a Route-2 calculator: the retained research runner still owns its source-bound
scalar, event-domain, HVP, and evidence checks independently.

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

Two AIMNet2 diagnostic files currently cross that guide for explicit audit
reasons, not because they are engines. `_aimnet2_float64_source.py` is the one
source/version/checkpoint-bound reconstruction boundary, including the
ordinary/second-order parity gate that cannot be separated from runtime
identity. `release/geometry_mediated_hessian.py` is a pure, side-effect-free
raw-record schema validator and reducer; its length is the explicit center,
five-direction, six-endpoint, topology, and admission validation contract. It
does not solve physics, call AIMNet2/Torch, or expose a capability. Further
physics or workflow behavior must go in a new owned module rather than expand
either file.

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
at tangency. The smooth Torch candidate now generates its fixed-source
coordinate partial and mixed drive/coordinate pullback from that scalar, but
no capability is enabled. The full construction and its fail-closed boundaries
are documented in
[HARMONIC_GALERKIN_TIER_V.md](HARMONIC_GALERKIN_TIER_V.md).

The same continuum scalar also has a separately named operational research
profile. That profile retains the original MACE density response rather than
claiming electronic energy/source conjugacy, excludes the field-conditioned
model energy difference, and uses the existing fixed-point adjoint to
differentiate `E_vac+G_harm`. Source-bound water, methanol, ethanol, acetone,
and acetonitrile canaries now pass their frozen local rotation gates; water
additionally passes three force directional checks, while the multi-atom
shards pass translation and identical-atom permutation checks. The harmonic
continuum is the
coefficient-space replacement for, not a repair of, the failed finite
laboratory-grid route. The profile is not Tier V and remains fully disabled
until physical-component, accuracy, complete PES/domain, Hessian/FREQ/MD, and
release panels pass.
