# Internal milestone: `torch-smooth-pcm-v1`

## Status and boundary

`torch-smooth-pcm-v1` is a **private, non-admitted, fixed-dimensional, smooth-partition moving-cavity finite-dielectric PCM scalar** implemented in PyTorch under
`maple/function/calculator/extra_correction/implicit/torch_smooth_pcm/`.

This milestone is a **new internal model family**, not a pyddx update, not a pyddx replacement, and not public MAPLE provider support. Historical evidence from the source lineage commit is **not transferable** to the new path, bytes, identities, or hashes.

This document records the exact private continuum contract. A separate
MACE-POLAR-EF stationary adapter now consumes this scalar, but the supplied EF
checkpoint fails its mandatory passivity gate before the PCM iteration. This
does **not** open any public capability, legal distribution claim, or
scientific admission.

## Frozen private identities

The implementation exposes four distinct identities:

| Layer | Value |
|---|---|
| model ID | `torch-smooth-pcm-v1` |
| provider ID | `maple.route2.continuum.torch-smooth-pcm-schwarz.provider.v1` |
| scalar ID | `maple.route2.scalar.torch-smooth-pcm-schwarz-finite-dielectric.v1` |
| configuration-contract ID | `maple.route2.continuum.torch-smooth-pcm-schwarz-config.v1` |

These IDs identify a **configurable internal algorithm family**. The family IDs stay fixed across parameter changes that preserve the same mathematics. The configuration-contract ID is the schema identity. Each exact parameterized instance carries its own `configuration_sha256`.

### Identity rules

- Changing atom order, radii, dielectric, transition width, clearance, harmonic orders, quadrature orders, or dtype/device policy changes `configuration_sha256`.
- Changing the mathematical semantics requires **new family IDs and a new configuration-contract ID**. This includes changes to the partition formula, source or field space, pairing, sign or normal conventions, topology policy, regularization, or scalar composition.
- `W` is the first frozen validation canary, **not** a constructor restriction. It is a validation anchor only.
- Any future admitted profile must bind one exact `configuration_sha256`, solvent identity, evidence bundle, and capability scope. No such admitted profile exists here.

## Exact mathematical authority and lineage

### External equation authority

The ddX equation authority recorded for this milestone is:

- remote repository: `ddsolvation/ddX`
- tag: `v0.8.0`
- annotated tag object: `8c41d83cec215881de0c43551ddf9482fca5c4b6`
- peeled commit: `4d79e3d9caeae5e602683572a71cb550414f9b09`

That authority is limited to **equations and independent mathematical reference**. It does **not** authorize copying or translating LGPL ddX source code or quadrature tables.

### MAPLE implementation lineage

The selected MAPLE implementation lineage is commit
`c578fda2b974dd44833b8f0813bd689751f47d4d`
(`feat(route2): add smooth AIMNet2 multisolvent ddPCM candidate`), authored and committed by **Jiahao Xie** on **2026-08-18**.

This commit is implementation and design lineage only. Its historical tests and evidence do **not** transfer to the new internal package.

### Selected reviewed lineage blobs

The transplant audit records these reviewed `c578` blob identities:

| File at `c578fda2` | Git blob ID |
|---|---|
| `maple/solvation/continuum/functional.py` | `8055b6ed017117c352bcbe3bff449a26ab6fc772` |
| `maple/solvation/continuum/harmonic_coefficients.py` | `7a2beb40b6cb7f0b7042f100ab43b4a692764bc1` |
| `maple/solvation/continuum/harmonic_ddcosmo_functional.py` | `7fdc690bc83801d2133f300fdec7b58268902ebb` |
| `maple/solvation/continuum/harmonic_ddpcm_functional.py` | `15d12a0a2e47d4cc028cedfc92a50df5a0d5767c` |
| `maple/solvation/continuum/harmonic_ddpcm_primitives.py` | `5c9683b0ae698acada52ac65d12b7f9245e61708` |
| `maple/solvation/continuum/harmonic_exposure.py` | `bb2eac180e60988e4ddc0963912b540ff1f8f861` |
| `maple/solvation/continuum/harmonic_point_source.py` | `d41c9669fab938459e9217114ebf4120ac2de365` |
| `maple/solvation/continuum/harmonic_schwarz_primitives.py` | `86ff10d10d24b0d75efe868ccca08d825540d1b2` |
| `maple/solvation/continuum/harmonic_single_layer.py` | `8674cd10dba2900b12a65ae1838e9b51a0e44ac3` |
| `maple/solvation/continuum/harmonic_torch_primitives.py` | `3f2c2653dda2820b8bcaf4e701def421a8c6e21c` |

These blob IDs identify review inputs. They do **not** require byte-for-byte copying into the new implementation.

## Reused current Route-2 contracts

`torch-smooth-pcm-v1` does not invent a new source or pairing contract. It reuses the current Route-2 atomic point-`l<=1` conventions:

- `ATOMIC_L1_PLUGIN_SOURCE_SPACE`
- `ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE`
- `MACE_POLAR_L1_PAIRING`
- `gto_density.py` sign, unit, and projection conventions

The frozen raw source order is `[q, p_y, p_z, p_x]` with units `[e, eÅ, eÅ, eÅ]`.

The frozen Cartesian field-dual order is `[V, g_x, g_y, g_z]` with units `[eV/e, eV/(eÅ), eV/(eÅ), eV/(eÅ)]`.

The raw-versus-Cartesian permutation is

```text
Q[V, g_x, g_y, g_z] = [V, g_y, g_z, g_x]
Q = [[1,0,0,0],
     [0,0,1,0],
     [0,0,0,1],
     [0,1,0,0]]
```

For `N` atoms, the implementation uses block-diagonal `Q_N`.

## Fixed-dimensional moving-cavity scope

This model is a **moving-cavity** construction: all operators depend on geometry `R`, and all source and coordinate derivatives are generated from one Torch scalar graph.

It is also **fixed dimensional**:

- every atom retains a complete local harmonic block of size `(L+1)^2`;
- smooth internal and external tangencies are treated as positive differentiability fixtures;
- there is no ddX-style compressed active-set topology, no geometry-dependent coefficient deletion, and no `U>0` hard-topology branch.

The smooth partition is the c578 Schwarz-family route selected by the user:

- centered compact-support `C∞` pair characteristic;
- polynomial-Shapley partition of unity in coefficient space;
- complete local reaction-potential blocks;
- invariant pair-axis radial quadrature;
- finite-band exact angular contractions;
- no laboratory-fixed cavity mask;
- no FMM.

## Frozen Schwarz equations and scalar

The private scalar freezes the following equations:

```text
F = -B c
A_eps G = A_inf F
L X = G
u = C_field X
C_raw = Q_N C_field
U(R,c) = 1/2 c.T C_raw X
```

where:

- `c` is the raw atom-major point-`l<=1` source;
- `B` maps the source to localized solute-potential coefficients;
- `A_eps` is the finite-dielectric smooth double-layer operator;
- `A_inf` is the conductor-limit smooth double-layer operator;
- `L` is the Schwarz local-potential overlap operator;
- `X`, `G`, and `F` are local-potential harmonic coefficient vectors;
- `C_field` maps reaction coefficients to the Cartesian receiver `[V, g_x, g_y, g_z]`;
- `C_raw` maps the same receiver into the raw dual order needed for the physical pairing.

The implementation is finite dielectric, with the committed principal-value and outward-source-normal convention. It is dense CPU `torch.float64` only.

## Full Lagrangian and KKT contract

The frozen Lagrangian is

```text
Lagr = 1/2 c.T C_raw X
     + lambda_F.T (F + B c)
     + lambda_G.T (A_eps G - A_inf F)
     + lambda_X.T (L X - G)
```

Every term is paired to **eV** under the declared source/field dual contract. The primal equations are the constraints. The transpose chain is

```text
L.T lambda_X = -1/2 C_raw.T c
A_eps.T lambda_G = lambda_X
lambda_F = A_inf.T lambda_G
```

At fixed geometry,

```text
dU/dc = 1/2 C_raw X + B.T lambda_F
drive_cartesian = Q_N.T @ dU/dc
```

This distinction is frozen:

- autograd authority is the raw energy gradient `dU/dc`;
- the internal Cartesian drive is `Q_N.T @ dU/dc`;
- the conventional receiver `C_field X` is a measured diagnostic, not the API authority.

If `C_field X` and `Q_N.T @ dU/dc` coincide on a reciprocal fixture, that is acceptable and does not change the contract.

## Same-scalar derivative contract

`torch-smooth-pcm-v1` is sealed to one internal scalar `_energy_torch(R, c)`. Public derivative methods are generated from that scalar and are final at class definition.

Let `H_raw = d²U/dc²`. Then the frozen response contract is

```text
source_jvp(v_raw)  = Q_N.T H_raw v_raw
source_vjp(w_cart) = H_raw.T Q_N w_cart
<w_cart, source_jvp(v_raw)> = <source_vjp(w_cart), v_raw>
mixed_coordinate_vjp(w_cart) = d/dR <w_cart, drive_cartesian(R,c)>
```

Important scope points:

- the standard field response is Cartesian;
- any raw source HVP is debug-only and must stay explicitly raw-named;
- coordinate differentiation includes the full geometry dependence of `B`, `C_field`, `C_raw`, `A_eps`, `A_inf`, `L`, and both solve chains.

## Direct internal construction API

This milestone is **directly constructible only inside its private package**. It is not wired into public provider selection, parser dispatch, profile registries, or calculator factories.

The current private entrypoint is the class
`maple.function.calculator.extra_correction.implicit.torch_smooth_pcm.TorchSmoothPCM`.

Its internal surface currently exposes:

- construction from explicit `atomic_numbers` and `radii_angstrom` plus numerical settings;
- `energy(...)`;
- `drive_cartesian(...)`;
- `coordinate_gradient(...)`;
- `energy_drive_gradient(...)`;
- `source_jvp(...)`;
- `source_vjp(...)`;
- `debug_source_hvp_raw(...)`;
- `mixed_coordinate_vjp(...)`;
- `joint_hvp(...)`;
- debug inspection via `debug_geometry_matrices(...)`, `debug_primal_state(...)`, and `audit(...)`;
- provenance accessors `configuration_sha256()` and `execution_provenance()`.

This is an **internal construction API only**. There is no public provider keyword, no public profile ID, and no public capability declaration.

## Configuration identity versus execution provenance

This milestone separates mathematical configuration identity from runtime execution identity.

### Configuration identity

`configuration_sha256` binds the exact parameterized instance, including:

- the four family/schema IDs;
- atomic numbers and atom order;
- explicit radii;
- transition width;
- harmonic truncation and quadrature orders;
- dielectric;
- source-shell clearance;
- dtype/device policy;
- reused source-space, field-space, and pairing identities;
- frozen raw and Cartesian order conventions;
- frozen equation string;
- source-lineage hash records;
- reused-contract hashes;
- current private implementation hashes;
- conditioning-policy values;
- empty capabilities.

It deliberately excludes actual runtime library versions, CPU identity, BLAS identity, and backend thread state.

### Execution provenance

`execution_provenance()` binds the runtime record for one execution environment, including:

- the `configuration_sha256`;
- Torch, NumPy, SciPy, and ASE versions;
- platform and CPU information;
- Torch build and parallel backend details;
- NumPy BLAS identity;
- the derived `execution_provenance_sha256`.

This distinction is mandatory: parameter changes alter the configuration hash, while runtime-environment changes alter execution provenance without redefining the mathematics.

## Frozen validation canary `W`

The first frozen validation canary is the canonical water fixture:

- `R = [[0,0,0], [0.9572,0,0], [-0.2399872,0.927297,0]] Å`
- `Z = [8,1,1]`
- `a = [2.294, 1.2, 1.2] Å`
- raw source rows:
  - `[-0.70, 0.04, -0.02, 0.03]`
  - `[ 0.35, 0.00, 0.01,-0.02]`
  - `[ 0.35,-0.01, 0.00, 0.02]`
- transition width: `0.08 Å²`
- surface `lmax = 3`
- partition `lmax = 6`
- partition/source/double-layer radial orders: `96 / 128 / 128`
- dielectric: `80.0`
- near-vacuum diagnostic: `1 + 1e-8`
- conductor diagnostic: `1e8`
- source-shell clearance: `0.05 Å`
- dtype/device: CPU `torch.float64`

`W` is the first frozen validation canary only. It is not a water-only constructor restriction and does not admit other configurations.

## Fail-closed conditions

This model must fail closed on all of the following:

- invalid geometry or source shapes;
- non-`float64` NumPy inputs at the Python boundary;
- invalid dtype or device requests;
- nonfinite geometry, source, radii, or parameters;
- `dielectric <= 1` or nonfinite dielectric;
- illegal atomic numbers or missing/nonpositive/nonfinite radii;
- unsupported source order above point-`l<=1`;
- coincident atomic centres;
- point-source target-shell clearance violation below `0.05 Å`;
- partition rank or partition-of-unity failure;
- operator singular-value or conditioning gate failure;
- catastrophic or fixture-derived amplification-indicator failure;
- provenance or implementation-hash drift.

No jitter, pseudoinverse, diagonal shift, block deletion, tolerance loosening after observing results, fitting, or regularization fallback is part of this contract.

## Protected current boundaries

This milestone is not allowed to modify or silently replace the current pyddx Route-2 path. In particular:

- pyddx/ddPCM provider, parser, profile, registry, cache, and ledger behavior remain unchanged;
- the protected files recorded in the transplant audit must remain byte-identical;
- the private model is rejected by public parser/provider selection;
- no new public provider string is exposed;
- `macepol-ef-v2.pt` is not part of this continuum milestone and is not loaded by the private continuum object.

This is therefore **not** pyddx parity work and not pyddx replacement work. The existing pyddx/ddPCM path remains the separate reference implementation.

## Explicitly closed scope and non-goals

The following remain closed for this milestone:

- public provider exposure;
- public parser support;
- public profile registration;
- capability admission of any kind;
- an admitted or passivity-bypassed MACE-POLAR-EF coupling;
- MACE-POLAR coupling in general;
- AIMNet2 coupling;
- SMD-CDS composition;
- public force, PES, OPT, TS, IRC, FREQ, or MD claims;
- FMM or other compressed acceleration paths;
- ddX exact active-set identity;
- the older `4dca73d7` weighted-Galerkin scalar;
- fitting, calibration, or benchmark-driven retuning;
- legal or redistribution conclusions.

## Legal and license block

The current repository `LICENSE` is internally inconsistent. It contains:

- BSD 3-Clause redistribution language;
- an academic-use-only notice;
- a commercial-use prohibition;
- an all-rights-reserved reservation.

Because of that conflict, this milestone must make **no** redistribution, relicensing, compatibility, or legal-clean-room claim.

Separately:

- ddX is treated here as external equation authority only;
- equation citation does not by itself resolve LGPL or downstream compatibility questions;
- this document does not edit the repository `LICENSE` and does not unblock distribution.

As of **2026-09-02**, distribution and legal claims for this milestone remain blocked pending owner or legal clarification.

## Current workspace evidence snapshot

The current private implementation lives in untracked local files under
`maple/function/calculator/extra_correction/implicit/torch_smooth_pcm/` with matching private tests under `tests/solvation/`.

The protected-file SHA-256 values observed in this workspace match the transplant audit baseline for:

- `ddpcm_smd.py`
- `pyddx_pcm_response.py`
- `correction.py`
- `set_calculator.py`
- `command_control.py`
- `route2_smd_profiles.py`

That match supports the protected-boundary claim only. It is **not** scientific admission and does **not** transfer historical evidence.

A fresh local targeted `torch_smooth_pcm` test run on **2026-09-03** produced
`186 passed`. The separate guarded MACE-POLAR-EF/smooth-PCM
connection tests produced `5 passed`; normal evaluation was verified to stop
at the failed electronic passivity gate before continuum iteration. These are
private mathematical and plumbing checks, not scientific admission.

## Stop condition for this document

This document is complete when it states, without overclaiming:

1. the exact private family and configuration-contract identities;
2. the ddX equation authority and `c578` implementation lineage;
3. the fixed-dimensional moving-cavity Schwarz equations and KKT chain;
4. the raw-to-Cartesian mapping and unit contract;
5. the configuration-versus-execution provenance split;
6. the fail-closed boundary;
7. the direct internal construction-only API;
8. the closed scope and non-goals; and
9. the unresolved repository license conflict.
