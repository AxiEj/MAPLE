# Route 2 scientific audit supplement

**Date:** 2026-07-26  
**Audited Route-2 head:** `d69049f8b5048c5c762d738f4151d54df822d310`  
**Parent report:** `ROUTE2_SCIENTIFIC_AUDIT_20260726.md`

This supplement preserves findings established after the first report was committed. It does not supersede or weaken the two P0 SMD defects already recorded there.

## 1. Additional confirmed integration regressions

### 1.1 A Route-2-only `#charge(...)` rejection became branch-global

`CommandControl._validate()` unconditionally rejects any parsed `charge` block, even when no Route-2 SMD calculation is selected. The correctly scoped SMD rejection already exists inside `_validate_solvation()`.

Impact: gas-phase and non-Route-2 charge workflows inherited from `enhance` are disabled. The unconditional post-validation rejection must be removed; the SMD-local guard must remain.

An independent red regression test is tracked in:

```text
tests/solvation/test_route2_input_regression_contract.py
```

### 1.2 `enhance` GBSA/QEq was replaced rather than extended

The base `enhance` parser and calculator factory support experimental `method=gbsa`. The Route-2 branch rejects every implicit method except `smd`, although the lower-level GBSA/QEq construction still exists.

This is not an acceptable merge architecture. Implicit methods must be registry-owned profiles, with SMD added alongside the existing GBSA route. Route-2 model/checkpoint/topology restrictions must never live in common implicit-solvent code.

### 1.3 Common `InputReader` applies SMD/MOL2/singlet rules to every implicit method

Any non-null `solv.implicit` currently requires one MOL2 structure with an explicit `0 1` line. This blocks the prior GBSA coordinate contract and conflates:

- canonical element-radius SMD;
- GAFF2 atom-type-dependent SMD profiles;
- future ionic/open-shell profiles.

Each profile must declare its own required input metadata.

### 1.4 Direct Python construction can silently produce gas-phase results

Directly constructing:

```python
MACEPolCalculator(implicit="smd", solvent="water")
```

sets Route-2-looking model/profile state, but the shared base initializer only constructs a correction for `gbsa`; for `smd` it leaves `solvent_correction=None`. The SMD correction is attached only later by `SetCalculator`.

A direct public calculator call can therefore appear configured for SMD while returning gas-phase energy/forces. Route-2 construction must have one fail-closed public factory path.

## 2. Additional confirmed chemistry/input defects

### 2.1 GAFF aromatic carbon can be parsed as calcium

For a MOL2 atom named `CA` with lowercase GAFF/GAFF2 type `ca`, `_element_from_mol2()` tests the title-cased two-letter atom name before the one-letter carbon fallback and returns element `Ca`.

The parser must distinguish proper two-letter element symbols from uppercase force-field atom names. A red contract test is tracked in:

```text
tests/solvation/test_route2_mol2_reference_contract.py
```

### 2.2 Closed-shell metadata is not checked against electron parity

The domain accepts `charge=0, multiplicity=1` without verifying the parity of `sum(Z)-charge`. Neutral NO can therefore be declared a singlet and pass the shared domain function.

Before loading MACE or constructing a cavity, require consistency between electron-count parity and multiplicity parity. A red contract test is tracked in:

```text
tests/solvation/test_route2_domain_reference_contract.py
```

## 3. Self-consistent-state and engine findings

### 3.1 PCMSolver final energy combines neighbouring iterates

The PCMSolver loop computes:

```text
c_n -> f_n=P(c_n) -> M(f_n)=c_(n+1)
```

After tentative convergence, it retains the MACE intrinsic energy evaluated at `f_n` but recomputes the final PCM state from `c_(n+1)`. Those terms belong to one identical discrete state only at an exact fixed point.

Production closure must explicitly choose one final density, recompute its field, reevaluate MACE, verify the returned density, and then evaluate every ledger component and identity at that same state.

### 3.2 The provider-neutral engine is not shared by the default provider

`Route2ContinuumEngine` drives ddPCM, while the default PCMSolver provider retains a separate SCF loop, validators, tolerances, ledger assembly and audit schema.

The two implementations already differ materially:

| Policy | PCMSolver path | ddPCM engine path |
|---|---:|---:|
| mixing | 0.5 | 1.0 |
| density tolerance | `1e-5` | `2e-12` |
| intrinsic-energy tolerance | `1e-5 eV` | `1e-10 eV` |
| maximum iterations | 50 | 100 |
| final same-state/identity gates | partial | stronger |

Provider-specific numerical settings are legitimate; duplicated definitions of a converged state and energy ledger are not. Select the cavity/provider first, then run one shared engine with profile-owned policy and protocol-based derivative capability.

## 4. Coordinate-derivative finding for the reciprocal fixed-box profile

The optional fixed-40-A reciprocal evaluator transforms coordinates before upstream MACE creates its autograd leaf:

```text
R_tilde = R - mean(R).
```

Upstream force/autograd therefore differentiates with respect to `R_tilde`. The declared composite profile requires the pullback:

```text
P = I - 11^T/N,

g_R = P g_tilde,
H_R = P H_tilde P.
```

The pullback must be owned by an explicit coordinate-transform contract and applied to:

- gas/fixed-field MACE force partials;
- density coordinate VJPs;
- any future Hessian.

A near-zero raw translation force in selected canaries is not a proof that the chain rule is implemented.

## 5. The two electrostatic interfaces are approximations, not direct density/potential coupling

### 5.1 Solute to continuum: point `l<=1` moments replace the smooth GTO density

The public PCMSolver and pyddx profiles use MACE coefficients as point monopoles/dipoles. This is not direct Gaussian-density PCM. A Gaussian has infinite support; being outside the atomic centre does not make its potential equal to the point limit.

For the declared `sigma=1.5 A`, the isolated Gaussian-monopole/point-monopole ratio is:

```text
erf(r/(sqrt(2)*sigma)).
```

Representative values are approximately:

| radius | Gaussian/point potential |
|---:|---:|
| 1.20 A | 0.576 |
| 1.52 A | 0.689 |
| 1.85 A | 0.783 |

The current method must be named cavity-sampled point-multipole coupling. A same-cavity Gaussian-MEP provider needs the exact forward, adjoint, energy and coordinate-derivative contract before the direct-density claim is available.

### 5.2 Continuum to MACE: only an atomwise first-order jet is supplied

MACE receives only:

```text
[V(R_i), grad V(R_i)].
```

That is exact for the upstream uniform-field training interface because the potential is linear. A PCM reaction potential generally has non-zero Hessian and higher spatial derivatives across the atom-centred GTO extent.

The implemented coupling is therefore a local first-order reaction-potential projection, not a complete external potential. It must be compared with exact/numerically integrated GTO projections or an explicit-external-charge interface on a preregistered curvature-sensitive panel.

### 5.3 Constant-potential gauge invariance is untested

Upstream uniform-field training constructs a barycentre-centred node potential. Route 2 injects absolute PCM node potentials. For the current neutral domain, adding a constant to every node potential must not alter the physical response or correction.

If the real model is not invariant, a training-compatible gauge transform must be introduced consistently into the residual, reaction map and adjoint; subtracting a mean in only the forward call is not sufficient.

## 6. Provenance and false-result findings

### 6.1 Actual model checkpoint bytes are not pinned

`mace-torch==0.3.16` maps `polar-1-m` to a release URL and reuses an existing cache file by basename. MAPLE records package version and alias, not the resolved model-file SHA256.

A production profile must own an expected checkpoint hash and audit the resolved path, size, SHA256 and source asset before deserialization.

### 6.2 ddPCM audit writes are not transactional

Changed geometries reuse the same `route2-ddpcm-result.json` and `route2-ddpcm-state.npz`. A later failed evaluation can leave a prior successful result in place. The wrapper manifest describes initialization geometry, which may not be the geometry represented by the surviving state archive.

Every evaluation needs a unique run directory and atomic success publication. Failure must publish a structured failure record and must never leave a prior success ambiguously current.

### 6.3 The frozen ISWIG discriminator does not cryptographically bind density to geometry

Its preregistration locks the MOL2 and density-archive hashes, but the older PCMSolver state archive does not contain source positions and the matching manifest is not locked/checked. The exact byte streams are preserved, but their density-at-this-geometry relation rests on directory association.

The rejected result can remain as a bounded negative result with this caveat. A replacement must lock and compare manifest positions, atomic order, profile, execution head and density archive.

## 7. Runtime safety finding

The default PCMSolver path executes a legacy C ABI inside the caller process, changes process-global cwd and redirects process stderr. Existing benchmark code already recognises that the native library can terminate a worker process.

Production PCMSolver calls must be isolated behind a worker process/service with timeout, signal/exit capture, structured IPC and immutable run directories. A Python lock cannot contain a native abort.

Severe PEDRA poor-tessellation diagnostics must also become publication-fatal under a fixed profile. Recording a warning is not enough for production.

## 8. Source-level checks that passed in the later review

The following were compared directly with maintained upstream source and did not reveal a formula/sign/order defect:

- PCMSolver v1.1.12 ctypes structure and function signatures;
- PCM half-coupling and direct/adjoint symmetrisation;
- neutral-subspace residual and GMRES adjoint algebra;
- point monopole/dipole potential, gradient and coordinate VJPs;
- pyddx real-spherical normalisation and bohr/eV conversions;
- PySCF SMD-CDS gradient conversion from Hartree/Bohr to Hartree/Angstrom;
- graph_longrange/MACE local-field component permutation;
- SMD SASA radii (`ASE/PySCF VDW + 0.4 A`) and area/energy units.

This positive list does not override the P0/P1 blockers.

## 9. Test-execution boundary

The review environment could not resolve `github.com` from the local container, so the repository could not be cloned and the red tests could not be executed independently here. GitHub source reads and writes succeeded through the connector.

The Route-2 head reports `242 passed, 2 skipped`, but that result predates and does not include the intentionally red independent tests in the review PR. No new real MACE, PCMSolver, pyddx, PySCF continuum or QM calculation was performed by this review.

The corrective branch must run the entire suite plus all optional real-runtime canaries in a clean, exact-version environment after applying the reference corrections.