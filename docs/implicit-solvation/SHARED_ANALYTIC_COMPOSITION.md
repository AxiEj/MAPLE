# Shared implicit-solvent composition

## Responsibilities

Implicit solvent is independent of the gas MLIP. Existing calculators retain
their model loading, molecular graph, scalar energy, precision, and gas-phase
derivative implementations. They join one shared solvent-composition boundary:

```text
gas calculator / third-party wrapper       common solvent correction
  gas E/F, H, or direct HVP                 solvent E/F, H, or direct HVP
             \                              /
              shared stateless composition
                    MAPLE task result
```

| Module | Responsibility |
|---|---|
| `analytic_protocol.py` | Immutable scalar/point identities, typed operations, pure addition and validation |
| `solvent_composition.py` | Three explicit stateless E/F, Hessian, and direct-HVP orchestration functions |
| `extra_correction/implicit/correction.py` | Common correction preparation, provider selection, and derivative audit |
| `independent_obc2.py`, `torch_obc2.py` | Independent OBC-II/mbondi2, ACE or NONE, and analytic solvent derivatives |
| Existing gas calculators | Model-specific input/forward/derivative operations, without solvent formulas |
| `uma/_analytic_adapter.py` | FAIR-Chem operation translation and same-scalar gas derivative provenance |
| `analytic_admission.py` | Resolve measured exact profiles; never infer qualification from capability |
| `set_calculator.py` | Construct components and bind a qualified composition through one model-neutral path |

Neither the solvent implementation nor the shared composition functions import
ANI, AIMNet2, MACE, UMA, or FAIR-Chem. UMA remains a `FAIRChemCalculator`; gas
calculators do not need a common inheritance hierarchy to share solvent logic.

## Scientific contract

For the current **fixed-charge** OBC-II model, the same coordinates and atom
order define both components:

\[
E=E_{gas}+E_{solv},\quad F=F_{gas}+F_{solv},\quad
H=H_{gas}+H_{solv},\quad Hv=H_{gas}v+H_{solv}v.
\]

The charge vector, topology, radii, dielectric constants, and nonpolar choice
are immutable solvent inputs. This is not a geometry-dependent charge-response
model or a mutually polarized gas/continuum model. An adapter for such a model
would require its own explicitly derived response contract.

All public results use Hartree and angstrom. Source-unit conversion occurs
once. An analytic operation records the exact checkpoint, gas options, device,
effective dtype, charge/multiplicity, geometry, and molecular graph. The solvent
point additionally binds its ordered charges and parameter payload. Relocating
identical checkpoint bytes does not change the qualified model profile, but the
live path and exact per-input identities remain part of operation provenance.

Composition checks identity and capability before calling a provider and
commits fresh calculator results only after the complete operation succeeds.
Solvent values are added once; a composed result cannot be reused as a gas
component. ASE's `free_energy` remains an alias of the configurational potential,
not a thermochemical Gibbs or hydration free-energy claim.

## Lazy derivatives and UMA

- Ordinary E/F does not prepare or calculate a Hessian.
- A full-H request evaluates only the requested analytic Hessians.
- Direct HVP uses a graph contraction, never a dense Hessian multiplication.
- Numerical differentiation is an explicit numerical mode or a validation
  oracle, never an analytic runtime fallback.

UMA uses FAIR-Chem for its gas model. Its normal E/F predictor remains
Hessian-disabled. Native full-H evaluation is a separate lazy operation;
HVP uses a Hessian-disabled same-scalar graph. Original and compatibility
checkpoint identities, the complete configuration transformation, and exact
tensor-payload equality are recorded independently of derivative results.
Enabling an upstream Hessian API is implemented capability, not scientific
qualification of every checkpoint/backend/device.

## Independence and installation

The explicit `implicit-obc2-torch` extra installs the independent solvent path.
It does not install OpenMM or FAIR-Chem. Model-specific runtimes remain managed
by the user: UMA still requires FAIR-Chem, while OBC-II does not.

```bash
pip install '.[implicit-obc2-torch]'
```

The four legacy OpenMM extras and the default OpenMM selection remain unchanged.
The independent runtime requires explicit `provider=torch`. It does not accept
OpenMM platform/device controls. The legacy OpenMM/Torch split-runtime
derivative route remains separately labeled and retains its live OpenMM source
compatibility check; the independent Torch kernel checks its packaged equation
profile without importing OpenMM.

## Capability, qualification, and activation

Installed analytic support follows a one-way evidence chain:

1. **Candidate qualification:** frozen numerical gates, original checkpoint
   lineage, solvent parity, derivative checks, actual public workflow paths,
   and a genuinely fresh no-OpenMM wheel installation. The harness alone may
   inject a private factory resolver; there is no input-header or environment
   bypass.
2. **Installed admission:** a reviewed data-only entry binds exact gas/solvent
   profiles and the qualification receipt's byte digest. Live capability or a
   matching model name cannot create an admission.
3. **Activation replay:** the same public workflows run without injection and
   bind the final installed mapping and complete current source snapshot.

The qualification snapshot excludes the later data-only allowlist and its own
receipt, avoiding a self-hash cycle. Runtime formula/profile data are not
excluded. Missing or mismatched evidence fails closed. Other checkpoints,
devices, dtypes, D4 settings, and response models are separate cells; a passing
cell must not be generalized to a model family.

Historical reports remain byte-identical. New source-compatibility records
document drift for archival audit only and never authorize execution, sealing,
scoring, or reuse of historical numerical claims against changed source.

## Verification entry points

- `validate_independent_obc2.py`: sealed 48 E/F comparisons, 526-case ACE/NONE
  replay, solvent Hessian and direct-HVP validation.
- `validate_shared_mlip_analytic.py`: exact model-cell gas/composite checks and
  candidate/public activation workflow evidence.
- `tests/solvation/run_no_openmm_install.py`: offline wheel/sdist build and
  installation in a fresh environment, distribution/import absence checks,
  and installed-provider execution outside the source checkout.
- `tests/solvation/test_shared_solvent_composition.py`: unit, identity,
  exactly-once, lazy-operation, and transactional regression contracts.

Only the resulting source-bound per-cell reports establish the supported
matrix. This architecture document does not assert that all cells are qualified
or that validation of a learned energy surface establishes physical accuracy.
