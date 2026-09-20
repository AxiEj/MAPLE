# Numerical LPB reference

## Three roles, four endpoints

```text
MAPLE Route 1 — fixed-charge additive solvent
├── FAST: OBC-II / ACE
│   └── Product endpoint; existing energy/force tasks unchanged
├── ACCURACY: CHA-GB/ALPB + PBSA cavity/dispersion
│   └── Experimental higher-accuracy SP endpoint; unchanged
└── REFERENCE: ddX / ddLPB
    ├── Experimental SP / OPT / numerical FREQ / P-RFO and dimer TS
    └── APBS: independent grid-based cross-check
```

This adds a reference role, not a new product default. Existing GB variants
remain options rather than new top-level development lines. The retired
ddPCM accuracy-screen runners and CQEq/GB implementation remain deleted.
Their archived numerical results are not recalculated or reused as proof for
this new reference provider.

## Explicit reference input

From this checkout root, install the optional reference extra in the intended
isolated environment (do not change another worktree's editable installation):

```bash
python -m pip install '.[implicit-ddlpb]'
```

This pins `pyddx==0.8.0` and `openmm==8.5.2`; it does not change the base or GB
extras. pyddx may require a source build with CMake, C++/Fortran, BLAS/LAPACK
and OpenMP. The initial runtime is checked on Python 3.11. Version 0.9 is not
implicitly substituted: higher-multipole and radius-response support are
outside this fixed-point-charge, fixed-radius release.

```text
#model=ani2x
#sp(verbose=1)
#device=cpu
#charge(source=mol2,label=your-fixed-charge-provenance)
#solv(implicit=water,method=pb,provider=ddx,model=lpb,profile=ddlpb-union-mbondi2-v1,nonpolar=none,solvent_kappa_inverse_angstrom=0.1,experimental=true)

0 1
MOL2 molecule.mol2
```

`verbose=0` requests energy; `verbose=1` also requests the complete composed
force. ANI2x is an input example, not a chemistry-accuracy recommendation.
Fixed MOL2, AM1-BCC and explicit ABCG2 charge preparation remain separate
choices. QEq/CQEq remains unavailable.

`solvent_kappa_inverse_angstrom` is the inverse screening length, not molarity:
`0.1` means a 10-angstrom screening length. It must be explicitly positive
and finite. Exact zero salt is the PCM limit of LPB, but this first runtime
does **not** silently switch to PCM. The zero-salt limit may be checked in a
separately labelled validation calculation. See [ddX theory](https://ddsolvation.github.io/ddX/md_docs_theory.html).

The first public profile fixes epsilon_in=1, epsilon_out=78.5, lmax=9,
302 Lebedev points, eta=0.1, shift=0, tolerance=1e-10, maximum 200 iterations,
20 DIIS iterates and one native thread, with FMM disabled. Header-level
numerical tuning, non-water defaults, variable charges/radii, nonpolar
pairing, prebuilt shells, SCAN/MD and multi-structure TS remain outside this
reference entry point. A separate validation API can compare stricter numerical settings.

## OPT, FREQ and TS opening (2026-09-10)

Use the same fixed-charge, positive-kappa solvent line above and replace the
model/task lines as follows:

| Task | Model line | Task line |
|---|---|---|
| OPT | `#model=ani2x` | `#opt(method=lbfgs,max_iter=100)` |
| FREQ | `#model=ani2x(hessian=numerical)` | `#freq(method=mw)` |
| P-RFO TS | `#model=ani2x(hessian=numerical)` | `#ts(method=prfo,max_iter=100)` |
| Dimer TS | `#model=ani2x` | `#ts(method=dimer,n_init=force,max_iter=100)` |

These tasks use the SAME gas-MLIP plus ddLPB polar scalar, with fixed charges
and radii. FREQ/P-RFO require numerical complete-force Hessians; dimer uses
force differences, not a gas-only HVP or a full Hessian. A completed candidate
search/iteration-limit exit does not certify a chemical transition state.
Interpret frequencies at stationary geometries and verify the final saddle
index and reaction connectivity separately. RRHO translation/rotation retain
the existing ideal-gas convention, not a solution standard-state free energy.

**ANI numerical precision is backend-owned, not a ddLPB patch.** Ordinary ANI
SP/OPT default inference is unchanged. Requesting a numerical Hessian, or
entering force-difference dimer, prepares the SAME loaded ANI checkpoint in
float64 before E/F/H evaluation, and recommends a `0.0005 angstrom` stencil.
No checkpoint file, tensor values, solvent radius or charge parameter is fit
or replaced. Explicit `dtype=float32` is rejected for ANI numerical curvature;
`dtype=float64` can be requested for other ANI tasks. There is no automatic
precision demotion midway through a workflow.

The generic Hessian default stays `0.002 angstrom` for backends without a
recommendation; dimer retains its existing default for those backends.
Explicit derivative steps still override a recommendation. Runtime output
reports actual precision and raw, pre-symmetrization Hessian diagnostics; the
latter are also available as `last_numerical_hessian_diagnostics`. These are
not a universal molecule-size-independent accuracy certificate.

The original float32/coarse-step failures are preserved in
`.omx/verification/ddlpb-workflows-20260909/`. Component-isolated evidence showed
clean ddLPB derivatives, while ANI precision AND stencil resolution required
repair. Float64 alone at `0.002 angstrom` did not pass. The fixed float64 plus
`0.0005` rule passed the unchanged `5e-4 Ha/angstrom^2` numerical test budget
against `0.00025` both at the initial water geometry (max raw-H change
`2.27e-4`) and an untouched OPT-final geometry (`3.22e-4`). This is bounded
numerical qualification, not chemical accuracy or broad TS convergence.

## What the energy and force mean

The reference correction is only `G_ddLPB,polar`; there is no GB energy,
APOLAR, ACE, cavity or dispersion term added to it. The normal MAPLE SP still
reports the composed scalar:

```text
E_solution(R) = E_MLIP,gas(R) + G_ddLPB,polar(R,q_fixed)
```

This is **not** a complete solvation free energy. The audit manifest marks
`role=Reference`, `product_contract=false`, and the polar-only formula.

Native ddX centers/radii use Bohr and its energy uses Hartree. Its two native
coordinate-derivative terms are summed, negated and converted to return
MAPLE force in Hartree/angstrom. Omitting either term is incorrect for moving
atom-centred charges; the sign is independently checked by finite differences.
See the [pinned upstream gradient test](https://github.com/ddsolvation/ddX/blob/4d79e3d9caeae5e602683572a71cb550414f9b09/src/test_gradients.py).

Successful result provenance records geometry/charge/radius identity, native
version/binary hash, numerical settings, primal/adjoint solved flags and
iteration counts. The setup manifest is not a post-solve convergence record.
No residual is invented, no failed solve produces a successful result, and
there is no fallback to a different solvent model or incomplete force.

## Independent APBS comparison

The existing APBS endpoint and defaults are unchanged. Its existing molecular
surface and APOLAR term are **not** directly equivalent to the ddX union-of-
mbondi2-balls, polar-only reference.

The separate validation harness starts with a shared spherical cavity. It
uses identical charges/radii, epsilon values and salt screening; APBS uses
zero probe/ion radius and no APOLAR term. Only the solvated APBS block contains
ions; the unscreened epsilon=1 reference contains none. The kappa/molarity
conversion and APBS-reported physical settings must be verified.

Solver agreement additionally needs grid/domain convergence or a defensible
extrapolation. A large coarse-grid error cannot inflate its own acceptance
window. Until these conditions are established, the result explicitly reports
`independent_crosscheck_complete=false`. That limits the agreement claim,
not the availability of already-validated experimental ddLPB workflows.

Run new evidence into a fresh directory; do not overwrite old artifacts:

```bash
python docs/implicit-solvation/benchmarks/run_ddlpb_reference_validation.py \
  --mol2 molecule.mol2 --solvent-kappa-inverse-angstrom 0.1 \
  --output-dir .omx/benchmarks/ddlpb-reference-new-run
```

Add `--apbs /path/to/apbs` for the optional independent comparator. Keep sphere
checks, all molecular FD components at both steps, refinement deltas, failures
and timings. Broader molecule/solvent coverage is separate from this first
bounded numerical-reference delivery.

## Initial verification checkpoint (2026-09-09)

Fresh real ANI2x SP plus ddLPB force execution closes against separate gas and
polar controls: energy addition error `5.77e-15 Ha`, force addition error
`3.47e-18 Ha/angstrom`. On the fixed water fixture at kappa=`0.1 angstrom^-1`,
all-coordinate FD maximum errors are `8.85e-10` and `1.59e-9 Ha/angstrom` at
steps `1e-4` and `5e-5 angstrom` respectively. These measure derivative
consistency, **not** experimental accuracy or exact-PB force error.

Working `(9,302,1e-10)` versus stricter `(11,590,1e-12)` numerical settings
change energy by `0.00216 kcal/mol` and have maximum force-component difference
`0.1603 kJ/mol/angstrom` on that fixture. Refinement remains separately visible;
no parameter or acceptance threshold was tuned to conceal the difference.

The optional APBS spherical comparison executed, but only one grid/domain
point is present and physical-parameter equivalence is not yet certified.
`independent_crosscheck_complete=false` remains the correct outcome. Do not
claim ddLPB/APBS converged agreement from this run.

Final raw input/output/source-bound evidence is local under
`.omx/benchmarks/route1-ddlpb-sp-20260909-final/` and
`.omx/benchmarks/route1-ddlpb-reference-20260909-final/`.
