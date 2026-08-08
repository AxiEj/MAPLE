# Route 2 reconstructed-density rho-DROP CPCM

## Status

This document records the implemented research boundary for
`route2-rhodrop-cpcm-operational-v1`.

The admitted claim is deliberately narrow:

> Route 2 can build a source-dependent DROP surface from a versioned
> reconstructed MACE-POLAR density, solve the stationary CPCM electrostatic
> problem on that surface, and differentiate the nonlinear source-to-reaction-
> field map with a local JVP/VJP at the exact cached forward state.

The profile is **electrostatics-only and energy-only**. It does not admit CDS,
analytic nuclear forces, optimization, frequencies, molecular dynamics, or a
full-functional model drive. The reconstructed density is not represented as
a learned full electron density and has not passed a QM isodensity-surface
benchmark.

## Scientific boundary

Route 2's public source remains the raw MACE-POLAR atomic block

\[
c_A=(q_A,p_{A,l=1,m=0},p_{A,l=1,m=1},p_{A,l=1,m=2}),
\]

not a complete electron density. The new cavity therefore uses

\[
n_{\mathrm{rec}}(\mathbf r;c,\mathbf R)
=n_0(\mathbf r;\mathbf R)-\rho_{\mathrm{res}}(\mathbf r;c,\mathbf R),
\qquad
S=1-\frac{n_{\mathrm{rec}}}{n_{\mathrm{iso}}}.
\]

`n_0` is a positive, versioned analytic Gaussian-mixture asset whose
components integrate to the nuclear charge. `rho_res` is the existing
MACE-POLAR Gaussian monopole/dipole source with width 1.5 angstrom. Thus

\[
\int n_{\mathrm{rec}}(\mathbf r)\,d\mathbf r
=\sum_A Z_A-\sum_A q_A,
\]

while the odd dipole contribution integrates to zero. This proves the charge
sum rule, not local QM-density accuracy.

The level-set provider is the sole owner of:

- raw real-spherical `l=1` to Cartesian ordering;
- angstrom/bohr conversion;
- analytic spatial derivatives through third order;
- source JVP/VJP and explicit level-set nuclear VJP;
- reconstructed-density nonnegativity, surface residual, gradient, and
  electron-count checks.

No production callback uses spatial finite differences or `numpy.interp`.

Frozen reference-density identities:

```text
Gaussian-mixture table SHA256
d81363187ce6092b23031f5626e0d794bbad4f44646f1e06e41b1112768fefc6

manifest SHA256
bc85c8375fee448ceb78f7394a23a459e9c17ea6e70f672f3e3937072112736f

generator SHA256
ff4d8a79ad69405879a740d6af8cd503eb1d998cf500ed0a06482ea062029b98

canonical unpacked-mixture content SHA256
6c98c98391fa93fda8690a491af2bc1318d891373f424196d6e40c5779f32e60
```

The last digest is recomputed from the ordered `(Z, N_k, alpha_k)` scientific
content. It prevents an in-memory object from claiming the frozen NPZ/manifest
hashes while carrying different Gaussian weights or exponents. Both asset
construction and the operational profile recompute this digest; it also enters
the level-set, provider-configuration, and audit identities.

## Module boundaries

| Module | Single responsibility |
|---|---|
| `route2_atomic_reference_density.py` | Versioned analytic reference-density asset, hashes, Gaussian screening, value through third spatial derivative, electron-count identity. |
| `route2_density_levelset.py` | Reconstructed density, scaled level set, unit/order conversion, source JVP/VJP, explicit nuclear VJP, boundary diagnostics. |
| `route2_moist_drop.py` | Pinned MOIST runtime verification, callback lifetime, cavity/operator assembly, immutable surface snapshot, MOIST surface-to-LSF adjoint contractions. |
| `route2_rhodrop_cpcm.py` | Stationary CPCM forward solve and nonlinear reaction-map JVP/VJP. |
| `route2_rhodrop_profile.py` | Versioned parameters, provenance, admission status, zero-CDS leaf, and composition with the existing Route 2 engine. |

The provider remains parallel to `route2_pcm_response.py`; the fixed-surface
module is not reinterpreted as a variable-cavity implementation.

## Stationary CPCM scalar

For the source-bound surface `Gamma[c,R]`, let `B_Gamma` map the atom-centred
source to the surface electrostatic potential. The implemented forward state
is

\[
v=B_\Gamma c,
\qquad
A_\Gamma q=-f_\epsilon v,
\qquad
f_\epsilon=\frac{\epsilon-1}{\epsilon},
\]

\[
f=B_\Gamma^*q,
\qquad
E_{\mathrm{pol}}=\frac12q^Tv
=\frac12\langle c,f\rangle.
\]

The solve requires a symmetric positive-definite MOIST CPCM matrix, a bounded
2-norm condition number, a passing linear residual, a non-positive stationary
polarization energy, and equality of the surface and source/field couplings.
Source, geometry, level-set, surface, operator, runtime, and forward state are
content-addressed. A changed source cannot reuse the cached energy or
linearization state.

MOIST may emit the same OpenMP-built surface in a different raw point order.
The adapter therefore canonicalizes all surface-indexed arrays by
`(owner, x, y, z)`, permutes both CPCM matrix axes, and retains the inverse
permutation solely for native adjoint calls. Total area and the
divergence-theorem volume are recomputed in that canonical order; MOIST's raw
parallel-reduction totals must agree within a tight floating-point tolerance.

## Nonlinear reaction-map derivative

Because the surface depends on `c`, `apply` and `adjoint` mean

\[
\operatorname{apply}(d)=J_f(c,R)d,
\qquad
\operatorname{adjoint}(y)=J_f(c,R)^Ty
\]

at the latest exact `apply_scf(c)` state. They are not calls to a global linear
operator.

For `L=<y,f>`, the reverse solve is

\[
A^Tz=B_\Gamma y,
\qquad
\bar v=-f_\epsilon z,
\]

with direct source cotangent `B_Gamma^* bar(v)`. The source-dependent surface
cotangent contracts the three contributions

\[
q^T(\delta B)y,
\qquad
\bar v^T(\delta B)c,
\qquad
-z^T(\delta A)q.
\]

MOIST converts these surface weights to level-set value/gradient/Hessian
weights, and `DensityLevelSetProvider.source_vjp()` returns the final source
cotangent. The implementation never materializes `dGamma/dc` and makes no
self-adjoint assumption about `J_f`.

The pinned MOIST API exposes the efficient reverse contraction but not the
corresponding forward surface JVP. The current exact JVP therefore evaluates
`d^T J_f^T e_i` over the `4N` output basis. This is an analytic reference path,
not a production-efficient Krylov operator:

```text
jvp_implementation = analytic-vjp-transpose-probe-reference-v1
operational_jvp_efficiency_admitted = False
```

An upstream forward-mode surface/LSF contraction is the preferred repair; a
cavity finite-difference loop is not an acceptable substitute.

## Half-coupling derivative and Route 2 integration

For a source-dependent map,

\[
\nabla_c\frac12\langle c,f(c)\rangle
=\frac12\,Qf+\frac12J_f(c)^TQ^Tc,
\]

where `Q` is the existing source/field pairing conversion. The generic
`pcm_half_coupling_source_gradient()` helper implements this formula after
checking that the source and field belong to the exact cached forward state.
Fixed-cavity reciprocal providers retain their existing fast path.

`RhoDropCPCMProfile.build_engine()` composes the provider with the existing
`Route2ContinuumEngine`; it does not introduce a second SCF or adjoint engine.
The operational profile drives MACE-POLAR with the conventional electrostatic
reaction-potential jet and uses only the `pcm-half-coupling-only-v1` energy
ledger. The engine is explicitly bound to that ledger, so an omitted or
requested legacy MACE-field-plus-PCM ledger fails before continuum work.
`RhoDropZeroCDSResult` makes the absence of a nonpolar term explicit.

All outer SCF/Anderson/adjoint numerical settings are part of the profile
identity. Total charge is the only dynamic engine setting and is bound
separately by the build argument and provider-cache signature. The operational
profile also requires exact within-runtime cold replay; callers cannot disable
that gate while retaining the v1 profile identity.

The frozen-surface electrostatic pairing is reciprocal, but the complete
nonlinear reaction-map Jacobian is explicitly marked
`reaction_jacobian_self_adjoint = False`.

## Pinned runtime and current evidence

MOIST is pinned to commit
[`6e94f4a5841f5d2e1987e1ca496c7b9baad75e59`](https://github.com/lukaswittmann/moist/tree/6e94f4a5841f5d2e1987e1ca496c7b9baad75e59),
source version `0.6.0-alpha.1`, Python/C API version `0.6.0`.
The upstream project describes itself as pre-release and warns that its public
APIs are still in development. The runtime verifier therefore binds both the
Python extension and shared library by SHA256 rather than accepting an
unversioned import. It additionally requires the canonical imported
`moist`, `moist.interface`, and `moist.library` modules, verifies the patched
Python source tree, and uses the dynamic loader to attest the shared library
that is actually bound to the extension.

The local reproducible build used for the real-backend tests has:

```text
Python extension SHA256
8b30baef76587221cd187c80ffc6d0cbbdde11d904969febeb4b961552156cd0

shared library SHA256
69d59b67627016223cbb4328f360b98754906b64c6be83dbf91c6c711a8c042d

patched Python package source SHA256
a9a907902c42242b9f65df58bf41a11df56899e08bca3eb1fcb3a41589af9720

build-only Python import patch SHA256
96608ddc4135efd35324e593fdef343f7481a73564bd88d34d2aac7688ba9190
```

The build-only patch removes a stale missing-symbol declaration from the
Python wrapper; it does not change DROP or CPCM numerical code. Its exact
content is archived at
`docs/implicit-solvation/patches/moist-6e94f4a-python-import.patch`.

A fresh build starts from the pinned commit and applies only that archived
patch before configuring MOIST's documented Meson Python build:

```bash
git clone https://github.com/lukaswittmann/moist.git
git -C moist checkout 6e94f4a5841f5d2e1987e1ca496c7b9baad75e59
git -C moist apply \
  "$MAPLE/docs/implicit-solvation/patches/moist-6e94f4a-python-import.patch"
meson setup moist/build moist \
  --prefix="$MOIST_PREFIX" \
  -Dpython=true -Dpython_version="$(command -v python)"
meson compile -C moist/build
meson install -C moist/build
```

Real-MOIST nonzero-dipole canary:

```text
source raw order     [[0.0, 0.002, -0.001, 0.0015]]
surface points       194
E_pol                -4.06526157419331e-07 hartree
linear residual inf  below 1e-17
condition number 2   20.792984988145154
```

The Gaussian moments used by the callback are accumulated in explicit frozen
component order rather than by threaded BLAS GEMV. Under the audited default
OpenBLAS/OpenMP environment, 30 repeated bare surface builds produced one
within-runtime snapshot identity, and ten independently constructed
operational providers each passed their internal exact two-build cavity and
field replay. The real-backend nonlinear JVP agrees with an independent
centred finite difference and its JVP/VJP dot product closes to floating-point
precision.

Surface/operator/forward SHA256 values remain exact execution-state identities,
not portable scientific constants: changing MOIST or BLAS thread counts can
change legitimate last bits in parallel matrix assembly or linear algebra.
Portable canaries therefore use physical tolerances, while the fail-closed cold
replay compares exact identities inside one fixed runtime. These are
implementation tests, not chemical-accuracy evidence.

A separate three-centre water smoke produced 413 retained surface points with
owner counts `(175, 119, 119)`, finite fields, exact equality of the two
coupling evaluations, and exact same-runtime surface replay. Its JVP check includes
a zero-sum interatomic monopole redistribution as well as dipole directions;
the analytic reference JVP agrees with the centred finite difference and the
VJP dot product closes. This validates the Python/Fortran owner-index and
source-response boundaries beyond the single-atom canary; it is still not a
hydration-free-energy or QM-density benchmark.

## Admission gates that remain closed

`RhoDropCavityForceGateEvidence` is separate from the existing Route 2 force
certificate so archived fixed-cavity evidence is not reinterpreted. Every gate
defaults to false. In particular, production force admission still requires:

1. QM isodensity-surface, area, volume, outlying-charge, and topology audits;
2. robust projection/branch tests including diffuse anions and stretched
   complexes;
3. an efficient forward DROP JVP suitable for outer Krylov iterations;
4. complete coordinate VJP = explicit level-set field term + reference-anchor
   term;
5. Gate B component finite-difference agreement;
6. Gate C end-to-end Route 2 force/PES validation.

The pinned callback reports zero active atoms and zero mixed nuclear level-set
derivatives. Consequently, its current public gradient route does not supply
the complete reference-anchor derivative. `nuclear_field_vjp()` is named and
documented as the field term only. It must not be exposed as
`full_position_vjp()`.

The separately declared full-functional-drive v0 is metadata-only:
`energy_available=False`, and both provider and engine construction reject it.
It cannot be used to imply even an experimental energy implementation before
the full PCM source-gradient drive is wired and validated. That future
energy-only mode needs the implemented first-order source VJP; analytic forces
for it remain separately blocked because they additionally require continuum
source Hessian-vector and source-coordinate mixed response.

The preferred upstream addition is a contracted anchor-coordinate API, for
example a routine that maps surface `xi/f/xyz` weights directly to nuclear
coordinates. A local copy of MOIST's full geometry chain or a giant exported
fourth-order tensor would be less maintainable.

### Minimal upstream API work

The remaining gaps do not require a second cavity implementation. They map to
three narrow extensions of MOIST's existing DROP machinery:

1. **Versioned callback tolerance.** Add an optional `tolerance` argument to a
   versioned isodensity-callback constructor and pass it unchanged to the
   existing Fortran `new_cavity_drop(..., tolerance=...)`. The old constructor
   must retain its `1e-10` behavior; the MAPLE profile can then request and
   attest `1e-12` without changing a compiled global default.
2. **Forward surface tangent.** Add the forward-mode counterpart of
   `contract_surface_lsf_weights`: accept directional level-set value,
   gradient, and Hessian jets and return directional `xi/f/xyz` surface data.
   Its dot product with the existing reverse contraction must close before it
   replaces the current `4N` transpose-probe JVP.
3. **Contracted anchor coordinate VJP.** After `compute_gradient_drop`, contract
   arbitrary `w_xi/w_f/w_xyz` directly to `(n_atoms,3)`. The callback LSF must
   enumerate all centres for the anchor loop while continuing to report zero
   explicit mixed nuclear LSF derivatives; MAPLE then adds its separately
   tested analytic level-set field term.

Acceptance requires unchanged old-constructor tests, `1e-12` projection tests,
forward/reverse dot products, component finite differences, translational and
rotational covariance, and no exported `xyz1_rA`-sized production tensor.

The [DROP preprint](https://doi.org/10.26434/chemrxiv.15003893/v2) presents a
general differentiable implicit-surface discretization and CPCM validation,
but it does not by itself validate MAPLE's reconstructed MACE-POLAR density.
The [Q-Chem 5.4 manual](https://manual.q-chem.com/5.4/subsec_SS%28V%29PE.html)
also records that gradients were unavailable for its isodensity SS(V)PE
implementation, illustrating why an isodensity energy alone cannot certify
forces. Density-dependent continuum interfaces and their variational response
have separate precedent in the [SCCS formulation](https://arxiv.org/abs/1112.5332),
while smooth atom-centred surface discretization is independently discussed in
the [SWIG/SES literature](https://doi.org/10.1080/00268976.2019.1644384).

## Reproduction

Normal targeted tests:

```bash
python -m pytest -q \
  tests/solvation/test_route2_atomic_reference_density.py \
  tests/solvation/test_route2_density_levelset.py \
  tests/solvation/test_route2_moist_drop.py \
  tests/solvation/test_route2_rhodrop_cpcm.py \
  tests/solvation/test_route2_rhodrop_profile.py \
  tests/solvation/test_route2_derivative.py \
  tests/solvation/test_route2_force_admission.py \
  tests/solvation/test_route2_response.py \
  tests/solvation/test_route2_energy_ledger.py
```

Real pinned-MOIST tests additionally require the built package/library paths:

```bash
export PYTHONPATH="$MOIST_PREFIX/lib/python3.11/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$MOIST_PREFIX/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export MAPLE_ROUTE2_MOIST_REAL=1
export MAPLE_ROUTE2_MOIST_LIBRARY="$MOIST_PREFIX/lib/x86_64-linux-gnu/libmoist.so.0.6.0"
python -m pytest -q \
  tests/solvation/test_route2_moist_drop.py \
  tests/solvation/test_route2_rhodrop_cpcm.py \
  tests/solvation/test_route2_rhodrop_profile.py
```

The real integration test uses a constant-source electronic-model test double
to exercise the existing Route 2 SCF/energy ledger. It is not a
MACE-POLAR-checkpoint end-to-end calculation.

Fresh verification on 2026-08-08:

```text
normal repository suite                     1430 passed, 15 skipped
pinned real-MOIST rho-DROP suite             37 passed
real suite threading              OpenBLAS=20, MOIST OpenMP=10
```

The normal-suite skips are optional-runtime tests; the separately enabled
real-MOIST run above is the evidence for this adapter. Neither count promotes
the still-open QM-isosurface, force, PES, or chemical-accuracy gates.
