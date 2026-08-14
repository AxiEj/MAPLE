# Harmonic-Galerkin Tier-V continuum boundary

## Decision

A conventional finite set of laboratory-fixed surface directions with a
pointwise buried/exposed mask is not a structural `SO(3)` representation. Grid
refinement can reduce its rotation error, but no fixed order proves global
continuous-rotation covariance for a generic sharp or smooth mask.

The strict Route-2 continuum branch therefore targets:

1. fixed, complete per-atom coefficient spaces
   \(V_i=\bigoplus_{\ell=0}^{L}\mathcal H_\ell\);
2. analytic or exact finite-band Galerkin assembly satisfying
   \(A(QR)=D(Q)A(R)D(Q)^{-1}\);
3. a source map satisfying
   \(S(QR)D_c(Q)=D(Q)S(R)\);
4. a receiver that is the exact discrete adjoint \(S^\dagger\);
5. one stationary electrostatic scalar.

A finite quadrature is allowed only as an exact backend for a proven finite
band-limited coefficient contraction. It must not sample a raw cavity mask or
an unprojected non-band-limited exposure function.

## Implemented reference slice

`maple.solvation.continuum.harmonic_galerkin` now freezes the following
dependency-light algebra:

- `PerAtomHarmonicSpace`: immutable atom-major complete real-harmonic irrep
  blocks with dimension `atom_count * (L + 1)^2`; reference contract v1 is
  fail-closed above `L=16` to bound dense diagnostic allocations;
- `real_wigner_matrix`: the coefficient action of
  \(f(u)\mapsto f(Q^{-1}u)\), contracted with a rule exact for the retained
  finite harmonic product;
- `radial_gto_source_rotation_matrix`: the exact active rotation in the
  authoritative two-width, eight-channel MACE-POLAR raw order;
- `FixedHarmonicGalerkinSnapshot`: content-addressed, immutable symmetric
  positive-definite \(A\) and source operator \(S\) for an independently
  supplied fixed cavity descriptor;
- `FixedHarmonicGalerkinCPCMCandidate`: a sealed scalar-first functional

  \[
    G(c)=-\frac12(Sc)^T A^{-1}(Sc).
  \]

Its drive, source HVP, JVP, and VJP come from the same Torch scalar graph. No
independent receiver or response implementation exists. Explicit coefficient
conjugation uses

\[
  A' = D A D^T,
  \qquad
  S' = D S D_c^T,
\]

and preserves the scalar and drive covariance to floating-point/solve error.

## What this slice does not implement

The snapshot matrices are external inputs. The repository does **not** yet
assemble them from nuclear geometry by solid-harmonic translations, analytic
Gaussian source coefficients, or coefficient-space overlap products.
Consequently this slice does not establish

\[
  A(QR)=D(Q)A(R)D(Q)^{-1}
\]

for a production geometry assembler. It proves only that once covariant
coefficient matrices are supplied, the finite representation, stationary
scalar, exact adjoint receiver, and derivative plumbing preserve that
structure.

It is therefore not:

- a production ddPCM/ddCOSMO implementation;
- a physical-accuracy result;
- a moving-cavity coordinate derivative;
- a conservative total Route-2 nuclear force;
- a completed electronic-continuum common functional;
- a Tier E/F/H/V/M admission.

The registered scalar/profile are distinct, disabled identities:

```text
route2-variational-macepolar-energygradient-fixedcavity-
harmonicgalerkin-cpcm-v1

route2-profile-variational-macepolar-energygradient-fixedcavity-
harmonicgalerkin-cpcm-v1
```

All capability bits remain false and there is no admission evidence.

## Moving-cavity choice remains explicit

The first reference treats the cavity descriptor as an independent fixed
input. A later moving-cavity branch must choose exactly one contract.

### Global differentiability

Use a rotationally scalar `C1` or preferably `C-infinity` overlap/partition
field, form its harmonic multiplication matrices in coefficient space, and
differentiate every coefficient through the same stationary scalar. This is a
regularized weighted-overlap model, not the exact sharp union boundary inside
the switching layer.

### Exact sharp union

Integrate sharp cap/spherical-polygon coefficients covariantly, certify a
fixed-topology stratum, and fail closed before tangency, patch birth/death,
triple-intersection changes, or loss of operator invertibility. The correct
claim is then exact `SO(3)` covariance and piecewise `C1`, not a global `C1`
PES.

Source-dependent rho-DROP and nonpolar terms remain outside the first strict
fixed-cavity milestone.

## Next implementation gate

The next continuum PR must provide a geometry-bound assembler whose only
directional inputs are inter-centre displacement vectors and whose blocks are
built from solid-harmonic translation/addition formulas or equivalent STF
tensor contractions. Exposure products must use exact Gaunt/Clebsch-Gordan
coefficient algebra or another proven covariant projection.

Before any capability is admitted, verify:

1. operator block covariance;
2. source-map covariance for the full eight-channel source;
3. the `S/S-dagger` dot-product identity;
4. scalar rotation invariance and drive/force covariance;
5. primal/adjoint residual and minimum-eigenvalue bounds;
6. energy/drive directional derivatives;
7. basis convergence in `L` separately from rotation covariance;
8. the model-side energy/source conjugacy, sign, gauge, passivity, root, and
   envelope-coordinate gates.

No failure may silently fall back to the legacy lab-grid mask, a principal
axis frame, finite subgroup averaging, or a detached receiver.
