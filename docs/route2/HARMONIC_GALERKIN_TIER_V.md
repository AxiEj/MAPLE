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

## Implemented smooth geometry descriptor

`maple.solvation.continuum.harmonic_exposure` now implements the first
geometry-dependent part of the global-smoothness branch without introducing a
surface-point mask:

- `smooth_flat_step` is a compact `C-infinity` switch with exact buried and
  exposed plateaus;
- each pair factor is reduced through the invariant one-dimensional integral

  \[
    k_\ell(d)=2\pi\int_{-1}^{1}
      s\!\left(\frac{a_i^2+d^2-2a_i d t-a_j^2}{\delta}\right)
      P_\ell(t)\,dt,
  \]

  followed by

  \[
    [e_{ij}]_{\ell m}=k_\ell(d)Y_{\ell m}(\widehat d_{ij});
  \]

- products are formed only after every factor has a declared finite harmonic
  bandwidth. The backend rule is exact for the full finite algebraic degree of
  the product and output test harmonic, so it is coefficient contraction rather
  than sampling of a raw non-band-limited cavity field;
- `harmonic_multiplication_matrix` constructs
  \(P_L M_{e_i}P_L\) in complete real-irrep blocks;
- `SmoothHarmonicExposureSnapshot` binds atoms, Å coordinates/radii, the
  transition width in Å², harmonic orders, invariant radial rule, pair states,
  coefficient arrays, multiplication matrices, implementation source, and
  fixed topology to immutable SHA-256 identities.

The descriptor has constant dimensions and no active node/coefficient
deletion. Tests cover arbitrary continuous rotations, rigid translations,
label permutations, finite-band product commutativity, multiplication-matrix
covariance, cold replay, content tampering, compact-support screening, and a
central-difference sweep through pair tangency. The radial quadrature error can
change only the invariant scalars `k_l(d)`; it cannot select a laboratory
orientation.

This defines a regularized smooth weighted-overlap cavity descriptor. It is
not mathematically identical to the sharp union of spheres inside the switching
layer. The declared smooth domain excludes coincident sphere centres; that
singular geometry fails closed rather than receiving a body-frame fallback.

## Implemented eight-channel source intertwiner

`maple.solvation.continuum.harmonic_gaussian_source` now assembles the
geometry-dependent source map `S(R)` for the authoritative two-width,
eight-channel radial-GTO space:

- the `1.5 Å` and `3.0 Å` widths are declared once with the canonical source
  space rather than copied from a legacy adapter;
- Gaussian monopole coefficients use invariant one-dimensional radial
  integrals;
- dipole columns are analytic derivatives of those monopole coefficients with
  respect to the source centre. Angular derivatives use exact real `SO(3)` Lie
  algebra generators, not finite-difference rotations;
- the raw potential coefficients are contracted with the smooth exposure
  multiplication operator;
- the receiver is exactly `S(R).T` under the declared identity radial pairing.
  There is no separately implemented receiver physics.

The raw source operator agrees with an independent high-order projection of the
legacy analytic Gaussian point kernel. Dipole columns agree with source-centre
finite differences, while arbitrary continuous rotations satisfy

\[
  S(QR)D_c(Q)=D(Q)S(R)
\]

to floating-point/invariant-quadrature error. The public continuum package now
resolves adapters lazily, so importing this dependency-light source module does
not execute the legacy exact-GTO adapter or load Torch.

## What this slice does not implement

The stationary-continuum snapshot matrices are still external inputs. The
repository now assembles smooth overlap coefficients, their per-sphere
multiplication matrices, and the full eight-channel Gaussian source map
`S(R)` from geometry. It does **not** yet assemble the continuum Green matrix
`A(R)` from solid-harmonic translations. Consequently the combined slice does
not yet establish

\[
  A(QR)=D(Q)A(R)D(Q)^{-1}
\]

for a complete production continuum assembler. It proves the exposure and
source-map intertwining identities and that, once the remaining covariant Green
matrix is supplied, the finite representation, stationary scalar, exact
adjoint receiver, and derivative plumbing preserve that structure.

It is therefore not:

- a production ddPCM/ddCOSMO implementation;
- a physical-accuracy result;
- an analytic moving-cavity coordinate JVP/VJP;
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

The next continuum PR must use the smooth descriptor and source map in a
geometry-bound Green-operator assembler whose only directional inputs are
inter-centre displacement vectors and whose blocks are built from
solid-harmonic translation/addition formulas or equivalent STF tensor
contractions. It must then differentiate those blocks, the source map, and the
exposure coefficients from the same scalar graph. The current dense exact
finite-band contraction is a bounded reference; a production implementation
may replace it with exact Gaunt/Clebsch-Gordan contractions without changing
the coefficient contract.

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
