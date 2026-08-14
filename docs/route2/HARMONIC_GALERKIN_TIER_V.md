# Harmonic-Galerkin Tier-V continuum boundary

## Decision

A conventional finite set of laboratory-fixed surface directions with a
pointwise buried/exposed mask is not a structural `SO(3)` representation. Grid
refinement can reduce its rotation error, but no fixed order proves global
continuous-rotation covariance for a generic sharp or smooth mask.

The strict Route-2 continuum branch therefore uses:

1. fixed complete per-atom coefficient spaces
   \(V_i=\bigoplus_{\ell=0}^{L}\mathcal H_\ell\);
2. invariant coefficient assembly satisfying
   \(A(QR)=D(Q)A(R)D(Q)^T\);
3. a source map satisfying
   \(S(QR)D_c(Q)=D(Q)S(R)\);
4. the receiver defined only as the exact discrete adjoint \(S^T\);
5. one stationary electrostatic scalar.

A finite quadrature is allowed only for invariant scalar coefficients or as an
exact backend for a declared finite-band contraction. It must not sample a raw
cavity mask or an unprojected non-band-limited exposure field.

## Implemented coefficient algebra

`maple.solvation.continuum.harmonic_coefficients` and
`harmonic_galerkin` provide:

- immutable atom-major complete real-harmonic irrep blocks;
- real Wigner matrices and Lie-algebra generators;
- the exact active rotation in the authoritative two-width, eight-channel
  radial-GTO source space;
- immutable symmetric-positive-definite coefficient snapshots;
- the sealed same-scalar candidate

  \[
    G(c)=-\frac12(Sc)^T A^{-1}(Sc).
  \]

The drive, source HVP, JVP, and VJP of the fixed snapshot are generated from
that Torch scalar. No independently coded receiver or force expression exists.

## Smooth rotationally scalar exposure

`harmonic_exposure` builds compact `C-infinity` pair switches through invariant
one-dimensional Legendre coefficients. Products are projected only after each
factor has a declared finite bandwidth. This produces a fixed-dimensional,
content-addressed regularized weighted-overlap descriptor with no active node
or coefficient deletion.

It is not the exact sharp union of spheres inside the switching layer. The
noncoincident-centre domain is explicit; coincident centres fail closed.

### The required rectangular product embedding

Let the retained unknown on chart \(i\) be

\[
  \alpha_i(u)=\sum_{\ell\le L,m}\alpha_{i\ell m}Y_{\ell m}(u),
\]

and let the finite exposure approximation have order \(L_e\). The physical
weighted shell charge is

\[
  q_i(u)=e_i(u)\alpha_i(u).
\]

The complete product belongs to \(V_P\), \(P=L+L_e\). The implementation
therefore constructs the rectangular exact finite-product map

\[
  E_i:V_L\longrightarrow V_P,
  \qquad E_i\alpha_i=[e_i\alpha_i]_{V_P},
\]

and the block-diagonal global map \(E\).

This is not interchangeable with the square matrix
\(P_LM_{e_i}P_L\). If an invertible square \(M\) is placed in both
\(A=MKM\) and \(S=MV\), then

\[
  -\frac12(Mv)^T(MKM)^{-1}(Mv)=-\frac12v^TK^{-1}v,
\]

so exposure cancels and merely reparameterizes the unknown. The rectangular
embedding is therefore a physical requirement, not an optimization detail.
Incomplete product bandwidths fail closed.

The transition layer can contain weighted contributions from more than one
chart. That is the declared regularized multi-shell model, not an assertion
that it is exactly the sharp union boundary or an exact partition of unity.

## Eight-channel Gaussian source and exact receiver

`harmonic_gaussian_source` assembles the raw Gaussian boundary-potential map
\(V(R)\) in the physical product space \(V_P\):

- both `1.5 Angstrom` and `3.0 Angstrom` radial widths are included;
- monopoles use invariant one-dimensional radial integrals;
- dipoles are analytic source-centre derivatives using real `SO(3)`
  generators;
- the retained-test source operator is

  \[
    S(R)=E(R)^T V(R);
  \]

- the only receiver is exactly \(S(R)^T\) under the declared radial pairing.

The raw map agrees with an independent high-order analytic-kernel projection,
and the weighted map satisfies
\(S(QR)D_c(Q)=D(Q)S(R)\) to floating-point/invariant-quadrature error.

## Geometry-assembled Coulomb single layer

`harmonic_single_layer` now assembles the physical Coulomb bilinear matrix
\(K_P(R)\) for charge per unit solid angle on complete per-sphere \(V_P\)
blocks. It covers nested, intersecting, tangent, and separated distinct-centre
spheres without a laboratory grid.

### Self block

For orthonormal real harmonics the exact self-sphere eigenvalue is

\[
  K_{i,\ell m; i,\ell' m'}
  = k_e\frac{4\pi}{a_i(2\ell+1)}
    \delta_{\ell\ell'}\delta_{mm'},
\]

where

\[
  k_e=(\text{Hartree-to-eV})(\text{Bohr-in-Angstrom})
\]

has units `eV Angstrom / e^2`.

### Cross block

For a pair aligned with the positive pair axis, the source-sphere harmonic is
integrated analytically first:

\[
  \int_{S^2}\frac{Y_{\ell m}(u')}{|x-a_j u'|}\,d\Omega'
  =\frac{4\pi}{2\ell+1}
  \begin{cases}
    r^\ell/a_j^{\ell+1}, & r<a_j,\\
    a_j^\ell/r^{\ell+1}, & r>a_j,
  \end{cases}
  Y_{\ell m}(\widehat x).
\]

The remaining target polar integral is one-dimensional and is split at the
sphere-intersection cosine. The finite azimuthal contraction is exact for the
retained bandwidth. Consequently no two-dimensional singular surface
quadrature remains even when the shells intersect.

The canonical pair-axis block commutes with the complete `SO(2)` stabilizer.
For any proper rotation \(Q_d\) mapping `+z` to the pair direction,

\[
  K_{ij}(d)=D(Q_d)K_{ij}^{z}(|d|)D(Q_d)^T.
\]

Any other section is \(Q_dR_z\); stabilizer commutation makes the reconstructed
block independent of that arbitrary transverse gauge. Reverse blocks are the
exact transpose, not a separately approximated receiver-side kernel.

## Complete weighted stationary assembly

`harmonic_weighted_galerkin` composes the three geometry-bound objects:

\[
  A(R)=E(R)^T K_P(R)E(R),
  \qquad
  S(R)=E(R)^T V(R),
\]

and freezes them into one `FixedHarmonicGalerkinSnapshot`. The parent scalar is

\[
  \Phi(R,c,\alpha)
  =\frac12\alpha^TA(R)\alpha-\alpha^TS(R)c,
\]

with stationary equation \(A\alpha=Sc\) and reduced scalar
\(G=-\tfrac12(Sc)^TA^{-1}(Sc)\).

Symmetry of \(A\) is by construction. In exact arithmetic, the Coulomb energy
is strictly positive for nonzero finite harmonic densities on distinct,
noncoincident spherical supports. The implementation additionally requires:

- positive raw \(K_P\) eigenvalues;
- full column rank of the rectangular weighted basis \(E\);
- positive weighted \(A\) eigenvalues;
- bounded condition numbers.

A fully buried compact-support chart can make \(E\) rank deficient. That state
fails closed. Coefficients are not deleted and eigenvalues are not clipped or
regularized, because either action would silently define different physics.

## Same-scalar moving-geometry differentiation candidate

`harmonic_torch_functional` independently reassembles the complete dependency
path

\[
  R\longmapsto E(R),K_P(R),V(R)
  \longmapsto A(R)=E^TK_PE,\ S(R)=E^TV
  \longmapsto -\tfrac12(Sc)^TA^{-1}(Sc)
\]

inside one sealed Torch scalar graph.  The public drive, source HVP/JVP/VJP,
fixed-source coordinate partial, and mixed drive/coordinate pullback are all
generated by `ContinuumEnergyFunctional`; the candidate cannot override them
with independent formulas.

The Torch implementation is checked matrix-by-matrix against the independent
NumPy/SciPy assembly.  Its Cartesian harmonic recurrence is polynomial at the
polar axes.  The pair-axis section uses two smooth local charts, and the
canonical block's exact `SO(2)` stabilizer removes their transverse gauge.
An explicit polar-axis regression prevents the earlier invalid shortcut of
returning a constant rotation matrix, which would erase transverse coordinate
derivatives at `+z` or `-z`.

This closes the implementation gap for first coordinate derivatives of the
diagnostic continuum scalar.  It does not by itself admit a total Route-2
force: the changed electronic model, common stationary root, topology/domain
guards, real-checkpoint panels, and release evidence remain separate gates.

## Exact claim boundary

The construction is structurally `SO(3)` covariant in its finite coefficient
representation. Changing `L`, `L_e`, or invariant radial quadrature changes
physical approximation accuracy, but must not systematically change rotation
covariance.

The exposure coefficients and Gaussian source are smooth on the declared
noncoincident domain. The exact delta-shell Coulomb cross operator is globally
`C1` but generally not `C2` at internal or external sphere tangency. Tests lock
the continuous first derivative and the second-derivative jump of the analytic
`l=0` shell block. Therefore this candidate supports only the following
regularity statement:

> `C1` on distinct-centre, full-rank, nonsingular weighted-shell states; no
> `C2`, Hessian, or frequency claim across shell tangencies.

The same-scalar Torch candidate now supplies analytic/AD fixed-source
coordinate partials and mixed drive/coordinate pullbacks.  Coordinate HVPs,
Hessians, and frequency admission are not implemented.  The present code does
not publish a nuclear force: callable internal derivatives do not grant a
capability.

## Executed structural evidence

The tests cover:

- dependency-light imports without Torch or legacy continuum adapters;
- exact self-sphere spectrum and independent Newton-shell `l=0` values;
- reciprocity across nested, intersecting, tangent, and separated spheres;
- pair-axis `SO(2)` commutation and transverse-gauge independence;
- positive definiteness and fail-closed coincident centres;
- arbitrary `SO(3)` block covariance, translation, and atom permutation;
- invariant-radial-quadrature convergence without rotation drift;
- `C1`/not-`C2` tangency behavior;
- exact rectangular finite products and their covariance;
- the square-exposure cancellation negative canary;
- `A=E.T K E`, `S=E.T V`, one-sphere analytic conductor energy;
- scalar rotation invariance, exact `S/S.T` adjoint identity, and source
  directional derivatives;
- matrix-by-matrix Torch/NumPy parity for `E`, `K`, `V`, `A`, and `S`;
- same-scalar coordinate partials and mixed pullbacks against central finite
  differences over multiple steps;
- energy and coordinate-gradient rotation/translation/permutation covariance
  to float64 roundoff, including an exactly polar pair axis;
- fixed dimensions and a continuous stationary scalar through external
  tangency;
- full-burial rank loss, tampering, and cold replay failures.

These are algebraic/representation results, not chemical validation.

## Capabilities deliberately left disabled

This slice is not:

- production ddPCM/ddCOSMO;
- the exact sharp union-of-spheres PCM boundary;
- a calibrated or accuracy-validated solvent model;
- a coordinate Hessian or tangency-crossing Tier-H domain;
- a conservative total Route-2 nuclear force;
- a real-checkpoint-admitted electronic-continuum common functional;
- an admitted Tier E/F/H/V/M profile.

The generic disabled common-stationarity kernel is implemented and accepts a
fixed external harmonic snapshot. Its present test uses a synthetic
same-scalar electronic oracle. That proves software composition only; it does
not establish the real checkpoint's sign, stability, unique root, coordinate
envelope, or Tier V admission.

The registered scalar/profile remain distinct disabled identities:

```text
route2-variational-macepolar-energygradient-fixedcavity-
harmonicgalerkin-cpcm-v1

route2-profile-variational-macepolar-energygradient-fixedcavity-
harmonicgalerkin-cpcm-v1
```

All capability bits remain false. No failure may fall back to a lab-grid mask,
body frame, finite rotation average, active coefficient deletion, detached
receiver, or empirical diagonal regularizer.

## Next gates

Before any admission, the implementation still needs:

1. analytic or scalar-AD coordinate JVP/VJP of exposure, `E`, `K`, and `S`;
2. force/energy directional derivatives across the full declared geometry
   domain, with explicit tangency/Hessian guards;
3. real-checkpoint sign and gauge validation of the integrated scalar-first
   eight-channel model/common-state kernel;
4. passivity, root uniqueness, and combined-Hessian gates;
5. real-checkpoint component and PES panels;
6. clean source/model/runtime-bound release evidence.

Tier V remains fail closed until every model and continuum gate passes.
