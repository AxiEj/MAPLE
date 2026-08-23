# Smooth-partition harmonic ddCOSMO candidate

## Decision

The old weighted-shell scalar

\[
A=E(e)^T K E(e),\qquad b=E(e)^T Vc,
\qquad G=-\tfrac12 b^T A^{-1}b
\]

is structurally rejected for a globally smooth moving union-of-spheres cavity.
Multiplying one chart by any scalar exposure amplitude \(\alpha>0\) changes
only its coefficient parameterization; it does not remove the physical trial
range. At \(\alpha=0\), complete burial changes the rank. Unless every removed
direction is invisible to every admitted source, the minimized energy has a
jump. A pseudoinverse, diagonal jitter, a positive exposure floor, or silent
chart deletion does not repair that mathematical defect.

The replacement keeps a fixed block of local reaction-potential coefficients
\(X_i\in\bigoplus_{l=0}^L\mathcal H_l\) on every atom sphere. It solves the
smooth Schwarz system

\[
(LX)_i=X_i-
\sum_{j\ne i}P_L[\omega_{ij}T_{ij}X_j]
=-P_L[e_i\Phi].
\]

The diagonal is the identity because \(X_i\) is a local potential trace, not
an apparent surface-charge coefficient. A fully buried chart therefore becomes
a consistency equation rather than a null block.

## Smooth partition

For pair inside factors \(f_{ij}\), whose underlying continuous support is
contained inside ball \(j\), define

\[
e_i=\prod_j(1-f_{ij}),
\qquad
\omega_{ij}=f_{ij}\int_0^1
\prod_{k\ne j}(1-tf_{ik})\,dt.
\]

Then

\[
e_i+\sum_j\omega_{ij}=1.
\]

For Boolean overlap factors with \(m\) covering balls, every active overlap
weight is \(1/m\), matching hard ddCOSMO equal sharing. Complete harmonic
irreps and invariant one-dimensional Legendre coefficients make every finite
truncation exactly SO(3)-covariant in exact arithmetic. Compact smooth weights
are not finite-band; auxiliary harmonic convergence remains a numerical gate.

## Scalar and derivatives

For point \(l\le1\) source coefficients \(c\), the implementation assembles

\[
LX=-Bc,
\qquad
G_{\rm ddCOSMO}(R,c)=
\frac{f_\epsilon}{2}c^TCX,
\qquad
f_\epsilon=\frac{\epsilon-1}{\epsilon}.
\]

\(C\) evaluates each local reaction potential and gradient at its atom centre.
The Schwarz matrix is generally nonsymmetric, so there is no claimed physical
quadratic in \(X\) alone. Torch differentiates the on-shell scalar through the
linear solve, which is equivalent to the standard primal-adjoint derivative
Lagrangian. Source gradients, coordinate gradients, mixed derivatives, and HVP
actions therefore come from one scalar graph.

This finite-dielectric scaling is ddCOSMO/CPCM, not full ddPCM. The next
continuum milestone is the exact-coefficient two-stage ddPCM system

\[
A_\epsilon G=A_\infty F,
\qquad
LX=G,
\]

with its composite adjoint.

## Frozen accuracy result

The preregistered MNSol-10/10-solvent profile uses:

- official unmodified MACE-POLAR-1-M zero-field point-\(l\le1\) source;
- SMD solvent-dependent Coulomb radii;
- \(L=5\), auxiliary partition cutoff 10;
- transition width \(0.18\ \text{A}^2\);
- invariant radial orders 96/128;
- PySCF SMD-CDS, unchanged from the point-ddPCM baseline.

Result:

| metric | value (kcal/mol) |
|---|---:|
| MAE | **0.9194258754** |
| RMSE | 1.0261979389 |
| maximum absolute error | 1.6479417114 |
| target | MAE <= 1.5 |

The profile passes the frozen accuracy target. Against the same point-source
pyddx ddPCM records, its mean absolute polarization-component difference is
0.1921734816 kcal/mol and the maximum is 0.6284443996 kcal/mol.

This does **not** admit public E/F/H/V/M. Remaining mainline work is:

1. exact-coefficient finite-dielectric ddPCM;
2. a checkpoint-native eight-channel continuum-to-MACE-POLAR field receiver;
3. mutual self-consistency and its operational implicit adjoint;
4. distorted-PES, root/domain, component, and release gates.

## Evidence

- preregistration:
  `docs/implicit-solvation/benchmarks/route2-mnsol10-harmonic-ddcosmo-preregistration-v1.json`
- aggregate result:
  `docs/route2/evidence/mace-polar-point-l1-harmonic-ddcosmo-mnsol10-accuracy-v1.json`
- implementation:
  `maple/solvation/continuum/harmonic_schwarz_primitives.py`
  and `maple/solvation/continuum/harmonic_ddcosmo_functional.py`
- tests:
  `tests/route2_vnext/test_harmonic_ddcosmo_functional.py`
