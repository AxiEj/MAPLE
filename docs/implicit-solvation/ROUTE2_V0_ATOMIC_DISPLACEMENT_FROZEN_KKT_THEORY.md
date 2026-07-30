# Route-2 V0-ADT: frozen-source direct-sum KKT gate

## Status and scope

`route2_v0_atomic_displacement_frozen_kkt.py` gives V0-ADT a single,
fixed-geometry **distributional source space** without projecting its
free-atom radial translation tangent into a chosen Gaussian width.  It joins:

1. the unmodified zero-field MACE-POLAR point monopole/dipole source;
2. the frozen free-atom V0-ADT induced-density tangent; and
3. one energy-conjugate reciprocal continuum response.

The result is a structural common-scalar gate only.  It does **not** admit a
physical V0 continuum, a full stationary electronic density functional, a
nonpolar term, forces, a PES, or a solvation prediction.  Its tests use a
synthetic linear reciprocal continuum and read no QM solvent or experimental
solvation value.

The machine-readable boundary is
[`route2-v0-atomic-displacement-frozen-kkt-prereg-v1.json`](benchmarks/route2-v0-atomic-displacement-frozen-kkt-prereg-v1.json).
The distinct continuum-admission audit remains negative:
[`route2-v0-atomic-displacement-continuum-admission-audit-v1.json`](benchmarks/route2-v0-atomic-displacement-continuum-admission-audit-v1.json).

## 1. Why V0-ADT must not be fitted into a GTO radial channel

The V0-ADT source is fixed by the enclosed free-atom electron count
(N_Z(r)).  Its charge-density tangent can be stated weakly as

\[
\eta_{a i}(\mathbf r)
=-\frac{1}{Z_a}\partial_i n_{Z_a}
 (\lvert\mathbf r-\mathbf R_a\rvert),
\qquad
\int\eta_{a i}=0,
\qquad
\int r_j\eta_{a i}=\delta_{ij}.
\]

Its Coulomb potential is exactly the already registered V0-ADT potential.  A
finite GTO expansion would require a projection metric, a radial set, and a
residual rule.  The unrepresented potential couples to a continuum charge as
(q^\mathsf T B(\eta-\Pi\eta)); choosing a width, metric, or residual
threshold after seeing an error would recreate the prohibited radial tuning
problem.  Therefore this branch makes **no** GTO projection.

## 2. One direct-sum source/dual representation

Let (c_0\in\mathbb R^{4N}\) be the frozen raw MACE point-multipole
coefficients and (p\in\mathbb R^3) the induced molecular dipole in
(e\,a_0).  Define the distributional source space

\[
\mathcal X_{\mathbf R}=
\operatorname{span}\left\{
\delta_a,\,-\partial_i\delta_a,\,\eta_{a i}
\right\}.
\]

`PointMultipoleSurfaceCoupling` builds the permanent map (S_{\mathbf R})
from the exact existing `point_multipole_potential` convention.  The existing
`AtomicDisplacementSurfaceCoupling` builds the induced map
(B_{\mathbf R}) directly from V0-ADT.  They must share exactly the same
nuclear geometry and continuum surface.  The complete source map is

\[
v_{\mathbf R}(c_0,p)=S_{\mathbf R}c_0+B_{\mathbf R}p,
\qquad
\mathcal S_{\mathbf R}=[S_{\mathbf R}\;B_{\mathbf R}].
\]

For integrated surface charge (q), the only permitted reaction dual is

\[
\mathcal S_{\mathbf R}^{\mathsf T}q=
\begin{bmatrix}
S_{\mathbf R}^{\mathsf T}q\\
B_{\mathbf R}^{\mathsf T}q
\end{bmatrix},
\qquad
q^\mathsf Tv=c_0^\mathsf TS_{\mathbf R}^{\mathsf T}q
+p^\mathsf TB_{\mathbf R}^{\mathsf T}q.
\]

This is one direct-sum source/dual pairing; it is not a claim that the frozen
MACE coefficients are an all-electron density or that the two families share
a Gaussian radial basis.

## 3. The fixed-source stationary scalar

Let (q=Q_{\mathbf R}v) be a **linear**, energy-conjugate reciprocal
continuum response and let the frozen MACE-MDP polarizability satisfy
\(\alpha_\theta\succ0\).  With (c_0) held fixed, the only stationary
variable is (p):

\[
\boxed{
g_{\mathbf R}(p;c_0)=
\frac12p^\mathsf T\alpha_\theta^{-1}p+
\frac12\left(S_{\mathbf R}c_0+B_{\mathbf R}p\right)^\mathsf T
Q_{\mathbf R}
\left(S_{\mathbf R}c_0+B_{\mathbf R}p\right).}
\]

The Euler equation is

\[
\left(\alpha_\theta^{-1}+B^\mathsf TQB\right)p
+B^\mathsf TQS c_0=0.
\]

The implementation obtains the permanent driving dual (B^\mathsf TQS c_0)
from the exact direct-sum transpose, delegates the induced solve to the
reduced KKT kernel, and then independently verifies that the full restricted
matrix is reciprocal,

\[
\mathcal S^\mathsf TQ\mathcal S=
(\mathcal S^\mathsf TQ\mathcal S)^\mathsf T,
\qquad \mathcal S=[S\;B],
\]

not merely its induced block \(B^\mathsf TQB\). It also requires homogeneous
linearity \(Q0=0\) and, at the declared \((c_0,p)\), independently verifies

\[
Q(S c_0+Bp)=QSc_0+QBp.
\]

These checks reject a cross-source nonreciprocal operator or an affine/nonlinear
provider even if it happens to look reciprocal in the three induced columns.

At the stationary solution, the reported ledger is only

\[
 g_{\rm el}=\frac12p^\mathsf T\alpha_\theta^{-1}p,
\qquad
 g_{\rm cont}=\frac12v^\mathsf Tq,
\qquad
 g=g_{\rm el}+g_{\rm cont}.
\]

The frozen-source scaling test verifies the envelope identity

\[
\frac{d g^*(s c_0)}{ds}\bigg|_{s=1}
=(S c_0)^\mathsf Tq^*.
\]

It is a mathematical differentiation check, **not** permission to rescale a
MACE density in a physical calculation.

## 4. What this does and does not solve

This construction removes a precise ambiguity: the permanent and induced
sources now have one exact surface pairing, so neither a separate receiver
map nor a GTO fitting error can enter the induced KKT ledger.  It also makes
V0-ADT response reciprocal and passive whenever the declared joint curvature
is positive.

It does **not** supply \(\partial F/\partial c_0=0\).  The permanent MACE
source remains a geometry-dependent frozen external source exactly as in
V0-FD.  Consequently it cannot yet be called the target Route-2V functional

\[
F_\theta(\mathbf R,c)+\frac12\sigma^\mathsf TA_{\mathbf R}\sigma+
\sigma^\mathsf TB_{\mathbf R}c.
\]

Nor does it solve the negative continuum audit: no checked-in PCMSolver, SWIG,
or diffuse-continuum configuration has independently sourced V0 cavity data
and a smooth-coordinate certificate.  No physical PCM run, QM solvent panel,
or FreeSolv/MNSol accuracy run is allowed from this kernel.

## 5. Next gates

1. Preserve the direct-sum source identity under rigid motion and validate its
   explicit coordinate derivatives before any force claim.
2. Bind a continuum only after its cavity/solvent construction and coordinate
   smoothness are independently source-bound; do not pick a radius from an
   error.
3. Run the frozen twelve-record, ten-functional-group **QM physics** panel
   without reading solvation labels.
4. Choose between an honestly frozen-source V0 endpoint and a future full
   electronic functional only before the experimental protocol is opened.
5. Freeze all choices, then and only then evaluate every record of the
   historical FreeSolv10 and broader development, confirmation, and blind
   panels against the strict maximum-error gates.
