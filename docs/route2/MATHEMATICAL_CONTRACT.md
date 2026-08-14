# Route 2 mathematical contract

This document is the normative mathematical boundary for the conservative
Route 2 kernel. Implementation capability and executed evidence are recorded
separately; an equation appearing here does not by itself open a public gate.

## 1. Spaces and the only energy pairing

For nuclear coordinates \(R\in\mathbb R^{3N}\), the electronic source is
\(c\in\mathcal C_q\), the energy-dual field is \(u\in\mathcal C^*\), and the
source/field pairing is

\[
\langle c,u\rangle_Q=c^\mathsf TQ(R)u.
\]

`Q` owns component order, spherical/Cartesian permutation, signs, gauge, and
unit conversion. Model and continuum adapters may not repeat those choices.
The source satisfies the affine charge constraint

\[
Ac=q_{\rm tot},\qquad c=c_{\rm ref}(R)+T(R)y,\qquad AT=0,
\]

where \(y\) is dimensionless. Residual and Krylov norms are evaluated in
\(y\), never in an unscaled concatenation of monopoles and dipoles.

## 2. Operational conservative PES

The production-v1 target uses the state equation

\[
r(R,y)=T^+\left[c_{\rm ref}+Ty-
\Pi_q M_\theta\!\left(R,P_R(c_{\rm ref}+Ty)\right)\right]=0.
\]

The reported scalar is

\[
\Phi_{\rm op}(R,y)=E_{\rm vac}(R)+
G_{\rm pcm}(R,c_{\rm ref}+Ty)+G_{\rm np}(R,c_{\rm ref}+Ty).
\]

For the first force-admitted fixed-topology reciprocal C-PCM profile,

\[
G_{\rm pcm}(R,c)=\tfrac12\langle c,P_R(c)\rangle_Q,
\qquad G_{\rm np}=0.
\]

For the unique admitted root \(y^*(R)\), the reduced PES is

\[
E_{\rm op}(R)=\Phi_{\rm op}(R,y^*(R)).
\]

The field-conditioned MACE energy difference is diagnostic only. It is not
silently added to this half-coupling scalar.

In the current radial-GTO candidate, the learned checkpoint's four source
coefficients are embedded into the `(sigma=1.5 Angstrom, l<=1)` block of an
eight-channel physical space; the independent `sigma=3.0 Angstrom` source block
is fixed to zero while both receiver-width field blocks drive the model through
the checkpoint projection transform. This defines one differentiable
operational state equation and one conjugate continuum operator. It does not
prove that the learned source is a physically calibrated finite-width density,
nor that this candidate has passed release admission.

The disabled ordered-pair-frame candidate evaluates that same half-coupling
with a distinct rotationally equivariant continuum discretization. For every
ordered atom pair \((i,j)\), a body frame is formed from
\(R_j-R_i\) and the nuclear-charge centroid offset from the pair midpoint. Its
member weight is \(w_{ij}=|(R_j-R_i)\times v_{ij}|^8\), and the declared map is

\[
P_R^{\rm pf}=\frac{\sum_{i\ne j}w_{ij}\,U_{ij}^{-1}
P_{U_{ij}R}^{\rm body}U_{ij}}{\sum_{i\ne j}w_{ij}}.
\]

The coordinate VJP differentiates the member C-PCM operator, body-frame
rotation, centroid, and normalized weights. This is a new ensemble C-PCM
discretization, not a claim that one conventional laboratory-fixed Lebedev
cavity was made exact. Exactly singular members remain in the topology with
zero weight; because \(w\) and its first derivative vanish there, their bounded
fallback orientation does not enter the map or force. Fully collinear
geometries fail closed. It remains diagnostic until the complete real-stack
panel and component-accuracy gates pass.

## 3. One adjoint and one force

Define

\[
\mathcal L(R,y,\lambda)=\Phi_{\rm op}(R,y)-\lambda^\mathsf Tr(R,y).
\]

The adjoint and total derivative are

\[
r_y^\mathsf T\lambda=\Phi_y^\mathsf T,
\qquad
\nabla_RE_{\rm op}=\Phi_R-\lambda^\mathsf Tr_R,
\qquad
F_{\rm Route2}=-\nabla_RE_{\rm op}.
\]

There is no independently coded force formula. A returned force must bind to
one scalar ID, state-equation ID, root identity, adjoint solve, and evidence
artifact.

## 4. Full block reference

Eliminating continuum variables is an implementation choice. The reference
form supports \(z=(y,\sigma,\ldots)\), a state equation \(g(R,z)=0\), and one
scalar \(\Phi(R,z)\):

\[
g_z^\mathsf T\lambda=\Phi_z^\mathsf T,
\qquad
\nabla_RE=\Phi_R-\lambda^\mathsf Tg_R.
\]

A Schur-complement or matrix-free implementation must be numerically
equivalent to this system and pass independent JVP/VJP tests.

## 5. Source-dependent reaction maps

For a nonlinear/source-dependent map \(f(c)\), the declared operational
half-coupling is

\[
G(c)=\tfrac12c^\mathsf TQf(c).
\]

Its source derivative is

\[
\nabla_cG=\tfrac12Qf(c)+\tfrac12J_f(c)^\mathsf TQc.
\]

Both terms are mandatory. If a profile instead represents charging work, it
must use a separately versioned scalar such as

\[
G_{\rm chg}(R,c)=\int_0^1\langle c,P_R(\lambda c)\rangle_Q\,d\lambda
\]

and differentiate that exact integral/quadrature. Charging work and half
coupling are not interchangeable.

## 6. Coupling adjoint

One operator supplies both directions:

\[
v_{\rm surface}=B(R)c,\qquad
f_{\rm model}=B(R)^*\sigma,
\]

with

\[
\langle Bc,\sigma\rangle_{\rm surface}
=\langle c,B^*\sigma\rangle_Q.
\]

An exact-GTO energy and a local-jet force cannot share a profile ID.

## 7. Optional strict variational capability

Tier V is disabled by default. A model may enter it only if the public
energy-dual convention satisfies

\[
D_uE_\theta(R,u)[\delta u]
=\langle M_\theta(R,u),\delta u\rangle_Q
\]

over the admitted finite-field domain, together with reciprocity, passivity,
local invertibility/root uniqueness, sign/gauge/origin, common-checkpoint, and
full-coordinate-derivative gates. Passing implementation JVP/VJP transpose
tests is not proof of this physical identity.

Until those gates pass, Route 2 is named an **operational self-consistent
differentiable surrogate PES**, not a common variational SCRF.

The separately registered disabled scalar-first candidate does not reuse the
original four-channel density head as its source.  It defines an anchored
field energy on the fixed-charge, gauge-reduced two-width radial chart and
generates a complete eight-channel effective source and all field HVP/mixed
coordinate derivatives from that single Torch scalar graph.  This is a new
model identity.  Structural conjugacy of its implementation is necessary but
does not admit Tier V without the remaining physical and release gates.

The disabled implementation composes such a scalar-first model with a
scalar-first fixed continuum through

\[
L_s(R,c,u)=E(R,u)-s\langle c,u\rangle_Q+sG(R,c),
\]

and reuses the charge-constrained reduced fixed-point solver for
`c=M_E(R,u)`, `u=grad_Q G(R,c)`. Its envelope derivative is exposed only as an
unadmitted diagnostic. Synthetic composition and finite-difference tests do
not replace the real-checkpoint sign, passivity, root-uniqueness,
combined-Hessian, coordinate, symmetry, or release gates.
