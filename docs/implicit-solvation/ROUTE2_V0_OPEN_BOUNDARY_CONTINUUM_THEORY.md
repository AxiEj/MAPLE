# Route-2 V0-AQ-C: open-boundary reciprocal reaction scalar

## Scope and status

This document defines the isolated-source electrostatic block implemented in
[`route2_v0_open_diffuse_continuum.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_open_diffuse_continuum.py)
and its matching open density-defined cavity composition in
[`route2_v0_open_iso_density_cavity.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_open_iso_density_cavity.py).
It exists because an AO molecular density on a finite box generally has a
visible quadrature/domain electron-count residual.  A periodic Poisson solve
must reject that source rather than introduce a neutralising background; it
therefore cannot be the direct continuum endpoint for a future AO stationary
source.

The open block has a zero-potential boundary at the *exterior faces* of a
finite cell-centred box.  It is symmetric, energy-conjugate, and supports a
non-neutral source.  The matching cavity has a zero-extended rather than a
periodically wrapped convolution.  Together they are still only a
**structural electrostatic control**: they supply no permanent electronic
functional, physical solvent density kernel or overlap threshold, cavitation,
Pauli/dispersion term, standard-state term, force/PES certificate, runtime
result, or experimental solvation value.

In particular, removing the periodic neutrality restriction does **not**
repair an AO electron-count error.  The source must report

\[
\Delta N_h=\Delta V\sum_g n_D(\mathbf r_g)-\operatorname{Tr}[DS]
\]

and converge it by enlarging/refining the declared box.  There is no density
renormalisation, compensating charge, radius adjustment, or response scale.

The nuclear source needs the same boundary discipline.  The open
[`route2_v0_open_bspline.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_open_bspline.py)
map preserves the exact nuclear charge and grid-field adjoint while its cubic
support lies inside the box, and fails rather than wrapping, clipping, or
renormalising a nucleus whose support reaches the boundary.

## 1. Cell-centred open finite-volume operator

Let cells \(i\) lie on a rectangular Cartesian grid with volume
\(\Delta V=h_xh_yh_z\).  The computational boundary is one half spacing beyond
each outer cell centre, and

\[
\phi\vert_{\partial\Omega_h}=0.
\]

For a supplied smooth occupancy \(m_i\in[0,1]\), define

\[
\epsilon_i=1+(\epsilon_b-1)m_i,
\qquad
\epsilon_{ij}=\tfrac12(\epsilon_i+\epsilon_j)
\]

on an interior face \(i\!\leftrightarrow\!j\).  For an exterior face attached
to \(i\), use \(\epsilon_i\) and the half-cell distance \(h_a/2\).  The
finite-volume operator is consequently

\[
(A_\epsilon\phi)_i=
\sum_{a}\left[
\sum_{j=i\pm\hat a\,\mathrm{inside}}
\frac{\epsilon_{ij}}{h_a^2}(\phi_i-\phi_j)
+\sum_{b\in\partial_a i}
\frac{2\epsilon_i}{h_a^2}\phi_i
\right].
\]

For every nonzero real cell vector \(v\),

\[
v^\mathsf TA_\epsilon v=
\sum_{\langle i,j\rangle}
\frac{\epsilon_{ij}}{h_{ij}^2}(v_i-v_j)^2
+\sum_{i,b\in\partial i}\frac{2\epsilon_i}{h_b^2}v_i^2>0.
\]

Thus \(A_\epsilon\) is symmetric positive definite, has no constant-potential
null mode, and needs no periodic neutrality condition.  The two fields are
defined from the same source density \(\rho\):

\[
A_\epsilon\phi_\epsilon=4\pi\rho,
\qquad
A_1\phi_1=4\pi\rho.
\]

## 2. One reaction scalar, reciprocity, and passivity

The only reported electrostatic quantity is the vacuum-subtracted scalar

\[
\boxed{
G_{\rm reac,h}[\rho,m]
=\frac{\Delta V}{2}\rho^\mathsf T
(\phi_\epsilon-\phi_1).
}
\]

It follows directly that

\[
\frac{\delta G_{\rm reac,h}}{\delta\rho}
=\phi_\epsilon-\phi_1\equiv\phi_{\rm reac},
\]

and the reaction map is self-adjoint in the \(\Delta V\)-weighted pairing:

\[
\Delta V\,\rho_a^\mathsf T P\rho_b
=\Delta V\,\rho_b^\mathsf T P\rho_a,
\qquad
P=4\pi(A_\epsilon^{-1}-A_1^{-1}).
\]

Because \(\epsilon_i\ge1\), \(A_\epsilon\succeq A_1\), so

\[
P\preceq0,
\qquad G_{\rm reac,h}\le0.
\]

This is a passivity statement for the declared finite box, not a proof that a
particular dielectric/cavity is a complete solvent model.

## 3. Exact occupancy envelope derivative

At the stationary fields, no \(d\phi/dm\) solve belongs in the derivative.  If

\[
g_{ij,a}=\frac{\phi_{\epsilon,j}-\phi_{\epsilon,i}}{h_a}
\]

on an interior face, and \(g_{i,b}=2\phi_{\epsilon,i}/h_a\) on an exterior
face, then the exact derivative of the same finite-volume scalar is

\[
\boxed{
\frac{\partial G_{\rm reac,h}}{\partial m_i}
=-\frac{(\epsilon_b-1)\Delta V}{16\pi}
\sum_{F\ni i}g_F^2.
}
\]

Every interior or exterior face incident on cell \(i\) appears once.  This is
why the code must retain the exterior-face terms: omitting them yields a
potential that is not the derivative of the reported open-box energy.

For a density-defined cavity \(m[n]\), the electronic potential is the
composed derivative

\[
u_n=-\phi_{\rm reac}
+\left(\frac{\delta m}{\delta n}\right)^\!*\frac{\partial G_{\rm reac}}{\partial m},
\]

and the AO Euler/Fock contribution is the exact bridge pullback

\[
F_{\rm reac}=\mathcal P^*u_n.
\]

No grid-field interpolation or separately assembled half coupling is allowed.

## 4. Matching open iso-density-product cavity

The periodic cavity control cannot be combined with the open reaction scalar:
a circular FFT would reintroduce a periodic image of an isolated AO density
through the opposite box face.  The open map therefore uses a finite odd-shape
relative kernel \(k_\Delta\), centered at \(\Delta=0\), and declares the
zero-extension convention explicitly:

\[
(Kn)_i=\Delta V\sum_{j\in\Omega_h}k_{i-j}n_j,
\qquad
k_\Delta=0\quad\text{outside the stored relative support}.
\]

No index is taken modulo the simulation-box shape.  A positive,
centrosymmetric \(k\) makes \(K\) self-adjoint in the same uniform grid
pairing,

\[
\Delta V\sum_i (Kn)_i v_i
=\Delta V\sum_i n_i(K^\dagger v)_i,
\qquad K^\dagger=K.
\]

For a positive independently bound overlap threshold \(q_c\), the smooth
occupancy is

\[
q_i=(Kn)_i,
\qquad
m_i[n]=\tfrac12\operatorname{erfc}\!\left(\log\frac{q_i}{q_c}\right),
\]

with

\[
h'(q_i)=-\frac{\exp[-\log^2(q_i/q_c)]}{\sqrt\pi\,q_i}
\quad(q_i>0).
\]

The mathematical form is an iso-density-product cavity construction, not an
import of its empirical numerical choices.  In particular, the universal
continuum work of [Gunceler and Arias](https://arxiv.org/abs/1403.6465)
illustrates the nonlocal density-overlap idea, but its fitted threshold and
surface-tension choices are expressly **not** Route-2 V0 inputs.

Let \(g_i=\partial G_{\rm reac,h}/\partial m_i\) be the coordinate
derivative in Section 3.  Applying the chain rule in the declared grid
pairing gives

\[
\begin{aligned}
\delta G_{\rm reac,h}
&=\sum_i g_i h'(q_i)(K\delta n)_i\\
&=\Delta V\sum_j\delta n_j
\left[K^\dagger\!\left(\frac{h'(q)g}{\Delta V}\right)\right]_j,
\\
\boxed{\quad
u_n=-\phi_{\rm reac}
+K^\dagger\!\left(\frac{h'(q)g}{\Delta V}\right).
\quad}
\end{aligned}
\]

Thus both the direct charge term and the cavity term are derivatives of the
same reported reaction scalar.  The code does not permit a post-solve cavity
force, a fitted atomic radius, or a periodic shortcut.  An open boundary is
not an infinite-domain proof: buffer and grid-refinement convergence must
expose the residual truncation error before this block can enter a physical
calculation.

## 5. Evidence and remaining gates

[`test_route2_v0_open_diffuse_continuum.py`](../../tests/solvation/test_route2_v0_open_diffuse_continuum.py)
proves only the fixed-occupancy discrete claims: a deliberately non-neutral
source is admitted without a background, uniform-dielectric scaling,
reciprocity, passivity, density finite-difference conjugacy, and the full
occupancy finite difference including boundary faces.  It also combines the
open reaction potential with the exact AO/grid pullback and verifies the
central AO-density directional derivative of the reaction scalar.

[`test_route2_v0_open_iso_density_cavity.py`](../../tests/solvation/test_route2_v0_open_iso_density_cavity.py)
adds an independent direct-loop reference for the zero-extended convolution,
an opposite-face no-wrap check, the weighted convolution adjoint identity,
the cavity VJP finite difference, the composed non-neutral reaction finite
difference, and the exact AO density-dual pullback.  These tests use an
arbitrary structural kernel and threshold and do not use a solvation label or
select a numerical parameter from any error.

The physical motivation is consistent with the variational solute--continuum
derivations in [Chai and Luber (2024)](https://arxiv.org/abs/2407.20404), which
explicitly derive nonlocal-interface contributions to the electronic potential
and analytic forces.  JDFT provides the broader electron--liquid common-scalar
template, including a classical liquid functional when one is available; see
the [JDFTx `fluid` documentation](https://jdftx.org/CommandFluid.html).  Those
references do not provide a frozen Route-2 solvent asset or license importing
their fitted interface parameters.

Before this block can contribute to V0 physics or accuracy, the route still
needs all of the following:

1. a source-bound stationary molecular electronic functional and an
   AO/grid/domain convergence certificate, plus the matching non-wrapping
   nuclear source;
2. a physical solvent density kernel and overlap threshold fixed from
   independent evidence, then open-domain grid/buffer convergence for the
   matching smooth cavity already implemented here;
3. a pre-minimisation nonpolar/Pauli/dispersion and standard-state scalar from
   independent solvent physics;
4. KKT residual, stability, force, grid/buffer, rigid-motion, and closed-loop
   work gates; and then
5. the immutable all-record 10+ functional-group/11-solvent/blind experiment
   protocol, with every absolute error below the declared threshold.
