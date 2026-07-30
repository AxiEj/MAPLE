# Route-2 V0-ADT: induced-only reduced KKT gate

## Status and strict scope

`route2_v0_atomic_displacement_kkt.py` implements an executable **reduced**
KKT gate for the frozen V0 atomic-displacement tangent (V0-ADT).  It proves a
small but important statement: at fixed geometry, the physical V0-ADT
induced-source map, a declared positive molecular polarizability, and one
reciprocal continuum response can be made derivatives of **one induced-only
scalar** without a guessed radial width, response mixing, Jacobian repair, or
solvation-label fit.

This is not yet the Route-2V electronic functional proposed for production.
It has no permanent solute density, no gas-phase total energy, no nonpolar
functional, no coordinate derivative, and no total-solvation prediction.  The
unit tests use a synthetic reciprocal continuum only to falsify algebraic
mistakes; they are not a PCM calculation or an accuracy result.  In
particular, passing this gate does not authorize a FreeSolv/MNSol run.

The machine-readable boundary is
[`route2-v0-atomic-displacement-reduced-kkt-prereg-v1.json`](benchmarks/route2-v0-atomic-displacement-reduced-kkt-prereg-v1.json).

## 1. Exact source and its declared dual

At a fixed nuclear geometry \(\mathbf R\), let \(p\in\mathbb R^3\) be the
induced molecular dipole in \(e\,a_0\).  The frozen atomic map supplies
\(p_a=W_a p\), with \(\sum_aW_a=I\), and the reproducible free-atom
translation tangent supplies the surface potential

\[
v=B_{\mathbf R}p,
\qquad
[B_{\mathbf R}]_{k i}
=V_{\rm ADT}(\mathbf s_k;e_i),
\]

where \(\mathbf s_k\) are the fixed continuum surface points and

\[
V_{\rm ADT}(\mathbf r;p)=
\sum_a
\frac{(W_ap)\cdot(\mathbf r-\mathbf R_a)}
{Z_a\lvert\mathbf r-\mathbf R_a\rvert^3}
N_{Z_a}(\lvert\mathbf r-\mathbf R_a\rvert).
\]

The columns are evaluated directly from the frozen radial table; no GTO fit,
Gaussian width, multipole patch, or response rescaling is introduced.  The
continuum contract returns integrated apparent surface charge \(q\) in the
pairing \(v^\mathsf Tq\).  Therefore the only allowed back-map is exactly

\[
 B_{\mathbf R}^{\mathsf T}q,
 \qquad
 q^\mathsf TB_{\mathbf R}p
 =p^\mathsf TB_{\mathbf R}^{\mathsf T}q.
\]

The code builds this transpose from the same dense matrix, rather than asking
the source and continuum implementations for separately written pullbacks.

## 2. Continuum elimination and the common scalar

Let \(Q_{\mathbf R}\) denote the energy-conjugate linear continuum response,
\(q=Q_{\mathbf R}v\), and let the frozen MACE-MDP molecular polarizability
be \(\alpha_\theta(\mathbf R)\succ0\) in \(a_0^3\).  A dipole dual \(f\)
uses the sign convention in which the external work is \(p^\mathsf Tf\).
After eliminating the continuum surface variables, define

\[
\boxed{
 g_{\mathbf R}(p;f)=
 \frac12p^\mathsf T\alpha_\theta^{-1}p
 +\frac12(B_{\mathbf R}p)^\mathsf TQ_{\mathbf R}(B_{\mathbf R}p)
 +p^\mathsf Tf.}
\]

The reduced stationary equation and Hessian are

\[
K_{\mathbf R}p+f=0,
\qquad
K_{\mathbf R}=
\alpha_\theta^{-1}+B_{\mathbf R}^\mathsf TQ_{\mathbf R}B_{\mathbf R}.
\]

Thus \(p^*=-K_{\mathbf R}^{-1}f\), \(v^*=B_{\mathbf R}p^*\), and
\(q^*=Q_{\mathbf R}v^*\).  The external-dual response is

\[
\frac{\partial p^*}{\partial f}=-K_{\mathbf R}^{-1}.
\]

When \(K_{\mathbf R}\succ0\), this response is symmetric and nonpositive
in the declared energy pairing.  Hence reciprocity and passivity follow from
the scalar; they are not imposed by symmetrizing a learned fixed-point
Jacobian.

The implementation reports the only permissible ledger:

\[
g_{\rm el}=\tfrac12p^\mathsf T\alpha_\theta^{-1}p,
\quad
g_{\rm cont}=\tfrac12v^\mathsf Tq,
\quad
g_{\rm ext}=p^\mathsf Tf,
\quad
g=g_{\rm el}+g_{\rm cont}+g_{\rm ext}.
\]

## 3. Executable rejection gates

Before returning a state, the kernel rejects all of the following rather than
repairing them:

1. a non-symmetric or non-positive MACE-MDP polarizability;
2. a continuum whose declared geometry, atomic numbers, or surface points do
   not match V0-ADT;
3. a material antisymmetry of \(B^\mathsf TQB\) in the V0-ADT source
   subspace;
4. loss of positive definiteness of either electronic curvature or
   \(K_{\mathbf R}\);
5. a surface charge that is not dual to the reported continuum curvature;
6. a failed stationary residual, half-coupling identity, KKT inverse identity,
   or passive external response.

There is intentionally no DIIS/mixing setting, eigenvalue clipping,
polarizability rescaling, cavity adjustment, or fit-to-error escape route.
The tests additionally verify the envelope identity
\(d g^*/d f_i=p_i^*\) by central finite difference.

## 4. Why this is not the full Route-2V functional

The reduced scalar contains only the three induced-dipole degrees of freedom.
It cannot represent a permanent density \(c_0\), charge transfer, higher
response channels, or the explicit coordinate dependence of a total
electronic functional.  It also does not establish that a physical PCM
operator is smooth under cavity motion.  Therefore it is narrower than the
target joint functional

\[
\mathcal L_{\mathbf R}(c,\sigma,\lambda)=
F_\theta(\mathbf R,c)
+\tfrac12\sigma^\mathsf TA_{\mathbf R}\sigma
+\sigma^\mathsf TB_{\mathbf R}c
+\lambda(u^\mathsf Tc-Q).
\]

The reduced construction is a source-space lower gate, not a substitute for a
single full density/source dual.  In particular, it must not be relabeled as
an energy for unmodified MACE-MDP or as a total implicit-solvation free energy.

## 5. Consequences for the main line

The next admissible stages are:

1. bind the actual frozen MACE-MDP \(\alpha_\theta\), the physical V0-ADT
   source, and a fixed reciprocal continuum at preregistered geometries;
2. run source/duality/reciprocity/passivity and QM response gates over the
   frozen twelve-record, ten-functional-group **physics** panel without
   reading experimental solvation values;
3. construct a full permanent-density/source dual (or reject V0-ADT and move
   to the separately source-bound V0-RK kernel);
4. establish a smooth-coordinate joint functional and envelope-force gates;
5. only then freeze every choice and evaluate the immutable broad experimental
   panels, including every historical FreeSolv10 record and the required
   multi-solvent/blind partitions.

No paper can prove the requested per-record \(<1.5\) or ultimate \(<1\)
kcal/mol bound for this new method; only the frozen, identity-bound
experimental protocol can do that.

## 6. Literature placement

* [Petrosyan *et al.*, joint density-functional theory](https://arxiv.org/abs/cond-mat/0606817)
  gives the governing principle: solute electrons and solvent must arise from
  one variational free energy rather than a fixed-point response plus a
  separately reported energy.
* [MACE-Field](https://arxiv.org/abs/2508.17870) demonstrates the architectural
  alternative of differentiating one learned electric enthalpy; it is not an
  admissible V0 dependency because the available work trains field-aware models
  for inorganic solids.
* [Lange and Herbert's SWIG PCM](https://doi.org/10.1021/jz900282c) motivates a
  smooth reciprocal continuum backend: switching weights and Gaussian surface
  charges address discontinuities and gradient oscillations of ordinary
  sphere-grid BEMs.
* [Universal iso-density JDFT PCM](https://arxiv.org/abs/1403.6465) is the most
  relevant mathematical template for future custom solvents, but its published
  construction still needs a full solute electron density and solvent bulk/
  molecular inputs; it cannot be dropped onto a three-component V0-ADT state.
* [SaLSA](https://arxiv.org/abs/1410.2273) derives parameter-free dielectric
  response from liquid susceptibility, but its reported total solvation model
  includes a fitted dispersion contribution.  It is an independent continuum
  research comparator, not a no-fit total-V0 shortcut.
