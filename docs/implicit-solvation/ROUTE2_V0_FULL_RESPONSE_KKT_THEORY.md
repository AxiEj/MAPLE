# Route-2 V0-RK: full-response common-scalar KKT gate

## Status and strict scope

\`route2_v0_full_response_kkt.py\` is the first V0-RK structural kernel that
uses the **full** frozen atomic independent-particle coefficient space rather
than reducing induction to a molecular three-component dipole. It combines:

1. the frozen positive response covariance \(C_{\mathbf R}\) and its exact
   support curvature \(C_{\mathbf R}^{+}\);
2. the exact GTO transition-density surface map \(B_{\mathbf R}\) and its
   transpose in the **same coefficient/dual space**; and
3. a declared reciprocal, passive continuum response \(Q_{\mathbf R}\).

It is not a permanent-density model, a physical continuum admission, a force
or PES result, a nonpolar free-energy model, or a solvation-accuracy result.
The unit tests use a synthetic linear continuum to reject algebraic mistakes;
they do not run PCM or read experimental solvation labels. The
machine-readable boundary is
[\`route2-v0-full-response-kkt-prereg-v1.json\`](benchmarks/route2-v0-full-response-kkt-prereg-v1.json).

## 1. Frozen coefficient/dual pairing

The atomic-HF table supplies neutral transition densities
\(\{\tau_m\}_{m=1}^M\) and the V0-RK completion supplies
\(C_{\mathbf R}\succeq0\). The induced electron-number density is

\[
\delta n(\mathbf r)=\sum_{m=1}^{M}x_m\tau_m(\mathbf r),
\qquad N_{\mathbf R}x=0,
\]

where rows of \(N_{\mathbf R}\) span exactly
\(\ker C_{\mathbf R}\). The latter constraint is not a penalty or an
arbitrary hardness: it excludes coefficients outside the declared response
support. Every atomic transition is intrinsically charge neutral, so there
is deliberately no redundant total-charge KKT row.

At fixed continuum surface points \(\mathbf s_k\), the source matrix includes
the physical electron-charge sign:

\[
[B_{\mathbf R}]_{km}=-\int
\frac{\tau_m(\mathbf r)}{|\mathbf r-\mathbf s_k|}\,d\mathbf r,
\qquad v_{\mathrm{ind}}=B_{\mathbf R}x.
\]

The reaction dual has no separately written pullback:

\[
q^\mathsf T B_{\mathbf R}x=x^\mathsf T B_{\mathbf R}^{\mathsf T}q.
\]

The code obtains both maps from the same dense AO-integral matrix. Thus a
surface map and a reaction dual cannot silently disagree by construction.

## 2. One scalar after continuum elimination

Let \(v_0\) be an optional permanent surface potential. It is intentionally
an explicit input, not a claim that a permanent electronic density has already
been constructed. Let \(f\) be an external coefficient dual and
\(q=Q_{\mathbf R}(v_0+B_{\mathbf R}x)\), where \(Q_{\mathbf R}\) is the
energy-conjugate continuum response. The only scalar evaluated by the kernel
is

\[
\boxed{
G_{\mathbf R}(x;v_0,f)=
\frac12x^\mathsf TC_{\mathbf R}^{+}x+
\frac12(v_0+B_{\mathbf R}x)^\mathsf T
Q_{\mathbf R}(v_0+B_{\mathbf R}x)+
x^\mathsf Tf,
\quad N_{\mathbf R}x=0.}
\]

Define

\[
H_{\mathbf R}=C_{\mathbf R}^{+}+B_{\mathbf R}^{\mathsf T}
Q_{\mathbf R}B_{\mathbf R},
\qquad
g_{\mathbf R}=B_{\mathbf R}^{\mathsf T}Q_{\mathbf R}v_0+f.
\]

The stationary state is the symmetric KKT solve

\[
\begin{bmatrix}
H_{\mathbf R}&N_{\mathbf R}^{\mathsf T}\\
N_{\mathbf R}&0
\end{bmatrix}
\begin{bmatrix}x\\\eta\end{bmatrix}
=-\begin{bmatrix}g_{\mathbf R}\\0\end{bmatrix}.
\]

The code requires \(H_{\mathbf R}\succ0\) on
\(\operatorname{Ran}C_{\mathbf R}\), rather than relying on an SCF-mixing
spectral radius. It checks that the continuum is reciprocal and nonpositive
on that same support before the KKT factorization. A positive/anti-screening
continuum mode, material antisymmetry, a missing support constraint, or an
unstable joint curvature is a rejection, not a reason to clip an eigenvalue.

## 3. Reciprocity, passivity, and the ledger

With \(U\) an orthonormal basis of
\(\operatorname{Ran}C_{\mathbf R}\), the external response is

\[
\frac{\partial x^*}{\partial f}
=-U(U^\mathsf TH_{\mathbf R}U)^{-1}U^\mathsf T.
\]

It is symmetric and negative semidefinite in the declared pairing. Therefore
Maxwell reciprocity and passivity arise from the scalar itself, not from a
post-hoc symmetrization of MACE's nonvariational fixed-point Jacobian.

At a stationary state, the reported terms are exactly

\[
G_{\mathrm{el}}=\tfrac12x^\mathsf TC^+x,
\qquad
G_{\mathrm{cont}}=\tfrac12(v_0+Bx)^\mathsf Tq,
\qquad
G_{\mathrm{ext}}=x^\mathsf Tf,
\qquad
G=G_{\mathrm{el}}+G_{\mathrm{cont}}+G_{\mathrm{ext}}.
\]

The tests independently lock:

* \(q^\mathsf TBx=x^\mathsf TB^\mathsf Tq\);
* support-constraint and KKT residuals;
* continuum linearity, reciprocity, and support passivity;
* the complete energy ledger;
* the external envelope identity \(dG^*/df=x^*\); and
* the permanent-potential envelope identity
  \(dG^*(a v_0,f)/da=v_0^\mathsf Tq^*\).

## 4. Permanent-reference identifiability is an independent gate

The optional \(v_0\) input makes the common scalar algebra explicit; it does
not identify, validate, or create the physical permanent source. A response
covariance and its source map determine the induced curvature but not the
affine molecular electronic reference. Two distinct neutral references can
therefore have identical reciprocal/passive external response while producing
different permanent surface potential, continuum charge, and on-shell energy.

[ROUTE2_V0_PERMANENT_REFERENCE_IDENTIFIABILITY.md](ROUTE2_V0_PERMANENT_REFERENCE_IDENTIFIABILITY.md)
states and tests this reference-shift nonidentifiability. It rejects the raw
MACE fixed point, a promolecular source, or the response covariance alone as a
complete permanent reference. This structural kernel remains a synthetic
control until that independent gate passes.

## 5. What this permits next

Only after the permanent-reference gate passes may a source-consistent next
implementation bind the stationary permanent source to \(v_0\) and use one
smooth-coordinate, reciprocal continuum whose \(B_{\mathbf R}\) and
\(B_{\mathbf R}^{\mathsf T}\) remain conjugate as nuclei move. It does not
permit taking a MACE density head as \(v_0\), borrowing a cavity radius from a
solvation error, or adding a fitted nonpolar correction.

Lange and Herbert's switching/Gaussian PCM construction shows why the
continuum backend must be smooth under cavity motion rather than merely
reciprocal at one geometry: ordinary overlapping-sphere BEM discretizations
can produce discontinuous PESs, whereas switching and Gaussian surface charge
regularization address singularities and gradient oscillations.
[Their SWIG PCM paper](https://doi.org/10.1021/jz900282c) is therefore a
backend-design reference, not an authorization to import a fitted cavity.

The broader common-functional requirement is also consistent with the recent
self-consistent electrostatic MLIP design-space analysis, which distinguishes
energy-functional formulations from fixed-point constructions and finds that
more expressive self-consistent models remain necessary.
[Baldwin *et al.*](https://arxiv.org/abs/2603.14700)

Before any experimental score, the next gates remain: a source-bound smooth
continuum/cavity, a stationary permanent electronic source, coordinate
KKT/envelope-force proof, the twelve-record QM physics panel, runtime
comparison, and only then the immutable all-record multi-solvent and blind
experimental protocols.
