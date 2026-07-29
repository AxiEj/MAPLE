# Route-2 V0-Q: no-training variational quadratic-response kernel

## Status and scope

V0-Q is a **structural research kernel**, not a chemical model, public Route-2
profile, force implementation, or accuracy result.  It adds no post-training,
fine-tuning, experimental-solvation fit, MAP/UQ calibration, density
projection, response damping, or update to the frozen MACE-POLAR checkpoint.

Its machine-readable preregistration is
[`route2-v0-variational-quadratic-prereg-v1.json`](benchmarks/route2-v0-variational-quadratic-prereg-v1.json).
The kernel exists to make the required mathematical gates executable *before*
someone binds a physical gas-phase response model to the frozen MACE density.
It supplies no default curvature and therefore cannot yet produce a chemistry
score.

V0-Q is distinct from both earlier zero-training controls:

- frozen field-space scalar V0 was rejected because the checkpoint's learned
  field response violates charge, reciprocity, and passivity gates;
- V0-FD freezes the solute density and minimizes only the continuum state;
- V0-Q retains the same frozen zero-field density but admits a new induced
  density only through a separately declared scalar electronic functional.

The last item is not a relabeling of the learned fixed point.  The rejected
MACE field update is never called by V0-Q.

## 1. One joint scalar and one stationary state

At fixed nuclear geometry, write the frozen gas-phase density as

\[
c_0(\mathbf R)=c_{\mathrm{MACE}}(\mathbf R,0),
\qquad u^\mathsf Tc_0=Q.
\]

The induced coefficient vector is \(\delta c\), with exact conservation
constraint \(u^\mathsf T\delta c=0\).  A caller supplies a symmetric
gas-phase electronic curvature \(H_{\mathbf R}\) in the **same declared GTO
coefficient/dual pairing** as the continuum operator.  It defines

\[
F_{\mathrm{ind}}(\mathbf R,\delta c)
=\frac12\delta c^\mathsf TH_{\mathbf R}\delta c.
\]

For a reciprocal same-basis GTO Galerkin PCM operator \(P_{\mathbf R}\),
the V0-Q scalar is

\[
\boxed{
\mathcal G_{\mathbf R}(\delta c)
=
\frac12\delta c^\mathsf TH_{\mathbf R}\delta c
+\frac12(c_0+\delta c)^\mathsf TP_{\mathbf R}(c_0+\delta c)
}
\]

up to the unchanged MACE gas energy.  The corresponding KKT equations are

\[
\begin{bmatrix}
H_{\mathbf R}+P_{\mathbf R} & u\\
u^\mathsf T & 0
\end{bmatrix}
\begin{bmatrix}\delta c\\\lambda\end{bmatrix}
=
\begin{bmatrix}-P_{\mathbf R}c_0\\0\end{bmatrix}.
\]

The code exposes the two ledger terms separately,

\[
G_{\mathrm{ind}}=\frac12\delta c^\mathsf TH\delta c,
\qquad
G_{\mathrm{PCM}}=\frac12(c_0+\delta c)^\mathsf TP(c_0+\delta c),
\]

and reports only their sum as the V0-Q solute--continuum correction.  It does
not splice a MACE field-conditioned energy onto a separately solved density.

## 2. Structural gates are enforced, not optimized away

Let \(Q_0\) be an orthonormal basis for \(u^\mathsf T\delta c=0\).  Before
solving, V0-Q requires

\[
Q_0^\mathsf THQ_0\succ0,
\qquad
Q_0^\mathsf T(H+P)Q_0\succ0.
\]

The first gate makes the declared electronic functional convex in the allowed
charge-conserving directions.  The second gate is the actual local stability
condition after dielectric feedback.  A failed gate is a rejected physical
candidate, not an instruction to add mixing or clip a negative eigenvalue.

For a perturbing coefficient-dual field \(f\), the same scalar gives

\[
\frac{d(c_0+\delta c)}{df}
=-Q_0\left[Q_0^\mathsf T(H+P)Q_0\right]^{-1}Q_0^\mathsf T.
\]

It is symmetric and nonpositive in the declared energy pairing by
construction.  Thus Maxwell reciprocity and passivity are architectural
properties of this candidate, not loss terms applied to the rejected learned
response.

## 3. Density representation boundary

The existing MACE-POLAR \(l\leq1\) source has one Gaussian radial width of
`1.5 Å`.  V0-Q embeds it exactly in one matching radial GTO channel:

\[
c_0^{\mathrm{GTO}}[a,0,:]=c_0^{\mathrm{MACE}}[a,:].
\]

No fit or projection is allowed.  A multi-radial response basis would require
a declared density decomposition and a new preregistration, so the current
kernel rejects it rather than silently inventing additional density channels.
This same-basis rule lets the Galerkin compression preserve
\(\sigma^\mathsf TBc=c^\mathsf TB^\mathsf T\sigma\) and the PCM half-coupling
identity.

V0-Q's Gaussian source representation is deliberately different from V0-FD's
legacy point-multipole control.  Their numerical scores must not be compared
as though they differed only by an induced response switch.

## 4. What remains required before it becomes a physical candidate

The kernel's `H` is intentionally an explicit input.  A valid binding must
freeze, before any target solvation error is read:

1. its complete coefficient-basis and unit conversion convention;
2. an independently sourced gas-phase physical construction (for example,
   ab-initio response data or a documented hardness/polarizability plus
   vacuum-interaction functional);
3. its elements, reference states, and uncertainty/provenance;
4. the cavity/operator and all nonpolar/standard-state terms.

An atomic hardness alone, an isolated-atom polarizability alone, or an
unversioned diagonal matrix is **not** enough to claim molecular electronic
physics.  Any simple QEq/induced-dipole form must include its stated
gas-phase Coulomb and charge-transfer assumptions and pass the exact same
structural gates.

MNSol, FreeSolv, SMD parameters, target solvation energies, or observed
per-solvent errors may not select or rescale `H`.  Until a physical `H` is
frozen, V0-Q is intentionally unable to enter the chemistry leaderboard.

## 5. Explicit exclusions

V0-Q does not currently provide:

- a default electronic curvature, element table, or trained response head;
- a nonpolar free-energy functional;
- an arbitrary-dielectric total-solvation interface;
- a smooth cavity, coordinate derivative, force, PES, optimization, scan,
  MD, or NVE claim;
- a replacement for the legacy fixed-point diagnostic or V0-FD control.

After physical inputs are frozen, it must still pass the preregistered
structural gates, per-record multi-solvent development gate, disjoint
confirmation gate, and external blind gate described by the V0-FD protocol.

## 6. Literature boundary

1. I. Batatia *et al.*, *MACE-POLAR-1: A Polarisable Electrostatic Foundation
   Model for Molecular Chemistry* (2026),
   [arXiv:2602.19411](https://arxiv.org/abs/2602.19411).  Its current
   non-self-consistent field construction does not furnish the `H` used here.
2. W. J. Baldwin *et al.*, *Design Space of Self-Consistent Electrostatic
   Machine Learning Interatomic Potentials* (2026),
   [arXiv:2603.14700](https://arxiv.org/abs/2603.14700).  It distinguishes
   energy-functional and fixed-point electrostatic constructions.
3. S. Petrosyan, J.-F. Briere, D. Roundy, and T. A. Arias, *Joint
   density-functional theory for electronic structure of solvated systems*,
   [arXiv:cond-mat/0606817](https://arxiv.org/abs/cond-mat/0606817).  It is
   the variational reference: solute and solvent arise from a common
   stationary functional.
4. R. G. Parr and R. G. Pearson, *Absolute hardness: companion parameter to
   absolute electronegativity*, *J. Am. Chem. Soc.* **105**, 7512--7516
   (1983), [DOI:10.1021/ja00364a005](https://doi.org/10.1021/ja00364a005).
   It motivates possible future independently sourced hardness data, but does
   not by itself validate a molecular curvature matrix.
