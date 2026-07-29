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

## 6. First frozen curvature binding: QEq-hardness monopole tangent

The first admissible no-training binding is deliberately narrow:
`route2-v0-rappe-goddard-hardness-same-basis-monopole-v1`.  It is a
**falsifier**, not a claim that QEq is an adequate molecular response model.

For elements with published Rappé--Goddard rows only

\[
\{\mathrm{H,Li,C,N,O,F,Na,Si,P,S,Cl,K,Br,Rb,I,Cs}\},
\]

it sets the monopole tangent curvature to

\[
H^{(q)}_{ij}=
\delta_{ij}\frac{\eta_i^{\mathrm{QEq}}}{E_h}
+(1-\delta_{ij})
\frac{\operatorname{erf}\!\left(R_{ij}/2\sigma_{\mathrm{MACE}}\right)}
{R_{ij}},
\qquad \sigma_{\mathrm{MACE}}=1.5\ \text{\AA},
\]

where distances are in bohr in the second term.  The hardness diagonals are
from the pinned local QEq table; the off-diagonal Coulomb metric deliberately
uses the **same MACE Gaussian density basis**, not the element-dependent QEq
screening radii.  This preserves the source/receiver basis rule for PCM while
using independently sourced atomic hardness as a local response scale.

The full electronic state remains

\[
c=c_0+\delta q, \qquad \delta\mu=0,
\]

with all raw \(l=1\) coefficients imposed as exact homogeneous KKT
constraints.  A large artificial dipole stiffness is forbidden.  QEq's
electronegativity linear term is intentionally not copied: the construction is
a reference-shifted tangent about frozen MACE \(c_0\), so its gas-phase
minimum is declared at \(\delta q=0\).  It is therefore not the full QEq
charge model and must never be marketed as one.

The original QEq paper takes atomic ionization potentials, electron affinities
and radii as input.  The implementation pins the local table bytes and refuses
the Open Babel rows that its own documentation labels unpublished UFF-derived
values.  It also inherits the documented fixed-neutral-hydrogen approximation
of the Gaussian QEq table.  These limits are why the binding is an independent
structural/physical control only; it does not authorize a chemistry score,
nonpolar term, force, or PES.

### 6.1 Frozen physical verdict

The structural acetone KKT canary passed, but that result tested only
variational consistency and stability.  It did not establish that the
published-hardness curvature represents the molecule's physical response.

A separate protocol was therefore frozen before execution:
`benchmarks/route2-v0-qeq-acetone-qm-field-prereg-v1.json`.  It compares the
analytic gas-phase V0-Q polarizability with an independent
ωB97M-V/def2-TZVPD central finite-field calculation at the identical archived
acetone geometry.  No experimental solvation value, response scale, fitted
hardness, field selection, MACE update, or continuum contribution enters the
comparison.

The QM calculation passed every registered numerical check.  The fixed
QEq-monopole response then failed every physical rejection gate:

\[
\frac{\|\alpha_{\mathrm{QEq}}-\alpha_{\mathrm{QM}}\|_F}
     {\|\alpha_{\mathrm{QM}}\|_F}
=1.4295>0.2,
\]

\[
\frac{\operatorname{tr}\alpha_{\mathrm{QEq}}}
     {\operatorname{tr}\alpha_{\mathrm{QM}}}
=1.9768\notin[0.8,1.2],
\]

and the maximum relative principal-value error is
\(2.2244>0.3\).  The frozen decision is
`reject-fixed-qeq-monopole-curvature`.

This is a negative physical result, not a failure of the common-energy KKT
formulation.  The V0-Q kernel remains structurally valid, but this curvature
must not be used for chemistry, solvation scoring, forces, or a public
profile.  Rescaling it or modifying its atomic hardnesses after observing the
QM result is explicitly forbidden.  The immutable result is
`benchmarks/route2-v0-qeq-acetone-qm-field-v1.json`.

## 7. Literature boundary

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
5. A. K. Rappé and W. A. Goddard III, *Charge equilibration for molecular
   dynamics simulations*, *J. Phys. Chem.* **95**, 3358--3363 (1991),
   [DOI:10.1021/j100161a070](https://doi.org/10.1021/j100161a070).  It is the
   independent source of the published atomic QEq input family, not a
   solvation-data fit.
6. J. Chen and T. J. Martínez, *QTPIE: Charge transfer with polarization
   current equalization*, *J. Chem. Phys.* **131**, 044114 (2009),
   [DOI:10.1063/1.3183167](https://doi.org/10.1063/1.3183167).  The Open Babel
   Gaussian QEq implementation documents its relation to this screened
   integral representation; V0-Q uses it only as a provenance control and
   retains the MACE-GTO metric for the actual pair term.
