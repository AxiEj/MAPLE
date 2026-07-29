# Route-2 V0-AQ: auxiliary quantum--liquid variational-difference contract

## Status and scope

V0-AQ is the no-training route for restoring a **stationary electronic
response** without claiming that the frozen MACE-POLAR field update is an
energy gradient.  It leaves the gas MACE model untouched and introduces a
separate, declared auxiliary electronic--liquid calculation only through a
gas-to-liquid free-energy difference.

It is not a public Route-2 profile, a replacement for V0-FD, a total-solvation
result, a force/PES implementation, or an accuracy result.  In particular,
the first V0-AQ-E control below is a fixed-geometry electronic-continuum
control: it contains neither an empirical SMD CDS term nor a complete liquid
non-electrostatic free energy.  It cannot be compared with experimental
solvation free energies.

The machine-readable commitment is
[`route2-v0-auxiliary-qm-liquid-prereg-v1.json`](benchmarks/route2-v0-auxiliary-qm-liquid-prereg-v1.json).

## 1. One scalar ledger without changing the MACE gas branch

At a nuclear geometry \(\mathbf R\), retain the unmodified MACE gas energy

\[
E_{\mathrm{MACE,gas}}(\mathbf R).
\]

Let \(n\) denote an **auxiliary** electronic density.  It is not a MACE
density, is not projected from MACE coefficients, and never drives a
field-conditioned MACE inference.  The corresponding auxiliary gas reference
is

\[
A_{\mathrm{aux}}^{\mathrm{gas}}(\mathbf R)
=\min_{n\in\mathcal N_e} A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R].
\]

For a physical liquid state with solvent degrees of freedom \(N\), define one
joint scalar

\[
A_{\mathrm{aux}}^{\mathrm{sol}}(\mathbf R;s)
=\operatorname*{stat}_{n,N,\lambda}
\left[
 A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R]
+\Phi_s[N]
+U_s[n,N;\mathbf R]
+\lambda\,C[n,N]
\right].
\]

The only admissible composite ledger is therefore

\[
\boxed{
G_{\mathrm{V0\text{-}AQ}}(\mathbf R;s)
=E_{\mathrm{MACE,gas}}(\mathbf R)
+\left[A_{\mathrm{aux}}^{\mathrm{sol}}(\mathbf R;s)
-A_{\mathrm{aux}}^{\mathrm{gas}}(\mathbf R)\right]
+\Delta G_s^\circ .
}
\]

The gas subtraction is mandatory.  Adding a solvated auxiliary electronic
energy directly to \(E_{\mathrm{MACE,gas}}\) would count a second gas solute
energy; adding a PCM half-coupling or a CDS term outside this same auxiliary
difference would create a second, nonconjugate ledger.  \(\Delta G_s^\circ\)
is permitted only when it is part of a declared, physical liquid
standard-state convention.  It is absent from the V0-AQ-E control.

This construction is deliberately an *auxiliary solvation difference*, not a
claim that MACE and the auxiliary quantum method describe the same electronic
state.  It preserves the MACE gas potential exactly while making the added
response the derivative of a scalar by construction.

## 2. The first bounded control: V0-AQ-E

V0-AQ-E uses a self-consistent auxiliary QM--PCM scalar only to test the
electronic/continuum stationarity boundary at fixed geometry.  It is **not a
total solvation free energy**.  For an
energy-conjugate PCM discretization with apparent surface charge \(\sigma\),
one representative Lagrangian is

\[
\mathcal L_{\mathrm{aux,PCM}}[n,\sigma]
=A_{\mathrm{aux}}^{\mathrm{gas}}[n;\mathbf R]
+\frac12\sigma^\mathsf T A_{\mathbf R}\sigma
+\sigma^\mathsf T B_{\mathbf R}n.
\]

Its stationary equations are

\[
\frac{\delta A_{\mathrm{aux}}^{\mathrm{gas}}}{\delta n}
+B_{\mathbf R}^\mathsf T\sigma=0,
\qquad
A_{\mathbf R}\sigma+B_{\mathbf R}n=0.
\]

The correct V0-AQ-E correction is the difference between the fully
self-consistent stationary total energies,

\[
\Delta A_{\mathrm{aux}}^{\mathrm{PCM}}
=A_{\mathrm{aux}}^{\mathrm{PCM\text{-}SCF}}
-A_{\mathrm{aux}}^{\mathrm{gas\text{-}SCF}},
\]

not an isolated implementation-specific ``PCM energy`` field and not a
frozen-density half-coupling estimate.  Thus the electron density and surface
charge are both stationary in the same scalar.  This directly avoids the
nonreciprocal learned fixed-point response diagnosed for legacy Route 2.

V0-AQ-E has the following hard limits:

* it is an electrostatic fixed-geometry control, not a total solvation free
  energy or a solvent-ranking calculation;
* it does not use SMD radii, SMD CDS, target solvation data, or any
  post-result radius/scale selection;
* a finite tessellation may still be unsuitable for a production PES.  An
  analytic gradient at an isolated geometry does not certify smooth
  rotation/translation, scan, optimization, MD, or NVE behavior;
* no auxiliary density may be relabelled as a MACE density or be sent into a
  MACE field-feature interface.

The first execution is deliberately a source-bound water control, frozen in
[`route2-v0-aq-water-pcm-prereg-v1.json`](benchmarks/route2-v0-aq-water-pcm-prereg-v1.json)
and run by
[`run_route2_v0_auxiliary_qm_pcm_water.py`](benchmarks/run_route2_v0_auxiliary_qm_pcm_water.py).
It records the gas and PCM-SCF **total** stationary energies and their
analytic nuclear gradients, then takes their difference.  It does not call
MACE, inspect a solvation label, or certify a force/PES; the separate V0-AQ
ledger owns the future addition to an unchanged MACE gas term.

## 3. Physical completion: V0-AQ-L

The full V0-AQ-L branch replaces the PCM control with a molecular liquid
functional \(\Phi_s[N]+U_s[n,N]\) that contains the solvent structure,
short-range interaction, and thermodynamic convention in the same scalar.
It needs frozen, independently sourced solvent assets before any target
solute is evaluated.  Required inputs include a molecular site model, bulk
state and equation of state, liquid correlation or classical-DFT functional,
short-range/dispersion physics, and a standard-state convention.  A solvent
name or scalar dielectric is insufficient.

There are two admissible upstream families, subject to their own independently
parameterized liquid assets:

1. **Joint/classical DFT.**  Joint density-functional theory couples an
   electronic functional and a classical liquid functional in one variational
   free energy.  JDFTx documents a `ClassicalDFT` fluid mode and named
   `fluid-solvent` assets, but those assets are inputs to freeze before
   scoring, not free parameters to tune after observing errors.
2. **QM--3D-RISM-SCF/MDFT.**  A molecular liquid density or RISM site-density
   functional can be jointly stationary with an auxiliary QM state.  The
   actual MACE source may not be substituted by GAFF/AM1-BCC topology charges;
   if a MACE-native interaction cannot be derived from one scalar, the
   auxiliary QM state remains the solute source for this branch.

Neither family is automatically ``parameter free``: solvent force-field,
correlation, and cavity/dispersion inputs must be recorded as independent
physical assets.  SaLSA is useful evidence that a nonlocal dielectric response
can be derived without empirical dielectric parameters, but its published
solvation model retains a fitted dispersion contribution and is therefore not
a strictly zero-fit total-free-energy endpoint here.

## 4. Conserved derivatives and acceptance gates

At stationary auxiliary gas and liquid states, the envelope theorem gives

\[
\frac{dG_{\mathrm{V0\text{-}AQ}}}{d\mathbf R}
=\frac{\partial E_{\mathrm{MACE,gas}}}{\partial\mathbf R}
+\left.
\frac{\partial\mathcal L_{\mathrm{aux}}^{\mathrm{sol}}}
{\partial\mathbf R}\right|_*
-\left.
\frac{\partial A_{\mathrm{aux}}^{\mathrm{gas}}}
{\partial\mathbf R}\right|_* .
\]

No response derivative from the legacy MACE field fixed point appears in this
expression.  Before any force claim, the candidate must prove that all terms
are derivatives of the same auxiliary scalar and pass the independent
smooth-cavity and central-difference gates.  V0-AQ-E is explicitly below that
threshold.

The prerequisite order is:

1. source-bind an auxiliary QM gas/solvent pair with the same functional,
   basis, geometry, charge/spin, numerical grid, and converged stationary
   states;
2. verify the exact gas subtraction and the absence of an extra half-coupling,
   CDS, fitted response scale, or MACE-field update;
3. for V0-AQ-L, register complete liquid assets for at least the frozen
   11-solvent set before inspecting target-solvation labels;
4. prove scalar/gradient/force gates, then freeze development, disjoint
   confirmation, and external-blind records; and
5. reject any failed structural or per-record accuracy gate without tuning,
   retraining, fine-tuning, response clipping, or per-record model selection.

## 5. Literature and upstream boundary

1. S. Petrosyan *et al.*, *Joint density-functional theory for electronic
   structure of solvated systems*,
   [arXiv:cond-mat/0606817](https://arxiv.org/abs/cond-mat/0606817).  It gives
   the joint electron--liquid variational principle used as the V0-AQ-L
   structural reference.
2. JDFTx, [`fluid`](https://jdftx.org/CommandFluid.html) and
   [`fluid-solvent`](https://jdftx.org/CommandFluidSolvent.html) documentation.
   The upstream interface distinguishes ClassicalDFT from continuum modes and
   lists named solvent assets plus explicit physical-property overrides; it
   does not authorize error-driven property selection.
3. N. Minezawa and S. Kato, *Efficient implementation of 3D-RISM-SCF*,
   *J. Chem. Phys.* **126**, 054511 (2007),
   [DOI:10.1063/1.2431809](https://doi.org/10.1063/1.2431809).  It reports
   direct electron-density-to-grid coupling and first derivatives of the
   free energy, motivating the V0-AQ-L scalar/force requirement.
4. R. Sundararaman *et al.*, *Spicing up continuum solvation models with
   SaLSA*, *J. Chem. Phys.* **142**, 054102 (2015),
   [arXiv:1410.2273](https://arxiv.org/abs/1410.2273).  Its dielectric response
   is parameter-free, while the reported full solvation model retains a
   dispersion fit; this is why it is not a no-fit total endpoint here.
