# Route-2 V0: no-fit accuracy research decision record (2026-07-30)

## Purpose and non-negotiable boundary

This is a literature-screening record, not a benchmark result.  Route-2 V0
may improve only by changing the **pre-minimisation common scalar** or by
strengthening source-provenanced liquid physics.  It must not use target
solvation energies, a fitted correction, fine tuning, post-training, MAP/UQ
calibration, per-solute route selection, an error-selected cavity radius, or a
post-hoc free-energy ledger.

A future candidate is accepted only if it preserves one scalar

\[
\Omega[\nu;\mathbf R],\qquad
\frac{\delta\Omega}{\delta\nu}=0,\qquad
\mathbf F=-\partial_{\mathbf R}\Omega\big|_{\nu^*},
\]

with explicit source, pressure, surface-tension, operator, and derivative
contracts.  A paper's reported MAE/MAD is never evidence that Route-2 V0 meets
its own all-record maximum-error gates.

## Current evidence boundary

No V0 calculation has passed the immutable historical FreeSolv10 gate.  That
same ten-record panel contains the historical ethyl-acetate outlier
`mobley_6973347` at 7.041442082076966 kcal/mol in the retired GTO/QEq--GBn2
calculation; any V0 candidate must recompute **all ten** and put every record
strictly below 1.5 kcal/mol.  The locked ten are also explicitly one each from
ten distinct chemical-function or scaffold classes: alkane, aromatic
hydrocarbon, alcohol, ether, ketone, ester, nitrile, aromatic amine,
haloalkane, and sulfoxide.  Because alkane and aromatic hydrocarbon are
nonfunctional scaffolds, they are not allowed to satisfy the user's separate
ten-**actual**-functional-group requirement.  The frozen FreeSolv12 superset
therefore retains all historical ten (including ethyl acetate) and adds an
amide plus a carboxylic acid: it has ten source-labelled actual functional
groups and two separately reported nonpolar controls.  Both validators reject
a weakened diversity gate; there is no valid “accuracy test” with fewer than
ten actual functional groups or without the historical outlier.  This is
additional to the future 11-solvent, confirmation, and disjoint blind panels;
it cannot be replaced by a lower-MAE subset.

## Candidate screen

| Candidate | Mathematical status | V0 decision | Reason |
| --- | --- | --- | --- |
| HNC-plus-weighted-density molecular bridge | Variational if inserted into \(\Omega\) before stationarity; \(A_s\) follows from the HNC vacuum pressure, while \(B_s\) is first fixed by the finite-density same-scalar coexistence continuation and only a remaining liquid length scale may be checked against pure-liquid \(\gamma_s\). | **Active V0 path** | It targets the HNC cavity/liquid--vapour defect without a solvation-label regression.  The source-bound certificate must bind solvent sources, SI identities, the inner coexistence branch, outer planar root/residual/grid evidence, and discrete \(D,K\) operators. |
| Lorentz/Yukawa nonlocal dielectric spectrum | A positive quadratic orientational-polarization functional gives \(\epsilon_s(k)=\epsilon_\infty+(\epsilon_0-\epsilon_\infty)/(1+\lambda_s^2k^2)\); its reaction field is the derivative of one passive scalar. | **Active custom-solvent electrostatic control only** | It supplies a mathematically constrained finite-\(k\) response from independently sourced \(\epsilon_0,\epsilon_\infty,\lambda_s\), with no target fit.  It has no cavity, dispersion, molecular \(C_{ab}(k)\), or standard-state term, so it cannot be scored as a total solvation method. |
| LCW-style two-reference cDFT | A common cDFT scalar with slowly varying reference density and direct-correlation/surface-tension inputs. | **Deferred research path** | Bui--Cox provides a genuinely variational length-scale construction, but it needs a complete, source-provenanced multi-field liquid functional.  It must retain every term and pass scalar/force checks before it can replace the current bridge; importing only a coarse-graining kernel would be an unjustified hybrid. |
| Full orientational mDFT / angular correlation functional | Variational in the full molecular configuration density when the angular direct-correlation functional is frozen. | **Deferred research path** | Route-2 already preserves a full \(SO(3)\) quadrature convention.  A new angular functional needs a real all-atom solvent source and its own common-scalar, grid/orientation, and force proof; a site-HNC table cannot be relabelled as that functional. |
| Native PSE-\(n\) / 3D-RISM | PSE-\(n\) has a closure-matched, path-independent **site-3D-RISM on-shell** chemical-potential expression. | **Excluded from the current MACE molecular V0 bridge** | The PSE3 `Cvv` is only a bulk site correlation, while the MACE source is one whole-molecule cluster potential.  Neither supplies the site \(u_\alpha,h_\alpha,c_\alpha\) state required by native PSE3.  Feeding `Cvv` into molecular HNC or adding the PSE3 correction afterward would break common-scalar stationarity.  Direct PSE3 literature results also give no basis for the hard all-record \(<1.5\) target without the prohibited empirical corrections; see [`ROUTE2_V0_PSE_N_ADMISSION_AUDIT_20260730.md`](ROUTE2_V0_PSE_N_ADMISSION_AUDIT_20260730.md). |
| 3D-RISM PC, PC+, UC, PMV correction | Often useful empirically, but applied after a base 3D-RISM energy or selected against errors. | **Excluded from V0 core** | It fails the route's no-post-hoc-ledger/no-error-fit boundary.  Pressure must instead be repaired inside the scalar before minimisation. |
| Ng bridge and other single-parameter hydration corrections | May use a variational liquid-theory starting point. | **Excluded unless independently re-derived** | The cited 3D-RISM-NgB work adjusts a constant to hydration free energies, which is target fitting under the V0 contract. |
| MILC / learned linear corrections | Cross-validated experimental FreeSolv correction. | **Excluded** | It is explicitly label-driven calibration, even when descriptors look physical. |
| Frozen MACE field fixed point plus a new response filter | Does not create an integrable electronic response. | **Excluded** | A numerically converged non-variational fixed point cannot be made a common energy by symmetric post-processing. |
| V1 learned electronic functional head | Could be variational only after new QM training and a separate architecture/validation protocol. | **Deferred outside V0** | The user constraint freezes training and fine tuning for V0. |

## Why the active bridge is not an accuracy shortcut

For the bridge

\[
F_B=\Delta v\sum_g\left[
A_s(\bar\rho_g-\rho_b)^3+
B_s\bar\rho_g^2(\bar\rho_g-\rho_b)^4\right],
\quad \bar\rho=KD\nu,
\]

the pure-liquid source fixes

\[
A_s=\frac{P_{\rm HNC}-P_s}{\rho_b^3},
\]

This only fixes the empty-density pressure identity; it does **not** establish
a finite-density stationary gas phase or liquid--gas coexistence.  Before a
planar-interface result is admissible, the exact same scalar must show a
zero-external-potential liquid \(x=1\) and gas \(0<x_g<1\) with full
configuration-gradient stationarity, positive curvatures, and

\[
|\Omega[\nu_{x_g};0]/V-\Omega[\nu_{1};0]/V|
\le\tau_{\rm coex}.
\]

At finite gas density, a positivity-only \(d\gamma/dB_s\) formula omits
bulk-phase terms; changing \(B_s\) can also move the gas phase or break
coexistence.  Thus neither monotonicity nor unique-root claims are accepted
without a predeclared coexistence-preserving path, dividing-surface convention,
and stationary branch proof.  The implementation requires the actual
phase/coexistence record before a future physical-scope certificate and rejects
a source, grid, operator, closure, pressure, surface-tension conversion, or
no-label-policy mismatch.  The current v1 certificate is control-only and
fails closed for physical admission.  There is no physical certificate, a
complete liquid backend, a chemistry score, or a force/PES claim.  The current
synthetic test source remains synthetic by design.

The new inner continuation makes the allowed barrier construction precise.
For the quartic coefficient functional

\[
S[\nu]=\Delta v\sum_g\bar\rho_g^2(\bar\rho_g-\rho_b)^4,
\qquad
C(B_s)=\frac{\Omega_{B_s}[\nu_g(B_s)]-\Omega_{B_s}[\nu_b]}{V},
\]

stationarity of both homogeneous branches gives

\[
\frac{dC}{dB_s}=\frac{S[\nu_g]-S[\nu_b]}{V}>0
\]

on the declared continuous finite-density gas branch.  The liquid term is
exactly zero and the finite gas term is positive, so a negative-to-positive
coexistence-gap bracket determines \(B_s\) without reading \(\gamma_s\) or
any solvation label.  This result is deliberately about the **coexistence
gap**, not \(d\gamma/dB_s\).  If a Gaussian width remains unknown, the future
outer root must solve \(\gamma(B_s^*(\sigma_s),\sigma_s)=\gamma_s\) only
after this inner gate, retaining every coexistence and planar branch point.

The inner result now feeds an exact **control** planar restriction rather than
a new 1D surrogate: \(\nu(x,y,z,\Omega)\) is constrained only to be invariant
in \(x,y\), retaining the complete Euler \(SO(3)\) quadrature at each \(z\).
The reduced gradient is the exact transverse-average derivative of the same
scalar.  A fixed equimolar molecule count can stabilize a periodic slab
numerically, but it is accepted only if the associated multiplier and full
unconstrained residual both return to zero.  The synthetic local control has
zero surface excess, which is the required negative control.  This makes the
planar algebra auditable, not physical: a source-bound all-atom correlation,
outer length-scale branch, grid/orientation convergence, and physical
certificate are still absent.

## Next evidence needed before any accuracy assertion

1. Produce one physical, closure-aligned HNC source with a reproducible
   all-atom liquid model, bulk direct correlation, a saturation/coexistence
   state convention, independent surface tension, an exact homogeneous
   liquid/gas continuation record that determines \(B_s\), and a stationary
   planar-interface certificate for any remaining liquid length scale.  The
   current executable one-width order is fixed: build \(K_\sigma\), solve
   \(C(B_s,\sigma)=0\), freeze \(B_s^*(\sigma)\), then solve the exact
   planar branch; only a separate outer evidence record may compare the
   resulting \(\Gamma(\sigma)\) to the independent pure-liquid target.
   The IAPWS ordinary-water $\gamma(298\,\mathrm{K})$ source is now frozen
   separately, but it cannot be relabelled as a cSPC/E state or substitute for
   the missing model-matched state/correlation evidence.
2. Verify the same scalar's derivative, Hessian reciprocity, stability,
   Cartesian/orientation/grid convergence, envelope forces, and energy
   conservation.
3. Build source-complete physical assets for the pre-registered 11-solvent
   panel plus a strict custom-solvent admission route.  Dielectric constants
   alone remain insufficient; the required inputs are a molecular model,
   density, pressure, surface tension, correlation or liquid functional,
   source-bound \(D,K\), and source-complete solute--solvent interaction.
   For a custom Lorentz nonlocal electrostatic diagnostic, \(\epsilon_0\),
   \(\epsilon_\infty\), and a finite-\(k\) polarization-correlation length
   must each be independently source-bound; this remains below the
   total-free-energy boundary.
4. Freeze the method before reading target values, then run both the historical
   FreeSolv10 gate and the twelve-record, ten-actual-functional-group
   FreeSolv12 superset, followed by 11-solvent development, disjoint
   confirmation, and an independent final blind dataset.  Every record, not
   just the MAE, must be below 1.5 kcal/mol; the stricter final objective is
   below 1 kcal/mol.

## External sources screened

1. C. Gageat, L. Belloni, D. Borgis, and M. Levesque, *Bridge functional for
   the molecular density functional theory with consistent pressure and
   surface tension and its importance for solvation in water*,
   [arXiv:1709.10139](https://arxiv.org/abs/1709.10139).  This supports a
   pure-liquid pressure/surface-tension bridge placed inside a molecular
   functional rather than a solvation-error correction.
2. D. Borgis, S. Luukkonen, L. Belloni, and G. Jeanmairet, *Simple
   Parameter-Free Bridge Functionals for Molecular Density Functional Theory*,
   [DOI:10.1021/acs.jpcb.0c04496](https://doi.org/10.1021/acs.jpcb.0c04496).
   Its water-model construction is a precedent for source-fixed liquid
   thermodynamic anchors, not a Route-2 accuracy result.
3. A. T. Bui and S. J. Cox, *A classical density functional theory for
   solvation across length scales*,
   [arXiv:2402.02873](https://arxiv.org/abs/2402.02873).  It derives a
   self-consistent cDFT based on direct correlations and liquid--vapour surface
   tension; it motivates the deferred multi-field path rather than a partial
   transplant.
4. W. J. Baldwin *et al.*, *Design Space of Self-Consistent Electrostatic
   Machine Learning Interatomic Potentials*,
   [arXiv:2603.14700](https://arxiv.org/abs/2603.14700).  It frames
   electrostatic MLIPs through energy-functional versus fixed-point design
   choices and reinforces the common-functional boundary.
5. *A Cavity Corrected 3D-RISM Functional for Accurate Solvation Free
   Energies*,
   [DOI:10.1021/ct4009359](https://doi.org/10.1021/ct4009359).  Its adjusted
   hydration free-energy constant disqualifies it from the V0 no-fit route.
6. *High-Precision Solvation Free Energy Calculation via Multi-Input Linear
   Correction in 3D-RISM Theory*,
   [DOI:10.1021/acs.jctc.5c02013](https://doi.org/10.1021/acs.jctc.5c02013).
   The reported cross-validation on FreeSolv makes it unsuitable as a V0
   physics-only improvement.
7. V. P. Sergiievskyi *et al.*, *Fast Computation of Solvation Free Energies
   with Molecular Density Functional Theory: Thermodynamic-Ensemble Partial
   Molar Volume Corrections*,
   [DOI:10.1021/jz500428s](https://doi.org/10.1021/jz500428s).  It explains
   the pressure/PMV issue, but V0 keeps that repair inside the scalar rather
   than adopting a post-hoc correction.
8. D. Xie, J.-L. Liu, and B. Eisenberg, *A Nonlocal Poisson-Fermi Model for
   Ionic Solvent*, *Phys. Rev. E* **94**, 012114 (2016),
   [DOI:10.1103/PhysRevE.94.012114](https://doi.org/10.1103/PhysRevE.94.012114).
   It provides the Lorentz/Yukawa finite-wavevector dielectric construction;
   V0 uses it only for a source-bound fixed-density electrostatic control,
   never as a cavity or fitted total-solvation correction.
