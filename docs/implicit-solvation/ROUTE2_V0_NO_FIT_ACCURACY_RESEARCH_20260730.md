# Route-2 V0: archived no-fit research alternatives and current implicit-PCM decision record (2026-07-30)

## Current route decision and non-negotiable boundary

This is a literature-screening record, not a benchmark result.  The active
Route-2 V0 main line is an **implicit PCM/continuum** construction: a
stationary gas electronic functional plus an energy-conjugate smooth continuum
operator and a separately declared implicit nonpolar/standard-state scalar.
It does not require explicit solvent molecules, molecular-liquid simulation,
GROMACS, 3D-RISM, or MDFT.  Those older molecular-liquid sections below are
archived alternative research only and must not be invoked by the V0 main
execution path or used to justify an accuracy test.

Route-2 V0 may improve only by changing the **pre-minimisation common scalar**
within that implicit route.  It must not use target solvation energies, a
fitted correction, fine tuning, post-training, MAP/UQ calibration, per-solute
route selection, an error-selected cavity radius, or a post-hoc free-energy
ledger.  The executable electronic-admission conditions and current
no-training verdict are recorded in
[`ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md`](ROUTE2_V0_VARIATIONAL_QUADRATIC_THEORY.md).

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
| HNC-plus-weighted-density molecular bridge | Variational only as a molecular-liquid theory. | **Archived; excluded from implicit V0 main path** | It is not needed for a PCM model and may not become a substitute electronic functional, nonpolar term, or accuracy shortcut. |
| Lorentz/Yukawa nonlocal dielectric spectrum | A positive quadratic orientational-polarization functional gives \(\epsilon_s(k)=\epsilon_\infty+(\epsilon_0-\epsilon_\infty)/(1+\lambda_s^2k^2)\); its reaction field is the derivative of one passive scalar. | **Active custom-solvent electrostatic control only** | It supplies a mathematically constrained finite-\(k\) response from independently sourced \(\epsilon_0,\epsilon_\infty,\lambda_s\), with no target fit.  It has no cavity, dispersion, molecular \(C_{ab}(k)\), or standard-state term, so it cannot be scored as a total solvation method. |
| V0-AQ-C diffuse stationary auxiliary continuum | One auxiliary electronic functional, smooth solvent-occupancy field, reaction-field difference, and nonpolar/standard-state functional are stationary in a single scalar. | **Active implicit-continuum architecture; electrostatic, cavitation, and exact standard-state primitives implemented, physical total assets pending** | It preserves the MACE gas branch and requires neither GROMACS nor a molecular-liquid trajectory.  The fixed-occupancy reaction scalar, density-defined nonlocal-cavity chain rule, pure-liquid weighted-density cavitation scalar, and exact \(RT\log(c^\circ RT/p^\circ)\) standard-state term pass their structural/finite-difference checks.  It still admits no SMD-CDS, fitted cavity threshold, fitted dispersion scale, or post-hoc ledger; its source-bound solvent-centre/electrostatic-cavity relation, Pauli/dispersion, auxiliary electronic functional, and custom-solvent assets must exist before any accuracy run.  See [`ROUTE2_V0_AQ_THEORY.md`](ROUTE2_V0_AQ_THEORY.md). |
| SaLSA iso-density-product cavity and nonlocal dielectric response | A density-overlap cavity and nonlocal dielectric response can be obtained from a liquid-density functional; the published dielectric response has no empirical parameter. | **Use the cavity/derivative mathematics only; reject the published total model as V0** | SaLSA's published dispersion term has a fitted solvent-dependent scale, and its reported RMS is not an all-record maximum-error certificate.  A V0 use requires separately source-bound isolated-solvent kernels and a pre-label overlap condition; neither the fitted dispersion scale nor a paper threshold may become a silent default. |
| Solvent-aware SCCS | A nonlocal density convolution with full functional derivatives gives a smooth density-defined interface and analytical force terms. | **Derivative/convolution reference only** | The recent implementation demonstrates why a purely local density cavity can create internal solvent islands.  Its published interface thresholds and nonpolar model are not V0 physical assets and must not be copied as error-selected defaults. |
| LCW-style two-reference cDFT | A common cDFT scalar with slowly varying reference density and direct-correlation/surface-tension inputs. | **Deferred research path** | Bui--Cox provides a genuinely variational length-scale construction, but it needs a complete, source-provenanced multi-field liquid functional.  It must retain every term and pass scalar/force checks before it can replace the current bridge; importing only a coarse-graining kernel would be an unjustified hybrid. |
| Full orientational mDFT / angular correlation functional | Variational in the full molecular configuration density when the angular direct-correlation functional is frozen. | **Deferred research path** | Route-2 already preserves a full \(SO(3)\) quadrature convention.  A new angular functional needs a real all-atom solvent source and its own common-scalar, grid/orientation, and force proof; a site-HNC table cannot be relabelled as that functional. |
| Native PSE-\(n\) / 3D-RISM | PSE-\(n\) has a closure-matched, path-independent **site-3D-RISM on-shell** chemical-potential expression. | **Excluded from the current MACE molecular V0 bridge** | The PSE3 `Cvv` is only a bulk site correlation, while the MACE source is one whole-molecule cluster potential.  Neither supplies the site \(u_\alpha,h_\alpha,c_\alpha\) state required by native PSE3.  Feeding `Cvv` into molecular HNC or adding the PSE3 correction afterward would break common-scalar stationarity.  Direct PSE3 literature results also give no basis for the hard all-record \(<1.5\) target without the prohibited empirical corrections; see [`ROUTE2_V0_PSE_N_ADMISSION_AUDIT_20260730.md`](ROUTE2_V0_PSE_N_ADMISSION_AUDIT_20260730.md). |
| 3D-RISM PC, PC+, UC, PMV correction | Often useful empirically, but applied after a base 3D-RISM energy or selected against errors. | **Excluded from V0 core** | It fails the route's no-post-hoc-ledger/no-error-fit boundary.  Pressure must instead be repaired inside the scalar before minimisation. |
| Ng bridge and other single-parameter hydration corrections | May use a variational liquid-theory starting point. | **Excluded unless independently re-derived** | The cited 3D-RISM-NgB work adjusts a constant to hydration free energies, which is target fitting under the V0 contract. |
| MILC / learned linear corrections | Cross-validated experimental FreeSolv correction. | **Excluded** | It is explicitly label-driven calibration, even when descriptors look physical. |
| Frozen MACE field fixed point plus a new response filter | Does not create an integrable electronic response. | **Excluded** | A numerically converged non-variational fixed point cannot be made a common energy by symmetric post-processing. |
| Frozen MACE intrinsic-energy-only Legendre--Fenchel construction | The exact energy Hessian is reciprocal, but the frozen local-field scalar is not concave on the charge-conserving coefficient-dual space. | **Rejected for the current checkpoint before PCM or accuracy** | The preregistered MACE-only water falsifier found \(\lambda_{\max}=+0.03441894\,\mathrm{eV}\), far above the passive \(10^{-8}\,\mathrm{eV}\) bound. A local eigenvalue clip, response tempering, or error-selected concave envelope would be a new post-hoc model, not a recovery of the frozen energy functional. |
| Frozen gas GFN2-xTB plus external pair fields | GFN2-xTB has a self-consistent semiempirical energy, but this installed point-charge interface does not make its reported full dipole the energy-conjugate source. | **Rejected before PCM or accuracy** | The preregistered no-runtime-QM acetone screen is fast (13 cases: 0.242 s versus the archived QM finite-field reference: 855.736 s) but fails both required layers: its dipole polarizability has \(0.45046\) relative Frobenius mismatch and \(0.54691\) trace ratio against QM, and its energy/dipole diagonal identity misses by \(0.48611\). This is not repairable by rescaling, parameter adjustment, or an external correction. xTB COSMO/GBSA/ALPB/CPCM-X remains excluded. |
| Official MACE-MDP response model | An official pretrained organic dipole/polarizability predictor; its documentation declares it for those properties **only**, not energies or forces. The sealed acetone coefficient, atomic-partition, and coordinate-derivative screens remain valid: raw \(\alpha_\theta\) is symmetric/positive and matches QM global polarizability at relative Frobenius \(0.003868\), its identity-bound partition has \(\sum_a\alpha_a\alpha^{-1}=I\) to \(4.06\times10^{-17}\), and coefficient derivatives match fixed finite differences at \(2.28\times10^{-8}\) (dipole) and \(3.98\times10^{-8}\) (polarizability). Its first one-radial induced-density source has now failed the independent vacuum QM MEP test: \(0.4597926523>0.20\) Frobenius and \(0.4965624773>0.30\) maximum directional error. | **Admit frozen \(\alpha_\theta(\mathbf R)\), its atom moment partition, and explicit coefficient derivatives only; reject the one-radial source and do not use MACE-MDP as a drop-in electronic functional** | A molecular polarizability fixes only a three-dimensional projection of a full density susceptibility, not the radial/null-space response that determines the MEP. The failure cannot be repaired by width selection, extra multipoles, rescaling, or experimental labels. A successor requires a separately source-bound full response kernel and must pass QM MEP physics gates before PCM/KKT or accuracy. This preserves the no-training boundary but makes no accuracy promise. [Coefficient artifact](benchmarks/route2-v0-mace-mdp-acetone-response-v1.json); [partition artifact](benchmarks/route2-v0-mace-mdp-atomic-map-acetone-v1.json); [source rejection](benchmarks/route2-v0-mace-mdp-induced-source-acetone-v1.json); [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/polarizability.html) |
| V0-ADT free-atom translation-tangent source | The frozen v2 radial tables come from independently generated spherical free-atom HF densities, not a width search. The exact translated-electron potential is combined only with the frozen raw MACE-MDP \(\alpha_\theta\) and atom partition. Its preregistered acetone QM-MEP source run passes all 516 exterior points: Frobenius \(0.1484314509<0.20\), maximum direction \(0.1611208415<0.30\), and dipole response \(0.0038678622<0.20\). | **Admit only the exact real-space source to the next same-basis source/dual and KKT gate; do not use it as a drop-in MACE functional** | This is the first no-fit evidence that a source-provenanced radial response can repair the Gaussian near-field failure. It remains one molecule and one gas-phase physics canary: no continuum, molecular density coefficient, energy/force, broad response, or solvation accuracy claim follows. If a non-arbitrary source/dual representation or the frozen broad physics panel fails, preserve the result and return to V0-RK; never select radial details from error. [Source result](benchmarks/route2-v0-atomic-displacement-source-acetone-v1.json); [repeatability audit](benchmarks/route2-v0-atomic-displacement-hf-def2-tzvpd-v2-reproducibility-audit.json) |
| V0-RK response-kernel completion | [`route2_v0_response_kernel.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_response_kernel.py) implements the Schur-complement completion from an independently source-bound positive-semidefinite GTO density-response kernel \(C_0\), atom-dipole map \(A\), and frozen MACE partition \(W\). It replaces only \(S_0=AC_0A^\mathsf T\) by \(\Gamma=W\alpha_\theta W^\mathsf T\): \(C=C_0-C_0A^\mathsf TS_0^{-1}AC_0+C_0A^\mathsf TS_0^{-1}\Gamma S_0^{-1}AC_0\). It proves \(C\succeq0\), \(ACA^\mathsf T=\Gamma\), exact global \(\alpha_\theta\), charge neutrality, and the support scalar \(\tfrac12\delta c^\mathsf TC^+\delta c\). | **Structural no-fit kernel only; no physical \(C_0\) asset, PCM, or accuracy result** | This retains a physically sourced radial/null-space response while using the frozen MACE tensor only for its observed molecular moment response. It is a fixed low-rank covariance constraint, not a width fit or response correction. A singular \(S_0\) is rejected rather than pseudoinverted. The baseline \(C_0\) must be fixed from independent electronic-response data before seeing any solvation label, and the completed kernel must first pass preregistered QM MEP, density, reciprocity, passivity, force, and speed gates. See [`ROUTE2_V0_RESPONSE_KERNEL_THEORY.md`](ROUTE2_V0_RESPONSE_KERNEL_THEORY.md). |

| MACE-Field | It learns one field-dependent enthalpy and differentiates it consistently, the correct **architecture**. Its released evidence is for inorganic solids and field-aware model training/fine-tuning. | **Excluded for molecular V0** | It has neither an already admissible molecular-organic field-energy checkpoint nor a no-training path for the FreeSolv chemistry domain. [MACE-Field](https://arxiv.org/abs/2508.17870) |
| FieldSchNet / FIREANN-style field models | They can model field-dependent molecular energies only after dedicated field-response training; published FieldSchNet data cover named molecule families rather than a universal frozen organic checkpoint. | **Excluded for molecular V0** | Adopting any such model now would violate the no-post-training/fine-tuning boundary. [FieldSchNet](https://arxiv.org/abs/2010.14942) |
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

1. Build one source-complete **V0-AQ-C** implicit-continuum functional:
   a stationary auxiliary electronic state, reaction-field subtraction,
   smooth occupancy/cavity state, and one pre-minimisation nonpolar and
   standard-state scalar.  Its solvent inputs must be frozen from
   independently sourced physical data or a pure-solvent theory; it does not
   require GROMACS, 3D-RISM, MDFT, or a molecular-liquid trajectory.
2. Verify the same scalar's derivative, Hessian reciprocity, stability,
   Cartesian/orientation/grid convergence, envelope forces, and energy
   conservation.  A numerical SCF/continuum control is not enough.
3. Build source-complete physical assets for the pre-registered 11-solvent
   panel plus a strict custom-solvent admission route.  Dielectric constants
   alone remain insufficient; the required inputs include bulk state,
   electrostatic response, smooth-cavity/nonpolar functional, short-range and
   dispersion physics, and a standard-state convention.  For a custom Lorentz
   nonlocal electrostatic diagnostic, \(\epsilon_0\), \(\epsilon_\infty\),
   and a finite-\(k\) polarization-correlation length must each be
   independently source-bound; this remains below the total-free-energy
   boundary.
4. Freeze the method before reading target values, then run both the
   historical FreeSolv10 gate and the twelve-record, ten-actual-functional-
   group FreeSolv12 superset, followed by 11-solvent development, disjoint
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
9. R. Sundararaman, K. A. Schwarz, K. Letchworth-Weaver, and T. A. Arias,
   *Spicing up continuum solvation models with SaLSA: the spherically-averaged
   liquid susceptibility ansatz*,
   [arXiv:1410.2273](https://arxiv.org/abs/1410.2273).  It derives a
   parameter-free dielectric response and a nonlocal density-overlap cavity,
   but the published total neutral-molecule model includes one fitted dispersion
   parameter.  V0 preserves only the mathematically differentiable cavity
   pattern until independent source assets exist.
10. Z. Chai and S. Luber, *Functional Analytic Derivation and CP2K
    Implementation of the SCCS Model Based on the Solvent-Aware Interface*,
    [arXiv:2407.20404](https://arxiv.org/abs/2407.20404).  It provides an
    independent derivation of the nonlocal-interface potential and force
    contributions and documents the failure of purely local density cavities;
    it does not supply V0 solvent assets or authorize use of its thresholds.
