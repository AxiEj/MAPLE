# Route-2 V0 exploration register

## Purpose

This is a factual register of completed Route-2 V0 explorations.  It preserves
negative results, source controls, and mathematical controls so that they are
not rediscovered, silently reused, or mistaken for chemical-accuracy evidence.
**No code or artifact listed here is deleted by this register.**

The active main line is an implicit-continuum model:

\[
\mathcal L(\mathbf R,c,\sigma,\lambda)
=F_{\rm gas}(\mathbf R,c)
+\tfrac12\sigma^\mathsf TA_{\mathbf R}\sigma
+\sigma^\mathsf TB_{\mathbf R}c
+\lambda(u^\mathsf Tc-Q).
\]

Thus the main line needs a physical gas electronic functional and an
energy-conjugate PCM operator.  It does **not** need explicit solvent
molecules, molecular-liquid MD, or GROMACS.  Molecular RISM/MDFT work stays
preserved as an archived comparison/research lane, not as a prerequisite or
fallback for the implicit PCM route.

A status of `pass` below means only the artifact's own stated numerical or
structural claim passed.  It never means that Route 2 has passed the
all-record experimental accuracy requirement.

## A. Electronic-response and common-energy explorations

| Exploration | Authoritative result | What was learned | Main-line disposition |
| --- | --- | --- | --- |
| Frozen checkpoint scalar-response V0 | [`route2-v0-scalar-response-water-v1.json`](benchmarks/route2-v0-scalar-response-water-v1.json): **fail** | On one fixed water geometry, zero-field anchoring and scalar finite differences passed, but nonzero field charge, reciprocity, and passivity failed. The neutral antisymmetric/symmetric ratio was \(5.6184\times10^{-5}>10^{-5}\); the largest response eigenvalue was \(+1.97098\times10^{-2}\,\mathrm{eV}>10^{-6}\). | Legacy `c=M_\theta(Pc)` remains a nonvariational diagnostic only. Do not lower SCF tolerance, symmetrize its Jacobian, or reuse it as \(\nabla_f W\). |
| MACE-energy-only V0-FE / Fenchel admission | [`route2-v0-energy-only-fenchel-water-v1.json`](benchmarks/route2-v0-energy-only-fenchel-water-v1.json): **fail** | The exact autograd Hessian of the frozen local-field **energy scalar** is reciprocal (antisymmetric ratio \(2.63\times10^{-10}\)) but not concave in the neutral coefficient-dual field: its largest eigenvalue is \(+3.441894\times10^{-2}\,\mathrm{eV}\), versus the preregistered passive bound \(10^{-8}\,\mathrm{eV}\). This test does not use MACE's density head, PCM, or experimental labels. | Reject the current checkpoint as a MACE-only Legendre--Fenchel V0 electronic functional. Do not clip positive modes, symmetrize/temper a response, or replace it with an error-selected envelope. The result is a structural negative, not an accuracy test. |
| V0-FD frozen MACE source + reciprocal continuum | [`route2_v0_frozen_density.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_frozen_density.py) and [`ROUTE2_V0_FD_THEORY.md`](ROUTE2_V0_FD_THEORY.md) | \(\frac12v^\mathsf Tq_{\rm energy}(v)\) is an auditable fixed-density electrostatic scalar when source and energy-conjugate continuum are paired. | Retain as a structural electrostatic baseline. It has no induced solute response, total nonpolar term, force/PES certificate, or chemistry score. |
| Same-GTO V0-Q KKT kernel | [`route2_v0_variational_quadratic.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_variational_quadratic.py) and [`route2-v0-qeq-monopole-acetone-v1.json`](benchmarks/route2-v0-qeq-monopole-acetone-v1.json): **pass, structural only** | A symmetric \(H\), reciprocal \(P\), and exact charge constraints give a stationary KKT state with reciprocal passive response. This proves the algebraic architecture, not molecular electronic physics. | Retain as the common-energy kernel. It accepts no default curvature and cannot enter a chemistry leaderboard by itself. |
| Published Rappé--Goddard QEq monopole curvature | [`route2-v0-qeq-acetone-qm-field-v1.json`](benchmarks/route2-v0-qeq-acetone-qm-field-v1.json): execution **pass**, scientific verdict **reject** | Independent acetone QM finite-field data rejected the frozen curvature: relative Frobenius polarizability mismatch \(1.4295>0.20\), trace ratio \(1.9768\notin[0.80,1.20]\), largest principal-value error \(2.2244>0.30\). | Preserve as a falsifier. Never rescale QEq hardnesses, add dipole penalties, fit a response scale, or compose it with PCM for chemistry scoring. |
| Auxiliary QM--PCM stationary-difference control | [`route2-v0-aq-water-pcm-control-v1.json`](benchmarks/route2-v0-aq-water-pcm-control-v1.json): **rejected before PCM state** | The source-bound water control aborted at the gas translation-gradient gate. No energy/force result was serialized and no parameter was changed afterward. | Preserve the failed control and its exact gas-subtraction ledger. It is not a replacement for the implicit PCM electronic functional and not an accuracy result. |
| Auxiliary SCF--ddCOSMO numerical discriminators | [`route2-v0-aq-electronic-preflight-20260730.json`](benchmarks/route2-v0-aq-electronic-preflight-20260730.json): **PBE0 rejected; RHF numerical preflight passes only** | PBE0/def2-TZVPD with its atom-centred grid missed the \(10^{-8}\,E_h/a_0\) translation gate at \(1.10\times10^{-6}\); integral RHF/def2-TZVPD plus the same source-defined ddCOSMO control reached \(8.50\times10^{-15}\). | Preserve this as an implementation discriminator, not a method choice or chemical claim. It proves neither a smooth production cavity nor a no-fit total solvent free energy. |

The executable `route2_v0_variational_admission.py` records these distinctions:
structural-only, gas-response rejection, and an electronic-component admission
that is still insufficient for total-solvation scoring.  The associated test
uses the frozen QEq artifact, so that curvature cannot be accidentally
promoted later.

## B. Implicit-continuum and custom-solvent explorations

| Exploration | Authoritative result | What was learned | Main-line disposition |
| --- | --- | --- | --- |
| Energy-conjugate PCM / GTO Galerkin operator | [`gto_galerkin.py`](../../maple/function/calculator/extra_correction/implicit/gto_galerkin.py) and V0-Q preregistration | A one-basis source/dual map preserves \(\sigma^\mathsf TBc=c^\mathsf TB^\mathsf T\sigma\); eliminating the stationary continuum gives \(P=-B^\mathsf TA^{-1}B\). | Reuse as the electrostatic pairing requirement. A smooth production cavity/coordinate derivative remains open. |
| V0-AQ-C diffuse fixed-occupancy reaction block | [`route2_v0_diffuse_continuum.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_diffuse_continuum.py) and [`test_route2_v0_diffuse_continuum.py`](../../tests/solvation/test_route2_v0_diffuse_continuum.py): **pass, structural only** | One symmetric finite-volume Poisson operator now supplies the reaction scalar, its density derivative, and its occupancy envelope derivative. The tests lock homogeneous-dielectric scaling, reciprocity, passivity, and both finite-difference conjugacy identities. | Retain as the fixed-occupancy V0-AQ-C electrostatic building block. It has no stationary cavity, auxiliary electronic functional, nonpolar scalar, total free energy, force/PES certificate, or chemistry score. |
| V0-AQ-C iso-density-product cavity composition | [`route2_v0_iso_density_cavity.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_iso_density_cavity.py) and [`test_route2_v0_iso_density_cavity.py`](../../tests/solvation/test_route2_v0_iso_density_cavity.py): **pass, structural only** | A nonlocal solvent-electron-density convolution determines the smooth occupancy, and the exact adjoint carries the occupancy derivative into the auxiliary electron-density derivative. The combined reaction scalar passes its full finite-difference derivative test; the cavity chain term is nonzero and cannot be dropped. | Retain as the no-atomic-radius cavity closure candidate. It has no source-bound solvent kernel/threshold, auxiliary QM state, nonpolar/dispersion scalar, standard-state term, total free energy, or accuracy score. |
| V0-AQ-C weighted-density cavitation scalar | [`route2_v0_weighted_density_cavity.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_weighted_density_cavity.py) and [`test_route2_v0_weighted_density_cavity.py`](../../tests/solvation/test_route2_v0_weighted_density_cavity.py): **pass, structural only** | A fourth-order cavity scalar is fixed by pure-liquid pressure, number density, vapour pressure, planar surface tension, and solvent vdW radius. Its exact shell-adjoint derivative, bulk/empty limits, finite-difference identity, and periodic covariance pass. | Retain as the implicit \(\Phi_{\rm cav}\) candidate, not the archived molecular-liquid WDA branch. It has no source-bound shell/properties, no proven solvent-centre/electrostatic-cavity relation, no Pauli/dispersion, total free energy, force/PES certificate, or accuracy score. |
| V0-AQ-C ideal standard-state conversion | [`route2_v0_standard_state.py`](../../maple/function/calculator/extra_correction/implicit/route2_v0_standard_state.py) and [`test_route2_v0_standard_state.py`](../../tests/solvation/test_route2_v0_standard_state.py): **pass, structural only** | The exact \(RT\log(c^\circ RT/p^\circ)\) conversion has its declared sign, units, temperature derivative, and zero nuclear-coordinate derivative.  It contains no solvent name, solvation label, fitted coefficient, or radius. | Retain as the universal \(\Phi_{\rm std}\) primitive. It cannot repair an incomplete excess-free-energy functional, and a total asset must bind its auxiliary reference to the same convention. |
| Lorentz/Yukawa nonlocal dielectric | [`route2-v0-lorentz-nonlocal-dielectric-prereg-v1.json`](benchmarks/route2-v0-lorentz-nonlocal-dielectric-prereg-v1.json): **preregistered control only** | A passive finite-\(k\) dielectric spectrum is a mathematically valid custom-solvent electrostatic control if its limits and correlation length are independently sourced. | Retain for custom-solvent electrostatics research only. It is explicitly not a cavity, dispersion, standard-state, total-solvation, or accuracy result. |
| Free-atom promolecular density source | [`route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json`](benchmarks/route2-v0-promolecular-atomic-hf-def2-tzvpd-v1.json): **pass, source only** | The artifact supplies a positive, frozen free-atom density source with documented provenance. | Preserve as a source control. It is not MACE density, a solute--solvent interaction, or a nonpolar free-energy functional. |
| Solvent-asset inventory | [`route2-v0-solvent-asset-inventory-v1.json`](benchmarks/route2-v0-solvent-asset-inventory-v1.json): **zero admitted physical default assets** | The inventory correctly fails closed rather than replacing missing physical terms with proxy solvents or error-selected parameters. | Keep as evidence of what remains missing; do not turn inventory entries into default solvent models. |
| xTB GFN2 `--cosmo 78.3553` source screen | [`route2-v0-xtb-cosmo-source-screen-v1.json`](benchmarks/route2-v0-xtb-cosmo-source-screen-v1.json): **excluded before candidate registration** | With a numeric dielectric, local xTB 6.7.1 enabled GBSA and emitted nonzero \(G_{\rm sasa}\), an internal parameter file, surface tension, density, and solvent mass. Isolating its reported \(G_{\rm elec}\) would splice an implementation component out of the stationary xTB energy. | Preserve as an external empirical-comparator screen; exclude xTB COSMO/ALPB/CPCM-X from the no-fit core. |

For a future user-defined implicit solvent, dielectric data alone may select an
electrostatic response family but cannot by itself determine a total
solvation free energy.  A total model still needs a declared smooth cavity,
nonpolar/dispersion functional, and standard-state convention, all frozen
before experimental solvation errors are examined.

## C. Archived molecular-liquid / RISM / MDFT research

These experiments are retained because their failure modes and scalar rules
are useful.  They are **not deleted**, but they are not required for the
implicit PCM main line.

| Exploration | Authoritative result | Preserved conclusion |
| --- | --- | --- |
| cSPC/E 1D-RISM bulk parser and Coulomb-tail split | [`route2-v0-rism1d-cspce-bulk-control-v1.json`](benchmarks/route2-v0-rism1d-cspce-bulk-control-v1.json): **bulk-only parser control** | Confirms parsing and analytic-tail bookkeeping only; not a MACE-coupled 3D-RISM result, physical solvent asset, or solvation value. |
| Molecular-HNC closure identity | [`route2-v0-molecular-hnc-closure-identity-v1.json`](benchmarks/route2-v0-molecular-hnc-closure-identity-v1.json): **source-complete water control, not physical admitted** | The scalar correctly refuses a bulk correlation generated by a different closure. This prevents cross-closure energy mixing. |
| Dichloromethane KH/DRISM numerical probe | [`route2-v0-dcm-kh-source-frozen-feasibility-audit-v1.json`](benchmarks/route2-v0-dcm-kh-source-frozen-feasibility-audit-v1.json): **rejected** | The frozen run missed its residual tolerance and produced no XVV/Cvv asset. It does not invalidate the source molecule, but it cannot be used as finite-\(k\) liquid data. |
| Native PSE3 shortcut | [`route2-v0-pse3-native-admission-audit-v1.json`](benchmarks/route2-v0-pse3-native-admission-audit-v1.json): **reject current shortcut** | A bulk site correlation plus a whole-molecule MACE source is not the native site-3D-RISM state required by PSE3. Post-hoc PSE3 insertion would not preserve the common scalar. |
| Weighted-density bridge and cubic WDA controls | [`route2-v0-weighted-density-bridge-prereg-v1.json`](benchmarks/route2-v0-weighted-density-bridge-prereg-v1.json) and [`route2-v0-molecular-cubic-wda-prereg-v1.json`](benchmarks/route2-v0-molecular-cubic-wda-prereg-v1.json): **preregistered controls** | They test mathematical scalar/derivative and coexistence logic only. No physical liquid asset or solvation accuracy claim exists. |
| Ordinary-water surface-tension source | [`route2-v0-ordinary-water-surface-tension-iapws-r1-76-2014-v1.json`](benchmarks/route2-v0-ordinary-water-surface-tension-iapws-r1-76-2014-v1.json): **source only** | An IAPWS target is not automatically a cSPC/E/HNC model state, bridge-width choice, or solvation correction. |

## D. Accuracy evidence and gates

No current V0 candidate has a valid total-solvation prediction set.  Therefore
there is no current MAE, maximum error, or claim of progress toward chemical
accuracy.

The immutable future gates are preserved rather than weakened:

1. [`route2-v0-historical-freesolv10-regression-v1.json`](benchmarks/route2-v0-historical-freesolv10-regression-v1.json) retains the historical ethyl-acetate record `mobley_6973347`, where the retired GTO/QEq/GBn2 calculation reached **7.041442082076966 kcal/mol** error.
2. [`route2-v0-freesolv12-functional-groups-v1.json`](benchmarks/route2-v0-freesolv12-functional-groups-v1.json) retains those ten and adds amide and carboxylic-acid records: ten actual functional groups plus methane and benzene controls. Every prediction must be **strictly below 1.5 kcal/mol**; equality fails.
3. [`route2-v0-fd-multisolvent-prereg-v1.json`](benchmarks/route2-v0-fd-multisolvent-prereg-v1.json) preserves the subsequent 11-solvent, confirmation, and independent blind-data gates. None has been executed for a physically admissible total model.

## E. Reuse rules

- Preserve every artifact above, including failed and control-only records.
- Do not delete or rerun a source-bound exploratory control merely to obtain a
  more favorable number.
- Do not use an archived molecular-liquid result to bypass the missing
  implicit electronic functional.
- Do not let a structural pass, source pass, or parser pass become an
  experimental accuracy claim.
- The next useful main-line evidence is a frozen physical gas electronic
  functional that passes the executable QM response gate and can be coupled to
  the same energy-conjugate, smooth implicit continuum scalar.
