You are reviewing one narrow, zero-training theoretical-chemistry decision for MAPLE Route 2. This is a new focused question. Do not propose fitting to experimental solvation energies, global rescaling, residual ML, deleting difficult molecules, changing targets after inspection, or inventing a new continuum method when a standard physical diagnostic exists.

SYSTEM UNDER REVIEW

The frozen hybrid uses three distinct source branches coupled to the same linear ddPCM operator at each geometry:

1. p: permanent MACE-MDP atom-centred point monopoles/dipoles (4N coefficients).
2. r(u): MACE-POLAR induced Gaussian residual multipoles after removing its zero-field source and its three-dimensional uniform-field tangent.
3. a(u): a canonical atomic-density-translation (ADT) dipole distribution whose molecular sum reproduces the same MACE-MDP molecular polarizability in the uniform-field subspace.

The continuum field u is returned in the native eight-channel MACE-POLAR receiver space. The operational fixed point is

    u = K_R [p + r(u) + a(u)]

where K_R includes source-to-boundary maps, ddPCM solve, and boundary-to-native-field receiver maps. The operational electrostatic ledger is the stationary linear-continuum half work

    G = 1/2 <p+r+a, K_R[p+r+a]>.

The construction is deliberately not claimed to be a common MACE-continuum variational functional. No parameters were fit to solvation targets. Stock SMD CDS is added only as a separately reported M1 total.

The response split currently uses an arithmetic three-dimensional uniform-field chart:

    r(u) = M_POLAR(u) - M_POLAR(0) - J_POLAR(0) U G u
    a(u) = P_ADT[-alpha_MDP G u]

where U embeds a uniform Cartesian field in the native eight-channel field space and G is its arithmetic left inverse. Therefore a continuum field with both uniform and nonuniform parts can excite both branches. The split guarantees the desired molecular alpha only on the exact three-dimensional uniform subspace, not orthogonality under the continuum energy metric.

NEW FIXED-GEOMETRY COMPONENT EVIDENCE

All values below are kcal/mol. Native reference lane is M05-2X/6-31G* + official PySCF 2.13.1 SMD/IEFPCM water at the identical frozen MNSol geometry. It decomposes the solution energy into positive vacuum electronic distortion, negative reaction-field stabilization, electrostatic sum, stock CDS, and total. It is a mechanistic reference lane, not a same-operator oracle. The stock CDS is numerically identical between native and hybrid for each geometry.

Urea:
  experiment total = -13.8000
  native distortion = +3.2606
  native reaction field = -19.3467
  native electrostatic = -16.0861
  native CDS = +3.4819
  native total = -12.6042, error +1.1958
  hybrid G = -11.6836
  hybrid G+CDS = -8.2017, error +5.5983
  hybrid G minus native electrostatic = +4.4024 (hybrid under-stabilizes)

Uracil:
  experiment total = -16.5900
  native distortion = +4.3592
  native reaction field = -23.2297
  native electrostatic = -18.8704
  native CDS = +5.2135
  native total = -13.6569, error +2.9331
  hybrid G = -16.8012
  hybrid G+CDS = -11.5876, error +5.0024
  hybrid G minus native electrostatic = +2.0693 (hybrid under-stabilizes)

1,4,5,8-tetraminoanthraquinone:
  experiment total = -8.9000
  native distortion = +4.6455
  native reaction field = -25.5147
  native electrostatic = -20.8691
  native CDS = +5.1621
  native total = -15.7071, error -6.8071
  hybrid G = -28.7509
  hybrid G+CDS = -23.5889, error -14.6889
  hybrid G minus native electrostatic = -7.8818 (hybrid over-stabilizes even more)

Thus neither deleting CDS, adding the missing positive distortion, one dielectric/source scale, nor one sign change can repair all three cases. The native continuum model itself is poor for the large conjugated sentinel, but the hybrid amplifies its over-stabilization by another 7.88 kcal/mol.

AN EXACT TARGET-FREE DIAGNOSTIC IS NOW QUEUED

At the identical converged hybrid root, linearity allows exact branch decomposition by solving all seven nonzero combinations. Let E_p, E_r, E_a be self terms and

    C_pr = E_{p+r} - E_p - E_r,
    C_pa = E_{p+a} - E_p - E_a,
    C_ra = E_{r+a} - E_r - E_a.

Then

    G = E_p + E_r + E_a + C_pr + C_pa + C_ra

up to numerical residual. We will also record branch source norms, the total-solution half-work allocation, fixed-point Jacobian singular values, and the MDP/POLAR molecular polarizabilities. This experiment does not refit or change the predictor.

QUESTIONS

1. Starting from the equations above, derive the most discriminating interpretation of the six energy terms and the response Jacobian. What exact patterns would distinguish:
   A. a bad permanent MDP near-field source,
   B. excessive MACE-POLAR residual response,
   C. overlap/double counting between the residual and ADT response branches,
   D. a cavity/operator problem,
   E. nonlinear fixed-point feedback amplification?

2. Is the arithmetic uniform/nonuniform split mathematically sufficient for a source direct sum used inside a continuum reaction field? Or can the non-orthogonality of U G under the physical field/source energy pairing create systematic overlap despite exact closure on uniform fields? Give a careful derivation. Do not merely assert that projections should be orthogonal: specify the metric/operator and whether a geometry-dependent K_R-orthogonal projector would remain a legitimate, differentiable, target-free model rather than a disguised fit.

3. If a physical projector is warranted, derive the minimal form. Candidates include a source-response tangent projector, a receiver-space projector under a positive metric, or a continuum-active Schur/Gram projector. State required invertibility, gauge treatment, SO(3) covariance, passivity, differentiability, and whether the construction changes the original MACE-POLAR nonuniform response on ker(G).

4. Could the free-atom-density ADT allocation itself explain size/conjugation-dependent over-polarization even though the total molecular alpha is exact and positive? Identify a target-free observable that would falsify this (for example atom-resolved induced dipole leverage, continuum-active participation ratios, or comparison against MDP atomwise alpha if that latent decomposition is physically meaningful). Avoid treating an arbitrary atomic alpha partition as truth.

5. Does the sign-changing hybrid-minus-native electrostatic discrepancy logically rule out any broad classes of repair? Separate what is proved from what is only suggested by three sentinels.

6. Rank exactly the next three no-fit experiments after the branch decomposition. Each must have an explicit pass/fail observation and a termination decision. Prefer standard electrostatics/response diagnostics over novel machinery. We need one decisive next implementation, not an open-ended list.

7. Critically challenge the premise: is the operational hybrid attempting to combine quantities that are individually valid within their parent MLIPs but do not define a transferable PCM source/response? If so, give the earliest decisive test before any more 505-scale runs.

Please provide full mathematical derivations, an adversarial review of hidden assumptions, and a finite decision tree. Distinguish structural facts, numerical evidence, and hypotheses. End with the exact terminal marker:

HYBRID-BRANCH-ATTRIBUTION-DECISION
