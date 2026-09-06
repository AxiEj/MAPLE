You are reviewing a preregistered post-pilot architecture decision for a scalar-first MACE-MDP + MACE-POLAR electrostatic response model. This is a mathematical/model-selection review, not a request to relax failed gates. Please make one finite next-step recommendation and explicit stop rules.

Scientific objective and prohibitions
- Long-term objective: improve a MACE-MDP permanent + learned passive response + reciprocal ddPCM hybrid toward 1 kcal/mol total solvation accuracy.
- MACE-MDP point monopoles/dipoles remain the permanent affine source in this branch.
- No experimental solvation labels, PCM energies, cavity-dependent labels, fitted per-molecule source coefficients, original MACE-POLAR response targets, or VQM24 total energies may enter training/model selection.
- The 32 train molecules are open. The 14 validation and 14 blind formulas remain unopened.
- Every induced source must be the derivative of one scalar; passivity and reciprocity may not be imposed by clipping after prediction.

Frozen observable data
- 32 chemically diverse neutral train formulas, H/C/N/O/F/P/S/Cl/Br.
- Four independent nonuniform exterior point-charge modes per molecule.
- Dense exterior MEP fit frame and independently rotated audit frame, each overdetermined relative to the source space.
- QM induced exterior MEP and induced molecular dipole from converged +/-0.001 e finite-field states.
- MEP/dipole nonlinearity is below 3.4e-4/3.1e-4, so a quadratic response is justified.

Current scalar and factor
E_response(R,eta) = -1/2 ||C_theta(R) eta||^2.
The induced P13 source is obtained by autograd with respect to eta. P13 contains two radial l<=1 Gaussian channels (8N) plus one Gaussian STF l=2 block (5N, sigma=1.5 Angstrom). C is currently block diagonal between radial and l=2 sectors.

A2 factor rows:
1. one atom-local radial-null scalar row per atom;
2. two atom-local Cartesian vector rows per atom, using I and the symmetric MDP atomic polarizability as equivariant carriers;
3. one isotropic local l=2 row block per atom.

A3 adds one symmetric local edge row for each unordered pair inside a C2 quintic 5 Angstrom cutoff. Atom/edge coefficients come from a tiny MDP-only invariant head: frozen MDP q, p, alpha, element embeddings and fixed neighbor sums; no new message passing. A2 has 432 parameters; A3 has 634. Both are below the frozen 2500 cap.

Training
- 1500 Adam epochs, lr 1e-3, weight decay 1e-4, float64 GPU.
- loss = 0.82 normalized fit-frame response-MEP MSE + 0.18 induced-dipole MSE.
- audit probes never enter gradients.
- Both models satisfy exact charge, reciprocity (~2.4e-16) and negative-semidefinite source-point response. Fit/audit ratios are essentially 1, so there is no probe-frame interpolation signal.

Results
Old MACE-POLAR nonuniform response mean error: about 46.5%.
A2:
- mean fit/audit response MEP: 13.20% / 13.15%
- worst audit: 21.99%
- mean/worst induced dipole: 8.62% / 13.00%
- mean/worst source-point response: 14.01% / 25.32%
- max absolute audit response error: 0.01442 Ha/e
- final normalized MEP loss at epochs 1300/1400/1499: 0.02250/0.02190/0.02127.

A3:
- mean fit/audit response MEP: 11.49% / 11.43%
- worst audit: 20.08%
- mean/worst induced dipole: 7.64% / 12.05%
- mean/worst source-point response: 12.93% / 23.08%
- max absolute audit response error: 0.01324 Ha/e
- score improves 12.85% over A2
- final normalized MEP loss at epochs 1300/1400/1499: 0.01727/0.01676/0.01633.

Frozen gates failed by A3:
- mean audit MEP <=8%; observed 11.43%
- worst audit MEP <=20%; observed 20.08%
- mean source-point response <=5%; observed 12.93%
- worst source-point response <=20%; observed 23.08%
- max absolute audit response <=0.002 Ha/e; observed 0.01324.
Dipole, charge, reciprocity, passivity, and fit-to-audit gates pass. The 46.5/11.43 ratio is 4.07, so the separate fourfold-improvement requirement barely passes.

Candidate next moves, none yet executed
O1. Keep identical A3 architecture/objective and continue optimization from its frozen checkpoint with a lower learning rate and a preregistered plateau criterion. This tests optimization underconvergence without adding capacity.
O2. Replace the slow nested autograd training backend by the exact analytic derivative of the same factor scalar, with a unit test against autograd, then use full-batch L-BFGS or longer Adam. This changes no mathematical model.
A4. Add only target-free zero-field MACE-POLAR latent descriptors to the coefficient heads, using train-only multiplicity-space PCA capped at 8 scalar and 4 vector irrep channels, no original source/response and no new message passing. Keep the same factor topology and <=4096 parameters.
C. Add radial-l2 cross factor rows, still inside one squared norm, so the susceptibility remains C^T C. Four external point modes jointly excite radial and l2 fields, but the data may not identify a transferable cross block.

Questions
1. Does fit approximately equal audit plus a still-decreasing training loss justify O1/O2 before A4, or do the error magnitudes already indicate approximation/capacity failure? Give a quantitative, preregisterable plateau rule.
2. Is an exact analytic derivative backend scientifically equivalent to source-from-scalar autograd when equality is locked by tests, or would it undermine the scalar-first guarantee?
3. If optimization continuation fails, should the next bounded candidate be A4 descriptors or cross-factor topology C? Analyze identifiability with 32 molecules x 4 nonuniform modes, SO(3) equivariance, passivity, dissociation/locality, and force differentiability.
4. The absolute 0.002 Ha/e gate is much stricter than the relative gate for several records. Do not change it merely because it failed. Explain whether it is physically defensible for this training-stage exterior-MEP response, and what independent error-budget derivation would be required before any future revision.
5. Give a finite sequence with hard stop rules that prevents indefinite capacity growth or development-set fitting. State exactly when to open grouped train-CV, the 14 validation formulas, and the blind set.
6. Critically challenge any hidden category error: MDP point q/p is the permanent source, while the learned response uses radial-GTO plus Gaussian-l2 induced sources. Is that direct-sum scalar physically coherent, and what coupling/energy identity must be tested before ddPCM integration?

Do not recommend experimental solvation fitting, residual corrections, eigenvalue clipping, geometry-dependent active spaces, or relaxing a gate after seeing these results. Distinguish advice from what is mathematically proved by the current evidence.
