You are reviewing a quantitative molecular-response data generator for a conservative implicit-solvation ML model. I need a mathematically exact static closed-shell Kohn-Sham linear-response derivation and a critical implementation review, not general advice.

Frozen reference protocol
- PySCF 2.13.1, neutral singlet RKS, omegaB97M-V/def2-TZVPD, density fitting.
- Semilocal and nonlocal grids are both PySCF level 3.
- A converged gas checkpoint stores C, occupations, orbital energies, and E0.
- Four external point-charge modes are centered at fixed exterior points s_k.
- A positive source amplitude q changes the electronic one-electron Hamiltonian by h'(s_k) = - int1e_rinv(s_k), since the scalar potential is +q/|r-s_k| and electron charge is negative.
- Current oracle runs independent SCFs at q = +/-3e-4 and +/-1e-3 e for every mode. It measures central derivatives of electronic exterior MEP, total dipole, and energy. Across CHNO and one F/P/S/Cl/Br molecule, step inconsistencies are only 1.3e-4 to 5.8e-4 relative and all energy curvatures are negative.
- This data is independent QM response supervision. It does not use PCM, a cavity, VQM24 energies, or experimental solvation targets.

Proposed acceleration
Use PySCF's existing ground-state static CPKS machinery for all four arbitrary h' right-hand sides in one solve:
1. construct a level-3 DFRKS object with the checkpoint orbitals/occupations/energies;
2. vresp = mf.gen_response(C, occ, singlet=None, hermi=1, with_nlc=True);
3. h_vo[k] = C_vir.T @ h'_k @ C_occ;
4. solve scf.cphf.solve(fvind, eps, occ, h_vo) with an fvind that maps orbital response amplitudes to the symmetrized closed-shell AO density response, applies vresp, and returns its virtual-occupied block;
5. reconstruct dD/dq and contract it with exterior int1e_rinv and int1e_r operators.

Please answer all of the following precisely.

A. Derive the static CPKS equations in the exact conventions above. State the shape and sign of the unknown orbital-response amplitudes and h_vo. State every closed-shell factor of two in dD and in fvind. Explain whether cphf.solve should receive only the virtual-occupied block or a full nmo x nocc block when the AO basis and nuclei are fixed.

B. Give the correct observable contractions per unit source charge:
- derivative of electronic MEP at an exterior point r: V_e(r) = -Tr[D J(r)];
- derivative of total dipole: mu = mu_nuc - Tr[D r];
- second derivative E''(0) for the energy under h(q)=h0+q h'.
Freeze signs and any factors of 1/2. Explain why E'' should be non-positive for a stable passive electronic ground state in this convention.

C. Audit the proposed PySCF implementation. In particular:
- whether mf.gen_response(..., singlet=None, hermi=1, with_nlc=True) is the correct ground-state RKS kernel for omegaB97M-V;
- whether density fitting and the level-3 NLC grid are retained without rerunning a zero-field SCF;
- which checkpoint quantities must be assigned to mf before gen_response;
- whether shared grids must be explicitly built;
- how to handle all four right-hand sides in one cphf.solve call;
- robust residual checks that directly test the solved CPKS equation.

D. Specify a decisive validation against the already completed finite-field oracle. Give separate relative/absolute metrics for dD, induced exterior MEP, induced dipole, E'', reciprocity, electron-number conservation, and passivity. The CPKS backend may be used for batch data only if it agrees with finite differences within the observed two-step truncation/noise budget. Identify any case where finite-field SCF and static CPKS can legitimately disagree (SCF branch change, near instability, inconsistent NLC kernel/grid, checkpoint mismatch, etc.).

E. Critically look for hidden category errors: Are these four point-charge response directions sufficient only as observable training probes, rather than a complete density label? Does replacing 16 finite-field SCFs with CPKS change the scientific target or merely compute its q->0 derivative more directly? State the fail-closed conditions.

Return equations, pseudocode close to PySCF 2.13.1 APIs, and a short implementation checklist. Do not suggest fitting solvation targets, density-partition coefficients, or an empirical correction.
