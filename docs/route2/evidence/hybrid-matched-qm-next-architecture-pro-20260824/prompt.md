You are reviewing a quantitative implicit-solvation architecture. I need a rigorous mathematical/physical decision, not coding advice and not a fitted solvation residual.

Frozen constraints
- Keep the official MACE-MDP and MACE-POLAR checkpoints frozen.
- No MNSol/FreeSolv/experimental solvation target may train or select the electrostatic repair.
- Independent QM electrostatic/density/finite-field response supervision is allowed only if the zero-training architecture is now mathematically/physically exhausted.
- Permanent source, induced response, continuum, and CDS must remain separately identifiable.
- Do not propose geometry-dependent spectral clipping, source scaling chosen from solvation errors, or a cavity-specific projector disguised as a coordinate change.

Current hybrid
- permanent source: MACE-MDP latent atom-centered point monopoles/dipoles;
- nonuniform response: zero-anchored MACE-POLAR residual after subtracting its uniform tangent;
- uniform response: canonical atomic-density-translation (ADT) allocation of the negative MACE-MDP molecular polarizability;
- continuum: separated-source ddPCM; total development ledger adds stock SMD-CDS.

Complete target-open development result (method frozen target-free)
- baseline hybrid 505 MAE = 1.696313 kcal/mol;
- ADT hybrid 505 MAE = 1.638962 kcal/mol, RMSE = 2.277837, maximum = 14.688878;
- water 306 MAE = 2.011536; nonwater is much better;
- ADT improves only 0.057351 kcal/mol.

Decisive target-free matched-QM acetone audit
Reference: omegaB97M-V/def2-TZVPD finite-field response, def2-TZVP basis control, identical frozen PCMSolver cavity.
1. MDP permanent point q/p versus vacuum QM boundary MEP:
   area-weighted relative error 0.312913;
   passive-continuum relative response error 0.350424;
   correlation 0.944323.
2. MDP molecular polarizability versus QM molecular induced-dipole derivative:
   relative errors only 0.0032-0.0042 on x/y/z. Thus the total molecular alpha is excellent.
3. ADT spatial response versus QM induced cavity-boundary potential:
   area-weighted relative errors x/y/z = 0.3524/0.4932/0.4462;
   passive-continuum relative errors = 0.4929/0.6870/0.6396;
   correlations = 0.8883/0.8171/0.8298;
   errors are 7.49/13.85/9.46 times the independently frozen finite-field/basis uncertainty.
All three axes are resolved failures beyond reference uncertainty.
4. Fixed-root branch decomposition shows the worst 505 tail is not ADT-driven: for 1,4,5,8-tetraminoanthraquinone, permanent self work is -21.63 kcal/mol, permanent-radial cross work -6.47, radial self -0.66, ADT approximately zero, total electrostatic -28.75 versus experimental total -8.9.
5. An earlier molecule-held-out MDP-feature MBIS q/p head reduced fixed-source ddPCM MAE from 8.19 to 2.53 kcal/mol but failed its physical gate. Error decomposition concluded q/p/Q is terminally insufficient and requires l<=3 or a compact density/potential basis. An oracle using exact MBIS q/p projected to the frozen MDP molecular dipole had 0.022 kcal/mol MAE, but this is evaluation-only and unavailable at inference.

Questions
A. Does this evidence terminate the unchanged MDP-point + POLAR-residual + ADT hybrid as a quantitative PCM source, or is there any legitimate zero-training loophole left? State necessary and sufficient caveats.
B. If a new small trained component is now unavoidable, give the minimal scientifically defensible architecture while freezing both backbones. Compare:
   (i) an SO(3)-equivariant permanent l<=3 multipole/density/potential head plus a separately supervised induced-response head;
   (ii) a scalar-first external-field energy head whose source and susceptibility are derivatives of one scalar;
   (iii) a compact atom-centered density/potential basis predicted from frozen features.
C. Should the excellent MDP molecular dipole/polarizability be imposed as hard affine constraints, soft losses, or only diagnostics? Derive the constraint/projection so total charge, molecular dipole, and linear polarizability are exact without making the cavity-boundary response nonlocal or cavity-specific.
D. Specify independent QM targets, losses, representation, and admission tests that test the actual continuum-active boundary observable. Include permanent MEP, induced MEP for three fields, electron-number conservation, reciprocity, passivity, rotational covariance, and held-out molecule/scaffold rules.
E. Decide whether the original MACE-POLAR nonlinear residual can remain as a separately identified branch, or whether the matched-QM result requires replacing all induced response. Give a finite terminal experiment rather than an open-ended tuning program.
F. Give a fail-closed decision tree. The goal is eventually <=1 kcal/mol without solvation-label fitting, but do not promise that the architecture can achieve it.

Please explicitly distinguish theorem-level conclusions, evidence-supported inferences, and unknowns. Challenge the premise if needed.