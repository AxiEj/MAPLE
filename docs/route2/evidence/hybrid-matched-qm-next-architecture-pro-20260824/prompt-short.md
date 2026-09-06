Give a rigorous architecture decision for one frozen-checkpoint implicit-solvation hybrid. No coding advice and no solvation-target fitting.

Frozen model:
- permanent source = MACE-MDP latent atom-centered point q/p;
- nonlinear nonuniform response = zero-anchored MACE-POLAR residual;
- uniform response = ADT allocation of the MACE-MDP molecular polarizability;
- separated ddPCM + stock SMD-CDS.

Complete development result: baseline 505 MAE 1.696313 kcal/mol; ADT hybrid 1.638962, water 2.011536, maximum 14.688878.

Decisive target-free acetone comparison on the identical PCMSolver cavity:
- MDP permanent MEP: area relative error 0.312913, passive-continuum relative error 0.350424, correlation 0.944323.
- MDP molecular polarizability is excellent: induced-dipole derivative differs from omegaB97M-V/def2-TZVPD finite-field QM by only 0.32-0.42%.
- Nevertheless ADT induced cavity-boundary potentials fail on x/y/z: area errors 0.352/0.493/0.446; passive-continuum errors 0.493/0.687/0.640; correlations 0.888/0.817/0.830. These are 7.49/13.85/9.46 times frozen basis/finite-field uncertainty.
- An earlier molecule-held-out MDP-feature MBIS q/p head reduced fixed-source ddPCM MAE 8.19 to 2.53 but failed. Evidence says q/p/Q is insufficient; l<=3 or a compact density/potential basis is needed.

Answer only these:
1. Is any legitimate zero-training loophole left for the unchanged MDP-point + POLAR-residual + ADT source? State caveats.
2. If not, choose the minimal new component while freezing both backbones:
   A) separate equivariant permanent l<=3/density head plus induced-response head;
   B) one scalar-first external-field energy head;
   C) another architecture.
3. Derive how exact total charge, MDP molecular dipole, and MDP molecular polarizability should be imposed without making the model cavity-specific.
4. Specify independent QM supervision and finite pass/fail gates for permanent and induced cavity-boundary potentials, reciprocity, passivity, rotations, and molecule-held-out validation.
5. Decide whether the original MACE-POLAR nonlinear residual can remain. Give one finite terminal experiment.
6. Distinguish theorem, evidence-supported inference, and unknown. Do not promise <=1 kcal/mol.