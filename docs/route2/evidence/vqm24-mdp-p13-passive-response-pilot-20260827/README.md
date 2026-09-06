# VQM24 MDP-only block-passive P13 response pilot

The preregistered A2 and A3 development pilots use only the 32 frozen train
formulas and QM exterior-MEP / molecular-dipole linear-response observables.
The 14 validation and 14 blind formulas remain unopened.  MACE-MDP point q/p
remains the permanent affine source; the learned induced response is generated
only as the derivative of `-1/2 ||C_theta(R) eta_P13||^2`.

Both candidates preserve charge, reciprocity and passivity to numerical
precision.  A3 improves the frozen score by 12.85% and reduces mean rotated-
audit response MEP error from the old approximately 46.5% response to 11.43%,
but fails the preregistered 8% mean, 20% worst, 5% source-point mean and 0.002
Ha/e absolute gates.  It is therefore not selected or admitted.  Its training
loss was still descending at epoch 1500, so the next decision must distinguish
optimization underconvergence from missing frozen-POLAR descriptors before any
capacity escalation.
