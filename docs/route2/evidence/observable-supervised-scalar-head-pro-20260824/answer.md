## Final decision — **GO, as a new unified scalar-response architecture**

Discard **all density-coefficient labels** and stop searching for a unique atomwise partition of the ETB-(\beta2) projection. The failed (0.707/1.011) kcal/mol rigorous-bound gate remains a failure; the (0.094) actual mean error does not override it.

The unpartitioned ETB-(\beta2) result instead establishes a narrower fact: **the fixed span has sufficient physical representation capacity, but its overlapping molecular coefficients are not observables**. Retain the functions as an internal source/field basis, not as supervised labels.

### Minimal passive functional

Let (B_Rz=\sum_a z_a\phi_a^R) be the source represented in fixed, cavity-independent atom-centred radial–spherical functions, and let

[
\eta_R(v)=B_R^{*}v
]

be the exactly dual projection of an external potential (v). Let (\mathcal C_Q) impose the exact molecular charge or electron count.

Define

[
A_\theta(R,z)
=============

\Psi_\theta^{\mathrm{cvx}}(h_R,z)
-b_\theta(h_R)^{T}z
+\frac{\mu}{2}z^{T}Dz ,
\qquad
\mu>0,\quad D\succ0 ,
]

where:

* (\Psi_\theta^{\mathrm{cvx}}) is convex in (z);
* (D) is a **fixed per-element/local metric**, preferably fixed once after isolated-atom whitening;
* (b_\theta) creates the geometry-dependent zero-field source;
* (h_R) contains only frozen geometry representations;
* strong convexity is required on the charge-conserving tangent space.

The field-conditioned scalar is

[
\boxed{
\mathcal E_\theta(R,v)
======================

E_{\mathrm{vac}}^{\mathrm{MACE\text{-}MDP}}(R)
+
\min_{z\in\mathcal C_Q}
\left[A_\theta(R,z)-\eta_R(v)^Tz\right]
---------------------------------------

\min_{z\in\mathcal C_Q}A_\theta(R,z)
}
]

The subtraction anchors the learned field contribution exactly to zero at (v=0). A convex neural scalar such as an ICNN is one admissible implementation of (\Psi_\theta^{\mathrm{cvx}}), although the architecture need not specifically be an ICNN. ([Proceedings of Machine Learning Research][1])

This convex-in-source/concave-in-potential structure is the appropriate finite-dimensional analogue of the convex dual formulation of electronic ground-state response. Learning one generalized energy and deriving response quantities from its derivatives enforces consistency that separate energy, dipole, and polarizability heads do not. ([arXiv][2])

### One derivative gives both sources

Let

[
z^\star(R,v)
============

\arg\min_{z\in\mathcal C_Q}
\left[A_\theta(R,z)-\eta_R(v)^Tz\right].
]

Using the convention that the source couples as (-\langle s,v\rangle),

[
s_\theta(R,v)
=============

# -\frac{\delta\mathcal E_\theta}{\delta v}

B_Rz^\star(R,v).
]

Therefore

[
s_{\mathrm{perm}}(R)=B_Rz^\star(R,0),
]

and

[
s_{\mathrm{ind}}(R,v)
=====================

B_R!\left[z^\star(R,v)-z^\star(R,0)\right].
]

Thus permanent and induced sources are not separate branches, not separately fitted objects, and not joined by ADT. They are the zero-field value and zero-anchored increment of **the same conjugate derivative**.

If (H=\nabla_z^2A_\theta), the response on the charge-neutral tangent space (T_Q) is

[
\chi_\theta
===========

# \frac{\delta s_\theta}{\delta v}

B_R
\left(H|_{T_Q}\right)^{-1}
B_R^{*}
\succeq0 .
]

Hence reciprocity and positive-semidefinite susceptibility follow from the scalar and strong convexity. The field energy is concave, and its induced part cannot generate energy by an infinitesimal passive cycle.

### What strong convexity fixes

The term (\frac{\mu}{2}z^TDz) makes (z^\star) unique even when overlapping ETB functions make (B_R) nearly singular or give it latent null directions. No molecular Coulomb pseudoinverse, spectral cutoff, geometry-dependent whitening, or coefficient target is needed.

This does **not** turn individual coefficients into physically identifiable observables. It instead defines a unique internal gauge as part of the model. Scientific outputs are only

[
\mathcal E_\theta,\qquad B_Rz^\star,\qquad
\frac{\delta B_Rz^\star}{\delta v},
]

not the entries of (z^\star). The choices of basis, (D), and convexity floor (\mu) must therefore be frozen as part of the architecture identity.

## Required training loss

Every target must supervise the scalar or an exact derivative/projection of it:

[
\mathcal L=
w_E\mathcal L_{\Delta E}
+w_V\mathcal L_{\mathrm{MEP}}
+w_\mu\mathcal L_{\mathrm{dipole}}
+w_\alpha\mathcal L_{\mathrm{polarizability}}
+w_{\mathrm{nr}}\mathcal L_{\mathrm{nonuniform}} .
]

Specifically:

* (\mathcal L_{\Delta E}): independent-QM energy changes under registered uniform and nonuniform external potentials.
* (\mathcal L_{\mathrm{MEP}}): exterior MEP obtained by applying the exact Coulomb operator to (B_Rz^\star), with no learned MEP head.
* (\mathcal L_{\mathrm{dipole}}): dipole from the first derivative with respect to a uniform field.
* (\mathcal L_{\mathrm{polarizability}}): polarizability from the corresponding second derivative.
* (\mathcal L_{\mathrm{nonuniform}}): induced exterior MEP/source response under a frozen family of smooth, cavity-independent nonuniform probes.

There is **no coefficient loss**, Hirshfeld loss, density-partition loss, MACE-POLAR-source imitation, ADT loss, PCM-energy loss, or experimental-solvation loss. Loss normalizations and weights must be frozen from QM numerical scales, not selected using PCMSolver or solvation performance.

## Required gates

1. **Structural gates:** exact charge; zero induced charge; SO(3) covariance and translation invariance; registered strong-convexity floor and Hessian condition cap; unique stationary solution; automatic-differentiation versus finite-difference agreement; reciprocity; positive-semidefinite response; and passive field-cycle work.

2. **Independent-QM gates:** geometry-grouped and field-family-held-out gates for field energies, zero-field MEP/dipole, polarizability, and genuinely nonuniform induced response. Each category must pass separately; a weighted aggregate cannot compensate for a failed response class.

3. **Continuum-transfer gate:** apply the already justified rigorous reaction-energy budget—mean bound (\le 0.25) and maximum bound (\le 0.50) kcal/mol—to a **new unopened panel**. PCMSolver may appear only here as an evaluator; its cavity and reaction fields may not be training inputs or labels. The opened 12-case Hirshfeld panel cannot select the new basis, (\mu), loss weights, or checkpoints.

Any gate failure is final for that registered model.

## Frozen MACE features

Frozen MACE-MDP and zero-field MACE-POLAR latent representations may condition (h_R). The native MACE-POLAR field channels may be retained only when they are a fixed linear, equivariant representation of (v) entering the conjugate coupling with the full derivative path preserved.

Arbitrary field-conditioned MACE-POLAR hidden activations must not bypass that coupling, because unconstrained nonlinear field dependence would destroy the concavity/passivity guarantee.

The following remain excluded:

* the original MACE-POLAR source output;
* its uniform tangent and ADT construction;
* any additive induced-source branch;
* MACE-MDP atomwise (q/p) as an independently asserted physical permanent source.

MACE-MDP may still provide the frozen vacuum scalar and geometry embeddings, but the electrostatic permanent and induced sources must both come from the derivative of the new scalar.

**Therefore the legitimate next architecture is a cavity-blind, independent-QM-supervised, strongly convex latent source functional. It preserves the accurate ETB-(\beta2) span while eliminating coefficient labels and global pseudoinverse identity from the scientific model.**

[1]: https://proceedings.mlr.press/v70/amos17b.html "Input Convex Neural Networks"
[2]: https://arxiv.org/html/2204.12216v1 "Lieb variation principle in density-functional theory"
