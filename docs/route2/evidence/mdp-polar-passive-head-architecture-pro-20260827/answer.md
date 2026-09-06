# Hard decision

**Use the direct reduced-field quadratic scalar for v1. Do not introduce an inner latent-source minimization yet.**

The v1 head should be

$$
\boxed{
E_\theta(R,\eta)
=
E_{\rm vac}(R)
+s_Q^{\mathsf T}\eta
+m_0(R)^{\mathsf T}\xi
-\frac12\|C_\theta(R)\xi\|^2
}
$$

with

$$
\xi=U_N^{\mathsf T}\eta,\qquad
K_\theta=C_\theta^{\mathsf T}C_\theta\succeq0,
$$

and therefore

$$
\boxed{
s_\theta(R,\eta)
=
\nabla_\eta E_\theta
=
s_Q+U_N\!\left[m_0-K_\theta\xi\right].
}
$$

Here \(s_\theta\in\mathbb R^{8N}\) is the full two-width \(l\leq1\) molecular source, \(s_Q\) is a fixed particular source with total charge \(Q\), and \(U_N\in\mathbb R^{8N\times(8N-1)}\) is the Householder charge-null chart.

This gives

$$
\frac{\partial s_\theta}{\partial \eta}
=
-U_NK_\theta U_N^{\mathsf T}\preceq0,
$$

so response is reciprocal and passive by construction. It also exactly matches the information content of the dataset: a reference source and first-order response. A nonlinear source functional is not identifiable from one small-amplitude pair per mode.

The current Baldwin et al. record is v2, revised August 8, 2026. Its analysis makes the relevant distinction particularly clear: quadratic source functionals and linear response maps can be dual descriptions, but electronically stiff or zero-susceptibility modes require extremely large source-functional curvature and are materially harder to train. It separately reiterates ordinary QEq’s fractional dissociation and unphysical \(N^3\) polarizability scaling. ([arXiv][1]) The same preprint’s coarse-graining derivation identifies reference-density fidelity, first-order-response fidelity, and control of higher-order response as distinct conditions, while retaining applied-field work explicitly. ([arXiv][2])

The concrete v1 is therefore:

> **MDP-seeded permanent source + sparse atom/pair factorized susceptibility, using MDP-only descriptors initially, no new message-passing layer, no inner source solve, and no more than about 2,500 trainable scalar parameters.**

Zero-field MACE-POLAR features are an escalation ablation, not the default.

---

# A. Direct quadratic field energy versus a latent source functional

## A.1 Exact relationship

At fixed geometry, consider the quadratic latent functional

$$
A(R,z)
=
E_{\rm vac}(R)
+\frac12(z-m_0)^{\mathsf T}H(R)(z-m_0),
\qquad H\succ0.
$$

Using the present sign convention, the external enthalpy is

$$
E(R,\xi)=\min_z\left[A(R,z)+\xi^{\mathsf T}z\right].
$$

Stationarity gives

$$
H(z-m_0)+\xi=0,
\qquad
z=m_0-H^{-1}\xi,
$$

and consequently

$$
E(R,\xi)
=
E_{\rm vac}
+m_0^{\mathsf T}\xi
-\frac12\xi^{\mathsf T}H^{-1}\xi.
$$

Thus

$$
K=H^{-1}=C^{\mathsf T}C.
$$

The alternative sign in \(A-\eta^{\mathsf T}z\) is equivalent after setting \(\eta=-\xi\).

Therefore a **quadratic strongly convex latent functional has no additional linear-response expressivity**. It simply parameterizes the inverse susceptibility rather than the susceptibility.

## A.2 Comparison

| Property                       | Direct quadratic field scalar                                    | Strongly convex latent source functional                                  |    |                                           |
| ------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------- | -- | ----------------------------------------- |
| Identifiable from current data | Yes: \(m_0\) and first-order \(K\)                               | Quadratic part yes; nonlinear part no                                     |    |                                           |
| Linear-response expressivity   | Every PSD \(K\), subject to chosen sparse factor topology        | Every SPD \(K=H^{-1}\), if \(H\) is unrestricted                          |    |                                           |
| Zero-susceptibility modes      | Natural: exact nullspace of \(K\)                                | Require infinite or extremely large curvature in \(H\)                    |    |                                           |
| Variable \(N\)                 | Sparse \(C\xi\), \(C^{\mathsf T}y\), (O(N+                       | E                                                                         | )) | Requires minimizing or solving with \(H\) |
| Local parameterization         | Local \(K\) remains local                                        | A local sparse \(H\) generally has a dense inverse                        |    |                                           |
| Passivity                      | Immediate from \(K=C^{\mathsf T}C\)                              | Immediate from convexity of \(A\)                                         |    |                                           |
| Solute-state uniqueness        | Source is explicit even when \(K\) is singular                   | Requires strong convexity                                                 |    |                                           |
| Coordinate forces              | One scalar graph; only PCM root is implicit                      | Additional source minimization and implicit derivatives                   |    |                                           |
| PCM coupling                   | One solute–continuum root                                        | Possible joint minimization, but combined convexity still must be checked |    |                                           |
| QEq risk                       | Avoided when \(C\) is finite-range and no matrix inverse is used | Local hardness inversion can recreate global equalization                 |    |                                           |

## A.3 Why the latent formulation is the wrong v1

A nonlinear \(A_\theta(R,z)\) would introduce unconstrained second- and higher-order field response that the present \(\pm10^{-3}e\) data do not identify. Infinitely many nonlinear functionals have the same minimum, reference source, and Hessian at the sampled state.

A quadratic \(A\) is mathematically equivalent to the direct model but computationally worse:

$$
H \text{ sparse and local}
\quad\not\Rightarrow\quad
H^{-1} \text{ sparse and local}.
$$

That inversion is precisely the dangerous direction for a small molecular dataset: global charge equalization, nonlocal fragment coupling, and potentially metallic size scaling.

The latent formulation becomes justified only after a sealed multi-amplitude dataset demonstrates statistically significant nonlinear MEP or dipole response that cannot be represented by a quadratic field energy. Until then, it adds an unidentifiable function and an unnecessary solve.

---

# B. Explicit variable-\(N\), equivariant parameterization

## B.1 Physical source and field spaces

For atom \(i\),

$$
s_i=
\left(
q_{i1},q_{i2},
\mathbf p_{i1},\mathbf p_{i2}
\right)
\in
2(0e)\oplus2(1o),
$$

where radial index \(a=1,2\) corresponds to \(\sigma=(1.5,3.0)\) Å.

Thus

$$
s\in\mathbb R^{D},\qquad D=8N.
$$

The conjugate field has the same decomposition,

$$
\eta_i=
\left(
\phi_{i1},\phi_{i2},
\mathbf e_{i1},\mathbf e_{i2}
\right).
$$

All formulas below assume the already frozen dual source/field normalization. If raw GTO coordinates require channel scaling, use a fixed dual transformation

$$
\tilde s=S^{-1}s,\qquad
\tilde\eta=S^{\mathsf T}\eta,
$$

which preserves

$$
s^{\mathsf T}\eta=\tilde s^{\mathsf T}\tilde\eta.
$$

Do not construct a new probe-dependent whitening or a fitted coefficient metric.

## B.2 Exact fixed-charge chart

Let \(g_N\in\mathbb R^{8N}\) be the analytic total-charge covector. It has nonzero entries only in the two scalar radial channels:

$$
g_N^{\mathsf T}s=Q.
$$

Define

$$
\widehat g=\frac{g_N}{\|g_N\|},
\qquad
s_Q=\frac{Q}{g_N^{\mathsf T}g_N}g_N.
$$

A Householder matrix \(H_N\) maps the first scalar coordinate onto \(\widehat g\). Let

$$
U_N=H_N[:,1:D],
$$

so

$$
U_N^{\mathsf T}U_N=I,\qquad
U_N^{\mathsf T}g_N=0,\qquad
U_NU_N^{\mathsf T}=I-\widehat g\widehat g^{\mathsf T}.
$$

The neural head must operate in the **full physical atomwise space**, not directly on arbitrary reduced indices. The Householder chart is used only for the final exact coordinate conversion:

$$
m_0=U_N^{\mathsf T}(s_0-s_Q),\qquad C=B_\theta U_N.
$$

That avoids atom-order-dependent reduced-coordinate learning.

## B.3 Radial moment decomposition

Let \(M_0\in\mathbb R^{1\times2}\) map the two scalar radial coefficients to physical charge, and \(M_1\in\mathbb R^{1\times2}\) map the two \(l=1\) radial coefficients to physical dipole.

Define

$$
w_l=M_l^{\mathsf T}(M_lM_l^{\mathsf T})^{-1},
\qquad
M_lw_l=1,
$$

and choose unit vectors

$$
n_l\in\ker M_l,
\qquad M_ln_l=0.
$$

Interpretation:

* \(w_0\): a unit physical charge distributed over the two scalar widths.
* \(n_0\): an intra-atomic radial redistribution with zero net charge.
* \(w_1\): a unit physical dipole distributed over the two vector widths.
* \(n_1\): a radial reshaping of a dipole that leaves its physical dipole moment unchanged.

This is the right way to mix the two widths. A free unconstrained \(2\times2\) mixing matrix would obscure charge and moment conservation.

## B.4 Permanent source \(s_0(R)\)

Use MACE-MDP \(q_i^{\rm MDP}\) and \(\mathbf p_i^{\rm MDP}\) as an initialization and descriptor, not as supervised atomic truth.

A minimal explicit construction is

$$
\begin{aligned}
\mathbf q_i^{\rm rad}
&=
(w_0+\beta_0n_0)\,q_i^{\rm MDP}
+n_0\,a_i
+(w_0+\bar\beta_0n_0)\sum_{j}t_{ij},
\\[3pt]
\mathbf p_i^{\rm rad}
&=
(w_1+\beta_1n_1)\otimes\mathbf p_i^{\rm MDP}
+w_1\otimes\mathbf u_i
+n_1\otimes\mathbf v_i .
\end{aligned}
$$

Here:

* \(a_i\in\mathbb R\) changes the radial scalar shape without changing atomic charge.
* \(\mathbf u_i,\mathbf v_i\in\mathbb R^3\) are equivariant dipolar corrections.
* \(t_{ij}=-t_{ji}\) is a local charge transfer.
* \(\beta_0,\bar\beta_0,\beta_1\) are at most three shared scalar radial-mixing parameters, not molecule-specific parameters.

Construct the antisymmetric transfer as

$$
t_{ij}
=
\frac12\chi(r_{ij})
\left[
f_\theta(h_i,h_j,r_{ij})
-
f_\theta(h_j,h_i,r_{ij})
\right],
$$

with a \(C^2\) or \(C^3\) cutoff \(\chi\).

Then

$$
\sum_i\sum_j t_{ij}=0
$$

and therefore

$$
g_N^{\mathsf T}s_0
=
\sum_iq_i^{\rm MDP}=Q.
$$

The vector carriers for \(\mathbf u_i\) and \(\mathbf v_i\) can be restricted to

$$
\left\{
\mathbf p_i^{\rm MDP},\;
\bar\alpha_i^{\rm MDP}\mathbf p_i^{\rm MDP},\;
\sum_j\chi_k(r_{ij})\widehat{\mathbf r}_{ij}
\right\},
$$

with scalar invariant coefficients generated by the small head. Optional frozen MACE-POLAR \(1o\) features may later be appended as additional vector carriers.

No atomic source coefficient is used as a target.

## B.5 Sparse factor \(B_\theta\)

Define an implicit full-space factor

$$
B_\theta(R):\mathbb R^{8N}\rightarrow\mathcal Y_N,
\qquad
C_\theta=B_\theta U_N.
$$

For v1, use three block types.

### 1. One scalar radial-null row per atom

$$
y_i^{(0)}
=
a_i^{(0)}\,n_0^{\mathsf T}\boldsymbol\phi_i,
\qquad
\boldsymbol\phi_i=(\phi_{i1},\phi_{i2}).
$$

This responds only to the difference between the two radial scalar channels and cannot change total charge.

### 2. Two vector rows per atom

Let

$$
\mathbf e_i=
\begin{bmatrix}
\mathbf e_{i1}\\
\mathbf e_{i2}
\end{bmatrix}.
$$

For \(k=1,2\),

$$
\boxed{
\mathbf y_{ik}^{(1)}
=
\sum_{a=1}^{2}
\left[
u_{ika}I_3+v_{ika}\bar\alpha_i^{\rm MDP}
\right]\mathbf e_{ia}.
}
$$

The scalar coefficients \(u_{ika},v_{ika}\) are outputs of a shared invariant atom head. Since

$$
\bar\alpha_i\mapsto R\bar\alpha_iR^{\mathsf T},
\qquad
\mathbf e_{ia}\mapsto R\mathbf e_{ia},
$$

each \(\mathbf y_{ik}^{(1)}\) transforms as a vector.

A fixed-MDP-polarizability initialization may be used only when the relevant symmetrized atomic tensor is PSD. In that case, with

$$
S_i^{\mathsf T}S_i=\alpha_i^{\rm MDP},
$$

initialize one vector row with the analytic uniform-field radial combination and \(S_i\). A learned correction is added **inside the same factor row**:

$$
T_{ia}=\beta_aS_i+\Delta T_{ia}.
$$

This permits both increasing and decreasing the baseline susceptibility while keeping \(K=B^{\mathsf T}B\) PSD.

Do not use

$$
K=K_{\rm MDP}+\Delta C^{\mathsf T}\Delta C
$$

as the only correction form, because it can only increase susceptibility.

If an atomic MDP polarizability is indefinite, do not eigenclip it and call it physical. Use it only as an equivariant descriptor.

### 3. One scalar row per local edge

For every unordered edge \(e=(i,j)\),

$$
\boxed{
y_{ij}^{(e)}
=
\chi(r_{ij})
\left[
c_{ij}\,\widetilde w_0^{\mathsf T}
(\boldsymbol\phi_i-\boldsymbol\phi_j)
+
d_{ij}\,
\widehat{\mathbf r}_{ij}\cdot
\widetilde w_1^{\mathsf T}
\frac{\mathbf e_i+\mathbf e_j}{2}
\right].
}
$$

Here

$$
\widetilde w_l=w_l+\gamma_ln_l
$$

allows one shared radial reshaping while preserving physical moments. The gains \(c_{ij},d_{ij}\) are symmetric under \(i\leftrightarrow j\).

Under edge reversal, both terms change sign, so \(y_{ij}^{(e)}\) changes sign and its square is invariant. This one factor generates reciprocal charge–charge, charge–dipole, and dipole–charge response.

Finally,

$$
\boxed{
\|B_\theta\eta\|^2
=
\sum_i(y_i^{(0)})^2
+\sum_{i,k}\|\mathbf y_{ik}^{(1)}\|^2
+\sum_{(i,j)}(y_{ij}^{(e)})^2.
}
$$

## B.6 Minimality and rank

The charge-constrained scalar source space has dimension

$$
2N-1.
$$

The proposed factor supplies:

* \(N\) atomic radial-null rows;
* \(N-1\) independent edge-incidence rows on a connected graph.

Thus it can span all \(2N-1\) scalar response modes.

The vector source space has dimension

$$
6N.
$$

Two three-component vector rows per atom supply exactly \(6N\) latent dimensions.

For a spanning tree,

$$
M=N+(N-1)+6N=8N-1,
$$

which is exactly the reduced source dimension. With additional local edges,

$$
M=7N+|E|,
$$

and the factor is overcomplete but still \(O(N)\).

This is a genuinely minimum variable-\(N\) full-rank topology; it is not a dense \(O(N^2)\) matrix output.

## B.7 Global PSD and reciprocity proof

For any reduced field \(\xi\),

$$
\xi^{\mathsf T}K_\theta\xi
=
\xi^{\mathsf T}C_\theta^{\mathsf T}C_\theta\xi
=
\|C_\theta\xi\|^2
\geq0.
$$

Therefore

$$
K_\theta\succeq0.
$$

The response Jacobian is

$$
J_s
=
\frac{\partial s}{\partial\eta}
=
-U_NK_\theta U_N^{\mathsf T},
$$

so

$$
J_s=J_s^{\mathsf T},
\qquad
J_s\preceq0.
$$

No numerical symmetrization, eigenvalue clipping, or reciprocity penalty is needed.

## B.8 Why this does not become QEq

There is no learned electronegativity vector followed by a global hardness solve. More importantly, the model uses

$$
K=B^{\mathsf T}B,
$$

not

$$
K=H^{-1}
$$

for a local graph hardness \(H\).

For a uniform field \(\mathbf F\):

* the two scalar radial channels at one atom see the same linear potential, so \(n_0^{\mathsf T}\phi_i=0\);
* edge scalar rows depend on potential differences and therefore scale as
  \(\mathbf F\cdot(\mathbf R_i-\mathbf R_j)\), bounded by the cutoff;
* vector rows see \(O(\|\mathbf F\|)\) local electric fields.

With bounded degree and bounded factor coefficients,

$$
\|B\eta_{\mathbf F}\|^2
\leq C(N+|E|)\|\mathbf F\|^2
=
O(N)\|\mathbf F\|^2.
$$

Hence molecular polarizability is \(O(N)\), not \(O(N^3)\).

At dissociation, \(\chi(r_{ij})\rightarrow0\), so interfragment edge rows vanish smoothly. The factor becomes a direct sum over fragments. Each fragment has zero induced net charge because its remaining scalar rows consist of radial-null modes and internal edge differences. There is therefore no asymptotic interfragment charge equalization.

The permanent fragment charges approach whatever the local frozen-MDP-plus-local-correction model assigns; they do not continue equalizing with distance. Guaranteeing integer charges for arbitrary charged dissociation channels would require explicit fragment charge-state conditioning and is outside the current neutral-singlet training domain.

---

# C. Scalar-only training loss

## C.1 Use observable projections, never fitted source coefficients

Let \(A_n^{\rm fit}(R)\in\mathbb R^{P_n\times8N}\) be the exact analytic source-to-MEP matrix on the dense fit probes, and let

$$
D_n(R)\in\mathbb R^{3\times8N}
$$

be the exact source-to-molecular-dipole map.

For point-charge mode \(k\), let \(a_{nk}\in\mathbb R^{8N}\) be the exact external-field covector. Exact adjointness means

$$
\eta_{nk}(q)=q\,a_{nk},
$$

and

$$
\frac{dE_\theta}{dq}
=
a_{nk}^{\mathsf T}s_\theta
=
V_\theta(\mathbf s_{nk}).
$$

This is why the full point-charge–nuclear work must remain inside the same scalar. If the learned source represents only the electronic part, the analytic nuclear source and work must be added before taking the derivative. Electronic MEP must never be compared against the full enthalpy slope.

## C.2 Derived QM targets

For \(q_0=10^{-3}e\), construct only the reference and first-order pieces:

$$
V_n^{(0)}=V_n(0),
$$

$$
V_{nk}^{(1)}
=
\frac{V_{nk}(+q_0)-V_{nk}(-q_0)}{2q_0},
$$

$$
\mu_n^{(0)}=\mu_n(0),
$$

$$
\mu_{nk}^{(1)}
=
\frac{\mu_{nk}(+q_0)-\mu_{nk}(-q_0)}{2q_0},
$$

and

$$
g_{nk}^{E}
=
\frac{H_{nk}(+q_0)-H_{nk}(-q_0)}{2q_0}.
$$

Do **not** train against

$$
\frac{H(+q_0)+H(-q_0)-2H(0)}{q_0^2}.
$$

The energy curvature is already implied by the learned MEP response through the common scalar. Using the noisy finite-difference curvature as a separate target would double-count response and inject the least stable observable.

The zero-field absolute energy is used only to form centered differences. It cannot train \(m_0\) or \(C\), because the head contributes zero at \(\eta=0\).

## C.3 Model predictions

At zero field,

$$
s_n^{(0)}=s_Q+U_Nm_0.
$$

For mode \(k\),

$$
s_{nk}^{(1)}
=
-U_NK_\theta U_N^{\mathsf T}a_{nk}.
$$

Then

$$
\widehat V_n^{(0)}
=
A_n^{\rm fit}s_n^{(0)},
$$

$$
\widehat V_{nk}^{(1)}
=
A_n^{\rm fit}s_{nk}^{(1)},
$$

$$
\widehat\mu_n^{(0)}
=
D_ns_n^{(0)},
$$

$$
\widehat\mu_{nk}^{(1)}
=
D_ns_{nk}^{(1)},
$$

and

$$
\widehat g_{nk}^{E}
=
a_{nk}^{\mathsf T}s_n^{(0)}.
$$

In implementation, compute \(s=\nabla_\eta E\) and the response JVP from the scalar graph rather than coding a separate response head.

## C.4 Frozen nondimensionalization

For each MEP frame, use frozen probe quadrature weights

$$
w_{np}\geq0,\qquad \sum_pw_{np}=1.
$$

Define

$$
\|x\|_{W_n}^2=\sum_pw_{np}x_p^2.
$$

Compute five train-only scales once and write them into the model manifest:

$$
S_b
=
\max\left[
\operatorname{median}_{\text{train groups}}
{\rm RMS}(b),
\ 10\sigma_b^{\rm numerical}
\right],
$$

for

$$
b\in\{V^{(0)},V^{(1)},\mu^{(0)},\mu^{(1)},g^E\}.
$$

The numerical floors must come from duplicated or tighter-convergence QM calculations, not from validation performance.

For nonzero total charge, subtract only the known fixed-charge source contribution when defining the scale:

$$
V_{\rm residual}^{(0)}
=
V^{(0)}-A_ns_Q.
$$

The loss still compares the full physical MEP.

## C.5 Fixed loss

Give every molecule equal weight and every mode within a molecule equal weight. Probes are not independent samples.

$$
\begin{aligned}
\mathcal L_{\rm data}
={}&
0.30\,\mathcal L_{V0}
+0.45\,\mathcal L_{V1}
+0.05\,\mathcal L_{\mu0}
\\
&+0.10\,\mathcal L_{\mu1}
+0.10\,\mathcal L_E,
\end{aligned}
$$

where, for example,

$$
\mathcal L_{V0}
=
\frac1{N_{\rm mol}}
\sum_n
\frac{
\|\widehat V_n^{(0)}-V_n^{(0)}\|_{W_n}^2
}{
S_{V0}^2
},
$$

and

$$
\mathcal L_{V1}
=
\frac1{4N_{\rm mol}}
\sum_{n,k}
\frac{
\|\widehat V_{nk}^{(1)}-V_{nk}^{(1)}\|_{W_n}^2
}{
S_{V1}^2
}.
$$

Use

$$
\mathcal L_{\rm reg}
=
10^{-4}\|\theta\|_2^2
+
10^{-6}\sum_{\text{optional blocks}}\|\theta_{\rm block}\|_2
+
10^{-3}\mathcal L_{\rm local\ operator\ cap}.
$$

The operator cap is a smooth penalty on atom/edge factor norms above a fixed train-manifest threshold. It is not a solvent-dependent rescaling and is not applied after prediction.

There is:

* no coefficient loss;
* no MDP atomic-charge loss;
* no original MACE-POLAR source loss;
* no response-matrix loss;
* no PCM or cavity loss;
* no experimental solvation loss.

The dense fit frame is used in \(\mathcal L_{\rm data}\). The independently rotated audit frame is never entered into a gradient.

## C.6 Higher-order data sufficiency gate

Although energy curvature is not a target, quadratic-response adequacy must be checked from MEP and dipole:

$$
\epsilon_{\rm nl}^{V}
=
\frac{
\left\|
\frac12[V(+q_0)+V(-q_0)]-V(0)
\right\|_W
}{
\frac12\|V(+q_0)-V(-q_0)\|_W+\tau
},
$$

with an analogous \(\epsilon_{\rm nl}^{\mu}\).

Require

$$
\operatorname{mean}\epsilon_{\rm nl}^{V,\mu}\leq0.02,
\qquad
\max\epsilon_{\rm nl}^{V,\mu}\leq0.05.
$$

Failure means the current one-amplitude dataset does not justify the direct quadratic model. The remedy is additional independent QM amplitudes, not a nonlinear network trained against one finite difference.

---

# D. Capacity, regularization, and ablations

## D.1 Concrete v1 capacity

The default v1 uses:

* no trainable backbone;
* no new MACE interaction layer;
* a 4-dimensional shared element embedding;
* one atom invariant MLP with hidden width 16;
* one edge MLP with hidden width 8;
* two vector factor channels per atom;
* one scalar factor channel per local edge;
* at most three shared radial-null mixing parameters.

The hard cap is

$$
\boxed{P_{\rm trainable}\leq2500}
$$

for the MDP-only candidate.

A MACE-POLAR-feature candidate may increase the cap to

$$
\boxed{P_{\rm trainable}\leq4096},
$$

but only by adding a small multiplicity-space compression of frozen zero-field features. It may not add another message-passing layer.

For POLAR features, retain at most:

$$
8\times 0e,\qquad 4\times1o
$$

compressed channels. A fixed train-only PCA within irrep multiplicity space is preferable to a large trainable projection; the same channel transformation is applied to all \(m\) components of an irrep.

All frozen backbone parameters have `requires_grad=False`, but their coordinate computations must remain attached to \(R\).

## D.2 Prespecified ablation sequence

| ID | Permanent source                                | Response                                      | Descriptors                    |
| -- | ----------------------------------------------- | --------------------------------------------- | ------------------------------ |
| A0 | Fixed MDP q/p radial lift                       | Fixed passive MDP polarizability lift         | MDP                            |
| A1 | Learned observable-only \(m_0\) correction      | Fixed response                                | MDP                            |
| A2 | Learned \(m_0\)                                 | Atomic radial + two atomic vector factor rows | MDP                            |
| A3 | A2 plus local edge charge-transfer/cross factor | Full proposed sparse \(B\)                    | MDP                            |
| A4 | Same topology as A3                             | Same topology as A3                           | MDP + zero-field POLAR latents |

A0 is a valid passive baseline only if its polarizability factorization passes PSD. An indefinite atomwise tensor is not repaired by spectral clipping.

The **implementation target is A3**. It is the smallest topology that spans all \(8N-1\) response dimensions while preserving fragment locality.

A2 may be released instead only if it passes every gate. A4 is considered only if A3 fails a predictive gate or if it produces a predeclared statistically decisive improvement.

## D.3 Selection protocol

1. Hyperparameters and epoch count are selected by formula-group cross-validation within the 32 training molecules.
2. Retrain each prespecified ablation on all 32 training molecules.
3. Evaluate each candidate once on the 14 validation formulas and on rotated audit probes.
4. Select the smallest all-gates-passing model.
5. Open the 14 blind formulas once.

The ranking score may be

$$
M_{\rm select}
=
0.55\epsilon_{V1}
+0.25\epsilon_{V0}
+0.10\epsilon_{\mu1}
+0.10\epsilon_E,
$$

but it is used only among candidates that pass every individual gate. A weighted score may not conceal failure of response MEP, charge, passivity, or rotations.

## D.4 Stop rule

An added block is justified only when all of the following hold:

$$
\frac{M_{\rm old}-M_{\rm new}}{M_{\rm old}}\geq0.10,
$$

the paired formula-bootstrap 95% lower confidence bound on improvement is positive, improvement occurs on both grouped train-CV and validation audit, and no individual gate worsens by more than 10%.

Therefore:

* If A2 passes all gates, stop at A2.
* If A3 passes and A4 improves only the fit partition or only the aggregate score, stop at A3.
* A4 cannot be retained merely because it lowers development-set loss.
* No backbone unfreezing is permitted for this dataset size.

---

# E. Reciprocal ddPCM coupling and force path

## E.1 Reaction operator

At fixed geometry, let reciprocal ddPCM induce a linear reaction-potential operator

$$
\eta_{\rm rf}=\mathcal R(R)s.
$$

After whitening in the exact PCM reciprocity metric,

$$
\mathcal R=\mathcal R^{\mathsf T}\preceq0.
$$

Define the reduced operator

$$
R_r=U_N^{\mathsf T}\mathcal R U_N,
\qquad
P_r=-R_r\succeq0.
$$

The self-consistent source is

$$
s^*
=
s_Q+U_N\left[m_0-K\xi^*\right],
$$

with

$$
\xi^*=U_N^{\mathsf T}\mathcal R s^*.
$$

Equivalently,

$$
\left(I-P_rK\right)\xi^*
=
U_N^{\mathsf T}\mathcal R(s_Q+U_Nm_0).
$$

## E.2 Structural passivity is not combined stability

The source head guarantees only

$$
K=C^{\mathsf T}C\succeq0.
$$

That does **not** guarantee a unique coupled state. Continuum feedback can still produce a polarization catastrophe.

A sufficient and, for the linear reduced problem, essentially sharp condition is

$$
\boxed{
\lambda_{\max}
\left(
P_r^{1/2}KP_r^{1/2}
\right)<1.
}
$$

Using the factor,

$$
\lambda_{\max}
\left(P_r^{1/2}C^{\mathsf T}CP_r^{1/2}\right)
=
\lambda_{\max}\left(CP_rC^{\mathsf T}\right).
$$

The production gate should require a margin:

$$
\boxed{
\lambda_{\max}(CP_rC^{\mathsf T})\leq0.90.
}
$$

In source-functional language, when \(K\) is invertible this is equivalent to

$$
K^{-1}-P_r\succ0.
$$

So a strongly convex isolated-solute functional would not, by itself, remove the need for the combined condition.

## E.3 Total coupled energy

With the present sign convention, once the reciprocal root is converged,

$$
\boxed{
G_{\rm coupled}
=
E_\theta(R,\eta_{\rm rf}^*)
-\frac12{s^*}^{\mathsf T}\eta_{\rm rf}^*
+G_{\rm CDS}(R).
}
$$

The subtraction removes the double counting of solute–reaction-field work. It is the field-space equivalent of internal polarization cost plus one-half reaction interaction.

## E.4 First force path

The first production force implementation should use implicit differentiation through the converged coupled root, not a detached source and not finite differences.

Let

$$
F(R,y)=0
$$

be the coupled residual in the native ddPCM surface variables \(y\). Solve the adjoint equation

$$
F_y^{\mathsf T}\lambda
=
\frac{\partial G_{\rm coupled}}{\partial y},
$$

then

$$
\boxed{
\frac{dG_{\rm coupled}}{dR}
=
\frac{\partial G_{\rm coupled}}{\partial R}
-
\lambda^{\mathsf T}\frac{\partial F}{\partial R}.
}
$$

At exact stationarity, this must agree with a joint stationary-Lagrangian/envelope implementation. That agreement is a force audit, not an excuse to detach the root.

The following coordinate dependencies must stay in the same autograd graph:

* frozen MACE-MDP \(q_i,\mathbf p_i,\alpha_i,E_{\rm vac}\);
* optional frozen MACE-POLAR zero-field node features;
* atom and edge head coefficients;
* radial mixing and all pair cutoff functions;
* analytic GTO basis centers and source/receiver kernels;
* \(m_0(R)\), \(B_\theta(R)\), and the mixed derivative
  \(\partial^2E/\partial R\,\partial\eta\);
* ddPCM cavity points, switching functions, surface partitions, matrices, and solves;
* the reaction-potential map \(\mathcal R(R)s\);
* CDS/SASA geometry terms.

Frozen weights are not detached coordinates.

---

# F. Fail-closed gates

Every threshold below is evaluated in float64. Any failure is `STOP`; no failed class may be averaged away by another class.

## F.1 Data and model-identity gates

| Gate                    | Required condition                                                                           |                                |      |   |               |
| ----------------------- | -------------------------------------------------------------------------------------------- | ------------------------------ | ---- | - | ------------- |
| QM enthalpy/MEP closure | Every mode: (\left                                                                           | g^E-V^{(0)}_{\rm source}\right | /(1+ | V | )\leq10^{-5}) |
| Higher-order MEP        | Mean \(\epsilon_{\rm nl}^{V}\leq2\%\), maximum \(\leq5\%\)                                   |                                |      |   |               |
| Higher-order dipole     | Mean \(\epsilon_{\rm nl}^{\mu}\leq2\%\), maximum \(\leq5\%\)                                 |                                |      |   |               |
| Probe adequacy          | Each fit and audit partition remains \(>8N\); no probe deletion after seeing errors          |                                |      |   |               |
| Audit isolation         | Rotated audit probes never appear in gradient calculations                                   |                                |      |   |               |
| Blind isolation         | The 14 blind formulas remain unopened until architecture, weights, and thresholds are sealed |                                |      |   |               |

The observed \(1.91\times10^{-7}\) enthalpy-slope closure comfortably satisfies the proposed data threshold.

## F.2 Exact structural gates

Let

$$
\epsilon_{\rm rel}(x,y)=\frac{\|x-y\|}{1+\|y\|}.
$$

| Gate                             | Required condition                                                                                       |                                               |                                                  |                  |                       |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------ | ---------------- | --------------------- |
| Charge                           | (                                                                                                        | g_N^{\mathsf T}s-Q                            | \leq10^{-12}e) for zero and all perturbed states |                  |                       |
| Architecture charge construction | Before final projection, charge residual already \(\leq10^{-13}e\); otherwise projection is hiding a bug |                                               |                                                  |                  |                       |
| Energy/source conjugacy          | \(\epsilon_{\rm rel}(\nabla_\eta E,s_{\rm analytic})\leq5\times10^{-11}\)                                |                                               |                                                  |                  |                       |
| Source-point MEP conjugacy       | (                                                                                                        | dE/dq-a_k^{\mathsf T}s                        | /(1+                                             | a_k^{\mathsf T}s | )\leq5\times10^{-11}) |
| Adjointness                      | (                                                                                                        | u^{\mathsf T}Av-(A^{\mathsf T}u)^{\mathsf T}v | /(1+\cdots)\leq5\times10^{-11})                  |                  |                       |
| Reciprocity                      | \(\|J_s-J_s^{\mathsf T}\|_F/(\|J_s\|_F+10^{-30})\leq10^{-10}\)                                           |                                               |                                                  |                  |                       |
| Passivity                        | \(\lambda_{\min}(K)\geq-10^{-12}\max(1,\lambda_{\max}K)\)                                                |                                               |                                                  |                  |                       |
| Directional passivity            | For 256 deterministic directions \(v\), \(v^{\mathsf T}Kv\geq-10^{-12}\|v\|^2\)                          |                                               |                                                  |                  |                       |
| Smooth cutoff                    | Energy, source, and first coordinate derivative go continuously to zero at every edge cutoff             |                                               |                                                  |                  |                       |

## F.3 Transferable electrostatic accuracy gates

Use formula-weighted relative RMS, not pointwise percentage error.

### Validation rotated-audit gate

$$
\begin{array}{ll}
\text{reference MEP mean} & \leq 5\%,\\
\text{reference MEP worst formula} & \leq 12\%,\\
\text{response MEP mean} & \leq 8\%,\\
\text{response MEP worst formula} & \leq 20\%,\\
\text{source-point response mean} & \leq 5\%,\\
\text{zero-field dipole error} & \leq 3\%,\\
\text{dipole-response error} & \leq 8\%,\\
\text{enthalpy-slope error} & \leq 2\%.
\end{array}
$$

### Blind 14-formula gate

$$
\begin{array}{ll}
\text{reference MEP mean} & \leq 6\%,\\
\text{reference MEP worst formula} & \leq 12\%,\\
\text{response MEP mean} & \leq 10\%,\\
\text{response MEP worst formula} & \leq 20\%.
\end{array}
$$

The response gate must also represent at least a fourfold reduction from the frozen original approximately 46% mean response error. A class-specific failure cannot pass because the overall average improves.

## F.4 Dense fit-to-audit generalization

For both reference and response MEP,

$$
\frac{\epsilon_{\rm audit}}{\epsilon_{\rm fit}}\leq1.5,
$$

and

$$
\epsilon_{\rm audit}-\epsilon_{\rm fit}
\leq2\text{ percentage points}.
$$

No individual molecule may have an audit/fit ratio above 2.0.

This gate is important because the unconstrained per-molecule source fits already demonstrate that the physical span is adequate. The neural head must generalize across probes rather than recreate a probe-frame interpolation.

## F.5 Rotation and reflection gates

For at least 24 deterministic random rotations plus one inversion/reflection:

### Head-only analytic source model

$$
|E(QR,Q\eta)-E(R,\eta)|\leq10^{-9}\ {\rm eV},
$$

$$
\frac{\|s(QR,Q\eta)-D(Q)s(R,\eta)\|}
{1+\|s\|}
\leq10^{-8},
$$

$$
\frac{\|\mu(QR)-Q\mu(R)\|}
{1+\|\mu\|}
\leq10^{-8},
$$

and force covariance error must be \(\leq10^{-7}\) relative.

### Coupled discretized ddPCM

The continuum has a separate numerical budget:

$$
|\Delta G_{\rm rotation}|\leq10^{-5}\ {\rm eV},
$$

$$
\epsilon_{\rm source/MEP\ covariance}\leq2\times10^{-4},
$$

$$
\epsilon_{\rm force\ covariance}\leq5\times10^{-4}.
$$

A head-only pass cannot mask a ddPCM rotation failure.

## F.6 Coupled-root gates

For every train, validation, blind-electrostatic, and unopened continuum-transfer geometry:

| Gate                     | Required condition                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| Combined spectral margin | \(\lambda_{\max}(CP_rC^{\mathsf T})\leq0.90\)                                                       |
| Root residual            | \(\|F\|_\infty/(1+\|y\|_\infty)\leq10^{-10}\)                                                       |
| Multi-start              | At least 8 starts: zero, MDP prior, continuation, and deterministic positive/negative random starts |
| State agreement          | All starts agree in source to \(10^{-8}\) relative                                                  |
| Energy agreement         | All starts agree to \(10^{-10}\,E_h\)                                                               |
| Adjoint solve            | Relative residual \(\leq10^{-10}\)                                                                  |
| Force check              | Central coordinate difference agrees with implicit force within the frozen force tolerance          |

No damping-dependent alternative root is accepted.

## F.7 Continuum-transfer and final solvation gates

After the electrostatic model, architecture, code, and weights are sealed:

1. **Unopened matched-QM continuum transfer**

   Retain the stringent source/continuum gate:

   $$
   \text{mean certified reaction-energy error bound}
   \leq0.25\ {\rm kcal/mol},
   $$

   $$
   \max\text{ certified bound}
   \leq0.50\ {\rm kcal/mol}.
   $$

   No cavity quantity or PCM energy from this panel may feed back into training.

2. **Final untouched full-solvation confirmation**

   Experimental identities and labels may be opened only for this terminal evaluation. The frozen project gate remains

   $$
   \boxed{\mathrm{MAE}\leq1.50\ {\rm kcal/mol}},
   $$

   together with

   $$
   |\mathrm{bias}|\leq0.50\ {\rm kcal/mol},
   $$

   and

   $$
   \text{one-sided 95\% geometry-bootstrap MAE UCB}
   \leq2.00\ {\rm kcal/mol}.
   $$

   RMSE, \(q_{95}\), and maximum error are reported but not retrospectively thresholded.

Failure of the terminal experimental gate does not authorize tuning the electrostatic head, radial mixing, cavity, or CDS against those labels. Any subsequent repair requires new independent QM-only data, a new model identity, and a newly sealed blind evaluation.

---

# Compact PyTorch-style skeleton

```python
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GeometryBatch:
    atomic_numbers: Tensor       # [N]
    positions: Tensor            # [N, 3], float64
    edge_index: Tensor           # [2, E]
    total_charge: float


class PassiveElectrostaticHead(nn.Module):
    """
    Geometry-conditioned scalar energy.
    Frozen backbones remain differentiable with respect to positions.
    """

    def __init__(self, frozen_mdp: nn.Module, atom_head: nn.Module,
                 edge_head: nn.Module) -> None:
        super().__init__()
        self.frozen_mdp = frozen_mdp
        for parameter in self.frozen_mdp.parameters():
            parameter.requires_grad_(False)

        self.atom_head = atom_head
        self.edge_head = edge_head

        # Shared radial-null mixing only; no molecule-specific parameters.
        self.beta_q = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.beta_p = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def permanent_source(
        self,
        batch: GeometryBatch,
        mdp_outputs: dict[str, Tensor],
        radial_maps: dict[str, Tensor],
    ) -> Tensor:
        """
        Returns full source s0: [N, 8].
        Charge conservation is architectural, not repaired by mean subtraction.
        """
        q_mdp = mdp_outputs["q"]          # [N]
        p_mdp = mdp_outputs["p"]          # [N, 3]
        alpha = mdp_outputs["alpha"]      # [N, 3, 3]
        scalar_desc = mdp_outputs["scalar_descriptors"]

        atom_out = self.atom_head(scalar_desc)
        radial_scalar = atom_out["radial_scalar"]      # [N]
        vector_coeffs = atom_out["vector_coeffs"]

        edge_out = self.edge_head(
            scalar_desc,
            batch.positions,
            batch.edge_index,
        )
        t_directed = edge_out["antisymmetric_transfer"]  # [E directed]

        # Scatter directed antisymmetric transfers to atoms.
        delta_atomic_q = antisymmetric_edge_scatter(
            t_directed, batch.edge_index, q_mdp.shape[0]
        )  # [N], sums exactly to zero

        u_vec, v_vec = equivariant_vector_readout(
            p_mdp=p_mdp,
            alpha=alpha,
            positions=batch.positions,
            edge_index=batch.edge_index,
            coefficients=vector_coeffs,
        )  # each [N, 3]

        w0, n0 = radial_maps["w0"], radial_maps["n0"]   # [2]
        w1, n1 = radial_maps["w1"], radial_maps["n1"]   # [2]

        q_rad = (
            q_mdp[:, None] * (w0 + self.beta_q * n0)[None, :]
            + radial_scalar[:, None] * n0[None, :]
            + delta_atomic_q[:, None] * w0[None, :]
        )  # [N, 2]

        p_rad = (
            p_mdp[:, None, :] * (w1 + self.beta_p * n1)[None, :, None]
            + u_vec[:, None, :] * w1[None, :, None]
            + v_vec[:, None, :] * n1[None, :, None]
        )  # [N, 2, 3]

        return pack_source(q_rad, p_rad)  # [N, 8]

    def factor_apply(
        self,
        batch: GeometryBatch,
        field_full: Tensor,          # [N, 8]
        mdp_outputs: dict[str, Tensor],
        radial_maps: dict[str, Tensor],
    ) -> Tensor:
        """
        Applies B(R) to a full physical field.
        Returns concatenated scalar/vector latent rows.
        Never constructs a dense 8N x 8N matrix.
        """
        phi, efield = unpack_field(field_full)
        # phi: [N, 2], efield: [N, 2, 3]

        alpha = mdp_outputs["alpha"]
        scalar_desc = mdp_outputs["scalar_descriptors"]
        atom_coeff = self.atom_head(scalar_desc)["factor_coefficients"]

        w0, n0 = radial_maps["w0"], radial_maps["n0"]
        w1 = radial_maps["w1"]

        # N scalar radial-null rows.
        y_radial = (
            atom_coeff["radial_gain"]
            * torch.einsum("na,a->n", phi, n0)
        )  # [N]

        # Two vector rows per atom: [N, 2, 3].
        y_vector = vector_factor_apply(
            efield=efield,
            alpha=alpha,
            coefficients=atom_coeff["vector_factor"],
        )

        # Local pair rows.
        edge_coeff = self.edge_head(
            scalar_desc, batch.positions, batch.edge_index
        )
        y_edge = pair_charge_dipole_factor(
            phi=phi,
            efield=efield,
            positions=batch.positions,
            edge_index=batch.edge_index,
            coefficients=edge_coeff,
            w0=w0,
            w1=w1,
        )  # [E]

        return torch.cat([
            y_radial.reshape(-1),
            y_vector.reshape(-1),
            y_edge.reshape(-1),
        ])

    def energy(
        self,
        batch: GeometryBatch,
        eta_full: Tensor,             # [N, 8], requires_grad=True
        chart: dict[str, Tensor],
        radial_maps: dict[str, Tensor],
    ) -> Tensor:
        mdp = self.frozen_mdp(
            batch.atomic_numbers,
            batch.positions,
        )
        e_vac = mdp["energy"]

        s0_full = self.permanent_source(batch, mdp, radial_maps).reshape(-1)

        U = chart["U"]                 # [8N, 8N-1]
        sQ = chart["sQ"]               # [8N]
        eta = eta_full.reshape(-1)
        xi = U.T @ eta

        # Convert full physical source to the authoritative reduced chart.
        m0 = U.T @ (s0_full - sQ)

        # B acts on a gauge-fixed full field; C = B U.
        eta_reduced_full = (U @ xi).reshape_as(eta_full)
        y = self.factor_apply(batch, eta_reduced_full, mdp, radial_maps)

        return (
            e_vac
            + torch.dot(sQ, eta)
            + torch.dot(m0, xi)
            - 0.5 * torch.dot(y, y)
        )

    def source(
        self,
        batch: GeometryBatch,
        eta_full: Tensor,
        chart: dict[str, Tensor],
        radial_maps: dict[str, Tensor],
    ) -> Tensor:
        eta = eta_full.requires_grad_(True)
        energy = self.energy(batch, eta, chart, radial_maps)
        source, = torch.autograd.grad(
            energy, eta, create_graph=True
        )
        return source


def observable_predictions(
    head: PassiveElectrostaticHead,
    batch: GeometryBatch,
    mode_field: Tensor,               # [N, 8]
    source_to_mep: Tensor,             # [P, 8N]
    source_to_dipole: Tensor,          # [3, 8N]
    chart: dict[str, Tensor],
    radial_maps: dict[str, Tensor],
) -> dict[str, Tensor]:
    zero_field = torch.zeros_like(mode_field, requires_grad=True)

    def source_fn(field: Tensor) -> Tensor:
        return head.source(batch, field, chart, radial_maps).reshape(-1)

    s0 = source_fn(zero_field)

    # Exact first-order response from the scalar graph.
    _, ds_dq = torch.func.jvp(
        source_fn,
        (zero_field,),
        (mode_field,),
    )

    return {
        "mep0": source_to_mep @ s0,
        "mep1": source_to_mep @ ds_dq,
        "dipole0": source_to_dipole @ s0,
        "dipole1": source_to_dipole @ ds_dq,
        "enthalpy_slope": torch.dot(mode_field.reshape(-1), s0),
    }
```

For PCM, solve in native surface variables rather than constructing a dense reaction matrix:

```python
def coupled_residual(surface_state: Tensor) -> Tensor:
    eta_rf = pcm.receiver(geometry, surface_state)        # [N, 8]
    source = head.source(geometry, eta_rf, chart, radial_maps)
    return pcm.boundary_residual(geometry, surface_state, source)


surface_star = newton_krylov_multistart(coupled_residual, starts)
eta_star = pcm.receiver(geometry, surface_star)
source_star = head.source(geometry, eta_star, chart, radial_maps)

g_total = (
    head.energy(geometry, eta_star, chart, radial_maps)
    - 0.5 * torch.dot(source_star.reshape(-1), eta_star.reshape(-1))
    + cds_energy(geometry)
)

forces = -implicit_root_gradient(
    scalar=g_total,
    residual=coupled_residual,
    root=surface_star,
    coordinates=geometry.positions,
)
```

---

# Ranked implementation plan

1. **Freeze the mathematical interfaces.** Lock GTO normalizations, source–field pairing, moment rows \(M_0,M_1\), Householder construction, source-to-MEP matrices, dipole matrices, and point-charge receiver covectors. Unit-test all adjoints before training.

2. **Implement the full-space permanent source.** Start from the MDP q/p lift, radial-null atom corrections, antisymmetric local edge transfer, and equivariant vector corrections. Require charge closure before applying the Householder chart.

3. **Implement the sparse factor \(B_\theta\).** Use one atomic scalar row, two atomic vector rows, and one local edge row. Never materialize or predict a dense \(C\) or \(K\).

4. **Close scalar differentiation.** Verify source, JVP, VJP, reciprocity, passivity, charge, and point-charge enthalpy/MEP conjugacy in float64 before any fit.

5. **Run A0–A3 with MDP-only descriptors.** Release the smallest all-gates-passing candidate. A3 is the expected minimum for nonuniform response.

6. **Run A4 only under the stop rule.** Use only zero-field POLAR node features. Do not execute or distill the original POLAR source or response.

7. **Seal the electrostatic head before PCM transfer.** Hash architecture, weights, train scales, cutoffs, thresholds, and software environment.

8. **Couple reciprocal ddPCM.** Enforce the spectral margin and multi-start root gates before force work.

9. **Close implicit forces and rotations.** Do not open optimization, dynamics, or full-solvation claims before coupled force and rotation gates pass.

10. **Open blind electrostatics, then continuum transfer, then experimental totals.** No result from a later stage may modify an earlier sealed stage.

---

# Shortcuts that must be rejected

**Per-molecule coefficient supervision:** reject. The dense fits establish representational sufficiency, but fitted coefficients are probe-frame-dependent latent choices, not physical labels.

**Original MACE-POLAR response distillation:** reject. It would reproduce the demonstrated failure and preserve the absence of a common scalar.

**Dense \(C(R)\in\mathbb R^{(8N-1)\times(8N-1)}\) output:** reject. It is not a transferable variable-\(N\) head and has \(O(N^2)\) outputs.

**Ordinary global QEq or a local hardness matrix followed by inversion:** reject. This is mathematically opposite to the desired local susceptibility construction and reopens dissociation and size-scaling risks. ([arXiv][2])

**Post-hoc matrix symmetrization:** reject. Symmetry does not imply PSD.

**Eigenvalue clipping of \(K\), \(H\), or atomic polarizabilities:** reject. It creates nonsmooth coordinate derivatives at eigenvalue crossings and can hide an invalid architecture.

**Global mean-charge subtraction as the primary conservation mechanism:** reject. It couples disconnected fragments. Charge must be conserved by radial-null and incidence constructions; Householder projection is a numerical representation, not a physical repair.

**Separate MEP, dipole, source, and energy heads:** reject. They recreate the original nonvariational problem.

**Numerical energy-curvature training:** reject. The response is already constrained through derivative MEP and dipole data.

**Stopping gradients through frozen backbones or PCM geometry:** reject. Frozen parameters do not mean frozen coordinate derivatives.

**Solvent-dependent rescaling of \(C\) to force convergence:** reject. The solute head must remain solvent independent; a failed combined spectral gate is a model failure.

**Adding POLAR latents solely because validation loss decreases:** reject. They are retained only when the prespecified paired improvement and all individual gates pass.

The direct sparse quadratic head is therefore not merely the simpler implementation. For this dataset and retained 8N physical space, it is the mathematically identified model class, the minimum full-rank variable-\(N\) construction, and the safer alternative to an inverse-hardness/QEq architecture.

[1]: https://arxiv.org/abs/2603.14700 "https://arxiv.org/abs/2603.14700"
[2]: https://arxiv.org/html/2603.14700v2 "https://arxiv.org/html/2603.14700v2"
