# Decision

**Yes.** The minimum scientifically justified v1 is now a **split-order direct-sum scalar**:

$$
\boxed{
\text{permanent source space}
=
\text{radial8}\oplus \text{point-}l=2,
\qquad
\text{induced-response space}
=
\text{radial8 only}.
}
$$

Equivalently:

$$
\boxed{
\text{P13/R8 model: permanent }13N\text{ channels, dynamic }8N-1\text{ channels}.
}
$$

The five atom-centred traceless quadrupole components are geometry-dependent but **field-rigid** in v1. The susceptibility has zero \(l=2\) rows, zero \(l=2\) columns, and therefore zero radial–quadrupole cross-response.

This is not a relaxation or patch to the failed radial8 registration. The radial8-complete model is closed. The repaired model needs a new architecture identity and new sealed manifest.

The evidence supports precisely this asymmetry:

* radial8 fails the reference-density gate: \(6.91\%>5\%\);
* adding permanent \(l=2\) reduces the independent audit error to \(1.64\%\);
* radial8 already passes the first-order-response gate: \(1.27\%<3\%\);
* adding induced \(l=2\) changes response only from \(1.27\%\) to \(1.15\%\), without crossing any gate.

The result establishes the **necessity of permanent \(l=2\)** and the **lack of present justification for induced \(l=2\)**. It does not yet establish that a transferable geometry head can predict the quadrupoles; that remains a model-validation question.

---

# 1. Permanent and induced source spaces

Let \(V_l^\pi\) denote the real \(E(3)\) irrep of angular order \(l\) and parity \(\pi\).

For each atom, the two-width radial-GTO \(l\leq1\) source space is

$$
\mathcal S_{r,i}
=
\left(\mathbb R^2\otimes V_0^+\right)
\oplus
\left(\mathbb R^2\otimes V_1^-\right),
$$

with dimension

$$
2+2(3)=8.
$$

Thus

$$
\mathcal S_{r,N}
=
\bigoplus_{i=1}^N\mathcal S_{r,i}
\cong\mathbb R^{8N}.
$$

The permanent quadrupole space is

$$
\mathcal S_{2,N}
=
\bigoplus_{i=1}^N V_2^+
\cong\mathbb R^{5N}.
$$

A convenient Cartesian representation is a symmetric traceless tensor

$$
\Theta_i\in\operatorname{STF}(3),
\qquad
\Theta_i^{\mathsf T}=\Theta_i,
\qquad
\operatorname{tr}\Theta_i=0,
$$

which has five independent components.

The full source and dual field spaces are therefore

$$
\boxed{
\mathcal S_N
=
\mathcal S_{r,N}\oplus\mathcal S_{2,N},
\qquad
\mathcal F_N
=
\mathcal F_{r,N}\oplus\mathcal F_{2,N}.
}
$$

Their dimensions are both \(13N\).

The permanent and induced manifolds are different:

$$
\boxed{
\mathcal S_{\mathrm{perm}}
=
\left(s_Q+\ker g_r^{\mathsf T}\right)
\oplus \mathcal S_{2,N},
}
$$

whereas

$$
\boxed{
\mathcal S_{\mathrm{ind}}^{v1}
=
\ker g_r^{\mathsf T}\oplus\{0\}_{2}.
}
$$

So the reference source has radial \(l\leq1\) and point \(l=2\) content, but the field-induced change has radial \(l\leq1\) content only.

This does **not** claim that the true QM atomic quadrupoles do not respond. It says only that, over the preregistered exterior-point-charge and ddPCM field class, the observable first-order response potential is adequately represented in the distributed radial8 span.

Distributed atom-centred \(l\leq1\) functions can reproduce molecular response fields containing higher global angular moments. The local angular label of an atomic basis function is not the same as the angular order of the total molecular far field.

---

# 2. Exact point-quadrupole dual convention

Choose one quadrupole convention and freeze it in the model identity. A clean convention uses the raw symmetric-traceless second moment \(\Theta_i\).

For an external scalar potential \(\phi\), define the conjugate quadrupolar field

$$
\boxed{
\Gamma_i[\phi]
=
\frac12
\operatorname{STF}
\left[
\nabla\nabla\phi(R_i)
\right].
}
$$

The quadrupole–field work is

$$
\langle\Theta,\Gamma\rangle
=
\sum_{i=1}^{N}
\Theta_i:\Gamma_i.
$$

The potential generated at an exterior point \(x\) is

$$
\boxed{
\phi_2(x)
=
\frac12
\sum_i
\Theta_i:
\nabla\nabla
\frac{1}{|x-R_i|}.
}
$$

In a five-component orthonormal STF basis,

$$
\theta_i\in\mathbb R^5,
\qquad
\gamma_i\in\mathbb R^5,
\qquad
\Theta_i:\Gamma_i=\theta_i^{\mathsf T}\gamma_i.
$$

A convention using the common \(Q_{ab}=3M_{ab}-\operatorname{tr}(M)\delta_{ab}\) quadrupole instead carries a \(1/6\) rather than \(1/2\) factor. The choice is immaterial provided the source, receiver, energy coupling, and force kernels use exactly the same convention.

Let

$$
A_r(R):\mathcal S_{r,N}\rightarrow\mathbb R^P
$$

be the analytic radial8 source-to-MEP map and

$$
A_2(R):\mathcal S_{2,N}\rightarrow\mathbb R^P
$$

the point-quadrupole source-to-MEP map. Then

$$
V_{\mathrm{MEP}}
=
A_rs_r+A_2\Theta.
$$

Their receivers must be the exact metric adjoints:

$$
\eta_r=A_r^\dagger w,
\qquad
\Gamma=A_2^\dagger w.
$$

For weighted probes with metric \(W\),

$$
A_r^\dagger=A_r^{\mathsf T}W,
\qquad
A_2^\dagger=A_2^{\mathsf T}W.
$$

No separately implemented “quadrupole field head” is permissible.

---

# 3. The corrected scalar

Let \(g_r\in\mathbb R^{8N}\) be the total-charge covector in radial8 space:

$$
g_r^{\mathsf T}s_r=Q.
$$

Construct the frozen Householder chart

$$
U_N\in\mathbb R^{8N\times(8N-1)}
$$

such that

$$
U_N^{\mathsf T}U_N=I,
\qquad
U_N^{\mathsf T}g_r=0.
$$

Choose the particular fixed-charge source

$$
s_Q=\frac{Q}{g_r^{\mathsf T}g_r}g_r.
$$

Let

$$
m_0(R)\in\mathbb R^{8N-1}
$$

be the learned permanent radial reduced source,

$$
\Theta_0(R)\in\mathbb R^{5N}
$$

the learned geometry-only permanent quadrupole source, and

$$
C(R):\mathbb R^{8N-1}\rightarrow\mathbb R^{M_N}
$$

the sparse passive radial response factor.

For a radial field \(\eta_r\), define

$$
\xi=U_N^{\mathsf T}\eta_r
$$

and

$$
K(R)=C(R)^{\mathsf T}C(R)\succeq0.
$$

The v1 scalar is

$$
\boxed{
\begin{aligned}
E_\theta(R,\eta_r,\Gamma)
={}&E_{\mathrm{vac}}(R)
+s_Q^{\mathsf T}\eta_r
+m_0(R)^{\mathsf T}\xi
\\
&+\Theta_0(R)^{\mathsf T}\Gamma
-\frac12\left\|C(R)\xi\right\|^2.
\end{aligned}
}
$$

The \(l=2\) term is affine in the applied field. There is no quadratic term involving \(\Gamma\).

The full source obtained from the scalar is

$$
\boxed{
s_r
=
\nabla_{\eta_r}E_\theta
=
s_Q+U_N\left[m_0-K\xi\right],
}
$$

and

$$
\boxed{
\Theta
=
\nabla_\Gamma E_\theta
=
\Theta_0(R).
}
$$

Therefore

$$
\frac{\partial\Theta}{\partial\eta_r}
=
\frac{\partial\Theta}{\partial\Gamma}
=
0,
\qquad
\frac{\partial s_r}{\partial\Gamma}=0.
$$

The quadrupoles are field-rigid but not coordinate-rigid:

$$
\frac{\partial\Theta_0}{\partial R}\neq0
$$

in general.

---

# 4. Structural properties

## Energy/source conjugacy

Because both source sectors are obtained by differentiation of the same scalar,

$$
s_r=\frac{\partial E}{\partial\eta_r},
\qquad
\Theta=\frac{\partial E}{\partial\Gamma}.
$$

For an exterior point-charge mode at \(x_k\),

$$
\eta_r(q)=q\,a_{r,k},
\qquad
\Gamma(q)=q\,a_{2,k},
$$

where \(a_{r,k}\) and \(a_{2,k}\) are the exact radial and quadrupolar receivers of a unit point charge.

Then

$$
\boxed{
\frac{dE}{dq}
=
a_{r,k}^{\mathsf T}s_r(q)
+
a_{2,k}^{\mathsf T}\Theta_0.
}
$$

By analytic adjointness, this is exactly the total model MEP at the point-charge location, including the permanent quadrupole contribution.

At \(q=0\),

$$
\left.\frac{dE}{dq}\right|_{q=0}
=
a_{r,k}^{\mathsf T}
\left(s_Q+U_Nm_0\right)
+
a_{2,k}^{\mathsf T}\Theta_0.
$$

Thus the same central-enthalpy-slope closure test remains applicable. All point-charge–nuclear work must remain in this same affine scalar convention.

## Fixed charge

Since

$$
g_r^{\mathsf T}U_N=0,
$$

we have

$$
g_r^{\mathsf T}s_r
=
g_r^{\mathsf T}s_Q
=
Q.
$$

A traceless point quadrupole has zero monopole and zero dipole, so

$$
g^{\mathsf T}
\begin{bmatrix}
s_r\\
\Theta
\end{bmatrix}
=
Q.
$$

The quadrupole head cannot alter total charge.

## Reciprocity and passivity

Define the positive susceptibility as

$$
\chi
=
-\frac{\partial(s_r,\Theta)}
{\partial(\eta_r,\Gamma)}.
$$

For the proposed scalar,

$$
\boxed{
\chi
=
\begin{pmatrix}
U_NKU_N^{\mathsf T} & 0\\
0 & 0
\end{pmatrix}
\succeq0.
}
$$

It is symmetric, so reciprocity is exact.

The polarization energy is

$$
\Delta E_{\mathrm{pol}}
=
-\frac12\xi^{\mathsf T}K\xi
\leq0.
$$

Thus structural passivity remains exact. The response spectrum consists of the original nonnegative susceptibility eigenvalues plus \(5N\) quadrupole-sector zeros and the fixed-charge gauge zero.

Equivalently, the source Jacobian has nonpositive eigenvalues:

$$
\operatorname{spec}
\left(
\frac{\partial(s_r,\Theta)}
{\partial(\eta_r,\Gamma)}
\right)
=
\operatorname{spec}(-U_NKU_N^{\mathsf T})
\cup\{0\}^{5N}.
$$

## Why all radial–quadrupole cross-response must also be zero

Suppose one attempted

$$
\chi
=
\begin{pmatrix}
A&B\\
B^{\mathsf T}&0
\end{pmatrix}
\succeq0
$$

while claiming zero induced quadrupole susceptibility.

For arbitrary \(x,y\) and scalar \(t\),

$$
\begin{bmatrix}x\\ty\end{bmatrix}^{\mathsf T}
\chi
\begin{bmatrix}x\\ty\end{bmatrix}
=
x^{\mathsf T}Ax+2t\,x^{\mathsf T}By.
$$

This must be nonnegative for every positive and negative \(t\). Therefore

$$
x^{\mathsf T}By=0
$$

for every \(x,y\), implying

$$
B=0.
$$

So a passive model cannot have nonzero radial–quadrupole cross-susceptibility while keeping the quadrupole diagonal susceptibility exactly zero.

Therefore the proposed block-direct-sum response is not merely convenient. It is mathematically forced by the combination of:

1. exact zero induced quadrupole response, and
2. global passivity.

---

# 5. Why this matches the Baldwin coarse-graining logic

The Baldwin analysis explicitly separates three questions:

1. whether the coarse-grained **reference density** reproduces the long-range potential and applied-field work;
2. whether its **first-order response** reproduces the long-range response potential;
3. whether higher-order responses are absent or otherwise controlled. ([arXiv][1])

For the reference state, the preprint notes that constant and uniform fields require correct total charge and dipole, while more complicated applied fields can require higher multipole information. ([arXiv][1]) Your exterior point charges are precisely such nonuniform fields.

The new evidence maps cleanly onto those separate conditions:

$$
\begin{array}{lll}
\text{Condition 1: reference potential} &
\text{radial8 fails;} &
\text{static }l=2\text{ repairs it},\\[2mm]
\text{Condition 2: first response} &
\text{radial8 passes;} &
\text{dynamic }l=2\text{ is unnecessary},\\[2mm]
\text{Condition 3: higher response} &
\text{quadratic scalar;} &
\nabla_\eta^3E=0.
\end{array}
$$

The paper’s first-response condition concerns whether the response basis reproduces the far-field response potential, not whether every angular channel present in the reference density must itself be dynamically variable. ([arXiv][1])

One important distinction:

> “Zero induced quadrupole susceptibility” is **not** Baldwin’s “absence of higher-order response.”

The former restricts which spatial source channels participate in the linear response. The latter concerns second and higher derivatives of the density with respect to perturbation. The proposed scalar satisfies both:

* first-order response exists in radial8;
* \(l=2\) has zero first-order response;
* all source response beyond first order vanishes because the field scalar is quadratic.

This is a controlled coarse-graining choice, not an empirical correction.

---

# 6. Comparison with the alternatives

| Design                                                   |                            Reference MEP |                             Response evidence | Structural cost                                                              | Decision       |
| -------------------------------------------------------- | ---------------------------------------: | --------------------------------------------: | ---------------------------------------------------------------------------- | -------------- |
| Permanent radial8 \(+\) static \(l=2\); radial8 response |              Repairs \(6.91\%\to1.64\%\) |           Retains passing \(1.27\%\) response | Adds only \(5N\) affine outputs; no new dynamic root                         | **Use for v1** |
| Full induced radial8 \(+\) \(l=2\)                       |                        Repairs reference | Only \(1.27\%\to1.15\%\) response improvement | Adds \(5N\) response directions plus \(r2\) and \(22\) susceptibility blocks | Reject for v1  |
| radial8 plus local short-range energy correction         | Cannot repair MEP unless field-dependent |       Does not address representation failure | Either ineffective or secretly another source head                           | Reject         |

## Alternative 1: full induced \(l=2\)

A full response model would use the reduced fixed-charge tangent

$$
\ker g_r^{\mathsf T}\oplus\mathcal S_{2,N},
$$

of dimension

$$
(8N-1)+5N=13N-1.
$$

Its susceptibility would be

$$
\chi_{\mathrm{full}}
=
\begin{pmatrix}
\chi_{rr}&\chi_{r2}\\
\chi_{r2}^{\mathsf T}&\chi_{22}
\end{pmatrix}
=
C_{\mathrm{full}}^{\mathsf T}C_{\mathrm{full}}.
$$

This would introduce three new physical objects:

$$
\chi_{r2},\qquad
\chi_{2r},\qquad
\chi_{22}.
$$

The present data do not identify them adequately:

* only four geometry-selected perturbation modes are available per molecule;
* the radial8 response already passes the \(3\%\) gate;
* adding \(l=2\) produces only a \(0.12\)-percentage-point audit improvement;
* no independent atom-centred \(l=2\) perturbations were applied;
* only 32 molecules are available for training.

The \(1.15\%\) result shows that induced \(l=2\) is capable of representing a little more response. It does not show that learning its susceptibility is necessary or transferable.

Adding it would also enlarge the ddPCM coupled Jacobian and require a new stability proof involving the full \(13N-1\) response space. With point quadrupoles, it would additionally introduce field-induced \(r^{-3}\) source components close to the cavity.

**Escalation trigger:** full induced \(l=2\) is reconsidered only if the radial8-only response span exceeds the unchanged \(3\%\) audit gate on a development/validation formula or if a future, separately preregistered field class includes independent atom-centred quadrupolar perturbations.

It is not added merely because \(1.15\%<1.27\%\).

## Alternative 2: local short-range energy correction

A geometry-only correction

$$
E_{\mathrm{sr}}(R)
$$

cannot change the exterior MEP because

$$
\frac{\partial E_{\mathrm{sr}}(R)}{\partial q_{\mathrm{ext}}}=0.
$$

Therefore

$$
E_{\mathrm{sr}}(R)
$$

is mathematically incapable of repairing the observed \(6.91\%\) zero-field MEP error.

If instead one writes a field-dependent local term such as

$$
E_{\mathrm{sr}}(R,\phi)
=
\sum_i
\Theta_i(R):
\frac12\operatorname{STF}\nabla\nabla\phi(R_i),
$$

then this is not an alternative to a quadrupole source. It **is exactly the affine point-quadrupole source coupling**, written under another name.

A nonlinear field-dependent “short-range correction” would introduce unvalidated susceptibility and higher-order response. A correction trained against PCM or experimental solvation energies would be an empirical patch and is prohibited.

Baldwin’s local residual terms are justified only after the explicit coarse-grained density captures the long-range reference potential and field coupling. Here, the exterior audit directly shows that radial8 does not do so for the sulfur molecule. ([arXiv][1])

A local vacuum-energy correction could later repair independent vacuum-energy errors, but it cannot substitute for the missing electrostatic source channel.

---

# 7. Permanent quadrupole head

## Required equivariant form

A scalar-invariant network may not output five unrelated Cartesian values. The quadrupole must be formed from \(l=2\), even-parity tensor carriers:

$$
\boxed{
\Theta_i(R)
=
\sum_{c=1}^{n_c}
a_{ic}(R)\,T_{ic}^{(2)}(R),
}
$$

where

$$
a_{ic}(R)\in V_0^+
$$

are invariant scalar coefficients and

$$
T_{ic}^{(2)}(R)\in V_2^+
$$

are equivariant STF carriers.

Under \(O\in E(3)\),

$$
T_{ic}^{(2)}(OR)
=
O\,T_{ic}^{(2)}(R)\,O^{\mathsf T},
$$

and therefore

$$
\Theta_i(OR)
=
O\,\Theta_i(R)\,O^{\mathsf T}.
$$

No atom-local coordinate frame should be introduced.

## Explicit pair carriers

A transparent carrier family is

$$
\boxed{
T_{ik}^{\mathrm{pair}}
=
\sum_{j\ne i}
\chi(r_{ij})\,
\rho_k(r_{ij})\,
\operatorname{STF}
\left(
\widehat r_{ij}\otimes\widehat r_{ij}
\right),
}
$$

with smooth cutoff \(\chi\) and a very small number of fixed radial functions \(\rho_k\).

This construction is:

* variable \(N\);
* \(E(3)\)-equivariant;
* even under inversion;
* local;
* smooth when the cutoff is smooth;
* independent of any atomic partition target.

Two radial pair carriers are enough for the first candidate.

## Frozen MDP and POLAR \(l=2\) features

They are permitted.

In particular, the following can be used as geometry carriers:

$$
T_i^{\alpha}
=
\operatorname{STF}
\left(
\alpha_i^{\mathrm{MDP}}
\right),
$$

and frozen zero-field MACE-POLAR node features

$$
X_{ic}^{(2,+)}.
$$

Their use is valid provided:

1. they are evaluated at zero field;
2. all backbone weights remain frozen;
3. their coordinate dependence remains in autograd;
4. they are used only as geometric equivariant descriptors;
5. no original MACE-POLAR source, induced source, or atomic coefficient is used as a target;
6. the learned \(\Theta_i\) is not interpreted as the MDP or POLAR atomic partition.

Using \(\operatorname{STF}\alpha_i^{\mathrm{MDP}}\) as a carrier does not assert that the permanent quadrupole is proportional to the atomic polarizability. It only supplies an environment-dependent \(l=2\) orientation tensor.

Explicit pair carriers are therefore **not mathematically mandatory** if reliable frozen \(l=2\) features are already available. At least one genuine \(V_2^+\) carrier family is mandatory.

## Recommended minimal candidate hierarchy

The first candidate should use

$$
\boxed{
\Theta_i
=
a_{i0}\operatorname{STF}(\alpha_i^{\mathrm{MDP}})
+
a_{i1}T_{i1}^{\mathrm{pair}}
+
a_{i2}T_{i2}^{\mathrm{pair}}.
}
$$

The three coefficients are produced by a small shared invariant head. No new message-passing layer is added.

Recommended limits:

$$
n_c\leq3,
\qquad
P_{\mathrm{new},\,l=2}\leq512.
$$

Only if this candidate fails the unchanged zero-field validation gate may a second and final candidate add at most two compressed frozen POLAR \(2e\) channels:

$$
\Theta_i
=
\Theta_i^{\mathrm{base}}
+
\sum_{c=1}^{2}b_{ic}X_{ic}^{(2,+)}.
$$

The total new \(l=2\) head should remain below approximately \(1024\) trainable parameters.

If the first candidate passes, stop. Do not evaluate a larger model merely to lower the mean error.

---

# 8. Training remains observable-only

The zero-field prediction is

$$
\boxed{
\widehat V^{(0)}
=
A_r
\left(
s_Q+U_Nm_0
\right)
+
A_2\Theta_0.
}
$$

For point-charge mode \(k\), with exactly zero induced quadrupole response,

$$
\boxed{
\widehat V_k^{(1)}
=
-A_rU_NKU_N^{\mathsf T}a_{r,k}.
}
$$

There is no \(A_2\) term in the central response because

$$
\frac{d\Theta_0}{dq}=0.
$$

The enthalpy slope is

$$
\boxed{
\widehat g_k^E
=
a_{r,k}^{\mathsf T}
\left(
s_Q+U_Nm_0
\right)
+
a_{2,k}^{\mathsf T}\Theta_0.
}
$$

Train only against:

* raw zero-field exterior MEP;
* raw \(\pm q\) exterior MEP through central response;
* raw full external enthalpy slopes;
* molecular dipole and dipole response.

There is no loss of the form

$$
\|\Theta_i-\Theta_i^{\mathrm{fit}}\|^2.
$$

The augmented per-molecule least-squares quadrupole coefficients are a **representation diagnostic only**. They must not be persisted, exposed to the network, used for initialization, or used as auxiliary labels.

Because intrinsic point quadrupoles have zero total dipole, the quadrupole head is identified primarily through the dense exterior MEP and external-enthalpy slope, not through molecular dipole labels.

---

# 9. ddPCM coupling

Let the reciprocal ddPCM reaction operator on the direct-sum source space be

$$
\begin{pmatrix}
\eta_r\\
\Gamma
\end{pmatrix}
=
\begin{pmatrix}
\mathcal R_{rr}&\mathcal R_{r2}\\
\mathcal R_{2r}&\mathcal R_{22}
\end{pmatrix}
\begin{pmatrix}
s_r\\
\Theta_0
\end{pmatrix},
$$

with exact metric reciprocity

$$
\mathcal R_{2r}
=
\mathcal R_{r2}^{\dagger}.
$$

Although \(\Theta_0\) does not respond, it contributes to the PCM boundary potential through \(\mathcal R_{r2}\Theta_0\), and the PCM reaction field exerts a force and torque through the quadrupole receiver.

The radial source remains

$$
s_r
=
s_r^0-U_NKU_N^{\mathsf T}\eta_r.
$$

The static quadrupole changes the forcing term of the coupled root but not its response Jacobian. In reduced radial coordinates, define

$$
P_r
=
-U_N^{\mathsf T}\mathcal R_{rr}U_N
\succeq0.
$$

The same uniqueness condition remains

$$
\boxed{
\lambda_{\max}
\left(
P_r^{1/2}KP_r^{1/2}
\right)<1.
}
$$

A production margin such as the already proposed \(0.90\) bound remains appropriate.

The quadrupole cross-block

$$
\mathcal R_{r2}\Theta_0
$$

appears on the right-hand side, but because \(\Theta_0\) is field-independent it does not change the linearized root matrix.

This is another reason not to add induced \(l=2\) without evidence: static \(l=2\) repairs the source while preserving the existing root dimension and stability proof.

---

# 10. Coordinate derivatives required for force

At fixed external potential representation, write

$$
y=C(R)\xi,
\qquad
\xi=U_N^{\mathsf T}\eta_r(R).
$$

Because \(U_N\) and \(s_Q\) are coordinate-independent for fixed \(N\), widths, charge, and atom ordering, the explicit coordinate derivative is

$$
\boxed{
\begin{aligned}
\frac{\partial E}{\partial R_a}
={}&
\frac{\partial E_{\mathrm{vac}}}{\partial R_a}
+
\left(
\frac{\partial m_0}{\partial R_a}
\right)^{\mathsf T}\xi
-
y^{\mathsf T}
\left(
\frac{\partial C}{\partial R_a}
\right)\xi
\\
&+
\left(
\frac{\partial\Theta_0}{\partial R_a}
\right)^{\mathsf T}\Gamma
+
s_r^{\mathsf T}
\frac{\partial\eta_r}{\partial R_a}
+
\Theta_0^{\mathsf T}
\frac{\partial\Gamma}{\partial R_a}.
\end{aligned}
}
$$

The following derivatives must remain in the same differentiable graph:

1. \(\partial_R E_{\mathrm{vac}}\).
2. \(\partial_R m_0\).
3. \(\partial_R C\), including all pair distances, directions, and smooth cutoffs.
4. \(\partial_R\Theta_0\), including frozen MDP/POLAR features and explicit STF pair carriers.
5. \(\partial_R A_r\): translations of both-width GTO source centres and radial receiver kernels.
6. \(\partial_R A_2\): translations of point-quadrupole centres.
7. \(\partial_R\Gamma\), which includes third spatial derivatives of the Coulomb kernel:

   $$
   \nabla\nabla\nabla |r-R_i|^{-1}.
   $$
8. Derivatives of all PCM cavity points, switching functions, quadrature weights, and boundary operators.
9. Derivatives of all four PCM source/receiver blocks:

   $$
   rr,\quad r2,\quad 2r,\quad22.
   $$
10. The mixed derivatives

    $$
    \frac{\partial^2E}{\partial R\,\partial\eta_r},
    \qquad
    \frac{\partial^2E}{\partial R\,\partial\Gamma}.
    $$
11. The implicit derivative of the converged radial-source/PCM-surface root.

For a root residual

$$
F(R,z^*)=0,
$$

solve

$$
F_z^{\mathsf T}\lambda
=
\frac{\partial G}{\partial z},
$$

and evaluate

$$
\boxed{
\frac{dG}{dR}
=
\frac{\partial G}{\partial R}
-
\lambda^{\mathsf T}
\frac{\partial F}{\partial R}.
}
$$

Frozen backbone weights must have `requires_grad=False`; their outputs must not be detached from \(R\).

Point quadrupoles are acceptable only on the frozen exterior/cavity domain where no evaluation or surface point approaches an atomic centre. If the ddPCM cavity violates that source-distance condition, the result is `STOP`; the next representation to test is a fixed-width \(l=2\) Gaussian, not a short-range energy correction.

---

# 11. Strict validation and stop rule

## Stage 1: representation closure

For the permanent augmented span, the charge-constrained dimension is

$$
(8N-1)+5N=13N-1.
$$

Every dense fit and rotated-audit partition must therefore contain more than \(13N-1\) independent probes, not merely more than \(8N\), and the nondimensionalized design matrix must have full numerical rank under the frozen SVD tolerance.

Per-molecule diagnostic coefficients are discarded immediately.

The unchanged gates are:

$$
\boxed{
\epsilon_{V0,\mathrm{audit}}
\left(
\mathrm{radial8}\oplus l=2_{\mathrm{perm}}
\right)
\leq5\%
}
$$

for every development/validation molecule, and

$$
\boxed{
\epsilon_{V1,\mathrm{audit}}
\left(
\mathrm{radial8\ response}
\right)
\leq3\%
}
$$

for every molecule and every preregistered response aggregate.

No mean-over-formulas rescue is allowed.

Consequences:

* any augmented permanent audit error \(>5\%\): this repair fails;
* any radial8 response audit error \(>3\%\): zero-induced-\(l=2\) v1 fails;
* neither failure may be repaired with a local energy term.

The reported sulfur values pass these absolute gates:

$$
1.64\%<5\%,
\qquad
1.27\%<3\%.
$$

However,

$$
\frac{1.64}{0.387}\approx4.24.
$$

So the excellent fit error must not be interpreted as proof of transferability. If an audit/fit-ratio gate was separately sealed in the actual registry, that gate remains binding; the \(1.64\%\) absolute pass cannot override it.

## Stage 2: transferable-head closure

Train only on the 32 training formulas.

For the first quadrupole-head candidate, evaluate all 14 validation formulas once and require, per formula,

$$
\epsilon_{V0,\mathrm{audit}}\leq5\%,
\qquad
\epsilon_{V1,\mathrm{audit}}\leq3\%.
$$

Also require unchanged:

* charge;
* energy/MEP conjugacy;
* adjointness;
* rotation/reflection covariance;
* reciprocity;
* passivity;
* PCM spectral margin;
* multi-start root agreement.

The quadrupole head must improve reference MEP without changing the mathematical response space. Any nonzero numerical derivative

$$
\frac{\partial\Theta}{\partial q}
$$

above the structural tolerance is an implementation failure.

## Stage 3: fixed capacity sequence

Use at most two preregistered permanent-quadrupole candidates:

1. **Q-A:** MDP \(\operatorname{STF}\alpha_i\) plus two explicit pair STF carriers; at most 512 new parameters.
2. **Q-B:** Q-A plus at most two compressed zero-field POLAR \(2e\) channels; at most 1024 new parameters.

Rules:

* If Q-A passes every gate, select Q-A and stop.
* Q-B is evaluated only if Q-A fails the zero-field MEP gate while all structural and response gates pass.
* If Q-B fails, return `STOP`.
* No additional message passing, hidden width, carrier count, or induced \(l=2\) is added using the same validation set.
* A lower aggregate error cannot displace an earlier, smaller all-gates-passing candidate.

## Stage 4: induced-\(l=2\) escalation rule

A full induced \(l=2\) model is allowed into a later preregistration only if at least one of these occurs before blind opening:

$$
\epsilon_{V1,\mathrm{audit}}(\mathrm{radial8})>3\%
$$

on a validation formula, or a new independent perturbation dataset directly probes atom-centred \(l=2\) fields and demonstrates a failed radial8 response representation.

The observed change

$$
1.27\%\rightarrow1.15\%
$$

is not such a trigger.

## Stage 5: blind opening

After architecture, carrier set, weights, thresholds, PCM coupling, and force code are sealed, open the 14 blind formulas once.

A blind failure invalidates the model identity. It does not authorize:

* adding induced quadrupoles;
* adding more POLAR channels;
* changing point quadrupole damping;
* changing the cavity;
* changing the \(5\%\) or \(3\%\) gates;
* training against experimental solvation data.

A subsequent attempt requires a new campaign identity and new independent holdout.

---

# Final ruling

$$
\boxed{
E_\theta
=
E_{\mathrm{vac}}
+
\underbrace{\langle s_{r,0},\eta_r\rangle
+\langle\Theta_0,\Gamma\rangle}_{\text{permanent radial8 + static }l=2}
-
\underbrace{\frac12\|C\,U_N^{\mathsf T}\eta_r\|^2}_{\text{radial8 passive response only}}
}
$$

is the correct minimum v1 candidate.

It repairs the demonstrated reference-density deficiency in the source representation, preserves the already adequate response representation, keeps fixed charge exact, preserves scalar conjugacy, reciprocity, and passivity, and adds no unsupported dynamic degrees of freedom.

The full induced-\(l=2\) model is scientifically premature. The local-energy alternative is mathematically incapable of repairing the exterior MEP unless it is secretly rewritten as exactly the same quadrupole field coupling.

[1]: https://arxiv.org/html/2603.14700v2 "Design Space of Self-Consistent Electrostatic Machine Learning Interatomic Potentials"
