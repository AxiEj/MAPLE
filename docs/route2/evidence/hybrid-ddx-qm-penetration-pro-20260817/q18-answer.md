# Verdict

**The exact-radius W97M‑V/def2‑TZVPD QM/ddPCM component ledger is presently inadmissible.** The evidence is already stronger than “ordinary SCF difficulty,” because the pathology appears at the *frozen gas density*, before electronic feedback: the electrostatic term changes to the wrong passivity sign, the reaction operator entering the Fock matrix becomes enormous, and a 10% cavity enlargement changes both quantities discontinuously in scale. The successful H(_2) directional-derivative test excludes a missing or incorrectly differentiated Fock contribution, but it does **not** establish passivity, boundedness, cavity-source admissibility, or existence of a stable benzene stationary branch.

The sharp diagnosis is:

> **A penetration-triggered instability of the composed QM–ddPCM functional, or a discrete cavity/operator pathology activated by the penetrating density, is strongly indicated. Literal unboundedness below has not yet been proved.**

The (s=1.5) calculation establishes a well-behaved **enlarged-cavity reference** only. It cannot validate either (G_{\rm cont}) or (\Delta E_{\rm raw}+G_{\rm cont}) at the original radii.

One fixed, target-independent stability-continuation experiment can still decide whether the exact finite-basis problem contains a recoverable stable solution. If that experiment fails, the matched QM ledger closes permanently for the unchanged profile. The operational hybrid need not automatically close, but it must then be defended only as a prospectively frozen empirical conservative scalar, with no QM component interpretation and no post-target choice between (\Phi_0) and (\Phi_{\rm raw}).

---

# I. Structural theorems

## 1. The three different notions of stability

At fixed geometry and cavity scale (s), write the discrete continuum equations abstractly as

[
A_s \sigma = B_s q(P),\qquad
R_s=C_sA_s^{-1}B_s ,
]

where (P) is the electronic density matrix, (q(P)) is the nuclear-plus-electronic source, and (R_s) is the composed reaction-field map. The coupled electronic scalar is

[
\mathcal F_s(P)
===============

E_0(P)+G_s(P),
\qquad
G_s(P)=\frac12\langle q(P),R_s q(P)\rangle .
]

Three questions are logically independent.

### Continuum linear-solve stability

This asks whether (A_s) is invertible and whether (\sigma) is obtained with controlled forward error. It is diagnosed by the conditioned residual, singular-value or condition estimates, adjoint consistency, and discretization refinement.

A tiny residual does **not** imply that the resulting reaction field is physically meaningful for the source. A well-conditioned boundary solver can faithfully evaluate a boundary formulation outside its source-support assumptions.

### Electronic stationary-point stability

A stationary RKS density satisfies

[
[\tilde F_s(P_*),\tilde P_*]=0
]

in an orthonormal representation. Its character is determined by the occupied–virtual orbital Hessian

[
H_s
===

\frac{\partial^2 \mathcal F_s}{\partial \kappa_{\rm ov}^2}\bigg|*{P**},
]

including the full ddPCM density response.

A strict local minimum requires, after removing genuine gauge modes,

[
\lambda_{\min}(H_s)>0.
]

A negative eigenvalue means that the SCF solution is a saddle. A zero eigenvalue means loss of local invertibility of the response equations; continuation of the stationary branch can bifurcate or terminate there. SCF stability analysis and direct orbital minimization are precisely the tools needed to distinguish a minimum from a saddle. ([arXiv][1])

### Iterative SCF convergence

The Roothaan/DIIS iteration has its own fixed-point Jacobian. Its spectral radius may exceed one even when (H_s) is positive definite. In that case damping, DIIS, augmented-Hessian methods, or direct minimization may recover the same stable minimum. Conversely, damping can appear to stabilize an iteration around a saddle.

Therefore:

[
\text{SCF divergence}
;\not\Rightarrow;
\text{no stationary minimum},
]

and

[
\text{SCF convergence}
;\not\Rightarrow;
\text{stable minimum}.
]

---

## 2. Finite-basis boundedness theorem

For a finite AO basis with a positive-definite overlap matrix, the admissible (N)-electron idempotent density matrices form a compact Grassmann manifold. If (A_s) is nonsingular and (E_0(P)) and (G_s(P)) are finite continuous functions on that manifold, then

[
\inf_P \mathcal F_s(P)>-\infty
]

and a finite-basis global minimum exists.

Consequently, the iteration trace reaching (-10^5,E_h) does **not**, by itself, prove that the finite def2‑TZVPD functional is mathematically unbounded below. It could mean:

1. convergence toward an absurd but finite spill-state minimum;
2. passage through nonrepresentable or numerically corrupted extrapolated states;
3. near-linear dependence or quadrature failure;
4. a nearly singular continuum map;
5. failure of the physical branch while another unphysical finite-basis minimum remains.

Literal unboundedness is instead a **basis-sequence statement**. It requires a sequence of admissible, increasingly complete electronic spaces (B_n) and densities (P_n) such that

[
\mathcal F_s^{B_n}(P_n)\longrightarrow -\infty .
]

An abstract sufficient stability condition is that the negative part of (G_s) be uniformly form-bounded relative to the stabilizing kinetic and Coulomb terms in (E_0), with relative bound below one. Unboundedness can occur only if this uniform bound fails.

No finite computation can prove the infinite limit. A nested variational descent can, however, provide a decisive numerical rejection of basis stability.

---

## 3. Passivity and source support

For a passive dielectric with (\epsilon_{\rm out}>\epsilon_{\rm in}), and for a charge distribution belonging to the source class assumed by the sharp-boundary PCM—most importantly, charge contained in the solute cavity—the properly paired reaction operator is negative semidefinite:

[
G_s[q]\le 0.
]

PCM can be written through a variational free-energy formulation, but that variational structure presupposes a correctly assembled dielectric problem and admissible source treatment. ([AIP Publishing][2])

A small cavity puts a non-negligible electronic tail outside the nominal solute domain. This “outlying charge” is a long-recognized serious problem for sharp-boundary continuum models; formulations with diffuse boundaries or explicit volume polarization are designed to avoid treating that exterior density through a surface-only relation. ([AIP Publishing][3])

Thus the original-radius value

[
G_{\rm dd}[\gamma_{\rm vac}]>0
]

has only four credible explanations:

1. the sign or energy pairing is wrong at that cell;
2. the linear solve is numerically unreliable;
3. the discretization has lost passivity;
4. the diffuse gas density violates the source-support assumptions, so the implemented surface map is no longer the intended passive dielectric functional.

Because the same code gives negative (G_{\rm dd}) after radius enlargement, a globally reversed sign is unlikely. The main remaining distinction is between a discrete operator pathology and a source-penetration pathology.

---

# II. What the present evidence establishes

## Established

The pathology precedes SCF feedback. A fixed gas density at the original cavity produces an anomalous positive polarization energy and an enormous reaction-field operator. Therefore this is not adequately described as “DIIS happened to fail.”

The clean (s=1.5) result, including exact closure,

[
D_{\rm QM}+S_{\rm QM}=T_{\rm QM},
]

demonstrates correct same-scalar bookkeeping at (s=1.5). It does not establish continuity or validity at (s=1.0).

The ML hybrid remaining stable at (s=1.0) does not show that the cavity is safe for QM. It shows that the ML source class—point permanent multipoles plus finite-width induced Gaussians—does not possess the freely variational diffuse electronic modes that can exploit the cavity boundary.

## Not established

The data do not yet distinguish whether (A_s) itself is nearly singular from whether a well-solved (A_s) is being driven by an inadmissible penetrating source.

They do not establish a stationary point at (s=1.0), because the reported energies come from a divergent iteration.

They do not establish unboundedness, because the path has not been restricted to normalized, idempotent densities and no nested basis lower-bound sequence has been produced.

They do not establish that the raw AO Frobenius norm is intrinsically divergent. (|V_{\rm dd}|_F) depends on AO representation. Comparisons within the same basis are informative, but the invariant cross-basis diagnostic is the spectral norm of

[
S^{-1/2}V_{\rm dd}S^{-1/2}
]

or the occupied–virtual matrix elements in an orthonormal MO basis. Likewise, (|\xi|_2) is meaningful only with a fixed scaling or energy norm.

## Current classification

**Suspected outlying-charge-triggered loss of the physical electronic branch, with a possible accompanying discrete cavity instability.**

That classification is sufficient to suspend the ledger now. It is not yet sufficient to write “the functional is mathematically unbounded below.”

---

# III. Sharp numerical certificate

## A. Recoverable SCF-convergence problem

Classify the event as merely algorithmic only if all of the following are obtained for the unmodified (s=1.0), def2‑TZVPD scalar:

1. A zero-level-shift, integer-occupation stationary solution with a small orbital gradient.
2. A positive occupied–virtual Hessian, including PCM response.
3. The same solution from several frozen, target-independent starts.
4. Direct orbital minimization converges even if ordinary DIIS does not.
5. Removal of all temporary damping, level shifting, fractional occupations, or constraints leaves the same stationary solution.
6. An exact orbital-rotation energy scan around the solution is locally upward in every computed low-eigenvalue direction.
7. The solution is stable to modest nested diffuse augmentation.

A small HOMO–LUMO gap or difficult DIIS trajectory is not enough to reject the functional if these conditions pass.

## B. Metastable minimum or saddle

A **saddle** is certified by

[
|g_{\rm ov}|\approx 0,\qquad
\lambda_{\min}(H_s)<-\tau_H.
]

The corresponding eigenvector supplies an explicit idempotent downhill orbital-rotation path. No SCF algorithm can convert that stationary point into a minimum without changing the scalar.

A **metastable minimum** has

[
\lambda_{\min}(H_s)>\tau_H,
]

but a second target-independent start converges to a lower stable solution. Reverse radius continuation may show hysteresis between the chemical and spill branches.

For the proposed matched ground-state ledger, neither result is acceptable:

* a saddle is not a stable electronic state;
* selecting a higher chemical minimum over a lower spill minimum would be an implicit density constraint or branch-selection rule that was not part of the original QM/ddPCM model.

Maximum-overlap or occupation-following methods may locate the metastable branch diagnostically, but they cannot turn it into the unconstrained ground-state reference.

## C. Genuine spill/polarization catastrophe

The strongest finite numerical certificate is the conjunction of:

1. (A_s) is accurately solvable and its interior-source reaction map passes reciprocity and passivity tests.
2. The physical stationary branch loses Hessian positivity as (s) approaches 1.
3. The unstable Hessian mode is spatially concentrated at or outside the cavity boundary.
4. The exterior electron population increases along the downhill mode.
5. Fully minimized energies decrease systematically under a nested diffuse-basis sequence.
6. The stabilization comes overwhelmingly from (G_{\rm dd}), rather than from a broken XC grid, DF metric, electron-count error, or overlap singularity.
7. A temporary level shift may suppress the spill state, but removing the shift restores the instability.
8. Gas, continuation, SAD, and deliberate spill starts either find no stable chemical minimum or find a lower spill minimum.

This certifies a **numerically catastrophic lack of a stable basis limit**, sufficient to reject the ledger. It still is not a formal proof that the exact continuum infimum equals (-\infty); that would require an analytic form-bound argument or an infinite constructive sequence.

## Fail-closed stop rule

After the single frozen experiment below:

* a positive Hessian above tolerance is “stable”;
* a negative Hessian below tolerance is “unstable”;
* a near-zero eigenvalue after one predefined precision refinement is “unresolved”;
* unresolved means **no ledger**, not more solver tuning;
* failure to find a stationary point without a certified downhill path is “numerically unresolved,” not automatically “catastrophe,” but it still closes the ledger;
* no new basis, radius, shift, occupation rule, or solver setting may be introduced afterward.

---

# IV. What options A–F can legitimately answer

| Option                                                                     | Legitimate question                                                                                                                                                                                  | Conclusion that cannot be transported to the original profile                                                                                                                                                                                           |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Exact cavity, non-diffuse basis**                                     | Does a compact finite-dimensional electronic subspace have a stable same-cavity branch? How does the fixed-source ddPCM operator behave when exterior density is suppressed by basis incompleteness? | It cannot establish (D_{\rm QM},S_{\rm QM},T_{\rm QM}) for W97M‑V/def2‑TZVPD, nor show that the diffuse-basis problem is stable. A compact basis may merely hide the destabilizing degree of freedom.                                                   |
| **B. Enlarged cavity**                                                     | Does a newly defined cavity profile possess a stable QM/ddPCM branch and a usable ledger? The (s=1.5) result already answers this locally in the affirmative.                                        | It cannot adjudicate the original-radii (G_{\rm cont}) or (\Delta E_{\rm raw}+G_{\rm cont}). Changing radii changes the continuum state, projected MACE field, induced source, energy, and eventually forces. It is a new hybrid profile.               |
| **C. Constrained or projected QM density**                                 | What is the stationary energy of a confined-electron or projected-subspace QM/ddPCM functional? With a proper constrained Lagrangian, this can be mathematically consistent.                         | It cannot be called unconstrained W97M‑V/ddPCM. The constraint supplies the missing short-range/exclusion physics and changes both energy and response.                                                                                                 |
| **D. IEFPCM/GEPOL density followed by ddPCM evaluation**                   | How transferable is a density between two continuum operators? What fixed-density ddPCM energy does an IEFPCM density produce?                                                                       | That density is not stationary for the ddPCM scalar. (D+S) is not the self-consistent ddPCM total and cannot select the hybrid ledger. Exact post-evaluation closure is irrelevant without same-scalar stationarity.                                    |
| **E. Frozen gas or foreign PCM densities**                                 | Fixed-source sign, magnitude, linearity, MEP, Fock derivative, and implementation comparisons. These are valuable operator diagnostics.                                                              | They cannot supply the self-consistent electronic deformation (D_{\rm QM}), the matched (T_{\rm QM}), or a QM polarization-component interpretation.                                                                                                    |
| **F. ddPCM regularization or short-range dielectric/exclusion functional** | Does a newly specified regularized continuum-electronic model have a bounded, stable stationary branch? A physically derived exclusion or diffuse-boundary model can answer this.                    | If used only on the QM side, it is an unmatched reference. If used on both QM and hybrid sides, it is a new Route‑2 profile requiring a full new freeze, validation, and force derivation. It cannot retroactively validate the original pyddx profile. |

Options A, C, D, and E are therefore **diagnostics or alternative reference definitions**, not rescues of the original matched ledger. Options B and F are **new models**.

---

# V. The one immediate highest-information experiment

Run exactly one experiment:

## Nested-basis/radius stationary-branch and spill-mode map

### Frozen physical and numerical settings

Keep the supplied benzene geometry, W97M‑V, RKS/DF settings, solvent dielectric, (\eta), shift, cavity construction, and baseline ddPCM discretization unchanged.

Use the fixed radius sequence

[
s\in
{1.50,;1.30,;1.20,;1.10,;1.05,;1.025,;1.00}.
]

Use the following basis sequence:

[
B_c=\text{def2-TZVP},
\qquad
B_0=\text{def2-TZVPD}.
]

For each element (X\in{\mathrm C,\mathrm H}) and (l\in{s,p}), let (\alpha_{Xl}) be the smallest primitive exponent of that angular momentum in (B_0). Define nested diagnostic augmentations

[
B_1
===

B_0+
\left{\alpha_{Xl}/3.2\right},
]

[
B_2
===

B_1+
\left{\alpha_{Xl}/3.2^2\right},
]

with one uncontracted primitive of each specified exponent placed on every atom of the corresponding element.

There is no exponent optimization, no deletion selected after seeing results, and no further basis. Use one fixed canonical-orthogonalization cutoff of (10^{-12}). If the retained basis rank changes because of linear dependence, the nested-basis classification is terminally unresolved.

### Stage 1: continuum operator audit

At every radius, before PCM-SCF:

1. Form or estimate the singular spectrum of a diagonally equilibrated (A_s).
2. Solve the gas-density right-hand side to relative residual at most (10^{-12}).
3. Require the estimated forward-error bound
   [
   \kappa_2(A_s),r_{\rm rel}\le 10^{-8}.
   ]
4. Check adjoint/reciprocity in the actual ddPCM energy pairing using 16 deterministic, geometry-hash-seeded interior Gaussian monopole/dipole probes. The relative bilinear reciprocity defect must be at most (10^{-9}).
5. Normalize the probes to unit vacuum Coulomb self-energy and require the largest eigenvalue of the symmetrized interior reaction matrix to be no greater than (10^{-8}) times its spectral norm.
6. At the gas density only, repeat the calculation at the two frozen diagnostic refinements
   [
   (l_{\max},n_{\rm leb})=(17,1730),;(19,2030).
   ]
   Require
   [
   |G_{15,1202}-G_{19,2030}|\le 10^{-6}\ E_h
   ]
   and
   [
   \frac{|
   \tilde V_{15,1202}-\tilde V_{19,2030}
   |*2}
   {\max(1,|\tilde V*{19,2030}|_2)}
   \le 10^{-4},
   ]
   where (\tilde V=S^{-1/2}VS^{-1/2}).

Failure here is classified as a **continuum/discretization failure**, not an SCF catastrophe. The refined discretizations are diagnostic only and cannot replace the frozen (15/1202) profile.

### Stage 2: unshifted electronic minimization

At every ((B,s)) cell, run a Riemannian trust-region orbital minimization of the original, zero-level-shift scalar. Conventional DIIS may be run only as an algorithmic comparator.

Use exactly these starts:

1. same-basis vacuum orbitals;
2. same-basis SAD orbitals;
3. the converged stationary solution from the immediately preceding larger radius;
4. two densities obtained by (\pm0.15)-radian occupied–virtual rotation along the lowest solvent-coupled Hessian mode from that preceding radius.

At (s=1.50), where the fourth start is unavailable, use the lowest mode obtained from the converged vacuum-start solution.

A temporary level shift is allowed only through the frozen sequence

[
0.5,;0.1,;0\ E_h.
]

No result is accepted until it reconverges at exactly zero shift. Fractional occupations, MOM/IMOM state following, density projection, and permanent damping are diagnostic only and cannot produce an accepted reference.

### Stage 3: stationary-point invariants

An accepted stationary point must satisfy:

[
|g_{\rm ov}|*{\rm RMS}\le10^{-8}\ E_h,
\qquad
|g*{\rm ov}|_\infty\le10^{-7}\ E_h,
]

[
|\Delta E|\le10^{-10}\ E_h
]

for the last five accepted iterations,

[
|\operatorname{Tr}(PS)-N|\le10^{-10},
\qquad
|C_{\rm occ}^{T}SC_{\rm occ}-I|_F\le10^{-10},
]

and the ddPCM residual criterion above.

Also record and audit kinetic, nuclear-attraction, Coulomb, exchange-correlation, VV10, and (G_{\rm dd}) contributions. Require:

* nonnegative kinetic energy;
* nonnegative Coulomb self-energy within (10^{-9},E_h);
* DFT-grid electron count within (10^{-7}) electron of (\operatorname{Tr}(PS));
* unchanged conclusions on a doubled XC/VV10 quadrature.

This prevents an XC-grid or DF failure from being mislabeled a dielectric catastrophe.

### Stage 4: full stability test

Compute the five lowest eigenpairs of the RKS occupied–virtual Hessian, including the fully relaxed ddPCM response, by matrix-free finite differences of the exact orbital gradient.

Use both orbital-rotation steps

[
h=10^{-4},\qquad h=5\times10^{-5}.
]

The corresponding lowest eigenvalues must agree within (2\times10^{-6},E_h).

Classify with

[
\tau_H=10^{-5},E_h:
]

* **stable:** (\lambda_{\min}>\tau_H);
* **saddle:** (\lambda_{\min}<-\tau_H);
* **borderline:** (|\lambda_{\min}|\le\tau_H).

For a borderline result, perform exactly one predefined refinement with (h=2.5\times10^{-5}) and ddPCM residual (10^{-13}). If it remains borderline, classification is terminally unresolved and the ledger is rejected.

Test the unrestricted/spin-breaking channel separately. A lower UKS solution is recorded as a distinct electronic instability, not silently mixed into the RKS ledger.

### Stage 5: exterior-density and energy-path diagnostics

For every stationary point and every negative or near-zero Hessian mode, compute

[
Q_{\rm out}
===========

\int_{\mathbb R^3\setminus\Omega_s}\rho_e(\mathbf r),d\mathbf r
]

and the exterior norm of the unstable orbital mode.

Use two deterministic nested real-space quadratures and require their (Q_{\rm out}) values to agree within (10^{-4}) electron.

For the lowest Hessian vector (v), evaluate the exact scalar on the idempotent orbital-rotation path

[
P(t)=e^{t\kappa_v}P_*e^{-t\kappa_v}
]

at

[
t=
0,;\pm0.02,;\pm0.05,;\pm0.10,;\pm0.20,;\pm0.40,;\pm0.80.
]

At every point, solve ddPCM fully. A negative Hessian eigenvalue must be accompanied by the predicted quadratic energy decrease at small (t); otherwise the Hessian calculation is not accepted.

### Stage 6: uniqueness and nested-basis stability

Solutions from the frozen starts count as the same minimum only when

[
|\Delta E|\le10^{-8}\ E_h,
]

[
\frac{|\Delta P|_F}{\sqrt{M}}\le10^{-6},
]

and

[
|\Delta Q_{\rm out}|\le10^{-4}\ e.
]

At (s=1.00), the chemical branch must remain stable through (B_0,B_1,B_2). Between (B_1) and (B_2), require

[
|T_{\rm QM}^{B_2}-T_{\rm QM}^{B_1}|
\le2\times10^{-4}\ E_h,
]

[
|Q_{\rm out}^{B_2}-Q_{\rm out}^{B_1}|
\le2\times10^{-3}\ e,
]

[
|\lambda_{\min}^{B_2}-\lambda_{\min}^{B_1}|
\le2\times10^{-5}\ E_h.
]

These are basis-stability tolerances, not accuracy targets.

### Terminal decision table

| Observation                                                                                                                                                   | Classification                                             | Exact-profile quantitative ledger                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| (A_s), reciprocity, passivity, forward error, or implementation invariants fail                                                                               | Continuum/discretization or electronic-integration failure | **Rejected**                                         |
| DIIS fails, but unshifted direct minimization finds one unique solution with positive Hessian and nested-basis stability                                      | Recoverable SCF algorithm failure                          | **Available**                                        |
| Stationary chemical solution has (\lambda_{\min}<-\tau_H)                                                                                                     | Saddle                                                     | **Rejected**                                         |
| Chemical solution is Hessian-stable, but a frozen start finds a lower stable spill solution                                                                   | Metastable chemical branch                                 | **Rejected**                                         |
| Lowest eigenvalue approaches/crosses zero on radius continuation, mode is exterior-localized, and diffuse augmentation lowers the stable minimum non-Cauchily | Penetration/polarization catastrophe                       | **Rejected**                                         |
| No stationary solution and no reproducible same-scalar classification within the fixed budget                                                                 | Numerically unresolved                                     | **Rejected**                                         |
| Exact (B_0,s=1) solution is unique and stable, but (B_1/B_2) reveal a lower spill state or fail the basis-stability tolerances                                | Finite-basis stabilization artifact                        | **Rejected as a chemically defensible QM benchmark** |
| Exact cell and nested augmentations pass all tests                                                                                                            | Stable matched branch                                      | **Compute (D_{\rm QM},S_{\rm QM},T_{\rm QM})**       |

This experiment does not inspect MACE agreement or hydration targets, and it has no adaptive model-selection loop.

---

# VI. Consequence for (\Phi_0) and (\Phi_{\rm raw})

## What the failed QM ledger would mean

If the exact cell is rejected, then the following claims close:

* “(G_{\rm cont}) is quantitatively the QM electrostatic component.”
* “(\Delta E_{\rm raw}+G_{\rm cont}) is quantitatively the QM polarization-plus-continuum component.”
* “QM/ddPCM selects between (\Phi_0) and (\Phi_{\rm raw}).”
* Any common electronic–continuum functional claim.

The failure does **not** logically show that either operational scalar is undefined or nonconservative.

## What can still survive

Both

[
\Phi_0(R)=E_{\rm vac}(R)+G_{\rm cont}(R,u_*(R))
]

and

[
\Phi_{\rm raw}(R)
=================

E_{\rm conditioned,raw}(R,u_*(R))
+
G_{\rm cont}(R,u_*(R))
]

are legitimate mathematical scalars if:

1. the operational fixed point (u_*(R)) is uniquely and reproducibly selected;
2. the fixed-point Jacobian is nonsingular;
3. every source, cavity, projector, and response term entering the scalar is differentiable;
4. forces differentiate that same composed scalar, including implicit fixed-point response;
5. the fixed point is stable from multiple target-independent starts over the claimed chemical domain.

Broad target-blind validation of QM source moments, external MEPs, field-response curves, total-charge invariance, fixed-source ddPCM energies, symmetry, and charging identities—followed by genuinely sealed total solvation-energy validation—can support an **empirical operational model**. It cannot establish the QM meaning of its internal components.

## The non-negotiable governance decision

The source/MEP tests cannot distinguish (\Phi_0) from (\Phi_{\rm raw}), because both candidates share the same source and continuum state. If the matched QM ledger is unavailable, experimental total errors may not be used to choose which accounting is “right.”

Therefore:

* If exactly one of (\Phi_0) or (\Phi_{\rm raw}) was prospectively frozen before total-target access, that scalar may proceed through the operational and sealed empirical gates.
* If both were prospectively registered as two distinct models, each may be accepted or rejected independently under predeclared multiplicity governance; one may not be selected afterward as the winner.
* If neither was prospectively designated and the project requires one final unchanged-source model, the unchanged-source energy lane closes. The missing QM ledger cannot be replaced by target-selected bookkeeping.
* A CDS term cannot repair this admission failure or establish which electrostatic accounting is valid.

Thus the correct decision is:

> **Close the exact-profile QM-component claim if the experiment fails, but do not automatically close a previously frozen empirical operational scalar. Close the entire unchanged-source lane only if no target-independent scalar choice exists or the operational stability/force gates fail.**

---

# VII. Analytic-force consequences

## 1. Conservative derivative of a declared operational scalar

Let the operational fixed-point residual be

[
H(R,u)=u-f(R,u)=0.
]

For either declared scalar (\Phi(R,u)),

[
\frac{d\Phi}{dR}
================

\Phi_R-\Phi_u H_u^{-1}H_R.
]

Equivalently, solve the adjoint equation

[
H_u^{T}\lambda=\Phi_u^{T}
]

and evaluate

[
\frac{d\Phi}{dR}
================

\Phi_R-\lambda^{T}H_R.
]

This derivative is conservative because it is the derivative of an explicit scalar composed with a uniquely selected root. It does not require the root to minimize (\Phi).

The force implementation must include:

* movement and deformation of the cavity;
* derivatives of permanent and induced source maps;
* derivatives of Gaussian widths/centers if geometry-dependent;
* ddPCM primal and adjoint response;
* surface-to-native-field projector derivatives;
* the full fixed-point implicit response;
* derivatives of the selected (E_{\rm vac}) or (E_{\rm conditioned,raw}) term.

Finite-difference agreement must be checked against the complete scalar, not against separately differentiated components assembled afterward.

## 2. Chemical validity

An exact derivative can be the exact force of a chemically wrong scalar. Conservativity establishes energy–force consistency, not physical accuracy. Chemical validity therefore remains an empirical and target-blind physics-validation question.

## 3. Common variational functional

A common electronic–continuum functional would impose stationarity with respect to the shared response variables and permit envelope-theorem simplifications. That claim is already closed for the unchanged checkpoint/source.

The operational adjoint derivative must not be called variational merely because it is analytic or conservative.

## 4. QM reference forces at an instability

For the matched QM/ddPCM problem:

* at a strict local minimum with nonsingular Hessian, an analytic branch derivative can exist;
* at a saddle with nonsingular Hessian, a mathematical branch derivative may still exist, but it is not a ground-state force;
* as (\lambda_{\min}(H)\to0), the electronic response and geometry derivative become ill-conditioned or divergent;
* if no stable stationary branch exists, there is no admissible matched ground-state QM force;
* forces evaluated on a foreign, frozen, constrained, or level-shifted density belong to that altered construction and cannot be transported to the original stationary model.

---

# VIII. Finite non-looping decision tree

1. **Run the single frozen nested-basis/radius experiment.**

2. **Does the exact (s=1) continuum operator pass forward-error, reciprocity, passivity, discretization, and electronic-integration invariants?**

   * **No:** terminate the original matched QM ledger as a continuum/discretization failure.
   * **Yes:** proceed to 3.

3. **Does zero-shift direct minimization produce an exact def2‑TZVPD stationary point?**

   * **No:** terminate the matched ledger as unavailable.
   * **Yes:** proceed to 4.

4. **Is the full occupied–virtual Hessian positive above tolerance?**

   * **No, negative:** terminate as saddle/branch instability.
   * **Borderline after the one refinement:** terminate as unresolved.
   * **Yes:** proceed to 5.

5. **Do all frozen starts reach the same stable minimum?**

   * **No, a lower spill minimum exists:** terminate as metastability/polarization collapse.
   * **Yes:** proceed to 6.

6. **Does the same chemical branch remain stable and Cauchy through (B_1,B_2)?**

   * **No:** terminate as finite-basis stabilization or basis-limit spill instability.
   * **Yes:** admit the exact-profile QM ledger and compute (D_{\rm QM},S_{\rm QM},T_{\rm QM}).

7. **If the matched ledger terminates, was one operational scalar prospectively frozen before target access?**

   * **No:** close the unchanged-source total-energy lane.
   * **Yes:** proceed to 8.
   * **Two separately preregistered scalars:** evaluate each independently; do not select a winner afterward.

8. **Does the frozen operational scalar pass unique-fixed-point, Jacobian-conditioning, scalar/force closure, target-blind source/MEP, fixed-source continuum, and sealed total-energy gates?**

   * **No:** close the unchanged-source route.
   * **Yes:** retain it only as an empirical operational conservative model, explicitly without QM component or common-functional interpretation.

HYBRID QM-DDPCM PENETRATION DECISION

[1]: https://arxiv.org/pdf/1912.12029 "https://arxiv.org/pdf/1912.12029"
[2]: https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.3454683/16050165/014106_1_online.pdf "https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.3454683/16050165/014106_1_online.pdf"
[3]: https://pubs.aip.org/aip/jcp/article/105/22/9972/477983/Treatment-of-the-outlying-charge-in-continuum "https://pubs.aip.org/aip/jcp/article/105/22/9972/477983/Treatment-of-the-outlying-charge-in-continuum"
