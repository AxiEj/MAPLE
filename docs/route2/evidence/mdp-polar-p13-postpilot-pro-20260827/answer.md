## Decision

**Execute O2 now as a one-time A3 optimization-closure experiment. Do not run O1 as a separate model-selection attempt.** Implement the exact analytic derivative of the existing scalar, verify it against the autograd reference, and use it for one bounded deterministic full-batch continuation of the frozen A3 checkpoint.

If—and only if—A3 reaches the preregistered plateau and still fails any frozen gate, the next and final architecture candidate should be **A4**, not cross-factor topology C. A4 is admissible only if its latent-feature provenance is clean and it remains inside the genuinely frozen parameter cap. If A4 fails, stop this branch. Do not proceed to C using the present four-mode data.

That recommendation does **not** mean optimization is likely to rescue A3. It means optimization has not yet been closed well enough to attribute failure to the model class.

## 1. What the present numbers establish

The factorization proves, for every fixed geometry and assuming \(C_\theta\) is independent of \(\eta\),

$$
\chi_\theta(R)=C_\theta(R)^{T}C_\theta(R)\succeq 0,
$$

and therefore

$$
E_{\rm response}
=-\frac12\eta^{T}\chi_\theta\eta,
\qquad
\nabla_\eta E_{\rm response}
=-\chi_\theta\eta .
$$

Thus reciprocity and the stated sign of the linear response are architectural properties, not empirical regularities. Deriving response quantities from one generalized scalar is precisely what enforces conservative response relations and associated symmetry constraints; automatic differentiation is one implementation of that relation, not the source of the relation itself. ([arXiv][1])

The current results also establish that A3 is not fitting one exterior probe frame at the expense of its rotated audit frame. They do **not** establish chemical transferability, full-operator identifiability, or global linearity outside the four probed directions.

The optimization evidence and gate distances point in different directions:

| Quantity                    |           A3 |       Gate | Relative reduction still needed |
| --------------------------- | -----------: | ---------: | ------------------------------: |
| Mean audit MEP              |       11.43% |         8% |                          30.01% |
| Worst audit MEP             |       20.08% |        20% |                           0.40% |
| Mean source-point response  |       12.93% |         5% |                          61.33% |
| Worst source-point response |       23.08% |        20% |                          13.34% |
| Max absolute audit response | 0.01324 Ha/e | 0.002 Ha/e |                          84.89% |

Meanwhile, the normalized training loss fell from 0.01727 to 0.01633 between epochs 1300 and 1499, a **5.44%** reduction; even from 1400 to 1499 it fell **2.57%**. That is not a credible plateau.

Under the favorable—but unproved—assumption that the relevant errors scale like the square root of the MEP loss, reaching 8% mean MEP would require approximately a **51% further reduction in MEP MSE**. Analogous proportional scaling would require approximately 85% and 97.7% MSE reductions for the mean source and maximum-absolute gates. Those latter metrics are not directly represented in the frozen objective, so even those estimates are optimistic heuristics.

Therefore:

* **Optimization underconvergence remains possible.**
* **Optimization-only rescue of all gates is unlikely.**
* **The current evidence does not mathematically prove a capacity failure.** It also cannot distinguish factor-topology error, coefficient-head insufficiency, P13 source-span error, and loss-to-gate misalignment.
* The fourfold improvement requirement passing at 4.07 does not offset any failed conjunctive gate.

## 2. Exact O2 protocol and plateau rule

### Analytic backend

Use the shared intermediate

$$
z=C_\theta(R)\eta
$$

and compute

$$
E_{\rm response}=-\frac12 z^{T}z,
\qquad
s_{\rm analytic}=-C_\theta(R)^{T}z.
$$

Here \(s_{\rm analytic}=\nabla_\eta E\) under the sign convention implied by your negative-semidefinite response. If another part of the code defines the physical source as \(-\nabla_\eta E\), that sign must be transformed once at the interface rather than changed between code paths.

The scalar must remain the normative definition. The analytic source should be a compiled algebraic consequence of the same \(C\), not an independently parameterized output.

Before optimization, require all of the following on randomized train geometries, all source-sector types, multiple \(\eta\) magnitudes, rotations, and points near—but not exactly on—cutoffs:

$$
\|s_{\rm analytic}-s_{\rm autograd}\|_\infty
\le 10^{-11}+10^{-10}\|s_{\rm autograd}\|_\infty,
$$

$$
\left|2E_{\rm response}-\eta^{T}s_{\rm analytic}\right|
\le 10^{-11}+10^{-10}|2E_{\rm response}|,
$$

and agreement of parameter gradients and mixed geometry–field derivatives to

$$
{\rm atol}\le 10^{-9},\qquad {\rm rtol}\le 10^{-8}.
$$

Also run finite-difference `gradcheck` and second-derivative `gradgradcheck` on reduced test systems. PyTorch’s `gradcheck` is specifically intended to compare computed derivatives with finite-difference derivatives, while `gradgradcheck` tests derivatives of derivatives. ([PyTorch Docs][2])

**Backend stop rule:** any failed source, parameter-gradient, mixed-\(R,\eta\), exact-charge, covariance, or second-derivative test terminates O2. Do not optimize with a backend that is only first-derivative equivalent.

### Optimization closure

Use one deterministic full-batch L-BFGS continuation from the exact frozen A3 checkpoint:

* Same 0.82/0.18 data objective and normalization.
* Same regularization. The \(10^{-4}\) weight decay must be represented with exactly the same semantics as in the Adam run; do not silently replace decoupled decay with an L2 penalty or vice versa.
* Strong-Wolfe line search.
* No restart, seed sweep, alternative loss weighting, or audit-based checkpoint choice.
* Maximum 500 accepted quasi-Newton steps and 5000 full-objective evaluations.
* Select the checkpoint with the lowest original training objective only.

Let \(L_k\) be the full training objective after accepted step \(k\), and

$$
b_k=\min_{0\le j\le k}L_j.
$$

Declare a plateau only when three consecutive nonoverlapping 50-step blocks each improve the best loss by less than 0.25%:

$$
\frac{b_{k-150}-b_{k-100}}{b_{k-150}}<0.0025,
$$

$$
\frac{b_{k-100}-b_{k-50}}{b_{k-100}}<0.0025,
$$

$$
\frac{b_{k-50}-b_k}{b_{k-50}}<0.0025.
$$

This objective-based rule is preferable to a parameter-step rule because the factorization has row-sign, row-permutation, and more general factor-gauge flat directions: \(C\) itself is not unique even when \(C^TC\) is.

**Optimization stop rules:**

* Plateau reached: evaluate every frozen gate exactly once at the minimum-loss checkpoint.
* Hard cap reached without plateau: stop the branch. Do not add capacity while optimization remains unresolved.
* NaN, Inf, or persistent line-search failure: stop the branch.
* Plateau reached but any frozen gate fails: reject A3 and proceed to the single conditional A4 attempt.
* All gates pass: freeze A3 and do not run A4.

Fit/audit metrics must not be monitored for early stopping during this continuation. Otherwise the audit frame becomes a model-selection set.

## 3. A4 should precede C—but only conditionally

### Why the four-mode data do not identify a general cross block

For molecule \(i\), let

$$
E_i=
\begin{bmatrix}
\eta_{i1}&\eta_{i2}&\eta_{i3}&\eta_{i4}
\end{bmatrix},
\qquad
U_i=
\begin{bmatrix}
s_{i1}&s_{i2}&s_{i3}&s_{i4}
\end{bmatrix}.
$$

Even if the dense exterior MEP determines each induced P13 source exactly, the data determine only

$$
\chi_i E_i=U_i.
$$

Since \({\rm rank}(E_i)\le 4\), they identify the action of the susceptibility on at most a four-dimensional subspace of a \(13N_i\)-dimensional field space. For any symmetric perturbation \(\Delta\chi_i\) satisfying

$$
\Delta\chi_i E_i=0,
$$

the four-mode predictions are unchanged. In a basis where the excitation subspace occupies the first \(r\le4\) coordinates, an arbitrary symmetric block on the remaining \(13N_i-r\) coordinates is invisible.

Dense MEP points give many equations for each **output source action**. They do not turn four input directions into a full susceptibility measurement.

Joint radial and \(l=2\) excitation also does not independently identify their coupling. Radial–\(l=2\) contributions can trade against within-sector terms along the four correlated excitation directions.

There is a second complication: a “cross factor row” is not a pure off-diagonal modification. A row of the form

$$
c=\begin{bmatrix}c_r&c_2\end{bmatrix}
$$

contributes

$$
c^Tc=
\begin{bmatrix}
c_r^Tc_r & c_r^Tc_2\\
c_2^Tc_r & c_2^Tc_2
\end{bmatrix}.
$$

It changes both diagonal blocks as well as the cross block. A training improvement would therefore not establish that physical cross coupling was the missing ingredient.

Rotational symmetry does not prohibit radial–\(l=2\) coupling in an anisotropic molecular environment; it only constrains how that coupling is constructed. It should vanish in a genuinely spherical atomic environment but may be mediated by molecular geometry tensors. Thus C is physically plausible, but not adequately identified by the present experiment.

### Why A4 is the bounded next candidate

A4 retains the tested factor topology and changes only the geometry-conditioned coefficient representation. That addresses a specific plausible deficiency: frozen \(q,p,\alpha\), element embeddings, and fixed neighbor sums may not distinguish enough local chemical environments.

MACE-style hidden features are equivariant tensors rather than arbitrary Cartesian features, but their locality and receptive field depend on the underlying cutoff and message-passing depth. “No new message passing” does not mean that inherited features are atom-local. ([arXiv][3])

A4 is admissible only under these conditions:

1. **No silent cap increase.** The stated 2500-parameter cap is described as frozen. A4 must remain at or below 2500 unless a separate 4096 tier was committed before the A3 results were opened. A post-failure increase from 2500 to 4096 is capacity relaxation, even though it is not a metric-gate relaxation.

2. **Clean representation provenance.** “Evaluated at zero field” is not the same as “uninformed by prohibited targets.” If the frozen latent extractor was trained using original MACE-POLAR response targets or prohibited VQM24 energies, those targets enter A4 indirectly through transfer learning. Either the upstream representation provenance must exclude those targets, or prior protocol text must explicitly permit that fixed information channel. Otherwise A4 is prohibited and the branch stops after A3.

3. **Use pre-response local latents.** Do not take features after field-dependent polarizable updates, global charge equilibration, source prediction, or global pooling. Contemporary MACE-POLAR architectures can contain nonlocal density updates and global equilibration, so the exact extraction layer matters. ([arXiv][4])

4. **Equivariant PCA only in multiplicity space.** For an \(l\)-irrep with multiplicity index \(a\) and magnetic component \(m\), construct covariance by contracting over \(m\), and apply the same PCA mixing matrix to every \(m\). Do not flatten \(x,y,z\) and perform ordinary Cartesian PCA. Do not mix opposite parity channels. Scalar centering is allowed; subtracting a finite-sample vector mean is not equivariant.

5. **Require \(O(3)\), not only \(SO(3)\).** Electrostatic energy is parity-even. An \(SO(3)\)-only implementation could admit pseudoscalar or pseudotensor contamination that is invisible without reflected examples.

6. **PCA must be fitted inside each CV training fold.** Computing PCA on all 32 formulas and then performing grouped CV leaks held-out molecular geometry information.

7. **Frozen weights must remain geometry-differentiable.** Freezing the descriptor-network parameters must not detach its output from \(R\). Otherwise total forces omit \(\partial C/\partial R\).

8. **Use nested initialization.** Copy the terminal A3 parameters and initialize all new latent-channel couplings to zero, so A4 initially reproduces A3 exactly. This removes an avoidable optimizer/initialization confound.

**A4 stop rule:** one training run, the same O2 backend, the same plateau rule, and one terminal evaluation. Failure of any full-train gate, failure to plateau, provenance failure, or cap violation ends the branch. Do not fall through to C.

C belongs to a separate future preregistration with either additional sector-balanced excitation modes or a target-free tangent-rank certificate showing that the proposed cross contribution to \(\chi\eta\), after projecting out the A3 tangent space, is identifiable. Rank must be assessed on the physical susceptibility action, not on factor parameters, because \(C\) has an orthogonal-row gauge.

## 4. The 0.002 Ha/e absolute gate

Using the CODATA Hartree conversion, 0.002 Ha/e corresponds to approximately **54.4 mV** of electrostatic potential or **1.255 kcal/mol per unit test charge**. ([NIST][5])

That makes the gate physically plausible as a conservative local-potential requirement. It prevents a model with tolerable relative averages from producing a chemically significant local exterior-potential spike. Relative and absolute gates address different failure modes and are reasonably conjunctive.

What is **not** established is that 0.002 Ha/e is the uniquely correct threshold for a 1 kcal/mol total-solvation objective. A pointwise exterior potential error is not itself a solvation free-energy error. The conversion depends on the downstream source distribution, cavity operator, quadrature pairing, self-consistent feedback amplification, and the fraction of the total error budget allocated to learned response.

A future revision would require an independent derivation of this form:

1. Allocate a response-model contribution \(\epsilon_{\rm response}\) from the total 1 kcal/mol budget before inspecting model results.

2. Define the exact physical domain and norm for exterior MEP error, including point placement, surface distance, rotations, and whether the intended norm is \(L^\infty\), weighted \(L^2\), or a dual electrostatic norm.

3. Derive a downstream bound such as

$$
|\delta G_{\rm el}|
\le
\|D_\chi G_{\rm el}\|\,
\|\delta\chi\|,
$$

including the closed-loop PCM resolvent, rather than assuming a one-pass test-charge interaction.

4. Establish a probe-coverage constant

$$
\|\delta\chi\|_{\rm admissible}
\le
c_{\rm probe}
\max_{m=1,\ldots,4}
\|\delta V_m\|_{\rm audit},
$$

or show that no finite useful \(c_{\rm probe}\) follows from four modes.

5. Include permanent-source representation error, PCM discretization, source-to-boundary projection, finite-field nonlinearity, solver residual, and force/geometry-derivative error.

6. Derive the threshold from analytic operator bounds or a separately sealed calibration campaign, not from the fact that 0.002 failed here.

Until such a derivation exists in a new preregistration, the current 0.002 gate remains binding.

## 5. Finite sequence and data-opening rules

| Stage | Action                                                                                                                                                | Pass condition                                                                                   | Failure action                                                                 |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 0     | Freeze O2 code specification, optimizer, plateau rule, seeds, metric code, A4 provenance, A4 cap, PCA construction, grouped folds, and opening policy | Hash-addressed commitment before more training                                                   | Stop                                                                           |
| 1     | Implement and test analytic derivative backend                                                                                                        | All source, energy, first-, mixed-, and second-derivative tests pass                             | Stop                                                                           |
| 2     | One bounded A3 optimization closure                                                                                                                   | Plateau reached and all frozen full-train gates pass                                             | If no plateau: stop. If plateau plus gate failure: proceed only to eligible A4 |
| 3     | One nested A4 run                                                                                                                                     | Plateau reached and all frozen full-train gates pass                                             | Stop; C remains unexecuted                                                     |
| 4     | Run label-free direct-sum/ddPCM mathematical coupling tests on open train geometries                                                                  | All energy, adjointness, stability, covariance, dissociation, and force tests pass               | Stop                                                                           |
| 5     | Open grouped train-CV results                                                                                                                         | All frozen gates pass on pooled out-of-fold predictions and on every defined worst-record metric | Stop; validation remains sealed                                                |
| 6     | Hash final full-train model/code and open the 14 validation formulas once                                                                             | Every frozen validation gate passes                                                              | Stop; blind remains sealed                                                     |
| 7     | Evaluate the exact same artifact on the blind 14 once                                                                                                 | Report the preregistered result                                                                  | Any failure is terminal; no blind-informed repair                              |

The grouped folds should be committed now, but **not run or inspected until the candidate search has closed**. All four modes, both exterior frames, and all records from one formula must remain in the same fold. A4 PCA must be recomputed using only the training portion of each fold.

Do not use grouped CV to choose between A3, A4, and C. Under this sequence, theory and the full-train gates choose at most one final topology; CV is a one-time transfer qualification.

Do not open validation until:

* candidate search has ended permanently;
* the final topology, descriptors, normalization, optimizer, and weights are fixed;
* all train gates and grouped-CV gates pass;
* all scalar/ddPCM coupling tests pass.

Do not open the blind formulas until validation passes. Nothing learned from validation may alter training, PCA, architecture, parameter cap, thresholds, source basis, optimizer, or ddPCM coupling.

## 6. Permanent point \(q/p\) plus induced GTO/\(l=2\): coherent only with a common dual pairing

The use of different permanent and induced source bases is **not itself a category error**. A permanent reference density may be represented by point monopoles/dipoles while the differential response uses a richer smooth Gaussian direct sum. Physical coherence does not require identical bases. It requires that both bases map into one physical charge-density space and that all couplings derive from one total scalar.

Let

$$
p(R)\in P_{\rm perm},
\qquad
s\in P_{\rm ind},
$$

and let

$$
B_p(R)p,\qquad B_i(R)s
$$

be their physical source-density maps. Let \(G_R\) be the ddPCM reaction operator, self-adjoint under the correct physical or quadrature-weighted pairing \(\langle\cdot,\cdot\rangle_Q\). The conjugate induced-field coordinates must be the exact adjoint projection

$$
\eta=B_i^\dagger G_R\rho,
\qquad
\rho=B_pp+B_is.
$$

They cannot merely be a conveniently normalized collection of sampled potentials unless that collection is proven to be the dual coordinate system of \(s\).

With \(\chi=C^TC\) and your convention \(s=-\chi\eta\), introduce a latent polarization variable \(z\) and the total scalar

$$
\mathcal F_R(z)
=
\frac12 z^Tz
+
\frac12
\left\langle
B_pp-B_iC^Tz,\,
G_R\!\left(B_pp-B_iC^Tz\right)
\right\rangle_Q .
$$

Set

$$
s=-C^Tz,
\qquad
\rho=B_pp+B_is.
$$

Stationarity gives

$$
\nabla_z\mathcal F_R
=
z-CB_i^\dagger G_R\rho
=0,
$$

and therefore

$$
s
=
-C^TCB_i^\dagger G_R\rho
=
-\chi\,\eta.
$$

This is the required fixed-point/energy closure. It also gives the stationary energy identity

$$
\mathcal F_R(z^\star)
=
\frac12
\left\langle
B_pp,\,
G_R\!\left(B_pp+B_is^\star\right)
\right\rangle_Q .
$$

Continuum electrostatic solvation energy has precisely the characteristic one-half reaction-field pairing, reflecting the work needed to polarize the environment, and self-consistent reaction-field formulations are variational when their constituent energy is variational. ([arXiv][6])

Before ddPCM integration, require the following tests:

1. **Field–source work identity**

   $$
   \eta^Ts=2E_{\rm response}=-\|C\eta\|^2.
   $$

2. **Mixed permanent/induced reciprocity**

   $$
   \langle B_pa,G_RB_ib\rangle_Q
   =
   \langle B_ib,G_RB_pa\rangle_Q
   $$

   for randomized \(a,b\).

3. **Fixed point equals scalar stationarity**

   The numerical fixed-point residual must equal \(\nabla_z\mathcal F_R\), not merely correlate with it.

4. **Stationary total-energy identity**

   Direct evaluation of \(\mathcal F_R(z^\star)\), the half permanent–total reaction pairing, and a charging-integral implementation must agree to the solver tolerance.

5. **Closed-loop stability**

   The latent Hessian

   $$
   H_z
   =
   I+
   C B_i^\dagger G_R B_i C^T
   $$

   must be positive definite under the frozen sign convention. Passivity of \(\chi\) alone does not prevent a solvent-feedback polarization catastrophe. A nonpositive eigenvalue is an immediate stop; a positive safety margin must be derived before production use.

6. **Weighted rather than assumed Euclidean adjointness**

   PCM continuum operators are reciprocal, but common discretizations do not necessarily preserve raw Euclidean self-adjointness. The correct quadrature-weighted pairing or explicit adjoint formulation is required. ([arXiv][6])

7. **Exact derivative of the discrete total energy**

   Forces must match finite differences of the same discrete scalar, including derivatives through \(p(R)\), \(C(R)\), Gaussian centers, widths if variable, cutoffs, and the cavity operator. ddPCM and related domain-decomposition methods explicitly treat energy and analytic first derivatives, and their force implementations are validated against finite differences of the discrete energy. ([PubMed][7])

8. **Basis-rescaling and dimensional covariance**

   Radial \(l\le1\) fields and \(l=2\) field gradients have different physical dimensions. Under a harmless induced-basis rescaling \(B_i\mapsto B_iS\), the coordinates and susceptibility must transform as

   $$
   \eta\mapsto S^T\eta,
   \qquad
   \chi\mapsto S^{-1}\chi S^{-T},
   $$

   leaving energy and physical density unchanged. This is especially important before any cross-sector factor rows are contemplated.

9. **Dissociation and locality**

   At infinite fragment separation, the learned susceptibility must become fragment block-diagonal. No factor row may aggregate disconnected fragments into one squared latent norm. Any inherited A4 latent must be taken before global equilibration or pooling.

10. **Gauge sensitivity of MDP \(q/p\)**

    If atomic MDP multipoles possess a decomposition gauge, their use as coefficient-head descriptors can make the response depend on an arbitrary atomic representation even when the exterior permanent potential is unchanged. Either the MDP gauge must be canonical and frozen as part of the model definition, or response sensitivity to exterior-potential-preserving gauge perturbations must be bounded.

Finally, the observed finite-field linearity establishes quadratic response only over the sampled amplitudes and four excitation directions. It does not prove that the self-consistent ddPCM reaction fields lie inside the same amplitude and directional domain. Passing all response gates therefore remains a necessary representation qualification, not proof of 1 kcal/mol total solvation accuracy.

**Finite endpoint:** O2 closure, then at most one admissible A4, then grouped CV, one validation opening, and one blind opening. C is outside this preregistration.

[1]: https://arxiv.org/html/2403.17207v2 "https://arxiv.org/html/2403.17207v2"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html "https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html"
[3]: https://arxiv.org/pdf/2206.07697 "https://arxiv.org/pdf/2206.07697"
[4]: https://arxiv.org/html/2602.19411v1 "https://arxiv.org/html/2602.19411v1"
[5]: https://physics.nist.gov/cgi-bin/cuu/Value?hrev=&utm_source=chatgpt.com "CODATA Value: Hartree energy in eV"
[6]: https://arxiv.org/pdf/2203.06846 "https://arxiv.org/pdf/2203.06846"
[7]: https://pubmed.ncbi.nlm.nih.gov/26584117/ "https://pubmed.ncbi.nlm.nih.gov/26584117/"
