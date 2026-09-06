## Decision

**Permanently reject `make_auxbasis`; retain the atom-centred Coulomb-projected auxiliary-density architecture behind one finite, cheapest-first standard-basis ladder.**

### Frozen candidate ladder

[
B_1=\mathrm{aug_etb}(\beta=2.0)
;\longrightarrow;
B_2=\mathrm{autoaux}
;\longrightarrow;
B_3=\mathrm{aug_etb}(\beta=1.5).
]

This order is locked by increasing expected representation size and cost, not by observed electrostatic performance. Therefore, the opened acetone AutoAux MEP result must not move AutoAux earlier. PySCF provides both ETB and AutoAux as standard generators, and recent density-coefficient work explicitly uses atom-centred ETB bases with (\beta=2.0) and (1.5), with smaller (\beta) producing the larger basis. ([PySCF][1])

### Frozen numerical gate

For every geometry (m), let (J_m) be the auxiliary Coulomb metric and define the dimensionless, basis-rescaling-invariant metric

[
\widehat J_m
============

D_m^{-1/2}J_mD_m^{-1/2},
\qquad
D_m=\operatorname{diag}(J_m).
]

Let (C_m) contain the electron-count and three electronic-first-moment constraints, and let (Z_m) span the unconstrained subspace

[
\ker!\left(C_mD_m^{-1/2}\right).
]

A candidate is numerically admissible only when, for **every** development geometry,

[
\operatorname{rank}(C_m)=4,
\qquad
\kappa_2!\left(Z_m^{T}\widehat J_m Z_m\right)\le 10^{10},
]

equivalently,

[
\frac{\lambda_{\min}}{\lambda_{\max}}\ge 10^{-10}.
]

The (10^{-10}) eigenvalue ratio is a **fail cutoff, not a truncation rule**: do not delete modes, change the cutoff, or rescue an ill-conditioned candidate with a geometry-dependent pseudoinverse. Retain the existing exact-constraint closure tolerance unchanged.

The previously reported AutoAux condition number of approximately (2\times10^{10}) does not automatically settle this gate unless it was computed from this same diagonal-normalized, constraint-restricted metric.

### Reaction-bound gate and stopping

For each numerically admissible candidate, use exactly the frozen projection, constraints, PCMSolver cavity, reaction construction, and rigorous bound calculation. It passes only if

[
\frac1{12}\sum_{i=1}^{12}U_i\le0.25\ \mathrm{kcal/mol},
\qquad
\max_i U_i\le0.50\ \mathrm{kcal/mol}.
]

The actual fixed-source errors, exterior-MEP errors, correlations, and basis dimensions are diagnostic only; none may override these inequalities.

Proceed sequentially:

1. Numerical or reaction-bound failure: advance to the next candidate.
2. First candidate passing both gates: **stop immediately and lock it**; do not evaluate a larger basis looking for a better score.
3. All three fail: close the standard auxiliary-basis lane. Do not add intermediate (\beta), change cutoffs, fit exponents, or introduce another generator under this experiment.

### Panel decision

Use the **same existing 12-case panel for the entire sequential development ladder**. Changing panels between candidates would confound basis choice with panel composition.

However, **reserve a fresh, chemically disjoint, target-free confirmation panel now**, before opening any new candidate results. Apply it exactly once to the first development passer, with the identical numerical and (0.25/0.50) gates. If confirmation fails, the frozen ladder fails; do not test the next basis on that confirmation panel.

**Minimal architecture:** the first standard basis passing both the common development panel and the sealed confirmation panel becomes the frozen representation. Otherwise, no standard atom-centred auxiliary-density representation is accepted.

[1]: https://pyscf.org/_modules/pyscf/df/addons.html "pyscf.df.addons — PySCF"
