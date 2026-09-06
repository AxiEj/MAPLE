## Decision

**Do not reject `make_auxbasis` on the present evidence. Retain it only as a quarantined candidate behind a new prospective operator-level gate.** The locked v1 gate remains permanently **FAIL**; nothing in v2 reinterprets or repairs that result.

Methane’s 19.632% relative error is consistent with a low-denominator instability, given its 0.99698 correlation, (6.94\times10^{-4}) Hartree/e maximum absolute error, and essentially exact charge/moment constraints. That is evidence that the **v1 metric is unsuitable for near-zero exterior potentials**, not proof that the representation is PCM-accurate.

## Exact v2 oracle and backend

Use:

* **Oracle source:** the frozen (\omega)B97M-V/def2-TZVPD AO density, evaluated directly—without another density fit—to obtain the total solute MEP on the cavity surface.
* **Candidate source:** the Coulomb-metric `make_auxbasis` projection with the locked electron-count and electronic-first-moment constraints.
* **Backend:** **PCMSolver’s external-cavity-MEP interface**, with identical nuclei, cavity geometry, radii, tessellation, dielectric, and PCM operator for oracle and candidate.
* **Do not use ddX’s volume-density source path for this gate.** Gaussian auxiliary densities have exterior tails, whereas the stated ddX derivation assumes density support inside the cavity. That would mix representation error with a violated backend assumption.

For multiple intended PCM operators or solvents, freeze that operator panel prospectively and gate the worst case; do not select the easiest operator afterward.

## Clean v2 quantity

Let (v_i) and (\tilde v_i) be the reference and auxiliary total MEP vectors on the same cavity, and let

[
\delta v_i=\tilde v_i-v_i .
]

Let (K_i) be PCMSolver’s energy-consistent, symmetric positive reaction kernel—schematically the properly weighted (A_i^{-1}). Define the fixed-source stabilization magnitude

[
P_i(v)=\frac12 v^{T}K_i v ,
]

the actual polarization-energy error

[
d_i=\left|P_i(\tilde v_i)-P_i(v_i)\right|,
]

and the positive error-source self-energy

[
e_i=\frac12\delta v_i^{T}K_i\delta v_i .
]

Then

[
d_i
===

\left|v_i^{T}K_i\delta v_i+\frac12\delta v_i^{T}K_i\delta v_i\right|
\le
2\sqrt{P_i(v_i)e_i}+e_i
\equiv U_i .
]

**Use (U_i) as the gate statistic and report (d_i) alongside it.** This is an absolute, operator-weighted, cancellation-resistant error bound in kcal/mol. It remains well behaved when the reference MEP is nearly zero.

## Budget

A 1 kcal/mol end-to-end objective does not uniquely determine the internal allocation. It must be frozen as an architecture policy. A defensible deterministic allocation is:

[
\boxed{\operatorname{mean}_i U_i \le 0.25\ {\rm kcal/mol}}
]

and

[
\boxed{\max_i U_i \le 0.50\ {\rm kcal/mol}}.
]

This reserves at least 0.75 kcal/mol of mean absolute-error budget for frozen MDP/POLAR model error, self-consistent response, continuum/cavity error, and CDS. It uses an additive budget rather than unjustified root-sum-square independence.

## Comparison of alternatives

**(a) Fixed-source PCM energy error:** chemically closest to the final observable and should always be reported. Alone, however, it can be small through cancellation between the cross term and the error-source self-energy.

**(b) (A^{-1}) norm:** the correct secondary structure. The derived (U_i) above combines it with the reference PCM energy and converts it into a rigorous kcal/mol certificate. A bare relative (A^{-1})-norm would reproduce the methane denominator problem.

**(c) Absolute-MEP floor or mixed absolute/relative metric:** inferior as the primary gate. Any floor chosen after observing methane would be post hoc, and ordinary surface (L^2) weights do not reflect which potential modes the continuum amplifies. It may remain a descriptive diagnostic only.

**(d) Excluding methane or other symmetry/nonpolar controls:** reject. That would be direct case deletion after failure. Such molecules are valuable controls for charge leakage, symmetry breaking, and spurious low-order moments. Keep methane or a prospectively selected equivalent in v2.

## Final disposition

**Status now: RETAIN–BLOCKED, not accepted.** Freeze the criterion, backend, operator panel, budget, and a fresh prospective molecular panel before calculating v2 results. If either the (0.25) mean or (0.50) per-case bound fails, terminate `make_auxbasis` for this role without exclusions or threshold revision. If both pass, retain it as an adequately controlled density representation—but not as evidence that the full MACE-MDP/MACE-POLAR model will achieve below 1 kcal/mol.
