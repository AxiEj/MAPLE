# Route-2 V0 accuracy bottleneck and response-kernel decision (2026-07-31)

## Status

There is **no valid current Route-2V0 solvation-accuracy value**. The only
honest answer to an accuracy question today is that the required physical
source and total free-energy scalar do not yet exist.

Two retained historical values must not be confused with progress:

* The retired GTO/QEq/GBn2 calculation missed the historical ethyl-acetate
  record by \(7.041442082076966\) kcal/mol. That exact record remains in the
  immutable acceptance panel.
* The legacy MACE exact-GTO ten-record report has a superficially lower MAE
  of \(0.9152863783\) kcal/mol, but its maximum is
  \(3.2473279957\) kcal/mol and its electronic state is nonvariational. It
  therefore fails both the all-record error rule and the common-scalar rule.

Neither number may be used to rank V0 or to justify an accuracy run.

## The observed bottleneck

The frozen MACE-MDP tensor accurately predicts the acetone **molecular**
dipole polarizability. Its first mathematically exact density embedding does
not predict the corresponding **spatial** response: the preregistered vacuum
QM MEP errors are \(0.4597926523\) in relative Frobenius norm and
\(0.4965624773\) in worst field direction, above the respective \(0.20\) and
\(0.30\) ceilings. The source still preserves net charge, induced molecular
dipole, source potential construction, and continuum duality at float64
precision.

This is the key diagnosis:

\[
\alpha_\theta=M_{\mathbf R} C_{\mathbf R}M_{\mathbf R}^{\mathsf T}
\]

only fixes a three-by-three projection of the full density susceptibility
\(C_{\mathbf R}\). The null space of \(M_{\mathbf R}\) leaves the molecular
dipole unchanged but changes the near-field electrostatic potential, reaction
field, and ultimately solvation energy. A radial width cannot be inferred
from \(\alpha_\theta\), and the failed one-radial map is an executable
demonstration rather than an invitation to tune that width.

This distinction is consistent with distributed-response literature:
polarizabilities are integrals of a density susceptibility, while local
polarizability densities retain information that can be hidden by the
integral. See [ISA-Pol](https://arxiv.org/abs/1806.06737) and
[origin-independent polarizability densities](https://doi.org/10.1021/acs.jpclett.1c02545).

## The only fast no-fit electronic candidate worth testing next

The next candidate is called **V0-RK**. It is not implemented and is not
admitted to PCM. Its purpose is to make the missing spatial response an
explicit physical asset rather than a guessed Gaussian.

Let:

* \(C_{0,\mathbf R}\succeq0\) be a fixed, independently sourced
  atom-centred GTO density-response kernel;
* \(A_{\mathbf R}\) map GTO coefficients to the stacked \(3N\) atomic induced
  dipoles;
* \(W_{\mathbf R}\) stack the already audited MACE-MDP atomic partitions
  \(W_a=\alpha_a\alpha^{-1}\);
* \(S_0=A C_0 A^\mathsf T\) on its declared nonsingular response support; and
* \(\Gamma=W\alpha_\theta W^\mathsf T\succeq0\).

Then use the covariance-completion identity

\[
\boxed{
C=C_0-C_0A^\mathsf TS_0^{-1}AC_0+
C_0A^\mathsf TS_0^{-1}\Gamma S_0^{-1}AC_0.
}
\]

On that support, it gives

\[
C\succeq0,\qquad ACA^\mathsf T=\Gamma,\qquad
\Bigl(\sum_a A_a\Bigr)C
\Bigl(\sum_a A_a\Bigr)^\mathsf T=\alpha_\theta.
\]

Thus the frozen MACE tensor fixes the observed molecular response, while the
independently sourced \(C_0\) fixes the radial and null-space response. The
electronic scalar is the quadratic form
\(\frac12\delta c^\mathsf TC^+\delta c\) on the response support. Its
field response is reciprocal and passive by construction; coupling it to a
reciprocal continuum then gives one KKT scalar rather than a learned
fixed-point update.

This is not a hidden fit:

* \(C_0\), its GTO basis, supported elements/charge states, and the support
  convention must be content-addressed before any solvation label is read.
* The formula has no adjustable width, solvation coefficient, MEP regression,
  response rescaling, or post-solve correction.
* A physical \(C_0\) may be generated once from an explicitly declared
  independent electronic-response procedure or a published response asset.
  It is an offline element/fragment asset, not a per-user QM calculation and
  not a fine-tune of MACE.
* If no such transferable source can pass the preregistered QM MEP gates, this
  branch is rejected. It must not be repaired by adding parameters.

The design is deliberately compatible with the speed requirement. Runtime
would consist of frozen MACE-MDP inference plus a fixed-basis KKT solve; the
expensive construction of \(C_0\) is offline. This is a design expectation,
not a performance result. A candidate is rejected as a production route if
its measured cold or warm same-geometry calculation is not faster than the
declared QM reference.

## Ordered evidence, before any experimental score

1. **Source provenance:** freeze one physical \(C_0\) construction, basis,
   elements, charge states, and hashes. No FreeSolv, MNSol, or target-solvent
   labels may participate.
2. **Response physics:** prove charge conservation, Euclidean covariance,
   \(C\succeq0\), atom/molecule moment identities, source/continuum duality,
   and reciprocal/passive response.
3. **QM spatial gate:** run finite-field density, MEP, induced-dipole, and
   reaction-potential tests on the already frozen twelve-record,
   ten-actual-functional-group geometry set. This is a physics panel, not an
   experimental accuracy panel; acetone alone cannot admit the method.
4. **Common scalar:** only after step 3 passes, combine V0-RK with the smooth
   density-defined cavity, nonpolar scalar, standard-state convention, and
   reciprocal continuum. Then prove KKT residuals, positive Schur complement,
   envelope forces, second-order finite differences, coordinate-loop work,
   and grid/cavity refinement.
5. **Runtime gate:** measure cold and warm end-to-end cost against the same
   QM task. If it is slower, it has no production advantage and is rejected
   regardless of numerical accuracy.
6. **Accuracy gate:** freeze all physical choices, then evaluate every
   individual record in historical FreeSolv10 and FreeSolv12. Each must be
   strictly below \(1.5\) kcal/mol, including ethyl acetate; ten actual
   functional groups are mandatory. Only afterward may the 11-solvent,
   custom-solvent, disjoint-confirmation, and truly blind experimental gates
   run. The final target remains every record below \(1\) kcal/mol.

The cavity/cavitation/dispersion branch remains a separate requirement.
Nonlocal dielectric response and smooth cavity mathematics are useful, but
SaLSA's reported total model includes a fitted dispersion scale and cannot
be imported as V0. The V0-AQ continuum controls remain useful for checking
scalar calculus, but an on-the-fly auxiliary QM calculation cannot be called
a fast production replacement. None of these steps requires GROMACS or a
molecular-liquid trajectory.

## Decision

Do not run a new solvation panel now. It would be testing a source already
shown to have the wrong spatial electronic response and would only create
misleading errors. The productive next task is to source-bind and
preregister \(C_0\), then try to falsify V0-RK on the full QM physics panel.
