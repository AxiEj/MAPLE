# Route-2 V0 accuracy bottleneck and response-kernel decision (2026-07-31)

## Status

There is **no valid current Route-2V0 solvation-accuracy value**. A
free-atom translation-tangent source has now passed one preregistered acetone
**gas-phase QM-MEP physics canary**. An induced-only reduced source-space KKT
structural kernel and a direct-sum frozen-permanent-source KKT gate also
exist. Neither has a physical continuum binding, force proof, stationary
permanent electronic density, or multi-functional-group physics certificate.
They therefore cannot be used to quote a solvation error.

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
dipole polarizability. Its first mathematically exact density embedding did
not predict the corresponding **spatial** response: the preregistered vacuum
QM MEP errors were \(0.4597926523\) in relative Frobenius norm and
\(0.4965624773\) in worst field direction, above the respective \(0.20\) and
\(0.30\) ceilings. That inherited one-radial Gaussian source remains
permanently rejected.

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

## What has now passed — and what has not

The preregistered
[`route2-v0-atomic-displacement-source-acetone-v1.json`](benchmarks/route2-v0-atomic-displacement-source-acetone-v1.json)
uses no fitted width. It translates independently generated spherical
free-atom HF electron densities infinitesimally and distributes the frozen
raw MACE-MDP induced dipole by the already identity-checked atomic partition.
The radial asset was generated twice with a preregistered one-thread runtime
and byte-identical archives before it was used.

Against **all 516** frozen exterior acetone QM-MEP points, this source passes
the predeclared physics ceilings:

\[
\frac{\|\partial_E V_{\rm ADT}-\partial_E V_{\rm QM}\|_F}
     {\|\partial_E V_{\rm QM}\|_F}=0.1484314509<0.20,
\qquad
\max_{\hat E}\operatorname{relerr}=0.1611208415<0.30,
\]

while its induced-dipole mismatch remains
\(0.0038678622<0.20\). This is evidence that the original accuracy bottleneck
was the guessed spatial radial response, not a need for an experimental
correction or a MACE-MDP re-fit.

It is nevertheless only a **single-molecule source falsifier**. It is not a
proof of transferable response, a full susceptibility kernel, a GTO
coefficient representation, a continuum source, an energy functional, a
force, or an accuracy result. V0-ADT now has an exact **direct-sum**
point-multipole-plus-radial-tangent surface/dual representation, which avoids
choosing a GTO width. The remaining issue is a stationary permanent electronic
functional and a source-bound physical continuum, not a missing algebraic
transpose.

## The remaining no-fit electronic construction

The passing source is called **V0-ADT** (atomic displacement tangent). It is
the first concrete physical radial response candidate. Its reduced
induced-only and frozen-permanent-source direct-sum KKT constructions are now
executable, but neither is a physical PCM or total-solvation model. Its next
**physical** gate is binding the direct-sum source to an actual reciprocal,
source-bound continuum. A future full density-functional formulation remains
separate. **V0-RK** now has an algebraic covariance-completion structural
kernel, but no independently source-bound physical \(C_0\) asset; it remains
the fallback if V0-ADT fails the frozen multi-molecule QM physics panel or
cannot be made energy-conjugate without an arbitrary projection.

Let:

* \(C_{0,\mathbf R}\succeq0\) be a fixed, independently sourced
  source/dual density-response kernel;
* \(A_{\mathbf R}\) map its declared response coefficients to the stacked \(3N\) atomic induced
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

* \(C_0\), its declared source/dual basis, supported elements/charge states, and the support
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

The next concrete candidate is the preregistered frozen atomic-HF
independent-particle transition-density source in
[`ROUTE2_V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_THEORY.md`](ROUTE2_V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_THEORY.md).
It supplies a broad neutral \(C_0\) without a fitted radial width and is
completed only through the existing MACE moment constraint.  Until the
hash-bound table and its frozen acetone source falsifier have executed, it is
still only a source candidate; it has no physical-continuum or accuracy claim.

## Ordered evidence, before any experimental score

1. **Source provenance:** freeze V0-ADT's physical radial tables, its exact
   direct-sum surface-source dual with the frozen permanent MACE source,
   supported elements/charge states, and hashes. Bind one physical reciprocal
   continuum before claiming anything beyond the synthetic KKT control. If that
   route fails, freeze one physical \(C_0\) construction instead. No FreeSolv,
   MNSol, or target-solvent labels may participate.
2. **Response physics:** prove charge conservation, Euclidean covariance,
   atom/molecule moment identities, source/continuum duality, and reciprocal/
   passive response. A full-kernel fallback must additionally prove
   \(C\succeq0\).
3. **QM spatial gate:** run finite-field density, MEP, induced-dipole, and
   reaction-potential tests on the already frozen twelve-record,
   ten-actual-functional-group geometry set. This is a physics panel, not an
   experimental accuracy panel; the acetone V0-ADT pass is a prerequisite,
   not a substitute.
4. **Common scalar:** only after step 3 passes, combine V0-ADT (or, if it is
   rejected, V0-RK) with the smooth density-defined cavity, nonpolar scalar,
   standard-state convention, and reciprocal continuum. Then prove KKT
   residuals, positive Schur complement, envelope forces, second-order finite
   differences, coordinate-loop work, and grid/cavity refinement.
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

Do not run a new solvation panel now. The previous one-radial source was
indeed physically wrong; V0-ADT has repaired that **one source-level acetone
test** without tuning, but still lacks the common scalar and the required
multi-functional-group physics evidence. The productive next task is to
construct and falsify its same-basis dual/KKT form, then run the full QM
physics panel. If that fails, preserve V0-ADT as a bounded positive result
and return to the separately source-bound \(C_0\) / V0-RK path.
