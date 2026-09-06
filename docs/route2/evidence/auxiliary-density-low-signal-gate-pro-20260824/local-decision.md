# Local decision

The Pro result is adopted with the following independent checks.

1. For a symmetric negative PCM response `q=R v`, define
   `||v||_K=sqrt(-v.T R v)`. With `e=v_candidate-v_reference`,

       |G(v+e)-G(v)| <= ||v||_K ||e||_K + 0.5 ||e||_K^2.

   This equals the advisor's `2 sqrt(P e_self) + e_self` notation and is a
   cancellation-resistant absolute energy bound.

2. The v1 relative-MEP result remains `FAIL`. Methane and every other frozen
   case remain in v2; there is no denominator floor, case deletion, or threshold
   rewrite.

3. PCMSolver's symmetric external-MEP interface is selected because it applies
   the identical physical operator to the direct AO MEP and the fitted-density
   MEP. The ddX volume-density path is not used because its compact-support
   assumption is not exact for Gaussian auxiliary functions.

4. The one-kcal/mol objective is an MAE target, so a mean bound of 0.25 kcal/mol
   is the relevant 25-percent representation allocation. A separate 0.50
   kcal/mol maximum bound guards the tail. Actual fixed-source energy error is
   reported but cannot replace the bound because cross and self terms may
   cancel.

Passing v2 will retain only the representation oracle. It will not admit a
trained source, full hybrid accuracy, or any MAPLE capability.
